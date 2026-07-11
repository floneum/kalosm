use std::hash::{Hash, Hasher};

use super::*;

#[derive(Clone, Copy)]
enum QmatmulDirectTokens {
    Workgroup,
    Qgemv(tile_ir_kernels::SubgroupConfig),
    Coop {
        subgroup: tile_ir_kernels::SubgroupConfig,
        coop: tile_ir::CoopMatrixToken,
    },
}

impl QMatMulOperation {
    /// Lower an M-padded matmul: copy the activation into a zero-padded
    /// scratch tensor (the same kernel a `resize` view lowers to) and run the
    /// matmul over the padded views. The output buffer's slack rows were
    /// allocated by `qmatmul_operation_inputs`, which makes the same
    /// `m_pad_target` decision.
    fn build_m_padded_kernels(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Option<Vec<DirectKernel>> {
        let device = graph.device();
        let padded_m = self.m_pad_target(KernelDeviceCaps::from_device(&device))?;
        // A pad target implies no epilogues: inputs are [input, matrix, output].
        let [input, MirValue::QMatrix(matrix), output] = inputs else {
            return None;
        };
        let input = input.as_tensor()?;
        let output = output.as_tensor()?;
        let in_shape = input.layout().shape();
        let out_shape = output.layout().shape();
        if in_shape.len() < 2 || out_shape.len() != in_shape.len() {
            return None;
        }
        let m_axis = in_shape.len() - 2;

        let mut padded_in_shape = in_shape.to_vec();
        padded_in_shape[m_axis] = padded_m;
        let scratch = TensorData::new_for_shape(&device, &padded_in_shape, input.datatype());
        let pad_copy = crate::view::ViewOperation {
            input: self.input,
            stages: vec![crate::view::ViewStage {
                layout: Layout::from_parts(
                    0,
                    padded_in_shape.into(),
                    Layout::continuous_strides(in_shape),
                ),
                input_shape: in_shape.into(),
                defined: in_shape.into(),
                fill: crate::view::zero_scalar(input.datatype()),
            }],
            datatype: input.datatype(),
        };
        let pad_workgroup = pad_copy
            .workgroup_shape_constraints(&device)
            .solve(device.max_subgroup_size(), &device.limits())?;
        let pad_inputs = [input.clone().into(), scratch.clone().into()];
        let pad_kernel = pad_copy.build_direct_kernel(graph, &pad_workgroup, &pad_inputs)?;

        let mut padded_out_shape = out_shape.to_vec();
        padded_out_shape[m_axis] = padded_m;
        let padded_out_layout = Layout::contiguous(&padded_out_shape);
        let padded_bytes =
            padded_out_shape.iter().product::<usize>() * output.datatype().element_size();
        if output.layout().offset() != 0 || (padded_bytes as u64) > output.buffer().size() {
            return None;
        }
        let padded_output = TensorData::new_from_parts(
            &device,
            output.buffer().clone(),
            padded_out_layout,
            output.datatype(),
        );

        let matmul = if matches!(matrix.datatype(), GgmlType::F32 | GgmlType::F16) {
            self.build_dense_direct_kernel(graph, &scratch, matrix, &padded_output)?
        } else {
            Self::direct_kernel_for_tensors(
                &device,
                DirectKernelTensors {
                    input: &scratch,
                    matrix,
                    pre_extra_tensors: &[],
                    post_extra_tensors: &[],
                    output: &padded_output,
                },
                self.name(),
                DirectKernelChains {
                    pre_expr: None,
                    post_expr: None,
                    post_accumulator_offsets: &[],
                },
                Some((self, inputs)),
            )?
        };
        Some(vec![pad_kernel, matmul])
    }

    /// F32/F16 ggml storage is dense values: read the matrix buffer as a
    /// transposed dense weight and run a regular matmul kernel.
    fn build_dense_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        input: &TensorData,
        matrix: &QMatrix,
        output: &TensorData,
    ) -> Option<DirectKernel> {
        let [n, k] = matrix.shape() else {
            return None;
        };
        let (n, k) = (*n, *k);
        let input_shape = input.layout().shape();
        let rank = input_shape.len();
        if rank < 2 {
            return None;
        }
        let mut dense_shape = input_shape.to_vec();
        dense_shape[rank - 2] = k;
        dense_shape[rank - 1] = n;
        let mut dense_strides = vec![0; rank];
        dense_strides[rank - 2] = 1;
        dense_strides[rank - 1] = k;
        let matrix_datatype = match matrix.datatype() {
            GgmlType::F32 => DataTypeEnum::F32,
            GgmlType::F16 => DataTypeEnum::F16,
            _ => return None,
        };
        if input.datatype() != matrix_datatype || output.datatype() != matrix_datatype {
            return None;
        }
        let dense_weight_t = TensorData::new_from_parts(
            matrix.device(),
            matrix.buffer().clone(),
            Layout::from_parts(
                0,
                dense_shape.into_boxed_slice(),
                dense_strides.into_boxed_slice(),
            ),
            matrix_datatype,
        );
        let device = graph.device();
        let dense_matmul = MatMulOperation::new(
            matrix_datatype,
            self.input,
            self.input,
            input.layout().shape(),
            dense_weight_t.layout().shape(),
            None,
            &device,
        );
        dense_matmul.build_direct_kernel(
            graph,
            &dense_matmul
                .workgroup_shape_constraints(&device)
                .solve(device.max_subgroup_size(), &device.limits())?,
            &[
                input.clone().into(),
                dense_weight_t.into(),
                output.clone().into(),
            ],
        )
    }

    /// Build a direct quantized-matmul kernel for the supplied tensors.
    /// `pre_chain`/`post_chain` are pre- and post-element-wise unary chains
    /// to fuse into the kernel; pass `None` to skip. `operation_key` ties the
    /// compiled module into an operation-bound cache slot; pass `None` for an
    /// ad-hoc call (e.g. the sampler path).
    pub(crate) fn direct_kernel_for_tensors(
        device: &Device,
        tensors: DirectKernelTensors<'_>,
        kernel_name: impl Into<String>,
        chains: DirectKernelChains<'_>,
        operation_key: Option<(&dyn Operation, &[MirValue])>,
    ) -> Option<DirectKernel> {
        let DirectKernelTensors {
            input,
            matrix,
            pre_extra_tensors,
            post_extra_tensors,
            output,
        } = tensors;
        let DirectKernelChains {
            pre_expr,
            post_expr,
            post_accumulator_offsets,
        } = chains;
        if input.datatype() != output.datatype() {
            return None;
        }
        let f16_storage = match input.datatype() {
            DataTypeEnum::F32 => false,
            DataTypeEnum::F16 if device.f16_supported() => true,
            DataTypeEnum::F16 | DataTypeEnum::U32 => return None,
        };
        if f16_storage
            && (!pre_extra_tensors.is_empty()
                || !post_extra_tensors.is_empty()
                || pre_expr.is_some()
                || post_expr.is_some())
        {
            return None;
        }
        if matches!(matrix.datatype(), GgmlType::F32 | GgmlType::F16) {
            return None;
        }
        let input_rank = input.layout().shape().len();
        if input_rank != output.layout().shape().len() {
            return None;
        }

        let format = qmatrix_direct_quant_format(matrix)?;
        let a_view = flatten_matrix_layout(input.layout())?;
        let y_view = flatten_matrix_layout(output.layout())?;
        let m = a_view.rows;
        let k = a_view.cols;
        let y_m = y_view.rows;
        let n = y_view.cols;
        let matrix_n = matrix.shape[0] as u32;
        let has_custom_accumulator_offsets = !post_accumulator_offsets.is_empty();
        let post_value_arity = post_accumulator_offsets.len().max(1);
        let max_accumulator_offset = post_accumulator_offsets.iter().copied().max().unwrap_or(0);
        if has_custom_accumulator_offsets && post_expr.is_none() {
            return None;
        }
        if m != y_m || k != matrix.shape[1] as u32 {
            return None;
        }
        let limits = device.limits();
        let caps = KernelDeviceCaps::from_device(device);
        let subgroup_size_range = [caps.min_subgroup_size, caps.max_subgroup_size];
        let max_workgroups = effective_qmatmul_max_workgroups_per_dimension(&limits);
        let y_supports_coop = tile_ir_kernels::cooperative_store_layout_supported(&y_view.layout);
        let mut variant = select_qmatmul_direct_variant(format, m, k, n, caps);
        if f16_storage {
            variant = QMatmulPath::Workgroup;
        }
        let direct_tokens = match variant {
            QMatmulPath::Workgroup => QmatmulDirectTokens::Workgroup,
            QMatmulPath::Q5SmallSingleRow | QMatmulPath::SingleRow => {
                QmatmulDirectTokens::Qgemv(device.subgroup_config()?)
            }
            QMatmulPath::Q8Wide(_) | QMatmulPath::Tile { .. } => QmatmulDirectTokens::Coop {
                subgroup: device.subgroup_config()?,
                coop: device.coop_token(CooperativeMatrixKind::F32F32M8N8K8)?,
            },
        };
        let use_workgroup_qmatmul = matches!(direct_tokens, QmatmulDirectTokens::Workgroup);
        if has_custom_accumulator_offsets {
            if !qmatmul_custom_accumulator_offsets_supported(
                format,
                variant,
                m,
                k,
                n,
                matrix_n,
                max_accumulator_offset,
                max_workgroups,
            ) {
                return None;
            }
        } else if n != matrix_n {
            return None;
        }
        let pre_extra_pointwise_views = pre_extra_tensors
            .iter()
            .map(|extra| {
                let layout = extra.layout();
                let column = extra.datatype() == DataTypeEnum::F32
                    && layout.shape().len() == 1
                    && layout.shape()[0] == k as usize
                    && layout.offset() == 0
                    && layout.strides() == [1];
                if column {
                    return Some(None);
                }
                if extra.datatype() != DataTypeEnum::F32 || layout.shape() != input.layout().shape()
                {
                    return None;
                }
                let view = flatten_matrix_layout(layout)?;
                (view.rows == m && view.cols == k).then_some(Some(view))
            })
            .collect::<Option<Vec<_>>>()?;
        let post_extra_pointwise_views = post_extra_tensors
            .iter()
            .map(|extra| {
                let layout = extra.layout();
                let column = extra.datatype() == DataTypeEnum::F32
                    && layout.shape().len() == 1
                    && layout.shape()[0] == n as usize
                    && layout.offset() == 0
                    && layout.strides() == [1];
                if column {
                    return Some(None);
                }
                if extra.datatype() != DataTypeEnum::F32
                    || layout.shape() != output.layout().shape()
                {
                    return None;
                }
                let view = flatten_matrix_layout(layout)?;
                (view.rows == m && view.cols == n).then_some(Some(view))
            })
            .collect::<Option<Vec<_>>>()?;
        let mut qmatmul_workgroups_x = 1;
        let use_f16_workgroup_tiles = f16_storage;
        let use_coop_acc_init_epilogue = !use_workgroup_qmatmul
            && !has_custom_accumulator_offsets
            && pre_expr.is_none()
            && post_expr.is_some_and(qmatmul_post_expr_is_column_add)
            && post_extra_pointwise_views.len() == 1
            && post_extra_pointwise_views[0].is_none()
            && qmatmul_variant_supports_coop_acc_init(variant, m, k, n, y_supports_coop);

        // Build the per-tile epilogue closures once. `None` if the resolver
        // didn't attach an expression; `Some` triggers the `_with_epilogue`
        // kernel variants. The closures capture expressions by clone so they
        // can live in the long-lived `tile_ir::tile::build` closure below.
        let pre_epilogue_with_extras = if let Some(expr) = pre_expr {
            let expression = expr.expression.clone();
            let input_datatype = expr.input_datatype;
            let output_datatype = expr.output_datatype;
            Some(tile_ir_kernels::UnaryEpilogueWithExtras::new(
                "qmatmul_pre_expr",
                pre_extra_tensors.len(),
                move |tile| {
                    let input = tile[0].clone();
                    let extras = tile[1..]
                        .iter()
                        .cloned()
                        .map(|value| (crate::nary_direct::ValueTile::F32(value), DataTypeEnum::F32))
                        .collect::<Vec<_>>();
                    apply_single_input_elementwise_expr(
                        input,
                        input_datatype,
                        &expression,
                        output_datatype,
                        &extras,
                    )
                    .expect("pre expression validated at fuse time")
                    .0
                },
            ))
        } else {
            None
        };
        let post_epilogue_with_extras = if use_coop_acc_init_epilogue {
            None
        } else if let Some(expr) = post_expr {
            let expression = expr.expression.clone();
            let input_datatype = expr.input_datatype;
            let output_datatype = expr.output_datatype;
            Some(
                tile_ir_kernels::UnaryEpilogueWithExtras::new_with_value_arity(
                    "qmatmul_post_expr",
                    post_value_arity,
                    post_extra_tensors.len(),
                    move |tile| {
                        let values = tile[..post_value_arity]
                            .iter()
                            .cloned()
                            .map(|value| (value, input_datatype))
                            .collect::<Vec<_>>();
                        let extras = tile[post_value_arity..]
                            .iter()
                            .cloned()
                            .map(|value| {
                                (crate::nary_direct::ValueTile::F32(value), DataTypeEnum::F32)
                            })
                            .collect::<Vec<_>>();
                        apply_multi_input_elementwise_expr(
                            &values,
                            &expression,
                            output_datatype,
                            &extras,
                        )
                        .expect("post expression validated at fuse time")
                        .0
                    },
                ),
            )
        } else {
            None
        };
        let mut accumulator_offsets_hasher = FxHasher::default();
        post_accumulator_offsets.hash(&mut accumulator_offsets_hasher);
        let accumulator_offsets_identity = accumulator_offsets_hasher.finish();
        let epilogue_identity = pre_epilogue_with_extras
            .as_ref()
            .map(|e| e.identity())
            .unwrap_or(0)
            ^ post_epilogue_with_extras
                .as_ref()
                .map(|e| e.identity())
                .unwrap_or(0)
            ^ if use_coop_acc_init_epilogue {
                0xB1A5_C001u64
            } else {
                0
            }
            ^ if use_f16_workgroup_tiles {
                0xF16C_A5A5u64
            } else {
                0
            }
            ^ if f16_storage { 0xF16F_0001u64 } else { 0 }
            ^ accumulator_offsets_identity;
        let fast_dispatch_size = if use_workgroup_qmatmul {
            // The workgroup-tiled kernel computes its own grid inside
            // `tile::build`; skip the pre-built-pipeline fast path.
            None
        } else {
            match variant {
                QMatmulPath::Q5SmallSingleRow | QMatmulPath::SingleRow => {
                    let qgemv_cols_per_workgroup =
                        qgemv_cols_per_workgroup_for_direct(format, k, n);
                    let qgemv_workgroups = n.div_ceil(qgemv_cols_per_workgroup);
                    let [dispatch_x, _] = split_workgroups_2d(qgemv_workgroups, max_workgroups)?;
                    qmatmul_workgroups_x = dispatch_x;
                    Some([
                        qmatmul_workgroups_x,
                        qgemv_workgroups.div_ceil(qmatmul_workgroups_x),
                        1,
                    ])
                }
                // The IR-build fallback (cached=false catch-all) is the only
                // path that defers the dispatch to the IR builder; every
                // tile-aligned coop variant has a precomputed `[n/BN, m/BM, 1]`.
                QMatmulPath::Tile {
                    cached: false,
                    tile,
                } if tile == CoopTile::new(64, 64, QMATMUL_COOP_BK) => None,
                QMatmulPath::Workgroup => None,
                QMatmulPath::Q8Wide(tile) | QMatmulPath::Tile { tile, .. } => {
                    Some([n / tile.bn, m / tile.bm, 1])
                }
            }
        };
        let kernel_name = kernel_name.into();
        // The pre-built-pipeline fast path can only be reused when there's no
        // epilogue attached — otherwise the cached pipeline encodes the wrong
        // (no-epilogue) kernel. Skip the fast path entirely when fusing.
        if pre_extra_tensors.is_empty()
            && post_extra_tensors.is_empty()
            && !has_custom_accumulator_offsets
            && pre_epilogue_with_extras.is_none()
            && post_epilogue_with_extras.is_none()
            && let Some(dispatch_size) = fast_dispatch_size
        {
            if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
                return None;
            }
            let pipeline_key = QMatMulDirectPipelineKey::new(
                matrix.datatype(),
                matrix.storage_layout(),
                crate::quantized::QMatMulShape { m, k, n: matrix_n },
                subgroup_size_range,
                dispatch_size,
                input.layout(),
                output.layout(),
            );
            if let Some(kernel) = cached_qmatmul_direct_kernel(
                &kernel_name,
                matrix,
                &pipeline_key,
                input,
                output,
                dispatch_size,
            ) {
                return Some(kernel);
            }
        }
        let pre_with_extras_for_ir = pre_epilogue_with_extras.clone();
        let post_with_extras_for_ir = post_epilogue_with_extras.clone();
        let post_accumulator_offsets_for_ir = post_accumulator_offsets.to_vec();
        let ir = tile_ir::tile::build(move |phase| {
            if f16_storage {
                let a = tile_storage_read_with_direct_layout_typed(
                    phase,
                    tile_ir::ElementType::F16,
                    a_view,
                );
                let b = tile_ir_kernels::quantized_matrix(phase, format, k, matrix_n);
                let y = tile_storage_write_with_direct_layout_typed(
                    phase,
                    tile_ir::ElementType::F16,
                    y_view,
                );
                let epilogues = tile_ir_kernels::QmatmulEpilogues::default();
                if m == 1 {
                    tile_ir_kernels::qgemv_workgroup_storage_f16_with_epilogue(
                        phase,
                        &a,
                        &b,
                        &y,
                        &epilogues,
                        max_workgroups,
                    );
                } else {
                    tile_ir_kernels::qmatmul_workgroup_storage_f16_with_epilogues(
                        phase,
                        &a,
                        &b,
                        &y,
                        &epilogues,
                        max_workgroups,
                    );
                }
                return;
            }
            let a = tile_storage_read_with_direct_layout(phase, a_view);
            let b = tile_ir_kernels::quantized_matrix(phase, format, k, matrix_n);
            let pre_extra_storage_defs = pre_extra_tensors
                .iter()
                .zip(pre_extra_pointwise_views.iter())
                .map(|(extra, pointwise_view)| {
                    if let Some(view) = pointwise_view.clone() {
                        QmatmulExtraStorage::Pointwise(tile_storage_read_with_direct_layout(
                            phase, view,
                        ))
                    } else {
                        let shape = extra.layout().shape();
                        assert_eq!(shape.len(), 1);
                        QmatmulExtraStorage::Column(phase.storage_read(
                            tile_ir::ElementType::F32,
                            tile_ir::Shape::new([shape[0] as u32]),
                        ))
                    }
                })
                .collect::<Vec<_>>();
            let pre_extra_storages = pre_extra_storage_defs
                .iter()
                .map(QmatmulExtraStorage::as_extra)
                .collect::<Vec<_>>();
            let post_extra_storage_defs = post_extra_tensors
                .iter()
                .zip(post_extra_pointwise_views.iter())
                .map(|(extra, pointwise_view)| {
                    if let Some(view) = pointwise_view.clone() {
                        QmatmulExtraStorage::Pointwise(tile_storage_read_with_direct_layout(
                            phase, view,
                        ))
                    } else {
                        let shape = extra.layout().shape();
                        assert_eq!(shape.len(), 1);
                        QmatmulExtraStorage::Column(phase.storage_read(
                            tile_ir::ElementType::F32,
                            tile_ir::Shape::new([shape[0] as u32]),
                        ))
                    }
                })
                .collect::<Vec<_>>();
            let post_extra_storages = post_extra_storage_defs
                .iter()
                .map(QmatmulExtraStorage::as_extra)
                .collect::<Vec<_>>();
            let y = tile_storage_write_with_direct_layout(phase, y_view);
            let epilogues = tile_ir_kernels::QmatmulEpilogues {
                pre: None,
                pre_with_extras: pre_with_extras_for_ir.as_ref(),
                pre_extra_inputs: &pre_extra_storages,
                post: None,
                post_with_extras: post_with_extras_for_ir.as_ref(),
                post_extra_inputs: &post_extra_storages,
                post_accumulator_offsets: &post_accumulator_offsets_for_ir,
                post_acc_init_col_vector: match post_extra_storages.first() {
                    Some(tile_ir_kernels::QmatmulExtra::Column(storage))
                        if use_coop_acc_init_epilogue =>
                    {
                        Some(*storage)
                    }
                    _ => None,
                },
            };
            if use_workgroup_qmatmul {
                if m == 1 {
                    if use_f16_workgroup_tiles {
                        tile_ir_kernels::qgemv_workgroup_f16_with_epilogue(
                            phase,
                            &a,
                            &b,
                            &y,
                            &epilogues,
                            max_workgroups,
                        );
                    } else {
                        tile_ir_kernels::qgemv_workgroup_with_epilogue(
                            phase,
                            &a,
                            &b,
                            &y,
                            &epilogues,
                            max_workgroups,
                        );
                    }
                } else {
                    if use_f16_workgroup_tiles {
                        tile_ir_kernels::qmatmul_workgroup_f16_with_epilogues(
                            phase,
                            &a,
                            &b,
                            &y,
                            &epilogues,
                            max_workgroups,
                        );
                    } else {
                        tile_ir_kernels::qmatmul_workgroup_with_epilogues(
                            phase,
                            &a,
                            &b,
                            &y,
                            &epilogues,
                            max_workgroups,
                        );
                    }
                }
                return;
            }
            // Map the selected variant to its cooperative tile dimensions.
            // The first two single-row variants short-circuit to
            // qgemv; the rest share the qmatmul_with_epilogue entry point.
            let tile = match variant {
                QMatmulPath::Q5SmallSingleRow | QMatmulPath::SingleRow => {
                    let QmatmulDirectTokens::Qgemv(subgroups) = direct_tokens else {
                        unreachable!("single-row qmatmul variant requires subgroup token");
                    };
                    tile_ir_kernels::qgemv_with_epilogue(
                        phase,
                        &a,
                        &b,
                        &y,
                        qmatmul_workgroups_x,
                        subgroups,
                        &epilogues,
                    );
                    return;
                }
                QMatmulPath::Workgroup => unreachable!("workgroup qmatmul returned above"),
                QMatmulPath::Q8Wide(tile) | QMatmulPath::Tile { tile, .. } => tile,
            };
            let QmatmulDirectTokens::Coop {
                subgroup: subgroups,
                coop: coop_token,
            } = direct_tokens
            else {
                unreachable!("direct qmatmul tile variant requires cooperative-matrix tokens");
            };
            tile_ir_kernels::qmatmul_with_epilogue(
                phase, &a, &b, &y, &epilogues, coop_token, subgroups, tile.bm, tile.bn, tile.bk,
            );
        });
        let dispatch_size = ir.grid;
        if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
            return None;
        }
        let pipeline_key = QMatMulDirectPipelineKey::new_with_epilogue(
            matrix.datatype(),
            matrix.storage_layout(),
            crate::quantized::QMatMulShape { m, k, n: matrix_n },
            epilogue_identity,
            subgroup_size_range,
            dispatch_size,
            input.layout(),
            output.layout(),
        );
        let cache_key = qmatmul_direct_cache_key::<QMatmulDirectEpilogueKernelVariant>(
            |state| {
                variant.hash(state);
                epilogue_identity.hash(state);
                subgroup_size_range.hash(state);
                QMATMUL_DIRECT_KERNEL_GENERATION.hash(state);
            },
            |state| {
                QMATMUL_DIRECT_KERNEL_GENERATION.hash(state);
                hash_qmatmul_shape(state, format, m, k, matrix_n);
                epilogue_identity.hash(state);
                subgroup_size_range.hash(state);
                hash_qmatmul_dispatch_layouts(
                    state,
                    dispatch_size,
                    input.layout(),
                    output.layout(),
                );
            },
            dispatch_size,
            if pre_extra_tensors.is_empty() && post_extra_tensors.is_empty() {
                None
            } else {
                operation_key
            },
        );
        qmatmul_direct_kernel_from_ir(
            device,
            kernel_name.clone(),
            kernel_name,
            cache_key,
            matrix,
            pipeline_key,
            input,
            pre_extra_tensors,
            post_extra_tensors,
            output,
            dispatch_size,
            || Some(ir),
        )
    }
}

fn hash_qmatmul_epilogue(state: &mut FxHasher, epilogue: &Option<ElementwiseEpilogue>) {
    match epilogue {
        Some(epilogue) => {
            true.hash(state);
            epilogue.expression.hash(state);
            epilogue.extras.len().hash(state);
            epilogue.input_datatype.hash(state);
            epilogue.output_datatype.hash(state);
        }
        None => false.hash(state),
    }
}

impl Operation for QMatMulOperation {
    /// Recognition and epilogue fusion only build operations the direct
    /// paths can lower (see `supports_elementwise_epilogue_fusion`), so a
    /// failure from every path is an invariant violation.
    fn build_direct_kernel_plan(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Result<DirectKernelPlan, DirectKernelLoweringError> {
        if inputs
            .last()
            .and_then(MirValue::as_tensor)
            .is_some_and(|output| output.layout().shape().contains(&0))
        {
            return Ok(DirectKernelPlan::empty());
        }

        if let Some(kernels) = self.build_m_padded_kernels(graph, inputs) {
            return Ok(DirectKernelPlan::many(kernels));
        }
        if let Some(kernel) = self.build_direct_kernel(graph, workgroup_shape, inputs) {
            return Ok(DirectKernelPlan::single(kernel));
        }
        Err(DirectKernelLoweringError::new(self.name()))
    }

    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        self.input_datatype.hash(state);
        self.in_shape.hash(state);
        self.out_shape.hash(state);
        self.matrix.datatype().hash(state);
        self.matrix.storage_layout().hash(state);
        self.matrix.shape().hash(state);
        hash_qmatmul_epilogue(state, &self.pre_element_wise_expr);
        hash_qmatmul_epilogue(state, &self.post_element_wise_expr);
        self.post_accumulator_offsets.hash(state);
    }

    fn workgroup_shape_constraints(
        &self,
        _device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        if self.m_size() == 1 {
            constraints.add_constraint(0, Constraint::Equals(1));
        } else {
            constraints.add_constraint(0, Constraint::Equals(32));
        }
        constraints.add_constraint(1, Constraint::Equals(1));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
    ) -> [u32; 3] {
        let n = self.n_size();
        let m = self.m_size();
        // Calculate batch size for dimensions beyond the last two (M, K)
        let batch_size: u32 = self
            .in_shape
            .iter()
            .rev()
            .skip(2)
            .map(|x| *x as u32)
            .product();

        if m == 1 {
            [n, 1, batch_size]
        } else {
            [n, m, batch_size]
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
        if let Some(epilogue) = &self.pre_element_wise_expr {
            for extra in &epilogue.extras {
                f(*extra);
            }
        }
        if let Some(epilogue) = &self.post_element_wise_expr {
            for extra in &epilogue.extras {
                f(*extra);
            }
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let m_pad = self.m_pad_target(KernelDeviceCaps::from_device(&nodes.device()));
        let base =
            qmatmul_operation_inputs(self.input, &self.matrix, &self.out_shape, m_pad, nodes);
        let pre_extras = self
            .pre_element_wise_expr
            .as_ref()
            .map(|epilogue| epilogue.extras.as_slice())
            .unwrap_or(&[]);
        let post_extras = self
            .post_element_wise_expr
            .as_ref()
            .map(|epilogue| epilogue.extras.as_slice())
            .unwrap_or(&[]);
        if pre_extras.is_empty() && post_extras.is_empty() {
            return base;
        }
        let mut result = Vec::with_capacity(base.len() + pre_extras.len() + post_extras.len());
        let (head, tail) = base.split_at(2);
        result.extend_from_slice(head);
        for extra in pre_extras.iter().chain(post_extras.iter()) {
            result.push(nodes.get_result_or_qmatrix(*extra).unwrap().into());
        }
        result.extend_from_slice(tail);
        result
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        if inputs.len() < 3 {
            return None;
        }
        let input = inputs[0].as_tensor()?;
        let MirValue::QMatrix(matrix) = &inputs[1] else {
            return None;
        };
        let output = inputs.last()?.as_tensor()?;
        // A rank-1 activation is a single matrix row: lower it through the
        // same [1, K] -> [1, N] views the rank-2 path uses.
        let input_row;
        let output_row;
        let (input, output) = if input.layout().rank() == 1 && output.layout().rank() == 1 {
            let in_len = input.layout().shape()[0];
            let out_len = output.layout().shape()[0];
            let out_stride = output.layout().strides()[0];
            input_row = TensorData::new_from_parts(
                input.device(),
                input.buffer().clone(),
                Layout::from_parts(
                    input.layout().offset(),
                    Box::new([1, in_len]),
                    Box::new([0, input.layout().strides()[0]]),
                ),
                input.datatype(),
            );
            output_row = TensorData::new_from_parts(
                output.device(),
                output.buffer().clone(),
                Layout::from_parts(
                    output.layout().offset(),
                    Box::new([1, out_len]),
                    Box::new([out_len * out_stride, out_stride]),
                ),
                output.datatype(),
            );
            (&input_row, &output_row)
        } else {
            (input, output)
        };
        let pre_extra_count = self
            .pre_element_wise_expr
            .as_ref()
            .map(|epilogue| epilogue.extras.len())
            .unwrap_or(0);
        let post_extra_count = self
            .post_element_wise_expr
            .as_ref()
            .map(|epilogue| epilogue.extras.len())
            .unwrap_or(0);
        if inputs.len() != 3 + pre_extra_count + post_extra_count {
            return None;
        }
        let extras = &inputs[2..inputs.len() - 1];
        let pre_extra_tensors = extras[..pre_extra_count]
            .iter()
            .map(|input| input.as_tensor())
            .collect::<Option<Vec<_>>>()?;
        let post_extra_tensors = extras[pre_extra_count..]
            .iter()
            .map(|input| input.as_tensor())
            .collect::<Option<Vec<_>>>()?;
        if input.datatype() != output.datatype()
            || !matches!(input.datatype(), DataTypeEnum::F32 | DataTypeEnum::F16)
        {
            return None;
        }
        if input.datatype() == DataTypeEnum::F16
            && (!pre_extra_tensors.is_empty()
                || !post_extra_tensors.is_empty()
                || self.pre_element_wise_expr.is_some()
                || self.post_element_wise_expr.is_some())
        {
            return None;
        }
        for extra in &pre_extra_tensors {
            let layout = extra.layout();
            let column = extra.datatype() == DataTypeEnum::F32
                && layout.shape().len() == 1
                && layout.shape()[0] == input.layout().shape().last().copied().unwrap_or(0)
                && layout.offset() == 0
                && layout.strides() == [1];
            let pointwise = extra.datatype() == DataTypeEnum::F32
                && layout.shape() == input.layout().shape()
                && flatten_matrix_layout(layout).is_some();
            if !column && !pointwise {
                return None;
            }
        }
        for extra in &post_extra_tensors {
            let layout = extra.layout();
            let column = extra.datatype() == DataTypeEnum::F32
                && layout.shape().len() == 1
                && layout.shape()[0] == output.layout().shape().last().copied().unwrap_or(0)
                && layout.offset() == 0
                && layout.strides() == [1];
            let pointwise = extra.datatype() == DataTypeEnum::F32
                && layout.shape() == output.layout().shape()
                && flatten_matrix_layout(layout).is_some();
            if !column && !pointwise {
                return None;
            }
        }
        if matches!(matrix.datatype(), GgmlType::F32 | GgmlType::F16) {
            // The dense kernel has no epilogue slots; fusion never attaches
            // epilogues to dense-storage operations (see
            // `supports_elementwise_epilogue_fusion`).
            if self.pre_element_wise_expr.is_some() || self.post_element_wise_expr.is_some() {
                return None;
            }
            return self.build_dense_direct_kernel(graph, input, matrix, output);
        }
        Self::direct_kernel_for_tensors(
            &graph.device(),
            DirectKernelTensors {
                input,
                matrix,
                pre_extra_tensors: &pre_extra_tensors,
                post_extra_tensors: &post_extra_tensors,
                output,
            },
            self.name(),
            DirectKernelChains {
                pre_expr: self.pre_element_wise_expr.as_ref(),
                post_expr: self.post_element_wise_expr.as_ref(),
                post_accumulator_offsets: &self.post_accumulator_offsets,
            },
            Some((self, inputs)),
        )
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        qmatmul_operation_output(inputs)
    }

    fn name(&self) -> String {
        qmatmul_operation_name("mul", self.input_datatype, &self.in_shape, &self.matrix)
    }
}
