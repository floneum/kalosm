use std::hash::Hash;

use super::*;

impl QMatMulOperation {
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
        if m != y_m || k != matrix.shape[1] as u32 || n != matrix.shape[0] as u32 {
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
        let limits = device.limits();
        let caps = KernelDeviceCaps::from_device(device);
        let max_workgroups = effective_qmatmul_max_workgroups_per_dimension(&limits);
        let mut qmatmul_workgroups_x = 1;
        let y_supports_coop = tile_cooperative_store_layout_supported(&y_view.layout);
        let variant = select_qmatmul_direct_variant(format, m, k, n, y_supports_coop, caps);
        // Every direct multi-row variant relies on cooperative-matrix or
        // subgroup-specialized IR. Route devices without the F32 coop path
        // through the workgroup-tiled qmatmul/qgemv variants.
        let use_workgroup_qmatmul = !qmatmul_coop_supported(caps) || f16_storage;
        let use_f16_workgroup_tiles = f16_storage;
        let use_coop_acc_init_epilogue = !use_workgroup_qmatmul
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
            Some(tile_ir_kernels::UnaryEpilogueWithExtras::new(
                "qmatmul_post_expr",
                post_extra_tensors.len(),
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
                    .expect("post expression validated at fuse time")
                    .0
                },
            ))
        } else {
            None
        };
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
            ^ if f16_storage { 0xF16F_0001u64 } else { 0 };
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
                } if tile == QCoopTile::new(64, 64) => None,
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
                crate::quantized::QMatMulShape { m, k, n },
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
            let cache_key = qmatmul_direct_module_key::<QMatmulDirectFastKernelVariant>(
                |state| {
                    variant.hash(state);
                    QMATMUL_DIRECT_KERNEL_GENERATION.hash(state);
                },
                |state| {
                    QMATMUL_DIRECT_KERNEL_GENERATION.hash(state);
                    hash_qmatmul_shape(state, format, m, k, n);
                    hash_qmatmul_dispatch_layouts(
                        state,
                        dispatch_size,
                        input.layout(),
                        output.layout(),
                    );
                },
                dispatch_size,
                operation_key,
            );
            if let Some(pipeline) = kernel_backend::three_buffer_pipeline_from_cached_module(
                device.kernel_cache(),
                &kernel_name,
                cache_key,
            ) {
                matrix
                    .direct_pipeline_cache()
                    .write()
                    .get_or_insert(pipeline_key, || pipeline.clone());
                return Some(
                    kernel_backend::DirectKernel::from_prepared_three_buffer_pipeline(
                        kernel_name.clone(),
                        pipeline,
                        input.buffer().clone(),
                        matrix.buffer().clone(),
                        output.buffer().clone(),
                        dispatch_size,
                    ),
                );
            }
        }
        let pre_with_extras_for_ir = pre_epilogue_with_extras.clone();
        let post_with_extras_for_ir = post_epilogue_with_extras.clone();
        let ir = tile_ir::tile::build(move |phase| {
            if f16_storage {
                let a = tile_storage_read_with_direct_layout_typed::<tile_ir::F16>(phase, a_view);
                let b = tile_ir_kernels::quantized_matrix(phase, format, k, n);
                let y = tile_storage_write_with_direct_layout_typed::<tile_ir::F16>(phase, y_view);
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
            let b = tile_ir_kernels::quantized_matrix(phase, format, k, n);
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
                        QmatmulExtraStorage::Column(phase.storage_read::<tile_ir::F32, 1>(
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
                        QmatmulExtraStorage::Column(phase.storage_read::<tile_ir::F32, 1>(
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
            // Map the selected variant to its (BM, BN) cooperative tile
            // dimensions. The first two single-row variants short-circuit to
            // qgemv; the rest share the qmatmul_with_epilogue entry point.
            // (BK is pinned to 32 inside the coop dispatcher.)
            let tile = match variant {
                QMatmulPath::Q5SmallSingleRow | QMatmulPath::SingleRow => {
                    tile_ir_kernels::qgemv_with_epilogue(
                        phase,
                        &a,
                        &b,
                        &y,
                        qmatmul_workgroups_x,
                        &epilogues,
                    );
                    return;
                }
                QMatmulPath::Q8Wide(tile) | QMatmulPath::Tile { tile, .. } => tile,
            };
            tile_ir_kernels::qmatmul_with_epilogue(phase, &a, &b, &y, &epilogues, tile.bm, tile.bn);
        });
        let dispatch_size = ir.body().grid;
        if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
            return None;
        }
        let pipeline_key = QMatMulDirectPipelineKey::new_with_epilogue(
            matrix.datatype(),
            matrix.storage_layout(),
            crate::quantized::QMatMulShape { m, k, n },
            epilogue_identity,
            dispatch_size,
            input.layout(),
            output.layout(),
        );
        let cache_key = qmatmul_direct_module_key::<QMatmulDirectEpilogueKernelVariant>(
            |state| {
                variant.hash(state);
                epilogue_identity.hash(state);
                QMATMUL_DIRECT_KERNEL_GENERATION.hash(state);
            },
            |state| {
                QMATMUL_DIRECT_KERNEL_GENERATION.hash(state);
                hash_qmatmul_shape(state, format, m, k, n);
                epilogue_identity.hash(state);
                hash_qmatmul_dispatch_layouts(
                    state,
                    dispatch_size,
                    input.layout(),
                    output.layout(),
                );
            },
            dispatch_size,
            operation_key,
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

impl Operation for QMatMulOperation {
    fn workgroup_shape_constraints(
        &self,
        _device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        if self.paired.is_some() {
            // Paired qgemv kernels are single-thread dispatched along x; the
            // tile shape is set inside the kernel.
            constraints.add_constraint(0, Constraint::Equals(1));
        } else if self.m_size() == 1 {
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
        if let Some(paired) = &self.paired {
            return [paired.pair_len as u32, self.m_size(), 1];
        }
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
        if let Some(paired) = &self.paired {
            for extra in &paired.extras {
                f(*extra);
            }
        }
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let base = qmatmul_operation_inputs(self.input, &self.matrix, &self.out_shape, nodes);
        if self.paired.is_none() {
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
                result.push(nodes.get_cached_result(*extra).unwrap().clone().into());
            }
            result.extend_from_slice(tail);
            return result;
        }
        let Some(paired) = &self.paired else {
            return base;
        };
        if paired.extras.is_empty() {
            return base;
        }
        // [input, qmatrix, extras..., output] — splice extras between qmatrix
        // and output so the layout stays a strict superset of the no-extras
        // case and `qmatmul_operation_output` still pattern-matches the tail.
        let mut result = Vec::with_capacity(base.len() + paired.extras.len());
        let (head, tail) = base.split_at(2);
        result.extend_from_slice(head);
        for extra in &paired.extras {
            result.push(nodes.get_cached_result(*extra).unwrap().clone().into());
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
        if let Some(paired) = &self.paired {
            return self.build_paired_direct_kernel(graph, paired, inputs);
        }
        if inputs.len() < 3 {
            return None;
        }
        let input = inputs[0].as_tensor()?;
        let MirValue::QMatrix(matrix) = &inputs[1] else {
            return None;
        };
        let output = inputs.last()?.as_tensor()?;
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
            },
            Some((self, inputs)),
        )
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        qmatmul_operation_output(inputs)
    }

    fn name(&self) -> String {
        let op_label = self
            .paired
            .as_ref()
            .map(|p| p.epilogue.label())
            .unwrap_or("mul");
        qmatmul_operation_name(op_label, self.input_datatype, &self.in_shape, &self.matrix)
    }
}
