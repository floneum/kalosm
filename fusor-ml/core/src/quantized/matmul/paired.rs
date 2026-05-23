use super::*;

impl QMatMulOperation {
    pub(super) fn build_paired_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        paired: &PairedConfig,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let extras_count = paired.extras.len();
        if inputs.len() != 2 + extras_count + 1 {
            return None;
        }
        let input = inputs[0].as_tensor()?;
        let MirValue::QMatrix(matrix) = &inputs[1] else {
            return None;
        };
        let extras_tensors: Vec<&TensorData> = inputs[2..2 + extras_count]
            .iter()
            .map(|v| v.as_tensor())
            .collect::<Option<Vec<_>>>()?;
        let output = inputs[2 + extras_count].as_tensor()?;
        if input.datatype() != DataTypeEnum::F32 || output.datatype() != DataTypeEnum::F32 {
            return None;
        }
        for extra in &extras_tensors {
            if extra.datatype() != DataTypeEnum::F32 || extra.layout().shape() != [paired.pair_len]
            {
                return None;
            }
        }
        // Paired only supports Q4K today.
        if matrix.datatype() != GgmlType::Q4K {
            return None;
        }
        // `qgemv_q4k_paired` emits `subgroup_id` / `subgroup_reduce_*` ops
        // with no workgroup-only fallback yet, so adapters without
        // a trusted subgroup path trip shader validation.
        let device = graph.device();
        if !device.subgroups_supported() {
            return None;
        }
        let format = tile_ir::GgmlQuantFormat::Q4K;
        let a_view = flatten_matrix_layout(input.layout())?;
        let y_view = flatten_matrix_layout(output.layout())?;
        let m = a_view.rows;
        let k = a_view.cols;
        let pair_len = paired.pair_len as u32;
        if m == 0
            || y_view.rows != m
            || y_view.cols != pair_len
            || k != self.matrix.shape[1] as u32
            || self.matrix.shape[0] as u32 != pair_len.checked_mul(2)?
        {
            return None;
        }

        let limits = graph.device().limits();
        let max_workgroups = effective_qmatmul_max_workgroups_per_dimension(&limits);
        let (dispatch_size, workgroups_x, paired_shape) =
            tile_ir_kernels::qgemv_q4k_paired_dispatch(pair_len, m, max_workgroups)?;

        let epilogue = paired.epilogue.clone();
        let paired_identity = {
            let mut hasher = FxHasher::default();
            epilogue.identity().hash(&mut hasher);
            paired_shape.hash(&mut hasher);
            hasher.finish()
        };
        let kernel_name = self.name();

        // Fast path: no extras → existing storage3 cached-pipeline path.
        if extras_count == 0 {
            let pipeline_key = QMatMulDirectPipelineKey::new_with_epilogue(
                matrix.datatype(),
                crate::quantized::QMatMulShape { m, k, n: pair_len },
                paired_identity,
                dispatch_size,
                input.layout(),
                output.layout(),
            );
            let cache_variant = kernel_backend::KernelVariantKey::with_payload::<
                QMatmulPairedKernelVariant,
            >(|state| {
                paired_identity.hash(state);
            });
            let cache_key =
                self.kernel_cache_key_with_dispatch(cache_variant, None, dispatch_size, inputs);
            return qmatmul_direct_kernel_from_ir(
                &graph.device(),
                "q_mat_paired".to_owned(),
                kernel_name,
                cache_key,
                matrix,
                pipeline_key,
                input,
                &[],
                &[],
                output,
                dispatch_size,
                || {
                    Some(tile_ir::tile::build(move |phase| {
                        let a = tile_storage_read_with_direct_layout(phase, a_view);
                        let b = tile_ir_kernels::quantized_matrix(phase, format, k, pair_len * 2);
                        let y = tile_storage_write_with_direct_layout(phase, y_view);
                        tile_ir_kernels::qgemv_q4k_paired(
                            phase,
                            tile_ir_kernels::Q4KPairedGgml {
                                a: &a,
                                b: &b,
                                y: &y,
                                pair_cols: pair_len,
                                m_rows: m,
                                workgroups_x,
                                shape: paired_shape,
                                epilogue: &epilogue,
                                extras: &[],
                            },
                        );
                    }))
                },
            );
        }

        // Extras path: build the IR with `(3 + extras_count)` storage bindings
        // and dispatch via `dynamic_kernel_from_ir`, which derives binding
        // counts from the lowered module. The storage3-specialized fast path
        // assumes exactly 3 bindings and doesn't apply here.

        // Convert each extra's host-side tensor layout (`fusor_types::Layout`)
        // into a 1D `tile_ir::Layout` suitable for `storage_read_with_layout`,
        // preserving its element stride and offset.
        struct ExtraView {
            tile_layout: tile_ir::Layout,
            offset: u32,
        }
        let extra_views: Option<Vec<ExtraView>> = extras_tensors
            .iter()
            .map(|t| {
                let shape = t.layout().shape();
                let strides = t.layout().strides();
                let length: u32 = (*shape.first()?).try_into().ok()?;
                let stride: u32 = (*strides.first()?).try_into().ok()?;
                let offset: u32 = t.layout().offset().try_into().ok()?;
                Some(ExtraView {
                    tile_layout: tile_ir::Layout::strided(
                        tile_ir::MemoryLevel::Storage,
                        tile_ir::Shape::new([length]),
                        &[stride],
                    ),
                    offset,
                })
            })
            .collect();
        let extra_views = extra_views?;
        let cache_variant = kernel_backend::KernelVariantKey::with_payload::<
            QMatmulPairedExtrasKernelVariant,
        >(|state| {
            paired_identity.hash(state);
        });
        let cache_key =
            self.kernel_cache_key_with_dispatch(cache_variant, None, dispatch_size, inputs);
        // Build IR + binding list together via `KernelBuilder` so the runtime
        // buffer order can't drift from the IR's storage declaration order.
        let mut kb = tile_ir::KernelBuilder::<std::sync::Arc<wgpu::Buffer>>::new();
        let a = kb.read::<tile_ir::F32, 2>(tile_ir::KernelTensorRef::with_offset(
            input.buffer().clone(),
            a_view.layout.clone(),
            a_view.offset,
        ));
        let b = tile_ir_kernels::quantized_matrix_for(
            &mut kb,
            matrix.buffer().clone(),
            format,
            k,
            pair_len * 2,
        );
        let extras: Vec<tile_ir::tile::Storage<tile_ir::F32, 1>> = extra_views
            .iter()
            .zip(extras_tensors.iter())
            .map(|(view, tensor)| {
                kb.read::<tile_ir::F32, 1>(tile_ir::KernelTensorRef::with_offset(
                    tensor.buffer().clone(),
                    view.tile_layout.clone(),
                    view.offset,
                ))
            })
            .collect();
        let y = kb.write::<tile_ir::F32, 2>(tile_ir::KernelTensorRef::with_offset(
            output.buffer().clone(),
            y_view.layout.clone(),
            y_view.offset,
        ));
        tile_ir_kernels::qgemv_q4k_paired(
            kb.program(),
            tile_ir_kernels::Q4KPairedGgml {
                a: &a,
                b: &b,
                y: &y,
                pair_cols: pair_len,
                m_rows: m,
                workgroups_x,
                shape: paired_shape,
                epilogue: &epilogue,
                extras: &extras,
            },
        );
        let (ir, buffers) = kb.finish();

        kernel_backend::dynamic_kernel_from_ir(
            graph.device().kernel_cache(),
            kernel_name,
            cache_key,
            move || Some(ir),
            buffers,
            dispatch_size,
        )
    }
}

impl QMatMulOperation {
    pub(crate) fn build_direct_kernels(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Result<QMatMulKernelPlan, QMatMulLoweringError> {
        if inputs
            .last()
            .and_then(MirValue::as_tensor)
            .is_some_and(|output| output.layout().shape().contains(&0))
        {
            return Ok(QMatMulKernelPlan::EmptyOutput);
        }

        if let Some(kernel) = self.build_direct_kernel(graph, workgroup_shape, inputs) {
            return Ok(QMatMulKernelPlan::Kernels(vec![kernel]));
        }

        self.build_dequantize_dense_fallback_direct_kernels(graph, inputs)
            .and_then(QMatMulKernelPlan::from_kernels)
            .ok_or_else(|| QMatMulLoweringError::new(self.name()))
    }

    pub(super) fn build_dense_direct_kernel(
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
        let dense_weight_t = TensorData::new_from_parts(
            matrix.device(),
            matrix.buffer().clone(),
            Layout::from_parts(
                0,
                dense_shape.into_boxed_slice(),
                dense_strides.into_boxed_slice(),
            ),
            DataTypeEnum::F32,
        );
        let device = graph.device();
        let dense_matmul = MatMulOperation::new(
            DataTypeEnum::F32,
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
                .solve(device.max_subgroup_size())?,
            &[
                input.clone().into(),
                dense_weight_t.into(),
                output.clone().into(),
            ],
        )
    }

    fn build_dense_qmatmul_fallback_direct_kernels(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        input: &TensorData,
        matrix: &QMatrix,
        output: &TensorData,
    ) -> Option<Vec<DirectKernel>> {
        if input.datatype() != DataTypeEnum::F32 || output.datatype() != DataTypeEnum::F32 {
            return None;
        }
        if matrix.datatype() == GgmlType::F32 {
            return self
                .build_dense_direct_kernel(graph, input, matrix, output)
                .map(|kernel| vec![kernel]);
        }

        let dense_weight =
            TensorData::new_for_shape(&graph.device(), matrix.shape(), DataTypeEnum::F32);
        let dequantize = DequantizeOperation::new(matrix.clone(), DataTypeEnum::F32);
        let dequantize_inputs = vec![matrix.clone().into(), dense_weight.clone().into()];
        let dequantize_workgroup = dequantize
            .workgroup_shape_constraints(&graph.device())
            .solve(graph.device().max_subgroup_size())?;
        let dequantize_kernel =
            dequantize.build_direct_kernel(graph, &dequantize_workgroup, &dequantize_inputs)?;
        let dense_matrix = QMatrix {
            device: graph.device(),
            shape: matrix.shape.clone(),
            buffer: dense_weight.buffer().clone(),
            datatype: GgmlType::F32,
            direct_pipeline_cache: matrix.direct_pipeline_cache.clone(),
        };
        let matmul_kernel = self.build_dense_direct_kernel(graph, input, &dense_matrix, output)?;
        Some(vec![dequantize_kernel, matmul_kernel])
    }

    fn build_nary_fallback_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        expression: NaryExpr,
        inputs: &[&TensorData],
        output: &TensorData,
        shape: &[usize],
        output_datatype: DataTypeEnum,
    ) -> Option<DirectKernel> {
        let operation = NaryOperation {
            inputs: (0..inputs.len()).map(NodeIndex::new).collect(),
            expression,
            shape: shape.into(),
            output_datatype,
        };
        let mut mir_inputs = inputs
            .iter()
            .map(|input| (*input).clone().into())
            .collect::<Vec<MirValue>>();
        mir_inputs.push(output.clone().into());
        let workgroup_shape = operation
            .workgroup_shape_constraints(&graph.device())
            .solve(graph.device().max_subgroup_size())?;
        operation.build_direct_kernel(graph, &workgroup_shape, &mir_inputs)
    }

    fn build_dequantize_dense_fallback_direct_kernels(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Option<Vec<DirectKernel>> {
        let [input, matrix, rest @ .., output] = inputs else {
            return None;
        };
        let mut input = input.as_tensor()?.clone();
        let MirValue::QMatrix(matrix) = matrix else {
            return None;
        };
        let output = output.as_tensor()?.clone();

        if self.paired.is_some() {
            return self.build_paired_dequantize_dense_fallback_direct_kernels(
                graph, &input, matrix, rest, &output, inputs,
            );
        }

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
        if rest.len() != pre_extra_count + post_extra_count {
            return None;
        }
        let extra_tensors = rest
            .iter()
            .map(MirValue::as_tensor)
            .collect::<Option<Vec<_>>>()?;
        let pre_extra_tensors = &extra_tensors[..pre_extra_count];
        let post_extra_tensors = &extra_tensors[pre_extra_count..];

        let mut kernels = Vec::new();
        if let Some(pre) = &self.pre_element_wise_expr {
            let pre_output = TensorData::new_for_shape(
                &graph.device(),
                input.layout().shape(),
                pre.output_datatype,
            );
            let mut nary_inputs = Vec::with_capacity(1 + pre_extra_tensors.len());
            nary_inputs.push(&input);
            nary_inputs.extend(pre_extra_tensors.iter().copied());
            kernels.push(self.build_nary_fallback_direct_kernel(
                graph,
                pre.expression.clone(),
                &nary_inputs,
                &pre_output,
                input.layout().shape(),
                pre.output_datatype,
            )?);
            input = pre_output;
        }

        let matmul_output = if self.post_element_wise_expr.is_some() {
            TensorData::new_for_shape(&graph.device(), output.layout().shape(), DataTypeEnum::F32)
        } else {
            output.clone()
        };
        kernels.extend(self.build_dense_qmatmul_fallback_direct_kernels(
            graph,
            &input,
            matrix,
            &matmul_output,
        )?);

        if let Some(post) = &self.post_element_wise_expr {
            let mut nary_inputs = Vec::with_capacity(1 + post_extra_tensors.len());
            nary_inputs.push(&matmul_output);
            nary_inputs.extend(post_extra_tensors.iter().copied());
            kernels.push(self.build_nary_fallback_direct_kernel(
                graph,
                post.expression.clone(),
                &nary_inputs,
                &output,
                output.layout().shape(),
                post.output_datatype,
            )?);
        }

        Some(kernels)
    }

    fn build_paired_dequantize_dense_fallback_direct_kernels(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        input: &TensorData,
        matrix: &QMatrix,
        extras: &[MirValue],
        output: &TensorData,
        operation_inputs: &[MirValue],
    ) -> Option<Vec<DirectKernel>> {
        let paired = self.paired.as_ref()?;
        if extras.len() != paired.extras.len()
            || input.datatype() != DataTypeEnum::F32
            || output.datatype() != DataTypeEnum::F32
            || matrix.shape()[0] != paired.pair_len * 2
            || matrix.shape()[1] != *input.layout().shape().last()?
        {
            return None;
        }
        let extras = extras
            .iter()
            .map(MirValue::as_tensor)
            .collect::<Option<Vec<_>>>()?;
        for extra in &extras {
            if extra.datatype() != DataTypeEnum::F32 || extra.layout().shape() != [paired.pair_len]
            {
                return None;
            }
        }

        let mut projected_shape = input.layout().shape().to_vec();
        *projected_shape.last_mut()? = matrix.shape()[0];
        let projected =
            TensorData::new_for_shape(&graph.device(), &projected_shape, DataTypeEnum::F32);

        let mut kernels =
            self.build_dense_qmatmul_fallback_direct_kernels(graph, input, matrix, &projected)?;
        kernels.push(self.build_paired_dense_fallback_epilogue_kernel(
            graph,
            paired,
            &projected,
            &extras,
            output,
            operation_inputs,
        )?);
        Some(kernels)
    }

    fn build_paired_dense_fallback_epilogue_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        paired: &PairedConfig,
        projected: &TensorData,
        extras: &[&TensorData],
        output: &TensorData,
        operation_inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let projected_view = flatten_matrix_layout(projected.layout())?;
        let output_view = flatten_matrix_layout(output.layout())?;
        let rows = projected_view.rows;
        let pair_len: u32 = paired.pair_len.try_into().ok()?;
        if rows == 0
            || pair_len == 0
            || projected_view.cols != pair_len.checked_mul(2)?
            || output_view.rows != rows
            || output_view.cols != pair_len
        {
            return None;
        }

        struct ExtraView {
            layout: tile_ir::Layout,
            offset: u32,
        }
        let extra_views = extras
            .iter()
            .map(|extra| {
                let shape = extra.layout().shape();
                let strides = extra.layout().strides();
                let length: u32 = (*shape.first()?).try_into().ok()?;
                let stride: u32 = (*strides.first()?).try_into().ok()?;
                let offset: u32 = extra.layout().offset().try_into().ok()?;
                Some(ExtraView {
                    layout: tile_ir::Layout::strided(
                        tile_ir::MemoryLevel::Storage,
                        tile_ir::Shape::new([length]),
                        &[stride],
                    ),
                    offset,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        const BLOCK: usize = 256;
        let total_outputs = rows.checked_mul(pair_len)?;
        let workgroups = total_outputs.div_ceil(BLOCK as u32);
        let max_workgroups =
            effective_qmatmul_max_workgroups_per_dimension(&graph.device().limits());
        let [dispatch_x, dispatch_y] = split_workgroups_2d(workgroups, max_workgroups)?;
        let dispatch_size = [dispatch_x, dispatch_y, 1];

        let epilogue = paired.epilogue.clone();
        let mut kb = tile_ir::KernelBuilder::<std::sync::Arc<wgpu::Buffer>>::new();
        let projected_storage = kb.read::<tile_ir::F32, 2>(tile_ir::KernelTensorRef::with_offset(
            projected.buffer().clone(),
            projected_view.layout,
            projected_view.offset,
        ));
        let extra_storages = extra_views
            .iter()
            .zip(extras.iter())
            .map(|(view, extra)| {
                kb.read::<tile_ir::F32, 1>(tile_ir::KernelTensorRef::with_offset(
                    extra.buffer().clone(),
                    view.layout.clone(),
                    view.offset,
                ))
            })
            .collect::<Vec<_>>();
        let output_storage = kb.write::<tile_ir::F32, 2>(tile_ir::KernelTensorRef::with_offset(
            output.buffer().clone(),
            output_view.layout,
            output_view.offset,
        ));
        kb.program()
            .program_grid::<BLOCK>(dispatch_size, |program| {
                let lane = program.lane();
                let group = program.program_id(tile_ir::WorkgroupAxis::X)
                    + program.program_id(tile_ir::WorkgroupAxis::Y) * dispatch_x;
                let flat_index = group * BLOCK as u32 + lane;
                let in_bounds = flat_index.clone().lt(total_outputs);
                let row = flat_index.clone() / pair_len;
                let col = flat_index % pair_len;
                let gate = program.load(
                    projected_storage.at((row.clone(), col.clone())),
                    in_bounds.clone(),
                    0.0,
                );
                let up = program.load(
                    projected_storage.at((row.clone(), col.clone() + pair_len)),
                    in_bounds.clone(),
                    0.0,
                );
                let extra_tiles = extra_storages
                    .iter()
                    .map(|extra| program.load(extra.at(col.clone()), in_bounds.clone(), 0.0))
                    .collect::<Vec<_>>();
                let value = epilogue.apply(gate, up, &extra_tiles);
                program.store(output_storage.at((row, col)), value, in_bounds);
            });
        let (ir, buffers) = kb.finish();
        let cache_variant = kernel_backend::KernelVariantKey::with_payload::<
            QMatmulPairedDenseFallbackKernelVariant,
        >(|state| {
            paired.epilogue.identity().hash(state);
            QMATMUL_DIRECT_KERNEL_GENERATION.hash(state);
        });
        let cache_key = self.kernel_cache_key_with_dispatch(
            cache_variant,
            None,
            dispatch_size,
            operation_inputs,
        );
        kernel_backend::dynamic_kernel_from_ir(
            graph.device().kernel_cache(),
            format!("{}_dense_paired_epilogue", self.name()),
            cache_key,
            move || Some(ir),
            buffers,
            dispatch_size,
        )
    }
}
