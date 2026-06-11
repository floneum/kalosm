use super::*;

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

    fn build_dense_qmatmul_fallback_direct_kernels(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        input: &TensorData,
        matrix: &QMatrix,
        output: &TensorData,
    ) -> Option<Vec<DirectKernel>> {
        if input.datatype() != output.datatype() {
            return None;
        }
        if matches!(matrix.datatype(), GgmlType::F32 | GgmlType::F16) {
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
            .solve(graph.device().max_subgroup_size(), &graph.device().limits())?;
        let dequantize_kernel =
            dequantize.build_direct_kernel(graph, &dequantize_workgroup, &dequantize_inputs)?;
        let dense_matrix = QMatrix {
            device: graph.device(),
            shape: matrix.shape.clone(),
            buffer: dense_weight.buffer().clone(),
            datatype: GgmlType::F32,
            storage_layout: QMatrixStorageLayout::Native,
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
        let operation = ElementwiseOperation {
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
            .solve(graph.device().max_subgroup_size(), &graph.device().limits())?;
        operation.build_direct_kernel(graph, &workgroup_shape, &mir_inputs)
    }

    fn build_dequantize_dense_fallback_direct_kernels(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Option<Vec<DirectKernel>> {
        if !self.post_accumulator_offsets.is_empty() {
            return None;
        }
        let [input, matrix, rest @ .., output] = inputs else {
            return None;
        };
        let mut input = input.as_tensor()?.clone();
        let MirValue::QMatrix(matrix) = matrix else {
            return None;
        };
        let output = output.as_tensor()?.clone();

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
}
