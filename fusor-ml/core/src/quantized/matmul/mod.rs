use crate::{
    DataType, DataTypeEnum, Device, Layout, Tensor, TensorData,
    compute_graph::NodeIndex,
    mir::{inputs::MirValue, kernel::GenericKernel, operation::Operation},
};
use fusor_gguf::GgmlType;

use super::QMatrix;

mod sgemm;
mod sgemv;

pub use sgemm::{ChunkedSgemmConfig, GeneralSgemmConfig};

#[derive(Debug, Clone)]
pub(crate) struct QMatMulOperation {
    pub(crate) input_datatype: DataTypeEnum,
    pub(crate) input: NodeIndex,
    pub(crate) matrix: QMatrix,
    pub(crate) in_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) chunked_config: Option<ChunkedSgemmConfig>,
    pub(crate) general_config: Option<GeneralSgemmConfig>,
}

impl QMatMulOperation {
    pub(crate) fn new(
        input_datatype: DataTypeEnum,
        input_shape: &[usize],
        input: NodeIndex,
        matrix: QMatrix,
    ) -> Self {
        let last_dim = input_shape.len() - 1;
        let mut out_shape = input_shape.to_vec();
        out_shape[last_dim] = matrix.shape[0];
        assert_eq!(input_shape[last_dim], matrix.shape[1]);
        let out_shape = out_shape.into_boxed_slice();
        QMatMulOperation {
            input_datatype,
            input,
            matrix,
            in_shape: input_shape.into(),
            out_shape,
            chunked_config: None,
            general_config: None,
        }
    }

    fn elements_per_block(&self) -> u32 {
        self.matrix.datatype.block_size() as u32
    }

    fn sgemv(&self) -> bool {
        let m_dim_idx = self.in_shape.len() - 2;
        let m = self.in_shape[m_dim_idx];
        // Use SGEMV for tall and skinny matrices (small M, any K)
        // SGEMV is more efficient when M is small because:
        // - Each workgroup processes one M value independently
        // - Less workgroup synchronization overhead
        // - Better cache utilization for the K dimension
        // SGEMM becomes more efficient for larger M where it can use
        // tile-based processing with 16x16 workgroups
        m <= 32
    }

    fn m_size(&self) -> u32 {
        let m_dim_idx = self.in_shape.len() - 2;
        self.in_shape[m_dim_idx] as u32
    }

    fn n_size(&self) -> u32 {
        self.matrix.shape[0] as u32
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        // For F16/F32 matrices, dequantize and use regular mat_mul
        // because they don't have block structure like quantized types
        if matches!(other.datatype(), GgmlType::F16 | GgmlType::F32) {
            let dequantized: Tensor<2, T> = other.dequantize();

            // The weight matrix is [out_features, in_features], need to transpose for mat_mul
            // self: [..., M, K] @ weight.T: [K, N] -> [..., M, N]
            // Reshape weight to add batch dimensions: [1, 1, ..., K, N]
            let weight_t: Tensor<2, T> =
                dequantized.restride_layout(Layout::contiguous(other.shape()).transpose(0, 1));

            // Create batch dimensions for the weight
            let weight_shape: [usize; R] = std::array::from_fn(|i| {
                if i < R - 2 {
                    1 // Broadcast batch dimensions
                } else if i == R - 2 {
                    other.shape()[1] // K dimension
                } else {
                    other.shape()[0] // N dimension
                }
            });
            let weight_reshaped: Tensor<R, T> = weight_t.reshape(weight_shape);

            // Broadcast to match input batch dimensions
            let self_shape = self.shape();
            let broadcast_shape: [usize; R] = std::array::from_fn(|i| {
                if i < R - 2 {
                    self_shape[i] // Match input batch dimensions
                } else if i == R - 2 {
                    other.shape()[1] // K dimension
                } else {
                    other.shape()[0] // N dimension
                }
            });
            let weight_broadcast: Tensor<R, T> = weight_reshaped.broadcast_as(broadcast_shape);

            return self.mat_mul(&weight_broadcast);
        }
        self.add_q_mat_mul(other)
    }
}

impl Operation for QMatMulOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        if self.sgemv() {
            sgemv::workgroup_shape_constraints(&self.matrix, device)
        } else {
            sgemm::workgroup_shape_constraints(&self.matrix, device)
        }
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
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

        if self.sgemv() {
            sgemv::dispatch_size(&self.matrix, n, m, batch_size)
        } else {
            sgemm::dispatch_size(self, workgroup_shape, &self.matrix, n, m, batch_size)
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_result(self.input).unwrap();
        let q_matrix = self.matrix.clone();
        let device = input.device();
        let output_tensor = TensorData::new_for_shape(device, &self.out_shape, input.datatype());
        vec![input.into(), q_matrix.into(), output_tensor.into()]
    }

    // Related files/PRs in llama.cpp for reference:
    // https://github.com/ggml-org/llama.cpp/pull/2290
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L4561
    // https://github.com/ggml-org/llama.cpp/blob/add2a3aa5a1571211aa5c7303b8e80c8d1824b91/ggml/src/ggml-metal/ggml-metal.metal#L5881
    // based on https://siboehm.com/articles/22/CUDA-MMM
    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
        generic_kernel: &mut GenericKernel,
    ) {
        let datatype = self.input_datatype;
        let rank = self.in_shape.len() as u32;
        let matrix_rank = self.matrix.shape.len() as u32;

        let input_a = generic_kernel.add_tensor_input(rank, false, datatype);
        let input_b = generic_kernel.add_q_matrix_input(matrix_rank, self.matrix.datatype);
        let output = generic_kernel.add_tensor_input(rank, true, datatype);

        // For batched operations, we need to get the correct dimension indices
        let k_size = input_a.shape_binding(rank - 1).to_string(); // Last dimension is K
        let m_size = input_a.shape_binding(rank - 2).to_string(); // Second-to-last dimension is M
        let n_size = input_b.shape_binding(0).to_string();

        // Check if this is a sgemv or sgemm operation
        let algo = if self.sgemv() {
            sgemv::sgemv
        } else {
            sgemm::sgemm
        };

        algo(
            self,
            generic_kernel,
            workgroup_shape,
            &input_a,
            &input_b,
            &output,
            &n_size,
            &m_size,
            &k_size,
            graph,
        );
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        let output_tensor = inputs[2].as_tensor().unwrap();
        output_tensor.clone().into()
    }

    fn name(&self) -> String {
        format!(
            "q_mat_mul_{}_{}_{}_{}",
            self.input_datatype,
            self.in_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.matrix.datatype,
            self.matrix
                .shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}
