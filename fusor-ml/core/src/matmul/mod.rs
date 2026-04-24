use crate::matmul::sgemm_params::gemm_parameters;
use crate::matmul::sgemv_params::gemv_parameters;
use crate::mir::operation::Operation;
use crate::{
    Device, ElementWiseFunctions, Tensor,
    compute_graph::NodeIndex,
    mir::kernel::GenericKernel,
    tensor::{DataType, DataTypeEnum, TensorData},
};

pub mod sgemm;
mod sgemm_params;
pub mod sgemv;
mod sgemv_params;

pub fn get_optimal_params(m: usize, n: usize, k: usize) -> MatMulParams {
    match (m, n, k) {
        (_, 0..=64, _) => MatMulParams::Vector(gemv_parameters(m, n, k)),
        (_, _, _) => MatMulParams::MatMul(gemm_parameters(m, n, k)),
    }
}

#[derive(Debug, Clone)]
pub enum MatMulParams {
    Vector(sgemv::SgemvParams),
    MatMul(sgemm::SgemmParams),
}

#[derive(Debug, Clone)]
pub(crate) struct MatMulOperation {
    pub(crate) datatype: DataTypeEnum,
    pub(crate) first: NodeIndex,
    pub(crate) second: NodeIndex,
    pub(crate) first_shape: Box<[usize]>,
    pub(crate) second_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) pre_element_wise: [ElementWiseFunctions; 2],
    pub(crate) post_element_wise: ElementWiseFunctions,
    pub(crate) parameters: MatMulParams,
}

impl MatMulOperation {
    pub fn new(
        datatype: DataTypeEnum,
        first: NodeIndex,
        second: NodeIndex,
        first_shape: &[usize],
        second_shape: &[usize],
        parameters: Option<MatMulParams>,
    ) -> Self {
        // Check if this is a matrix-vector multiplication (second matrix has 1 column and first matrix has multiple rows)
        let parameters = parameters.unwrap_or_else(|| {
            let n = second_shape[second_shape.len() - 1];
            let m = first_shape[first_shape.len() - 2];
            let k = first_shape[first_shape.len() - 1];
            get_optimal_params(m, n, k)
        });
        Self::new_with_parameters(
            datatype,
            first,
            second,
            first_shape,
            second_shape,
            parameters,
        )
    }

    pub(crate) fn new_with_parameters(
        datatype: DataTypeEnum,
        first: NodeIndex,
        second: NodeIndex,
        first_shape: &[usize],
        second_shape: &[usize],
        parameters: MatMulParams,
    ) -> Self {
        let last_dim = first_shape.len() - 1;
        let second_to_last_dim = first_shape.len() - 2;
        let mut out_shape = first_shape.to_vec();
        out_shape[second_to_last_dim] = first_shape[second_to_last_dim];
        out_shape[last_dim] = second_shape[last_dim];
        assert_eq!(first_shape[last_dim], second_shape[second_to_last_dim]);
        assert!(
            first_shape
                .iter()
                .rev()
                .skip(2)
                .zip(second_shape.iter().rev().skip(2))
                .all(|(a, b)| a == b)
        );

        Self {
            first,
            second,
            first_shape: first_shape.into(),
            second_shape: second_shape.into(),
            out_shape: out_shape.into(),
            datatype,
            pre_element_wise: [
                ElementWiseFunctions::empty(datatype),
                ElementWiseFunctions::empty(datatype),
            ],
            post_element_wise: ElementWiseFunctions::empty(datatype),
            parameters,
        }
    }

    pub fn matmul_dtype(&self) -> DataTypeEnum {
        self.pre_element_wise[0].out_datatype()
    }

    pub fn rank(&self) -> u32 {
        self.out_shape.len() as u32
    }
}

impl Operation for MatMulOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        match &self.parameters {
            MatMulParams::Vector(sgemv_params) => {
                sgemv::workgroup_shape_constraints(self, device, sgemv_params)
            }
            MatMulParams::MatMul(sgemm_params) => {
                sgemm::workgroup_shape_constraints(self, device, sgemm_params)
            }
        }
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        let [input_a, input_b, _output] = inputs else {
            panic!("MatMulOperation requires 3 inputs");
        };
        let input_a = input_a.as_tensor().unwrap();
        let input_b = input_b.as_tensor().unwrap();
        let a_shape = input_a.layout().shape();
        let b_shape = input_b.layout().shape();
        let last_dim = self.rank() as usize - 1;
        let last_dim_size = b_shape[last_dim];
        let second_to_last_dim = self.rank() as usize - 2;
        let second_to_last_dim_size = a_shape[second_to_last_dim];
        let batch_size = a_shape.iter().rev().skip(2).product::<usize>();

        match &self.parameters {
            MatMulParams::Vector(sgemv_params) => sgemv::dispatch_size(
                second_to_last_dim_size as u32,
                last_dim_size as u32,
                batch_size as u32,
                workgroup_shape,
                sgemv_params,
            ),
            MatMulParams::MatMul(sgemm_params) => sgemm::dispatch_size(
                last_dim_size,
                second_to_last_dim_size,
                batch_size,
                workgroup_shape,
                sgemm_params,
            ),
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.first);
        f(self.second);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        let a = nodes.get_result(self.first).unwrap();
        let b = nodes.get_result(self.second).unwrap();
        let last_dim = self.rank() as usize - 1;
        let second_to_last_dim = self.rank() as usize - 2;
        let device = a.device();
        let a_shape = a.layout().shape();
        let b_shape = b.layout().shape();
        let mut out_shape = a_shape.to_vec();
        out_shape[second_to_last_dim] = a_shape[second_to_last_dim];
        out_shape[last_dim] = b_shape[last_dim];
        let output_tensor =
            TensorData::new_for_shape(device, &out_shape, self.post_element_wise.out_datatype());
        vec![a.into(), b.into(), output_tensor.into()]
    }

    // 1000x1000 dense matmul time on M2 mac pro 1.4743 ms
    fn build_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
        generic_kernel: &mut GenericKernel,
    ) {
        match &self.parameters {
            MatMulParams::Vector(sgemv_params) => {
                let [input_a, input_b, _] = inputs else {
                    panic!("MatMulOperation requires 3 inputs");
                };
                let input_a = input_a.as_tensor().unwrap();
                let input_b = input_b.as_tensor().unwrap();

                let input_a =
                    generic_kernel.add_tensor_input(self.rank(), false, input_a.datatype());
                let input_b =
                    generic_kernel.add_tensor_input(self.rank(), false, input_b.datatype());
                let output = generic_kernel.add_tensor_input(
                    self.rank(),
                    true,
                    self.post_element_wise.out_datatype(),
                );

                // Get dimension bindings
                let k_size = input_a.shape_binding(self.rank() - 1).to_string();
                let m_size = input_a.shape_binding(self.rank() - 2).to_string();
                let n_size = input_b.shape_binding(self.rank() - 1).to_string();

                sgemv::sgemv(
                    self,
                    generic_kernel,
                    workgroup_shape,
                    &input_a,
                    &input_b,
                    &output,
                    &n_size,
                    &m_size,
                    &k_size,
                    sgemv_params,
                    graph,
                )
            }
            MatMulParams::MatMul(sgemm_params) => sgemm::build_kernel(
                self,
                graph,
                workgroup_shape,
                inputs,
                generic_kernel,
                sgemm_params,
            ),
        }
    }

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> crate::mir::inputs::MirValue {
        let output_tensor = inputs[2].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!(
            "matmul_{}_{}_by_{}",
            self.datatype,
            self.first_shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.second_shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn mat_mul(&self, other: &Self) -> Self {
        self.add_mat_mul(other, None)
    }

    pub fn mat_mul_with_parameters(&self, other: &Self, parameters: MatMulParams) -> Self {
        self.add_mat_mul(other, Some(parameters))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_matrix_vector_mul() {
    let device = Device::test_instance();

    // Test matrix-vector multiplication: [2x3] * [3x1] = [2x1]
    let matrix = [[1., 2., 3.], [4., 5., 6.]];
    let vector = [[7.], [8.], [9.]];
    let tensor_matrix = Tensor::new(&device, &matrix);
    let tensor_vector = Tensor::new(&device, &vector);
    let result = tensor_matrix.mat_mul(&tensor_vector);
    let as_slice = result.as_slice().await.unwrap();

    // Expected: [1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9] = [50, 122]
    assert_eq!(as_slice[[0, 0]], 50.);
    assert_eq!(as_slice[[1, 0]], 122.);
}

#[cfg(test)]
#[tokio::test]
async fn test_matrix_vector_mul_non_contiguous() {
    let device = Device::test_instance();

    // Test with non-contiguous tensors
    let matrix = [[1., 2., 3., 10.], [4., 5., 6., 11.]];
    let vector = [[7.], [8.], [9.]];

    // Take a slice of the matrix to make it non-contiguous
    let tensor_matrix = Tensor::new(&device, &matrix).narrow(1, 0, 3);
    let tensor_vector = Tensor::new(&device, &vector);
    let result = tensor_matrix.mat_mul(&tensor_vector);
    let as_slice = result.as_slice().await.unwrap();

    // Expected: same as before since we removed the last column
    assert_eq!(as_slice[[0, 0]], 50.);
    assert_eq!(as_slice[[1, 0]], 122.);
}

#[cfg(test)]
#[tokio::test]
async fn test_large_skinny_k_matmul_matches_cpu_reference() {
    let device = Device::test_instance();
    let m = 100;
    let k = 3200;
    let n = 320;

    let lhs_data: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.001)
        .collect();
    let rhs_data: Vec<f32> = (0..k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.0015)
        .collect();
    let lhs_nested: Vec<Vec<f32>> = lhs_data.chunks_exact(k).map(|row| row.to_vec()).collect();
    let rhs_nested: Vec<Vec<f32>> = rhs_data.chunks_exact(n).map(|row| row.to_vec()).collect();
    let lhs = Tensor::new(&device, &lhs_nested);
    let rhs = Tensor::new(&device, &rhs_nested);
    let result = lhs.mat_mul(&rhs).as_slice().await.unwrap();

    let check_positions = [
        (0, 0),
        (0, n - 1),
        (m / 2, n / 2),
        (m - 1, 0),
        (m - 1, n - 1),
    ];

    for (row, col) in check_positions {
        let expected = (0..k)
            .map(|idx| lhs_data[row * k + idx] * rhs_data[idx * n + col])
            .sum::<f32>();
        let actual = result[[row, col]];
        assert!(
            (actual - expected).abs() < 1e-3,
            "Mismatch at [{row}, {col}]: actual={actual}, expected={expected}"
        );
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_moonshine_attention_projection_matmul_matches_cpu_reference() {
    let device = Device::test_instance();
    let batch = 1;
    let m = 100;
    let k = 320;
    let n = 320;

    let lhs_data: Vec<f32> = (0..batch * m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.002)
        .collect();
    let rhs_data: Vec<f32> = (0..batch * k * n)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.0015)
        .collect();
    let lhs_nested: Vec<Vec<Vec<f32>>> = lhs_data
        .chunks_exact(m * k)
        .map(|batch| batch.chunks_exact(k).map(|row| row.to_vec()).collect())
        .collect();
    let rhs_nested: Vec<Vec<Vec<f32>>> = rhs_data
        .chunks_exact(k * n)
        .map(|batch| batch.chunks_exact(n).map(|row| row.to_vec()).collect())
        .collect();
    let lhs = Tensor::new(&device, &lhs_nested);
    let rhs = Tensor::new(&device, &rhs_nested);
    let result = lhs.mat_mul(&rhs).as_slice().await.unwrap();

    for row in [0, m / 2, m - 1] {
        for col in [0, n / 2, n - 1] {
            let expected = (0..k)
                .map(|idx| lhs_data[row * k + idx] * rhs_data[idx * n + col])
                .sum::<f32>();
            let actual = result[[0, row, col]];
            assert!(
                (actual - expected).abs() < 1e-3,
                "Mismatch at [0, {row}, {col}]: actual={actual}, expected={expected}"
            );
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_multi_row_matrix_vector_mul() {
    let device = Device::test_instance();

    // Test matrix-vector multiplication with multiple rows: [3x2] * [2x1] = [3x1]
    let matrix = [[1., 2.], [3., 4.], [5., 6.]];
    let vector = [[7.], [8.]];
    let tensor_matrix = Tensor::new(&device, &matrix);
    let tensor_vector = Tensor::new(&device, &vector);
    let result = tensor_matrix.mat_mul(&tensor_vector);
    let as_slice = result.as_slice().await.unwrap();

    // Expected: [1*7 + 2*8, 3*7 + 4*8, 5*7 + 6*8] = [23, 53, 83]
    assert_eq!(as_slice[[0, 0]], 23.);
    assert_eq!(as_slice[[1, 0]], 53.);
    assert_eq!(as_slice[[2, 0]], 83.);
}

#[cfg(test)]
#[tokio::test]
async fn test_batched_matrix_vector_mul() {
    let device = Device::test_instance();

    // Test simpler batched case first: [1x2x3] * [1x3x1] = [1x2x1]
    let matrices = [[[1., 2., 3.], [4., 5., 6.]]];
    let vectors = [[[7.], [8.], [9.]]];

    let tensor_matrices = Tensor::new(&device, &matrices);
    let tensor_vectors = Tensor::new(&device, &vectors);
    let result = tensor_matrices.mat_mul(&tensor_vectors);
    let as_slice = result.as_slice().await.unwrap();

    // Expected: [1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9] = [50, 122]
    assert_eq!(as_slice[[0, 0, 0]], 50.);
    assert_eq!(as_slice[[0, 1, 0]], 122.);
}

#[cfg(test)]
#[tokio::test]
async fn test_full_batched_matrix_vector_mul() {
    let device = Device::test_instance();

    // Test batched matrix-vector multiplication: [2x2x3] * [2x3x1] = [2x2x1]
    let matrices = [
        [[1., 2., 3.], [4., 5., 6.]],
        [[7., 8., 9.], [10., 11., 12.]],
    ];
    let vectors = [[[13.], [14.], [15.]], [[16.], [17.], [18.]]];

    let tensor_matrices = Tensor::new(&device, &matrices);
    let tensor_vectors = Tensor::new(&device, &vectors);
    let result = tensor_matrices.mat_mul(&tensor_vectors);
    let as_slice = result.as_slice().await.unwrap();

    // First batch: [1*13 + 2*14 + 3*15, 4*13 + 5*14 + 6*15] = [86, 212]
    assert_eq!(as_slice[[0, 0, 0]], 86.);
    assert_eq!(as_slice[[0, 1, 0]], 212.);

    // Second batch: [7*16 + 8*17 + 9*18, 10*16 + 11*17 + 12*18] = [410, 563]
    assert_eq!(as_slice[[1, 0, 0]], 410.);
    assert_eq!(as_slice[[1, 1, 0]], 563.);
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul() {
    let device = Device::test_instance();

    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_asymetric_matmul() {
    let device = Device::test_instance();

    let data_a = [[1., 2.], [3., 4.], [5., 6.]];
    let data_b = [[1., 2.], [3., 4.]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. * 1. + 2. * 3.);
    assert_eq!(as_slice[[0, 1]], 1. * 2. + 2. * 4.);
    assert_eq!(as_slice[[1, 0]], 3. * 1. + 4. * 3.);
    assert_eq!(as_slice[[1, 1]], 3. * 2. + 4. * 4.);
    assert_eq!(as_slice[[2, 0]], 5. * 1. + 6. * 3.);
    assert_eq!(as_slice[[2, 1]], 5. * 2. + 6. * 4.);
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul_fused() {
    let device = Device::test_instance();

    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a) * 2.;
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b) / 4.;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 1. / 2.);
    assert_eq!(as_slice[[0, 1]], 2. / 2.);
    assert_eq!(as_slice[[1, 0]], 3. / 2.);
    assert_eq!(as_slice[[1, 1]], 6. / 2.);
}

#[cfg(test)]
#[tokio::test]
async fn test_transposed_matmul() {
    use candle_core::IndexOp;

    let device = Device::test_instance();
    // This test uses regular tensor ops, not quantized matmul, so CPU is fine
    let candle_device = candle_core::Device::Cpu;

    let data_a = [[1.], [3.]];
    let data_b = [[1., 2.]];
    let tensor_a = Tensor::new(&device, &data_a).t();
    let tensor_b = Tensor::new(&device, &data_b).t();
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], 7.);

    let data_a = [[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]]];
    let data_b = [[[[9., 10.], [11., 12.]], [[13., 14.], [15., 16.]]]];
    let tensor_a = Tensor::new(&device, &data_a).transpose(1, 2);
    let tensor_b = Tensor::new(&device, &data_b).transpose(1, 2).t();
    let candle_tensor_a = candle_core::Tensor::new(&data_a, &candle_device)
        .unwrap()
        .transpose(1, 2)
        .unwrap();
    let candle_tensor_b = candle_core::Tensor::new(&data_b, &candle_device)
        .unwrap()
        .transpose(1, 2)
        .unwrap()
        .t()
        .unwrap();
    let tensor = tensor_a.mat_mul(&tensor_b);
    let candle_tensor = candle_tensor_a.matmul(&candle_tensor_b).unwrap();
    let candle_as_slice = candle_tensor
        .i((0, .., .., ..))
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    let as_slice = tensor.as_slice().await.unwrap();
    println!("fusor: {as_slice:?}");
    println!("candle: {candle_as_slice:?}");

    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                assert_eq!(as_slice[[0, z, y, x]], candle_as_slice[z][y][x]);
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_attention_chain_single_resolution_matches_staged() {
    let device = Device::test_instance();
    let batch = 1;
    let seq_len = 20;
    let heads = 8;
    let head_dim = 40;
    let hidden = heads * head_dim;

    fn tensor3(
        device: &Device,
        batch: usize,
        seq_len: usize,
        hidden: usize,
        seed: usize,
    ) -> Tensor<3, f32> {
        let data = (0..batch)
            .map(|batch_idx| {
                (0..seq_len)
                    .map(|seq_idx| {
                        (0..hidden)
                            .map(|hidden_idx| {
                                let value =
                                    (batch_idx * 131 + seq_idx * 17 + hidden_idx * 7 + seed) % 97;
                                (value as f32 - 48.0) * 0.01
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Tensor::new(device, &data)
    }

    fn tensor4_mask(device: &Device, batch: usize, heads: usize, seq_len: usize) -> Tensor<4, f32> {
        let data = (0..batch)
            .map(|_| {
                (0..heads)
                    .map(|_| {
                        (0..seq_len)
                            .map(|q_idx| {
                                (0..seq_len)
                                    .map(|k_idx| if k_idx <= q_idx { 0.0 } else { -1.0e9 })
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Tensor::new(device, &data)
    }

    fn attention(
        q: &Tensor<3, f32>,
        k: &Tensor<3, f32>,
        v: &Tensor<3, f32>,
        mask: &Tensor<4, f32>,
        heads: usize,
        head_dim: usize,
    ) -> Tensor<3, f32> {
        let [batch, seq_len, _] = *q.shape();
        let q = q.reshape([batch, seq_len, heads, head_dim]).transpose(1, 2);
        let k = k.reshape([batch, seq_len, heads, head_dim]).transpose(1, 2);
        let v = v.reshape([batch, seq_len, heads, head_dim]).transpose(1, 2);
        let k_t = k.transpose(2, 3);
        let scores = q.mat_mul(&k_t) * (head_dim as f32).powf(-0.5);
        let scores = scores.add_(mask);
        let weights = scores.softmax_last_dim::<3>();
        weights
            .mat_mul(&v)
            .transpose(1, 2)
            .reshape([batch, seq_len, heads * head_dim])
    }

    let q = tensor3(&device, batch, seq_len, hidden, 3);
    let k = tensor3(&device, batch, seq_len, hidden, 11);
    let v = tensor3(&device, batch, seq_len, hidden, 19);
    let mask = tensor4_mask(&device, batch, heads, seq_len);

    let single_pass = attention(&q, &k, &v, &mask, heads, head_dim);
    let single_pass = single_pass.as_slice().await.unwrap();

    let q_heads = q.reshape([batch, seq_len, heads, head_dim]).transpose(1, 2);
    drop(q_heads.as_slice().await.unwrap());
    let k_heads = k.reshape([batch, seq_len, heads, head_dim]).transpose(1, 2);
    drop(k_heads.as_slice().await.unwrap());
    let v_heads = v.reshape([batch, seq_len, heads, head_dim]).transpose(1, 2);
    drop(v_heads.as_slice().await.unwrap());
    let k_t = k_heads.transpose(2, 3);
    drop(k_t.as_slice().await.unwrap());
    let scores = q_heads.mat_mul(&k_t) * (head_dim as f32).powf(-0.5);
    drop(scores.as_slice().await.unwrap());
    let scores = scores.add_(&mask);
    drop(scores.as_slice().await.unwrap());
    let weights = scores.softmax_last_dim::<3>();
    drop(weights.as_slice().await.unwrap());
    let staged = weights
        .mat_mul(&v_heads)
        .transpose(1, 2)
        .reshape([batch, seq_len, hidden]);
    let staged = staged.as_slice().await.unwrap();

    assert_eq!(single_pass.shape(), staged.shape());
    let mut max_diff = 0.0f32;
    for batch_idx in 0..batch {
        for seq_idx in 0..seq_len {
            for hidden_idx in 0..hidden {
                max_diff = max_diff.max(
                    (single_pass[[batch_idx, seq_idx, hidden_idx]]
                        - staged[[batch_idx, seq_idx, hidden_idx]])
                    .abs(),
                );
            }
        }
    }
    assert!(
        max_diff < 1e-4,
        "single-pass attention chain diverged from staged execution: max_diff={max_diff}"
    );
}

#[cfg(test)]
#[tokio::test]
async fn test_batched_matmul() {
    let device = Device::test_instance();

    let data_a = [[[1.], [3.]], [[2.], [6.]]];
    let data_b = [[[1., 2.]], [[2., 4.]]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);
    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0, 0]], 1.);
    assert_eq!(as_slice[[0, 0, 1]], 2.);
    assert_eq!(as_slice[[0, 1, 0]], 3.);
    assert_eq!(as_slice[[0, 1, 1]], 6.);

    assert_eq!(as_slice[[1, 0, 0]], 4.);
    assert_eq!(as_slice[[1, 0, 1]], 8.);
    assert_eq!(as_slice[[1, 1, 0]], 12.);
    assert_eq!(as_slice[[1, 1, 1]], 24.);
}

#[cfg(test)]
#[tokio::test]
async fn test_matmul_f16() {
    let device = Device::test_instance();
    if !device.f16_supported() {
        return;
    }

    let data_a = [[half::f16::from_f32(1.)], [half::f16::from_f32(3.)]];
    let data_b = [[half::f16::from_f32(1.), half::f16::from_f32(2.)]];
    let tensor_a = Tensor::new(&device, &data_a);
    let tensor_b = Tensor::new(&device, &data_b);

    let tensor = tensor_a.mat_mul(&tensor_b);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");

    assert_eq!(as_slice[[0, 0]], half::f16::from_f32(1.));
    assert_eq!(as_slice[[0, 1]], half::f16::from_f32(2.));
    assert_eq!(as_slice[[1, 0]], half::f16::from_f32(3.));
    assert_eq!(as_slice[[1, 1]], half::f16::from_f32(6.));
}

#[cfg(test)]
#[tokio::test]
async fn fuzz_matmul() {
    use rand::Rng;

    let device = Device::test_instance();

    let min_size = 1;
    let max_size = 512;
    let iterations = if cfg!(debug_assertions) { 10 } else { 100 };

    for _ in 0..iterations {
        let size1 = rand::rng().random_range(min_size..max_size);
        let size2 = rand::rng().random_range(min_size..max_size);
        let size3 = rand::rng().random_range(min_size..max_size);

        let data_a: Vec<Vec<f32>> = (0..size1)
            .map(|_| (0..size2).map(|_| rand::random()).collect())
            .collect();
        let data_b: Vec<Vec<f32>> = (0..size2)
            .map(|_| (0..size3).map(|_| rand::random()).collect())
            .collect();

        let tensor_a = Tensor::new(&device, &data_a);
        let tensor_b = Tensor::new(&device, &data_b);

        let mut ndarray_a = ndarray::Array2::zeros((size1, size2));
        for i in 0..size1 {
            for j in 0..size2 {
                ndarray_a[[i, j]] = data_a[i][j];
            }
        }
        let mut ndarray_b = ndarray::Array2::zeros((size2, size3));
        for i in 0..size2 {
            for j in 0..size3 {
                ndarray_b[[i, j]] = data_b[i][j];
            }
        }

        let dot = ndarray_a.dot(&ndarray_b);

        let tensor = tensor_a.mat_mul(&tensor_b);
        let as_slice = tensor.as_slice().await.unwrap();
        for i in 0..size1 {
            for j in 0..size3 {
                if (as_slice[[i, j]] - dot[[i, j]]).abs() > 0.001 {
                    println!(
                        "Mismatch at ({}, {}): {} != {}",
                        i,
                        j,
                        as_slice[[i, j]],
                        dot[[i, j]]
                    );
                    panic!("fuzz failed with size ({size1}x{size2})*({size2}x{size3})");
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn fuzz_batched_matmul() {
    use rand::Rng;
    let device = Device::test_instance();

    let min_batch_size = 2;
    let max_batch_size = 4;
    let min_size = 1;
    let max_size = 512;
    let iterations = if cfg!(debug_assertions) { 10 } else { 100 };

    for _ in 0..iterations {
        let batch_size_1 = rand::rng().random_range(min_batch_size..max_batch_size);
        let batch_size_2 = rand::rng().random_range(min_batch_size..max_batch_size);
        let size1 = rand::rng().random_range(min_size..max_size);
        let size2 = rand::rng().random_range(min_size..max_size);
        let size3 = rand::rng().random_range(min_size..max_size);

        let data_a: Vec<Vec<Vec<Vec<f32>>>> = (0..batch_size_1)
            .map(|_| {
                (0..batch_size_2)
                    .map(|_| {
                        (0..size1)
                            .map(|_| (0..size2).map(|_| rand::random()).collect())
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let data_b: Vec<Vec<Vec<Vec<f32>>>> = (0..batch_size_1)
            .map(|_| {
                (0..batch_size_2)
                    .map(|_| {
                        (0..size2)
                            .map(|_| (0..size3).map(|_| rand::random()).collect())
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let tensor_a = Tensor::new(&device, &data_a);
        let tensor_b = Tensor::new(&device, &data_b);

        let ndarray_a = (0..batch_size_1)
            .map(|i_1| {
                (0..batch_size_2)
                    .map(|i_2| {
                        let mut array = ndarray::Array2::zeros((size1, size2));
                        for j in 0..size1 {
                            for k in 0..size2 {
                                array[[j, k]] = data_a[i_1][i_2][j][k];
                            }
                        }
                        array
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let ndarray_b = (0..batch_size_1)
            .map(|i_1| {
                (0..batch_size_2)
                    .map(|i_2| {
                        let mut array = ndarray::Array2::zeros((size2, size3));
                        for j in 0..size2 {
                            for k in 0..size3 {
                                array[[j, k]] = data_b[i_1][i_2][j][k];
                            }
                        }
                        array
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let dot = ndarray_a
            .iter()
            .zip(ndarray_b.iter())
            .map(|(a, b)| {
                a.iter()
                    .zip(b.iter())
                    .map(|(a, b)| a.dot(b))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let tensor = tensor_a.mat_mul(&tensor_b);
        let as_slice = tensor.as_slice().await.unwrap();
        for batch_1 in 0..batch_size_1 {
            for batch_2 in 0..batch_size_2 {
                for i in 0..size1 {
                    for j in 0..size3 {
                        if (as_slice[[batch_1, batch_2, i, j]] - dot[batch_1][batch_2][[i, j]])
                            .abs()
                            > 0.001
                        {
                            println!(
                                "Mismatch at ({}, {}): {} != {}",
                                i,
                                j,
                                as_slice[[batch_1, batch_2, i, j]],
                                dot[batch_1][batch_2][[i, j]]
                            );
                            panic!("fuzz failed with size ({size1}x{size2})*({size2}x{size3})");
                        }
                    }
                }
            }
        }
    }
}
