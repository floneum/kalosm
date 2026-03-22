use crate::matmul::sgemm_params::gemm_parameters;
use crate::matmul::sgemv_params::gemv_parameters;
use crate::mir::operation::Operation;
use crate::{
    Device, Tensor,
    compute_graph::NodeIndex,
    mir::kernel::GenericKernel,
    nary_wise::UnaryFunctionChain,
    tensor::{DataType, DataTypeEnum, TensorData},
};

pub mod coop_gemm;
pub mod sgemm;
mod sgemm_params;
pub mod sgemv;
mod sgemv_params;

pub fn get_optimal_params(m: usize, n: usize, k: usize, device: &Device) -> MatMulParams {
    let coop = coop_gemm::CoopGemmParams::default();
    if device.cooperative_matrix_supported()
        && m >= coop.block_m as usize
        && n >= coop.block_n as usize
        && k >= coop.block_k as usize
        && m % coop.block_m as usize == 0
        && n % coop.block_n as usize == 0
    {
        return MatMulParams::CoopMatMul(coop);
    }
    match (m, n, k) {
        (_, 0..=64, _) => MatMulParams::Vector(gemv_parameters(m, n, k)),
        (_, _, _) => MatMulParams::MatMul(gemm_parameters(m, n, k)),
    }
}

#[derive(Debug, Clone)]
pub enum MatMulParams {
    Vector(sgemv::SgemvParams),
    MatMul(sgemm::SgemmParams),
    CoopMatMul(coop_gemm::CoopGemmParams),
}

#[derive(Debug, Clone)]
pub(crate) struct MatMulOperation {
    pub(crate) datatype: DataTypeEnum,
    pub(crate) first: NodeIndex,
    pub(crate) second: NodeIndex,
    pub(crate) first_shape: Box<[usize]>,
    pub(crate) second_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) pre_element_wise: [UnaryFunctionChain; 2],
    pub(crate) post_element_wise: UnaryFunctionChain,
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
        device: &Device,
    ) -> Self {
        let parameters = parameters.unwrap_or_else(|| {
            let n = second_shape[second_shape.len() - 1];
            let m = first_shape[first_shape.len() - 2];
            let k = first_shape[first_shape.len() - 1];
            get_optimal_params(m, n, k, device)
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
                UnaryFunctionChain::empty(datatype),
                UnaryFunctionChain::empty(datatype),
            ],
            post_element_wise: UnaryFunctionChain::empty(datatype),
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
            MatMulParams::CoopMatMul(coop_params) => {
                coop_gemm::workgroup_shape_constraints(self, device, coop_params)
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
            MatMulParams::CoopMatMul(coop_params) => coop_gemm::dispatch_size(
                last_dim_size,
                second_to_last_dim_size,
                batch_size,
                workgroup_shape,
                coop_params,
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
            MatMulParams::CoopMatMul(coop_params) => coop_gemm::build_kernel(
                self,
                graph,
                workgroup_shape,
                inputs,
                generic_kernel,
                coop_params,
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
    let tensor_matrix = Tensor::new(&device, &matrix).restride([
        crate::StrideSpec::dim(0, 2),
        crate::StrideSpec::dim(1, 3),
    ]);
    let tensor_vector = Tensor::new(&device, &vector);
    let result = tensor_matrix.mat_mul(&tensor_vector);
    let as_slice = result.as_slice().await.unwrap();

    // Expected: same as before since we removed the last column
    assert_eq!(as_slice[[0, 0]], 50.);
    assert_eq!(as_slice[[1, 0]], 122.);
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
async fn test_coop_matmul_aligned() {
    // Test with perfectly aligned sizes for cooperative matmul (128x64 block)
    let device = Device::test_instance();
    for (size1, size2, size3) in [
        (128, 16, 64),
        (128, 17, 64),
        (128, 32, 64),
        (128, 33, 64),
        (256, 100, 128),
    ] {
        // Use all-ones; expected result = size2 at every position
        let data_a: Vec<Vec<f32>> = (0..size1)
            .map(|_| (0..size2).map(|_| 1.0f32).collect())
            .collect();
        let data_b: Vec<Vec<f32>> = (0..size2)
            .map(|_| (0..size3).map(|_| 1.0f32).collect())
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

        let mut max_err = 0.0f32;
        for i in 0..size1 {
            for j in 0..size3 {
                let err = (as_slice[[i, j]] - dot[[i, j]]).abs();
                max_err = max_err.max(err);
                if err > 0.01 {
                    panic!(
                        "Mismatch at ({i}, {j}): gpu={} ref={} err={err} for ({size1}x{size2})*({size2}x{size3})",
                        as_slice[[i, j]],
                        dot[[i, j]]
                    );
                }
            }
        }
        println!("({size1}x{size2})*({size2}x{size3}): max_err={max_err}");
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
