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

