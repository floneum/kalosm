use crate::matmul::sgemm_params::gemm_parameters;
use crate::matmul::sgemv_params::gemv_parameters;
use crate::mir::operation::Operation;
use crate::{
    Device, Tensor,
    compute_graph::NodeIndex,
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding},
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout,
            tile_storage_write_with_direct_layout,
        },
    },
    nary_wise::UnaryFunctionChain,
    tensor::{DataType, DataTypeEnum, TensorData},
};
use phase_token_prototype as tile_ir;

pub mod coop_gemm;
mod direct;
pub mod sgemm;
mod sgemm_params;
pub mod sgemv;
mod sgemv_params;

pub fn get_optimal_params(m: usize, n: usize, k: usize, device: &Device) -> MatMulParams {
    if let Some(coop) = coop_gemm::optimal_params(m, n, k, device) {
        return MatMulParams::CoopMatMul(coop);
    }

    if m <= 32 || n <= 64 {
        return MatMulParams::Vector(gemv_parameters(m, n, k));
    }

    MatMulParams::MatMul(gemm_parameters(m, n, k))
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
            if datatype == DataTypeEnum::F32 {
                get_optimal_params(m, n, k, device)
            } else if m <= 32 || n <= 64 {
                MatMulParams::Vector(gemv_parameters(m, n, k))
            } else {
                MatMulParams::MatMul(gemm_parameters(m, n, k))
            }
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

    fn can_use_direct_tile_matmul(&self) -> bool {
        self.datatype == DataTypeEnum::F32
            && self
                .pre_element_wise
                .iter()
                .all(|chain| chain.functions.is_empty())
            && self.post_element_wise.functions.is_empty()
    }

    fn build_direct_tile_matmul(
        &self,
        device: &Device,
        input_a: &TensorData,
        input_b: &TensorData,
        output: &TensorData,
    ) -> Option<DirectKernel> {
        let a_view = flatten_matrix_layout(input_a.layout())?;
        let b_view = flatten_matrix_layout(input_b.layout())?;
        let y_view = flatten_matrix_layout(output.layout())?;
        let m = a_view.rows;
        let k = a_view.cols;
        let k_b = b_view.rows;
        let n = b_view.cols;
        if k != k_b || y_view.rows != m || y_view.cols != n {
            return None;
        }

        let ir = tile_ir::tile::build(move |phase| {
            let a = tile_storage_read_with_direct_layout(phase, a_view);
            let b = tile_storage_read_with_direct_layout(phase, b_view);
            let y = tile_storage_write_with_direct_layout(phase, y_view);
            phase.matmul::<256>(&a, &b, &y);
        });
        let dispatch_size = ir.single_tile_program_grid()?;
        let max_workgroups = device.limits().max_compute_workgroups_per_dimension;
        if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
            return None;
        }
        let cache_key = format!(
            "{}:matmul-tile-program:{:?}:{:?}:{:?}:dispatch={dispatch_size:?}",
            self.name(),
            input_a.layout(),
            input_b.layout(),
            output.layout()
        );

        let module = if let Some(module) = device.naga_module_cache().write().get(&cache_key) {
            module.clone()
        } else {
            let module = ir.lower_to_naga().ok()?.module().clone();
            device
                .naga_module_cache()
                .write()
                .get_or_insert(cache_key.clone(), || module.clone())
                .clone()
        };
        Some(DirectKernel::new_with_cache_key(
            self.name(),
            cache_key,
            module,
            vec![
                DirectKernelBinding::Storage {
                    binding: 0,
                    buffer: input_a.buffer().clone(),
                    read_only: true,
                },
                DirectKernelBinding::Storage {
                    binding: 1,
                    buffer: input_b.buffer().clone(),
                    read_only: true,
                },
                DirectKernelBinding::Storage {
                    binding: 2,
                    buffer: output.buffer().clone(),
                    read_only: false,
                },
            ],
            dispatch_size,
        ))
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

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> Option<DirectKernel> {
        let [input_a, input_b, output] = inputs else {
            return None;
        };
        let input_a = input_a.as_tensor()?;
        let input_b = input_b.as_tensor()?;
        let output = output.as_tensor()?;
        if self.can_use_direct_tile_matmul()
            && input_a.datatype() == DataTypeEnum::F32
            && input_b.datatype() == DataTypeEnum::F32
            && output.datatype() == DataTypeEnum::F32
            && input_a.layout().rank() == 2
            && input_b.layout().rank() == 2
            && output.layout().rank() == 2
            && let Some(kernel) =
                self.build_direct_tile_matmul(&graph.device(), input_a, input_b, output)
        {
            return Some(kernel);
        }

        direct::build_serial_matmul_direct_kernel(self, graph, workgroup_shape, inputs)
    }

    fn requires_single_kernel_batch(&self) -> bool {
        true
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
