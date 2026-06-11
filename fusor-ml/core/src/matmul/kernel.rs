use std::hash::Hash;

use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use rustc_hash::FxHasher;

use crate::{
    Device,
    compute_graph::NodeIndex,
    mir::{
        kernel_backend::{self, DirectKernel},
        operation::Operation,
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout_typed,
            tile_storage_write_with_direct_layout_typed,
        },
    },
    nary_wise::{NaryExpr, NaryFunction, NaryOp, NaryScalar, UnaryFunctionChain},
    reduce::{ReduceFunction, ReduceOp, ReduceOperation},
    tensor::{DataTypeEnum, TensorData},
};

use super::{
    MatMulOperation, MatMulParams, coop_gemm, sgemm, sgemv,
    variants::{CoopTile, dense_coop_kinds_from_datatype, select_dense_matmul_params},
};

fn device_supported<T>(value: Option<T>) -> Result<T, kernel_backend::DeviceNotSupported> {
    value.ok_or(kernel_backend::DeviceNotSupported)
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
            select_dense_matmul_params(m, n, k, device, dense_coop_kinds_from_datatype(datatype))
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

    pub fn rank(&self) -> u32 {
        self.out_shape.len() as u32
    }

    fn can_use_hardware_matmul(&self) -> bool {
        matches!(self.datatype, DataTypeEnum::F32 | DataTypeEnum::F16)
    }

    /// The contraction in its composed map-reduce form: a multiply over the
    /// `[batch.., m, n, k]` index space summed along `k`, with the fused
    /// pre/post chains inlined and accumulation upgraded to f32 (matching
    /// the dedicated kernels' accumulator). Routes that aren't
    /// hardware-specialized lower through this — the same generic tiled
    /// reduce any composed contraction gets.
    fn as_fused_reduce(&self) -> ReduceOperation {
        let rank = self.first_shape.len();
        let batch = rank - 2;
        let (m_dim, n_dim, k_dim) = (batch, batch + 1, batch + 2);
        let mut index_space: Vec<usize> = self.first_shape[..batch].to_vec();
        index_space.extend([
            self.first_shape[batch],
            self.second_shape[batch + 1],
            self.first_shape[batch + 1],
        ]);

        let apply_chain = |mut expr: NaryExpr, chain: &UnaryFunctionChain| {
            for function in &chain.functions {
                expr = NaryExpr::Op {
                    children: vec![expr],
                    function: function.clone(),
                };
            }
            expr
        };
        let cast_to = |expr: NaryExpr, from: DataTypeEnum, to: DataTypeEnum| {
            if from == to {
                expr
            } else {
                NaryExpr::Op {
                    children: vec![expr],
                    function: NaryFunction::unary(Some("cast".to_string()), NaryOp::Cast, from, to),
                }
            }
        };

        let a_indices: Vec<NaryExpr> = (0..batch)
            .chain([m_dim, k_dim])
            .map(NaryExpr::DimIndex)
            .collect();
        let b_indices: Vec<NaryExpr> = (0..batch)
            .chain([k_dim, n_dim])
            .map(NaryExpr::DimIndex)
            .collect();

        let acc_dtype = match self.pre_element_wise[0].out_datatype() {
            DataTypeEnum::U32 => DataTypeEnum::U32,
            DataTypeEnum::F32 | DataTypeEnum::F16 => DataTypeEnum::F32,
        };
        let a = apply_chain(
            NaryExpr::indexed_input(0, a_indices),
            &self.pre_element_wise[0],
        );
        let a = cast_to(a, self.pre_element_wise[0].out_datatype(), acc_dtype);
        let b = apply_chain(
            NaryExpr::indexed_input(1, b_indices),
            &self.pre_element_wise[1],
        );
        let b = cast_to(b, self.pre_element_wise[1].out_datatype(), acc_dtype);
        let expression = NaryExpr::mul(a, b, acc_dtype);

        let initial_value = match acc_dtype {
            DataTypeEnum::U32 => NaryScalar::U32(0),
            _ => NaryScalar::F32(0.0),
        };
        let result_dtype = self.post_element_wise.input_datatype();
        let mut post_functions = Vec::new();
        if acc_dtype != result_dtype {
            post_functions.push(NaryFunction::unary(
                Some("cast".to_string()),
                NaryOp::Cast,
                acc_dtype,
                result_dtype,
            ));
        }
        post_functions.extend(self.post_element_wise.functions.iter().cloned());

        ReduceOperation {
            inputs: vec![self.first, self.second],
            expression,
            shape: index_space.into(),
            function: ReduceFunction {
                name: Some("sum".to_string()),
                op: ReduceOp::Sum,
                initial_value,
                datatype: acc_dtype,
            },
            post_element_wise: UnaryFunctionChain::new(post_functions, acc_dtype),
            axis: k_dim,
        }
    }

    fn build_hardware_matmul(
        &self,
        device: &Device,
        input_a: &TensorData,
        input_b: &TensorData,
        output: &TensorData,
    ) -> Result<DirectKernel, kernel_backend::DeviceNotSupported> {
        let a_view = device_supported(flatten_matrix_layout(input_a.layout()))?;
        let b_view = device_supported(flatten_matrix_layout(input_b.layout()))?;
        let y_view = device_supported(flatten_matrix_layout(output.layout()))?;

        let rank = self.first_shape.len();
        let m: u32 = self.first_shape[rank - 2]
            .try_into()
            .map_err(|_| kernel_backend::DeviceNotSupported)?;
        let k: u32 = self.first_shape[rank - 1]
            .try_into()
            .map_err(|_| kernel_backend::DeviceNotSupported)?;
        let n: u32 = self.second_shape[rank - 1]
            .try_into()
            .map_err(|_| kernel_backend::DeviceNotSupported)?;
        let batch: u32 = device_supported(
            self.first_shape[..rank - 2]
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim)),
        )?
        .try_into()
        .map_err(|_| kernel_backend::DeviceNotSupported)?;
        let batch_m = device_supported(batch.checked_mul(m))?;
        let batch_k = device_supported(batch.checked_mul(k))?;
        if a_view.rows != batch_m
            || a_view.cols != k
            || b_view.rows != batch_k
            || b_view.cols != n
            || y_view.rows != batch_m
            || y_view.cols != n
        {
            return Err(kernel_backend::DeviceNotSupported);
        }
        let shape = tile_ir_kernels::DenseMatmulShape { batch, m, k, n };

        // Only the cooperative-matrix route stays hand-specialized; gemv
        // shapes lower through the generic subgroup-per-output reduce, and
        // fused chains lower through the generic tiled reduce.
        if !self.pre_element_wise[0].functions.is_empty()
            || !self.pre_element_wise[1].functions.is_empty()
            || !self.post_element_wise.functions.is_empty()
        {
            return Err(kernel_backend::DeviceNotSupported);
        }
        let subgroup_config = device_supported(device.subgroup_config())?;
        if !subgroup_config.is_fixed() {
            return Err(kernel_backend::DeviceNotSupported);
        }
        let MatMulParams::CoopMatMul(params) = &self.parameters else {
            return Err(kernel_backend::DeviceNotSupported);
        };
        let kind = params.kind();
        let coop = device_supported(device.coop_token(kind))?;
        let tile = device_supported(CoopTile::select(
            m,
            k,
            n,
            device
                .limits()
                .max_compute_workgroup_size_x
                .min(device.limits().max_compute_invocations_per_workgroup),
            subgroup_config.max_size(),
        ))?;

        let max_wg_per_dim = device.limits().max_compute_workgroups_per_dimension;
        let datatype = self.datatype;
        let used = std::cell::Cell::new(false);
        let ir = tile_ir::tile::build(|phase| {
            let element = match datatype {
                DataTypeEnum::F32 => tile_ir::ElementType::F32,
                DataTypeEnum::F16 => tile_ir::ElementType::F16,
                _ => unreachable!("hardware matmul only supports f32/f16"),
            };
            let a = tile_storage_read_with_direct_layout_typed(phase, element, a_view.clone());
            let b = tile_storage_read_with_direct_layout_typed(phase, element, b_view.clone());
            let y = tile_storage_write_with_direct_layout_typed(phase, element, y_view.clone());
            used.set(tile_ir_kernels::try_batched_coop_matmul(
                phase,
                tile_ir_kernels::DenseMatmulTensors {
                    a: &a,
                    b: &b,
                    y: &y,
                },
                shape,
                &tile_ir_kernels::DenseMatmulEpilogues {
                    pre_a: None,
                    pre_b: None,
                    post: None,
                },
                max_wg_per_dim,
                tile_ir_kernels::DenseCoopMatmulConfig {
                    coop,
                    subgroups: subgroup_config,
                    tile: tile_ir_kernels::DenseCoopMatmulTile {
                        bm: tile.bm,
                        bn: tile.bn,
                        bk: tile.bk,
                    },
                },
            ));
        });
        if !used.get() {
            return Err(kernel_backend::DeviceNotSupported);
        }
        let dispatch_size = ir.grid;
        if dispatch_size.iter().any(|dim| *dim > max_wg_per_dim) {
            return Err(kernel_backend::DeviceNotSupported);
        }
        let inputs = [
            input_a.clone().into(),
            input_b.clone().into(),
            output.clone().into(),
        ];
        let variant =
            kernel_backend::KernelVariantKey::with_payload::<HardwareMatmulVariant>(|state| {
                tile.hash(state);
                subgroup_config.hash(state);
            });
        let cache_key = self.kernel_cache_key_with_dispatch(variant, None, dispatch_size, &inputs);

        let name = self.name();
        let pipeline = kernel_backend::three_buffer_pipeline_from_ir(
            device.kernel_cache(),
            &name,
            cache_key,
            || Some(ir),
        )
        .ok_or(kernel_backend::DeviceNotSupported)?;
        Ok(
            kernel_backend::DirectKernel::from_prepared_three_buffer_pipeline(
                name,
                pipeline,
                input_a.buffer().clone(),
                input_b.buffer().clone(),
                output.buffer().clone(),
                dispatch_size,
            ),
        )
    }
}

struct HardwareMatmulVariant;

impl Operation for MatMulOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        self.datatype.hash(state);
        self.first_shape.hash(state);
        self.second_shape.hash(state);
        self.out_shape.hash(state);
        self.pre_element_wise.hash(state);
        self.post_element_wise.hash(state);
        self.parameters.hash(state);
    }

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
                input_a
                    .device()
                    .limits()
                    .max_compute_workgroups_per_dimension,
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
        if self.can_use_hardware_matmul()
            && input_a.datatype() == self.datatype
            && input_b.datatype() == self.datatype
            && output.datatype() == self.datatype
            && (self.datatype != DataTypeEnum::F16 || graph.device().f16_supported())
            && let Ok(kernel) =
                self.build_hardware_matmul(&graph.device(), input_a, input_b, output)
        {
            return Some(kernel);
        }
        // Everything else is the composed contraction's own lowering: the
        // generic tiled (or serial) fused reduce, identical to what any
        // unrecognized contraction gets.
        let reduce = self.as_fused_reduce();
        crate::reduce_tiled::build_reduce_tiled_kernel(&reduce, graph, workgroup_shape, inputs)
            .or_else(|| {
                crate::reduce_direct::build_reduce_direct_kernel(
                    &reduce,
                    graph,
                    workgroup_shape,
                    inputs,
                )
            })
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
