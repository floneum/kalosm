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
            DirectMatrixLayout, flatten_matrix_layout, tile_storage_read_with_direct_layout_typed,
            tile_storage_write_with_direct_layout_typed,
        },
    },
    nary_direct::apply_unary_function_chain,
    nary_wise::UnaryFunctionChain,
    tensor::{DataTypeEnum, TensorData},
};

use super::{
    MatMulOperation, MatMulParams, coop_gemm, direct, sgemm, sgemv,
    variants::{
        CoopTile, DirectTileMatmulVariant, dense_coop_kinds_from_datatype,
        select_dense_matmul_params, select_direct_tile_matmul_variant,
    },
};

struct MatmulTileDirectKernelVariant;

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

    fn can_use_direct_tile_matmul(&self) -> bool {
        matches!(self.datatype, DataTypeEnum::F32 | DataTypeEnum::F16)
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

        let rank = self.first_shape.len();
        let m: u32 = self.first_shape[rank - 2].try_into().ok()?;
        let k: u32 = self.first_shape[rank - 1].try_into().ok()?;
        let n: u32 = self.second_shape[rank - 1].try_into().ok()?;
        let batch: u32 = self.first_shape[..rank - 2]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?
            .try_into()
            .ok()?;
        if a_view.rows != batch.checked_mul(m)?
            || a_view.cols != k
            || b_view.rows != batch.checked_mul(k)?
            || b_view.cols != n
            || y_view.rows != batch.checked_mul(m)?
            || y_view.cols != n
        {
            return None;
        }
        let shape = tile_ir_kernels::DenseMatmulShape { batch, m, k, n };

        // The Gemv and shared-tile MatMul variants reduce through subgroup
        // operations. Use the register-tiled kernel unless the device exposes
        // a subgroup path we trust.
        let variant = if device.subgroups_supported() {
            select_direct_tile_matmul_variant(m, k, n)
        } else {
            DirectTileMatmulVariant::MatMul
        };
        let pre_a = self.pre_element_wise[0]
            .functions
            .is_empty()
            .then_some(())
            .is_none()
            .then(|| {
                let chain = self.pre_element_wise[0].clone();
                let datatype = chain.input_datatype();
                tile_ir_kernels::UnaryEpilogue::new("matmul_pre_a_chain", move |tile| {
                    apply_unary_function_chain(tile, datatype, &chain)
                        .expect("pre-chain validated at fuse time")
                        .0
                })
            });
        let pre_b = self.pre_element_wise[1]
            .functions
            .is_empty()
            .then_some(())
            .is_none()
            .then(|| {
                let chain = self.pre_element_wise[1].clone();
                let datatype = chain.input_datatype();
                tile_ir_kernels::UnaryEpilogue::new("matmul_pre_b_chain", move |tile| {
                    apply_unary_function_chain(tile, datatype, &chain)
                        .expect("pre-chain validated at fuse time")
                        .0
                })
            });
        let post = self
            .post_element_wise
            .functions
            .is_empty()
            .then_some(())
            .is_none()
            .then(|| {
                let chain = self.post_element_wise.clone();
                let datatype = chain.input_datatype();
                tile_ir_kernels::UnaryEpilogue::new("matmul_post_chain", move |tile| {
                    apply_unary_function_chain(tile, datatype, &chain)
                        .expect("post-chain validated at fuse time")
                        .0
                })
            });
        let epilogue_identity = pre_a.as_ref().map(|e| e.identity()).unwrap_or(0)
            ^ pre_b.as_ref().map(|e| e.identity()).unwrap_or(0)
            ^ post.as_ref().map(|e| e.identity()).unwrap_or(0);
        let coop_kind = match &self.parameters {
            MatMulParams::CoopMatMul(params) => Some(params.kind()),
            _ => None,
        };
        let coop_property_supported =
            coop_kind.is_some_and(|kind| device.cooperative_matrix_caps().supports(kind));
        let use_coop = coop_kind.is_some()
            && coop_property_supported
            && device.subgroups_supported()
            && device.max_subgroup_size() >= 32
            && device.min_subgroup_size() <= 32;
        let coop_variant = if use_coop {
            CoopTile::select(m, k, n, device.limits().max_compute_workgroup_size_x)
        } else {
            None
        };
        // The shared-tile kernel uses `div_ceil` for tile counts and
        // bounds-checks both A/B loads and Y stores, so it is correct for any
        // M/N/K. The register-tile path is only worth the fixed-overhead win
        // for shapes too small to amortize the workgroup-memory tile.
        let use_shared_tile = m >= 32 && n >= 32 && k >= 8;
        let max_wg_per_dim = device.limits().max_compute_workgroups_per_dimension;
        let datatype = self.datatype;
        let ir = tile_ir::tile::build(move |phase| {
            let epilogues = tile_ir_kernels::DenseMatmulEpilogues {
                pre_a: pre_a.as_ref(),
                pre_b: pre_b.as_ref(),
                post: post.as_ref(),
            };
            match datatype {
                DataTypeEnum::F32 => dispatch_direct_tile_matmul::<tile_ir::F32>(
                    phase,
                    a_view.clone(),
                    b_view.clone(),
                    y_view.clone(),
                    coop_variant,
                    variant,
                    use_shared_tile,
                    shape,
                    &epilogues,
                    max_wg_per_dim,
                ),
                DataTypeEnum::F16 => dispatch_direct_tile_matmul::<tile_ir::F16>(
                    phase,
                    a_view.clone(),
                    b_view.clone(),
                    y_view.clone(),
                    coop_variant,
                    variant,
                    use_shared_tile,
                    shape,
                    &epilogues,
                    max_wg_per_dim,
                ),
                _ => unreachable!("direct tile matmul only supports f32/f16"),
            }
        });
        let dispatch_size = ir.body().grid;
        if dispatch_size.iter().any(|dim| *dim > max_wg_per_dim) {
            return None;
        }
        let inputs = [
            input_a.clone().into(),
            input_b.clone().into(),
            output.clone().into(),
        ];
        let variant = kernel_backend::KernelVariantKey::with_payload::<MatmulTileDirectKernelVariant>(
            |state| {
                use_shared_tile.hash(state);
                variant.hash(state);
                coop_variant.hash(state);
                coop_kind.hash(state);
                epilogue_identity.hash(state);
            },
        );
        let cache_key = self.kernel_cache_key_with_dispatch(variant, None, dispatch_size, &inputs);

        kernel_backend::dynamic_kernel_from_ir(
            device.kernel_cache(),
            self.name(),
            cache_key,
            || Some(ir),
            [
                input_a.buffer().clone(),
                input_b.buffer().clone(),
                output.buffer().clone(),
            ],
            dispatch_size,
        )
    }
}

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
        if self.can_use_direct_tile_matmul()
            && input_a.datatype() == self.datatype
            && input_b.datatype() == self.datatype
            && output.datatype() == self.datatype
            && (self.datatype != DataTypeEnum::F16 || graph.device().f16_supported())
            && let Some(kernel) =
                self.build_direct_tile_matmul(&graph.device(), input_a, input_b, output)
        {
            return Some(kernel);
        }

        direct::build_serial_matmul_direct_kernel(self, graph, workgroup_shape, inputs)
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

#[allow(clippy::too_many_arguments)]
fn dispatch_direct_tile_matmul<
    T: tile_ir::CoopElement + tile_ir_kernels::AccumCast<tile_ir::F32>,
>(
    phase: &mut tile_ir::tile::Program,
    a_view: DirectMatrixLayout,
    b_view: DirectMatrixLayout,
    y_view: DirectMatrixLayout,
    coop_variant: Option<CoopTile>,
    variant: DirectTileMatmulVariant,
    use_shared_tile: bool,
    shape: tile_ir_kernels::DenseMatmulShape,
    epilogues: &tile_ir_kernels::DenseMatmulEpilogues<'_>,
    max_wg_per_dim: u32,
) {
    let a = tile_storage_read_with_direct_layout_typed::<T>(phase, a_view);
    let b = tile_storage_read_with_direct_layout_typed::<T>(phase, b_view);
    let y = tile_storage_write_with_direct_layout_typed::<T>(phase, y_view);
    if let Some(tile) = coop_variant
        && tile_ir_kernels::try_batched_coop_matmul::<T>(
            phase,
            tile_ir_kernels::DenseMatmulTensors {
                a: &a,
                b: &b,
                y: &y,
            },
            shape,
            epilogues,
            max_wg_per_dim,
            tile_ir_kernels::DenseCoopMatmulTile {
                bm: tile.bm,
                bn: tile.bn,
                bk: tile.bk,
            },
        )
    {
        return;
    }
    match variant {
        DirectTileMatmulVariant::Gemv => tile_ir_kernels::batched_gemv_with_epilogues::<T>(
            phase,
            &a,
            &b,
            &y,
            shape,
            epilogues,
            max_wg_per_dim,
        ),
        DirectTileMatmulVariant::MatMul => {
            if use_shared_tile {
                tile_ir_kernels::batched_matmul_with_epilogues::<T>(
                    phase,
                    &a,
                    &b,
                    &y,
                    shape,
                    epilogues,
                    max_wg_per_dim,
                )
            } else {
                tile_ir_kernels::batched_matmul_register_with_epilogues::<T>(
                    phase,
                    &a,
                    &b,
                    &y,
                    shape,
                    epilogues,
                    max_wg_per_dim,
                )
            }
        }
    }
}
