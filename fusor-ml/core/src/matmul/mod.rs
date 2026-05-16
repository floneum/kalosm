use std::hash::Hash;

use crate::matmul::sgemm_params::gemm_parameters;
use crate::matmul::sgemv_params::gemv_parameters;
use crate::mir::operation::Operation;
use crate::{
    Device, Tensor,
    compute_graph::NodeIndex,
    kernel_selection::{Axis, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, eq, range},
    mir::{
        direct_kernel::DirectKernel,
        kernel_backend,
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout_typed,
            tile_storage_write_with_direct_layout_typed,
        },
    },
    nary_direct::apply_unary_function_chain,
    nary_wise::UnaryFunctionChain,
    tensor::{DataType, DataTypeEnum, TensorData},
};
use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use rustc_hash::FxHasher;

pub mod coop_gemm;
mod direct;
pub mod sgemm;
mod sgemm_params;
pub mod sgemv;
mod sgemv_params;

pub fn get_optimal_params(m: usize, n: usize, k: usize, device: &Device) -> MatMulParams {
    select_dense_matmul_params(m, n, k, device, true)
}

const DENSE_M: Axis<0> = Axis;
const DENSE_K: Axis<1> = Axis;
const DENSE_N: Axis<2> = Axis;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DenseMatmulVariant {
    Coop,
    Vector,
    MatMul,
}

#[derive(Clone, Copy, Debug)]
struct DenseMatmulCtx {
    allow_coop: bool,
}

fn dense_matmul_selector() -> ShapeSelector<3, DenseMatmulCtx, DenseMatmulVariant> {
    ShapeSelector::new()
        .rule(
            DenseMatmulVariant::Coop,
            ShapeRule::new().when(|shape: KernelShape<3>, ctx: &DenseMatmulCtx, caps| {
                ctx.allow_coop
                    && coop_gemm_params_from_caps(
                        shape[DENSE_M],
                        shape[DENSE_N],
                        shape[DENSE_K],
                        caps,
                    )
                    .is_some()
            }),
        )
        .rule(
            DenseMatmulVariant::Vector,
            ShapeRule::new().axis(DENSE_M, range(0..=32)),
        )
        .rule(
            DenseMatmulVariant::Vector,
            ShapeRule::new().axis(DENSE_N, range(0..=64)),
        )
        .rule(DenseMatmulVariant::MatMul, ShapeRule::new())
}

fn select_dense_matmul_params(
    m: usize,
    n: usize,
    k: usize,
    device: &Device,
    allow_coop: bool,
) -> MatMulParams {
    let shape = KernelShape::new([m, k, n]);
    let ctx = DenseMatmulCtx { allow_coop };
    let caps = KernelDeviceCaps::from_device(device);
    match dense_matmul_selector()
        .select(shape, &ctx, caps)
        .expect("dense matmul selector has a catch-all rule")
    {
        DenseMatmulVariant::Coop => MatMulParams::CoopMatMul(
            coop_gemm::optimal_params(m, n, k, device)
                .expect("coop selector and coop parameter selection disagree"),
        ),
        DenseMatmulVariant::Vector => MatMulParams::Vector(gemv_parameters(m, n, k)),
        DenseMatmulVariant::MatMul => MatMulParams::MatMul(gemm_parameters(m, n, k)),
    }
}

fn coop_gemm_params_from_caps(
    m: usize,
    n: usize,
    _k: usize,
    caps: KernelDeviceCaps,
) -> Option<coop_gemm::CoopGemmParams> {
    if !caps.cooperative_matrix_supported
        || !caps.subgroups_supported
        || caps.min_subgroup_size != 32
        || caps.max_subgroup_size != 32
        || caps.max_compute_workgroup_size_x < 64
    {
        return None;
    }

    let mut params = coop_gemm::CoopGemmParams::default();
    if n <= 16 {
        params.block_n = 16;
        params.n_passes = 1;
    } else if n <= 32 {
        params.block_n = 32;
        params.n_passes = 2;
    }

    if m <= 16 {
        params.block_m = 16;
        params.wg_threads = 64;
    } else if m < params.block_m as usize {
        params.block_m = 64;
        params.wg_threads = 128;
    }

    (params.wg_threads <= caps.max_compute_workgroup_size_x).then_some(params)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectTileMatmulVariant {
    Gemv,
    MatMul,
}

struct MatmulTileDirectKernelVariant;

fn direct_tile_matmul_selector() -> ShapeSelector<3, (), DirectTileMatmulVariant> {
    ShapeSelector::new()
        .rule(
            DirectTileMatmulVariant::Gemv,
            ShapeRule::new().axis(DENSE_N, eq(1)),
        )
        .rule(DirectTileMatmulVariant::MatMul, ShapeRule::new())
}

fn select_direct_tile_matmul_variant(m: u32, k: u32, n: u32) -> DirectTileMatmulVariant {
    direct_tile_matmul_selector()
        .select(
            KernelShape::new([m as usize, k as usize, n as usize]),
            &(),
            KernelDeviceCaps {
                subgroups_supported: false,
                cooperative_matrix_supported: false,
                min_subgroup_size: 0,
                max_subgroup_size: 0,
                max_compute_invocations_per_workgroup: 0,
                max_compute_workgroup_storage_size: 0,
                max_compute_workgroup_size_x: 0,
                max_compute_workgroups_per_dimension: 0,
            },
        )
        .expect("direct tile matmul selector has a catch-all rule")
}

#[derive(Debug, Clone, Hash)]
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
            select_dense_matmul_params(m, n, k, device, datatype == DataTypeEnum::F32)
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

        let variant = select_direct_tile_matmul_variant(m, k, n);
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
        let use_coop = matches!(self.parameters, MatMulParams::CoopMatMul(_))
            && device.cooperative_matrix_supported()
            && device.subgroups_supported()
            && device.min_subgroup_size() == 32
            && device.max_subgroup_size() == 32;
        let use_shared_tile = m.is_multiple_of(32) && n.is_multiple_of(32) && k.is_multiple_of(8);
        let datatype = self.datatype;
        let ir = tile_ir::tile::build(move |phase| {
            let epilogues = tile_ir_kernels::DenseMatmulEpilogues {
                pre_a: pre_a.as_ref(),
                pre_b: pre_b.as_ref(),
                post: post.as_ref(),
            };
            match datatype {
                DataTypeEnum::F32 => {
                    let a = tile_storage_read_with_direct_layout_typed::<tile_ir::F32>(
                        phase,
                        a_view.clone(),
                    );
                    let b = tile_storage_read_with_direct_layout_typed::<tile_ir::F32>(
                        phase,
                        b_view.clone(),
                    );
                    let y = tile_storage_write_with_direct_layout_typed::<tile_ir::F32>(
                        phase,
                        y_view.clone(),
                    );
                    if use_coop
                        && tile_ir_kernels::try_batched_coop_matmul_f32::<64, 64, 32>(
                            phase, &a, &b, &y, shape, &epilogues,
                        )
                    {
                        return;
                    }
                    match variant {
                        DirectTileMatmulVariant::Gemv | DirectTileMatmulVariant::MatMul => {
                            if use_shared_tile {
                                tile_ir_kernels::batched_matmul_with_epilogues::<
                                    tile_ir::F32,
                                    32,
                                    32,
                                    8,
                                >(
                                    phase,
                                    &a,
                                    &b,
                                    &y,
                                    shape,
                                    tile_ir::TileLiteral::f32(0.0),
                                    &epilogues,
                                )
                            } else {
                                tile_ir_kernels::batched_matmul_register_with_epilogues::<
                                    tile_ir::F32,
                                    32,
                                    32,
                                    8,
                                >(
                                    phase,
                                    &a,
                                    &b,
                                    &y,
                                    shape,
                                    tile_ir::TileLiteral::f32(0.0),
                                    &epilogues,
                                )
                            }
                        }
                    }
                }
                DataTypeEnum::F16 => {
                    let a = tile_storage_read_with_direct_layout_typed::<tile_ir::F16>(
                        phase,
                        a_view.clone(),
                    );
                    let b = tile_storage_read_with_direct_layout_typed::<tile_ir::F16>(
                        phase,
                        b_view.clone(),
                    );
                    let y = tile_storage_write_with_direct_layout_typed::<tile_ir::F16>(
                        phase,
                        y_view.clone(),
                    );
                    if use_shared_tile {
                        tile_ir_kernels::batched_matmul_f16_accum_f32_with_epilogues::<32, 32, 8>(
                            phase, &a, &b, &y, shape, &epilogues,
                        )
                    } else {
                        tile_ir_kernels::batched_matmul_f16_accum_f32_register_with_epilogues::<
                            32,
                            32,
                            8,
                        >(phase, &a, &b, &y, shape, &epilogues)
                    }
                }
                _ => unreachable!("direct tile matmul only supports f32/f16"),
            }
        });
        let dispatch_size = ir.body().grid;
        let max_workgroups = device.limits().max_compute_workgroups_per_dimension;
        if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
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
                epilogue_identity.hash(state);
            },
        );
        let cache_key = self.kernel_cache_key_with_dispatch(variant, None, dispatch_size, &inputs);

        kernel_backend::dynamic_kernel_from_ir(
            device,
            self.name(),
            cache_key,
            || Some(ir),
            kernel_backend::buffers_from_tensors([&input_a, &input_b, &output]),
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

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::kernel_selection::DeterministicShapeRng;

    fn caps(coop: bool) -> KernelDeviceCaps {
        KernelDeviceCaps {
            subgroups_supported: coop,
            cooperative_matrix_supported: coop,
            min_subgroup_size: 32,
            max_subgroup_size: 32,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_storage_size: 64 * 1024,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroups_per_dimension: 65_535,
        }
    }

    #[test]
    fn dense_selector_generates_each_variant() {
        let selector = dense_matmul_selector();
        let cases = [
            (
                DenseMatmulVariant::Coop,
                DenseMatmulCtx { allow_coop: true },
                caps(true),
            ),
            (
                DenseMatmulVariant::Vector,
                DenseMatmulCtx { allow_coop: false },
                caps(false),
            ),
            (
                DenseMatmulVariant::MatMul,
                DenseMatmulCtx { allow_coop: false },
                caps(false),
            ),
        ];
        let mut rng = DeterministicShapeRng::default();

        for (variant, ctx, caps) in cases {
            let shape = selector
                .generate_for(variant, &ctx, caps, &mut rng)
                .expect("variant should generate");
            assert_eq!(selector.select(shape, &ctx, caps), Some(variant));
        }
    }

    #[test]
    fn direct_tile_selector_generates_each_variant() {
        let selector = direct_tile_matmul_selector();
        let caps = KernelDeviceCaps {
            subgroups_supported: false,
            cooperative_matrix_supported: false,
            min_subgroup_size: 0,
            max_subgroup_size: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroup_storage_size: 0,
            max_compute_workgroup_size_x: 0,
            max_compute_workgroups_per_dimension: 0,
        };
        let mut rng = DeterministicShapeRng::default();

        for variant in [
            DirectTileMatmulVariant::Gemv,
            DirectTileMatmulVariant::MatMul,
        ] {
            let shape = selector
                .generate_for(variant, &(), caps, &mut rng)
                .expect("variant should generate");
            assert_eq!(selector.select(shape, &(), caps), Some(variant));
        }
    }
}
