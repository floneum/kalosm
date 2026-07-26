use crate::{
    Device,
    kernel_selection::{
        Axis, CooperativeMatrixKind, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, range,
    },
    matmul::sgemm_params::gemm_parameters,
    matmul::sgemv_params::gemv_parameters,
    tensor::DataTypeEnum,
};

use super::MatMulParams;

pub(super) const DENSE_M: Axis<0> = Axis;
pub(super) const DENSE_N: Axis<2> = Axis;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DenseMatmulVariant {
    Coop,
    Vector,
    MatMul,
}

pub(super) fn dense_coop_kinds_from_datatype(
    datatype: DataTypeEnum,
) -> &'static [CooperativeMatrixKind] {
    match datatype {
        DataTypeEnum::F32 => &[CooperativeMatrixKind::F32F32M8N8K8],
        DataTypeEnum::F16 => &[CooperativeMatrixKind::F16F16M8N8K8],
        DataTypeEnum::U32 => &[],
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct DenseMatmulCtx {
    pub(super) coop_kinds: &'static [CooperativeMatrixKind],
}

pub(super) fn dense_matmul_selector() -> ShapeSelector<3, DenseMatmulCtx, DenseMatmulVariant> {
    ShapeSelector::new()
        .rule(
            DenseMatmulVariant::Coop,
            ShapeRule::new().when(|_shape: KernelShape<3>, ctx: &DenseMatmulCtx, caps| {
                coop_supported(caps, ctx.coop_kinds)
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

pub(super) fn select_dense_matmul_params(
    m: usize,
    n: usize,
    k: usize,
    device: &Device,
    coop_kinds: &'static [CooperativeMatrixKind],
) -> MatMulParams {
    let shape = KernelShape::new([m, k, n]);
    let ctx = DenseMatmulCtx { coop_kinds };
    let caps = KernelDeviceCaps::from_device(device);
    match dense_matmul_selector()
        .select(shape, &ctx, caps)
        .expect("dense matmul selector has a catch-all rule")
    {
        DenseMatmulVariant::Coop => MatMulParams::CoopMatMul,
        DenseMatmulVariant::Vector => MatMulParams::Vector(gemv_parameters(m, n, k)),
        DenseMatmulVariant::MatMul => MatMulParams::MatMul(gemm_parameters(m, n, k)),
    }
}

/// Whether the cooperative-matrix family is available at all on this
/// device: fixed-width subgroups, a supported cooperative kind, and room
/// for at least the smallest coop workgroup. Geometry is not consulted —
/// the scored tile selection decides it per kernel build, and shapes it
/// declines lower through the generic fused reduction.
pub(super) fn coop_supported(caps: KernelDeviceCaps, coop_kinds: &[CooperativeMatrixKind]) -> bool {
    caps.subgroups_supported
        && coop_kinds
            .iter()
            .any(|kind| caps.cooperative_matrix.supports(*kind))
        && caps.min_subgroup_size == caps.max_subgroup_size
        && caps.max_compute_workgroup_size_x >= 64
}

#[cfg(test)]
pub(super) fn select_coop_kind(
    caps: KernelDeviceCaps,
    coop_kinds: &[CooperativeMatrixKind],
) -> CooperativeMatrixKind {
    coop_kinds
        .iter()
        .copied()
        .find(|kind| caps.cooperative_matrix.supports(*kind))
        .expect("coop selector called with no supported cooperative matrix kind")
}

/// (BM, BN, BK) tile dimensions for a cooperative-matrix matmul tile. Chosen
/// by [`super::cost::plan_coop_tile`], which also derives the subgroup split
/// the kernel runs with; the kernel layer looks the remaining execution
/// properties (N_PASSES, BLOCK) up by geometry in its own table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CoopTile {
    pub(crate) bm: u32,
    pub(crate) bn: u32,
    pub(crate) bk: u32,
}

impl CoopTile {
    pub(crate) const fn new(bm: u32, bn: u32, bk: u32) -> Self {
        Self { bm, bn, bk }
    }

}
