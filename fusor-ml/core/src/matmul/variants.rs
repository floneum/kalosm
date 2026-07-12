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
        DenseMatmulVariant::Coop => {
            MatMulParams::CoopMatMul(select_coop_kind(caps, coop_kinds))
        }
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

/// (BM, BN, BK) tile dimensions for a cooperative-matrix matmul tile. The
/// `select` helper below returns `Option<CoopTile>` (`None` = no coop variant
/// fits the shape); the kernel layer uses the tuple to look up the matching
/// ROW_GROUPS/COL_GROUPS/N_PASSES/BLOCK in its internal table.
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

    /// Subgroups per workgroup for this geometry, from the kernel table —
    /// the single source of truth for coop tile execution properties.
    /// Zero means the geometry has no kernel entry and is unselectable.
    pub(super) fn subgroup_groups(self) -> u32 {
        fusor_tile_ir_kernels::coop_tile_entries()
            .iter()
            .find(|entry| {
                entry.tile.bm == self.bm && entry.tile.bn == self.bn && entry.tile.bk == self.bk
            })
            .map(|entry| entry.row_groups * entry.col_groups)
            .unwrap_or(0)
    }

    /// The merged kernel shares one double-buffered workgroup-tile pair
    /// across guarded segments. The 256x256 standalone profile is deliberately
    /// single-buffered because a second pair would exceed the workgroup-memory
    /// budget, so it must remain a standalone dispatch.
    pub(super) const fn supports_horizontal_merge(self) -> bool {
        !matches!((self.bm, self.bn, self.bk), (256, 256, 16))
    }

    fn workgroup_size_supported(self, max_workgroup_size_x: u32, max_subgroup_size: u32) -> bool {
        self.subgroup_groups()
            .checked_mul(max_subgroup_size)
            .is_some_and(|block| block <= max_workgroup_size_x)
    }

    /// Pick a cooperative-matrix tile for the given (m, k, n) shape, or
    /// `None` when no coop tile is worth it (degenerate contractions route
    /// to the vector/generic families). Selection is the general scored
    /// argmin over the full kernel tile table — see [`super::cost`].
    pub(super) fn select(
        m: u32,
        k: u32,
        n: u32,
        policy: &crate::occupancy::DispatchPolicy,
        max_subgroup_size: u32,
    ) -> Option<Self> {
        if let Some(forced) = Self::forced_tile(policy.max_workgroup_lanes(), max_subgroup_size) {
            return Some(forced);
        }
        super::cost::select_coop_tile(m, k, n, policy, max_subgroup_size)
    }

    /// Debug override: `FUSOR_FORCE_COOP_TILE=<bm>x<bn>` forces a specific
    /// tile geometry for every coop matmul (bk is always 16). Used by the
    /// per-tile conformance sweep and A/B tile experiments; unset in normal
    /// operation.
    fn forced_tile(max_workgroup_size_x: u32, max_subgroup_size: u32) -> Option<Self> {
        let value = std::env::var("FUSOR_FORCE_COOP_TILE").ok()?;
        let (bm, bn) = value.split_once('x')?;
        let tile = Self::new(bm.parse().ok()?, bn.parse().ok()?, 16);
        tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size)
            .then_some(tile)
    }

}
