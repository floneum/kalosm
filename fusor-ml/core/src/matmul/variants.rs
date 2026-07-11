use crate::{
    Device,
    kernel_selection::{
        Axis, CooperativeMatrixKind, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, range,
    },
    matmul::sgemm_params::gemm_parameters,
    matmul::sgemv_params::gemv_parameters,
    tensor::DataTypeEnum,
};

use super::{MatMulParams, coop_gemm};

pub(super) const DENSE_M: Axis<0> = Axis;
pub(super) const DENSE_K: Axis<1> = Axis;
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
            ShapeRule::new().when(|shape: KernelShape<3>, ctx: &DenseMatmulCtx, caps| {
                coop_gemm_params_from_caps(
                    shape[DENSE_M],
                    shape[DENSE_N],
                    shape[DENSE_K],
                    caps,
                    ctx.coop_kinds,
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
        DenseMatmulVariant::Coop => MatMulParams::CoopMatMul(
            coop_gemm::optimal_params(m, n, k, device, select_coop_kind(caps, coop_kinds))
                .expect("coop selector and coop parameter selection disagree"),
        ),
        DenseMatmulVariant::Vector => MatMulParams::Vector(gemv_parameters(m, n, k)),
        DenseMatmulVariant::MatMul => MatMulParams::MatMul(gemm_parameters(m, n, k)),
    }
}

pub(super) fn coop_gemm_params_from_caps(
    m: usize,
    n: usize,
    _k: usize,
    caps: KernelDeviceCaps,
    coop_kinds: &[CooperativeMatrixKind],
) -> Option<coop_gemm::CoopGemmParams> {
    if !caps.subgroups_supported
        || !coop_kinds
            .iter()
            .any(|kind| caps.cooperative_matrix.supports(*kind))
        || caps.min_subgroup_size != caps.max_subgroup_size
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

    params.kind = select_coop_kind(caps, coop_kinds);
    (params.wg_threads <= caps.max_compute_workgroup_size_x).then_some(params)
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

    const fn subgroup_groups(self) -> u32 {
        match (self.bm, self.bn, self.bk) {
            (256, 256, 16) => 8,
            (128, 512, 16) => 8,
            (128, 256, 16) => 8,
            (128, 128, 16) => 16,
            (128, 64, 16) => 8,
            (64, 128, 16) => 8,
            (64, 64, 16) => 4,
            (64, 16, 16) => 4,
            (16, 64, 16) => 4,
            _ => 0,
        }
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

    /// Pick a cooperative-matrix tile for the given (m, k, n) shape, returning
    /// `None` when no coop tile fits. All entries use BK=16 to keep
    /// double-buffered workgroup tiles inside Apple's 32 KB limit; the
    /// (256, 256, 16) entry runs single-buffered in the inner perf kernel.
    /// Heuristic: bigger tiles only fire when (M/BM)*(N/BN) clears a minimum
    /// tile count so there's enough work for the GPU.
    pub(super) fn select(
        m: u32,
        k: u32,
        n: u32,
        max_workgroup_size_x: u32,
        max_subgroup_size: u32,
    ) -> Option<Self> {
        Self::select_primary(m, k, n, max_workgroup_size_x, max_subgroup_size)
            .or_else(|| Self::select_small_side(m, k, n, max_workgroup_size_x, max_subgroup_size))
    }

    /// The primary tile ladder: every selection that predates the small-side
    /// tiles goes through here unchanged, so shapes that already reached the
    /// coop kernel keep the exact same tile.
    fn select_primary(
        m: u32,
        k: u32,
        n: u32,
        max_workgroup_size_x: u32,
        max_subgroup_size: u32,
    ) -> Option<Self> {
        let tiles_for = |bm: u32, bn: u32| -> u32 { (m / bm) * (n / bn) };
        if m == 0 || n == 0 || k == 0 {
            return None;
        }
        // Tile256x256 single-buffer has lower memory traffic (sqrt-min) but
        // 2× the barriers of Tile128x512 double-buffer; only fires when N
        // is divisible by 256 but not by 512.
        if m.is_multiple_of(256)
            && n.is_multiple_of(256)
            && !n.is_multiple_of(512)
            && tiles_for(256, 256) >= 256
        {
            let tile = Self::new(256, 256, 16);
            if tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size) {
                return Some(tile);
            }
        }
        if m.is_multiple_of(128) && n.is_multiple_of(512) && tiles_for(128, 512) >= 256 {
            let tile = Self::new(128, 512, 16);
            if tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size) {
                return Some(tile);
            }
        }
        if m.is_multiple_of(128) && n.is_multiple_of(256) && tiles_for(128, 256) >= 256 {
            let tile = Self::new(128, 256, 16);
            if tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size) {
                return Some(tile);
            }
        }
        if m.is_multiple_of(128) && n.is_multiple_of(64) {
            let tile = Self::new(128, 64, 16);
            if tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size) {
                return Some(tile);
            }
        }
        if m.is_multiple_of(64) && n.is_multiple_of(128) {
            let tile = Self::new(64, 128, 16);
            if tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size) {
                return Some(tile);
            }
        }
        if m.is_multiple_of(64) && n.is_multiple_of(64) {
            let tile = Self::new(64, 64, 16);
            if tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size) {
                return Some(tile);
            }
        }

        // Shapes that divide no tile run with masked edge tiles: pick the
        // candidate minimizing padded work, in preference order on ties.
        // Selections whose padding inflates the output by more than a
        // quarter stay on the generic path — that bound also keeps
        // gemv-shaped contractions (tiny M or N) off the tile kernels.
        // Candidates stick to geometries the aligned rules already reach
        // (the (128, 128) table entry was never selectable and miscomputes).
        let mut best: Option<(u64, Self)> = None;
        for (bm, bn) in [(128, 64), (64, 128), (64, 64)] {
            let tile = Self::new(bm, bn, 16);
            if !tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size) {
                continue;
            }
            let padded = u64::from(m.div_ceil(bm) * bm) * u64::from(n.div_ceil(bn) * bn);
            if padded * 4 > u64::from(m) * u64::from(n) * 5 {
                continue;
            }
            if best.is_none_or(|(best_padded, _)| padded < best_padded) {
                best = Some((padded, tile));
            }
        }
        best.map(|(_, tile)| tile)
    }

    /// Second-chance selection for contractions with a 16-wide (or 16-padded)
    /// M or N side — batched attention contractions like `P@V` (n = head_dim)
    /// and `Qᵀ@dS` (m = head_dim), and narrow-vocab lm_head shapes. Runs only
    /// after [`Self::select_primary`] declines, so no shape that reaches the
    /// coop kernel today changes tile. The masked-edge candidates keep the
    /// primary ladder's bound: padding may inflate the output by at most a
    /// quarter, which still keeps gemv-shaped contractions (M and N both
    /// tiny) on the generic path.
    fn select_small_side(
        m: u32,
        k: u32,
        n: u32,
        max_workgroup_size_x: u32,
        max_subgroup_size: u32,
    ) -> Option<Self> {
        const SMALL_SIDE_TILES: [(u32, u32); 2] = [(64, 16), (16, 64)];
        if m == 0 || n == 0 || k == 0 {
            return None;
        }
        for (bm, bn) in SMALL_SIDE_TILES {
            if m.is_multiple_of(bm) && n.is_multiple_of(bn) {
                let tile = Self::new(bm, bn, 16);
                if tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size) {
                    return Some(tile);
                }
            }
        }
        let mut best: Option<(u64, Self)> = None;
        for (bm, bn) in SMALL_SIDE_TILES {
            let tile = Self::new(bm, bn, 16);
            if !tile.workgroup_size_supported(max_workgroup_size_x, max_subgroup_size) {
                continue;
            }
            let padded = u64::from(m.div_ceil(bm) * bm) * u64::from(n.div_ceil(bn) * bn);
            if padded * 4 > u64::from(m) * u64::from(n) * 5 {
                continue;
            }
            if best.is_none_or(|(best_padded, _)| padded < best_padded) {
                best = Some((padded, tile));
            }
        }
        best.map(|(_, tile)| tile)
    }
}
