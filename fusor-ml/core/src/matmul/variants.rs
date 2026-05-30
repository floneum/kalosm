use crate::{
    Device,
    kernel_selection::{
        Axis, CooperativeMatrixCaps, CooperativeMatrixKind, KernelDeviceCaps, KernelShape,
        ShapeRule, ShapeSelector, eq, range,
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
    // Apple's coopMatrix instructions execute on 32-thread SIMD groups even
    // when the device's wgpu-reported subgroup-size range straddles 32 (M-series
    // reports min=4, max=64). Match `floneum/main`: gate only on the
    // cooperative-matrix and subgroup capabilities plus workgroup size.
    if !caps.subgroups_supported
        || !coop_kinds
            .iter()
            .any(|kind| caps.cooperative_matrix.supports(*kind))
        || caps.max_subgroup_size < 32
        || caps.min_subgroup_size > 32
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum DirectTileMatmulVariant {
    Gemv,
    MatMul,
}

/// (BM, BN, BK) tile dimensions for a cooperative-matrix matmul tile. The
/// `select` helper below returns `Option<CoopTile>` (`None` = no coop variant
/// fits the shape); the kernel layer uses the tuple to look up the matching
/// ROW_GROUPS/COL_GROUPS/N_PASSES/BLOCK in its internal table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct CoopTile {
    pub(super) bm: u32,
    pub(super) bn: u32,
    pub(super) bk: u32,
}

impl CoopTile {
    pub(super) const fn new(bm: u32, bn: u32, bk: u32) -> Self {
        Self { bm, bn, bk }
    }

    /// Pick a cooperative-matrix tile for the given (m, k, n) shape, returning
    /// `None` when no coop tile fits. All entries use BK=16 to keep
    /// double-buffered workgroup tiles inside Apple's 32 KB limit; the
    /// (256, 256, 16) entry runs single-buffered in the inner perf kernel.
    /// Heuristic: bigger tiles only fire when (M/BM)*(N/BN) clears a minimum
    /// tile count so there's enough work for the GPU.
    pub(super) fn select(m: u32, k: u32, n: u32, max_workgroup_size_x: u32) -> Option<Self> {
        let tiles_for = |bm: u32, bn: u32| -> u32 { (m / bm) * (n / bn) };
        if !k.is_multiple_of(16) {
            return None;
        }
        // Tile256x256 single-buffer has lower memory traffic (sqrt-min) but
        // 2× the barriers of Tile128x512 double-buffer; only fires when N
        // is divisible by 256 but not by 512.
        if m.is_multiple_of(256)
            && n.is_multiple_of(256)
            && !n.is_multiple_of(512)
            && max_workgroup_size_x >= 256
            && tiles_for(256, 256) >= 256
        {
            return Some(Self::new(256, 256, 16));
        }
        if m.is_multiple_of(128)
            && n.is_multiple_of(512)
            && max_workgroup_size_x >= 256
            && tiles_for(128, 512) >= 256
        {
            return Some(Self::new(128, 512, 16));
        }
        if m.is_multiple_of(128)
            && n.is_multiple_of(256)
            && max_workgroup_size_x >= 256
            && tiles_for(128, 256) >= 256
        {
            return Some(Self::new(128, 256, 16));
        }
        if m.is_multiple_of(128) && n.is_multiple_of(64) && max_workgroup_size_x >= 256 {
            return Some(Self::new(128, 64, 16));
        }
        if m.is_multiple_of(64) && n.is_multiple_of(128) && max_workgroup_size_x >= 256 {
            return Some(Self::new(64, 128, 16));
        }
        if m.is_multiple_of(64) && n.is_multiple_of(64) && max_workgroup_size_x >= 128 {
            return Some(Self::new(64, 64, 16));
        }
        None
    }
}

pub(super) fn direct_tile_matmul_selector() -> ShapeSelector<3, (), DirectTileMatmulVariant> {
    ShapeSelector::new()
        .rule(
            DirectTileMatmulVariant::Gemv,
            ShapeRule::new().axis(DENSE_N, eq(1)),
        )
        .rule(DirectTileMatmulVariant::MatMul, ShapeRule::new())
}

pub(super) fn select_direct_tile_matmul_variant(m: u32, k: u32, n: u32) -> DirectTileMatmulVariant {
    direct_tile_matmul_selector()
        .select(
            KernelShape::new([m as usize, k as usize, n as usize]),
            &(),
            KernelDeviceCaps {
                subgroups_supported: false,
                cooperative_matrix: CooperativeMatrixCaps::default(),
                min_subgroup_size: 0,
                max_subgroup_size: 0,
                max_compute_invocations_per_workgroup: 0,
                max_compute_workgroup_storage_size: 0,
                max_compute_workgroup_size_x: 0,
                backend: wgpu::Backend::Noop,
            },
        )
        .expect("direct tile matmul selector has a catch-all rule")
}
