use crate::{Device, MatMulOperation};

/// Parameters for cooperative matrix matmul.
#[derive(Debug, Clone, PartialEq, Hash)]
pub struct CoopGemmParams {
    pub block_m: u32,
    pub block_n: u32,
    pub block_k: u32,
    pub n_passes: u32,
    pub mma_size: u32,
    pub wg_threads: u32,
}

impl Default for CoopGemmParams {
    fn default() -> Self {
        Self {
            block_m: 128,
            block_n: 64,
            block_k: 16,
            n_passes: 4,
            mma_size: 8,
            wg_threads: 256,
        }
    }
}

pub(super) fn optimal_params(
    m: usize,
    n: usize,
    k: usize,
    device: &Device,
) -> Option<CoopGemmParams> {
    // Apple's coopMatrix instructions run on 32-thread SIMD groups even when
    // the wgpu-reported subgroup-size range straddles 32. Match
    // `floneum/main`'s gate: only require coop-matrix + subgroups, not exact
    // equality of min/max subgroup size.
    if !device.cooperative_matrix_supported()
        || !device.subgroups_supported()
        || device.max_subgroup_size() < 32
        || device.min_subgroup_size() > 32
        || device.limits().max_compute_workgroup_size_x < 64
    {
        return None;
    }

    let mut params = CoopGemmParams::default();
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

    if params.wg_threads > device.limits().max_compute_workgroup_size_x {
        return None;
    }

    let _ = k;
    Some(params)
}

pub(super) fn workgroup_shape_constraints(
    _: &MatMulOperation,
    _: &Device,
    params: &CoopGemmParams,
) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
    let mut constraints = crate::mir::workgroup_shape::WorkgroupShapeConstraints::default();
    constraints.add_constraint(
        0,
        crate::mir::workgroup_shape::Constraint::Equals(params.wg_threads),
    );
    constraints.add_constraint(1, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints.add_constraint(2, crate::mir::workgroup_shape::Constraint::Equals(1));
    constraints
}

pub(super) fn dispatch_size(
    last_dim_size: usize,
    second_to_last_dim_size: usize,
    batch_size: usize,
    workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
    params: &CoopGemmParams,
) -> [u32; 3] {
    [
        (second_to_last_dim_size as u32).div_ceil(params.block_m),
        (last_dim_size as u32).div_ceil(params.block_n),
        (batch_size as u32).div_ceil(workgroup_shape.z()),
    ]
}
