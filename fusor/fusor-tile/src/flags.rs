//! Measurement switches read from `FUSOR_*` environment variables, parsed once.

use std::sync::OnceLock;

pub(crate) struct Flags {
    /// `FUSOR_PIN_COOP="bm,bn,bk[,..]"`: restrict the coop domain to one geometry.
    pub pin_coop: Option<Vec<u32>>,
    /// Leave smaller contractions on their default domain when tuning prefill.
    pub pin_coop_min_m: u64,
    /// `FUSOR_PIN_SGEMV="vector,subgroups,cols[,parts,gap]"`: restrict the
    /// sgemv domain to one cell.
    pub pin_sgemv: Option<Vec<u32>>,
    /// Restrict small dense contractions (M <= 4, K <= 512, N != 1).
    pub pin_small_dense_sgemv: Option<Vec<u32>>,
    /// Restrict Q4 decode projections with N >= 16384 and K >= 4096.
    pub pin_large_q4_sgemv: Option<Vec<u32>>,
    /// Restrict Q6 decode projections with N >= 16384 and K >= 4096.
    pub pin_large_q6_sgemv: Option<Vec<u32>>,
}

pub(crate) fn for_profile(profile: fusor_ir::execution::ExecutionProfile) -> &'static Flags {
    static FLAGS: OnceLock<Flags> = OnceLock::new();
    static INFERENCE: OnceLock<Flags> = OnceLock::new();
    let flags = if profile == fusor_ir::execution::ExecutionProfile::LowLatency {
        &INFERENCE
    } else {
        &FLAGS
    };
    flags.get_or_init(|| {
        let list = |name| {
            profile.var(name).ok().map(|v| {
                v.split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect::<Vec<u32>>()
            })
        };
        Flags {
            pin_coop: list("FUSOR_PIN_COOP"),
            pin_coop_min_m: profile
                .var("FUSOR_PIN_COOP_MIN_M")
                .map(|s| s.parse().expect("FUSOR_PIN_COOP_MIN_M must be an integer"))
                .unwrap_or(1),
            pin_sgemv: list("FUSOR_PIN_SGEMV"),
            pin_small_dense_sgemv: list("FUSOR_PIN_SMALL_DENSE_SGEMV"),
            pin_large_q4_sgemv: list("FUSOR_PIN_LARGE_Q4_SGEMV"),
            pin_large_q6_sgemv: list("FUSOR_PIN_LARGE_Q6_SGEMV"),
        }
    })
}
