//! Debugging switches read from `FUSOR_*` environment variables, parsed once.

use std::path::PathBuf;
use std::sync::OnceLock;

pub(crate) struct Flags {
    /// `FUSOR_GAPSTEP`: one resolve-phase stopwatch line per resolve.
    pub gapstep: bool,
    /// `FUSOR_COLDLIST`: under `gapstep`, one line per launch the probe missed.
    pub coldlist: bool,
    /// `FUSOR_VERIFY_ARTIFACT_CACHE`: relower every cache hit and compare
    /// (compiler-test builds only).
    pub verify_artifact_cache: bool,
    /// `FUSOR_MISMATCH_DUMP`: where a failed cache verification dumps the body.
    pub mismatch_dump: Option<PathBuf>,
    /// `FUSOR_NO_PIPELINE_SHARE`: compile every body, sharing no pipeline.
    pub no_pipeline_share: bool,
    /// `FUSOR_WGSL_DUMP`: a directory every emitted module is written into.
    pub wgsl_dump: Option<PathBuf>,
    /// `FUSOR_COMPILE_THREADS`: cap on parallel lower-and-compile workers.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    pub compile_threads: Option<usize>,
    /// `FUSOR_TIME_PLAN`: print each plan's GPU span as `TPLAN <us>`.
    pub time_plan: bool,
    /// `FUSOR_TIME_RANGE`: time live dispatches from this index as `TSPAN`.
    pub time_range: Option<usize>,
    /// `FUSOR_TRACE_DISPATCH`: one dispatch per submission, each waited on.
    pub trace_dispatch: bool,
    /// `FUSOR_CACHE_STATS`: retained-object counts every 64 resolves.
    pub cache_stats: bool,
    /// `FUSOR_PASS_SIZE`: dispatches per compute pass.
    pub pass_size: Option<usize>,
    /// `FUSOR_NO_TILE_ALIAS`: give every workgroup tile its own allocation.
    pub no_tile_alias: bool,
    /// `FUSOR_DUMP_CONTRACT`: the family, point and shape of each contraction.
    pub dump_contract: bool,
    /// `FUSOR_POOL_DEBUG`: log pool allocation failures.
    pub pool_debug: bool,
}

pub(crate) fn flags() -> &'static Flags {
    static FLAGS: OnceLock<Flags> = OnceLock::new();
    FLAGS.get_or_init(|| {
        let on = |name| std::env::var_os(name).is_some();
        let path = |name| std::env::var_os(name).map(PathBuf::from);
        let number = |name| std::env::var(name).ok().and_then(|v| v.parse().ok());
        Flags {
            gapstep: on("FUSOR_GAPSTEP"),
            coldlist: on("FUSOR_COLDLIST"),
            verify_artifact_cache: cfg!(feature = "compiler-tests")
                && on("FUSOR_VERIFY_ARTIFACT_CACHE"),
            mismatch_dump: path("FUSOR_MISMATCH_DUMP"),
            no_pipeline_share: on("FUSOR_NO_PIPELINE_SHARE"),
            wgsl_dump: path("FUSOR_WGSL_DUMP"),
            compile_threads: number("FUSOR_COMPILE_THREADS"),
            time_plan: on("FUSOR_TIME_PLAN"),
            time_range: number("FUSOR_TIME_RANGE"),
            trace_dispatch: on("FUSOR_TRACE_DISPATCH"),
            cache_stats: on("FUSOR_CACHE_STATS"),
            pass_size: number("FUSOR_PASS_SIZE"),
            no_tile_alias: on("FUSOR_NO_TILE_ALIAS"),
            dump_contract: on("FUSOR_DUMP_CONTRACT"),
            pool_debug: on("FUSOR_POOL_DEBUG"),
        }
    })
}
