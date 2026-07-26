//! Process configuration, parsed from the environment exactly once.
//!
//! Every runtime knob and trace flag lives here instead of being read from
//! the process environment at its point of use. [`FusorConfig::from_env`]
//! runs once at device creation; the value is threaded through the
//! constructors that need it ([`crate::KernelCache`], [`crate::BufferPool`],
//! and the device layer above), so configuration is explicit, inspectable,
//! and settable programmatically without touching the environment.

use std::path::PathBuf;

/// All Fusor runtime knobs and trace flags.
///
/// `Default` disables every flag and leaves every knob at its built-in
/// policy; `from_env` reads the documented `FUSOR_*` variables.
#[derive(Debug, Clone, Default)]
pub struct FusorConfig {
    /// Log resolver dispatch-category counts (`FUSOR_TRACE_RESOLVE`).
    pub trace_resolve: bool,
    /// Log host-side resolver pass timings (`FUSOR_TRACE_RESOLVE_HOST`).
    pub trace_resolve_host: bool,
    /// Log decode dispatch counts (`FUSOR_TRACE_DECODE`).
    pub trace_decode: bool,
    /// Log per-kernel dispatch names (`FUSOR_TRACE_DECODE_NAMES`).
    pub trace_decode_names: bool,
    /// Log decode timing at the model layer (`FUSOR_TRACE_DECODE_TIMING`;
    /// `KALOSM_TRACE_DECODE_TIMING` is honored for compatibility).
    pub trace_decode_timing: bool,
    /// Request GPU timestamp queries and print per-kernel GPU timings
    /// (`FUSOR_TRACE_GPU_KERNELS`).
    pub trace_gpu_kernels: bool,
    /// Log sampler pipeline decisions (`FUSOR_TRACE_SAMPLER`).
    pub trace_sampler: bool,
    /// Log split-K matmul selection (`FUSOR_TRACE_SPLITK`).
    pub trace_splitk: bool,
    /// Log horizontal matmul-merge decisions (`FUSOR_TRACE_MATMUL_MERGE`).
    pub trace_matmul_merge: bool,
    /// Log row-program fusion decisions (`FUSOR_TRACE_ROW_FUSION`).
    pub trace_row_fusion: bool,
    /// Log tiled-reduce lowering decisions (`FUSOR_TRACE_REDUCE_TILED`).
    pub trace_reduce_tiled: bool,
    /// Log per-kernel build times (`FUSOR_TRACE_BUILD_TIMES`).
    pub trace_build_times: bool,
    /// Log workgroup-tile liveness, arena packing, and barrier elision
    /// (`FUSOR_TRACE_ARENA`). Pushed into tile-ir via
    /// [`fusor_tile_ir::set_liveness_trace`] when the config is applied.
    pub trace_arena: bool,
    /// Log every shader-module / pipeline compilation
    /// (`FUSOR_TRACE_PIPELINE_COMPILES`).
    pub trace_pipeline_compiles: bool,
    /// Validate sampler outputs against a CPU reference
    /// (`FUSOR_DEBUG_SAMPLER`).
    pub debug_sampler: bool,
    /// Cross-check structurally shared fusion plans against fresh planning
    /// (`FUSOR_VERIFY_PLAN_SHARING`).
    pub verify_plan_sharing: bool,
    /// Log the per-resolve ingest and window-capture ledgers of the
    /// recognition-hoisting spike (`FUSOR_SPIKE_HOISTING`; see
    /// `compute_graph/resolve/egraph/HOISTING_SPIKE.md`). Measurement only:
    /// the ledgers change no decision.
    pub spike_hoisting: bool,
    /// Skip the pre-ingest recognition sweep for resolves with at most this
    /// many execution nodes (`FUSOR_SPIKE_NO_RECOGNITION`), so the e-graph
    /// ingests the un-preshrunk graph. This is the cost side of hoisting
    /// every recognizer into a fusion generator; it is scoped by graph size
    /// because an un-preshrunk training step does not fit in unified memory.
    pub spike_no_recognition: Option<usize>,
    /// Override the structural fusion-plan window horizon
    /// (`FUSOR_SPIKE_WINDOW_DEPTH`); unset keeps the built-in stub depth.
    pub spike_window_depth: Option<u32>,
    /// Log what the extraction duplication gate refused
    /// (`FUSOR_SPIKE_DUP_LEDGER`): the producers a fusion candidate would
    /// have inlined while they still materialize for someone else. This is
    /// the rematerialization surface. Measurement only; changes no decision.
    pub spike_dup_ledger: bool,
    /// Score the extraction byte term as total traffic — input reads plus the
    /// output write — instead of the output write alone
    /// (`FUSOR_SPIKE_READ_TRAFFIC`). Without it, inlining a producer looks
    /// free because only the deleted write is counted and the added reads of
    /// that producer's own inputs are not.
    pub spike_read_traffic: bool,
    /// Rewrite every built-in row-phase combine into an equivalent general
    /// combine before emission (`FUSOR_SPIKE_GENERAL_COMBINE`). The results
    /// must be bit-identical: this is how the general fold path is validated
    /// against the closed-operator path on real reductions.
    pub spike_general_combine: bool,
    /// Compare extraction costs on one clock — dispatches, bytes and work
    /// converted to nanoseconds by measured roofline constants — instead of
    /// the lexicographic tuple (`FUSOR_SPIKE_SCALAR_COST`). Implies
    /// `spike_read_traffic`. Under the tuple, dispatch count is effectively
    /// infinite and no amount of bandwidth or arithmetic can outweigh it.
    pub spike_scalar_cost: bool,
    /// Let candidates that duplicate a live producer through to the cost
    /// model instead of rejecting them outright (`FUSOR_SPIKE_NO_DUP_GATE`).
    /// Measures whether the lexicographic cost tuple can price duplication
    /// on its own. Experimental: the cost model does not count input reads,
    /// so this is expected to over-fuse.
    pub spike_no_dup_gate: bool,
    /// Write every generated shader to this directory (`FUSOR_DUMP_SHADERS`).
    pub dump_shaders: Option<PathBuf>,
    /// Write a Graphviz digraph of the execution graph after each resolver
    /// stage to this directory (`FUSOR_DUMP_STAGES`). Requires the `graphvis`
    /// feature; one `resolve<n>-<stage>.dot` per stage per resolve.
    pub dump_stages: Option<PathBuf>,
    /// Override the lazy graph's auto-flush node threshold
    /// (`FUSOR_GRAPH_FLUSH_THRESHOLD`; 0 disables auto-flush).
    pub graph_flush_threshold: Option<usize>,
    /// Override dispatches recorded per compute pass on giant graphs
    /// (`FUSOR_RESOLVE_DISPATCHES_PER_PASS`).
    pub resolve_dispatches_per_pass: Option<usize>,
    /// Override dispatches per queue submit on giant graphs
    /// (`FUSOR_RESOLVE_DISPATCHES_PER_SUBMIT`).
    pub resolve_dispatches_per_submit: Option<usize>,
    /// Override the top-k chunking policy's minimum candidates per chunk
    /// (`FUSOR_TOP_K_MIN_CANDIDATES_PER_CHUNK`).
    pub top_k_min_candidates_per_chunk: Option<usize>,
    /// Cap on pooled GPU memory before allocation panics; used to catch
    /// runaway graphs in tests (`FUSOR_MAX_GPU_MEMORY_BYTES`).
    pub max_gpu_memory_bytes: Option<u64>,
    /// Override the on-disk kernel-plan cache directory
    /// (`FUSOR_KERNEL_CACHE_DIR`); platform cache conventions apply when
    /// unset.
    pub kernel_cache_dir: Option<PathBuf>,
    /// Preferred wgpu adapter substring match (`WGPU_ADAPTER_NAME`).
    pub adapter_name: Option<String>,
}

fn flag(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

fn parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.parse().ok()
}

impl FusorConfig {
    /// Read every documented variable from the process environment.
    pub fn from_env() -> Self {
        Self {
            trace_resolve: flag("FUSOR_TRACE_RESOLVE"),
            trace_resolve_host: flag("FUSOR_TRACE_RESOLVE_HOST"),
            trace_decode: flag("FUSOR_TRACE_DECODE"),
            trace_decode_names: flag("FUSOR_TRACE_DECODE_NAMES"),
            trace_decode_timing: flag("FUSOR_TRACE_DECODE_TIMING")
                || flag("KALOSM_TRACE_DECODE_TIMING"),
            trace_gpu_kernels: flag("FUSOR_TRACE_GPU_KERNELS"),
            trace_sampler: flag("FUSOR_TRACE_SAMPLER"),
            trace_splitk: flag("FUSOR_TRACE_SPLITK"),
            trace_matmul_merge: flag("FUSOR_TRACE_MATMUL_MERGE"),
            trace_row_fusion: flag("FUSOR_TRACE_ROW_FUSION"),
            trace_reduce_tiled: flag("FUSOR_TRACE_REDUCE_TILED"),
            trace_build_times: flag("FUSOR_TRACE_BUILD_TIMES"),
            trace_arena: flag("FUSOR_TRACE_ARENA"),
            trace_pipeline_compiles: flag("FUSOR_TRACE_PIPELINE_COMPILES"),
            debug_sampler: flag("FUSOR_DEBUG_SAMPLER"),
            verify_plan_sharing: flag("FUSOR_VERIFY_PLAN_SHARING"),
            spike_hoisting: flag("FUSOR_SPIKE_HOISTING"),
            spike_no_recognition: parse("FUSOR_SPIKE_NO_RECOGNITION"),
            spike_window_depth: parse("FUSOR_SPIKE_WINDOW_DEPTH"),
            spike_dup_ledger: flag("FUSOR_SPIKE_DUP_LEDGER"),
            spike_read_traffic: flag("FUSOR_SPIKE_READ_TRAFFIC"),
            spike_scalar_cost: flag("FUSOR_SPIKE_SCALAR_COST"),
            spike_general_combine: flag("FUSOR_SPIKE_GENERAL_COMBINE"),
            spike_no_dup_gate: flag("FUSOR_SPIKE_NO_DUP_GATE"),
            dump_shaders: std::env::var_os("FUSOR_DUMP_SHADERS").map(PathBuf::from),
            dump_stages: std::env::var_os("FUSOR_DUMP_STAGES").map(PathBuf::from),
            graph_flush_threshold: parse("FUSOR_GRAPH_FLUSH_THRESHOLD"),
            resolve_dispatches_per_pass: parse("FUSOR_RESOLVE_DISPATCHES_PER_PASS")
                .filter(|&v: &usize| v > 0),
            resolve_dispatches_per_submit: parse("FUSOR_RESOLVE_DISPATCHES_PER_SUBMIT")
                .filter(|&v: &usize| v > 0),
            top_k_min_candidates_per_chunk: parse("FUSOR_TOP_K_MIN_CANDIDATES_PER_CHUNK"),
            max_gpu_memory_bytes: parse("FUSOR_MAX_GPU_MEMORY_BYTES"),
            kernel_cache_dir: std::env::var_os("FUSOR_KERNEL_CACHE_DIR").map(PathBuf::from),
            adapter_name: std::env::var("WGPU_ADAPTER_NAME").ok(),
        }
    }
}
