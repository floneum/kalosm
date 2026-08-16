//! [`GpuTarget`] — the [`Target`] implementation tying device, lowering,
//! emission, the pool, the plan cache and the launcher together.

use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use fusor2_ir::Result;
use fusor2_ir::cost::DeviceFacts;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::Persistence;
use fusor2_ir::egraph::{EGraph, Id, Rule};
use fusor2_ir::error::Error;
use fusor2_ir::extract::{Plan, PlanHash};
use fusor2_ir::ir::Node;
use fusor2_ir::ir::launch::SchedPoint;
use fusor2_ir::ir::kernel::KernelIr;
use fusor2_ir::shape::SymId;
use fusor2_ir::target::{Artifact, Buf, EmitError, LowerCtx, Target, Uniforms};
use rustc_hash::{FxHashMap, FxHasher};

use crate::device::GpuDevice;
use crate::launch::{
    BuildCursor, CommandRecord, GpuArtifact, KernelProfile, Launcher, TimingMode,
};
use crate::plan_cache::PlanCache;
use crate::pool::BufferPool;
use crate::uniforms::UniformPack;

/// Runtime policy.
#[derive(Clone, Debug)]
pub struct GpuConfig {
    /// Override the platform memory ceiling.
    pub max_gpu_memory_bytes: Option<u64>,
    /// Pre-fill fresh allocations with `0xCD` so a zero-init assumption fails
    /// loudly instead of reading the last tenant's bytes.
    pub poison_allocations: bool,
    /// Back-pressure window: the runtime blocks when more than this many
    /// submissions are outstanding.
    pub max_in_flight_submits: usize,
    /// Allocate a timestamp query set and fold the samples into
    /// [`KernelProfile`]s.
    pub trace_gpu_kernels: bool,
    /// Root of the on-disk plan tier; `None` disables it.
    pub cache_dir: Option<PathBuf>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory_bytes: None,
            poison_allocations: false,
            max_in_flight_submits: 8,
            trace_gpu_kernels: false,
            cache_dir: default_cache_dir(),
        }
    }
}

fn default_cache_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return Some(PathBuf::from(xdg));
    }
    let home = std::env::var_os("HOME")?;
    let home = PathBuf::from(home);
    Some(if cfg!(target_vendor = "apple") {
        home.join("Library").join("Caches")
    } else {
        home.join(".cache")
    })
}

/// Live compiled pipelines retained per target. Must sit above any one plan's
/// whole launch set: a plan bigger than the cache evicts and recompiles every
/// pipeline every resolve.
pub const ARTIFACT_CAPACITY: usize = 65_536;

/// Everything the emitted kernel body depends on, *except* the dim binding.
///
/// `launch` is the dispatch itself — root, inlined members, bindings in
/// binding order, grid and block. `context` is everything *else* the lowering
/// of that dispatch reads out of the plan it sits in:
///
/// * the `BufferPlan` of every value the launch binds, and of its root —
///   `lower::bound_layout` treats those as the authoritative padded stride
///   set (a value with no `BufferPlan` is a leaf, answered from facts);
/// * `theta[root]`, the schedule point `lower_node` is called at;
/// * the [`UniformPack`] word layout, which bakes binding-0 slot indices.
///
/// The binding is not in the key: which of its values a lowering depends on
/// is only known after lowering ([`DimBinding::body_consulted`]), so the cached
/// [`ArtifactEntry`] carries that set and discriminates variants on those
/// values alone.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct ArtifactKey {
    /// Which arena the ids below index. An `Id` names a node only together
    /// with its graph, and many graphs resolve against one target.
    arena: u64,
    launch: u64,
    context: u64,
}

/// Every launch's [`ArtifactKey`] for one plan, computed in one pass.
fn plan_artifact_keys(plan: &Plan, pack: &UniformPack, arena: u64) -> Vec<ArtifactKey> {
    let mut digests: FxHashMap<fusor2_ir::egraph::Id, u64> =
        FxHashMap::with_capacity_and_hasher(plan.buffers.len(), Default::default());
    for buffer in &plan.buffers {
        let mut bh = FxHasher::default();
        buffer.hash(&mut bh);
        digests.insert(buffer.value, bh.finish());
    }
    let pack_digest = pack.digest();
    plan.launches
        .iter()
        .map(|launch| {
            let mut lh = FxHasher::default();
            launch.hash(&mut lh);
            let mut ch = FxHasher::default();
            ch.write_u64(pack_digest);
            plan.extraction
                .theta
                .get(&launch.root)
                .copied()
                .hash(&mut ch);
            // Binding order, so the digest tracks which slot reads which
            // layout, not just the multiset of layouts.
            for b in &launch.bindings {
                ch.write_u64(digests.get(&b.value).copied().unwrap_or(0));
            }
            ch.write_u64(digests.get(&launch.root).copied().unwrap_or(0));
            ArtifactKey {
                arena,
                launch: lh.finish(),
                context: ch.finish(),
            }
        })
        .collect()
}

/// The built variants of one launch.
struct ArtifactEntry {
    /// The symbols the last lowering's **body** consulted, sorted.
    consulted: Vec<fusor2_ir::shape::SymId>,
    /// How that lowering folded its dispatch grid, when the fold is
    /// replayable. Present means the grid — and only the grid — moves with
    /// the binding, so a length change is answered by replaying the fold
    /// rather than by re-lowering.
    grid_space: Option<crate::lower::GridSpec>,
    /// `hash(consulted syms + their bound values)` -> compiled kernel, the
    /// grid the lowering finished with, and the lowered body's identity hash
    /// (for cache verification).
    variants: lru::LruCache<u64, (Artifact, [u32; 3], u128)>,
}

/// Variants kept per launch: a decode loop in flight sees a handful of
/// active lengths (the racing autotuner's, plus the current one).
const VARIANTS_PER_LAUNCH: usize = 8;

/// One kernel body's `verify_kernel` verdict, or the verify in flight for it.
/// `true` once the body has passed; a failed verify leaves it `false`, so the
/// next caller retries rather than inheriting an error it cannot clone.
type VerifySlot = Arc<parking_lot::Mutex<bool>>;

/// One kernel body's compiled pipeline, or the compile in flight for it.
///
/// A failed build leaves the slot empty, so the next caller retries rather
/// than inheriting an error it cannot clone.
type PipelineSlot = Arc<parking_lot::Mutex<Option<Artifact>>>;

static LAST_EXIT: parking_lot::Mutex<Option<Instant>> = parking_lot::Mutex::new(None);
pub static COMPILE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LOWER_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static VERIFY_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static VERIFY_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
struct CompileGuard(Instant);
impl Drop for CompileGuard {
    fn drop(&mut self) {
        COMPILE_US.fetch_add(
            self.0.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
    }
}
fn scopeguard_compile(t: Instant) -> CompileGuard {
    CompileGuard(t)
}
fn gapstep() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR2_GAPSTEP").is_some())
}

/// Whether `FUSOR2_VERIFY_ARTIFACT_CACHE` is set, read once.
fn verify_artifact_cache() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR2_VERIFY_ARTIFACT_CACHE").is_some())
}

/// Whether `FUSOR2_NO_PIPELINE_SHARE` is set, read once.
fn no_pipeline_share() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FUSOR2_NO_PIPELINE_SHARE").is_some())
}

/// The directory `FUSOR2_WGSL_DUMP` names, read once.
fn wgsl_dump_dir() -> Option<&'static std::path::Path> {
    static DIR: std::sync::OnceLock<Option<std::path::PathBuf>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("FUSOR2_WGSL_DUMP").map(std::path::PathBuf::from))
        .as_deref()
}

/// `hash((sym, value) for sym in consulted)` under the current binding.
/// `None` when a consulted symbol is now unbound — the caller must re-lower.
fn variant_hash(consulted: &[fusor2_ir::shape::SymId], binds: &BindingEnv) -> Option<u64> {
    let mut h = FxHasher::default();
    for sym in consulted {
        h.write_u32(sym.0);
        h.write_u64(binds.dims.get(sym).copied()?);
    }
    Some(h.finish())
}

/// The body identity a compiled pipeline is deduplicated on: everything in
/// the `KernelIr` except the dispatch grid. Two lowerings at two sequence
/// lengths produce byte-identical WGSL whenever the length only moved the
/// grid, and the Metal compile happens once.
fn pipeline_hash(ir: &fusor2_ir::ir::kernel::KernelIr) -> u128 {
    // Structural, pointer-free and 128-bit: the derived `Hash` of a `Stmt`
    // folds in `Arc` addresses, so it neither matches across relowers nor is
    // safe against allocator reuse, and a collision here is a silent wrong
    // kernel.
    fusor2_tile::planner::kernel_identity(ir)
}

/// The wgpu backend.
pub struct GpuTarget {
    device: Arc<GpuDevice>,
    pool: BufferPool,
    cache: PlanCache,
    artifacts: parking_lot::Mutex<lru::LruCache<ArtifactKey, ArtifactEntry>>,
    /// Compiled pipelines by kernel-body identity ([`pipeline_hash`]), shared
    /// across launches and bindings.
    ///
    /// The slot is the *single-flight* claim, not just the answer: it is held
    /// across the compile, so whoever takes it compiles and everyone else
    /// finds the artifact. The cohort takes the claim with `try_lock` and
    /// moves on when it is held (see `try_pipeline_for`); only the serial
    /// tail waits.
    pipelines: parking_lot::Mutex<lru::LruCache<u128, PipelineSlot>>,
    /// Compiled pipelines by emitted-WGSL identity — the last-resort dedup,
    /// keyed on the full source text so a collision is impossible. A relower
    /// at a new sequence length can change the IR hash without changing one
    /// byte of the emitted module; this tier catches that and skips the
    /// Metal compile.
    pipelines_by_source: parking_lot::Mutex<lru::LruCache<String, Artifact>>,
    /// Body identities already through [`fusor2_tile::verify_kernel`] on this
    /// target's caps.
    ///
    /// `verify_kernel` is a pure function of `(body, caps)` and this target's
    /// caps are fixed for its life, so a body whose [`pipeline_hash`] is here
    /// has already been verified. It is never an opt-out: a body reaching
    /// Kernel for the first time is always verified.
    ///
    /// The entry is a slot claimed before the verify and held across it,
    /// exactly as [`Self::pipelines`] is; a check-then-insert would let a
    /// whole cohort past an empty entry for the same body.
    verified: parking_lot::Mutex<lru::LruCache<u128, VerifySlot>>,
    launcher: Launcher,
    config: GpuConfig,
}

impl GpuTarget {
    /// Probe an adapter at WebGPU baseline limits and build the target.
    pub async fn new() -> Result<Self> {
        Self::with_config(GpuConfig::default()).await
    }

    pub async fn with_config(config: GpuConfig) -> Result<Self> {
        let device = Arc::new(GpuDevice::request(None).await?);
        Self::from_device(device, config)
    }

    /// Build over an already-requested device.
    pub fn from_device(device: Arc<GpuDevice>, config: GpuConfig) -> Result<Self> {
        let wgpu_device = Arc::new(device.device().clone());
        let queue = Arc::new(device.queue().clone());
        let backend = device.adapter().get_info().backend;
        let pool = BufferPool::new(wgpu_device.clone(), queue.clone(), &config);
        let cache = PlanCache::with_facts(device.facts(), config.cache_dir.clone());
        let launcher = Launcher::new(wgpu_device, queue, backend, config.clone());
        Ok(Self {
            device,
            pool,
            cache,
            artifacts: parking_lot::Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(ARTIFACT_CAPACITY).expect("ARTIFACT_CAPACITY is nonzero"),
            )),
            pipelines: parking_lot::Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(ARTIFACT_CAPACITY).expect("ARTIFACT_CAPACITY is nonzero"),
            )),
            verified: parking_lot::Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(ARTIFACT_CAPACITY).expect("ARTIFACT_CAPACITY is nonzero"),
            )),
            pipelines_by_source: parking_lot::Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(ARTIFACT_CAPACITY).expect("ARTIFACT_CAPACITY is nonzero"),
            )),
            launcher,
            config,
        })
    }

    /// `pollster`-blocking convenience for non-async callers.
    pub fn new_blocking() -> Result<Self> {
        pollster::block_on(Self::new())
    }

    pub fn device(&self) -> &Arc<GpuDevice> {
        &self.device
    }
    pub fn pool(&self) -> &BufferPool {
        &self.pool
    }
    pub fn plan_cache(&self) -> &PlanCache {
        &self.cache
    }
    pub fn launcher(&self) -> &Launcher {
        &self.launcher
    }
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    pub fn take_kernel_profiles(&self) -> Vec<KernelProfile> {
        self.launcher.take_kernel_profiles()
    }

    /// The plan is the cache key.
    pub const fn plan_key(&self, plan: &Plan) -> PlanHash {
        plan.hash
    }

    /// Read a device buffer back to the host. **One of exactly three host
    /// syncs.**
    pub async fn readback(&self, buf: &Buf, bytes: u64) -> Result<Vec<u8>> {
        self.launcher.readback(&self.pool, buf, bytes)
    }

    /// The whole-plan entry point `fusor2::Session` calls.
    ///
    /// Three phases, in this order and no other:
    ///
    /// 1. **Serial, plan order** — bind buffers per `Launch::bindings` (binding
    ///    0 is the uniform block), allocate outputs from `Plan::buffers`
    ///    through the pool, resolve grids.
    /// 2. **Parallel** — plan-cache lookup by [`PlanHash`], else lower, verify
    ///    Kernel, emit and create the pipeline. A serial probe runs first so a warm
    ///    cache never touches the thread pool.
    /// 3. **Serial, exact plan order** — push command records and release
    ///    consumed buffers.
    pub fn resolve(&self, plan: &Plan, graph: &EGraph, binds: &BindingEnv) -> Result<()> {
        let start = Instant::now();
        // `FUSOR2_GAPSTEP` — the resolve-phase stopwatch, one line per resolve:
        //
        // - `outside`  ms between the previous resolve returning and this one
        //   starting.
        // - `p1`/`probe`/`build`/`bind`/`enc`/`tail`/`tot` the phases below,
        //   in order; `cold` is how many launches the warm probe missed and
        //   `build` therefore had to lower.
        // - `lowus`/`compus`/`verus`/`vern` CPU microseconds this resolve
        //   spent lowering, in the Metal compiler, and in `verify_kernel` (with
        //   the number of bodies verified) — summed across the build cohort,
        //   so they exceed `build` whenever the workers overlap.
        // - `chunkwait`/`pollus` how long the host was blocked on the GPU
        //   (chunked-submit backpressure, and `poll_wait`).
        let gap = gapstep();
        if gap {
            let prev = LAST_EXIT.lock().replace(start);
            let outside = prev.map(|p| start.duration_since(p).as_secs_f64() * 1e3);
            eprint!("GAPSTEP outside={:.2} ", outside.unwrap_or(0.0));
        }
        // One pack for the whole resolve; every lowering this resolve drives
        // needs it.
        let pack = Arc::new(UniformPack::new(plan));
        let uniforms = pack.fill(plan, &binds.dims, &binds.scalars)?;

        // Phase 1: serial, plan order.
        let uniform_buf = self
            .pool
            .alloc_with_usage(pack.byte_len(), crate::pool::TENSOR_USAGE)?;
        self.launcher.write_uniforms(&uniform_buf, &uniforms)?;

        // `plan.buffers` excludes external leaves. Every binding must still
        // resolve, so the caller-owned buffers seed the map before anything
        // is allocated on top of them.
        let mut resolved: FxHashMap<Id, Buf> = binds.buffers.clone();
        for buffer in &plan.buffers {
            if resolved.contains_key(&buffer.value) {
                continue;
            }
            // `BufferPlan::elements` is the derived placeholder whenever any
            // extent is symbolic; the layout is the authority then. Padding
            // lives in the strides, so the extent of the plan's row-major
            // layouts is `shape[0] * strides[0]` — never the shape product,
            // which undercounts a padded buffer. A `DERIVED_STRIDE` in slot 0
            // implies no padding (plan derivation refuses the combination),
            // so it resolves as the product of the remaining extents.
            let elements = match buffer.elements {
                d if d == fusor2_ir::shape::Dim::Sym(crate::uniforms::DERIVED_STRIDE) => {
                    match (buffer.layout.shape().first(), buffer.layout.strides().first()) {
                        (Some(first), Some(stride0)) => {
                            let stride0 = match stride0 {
                                fusor2_ir::shape::Dim::Sym(s)
                                    if *s == crate::uniforms::DERIVED_STRIDE =>
                                {
                                    buffer.layout.shape()[1..].iter().try_fold(1u64, |acc, d| {
                                        Some(acc.saturating_mul(binds.dim(*d)?))
                                    })
                                }
                                d => binds.dim(*d),
                            };
                            stride0.and_then(|s| Some(binds.dim(*first)?.saturating_mul(s)))
                        }
                        _ => Some(1),
                    }
                }
                d => binds.dim(d),
            }
            .ok_or_else(|| {
                Error::Plan(format!("buffer {} has an unbound extent", buffer.value))
            })?;
            let bytes = elements.saturating_mul(buffer.dtype.byte_size()).max(4);
            resolved.insert(
                buffer.value,
                self.pool.alloc(bytes, buffer.persistence)?,
            );
        }

        let mut work: Vec<LaunchWork> = Vec::with_capacity(plan.launches.len());
        for (launch_ix, launch) in plan.launches.iter().enumerate() {
            let mut ordered: Vec<_> = launch.bindings.iter().collect();
            ordered.sort_by_key(|b| b.binding);
            let mut buffers = Vec::with_capacity(ordered.len() + 1);
            buffers.push(uniform_buf.clone());
            for b in &ordered {
                let buf = resolved.get(&b.value).cloned().ok_or_else(|| {
                    Error::Plan(format!("launch binds {} which the plan never allocates", b.value))
                })?;
                buffers.push(buf);
            }
            work.push(LaunchWork {
                root: launch.root,
                launch_ix,
                grid: launch.grid,
                buffers,
                artifact: None,
            });
        }


        // Phase 2: probe the cache, then build the cold set.
        let __t_p1 = start.elapsed();
        let keys = plan_artifact_keys(plan, &pack, graph.arena_id());
        let queue_len = work.len();
        // The probe is the partition: every launch whose variant is already
        // built finishes here with a hash lookup, and what is left is the
        // cold set, which goes to the cohort whole.
        let mut cold: Vec<usize> = Vec::new();
        // One binding for the whole probe. Grid replay only *reads* it, and
        // the reads it records are the caller's, not a lowering's.
        let probe_binding =
            crate::lower::DimBinding::from_pairs(binds.dims.iter().map(|(k, v)| (*k, *v)));
        for i in 0..queue_len {
            match self.cached_artifact(plan, graph, &work[i], binds, &probe_binding, &keys, &pack)? {
                Some((artifact, grid)) => {
                    work[i].artifact = Some(artifact);
                    work[i].grid = grid;
                }
                None => cold.push(i),
            }
        }
        let __t_probe = start.elapsed();
        let __cold = cold.len();
        if !cold.is_empty() {
            let len = cold.len();
            let items: Vec<LaunchWork> = cold
                .iter()
                .map(|&i| LaunchWork {
                    root: work[i].root,
                    launch_ix: work[i].launch_ix,
                    grid: work[i].grid,
                    buffers: work[i].buffers.clone(),
                    artifact: None,
                })
                .collect();

            // Pass A: lower, and compile what nobody else is on. A worker
            // that has just lowered a body claims its pipeline slot and
            // compiles it there and then, so compiles overlap the remaining
            // lowerings. A slot another worker already holds is skipped,
            // never waited on — waiting parks a core for the whole compile.
            let cursor = BuildCursor::new();
            let lowered: Vec<Mutexed<(Lowered, Option<Artifact>)>> =
                (0..len).map(|_| Mutexed::default()).collect();
            std::thread::scope(|scope| {
                let threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
                    .min(len);
                for _ in 0..threads {
                    scope.spawn(|| {
                        while let Some(j) = cursor.take(len) {
                            let built = self
                                .lower_uncached(plan, graph, &items[j], binds, &pack)
                                .and_then(|l| {
                                    let a = self.try_pipeline_for(&l.ir, l.ph)?;
                                    Ok((l, a))
                                });
                            *lowered[j].0.lock() = Some(built);
                        }
                    });
                }
            });
            let lowered: Vec<(Lowered, Option<Artifact>)> = lowered
                .into_iter()
                .map(|slot| {
                    slot.0
                        .into_inner()
                        .ok_or_else(|| Error::Device("a build worker dropped its slot".into()))?
                })
                .collect::<Result<_>>()?;

            // Pass B: file every launch's variant. Whatever Pass A skipped is
            // finished here, by which time the worker that claimed it has
            // published the artifact.
            for (j, (l, artifact)) in lowered.into_iter().enumerate() {
                let artifact = match artifact {
                    Some(a) => a,
                    None => self.pipeline_for(&l.ir, l.ph)?,
                };
                let i = cold[j];
                if gap && std::env::var_os("FUSOR2_COLDLIST").is_some() {
                    let gs = l.binding.grid_derivation(l.ir.grid, &self.caps().limits);
                    eprintln!(
                        "COLD ix={} name={} ph={:x} replay={} consulted={:?} vals={:?}",
                        items[j].launch_ix,
                        l.ir.name,
                        l.ph as u64,
                        gs.is_some(),
                        l.binding.body_consulted(gs.is_some()),
                        l.binding
                            .body_consulted(gs.is_some())
                            .iter()
                            .map(|s| binds.dims.get(s).copied().unwrap_or(0))
                            .collect::<Vec<_>>(),
                    );
                }
                let grid = self.record_variant(keys[items[j].launch_ix], l, &artifact, binds)?;
                work[i].artifact = Some(artifact);
                work[i].grid = grid;
            }
        }

        // Phase 3: serial, exact plan order.
        let __t_p2 = start.elapsed();
        let total = work.len();
        // A plan too large for a full per-dispatch query set can still time
        // one launch: the tuner names a plan index, and two slots bracket
        // that dispatch alone.
        let focus_plan_ix = self.launcher.take_tuning_focus();
        let mut records = Vec::with_capacity(total);
        for item in &work {
            let artifact = item
                .artifact
                .as_ref()
                .ok_or_else(|| Error::Device("a launch was never built".into()))?;
            let gpu = artifact
                .downcast_ref::<GpuArtifact>()
                .ok_or_else(|| Error::Device("artifact is not a gpu pipeline".into()))?;
            let bind_group = self.launcher.bind_group(gpu, &item.buffers)?;
            records.push(CommandRecord::Dispatch {
                name: gpu.name,
                pipeline: gpu.pipeline.clone(),
                bind_group,
                grid: item.grid,
            });
        }
        let __t_bind = start.elapsed();
        // The focused indices are stated in plan order; the encoder counts
        // *live* dispatches, so restate them. An empty dispatch never
        // reaches the encoder and cannot be timed at all.
        let focus_plan_ix = focus_plan_ix.map(|mut ixs| {
            // The sparse slot map binary-searches the live list, so it must
            // be ascending and duplicate-free.
            ixs.sort_unstable();
            ixs.dedup();
            ixs
        });
        let focus_pairs: Vec<(usize, usize)> = focus_plan_ix
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .filter_map(|&ix| {
                records
                    .get(ix)
                    .filter(|r| !r.is_empty_dispatch())
                    .map(|_| {
                        let live = records[..ix]
                            .iter()
                            .filter(|r| !r.is_empty_dispatch())
                            .count();
                        (ix, live)
                    })
            })
            .collect();
        let focus_live: Vec<usize> = focus_pairs.iter().map(|&(_, l)| l).collect();
        // A named focus wins over whole-plan timing: a backend without
        // `TIMESTAMP_QUERY_INSIDE_PASSES` writes boundary samples only, so
        // each timed dispatch takes its own compute pass. A caller that
        // named the launches it will read gets those and no others.
        let (query_set, mode) = if focus_pairs.len() == 1 {
            (
                self.launcher.timestamp_query_set(1),
                TimingMode::Focus(focus_live[0]),
            )
        } else if !focus_pairs.is_empty() {
            (
                self.launcher.timestamp_query_set(focus_pairs.len()),
                TimingMode::Sparse(&focus_live),
            )
        } else if self.launcher.can_time_whole(total) {
            (self.launcher.timestamp_query_set(total), TimingMode::All)
        } else if let Some(start) = std::env::var("FUSOR2_TIME_RANGE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
        {
            // Times live dispatches `[start, start+cap)` of an over-cap plan
            // and prints each span.
            let live = records.iter().filter(|r| !r.is_empty_dispatch()).count();
            let cap = (wgpu::QUERY_SET_MAX_QUERIES as usize / 2).min(live.saturating_sub(start));
            if cap == 0 {
                (None, TimingMode::All)
            } else {
                self.launcher.set_tuning(true);
                (
                    self.launcher.timestamp_query_set(cap),
                    TimingMode::Range { start, n: cap },
                )
            }
        } else {
            (None, TimingMode::All)
        };
        self.launcher
            .encode_command_records(&records, query_set.as_ref(), mode)?;
        let __t_enc = start.elapsed();
        // Release step-local buffers back to the pool in exact plan order.
        for buffer in &plan.buffers {
            if buffer.persistence == Persistence::Step
                && let Some(buf) = resolved.remove(&buffer.value)
                && !binds.buffers.contains_key(&buffer.value)
            {
                self.pool.recycle(buf);
            }
        }
        self.pool.recycle(uniform_buf);
        self.pool.repoison_free_buffers();
        if gap {
            let end = start.elapsed();
            eprintln!(
                "p1={:.2} probe={:.2} cold={} build={:.2} lowus={} compus={} verus={} vern={} bind={:.2} enc={:.2} tail={:.2} tot={:.2} n={} compiles={} pollwait={} chunkwait={:.2} pollus={:.2}",
                __t_p1.as_secs_f64() * 1e3,
                (__t_probe - __t_p1).as_secs_f64() * 1e3,
                __cold,
                (__t_p2 - __t_probe).as_secs_f64() * 1e3,
                LOWER_US.swap(0, std::sync::atomic::Ordering::Relaxed),
                COMPILE_US.swap(0, std::sync::atomic::Ordering::Relaxed),
                VERIFY_US.swap(0, std::sync::atomic::Ordering::Relaxed),
                VERIFY_N.swap(0, std::sync::atomic::Ordering::Relaxed),
                (__t_bind - __t_p2).as_secs_f64() * 1e3,
                (__t_enc - __t_bind).as_secs_f64() * 1e3,

                (end - __t_enc).as_secs_f64() * 1e3,
                end.as_secs_f64() * 1e3,
                total,
                self.launcher.pipeline_compiles(),
                self.launcher.poll_wait_count(),
                crate::launch::CHUNK_WAIT_US.swap(0, std::sync::atomic::Ordering::Relaxed) as f64
                    / 1e3,
                crate::launch::POLL_WAIT_US.swap(0, std::sync::atomic::Ordering::Relaxed) as f64
                    / 1e3,
            );
            *LAST_EXIT.lock() = Some(Instant::now());
        }

        if let Some(set) = query_set.as_ref() {
            // The query set is resolved from a command buffer submitted after
            // a `poll_wait`: Metal's writeback of the final encoder's boundary
            // samples races a resolve encoded behind it and leaves slots zero.
            self.launcher.poll_wait()?;
            if matches!(mode, TimingMode::Focus(_) | TimingMode::Sparse(_)) {
                // Only the focused dispatches were timed; each span lands at
                // its own plan index and every other slot reads zero, which
                // the consumers already treat as "not timed".
                let samples =
                    self.launcher
                        .read_timestamps(&self.pool, set, focus_pairs.len())?;
                if samples.iter().any(|s| *s > 0.0) {
                    let mut per_launch = vec![0.0; records.len()];
                    for (k, &(plan_ix, _)) in focus_pairs.iter().enumerate() {
                        if let Some(slot) = per_launch.get_mut(plan_ix) {
                            *slot = samples.get(k).copied().unwrap_or(0.0);
                        }
                    }
                    self.launcher.set_last_profile(per_launch);
                }
                return Ok(());
            }
            if let TimingMode::Range { start, n } = mode {
                let samples = self.launcher.read_timestamps(&self.pool, set, n)?;
                let names: Vec<&'static str> = work
                    .iter()
                    .map(|w| {
                        w.artifact
                            .as_ref()
                            .and_then(|a| a.downcast_ref::<GpuArtifact>())
                            .map(|g| g.name)
                            .unwrap_or("?")
                    })
                    .collect();
                for (j, us) in samples.iter().enumerate() {
                    let ix = start + j;
                    eprintln!(
                        "TSPAN {ix} {} {us:.1}",
                        names.get(ix).copied().unwrap_or("?")
                    );
                }
                return Ok(());
            }
            let live = records.iter().filter(|r| !r.is_empty_dispatch()).count();
            let samples = self.launcher.read_timestamps(&self.pool, set, live)?;
            // Back to plan order: a zero-grid launch never reached the encoder
            // and owns no sample.
            let mut next = 0usize;
            let mut per_launch = Vec::with_capacity(records.len());
            for record in &records {
                if record.is_empty_dispatch() {
                    per_launch.push(0.0);
                } else {
                    per_launch.push(samples.get(next).copied().unwrap_or(0.0));
                    next += 1;
                }
            }
            if self.config.trace_gpu_kernels {
                let named: Vec<(String, f64)> = work
                    .iter()
                    .zip(&per_launch)
                    .filter_map(|(w, us)| {
                        w.artifact
                            .as_ref()?
                            .downcast_ref::<GpuArtifact>()
                            .map(|g| (g.name.to_string(), *us))
                    })
                    .collect();
                self.launcher.push_profile(KernelProfile::from_samples(
                    start.elapsed().as_secs_f64() * 1000.0,
                    &named,
                ));
            }
            // An all-zero read is a device that did not write the slots, not a
            // plan that took no time. Publishing it would make every candidate
            // look infinitely fast, so the tuner falls back to the wall clock.
            if per_launch.iter().any(|us| *us > 0.0) {
                self.launcher.set_last_profile(per_launch);
            }
        }
        Ok(())
    }

    /// The artifact this launch already carries under `binding`, or `None`
    /// when it has to be built.
    ///
    /// `binding` is the resolve's, built once and handed down.
    fn cached_artifact(
        &self,
        plan: &Plan,
        graph: &EGraph,
        item: &LaunchWork,
        binds: &BindingEnv,
        binding: &crate::lower::DimBinding,
        keys: &[ArtifactKey],
        pack: &Arc<UniformPack>,
    ) -> Result<Option<(Artifact, [u32; 3])>> {
        let launch = plan
            .launches
            .get(item.launch_ix)
            .filter(|l| l.root == item.root)
            .ok_or_else(|| Error::Plan(format!("no launch roots at {}", item.root)))?;
        let key = keys[item.launch_ix];
        let cached = {
            let mut lock = self.artifacts.lock();
            match lock.get_mut(&key) {
                Some(entry) => match variant_hash(&entry.consulted, binds)
                    .and_then(|vh| entry.variants.get(&vh).cloned())
                {
                    // The stored grid belongs to the binding that built the
                    // entry, so whenever a fold was recorded it is replayed
                    // here rather than reused.
                    Some((artifact, grid, ph)) => {
                        let grid = match &entry.grid_space {
                            Some(spec) => crate::lower::grid_from(
                                &spec.space,
                                spec.block,
                                binding,
                                &self.caps().limits,
                            )?,
                            None => grid,
                        };
                        Some((artifact, grid, ph))
                    }
                    None => None,
                },
                None => None,
            }
        };
        let Some((artifact, grid, cached_ph)) = cached else {
            return Ok(None);
        };
        {
            if !verify_artifact_cache() {
                return Ok(Some((artifact, grid)));
            }
            // Verification mode: relower and compare identity + grid.
            let cx = LowerCtx {
                plan,
                launch,
                graph,
                symbols: &plan.symbols,
            };
            let theta = plan
                .extraction
                .theta
                .get(&launch.root)
                .copied()
                .unwrap_or(SchedPoint::Point);
            let node = graph.node(launch.root);
            let binding = crate::lower::DimBinding::from_pairs(
                binds.dims.iter().map(|(k, v)| (*k, *v)),
            );
            let mut kernels =
                crate::lower::lower_node(self.caps(), node, theta, &cx, binding, pack.clone())?;
            let ir = kernels.remove(0);
            if pipeline_hash(&ir) != cached_ph || ir.grid != grid {
                eprintln!(
                    "[artifact-cache] MISMATCH root {} name {}: body {} grid {:?} cached ({}, {:?})",
                    launch.root,
                    ir.name,
                    pipeline_hash(&ir),
                    ir.grid,
                    cached_ph,
                    grid,
                );
                if let Some(dir) = std::env::var_os("FUSOR2_MISMATCH_DUMP") {
                    static N: std::sync::atomic::AtomicU64 =
                        std::sync::atomic::AtomicU64::new(0);
                    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n < 40 {
                        let dir = std::path::PathBuf::from(dir);
                        let _ = std::fs::write(
                            dir.join(format!("mismatch_{n}_{}.txt", launch.root)),
                            format!("{ir:#?}"),
                        );
                    }
                }
            }
            Ok(Some((artifact, grid)))
        }
    }

    /// Emit one lowered body and compile it, consulting the WGSL-identity
    /// tier first. Called with the body's [`pipeline_hash`] slot held, so at
    /// most one worker is ever inside it for a given body.
    fn compile_body(&self, ir: &KernelIr, ph: u128) -> Result<Artifact> {
        let __t = Instant::now();
        let _g = scopeguard_compile(__t);
        let share = !no_pipeline_share();
        let emitted = crate::emit::emit(ir, self.caps()).map_err(Error::from)?;
        let mut flags = naga::back::wgsl::WriterFlags::empty();
        flags.set(naga::back::wgsl::WriterFlags::EXPLICIT_TYPES, true);
        let text = naga::back::wgsl::write_string(&emitted.module, &emitted.info, flags)
            .map_err(|e| Error::Device(format!("wgsl serialization: {e}")))?;
        if let Some(dir) = wgsl_dump_dir() {
            let _ = std::fs::create_dir_all(dir);
            let _ = std::fs::write(dir.join(format!("{}_{:016x}.wgsl", ir.name, ph)), &text);
        }
        let source_key = format!("{}\n{}", ir.block, text);
        let hit = if share {
            self.pipelines_by_source.lock().get(&source_key).cloned()
        } else {
            None
        };
        match hit {
            Some(a) => Ok(a),
            None => {
                let a = self
                    .compile_emitted(ir.name, ir.block, emitted)
                    .map_err(Error::from)?;
                self.pipelines_by_source.lock().put(source_key, a.clone());
                Ok(a)
            }
        }
    }

    /// Lower, verify, emit and compile a launch whose artifact is not
    /// cached, returning it with **the grid the lowering indexed its body
    /// against**. `Launch::grid` is the cost model's workgroup count, derived
    /// from the schedule point; `KernelIr::grid` is what the kernel body
    /// actually assumes. Dispatching the former silently computes a prefix of
    /// the output.
    fn lower_uncached(
        &self,
        plan: &Plan,
        graph: &EGraph,
        item: &LaunchWork,
        binds: &BindingEnv,
        pack: &Arc<UniformPack>,
    ) -> Result<Lowered> {
        let launch = plan
            .launches
            .get(item.launch_ix)
            .filter(|l| l.root == item.root)
            .ok_or_else(|| Error::Plan(format!("no launch roots at {}", item.root)))?;
        let cx = LowerCtx {
            plan,
            launch,
            graph,
            symbols: &plan.symbols,
        };
        let theta = plan
            .extraction
            .theta
            .get(&launch.root)
            .copied()
            .unwrap_or(SchedPoint::Point);
        let node = graph.node(launch.root);
        let binding = crate::lower::DimBinding::from_pairs(
            binds.dims.iter().map(|(k, v)| (*k, *v)),
        );
        let __tl = Instant::now();
        let mut kernels =
            crate::lower::lower_node(self.caps(), node, theta, &cx, binding.clone(), pack.clone())?;
        LOWER_US.fetch_add(
            __tl.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let ir = if kernels.is_empty() {
            return Err(Error::Plan("a launch lowered to no kernel".into()));
        } else {
            kernels.remove(0)
        };
        let ph = pipeline_hash(&ir);
        // `verify_kernel` is never optional: a failure is `Error::Lower`. A
        // body already verified on these caps is not verified twice, and the
        // slot is held across the check so a cohort lowering the same body
        // waits for the one verify.
        let slot: VerifySlot = {
            let mut lock = self.verified.lock();
            Arc::clone(lock.get_or_insert(ph, VerifySlot::default))
        };
        let mut done = slot.lock();
        if !*done {
            let __tv = Instant::now();
            fusor2_tile::verify_kernel(&ir, self.caps())?;
            VERIFY_US.fetch_add(
                __tv.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            VERIFY_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            *done = true;
        }
        drop(done);
        Ok(Lowered { ir, ph, binding })
    }

    /// The compiled pipeline for one lowered body, deduplicated on the kernel
    /// body. The slot is claimed before the compile and held across it, so a
    /// second caller on the same body waits for the first.
    fn try_pipeline_for(&self, ir: &KernelIr, ph: u128) -> Result<Option<Artifact>> {
        if no_pipeline_share() {
            return self.compile_body(ir, ph).map(Some);
        }
        let slot: PipelineSlot = {
            let mut lock = self.pipelines.lock();
            Arc::clone(lock.get_or_insert(ph, PipelineSlot::default))
        };
        let Some(mut built) = slot.try_lock() else {
            return Ok(None);
        };
        match built.as_ref() {
            Some(a) => Ok(Some(a.clone())),
            None => {
                let a = self.compile_body(ir, ph)?;
                *built = Some(a.clone());
                Ok(Some(a))
            }
        }
    }

    fn pipeline_for(&self, ir: &KernelIr, ph: u128) -> Result<Artifact> {
        if no_pipeline_share() {
            return self.compile_body(ir, ph);
        }
        let slot: PipelineSlot = {
            let mut lock = self.pipelines.lock();
            Arc::clone(lock.get_or_insert(ph, PipelineSlot::default))
        };
        let mut built = slot.lock();
        match built.as_ref() {
            Some(a) => Ok(a.clone()),
            None => {
                let a = self.compile_body(ir, ph)?;
                *built = Some(a.clone());
                Ok(a)
            }
        }
    }

    /// File one lowering's artifact under its launch key, returning **the
    /// grid the lowering indexed its body against**. `Launch::grid` is the
    /// cost model's workgroup count, derived from the schedule point;
    /// `KernelIr::grid` is what the kernel body actually assumes. Dispatching
    /// the former silently computes a prefix of the output.
    fn record_variant(
        &self,
        key: ArtifactKey,
        lowered: Lowered,
        artifact: &Artifact,
        binds: &BindingEnv,
    ) -> Result<[u32; 3]> {
        let Lowered { ir, ph, binding } = lowered;
        let grid = ir.grid;
        // A fold that replays to the grid this lowering finished with can be
        // evaluated at any later length; otherwise the grid's symbols stay in
        // the variant key, so a grid nobody can recompute is never reused
        // under a binding that would move it.
        let grid_space = binding.grid_derivation(grid, &self.caps().limits);
        let consulted = binding.body_consulted(grid_space.is_some());
        let vh = variant_hash(&consulted, binds).ok_or_else(|| {
            Error::Plan("a lowering consulted a symbol the dispatch does not bind".into())
        })?;
        let mut lock = self.artifacts.lock();
        let entry = lock.get_or_insert_mut(key, || ArtifactEntry {
            consulted: Vec::new(),
            grid_space: None,
            variants: lru::LruCache::new(
                NonZeroUsize::new(VARIANTS_PER_LAUNCH).expect("nonzero"),
            ),
        });
        entry.grid_space = grid_space;
        entry.consulted = consulted;
        entry.variants.put(vh, (artifact.clone(), grid, ph));
        Ok(grid)
    }
}

/// One launch lowered and verified, before any compile.
struct Lowered {
    ir: KernelIr,
    ph: u128,
    binding: crate::lower::DimBinding,
}

/// A slot a build worker fills.
struct Mutexed<T>(parking_lot::Mutex<Option<Result<T>>>);

impl<T> Default for Mutexed<T> {
    fn default() -> Self {
        Self(parking_lot::Mutex::new(None))
    }
}


struct LaunchWork {
    root: Id,
    /// Index into `plan.launches` (and the per-plan key vector).
    launch_ix: usize,
    grid: [u32; 3],
    buffers: Vec<Buf>,
    artifact: Option<Artifact>,
}

/// Everything a resolve needs that is not in the plan: the symbol bindings,
/// the runtime scalars and any caller-owned buffers.
#[derive(Clone, Debug, Default)]
pub struct BindingEnv {
    pub dims: FxHashMap<SymId, u64>,
    pub scalars: FxHashMap<SymId, f32>,
    pub buffers: FxHashMap<Id, Buf>,
}

impl BindingEnv {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_dim(mut self, sym: SymId, value: u64) -> Self {
        self.dims.insert(sym, value);
        self
    }
    pub fn with_scalar(mut self, sym: SymId, value: f32) -> Self {
        self.scalars.insert(sym, value);
        self
    }
    pub fn with_buffer(mut self, id: Id, buf: Buf) -> Self {
        self.buffers.insert(id, buf);
        self
    }
    fn dim(&self, d: fusor2_ir::shape::Dim) -> Option<u64> {
        match d {
            fusor2_ir::shape::Dim::Const(v) => Some(v),
            fusor2_ir::shape::Dim::Sym(s) => self.dims.get(&s).copied(),
        }
    }
}

impl GpuTarget {
    /// Compile an already-emitted module into a pipeline artifact.
    fn compile_emitted(
        &self,
        name: &'static str,
        block: u32,
        emitted: crate::emit::EmittedModule,
    ) -> std::result::Result<Artifact, EmitError> {
        let module = emitted.module;
        let bindings: Vec<(u32, bool)> = crate::bindings::bindings_from_module(&module)
            .into_iter()
            .map(|b| (b.binding, b.read_only))
            .collect();
        let entries = crate::bindings::layout_entries(
            &crate::bindings::bindings_from_module(&module),
        );
        let device = self.device.device();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(name),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(name),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        // SAFETY: every load this compiler emits is masked or provably in
        // range and every loop is counted — `verify_kernel` establishes both
        // before emission.
        let module = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some(name),
                    source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        self.launcher.note_pipeline_compile();
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        static NEXT_ARTIFACT_ID: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        Ok(Artifact::new(GpuArtifact {
            id: NEXT_ARTIFACT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            name,
            pipeline: Arc::new(pipeline),
            layout: Arc::new(layout),
            bindings,
            block,
        }))
    }
}

impl Target for GpuTarget {
    fn name(&self) -> &'static str {
        "gpu"
    }

    fn caps(&self) -> &Caps {
        self.device.caps()
    }

    fn facts(&self) -> &DeviceFacts {
        self.device.facts()
    }

    fn rules(&self) -> &'static [Rule] {
        crate::rules::GPU_RULES
    }

    fn lower(&self, node: &Node, id: Id, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
        crate::lower::lower(self.caps(), node, id, theta, cx)
    }

    fn emit(&self, ir: &KernelIr) -> std::result::Result<Artifact, EmitError> {
        let emitted = crate::emit::emit(ir, self.caps())?;
        self.compile_emitted(ir.name, ir.block, emitted)
    }

    fn launch(
        &self,
        artifact: &Artifact,
        grid: [u32; 3],
        binds: &[Buf],
        uniforms: &Uniforms,
    ) -> Result<()> {
        self.launcher.encode(artifact, grid, binds, uniforms)
    }

    fn alloc(&self, bytes: u64, persistence: Persistence) -> Result<Buf> {
        self.pool.alloc(bytes, persistence)
    }

    fn wait(&self) -> Result<()> {
        self.launcher.poll_wait()
    }
}
