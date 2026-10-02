//! [`GpuTarget`] — the [`Target`] implementation tying device, lowering,
//! emission, compiled pipelines, the pool and the launcher together.

use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;
use web_time::Instant;

use fusor_ir::Result;
use fusor_ir::cost::DeviceFacts;
use fusor_ir::device::Caps;
use fusor_ir::dtype::Persistence;
use fusor_ir::egraph::{EGraph, Id, Rule};
use fusor_ir::error::Error;
use fusor_ir::extract::Plan;
use fusor_ir::ir::Node;
use fusor_ir::ir::kernel::KernelIr;
use fusor_ir::ir::launch::SchedPoint;
use fusor_ir::shape::SymId;
use fusor_ir::target::{Artifact, Buf, EmitError, LowerCtx, Target, Uniforms};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::device::GpuDevice;
use crate::launch::{
    BuildCursor, CommandRecord, GpuArtifact, KernelProfile, Launcher, Stopwatch, lru_retain,
};
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
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            max_gpu_memory_bytes: None,
            poison_allocations: false,
            max_in_flight_submits: 8,
            trace_gpu_kernels: false,
        }
    }
}

/// Live compiled pipelines retained per target; above any one plan's launch
/// set, or a plan recompiles every pipeline every resolve.
pub const ARTIFACT_CAPACITY: usize = 65_536;

/// Everything the emitted kernel body depends on except the dim binding:
/// `launch` is the dispatch; `context` is the plan state its lowering reads
/// (bound values' `BufferPlan`s, `theta[root]`, the [`UniformPack`] layout).
/// Which binding values matter is known only after lowering, so the entry
/// discriminates variants on [`DimBinding::body_consulted`].
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct ArtifactKey {
    /// Which graph arena the ids below index.
    arena: u64,
    launch: u64,
    context: u64,
}

/// Every launch's [`ArtifactKey`] for one plan, computed in one pass.
fn plan_artifact_keys(plan: &Plan, pack: &UniformPack, arena: u64) -> Vec<ArtifactKey> {
    let mut digests: FxHashMap<fusor_ir::egraph::Id, u64> =
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
    consulted: Vec<fusor_ir::shape::SymId>,
    /// `hash(consulted syms + values)` -> kernel, grid and body identity hash.
    variants: lru::LruCache<u64, ArtifactVariant>,
}

#[derive(Clone)]
struct ArtifactVariant {
    artifact: Artifact,
    grid: [u32; 3],
    grid_space: Option<crate::lower::GridSpec>,
    ph: u128,
}

/// Variants kept per launch: a decode loop sees a handful of active lengths.
const VARIANTS_PER_LAUNCH: usize = 8;

/// One kernel body's compiled pipeline, or the compile in flight for it. A
/// failed build leaves the slot empty so the next caller retries.
type PipelineSlot = Arc<parking_lot::Mutex<Option<Artifact>>>;

static LAST_EXIT: parking_lot::Mutex<Option<Instant>> = parking_lot::Mutex::new(None);
pub static COMPILE_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static LOWER_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `hash((sym, value) for sym in consulted)` under the current binding.
/// `None` when a consulted symbol is now unbound — the caller must re-lower.
fn variant_hash(consulted: &[fusor_ir::shape::SymId], binds: &BindingEnv) -> Option<u64> {
    let mut h = FxHasher::default();
    for sym in consulted {
        h.write_u32(sym.0);
        h.write_u64(binds.dims.get(sym).copied()?);
    }
    Some(h.finish())
}

/// The body identity pipelines dedup on: the `KernelIr` minus its grid, so a
/// length that only moved the grid compiles once.
fn pipeline_hash(ir: &fusor_ir::ir::kernel::KernelIr) -> u128 {
    // Structural, pointer-free and 128-bit: derived `Hash` folds in `Arc`
    // addresses, and a collision here is a silently wrong kernel.
    fusor_tile::planner::kernel_identity(ir)
}

fn wgsl_text(emitted: &crate::emit::EmittedModule) -> Result<String> {
    naga::back::wgsl::write_string(
        &emitted.module,
        &emitted.info,
        naga::back::wgsl::WriterFlags::EXPLICIT_TYPES,
    )
    .map_err(|e| Error::Device(format!("wgsl serialization: {e}")))
}

/// The wgpu backend.
pub struct GpuTarget {
    device: Arc<GpuDevice>,
    pool: BufferPool,
    artifacts: parking_lot::Mutex<lru::LruCache<ArtifactKey, ArtifactEntry>>,
    /// Compiled pipelines by kernel-body identity ([`pipeline_hash`]). The
    /// slot is a single-flight claim held across the compile; the cohort
    /// `try_lock`s it and moves on, only the serial tail waits.
    pipelines: parking_lot::Mutex<lru::LruCache<u128, PipelineSlot>>,
    /// Compiled pipelines by full WGSL text: catches a relower whose IR hash
    /// changed but whose module did not, skipping the Metal compile.
    pipelines_by_source: parking_lot::Mutex<lru::LruCache<String, Artifact>>,
    launcher: Launcher,
    config: GpuConfig,
}

// Explicit `Send + Sync`: the auto-trait solver otherwise recurses through
// wgpu_core's resource graph and overflows (E0275) in downstream `Send`
// futures holding a `fusor::Session`.
//
// SAFETY: `gpu_target_fields_are_send_sync` asserts `Send + Sync` for every
// field type, exactly what the auto impls would require.
unsafe impl Send for GpuTarget {}
unsafe impl Sync for GpuTarget {}

#[allow(dead_code)]
fn gpu_target_fields_are_send_sync() {
    fn assert<T: Send + Sync>() {}
    assert::<Arc<GpuDevice>>();
    assert::<BufferPool>();
    assert::<parking_lot::Mutex<lru::LruCache<ArtifactKey, ArtifactEntry>>>();
    assert::<parking_lot::Mutex<lru::LruCache<u128, PipelineSlot>>>();
    assert::<parking_lot::Mutex<lru::LruCache<String, Artifact>>>();
    assert::<Launcher>();
    assert::<GpuConfig>();
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
        let lost = device.lost().clone();
        let pool = BufferPool::new(wgpu_device.clone(), queue.clone(), &config, lost.clone());
        let launcher = Launcher::new(wgpu_device, queue, backend, config.clone(), lost);
        fn lru<K: Hash + Eq, V>() -> parking_lot::Mutex<lru::LruCache<K, V>> {
            let cap = NonZeroUsize::new(ARTIFACT_CAPACITY).expect("nonzero");
            parking_lot::Mutex::new(lru::LruCache::new(cap))
        }
        Ok(Self {
            device,
            pool,
            artifacts: lru(),
            pipelines: lru(),
            pipelines_by_source: lru(),
            launcher,
            config,
        })
    }

    /// `pollster`-blocking convenience for non-async callers. Native only.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_blocking() -> Result<Self> {
        pollster::block_on(Self::new())
    }

    pub fn device(&self) -> &Arc<GpuDevice> {
        &self.device
    }
    pub fn pool(&self) -> &BufferPool {
        &self.pool
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

    /// Prepare a selected candidate's uploads and changed launches off-thread.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn prepare_resources(
        self: &Arc<Self>,
        plan: &Plan,
        graph: &EGraph,
        dims: &[(SymId, u64)],
        launches: &[usize],
        uploads: Vec<Arc<Vec<u8>>>,
    ) -> Result<Option<std::thread::JoinHandle<Result<Vec<Buf>>>>> {
        if crate::flags().no_pipeline_share || launches.is_empty() {
            return Ok(None);
        }
        let pack = Arc::new(UniformPack::new(plan));
        let binds = BindingEnv {
            dims: dims.iter().copied().collect(),
            ..BindingEnv::default()
        };
        let mut kernels = launches
            .iter()
            .map(|&launch_ix| {
                let lowered = self.lower_uncached(plan, graph, launch_ix, &binds, &pack)?;
                Ok((lowered.ir, lowered.ph))
            })
            .collect::<Result<Vec<_>>>()?;
        kernels.retain(|(_, ph)| {
            self.pipelines
                .lock()
                .peek(ph)
                .is_none_or(|slot| slot.try_lock().is_none_or(|built| built.is_none()))
        });
        if kernels.is_empty() && uploads.is_empty() {
            return Ok(None);
        }
        let target = Arc::clone(self);
        Ok(Some(std::thread::spawn(move || {
            let buffers = uploads
                .iter()
                .map(|bytes| {
                    target
                        .pool
                        .create_buffer_init(bytes, crate::pool::TENSOR_USAGE)
                })
                .collect::<Result<Vec<_>>>()?;
            if !buffers.is_empty() {
                // Flush write_buffer uploads as well as the pool's submitted copies.
                target.device.queue().submit([]);
            }
            for (ir, ph) in kernels {
                target.pipeline(&ir, ph, true)?;
            }
            Ok(buffers)
        })))
    }

    /// [`Launcher::wait_async`]: every submission so far has retired.
    pub async fn wait_async(&self) -> Result<()> {
        self.launcher.wait_async().await
    }

    /// Read a device buffer back to the host. One of three host syncs.
    pub async fn readback(&self, buf: &Buf, bytes: u64) -> Result<Vec<u8>> {
        self.launcher.readback(&self.pool, buf, bytes).await
    }

    /// The whole-plan entry point `fusor::Session` calls, in three phases:
    /// serial allocation in plan order; parallel lower/emit/compile of the
    /// launches a warm probe missed; serial encode in exact plan order.
    pub fn resolve(&self, plan: &Plan, graph: &EGraph, binds: &BindingEnv) -> Result<()> {
        self.cache_stats();
        let start = Instant::now();
        // `FUSOR_GAPSTEP`: one line per resolve. `outside` = ms since the last
        // resolve; `p1`/`probe`/`build`/`bind`/`enc`/`tail`/`tot` = phases;
        // `cold` = probe misses; `lowus`/`compus` = cohort-summed lower/compile
        // us; `chunkwait`/`pollus` = host blocked on the GPU.
        let flags = crate::flags();
        let gap = flags.gapstep;
        if gap {
            let prev = LAST_EXIT.lock().replace(start);
            let outside = prev.map(|p| start.duration_since(p).as_secs_f64() * 1e3);
            eprint!("GAPSTEP outside={:.2} ", outside.unwrap_or(0.0));
        }
        // One pack for the whole resolve.
        let pack = Arc::new(UniformPack::new(plan));
        let uniforms = pack.fill(&binds.dims, &binds.scalars)?;

        // Phase 1: serial, plan order.
        let uniform_buf = self
            .pool
            .alloc_with_usage(pack.byte_len(), crate::pool::TENSOR_USAGE)?;
        self.launcher.write_uniforms(&uniform_buf, &uniforms)?;

        // `plan.buffers` excludes external leaves: caller buffers seed the map.
        let mut resolved: FxHashMap<Id, Buf> = binds.buffers.clone();
        let mut pending: FxHashMap<Id, (u64, Persistence)> = FxHashMap::default();
        // The step arena holds every packed intermediate for the resolve.
        let arena_buf = if plan.arena_bytes > 0 {
            Some(self.pool.alloc(plan.arena_bytes, Persistence::Step)?)
        } else {
            None
        };
        for buffer in &plan.buffers {
            if resolved.contains_key(&buffer.value) || buffer.arena.is_some() {
                continue;
            }
            let elements = binds.dim(buffer.elements).ok_or_else(|| {
                Error::Plan(format!("buffer {} has an unbound extent", buffer.value))
            })?;
            let bytes = elements.saturating_mul(buffer.dtype.byte_size()).max(4);
            // Allocated lazily in phase 3 over its live interval, so
            // intermediates share pool buffers instead of all being resident.
            pending.insert(buffer.value, (bytes, buffer.persistence));
        }
        // The last launch binding each value; its buffer is recycled after it.
        let mut last_use: FxHashMap<Id, usize> = FxHashMap::default();
        for (launch_ix, launch) in plan.launches.iter().enumerate() {
            for b in &launch.bindings {
                last_use.insert(b.value, launch_ix);
            }
        }

        // Phase 2: probe the cache, then build the cold set.
        let __t_p1 = start.elapsed();
        let keys = plan_artifact_keys(plan, &pack, graph.arena_id());
        // Warm launches finish here with a hash lookup; the rest is the cold
        // set for the build cohort.
        let mut cold: Vec<usize> = Vec::new();
        // One binding for the whole probe; grid replay only reads it.
        let probe_binding =
            crate::lower::DimBinding::from_pairs(binds.dims.iter().map(|(k, v)| (*k, *v)));
        let mut built: Vec<Option<(Artifact, [u32; 3])>> = Vec::with_capacity(plan.launches.len());
        for launch_ix in 0..plan.launches.len() {
            let hit =
                self.cached_artifact(plan, graph, launch_ix, binds, &probe_binding, &keys, &pack)?;
            if hit.is_none() {
                cold.push(launch_ix);
            }
            built.push(hit);
        }
        let __t_probe = start.elapsed();
        let __cold = cold.len();
        if !cold.is_empty() {
            let len = cold.len();
            // Pass A: lower, then compile whatever slot nobody else holds, so
            // compiles overlap lowerings; a held slot is skipped, never waited on.
            let cursor = BuildCursor::new();
            let lowered: Vec<Mutexed<(Lowered, Option<Artifact>)>> =
                (0..len).map(|_| Mutexed::default()).collect();
            let worker = || {
                while let Some(j) = cursor.take(len) {
                    let built = self
                        .lower_uncached(plan, graph, cold[j], binds, &pack)
                        .and_then(|l| {
                            let a = self.pipeline(&l.ir, l.ph, false)?;
                            Ok((l, a))
                        });
                    *lowered[j].0.lock() = Some(built);
                }
            };
            // wasm32 has no threads: one worker drains the cursor.
            #[cfg(target_arch = "wasm32")]
            worker();
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::scope(|scope| {
                // `FUSOR_COMPILE_THREADS` caps the parallel compiles.
                let threads = flags
                    .compile_threads
                    .or_else(|| std::thread::available_parallelism().map(|n| n.get()).ok())
                    .unwrap_or(1)
                    .min(len);
                for _ in 0..threads {
                    scope.spawn(worker);
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

            // Pass B: finish what Pass A skipped and file every variant.
            for (&launch_ix, (l, artifact)) in cold.iter().zip(lowered) {
                let artifact = match artifact {
                    Some(a) => a,
                    None => self
                        .pipeline(&l.ir, l.ph, true)?
                        .ok_or_else(|| Error::Device("a pipeline slot came back unbuilt".into()))?,
                };
                if gap && flags.coldlist {
                    let gs = l.binding.grid_derivation(l.ir.grid, &self.caps().limits);
                    let consulted = l.binding.body_consulted(gs.is_some());
                    eprintln!(
                        "COLD ix={launch_ix} name={} ph={:x} replay={} consulted={consulted:?} vals={:?}",
                        l.ir.name,
                        l.ph as u64,
                        gs.is_some(),
                        consulted
                            .iter()
                            .map(|s| binds.dims.get(s).copied().unwrap_or(0))
                            .collect::<Vec<_>>(),
                    );
                }
                let grid = self.record_variant(keys[launch_ix], l, &artifact, binds)?;
                built[launch_ix] = Some((artifact, grid));
            }
        }

        // Phase 3: serial, exact plan order.
        let __t_p2 = start.elapsed();
        let total = built.len();
        let focus = self.launcher.take_tuning_focus();
        let mut records = Vec::with_capacity(total);
        for (launch_ix, slot) in built.iter().enumerate() {
            let (artifact, grid) = slot
                .as_ref()
                .ok_or_else(|| Error::Device("a launch was never built".into()))?;
            let gpu = artifact
                .downcast_ref::<GpuArtifact>()
                .ok_or_else(|| Error::Device("artifact is not a gpu pipeline".into()))?;
            // This launch's buffers, plan buffers allocated at first use.
            let launch = &plan.launches[launch_ix];
            let mut ordered: Vec<_> = launch.bindings.iter().collect();
            ordered.sort_by_key(|b| b.binding);
            let mut buffers = Vec::with_capacity(ordered.len() + 1);
            buffers.push(uniform_buf.clone());
            let mut last_binding: Option<u32> = None;
            for b in &ordered {
                // Arena values share a binding: bound once.
                if last_binding == Some(b.binding) {
                    continue;
                }
                last_binding = Some(b.binding);
                if b.arena {
                    let arena = arena_buf.clone().ok_or_else(|| {
                        Error::Plan("launch binds the arena, which the plan never sized".into())
                    })?;
                    buffers.push(arena);
                    continue;
                }
                let buf = match resolved.get(&b.value) {
                    Some(buf) => buf.clone(),
                    None => {
                        let (bytes, persistence) = pending.remove(&b.value).ok_or_else(|| {
                            Error::Plan(format!(
                                "launch binds {} which the plan never allocates",
                                b.value
                            ))
                        })?;
                        let buf = self.pool.alloc(bytes, persistence).map_err(|e| {
                            let held: u64 = resolved
                                .values()
                                .filter_map(|b| b.downcast_ref::<crate::pool::GpuBuffer>())
                                .map(|g| g.size)
                                .sum();
                            Error::Device(format!(
                                "{e} (at launch {} of {}: {} plan buffers live holding {} MB, \
                                 {} not yet allocated)",
                                launch_ix,
                                total,
                                resolved.len(),
                                held >> 20,
                                pending.len()
                            ))
                        })?;
                        resolved.insert(b.value, buf.clone());
                        buf
                    }
                };
                buffers.push(buf);
            }
            crate::launch::trace_binds(gpu.name, *grid, &buffers);
            let bind_group = self.launcher.bind_group(gpu, &buffers)?;
            // Drop pool handles now: the bind group holds the device buffers,
            // and a handle kept longer would pin every intermediate.
            drop(buffers);
            records.push(CommandRecord::Dispatch {
                name: gpu.name,
                pipeline: gpu.pipeline.clone(),
                bind_group,
                grid: *grid,
            });
            // Recycle step-local buffers at their last use; encoding order
            // makes reuse by a later launch safe.
            for b in &ordered {
                if !b.arena
                    && last_use.get(&b.value) == Some(&launch_ix)
                    && !binds.buffers.contains_key(&b.value)
                    && plan
                        .buffers
                        .iter()
                        .any(|p| p.value == b.value && p.persistence == Persistence::Step)
                    && let Some(buf) = resolved.remove(&b.value)
                {
                    self.pool.recycle(buf);
                }
            }
        }
        let __t_bind = start.elapsed();
        let timing = self.launcher.timing_plan(&records, focus);
        self.launcher
            .encode_command_records(&records, timing.set(), timing.mode())?;
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
        if let Some(arena) = arena_buf {
            self.pool.recycle(arena);
        }
        self.pool.repoison_free_buffers();
        if self.pool.take_retired_buffers() {
            self.launcher.prune_dead_bind_groups();
        }
        if gap {
            let end = start.elapsed();
            eprintln!(
                "p1={:.2} probe={:.2} cold={} build={:.2} lowus={} compus={} bind={:.2} enc={:.2} tail={:.2} tot={:.2} n={} compiles={} pollwait={} chunkwait={:.2} pollus={:.2}",
                __t_p1.as_secs_f64() * 1e3,
                (__t_probe - __t_p1).as_secs_f64() * 1e3,
                __cold,
                (__t_p2 - __t_probe).as_secs_f64() * 1e3,
                LOWER_US.swap(0, std::sync::atomic::Ordering::Relaxed),
                COMPILE_US.swap(0, std::sync::atomic::Ordering::Relaxed),
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

        self.launcher
            .publish_timing(&self.pool, &timing, &records, start)
    }

    /// The artifact this launch already carries under `binding`, or `None`.
    #[allow(clippy::too_many_arguments)]
    fn cached_artifact(
        &self,
        plan: &Plan,
        graph: &EGraph,
        launch_ix: usize,
        binds: &BindingEnv,
        binding: &crate::lower::DimBinding,
        keys: &[ArtifactKey],
        pack: &Arc<UniformPack>,
    ) -> Result<Option<(Artifact, [u32; 3])>> {
        let key = keys[launch_ix];
        let cached = {
            let mut lock = self.artifacts.lock();
            match lock.get_mut(&key) {
                Some(entry) => match variant_hash(&entry.consulted, binds)
                    .and_then(|vh| entry.variants.get(&vh).cloned())
                {
                    // The stored grid is the building binding's; replay it.
                    Some(ArtifactVariant {
                        artifact,
                        grid,
                        grid_space,
                        ph,
                    }) => {
                        let grid = match &grid_space {
                            Some(spec) => spec.grid(binding, &self.caps().limits)?,
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
        if !crate::flags().verify_artifact_cache {
            return Ok(Some((artifact, grid)));
        }
        // Verification mode: relower and compare identity + grid.
        let Lowered { ir, ph, .. } = self.lower_uncached(plan, graph, launch_ix, binds, pack)?;
        let same_body = ph == cached_ph || {
            let emitted = crate::emit::emit(&ir, self.caps()).map_err(Error::from)?;
            let key = format!("{}\n{}", ir.block, wgsl_text(&emitted)?);
            self.pipelines_by_source
                .lock()
                .peek(&key)
                .and_then(|a| a.downcast_ref::<GpuArtifact>())
                .zip(artifact.downcast_ref::<GpuArtifact>())
                .is_some_and(|(a, b)| a.id == b.id)
        };
        if !same_body || ir.grid != grid {
            eprintln!(
                "[artifact-cache] MISMATCH launch {launch_ix} name {}: body {ph} grid {:?} cached ({cached_ph}, {grid:?})",
                ir.name, ir.grid,
            );
            if let Some(dir) = &crate::flags().mismatch_dump {
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if n < 40 {
                    let _ = std::fs::write(
                        dir.join(format!("mismatch_{n}_{launch_ix}.txt")),
                        format!("{ir:#?}"),
                    );
                }
            }
        }
        Ok(Some((artifact, grid)))
    }

    /// Emit and compile one lowered body, checking the WGSL tier first. Runs
    /// with the body's [`pipeline_hash`] slot held.
    fn compile_body(&self, ir: &KernelIr, ph: u128) -> Result<Artifact> {
        let _g = Stopwatch::start(&COMPILE_US);
        let share = !crate::flags().no_pipeline_share;
        let emitted = crate::emit::emit(ir, self.caps()).map_err(Error::from)?;
        let text = wgsl_text(&emitted)?;
        if let Some(dir) = &crate::flags().wgsl_dump {
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

    /// Lower a launch whose artifact is not cached. Dispatch must use the
    /// `KernelIr::grid` the body was indexed against, not `Launch::grid`.
    fn lower_uncached(
        &self,
        plan: &Plan,
        graph: &EGraph,
        launch_ix: usize,
        binds: &BindingEnv,
        pack: &Arc<UniformPack>,
    ) -> Result<Lowered> {
        let launch = plan
            .launches
            .get(launch_ix)
            .ok_or_else(|| Error::Plan(format!("no launch at index {launch_ix}")))?;
        let cx = LowerCtx {
            plan,
            launch,
            graph,
            symbols: &plan.symbols,
            dim_bindings: &[],
        };
        let theta = plan
            .extraction
            .theta
            .get(&launch.root)
            .copied()
            .unwrap_or(SchedPoint::Point);
        let node = graph.node(launch.root);
        let binding =
            crate::lower::DimBinding::from_pairs(binds.dims.iter().map(|(k, v)| (*k, *v)));
        let ir = {
            let _t = Stopwatch::start(&LOWER_US);
            crate::lower::lower_node(self.caps(), node, theta, &cx, binding.clone(), pack.clone())?
        };
        let ph = pipeline_hash(&ir);
        #[cfg(any(test, feature = "compiler-tests"))]
        fusor_tile::verify_kernel(&ir, self.caps())?;
        Ok(Lowered { ir, ph, binding })
    }

    /// The compiled pipeline for one body, claimed single-flight. With `wait`
    /// a slot another worker holds is waited on; otherwise `None`.
    fn pipeline(&self, ir: &KernelIr, ph: u128, wait: bool) -> Result<Option<Artifact>> {
        if crate::flags().no_pipeline_share {
            return self.compile_body(ir, ph).map(Some);
        }
        let slot: PipelineSlot = {
            let mut lock = self.pipelines.lock();
            Arc::clone(lock.get_or_insert(ph, PipelineSlot::default))
        };
        let mut built = match wait {
            true => slot.lock(),
            false => match slot.try_lock() {
                Some(built) => built,
                None => return Ok(None),
            },
        };
        if let Some(a) = built.as_ref() {
            return Ok(Some(a.clone()));
        }
        let a = self.compile_body(ir, ph)?;
        *built = Some(a.clone());
        Ok(Some(a))
    }

    /// File one lowering's artifact under its launch key, returning the grid
    /// the body was indexed against.
    fn record_variant(
        &self,
        key: ArtifactKey,
        lowered: Lowered,
        artifact: &Artifact,
        binds: &BindingEnv,
    ) -> Result<[u32; 3]> {
        let Lowered { ir, ph, binding } = lowered;
        let grid = ir.grid;
        // A replayable grid is recomputed per binding; otherwise its symbols
        // stay in the variant key.
        let grid_space = binding.grid_derivation(grid, &self.caps().limits);
        let consulted = binding.body_consulted(grid_space.is_some());
        let vh = variant_hash(&consulted, binds).ok_or_else(|| {
            Error::Plan("a lowering consulted a symbol the dispatch does not bind".into())
        })?;
        let mut lock = self.artifacts.lock();
        let entry = lock.get_or_insert_mut(key, || ArtifactEntry {
            consulted: Vec::new(),
            variants: lru::LruCache::new(NonZeroUsize::new(VARIANTS_PER_LAUNCH).expect("nonzero")),
        });
        entry.consulted = consulted;
        entry.variants.put(
            vh,
            ArtifactVariant {
                artifact: artifact.clone(),
                grid,
                grid_space,
                ph,
            },
        );
        Ok(grid)
    }
}

/// One lowered launch, before compilation.
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

/// A resolve's symbol bindings, runtime scalars and caller-owned buffers.
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
    fn dim(&self, d: fusor_ir::shape::Dim) -> Option<u64> {
        // A derived symbol evaluates through the symbols it reaches.
        d.evaluate(&mut |s| self.dims.get(&s).copied())
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
        let descs = crate::bindings::bindings_from_module(&module);
        let bindings: Vec<(u32, bool)> = descs.iter().map(|b| (b.binding, b.read_only)).collect();
        let entries = crate::bindings::layout_entries(&descs);
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
        // SAFETY: every emitted load is masked or provably in range and every
        // loop is counted; the compiler test harness checks this.
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
        let trace = crate::flags().trace_dispatch;
        if trace {
            eprintln!("[compile] start {name} bindings={}", entries.len());
        }
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        if trace {
            eprintln!("[compile] done {name}");
        }
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

    fn lower(&self, node: &Node, _: Id, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
        crate::lower::lower(self.caps(), node, theta, cx)
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

    fn copy(&self, src: &Buf) -> Result<Buf> {
        let source = crate::pool::GpuBuffer::of(src, "copy source")?;
        let dst = self.pool.alloc_with_usage(source.size, source.usage)?;
        self.launcher.copy_buffer(src, &dst, source.size)?;
        Ok(dst)
    }

    fn wait(&self) -> Result<()> {
        self.launcher.poll_wait()
    }
}

impl GpuTarget {
    /// Under `FUSOR_CACHE_STATS`, print retained-object counts every 64
    /// resolves.
    fn cache_stats(&self) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        if !crate::flags().cache_stats {
            return;
        }
        let n = N.fetch_add(1, Ordering::Relaxed);
        if !n.is_multiple_of(64) {
            return;
        }
        let pool = self.pool.counters();
        eprintln!(
            "[cache-stats] resolves={n} artifacts={} pipelines={} by_source={} \
             bind_groups={} pool_live_mib={} pool_created={}",
            self.artifacts.lock().len(),
            self.pipelines.lock().len(),
            self.pipelines_by_source.lock().len(),
            self.launcher.bind_group_count(),
            pool.live_bytes >> 20,
            pool.created,
        );
        use std::sync::atomic::Ordering::Relaxed;
        eprintln!(
            "[upload-stats] uniform_kib={} init_kib={} copy_kib={} poison_kib={}",
            crate::pool::UPLOAD_UNIFORM.load(Relaxed) >> 10,
            crate::pool::UPLOAD_INIT.load(Relaxed) >> 10,
            crate::pool::COPY_BYTES.load(Relaxed) >> 10,
            crate::pool::POISON_BYTES.load(Relaxed) >> 10,
        );
    }
}

impl GpuTarget {
    /// Forget every compiled kernel only a losing race candidate used, so
    /// autotuning and the member sweep don't grow the caches without bound.
    pub fn release_candidates(&self, arena: u64, candidates: &[Arc<Plan>], keep: &Plan) {
        let keep_keys: FxHashSet<ArtifactKey> =
            plan_artifact_keys(keep, &UniformPack::new(keep), arena)
                .into_iter()
                .collect();
        {
            let mut artifacts = self.artifacts.lock();
            for candidate in candidates {
                let pack = UniformPack::new(candidate);
                for key in plan_artifact_keys(candidate, &pack, arena) {
                    if !keep_keys.contains(&key) {
                        artifacts.pop(&key);
                    }
                }
            }
        }
        self.sweep_unreferenced();
    }

    /// Forget every compiled kernel of graph `arena`, which can never be
    /// looked up again.
    pub fn release_arena(&self, arena: u64) {
        if lru_retain(&mut self.artifacts.lock(), |k, _| k.arena != arena) {
            self.sweep_unreferenced();
        }
    }

    /// Drop pipelines and bind groups no artifact entry references.
    fn sweep_unreferenced(&self) {
        let live: FxHashSet<u64> = self
            .artifacts
            .lock()
            .iter()
            .flat_map(|(_, entry)| entry.variants.iter())
            .filter_map(|(_, variant)| variant.artifact.downcast_ref::<GpuArtifact>())
            .map(|a| a.id)
            .collect();
        let id_of = |artifact: &Artifact| artifact.downcast_ref::<GpuArtifact>().map(|a| a.id);
        let dead = |id: Option<u64>| id.is_some_and(|id| !live.contains(&id));
        lru_retain(&mut self.pipelines_by_source.lock(), |_, a| !dead(id_of(a)));
        // A slot still being compiled is `None`; it stays.
        lru_retain(&mut self.pipelines.lock(), |_, slot| {
            !dead(slot.try_lock().and_then(|a| a.as_ref().and_then(id_of)))
        });
        self.launcher.retain_bind_groups(&live);
    }
}
