//! [`GpuTarget`] — the [`Target`] implementation tying device, lowering,
//! emission, the pool, the artifact cache and the launcher together.

use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::Instant;

use fusor2_ir::Result;
use fusor2_ir::cost::DeviceFacts;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::Persistence;
use fusor2_ir::egraph::{EGraph, Id, Rule};
use fusor2_ir::error::Error;
use fusor2_ir::extract::{Launch, Plan, PlanHash};
use fusor2_ir::ir::Node;
use fusor2_ir::ir::level1::SchedPoint;
use fusor2_ir::ir::level2::KernelIr;
use fusor2_ir::shape::SymId;
use fusor2_ir::target::{Artifact, Buf, EmitError, LowerCtx, Target, Uniforms};
use rustc_hash::{FxHashMap, FxHasher};

use crate::device::GpuDevice;
use crate::launch::{
    BuildCursor, CommandRecord, GpuArtifact, KernelProfile, Launcher,
    should_parallelize_build_remainder,
};
use crate::pool::BufferPool;
use crate::uniforms::UniformPack;

/// Runtime policy. Every field names a decision the library owns; none is read
/// from the environment at point of use, and none gates an *optimizer*
/// behaviour — those are all per-shape cost-model calls.
#[derive(Clone, Debug)]
pub struct GpuConfig {
    /// Override the platform memory ceiling.
    pub max_gpu_memory_bytes: Option<u64>,
    /// Pre-fill fresh allocations with `0xCD` so a zero-init assumption fails
    /// loudly instead of reading the last tenant's bytes.
    pub poison_allocations: bool,
    /// Back-pressure window: the runtime blocks when more than this many
    /// submissions are outstanding, so a training script never counts steps by
    /// hand.
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

/// Live compiled pipelines retained per target. A transformer training step's
/// whole distinct kernel set fits well inside this; past it the LRU drops the
/// coldest, which costs one rebuild and never correctness.
pub const ARTIFACT_CAPACITY: usize = 1024;

/// Everything the emitted kernel body depends on.
///
/// `PlanHash` is `hash(realized DAG term + M + theta + DeviceFacts)`: it fixes
/// the launch set, every member's op with symbols *as symbols*, every leaf
/// operand's dtype and shape, `materialized`, `theta`, and the device. What it
/// deliberately leaves out is exactly what the rest of this key adds back:
/// `Plan::buffers`, which `lower::bound_layout` treats as the authoritative
/// padded stride set; `Plan::symbols`, which `UniformPack::new` turns into
/// baked uniform slot indices; and the concrete extents `DimBinding` resolves
/// `KernelIr::grid` against. `launch` pins which dispatch of the plan this is.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
struct ArtifactKey {
    plan: PlanHash,
    launch: u64,
    tail: u64,
    dims: u64,
}

fn artifact_key(plan: &Plan, launch: &Launch, tail: u64, dims: u64) -> ArtifactKey {
    let mut lh = FxHasher::default();
    launch.hash(&mut lh);
    ArtifactKey {
        plan: plan.hash,
        launch: lh.finish(),
        tail,
        dims,
    }
}

/// The resolve-invariant halves of [`ArtifactKey`]: `Plan::buffers` +
/// `Plan::symbols`, and the bound extents. Hashed once per resolve, not once
/// per launch.
fn resolve_key_parts(plan: &Plan, binds: &BindingEnv) -> (u64, u64) {
    let mut th = FxHasher::default();
    plan.buffers.hash(&mut th);
    plan.symbols.hash(&mut th);

    // An `FxHashMap` has no stable iteration order, so the dim fold has to be
    // commutative. Keys are distinct `SymId`s, so no two terms cancel.
    let mut dims = 0u64;
    for (sym, value) in &binds.dims {
        let mut dh = FxHasher::default();
        dh.write_u32(sym.0);
        dh.write_u64(*value);
        dims = dims.wrapping_add(dh.finish());
    }
    (th.finish(), dims)
}

/// The wgpu backend.
pub struct GpuTarget {
    device: Arc<GpuDevice>,
    pool: BufferPool,
    artifacts: parking_lot::Mutex<lru::LruCache<ArtifactKey, (Artifact, [u32; 3])>>,
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
        let launcher = Launcher::new(wgpu_device, queue, backend, config.clone());
        Ok(Self {
            device,
            pool,
            artifacts: parking_lot::Mutex::new(lru::LruCache::new(
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
    pub fn launcher(&self) -> &Launcher {
        &self.launcher
    }
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    pub fn take_kernel_profiles(&self) -> Vec<KernelProfile> {
        self.launcher.take_kernel_profiles()
    }

    /// The plan **is** the cache key: one hash recipe, so a new decision
    /// variable cannot be forgotten in some other one.
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
    /// 2. **Parallel** — artifact-cache lookup by [`ArtifactKey`], else lower,
    ///    verify L2, emit and create the pipeline. A serial probe runs first so
    ///    a warm cache never touches the thread pool.
    /// 3. **Serial, exact plan order** — push command records and release
    ///    consumed buffers.
    pub fn resolve(&self, plan: &Plan, graph: &EGraph, binds: &BindingEnv) -> Result<()> {
        let start = Instant::now();
        let (key_tail, key_dims) = resolve_key_parts(plan, binds);
        let pack = UniformPack::new(plan);
        let uniforms = pack.fill(plan, &binds.dims, &binds.scalars)?;

        // Allocate and upload uniforms in plan order.
        let uniform_buf = self
            .pool
            .alloc_with_usage(pack.byte_len(), crate::pool::TENSOR_USAGE)?;
        self.launcher.write_uniforms(&uniform_buf, &uniforms)?;

        // `plan.buffers` deliberately excludes external leaves: allocation is
        // derived from what the plan *produces*, and a leaf is supplied. Every
        // binding must still resolve, so the caller-owned buffers seed the map
        // before anything is allocated on top of them.
        let mut resolved: FxHashMap<Id, Buf> = binds.buffers.clone();
        for buffer in &plan.buffers {
            if resolved.contains_key(&buffer.value) {
                continue;
            }
            let elements = binds
                .dim(buffer.elements)
                .ok_or_else(|| Error::Plan(format!("buffer {} has an unbound extent", buffer.value)))?;
            let bytes = elements.saturating_mul(buffer.dtype.byte_size()).max(4);
            resolved.insert(
                buffer.value,
                self.pool.alloc(bytes, buffer.persistence)?,
            );
        }

        let mut work: Vec<LaunchWork> = Vec::with_capacity(plan.launches.len());
        for launch in &plan.launches {
            // `derive_bindings` is the only producer of `Launch::bindings` and
            // emits them in ascending binding order.
            debug_assert!(launch.bindings.windows(2).all(|w| w[0].binding < w[1].binding));
            let mut buffers = Vec::with_capacity(launch.bindings.len() + 1);
            buffers.push(uniform_buf.clone());
            for b in &launch.bindings {
                let buf = resolved.get(&b.value).cloned().ok_or_else(|| {
                    Error::Plan(format!("launch binds {} which the plan never allocates", b.value))
                })?;
                buffers.push(buf);
            }
            work.push(LaunchWork {
                grid: launch.grid,
                buffers,
                artifact: None,
            });
        }

        // Build the queue: probe serially, then cut over to parallel.
        // `work[i]` was built from `plan.launches[i]`, so the index carries
        // the launch identity through both build passes.
        let queue_len = work.len();
        let mut cutover = queue_len;
        for i in 0..queue_len {
            let began = Instant::now();
            let (artifact, grid) =
                self.build_one(plan, graph, &plan.launches[i], binds, key_tail, key_dims)?;
            work[i].artifact = Some(artifact);
            work[i].grid = grid;
            let remaining = queue_len - i - 1;
            if should_parallelize_build_remainder(queue_len, remaining, began.elapsed()) {
                cutover = i + 1;
                break;
            }
        }
        if cutover < queue_len {
            // Every compiled artifact lives behind a `OnceLock` on the cached
            // kernel, so a cohort can only duplicate work, never observe a
            // half-built pipeline.
            let cursor = BuildCursor::new();
            let tail = &mut work[cutover..];
            let len = tail.len();
            let results: Vec<Mutexed> = (0..len).map(|_| Mutexed::default()).collect();
            std::thread::scope(|scope| {
                let threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
                    .min(len);
                for _ in 0..threads {
                    scope.spawn(|| {
                        while let Some(i) = cursor.take(len) {
                            let built = self.build_one(
                                plan,
                                graph,
                                &plan.launches[cutover + i],
                                binds,
                                key_tail,
                                key_dims,
                            );
                            *results[i].0.lock() = Some(built);
                        }
                    });
                }
            });
            for (i, slot) in results.into_iter().enumerate() {
                let (artifact, grid) = slot
                    .0
                    .into_inner()
                    .ok_or_else(|| Error::Device("a build worker dropped its slot".into()))??;
                tail[i].artifact = Some(artifact);
                tail[i].grid = grid;
            }
        }

        // Record dispatches serially, in exact plan order.
        let total = work.len();
        let query_set = self.launcher.timestamp_query_set(total);
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
                bind_group: Arc::new(bind_group),
                grid: item.grid,
            });
        }
        self.launcher
            .encode_command_records(&records, query_set.as_ref())?;

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

        if let Some(set) = query_set.as_ref() {
            // The query set is resolved from a command buffer submitted after
            // a `poll_wait`: Metal's writeback of the final encoder's boundary
            // samples races a resolve encoded behind it and leaves slots zero.
            self.launcher.poll_wait()?;
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

    /// Lower, verify, emit and compile one launch, returning it with **the
    /// grid the lowering indexed its body against**. `Launch::grid` is the
    /// cost model's workgroup count, derived from the schedule point;
    /// `KernelIr::grid` is what the kernel body actually assumes. Dispatching
    /// the former silently computes a prefix of the output.
    fn build_one(
        &self,
        plan: &Plan,
        graph: &EGraph,
        launch: &Launch,
        binds: &BindingEnv,
        key_tail: u64,
        key_dims: u64,
    ) -> Result<(Artifact, [u32; 3])> {
        let key = artifact_key(plan, launch, key_tail, key_dims);
        if let Some(hit) = self.artifacts.lock().get(&key).cloned() {
            return Ok(hit);
        }
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
            crate::lower::lower_node(self.caps(), node, theta, &cx, binding)?;
        let ir = if kernels.is_empty() {
            return Err(Error::Plan("a launch lowered to no kernel".into()));
        } else {
            kernels.remove(0)
        };
        // `verify_l2` is never optional and never a fallback: a failure is
        // `Error::Lower`.
        fusor2_tile::verify_l2(&ir, self.caps())?;
        let grid = ir.grid;
        let artifact = self.emit(&ir).map_err(Error::from)?;
        self.artifacts.lock().put(key, (artifact.clone(), grid));
        Ok((artifact, grid))
    }
}

/// A slot a build worker fills.
#[derive(Default)]
struct Mutexed(parking_lot::Mutex<Option<Result<(Artifact, [u32; 3])>>>);

struct LaunchWork {
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
        let module = crate::emit::emit(ir, self.caps())?;
        let bindings: Vec<(u32, bool)> = crate::bindings::bindings_from_module(&module)
            .into_iter()
            .map(|b| (b.binding, b.read_only))
            .collect();
        let entries = crate::bindings::layout_entries(
            &crate::bindings::bindings_from_module(&module),
        );
        let device = self.device.device();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(ir.name),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(ir.name),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        // SAFETY: every load this compiler emits is masked or provably in
        // range and every loop is counted — `verify_l2` establishes both
        // before emission, which is exactly what licenses the trusted path.
        let module = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some(ir.name),
                    source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        self.launcher.note_pipeline_compile();
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(ir.name),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Ok(Artifact::new(GpuArtifact {
            name: ir.name,
            pipeline: Arc::new(pipeline),
            layout: Arc::new(layout),
            bindings,
            block: ir.block,
        }))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn back_pressure_is_a_library_policy_with_a_real_default() {
        let c = GpuConfig::default();
        assert_eq!(c.max_in_flight_submits, 8);
        assert!(!c.poison_allocations);
        assert!(!c.trace_gpu_kernels);
        assert!(c.max_gpu_memory_bytes.is_none());
    }

    #[test]
    fn binding_env_resolves_both_kinds_of_symbol() {
        let s = SymId(3);
        let lr = SymId(9);
        let env = BindingEnv::new().with_dim(s, 512).with_scalar(lr, 1e-3);
        assert_eq!(env.dim(fusor2_ir::shape::Dim::Sym(s)), Some(512));
        assert_eq!(env.dim(fusor2_ir::shape::Dim::Const(7)), Some(7));
        assert_eq!(env.dim(fusor2_ir::shape::Dim::Sym(SymId(99))), None);
        assert_eq!(env.scalars.get(&lr).copied(), Some(1e-3));
    }
}
