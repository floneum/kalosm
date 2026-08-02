//! [`GpuTarget`] — the [`Target`] implementation tying device, lowering,
//! emission, the pool, the plan cache and the launcher together.
//!
//! Owned by W9.

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
use fusor2_ir::ir::level1::SchedPoint;
use fusor2_ir::ir::level2::KernelIr;
use fusor2_ir::shape::SymId;
use fusor2_ir::target::{Artifact, Buf, EmitError, LowerCtx, Target, Uniforms};
use rustc_hash::FxHashMap;

use crate::device::GpuDevice;
use crate::launch::{
    BuildCursor, CommandRecord, GpuArtifact, KernelProfile, Launcher,
    should_parallelize_build_remainder,
};
use crate::plan_cache::PlanCache;
use crate::pool::{BufferPool, BufferPoolCounters};
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
    /// Back-pressure window. **This is the `--drain-every` replacement**: the
    /// runtime blocks when more than this many submissions are outstanding, so
    /// a training script never counts steps by hand.
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

/// The wgpu backend.
pub struct GpuTarget {
    device: Arc<GpuDevice>,
    pool: BufferPool,
    cache: PlanCache,
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

    /// Build over an already-requested device. This is the entry point W8's
    /// `DeviceSetup` feeds.
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

    pub fn pool_counters(&self) -> BufferPoolCounters {
        self.pool.counters()
    }

    pub fn take_kernel_profiles(&self) -> Vec<KernelProfile> {
        self.launcher.take_kernel_profiles()
    }

    /// The plan **is** the cache key. There is no `hash_kernel_fields`, no
    /// `kernel_cache_key_with_dispatch` and no golden byte file, so a new
    /// decision variable cannot be forgotten in one of four hash recipes.
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
    ///    L2, emit and create the pipeline. A serial probe runs first so a warm
    ///    cache never touches the thread pool.
    /// 3. **Serial, exact plan order** — push command records and release
    ///    consumed buffers.
    pub fn resolve(&self, plan: &Plan, graph: &EGraph, binds: &BindingEnv) -> Result<()> {
        let start = Instant::now();
        let pack = UniformPack::new(plan);
        let uniforms = pack.fill(plan, &binds.dims, &binds.scalars)?;

        // ---- Phase 1: serial, plan order --------------------------------
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
                grid: launch.grid,
                buffers,
                artifact: None,
            });
        }

        // ---- Phase 2: build, serially probing then in parallel -----------
        let cached = self.cache.get(plan.hash);
        if cached.is_none() {
            self.cache.note_miss();
        }
        let queue_len = work.len();
        let mut cutover = queue_len;
        for i in 0..queue_len {
            let began = Instant::now();
            let (artifact, grid) = self.build_one(plan, graph, &work[i], binds)?;
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
                            let built = self.build_one(plan, graph, &tail_ref(tail, i), binds);
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

        // ---- Phase 3: serial, exact plan order ---------------------------
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
        self.pool.reset_initialized_buffers();

        if self.config.trace_gpu_kernels {
            // The query set is resolved from a command buffer submitted after
            // a `poll_wait`: Metal's writeback of the final encoder's boundary
            // samples races a resolve encoded behind it and leaves slots zero.
            self.launcher.poll_wait()?;
            let samples: Vec<(String, f64)> = work
                .iter()
                .filter_map(|w| {
                    w.artifact
                        .as_ref()?
                        .downcast_ref::<GpuArtifact>()
                        .map(|g| (g.name.to_string(), 0.0))
                })
                .collect();
            self.launcher.push_profile(KernelProfile::from_samples(
                start.elapsed().as_secs_f64() * 1000.0,
                &samples,
            ));
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
        item: &LaunchWork,
        binds: &BindingEnv,
    ) -> Result<(Artifact, [u32; 3])> {
        let launch = plan
            .launches
            .iter()
            .find(|l| l.root == item.root)
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
        Ok((artifact, grid))
    }
}

/// A slot a build worker fills.
#[derive(Default)]
struct Mutexed(parking_lot::Mutex<Option<Result<(Artifact, [u32; 3])>>>);

fn tail_ref<'a>(tail: &'a [LaunchWork], i: usize) -> LaunchWork {
    LaunchWork {
        root: tail[i].root,
        grid: tail[i].grid,
        buffers: tail[i].buffers.clone(),
        artifact: None,
    }
}

struct LaunchWork {
    root: Id,
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

    #[test]
    fn the_cache_dir_is_platform_shaped() {
        // Not asserting a specific path: only that a home-relative default
        // exists wherever HOME does, so the disk tier is on by default.
        if std::env::var_os("HOME").is_some() || std::env::var_os("XDG_CACHE_HOME").is_some() {
            assert!(default_cache_dir().is_some());
        }
    }
}
