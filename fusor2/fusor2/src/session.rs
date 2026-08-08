//! `Session` and `Device`. The session owns the target, the cost model, the
//! extractor and the plan cache; `resolve` is the one place saturation,
//! extraction and dispatch happen.
//!
//! Owned by W13.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use fusor2_cost::tune_cache::Verdict;
use fusor2_cost::{LocalSearch, ReplayMemo, Roofline};
use fusor2_cpu::CpuTarget;
use fusor2_gpu::GpuTarget;
use fusor2_ir::CORE_RULES;
use fusor2_ir::cost::CostModel;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::{Dtype, Persistence};
use fusor2_ir::egraph::{EGraph, Id, Rule, Saturate, SaturationBudget, SaturationDelta};
use fusor2_ir::extract::{ExtractBudget, Extractor, Plan, ReplayKey};
use fusor2_ir::ir::level0::{L0, LeafKind};
use fusor2_ir::ir::level2::ArenaPlanner;
use fusor2_ir::ir::{Op, OpDefRegistry, Semantics};
use fusor2_ir::saturate::Driver;
use fusor2_ir::shape::Dim;
use fusor2_ir::target::{Buf, LowerCtx, Target, Uniforms};
use fusor2_tile::{Planner, SCHED_RULES};
use rustc_hash::FxHashMap;

use crate::composite::register_macro_ops;
use crate::graph::GraphRef;
use crate::tensor::Tensor;
use crate::{Error, Result};

/// Submitted-but-unretired plans the session will let pile up before it
/// blocks in `resolve`. Back-pressure is a **runtime policy the library
/// owns** — the trainer's `--drain-every` counter disappears.
pub const MAX_INFLIGHT_PLANS: u32 = 8;

/// Contractions below this never pay for a measurement round. Override with
/// `FUSOR2_AUTOTUNE_MIN_MACS`; `0` tunes everything, which is how the
/// conformance suite is made to exercise the tuner.
pub const AUTOTUNE_MIN_MACS: u64 = 64 << 20;
/// Runs per candidate: the first pays pipeline compilation, the second is the
/// sample, and the minimum of the two is taken. A cold-path cost paid once
/// per `ReplayKey`.
/// Timed repeats per candidate, min taken.
///
/// Raised from 2 when tuning stopped being contraction-only. With every node
/// family eligible, one plan can offer candidates at six launches instead of
/// one, and each adoption is a chance for a lucky sample to displace a better
/// plan. Measured at 2 repeats and a 3% margin, attention oscillated between
/// 2.6 ms and 5.4 ms run to run on an unchanged graph; the winner was noise,
/// not a decision. Four repeats and a wider margin make an adoption mean
/// something. The extra timing is exactly what the per-machine cache exists to
/// amortize.
const TUNE_RUNS: usize = 4;
/// How much better a candidate must be to displace the incumbent. Wide enough
/// that run-to-run noise cannot drive an adoption — see [`TUNE_RUNS`].
///
/// **Measured on the launch's own kernel span** wherever a device timer exists,
/// and on the whole plan only when one does not. A candidate differs from the
/// incumbent at exactly one launch, so on the plan sum the rule reads
/// `s_c < s_b - m*sum_b`: the relative win demanded at that launch is
/// `m*sum_b/s_b`, which makes any launch smaller than `m` of the plan
/// unadoptable no matter what its field contains. A warming M2 Max moves a
/// 2048-cube matmul 2-3% between back-to-back runs on the *wall clock* that
/// this constant was originally sized against; a min over [`TUNE_RUNS`] GPU
/// timestamp spans of one kernel is far tighter than that.
const TUNE_MARGIN: f64 = 0.08;

/// Proof that the holder owns a graph's `resolve_lock`.
///
/// Passed rather than taken by the `_locked` entry points, so "the caller
/// already holds it" is checked by the borrow checker instead of by comment.
pub(crate) type ResolveGuard<'a> = parking_lot::MutexGuard<'a, ()>;

/// Which backend a session runs on.
#[derive(Clone)]
pub enum Device {
    Cpu(Arc<CpuTarget>),
    Gpu(Arc<GpuTarget>),
}

impl Device {
    pub fn cpu() -> Result<Self> {
        Ok(Self::Cpu(Arc::new(CpuTarget::new()?)))
    }

    pub async fn gpu() -> Result<Self> {
        Ok(Self::Gpu(Arc::new(GpuTarget::new().await?)))
    }

    pub fn gpu_blocking() -> Result<Self> {
        Ok(Self::Gpu(Arc::new(GpuTarget::new_blocking()?)))
    }

    pub fn target(&self) -> Arc<dyn Target> {
        match self {
            Self::Cpu(t) => Arc::clone(t) as Arc<dyn Target>,
            Self::Gpu(t) => Arc::clone(t) as Arc<dyn Target>,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            Self::Gpu(_) => "gpu",
        }
    }

    pub fn caps(&self) -> Caps {
        match self {
            Self::Cpu(t) => t.caps().clone(),
            Self::Gpu(t) => t.caps().clone(),
        }
    }

    /// Upload host bytes into a fresh device buffer.
    ///
    /// `Target` deliberately exposes only `alloc`: a write path would have to
    /// name a byte layout, and only the backend knows one. This is the one
    /// place the session distinguishes the two backends, and it distinguishes
    /// them on *upload* — never on lowering, cost or selection.
    pub(crate) fn upload(&self, bytes: &[u8], persistence: Persistence) -> Result<Buf> {
        match self {
            Self::Gpu(t) => t
                .pool()
                .create_buffer_init(bytes, fusor2_gpu::pool::TENSOR_USAGE),
            Self::Cpu(t) => {
                let buf = t.alloc(bytes.len().max(4) as u64, persistence)?;
                let aligned = buf
                    .downcast_ref::<fusor2_cpu::AlignedBuf>()
                    .ok_or_else(|| Error::Device("cpu buffer was not an AlignedBuf".into()))?;
                // `AlignedBuf::as_mut_ptr` is the backend's own interior-mutable
                // handle; the buffer was just allocated, so nothing else reads it.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(aligned.as_mut_ptr(), aligned.len()) };
                dst[..bytes.len()].copy_from_slice(bytes);
                Ok(buf)
            }
        }
    }

    /// Copy a device buffer back to the host. One of exactly three host syncs.
    fn download(&self, buf: &Buf, bytes: u64) -> Result<Vec<u8>> {
        match self {
            Self::Gpu(t) => pollster::block_on(t.readback(buf, bytes)),
            Self::Cpu(_) => {
                let aligned = buf
                    .downcast_ref::<fusor2_cpu::AlignedBuf>()
                    .ok_or_else(|| Error::Device("cpu buffer was not an AlignedBuf".into()))?;
                let len = (bytes as usize).min(aligned.len());
                Ok(aligned.as_slice()[..len].to_vec())
            }
        }
    }
}

/// One device, one cost model, one extractor, one plan cache.
#[derive(Clone)]
pub struct Session {
    inner: Arc<SessionInner>,
}

pub struct SessionInner {
    pub device: Device,
    pub cost: Arc<dyn CostModel>,
    pub extractor: Arc<dyn Extractor>,
    pub planner: Arc<dyn ArenaPlanner>,
    pub registry: OpDefRegistry,
    semantics: Arc<dyn Semantics>,
    rules: Vec<Rule>,
    replay: ReplayMemo,
    /// Recorded saturations. A tier *below* `replay`: it removes the work
    /// that produces the graph a plan is extracted from, where `replay`
    /// removes the extraction itself.
    saturation: SaturationMemo,
    /// What this machine has already learned about which kernels are cheap.
    /// Persisted per caps fingerprint, so a second process starts from the
    /// first one's measurements instead of re-timing them.
    tune: fusor2_cost::tune_cache::TuneCache,
    launches: AtomicU64,
    in_flight: AtomicU32,
}

impl Session {
    pub fn new(device: Device) -> Result<Self> {
        let planner = Planner::shared();
        let device_fingerprint = device.caps().fingerprint();

        // The one registration point. Ids follow table order because
        // `PlanHash` reads registration order.
        let mut registry = OpDefRegistry::new();
        register_macro_ops(&mut registry);

        let semantics =
            fusor2_ir::CoreSemantics::with_registry(Arc::clone(&planner), registry.clone());
        let target = device.target();
        let cost: Arc<dyn CostModel> = Arc::new(Roofline::new(target.facts().clone()));
        let extractor: Arc<dyn Extractor> = Arc::new(
            LocalSearch::new(Arc::clone(&planner), target.caps().clone())
                .with_registry(registry.clone()),
        );

        // Rule order carries no semantics; the fixed order exists only for
        // reproducibility.
        let mut rules: Vec<Rule> = Vec::new();
        rules.extend_from_slice(CORE_RULES);
        rules.extend_from_slice(SCHED_RULES);
        rules.extend_from_slice(fusor2_autograd::ADJOINT_RULES);
        rules.extend_from_slice(target.rules());

        Ok(Self {
            inner: Arc::new(SessionInner {
                device,
                cost,
                extractor,
                planner,
                registry,
                semantics,
                rules,
                replay: ReplayMemo::new(),
                tune: fusor2_cost::tune_cache::TuneCache::load(device_fingerprint),
                saturation: SaturationMemo::default(),
                launches: AtomicU64::new(0),
                in_flight: AtomicU32::new(0),
            }),
        })
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    pub fn caps(&self) -> Caps {
        self.inner.device.caps()
    }

    pub fn semantics(&self) -> Arc<dyn Semantics> {
        Arc::clone(&self.inner.semantics)
    }

    pub fn registry(&self) -> &OpDefRegistry {
        &self.inner.registry
    }

    pub fn rules(&self) -> &[Rule] {
        &self.inner.rules
    }

    /// Saturate, extract, lower, emit and dispatch everything `values` needs.
    ///
    /// Atomic against every other resolve and readback on the same graph — see
    /// [`crate::graph::GraphInner::resolve_lock`] for why the e-graph's own
    /// mutex cannot do that job.
    pub fn resolve(&self, values: &[Tensor]) -> Result<()> {
        let Some(first) = values.first() else {
            return Ok(());
        };
        let graph = first.graph.clone();
        let resolving = graph.resolve_lock.lock();
        self.resolve_locked(&resolving, values)
    }

    /// [`Self::resolve`]'s body, for a caller already holding the graph's
    /// `resolve_lock` — `read_back` does, so that a resolve and the readback
    /// that follows it are one section.
    pub(crate) fn resolve_locked(
        &self,
        resolving: &ResolveGuard<'_>,
        values: &[Tensor],
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let graph = values[0].graph.clone();
        for v in values {
            if !Arc::ptr_eq(&v.graph, &graph) {
                return Err(Error::Device(
                    "operands come from two different graphs".into(),
                ));
            }
        }

        // Own the back-pressure: the trainer's `--drain-every` is a runtime
        // policy, not a counter in a training script.
        if self.inner.in_flight.load(Ordering::Relaxed) >= MAX_INFLIGHT_PLANS {
            self.wait()?;
        }

        let caps = self.caps();
        let binding: Vec<Dim> = graph
            .dim_bindings()
            .into_iter()
            .map(|(_, v)| Dim::Const(v))
            .collect();

        let (plan, roots, key, missed) = {
            let mut g = graph.egraph.lock();
            for v in values {
                g.add_root(v.id);
            }
            let __t_sat = Instant::now();
            let __pre_nodes = g.len();
            // Saturation is a pure function of `(graph, caps, rules, budget)`
            // and a `Session` fixes the last three for its whole life, so a
            // graph in a pre-state seen before saturates to a graph seen
            // before. Replaying that recording is the *same* answer the driver
            // would compute, and the memo's validity check is an exact
            // node-by-node comparison, not a fingerprint — a mismatch falls
            // through to the driver, so a miss is slow and never wrong.
            let __replayed = self.inner.saturation.replay(&mut g);
            if !__replayed {
                let pre = g.pre_saturation();
                Driver::new().saturate(
                    &mut g,
                    &caps,
                    &self.inner.rules,
                    SaturationBudget::default(),
                )?;
                self.inner.saturation.insert(g.record_saturation(pre));
            }
            let __sat_us = __t_sat.elapsed().as_micros();

            let roots: Vec<Id> = g.roots().to_vec();
            let __t_rest = Instant::now();
            let key = ReplayKey {
                l0_term: fusor2_cost::replay::l0_term_hash(&g, &roots),
                device: self.inner.cost.facts().fingerprint(),
                binding: fusor2_cost::replay::binding_hash(&binding),
            };
            // Tuning is a cold-path cost: it happens on a memo miss and the
            // winner is what every later resolve of this key replays.
            let missed = self.inner.replay.get(key).is_none();
            let graph_ref: &EGraph = &g;
            let (plan, _unchanged) = self.inner.replay.get_or_extract(key, || {
                self.inner.extractor.extract(
                    graph_ref,
                    &roots,
                    self.inner.cost.as_ref(),
                    ExtractBudget::default(),
                )
            })?;
            self.inner.extractor.verify_plan(graph_ref, &plan)?;
            if resolve_profile() {
                eprintln!(
                    "[profile] saturate{} {} us ({} -> {} nodes), extract+verify {} us",
                    if __replayed { " (replayed)" } else { "" },
                    __sat_us,
                    __pre_nodes,
                    g.len(),
                    __t_rest.elapsed().as_micros()
                );
            }
            (plan, roots, key, missed)
        };

        let plan = if missed && matches!(self.inner.device, Device::Gpu(_)) {
            let tuned = self.autotune(resolving, &graph, &roots, plan, values)?;
            self.inner.replay.insert(key, (*tuned).clone());
            tuned
        } else {
            plan
        };

        self.run(&graph, &plan, values)?;
        self.inner
            .launches
            .fetch_add(plan.launches.len() as u64, Ordering::Relaxed);
        self.inner.in_flight.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Submit whatever is encoded without waiting.
    ///
    /// Every backend encodes and submits inside `launch`, so there is no
    /// deferred encoder to push. This exists so a caller need not know that.
    pub fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Block until every submitted dispatch has retired.
    pub fn wait(&self) -> Result<()> {
        self.inner.device.target().wait()?;
        self.inner.in_flight.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Dispatches issued since construction. The conformance
    /// `resolves_in::<N>` asserts read this, so it counts **dispatches**, not
    /// encoder submissions.
    pub fn launch_count(&self) -> u64 {
        self.inner.launches.load(Ordering::Relaxed)
    }

    /// Resolve `values` and report whether it took exactly `N` dispatches.
    ///
    /// The two counter reads bracket the resolve *inside* the graph's
    /// resolve lock, so the difference is this call's own dispatches even
    /// when another thread is resolving the same graph. Counting outside it
    /// would attribute the other thread's launches to this assert.
    pub fn resolves_in<const N: u64>(&self, values: &[Tensor]) -> Result<bool> {
        let Some(first) = values.first() else {
            return Ok(N == 0);
        };
        let graph = first.graph.clone();
        let resolving = graph.resolve_lock.lock();
        let before = self.launch_count();
        self.resolve_locked(&resolving, values)?;
        Ok(self.launch_count() - before == N)
    }

    /// Bytes of an already-resolved value.
    ///
    /// `_resolving` is a witness that the caller holds the graph's
    /// `resolve_lock`, not a parameter this reads. Downloading a buffer is
    /// only meaningful while no other thread can be part-way through binding
    /// and dispatching a plan that writes it, and taking the guard here
    /// instead would leave a window open between a caller's `resolve` and its
    /// read. Passing the guard makes "I already hold it" the only way to
    /// call this.
    pub(crate) fn read_bytes_locked(
        &self,
        _resolving: &ResolveGuard<'_>,
        graph: &GraphRef,
        id: Id,
    ) -> Result<Vec<u8>> {
        let buf = graph
            .device_buf(id)
            .ok_or_else(|| Error::Plan(format!("{id} has no device buffer; resolve it first")))?;
        let facts = graph.facts(id);
        let elem = facts.dtype.byte_size();
        // **No drain here.** `download` records its copy on the same queue the
        // plan was just submitted to and wgpu orders submissions, so the copy
        // cannot observe an unfinished dispatch; the `map_async` inside
        // `Launcher::readback` is what blocks, and wgpu defers that callback
        // until the submission that wrote the staging buffer has retired.
        //
        // Draining first stalls the host until the GPU is *idle* and only then
        // allocates staging, records the copy and commits it, so the readback
        // command buffer pays a cold GPU wake-up every resolve. The reference
        // does not: `fusor-ml`'s `Tensor::as_slice` goes through
        // `materialize_with_tail(enqueue_download)`, which encodes
        // `copy_buffer_to_buffer` into the *resolve's own* encoder and maps
        // once. Two round trips became one.
        //
        // Measured round 7, Apple M2 Max, median ms: passthrough 2.83 -> 2.65,
        // matmul 2048-cube 9.63 -> 8.72, matmul+epilogue 11.04 -> 10.42,
        // elementwise x20 5.51 -> 5.04, attention [1,8,1024,64] 3.60 -> 3.48.
        // Conformance failure list byte-identical before and after.
        //
        // `in_flight` is back-pressure bookkeeping only (`MAX_INFLIGHT_PLANS`),
        // and the download on the next line blocks on the device regardless.
        self.inner.in_flight.store(0, Ordering::Relaxed);

        // A selected `Coop`/`Sgemm` geometry pads the output buffer to its
        // tile multiple, so the bytes on the device are *not* the value's own
        // dense shape. Read the whole padded buffer and gather the value out
        // of it; a dense layout takes the straight path.
        //
        // The registered layout is stated against the *selected member's*
        // shape, and the id being read may be a reshaped spelling of the same
        // class — attention reads `[b, h, q, d]` off a contract whose padded
        // buffer is `[b*h, m_pad, n_pad]`. Restating the layout over the
        // reader's shape is exact (`restate_layout`); what is **not** allowed
        // is dropping it. The old filter did exactly that on any rank
        // mismatch, and a dense read of a padded buffer returns the top-left
        // corner plus padding zeros as if they were the value.
        let padded = graph
            .device_layout(id)
            .filter(|l| l.shape() != &facts.shape[..])
            .map(|l| {
                restate_layout(&l, &facts.shape, graph).ok_or_else(|| {
                    Error::Plan(format!(
                        "value {id} is shaped {:?} over a device buffer laid out {:?}; \
                         the shapes do not factor",
                        facts.shape,
                        l.shape()
                    ))
                })
            })
            .transpose()?;
        let Some(layout) = padded else {
            let elements = resolve_elements(&facts.shape, graph)?;
            return self.inner.device.download(&buf, elements * elem);
        };

        let base = resolve_dim(layout.offset(), graph)?;
        let strides: Vec<u64> = layout
            .strides()
            .iter()
            .map(|d| resolve_dim(*d, graph))
            .collect::<Result<_>>()?;
        // The bytes to pull are the layout's *address span*, not its element
        // count: a restated layout addresses far past `product(shape)` — the
        // `[2,2,3,4]` read of a `[4,16,16]`-padded contract touches flat
        // index 803 while holding 48 elements, and a short download gathers
        // zeros for everything past its end.
        let mut span = 1u64;
        for (d, s) in layout.shape().iter().zip(&strides) {
            span += resolve_dim(*d, graph)?.saturating_sub(1).saturating_mul(*s);
        }
        let raw = self.inner.device.download(&buf, (base + span) * elem)?;
        let extents: Vec<u64> = facts
            .shape
            .iter()
            .map(|d| resolve_dim(*d, graph))
            .collect::<Result<_>>()?;
        let count = extents.iter().product::<u64>() as usize;
        let mut out = Vec::with_capacity(count * elem as usize);
        let mut idx = vec![0u64; extents.len()];
        for _ in 0..count {
            let flat = base
                + idx
                    .iter()
                    .zip(&strides)
                    .map(|(i, s)| i * s)
                    .sum::<u64>();
            let start = (flat * elem) as usize;
            let end = start + elem as usize;
            match raw.get(start..end) {
                Some(slice) => out.extend_from_slice(slice),
                None => out.extend(std::iter::repeat_n(0u8, elem as usize)),
            }
            for axis in (0..extents.len()).rev() {
                idx[axis] += 1;
                if idx[axis] < extents[axis] {
                    break;
                }
                idx[axis] = 0;
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------
    // Running a plan
    // -----------------------------------------------------------------

    /// The class member this plan selected for `id`.
    ///
    /// `Extraction::sigma` is `ClassId -> Id`, so it only runs one way: from
    /// the class a value belongs to, to the member the extractor picked. The
    /// facade holds the `L0` id the user built; the plan names the selected
    /// member. Those diverge the moment any rewrite fires, which is always.
    fn selected(&self, graph: &GraphRef, plan: &Plan, id: Id) -> Id {
        let class = graph.egraph.lock().class_of(id);
        plan.extraction.selected(class).unwrap_or(id)
    }

    /// Register `buf` under **every** id in `id`'s e-class, `Union` spine
    /// included.
    ///
    /// Registering only the selected member is what left `read_bytes` looking
    /// in a map its key was never inserted into. The class — not the member —
    /// is the stable identity of a value: `sigma` is keyed by it, and which
    /// member wins is an artifact of one extraction that a later resolve may
    /// change. Writing the whole class also means a later extraction that
    /// selects a different member overwrites every stale entry instead of
    /// leaving one to shadow.
    ///
    /// `EGraph::members` is the *selectable* set and drops the `Union` nodes,
    /// but `macro_op` hands the caller the id `union(defn, sugar)` returned —
    /// so every sugared spelling (`softmax`, `rms_norm`, `rope`, `attention`,
    /// every windowed view) names its value by a spine node. Binding only the
    /// members left all of those unreadable.
    fn bind_class(
        &self,
        graph: &GraphRef,
        id: Id,
        buf: &Buf,
        layout: Option<&fusor2_ir::shape::Layout>,
    ) {
        let members = {
            let g = graph.egraph.lock();
            g.class_ids(g.class_of(id))
        };
        for m in members {
            match layout {
                Some(l) => graph.set_device_buf_with_layout(m, buf.clone(), l.clone()),
                None => graph.set_device_buf(m, buf.clone()),
            }
        }
    }

    fn run(&self, graph: &GraphRef, plan: &Plan, values: &[Tensor]) -> Result<()> {
        // What the extractor selected for each requested value. `sigma` is
        // keyed by `ClassId`, so this is the only direction it runs in, and
        // the id the user's `Tensor` holds is generally *not* it.
        let wanted: Vec<Id> = values
            .iter()
            .map(|v| self.selected(graph, plan, v.id))
            .collect();
        // Every external leaf the plan reads, uploaded once. A `Persistent`
        // leaf keeps its buffer across resolves, which is what makes an
        // optimizer's state stay on device with no host round trip.
        let mut supplied: FxHashMap<Id, Buf> = FxHashMap::default();
        for launch in &plan.launches {
            for binding in &launch.bindings {
                if supplied.contains_key(&binding.value) {
                    continue;
                }
                if let Some(buf) = self.leaf_buffer(graph, binding.value)? {
                    supplied.insert(binding.value, buf);
                }
            }
        }
        // Root outputs are allocated here rather than inside the backend so
        // the handle survives for readback.
        for buffer in &plan.buffers {
            if supplied.contains_key(&buffer.value) {
                continue;
            }
            let is_launch_root = plan.launches.iter().any(|l| l.root == buffer.value);
            if !is_launch_root && !wanted.contains(&buffer.value) {
                continue;
            }
            let elements = resolve_dim(buffer.elements, graph)?;
            let bytes = (elements * buffer.dtype.byte_size()).max(4);
            let buf = self.inner.device.target().alloc(bytes, buffer.persistence)?;
            self.bind_class(graph, buffer.value, &buf, Some(&buffer.layout));
            supplied.insert(buffer.value, buf);
        }
        // A value the caller asked for that the plan never had to allocate: a
        // bare leaf, or a leaf that only ever appears as an operand. Reading
        // one back is still a legal request, and a graph of nothing but a leaf
        // has no launches at all.
        for (value, selected) in values.iter().zip(&wanted) {
            if graph.device_buf(value.id).is_some() {
                continue;
            }
            let buf = match supplied.get(selected) {
                Some(buf) => buf.clone(),
                None => match self.free_leaf_buffer(graph, *selected)? {
                    Some(buf) => {
                        supplied.insert(*selected, buf.clone());
                        buf
                    }
                    None => continue,
                },
            };
            self.bind_class(graph, *selected, &buf, None);
        }

        match &self.inner.device {
            Device::Gpu(target) => {
                let mut env = fusor2_gpu::target::BindingEnv::new();
                for (sym, value) in graph.dim_bindings() {
                    env = env.with_dim(sym, value);
                }
                for (sym, value) in graph.uniform_scalars() {
                    env = env.with_scalar(sym, value);
                }
                for (id, buf) in &supplied {
                    env = env.with_buffer(*id, buf.clone());
                }
                let g = graph.egraph.lock();
                target.resolve(plan, &g, &env)
            }
            Device::Cpu(target) => {
                let target = Arc::clone(target);
                self.run_cpu(target.as_ref(), graph, plan, &mut supplied)
            }
        }
    }

    /// The generic runner: one `lower -> emit -> launch` per plan launch, in
    /// plan order. The GPU takes `GpuTarget::resolve` instead, which adds the
    /// plan cache, the parallel build cohort and one encoder per resolve.
    fn run_cpu(
        &self,
        target: &CpuTarget,
        graph: &GraphRef,
        plan: &Plan,
        supplied: &mut FxHashMap<Id, Buf>,
    ) -> Result<()> {
        let uniforms = self.uniforms_for(plan, graph)?;

        for buffer in &plan.buffers {
            if supplied.contains_key(&buffer.value) {
                continue;
            }
            let elements = resolve_dim(buffer.elements, graph)?;
            let bytes = (elements * buffer.dtype.byte_size()).max(4);
            supplied.insert(buffer.value, target.alloc(bytes, buffer.persistence)?);
        }

        let g = graph.egraph.lock();
        for launch in &plan.launches {
            let theta = plan
                .extraction
                .theta
                .get(&launch.root)
                .copied()
                .unwrap_or(fusor2_ir::ir::level1::SchedPoint::Point);
            let cx = LowerCtx {
                plan,
                launch,
                graph: &g,
                symbols: &plan.symbols,
            };
            let ir = target.lower(g.node(launch.root), launch.root, theta, &cx)?;
            let artifact = target.emit(&ir)?;

            let mut ordered: Vec<_> = launch.bindings.iter().collect();
            ordered.sort_by_key(|b| b.binding);
            let mut binds = Vec::with_capacity(ordered.len());
            for b in ordered {
                let buf = supplied.get(&b.value).cloned().ok_or_else(|| {
                    Error::Plan(format!("launch binds {} which nothing allocates", b.value))
                })?;
                binds.push(buf);
            }
            // **The kernel's own grid, not the plan's.** `Launch::grid` is the
            // cost model's workgroup count, derived from the schedule point;
            // `KernelIr::grid` is what the lowering actually indexed the body
            // against. When they disagree the kernel silently computes a
            // prefix of its output.
            target.launch(&artifact, ir.grid, &binds, &uniforms)?;
        }
        Ok(())
    }

    /// Binding 0 for the CPU launcher, indexed by raw `SymId`: a dim symbol
    /// contributes its extent, a scalar symbol its `f32` bits, and the
    /// emitter bitcasts on read. Neither ever enters a kernel's identity.
    fn uniforms_for(&self, plan: &Plan, graph: &GraphRef) -> Result<Uniforms> {
        let dims = graph.dim_bindings();
        let scalars = graph.uniform_scalars();
        let highest = plan
            .symbols
            .iter()
            .map(|s| s.0)
            .chain(dims.iter().map(|(s, _)| s.0))
            .chain(scalars.iter().map(|(s, _)| s.0))
            .filter(|s| *s != u32::MAX)
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(1);
        let mut words = vec![0u32; highest.max(1)];
        for (sym, value) in dims {
            words[sym.0 as usize] = u32::try_from(value)
                .map_err(|_| Error::Plan(format!("extent {value} exceeds a u32")))?;
        }
        for (sym, value) in scalars {
            words[sym.0 as usize] = value.to_bits();
        }
        Ok(Uniforms {
            dims: words,
            scalars: Vec::new(),
        })
    }

    /// The buffer backing a leaf the caller asked to read back.
    ///
    /// An external leaf uploads its host bytes. A `Const` splat and a
    /// `Uniform` scalar have **no buffer by design** — a constant is folded
    /// into the kernel and a uniform is a word in binding 0 — but reading one
    /// back is still a legal request (a comparison's adjoint *is* a splat of
    /// zero), so their bytes are materialized here rather than reported as a
    /// missing buffer.
    fn free_leaf_buffer(&self, graph: &GraphRef, id: Id) -> Result<Option<Buf>> {
        if let Some(buf) = self.leaf_buffer(graph, id)? {
            return Ok(Some(buf));
        }
        let leaf = {
            let g = graph.egraph.lock();
            match &g.node(id).op {
                Op::L0(L0::Leaf(k @ (LeafKind::Const { .. } | LeafKind::Uniform { .. }))) => {
                    k.clone()
                }
                _ => return Ok(None),
            }
        };
        let facts = graph.facts(id);
        let unit = match &leaf {
            LeafKind::Const { value, .. } => splat_bytes(*value),
            LeafKind::Uniform { sym, .. } => {
                splat_bytes(fusor2_autograd::tape::splat_of(
                    facts.dtype,
                    graph.uniform_value(*sym).unwrap_or(0.0),
                )?)
            }
            _ => return Ok(None),
        };
        let elements = resolve_elements(&facts.shape, graph)? as usize;
        let mut bytes = Vec::with_capacity(elements * unit.len());
        for _ in 0..elements {
            bytes.extend_from_slice(&unit);
        }
        let buf = self.inner.device.upload(&bytes, facts.persistence)?;
        graph.set_device_buf(id, buf.clone());
        Ok(Some(buf))
    }

    /// Time the base plan and every `(family, geometry)` alternative of each
    /// contraction launch, keep the fastest that reproduces the base's
    /// values.
    ///
    /// Coordinate descent, incumbent carried forward, so attention's two
    /// contractions are tuned against each other rather than in isolation.
    /// **The measurement travels with the plan it measured**: a round-3 probe
    /// kept them in parallel lists, desynced them, and applied a tile it had
    /// not timed.
    fn autotune(
        &self,
        guard: &ResolveGuard<'_>,
        graph: &GraphRef,
        roots: &[Id],
        base: Arc<Plan>,
        values: &[Tensor],
    ) -> Result<Arc<Plan>> {
        let min_macs = std::env::var("FUSOR2_AUTOTUNE_MIN_MACS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(AUTOTUNE_MIN_MACS);
        let log = std::env::var_os("FUSOR2_AUTOTUNE_LOG").is_some();

        // One probe pass over the base plan. `launch_variants` is where both
        // the MAC gate and the `Effect::Pure` guard live, so "every launch
        // offered nothing" is simultaneously "not worth tuning" and "not safe
        // to re-run" — and neither the base measurement nor any candidate run
        // may happen before it has answered.
        let probe: Vec<Vec<(String, Plan)>> = {
            let g = graph.egraph.lock();
            (0..base.launches.len())
                .map(|ix| {
                    self.inner.extractor.launch_variants(
                        &g,
                        roots,
                        &base,
                        ix,
                        self.inner.cost.as_ref(),
                        min_macs,
                    )
                })
                .collect()
        };
        if probe.iter().all(Vec::is_empty) {
            return Ok(base);
        }

        // Every measurement below is scored on the kernel's own GPU span rather
        // than on the whole plan's wall clock, which needs the per-dispatch
        // timestamp path on for the duration of the pass and off again after
        // it. A resolve that is not being tuned never allocates a query set.
        let _clock = TuningClock::new(&self.inner.device);

        let Some(reference) = self.timed_run(guard, graph, &base, values)? else {
            return Ok(base);
        };
        // The plan's identity across processes: every launch signature in
        // order. A cached combination is only replayable onto the same plan
        // shape, so this is what it is keyed on.
        let plan_sig: String = {
            let g = graph.egraph.lock();
            base.launches
                .iter()
                .map(|l| fusor2_cost::extract::launch_signature(&g, l))
                .collect::<Vec<_>>()
                .join(";")
        };
        // What this pass actually adopts, per launch, so the *combination* can
        // be recorded rather than reassembled later from per-launch minima
        // that were never measured together.
        let mut picks: Vec<Option<String>> = vec![None; base.launches.len()];

        let mut best = Arc::clone(&base);
        let base_ns = plan_ns(&reference);
        let mut best_ns = base_ns;
        // The incumbent's own per-launch spans, plan order. A candidate differs
        // from the incumbent at exactly one launch, so the plan difference
        // between them *is* that launch's span difference — which makes the
        // launch's own span, not the sum, the term `TUNE_MARGIN` belongs on.
        let mut best_spans: Option<Vec<f64>> = reference.gpu_us.clone();
        if log {
            eprintln!(
                "[tune] base {best_ns:.0} ns ({} ns wall), {} launches, {}",
                reference.nanos,
                best.launches.len(),
                if reference.gpu_us.is_some() {
                    "per-kernel gpu timestamps"
                } else {
                    "wall clock only"
                }
            );
        }

        for (ix, probed) in probe.into_iter().enumerate() {
            if probed.is_empty() {
                continue;
            }
            // The incumbent is carried across launches, so once a tile has
            // been adopted the next launch's alternatives must be re-derived
            // against it; while nothing has moved the probe is still exact.
            let variants = if Arc::ptr_eq(&best, &base) {
                probed
            } else {
                let g = graph.egraph.lock();
                self.inner.extractor.launch_variants(
                    &g,
                    roots,
                    &best,
                    ix,
                    self.inner.cost.as_ref(),
                    min_macs,
                )
            };
            // What this machine already knows about this launch decides where
            // the time goes: re-confirm a known incumbent, explore a bounded
            // number of never-tried points, and do not rebuild the tail this
            // device has already shown to be hopeless. The signature is
            // structural, so it is the *same* key the previous process wrote.
            let sig = {
                let g = graph.egraph.lock();
                best.launches
                    .get(ix)
                    .map(|l| fusor2_cost::extract::launch_signature(&g, l))
            };
            let variants: Vec<(String, Plan)> = match &sig {
                Some(sig) => {
                    let names: Vec<String> =
                        variants.iter().map(|(n, _)| n.clone()).collect();
                    // Nothing left to learn about this launch: apply the
                    // accumulated winner instead of re-racing the field on this
                    // run's clock. The minimum over every past run is a much
                    // better estimate than one fresh noisy sample, and it is
                    // the difference between converging on 2.58 ms and settling
                    // at ~3.0 ms. Still timed and still value-checked — the
                    // cache proposes, the measurement disposes.
                    // A jointly-measured combination wins over anything
                    // assembled per launch: replay this launch's pick from it.
                    // `None` means the winning plan left this launch alone, so
                    // there is nothing to try here at all.
                    // **Replay only once there is nothing left to learn.**
                    // A combination recorded while the space was still being
                    // explored is just the best of the handful tried so far,
                    // and replaying it unconditionally freezes that answer
                    // forever — measured, attention locked at 4.2 ms because
                    // run 1's bounded sweep was cached and never revisited.
                    // So the combination is authoritative only when every
                    // candidate for this launch has been measured; until then
                    // exploration continues and the combination is rewritten
                    // at the end of each pass.
                    // **The cache orders and prunes; it never replaces the
                    // race.** Two stronger uses were built and measured and
                    // both lost time:
                    //
                    //  * applying each launch's cached arg-min — per-launch
                    //    minima are scored in whatever context the descent was
                    //    in when that launch's turn came, so assembling them
                    //    produces a plan nobody ran (attention 4.2 ms against
                    //    a 2.67 ms descent);
                    //  * replaying the whole recorded winning combination —
                    //    `launch_variants` re-derives against the carried
                    //    incumbent, so a label does not name the same plan on
                    //    the next pass, and the replay did not reproduce its
                    //    own recording (4.2-5.5 ms against 2.67 ms).
                    //
                    // Ordering is sound because it changes only *which
                    // candidate is tried first*, and every candidate is still
                    // built, timed and value-checked. Making the cache
                    // authoritative over the measurement is what broke; making
                    // it a prior over the measurement is what works.
                    let (run, skipped) = self.inner.tune.plan_candidates(sig, &names);
                    if log && !skipped.is_empty() {
                        eprintln!(
                            "[tune]   L{ix} skipping {} variant(s) this device has \
                             already ruled out",
                            skipped.len()
                        );
                    }
                    let order: Vec<String> = run.into_iter().cloned().collect();
                    let mut by_name: std::collections::HashMap<String, Plan> =
                        variants.into_iter().collect();
                    order
                        .into_iter()
                        .filter_map(|n| by_name.remove(&n).map(|p| (n, p)))
                        .collect()
                }
                None => variants,
            };
            for (label, candidate) in variants {
                let candidate = Arc::new(candidate);
                let Ok(Some(sample)) = self.timed_run(guard, graph, &candidate, values) else {
                    continue;
                };
                let sample_ns = plan_ns(&sample);
                // A different tile is a different reduction order, so bit
                // equality is the wrong test — but a *wrong kernel* is off by
                // orders of magnitude. The round-3 probe found Sgemv
                // returning 0.40 where the answer is 0.92 on a 2048-cube
                // matmul, and 14x faster. This gate is not optional.
                //
                // **It runs before the cache write, not after.** A candidate
                // differs from the incumbent at exactly one class, so a
                // whole-plan disagreement is this variant's, and it is the
                // single most important fact this device can remember about it.
                let ok = reference.bytes.len() == sample.bytes.len()
                    && reference
                        .bytes
                        .iter()
                        .zip(&sample.bytes)
                        .all(|((dt, a), (_, b))| agrees(*dt, a, b));
                // `replan` rebuilds the *whole* plan from a one-node edit, so a
                // per-launch quantity may be attributed to index `ix` only when
                // every other launch is untouched. This licenses both the cache
                // write below and the adoption test further down.
                //
                // Equal roots are not that guarantee. A launch is its root *and*
                // its members, grid and block, and a replan can move a fused
                // member across an existing launch boundary — or flip a
                // neighbour between reading a materialized producer and
                // recomputing it, which inlines that producer's members — while
                // both roots and the launch count stay put. `gpu_us[ix]` is then
                // small because this launch shed work onto its neighbour, and
                // adopting on it buys a slower plan and files that span under
                // this kernel's name for every later process.
                let aligned = candidate.launches.len() == best.launches.len()
                    && candidate
                        .launches
                        .iter()
                        .zip(&best.launches)
                        .enumerate()
                        .all(|(j, (c, b))| {
                            j == ix
                                || (c.root == b.root
                                    && c.grid == b.grid
                                    && c.block == b.block
                                    && {
                                        // Member *order* is a realization
                                        // detail — `launch_signature` sorts for
                                        // the same reason — but the member set
                                        // is the work.
                                        let mut cm: Vec<Id> = c.members.to_vec();
                                        let mut bm: Vec<Id> = b.members.to_vec();
                                        cm.sort_unstable();
                                        bm.sort_unstable();
                                        cm == bm
                                    })
                        });
                if let Some(sig) = &sig {
                    // `sig` names *one launch*, so only a number that is a
                    // property of that launch may be filed under it. That is
                    // the launch's own kernel span, which the device timer
                    // measures directly; the whole-plan wall clock is not one,
                    // because coordinate descent carries an incumbent and the
                    // same variant clocks differently depending on when its
                    // turn came. Hence the `aligned` filter below: a *time* is
                    // attributable only when index `ix` really is this launch.
                    // A *wrong answer* needs no such guard — it is not a timing
                    // property, it does not depend on a device timer, and it is
                    // identical on the CPU target.
                    let verdict = if !ok {
                        Some(Verdict::Wrong)
                    } else {
                        match launch_ns(&sample, ix).filter(|_| aligned) {
                            // Absolute nanoseconds are comparable across
                            // processes because `launch_signature` pins family,
                            // dtype and shape, and the file is keyed by device.
                            Some(ns) => Some(Verdict::Ran(ns)),
                            // No device timer at all: keep the ppm-of-base
                            // ratio, the best a host clock supports. A GPU that
                            // *can* time kernels but did not time this plan
                            // records nothing, rather than mixing two units in
                            // one device's file.
                            None if matches!(self.inner.device, Device::Cpu(_)) => {
                                Some(Verdict::Ran(ratio_ppm(sample_ns, base_ns)))
                            }
                            None => None,
                        }
                    };
                    if let Some(verdict) = verdict {
                        self.inner.tune.record(sig, &label, verdict);
                    }
                }
                if log {
                    eprintln!(
                        "[tune]   L{ix} {sample_ns:.0} ns  (own {} ns, {} ns wall)  {label}{}{}",
                        launch_ns(&sample, ix)
                            .map_or_else(|| "-".to_string(), |ns| ns.to_string()),
                        sample.nanos,
                        if ok { "" } else { "  REJECTED: wrong values" },
                        if aligned {
                            ""
                        } else {
                            "  PERTURBED: >1 launch differs"
                        }
                    );
                }
                // Only launch `ix` differs, so `sum_c - sum_b == s_c - s_b`
                // exactly and the launch's own span is what `TUNE_MARGIN`
                // applies to. On the sum the rule reads `s_c < s_b - m*sum_b`:
                // the required *relative* win at this launch is `m*sum_b/s_b`,
                // so a launch under `m` of the plan total can never be adopted
                // at all (attention: anything under 0.21 ms of its 2.65 ms) and
                // even a launch at 1.0 ms of that plan must get 21% faster to
                // clear an "8%" margin. Same quantity the cache records, so the
                // ordering it hands back and the decision made on it finally
                // agree.
                let improved = match (&best_spans, &sample.gpu_us) {
                    (Some(prev), Some(now))
                        if aligned && prev.len() == now.len() && ix < prev.len() =>
                    {
                        now[ix] < prev[ix] * (1.0 - TUNE_MARGIN)
                    }
                    // No device timer, or `replan` moved the launches: the
                    // whole-plan number is all there is, and is what the host
                    // clock always had.
                    _ => sample_ns < best_ns * (1.0 - TUNE_MARGIN),
                };
                if ok && improved {
                    best_ns = sample_ns;
                    // The adopted candidate *is* the new incumbent, so its
                    // profile is the incumbent's profile for every later
                    // launch's comparison.
                    if sample.gpu_us.is_some() {
                        best_spans = sample.gpu_us.clone();
                    }
                    best = candidate;
                    if let Some(slot) = picks.get_mut(ix) {
                        *slot = Some(label.clone());
                    }
                }
            }
        }
        // Record the combination that actually won. A combination spans every
        // launch, so unlike a per-launch record it stays a ratio against this
        // pass's own base — that is what makes it comparable across runs.
        let combo_score = ratio_ppm(best_ns, base_ns);
        self.inner.tune.record_combo(&plan_sig, picks, combo_score);
        // One write per tuning pass, atomic, and a no-op when nothing new was
        // measured — a fully-learned shape costs zero IO.
        self.inner.tune.save();
        if log {
            eprintln!("[tune] winner {best_ns:.0} ns");
        }
        Ok(best)
    }

    /// Run `plan` [`TUNE_RUNS`] times, keep the fastest, and read every
    /// requested value back. `None` when nothing was readable.
    fn timed_run(
        &self,
        guard: &ResolveGuard<'_>,
        graph: &GraphRef,
        plan: &Plan,
        values: &[Tensor],
    ) -> Result<Option<TuneSample>> {
        let mut nanos = u64::MAX;
        let mut gpu_us: Option<Vec<f64>> = None;
        for _ in 0..TUNE_RUNS {
            let t = Instant::now();
            self.run(graph, plan, values)?;
            self.inner.device.target().wait()?;
            self.inner.in_flight.store(0, Ordering::Relaxed);
            nanos = nanos.min(t.elapsed().as_nanos() as u64);
            if let Device::Gpu(target) = &self.inner.device
                && let Some(us) = target.launcher().take_last_profile()
            {
                // Element-wise minimum, for the same reason the wall clock is a
                // minimum: a slow sample is contention, a fast one is the
                // kernel.
                gpu_us = Some(match gpu_us {
                    Some(prev) if prev.len() == us.len() => {
                        prev.iter().zip(&us).map(|(a, b)| a.min(*b)).collect()
                    }
                    _ => us,
                });
            }
        }
        let mut bytes = Vec::with_capacity(values.len());
        for v in values {
            match self.read_bytes_locked(guard, graph, v.id) {
                Ok(b) => bytes.push((graph.facts(v.id).dtype, b)),
                Err(_) => return Ok(None),
            }
        }
        Ok(Some(TuneSample {
            nanos,
            gpu_us,
            bytes,
        }))
    }

    /// The device buffer of an external leaf, uploading it on first use.
    fn leaf_buffer(&self, graph: &GraphRef, id: Id) -> Result<Option<Buf>> {
        let is_external = {
            let g = graph.egraph.lock();
            matches!(
                &g.node(id).op,
                Op::L0(L0::Leaf(
                    LeafKind::Buffer { .. } | LeafKind::Param { .. } | LeafKind::Quantized { .. }
                ))
            )
        };
        if !is_external {
            return Ok(None);
        }
        if let Some(buf) = graph.device_buf(id) {
            return Ok(Some(buf));
        }
        let facts = graph.facts(id);
        let buf = match graph.with_leaf_bytes(id, |bytes| {
            self.inner.device.upload(bytes, facts.persistence)
        }) {
            Some(uploaded) => uploaded?,
            None => {
                let elements = resolve_elements(&facts.shape, graph)?;
                let bytes = (elements * facts.dtype.byte_size()).max(4);
                self.inner.device.target().alloc(bytes, facts.persistence)?
            }
        };
        graph.set_device_buf(id, buf.clone());
        Ok(Some(buf))
    }
}

/// Whether `FUSOR2_RESOLVE_PROFILE` is set. Read once: `resolve` is the hot
/// path and an env lookup per call is a per-resolve allocation.
fn resolve_profile() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR2_RESOLVE_PROFILE").is_some())
}

/// Whether `FUSOR2_NO_SAT_MEMO` is set. The A/B switch that shows a
/// suspected miscompile is or is not the saturation memo without a rebuild;
/// read once, because `resolve` is the hot path.
fn saturation_memo_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FUSOR2_NO_SAT_MEMO").is_some())
}

/// Recorded saturations, bounded and FIFO-evicted.
///
/// Not keyed on a hash. [`EGraph::replay_saturation`] checks the *whole*
/// pre-state by value, so this is a list to scan rather than a map to look
/// up — and `could_apply_to` rejects on node count and symbol counter before
/// a single node is compared, which is O(1) for every entry that is not the
/// one wanted.
#[derive(Default)]
struct SaturationMemo(parking_lot::Mutex<Vec<Arc<SaturationDelta>>>);

/// Recordings kept. Matches `fusor2_cost::replay::CAPACITY`, so the two tiers
/// hold the same number of distinct terms.
const SATURATION_MEMO_CAPACITY: usize = 64;

/// Nodes the whole memo may hold. A recording carries a copy of its graph, so
/// the entry count alone is not a memory bound: a caller that *grows* one
/// graph — a training loop building a longer tape every step — misses every
/// time, and would otherwise leave 64 ever-larger recordings behind that can
/// never be hit again. This is what keeps that case flat instead of linear.
const SATURATION_MEMO_NODES: usize = 256 << 10;

impl SaturationMemo {
    /// Replay a recording onto `graph` if one was taken against exactly this
    /// pre-state. `false` means the caller must saturate for real.
    fn replay(&self, graph: &mut EGraph) -> bool {
        if saturation_memo_disabled() {
            return false;
        }
        // Cloned out of the lock: `replay_saturation` needs `&mut EGraph`
        // and the delta at once, and the entries are `Arc`s precisely so
        // that costs a refcount rather than a copy.
        let candidates: Vec<Arc<SaturationDelta>> = {
            let entries = self.0.lock();
            entries
                .iter()
                .rev()
                .filter(|d| d.could_apply_to(graph))
                .cloned()
                .collect()
        };
        candidates.iter().any(|d| graph.replay_saturation(d))
    }

    fn insert(&self, delta: SaturationDelta) {
        let mut entries = self.0.lock();
        entries.push(Arc::new(delta));
        // Oldest first, and never down to nothing: a single recording larger
        // than the whole node budget is still worth keeping, because the
        // caller that produced it is the one likeliest to ask again.
        while entries.len() > 1
            && (entries.len() > SATURATION_MEMO_CAPACITY
                || entries.iter().map(|d| d.prefix() + d.added()).sum::<usize>()
                    > SATURATION_MEMO_NODES)
        {
            entries.remove(0);
        }
    }
}

/// One candidate's measurement and the values it produced, together — never
/// in parallel lists.
struct TuneSample {
    /// Wall clock around the whole `run`, in nanoseconds. Includes buffer
    /// allocation, binding, the plan-cache lookup, submission and the poll
    /// spin, so it is a property of the *process*, not of any one kernel.
    nanos: u64,
    /// Per-launch GPU kernel spans in microseconds, plan order, when the device
    /// could time them. This is the number a tuning decision wants: it excludes
    /// every host cost above and is a property of one launch.
    gpu_us: Option<Vec<f64>>,
    bytes: Vec<(Dtype, Vec<u8>)>,
}

/// The number a tuning decision is made on, in nanoseconds: the GPU's
/// per-launch spans summed, so every host cost and every inter-pass gap is
/// excluded. Without a device timer this is the whole-plan wall clock,
/// unchanged.
fn plan_ns(s: &TuneSample) -> f64 {
    match &s.gpu_us {
        Some(us) => us.iter().sum::<f64>() * 1000.0,
        None => s.nanos as f64,
    }
}

/// What `(launch, variant)` may record: that launch's own kernel span, in ns.
fn launch_ns(s: &TuneSample, ix: usize) -> Option<u64> {
    Some((s.gpu_us.as_ref()?.get(ix).copied()? * 1000.0) as u64)
}

/// Parts-per-million of the base plan's time. The best a host clock supports,
/// and what a device with no kernel timer still records.
fn ratio_ppm(sample: f64, base: f64) -> u64 {
    ((sample / base.max(1.0)) * 1_000_000.0).clamp(0.0, u64::MAX as f64) as u64
}

/// Turns the per-dispatch timestamp path on for a tuning pass and off again on
/// every exit path, so a production resolve never pays for a query set.
struct TuningClock<'a>(Option<&'a GpuTarget>);

impl<'a> TuningClock<'a> {
    fn new(device: &'a Device) -> Self {
        let target = match device {
            Device::Gpu(t) => Some(t.as_ref()),
            Device::Cpu(_) => None,
        };
        if let Some(t) = target {
            t.launcher().set_tuning(true);
        }
        Self(target)
    }
}

impl Drop for TuningClock<'_> {
    fn drop(&mut self) {
        if let Some(t) = self.0 {
            t.launcher().set_tuning(false);
            let _ = t.launcher().take_last_profile();
        }
    }
}

/// Byte-identical, or — for f32 — within 1e-3 of the reference's own
/// magnitude. NaN fails the comparison and is therefore rejected.
fn agrees(dtype: Dtype, a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if a == b {
        return true;
    }
    if dtype != Dtype::F32 {
        return false;
    }
    let f = |s: &[u8]| {
        s.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>()
    };
    let (x, y) = (f(a), f(b));
    let scale = x.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    x.iter().zip(&y).all(|(p, q)| (p - q).abs() <= 1e-3 * scale)
}

/// One element's little-endian bytes, in the splat's own dtype.
fn splat_bytes(s: fusor2_ir::dtype::Splat) -> Vec<u8> {
    use fusor2_ir::dtype::Splat;
    match s {
        Splat::F32(v) => v.to_le_bytes().to_vec(),
        Splat::F16(v) | Splat::BF16(v) => v.to_le_bytes().to_vec(),
        Splat::U32(v) => v.to_le_bytes().to_vec(),
        Splat::I32(v) => v.to_le_bytes().to_vec(),
    }
}

/// Restate a device buffer layout over a reader's shape.
///
/// The layout is stated against the selected member's own shape — a contract
/// says `[batch, m_pad, n_pad]` — while the reader may hold a reshaped
/// spelling of the same value, `[b, h, q, d]`. Same bytes, different
/// coordinates. Each layout axis is either
///
/// - **unpadded** (a run of reader dims multiplies to exactly its extent):
///   the run splits it row-major, so `[b, h]` over an extent-`b*h` batch axis
///   takes strides `[s*h, s]`; or
/// - **padded** (no run can reach the extent): the one next reader dim maps
///   alone at the axis's stride and reads the unpadded prefix, which is what
///   `m = 3` inside `m_pad = 16` means.
///
/// `None` when the shapes do not factor this way — the caller turns that
/// into an error rather than a dense read, because reading a padded buffer
/// densely returns padding as data.
fn restate_layout(
    layout: &fusor2_ir::shape::Layout,
    shape: &[Dim],
    graph: &GraphRef,
) -> Option<fusor2_ir::shape::Layout> {
    if layout.shape() == shape {
        return Some(layout.clone());
    }
    let l_ext: Vec<u64> = layout
        .shape()
        .iter()
        .map(|d| resolve_dim(*d, graph).ok())
        .collect::<Option<_>>()?;
    let l_str: Vec<u64> = layout
        .strides()
        .iter()
        .map(|d| resolve_dim(*d, graph).ok())
        .collect::<Option<_>>()?;
    let r_ext: Vec<u64> = shape
        .iter()
        .map(|d| resolve_dim(*d, graph).ok())
        .collect::<Option<_>>()?;

    // Backtracking assignment. An exact run and a padded singleton can both
    // look viable locally — `[2, 2, 4, 4]` over `[4, 16, 16]` has `4 * 4`
    // exactly filling the padded 16 — and only the remainder decides which
    // reading was right, so a greedy walk mis-factors exactly the shapes a
    // backward pass produces.
    fn assign(
        l_ext: &[u64],
        l_str: &[u64],
        r_ext: &[u64],
        strides: &mut Vec<u64>,
    ) -> bool {
        let Some((&ext, l_rest)) = l_ext.split_first() else {
            // Layout exhausted: only unit reader dims may remain.
            if r_ext.iter().all(|&e| e == 1) {
                strides.extend(std::iter::repeat_n(0, r_ext.len()));
                return true;
            }
            return false;
        };
        let (&stride, s_rest) = l_str.split_first().expect("shapes and strides zip");
        if ext == 1 {
            return assign(l_rest, s_rest, r_ext, strides);
        }
        // Exact runs first — every prefix of reader dims whose product is
        // the extent — longest first so a `[a, b]` split is preferred over
        // treating the axis as padded when both parse.
        let mut prod = 1u64;
        let mut end = 0usize;
        while prod < ext && end < r_ext.len() {
            prod = prod.saturating_mul(r_ext[end]);
            end += 1;
        }
        if prod == ext {
            let mark = strides.len();
            let mut inner = stride;
            let mut group = vec![0u64; end];
            for j in (0..end).rev() {
                group[j] = inner;
                inner = inner.saturating_mul(r_ext[j]);
            }
            strides.extend_from_slice(&group);
            if assign(l_rest, s_rest, &r_ext[end..], strides) {
                return true;
            }
            strides.truncate(mark);
        }
        // Padded singleton: one reader dim strictly inside the extent.
        if let Some((&first, r_rest)) = r_ext.split_first() {
            if first <= ext {
                let mark = strides.len();
                strides.push(stride);
                if assign(l_rest, s_rest, r_rest, strides) {
                    return true;
                }
                strides.truncate(mark);
            }
        }
        false
    }

    let mut strides: Vec<u64> = Vec::with_capacity(r_ext.len());
    if !assign(&l_ext, &l_str, &r_ext, &mut strides) {
        return None;
    }
    fusor2_ir::shape::Layout::from_parts(
        layout.offset(),
        shape,
        &strides.iter().map(|&s| Dim::Const(s)).collect::<Vec<_>>(),
    )
    .ok()
}

fn resolve_dim(d: Dim, graph: &GraphRef) -> Result<u64> {
    match d {
        Dim::Const(v) => Ok(v),
        Dim::Sym(s) => graph
            .dim_binding(s)
            .ok_or_else(|| Error::Plan(format!("dim {s} is unbound at dispatch"))),
    }
}

fn resolve_elements(shape: &[Dim], graph: &GraphRef) -> Result<u64> {
    let mut acc = 1u64;
    for d in shape {
        acc = acc.saturating_mul(resolve_dim(*d, graph)?);
    }
    Ok(acc.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cpu_session_registers_every_macro_op_in_table_order() {
        let s = Session::new(Device::cpu().unwrap()).unwrap();
        for op in crate::composite::MacroOp::ALL {
            assert_eq!(s.registry().get(op.def_id()).unwrap().name, op.name());
        }
    }

    #[test]
    fn the_rule_table_is_the_union_of_every_contributor() {
        let s = Session::new(Device::cpu().unwrap()).unwrap();
        let expected = CORE_RULES.len()
            + SCHED_RULES.len()
            + fusor2_autograd::ADJOINT_RULES.len()
            + fusor2_cpu::CPU_RULES.len();
        assert_eq!(s.rules().len(), expected);
    }

    #[test]
    fn back_pressure_is_a_library_policy() {
        assert_eq!(MAX_INFLIGHT_PLANS, 8);
        let s = Session::new(Device::cpu().unwrap()).unwrap();
        assert_eq!(s.launch_count(), 0);
        s.wait().unwrap();
    }
}
