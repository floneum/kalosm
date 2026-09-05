//! `Session` and `Backend`. The session owns the target, the cost model, the
//! extractor and the plan cache; `resolve` is the one place saturation,
//! extraction and dispatch happen.

#[cfg(feature = "cpu")]
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Instant;

use fusor_cost::tune_cache::Verdict;
use fusor_cost::{LocalSearch, ReplayMemo, Roofline};
#[cfg(feature = "cpu")]
use fusor_cpu::CpuTarget;
#[cfg(feature = "gpu")]
use fusor_gpu::GpuTarget;
use fusor_ir::CORE_RULES;
use fusor_ir::cost::CostModel;
use fusor_ir::device::Caps;
use fusor_ir::dtype::{Dtype, Persistence};
use fusor_ir::egraph::{EGraph, Id, Rule, Saturate, SaturationBudget, SaturationDelta};
use fusor_ir::extract::{ExtractBudget, Extractor, Plan, ReplayKey};
use fusor_ir::ir::launch::Effect;
#[cfg(feature = "cpu")]
use fusor_ir::ir::launch::Launch;
#[cfg(feature = "cpu")]
use fusor_ir::ir::logical::BufferId;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::ir::{Op, OpDefRegistry, Semantics};
use fusor_ir::saturate::Driver;
use fusor_ir::shape::Dim;
#[cfg(feature = "cpu")]
use fusor_ir::target::{Artifact, LowerCtx, Uniforms};
use fusor_ir::target::{Buf, Target};
use fusor_tile::{Planner, SCHED_RULES};
use rustc_hash::FxHashMap;

use crate::composite::register_macro_ops;
use crate::graph::GraphRef;
#[cfg(feature = "cpu")]
use crate::graph::WeakGraphRef;
use crate::tensor::Tensor;
use crate::{Error, Result};

#[cfg(feature = "gpu")]
mod explore;

/// Submitted-but-unretired plans the session will let pile up before it
/// blocks in `resolve`.
const MAX_INFLIGHT_PLANS: u32 = 8;

/// Contractions below this never pay for a measurement round. Override with
/// `FUSOR_AUTOTUNE_MIN_MACS`; `0` tunes everything.
const AUTOTUNE_MIN_MACS: u64 = 64 << 20;
/// Timed repeats per candidate, min taken.
const TUNE_RUNS: usize = 4;
/// How much better a candidate must be to displace the incumbent. Wide enough
/// that run-to-run noise cannot drive an adoption.
///
/// Measured on the launch's own kernel span wherever a device timer exists,
/// and on the whole plan only when one does not.
const TUNE_MARGIN: f64 = 0.08;

/// Class members the tune race has caught computing wrong values, process
/// wide. Every entry is a live miscompile: a member of some e-class whose
/// value disagrees with its siblings'. The conformance harness races every
/// class member (`FUSOR_VERIFY_MEMBERS`) and fails the run when this is
/// nonzero.
static WRONG_MEMBERS: AtomicU64 = AtomicU64::new(0);

/// Number of live member-verification failures observed by this process.
pub fn wrong_member_count() -> u64 {
    WRONG_MEMBERS.load(Ordering::Relaxed)
}

/// Proof that the holder owns a graph's `resolve_lock`.
pub(crate) type ResolveGuard<'a> = parking_lot::MutexGuard<'a, ()>;

/// Which backend a session runs on.
#[derive(Clone)]
pub enum Backend {
    /// Native CPU execution.
    #[cfg(feature = "cpu")]
    Cpu(Arc<CpuTarget>),
    /// WebGPU execution.
    #[cfg(feature = "gpu")]
    Gpu(Arc<GpuTarget>),
}

impl Backend {
    /// Create a CPU backend.
    #[cfg(feature = "cpu")]
    pub fn cpu() -> Result<Self> {
        Ok(Self::Cpu(Arc::new(CpuTarget::new()?)))
    }

    /// Create a GPU backend asynchronously.
    #[cfg(feature = "gpu")]
    pub async fn gpu() -> Result<Self> {
        Ok(Self::Gpu(Arc::new(GpuTarget::new().await?)))
    }

    /// Create a GPU backend and block until adapter initialization completes.
    #[cfg(feature = "gpu")]
    pub fn gpu_blocking() -> Result<Self> {
        Ok(Self::Gpu(Arc::new(GpuTarget::new_blocking()?)))
    }

    /// Whether this is the CPU backend.
    pub fn is_cpu(&self) -> bool {
        #[cfg(feature = "cpu")]
        return matches!(self, Self::Cpu(_));
        #[cfg(not(feature = "cpu"))]
        false
    }

    /// Whether this is the GPU backend.
    pub fn is_gpu(&self) -> bool {
        #[cfg(feature = "gpu")]
        return matches!(self, Self::Gpu(_));
        #[cfg(not(feature = "gpu"))]
        false
    }

    /// The GPU target, when this is the GPU backend.
    #[cfg(feature = "gpu")]
    pub fn gpu_target(&self) -> Option<&GpuTarget> {
        match self {
            Self::Gpu(t) => Some(t.as_ref()),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => None,
        }
    }

    /// The backend's compiler and execution target.
    pub fn target(&self) -> Arc<dyn Target> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(t) => Arc::clone(t) as Arc<dyn Target>,
            #[cfg(feature = "gpu")]
            Self::Gpu(t) => Arc::clone(t) as Arc<dyn Target>,
        }
    }

    /// The reason the GPU device was lost, once the driver has reported it.
    /// Always `None` on the CPU.
    pub fn device_lost(&self) -> Option<String> {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => None,
            #[cfg(feature = "gpu")]
            Self::Gpu(t) => t.device().lost().reason(),
        }
    }

    /// The backend name, either `"cpu"` or `"gpu"`.
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => "cpu",
            #[cfg(feature = "gpu")]
            Self::Gpu(_) => "gpu",
        }
    }

    /// A snapshot of the backend capabilities used for planning.
    pub fn caps(&self) -> Caps {
        match self {
            #[cfg(feature = "cpu")]
            Self::Cpu(t) => t.caps().clone(),
            #[cfg(feature = "gpu")]
            Self::Gpu(t) => t.caps().clone(),
        }
    }

    /// Upload host bytes into a fresh device buffer.
    pub(crate) fn upload(&self, bytes: &[u8], persistence: Persistence) -> Result<Buf> {
        // Only the CPU allocator distinguishes persistence classes.
        #[cfg(not(feature = "cpu"))]
        let _ = persistence;
        match self {
            #[cfg(feature = "gpu")]
            Self::Gpu(t) => t
                .pool()
                .create_buffer_init(bytes, fusor_gpu::pool::TENSOR_USAGE),
            #[cfg(feature = "cpu")]
            Self::Cpu(t) => {
                let buf = t.alloc(bytes.len().max(4) as u64, persistence)?;
                let aligned = buf
                    .downcast_ref::<fusor_cpu::AlignedBuf>()
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
            #[cfg(feature = "gpu")]
            Self::Gpu(t) => pollster::block_on(t.readback(buf, bytes)),
            #[cfg(feature = "cpu")]
            Self::Cpu(_) => {
                let aligned = buf
                    .downcast_ref::<fusor_cpu::AlignedBuf>()
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

pub(crate) struct SessionInner {
    pub device: Backend,
    pub cost: Arc<dyn CostModel>,
    pub extractor: Arc<dyn Extractor>,
    semantics: Arc<dyn Semantics>,
    rules: Vec<Rule>,
    replay: ReplayMemo,
    /// Recorded saturations. A tier below `replay`: it removes the work
    /// that produces the graph a plan is extracted from, where `replay`
    /// removes the extraction itself.
    saturation: SaturationMemo,
    /// What this machine has already learned about which kernels are cheap,
    /// persisted per caps fingerprint.
    tune: fusor_cost::tune_cache::TuneCache,
    /// The online explorer's per-key state: deterministic resolve counters
    /// and the candidate arms production sampling is working through.
    #[cfg(feature = "gpu")]
    explore: parking_lot::Mutex<explore::ExploreState>,
    /// Verified CPU plans and native launch lists, keyed by an unsaturated
    /// term modulo fresh step-buffer names.
    #[cfg(feature = "cpu")]
    cpu_structural: parking_lot::Mutex<FxHashMap<CpuStructuralKey, CpuStructuralEntry>>,
    launches: AtomicU64,
    in_flight: AtomicU32,
}

#[cfg(feature = "cpu")]
struct CpuExecutableLaunch {
    artifact: Artifact,
    grid: [u32; 3],
    bindings: Vec<Id>,
}

#[cfg(feature = "cpu")]
struct CpuExecutable {
    launches: Vec<CpuExecutableLaunch>,
}

#[cfg(feature = "cpu")]
#[derive(Clone, PartialEq, Eq, Hash)]
struct CpuTerm {
    nodes: Vec<Op>,
    roots: Vec<Id>,
}

#[cfg(feature = "cpu")]
#[derive(Clone, PartialEq, Eq, Hash)]
struct CpuStructuralKey {
    graph: usize,
    dim_values: u64,
    term: CpuTerm,
}

#[cfg(feature = "cpu")]
struct CpuStructuralEntry {
    graph: WeakGraphRef,
    roots: Vec<Id>,
    inputs: Vec<Id>,
    plan: Arc<Plan>,
    executable: Arc<CpuExecutable>,
}

#[cfg(feature = "cpu")]
struct CpuStructuralHit {
    roots: Vec<Id>,
    plan: Arc<Plan>,
    executable: Arc<CpuExecutable>,
    inputs: FxHashMap<Id, Buf>,
}

#[cfg(feature = "cpu")]
struct CpuTemplateBindings {
    inputs: FxHashMap<Id, Buf>,
    outputs: Vec<Id>,
    executable: Arc<CpuExecutable>,
}

/// Uninhabited without the `cpu` backend: `run` still names the type, and a
/// GPU resolve never has one to pass.
#[cfg(not(feature = "cpu"))]
enum CpuExecutable {}

/// Uninhabited without the `cpu` backend; see [`CpuExecutable`].
#[cfg(not(feature = "cpu"))]
enum CpuTemplateBindings {}

impl Session {
    /// Create a planner, compiler, and execution session for `device`.
    pub fn new(device: Backend) -> Result<Self> {
        let planner = Planner::shared();
        let device_fingerprint = device.caps().fingerprint();

        // The one registration point. Ids follow table order because
        // `PlanHash` reads registration order.
        let mut registry = OpDefRegistry::new();
        register_macro_ops(&mut registry);

        let semantics =
            fusor_ir::CoreSemantics::with_registry(Arc::clone(&planner), registry.clone());
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
        rules.extend_from_slice(fusor_autograd::ADJOINT_RULES);
        rules.extend_from_slice(target.rules());

        Ok(Self {
            inner: Arc::new(SessionInner {
                device,
                cost,
                extractor,
                semantics,
                rules,
                replay: ReplayMemo::new(),
                tune: fusor_cost::tune_cache::TuneCache::load(device_fingerprint),
                #[cfg(feature = "gpu")]
                explore: parking_lot::Mutex::new(explore::ExploreState::default()),
                saturation: SaturationMemo::default(),
                #[cfg(feature = "cpu")]
                cpu_structural: parking_lot::Mutex::new(FxHashMap::default()),
                launches: AtomicU64::new(0),
                in_flight: AtomicU32::new(0),
            }),
        })
    }

    /// The selected backend.
    pub fn device(&self) -> &Backend {
        &self.inner.device
    }

    /// A snapshot of the backend capabilities used for planning.
    pub fn caps(&self) -> Caps {
        self.inner.device.caps()
    }

    pub(crate) fn semantics(&self) -> Arc<dyn Semantics> {
        Arc::clone(&self.inner.semantics)
    }

    /// Saturate, extract, lower, emit and dispatch everything `values` needs.
    ///
    /// Atomic against every other resolve and readback on the same graph; the
    /// e-graph mutex alone cannot cover dispatch followed by readback.
    pub fn resolve(&self, values: &[Tensor]) -> Result<()> {
        let Some(first) = values.first() else {
            return Ok(());
        };
        let graph = first.graph.clone();
        let resolving = graph.state().resolve_lock.lock();
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
            if !GraphRef::ptr_eq(&v.graph, &graph) {
                return Err(Error::Device(
                    "operands come from two different graphs".into(),
                ));
            }
        }

        // Every requested value already has a device buffer: nothing to plan.
        if values.iter().all(|v| graph.device_buf(v.id).is_some()) {
            return Ok(());
        }

        if self.inner.in_flight.load(Ordering::Relaxed) >= MAX_INFLIGHT_PLANS {
            self.wait()?;
        }

        // Inference frontends commonly rebuild the same expression with a
        // fresh `from_slice` leaf on every call. On CPU, reuse the already
        // verified plan and compiled executable for an isomorphic raw term,
        // rebinding only the fresh input and requested output buffers. This
        // deliberately runs before saturation and extraction: those dominate
        // the cost of replaying a model on an append-only graph.
        #[cfg(feature = "cpu")]
        if self.inner.device.is_cpu()
            && let Some(hit) = self.cpu_structural_hit(&graph, values)
        {
            let old_values: Vec<Tensor> = hit.roots.iter().map(|id| graph.tensor(*id)).collect();
            let bindings = CpuTemplateBindings {
                inputs: hit.inputs,
                outputs: values.iter().map(|value| value.id).collect(),
                executable: hit.executable,
            };
            let started = Instant::now();
            let (launched, _) = self.run(&graph, &hit.plan, &old_values, Some(&bindings))?;
            if resolve_profile() {
                eprintln!(
                    "[profile] structural replay hit: run {} us ({} launches)",
                    started.elapsed().as_micros(),
                    launched
                );
            }
            self.inner
                .launches
                .fetch_add(launched as u64, Ordering::Relaxed);
            self.inner.in_flight.fetch_add(1, Ordering::Relaxed);
            return Ok(());
        }

        let caps = self.caps();
        // The key discriminates on which symbols are bound, never on their
        // values: one plan serves the whole shape family, and the values
        // reach the dispatch through the uniform block and `grid_for`.
        let binding: Vec<Dim> = graph
            .dim_bindings()
            .into_iter()
            .map(|(s, _)| Dim::Sym(s))
            .collect();

        let (plan, roots, key, missed) = {
            let mut g = graph.state().egraph.lock();
            // The root set is per-resolve: planning covers exactly the values
            // this call requested.
            g.clear_roots();
            for v in values {
                g.add_root(v.id);
            }
            let __t_sat = Instant::now();
            let __pre_nodes = g.len();
            // Saturation is a pure function of `(graph, caps, rules, budget)`
            // and a `Session` fixes the last three for its whole life, so a
            // graph in a pre-state seen before saturates to a graph seen
            // before. The memo's validity check is an exact node-by-node
            // comparison, so a miss is slow and never wrong. Recording clones
            // the whole node/facts/parent tables, so model-scale graphs skip
            // the memo entirely.
            const SATURATION_MEMO_MAX_NODES: usize = 50_000;
            // A graph whose node count is unchanged since its last completed
            // saturation is exactly the graph saturation last ran on
            // (`add` is the only structural mutation), so there is nothing
            // to do.
            let __skipped = g.saturated_at_len == Some(g.len());
            let memo_eligible = !__skipped && g.len() <= SATURATION_MEMO_MAX_NODES;
            let __replayed = memo_eligible && self.inner.saturation.replay(&mut g);
            if !__skipped && !__replayed {
                let pre = memo_eligible.then(|| g.pre_saturation());
                // A flat `max_applications` exhausts mid-walk on a model-scale
                // graph, leaving nodes past the exhaustion point without their
                // `lower_*` kernel members; scale the ceiling with the
                // pre-saturation node count.
                let mut budget = SaturationBudget::default();
                budget.max_applications = budget
                    .max_applications
                    .max((g.len() as u32).saturating_mul(16));
                Driver::new().saturate(&mut g, &caps, &self.inner.rules, budget)?;
                // The frontier below `len` is the driver's own exhaustion
                // signal: double and continue until every node has been
                // offered every rule. Bounded, so a genuinely exploding
                // graph still terminates.
                for _ in 0..8 {
                    if g.saturation_frontier >= g.len() {
                        break;
                    }
                    budget.max_applications = budget.max_applications.saturating_mul(2);
                    Driver::new().saturate(&mut g, &caps, &self.inner.rules, budget)?;
                }
                if let Some(pre) = pre {
                    self.inner.saturation.insert(g.record_saturation(pre));
                }
            }
            g.saturated_at_len = Some(g.len());
            let __sat_us = __t_sat.elapsed().as_micros();

            let roots: Vec<Id> = g.roots().to_vec();
            let __t_rest = Instant::now();
            let l0_term = match &g.l0_term_memo {
                Some((r, len, hash)) if *len == g.len() && r == &roots => *hash,
                _ => {
                    let hash = fusor_cost::replay::l0_term_hash(&g, &roots);
                    g.l0_term_memo = Some((roots.clone(), g.len(), hash));
                    hash
                }
            };
            let key = ReplayKey {
                l0_term,
                device: self.inner.cost.facts().fingerprint(),
                binding: fusor_cost::replay::binding_hash(&binding),
            };
            // Tuning happens on a memo miss; the winner is what every later
            // resolve of this key replays.
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
            let __t_verify = Instant::now();
            // `verify_plan` is a pure function of `(key, plan)`, so the same
            // verdict is not re-derived every dispatch. A plan reaching
            // Kernel for the first time is always verified, and an entry
            // replaced by a tuning winner is verified on its own hash.
            if !self.inner.replay.is_verified(key, plan.hash) {
                self.inner.extractor.verify_plan(graph_ref, &plan)?;
                self.inner.replay.mark_verified(key, plan.hash);
            }
            let __verify_us = __t_verify.elapsed().as_micros();
            if resolve_profile() {
                eprintln!(
                    "[profile] saturate{} {} us ({} -> {} nodes), extract+verify {} us (verify {__verify_us}), replay {}",
                    if __skipped {
                        " (skipped)"
                    } else if __replayed {
                        " (replayed)"
                    } else {
                        ""
                    },
                    __sat_us,
                    __pre_nodes,
                    g.len(),
                    __t_rest.elapsed().as_micros(),
                    if missed { "MISS" } else { "hit" },
                );
            }
            (plan, roots, key, missed)
        };

        let plan = if missed && self.inner.device.is_gpu() {
            let tuned = self.autotune(resolving, &graph, &roots, plan, values)?;
            self.inner.replay.insert(key, (*tuned).clone());
            tuned
        } else {
            plan
        };

        // Online tuning: on a replay hit, occasionally substitute one legal
        // arm for the incumbent and let this production dispatch's own GPU
        // spans feed the tuner's windows. Every arm is a verify_plan-checked
        // member plan.
        #[cfg(feature = "gpu")]
        let explored = if !missed && self.inner.device.is_gpu() {
            self.explore_step(&graph, &roots, key, &plan)
        } else {
            None
        };
        #[cfg(feature = "gpu")]
        let (plan, _explore_clock) = match &explored {
            Some(sel) => (
                Arc::clone(sel.plan()),
                Some(TuningClock::new(&self.inner.device)),
            ),
            None => (plan, None),
        };

        // Dumps the launch and incumbent signatures of the plan that actually
        // executes when `FUSOR_DUMP_EXEC` is set, once per distinct plan
        // hash.
        if dump_exec() {
            use std::collections::HashSet;
            use std::sync::{Mutex as StdMutex, OnceLock};
            static SEEN: OnceLock<StdMutex<HashSet<u128>>> = OnceLock::new();
            let seen = SEEN.get_or_init(|| StdMutex::new(HashSet::new()));
            if seen.lock().unwrap().insert(plan.hash.0) {
                let g = graph.state().egraph.lock();
                eprintln!(
                    "EXEC plan hash={:x} launches={}",
                    plan.hash.0,
                    plan.launches.len()
                );
                for ix in 0..plan.launches.len() {
                    eprintln!(
                        "  E{ix}: {} :: {}",
                        fusor_cost::extract::launch_signature(&g, &plan.launches[ix]),
                        fusor_cost::extract::incumbent_signature(&g, &plan, ix)
                            .unwrap_or_else(|| "base".to_string()),
                    );
                }
            }
        }

        let __t_run = Instant::now();
        let (launched, executable) = self.run(&graph, &plan, values, None)?;
        #[cfg(feature = "gpu")]
        if let Some(sel) = explored {
            // Reads the profile the armed clock captured; must run before the
            // clock drops (its drop clears the last profile).
            self.explore_record(sel);
        }
        if resolve_profile() {
            eprintln!(
                "[profile] run {} us ({} launches)",
                __t_run.elapsed().as_micros(),
                launched
            );
        }
        self.inner
            .launches
            .fetch_add(launched as u64, Ordering::Relaxed);
        self.inner.in_flight.fetch_add(1, Ordering::Relaxed);
        #[cfg(feature = "cpu")]
        if let Some(executable) = executable {
            self.insert_cpu_structural(&graph, &roots, Arc::clone(&plan), executable);
        }
        #[cfg(not(feature = "cpu"))]
        let _ = executable;
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

    #[cfg(feature = "cpu")]
    fn cpu_structural_hit(&self, graph: &GraphRef, values: &[Tensor]) -> Option<CpuStructuralHit> {
        let roots: Vec<Id> = values.iter().map(|value| value.id).collect();
        let (key, current_inputs) = cpu_cache_key(graph, &roots)?;
        let mut entries = self.inner.cpu_structural.lock();
        entries.retain(|_, entry| entry.graph.strong_count() > 0);
        let entry = entries.get(&key)?;
        if !entry
            .graph
            .upgrade()
            .is_some_and(|owner| GraphRef::ptr_eq(&owner, graph))
        {
            return None;
        }
        let mut inputs = FxHashMap::default();
        for (&old, current) in entry.inputs.iter().zip(current_inputs) {
            if old == current {
                continue;
            }
            let buffer = graph
                .device_buf(current)
                .or_else(|| self.leaf_buffer(graph, current).ok().flatten())?;
            inputs.insert(old, buffer);
        }
        Some(CpuStructuralHit {
            roots: entry.roots.clone(),
            plan: Arc::clone(&entry.plan),
            executable: Arc::clone(&entry.executable),
            inputs,
        })
    }

    #[cfg(feature = "cpu")]
    fn insert_cpu_structural(
        &self,
        graph: &GraphRef,
        roots: &[Id],
        plan: Arc<Plan>,
        executable: Arc<CpuExecutable>,
    ) {
        let Some((key, inputs)) = cpu_cache_key(graph, roots) else {
            return;
        };
        let mut entries = self.inner.cpu_structural.lock();
        entries.retain(|_, entry| entry.graph.strong_count() > 0);
        if entries.len() >= fusor_cost::replay::CAPACITY && !entries.contains_key(&key) {
            entries.clear();
        }
        entries.insert(
            key,
            CpuStructuralEntry {
                graph: GraphRef::downgrade(graph),
                roots: roots.to_vec(),
                inputs,
                plan,
                executable,
            },
        );
    }

    /// Dispatches issued since construction, not encoder submissions.
    pub fn launch_count(&self) -> u64 {
        self.inner.launches.load(Ordering::Relaxed)
    }

    /// Bytes of an already-resolved value.
    ///
    /// `_resolving` is a witness that the caller holds the graph's
    /// `resolve_lock`: downloading a buffer is only meaningful while no other
    /// thread can be part-way through dispatching a plan that writes it.
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
        // No drain here: `download` records its copy on the same queue the
        // plan was just submitted to and wgpu orders submissions, so the copy
        // cannot observe an unfinished dispatch. `in_flight` is back-pressure
        // bookkeeping only, and the download blocks on the device regardless.
        self.inner.in_flight.store(0, Ordering::Relaxed);

        // A selected `Coop`/`Sgemm` geometry pads the output buffer to its
        // tile multiple, so the bytes on the device are not the value's own
        // dense shape. Read the whole padded buffer and gather the value out
        // of it; a dense layout takes the straight path.
        //
        // The registered layout is stated against the selected member's
        // shape, and the id being read may be a reshaped spelling of the same
        // class, so the layout is restated over the reader's shape
        // (`restate_layout`) — never dropped, because a dense read of a
        // padded buffer returns padding zeros as if they were the value.
        // Padding lives in the strides, never in the shape: a padded buffer
        // is detected by its strides departing from the row-major set or a
        // nonzero offset, not by its shape.
        let padded = graph
            .device_layout(id)
            .filter(|l| {
                l.shape() != &facts.shape[..]
                    || !l.offset().known_eq(fusor_ir::shape::Dim::Const(0))
                    || l.strides() != &fusor_ir::shape::Layout::row_major_strides(l.shape())[..]
            })
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
        let strides: Vec<u64> = resolve_strides(&layout, graph)?;
        // The bytes to pull are the layout's address span, not its element
        // count: a restated layout addresses far past `product(shape)`, and a
        // short download gathers zeros for everything past its end.
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
            let flat = base + idx.iter().zip(&strides).map(|(i, s)| i * s).sum::<u64>();
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

    /// The class member this plan selected for `id`.
    ///
    /// The facade holds the `Logical` id the user built; the plan names the
    /// selected member, and those diverge the moment any rewrite fires.
    fn selected(&self, graph: &GraphRef, plan: &Plan, id: Id) -> Id {
        let class = graph.state().egraph.lock().class_of(id);
        plan.extraction.selected(class).unwrap_or(id)
    }

    /// Register `buf` under every id in `id`'s e-class, `Union` spine
    /// included.
    ///
    /// The class — not the member — is the stable identity of a value: which
    /// member wins is an artifact of one extraction that a later resolve may
    /// change. `macro_op` hands the caller a `Union` spine node, so binding
    /// only the selectable members would leave every sugared spelling
    /// unreadable.
    fn bind_class(
        &self,
        graph: &GraphRef,
        id: Id,
        buf: &Buf,
        layout: Option<&fusor_ir::shape::Layout>,
    ) {
        let members = {
            let g = graph.state().egraph.lock();
            g.class_ids(g.class_of(id))
        };
        let layout = layout.cloned().map(Arc::new);
        graph.set_device_buf_class(&members, buf, layout.as_ref());
    }

    fn run(
        &self,
        graph: &GraphRef,
        plan: &Plan,
        values: &[Tensor],
        template: Option<&CpuTemplateBindings>,
    ) -> Result<(usize, Option<Arc<CpuExecutable>>)> {
        // What the extractor selected for each requested value.
        let wanted: Vec<Id> = values
            .iter()
            .map(|v| self.selected(graph, plan, v.id))
            .collect();
        #[cfg(not(feature = "cpu"))]
        let output_aliases: FxHashMap<Id, Id> = {
            let _ = template;
            FxHashMap::default()
        };
        #[cfg(feature = "cpu")]
        let output_aliases: FxHashMap<Id, Id> = template
            .map(|bindings| {
                wanted
                    .iter()
                    .copied()
                    .zip(bindings.outputs.iter().copied())
                    .collect()
            })
            .unwrap_or_default();
        let launch_roots: rustc_hash::FxHashSet<Id> =
            plan.launches.iter().map(|launch| launch.root).collect();
        // Every external leaf the plan reads, uploaded once. A `Persistent`
        // leaf keeps its buffer across resolves. The classification runs once
        // over the distinct bound values under one lock apiece rather than
        // per binding; only the values that are genuinely unbacked reach
        // `leaf_buffer` and upload.
        let mut supplied: FxHashMap<Id, Buf> = FxHashMap::default();
        let mut bound: rustc_hash::FxHashSet<Id> = rustc_hash::FxHashSet::default();
        let mut distinct: Vec<Id> = Vec::with_capacity(plan.launches.len());
        for launch in &plan.launches {
            for binding in &launch.bindings {
                if bound.insert(binding.value) {
                    distinct.push(binding.value);
                }
            }
        }
        for (id, existing) in graph.external_leaf_buffers(&distinct) {
            match existing {
                Some(buf) => {
                    supplied.insert(id, buf);
                }
                None => {
                    if let Some(buf) = self.leaf_buffer(graph, id)? {
                        supplied.insert(id, buf);
                    }
                }
            }
        }
        #[cfg(feature = "cpu")]
        if let Some(template) = template {
            for (id, buf) in &template.inputs {
                supplied.insert(*id, buf.clone());
            }
        }
        // Root outputs are allocated here rather than inside the backend so
        // the handle survives for readback. The root set is built once;
        // rescanning the launch list per buffer would be quadratic.
        let mut to_bind: Vec<(Id, Buf, Option<Arc<fusor_ir::shape::Layout>>)> = Vec::new();
        for buffer in &plan.buffers {
            if let Some(target) = output_aliases.get(&buffer.value).copied() {
                let elements = resolve_buffer_elements(buffer.elements, &buffer.layout, graph)?;
                let bytes = (elements * buffer.dtype.byte_size()).max(4);
                let buf = self
                    .inner
                    .device
                    .target()
                    .alloc(bytes, buffer.persistence)?;
                to_bind.push((target, buf.clone(), Some(Arc::new(buffer.layout.clone()))));
                supplied.insert(buffer.value, buf);
                continue;
            }
            if supplied.contains_key(&buffer.value) {
                continue;
            }
            let is_launch_root = launch_roots.contains(&buffer.value);
            if !is_launch_root && !wanted.contains(&buffer.value) {
                continue;
            }
            let elements = resolve_buffer_elements(buffer.elements, &buffer.layout, graph)?;
            let bytes = (elements * buffer.dtype.byte_size()).max(4);
            #[cfg(feature = "cpu")]
            if self.inner.device.is_cpu()
                && let Some(existing) = graph.device_buf(buffer.value)
                && existing
                    .downcast_ref::<fusor_cpu::AlignedBuf>()
                    .is_some_and(|buf| buf.len() as u64 >= bytes)
            {
                supplied.insert(buffer.value, existing);
                continue;
            }
            let buf = self
                .inner
                .device
                .target()
                .alloc(bytes, buffer.persistence)?;
            to_bind.push((
                buffer.value,
                buf.clone(),
                Some(Arc::new(buffer.layout.clone())),
            ));
            supplied.insert(buffer.value, buf);
        }
        graph.bind_classes(&to_bind);
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
            #[cfg(feature = "gpu")]
            Backend::Gpu(target) => {
                let mut env = fusor_gpu::target::BindingEnv::new();
                for (sym, value) in graph.dim_bindings() {
                    env = env.with_dim(sym, value);
                }
                for (sym, value) in graph.uniform_scalars() {
                    env = env.with_scalar(sym, value);
                }
                for (id, buf) in &supplied {
                    env = env.with_buffer(*id, buf.clone());
                }
                let g = graph.state().egraph.lock();
                target.resolve(plan, &g, &env)?;
                Ok((plan.launches.len(), None))
            }
            #[cfg(feature = "cpu")]
            Backend::Cpu(target) => {
                let target = Arc::clone(target);
                let (launched, executable) = self.run_cpu(
                    target.as_ref(),
                    graph,
                    plan,
                    &mut supplied,
                    template.map(|bindings| Arc::clone(&bindings.executable)),
                )?;
                Ok((launched, Some(executable)))
            }
        }
    }

    /// The generic runner: one `lower -> emit -> launch` per plan launch, in
    /// plan order. The GPU takes `GpuTarget::resolve` instead, which adds the
    /// plan cache, the parallel build cohort and one encoder per resolve.
    #[cfg(feature = "cpu")]
    fn run_cpu(
        &self,
        target: &CpuTarget,
        graph: &GraphRef,
        plan: &Plan,
        supplied: &mut FxHashMap<Id, Buf>,
        cached: Option<Arc<CpuExecutable>>,
    ) -> Result<(usize, Arc<CpuExecutable>)> {
        let uniforms = self.uniforms_for(plan, graph)?;
        let dim_bindings = graph.dim_bindings();

        for buffer in &plan.buffers {
            if supplied.contains_key(&buffer.value) {
                continue;
            }
            let elements = resolve_buffer_elements(buffer.elements, &buffer.layout, graph)?;
            let bytes = (elements * buffer.dtype.byte_size()).max(4);
            if let Some(existing) = graph.device_buf(buffer.value)
                && existing
                    .downcast_ref::<fusor_cpu::AlignedBuf>()
                    .is_some_and(|buf| buf.len() as u64 >= bytes)
            {
                supplied.insert(buffer.value, existing);
                continue;
            }
            supplied.insert(buffer.value, target.alloc(bytes, buffer.persistence)?);
        }

        let executable = if let Some(cached) = cached {
            cached
        } else {
            let g = graph.state().egraph.lock();
            let mut launches = Vec::with_capacity(plan.launches.len());
            for launch in &plan.launches {
                let theta = plan
                    .extraction
                    .theta
                    .get(&launch.root)
                    .copied()
                    .unwrap_or(fusor_ir::ir::launch::SchedPoint::Point);
                let cx = LowerCtx {
                    plan,
                    launch,
                    graph: &g,
                    symbols: &plan.symbols,
                    dim_bindings: &dim_bindings,
                };
                let ir = target.lower(g.node(launch.root), launch.root, theta, &cx)?;
                let artifact = target.emit(&ir)?;
                let mut ordered: Vec<_> = launch.bindings.iter().collect();
                ordered.sort_by_key(|binding| binding.binding);
                launches.push(CpuExecutableLaunch {
                    artifact,
                    grid: ir.grid,
                    bindings: ordered.into_iter().map(|binding| binding.value).collect(),
                });
            }
            Arc::new(CpuExecutable { launches })
        };

        let mut launched = 0usize;
        for launch in &executable.launches {
            let mut binds = Vec::with_capacity(launch.bindings.len());
            for value in &launch.bindings {
                let buf = supplied.get(value).cloned().ok_or_else(|| {
                    Error::Plan(format!("launch binds {value} which nothing allocates"))
                })?;
                binds.push(buf);
            }
            // The kernel's own grid, not the plan's: `Launch::grid` is the
            // cost model's workgroup count; `KernelIr::grid` is what the
            // lowering indexed the body against. When they disagree the
            // kernel silently computes a prefix of its output.
            target.launch(&launch.artifact, launch.grid, &binds, &uniforms)?;
            launched += 1;
        }
        Ok((launched, executable))
    }

    /// Binding 0 for the CPU launcher, indexed by raw `SymId`: a dim symbol
    /// contributes its extent, a scalar symbol its `f32` bits, and the
    /// emitter bitcasts on read. Neither ever enters a kernel's identity.
    #[cfg(feature = "cpu")]
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
            let g = graph.state().egraph.lock();
            match &g.node(id).op {
                Op::Logical(Logical::Leaf(
                    k @ (LeafKind::Const { .. } | LeafKind::Uniform { .. }),
                )) => k.clone(),
                _ => return Ok(None),
            }
        };
        let facts = graph.facts(id);
        let unit = match &leaf {
            LeafKind::Const { value, .. } => splat_bytes(*value),
            LeafKind::Uniform { sym, .. } => splat_bytes(fusor_autograd::tape::splat_of(
                facts.dtype,
                graph.uniform_value(*sym).unwrap_or(0.0),
            )?),
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
    /// The measurement travels with the plan it measured.
    /// The per-dispatch spans the device timed for the last resolve, when it
    /// has a timer and it was armed. Never on the CPU.
    fn take_last_profile(&self) -> Option<Vec<f64>> {
        #[cfg(feature = "gpu")]
        return self
            .inner
            .device
            .gpu_target()
            .and_then(|t| t.launcher().take_last_profile());
        #[cfg(not(feature = "gpu"))]
        None
    }

    fn autotune(
        &self,
        guard: &ResolveGuard<'_>,
        graph: &GraphRef,
        roots: &[Id],
        base: Arc<Plan>,
        values: &[Tensor],
    ) -> Result<Arc<Plan>> {
        // Member verification: race every candidate of every launch so each
        // gets value-checked, but adopt none — a plan that changes under
        // measurement would make suite dispatch counts nondeterministic.
        let verify_members = std::env::var_os("FUSOR_VERIFY_MEMBERS").is_some();
        let min_macs = if verify_members {
            0
        } else {
            autotune_min_macs()
        };
        let log = std::env::var_os("FUSOR_AUTOTUNE_LOG").is_some();

        // Timing a plan re-runs it, and an in-place node makes a re-run
        // destructive, so an impure plan is never raced. It is still tuned:
        // the production explorer substitutes one candidate exactly once, in
        // place of the incumbent's own dispatch.
        {
            let g = graph.state().egraph.lock();
            if base.launches.iter().any(|l| {
                l.members
                    .iter()
                    .any(|m| g.semantics().effect(&g.node(*m).op) != Effect::Pure)
            }) {
                return Ok(base);
            }
        }

        // One probe pass over the base plan. `launch_variants` holds the work
        // gate, so "every launch offered nothing" is "not worth tuning".
        let probe: Vec<Vec<(String, Plan)>> = {
            let g = graph.state().egraph.lock();
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

        // A member sweep is a correctness pass, not a benchmark: its timings
        // are discarded below, so one execution covers each candidate. Normal
        // autotuning keeps the repeated samples and per-dispatch timestamps it
        // needs for stable comparisons.
        let repetitions = if verify_members { 1 } else { TUNE_RUNS };
        #[cfg(feature = "gpu")]
        let _clock = (!verify_members).then(|| TuningClock::new(&self.inner.device));

        let Some(reference) = self.timed_run(guard, graph, &base, values, repetitions)? else {
            return Ok(base);
        };
        // The plan's identity across processes: every launch signature in
        // order. A cached combination is only replayable onto the same plan
        // shape, so this is what it is keyed on.
        let plan_sig: String = {
            let g = graph.state().egraph.lock();
            base.launches
                .iter()
                .map(|l| fusor_cost::extract::launch_signature(&g, l))
                .collect::<Vec<_>>()
                .join(";")
        };
        // What this pass actually adopts, per launch, so the combination can
        // be recorded rather than reassembled from per-launch minima that
        // were never measured together.
        let mut picks: Vec<Option<String>> = vec![None; base.launches.len()];

        let mut best = Arc::clone(&base);
        let base_ns = plan_ns(&reference);
        let mut best_ns = base_ns;
        // The incumbent's own per-launch spans, plan order. A candidate
        // differs from the incumbent at exactly one launch, so the launch's
        // own span — not the sum — is the term `TUNE_MARGIN` belongs on.
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
                let g = graph.state().egraph.lock();
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
            // number of never-tried points, skip variants this device has
            // already ruled out. The signature is structural, so it is the
            // same key the previous process wrote.
            let sig = {
                let g = graph.state().egraph.lock();
                best.launches
                    .get(ix)
                    .map(|l| fusor_cost::extract::launch_signature(&g, l))
            };
            let variants: Vec<(String, Plan)> = match &sig {
                // The member sweep is a coverage tool: every candidate must be
                // built and value-checked, so the cache must not narrow it.
                Some(_) if verify_members => variants,
                Some(sig) => {
                    // Each candidate travels with the cost model's prior for
                    // the plan it denotes: on a cold signature the cache races
                    // only the model's top-`RACE_TOP_K` picks.
                    let names: Vec<(String, u64)> = variants
                        .iter()
                        .map(|(n, p)| (n.clone(), p.cost.0))
                        .collect();
                    // The cache orders and prunes; it never replaces the race.
                    // Every candidate it hands back is still built, timed and
                    // value-checked, and a recorded combination is
                    // authoritative only once every candidate for the launch
                    // has been measured.
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
                let sample = match self.timed_run(guard, graph, &candidate, values, repetitions) {
                    Ok(Some(sample)) => sample,
                    // A candidate this device cannot build or read is not a
                    // wrong answer; it is skipped. A candidate that took the
                    // device down is a different matter: nothing after it
                    // can run, and the name of the kernel is the whole
                    // diagnosis.
                    outcome => {
                        if let Some(reason) = self.inner.device.device_lost() {
                            let detail = match outcome {
                                Err(e) => format!(" ({e})"),
                                Ok(None) => String::new(),
                                Ok(Some(_)) => unreachable!(),
                            };
                            return Err(Error::Device(format!(
                                "the device was lost while racing candidate `{label}` of \
                                 launch {ix}: {reason}{detail}"
                            )));
                        }
                        continue;
                    }
                };
                let sample_ns = plan_ns(&sample);
                // A different tile is a different reduction order, so bit
                // equality is the wrong test — but a wrong kernel is off by
                // orders of magnitude. This runs before the cache write: a
                // whole-plan disagreement is this variant's.
                let ok = reference.bytes.len() == sample.bytes.len()
                    && reference
                        .bytes
                        .iter()
                        .zip(&sample.bytes)
                        .all(|((dt, a), (_, b))| agrees(*dt, a, b));
                // `replan` rebuilds the whole plan from a one-node edit, so a
                // per-launch quantity may be attributed to index `ix` only
                // when every other launch is untouched. Equal roots are not
                // that guarantee: a replan can move a fused member across a
                // launch boundary while both roots and the launch count stay
                // put, leaving `gpu_us[ix]` small because the launch shed
                // work onto its neighbour.
                let aligned = plans_align(&candidate, &best, ix);
                if let Some(sig) = &sig {
                    // `sig` names one launch, so only a property of that
                    // launch may be filed under it: its own kernel span,
                    // hence the `aligned` filter. A wrong answer needs no
                    // such guard — it is not a timing property.
                    let verdict = if !ok {
                        WRONG_MEMBERS.fetch_add(1, Ordering::Relaxed);
                        let detail = reference
                            .bytes
                            .iter()
                            .zip(&sample.bytes)
                            .enumerate()
                            .find_map(|(o, ((dt, a), (_, b)))| {
                                (*dt == Dtype::F32)
                                    .then(|| first_mismatch(a, b).map(|m| (o, m)))
                                    .flatten()
                            });
                        let detail = detail.map_or_else(String::new, |(o, (i, p, q, w))| {
                            format!(" (out {o} elem {i}: incumbent {p} vs {q}, worst |d| {w})")
                        });
                        // A tiny output is worth printing whole: the wrong
                        // pattern (a column, a row tail, a stripe) names the
                        // bug faster than any single element.
                        if let Some(((_, a), (_, b))) =
                            reference.bytes.first().zip(sample.bytes.first())
                            && a.len() <= 128
                            && a.len() % 4 == 0
                        {
                            let f = |s: &[u8]| {
                                s.as_chunks::<4>()
                                    .0
                                    .iter()
                                    .map(|c| f32::from_le_bytes(*c))
                                    .map(|v| format!("{v:.3}"))
                                    .collect::<Vec<_>>()
                                    .join(" ")
                            };
                            eprintln!("[tune]   incumbent: {}", f(a));
                            eprintln!("[tune]   candidate: {}", f(b));
                        }
                        eprintln!(
                            "[tune] MISCOMPILE: candidate `{label}` of launch {ix} computes \
                             different values from the incumbent plan{detail}"
                        );
                        // Two members of one e-class disagreeing on bytes is
                        // a violated compiler invariant. In production the
                        // resolve fails loudly; only the CI member sweep
                        // (`FUSOR_VERIFY_MEMBERS`) records and continues,
                        // so one run can enumerate every such bug.
                        if !verify_members {
                            return Err(Error::Plan(format!(
                                "internal compiler error: class member `{label}` of launch \
                                 {ix} computes different values from its siblings; every \
                                 member of a class must compute the same value. This is a \
                                 compiler bug — reduce and fix the kernel or rule, do not \
                                 route around it."
                            )));
                        }
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
                            None if self.inner.device.is_cpu() => {
                                Some(Verdict::Ran(ratio_ppm(sample_ns, base_ns)))
                            }
                            None => None,
                        }
                    };
                    // The member sweep's spans are measured under contention
                    // at sizes production never tunes. A `Wrong` verdict is a
                    // property of the kernel and is kept; a `Ran` time is a
                    // property of the sweep and is dropped.
                    let keep = !verify_members || matches!(verdict, Some(Verdict::Wrong));
                    if let Some(verdict) = verdict.filter(|_| keep) {
                        self.inner.tune.record(sig, &label, verdict);
                    }
                }
                if log {
                    eprintln!(
                        "[tune]   L{ix} {sample_ns:.0} ns  (own {} ns, {} ns wall)  {label}{}{}",
                        launch_ns(&sample, ix).map_or_else(|| "-".to_string(), |ns| ns.to_string()),
                        sample.nanos,
                        if ok { "" } else { "  REJECTED: wrong values" },
                        if aligned {
                            ""
                        } else {
                            "  PERTURBED: >1 launch differs"
                        }
                    );
                }
                // Only launch `ix` differs, so the launch's own span is what
                // `TUNE_MARGIN` applies to; on the plan sum, a launch under
                // the margin fraction of the total could never be adopted.
                // Same quantity the cache records.
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
                if ok && improved && !verify_members {
                    best_ns = sample_ns;
                    // The adopted candidate is the new incumbent, so its
                    // profile serves every later launch's comparison.
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
        // launch, so it stays a ratio against this pass's own base, which is
        // what makes it comparable across runs.
        let combo_score = ratio_ppm(best_ns, base_ns);
        if !verify_members {
            self.inner.tune.record_combo(&plan_sig, picks, combo_score);
        }
        // One write per tuning pass, atomic, and a no-op when nothing new was
        // measured — a fully-learned shape costs zero IO.
        self.inner.tune.save();
        if log {
            eprintln!("[tune] winner {best_ns:.0} ns");
        }
        Ok(best)
    }

    /// Run `plan` `repetitions` times, keep the fastest, and read every
    /// requested value back. `None` when nothing was readable.
    fn timed_run(
        &self,
        guard: &ResolveGuard<'_>,
        graph: &GraphRef,
        plan: &Plan,
        values: &[Tensor],
        repetitions: usize,
    ) -> Result<Option<TuneSample>> {
        let mut nanos = u64::MAX;
        let mut gpu_us: Option<Vec<f64>> = None;
        for _ in 0..repetitions {
            let t = Instant::now();
            let _ = self.run(graph, plan, values, None)?;
            self.inner.device.target().wait()?;
            self.inner.in_flight.store(0, Ordering::Relaxed);
            nanos = nanos.min(t.elapsed().as_nanos() as u64);
            if let Some(us) = self.take_last_profile() {
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
            let g = graph.state().egraph.lock();
            matches!(
                &g.node(id).op,
                Op::Logical(Logical::Leaf(
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

/// The launch-work gate below which nothing is measured, env-overridable.
/// Shared by the cold race and the online explorer so "worth tuning" means
/// one thing.
pub(crate) fn autotune_min_macs() -> u64 {
    std::env::var("FUSOR_AUTOTUNE_MIN_MACS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(AUTOTUNE_MIN_MACS)
}

/// Whether `FUSOR_RESOLVE_PROFILE` is set. Read once: `resolve` is the hot
/// path and an env lookup per call is a per-resolve allocation.
fn resolve_profile() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR_RESOLVE_PROFILE").is_some())
}

/// Whether `FUSOR_DUMP_EXEC` is set. Prints the launch and incumbent
/// signatures of each distinct executed plan, so a per-dispatch span profile
/// can be joined to the kernel that actually ran.
fn dump_exec() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR_DUMP_EXEC").is_some())
}

/// Canonicalize an unsaturated term while replacing each runtime buffer name
/// with its traversal-order input slot. The resulting ordinary Op vector
/// provides exact equality and hashing without a parallel hand-written matcher.
#[cfg(feature = "cpu")]
fn canonical_cpu_term(graph: &GraphRef, roots: &[Id]) -> Option<(CpuTerm, Vec<Id>)> {
    fn rebuild(op: &Op, source: Id, children: &[Id], inputs: &mut Vec<Id>) -> Option<Op> {
        let child = |index: usize| children.get(index).copied();
        Some(match op {
            Op::Logical(Logical::Leaf(LeafKind::Buffer { dtype, shape, .. })) => {
                let name = BufferId(inputs.len() as u32);
                inputs.push(source);
                Op::Logical(Logical::Leaf(LeafKind::Buffer {
                    name,
                    dtype: *dtype,
                    shape: shape.clone(),
                }))
            }
            Op::Logical(Logical::Leaf(leaf)) => Op::Logical(Logical::Leaf(leaf.clone())),
            Op::Logical(Logical::Map { expr, outs, .. }) => Op::Logical(Logical::Map {
                expr: expr.clone(),
                ins: children.iter().copied().collect(),
                outs: *outs,
            }),
            Op::Logical(Logical::Fold {
                carrier, axis, acc, ..
            }) => Op::Logical(Logical::Fold {
                carrier: carrier.clone(),
                axis: *axis,
                acc: *acc,
                ins: children.iter().copied().collect(),
            }),
            Op::Logical(Logical::Contract {
                spec, acc, outs, ..
            }) => Op::Logical(Logical::Contract {
                spec: spec.clone(),
                acc: *acc,
                a: child(0)?,
                b: child(1)?,
                outs: *outs,
            }),
            Op::Logical(Logical::Restride { specs, bounds, .. }) => {
                Op::Logical(Logical::Restride {
                    specs: specs.clone(),
                    bounds: *bounds,
                    x: child(0)?,
                })
            }
            Op::Logical(Logical::Window { specs, .. }) => Op::Logical(Logical::Window {
                specs: specs.clone(),
                x: child(0)?,
            }),
            Op::Logical(Logical::Gather { axis, .. }) => Op::Logical(Logical::Gather {
                axis: *axis,
                x: child(0)?,
                idx: child(1)?,
            }),
            Op::Logical(Logical::Scatter {
                axis,
                combine,
                unique,
                ..
            }) => Op::Logical(Logical::Scatter {
                axis: *axis,
                combine: *combine,
                base: child(0)?,
                idx: child(1)?,
                upd: child(2)?,
                unique: *unique,
            }),
            Op::Logical(Logical::Dequant { fmt, layout, .. }) => Op::Logical(Logical::Dequant {
                fmt: *fmt,
                layout: *layout,
                x: child(0)?,
            }),
            Op::Logical(Logical::Project { slot, .. }) => Op::Logical(Logical::Project {
                slot: *slot,
                x: child(0)?,
            }),
            Op::Launch(Launch::Ext { def, ops, attrs }) => {
                let mut ops = ops.clone();
                if ops.len() != children.len() {
                    return None;
                }
                for (operand, child) in ops.iter_mut().zip(children) {
                    operand.src = *child;
                }
                Op::Launch(Launch::Ext {
                    def: *def,
                    ops,
                    attrs: *attrs,
                })
            }
            Op::Union(..) => Op::Union(child(0)?, child(1)?),
            Op::Launch(_) => return None,
        })
    }

    fn visit(
        graph: &EGraph,
        id: Id,
        memo: &mut FxHashMap<Id, Id>,
        nodes: &mut Vec<Op>,
        inputs: &mut Vec<Id>,
    ) -> Option<Id> {
        if let Some(id) = memo.get(&id).copied() {
            return Some(id);
        }
        let node = graph.node(id);
        let children = node
            .children
            .iter()
            .map(|child| visit(graph, *child, memo, nodes, inputs))
            .collect::<Option<Vec<_>>>()?;
        let op = rebuild(&node.op, id, &children, inputs)?;
        let canonical = Id(nodes.len() as u32);
        nodes.push(op);
        memo.insert(id, canonical);
        Some(canonical)
    }

    let graph = graph.state().egraph.lock();
    let mut memo = FxHashMap::default();
    let mut nodes = Vec::new();
    let mut inputs = Vec::new();
    let roots = roots
        .iter()
        .map(|root| visit(&graph, *root, &mut memo, &mut nodes, &mut inputs))
        .collect::<Option<Vec<_>>>()?;
    Some((CpuTerm { nodes, roots }, inputs))
}

#[cfg(feature = "cpu")]
fn cpu_cache_key(graph: &GraphRef, roots: &[Id]) -> Option<(CpuStructuralKey, Vec<Id>)> {
    let (term, inputs) = canonical_cpu_term(graph, roots)?;
    let mut dims = rustc_hash::FxHasher::default();
    graph.dim_bindings().hash(&mut dims);
    Some((
        CpuStructuralKey {
            graph: GraphRef::as_ptr(graph) as usize,
            dim_values: dims.finish(),
            term,
        },
        inputs,
    ))
}
fn saturation_memo_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var_os("FUSOR_NO_SAT_MEMO").is_some())
}

/// Recorded saturations, bounded and FIFO-evicted.
///
/// [`EGraph::replay_saturation`] checks the whole pre-state by value, so this
/// is a list to scan; `could_apply_to` rejects on node count and symbol
/// counter before a single node is compared.
#[derive(Default)]
struct SaturationMemo(parking_lot::Mutex<Vec<Arc<SaturationDelta>>>);

/// Recordings kept. Matches `fusor_cost::replay::CAPACITY`, so the two tiers
/// hold the same number of distinct terms.
const SATURATION_MEMO_CAPACITY: usize = 64;

/// Nodes the whole memo may hold. A recording carries a copy of its graph, so
/// the entry count alone is not a memory bound.
const SATURATION_MEMO_NODES: usize = 256 << 10;

impl SaturationMemo {
    /// Replay a recording onto `graph` if one was taken against exactly this
    /// pre-state. `false` means the caller must saturate for real.
    fn replay(&self, graph: &mut EGraph) -> bool {
        if saturation_memo_disabled() {
            return false;
        }
        // Cloned out of the lock: `replay_saturation` needs `&mut EGraph`
        // and the delta at once.
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
        // than the whole node budget is still worth keeping.
        while entries.len() > 1
            && (entries.len() > SATURATION_MEMO_CAPACITY
                || entries
                    .iter()
                    .map(|d| d.prefix() + d.added())
                    .sum::<usize>()
                    > SATURATION_MEMO_NODES)
        {
            entries.remove(0);
        }
    }
}

/// One candidate's measurement and the values it produced, together.
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

/// Whether `candidate` differs from `incumbent` at exactly launch `ix`, so a
/// per-launch quantity may be attributed to that index. Equal roots are not
/// that guarantee: a launch is its root *and* its members, grid and block. A
/// candidate that fails this is compared at whole-plan granularity instead.
fn plans_align(candidate: &Plan, incumbent: &Plan, ix: usize) -> bool {
    candidate.launches.len() == incumbent.launches.len()
        && candidate
            .launches
            .iter()
            .zip(&incumbent.launches)
            .enumerate()
            .all(|(j, (c, b))| {
                j == ix
                    || (c.root == b.root && c.grid == b.grid && c.block == b.block && {
                        // Member *order* is a realization detail —
                        // `launch_signature` sorts for the same reason —
                        // but the member set is the work.
                        let mut cm: Vec<Id> = c.members.to_vec();
                        let mut bm: Vec<Id> = b.members.to_vec();
                        cm.sort_unstable();
                        bm.sort_unstable();
                        cm == bm
                    })
            })
}

/// [`plans_align`] over a *set* of swapped launches: `candidate` must differ
/// from `incumbent` at most at the swapped indices — the check a batched
/// prior adoption holds its composed replan to before trusting per-launch
/// windows measured against single swaps.
#[cfg(feature = "gpu")]
fn batch_aligns(candidate: &Plan, incumbent: &Plan, swaps: &[(usize, String)]) -> bool {
    candidate.launches.len() == incumbent.launches.len()
        && candidate
            .launches
            .iter()
            .zip(&incumbent.launches)
            .enumerate()
            .all(|(j, (c, b))| swaps.iter().any(|(s, _)| *s == j) || launch_key(c) == launch_key(b))
}

/// One launch's identity for plan diffing, hashed: the same fields
/// [`plans_align`] compares — root, grid, block and the member *set* — so two
/// launches with equal keys are the same work on the same schedule. `Id`s are
/// process-local, which is fine: a diff always compares two plans over the
/// same graph.
#[cfg(feature = "gpu")]
fn launch_key(launch: &fusor_ir::extract::Dispatch) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    launch.root.hash(&mut h);
    launch.grid.hash(&mut h);
    launch.block.hash(&mut h);
    let mut members: Vec<Id> = launch.members.to_vec();
    members.sort_unstable();
    members.hash(&mut h);
    h.finish()
}

/// The launches on which `candidate` and `incumbent` disagree, as index sets
/// `(changed_in_candidate, changed_in_incumbent)`, when an alignment with at
/// most `max_d` edits exists (Myers O((N+M)·D)). `None` means the plans are
/// either identical or too different to attribute a window: a candidate that
/// removes a producer launch and re-spells a consumer diffs as (one inserted
/// launch, two deleted launches), and the summed spans of each side's changed
/// set are the honest comparison for a plan too large to time whole.
#[cfg(feature = "gpu")]
fn plan_sparse_diff(
    candidate: &Plan,
    incumbent: &Plan,
    max_d: usize,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let a: Vec<u64> = candidate.launches.iter().map(launch_key).collect();
    let b: Vec<u64> = incumbent.launches.iter().map(launch_key).collect();
    sparse_diff(&a, &b, max_d)
}

/// The alignment itself, over launch keys. Split from [`plan_sparse_diff`]
/// so the path walk is testable without building plans.
#[cfg(feature = "gpu")]
fn sparse_diff(a: &[u64], b: &[u64], max_d: usize) -> Option<(Vec<usize>, Vec<usize>)> {
    let (n, m) = (a.len(), b.len());
    if n.abs_diff(m) > max_d {
        return None;
    }
    // Myers' greedy forward search over edit-distance frontiers. `v[k]` is
    // the furthest `x` on diagonal `k = x - y`; `trace` keeps each frontier
    // so the path can be walked back into the two changed sets.
    let width = 2 * max_d + 1;
    let off = max_d as isize;
    let mut v = vec![0usize; width];
    let mut trace: Vec<Vec<usize>> = Vec::new();
    let mut found_d = None;
    'outer: for d in 0..=max_d {
        let di = d as isize;
        let mut k = -di;
        while k <= di {
            let ki = (k + off) as usize;
            let mut x = if k == -di || (k != di && v[ki - 1] < v[ki + 1]) {
                v[ki + 1] // down: consume one of `b` (deletion from incumbent)
            } else {
                v[ki - 1] + 1 // right: consume one of `a` (insertion)
            };
            let mut y = (x as isize - k) as usize;
            while x < n && y < m && a[x] == b[y] {
                x += 1;
                y += 1;
            }
            v[ki] = x;
            if x >= n && y >= m {
                trace.push(v.clone());
                found_d = Some(d);
                break 'outer;
            }
            k += 2;
        }
        trace.push(v.clone());
    }
    let d_final = found_d?;
    if d_final == 0 {
        return None; // identical plans: nothing to attribute
    }
    // Walk the path back, collecting the non-diagonal steps. Each frontier
    // step is: pre-edit position on the previous frontier, one edit, then a
    // diagonal snake — undone here in reverse order.
    let mut changed_a = Vec::new();
    let mut changed_b = Vec::new();
    let (mut x, mut y) = (n, m);
    for d in (1..=d_final).rev() {
        let vprev = &trace[d - 1];
        let di = d as isize;
        let k = x as isize - y as isize;
        let ki = (k + off) as usize;
        let down = k == -di || (k != di && vprev[ki - 1] < vprev[ki + 1]);
        let prev_k = if down { k + 1 } else { k - 1 };
        // The pre-edit position, on the previous frontier's diagonal.
        let x0 = vprev[(prev_k + off) as usize];
        let y0 = (x0 as isize - prev_k) as usize;
        // Undo the snake, back to just after the edit.
        let (post_x, post_y) = if down { (x0, y0 + 1) } else { (x0 + 1, y0) };
        while x > post_x && y > post_y {
            x -= 1;
            y -= 1;
        }
        // Undo the edit.
        if down {
            changed_b.push(y0);
        } else {
            changed_a.push(x0);
        }
        x = x0;
        y = y0;
    }
    changed_a.reverse();
    changed_b.reverse();
    Some((changed_a, changed_b))
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
#[cfg(feature = "gpu")]
struct TuningClock<'a>(Option<&'a GpuTarget>);

#[cfg(feature = "gpu")]
impl<'a> TuningClock<'a> {
    fn new(device: &'a Backend) -> Self {
        let target = device.gpu_target();
        if let Some(t) = target {
            t.launcher().set_tuning(true);
        }
        Self(target)
    }
}

#[cfg(feature = "gpu")]
impl Drop for TuningClock<'_> {
    fn drop(&mut self) {
        if let Some(t) = self.0 {
            t.launcher().set_tuning(false);
            let _ = t.launcher().take_last_profile();
        }
    }
}

/// The first disagreeing f32 element and the worst one, for the MISCOMPILE
/// report: `(first_index, expected, got, worst_abs_diff)`.
fn first_mismatch(a: &[u8], b: &[u8]) -> Option<(usize, f32, f32, f32)> {
    let f = |s: &[u8]| {
        s.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<f32>>()
    };
    let (x, y) = (f(a), f(b));
    let scale = x.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    let mut first = None;
    let mut worst = 0.0f32;
    for (i, (p, q)) in x.iter().zip(&y).enumerate() {
        let d = (p - q).abs();
        if d > 1e-3 * scale && first.is_none() {
            first = Some((i, *p, *q));
        }
        worst = worst.max(d);
    }
    first.map(|(i, p, q)| (i, p, q, worst))
}

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
        s.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect::<Vec<f32>>()
    };
    let (x, y) = (f(a), f(b));
    let scale = x.iter().fold(1.0f32, |m, v| m.max(v.abs()));
    x.iter().zip(&y).all(|(p, q)| (p - q).abs() <= 1e-3 * scale)
}

/// One element's little-endian bytes, in the splat's own dtype.
fn splat_bytes(s: fusor_ir::dtype::Splat) -> Vec<u8> {
    use fusor_ir::dtype::Splat;
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
    layout: &fusor_ir::shape::Layout,
    shape: &[Dim],
    graph: &GraphRef,
) -> Option<fusor_ir::shape::Layout> {
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
    fn assign(l_ext: &[u64], l_str: &[u64], r_ext: &[u64], strides: &mut Vec<u64>) -> bool {
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
        // Padded run: one or more reader dims whose product fits inside the
        // extent, laid out row-major over the value's own extents and reading
        // the unpadded prefix. Shortest run first; longer runs are only
        // reached once the singleton reading has failed the remainder.
        //
        // A padded axis holds exactly one logical axis, so the run may
        // contain at most one non-unit reader dim: a second real axis nested
        // inside the padded one would read the padding between the logical
        // extent and the block edge as data.
        let mut prod = 1u64;
        let mut non_unit = 0usize;
        for take in 1..=r_ext.len() {
            prod = prod.saturating_mul(r_ext[take - 1]);
            if prod > ext {
                break;
            }
            if r_ext[take - 1] != 1 {
                non_unit += 1;
                if non_unit > 1 {
                    break;
                }
            }
            let mark = strides.len();
            let mut inner = stride;
            let mut group = vec![0u64; take];
            for j in (0..take).rev() {
                group[j] = inner;
                inner = inner.saturating_mul(r_ext[j]);
            }
            strides.extend_from_slice(&group);
            if assign(l_rest, s_rest, &r_ext[take..], strides) {
                return true;
            }
            strides.truncate(mark);
        }
        false
    }

    let mut strides: Vec<u64> = Vec::with_capacity(r_ext.len());
    if !assign(&l_ext, &l_str, &r_ext, &mut strides) {
        return None;
    }
    // The padded parse is ambiguous in principle (only the producer knows its
    // logical extents), so this records which parse won.
    if std::env::var_os("FUSOR_RESTATE_LOG").is_some() {
        eprintln!("[restate] layout={l_ext:?}/{l_str:?} reader={r_ext:?} -> strides={strides:?}");
    }
    fusor_ir::shape::Layout::from_parts(
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

/// The `row_major_strides` placeholder (`SymId(u32::MAX)`): a stride past a
/// symbolic axis, derived at dispatch as the product of the following
/// extents. Mirrors `UniformPack::resolve_stride`.
const DERIVED_STRIDE: fusor_ir::shape::SymId = fusor_ir::shape::SymId(u32::MAX);

/// Concrete strides of a layout at the current binding, deriving any
/// placeholder from the (now concrete) shape.
fn resolve_strides(layout: &fusor_ir::shape::Layout, graph: &GraphRef) -> Result<Vec<u64>> {
    let shape = layout.shape();
    layout
        .strides()
        .iter()
        .enumerate()
        .map(|(axis, d)| match d {
            Dim::Sym(s) if *s == DERIVED_STRIDE => {
                let mut acc = 1u64;
                for e in &shape[axis + 1..] {
                    acc = acc.saturating_mul(resolve_dim(*e, graph)?);
                }
                Ok(acc)
            }
            other => resolve_dim(*other, graph),
        })
        .collect()
}

/// A buffer's element count at the current binding. `BufferPlan::elements`
/// is the placeholder whenever any extent is symbolic; the layout's shape is
/// the authority then.
fn resolve_buffer_elements(
    elements: Dim,
    layout: &fusor_ir::shape::Layout,
    graph: &GraphRef,
) -> Result<u64> {
    match elements {
        Dim::Sym(s) if s == DERIVED_STRIDE => {
            // Padding lives in the strides: for the plan's row-major layouts
            // the buffer extent is `shape[0] * strides[0]`, never the shape
            // product, which undercounts a padded buffer.
            let Some(first) = layout.shape().first() else {
                return Ok(1);
            };
            let strides = resolve_strides(layout, graph)?;
            Ok(resolve_dim(*first, graph)?
                .saturating_mul(strides[0])
                .max(1))
        }
        d => resolve_dim(d, graph),
    }
}

fn resolve_elements(shape: &[Dim], graph: &GraphRef) -> Result<u64> {
    let mut acc = 1u64;
    for d in shape {
        acc = acc.saturating_mul(resolve_dim(*d, graph)?);
    }
    Ok(acc.max(1))
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    #[cfg(feature = "cpu")]
    fn a_fresh_step_leaf_reuses_the_cpu_plan_and_executable() {
        let session = Session::new(Backend::cpu().unwrap()).unwrap();
        let graph = Graph::new(&session);
        let run = |values: &[f32]| {
            let x =
                Tensor::from_elements(graph.handle(), &[Dim::Const(values.len() as u64)], values)
                    .unwrap();
            let y = x.add_scalar(1.0).unwrap();
            let bytes = graph.handle().read_back(y.id).unwrap();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        };

        assert_eq!(run(&[1.0, 2.0, 3.0]), vec![2.0, 3.0, 4.0]);
        assert_eq!(session.inner.cpu_structural.lock().len(), 1);

        assert_eq!(run(&[10.0, 11.0, 12.0]), vec![11.0, 12.0, 13.0]);
        assert_eq!(
            session.inner.cpu_structural.lock().len(),
            1,
            "a replay hit must not extract and record another plan"
        );
    }
}
