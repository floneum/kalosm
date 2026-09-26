//! `Session` and `Backend`. The session owns the target, the cost model, the
//! extractor and the plan cache; `resolve` is the one place saturation,
//! extraction and dispatch happen.

use std::hash::Hash;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use web_time::Instant;

use fusor_cost::{LocalSearch, ReplayMemo, Roofline};
#[cfg(feature = "cpu")]
use fusor_cpu::CpuTarget;
#[cfg(feature = "gpu")]
use fusor_gpu::GpuTarget;
use fusor_ir::CORE_RULES;
use fusor_ir::cost::CostModel;
use fusor_ir::device::Caps;
#[cfg(feature = "compiler-tests")]
use fusor_ir::dtype::Dtype;
use fusor_ir::dtype::Persistence;
use fusor_ir::egraph::{ClassId, EGraph, Id, Rule, Saturate, SaturationBudget, SaturationDelta};
use fusor_ir::extract::{ExtractBudget, Extractor, Plan, ReplayKey};
use fusor_ir::ir::launch::Effect;
use fusor_ir::ir::logical::BufferId;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::ir::{Level, Op, Semantics};
use fusor_ir::saturate::Driver;
use fusor_ir::shape::{Dim, Layout, SymId};
#[cfg(feature = "cpu")]
use fusor_ir::target::{Artifact, LowerCtx, Uniforms};
use fusor_ir::target::{Buf, Target};
use fusor_tile::{Planner, SCHED_RULES};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::graph::GraphRef;
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

#[cfg(feature = "compiler-tests")]
mod testing;
#[cfg(feature = "compiler-tests")]
use testing::{agrees, first_mismatch};
#[cfg(feature = "compiler-tests")]
pub use testing::{set_verify_members, verify_members, wrong_member_count};
#[cfg(all(not(feature = "compiler-tests"), feature = "gpu"))]
fn verify_members() -> bool {
    false
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
    #[cfg(not(target_arch = "wasm32"))]
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

    /// Release the kernels only losing race candidates used; see
    /// `GpuTarget::release_candidates`. The CPU keeps nothing per candidate.
    pub(crate) fn release_candidates(&self, _arena: u64, _candidates: &[Arc<Plan>], _keep: &Plan) {
        match self {
            #[cfg(feature = "gpu")]
            Self::Gpu(t) => t.release_candidates(_arena, _candidates, _keep),
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    /// Release every kernel compiled for graph `arena`; called when the
    /// graph is dropped. See `GpuTarget::release_arena`.
    pub(crate) fn release_arena(&self, _arena: u64) {
        match self {
            #[cfg(feature = "gpu")]
            Self::Gpu(t) => t.release_arena(_arena),
            #[allow(unreachable_patterns)]
            _ => {}
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

    /// Copy a device buffer back to the host. One of exactly three host
    /// syncs, and the one that is awaited: on WebGPU the copy completes only
    /// when the browser's event loop runs, so it cannot be spun on.
    fn copy(&self, buf: &Buf) -> Result<Buf> {
        self.target().copy(buf)
    }

    async fn download(&self, buf: &Buf, bytes: u64) -> Result<Vec<u8>> {
        match self {
            #[cfg(feature = "gpu")]
            Self::Gpu(t) => t.readback(buf, bytes).await,
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
    /// The CPU's compiled launch lists, per plan and per binding (its
    /// lowering folds the symbols' values in). The GPU target keeps its
    /// own pipeline cache.
    #[cfg(feature = "cpu")]
    cpu_executables: parking_lot::Mutex<FxHashMap<(u128, u64), Arc<CpuExecutable>>>,
    /// Shape families: terms equal modulo the constants of their step
    /// buffers and views. A family whose constants vary across calls gets a
    /// symbolic twin term planned once and re-bound per call.
    families: parking_lot::Mutex<FxHashMap<FamilyKey, Family>>,
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

/// The raw (unsaturated) term below a root set, with every step buffer leaf
/// renamed positionally: what a plan is a function of, once buffers are
/// rebound.
#[derive(Clone, PartialEq, Eq, Hash)]
struct CanonicalTerm {
    nodes: Vec<Op>,
    roots: Vec<Id>,
}

/// Uninhabited without the `cpu` backend: the memo still names the type, and
/// a GPU resolve never has one to hold.
#[cfg(not(feature = "cpu"))]
enum CpuExecutable {}

/// A term with the constants of its step-buffer shapes and view specs
/// abstracted into slots: what two calls of one model step have in common
/// when only a sequence length or a view offset moved.
struct FamilyTerm {
    /// The term with every abstracted constant spelled as a slot symbol
    /// (see [`slot_dim`]), step buffers renamed positionally.
    term: CanonicalTerm,
    /// The abstracted constants, slot order.
    consts: Vec<u64>,
    /// The graph node each canonical node came from.
    sources: Vec<Id>,
    /// The step buffers, in the order the term names them.
    inputs: Vec<Id>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct FamilyKey {
    graph: usize,
    term: CanonicalTerm,
}

/// What the session knows about one shape family.
struct Family {
    graph: WeakGraphRef,
    /// The constants of the first call seen.
    consts: Vec<u64>,
    /// Slots that have differed between two calls: the twin's symbols.
    varying: Vec<bool>,
    /// Which symbol each varying slot takes: slots whose values have agreed
    /// in every call seen share one, so a length that a term spells in its
    /// leaf shape, its views and its reshapes stays one extent to the
    /// planner. A slot pair that later disagrees splits and rebuilds.
    group: Vec<usize>,
    /// Members planned concretely, one per distinct constant vector, newest
    /// last: a shape seen again replays its own plan, whose kernels were
    /// selected for its exact extents. Bounded by [`CONCRETE_SHAPES`].
    concrete: Vec<Twin>,
    /// The symbolic twin, built once the family has shown more distinct
    /// shapes than [`CONCRETE_SHAPES`]: from then on every new shape is a
    /// re-binding, not an extraction.
    symbolic: Option<Twin>,
    /// The symbolic twin could not be built or planned (an op that needs the
    /// constant); every new shape of this family plans concretely.
    blocked: bool,
}

/// Root sets the per-graph saturation and closure-hash memos hold before
/// they are cleared: the twins of every live family, with room to spare.
const ROOT_SET_MEMOS: usize = 256;

/// Distinct shapes a family replays concretely before it goes symbolic. A
/// batch of embeddings sees a handful of padded lengths and every one of
/// them deserves the kernels its exact extents select; a decode loop sees a
/// new length every token and would extract forever.
const CONCRETE_SHAPES: usize = 2;

/// The symbolic twin of a family: the same term with the varying slots as
/// symbols, so one plan serves every member. The twin's nodes live in the
/// graph beside the members' and are the ids a replay hits on.
#[derive(Clone)]
struct Twin {
    /// The constant vector this twin is exact for; empty for the symbolic
    /// twin.
    consts: Vec<u64>,
    syms: Vec<Option<SymId>>,
    roots: Vec<Id>,
    /// The twin's step-buffer leaves and the index of the member input each
    /// stands in for: a fresh leaf where the shape carries a symbol, the
    /// building member's own leaf otherwise. Every call rebinds them to its
    /// member's buffers.
    leaves: Vec<(Id, usize)>,
}

/// A family invocation borrows node names without changing their callers' bindings.
struct FamilyBindings<'a> {
    graph: &'a GraphRef,
    values: Vec<Id>,
    saved: FxHashMap<Id, (Buf, Option<Arc<Layout>>)>,
}

impl<'a> FamilyBindings<'a> {
    fn new(graph: &'a GraphRef, roots: &[Id]) -> Result<Self> {
        let mut g = graph.state().egraph.lock();
        let values = logical_postorder(&g, roots)
            .ok_or_else(|| Error::Plan("shape family has no logical program".into()))?;
        let mut saved = FxHashMap::default();
        let mut leaves = graph.state().leaves.lock();
        for value in &values {
            let class = g.class_of(*value);
            let members = g.class_ids_cached(class);
            let external = members
                .iter()
                .any(|id| matches!(g.node(*id).op, Op::Logical(Logical::Leaf(_))));
            for id in members.iter() {
                if let Some(binding) = leaves.device.get(id) {
                    saved.entry(*id).or_insert_with(|| binding.clone());
                }
                if !external {
                    leaves.device.remove(id);
                }
            }
        }
        Ok(Self {
            graph,
            values,
            saved,
        })
    }
}

impl Drop for FamilyBindings<'_> {
    fn drop(&mut self) {
        let mut g = self.graph.state().egraph.lock();
        let mut leaves = self.graph.state().leaves.lock();
        for value in &self.values {
            let class = g.class_of(*value);
            for id in g.class_ids_cached(class).iter() {
                leaves.device.remove(id);
            }
        }
        leaves.device.extend(self.saved.drain());
    }
}

/// Resolve symbolic extents and strides without changing the physical layout.
fn concrete_layout(layout: &Layout, bindings: &FxHashMap<SymId, u64>) -> Result<Layout> {
    let value = |d: Dim| {
        d.evaluate(&mut |s| bindings.get(&s).copied())
            .map(Dim::Const)
            .ok_or_else(|| Error::Plan(format!("unbound layout dimension {d}")))
    };
    let shape = layout
        .shape()
        .iter()
        .map(|d| value(*d))
        .collect::<Result<Vec<_>>>()?;
    let strides = layout
        .strides()
        .iter()
        .map(|d| value(*d))
        .collect::<Result<Vec<_>>>()?;
    Layout::from_parts(value(layout.offset())?, &shape, &strides)
}

/// Family slots are spelled as symbols above any the graph mints and above
/// the derived range ([`fusor_ir::shape::DERIVED_END`]).
const FAMILY_SLOT_BASE: u32 = fusor_ir::shape::DERIVED_END;

fn slot_dim(slot: usize) -> Dim {
    Dim::Sym(SymId(FAMILY_SLOT_BASE + slot as u32))
}

fn slot_of(d: Dim) -> Option<usize> {
    match d {
        // The opaque-dimension sentinel is not a family slot.
        Dim::Sym(s) if s.0 >= FAMILY_SLOT_BASE && s.0 != u32::MAX => {
            Some((s.0 - FAMILY_SLOT_BASE) as usize)
        }
        _ => None,
    }
}

impl Session {
    /// The backend this session runs on, for building another session over
    /// the same device.
    pub fn backend(&self) -> Backend {
        self.inner.device.clone()
    }

    /// Create a planner, compiler, and execution session for `device`.
    pub fn new(device: Backend) -> Result<Self> {
        let planner = Planner::shared();
        let device_fingerprint = device.caps().fingerprint();

        let semantics = fusor_ir::CoreSemantics::new(Arc::clone(&planner));
        let target = device.target();
        let cost: Arc<dyn CostModel> = Arc::new(Roofline::new(target.facts().clone()));
        let extractor: Arc<dyn Extractor> = Arc::new(LocalSearch::new(
            Arc::clone(&planner),
            target.caps().clone(),
        ));

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
                cpu_executables: parking_lot::Mutex::new(FxHashMap::default()),
                families: parking_lot::Mutex::new(FxHashMap::default()),
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

    /// Put an external leaf's host bytes on the device now, bound to the
    /// leaf, without resolving anything that reads it. A no-op once the leaf
    /// has a device buffer. What a weight needs before another leaf can
    /// [`Tensor::adopt_buffer`] it.
    pub fn upload_leaf(&self, leaf: &Tensor) -> Result<()> {
        if leaf.graph.device_buf(leaf.id).is_some() {
            return Ok(());
        }
        let graph = leaf.graph.clone();
        let _resolving = graph.state().resolve_lock.lock();
        if self.leaf_buffer(&graph, leaf.id)?.is_none() {
            return Err(Error::Plan(
                "upload_leaf targets an external leaf; this value is computed".into(),
            ));
        }
        Ok(())
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
        self.resolve_locked_plan(resolving, values).map(|_| ())
    }

    /// [`Self::resolve_locked`], also handing back the plan when the values
    /// were planned and run here (`None` when a memo or a restatement ran
    /// them instead).
    fn resolve_locked_plan(
        &self,
        resolving: &ResolveGuard<'_>,
        values: &[Tensor],
    ) -> Result<Option<Arc<Plan>>> {
        if values.is_empty() {
            return Ok(None);
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
            return Ok(None);
        }

        // Release dead values before allocating a new execution's buffers.
        graph.reap_dead();

        if self.inner.in_flight.load(Ordering::Relaxed) >= MAX_INFLIGHT_PLANS {
            self.wait()?;
        }

        // A value below the roots that already has a device buffer is an
        // input, not something to recompute: a reader of one chunk of a
        // resolved batch must not re-run the whole batch. Planning cannot
        // see bindings, so the term is restated over a leaf adopting that
        // buffer, resolved, and the requested values take the results.
        // A resolve the replay memo already answers — an unchanged graph
        // under the roots it last hashed, a decode step — takes that path,
        // where the online explorer samples its arms and the step's own
        // last outputs stay roots rather than becoming inputs. Everything
        // else (a grown graph, or new roots over an unchanged one such as
        // the next chunk of a resolved batch) is restated and memoized
        // structurally first.
        // A requested pure view of an unresolved computed value plans that
        // value first: every chunk of a batch needs the same producers, and
        // computing the whole once leaves each chunk a copy of a bound buffer
        // (the restatement below) instead of a run of the whole batch.
        let bases = self.unresolved_view_bases(&graph, values);
        if !bases.is_empty() {
            let bases: Vec<Tensor> = bases.into_iter().map(|id| graph.tensor(id)).collect();
            self.resolve_locked(resolving, &bases)?;
        }
        let replays = self.replay_would_hit(&graph, values);
        if !replays && let Some(cut) = self.cut_at_bound(&graph, values)? {
            if resolve_profile() {
                eprintln!(
                    "[profile] cut at bound values: {:?} -> {:?}",
                    values.iter().map(|v| v.id).collect::<Vec<_>>(),
                    cut.iter().map(|v| v.id).collect::<Vec<_>>()
                );
            }
            self.resolve_locked(resolving, &cut)?;
            for (value, root) in values.iter().zip(&cut) {
                if let Some(buf) = graph.device_buf(root.id) {
                    let layout = graph.device_layout(root.id).map(Arc::new);
                    graph.bind_classes(&[(value.id, buf, layout)]);
                }
            }
            return Ok(None);
        }

        // Inference frontends rebuild the same expression with fresh
        // `from_slice` leaves on every call, often with a length or a view
        // offset moved: the graph grows every call and the replay key (a
        // hash over the ids the roots reach) can never hit. Shape families
        // catch that: the first call records the term with its constants
        // abstracted, an identical later call re-binds the recorded nodes'
        // step buffers and replays their plan, and a call whose constants
        // differ plans one symbolic twin that every call after re-binds.
        // This runs before saturation and extraction, which dominate a
        // replay on an append-only graph. A graph unchanged since its last
        // saturation is left to the replay memo, which is where the online
        // explorer samples its arms.
        if !replays
            && let Some(term) = {
                let roots: Vec<Id> = values.iter().map(|v| v.id).collect();
                let term = canonical_family(&graph, &roots);
                if resolve_profile() && term.is_none() {
                    eprintln!("[profile] shape family: term not canonicalizable");
                }
                term
            }
            && self.family_step(resolving, &graph, values, term)?
        {
            return Ok(None);
        }

        let caps = self.caps();

        let (plan, roots, key, missed) = {
            let mut g = graph.state().egraph.lock();
            // The root set is per-resolve: planning covers exactly the values
            // this call requested. The bindings ride along as costing hints.
            g.dim_hints = graph.dim_bindings().into_iter().collect();
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
            // Saturation is scoped to what the roots reach, so an unchanged
            // arena is saturated for these roots only if they are the roots
            // it last ran for.
            let __skipped = g.saturated_root_sets.contains(g.roots());
            let memo_eligible = !__skipped && g.len() <= SATURATION_MEMO_MAX_NODES;
            let __replayed = memo_eligible && self.inner.saturation.replay(&mut g);
            if !__skipped && !__replayed {
                let pre = memo_eligible.then(|| g.pre_saturation());
                // Scale search with newly reached work. The driver finishes
                // lowering when this budget is exhausted.
                let budget = SaturationBudget {
                    application_slope: 16,
                    ..SaturationBudget::default()
                };
                Driver::new().saturate(&mut g, &caps, &self.inner.rules, budget)?;
                if let Some(pre) = pre {
                    self.inner.saturation.insert(g.record_saturation(pre));
                }
            }
            g.saturated_at_len = Some(g.len());
            if g.saturated_root_sets.len() >= ROOT_SET_MEMOS {
                g.saturated_root_sets.clear();
            }
            let roots_now = g.roots().to_vec();
            g.saturated_root_sets.insert(roots_now);
            let __sat_us = __t_sat.elapsed().as_micros();

            let roots: Vec<Id> = g.roots().to_vec();
            let __t_rest = Instant::now();
            let l0_term = match g.l0_term_memo.get(&roots) {
                Some(hash) => *hash,
                None => {
                    let hash = fusor_cost::replay::l0_term_hash(&g, &roots);
                    if g.l0_term_memo.len() >= ROOT_SET_MEMOS {
                        g.l0_term_memo.clear();
                    }
                    g.l0_term_memo.insert(roots.clone(), hash);
                    hash
                }
            };
            let key = ReplayKey {
                l0_term,
                device: self.inner.cost.facts().fingerprint(),
            };
            // Tuning happens on a memo miss; the winner is what every later
            // resolve of this key replays.
            let missed = self.inner.replay.get(key).is_none();
            let graph_ref: &EGraph = &g;
            let seed = graph.state().extraction_seed.lock().clone();
            let (plan, _unchanged) =
                self.inner
                    .replay
                    .get_or_extract(key, graph_ref, || match seed.as_deref() {
                        Some(seed) => self.inner.extractor.extract_seeded(
                            graph_ref,
                            &roots,
                            self.inner.cost.as_ref(),
                            ExtractBudget::default(),
                            seed,
                        ),
                        None => self.inner.extractor.extract(
                            graph_ref,
                            &roots,
                            self.inner.cost.as_ref(),
                            ExtractBudget::default(),
                        ),
                    })?;
            *graph.state().extraction_seed.lock() = Some(Arc::clone(&plan));
            #[cfg(feature = "compiler-tests")]
            self.inner.extractor.verify_plan(graph_ref, &plan)?;
            if resolve_profile() {
                eprintln!(
                    "[profile] saturate{} {} us ({} -> {} nodes), extract {} us, replay {}",
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
            #[cfg(feature = "gpu")]
            let tuned = self.explore_prior(&graph, &roots, key, tuned);
            self.inner.replay.insert(key, (*tuned).clone());
            tuned
        } else {
            plan
        };

        // Online tuning: on a replay hit, occasionally substitute one legal
        // arm for the incumbent and let this production dispatch's own GPU
        // spans feed the tuner's windows. Every arm is a constructed
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
        let (launched, executable) = self.run(&graph, &plan, values)?;
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
            self.inner
                .cpu_executables
                .lock()
                .insert((plan.hash.0, dims_hash(&graph)), executable);
        }
        #[cfg(not(feature = "cpu"))]
        let _ = executable;
        Ok(Some(plan))
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

    /// [`Self::wait`], awaited: the form a browser can use, where nothing
    /// can block on the device.
    pub async fn wait_async(&self) -> Result<()> {
        match &self.inner.device {
            #[cfg(feature = "gpu")]
            Backend::Gpu(t) => t.wait_async().await?,
            #[allow(unreachable_patterns)]
            _ => self.inner.device.target().wait()?,
        }
        self.inner.in_flight.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// For each requested value that is a chain of pure views over a computed
    /// value with no device buffer, that base value (deduplicated).
    fn unresolved_view_bases(&self, graph: &GraphRef, values: &[Tensor]) -> Vec<Id> {
        let g = graph.state().egraph.lock();
        let mut out: Vec<Id> = Vec::new();
        for value in values {
            let mut id = value.id;
            loop {
                match logical_children(&g, id) {
                    Some(kids) if matches!(g.node(id).op, Op::Union(..)) && kids.len() == 1 => {
                        id = kids[0];
                    }
                    Some(kids)
                        if matches!(g.node(id).op, Op::Logical(Logical::Restride { .. })) =>
                    {
                        id = kids[0];
                    }
                    _ => break,
                }
            }
            if id != value.id
                && !matches!(g.node(id).op, Op::Logical(Logical::Leaf(_)))
                && graph.device_buf(id).is_none()
                && !out.contains(&id)
            {
                out.push(id);
            }
        }
        out
    }

    /// Whether the replay memo holds a plan for exactly these values on the
    /// graph as it stands: its key is the closure hash the graph last cached
    /// for this root set at this length.
    fn replay_would_hit(&self, graph: &GraphRef, values: &[Tensor]) -> bool {
        let g = graph.state().egraph.lock();
        let roots: Vec<Id> = values.iter().map(|v| v.id).collect();
        let Some(hash) = g.l0_term_memo.get(&roots) else {
            return false;
        };
        let key = ReplayKey {
            l0_term: *hash,
            device: self.inner.cost.facts().fingerprint(),
        };
        self.inner.replay.get(key).is_some()
    }

    /// `values`' term with every bound non-leaf value below the roots
    /// replaced by a fresh external leaf adopting its buffer. `None` when
    /// nothing below the roots is bound — the common case — so no node is
    /// minted; otherwise the restated roots, hash-consed onto the original
    /// term wherever nothing changed.
    fn cut_at_bound(&self, graph: &GraphRef, values: &[Tensor]) -> Result<Option<Vec<Tensor>>> {
        let roots: Vec<Id> = values.iter().map(|v| v.id).collect();
        // Cheap pre-check under the leaf lock: any bound value at all?
        let bound: FxHashSet<Id> = graph.bound_values();
        if bound.is_empty() || bound.iter().all(|id| roots.contains(id)) {
            return Ok(None);
        }
        let mut g = graph.state().egraph.lock();
        let Some(order) = logical_postorder(&g, &roots) else {
            return Ok(None);
        };
        // A root is requested by class: its Logical member below a union
        // spelling is the root itself, not an input to cut at.
        let root_classes: FxHashSet<ClassId> = roots.iter().map(|r| g.class_of(*r)).collect();
        let mut memo: FxHashMap<Id, Id> = FxHashMap::default();
        let mut cut_any = false;
        for id in order {
            let node = g.node(id).clone();
            let kids = logical_children(&g, id).expect("walked above");
            let is_root = root_classes.contains(&g.class_of(id));
            let is_leaf = matches!(node.op, Op::Logical(Logical::Leaf(_)));
            let out = if let Op::Union(..) = node.op {
                let mapped: Vec<Id> = kids.iter().map(|c| memo[c]).collect();
                match mapped.as_slice() {
                    [one] => *one,
                    [x, y] if *x == kids[0] && *y == kids[1] => id,
                    [x, y] => g.union(*x, *y)?,
                    _ => id,
                }
            } else if !is_root && !is_leaf && bound.contains(&id) && dense_bound(graph, id) {
                if resolve_profile() {
                    eprintln!("[profile]   cutting {id} ({:?})", node.op.tag());
                }
                let facts = g.facts(id).clone();
                let leaf = g.add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                    name: graph.fresh_buffer_id(),
                    dtype: facts.dtype,
                    shape: facts.shape.clone(),
                })))?;
                if let Some(buf) = graph.device_buf(id) {
                    let layout = graph.device_layout(id).map(Arc::new);
                    graph.bind_leaf(leaf, buf, layout);
                }
                cut_any = true;
                leaf
            } else if is_leaf {
                id
            } else {
                let children: Vec<Id> = kids.iter().map(|c| memo[c]).collect();
                if children == kids {
                    id
                } else {
                    let rebuilt = rebuild_op(&node.op, &children, &mut |d| d).ok_or_else(|| {
                        Error::Plan("a bound intermediate lies below a lowered launch".into())
                    })?;
                    g.add(rebuilt)?
                }
            };
            memo.insert(id, out);
        }
        if !cut_any {
            return Ok(None);
        }
        let out: Vec<Id> = roots.iter().map(|r| memo[r]).collect();
        drop(g);
        Ok(Some(out.into_iter().map(|id| graph.tensor(id)).collect()))
    }

    /// Run `values` through their shape family: the concrete twin recorded
    /// for exactly these constants, or the symbolic twin once the family has
    /// shown more shapes than it keeps concrete plans for. `Ok(false)` when
    /// this call is a new shape the family still plans concretely (and
    /// records), or the family is blocked.
    fn family_step(
        &self,
        resolving: &ResolveGuard<'_>,
        graph: &GraphRef,
        values: &[Tensor],
        term: FamilyTerm,
    ) -> Result<bool> {
        let key = FamilyKey {
            graph: GraphRef::as_ptr(graph) as usize,
            term: term.term.clone(),
        };
        let own_nodes = || Twin {
            consts: term.consts.clone(),
            syms: vec![None; term.consts.len()],
            roots: term
                .term
                .roots
                .iter()
                .map(|r| term.sources[r.index()])
                .collect(),
            leaves: (0..term.inputs.len())
                .map(|ix| (term.inputs[ix], ix))
                .collect(),
        };
        let twin = {
            let mut fams = self.inner.families.lock();
            fams.retain(|_, f| f.graph.strong_count() > 0);
            if fams.len() >= fusor_cost::replay::CAPACITY && !fams.contains_key(&key) {
                fams.clear();
            }
            let is_new = !fams.contains_key(&key);
            let fam = fams.entry(key.clone()).or_insert_with(|| Family {
                graph: GraphRef::downgrade(graph),
                consts: term.consts.clone(),
                varying: vec![false; term.consts.len()],
                group: vec![0; term.consts.len()],
                concrete: vec![own_nodes()],
                symbolic: None,
                blocked: false,
            });
            if is_new {
                if resolve_profile() {
                    eprintln!(
                        "[profile] shape family: first sighting ({} slots, {} families)",
                        term.consts.len(),
                        fams.len()
                    );
                }
                return Ok(false);
            }
            // These values are a recorded twin's own roots: plan them directly.
            let is_own = |t: &Twin| t.roots.iter().zip(values).all(|(r, v)| *r == v.id);
            if fam.concrete.iter().chain(fam.symbolic.iter()).any(is_own) {
                return Ok(false);
            }
            if let Some(i) = fam.concrete.iter().position(|t| t.consts == term.consts) {
                // This exact shape again: its own plan, newest last.
                let t = fam.concrete.remove(i);
                fam.concrete.push(t);
                fam.concrete.last().cloned().expect("pushed")
            } else if fam.blocked || fam.concrete.len() < CONCRETE_SHAPES {
                // A new shape the family still plans concretely.
                if fam.concrete.len() >= CONCRETE_SHAPES {
                    fam.concrete.remove(0);
                }
                fam.concrete.push(own_nodes());
                if resolve_profile() {
                    eprintln!(
                        "[profile] shape family: new shape ({} of {} concrete)",
                        fam.concrete.len(),
                        CONCRETE_SHAPES
                    );
                }
                return Ok(false);
            } else {
                // Shapes keep changing: the symbolic twin. Slots that have
                // differed from the first shape are its symbols, grouped by
                // value history (two slots share a symbol only while they
                // have agreed in every call); a group that splits rebuilds.
                let mut widened = fam.symbolic.is_none();
                for (slot, (a, b)) in fam.consts.iter().zip(&term.consts).enumerate() {
                    if a != b && !fam.varying[slot] {
                        fam.varying[slot] = true;
                        widened = true;
                    }
                }
                let mut by_value: FxHashMap<(usize, u64, u64), usize> = FxHashMap::default();
                let mut regrouped = fam.group.clone();
                for (slot, group) in regrouped.iter_mut().enumerate() {
                    if !fam.varying[slot] {
                        continue;
                    }
                    let next = by_value.len();
                    *group = *by_value
                        .entry((fam.group[slot], fam.consts[slot], term.consts[slot]))
                        .or_insert(next);
                }
                let split = (0..term.consts.len()).any(|a| {
                    fam.varying[a]
                        && (0..a).any(|b| {
                            fam.varying[b]
                                && fam.group[a] == fam.group[b]
                                && regrouped[a] != regrouped[b]
                        })
                });
                widened |= split;
                fam.group = regrouped;
                if widened {
                    match self.build_twin(graph, &term, &fam.varying, &fam.group) {
                        Ok(twin) => fam.symbolic = Some(twin),
                        Err(err) => {
                            if resolve_profile() {
                                eprintln!("[profile] shape family blocked: {err}");
                            }
                            fam.blocked = true;
                            fam.symbolic = None;
                            fam.concrete.push(own_nodes());
                            return Ok(false);
                        }
                    }
                }
                fam.symbolic.clone().expect("built above")
            }
        };
        let symbolic = twin.consts.is_empty();
        let __t = Instant::now();
        for (slot, sym) in twin.syms.iter().enumerate() {
            if let Some(sym) = sym {
                graph.bind_dim(*sym, term.consts[slot]);
            }
        }
        // Capture sources before rebinding: a member may permute the twin's inputs.
        let mut inputs = Vec::new();
        for (leaf, input_ix) in &twin.leaves {
            let source = term.inputs[*input_ix];
            if *leaf == source {
                continue;
            }
            let Some(buf) = graph
                .device_buf(source)
                .or(self.leaf_buffer(graph, source)?)
            else {
                return Ok(false);
            };
            let layout = graph.device_layout(source).map(Arc::new);
            inputs.push((*leaf, buf, layout));
        }
        let bindings_scope = FamilyBindings::new(graph, &twin.roots)?;
        graph.bind_classes(&inputs);
        let twin_values: Vec<Tensor> = twin.roots.iter().map(|id| graph.tensor(*id)).collect();
        // The twin's roots are still bound to the last call's outputs; a
        // bound value is "nothing to plan", so they are unbound first.
        for root in &twin.roots {
            graph.clear_class_device_buf(*root);
        }
        // The twin's own nodes never change between calls, so this is a
        // replay of its plan through the one memo that holds plans. A
        // symbolic twin that fails to plan blocks the family; a concrete one
        // that fails is dropped.
        if let Err(err) = self.resolve_locked(resolving, &twin_values) {
            {
                if resolve_profile() {
                    eprintln!("[profile] shape family blocked at planning: {err}");
                }
                for root in &twin.roots {
                    graph.clear_class_device_buf(*root);
                }
                if let Some(fam) = self.inner.families.lock().get_mut(&key) {
                    if symbolic {
                        fam.blocked = true;
                        fam.symbolic = None;
                    } else {
                        fam.concrete.retain(|t| t.roots != twin.roots);
                    }
                }
                return Ok(false);
            }
        }
        // The member's value is concrete, so its binding carries the twin's
        // layout with this call's values in place of the symbols.
        let bindings: FxHashMap<SymId, u64> = graph.dim_bindings().into_iter().collect();
        let mut outputs = Vec::with_capacity(values.len());
        for (value, root) in values.iter().zip(&twin.roots) {
            let Some(buf) = graph.device_buf(*root) else {
                return Err(Error::Plan(format!(
                    "shape family twin root {root} resolved without a buffer"
                )));
            };
            let layout = match graph.device_layout(*root) {
                Some(layout) => Some(Arc::new(concrete_layout(&layout, &bindings)?)),
                None => None,
            };
            outputs.push((value.id, buf, layout));
        }
        drop(bindings_scope);
        graph.bind_classes(&outputs);
        if resolve_profile() {
            eprintln!(
                "[profile] shape family hit: {} ({} symbol(s), {} us)",
                if symbolic { "symbolic" } else { "concrete" },
                twin.syms.iter().flatten().count(),
                __t.elapsed().as_micros()
            );
        }
        // `FUSOR_VERIFY_FAMILIES`: also plan this member concretely and
        // compare every output byte-for-byte. A twin computing something
        // its member does not is a compiler bug in symbolic lowering.
        #[cfg(feature = "compiler-tests")]
        if verify_families() {
            let mut from_twin = Vec::with_capacity(values.len());
            for value in values {
                from_twin.push(self.read_bytes_locked(resolving, graph, value.id)?);
                graph.clear_class_device_buf(value.id);
            }
            let saved = self.inner.families.lock().remove(&key);
            self.resolve_locked(resolving, values)?;
            if let Some(saved) = saved {
                self.inner.families.lock().insert(key, saved);
            }
            for (o, (value, twin_bytes)) in values.iter().zip(&from_twin).enumerate() {
                let concrete = self.read_bytes_locked(resolving, graph, value.id)?;
                let dtype = graph.facts(value.id).dtype;
                if !agrees(dtype, &concrete, twin_bytes) {
                    let detail = first_mismatch(dtype, &concrete, twin_bytes).map_or_else(
                        String::new,
                        |(i, p, q, w)| {
                            format!(" (elem {i}: concrete {p} vs twin {q}, worst |d| {w})")
                        },
                    );
                    return Err(Error::Plan(format!(
                        "shape family twin disagrees with its member on output {o} ({} vs {} \
                         bytes){detail}",
                        concrete.len(),
                        twin_bytes.len()
                    )));
                }
            }
            eprintln!(
                "[verify] shape family twin agrees on {} output(s)",
                values.len()
            );
        }
        Ok(true)
    }

    /// Mint the family's symbolic twin: the canonical term rebuilt into the
    /// graph with every varying slot a fresh symbol and every other slot its
    /// constant. Step buffers with a symbolic shape become fresh leaves; all
    /// other leaves are the members' own nodes, and hash-consing folds every
    /// unchanged node onto the member's.
    fn build_twin(
        &self,
        graph: &GraphRef,
        term: &FamilyTerm,
        varying: &[bool],
        group: &[usize],
    ) -> Result<Twin> {
        let mut g = graph.state().egraph.lock();
        let mut by_group: FxHashMap<usize, SymId> = FxHashMap::default();
        let syms: Vec<Option<SymId>> = varying
            .iter()
            .zip(group)
            .map(|(v, grp)| v.then(|| *by_group.entry(*grp).or_insert_with(|| g.fresh_sym())))
            .collect();
        let mut subst = |d: Dim| -> Dim {
            match slot_of(d) {
                Some(slot) => match syms[slot] {
                    Some(sym) => Dim::Sym(sym),
                    None => Dim::Const(term.consts[slot]),
                },
                None => d,
            }
        };
        let mut map: Vec<Id> = Vec::with_capacity(term.term.nodes.len());
        let mut leaves = Vec::new();
        for (k, op) in term.term.nodes.iter().enumerate() {
            let id = match op {
                Op::Logical(Logical::Leaf(LeafKind::Buffer { name, dtype, shape })) => {
                    let input_ix = name.0 as usize;
                    let symbolic = shape
                        .iter()
                        .any(|d| slot_of(*d).is_some_and(|slot| syms[slot].is_some()));
                    let id = if symbolic {
                        g.add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                            name: graph.fresh_buffer_id(),
                            dtype: *dtype,
                            shape: shape.iter().map(|d| subst(*d)).collect(),
                        })))?
                    } else {
                        term.inputs[input_ix]
                    };
                    leaves.push((id, input_ix));
                    id
                }
                Op::Logical(Logical::Leaf(LeafKind::Const { .. })) => {
                    let rebuilt = rebuild_op(op, &[], &mut subst).expect("a leaf rebuilds");
                    g.add(rebuilt)?
                }
                Op::Logical(Logical::Leaf(_)) => term.sources[k],
                Op::Union(a, b) => g.union(map[a.index()], map[b.index()])?,
                other => {
                    let children: Vec<Id> = g
                        .semantics()
                        .children(other)
                        .iter()
                        .map(|c| map[c.index()])
                        .collect();
                    let rebuilt = rebuild_op(other, &children, &mut subst).ok_or_else(|| {
                        Error::Plan("a shape family term holds an op it cannot rebuild".into())
                    })?;
                    g.add(rebuilt)?
                }
            };
            map.push(id);
        }
        Ok(Twin {
            consts: Vec::new(),
            syms,
            roots: term.term.roots.iter().map(|r| map[r.index()]).collect(),
            leaves,
        })
    }

    /// Dispatches issued since construction, not encoder submissions.
    pub fn launch_count(&self) -> u64 {
        self.inner.launches.load(Ordering::Relaxed)
    }

    /// Everything a readback of `id` needs once the graph lock is released:
    /// the device buffer, how many bytes to pull, and — for a padded layout
    /// — how to gather the value out of them. See [`Self::read_plan_locked`].
    ///
    /// `_resolving` is a witness that the caller holds the graph's
    /// `resolve_lock`: the plan is only meaningful while no other thread can
    /// be part-way through dispatching a plan that writes the buffer. The
    /// download itself needs no lock: the plan holds its own handle on the
    /// buffer, so the pool cannot hand it to a later resolve, and the copy
    /// is queued after the dispatch that produced it.
    pub(crate) fn read_plan_locked(
        &self,
        _resolving: &ResolveGuard<'_>,
        graph: &GraphRef,
        id: Id,
    ) -> Result<ReadPlan> {
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

        // The registered layout retains the value's logical axes; padding is
        // represented by its strides and is gathered only at readback.
        let padded = graph
            .device_layout(id)
            .filter(|layout| !layout.is_contiguous());
        let Some(layout) = padded else {
            let elements = resolve_elements(&facts.shape, graph)?;
            return Ok(ReadPlan {
                buf,
                bytes: elements * elem,
                gather: None,
            });
        };

        let base = resolve_dim(layout.offset(), graph)?;
        let strides: Vec<u64> = resolve_strides(&layout, graph)?;
        // The bytes to pull are the layout's address span, not its element
        // count: a padded layout addresses past `product(shape)`, and a
        // short download gathers zeros for everything past its end.
        let mut span = 1u64;
        for (d, s) in layout.shape().iter().zip(&strides) {
            span += resolve_dim(*d, graph)?.saturating_sub(1).saturating_mul(*s);
        }
        let extents: Vec<u64> = facts
            .shape
            .iter()
            .map(|d| resolve_dim(*d, graph))
            .collect::<Result<_>>()?;
        Ok(ReadPlan {
            buf,
            bytes: (base + span) * elem,
            gather: Some(Gather {
                base,
                strides,
                extents,
                elem,
            }),
        })
    }

    /// Pull the bytes a [`ReadPlan`] names. Awaited: see [`Backend::download`].
    pub(crate) async fn read_bytes(&self, plan: ReadPlan) -> Result<Vec<u8>> {
        let raw = self.inner.device.download(&plan.buf, plan.bytes).await?;
        Ok(match plan.gather {
            None => raw,
            Some(g) => g.apply(&raw),
        })
    }

    /// A device-side copy of an already-resolved value's buffer, with the
    /// layout that buffer carries. Nothing crosses to the host, so this is
    /// the same call on every platform.
    pub(crate) fn copy_device_locked(
        &self,
        _resolving: &ResolveGuard<'_>,
        graph: &GraphRef,
        id: Id,
    ) -> Result<(Buf, Option<fusor_ir::shape::Layout>)> {
        let buf = graph
            .device_buf(id)
            .ok_or_else(|| Error::Plan(format!("{id} has no device buffer; resolve it first")))?;
        let copy = self.inner.device.copy(&buf)?;
        Ok((copy, graph.device_layout(id)))
    }

    /// Bytes of an already-resolved value, blocking. Native only: on wasm a
    /// readback can only be awaited.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn read_bytes_locked(
        &self,
        resolving: &ResolveGuard<'_>,
        graph: &GraphRef,
        id: Id,
    ) -> Result<Vec<u8>> {
        let plan = self.read_plan_locked(resolving, graph, id)?;
        pollster::block_on(self.read_bytes(plan))
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn read_bytes_locked(
        &self,
        _resolving: &ResolveGuard<'_>,
        _graph: &GraphRef,
        _id: Id,
    ) -> Result<Vec<u8>> {
        Err(Error::Device(
            "a blocking readback is not available on wasm; await the async readback".into(),
        ))
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
    /// A handle may name any member, including a union created by a rewrite.
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
    ) -> Result<(usize, Option<Arc<CpuExecutable>>)> {
        // What the extractor selected for each requested value.
        let wanted: Vec<Id> = values
            .iter()
            .map(|v| self.selected(graph, plan, v.id))
            .collect();
        // An in-place launch that writes through a *persistent* leaf (a
        // cache store) produces that leaf's next contents, so its output
        // stays bound whether or not it was requested: the cache commits it
        // after the step, and it aliases the store, pinning nothing. One that
        // writes through a step-local buffer (a `cat` scattering onto a
        // constant) must not: that buffer returns to the pool with the plan.
        let in_place_roots: rustc_hash::FxHashSet<Id> = {
            let g = graph.state().egraph.lock();
            plan.launches
                .iter()
                .filter(|launch| {
                    launch.members.iter().any(|m| {
                        let node = g.node(*m);
                        match g.semantics().effect(&node.op) {
                            Effect::Pure => false,
                            Effect::InPlace(role) => g
                                .semantics()
                                .children(&node.op)
                                .get(role.0 as usize)
                                .is_some_and(|written| {
                                    matches!(
                                        g.node(*written).op,
                                        Op::Logical(Logical::Leaf(
                                            LeafKind::Buffer { .. } | LeafKind::Param { .. }
                                        ))
                                    ) && graph.device_buf(*written).is_some()
                                }),
                        }
                    })
                })
                .map(|launch| launch.root)
                .collect()
        };
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
        // Root outputs are allocated here rather than inside the backend so
        // the handle survives for readback. The root set is built once;
        // rescanning the launch list per buffer would be quadratic.
        let mut to_bind: Vec<(Id, Buf, Option<Arc<fusor_ir::shape::Layout>>)> = Vec::new();
        for buffer in &plan.buffers {
            if supplied.contains_key(&buffer.value) {
                continue;
            }
            // Only a requested value (and an in-place output through a
            // persistent leaf) is allocated here, where its handle can be
            // bound for readback. Every other plan buffer is the backend's:
            // it allocates each at its first launch and recycles it after
            // its last, so a plan's intermediates share pool buffers.
            if !wanted.contains(&buffer.value) && !in_place_roots.contains(&buffer.value) {
                continue;
            }
            let elements = resolve_dim(buffer.elements, graph)?;
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
            // Only a requested value keeps its buffer bound to the graph. A
            // launch root that is merely an intermediate of this resolve
            // gets scratch that returns to the pool with the plan; binding
            // it too would pin every intermediate of every resolve for the
            // life of the graph — a 32-block vision tower held 22 GB of
            // hidden states that way. A later read of an intermediate
            // recomputes it.
            if wanted.contains(&buffer.value) || in_place_roots.contains(&buffer.value) {
                to_bind.push((
                    buffer.value,
                    buf.clone(),
                    Some(Arc::new(buffer.layout.clone())),
                ));
            }
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
                let cached = self
                    .inner
                    .cpu_executables
                    .lock()
                    .get(&(plan.hash.0, dims_hash(graph)))
                    .cloned();
                let (launched, executable) =
                    self.run_cpu(target.as_ref(), graph, plan, &mut supplied, cached)?;
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
            let elements = resolve_dim(buffer.elements, graph)?;
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
        for (launch_ix, launch) in executable.launches.iter().enumerate() {
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
            if std::env::var_os("FUSOR_DEBUG_CPU_NAN").is_some() {
                let root = plan.launches[launch_ix].root;
                if let Some(buffer) = supplied
                    .get(&root)
                    .and_then(|buffer| buffer.downcast_ref::<fusor_cpu::AlignedBuf>())
                {
                    let values: Vec<f32> = buffer
                        .as_slice()
                        .chunks_exact(4)
                        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                        .collect();
                    eprintln!("[DEBUG-cpu-nan] launch={launch_ix} root={root} values={values:?}");
                    if values.iter().any(|v| v.is_nan()) {
                        eprintln!(
                            "[DEBUG-cpu-nan] OP {:?}",
                            graph.state().egraph.lock().node(root).op
                        );
                    }
                }
            }
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
            // Derived symbols have no word: the CPU lowering evaluates
            // them through the binding.
            .filter(|s| *s < fusor_ir::shape::DERIVED_BASE)
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
        _guard: &ResolveGuard<'_>,
        graph: &GraphRef,
        roots: &[Id],
        base: Arc<Plan>,
        values: &[Tensor],
    ) -> Result<Arc<Plan>> {
        #[cfg(feature = "compiler-tests")]
        if verify_members() {
            return self.check_members(_guard, graph, roots, base, values);
        }
        let min_macs = autotune_min_macs();
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
                if log {
                    eprintln!("[tune] not raced: the plan has an in-place launch");
                }
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
            if log {
                eprintln!(
                    "[tune] not raced: no launch of {} offers a variant above {min_macs} macs",
                    base.launches.len()
                );
            }
            return Ok(base);
        }

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
        // A combination this machine has already raced to a verdict is
        // applied as recorded, not raced again: the race costs seconds per
        // plan shape (every candidate is built, compiled and timed several
        // times), and re-running it in every process would put that on
        // every first transcription or embedding. Production sampling keeps
        // exploring from there. The member sweep must measure everything.
        if let Some(picks) = self.inner.tune.combo(&plan_sig)
            && let Some(plan) = self.apply_combo(graph, roots, &base, &picks, min_macs, log)?
        {
            return Ok(plan);
        }

        #[cfg(feature = "gpu")]
        let _clock = TuningClock::new(&self.inner.device);
        let reference = self.timed_run(graph, &base, values, TUNE_RUNS)?;
        // What this pass actually adopts, per launch, so the combination can
        // be recorded rather than reassembled from per-launch minima that
        // were never measured together.
        let mut picks: Vec<Option<String>> = vec![None; base.launches.len()];

        let mut best = Arc::clone(&base);
        // Every plan built for this race, so what lost can be released.
        let mut raced: Vec<Arc<Plan>> = vec![Arc::clone(&base)];
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
                Some(sig) => {
                    // Each candidate travels with the cost model's prior for
                    // the plan it denotes: on a cold signature the cache races
                    // only the model's top-`RACE_TOP_K` picks.
                    let names: Vec<(String, u64)> = variants
                        .iter()
                        .map(|(n, p)| (n.clone(), p.cost.0))
                        .collect();
                    // The cache orders and prunes; it never replaces the race.
                    // Every candidate it hands back is still built and timed;
                    // a recorded combination is
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
                raced.push(Arc::clone(&candidate));
                let sample = self.timed_run(graph, &candidate, values, TUNE_RUNS)?;
                let sample_ns = plan_ns(&sample);
                // `replan` rebuilds the whole plan from a one-node edit, so a
                // per-launch quantity may be attributed to index `ix` only
                // when every other launch is untouched. Equal roots are not
                // that guarantee: a replan can move a fused member across a
                // launch boundary while both roots and the launch count stay
                // put, leaving `gpu_us[ix]` small because the launch shed
                // work onto its neighbour.
                let aligned = plans_align(&candidate, &best, ix);
                if let Some(sig) = &sig {
                    let measurement = launch_ns(&sample, ix).filter(|_| aligned).or_else(|| {
                        self.inner
                            .device
                            .is_cpu()
                            .then(|| ratio_ppm(sample_ns, base_ns))
                    });
                    if let Some(ns) = measurement {
                        self.inner.tune.observe(sig, &label, ns);
                    }
                }
                if log {
                    eprintln!(
                        "[tune]   L{ix} {sample_ns:.0} ns  (own {} ns, {} ns wall)  {label}{}",
                        launch_ns(&sample, ix).map_or_else(|| "-".to_string(), |ns| ns.to_string()),
                        sample.nanos,
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
                if improved {
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
        self.inner.tune.record_combo(&plan_sig, picks, combo_score);
        // One write per tuning pass, atomic, and a no-op when nothing new was
        // measured — a fully-learned shape costs zero IO.
        self.inner.tune.save();
        if log {
            eprintln!("[tune] winner {best_ns:.0} ns");
        }
        let arena = graph.state().egraph.lock().arena_id();
        self.inner.device.release_candidates(arena, &raced, &best);
        Ok(best)
    }

    /// Rebuild a recorded tuning combination onto `base`: launch by launch,
    /// the variant the race adopted, re-derived against the plan as it stands
    /// after the earlier substitutions (a substitution can change what the
    /// next launch's alternatives are). `None` when a recorded pick no longer
    /// exists — a renamed kernel family — so the caller races afresh.
    fn apply_combo(
        &self,
        graph: &GraphRef,
        roots: &[Id],
        base: &Arc<Plan>,
        picks: &[Option<String>],
        min_macs: u64,
        log: bool,
    ) -> Result<Option<Arc<Plan>>> {
        if picks.len() != base.launches.len() {
            return Ok(None);
        }
        let mut best = Arc::clone(base);
        let mut applied = 0usize;
        for (ix, pick) in picks.iter().enumerate() {
            let Some(name) = pick else {
                continue;
            };
            let variants = {
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
            let Some((_, plan)) = variants.into_iter().find(|(n, _)| n == name) else {
                if log {
                    eprintln!(
                        "[tune] recorded pick `{name}` for launch {ix} no longer exists; racing"
                    );
                }
                return Ok(None);
            };
            best = Arc::new(plan);
            applied += 1;
        }
        #[cfg(feature = "compiler-tests")]
        {
            let g = graph.state().egraph.lock();
            self.inner.extractor.verify_plan(&g, &best)?;
        }
        if log {
            eprintln!(
                "[tune] applied the recorded combination: {applied} of {} launches substituted",
                picks.len()
            );
        }
        Ok(Some(best))
    }

    /// Run `plan` repeatedly and retain timings. No output readback or
    /// correctness decisions belong to the performance selector.
    fn timed_run(
        &self,
        graph: &GraphRef,
        plan: &Plan,
        values: &[Tensor],
        repetitions: usize,
    ) -> Result<TuneSample> {
        let mut nanos = u64::MAX;
        let mut gpu_us: Option<Vec<f64>> = None;
        for _ in 0..repetitions {
            let t = Instant::now();
            let _ = self.run(graph, plan, values)?;
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
        Ok(TuneSample { nanos, gpu_us })
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
/// Whether `FUSOR_VERIFY_FAMILIES` is set: every shape-family hit is
/// cross-checked against a concrete plan of the same member.
#[cfg(feature = "compiler-tests")]
fn verify_families() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR_VERIFY_FAMILIES").is_some())
}

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

/// `op` over `children`, with every extent-carrying field passed through
/// `dims`. `None` for a lowered launch: only raw terms are canonicalized.
fn rebuild_op(op: &Op, children: &[Id], dims: &mut dyn FnMut(Dim) -> Dim) -> Option<Op> {
    let child = |index: usize| children.get(index).copied();
    Some(match op {
        Op::Logical(Logical::Leaf(LeafKind::Buffer { name, dtype, shape })) => {
            Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: *name,
                dtype: *dtype,
                shape: shape.iter().map(|d| dims(*d)).collect(),
            }))
        }
        Op::Logical(Logical::Leaf(LeafKind::Const { value, shape })) => {
            Op::Logical(Logical::Leaf(LeafKind::Const {
                value: *value,
                shape: shape.iter().map(|d| dims(*d)).collect(),
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
        Op::Logical(Logical::Restride { specs, bounds, .. }) => Op::Logical(Logical::Restride {
            specs: specs
                .iter()
                .map(|spec| {
                    let mut spec = *spec;
                    spec.size = dims(spec.size);
                    spec.offset = dims(spec.offset);
                    spec
                })
                .collect(),
            bounds: *bounds,
            x: child(0)?,
        }),
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
        Op::Union(..) => Op::Union(child(0)?, child(1)?),
        Op::Launch(_) => return None,
    })
}

/// Canonicalize an unsaturated term: each runtime buffer name becomes its
/// traversal-order input slot, and (`abstract_consts`) each constant extent
/// of a step buffer's shape or a view spec becomes a slot symbol whose value
/// is collected. The resulting ordinary Op vector provides exact equality and
/// hashing without a parallel hand-written matcher.
/// The children a raw-term walk follows: a `Union`'s Logical members only
/// (a value handed out as its class root is a `Union` whose other members
/// are the launches saturation minted). `None` when a Union has none.
fn logical_children(g: &EGraph, id: Id) -> Option<Vec<Id>> {
    let node = g.node(id);
    match node.op {
        Op::Union(a, b) => {
            let logical: Vec<Id> = [a, b]
                .into_iter()
                .filter(|c| g.node(*c).level == Level::Logical)
                .collect();
            (!logical.is_empty()).then_some(logical)
        }
        _ => Some(node.children.iter().copied().collect()),
    }
}

/// Every node of the raw term below `roots`, children before parents, each
/// once. Iterative: a decoder's term is thousands of nodes deep.
fn logical_postorder(g: &EGraph, roots: &[Id]) -> Option<Vec<Id>> {
    let mut order = Vec::new();
    // 1: expanded (its marker is on the stack), 2: emitted.
    let mut state: FxHashMap<Id, u8> = FxHashMap::default();
    let mut stack: Vec<(Id, bool)> = roots.iter().rev().map(|r| (*r, false)).collect();
    while let Some((id, expanded)) = stack.pop() {
        match state.get(&id) {
            Some(2) => continue,
            Some(1) if !expanded => continue,
            _ => {}
        }
        if expanded {
            state.insert(id, 2);
            order.push(id);
            continue;
        }
        state.insert(id, 1);
        stack.push((id, true));
        for c in logical_children(g, id)?.into_iter().rev() {
            if state.get(&c) != Some(&2) {
                stack.push((c, false));
            }
        }
    }
    Some(order)
}

fn canonicalize(graph: &GraphRef, roots: &[Id], abstract_consts: bool) -> Option<FamilyTerm> {
    let g = graph.state().egraph.lock();
    let order = logical_postorder(&g, roots)?;
    let mut memo: FxHashMap<Id, Id> = FxHashMap::default();
    let mut nodes: Vec<Op> = Vec::with_capacity(order.len());
    let mut sources: Vec<Id> = Vec::with_capacity(order.len());
    let mut inputs: Vec<Id> = Vec::new();
    let mut consts: Vec<u64> = Vec::new();
    for id in order {
        let node = g.node(id);
        let kids = logical_children(&g, id)?;
        // One Logical member stands for its union.
        if matches!(node.op, Op::Union(..)) && kids.len() == 1 {
            let canonical = memo[&kids[0]];
            memo.insert(id, canonical);
            continue;
        }
        let children: Vec<Id> = kids.iter().map(|c| memo[c]).collect();
        let mut slot = |d: Dim| -> Dim {
            match d {
                Dim::Const(v) if abstract_consts => {
                    consts.push(v);
                    slot_dim(consts.len() - 1)
                }
                other => other,
            }
        };
        let op = match &node.op {
            Op::Logical(Logical::Leaf(LeafKind::Buffer { dtype, shape, .. })) => {
                let name = BufferId(inputs.len() as u32);
                inputs.push(id);
                let shape = shape.iter().map(|d| slot(*d)).collect();
                Op::Logical(Logical::Leaf(LeafKind::Buffer {
                    name,
                    dtype: *dtype,
                    shape,
                }))
            }
            other => match rebuild_op(other, &children, &mut slot) {
                Some(op) => op,
                None => {
                    if resolve_profile() {
                        eprintln!("[profile]   not canonicalizable: {id} is {:?}", other.tag());
                    }
                    return None;
                }
            },
        };
        let canonical = Id(nodes.len() as u32);
        nodes.push(op);
        sources.push(id);
        memo.insert(id, canonical);
    }
    let roots = roots.iter().map(|r| memo[r]).collect();
    Some(FamilyTerm {
        term: CanonicalTerm { nodes, roots },
        consts,
        sources,
        inputs,
    })
}

/// The family form of `roots`' term. `None` when the term reaches a lowered
/// launch, or abstracts nothing.
fn canonical_family(graph: &GraphRef, roots: &[Id]) -> Option<FamilyTerm> {
    let t = canonicalize(graph, roots, true)?;
    (!t.consts.is_empty()).then_some(t)
}

/// `hash(sym, value)` over the graph's dim bindings: what a CPU executable,
/// which folds the values in, is specific to.
#[cfg(feature = "cpu")]
fn dims_hash(graph: &GraphRef) -> u64 {
    use std::hash::Hasher;
    let mut h = rustc_hash::FxHasher::default();
    for (sym, value) in graph.dim_bindings() {
        sym.hash(&mut h);
        value.hash(&mut h);
    }
    h.finish()
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

/// One candidate's timing measurements.
struct TuneSample {
    /// Wall clock around the whole `run`, in nanoseconds. Includes buffer
    /// allocation, binding, the plan-cache lookup, submission and the poll
    /// spin, so it is a property of the *process*, not of any one kernel.
    nanos: u64,
    /// Per-launch GPU kernel spans in microseconds, plan order, when the device
    /// could time them. This is the number a tuning decision wants: it excludes
    /// every host cost above and is a property of one launch.
    gpu_us: Option<Vec<f64>>,
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

/// A readback, planned under the graph lock and executed after it.
pub(crate) struct ReadPlan {
    buf: Buf,
    bytes: u64,
    gather: Option<Gather>,
}

/// How to pull a value out of a padded device layout.
struct Gather {
    base: u64,
    strides: Vec<u64>,
    extents: Vec<u64>,
    elem: u64,
}

impl Gather {
    fn apply(&self, raw: &[u8]) -> Vec<u8> {
        let elem = self.elem as usize;
        let count = self.extents.iter().product::<u64>() as usize;
        let mut out = Vec::with_capacity(count * elem);
        let mut idx = vec![0u64; self.extents.len()];
        for _ in 0..count {
            let flat = self.base
                + idx
                    .iter()
                    .zip(&self.strides)
                    .map(|(i, s)| i * s)
                    .sum::<u64>();
            let start = (flat as usize) * elem;
            match raw.get(start..start + elem) {
                Some(slice) => out.extend_from_slice(slice),
                None => out.extend(std::iter::repeat_n(0u8, elem)),
            }
            for axis in (0..self.extents.len()).rev() {
                idx[axis] += 1;
                if idx[axis] < self.extents[axis] {
                    break;
                }
                idx[axis] = 0;
            }
        }
        out
    }
}

/// Whether `id`'s bound device buffer holds its value densely.
///
/// A `Coop` output is padded to whole blocks, and the padding lives in the
/// layout the buffer was bound with. An external leaf carries no
/// `BufferPlan`, so `repad_index` has nothing to correct a read of it with
/// and would take the padding for data; such a value is recomputed rather
/// than cut at.
fn dense_bound(graph: &GraphRef, id: Id) -> bool {
    let Some(layout) = graph.device_layout(id) else {
        return true;
    };
    layout.offset().known_eq(Dim::Const(0))
        && layout.strides() == &fusor_ir::shape::Layout::row_major_strides(layout.shape())[..]
}

fn resolve_dim(d: Dim, graph: &GraphRef) -> Result<u64> {
    // A derived symbol evaluates through the symbols its expression reaches.
    d.evaluate(&mut |s| graph.dim_binding(s))
        .ok_or_else(|| Error::Plan(format!("dim {d} is unbound at dispatch")))
}

fn resolve_strides(layout: &fusor_ir::shape::Layout, graph: &GraphRef) -> Result<Vec<u64>> {
    layout
        .strides()
        .iter()
        .map(|d| resolve_dim(*d, graph))
        .collect()
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
    use crate::graph::Graph;

    #[test]
    #[cfg(all(feature = "cpu", not(target_arch = "wasm32")))]
    fn streamed_reductions_execute_without_the_producer_buffer() {
        use fusor_cost::realize::NodeCache;
        use fusor_ir::extract::Extraction;
        use fusor_ir::ir::launch::Launch;

        let backend = if std::env::var_os("FUSOR_CONFORMANCE_REQUIRE_GPU").is_some() {
            #[cfg(feature = "gpu")]
            {
                Backend::gpu_blocking().expect("GPU backend required")
            }
            #[cfg(not(feature = "gpu"))]
            {
                panic!("GPU feature required")
            }
        } else {
            Backend::cpu().unwrap()
        };
        let session = Session::new(backend).unwrap();
        for producer_axis in [1, 2] {
            let graph = Graph::new(&session);
            let h = graph.handle();
            let symbols = [h.fresh_sym(), h.fresh_sym(), h.fresh_sym()];
            for (sym, value) in symbols.iter().zip([2, 17, 33]) {
                h.bind_dim(*sym, value);
            }
            let shape = if producer_axis == 2 {
                symbols
            } else {
                [symbols[0], symbols[2], symbols[1]]
            };
            let input = graph
                .leaf("x", &shape.map(Dim::Sym), crate::Dtype::F32)
                .unwrap();
            let out = input
                .sum(producer_axis)
                .unwrap()
                .add_scalar(0.125)
                .unwrap()
                .sqr()
                .unwrap()
                .sum(1)
                .unwrap();
            let plan = {
                let mut g = h.state().egraph.lock();
                g.add_root(out.id);
                Driver::new()
                    .saturate(
                        &mut g,
                        &session.caps(),
                        &session.inner.rules,
                        Default::default(),
                    )
                    .unwrap();
                let stream = g
                    .members(g.class_of(out.id))
                    .into_iter()
                    .find(|id| {
                        matches!(g.node(*id).op, Op::Launch(Launch::StreamFold { .. }))
                            && g.node(*id).children.iter().all(|child| *child == input.id)
                    })
                    .expect("ordinary nested reductions must offer a streaming candidate");
                let mut extraction = Extraction::default();
                extraction.sigma.insert(g.class_of(input.id), input.id);
                extraction.sigma.insert(g.class_of(out.id), stream);
                let Op::Launch(op) = &g.node(stream).op else {
                    unreachable!()
                };
                extraction
                    .theta
                    .insert(stream, op.schedule().unwrap().iter().next().unwrap());
                LocalSearch::new(Arc::new(Planner::new()), session.caps())
                    .replan(
                        &g,
                        &[out.id],
                        &mut extraction,
                        session.inner.cost.as_ref(),
                        &mut NodeCache::new(g.len()),
                    )
                    .unwrap()
            };
            assert_eq!(plan.launches.len(), 1);
            let resolving = h.state().resolve_lock.lock();
            for (step, [rows, columns, width]) in [[2, 1, 17], [3, 17, 1], [2, 65, 33], [1, 17, 65]]
                .into_iter()
                .enumerate()
            {
                for (sym, value) in symbols.iter().zip([rows, columns, width]) {
                    h.bind_dim(*sym, value);
                }
                let values: Vec<f32> = (0..rows * columns * width)
                    .map(|i| ((i * 13 + step as u64 * 7) % 61) as f32 / 32.0 - 1.0)
                    .collect();
                input
                    .set_bytes(bytemuck::cast_slice(&values).to_vec())
                    .unwrap();
                session.run(h, &plan, std::slice::from_ref(&out)).unwrap();
                let bytes = session.read_bytes_locked(&resolving, h, out.id).unwrap();
                let actual = bytemuck::cast_slice::<u8, f32>(&bytes);
                assert_eq!(actual.len(), rows as usize);
                for (row, got) in actual.iter().enumerate() {
                    let expected: f64 = (0..columns)
                        .map(|column| {
                            let sum: f64 = (0..width)
                                .map(|k| {
                                    let offset = if producer_axis == 2 {
                                        column * width + k
                                    } else {
                                        k * columns + column
                                    };
                                    f64::from(
                                        values[row * (columns * width) as usize + offset as usize],
                                    )
                                })
                                .sum();
                            (sum + 0.125).powi(2)
                        })
                        .sum();
                    assert!(
                        (f64::from(*got) - expected).abs() < 2e-5 * expected.abs().max(1.0),
                        "shape=[{rows},{columns},{width}] row={row}: {got} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(all(feature = "cpu", not(target_arch = "wasm32")))]
    fn ordinary_attention_discovers_and_executes_a_streamed_weighted_reduction() {
        use fusor_cost::realize::NodeCache;
        use fusor_ir::carrier::SlotTy;
        use fusor_ir::extract::Extraction;
        use fusor_ir::ir::launch::Launch;

        let backend = if std::env::var_os("FUSOR_CONFORMANCE_REQUIRE_GPU").is_some() {
            #[cfg(feature = "gpu")]
            {
                Backend::gpu_blocking().expect("GPU backend required")
            }
            #[cfg(not(feature = "gpu"))]
            {
                panic!("GPU feature required")
            }
        } else {
            Backend::cpu().unwrap()
        };
        let session = Session::new(backend).unwrap();
        let graph = Graph::new(&session);
        let h = graph.handle();
        let seq = h.fresh_sym();
        h.bind_dim(seq, 7);
        let q = graph
            .leaf("q", &[Dim::Const(2), Dim::Const(4)], crate::Dtype::F32)
            .unwrap();
        let k = graph
            .leaf("k", &[Dim::Sym(seq), Dim::Const(4)], crate::Dtype::F32)
            .unwrap();
        let v = graph
            .leaf("v", &[Dim::Sym(seq), Dim::Const(4)], crate::Dtype::F32)
            .unwrap();
        let out = q
            .matmul_t(&k)
            .unwrap()
            .softmax_last_dim()
            .unwrap()
            .matmul(&v)
            .unwrap();
        let (stream, plan) = {
            let mut g = h.state().egraph.lock();
            g.add_root(out.id);
            Driver::new()
                .saturate(
                    &mut g,
                    &session.caps(),
                    &session.inner.rules,
                    Default::default(),
                )
                .unwrap();
            let reachable = g.reachable_from_roots();
            let stream = reachable
                .ones()
                .map(|i| Id(i as u32))
                .find(|id| {
                    let Op::Launch(Launch::StreamFold { producer, fold, .. }) = &g.node(*id).op
                    else {
                        return false;
                    };
                    let Launch::Fold { carrier, .. } = fold.as_ref() else {
                        return false;
                    };
                    let Launch::Fold { ops, .. } = producer.as_ref() else {
                        return false;
                    };
                    carrier.slots.as_slice()
                        == [
                            SlotTy::Scalar,
                            SlotTy::Scalar,
                            SlotTy::Vector(Dim::Const(4)),
                        ]
                        && ops.iter().any(|o| o.src == q.id)
                        && ops.iter().any(|o| o.src == k.id)
                        && g.node(*id)
                            .children
                            .iter()
                            .copied()
                            .collect::<FxHashSet<_>>()
                            == [q.id, k.id, v.id].into_iter().collect()
                })
                .expect(
                    "ordinary QK, softmax and PV must derive a single streamed weighted reduction",
                );
            g.clear_roots();
            g.add_root(stream);
            let mut extraction = Extraction::default();
            for input in [q.id, k.id, v.id, stream] {
                extraction.sigma.insert(g.class_of(input), input);
            }
            let Op::Launch(op) = &g.node(stream).op else {
                unreachable!()
            };
            extraction
                .theta
                .insert(stream, op.schedule().unwrap().iter().next().unwrap());
            let plan = LocalSearch::new(Arc::new(Planner::new()), session.caps())
                .replan(
                    &g,
                    &[stream],
                    &mut extraction,
                    session.inner.cost.as_ref(),
                    &mut NodeCache::new(g.len()),
                )
                .unwrap();
            (stream, plan)
        };
        assert_eq!(plan.launches.len(), 1);
        let state = h.tensor(stream);
        let queries = [1.0f32, 0.25, 0.5, 0.75, 0.5, 0.75, 0.25, 1.0];
        q.set_bytes(bytemuck::cast_slice(&queries).to_vec())
            .unwrap();
        let resolving = h.state().resolve_lock.lock();
        for (length, nonfinite) in [
            (7, 0),
            (7, 1),
            (7, 2),
            (7, 3),
            (7, 4),
            (1, 0),
            (0, 0),
            (9, 0),
        ] {
            h.bind_dim(seq, length);
            let mut keys: Vec<f32> = (0..length * 4)
                .map(|i| (i * 11 % 19) as f32 / 13.0 - 0.5)
                .collect();
            let values: Vec<f32> = (0..length * 4)
                .map(|i| (i * 7 % 23) as f32 / 11.0 - 0.75)
                .collect();
            for row in 0..length as usize {
                if nonfinite == 2 || nonfinite == 1 && row % 2 == 0 {
                    keys[row * 4] = f32::NEG_INFINITY;
                }
            }
            if nonfinite == 3 {
                keys[0] = f32::INFINITY;
            }
            if nonfinite == 4 {
                keys[0] = f32::NAN;
            }
            k.set_bytes(bytemuck::cast_slice(&keys).to_vec()).unwrap();
            v.set_bytes(bytemuck::cast_slice(&values).to_vec()).unwrap();
            session.run(h, &plan, std::slice::from_ref(&state)).unwrap();
            let bytes = session.read_bytes_locked(&resolving, h, stream).unwrap();
            let actual = bytemuck::cast_slice::<u8, f32>(&bytes);
            assert_eq!(actual.len(), 12);
            for row in 0..2 {
                let scores: Vec<f64> = keys
                    .chunks_exact(4)
                    .map(|key| {
                        (0..4)
                            .map(|d| f64::from(queries[row * 4 + d]) * f64::from(key[d]))
                            .sum()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f64 = weights.iter().sum();
                for d in 0..4 {
                    let expected: f64 = weights
                        .iter()
                        .enumerate()
                        .map(|(i, w)| w / sum * f64::from(values[i * 4 + d]))
                        .sum();
                    let got = if length == 0 {
                        assert_eq!(actual[row * 6 + 1], 0.0);
                        actual[row * 6 + 2 + d]
                    } else {
                        actual[row * 6 + 2 + d] / actual[row * 6 + 1]
                    };
                    assert!(
                        expected.is_nan() && got.is_nan()
                            || (f64::from(got) - expected).abs() < 3e-5 * expected.abs().max(1.0),
                        "length={length} nonfinite={nonfinite} row={row} d={d}: {got} != {expected}"
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn shape_family_replay_preserves_inputs_and_prior_results() {
        let session = Session::new(Backend::cpu().unwrap()).unwrap();
        let graph = Graph::new(&session);
        let x = graph
            .leaf("x", &[Dim::Const(2)], crate::Dtype::F32)
            .unwrap();
        let y = graph
            .leaf("y", &[Dim::Const(2)], crate::Dtype::F32)
            .unwrap();
        x.set_bytes(bytemuck::cast_slice(&[1.0f32, 3.0]).to_vec())
            .unwrap();
        y.set_bytes(bytemuck::cast_slice(&[10.0f32, 20.0]).to_vec())
            .unwrap();
        let first = x.sub(&y).unwrap();
        assert_eq!(first.to_vec_f32().unwrap(), [-9.0, -17.0]);
        let swapped = y.sub(&x).unwrap();
        assert_eq!(swapped.to_vec_f32().unwrap(), [9.0, 17.0]);
        assert_eq!(first.to_vec_f32().unwrap(), [-9.0, -17.0]);
        assert_eq!(x.to_vec_f32().unwrap(), [1.0, 3.0]);
        assert_eq!(y.to_vec_f32().unwrap(), [10.0, 20.0]);
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn dead_input_buffers_live_until_their_last_reader_drops() {
        let session = Session::new(Backend::cpu().unwrap()).unwrap();
        let graph = Graph::new(&session);
        let h = graph.handle();
        let input = Tensor::from_elements(h, &[Dim::Const(2)], &[1.0f32, 2.0]).unwrap();
        session.upload_leaf(&input).unwrap();
        let input_id = input.id;
        let reader = input.add_scalar(1.0).unwrap();
        let descendant = reader.add_scalar(2.0).unwrap();
        assert!(h.device_buf(reader.id).is_none());
        assert!(h.device_buf(descendant.id).is_none());
        drop(input);
        h.reap_dead();
        assert!(h.device_buf(input_id).is_some());
        h.reap_dead();
        assert!(h.device_buf(input_id).is_some());
        drop(reader);
        h.reap_dead();
        assert!(h.device_buf(input_id).is_some());
        drop(descendant);
        h.reap_dead();
        assert!(h.device_buf(input_id).is_none());
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn symbolic_contractions_reuse_pipelines_across_tile_boundaries() {
        use fusor_cost::realize::NodeCache;
        use fusor_ir::dtype::Dtype;
        use fusor_ir::extract::Extraction;
        use fusor_ir::ir::launch::{Family, Launch, SchedPoint};
        use fusor_tile::rules::contract::{lower_coop, lower_sgemm, lower_sgemv};

        let backend = match Backend::gpu_blocking() {
            Ok(backend) => backend,
            Err(error) => {
                let required = std::env::var("FUSOR_CONFORMANCE_REQUIRE_GPU").is_ok_and(|value| {
                    !matches!(value.trim(), "" | "0") && !value.trim().eq_ignore_ascii_case("false")
                });
                assert!(!required, "GPU regression is required: {error}");
                eprintln!("symbolic contraction regression skipped: {error}");
                return;
            }
        };
        let session = Session::new(backend).unwrap();
        let target = match &session.inner.device {
            Backend::Gpu(target) => target,
            #[cfg(feature = "cpu")]
            Backend::Cpu(_) => unreachable!(),
        };
        let caps = session.caps();
        eprintln!("symbolic contraction regression on {}", caps.name);
        let mut executions = 0;
        for (family, varying_k, columns, kv_cache) in [
            (Family::Sgemm, false, false, false),
            (Family::Sgemm, true, false, false),
            (Family::Sgemv, false, false, false),
            (Family::Sgemv, false, true, false),
            (Family::Sgemv, true, false, false),
            (Family::Sgemv, true, true, false),
            (Family::Coop, true, false, false),
            (Family::Sgemv, false, false, true),
            (Family::Sgemv, false, true, true),
            (Family::Sgemv, true, false, true),
            (Family::Sgemv, true, true, true),
            (Family::Coop, true, false, true),
        ] {
            if (family == Family::Coop && caps.coop_for(Dtype::F32, Dtype::F32).is_none())
                || (columns && !caps.subgroups.is_some_and(|s| s.is_fixed()))
            {
                continue;
            }
            let graph = Graph::new(&session);
            let h = graph.handle();
            let sym = h.fresh_sym();
            h.bind_dim(sym, 1);
            let (batch, rows, fixed_n, fixed_k) = if kv_cache {
                (8, 4, 128, 128)
            } else {
                (2, 3, 7, 19)
            };
            let capacity_sym = h.fresh_sym();
            h.bind_dim(capacity_sym, 64);
            let capacity_dim = Dim::Sym(capacity_sym);
            let n = if varying_k {
                Dim::Const(fixed_n)
            } else {
                Dim::Sym(sym)
            };
            let k = if varying_k {
                Dim::Sym(sym)
            } else {
                Dim::Const(fixed_k)
            };
            let a = graph
                .leaf("a", &[Dim::Const(batch), Dim::Const(rows), k], Dtype::F32)
                .unwrap();
            let b_shape = if kv_cache {
                [Dim::Const(batch), capacity_dim, Dim::Const(128)]
            } else {
                [Dim::Const(batch), n, k]
            };
            let b = graph.leaf("b", &b_shape, Dtype::F32).unwrap();
            let viewed = kv_cache.then(|| {
                use fusor_ir::shape::StrideSpec;
                b.restride(&[
                    StrideSpec::dim(0, Dim::Const(batch)),
                    StrideSpec::dim(1, Dim::Sym(sym)),
                    StrideSpec::dim(2, Dim::Const(128)),
                ])
                .unwrap()
            });
            let out = if let Some(viewed) = &viewed {
                if varying_k {
                    a.matmul(viewed).unwrap()
                } else {
                    a.matmul_t(viewed).unwrap()
                }
            } else {
                a.matmul_t(&b).unwrap()
            };
            let plan = {
                let mut g = h.state().egraph.lock();
                let node = g.node(out.id).clone();
                let facts = g.facts_view(out.id, &caps);
                let lower = match family {
                    Family::Sgemm => lower_sgemm,
                    Family::Sgemv => lower_sgemv,
                    Family::Coop => lower_coop,
                };
                let mut variant = lower(&mut g.builder(&caps), out.id, &node, &facts)
                    .unwrap_or_else(|| {
                        panic!("no {family:?} varying_k={varying_k} kv_cache={kv_cache}")
                    });
                if kv_cache {
                    // Read the live KV prefix at its physical head stride.
                    let mut op = g.node(variant).op.clone();
                    let Op::Launch(Launch::Contract { b: side, .. }) = &mut op else {
                        unreachable!()
                    };
                    side.ops[0].src = b.id;
                    side.ops[0].layout = fusor_ir::shape::Layout::from_parts(
                        Dim::Const(0),
                        &[Dim::Const(batch), k, n],
                        &[
                            capacity_dim * Dim::Const(128),
                            Dim::Const(if varying_k { 128 } else { 1 }),
                            Dim::Const(if varying_k { 1 } else { 128 }),
                        ],
                    )
                    .unwrap();
                    variant = g.add(op).unwrap();
                    g.union(out.id, variant).unwrap();
                }
                let Op::Launch(Launch::Contract { sched, .. }) = &g.node(variant).op else {
                    unreachable!()
                };
                let point = sched
                    .iter()
                    .find(|point| match point {
                        SchedPoint::Sgemv(p) if kv_cache => {
                            *p == if columns {
                                fusor_ir::ir::launch::SgemvParams {
                                    vector: 32,
                                    subgroups: 2,
                                    cols: 4,
                                    parts: 4,
                                    gap: 32,
                                }
                            } else {
                                fusor_ir::ir::launch::SgemvParams {
                                    vector: 16,
                                    subgroups: 4,
                                    cols: 1,
                                    parts: 1,
                                    gap: 0,
                                }
                            }
                        }
                        SchedPoint::Sgemv(p) => p.cols == if columns { 4 } else { 1 },
                        SchedPoint::Sgemm(p) => p.tn > 1,
                        _ => true,
                    })
                    .unwrap();
                let mut extraction = Extraction::default();
                for id in [a.id, b.id, variant] {
                    extraction.sigma.insert(g.class_of(id), id);
                }
                extraction.theta.insert(variant, point);
                LocalSearch::new(Arc::new(Planner::new()), session.caps())
                    .replan(
                        &g,
                        &[out.id],
                        &mut extraction,
                        session.inner.cost.as_ref(),
                        &mut NodeCache::new(g.len()),
                    )
                    .unwrap()
            };
            assert_eq!(plan.launches.len(), 1);
            let resolving = h.state().resolve_lock.lock();
            h.bind_dim(sym, 33);
            let dispatches = target.launcher().dispatch_count();
            let before = target.launcher().pipeline_compiles();
            let compiled = before + 1;
            let lengths = [33, 1, 15, 16, 17, 65]
                .into_iter()
                .chain([512].into_iter().filter(|_| kv_cache))
                .chain(
                    [1024, 2048, 2049]
                        .into_iter()
                        .filter(|_| family == Family::Sgemv && varying_k && !kv_cache),
                );
            for len in lengths {
                h.bind_dim(sym, len);
                let capacity = len.next_power_of_two().max(64);
                h.bind_dim(capacity_sym, capacity);
                let (n, k) = if varying_k {
                    (fixed_n, len)
                } else {
                    (len, fixed_k)
                };
                let mut av: Vec<f32> = (0..batch * rows * k)
                    .map(|i| ((i * 7 % 17) as f32 - 8.0) / 8.0)
                    .collect();
                let b_elements = if kv_cache {
                    batch * capacity * 128
                } else {
                    batch * n * k
                };
                let bv: Vec<f32> = (0..b_elements)
                    .map(|i| ((i * 11 % 19) as f32 - 9.0) / 8.0)
                    .collect();
                a.set_bytes(bytemuck::cast_slice(&av).to_vec()).unwrap();
                b.set_bytes(bytemuck::cast_slice(&bv).to_vec()).unwrap();
                if len == 33 {
                    let a_bytes = h.leaf_bytes_shared(a.id).unwrap();
                    let b_bytes = h.leaf_bytes_shared(b.id).unwrap();
                    let buffers = target
                        .prepare_resources(
                            &plan,
                            &h.state().egraph.lock(),
                            &h.dim_bindings(),
                            &[0],
                            vec![Arc::clone(&a_bytes), Arc::clone(&b_bytes)],
                        )
                        .unwrap()
                        .unwrap()
                        .join()
                        .unwrap()
                        .unwrap();
                    assert_eq!(target.launcher().pipeline_compiles(), compiled);
                    assert_eq!(target.launcher().dispatch_count(), dispatches);
                    h.bind_prepared_leaf(a.id, &a_bytes, buffers[0].clone());
                    h.bind_prepared_leaf(b.id, &b_bytes, buffers[1].clone());
                    assert_eq!(h.device_buf(a.id).unwrap().addr(), buffers[0].addr());
                    assert_eq!(h.device_buf(b.id).unwrap().addr(), buffers[1].addr());
                    if family == Family::Sgemm && !varying_k {
                        av[0] += 1.0;
                        a.set_bytes(bytemuck::cast_slice(&av).to_vec()).unwrap();
                        h.bind_prepared_leaf(a.id, &a_bytes, buffers[0].clone());
                        assert!(h.device_buf(a.id).is_none());
                        h.bind_prepared_leaf(b.id, &b_bytes, buffers[0].clone());
                        assert_eq!(h.device_buf(b.id).unwrap().addr(), buffers[1].addr());
                    }
                }
                session.run(h, &plan, std::slice::from_ref(&out)).unwrap_or_else(|error| {
                    panic!("{family:?} varying_k={varying_k} columns={columns} kv_cache={kv_cache} length={len}: {error}")
                });
                let bytes = session.read_bytes_locked(&resolving, h, out.id).unwrap();
                let actual = bytemuck::cast_slice::<u8, f32>(&bytes);
                let (batch, rows, n, k, capacity) = (
                    batch as usize,
                    rows as usize,
                    n as usize,
                    k as usize,
                    capacity as usize,
                );
                assert_eq!(actual.len(), batch * rows * n);
                for row in 0..batch * rows {
                    for col in 0..n {
                        let expected: f32 = (0..k)
                            .map(|i| {
                                let b_index = if kv_cache {
                                    (row / rows) * capacity * 128
                                        + if varying_k {
                                            i * 128 + col
                                        } else {
                                            col * 128 + i
                                        }
                                } else {
                                    ((row / rows) * n + col) * k + i
                                };
                                av[row * k + i] * bv[b_index]
                            })
                            .sum();
                        assert!(
                            (actual[row * n + col] - expected).abs() < 1e-4,
                            "{family:?} k={k} n={n} at [{row}, {col}]: {} != {expected}",
                            actual[row * n + col]
                        );
                    }
                }
                executions += 1;
                let count = target.launcher().pipeline_compiles();
                assert_eq!(compiled, count, "{family:?} recompiled at length {len}");
            }
        }
        eprintln!("verified {executions} matrix executions and pipeline reuse");

        // A persisted winner must be the first compiled production plan.
        let graph = Graph::new(&session);
        let h = graph.handle();
        let a =
            Tensor::from_elements(h, &[Dim::ONE, Dim::Const(521)], &vec![0.125f32; 521]).unwrap();
        let b = Tensor::from_elements(
            h,
            &[Dim::Const(1009), Dim::Const(521)],
            &vec![0.25f32; 1009 * 521],
        )
        .unwrap();
        let out = a.matmul_t(&b).unwrap();
        let (base, key, field, base_label, winner) = {
            use fusor_cost::extract::{incumbent_signature, launch_signature};
            let mut g = h.state().egraph.lock();
            let node = g.node(out.id).clone();
            let facts = g.facts_view(out.id, &caps);
            let variant = lower_sgemv(&mut g.builder(&caps), out.id, &node, &facts).unwrap();
            let Op::Launch(Launch::Contract { sched, .. }) = &g.node(variant).op else {
                unreachable!()
            };
            let mut extraction = Extraction::default();
            for id in [a.id, b.id, variant] {
                extraction.sigma.insert(g.class_of(id), id);
            }
            extraction
                .theta
                .insert(variant, sched.iter().next().unwrap());
            let base = Arc::new(
                LocalSearch::new(Arc::new(Planner::new()), caps.clone())
                    .replan(
                        &g,
                        &[out.id],
                        &mut extraction,
                        session.inner.cost.as_ref(),
                        &mut NodeCache::new(g.len()),
                    )
                    .unwrap(),
            );
            let field = launch_signature(&g, &base.launches[0]);
            let label = incumbent_signature(&g, &base, 0).unwrap();
            let winner = session
                .inner
                .extractor
                .launch_variant_labels(
                    &g,
                    &[out.id],
                    &base,
                    0,
                    session.inner.cost.as_ref(),
                    autotune_min_macs(),
                )
                .into_iter()
                .next()
                .unwrap()
                .0;
            let key = ReplayKey {
                l0_term: fusor_cost::replay::l0_term_hash(&g, &[out.id]),
                device: session.inner.cost.facts().fingerprint(),
            };
            (base, key, field, label, winner)
        };
        for _ in 0..2 {
            session.inner.tune.observe(&field, &base_label, 100);
            session.inner.tune.observe(&field, &winner, 1);
        }
        let before = target.launcher().pipeline_compiles();
        let selected = session.explore_prior(h, &[out.id], key, base);
        assert_eq!(target.launcher().pipeline_compiles(), before);
        assert_eq!(
            fusor_cost::extract::incumbent_signature(&h.state().egraph.lock(), &selected, 0),
            Some(winner)
        );
        let resolving = h.state().resolve_lock.lock();
        session
            .run(h, &selected, std::slice::from_ref(&out))
            .unwrap();
        let bytes = session.read_bytes_locked(&resolving, h, out.id).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&bytes), &[16.28125; 1009]);
        assert_eq!(target.launcher().pipeline_compiles(), before + 1);
        eprintln!("verified persisted winner executes with one pipeline compile");
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn saturation_budget_ignores_completed_graph_history() {
        let mut applications = Vec::new();
        for budget in [
            SaturationBudget {
                node_slope: 2,
                node_slack: 0,
                ..Default::default()
            },
            SaturationBudget {
                node_slope: 256,
                max_applications: 0,
                application_slope: 1,
                ..Default::default()
            },
            SaturationBudget {
                max_applications: 0,
                ..Default::default()
            },
        ] {
            let mut expansions = Vec::new();
            for history in [0, 32] {
                let session = Session::new(Backend::cpu().unwrap()).unwrap();
                let graph = Graph::new(&session);
                let h = graph.handle();
                if history > 0 {
                    let input = Tensor::from_elements(h, &[Dim::ONE], &[2.0f32]).unwrap();
                    let old: Vec<_> = (0..history)
                        .map(|i| input.add_scalar(i as f32).unwrap())
                        .collect();
                    let mut g = h.state().egraph.lock();
                    for value in &old {
                        g.add_root(value.id);
                    }
                    Driver::new()
                        .saturate(
                            &mut g,
                            &session.caps(),
                            &session.inner.rules,
                            Default::default(),
                        )
                        .unwrap();
                }
                let a = Tensor::from_elements(
                    h,
                    &[Dim::Const(2), Dim::Const(3)],
                    &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
                )
                .unwrap();
                let b = Tensor::from_elements(
                    h,
                    &[Dim::Const(3), Dim::Const(2)],
                    &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
                )
                .unwrap();
                let out = a
                    .matmul(&b)
                    .unwrap()
                    .add_scalar(1.0)
                    .unwrap()
                    .sum(1)
                    .unwrap();
                let resolving = h.state().resolve_lock.lock();
                let plan = {
                    let mut g = h.state().egraph.lock();
                    g.clear_roots();
                    g.add_root(out.id);
                    let report = Driver::new()
                        .saturate(&mut g, &session.caps(), &session.inner.rules, budget)
                        .unwrap();
                    assert!(!report.saturated);
                    expansions.push((
                        report.final_nodes - report.initial_nodes,
                        report.applications,
                    ));
                    session
                        .inner
                        .extractor
                        .extract(
                            &g,
                            &[out.id],
                            session.inner.cost.as_ref(),
                            ExtractBudget::default(),
                        )
                        .unwrap()
                };
                session.run(h, &plan, std::slice::from_ref(&out)).unwrap();
                let bytes = session.read_bytes_locked(&resolving, h, out.id).unwrap();
                assert_eq!(bytemuck::cast_slice::<u8, f32>(&bytes), [124.0, 295.0]);
            }
            assert_eq!(expansions[0], expansions[1]);
            applications.push(expansions[0].1);
        }
        assert!(applications[1] > applications[2]);
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn exhausted_saturation_still_executes_the_lowering_floor() {
        let session = Session::new(Backend::cpu().unwrap()).unwrap();
        let graph = Graph::new(&session);
        let h = graph.handle();
        let a = Tensor::from_elements(
            h,
            &[Dim::Const(2), Dim::Const(3)],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let b = Tensor::from_elements(
            h,
            &[Dim::Const(3), Dim::Const(2)],
            &[7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0],
        )
        .unwrap();
        let out = a
            .matmul(&b)
            .unwrap()
            .add_scalar(1.0)
            .unwrap()
            .sum(1)
            .unwrap();
        let unrelated = a.add_scalar(5.0).unwrap();
        let resolving = h.state().resolve_lock.lock();
        let execute_floor = |out: &Tensor| {
            let plan = {
                let mut g = h.state().egraph.lock();
                g.clear_roots();
                g.add_root(out.id);
                let report = Driver::new()
                    .saturate(
                        &mut g,
                        &session.caps(),
                        &session.inner.rules,
                        SaturationBudget {
                            max_applications: 0,
                            ..SaturationBudget::default()
                        },
                    )
                    .unwrap();
                assert!(!report.saturated);
                session
                    .inner
                    .extractor
                    .extract(
                        &g,
                        &[out.id],
                        session.inner.cost.as_ref(),
                        ExtractBudget::default(),
                    )
                    .unwrap()
            };
            session.run(h, &plan, std::slice::from_ref(out)).unwrap();
            let bytes = session.read_bytes_locked(&resolving, h, out.id).unwrap();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        };
        assert_eq!(execute_floor(&out), [124.0, 295.0]);
        {
            let mut g = h.state().egraph.lock();
            g.add_root(a.id);
            let before = g.len();
            let report = Driver::new()
                .saturate(
                    &mut g,
                    &session.caps(),
                    &session.inner.rules,
                    SaturationBudget::default(),
                )
                .unwrap();
            assert_eq!(
                report.applications, 0,
                "revisited an exhausted root closure"
            );
            assert_eq!(g.len(), before);
        }
        assert_eq!(execute_floor(&unrelated), [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    /// The same expression rebuilt over a fresh input leaf must run the
    /// recorded plan again rather than extract another: the graph grew, so
    /// the replay key cannot hit, and the structural memo is what remains.
    fn a_fresh_step_leaf_reuses_the_plan(session: Session) {
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
        assert_eq!(session.inner.families.lock().len(), 1);

        assert_eq!(run(&[10.0, 11.0, 12.0]), vec![11.0, 12.0, 13.0]);
        assert_eq!(
            session.inner.families.lock().len(),
            1,
            "a replay hit must not extract and record another plan"
        );
    }

    /// A `Coop` contraction pads its output to whole blocks, and a view of
    /// that output is served by cutting the graph at the bound buffer. The
    /// cut mints an external leaf, which carries no `BufferPlan` for
    /// `repad_index` to correct the read with, so the view must not be cut
    /// there — it read the padding as data.
    #[test]
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn a_view_of_a_padded_contraction_reads_the_value_not_its_padding() {
        let Ok(backend) = Backend::gpu_blocking() else {
            return;
        };
        let session = Session::new(backend).unwrap();
        let graph = Graph::new(&session);
        let h = graph.handle();
        // `n = 130` is not a multiple of any block width, so every geometry
        // the extractor can pick pads it; the extents are large enough that
        // it picks a cooperative one.
        const T: u64 = 512;
        const K: u64 = 512;
        const N: u64 = 130;
        let xs: Vec<f32> = (0..T * K)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let ws: Vec<f32> = (0..N * K)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0)
            .collect();
        let x = Tensor::from_elements(h, &[Dim::Const(T), Dim::Const(K)], &xs).unwrap();
        let w = Tensor::from_elements(h, &[Dim::Const(N), Dim::Const(K)], &ws).unwrap();
        let y = x.matmul_t(&w).unwrap();
        let flat = y
            .reshape_dims(&[Dim::Const(1), Dim::Const(T), Dim::Const(N)])
            .unwrap();

        let full = h.read_back(y.id).unwrap();
        let full = bytemuck::cast_slice::<u8, f32>(&full).to_vec();
        let view = h.read_back(flat.id).unwrap();
        let view = bytemuck::cast_slice::<u8, f32>(&view).to_vec();
        assert_eq!(full.len(), view.len());
        let worst = full
            .iter()
            .zip(&view)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-4, "a view of the contraction differs by {worst}");
    }

    /// A model step rebuilt with a longer cache every call (a `cat` onto a
    /// re-leafed cache, a view at a moving offset, a contraction over the
    /// cache length): from the second call on the session plans a symbolic
    /// twin, and every call's values must equal the host's.
    #[test]
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn a_shape_family_twin_computes_what_its_members_do() {
        let Ok(backend) = Backend::gpu_blocking() else {
            return;
        };
        let session = Session::new(backend).unwrap();
        let graph = Graph::new(&session);
        let h = graph.handle();
        const D: usize = 8;
        let w_host: Vec<f32> = (0..D * D)
            .map(|i| ((i * 7) % 11) as f32 * 0.1 - 0.4)
            .collect();
        let w = Tensor::from_elements(h, &[Dim::Const(D as u64), Dim::Const(D as u64)], &w_host)
            .unwrap();
        let mut cache_host: Vec<f32> = Vec::new();
        for step in 0..(CONCRETE_SHAPES + 4) {
            let new_row: Vec<f32> = (0..D).map(|j| (step * D + j) as f32 * 0.05 - 0.3).collect();
            // The step: cache' = cat(cache, row); q = row @ w; scores =
            // q @ cache'^T (contraction over D, N = len); s = softmax(scores);
            // out = s @ cache' (contraction over len); tail = cache' narrowed
            // at the moving offset `step`.
            let row =
                Tensor::from_elements(h, &[Dim::Const(1), Dim::Const(D as u64)], &new_row).unwrap();
            let cache = if cache_host.is_empty() {
                row.clone()
            } else {
                let prev = Tensor::from_elements(
                    h,
                    &[
                        Dim::Const((cache_host.len() / D) as u64),
                        Dim::Const(D as u64),
                    ],
                    &cache_host,
                )
                .unwrap();
                Tensor::cat(&[prev, row.clone()], 0).unwrap()
            };
            cache_host.extend_from_slice(&new_row);
            let len = cache_host.len() / D;
            let q = row.matmul(&w).unwrap();
            let scores = q.matmul(&cache.t().unwrap()).unwrap();
            let s = scores.softmax(1).unwrap();
            let out = s.matmul(&cache).unwrap();
            let tail = cache.narrow(0, step, 1).unwrap();
            let got_out: Vec<f32> = bytemuck::cast_slice(&h.read_back(out.id).unwrap()).to_vec();
            let got_tail: Vec<f32> = bytemuck::cast_slice(&h.read_back(tail.id).unwrap()).to_vec();
            let got_cache: Vec<f32> =
                bytemuck::cast_slice(&h.read_back(cache.id).unwrap()).to_vec();

            // Host reference.
            let q_h: Vec<f32> = (0..D)
                .map(|j| (0..D).map(|k| new_row[k] * w_host[k * D + j]).sum())
                .collect();
            let sc: Vec<f32> = (0..len)
                .map(|r| (0..D).map(|k| q_h[k] * cache_host[r * D + k]).sum())
                .collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let e: Vec<f32> = sc.iter().map(|v| (v - m).exp()).collect();
            let z: f32 = e.iter().sum();
            let out_h: Vec<f32> = (0..D)
                .map(|j| (0..len).map(|r| e[r] / z * cache_host[r * D + j]).sum())
                .collect();
            let close = |a: &[f32], b: &[f32]| {
                a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4)
            };
            assert!(
                close(&got_cache, &cache_host),
                "step {step}: cache {got_cache:?}"
            );
            assert!(
                close(&got_tail, &cache_host[step * D..(step + 1) * D]),
                "step {step}: tail {got_tail:?}"
            );
            assert!(
                close(&got_out, &out_h),
                "step {step}: out {got_out:?} vs {out_h:?}"
            );
        }
        assert!(
            session
                .inner
                .families
                .lock()
                .values()
                .any(|f| f.symbolic.is_some()),
            "the step's shape family never went symbolic"
        );
    }

    /// The smallest moving-offset view: a fresh `[len, D]` leaf narrowed at
    /// row `step`. The twin's view offset is a derived symbol.
    fn shape_family_symbolic_offset(session: Session) {
        let graph = Graph::new(&session);
        let h = graph.handle();
        const D: usize = 4;
        for step in 0..(CONCRETE_SHAPES + 4) {
            let len = step + 2;
            let host: Vec<f32> = (0..len * D).map(|i| i as f32).collect();
            let x =
                Tensor::from_elements(h, &[Dim::Const(len as u64), Dim::Const(D as u64)], &host)
                    .unwrap();
            let tail = x.narrow(0, step, 1).unwrap().add_scalar(0.0).unwrap();
            let got: Vec<f32> = bytemuck::cast_slice(&h.read_back(tail.id).unwrap()).to_vec();
            assert_eq!(got, host[step * D..(step + 1) * D], "step {step}");
        }
        assert!(
            session
                .inner
                .families
                .lock()
                .values()
                .any(|family| family.symbolic.is_some() && !family.blocked),
            "symbolic offsets must execute without falling back to concrete plans"
        );
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn a_shape_family_twin_reads_a_view_at_a_symbolic_offset() {
        let Ok(backend) = Backend::gpu_blocking() else {
            return;
        };
        shape_family_symbolic_offset(Session::new(backend).unwrap());
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn a_shape_family_twin_reads_a_cpu_view_at_a_symbolic_offset() {
        shape_family_symbolic_offset(Session::new(Backend::cpu().unwrap()).unwrap());
    }

    #[test]
    #[cfg(feature = "cpu")]
    fn a_fresh_step_leaf_reuses_the_cpu_plan_and_executable() {
        a_fresh_step_leaf_reuses_the_plan(Session::new(Backend::cpu().unwrap()).unwrap());
    }

    #[test]
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    fn a_fresh_step_leaf_reuses_the_gpu_plan() {
        let Ok(backend) = Backend::gpu_blocking() else {
            return;
        };
        a_fresh_step_leaf_reuses_the_plan(Session::new(backend).unwrap());
    }
}
