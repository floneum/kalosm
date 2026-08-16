//! `Session` and `Backend`. The session owns the target, the cost model, the
//! extractor and the plan cache; `resolve` is the one place saturation,
//! extraction and dispatch happen.

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
use fusor2_ir::ir::level1::Effect;
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

mod explore;

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

/// Class members the tune race has caught computing wrong values, process
/// wide. Every entry is a live miscompile: a member of some e-class whose
/// value disagrees with its siblings', which extraction could select on some
/// machine. The race already detects these — it value-checks every candidate
/// it times — and detection must be loud: a silent skip would let a wrong
/// staged decode stay green for as
/// long as no case's *selected* member happened to be the broken one. The
/// conformance harness races every class member (`FUSOR2_VERIFY_MEMBERS`) and
/// fails the run when this is nonzero.
static WRONG_MEMBERS: AtomicU64 = AtomicU64::new(0);

/// See [`WRONG_MEMBERS`].
pub fn wrong_member_count() -> u64 {
    WRONG_MEMBERS.load(Ordering::Relaxed)
}

/// Proof that the holder owns a graph's `resolve_lock`.
///
/// Passed rather than taken by the `_locked` entry points, so "the caller
/// already holds it" is checked by the borrow checker instead of by comment.
pub(crate) type ResolveGuard<'a> = parking_lot::MutexGuard<'a, ()>;

/// Which backend a session runs on.
#[derive(Clone)]
pub enum Backend {
    Cpu(Arc<CpuTarget>),
    Gpu(Arc<GpuTarget>),
}

impl Backend {
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
    pub device: Backend,
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
    /// The online explorer's per-key state: deterministic resolve counters
    /// and the candidate arms production sampling is working through.
    explore: parking_lot::Mutex<explore::ExploreState>,
    launches: AtomicU64,
    in_flight: AtomicU32,
}

impl Session {
    pub fn new(device: Backend) -> Result<Self> {
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
                explore: parking_lot::Mutex::new(explore::ExploreState::default()),
                saturation: SaturationMemo::default(),
                launches: AtomicU64::new(0),
                in_flight: AtomicU32::new(0),
            }),
        })
    }

    pub fn device(&self) -> &Backend {
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

        // Every requested value already has a device buffer: nothing to plan.
        // Without this, a readback that follows a batched resolve re-enters
        // saturation and extraction on the whole long-lived graph just to
        // download bytes it already has.
        if values.iter().all(|v| graph.device_buf(v.id).is_some()) {
            return Ok(());
        }

        // Own the back-pressure: the trainer's `--drain-every` is a runtime
        // policy, not a counter in a training script.
        if self.inner.in_flight.load(Ordering::Relaxed) >= MAX_INFLIGHT_PLANS {
            self.wait()?;
        }

        let caps = self.caps();
        // The key discriminates on *which* symbols are bound, never on their
        // values: extraction reads the e-graph (symbols as symbols) and the
        // device, so one plan serves the whole shape family and a decode
        // step's length change replays instead of re-extracting. The values
        // reach the dispatch through the uniform block and `grid_for`.
        let binding: Vec<Dim> = graph
            .dim_bindings()
            .into_iter()
            .map(|(s, _)| Dim::Sym(s))
            .collect();

        let (plan, roots, key, missed) = {
            let mut g = graph.egraph.lock();
            // The root set is per-resolve: planning (and `verify_plan`'s
            // clause 6) covers exactly the values this call requested.
            // Carrying every historical root forward made each decode step
            // replan and re-dispatch the whole generation history.
            g.clear_roots();
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
            // Recording a saturation clones the whole node/facts/parent
            // tables. On a decode loop's long-lived graph that is hundreds of
            // MB of memcpy per token for a memo that can never hit (every
            // step's pre-state differs), so model-scale graphs skip the memo
            // entirely. The bound keeps every suite graph's behavior intact.
            const SATURATION_MEMO_MAX_NODES: usize = 50_000;
            // A graph whose node count is unchanged since its last completed
            // saturation is *exactly* the graph saturation last ran on
            // (`add` is the only structural mutation), so there is nothing
            // to do. This is what makes a decode step whose rebuild was all
            // hash-cons hits skip saturation outright.
            let __skipped = g.saturated_at_len == Some(g.len());
            let memo_eligible = !__skipped && g.len() <= SATURATION_MEMO_MAX_NODES;
            let __replayed = memo_eligible && self.inner.saturation.replay(&mut g);
            if !__skipped && !__replayed {
                let pre = memo_eligible.then(|| g.pre_saturation());
                // `max_applications`' flat default is sized against suite
                // graphs. A model-scale graph (a transformer forward is 40k+
                // nodes) exhausts it mid-walk, so nodes past the exhaustion
                // point never receive their `lower_*` kernel members and the
                // extractor is left choosing among defn expansions only —
                // which is how an 8B forward selected fold-over-materialized-
                // dequant for every late-layer matmul. Scale the ceiling with
                // the pre-saturation node count; suite-sized graphs keep the
                // default exactly.
                let mut budget = SaturationBudget::default();
                budget.max_applications = budget
                    .max_applications
                    .max((g.len() as u32).saturating_mul(16));
                Driver::new().saturate(&mut g, &caps, &self.inner.rules, budget)?;
                // A budget sized off the *pre-growth* node count exhausts
                // mid-walk on a model-scale graph, and every node past the
                // exhaustion point never receives its `lower_*` members —
                // the extractor is left picking defn expansions. The
                // frontier below `len` is the driver's own exhaustion
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
                    let hash = fusor2_cost::replay::l0_term_hash(&g, &roots);
                    g.l0_term_memo = Some((roots.clone(), g.len(), hash));
                    hash
                }
            };
            let key = ReplayKey {
                l0_term,
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
            let __t_verify = Instant::now();
            // `verify_plan` is a pure function of the plan and the graph term
            // it was extracted from, and `key` is that term's identity: the
            // same key already decides *which* plan executes, so re-deriving
            // the same verdict for the same (key, plan) every dispatch is
            // pure repetition — 1.5 ms of every 45 ms decode token, all of it
            // with the GPU idle. A plan reaching L2 for the first time is
            // always verified, an entry replaced by a tuning winner is
            // verified on its own hash, and the member sweep is untouched.
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

        let plan = if missed && matches!(self.inner.device, Backend::Gpu(_)) {
            let tuned = self.autotune(resolving, &graph, &roots, plan, values)?;
            self.inner.replay.insert(key, (*tuned).clone());
            tuned
        } else {
            plan
        };

        // Online tuning: on a replay hit, occasionally substitute one legal
        // arm for the incumbent (or re-sample the incumbent itself) and let
        // this production dispatch's own GPU spans feed the tuner's windows.
        // Every arm is a verify_plan-checked member plan — there is no
        // correctness machinery here, only performance.
        let explored = if !missed && matches!(self.inner.device, Backend::Gpu(_)) {
            self.explore_step(&graph, &roots, key, &plan)
        } else {
            None
        };
        let (plan, _explore_clock) = match &explored {
            Some(sel) => (
                Arc::clone(sel.plan()),
                Some(TuningClock::new(&self.inner.device)),
            ),
            None => (plan, None),
        };

        // Dumps the launch and incumbent signatures of the plan that actually
        // *executes* (post prior-adoption, post explorer substitution) when
        // `FUSOR2_DUMP_EXEC` is set, once per distinct plan hash, so
        // per-dispatch span indices join to the executed kernel rather than to
        // the extraction.
        if dump_exec() {
            use std::collections::HashSet;
            use std::sync::{Mutex as StdMutex, OnceLock};
            static SEEN: OnceLock<StdMutex<HashSet<u128>>> = OnceLock::new();
            let seen = SEEN.get_or_init(|| StdMutex::new(HashSet::new()));
            if seen.lock().unwrap().insert(plan.hash.0) {
                let g = graph.egraph.lock();
                eprintln!(
                    "EXEC plan hash={:x} launches={}",
                    plan.hash.0,
                    plan.launches.len()
                );
                for ix in 0..plan.launches.len() {
                    eprintln!(
                        "  E{ix}: {} :: {}",
                        fusor2_cost::extract::launch_signature(&g, &plan.launches[ix]),
                        fusor2_cost::extract::incumbent_signature(&g, &plan, ix)
                            .unwrap_or_else(|| "base".to_string()),
                    );
                }
            }
        }

        let __t_run = Instant::now();
        self.run(&graph, &plan, values)?;
        if let Some(sel) = explored {
            // Reads the profile the armed clock captured; must run before the
            // clock drops (its drop clears the last profile).
            self.explore_record(sel);
        }
        if resolve_profile() {
            eprintln!(
                "[profile] run {} us ({} launches)",
                __t_run.elapsed().as_micros(),
                plan.launches.len()
            );
        }
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
        // Padding lives in the strides, never in the shape: a padded buffer
        // is detected by its strides departing from the row-major set (or a
        // nonzero offset), not by its shape — the plan layout's shape *is*
        // the logical shape. A shape mismatch still routes here: the reader
        // may be a reshaped spelling of the selected member.
        let padded = graph
            .device_layout(id)
            .filter(|l| {
                l.shape() != &facts.shape[..]
                    || !l.offset().known_eq(fusor2_ir::shape::Dim::Const(0))
                    || l.strides()
                        != &fusor2_ir::shape::Layout::row_major_strides(l.shape())[..]
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
        let layout = layout.cloned().map(Arc::new);
        graph.set_device_buf_class(&members, buf, layout.as_ref());
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
        //
        // The classification runs once over the *distinct* bound values under
        // one lock apiece rather than per binding: the per-id `leaf_buffer`
        // asks the e-graph "is this an external leaf?" and the leaf store "is
        // it already on the device?" behind a mutex each, and a decode plan
        // binds ~7,000 values a step. Only the handful that are genuinely
        // unbacked reach `leaf_buffer` and upload.
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
        // the handle survives for readback. The root set is built once: the
        // membership test is per buffer, and rescanning the launch list for
        // each made this loop quadratic — 1,731 launches against as many
        // buffers on a decode step.
        let launch_roots: rustc_hash::FxHashSet<Id> =
            plan.launches.iter().map(|l| l.root).collect();
        let mut to_bind: Vec<(Id, Buf, Option<Arc<fusor2_ir::shape::Layout>>)> = Vec::new();
        for buffer in &plan.buffers {
            if supplied.contains_key(&buffer.value) {
                continue;
            }
            let is_launch_root = launch_roots.contains(&buffer.value);
            if !is_launch_root && !wanted.contains(&buffer.value) {
                continue;
            }
            let elements = resolve_buffer_elements(buffer.elements, &buffer.layout, graph)?;
            let bytes = (elements * buffer.dtype.byte_size()).max(4);
            let buf = self.inner.device.target().alloc(bytes, buffer.persistence)?;
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
            Backend::Gpu(target) => {
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
            Backend::Cpu(target) => {
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
            let elements = resolve_buffer_elements(buffer.elements, &buffer.layout, graph)?;
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
        // Member verification: race every candidate of every launch so each
        // gets value-checked, but adopt none — the sweep is for coverage, and
        // a plan that changes under measurement would make every dispatch
        // count in the suite nondeterministic.
        let verify_members = std::env::var_os("FUSOR2_VERIFY_MEMBERS").is_some();
        let min_macs = if verify_members {
            0
        } else {
            autotune_min_macs()
        };
        let log = std::env::var_os("FUSOR2_AUTOTUNE_LOG").is_some();

        // Timing a plan re-runs it — `TUNE_RUNS` samples per candidate, plus
        // a value readback per run — and an in-place node makes a re-run
        // destructive, so an impure plan is never *raced*. It is still tuned:
        // the production explorer substitutes one candidate exactly once, in
        // place of the incumbent's own dispatch, which is the same effect
        // either plan would have had. The decode plan (KV append is a
        // `Scatter{Set}`) takes only that path.
        {
            let g = graph.egraph.lock();
            if base.launches.iter().any(|l| {
                l.members
                    .iter()
                    .any(|m| g.semantics().effect(&g.node(*m).op) != Effect::Pure)
            }) {
                return Ok(base);
            }
        }

        // One probe pass over the base plan. `launch_variants` is where the
        // work gate lives, so "every launch offered nothing" is "not worth
        // tuning" — and neither the base measurement nor any candidate run
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
                // The member sweep is a coverage tool: every candidate must be
                // built and value-checked, so the cache must not narrow it.
                Some(_) if verify_members => variants,
                Some(sig) => {
                    // Each candidate travels with the cost model's prior for
                    // the plan it denotes: on a cold signature the cache races
                    // only the model's top-`RACE_TOP_K` picks, so a first
                    // resolve is 3 races per launch instead of 16.
                    let names: Vec<(String, u64)> = variants
                        .iter()
                        .map(|(n, p)| (n.clone(), p.cost.0))
                        .collect();
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
                let aligned = plans_align(&candidate, &best, ix);
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
                        // A tiny output is worth printing whole: the zero/wrong
                        // *pattern* (a column, a row tail, a stripe) names the
                        // bug faster than any single element.
                        if let Some(((_, a), (_, b))) =
                            reference.bytes.first().zip(sample.bytes.first())
                            && a.len() <= 128
                            && a.len() % 4 == 0
                        {
                            let f = |s: &[u8]| {
                                s.chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
                        // a violated compiler invariant, full stop. In
                        // production that is an internal compiler error the
                        // resolve fails loudly on — never a skip, never a
                        // workaround; the kernel or rule gets fixed. Only
                        // the CI member sweep (`FUSOR2_VERIFY_MEMBERS`)
                        // records and continues, so one run can enumerate
                        // every such bug and fail the suite at the end.
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
                            None if matches!(self.inner.device, Backend::Cpu(_)) => {
                                Some(Verdict::Ran(ratio_ppm(sample_ns, base_ns)))
                            }
                            None => None,
                        }
                    };
                    // The member sweep races *every* member of every class
                    // back to back with `min_macs` at zero, so its spans are
                    // measured under contention and at sizes production never
                    // tunes. A `Wrong` verdict is a property of the kernel and
                    // is kept; a `Ran` time is a property of the sweep, and
                    // filing it would let a CI run decide production's
                    // incumbents from numbers production would never observe.
                    let keep = !verify_members || matches!(verdict, Some(Verdict::Wrong));
                    if let Some(verdict) = verdict.filter(|_| keep) {
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
                if ok && improved && !verify_members {
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
            if let Backend::Gpu(target) = &self.inner.device
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

/// The launch-work gate below which nothing is measured, env-overridable.
/// Shared by the cold race and the online explorer so "worth tuning" means
/// one thing.
pub(crate) fn autotune_min_macs() -> u64 {
    std::env::var("FUSOR2_AUTOTUNE_MIN_MACS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(AUTOTUNE_MIN_MACS)
}

/// Whether `FUSOR2_RESOLVE_PROFILE` is set. Read once: `resolve` is the hot
/// path and an env lookup per call is a per-resolve allocation.
fn resolve_profile() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR2_RESOLVE_PROFILE").is_some())
}

/// Whether `FUSOR2_DUMP_EXEC` is set. Prints the launch and incumbent
/// signatures of each distinct executed plan, so a per-dispatch span profile
/// can be joined to the kernel that actually ran.
fn dump_exec() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR2_DUMP_EXEC").is_some())
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

/// Whether `candidate` differs from `incumbent` at exactly launch `ix`, so a
/// per-launch quantity may be attributed to that index and a variant swap is
/// layout-compatible per launch. Equal roots are not that guarantee: a launch
/// is its root *and* its members, grid and block, and a replan can move a
/// fused member across a launch boundary while both roots and the launch
/// count stay put — `gpu_us[ix]` is then small because launch `ix` shed work
/// onto its neighbour. A candidate that fails this is compared and explored
/// at whole-plan granularity instead.
fn plans_align(candidate: &Plan, incumbent: &Plan, ix: usize) -> bool {
    candidate.launches.len() == incumbent.launches.len()
        && candidate
            .launches
            .iter()
            .zip(&incumbent.launches)
            .enumerate()
            .all(|(j, (c, b))| {
                j == ix
                    || (c.root == b.root
                        && c.grid == b.grid
                        && c.block == b.block
                        && {
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
fn batch_aligns(candidate: &Plan, incumbent: &Plan, swaps: &[(usize, String)]) -> bool {
    candidate.launches.len() == incumbent.launches.len()
        && candidate
            .launches
            .iter()
            .zip(&incumbent.launches)
            .enumerate()
            .all(|(j, (c, b))| {
                swaps.iter().any(|(s, _)| *s == j) || launch_key(c) == launch_key(b)
            })
}

/// One launch's identity for plan diffing, hashed: the same fields
/// [`plans_align`] compares — root, grid, block and the member *set* — so two
/// launches with equal keys are the same work on the same schedule. `Id`s are
/// process-local, which is fine: a diff always compares two plans over the
/// same graph.
fn launch_key(launch: &fusor2_ir::extract::Launch) -> u64 {
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
struct TuningClock<'a>(Option<&'a GpuTarget>);

impl<'a> TuningClock<'a> {
    fn new(device: &'a Backend) -> Self {
        let target = match device {
            Backend::Gpu(t) => Some(t.as_ref()),
            Backend::Cpu(_) => None,
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

/// Byte-identical, or — for f32 — within 1e-3 relative magnitude. NaN fails
/// the comparison and is therefore rejected.
/// The first disagreeing f32 element and the worst one, for the MISCOMPILE
/// report: `(first_index, expected, got, worst_abs_diff)`. A miscompile
/// message that names an element turns "run the sweep again with a debugger"
/// into "read the line".
fn first_mismatch(a: &[u8], b: &[u8]) -> Option<(usize, f32, f32, f32)> {
    let f = |s: &[u8]| {
        s.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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
        // Padded run: one *or more* reader dims whose product fits inside the
        // extent, laid out row-major over the value's own extents and reading
        // the unpadded prefix.
        //
        // `m = 3` inside `m_pad = 16` is the length-1 case. A leading batch
        // axis in front of it is the length-2 one — whisper's cross-attention
        // K/V is `[1, 1500, 384]` over a contract padded to `[1504, 384]`, so
        // the reader dims `[1, 1500]` have to share the padded axis — and a
        // singleton-only rule cannot state that at all: it spends the `1` on
        // the padded axis and then finds `1500` will not fit in `384`. The
        // readback failed outright, which is how a whisper decode step died
        // re-leafing its caches.
        //
        // Shortest run first; the longer runs are only reached
        // once the singleton reading has failed the remainder.
        //
        // A padded axis holds exactly **one** logical axis, so the run may
        // contain at most one non-unit reader dim: whisper's `[1, 1500]` over
        // a padded 1504 is one real axis behind a unit batch, and is real
        // data — but `[1, 3, 4]` over a padded 16 nests a second real axis
        // *inside* the padded one at its element stride, and elements 4..11
        // of that reading are the padding between the logical extent and the
        // block edge, returned as data. That parse also races the correct one
        // (`[1, 1, 3]` on the padded m axis, `[4]` on n) in this backtracker,
        // and whichever is tried first wins — which is how a coop-padded
        // `[16, 16]` attention output read back as `[1, 1, 3, 4]` produced
        // row 0 followed by zeros.
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
    // `FUSOR2_RESTATE_LOG` prints every non-trivial restatement: the padded
    // parse is ambiguous in principle (only the producer knows its logical
    // extents), so when a wrong read is suspected this is the record that
    // shows which parse won.
    if std::env::var_os("FUSOR2_RESTATE_LOG").is_some() {
        eprintln!("[restate] layout={l_ext:?}/{l_str:?} reader={r_ext:?} -> strides={strides:?}");
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

/// The `row_major_strides` placeholder (`SymId(u32::MAX)`): a stride past a
/// symbolic axis, derived at dispatch as the product of the following
/// extents. Mirrors `UniformPack::resolve_stride`.
const DERIVED_STRIDE: fusor2_ir::shape::SymId = fusor2_ir::shape::SymId(u32::MAX);

/// Concrete strides of a layout at the current binding, deriving any
/// placeholder from the (now concrete) shape.
fn resolve_strides(layout: &fusor2_ir::shape::Layout, graph: &GraphRef) -> Result<Vec<u64>> {
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
    layout: &fusor2_ir::shape::Layout,
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

#[cfg(test)]
mod tests {
    use super::*;

    /// `restate_layout` has to state a padded buffer over a reader shape whose
    /// dims *share* the padded axis, not just one dim inside it.
    ///
    /// The failing shape is whisper's cross-attention K/V: the value is
    /// `[1, 1500, 384]` and the contract that produced it padded its row axis
    /// to `[1504, 384]`. Reader dims `[1, 1500]` both belong to the 1504 axis.
    /// The singleton-only rule spent the `1` there and then could not fit
    /// `1500` into `384`, so the readback errored and `detach` — the whole
    /// re-leaf-the-cache step of a decode loop — died with "the shapes do not
    /// factor".
    #[test]
    fn a_padded_axis_can_carry_a_run_of_reader_dims() {
        use crate::graph::Graph;
        use fusor2_ir::shape::Layout;
        let g = Graph::new(&Session::new(Backend::cpu().unwrap()).unwrap());
        let dims = |v: &[u64]| -> Vec<Dim> { v.iter().copied().map(Dim::Const).collect() };
        let padded = Layout::from_parts(
            Dim::Const(0),
            &dims(&[1504, 384]),
            &dims(&[384, 1]),
        )
        .unwrap();

        let got = restate_layout(&padded, &dims(&[1, 1500, 384]), g.handle())
            .expect("[1, 1500] shares the padded 1504 axis");
        // Row-major over the value's own extents: the batch axis steps a
        // whole 1500-row block, the row axis one row, the column axis one
        // element. Nothing addresses the four rows of padding.
        assert_eq!(
            got.strides(),
            &dims(&[1500 * 384, 384, 1])[..],
            "a padded run lays out over the value's extents, not the padded ones"
        );

        // The length-1 padded reading is unchanged: `m = 3` inside `m_pad = 16`.
        let tile = Layout::from_parts(Dim::Const(0), &dims(&[16, 16]), &dims(&[16, 1])).unwrap();
        let got = restate_layout(&tile, &dims(&[3, 4]), g.handle()).unwrap();
        assert_eq!(got.strides(), &dims(&[16, 1])[..]);

        // And so is the exact-run one, which must still beat a padded parse:
        // `[2, 2]` fills the `4` batch axis rather than sitting inside it.
        let contract =
            Layout::from_parts(Dim::Const(0), &dims(&[4, 16, 16]), &dims(&[256, 16, 1])).unwrap();
        let got = restate_layout(&contract, &dims(&[2, 2, 4, 4]), g.handle()).unwrap();
        assert_eq!(got.strides(), &dims(&[512, 256, 16, 1])[..]);

        // A run that overflows the padded extent is still no factoring at all.
        assert!(restate_layout(&tile, &dims(&[17, 4]), g.handle()).is_none());
    }

    #[test]
    fn a_cpu_session_registers_every_macro_op_in_table_order() {
        let s = Session::new(Backend::cpu().unwrap()).unwrap();
        for op in crate::composite::MacroOp::ALL {
            assert_eq!(s.registry().get(op.def_id()).unwrap().name, op.name());
        }
    }

    #[test]
    fn the_rule_table_is_the_union_of_every_contributor() {
        let s = Session::new(Backend::cpu().unwrap()).unwrap();
        let expected = CORE_RULES.len()
            + SCHED_RULES.len()
            + fusor2_autograd::ADJOINT_RULES.len()
            + fusor2_cpu::CPU_RULES.len();
        assert_eq!(s.rules().len(), expected);
    }

    #[test]
    fn back_pressure_is_a_library_policy() {
        assert_eq!(MAX_INFLIGHT_PLANS, 8);
        let s = Session::new(Backend::cpu().unwrap()).unwrap();
        assert_eq!(s.launch_count(), 0);
        s.wait().unwrap();
    }

    /// The changed-launch attribution behind [`Gran::Diff`]: the candidate
    /// that drops a producer launch and re-spells the consumer (the
    /// dequantize-once vs decode-in-the-fill pair) diffs as exactly those
    /// launches, everything between attributed to neither side.
    #[test]
    fn sparse_diff_attributes_the_changed_launches() {
        // Identical sequences: no diff to attribute.
        assert_eq!(sparse_diff(&[1, 2, 3], &[1, 2, 3], 4), None);
        // The lm_head shape: incumbent [dequant, A, B, C, fold_f32] vs
        // candidate [A, B, C, fold_native] — a deletion at the front and a
        // substitution at the back, with the shared middle matched.
        let inc = [10, 1, 2, 3, 20];
        let cand = [1, 2, 3, 21];
        let (ca, ib) = sparse_diff(&cand, &inc, 4).unwrap();
        assert_eq!(ca, vec![3]);
        assert_eq!(ib, vec![0, 4]);
        // Pure insertion in the middle.
        let (ca, ib) = sparse_diff(&[1, 9, 2], &[1, 2], 4).unwrap();
        assert_eq!(ca, vec![1]);
        assert_eq!(ib, Vec::<usize>::new());
        // Pure deletion in the middle.
        let (ca, ib) = sparse_diff(&[1, 2], &[1, 9, 2], 4).unwrap();
        assert_eq!(ca, Vec::<usize>::new());
        assert_eq!(ib, vec![1]);
        // Too different for the budget: no attribution.
        assert_eq!(sparse_diff(&[1, 2, 3, 4], &[5, 6, 7, 8], 3), None);
        // Empty against non-empty stays within the cap.
        let (ca, ib) = sparse_diff(&[], &[1, 2], 4).unwrap();
        assert_eq!(ca, Vec::<usize>::new());
        assert_eq!(ib, vec![0, 1]);
        // Repeated keys (2357-launch decode plans are mostly repeated tiny
        // maps): the alignment still isolates a single substitution.
        let inc = [7, 7, 7, 7, 7, 5, 7, 7];
        let cand = [7, 7, 7, 7, 7, 6, 7, 7];
        let (ca, ib) = sparse_diff(&cand, &inc, 4).unwrap();
        assert_eq!(ca, vec![5]);
        assert_eq!(ib, vec![5]);
    }
}
