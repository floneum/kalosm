//! Online tuning from production dispatches: the epsilon explorer.
//!
//! On a deterministic schedule (a per-key resolve counter, never an RNG), one
//! replay-hit resolve in [`EXPLORE_EPSILON`] substitutes a single candidate
//! plan for the incumbent, arms the per-dispatch timestamp path for that one
//! resolve, and files the kernel's GPU span into the tune cache's sliding
//! windows. A candidate displaces the incumbent in the replay memo only when
//! its window minimum beats the incumbent's by [`TUNE_MARGIN`], on at least
//! [`MIN_OBS`] samples.
//!
//! Granularity follows layout. A candidate that leaves every other launch
//! untouched (`plans_align`) is a per-launch swap: its span files under the
//! launch's own signature. A restructuring candidate is timed whole when the
//! plan fits a query set, or at diff granularity ([`Gran::Diff`]): the arm's
//! changed launches are timed together and their sum is judged against the
//! sum of the incumbent's per-launch windows at its changed launches.
//!
//! Every arm is a `verify_plan`-checked plan over members of the same
//! e-classes, so selection is pure performance. A substitution is safe on an
//! impure plan (the decode step's KV append): the arm runs once, instead of
//! the incumbent, so the plan's one write happens exactly once either way.
//!
//! A step builds the one arm it runs and no others: opening a launch's
//! candidate field names its candidates (`launch_variant_labels`); a name
//! becomes a plan in `materialize_arm` for the single label the step selects.
//! A step also times only the launches its comparison reads, unless the arm's
//! granularity is the whole plan.

use std::sync::Arc;

use fusor2_cost::extract::{incumbent_signature, launch_signature, launch_work};
use fusor2_ir::egraph::Id;
use fusor2_ir::extract::{Plan, ReplayKey};
use rustc_hash::FxHashMap;

use super::{
    Backend, Session, TUNE_MARGIN, autotune_min_macs, batch_aligns, plan_sparse_diff, plans_align,
};
use crate::graph::GraphRef;

/// One in this many replay-hit resolves of a key is an exploration step.
/// Deterministic — a per-key counter, never an RNG — so a run is exactly
/// reproducible. `FUSOR2_EXPLORE_EPS` overrides; `0` disables the explorer.
const EXPLORE_EPSILON: u64 = 16;

/// Window samples an arm needs before its window-min may displace the
/// incumbent, and before the explorer stops considering it under-explored.
const MIN_OBS: usize = 2;

/// Keys tracked before the oldest is dropped. Matches the replay memo's
/// capacity.
const KEY_CAP: usize = 64;

/// Edit-distance cap for attributing a restructuring candidate to a changed
/// launch window (see `plan_sparse_diff`). Anything wider is judged
/// whole-plan or not at all.
const MAX_DIFF_EDITS: usize = 8;

/// How many of the heaviest launches the persisted-prior scan builds
/// restructuring variants for. Each variant is a full replan, so this is
/// bounded tightly.
const DIFF_SCAN_TOP_K: usize = 4;

/// How much more step-transient memory an arm may allocate than the
/// incumbent. An arm runs in production in place of the incumbent, so its
/// working set must stay near what production already survived; an
/// over-budget arm is an allocation crash risk, not a measurement.
const ARM_EXTRA_BYTES: u64 = 256 << 20;

/// Step-transient bytes a plan allocates, over its constant-extent buffers.
/// Symbolic extents are decode-step scratch (sequence-length shaped, small);
/// the constant ones are where a restructuring can balloon.
fn plan_step_bytes(plan: &Plan) -> u64 {
    plan.buffers
        .iter()
        .filter(|b| b.persistence == fusor2_ir::dtype::Persistence::Step)
        .filter_map(|b| {
            b.elements
                .as_const()
                .map(|e| e.saturating_mul(b.dtype.byte_size()))
        })
        .sum()
}

fn epsilon() -> u64 {
    static EPS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *EPS.get_or_init(|| {
        std::env::var("FUSOR2_EXPLORE_EPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(EXPLORE_EPSILON)
    })
}

/// The member-verification sweep needs a deterministic dispatch stream —
/// it adopts nothing and counts everything — so the explorer stands down.
fn verify_members() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR2_VERIFY_MEMBERS").is_some())
}

fn tune_log() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("FUSOR2_AUTOTUNE_LOG").is_some())
}

/// A short, process-stable key for whole-plan windows.
fn plan_key(plan_sig: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    plan_sig.hash(&mut h);
    format!("plan:{:016x}", h.finish())
}

#[derive(Default)]
pub(super) struct ExploreState {
    keys: FxHashMap<ReplayKey, KeyState>,
    /// Insertion order, for FIFO eviction.
    order: Vec<ReplayKey>,
}

#[derive(Default)]
struct KeyState {
    /// Replay-hit resolves of this key, the deterministic epsilon source.
    counter: u64,
    /// Built lazily on the first exploration step, dropped on adoption (the
    /// arms were derived against the displaced incumbent).
    arms: Option<ArmSet>,
}

/// The exploration frontier for one replay key's incumbent plan.
struct ArmSet {
    /// Which incumbent these arms and labels describe; everything below goes
    /// stale on adoption.
    incumbent_hash: fusor2_ir::extract::PlanHash,
    /// `launch_signature` per launch, plan order: the field each per-launch
    /// window lives in. Shared with the cold race.
    launch_sigs: Vec<String>,
    /// The incumbent's own per-launch labels — the same `(family, schedule
    /// point)` strings its challengers get, so both rank in one field.
    incumbent_labels: Vec<String>,
    /// Whole-plan window field ([`plan_key`] digest) and the incumbent's
    /// label in it.
    plan_field: String,
    incumbent_plan_label: String,
    /// Launch indices whose candidate fields have not been opened yet,
    /// popped back-to-front — stored ascending by launch work, so the
    /// heaviest launch's field opens first.
    pending: Vec<usize>,
    /// The incumbent's step-transient bytes: the memory bar arms are held to.
    step_bytes: u64,
    /// Per label, the smallest step-transient total any plan realizing it has
    /// come out at against this incumbent. A label is skipped only when even
    /// its cheapest realization is over budget, so a label that has ever fit
    /// is never skipped.
    over_budget: FxHashMap<String, u64>,
    /// Arms of the launch currently being explored, one launch at a time.
    arms: Vec<Arm>,
}

/// How one arm is measured and judged against the incumbent.
#[derive(Clone)]
enum Gran {
    /// Differs from the incumbent at exactly launch `ix`: its span files
    /// under the launch's own signature, against the incumbent's kernel.
    Launch,
    /// Restructured the plan, and the plan is small enough to time whole.
    Whole,
    /// Restructured the plan, but the changed launches on each side are a
    /// small set (`plan_sparse_diff`): the arm's changed launches are timed
    /// together and their summed span — filed under `field` — is judged
    /// against the sum of the incumbent's own per-launch window minimums at
    /// its changed launches.
    Diff {
        cand: Vec<usize>,
        inc: Vec<usize>,
        field: String,
    },
}

/// One candidate of the launch currently being explored.
///
/// An arm is named by the label its offering class carries and only built
/// (a whole-plan replan) on the step it runs.
struct Arm {
    ix: usize,
    label: String,
    /// The candidate itself, built by [`Session::materialize_arm`] on the
    /// step this arm is first run, and `None` until then.
    plan: Option<Arc<Plan>>,
    /// How the arm is measured. Known only once the plan exists — it is a
    /// property of how the candidate's launches line up against the
    /// incumbent's. An unbuilt arm is read as [`Gran::Launch`].
    gran: Option<Gran>,
}

impl Arm {
    /// The granularity an arm is read at before it is built.
    fn gran(&self) -> &Gran {
        self.gran.as_ref().unwrap_or(&Gran::Launch)
    }
}

/// The stats field a diff arm's summed window files under: stable across
/// processes, derived from the plan field and the incumbent-side changed
/// launches' signatures, so every restructuring that displaces the same
/// incumbent work ranks in one field.
fn diff_field(plan_field: &str, launch_sigs: &[String], inc: &[usize]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = rustc_hash::FxHasher::default();
    plan_field.hash(&mut h);
    for &j in inc {
        launch_sigs.get(j).hash(&mut h);
        j.hash(&mut h);
    }
    format!("diff:{:016x}", h.finish())
}

impl ArmSet {
    fn metadata(graph: &GraphRef, incumbent: &Arc<Plan>) -> Self {
        let g = graph.state().egraph.lock();
        let launch_sigs: Vec<String> = incumbent
            .launches
            .iter()
            .map(|l| launch_signature(&g, l))
            .collect();
        let incumbent_labels: Vec<String> = (0..incumbent.launches.len())
            .map(|ix| {
                incumbent_signature(&g, incumbent, ix).unwrap_or_else(|| "base".to_string())
            })
            .collect();
        // Heaviest work first: `pending` pops from the back, so it is stored
        // ascending by work.
        let mut pending: Vec<usize> = (0..incumbent.launches.len()).collect();
        let works: Vec<u64> = pending
            .iter()
            .map(|&ix| launch_work(&g, incumbent, ix))
            .collect();
        pending.sort_by_key(|&ix| (works[ix], std::cmp::Reverse(ix)));
        let plan_field = plan_key(&launch_sigs.join(";"));
        Self {
            incumbent_hash: incumbent.hash,
            pending,
            launch_sigs,
            incumbent_labels,
            plan_field,
            incumbent_plan_label: format!("{:032x}", incumbent.hash.0),
            arms: Vec::new(),
            step_bytes: plan_step_bytes(incumbent),
            over_budget: FxHashMap::default(),
        }
    }

    /// Whether a named candidate of launch `ix` is worth the whole-plan
    /// replan that building it costs: a window too thin to judge, or a window
    /// thick enough to judge that says this label wins. The read is
    /// `launch_sigs[ix]` — an arm that turns out to restructure files
    /// elsewhere, reads zero here, and so counts as thin.
    fn worth_building(
        &self,
        tune: &fusor2_cost::tune_cache::TuneCache,
        ix: usize,
        label: &str,
    ) -> bool {
        let Some(field) = self.launch_sigs.get(ix) else {
            return false;
        };
        if self
            .over_budget
            .get(label)
            .is_some_and(|&b| b > self.step_bytes.saturating_add(ARM_EXTRA_BYTES))
        {
            return false;
        }
        if tune.observations(field, label) < MIN_OBS {
            return true;
        }
        let Some(inc) = self.incumbent_labels.get(ix) else {
            return false;
        };
        match (tune.window_min(field, label), tune.window_min(field, inc)) {
            (Some(a), Some(b)) => (a as f64) < b as f64 * (1.0 - TUNE_MARGIN),
            _ => false,
        }
    }

    /// The stats key one arm's observations file under.
    fn arm_field<'a>(&'a self, arm: &'a Arm) -> (&'a str, &'a str) {
        match arm.gran() {
            Gran::Launch => (self.launch_sigs[arm.ix].as_str(), arm.label.as_str()),
            Gran::Whole => (self.plan_field.as_str(), arm.label.as_str()),
            Gran::Diff { field, .. } => (field.as_str(), arm.label.as_str()),
        }
    }

    /// How many observations the *incumbent's* side of the comparison holds.
    /// For a diff arm that is the thinnest window among its changed launches:
    /// every one of them needs a span before the sum means anything.
    fn incumbent_obs(&self, tune: &fusor2_cost::tune_cache::TuneCache, arm: &Arm) -> usize {
        match arm.gran() {
            Gran::Launch => tune.observations(
                self.launch_sigs[arm.ix].as_str(),
                self.incumbent_labels[arm.ix].as_str(),
            ),
            Gran::Whole => {
                tune.observations(self.plan_field.as_str(), self.incumbent_plan_label.as_str())
            }
            Gran::Diff { inc, .. } => inc
                .iter()
                .map(|&j| {
                    tune.observations(
                        self.launch_sigs[j].as_str(),
                        self.incumbent_labels[j].as_str(),
                    )
                })
                .min()
                .unwrap_or(0),
        }
    }

    /// The incumbent-side number an arm must beat, in ns. For a diff arm the
    /// sum of per-launch window minimums is at most the minimum of sums, so
    /// the comparison is biased *for* the incumbent — a diff adoption is
    /// conservative by construction.
    fn incumbent_min(&self, tune: &fusor2_cost::tune_cache::TuneCache, arm: &Arm) -> Option<u64> {
        match arm.gran() {
            Gran::Launch => tune.window_min(
                self.launch_sigs[arm.ix].as_str(),
                self.incumbent_labels[arm.ix].as_str(),
            ),
            Gran::Whole => {
                tune.window_min(self.plan_field.as_str(), self.incumbent_plan_label.as_str())
            }
            Gran::Diff { inc, .. } => inc
                .iter()
                .map(|&j| {
                    tune.window_min(
                        self.launch_sigs[j].as_str(),
                        self.incumbent_labels[j].as_str(),
                    )
                })
                .sum::<Option<u64>>(),
        }
    }
}

/// What one exploration step decided to run. Returned to `resolve_locked`,
/// which runs the plan with the timestamp clock armed and hands the token
/// back to [`Session::explore_record`].
pub(super) struct ExploreRun {
    plan: Arc<Plan>,
    key: ReplayKey,
    /// `None`: the incumbent itself is being sampled.
    pick: Option<Pick>,
    /// Whether the plan was timed whole. A focused resolve yields one
    /// launch's span; a plan-level total exists only when every dispatch
    /// was timed.
    whole: bool,
}

struct Pick {
    ix: usize,
    label: String,
    gran: Gran,
}

impl ExploreRun {
    pub(super) fn plan(&self) -> &Arc<Plan> {
        &self.plan
    }
}

impl Session {
    /// Decide what this replay-hit resolve runs. `None` on most resolves —
    /// not an epsilon step, or the explorer is off.
    pub(super) fn explore_step(
        &self,
        graph: &GraphRef,
        roots: &[Id],
        key: ReplayKey,
        incumbent: &Arc<Plan>,
    ) -> Option<ExploreRun> {
        let eps = epsilon();
        if eps == 0 || verify_members() {
            return None;
        }
        let mut state = self.inner.explore.lock();
        let ks = state.key_mut(key);
        ks.counter += 1;
        // First replay hit of this key in this process: turn the persisted
        // windows into adoptions in one pass, before any sequential
        // exploration. Every swap clears the same bar `maybe_adopt` enforces.
        if ks.counter == 1 && self.adopt_persisted(graph, roots, key, incumbent) {
            // The next resolve replays the adopted plan; exploring the
            // displaced incumbent now would file arms against dead labels.
            return None;
        }
        if ks.counter % eps != 0 {
            return None;
        }

        // The arms must describe the plan that is actually incumbent.
        if ks
            .arms
            .as_ref()
            .is_some_and(|a| a.incumbent_hash != incumbent.hash)
        {
            ks.arms = None;
        }
        // Whether this plan can be timed whole. Past the query-set cap only a
        // focused set of launches can be timed per resolve, which bounds this
        // key's exploration to per-launch granularity.
        let whole = match &self.inner.device {
            Backend::Gpu(t) => t.launcher().can_time_whole(incumbent.launches.len()),
            Backend::Cpu(_) => return None,
        };
        let set = ks
            .arms
            .get_or_insert_with(|| ArmSet::metadata(graph, incumbent));

        // Adoption sweep over the current arms before anything else: an arm's
        // own runs and the incumbent's window fills arrive on different
        // steps, so the win often becomes visible on a step that runs
        // neither. The winner is found from window lookups alone and built
        // only if it is not built already.
        let winner = {
            let tune = &self.inner.tune;
            set.arms
                .iter()
                .enumerate()
                .filter(|(_, a)| {
                    let (f, l) = set.arm_field(a);
                    tune.observations(f, l) >= MIN_OBS && set.incumbent_obs(tune, a) >= MIN_OBS
                })
                .filter_map(|(i, a)| {
                    let (f, l) = set.arm_field(a);
                    let am = tune.window_min(f, l)?;
                    let bm = set.incumbent_min(tune, a)?;
                    ((am as f64) < bm as f64 * (1.0 - TUNE_MARGIN))
                        .then(|| (i, l.to_string(), am, bm))
                })
                .min_by_key(|(_, _, am, _)| *am)
        };
        if let Some((i, label, a, b)) = winner {
            let built = set.arms[i].plan.is_some()
                || self.materialize_arm(graph, roots, incumbent, set, i, whole);
            match built.then(|| set.arms[i].plan.clone()).flatten() {
                Some(plan) => {
                    if tune_log() {
                        eprintln!(
                            "[tune] ADOPT `{label}`: window-min {a} ns vs incumbent {b} ns, \
                             from the arm sweep"
                        );
                    }
                    self.inner.replay.insert(key, (*plan).clone());
                    // Everything in the arm set described the displaced
                    // incumbent.
                    ks.arms = None;
                    return None;
                }
                // The window ranks a label this incumbent cannot realize.
                None => {
                    set.arms.remove(i);
                }
            }
        }
        let set = ks.arms.as_mut().expect("ensured above");

        // Advance the frontier: when the current launch's field is explored
        // out (every arm has MIN_OBS window samples), open the next launch's.
        // Opening a field names its candidates and builds none of them.
        if set
            .arms
            .iter()
            .all(|a| self.inner.tune.observations(set.arm_field(a).0, set.arm_field(a).1) >= MIN_OBS)
        {
            set.arms.clear();
            while let Some(ix) = set.pending.pop() {
                let labels = {
                    let g = graph.state().egraph.lock();
                    self.inner.extractor.launch_variant_labels(
                        &g,
                        incumbent,
                        ix,
                        autotune_min_macs(),
                    )
                };
                let tune = &self.inner.tune;
                let worth: Vec<Arm> = labels
                    .into_iter()
                    .filter(|label| set.worth_building(tune, ix, label))
                    .map(|label| Arm {
                        ix,
                        label,
                        plan: None,
                        gran: None,
                    })
                    .collect();
                set.arms.extend(worth);
                if !set.arms.is_empty() {
                    break;
                }
            }
        }

        // Least-observed arm first; ties break by label so a run is
        // reproducible. The chosen arm is built here and nowhere else; a
        // label that cannot be realized is dropped and the
        // next-least-observed label takes the step.
        let arm_ix = loop {
            let Some(i) = set
                .arms
                .iter()
                .enumerate()
                .min_by_key(|(_, a)| {
                    let (f, l) = set.arm_field(a);
                    (self.inner.tune.observations(f, l), a.label.clone())
                })
                .map(|(i, _)| i)
            else {
                break None;
            };
            if set.arms[i].plan.is_some() {
                break Some(i);
            }
            if self.materialize_arm(graph, roots, incumbent, set, i, whole) {
                break Some(i);
            }
            set.arms.remove(i);
        };
        let counter = ks.counter;
        let (run, focus) = match arm_ix {
            Some(i) => {
                let arm = &set.arms[i];
                let (af, al) = set.arm_field(arm);
                let arm_obs = self.inner.tune.observations(af, al);
                let inc_obs = set.incumbent_obs(&self.inner.tune, arm);
                let whole = matches!(arm.gran(), Gran::Whole);
                if inc_obs < MIN_OBS && inc_obs <= arm_obs {
                    // Sample the incumbent at the launches under contest, so
                    // both sides of the adoption comparison fill.
                    let focus = match arm.gran() {
                        Gran::Diff { inc, .. } => inc.clone(),
                        _ => vec![arm.ix],
                    };
                    (
                        ExploreRun {
                            plan: Arc::clone(incumbent),
                            key,
                            pick: None,
                            whole,
                        },
                        focus,
                    )
                } else {
                    let focus = match arm.gran() {
                        Gran::Diff { cand, .. } => cand.clone(),
                        _ => vec![arm.ix],
                    };
                    (
                        ExploreRun {
                            plan: arm
                                .plan
                                .clone()
                                .expect("the chosen arm was materialized above"),
                            key,
                            pick: Some(Pick {
                                ix: arm.ix,
                                label: arm.label.clone(),
                                gran: arm.gran().clone(),
                            }),
                            whole,
                        },
                        focus,
                    )
                }
            }
            // Frontier exhausted: keep the incumbent's windows fresh, one
            // rotating launch per step. The stale-prior scan in
            // `explore_record` re-opens a launch when the persisted windows
            // say some arm now beats the incumbent.
            None => {
                let rotate = (counter as usize) % incumbent.launches.len().max(1);
                (
                    ExploreRun {
                        plan: Arc::clone(incumbent),
                        key,
                        pick: None,
                        whole: false,
                    },
                    vec![rotate],
                )
            }
        };
        // Time exactly the launches the comparison reads; only a whole-plan
        // arm leaves the focus unset.
        if !run.whole && let Backend::Gpu(t) = &self.inner.device {
            t.launcher().set_tuning_focus(Some(focus));
        }
        Some(run)
    }

    /// Build the candidate one arm names and classify how it is measured.
    /// `false` when the label realizes no plan this step may run — an
    /// illegal selection, a plan that fails to build or verify, a working
    /// set past the memory bar, or a restructuring with no measurable
    /// window; the caller drops the arm.
    fn materialize_arm(
        &self,
        graph: &GraphRef,
        roots: &[Id],
        incumbent: &Arc<Plan>,
        set: &mut ArmSet,
        i: usize,
        whole: bool,
    ) -> bool {
        let ix = set.arms[i].ix;
        let label = set.arms[i].label.clone();
        let plan = {
            let g = graph.state().egraph.lock();
            self.inner.extractor.replan_with_variants(
                &g,
                roots,
                incumbent,
                self.inner.cost.as_ref(),
                autotune_min_macs(),
                &[(ix, label.clone())],
            )
        };
        let Some(plan) = plan else { return false };
        let bytes = plan_step_bytes(&plan);
        let entry = set.over_budget.entry(label.clone()).or_insert(bytes);
        *entry = (*entry).min(bytes);
        if bytes > set.step_bytes.saturating_add(ARM_EXTRA_BYTES) {
            return false;
        }
        let gran = if plans_align(&plan, incumbent, ix) {
            Gran::Launch
        } else if whole {
            Gran::Whole
        } else if let Some((cand, inc)) = plan_sparse_diff(&plan, incumbent, MAX_DIFF_EDITS)
            && !cand.is_empty()
            && !inc.is_empty()
        {
            let field = diff_field(&set.plan_field, &set.launch_sigs, &inc);
            if tune_log() {
                eprintln!(
                    "[tune] diff arm L{ix} `{label}`: cand {cand:?} inc {inc:?} \
                     ({} vs {} launches) -> {field}",
                    plan.launches.len(),
                    incumbent.launches.len(),
                );
            }
            Gran::Diff { cand, inc, field }
        } else {
            // Restructured too widely to attribute a window, and the plan is
            // too large to time whole.
            return false;
        };
        set.arms[i].plan = Some(Arc::new(plan));
        set.arms[i].gran = Some(gran);
        true
    }

    /// File the armed resolve's GPU spans into the windows and adopt on a
    /// window-min win. Called after `run`, while the tuning clock is alive.
    pub(super) fn explore_record(&self, run: ExploreRun) {
        let Backend::Gpu(target) = &self.inner.device else {
            return;
        };
        let Some(spans_us) = target.launcher().take_last_profile() else {
            // The device could not time this resolve; a wall clock is not a
            // kernel span, so nothing is recorded.
            if tune_log() {
                eprintln!("[tune] explore step yielded no profile");
            }
            return;
        };
        if spans_us.len() != run.plan.launches.len() {
            if tune_log() {
                eprintln!(
                    "[tune] explore profile length mismatch: {} spans vs {} launches",
                    spans_us.len(),
                    run.plan.launches.len()
                );
            }
            return;
        }
        let ns = |us: f64| (us * 1000.0) as u64;
        let total_ns = ns(spans_us.iter().sum());

        let mut state = self.inner.explore.lock();
        let Some(ks) = state.keys.get_mut(&run.key) else {
            return;
        };
        let Some(set) = ks.arms.as_ref() else { return };
        let tune = &self.inner.tune;

        match &run.pick {
            None => {
                // The incumbent's own production sample: every launch ran its
                // own kernel, so every span files under its own label, and
                // the plan total under the plan field.
                let mut timed = 0usize;
                for (j, us) in spans_us.iter().enumerate() {
                    if *us > 0.0 {
                        tune.observe(&set.launch_sigs[j], &set.incumbent_labels[j], ns(*us));
                        timed += 1;
                    }
                }
                if tune_log() && !run.whole {
                    eprintln!("[tune] incumbent sample: {timed} focused span(s) filed");
                }
                if run.whole && total_ns > 0 {
                    tune.observe(&set.plan_field, &set.incumbent_plan_label, total_ns);
                }
                // Stale-prior scan: when the accumulated windows say some
                // launch's best variant beats the incumbent's own kernel,
                // re-open that launch so the arm gets rebuilt, re-observed
                // and — if it holds up — adopted.
                let reopen: Vec<usize> = (0..set.launch_sigs.len())
                    .filter(|&j| {
                        let inc = tune.window_min(&set.launch_sigs[j], &set.incumbent_labels[j]);
                        match (tune.best(&set.launch_sigs[j]), inc) {
                            (Some((name, best)), Some(inc)) => {
                                name != set.incumbent_labels[j]
                                    && (best as f64) < inc as f64 * (1.0 - TUNE_MARGIN)
                            }
                            _ => false,
                        }
                    })
                    .collect();
                if !reopen.is_empty() {
                    let set = ks.arms.as_mut().expect("checked above");
                    for j in reopen {
                        if !set.pending.contains(&j) && !set.arms.iter().any(|a| a.ix == j) {
                            set.pending.push(j);
                        }
                    }
                }
            }
            Some(pick) => {
                match &pick.gran {
                    Gran::Launch => {
                        // Launch `ix` ran the arm's kernel, every other
                        // launch ran the incumbent's, so each span files
                        // under the kernel that produced it.
                        for (j, us) in spans_us.iter().enumerate() {
                            if *us <= 0.0 {
                                continue;
                            }
                            if j == pick.ix {
                                tune.observe(&set.launch_sigs[j], &pick.label, ns(*us));
                            } else {
                                tune.observe(
                                    &set.launch_sigs[j],
                                    &set.incumbent_labels[j],
                                    ns(*us),
                                );
                            }
                        }
                    }
                    Gran::Whole => {
                        // Only the whole is comparable.
                        if run.whole && total_ns > 0 {
                            tune.observe(&set.plan_field, &pick.label, total_ns);
                        }
                    }
                    Gran::Diff { cand, field, .. } => {
                        // The arm's changed launches were timed together; the
                        // summed span is the arm's number. A zero span among
                        // them is a slot the device did not write — filing a
                        // partial sum would flatter the arm, so nothing is
                        // recorded.
                        let spans: Vec<f64> = cand
                            .iter()
                            .filter_map(|&j| spans_us.get(j).copied())
                            .collect();
                        if spans.len() == cand.len() && spans.iter().all(|us| *us > 0.0) {
                            tune.observe(field, &pick.label, ns(spans.iter().sum()));
                        } else if tune_log() {
                            eprintln!(
                                "[tune] diff arm `{}` at {field}: {} of {} spans timed, \
                                 nothing filed",
                                pick.label,
                                spans.iter().filter(|us| **us > 0.0).count(),
                                cand.len()
                            );
                        }
                    }
                }
                self.maybe_adopt(&run, ks);
            }
        }
        // One atomic write per exploration step, a no-op when nothing new
        // was measured.
        tune.save();
    }

    /// Adopt every launch whose persisted windows already hold a qualifying
    /// winner, in one sequential pass. Returns whether the replay memo was
    /// updated.
    ///
    /// Each swap clears exactly the bar [`Self::maybe_adopt`] enforces, but
    /// runs off windows written by past processes, so a prior becomes a plan
    /// at the first replay hit. Later launches replan against the
    /// already-adopted earlier swaps.
    fn adopt_persisted(
        &self,
        graph: &GraphRef,
        roots: &[Id],
        key: ReplayKey,
        incumbent: &Arc<Plan>,
    ) -> bool {
        let tune = &self.inner.tune;
        let (sigs, mut labels) = {
            let g = graph.state().egraph.lock();
            let sigs: Vec<String> = incumbent
                .launches
                .iter()
                .map(|l| launch_signature(&g, l))
                .collect();
            let labels: Vec<String> = (0..incumbent.launches.len())
                .map(|ix| {
                    incumbent_signature(&g, incumbent, ix).unwrap_or_else(|| "base".to_string())
                })
                .collect();
            (sigs, labels)
        };
        let mut current = Arc::clone(incumbent);
        let mut adopted = 0usize;
        // The qualifying winners, from window lookups alone: same window-min
        // bar and MIN_OBS floor as the sequential loop below.
        let winners: Vec<(usize, String)> = (0..sigs.len())
            .filter_map(|j| {
                let (best_name, best) = tune.best(&sigs[j])?;
                if best_name == labels[j] || tune.observations(&sigs[j], &best_name) < MIN_OBS {
                    return None;
                }
                let inc_min = tune.window_min(&sigs[j], &labels[j])?;
                ((best as f64) < inc_min as f64 * (1.0 - TUNE_MARGIN)).then_some((j, best_name))
            })
            .collect();
        // Batch first: every winner composed onto one extraction, one replan,
        // one verify. Each swapped launch must come out running its winner
        // and nothing outside the swapped set may move; anything less falls
        // back to the sequential loop.
        if !winners.is_empty() {
            let batch = {
                let g = graph.state().egraph.lock();
                self.inner
                    .extractor
                    .replan_with_variants(
                        &g,
                        roots,
                        &current,
                        self.inner.cost.as_ref(),
                        autotune_min_macs(),
                        &winners,
                    )
                    .filter(|plan| {
                        batch_aligns(plan, &current, &winners)
                            && winners.iter().all(|(j, name)| {
                                incumbent_signature(&g, plan, *j).as_deref() == Some(name)
                            })
                    })
            };
            if let Some(plan) = batch {
                if tune_log() {
                    for (j, name) in &winners {
                        eprintln!(
                            "[tune] ADOPT(prior) L{j} `{name}` over `{}` for {}: from \
                             persisted windows (batched)",
                            labels[*j], sigs[*j]
                        );
                    }
                }
                adopted = winners.len();
                for (j, name) in &winners {
                    labels[*j] = name.clone();
                }
                current = Arc::new(plan);
            }
        }
        if adopted == 0 {
            for (j, best_name) in winners {
                let inc_label = labels[j].clone();
                let Some(inc_min) = tune.window_min(&sigs[j], &inc_label) else {
                    continue;
                };
                let Some((_, best)) = tune.best(&sigs[j]) else {
                    continue;
                };
                let variants = {
                    let g = graph.state().egraph.lock();
                    self.inner.extractor.launch_variants(
                        &g,
                        roots,
                        &current,
                        j,
                        self.inner.cost.as_ref(),
                        autotune_min_macs(),
                    )
                };
                let Some((label, plan)) = variants
                    .into_iter()
                    .find(|(l, p)| *l == best_name && plans_align(p, &current, j))
                else {
                    continue;
                };
                if tune_log() {
                    eprintln!(
                        "[tune] ADOPT(prior) L{j} `{best_name}` over `{inc_label}` for {}: \
                         window-min {best} ns vs {inc_min} ns, from persisted windows",
                        sigs[j]
                    );
                }
                current = Arc::new(plan);
                labels[j] = label;
                adopted += 1;
            }
        }
        // Second pass: restructuring candidates for the heaviest launches,
        // raced at diff granularity (see `Gran::Diff`) off the persisted
        // windows.
        let mut rounds = 0usize;
        'diff: while rounds < DIFF_SCAN_TOP_K {
            rounds += 1;
            let (sigs, labels, works) = {
                let g = graph.state().egraph.lock();
                let sigs: Vec<String> = current
                    .launches
                    .iter()
                    .map(|l| launch_signature(&g, l))
                    .collect();
                let labels: Vec<String> = (0..current.launches.len())
                    .map(|ix| {
                        incumbent_signature(&g, &current, ix)
                            .unwrap_or_else(|| "base".to_string())
                    })
                    .collect();
                let works: Vec<u64> = (0..current.launches.len())
                    .map(|ix| launch_work(&g, &current, ix))
                    .collect();
                (sigs, labels, works)
            };
            let plan_field = plan_key(&sigs.join(";"));
            let mut order: Vec<usize> = (0..works.len()).collect();
            order.sort_by_key(|&j| (std::cmp::Reverse(works[j]), j));
            for &j in order.iter().take(DIFF_SCAN_TOP_K) {
                let variants = {
                    let g = graph.state().egraph.lock();
                    self.inner.extractor.launch_variants(
                        &g,
                        roots,
                        &current,
                        j,
                        self.inner.cost.as_ref(),
                        autotune_min_macs(),
                    )
                };
                let budget = plan_step_bytes(&current).saturating_add(ARM_EXTRA_BYTES);
                for (label, plan) in variants {
                    if plans_align(&plan, &current, j) {
                        continue;
                    }
                    if plan_step_bytes(&plan) > budget {
                        continue;
                    }
                    let Some((cand, inc)) = plan_sparse_diff(&plan, &current, MAX_DIFF_EDITS)
                    else {
                        continue;
                    };
                    if cand.is_empty() || inc.is_empty() {
                        continue;
                    }
                    let field = diff_field(&plan_field, &sigs, &inc);
                    if tune.observations(&field, &label) < MIN_OBS {
                        continue;
                    }
                    let Some(a) = tune.window_min(&field, &label) else {
                        continue;
                    };
                    let Some(b) = inc
                        .iter()
                        .map(|&t| tune.window_min(&sigs[t], &labels[t]))
                        .sum::<Option<u64>>()
                    else {
                        continue;
                    };
                    if (a as f64) < b as f64 * (1.0 - TUNE_MARGIN) {
                        if tune_log() {
                            eprintln!(
                                "[tune] ADOPT(prior) diff L{j} `{label}` for {field}: \
                                 summed window-min {a} ns vs incumbent {b} ns"
                            );
                        }
                        current = Arc::new(plan);
                        adopted += 1;
                        continue 'diff;
                    }
                }
            }
            break;
        }
        if adopted == 0 {
            return false;
        }
        if tune_log() {
            eprintln!("[tune] adopted {adopted} launch(es) from the persisted prior");
        }
        self.inner.replay.insert(key, (*current).clone());
        true
    }

    /// Adopt the just-run arm if its window minimum beats the incumbent's by
    /// the same hysteresis the cold race uses.
    fn maybe_adopt(&self, run: &ExploreRun, ks: &mut KeyState) {
        let Some(pick) = &run.pick else { return };
        let Some(set) = ks.arms.as_ref() else { return };
        let Some(arm) = set
            .arms
            .iter()
            .find(|a| a.ix == pick.ix && a.label == pick.label && a.plan.is_some())
        else {
            return;
        };
        let (af, al) = set.arm_field(arm);
        let tune = &self.inner.tune;
        if tune.observations(af, al) < MIN_OBS || set.incumbent_obs(tune, arm) < MIN_OBS {
            return;
        }
        let (Some(a), Some(b)) = (tune.window_min(af, al), set.incumbent_min(tune, arm)) else {
            return;
        };
        if (a as f64) < b as f64 * (1.0 - TUNE_MARGIN) {
            if tune_log() {
                eprintln!(
                    "[tune] ADOPT L{} `{}` for {}: window-min {} ns vs incumbent {} ns, \
                     from production samples",
                    pick.ix, al, af, a, b
                );
            }
            if let Some(plan) = &arm.plan {
                self.inner.replay.insert(run.key, (**plan).clone());
            }
            // Everything in the arm set described the displaced incumbent.
            ks.arms = None;
        }
    }
}

impl ExploreState {
    fn key_mut(&mut self, key: ReplayKey) -> &mut KeyState {
        if !self.keys.contains_key(&key) {
            self.keys.insert(key, KeyState::default());
            self.order.push(key);
            while self.order.len() > KEY_CAP {
                let evicted = self.order.remove(0);
                self.keys.remove(&evicted);
            }
        }
        self.keys.get_mut(&key).expect("just inserted")
    }
}
