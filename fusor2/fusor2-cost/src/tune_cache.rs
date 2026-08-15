//! The per-machine tuning cache: what this device has already learned about
//! which kernels are cheap.
//!
//! # What it is for
//!
//! Autotuning measures, and measuring costs a kernel run per candidate. Without
//! memory that bill is paid *every process*: measured on this workspace's own
//! benchmark, a cold matmul resolve spent ~2.3 s and a cold attention resolve
//! ~3.1 s re-discovering, from scratch, facts the previous run had already
//! established. That is fine for a training loop, where it amortises over
//! thousands of steps, and unacceptable for anything short-lived.
//!
//! So the tuner writes down what it measured, keyed by device, and the next
//! process starts from it.
//!
//! # Learning, not just caching
//!
//! A pure cache answers "have I seen this exact thing". This answers the more
//! useful question — *which candidates are worth my time* — and gets better
//! the more it is used:
//!
//! * A variant already timed is never timed again. Its score is read back.
//! * A variant this device caught computing a *different function* is never
//!   timed again either, and can never be the incumbent ([`Verdict::Wrong`]).
//! * A variant known to be much slower than the entry's best is not even
//!   built, let alone run ([`SKIP_RATIO`]).
//! * Untried variants are explored a few per resolve ([`EXPLORE_PER_RESOLVE`]),
//!   so a first run is not a full sweep and successive runs converge on the
//!   optimum instead of re-measuring the incumbent.
//!
//! The result is a cost curve that falls with use: run 1 explores a handful,
//! run N applies the winner and times almost nothing.
//!
//! # Why it is safe to trust
//!
//! It never selects a plan on its own. It **orders** candidates and **skips
//! re-timing**; the plan is still built by the extractor and its outputs are
//! still value-checked against the base before adoption. A stale, wrong or
//! corrupt entry costs a worse starting order or a missed candidate — never a
//! wrong answer. That is deliberate: this file is a heuristic store, and the
//! measurement in `Session::autotune` is the authority.
//!
//! # Keyed per machine
//!
//! By `Caps::fingerprint()`, alongside `crate::cache`'s device facts, because a
//! tile that wins on one GPU says nothing about another. A different device
//! reads a different file; an unknown device reads nothing and tunes normally.

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Candidates whose recorded score is worse than `best * SKIP_RATIO` are not
/// rebuilt on later resolves.
///
/// **Deliberately unbounded, and this is now a settled result rather than a
/// pending experiment.** See [`RERACE_PER_RESOLVE`] for the measurement: a
/// per-launch score being *honest* does not make per-launch pruning *sound*,
/// because the plan optimum is not the per-launch argmin.
///
/// Re-measured at 3.0 on the current footing — per-launch GPU spans, per-kernel
/// keys, this device's own converged file — pruning drops 6 of 7 correct
/// candidates on the 2048-cube and 9 of 10 on attention's `q@k^T`, always
/// keeping each launch's own best. Cold collapsed (matmul 1.38 s -> 0.08 s,
/// attention 2.12 s -> 0.15 s) and **attention's median went 2.57 -> 4.52 ms**,
/// the same regression the 1.5x and 3.0x attempts hit when the score was a
/// whole-plan wall time. The cause is different and structural: attention is
/// six launches whose choices interact, and cutting each launch's field to its
/// own argmin denies the coordinate descent the combination that wins. That is
/// the same reason `Combo` records a *joint* measurement rather than assembling
/// per-launch minima.
pub const SKIP_RATIO: f64 = f64::INFINITY;

/// How many never-measured variants one tuning race will spend time on: the
/// top-K of the *cost model's* ordering. A cold signature races 3 candidates,
/// not the full 16 — the model orders, the measurement decides — and the rest
/// of the field is explored later from production samples (see the session's
/// epsilon explorer), where a sample costs one substituted dispatch instead of
/// a `TUNE_RUNS`-deep race.
pub const RACE_TOP_K: usize = 3;

/// Observations one `(launch, variant)` window holds. The stat every decision
/// reads is the **minimum over the window** — timings are noisy upward, so the
/// min is the kernel — and the window is what lets a stale record die: a
/// fossil minimum from an older driver, thermal regime or build is not carried
/// forever, it ages out after `WINDOW` fresh observations and the entry
/// re-earns its rank or loses it. That is the re-race-on-loss decay the
/// fossil semantics want, with no timestamped bookkeeping.
pub const WINDOW: usize = 8;

/// How many already-known variants one resolve re-races, best-scored first.
///
/// **Still unbounded, and the reason is no longer the one it used to be.**
///
/// Originally a score here was a *whole-plan* wall time attributed to a single
/// launch. The tuner is a coordinate descent that carries an incumbent, so the
/// same variant clocked differently depending on when its turn came, and the
/// score was not a property of `(launch, variant)` at all. Four ways of
/// exploiting it were built and measured, and every one traded quality for
/// compile time:
///
/// | policy | cold | attention |
/// |---|---|---|
/// | prune at `best * 1.5` | 55-65 ms | 4.2 ms |
/// | prune at `best * 3.0` | 260-280 ms | 4.2 ms |
/// | re-race best 6 | 280-930 ms | 2.7-5.6 ms, unstable |
/// | **no bound (shipped)** | 1.9-2.3 s | **2.73 ms** |
///
/// That table was read as a record of a *measurement* defect, so both causes
/// were removed — the score became the launch's own GPU span (see [`Record`]),
/// and the key that span is filed under became one that names a kernel rather
/// than a root (see `fusor2_cost::extract::launch_signature`). Both bounds were
/// then re-measured on that footing, and **the table reproduced**:
///
/// | policy, honest per-launch spans | cold matmul / attention | attention median |
/// |---|---|---|
/// | `SKIP_RATIO` 3.0 + re-race 6 | 0.08 s / 0.15 s | 4.52, 4.53 ms |
/// | re-race 6 alone | 0.79 s / 0.51 s | 2.69, **4.55** ms — unstable |
/// | **no bound (shipped)** | 1.37 s / 2.12 s | **2.57, 2.59 ms** |
///
/// So the defect was never the score's honesty. **A launch's candidate field
/// cannot be narrowed on that launch's own score, because the plan optimum is
/// not the per-launch argmin.** Attention is six launches whose choices
/// interact; keeping only each launch's locally-best points denies the descent
/// the combination that wins, and the loss is the same ~1.9 ms whether the
/// number doing the narrowing was noise or a perfectly measured kernel span.
/// This is the same fact `Combo` exists for.
///
/// A bound here is therefore not blocked on better *measurement*. It is blocked
/// on a candidate ordering that is a property of the plan rather than of one
/// launch — a joint sweep, or a cheap model of the interaction.
pub const RERACE_PER_RESOLVE: usize = usize::MAX;

/// What this device learned about one `(launch, variant)` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Reproduced the base plan's values, in the unit [`Record`] documents.
    Ran(u64),
    /// Measured, and its outputs disagreed with the base plan's.
    ///
    /// **Deliberately not stored as a time.** A member that computes a
    /// different function skips the work the right one must do, so it is
    /// usually the *fastest* thing in the e-class: this device's own file had
    /// `Sgemv(chunk 2, vector 4)` at 14_875 ns against a correct best of
    /// 2_899_291 ns on the 2048-cube, which is 1,155 TFLOP/s on a 13 TFLOP/s
    /// part. Filed as a duration it becomes [`TuneCache::best`], and every
    /// finite [`SKIP_RATIO`] then prunes the *correct* candidates instead.
    Wrong,
}

/// One learned `(launch, variant)` pair: its observation window.
///
/// `window` is `None` for a variant that disagreed with the base — see
/// [`Verdict`] — and otherwise holds up to [`WINDOW`] samples, oldest first,
/// each meaning one of two things, which [`FORMAT`] keeps out of the same
/// file:
///
/// * **On a device that can time kernels** — this launch's own GPU span, in
///   absolute nanoseconds, taken from the resolve's timestamp query set. Host
///   cost, submission and inter-pass gaps are all excluded, so it is a property
///   of the launch and comparable across processes: `launch_signature` pins
///   family, dtype and shape, and the file is keyed by device.
/// * **On a device with no kernel timer (the CPU target)** — parts-per-million
///   of the base plan's time in the pass that measured it, so 1_000_000 means
///   "no better than the base". A whole-plan wall clock is not comparable
///   between passes; the ratio divides the context out, and it is the best a
///   host clock supports.
///
/// The two units never share a file: a device that *can* time kernels but did
/// not time a particular plan records nothing for it.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    launch: String,
    variant: String,
    /// `None` for a variant that disagreed with the base; see [`Verdict`].
    window: Option<Vec<u64>>,
}

/// What is held in memory for one `(launch, variant)` pair.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Learned {
    /// This device caught the variant computing a different function.
    /// Absorbing within a process — see [`TuneCache::record`].
    Wrong,
    /// The last up-to-[`WINDOW`] observations, oldest first. Never empty.
    Window(Vec<u64>),
}

impl Learned {
    fn verdict(&self) -> Verdict {
        match self {
            Learned::Wrong => Verdict::Wrong,
            Learned::Window(w) => {
                Verdict::Ran(w.iter().copied().min().unwrap_or(u64::MAX))
            }
        }
    }
}

/// A whole-plan outcome: the per-launch variants that were fastest **when
/// measured together**.
///
/// Per-launch minima do not compose. Each `Record` above is scored in the
/// context the coordinate descent happened to be in when its turn came, so
/// taking the arg-min of each launch independently assembles a configuration
/// that was never actually run — measured, that assembled plan clocked ~4.2 ms
/// on attention where the descent's own winner was 2.75 ms. This is the
/// combination that really won, stored whole.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Combo {
    plan: String,
    /// One entry per launch, in launch order. `None` means "the base plan's
    /// own choice", which is not a variant and has no label.
    picks: Vec<Option<String>>,
    score: u64,
}

/// The unit [`Record::nanos`] is written in.
///
/// Bumped whenever that meaning changes, because the old and new units are the
/// same order of magnitude and would sort against each other in silence: a
/// format-1 file holds ~1e6 parts-per-million of a base plan, a format-2 file
/// holds a 2 ms kernel as 2_000_000 ns. A file at any other format is read as
/// an empty cache, which costs one tuning pass and never a wrong ordering.
///
/// Bumped to 3: format 2 filed a variant that produced *wrong values* as if it
/// were simply fast, so `best()` in such a file is frequently a kernel that
/// computes a different function, and nothing in a format-2 file distinguishes
/// the two.
///
/// Bumped to 4: format 3's keys were a function of the launch *root* only, so
/// launches with different reduced extents or different fused bodies shared an
/// entry and [`TuneCache::record`]'s minimum-merge filed the cheapest one's
/// span for all of them. Those records name kernels that cannot be identified.
///
/// Bumped to 5: a record became an observation *window* rather than a single
/// all-time minimum, so production samples can both feed the prior and age a
/// stale minimum out. A format-4 `nanos` cannot be told apart from a window
/// of one, but the semantics differ (it was a min over an unbounded past), so
/// the file is dropped like every other stale format.
pub const FORMAT: u32 = 5;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Disk {
    #[serde(default)]
    format: u32,
    #[serde(default)]
    records: Vec<Record>,
    #[serde(default)]
    combos: Vec<Combo>,
}

impl Disk {
    /// Split a parsed file into `(records, combos)`, empty unless it was
    /// written in the unit this build reads.
    fn accept(self) -> (Vec<Record>, Vec<Combo>) {
        if self.format == FORMAT {
            (self.records, self.combos)
        } else {
            (Vec::new(), Vec::new())
        }
    }
}

/// Fold a file at `path` into the in-memory tables.
fn read_tables(
    path: &Path,
) -> (
    FxHashMap<String, FxHashMap<String, Learned>>,
    FxHashMap<String, (Vec<Option<String>>, u64)>,
) {
    let mut seen: FxHashMap<String, FxHashMap<String, Learned>> = FxHashMap::default();
    let mut combos: FxHashMap<String, (Vec<Option<String>>, u64)> = FxHashMap::default();
    if let Ok(body) = std::fs::read_to_string(path)
        && let Ok(disk) = serde_json::from_str::<Disk>(&body)
    {
        let (records, stored) = disk.accept();
        for r in records {
            let learned = match r.window {
                Some(w) if !w.is_empty() => {
                    let start = w.len().saturating_sub(WINDOW);
                    Learned::Window(w[start..].to_vec())
                }
                // An empty window is not a thing `save` writes; read it as
                // unmeasured rather than inventing a sample.
                Some(_) => continue,
                None => Learned::Wrong,
            };
            seen.entry(r.launch).or_default().insert(r.variant, learned);
        }
        for c in stored {
            combos.insert(c.plan, (c.picks, c.score));
        }
    }
    (seen, combos)
}

/// What this device has learned. Cheap to clone-free share; all mutation is
/// behind one lock because a resolve is already serialized.
#[derive(Debug, Default)]
pub struct TuneCache {
    path: Option<PathBuf>,
    /// `launch signature -> variant signature -> observation window`.
    seen: Mutex<FxHashMap<String, FxHashMap<String, Learned>>>,
    /// `plan signature -> (picks, score)`, the jointly-measured outcome.
    combos: Mutex<FxHashMap<String, (Vec<Option<String>>, u64)>>,
    /// Set when anything changed, so an unchanged process writes nothing.
    dirty: Mutex<bool>,
}

/// `$XDG_CACHE_HOME/fusor2/tune/<fingerprint>.json`, or `$HOME/.cache/...`.
/// `FUSOR2_TUNE_CACHE` overrides the whole path; `FUSOR2_NO_TUNE_CACHE`
/// disables persistence entirely, which is the A/B switch that shows whether a
/// result came from the cache or from this run.
pub fn cache_path(caps_fingerprint: u64) -> Option<PathBuf> {
    if std::env::var_os("FUSOR2_NO_TUNE_CACHE").is_some() {
        return None;
    }
    if let Some(p) = std::env::var_os("FUSOR2_TUNE_CACHE").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(p));
    }
    let base = if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(xdg)
    } else {
        PathBuf::from(std::env::var_os("HOME").filter(|v| !v.is_empty())?)
            .join(".cache")
    };
    Some(
        base.join("fusor2")
            .join("tune")
            .join(format!("{caps_fingerprint:016x}.json")),
    )
}

impl TuneCache {
    /// Read this device's file, or start empty. A malformed or unreadable file
    /// is an empty cache, never an error: the worst it can cost is a tuning
    /// pass this process would have done anyway.
    pub fn load(caps_fingerprint: u64) -> Self {
        let path = cache_path(caps_fingerprint);
        let (seen, combos) = match &path {
            Some(p) => read_tables(p),
            None => Default::default(),
        };
        Self {
            path,
            seen: Mutex::new(seen),
            combos: Mutex::new(combos),
            dirty: Mutex::new(false),
        }
    }

    /// How many launches this device has learned anything about.
    pub fn len(&self) -> usize {
        self.seen.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// What this device learned about one candidate, if it has ever been
    /// measured here. A `Ran` carries the window minimum.
    pub fn known(&self, launch: &str, variant: &str) -> Option<Verdict> {
        Some(self.seen.lock().get(launch)?.get(variant)?.verdict())
    }

    /// The minimum over the last [`WINDOW`] observations of one candidate.
    /// `None` for a never-measured or wrong-valued variant.
    pub fn window_min(&self, launch: &str, variant: &str) -> Option<u64> {
        match self.seen.lock().get(launch)?.get(variant)? {
            Learned::Window(w) => w.iter().copied().min(),
            Learned::Wrong => None,
        }
    }

    /// How many observations one candidate's window currently holds. What the
    /// explorer's "least-observed arm first" policy reads.
    pub fn observations(&self, launch: &str, variant: &str) -> usize {
        match self.seen.lock().get(launch).and_then(|e| e.get(variant)) {
            Some(Learned::Window(w)) => w.len(),
            _ => 0,
        }
    }

    /// The fastest variant recorded for this launch **that reproduced the
    /// base's values**, and its window-min time. A wrong answer has no time,
    /// so it can never be the incumbent.
    pub fn best(&self, launch: &str) -> Option<(String, u64)> {
        let seen = self.seen.lock();
        seen.get(launch)?
            .iter()
            .filter_map(|(name, l)| match l {
                Learned::Window(w) => Some((name.clone(), w.iter().copied().min()?)),
                Learned::Wrong => None,
            })
            .min_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
    }

    /// Push one production observation into the candidate's window.
    pub fn observe(&self, launch: &str, variant: &str, nanos: u64) {
        self.record(launch, variant, Verdict::Ran(nanos));
    }

    pub fn record(&self, launch: &str, variant: &str, verdict: Verdict) {
        let mut seen = self.seen.lock();
        let entry = seen.entry(launch.to_string()).or_default();
        match (entry.get_mut(variant), verdict) {
            // Wrongness is a property of the kernel, not of the run: once this
            // device has seen a variant disagree, no later timing of it means
            // anything within this process, so `Wrong` is absorbing. (Across
            // processes it is a fossil of a since-fixed compiler bug and
            // re-races as unmeasured — see `plan_candidates`.)
            (Some(slot), Verdict::Wrong) => *slot = Learned::Wrong,
            (Some(Learned::Wrong), Verdict::Ran(_)) => {}
            // The window keeps the most recent `WINDOW` observations; every
            // reader takes the min over it. A slow sample is contention, a
            // fast one is the kernel — but a fast sample that `WINDOW` newer
            // runs never reproduce ages out, which is how a stale record
            // loses to fresh observations instead of pinning forever.
            (Some(Learned::Window(w)), Verdict::Ran(ns)) => {
                w.push(ns);
                if w.len() > WINDOW {
                    let excess = w.len() - WINDOW;
                    w.drain(..excess);
                }
            }
            (None, Verdict::Wrong) => {
                entry.insert(variant.to_string(), Learned::Wrong);
            }
            (None, Verdict::Ran(ns)) => {
                entry.insert(variant.to_string(), Learned::Window(vec![ns]));
            }
        }
        *self.dirty.lock() = true;
    }


    /// The jointly-measured winning combination for a whole plan.
    pub fn combo(&self, plan: &str) -> Option<Vec<Option<String>>> {
        self.combos.lock().get(plan).map(|(p, _)| p.clone())
    }

    /// Record a combination, keeping the better score.
    pub fn record_combo(&self, plan: &str, picks: Vec<Option<String>>, score: u64) {
        let mut combos = self.combos.lock();
        match combos.get(plan) {
            Some((_, best)) if *best <= score => return,
            _ => {}
        }
        combos.insert(plan.to_string(), (picks, score));
        *self.dirty.lock() = true;
    }

    /// Whether every candidate offered for this launch has already been
    /// measured here, so there is nothing left to learn. A variant proven to
    /// compute the wrong function counts: it is still something this device has
    /// nothing left to learn about.
    ///
    /// This is what turns the cache from "cheaper tuning" into "no tuning".
    /// While an entry is still exploring, the tuner races candidates and picks
    /// by *this run's* clock, which is noisy: measured, that re-selection
    /// settled attention at ~3.0 ms when the accumulated best was 2.58 ms. Once
    /// there is nothing new to try, the accumulated minimum over every past run
    /// is a far better estimate than one fresh sample, so the tuner should
    /// apply it rather than re-derive it.
    pub fn converged(&self, launch: &str, candidates: &[String]) -> bool {
        let seen = self.seen.lock();
        let Some(entry) = seen.get(launch) else {
            return false;
        };
        !candidates.is_empty() && candidates.iter().all(|c| entry.contains_key(c.as_str()))
    }

    /// Split candidates into what to race and what to skip, best prior first.
    ///
    /// Each candidate arrives with the **cost model's prior** for the plan it
    /// denotes, in picoseconds. Returns `(to_measure, skipped)`. Ordering is:
    /// measured variants by their window minimum (re-confirm the incumbent
    /// first), then never-measured ones by the model's prior, capped at
    /// [`RACE_TOP_K`] — so a cold signature races the model's top-3 picks
    /// rather than the full field, and the rest of the field is left to the
    /// production explorer, which pays one substituted dispatch per sample
    /// instead of a race. Ties break by name so a run is reproducible.
    pub fn plan_candidates<'a>(
        &self,
        launch: &str,
        candidates: &'a [(String, u64)],
    ) -> (Vec<&'a String>, Vec<&'a String>) {
        let best = self.best(launch).map(|(_, ns)| ns);
        let seen = self.seen.lock();
        let entry = seen.get(launch);

        let mut known: Vec<(&'a String, u64)> = Vec::new();
        let mut fresh: Vec<(&'a String, u64)> = Vec::new();
        let mut skipped: Vec<&'a String> = Vec::new();

        for (c, prior) in candidates {
            match entry.and_then(|e| e.get(c.as_str())) {
                // A recorded `Wrong` names a **compiler bug that has since
                // been fixed** — production halts on divergence before any
                // verdict is written (see `Session::autotune`), so an entry
                // can only be a fossil of an older build. Skipping on it
                // would silently pin the repaired kernel out of selection
                // forever; the entry is treated as unmeasured instead.
                Some(Learned::Wrong) => fresh.push((c, *prior)),
                Some(Learned::Window(w)) => {
                    let ns = w.iter().copied().min().unwrap_or(u64::MAX);
                    let hopeless = best.is_some_and(|b| ns as f64 > b as f64 * SKIP_RATIO);
                    if hopeless {
                        skipped.push(c);
                    } else {
                        known.push((c, ns));
                    }
                }
                None => fresh.push((c, *prior)),
            }
        }
        known.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        // Stable by prior only: candidates arrive in the enumerator's offer
        // order (round-robin over *belief-ordered* schedule domains), which
        // is exactly as reproducible as a name sort and, unlike one, means
        // something on a tie. The model prices every cell of a family
        // identically, so with a name tie-break the race's one fresh sgemv
        // slot went to whichever cell's `Debug` string sorted first —
        // `chunk: 4` before `chunk: 8` — and the domain's believed-best cell
        // was never cold-raced at all.
        fresh.sort_by(|a, b| a.1.cmp(&b.1));

        let mut out: Vec<&'a String> = known
            .into_iter()
            .take(RERACE_PER_RESOLVE)
            .map(|(c, _)| c)
            .collect();
        for (c, _) in fresh.into_iter().take(RACE_TOP_K) {
            out.push(c);
        }
        (out, skipped)
    }

    /// Persist, atomically, if anything changed. Best-effort: a cache that
    /// cannot be written is still a correct cache for this process.
    pub fn save(&self) {
        if !*self.dirty.lock() {
            return;
        }
        let Some(path) = &self.path else { return };
        let disk = {
            let seen = self.seen.lock();
            let mut records: Vec<Record> = seen
                .iter()
                .flat_map(|(l, vs)| {
                    vs.iter().map(move |(v, learned)| Record {
                        launch: l.clone(),
                        variant: v.clone(),
                        window: match learned {
                            Learned::Window(w) => Some(w.clone()),
                            Learned::Wrong => None,
                        },
                    })
                })
                .collect();
            // Sorted so the file is stable across runs and diffable by hand.
            records.sort_by(|a, b| {
                a.launch
                    .cmp(&b.launch)
                    .then_with(|| a.variant.cmp(&b.variant))
            });
            let mut combos: Vec<Combo> = self
                .combos
                .lock()
                .iter()
                .map(|(plan, (picks, score))| Combo {
                    plan: plan.clone(),
                    picks: picks.clone(),
                    score: *score,
                })
                .collect();
            combos.sort_by(|a, b| a.plan.cmp(&b.plan));
            Disk {
                format: FORMAT,
                records,
                combos,
            }
        };
        let Ok(body) = serde_json::to_string_pretty(&disk) else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        // Write-then-rename: a crash mid-write leaves the old cache, not a
        // truncated one that would parse as empty and silently re-tune.
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, body).is_ok() && std::fs::rename(&tmp, path).is_ok() {
            *self.dirty.lock() = false;
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// Round-trip a cache through a specific path. Used by the tests and by
/// anything that wants a scratch cache rather than the device's.
pub fn at_path(path: &Path) -> TuneCache {
    let (seen, combos) = read_tables(path);
    TuneCache {
        path: Some(path.to_path_buf()),
        seen: Mutex::new(seen),
        combos: Mutex::new(combos),
        dirty: Mutex::new(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("fusor2-tune-test-{name}.json"))
    }

    #[test]
    fn records_survive_a_round_trip() {
        let p = scratch("roundtrip");
        let _ = std::fs::remove_file(&p);
        let c = at_path(&p);
        c.record("L", "v1", Verdict::Ran(100));
        c.record("L", "v2", Verdict::Ran(250));
        c.save();

        let d = at_path(&p);
        assert_eq!(d.known("L", "v1"), Some(Verdict::Ran(100)));
        assert_eq!(d.known("L", "v2"), Some(Verdict::Ran(250)));
        assert_eq!(d.best("L"), Some(("v1".to_string(), 100)));
        let _ = std::fs::remove_file(&p);
    }

    /// The fastest observation in the window wins: timings are noisy upward,
    /// so a later slow sample must not displace a good one that is still
    /// inside the window.
    #[test]
    fn keeps_the_fastest_observation() {
        let c = TuneCache::default();
        c.record("L", "v", Verdict::Ran(100));
        c.record("L", "v", Verdict::Ran(400));
        assert_eq!(c.known("L", "v"), Some(Verdict::Ran(100)));
        c.record("L", "v", Verdict::Ran(60));
        assert_eq!(c.known("L", "v"), Some(Verdict::Ran(60)));
        assert_eq!(c.window_min("L", "v"), Some(60));
        assert_eq!(c.observations("L", "v"), 3);
    }

    /// The decay: a minimum that `WINDOW` fresh observations never reproduce
    /// ages out, so a stale record loses to fresh production samples instead
    /// of pinning a fossil ranking forever.
    #[test]
    fn a_stale_minimum_ages_out_of_the_window() {
        let c = TuneCache::default();
        c.observe("L", "v", 100);
        for _ in 0..WINDOW - 1 {
            c.observe("L", "v", 500);
        }
        // The fast sample is still the newest window's oldest entry.
        assert_eq!(c.window_min("L", "v"), Some(100));
        c.observe("L", "v", 500);
        // One more observation pushes it out.
        assert_eq!(c.window_min("L", "v"), Some(500));
        assert_eq!(c.observations("L", "v"), WINDOW);
    }

    /// Windows survive the file whole, not just their minimum: the next
    /// process continues the same decay instead of restarting it.
    #[test]
    fn windows_round_trip_whole() {
        let p = scratch("window");
        let _ = std::fs::remove_file(&p);
        let c = at_path(&p);
        c.observe("L", "v", 300);
        c.observe("L", "v", 100);
        c.observe("L", "v", 200);
        c.save();
        let d = at_path(&p);
        assert_eq!(d.observations("L", "v"), 3);
        assert_eq!(d.window_min("L", "v"), Some(100));
        // Continue the window: WINDOW slower samples must displace the 100.
        for _ in 0..WINDOW {
            d.observe("L", "v", 400);
        }
        assert_eq!(d.window_min("L", "v"), Some(400));
        let _ = std::fs::remove_file(&p);
    }

    /// A `Wrong` record never ranks — filed as a duration it would become
    /// `best()` and prune the correct candidates — but it never *pins*
    /// either. Production halts on divergence before writing a verdict, so
    /// a stored `Wrong` can only be a fossil of a compiler bug that has
    /// since been fixed; the variant is re-explored like an unmeasured one
    /// rather than skipped forever.
    #[test]
    fn a_wrong_variant_is_excluded_not_ranked() {
        let c = TuneCache::default();
        c.record("L", "fast_but_wrong", Verdict::Wrong);
        c.record("L", "slow_and_right", Verdict::Ran(2_899_291));
        assert_eq!(c.best("L"), Some(("slow_and_right".to_string(), 2_899_291)));
        let cands: Vec<(String, u64)> = [("fast_but_wrong", 10), ("slow_and_right", 20)]
            .iter()
            .map(|(s, p)| (s.to_string(), *p))
            .collect();
        let (run, skip) = c.plan_candidates("L", &cands);
        assert!(skip.is_empty());
        // Ranked entries lead (re-confirm the incumbent first), the fossil
        // re-races with the unmeasured tail.
        assert_eq!(
            run,
            vec![&"slow_and_right".to_string(), &"fast_but_wrong".to_string()]
        );
    }

    /// The verdict survives the file, so a later process does not re-race a
    /// kernel this device already caught computing a different function.
    #[test]
    fn a_wrong_verdict_round_trips() {
        let p = scratch("verdict");
        let _ = std::fs::remove_file(&p);
        let c = at_path(&p);
        c.record("L", "wrong", Verdict::Wrong);
        c.record("L", "right", Verdict::Ran(500));
        c.save();
        let d = at_path(&p);
        assert_eq!(d.known("L", "wrong"), Some(Verdict::Wrong));
        assert_eq!(d.best("L"), Some(("right".to_string(), 500)));
        let _ = std::fs::remove_file(&p);
    }

    /// The learning behaviour: a warm entry re-races its best-known first,
    /// races a bounded number of new candidates, and excludes nothing.
    #[test]
    fn reraces_best_first_and_bounds_exploration() {
        let c = TuneCache::default();
        c.record("L", "good", Verdict::Ran(100));
        c.record("L", "awful", Verdict::Ran(1_000)); // > 100 * SKIP_RATIO
        let cands: Vec<(String, u64)> = ["good", "awful", "n1", "n2", "n3", "n4", "n5", "n6"]
            .iter()
            .map(|s| (s.to_string(), 50))
            .collect();
        let (run, skip) = c.plan_candidates("L", &cands);
        assert!(skip.is_empty(), "score-based exclusion is unsound: {skip:?}");
        assert_eq!(run[0], &"good".to_string(), "incumbent first: {run:?}");
        assert_eq!(
            run.len(),
            2 + RACE_TOP_K,
            "every known candidate stays eligible; only the fresh tail is bounded: {run:?}"
        );
    }

    /// A cold signature races exactly the model's top-K picks, in the
    /// model's own order — not the full field, and not name order.
    #[test]
    fn a_cold_entry_races_the_models_top_k() {
        let c = TuneCache::default();
        let cands: Vec<(String, u64)> = (0..10)
            .map(|i| (format!("v{i}"), 1_000 - i as u64 * 100))
            .collect();
        let (run, skip) = c.plan_candidates("unseen", &cands);
        assert!(skip.is_empty());
        // Cheapest priors are the highest indices.
        assert_eq!(
            run,
            vec![&"v9".to_string(), &"v8".to_string(), &"v7".to_string()],
            "the model orders, the measurement decides"
        );
        assert_eq!(run.len(), RACE_TOP_K);
    }

    /// Convergence is "nothing left to try", which is what lets the tuner
    /// apply the accumulated winner instead of re-racing noisy samples.
    #[test]
    fn converged_only_when_every_candidate_is_known() {
        let c = TuneCache::default();
        let cands: Vec<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        assert!(!c.converged("L", &cands));
        c.record("L", "a", Verdict::Ran(900_000));
        assert!(!c.converged("L", &cands));
        c.record("L", "b", Verdict::Ran(800_000));
        assert!(c.converged("L", &cands));
        assert_eq!(c.best("L"), Some(("b".to_string(), 800_000)));
    }

    /// The whole-plan combination round-trips and keeps the better score,
    /// because per-launch minima do not compose into a plan anyone ran.
    #[test]
    fn combos_round_trip_and_keep_the_better_score() {
        let p = scratch("combo");
        let _ = std::fs::remove_file(&p);
        let c = at_path(&p);
        c.record_combo("P", vec![Some("a".into()), None], 900_000);
        c.record_combo("P", vec![Some("b".into()), None], 950_000); // worse, ignored
        c.record_combo("P", vec![Some("c".into()), None], 800_000); // better, kept
        c.save();
        let d = at_path(&p);
        assert_eq!(d.combo("P"), Some(vec![Some("c".into()), None]));
        let _ = std::fs::remove_file(&p);
    }

    /// A file written in an older unit is dropped rather than compared
    /// against: ppm-of-base and absolute kernel nanoseconds are the same order
    /// of magnitude, so mixing them would silently reorder every candidate.
    #[test]
    fn a_stale_format_reads_as_empty() {
        let p = scratch("format");
        std::fs::write(
            &p,
            format!(
                "{{\"format\":{},\"records\":[{{\"launch\":\"L\",\"variant\":\"v\",\"nanos\":7}}]}}",
                FORMAT - 1
            ),
        )
        .unwrap();
        assert!(at_path(&p).is_empty(), "an old unit must not be read back");

        // The current unit round-trips through the same path.
        let c = at_path(&p);
        c.record("L", "v", Verdict::Ran(7));
        c.save();
        assert_eq!(at_path(&p).known("L", "v"), Some(Verdict::Ran(7)));
        let _ = std::fs::remove_file(&p);
    }

    /// A corrupt file is an empty cache, never a panic and never an error.
    #[test]
    fn a_corrupt_file_reads_as_empty() {
        let p = scratch("corrupt");
        std::fs::write(&p, "{ not json").unwrap();
        assert!(at_path(&p).is_empty());
        let _ = std::fs::remove_file(&p);
    }
}
