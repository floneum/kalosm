//! The per-machine tuning cache: what this device has already learned about
//! which kernels are cheap.
//!
//! The tuner writes down what it measured, keyed by device, and the next
//! process starts from it. A variant already timed is never timed again; a
//! variant caught computing a different function can never be the incumbent
//! ([`Verdict::Wrong`]); untried variants are explored a few per resolve.
//!
//! It never selects a plan on its own — it orders candidates and skips
//! re-timing; plans are still built by the extractor and value-checked against
//! the base before adoption. A stale, wrong or corrupt entry costs a worse
//! starting order or a missed candidate, never a wrong answer.
//!
//! Keyed by `Caps::fingerprint()`: a different device reads a different file;
//! an unknown device reads nothing and tunes normally.

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Candidates whose recorded score is worse than `best * SKIP_RATIO` are not
/// rebuilt on later resolves.
///
/// Unbounded: a launch's candidate field cannot be narrowed on that launch's
/// own score, because the plan optimum is not the per-launch argmin —
/// multi-launch plans have interdependent choices.
pub const SKIP_RATIO: f64 = f64::INFINITY;

/// How many never-measured variants one tuning race will spend time on: the
/// top-K of the cost model's ordering. The rest of the field is explored later
/// from production samples via the session's epsilon explorer.
pub const RACE_TOP_K: usize = 3;

/// Observations one `(launch, variant)` window holds. Every decision reads the
/// minimum over the window — timings are noisy upward, so the min is the
/// kernel — and a stale minimum ages out after `WINDOW` fresh observations.
pub const WINDOW: usize = 8;

/// How many already-known variants one resolve re-races, best-scored first.
///
/// Unbounded: narrowing each launch's field independently denies the descent
/// the combination that actually wins.
pub const RERACE_PER_RESOLVE: usize = usize::MAX;

/// What this device learned about one `(launch, variant)` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Reproduced the base plan's values, in the unit the stored record documents.
    Ran(u64),
    /// Measured, and its outputs disagreed with the base plan's.
    ///
    /// Not stored as a time: a member that computes a different function skips
    /// work, so it is usually the fastest thing in the e-class. Filed as a
    /// duration it would become [`TuneCache::best`] and prune the correct
    /// candidates.
    Wrong,
}

/// One learned `(launch, variant)` pair: its observation window.
///
/// `window` is `None` for a variant that disagreed with the base — see
/// [`Verdict`] — and otherwise holds up to [`WINDOW`] samples, oldest first.
/// On a device that can time kernels a sample is the launch's GPU span in
/// absolute nanoseconds; on a device with no kernel timer (the CPU target) it
/// is parts-per-million of the base plan's time in the pass that measured it.
/// The two units never share a file ([`FORMAT`]).
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
            Learned::Window(w) => Verdict::Ran(w.iter().copied().min().unwrap_or(u64::MAX)),
        }
    }
}

/// A whole-plan outcome: the per-launch variants that were fastest when
/// measured together.
///
/// Per-launch minima do not compose — each `Record` is scored in whatever
/// context the coordinate descent was in when its turn came — so the winning
/// combination is stored whole.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Combo {
    plan: String,
    /// One entry per launch, in launch order. `None` means "the base plan's
    /// own choice", which is not a variant and has no label.
    picks: Vec<Option<String>>,
    score: u64,
}

/// The on-disk format version. A file at a different format is read as an
/// empty cache: mismatch is never a wrong ordering, only a re-tuning pass.
pub const FORMAT: u32 = 6;

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

/// What this device has learned. All mutation is behind one lock because a
/// resolve is already serialized.
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

/// `$XDG_CACHE_HOME/fusor/tune/<fingerprint>.json`, or `$HOME/.cache/...`.
/// `FUSOR_TUNE_CACHE` overrides the whole path; `FUSOR_NO_TUNE_CACHE`
/// disables persistence entirely.
pub fn cache_path(caps_fingerprint: u64) -> Option<PathBuf> {
    if std::env::var_os("FUSOR_NO_TUNE_CACHE").is_some() {
        return None;
    }
    if let Some(p) = std::env::var_os("FUSOR_TUNE_CACHE").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(p));
    }
    let base = if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(xdg)
    } else {
        PathBuf::from(std::env::var_os("HOME").filter(|v| !v.is_empty())?).join(".cache")
    };
    Some(
        base.join("fusor")
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
            // `Wrong` is absorbing within a process: once this device has seen
            // a variant disagree, no later timing of it means anything. Across
            // processes it re-races as unmeasured — see `plan_candidates`.
            (Some(slot), Verdict::Wrong) => *slot = Learned::Wrong,
            (Some(Learned::Wrong), Verdict::Ran(_)) => {}
            // The window keeps the most recent `WINDOW` observations; a fast
            // sample that `WINDOW` newer runs never reproduce ages out.
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
    /// compute the wrong function counts.
    ///
    /// Once there is nothing new to try, the accumulated minimum over every
    /// past run is a better estimate than one fresh noisy sample, so the tuner
    /// applies it rather than re-deriving it.
    pub fn converged(&self, launch: &str, candidates: &[String]) -> bool {
        let seen = self.seen.lock();
        let Some(entry) = seen.get(launch) else {
            return false;
        };
        !candidates.is_empty() && candidates.iter().all(|c| entry.contains_key(c.as_str()))
    }

    /// Split candidates into what to race and what to skip, best prior first.
    ///
    /// Each candidate arrives with the cost model's prior for the plan it
    /// denotes, in picoseconds. Returns `(to_measure, skipped)`. Ordering:
    /// measured variants by their window minimum (re-confirm the incumbent
    /// first), then never-measured ones by the model's prior, capped at
    /// [`RACE_TOP_K`]. Ties break by name so a run is reproducible.
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
                // A recorded `Wrong` can only be a fossil of an older build —
                // production halts on divergence before any verdict is written
                // (see `Session::autotune`) — so the entry is treated as
                // unmeasured rather than pinning the repaired kernel out of
                // selection forever.
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
        // order (round-robin over belief-ordered schedule domains), so a tie
        // keeps the domain's believed-best cell first.
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
