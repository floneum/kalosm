//! The per-machine tuning cache: what this device has already learned about
//! which kernels are cheap.
//!
//! Records only timing observations. Candidate equivalence is established by
//! construction and tested by conformance; this cache has no correctness
//! verdicts or blacklist. Stale timings can affect performance, never which
//! computations the compiler considers equivalent.
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

/// A candidate's last up-to-[`WINDOW`] timing observations, oldest first.
/// GPU samples are launch nanoseconds; CPU samples are ppm of the base plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    launch: String,
    variant: String,
    window: Vec<u64>,
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
pub const FORMAT: u32 = 7;

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

/// Per-signature timing windows, keyed by candidate label.
type LearnedTable = FxHashMap<String, FxHashMap<String, Vec<u64>>>;
/// Per-plan-signature adopted combination and its measured nanoseconds.
type ComboTable = FxHashMap<String, (Vec<Option<String>>, u64)>;

/// Fold a file at `path` into the in-memory tables.
fn read_tables(path: &Path) -> (LearnedTable, ComboTable) {
    let mut seen: FxHashMap<String, FxHashMap<String, Vec<u64>>> = FxHashMap::default();
    let mut combos: FxHashMap<String, (Vec<Option<String>>, u64)> = FxHashMap::default();
    if let Ok(body) = std::fs::read_to_string(path)
        && let Ok(disk) = serde_json::from_str::<Disk>(&body)
    {
        let (records, stored) = disk.accept();
        for r in records {
            if r.window.is_empty() {
                continue;
            }
            let start = r.window.len().saturating_sub(WINDOW);
            seen.entry(r.launch)
                .or_default()
                .insert(r.variant, r.window[start..].to_vec());
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
    seen: Mutex<LearnedTable>,
    /// `plan signature -> (picks, score)`, the jointly-measured outcome.
    combos: Mutex<ComboTable>,
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

    /// The minimum over the last [`WINDOW`] observations of one candidate.
    pub fn window_min(&self, launch: &str, variant: &str) -> Option<u64> {
        self.seen
            .lock()
            .get(launch)?
            .get(variant)?
            .iter()
            .copied()
            .min()
    }

    /// Observations available for the explorer's least-observed-first policy.
    pub fn observations(&self, launch: &str, variant: &str) -> usize {
        self.seen
            .lock()
            .get(launch)
            .and_then(|e| e.get(variant))
            .map_or(0, Vec::len)
    }

    /// The fastest observed variant and its window-min time.
    pub fn best(&self, launch: &str) -> Option<(String, u64)> {
        self.seen
            .lock()
            .get(launch)?
            .iter()
            .filter_map(|(name, w)| Some((name.clone(), w.iter().copied().min()?)))
            .min_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
    }

    /// Push one timing observation; old minima age out of the bounded window.
    pub fn observe(&self, launch: &str, variant: &str, nanos: u64) {
        let mut seen = self.seen.lock();
        let w = seen
            .entry(launch.to_string())
            .or_default()
            .entry(variant.to_string())
            .or_default();
        w.push(nanos);
        if w.len() > WINDOW {
            w.drain(..w.len() - WINDOW);
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
    /// measured here, so there is nothing left to learn.
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
                Some(w) => {
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
        fresh.sort_by_key(|a| a.1);

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
                        window: learned.clone(),
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

    #[test]
    fn observations_age_out_and_only_rank_existing_candidates() {
        let cache = TuneCache::default();
        cache.observe("launch", "a", 1);
        for _ in 0..WINDOW {
            cache.observe("launch", "a", 100);
        }
        cache.observe("launch", "b", 50);
        cache.observe("launch", "not-in-graph", 0);
        assert_eq!(cache.window_min("launch", "a"), Some(100));
        assert_eq!(cache.observations("launch", "a"), WINDOW);
        let candidates = vec![("a".into(), 0), ("b".into(), 1000)];
        let (ranked, _) = cache.plan_candidates("launch", &candidates);
        assert_eq!(ranked, vec![&candidates[1].0, &candidates[0].0]);
    }

    #[test]
    fn persisted_timings_require_a_matching_format() {
        let path =
            std::env::temp_dir().join(format!("fusor-timing-test-{}.json", std::process::id()));
        let cache = TuneCache {
            path: Some(path.clone()),
            ..TuneCache::default()
        };
        cache.observe("launch", "a", 40);
        cache.observe("launch", "a", 50);
        cache.record_combo("plan", vec![Some("a".into())], 12);
        cache.save();
        let restored = at_path(&path);
        assert_eq!(restored.window_min("launch", "a"), Some(40));
        assert_eq!(restored.observations("launch", "a"), 2);
        assert_eq!(restored.combo("plan"), Some(vec![Some("a".into())]));
        let mut disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        disk["format"] = (FORMAT + 1).into();
        std::fs::write(&path, serde_json::to_string(&disk).unwrap()).unwrap();
        let obsolete = at_path(&path);
        assert!(obsolete.is_empty());
        assert_eq!(obsolete.combo("plan"), None);
        std::fs::remove_file(path).unwrap();
    }
}
