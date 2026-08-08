//! The per-machine tuning cache: what this device has already learned about
//! which kernels are cheap. The tuner writes down what it measured and the next
//! process starts from it.
//!
//! * A variant already timed is never timed again; its score is read back.
//! * A variant caught computing a different function is never timed again and
//!   can never be the incumbent ([`Verdict::Wrong`]).
//! * A variant much slower than the entry's best is not built ([`SKIP_RATIO`]).
//! * Untried variants are explored a few per resolve ([`EXPLORE_PER_RESOLVE`]).
//!
//! It never selects a plan: it orders candidates and skips re-timing, while the
//! plan is built by the extractor and value-checked against the base before
//! adoption. A stale, wrong or corrupt entry costs a worse starting order or a
//! missed candidate, never a wrong answer.
//!
//! Keyed by `Caps::fingerprint()`, alongside `crate::cache`'s device facts. A
//! different device reads a different file; an unknown device tunes normally.

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Candidates whose recorded score is worse than `best * SKIP_RATIO` are not
/// rebuilt on later resolves.
///
/// Infinite, so nothing is pruned on score: a launch's candidate field cannot
/// be narrowed on that launch's own score, because the plan optimum is not the
/// per-launch argmin. That is also why `Combo` records a joint measurement
/// rather than assembling per-launch minima.
pub const SKIP_RATIO: f64 = f64::INFINITY;

/// How many never-before-seen variants one resolve will spend time on, so a
/// first run is not a full sweep.
pub const EXPLORE_PER_RESOLVE: usize = 4;

/// How many already-known variants one resolve re-races, best-scored first.
///
/// Unbounded, for the same reason [`SKIP_RATIO`] is infinite: the plan optimum
/// is not the per-launch argmin. A bound here needs a candidate ordering that
/// is a property of the plan rather than of one launch.
pub const RERACE_PER_RESOLVE: usize = usize::MAX;

/// What this device learned about one `(launch, variant)` pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Reproduced the base plan's values, in the unit [`Record`] documents.
    Ran(u64),
    /// Measured, and its outputs disagreed with the base plan's.
    ///
    /// Not stored as a time: a member computing a different function skips the
    /// work the right one must do, so it is usually the fastest thing in the
    /// e-class. Filed as a duration it would become [`TuneCache::best`] and a
    /// finite [`SKIP_RATIO`] would prune the correct candidates instead.
    Wrong,
}

/// One measured `(launch, variant)` pair.
///
/// `nanos` is `None` for a variant that disagreed with the base (see
/// [`Verdict`]) and otherwise carries one of two units, which [`FORMAT`] keeps
/// out of the same file: with a kernel timer, this launch's own GPU span in
/// absolute nanoseconds, excluding host cost, submission and inter-pass gaps;
/// without one (the CPU target), parts-per-million of the base plan's time in
/// the pass that measured it, so 1_000_000 means no better than the base.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    launch: String,
    variant: String,
    /// `None` for a variant that disagreed with the base; see [`Verdict`].
    nanos: Option<u64>,
}

/// A whole-plan outcome: the per-launch variants that were fastest when
/// measured together.
///
/// Per-launch minima do not compose — each `Record` above is scored in
/// whatever context the coordinate descent was in when its turn came — so the
/// winning combination is stored whole.
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
/// Bump whenever that meaning, or the key a span is filed under, changes:
/// different units are the same order of magnitude and would sort against each
/// other in silence. A file at any other format reads as an empty cache.
pub const FORMAT: u32 = 4;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct Disk {
    #[serde(default)]
    format: u32,
    #[serde(default)]
    records: Vec<Record>,
    #[serde(default)]
    combos: Vec<Combo>,
}

/// Fold a file at `path` into the in-memory tables. A file written in any
/// other unit than [`FORMAT`] is a miss, never an error.
fn read_tables(
    path: &Path,
) -> (
    FxHashMap<String, FxHashMap<String, Verdict>>,
    FxHashMap<String, (Vec<Option<String>>, u64)>,
) {
    let mut seen: FxHashMap<String, FxHashMap<String, Verdict>> = FxHashMap::default();
    let mut combos: FxHashMap<String, (Vec<Option<String>>, u64)> = FxHashMap::default();
    if let Some(disk) = crate::cache::load_versioned::<Disk>(path, |d| d.format == FORMAT) {
        for r in disk.records {
            let verdict = r.nanos.map_or(Verdict::Wrong, Verdict::Ran);
            seen.entry(r.launch).or_default().insert(r.variant, verdict);
        }
        for c in disk.combos {
            combos.insert(c.plan, (c.picks, c.score));
        }
    }
    (seen, combos)
}

/// What this device has learned. All mutation is behind one lock.
#[derive(Debug, Default)]
pub struct TuneCache {
    path: Option<PathBuf>,
    /// `launch signature -> variant signature -> verdict`.
    seen: Mutex<FxHashMap<String, FxHashMap<String, Verdict>>>,
    /// `plan signature -> (picks, score)`, the jointly-measured outcome.
    combos: Mutex<FxHashMap<String, (Vec<Option<String>>, u64)>>,
    /// Set when anything changed, so an unchanged process writes nothing.
    dirty: Mutex<bool>,
}

/// `$XDG_CACHE_HOME/fusor2/tune/<fingerprint>.json`, or `$HOME/.cache/...`.
/// `FUSOR2_TUNE_CACHE` overrides the whole path; `FUSOR2_NO_TUNE_CACHE`
/// disables persistence entirely.
pub fn cache_path(caps_fingerprint: u64) -> Option<PathBuf> {
    if std::env::var_os("FUSOR2_NO_TUNE_CACHE").is_some() {
        return None;
    }
    if let Some(p) = std::env::var_os("FUSOR2_TUNE_CACHE").filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(p));
    }
    Some(crate::cache::user_cache_dir("tune")?.join(format!("{caps_fingerprint:016x}.json")))
}

impl TuneCache {
    /// Read this device's file, or start empty. A malformed or unreadable file
    /// is an empty cache, never an error.
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
    /// measured here.
    pub fn known(&self, launch: &str, variant: &str) -> Option<Verdict> {
        self.seen.lock().get(launch)?.get(variant).copied()
    }

    /// The fastest variant recorded for this launch that reproduced the base's
    /// values, and its time. A wrong answer has no time, so it can never be the
    /// incumbent.
    pub fn best(&self, launch: &str) -> Option<(String, u64)> {
        let seen = self.seen.lock();
        seen.get(launch)?
            .iter()
            .filter_map(|(name, v)| match v {
                Verdict::Ran(ns) => Some((name.clone(), *ns)),
                Verdict::Wrong => None,
            })
            .min_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)))
    }

    pub fn record(&self, launch: &str, variant: &str, verdict: Verdict) {
        let mut seen = self.seen.lock();
        let slot = seen
            .entry(launch.to_string())
            .or_default()
            .entry(variant.to_string())
            .or_insert(verdict);
        *slot = match (*slot, verdict) {
            // Wrongness is a property of the kernel, not of the run, so
            // `Wrong` is absorbing.
            (Verdict::Wrong, _) | (_, Verdict::Wrong) => Verdict::Wrong,
            // Keep the fastest observation: timings are noisy upward, so a slow
            // sample is contention and a fast one is the kernel.
            (Verdict::Ran(a), Verdict::Ran(b)) => Verdict::Ran(a.min(b)),
        };
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
    /// While an entry is still exploring, the tuner races candidates and picks
    /// by this run's noisy clock. Once there is nothing new to try, the
    /// accumulated minimum over every past run is the better estimate, so the
    /// tuner applies it rather than re-deriving it.
    pub fn converged(&self, launch: &str, candidates: &[String]) -> bool {
        let seen = self.seen.lock();
        let Some(entry) = seen.get(launch) else {
            return false;
        };
        !candidates.is_empty() && candidates.iter().all(|c| entry.contains_key(c.as_str()))
    }

    /// Split candidates into what to run and what to skip, newest knowledge
    /// first.
    ///
    /// Returns `(to_measure, skipped)`. Ordering is: never-tried variants
    /// (capped at [`EXPLORE_PER_RESOLVE`]) after any known-good ones, so a warm
    /// entry re-confirms its incumbent cheaply and still makes progress on the
    /// unexplored tail. Ties break by name so a run is reproducible.
    pub fn plan_candidates<'a>(
        &self,
        launch: &str,
        candidates: &'a [String],
    ) -> (Vec<&'a String>, Vec<&'a String>) {
        let best = self.best(launch).map(|(_, ns)| ns);
        let seen = self.seen.lock();
        let entry = seen.get(launch);

        let mut known: Vec<(&'a String, u64)> = Vec::new();
        let mut fresh: Vec<&'a String> = Vec::new();
        let mut skipped: Vec<&'a String> = Vec::new();

        for c in candidates {
            match entry.and_then(|e| e.get(c.as_str())).copied() {
                // Proven on this device to compute a different function.
                // Re-racing it cannot change that and is not free.
                Some(Verdict::Wrong) => skipped.push(c),
                Some(Verdict::Ran(ns)) => {
                    let hopeless = best.is_some_and(|b| ns as f64 > b as f64 * SKIP_RATIO);
                    if hopeless {
                        skipped.push(c);
                    } else {
                        known.push((c, ns));
                    }
                }
                None => fresh.push(c),
            }
        }
        known.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        fresh.sort();

        let mut out: Vec<&'a String> = known
            .into_iter()
            .take(RERACE_PER_RESOLVE)
            .map(|(c, _)| c)
            .collect();
        for c in fresh.into_iter().take(EXPLORE_PER_RESOLVE) {
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
                    vs.iter().map(move |(v, verdict)| Record {
                        launch: l.clone(),
                        variant: v.clone(),
                        nanos: match verdict {
                            Verdict::Ran(ns) => Some(*ns),
                            Verdict::Wrong => None,
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
        // Write-then-rename: a crash mid-write leaves the old cache, not a
        // truncated one that would parse as empty and silently re-tune.
        if crate::cache::write_atomic(path, &body).is_ok() {
            *self.dirty.lock() = false;
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

    /// The fastest observation wins: timings are noisy upward, so a later slow
    /// sample must not overwrite a good one.
    #[test]
    fn keeps_the_fastest_observation() {
        let c = TuneCache::default();
        c.record("L", "v", Verdict::Ran(100));
        c.record("L", "v", Verdict::Ran(400));
        assert_eq!(c.known("L", "v"), Some(Verdict::Ran(100)));
        c.record("L", "v", Verdict::Ran(60));
        assert_eq!(c.known("L", "v"), Some(Verdict::Ran(60)));
    }

    /// A wrong answer is never the incumbent and is never re-raced.
    #[test]
    fn a_wrong_variant_is_excluded_not_ranked() {
        let c = TuneCache::default();
        c.record("L", "fast_but_wrong", Verdict::Wrong);
        c.record("L", "slow_and_right", Verdict::Ran(2_899_291));
        assert_eq!(c.best("L"), Some(("slow_and_right".to_string(), 2_899_291)));
        let cands: Vec<String> = ["fast_but_wrong", "slow_and_right"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (run, skip) = c.plan_candidates("L", &cands);
        assert_eq!(skip, vec![&"fast_but_wrong".to_string()]);
        assert_eq!(run, vec![&"slow_and_right".to_string()]);
        // And it stays wrong: a later fast timing does not resurrect it.
        c.record("L", "fast_but_wrong", Verdict::Ran(14_875));
        assert_eq!(c.known("L", "fast_but_wrong"), Some(Verdict::Wrong));
        // Nothing left to learn: a proven-wrong variant still counts as known.
        assert!(c.converged("L", &cands));
    }

    /// The verdict survives the file, so a later process does not re-race a
    /// kernel already caught computing a different function.
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

    /// A warm entry re-races its best-known first, explores a bounded number
    /// of new candidates, and excludes nothing.
    #[test]
    fn reraces_best_first_and_bounds_exploration() {
        let c = TuneCache::default();
        c.record("L", "good", Verdict::Ran(100));
        c.record("L", "awful", Verdict::Ran(1_000)); // > 100 * SKIP_RATIO
        let cands: Vec<String> = ["good", "awful", "n1", "n2", "n3", "n4", "n5", "n6"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (run, skip) = c.plan_candidates("L", &cands);
        assert!(skip.is_empty(), "score-based exclusion is unsound: {skip:?}");
        assert_eq!(run[0], &"good".to_string(), "incumbent first: {run:?}");
        assert_eq!(
            run.len(),
            2 + EXPLORE_PER_RESOLVE,
            "every known candidate stays eligible; only exploration is bounded: {run:?}"
        );
    }

    /// An unknown launch explores, so a cold cache still makes progress.
    #[test]
    fn a_cold_entry_explores() {
        let c = TuneCache::default();
        let cands: Vec<String> = (0..10).map(|i| format!("v{i}")).collect();
        let (run, skip) = c.plan_candidates("unseen", &cands);
        assert!(skip.is_empty());
        assert_eq!(run.len(), EXPLORE_PER_RESOLVE);
    }

    /// Convergence is nothing left to try, so the tuner applies the
    /// accumulated winner instead of re-racing noisy samples.
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

    /// The whole-plan combination round-trips and keeps the better score.
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

    /// A file written in an older unit is dropped: ppm-of-base and absolute
    /// kernel nanoseconds are the same order of magnitude, so mixing them
    /// would silently reorder every candidate.
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
