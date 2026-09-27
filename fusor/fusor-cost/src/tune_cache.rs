//! The per-machine tuning cache: timing observations only, never correctness
//! verdicts, keyed by `Caps::fingerprint()`.

use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Never-measured variants one race times: the model's top K.
pub const RACE_TOP_K: usize = 3;

/// Observations one `(launch, variant)` window holds; decisions read its min.
pub const WINDOW: usize = 8;

/// A candidate's last up-to-[`WINDOW`] timing observations, oldest first.
/// GPU samples are launch nanoseconds; CPU samples are ppm of the base plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Record {
    launch: String,
    variant: String,
    window: Vec<u64>,
}

/// The per-launch variants fastest when measured together; per-launch
/// minima do not compose.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct Combo {
    plan: String,
    /// One per launch; `None` is the base plan's own choice.
    picks: Vec<Option<String>>,
    score: u64,
}

/// The on-disk format version; any other reads as empty.
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
        && disk.format == FORMAT
    {
        for r in disk.records {
            if r.window.is_empty() {
                continue;
            }
            let start = r.window.len().saturating_sub(WINDOW);
            seen.entry(r.launch)
                .or_default()
                .insert(r.variant, r.window[start..].to_vec());
        }
        for c in disk.combos {
            combos.insert(c.plan, (c.picks, c.score));
        }
    }
    (seen, combos)
}

/// What this device has learned.
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
    /// Read this device's file; anything unreadable is an empty cache.
    pub fn load(caps_fingerprint: u64) -> Self {
        Self::at(cache_path(caps_fingerprint))
    }

    /// A cache persisted at `path`, or in memory only.
    fn at(path: Option<PathBuf>) -> Self {
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

    /// Whether a family of timing fields contains enough samples to compare.
    pub fn has_observations_with_prefix(&self, prefix: &str, minimum: usize) -> bool {
        self.seen.lock().iter().any(|(field, variants)| {
            field.starts_with(prefix) && variants.values().any(|window| window.len() >= minimum)
        })
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

    /// Split `(candidate, prior)`s into `(to_measure, skipped)`: measured
    /// ones by window min, then the top [`RACE_TOP_K`] fresh ones by prior.
    pub fn plan_candidates<'a>(
        &self,
        launch: &str,
        candidates: &'a [(String, u64)],
    ) -> (Vec<&'a String>, Vec<&'a String>) {
        let seen = self.seen.lock();
        let entry = seen.get(launch);
        let mut known: Vec<(&'a String, u64)> = Vec::new();
        let mut fresh: Vec<(&'a String, u64)> = Vec::new();
        for (c, prior) in candidates {
            match entry.and_then(|e| e.get(c.as_str())) {
                Some(w) => known.push((c, w.iter().copied().min().unwrap_or(u64::MAX))),
                None => fresh.push((c, *prior)),
            }
        }
        known.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(b.0)));
        // Stable: a tie keeps the enumerator's believed-best first.
        fresh.sort_by_key(|a| a.1);

        let out = known.into_iter().chain(fresh.into_iter().take(RACE_TOP_K));
        // Every known variant re-races: the plan optimum is not the
        // per-launch argmin, so nothing is skipped.
        (out.map(|(c, _)| c).collect(), Vec::new())
    }

    /// Persist atomically, best-effort, if anything changed.
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
        // Write-then-rename, so a crash leaves the old cache.
        let tmp = path.with_extension("json.tmp");
        if std::fs::write(&tmp, body).is_ok() && std::fs::rename(&tmp, path).is_ok() {
            *self.dirty.lock() = false;
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
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
        cache.observe("diff:plan", "candidate", 30);
        assert!(!cache.has_observations_with_prefix("diff:", 2));
        cache.observe("diff:plan", "candidate", 40);
        cache.save();
        let restored = TuneCache::at(Some(path.clone()));
        assert_eq!(restored.window_min("launch", "a"), Some(40));
        assert_eq!(restored.observations("launch", "a"), 2);
        assert_eq!(restored.combo("plan"), Some(vec![Some("a".into())]));
        assert!(restored.has_observations_with_prefix("diff:", 2));
        assert!(!restored.has_observations_with_prefix("other:", 2));
        let mut disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        disk["format"] = (FORMAT + 1).into();
        std::fs::write(&path, serde_json::to_string(&disk).unwrap()).unwrap();
        let obsolete = TuneCache::at(Some(path.clone()));
        assert!(obsolete.is_empty());
        assert!(!obsolete.has_observations_with_prefix("diff:", 2));
        assert_eq!(obsolete.combo("plan"), None);
        std::fs::remove_file(path).unwrap();
    }
}
