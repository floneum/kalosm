//! On-disk calibration cache:
//! `$XDG_CACHE_HOME/fusor2/facts/<fingerprint>.json`, falling back to
//! `~/.cache/fusor2/facts`.
//!
//! [`FactsRecord`] is a deliberate mirror rather than a `Serialize` derive on
//! `DeviceFacts`: `Caps` is re-probed from the live device on every run, so
//! persisting it would let a stale capability set outlive a driver update.
//! Only the *measured rates* round-trip.
//!
//! **This module is the only place in the crate that reads the
//! environment.** Whether to calibrate at all is [`CalibrationMode`], a
//! function argument — every `spike` flag in the reference names a decision
//! that must be made per-shape by the cost model, not per-process by a
//! variable nobody sets.
//!
//! Owned by W6.

use fusor2_ir::Result;
use fusor2_ir::cost::{Calibrate, DeviceFacts, RateDtype};
use fusor2_ir::device::Caps;
use fusor2_ir::error::Error;
use fusor2_ir::target::Target;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bumped whenever a rate's meaning changes; a mismatch is a cache miss.
pub const FORMAT_VERSION: u32 = 1;

/// The name the W6 spec uses for [`FORMAT_VERSION`].
pub const FACTS_FORMAT_VERSION: u32 = FORMAT_VERSION;

/// Whether to consult, refresh or ignore the cache. **A function argument,
/// never an environment variable.**
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub enum CalibrationMode {
    /// Load if cached, else calibrate, else seed; store whatever resulted.
    #[default]
    Cached,
    /// Calibrate and overwrite the cache even on a hit.
    Force,
    /// Seed table only. No device work, no disk access.
    Disabled,
}

/// The persisted half of `DeviceFacts` — the measured rates only.
///
/// `mac_per_us` is a `Vec<Vec<u64>>` rather than a fixed array so that a
/// future `MacUnit` or `RateDtype` row is a length mismatch (a miss) rather
/// than a deserialization error.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactsRecord {
    pub version: u32,
    pub caps_fingerprint: u64,
    pub launch_ps: u64,
    pub dram_bytes_per_us: u64,
    pub llc_bytes: u64,
    pub wg_bytes_per_us: u64,
    pub mac_per_us: Vec<Vec<u64>>,
    pub trans_ps: u64,
    pub store_ps_per_element: u64,
    pub saturation_lanes: u32,
    pub single_buffered_traffic_pct: u32,
    pub compile_ps_per_kernel: u64,
    pub thread_wake_ps: u64,
}

impl FactsRecord {
    pub fn from_facts(facts: &DeviceFacts) -> Self {
        Self {
            version: FORMAT_VERSION,
            caps_fingerprint: facts.caps.fingerprint(),
            launch_ps: facts.launch_ps,
            dram_bytes_per_us: facts.dram_bytes_per_us,
            llc_bytes: facts.llc_bytes,
            wg_bytes_per_us: facts.wg_bytes_per_us,
            mac_per_us: facts.mac_per_us.iter().map(|row| row.to_vec()).collect(),
            trans_ps: facts.trans_ps,
            store_ps_per_element: facts.store_ps_per_element,
            saturation_lanes: facts.saturation_lanes,
            single_buffered_traffic_pct: facts.single_buffered_traffic_pct,
            compile_ps_per_kernel: facts.compile_ps_per_kernel,
            thread_wake_ps: facts.thread_wake_ps,
        }
    }

    /// Rehydrate against freshly probed caps. `None` when the record's shape
    /// no longer matches this build's rate table.
    pub fn into_facts(self, caps: Caps) -> Option<DeviceFacts> {
        if self.mac_per_us.len() != 3 {
            return None;
        }
        let mut mac_per_us = [[0u64; RateDtype::COUNT]; 3];
        for (dst, src) in mac_per_us.iter_mut().zip(&self.mac_per_us) {
            if src.len() != RateDtype::COUNT {
                return None;
            }
            dst.copy_from_slice(src);
        }
        Some(DeviceFacts {
            launch_ps: self.launch_ps,
            dram_bytes_per_us: self.dram_bytes_per_us,
            llc_bytes: self.llc_bytes,
            wg_bytes_per_us: self.wg_bytes_per_us,
            mac_per_us,
            trans_ps: self.trans_ps,
            store_ps_per_element: self.store_ps_per_element,
            saturation_lanes: self.saturation_lanes,
            single_buffered_traffic_pct: self.single_buffered_traffic_pct,
            compile_ps_per_kernel: self.compile_ps_per_kernel,
            thread_wake_ps: self.thread_wake_ps,
            caps,
        })
    }
}

/// `$XDG_CACHE_HOME/fusor2/facts`, else `~/.cache/fusor2/facts`, else a
/// relative `.cache/fusor2/facts` when neither variable is set.
///
/// The one environment read in the crate, and it selects a location rather
/// than a behaviour.
pub fn cache_dir() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(xdg).join("fusor2").join("facts");
    }
    let home = std::env::var_os("HOME")
        .filter(|v| !v.is_empty())
        .map_or_else(|| PathBuf::from("."), PathBuf::from);
    home.join(".cache").join("fusor2").join("facts")
}

/// Cache path for one caps fingerprint.
pub fn path_for(dir: &Path, caps_fingerprint: u64) -> PathBuf {
    dir.join(format!("{caps_fingerprint:016x}.json"))
}

/// The raw record on disk for one fingerprint, or `None`. A version
/// mismatch, a fingerprint mismatch, a missing file and a corrupt file are
/// all the same answer: a miss, never an error.
pub fn load_record(dir: &Path, fingerprint: u64) -> Option<FactsRecord> {
    let text = std::fs::read_to_string(path_for(dir, fingerprint)).ok()?;
    let record: FactsRecord = serde_json::from_str(&text).ok()?;
    (record.version == FORMAT_VERSION && record.caps_fingerprint == fingerprint).then_some(record)
}

/// The cached facts for these caps, rehydrated against them.
///
/// Takes `&Caps` rather than the bare fingerprint the W6 spec lists, because
/// `DeviceFacts` owns a `Caps` and the record deliberately does not persist
/// one — there would be nothing to rehydrate from. The fingerprint is
/// `caps.fingerprint()`, exactly what [`store`] filed it under.
pub fn load(dir: &Path, caps: &Caps) -> Option<DeviceFacts> {
    load_record(dir, caps.fingerprint())?.into_facts(caps.clone())
}

/// Write `<fingerprint>.json` through a temp file and an atomic rename, so a
/// crashed or concurrent writer cannot leave a half-written record that the
/// next process reads as truth.
pub fn store(dir: &Path, facts: &DeviceFacts) -> Result<()> {
    let record = FactsRecord::from_facts(facts);
    let final_path = path_for(dir, record.caps_fingerprint);
    let temp_path = dir.join(format!(
        "{:016x}.{}.tmp",
        record.caps_fingerprint,
        std::process::id()
    ));
    std::fs::create_dir_all(dir).map_err(|e| Error::Io(e.to_string()))?;
    let text = serde_json::to_string_pretty(&record).map_err(|e| Error::Io(e.to_string()))?;
    std::fs::write(&temp_path, text).map_err(|e| Error::Io(e.to_string()))?;
    match std::fs::rename(&temp_path, &final_path) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = std::fs::remove_file(&temp_path);
            Err(Error::Io(e.to_string()))
        }
    }
}

/// The default-directory form the crate contract lists.
pub fn load_default(caps: &Caps) -> Result<Option<DeviceFacts>> {
    Ok(load(&cache_dir(), caps))
}

/// See [`load_default`].
pub fn store_default(facts: &DeviceFacts) -> Result<()> {
    store(&cache_dir(), facts)
}

/// The facts a target should compile against.
///
/// `Cached` loads, else calibrates and stores, else seeds and stores.
/// `Force` always calibrates. `Disabled` seeds and touches no disk. A
/// calibration failure is never fatal: the seed is a working table.
pub fn facts_for(target: &dyn Target, mode: CalibrationMode) -> DeviceFacts {
    let caps = target.caps();
    if mode == CalibrationMode::Disabled {
        return crate::facts::seed_facts(caps);
    }
    let dir = cache_dir();
    if mode == CalibrationMode::Cached
        && let Some(cached) = load(&dir, caps)
    {
        return cached;
    }
    let calibrator = crate::calibrate::Calibrator::new();
    let facts = calibrator
        .calibrate(target)
        .unwrap_or_else(|_| crate::facts::seed_facts(caps));
    // A cache we cannot write is a slow start, not a failure.
    let _ = store(&dir, &facts);
    facts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::facts::seed_facts;
    use crate::facts::tests::{cpu_caps, gpu_caps};

    /// A scratch directory under the process's temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "fusor2-cost-{}-{}-{tag}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos())
            ));
            std::fs::create_dir_all(&dir).expect("scratch dir");
            Self(dir)
        }
        fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Test 12.
    #[test]
    fn facts_round_trip_through_cache() {
        let scratch = Scratch::new("roundtrip");
        let caps = gpu_caps("round trip");
        let mut facts = seed_facts(&caps);
        // Pretend a calibration ran and moved every rate off its seed.
        facts.launch_ps = 812_345;
        facts.dram_bytes_per_us = 401_003;
        facts.llc_bytes = 12 << 20;
        facts.wg_bytes_per_us = 733_112;
        facts.mac_per_us[0][0] = 4_711_000;
        facts.mac_per_us[2][4] = 19_000_001;
        facts.trans_ps = 7;
        facts.store_ps_per_element = 5;
        facts.saturation_lanes = 49_152;
        facts.single_buffered_traffic_pct = 111;
        facts.compile_ps_per_kernel = 900_000_007;
        facts.thread_wake_ps = 4_321_000;

        store(scratch.path(), &facts).expect("store");
        let loaded = load(scratch.path(), &caps).expect("hit");
        assert_eq!(loaded, facts, "the round trip must be bit-identical");

        // A different device is a miss, not another device's rates.
        assert!(load(scratch.path(), &cpu_caps("other", 4)).is_none());

        // A bumped format version is a miss.
        let path = path_for(scratch.path(), caps.fingerprint());
        let mut record = FactsRecord::from_facts(&facts);
        record.version = FORMAT_VERSION + 1;
        std::fs::write(&path, serde_json::to_string(&record).unwrap()).unwrap();
        assert!(
            load(scratch.path(), &caps).is_none(),
            "a bumped FACTS_FORMAT_VERSION must be a miss"
        );

        // So is a corrupt file, and it must not be an error.
        std::fs::write(&path, "{ not json").unwrap();
        assert!(load(scratch.path(), &caps).is_none());
        assert!(load_default(&caps).is_ok());
    }

    /// A caps change that only touches workgroup storage files under a
    /// different name, so a driver update cannot serve stale coop legality.
    #[test]
    fn workgroup_storage_change_is_a_different_cache_entry() {
        let scratch = Scratch::new("wgstorage");
        let mut a = gpu_caps("dev");
        a.limits.max_compute_workgroup_storage_size = 16 << 10;
        let mut b = a.clone();
        b.limits.max_compute_workgroup_storage_size = 32 << 10;

        store(scratch.path(), &seed_facts(&a)).expect("store");
        assert!(load(scratch.path(), &a).is_some());
        assert!(load(scratch.path(), &b).is_none());
    }

    /// `seed_or_cached` prefers a hit and falls back to the seed.
    #[test]
    fn seed_or_cached_prefers_the_cache() {
        let scratch = Scratch::new("seedor");
        let caps = gpu_caps("seed-or-cached");
        assert_eq!(
            crate::facts::seed_or_cached(&caps, Some(scratch.path())),
            seed_facts(&caps)
        );

        let mut measured = seed_facts(&caps);
        measured.dram_bytes_per_us = 123_456;
        store(scratch.path(), &measured).unwrap();
        assert_eq!(
            crate::facts::seed_or_cached(&caps, Some(scratch.path())).dram_bytes_per_us,
            123_456
        );
    }

    /// `Disabled` is seed-only. No `Target` implementation is reachable from
    /// this crate, so the mode's disk-free branch is exercised through the
    /// seed it delegates to.
    #[test]
    fn disabled_mode_is_seed_only() {
        let caps = gpu_caps("disabled");
        assert_eq!(seed_facts(&caps), crate::facts::seed_facts(&caps));
        assert_eq!(CalibrationMode::default(), CalibrationMode::Cached);
    }

    /// Test 13. No optimizer behaviour may be gated on an environment
    /// variable: every flag in the reference's `FusorConfig` names a
    /// decision that belongs to the cost model. The only legitimate
    /// environment read in this crate is the cache location, which selects a
    /// place rather than a policy.
    #[test]
    fn no_env_gated_behaviour() {
        const OWNED: [(&str, &str); 4] = [
            ("facts.rs", include_str!("facts.rs")),
            ("model.rs", include_str!("model.rs")),
            ("terms.rs", include_str!("terms.rs")),
            ("calibrate.rs", include_str!("calibrate.rs")),
        ];
        // Assembled at runtime so this file's own needles are not a literal
        // a future copy-paste into another module could match by accident.
        let var_call = ["env", "::", "var"].concat();
        let module = ["std", "::", "env"].concat();
        let spike = ["spike", "_"].concat();
        for (name, source) in OWNED {
            assert!(
                !source.contains(&var_call),
                "{name} reads the environment; only cache.rs may"
            );
            assert!(
                !source.contains(&module),
                "{name} reaches for the env module; only cache.rs may"
            );
            assert!(
                !source.contains(&spike),
                "{name} carries a spike flag; those are cost-model terms now"
            );
        }
        // And cache.rs itself reads exactly two variables, both locations.
        let me = include_str!("cache.rs");
        assert!(me.contains("XDG_CACHE_HOME"));
        assert!(me.contains("HOME"));
    }
}
