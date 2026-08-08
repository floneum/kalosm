//! The plan cache: memory LRU plus a disk tier, salted by exe identity and the
//! `DeviceFacts` fingerprint — which **includes**
//! `max_compute_workgroup_storage_size`, a field the reference omits while its
//! coop legality filter reads it, a live staleness hazard.
//!
//! A [`CachedKernelPlan`] is bufferless: the compiled template plus, per
//! binding slot, the caller-buffer index it wants, plus, per caller position,
//! the first position holding the same buffer. `record_plan` refuses (silently,
//! returning `None`) to record a kernel that binds a buffer the caller did not
//! present, and [`binding_shape_matches`] requires the caller's aliasing
//! pattern to reproduce the recorded one **exactly** — a kernel body is only
//! correct for callers with the identical aliasing pattern.
//!
//! Owned by W9.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use fusor2_ir::Result;
use fusor2_ir::cost::DeviceFacts;
use fusor2_ir::error::Error;
use fusor2_ir::extract::PlanHash;
use fusor2_ir::target::Buf;
use parking_lot::Mutex;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

/// Bumped whenever the on-disk record layout changes. A mismatch is a miss.
pub const DISK_PLAN_FORMAT_VERSION: u32 = 1;
/// Memory LRU capacity.
pub const MEMORY_CAPACITY: usize = if cfg!(target_arch = "wasm32") { 512 } else { 4096 };
/// Salt directories untouched this long are removed on open.
pub const SALT_TTL: Duration = Duration::from_secs(14 * 24 * 60 * 60);

/// One binding of a bufferless kernel template.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TemplateBinding {
    pub binding: u32,
    pub read_only: bool,
}

/// A compiled kernel with its buffers stripped out, so a hit rebinds
/// positionally instead of recompiling.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelTemplate {
    pub name: String,
    pub grid: [u32; 3],
    pub block: u32,
    pub bindings: Vec<TemplateBinding>,
    /// Present only when the emitter could serialize the module. Absent means
    /// the disk tier stores the *plan* and the driver's own pipeline cache
    /// stores the binary; both are misses, never corruption.
    pub source: Option<String>,
}

/// A recorded plan: the template plus the caller-buffer index per binding slot
/// and, per caller position, the first position holding the same buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CachedKernelPlan {
    pub template: KernelTemplate,
    /// `permutation[i]` is the caller-buffer index bound at binding slot `i`.
    pub permutation: Vec<usize>,
    /// `alias_class[i]` is the first caller position holding the same buffer as
    /// position `i`. Equal entries mean the two positions were aliased.
    pub alias_class: Vec<usize>,
}

/// Compiled artifacts keyed by [`PlanHash`].
pub struct PlanCache {
    memory: Mutex<lru::LruCache<PlanHash, Arc<[CachedKernelPlan]>>>,
    disk: Option<DiskPlanCache>,
    salt: u64,
    hits: Mutex<CacheCounters>,
}

/// What the cache has served.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CacheCounters {
    pub memory_hits: u64,
    pub disk_hits: u64,
}

impl PlanCache {
    /// Build a cache. `dir` is the root under which `fusor2/plans/<salt>/`
    /// lives; `None` disables the disk tier entirely.
    pub fn new(capacity: usize, salt: u64, dir: Option<PathBuf>) -> Self {
        let disk = dir.map(|d| DiskPlanCache::open(d, salt));
        Self {
            memory: Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(capacity.max(1)).expect("capacity clamped to >= 1"),
            )),
            disk,
            salt,
            hits: Mutex::new(CacheCounters::default()),
        }
    }

    /// The default: `MEMORY_CAPACITY` entries plus a disk tier under the
    /// platform cache directory.
    pub fn with_facts(facts: &DeviceFacts, dir: Option<PathBuf>) -> Self {
        Self::new(MEMORY_CAPACITY, disk_salt(facts), dir)
    }

    /// Look a plan up. Memory first, then disk; a disk hit is promoted.
    pub fn get(&self, hash: PlanHash) -> Option<Arc<[CachedKernelPlan]>> {
        if let Some(hit) = self.memory.lock().get(&hash).cloned() {
            self.hits.lock().memory_hits += 1;
            return Some(hit);
        }
        let disk = self.disk.as_ref()?;
        // Every failure path below — missing file, decode error, version
        // mismatch — is a miss, never a corruption.
        let plans = disk.load(hash)?;
        let plans: Arc<[CachedKernelPlan]> = plans.into();
        self.memory.lock().put(hash, plans.clone());
        self.hits.lock().disk_hits += 1;
        Some(plans)
    }

    /// Record a plan in both tiers.
    pub fn insert(&self, hash: PlanHash, plans: Arc<[CachedKernelPlan]>) {
        if let Some(disk) = &self.disk {
            disk.store(hash, &plans);
        }
        self.memory.lock().put(hash, plans);
    }

    /// Build a [`CachedKernelPlan`] for one kernel.
    ///
    /// Returns `None` — and *skips caching silently* — when the kernel binds a
    /// buffer the caller did not present. That is the internal-scratch case;
    /// recording it would produce a template no caller can rebind.
    pub fn record_plan(
        template: KernelTemplate,
        kernel_buffers: &[Buf],
        caller_buffers: &[Buf],
    ) -> Option<CachedKernelPlan> {
        let mut permutation = Vec::with_capacity(kernel_buffers.len());
        for kb in kernel_buffers {
            let pos = caller_buffers.iter().position(|cb| cb.addr() == kb.addr())?;
            permutation.push(pos);
        }
        Some(CachedKernelPlan {
            template,
            permutation,
            alias_class: alias_classes(caller_buffers),
        })
    }

    pub fn counters(&self) -> CacheCounters {
        *self.hits.lock()
    }

    /// Exe identity + device fingerprint. A rebuild or a driver update must
    /// invalidate the disk tier.
    pub fn disk_salt(&self) -> u64 {
        self.salt
    }
}

impl CachedKernelPlan {
    /// Rebind this template's buffers from a caller's list.
    pub fn rebind(&self, caller_buffers: &[Buf]) -> Option<Vec<Buf>> {
        if !self.binding_shape_matches(caller_buffers) {
            return None;
        }
        self.permutation
            .iter()
            .map(|i| caller_buffers.get(*i).cloned())
            .collect()
    }

    /// The caller's aliasing pattern must reproduce the recorded one
    /// **exactly**: same positions aliased, same positions distinct. A kernel
    /// body compiled for distinct buffers is not correct for an aliased pair,
    /// and vice versa.
    pub fn binding_shape_matches(&self, caller_buffers: &[Buf]) -> bool {
        if caller_buffers.len() != self.alias_class.len() {
            return false;
        }
        if self
            .permutation
            .iter()
            .any(|i| *i >= caller_buffers.len())
        {
            return false;
        }
        alias_classes(caller_buffers) == self.alias_class
    }
}

/// Per position, the first position holding the same buffer.
pub fn alias_classes(buffers: &[Buf]) -> Vec<usize> {
    let mut out = Vec::with_capacity(buffers.len());
    for (i, b) in buffers.iter().enumerate() {
        let first = buffers[..i]
            .iter()
            .position(|other| other.addr() == b.addr())
            .unwrap_or(i);
        out.push(first);
    }
    out
}

/// The disk salt: executable identity plus [`DeviceFacts::fingerprint`], which
/// includes `Caps::limits.max_compute_workgroup_storage_size`.
pub fn disk_salt(facts: &DeviceFacts) -> u64 {
    let mut h = FxHasher::default();
    if let Ok(exe) = std::env::current_exe() {
        exe.hash(&mut h);
        if let Ok(meta) = std::fs::metadata(&exe) {
            meta.len().hash(&mut h);
            if let Ok(mtime) = meta.modified()
                && let Ok(since) = mtime.duration_since(SystemTime::UNIX_EPOCH)
            {
                since.as_secs().hash(&mut h);
            }
        }
    }
    facts.fingerprint().hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// Disk tier
// ---------------------------------------------------------------------------

/// One file per [`PlanHash`] under `<dir>/fusor2/plans/<salt>/<hi><lo>.plan`.
pub struct DiskPlanCache {
    root: PathBuf,
}

impl DiskPlanCache {
    /// Open (and lazily create) the salt directory, removing salt directories
    /// untouched for [`SALT_TTL`].
    pub fn open(dir: PathBuf, salt: u64) -> Self {
        let plans = dir.join("fusor2").join("plans");
        let root = plans.join(format!("{salt:016x}"));
        let _ = std::fs::create_dir_all(&root);
        gc_stale_salts(&plans, &root);
        Self { root }
    }

    pub fn path_for(&self, hash: PlanHash) -> PathBuf {
        let hi = (hash.0 >> 64) as u64;
        let lo = hash.0 as u64;
        self.root.join(format!("{hi:016x}{lo:016x}.plan"))
    }

    /// Every failure — missing, truncated, version mismatch, trailing bytes —
    /// is a miss.
    pub fn load(&self, hash: PlanHash) -> Option<Vec<CachedKernelPlan>> {
        let bytes = std::fs::read(self.path_for(hash)).ok()?;
        decode(&bytes).ok()
    }

    /// Atomic temp-file plus rename, so a crashed write is never observed as a
    /// truncated record.
    pub fn store(&self, hash: PlanHash, plans: &[CachedKernelPlan]) {
        let final_path = self.path_for(hash);
        let tmp = final_path.with_extension("plan.tmp");
        if std::fs::write(&tmp, encode(plans)).is_ok() {
            let _ = std::fs::rename(&tmp, &final_path);
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

fn gc_stale_salts(plans: &Path, keep: &Path) {
    let Ok(entries) = std::fs::read_dir(plans) else {
        return;
    };
    let now = SystemTime::now();
    for entry in entries.flatten() {
        let path = entry.path();
        if path == keep || !path.is_dir() {
            continue;
        }
        let stale = entry
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| now.duration_since(t).ok())
            .is_some_and(|age| age > SALT_TTL);
        if stale {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

// ---------------------------------------------------------------------------
// Codec
// ---------------------------------------------------------------------------
//
// A hand-rolled little-endian codec rather than a serde dependency: the record
// is six field kinds deep and the format version is the only compatibility
// contract, so a derive would buy nothing and would pull a wire format into a
// crate that has no other need for one.

fn put_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}
fn put_str(out: &mut Vec<u8>, s: &str) {
    put_u32(out, s.len() as u32);
    out.extend_from_slice(s.as_bytes());
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn u32(&mut self) -> Result<u32> {
        let end = self.at + 4;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| Error::Io("plan record truncated".into()))?;
        self.at = end;
        Ok(u32::from_le_bytes(slice.try_into().expect("4 bytes")))
    }
    fn u64(&mut self) -> Result<u64> {
        let end = self.at + 8;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| Error::Io("plan record truncated".into()))?;
        self.at = end;
        Ok(u64::from_le_bytes(slice.try_into().expect("8 bytes")))
    }
    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        let end = self.at + len;
        let slice = self
            .bytes
            .get(self.at..end)
            .ok_or_else(|| Error::Io("plan record truncated".into()))?;
        self.at = end;
        String::from_utf8(slice.to_vec()).map_err(|_| Error::Io("plan record is not utf-8".into()))
    }
}

/// Serialize a plan list.
pub fn encode(plans: &[CachedKernelPlan]) -> Vec<u8> {
    let mut out = Vec::new();
    put_u32(&mut out, DISK_PLAN_FORMAT_VERSION);
    put_u32(&mut out, plans.len() as u32);
    for plan in plans {
        put_str(&mut out, &plan.template.name);
        for d in plan.template.grid {
            put_u32(&mut out, d);
        }
        put_u32(&mut out, plan.template.block);
        put_u32(&mut out, plan.template.bindings.len() as u32);
        for b in &plan.template.bindings {
            put_u32(&mut out, b.binding);
            put_u32(&mut out, u32::from(b.read_only));
        }
        match &plan.template.source {
            Some(s) => {
                put_u32(&mut out, 1);
                put_str(&mut out, s);
            }
            None => put_u32(&mut out, 0),
        }
        put_u32(&mut out, plan.permutation.len() as u32);
        for p in &plan.permutation {
            put_u64(&mut out, *p as u64);
        }
        put_u32(&mut out, plan.alias_class.len() as u32);
        for a in &plan.alias_class {
            put_u64(&mut out, *a as u64);
        }
    }
    out
}

/// Deserialize a plan list. Trailing bytes are an error, so a partially
/// overwritten file is a miss rather than a half-read record.
pub fn decode(bytes: &[u8]) -> Result<Vec<CachedKernelPlan>> {
    let mut r = Reader { bytes, at: 0 };
    let version = r.u32()?;
    if version != DISK_PLAN_FORMAT_VERSION {
        return Err(Error::Io(format!(
            "plan record version {version} != {DISK_PLAN_FORMAT_VERSION}"
        )));
    }
    let count = r.u32()? as usize;
    let mut plans = Vec::with_capacity(count.min(1024));
    for _ in 0..count {
        let name = r.string()?;
        let grid = [r.u32()?, r.u32()?, r.u32()?];
        let block = r.u32()?;
        let n_bindings = r.u32()? as usize;
        let mut bindings = Vec::with_capacity(n_bindings.min(64));
        for _ in 0..n_bindings {
            bindings.push(TemplateBinding {
                binding: r.u32()?,
                read_only: r.u32()? != 0,
            });
        }
        let source = if r.u32()? != 0 {
            Some(r.string()?)
        } else {
            None
        };
        let n_perm = r.u32()? as usize;
        let mut permutation = Vec::with_capacity(n_perm.min(64));
        for _ in 0..n_perm {
            permutation.push(r.u64()? as usize);
        }
        let n_alias = r.u32()? as usize;
        let mut alias_class = Vec::with_capacity(n_alias.min(64));
        for _ in 0..n_alias {
            alias_class.push(r.u64()? as usize);
        }
        plans.push(CachedKernelPlan {
            template: KernelTemplate {
                name,
                grid,
                block,
                bindings,
                source,
            },
            permutation,
            alias_class,
        });
    }
    if r.at != bytes.len() {
        return Err(Error::Io("plan record has trailing bytes".into()));
    }
    Ok(plans)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::cost::{DeviceFacts, RateDtype};
    use fusor2_ir::device::{Caps, DeviceKind, Limits};

    fn facts(wg_storage: u32) -> DeviceFacts {
        DeviceFacts {
            launch_ps: 1,
            dram_bytes_per_us: 1,
            llc_bytes: 1,
            wg_bytes_per_us: 1,
            mac_per_us: [[1; RateDtype::COUNT]; 3],
            trans_ps: 1,
            store_ps_per_element: 1,
            saturation_lanes: 1,
            single_buffered_traffic_pct: 1,
            compile_ps_per_kernel: 1,
            thread_wake_ps: 1,
            caps: Caps {
                kind: DeviceKind::Gpu,
                name: "test".into(),
                limits: Limits {
                    max_compute_workgroup_storage_size: wg_storage,
                    ..Limits::default()
                },
                subgroups: None,
                f16: false,
                bf16: false,
                coop: Default::default(),
                atomic_f32: false,
                workgroup_alias: false,
                mixed_precision_coop_store: false,
                pipeline_cache: false,
                timestamp_query: false,
                simd_widths: Default::default(),
                threads: 1,
            },
        }
    }

    fn plan(permutation: Vec<usize>, alias_class: Vec<usize>) -> CachedKernelPlan {
        CachedKernelPlan {
            template: KernelTemplate {
                name: "k".into(),
                grid: [1, 2, 3],
                block: 256,
                bindings: vec![
                    TemplateBinding {
                        binding: 0,
                        read_only: true,
                    },
                    TemplateBinding {
                        binding: 1,
                        read_only: false,
                    },
                ],
                source: Some("@compute fn main() {}".into()),
            },
            permutation,
            alias_class,
        }
    }

    /// Test 9: two `DeviceFacts` differing **only** in
    /// `max_compute_workgroup_storage_size` must land in different salt
    /// directories. The reference omits this field while its coop legality
    /// filter reads it.
    #[test]
    fn disk_salt_includes_workgroup_storage() {
        let a = disk_salt(&facts(16384));
        let b = disk_salt(&facts(32768));
        assert_ne!(a, b, "the salt must separate two workgroup-storage classes");
        assert_eq!(a, disk_salt(&facts(16384)), "the salt must be stable");
    }

    #[test]
    fn the_fingerprint_itself_carries_the_field() {
        assert_ne!(facts(16384).fingerprint(), facts(32768).fingerprint());
    }

    /// Test 8: a plan recorded with distinct buffers is not reused when the
    /// caller passes an aliased pair, and vice versa.
    #[test]
    fn plan_cache_rejects_alias_mismatch() {
        let x = Buf::new(1u32);
        let y = Buf::new(2u32);
        let distinct = vec![x.clone(), y.clone()];
        let aliased = vec![x.clone(), x.clone()];

        let recorded_distinct = plan(vec![0, 1], alias_classes(&distinct));
        let recorded_aliased = plan(vec![0, 1], alias_classes(&aliased));
        assert_eq!(recorded_distinct.alias_class, vec![0, 1]);
        assert_eq!(recorded_aliased.alias_class, vec![0, 0]);

        assert!(recorded_distinct.binding_shape_matches(&distinct));
        assert!(!recorded_distinct.binding_shape_matches(&aliased));
        assert!(recorded_aliased.binding_shape_matches(&aliased));
        assert!(!recorded_aliased.binding_shape_matches(&distinct));

        assert!(recorded_distinct.rebind(&aliased).is_none());
        assert_eq!(recorded_distinct.rebind(&distinct).unwrap().len(), 2);
    }

    /// A kernel binding a buffer the caller did not present is skipped
    /// silently, because a recorded template no caller can rebind is worse
    /// than a miss.
    #[test]
    fn record_plan_skips_internal_scratch() {
        let caller = vec![Buf::new(1u32), Buf::new(2u32)];
        let scratch = Buf::new(3u32);
        let template = plan(vec![], vec![]).template;
        let with_scratch = vec![caller[0].clone(), scratch];
        assert!(PlanCache::record_plan(template.clone(), &with_scratch, &caller).is_none());
        assert!(PlanCache::record_plan(template, &caller, &caller).is_some());
    }

    #[test]
    fn codec_round_trips() {
        let plans = vec![plan(vec![0, 1], vec![0, 1]), plan(vec![1, 0], vec![0, 0])];
        let bytes = encode(&plans);
        assert_eq!(decode(&bytes).unwrap(), plans);
    }

    #[test]
    fn every_decode_failure_is_a_miss_not_a_corruption() {
        let plans = vec![plan(vec![0], vec![0])];
        let mut bytes = encode(&plans);
        // Truncation.
        assert!(decode(&bytes[..bytes.len() - 3]).is_err());
        // Trailing bytes.
        bytes.push(0);
        assert!(decode(&bytes).is_err());
        // Version mismatch.
        let mut wrong = encode(&plans);
        wrong[0] = wrong[0].wrapping_add(1);
        assert!(decode(&wrong).is_err());
        // Empty.
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn a_disk_round_trip_survives_a_new_cache_object() {
        let dir = std::env::temp_dir().join(format!(
            "fusor2-plan-cache-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let hash = PlanHash(0xdead_beef_0000_0001u128 << 32);
        let plans: Arc<[CachedKernelPlan]> = vec![plan(vec![0, 1], vec![0, 1])].into();

        let a = PlanCache::new(8, 0x1234, Some(dir.clone()));
        a.insert(hash, plans.clone());
        drop(a);

        // A second process-equivalent open must find it on disk.
        let b = PlanCache::new(8, 0x1234, Some(dir.clone()));
        let hit = b.get(hash).expect("the disk tier must survive a restart");
        assert_eq!(&hit[..], &plans[..]);
        assert_eq!(b.counters().disk_hits, 1);
        assert_eq!(b.counters().memory_hits, 0);

        // And a second lookup is served from memory.
        assert!(b.get(hash).is_some());
        assert_eq!(b.counters().memory_hits, 1);

        // A different salt must not see it.
        let c = PlanCache::new(8, 0x5678, Some(dir.clone()));
        assert!(c.get(hash).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
