//! The plan cache: memory LRU plus a disk tier, salted by exe identity and the
//! `DeviceFacts` fingerprint — which includes `max_compute_workgroup_storage_size`.
//!
//! A [`CachedKernelPlan`] is bufferless: the compiled template plus, per
//! binding slot, the caller-buffer index it wants, plus, per caller position,
//! the first position holding the same buffer. `record_plan` refuses (silently,
//! returning `None`) to record a kernel that binds a buffer the caller did not
//! present, and [`binding_shape_matches`] requires the caller's aliasing
//! pattern to reproduce the recorded one exactly — a kernel body is only
//! correct for callers with the identical aliasing pattern.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use fusor_ir::Result;
use fusor_ir::cost::DeviceFacts;
use fusor_ir::error::Error;
use fusor_ir::extract::PlanHash;
use fusor_ir::target::Buf;
use parking_lot::Mutex;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

/// Bumped whenever the on-disk record layout changes. A mismatch is a miss.
pub(crate) const DISK_PLAN_FORMAT_VERSION: u32 = 1;
/// Memory LRU capacity.
pub(crate) const MEMORY_CAPACITY: usize = if cfg!(target_arch = "wasm32") {
    512
} else {
    4096
};
/// Salt directories untouched this long are removed on open.
pub(crate) const SALT_TTL: Duration = Duration::from_secs(14 * 24 * 60 * 60);

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
    /// Plans currently held in the memory tier.
    pub fn memory_len(&self) -> usize {
        self.memory.lock().len()
    }

    /// Build a cache. `dir` is the root under which `fusor/plans/<salt>/`
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

    /// Build the cached plan record for one kernel.
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
            let pos = caller_buffers
                .iter()
                .position(|cb| cb.addr() == kb.addr())?;
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
        if self.permutation.iter().any(|i| *i >= caller_buffers.len()) {
            return false;
        }
        alias_classes(caller_buffers) == self.alias_class
    }
}

/// Per position, the first position holding the same buffer.
pub(crate) fn alias_classes(buffers: &[Buf]) -> Vec<usize> {
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
pub(crate) fn disk_salt(facts: &DeviceFacts) -> u64 {
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

/// One file per [`PlanHash`] under `<dir>/fusor/plans/<salt>/<hi><lo>.plan`.
pub(crate) struct DiskPlanCache {
    root: PathBuf,
}

impl DiskPlanCache {
    /// Open (and lazily create) the salt directory, removing salt directories
    /// untouched for [`SALT_TTL`].
    pub(crate) fn open(dir: PathBuf, salt: u64) -> Self {
        let plans = dir.join("fusor").join("plans");
        let root = plans.join(format!("{salt:016x}"));
        let _ = std::fs::create_dir_all(&root);
        gc_stale_salts(&plans, &root);
        Self { root }
    }

    pub(crate) fn path_for(&self, hash: PlanHash) -> PathBuf {
        let hi = (hash.0 >> 64) as u64;
        let lo = hash.0 as u64;
        self.root.join(format!("{hi:016x}{lo:016x}.plan"))
    }

    /// Every failure — missing, truncated, version mismatch, trailing bytes —
    /// is a miss.
    pub(crate) fn load(&self, hash: PlanHash) -> Option<Vec<CachedKernelPlan>> {
        let bytes = std::fs::read(self.path_for(hash)).ok()?;
        decode(&bytes).ok()
    }

    /// Atomic temp-file plus rename, so a crashed write is never observed as a
    /// truncated record.
    pub(crate) fn store(&self, hash: PlanHash, plans: &[CachedKernelPlan]) {
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
pub(crate) fn encode(plans: &[CachedKernelPlan]) -> Vec<u8> {
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
pub(crate) fn decode(bytes: &[u8]) -> Result<Vec<CachedKernelPlan>> {
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
