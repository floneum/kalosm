//! Persistent cache of lowered direct-kernel plans.
//!
//! Kernel construction (tile-ir program building and Naga lowering) dominates
//! first-sight resolves; the resulting plans are pure functions of the
//! 128-bit structural [`KernelCacheKey`], so they can be reused across
//! processes. Plans are stored bufferless (the in-memory plan cache rebinds
//! the caller's buffers positionally) as one file per key under a salt
//! directory that encodes the executable identity and the device capability
//! fingerprint — any compiler change or capability change starts a cold
//! cache rather than risking a stale kernel.
//!
//! Every failure path (missing file, decode error, version or key mismatch,
//! revalidation failure) falls back to rebuilding the kernel, so the cache
//! can only miss, never corrupt.

use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use rustc_hash::FxHasher;
use serde::{Deserialize, Serialize};

use crate::cache::KernelCacheKey;

pub(crate) const DISK_PLAN_FORMAT_VERSION: u32 = 3;
/// Salt directories untouched for this long are removed on open: they belong
/// to executables that have since been rebuilt.
const STALE_SALT_AGE: std::time::Duration = std::time::Duration::from_secs(14 * 24 * 60 * 60);

#[derive(Serialize, Deserialize)]
pub(crate) struct DiskPlanFile {
    pub(crate) format: u32,
    pub(crate) key: [u64; 2],
    pub(crate) plans: Vec<DiskPlan>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct DiskPlan {
    /// Caller-buffer index per kernel binding slot.
    pub(crate) permutation: Vec<usize>,
    /// For each caller-buffer position, the first position holding the same
    /// buffer at record time.
    pub(crate) alias_class: Vec<usize>,
    pub(crate) template: DiskTemplate,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct DiskTemplate {
    pub(crate) name: String,
    pub(crate) dispatch_size: [u32; 3],
    pub(crate) kind: DiskTemplateKind,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum DiskTemplateKind {
    Dynamic {
        module: naga::Module,
        subgroups: bool,
        /// (binding index, read only) per buffer, in binding order.
        bindings: Vec<(u32, bool)>,
    },
    /// The singleton three-buffer (input, weight, output) fast-path layout;
    /// the pipeline is rebuilt from the module on revival.
    Storage3 {
        module: naga::Module,
        subgroups: bool,
    },
    Sequence(Vec<DiskTemplate>),
}

pub(crate) struct DiskPlanCache {
    dir: PathBuf,
}

impl DiskPlanCache {
    /// Open (creating if needed) the plan directory for this executable and
    /// device fingerprint, or `None` when its location is unresolvable.
    /// `dir_override` replaces the platform cache directory when set.
    pub(crate) fn open(device_fingerprint: u64, dir_override: Option<PathBuf>) -> Option<Self> {
        let base = match dir_override {
            Some(dir) => dir,
            None => default_cache_dir()?,
        };
        remove_stale_salts(&base);
        let dir = base.join(format!("{:016x}", salt(device_fingerprint)?));
        std::fs::create_dir_all(&dir).ok()?;
        Some(Self { dir })
    }

    fn path(&self, key: KernelCacheKey) -> PathBuf {
        let [a, b] = key.parts();
        self.dir.join(format!("{a:016x}{b:016x}.plan"))
    }

    pub(crate) fn load(&self, key: KernelCacheKey) -> Option<DiskPlanFile> {
        let bytes = std::fs::read(self.path(key)).ok()?;
        let file: DiskPlanFile = bincode::deserialize(&bytes).ok()?;
        (file.format == DISK_PLAN_FORMAT_VERSION && file.key == key.parts()).then_some(file)
    }

    /// Persist a plan. The write is synchronous — callers run on the
    /// parallel kernel-build workers, and a detached write racing process
    /// exit would silently drop exactly the largest (most valuable) plans.
    /// It is atomic (temp file + rename) so concurrent processes see either
    /// the whole file or none.
    pub(crate) fn store(&self, file: DiskPlanFile) {
        let path = self.path(KernelCacheKey::from_parts(file.key));
        let Ok(bytes) = bincode::serialize(&file) else {
            return;
        };
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        if std::fs::write(&tmp, bytes).is_ok() {
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

/// A salt covering everything that changes generated kernels: the exact
/// executable (any rebuild may change kernel emission) and the device
/// capability fingerprint (features and limits steer codegen).
fn salt(device_fingerprint: u64) -> Option<u64> {
    let exe = std::env::current_exe().ok()?;
    let meta = std::fs::metadata(&exe).ok()?;
    let mut hasher = FxHasher::default();
    exe.hash(&mut hasher);
    meta.len().hash(&mut hasher);
    meta.modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos()
        .hash(&mut hasher);
    device_fingerprint.hash(&mut hasher);
    Some(hasher.finish())
}

fn default_cache_dir() -> Option<PathBuf> {
    #[cfg(target_vendor = "apple")]
    let base = std::env::var_os("HOME").map(|home| PathBuf::from(home).join("Library/Caches"));
    #[cfg(not(target_vendor = "apple"))]
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")));
    Some(base?.join("fusor-ml/kernel-plans"))
}

fn remove_stale_salts(base: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(base) else {
        return;
    };
    let now = std::time::SystemTime::now();
    for entry in entries.flatten() {
        let stale = entry
            .metadata()
            .ok()
            .and_then(|meta| meta.modified().ok())
            .is_some_and(|modified| {
                now.duration_since(modified)
                    .is_ok_and(|age| age > STALE_SALT_AGE)
            });
        if stale {
            let _ = std::fs::remove_dir_all(entry.path());
        }
    }
}
