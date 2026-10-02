//! The pooled allocator: keyed `(size, usage)` with `strong_count == 1` reuse
//! and a memory ceiling that blocks and retries before failing (on macOS,
//! exceeding unified memory kills the OS).

pub static UPLOAD_UNIFORM: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static UPLOAD_INIT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static COPY_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static POISON_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use fusor_ir::Result;
use fusor_ir::dtype::Persistence;
use fusor_ir::error::Error;
use fusor_ir::target::Buf;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::target::GpuConfig;

/// Minimum idle buffers retained per size class, below its observed peak usage.
pub const FREE_PER_BUCKET: usize = 4;

/// Usage set for a tensor buffer.
pub const TENSOR_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);
/// Usage set for a readback staging buffer.
pub const READBACK_USAGE: wgpu::BufferUsages =
    wgpu::BufferUsages::COPY_DST.union(wgpu::BufferUsages::MAP_READ);

/// Pool key. `usage` is a `wgpu::BufferUsages` bit set.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PoolKey {
    pub size: u64,
    pub usage: u32,
}

impl PoolKey {
    fn of(buf: &Buf) -> Option<Self> {
        buf.downcast_ref::<GpuBuffer>().map(|g| Self {
            size: g.size,
            usage: g.usage.bits(),
        })
    }
}

/// What the pool has done since it was created.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BufferPoolCounters {
    /// Allocation requests, served from the cache or not.
    pub requested: u64,
    /// Buffers actually created on the device.
    pub created: u64,
    /// Bytes currently handed out plus bytes parked in the free lists.
    pub live_bytes: u64,
    /// Times the ceiling forced a `poll(wait_indefinitely)` and a retry.
    pub cap_retries: u64,
}

/// Buffers and peak concurrent usage for one size class.
#[derive(Debug, Default)]
struct Bucket {
    bufs: Vec<Buf>,
    peak: usize,
    last_used: u64,
}

impl Bucket {
    fn take(&mut self, request: u64) -> Option<Buf> {
        let hit = self.bufs.iter().find(|b| b.refcount() == 1).cloned();
        if hit.is_some() {
            self.last_used = self.last_used.max(request);
            self.observe();
            self.peak = self.peak.max(1);
        }
        hit
    }

    /// Note how many of this size are currently handed out.
    fn observe(&mut self) {
        let in_use = self.bufs.iter().filter(|b| b.refcount() > 1).count();
        self.peak = self.peak.max(in_use);
    }

    /// Idle buffers worth keeping: the working set, floored at [`FREE_PER_BUCKET`].
    fn keep(&self) -> usize {
        self.peak.max(FREE_PER_BUCKET)
    }
}

/// A pooled device buffer. `Buf` wraps this in an `Arc<dyn Any>`, so
/// `Buf::refcount() == 1` means the pool holds the only handle.
#[derive(Debug)]
pub struct GpuBuffer {
    pub buffer: wgpu::Buffer,
    pub size: u64,
    pub usage: wgpu::BufferUsages,
}

impl GpuBuffer {
    /// The pooled buffer behind `buf`; `what` names it in the error.
    pub fn of<'a>(buf: &'a Buf, what: &str) -> Result<&'a Self> {
        buf.downcast_ref()
            .ok_or_else(|| Error::Device(format!("{what} is not a pooled buffer")))
    }
}

/// Recycling buffer pool with a hard memory ceiling.
pub struct BufferPool {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Buckets retain all live buffers so `live_bytes` accounts for every allocation.
    free: Mutex<FxHashMap<PoolKey, Bucket>>,
    counters: Mutex<BufferPoolCounters>,
    ceiling_bytes: Mutex<u64>,
    poison: bool,
    upload_staging: Mutex<Vec<StagingChunk>>,
    retired_buffers: AtomicBool,
    lost: crate::device::LostFlag,
}

/// A reusable upload staging buffer kept mapped between uses, so large
/// uploads write resident pages instead of `write_buffer_with`'s fresh ones.
struct StagingChunk {
    buffer: wgpu::Buffer,
    size: u64,
    /// Armed by the `map_async` callback, cleared while the copy is in
    /// flight; a failed remap never re-arms and uploads take the belt.
    mapped: Arc<std::sync::atomic::AtomicBool>,
}

/// Staging chunks kept at most; further uploads take the belt.
const UPLOAD_STAGING_CHUNKS: usize = 4;
/// Below this an upload takes the belt: small copies aren't page-fault bound.
const UPLOAD_STAGING_MIN: u64 = 1 << 20;

impl BufferPool {
    /// Build a pool over a live device; the ceiling defaults to
    /// [`default_ceiling`], overridable by [`GpuConfig::max_gpu_memory_bytes`].
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: &GpuConfig,
        lost: crate::device::LostFlag,
    ) -> Self {
        let ceiling = config.max_gpu_memory_bytes.unwrap_or_else(default_ceiling);
        Self {
            device,
            queue,
            free: Mutex::new(FxHashMap::default()),
            counters: Mutex::new(BufferPoolCounters::default()),
            ceiling_bytes: Mutex::new(ceiling),
            poison: config.poison_allocations,
            upload_staging: Mutex::new(Vec::new()),
            retired_buffers: AtomicBool::new(false),
            lost,
        }
    }

    /// Allocate or recycle a tensor buffer.
    pub fn alloc(&self, bytes: u64, persistence: Persistence) -> Result<Buf> {
        let _ = persistence;
        self.alloc_with_usage(bytes, TENSOR_USAGE)
    }

    /// Allocate or recycle at an explicit usage set, blocking and retrying at
    /// the ceiling (one of the three host syncs).
    pub fn alloc_with_usage(&self, bytes: u64, usage: wgpu::BufferUsages) -> Result<Buf> {
        let size = allocation_size(bytes)?;
        let key = PoolKey {
            size,
            usage: usage.bits(),
        };
        let request = {
            let mut counters = self.counters.lock();
            counters.requested += 1;
            counters.requested
        };

        if let Some(hit) = self.take_free(key, request) {
            return Ok(hit);
        }

        let ceiling = *self.ceiling_bytes.lock();
        // Idle storage gets a small share of the budget so old shape classes
        // don't accumulate.
        self.trim((ceiling / 16).min(256 << 20));
        if self.counters.lock().live_bytes.saturating_add(size) > ceiling {
            // Retire everything in flight, then retry the cache.
            self.counters.lock().cap_retries += 1;
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            self.reclaim();
            if let Some(hit) = self.take_free(key, request) {
                return Ok(hit);
            }
            let live = self.counters.lock().live_bytes;
            if live.saturating_add(size) > ceiling {
                // Who holds the ceiling: pinned and idle bytes per bucket.
                if crate::flags().pool_debug {
                    let free = self.free.lock();
                    let mut rows: Vec<(u64, usize, usize)> = free
                        .iter()
                        .map(|(k, b)| {
                            let pinned = b.bufs.iter().filter(|x| x.refcount() > 1).count();
                            (k.size, pinned, b.bufs.len() - pinned)
                        })
                        .collect();
                    rows.sort_by_key(|(size, pinned, _)| std::cmp::Reverse(size * *pinned as u64));
                    for (size, pinned, idle) in rows.iter().take(12) {
                        eprintln!(
                            "[pool] size {size}: {pinned} pinned ({} MB), {idle} idle",
                            size * *pinned as u64 / (1 << 20)
                        );
                    }
                    let tracked: u64 = free
                        .iter()
                        .map(|(k, b)| k.size.saturating_mul(b.bufs.len() as u64))
                        .sum();
                    let entries: usize = free.values().map(|b| b.bufs.len()).sum();
                    eprintln!(
                        "[pool] tracked {} MB in {entries} entries across {} buckets; live_bytes {} MB",
                        tracked >> 20,
                        free.len(),
                        live >> 20
                    );
                }
                return Err(Error::Device(format!(
                    "gpu allocation of {size} bytes would exceed the {ceiling}-byte ceiling \
                     with {live} bytes live"
                )));
            }
        }

        // A lost device returns an invalid buffer without error; name the loss.
        self.lost.check()?;
        Ok(self.create(size, usage, request))
    }

    /// Upload initial contents, padded to `COPY_BUFFER_ALIGNMENT`.
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> Result<Buf> {
        UPLOAD_INIT.fetch_add(data.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let size = padded_copy_size(data.len() as u64);
        let buf = self.alloc_with_usage(size, usage)?;
        let gpu = GpuBuffer::of(&buf, "uploaded buffer")?;
        if size >= UPLOAD_STAGING_MIN && self.upload_via_staging(gpu, data, size) {
            return Ok(buf);
        }
        match self.queue.write_buffer_with(
            &gpu.buffer,
            0,
            std::num::NonZeroU64::new(size).expect("padded size is nonzero"),
        ) {
            Some(mut view) => {
                // The belt's padding tail is stale, so zero it.
                parallel_copy(view.slice(..data.len()), data);
                view.slice(data.len()..).fill(0);
            }
            None => {
                // The staging belt is full; the plain path pads the same way.
                let mut padded = data.to_vec();
                padded.resize(size as usize, 0);
                self.queue.write_buffer(&gpu.buffer, 0, &padded);
            }
        }
        Ok(buf)
    }

    /// Upload through the pool's remappable staging ring; `false` sends the
    /// caller to the belt. The copy is submitted before the plan's submit, so
    /// dispatches see it; the re-arm completes on the resolve's next poll.
    fn upload_via_staging(&self, dst: &GpuBuffer, data: &[u8], size: u64) -> bool {
        use std::sync::atomic::{AtomicBool, Ordering};
        let chunk = {
            let mut ring = self.upload_staging.lock();
            if let Some(i) = ring
                .iter()
                .position(|c| c.size >= size && c.mapped.load(Ordering::Acquire))
            {
                Some(ring.swap_remove(i))
            } else if ring.len() < UPLOAD_STAGING_CHUNKS {
                let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("fusor upload staging"),
                    size,
                    usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: true,
                });
                Some(StagingChunk {
                    buffer,
                    size,
                    mapped: Arc::new(AtomicBool::new(true)),
                })
            } else {
                None
            }
        };
        let Some(chunk) = chunk else {
            return false;
        };
        {
            let mut view = chunk.buffer.slice(0..size).get_mapped_range_mut();
            parallel_copy(view.slice(..data.len()), data);
            view.slice(data.len()..).fill(0);
        }
        chunk.buffer.unmap();
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fusor upload staging copy"),
            });
        enc.copy_buffer_to_buffer(&chunk.buffer, 0, &dst.buffer, 0, size);
        self.queue.submit([enc.finish()]);
        chunk.mapped.store(false, Ordering::Release);
        let armed = Arc::clone(&chunk.mapped);
        chunk
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Write, move |r| {
                if r.is_ok() {
                    armed.store(true, Ordering::Release);
                }
            });
        self.upload_staging.lock().push(chunk);
        true
    }

    /// Return a buffer whose only remaining handle is the caller's; one still
    /// referenced elsewhere is never handed out twice.
    pub fn recycle(&self, buf: Buf) {
        let Some(key) = PoolKey::of(&buf) else {
            return;
        };
        let addr = buf.addr();
        // Pool-created buffers are already tracked, so this only drops the
        // caller's clone; a foreign handle is adopted.
        let released = {
            let mut free = self.free.lock();
            let bucket = free.entry(key).or_default();
            if !bucket.bufs.iter().any(|b| b.addr() == addr) {
                bucket.bufs.push(buf);
            }
            prune_bucket(bucket)
        };
        self.released(released.saturating_mul(key.size), 0);
    }

    /// Forget `buf` entirely so it is destroyed with the caller's clone: for
    /// a buffer in an unknown state, e.g. staging whose map failed part-way.
    pub fn discard(&self, buf: Buf) {
        let Some(key) = PoolKey::of(&buf) else {
            return;
        };
        let addr = buf.addr();
        let removed = {
            let mut free = self.free.lock();
            match free.get_mut(&key) {
                Some(bucket) => {
                    let before = bucket.bufs.len();
                    bucket.bufs.retain(|b| b.addr() != addr);
                    before - bucket.bufs.len()
                }
                None => 0,
            }
        };
        if removed > 0 {
            let mut counters = self.counters.lock();
            counters.live_bytes = counters.live_bytes.saturating_sub(key.size);
        }
        drop(buf);
    }

    /// Drop every free buffer only the pool holds.
    pub fn reclaim(&self) {
        self.trim(0);
    }

    fn trim(&self, idle_budget: u64) {
        let released = trim_idle(&mut self.free.lock(), idle_budget);
        self.released(released, 0);
    }

    /// Account `released` bytes dropped from the buckets and `created` new
    /// ones; a drop also flags the bind group cache for pruning.
    fn released(&self, released: u64, created: u64) {
        if released > 0 {
            self.retired_buffers.store(true, Ordering::Release);
        }
        let mut counters = self.counters.lock();
        counters.live_bytes = counters
            .live_bytes
            .saturating_add(created)
            .saturating_sub(released);
    }

    pub(crate) fn take_retired_buffers(&self) -> bool {
        self.retired_buffers.swap(false, Ordering::Acquire)
    }

    /// Refill idle buffers with `0xCD` after a resolve (when poisoning), so
    /// zero-init assumptions fail loudly.
    pub fn repoison_free_buffers(&self) {
        if !self.poison {
            return;
        }
        let free = self.free.lock();
        for bucket in free.values() {
            // In-use buffers are tracked too; never poison a live tensor.
            for buf in bucket.bufs.iter().filter(|b| b.refcount() == 1) {
                if let Some(gpu) = buf.downcast_ref::<GpuBuffer>() {
                    self.poison_fill(gpu);
                }
            }
        }
    }

    pub fn ceiling(&self) -> u64 {
        *self.ceiling_bytes.lock()
    }

    pub fn counters(&self) -> BufferPoolCounters {
        *self.counters.lock()
    }

    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    fn take_free(&self, key: PoolKey, request: u64) -> Option<Buf> {
        let mut free = self.free.lock();
        let bucket = free.get_mut(&key)?;
        // `refcount() == 1` means no caller holds it; the clone stays tracked,
        // so a dropped buffer is reusable without `recycle`.
        bucket.take(request)
    }

    fn create(&self, size: u64, usage: wgpu::BufferUsages, request: u64) -> Buf {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fusor pooled buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });
        let gpu = GpuBuffer {
            buffer,
            size,
            usage,
        };
        if self.poison {
            self.poison_fill(&gpu);
        }
        let buf = Buf::new(gpu);
        // The pool keeps a handle to everything it creates, or unrecycled plan
        // outputs and leaves would be recreated every resolve.
        let key = PoolKey {
            size,
            usage: usage.bits(),
        };
        let released = {
            let mut free = self.free.lock();
            let bucket = free.entry(key).or_default();
            bucket.last_used = bucket.last_used.max(request);
            // Creating means the working set exceeds what the bucket holds.
            bucket.observe();
            bucket.peak = bucket.peak.saturating_add(1);
            let released = prune_bucket(bucket);
            bucket.bufs.push(buf.clone());
            released
        };
        self.counters.lock().created += 1;
        self.released(released.saturating_mul(size), size);
        buf
    }

    /// Pre-fill with `0xCD` so zero-init assumptions fail loudly.
    fn poison_fill(&self, gpu: &GpuBuffer) {
        POISON_BYTES.fetch_add(gpu.size, std::sync::atomic::Ordering::Relaxed);
        if !gpu.usage.contains(wgpu::BufferUsages::COPY_DST) {
            return;
        }
        let chunk = vec![0xCDu8; gpu.size.min(1 << 20) as usize];
        let mut offset = 0u64;
        while offset < gpu.size {
            let len = chunk.len().min((gpu.size - offset) as usize);
            self.queue.write_buffer(&gpu.buffer, offset, &chunk[..len]);
            offset += len as u64;
        }
    }
}

fn trim_idle(buckets: &mut FxHashMap<PoolKey, Bucket>, budget: u64) -> u64 {
    let mut idle = 0u64;
    let mut oldest: Vec<_> = buckets
        .iter()
        .filter_map(|(key, bucket)| {
            let count = bucket.bufs.iter().filter(|b| b.refcount() == 1).count() as u64;
            idle = idle.saturating_add(key.size.saturating_mul(count));
            (count > 0).then_some((bucket.last_used, *key))
        })
        .collect();
    if idle <= budget {
        return 0;
    }
    oldest.sort_unstable_by_key(|(used, key)| (*used, key.size, key.usage));
    let mut released = 0u64;
    for (_, key) in oldest {
        let bucket = buckets.get_mut(&key).expect("collected above");
        bucket.bufs.retain(|b| {
            if idle > budget && b.refcount() == 1 {
                idle = idle.saturating_sub(key.size);
                released = released.saturating_add(key.size);
                false
            } else {
                true
            }
        });
        bucket.peak = bucket.peak.min(bucket.bufs.len());
        if bucket.bufs.is_empty() {
            buckets.remove(&key);
        }
        if idle <= budget {
            break;
        }
    }
    released
}

/// Drop idle entries past the bucket's keep, returning how many went. An
/// entry a caller still holds is always kept: the pool's clone tracks it.
fn prune_bucket(bucket: &mut Bucket) -> u64 {
    let keep = bucket.keep();
    let mut idle = 0usize;
    let mut released = 0u64;
    bucket.bufs.retain(|b| {
        if b.refcount() > 1 {
            return true;
        }
        idle += 1;
        if idle <= keep {
            true
        } else {
            released += 1;
            false
        }
    });
    released
}

/// Copy `src` into a staging view with up to six threads: fresh pages copy at
/// page-fault speed, which parallelizes. Small uploads never spawn.
fn parallel_copy(mut dst: wgpu::WriteOnly<'_, [u8]>, src: &[u8]) {
    const PARALLEL_COPY_CHUNK: usize = 2 << 20;
    let threads = if cfg!(target_arch = "wasm32") {
        1
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(src.len().div_ceil(PARALLEL_COPY_CHUNK))
            .min(6)
    };
    if threads <= 1 {
        dst.copy_from_slice(src);
        return;
    }
    /// A base pointer a copy worker writes through; ranges are disjoint.
    struct SendPtr(*mut u8);
    unsafe impl Send for SendPtr {}
    let base = dst.as_raw_element_ptr().as_ptr();
    let per = src.len().div_ceil(threads);
    std::thread::scope(|scope| {
        for chunk in 0..threads {
            let start = chunk * per;
            let end = ((chunk + 1) * per).min(src.len());
            if start >= end {
                break;
            }
            let part = &src[start..end];
            let to = SendPtr(unsafe { base.add(start) });
            scope.spawn(move || {
                // Rebind the whole struct: disjoint capture would take the
                // non-`Send` field `to.0` alone.
                let to = to;
                unsafe { std::ptr::copy_nonoverlapping(part.as_ptr(), to.0, part.len()) };
            });
        }
    });
}

pub fn padded_copy_size(bytes: u64) -> u64 {
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    bytes.div_ceil(align).max(1) * align
}

/// Geometric size classes, at most 64 KiB apart; copy lengths stay exact.
fn allocation_size(bytes: u64) -> Result<u64> {
    let bytes = bytes.max(wgpu::COPY_BUFFER_ALIGNMENT);
    let quantum = 1u64 << bytes.ilog2().saturating_sub(4).clamp(2, 16);
    bytes
        .checked_next_multiple_of(quantum)
        .ok_or_else(|| Error::Device("buffer allocation size overflows u64".into()))
}

/// The platform memory ceiling: two thirds of `hw.memsize` on Apple silicon,
/// where exceeding unified memory panics macOS; elsewhere unlimited.
pub fn default_ceiling() -> u64 {
    // A browser tab reports no budget, and without a ceiling `reclaim` never
    // runs.
    #[cfg(target_arch = "wasm32")]
    {
        512 << 20
    }
    #[cfg(all(not(target_arch = "wasm32"), target_vendor = "apple"))]
    {
        if let Some(total) = hw_memsize() {
            return total / 3 * 2;
        }
        u64::MAX
    }
    #[cfg(all(not(target_arch = "wasm32"), not(target_vendor = "apple")))]
    {
        u64::MAX
    }
}

#[cfg(all(not(target_arch = "wasm32"), target_vendor = "apple"))]
fn hw_memsize() -> Option<u64> {
    // SAFETY: `sysctlbyname` writes at most `len` bytes into the live `u64`
    // and reads a NUL-terminated name.
    unsafe {
        unsafe extern "C" {
            fn sysctlbyname(
                name: *const std::ffi::c_char,
                oldp: *mut std::ffi::c_void,
                oldlenp: *mut usize,
                newp: *mut std::ffi::c_void,
                newlen: usize,
            ) -> std::ffi::c_int;
        }
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = c"hw.memsize";
        let rc = sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &raw mut len,
            std::ptr::null_mut(),
            0,
        );
        (rc == 0 && value > 0).then_some(value)
    }
}

// Explicit auto-trait impls, as on `GpuTarget` (E0275 in `Send` futures).
//
// SAFETY: the `*_fields_are_send_sync` functions assert `Send + Sync` for
// every field type, exactly what the auto impls would require.
unsafe impl Send for BufferPool {}
unsafe impl Sync for BufferPool {}
unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

#[allow(dead_code)]
fn pool_fields_are_send_sync() {
    fn assert<T: Send + Sync>() {}
    assert::<Arc<wgpu::Device>>();
    assert::<Arc<wgpu::Queue>>();
    assert::<Mutex<FxHashMap<PoolKey, Bucket>>>();
    assert::<Mutex<BufferPoolCounters>>();
    assert::<Mutex<u64>>();
    assert::<bool>();
    assert::<Mutex<Vec<StagingChunk>>>();
    assert::<AtomicBool>();
    assert::<crate::device::LostFlag>();
    assert::<wgpu::Buffer>();
    assert::<u64>();
    assert::<wgpu::BufferUsages>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idle_budget_bounds_growing_kv_scratch_and_keeps_live_buffers() {
        let mut buckets = FxHashMap::default();
        let pinned = Buf::new(());
        let pinned_key = PoolKey {
            size: 2 << 30,
            usage: TENSOR_USAGE.bits(),
        };
        buckets.insert(
            pinned_key,
            Bucket {
                bufs: vec![pinned.clone()],
                ..Bucket::default()
            },
        );
        let budget = 256 << 20;
        for len in 1..=8192 {
            let key = PoolKey {
                size: allocation_size(8 * len * 128 * 4).unwrap(),
                usage: TENSOR_USAGE.bits(),
            };
            if buckets.get_mut(&key).and_then(|b| b.take(len)).is_some() {
                continue;
            }
            let before: u64 = buckets
                .iter()
                .map(|(k, b)| k.size * b.bufs.len() as u64)
                .sum();
            let released = trim_idle(&mut buckets, budget);
            let after: u64 = buckets
                .iter()
                .map(|(k, b)| k.size * b.bufs.len() as u64)
                .sum();
            assert_eq!(before - after, released);
            assert!(after <= pinned_key.size + budget);
            buckets.insert(
                key,
                Bucket {
                    bufs: vec![Buf::new(())],
                    last_used: len,
                    ..Bucket::default()
                },
            );
        }
        trim_idle(&mut buckets, 0);
        assert_eq!(buckets.len(), 1);
        assert_eq!(buckets[&pinned_key].bufs[0].addr(), pinned.addr());
    }

    #[test]
    fn reuse_keeps_a_size_class_ahead_of_older_idle_storage() {
        let keys = [64, 128, 256].map(|size| PoolKey { size, usage: 0 });
        let mut buckets: FxHashMap<_, _> = keys
            .into_iter()
            .enumerate()
            .map(|(i, key)| {
                (
                    key,
                    Bucket {
                        bufs: vec![Buf::new(())],
                        last_used: i as u64,
                        ..Bucket::default()
                    },
                )
            })
            .collect();
        drop(buckets.get_mut(&keys[0]).unwrap().take(3).unwrap());
        assert_eq!(trim_idle(&mut buckets, 320), 128);
        assert!(buckets.contains_key(&keys[0]));
        assert!(!buckets.contains_key(&keys[1]));
        assert!(buckets.contains_key(&keys[2]));
    }

    #[test]
    fn growing_attention_scratch_reuses_size_classes() {
        let capacities: rustc_hash::FxHashSet<_> = (1..=8192)
            .map(|len| allocation_size(32 * len * 4).unwrap())
            .collect();
        assert!(capacities.len() <= 160);
        assert!(capacities.iter().sum::<u64>() < 25 << 20);
    }

    #[test]
    fn allocation_slack_is_bounded_without_padding_copy_lengths() {
        let boundaries = (2..=62).flat_map(|shift| {
            let size = 1u64 << shift;
            [size - 1, size, size + 1]
        });
        for bytes in (0..=256).chain(boundaries).chain([4_700_000_000]) {
            let capacity = allocation_size(bytes).unwrap();
            let copied = padded_copy_size(bytes);
            assert!(capacity >= copied);
            assert_eq!(capacity % wgpu::COPY_BUFFER_ALIGNMENT, 0);
            assert_eq!(allocation_size(capacity).unwrap(), capacity);
            assert!(capacity - bytes < 65536);
            assert!(capacity - bytes <= (bytes / 16).max(4));
            assert!(copied - bytes <= 4);
        }
        assert!(allocation_size(u64::MAX).is_err());
    }
}
