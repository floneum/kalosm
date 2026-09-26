//! The pooled allocator: keyed `(size, usage)` with `strong_count == 1` reuse
//! and a platform memory ceiling that blocks and retries before failing.
//! On macOS, exceeding unified memory kills the OS rather than erroring, which
//! is why the ceiling is a hard gate and not a warning.

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

    /// Idle buffers worth keeping: the working set, floored at
    /// [`FREE_PER_BUCKET`].
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

/// A reusable upload staging buffer, kept mapped between uses.
///
/// `queue.write_buffer_with`'s staging is a fresh allocation per call, so a
/// large upload memcpy runs at page-fault speed. Writing into the same mapped
/// buffer every time keeps the pages resident; the chunk is unmapped only
/// while its copy is in flight and `map_async` re-arms it during the
/// resolve's own wait.
struct StagingChunk {
    buffer: wgpu::Buffer,
    size: u64,
    /// Armed by the `map_async` callback; cleared while the copy is in
    /// flight. A chunk whose remap failed simply never re-arms, and uploads
    /// fall back to the belt.
    mapped: Arc<std::sync::atomic::AtomicBool>,
}

/// Staging chunks kept alive at most. Sized by the largest concurrent
/// uploads a resolve issues; anything past this takes the belt path.
const UPLOAD_STAGING_CHUNKS: usize = 4;
/// Below this an upload takes the belt: a small copy is not page-fault bound
/// and the belt already batches it into the next submit.
const UPLOAD_STAGING_MIN: u64 = 1 << 20;

impl BufferPool {
    /// Build a pool over a live device.
    ///
    /// The ceiling is `hw.memsize / 3 * 2` on Apple silicon and `u64::MAX`
    /// elsewhere, overridable by [`GpuConfig::max_gpu_memory_bytes`].
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

    /// Allocate or recycle at an explicit usage set.
    ///
    /// Blocks and retries at the ceiling rather than failing — one of exactly
    /// three host syncs in the whole runtime.
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
        // Idle storage gets a small share of the device budget; old shape
        // classes must not accumulate up to the device's allocation ceiling.
        self.trim((ceiling / 16).min(256 << 20));
        if self.counters.lock().live_bytes.saturating_add(size) > ceiling {
            // Retire everything in flight, then retry the cache. Only after
            // both fail is the working set genuinely over the cap.
            self.counters.lock().cap_retries += 1;
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            self.reclaim();
            if let Some(hit) = self.take_free(key, request) {
                return Ok(hit);
            }
            let live = self.counters.lock().live_bytes;
            if live.saturating_add(size) > ceiling {
                // Who holds the ceiling: every bucket's pinned (refcount > 1)
                // and idle bytes, largest first.
                if std::env::var_os("FUSOR_POOL_DEBUG").is_some() {
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

        // A lost device hands back an invalid handle from `create_buffer`
        // without an error; the first write to it would then fail as a
        // validation panic with no mention of the loss.
        self.lost.check()?;
        Ok(self.create(size, usage, request))
    }

    /// Upload initial contents through `queue.write_buffer_with`, padding to
    /// `COPY_BUFFER_ALIGNMENT`.
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> Result<Buf> {
        UPLOAD_INIT.fetch_add(data.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let size = padded_copy_size(data.len() as u64);
        let buf = self.alloc_with_usage(size, usage)?;
        let gpu = buf
            .downcast_ref::<GpuBuffer>()
            .ok_or_else(|| Error::Device("pool handed back a foreign buffer".into()))?;
        if size >= UPLOAD_STAGING_MIN && self.upload_via_staging(gpu, data, size) {
            return Ok(buf);
        }
        match self.queue.write_buffer_with(
            &gpu.buffer,
            0,
            std::num::NonZeroU64::new(size).expect("padded size is nonzero"),
        ) {
            Some(mut view) => {
                // Write straight into the staging belt: the padding tail is
                // whatever the belt held, so it is zeroed explicitly.
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

    /// Upload through the pool's own remappable staging ring. `false` when no
    /// chunk is ready and the ring is full — the caller takes the belt.
    ///
    /// Ordering: the copy is submitted *here*, before the plan's own submit,
    /// and wgpu executes submissions in order, so a dispatch can never read
    /// the destination before the copy. The `map_async` re-arm completes on
    /// any later device poll — every resolve ends in one (readback or
    /// `poll_wait`), which is what keeps the ring warm with no poll of its
    /// own.
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

    /// Return a buffer whose only remaining handle is the caller's.
    ///
    /// Reuse is gated on `strong_count == 1`: a buffer still referenced by a
    /// live tensor is dropped from the pool's view rather than handed out
    /// twice.
    pub fn recycle(&self, buf: Buf) {
        // `map` ends the borrow before `buf` may be moved into the bucket.
        let Some((size, usage)) = buf
            .downcast_ref::<GpuBuffer>()
            .map(|g| (g.size, g.usage.bits()))
        else {
            return;
        };
        let key = PoolKey { size, usage };
        let addr = buf.addr();
        // Everything this pool created is already tracked, so recycling is
        // dropping the caller's clone; only a foreign handle is adopted. A
        // tracked buffer has refcount 2 (pool + caller) here, and a caller
        // that still holds another clone fails `take_free`'s `refcount() == 1`
        // test until it drops it.
        let released = {
            let mut free = self.free.lock();
            let bucket = free.entry(key).or_default();
            if !bucket.bufs.iter().any(|b| b.addr() == addr) {
                bucket.bufs.push(buf);
            }
            prune_bucket(bucket)
        };
        if released > 0 {
            self.retired_buffers.store(true, Ordering::Release);
            let mut counters = self.counters.lock();
            counters.live_bytes = counters
                .live_bytes
                .saturating_sub(released.saturating_mul(key.size));
        }
    }

    /// Forget `buf` entirely: the pool drops its own handle, so the driver
    /// buffer is destroyed as soon as the caller's clone goes.
    ///
    /// For a buffer whose state is no longer known to be clean — a readback
    /// staging buffer whose map failed part-way is still marked mapped on
    /// the wgpu side, and the next `map_async` on it would panic. Such a
    /// buffer must never be handed out again.
    pub fn discard(&self, buf: Buf) {
        let Some((size, usage)) = buf
            .downcast_ref::<GpuBuffer>()
            .map(|g| (g.size, g.usage.bits()))
        else {
            return;
        };
        let key = PoolKey { size, usage };
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
            counters.live_bytes = counters.live_bytes.saturating_sub(size);
        }
        drop(buf);
    }

    /// Drop every free buffer whose only handle is the pool's, releasing their
    /// bytes back to the ceiling budget.
    pub fn reclaim(&self) {
        self.trim(0);
    }

    fn trim(&self, idle_budget: u64) {
        let released = trim_idle(&mut self.free.lock(), idle_budget);
        if released > 0 {
            self.retired_buffers.store(true, Ordering::Release);
        }
        let mut counters = self.counters.lock();
        counters.live_bytes = counters.live_bytes.saturating_sub(released);
    }

    pub(crate) fn take_retired_buffers(&self) -> bool {
        self.retired_buffers.swap(false, Ordering::Acquire)
    }

    /// Refill every free buffer with `0xCD` at the end of a resolve, so the
    /// next tenant that assumes zero-initialized storage fails loudly. A
    /// no-op unless poisoning is on.
    pub fn repoison_free_buffers(&self) {
        if !self.poison {
            return;
        }
        let free = self.free.lock();
        for bucket in free.values() {
            // The pool tracks in-use buffers now; poisoning one would
            // overwrite a live tensor.
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
        // The pool holds its own handle, so `refcount() == 1` is exactly "no
        // caller has this one". Handing back a clone leaves the entry tracked,
        // which makes a dropped buffer reusable with no `recycle` call.
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
        // The pool keeps its own handle to everything it creates. Without it,
        // a buffer that is never explicitly `recycle`d is destroyed with its
        // last caller handle and re-created from the driver next resolve —
        // and nothing recycles a plan output or an uploaded leaf.
        let key = PoolKey {
            size,
            usage: usage.bits(),
        };
        let released = {
            let mut free = self.free.lock();
            let bucket = free.entry(key).or_default();
            bucket.last_used = bucket.last_used.max(request);
            // Creating means the bucket could not serve the request, so its
            // working set is at least one larger than what it holds.
            bucket.observe();
            bucket.peak = bucket.peak.saturating_add(1);
            let released = prune_bucket(bucket);
            bucket.bufs.push(buf.clone());
            released
        };
        if released > 0 {
            self.retired_buffers.store(true, Ordering::Release);
        }
        let mut counters = self.counters.lock();
        counters.created += 1;
        counters.live_bytes = counters
            .live_bytes
            .saturating_add(size)
            .saturating_sub(released.saturating_mul(size));
        buf
    }

    /// Pre-fill with `0xCD` so a kernel that assumes zero-initialized storage
    /// fails loudly instead of reading whatever the last tenant left.
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

/// Drop idle entries past [`FREE_PER_BUCKET`], returning how many were
/// released so the caller can decrement `live_bytes`.
///
/// An entry with an outstanding caller handle (`refcount() > 1`) is **always**
/// kept: the pool's clone is what tracks the buffer, and dropping it would
/// untrack a live allocation and lose the reuse this pool exists for.
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

/// Copy `src` into a staging view with one thread per ~4 MB chunk.
///
/// A staging allocation is fresh pages, so a serial `copy_from_slice` runs at
/// page-fault speed. Page faults parallelize almost linearly; threads are
/// only spawned past [`PARALLEL_COPY_CHUNK`], so small uploads never pay a
/// spawn.
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
    /// A base pointer a copy worker may write through. The parent's
    /// `WriteOnly` view outlives the scope and each worker owns a disjoint
    /// range, so the writes never alias.
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
                // Rebind the whole struct first: RFC 2229 disjoint capture
                // would otherwise capture the field `to.0` alone — a bare
                // pointer, which is not `Send`.
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

/// Geometric size classes, capped at 64 KiB between capacities.
/// Copy and readback lengths remain independent of allocation capacity.
fn allocation_size(bytes: u64) -> Result<u64> {
    let bytes = bytes.max(wgpu::COPY_BUFFER_ALIGNMENT);
    let quantum = 1u64 << bytes.ilog2().saturating_sub(4).clamp(2, 16);
    bytes
        .checked_next_multiple_of(quantum)
        .ok_or_else(|| Error::Device("buffer allocation size overflows u64".into()))
}

/// The platform memory ceiling.
///
/// On Apple silicon, exceeding unified memory panics macOS rather than
/// returning an error, so two thirds of `hw.memsize` is a hard gate. Elsewhere
/// the driver reports allocation failure and the pool does not need to guess.
pub fn default_ceiling() -> u64 {
    // A browser tab has no business holding a workstation's worth of GPU
    // buffers, and nothing there reports how much it may have. Without a
    // ceiling `reclaim` never runs, so a bucket's retained working set is
    // whatever the largest thing that ever happened needed.
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
    // SAFETY: `sysctlbyname` writes at most `len` bytes into `value`, which is
    // a live `u64`, and reads a NUL-terminated name. Both preconditions hold
    // by construction here. There is no safe wrapper in std for this.
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

// Explicit auto-trait impls; see the note on `GpuTarget`. Without them the
// auto-trait walk recurses through wgpu_core's resource graph and overflows
// (E0275) in downstream crates whose `Send` futures reach a pool or buffer.
//
// SAFETY: the `*_fields_are_send_sync` functions assert `Send + Sync` for
// every field type, which is exactly what the auto impls would require.
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
