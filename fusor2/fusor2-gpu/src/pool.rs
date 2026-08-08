//! The pooled allocator: keyed `(size, usage)` with `strong_count == 1` reuse
//! and a platform memory ceiling that blocks and retries before failing. On
//! macOS, exceeding unified memory kills the OS rather than erroring, so the
//! ceiling is a hard gate.

use std::num::NonZeroUsize;
use std::sync::Arc;

use fusor2_ir::Result;
use fusor2_ir::dtype::Persistence;
use fusor2_ir::error::Error;
use fusor2_ir::target::Buf;
use parking_lot::Mutex;

use crate::target::GpuConfig;

/// Buckets kept alive in the LRU.
pub const POOL_BUCKETS: usize = if cfg!(target_arch = "wasm32") { 32 } else { 128 };
/// Free buffers retained per bucket.
pub const FREE_PER_BUCKET: usize = if cfg!(target_arch = "wasm32") { 1 } else { 4 };

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
    free: Mutex<lru::LruCache<PoolKey, Vec<Buf>>>,
    counters: Mutex<BufferPoolCounters>,
    ceiling_bytes: Mutex<u64>,
    poison: bool,
}

impl BufferPool {
    /// Build a pool over a live device.
    ///
    /// The ceiling is `hw.memsize / 3 * 2` on Apple silicon and `u64::MAX`
    /// elsewhere, overridable by [`GpuConfig::max_gpu_memory_bytes`].
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, config: &GpuConfig) -> Self {
        let ceiling = config.max_gpu_memory_bytes.unwrap_or_else(default_ceiling);
        Self {
            device,
            queue,
            free: Mutex::new(lru::LruCache::new(
                NonZeroUsize::new(POOL_BUCKETS).expect("POOL_BUCKETS is nonzero"),
            )),
            counters: Mutex::new(BufferPoolCounters::default()),
            ceiling_bytes: Mutex::new(ceiling),
            poison: config.poison_allocations,
        }
    }

    /// Allocate or recycle a tensor buffer.
    pub fn alloc(&self, bytes: u64, persistence: Persistence) -> Result<Buf> {
        let _ = persistence;
        self.alloc_with_usage(bytes, TENSOR_USAGE)
    }

    /// Allocate or recycle at an explicit usage set.
    ///
    /// Blocks and retries at the ceiling rather than failing.
    pub fn alloc_with_usage(&self, bytes: u64, usage: wgpu::BufferUsages) -> Result<Buf> {
        let size = padded_copy_size(bytes.max(4));
        let key = PoolKey {
            size,
            usage: usage.bits(),
        };
        self.counters.lock().requested += 1;

        if let Some(hit) = self.take_free(key) {
            return Ok(hit);
        }

        let ceiling = *self.ceiling_bytes.lock();
        if self.counters.lock().live_bytes.saturating_add(size) > ceiling {
            // Retire everything in flight, then retry the cache.
            self.counters.lock().cap_retries += 1;
            self.device.poll(wgpu::PollType::wait_indefinitely()).ok();
            self.reclaim();
            if let Some(hit) = self.take_free(key) {
                return Ok(hit);
            }
            let live = self.counters.lock().live_bytes;
            if live.saturating_add(size) > ceiling {
                return Err(Error::Device(format!(
                    "gpu allocation of {size} bytes would exceed the {ceiling}-byte ceiling \
                     with {live} bytes live"
                )));
            }
        }

        Ok(self.create(size, usage))
    }

    /// Upload initial contents through `queue.write_buffer_with`, padding to
    /// `COPY_BUFFER_ALIGNMENT`.
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> Result<Buf> {
        let size = padded_copy_size(data.len() as u64);
        let buf = self.alloc_with_usage(size, usage)?;
        let gpu = buf
            .downcast_ref::<GpuBuffer>()
            .ok_or_else(|| Error::Device("pool handed back a foreign buffer".into()))?;
        match self.queue.write_buffer_with(
            &gpu.buffer,
            0,
            std::num::NonZeroU64::new(size).expect("padded size is nonzero"),
        ) {
            Some(mut view) => {
                // Write straight into the staging belt: the padding tail is
                // whatever the belt held, so it is zeroed explicitly.
                view.slice(..data.len()).copy_from_slice(data);
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

    /// Return a buffer whose only remaining handle is the caller's.
    ///
    /// Reuse is gated on `strong_count == 1`: a buffer still referenced by a
    /// live tensor is dropped from the pool's view rather than handed out
    /// twice.
    ///
    /// # Known risk: the refcount proves nothing about the GPU
    ///
    /// `refcount() == 1` establishes that no *host* handle remains. It does not
    /// establish that the device has finished reading the buffer. A buffer whose
    /// last host handle drops while its submission is still in flight is
    /// recycled here and handed to the next allocation, which then writes into
    /// memory a running kernel is still reading. Under allocation pressure that
    /// shows up as a single wrong value out of an otherwise correct kernel.
    ///
    /// Recording the `SubmissionIndex` at recycle time and withholding the
    /// buffer until `device.poll(WaitForSubmissionIndex(..))` has passed it
    /// would close the window, at the cost of an allocator that deadlocks the
    /// trainer whenever a submission never completes.
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
        // holding a further clone simply fails `take_free`'s `refcount() == 1`
        // test until it drops it. `self.counters` is locked once, outside the
        // `self.free` critical section: nesting the two deadlocks.
        let released = {
            let mut free = self.free.lock();
            let bucket = free.get_or_insert_mut(key, Vec::new);
            if !bucket.iter().any(|b| b.addr() == addr) {
                bucket.push(buf);
            }
            prune_bucket(bucket)
        };
        if released > 0 {
            let mut counters = self.counters.lock();
            counters.live_bytes = counters
                .live_bytes
                .saturating_sub(released.saturating_mul(key.size));
        }
    }

    /// Drop every free buffer whose only handle is the pool's, releasing their
    /// bytes back to the ceiling budget.
    pub fn reclaim(&self) {
        let mut free = self.free.lock();
        let mut released = 0u64;
        let keys: Vec<PoolKey> = free.iter().map(|(k, _)| *k).collect();
        for key in keys {
            if let Some(bucket) = free.get_mut(&key) {
                bucket.retain(|b| {
                    if b.refcount() == 1 {
                        released = released.saturating_add(key.size);
                        false
                    } else {
                        true
                    }
                });
            }
        }
        let mut counters = self.counters.lock();
        counters.live_bytes = counters.live_bytes.saturating_sub(released);
    }

    /// Refill every free buffer with `0xCD` at the end of a resolve, so the
    /// next tenant that assumes zero-initialized storage fails loudly. A
    /// no-op unless poisoning is on.
    pub fn repoison_free_buffers(&self) {
        if !self.poison {
            return;
        }
        let free = self.free.lock();
        for (_, bucket) in free.iter() {
            // The pool tracks in-use buffers too; poisoning one would
            // overwrite a live tensor.
            for buf in bucket.iter().filter(|b| b.refcount() == 1) {
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

    fn take_free(&self, key: PoolKey) -> Option<Buf> {
        let mut free = self.free.lock();
        let bucket = free.get_mut(&key)?;
        // The pool holds its own handle, so `refcount() == 1` is exactly "no
        // caller has this one". Handing back a **clone** leaves the entry
        // tracked, which is what makes a dropped buffer reusable with no
        // `recycle` call at all.
        bucket.iter().find(|b| b.refcount() == 1).cloned()
    }

    fn create(&self, size: u64, usage: wgpu::BufferUsages) -> Buf {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fusor2 pooled buffer"),
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
        // **The pool keeps its own handle to everything it creates.** Without
        // it, a buffer that is never explicitly `recycle`d is destroyed with
        // its last caller handle and re-created from the driver next resolve
        // — and nothing recycles a plan output (`GpuTarget::resolve` skips
        // every value in `binds.buffers`, which `Session::run` fills with
        // every launch root) or an uploaded leaf.
        let key = PoolKey {
            size,
            usage: usage.bits(),
        };
        let released = {
            let mut free = self.free.lock();
            let bucket = free.get_or_insert_mut(key, Vec::new);
            let released = prune_bucket(bucket);
            bucket.push(buf.clone());
            released
        };
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
        if !gpu.usage.contains(wgpu::BufferUsages::COPY_DST) {
            return;
        }
        let chunk = vec![0xCDu8; gpu.size.min(1 << 20) as usize];
        let mut offset = 0u64;
        while offset < gpu.size {
            let len = chunk.len().min((gpu.size - offset) as usize);
            self.queue
                .write_buffer(&gpu.buffer, offset, &chunk[..len]);
            offset += len as u64;
        }
    }
}

/// Drop idle entries past [`FREE_PER_BUCKET`], returning how many were
/// released so the caller can decrement `live_bytes`.
///
/// An entry with an outstanding caller handle (`refcount() > 1`) is **always**
/// kept: the pool's clone is what tracks the buffer, and dropping it would
/// untrack a live allocation and lose the reuse this pool exists for.
fn prune_bucket(bucket: &mut Vec<Buf>) -> u64 {
    let mut idle = 0usize;
    let mut released = 0u64;
    bucket.retain(|b| {
        if b.refcount() > 1 {
            return true;
        }
        idle += 1;
        if idle <= FREE_PER_BUCKET {
            true
        } else {
            released += 1;
            false
        }
    });
    released
}

/// Round up to `wgpu::COPY_BUFFER_ALIGNMENT`, which every
/// `copy_buffer_to_buffer` and `write_buffer_with` requires.
pub fn padded_copy_size(bytes: u64) -> u64 {
    let align = wgpu::COPY_BUFFER_ALIGNMENT;
    bytes.div_ceil(align).max(1) * align
}

/// The platform memory ceiling.
///
/// On Apple silicon, exceeding unified memory panics macOS rather than
/// returning an error, so two thirds of `hw.memsize` is a hard gate. Elsewhere
/// the driver reports allocation failure and the pool does not need to guess.
pub fn default_ceiling() -> u64 {
    #[cfg(target_vendor = "apple")]
    {
        if let Some(total) = hw_memsize() {
            return total / 3 * 2;
        }
        u64::MAX
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        u64::MAX
    }
}

#[cfg(target_vendor = "apple")]
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

#[cfg(test)]
mod tests {
    use super::*;

    // Adapter-gated. These skip cleanly when no GPU is present.

    /// A raw wgpu device at WebGPU baseline limits, independent of capability
    /// probing so a pool test cannot be broken by a capability change.
    fn baseline_device() -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fusor2 pool test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            },
        ))
        .ok()?;
        Some((Arc::new(device), Arc::new(queue)))
    }

    /// Allocate, drop, reallocate the same `(size, usage)` — `created`
    /// increments once, `requested` twice.
    #[test]
    fn pool_reuses_on_strong_count_one() {
        let Some((device, queue)) = baseline_device() else {
            eprintln!("no adapter; skipping pool_reuses_on_strong_count_one");
            return;
        };
        let pool = BufferPool::new(device, queue, &GpuConfig::default());
        let a = pool.alloc(4096, Persistence::Step).unwrap();
        assert_eq!(pool.counters().requested, 1);
        assert_eq!(pool.counters().created, 1);

        pool.recycle(a);
        let b = pool.alloc(4096, Persistence::Step).unwrap();
        assert_eq!(pool.counters().requested, 2, "both requests are counted");
        assert_eq!(
            pool.counters().created,
            1,
            "the second request must be served from the free list"
        );
        drop(b);
    }

    /// A buffer with an outstanding handle is never handed out twice.
    #[test]
    fn a_live_handle_is_never_recycled() {
        let Some((device, queue)) = baseline_device() else {
            eprintln!("no adapter; skipping a_live_handle_is_never_recycled");
            return;
        };
        let pool = BufferPool::new(device, queue, &GpuConfig::default());
        let a = pool.alloc(2048, Persistence::Step).unwrap();
        let alias = a.clone();
        pool.recycle(a);
        let b = pool.alloc(2048, Persistence::Step).unwrap();
        assert_eq!(
            pool.counters().created,
            2,
            "a pinned buffer must not be reissued"
        );
        drop((alias, b));
    }

    /// With the ceiling set just under the working set, the allocator polls
    /// and retries at least once before erroring.
    #[test]
    fn pool_cap_polls_before_failing() {
        let Some((device, queue)) = baseline_device() else {
            eprintln!("no adapter; skipping pool_cap_polls_before_failing");
            return;
        };
        let config = GpuConfig {
            max_gpu_memory_bytes: Some(8192),
            ..GpuConfig::default()
        };
        let pool = BufferPool::new(device, queue, &config);
        let held = pool.alloc(4096, Persistence::Step).unwrap();
        let _second = pool.alloc(4096, Persistence::Step).unwrap();
        assert_eq!(pool.counters().cap_retries, 0, "no retry was needed yet");

        // The third request cannot fit and both live buffers are pinned.
        let err = pool.alloc(4096, Persistence::Step).unwrap_err();
        assert!(
            matches!(&err, Error::Device(m) if m.contains("ceiling")),
            "{err}"
        );
        assert!(
            pool.counters().cap_retries >= 1,
            "the allocator must poll and retry before erroring"
        );
        drop(held);
    }

    #[test]
    fn copy_size_is_aligned_and_never_zero() {
        assert_eq!(padded_copy_size(0), wgpu::COPY_BUFFER_ALIGNMENT);
        assert_eq!(padded_copy_size(1), wgpu::COPY_BUFFER_ALIGNMENT);
        assert_eq!(padded_copy_size(4), 4);
        assert_eq!(padded_copy_size(5), 8);
        for n in 0..64u64 {
            assert_eq!(padded_copy_size(n) % wgpu::COPY_BUFFER_ALIGNMENT, 0);
            assert!(padded_copy_size(n) >= n);
        }
    }

    #[test]
    fn the_ceiling_is_a_real_number_on_apple_and_unbounded_elsewhere() {
        let c = default_ceiling();
        if cfg!(target_vendor = "apple") {
            assert!(c > 0);
            assert!(c < u64::MAX, "apple must report a real ceiling");
        } else {
            assert_eq!(c, u64::MAX);
        }
    }

    #[test]
    fn tensor_and_readback_usages_are_disjoint_in_intent() {
        assert!(TENSOR_USAGE.contains(wgpu::BufferUsages::STORAGE));
        assert!(TENSOR_USAGE.contains(wgpu::BufferUsages::COPY_SRC));
        assert!(TENSOR_USAGE.contains(wgpu::BufferUsages::COPY_DST));
        assert!(!TENSOR_USAGE.contains(wgpu::BufferUsages::MAP_READ));
        assert!(READBACK_USAGE.contains(wgpu::BufferUsages::MAP_READ));
        assert!(!READBACK_USAGE.contains(wgpu::BufferUsages::STORAGE));
    }

    #[test]
    fn pool_key_distinguishes_usage() {
        let a = PoolKey {
            size: 1024,
            usage: TENSOR_USAGE.bits(),
        };
        let b = PoolKey {
            size: 1024,
            usage: READBACK_USAGE.bits(),
        };
        assert_ne!(a, b);
    }

    /// `pool_reuses_on_strong_count_one`'s bookkeeping half, checkable with no
    /// adapter: `Buf` reports the pool's own handle count, so an outstanding
    /// clone makes a buffer unrecyclable and dropping it makes it recyclable
    /// again.
    #[test]
    fn reuse_is_gated_on_the_pool_holding_the_only_handle() {
        let buf = Buf::new(PoolKey {
            size: 4096,
            usage: TENSOR_USAGE.bits(),
        });
        assert_eq!(buf.refcount(), 1, "a fresh handle is recyclable");
        let alias = buf.clone();
        assert_eq!(buf.refcount(), 2, "an outstanding tensor pins the buffer");
        drop(alias);
        assert_eq!(buf.refcount(), 1, "the pool may claim it once more");
    }
}
