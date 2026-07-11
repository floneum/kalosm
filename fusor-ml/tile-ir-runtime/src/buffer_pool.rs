use std::{
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxBuildHasher;
use wgpu::{BufferUsages, COPY_BUFFER_ALIGNMENT};

#[cfg(not(target_arch = "wasm32"))]
const MAX_FREE_BUFFERS_PER_BUCKET: usize = 4;
#[cfg(target_arch = "wasm32")]
const MAX_FREE_BUFFERS_PER_BUCKET: usize = 1;
#[cfg(not(target_arch = "wasm32"))]
const BUFFER_ALLOCATION_CACHE_SIZE: usize = 128;
#[cfg(target_arch = "wasm32")]
const BUFFER_ALLOCATION_CACHE_SIZE: usize = 32;

/// Byte written into freshly handed-out (non-initialized) buffers when a tensor
/// is allocated on a poisoned device (see `Device::with_poisoned_allocations`).
/// Picked to be clearly non-zero in both f32 (`0xCDCDCDCD` ≈ -4.3e8) and integer
/// interpretations so any kernel that reads a region it did not write surfaces
/// an obviously wrong value.
const DIRTY_FILL_BYTE: u8 = 0xCD;

fn padded_copy_size(size: u64) -> u64 {
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;
    ((size + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
}

#[derive(Debug)]
struct CachedBuffer {
    writen: bool,
    buffer: Arc<wgpu::Buffer>,
}

impl CachedBuffer {
    fn new(buffer: Arc<wgpu::Buffer>, writen: bool) -> Self {
        Self { writen, buffer }
    }

    fn initialized(&self) -> bool {
        self.writen
    }

    fn set_initialized(&mut self) {
        self.writen = true;
    }
}

fn prune_cached_buffers(buffers: &mut Vec<CachedBuffer>) {
    let mut kept_free_buffers = 0;
    buffers.retain(|cached| {
        let is_free = Arc::strong_count(&cached.buffer) == 1;
        if !is_free {
            return true;
        }

        if kept_free_buffers < MAX_FREE_BUFFERS_PER_BUCKET {
            kept_free_buffers += 1;
            true
        } else {
            false
        }
    });
}

/// Cumulative allocation statistics for a [`BufferPool`]. `requested` counts
/// every buffer handed out; `created` counts only the ones that missed the
/// pool cache and hit the wgpu allocator. Snapshot before/after a step and
/// diff to measure allocations per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BufferPoolCounters {
    pub requested: u64,
    pub created: u64,
}

/// Per-device buffer pool keyed by `(size, usage)`. Reuses freed buffer
/// storage so common tensor allocations skip the wgpu allocator.
pub struct BufferPool {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffer_allocation_cache:
        RwLock<LruCache<(u64, BufferUsages), Vec<CachedBuffer>, FxBuildHasher>>,
    initialized_buffers_dirty: AtomicBool,
    initialized_buffer_keys: Mutex<Vec<(u64, BufferUsages)>>,
    buffers_requested: AtomicU64,
    buffers_created: AtomicU64,
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool").finish_non_exhaustive()
    }
}

impl BufferPool {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let buffer_allocation_cache = RwLock::new(LruCache::with_hasher(
            const { std::num::NonZeroUsize::new(BUFFER_ALLOCATION_CACHE_SIZE).unwrap() },
            Default::default(),
        ));
        Self {
            device,
            queue,
            buffer_allocation_cache,
            initialized_buffers_dirty: AtomicBool::new(false),
            initialized_buffer_keys: Mutex::new(Vec::new()),
            buffers_requested: AtomicU64::new(0),
            buffers_created: AtomicU64::new(0),
        }
    }

    /// Snapshot the cumulative allocation counters.
    pub fn counters(&self) -> BufferPoolCounters {
        BufferPoolCounters {
            requested: self.buffers_requested.load(Ordering::Relaxed),
            created: self.buffers_created.load(Ordering::Relaxed),
        }
    }

    /// Whether `buffer` is one of the pool's tracked buffers in the
    /// `(size, usage)` bucket — i.e. the pool holds its own strong clone of
    /// it. Liveness accounting (allocation-reuse ledger) uses this to
    /// enumerate the pool as an expected `Arc` holder. Read-only: does not
    /// touch LRU order.
    pub fn is_tracked(&self, size: u64, usage: BufferUsages, buffer: &Arc<wgpu::Buffer>) -> bool {
        let cache = self.buffer_allocation_cache.read();
        cache
            .peek(&(size, usage))
            .is_some_and(|buffers| buffers.iter().any(|c| Arc::ptr_eq(&c.buffer, buffer)))
    }

    /// Reset the initialized flag on all cached buffers.
    pub fn reset_initialized_buffers(&self) {
        if !self.initialized_buffers_dirty.swap(false, Ordering::AcqRel) {
            return;
        }
        let keys = {
            let mut keys = self.initialized_buffer_keys.lock();
            std::mem::take(&mut *keys)
        };
        let mut cache = self.buffer_allocation_cache.write();
        for key in keys {
            if let Some(buffers) = cache.get_mut(&key) {
                for buffer in buffers.iter_mut() {
                    buffer.writen = false;
                }
                prune_cached_buffers(buffers);
            }
        }
    }

    /// Try to get a buffer from the allocation cache. Returns None if no
    /// buffer of the requested size is available.
    pub fn get_cached_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        to_initilize: bool,
    ) -> Option<Arc<wgpu::Buffer>> {
        let mut cache = self.buffer_allocation_cache.write();
        let items = cache.get_mut(&(size, usage))?;
        items.iter_mut().find_map(|a| {
            if Arc::strong_count(&a.buffer) == 1 {
                if !to_initilize && a.initialized() {
                    return None;
                }
                if to_initilize {
                    if a.initialized() {
                        return None;
                    }
                    a.set_initialized();
                }
                Some(a.buffer.clone())
            } else {
                None
            }
        })
    }

    fn create_buffer_inner(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        to_initilize: bool,
        poison: bool,
    ) -> Arc<wgpu::Buffer> {
        if to_initilize {
            self.initialized_buffers_dirty
                .store(true, Ordering::Release);
            self.initialized_buffer_keys.lock().push((size, usage));
        }
        self.buffers_requested.fetch_add(1, Ordering::Relaxed);
        let buffer = self
            .get_cached_buffer(size, usage, to_initilize)
            .unwrap_or_else(|| {
                self.buffers_created.fetch_add(1, Ordering::Relaxed);
                let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Tensor Buffer"),
                    size,
                    usage,
                    mapped_at_creation: false,
                });

                let buffer = Arc::new(new_buffer);
                self.buffer_allocation_cache
                    .write()
                    .get_or_insert_mut((size, usage), Vec::new)
                    .push(CachedBuffer::new(buffer.clone(), to_initilize));
                if let Some(buffers) = self.buffer_allocation_cache.write().get_mut(&(size, usage))
                {
                    prune_cached_buffers(buffers);
                }
                buffer
            });
        // Buffers created with init data are fully overwritten by the caller, so
        // only the to-be-written-by-a-kernel buffers need poisoning to surface
        // zero-initialization assumptions.
        if poison && !to_initilize {
            self.poison_buffer(&buffer, usage);
        }
        buffer
    }

    /// Overwrite a buffer with [`DIRTY_FILL_BYTE`] so a later kernel that reads
    /// an unwritten region sees poison instead of zeros. Only storage buffers
    /// that can be a copy destination are poisoned; readback/staging buffers
    /// are left alone.
    fn poison_buffer(&self, buffer: &wgpu::Buffer, usage: wgpu::BufferUsages) {
        if !usage.contains(BufferUsages::STORAGE) || !usage.contains(BufferUsages::COPY_DST) {
            return;
        }
        if let Some(len) = NonZeroU64::new(buffer.size())
            && let Some(mut write) = self.queue.write_buffer_with(buffer, 0, len)
        {
            write.slice(..).fill(DIRTY_FILL_BYTE);
        }
    }

    /// Get or create a buffer of the specified size. When `poison` is set, the
    /// (kernel-written) buffer is pre-filled with [`DIRTY_FILL_BYTE`] so any
    /// kernel that relies on zero-initialized output storage is surfaced — this
    /// is driven by the allocating device (`Device::with_poisoned_allocations`).
    pub fn create_buffer(
        &self,
        size: u64,
        usage: wgpu::BufferUsages,
        poison: bool,
    ) -> Arc<wgpu::Buffer> {
        self.create_buffer_inner(size, usage, false, poison)
    }

    /// Get or create a buffer initialized with the supplied bytes. Init buffers
    /// are fully overwritten here, so they are never poisoned.
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        let padded_len = padded_copy_size(data.len() as u64);
        let buffer = self.create_buffer_inner(padded_len, usage, true, false);
        let mut write = self
            .queue
            .write_buffer_with(&buffer, 0, NonZeroU64::new(padded_len).unwrap())
            .expect("failed to map buffer for writing");
        write.slice(..data.len()).copy_from_slice(data);
        write.slice(data.len()..).fill(0);
        buffer
    }

    /// Get or create a buffer initialized from a byte iterator.
    pub fn create_buffer_init_iter(
        &self,
        data: impl IntoIterator<Item = u8>,
        usage: wgpu::BufferUsages,
        len: u64,
    ) -> Arc<wgpu::Buffer> {
        let mut iter = data.into_iter();
        let padded_len = padded_copy_size(len);
        let buffer = self.create_buffer_inner(padded_len, usage, true, false);
        if let Some(len) = NonZeroU64::new(buffer.size()) {
            if let Some(mut write) = self.queue.write_buffer_with(&buffer, 0, len) {
                let write_len = write.len();
                write
                    .slice(..)
                    .write_iter((0..write_len).map(|_| iter.next().unwrap_or(0)));
            } else {
                panic!("Failed to map buffer for writing");
            }
        } else {
            panic!("Failed to map buffer for writing");
        }
        buffer
    }
}
