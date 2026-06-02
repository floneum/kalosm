use std::{
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use lru::LruCache;
use parking_lot::{Mutex, RwLock};
use rustc_hash::FxBuildHasher;
use wgpu::{BufferUsages, COPY_BUFFER_ALIGNMENT};

const MAX_FREE_BUFFERS_PER_BUCKET: usize = 4;
const BUFFER_ALLOCATION_CACHE_SIZE: usize = 128;

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

/// Per-device buffer pool keyed by `(size, usage)`. Reuses freed buffer
/// storage so common tensor allocations skip the wgpu allocator.
pub struct BufferPool {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffer_allocation_cache:
        RwLock<LruCache<(u64, BufferUsages), Vec<CachedBuffer>, FxBuildHasher>>,
    initialized_buffers_dirty: AtomicBool,
    initialized_buffer_keys: Mutex<Vec<(u64, BufferUsages)>>,
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
        }
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
    ) -> Arc<wgpu::Buffer> {
        if to_initilize {
            self.initialized_buffers_dirty
                .store(true, Ordering::Release);
            self.initialized_buffer_keys.lock().push((size, usage));
        }
        self.get_cached_buffer(size, usage, to_initilize)
            .unwrap_or_else(|| {
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
            })
    }

    /// Get or create a buffer of the specified size.
    pub fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        self.create_buffer_inner(size, usage, false)
    }

    /// Get or create a buffer initialized with the supplied bytes.
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages) -> Arc<wgpu::Buffer> {
        let padded_len = padded_copy_size(data.len() as u64);
        let buffer = self.create_buffer_inner(padded_len, usage, true);
        let mut write = self
            .queue
            .write_buffer_with(&buffer, 0, NonZeroU64::new(padded_len).unwrap())
            .expect("failed to map buffer for writing");
        write[..data.len()].copy_from_slice(data);
        write[data.len()..].fill(0);
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
        let buffer = self.create_buffer_inner(padded_len, usage, true);
        if let Some(len) = NonZeroU64::new(buffer.size()) {
            if let Some(mut write) = self.queue.write_buffer_with(&buffer, 0, len) {
                for byte in write.iter_mut() {
                    *byte = iter.next().unwrap_or(0);
                }
            } else {
                panic!("Failed to map buffer for writing");
            }
        } else {
            panic!("Failed to map buffer for writing");
        }
        buffer
    }
}
