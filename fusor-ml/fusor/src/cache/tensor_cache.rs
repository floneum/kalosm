//! Growable tensor cache implementation.

use std::ops::Range;

use crate::gpu::{DataType, StrideSpec};
use crate::{Device, SimdElement, Tensor, cat};

const GPU_CACHE_MIN_ALLOC_SEQ_LEN: usize = 256;

/// A growable tensor cache.
/// This cache manages tensor data with exponentially larger allocations as the sequence length increases.
///
/// On GPU the cache owns a backing buffer and writes newly appended data into
/// it in place. Lazy writes are chained through the latest full backing tensor,
/// so resolving only the newest cache view still observes any earlier
/// unresolved appends. Growth (a rare power-of-two reallocation) stays lazy too,
/// so neither appends nor grows trigger a separate GPU submission — everything
/// resolves in-band with the forward pass.
#[derive(Clone)]
pub struct TensorCache<const R: usize, D: SimdElement> {
    /// The latest valid `[0..current_seq_len]` view of the cache. This is what
    /// `current_data` and `append` hand back; on GPU it carries the dependency
    /// on the most recent in-place write so readers observe the appended data.
    all_data: Option<Tensor<R, D>>,
    /// GPU only: the latest full tensor for the allocated backing buffer. It
    /// aliases the same allocation across appends; before resolution it may be
    /// an in-place write node so later appends preserve pending writes. `None`
    /// on CPU, where tensors are eager and need no backing.
    backing: Option<Tensor<R, D>>,
    current_seq_len: usize,
    allocated_seq_len: usize,
    concat_dim: usize,
    max_sequence_len: usize,
}

impl<const R: usize, D: SimdElement + DataType + Default> TensorCache<R, D>
where
    crate::AddOp: crate::cpu::SimdBinaryOp<D>,
    D: Copy,
{
    /// Create a new cache with the given concatenation dimension
    pub fn new(concat_dim: usize, max_sequence_len: usize) -> Self {
        assert!(concat_dim < R, "concat_dim must be less than tensor rank R");
        Self {
            all_data: None,
            backing: None,
            current_seq_len: 0,
            allocated_seq_len: 0,
            concat_dim,
            max_sequence_len,
        }
    }

    /// Get the current data in the cache
    pub fn current_data(&self) -> Option<&Tensor<R, D>> {
        self.all_data.as_ref()
    }

    /// Reset the cache
    pub fn reset(&mut self) {
        self.all_data = None;
        self.backing = None;
        self.current_seq_len = 0;
        self.allocated_seq_len = 0;
    }

    /// Append a new value to the cache
    ///
    /// Returns the full cached tensor including the newly appended data
    pub fn append(&mut self, device: &Device, v: &Tensor<R, D>) -> Tensor<R, D> {
        if v.is_gpu() {
            self.append_gpu(device, v)
        } else {
            self.append_cpu(device, v)
        }
    }

    /// GPU append: write `v` into the concrete backing buffer in place and
    /// return a view of the valid region. Only grows (and materializes) the
    /// backing buffer when it would otherwise overflow.
    fn append_gpu(&mut self, device: &Device, v: &Tensor<R, D>) -> Tensor<R, D> {
        let v_shape = v.shape();
        let seq_len = v_shape[self.concat_dim];
        let required_seq_len = self.current_seq_len + seq_len;

        // Sliding-window overflow: drop the oldest tokens and keep the last
        // `max_sequence_len` of `[existing .. v]`.
        if required_seq_len > self.max_sequence_len {
            return self.append_gpu_evict(v, &v_shape);
        }

        self.ensure_capacity_gpu(device, required_seq_len, &v_shape);

        let v_gpu = v.as_gpu().expect("append_gpu requires a GPU tensor");
        let backing = self
            .backing
            .as_ref()
            .and_then(|b| b.as_gpu())
            .expect("gpu backing present after ensure_capacity");
        let slice: [Range<usize>; R] = std::array::from_fn(|i| {
            if i == self.concat_dim {
                self.current_seq_len..required_seq_len
            } else {
                0..v_shape[i]
            }
        });
        let written = backing.slice_assign_in_place(slice, v_gpu);
        self.current_seq_len = required_seq_len;

        self.backing = Some(Tensor::Gpu(written.clone()));
        let view = Self::gpu_view(&written, self.concat_dim, 0, self.current_seq_len);
        self.all_data = Some(Tensor::Gpu(view.clone()));
        Tensor::Gpu(view)
    }

    /// Ensure the GPU backing buffer can hold `required_seq_len` tokens,
    /// (re)allocating a power-of-two buffer that preserves existing data. The
    /// new buffer is built from the latest backing tensor (which may include
    /// unresolved in-place writes) with `cat`, and left lazy: it is resolved
    /// in-band by the forward pass's own resolve, so a grow does not trigger a
    /// separate mid-forward GPU submission.
    fn ensure_capacity_gpu(
        &mut self,
        device: &Device,
        required_seq_len: usize,
        v_shape: &[usize; R],
    ) {
        if self.backing.is_some() && required_seq_len <= self.allocated_seq_len {
            return;
        }

        let new_allocated = gpu_allocation_seq_len(required_seq_len, self.max_sequence_len);

        let padded = if self.current_seq_len > 0
            && let Some(old) = self.backing.as_ref()
        {
            let valid = old
                .narrow(self.concat_dim, 0, self.current_seq_len)
                .to_concrete();
            let pad_shape: [usize; R] = std::array::from_fn(|i| {
                if i == self.concat_dim {
                    new_allocated - self.current_seq_len
                } else {
                    v_shape[i]
                }
            });
            let zeros = Tensor::zeros(device, pad_shape);
            cat([valid, zeros], self.concat_dim)
        } else {
            // First allocation: build a real contiguous backing buffer. `zeros`
            // (a stride-0 splat) cannot be used as an in-place target directly.
            // The valid slice is overwritten before it is exposed, so this does
            // not need a CPU-filled zero upload.
            let shape: [usize; R] = std::array::from_fn(|i| {
                if i == self.concat_dim {
                    new_allocated
                } else {
                    v_shape[i]
                }
            });
            let Device::Gpu(gpu_device) = device else {
                unreachable!("append_gpu only runs on GPU tensors");
            };
            Tensor::Gpu(crate::gpu::Tensor::uninit(gpu_device, shape))
        };

        self.backing = Some(padded);
        self.allocated_seq_len = new_allocated;
    }

    /// GPU sliding-window append: build a fresh backing holding the last
    /// `max_sequence_len` tokens of `[existing .. v]`. Mirrors the CPU overflow
    /// path and stays lazy, so the eviction resolves in-band with the forward
    /// pass rather than as a separate mid-forward submission.
    fn append_gpu_evict(&mut self, v: &Tensor<R, D>, v_shape: &[usize; R]) -> Tensor<R, D> {
        let max = self.max_sequence_len;
        let seq_len = v_shape[self.concat_dim];
        let required = self.current_seq_len + seq_len;
        let new_start = required - max;

        let mut tensors = Vec::new();
        if let Some(old) = self.backing.as_ref()
            && self.current_seq_len > new_start
        {
            tensors.push(
                old.narrow(self.concat_dim, new_start, self.current_seq_len - new_start)
                    .to_concrete(),
            );
        }
        tensors.push(v.clone());
        let combined = cat(tensors, self.concat_dim);
        let combined_len = combined.shape()[self.concat_dim];
        let backing = combined
            .narrow(self.concat_dim, combined_len - max, max)
            .to_concrete();

        self.allocated_seq_len = max;
        self.current_seq_len = max;
        self.all_data = Some(backing.clone());
        self.backing = Some(backing.clone());
        backing
    }

    /// Build a `[start..start + len]` view of `t` along `concat_dim` without
    /// copying. The result aliases `t`'s buffer (and carries `t`'s graph
    /// dependency).
    fn gpu_view(
        t: &crate::gpu::Tensor<R, D>,
        concat_dim: usize,
        start: usize,
        len: usize,
    ) -> crate::gpu::Tensor<R, D> {
        let shape = t.shape();
        let specs: [StrideSpec; R] = std::array::from_fn(|i| {
            if i == concat_dim {
                StrideSpec::dim(i, len).with_offset(start)
            } else {
                StrideSpec::dim(i, shape[i])
            }
        });
        t.restride(specs)
    }

    /// CPU append: eager concatenation/assignment. CPU tensors are not lazy, so
    /// there is no compute graph to bound and nothing to materialize by hand.
    fn append_cpu(&mut self, device: &Device, v: &Tensor<R, D>) -> Tensor<R, D> {
        let v_shape = v.shape();
        let seq_len = v_shape[self.concat_dim];
        // First find the required new sequence length
        let required_seq_len = self.current_seq_len + seq_len;

        // If the required size is larger than the max sequence length, cut the start of the cache.
        if required_seq_len > self.max_sequence_len {
            let max_seq_len = self.max_sequence_len;
            let new_start = required_seq_len - max_seq_len;
            let mut tensors = Vec::new();
            // Cut the start of the cache.
            if let Some(all_data) = self.all_data.as_ref() {
                tensors.push(
                    all_data
                        .narrow(self.concat_dim, new_start, self.current_seq_len - new_start)
                        .to_concrete(),
                );
            }
            tensors.push(v.clone());
            let all_data = cat(tensors, self.concat_dim);
            let all_data_len = all_data.shape()[self.concat_dim];
            self.all_data = Some(
                all_data
                    .narrow(self.concat_dim, all_data_len - max_seq_len, max_seq_len)
                    .to_concrete(),
            );
            self.current_seq_len = max_seq_len;
            self.allocated_seq_len = max_seq_len;
            return self.all_data.clone().unwrap();
        }

        if let Some(cached) = &mut self.all_data {
            // Check if we need to grow the allocation
            if required_seq_len > self.allocated_seq_len {
                // Double the allocation until it's large enough
                let new_allocated_seq_len = required_seq_len.next_power_of_two();
                self.allocated_seq_len = new_allocated_seq_len;
                let new_data_shape: [usize; R] = std::array::from_fn(|i| {
                    if i == self.concat_dim {
                        new_allocated_seq_len - self.current_seq_len
                    } else {
                        v_shape[i]
                    }
                });
                // Allocate new tensor with larger size
                let new_data = Tensor::zeros(device, new_data_shape);
                *cached = cat([cached.clone(), new_data], self.concat_dim);
            }
            // Assign the new data into the cached tensor
            let slice: [std::ops::Range<usize>; R] = std::array::from_fn(|i| {
                if i == self.concat_dim {
                    self.current_seq_len..required_seq_len
                } else {
                    0..v_shape[i]
                }
            });
            *cached = cached.slice_assign(slice, v);
            self.current_seq_len = required_seq_len;
            // Return only the valid portion of the cache, not the full allocated tensor
            cached
                .narrow(self.concat_dim, 0, self.current_seq_len)
                .to_concrete()
        } else {
            // First append - just store it
            self.all_data = Some(v.clone());
            self.current_seq_len = seq_len;
            self.allocated_seq_len = seq_len;
            v.clone()
        }
    }

    /// Get the current sequence length
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }
}

fn gpu_allocation_seq_len(required_seq_len: usize, max_sequence_len: usize) -> usize {
    debug_assert!(required_seq_len <= max_sequence_len);

    let min_alloc = GPU_CACHE_MIN_ALLOC_SEQ_LEN.min(max_sequence_len);
    required_seq_len
        .next_power_of_two()
        .max(min_alloc)
        .min(max_sequence_len)
}

#[cfg(test)]
mod tests {
    use super::{GPU_CACHE_MIN_ALLOC_SEQ_LEN, gpu_allocation_seq_len};

    #[test]
    fn gpu_allocation_skips_small_decode_growth_cliffs() {
        assert_eq!(GPU_CACHE_MIN_ALLOC_SEQ_LEN, 256);
        assert_eq!(gpu_allocation_seq_len(1, 4096), 256);
        assert_eq!(gpu_allocation_seq_len(64, 4096), 256);
        assert_eq!(gpu_allocation_seq_len(65, 4096), 256);
        assert_eq!(gpu_allocation_seq_len(256, 4096), 256);
        assert_eq!(gpu_allocation_seq_len(257, 4096), 512);
    }

    #[test]
    fn gpu_allocation_respects_short_contexts() {
        assert_eq!(gpu_allocation_seq_len(1, 96), 96);
        assert_eq!(gpu_allocation_seq_len(65, 96), 96);
        assert_eq!(gpu_allocation_seq_len(96, 96), 96);
    }
}
