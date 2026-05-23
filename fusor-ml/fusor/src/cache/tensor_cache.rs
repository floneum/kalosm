//! Growable tensor cache implementation.

use crate::{Device, SimdElement, Tensor, cat};
use fusor_core::DataType;

/// A growable tensor cache.
/// This cache manages tensor data with exponentially larger allocations as the sequence length increases.
#[derive(Clone)]
pub struct TensorCache<const R: usize, D: SimdElement> {
    all_data: Option<Tensor<R, D>>,
    current_seq_len: usize,
    allocated_seq_len: usize,
    concat_dim: usize,
    max_sequence_len: usize,
}

impl<const R: usize, D: SimdElement + DataType + Default> TensorCache<R, D>
where
    crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
    D: Copy,
{
    /// Create a new cache with the given concatenation dimension
    pub fn new(concat_dim: usize, max_sequence_len: usize) -> Self {
        assert!(concat_dim < R, "concat_dim must be less than tensor rank R");
        Self {
            all_data: None,
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
        self.current_seq_len = 0;
        self.allocated_seq_len = 0;
    }

    /// Append a new value to the cache
    ///
    /// Returns the full cached tensor including the newly appended data
    pub fn append(&mut self, device: &Device, v: &Tensor<R, D>) -> Tensor<R, D> {
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
            *cached = match (&*cached, v) {
                (Tensor::Gpu(cached), Tensor::Gpu(v)) => {
                    Tensor::Gpu(cached.slice_assign_in_place(slice, v))
                }
                _ => cached.slice_assign(slice, v),
            };
            self.current_seq_len = required_seq_len;
            // Return only the valid portion of the cache, not the full allocated tensor
            match &*cached {
                Tensor::Gpu(cached) => {
                    let specs: [fusor_core::StrideSpec; R] = std::array::from_fn(|i| {
                        let len = if i == self.concat_dim {
                            self.current_seq_len
                        } else {
                            cached.shape()[i]
                        };
                        fusor_core::StrideSpec::dim(i, len)
                    });
                    Tensor::Gpu(cached.restride(specs))
                }
                _ => cached
                    .narrow(self.concat_dim, 0, self.current_seq_len)
                    .to_concrete(),
            }
        } else {
            // First append - just store it
            self.all_data = Some(v.clone());
            self.current_seq_len = seq_len;
            self.allocated_seq_len = seq_len;
            v.clone()
        }
    }

    /// Reserve enough sequence storage to avoid growth during future appends.
    pub fn reserve(&mut self, device: &Device, target_seq_len: usize) -> Option<crate::NodeIndex> {
        let target_seq_len = target_seq_len.min(self.max_sequence_len);
        if target_seq_len <= self.allocated_seq_len {
            return None;
        }

        let Some(cached) = &mut self.all_data else {
            return None;
        };

        let new_allocated_seq_len = target_seq_len
            .next_power_of_two()
            .min(self.max_sequence_len);
        if new_allocated_seq_len <= self.allocated_seq_len {
            return None;
        }

        let cached_shape = cached.shape();
        let new_data_shape: [usize; R] = std::array::from_fn(|i| {
            if i == self.concat_dim {
                new_allocated_seq_len - self.allocated_seq_len
            } else {
                cached_shape[i]
            }
        });
        let new_data = Tensor::zeros(device, new_data_shape);
        *cached = cat([cached.clone(), new_data], self.concat_dim);
        self.allocated_seq_len = new_allocated_seq_len;
        cached.gpu_key()
    }

    /// Add this cache tensor's GPU node to a batch of already-resolved nodes
    /// that should be rebased to graph leaves.
    pub fn detach_key(&self, keys: &mut Vec<crate::NodeIndex>) {
        if let Some(cached) = &self.all_data
            && let Some(key) = cached.gpu_key()
        {
            keys.push(key);
        }
    }

    /// Get the current sequence length
    pub fn current_seq_len(&self) -> usize {
        self.current_seq_len
    }

}
