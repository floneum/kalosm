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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tensor_cache_first_append() {
        let device = Device::cpu();
        let mut cache: TensorCache<3, f32> = TensorCache::new(1, 2);

        let data = [1.0f32, 2.0, 3.0, 4.0];
        let tensor: Tensor<3, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 1, 2], &data));

        let result = cache.append(&device, &tensor);

        assert_eq!(result.shape(), [2, 1, 2]);
        assert_eq!(cache.current_seq_len(), 1);

        let output = result.as_slice().await.unwrap();
        assert_eq!(output[[0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 1]], 2.0);
        assert_eq!(output[[1, 0, 0]], 3.0);
        assert_eq!(output[[1, 0, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_tensor_cache_multiple_appends() {
        let device = Device::cpu();
        let mut cache: TensorCache<3, f32> = TensorCache::new(1, 3);

        let data1 = [1.0f32, 2.0];
        let data2 = [3.0f32, 4.0];
        let tensor1: Tensor<3, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2], &data1));
        let tensor2: Tensor<3, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2], &data2));

        cache.append(&device, &tensor1);
        let result = cache.append(&device, &tensor2);

        assert_eq!(result.shape(), [1, 2, 2]);
        assert_eq!(cache.current_seq_len(), 2);

        let output = result.as_slice().await.unwrap();
        assert_eq!(output[[0, 0, 0]], 1.0);
        assert_eq!(output[[0, 0, 1]], 2.0);
        assert_eq!(output[[0, 1, 0]], 3.0);
        assert_eq!(output[[0, 1, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_tensor_cache_reset() {
        let device = Device::cpu();
        let mut cache: TensorCache<3, f32> = TensorCache::new(1, 3);

        let data = [1.0f32, 2.0];
        let tensor: Tensor<3, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2], &data));
        cache.append(&device, &tensor);

        cache.reset();

        assert_eq!(cache.current_seq_len(), 0);
        assert!(cache.current_data().is_none());
    }
}
