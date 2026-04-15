//! Mask cache implementation for efficient attention mask management.

use crate::{ConcreteTensor, Device, SimdElement, Tensor};
use fusor_core::FloatDataType;

use super::AttentionMask;

/// Mask cache for efficiently managing attention masks
#[derive(Clone)]
pub struct MaskCache<D: SimdElement> {
    #[allow(clippy::type_complexity)]
    masks: std::sync::Arc<
        std::sync::RwLock<std::collections::HashMap<(usize, Option<usize>), AttentionMask<D>>>,
    >,
}

impl<D: SimdElement> Default for MaskCache<D> {
    fn default() -> Self {
        Self {
            masks: Default::default(),
        }
    }
}

impl<D: SimdElement + FloatDataType + Default> MaskCache<D>
where
    crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
{
    /// Get or create a causal mask for the given sequence length
    ///
    /// # Arguments
    /// * `seq_len` - The sequence length
    /// * `seqlen_offset` - The offset for the sequence (used for padding when not the first token)
    /// * `sliding_window_size` - Optional sliding window size for sliding window attention
    /// * `device` - The device to create the mask on
    pub fn get_mask(
        &self,
        seq_len: usize,
        seqlen_offset: usize,
        sliding_window_size: Option<usize>,
        device: &Device,
    ) -> AttentionMask<D> {
        let (seq_len, seqlen_offset) = if let Some(sliding_window_size) = sliding_window_size {
            // offset + seqlen_offset should not exceed sliding_window_size
            let offset = seqlen_offset.min(sliding_window_size.saturating_sub(seq_len));
            (seq_len, offset)
        } else {
            (seq_len, seqlen_offset)
        };

        // Get or create the base mask
        let mask = {
            let masks = self.masks.read().unwrap();
            masks.get(&(seq_len, sliding_window_size)).cloned()
        };

        let mask = if let Some(mask) = mask {
            mask
        } else {
            // Create the mask based on whether we have a sliding window
            let mask = if let Some(sliding_window_size) = sliding_window_size {
                Self::create_sliding_window_mask(device, seq_len, sliding_window_size)
            } else {
                AttentionMask::causal(device, seq_len)
            };

            let mut masks = self.masks.write().unwrap();
            masks.insert((seq_len, sliding_window_size), mask.clone());
            mask
        };

        // If we have an offset, we need to pad the mask
        if seqlen_offset > 0 {
            // Pad the mask on the left with zeros
            let mask_tensor = mask.mask();
            let [rows, cols] = mask_tensor.shape();
            let zeros: Tensor<2, D> = Tensor::zeros(device, [rows, seqlen_offset + cols]);
            let padded_mask = zeros.slice_assign(
                [0..rows, seqlen_offset..(seqlen_offset + cols)],
                mask_tensor,
            );
            AttentionMask::new(padded_mask)
        } else {
            mask
        }
    }

    fn create_sliding_window_mask(
        device: &Device,
        seq_len: usize,
        sliding_window_size: usize,
    ) -> AttentionMask<D> {
        // Create a mask that prevents attending to:
        // 1. Future positions (i < j)
        // 2. Positions outside the sliding window (j + sliding_window_size <= i)
        let mut mask_data = vec![D::zero(); seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if i < j || j + sliding_window_size <= i {
                    mask_data[i * seq_len + j] = D::from_f32(f32::NEG_INFINITY);
                }
            }
        }

        let mask: Tensor<2, D, ConcreteTensor<D, 2>> = match device {
            Device::Cpu => Tensor::Cpu(fusor_cpu::Tensor::from_slice(
                [seq_len, seq_len],
                &mask_data,
            )),
            Device::Gpu(gpu) => {
                let data_chunks: Vec<&[D]> = mask_data.chunks(seq_len).collect();
                Tensor::Gpu(fusor_core::Tensor::new(gpu, data_chunks))
            }
        };
        AttentionMask::new(mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mask_cache() {
        let device = Device::cpu();
        let cache: MaskCache<f32> = MaskCache::default();

        let mask1 = cache.get_mask(3, 0, None, &device);
        let mask2 = cache.get_mask(3, 0, None, &device);

        // Should be cached (same shape)
        assert_eq!(mask1.mask().shape(), [3, 3]);
        assert_eq!(mask2.mask().shape(), [3, 3]);

        let mask3 = cache.get_mask(5, 0, None, &device);
        assert_eq!(mask3.mask().shape(), [5, 5]);
    }

    #[tokio::test]
    async fn test_mask_cache_with_offset() {
        let device = Device::cpu();
        let cache: MaskCache<f32> = MaskCache::default();

        // Test with seqlen_offset
        let mask = cache.get_mask(2, 3, None, &device);
        // Mask should be padded: [2, 3+2] = [2, 5]
        assert_eq!(mask.mask().shape(), [2, 5]);
    }

    #[tokio::test]
    async fn test_mask_cache_sliding_window() {
        let device = Device::cpu();
        let cache: MaskCache<f32> = MaskCache::default();

        let mask = cache.get_mask(4, 0, Some(2), &device);
        assert_eq!(mask.mask().shape(), [4, 4]);

        let mask_data = mask.mask().clone().as_slice().await.unwrap();

        // Should match the sliding window pattern:
        // Row 0: [0, -inf, -inf, -inf] (can only attend to self)
        // Row 1: [0, 0, -inf, -inf] (can attend to 0,1)
        // Row 2: [-inf, 0, 0, -inf] (can attend to 1,2 - sliding window of 2)
        // Row 3: [-inf, -inf, 0, 0] (can attend to 2,3 - sliding window of 2)
        assert_eq!(mask_data[[0, 0]], 0.0);
        assert_eq!(mask_data[[0, 1]], f32::NEG_INFINITY);
        assert_eq!(mask_data[[1, 0]], 0.0);
        assert_eq!(mask_data[[1, 1]], 0.0);
        assert_eq!(mask_data[[2, 0]], f32::NEG_INFINITY);
        assert_eq!(mask_data[[2, 1]], 0.0);
        assert_eq!(mask_data[[2, 2]], 0.0);
        assert_eq!(mask_data[[3, 1]], f32::NEG_INFINITY);
        assert_eq!(mask_data[[3, 2]], 0.0);
        assert_eq!(mask_data[[3, 3]], 0.0);
    }
}
