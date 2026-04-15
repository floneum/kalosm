//! KV cache implementation for attention layers.

use crate::{ConcreteTensor, Device, SimdElement, Tensor};
use fusor_core::DataType;

use super::TensorCache;

/// A growable KV cache for attention layers
///
/// Manages key and value caches separately, growing them as needed
#[derive(Clone)]
pub struct KvCache<D: SimdElement> {
    key: TensorCache<4, D>,
    value: TensorCache<4, D>,
}

impl<D: SimdElement + DataType + Default> KvCache<D>
where
    crate::AddOp: fusor_cpu::SimdBinaryOp<D>,
{
    /// Create a new KV cache
    ///
    /// concat_dim: The dimension along which to concatenate new key/value tensors (typically 1 for sequence length)
    pub fn new(concat_dim: usize, max_sequence_len: usize) -> Self {
        Self {
            key: TensorCache::new(concat_dim, max_sequence_len),
            value: TensorCache::new(concat_dim, max_sequence_len),
        }
    }

    /// Get the current key data in the cache
    pub fn k(&self) -> Option<&Tensor<4, D, ConcreteTensor<D, 4>>> {
        self.key.current_data()
    }

    /// Get the current value data in the cache
    pub fn v(&self) -> Option<&Tensor<4, D, ConcreteTensor<D, 4>>> {
        self.value.current_data()
    }

    /// Reset the cache
    pub fn reset(&mut self) {
        self.key.reset();
        self.value.reset();
    }

    /// Append a new key/value pair to the cache
    ///
    /// Returns (full_keys, full_values) including the newly appended data
    pub fn append(
        &mut self,
        device: &Device,
        k: &Tensor<4, D, ConcreteTensor<D, 4>>,
        v: &Tensor<4, D, ConcreteTensor<D, 4>>,
    ) -> (
        Tensor<4, D, ConcreteTensor<D, 4>>,
        Tensor<4, D, ConcreteTensor<D, 4>>,
    ) {
        let keys = self.key.append(device, k);
        let values = self.value.append(device, v);
        (keys, values)
    }

    /// Get the current sequence length
    pub fn current_seq_len(&self) -> usize {
        self.key.current_seq_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_kv_cache_first_append() {
        let device = Device::cpu();
        let mut cache: KvCache<f32> = KvCache::new(1, 2);

        let key_data = [1.0f32, 2.0];
        let value_data = [3.0f32, 4.0];
        let key: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 1, 2], &key_data));
        let value: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 1, 2], &value_data));

        let (k_result, v_result) = cache.append(&device, &key, &value);

        assert_eq!(k_result.shape(), [1, 1, 1, 2]);
        assert_eq!(v_result.shape(), [1, 1, 1, 2]);

        let k_output = k_result.as_slice().await.unwrap();
        let v_output = v_result.as_slice().await.unwrap();

        assert_eq!(k_output[[0, 0, 0, 0]], 1.0);
        assert_eq!(k_output[[0, 0, 0, 1]], 2.0);
        assert_eq!(v_output[[0, 0, 0, 0]], 3.0);
        assert_eq!(v_output[[0, 0, 0, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_kv_cache_multiple_appends() {
        let device = Device::cpu();
        let mut cache: KvCache<f32> = KvCache::new(1, 3);

        let key_data1 = [1.0f32, 2.0];
        let value_data1 = [3.0f32, 4.0];
        let key1: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 1, 2], &key_data1));
        let value1: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 1, 2], &value_data1));

        cache.append(&device, &key1, &value1);

        let key_data2 = [5.0f32, 6.0];
        let value_data2 = [7.0f32, 8.0];
        let key2: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 1, 2], &key_data2));
        let value2: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 1, 2], &value_data2));

        let (k_result, v_result) = cache.append(&device, &key2, &value2);

        assert_eq!(k_result.shape(), [1, 2, 1, 2]);
        assert_eq!(v_result.shape(), [1, 2, 1, 2]);

        let k_output = k_result.as_slice().await.unwrap();
        let v_output = v_result.as_slice().await.unwrap();

        assert_eq!(k_output[[0, 0, 0, 0]], 1.0);
        assert_eq!(k_output[[0, 0, 0, 1]], 2.0);
        assert_eq!(k_output[[0, 1, 0, 0]], 5.0);
        assert_eq!(k_output[[0, 1, 0, 1]], 6.0);
        assert_eq!(v_output[[0, 0, 0, 0]], 3.0);
        assert_eq!(v_output[[0, 0, 0, 1]], 4.0);
        assert_eq!(v_output[[0, 1, 0, 0]], 7.0);
        assert_eq!(v_output[[0, 1, 0, 1]], 8.0);
    }

    #[tokio::test]
    async fn test_kv_cache_reset() {
        let device = Device::cpu();
        let mut cache: KvCache<f32> = KvCache::new(1, 3);

        let key_data = [1.0f32, 2.0];
        let value_data = [3.0f32, 4.0];
        let key: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 1, 2], &key_data));
        let value: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 1, 2], &value_data));

        cache.append(&device, &key, &value);

        cache.reset();

        assert!(cache.k().is_none());
        assert!(cache.v().is_none());
    }
}
