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

    /// Reserve enough key/value sequence storage to avoid growth during future appends.
    pub fn reserve(
        &mut self,
        device: &Device,
        target_seq_len: usize,
        keys: &mut Vec<crate::NodeIndex>,
    ) {
        if let Some(key) = self.key.reserve(device, target_seq_len) {
            keys.push(key);
        }
        if let Some(key) = self.value.reserve(device, target_seq_len) {
            keys.push(key);
        }
    }

    /// Get the current sequence length
    pub fn current_seq_len(&self) -> usize {
        self.key.current_seq_len()
    }
}
