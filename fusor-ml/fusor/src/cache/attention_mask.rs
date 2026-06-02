//! Attention mask implementation.

use crate::gpu::FloatDataType;
use crate::{ConcreteTensor, Device, SimdElement, Tensor};

/// Attention mask for causal (decoder) attention
///
/// Prevents attending to future positions
#[derive(Clone)]
pub struct AttentionMask<D: SimdElement> {
    mask: Tensor<2, D, ConcreteTensor<D, 2>>,
    /// `true` when the mask is exactly the strict lower-triangular causal
    /// mask of shape `[n, n]`. Hint to GPU flash attention so it can skip the
    /// mask tensor entirely and prune upper-triangle work.
    is_strict_causal: bool,
}

impl<D: SimdElement + FloatDataType + Default> AttentionMask<D>
where
    crate::AddOp: crate::cpu::SimdBinaryOp<D>,
{
    /// Create a new attention mask
    pub fn new(mask: Tensor<2, D, ConcreteTensor<D, 2>>) -> Self {
        Self {
            mask,
            is_strict_causal: false,
        }
    }

    /// Returns true if this is a strict lower-triangular causal mask. The
    /// GPU flash attention kernel can then skip masking work entirely.
    pub fn is_strict_causal(&self) -> bool {
        self.is_strict_causal
    }

    /// Marks this mask as a strict lower-triangular causal mask.
    pub fn mark_strict_causal(mut self) -> Self {
        self.is_strict_causal = true;
        self
    }

    /// Create a causal mask for the given sequence length
    ///
    /// mask[i, j] = -inf if j > i (can't attend to future), 0 otherwise
    pub fn causal(device: &Device, seq_len: usize) -> Self {
        // Create a lower triangular matrix of 0s and upper triangular of -inf
        let mut mask_data = vec![D::zero(); seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = D::from_f32(f32::NEG_INFINITY);
            }
        }

        let mask: Tensor<2, D> = match device {
            Device::Cpu => Tensor::Cpu(crate::cpu::TypedTensor::from_slice(
                [seq_len, seq_len],
                &mask_data,
            )),
            Device::Gpu(gpu) => {
                let data_chunks: Vec<&[D]> = mask_data.chunks(seq_len).collect();
                Tensor::Gpu(crate::gpu::Tensor::new(gpu, data_chunks))
            }
        };
        Self::new(mask).mark_strict_causal()
    }

    /// Apply the mask to attention scores
    ///
    /// attention_scores: (batch, heads, seq_len, seq_len) or similar ranks
    /// Returns: masked attention scores
    ///
    /// The mask will be broadcast to match the attention scores shape
    pub fn apply<'a, const R: usize>(
        &'a self,
        attention_scores: &'a Tensor<R, D>,
    ) -> Tensor<R, D, crate::cpu::Add<D, R, ConcreteTensor<D, R>, &'a ConcreteTensor<D, R>>>
    where
        D: std::ops::Add<Output = D>,
        (crate::gpu::Tensor<2, D>, crate::gpu::Tensor<R, D>): crate::gpu::MaxRank<R, D>,
    {
        // Broadcast the mask to match the attention scores shape
        let mask_broadcast: Tensor<R, D, _> = self.mask.broadcast_as(attention_scores.shape());
        match (mask_broadcast, attention_scores) {
            (Tensor::Cpu(m), Tensor::Cpu(a)) => Tensor::Cpu(m.to_concrete() + a),
            (Tensor::Gpu(m), Tensor::Gpu(a)) => Tensor::Gpu(m + a),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        }
    }

    pub fn forward<const R: usize>(&self, attention_scores: &mut Tensor<R, D>)
    where
        D: std::ops::Add<Output = D>,
        (crate::gpu::Tensor<2, D>, crate::gpu::Tensor<R, D>): crate::gpu::MaxRank<R, D>,
    {
        *attention_scores = self.apply(attention_scores).to_concrete();
    }

    pub fn mask(&self) -> &Tensor<2, D, ConcreteTensor<D, 2>> {
        &self.mask
    }
}
