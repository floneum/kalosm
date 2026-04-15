//! Flash attention operations that work on both CPU and GPU backends.

use crate::{
    AddOp, ConcreteTensor, DivOp, ExpOp, FloatOps, MulOp, SimdBinaryOp, SimdElement, SimdUnaryOp,
    SubOp, Tensor,
};
use fusor_core::{DataType, FloatDataType};
use fusor_cpu::{MatmulImpl, MaxOp, SimdReduceOp, SumOp};

/// Describes how to interpret a 2D attention mask.
#[derive(Debug, Clone, Copy)]
pub enum MaskKind {
    /// Mask is [q_seq_len, kv_seq_len] — applied identically to every (batch, head) pair.
    /// Used for causal masks in decoder models.
    QKMask,
    /// Mask is [batch, kv_seq_len] — per-token validity mask broadcast across heads and queries.
    /// Used for padding masks in encoder/embedding models.
    BatchKeyMask,
}

impl<D> Tensor<4, D, ConcreteTensor<D, 4>>
where
    D: SimdElement
        + DataType
        + FloatDataType
        + FloatOps
        + Default
        + MatmulImpl
        + std::ops::Add<Output = D>
        + std::ops::Sub<Output = D>
        + std::ops::Mul<Output = D>
        + std::ops::Div<Output = D>,
    AddOp: SimdBinaryOp<D>,
    SubOp: SimdBinaryOp<D>,
    MulOp: SimdBinaryOp<D>,
    DivOp: SimdBinaryOp<D>,
    MaxOp: SimdReduceOp<D>,
    SumOp: SimdReduceOp<D>,
    ExpOp: SimdUnaryOp<D>,
{
    /// Computes flash attention with optional masking.
    ///
    /// Supports grouped-query attention (GQA) and multi-query attention (MQA) where
    /// K and V may have fewer heads than Q. The number of Q heads must be divisible
    /// by the number of K/V heads.
    ///
    /// Args:
    ///   - k: Key tensor with shape [batch, num_kv_heads, kv_seq_len, head_dim]
    ///   - v: Value tensor with shape [batch, num_kv_heads, kv_seq_len, head_dim]
    ///   - scale: Scale factor (typically 1/sqrt(head_dim))
    ///   - mask: Optional attention mask with a [`MaskKind`] describing its layout
    pub fn flash_attention(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<(&Tensor<2, D, ConcreteTensor<D, 2>>, MaskKind)>,
    ) -> Self {
        match (self, k, v) {
            // GPU path - use the optimized fused kernel (QKMask only)
            (Tensor::Gpu(q), Tensor::Gpu(k), Tensor::Gpu(v))
                if !matches!(mask, Some((_, MaskKind::BatchKeyMask))) =>
            {
                let gpu_mask = mask.map(|(m, _kind)| match m {
                    Tensor::Gpu(mask) => mask,
                    _ => panic!("Mask must be on the same device as other tensors"),
                });
                Tensor::Gpu(q.flash_attention(k, v, scale, gpu_mask))
            }
            // CPU path and GPU+BatchKeyMask fallback - use composite operations via Tensor methods
            _ => self.flash_attention_composite_impl(k, v, scale, mask),
        }
    }

    /// Implementation of flash attention using Tensor composite operations.
    /// Works on both CPU and GPU tensors (GPU uses individual ops instead of fused kernel).
    fn flash_attention_composite_impl(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<(&Tensor<2, D, ConcreteTensor<D, 2>>, MaskKind)>,
    ) -> Self {
        let q_shape = self.shape();
        let k_shape = k.shape();

        let batch = q_shape[0];
        let num_heads = q_shape[1];
        let q_seq_len = q_shape[2];
        let head_dim = q_shape[3];
        let num_kv_heads = k_shape[1];
        let kv_seq_len = k_shape[2];

        assert!(
            num_heads.is_multiple_of(num_kv_heads),
            "Number of Q heads ({}) must be divisible by number of K/V heads ({})",
            num_heads,
            num_kv_heads
        );

        let num_key_value_groups = num_heads / num_kv_heads;

        // For GQA/MQA, we need to expand K and V to match Q heads
        let (k_expanded, v_expanded): (Tensor<4, D, _>, Tensor<4, D, _>) =
            if num_key_value_groups > 1 {
                // Expand K and V from [batch, num_kv_heads, kv_seq_len, head_dim]
                // to [batch, num_heads, kv_seq_len, head_dim]
                let k_reshaped: Tensor<5, D, _> =
                    k.reshape([batch, num_kv_heads, 1, kv_seq_len, head_dim]);
                let v_reshaped: Tensor<5, D, _> =
                    v.reshape([batch, num_kv_heads, 1, kv_seq_len, head_dim]);

                let k_broadcast = k_reshaped.broadcast_as([
                    batch,
                    num_kv_heads,
                    num_key_value_groups,
                    kv_seq_len,
                    head_dim,
                ]);
                let v_broadcast = v_reshaped.broadcast_as([
                    batch,
                    num_kv_heads,
                    num_key_value_groups,
                    kv_seq_len,
                    head_dim,
                ]);

                (
                    k_broadcast
                        .reshape([batch, num_heads, kv_seq_len, head_dim])
                        .to_concrete(),
                    v_broadcast
                        .reshape([batch, num_heads, kv_seq_len, head_dim])
                        .to_concrete(),
                )
            } else {
                (k.clone(), v.clone())
            };

        // Q @ K^T -> [batch, num_heads, q_seq_len, kv_seq_len]
        let k_t = k_expanded.transpose(2, 3);
        let scores = self.mat_mul(&k_t);

        // Scale the scores
        let scores_scaled = scores.mul_scalar(D::from_f32(scale));

        // Apply mask if provided
        let scores_masked = if let Some((m, kind)) = mask {
            let m_shape = m.shape();
            let mask_4d: Tensor<4, D, _> = match kind {
                MaskKind::QKMask => {
                    // Mask is [q_seq_len, kv_seq_len]
                    assert_eq!(
                        m_shape,
                        [q_seq_len, kv_seq_len],
                        "QKMask shape {:?} does not match expected [{}, {}]",
                        m_shape,
                        q_seq_len,
                        kv_seq_len
                    );
                    m.reshape([1, 1, q_seq_len, kv_seq_len])
                }
                MaskKind::BatchKeyMask => {
                    // Mask is [batch, kv_seq_len] — per-token validity mask
                    assert_eq!(
                        m_shape,
                        [batch, kv_seq_len],
                        "BatchKeyMask shape {:?} does not match expected [{}, {}]",
                        m_shape,
                        batch,
                        kv_seq_len
                    );
                    m.reshape([m_shape[0], 1, 1, m_shape[1]])
                }
            };
            let mask_broadcast = mask_4d.broadcast_as([batch, num_heads, q_seq_len, kv_seq_len]);
            (scores_scaled + mask_broadcast).to_concrete()
        } else {
            scores_scaled
        };

        // Softmax along last dimension
        // max(scores) for numerical stability
        let scores_shape = scores_masked.shape();
        let max_scores = scores_masked
            .max_keepdim::<3>(3)
            .broadcast_as(scores_shape)
            .to_concrete();
        let scores_shifted = scores_masked - max_scores;
        // Materialize exp_scores since sum_keepdim is a reduction that needs concrete data
        let exp_scores = scores_shifted.exp();
        let sum_exp = exp_scores
            .sum_keepdim::<3>(3)
            .broadcast_as(scores_shape)
            .to_concrete();
        let attn_weights = exp_scores / sum_exp;

        // attn_weights @ V -> [batch, num_heads, q_seq_len, head_dim]
        attn_weights.mat_mul(&v_expanded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_flash_attention_cpu() {
        // Test flash attention - 4D tensors [batch, heads, seq, dim]
        let q_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let k_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let v_data = vec![1.0f32, 2.0, 3.0, 4.0];

        let q: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2, 2], &q_data));
        let k: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2, 2], &k_data));
        let v: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2, 2], &v_data));

        let scale = 1.0 / (2.0_f32.sqrt());

        let output = q.flash_attention(&k, &v, scale, None);
        let result = output.as_slice().await.unwrap();

        // Verify output shape
        assert_eq!(result.shape(), &[1, 1, 2, 2]);

        // Verify output is finite (not NaN or infinity)
        for i in 0..2 {
            for j in 0..2 {
                let val = result[[0, 0, i, j]];
                assert!(
                    val.is_finite(),
                    "Non-finite value at [{}, {}]: {}",
                    i,
                    j,
                    val
                );
            }
        }
    }

    #[tokio::test]
    async fn test_flash_attention_cpu_with_mask() {
        // Test flash attention with causal mask
        let q_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let k_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let v_data = vec![1.0f32, 2.0, 3.0, 4.0];

        let q: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2, 2], &q_data));
        let k: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2, 2], &k_data));
        let v: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 1, 2, 2], &v_data));

        // Causal mask: [[0, -inf], [0, 0]]
        let neg_inf = f32::NEG_INFINITY;
        let mask_data = vec![0.0f32, neg_inf, 0.0, 0.0];
        let mask: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 2], &mask_data));

        let scale = 1.0 / (2.0_f32.sqrt());

        let output = q.flash_attention(&k, &v, scale, Some((&mask, MaskKind::QKMask)));
        let result = output.as_slice().await.unwrap();

        // With causal mask, first row should only attend to first position
        // So first row output should equal first row of V
        let tolerance = 0.01;
        assert!(
            (result[[0, 0, 0, 0]] - v_data[0]).abs() < tolerance,
            "First position should attend only to itself with causal mask: got {}, expected {}",
            result[[0, 0, 0, 0]],
            v_data[0]
        );
        assert!(
            (result[[0, 0, 0, 1]] - v_data[1]).abs() < tolerance,
            "First position should attend only to itself with causal mask: got {}, expected {}",
            result[[0, 0, 0, 1]],
            v_data[1]
        );
    }

    #[tokio::test]
    async fn test_flash_attention_gqa_cpu() {
        // Test GQA where K/V have fewer heads than Q
        // Q: [1, 4, 2, 2] - 4 heads
        // K/V: [1, 2, 2, 2] - 2 heads (each shared by 2 Q heads)
        let q_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let v_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 + 2.0).collect();

        let q: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 4, 2, 2], &q_data));
        let k: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2, 2], &k_data));
        let v: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 2, 2, 2], &v_data));

        let scale = 1.0 / (2.0_f32.sqrt());

        let output = q.flash_attention(&k, &v, scale, None);
        let result = output.as_slice().await.unwrap();

        // Verify output shape matches Q shape
        assert_eq!(result.shape(), &[1, 4, 2, 2]);

        // Verify output is finite
        for h in 0..4 {
            for s in 0..2 {
                for d in 0..2 {
                    let val = result[[0, h, s, d]];
                    assert!(
                        val.is_finite(),
                        "Non-finite value at [0, {}, {}, {}]: {}",
                        h,
                        s,
                        d,
                        val
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_flash_attention_cpu_with_batch_key_mask() {
        // Test flash attention with a BatchKeyMask (padding mask)
        // batch=2, heads=1, seq_len=3, dim=2
        // Second sentence has last token masked (padding)
        let q_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let v_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 2.0).collect();

        let q: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 1, 3, 2], &q_data));
        let k: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 1, 3, 2], &k_data));
        let v: Tensor<4, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 1, 3, 2], &v_data));

        // BatchKeyMask: [batch=2, kv_seq_len=3]
        // First sentence: all tokens valid. Second sentence: last token is padding.
        let neg_inf = f32::NEG_INFINITY;
        let mask_data = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, neg_inf];
        let mask: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &mask_data));

        let scale = 1.0 / (2.0_f32.sqrt());
        let output = q.flash_attention(&k, &v, scale, Some((&mask, MaskKind::BatchKeyMask)));
        let result = output.as_slice().await.unwrap();

        assert_eq!(result.shape(), &[2, 1, 3, 2]);

        // Verify all values are finite
        for b in 0..2 {
            for s in 0..3 {
                for d in 0..2 {
                    let val = result[[b, 0, s, d]];
                    assert!(
                        val.is_finite(),
                        "Non-finite value at [{}, 0, {}, {}]: {}",
                        b,
                        s,
                        d,
                        val
                    );
                }
            }
        }
    }
}
