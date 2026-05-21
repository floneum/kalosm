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
        + std::ops::Div<Output = D>
        + Copy,
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
                if !q.device().subgroup_kernels_supported() {
                    let cpu_q = tensor4_to_cpu(q);
                    let cpu_k = tensor4_to_cpu(k);
                    let cpu_v = tensor4_to_cpu(v);
                    let cpu_mask = mask.map(|(m, kind)| {
                        let Tensor::Gpu(mask) = m else {
                            panic!("Mask must be on the same device as other tensors");
                        };
                        (tensor2_to_cpu(mask), kind)
                    });
                    let cpu_mask_ref = cpu_mask.as_ref().map(|(mask, kind)| (mask, *kind));
                    let cpu_output = cpu_q.flash_attention(&cpu_k, &cpu_v, scale, cpu_mask_ref);
                    return tensor4_to_gpu(cpu_output, q.device());
                }
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

fn tensor4_to_cpu<D>(tensor: &fusor_core::Tensor<4, D>) -> Tensor<4, D>
where
    D: SimdElement + DataType + Copy,
{
    let shape = *tensor.shape();
    let slice = pollster::block_on(tensor.as_slice()).expect("failed to read tensor");
    let mut values = Vec::with_capacity(shape.iter().product());
    for b in 0..shape[0] {
        for h in 0..shape[1] {
            for s in 0..shape[2] {
                for d in 0..shape[3] {
                    values.push(slice[[b, h, s, d]]);
                }
            }
        }
    }
    Tensor::Cpu(fusor_cpu::Tensor::from_slice(shape, &values))
}

fn tensor4_to_gpu<D>(tensor: Tensor<4, D>, device: &fusor_core::Device) -> Tensor<4, D>
where
    D: SimdElement + DataType + Copy,
{
    let Tensor::Cpu(tensor) = tensor else {
        unreachable!("subgroup fallback should produce a CPU tensor");
    };
    let shape = tensor.shape();
    let slice = tensor.as_slice();
    let mut values = Vec::with_capacity(shape.iter().product());
    for b in 0..shape[0] {
        for h in 0..shape[1] {
            for s in 0..shape[2] {
                for d in 0..shape[3] {
                    values.push(slice[[b, h, s, d]]);
                }
            }
        }
    }
    Tensor::Gpu(fusor_core::Tensor::from_slice(device, shape, &values))
}

fn tensor2_to_cpu<D>(tensor: &fusor_core::Tensor<2, D>) -> Tensor<2, D>
where
    D: SimdElement + DataType + Copy,
{
    let shape = *tensor.shape();
    let slice = pollster::block_on(tensor.as_slice()).expect("failed to read tensor");
    let mut values = Vec::with_capacity(shape.iter().product());
    for row in 0..shape[0] {
        for col in 0..shape[1] {
            values.push(slice[[row, col]]);
        }
    }
    Tensor::Cpu(fusor_cpu::Tensor::from_slice(shape, &values))
}
