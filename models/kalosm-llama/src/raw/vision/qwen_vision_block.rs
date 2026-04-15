use fusor::{
    cache::{AttentionMask, KvCache},
    layers::{Linear, RmsNorm},
    CastTensor, CastTo, Device, FloatDataType, SimdElement, Tensor, VarBuilder,
};

use fusor::RopeCache;

use crate::raw::attention_layer::{forward_attention_qkv_f32, LlamaFeedForward};

pub(crate) struct VisionBlock<F: FloatDataType + SimdElement> {
    norm1: RmsNorm<1, F>,
    norm2: RmsNorm<1, F>,
    mlp: LlamaFeedForward<F>,
    attn: VisionAttention<F>,
}

impl<F: FloatDataType + SimdElement + Default> VisionBlock<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    pub(crate) fn new(
        vb: &mut VarBuilder,
        device: &Device,
        head_count: usize,
        head_dim: usize,
        embed_dim: usize,
        layer_norm_eps: f64,
    ) -> fusor::Result<Self> {
        // norm1, norm2
        let norm1_weight: Tensor<1, F> = vb.get("ln1.weight", device)?.dequantize().cast();
        let norm1 = RmsNorm::new(norm1_weight, None, layer_norm_eps as f32);

        let norm2_weight: Tensor<1, F> = vb.get("ln2.weight", device)?.dequantize().cast();
        let norm2 = RmsNorm::new(norm2_weight, None, layer_norm_eps as f32);

        // MLP
        let gate = vb.get("ffn_gate.weight", device)?;
        let gate_bias: Tensor<1, F> = vb.get("ffn_gate.bias", device)?.dequantize().cast();
        let down = vb.get("ffn_down.weight", device)?;
        let down_bias: Tensor<1, F> = vb.get("ffn_down.bias", device)?.dequantize().cast();
        let up = vb.get("ffn_up.weight", device)?;
        let up_bias: Tensor<1, F> = vb.get("ffn_up.bias", device)?.dequantize().cast();
        let mlp = LlamaFeedForward::new_with_bias(
            gate,
            Some(gate_bias),
            down,
            Some(down_bias),
            up,
            Some(up_bias),
        );

        let attn = VisionAttention::new(vb, device, head_count, head_dim, embed_dim)?;

        Ok(Self {
            norm1,
            norm2,
            mlp,
            attn,
        })
    }

    pub(crate) fn forward(
        &self,
        xs: &Tensor<2, F>,
        cu_seqlens: &[u32],
        rope_cache: &RopeCache,
        cache: Option<&mut KvCache<f32>>,
    ) -> fusor::Result<Tensor<2, F>> {
        let xs_3d = xs.unsqueeze(0).to_concrete(); // [1, seq, dim]
        let after_norm = self.norm1.forward_generic(&xs_3d);
        let after_attention = self
            .attn
            .forward(&after_norm, cu_seqlens, rope_cache, cache)?;

        // Work in f32 for tensor addition
        let xs_3d_f32: Tensor<3, f32> = xs_3d.cast();
        let after_attention_f32: Tensor<3, f32> = after_attention.cast();
        let xs_3d: Tensor<3, F> = (xs_3d_f32 + after_attention_f32).cast();

        let after_norm2 = self.norm2.forward_generic(&xs_3d);
        let mlp_out = self.mlp.forward(&after_norm2); // LlamaFeedForward expects Tensor<3, F>

        // Work in f32 for tensor addition
        let xs_3d_f32: Tensor<3, f32> = xs_3d.cast();
        let mlp_out_f32: Tensor<3, f32> = mlp_out.cast();
        let out: Tensor<3, F> = (xs_3d_f32 + mlp_out_f32).cast();

        Ok(out.squeeze(0).to_concrete())
    }
}

struct VisionAttention<F: FloatDataType + SimdElement> {
    q: Linear<F>,
    k: Linear<F>,
    v: Linear<F>,
    proj: Linear<F>,
    head_count: usize,
    head_dim: usize,
    embed_dim: usize,
}

impl<F: FloatDataType + SimdElement + Default> VisionAttention<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    fn new(
        vb: &mut VarBuilder,
        device: &Device,
        head_count: usize,
        head_dim: usize,
        embed_dim: usize,
    ) -> fusor::Result<Self> {
        let q = Linear::new(
            vb.get("attn_q.weight", device)?,
            Some(vb.get("attn_q.bias", device)?.dequantize().cast()),
        );
        let k = Linear::new(
            vb.get("attn_k.weight", device)?,
            Some(vb.get("attn_k.bias", device)?.dequantize().cast()),
        );
        let v = Linear::new(
            vb.get("attn_v.weight", device)?,
            Some(vb.get("attn_v.bias", device)?.dequantize().cast()),
        );
        let proj = Linear::new(
            vb.get("attn_out.weight", device)?,
            Some(vb.get("attn_out.bias", device)?.dequantize().cast()),
        );

        Ok(Self {
            q,
            k,
            v,
            proj,
            head_count,
            head_dim,
            embed_dim,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<3, F>, // [1, seq, dim]
        cu_seqlens: &[u32],
        rope_cache: &RopeCache,
        cache: Option<&mut KvCache<f32>>,
    ) -> fusor::Result<Tensor<3, F>> {
        let [bsz, seq_len, _] = xs.shape();

        // Work in f32 for qkv linear ops
        let q: Tensor<3, f32> = self
            .q
            .forward_generic(xs)
            .reshape([seq_len, self.head_count, self.head_dim])
            .cast();
        let k: Tensor<3, f32> = self
            .k
            .forward_generic(xs)
            .reshape([seq_len, self.head_count, self.head_dim])
            .cast();
        let v: Tensor<3, f32> = self
            .v
            .forward_generic(xs)
            .reshape([seq_len, self.head_count, self.head_dim])
            .cast();

        let sin = rope_cache.sin().clone();
        let cos = rope_cache.cos().clone();

        // sin/cos: [total_seq, head_dim/2]
        // Expand to [total_seq, head_count, head_dim]
        let last_dim = 1; // The last dimension index for 2D tensor
        let sin = Tensor::cat(vec![sin.clone(); 2], last_dim)
            .unsqueeze(1)
            .expand([seq_len, self.head_count, self.head_dim])
            .to_concrete();
        let cos = Tensor::cat(vec![cos.clone(); 2], last_dim)
            .unsqueeze(1)
            .expand([seq_len, self.head_count, self.head_dim])
            .to_concrete();

        // Rotate half (in f32)
        let rotate_half = |x: &Tensor<3, f32>| {
            // x: [seq, heads, dim]
            let last_dim = x.shape().last().copied().unwrap();
            let half = last_dim / 2;
            let x1 = x.narrow(2, 0, half).to_concrete();
            let x2 = x.narrow(2, half, half).to_concrete();
            let neg_x2 = (-x2).to_concrete();
            Tensor::cat([neg_x2, x1], 2)
        };

        let q_embed = ((q.clone() * cos.clone()) + (rotate_half(&q) * sin.clone())).to_concrete();
        let k_embed = ((k.clone() * cos) + (rotate_half(&k) * sin)).to_concrete();

        // Transpose to [heads, seq, dim] -> [1, heads, seq, dim] (batch=1)
        let query_states = q_embed.transpose(0, 1).unsqueeze(0).to_concrete();
        let key_states = k_embed.transpose(0, 1).unsqueeze(0).to_concrete();
        let value_states = v.transpose(0, 1).unsqueeze(0).to_concrete();

        // Cache append (cache uses f32 for SIMD operations)
        // query_states, key_states, value_states are already f32 from the rope computation
        let (key_states_f32, value_states_f32): (Tensor<4, f32>, Tensor<4, f32>) = match cache {
            None => (key_states.to_concrete(), value_states.to_concrete()),
            Some(cache) => cache.append(&xs.device(), &key_states, &value_states),
        };

        // Mask
        let mut mask_vec = vec![f32::NEG_INFINITY; seq_len * seq_len];
        for pair in cu_seqlens.windows(2) {
            let last = pair[0] as usize;
            let next = pair[1] as usize;
            for i in last..next {
                for j in last..next {
                    mask_vec[i * seq_len + j] = 0.0;
                }
            }
        }

        let mask_tensor: Tensor<2, f32> = Tensor::new(&xs.device(), &mask_vec)
            .reshape([seq_len, seq_len])
            .to_concrete();
        let mask = AttentionMask::new(mask_tensor);

        // query_states is already f32
        let query_f32 = query_states;
        let key_f32 = key_states_f32;
        let value_f32 = value_states_f32;

        let output = forward_attention_qkv_f32(
            &query_f32,
            &key_f32,
            &value_f32,
            &self.proj,
            Some(&mask),
            self.head_dim,
            bsz,
            seq_len,
            self.embed_dim,
        );

        Ok(output)
    }
}
