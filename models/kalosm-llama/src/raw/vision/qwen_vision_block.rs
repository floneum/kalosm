use fusor::{
    cache::KvCache,
    layers::{Linear, RmsNorm},
    CastTensor, CastTo, Device, FloatDataType, QMatrix, SimdElement, StrideSpec, Tensor,
    VarBuilder,
};

use fusor::RopeCache;

use crate::raw::attention_layer::LlamaFeedForward;

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
        let trace = std::env::var_os("KALOSM_TRACE_VBLOCK").is_some();
        let flush = |t: &Tensor<3, F>| {
            if trace {
                t.as_gpu().map(|g| g.materialize_sync());
            }
        };
        let xs_3d = xs.unsqueeze(0).to_concrete(); // [1, seq, dim]
        flush(&xs_3d);
        let t0 = std::time::Instant::now();
        let after_norm = self.norm1.forward_generic(&xs_3d);
        flush(&after_norm);
        if trace {
            eprintln!("    norm1: {:.2?}", t0.elapsed());
        }
        let t1 = std::time::Instant::now();
        let after_attention = self
            .attn
            .forward(&after_norm, cu_seqlens, rope_cache, cache)?;
        flush(&after_attention);
        if trace {
            eprintln!("    attn:  {:.2?}", t1.elapsed());
        }

        // Work in f32 for tensor addition
        let xs_3d_f32: Tensor<3, f32> = xs_3d.cast();
        let after_attention_f32: Tensor<3, f32> = after_attention.cast();
        let t2 = std::time::Instant::now();
        let xs_3d: Tensor<3, F> = (xs_3d_f32 + after_attention_f32).cast();
        flush(&xs_3d);
        if trace {
            eprintln!("    res1:  {:.2?}", t2.elapsed());
        }

        let t3 = std::time::Instant::now();
        let after_norm2 = self.norm2.forward_generic(&xs_3d);
        flush(&after_norm2);
        if trace {
            eprintln!("    norm2: {:.2?}", t3.elapsed());
        }
        let t4 = std::time::Instant::now();
        let mlp_out = self.mlp.forward(&after_norm2);
        flush(&mlp_out);
        if trace {
            eprintln!("    mlp:   {:.2?}", t4.elapsed());
        }

        // Work in f32 for tensor addition
        let xs_3d_f32: Tensor<3, f32> = xs_3d.cast();
        let mlp_out_f32: Tensor<3, f32> = mlp_out.cast();
        let t5 = std::time::Instant::now();
        let out: Tensor<3, F> = (xs_3d_f32 + mlp_out_f32).cast();
        flush(&out);
        if trace {
            eprintln!("    res2:  {:.2?}", t5.elapsed());
        }

        Ok(out.squeeze(0).to_concrete())
    }
}

struct VisionAttention<F: FloatDataType + SimdElement> {
    /// Fused Q/K/V projection: a single Linear whose weight is the row-wise
    /// concatenation of the per-tensor q/k/v weights. The previous code
    /// dispatched 3 separate matmuls per layer (96 dispatches across 32
    /// vision blocks); one fused matmul of triple output width does the same
    /// arithmetic with a third the dispatch count, and the wider N better
    /// saturates the shared-memory tile reuse.
    qkv: VisionQkv<F>,
    proj: Linear<F>,
    head_count: usize,
    head_dim: usize,
    embed_dim: usize,
}

enum VisionQkv<F: SimdElement> {
    Fused(Linear<F>),
    Split {
        q: Linear<F>,
        k: Linear<F>,
        v: Linear<F>,
    },
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
        let q_w = vb.get("attn_q.weight", device)?;
        let k_w = vb.get("attn_k.weight", device)?;
        let v_w = vb.get("attn_v.weight", device)?;
        // Concatenate biases on the same axis so the fused matmul's epilogue
        // bias add covers all three projections at once.
        let q_b: Tensor<1, F> = vb.get("attn_q.bias", device)?.dequantize().cast();
        let k_b: Tensor<1, F> = vb.get("attn_k.bias", device)?.dequantize().cast();
        let v_b: Tensor<1, F> = vb.get("attn_v.bias", device)?.dequantize().cast();
        let qkv = if let Some(qkv_weight) = QMatrix::concat_rows(&[&q_w, &k_w, &v_w]) {
            let qkv_bias: Tensor<1, F> = fusor::cat([q_b, k_b, v_b], 0).to_concrete();
            VisionQkv::Fused(Linear::new(qkv_weight, Some(qkv_bias)))
        } else {
            VisionQkv::Split {
                q: Linear::new(q_w, Some(q_b)),
                k: Linear::new(k_w, Some(k_b)),
                v: Linear::new(v_w, Some(v_b)),
            }
        };
        let proj = Linear::new(
            vb.get("attn_out.weight", device)?,
            Some(vb.get("attn_out.bias", device)?.dequantize().cast()),
        );

        Ok(Self {
            qkv,
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
        let trace_attn = std::env::var_os("KALOSM_TRACE_ATTN").is_some();
        let [bsz, seq_len, _] = xs.shape();
        let t_qkv = std::time::Instant::now();

        // One fused qkv matmul (output is [1, seq, 3 * embed_dim]); narrow
        // out the q/k/v slices as views — narrow is a layout-only op so we
        // pay one matmul + three free splits instead of three matmuls.
        let (q, k, v): (Tensor<3, f32>, Tensor<3, f32>, Tensor<3, f32>) = match &self.qkv {
            VisionQkv::Fused(qkv) => {
                let qkv: Tensor<3, f32> = qkv.forward_generic(xs).cast();
                if trace_attn {
                    qkv.as_gpu().map(|g| g.materialize_sync());
                    eprintln!("      qkv:   {:.2?}", t_qkv.elapsed());
                }
                let q = qkv
                    .narrow(2, 0, self.embed_dim)
                    .reshape([seq_len, self.head_count, self.head_dim])
                    .to_concrete();
                let k = qkv
                    .narrow(2, self.embed_dim, self.embed_dim)
                    .reshape([seq_len, self.head_count, self.head_dim])
                    .to_concrete();
                let v = qkv
                    .narrow(2, 2 * self.embed_dim, self.embed_dim)
                    .reshape([seq_len, self.head_count, self.head_dim])
                    .to_concrete();
                (q, k, v)
            }
            VisionQkv::Split { q, k, v } => {
                let q = q
                    .forward_generic(xs)
                    .cast()
                    .reshape([seq_len, self.head_count, self.head_dim])
                    .to_concrete();
                let k = k
                    .forward_generic(xs)
                    .cast()
                    .reshape([seq_len, self.head_count, self.head_dim])
                    .to_concrete();
                let v = v
                    .forward_generic(xs)
                    .cast()
                    .reshape([seq_len, self.head_count, self.head_dim])
                    .to_concrete();
                if trace_attn {
                    v.as_gpu().map(|g| g.materialize_sync());
                    eprintln!("      qkv:   {:.2?} (split)", t_qkv.elapsed());
                }
                (q, k, v)
            }
        };

        // Transpose to [heads, seq, dim] -> [1, heads, seq, dim] (batch=1) so
        // rope_normal_pair_fused can run as a single GPU kernel per layer
        // covering both q and k. Each prior layer dispatched ~16 element-wise
        // kernels (cat+neg+mul+add+slice_assign) per q/k for the unfused rope
        // composite; the fused kernel collapses all of that into one dispatch.
        let q_4d: Tensor<4, f32> = q.transpose(0, 1).unsqueeze(0).to_concrete();
        let k_4d: Tensor<4, f32> = k.transpose(0, 1).unsqueeze(0).to_concrete();

        let cos = rope_cache.cos().narrow(0, 0, seq_len).to_concrete();
        let sin = rope_cache.sin().narrow(0, 0, seq_len).to_concrete();
        let (query_states, key_states) = q_4d.rope_normal_pair_fused(&k_4d, &cos, &sin);

        // CRITICAL: also force V to a fresh row-major buffer here. We need
        // V's slices (later, per window) to start from a contiguous tensor;
        // narrowing a transpose-view produces another non-contiguous view.
        let value_states = v.transpose(0, 1).unsqueeze(0).to_concrete();
        let t_after_rope = std::time::Instant::now();
        if trace_attn {
            value_states.as_gpu().map(|g| g.materialize_sync());
            eprintln!(
                "      rope:  {:.2?} (incl. q/k/v split + transpose)",
                t_qkv.elapsed()
            );
        }

        // Cache append (cache uses f32 for SIMD operations)
        // query_states, key_states, value_states are already f32 from the rope computation
        let (key_states_f32, value_states_f32): (Tensor<4, f32>, Tensor<4, f32>) = match cache {
            None => (key_states.to_concrete(), value_states.to_concrete()),
            Some(cache) => cache.append(&xs.device(), &key_states, &value_states),
        };

        if trace_attn {
            eprintln!("      mask:  {:.2?}", t_after_rope.elapsed());
        }

        // The attention pattern is block-diagonal per `cu_seqlens`. The dense
        // masked path runs the full M×M flash kernel even when most of the
        // mask is -inf — that's where ~95% of vision-encoder GPU time goes.
        // For window-attention layers we instead slice Q/K/V per window and
        // run dense (unmasked) flash on each — same arithmetic the
        // block-diagonal mask intended, but at window² scale (~64²) rather
        // than seq² (1944²). Full-attention layers (cu_seqlens=[0, seq])
        // skip the slicing and just run one regular flash call.
        let t_flash = std::time::Instant::now();
        let query_f32 = query_states;
        let key_f32 = key_states_f32;
        let value_f32 = value_states_f32;
        let is_full_attn =
            cu_seqlens.len() == 2 && cu_seqlens[0] == 0 && cu_seqlens[1] as usize == seq_len;
        let attn_out_4d = if is_full_attn {
            query_f32.flash_attention(
                &key_f32,
                &value_f32,
                1.0 / (self.head_dim as f64).sqrt() as f32,
                None,
            )
        } else {
            let scale = 1.0 / (self.head_dim as f64).sqrt() as f32;
            let mut windows = Vec::with_capacity(cu_seqlens.len().saturating_sub(1));
            for pair in cu_seqlens.windows(2) {
                let start = pair[0] as usize;
                let end = pair[1] as usize;
                let len = end - start;
                if len == 0 {
                    continue;
                }
                windows.push((start, len));
            }

            let mut run_outputs = Vec::new();
            let mut window_idx = 0;
            while window_idx < windows.len() {
                let (start, len) = windows[window_idx];
                let mut run_count = 1;
                while window_idx + run_count < windows.len()
                    && windows[window_idx + run_count].1 == len
                {
                    run_count += 1;
                }

                let run_specs = [
                    StrideSpec::dim(1, self.head_count),
                    StrideSpec::dim_with(2, run_count, len).with_offset(start),
                    StrideSpec::dim(2, len),
                    StrideSpec::dim(3, self.head_dim),
                ];
                let q_run = query_f32.restride(run_specs).to_concrete();
                let k_run = key_f32.restride(run_specs).to_concrete();
                let v_run = value_f32.restride(run_specs).to_concrete();

                let run_out = q_run.flash_attention(&k_run, &v_run, scale, None);
                let run_out = run_out
                    .reshape([self.head_count, run_count * len, self.head_dim])
                    .unsqueeze(0)
                    .to_concrete();
                run_outputs.push(run_out);
                window_idx += run_count;
            }

            fusor::cat(run_outputs, 2).to_concrete()
        };

        // After flash: transpose+reshape to [b_sz, seq, hidden_size] then proj.
        let attn_output = attn_out_4d.transpose(1, 2);
        let attn_output = attn_output.reshape([bsz, seq_len, self.embed_dim]);
        let output: Tensor<3, F> = self.proj.forward_generic(&attn_output.cast());
        if trace_attn {
            output.as_gpu().map(|g| g.materialize_sync());
            eprintln!("      flash+proj: {:.2?}", t_flash.elapsed());
        }

        Ok(output)
    }
}
