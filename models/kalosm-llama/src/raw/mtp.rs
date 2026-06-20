use super::*;

pub(crate) struct Gemma4MtpAssistant<F: FloatDataType + SimdElement = f32> {
    config: Arc<LlamaConfig<F>>,
    pre_projection: QMatrix,
    post_projection: QMatrix,
    layers: Vec<LlamaAttention<F>>,
    norm: RmsNorm<1, F>,
    output: QMatrix,
    layer_is_sliding: Vec<bool>,
}

pub(crate) struct Gemma4MtpStep {
    pub(crate) logits: Tensor<1, f32>,
    pub(crate) h_nextn: Tensor<2, f32>,
}

impl<F: FloatDataType + SimdElement + FloatOps + MatmulImpl> Gemma4MtpAssistant<F>
where
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn from_gguf<R: std::io::Seek + std::io::Read>(
        source: &mut ShardedVarBuilder<R>,
        device: &Device,
    ) -> std::result::Result<Self, LlamaSourceError>
    where
        f32: CastTensor<F> + CastTo<F>,
        F: CastTensor<f32> + CastTo<f32>,
    {
        super::block_on_ready(Self::from_var_source(source, device))
    }

    pub(crate) async fn from_var_source<S: LlamaVarSource>(
        source: &mut S,
        device: &Device,
    ) -> std::result::Result<Self, LlamaSourceError>
    where
        f32: CastTensor<F> + CastTo<F>,
        F: CastTensor<f32> + CastTo<f32>,
    {
        let dequantize_1d = |qmatrix: QMatrix| -> Tensor<1, F> {
            let shape = qmatrix.shape();
            if shape.len() == 1 {
                let w1d: Tensor<1, f32> = qmatrix.dequantize();
                w1d.cast()
            } else if shape.len() == 2 {
                let w2d: Tensor<2, f32> = qmatrix.dequantize();
                w2d.reshape([w2d.shape()[0] * w2d.shape()[1]])
                    .to_concrete()
                    .cast()
            } else {
                panic!(
                    "Expected 1D or 2D tensor for dequantize_1d, got {}D",
                    shape.len()
                )
            }
        };
        let decode_norm = |qmatrix: QMatrix, eps: f64| -> Result<RmsNorm<1, F>> {
            let weight = dequantize_1d(qmatrix);
            Ok(RmsNorm::new(weight, None, eps as f32))
        };

        let architecture = source.get("general.architecture")?.to_string()?.clone();
        if architecture.as_ref() != "gemma4-assistant" {
            return Err(fusor::Error::msg(format!(
                "MTP assistant architecture must be gemma4-assistant, got {architecture}"
            ))
            .into());
        }

        let block_count = source.get(".block_count")?.to_u32()? as usize;
        let context_length = source.get(".context_length")?.to_u32()? as usize;
        let embedding_length = source.get(".embedding_length")?.to_u32()? as usize;
        let target_embedding_length = source.get(".embedding_length_out")?.to_u32()? as usize;
        let head_count = source.get(".attention.head_count")?.to_u32()? as usize;
        let head_count_kv = source.get(".attention.head_count_kv")?.to_u32()? as usize;
        let rms_norm_eps = source.get(".attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = source
            .get(".rope.freq_base")
            .and_then(|m| Ok(m.to_f32()?))
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let rope_freq_base_sliding = source
            .get(".rope.freq_base_swa")
            .and_then(|m| Ok(m.to_f32()?))
            .ok()
            .or_else(|| {
                source
                    .get(".rope.local_freq_base")
                    .and_then(|m| Ok(m.to_f32()?))
                    .ok()
            })
            .unwrap_or(GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING);
        let sliding_window_size = source
            .get(".attention.sliding_window")
            .and_then(|m| Ok(m.to_u32()?))
            .ok()
            .map(|x| x as usize);
        let head_dim = source
            .get(".attention.key_length")
            .and_then(|m| Ok(m.to_u32()?))
            .ok()
            .map(|x| x as usize)
            .unwrap_or_else(|| target_embedding_length / head_count);
        let head_dim_swa = source
            .get(".attention.key_length_swa")
            .and_then(|m| Ok(m.to_u32()?))
            .ok()
            .map(|x| x as usize)
            .unwrap_or(head_dim);
        let sliding_window_pattern = source
            .get(".attention.sliding_window_pattern")
            .ok()
            .and_then(|m| {
                let values = m.to_array().ok()?;
                values
                    .iter()
                    .map(|value| value.to_bool().ok())
                    .collect::<Option<Vec<_>>>()
            });
        let layer_is_sliding = sliding_window_pattern.unwrap_or_else(|| {
            (0..block_count)
                .map(|idx| sliding_window_size.is_some() && idx + 1 < block_count)
                .collect()
        });
        let layer_sliding_window_sizes = sliding_window_size.map(|window| {
            layer_is_sliding
                .iter()
                .map(|is_sliding| is_sliding.then_some(window))
                .collect::<Vec<_>>()
        });
        let rope_freq_weight: Option<Tensor<1, F>> = source
            .tensor("rope_freqs.weight", device)
            .await
            .ok()
            .map(&dequantize_1d);

        let config = Arc::new(LlamaConfig {
            rope_freq_weight,
            rope_theta: rope_freq_base,
            context_length,
            head_dimension: head_dim,
            n_layer: block_count,
            start_token_string: String::new(),
            stop_tokens: Vec::new(),
            stop_token_string: String::new(),
            chat_template: None,
            rope_scaling: None,
            sliding_window_type: None,
            sliding_window_size,
            layer_sliding_window_sizes,
            final_logit_softcapping: None,
            per_layer_embedding_length: None,
            vision_start_token: None,
            _vision_end_token: None,
            image_pad_token: None,
            image_start_token: None,
            image_end_token: None,
            video_pad_token: None,
            mrope_sections: None,
        });

        let rope = RopeImplementation::new_with_head_dimension(
            &config,
            head_dim,
            config.rope_freq_weight.as_ref(),
            rope_freq_base,
            device,
        )?;
        let sliding_rope = RopeImplementation::new_with_head_dimension(
            &config,
            head_dim_swa,
            None,
            rope_freq_base_sliding,
            device,
        )?;

        let pre_projection = source.tensor("nextn.pre_projection.weight", device).await?;
        let post_projection = source
            .tensor("nextn.post_projection.weight", device)
            .await?;
        let output_norm = source.tensor("output_norm.weight", device).await?;
        let norm = decode_norm(output_norm, rms_norm_eps)?;
        let token_embd = source.tensor("token_embd.weight", device).await?;
        let output = source
            .tensor("output.weight", device)
            .await
            .unwrap_or_else(|_| token_embd.clone());

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let layer_is_sliding = layer_is_sliding.get(layer_idx).copied().unwrap_or(false);
            let layer_head_dim = if layer_is_sliding {
                head_dim_swa
            } else {
                head_dim
            };
            let layer_attention_width = head_count * layer_head_dim;
            let layer_sliding_window_size = config
                .layer_sliding_window_sizes
                .as_ref()
                .and_then(|sizes| sizes.get(layer_idx).copied().flatten());
            let rope_cache = if layer_is_sliding {
                sliding_rope.clone()
            } else {
                rope.clone()
            };
            let prefix = format!("blk.{layer_idx}");
            let q = source
                .tensor(&format!("{prefix}.attn_q.weight"), device)
                .await?;
            let q_norm = source
                .tensor(&format!("{prefix}.attn_q_norm.weight"), device)
                .await
                .ok();
            let attention_variant = AttentionVariant::Separate(Box::new(SeparateAttention {
                attention_wq: q,
                attention_qkv: None,
                attention_q_norm: q_norm
                    .map(|norm| decode_norm(norm, rms_norm_eps))
                    .transpose()?,
                attention_wk: None,
                attention_k_norm: None,
                attention_wv: None,
                attention_v_norm: None,
                interleaved_rope: false,
                bias: None,
            }));

            let attention_wo = source
                .tensor(&format!("{prefix}.attn_output.weight"), device)
                .await?;
            let feed_forward_w1 = source
                .tensor(&format!("{prefix}.ffn_gate.weight"), device)
                .await?;
            let feed_forward_w2 = source
                .tensor(&format!("{prefix}.ffn_down.weight"), device)
                .await?;
            let feed_forward_w3 = source
                .tensor(&format!("{prefix}.ffn_up.weight"), device)
                .await?;
            let attention_norm = source
                .tensor(&format!("{prefix}.attn_norm.weight"), device)
                .await?;
            let post_attention_norm = source
                .tensor(&format!("{prefix}.post_attention_norm.weight"), device)
                .await
                .ok();
            let ffn_norm = source
                .tensor(&format!("{prefix}.ffn_norm.weight"), device)
                .await?;
            let ffn_post_norm = source
                .tensor(&format!("{prefix}.post_ffw_norm.weight"), device)
                .await
                .ok();
            let layer_output_scale = source
                .tensor(&format!("{prefix}.layer_output_scale.weight"), device)
                .await
                .ok()
                .map(&dequantize_1d);

            layers.push(LlamaAttention {
                attention_variant,
                attention_wo: Linear::new(attention_wo, None),
                attention_norm: decode_norm(attention_norm, rms_norm_eps)?,
                post_attention_norm: post_attention_norm
                    .map(|norm| decode_norm(norm, rms_norm_eps))
                    .transpose()?,
                feed_forward_variant: FeedForwardVariant::Llama(Box::new(
                    LlamaFeedForward::new_with_activation(
                        feed_forward_w1,
                        feed_forward_w2,
                        feed_forward_w3,
                        FeedForwardActivation::Gelu,
                    ),
                )),
                ffn_norm: decode_norm(ffn_norm, rms_norm_eps)?,
                post_ffn_norm: ffn_post_norm
                    .map(|norm| decode_norm(norm, rms_norm_eps))
                    .transpose()?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: layer_head_dim,
                hidden_size: layer_attention_width,
                rope_cache,
                sliding_window_size: layer_sliding_window_size,
                attention_scale: 1.0,
                shared_kv_layer: None,
                per_layer_inp_gate: None,
                per_layer_proj: None,
                per_layer_post_norm: None,
                layer_output_scale,
            });
        }

        if pre_projection.shape().get(1).copied() != Some(target_embedding_length * 2) {
            return Err(fusor::Error::msg(format!(
                "unexpected Gemma4 MTP pre_projection input width {:?}, target hidden {target_embedding_length}",
                pre_projection.shape()
            ))
            .into());
        }
        if post_projection.shape().first().copied() != Some(target_embedding_length) {
            return Err(fusor::Error::msg(format!(
                "unexpected Gemma4 MTP post_projection output width {:?}, target hidden {target_embedding_length}",
                post_projection.shape()
            ))
            .into());
        }
        if embedding_length == 0 {
            return Err(fusor::Error::msg("Gemma4 MTP assistant hidden size is zero").into());
        }

        Ok(Self {
            config,
            pre_projection,
            post_projection,
            layers,
            norm,
            output,
            layer_is_sliding,
        })
    }
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> Gemma4MtpAssistant<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    pub(crate) fn draft_step(
        &self,
        target: &Model<F>,
        token: u32,
        h_nextn: &Tensor<2, f32>,
        target_cache: &LlamaCache,
        device: &Device,
        position: usize,
    ) -> Result<Gemma4MtpStep> {
        let token_tensor: Tensor<2, u32> =
            Tensor::from_slice(device, [1, 1], &[token]).to_concrete();
        let mut token_embedding = target.tok_embeddings.forward::<2, 3, _>(&token_tensor);
        if let Some(scale) = target.tok_embedding_scale {
            token_embedding = (token_embedding * scale).to_concrete();
        }
        let h_nextn = h_nextn.unsqueeze(0).to_concrete();
        let projected = fusor::cat([token_embedding, h_nextn], 2)
            .to_concrete()
            .q_mat_mul(&self.pre_projection);
        let mut layer_in: Tensor<3, F> = projected.cast();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let target_kv_layer = self.target_kv_layer(target, layer_idx).ok_or_else(|| {
                fusor::Error::msg("Gemma4 MTP could not find a matching target KV cache layer")
            })?;
            let key = target_cache.blocks[target_kv_layer]
                .k()
                .cloned()
                .ok_or_else(|| fusor::Error::msg("Gemma4 MTP source key cache is empty"))?;
            let value = target_cache.blocks[target_kv_layer]
                .v()
                .cloned()
                .ok_or_else(|| fusor::Error::msg("Gemma4 MTP source value cache is empty"))?;

            let x = layer_in;
            let residual: Tensor<3, f32> = x.cast();
            let x = layer.attention_norm.forward_generic(&x);
            let mut attn = layer.forward_with_shared_kv(&x, None, position, None, &key, &value);
            if let Some(post_attention_norm) = &layer.post_attention_norm {
                attn = post_attention_norm.forward_generic(&attn);
            }
            let attn_f32: Tensor<3, f32> = attn.cast();
            let x = layer.ffn_norm.forward_residual_f32(&attn_f32, &residual);
            let mut x = layer.feed_forward_variant.forward(&x);
            if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                x = post_ffn_norm.forward_generic(&x);
            }
            let x_f32: Tensor<3, f32> = x.cast();
            layer_in = (x_f32 + attn_f32 + residual).cast();
            if let Some(layer_output_scale) = &layer.layer_output_scale {
                let scale = layer_output_scale
                    .reshape([1, 1, 1])
                    .broadcast_as(layer_in.shape())
                    .to_concrete();
                layer_in = (layer_in * scale).to_concrete();
            }
        }

        let normed = self.norm.forward_generic(&layer_in);
        let logits: Tensor<1, f32> = target
            .apply_final_logit_softcap(normed.cast::<f32>().q_mat_mul(&self.output))
            .squeeze(0)
            .squeeze(0)
            .to_concrete();
        let h_nextn: Tensor<2, f32> = normed
            .cast::<f32>()
            .q_mat_mul(&self.post_projection)
            .squeeze(0)
            .to_concrete();
        Ok(Gemma4MtpStep { logits, h_nextn })
    }

    fn target_kv_layer(&self, target: &Model<F>, assistant_layer_idx: usize) -> Option<usize> {
        let wants_sliding = self
            .layer_is_sliding
            .get(assistant_layer_idx)
            .copied()
            .unwrap_or(false);
        target
            .layers
            .iter()
            .enumerate()
            .rev()
            .find(|(_, layer)| layer.sliding_window_size.is_some() == wants_sliding)
            .map(|(idx, layer)| layer.shared_kv_layer.unwrap_or(idx))
    }

    pub(crate) fn draft_n(&self) -> usize {
        self.config.n_layer.max(1)
    }
}
