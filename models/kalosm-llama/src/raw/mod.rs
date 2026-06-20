use std::collections::HashMap;
use std::future::Future;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};
use web_time::{Duration, Instant};

#[cfg(feature = "vision")]
pub(crate) fn debug_check_nan_f32<const R: usize>(
    t: &fusor::Tensor<R, f32>,
    layer: usize,
    label: &str,
    index_pos: usize,
) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (t, layer, label, index_pos);
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if layer != 0 && layer != usize::MAX {
            return;
        }
        let Ok(slice) = pollster::block_on(t.as_slice()) else {
            return;
        };
        let mut nan = 0usize;
        let mut pos_inf = 0usize;
        let mut neg_inf = 0usize;
        let mut max_abs = 0f32;
        let mut sample_idx = 0usize;
        let mut sample_vals = [0usize; 4];
        for (i, v) in slice.as_slice().iter().enumerate() {
            let v = *v;
            if v.is_nan() {
                nan += 1;
                if sample_idx < sample_vals.len() {
                    sample_vals[sample_idx] = i;
                    sample_idx += 1;
                }
            } else if v == f32::INFINITY {
                pos_inf += 1;
            } else if v == f32::NEG_INFINITY {
                neg_inf += 1;
            } else if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }
        if nan > 0 || pos_inf > 0 || neg_inf > 0 {
            tracing::warn!(
                "trace_nan layer={layer} label={label} index_pos={index_pos} shape={:?} nan={nan} (first_nan_indices={:?}) +inf={pos_inf} -inf={neg_inf} max_abs={max_abs}",
                t.shape(),
                &sample_vals[..sample_idx]
            );
        }
    }
}

#[cfg(feature = "vision")]
pub(crate) fn debug_tensor_stats_f32<const R: usize>(t: &fusor::Tensor<R, f32>, label: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (t, label);
        return;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        if std::env::var_os("KALOSM_TRACE_VISION_STATS").is_none() {
            return;
        }
        let Ok(slice) = pollster::block_on(t.as_slice()) else {
            tracing::warn!(
                "[vision_stats] {label} shape={:?} readback failed",
                t.shape()
            );
            return;
        };
        let values = slice.as_slice();
        let mut nan = 0usize;
        let mut pos_inf = 0usize;
        let mut neg_inf = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        for value in values.iter().copied() {
            if value.is_nan() {
                nan += 1;
            } else if value == f32::INFINITY {
                pos_inf += 1;
            } else if value == f32::NEG_INFINITY {
                neg_inf += 1;
            } else {
                min = min.min(value);
                max = max.max(value);
                sum += value as f64;
            }
        }
        let finite = values.len().saturating_sub(nan + pos_inf + neg_inf);
        let mean = if finite == 0 {
            0.0
        } else {
            (sum / finite as f64) as f32
        };
        let sample = values.iter().copied().take(8).collect::<Vec<_>>();
        tracing::info!(
            "[vision_stats] {label} shape={:?} len={} finite={} nan={} +inf={} -inf={} min={} max={} mean={} sample={:?}",
            t.shape(),
            values.len(),
            finite,
            nan,
            pos_inf,
            neg_inf,
            min,
            max,
            mean,
            sample
        );
    }
}

#[cfg(not(feature = "vision"))]
pub(crate) fn debug_check_nan_f32<const R: usize>(
    _: &fusor::Tensor<R, f32>,
    _: usize,
    _: &str,
    _: usize,
) {
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_intermediate_hidden_f32(tensor: &fusor::Tensor<2, f32>) {
    let marker = tensor.clone().mul_scalar(1.0).to_concrete();
    if let Some(gpu_marker) = marker.as_gpu() {
        gpu_marker.materialize_sync();
    } else {
        std::mem::drop(marker.to_concrete());
    }
}

#[cfg(target_arch = "wasm32")]
fn resolve_intermediate_hidden_f32(_: &fusor::Tensor<2, f32>) {}

#[cfg(all(feature = "vision", not(target_arch = "wasm32")))]
fn copy_image_embeddings_to_device(
    tensor: Tensor<2, f32>,
    device: &Device,
) -> Result<Tensor<2, f32>> {
    let shape = tensor.shape();
    let expected_len = shape.iter().product::<usize>();
    let values = pollster::block_on(tensor.as_slice())?;
    let values = values.as_slice();
    if values.len() != expected_len {
        return Err(fusor::Error::msg(format!(
            "Image embedding transfer expected {expected_len} values for shape {shape:?}, got {}",
            values.len()
        )));
    }
    Ok(Tensor::from_slice(device, shape, values))
}

#[cfg(all(feature = "vision", target_arch = "wasm32"))]
fn copy_image_embeddings_to_device(
    tensor: Tensor<2, f32>,
    _device: &Device,
) -> Result<Tensor<2, f32>> {
    Ok(tensor)
}

use crate::chat_template::HuggingFaceChatTemplate;
use crate::raw::attention_layer::LlamaAttention;
use crate::raw::rope::RopeImplementation;
use crate::LlamaSourceError;
use attention_layer::AttentionBias;
use attention_layer::AttentionVariant;
use attention_layer::FeedForwardActivation;
use attention_layer::FeedForwardVariant;
use attention_layer::GroupedAttention;
use attention_layer::LlamaFeedForward;
use attention_layer::PhiFeedForward;
use attention_layer::SeparateAttention;
use fusor::cache::{AttentionMask, MaskCache};
use fusor::layers::Embedding;
use fusor::layers::Linear;
use fusor::layers::RmsNorm;
use fusor::QMatrix;
use fusor::ShardedVarBuilder;
use fusor::{
    AddOp, CastTensor, CastTo, FloatDataType, FloatOps, Fusion, MatmulImpl, MulOp, SimdBinaryOp,
    SimdElement, SimdReduceOp, SumOp,
};
use fusor::{AsyncReadRange, AsyncShardedVarBuilder};
use fusor::{Device, Result, Tensor};
use fusor_gguf::GgufMetadata;
use fusor_gguf::GgufValue;

mod attention_layer;
pub mod cache;
mod mtp;
mod rope;
#[cfg(feature = "vision")]
mod vision;

use crate::LlamaImage;
use cache::LlamaCache;
pub(crate) use mtp::Gemma4MtpAssistant;

pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const GEMMA_DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

/// Build the additive attention-mask values (`0.0` allowed, `-inf` blocked)
/// for a `[seq_len, index_pos + seq_len]` score matrix.
///
/// Tokens are causal by default. Any query/key position that falls inside the
/// same entry of `non_causal_token_ranges` may attend to its peers regardless
/// of ordering (this is how image-token blocks attend bidirectionally), and an
/// optional sliding `window` blocks keys older than `window` positions.
fn non_causal_mask_data(
    seq_len: usize,
    index_pos: usize,
    sliding_window_size: Option<usize>,
    non_causal_token_ranges: &[Range<usize>],
) -> Vec<f32> {
    let cols = index_pos + seq_len;
    let mut mask_data = vec![0.0_f32; seq_len * cols];
    for row in 0..seq_len {
        let global_row = index_pos + row;
        for col in 0..cols {
            let same_non_causal_range = col >= index_pos
                && non_causal_token_ranges
                    .iter()
                    .any(|range| range.contains(&row) && range.contains(&(col - index_pos)));
            let future = col > global_row && !same_non_causal_range;
            let outside_window = sliding_window_size
                .map(|window| col + window <= global_row)
                .unwrap_or(false);
            if future || outside_window {
                mask_data[row * cols + col] = f32::NEG_INFINITY;
            }
        }
    }
    mask_data
}

/// The configuration of a Llama model.
pub struct LlamaConfig<F: FloatDataType + SimdElement = f32> {
    pub(crate) rope_freq_weight: Option<Tensor<1, F>>,
    pub(crate) rope_theta: f32,
    pub(crate) context_length: usize,
    pub(crate) head_dimension: usize,
    pub(crate) n_layer: usize,
    pub(crate) start_token_string: String,
    pub(crate) stop_tokens: Vec<u32>,
    pub(crate) stop_token_string: String,
    pub(crate) chat_template: Option<HuggingFaceChatTemplate>,
    pub(crate) rope_scaling: Option<RopeScalingConfig>,
    pub(crate) sliding_window_type: Option<usize>,
    pub(crate) sliding_window_size: Option<usize>,
    pub(crate) layer_sliding_window_sizes: Option<Vec<Option<usize>>>,
    pub(crate) final_logit_softcapping: Option<f32>,
    pub(crate) per_layer_embedding_length: Option<usize>,
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) vision_start_token: Option<u32>,
    pub(crate) _vision_end_token: Option<u32>,
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) image_pad_token: Option<u32>,
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) image_start_token: Option<u32>,
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) image_end_token: Option<u32>,
    #[cfg_attr(not(feature = "vision"), allow(dead_code))]
    pub(crate) video_pad_token: Option<u32>,
    pub(crate) mrope_sections: Option<Vec<usize>>,
}

impl<F: FloatDataType + SimdElement> LlamaConfig<F> {
    #[cfg(test)]
    pub(crate) fn mock_test() -> Self {
        Self {
            rope_freq_weight: None,
            rope_theta: 5000.,
            context_length: 6,
            head_dimension: 2,
            n_layer: 0,
            start_token_string: "<|startoftext|>".to_string(),
            stop_tokens: vec![0],
            stop_token_string: "<|endoftext|>".to_string(),
            sliding_window_type: None,
            sliding_window_size: None,
            layer_sliding_window_sizes: None,
            final_logit_softcapping: None,
            per_layer_embedding_length: None,
            chat_template: None,
            rope_scaling: None,
            vision_start_token: None,
            _vision_end_token: None,
            image_pad_token: None,
            image_start_token: None,
            image_end_token: None,
            video_pad_token: None,
            mrope_sections: None,
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeScalingConfig {
    pub(crate) factor: f32,
    pub(crate) high_freq_factor: f32,
    pub(crate) low_freq_factor: f32,
    pub(crate) original_max_position_embeddings: usize,
}

pub struct Model<F: FloatDataType + SimdElement = f32> {
    pub(crate) config: Arc<LlamaConfig<F>>,
    #[cfg(feature = "vision")]
    vision_encoder: Option<vision::VisionTransformer<F>>,
    tok_embeddings: Embedding<f32>,
    tok_embedding_scale: Option<f32>,
    per_layer_tok_embeddings: Option<Embedding<f32>>,
    per_layer_model_proj: Option<QMatrix>,
    per_layer_proj_norm: Option<RmsNorm<1, F>>,
    layers: Vec<LlamaAttention<F>>,
    norm: RmsNorm<1, F>,
    output: QMatrix,
    /// Mask cache always uses f32 for SIMD compatibility
    masks: MaskCache<f32>,
}

pub(crate) struct TargetBatchOutput {
    pub(crate) logits: Tensor<2, f32>,
    pub(crate) h_nextn: Tensor<2, f32>,
}

struct PreNormForwardOutput<F: FloatDataType + SimdElement> {
    hidden: Tensor<3, F>,
    seq_len: usize,
}

/// The embedded token inputs produced by [`Model::encode_tokens`], ready to be
/// run through the transformer layers.
pub(crate) struct EncodedTokens<F: FloatDataType + SimdElement> {
    embeddings: Tensor<3, F>,
    per_layer_inputs: Option<Tensor<4, F>>,
    seq_len: usize,
    index_pos: usize,
    pos_ids: Option<Tensor<2, F>>,
    non_causal_token_ranges: Vec<Range<usize>>,
}

pub(crate) trait LlamaVarSource {
    fn get(&self, name: &str) -> Result<&GgufValue>;

    fn tensor<'a>(
        &'a mut self,
        name: &'a str,
        device: &'a Device,
    ) -> Pin<Box<dyn Future<Output = Result<QMatrix>> + 'a>>;
}

impl<R: std::io::Read + std::io::Seek> LlamaVarSource for ShardedVarBuilder<R> {
    fn get(&self, name: &str) -> Result<&GgufValue> {
        ShardedVarBuilder::get(self, name)
    }

    fn tensor<'a>(
        &'a mut self,
        name: &'a str,
        device: &'a Device,
    ) -> Pin<Box<dyn Future<Output = Result<QMatrix>> + 'a>> {
        Box::pin(std::future::ready(ShardedVarBuilder::tensor(
            self, name, device,
        )))
    }
}

impl<R: AsyncReadRange> LlamaVarSource for AsyncShardedVarBuilder<R> {
    fn get(&self, name: &str) -> Result<&GgufValue> {
        AsyncShardedVarBuilder::get(self, name)
    }

    fn tensor<'a>(
        &'a mut self,
        name: &'a str,
        device: &'a Device,
    ) -> Pin<Box<dyn Future<Output = Result<QMatrix>> + 'a>> {
        Box::pin(AsyncShardedVarBuilder::tensor(self, name, device))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn block_on_ready<F: Future>(future: F) -> F::Output {
    fn clone(_: *const ()) -> RawWaker {
        noop_raw_waker()
    }

    fn wake(_: *const ()) {}

    fn noop_raw_waker() -> RawWaker {
        RawWaker::new(
            std::ptr::null(),
            &RawWakerVTable::new(clone, wake, wake, wake),
        )
    }

    let waker = unsafe { Waker::from_raw(noop_raw_waker()) };
    let mut cx = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    match future.as_mut().poll(&mut cx) {
        Poll::Ready(output) => output,
        Poll::Pending => panic!("synchronous GGUF model loading unexpectedly yielded"),
    }
}

impl<F: FloatDataType + SimdElement + FloatOps + MatmulImpl> Model<F>
where
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        source: &mut ShardedVarBuilder<R>,
        vision_ct: Option<GgufMetadata>,
        vision_bytes: Option<Vec<u8>>,
        device: &Device,
        override_stop_token_string: Option<String>,
        override_chat_template: Option<String>,
        rope_scaling: Option<RopeScalingConfig>,
    ) -> std::result::Result<Self, LlamaSourceError>
    where
        f32: CastTensor<F> + CastTo<F>,
        F: CastTensor<f32> + CastTo<f32>,
    {
        block_on_ready(Self::from_var_source(
            source,
            vision_ct,
            vision_bytes,
            device,
            override_stop_token_string,
            override_chat_template,
            rope_scaling,
        ))
    }

    pub(crate) async fn from_var_source<S: LlamaVarSource>(
        source: &mut S,
        vision_ct: Option<GgufMetadata>,
        vision_bytes: Option<Vec<u8>>,
        device: &Device,
        override_stop_token_string: Option<String>,
        override_chat_template: Option<String>,
        rope_scaling: Option<RopeScalingConfig>,
    ) -> std::result::Result<Self, LlamaSourceError>
    where
        f32: CastTensor<F> + CastTo<F>,
        F: CastTensor<f32> + CastTo<f32>,
    {
        #[cfg(not(feature = "vision"))]
        let _ = (vision_ct, vision_bytes);

        // Helper to dequantize a QMatrix to 1D tensor
        // VarBuilder preserves original shapes, so 1D tensors stay 1D
        let dequantize_1d = |qmatrix: QMatrix| -> Tensor<1, F> {
            let shape = qmatrix.shape();
            if shape.len() == 1 {
                // Already 1D, dequantize directly
                let w1d: Tensor<1, f32> = qmatrix.dequantize();
                w1d.cast()
            } else if shape.len() == 2 {
                // 2D tensor, reshape to 1D (for backwards compatibility)
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

        // Get the eos and bos tokens from the metadata
        let tokens: Box<[GgufValue]> = source.get("tokenizer.ggml.tokens")?.clone().try_into()?;
        let tokens: Result<Vec<Box<str>>, LlamaSourceError> = tokens
            .iter()
            .map(|v| {
                let v: Box<str> = v.try_into()?;
                Ok(v)
            })
            .collect();
        let tokens = tokens?;
        let start_token: Option<u32> = source
            .get("tokenizer.ggml.bos_token_id")
            .ok()
            .and_then(|v| v.try_into().ok());
        let eos_token: u32 = source
            .get("tokenizer.ggml.eos_token_id")?
            .clone()
            .try_into()?;
        let stop_token = if let Some(override_stop_token_string) = override_stop_token_string {
            tokens
                .iter()
                .position(|v| **v == override_stop_token_string)
                .unwrap_or(0) as u32
        } else {
            eos_token
        };
        let mut stop_tokens = vec![stop_token];
        if eos_token != stop_token {
            stop_tokens.push(eos_token);
        }
        let start_token_string = start_token
            .map(|v| tokens[v as usize].to_string())
            .unwrap_or_default();
        let stop_token_string = tokens[stop_token as usize].to_string();
        let chat_template = override_chat_template.or_else(|| {
            source
                .get("tokenizer.chat_template")
                .ok()
                .and_then(|v| v.to_string().ok())
                .map(|s| s.to_string())
        });
        let chat_template = match chat_template {
            Some(chat_template) => {
                let chat_template = HuggingFaceChatTemplate::create(chat_template)
                    .map_err(LlamaSourceError::ChatTemplate)?;
                Some(chat_template)
            }
            None => None,
        };

        // Parameter extraction from metadata.
        let architecture = source.get("general.architecture")?.to_string()?.clone();
        let is_gemma4 = architecture.as_ref() == "gemma4";
        let head_count = source.get(".attention.head_count")?.to_u32()? as usize;
        let head_count_kv = source.get(".attention.head_count_kv")?.to_u32()? as usize;
        let block_count = source.get(".block_count")?.to_u32()? as usize;
        let embedding_length = source.get(".embedding_length")?.to_u32()? as usize;
        let per_layer_embedding_length = source
            .get(".embedding_length_per_layer_input")
            .and_then(|m| Ok(m.to_u32()?))
            .ok()
            .map(|x| x as usize);
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = source.get(".attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let final_logit_softcapping = source
            .get(".final_logit_softcapping")
            .and_then(|m| Ok(m.to_f32()?))
            .ok();

        let rope_freq_base = source
            .get(".rope.freq_base")
            .and_then(|m| Ok(m.to_f32()?))
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);
        let sliding_window_size = source
            .get(".attention.sliding_window")
            .and_then(|m| Ok(m.to_u32()?))
            .ok()
            .map(|x| x as usize);
        let sliding_window_type = source
            .get(".attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()?))
            .ok()
            .map(|x| x as usize)
            .or_else(|| (&*architecture == "gemma3").then_some(GEMMA_DEFAULT_SLIDING_WINDOW_TYPE));

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
            .or_else(|| {
                (&*architecture == "gemma3" || is_gemma4)
                    .then_some(GEMMA_DEFAULT_ROPE_FREQUENCY_SLIDING)
            });

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
        let layer_is_sliding: Vec<bool> = if let Some(pattern) = sliding_window_pattern {
            pattern
        } else if let Some(sliding_window_type) = sliding_window_type {
            (0..block_count)
                .map(|layer_idx| (layer_idx + 1) % sliding_window_type != 0)
                .collect()
        } else {
            vec![false; block_count]
        };
        let layer_sliding_window_sizes = sliding_window_size.map(|sliding_window_size| {
            layer_is_sliding
                .iter()
                .map(|is_sliding| is_sliding.then_some(sliding_window_size))
                .collect::<Vec<_>>()
        });

        let shared_kv_layers = source
            .get(".attention.shared_kv_layers")
            .and_then(|m| Ok(m.to_u32()?))
            .ok()
            .map(|x| x as usize)
            .unwrap_or_default();
        let n_layer_kv_from_start = block_count.saturating_sub(shared_kv_layers);

        let head_dim = source
            .get(".attention.key_length")
            .and_then(|v| Ok(v.to_u32()?))
            .ok()
            .map(|x| x as usize)
            .unwrap_or_else(|| embedding_length / head_count);
        let head_dim_swa = source
            .get(".attention.key_length_swa")
            .and_then(|v| Ok(v.to_u32()?))
            .ok()
            .map(|x| x as usize)
            .unwrap_or(head_dim);

        let context_length = source.get(".context_length")?.to_u32()? as usize;

        let rope_freq_weight: Option<Tensor<1, F>> = source
            .tensor("rope_freqs.weight", device)
            .await
            .ok()
            .map(&dequantize_1d);

        let config = LlamaConfig {
            rope_freq_weight,
            rope_theta: rope_freq_base,
            context_length,
            head_dimension: head_dim,
            n_layer: block_count,
            start_token_string,
            stop_tokens,
            stop_token_string,
            chat_template,
            rope_scaling,
            sliding_window_type,
            sliding_window_size,
            layer_sliding_window_sizes,
            final_logit_softcapping,
            per_layer_embedding_length,
            vision_start_token: tokens
                .iter()
                .position(|v| &**v == "<|vision_start|>")
                .map(|v| v as u32),
            _vision_end_token: tokens
                .iter()
                .position(|v| &**v == "<|vision_end|>")
                .map(|v| v as u32),
            image_pad_token: tokens
                .iter()
                .position(|v| &**v == "<|image_pad|>")
                .or_else(|| tokens.iter().position(|v| &**v == "<|image|>"))
                .map(|v| v as u32),
            image_start_token: tokens
                .iter()
                .position(|v| &**v == "<|image>")
                .map(|v| v as u32),
            image_end_token: tokens
                .iter()
                .position(|v| &**v == "<image|>")
                .map(|v| v as u32),
            video_pad_token: tokens
                .iter()
                .position(|v| &**v == "<|video_pad|>")
                .map(|v| v as u32),
            mrope_sections: source
                .get(".rope.dimension_sections")
                .ok()
                .and_then(|m| {
                    m.to_array()
                        .ok()
                        .map(|v| v.iter().map(|x| x.to_i32().map(|x| x as usize)).collect())
                })
                .transpose()?,
        };
        let config = Arc::new(config);

        let rope: RopeImplementation<F> = rope::RopeImplementation::new_with_head_dimension(
            &config,
            head_dim,
            config.rope_freq_weight.as_ref(),
            config.rope_theta,
            device,
        )?;
        let sliding_rope: Option<RopeImplementation<F>> = rope_freq_base_sliding
            .filter(|_| layer_is_sliding.iter().any(|is_sliding| *is_sliding))
            .map(|rope_freq_base_sliding| {
                let rope_freq_weight = (!is_gemma4)
                    .then_some(config.rope_freq_weight.as_ref())
                    .flatten();
                RopeImplementation::new_with_head_dimension(
                    &config,
                    head_dim_swa,
                    rope_freq_weight,
                    rope_freq_base_sliding,
                    device,
                )
            })
            .transpose()?;

        let tok_embeddings_q = source.tensor("token_embd.weight", device).await?;
        let tok_embedding_scale =
            (&*architecture == "gemma3" || is_gemma4).then(|| (embedding_length as f32).sqrt());
        let tok_embeddings = Embedding::new(tok_embeddings_q.clone());

        let (per_layer_tok_embeddings, per_layer_model_proj, per_layer_proj_norm) =
            if per_layer_embedding_length.is_some() {
                let embeddings = source.tensor("per_layer_token_embd.weight", device).await?;
                let model_proj = source.tensor("per_layer_model_proj.weight", device).await?;
                let proj_norm = source.tensor("per_layer_proj_norm.weight", device).await?;
                (
                    Some(Embedding::new(embeddings)),
                    Some(model_proj),
                    Some(decode_norm(proj_norm, rms_norm_eps)?),
                )
            } else {
                (None, None, None)
            };

        let norm = source.tensor("output_norm.weight", device).await?;
        let norm = decode_norm(norm, rms_norm_eps)?;
        let output = match source.tensor("output.weight", device).await {
            Ok(output) => output,
            Err(_) => {
                // If there is no output layer, assume the word embeddings are tied to the output
                tok_embeddings_q.clone()
            }
        };

        let mut layers = Vec::with_capacity(block_count);
        let interleaved_rope = architecture.as_ref() != "qwen2"
            && architecture.as_ref() != "qwen3"
            && architecture.as_ref() != "gemma3"
            && !is_gemma4;

        for layer_idx in 0..block_count {
            let layer_is_sliding = layer_is_sliding.get(layer_idx).copied().unwrap_or(false);
            let layer_head_dim = if is_gemma4 && layer_is_sliding {
                head_dim_swa
            } else {
                head_dim
            };
            let layer_attention_width = head_count * layer_head_dim;
            let layer_sliding_window_size = config
                .layer_sliding_window_sizes
                .as_ref()
                .and_then(|sizes| sizes.get(layer_idx).copied().flatten());

            let has_kv = !is_gemma4 || layer_idx < n_layer_kv_from_start;
            let shared_kv_layer = if is_gemma4 && !has_kv {
                let offset = if layer_is_sliding { 2 } else { 1 };
                Some(n_layer_kv_from_start.saturating_sub(offset))
            } else {
                None
            };

            let rope_cache = if layer_is_sliding {
                sliding_rope
                    .as_ref()
                    .cloned()
                    .unwrap_or_else(|| rope.clone())
            } else {
                rope.clone()
            };

            let prefix = format!("blk.{layer_idx}");
            let attention_variant = if let Ok(attention_qkv) = source
                .tensor(&format!("{prefix}.attn_qkv.weight"), device)
                .await
            {
                AttentionVariant::Grouped(GroupedAttention {
                    attention_qkv,
                    interleaved_rope,
                })
            } else {
                let q = source
                    .tensor(&format!("{prefix}.attn_q.weight"), device)
                    .await?;
                let k = if has_kv {
                    Some(
                        source
                            .tensor(&format!("{prefix}.attn_k.weight"), device)
                            .await?,
                    )
                } else {
                    None
                };
                let v = if has_kv {
                    Some(
                        source
                            .tensor(&format!("{prefix}.attn_v.weight"), device)
                            .await?,
                    )
                } else {
                    None
                };
                let qkv = if let (Some(k), Some(v)) = (&k, &v) {
                    QMatrix::concat_rows(&[&q, k, v])
                } else {
                    None
                };
                let bias_q = source
                    .tensor(&format!("{prefix}.attn_q.bias"), device)
                    .await;
                let bias_k = source
                    .tensor(&format!("{prefix}.attn_k.bias"), device)
                    .await;
                let bias_v = source
                    .tensor(&format!("{prefix}.attn_v.bias"), device)
                    .await;
                let bias = if let (Ok(bias_q), Ok(bias_k), Ok(bias_v)) = (bias_q, bias_k, bias_v) {
                    Some(AttentionBias::new(
                        dequantize_1d(bias_q),
                        dequantize_1d(bias_k),
                        dequantize_1d(bias_v),
                    ))
                } else {
                    None
                };
                let q_norm = source
                    .tensor(&format!("{prefix}.attn_q_norm.weight"), device)
                    .await
                    .ok();
                let k_norm = source
                    .tensor(&format!("{prefix}.attn_k_norm.weight"), device)
                    .await
                    .ok();
                let attention_v_norm = if is_gemma4 && has_kv {
                    Some(RmsNorm::new(
                        Tensor::ones(device, [layer_head_dim]),
                        None,
                        rms_norm_eps as f32,
                    ))
                } else {
                    None
                };
                let separate = SeparateAttention {
                    attention_wq: q,
                    attention_qkv: qkv,
                    attention_q_norm: q_norm
                        .map(|norm| decode_norm(norm, rms_norm_eps))
                        .transpose()?,
                    attention_wk: k,
                    attention_k_norm: k_norm
                        .map(|norm| decode_norm(norm, rms_norm_eps))
                        .transpose()?,
                    attention_wv: v,
                    attention_v_norm,
                    interleaved_rope,
                    bias,
                };
                AttentionVariant::Separate(Box::new(separate))
            };
            let attention_wo = source
                .tensor(&format!("{prefix}.attn_output.weight"), device)
                .await?;
            // Try to read from the up, down and gate weights
            let feed_forward_variant = if let Ok(ffn_gate) = source
                .tensor(&format!("{prefix}.ffn_gate.weight"), device)
                .await
            {
                let feed_forward_w1 = ffn_gate;
                let feed_forward_w2 = source
                    .tensor(&format!("{prefix}.ffn_down.weight"), device)
                    .await?;
                let feed_forward_w3 = source
                    .tensor(&format!("{prefix}.ffn_up.weight"), device)
                    .await?;
                let activation = if is_gemma4 {
                    FeedForwardActivation::Gelu
                } else {
                    FeedForwardActivation::Silu
                };
                FeedForwardVariant::Llama(Box::new(LlamaFeedForward::new_with_activation(
                    feed_forward_w1,
                    feed_forward_w2,
                    feed_forward_w3,
                    activation,
                )))
            } else {
                // Otherwise, try to read from the up, and down weights
                let up = source
                    .tensor(&format!("{prefix}.ffn_up.weight"), device)
                    .await?;
                // Transpose the down tensor
                let down = source
                    .tensor(&format!("{prefix}.ffn_down.weight"), device)
                    .await?;
                let feed_forward_length = source.get(".feed_forward_length")?.to_u32()? as usize;

                FeedForwardVariant::Phi(PhiFeedForward {
                    up,
                    down,
                    feed_forward_length,
                })
            };
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

            let per_layer_inp_gate = if per_layer_embedding_length.is_some() {
                Some(
                    source
                        .tensor(&format!("{prefix}.inp_gate.weight"), device)
                        .await?,
                )
            } else {
                None
            };
            let per_layer_proj = if per_layer_embedding_length.is_some() {
                Some(
                    source
                        .tensor(&format!("{prefix}.proj.weight"), device)
                        .await?,
                )
            } else {
                None
            };
            let per_layer_post_norm = if per_layer_embedding_length.is_some() {
                let norm = source
                    .tensor(&format!("{prefix}.post_norm.weight"), device)
                    .await?;
                Some(decode_norm(norm, rms_norm_eps)?)
            } else {
                None
            };
            let layer_output_scale = source
                .tensor(&format!("{prefix}.layer_output_scale.weight"), device)
                .await
                .ok()
                .map(&dequantize_1d);
            // Gemma 4 folds the query pre-attention scaling into the exported
            // `attn_q_norm` weights, so the softmax logits are already scaled and
            // flash-attention must run with a unit scale. Every other supported
            // architecture applies the usual 1/sqrt(head_dim) here.
            let attention_scale = if is_gemma4 {
                1.0
            } else {
                1.0 / (layer_head_dim as f32).sqrt()
            };

            layers.push(LlamaAttention {
                attention_variant,
                attention_wo: Linear::new(attention_wo, None),
                attention_norm: decode_norm(attention_norm, rms_norm_eps)?,
                post_attention_norm: post_attention_norm
                    .map(|norm| decode_norm(norm, rms_norm_eps))
                    .transpose()?,
                feed_forward_variant,
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
                attention_scale,
                shared_kv_layer,
                per_layer_inp_gate,
                per_layer_proj,
                per_layer_post_norm,
                layer_output_scale,
            })
        }

        // If the model is a vision model, load the vision encoder
        #[cfg(feature = "vision")]
        let vision_encoder =
            if let (Some(vision_ct), Some(vision_bytes)) = (vision_ct, vision_bytes) {
                Some(vision::VisionTransformer::from_gguf(
                    vision_ct,
                    &vision_bytes,
                    device,
                )?)
            } else {
                None
            };
        Ok(Self {
            config,
            tok_embeddings,
            tok_embedding_scale,
            per_layer_tok_embeddings,
            per_layer_model_proj,
            per_layer_proj_norm,
            layers,
            norm,
            output,
            masks: Default::default(),
            #[cfg(feature = "vision")]
            vision_encoder,
        })
    }
}

impl<F: FloatDataType + SimdElement + Default + FloatOps + MatmulImpl> Model<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    pub(crate) fn supports_gpu_token_run_ahead(&self) -> bool {
        #[cfg(feature = "vision")]
        {
            self.vision_encoder.is_none()
                || matches!(
                    self.vision_encoder,
                    Some(vision::VisionTransformer::Gemma(_))
                )
        }

        #[cfg(not(feature = "vision"))]
        {
            true
        }
    }

    /// Compute the Gemma "per-layer input" embeddings that are blended into each
    /// decoder layer (the `inp_gate`/`proj`/`post_norm` path). Returns `None`
    /// for models without per-layer embeddings.
    ///
    /// `per_layer_token_ids` is invoked lazily, so models without per-layer
    /// embeddings never pay for building the token id tensor. It yields the
    /// `[batch, positions]` ids used for the per-layer token lookup, with
    /// image/control tokens already zeroed by the caller. A single position is
    /// broadcast across the whole sequence (image chunks share one zeroed
    /// per-layer token).
    fn compute_per_layer_inputs<B>(
        &self,
        embeddings_f32: &Tensor<3, f32>,
        per_layer_token_ids: impl FnOnce() -> Tensor<2, u32, B>,
    ) -> Option<Tensor<4, F>>
    where
        B: Fusion<2, u32>,
    {
        let (
            per_layer_tok_embeddings,
            per_layer_model_proj,
            per_layer_proj_norm,
            per_layer_embedding_length,
        ) = match (
            &self.per_layer_tok_embeddings,
            &self.per_layer_model_proj,
            &self.per_layer_proj_norm,
            self.config.per_layer_embedding_length,
        ) {
            (Some(embeddings), Some(model_proj), Some(proj_norm), Some(length)) => {
                (embeddings, model_proj, proj_norm, length)
            }
            _ => return None,
        };

        let [batch, seq, embedding_dim] = embeddings_f32.shape();
        let n_layer = self.config.n_layer;
        let per_layer_token_ids = per_layer_token_ids();
        let positions = per_layer_token_ids.shape()[1];

        let token_inputs = per_layer_tok_embeddings.forward::<2, 3, _>(&per_layer_token_ids)
            * (per_layer_embedding_length as f32).sqrt();
        let token_inputs =
            token_inputs.reshape([batch, positions, n_layer, per_layer_embedding_length]);
        let token_inputs: Tensor<4, f32> = if positions == seq {
            token_inputs.to_concrete()
        } else {
            token_inputs
                .broadcast_as([batch, seq, n_layer, per_layer_embedding_length])
                .to_concrete()
        };

        let projected_inputs = embeddings_f32.q_mat_mul(per_layer_model_proj)
            * (1.0 / (embedding_dim as f32).sqrt());
        let projected_inputs = projected_inputs
            .reshape([batch, seq, n_layer, per_layer_embedding_length])
            .to_concrete();
        let projected_inputs: Tensor<4, F> =
            per_layer_proj_norm.forward_generic_4d(&projected_inputs.cast());

        Some(
            ((projected_inputs.cast::<f32>() + token_inputs) * (1.0 / 2.0_f32.sqrt()))
                .to_concrete()
                .cast(),
        )
    }

    pub fn encode_tokens(
        &self,
        raw_tokens: &[u32],
        raw_images: &[LlamaImage],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<EncodedTokens<F>> {
        #[cfg(feature = "vision")]
        let (tokens, images, grid_thw, image_token_ranges) = {
            let mut grid_thw = Vec::new();
            let mut images = Vec::new();
            let mut image_token_ranges = Vec::new();
            // Embed all images
            if let Some(vision_encoder) = &self.vision_encoder {
                for (image, hints) in raw_images {
                    let min_pixels = hints.min_tokens();
                    let max_pixels = hints.max_tokens();
                    let (image, thw) =
                        vision_encoder.preprocess_image(image, min_pixels, max_pixels)?;
                    images.push(image);
                    grid_thw.push(thw)
                }
            } else if !raw_images.is_empty() {
                return Err(fusor::Error::msg(
                    "Media inputs require a loaded vision encoder.",
                ));
            }

            // Add image padding tokens for any placeholders in the prompt.
            let tokens = if let (Some(image_pad_token), Some(vision)) =
                (self.config.image_pad_token, &self.vision_encoder)
            {
                let (tokens, ranges) = vision.expand_image_tokens(
                    raw_tokens,
                    image_pad_token,
                    self.config.vision_start_token,
                    self.config.image_start_token,
                    self.config.image_end_token,
                    &grid_thw,
                )?;
                image_token_ranges = ranges;
                tokens
            } else {
                raw_tokens.to_vec()
            };

            (tokens, images, grid_thw, image_token_ranges)
        };
        #[cfg(not(feature = "vision"))]
        let tokens = {
            let _ = raw_images;
            raw_tokens.to_vec()
        };

        let mut seq_len = tokens.len();
        let cached_tokens = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
        // We use a lower cutoff than the context length to avoid recomputing the attention every single token
        let cutoff_len: usize = self.config.context_length.saturating_sub(32).max(8);
        let (tokens, index_pos, start_time) = if seq_len + cached_tokens
            > self.config.context_length
        {
            let all_tokens = if let Some(cache) = cache.as_mut() {
                cache.clear();
                let mut all_tokens = cache.tokens.clone();
                all_tokens.extend(tokens);
                all_tokens
            } else {
                tokens.to_vec()
            };
            let start = all_tokens.len() - cutoff_len;
            seq_len = cutoff_len;
            tracing::trace!(
                "The context is full, trimming start of the context to fit new tokens. The first {} tokens were truncated.",
                start
            );
            let all_tokens = &all_tokens[start..];
            if let Some(cache) = cache.as_mut() {
                cache.tokens = all_tokens.to_vec();
            }
            assert!(all_tokens.len() <= self.config.context_length);
            (all_tokens.to_vec(), 0, 0)
        } else {
            let index_pos = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
            let start_time = cache.as_ref().map(|c| c.start_time).unwrap_or_default();
            if let Some(cache) = cache.as_mut() {
                cache.tokens.extend_from_slice(&tokens);
            }
            (tokens, index_pos, start_time)
        };
        #[cfg(not(feature = "vision"))]
        let _ = start_time;
        let x_base = Tensor::new(device, tokens.as_slice());
        let x = x_base.unsqueeze(0);

        let mut embeddings_f32 = self.tok_embeddings.forward(&x);
        if let Some(scale) = self.tok_embedding_scale {
            embeddings_f32 = (embeddings_f32 * scale).to_concrete();
        }
        #[cfg(feature = "vision")]
        let mut pos_ids = None;
        #[cfg(not(feature = "vision"))]
        let pos_ids = None;

        #[cfg(feature = "vision")]
        if let Some(vision_encoder) = &self.vision_encoder {
            let batch_size = embeddings_f32.shape()[0];
            let embed_dim = embeddings_f32.shape()[2];
            for ((pixels, grid), range) in
                images.iter().zip(&grid_thw).zip(image_token_ranges.iter())
            {
                let pixels_f: Tensor<2, F> = pixels.cast();
                let image_embeds = vision_encoder.forward_image(&pixels_f, *grid)?;
                let mut image_embeds_f32: Tensor<2, f32> = image_embeds.cast();
                if vision_encoder.outputs_on_isolated_device() {
                    image_embeds_f32 = copy_image_embeddings_to_device(image_embeds_f32, device)?;
                }
                debug_tensor_stats_f32(&image_embeds_f32, "image_embeds_projected");
                let image_embeds_3d: Tensor<3, f32> = image_embeds_f32.unsqueeze(0).to_concrete();
                embeddings_f32 = embeddings_f32.slice_assign(
                    [0..batch_size, range.clone(), 0..embed_dim],
                    &image_embeds_3d,
                );
            }
            if let Some((new_pos_ids, new_start_time)) =
                vision_encoder.get_rope_index(&tokens, &grid_thw, &self.config, start_time)?
            {
                if let Some(cache) = cache.as_mut() {
                    cache.start_time = new_start_time;
                }
                let pos_f32: Tensor<2, f32> = new_pos_ids.cast();
                let pos_f: Tensor<2, F> = pos_f32.cast();
                pos_ids = Some(pos_f);
            }
        }

        let per_layer_inputs = self.compute_per_layer_inputs(&embeddings_f32, || {
            #[cfg(feature = "vision")]
            {
                let mut per_layer_tokens = tokens.clone();
                for range in &image_token_ranges {
                    per_layer_tokens[range.clone()].fill(0);
                }
                if let Some(image_start_token) = self.config.image_start_token {
                    for token in &mut per_layer_tokens {
                        if *token == image_start_token {
                            *token = 0;
                        }
                    }
                }
                if let Some(image_end_token) = self.config.image_end_token {
                    for token in &mut per_layer_tokens {
                        if *token == image_end_token {
                            *token = 0;
                        }
                    }
                }
                Tensor::from_slice(device, [1, per_layer_tokens.len()], per_layer_tokens.as_slice())
            }
            #[cfg(not(feature = "vision"))]
            {
                x.clone()
            }
        });
        let embeddings: Tensor<3, F> = embeddings_f32.cast();

        Ok(EncodedTokens {
            embeddings,
            per_layer_inputs,
            seq_len,
            index_pos,
            pos_ids,
            #[cfg(feature = "vision")]
            non_causal_token_ranges: image_token_ranges,
            #[cfg(not(feature = "vision"))]
            non_causal_token_ranges: Vec::new(),
        })
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        images: &[LlamaImage],
        device: &Device,
        cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2, F>>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        let x_f32 = self.forward_last_hidden_f32(tokens, images, device, cache)?;
        self.forward_logits_from_hidden_f32(x_f32)
    }

    fn forward_logits_from_hidden_f32(&self, x_f32: Tensor<2, f32>) -> Result<Tensor<2, F>>
    where
        f32: CastTo<F> + CastTensor<F>,
    {
        Ok(self.logits_from_hidden_f32(x_f32).cast())
    }

    pub(crate) fn logits_from_hidden_f32<const R: usize>(
        &self,
        x_f32: Tensor<R, f32>,
    ) -> Tensor<R, f32> {
        self.apply_final_logit_softcap(x_f32.q_mat_mul(&self.output))
    }

    pub(crate) fn apply_final_logit_softcap<const R: usize>(
        &self,
        logits: Tensor<R, f32>,
    ) -> Tensor<R, f32> {
        if let Some(softcap) = self.config.final_logit_softcapping {
            logits.mul_scalar(1.0 / softcap).tanh().mul_scalar(softcap)
        } else {
            logits
        }
    }

    pub(crate) fn should_chunk_multimodal_prompt(&self) -> bool {
        if std::env::var_os("KALOSM_LLAMA_DISABLE_MULTIMODAL_CHUNK").is_some() {
            return false;
        }
        if std::env::var_os("KALOSM_LLAMA_FORCE_MULTIMODAL_CHUNK").is_some() {
            return self.config.image_pad_token.is_some();
        }
        #[cfg(feature = "vision")]
        {
            matches!(
                self.vision_encoder,
                Some(vision::VisionTransformer::Gemma(_))
            ) && self.config.image_pad_token.is_some()
        }
        #[cfg(not(feature = "vision"))]
        {
            false
        }
    }

    pub(crate) fn forward_chunked_multimodal(
        &self,
        tokens: &[u32],
        images: &[LlamaImage],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2, F>>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        let Some(image_pad_token) = self.config.image_pad_token else {
            return self.forward(tokens, images, device, cache);
        };
        if images.is_empty() {
            return self.forward(tokens, images, device, cache);
        }

        let mut image_index = 0;
        let mut segment_start = 0;
        let mut text_prefix = Vec::new();
        let mut last_logits = None;
        for (index, token) in tokens.iter().copied().enumerate() {
            if token != image_pad_token || image_index >= images.len() {
                continue;
            }
            let mut text_tokens = std::mem::take(&mut text_prefix);
            text_tokens.extend_from_slice(&tokens[segment_start..index]);
            if let Some(image_start_token) = self.config.image_start_token {
                text_tokens.push(image_start_token);
            }
            if !text_tokens.is_empty() {
                self.forward_text_chunk_for_multimodal(
                    &text_tokens,
                    device,
                    cache.as_deref_mut(),
                    false,
                )?;
            }
            let image_hidden = self.forward_image_embeddings_only_hidden_f32(
                &images[image_index..image_index + 1],
                device,
                cache.as_deref_mut(),
            )?;
            image_index += 1;
            segment_start = index + 1;
            if let Some(image_end_token) = self.config.image_end_token {
                text_prefix.push(image_end_token);
            }
            if segment_start < tokens.len() || !text_prefix.is_empty() {
                resolve_intermediate_hidden_f32(&image_hidden);
            } else {
                last_logits = Some(self.forward_logits_from_hidden_f32(image_hidden)?);
            }
        }

        if segment_start < tokens.len() || !text_prefix.is_empty() {
            let mut text_tokens = text_prefix;
            text_tokens.extend_from_slice(&tokens[segment_start..]);
            last_logits = self.forward_text_chunk_for_multimodal(
                &text_tokens,
                device,
                cache.as_deref_mut(),
                true,
            )?;
        }

        last_logits.ok_or_else(|| fusor::Error::msg("No tokens to forward"))
    }

    fn forward_text_chunk_for_multimodal(
        &self,
        text_tokens: &[u32],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
        return_logits: bool,
    ) -> Result<Option<Tensor<2, F>>>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        if text_tokens.is_empty() {
            return Ok(None);
        }

        let has_cache_prefix = cache
            .as_ref()
            .map(|cache| !cache.tokens.is_empty())
            .unwrap_or(false);
        let incremental = device.is_gpu()
            && has_cache_prefix
            && text_tokens.len() > 1
            && std::env::var_os("KALOSM_LLAMA_ENABLE_MULTIMODAL_TEXT_INCREMENTAL").is_some();

        if incremental {
            let last = text_tokens.len() - 1;
            for (index, token) in text_tokens.iter().copied().enumerate() {
                let one = [token];
                if return_logits && index == last {
                    return Ok(Some(self.forward(
                        &one,
                        &[],
                        device,
                        cache.as_deref_mut(),
                    )?));
                }
                let hidden =
                    self.forward_last_hidden_f32(&one, &[], device, cache.as_deref_mut())?;
                resolve_intermediate_hidden_f32(&hidden);
            }
            return Ok(None);
        }

        if return_logits {
            Ok(Some(self.forward(text_tokens, &[], device, cache)?))
        } else {
            let hidden = self.forward_last_hidden_f32(text_tokens, &[], device, cache)?;
            resolve_intermediate_hidden_f32(&hidden);
            Ok(None)
        }
    }

    #[cfg(feature = "vision")]
    fn forward_image_embeddings_only_hidden_f32(
        &self,
        raw_images: &[LlamaImage],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2, f32>>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        let vision_encoder = self
            .vision_encoder
            .as_ref()
            .ok_or_else(|| fusor::Error::msg("Image chunk requires a vision encoder"))?;
        let image_pad_token = self
            .config
            .image_pad_token
            .ok_or_else(|| fusor::Error::msg("Image chunk requires an image token"))?;
        let (image, hints) = raw_images
            .first()
            .ok_or_else(|| fusor::Error::msg("Image chunk requires an image"))?;
        let t_encode = Instant::now();
        let (pixels, grid) =
            vision_encoder.preprocess_image(image, hints.min_tokens(), hints.max_tokens())?;
        let pixels_f: Tensor<2, F> = pixels.cast();
        let image_embeds = vision_encoder.forward_image(&pixels_f, grid)?;
        let mut embeddings_f32: Tensor<2, f32> = image_embeds.cast();
        if vision_encoder.outputs_on_isolated_device() {
            embeddings_f32 = copy_image_embeddings_to_device(embeddings_f32, device)?;
        }
        debug_tensor_stats_f32(&embeddings_f32, "image_embeds_projected");
        let seq_len = embeddings_f32.shape()[0];
        let embeddings_f32 = embeddings_f32.unsqueeze(0).to_concrete();

        let index_pos = cache.as_ref().map(|c| c.tokens.len()).unwrap_or_default();
        if let Some(cache) = cache.as_mut() {
            cache
                .tokens
                .extend(std::iter::repeat_n(image_pad_token, seq_len));
        }

        // Image chunks carry no text tokens, so every position shares a single
        // zeroed per-layer token that the helper broadcasts across the chunk.
        let per_layer_inputs = self
            .compute_per_layer_inputs(&embeddings_f32, || Tensor::from_slice(device, [1, 1], &[0u32]));

        let encoded = EncodedTokens {
            embeddings: embeddings_f32.cast(),
            per_layer_inputs,
            seq_len,
            index_pos,
            pos_ids: None,
            // The whole image chunk attends bidirectionally.
            non_causal_token_ranges: std::iter::once(0..seq_len).collect(),
        };
        self.forward_last_hidden_from_embeddings(encoded, device, cache, Some(t_encode.elapsed()))
    }

    #[cfg(not(feature = "vision"))]
    fn forward_image_embeddings_only_hidden_f32(
        &self,
        _raw_images: &[LlamaImage],
        _device: &Device,
        _cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2, f32>>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        Err(fusor::Error::msg(
            "Image chunks require the `vision` feature",
        ))
    }

    pub(crate) fn forward_last_hidden_f32(
        &self,
        tokens: &[u32],
        images: &[LlamaImage],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<Tensor<2, f32>>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        let t_encode = Instant::now();
        let encoded = self.encode_tokens(tokens, images, device, cache.as_deref_mut())?;
        self.forward_last_hidden_from_embeddings(encoded, device, cache, Some(t_encode.elapsed()))
    }

    pub(crate) fn forward_logits_and_nextn_f32(
        &self,
        tokens: &[u32],
        images: &[LlamaImage],
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
    ) -> Result<TargetBatchOutput>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        let t_encode = Instant::now();
        let encoded = self.encode_tokens(tokens, images, device, cache.as_deref_mut())?;
        let pre_norm = self.forward_pre_norm_hidden_from_embeddings(
            encoded,
            device,
            cache,
            Some(t_encode.elapsed()),
        )?;
        let normed = self.norm.forward_generic(&pre_norm.hidden);
        let h_nextn: Tensor<2, f32> = normed.clone().cast::<f32>().squeeze(0).to_concrete();
        let logits = self.logits_from_hidden_f32(normed.cast::<f32>());
        let logits: Tensor<2, f32> = logits.squeeze(0).to_concrete();
        Ok(TargetBatchOutput { logits, h_nextn })
    }

    pub(crate) fn forward_last_hidden_f32_gpu_token(
        &self,
        token: &Tensor<1, u32>,
        device: &Device,
        cache: &mut LlamaCache,
    ) -> Result<(Tensor<2, f32>, usize)>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        #[cfg(feature = "vision")]
        if !self.supports_gpu_token_run_ahead() {
            return Err(fusor::Error::msg(
                "GPU token run-ahead is not available for this vision model",
            ));
        }

        if cache.tokens.len() + 1 > self.config.context_length {
            return Err(fusor::Error::msg(
                "GPU token run-ahead cannot trim a full context",
            ));
        }

        let cache_slot = cache.tokens.len();
        cache.tokens.push(0);
        let x = token.unsqueeze(0);
        let mut embeddings_f32 = self.tok_embeddings.forward(&x);
        if let Some(scale) = self.tok_embedding_scale {
            embeddings_f32 = (embeddings_f32 * scale).to_concrete();
        }
        let per_layer_inputs = self.compute_per_layer_inputs(&embeddings_f32, || x.clone());
        let embeddings: Tensor<3, F> = embeddings_f32.cast();
        let encoded = EncodedTokens {
            embeddings,
            per_layer_inputs,
            seq_len: 1,
            index_pos: cache_slot,
            pos_ids: None,
            non_causal_token_ranges: Vec::new(),
        };
        let hidden =
            self.forward_last_hidden_from_embeddings(encoded, device, Some(cache), None)?;
        Ok((hidden, cache_slot))
    }

    fn get_attention_mask(
        &self,
        seq_len: usize,
        index_pos: usize,
        sliding_window_size: Option<usize>,
        non_causal_token_ranges: &[Range<usize>],
        device: &Device,
    ) -> AttentionMask<f32> {
        if non_causal_token_ranges.is_empty() {
            return self
                .masks
                .get_mask(seq_len, index_pos, sliding_window_size, device);
        }

        let cols = index_pos + seq_len;
        let mask_data = non_causal_mask_data(
            seq_len,
            index_pos,
            sliding_window_size,
            non_causal_token_ranges,
        );
        let mask: Tensor<2, f32> = Tensor::new(device, mask_data.as_slice())
            .reshape([seq_len, cols])
            .to_concrete();
        AttentionMask::<f32>::new(mask)
    }

    fn can_skip_attention_mask(
        seq_len: usize,
        index_pos: usize,
        sliding_window_size: Option<usize>,
        non_causal_token_ranges: &[Range<usize>],
    ) -> bool {
        if std::env::var_os("KALOSM_LLAMA_DISABLE_NON_CAUSAL_MASK_SKIP").is_some() {
            return false;
        }
        let all_query_tokens_are_non_causal = non_causal_token_ranges
            .iter()
            .any(|range| range.start == 0 && range.end >= seq_len);
        if !all_query_tokens_are_non_causal {
            return false;
        }

        match sliding_window_size {
            Some(window) => window >= index_pos + seq_len,
            None => true,
        }
    }

    fn forward_last_hidden_from_embeddings(
        &self,
        encoded: EncodedTokens<F>,
        device: &Device,
        cache: Option<&mut LlamaCache>,
        encode_elapsed: Option<Duration>,
    ) -> Result<Tensor<2, f32>>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        let output =
            self.forward_pre_norm_hidden_from_embeddings(encoded, device, cache, encode_elapsed)?;
        let x = output.hidden.i((.., output.seq_len - 1, ..));
        let x = self.norm.forward_generic_2d(&x);
        Ok(x.cast::<f32>())
    }

    fn forward_pre_norm_hidden_from_embeddings(
        &self,
        encoded: EncodedTokens<F>,
        device: &Device,
        mut cache: Option<&mut LlamaCache>,
        encode_elapsed: Option<Duration>,
    ) -> Result<PreNormForwardOutput<F>>
    where
        F: CastTo<f32> + CastTensor<f32> + Default,
        f32: CastTo<F> + CastTensor<F>,
    {
        let EncodedTokens {
            embeddings: mut layer_in,
            per_layer_inputs,
            seq_len,
            index_pos,
            pos_ids,
            non_causal_token_ranges,
        } = encoded;
        let _trace_text_prefill = seq_len > 1 && std::env::var_os("KALOSM_TRACE_TEXT").is_some();
        let trace_forward_timing =
            seq_len > 1 || std::env::var_os("KALOSM_TRACE_FORWARD_TIMING").is_some();
        if trace_forward_timing {
            if let Some(encode_elapsed) = encode_elapsed {
                tracing::info!(
                    "[timing] encode_tokens (incl. vision): {:.2?} seq_len={}",
                    encode_elapsed,
                    seq_len
                );
            }
        }
        let t_text_layers = Instant::now();
        let trace_layer_nan = seq_len == 1 && std::env::var_os("KALOSM_TRACE_LAYER_NAN").is_some();
        if trace_layer_nan {
            let probe: fusor::Tensor<3, f32> = layer_in.cast();
            debug_check_nan_f32(&probe, usize::MAX, "embed", index_pos);
        }

        let mut non_causal_masks: HashMap<Option<usize>, AttentionMask<f32>> = HashMap::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let x = layer_in;
            let residual: Tensor<3, f32> = x.cast();
            let x = layer.attention_norm.forward_generic(&x);
            if trace_layer_nan {
                let probe: fusor::Tensor<3, f32> = x.clone().cast();
                debug_check_nan_f32(&probe, i, "post_attn_norm", index_pos);
            }
            let mask = if seq_len > 1 {
                if Self::can_skip_attention_mask(
                    seq_len,
                    index_pos,
                    layer.sliding_window_size,
                    &non_causal_token_ranges,
                ) {
                    None
                } else if non_causal_token_ranges.is_empty() {
                    Some(self.get_attention_mask(
                        seq_len,
                        index_pos,
                        layer.sliding_window_size,
                        &non_causal_token_ranges,
                        device,
                    ))
                } else {
                    Some(
                        non_causal_masks
                            .entry(layer.sliding_window_size)
                            .or_insert_with(|| {
                                self.get_attention_mask(
                                    seq_len,
                                    index_pos,
                                    layer.sliding_window_size,
                                    &non_causal_token_ranges,
                                    device,
                                )
                            })
                            .clone(),
                    )
                }
            } else {
                None
            };
            let shared_kv = if let Some(shared_kv_layer) = layer.shared_kv_layer {
                let cache_ref = cache.as_deref().ok_or_else(|| {
                    fusor::Error::msg("Gemma 4 shared KV attention requires a populated cache")
                })?;
                let key = cache_ref.blocks[shared_kv_layer]
                    .k()
                    .cloned()
                    .ok_or_else(|| {
                        fusor::Error::msg("Gemma 4 shared KV source key cache is empty")
                    })?;
                let value = cache_ref.blocks[shared_kv_layer]
                    .v()
                    .cloned()
                    .ok_or_else(|| {
                        fusor::Error::msg("Gemma 4 shared KV source value cache is empty")
                    })?;
                Some((key, value))
            } else {
                None
            };
            let mut attn = {
                if let Some((shared_key, shared_value)) = shared_kv.as_ref() {
                    layer.forward_with_shared_kv(
                        &x,
                        mask.as_ref(),
                        index_pos,
                        pos_ids.as_ref(),
                        shared_key,
                        shared_value,
                    )
                } else {
                    #[cfg(feature = "vision")]
                    {
                        if trace_layer_nan {
                            layer.forward_with_trace(
                                &x,
                                mask.as_ref(),
                                index_pos,
                                pos_ids.as_ref(),
                                cache.as_mut().map(|c| &mut c.blocks[i]),
                                i,
                            )
                        } else {
                            layer.forward(
                                &x,
                                mask.as_ref(),
                                index_pos,
                                pos_ids.as_ref(),
                                cache.as_mut().map(|c| &mut c.blocks[i]),
                            )
                        }
                    }
                    #[cfg(not(feature = "vision"))]
                    {
                        layer.forward(
                            &x,
                            mask.as_ref(),
                            index_pos,
                            pos_ids.as_ref(),
                            cache.as_mut().map(|c| &mut c.blocks[i]),
                        )
                    }
                }
            };
            if trace_layer_nan {
                let probe: fusor::Tensor<3, f32> = attn.clone().cast();
                debug_check_nan_f32(&probe, i, "attn_out", index_pos);
            }
            if let Some(post_attention_norm) = &layer.post_attention_norm {
                attn = post_attention_norm.forward_generic(&attn);
            }
            let attn_f32: Tensor<3, f32> = attn.cast();

            // MLP over RMSNorm(attention_output + residual). The fused path avoids
            // materializing the mid-block residual add just to feed normalization.
            let x = layer.ffn_norm.forward_residual_f32(&attn_f32, &residual);
            if layer.post_ffn_norm.is_none() {
                if let Some(layer_out) = layer
                    .feed_forward_variant
                    .forward_add_residuals(&x, &attn_f32, &residual)
                {
                    layer_in = layer_out;
                } else {
                    let x = layer.feed_forward_variant.forward(&x);
                    let x_f32: Tensor<3, f32> = x.cast();
                    layer_in = (x_f32 + attn_f32 + residual).cast();
                }
            } else {
                let mut x = layer.feed_forward_variant.forward(&x);
                if let Some(post_ffn_norm) = &layer.post_ffn_norm {
                    x = post_ffn_norm.forward_generic(&x);
                }
                let x_f32: Tensor<3, f32> = x.cast();
                layer_in = (x_f32 + attn_f32 + residual).cast();
            }
            if let (
                Some(per_layer_inputs),
                Some(per_layer_inp_gate),
                Some(per_layer_proj),
                Some(per_layer_post_norm),
            ) = (
                per_layer_inputs.as_ref(),
                layer.per_layer_inp_gate.as_ref(),
                layer.per_layer_proj.as_ref(),
                layer.per_layer_post_norm.as_ref(),
            ) {
                let pe_in: Tensor<3, f32> = layer_in.cast();
                let gate = pe_in.q_mat_mul(per_layer_inp_gate).gelu();
                let layer_input: Tensor<3, F> = per_layer_inputs
                    .narrow(2, i, 1)
                    .squeeze::<3>(2)
                    .to_concrete();
                let projected = (gate.cast::<F>() * layer_input)
                    .to_concrete()
                    .cast::<f32>()
                    .q_mat_mul(per_layer_proj)
                    .cast();
                let projected = per_layer_post_norm.forward_generic(&projected);
                let projected_f32: Tensor<3, f32> = projected.cast();
                layer_in = (pe_in + projected_f32).cast();
            }
            if let Some(layer_output_scale) = &layer.layer_output_scale {
                let scale = layer_output_scale
                    .reshape([1, 1, 1])
                    .broadcast_as(layer_in.shape())
                    .to_concrete();
                layer_in = (layer_in * scale).to_concrete();
            }
            if trace_layer_nan {
                let probe: fusor::Tensor<3, f32> = layer_in.cast();
                debug_check_nan_f32(&probe, i, "ffn_unfused", index_pos);
            }
        }
        if trace_forward_timing {
            tracing::info!("[timing] text layer loop: {:.2?}", t_text_layers.elapsed());
        }
        Ok(PreNormForwardOutput {
            hidden: layer_in,
            seq_len,
        })
    }

    pub(crate) fn output_matrix(&self) -> &QMatrix {
        &self.output
    }
}
