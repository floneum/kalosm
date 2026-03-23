use fusor::{
    Device, MaskKind, RopeCache, Tensor, VarBuilder,
    cache::AttentionMask,
    layers::{Embedding, RecurrentWeights, recurrent_forward},
};

use crate::data::{ACTION_DIRECTION_COUNT, ACTION_MODE_COUNT, CanvasStateSpec};

#[derive(Clone, Copy)]
struct ModelShape {
    block_size: usize,
    n_embd: usize,
    n_head: usize,
    n_kv_head: usize,
    n_ff: usize,
    conv_kernel_size: usize,
    eps: f32,
}

#[derive(Clone)]
pub struct InteractiveNanoChatModel {
    shape: ModelShape,
    use_extra_norms: bool,
    token_input_mode: Tensor<2, f32>,
    token_input_direction: Tensor<2, f32>,
    token_input_count: Tensor<2, f32>,
    wpe: Option<Tensor<2, f32>>,
    canvas_state: Option<CanvasStateEmbeddings>,
    rotary: Option<RopeCache>,
    ln_in_weight: Tensor<1, f32>,
    ln_in_bias: Tensor<1, f32>,
    blocks: Vec<TransformerBlock>,
    ln_f_weight: Tensor<1, f32>,
    ln_f_bias: Tensor<1, f32>,
    max_count: usize,
    action_head: OutputHead,
}

#[derive(Clone)]
struct OutputHead {
    weight: Tensor<2, f32>,
    bias: Tensor<1, f32>,
}

pub struct InteractiveActionLogits {
    pub mode: Tensor<3, f32>,
    pub direction: Tensor<3, f32>,
    pub count: Tensor<3, f32>,
}

#[derive(Clone)]
struct TransformerBlock {
    ln_1_weight: Tensor<1, f32>,
    ln_1_bias: Tensor<1, f32>,
    mixer: SequenceMixer,
    ln_attn_out_weight: Tensor<1, f32>,
    ln_attn_out_bias: Tensor<1, f32>,
    ln_2_weight: Tensor<1, f32>,
    ln_2_bias: Tensor<1, f32>,
    mlp: Mlp,
    ln_mlp_out_weight: Tensor<1, f32>,
    ln_mlp_out_bias: Tensor<1, f32>,
}

#[derive(Clone)]
enum SequenceMixer {
    Attention(CausalSelfAttention),
    Conv(ConvMixer),
    Recurrent(RecurrentMixer),
}

#[derive(Clone)]
struct CausalSelfAttention {
    c_attn_q: Tensor<2, f32>,
    c_attn_k: Tensor<2, f32>,
    c_attn_v: Tensor<2, f32>,
    c_proj: Tensor<2, f32>,
}

#[derive(Clone)]
struct RecurrentMixer {
    input_proj: Tensor<2, f32>,
    state_proj: Tensor<2, f32>,
    gate_input_proj: Tensor<2, f32>,
    gate_state_proj: Tensor<2, f32>,
    out_proj: Tensor<2, f32>,
}

#[derive(Clone)]
struct ConvMixer {
    kernels: Vec<Tensor<2, f32>>,
    bias: Tensor<1, f32>,
    out_proj: Tensor<2, f32>,
}

#[derive(Clone)]
struct Mlp {
    c_fc: Tensor<2, f32>,
    c_fc_bias: Tensor<1, f32>,
    c_proj: Tensor<2, f32>,
    c_proj_bias: Tensor<1, f32>,
}

#[derive(Clone)]
struct CanvasStateEmbeddings {
    spec: CanvasStateSpec,
    cursor_x: Tensor<2, f32>,
    cursor_y: Tensor<2, f32>,
    pen_state: Tensor<2, f32>,
}

impl InteractiveNanoChatModel {
    pub fn load(device: &Device, vb: &mut VarBuilder) -> fusor::Result<Self> {
        let block_size = metadata_u32(vb, "nanochat.block_size")? as usize;
        let n_embd = metadata_u32(vb, "nanochat.embedding_length")? as usize;
        let n_head = vb
            .get_metadata("nanochat.head_count")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or(1)
            .max(1);
        let n_kv_head = vb
            .get_metadata("nanochat.kv_head_count")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or(n_head)
            .max(1);
        let n_ff = metadata_u32(vb, "nanochat.feed_forward_length")? as usize;
        let conv_kernel_size = vb
            .get_metadata("nanochat.conv_kernel_size")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or(3)
            .max(1);
        let n_layer = metadata_u32(vb, "nanochat.block_count")? as usize;
        let attention_period = vb
            .get_metadata("nanochat.attention_period")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or(4)
            .max(1);
        let max_count = vb
            .get_metadata("tokenizer.nanochat.max_count")
            .and_then(|value| value.to_u32().ok())
            .map(|value| value as usize)
            .unwrap_or(8)
            .max(1);
        let eps = metadata_f32(vb, "nanochat.eps")?;
        let use_rope = vb
            .get_metadata("nanochat.use_rope")
            .and_then(|value| value.to_bool().ok())
            .unwrap_or(false);
        let rope_theta = vb
            .get_metadata("nanochat.rope_theta")
            .and_then(|value| value.to_f32().ok())
            .unwrap_or(10_000.0);
        let use_extra_norms = vb
            .get_metadata("nanochat.use_extra_norms")
            .and_then(|value| value.to_bool().ok())
            .unwrap_or_else(|| {
                vb.contains_key("input_norm.weight")
                    || vb.contains_key("blk.0.attn_out_norm.weight")
                    || vb.contains_key("blk.0.mlp_out_norm.weight")
            });
        let use_canvas_state_embeddings = vb
            .get_metadata("nanochat.use_canvas_state_embeddings")
            .and_then(|value| value.to_bool().ok())
            .unwrap_or_else(|| vb.contains_key("cursor_x_embd.weight"));
        let shape = ModelShape {
            block_size,
            n_embd,
            n_head,
            n_kv_head,
            n_ff,
            conv_kernel_size,
            eps,
        };
        assert_eq!(
            shape.n_embd % shape.n_head,
            0,
            "checkpoint embedding length {} must be divisible by head count {}",
            shape.n_embd,
            shape.n_head
        );
        assert_eq!(
            shape.n_head % shape.n_kv_head,
            0,
            "checkpoint head count {} must be divisible by kv head count {}",
            shape.n_head,
            shape.n_kv_head
        );
        if use_rope {
            assert_eq!(
                shape.head_dim() % 2,
                0,
                "RoPE requires an even head dimension, got {}",
                shape.head_dim()
            );
        }

        let token_input_mode: Tensor<2, f32> =
            vb.get("token_input_mode.weight", device)?.dequantize();
        let token_input_direction: Tensor<2, f32> =
            vb.get("token_input_direction.weight", device)?.dequantize();
        let token_input_count: Tensor<2, f32> =
            vb.get("token_input_count.weight", device)?.dequantize();
        let wpe = if vb.contains_key("position_embd.weight") {
            Some(vb.get("position_embd.weight", device)?.dequantize())
        } else if use_rope {
            None
        } else {
            return Err(fusor::Error::VarBuilder(
                "missing position_embd.weight for non-RoPE checkpoint".into(),
            ));
        };
        let canvas_state = if use_canvas_state_embeddings {
            let cursor_x: Tensor<2, f32> = vb.get("cursor_x_embd.weight", device)?.dequantize();
            let cursor_y: Tensor<2, f32> = vb.get("cursor_y_embd.weight", device)?.dequantize();
            let pen_state: Tensor<2, f32> = vb.get("pen_state_embd.weight", device)?.dequantize();
            let coordinate_vocab_size = vb
                .get_metadata("nanochat.canvas_coordinate_vocab")
                .and_then(|value| value.to_u32().ok())
                .map(|value| value as usize)
                .unwrap_or(cursor_x.shape()[0]);
            let coordinate_offset = vb
                .get_metadata("nanochat.canvas_coordinate_offset")
                .and_then(|value| value.to_u32().ok())
                .map(|value| value as i32)
                .unwrap_or_else(|| coordinate_vocab_size.saturating_sub(1) as i32 / 2);
            Some(CanvasStateEmbeddings {
                spec: CanvasStateSpec {
                    coordinate_vocab_size,
                    coordinate_offset,
                },
                cursor_x,
                cursor_y,
                pen_state,
            })
        } else {
            None
        };
        let rotary = if use_rope {
            Some(RopeCache::new(
                shape.head_dim(),
                shape.block_size,
                rope_theta,
                device,
            )?)
        } else {
            None
        };
        let ln_in_weight = get_tensor1_or_default(
            vb,
            device,
            &["input_norm.weight"],
            ones_1d(device, shape.n_embd),
        )?;
        let ln_in_bias = get_tensor1_or_default(
            vb,
            device,
            &["input_norm.bias"],
            zeros_1d(device, shape.n_embd),
        )?;
        let blocks = (0..n_layer)
            .map(|index| {
                TransformerBlock::load(
                    device,
                    &mut vb.pp(format!("blk.{index}")),
                    is_attention_layer(index, attention_period),
                    shape,
                )
            })
            .collect::<fusor::Result<Vec<_>>>()?;
        let ln_f_weight: Tensor<1, f32> = vb.get("output_norm.weight", device)?.dequantize();
        let ln_f_bias: Tensor<1, f32> = vb.get("output_norm.bias", device)?.dequantize();
        let action_head = OutputHead::load(device, vb, "output_action")?;

        Ok(Self {
            shape,
            use_extra_norms,
            token_input_mode,
            token_input_direction,
            token_input_count,
            wpe,
            canvas_state,
            rotary,
            ln_in_weight,
            ln_in_bias,
            blocks,
            ln_f_weight,
            ln_f_bias,
            max_count,
            action_head,
        })
    }

    pub fn block_size(&self) -> usize {
        self.shape.block_size
    }

    pub fn canvas_state_spec(&self) -> Option<CanvasStateSpec> {
        self.canvas_state.as_ref().map(|state| state.spec)
    }

    #[allow(dead_code)]
    pub fn forward(
        &self,
        mode_inputs: &Tensor<2, u32>,
        direction_inputs: &Tensor<2, u32>,
        count_inputs: &Tensor<2, u32>,
        position_inputs: &Tensor<2, u32>,
        cursor_x_inputs: &Tensor<2, u32>,
        cursor_y_inputs: &Tensor<2, u32>,
        pen_state_inputs: &Tensor<2, u32>,
        causal_mask: &AttentionMask<f32>,
    ) -> InteractiveActionLogits {
        let batch_size = mode_inputs.shape()[0];
        let mode_emb: Tensor<3, f32> =
            Embedding::new_from_tensor(self.token_input_mode.clone()).forward(mode_inputs);
        let dir_emb: Tensor<3, f32> =
            Embedding::new_from_tensor(self.token_input_direction.clone())
                .forward(direction_inputs);
        let count_emb: Tensor<3, f32> =
            Embedding::new_from_tensor(self.token_input_count.clone()).forward(count_inputs);
        let mut x: Tensor<3, f32> = (mode_emb + dir_emb + count_emb).to_concrete();
        if let Some(wpe) = &self.wpe {
            let position_embeddings: Tensor<3, f32> =
                Embedding::new_from_tensor(wpe.clone()).forward(position_inputs);
            x = (x + position_embeddings).to_concrete();
        }
        if let Some(canvas_state) = &self.canvas_state {
            let cursor_x_embeddings: Tensor<3, f32> =
                Embedding::new_from_tensor(canvas_state.cursor_x.clone()).forward(cursor_x_inputs);
            let cursor_y_embeddings: Tensor<3, f32> =
                Embedding::new_from_tensor(canvas_state.cursor_y.clone()).forward(cursor_y_inputs);
            let pen_state_embeddings: Tensor<3, f32> =
                Embedding::new_from_tensor(canvas_state.pen_state.clone())
                    .forward(pen_state_inputs);
            x = (x + cursor_x_embeddings + cursor_y_embeddings + pen_state_embeddings)
                .to_concrete();
        }
        if self.use_extra_norms {
            let ln_in_weight = self.ln_in_weight.broadcast_as(x.shape());
            let ln_in_bias = self.ln_in_bias.broadcast_as(x.shape());
            x = x
                .layer_norm(&ln_in_weight, Some(&ln_in_bias), self.shape.eps, true)
                .to_concrete();
        }

        for block in &self.blocks {
            x = block.forward(
                x,
                causal_mask,
                batch_size,
                self.shape,
                self.rotary.as_ref(),
                self.use_extra_norms,
            );
        }

        let ln_f_weight = self.ln_f_weight.broadcast_as(x.shape());
        let ln_f_bias = self.ln_f_bias.broadcast_as(x.shape());
        let x = x
            .layer_norm(&ln_f_weight, Some(&ln_f_bias), self.shape.eps, true)
            .to_concrete();
        let action_logits = self.action_head.project(&x, batch_size, self.shape.n_embd);
        let seq_len = action_logits.shape()[1];
        let mode_end = ACTION_MODE_COUNT;
        let direction_end = mode_end + ACTION_DIRECTION_COUNT;
        let count_end = direction_end + self.max_count;
        InteractiveActionLogits {
            mode: action_logits
                .slice([0..batch_size, 0..seq_len, 0..mode_end])
                .to_concrete(),
            direction: action_logits
                .slice([0..batch_size, 0..seq_len, mode_end..direction_end])
                .to_concrete(),
            count: action_logits
                .slice([0..batch_size, 0..seq_len, direction_end..count_end])
                .to_concrete(),
        }
    }
}

impl TransformerBlock {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        is_attention_layer: bool,
        shape: ModelShape,
    ) -> fusor::Result<Self> {
        let ln_1_weight: Tensor<1, f32> = vb.get("ln_1.weight", device)?.dequantize();
        let ln_1_bias: Tensor<1, f32> = vb.get("ln_1.bias", device)?.dequantize();
        let mixer = if is_attention_layer {
            SequenceMixer::Attention(CausalSelfAttention::load(device, vb)?)
        } else if vb.contains_key("conv_kernel.0.weight") {
            SequenceMixer::Conv(ConvMixer::load(device, vb, shape)?)
        } else {
            SequenceMixer::Recurrent(RecurrentMixer::load(device, vb)?)
        };
        let ln_attn_out_weight = get_tensor1_or_default(
            vb,
            device,
            &["attn_out_norm.weight"],
            ones_1d(device, shape.n_embd),
        )?;
        let ln_attn_out_bias = get_tensor1_or_default(
            vb,
            device,
            &["attn_out_norm.bias"],
            zeros_1d(device, shape.n_embd),
        )?;
        let ln_2_weight: Tensor<1, f32> = vb.get("ln_2.weight", device)?.dequantize();
        let ln_2_bias: Tensor<1, f32> = vb.get("ln_2.bias", device)?.dequantize();
        let mlp = Mlp::load(device, vb)?;
        let ln_mlp_out_weight = get_tensor1_or_default(
            vb,
            device,
            &["mlp_out_norm.weight"],
            ones_1d(device, shape.n_embd),
        )?;
        let ln_mlp_out_bias = get_tensor1_or_default(
            vb,
            device,
            &["mlp_out_norm.bias"],
            zeros_1d(device, shape.n_embd),
        )?;
        Ok(Self {
            ln_1_weight,
            ln_1_bias,
            mixer,
            ln_attn_out_weight,
            ln_attn_out_bias,
            ln_2_weight,
            ln_2_bias,
            mlp,
            ln_mlp_out_weight,
            ln_mlp_out_bias,
        })
    }

    fn forward(
        &self,
        x: Tensor<3, f32>,
        causal_mask: &AttentionMask<f32>,
        batch_size: usize,
        shape: ModelShape,
        rotary: Option<&RopeCache>,
        use_extra_norms: bool,
    ) -> Tensor<3, f32> {
        let ln_1_weight = self.ln_1_weight.broadcast_as(x.shape());
        let ln_1_bias = self.ln_1_bias.broadcast_as(x.shape());
        let attn_input = x
            .layer_norm(&ln_1_weight, Some(&ln_1_bias), shape.eps, true)
            .to_concrete();
        let attn_output = self
            .mixer
            .forward(&attn_input, causal_mask, batch_size, shape, rotary);
        let attn_output = if use_extra_norms {
            let ln_attn_out_weight = self.ln_attn_out_weight.broadcast_as(attn_output.shape());
            let ln_attn_out_bias = self.ln_attn_out_bias.broadcast_as(attn_output.shape());
            attn_output
                .layer_norm(
                    &ln_attn_out_weight,
                    Some(&ln_attn_out_bias),
                    shape.eps,
                    true,
                )
                .to_concrete()
        } else {
            attn_output
        };
        let x: Tensor<3, f32> = (x + attn_output).to_concrete();

        let ln_2_weight = self.ln_2_weight.broadcast_as(x.shape());
        let ln_2_bias = self.ln_2_bias.broadcast_as(x.shape());
        let mlp_input = x
            .layer_norm(&ln_2_weight, Some(&ln_2_bias), shape.eps, true)
            .to_concrete();
        let mlp_output = self.mlp.forward(&mlp_input, batch_size, shape);
        let mlp_output = if use_extra_norms {
            let ln_mlp_out_weight = self.ln_mlp_out_weight.broadcast_as(mlp_output.shape());
            let ln_mlp_out_bias = self.ln_mlp_out_bias.broadcast_as(mlp_output.shape());
            mlp_output
                .layer_norm(&ln_mlp_out_weight, Some(&ln_mlp_out_bias), shape.eps, true)
                .to_concrete()
        } else {
            mlp_output
        };
        (x + mlp_output).to_concrete()
    }
}

impl CausalSelfAttention {
    fn load(device: &Device, vb: &mut VarBuilder) -> fusor::Result<Self> {
        Ok(Self {
            c_attn_q: get_tensor2(vb, device, &["attn_q.weight", "attn.attn_q.weight"])?,
            c_attn_k: get_tensor2(vb, device, &["attn_k.weight", "attn.attn_k.weight"])?,
            c_attn_v: get_tensor2(vb, device, &["attn_v.weight", "attn.attn_v.weight"])?,
            c_proj: get_tensor2(vb, device, &["attn_proj.weight", "attn.attn_proj.weight"])?,
        })
    }

    fn forward(
        &self,
        x: &Tensor<3, f32>,
        causal_mask: &AttentionMask<f32>,
        batch_size: usize,
        shape: ModelShape,
        rotary: Option<&RopeCache>,
    ) -> Tensor<3, f32> {
        let head_dim = shape.head_dim();
        let seq_len = x.shape()[1];
        let q = x
            .mat_mul(
                &self
                    .c_attn_q
                    .broadcast_as([batch_size, shape.n_embd, shape.n_embd]),
            )
            .to_concrete();
        let k = x
            .mat_mul(
                &self
                    .c_attn_k
                    .broadcast_as([batch_size, shape.n_embd, shape.kv_dim()]),
            )
            .to_concrete();
        let v = x
            .mat_mul(
                &self
                    .c_attn_v
                    .broadcast_as([batch_size, shape.n_embd, shape.kv_dim()]),
            )
            .to_concrete();

        let q = q
            .reshape([batch_size, seq_len, shape.n_head, head_dim])
            .transpose(1, 2)
            .to_concrete();
        let k = k
            .reshape([batch_size, seq_len, shape.n_kv_head, head_dim])
            .transpose(1, 2)
            .to_concrete();
        let v = v
            .reshape([batch_size, seq_len, shape.n_kv_head, head_dim])
            .transpose(1, 2)
            .to_concrete();
        let (q, k) = match rotary {
            Some(cache) => cache.forward(&q, &k, 0),
            None => (q, k),
        };
        let attended = q
            .flash_attention(
                &k,
                &v,
                1.0 / (head_dim as f32).sqrt(),
                Some((causal_mask.mask(), MaskKind::QKMask)),
            )
            .to_concrete();
        let merged = attended
            .transpose(1, 2)
            .reshape([batch_size, seq_len, shape.n_embd])
            .to_concrete();

        merged
            .mat_mul(
                &self
                    .c_proj
                    .broadcast_as([batch_size, shape.n_embd, shape.n_embd]),
            )
            .to_concrete()
    }
}

impl SequenceMixer {
    fn forward(
        &self,
        x: &Tensor<3, f32>,
        causal_mask: &AttentionMask<f32>,
        batch_size: usize,
        shape: ModelShape,
        rotary: Option<&RopeCache>,
    ) -> Tensor<3, f32> {
        match self {
            SequenceMixer::Attention(attn) => {
                attn.forward(x, causal_mask, batch_size, shape, rotary)
            }
            SequenceMixer::Conv(conv) => conv.forward(x, batch_size, shape),
            SequenceMixer::Recurrent(recurrent) => recurrent.forward(x, batch_size, shape),
        }
    }
}

impl RecurrentMixer {
    fn load(device: &Device, vb: &mut VarBuilder) -> fusor::Result<Self> {
        Ok(Self {
            input_proj: get_tensor2(vb, device, &["recurrent_in.weight"])?,
            state_proj: get_tensor2(vb, device, &["recurrent_state.weight"])?,
            gate_input_proj: get_tensor2(vb, device, &["recurrent_gate_in.weight"])?,
            gate_state_proj: get_tensor2(vb, device, &["recurrent_gate_state.weight"])?,
            out_proj: get_tensor2(vb, device, &["recurrent_out.weight"])?,
        })
    }

    fn forward(&self, x: &Tensor<3, f32>, batch_size: usize, shape: ModelShape) -> Tensor<3, f32> {
        debug_assert_eq!(batch_size, x.shape()[0]);
        debug_assert_eq!(shape.n_embd, x.shape()[2]);
        recurrent_forward(
            x,
            &RecurrentWeights::new(
                self.input_proj.clone(),
                self.state_proj.clone(),
                self.gate_input_proj.clone(),
                self.gate_state_proj.clone(),
                self.out_proj.clone(),
            ),
        )
    }
}

impl ConvMixer {
    fn load(device: &Device, vb: &mut VarBuilder, shape: ModelShape) -> fusor::Result<Self> {
        let mut kernels = Vec::with_capacity(shape.conv_kernel_size);
        for index in 0..shape.conv_kernel_size {
            let key = format!("conv_kernel.{index}.weight");
            let tensor = if vb.contains_key(&key) {
                vb.get(&key, device)?.dequantize()
            } else {
                return Err(fusor::Error::VarBuilder(format!(
                    "missing conv kernel in GGUF metadata: {key}"
                )));
            };
            kernels.push(tensor);
        }
        Ok(Self {
            kernels,
            bias: get_tensor1(vb, device, &["conv_bias"])?,
            out_proj: get_tensor2(vb, device, &["conv_proj.weight"])?,
        })
    }

    fn forward(&self, x: &Tensor<3, f32>, batch_size: usize, shape: ModelShape) -> Tensor<3, f32> {
        let seq_len = x.shape()[1];
        let mut mixed: Tensor<3, f32> =
            Tensor::zeros(&x.device(), [batch_size, seq_len, shape.n_embd]);

        for (offset, kernel) in self.kernels.iter().enumerate() {
            let shifted = causal_shift(x, offset);
            let projected =
                shifted.mat_mul(&kernel.broadcast_as([batch_size, shape.n_embd, shape.n_embd]));
            mixed = (mixed + projected).to_concrete();
        }

        mixed
            .add_(&self.bias.broadcast_as([batch_size, seq_len, shape.n_embd]))
            .relu()
            .mat_mul(
                &self
                    .out_proj
                    .broadcast_as([batch_size, shape.n_embd, shape.n_embd]),
            )
            .to_concrete()
    }
}

impl Mlp {
    fn load(device: &Device, vb: &mut VarBuilder) -> fusor::Result<Self> {
        Ok(Self {
            c_fc: get_tensor2(vb, device, &["mlp_fc.weight", "mlp.mlp_fc.weight"])?,
            c_fc_bias: get_tensor1(vb, device, &["mlp_fc.bias", "mlp.mlp_fc.bias"])?,
            c_proj: get_tensor2(vb, device, &["mlp_proj.weight", "mlp.mlp_proj.weight"])?,
            c_proj_bias: get_tensor1(vb, device, &["mlp_proj.bias", "mlp.mlp_proj.bias"])?,
        })
    }

    fn forward(&self, x: &Tensor<3, f32>, batch_size: usize, shape: ModelShape) -> Tensor<3, f32> {
        let hidden = x
            .mat_mul(
                &self
                    .c_fc
                    .broadcast_as([batch_size, shape.n_embd, shape.n_ff]),
            )
            .add_(&self.c_fc_bias)
            .relu()
            .to_concrete();

        hidden
            .mat_mul(
                &self
                    .c_proj
                    .broadcast_as([batch_size, shape.n_ff, shape.n_embd]),
            )
            .add_(&self.c_proj_bias)
            .to_concrete()
    }
}

impl OutputHead {
    fn load(device: &Device, vb: &mut VarBuilder, prefix: &str) -> fusor::Result<Self> {
        Ok(Self {
            weight: vb.get(&format!("{prefix}.weight"), device)?.dequantize(),
            bias: vb.get(&format!("{prefix}.bias"), device)?.dequantize(),
        })
    }

    fn project(&self, x: &Tensor<3, f32>, batch_size: usize, n_embd: usize) -> Tensor<3, f32> {
        let out_dim = self.weight.shape()[1];
        x.mat_mul(&self.weight.broadcast_as([batch_size, n_embd, out_dim]))
            .add_(&self.bias.broadcast_as([batch_size, x.shape()[1], out_dim]))
            .to_concrete()
    }
}

fn metadata_u32(vb: &VarBuilder, key: &str) -> fusor::Result<u32> {
    vb.get_metadata(key)
        .ok_or_else(|| fusor::Error::msg(format!("Key '{key}' not found in GGUF metadata")))?
        .to_u32()
        .map_err(|error| fusor::Error::msg(error.to_string()))
}

fn metadata_f32(vb: &VarBuilder, key: &str) -> fusor::Result<f32> {
    vb.get_metadata(key)
        .ok_or_else(|| fusor::Error::msg(format!("Key '{key}' not found in GGUF metadata")))?
        .to_f32()
        .map_err(|error| fusor::Error::msg(error.to_string()))
}

fn get_tensor1(
    vb: &mut VarBuilder,
    device: &Device,
    keys: &[&str],
) -> fusor::Result<Tensor<1, f32>> {
    for key in keys {
        if vb.contains_key(key) {
            return Ok(vb.get(key, device)?.dequantize());
        }
    }
    Err(fusor::Error::VarBuilder(format!(
        "none of the candidate keys were found in GGUF metadata: {}",
        keys.join(", ")
    )))
}

fn get_tensor1_or_default(
    vb: &mut VarBuilder,
    device: &Device,
    keys: &[&str],
    default: Tensor<1, f32>,
) -> fusor::Result<Tensor<1, f32>> {
    for key in keys {
        if vb.contains_key(key) {
            return Ok(vb.get(key, device)?.dequantize());
        }
    }
    Ok(default)
}

fn get_tensor2(
    vb: &mut VarBuilder,
    device: &Device,
    keys: &[&str],
) -> fusor::Result<Tensor<2, f32>> {
    for key in keys {
        if vb.contains_key(key) {
            return Ok(vb.get(key, device)?.dequantize());
        }
    }
    Err(fusor::Error::VarBuilder(format!(
        "none of the candidate keys were found in GGUF metadata: {}",
        keys.join(", ")
    )))
}

fn ones_1d(device: &Device, len: usize) -> Tensor<1, f32> {
    Tensor::new(device, &vec![1.0; len])
}

fn zeros_1d(device: &Device, len: usize) -> Tensor<1, f32> {
    Tensor::new(device, &vec![0.0; len])
}

fn is_attention_layer(index: usize, attention_period: usize) -> bool {
    (index + 1) % attention_period.max(1) == 0
}

fn causal_shift(x: &Tensor<3, f32>, offset: usize) -> Tensor<3, f32> {
    if offset == 0 {
        return x.clone();
    }

    let [batch_size, seq_len, n_embd] = x.shape();
    if offset >= seq_len {
        return Tensor::zeros(&x.device(), [batch_size, seq_len, n_embd]);
    }

    let prefix: Tensor<3, f32> = Tensor::zeros(&x.device(), [batch_size, offset, n_embd]);
    let shifted = x
        .slice([0..batch_size, 0..seq_len - offset, 0..n_embd])
        .to_concrete();
    Tensor::cat(vec![prefix, shifted], 1).to_concrete()
}

impl ModelShape {
    fn head_dim(self) -> usize {
        self.n_embd / self.n_head.max(1)
    }

    fn kv_dim(self) -> usize {
        self.n_kv_head * self.head_dim()
    }
}
