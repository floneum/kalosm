use fusor::{
    Device, NodeIndex, Tensor as RawTensor, ToVec1, ToVec2,
    autograd::{Gradients, Graph, Tensor},
    base_inverse_frequency,
};
use fusor_train::{AdamMoments, AdamWModel, AdamWSettings, adamw_update_raw, extract_gradient};
use rand::{Rng, rngs::StdRng};
use std::{io::Write, time::Instant};

use fusor_nanochat::{
    RuntimeConfig,
    data::{
        ACTION_DIRECTION_COUNT, ACTION_MODE_COUNT, CanvasStateSpec, INPUT_MODE_COUNT,
        StrokeTokenizer, canvas_state_spec,
    },
};

pub struct NamedTensor {
    pub name: String,
    pub shape: Vec<u32>,
    pub values: Vec<f32>,
}

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
struct RotaryEmbeddings {
    cos: Tensor<2>,
    sin: Tensor<2>,
}

#[derive(Clone)]
struct CanvasStateEmbeddings {
    spec: CanvasStateSpec,
    cursor_x: Tensor<2>,
    cursor_y: Tensor<2>,
    pen_state: Tensor<2>,
}

#[derive(Clone)]
struct TokenInputEmbeddings {
    mode: Tensor<2>,      // [INPUT_MODE_COUNT=4, n_embd]
    direction: Tensor<2>, // [ACTION_DIRECTION_COUNT=8, n_embd]
    count: Tensor<2>,     // [max_count, n_embd]
}

struct TokenInputAdamState {
    mode: AdamMoments<2>,
    direction: AdamMoments<2>,
    count: AdamMoments<2>,
}

impl TokenInputAdamState {
    fn new(device: &Device, embeddings: &TokenInputEmbeddings) -> Self {
        Self {
            mode: AdamMoments::zeros_like(device, &embeddings.mode),
            direction: AdamMoments::zeros_like(device, &embeddings.direction),
            count: AdamMoments::zeros_like(device, &embeddings.count),
        }
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        self.mode.collect_gpu_keys(keys);
        self.direction.collect_gpu_keys(keys);
        self.count.collect_gpu_keys(keys);
    }
}

#[derive(Clone)]
struct OutputHead {
    weight: Tensor<2>,
    bias: Tensor<1>,
}

pub struct ActionLogits {
    pub mode: Tensor<3>,
    pub direction: Tensor<3>,
    pub count: Tensor<3>,
}

#[derive(Clone)]
pub struct NanoChatModel {
    graph: Graph,
    shape: ModelShape,
    attention_period: usize,
    vocab_size: usize,
    max_count: usize,
    use_rope: bool,
    rope_theta: f32,
    use_extra_norms: bool,
    token_input: TokenInputEmbeddings,
    wpe: Option<Tensor<2>>,
    canvas_state: Option<CanvasStateEmbeddings>,
    rotary: Option<RotaryEmbeddings>,
    ln_in_weight: Tensor<1>,
    ln_in_bias: Tensor<1>,
    blocks: Vec<TransformerBlock>,
    ln_f_weight: Tensor<1>,
    ln_f_bias: Tensor<1>,
    action_head: OutputHead,
}

pub struct NanoChatAdamState {
    token_input: TokenInputAdamState,
    wpe: Option<AdamMoments<2>>,
    canvas_state: Option<CanvasStateAdamState>,
    ln_in_weight: AdamMoments<1>,
    ln_in_bias: AdamMoments<1>,
    blocks: Vec<AdamBlockState>,
    ln_f_weight: AdamMoments<1>,
    ln_f_bias: AdamMoments<1>,
    action_head: AdamOutputHeadState,
}

impl NanoChatAdamState {
    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        self.token_input.collect_gpu_keys(keys);
        if let Some(wpe) = &self.wpe {
            wpe.collect_gpu_keys(keys);
        }
        if let Some(cs) = &self.canvas_state {
            cs.collect_gpu_keys(keys);
        }
        self.ln_in_weight.collect_gpu_keys(keys);
        self.ln_in_bias.collect_gpu_keys(keys);
        for block in &self.blocks {
            block.collect_gpu_keys(keys);
        }
        self.ln_f_weight.collect_gpu_keys(keys);
        self.ln_f_bias.collect_gpu_keys(keys);
        self.action_head.collect_gpu_keys(keys);
    }
}

#[derive(Clone)]
struct TransformerBlock {
    ln_1_weight: Tensor<1>,
    ln_1_bias: Tensor<1>,
    mixer: SequenceMixer,
    ln_attn_out_weight: Tensor<1>,
    ln_attn_out_bias: Tensor<1>,
    ln_2_weight: Tensor<1>,
    ln_2_bias: Tensor<1>,
    mlp: Mlp,
    ln_mlp_out_weight: Tensor<1>,
    ln_mlp_out_bias: Tensor<1>,
}

#[derive(Clone)]
enum SequenceMixer {
    Attention(CausalSelfAttention),
    Conv(ConvMixer),
}

#[derive(Clone)]
struct CausalSelfAttention {
    c_attn_q: Tensor<2>,
    c_attn_k: Tensor<2>,
    c_attn_v: Tensor<2>,
    c_proj: Tensor<2>,
}

#[derive(Clone)]
struct ConvMixer {
    kernels: Vec<Tensor<2>>,
    bias: Tensor<1>,
    out_proj: Tensor<2>,
}

#[derive(Clone)]
struct Mlp {
    c_fc: Tensor<2>,
    c_fc_bias: Tensor<1>,
    c_proj: Tensor<2>,
    c_proj_bias: Tensor<1>,
}

struct AdamBlockState {
    ln_1_weight: AdamMoments<1>,
    ln_1_bias: AdamMoments<1>,
    mixer: AdamMixerState,
    ln_attn_out_weight: AdamMoments<1>,
    ln_attn_out_bias: AdamMoments<1>,
    ln_2_weight: AdamMoments<1>,
    ln_2_bias: AdamMoments<1>,
    mlp: AdamMlpState,
    ln_mlp_out_weight: AdamMoments<1>,
    ln_mlp_out_bias: AdamMoments<1>,
}

enum AdamMixerState {
    Attention(AdamAttentionState),
    Conv(AdamConvState),
}

struct AdamAttentionState {
    c_attn_q: AdamMoments<2>,
    c_attn_k: AdamMoments<2>,
    c_attn_v: AdamMoments<2>,
    c_proj: AdamMoments<2>,
}

struct AdamConvState {
    kernels: Vec<AdamMoments<2>>,
    bias: AdamMoments<1>,
    out_proj: AdamMoments<2>,
}

struct AdamMlpState {
    c_fc: AdamMoments<2>,
    c_fc_bias: AdamMoments<1>,
    c_proj: AdamMoments<2>,
    c_proj_bias: AdamMoments<1>,
}

struct CanvasStateAdamState {
    cursor_x: AdamMoments<2>,
    cursor_y: AdamMoments<2>,
    pen_state: AdamMoments<2>,
}

struct AdamOutputHeadState {
    weight: AdamMoments<2>,
    bias: AdamMoments<1>,
}

impl AdamBlockState {
    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        self.ln_1_weight.collect_gpu_keys(keys);
        self.ln_1_bias.collect_gpu_keys(keys);
        self.mixer.collect_gpu_keys(keys);
        self.ln_attn_out_weight.collect_gpu_keys(keys);
        self.ln_attn_out_bias.collect_gpu_keys(keys);
        self.ln_2_weight.collect_gpu_keys(keys);
        self.ln_2_bias.collect_gpu_keys(keys);
        self.mlp.collect_gpu_keys(keys);
        self.ln_mlp_out_weight.collect_gpu_keys(keys);
        self.ln_mlp_out_bias.collect_gpu_keys(keys);
    }
}

impl AdamMixerState {
    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        match self {
            Self::Attention(attn) => attn.collect_gpu_keys(keys),
            Self::Conv(conv) => conv.collect_gpu_keys(keys),
        }
    }
}

impl AdamAttentionState {
    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        self.c_attn_q.collect_gpu_keys(keys);
        self.c_attn_k.collect_gpu_keys(keys);
        self.c_attn_v.collect_gpu_keys(keys);
        self.c_proj.collect_gpu_keys(keys);
    }
}

impl AdamConvState {
    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        for k in &self.kernels {
            k.collect_gpu_keys(keys);
        }
        self.bias.collect_gpu_keys(keys);
        self.out_proj.collect_gpu_keys(keys);
    }
}

impl AdamMlpState {
    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        self.c_fc.collect_gpu_keys(keys);
        self.c_fc_bias.collect_gpu_keys(keys);
        self.c_proj.collect_gpu_keys(keys);
        self.c_proj_bias.collect_gpu_keys(keys);
    }
}

impl CanvasStateAdamState {
    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        self.cursor_x.collect_gpu_keys(keys);
        self.cursor_y.collect_gpu_keys(keys);
        self.pen_state.collect_gpu_keys(keys);
    }
}

impl AdamOutputHeadState {
    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        self.weight.collect_gpu_keys(keys);
        self.bias.collect_gpu_keys(keys);
    }
}

impl NanoChatModel {
    pub fn new(
        device: &Device,
        rng: &mut StdRng,
        tokenizer: &StrokeTokenizer,
        config: &RuntimeConfig,
    ) -> Self {
        let vocab_size = tokenizer.vocab_size();
        let graph = Graph::new();
        let shape = ModelShape {
            block_size: config.block_size,
            n_embd: config.n_embd,
            n_head: config.n_head.max(1),
            n_kv_head: config.n_kv_head.max(1),
            n_ff: config.n_ff,
            conv_kernel_size: config.conv_kernel_size.max(1),
            eps: config.eps,
        };
        assert_eq!(
            shape.n_embd % shape.n_head,
            0,
            "NANOCHAT_N_EMBD ({}) must be divisible by NANOCHAT_N_HEAD ({})",
            shape.n_embd,
            shape.n_head
        );
        assert_eq!(
            shape.n_head % shape.n_kv_head,
            0,
            "NANOCHAT_N_HEAD ({}) must be divisible by NANOCHAT_N_KV_HEAD ({})",
            shape.n_head,
            shape.n_kv_head
        );
        if config.use_rope {
            assert_eq!(
                shape.head_dim() % 2,
                0,
                "RoPE requires an even head dimension, got {}",
                shape.head_dim()
            );
        }

        Self {
            graph: graph.clone(),
            shape,
            attention_period: config.attention_period.max(1),
            vocab_size,
            max_count: tokenizer.max_count(),
            use_rope: config.use_rope,
            rope_theta: config.rope_theta,
            use_extra_norms: config.use_extra_norms,
            token_input: TokenInputEmbeddings::new(
                &graph,
                device,
                rng,
                tokenizer.max_count(),
                shape.n_embd,
                config.init_scale,
            ),
            wpe: (!config.use_rope).then(|| {
                random_matrix(
                    &graph,
                    device,
                    rng,
                    shape.block_size,
                    shape.n_embd,
                    config.init_scale,
                )
            }),
            canvas_state: config.use_canvas_state_embeddings.then(|| {
                CanvasStateEmbeddings::new(
                    &graph,
                    device,
                    rng,
                    canvas_state_spec(tokenizer, config.block_size),
                    shape.n_embd,
                    config.init_scale,
                )
            }),
            rotary: config
                .use_rope
                .then(|| RotaryEmbeddings::new(&graph, device, shape, config.rope_theta)),
            ln_in_weight: ones(&graph, device, shape.n_embd),
            ln_in_bias: zeros(&graph, device, shape.n_embd),
            blocks: (0..config.n_layer)
                .map(|index| {
                    TransformerBlock::new(
                        &graph,
                        device,
                        rng,
                        shape,
                        config.init_scale,
                        is_attention_layer(index, config.attention_period.max(1)),
                    )
                })
                .collect(),
            ln_f_weight: ones(&graph, device, shape.n_embd),
            ln_f_bias: zeros(&graph, device, shape.n_embd),
            action_head: OutputHead::new(
                &graph,
                device,
                rng,
                shape.n_embd,
                ACTION_MODE_COUNT + ACTION_DIRECTION_COUNT + tokenizer.max_count(),
                config.init_scale,
            ),
        }
    }

    pub fn num_parameters(&self) -> usize {
        self.token_input.num_parameters()
            + self.wpe.as_ref().map_or(0, tensor_len)
            + self
                .canvas_state
                .as_ref()
                .map_or(0, CanvasStateEmbeddings::num_parameters)
            + tensor_len(&self.ln_in_weight)
            + tensor_len(&self.ln_in_bias)
            + self
                .blocks
                .iter()
                .map(TransformerBlock::num_parameters)
                .sum::<usize>()
            + tensor_len(&self.ln_f_weight)
            + tensor_len(&self.ln_f_bias)
            + self.action_head.num_parameters()
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn block_size(&self) -> usize {
        self.shape.block_size
    }

    pub fn n_embd(&self) -> usize {
        self.shape.n_embd
    }

    pub fn n_head(&self) -> usize {
        self.shape.n_head
    }

    pub fn n_kv_head(&self) -> usize {
        self.shape.n_kv_head
    }

    pub fn n_ff(&self) -> usize {
        self.shape.n_ff
    }

    pub fn n_layer(&self) -> usize {
        self.blocks.len()
    }

    pub fn eps(&self) -> f32 {
        self.shape.eps
    }

    pub fn conv_kernel_size(&self) -> usize {
        self.shape.conv_kernel_size
    }

    pub fn attention_period(&self) -> usize {
        self.attention_period
    }

    pub fn use_rope(&self) -> bool {
        self.use_rope
    }

    pub fn rope_theta(&self) -> f32 {
        self.rope_theta
    }

    pub fn use_extra_norms(&self) -> bool {
        self.use_extra_norms
    }

    pub fn canvas_state_spec(&self) -> Option<CanvasStateSpec> {
        self.canvas_state.as_ref().map(|state| state.spec)
    }

    pub fn forward(
        &self,
        mode_inputs: &RawTensor<2, u32>,
        direction_inputs: &RawTensor<2, u32>,
        count_inputs: &RawTensor<2, u32>,
        position_inputs: &RawTensor<2, u32>,
        cursor_x_inputs: &RawTensor<2, u32>,
        cursor_y_inputs: &RawTensor<2, u32>,
        pen_state_inputs: &RawTensor<2, u32>,
        causal_mask: &Tensor<3>,
    ) -> ActionLogits {
        let batch_size = mode_inputs.shape()[0];
        let mode_embeddings = self.token_input.mode.embedding(mode_inputs);
        let direction_embeddings = self.token_input.direction.embedding(direction_inputs);
        let count_embeddings = self.token_input.count.embedding(count_inputs);
        let mut x = mode_embeddings
            .add(&direction_embeddings)
            .add(&count_embeddings);
        if let Some(wpe) = &self.wpe {
            let position_embeddings = wpe.embedding(position_inputs);
            x = x.add(&position_embeddings);
        }
        if let Some(canvas_state) = &self.canvas_state {
            x = x
                .add(&canvas_state.cursor_x.embedding(cursor_x_inputs))
                .add(&canvas_state.cursor_y.embedding(cursor_y_inputs))
                .add(&canvas_state.pen_state.embedding(pen_state_inputs));
        }
        if self.use_extra_norms {
            x = x.layer_norm(&self.ln_in_weight, Some(&self.ln_in_bias), self.shape.eps);
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

        let x = x.layer_norm(&self.ln_f_weight, Some(&self.ln_f_bias), self.shape.eps);
        let action_logits = self.action_head.project(&x, batch_size, self.shape.n_embd);
        let seq_len = action_logits.shape()[1];
        let mode_end = ACTION_MODE_COUNT;
        let direction_end = mode_end + ACTION_DIRECTION_COUNT;
        let count_end = direction_end + self.max_count;
        ActionLogits {
            mode: action_logits.slice([0..batch_size, 0..seq_len, 0..mode_end]),
            direction: action_logits.slice([0..batch_size, 0..seq_len, mode_end..direction_end]),
            count: action_logits.slice([0..batch_size, 0..seq_len, direction_end..count_end]),
        }
    }

    pub async fn named_tensors(&self) -> Vec<NamedTensor> {
        let mut tensors = Vec::new();
        self.token_input.append_named_tensors(&mut tensors).await;
        if let Some(wpe) = &self.wpe {
            log_materialize_start("position_embd.weight", &wpe.shape());
            push_tensor_2d(&mut tensors, "position_embd.weight", wpe).await;
        }
        if let Some(canvas_state) = &self.canvas_state {
            canvas_state.append_named_tensors(&mut tensors).await;
        }
        log_materialize_start("input_norm.weight", &self.ln_in_weight.shape());
        push_tensor_1d(&mut tensors, "input_norm.weight", &self.ln_in_weight).await;
        log_materialize_start("input_norm.bias", &self.ln_in_bias.shape());
        push_tensor_1d(&mut tensors, "input_norm.bias", &self.ln_in_bias).await;

        for (index, block) in self.blocks.iter().enumerate() {
            println!("materializing block {index} tensors...");
            let _ = std::io::stdout().flush();
            block.append_named_tensors(index, &mut tensors).await;
        }

        log_materialize_start("output_norm.weight", &self.ln_f_weight.shape());
        push_tensor_1d(&mut tensors, "output_norm.weight", &self.ln_f_weight).await;
        log_materialize_start("output_norm.bias", &self.ln_f_bias.shape());
        push_tensor_1d(&mut tensors, "output_norm.bias", &self.ln_f_bias).await;
        self.action_head
            .append_named_tensors("output_action", &mut tensors)
            .await;
        tensors
    }
}

impl AdamWModel for NanoChatModel {
    type State = NanoChatAdamState;

    fn adamw_state(device: &Device, model: &Self) -> Self::State {
        NanoChatAdamState {
            token_input: TokenInputAdamState::new(device, &model.token_input),
            wpe: model
                .wpe
                .as_ref()
                .map(|wpe| AdamMoments::zeros_like(device, wpe)),
            canvas_state: model
                .canvas_state
                .as_ref()
                .map(|state| CanvasStateAdamState::new(device, state)),
            ln_in_weight: AdamMoments::zeros_like(device, &model.ln_in_weight),
            ln_in_bias: AdamMoments::zeros_like(device, &model.ln_in_bias),
            blocks: model
                .blocks
                .iter()
                .map(|block| AdamBlockState::new(device, block))
                .collect(),
            ln_f_weight: AdamMoments::zeros_like(device, &model.ln_f_weight),
            ln_f_bias: AdamMoments::zeros_like(device, &model.ln_f_bias),
            action_head: AdamOutputHeadState::new(device, &model.action_head),
        }
    }

    fn adamw_step(
        self,
        state: &mut Self::State,
        gradients: &Gradients,
        step: usize,
        settings: AdamWSettings,
    ) -> Self {
        // Phase 1: Extract all gradient RawTensors while the old autograd graph is
        // alive (needed for NodeId lookups). This only clones lazy handles — no GPU
        // work happens yet.
        let extracted = ExtractedModelGradients::extract(&self, gradients);
        let extracted_blocks: Vec<_> = self
            .blocks
            .iter()
            .map(|block| ExtractedBlockGradients::extract(block, gradients))
            .collect();

        // Phase 2: Drop the old autograd graph by destructuring the model and
        // immediately dropping the graph handle and all autograd tensors. This
        // releases the backward closures and their captured RawTensors, which
        // decrements GPU compute-graph reference counts and allows the resolver
        // to free intermediate buffers during the updates below.
        let NanoChatModel {
            graph: _old_graph,
            shape,
            attention_period,
            vocab_size,
            max_count,
            use_rope,
            rope_theta,
            use_extra_norms,
            token_input: _,
            wpe: _,
            canvas_state: _,
            rotary,
            ln_in_weight: _,
            ln_in_bias: _,
            blocks: _,
            ln_f_weight: _,
            ln_f_bias: _,
            action_head: _,
        } = self;
        // Old graph is now dropped — backward closures and their captured GPU
        // tensor references are freed.

        // Phase 2.5: Batch-resolve all extracted gradient and parameter-value
        // GPU tensors in a single resolver pass. The batch resolver builds one
        // shared execution graph so intermediate buffers can be freed as soon as
        // every consumer within the batch is computed. Without this, resolving
        // tensors one-by-one in adamw_update_raw keeps all shared intermediates
        // alive until every individual resolve finishes, causing GPU OOM.
        {
            let mut keys = Vec::new();
            extracted.collect_gpu_keys(&mut keys);
            for block in &extracted_blocks {
                block.collect_gpu_keys(&mut keys);
            }
            // Also collect keys from the AdamW moment tensors (m, v) since they
            // will be read during the update step.
            state.collect_gpu_keys(&mut keys);
            if !keys.is_empty() {
                // All GPU tensors share the same device; grab it from the first key's tensor.
                let device = extracted.device();
                device.resolve_batch(&keys);
            }
        }

        // Phase 3: Apply AdamW updates. Each parameter's gradient is
        // materialized and then dropped before the next one is processed,
        // so shared intermediate buffers in the compute graph can be freed
        // incrementally.
        let new_graph = Graph::new();

        let token_input = extracted
            .token_input
            .apply(&new_graph, &mut state.token_input, step, settings);

        let wpe = match (extracted.wpe, state.wpe.as_mut()) {
            (Some(Some((grad, val))), Some(moments)) => Some(Tensor::from_raw(
                &new_graph,
                adamw_update_raw(val, grad, moments, step, settings),
            )),
            (Some(None), Some(_)) => {
                Some(Tensor::from_raw(&new_graph, extracted.wpe_value.unwrap()))
            }
            (None, None) => None,
            _ => unreachable!("optimizer state does not match optional parameter"),
        };

        let canvas_state = match (extracted.canvas_state, state.canvas_state.as_mut()) {
            (Some(cs), Some(cs_state)) => Some(cs.apply(&new_graph, cs_state, step, settings)),
            (None, None) => None,
            _ => unreachable!("optimizer state does not match canvas state embeddings"),
        };

        let ln_in_weight = extracted
            .ln_in_weight
            .map(|(grad, val)| {
                Tensor::from_raw(
                    &new_graph,
                    adamw_update_raw(val, grad, &mut state.ln_in_weight, step, settings),
                )
            })
            .unwrap_or_else(|| Tensor::from_raw(&new_graph, extracted.ln_in_weight_value));

        let ln_in_bias = extracted
            .ln_in_bias
            .map(|(grad, val)| {
                Tensor::from_raw(
                    &new_graph,
                    adamw_update_raw(val, grad, &mut state.ln_in_bias, step, settings),
                )
            })
            .unwrap_or_else(|| Tensor::from_raw(&new_graph, extracted.ln_in_bias_value));

        let blocks = extracted_blocks
            .into_iter()
            .zip(state.blocks.iter_mut())
            .map(|(extracted_block, block_state)| {
                extracted_block.apply(&new_graph, block_state, step, settings)
            })
            .collect();

        let ln_f_weight = extracted
            .ln_f_weight
            .map(|(grad, val)| {
                Tensor::from_raw(
                    &new_graph,
                    adamw_update_raw(val, grad, &mut state.ln_f_weight, step, settings),
                )
            })
            .unwrap_or_else(|| Tensor::from_raw(&new_graph, extracted.ln_f_weight_value));

        let ln_f_bias = extracted
            .ln_f_bias
            .map(|(grad, val)| {
                Tensor::from_raw(
                    &new_graph,
                    adamw_update_raw(val, grad, &mut state.ln_f_bias, step, settings),
                )
            })
            .unwrap_or_else(|| Tensor::from_raw(&new_graph, extracted.ln_f_bias_value));

        let action_head =
            extracted
                .action_head
                .apply(&new_graph, &mut state.action_head, step, settings);

        let rotary = rotary.map(|cache| cache.into_graph(&new_graph));

        NanoChatModel {
            graph: new_graph,
            shape,
            attention_period,
            vocab_size,
            max_count,
            use_rope,
            rope_theta,
            use_extra_norms,
            token_input,
            wpe,
            canvas_state,
            rotary,
            ln_in_weight,
            ln_in_bias,
            blocks,
            ln_f_weight,
            ln_f_bias,
            action_head,
        }
    }
}

impl CanvasStateEmbeddings {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        spec: CanvasStateSpec,
        n_embd: usize,
        init_scale: f32,
    ) -> Self {
        Self {
            spec,
            cursor_x: random_matrix(
                graph,
                device,
                rng,
                spec.coordinate_vocab_size,
                n_embd,
                init_scale,
            ),
            cursor_y: random_matrix(
                graph,
                device,
                rng,
                spec.coordinate_vocab_size,
                n_embd,
                init_scale,
            ),
            pen_state: random_matrix(graph, device, rng, 2, n_embd, init_scale),
        }
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.cursor_x) + tensor_len(&self.cursor_y) + tensor_len(&self.pen_state)
    }

    async fn append_named_tensors(&self, tensors: &mut Vec<NamedTensor>) {
        log_materialize_start("cursor_x_embd.weight", &self.cursor_x.shape());
        push_tensor_2d(tensors, "cursor_x_embd.weight", &self.cursor_x).await;
        log_materialize_start("cursor_y_embd.weight", &self.cursor_y.shape());
        push_tensor_2d(tensors, "cursor_y_embd.weight", &self.cursor_y).await;
        log_materialize_start("pen_state_embd.weight", &self.pen_state.shape());
        push_tensor_2d(tensors, "pen_state_embd.weight", &self.pen_state).await;
    }
}

impl CanvasStateAdamState {
    fn new(device: &Device, state: &CanvasStateEmbeddings) -> Self {
        Self {
            cursor_x: AdamMoments::zeros_like(device, &state.cursor_x),
            cursor_y: AdamMoments::zeros_like(device, &state.cursor_y),
            pen_state: AdamMoments::zeros_like(device, &state.pen_state),
        }
    }
}

impl TokenInputEmbeddings {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        max_count: usize,
        n_embd: usize,
        init_scale: f32,
    ) -> Self {
        Self {
            mode: random_matrix(graph, device, rng, INPUT_MODE_COUNT, n_embd, init_scale),
            direction: random_matrix(
                graph,
                device,
                rng,
                ACTION_DIRECTION_COUNT,
                n_embd,
                init_scale,
            ),
            count: random_matrix(graph, device, rng, max_count, n_embd, init_scale),
        }
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.mode) + tensor_len(&self.direction) + tensor_len(&self.count)
    }

    async fn append_named_tensors(&self, tensors: &mut Vec<NamedTensor>) {
        log_materialize_start("token_input_mode.weight", &self.mode.shape());
        push_tensor_2d(tensors, "token_input_mode.weight", &self.mode).await;
        log_materialize_start("token_input_direction.weight", &self.direction.shape());
        push_tensor_2d(tensors, "token_input_direction.weight", &self.direction).await;
        log_materialize_start("token_input_count.weight", &self.count.shape());
        push_tensor_2d(tensors, "token_input_count.weight", &self.count).await;
    }
}

impl OutputHead {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        input_dim: usize,
        output_dim: usize,
        init_scale: f32,
    ) -> Self {
        Self {
            weight: random_matrix(graph, device, rng, input_dim, output_dim, init_scale),
            bias: zeros(graph, device, output_dim),
        }
    }

    fn project(&self, x: &Tensor<3>, batch_size: usize, input_dim: usize) -> Tensor<3> {
        let seq_len = x.shape()[1];
        x.mat_mul(
            &self
                .weight
                .broadcast_as([batch_size, input_dim, self.bias.shape()[0]]),
        )
        .add(
            &self
                .bias
                .broadcast_as([batch_size, seq_len, self.bias.shape()[0]]),
        )
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.weight) + tensor_len(&self.bias)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        log_materialize_start(&format!("{prefix}.weight"), &self.weight.shape());
        push_tensor_2d(tensors, &format!("{prefix}.weight"), &self.weight).await;
        log_materialize_start(&format!("{prefix}.bias"), &self.bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.bias"), &self.bias).await;
    }
}

impl AdamOutputHeadState {
    fn new(device: &Device, head: &OutputHead) -> Self {
        Self {
            weight: AdamMoments::zeros_like(device, &head.weight),
            bias: AdamMoments::zeros_like(device, &head.bias),
        }
    }
}

impl TransformerBlock {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        shape: ModelShape,
        init_scale: f32,
        is_attention_layer: bool,
    ) -> Self {
        Self {
            ln_1_weight: ones(graph, device, shape.n_embd),
            ln_1_bias: zeros(graph, device, shape.n_embd),
            mixer: if is_attention_layer {
                SequenceMixer::Attention(CausalSelfAttention::new(
                    graph, device, rng, shape, init_scale,
                ))
            } else {
                SequenceMixer::Conv(ConvMixer::new(graph, device, rng, shape, init_scale))
            },
            ln_attn_out_weight: ones(graph, device, shape.n_embd),
            ln_attn_out_bias: zeros(graph, device, shape.n_embd),
            ln_2_weight: ones(graph, device, shape.n_embd),
            ln_2_bias: zeros(graph, device, shape.n_embd),
            mlp: Mlp::new(graph, device, rng, shape, init_scale),
            ln_mlp_out_weight: ones(graph, device, shape.n_embd),
            ln_mlp_out_bias: zeros(graph, device, shape.n_embd),
        }
    }

    fn forward(
        &self,
        x: Tensor<3>,
        causal_mask: &Tensor<3>,
        batch_size: usize,
        shape: ModelShape,
        rotary: Option<&RotaryEmbeddings>,
        use_extra_norms: bool,
    ) -> Tensor<3> {
        let attn_input = x.layer_norm(&self.ln_1_weight, Some(&self.ln_1_bias), shape.eps);
        let mut attn_output =
            self.mixer
                .forward(&attn_input, causal_mask, batch_size, shape, rotary);
        if use_extra_norms {
            attn_output = attn_output.layer_norm(
                &self.ln_attn_out_weight,
                Some(&self.ln_attn_out_bias),
                shape.eps,
            );
        }
        let x = x.add(&attn_output);
        let mlp_input = x.layer_norm(&self.ln_2_weight, Some(&self.ln_2_bias), shape.eps);
        let mut mlp_output = self.mlp.forward(&mlp_input, batch_size, shape);
        if use_extra_norms {
            mlp_output = mlp_output.layer_norm(
                &self.ln_mlp_out_weight,
                Some(&self.ln_mlp_out_bias),
                shape.eps,
            );
        }
        x.add(&mlp_output)
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.ln_1_weight)
            + tensor_len(&self.ln_1_bias)
            + self.mixer.num_parameters()
            + tensor_len(&self.ln_attn_out_weight)
            + tensor_len(&self.ln_attn_out_bias)
            + tensor_len(&self.ln_2_weight)
            + tensor_len(&self.ln_2_bias)
            + self.mlp.num_parameters()
            + tensor_len(&self.ln_mlp_out_weight)
            + tensor_len(&self.ln_mlp_out_bias)
    }

    async fn append_named_tensors(&self, index: usize, tensors: &mut Vec<NamedTensor>) {
        let prefix = format!("blk.{index}");
        log_materialize_start(&format!("{prefix}.ln_1.weight"), &self.ln_1_weight.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_1.weight"), &self.ln_1_weight).await;
        log_materialize_start(&format!("{prefix}.ln_1.bias"), &self.ln_1_bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_1.bias"), &self.ln_1_bias).await;
        self.mixer.append_named_tensors(&prefix, tensors).await;
        log_materialize_start(
            &format!("{prefix}.attn_out_norm.weight"),
            &self.ln_attn_out_weight.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.attn_out_norm.weight"),
            &self.ln_attn_out_weight,
        )
        .await;
        log_materialize_start(
            &format!("{prefix}.attn_out_norm.bias"),
            &self.ln_attn_out_bias.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.attn_out_norm.bias"),
            &self.ln_attn_out_bias,
        )
        .await;
        log_materialize_start(&format!("{prefix}.ln_2.weight"), &self.ln_2_weight.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_2.weight"), &self.ln_2_weight).await;
        log_materialize_start(&format!("{prefix}.ln_2.bias"), &self.ln_2_bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.ln_2.bias"), &self.ln_2_bias).await;
        self.mlp.append_named_tensors(&prefix, tensors).await;
        log_materialize_start(
            &format!("{prefix}.mlp_out_norm.weight"),
            &self.ln_mlp_out_weight.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.mlp_out_norm.weight"),
            &self.ln_mlp_out_weight,
        )
        .await;
        log_materialize_start(
            &format!("{prefix}.mlp_out_norm.bias"),
            &self.ln_mlp_out_bias.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.mlp_out_norm.bias"),
            &self.ln_mlp_out_bias,
        )
        .await;
    }
}

impl CausalSelfAttention {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        shape: ModelShape,
        init_scale: f32,
    ) -> Self {
        Self {
            c_attn_q: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
            c_attn_k: random_matrix(graph, device, rng, shape.n_embd, shape.kv_dim(), init_scale),
            c_attn_v: random_matrix(graph, device, rng, shape.n_embd, shape.kv_dim(), init_scale),
            c_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
        }
    }

    fn forward(
        &self,
        x: &Tensor<3>,
        causal_mask: &Tensor<3>,
        batch_size: usize,
        shape: ModelShape,
        rotary: Option<&RotaryEmbeddings>,
    ) -> Tensor<3> {
        let head_dim = shape.head_dim();
        let q = x.mat_mul(
            &self
                .c_attn_q
                .broadcast_as([batch_size, shape.n_embd, shape.n_embd]),
        );
        let k = x.mat_mul(
            &self
                .c_attn_k
                .broadcast_as([batch_size, shape.n_embd, shape.kv_dim()]),
        );
        let v = x.mat_mul(
            &self
                .c_attn_v
                .broadcast_as([batch_size, shape.n_embd, shape.kv_dim()]),
        );
        let q_heads = (0..shape.n_head)
            .map(|head| {
                let start = head * head_dim;
                let end = start + head_dim;
                let head = q.slice([0..batch_size, 0..q.shape()[1], start..end]);
                rotary.map_or(head.clone(), |cache| cache.apply(&head))
            })
            .collect::<Vec<_>>();
        let k_heads = (0..shape.n_kv_head)
            .map(|head| {
                let start = head * head_dim;
                let end = start + head_dim;
                let head = k.slice([0..batch_size, 0..k.shape()[1], start..end]);
                rotary.map_or(head.clone(), |cache| cache.apply(&head))
            })
            .collect::<Vec<_>>();
        let v_heads = (0..shape.n_kv_head)
            .map(|head| {
                let start = head * head_dim;
                let end = start + head_dim;
                v.slice([0..batch_size, 0..v.shape()[1], start..end])
            })
            .collect::<Vec<_>>();
        let heads = (0..shape.n_head)
            .map(|head| {
                let kv_head = head / shape.num_kv_groups();
                let q_head = q_heads[head].clone();
                let k_head = k_heads[kv_head].clone();
                let v_head = v_heads[kv_head].clone();
                let scores = q_head
                    .mat_mul(&k_head.transpose(1, 2))
                    .div_scalar((head_dim as f32).sqrt());
                let masked = scores.add(causal_mask);
                let weights_exp = masked.exp();
                let attention =
                    weights_exp.div(&weights_exp.sum_keepdim(2).broadcast_as(weights_exp.shape()));
                attention.mat_mul(&v_head)
            })
            .collect::<Vec<_>>();
        Tensor::cat(heads, 2).mat_mul(&self.c_proj.broadcast_as([
            batch_size,
            shape.n_embd,
            shape.n_embd,
        ]))
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.c_attn_q)
            + tensor_len(&self.c_attn_k)
            + tensor_len(&self.c_attn_v)
            + tensor_len(&self.c_proj)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        log_materialize_start(&format!("{prefix}.attn_q.weight"), &self.c_attn_q.shape());
        push_tensor_2d(tensors, &format!("{prefix}.attn_q.weight"), &self.c_attn_q).await;
        log_materialize_start(&format!("{prefix}.attn_k.weight"), &self.c_attn_k.shape());
        push_tensor_2d(tensors, &format!("{prefix}.attn_k.weight"), &self.c_attn_k).await;
        log_materialize_start(&format!("{prefix}.attn_v.weight"), &self.c_attn_v.shape());
        push_tensor_2d(tensors, &format!("{prefix}.attn_v.weight"), &self.c_attn_v).await;
        log_materialize_start(&format!("{prefix}.attn_proj.weight"), &self.c_proj.shape());
        push_tensor_2d(tensors, &format!("{prefix}.attn_proj.weight"), &self.c_proj).await;
    }
}

impl ConvMixer {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        shape: ModelShape,
        init_scale: f32,
    ) -> Self {
        Self {
            kernels: (0..shape.conv_kernel_size)
                .map(|_| random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale))
                .collect(),
            bias: zeros(graph, device, shape.n_embd),
            out_proj: random_matrix(graph, device, rng, shape.n_embd, shape.n_embd, init_scale),
        }
    }

    fn forward(&self, x: &Tensor<3>, batch_size: usize, shape: ModelShape) -> Tensor<3> {
        let seq_len = x.shape()[1];
        let mut mixed = Tensor::zeros(&x.graph(), &x.device(), [batch_size, seq_len, shape.n_embd]);

        for (offset, kernel) in self.kernels.iter().enumerate() {
            let shifted = causal_shift_autograd(x, offset);
            let projected =
                shifted.mat_mul(&kernel.broadcast_as([batch_size, shape.n_embd, shape.n_embd]));
            mixed = mixed.add(&projected);
        }

        mixed
            .add(&self.bias.broadcast_as([batch_size, seq_len, shape.n_embd]))
            .relu()
            .mat_mul(
                &self
                    .out_proj
                    .broadcast_as([batch_size, shape.n_embd, shape.n_embd]),
            )
    }

    fn num_parameters(&self) -> usize {
        self.kernels.iter().map(tensor_len).sum::<usize>()
            + tensor_len(&self.bias)
            + tensor_len(&self.out_proj)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        for (index, kernel) in self.kernels.iter().enumerate() {
            log_materialize_start(
                &format!("{prefix}.conv_kernel.{index}.weight"),
                &kernel.shape(),
            );
            push_tensor_2d(
                tensors,
                &format!("{prefix}.conv_kernel.{index}.weight"),
                kernel,
            )
            .await;
        }
        log_materialize_start(&format!("{prefix}.conv_bias"), &self.bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.conv_bias"), &self.bias).await;
        log_materialize_start(
            &format!("{prefix}.conv_proj.weight"),
            &self.out_proj.shape(),
        );
        push_tensor_2d(
            tensors,
            &format!("{prefix}.conv_proj.weight"),
            &self.out_proj,
        )
        .await;
    }
}

impl SequenceMixer {
    fn forward(
        &self,
        x: &Tensor<3>,
        causal_mask: &Tensor<3>,
        batch_size: usize,
        shape: ModelShape,
        rotary: Option<&RotaryEmbeddings>,
    ) -> Tensor<3> {
        match self {
            SequenceMixer::Attention(attn) => {
                attn.forward(x, causal_mask, batch_size, shape, rotary)
            }
            SequenceMixer::Conv(conv) => conv.forward(x, batch_size, shape),
        }
    }

    fn num_parameters(&self) -> usize {
        match self {
            SequenceMixer::Attention(attn) => attn.num_parameters(),
            SequenceMixer::Conv(conv) => conv.num_parameters(),
        }
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        match self {
            SequenceMixer::Attention(attn) => attn.append_named_tensors(prefix, tensors).await,
            SequenceMixer::Conv(conv) => conv.append_named_tensors(prefix, tensors).await,
        }
    }
}

impl RotaryEmbeddings {
    fn new(graph: &Graph, device: &Device, shape: ModelShape, rope_theta: f32) -> Self {
        let inverse_frequency = base_inverse_frequency(shape.head_dim(), rope_theta);
        let mut cos = vec![vec![0.0; inverse_frequency.len()]; shape.block_size];
        let mut sin = vec![vec![0.0; inverse_frequency.len()]; shape.block_size];
        for position in 0..shape.block_size {
            for (index, frequency) in inverse_frequency.iter().enumerate() {
                let angle = position as f32 * frequency;
                cos[position][index] = angle.cos();
                sin[position][index] = angle.sin();
            }
        }

        Self {
            cos: Tensor::constant_from_raw(graph, RawTensor::new(device, &cos)),
            sin: Tensor::constant_from_raw(graph, RawTensor::new(device, &sin)),
        }
    }

    fn apply(&self, x: &Tensor<3>) -> Tensor<3> {
        let [batch_size, sequence_length, head_dim] = x.shape();
        let half_dim = head_dim / 2;
        let cos = self
            .cos
            .slice([0..sequence_length, 0..half_dim])
            .unsqueeze(0)
            .broadcast_as([batch_size, sequence_length, half_dim]);
        let sin = self
            .sin
            .slice([0..sequence_length, 0..half_dim])
            .unsqueeze(0)
            .broadcast_as([batch_size, sequence_length, half_dim]);
        let first_half = x.slice([0..batch_size, 0..sequence_length, 0..half_dim]);
        let second_half = x.slice([0..batch_size, 0..sequence_length, half_dim..head_dim]);
        let rotated_first = first_half.mul(&cos).sub(&second_half.mul(&sin));
        let rotated_second = second_half.mul(&cos).add(&first_half.mul(&sin));
        Tensor::cat(vec![rotated_first, rotated_second], 2)
    }

    fn into_graph(self, graph: &Graph) -> Self {
        Self {
            cos: regraph_constant_tensor(graph, self.cos),
            sin: regraph_constant_tensor(graph, self.sin),
        }
    }
}

impl Mlp {
    fn new(
        graph: &Graph,
        device: &Device,
        rng: &mut StdRng,
        shape: ModelShape,
        init_scale: f32,
    ) -> Self {
        Self {
            c_fc: random_matrix(graph, device, rng, shape.n_embd, shape.n_ff, init_scale),
            c_fc_bias: zeros(graph, device, shape.n_ff),
            c_proj: random_matrix(graph, device, rng, shape.n_ff, shape.n_embd, init_scale),
            c_proj_bias: zeros(graph, device, shape.n_embd),
        }
    }

    fn forward(&self, x: &Tensor<3>, batch_size: usize, shape: ModelShape) -> Tensor<3> {
        let seq_len = x.shape()[1];
        let hidden = x
            .mat_mul(
                &self
                    .c_fc
                    .broadcast_as([batch_size, shape.n_embd, shape.n_ff]),
            )
            .add(
                &self
                    .c_fc_bias
                    .broadcast_as([batch_size, seq_len, shape.n_ff]),
            )
            .relu();

        hidden
            .mat_mul(
                &self
                    .c_proj
                    .broadcast_as([batch_size, shape.n_ff, shape.n_embd]),
            )
            .add(
                &self
                    .c_proj_bias
                    .broadcast_as([batch_size, seq_len, shape.n_embd]),
            )
    }

    fn num_parameters(&self) -> usize {
        tensor_len(&self.c_fc)
            + tensor_len(&self.c_fc_bias)
            + tensor_len(&self.c_proj)
            + tensor_len(&self.c_proj_bias)
    }

    async fn append_named_tensors(&self, prefix: &str, tensors: &mut Vec<NamedTensor>) {
        log_materialize_start(&format!("{prefix}.mlp_fc.weight"), &self.c_fc.shape());
        push_tensor_2d(tensors, &format!("{prefix}.mlp_fc.weight"), &self.c_fc).await;
        log_materialize_start(&format!("{prefix}.mlp_fc.bias"), &self.c_fc_bias.shape());
        push_tensor_1d(tensors, &format!("{prefix}.mlp_fc.bias"), &self.c_fc_bias).await;
        log_materialize_start(&format!("{prefix}.mlp_proj.weight"), &self.c_proj.shape());
        push_tensor_2d(tensors, &format!("{prefix}.mlp_proj.weight"), &self.c_proj).await;
        log_materialize_start(
            &format!("{prefix}.mlp_proj.bias"),
            &self.c_proj_bias.shape(),
        );
        push_tensor_1d(
            tensors,
            &format!("{prefix}.mlp_proj.bias"),
            &self.c_proj_bias,
        )
        .await;
    }
}

impl AdamBlockState {
    fn new(device: &Device, block: &TransformerBlock) -> Self {
        Self {
            ln_1_weight: AdamMoments::zeros_like(device, &block.ln_1_weight),
            ln_1_bias: AdamMoments::zeros_like(device, &block.ln_1_bias),
            mixer: AdamMixerState::new(device, &block.mixer),
            ln_attn_out_weight: AdamMoments::zeros_like(device, &block.ln_attn_out_weight),
            ln_attn_out_bias: AdamMoments::zeros_like(device, &block.ln_attn_out_bias),
            ln_2_weight: AdamMoments::zeros_like(device, &block.ln_2_weight),
            ln_2_bias: AdamMoments::zeros_like(device, &block.ln_2_bias),
            mlp: AdamMlpState::new(device, &block.mlp),
            ln_mlp_out_weight: AdamMoments::zeros_like(device, &block.ln_mlp_out_weight),
            ln_mlp_out_bias: AdamMoments::zeros_like(device, &block.ln_mlp_out_bias),
        }
    }
}

impl AdamMixerState {
    fn new(device: &Device, mixer: &SequenceMixer) -> Self {
        match mixer {
            SequenceMixer::Attention(attn) => {
                Self::Attention(AdamAttentionState::new(device, attn))
            }
            SequenceMixer::Conv(conv) => Self::Conv(AdamConvState::new(device, conv)),
        }
    }
}

impl AdamAttentionState {
    fn new(device: &Device, attn: &CausalSelfAttention) -> Self {
        Self {
            c_attn_q: AdamMoments::zeros_like(device, &attn.c_attn_q),
            c_attn_k: AdamMoments::zeros_like(device, &attn.c_attn_k),
            c_attn_v: AdamMoments::zeros_like(device, &attn.c_attn_v),
            c_proj: AdamMoments::zeros_like(device, &attn.c_proj),
        }
    }
}

impl AdamConvState {
    fn new(device: &Device, conv: &ConvMixer) -> Self {
        Self {
            kernels: conv
                .kernels
                .iter()
                .map(|kernel| AdamMoments::zeros_like(device, kernel))
                .collect(),
            bias: AdamMoments::zeros_like(device, &conv.bias),
            out_proj: AdamMoments::zeros_like(device, &conv.out_proj),
        }
    }
}

impl AdamMlpState {
    fn new(device: &Device, mlp: &Mlp) -> Self {
        Self {
            c_fc: AdamMoments::zeros_like(device, &mlp.c_fc),
            c_fc_bias: AdamMoments::zeros_like(device, &mlp.c_fc_bias),
            c_proj: AdamMoments::zeros_like(device, &mlp.c_proj),
            c_proj_bias: AdamMoments::zeros_like(device, &mlp.c_proj_bias),
        }
    }
}

fn random_matrix(
    graph: &Graph,
    device: &Device,
    rng: &mut StdRng,
    rows: usize,
    cols: usize,
    scale: f32,
) -> Tensor<2> {
    let data: Vec<Vec<f32>> = (0..rows)
        .map(|_| (0..cols).map(|_| rng.random_range(-scale..scale)).collect())
        .collect();
    Tensor::new(graph, device, &data)
}

fn ones(graph: &Graph, device: &Device, len: usize) -> Tensor<1> {
    Tensor::new(graph, device, &vec![1.0; len])
}

fn zeros(graph: &Graph, device: &Device, len: usize) -> Tensor<1> {
    Tensor::new(graph, device, &vec![0.0; len])
}

fn tensor_len<const R: usize>(tensor: &Tensor<R>) -> usize {
    tensor.shape().iter().product()
}

fn regraph_constant_tensor<const R: usize>(graph: &Graph, tensor: Tensor<R>) -> Tensor<R> {
    Tensor::constant_from_raw(graph, tensor.into_raw())
}

async fn push_tensor_1d(tensors: &mut Vec<NamedTensor>, name: &str, tensor: &Tensor<1>) {
    let start = Instant::now();
    let values = tensor.raw().clone().as_slice().await.unwrap().to_vec1();
    println!(
        "materialized tensor {name} shape={:?} elems={} bytes={} in {:.2?}",
        tensor.shape(),
        values.len(),
        values.len() * std::mem::size_of::<f32>(),
        start.elapsed(),
    );
    tensors.push(NamedTensor {
        name: name.to_string(),
        shape: tensor.shape().iter().map(|&dim| dim as u32).collect(),
        values,
    });
}

async fn push_tensor_2d(tensors: &mut Vec<NamedTensor>, name: &str, tensor: &Tensor<2>) {
    let start = Instant::now();
    let values = tensor.raw().clone().as_slice().await.unwrap().to_vec2();
    let flat = values.into_iter().flatten().collect::<Vec<_>>();
    println!(
        "materialized tensor {name} shape={:?} elems={} bytes={} in {:.2?}",
        tensor.shape(),
        flat.len(),
        flat.len() * std::mem::size_of::<f32>(),
        start.elapsed(),
    );
    tensors.push(NamedTensor {
        name: name.to_string(),
        shape: tensor.shape().iter().map(|&dim| dim as u32).collect(),
        values: flat,
    });
}

fn log_materialize_start<const R: usize>(name: &str, shape: &[usize; R]) {
    let elements: usize = shape.iter().product();
    println!(
        "starting materialize {name} shape={shape:?} elems={} bytes={}",
        elements,
        elements * std::mem::size_of::<f32>(),
    );
    let _ = std::io::stdout().flush();
}

fn is_attention_layer(index: usize, attention_period: usize) -> bool {
    (index + 1) % attention_period.max(1) == 0
}

fn causal_shift_autograd(x: &Tensor<3>, offset: usize) -> Tensor<3> {
    if offset == 0 {
        return x.clone();
    }

    let [batch_size, seq_len, n_embd] = x.shape();
    if offset >= seq_len {
        return Tensor::zeros(&x.graph(), &x.device(), [batch_size, seq_len, n_embd]);
    }

    let prefix = Tensor::zeros(&x.graph(), &x.device(), [batch_size, offset, n_embd]);
    let shifted = x.slice([0..batch_size, 0..seq_len - offset, 0..n_embd]);
    Tensor::cat(vec![prefix, shifted], 1)
}

impl ModelShape {
    fn head_dim(self) -> usize {
        self.n_embd / self.n_head.max(1)
    }

    fn kv_dim(self) -> usize {
        self.n_kv_head * self.head_dim()
    }

    fn num_kv_groups(self) -> usize {
        self.n_head / self.n_kv_head.max(1)
    }
}

// ---------------------------------------------------------------------------
// GPU batch-resolve helpers
// ---------------------------------------------------------------------------

/// Collect the GPU compute-graph key from a `RawTensor`, if it lives on GPU.
fn collect_tensor_gpu_key<const R: usize>(
    tensor: &fusor::Tensor<R, f32>,
    keys: &mut Vec<NodeIndex>,
) {
    if let Some(key) = tensor.gpu_key() {
        keys.push(key);
    }
}

/// Collect GPU keys from an optional (gradient, param_value) pair.
fn collect_grad_pair_gpu_keys<const R: usize>(
    pair: &GradPair<R>,
    keys: &mut Vec<NodeIndex>,
) {
    if let Some((grad, val)) = pair {
        collect_tensor_gpu_key(grad, keys);
        collect_tensor_gpu_key(val, keys);
    }
}


// ---------------------------------------------------------------------------
// Extracted-gradient structs for the two-phase optimizer step.
//
// Phase 1 extracts lazy (RawTensor, RawTensor) pairs keyed by the old autograd
// NodeIds.  Phase 2 drops the old autograd graph (freeing backward-closure
// captured GPU buffers) then resolves the extracted gradients.
// ---------------------------------------------------------------------------

type GradPair<const R: usize> = Option<(fusor::Tensor<R, f32>, fusor::Tensor<R, f32>)>;

struct ExtractedModelGradients {
    token_input: ExtractedTokenInputGradients,
    wpe: Option<GradPair<2>>,
    wpe_value: Option<fusor::Tensor<2, f32>>,
    canvas_state: Option<ExtractedCanvasStateGradients>,
    ln_in_weight: GradPair<1>,
    ln_in_weight_value: fusor::Tensor<1, f32>,
    ln_in_bias: GradPair<1>,
    ln_in_bias_value: fusor::Tensor<1, f32>,
    ln_f_weight: GradPair<1>,
    ln_f_weight_value: fusor::Tensor<1, f32>,
    ln_f_bias: GradPair<1>,
    ln_f_bias_value: fusor::Tensor<1, f32>,
    action_head: ExtractedOutputHeadGradients,
}

impl ExtractedModelGradients {
    fn extract(model: &NanoChatModel, gradients: &Gradients) -> Self {
        Self {
            token_input: ExtractedTokenInputGradients::extract(&model.token_input, gradients),
            wpe: model
                .wpe
                .as_ref()
                .map(|wpe| extract_gradient(wpe, gradients)),
            wpe_value: model.wpe.as_ref().map(|wpe| wpe.raw().clone()),
            canvas_state: model
                .canvas_state
                .as_ref()
                .map(|cs| ExtractedCanvasStateGradients::extract(cs, gradients)),
            ln_in_weight: extract_gradient(&model.ln_in_weight, gradients),
            ln_in_weight_value: model.ln_in_weight.raw().clone(),
            ln_in_bias: extract_gradient(&model.ln_in_bias, gradients),
            ln_in_bias_value: model.ln_in_bias.raw().clone(),
            ln_f_weight: extract_gradient(&model.ln_f_weight, gradients),
            ln_f_weight_value: model.ln_f_weight.raw().clone(),
            ln_f_bias: extract_gradient(&model.ln_f_bias, gradients),
            ln_f_bias_value: model.ln_f_bias.raw().clone(),
            action_head: ExtractedOutputHeadGradients::extract(&model.action_head, gradients),
        }
    }

    fn device(&self) -> Device {
        self.token_input.mode_value.device()
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        self.token_input.collect_gpu_keys(keys);
        if let Some(wpe) = &self.wpe {
            collect_grad_pair_gpu_keys(wpe, keys);
        }
        if let Some(val) = &self.wpe_value {
            collect_tensor_gpu_key(val, keys);
        }
        if let Some(cs) = &self.canvas_state {
            cs.collect_gpu_keys(keys);
        }
        collect_grad_pair_gpu_keys(&self.ln_in_weight, keys);
        collect_tensor_gpu_key(&self.ln_in_weight_value, keys);
        collect_grad_pair_gpu_keys(&self.ln_in_bias, keys);
        collect_tensor_gpu_key(&self.ln_in_bias_value, keys);
        collect_grad_pair_gpu_keys(&self.ln_f_weight, keys);
        collect_tensor_gpu_key(&self.ln_f_weight_value, keys);
        collect_grad_pair_gpu_keys(&self.ln_f_bias, keys);
        collect_tensor_gpu_key(&self.ln_f_bias_value, keys);
        self.action_head.collect_gpu_keys(keys);
    }
}

struct ExtractedTokenInputGradients {
    mode: GradPair<2>,
    mode_value: fusor::Tensor<2, f32>,
    direction: GradPair<2>,
    direction_value: fusor::Tensor<2, f32>,
    count: GradPair<2>,
    count_value: fusor::Tensor<2, f32>,
}

impl ExtractedTokenInputGradients {
    fn extract(embeddings: &TokenInputEmbeddings, gradients: &Gradients) -> Self {
        Self {
            mode: extract_gradient(&embeddings.mode, gradients),
            mode_value: embeddings.mode.raw().clone(),
            direction: extract_gradient(&embeddings.direction, gradients),
            direction_value: embeddings.direction.raw().clone(),
            count: extract_gradient(&embeddings.count, gradients),
            count_value: embeddings.count.raw().clone(),
        }
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        collect_grad_pair_gpu_keys(&self.mode, keys);
        collect_tensor_gpu_key(&self.mode_value, keys);
        collect_grad_pair_gpu_keys(&self.direction, keys);
        collect_tensor_gpu_key(&self.direction_value, keys);
        collect_grad_pair_gpu_keys(&self.count, keys);
        collect_tensor_gpu_key(&self.count_value, keys);
    }

    fn apply(
        self,
        graph: &Graph,
        state: &mut TokenInputAdamState,
        step: usize,
        settings: AdamWSettings,
    ) -> TokenInputEmbeddings {
        TokenInputEmbeddings {
            mode: apply_extracted(
                graph,
                self.mode,
                self.mode_value,
                &mut state.mode,
                step,
                settings,
            ),
            direction: apply_extracted(
                graph,
                self.direction,
                self.direction_value,
                &mut state.direction,
                step,
                settings,
            ),
            count: apply_extracted(
                graph,
                self.count,
                self.count_value,
                &mut state.count,
                step,
                settings,
            ),
        }
    }
}

struct ExtractedCanvasStateGradients {
    cursor_x: GradPair<2>,
    cursor_x_value: fusor::Tensor<2, f32>,
    cursor_y: GradPair<2>,
    cursor_y_value: fusor::Tensor<2, f32>,
    pen_state: GradPair<2>,
    pen_state_value: fusor::Tensor<2, f32>,
    spec: CanvasStateSpec,
}

impl ExtractedCanvasStateGradients {
    fn extract(cs: &CanvasStateEmbeddings, gradients: &Gradients) -> Self {
        Self {
            cursor_x: extract_gradient(&cs.cursor_x, gradients),
            cursor_x_value: cs.cursor_x.raw().clone(),
            cursor_y: extract_gradient(&cs.cursor_y, gradients),
            cursor_y_value: cs.cursor_y.raw().clone(),
            pen_state: extract_gradient(&cs.pen_state, gradients),
            pen_state_value: cs.pen_state.raw().clone(),
            spec: cs.spec,
        }
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        collect_grad_pair_gpu_keys(&self.cursor_x, keys);
        collect_tensor_gpu_key(&self.cursor_x_value, keys);
        collect_grad_pair_gpu_keys(&self.cursor_y, keys);
        collect_tensor_gpu_key(&self.cursor_y_value, keys);
        collect_grad_pair_gpu_keys(&self.pen_state, keys);
        collect_tensor_gpu_key(&self.pen_state_value, keys);
    }

    fn apply(
        self,
        graph: &Graph,
        state: &mut CanvasStateAdamState,
        step: usize,
        settings: AdamWSettings,
    ) -> CanvasStateEmbeddings {
        CanvasStateEmbeddings {
            spec: self.spec,
            cursor_x: apply_extracted(
                graph,
                self.cursor_x,
                self.cursor_x_value,
                &mut state.cursor_x,
                step,
                settings,
            ),
            cursor_y: apply_extracted(
                graph,
                self.cursor_y,
                self.cursor_y_value,
                &mut state.cursor_y,
                step,
                settings,
            ),
            pen_state: apply_extracted(
                graph,
                self.pen_state,
                self.pen_state_value,
                &mut state.pen_state,
                step,
                settings,
            ),
        }
    }
}

struct ExtractedOutputHeadGradients {
    weight: GradPair<2>,
    weight_value: fusor::Tensor<2, f32>,
    bias: GradPair<1>,
    bias_value: fusor::Tensor<1, f32>,
}

impl ExtractedOutputHeadGradients {
    fn extract(head: &OutputHead, gradients: &Gradients) -> Self {
        Self {
            weight: extract_gradient(&head.weight, gradients),
            weight_value: head.weight.raw().clone(),
            bias: extract_gradient(&head.bias, gradients),
            bias_value: head.bias.raw().clone(),
        }
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        collect_grad_pair_gpu_keys(&self.weight, keys);
        collect_tensor_gpu_key(&self.weight_value, keys);
        collect_grad_pair_gpu_keys(&self.bias, keys);
        collect_tensor_gpu_key(&self.bias_value, keys);
    }

    fn apply(
        self,
        graph: &Graph,
        state: &mut AdamOutputHeadState,
        step: usize,
        settings: AdamWSettings,
    ) -> OutputHead {
        OutputHead {
            weight: apply_extracted(
                graph,
                self.weight,
                self.weight_value,
                &mut state.weight,
                step,
                settings,
            ),
            bias: apply_extracted(
                graph,
                self.bias,
                self.bias_value,
                &mut state.bias,
                step,
                settings,
            ),
        }
    }
}

struct ExtractedBlockGradients {
    ln_1_weight: GradPair<1>,
    ln_1_weight_value: fusor::Tensor<1, f32>,
    ln_1_bias: GradPair<1>,
    ln_1_bias_value: fusor::Tensor<1, f32>,
    mixer: ExtractedMixerGradients,
    ln_attn_out_weight: GradPair<1>,
    ln_attn_out_weight_value: fusor::Tensor<1, f32>,
    ln_attn_out_bias: GradPair<1>,
    ln_attn_out_bias_value: fusor::Tensor<1, f32>,
    ln_2_weight: GradPair<1>,
    ln_2_weight_value: fusor::Tensor<1, f32>,
    ln_2_bias: GradPair<1>,
    ln_2_bias_value: fusor::Tensor<1, f32>,
    mlp: ExtractedMlpGradients,
    ln_mlp_out_weight: GradPair<1>,
    ln_mlp_out_weight_value: fusor::Tensor<1, f32>,
    ln_mlp_out_bias: GradPair<1>,
    ln_mlp_out_bias_value: fusor::Tensor<1, f32>,
}

impl ExtractedBlockGradients {
    fn extract(block: &TransformerBlock, gradients: &Gradients) -> Self {
        Self {
            ln_1_weight: extract_gradient(&block.ln_1_weight, gradients),
            ln_1_weight_value: block.ln_1_weight.raw().clone(),
            ln_1_bias: extract_gradient(&block.ln_1_bias, gradients),
            ln_1_bias_value: block.ln_1_bias.raw().clone(),
            mixer: ExtractedMixerGradients::extract(&block.mixer, gradients),
            ln_attn_out_weight: extract_gradient(&block.ln_attn_out_weight, gradients),
            ln_attn_out_weight_value: block.ln_attn_out_weight.raw().clone(),
            ln_attn_out_bias: extract_gradient(&block.ln_attn_out_bias, gradients),
            ln_attn_out_bias_value: block.ln_attn_out_bias.raw().clone(),
            ln_2_weight: extract_gradient(&block.ln_2_weight, gradients),
            ln_2_weight_value: block.ln_2_weight.raw().clone(),
            ln_2_bias: extract_gradient(&block.ln_2_bias, gradients),
            ln_2_bias_value: block.ln_2_bias.raw().clone(),
            mlp: ExtractedMlpGradients::extract(&block.mlp, gradients),
            ln_mlp_out_weight: extract_gradient(&block.ln_mlp_out_weight, gradients),
            ln_mlp_out_weight_value: block.ln_mlp_out_weight.raw().clone(),
            ln_mlp_out_bias: extract_gradient(&block.ln_mlp_out_bias, gradients),
            ln_mlp_out_bias_value: block.ln_mlp_out_bias.raw().clone(),
        }
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        collect_grad_pair_gpu_keys(&self.ln_1_weight, keys);
        collect_tensor_gpu_key(&self.ln_1_weight_value, keys);
        collect_grad_pair_gpu_keys(&self.ln_1_bias, keys);
        collect_tensor_gpu_key(&self.ln_1_bias_value, keys);
        self.mixer.collect_gpu_keys(keys);
        collect_grad_pair_gpu_keys(&self.ln_attn_out_weight, keys);
        collect_tensor_gpu_key(&self.ln_attn_out_weight_value, keys);
        collect_grad_pair_gpu_keys(&self.ln_attn_out_bias, keys);
        collect_tensor_gpu_key(&self.ln_attn_out_bias_value, keys);
        collect_grad_pair_gpu_keys(&self.ln_2_weight, keys);
        collect_tensor_gpu_key(&self.ln_2_weight_value, keys);
        collect_grad_pair_gpu_keys(&self.ln_2_bias, keys);
        collect_tensor_gpu_key(&self.ln_2_bias_value, keys);
        self.mlp.collect_gpu_keys(keys);
        collect_grad_pair_gpu_keys(&self.ln_mlp_out_weight, keys);
        collect_tensor_gpu_key(&self.ln_mlp_out_weight_value, keys);
        collect_grad_pair_gpu_keys(&self.ln_mlp_out_bias, keys);
        collect_tensor_gpu_key(&self.ln_mlp_out_bias_value, keys);
    }

    fn apply(
        self,
        graph: &Graph,
        state: &mut AdamBlockState,
        step: usize,
        settings: AdamWSettings,
    ) -> TransformerBlock {
        TransformerBlock {
            ln_1_weight: apply_extracted(
                graph,
                self.ln_1_weight,
                self.ln_1_weight_value,
                &mut state.ln_1_weight,
                step,
                settings,
            ),
            ln_1_bias: apply_extracted(
                graph,
                self.ln_1_bias,
                self.ln_1_bias_value,
                &mut state.ln_1_bias,
                step,
                settings,
            ),
            mixer: self.mixer.apply(graph, &mut state.mixer, step, settings),
            ln_attn_out_weight: apply_extracted(
                graph,
                self.ln_attn_out_weight,
                self.ln_attn_out_weight_value,
                &mut state.ln_attn_out_weight,
                step,
                settings,
            ),
            ln_attn_out_bias: apply_extracted(
                graph,
                self.ln_attn_out_bias,
                self.ln_attn_out_bias_value,
                &mut state.ln_attn_out_bias,
                step,
                settings,
            ),
            ln_2_weight: apply_extracted(
                graph,
                self.ln_2_weight,
                self.ln_2_weight_value,
                &mut state.ln_2_weight,
                step,
                settings,
            ),
            ln_2_bias: apply_extracted(
                graph,
                self.ln_2_bias,
                self.ln_2_bias_value,
                &mut state.ln_2_bias,
                step,
                settings,
            ),
            mlp: self.mlp.apply(graph, &mut state.mlp, step, settings),
            ln_mlp_out_weight: apply_extracted(
                graph,
                self.ln_mlp_out_weight,
                self.ln_mlp_out_weight_value,
                &mut state.ln_mlp_out_weight,
                step,
                settings,
            ),
            ln_mlp_out_bias: apply_extracted(
                graph,
                self.ln_mlp_out_bias,
                self.ln_mlp_out_bias_value,
                &mut state.ln_mlp_out_bias,
                step,
                settings,
            ),
        }
    }
}

enum ExtractedMixerGradients {
    Attention(ExtractedAttentionGradients),
    Conv(ExtractedConvGradients),
}

impl ExtractedMixerGradients {
    fn extract(mixer: &SequenceMixer, gradients: &Gradients) -> Self {
        match mixer {
            SequenceMixer::Attention(attn) => {
                Self::Attention(ExtractedAttentionGradients::extract(attn, gradients))
            }
            SequenceMixer::Conv(conv) => {
                Self::Conv(ExtractedConvGradients::extract(conv, gradients))
            }
        }
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        match self {
            Self::Attention(attn) => attn.collect_gpu_keys(keys),
            Self::Conv(conv) => conv.collect_gpu_keys(keys),
        }
    }

    fn apply(
        self,
        graph: &Graph,
        state: &mut AdamMixerState,
        step: usize,
        settings: AdamWSettings,
    ) -> SequenceMixer {
        match (self, state) {
            (Self::Attention(extracted), AdamMixerState::Attention(state)) => {
                SequenceMixer::Attention(extracted.apply(graph, state, step, settings))
            }
            (Self::Conv(extracted), AdamMixerState::Conv(state)) => {
                SequenceMixer::Conv(extracted.apply(graph, state, step, settings))
            }
            _ => unreachable!("mixer schedule changed after optimizer init"),
        }
    }
}

struct ExtractedAttentionGradients {
    c_attn_q: GradPair<2>,
    c_attn_q_value: fusor::Tensor<2, f32>,
    c_attn_k: GradPair<2>,
    c_attn_k_value: fusor::Tensor<2, f32>,
    c_attn_v: GradPair<2>,
    c_attn_v_value: fusor::Tensor<2, f32>,
    c_proj: GradPair<2>,
    c_proj_value: fusor::Tensor<2, f32>,
}

impl ExtractedAttentionGradients {
    fn extract(attn: &CausalSelfAttention, gradients: &Gradients) -> Self {
        Self {
            c_attn_q: extract_gradient(&attn.c_attn_q, gradients),
            c_attn_q_value: attn.c_attn_q.raw().clone(),
            c_attn_k: extract_gradient(&attn.c_attn_k, gradients),
            c_attn_k_value: attn.c_attn_k.raw().clone(),
            c_attn_v: extract_gradient(&attn.c_attn_v, gradients),
            c_attn_v_value: attn.c_attn_v.raw().clone(),
            c_proj: extract_gradient(&attn.c_proj, gradients),
            c_proj_value: attn.c_proj.raw().clone(),
        }
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        collect_grad_pair_gpu_keys(&self.c_attn_q, keys);
        collect_tensor_gpu_key(&self.c_attn_q_value, keys);
        collect_grad_pair_gpu_keys(&self.c_attn_k, keys);
        collect_tensor_gpu_key(&self.c_attn_k_value, keys);
        collect_grad_pair_gpu_keys(&self.c_attn_v, keys);
        collect_tensor_gpu_key(&self.c_attn_v_value, keys);
        collect_grad_pair_gpu_keys(&self.c_proj, keys);
        collect_tensor_gpu_key(&self.c_proj_value, keys);
    }

    fn apply(
        self,
        graph: &Graph,
        state: &mut AdamAttentionState,
        step: usize,
        settings: AdamWSettings,
    ) -> CausalSelfAttention {
        CausalSelfAttention {
            c_attn_q: apply_extracted(
                graph,
                self.c_attn_q,
                self.c_attn_q_value,
                &mut state.c_attn_q,
                step,
                settings,
            ),
            c_attn_k: apply_extracted(
                graph,
                self.c_attn_k,
                self.c_attn_k_value,
                &mut state.c_attn_k,
                step,
                settings,
            ),
            c_attn_v: apply_extracted(
                graph,
                self.c_attn_v,
                self.c_attn_v_value,
                &mut state.c_attn_v,
                step,
                settings,
            ),
            c_proj: apply_extracted(
                graph,
                self.c_proj,
                self.c_proj_value,
                &mut state.c_proj,
                step,
                settings,
            ),
        }
    }
}

struct ExtractedConvGradients {
    kernels: Vec<GradPair<2>>,
    kernel_values: Vec<fusor::Tensor<2, f32>>,
    bias: GradPair<1>,
    bias_value: fusor::Tensor<1, f32>,
    out_proj: GradPair<2>,
    out_proj_value: fusor::Tensor<2, f32>,
}

impl ExtractedConvGradients {
    fn extract(conv: &ConvMixer, gradients: &Gradients) -> Self {
        Self {
            kernels: conv
                .kernels
                .iter()
                .map(|k| extract_gradient(k, gradients))
                .collect(),
            kernel_values: conv.kernels.iter().map(|k| k.raw().clone()).collect(),
            bias: extract_gradient(&conv.bias, gradients),
            bias_value: conv.bias.raw().clone(),
            out_proj: extract_gradient(&conv.out_proj, gradients),
            out_proj_value: conv.out_proj.raw().clone(),
        }
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        for (pair, val) in self.kernels.iter().zip(&self.kernel_values) {
            collect_grad_pair_gpu_keys(pair, keys);
            collect_tensor_gpu_key(val, keys);
        }
        collect_grad_pair_gpu_keys(&self.bias, keys);
        collect_tensor_gpu_key(&self.bias_value, keys);
        collect_grad_pair_gpu_keys(&self.out_proj, keys);
        collect_tensor_gpu_key(&self.out_proj_value, keys);
    }

    fn apply(
        self,
        graph: &Graph,
        state: &mut AdamConvState,
        step: usize,
        settings: AdamWSettings,
    ) -> ConvMixer {
        let kernels = self
            .kernels
            .into_iter()
            .zip(self.kernel_values)
            .zip(state.kernels.iter_mut())
            .map(|((grad_pair, value), moments)| {
                apply_extracted(graph, grad_pair, value, moments, step, settings)
            })
            .collect();
        ConvMixer {
            kernels,
            bias: apply_extracted(
                graph,
                self.bias,
                self.bias_value,
                &mut state.bias,
                step,
                settings,
            ),
            out_proj: apply_extracted(
                graph,
                self.out_proj,
                self.out_proj_value,
                &mut state.out_proj,
                step,
                settings,
            ),
        }
    }
}

struct ExtractedMlpGradients {
    c_fc: GradPair<2>,
    c_fc_value: fusor::Tensor<2, f32>,
    c_fc_bias: GradPair<1>,
    c_fc_bias_value: fusor::Tensor<1, f32>,
    c_proj: GradPair<2>,
    c_proj_value: fusor::Tensor<2, f32>,
    c_proj_bias: GradPair<1>,
    c_proj_bias_value: fusor::Tensor<1, f32>,
}

impl ExtractedMlpGradients {
    fn extract(mlp: &Mlp, gradients: &Gradients) -> Self {
        Self {
            c_fc: extract_gradient(&mlp.c_fc, gradients),
            c_fc_value: mlp.c_fc.raw().clone(),
            c_fc_bias: extract_gradient(&mlp.c_fc_bias, gradients),
            c_fc_bias_value: mlp.c_fc_bias.raw().clone(),
            c_proj: extract_gradient(&mlp.c_proj, gradients),
            c_proj_value: mlp.c_proj.raw().clone(),
            c_proj_bias: extract_gradient(&mlp.c_proj_bias, gradients),
            c_proj_bias_value: mlp.c_proj_bias.raw().clone(),
        }
    }

    fn collect_gpu_keys(&self, keys: &mut Vec<NodeIndex>) {
        collect_grad_pair_gpu_keys(&self.c_fc, keys);
        collect_tensor_gpu_key(&self.c_fc_value, keys);
        collect_grad_pair_gpu_keys(&self.c_fc_bias, keys);
        collect_tensor_gpu_key(&self.c_fc_bias_value, keys);
        collect_grad_pair_gpu_keys(&self.c_proj, keys);
        collect_tensor_gpu_key(&self.c_proj_value, keys);
        collect_grad_pair_gpu_keys(&self.c_proj_bias, keys);
        collect_tensor_gpu_key(&self.c_proj_bias_value, keys);
    }

    fn apply(
        self,
        graph: &Graph,
        state: &mut AdamMlpState,
        step: usize,
        settings: AdamWSettings,
    ) -> Mlp {
        Mlp {
            c_fc: apply_extracted(
                graph,
                self.c_fc,
                self.c_fc_value,
                &mut state.c_fc,
                step,
                settings,
            ),
            c_fc_bias: apply_extracted(
                graph,
                self.c_fc_bias,
                self.c_fc_bias_value,
                &mut state.c_fc_bias,
                step,
                settings,
            ),
            c_proj: apply_extracted(
                graph,
                self.c_proj,
                self.c_proj_value,
                &mut state.c_proj,
                step,
                settings,
            ),
            c_proj_bias: apply_extracted(
                graph,
                self.c_proj_bias,
                self.c_proj_bias_value,
                &mut state.c_proj_bias,
                step,
                settings,
            ),
        }
    }
}

/// Helper: apply an extracted gradient pair or fall back to the unchanged value.
fn apply_extracted<const R: usize>(
    graph: &Graph,
    grad_pair: GradPair<R>,
    fallback_value: fusor::Tensor<R, f32>,
    moments: &mut AdamMoments<R>,
    step: usize,
    settings: AdamWSettings,
) -> Tensor<R> {
    match grad_pair {
        Some((gradient, param_value)) => Tensor::from_raw(
            graph,
            adamw_update_raw(param_value, gradient, moments, step, settings),
        ),
        None => Tensor::from_raw(graph, fallback_value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_nanochat::{
        RuntimeConfig, SaveQuantization,
        data::{
            StrokeTokenizer, canvas_state_indexes, position_indexes, token_component_indexes,
        },
    };
    use rand::SeedableRng;

    #[tokio::test]
    async fn rope_mqa_model_uses_shared_kv_and_skips_position_embeddings() {
        let device = Device::cpu();
        let mut rng = StdRng::seed_from_u64(7);
        let config = RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.01,
            log_every: 1,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 8,
            batch_size: 2,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 1,
            n_ff: 32,
            n_layer: 1,
            conv_kernel_size: 3,
            attention_period: 1,
            use_rope: true,
            rope_theta: 10_000.0,
            use_canvas_state_embeddings: true,
            use_extra_norms: false,
            eps: 1e-5,
            init_scale: 0.05,
            seed: 7,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: SaveQuantization::F32,
            train_examples: 1,
            validation_examples: 1,
            test_examples: 1,
            dataset_path: None,
            include_synthetic_data: false,
            synthetic_multiplier: 1,
            gguf_path: "test.gguf".into(),
            sample_output_path: "sample.svg".into(),
        };

        let tokenizer = StrokeTokenizer::new();
        let model = NanoChatModel::new(&device, &mut rng, &tokenizer, &config);
        assert_eq!(model.n_kv_head(), 1);
        assert!(model.use_rope());
        assert!(model.canvas_state_spec().is_some());

        let tensors = model.named_tensors().await;
        assert!(
            !tensors
                .iter()
                .any(|tensor| tensor.name == "position_embd.weight"),
            "RoPE models should not serialize learned position embeddings"
        );
        assert!(
            tensors
                .iter()
                .any(|tensor| tensor.name == "cursor_x_embd.weight"),
            "canvas-state models should serialize cursor x embeddings"
        );

        let attn_k = tensors
            .iter()
            .find(|tensor| tensor.name == "blk.0.attn_k.weight")
            .unwrap();
        let attn_v = tensors
            .iter()
            .find(|tensor| tensor.name == "blk.0.attn_v.weight")
            .unwrap();
        let cursor_x = tensors
            .iter()
            .find(|tensor| tensor.name == "cursor_x_embd.weight")
            .unwrap();
        assert_eq!(attn_k.shape, vec![16, 4]);
        assert_eq!(attn_v.shape, vec![16, 4]);
        assert_eq!(cursor_x.shape, vec![129, 16]);
    }

    #[test]
    fn forward_accepts_sequences_shorter_than_block_size() {
        let device = Device::cpu();
        let mut rng = StdRng::seed_from_u64(11);
        let config = RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.01,
            log_every: 1,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 8,
            batch_size: 2,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 1,
            n_ff: 32,
            n_layer: 1,
            conv_kernel_size: 3,
            attention_period: 1,
            use_rope: true,
            rope_theta: 10_000.0,
            use_canvas_state_embeddings: true,
            use_extra_norms: false,
            eps: 1e-5,
            init_scale: 0.05,
            seed: 11,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: SaveQuantization::F32,
            train_examples: 1,
            validation_examples: 1,
            test_examples: 1,
            dataset_path: None,
            include_synthetic_data: false,
            synthetic_multiplier: 1,
            gguf_path: "test.gguf".into(),
            sample_output_path: "sample.svg".into(),
        };

        let tokenizer = StrokeTokenizer::new();
        let model = NanoChatModel::new(&device, &mut rng, &tokenizer, &config);
        let inputs = vec![
            vec![
                tokenizer.bos_token(),
                tokenizer.token_from_components(1, 2, 1),
                tokenizer.eot_token(),
            ],
            vec![
                tokenizer.bos_token(),
                tokenizer.token_from_components(0, 4, 1),
                tokenizer.eot_token(),
            ],
        ];
        let components = token_component_indexes(&tokenizer, &inputs);
        let mode_inputs: RawTensor<2, u32> = RawTensor::new(&device, &components.mode);
        let direction_inputs: RawTensor<2, u32> = RawTensor::new(&device, &components.direction);
        let count_inputs: RawTensor<2, u32> = RawTensor::new(&device, &components.count);
        let position_values = position_indexes(2, 3);
        let position_inputs: RawTensor<2, u32> = RawTensor::new(&device, &position_values);
        let canvas = canvas_state_indexes(
            &tokenizer,
            &inputs,
            model
                .canvas_state_spec()
                .expect("canvas embeddings enabled"),
        );
        let cursor_x_inputs: RawTensor<2, u32> = RawTensor::new(&device, &canvas.cursor_x);
        let cursor_y_inputs: RawTensor<2, u32> = RawTensor::new(&device, &canvas.cursor_y);
        let pen_state_inputs: RawTensor<2, u32> = RawTensor::new(&device, &canvas.pen_state);
        let causal_mask_values = vec![
            vec![
                vec![0.0f32, f32::NEG_INFINITY, f32::NEG_INFINITY],
                vec![0.0f32, 0.0f32, f32::NEG_INFINITY],
                vec![0.0f32, 0.0f32, 0.0f32]
            ];
            2
        ];
        let causal_mask =
            Tensor::constant_from_raw(model.graph(), RawTensor::new(&device, &causal_mask_values));

        let logits = model.forward(
            &mode_inputs,
            &direction_inputs,
            &count_inputs,
            &position_inputs,
            &cursor_x_inputs,
            &cursor_y_inputs,
            &pen_state_inputs,
            &causal_mask,
        );

        assert_eq!(logits.mode.shape(), [2, 3, 3]);
        assert_eq!(logits.direction.shape(), [2, 3, 8]);
        assert_eq!(logits.count.shape(), [2, 3, tokenizer.max_count()]);
    }
}
