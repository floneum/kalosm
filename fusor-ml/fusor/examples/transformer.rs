//! Train a small GPT-style transformer on tiny Shakespeare with the autograd API.
//!
//! Downloads the dataset on first run (cached under `examples/data`), builds a
//! character-level vocabulary, then trains a 2-layer pre-norm transformer
//! (learned positional embeddings, 4-head causal self-attention, gelu MLP)
//! with softmax cross-entropy and Adam, rebuilding the tape each step.
//! Finishes by reporting validation loss and sampling text from the model.
//!
//! Run with:
//! ```sh
//! cargo run --release --example transformer
//! ```

use std::collections::BTreeSet;
use std::io::Read;
use std::path::PathBuf;

use fusor::autograd::layers::{Embedding, LayerNorm, Linear};
use fusor::autograd::{Graph, Tensor};
use fusor::{Device, Tensor as RawTensor, ToVec1, ToVec2};

const CONTEXT: usize = 64;
const BATCH_SIZE: usize = 32;
const DIM: usize = 64;
const HEADS: usize = 4;
const HEAD_DIM: usize = DIM / HEADS;
const MLP_DIM: usize = 4 * DIM;
const LAYERS: usize = 2;
const STEPS: usize = 300;
const LEARNING_RATE: f32 = 1e-3;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;
const LAYER_NORM_EPS: f32 = 1e-5;
const MASK_VALUE: f32 = -1e9;
const TEMPERATURE: f32 = 0.8;

const DATA_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data")
}

/// Download the tiny Shakespeare corpus, caching it on disk.
fn fetch_text() -> String {
    let path = data_dir().join("tinyshakespeare.txt");
    if let Ok(text) = std::fs::read_to_string(&path) {
        return text;
    }
    println!("downloading tinyshakespeare.txt");
    let mut response = ureq::get(DATA_URL)
        .call()
        .unwrap_or_else(|err| panic!("failed to download tiny shakespeare: {err}"));
    let mut text = String::new();
    response
        .body_mut()
        .as_reader()
        .read_to_string(&mut text)
        .unwrap();
    std::fs::create_dir_all(data_dir()).unwrap();
    std::fs::write(&path, &text).unwrap();
    text
}

/// Deterministic LCG so runs are reproducible without a rand dependency.
struct Lcg(u64);

impl Lcg {
    /// Uniform in [0, 1).
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as f32 / (1u64 << 32) as f32
    }

    /// Uniform in (-bound, bound) with `bound = sqrt(1 / fan_in)`, the
    /// PyTorch default init for linear layers.
    fn kaiming(&mut self, count: usize, fan_in: usize) -> Vec<f32> {
        let bound = (1.0 / fan_in as f32).sqrt();
        (0..count)
            .map(|_| (self.next_f32() - 0.5) * 2.0 * bound)
            .collect()
    }
}

/// Raw parameter tensors that persist across training steps.
struct BlockParams {
    ln1_weight: RawTensor<1, f32>,
    ln1_bias: RawTensor<1, f32>,
    query: RawTensor<2, f32>,
    key: RawTensor<2, f32>,
    value: RawTensor<2, f32>,
    output: RawTensor<2, f32>,
    ln2_weight: RawTensor<1, f32>,
    ln2_bias: RawTensor<1, f32>,
    mlp_up: RawTensor<2, f32>,
    mlp_up_bias: RawTensor<1, f32>,
    mlp_down: RawTensor<2, f32>,
    mlp_down_bias: RawTensor<1, f32>,
}

struct Params {
    token_embedding: RawTensor<2, f32>,
    position_embedding: RawTensor<2, f32>,
    blocks: Vec<BlockParams>,
    ln_final_weight: RawTensor<1, f32>,
    ln_final_bias: RawTensor<1, f32>,
    lm_head: RawTensor<2, f32>,
}

impl Params {
    fn new(device: &Device, vocab_size: usize) -> Self {
        let mut rng = Lcg(42);
        let blocks = (0..LAYERS)
            .map(|_| BlockParams {
                ln1_weight: RawTensor::splat(device, 1.0, [DIM]),
                ln1_bias: RawTensor::zeros(device, [DIM]),
                query: RawTensor::from_slice(device, [DIM, DIM], &rng.kaiming(DIM * DIM, DIM)),
                key: RawTensor::from_slice(device, [DIM, DIM], &rng.kaiming(DIM * DIM, DIM)),
                value: RawTensor::from_slice(device, [DIM, DIM], &rng.kaiming(DIM * DIM, DIM)),
                output: RawTensor::from_slice(device, [DIM, DIM], &rng.kaiming(DIM * DIM, DIM)),
                ln2_weight: RawTensor::splat(device, 1.0, [DIM]),
                ln2_bias: RawTensor::zeros(device, [DIM]),
                mlp_up: RawTensor::from_slice(
                    device,
                    [MLP_DIM, DIM],
                    &rng.kaiming(MLP_DIM * DIM, DIM),
                ),
                mlp_up_bias: RawTensor::zeros(device, [MLP_DIM]),
                mlp_down: RawTensor::from_slice(
                    device,
                    [DIM, MLP_DIM],
                    &rng.kaiming(DIM * MLP_DIM, MLP_DIM),
                ),
                mlp_down_bias: RawTensor::zeros(device, [DIM]),
            })
            .collect();
        Self {
            token_embedding: RawTensor::from_slice(
                device,
                [vocab_size, DIM],
                &rng.kaiming(vocab_size * DIM, DIM),
            ),
            position_embedding: RawTensor::from_slice(
                device,
                [CONTEXT, DIM],
                &rng.kaiming(CONTEXT * DIM, DIM),
            ),
            blocks,
            ln_final_weight: RawTensor::splat(device, 1.0, [DIM]),
            ln_final_bias: RawTensor::zeros(device, [DIM]),
            lm_head: RawTensor::from_slice(
                device,
                [vocab_size, DIM],
                &rng.kaiming(vocab_size * DIM, DIM),
            ),
        }
    }

    /// Parameters in a fixed order; must match [`Gpt::leaves`].
    fn tensors_mut(&mut self) -> (Vec<&mut RawTensor<1, f32>>, Vec<&mut RawTensor<2, f32>>) {
        let mut rank1 = vec![&mut self.ln_final_weight, &mut self.ln_final_bias];
        let mut rank2 = vec![
            &mut self.token_embedding,
            &mut self.position_embedding,
            &mut self.lm_head,
        ];
        for block in &mut self.blocks {
            rank1.extend([
                &mut block.ln1_weight,
                &mut block.ln1_bias,
                &mut block.ln2_weight,
                &mut block.ln2_bias,
                &mut block.mlp_up_bias,
                &mut block.mlp_down_bias,
            ]);
            rank2.extend([
                &mut block.query,
                &mut block.key,
                &mut block.value,
                &mut block.output,
                &mut block.mlp_up,
                &mut block.mlp_down,
            ]);
        }
        (rank1, rank2)
    }
}

/// Pre-norm transformer block: causal self-attention then a gelu MLP,
/// each behind a residual connection.
struct Block {
    ln1: LayerNorm<1>,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    ln2: LayerNorm<1>,
    mlp_up: Linear,
    mlp_down: Linear,
}

impl Block {
    /// Multi-head causal self-attention for `x` of shape (batch, seq, dim).
    fn attention(&self, x: &Tensor<3>, mask: &Tensor<2>) -> Tensor<3> {
        let [batch, seq, dim] = x.shape();
        // (batch, seq, dim) -> (batch, heads, seq, head_dim)
        let split = |projected: &Tensor<3>| {
            projected
                .reshape([batch, seq, HEADS, HEAD_DIM])
                .transpose(1, 2)
        };
        let query = split(&self.query.forward(x));
        let key = split(&self.key.forward(x));
        let value = split(&self.value.forward(x));

        let scores = query
            .mat_mul(&key.transpose(2, 3))
            .mul_scalar(1.0 / (HEAD_DIM as f32).sqrt())
            .add_::<4, 4>(&mask.broadcast_as([batch, HEADS, seq, seq]));
        let attention = scores.softmax_last_dim::<3>();
        let context = attention
            .mat_mul(&value)
            .transpose(1, 2)
            .reshape([batch, seq, dim]);
        self.output.forward(&context)
    }

    fn forward(&self, x: &Tensor<3>, mask: &Tensor<2>) -> Tensor<3> {
        let x = x.add(&self.attention(&self.ln1.forward(x), mask));
        let mlp = self
            .mlp_down
            .forward(&self.mlp_up.forward(&self.ln2.forward(&x)).gelu());
        x.add(&mlp)
    }
}

struct Gpt {
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<Block>,
    ln_final: LayerNorm<1>,
    lm_head: Linear,
}

impl Gpt {
    /// Build the model on `graph`, as trainable leaves (training) or
    /// constants (evaluation).
    fn new(graph: &Graph, params: &Params, trainable: bool) -> Self {
        fn wrap<const R: usize>(
            graph: &Graph,
            value: &RawTensor<R, f32>,
            trainable: bool,
        ) -> Tensor<R> {
            if trainable {
                graph.leaf(value.clone())
            } else {
                Tensor::constant_from_raw(graph, value.clone())
            }
        }
        let blocks = params
            .blocks
            .iter()
            .map(|block| Block {
                ln1: LayerNorm::new(
                    wrap(graph, &block.ln1_weight, trainable),
                    Some(wrap(graph, &block.ln1_bias, trainable)),
                    LAYER_NORM_EPS,
                ),
                query: Linear::new(wrap(graph, &block.query, trainable), None),
                key: Linear::new(wrap(graph, &block.key, trainable), None),
                value: Linear::new(wrap(graph, &block.value, trainable), None),
                output: Linear::new(wrap(graph, &block.output, trainable), None),
                ln2: LayerNorm::new(
                    wrap(graph, &block.ln2_weight, trainable),
                    Some(wrap(graph, &block.ln2_bias, trainable)),
                    LAYER_NORM_EPS,
                ),
                mlp_up: Linear::new(
                    wrap(graph, &block.mlp_up, trainable),
                    Some(wrap(graph, &block.mlp_up_bias, trainable)),
                ),
                mlp_down: Linear::new(
                    wrap(graph, &block.mlp_down, trainable),
                    Some(wrap(graph, &block.mlp_down_bias, trainable)),
                ),
            })
            .collect();
        Self {
            token_embedding: Embedding::new_from_tensor(wrap(
                graph,
                &params.token_embedding,
                trainable,
            )),
            position_embedding: Embedding::new_from_tensor(wrap(
                graph,
                &params.position_embedding,
                trainable,
            )),
            blocks,
            ln_final: LayerNorm::new(
                wrap(graph, &params.ln_final_weight, trainable),
                Some(wrap(graph, &params.ln_final_bias, trainable)),
                LAYER_NORM_EPS,
            ),
            lm_head: Linear::new(wrap(graph, &params.lm_head, trainable), None),
        }
    }

    /// Parameter leaves in a fixed order; must match [`Params::tensors_mut`].
    fn leaves(&self) -> (Vec<&Tensor<1>>, Vec<&Tensor<2>>) {
        let mut rank1 = vec![self.ln_final.weight(), self.ln_final.bias().unwrap()];
        let mut rank2 = vec![
            self.token_embedding.embeddings(),
            self.position_embedding.embeddings(),
            self.lm_head.weight(),
        ];
        for block in &self.blocks {
            rank1.extend([
                block.ln1.weight(),
                block.ln1.bias().unwrap(),
                block.ln2.weight(),
                block.ln2.bias().unwrap(),
                block.mlp_up.bias().unwrap(),
                block.mlp_down.bias().unwrap(),
            ]);
            rank2.extend([
                block.query.weight(),
                block.key.weight(),
                block.value.weight(),
                block.output.weight(),
                block.mlp_up.weight(),
                block.mlp_down.weight(),
            ]);
        }
        (rank1, rank2)
    }

    /// Input shape: (batch, seq) token ids. Output shape: (batch, seq, vocab).
    fn forward(
        &self,
        tokens: &RawTensor<2, u32>,
        positions: &RawTensor<1, u32>,
        mask: &Tensor<2>,
    ) -> Tensor<3> {
        let [batch, seq] = tokens.shape();
        let position_embedded = self.position_embedding.forward_1d(positions);
        let mut x = self
            .token_embedding
            .forward(tokens)
            .add(&position_embedded.broadcast_as([batch, seq, DIM]));
        for block in &self.blocks {
            x = block.forward(&x, mask);
        }
        self.lm_head.forward(&self.ln_final.forward(&x))
    }
}

/// Additive causal mask: 0 where position j <= i, a large negative
/// value elsewhere so softmax zeroes out future positions.
fn causal_mask(device: &Device, seq: usize) -> RawTensor<2, f32> {
    let values: Vec<f32> = (0..seq * seq)
        .map(|i| if i % seq <= i / seq { 0.0 } else { MASK_VALUE })
        .collect();
    RawTensor::from_slice(device, [seq, seq], &values)
}

/// Numerically stable softmax cross-entropy averaged over all positions:
/// log softmax via log-sum-exp so a saturated class cannot underflow.
fn cross_entropy(logits: &Tensor<2>, targets: &RawTensor<1, u32>) -> Tensor<0> {
    let batch = logits.shape()[0];
    let shifted = logits.sub_::<2, 2>(&logits.max_keepdim::<1>(1));
    let log_sum_exp = shifted.exp().sum_keepdim(1).log();
    let label_log_probs = shifted.sub_::<2, 2>(&log_sum_exp).gather_last(targets);
    label_log_probs.sum().mul_scalar(-1.0 / batch as f32)
}

/// First and second moment estimates for one parameter tensor, stored flat
/// so the update math is rank-independent.
struct AdamState {
    momentum: RawTensor<1, f32>,
    variance: RawTensor<1, f32>,
}

impl AdamState {
    fn zeros(device: &Device, elements: usize) -> Self {
        Self {
            momentum: RawTensor::zeros(device, [elements]),
            variance: RawTensor::zeros(device, [elements]),
        }
    }
}

/// One Adam update. `lr` is the learning rate with warmup and the bias
/// correction for step `t` already folded in. It is passed as a [1] tensor
/// rather than an f32 on purpose: scalar constants are baked into the
/// generated kernel source, so a per-step scalar would force a shader
/// recompile every step, while tensor *contents* are runtime data and the
/// cached pipelines are reused.
fn adam_step<const R: usize>(
    param: &mut RawTensor<R, f32>,
    state: &mut AdamState,
    gradient: RawTensor<R, f32>,
    lr: &RawTensor<1, f32>,
) {
    let shape = param.shape();
    let elements = shape.iter().product();
    let gradient = gradient.reshape([elements]);
    state.momentum =
        (state.momentum.clone() * BETA1 + gradient.clone() * (1.0 - BETA1)).to_concrete();
    state.variance =
        (state.variance.clone() * BETA2 + (gradient.clone() * gradient) * (1.0 - BETA2))
            .to_concrete();
    let update: RawTensor<1, f32> = state
        .momentum
        .mul_(lr)
        .div_(&state.variance.sqrt().add_scalar(EPSILON).to_concrete());
    *param = (param.clone() - update.reshape(shape)).to_concrete();
}

/// Pick `batch_size` random windows and return (inputs, next-char targets),
/// each flattened to batch_size * CONTEXT ids.
fn sample_batch(tokens: &[u32], rng: &mut Lcg, batch_size: usize) -> (Vec<u32>, Vec<u32>) {
    let mut inputs = Vec::with_capacity(batch_size * CONTEXT);
    let mut targets = Vec::with_capacity(batch_size * CONTEXT);
    for _ in 0..batch_size {
        let start = (rng.next_f32() * (tokens.len() - CONTEXT - 1) as f32) as usize;
        inputs.extend_from_slice(&tokens[start..start + CONTEXT]);
        targets.extend_from_slice(&tokens[start + 1..start + CONTEXT + 1]);
    }
    (inputs, targets)
}

async fn to_scalar(value: RawTensor<0, f32>) -> f32 {
    value.reshape([1]).as_slice().await.unwrap().to_vec1()[0]
}

/// Sample a token id from logits with temperature.
fn sample(logits: &[f32], rng: &mut Lcg) -> u32 {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = logits
        .iter()
        .map(|logit| ((logit - max) / TEMPERATURE).exp())
        .collect();
    let mut remaining = rng.next_f32() * weights.iter().sum::<f32>();
    for (index, weight) in weights.iter().enumerate() {
        remaining -= weight;
        if remaining <= 0.0 {
            return index as u32;
        }
    }
    (weights.len() - 1) as u32
}

/// Autoregressively extend `tokens` by `length` sampled characters. The
/// prompt must be at least CONTEXT tokens so every forward pass sees the
/// same shapes and reuses the same compiled kernels.
async fn generate(
    device: &Device,
    params: &Params,
    mut tokens: Vec<u32>,
    length: usize,
    rng: &mut Lcg,
) -> Vec<u32> {
    assert!(tokens.len() >= CONTEXT, "prompt shorter than CONTEXT");
    let positions: Vec<u32> = (0..CONTEXT as u32).collect();
    let positions = RawTensor::from_slice(device, [CONTEXT], &positions);
    let mask = causal_mask(device, CONTEXT);
    for _ in 0..length {
        let window = &tokens[tokens.len() - CONTEXT..];
        let graph = Graph::new();
        let model = Gpt::new(&graph, params, false);
        let input = RawTensor::from_slice(device, [1, CONTEXT], window);
        let mask = Tensor::constant_from_raw(&graph, mask.clone());
        let logits = model.forward(&input, &positions, &mask);
        let vocab_size = logits.shape()[2];
        let rows = logits
            .raw()
            .clone()
            .reshape([CONTEXT, vocab_size])
            .as_slice()
            .await
            .unwrap()
            .to_vec2();
        tokens.push(sample(&rows[CONTEXT - 1], rng));
    }
    tokens
}

#[tokio::main]
async fn main() {
    if std::env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    }
    let text = fetch_text();
    // Character-level vocabulary over the corpus bytes (the text is ASCII).
    let vocab: Vec<u8> = text.bytes().collect::<BTreeSet<u8>>().into_iter().collect();
    let index_of: std::collections::HashMap<u8, u32> = vocab
        .iter()
        .enumerate()
        .map(|(index, &byte)| (byte, index as u32))
        .collect();
    let tokens: Vec<u32> = text.bytes().map(|byte| index_of[&byte]).collect();
    let split = tokens.len() * 9 / 10;
    let (train_tokens, val_tokens) = tokens.split_at(split);
    println!(
        "corpus: {} chars, vocab {}, {} train / {} validation",
        tokens.len(),
        vocab.len(),
        train_tokens.len(),
        val_tokens.len()
    );

    let device = match Device::gpu().await {
        Ok(gpu) => gpu,
        Err(_) => {
            println!("GPU unavailable, training on CPU");
            Device::cpu()
        }
    };

    let mut params = Params::new(&device, vocab.len());
    let (mut adam1, mut adam2) = {
        let (rank1, rank2) = params.tensors_mut();
        (
            rank1
                .iter()
                .map(|tensor| AdamState::zeros(&device, tensor.shape().iter().product()))
                .collect::<Vec<_>>(),
            rank2
                .iter()
                .map(|tensor| AdamState::zeros(&device, tensor.shape().iter().product()))
                .collect::<Vec<_>>(),
        )
    };

    let positions: Vec<u32> = (0..CONTEXT as u32).collect();
    let positions = RawTensor::from_slice(&device, [CONTEXT], &positions);
    let mask = causal_mask(&device, CONTEXT);

    let mut rng = Lcg(7);
    let start = std::time::Instant::now();
    for step in 0..STEPS {
        let (inputs, targets) = sample_batch(train_tokens, &mut rng, BATCH_SIZE);
        let inputs = RawTensor::from_slice(&device, [BATCH_SIZE, CONTEXT], &inputs);
        let targets = RawTensor::from_slice(&device, [BATCH_SIZE * CONTEXT], &targets);

        let graph = Graph::new();
        let model = Gpt::new(&graph, &params, true);
        let mask = Tensor::constant_from_raw(&graph, mask.clone());
        let logits = model.forward(&inputs, &positions, &mask);
        let flat_logits = logits.reshape([BATCH_SIZE * CONTEXT, vocab.len()]);
        let loss = cross_entropy(&flat_logits, &targets);

        // Gradients stay lazy: no per-step readback. The whole step
        // (forward, backward, and the optimizer updates below) is submitted
        // to the GPU by the flush at the bottom of the loop.
        let gradients = loss.backward().unwrap();

        // Adam with warmup and bias correction folded into the learning rate.
        let t = step as i32 + 1;
        let warmup = (STEPS / 10).max(1);
        let lr_value = LEARNING_RATE
            * ((step + 1) as f32 / warmup as f32).min(1.0)
            * (1.0 - BETA2.powi(t)).sqrt()
            / (1.0 - BETA1.powi(t));
        let lr = RawTensor::from_slice(&device, [1], &[lr_value]);
        let (leaves1, leaves2) = model.leaves();
        let (tensors1, tensors2) = params.tensors_mut();
        for ((param, state), leaf) in tensors1.into_iter().zip(&mut adam1).zip(leaves1) {
            adam_step(param, state, gradients.get(leaf).expect("missing gradient"), &lr);
        }
        for ((param, state), leaf) in tensors2.into_iter().zip(&mut adam2).zip(leaves2) {
            adam_step(param, state, gradients.get(leaf).expect("missing gradient"), &lr);
        }

        if step % 100 == 0 || step + 1 == STEPS {
            // Reading the loss synchronizes with the GPU; only do it when
            // printing progress.
            let loss_value = to_scalar(loss.raw().clone()).await;
            println!("step {step}/{STEPS}: loss {loss_value:.4}");
        } else {
            // Submit the step's work without waiting for the GPU: keeps the
            // pending graph bounded while the host races ahead.
            device.flush();
        }
    }
    let elapsed = start.elapsed();
    println!(
        "trained {STEPS} steps ({} tokens) in {elapsed:.2?} ({:.1} steps/s)",
        STEPS * BATCH_SIZE * CONTEXT,
        STEPS as f64 / elapsed.as_secs_f64(),
    );

    // Validation loss over held-out batches.
    const VAL_BATCHES: usize = 10;
    let mut total = 0.0;
    for _ in 0..VAL_BATCHES {
        let (inputs, targets) = sample_batch(val_tokens, &mut rng, BATCH_SIZE);
        let inputs = RawTensor::from_slice(&device, [BATCH_SIZE, CONTEXT], &inputs);
        let targets = RawTensor::from_slice(&device, [BATCH_SIZE * CONTEXT], &targets);
        let graph = Graph::new();
        let model = Gpt::new(&graph, &params, false);
        let mask = Tensor::constant_from_raw(&graph, mask.clone());
        let logits = model.forward(&inputs, &positions, &mask);
        let flat_logits = logits.reshape([BATCH_SIZE * CONTEXT, vocab.len()]);
        total += to_scalar(cross_entropy(&flat_logits, &targets).raw().clone()).await;
    }
    println!("validation loss: {:.4}", total / VAL_BATCHES as f32);

    // Prompt with the tail of the corpus so the window is always full.
    let prompt = tokens[tokens.len() - CONTEXT..].to_vec();
    let generated = generate(&device, &params, prompt, 400, &mut rng).await;
    let text: String = generated[CONTEXT..]
        .iter()
        .map(|&id| vocab[id as usize] as char)
        .collect();
    println!("--- sample ---\n{text}");
}
