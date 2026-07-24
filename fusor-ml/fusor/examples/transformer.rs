//! Train a small GPT-style transformer on tiny Shakespeare with the autograd API.
//!
//! Downloads the dataset on first run (cached under `examples/data`), builds a
//! character-level vocabulary, then trains a 2-layer pre-norm transformer
//! (learned positional embeddings, 4-head causal self-attention, gelu MLP)
//! with softmax cross-entropy and Adam, rebuilding the tape each step.
//! Reports train/test next-token accuracy while training, then samples text.
//!
//! The model body is written once over an element type alias and stamped out
//! per dtype: `--dtype f16` runs the standard mixed-precision recipe (f32
//! master weights and optimizer, f16 model compute, final norm + lm head +
//! loss in f32), the default runs everything in f32.
//!
//! Run with:
//! ```sh
//! cargo run --release --example transformer
//! ```

use std::collections::{BTreeSet, VecDeque};
use std::io::Read;
use std::path::PathBuf;

use fusor::autograd::layers::{Embedding, LayerNorm, Linear};
use fusor::autograd::{Graph, Tensor};
use fusor::{
    Device, FloatDataType, MaskKind, StandardSamplerParams, Tensor as RawTensor, ToVec, cat,
};

const CONTEXT: usize = 256;
const BATCH_SIZE: usize = 64;
const DIM: usize = 512;
const HEADS: usize = 8;
const HEAD_DIM: usize = DIM / HEADS;
const MLP_DIM: usize = 4 * DIM;
const LAYERS: usize = 6;
const STEPS: usize = 300;
const LEARNING_RATE: f32 = 1e-3;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPSILON: f32 = 1e-8;
const LAYER_NORM_EPS: f32 = 1e-5;
const MASK_VALUE: f32 = -1e9;
const TEMPERATURE: f32 = 0.8;
const GENERATION_RUN_AHEAD: usize = 32;
const PROGRESS_EVERY: usize = 25;

const DATA_URL: &str =
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

#[derive(Clone, Copy, PartialEq)]
enum Dtype {
    F32,
    F16,
}

#[derive(Clone, Copy)]
struct RunConfig {
    steps: usize,
    min_steps_per_second: Option<f64>,
    skip_eval: bool,
    progress_every: usize,
    trace_host: bool,
    trace_resolve: bool,
    trace_names: bool,
    dtype: Dtype,
}

impl RunConfig {
    fn from_args() -> Self {
        let mut config = Self {
            steps: STEPS,
            min_steps_per_second: None,
            skip_eval: false,
            progress_every: PROGRESS_EVERY,
            trace_host: false,
            trace_resolve: false,
            trace_names: false,
            dtype: Dtype::F32,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--steps" => {
                    let value = args.next().expect("--steps requires a value");
                    config.steps = value.parse().expect("--steps must be a positive integer");
                }
                "--min-steps-per-sec" | "--min-steps-per-second" => {
                    let value = args.next().expect("--min-steps-per-sec requires a value");
                    config.min_steps_per_second =
                        Some(value.parse().expect("--min-steps-per-sec must be a number"));
                }
                "--skip-eval" => config.skip_eval = true,
                "--progress-every" => {
                    let value = args.next().expect("--progress-every requires a value");
                    config.progress_every = value
                        .parse()
                        .expect("--progress-every must be a non-negative integer");
                }
                "--dtype" => {
                    let value = args.next().expect("--dtype requires a value");
                    config.dtype = match value.as_str() {
                        "f32" => Dtype::F32,
                        "f16" => Dtype::F16,
                        _ => panic!("--dtype must be f32 or f16"),
                    };
                }
                "--trace-host" => config.trace_host = true,
                "--trace-resolve" => config.trace_resolve = true,
                "--trace-names" => {
                    config.trace_resolve = true;
                    config.trace_names = true;
                }
                _ => panic!("unknown argument: {arg}"),
            }
        }
        assert!(config.steps > 0, "--steps must be greater than zero");
        config
    }
}

/// Block until all submitted GPU work has completed (no-op on CPU).
fn wait_for_gpu(device: &Device) {
    if let Device::Gpu(gpu) = device {
        gpu.poll_wait();
    }
}

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

/// Raw parameter tensors that persist across training steps. Masters stay in
/// f32 regardless of the model compute dtype.
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

/// Numerically stable softmax cross-entropy averaged over all positions:
/// log softmax via log-sum-exp so a saturated class cannot underflow.
fn cross_entropy(logits: &Tensor<2>, targets: &RawTensor<1, u32>) -> Tensor<0> {
    logits.softmax_cross_entropy(targets)
}

fn correct_predictions(logits: &Tensor<2>, targets: &RawTensor<1, u32>) -> Tensor<0> {
    assert_eq!(logits.shape()[0], targets.shape()[0]);
    logits
        .gather_last(targets)
        .eq_tensor(&logits.max::<1>(1))
        .sum()
}

/// First and second moment estimates for one parameter tensor, stored flat
/// so the update math is rank-independent.
struct AdamState {
    momentum: RawTensor<1, f32>,
    variance: RawTensor<1, f32>,
}

impl AdamState {
    fn zeros(device: &Device, elements: usize) -> Self {
        let zeros = vec![0.0; elements];
        Self {
            momentum: RawTensor::from_slice(device, [elements], &zeros),
            variance: RawTensor::from_slice(device, [elements], &zeros),
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
        (state.momentum.clone() * BETA1 + gradient.clone() * (1.0 - BETA1)).into_concrete();
    state.variance = (state.variance.clone() * BETA2
        + (gradient.clone() * gradient) * (1.0 - BETA2))
        .into_concrete();
    let update: RawTensor<1, f32> = state
        .momentum
        .mul_(lr)
        .div_(&state.variance.sqrt().add_scalar(EPSILON).into_concrete());
    *param = (param.clone().reshape([elements]) - update)
        .reshape(shape)
        .into_concrete();
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

async fn read_metrics(
    loss: RawTensor<0, f32>,
    correct: RawTensor<0, f32>,
    tokens: usize,
) -> (f32, f32) {
    let metrics = cat([loss.reshape([1]), correct.reshape([1])], 0)
        .as_slice()
        .await
        .unwrap()
        .to_vec();
    (metrics[0], metrics[1] / tokens as f32)
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

/// The model and training loop, written once over the element alias `E` and
/// stamped out per dtype. Masters, the optimizer, the final norm, the lm
/// head, and the loss always run in f32; everything between the embedding
/// lookup and the final norm runs in `E`.
macro_rules! gpt_model {
    ($mod_name:ident, $elem:ty) => {
        mod $mod_name {
            use super::*;

            type E = $elem;

            /// Model-dtype view of an f32 leaf: the identity for `E == f32`
            /// (keeping that graph byte-identical to the pure-f32 model), a
            /// differentiable cast otherwise (its backward casts the weight
            /// gradient back to the f32 master).
            fn to_model<const R: usize>(leaf: &Tensor<R>) -> Tensor<R, E> {
                let any: &dyn std::any::Any = leaf;
                match any.downcast_ref::<Tensor<R, E>>() {
                    Some(same) => same.clone(),
                    None => leaf.cast::<E>(),
                }
            }

            /// Back to f32 at norm and head boundaries; identity for `E == f32`.
            fn to_f32<const R: usize>(hidden: &Tensor<R, E>) -> Tensor<R> {
                let any: &dyn std::any::Any = hidden;
                match any.downcast_ref::<Tensor<R>>() {
                    Some(same) => same.clone(),
                    None => hidden.cast::<f32>(),
                }
            }

            /// Additive causal mask: 0 where position j <= i, a large negative
            /// value elsewhere so softmax zeroes out future positions. The
            /// mask value stays finite in the model dtype: -1e9 overflows f16
            /// to -inf, which turns masked-lane arithmetic into NaNs.
            pub fn causal_mask(device: &Device, seq: usize) -> RawTensor<2, E> {
                let mask_value = if std::mem::size_of::<E>() == 2 {
                    -6.0e4
                } else {
                    MASK_VALUE
                };
                let values: Vec<E> = (0..seq * seq)
                    .map(|i| {
                        E::from_f32(if i % seq <= i / seq { 0.0 } else { mask_value })
                    })
                    .collect();
                RawTensor::from_slice(device, [seq, seq], &values)
            }

            /// Pre-norm transformer block: causal self-attention then a gelu
            /// MLP, each behind a residual connection.
            pub struct Block {
                ln1: LayerNorm<1>,
                query: Linear<E>,
                key: Linear<E>,
                value: Linear<E>,
                output: Linear<E>,
                ln2: LayerNorm<1>,
                mlp_up: Linear<E>,
                mlp_down: Linear<E>,
            }

            impl Block {
                /// Multi-head causal self-attention for `x` of shape
                /// (batch, seq, dim). The fused attention primitive keeps the
                /// softmax probabilities out of the graph (its backward
                /// replays a recompute), so the compiler's flash kernel claims
                /// the forward cluster.
                fn attention(&self, x: &Tensor<3, E>, mask: &RawTensor<2, E>) -> Tensor<3, E> {
                    let [batch, seq, dim] = x.shape();
                    // (batch, seq, dim) -> (batch, heads, seq, head_dim)
                    let split = |projected: &Tensor<3, E>| {
                        projected
                            .reshape([batch, seq, HEADS, HEAD_DIM])
                            .transpose(1, 2)
                    };
                    let query = split(&self.query.forward(x));
                    let key = split(&self.key.forward(x));
                    let value = split(&self.value.forward(x));

                    let context = query
                        .attention(
                            &key,
                            &value,
                            1.0 / (HEAD_DIM as f32).sqrt(),
                            Some((mask, MaskKind::Causal)),
                        )
                        .transpose(1, 2)
                        .reshape([batch, seq, dim]);
                    self.output.forward(&context)
                }

                fn forward(&self, x: &Tensor<3, E>, mask: &RawTensor<2, E>) -> Tensor<3, E> {
                    // Norm statistics stay in f32 (the standard mixed-precision
                    // recipe); everything else runs in the model dtype.
                    let normed = to_model(&self.ln1.forward(&to_f32(x)));
                    let x = x.add(&self.attention(&normed, mask));
                    let normed = to_model(&self.ln2.forward(&to_f32(&x)));
                    let mlp = self.mlp_down.forward(&self.mlp_up.forward(&normed).gelu());
                    x.add(&mlp)
                }
            }

            pub struct Gpt {
                leaf_rank1: Vec<Tensor<1>>,
                leaf_rank2: Vec<Tensor<2>>,
                token_embedding: Embedding<E>,
                position_embedding: Embedding<E>,
                blocks: Vec<Block>,
                ln_final: LayerNorm<1>,
                lm_head: Linear,
            }

            impl Gpt {
                /// Build the model on `graph`, as trainable leaves (training)
                /// or constants (evaluation). Leaves are always f32 masters;
                /// the model weights are their `E` views.
                pub fn new(graph: &Graph, params: &Params, trainable: bool) -> Self {
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
                    // Leaf order must match [`Params::tensors_mut`].
                    let mut leaf_rank1 = vec![
                        wrap(graph, &params.ln_final_weight, trainable),
                        wrap(graph, &params.ln_final_bias, trainable),
                    ];
                    let mut leaf_rank2 = vec![
                        wrap(graph, &params.token_embedding, trainable),
                        wrap(graph, &params.position_embedding, trainable),
                        wrap(graph, &params.lm_head, trainable),
                    ];
                    let blocks = params
                        .blocks
                        .iter()
                        .map(|block| {
                            let ln1_weight = wrap(graph, &block.ln1_weight, trainable);
                            let ln1_bias = wrap(graph, &block.ln1_bias, trainable);
                            let ln2_weight = wrap(graph, &block.ln2_weight, trainable);
                            let ln2_bias = wrap(graph, &block.ln2_bias, trainable);
                            let mlp_up_bias = wrap(graph, &block.mlp_up_bias, trainable);
                            let mlp_down_bias = wrap(graph, &block.mlp_down_bias, trainable);
                            let query = wrap(graph, &block.query, trainable);
                            let key = wrap(graph, &block.key, trainable);
                            let value = wrap(graph, &block.value, trainable);
                            let output = wrap(graph, &block.output, trainable);
                            let mlp_up = wrap(graph, &block.mlp_up, trainable);
                            let mlp_down = wrap(graph, &block.mlp_down, trainable);
                            let built = Block {
                                ln1: LayerNorm::new(
                                    ln1_weight.clone(),
                                    Some(ln1_bias.clone()),
                                    LAYER_NORM_EPS,
                                ),
                                query: Linear::new(to_model(&query), None),
                                key: Linear::new(to_model(&key), None),
                                value: Linear::new(to_model(&value), None),
                                output: Linear::new(to_model(&output), None),
                                ln2: LayerNorm::new(
                                    ln2_weight.clone(),
                                    Some(ln2_bias.clone()),
                                    LAYER_NORM_EPS,
                                ),
                                mlp_up: Linear::new(
                                    to_model(&mlp_up),
                                    Some(to_model(&mlp_up_bias)),
                                ),
                                mlp_down: Linear::new(
                                    to_model(&mlp_down),
                                    Some(to_model(&mlp_down_bias)),
                                ),
                            };
                            leaf_rank1.extend([
                                ln1_weight,
                                ln1_bias,
                                ln2_weight,
                                ln2_bias,
                                mlp_up_bias,
                                mlp_down_bias,
                            ]);
                            leaf_rank2
                                .extend([query, key, value, output, mlp_up, mlp_down]);
                            built
                        })
                        .collect();
                    Self {
                        token_embedding: Embedding::new_from_tensor(to_model(&leaf_rank2[0])),
                        position_embedding: Embedding::new_from_tensor(to_model(
                            &leaf_rank2[1],
                        )),
                        blocks,
                        ln_final: LayerNorm::new(
                            leaf_rank1[0].clone(),
                            Some(leaf_rank1[1].clone()),
                            LAYER_NORM_EPS,
                        ),
                        lm_head: Linear::new(leaf_rank2[2].clone(), None),
                        leaf_rank1,
                        leaf_rank2,
                    }
                }

                /// Parameter leaves in a fixed order; must match
                /// [`Params::tensors_mut`].
                pub fn leaves(&self) -> (Vec<&Tensor<1>>, Vec<&Tensor<2>>) {
                    (
                        self.leaf_rank1.iter().collect(),
                        self.leaf_rank2.iter().collect(),
                    )
                }

                /// Input shape: (batch, seq) token ids. Output shape:
                /// (batch, seq, vocab), always f32.
                pub fn forward(
                    &self,
                    tokens: &RawTensor<2, u32>,
                    positions: &RawTensor<1, u32>,
                    mask: &RawTensor<2, E>,
                ) -> Tensor<3> {
                    let [batch, seq] = tokens.shape();
                    let position_embedded = self.position_embedding.forward(positions);
                    let mut x = self
                        .token_embedding
                        .forward(tokens)
                        .add(&position_embedded.broadcast_as([batch, seq, DIM]));
                    for block in &self.blocks {
                        x = block.forward(&x, mask);
                    }
                    self.lm_head.forward(&self.ln_final.forward(&to_f32(&x)))
                }
            }

            pub async fn evaluate_batch(
                device: &Device,
                params: &Params,
                inputs: &[u32],
                targets: &[u32],
                positions: &RawTensor<1, u32>,
                mask: &RawTensor<2, E>,
                vocab_size: usize,
            ) -> (f32, f32) {
                assert_eq!(inputs.len(), targets.len());
                assert!(inputs.len().is_multiple_of(CONTEXT));
                let batch_size = inputs.len() / CONTEXT;
                let inputs = RawTensor::from_slice(device, [batch_size, CONTEXT], inputs);
                let targets = RawTensor::from_slice(device, [targets.len()], targets);
                let graph = Graph::new();
                let model = Gpt::new(&graph, params, false);
                let logits = model.forward(&inputs, positions, mask);
                let flat_logits = logits.reshape([batch_size * CONTEXT, vocab_size]);
                let loss = cross_entropy(&flat_logits, &targets);
                let correct = correct_predictions(&flat_logits, &targets);
                read_metrics(
                    loss.raw().clone(),
                    correct.raw().clone(),
                    targets.shape()[0],
                )
                .await
            }

            async fn generate_synchronized(
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
                    let logits = model.forward(&input, &positions, &mask);
                    let vocab_size = logits.shape()[2];
                    let last_logits = logits
                        .raw()
                        .clone()
                        .narrow(1, CONTEXT - 1, 1)
                        .reshape([vocab_size])
                        .into_concrete();
                    drop(logits);
                    let last_logits = last_logits.as_slice().await.unwrap().to_vec();
                    tokens.push(sample(&last_logits, rng));
                }
                tokens
            }

            /// Keep a bounded window of autoregressive steps in flight using
            /// GPU-resident sampled tokens. Queue ordering preserves the
            /// sliding-window dependency; host readbacks are only needed to
            /// assemble the final output text.
            async fn generate_gpu_run_ahead(
                device: &Device,
                params: &Params,
                mut tokens: Vec<u32>,
                length: usize,
                rng: &mut Lcg,
            ) -> Option<Vec<u32>> {
                assert!(tokens.len() >= CONTEXT, "prompt shorter than CONTEXT");
                let positions: Vec<u32> = (0..CONTEXT as u32).collect();
                let positions = RawTensor::from_slice(device, [CONTEXT], &positions);
                let mask = causal_mask(device, CONTEXT);
                let prompt =
                    RawTensor::from_slice(device, [1, CONTEXT], &tokens[tokens.len() - CONTEXT..]);
                let mut input = prompt;
                let mut pending_tokens = VecDeque::with_capacity(GENERATION_RUN_AHEAD + 1);

                for _ in 0..length {
                    let graph = Graph::new();
                    let model = Gpt::new(&graph, params, false);
                    let logits = model.forward(&input, &positions, &mask);
                    let vocab_size = logits.shape()[2];
                    let last_logits = logits
                        .raw()
                        .clone()
                        .narrow(1, CONTEXT - 1, 1)
                        .reshape([vocab_size])
                        .into_concrete();
                    drop(logits);

                    let pending = last_logits
                        .sample_standard_token_pending(
                            &[],
                            None,
                            StandardSamplerParams {
                                top_k: vocab_size,
                                temperature: TEMPERATURE,
                                repetition_penalty: 1.0,
                                top_p: 1.0,
                                min_p: 0.0,
                                random: rng.next_f32(),
                            },
                        )
                        .ok()??;
                    let next = pending.token_tensor().reshape([1, 1]).into_concrete();
                    let retained = input.narrow(1, 1, CONTEXT - 1).into_concrete();
                    input = cat([retained, next], 1);
                    pending_tokens.push_back(pending);
                    if pending_tokens.len() > GENERATION_RUN_AHEAD {
                        let token = pending_tokens
                            .pop_front()
                            .expect("pending token queue cannot be empty")
                            .read_token()
                            .await
                            .ok()??;
                        tokens.push(token);
                    }
                }
                while let Some(pending) = pending_tokens.pop_front() {
                    let token = pending.read_token().await.ok()??;
                    tokens.push(token);
                }
                Some(tokens)
            }

            /// Autoregressively extend `tokens` by `length` sampled
            /// characters. The prompt must be at least CONTEXT tokens so every
            /// forward pass sees the same shapes and reuses the same compiled
            /// kernels.
            pub async fn generate(
                device: &Device,
                params: &Params,
                tokens: Vec<u32>,
                length: usize,
                rng: &mut Lcg,
            ) -> Vec<u32> {
                match device {
                    Device::Gpu(_) => {
                        let rng_state = rng.0;
                        if let Some(generated) =
                            generate_gpu_run_ahead(device, params, tokens.clone(), length, rng)
                                .await
                        {
                            generated
                        } else {
                            rng.0 = rng_state;
                            generate_synchronized(device, params, tokens, length, rng).await
                        }
                    }
                    Device::Cpu => {
                        generate_synchronized(device, params, tokens, length, rng).await
                    }
                }
            }

            pub async fn run(
                config: RunConfig,
                device: Device,
                vocab: Vec<u8>,
                train_tokens: &[u32],
                test_tokens: &[u32],
                all_tokens: &[u32],
            ) {
                let mut params = Params::new(&device, vocab.len());
                let (mut adam1, mut adam2) = {
                    let (rank1, rank2) = params.tensors_mut();
                    (
                        rank1
                            .iter()
                            .map(|tensor| {
                                AdamState::zeros(&device, tensor.shape().iter().product())
                            })
                            .collect::<Vec<_>>(),
                        rank2
                            .iter()
                            .map(|tensor| {
                                AdamState::zeros(&device, tensor.shape().iter().product())
                            })
                            .collect::<Vec<_>>(),
                    )
                };

                let positions: Vec<u32> = (0..CONTEXT as u32).collect();
                let positions = RawTensor::from_slice(&device, [CONTEXT], &positions);
                let mask = causal_mask(&device, CONTEXT);
                let (progress_test_inputs, progress_test_targets) = {
                    let mut test_rng = Lcg(0x5eed);
                    sample_batch(test_tokens, &mut test_rng, BATCH_SIZE)
                };

                let mut rng = Lcg(7);
                let start = std::time::Instant::now();
                for step in 0..config.steps {
                    let report_progress = config.progress_every != 0
                        && ((step + 1).is_multiple_of(config.progress_every)
                            || step + 1 == config.steps);
                    let progress = {
                        let (inputs, targets) =
                            sample_batch(train_tokens, &mut rng, BATCH_SIZE);
                        let inputs =
                            RawTensor::from_slice(&device, [BATCH_SIZE, CONTEXT], &inputs);
                        let targets =
                            RawTensor::from_slice(&device, [BATCH_SIZE * CONTEXT], &targets);

                        let graph = Graph::new();
                        let model = Gpt::new(&graph, &params, true);
                        let logits = model.forward(&inputs, &positions, &mask);
                        let flat_logits =
                            logits.reshape([BATCH_SIZE * CONTEXT, vocab.len()]);
                        let loss = cross_entropy(&flat_logits, &targets);
                        let progress = report_progress.then(|| {
                            let correct = correct_predictions(&flat_logits, &targets);
                            (loss.raw().clone(), correct.raw().clone())
                        });

                        // Gradients stay lazy: no per-step readback. The whole
                        // step (forward, backward, and the optimizer updates
                        // below) is submitted to the GPU by the flush after
                        // step-local temporaries are dropped.
                        //
                        // The f16 model seeds the backward with a loss scale:
                        // activation gradients entering the f16 blocks sit
                        // near 1e-6, deep in f16's subnormal range, and would
                        // quantize to noise. Adam's update is invariant to a
                        // constant gradient scale (m and sqrt(v) scale
                        // together), so no unscaling is needed.
                        let loss_scale = if std::mem::size_of::<E>() == 2 {
                            1024.0
                        } else {
                            1.0
                        };
                        let seed = RawTensor::splat(&device, loss_scale, []);
                        let gradients = loss.backward_with(seed).unwrap();

                        // Adam with warmup and bias correction folded into the
                        // learning rate.
                        let t = step as i32 + 1;
                        let warmup = (config.steps / 10).max(1);
                        let lr_value = LEARNING_RATE
                            * ((step + 1) as f32 / warmup as f32).min(1.0)
                            * (1.0 - BETA2.powi(t)).sqrt()
                            / (1.0 - BETA1.powi(t));
                        let lr = RawTensor::from_slice(&device, [1], &[lr_value]);
                        let (leaves1, leaves2) = model.leaves();
                        let (tensors1, tensors2) = params.tensors_mut();
                        for ((param, state), leaf) in
                            tensors1.into_iter().zip(&mut adam1).zip(leaves1)
                        {
                            adam_step(
                                param,
                                state,
                                gradients.get(leaf).expect("missing gradient"),
                                &lr,
                            );
                        }
                        for ((param, state), leaf) in
                            tensors2.into_iter().zip(&mut adam2).zip(leaves2)
                        {
                            adam_step(
                                param,
                                state,
                                gradients.get(leaf).expect("missing gradient"),
                                &lr,
                            );
                        }

                        progress
                    };

                    // Submit parameter updates after step-local
                    // forward/backward handles have dropped. Reporting steps
                    // intentionally retain two scalar metric targets; ordinary
                    // steps exclude loss, logits, and gradient views.
                    device.flush();

                    if let Some((loss, correct)) = progress {
                        let (loss, train_accuracy) =
                            read_metrics(loss, correct, BATCH_SIZE * CONTEXT).await;
                        let (_, test_accuracy) = evaluate_batch(
                            &device,
                            &params,
                            &progress_test_inputs,
                            &progress_test_targets,
                            &positions,
                            &mask,
                            vocab.len(),
                        )
                        .await;
                        println!(
                            "step {}/{}: loss {loss:.4} | train accuracy {:.2}% | test accuracy {:.2}%",
                            step + 1,
                            config.steps,
                            train_accuracy * 100.0,
                            test_accuracy * 100.0,
                        );
                    }
                }
                // Drain the GPU before timing and before exit: without this,
                // sparse progress reporting lets the loop report throughput
                // (or terminate the process) with steps still queued on the
                // GPU.
                wait_for_gpu(&device);
                let elapsed = start.elapsed();
                let steps_per_second = config.steps as f64 / elapsed.as_secs_f64();
                println!(
                    "trained {} steps ({} tokens) in {elapsed:.2?} ({steps_per_second:.1} steps/s)",
                    config.steps,
                    config.steps * BATCH_SIZE * CONTEXT,
                );

                if let Some(min_steps_per_second) = config.min_steps_per_second
                    && steps_per_second < min_steps_per_second
                {
                    panic!(
                        "transformer throughput {steps_per_second:.1} steps/s below required {min_steps_per_second:.1}"
                    );
                }

                if config.skip_eval {
                    return;
                }

                // Final metrics over a larger held-out sample.
                const TEST_BATCHES: usize = 10;
                const TEST_BATCH_SIZE: usize = TEST_BATCHES * BATCH_SIZE;
                let (test_inputs, test_targets) =
                    sample_batch(test_tokens, &mut rng, TEST_BATCH_SIZE);
                let (test_loss, test_accuracy) = evaluate_batch(
                    &device,
                    &params,
                    &test_inputs,
                    &test_targets,
                    &positions,
                    &mask,
                    vocab.len(),
                )
                .await;
                println!(
                    "test loss: {test_loss:.4} | test accuracy {:.2}%",
                    test_accuracy * 100.0
                );

                // Prompt with the tail of the corpus so the window is always
                // full.
                let prompt = all_tokens[all_tokens.len() - CONTEXT..].to_vec();
                let generated = generate(&device, &params, prompt, 400, &mut rng).await;
                let text: String = generated[CONTEXT..]
                    .iter()
                    .map(|&id| vocab[id as usize] as char)
                    .collect();
                println!("--- sample ---\n{text}");
            }
        }
    };
}

gpt_model!(gpt_f32, f32);
gpt_model!(gpt_f16, half::f16);

#[tokio::main]
async fn main() {
    let config = RunConfig::from_args();
    if std::env::var_os("RUST_LOG").is_some() || config.trace_host || config.trace_resolve {
        let env_filter = if std::env::var_os("RUST_LOG").is_some() {
            tracing_subscriber::EnvFilter::from_default_env()
        } else {
            tracing_subscriber::EnvFilter::new(
                "fusor_core::compute_graph::resolve=info,fusor_tile_ir_runtime::plan_cache=warn",
            )
        };
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
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
    let (train_tokens, test_tokens) = tokens.split_at(split);
    println!(
        "corpus: {} chars, vocab {}, {} train / {} test",
        tokens.len(),
        vocab.len(),
        train_tokens.len(),
        test_tokens.len()
    );

    let device = match Device::gpu().await {
        Ok(gpu) => gpu,
        Err(_) => {
            println!("GPU unavailable, training on CPU");
            Device::cpu()
        }
    };
    if config.trace_host || config.trace_resolve {
        // SAFETY: this example sets resolver trace flags before building any
        // training graph; the flags are read by Fusor during subsequent
        // single-threaded graph resolution.
        unsafe {
            std::env::set_var("FUSOR_TRACE_RESOLVE_HOST", "1");
            if config.trace_resolve {
                std::env::set_var("FUSOR_TRACE_RESOLVE", "1");
            }
            if config.trace_names {
                std::env::set_var("FUSOR_TRACE_DECODE_NAMES", "1");
            }
        }
    }

    match config.dtype {
        Dtype::F32 => {
            gpt_f32::run(config, device, vocab, train_tokens, test_tokens, &tokens).await
        }
        Dtype::F16 => {
            gpt_f16::run(config, device, vocab, train_tokens, test_tokens, &tokens).await
        }
    }
}
