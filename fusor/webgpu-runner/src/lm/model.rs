//! Character-level transformer with reusable training and observation programs.
//! Inputs change between steps; parameters and Adam moments remain on the GPU.
//! Session execution is available as a numerical reference and benchmark.

use fusor::cache::MaskKind;
use fusor::layers::{Embedding, Linear};
use fusor::optim::cosine_decay;
use fusor::tensor::Dyn;
use fusor::{Device, Dtype, Result, Tensor};

use super::corpus::{Corpus, Split};
use super::rng::Rng;

use super::config::ModelConfig;

/// Peak step size, reached at the end of warmup, then cosine-decayed.
pub const LEARNING_RATE: f32 = 3e-3;
/// Where the schedule settles.
pub const FLOOR_RATE: f32 = 3e-4;
/// Learning-rate warmup steps.
const WARMUP: u64 = 40;
/// Steps from warmup to the floor.
const DECAY: u64 = 1400;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
/// Parameter standard deviation, scaled by depth for residual projections.
const INIT_STD: f32 = 0.02;
/// Finite mask magnitude whose negative exponential underflows to zero.
const MASK_FLOOR: f32 = 1e9;

/// What one run of steps measured.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct StepStats {
    /// Optimizer steps completed in total.
    pub step: u64,
    /// Cross-entropy in nats on the last training batch.
    pub loss: f32,
    /// The step size that step was taken at.
    pub rate: f32,
}

impl StepStats {
    /// Per-character perplexity: how many characters the model is effectively
    /// choosing between at each position.
    pub fn perplexity(&self) -> f32 {
        self.loss.exp()
    }
}

/// What the held-out split says.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct Evaluation {
    /// Cross-entropy in nats on text the optimizer never saw.
    pub loss: f32,
    /// Share of positions whose true next character was the model's argmax.
    pub accuracy: f32,
}

impl Evaluation {
    /// Per-character perplexity on held-out text.
    pub fn perplexity(&self) -> f32 {
        self.loss.exp()
    }
}

/// One transformer block's parameters.
#[derive(Clone)]
struct Block {
    attn_norm: Tensor<1, f32>,
    q: Tensor<2, f32>,
    k: Tensor<2, f32>,
    v: Tensor<2, f32>,
    proj: Tensor<2, f32>,
    mlp_norm: Tensor<1, f32>,
    up: Tensor<2, f32>,
    down: Tensor<2, f32>,
}

struct Weights {
    embed: Tensor<2, f32>,
    positions: Tensor<2, f32>,
    blocks: Vec<Block>,
    final_norm: Tensor<1, f32>,
    head: Tensor<2, f32>,
}

/// The batch-of-one graph the generator and the interpretability panels run.
struct Single {
    /// An observation program snapshots parameters at this optimizer step.
    compiled: Option<(u64, fusor::program::TrainingProgram)>,
    /// `[1, context]` token ids, left-aligned by the caller.
    tokens: Tensor<2, u32>,
    /// `[1, context, vocab]`, through the same fused attention the training
    /// graph uses.
    logits: Tensor<3, f32>,
    /// The same logits through an unfused, explicitly-masked softmax — the
    /// path the attention maps come out of.
    lens_logits: Tensor<3, f32>,
    /// `[heads, context, context]` attention probabilities per block.
    attention: Vec<Tensor<3, f32>>,
    /// Everything to invalidate when the parameters move.
    chain: Vec<Dyn>,
}

/// The model, its optimizer state, and the graphs that read it.
pub struct Lm {
    compiled: Option<fusor::program::TrainingProgram>,
    published_step: Option<u64>,
    device: Device,
    vocab: usize,
    config: ModelConfig,
    schedule: (u64, u64),

    tokens: Tensor<2, u32>,
    labels: Tensor<1, f32>,
    alpha: Tensor<1, f32>,
    loss: Tensor<0, f32>,
    /// `[loss, accuracy]` over the current batch, for [`Lm::evaluate`].
    scored: Tensor<1, f32>,

    weights: Weights,
    feedback: Vec<(Dyn, Dyn)>,
    roots: Vec<Dyn>,
    /// The forward's intermediates, cleared whenever a leaf's bytes change.
    chain: Vec<Dyn>,

    single: Option<Single>,
    step: u64,
    rng: Rng,
    pub learning_rate: f32,
    pub floor_rate: f32,
}

/// Uniform samples with mean zero and standard deviation `std`.
fn init(rng: &mut Rng, n: usize, std: f32) -> Vec<f32> {
    (0..n).map(|_| rng.signed() * std * 1.732_050_8).collect()
}

impl Lm {
    pub fn config(&self) -> ModelConfig {
        self.config
    }

    /// Set the schedule for this run before its first optimizer step.
    pub fn set_training_steps(&mut self, steps: u64) -> Result<()> {
        if self.step != 0 || steps == 0 {
            return Err(fusor::Error::Plan(
                "Set a positive training budget before training starts.".into(),
            ));
        }
        self.schedule = ((steps / 10).clamp(1, WARMUP), steps.max(2));
        Ok(())
    }

    fn validate_corpus(&self, corpus: &Corpus) -> Result<()> {
        if corpus.vocab_size() != self.vocab
            || corpus.split_len(Split::Train) <= self.config.context
            || corpus.split_len(Split::Test) <= self.config.context
        {
            return Err(fusor::Error::Shape("The corpus must match the vocabulary and contain a full context plus target in each split.".into()));
        }
        Ok(())
    }

    /// Build the device, the parameters and the step graph.
    pub async fn new(vocab: usize, seed: u32, config: ModelConfig) -> Result<Self> {
        config.validate(vocab).map_err(fusor::Error::Shape)?;
        let device = Device::gpu().await?;
        let mut rng = Rng::new(seed);

        // Every leaf here is an *external* leaf — `Tensor::zeros` would mint a
        // constant, and neither `set_elements` nor `adopt_buffer` accepts one.
        let tokens = Tensor::<2, u32>::from_slice(
            &device,
            [config.batch, config.context],
            &vec![0u32; config.tokens()],
        );
        let labels =
            Tensor::<1, f32>::from_slice(&device, [config.tokens()], &vec![0.0; config.tokens()]);
        let alpha = Tensor::<1, f32>::from_slice(&device, [1], &[0.0]);

        let weight = |rng: &mut Rng, inn: usize, out: usize, std: f32| {
            Tensor::<2, f32>::from_slice(&device, [out, inn], &init(rng, out * inn, std))
        };
        let ones = |n: usize| Tensor::<1, f32>::from_slice(&device, [n], &vec![1.0; n]);
        // A residual projection is summed into the stream once per block, so
        // its share of the variance is divided by how many such sums there are.
        let residual = INIT_STD / ((2 * config.blocks) as f32).sqrt();

        let embed = Tensor::<2, f32>::from_slice(
            &device,
            [vocab, config.dim],
            &init(&mut rng, vocab * config.dim, INIT_STD),
        );
        let positions = Tensor::<2, f32>::from_slice(
            &device,
            [config.context, config.dim],
            &init(&mut rng, config.context * config.dim, INIT_STD / 2.0),
        );
        let blocks: Vec<Block> = (0..config.blocks)
            .map(|_| Block {
                attn_norm: ones(config.dim),
                q: weight(&mut rng, config.dim, config.dim, INIT_STD),
                k: weight(&mut rng, config.dim, config.dim, INIT_STD),
                v: weight(&mut rng, config.dim, config.dim, INIT_STD),
                proj: weight(&mut rng, config.dim, config.dim, residual),
                mlp_norm: ones(config.dim),
                up: weight(&mut rng, config.dim, config.mlp, INIT_STD),
                down: weight(&mut rng, config.mlp, config.dim, residual),
            })
            .collect();
        let final_norm = ones(config.dim);
        let head = weight(&mut rng, config.dim, vocab, INIT_STD);

        let weights = Weights {
            embed,
            positions,
            blocks,
            final_norm,
            head,
        };
        let mut chain = Vec::new();
        let logits = weights.forward(&tokens, config, &mut chain, None);
        let flat = logits.reshape([config.tokens(), vocab]);
        chain.push(flat.as_dyn().clone());

        // Cross-entropy against a one-hot the *device* builds: uploading a
        // dense `[config.tokens(), vocab]` target every step would be a quarter of a
        // megabyte of host traffic for what is one comparison in a kernel.
        let ids = Tensor::<1, f32>::arange(&device, 0.0, vocab as f64);
        let onehot = ids
            .reshape([1, vocab])
            .sub_::<2, 2, _>(&labels.reshape([config.tokens(), 1]))
            .abs()
            .lte_scalar(0.5);
        let shifted = flat.sub_::<2, 2, _>(&flat.max_keepdim(1usize));
        let log_prob = shifted.sub_::<2, 2, _>(&shifted.exp().sum_keepdim(1usize).log());
        let loss = log_prob
            .mul(&onehot)
            .sum::<1>(1usize)
            .sum::<0>(0usize)
            .neg()
            .div_scalar(config.tokens() as f32);

        // Next-character accuracy, scored where the logits already are. The
        // true character's logit against the row's best: reading a
        // `[1024, 65]` tile back to count matches on the host is a quarter of
        // a megabyte per batch, and in a browser every readback is a frame.
        let picked = flat.mul(&onehot).sum::<1>(1usize);
        let best = flat.max::<1>(1usize);
        let hits = picked
            .sub(&best)
            .add_scalar(1e-6)
            .gte_scalar(0.0)
            .sum::<0>(0usize)
            .div_scalar(config.tokens() as f32);
        // One value to read rather than two: a second readback costs another
        // trip through the browser's event loop for four bytes.
        let scored = fusor::stack::<0, 1, f32, _>([loss.clone(), hits], 0);

        let parameters = weights.parameters();

        // Forward and backward are one graph with one root set; the extractor
        // decides what to keep and what to recompute.
        let gradients = device.graph().backward_with(loss.as_dyn(), &parameters)?;

        let alpha_dyn = alpha.as_dyn().clone();
        let mut feedback = Vec::with_capacity(parameters.len() * 3);
        let mut roots = vec![loss.as_dyn().clone()];
        for value in parameters {
            let gradient = gradients
                .get(&value)
                .ok_or_else(|| fusor::Error::Plan("a parameter received no gradient".into()))?;
            let count = value.elem_count().ok_or_else(|| {
                fusor::Error::Shape(
                    "a parameter with a symbolic extent cannot hold Adam state".into(),
                )
            })? as usize;
            let bytes = vec![0u8; count * 4];
            let handle = device.graph().handle();
            let shape = value.shape();
            let m = Dyn::from_slice(handle, Dtype::F32, &shape, &bytes)?;
            let v = Dyn::from_slice(handle, Dtype::F32, &shape, &bytes)?;

            // Adam, with the bias correction folded into `alpha` on the host.
            let next_m = m
                .mul_scalar(BETA1)?
                .add(&gradient.mul_scalar(1.0 - BETA1)?)?;
            let next_v = v
                .mul_scalar(BETA2)?
                .add(&gradient.sqr()?.mul_scalar(1.0 - BETA2)?)?;
            let update = next_m
                .mul_(&alpha_dyn)?
                .div(&next_v.sqrt()?.add_scalar(EPS)?)?;
            let next_value = value.sub(&update)?;
            let next = [next_value, next_m, next_v];
            roots.extend(next.iter().cloned());
            feedback.extend([value, m, v].into_iter().zip(next));
        }

        Ok(Self {
            compiled: None,
            published_step: None,
            device,
            vocab,
            config,
            schedule: (WARMUP, WARMUP + DECAY),
            tokens,
            labels,
            alpha,
            loss,
            scored,
            weights,
            feedback,
            roots,
            chain,
            single: None,
            step: 0,
            rng: Rng::new(seed ^ 0x9e37_79b9),
            learning_rate: LEARNING_RATE,
            floor_rate: FLOOR_RATE,
        })
    }

    /// Opt into a compiled logical training step. The ordinary compiler stays
    /// available so callers can compare full-step performance on their device.
    #[allow(dead_code)] // Optional executor; also used by the headless benchmark.
    pub async fn compile_training(
        &mut self,
        options: fusor::program::ProgramOptions,
    ) -> Result<()> {
        if let Some(program) = &self.compiled {
            let state: Vec<_> = self
                .feedback
                .iter()
                .map(|(state, _)| state.clone())
                .collect();
            program.export(&state)?;
        }
        let program = fusor::program::TrainingProgram::compile_with_options(
            &self.roots,
            &self.feedback,
            options,
        )
        .await?;
        self.compiled = Some(program);
        self.published_step = None;
        Ok(())
    }
    #[allow(dead_code)] // Diagnostics for callers selecting the optional executor.
    pub fn program_stats(&self) -> Option<&fusor::program::ProgramStats> {
        self.compiled.as_ref().map(|p| p.stats())
    }
    fn publish_parameters(&mut self) -> Result<()> {
        if self.published_step != Some(self.step) {
            if let Some(program) = &self.compiled {
                program.export(&self.weights.parameters())?;
            }
            self.published_step = Some(self.step);
        }
        Ok(())
    }

    /// Completed optimizer steps.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// GPU dispatches issued since device creation.
    pub fn dispatch_count(&self) -> u64 {
        self.compiled.as_ref().map_or_else(
            || self.device.session().launch_count(),
            |p| p.dispatch_count(),
        )
    }

    /// Queue `steps` optimizer steps, then read the final loss and step size.
    pub async fn train(&mut self, corpus: &Corpus, steps: usize) -> Result<StepStats> {
        self.validate_corpus(corpus)?;
        let mut rate = 0.0;
        for _ in 0..steps.max(1) {
            rate = self.dispatch(corpus).await?;
        }
        let loss = if let Some(program) = &self.compiled {
            let bytes = program.read(self.loss.as_dyn()).await?;
            vec![f32::from_le_bytes(bytes.try_into().map_err(|_| {
                fusor::Error::Shape("loss must be scalar f32".into())
            })?)]
        } else {
            self.loss.to_vec_f32_async().await?
        };
        Ok(StepStats {
            step: self.step,
            loss: loss.first().copied().unwrap_or(f32::NAN),
            rate,
        })
    }

    /// One step, dispatched but not waited on. Returns the step size taken.
    async fn dispatch(&mut self, corpus: &Corpus) -> Result<f32> {
        let (tokens, labels) = self.draw(corpus, Split::Train);

        self.step += 1;
        let t = self.step.min(i32::MAX as u64) as i32;
        let rate = cosine_decay(
            self.step,
            self.schedule.0,
            self.schedule.1,
            self.learning_rate,
            self.floor_rate,
        );
        // `alpha = lr * sqrt(1 - beta2^t) / (1 - beta1^t)`, the Keras form.
        let alpha = rate * (1.0 - BETA2.powi(t)).sqrt() / (1.0 - BETA1.powi(t));

        if let Some(program) = &mut self.compiled {
            let token_bytes: Vec<_> = tokens.iter().flat_map(|x| x.to_le_bytes()).collect();
            let label_bytes: Vec<_> = labels.iter().flat_map(|x| x.to_le_bytes()).collect();
            program.write(self.tokens.as_dyn(), &token_bytes)?;
            program.write(self.labels.as_dyn(), &label_bytes)?;
            program.write(self.alpha.as_dyn(), &alpha.to_le_bytes())?;
            program.run_async().await?;
            return Ok(rate);
        }
        self.tokens.set_elements(&tokens);
        self.labels.set_elements(&labels);
        self.alpha.set_elements(&[alpha]);

        // `set_elements` invalidates the leaves; the values computed from them
        // still hold last step's buffers, and a resolve returns early for a
        // root that already has one.
        for value in self.chain.iter().chain(&self.roots) {
            value.clear_device_buf();
        }
        self.device.session().resolve(&self.roots)?;

        // Device-side detach: every leaf takes over the buffer its update
        // landed in, so the next step re-runs the very same graph.
        for (state, next) in &self.feedback {
            state.adopt_buffer(next)?;
        }
        Ok(rate)
    }

    /// Score held-out batches without updating parameters or optimizer state.
    pub async fn evaluate(&mut self, corpus: &Corpus, batches: usize) -> Result<Evaluation> {
        self.validate_corpus(corpus)?;
        self.publish_parameters()?;
        // Use the selected executor for observations as well as updates. This
        // program has no feedback, so held-out inputs cannot train the model.
        let mut compiled = if self.compiled.is_some() {
            Some(
                fusor::program::TrainingProgram::compile(
                    std::slice::from_ref(self.scored.as_dyn()),
                    &[],
                )
                .await?,
            )
        } else {
            None
        };
        let (mut loss, mut accuracy) = (0.0f32, 0.0f32);
        let runs = batches.max(1);
        for _ in 0..runs {
            let (tokens, labels) = self.draw(corpus, Split::Test);
            if let Some(program) = &mut compiled {
                let token_bytes: Vec<_> = tokens.iter().flat_map(|x| x.to_le_bytes()).collect();
                let label_bytes: Vec<_> = labels.iter().flat_map(|x| x.to_le_bytes()).collect();
                program.write(self.tokens.as_dyn(), &token_bytes)?;
                program.write(self.labels.as_dyn(), &label_bytes)?;
                program.run_async().await?;
                let bytes = program.read(self.scored.as_dyn()).await?;
                let read: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|x| f32::from_le_bytes(*x))
                    .collect();
                loss += read[0];
                accuracy += read[1];
                continue;
            }
            self.tokens.set_elements(&tokens);
            self.labels.set_elements(&labels);
            for value in self.chain.iter().chain(&self.roots) {
                value.clear_device_buf();
            }
            self.scored.as_dyn().clear_device_buf();
            self.device
                .session()
                .resolve(std::slice::from_ref(self.scored.as_dyn()))?;
            let read = self.scored.to_vec_f32_async().await?;
            loss += read.first().copied().unwrap_or(f32::NAN);
            accuracy += read.get(1).copied().unwrap_or(0.0);
        }
        Ok(Evaluation {
            loss: loss / runs as f32,
            accuracy: accuracy / runs as f32,
        })
    }

    /// A batch of windows and their next-character targets.
    fn draw(&mut self, corpus: &Corpus, split: Split) -> (Vec<u32>, Vec<f32>) {
        let mut tokens = vec![0u32; self.config.tokens()];
        let mut labels = vec![0.0f32; self.config.tokens()];
        for row in 0..self.config.batch {
            let window = corpus.window(split, self.rng.below(corpus.len()), self.config.context);
            for position in 0..self.config.context {
                let at = row * self.config.context + position;
                tokens[at] = u32::from(window[position]);
                labels[at] = f32::from(window[position + 1]);
            }
        }
        (tokens, labels)
    }

    /// The next-character distribution after `context`, at `temperature`.
    ///
    /// `context` is the tail of what has been written so far; only its last
    /// `config.context` characters reach the model.
    pub async fn next_char(&mut self, context: &[u8], temperature: f32) -> Result<Vec<f32>> {
        let (row, filled) = window_of(context, self.config.context);
        self.single_forward(&row).await?;
        let single = self.expect_single()?;
        let logits = single.logits.to_vec_f32_async().await?;
        let last = (filled - 1) * self.vocab;
        Ok(soften(
            &logits[last..last + self.vocab],
            temperature.max(0.01),
        ))
    }

    /// Generate `count` characters; each character requires a logits readback.
    pub async fn generate(
        &mut self,
        corpus: &Corpus,
        prompt: &[u8],
        count: usize,
        temperature: f32,
    ) -> Result<String> {
        let mut written: Vec<u8> = prompt.to_vec();
        let mut out = String::new();
        for _ in 0..count {
            let probabilities = self.next_char(&written, temperature).await?;
            let picked = self.rng.pick(&probabilities);
            out.push_str(corpus.decode(picked));
            written.push(picked as u8);
        }
        Ok(out)
    }

    /// Attention probabilities for `context`, block by block and head by head.
    ///
    /// Returns `[self.config.blocks][self.config.heads][self.config.context * self.config.context]` row-major probabilities
    /// and how many of the positions are real rather than right padding.
    pub async fn attention(&mut self, context: &[u8]) -> Result<Attention> {
        let (row, filled) = window_of(context, self.config.context);
        self.single_forward(&row).await?;
        let single = self.expect_single()?;
        let mut maps = Vec::with_capacity(self.config.blocks);
        for block in &single.attention {
            let flat = block.to_vec_f32_async().await?;
            maps.push(
                flat.chunks_exact(self.config.context * self.config.context)
                    .map(<[f32]>::to_vec)
                    .collect::<Vec<_>>(),
            );
        }
        // The lens is a second implementation of the model's own attention;
        // if the two disagree the maps are a picture of something else.
        let fused = single.logits.to_vec_f32_async().await?;
        let lens = single.lens_logits.to_vec_f32_async().await?;
        let start = (filled - 1) * self.vocab;
        let disagreement = fused[start..start + self.vocab]
            .iter()
            .zip(&lens[start..start + self.vocab])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        Ok(Attention {
            maps,
            context: self.config.context,
            filled,
            disagreement,
        })
    }

    /// Pairwise cosine similarity between token embeddings.
    pub async fn embedding_similarity(&mut self) -> Result<Vec<f32>> {
        self.publish_parameters()?;
        self.weights.embed.as_dyn().clear_device_buf();
        let rows = self.weights.embed.to_vec_f32_async().await?;
        let v = self.vocab;
        let norms: Vec<f32> = (0..v)
            .map(|i| {
                rows[i * self.config.dim..(i + 1) * self.config.dim]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-6)
            })
            .collect();
        let mut out = vec![0.0f32; v * v];
        for i in 0..v {
            for j in 0..v {
                let dot: f32 = rows[i * self.config.dim..(i + 1) * self.config.dim]
                    .iter()
                    .zip(&rows[j * self.config.dim..(j + 1) * self.config.dim])
                    .map(|(a, b)| a * b)
                    .sum();
                out[i * v + j] = dot / (norms[i] * norms[j]);
            }
        }
        Ok(out)
    }

    /// Run the batch-of-one graph over `row`, building it on first use.
    async fn single_forward(&mut self, row: &[u32]) -> Result<()> {
        self.publish_parameters()?;
        if self.single.is_none() {
            self.single = Some(self.build_single()?);
        }
        let Some(single) = self.single.as_mut() else {
            return Err(fusor::Error::Plan("the read-out graph went missing".into()));
        };
        single.tokens.set_elements(row);
        for value in &single.chain {
            value.clear_device_buf();
        }
        let mut roots = vec![
            single.logits.as_dyn().clone(),
            single.lens_logits.as_dyn().clone(),
        ];
        roots.extend(single.attention.iter().map(|a| a.as_dyn().clone()));
        if self.compiled.is_some() {
            if single.compiled.as_ref().map(|(step, _)| *step) != Some(self.step) {
                single.compiled = Some((
                    self.step,
                    fusor::program::TrainingProgram::compile(&roots, &[]).await?,
                ));
            }
            let program = &mut single.compiled.as_mut().unwrap().1;
            let token_bytes: Vec<_> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
            program.write(single.tokens.as_dyn(), &token_bytes)?;
            program.run_async().await?;
            program.export(&roots)?;
            return Ok(());
        }
        self.device.session().resolve(&roots)?;
        Ok(())
    }

    fn expect_single(&self) -> Result<&Single> {
        self.single
            .as_ref()
            .ok_or_else(|| fusor::Error::Plan("the read-out graph went missing".into()))
    }

    /// The batch-of-one graph: the model's own forward, plus an unfused
    /// attention beside it that the maps are read out of.
    fn build_single(&self) -> Result<Single> {
        let tokens = Tensor::<2, u32>::from_slice(
            &self.device,
            [1, self.config.context],
            &vec![0u32; self.config.context],
        );
        let mut chain = Vec::new();
        let logits = self.weights.forward(&tokens, self.config, &mut chain, None);
        let mut attention = Vec::with_capacity(self.config.blocks);
        let lens_logits =
            self.weights
                .forward(&tokens, self.config, &mut chain, Some(&mut attention));

        Ok(Single {
            compiled: None,
            tokens,
            logits,
            lens_logits,
            attention,
            chain,
        })
    }
}

impl Block {
    /// Pre-norm feed-forward, added back into the residual stream.
    fn feed_forward(
        &self,
        x: &Tensor<3, f32>,
        rows: usize,
        config: ModelConfig,
        chain: &mut Vec<Dyn>,
    ) -> Tensor<3, f32> {
        let h = x
            .rms_norm(&self.mlp_norm, 1e-5)
            .reshape([rows * config.context, config.dim]);
        chain.push(h.as_dyn().clone());
        let up: Tensor<2, f32> = Linear::new(self.up.clone(), None).forward(&h);
        chain.push(up.as_dyn().clone());
        let down: Tensor<2, f32> = Linear::new(self.down.clone(), None).forward(&up.gelu());
        let out = x.add(&down.reshape([rows, config.context, config.dim]));
        chain.push(out.as_dyn().clone());
        out
    }
}

impl Weights {
    fn parameters(&self) -> Vec<Dyn> {
        let mut parameters = vec![self.embed.as_dyn().clone(), self.positions.as_dyn().clone()];
        for block in &self.blocks {
            parameters.extend(
                [
                    block.attn_norm.as_dyn(),
                    block.q.as_dyn(),
                    block.k.as_dyn(),
                    block.v.as_dyn(),
                    block.proj.as_dyn(),
                    block.mlp_norm.as_dyn(),
                    block.up.as_dyn(),
                    block.down.as_dyn(),
                ]
                .into_iter()
                .cloned(),
            );
        }
        parameters.extend([self.final_norm.as_dyn().clone(), self.head.as_dyn().clone()]);
        parameters
    }

    /// The same transformer builds training, generation and attention-map graphs.
    /// Requesting maps exposes the probabilities through explicit masked attention.
    fn forward(
        &self,
        tokens: &Tensor<2, u32>,
        config: ModelConfig,
        chain: &mut Vec<Dyn>,
        mut maps: Option<&mut Vec<Tensor<3, f32>>>,
    ) -> Tensor<3, f32> {
        let rows = tokens.shape()[0];
        let mask = maps.as_ref().map(|_| {
            let causal: Vec<_> = (0..config.context * config.context)
                .map(|i| {
                    if i % config.context > i / config.context {
                        -MASK_FLOOR
                    } else {
                        0.0
                    }
                })
                .collect();
            Tensor::<2, f32>::from_slice(
                &tokens.device(),
                [config.context, config.context],
                &causal,
            )
        });
        let embedded: Tensor<3, f32> = Embedding::new(self.embed.clone()).forward(tokens);
        let mut x = embedded.add_::<2, 3, _>(&self.positions);
        chain.push(x.as_dyn().clone());
        for block in &self.blocks {
            let h = x
                .rms_norm(&block.attn_norm, 1e-5)
                .reshape([rows * config.context, config.dim]);
            chain.push(h.as_dyn().clone());
            let heads = |w: &Tensor<2, f32>| -> Tensor<4, f32> {
                let projected: Tensor<2, f32> = Linear::new(w.clone(), None).forward(&h);
                projected
                    .reshape([rows, config.context, config.heads, config.head_dim()])
                    .permute([0, 2, 1, 3])
            };
            let (q, k, v) = (heads(&block.q), heads(&block.k), heads(&block.v));
            let attended = match (&mask, maps.as_mut()) {
                (Some(mask), Some(maps)) => {
                    let probabilities = q
                        .matmul_t(&k)
                        .mul_scalar(1.0 / (config.head_dim() as f32).sqrt())
                        .add_::<2, 4, _>(mask)
                        .softmax(3usize);
                    let map = probabilities.reshape([
                        rows * config.heads,
                        config.context,
                        config.context,
                    ]);
                    chain.extend([probabilities.as_dyn().clone(), map.as_dyn().clone()]);
                    maps.push(map);
                    probabilities.matmul(&v)
                }
                _ => q.attention(&k, &v, MaskKind::Causal, None),
            };
            let merged = attended
                .permute([0, 2, 1, 3])
                .reshape([rows * config.context, config.dim]);
            chain.push(merged.as_dyn().clone());
            let projected: Tensor<2, f32> = Linear::new(block.proj.clone(), None).forward(&merged);
            x = x.add(&projected.reshape([rows, config.context, config.dim]));
            chain.push(x.as_dyn().clone());
            x = block.feed_forward(&x, rows, config, chain);
        }
        let normed = x.rms_norm(&self.final_norm, 1e-5);
        let logits = Linear::new(self.head.clone(), None).forward(&normed);
        chain.extend([normed.as_dyn().clone(), logits.as_dyn().clone()]);
        logits
    }
}

/// What the attention panel draws.
pub struct Attention {
    /// `[blocks][heads]` maps of `context * context` probabilities.
    pub maps: Vec<Vec<Vec<f32>>>,
    /// How many of the `context` positions carry real characters; the rest
    /// are right padding the reader should not be shown.
    pub filled: usize,
    pub context: usize,
    /// The largest gap between the model's own logits and the lens's, on the
    /// position the panel is about. A picture of the model is only a picture
    /// of the model while this is small.
    pub disagreement: f32,
}

/// The last configured window of characters of `context`, laid into a window from the
/// left, and how many of them are real.
///
/// The trailing slots stay zero. Nothing reads them: the caller takes the
/// last *real* position, and a causal mask means that position attends to
/// nothing after itself.
fn window_of(context: &[u8], size: usize) -> (Vec<u32>, usize) {
    let take = context.len().clamp(1, size);
    let from = context.len().saturating_sub(take);
    let mut row = vec![0u32; size];
    for (slot, token) in row.iter_mut().zip(&context[from..]) {
        *slot = u32::from(*token);
    }
    (row, take)
}

/// Softmax at `temperature`. Lower is more confident, and the panel shows the
/// same numbers the sampler draws from.
fn soften(logits: &[f32], temperature: f32) -> Vec<f32> {
    let peak = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits
        .iter()
        .map(|l| ((l - peak) / temperature).exp())
        .collect();
    let total: f32 = exp.iter().sum();
    exp.iter().map(|e| e / total.max(1e-30)).collect()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn compiled_training_matches_reference_state_and_observations() {
        pollster::block_on(async {
            for config in [
                ModelConfig::TINY,
                ModelConfig::default(),
                ModelConfig {
                    blocks: 2,
                    dim: 42,
                    heads: 3,
                    mlp: 75,
                    context: 19,
                    batch: 3,
                },
            ] {
                eprintln!("checking {config:?}");
                let corpus = if config == ModelConfig::TINY {
                    Corpus::benchmark().await.unwrap()
                } else {
                    Corpus::load().await.unwrap()
                };
                let mut reference = Lm::new(corpus.vocab_size(), 0x51ed_c0de, config)
                    .await
                    .unwrap();
                let mut compiled = Lm::new(corpus.vocab_size(), 0x51ed_c0de, config)
                    .await
                    .unwrap();
                compiled.compile_training(Default::default()).await.unwrap();
                let allocated: usize = compiled
                    .weights
                    .parameters()
                    .iter()
                    .map(|s| s.elem_count().unwrap() as usize)
                    .sum();
                assert_eq!(allocated, config.parameters(corpus.vocab_size()));
                for _ in 0..3 {
                    let a = reference.train(&corpus, 8).await.unwrap();
                    let b = compiled.train(&corpus, 8).await.unwrap();
                    assert!((a.loss - b.loss).abs() < 2e-4, "{} vs {}", a.loss, b.loss);
                }
                let state: Vec<_> = compiled
                    .feedback
                    .iter()
                    .map(|(state, _)| state.clone())
                    .collect();
                compiled.compiled.as_ref().unwrap().export(&state).unwrap();
                let expected: Vec<_> = reference
                    .feedback
                    .iter()
                    .map(|(state, _)| state.clone())
                    .collect();
                for (index, (a, b)) in expected.iter().zip(&state).enumerate() {
                    let a = a.to_bytes_async().await.unwrap();
                    let b = b.to_bytes_async().await.unwrap();
                    for (a, b) in a.as_chunks::<4>().0.iter().zip(b.as_chunks::<4>().0.iter()) {
                        let a = f32::from_le_bytes(*a);
                        let b = f32::from_le_bytes(*b);
                        assert!(
                            (a - b).abs() <= 2e-5 + 1e-3 * a.abs(),
                            "state {index}: {a} vs {b}"
                        );
                    }
                }
                let a = reference.evaluate(&corpus, 1).await.unwrap();
                let b = compiled.evaluate(&corpus, 1).await.unwrap();
                assert!((a.loss - b.loss).abs() < 2e-4);
                assert!((a.accuracy - b.accuracy).abs() < 1e-4);
                let a = reference.next_char(&[0, 1, 2, 3], 1.).await.unwrap();
                let b = compiled.next_char(&[0, 1, 2, 3], 1.).await.unwrap();
                for (a, b) in a.iter().zip(b) {
                    assert!((a - b).abs() < 2e-4);
                }
                let probe =
                    corpus.encode_all("Once upon a time, there was a little girl named Lily.");
                let lens = compiled.attention(&probe).await.unwrap();
                assert_eq!(lens.context, config.context);
                assert_eq!(lens.filled, config.context.min(probe.len()));
                assert_eq!(lens.maps.len(), config.blocks);
                assert!(lens.disagreement < 1e-2, "{}", lens.disagreement);
                for heads in &lens.maps {
                    assert_eq!(heads.len(), config.heads);
                    for map in heads {
                        assert_eq!(map.len(), config.context * config.context);
                        for q in 0..lens.filled {
                            let row = &map[q * config.context..(q + 1) * config.context];
                            assert!((row.iter().sum::<f32>() - 1.).abs() < 1e-4);
                            assert!(row[q + 1..].iter().all(|p| p.abs() < 1e-6));
                        }
                    }
                }
                // Four tokens, each at least one character.
                assert!(
                    compiled
                        .generate(&corpus, &probe, 4, 0.8)
                        .await
                        .unwrap()
                        .chars()
                        .count()
                        >= 4
                );
                assert_eq!(
                    compiled.embedding_similarity().await.unwrap().len(),
                    corpus.vocab_size().pow(2)
                );
                if config == ModelConfig::TINY {
                    let trained = compiled.train(&corpus, 376).await.unwrap();
                    assert!(
                        trained.loss.is_finite() && trained.loss < 2.5,
                        "loss after 400 steps: {}",
                        trained.loss
                    );
                }
            }
        });
    }

    #[test]
    fn training_improves_held_out_loss_and_preserves_causal_attention() {
        let corpus = pollster::block_on(Corpus::benchmark()).unwrap();
        let mut model =
            pollster::block_on(Lm::new(corpus.vocab_size(), 0x51ed_c0de, ModelConfig::TINY))
                .expect("create training model");
        let uniform = (corpus.vocab_size() as f32).ln();

        let before = pollster::block_on(model.evaluate(&corpus, 2)).expect("evaluate");
        eprintln!("before: loss {:.5} acc {:.4}", before.loss, before.accuracy);
        assert!(
            (before.loss - uniform).abs() < 0.2,
            "an untrained model should be near {uniform:.2} nats, not {:.2}",
            before.loss,
        );
        let chance = 1.0 / corpus.vocab_size() as f32;
        assert!(
            before.accuracy > chance * 0.2 && before.accuracy < chance * 6.0,
            "untrained accuracy should be near chance ({:.1}%), not {:.1}%",
            chance * 100.0,
            before.accuracy * 100.0,
        );

        pollster::block_on(model.train(&corpus, 400)).expect("train");
        let after = pollster::block_on(model.evaluate(&corpus, 4)).expect("evaluate");
        eprintln!("after: loss {:.5} acc {:.4}", after.loss, after.accuracy);
        assert!(
            after.loss < uniform - 1.5,
            "400 steps should get held-out loss under {:.2} nats, not {:.2}",
            uniform - 1.5,
            after.loss,
        );
        assert!(
            after.accuracy > 0.3,
            "400 steps should get next-character top-1 over 30%, not {:.0}%",
            after.accuracy * 100.0,
        );

        // The interpretability lens is a second implementation of attention.
        let prompt = corpus.encode_all("Once upon a time, there was a little girl named");
        let read = pollster::block_on(model.attention(&prompt)).expect("attention");
        assert!(
            read.disagreement < 1e-2,
            "the lens disagrees with the model by {:.4} logits",
            read.disagreement,
        );

        // Causality: no query may read a key after it.
        for (block, heads) in read.maps.iter().enumerate() {
            for (head, map) in heads.iter().enumerate() {
                for query in 0..read.filled {
                    let leak: f32 = ((query + 1)..model.config.context)
                        .map(|k| map[query * model.config.context + k])
                        .sum();
                    assert!(
                        leak < 1e-5,
                        "block {block} head {head} query {query} reads {leak:.6} of its mass \
                         from the future",
                    );
                }
            }
        }

        // And the sampler produces text drawn from the corpus's alphabet.
        let written = pollster::block_on(model.generate(&corpus, &prompt, 64, 0.8)).expect("write");
        assert_eq!(written.chars().count(), 64);
        let known: String = corpus
            .encode_all(&written)
            .iter()
            .map(|id| corpus.decode(*id as usize))
            .collect();
        assert_eq!(known, written);
    }
}

#[cfg(target_arch = "wasm32")]
#[path = "training_checks.rs"]
mod training_checks;
