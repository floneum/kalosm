//! Rookie's model: one Fusor graph trained on the GPU through `Session`, and
//! the same network compiled for the CPU search (see `cpu`).
use fusor::{Device, Dtype, Error, Result, Tensor, tensor::Dyn};

/// Positions per training and inference batch.
pub const BATCH: usize = 1024;
pub const INPUT: usize = 832;
pub mod cpu;
/// Wall-clock milliseconds; `std::time::Instant` panics in the browser.
pub mod clock {
    #[cfg(target_arch = "wasm32")]
    #[wasm_bindgen::prelude::wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = Date, js_name = now)]
        fn date_now() -> f64;
    }
    #[cfg(target_arch = "wasm32")]
    pub fn millis() -> f64 {
        date_now()
    }
    #[cfg(not(target_arch = "wasm32"))]
    pub fn millis() -> f64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64() * 1000.
    }
}
pub mod network;
pub use network::Config;

/// The value network with its AdamW state on the GPU. One graph serves both
/// training and prediction: a step sets the input leaves, resolves the loss
/// and every updated state, and each state leaf adopts the buffer its update
/// landed in.
pub struct Model {
    device: Device,
    parameters: Vec<Dyn>,
    /// Each parameter and its two moments, with their next values.
    feedback: Vec<(Dyn, Dyn)>,
    /// What a training step resolves: the loss and every next state.
    roots: Vec<Dyn>,
    x: Tensor<2, f32>,
    /// Value target interval per row: exact scores have low == high.
    low: Tensor<2, f32>,
    high: Tensor<2, f32>,
    alpha: Tensor<1, f32>,
    rate: Tensor<1, f32>,
    value: Tensor<2, f32>,
    loss: Tensor<1, f32>,
    pub step: u32,
    /// Peak AdamW learning rate, reached after warm-up.
    pub learning_rate: f32,
    pub batch: usize,
    pub config: Config,
}
fn shape(ok: bool) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error::Shape("invalid chess batch or checkpoint".into()))
    }
}
/// Scale of the piece-count inputs (782..794): a learned factorization sharing
/// one weight per piece type across its squares (ROOKIE_COUNT_SCALE natively).
pub fn count_scale() -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(v) = std::env::var("ROOKIE_COUNT_SCALE").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    0.25
}
/// Step-size multiplier of the linear value path (ROOKIE_LINEAR_STEP natively).
fn linear_step_scale() -> f32 {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(v) = std::env::var("ROOKIE_LINEAR_STEP").ok().and_then(|v| v.parse().ok()) {
        return v;
    }
    1.
}
impl Model {
    pub async fn new(config: Config) -> Result<Self> {
        config.validate()?;
        let batch = BATCH;
        let device = Device::gpu().await?;
        let x = Tensor::from_slice(&device, [batch, INPUT], &vec![0f32; batch * INPUT]);
        let low = Tensor::from_slice(&device, [batch, 1], &vec![0f32; batch]);
        let high = Tensor::from_slice(&device, [batch, 1], &vec![0f32; batch]);
        let alpha = Tensor::from_slice(&device, [1], &[0f32]);
        let rate = Tensor::from_slice(&device, [1], &[0f32]);
        let (parameters, raw_value) = network::build(&device, &x, config)?;
        // The model alone scores a position: no hand-written evaluation.
        let value = raw_value.tanh();
        // Bound-aware loss: only a value outside [low, high] is wrong.
        let loss = low
            .sub(&value)
            .relu()
            .sqr()
            .add(&value.sub(&high).relu().sqr())
            .sum::<1>(1usize)
            .sum::<0>(0usize)
            .div_scalar(batch as f32)
            .reshape([1]);
        let gradients = device.graph().backward_with(loss.as_dyn(), &parameters)?;
        let mut roots = vec![loss.as_dyn().clone()];
        let mut feedback = Vec::new();
        // The linear input-to-value path (the last parameter) is convex and learns
        // piece-square values from few games; it may take larger steps.
        let linear_step = linear_step_scale();
        for (index, parameter) in parameters.iter().enumerate() {
            let step = if index + 1 == parameters.len() { linear_step } else { 1. };
            let gradient = gradients
                .get(parameter)
                .ok_or_else(|| Error::Plan("missing chess gradient".into()))?
                .clamp(-1., 1.)?;
            let bytes = vec![0u8; parameter.elem_count().unwrap() as usize * 4];
            let handle = device.graph().handle();
            let m = Dyn::from_slice(handle, Dtype::F32, &parameter.shape(), &bytes)?;
            let v = Dyn::from_slice(handle, Dtype::F32, &parameter.shape(), &bytes)?;
            let next_m = m.mul_scalar(0.9)?.add(&gradient.mul_scalar(0.1)?)?;
            let next_v = v
                .mul_scalar(0.999)?
                .add(&gradient.sqr()?.mul_scalar(0.001)?)?;
            let update = next_m
                .mul_(alpha.as_dyn())?
                .div(&next_v.sqrt()?.add_scalar(1e-8)?)?
                .mul_scalar(step)?;
            let decay = parameter.mul_(rate.as_dyn())?.mul_scalar(0.01)?;
            let next_p = parameter.sub(&update)?.sub(&decay)?;
            for (old, next) in [parameter.clone(), m, v]
                .into_iter()
                .zip([next_p, next_m, next_v])
            {
                roots.push(next.clone());
                feedback.push((old, next));
            }
        }
        Ok(Self {
            device,
            parameters,
            feedback,
            roots,
            x,
            low,
            high,
            alpha,
            rate,
            value,
            loss,
            step: 0,
            // 3e-3 learned fastest from scratch (1e-3 slower, 1e-2 unstable).
            learning_rate: 0.003,
            batch,
            config,
        })
    }
    /// Forget every buffer computed from the input leaves, which just changed.
    fn invalidate(&self) {
        self.value.as_dyn().clear_device_buf();
        for root in &self.roots {
            root.clear_device_buf();
        }
    }
    /// The value of each position, for its side to move.
    pub async fn predict(&mut self, features: &[f32]) -> Result<Vec<f32>> {
        shape(features.len() == self.batch * INPUT)?;
        self.x.set_elements(features);
        self.invalidate();
        self.device.session().resolve(std::slice::from_ref(self.value.as_dyn()))?;
        self.value.to_vec_f32_async().await
    }
    /// Train `steps` batches toward `values` (one target per position).
    pub async fn train(&mut self, features: &[f32], values: &[f32], steps: u32) -> Result<()> {
        shape(features.len() == self.batch * INPUT && values.len() == self.batch && steps <= 32)?;
        self.x.set_elements(features);
        self.low.set_elements(values);
        self.high.set_elements(values);
        for _ in 0..steps {
            self.step += 1;
            // Automatic warm-up and inverse-square-root decay; no user tuning.
            let rate = self.learning_rate
                * (self.step as f32 / 32.).min(1.)
                * (2048. / self.step.max(2048) as f32).sqrt();
            let t = self.step.min(i32::MAX as u32) as i32;
            let alpha = rate * (1. - 0.999f32.powi(t)).sqrt() / (1. - 0.9f32.powi(t));
            self.alpha.set_elements(&[alpha]);
            self.rate.set_elements(&[rate]);
            self.invalidate();
            self.device.session().resolve(&self.roots)?;
            for (state, next) in &self.feedback {
                state.adopt_buffer(next)?;
            }
        }
        Ok(())
    }
    /// The parameters, each in `Config::shapes` order.
    pub async fn value_parameters(&self) -> Result<Vec<Vec<f32>>> {
        let mut parameters = Vec::new();
        for p in &self.parameters {
            parameters.push(p.to_vec_f32_async().await?);
        }
        Ok(parameters)
    }
    /// The network for the CPU search.
    pub async fn cpu_net(&self) -> Result<cpu::Net> {
        Ok(cpu::Net::from_parameters(self.config, &self.value_parameters().await?))
    }
    /// The parameters concatenated (for `cpu::Net::from_flat`).
    pub async fn value_weights(&self) -> Result<Vec<f32>> {
        Ok(self.value_parameters().await?.concat())
    }
    /// The last training step's loss.
    pub async fn losses(&self) -> Result<Vec<f32>> {
        self.loss.to_vec_f32_async().await
    }
    pub fn info(&self) -> String {
        format!(
            "Fusor · MLP {}×{} (hidden {}) · {} parameters · batches of {}",
            self.config.width,
            self.config.depth,
            self.config.hidden,
            self.config.parameter_count(),
            self.batch,
        )
    }
    /// The step count, then every parameter followed by its two moments.
    pub async fn save(&self) -> Result<Vec<f32>> {
        let mut out = vec![self.step as f32];
        for (state, _) in &self.feedback {
            out.extend(state.to_vec_f32_async().await?);
        }
        Ok(out)
    }
    pub fn load(&mut self, data: &[f32]) -> Result<()> {
        shape(
            data.len() == 1 + self.config.parameter_count() * 3
                && data.iter().all(|v| v.is_finite())
                && data[0] >= 0.
                && data[0].fract() == 0.,
        )?;
        let mut offset = 1;
        for (state, _) in &self.feedback {
            let end = offset + state.elem_count().unwrap() as usize;
            state.set_bytes(bytemuck::cast_slice(&data[offset..end]).to_vec())?;
            offset = end;
        }
        self.invalidate();
        self.step = data[0] as u32;
        Ok(())
    }
}

#[cfg(target_arch = "wasm32")]
mod browser {
    use super::*;
    use wasm_bindgen::prelude::*;
    fn js(error: Error) -> JsValue {
        JsValue::from_str(&error.to_string())
    }
    fn config(width: usize, depth: usize, hidden: usize) -> std::result::Result<Config, JsValue> {
        let config = Config { width, depth, hidden };
        config.validate().map_err(js)?;
        Ok(config)
    }
    /// CPU self-play for a worker: plays games with the weights it is given.
    #[wasm_bindgen]
    pub struct CpuEngine {
        net: cpu::Net,
        ready: bool,
        seed: u32,
        table: cpu::Table,
    }
    #[wasm_bindgen]
    impl CpuEngine {
        #[wasm_bindgen(constructor)]
        pub fn new(width: usize, depth: usize, hidden: usize, seed: u32) -> std::result::Result<CpuEngine, JsValue> {
            console_error_panic_hook::set_once();
            Ok(CpuEngine {
                net: cpu::Net::new(config(width, depth, hidden)?),
                ready: false,
                seed: seed | 1,
                table: cpu::Table::new(),
            })
        }
        pub fn set_weights(&mut self, flat: Vec<f32>) {
            self.net.set_flat(&flat);
            self.ready = true;
        }
        /// One game: 833 numbers per searched position (features, then the value
        /// target), followed by the final game state (768 words as f32 bits) and
        /// the result for White. Empty until weights are set.
        pub fn play_game(&mut self, nodes: u32, result_weight: f32, lambda: f32, random_plies: u32) -> Vec<f32> {
            if !self.ready {
                return Vec::new();
            }
            let mut seed = self.seed;
            let mut random = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                seed
            };
            let clock = clock::millis;
            let game = cpu::self_play_game(&self.net, u64::from(nodes), result_weight, lambda, random_plies, &mut random, &clock, &mut self.table);
            self.seed = random();
            let mut out = Vec::with_capacity(game.rows.len() * 833 + 769);
            for (x, target) in game.rows {
                out.extend(x);
                out.push(target);
            }
            out.extend(game.last.iter().map(|w| f32::from_bits(*w)));
            out.push(game.result as f32);
            out
        }
    }
    /// The model on the GPU, and the search that plays a human.
    #[wasm_bindgen]
    pub struct ChessGpu {
        model: Model,
        /// The network on the CPU, and the training step it matches.
        net: cpu::Net,
        net_step: Option<u32>,
        /// The search's transposition table, kept across the human game.
        table: Option<cpu::Table>,
    }
    #[wasm_bindgen]
    impl ChessGpu {
        pub async fn create(width: usize, depth: usize, hidden: usize) -> std::result::Result<ChessGpu, JsValue> {
            console_error_panic_hook::set_once();
            let config = config(width, depth, hidden)?;
            Ok(Self {
                model: Model::new(config).await.map_err(js)?,
                net: cpu::Net::new(config),
                net_step: None,
                table: None,
            })
        }
        /// Search one position (a 768-word game state) for at most `millis`
        /// and return the state after the chosen move.
        pub async fn play(&mut self, states: Vec<u32>, millis: f64) -> std::result::Result<Vec<u32>, JsValue> {
            if self.net_step != Some(self.model.step) {
                self.net.set_parameters(&self.model.value_parameters().await.map_err(js)?);
                self.net_step = Some(self.model.step);
            }
            let clock = clock::millis;
            let mut search = cpu::Search::with_table(&self.net, &states, &clock, self.table.take().unwrap_or_default());
            let found = search.think(millis);
            self.table = Some(search.into_table());
            let found = found.ok_or_else(|| js(Error::Shape("no legal move".into())))?;
            Ok(cpu::answer(&states, &found))
        }
        pub fn info(&self) -> String {
            self.model.info()
        }
        /// Rows of every training and prediction batch.
        pub fn batch(&self) -> usize {
            self.model.batch
        }
        pub async fn predict(&mut self, x: Vec<f32>) -> std::result::Result<Vec<f32>, JsValue> {
            self.model.predict(&x).await.map_err(js)
        }
        pub async fn train(&mut self, x: Vec<f32>, values: Vec<f32>, steps: u32) -> std::result::Result<(), JsValue> {
            self.model.train(&x, &values, steps).await.map_err(js)
        }
        pub async fn losses(&self) -> std::result::Result<Vec<f32>, JsValue> {
            self.model.losses().await.map_err(js)
        }
        /// The value weights, for the CPU self-play workers.
        pub async fn value_weights(&self) -> std::result::Result<Vec<f32>, JsValue> {
            self.model.value_weights().await.map_err(js)
        }
        pub async fn save(&self) -> std::result::Result<Vec<f32>, JsValue> {
            self.model.save().await.map_err(js)
        }
        pub fn load(&mut self, data: Vec<f32>) -> std::result::Result<(), JsValue> {
            self.model.load(&data).map_err(js)
        }
    }
}
