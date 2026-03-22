use std::{env, fmt::Display, path::PathBuf, str::FromStr};

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeConfig {
    pub epochs: usize,
    pub warmup_steps: usize,
    pub learning_rate: f32,
    pub min_learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub adam_eps: f32,
    pub weight_decay: f32,
    pub log_every: usize,
    pub eval_batches: usize,
    pub sample_tokens: usize,
    pub sample_prefix_tokens: usize,
    pub sample_temperature: f32,
    pub sample_top_k: usize,
    pub block_size: usize,
    pub batch_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_ff: usize,
    pub n_layer: usize,
    pub conv_kernel_size: usize,
    pub attention_period: usize,
    pub use_rope: bool,
    pub rope_theta: f32,
    pub use_canvas_state_embeddings: bool,
    pub use_extra_norms: bool,
    pub eps: f32,
    pub init_scale: f32,
    pub seed: u64,
    pub save_every_steps: usize,
    pub save_final_model: bool,
    pub save_quantization: SaveQuantization,
    pub train_examples: usize,
    pub validation_examples: usize,
    pub test_examples: usize,
    pub dataset_path: Option<PathBuf>,
    pub include_synthetic_data: bool,
    pub synthetic_multiplier: usize,
    pub gguf_path: PathBuf,
    pub sample_output_path: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaveQuantization {
    F32,
    F16,
    Q4_0,
    Q8_0,
}

impl SaveQuantization {
    pub fn as_str(self) -> &'static str {
        match self {
            SaveQuantization::F32 => "f32",
            SaveQuantization::F16 => "f16",
            SaveQuantization::Q4_0 => "q4_0",
            SaveQuantization::Q8_0 => "q8_0",
        }
    }
}

impl FromStr for SaveQuantization {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" => Ok(Self::F32),
            "f16" => Ok(Self::F16),
            "q4_0" | "q40" | "q4" => Ok(Self::Q4_0),
            "q8_0" | "q80" | "q8" => Ok(Self::Q8_0),
            other => Err(format!(
                "unsupported save quantization {other:?}; expected one of: f32, f16, q4_0, q8_0"
            )),
        }
    }
}

impl RuntimeConfig {
    pub fn load() -> Self {
        Self::try_load().unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn try_load() -> Result<Self, String> {
        let env_path = env_path();
        dotenvy::from_path(&env_path).map_err(|error| {
            format!(
                "failed to load nanochat config from {}: {error}",
                env_path.display()
            )
        })?;

        let batch_size = try_read_env("NANOCHAT_BATCH_SIZE", 12)?;
        let n_head = try_read_env("NANOCHAT_N_HEAD", 4)?;
        let n_kv_head = try_read_env_optional("NANOCHAT_N_KV_HEAD")?.unwrap_or(n_head);

        Ok(Self {
            epochs: try_read_env("NANOCHAT_EPOCHS", 1)?,
            warmup_steps: try_read_env("NANOCHAT_WARMUP_STEPS", 20)?,
            learning_rate: try_read_env("NANOCHAT_LEARNING_RATE", 1e-3)?,
            min_learning_rate: try_read_env("NANOCHAT_MIN_LEARNING_RATE", 1e-4)?,
            beta1: try_read_env("NANOCHAT_BETA1", 0.9)?,
            beta2: try_read_env("NANOCHAT_BETA2", 0.95)?,
            adam_eps: try_read_env("NANOCHAT_ADAM_EPS", 1e-8)?,
            weight_decay: try_read_env("NANOCHAT_WEIGHT_DECAY", 0.1)?,
            log_every: try_read_env("NANOCHAT_LOG_EVERY", 80)?,
            eval_batches: try_read_env("NANOCHAT_EVAL_BATCHES", 2)?,
            sample_tokens: try_read_env("NANOCHAT_SAMPLE_TOKENS", 256)?,
            sample_prefix_tokens: try_read_env("NANOCHAT_SAMPLE_PREFIX_TOKENS", 128)?,
            sample_temperature: try_read_env("NANOCHAT_SAMPLE_TEMPERATURE", 0.7)?,
            sample_top_k: try_read_env("NANOCHAT_SAMPLE_TOP_K", 8)?,
            block_size: try_read_env("NANOCHAT_BLOCK_SIZE", 32)?,
            batch_size,
            n_embd: try_read_env("NANOCHAT_N_EMBD", 64)?,
            n_head,
            n_kv_head,
            n_ff: try_read_env("NANOCHAT_N_FF", 256)?,
            n_layer: try_read_env("NANOCHAT_N_LAYER", 4)?,
            conv_kernel_size: try_read_env("NANOCHAT_CONV_KERNEL_SIZE", 3)?,
            attention_period: try_read_env("NANOCHAT_ATTENTION_PERIOD", 1)?,
            use_rope: try_read_env("NANOCHAT_USE_ROPE", false)?,
            rope_theta: try_read_env("NANOCHAT_ROPE_THETA", 10_000.0)?,
            use_canvas_state_embeddings: try_read_env(
                "NANOCHAT_USE_CANVAS_STATE_EMBEDDINGS",
                true,
            )?,
            use_extra_norms: try_read_env("NANOCHAT_USE_EXTRA_NORMS", false)?,
            eps: try_read_env("NANOCHAT_EPS", 1e-5)?,
            init_scale: try_read_env("NANOCHAT_INIT_SCALE", 0.08)?,
            seed: try_read_env("NANOCHAT_SEED", 1337)?,
            save_every_steps: try_read_env("NANOCHAT_SAVE_EVERY_STEPS", 0)?,
            save_final_model: try_read_env("NANOCHAT_SAVE_FINAL_MODEL", true)?,
            save_quantization: try_read_env("NANOCHAT_SAVE_QUANTIZATION", SaveQuantization::F32)?,
            train_examples: try_read_env("NANOCHAT_TRAIN_EXAMPLES", 64)?,
            validation_examples: try_read_env("NANOCHAT_VALID_EXAMPLES", 16)?,
            test_examples: try_read_env("NANOCHAT_TEST_EXAMPLES", 16)?,
            dataset_path: try_read_env_path_optional("NANOCHAT_DATASET_PATH")?,
            include_synthetic_data: try_read_env("NANOCHAT_INCLUDE_SYNTHETIC_DATA", false)?,
            synthetic_multiplier: try_read_env("NANOCHAT_SYNTHETIC_MULTIPLIER", 1)?,
            gguf_path: try_read_env_path("NANOCHAT_GGUF_PATH", "nanochat.gguf")?,
            sample_output_path: try_read_env_path(
                "NANOCHAT_SAMPLE_OUTPUT_PATH",
                "nanochat-sample.svg",
            )?,
        })
    }
}

fn env_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env")
}

fn try_read_env<T>(key: &str, default: T) -> Result<T, String>
where
    T: FromStr + Copy,
    T::Err: Display,
{
    match env::var(key) {
        Ok(value) => value
            .parse()
            .map_err(|error| format!("invalid value for {key}: {value:?} ({error})")),
        Err(env::VarError::NotPresent) => Ok(default),
        Err(error) => Err(format!("failed to read {key}: {error}")),
    }
}

fn try_read_env_optional<T>(key: &str) -> Result<Option<T>, String>
where
    T: FromStr,
    T::Err: Display,
{
    match env::var(key) {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => value
            .parse()
            .map(Some)
            .map_err(|error| format!("invalid value for {key}: {value:?} ({error})")),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("failed to read {key}: {error}")),
    }
}

fn try_read_env_path(key: &str, default: &str) -> Result<PathBuf, String> {
    match env::var(key) {
        Ok(value) => Ok(PathBuf::from(value)),
        Err(env::VarError::NotPresent) => Ok(PathBuf::from(default)),
        Err(error) => Err(format!("failed to read {key}: {error}")),
    }
}

fn try_read_env_path_optional(key: &str) -> Result<Option<PathBuf>, String> {
    match env::var(key) {
        Ok(value) if value.trim().is_empty() => Ok(None),
        Ok(value) => Ok(Some(PathBuf::from(value))),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("failed to read {key}: {error}")),
    }
}
