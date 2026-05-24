#![recursion_limit = "256"]

use fusor::Device;
use kalosm_llama::prelude::*;
use kalosm_model_types::ModelLoadingProgress;
use std::time::Instant;

const PROBE_TOKENS: usize = 12;
const WARMUP_TOKENS: usize = 8;
const DEFAULT_GPU_SAMPLE_TOP_K: &str = "1";

fn main() {
    pollster::block_on(async {
        fn progress(_: ModelLoadingProgress) {}

        if std::env::var_os("KALOSM_LLAMA_GPU_SAMPLE_TOP_K").is_none() {
            std::env::set_var("KALOSM_LLAMA_GPU_SAMPLE_TOP_K", DEFAULT_GPU_SAMPLE_TOP_K);
        }

        let model = Llama::builder()
            .with_source(LlamaSource::llama_3_1_8b_chat())
            .with_device(Device::gpu().await.unwrap())
            .build_with_loading_handler(progress)
            .await
            .unwrap();

        run_probe(&model, "warmup", WARMUP_TOKENS, false).await;
        run_probe(&model, "steady", PROBE_TOKENS, true).await;
    });
}

async fn run_probe(model: &Llama, label: &str, tokens_to_generate: usize, print_tokens: bool) {
    let sampler = GenerationParameters::new()
        .with_max_length(tokens_to_generate as u32)
        .with_seed(0);
    let mut stream = model.complete("Hello world").with_sampler(sampler);
    let total_start = Instant::now();
    let mut last = total_start;
    let mut first_token_at = None;
    let mut last_token_at = None;
    let mut tokens = 0usize;

    while let Some(token) = stream.next().await {
        let now = Instant::now();
        tokens += 1;
        first_token_at.get_or_insert(now);
        last_token_at = Some(now);
        if print_tokens {
            eprintln!(
                "{label} token {tokens:02}: step={:?} total={:?} text={token:?}",
                now.duration_since(last),
                now.duration_since(total_start)
            );
        }
        last = now;
    }

    if let (Some(first), Some(last)) = (first_token_at, last_token_at) {
        let elapsed = last.duration_since(first);
        let measured_tokens = tokens.saturating_sub(1);
        let tps = if measured_tokens > 0 && !elapsed.is_zero() {
            measured_tokens as f64 / elapsed.as_secs_f64()
        } else {
            0.0
        };
        eprintln!(
            "{label} decode: {measured_tokens} tokens in {:.3}s = {tps:.2} t/s",
            elapsed.as_secs_f64()
        );
    }
}
