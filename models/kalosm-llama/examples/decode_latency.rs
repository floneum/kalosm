use kalosm_llama::*;
use kalosm_model_types::ModelLoadingProgress;
use prelude::{StreamExt, TextCompletionModelExt};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn main() {
    pollster::block_on(async {
        let warmup = env_usize("KALOSM_DECODE_BENCH_WARMUP", 8);
        let measured = env_usize("KALOSM_DECODE_BENCH_TOKENS", 64);
        let prompt = std::env::var("KALOSM_DECODE_BENCH_PROMPT")
            .unwrap_or_else(|_| "Write a compact list of practical Rust performance tips:".into());

        let model = Llama::builder()
            .with_source(LlamaSource::llama_8b())
            .build_with_loading_handler(|_: ModelLoadingProgress| {})
            .await
            .unwrap();

        let mut stream = model.complete(&prompt).take(warmup + measured);
        for _ in 0..warmup {
            if stream.next().await.is_none() {
                eprintln!("stream ended during warmup");
                return;
            }
        }

        let start = std::time::Instant::now();
        let mut tokens = 0usize;
        while tokens < measured {
            if stream.next().await.is_none() {
                break;
            }
            tokens += 1;
        }
        let elapsed = start.elapsed();
        let per_token_ms = elapsed.as_secs_f64() * 1_000.0 / tokens.max(1) as f64;
        println!(
            "decode_latency tokens={tokens} elapsed={elapsed:?} per_token_ms={per_token_ms:.3}"
        );
    });
}
