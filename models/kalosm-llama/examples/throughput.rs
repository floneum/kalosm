//! Decode throughput probe against the locally-cached Llama 3.1 8B Q4_K_M.
//! Warms up, then measures steady-state decode tok/s. Release builds only.

use kalosm_llama::*;
use kalosm_model_types::ModelLoadingProgress;
use prelude::{StreamExt, TextCompletionModelExt};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() {
    pollster::block_on(async {
        let warmup = env_usize("WARMUP", 8);
        let measured = env_usize("TOKENS", 128);
        let prompt = "Write a detailed explanation of how a CPU executes instructions:";

        let load = std::time::Instant::now();
        let model = Llama::builder()
            .with_source(LlamaSource::llama_3_1_8b_chat())
            .build_with_loading_handler(|_: ModelLoadingProgress| {})
            .await
            .unwrap();
        println!("load: {:.2}s", load.elapsed().as_secs_f64());

        let mut stream = model.complete(prompt).take(warmup + measured);
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
        let tps = tokens as f64 / elapsed.as_secs_f64();
        let per_token_ms = elapsed.as_secs_f64() * 1_000.0 / tokens.max(1) as f64;
        println!(
            "decode: {tokens} tok in {:.3}s  =>  {tps:.2} tok/s  ({per_token_ms:.3} ms/tok)",
            elapsed.as_secs_f64()
        );
    });
}
