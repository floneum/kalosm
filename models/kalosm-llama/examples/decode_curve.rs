#![recursion_limit = "256"]

//! Decode throughput as a function of context length.
//!
//! Prints one line per 128-token window so a slowdown that only appears once
//! the KV cache is long shows up as a curve rather than one average.

use kalosm_llama::*;
use kalosm_model_types::ModelLoadingProgress;
use prelude::{StreamExt, TextCompletionModelExt};

fn main() {
    let _ = tracing_subscriber::fmt::try_init();
    pollster::block_on(async {
        let total: usize = std::env::var("DECODE_CURVE_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024);
        let window: usize = 128;

        let model = Llama::builder()
            .with_source(LlamaSource::qwen_3_0_6b_instruct())
            .build_with_loading_handler(|_: ModelLoadingProgress| {})
            .await
            .unwrap();

        let prompt = "Write a long, detailed essay about the history of computing:";
        let mut stream = model.complete(prompt).take(total);
        let mut tokens = 0usize;
        let mut mark = std::time::Instant::now();
        while stream.next().await.is_some() {
            tokens += 1;
            if tokens.is_multiple_of(window) {
                let dt = mark.elapsed();
                println!(
                    "tokens {}-{}: {:.2} tok/s",
                    tokens - window,
                    tokens,
                    window as f64 / dt.as_secs_f64()
                );
                mark = std::time::Instant::now();
            }
        }
    });
}
