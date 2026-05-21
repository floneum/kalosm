//! Tiny non-interactive perf measurement matching the chat steady-state
//! tok/s metric. Used to bisect a 27 → 13 t/s regression.

use kalosm_llama::*;
use prelude::ChatModelExt;
use prelude::GenerationParameters;
use prelude::StreamExt;
use std::time::Instant;

#[tokio::main]
async fn main() {
    let model = Llama::new_chat().await.unwrap();
    let mut chat = model
        .chat()
        .with_system_prompt("The assistant will act like a pirate");

    // Warm: short prompt to trigger pipeline cache for the chat path.
    let warmup_sampler = GenerationParameters::new().with_max_length(4);
    let _ = chat(&"hi".to_string())
        .with_sampler(warmup_sampler)
        .collect::<Vec<_>>()
        .await;

    // Measured: same prompt as the user's reported run.
    let measured_tokens = std::env::var("KALOSM_PERF_PROBE_TOKENS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(64);
    let sampler = GenerationParameters::new().with_max_length(measured_tokens);
    let mut response = chat(&"write a short story".to_string()).with_sampler(sampler);
    let mut first: Option<Instant> = None;
    let mut last: Option<Instant> = None;
    let mut tokens: usize = 0;
    while response.next().await.is_some() {
        let now = Instant::now();
        first.get_or_insert(now);
        last = Some(now);
        tokens += 1;
    }
    if let (Some(f), Some(l)) = (first, last) {
        let elapsed = (l - f).as_secs_f64();
        let rate = if elapsed > 0.0 && tokens > 1 {
            (tokens - 1) as f64 / elapsed
        } else {
            0.0
        };
        println!("PERF_PROBE tokens={tokens} elapsed_s={elapsed:.3} tok_s={rate:.2}");
    } else {
        println!("PERF_PROBE no_tokens");
    }
}
