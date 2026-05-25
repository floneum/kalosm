//! Tiny non-interactive perf measurement matching the chat steady-state
//! tok/s metric. Used to bisect a 27 → 13 t/s regression.

use futures_util::Stream;
use kalosm_llama::*;
use prelude::ChatModelExt;
use prelude::GenerationParameters;
use prelude::StreamExt;
use std::time::Instant;

async fn measure_response<S>(mut response: S) -> (usize, Option<Instant>, Option<Instant>)
where
    S: Stream + Unpin,
{
    let mut first: Option<Instant> = None;
    let mut last: Option<Instant> = None;
    let mut tokens: usize = 0;
    while response.next().await.is_some() {
        let now = Instant::now();
        first.get_or_insert(now);
        last = Some(now);
        tokens += 1;
    }
    (tokens, first, last)
}

fn main() {
    pollster::block_on(async {
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
        let unbounded = std::env::var_os("KALOSM_PERF_PROBE_UNBOUNDED").is_some();
        let sampler = if unbounded {
            GenerationParameters::new()
        } else {
            GenerationParameters::new().with_max_length(measured_tokens)
        };
        let mut prompt = std::env::var("KALOSM_PERF_PROBE_PROMPT")
            .unwrap_or_else(|_| "write a short story".to_string());
        if let Some(words) = std::env::var("KALOSM_PERF_PROBE_CONTEXT_WORDS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
        {
            let prefix = "context ".repeat(words);
            prompt = format!(
                "{prefix}\nContinue with a numbered list of short software performance notes."
            );
        }
        let response = chat(&prompt).with_sampler(sampler);
        let (tokens, first, last) = if unbounded {
            measure_response(response).await
        } else {
            measure_response(response.take(measured_tokens as usize)).await
        };
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
    });
}
