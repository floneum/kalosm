#![recursion_limit = "256"]

use kalosm_llama::prelude::*;
use kalosm_model_types::ModelLoadingProgress;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, Layer};

const TOKEN_TARGET: &str = "kalosm_llama::decode_token";

#[derive(Clone, Default)]
struct TokenObserver(Arc<Mutex<Option<TokenSamples>>>);

struct TokenSamples {
    events: Vec<(u32, Instant)>,
    expected: usize,
    invalid: bool,
}

#[derive(Default)]
struct TokenFields {
    token_id: Option<u32>,
    ordinal: Option<usize>,
}

impl tracing::field::Visit for TokenFields {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        match field.name() {
            "token_id" => self.token_id = u32::try_from(value).ok(),
            "ordinal" => self.ordinal = usize::try_from(value).ok(),
            _ => {}
        }
    }

    fn record_debug(&mut self, _: &tracing::field::Field, _: &dyn std::fmt::Debug) {}
}

impl<S: tracing::Subscriber> Layer<S> for TokenObserver {
    fn on_event(&self, event: &tracing::Event<'_>, _: tracing_subscriber::layer::Context<'_, S>) {
        if event.metadata().target() != TOKEN_TARGET {
            return;
        }
        let now = Instant::now();
        let mut fields = TokenFields::default();
        event.record(&mut fields);
        if let Some(samples) = self.0.lock().unwrap().as_mut() {
            match (fields.token_id, fields.ordinal) {
                (Some(id), Some(ordinal))
                    if ordinal == samples.events.len() + 1
                        && samples.events.len() < samples.expected =>
                {
                    samples.events.push((id, now));
                }
                _ => samples.invalid = true,
            }
        }
    }
}

impl TokenSamples {
    fn measured_gaps_ms(&self, warmup: usize) -> Vec<f64> {
        assert!(warmup > 0 && warmup < self.expected);
        assert!(!self.invalid, "invalid or out-of-order token events");
        assert_eq!(self.events.len(), self.expected, "incomplete token sample");
        self.events[warmup - 1..]
            .windows(2)
            .map(|pair| pair[1].1.duration_since(pair[0].1).as_secs_f64() * 1_000.)
            .collect()
    }
}

async fn measure_stream<S>(mut stream: S, observer: &TokenObserver, warmup: usize, measured: usize)
where
    S: futures_util::Stream<Item = String> + Unpin,
{
    *observer.0.lock().unwrap() = Some(TokenSamples {
        events: Vec::with_capacity(warmup + measured),
        expected: warmup + measured,
        invalid: false,
    });
    // Drain the normal chat/completion path through its exact token limit.
    // Text callbacks can combine tokens; only sampled-token events define time.
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        text.push_str(&chunk);
    }
    let samples = observer.0.lock().unwrap().take().unwrap();
    let gaps_ms = samples.measured_gaps_ms(warmup);
    let token_ids: Vec<_> = samples.events.iter().map(|&(id, _)| id).collect();
    let per_token_ms = gaps_ms.iter().sum::<f64>() / measured as f64;
    let tokens_per_second = 1_000. / per_token_ms;
    let mut sorted = gaps_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let p50_ms = sorted[(measured - 1) / 2];
    let p95_ms = sorted[(measured * 95).div_ceil(100) - 1];
    let max_ms = sorted[measured - 1];
    println!(
        "llama_forward_profile tokens={measured} requested={measured} warmup={warmup} per_token_ms={per_token_ms:.3} tokens_per_second={tokens_per_second:.3} p50_ms={p50_ms:.3} p95_ms={p95_ms:.3} max_ms={max_ms:.3}"
    );
    println!("llama_forward_gaps_ms {gaps_ms:?}");
    println!("llama_forward_token_ids {token_ids:?}");
    println!("llama_forward_output {text:?}");
    if let Ok(max_ms) = std::env::var("KALOSM_PROFILE_LLAMA_MAX_MS") {
        let max_ms = max_ms.parse::<f64>().expect("MAX_MS must be a number");
        assert!(
            per_token_ms <= max_ms,
            "decode regression: {per_token_ms:.3} ms/token exceeds {max_ms:.3} ms/token"
        );
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn source() -> LlamaSource {
    if let Ok(path) = std::env::var("KALOSM_PROFILE_LLAMA_LOCAL_PATH") {
        return LlamaSource::new(FileSource::local(path.into()));
    }

    if let (Ok(model_id), Ok(file)) = (
        std::env::var("KALOSM_PROFILE_LLAMA_HF_REPO"),
        std::env::var("KALOSM_PROFILE_LLAMA_HF_FILE"),
    ) {
        let revision =
            std::env::var("KALOSM_PROFILE_LLAMA_HF_REVISION").unwrap_or_else(|_| "main".into());
        return LlamaSource::new(FileSource::huggingface(model_id, revision, file));
    }

    match std::env::var("KALOSM_PROFILE_LLAMA_SOURCE").as_deref() {
        Ok("default-chat") => LlamaSource::llama_3_1_8b_chat(),
        Ok("llama-8b") => LlamaSource::llama_8b(),
        Ok("llama-8b-chat") => LlamaSource::llama_8b_chat(),
        Ok("llama-3.1-8b-chat") => LlamaSource::llama_3_1_8b_chat(),
        Ok("tiny-llama") => LlamaSource::tiny_llama_1_1b_chat(),
        Ok("qwen2.5-0.5b") => LlamaSource::qwen_2_5_0_5b_instruct(),
        Ok("qwen2.5-1.5b") => LlamaSource::qwen_2_5_1_5b_instruct(),
        Ok("qwen3-0.6b") => LlamaSource::qwen_3_0_6b_instruct(),
        Ok("gemma3-270m") => LlamaSource::gemma_3_270m_chat(),
        Ok("gemma3-1b") => LlamaSource::gemma_3_1b_chat(),
        Ok("gemma3-4b") => LlamaSource::gemma_3_4b_chat(),
        _ => LlamaSource::new(FileSource::huggingface(
            "unsloth/SmolLM2-135M-Instruct-GGUF",
            "main",
            "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        )),
    }
}

fn main() {
    let observer = TokenObserver::default();
    tracing_subscriber::registry()
        .with(
            observer
                .clone()
                .with_filter(tracing_subscriber::filter::filter_fn(|metadata| {
                    metadata.target() == TOKEN_TARGET
                })),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_filter(tracing_subscriber::filter::LevelFilter::WARN),
        )
        .init();

    pollster::block_on(async {
        let warmup = env_usize("KALOSM_PROFILE_LLAMA_WARMUP", 4);
        let measured = env_usize("KALOSM_PROFILE_LLAMA_TOKENS", 16);
        let seed = env_usize("KALOSM_PROFILE_LLAMA_SEED", 0) as u64;
        assert!((1..=512).contains(&warmup) && (1..=512).contains(&measured));
        let prompt = std::env::var("KALOSM_PROFILE_LLAMA_PROMPT")
            .unwrap_or_else(|_| "Write one compact Rust performance tip:".into());

        let model_source = source();
        let source_description = format!("{model_source:?}");
        let model = Llama::builder()
            .with_source(model_source)
            .build_with_loading_handler(|_: ModelLoadingProgress| {})
            .await
            .unwrap();

        let prerun = env_usize("KALOSM_PROFILE_LLAMA_PRERUN", 0);
        assert!(prerun <= 512);
        if prerun > 0 {
            let prerun_prompt = std::env::var("KALOSM_PROFILE_LLAMA_PRERUN_PROMPT")
                .unwrap_or_else(|_| "Warm up the model with a short answer.".into());
            let prerun_sampler = GenerationParameters::default()
                .with_max_length((prerun + 1) as u32)
                .with_seed(seed);
            let prerun_stream = model.complete(&prerun_prompt).with_sampler(prerun_sampler);
            measure_stream(prerun_stream, &observer, 1, prerun).await;
        }

        let sampler = GenerationParameters::default().with_max_length((warmup + measured) as u32);
        let sampler = match std::env::var("KALOSM_PROFILE_LLAMA_TOP_K")
            .ok()
            .and_then(|value| value.parse().ok())
        {
            Some(top_k) => sampler.with_top_k(top_k),
            None => sampler,
        };
        let repeats = env_usize("KALOSM_PROFILE_LLAMA_REPEATS", 1);
        assert!((1..=10).contains(&repeats));
        println!(
            "llama_forward_config warmup={warmup} measured={measured} seed={seed} repeats={repeats} chat={} prompt={prompt:?} source={source_description} sampler={sampler:?}",
            std::env::var_os("KALOSM_PROFILE_LLAMA_CHAT").is_some()
        );
        for _ in 0..repeats {
            if std::env::var_os("KALOSM_PROFILE_LLAMA_CHAT").is_some() {
                let mut chat = model.chat();
                let stream = chat
                    .add_message(prompt.clone())
                    .with_sampler(sampler.clone().with_seed(seed));
                measure_stream(stream, &observer, warmup, measured).await;
            } else {
                let stream = model
                    .complete(&prompt)
                    .with_sampler(sampler.clone().with_seed(seed));
                measure_stream(stream, &observer, warmup, measured).await;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn token_intervals_exclude_prefill_and_keep_stalls() {
        let start = Instant::now();
        let samples = TokenSamples {
            events: [1000, 2000, 2010, 2020, 2100]
                .into_iter()
                .enumerate()
                .map(|(id, ms)| (id as u32, start + Duration::from_millis(ms)))
                .collect(),
            expected: 5,
            invalid: false,
        };
        assert_eq!(samples.measured_gaps_ms(2), [10., 10., 80.]);
    }

    #[test]
    fn trace_observer_counts_tokens_independently_of_text_callbacks() {
        let observer = TokenObserver::default();
        let subscriber = tracing_subscriber::registry().with(observer.clone());
        tracing::subscriber::with_default(subscriber, || {
            // One UTF-8 callback could contain both tokens; neither text decoding
            // nor the callback count participates in this measurement.
            *observer.0.lock().unwrap() = Some(TokenSamples {
                events: Vec::with_capacity(2),
                expected: 2,
                invalid: false,
            });
            tracing::trace!(target: TOKEN_TARGET, token_id = 12_u32, ordinal = 1_u32);
            tracing::trace!(target: "unrelated", token_id = 999_u32, ordinal = 1_u32);
            tracing::trace!(target: TOKEN_TARGET, token_id = 34_u32, ordinal = 2_u32);
        });
        let samples = observer.0.lock().unwrap().take().unwrap();
        assert!(!samples.invalid);
        assert_eq!(
            samples.events.iter().map(|&(id, _)| id).collect::<Vec<_>>(),
            [12, 34]
        );
    }

    #[test]
    fn trace_observer_rejects_overflow_and_wrong_sequence() {
        for ordinal in [1_u32, 2, 3] {
            let observer = TokenObserver::default();
            *observer.0.lock().unwrap() = Some(TokenSamples {
                events: Vec::with_capacity(1),
                expected: 1,
                invalid: false,
            });
            let subscriber = tracing_subscriber::registry().with(observer.clone());
            tracing::subscriber::with_default(subscriber, || {
                tracing::trace!(target: TOKEN_TARGET, token_id = 12_u32, ordinal = 1_u32);
                tracing::trace!(target: TOKEN_TARGET, token_id = 34_u32, ordinal);
            });
            let samples = observer.0.lock().unwrap().take().unwrap();
            assert!(samples.invalid);
            assert_eq!(samples.events.len(), 1);
        }
    }
}
