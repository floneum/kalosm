use kalosm_llama::prelude::*;
use kalosm_model_types::ModelLoadingProgress;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn source() -> LlamaSource {
    if let (Ok(model_id), Ok(file)) = (
        std::env::var("KALOSM_PROFILE_LLAMA_HF_REPO"),
        std::env::var("KALOSM_PROFILE_LLAMA_HF_FILE"),
    ) {
        let revision =
            std::env::var("KALOSM_PROFILE_LLAMA_HF_REVISION").unwrap_or_else(|_| "main".into());
        return LlamaSource::new(FileSource::huggingface(model_id, revision, file));
    }

    match std::env::var("KALOSM_PROFILE_LLAMA_SOURCE").as_deref() {
        Ok("llama-8b") => LlamaSource::llama_8b(),
        Ok("llama-8b-chat") => LlamaSource::llama_8b_chat(),
        Ok("llama-3.1-8b-chat") => LlamaSource::llama_3_1_8b_chat(),
        Ok("tiny-llama") => LlamaSource::tiny_llama_1_1b_chat(),
        _ => LlamaSource::new(FileSource::huggingface(
            "unsloth/SmolLM2-135M-Instruct-GGUF",
            "main",
            "SmolLM2-135M-Instruct-Q4_K_M.gguf",
        )),
    }
}

#[tokio::main]
async fn main() {
    let warmup = env_usize("KALOSM_PROFILE_LLAMA_WARMUP", 4);
    let measured = env_usize("KALOSM_PROFILE_LLAMA_TOKENS", 16);
    let prompt = std::env::var("KALOSM_PROFILE_LLAMA_PROMPT")
        .unwrap_or_else(|_| "Write one compact Rust performance tip:".into());

    let model = Llama::builder()
        .with_source(source())
        .build_with_loading_handler(|_: ModelLoadingProgress| {})
        .await
        .unwrap();

    let sampler = GenerationParameters::default().with_max_length((warmup + measured) as u32);
    let mut stream = model
        .complete(&prompt)
        .with_sampler(sampler)
        .take(warmup + measured);
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
        "llama_forward_profile tokens={tokens} elapsed={elapsed:?} per_token_ms={per_token_ms:.3}"
    );
}
