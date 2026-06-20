use kalosm::language::*;
use std::time::Instant;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let t_load_start = Instant::now();
    let mut builder = Llama::builder().with_source(
        LlamaSource::gemma_4_e2b_it_qat_chat().with_vision_model(FileSource::HuggingFace {
            model_id: "unsloth/gemma-4-E2B-it-qat-GGUF".into(),
            revision: "main".into(),
            file: "mmproj-F16.gguf".into(),
        }),
    );
    builder = if std::env::var_os("KALOSM_VISION_CPU").is_some() {
        builder.with_device(Device::Cpu)
    } else {
        builder.with_device(Device::gpu().await.expect(
            "The vision example requires a GPU by default; set KALOSM_VISION_CPU=1 to run the slow CPU path.",
        ))
    };
    let model = builder.build().await.unwrap();
    tracing::info!("[timing] model load: {:.2?}", t_load_start.elapsed());

    let mut chat = model.chat();
    let max_tokens = std::env::var("KALOSM_VISION_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(64);
    let t_total = Instant::now();
    let image_source = if let Ok(url) = std::env::var("KALOSM_VISION_URL") {
        MediaSource::url(url)
    } else if let Ok(path) = std::env::var("KALOSM_VISION_IMAGE") {
        MediaSource::file(path).unwrap()
    } else {
        MediaSource::bytes(include_bytes!("landscape.jpg").as_slice())
    };
    let mut sampler = GenerationParameters::new()
        .with_standard_sampler()
        .with_temperature(0.0);
    if let Some(seed) = std::env::var("KALOSM_VISION_SEED")
        .ok()
        .and_then(|seed| seed.parse::<u64>().ok())
    {
        sampler = sampler.with_seed(seed);
    }
    let mut response = chat(&(
        MediaChunk::new(image_source, MediaType::Image),
        "Describe this image.",
    ))
    .with_sampler(sampler);
    let mut first_token_at: Option<std::time::Duration> = None;
    let mut token_count = 0u64;
    let t_prefill = Instant::now();
    while let Some(token) = response.next().await {
        if first_token_at.is_none() {
            first_token_at = Some(t_prefill.elapsed());
            tracing::info!(
                "[timing] first token (prefill): {:.2?}",
                first_token_at.unwrap()
            );
        }
        token_count += 1;
        print!("{}", token);
        if token_count >= max_tokens {
            break;
        }
    }
    println!();
    let total = t_total.elapsed();
    let prefill = first_token_at.unwrap_or_default();
    let decode = total.saturating_sub(prefill);
    let decode_tokens = token_count.saturating_sub(1);
    let toks_per_sec = if decode.as_secs_f64() > 0.0 {
        decode_tokens as f64 / decode.as_secs_f64()
    } else {
        0.0
    };
    tracing::info!(
        "[timing] total: {:.2?} | prefill: {:.2?} | decode: {:.2?} ({} tok, {:.1} tok/s)",
        total,
        prefill,
        decode,
        decode_tokens,
        toks_per_sec
    );
}
