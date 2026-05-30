use kalosm::language::*;
use std::time::Instant;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let t_load_start = Instant::now();
    let model = Llama::builder()
        .with_source(LlamaSource::qwen_2_5_3b_vl_chat_q4())
        .build()
        .await
        .unwrap();
    eprintln!("[timing] model load: {:.2?}", t_load_start.elapsed());

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
        MediaSource::url("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
    };
    let mut response = chat(&(
        MediaChunk::new(image_source, MediaType::Image),
        "Describe this image.",
    ));
    if let Some(seed) = std::env::var("KALOSM_VISION_SEED")
        .ok()
        .and_then(|seed| seed.parse::<u64>().ok())
    {
        response = response.with_sampler(GenerationParameters::new().with_seed(seed));
    }
    let mut first_token_at: Option<std::time::Duration> = None;
    let mut token_count = 0u64;
    let t_prefill = Instant::now();
    while let Some(token) = response.next().await {
        if first_token_at.is_none() {
            first_token_at = Some(t_prefill.elapsed());
            eprintln!(
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
    eprintln!(
        "[timing] total: {:.2?} | prefill: {:.2?} | decode: {:.2?} ({} tok, {:.1} tok/s)",
        total, prefill, decode, decode_tokens, toks_per_sec
    );
}
