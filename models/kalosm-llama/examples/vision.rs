#![recursion_limit = "256"]

use kalosm_llama::prelude::*;
use kalosm_streams::text_stream::TextStream;
use std::time::Instant;

async fn collect_or_bench(mut response: impl TextStream + Unpin, max_tokens: u32) {
    let bench = std::env::var_os("KALOSM_VISION_BENCH").is_some();
    let total_start = Instant::now();
    let mut first = None;
    let mut last = None;
    let mut tokens = 0usize;
    let mut output = String::new();
    while let Some(token) = response.next().await {
        let now = Instant::now();
        first.get_or_insert(now);
        last = Some(now);
        tokens += 1;
        output.push_str(&token);
    }

    if !bench {
        print!("{output}");
        println!("\n");
        return;
    }

    let total_elapsed = total_start.elapsed().as_secs_f64();
    let steady_elapsed = first
        .zip(last)
        .map(|(first, last)| (last - first).as_secs_f64())
        .unwrap_or_default();
    let steady_tokens = tokens.saturating_sub(1);
    let total_tps = if total_elapsed > 0.0 {
        tokens as f64 / total_elapsed
    } else {
        0.0
    };
    let steady_tps = if steady_elapsed > 0.0 {
        steady_tokens as f64 / steady_elapsed
    } else {
        0.0
    };

    print!("{output}");
    println!("\n");
    eprintln!(
        "vision_bench tokens={tokens} max_tokens={max_tokens} total_s={total_elapsed:.3} total_tok_s={total_tps:.2} steady_tok_s={steady_tps:.2}"
    );
}

// The demo image may be fetched over HTTP, which needs a tokio reactor.
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let source = match std::env::var("KALOSM_VISION_SOURCE").as_deref() {
        Ok("gemma4") => {
            let mut source = LlamaSource::gemma_4_e2b_it_qat_chat().with_vision_model(
                kalosm_model_types::FileSource::HuggingFace {
                    model_id: "unsloth/gemma-4-E2B-it-qat-GGUF".into(),
                    revision: "main".into(),
                    file: "mmproj-F16.gguf".into(),
                },
            );
            if std::env::var_os("KALOSM_LLAMA_MTP_DRAFT").is_some() {
                let file = std::env::var("KALOSM_LLAMA_MTP_FILE")
                    .unwrap_or_else(|_| "MTP/gemma-4-E2B-it-Q4_0-MTP.gguf".into());
                source = source.with_mtp_model(kalosm_model_types::FileSource::HuggingFace {
                    model_id: "unsloth/gemma-4-E2B-it-qat-GGUF".into(),
                    revision: "main".into(),
                    file,
                });
            }
            source
        }
        _ => LlamaSource::qwen_2_5_3b_vl_chat_q4(),
    };
    let mut builder = Llama::builder().with_source(source);
    if std::env::var_os("KALOSM_VISION_CPU").is_some() {
        builder = builder.with_device(kalosm_llama::Device::Cpu);
    }
    let model = builder.build().await.unwrap();

    let mut chat = model.chat();
    let max_tokens = std::env::var("KALOSM_VISION_MAX_TOKENS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(64);
    let mut sampler = GenerationParameters::new().with_max_length(max_tokens);
    if let Ok(seed) = std::env::var("KALOSM_VISION_SEED") {
        if let Ok(seed) = seed.parse::<u64>() {
            sampler = sampler.with_seed(seed);
        }
    }
    if std::env::var_os("KALOSM_VISION_STANDARD").is_some() {
        sampler = sampler.with_standard_sampler();
    }
    if let Ok(temperature) = std::env::var("KALOSM_VISION_TEMPERATURE") {
        if let Ok(temperature) = temperature.parse::<f32>() {
            sampler = sampler.with_temperature(temperature);
            if temperature <= 0.0 {
                sampler = sampler.with_standard_sampler();
            }
        }
    }
    if std::env::var_os("KALOSM_VISION_TEXT_ONLY").is_some() {
        let prompt = std::env::var("KALOSM_VISION_TEXT_PROMPT").unwrap_or_else(|_| {
            "Answer in one short sentence: what color is a ripe banana?".into()
        });
        let mut response = chat(&prompt).with_sampler(sampler);
        if std::env::var_os("KALOSM_VISION_COLLECT").is_some() {
            collect_or_bench(response, max_tokens).await;
            return;
        }
        response.to_std_out().await.unwrap();
        println!("\n");
        return;
    }

    let image_source = if let Ok(url) = std::env::var("KALOSM_VISION_URL") {
        MediaSource::url(url)
    } else if let Ok(path) = std::env::var("KALOSM_VISION_IMAGE") {
        MediaSource::file(path).unwrap()
    } else {
        MediaSource::url("https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg")
    };
    let prompt =
        std::env::var("KALOSM_VISION_PROMPT").unwrap_or_else(|_| "Describe this image.".into());

    let mut response = chat(&(
        MediaChunk::new(image_source, MediaType::Image),
        prompt.as_str(),
    ))
    .with_sampler(sampler);
    if std::env::var_os("KALOSM_VISION_COLLECT").is_some() {
        collect_or_bench(response, max_tokens).await;
        return;
    }
    response.to_std_out().await.unwrap();
    println!("\n");
}
