//! Temporary perf probe for the Qwen2.5-VL mrope split+op+cat path.

use kalosm_llama::*;
use kalosm_model_types::ModelLoadingProgress;
use prelude::{StreamExt, TextCompletionModelExt};

fn main() {
    let _ = tracing_subscriber::fmt::try_init();
    pollster::block_on(async {
        let model = Llama::builder()
            .with_source(LlamaSource::qwen_2_5_3b_vl_chat_q4())
            .build_with_loading_handler(|_: ModelLoadingProgress| {})
            .await
            .unwrap();

        let prompt = "The capital of France is";
        let mut stream = model.complete(prompt).take(72);
        let mut text = String::new();
        // Warmup: first tokens pay one-time kernel compilation.
        for _ in 0..8 {
            if let Some(token) = stream.next().await {
                text.push_str(&token);
            }
        }
        let start = std::time::Instant::now();
        let mut tokens = 0usize;
        while let Some(token) = stream.next().await {
            text.push_str(&token);
            tokens += 1;
        }
        let elapsed = start.elapsed();
        println!("OUTPUT: {}", text.replace('\n', " | "));
        println!(
            "steady-state tokens={tokens} elapsed={elapsed:?} tps={:.2}",
            tokens as f64 / elapsed.as_secs_f64()
        );
    });
}
