use fusor::Device;
use kalosm_llama::prelude::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
    let model = Llama::builder()
        .with_source(LlamaSource::qwen_2_5_0_5b_instruct())
        .with_device(Device::cpu())
        .build()
        .await
        .unwrap();

    let mut story = model("Once upon a time there was a penguin named Peng.");

    let start = std::time::Instant::now();
    let mut tokens = 0;
    while let Some(token) = story.next().await {
        print!("{}", token);
        std::io::stdout().flush().unwrap();
        tokens += 1;
    }
    let elapsed = start.elapsed();
    println!();
    println!(
        "{} tokens in {:.2} seconds ({:.2} tokens/second)",
        tokens,
        elapsed.as_secs_f64(),
        tokens as f64 / elapsed.as_secs_f64()
    );
}
