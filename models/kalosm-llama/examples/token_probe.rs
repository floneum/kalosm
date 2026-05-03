use fusor::Device;
use kalosm_llama::prelude::*;
use kalosm_model_types::ModelLoadingProgress;
use std::time::Instant;

#[tokio::main]
async fn main() {
    fn progress(_: ModelLoadingProgress) {}

    let model = Llama::builder()
        .with_source(LlamaSource::llama_3_1_8b_chat())
        .with_device(Device::gpu().await.unwrap())
        .build_with_loading_handler(progress)
        .await
        .unwrap();

    let mut stream = model.complete("Hello world").take(8);
    let total_start = Instant::now();
    let mut last = total_start;
    let mut tokens = 0usize;

    while let Some(token) = stream.next().await {
        let now = Instant::now();
        tokens += 1;
        eprintln!(
            "token {tokens:02}: step={:?} total={:?} text={token:?}",
            now.duration_since(last),
            now.duration_since(total_start)
        );
        last = now;
    }
}
