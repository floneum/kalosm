// Generates a short completion on the GPU. Run it twice to check whether the
// dirty-buffer reuse condition corrupts generation:
//
//   cargo run -p kalosm-llama --example dirty_probe
//   FUSOR_DIRTY_BUFFERS=1 cargo run -p kalosm-llama --example dirty_probe
//
// If the second run produces garbage while the first is coherent, the model
// relies on zero-initialized pooled buffers somewhere (the web bug).

use fusor::Device;
use kalosm_llama::prelude::*;
use std::io::Write;

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt::init();
        let device = Device::gpu().await.expect("no GPU device available");
        let model = Llama::builder()
            .with_source(LlamaSource::qwen_2_5_0_5b_instruct())
            .with_device(device)
            .build()
            .await
            .unwrap();

        let mut story = model("Once upon a time there was a penguin named Peng.");

        let mut tokens = 0;
        while let Some(token) = story.next().await {
            print!("{}", token);
            std::io::stdout().flush().unwrap();
            tokens += 1;
            if tokens >= 60 {
                break;
            }
        }
        println!();
    });
}
