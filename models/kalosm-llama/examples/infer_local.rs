#![recursion_limit = "256"]

//! Run inference against a local GGUF file (no network), on the GPU.
//!
//! Usage: cargo run --release --example infer_local [path-to-gguf]

use kalosm_llama::prelude::*;
use kalosm_llama::Device;
use std::io::Write;
use std::path::PathBuf;

const DEFAULT_MODEL: &str = "/Users/evanalmloff/.cache/huggingface/hub/models--lmstudio-community--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/8601e6db71269a2b12255ebdf09ab75becf22cc8/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt::init();
        let path = std::env::args()
            .nth(1)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
        assert!(path.exists(), "model file not found: {}", path.display());

        let model = Llama::builder()
            .with_source(LlamaSource::new(FileSource::Local(path)))
            .with_device(Device::gpu().await.unwrap())
            .build()
            .await
            .unwrap();

        let prompt = "Once upon a time there was a penguin named Peng.";
        let mut story = model(prompt).with_sampler(GenerationParameters::new().with_max_length(64));

        print!("{prompt}");
        std::io::stdout().flush().unwrap();
        let mut start = None;
        let mut tokens = 0;
        while let Some(token) = story.next().await {
            // Start the clock at the first emitted token so prefill/compile
            // time does not pollute the decode rate.
            start.get_or_insert_with(std::time::Instant::now);
            print!("{token}");
            std::io::stdout().flush().unwrap();
            tokens += 1;
        }
        let elapsed = start.map(|s| s.elapsed()).unwrap_or_default();
        println!();
        println!(
            "{} tokens in {:.2} seconds ({:.2} tokens/second)",
            tokens,
            elapsed.as_secs_f64(),
            tokens as f64 / elapsed.as_secs_f64()
        );
    });
}
