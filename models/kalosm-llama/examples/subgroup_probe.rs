// Deterministic greedy generation probe for bisecting the no-subgroup bug.
//
//   cargo run -p kalosm-llama --example subgroup_probe                 # subgroups on
//   FUSOR_DISABLE_SUBGROUPS=1 cargo run -p kalosm-llama --example subgroup_probe
//
// Greedy (top_k=1) + fixed seed makes the run deterministic, so any difference
// between the two runs localizes the bug. Pair with KALOSM_LLAMA_GPU_SAMPLE_TOP_K
// to compare greedy (top_k=1) vs a chunked top-k path.

use fusor::Device;
use kalosm_llama::prelude::*;
use std::io::Write;

fn main() {
    pollster::block_on(async {
        if std::env::var_os("KALOSM_LLAMA_GPU_SAMPLE_TOP_K").is_none() {
            std::env::set_var("KALOSM_LLAMA_GPU_SAMPLE_TOP_K", "1");
        }
        let device = Device::gpu().await.expect("no GPU device available");
        let model = Llama::builder()
            .with_source(LlamaSource::qwen_2_5_0_5b_instruct())
            .with_device(device)
            .build()
            .await
            .unwrap();

        let sampler = GenerationParameters::new().with_max_length(40).with_seed(0);
        let mut stream = model
            .complete("Once upon a time there was a penguin named Peng.")
            .with_sampler(sampler);
        while let Some(token) = stream.next().await {
            print!("{token}");
            std::io::stdout().flush().unwrap();
        }
        println!();
    });
}
