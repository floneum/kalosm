#![recursion_limit = "512"]

use fusor::Device;
use kalosm_llama::prelude::*;
use std::time::Instant;

fn main() {
    pollster::block_on(async {
        let device = Device::new().await.expect("gpu");
        let t = Instant::now();
        let model = Llama::builder()
            .with_source(LlamaSource::llama_3_1_8b_chat())
            .with_device(device)
            .build()
            .await
            .unwrap();
        println!("startup {:.3}s", t.elapsed().as_secs_f64());

        let mut stream = model("The capital of France is");
        let mut out = String::new();
        for _ in 0..10 {
            match stream.next().await {
                Some(tok) => out.push_str(&tok),
                None => break,
            }
        }
        println!("GEN: \"The capital of France is{out}\"");
    });
}
