// Native GPU reproduction for the decode-plan-cache gibberish bug.
// Generates a short completion on the default (GPU) device and prints it, so
// coherence can be eyeballed. Run with the cache on (default) vs off
// (FUSOR_DISABLE_DECODE_PLAN_CACHE=1) to compare.
use kalosm_llama::prelude::*;
use kalosm_model_types::ModelLoadingProgress;
use std::io::Write;

fn main() {
    pollster::block_on(async {
        let model = Llama::builder()
            .with_source(LlamaSource::qwen_2_5_0_5b_instruct())
            .build_with_loading_handler(|_: ModelLoadingProgress| {})
            .await
            .unwrap();

        let mut stream = model("Once upon a time there was a penguin named Peng.").take(40);
        while let Some(token) = stream.next().await {
            print!("{token}");
            std::io::stdout().flush().unwrap();
        }
        println!();
    });
}
