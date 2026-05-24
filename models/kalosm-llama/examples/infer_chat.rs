use kalosm_llama::prelude::*;
use kalosm_streams::text_stream::TextStream;

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt::init();
        let model = Llama::builder()
            .with_source(LlamaSource::qwen_2_5_7b_instruct())
            .build()
            .await
            .unwrap();

        println!("First message\n");

        let mut chat = model.chat();
        loop {
            let mut response = chat(&prompt_input("> ").unwrap());
            response.to_std_out().await.unwrap();
            println!("\n");
        }
    });
}
