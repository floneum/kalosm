use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt::init();
        let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
        let llm = Llama::builder()
            .with_source(LlamaSource::qwen_3_0_6b_instruct())
            .build()
            .await
            .unwrap();

        print!("{prompt}");

        llm(prompt).to_std_out().await.unwrap();
    });
}
