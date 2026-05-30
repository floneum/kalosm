use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        let model = Llama::builder()
            .with_source(LlamaSource::zephyr_7b_beta())
            .build()
            .await
            .unwrap();
        let prompt = "<|system|>

    </s>
    <|user|>
    What is your favorite story from your adventures?</s>
    <|assistant|>";

        print!("{prompt}");
        model(prompt).to_std_out().await.unwrap();
    });
}
