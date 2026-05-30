use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        let model = Llama::builder()
            // To use a custom model, you can set the LlamaSource to a custom model
            .with_source(LlamaSource::new(
                // Llama source takes a gguf file to load the model, tokenizer, and chat template from
                FileSource::HuggingFace {
                    model_id: "QuantFactory/SmolLM-1.7B-Instruct-GGUF".to_string(),
                    revision: "main".to_string(),
                    file: "SmolLM-1.7B-Instruct.Q4_K_M.gguf".to_string(),
                },
            ))
            .build()
            .await
            .unwrap();

        let mut chat = model
            .chat()
            .with_system_prompt("The assistant will act like a pirate");

        loop {
            chat(&prompt_input("\n> ").unwrap())
                .to_std_out()
                .await
                .unwrap();
        }
    });
}
