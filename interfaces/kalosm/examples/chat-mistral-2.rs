use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        let model = Llama::builder()
            .with_source(LlamaSource::mistral_7b_instruct_2())
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
