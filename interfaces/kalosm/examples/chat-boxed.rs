#![allow(unused)]
use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        let model = loop {
            let input = prompt_input("Choose Model (gpt, claude, llama, or phi): ").unwrap();
            match input.to_lowercase().as_str() {
                "gpt" => {
                    break OpenAICompatibleChatModel::builder()
                        .with_gpt_4o_mini()
                        .build()
                        .boxed_chat_model()
                }
                "claude" => {
                    break AnthropicCompatibleChatModel::builder()
                        .with_claude_3_5_haiku()
                        .build()
                        .boxed_chat_model()
                }
                "llama" => {
                    break Llama::builder()
                        .with_source(LlamaSource::llama_3_1_8b_chat())
                        .build()
                        .await
                        .unwrap()
                        .boxed_chat_model()
                }
                "phi" => {
                    break Llama::builder()
                        .with_source(LlamaSource::phi_3_5_mini_4k_instruct())
                        .build()
                        .await
                        .unwrap()
                        .boxed_chat_model()
                }
                _ => {}
            }
        };

        let mut chat = model
            .chat()
            .with_system_prompt("The assistant will act like a pirate");

        // Then chat with the session
        loop {
            chat(&prompt_input("\n> ").unwrap())
                .to_std_out()
                .await
                .unwrap();
        }
    });
}
