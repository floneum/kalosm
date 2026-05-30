#![allow(unused)]
use kalosm::language::*;
use serde::Deserialize;

fn main() {
    pollster::block_on(async {
        // You can derive an efficient parser for your struct with the `Parse` trait
        // OpenAI doesn't support root anyof schemas, so we need to wrap the constraints in a struct
        #[derive(Parse, Clone, Schema, Deserialize, Debug)]
        struct Response {
            action: Action,
        }

        #[derive(Parse, Clone, Schema, Deserialize, Debug)]
        #[serde(tag = "type")]
        #[serde(content = "data")]
        pub enum Action {
            Do(String),
            Say(String),
        }

        let model: BoxedStructuredChatModel<Response> = loop {
            let input = prompt_input("Choose Model (gpt, llama, or phi): ").unwrap();
            match input.to_lowercase().as_str() {
                "gpt" => {
                    break OpenAICompatibleChatModel::builder()
                        .with_gpt_4o_mini()
                        .build()
                        .boxed_typed_chat_model()
                }
                "llama" => {
                    break Llama::builder()
                        .with_source(LlamaSource::llama_3_1_8b_chat())
                        .build()
                        .await
                        .unwrap()
                        .boxed_typed_chat_model()
                }
                "phi" => {
                    break Llama::builder()
                        .with_source(LlamaSource::phi_3_5_mini_4k_instruct())
                        .build()
                        .await
                        .unwrap()
                        .boxed_typed_chat_model()
                }
                _ => {}
            }
        };

        let mut chat = model
            .chat()
            .with_system_prompt("The assistant will act like a pirate. You will respond with either something you do or something you say. Respond with JSON in the format { \"type\": \"Say\", \"data\": \"hello\" } or { \"type\": \"Do\", \"data\": \"run away\" }");

        // Then chat with the session
        loop {
            let mut response = chat(&prompt_input("\n> ").unwrap()).typed::<Response>();
            response.to_std_out().await.unwrap();
            println!("{:?}", response.await);
        }
    });
}
