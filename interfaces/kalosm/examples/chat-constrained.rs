#![allow(unused)]
use kalosm::language::*;
use std::sync::Arc;

fn main() {
    pollster::block_on(async {
        // You can derive an efficient parser for your struct with the `Parse` trait
        #[derive(Parse, Clone)]
        pub enum Response {
            Do(String),
            Say(String),
        }

        // Create a parser and warp it in Arc so it implements Clone
        let parser = Arc::new(Response::new_parser());

        // Create a chat session with the default chat model with the parser as constraints
        let model = Llama::new_chat().await.unwrap();
        let mut chat = model
            .chat()
            .with_system_prompt("The assistant will act like a pirate. You will respond with either something you do or something you say. Respond with JSON in the format { \"type\": \"Say\", \"data\": \"hello\" } or { \"type\": \"Do\", \"data\": \"run away\" }");

        // Then chat with the session
        loop {
            chat(&prompt_input("\n> ").unwrap())
                .with_constraints(parser.clone())
                .to_std_out()
                .await
                .unwrap();
        }
    });
}
