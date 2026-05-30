//! You can have multiple chat instances with the same model.

use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        let model = Llama::builder()
            .with_source(LlamaSource::qwen_2_5_1_5b_instruct())
            .build()
            .await
            .unwrap();
        let mut agent1 = model
            .chat()
            .with_system_prompt("The assistant will act like a pirate.");
        let mut agent2 = model
            .chat()
            .with_system_prompt("The assistant is curious and will ask questions about the world.");

        let mut response = String::from("Is there anything you want to know about the world?");
        loop {
            println!("User:");
            let mut stream = agent2(&response);
            stream.to_std_out().await.unwrap();
            let user_question = stream.await.unwrap();
            println!();

            println!("Assistant:");
            let mut stream = agent1(&user_question);
            stream.to_std_out().await.unwrap();
            response = stream.await.unwrap();
            println!();
        }
    });
}
