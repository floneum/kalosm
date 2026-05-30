//! This example works for any endpoint that has the same interface as OpenAI's API.
//! This can be useful if you want to self-host a remote model.
//!
//! If you would like to self host a llama model, you can use a tool like litellm to host the model: https://github.com/BerriAI/litellm#openai-proxy---docs

use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt::init();

        let base_url = std::env::var("OPENAI_API_BASE").expect("Custom OPENAI_API_BASE not set");
        let model = std::env::var("OPENAI_API_MODEL").expect("Custom OPENAI_API_MODEL not set");
        let client = OpenAICompatibleClient::new().with_base_url(base_url);
        let llm = OpenAICompatibleChatModel::builder()
            .with_client(client)
            .with_model(model)
            .build();
        let prompt = "Write a 300 word essay about why the capital of France is Paris";
        print!("{prompt}");

        let mut chat = llm.chat();
        chat(&prompt).to_std_out().await.unwrap();
    });
}
