// You must set the environment variable OPENAI_API_KEY (https://platform.openai.com/account/api-keys) to run this example.

use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt::init();

        let llm = OpenAICompatibleChatModel::builder()
            .with_gpt_4o_mini()
            .build();
        let prompt = "Write a 300 word essay about why the capital of France is Paris";
        print!("{prompt}");

        let mut chat = llm.chat();
        chat(&prompt).to_std_out().await.unwrap();
    });
}
