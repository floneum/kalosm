// You must set the environment variable `ANTHROPIC_API_KEY` (https://console.anthropic.com/settings/keys) to run this example.

use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt::init();

        let llm = AnthropicCompatibleChatModel::builder()
            .with_claude_3_5_haiku()
            .build();
        let prompt = "Write a 300 word essay about why the capital of France is Paris";
        print!("{prompt}");

        let mut chat = llm.chat();
        chat(&prompt).to_std_out().await.unwrap();
    });
}
