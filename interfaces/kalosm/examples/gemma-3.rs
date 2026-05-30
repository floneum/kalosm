use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        let model = Llama::builder()
            .with_source(LlamaSource::gemma_3_4b_chat())
            .build()
            .await
            .unwrap();

        let mut chat = model.chat();

        loop {
            println!();
            let mut response = chat(&prompt_input("\n> ").unwrap());
            response.to_std_out().await.unwrap();
            response.await.unwrap();
        }
    });
}
