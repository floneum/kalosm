use kalosm_llama::*;
use kalosm_model_types::ModelLoadingProgress;
use prelude::StreamExt;
use prelude::TextCompletionModelExt;

#[tokio::main]
async fn main() {
    fn progress(_: ModelLoadingProgress) {}

    #[inline(never)]
    async fn load_small() {
        let model = Llama::builder()
            .with_source(LlamaSource::llama_8b())
            .build_with_loading_handler(progress)
            .await
            .unwrap();
        let prompt = "Hello world";

        let tokens = model.tokenizer().encode(prompt, false).unwrap().len();
        for _ in 0..100 {
            let start_time = std::time::Instant::now();
            let _ = model.complete(prompt).next().await.unwrap();
            let elapsed = start_time.elapsed();
            println!("\n\nLoaded {tokens} tokens in {elapsed:?}");
            println!(
                "Tokens per second: {:.2}",
                tokens as f64 / elapsed.as_secs_f64()
            );
        }
    }

    #[inline(never)]
    async fn load_large() {
        let model = Llama::builder()
            .with_source(LlamaSource::llama_8b())
            .build_with_loading_handler(progress)
            .await
            .unwrap();
        let prompt = "Hello world".repeat(600);

        let tokens = model.tokenizer().encode(&prompt, false).unwrap().len();
        for _ in 0..100 {
            let start_time = std::time::Instant::now();
            let _ = model.complete(&prompt).next().await.unwrap();
            let elapsed = start_time.elapsed();
            println!("\n\nLoaded {tokens} tokens in {elapsed:?}");
            println!(
                "Tokens per second: {:.2}",
                tokens as f64 / elapsed.as_secs_f64()
            );
        }
    }

    #[inline(never)]
    async fn generate() {
        let model = Llama::builder()
            .with_source(LlamaSource::llama_8b())
            .build_with_loading_handler(progress)
            .await
            .unwrap();
        let prompt = "Hello world";

        for _ in 0..100 {
            let start_time = std::time::Instant::now();
            let mut tokens = 0;
            let mut stream = model.complete(prompt).take(100);
            while (stream.next().await).is_some() {
                tokens += 1;
            }
            let elapsed = start_time.elapsed();
            println!("\n\nGenerated {tokens} tokens in {elapsed:?}");
            println!(
                "Tokens per second: {:.2}",
                tokens as f64 / elapsed.as_secs_f64()
            );
        }
    }

    load_small().await;
    generate().await;
    load_large().await;
}
