// KALOSM_TRACE_DECODE_TIMING=1 \
// FUSOR_TRACE_DECODE=1 \
// KALOSM_LLAMA_GPU_SAMPLE_TOKEN=1 \
// KALOSM_LLAMA_GPU_FUSED_LOGITS=1 \
// KALOSM_LLAMA_GPU_SAMPLE_TOP_K=128 \
// KALOSM_LLAMA_UNBOUNDED_DECODE_RESERVE=512 \
// cargo run -p kalosm --example chat --features llama --release
use kalosm::language::*;
use std::io::{BufWriter, Write};
use std::time::Duration;
use std::time::Instant;

#[tokio::main]
async fn main() {
        let model = Llama::new_chat().await.unwrap();
        let system_prompt = "The assistant will act like a pirate";

        let mut warmup_chat = model.chat().with_system_prompt(system_prompt);
        let warmup_sampler = GenerationParameters::new().with_max_length(4);
        let _ = warmup_chat(&"hi".to_string())
            .with_sampler(warmup_sampler)
            .collect::<Vec<_>>()
            .await;

        let mut chat = model.chat().with_system_prompt(system_prompt);

        loop {
            let mut response = chat(&prompt_input("\n> ").unwrap());
            let stdout = std::io::stdout();
            let mut stdout = BufWriter::with_capacity(8192, stdout.lock());
            let mut last_flush = Instant::now();
            // Time between the first and last streamed chunks. This excludes
            // prefill latency before the first token and any task-cleanup work
            // after the last token, isolating steady-state generation.
            let mut first: Option<Instant> = None;
            let mut last: Option<Instant> = None;
            let mut tokens: usize = 0;
            while let Some(chunk) = response.next().await {
                let now = Instant::now();
                first.get_or_insert(now);
                last = Some(now);
                tokens += 1;
                stdout.write_all(chunk.as_bytes()).unwrap();
                if last_flush.elapsed() >= Duration::from_millis(50) {
                    stdout.flush().unwrap();
                    last_flush = Instant::now();
                }
            }
            stdout.flush().unwrap();
            if let (Some(first), Some(last)) = (first, last) {
                let elapsed = (last - first).as_secs_f64();
                // tokens-1 inter-token gaps fit inside `elapsed`.
                let rate = if elapsed > 0.0 && tokens > 1 {
                    (tokens - 1) as f64 / elapsed
                } else {
                    0.0
                };
                println!("\n[{tokens} tokens in {elapsed:.2}s — {rate:.2} tok/s]");
            }
        }
}
