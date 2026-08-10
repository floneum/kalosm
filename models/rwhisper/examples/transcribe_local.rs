//! Transcribe a local wav file with the cached tiny-en model, printing the
//! transcript to stdout. No network access is needed once the model is
//! cached.
//!
//! Usage: cargo run --release -p rwhisper --example transcribe_local [path-to-wav]

use futures_util::StreamExt;
use rwhisper::{WhisperBuilder, WhisperSource};

fn main() -> Result<(), anyhow::Error> {
    let _ = tracing_subscriber::fmt::try_init();
    pollster::block_on(async {
        let path = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "./models/rwhisper/examples/samples_jfk.wav".to_string());
        let contents = std::fs::read(&path)?;
        let audio = rodio::Decoder::new(std::io::Cursor::new(contents))?;

        let model = WhisperBuilder::default()
            .with_source(WhisperSource::tiny_en())
            .build()
            .await?;

        let start = std::time::Instant::now();
        let mut text = model.transcribe(audio);
        let mut transcript = String::new();
        while let Some(segment) = text.next().await {
            print!("{}", segment.text());
            transcript += segment.text();
        }
        println!();
        println!(
            "transcribed {} chars in {:.2} seconds",
            transcript.len(),
            start.elapsed().as_secs_f64()
        );
        Ok(())
    })
}
