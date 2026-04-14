use futures_util::Stream;
use kalosm::sound::*;
use std::path::PathBuf;
use tokio::time::{Duration, Instant};

async fn print_segments<S>(mut text: S) -> Result<(), anyhow::Error>
where
    S: Stream<Item = Segment> + Unpin + Send,
{
    text.to_std_out().await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let source = if let Ok(path) = std::env::var("RWHISPER_COHERE_GGUF") {
        WhisperSource::cohere_transcribe_03_2026_local(path)
    } else if let Ok(path) = std::env::var("RWHISPER_MOONSHINE_GGUF") {
        WhisperSource::moonshine_streaming_local(path)
    } else if std::env::var("RWHISPER_COHERE").is_ok() {
        WhisperSource::cohere_transcribe_03_2026()
    } else if let Ok(dir) = std::env::var("RWHISPER_WHISPER_DIR") {
        let dir = PathBuf::from(dir);
        let model_path = dir.join("whisper-tiny-en.gguf.real");
        let model_path = if model_path.exists() {
            model_path
        } else {
            dir.join("whisper-tiny-en.gguf")
        };
        WhisperSource::new(
            FileSource::local(model_path),
            FileSource::local(dir.join("tokenizer-tiny-en.json")),
            FileSource::local(dir.join("config-tiny-en.json")),
            false,
            Some(&[
                [1, 0],
                [2, 0],
                [2, 5],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
                [3, 4],
            ]),
        )
    } else if std::env::var("RWHISPER_MOONSHINE").is_ok() {
        WhisperSource::moonshine_streaming_tiny()
    } else {
        WhisperSource::default()
    };
    let duration = std::env::var("RWHISPER_MAX_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(60);
    let timestamped = std::env::var("RWHISPER_TIMESTAMPED").ok().as_deref() == Some("1");

    let model = WhisperBuilder::default()
        .with_source(source)
        .build()
        .await?;

    let audio = MicInput::default()
        .record_until(Instant::now() + Duration::from_secs(duration))
        .await;
    let mut task = model.transcribe(audio);
    if timestamped {
        task = task.timestamped();
    }
    print_segments(task).await?;

    Ok(())
}
