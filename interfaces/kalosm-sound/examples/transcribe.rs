use futures_util::Stream;
use kalosm::sound::*;
use std::{path::PathBuf, pin::Pin};
use tokio::time::{Duration, Instant};

struct TimedAsyncSource<S> {
    source: S,
    deadline: Instant,
}

impl<S> TimedAsyncSource<S> {
    fn new(source: S, deadline: Instant) -> Self {
        Self { source, deadline }
    }
}

impl<S: AsyncSource + Unpin> Stream for TimedAsyncSource<S> {
    type Item = f32;

    fn poll_next(
        self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let myself = self.get_mut();
        if Instant::now() >= myself.deadline {
            return std::task::Poll::Ready(None);
        }

        let stream = myself.source.as_stream();
        let mut stream = std::pin::pin!(stream);
        stream.as_mut().poll_next(cx)
    }
}

impl<S: AsyncSource + Unpin> AsyncSource for TimedAsyncSource<S> {
    fn as_stream(&mut self) -> impl Stream<Item = f32> + '_ {
        self
    }

    fn sample_rate(&self) -> u32 {
        self.source.sample_rate()
    }
}

async fn print_segments<S>(mut text: S) -> Result<(), anyhow::Error>
where
    S: Stream<Item = Segment> + Unpin + Send,
{
    text.to_std_out().await?;
    Ok(())
}

async fn run() -> Result<(), anyhow::Error> {
    let source = if let Ok(dir) = std::env::var("RWHISPER_COHERE_DIR") {
        WhisperSource::cohere_transcribe_03_2026_local(dir)
    } else if let Ok(dir) = std::env::var("RWHISPER_MOONSHINE_DIR") {
        WhisperSource::moonshine_streaming_local(dir)
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
        WhisperSource::tiny_en()
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

    let audio = TimedAsyncSource::new(
        MicInput::default().stream(),
        Instant::now() + Duration::from_secs(duration),
    );
    let mut task = audio.transcribe(model);
    if timestamped {
        task = task.timestamped();
    }
    print_segments(task).await?;

    Ok(())
}

fn main() -> Result<(), anyhow::Error> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(run())
}
