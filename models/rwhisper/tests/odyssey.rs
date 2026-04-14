use kalosm::sound::*;
use rodio::{Decoder, Source};
use std::{io::Cursor, time::Duration};

#[tokio::test]
async fn transcribe_the_odyssey() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt::init();
    // First download the audio file from the internet.
    // archive.org redirects through CDN nodes; reqwest follows redirects by
    // default. We surface the final status and a couple of retries so a
    // transient 5xx doesn't fail the whole test run.
    let url = "https://archive.org/download/odyssey_2409_librivox/odyssey_00_homer.mp3";
    let mut last_err: Option<anyhow::Error> = None;
    let content = 'fetched: {
        for attempt in 0..3 {
            match reqwest::get(url).await {
                Ok(r) if r.status().is_success() => match r.bytes().await {
                    Ok(b) => break 'fetched b,
                    Err(e) => last_err = Some(e.into()),
                },
                Ok(r) => last_err = Some(anyhow::anyhow!("HTTP {} from {url}", r.status())),
                Err(e) => last_err = Some(e.into()),
            }
            tracing::warn!("odyssey download attempt {} failed; retrying", attempt + 1);
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
        return Err(last_err.unwrap_or_else(|| anyhow::anyhow!("odyssey download failed")));
    };

    // Create a new small whisper model
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::tiny_en())
        .build()
        .await?;

    let source = Cursor::new(content);
    // Decode that sound file into a source
    let audio = Decoder::new(source).unwrap();
    let audio = audio.take_duration(Duration::from_secs(10));

    // Transcribe the source audio into text
    let mut text = model.transcribe(audio);

    // As the model transcribes the audio, print the text to the console
    while let Some(segment) = text.next().await {
        for chunk in segment.chunks() {
            if let Some(timestamp) = chunk.timestamp() {
                println!("{:0.2}..{:0.2}", timestamp.start, timestamp.end);
                println!("{chunk}");
            } else {
                println!("no timestamp for {chunk}");
            }
        }
    }

    Ok(())
}
