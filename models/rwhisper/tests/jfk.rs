use futures_util::StreamExt;
use rwhisper::{WhisperBuilder, WhisperSource};

/// Run explicitly because this loads model weights (and downloads them if uncached).
/// cargo test -p rwhisper --test jfk --release -- --ignored --nocapture
#[test]
#[ignore = "requires Whisper model weights and runs inference"]
fn transcribes_jfk() -> anyhow::Result<()> {
    pollster::block_on(async {
        let model = WhisperBuilder::default()
            .with_source(WhisperSource::tiny_en())
            .build()
            .await?;
        let audio = rodio::Decoder::new(std::io::Cursor::new(include_bytes!(
            "../examples/samples_jfk.wav"
        )))?;
        let mut stream = model.transcribe(audio);
        let mut transcript = String::new();
        while let Some(segment) = stream.next().await {
            transcript.push_str(segment.text());
        }
        println!("Transcript: {transcript}");
        let normalized = transcript
            .split(|c: char| !c.is_alphanumeric())
            .filter(|word| !word.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase();
        assert_eq!(
            normalized,
            "and so my fellow americans ask not what your country can do for you ask what you can do for your country",
            "JFK transcription differs from the reference (ignoring punctuation and case)"
        );
        Ok(())
    })
}
