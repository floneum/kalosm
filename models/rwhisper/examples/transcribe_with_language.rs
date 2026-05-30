use kalosm::sound::*;
use std::time::{Duration, Instant};

fn main() -> Result<(), anyhow::Error> {
    pollster::block_on(async {
        // Record audio from the microphone for 10 seconds.
        let audio =
            MicInput::default().record_until_blocking(Instant::now() + Duration::from_secs(10));

        // Create a new small whisper model.
        let model = WhisperBuilder::default()
            .with_source(WhisperSource::large_v3_turbo())
            .build()
            .await?;

        // Transcribe the audio.
        let mut text = model
            .transcribe(audio)
            .with_language(WhisperLanguage::Hindi);

        // As the model transcribes the audio, print the text to the console.
        text.to_std_out().await?;

        Ok(())
    })
}
