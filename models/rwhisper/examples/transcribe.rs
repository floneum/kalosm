use kalosm::sound::*;
use std::time::{Duration, Instant};

fn main() -> Result<(), anyhow::Error> {
    pollster::block_on(async {
        // Record audio from the microphone for 60 seconds.
        let audio =
            MicInput::default().record_until_blocking(Instant::now() + Duration::from_secs(60));

        // Create a new small whisper model.
        let model = WhisperBuilder::default().build().await?;

        // Transcribe the audio.
        let mut text = model.transcribe(audio);

        // As the model transcribes the audio, print the text to the console.
        text.to_std_out().await?;

        Ok(())
    })
}
