use kalosm::sound::*;

fn main() -> Result<(), anyhow::Error> {
    pollster::block_on(async {
        // Create a new whisper model.
        let model = Whisper::new().await?;

        // Stream audio from the microphone
        let mic = MicInput::default();
        let stream = mic.stream();

        // Transcribe the audio into text in chunks based on voice activity.
        let mut text_stream = stream.transcribe(model);

        // Finally, print the text to the console
        text_stream.to_std_out().await.unwrap();

        Ok(())
    })
}
