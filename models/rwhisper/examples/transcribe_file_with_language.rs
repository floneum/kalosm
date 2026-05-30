use kalosm::sound::*;
use rodio::Decoder;

fn main() -> Result<(), anyhow::Error> {
    pollster::block_on(async {
        // tracing_subscriber::fmt::init();

        // Create a new small whisper model
        let model = WhisperBuilder::default()
            .with_source(WhisperSource::large_v3_turbo())
            .build()
            .await?;

        // Load audio from a file
        let contents = std::fs::read("./media/audio_samples/sample_spanish.mp3").unwrap();
        let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();

        // Transcribe the source audio into text
        let mut text = model
            .transcribe(audio)
            .with_language(WhisperLanguage::Spanish);

        text.to_std_out().await?;
        Ok(())
    })
}
