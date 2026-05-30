// This example shows how to cancel a transcription early
// When you drop the stream, the model will stop transcribing after the current segment

use futures_util::StreamExt;
use rodio::Decoder;
use rwhisper::*;
use std::fs::File;
use std::io::BufReader;

fn main() -> Result<(), anyhow::Error> {
    pollster::block_on(async {
        // Read a path from the CLI arguments
        let path = std::env::args().nth(1).unwrap();

        // Create a new small whisper model
        let model = WhisperBuilder::default().build().await?;

        for _ in 0..10 {
            // Load audio from a file
            let file = BufReader::new(File::open(path.clone()).unwrap());
            // Decode that sound file into a source
            let audio = Decoder::new(file).unwrap();

            // Transcribe the source audio into text
            // Only transcribe the first segment
            let mut text = model.transcribe(audio).take(1);

            // As the model transcribes the audio, print the text to the console
            while let Some(text) = text.next().await {
                print!("{}", text.text());
            }
        }

        Ok(())
    })
}
