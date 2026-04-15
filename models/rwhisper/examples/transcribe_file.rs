use kalosm::sound::*;
use rodio::Decoder;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    eprintln!("Starting transcription...");

    // Create a new large whisper model
    eprintln!("Building model...");
    let model = WhisperBuilder::default()
        .with_source(WhisperSource::tiny_en())
        .build()
        .await?;
    eprintln!("Model built successfully");

    // Load audio from a file
    let contents = std::fs::read("./models/rwhisper/examples/samples_jfk.wav").unwrap();
    let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();

    let (_stream, stream_handle) = rodio::OutputStream::try_default()?;
    let sink = rodio::Sink::try_new(&stream_handle).unwrap();
    let rate = audio.sample_rate() as f32;

    // Transcribe the source audio into text
    eprintln!("Starting transcription...");
    let mut text = model.transcribe(audio);

    eprintln!("Waiting for segments...");
    let mut segment_count = 0;
    // As the model transcribes the audio, print the text to the console
    while let Some(segment) = text.next().await {
        segment_count += 1;
        let chunks: Vec<_> = segment.chunks().collect();
        eprintln!(
            "Received segment {}: {} chunks",
            segment_count,
            chunks.len()
        );
        for chunk in chunks {
            eprintln!("Chunk: {:?}", chunk);
            print!("{chunk}");
            // Play the audio chunk
            if let Some(timestamp) = chunk.timestamp() {
                let start = timestamp.start;
                let end = timestamp.end;
                let start = (start * rate) as usize;
                let end = (end * rate) as usize;
                let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();
                let audio_chunk = audio.skip(start).take(end - start).collect::<Vec<_>>();
                let audio_source = rodio::buffer::SamplesBuffer::new(1, rate as u32, audio_chunk);
                sink.append(audio_source);
                sink.sleep_until_end();
            }
        }
    }

    eprintln!("Transcription complete. Total segments: {}", segment_count);
    Ok(())
}
