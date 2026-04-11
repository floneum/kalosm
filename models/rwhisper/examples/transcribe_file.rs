use kalosm::sound::*;
use rodio::Decoder;
use rodio::Source;
use std::path::PathBuf;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    eprintln!("Starting transcription...");

    let source = if let Ok(dir) = std::env::var("RWHISPER_COHERE_DIR") {
        WhisperSource::cohere_transcribe_03_2026_local(dir)
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
    } else {
        WhisperSource::large_v3_turbo()
    };

    eprintln!("Building model...");
    let model = WhisperBuilder::default()
        .with_source(source)
        .build()
        .await?;
    eprintln!("Model built successfully");

    // Load audio from a file
    let contents = std::fs::read("./models/rwhisper/examples/samples_jfk.wav").unwrap();
    let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();
    let max_seconds = std::env::var("RWHISPER_MAX_SECONDS")
        .ok()
        .and_then(|value| value.parse::<f32>().ok());
    let rate = rodio::Source::sample_rate(&audio) as f32;

    // Transcribe the source audio into text
    eprintln!("Starting transcription...");
    let mut text = if let Some(max_seconds) = max_seconds {
        model.transcribe(audio.take_duration(Duration::from_secs_f32(max_seconds)))
    } else {
        model.transcribe(audio)
    };
    if std::env::var("RWHISPER_TIMESTAMPED").ok().as_deref() == Some("1") {
        text = text.timestamped();
    }

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
            if let Some(timestamp) = chunk.timestamp() {
                println!(
                    "{:?}\t{:.3}..{:.3}",
                    chunk.text(),
                    timestamp.start,
                    timestamp.end
                );
                // Play the audio chunk
                let (_stream, stream_handle) = rodio::OutputStream::try_default()?;
                let sink = rodio::Sink::try_new(&stream_handle).unwrap();
                let start = timestamp.start;
                let end = timestamp.end;
                let start = (start * rate) as usize;
                let end = (end * rate) as usize;
                let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();
                let audio_chunk = audio.skip(start).take(end - start).collect::<Vec<_>>();
                let audio_source = rodio::buffer::SamplesBuffer::new(1, rate as u32, audio_chunk);
                sink.append(audio_source);
                sink.sleep_until_end();
            } else {
                print!("{chunk}");
            }
        }
    }

    eprintln!("Transcription complete. Total segments: {}", segment_count);
    Ok(())
}
