use futures_util::{stream, Stream};
use kalosm::sound::*;
use rodio::Decoder;
use rodio::Source;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    eprintln!("Starting transcription...");

    let source = if let Ok(path) = std::env::var("RWHISPER_COHERE_GGUF") {
        WhisperSource::cohere_transcribe_03_2026_local(path)
    } else if let Ok(_) = std::env::var("RWHISPER_MOONSHINE") {
        WhisperSource::moonshine_streaming_tiny()
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
    let max_seconds = std::env::var("RWHISPER_MAX_SECONDS")
        .ok()
        .and_then(|value| value.parse::<f32>().ok());
    let streaming_chunk_ms = std::env::var("RWHISPER_STREAMING_CHUNK_MS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok());
    let timestamped = std::env::var("RWHISPER_TIMESTAMPED").ok().as_deref() == Some("1");

    let audio = Decoder::new(std::io::Cursor::new(contents.clone())).unwrap();
    let rate = rodio::Source::sample_rate(&audio) as f32;

    // Transcribe the source audio into text
    eprintln!("Starting transcription...");
    let mut text: Pin<Box<dyn Stream<Item = Segment>>> = if let Some(chunk_ms) = streaming_chunk_ms
    {
        let channels = rodio::Source::channels(&audio);
        let sample_rate = rodio::Source::sample_rate(&audio);
        let samples = if let Some(max_seconds) = max_seconds {
            audio
                .take_duration(Duration::from_secs_f32(max_seconds))
                .collect::<Vec<_>>()
        } else {
            audio.collect::<Vec<_>>()
        };
        let samples_per_chunk =
            ((sample_rate as usize * chunk_ms) / 1000).max(1) * channels as usize;
        let chunks = samples
            .chunks(samples_per_chunk)
            .map(|chunk| rodio::buffer::SamplesBuffer::new(channels, sample_rate, chunk.to_vec()))
            .collect::<Vec<_>>();
        let mut task = stream::iter(chunks).transcribe(model);
        if timestamped {
            task = task.timestamped();
        }
        Box::pin(task)
    } else if let Some(max_seconds) = max_seconds {
        let mut task = model.transcribe(audio.take_duration(Duration::from_secs_f32(max_seconds)));
        if timestamped {
            task = task.timestamped();
        }
        Box::pin(task)
    } else {
        let mut task = model.transcribe(audio);
        if timestamped {
            task = task.timestamped();
        }
        Box::pin(task)
    };

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
