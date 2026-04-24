use std::{fs, io::Cursor, path::Path};

use anyhow::{Context, Result};
use futures_channel::mpsc;
use futures_util::{stream, StreamExt};
use kalosm::sound::*;
use rodio::Decoder;
use rwhisper::TranscribeChunkedAudioStreamExt;

// Reference text from JFK's January 20, 1961 inaugural address:
// "And so, my fellow Americans, ask not what your country can do for you,
// ask what you can do for your country."
const JFK_REFERENCE: &str =
    "And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.";

fn normalize_text(text: &str) -> String {
    let filtered = text
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c.is_ascii_whitespace() {
                c.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>();
    filtered.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let reference = normalize_text(reference);
    let hypothesis = normalize_text(hypothesis);
    let reference = reference.split_whitespace().collect::<Vec<_>>();
    let hypothesis = hypothesis.split_whitespace().collect::<Vec<_>>();

    let mut dp = vec![vec![0usize; hypothesis.len() + 1]; reference.len() + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }

    for i in 1..=reference.len() {
        for j in 1..=hypothesis.len() {
            let substitution_cost = usize::from(reference[i - 1] != hypothesis[j - 1]);
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + substitution_cost);
        }
    }

    if reference.is_empty() {
        0.0
    } else {
        dp[reference.len()][hypothesis.len()] as f32 / reference.len() as f32
    }
}

fn moonshine_source(size: &str) -> WhisperSource {
    let local = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts")
        .join(format!("moonshine-streaming-{size}.gguf"));
    if local.exists() {
        return WhisperSource::moonshine_streaming_local(local);
    }
    match size {
        "tiny" => WhisperSource::moonshine_streaming_tiny(),
        "small" => WhisperSource::moonshine_streaming_small(),
        "medium" => WhisperSource::moonshine_streaming_medium(),
        other => panic!("unknown Moonshine size: {other}"),
    }
}

async fn transcribe_jfk_sample(source: WhisperSource) -> Result<String> {
    let model = WhisperBuilder::default()
        .with_source(source)
        .build()
        .await
        .context("failed to build model")?;

    let contents = fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/samples_jfk.wav"))
        .context("failed to read JFK sample")?;
    let audio = Decoder::new(Cursor::new(contents)).context("failed to decode JFK sample")?;

    let mut stream = model.transcribe(audio);
    let mut transcript = String::new();
    while let Some(segment) = stream.next().await {
        if !transcript.is_empty() {
            transcript.push(' ');
        }
        transcript.push_str(segment.text().trim());
    }

    Ok(transcript)
}

async fn transcribe_jfk_sample_streaming(
    source: WhisperSource,
    chunk_ms: usize,
    max_seconds: Option<f32>,
) -> Result<String> {
    let model = WhisperBuilder::default()
        .with_source(source)
        .build()
        .await
        .context("failed to build model")?;

    let contents = fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/samples_jfk.wav"))
        .context("failed to read JFK sample")?;
    let audio = Decoder::new(Cursor::new(contents)).context("failed to decode JFK sample")?;
    let sample_rate = rodio::Source::sample_rate(&audio);
    let channels = rodio::Source::channels(&audio);
    let samples: Vec<i16> = if let Some(max_seconds) = max_seconds {
        rodio::Source::take_duration(audio, std::time::Duration::from_secs_f32(max_seconds))
            .collect()
    } else {
        audio.collect()
    };
    let samples_per_chunk = ((sample_rate as usize * chunk_ms) / 1000).max(1) * channels as usize;
    let chunks = samples
        .chunks(samples_per_chunk)
        .map(|chunk| rodio::buffer::SamplesBuffer::new(channels, sample_rate, chunk.to_vec()))
        .collect::<Vec<_>>();

    let mut stream = stream::iter(chunks).transcribe(model);
    let mut transcript = String::new();
    while let Some(segment) = stream.next().await {
        if !transcript.is_empty() {
            transcript.push(' ');
        }
        transcript.push_str(segment.text().trim());
    }

    Ok(transcript)
}

async fn first_streaming_emission_chunk(
    source: WhisperSource,
    chunk_ms: usize,
) -> Result<Option<usize>> {
    let model = WhisperBuilder::default()
        .with_source(source)
        .build()
        .await
        .context("failed to build model")?;

    let contents = fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/samples_jfk.wav"))
        .context("failed to read JFK sample")?;
    let audio = Decoder::new(Cursor::new(contents)).context("failed to decode JFK sample")?;
    let sample_rate = rodio::Source::sample_rate(&audio);
    let channels = rodio::Source::channels(&audio);
    let samples: Vec<i16> = audio.collect();
    let samples_per_chunk = ((sample_rate as usize * chunk_ms) / 1000).max(1) * channels as usize;
    let chunks = samples
        .chunks(samples_per_chunk)
        .map(|chunk| rodio::buffer::SamplesBuffer::new(channels, sample_rate, chunk.to_vec()))
        .collect::<Vec<_>>();
    let chunk_count = chunks.len();

    let (tx, rx) = mpsc::unbounded();
    let mut stream = rx.transcribe(model);

    for (index, chunk) in chunks.into_iter().enumerate() {
        tx.unbounded_send(chunk)
            .map_err(|_| anyhow::anyhow!("failed to send chunk to transcription stream"))?;
        if let Ok(Some(segment)) =
            tokio::time::timeout(std::time::Duration::from_millis(250), stream.next()).await
        {
            if !segment.text().trim().is_empty() {
                return Ok(Some(index));
            }
        }
    }

    drop(tx);
    while let Some(segment) = stream.next().await {
        if !segment.text().trim().is_empty() {
            return Ok(Some(chunk_count));
        }
    }

    Ok(None)
}

#[test]
fn normalize_text_keeps_the_reference_stable() {
    assert_eq!(
        normalize_text(JFK_REFERENCE),
        "and so my fellow americans ask not what your country can do for you ask what you can do for your country"
    );
}

#[test]
fn word_error_rate_is_zero_for_exact_match() {
    assert_eq!(word_error_rate(JFK_REFERENCE, JFK_REFERENCE), 0.0);
}

#[tokio::test]
async fn moonshine_tiny_matches_the_jfk_reference() -> Result<()> {
    let transcript = transcribe_jfk_sample(moonshine_source("tiny")).await?;
    assert_eq!(
        normalize_text(&transcript),
        normalize_text(JFK_REFERENCE),
        "unexpected Moonshine transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
async fn moonshine_small_matches_the_jfk_reference() -> Result<()> {
    let transcript = transcribe_jfk_sample(moonshine_source("small")).await?;
    assert_eq!(
        normalize_text(&transcript),
        normalize_text(JFK_REFERENCE),
        "unexpected Moonshine transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
async fn moonshine_medium_matches_the_jfk_reference() -> Result<()> {
    let transcript = transcribe_jfk_sample(moonshine_source("medium")).await?;
    assert_eq!(
        normalize_text(&transcript),
        normalize_text(JFK_REFERENCE),
        "unexpected Moonshine transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
async fn moonshine_tiny_streaming_matches_the_jfk_reference() -> Result<()> {
    let transcript = transcribe_jfk_sample_streaming(moonshine_source("tiny"), 250, None).await?;
    assert_eq!(
        normalize_text(&transcript),
        normalize_text(JFK_REFERENCE),
        "unexpected Moonshine streaming transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
async fn moonshine_tiny_streaming_short_prefix_smoke() -> Result<()> {
    let transcript =
        transcribe_jfk_sample_streaming(moonshine_source("tiny"), 1_000, Some(2.0)).await?;
    assert_eq!(
        normalize_text(&transcript),
        "and so my fellow americans",
        "unexpected short Moonshine streaming transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
async fn moonshine_tiny_streaming_emits_before_the_clip_ends() -> Result<()> {
    let first_chunk = first_streaming_emission_chunk(moonshine_source("tiny"), 250).await?;
    let first_chunk =
        first_chunk.context("streaming Moonshine never emitted a non-empty segment")?;
    assert!(
        first_chunk <= 8,
        "expected streaming Moonshine to emit within the first 2 seconds; first emission chunk index was {first_chunk}"
    );
    Ok(())
}

#[tokio::test]
async fn cohere_transcribe_matches_the_jfk_reference() -> Result<()> {
    let transcript = transcribe_jfk_sample(WhisperSource::cohere_transcribe_03_2026()).await?;
    assert_eq!(
        normalize_text(&transcript),
        normalize_text(JFK_REFERENCE),
        "unexpected Cohere transcript: {transcript}"
    );
    Ok(())
}
