use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{bail, Context, Result};
use futures_channel::mpsc;
use futures_util::{stream, StreamExt};
use kalosm::sound::*;
use kalosm_model_types::FileSource;
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

fn repo_artifact_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("artifacts")
        .join(name)
}

fn moonshine_dir(size: &str) -> Option<PathBuf> {
    let env_var = format!("RWHISPER_MOONSHINE_{}_DIR", size.to_ascii_uppercase());
    std::env::var(&env_var).ok().map(PathBuf::from).or_else(|| {
        let dir = repo_artifact_dir(&format!("moonshine-streaming-{size}"));
        dir.exists().then_some(dir)
    })
}

fn moonshine_source(size: &str) -> Option<WhisperSource> {
    let dir = moonshine_dir(size)?;
    Some(WhisperSource::moonshine_streaming_local(dir))
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock drifted backwards")
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}-{}-{timestamp}", std::process::id()))
}

fn whisper_source() -> Option<WhisperSource> {
    let explicit_model = std::env::var("RWHISPER_WHISPER_MODEL").ok();
    let explicit_tokenizer = std::env::var("RWHISPER_WHISPER_TOKENIZER").ok();
    let explicit_config = std::env::var("RWHISPER_WHISPER_CONFIG").ok();
    if let (Some(model), Some(tokenizer), Some(config)) =
        (explicit_model, explicit_tokenizer, explicit_config)
    {
        return Some(WhisperSource::new(
            FileSource::local(model.into()),
            FileSource::local(tokenizer.into()),
            FileSource::local(config.into()),
            false,
            None,
        ));
    }

    let dir = std::env::var("RWHISPER_WHISPER_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let dir = repo_artifact_dir("whisper-tiny-en");
            dir.exists().then_some(dir)
        })?;

    Some(WhisperSource::new(
        FileSource::local(dir.join("whisper-tiny-en.gguf")),
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
    ))
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
#[ignore = "requires a local Moonshine artifact directory"]
async fn moonshine_tiny_matches_the_jfk_reference() -> Result<()> {
    let Some(source) = moonshine_source("tiny") else {
        bail!("missing Moonshine tiny artifact dir; set RWHISPER_MOONSHINE_TINY_DIR or place files in artifacts/moonshine-streaming-tiny");
    };

    let transcript = transcribe_jfk_sample(source).await?;
    assert_eq!(
        normalize_text(&transcript),
        normalize_text(JFK_REFERENCE),
        "unexpected Moonshine transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires a local Moonshine artifact directory"]
async fn moonshine_small_matches_the_jfk_reference() -> Result<()> {
    let Some(source) = moonshine_source("small") else {
        bail!("missing Moonshine small artifact dir; set RWHISPER_MOONSHINE_SMALL_DIR or place files in artifacts/moonshine-streaming-small");
    };

    let transcript = transcribe_jfk_sample(source).await?;
    assert_eq!(
        normalize_text(&transcript),
        normalize_text(JFK_REFERENCE),
        "unexpected Moonshine transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires a local Moonshine artifact directory"]
async fn moonshine_medium_matches_the_jfk_reference() -> Result<()> {
    let Some(source) = moonshine_source("medium") else {
        bail!("missing Moonshine medium artifact dir; set RWHISPER_MOONSHINE_MEDIUM_DIR or place files in artifacts/moonshine-streaming-medium");
    };

    let transcript = transcribe_jfk_sample(source).await?;
    assert_eq!(
        normalize_text(&transcript),
        normalize_text(JFK_REFERENCE),
        "unexpected Moonshine transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires a local Moonshine tiny artifact directory with embedded GGUF metadata"]
async fn moonshine_tiny_loads_from_embedded_metadata_without_sidecars() -> Result<()> {
    let Some(source_dir) = moonshine_dir("tiny") else {
        bail!("missing Moonshine tiny artifact dir; set RWHISPER_MOONSHINE_TINY_DIR or place files in artifacts/moonshine-streaming-tiny");
    };

    let temp_dir = unique_temp_dir("rwhisper-moonshine-embedded");
    fs::create_dir_all(&temp_dir)?;
    fs::copy(source_dir.join("model.gguf"), temp_dir.join("model.gguf"))
        .context("failed to copy GGUF into temp directory")?;
    assert!(!temp_dir.join("tokenizer.json").exists());
    assert!(!temp_dir.join("config.json").exists());

    let _model = WhisperBuilder::default()
        .with_source(WhisperSource::moonshine_streaming_local(&temp_dir))
        .build()
        .await
        .context("failed to load Moonshine from embedded GGUF metadata only")?;

    let _ = fs::remove_dir_all(&temp_dir);
    Ok(())
}

#[tokio::test]
#[ignore = "requires a local Moonshine artifact directory"]
async fn moonshine_tiny_streaming_matches_the_jfk_reference() -> Result<()> {
    let Some(source) = moonshine_source("tiny") else {
        bail!("missing Moonshine tiny artifact dir; set RWHISPER_MOONSHINE_TINY_DIR or place files in artifacts/moonshine-streaming-tiny");
    };

    let transcript = transcribe_jfk_sample_streaming(source, 250, None).await?;
    assert_eq!(
        normalize_text(&transcript),
        normalize_text(JFK_REFERENCE),
        "unexpected Moonshine streaming transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires a local Moonshine artifact directory"]
async fn moonshine_tiny_streaming_short_prefix_smoke() -> Result<()> {
    let Some(source) = moonshine_source("tiny") else {
        bail!("missing Moonshine tiny artifact dir; set RWHISPER_MOONSHINE_TINY_DIR or place files in artifacts/moonshine-streaming-tiny");
    };

    let transcript = transcribe_jfk_sample_streaming(source, 1_000, Some(2.0)).await?;
    assert_eq!(
        normalize_text(&transcript),
        "and so my fellow america",
        "unexpected short Moonshine streaming transcript: {transcript}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires a local Moonshine artifact directory"]
async fn moonshine_tiny_streaming_emits_before_the_clip_ends() -> Result<()> {
    let Some(source) = moonshine_source("tiny") else {
        bail!("missing Moonshine tiny artifact dir; set RWHISPER_MOONSHINE_TINY_DIR or place files in artifacts/moonshine-streaming-tiny");
    };

    let first_chunk = first_streaming_emission_chunk(source, 250).await?;
    let Some(first_chunk) = first_chunk else {
        bail!("streaming Moonshine never emitted a non-empty segment");
    };
    assert!(
        first_chunk <= 8,
        "expected streaming Moonshine to emit within the first 2 seconds; first emission chunk index was {first_chunk}"
    );
    Ok(())
}

#[tokio::test]
#[ignore = "requires local Moonshine and Whisper artifact directories"]
async fn moonshine_tiny_beats_whisper_tiny_on_the_jfk_sample() -> Result<()> {
    let Some(moonshine_source) = moonshine_source("tiny") else {
        bail!("missing Moonshine tiny artifact dir; set RWHISPER_MOONSHINE_TINY_DIR or place files in artifacts/moonshine-streaming-tiny");
    };
    let Some(whisper_source) = whisper_source() else {
        bail!("missing Whisper artifact paths; set RWHISPER_WHISPER_MODEL/RWHISPER_WHISPER_TOKENIZER/RWHISPER_WHISPER_CONFIG or RWHISPER_WHISPER_DIR");
    };

    let moonshine = transcribe_jfk_sample(moonshine_source).await?;
    let whisper = transcribe_jfk_sample(whisper_source).await?;
    let moonshine_wer = word_error_rate(JFK_REFERENCE, &moonshine);
    let whisper_wer = word_error_rate(JFK_REFERENCE, &whisper);

    assert!(
        moonshine_wer < whisper_wer,
        "expected Moonshine to be closer to the JFK reference on this sample\nmoonshine: {moonshine:?} (wer={moonshine_wer:.3})\nwhisper: {whisper:?} (wer={whisper_wer:.3})"
    );
    Ok(())
}
