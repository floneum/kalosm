//! Handles chunking audio with a voice audio detection model
use std::{
    pin::Pin,
    sync::{Arc, RwLock},
    task::{Context, Poll},
};

use futures_core::{ready, Stream};
use futures_util::FutureExt;
use rodio::buffer::SamplesBuffer;
use voice_activity_detector::VoiceActivityDetector;

use crate::{AsyncSource, ResampledAsyncSource, VoiceActivityDetectorOutput};

/// An extension trait for audio streams that adds a voice activity detection information. Based on the [voice_activity_detector](https://github.com/nkeenan38/voice_activity_detector) crate.
pub trait VoiceActivityDetectorExt: AsyncSource {
    /// Transform the audio stream to a stream of [`SamplesBuffer`]s with voice activity detection information
    fn voice_activity_stream(self) -> VoiceActivityDetectorStream<Self>
    where
        Self: Sized + Unpin,
    {
        let (source, closest) = resample_to_nearest_supported_rate(self);
        let vad = closest.vad();

        VoiceActivityDetectorStream::new(source, vad, closest.chunk_sizes[0])
    }
}

impl<S: AsyncSource> VoiceActivityDetectorExt for S {}

/// A stream of [`SamplesBuffer`]s with voice activity detection information
pub struct VoiceActivityDetectorStream<S: AsyncSource + Unpin> {
    source: ResampledAsyncSource<S>,
    buffer: Vec<f32>,
    chunk_size: usize,
    vad: Arc<RwLock<VoiceActivityDetector>>,
    task: Option<tokio::task::JoinHandle<VoiceActivityDetectorOutput>>,
}

impl<S: AsyncSource + Unpin> VoiceActivityDetectorStream<S> {
    fn new(source: ResampledAsyncSource<S>, vad: VoiceActivityDetector, chunk_size: usize) -> Self {
        Self {
            source,
            buffer: Vec::with_capacity(chunk_size),
            chunk_size,
            vad: Arc::new(RwLock::new(vad)),
            task: None,
        }
    }
}

impl<S: AsyncSource + Unpin> Stream for VoiceActivityDetectorStream<S> {
    type Item = VoiceActivityDetectorOutput;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let sample_rate = this.source.sample_rate();

        loop {
            if let Some(task) = &mut this.task {
                let output = ready!(task.poll_unpin(cx));
                this.task = None;
                match output {
                    Ok(output) => return Poll::Ready(Some(output)),
                    Err(err) => tracing::error!("Error in voice activity detector: {err}"),
                }
            }

            let stream = this.source.as_stream();
            let mut stream = std::pin::pin!(stream);
            while this.buffer.len() < this.chunk_size {
                let sample = ready!(stream.as_mut().poll_next(cx));
                if let Some(sample) = sample {
                    this.buffer.push(sample);
                } else {
                    return Poll::Ready(None);
                }
            }
            let data = std::mem::take(&mut this.buffer);
            let model = this.vad.clone();
            let vad = tokio::task::spawn_blocking(move || {
                let mut locked = model.write().unwrap();
                let vad = locked.predict(data.iter().copied());
                VoiceActivityDetectorOutput {
                    probability: vad,
                    samples: SamplesBuffer::new(1, sample_rate, data),
                }
            });
            this.task = Some(vad);
        }
    }
}

/// Resample the audio to the closest supported sample rate
fn resample_to_nearest_supported_rate<S: AsyncSource + Unpin>(
    source: S,
) -> (ResampledAsyncSource<S>, SupportedSampleRate) {
    let sample_rate = source.sample_rate();
    let closet = SupportedSampleRate::closest(sample_rate);
    let resampled = source.resample(closet.sample_rate);
    (resampled, closet)
}

#[derive(Clone, Copy)]
struct SupportedSampleRate {
    sample_rate: u32,
    chunk_sizes: [usize; 3],
}

impl SupportedSampleRate {
    fn closest(sample_rate: u32) -> Self {
        *SUPPORTED_SAMPLE_RATES
            .iter()
            .min_by_key(|sr| (sr.sample_rate as i64 - sample_rate as i64).abs())
            .unwrap()
    }

    fn vad(&self) -> VoiceActivityDetector {
        VoiceActivityDetector::builder()
            .sample_rate(self.sample_rate)
            .chunk_size(self.chunk_sizes[0])
            .build()
            .unwrap()
    }
}

const SUPPORTED_SAMPLE_RATES: [SupportedSampleRate; 2] = [
    SupportedSampleRate {
        sample_rate: 8000,
        chunk_sizes: [256, 512, 768],
    },
    SupportedSampleRate {
        sample_rate: 16000,
        chunk_sizes: [512, 768, 1024],
    },
];
