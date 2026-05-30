//! Handles denoising audio streams
use std::{
    pin::Pin,
    task::{Context, Poll},
};

use futures_core::{ready, Stream};
use nnnoiseless::DenoiseState;
use rodio::buffer::SamplesBuffer;

use crate::{AsyncSource, ResampledAsyncSource, VoiceActivityDetectorOutput};

/// An extension trait for audio streams for denoising. Based on the [nnnoiseless](https://github.com/rust-dsp/nnnoiseless) crate.
pub trait DenoisedExt: AsyncSource {
    /// Transform the audio stream to a stream of [`SamplesBuffer`]s that have been denoised
    ///
    /// NOTE: The detection in [`crate::VoiceActivityDetectorExt::voice_activity_stream`] tends to be more consistent than this method.
    fn denoise_and_detect_voice_activity(self) -> DenoisedStream<Self>
    where
        Self: Sized + Unpin,
    {
        DenoisedStream::new(self)
    }
}

impl<S: AsyncSource> DenoisedExt for S {}

const SAMPLE_RATE: u32 = 48_000;
const SCALE_FACTOR: f32 = i16::MAX as f32;

/// A stream of [`SamplesBuffer`]s with voice activity detection information
pub struct DenoisedStream<S: AsyncSource + Unpin> {
    source: ResampledAsyncSource<S>,
    denoiser: Box<nnnoiseless::DenoiseState<'static>>,
    fill_index: usize,
    input_buffer: [f32; DenoiseState::FRAME_SIZE],
    output: [f32; DenoiseState::FRAME_SIZE],
}

impl<S: AsyncSource + Unpin> DenoisedStream<S> {
    fn new(source: S) -> Self {
        Self {
            source: source.resample(SAMPLE_RATE),
            denoiser: DenoiseState::new(),
            fill_index: 0,
            input_buffer: [0f32; DenoiseState::FRAME_SIZE],
            output: [0f32; DenoiseState::FRAME_SIZE],
        }
    }
}

impl<S: AsyncSource + Unpin> Stream for DenoisedStream<S> {
    type Item = VoiceActivityDetectorOutput;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let sample_rate = this.source.sample_rate();
        let stream = this.source.as_stream();
        let mut stream = std::pin::pin!(stream);
        // Fill the input buffer
        while this.fill_index < DenoiseState::FRAME_SIZE {
            let sample = ready!(stream.as_mut().poll_next(cx));
            if let Some(sample) = sample {
                let scaled = sample * SCALE_FACTOR;
                this.input_buffer[this.fill_index] = scaled;
                this.fill_index += 1;
            } else {
                return Poll::Ready(None);
            }
        }

        this.fill_index = 0;

        // Once we have enough samples, denoise the buffer and copy the output to the output buffer
        let vad = this
            .denoiser
            .process_frame(&mut this.output, &this.input_buffer);
        // Rescale the output
        for output in &mut this.output {
            *output /= SCALE_FACTOR;
        }
        let samples = SamplesBuffer::new(1, sample_rate, this.output);
        Poll::Ready(Some(VoiceActivityDetectorOutput {
            probability: vad,
            samples,
        }))
    }
}
