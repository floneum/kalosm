use dasp::interpolate::Interpolator as _;
use futures_core::{Future, Stream};
use futures_util::StreamExt;
use rodio::buffer::SamplesBuffer;
use std::time::Duration;

#[cfg(feature = "source")]
mod mic;
#[cfg(feature = "source")]
pub use mic::*;

/// A streaming audio source for single channel audio. This trait is implemented for all types that implement `rodio::Source` automatically.
pub trait AsyncSource {
    /// Get the stream of the source
    fn as_stream(&mut self) -> impl Stream<Item = f32> + '_;

    /// Get the sample rate of the stream
    fn sample_rate(&self) -> u32;

    /// Read the next n samples from the stream or until the stream is exhausted
    fn read_samples(&mut self, samples: usize) -> impl Future<Output = SamplesBuffer<f32>> {
        async move {
            let channels = 1;
            let sample_rate = self.sample_rate();
            let stream = self.as_stream();
            let mut stream = std::pin::pin!(stream);
            let mut buffer = Vec::with_capacity(samples);
            for _ in 0..samples {
                match stream.next().await {
                    Some(data) => buffer.push(data),
                    None => break,
                }
            }

            SamplesBuffer::new(channels, sample_rate, buffer)
        }
    }

    /// Read the next duration of samples from the stream
    fn read_duration(&mut self, duration: Duration) -> impl Future<Output = SamplesBuffer<f32>> {
        let samples = duration.as_secs_f32() * self.sample_rate() as f32;
        self.read_samples(samples as usize)
    }

    /// Resample the stream to the given sample rate
    fn resample(self, sample_rate: u32) -> ResampledAsyncSource<Self>
    where
        Self: Sized + Unpin,
    {
        ResampledAsyncSource::new(self, sample_rate)
    }
}

impl<S: rodio::Source> AsyncSource for S
where
    <S as std::iter::Iterator>::Item: rodio::Sample + dasp::sample::ToSample<f32>,
{
    fn as_stream(&mut self) -> impl Stream<Item = f32> + '_ {
        futures_util::stream::iter(
            self.step_by(self.channels() as usize)
                .map(dasp::Sample::to_sample),
        )
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate()
    }
}

/// An async audio source from a stream of samples.
pub struct AsyncSourceFromStream<I: Stream<Item = f32>> {
    stream: I,
    sample_rate: u32,
}

impl<I: Stream<Item = f32>> AsyncSourceFromStream<I> {
    /// Create a new async audio source from a stream of samples.
    pub fn new(stream: I, sample_rate: u32) -> Self {
        Self {
            stream,
            sample_rate,
        }
    }
}

impl<I: Stream<Item = f32> + std::marker::Unpin> AsyncSource for AsyncSourceFromStream<I> {
    fn as_stream(&mut self) -> impl Stream<Item = f32> + '_ {
        &mut self.stream
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// A resampled async audio source
pub struct ResampledAsyncSource<S: AsyncSource> {
    source: S,
    source_output_sample_ratio: f64,
    sample_position: f64,
    sample_rate: u32,
    resampler: dasp::interpolate::linear::Linear<f32>,
}

impl<S: AsyncSource> ResampledAsyncSource<S> {
    fn new(source: S, sample_rate: u32) -> Self {
        let source_output_sample_ratio = source.sample_rate() as f64 / sample_rate as f64;
        Self {
            source,
            source_output_sample_ratio,
            sample_position: source_output_sample_ratio,
            sample_rate,
            resampler: dasp::interpolate::linear::Linear::new(0.0, 0.0),
        }
    }
}

impl<S: AsyncSource + Unpin> Stream for ResampledAsyncSource<S> {
    type Item = f32;

    fn poll_next(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let myself = self.get_mut();
        let source = myself.source.as_stream();
        let mut source = std::pin::pin!(source);
        let source_output_sample_ratio = myself.source_output_sample_ratio;

        while myself.sample_position >= 1.0 {
            myself.sample_position -= 1.0;
            myself
                .resampler
                .next_source_frame(match source.as_mut().poll_next(cx) {
                    std::task::Poll::Ready(Some(frame)) => frame,
                    std::task::Poll::Ready(None) => return std::task::Poll::Ready(None),
                    std::task::Poll::Pending => return std::task::Poll::Pending,
                })
        }

        // Get the interpolated value
        let interpolated = myself.resampler.interpolate(myself.sample_position);

        // Advance the sample position
        myself.sample_position += source_output_sample_ratio;

        std::task::Poll::Ready(Some(interpolated))
    }
}

impl<S: AsyncSource + Unpin> AsyncSource for ResampledAsyncSource<S> {
    fn as_stream(&mut self) -> impl Stream<Item = f32> + '_ {
        self
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
