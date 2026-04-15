//! Handles denoising audio streams
use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use futures_core::{ready, Stream};
use futures_util::FutureExt;
use rodio::buffer::SamplesBuffer;

use crate::{AsyncSource, ResampledAsyncSource, VoiceActivityDetectorOutput};

const SAMPLE_RATE: u32 = 48_000;
const SCALE_FACTOR: f32 = i16::MAX as f32;
const FRAME_SIZE: usize = nnnoiseless::FRAME_SIZE;

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

mod fusor_impl {
    use super::*;
    use nnnoiseless::{
        DenoiseFeatures, FusorRnnoise, FusorRnnoiseState, FREQ_SIZE, NB_BANDS, NB_FEATURES,
    };
    use std::sync::Arc;

    type InitFuture = Pin<Box<dyn Future<Output = fusor::Result<FusorRnnoise>>>>;

    /// A stream of [`SamplesBuffer`]s with voice activity detection information.
    ///
    /// This version uses GPU acceleration via fusor for the RNN inference,
    /// while keeping feature extraction on the CPU.
    pub struct DenoisedStream<S: AsyncSource + Unpin> {
        source: ResampledAsyncSource<S>,
        /// CPU-side feature extraction
        features: Box<DenoiseFeatures>,
        /// GPU-side RNN model (lazily initialized)
        rnn: Option<Arc<FusorRnnoise>>,
        /// RNN hidden state (lazily initialized)
        rnn_state: Option<FusorRnnoiseState>,
        /// Future for initializing the RNN model
        init_future: Option<InitFuture>,
        /// Previous gains for smoothing
        lastg: [f32; NB_BANDS],
        /// Buffer fill index
        fill_index: usize,
        /// Input sample buffer
        input_buffer: [f32; FRAME_SIZE],
        /// Output sample buffer
        output: [f32; FRAME_SIZE],
    }

    impl<S: AsyncSource + Unpin> DenoisedStream<S> {
        pub(super) fn new(source: S) -> Self {
            Self {
                source: source.resample(SAMPLE_RATE),
                features: Box::new(DenoiseFeatures::new()),
                rnn: None,
                rnn_state: None,
                init_future: None,
                lastg: [0.0; NB_BANDS],
                fill_index: 0,
                input_buffer: [0f32; FRAME_SIZE],
                output: [0f32; FRAME_SIZE],
            }
        }
    }

    impl<S: AsyncSource + Unpin> Stream for DenoisedStream<S> {
        type Item = VoiceActivityDetectorOutput;

        fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            let this = self.get_mut();

            // Lazily initialize the RNN model before yielding any items
            if this.rnn.is_none() {
                // Start the initialization future if not already started
                if this.init_future.is_none() {
                    this.init_future = Some(Box::pin(FusorRnnoise::new()));
                }

                // Poll the initialization future
                if let Some(fut) = &mut this.init_future {
                    match fut.poll_unpin(cx) {
                        Poll::Ready(Ok(rnn)) => {
                            let rnn_state = rnn.new_state();
                            this.rnn = Some(Arc::new(rnn));
                            this.rnn_state = Some(rnn_state);
                            this.init_future = None;
                        }
                        Poll::Ready(Err(err)) => {
                            tracing::error!("Failed to initialize RNNoise model: {err}");
                            this.init_future = None;
                            return Poll::Ready(None);
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
            }

            let rnn = this.rnn.as_ref().expect("RNN should be initialized");
            let rnn_state = this
                .rnn_state
                .as_mut()
                .expect("RNN state should be initialized");

            let sample_rate = this.source.sample_rate();
            let stream = this.source.as_stream();
            let mut stream = std::pin::pin!(stream);

            // Fill the input buffer
            while this.fill_index < FRAME_SIZE {
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

            // Process the frame
            // Step 1: CPU - shift and filter input
            this.features.shift_and_filter_input(&this.input_buffer);

            // Step 2: CPU - compute frame features
            let silence = this.features.compute_frame_features();

            let vad_prob = if !silence {
                // Step 3: GPU - run RNN inference
                let features_slice = this.features.features();
                let mut features_array = [0.0f32; NB_FEATURES];
                features_array.copy_from_slice(features_slice);

                let (gains, vad) = rnn.forward_sync(&features_array, rnn_state);

                // Convert gains to array
                let mut g = [0.0f32; NB_BANDS];
                for (i, &gain) in gains.iter().enumerate().take(NB_BANDS) {
                    g[i] = gain;
                }

                // Step 4: CPU - pitch filter
                this.features.pitch_filter(&g);

                // Step 5: Apply gain smoothing
                for i in 0..NB_BANDS {
                    g[i] = g[i].max(0.6 * this.lastg[i]);
                    this.lastg[i] = g[i];
                }

                // Step 6: Interpolate band gains to frequency bins
                let mut gf = [1.0f32; FREQ_SIZE];
                nnnoiseless::interp_band_gain(&mut gf, &g);

                // Step 7: CPU - apply gain
                this.features.apply_gain(&gf);

                vad
            } else {
                0.0
            };

            // Step 8: CPU - frame synthesis
            this.features.frame_synthesis(&mut this.output);

            // Rescale the output
            for output in &mut this.output {
                *output /= SCALE_FACTOR;
            }

            let samples = SamplesBuffer::new(1, sample_rate, this.output);
            Poll::Ready(Some(VoiceActivityDetectorOutput {
                probability: vad_prob,
                samples,
            }))
        }
    }
}

pub use fusor_impl::DenoisedStream;
