// Audio processing code, adapted from whisper.cpp
// https://github.com/ggerganov/whisper.cpp
// In turn adapted from candle:
// https://github.com/huggingface/candle/blob/7669ed1eb37a0ca6837757ad0adc79639a424bed/candle-transformers/src/models/whisper/audio.rs

use std::sync::Arc;
use std::thread;

use rustfft::{num_complex::Complex, FftPlanner};

use crate::config::{Config, CHUNK_LENGTH, HOP_LENGTH, N_FFT};

fn get_num_threads() -> usize {
    if cfg!(target_arch = "wasm32") {
        1
    } else {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

fn fft(input: &mut [Complex<f32>], fft_out: &mut Vec<f32>) {
    let mut fft = FftPlanner::new();
    let fft = fft.plan_fft_forward(input.len());
    fft.process(input);
    fft_out.clear();
    for c in input.iter() {
        fft_out.push(c.re);
        fft_out.push(c.im);
    }
}

#[allow(clippy::too_many_arguments)]
// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2414
fn log_mel_spectrogram_w(
    ith: usize,
    hann: &[f32],
    samples: &[f32],
    filters: &[f32],
    fft_size: usize,
    fft_step: usize,
    n_len: usize,
    n_mel: usize,
    n_threads: usize,
) -> Vec<f32> {
    let n_fft = 1 + fft_size / 2;

    let mut fft_in = vec![Complex::ZERO; fft_size];
    let mut mel = vec![0.0; n_len * n_mel];
    let n_samples = samples.len();
    let end = std::cmp::min(n_samples / fft_step + 1, n_len);
    let mut fft_out = Vec::new();

    for i in (ith..end).step_by(n_threads) {
        let offset = i * fft_step;

        // apply Hanning window
        for j in 0..std::cmp::min(fft_size, n_samples - offset) {
            fft_in[j] = Complex::new(hann[j] * samples[offset + j], 0.0);
        }

        // fill the rest with zeros
        if n_samples - offset < fft_size {
            fft_in[n_samples - offset..].fill(Complex::ZERO);
        }

        // FFT
        fft(&mut fft_in, &mut fft_out);

        // Calculate modulus^2 of complex numbers
        for j in 0..fft_size {
            fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }
        for j in 1..fft_size / 2 {
            let v = fft_out[fft_size - j];
            fft_out[j] += v;
        }

        // mel spectrogram
        for j in 0..n_mel {
            let mut sum = 0.0;
            let mut k = 0;
            // Unroll loop
            while k < n_fft.saturating_sub(3) {
                sum += fft_out[k] * filters[j * n_fft + k]
                    + fft_out[k + 1] * filters[j * n_fft + k + 1]
                    + fft_out[k + 2] * filters[j * n_fft + k + 2]
                    + fft_out[k + 3] * filters[j * n_fft + k + 3];
                k += 4;
            }
            // Handle remainder
            while k < n_fft {
                sum += fft_out[k] * filters[j * n_fft + k];
                k += 1;
            }
            mel[j * n_len + i] = f32::max(sum, 1e-10).log10();
        }
    }
    mel
}

pub fn log_mel_spectrogram_(
    samples: &[f32],
    filters: &[f32],
    fft_size: usize,
    fft_step: usize,
    n_mel: usize,
) -> Vec<f32> {
    let fft_size_t = fft_size as f32;

    let hann: Vec<f32> = (0..fft_size)
        .map(|i| (1.0 - ((std::f32::consts::TAU * i as f32) / fft_size_t).cos()) / 2.0)
        .collect();
    let n_len = samples.len() / fft_step;

    // pad audio with at least one extra chunk of zeros
    let pad = 100 * CHUNK_LENGTH / 2;
    let n_len = if !n_len.is_multiple_of(pad) {
        (n_len / pad + 1) * pad
    } else {
        n_len
    };
    let n_len = n_len + pad;
    let samples = {
        let mut samples_padded = samples.to_vec();
        let to_add = n_len * fft_step - samples.len();
        samples_padded.extend(std::iter::repeat_n(0.0, to_add));
        samples_padded
    };

    // ensure that the number of threads is even and less than 12
    let n_threads = std::cmp::min(get_num_threads() - get_num_threads() % 2, 12);
    let n_threads = std::cmp::max(n_threads, 2);

    let hann = Arc::new(hann);
    let samples = Arc::new(samples);
    let filters = Arc::new(filters);

    // use scope to allow for non static references to be passed to the threads
    // and directly collect the results into a single vector
    let all_outputs = if cfg!(target_arch = "wasm32") {
        (0..n_threads)
            .map(|thread_id| {
                log_mel_spectrogram_w(
                    thread_id, &hann, &samples, &filters, fft_size, fft_step, n_len, n_mel,
                    n_threads,
                )
            })
            .collect::<Vec<_>>()
    } else {
        thread::scope(|s| {
            (0..n_threads)
                // create threads and return their handles
                .map(|thread_id| {
                    let hann = Arc::clone(&hann);
                    let samples = Arc::clone(&samples);
                    let filters = Arc::clone(&filters);
                    // spawn new thread and start work
                    s.spawn(move || {
                        log_mel_spectrogram_w(
                            thread_id, &hann, &samples, &filters, fft_size, fft_step, n_len, n_mel,
                            n_threads,
                        )
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                // wait for each thread to finish and collect their results
                .map(|handle| handle.join().expect("Thread failed"))
                .collect::<Vec<_>>()
        })
    };

    let l = all_outputs[0].len();
    let mut mel = vec![0.0; l];

    // iterate over mel spectrogram segments, dividing work by threads.
    for segment_start in (0..l).step_by(n_threads) {
        // go through each thread's output.
        for thread_output in all_outputs.iter() {
            // add each thread's piece to our mel spectrogram.
            for offset in 0..n_threads {
                let mel_index = segment_start + offset; // find location in mel.
                if let Some(mel) = mel.get_mut(mel_index) {
                    // Make sure we don't go out of bounds.
                    *mel += thread_output[mel_index];
                }
            }
        }
    }

    let mmax = mel
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .unwrap_or_default()
        - 8.0_f32;
    for m in mel.iter_mut() {
        let v = f32::max(*m, mmax);
        *m = v / 4.0 + 1.0
    }
    mel
}

pub fn pcm_to_mel(cfg: &Config, samples: &[f32], filters: &[f32]) -> Vec<f32> {
    log_mel_spectrogram_(samples, filters, N_FFT, HOP_LENGTH, cfg.num_mel_bins)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft() {
        let mut input = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0),
            Complex::new(0.0, 0.0),
        ];
        let mut output = Vec::new();
        fft(&mut input, &mut output);
        assert_eq!(output, vec![1.0, 0.0, 0.0, -1.0, -1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_log_mel_spectrogram() {
        let samples = vec![0.0; 1000];
        let filters = vec![0.0; 1000];
        let output = log_mel_spectrogram_(&samples, &filters, 100, 10, 10);
        assert_eq!(output.len(), 30_000);
    }

    #[test]
    fn test_tiny_log_mel_spectrogram() {
        let samples = vec![0.0; 100];
        let filters = vec![0.0; 100];
        let output = log_mel_spectrogram_(&samples, &filters, 20, 2, 2);
        assert_eq!(output.len(), 6_000);
    }
}
