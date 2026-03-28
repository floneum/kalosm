use half::bf16;
use rand::{Rng, SeedableRng};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::cohere_config::CohereConfig;

const PREEMPH: f32 = 0.97;
const LOG_ZERO_GUARD: f32 = 5.960_464_5e-8_f32;

fn quantize_bf16(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

fn sample_standard_normal(rng: &mut rand::rngs::StdRng) -> f32 {
    let u1 = rng.random::<f32>().clamp(f32::MIN_POSITIVE, 1.0);
    let u2 = rng.random::<f32>();
    (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
}

pub fn pcm_to_features(cfg: &CohereConfig, samples: &[f32], filters: &[f32]) -> (Vec<f32>, usize, usize) {
    let sample_rate = cfg.preprocessor.sample_rate;
    let n_fft = cfg.preprocessor.n_fft;
    let win_length = (cfg.preprocessor.window_size * sample_rate as f32).round() as usize;
    let hop_length = (cfg.preprocessor.window_stride * sample_rate as f32).round() as usize;
    let n_mels = cfg.preprocessor.features;
    let n_freqs = n_fft / 2 + 1;
    let pad = n_fft / 2;
    let total_frames = samples.len() / hop_length + 1;
    let valid_frames = samples.len() / hop_length;

    let mut waveform = samples.to_vec();
    if cfg.preprocessor.dither > 0.0 {
        let mut rng = rand::rngs::StdRng::seed_from_u64(samples.len() as u64);
        for sample in &mut waveform {
            *sample += cfg.preprocessor.dither * sample_standard_normal(&mut rng);
        }
    }

    if !waveform.is_empty() {
        for i in (1..waveform.len()).rev() {
            waveform[i] -= PREEMPH * waveform[i - 1];
        }
    }

    let mut padded = vec![0.0f32; waveform.len() + 2 * pad];
    padded[pad..pad + waveform.len()].copy_from_slice(&waveform);

    let mut window = vec![0.0f32; n_fft];
    let pad_left = (n_fft - win_length) / 2;
    for i in 0..win_length {
        let phase = std::f32::consts::TAU * i as f32 / (win_length.saturating_sub(1)) as f32;
        window[pad_left + i] = quantize_bf16(0.5 - 0.5 * phase.cos());
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut frame = vec![Complex::ZERO; n_fft];
    let mut fft_out = vec![0.0f32; n_freqs];
    let mut features = vec![cfg.preprocessor.pad_value; n_mels * total_frames];

    for frame_idx in 0..total_frames {
        let start = frame_idx * hop_length;
        for i in 0..n_fft {
            frame[i] = Complex::new(padded[start + i] * window[i], 0.0);
        }
        fft.process(&mut frame);

        for i in 0..n_freqs {
            let re = frame[i].re;
            let im = frame[i].im;
            fft_out[i] = re * re + im * im;
        }

        for mel in 0..n_mels {
            let filter = &filters[mel * n_freqs..(mel + 1) * n_freqs];
            let mut sum = 0.0f32;
            for i in 0..n_freqs {
                sum += filter[i] * fft_out[i];
            }
            features[mel * total_frames + frame_idx] = (sum + LOG_ZERO_GUARD).ln();
        }
    }

    if cfg.preprocessor.normalize == "per_feature" && valid_frames > 1 {
        for mel in 0..n_mels {
            let row = &mut features[mel * total_frames..(mel + 1) * total_frames];
            let mean = row[..valid_frames].iter().copied().sum::<f32>() / valid_frames as f32;
            let var = row[..valid_frames]
                .iter()
                .map(|value| {
                    let diff = *value - mean;
                    diff * diff
                })
                .sum::<f32>()
                / (valid_frames as f32 - 1.0);
            let std = var.sqrt() + 1e-5;
            for value in &mut row[..valid_frames] {
                *value = (*value - mean) / std;
            }
            for value in &mut row[valid_frames..] {
                *value = cfg.preprocessor.pad_value;
            }
        }
    }

    (features, total_frames, valid_frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cohere_config::{
        CohereConfig, CohereDecoderConfig, CohereDecoderWrapperConfig, CohereEncoderConfig, CohereHeadConfig,
        CoherePreprocessorConfig,
    };

    fn test_config() -> CohereConfig {
        CohereConfig {
            batch_size: 64,
            sample_rate: 16_000,
            max_audio_clip_s: 35,
            overlap_chunk_second: 5,
            min_energy_window_samples: 1600,
            supported_languages: vec!["en".to_owned()],
            preprocessor: CoherePreprocessorConfig {
                features: 128,
                sample_rate: 16_000,
                window_size: 0.025,
                window_stride: 0.01,
                n_fft: 512,
                normalize: "per_feature".to_owned(),
                dither: 1e-5,
                pad_to: 0,
                pad_value: 0.0,
                window: "hann".to_owned(),
            },
            encoder: CohereEncoderConfig {
                att_context_size: [-1, -1],
                conv_kernel_size: 9,
                conv_norm_type: "batch_norm".to_owned(),
                d_model: 1280,
                dropout: 0.0,
                feat_in: 128,
                feat_out: 1280,
                ff_expansion_factor: 4,
                n_heads: 8,
                n_layers: 48,
                pos_emb_max_len: 5000,
                self_attention_model: "rel_pos".to_owned(),
                subsampling: "dw_striding".to_owned(),
                subsampling_conv_channels: 256,
                subsampling_factor: 8,
                untie_biases: false,
                xscaling: true,
            },
            transf_decoder: CohereDecoderWrapperConfig {
                config_dict: CohereDecoderConfig {
                    hidden_act: "relu".to_owned(),
                    hidden_size: 1024,
                    inner_size: 4096,
                    learn_positional_encodings: false,
                    max_sequence_length: 1024,
                    num_attention_heads: 8,
                    num_layers: 8,
                },
            },
            head: CohereHeadConfig {
                hidden_size: 1024,
                log_softmax: true,
                num_classes: 16_384,
            },
            vocab_size: 16_384,
        }
    }

    #[test]
    fn debug_feature_stats() {
        let cfg = test_config();
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/samples_jfk.wav");
        let reader = hound::WavReader::open(path).unwrap();
        let samples = reader
            .into_samples::<i16>()
            .map(|sample| sample.unwrap() as f32 / i16::MAX as f32)
            .take(16_000)
            .collect::<Vec<_>>();
        let filter_bytes = include_bytes!("cohere_melfilters128.bytes").as_slice();
        let mut filterbank = vec![0.0f32; filter_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(filter_bytes, &mut filterbank);

        let (features, total_frames, valid_frames) = pcm_to_features(&cfg, &samples, &filterbank);
        let sum: f32 = features.iter().sum();
        let mean = sum / features.len() as f32;
        let var = features
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f32>()
            / features.len() as f32;
        println!("frames total={total_frames} valid={valid_frames}");
        println!("sum={sum} mean={mean} std={}", var.sqrt());
        println!("first20={:?}", &features[..20]);
    }
}
