//! A/B timing for attention at the training-example shape
//! (batch 64, 6 heads, seq 256, head dim 64): the recognized fused row
//! program vs the composed matmul + softmax + matmul cluster (forced by
//! keeping the softmax probabilities alive as a second output, which fails
//! the exclusive-consumption gate exactly like a training graph does).

use fusor_core::{Device, Tensor};
use std::time::Instant;

const BATCH: usize = 64;
const HEADS: usize = 6;
const SEQ: usize = 256;
const HEAD_DIM: usize = 64;
const MASK_VALUE: f32 = -1e9;

fn values(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * scale).sin()).collect()
}

fn time_case(name: &str, warmup: usize, iters: usize, mut run: impl FnMut()) {
    for _ in 0..warmup {
        run();
    }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Instant::now();
        run();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{name}: median {:.3} ms, min {:.3} ms over {iters} iters",
        samples[samples.len() / 2],
        samples[0]
    );
}

fn main() {
    pollster::block_on(async {
        let device = Device::new().await.expect("gpu device");
        let elems = BATCH * HEADS * SEQ * HEAD_DIM;
        let q = Tensor::from_slice(&device, [BATCH, HEADS, SEQ, HEAD_DIM], &values(elems, 0.13));
        let k = Tensor::from_slice(&device, [BATCH, HEADS, SEQ, HEAD_DIM], &values(elems, 0.07));
        let v = Tensor::from_slice(&device, [BATCH, HEADS, SEQ, HEAD_DIM], &values(elems, 0.11));
        let scale = (HEAD_DIM as f32).powf(-0.5);

        let mask_values: Vec<f32> = (0..SEQ * SEQ)
            .map(|i| if i % SEQ > i / SEQ { MASK_VALUE } else { 0.0 })
            .collect();
        let mask = Tensor::from_slice(&device, [SEQ, SEQ], &mask_values);

        time_case("attention_recognized_mask", 3, 20, || {
            let out = q.attention(&k, &v, scale, Some(&mask));
            out.materialize_sync();
        });

        time_case("attention_recognized_causal", 3, 20, || {
            let out = q.attention_causal(&k, &v, scale);
            out.materialize_sync();
        });

        // Composed baseline: identical math, but the probabilities double as
        // an output so recognition cannot claim the cluster (the training
        // situation, where backward reads them).
        // K pre-transposed on the host: [B, H, HEAD_DIM, SEQ].
        let k_values = values(elems, 0.07);
        let mut k_t_values = vec![0.0f32; elems];
        for b in 0..BATCH * HEADS {
            for s in 0..SEQ {
                for d in 0..HEAD_DIM {
                    k_t_values[b * SEQ * HEAD_DIM + d * SEQ + s] =
                        k_values[b * SEQ * HEAD_DIM + s * HEAD_DIM + d];
                }
            }
        }
        let k_t = Tensor::from_slice(&device, [BATCH, HEADS, HEAD_DIM, SEQ], &k_t_values);

        time_case("attention_composed_probs_kept", 3, 20, || {
            let scores = q.mat_mul(&k_t) * scale;
            let masked = scores
                + mask.reshape([1, 1, SEQ, SEQ]).broadcast_as([
                    BATCH,
                    HEADS,
                    SEQ,
                    SEQ,
                ]);
            let probs = masked.softmax(3);
            let out = probs.mat_mul(&v);
            let (probs_r, out_r) = (probs.clone(), out.clone());
            probs_r.materialize_sync();
            out_r.materialize_sync();
        });
    });
}
