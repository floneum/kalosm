//! Regression test for the qwen-vision multi-block resolve pattern.
//!
//! Each "block" mimics the shape of `VisionBlock::forward`: narrow a fused QKV
//! tensor into Q/K/V, run windowed flash attention (multiple narrows + flash
//! per block, then sum), add a residual, then an MLP-shaped elementwise pass.
//! The point isn't numerical correctness against a reference, it's to exercise
//! the same fusion+resolve interaction the qwen vision encoder hits — without
//! depending on GGUF weights or a real model load.
//!
//! Pre-fix this graph used to need a manual `materialize_sync` flush every
//! 4 blocks (qwen.rs `FLUSH_EVERY = 4`) because the resolver's freeing
//! predicate kept every intermediate alive while the held final tensor stayed
//! uncached. The new `alive_uncached` counter + per-step propagation in
//! `set_cached_result` makes the single end-of-loop resolve handle it.
use fusor::Tensor;
use fusor_conformance::available_devices;

// Dimensions match qwen2.5-VL's vision encoder defaults (clip.vision.* in the
// GGUF metadata). SEQ_LEN here is the smaller-image path (~512); the full
// 1944-token path is wider but exercises the same fusion topology.
const BLOCKS: usize = 32;
const SEQ_LEN: usize = 512;
const HEAD_COUNT: usize = 16;
const HEAD_DIM: usize = 80;
const EMBED_DIM: usize = HEAD_COUNT * HEAD_DIM; // 1280
const WINDOW_LEN: usize = 64;
const WINDOWS_PER_SEQ: usize = SEQ_LEN / WINDOW_LEN;
const MLP_INTERMEDIATE: usize = 3420;

fn ramp_data(len: usize, scale: f32) -> Vec<f32> {
    // Small magnitudes keep the values finite through many stacked blocks of
    // matmul + flash + residual + mlp without depending on real
    // initialization.
    (0..len)
        .map(|i| (((i % 23) as f32) / 23.0 - 0.5) * scale + 0.001)
        .collect()
}

async fn run_blocks(device: &fusor::Device, flush_every: Option<usize>) -> Vec<f32> {
    let xs_data = ramp_data(SEQ_LEN * EMBED_DIM, 0.5);
    let qkv_w_data = ramp_data(EMBED_DIM * (3 * EMBED_DIM), 0.02);
    let proj_w_data = ramp_data(EMBED_DIM * EMBED_DIM, 0.02);
    let mlp_up_data = ramp_data(EMBED_DIM * MLP_INTERMEDIATE, 0.02);
    let mlp_down_data = ramp_data(MLP_INTERMEDIATE * EMBED_DIM, 0.02);

    let mut xs: Tensor<2, f32> = Tensor::from_slice(device, [SEQ_LEN, EMBED_DIM], &xs_data);
    let qkv_w: Tensor<2, f32> = Tensor::from_slice(device, [EMBED_DIM, 3 * EMBED_DIM], &qkv_w_data);
    let proj_w: Tensor<2, f32> = Tensor::from_slice(device, [EMBED_DIM, EMBED_DIM], &proj_w_data);
    let mlp_up_w: Tensor<2, f32> =
        Tensor::from_slice(device, [EMBED_DIM, MLP_INTERMEDIATE], &mlp_up_data);
    let mlp_down_w: Tensor<2, f32> =
        Tensor::from_slice(device, [MLP_INTERMEDIATE, EMBED_DIM], &mlp_down_data);

    for block in 0..BLOCKS {
        let qkv = xs.mat_mul(&qkv_w);
        let q = qkv
            .narrow(1, 0, EMBED_DIM)
            .reshape([SEQ_LEN, HEAD_COUNT, HEAD_DIM])
            .transpose(0, 1)
            .unsqueeze(0)
            .to_concrete();
        let k = qkv
            .narrow(1, EMBED_DIM, EMBED_DIM)
            .reshape([SEQ_LEN, HEAD_COUNT, HEAD_DIM])
            .transpose(0, 1)
            .unsqueeze(0)
            .to_concrete();
        let v = qkv
            .narrow(1, 2 * EMBED_DIM, EMBED_DIM)
            .reshape([SEQ_LEN, HEAD_COUNT, HEAD_DIM])
            .transpose(0, 1)
            .unsqueeze(0)
            .to_concrete();

        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let mut window_outputs = Vec::with_capacity(WINDOWS_PER_SEQ);
        for w in 0..WINDOWS_PER_SEQ {
            let start = w * WINDOW_LEN;
            let qw = q.narrow(2, start, WINDOW_LEN).to_concrete();
            let kw = k.narrow(2, start, WINDOW_LEN).to_concrete();
            let vw = v.narrow(2, start, WINDOW_LEN).to_concrete();
            let attn_w: Tensor<4, f32> = qw.flash_attention(&kw, &vw, scale, None);
            window_outputs.push(attn_w);
        }
        let attn_out: Tensor<4, f32> = fusor::cat(window_outputs, 2).to_concrete();
        let attn_out_2d: Tensor<2, f32> = attn_out
            .squeeze(0)
            .transpose(0, 1)
            .reshape([SEQ_LEN, EMBED_DIM])
            .to_concrete();
        let attn_proj = attn_out_2d.mat_mul(&proj_w);
        let xs_after_attn: Tensor<2, f32> = (&xs + &attn_proj).to_concrete();
        let mlp_hidden = xs_after_attn.mat_mul(&mlp_up_w);
        let mlp_act = &mlp_hidden * mlp_hidden.tanh();
        let mlp_out = mlp_act.mat_mul(&mlp_down_w);
        xs = (&xs_after_attn + &mlp_out).to_concrete();

        // Optionally simulate the old qwen.rs FLUSH_EVERY workaround by
        // materializing the running tensor every N blocks.
        if let Some(n) = flush_every
            && (block + 1) % n == 0
            && let Some(g) = xs.as_gpu()
        {
            g.materialize_sync();
        }
    }

    let result_data = xs.as_slice().await.unwrap();
    assert_eq!(result_data.shape(), &[SEQ_LEN, EMBED_DIM]);
    // We deliberately skip a finite-value check: the mock weights aren't
    // normalized (no RmsNorm) so the activations overflow over 32 blocks.
    // The test is about graph build/resolve, not numerical correctness.
    result_data.as_slice().to_vec()
}

#[tokio::test]
async fn vision_block_pattern_resolves_without_periodic_flush() {
    for device in available_devices().await {
        let _ = run_blocks(&device, None).await;
    }
}

/// Compare wall-clock of the same vision-block graph with no periodic flush
/// vs. with the old qwen.rs `FLUSH_EVERY = 4` workaround. Doesn't assert a
/// ratio (CI noise) — just prints both numbers so a human can sanity-check
/// that dropping the flush isn't a regression.
#[tokio::test]
async fn vision_block_pattern_flush_vs_no_flush_timing() {
    if std::env::var_os("FUSOR_FLUSH_TIMING").is_none() {
        // Off by default — only run when explicitly requested, since
        // micro-benchmarks in `cargo test` are noisy.
        return;
    }
    // CPU is too slow at qwen scale to time in cargo test (single iteration
    // takes minutes). The interesting comparison is GPU wall-clock — that's
    // what matters for the qwen vision pipeline.
    for device in available_devices().await {
        if device.as_gpu().is_none() {
            continue;
        }
        // Warm-up: kernels need to be compiled/cached once. Both paths share
        // the same kernels so a single warm-up benefits both timings.
        let _ = run_blocks(&device, None).await;

        const ITERS: usize = 3;

        let mut no_flush_total = std::time::Duration::ZERO;
        for _ in 0..ITERS {
            let t = std::time::Instant::now();
            let _ = run_blocks(&device, None).await;
            no_flush_total += t.elapsed();
        }

        let mut flush_total = std::time::Duration::ZERO;
        for _ in 0..ITERS {
            let t = std::time::Instant::now();
            let _ = run_blocks(&device, Some(4)).await;
            flush_total += t.elapsed();
        }

        eprintln!(
            "device={device:?} blocks={BLOCKS} iters={ITERS}: no_flush_avg={:.2?} flush_every_4_avg={:.2?}",
            no_flush_total / ITERS as u32,
            flush_total / ITERS as u32,
        );
    }
}
