//! Composed lm-head + softmax-cross-entropy cost at a real vocabulary size:
//! the baseline the streaming (logits-free) CE kernels must beat. The
//! example transformer's vocab of 65 makes this cluster ~1% of a step;
//! at 32k vocab the materialized logits and their gradient dominate.

use fusor::autograd::{Graph, Tensor};
use fusor::{Device, Tensor as RawTensor};

fn main() {
    if std::env::var_os("RUST_LOG").is_some() {
        tracing_subscriber::fmt()
            .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
            .init();
    }
    pollster::block_on(async {
        let device = Device::gpu().await.expect("gpu device");
        let rows: usize = std::env::var("ROWS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(16384);
        let dim: usize = 384;
        let vocab: usize = std::env::var("VOCAB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(32768);

        let mut state = 0x5eed_u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        };
        let x_data: Vec<f32> = (0..rows * dim).map(|_| next() * 0.1).collect();
        let w_data: Vec<f32> = (0..vocab * dim).map(|_| next() * 0.05).collect();
        let targets: Vec<u32> = (0..rows).map(|i| (i * 2654435761 % vocab) as u32).collect();
        let targets = RawTensor::from_slice(&device, [rows], &targets);

        let step = || {
            let graph = Graph::new();
            let x: Tensor<2> = Tensor::from_slice(&graph, &device, [rows, dim], &x_data);
            let w: Tensor<2> = Tensor::from_slice(&graph, &device, [vocab, dim], &w_data);
            let logits = x.mat_mul_transposed_rhs(&w);
            let loss = logits.softmax_cross_entropy(&targets);
            let gradients = loss.backward().unwrap();
            let dw = gradients.get(&w).expect("dW");
            let dx = gradients.get(&x).expect("dX");
            (loss, dx, dw)
        };

        // Warm (compile kernels).
        {
            let (_loss, dx, dw) = step();
            pollster::block_on(dx.materialize());
            pollster::block_on(dw.materialize());
        }
        let iters = 5;
        let mut best = f64::MAX;
        for _ in 0..3 {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let (_loss, dx, dw) = step();
                pollster::block_on(dx.materialize());
                pollster::block_on(dw.materialize());
            }
            best = best.min(start.elapsed().as_secs_f64() / iters as f64);
        }
        let logits_bytes = (rows * vocab * 4) as f64;
        println!(
            "composed lm_head+CE fwd+bwd rows={rows} dim={dim} vocab={vocab}: {:.2} ms/iter (logits slab {:.2} GB)",
            best * 1e3,
            logits_bytes / 1e9
        );
    });
}
