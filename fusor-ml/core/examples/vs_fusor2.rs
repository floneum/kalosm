//! fusor-ml (reference) half of the fusor-vs-fusor2 comparison.
//!
//! Same workloads, same shapes, same protocol as
//! `fusor2/fusor2/examples/vs_fusor1.rs`. Prints one TSV row per workload:
//!
//!     workload  cold_ms  min_ms  median_ms  launches
//!
//! Protocol per iteration: fresh tensors uploaded from host slices, build the
//! expression, force resolve + wait + readback. Upload is inside the timed
//! region on both sides.

use fusor_core::{Device, Tensor};
use pollster::block_on;
use std::time::{Duration, Instant};

const WARMUP: usize = 3;
const ITERS: usize = 10;

/// Deterministic pseudo-random host data, identical formula on both sides.
fn make(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32 * seed).sin() * scale).clamp(-1.0, 1.0))
        .collect()
}

struct Timing {
    cold: Duration,
    min: Duration,
    median: Duration,
    launches: usize,
}

fn time_it(mut f: impl FnMut() -> usize) -> Timing {
    let t = Instant::now();
    let launches = f();
    let cold = t.elapsed();

    for _ in 0..WARMUP {
        f();
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f();
        samples.push(t.elapsed());
    }
    samples.sort();
    Timing {
        cold,
        min: samples[0],
        median: samples[samples.len() / 2],
        launches,
    }
}

fn row(name: &str, t: &Timing) {
    println!(
        "{name}\t{:.3}\t{:.3}\t{:.3}\t{}",
        t.cold.as_secs_f64() * 1e3,
        t.min.as_secs_f64() * 1e3,
        t.median.as_secs_f64() * 1e3,
        t.launches
    );
}

/// A 20-op elementwise chain: enough arithmetic that the kernel, not the
/// upload, is what the row measures.
fn chain(x: &Tensor, y: &Tensor) -> Tensor {
    let mut z = x.clone() + y;
    for _ in 0..5 {
        z = ((z * y).tanh() + x).abs();
    }
    z
}

fn main() {
    let device = match block_on(Device::new()) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no gpu: {e}");
            return;
        }
    };
    println!("# fusor-ml (reference)");
    println!("workload\tcold_ms\tmin_ms\tmedian_ms\tlaunches");

    // Launch counts are taken once, outside the timed loop, because
    // `count_kernels_to_resolve` resolves the tensor itself.
    let mut count_of = |t: &Tensor| t.count_kernels_to_resolve();

    // ---- 0. upload + readback floor, no compute. Both sides move the same
    //         bytes, so every row below is this plus its kernel. ----
    {
        let n = 2048usize;
        let xd = make(n * n, 0.013, 0.9);
        let t = time_it(|| {
            let x = Tensor::from_slice(&device, [n, n], &xd);
            let z = x.sum(1);
            block_on(z.as_slice::<1, f32>()).unwrap();
            1
        });
        row("passthrough_2048", &t);
    }

    // ---- 1. matmul 1024^3 ----
    {
        let n = 2048usize;
        let ad = make(n * n, 0.013, 0.5);
        let bd = make(n * n, 0.017, 0.5);
        let launches = {
            let a = Tensor::from_slice(&device, [n, n], &ad);
            let b = Tensor::from_slice(&device, [n, n], &bd);
            count_of(&a.mat_mul(&b).sum(1))
        };
        let t = time_it(|| {
            let a = Tensor::from_slice(&device, [n, n], &ad);
            let b = Tensor::from_slice(&device, [n, n], &bd);
            let c = a.mat_mul(&b).sum(1);
            block_on(c.as_slice::<1, f32>()).unwrap();
            launches
        });
        row("matmul_2048", &t);
    }

    // ---- 2. matmul + bias + tanh (epilogue fusion) ----
    {
        let n = 2048usize;
        let ad = make(n * n, 0.013, 0.5);
        let bd = make(n * n, 0.017, 0.5);
        let bias = make(n * n, 0.023, 0.1);
        let launches = {
            let a = Tensor::from_slice(&device, [n, n], &ad);
            let b = Tensor::from_slice(&device, [n, n], &bd);
            let bs = Tensor::from_slice(&device, [n, n], &bias);
            count_of(&(a.mat_mul(&b) + &bs).tanh().sum(1))
        };
        let t = time_it(|| {
            let a = Tensor::from_slice(&device, [n, n], &ad);
            let b = Tensor::from_slice(&device, [n, n], &bd);
            let bs = Tensor::from_slice(&device, [n, n], &bias);
            let c = (a.mat_mul(&b) + &bs).tanh().sum(1);
            block_on(c.as_slice::<1, f32>()).unwrap();
            launches
        });
        row("matmul_epilogue_2048", &t);
    }

    // ---- 3. elementwise chain, 2048^2 ----
    {
        let n = 2048usize;
        let xd = make(n * n, 0.013, 0.9);
        let yd = make(n * n, 0.017, 0.9);
        let launches = {
            let x = Tensor::from_slice(&device, [n, n], &xd);
            let y = Tensor::from_slice(&device, [n, n], &yd);
            count_of(&chain(&x, &y).sum(1))
        };
        let t = time_it(|| {
            let x = Tensor::from_slice(&device, [n, n], &xd);
            let y = Tensor::from_slice(&device, [n, n], &yd);
            let z = chain(&x, &y).sum(1);
            block_on(z.as_slice::<1, f32>()).unwrap();
            launches
        });
        row("elementwise_chain_2048", &t);
    }

    // ---- 4. softmax last dim, 2048^2 ----
    {
        let n = 2048usize;
        let xd = make(n * n, 0.013, 2.0);
        let launches = {
            let x = Tensor::from_slice(&device, [n, n], &xd);
            count_of(&x.softmax_last_dim().sum(1))
        };
        let t = time_it(|| {
            let x = Tensor::from_slice(&device, [n, n], &xd);
            let z = x.softmax_last_dim().sum(1);
            block_on(z.as_slice::<1, f32>()).unwrap();
            launches
        });
        row("softmax_2048", &t);
    }

    // ---- 5. rms_norm, 2048^2 ----
    {
        let n = 2048usize;
        let xd = make(n * n, 0.013, 1.0);
        let wd = make(n, 0.031, 1.0);
        let launches = {
            let x = Tensor::from_slice(&device, [n, n], &xd);
            let w = Tensor::from_slice(&device, [n], &wd);
            count_of(&x.rms_norm_fused_no_bias(&w, 1e-5).sum(1))
        };
        let t = time_it(|| {
            let x = Tensor::from_slice(&device, [n, n], &xd);
            let w = Tensor::from_slice(&device, [n], &wd);
            let z = x.rms_norm_fused_no_bias(&w, 1e-5).sum(1);
            block_on(z.as_slice::<1, f32>()).unwrap();
            launches
        });
        row("rms_norm_2048", &t);
    }

    // ---- 6. attention forward, [1,8,512,64] ----
    {
        let shape = [1usize, 8, 1024, 64];
        let numel: usize = shape.iter().product();
        let qd = make(numel, 0.013, 0.1);
        let kd = make(numel, 0.017, 0.7);
        let vd = make(numel, 0.019, 1.3);
        let scale = 1.0f32 / (64f32).sqrt();
        let launches = {
            let q = Tensor::from_slice(&device, shape, &qd);
            let k = Tensor::from_slice(&device, shape, &kd);
            let v = Tensor::from_slice(&device, shape, &vd);
            count_of(&q.attention(&k, &v, scale, None).sum(3))
        };
        let t = time_it(|| {
            let q = Tensor::from_slice(&device, shape, &qd);
            let k = Tensor::from_slice(&device, shape, &kd);
            let v = Tensor::from_slice(&device, shape, &vd);
            let o = q.attention(&k, &v, scale, None).sum(3);
            block_on(o.as_slice::<3, f32>()).unwrap();
            launches
        });
        row("attention_1x8x1024x64", &t);
    }

    // ---- 7. quantized matmul, Q4K weights, same shape and bytes as the
    //         fusor2 row: weight [n, k] = [4096, 4096] of 0x11 blocks,
    //         activation [256, 4096]. Weight upload inside the timed region
    //         on both sides. ----
    {
        use fusor_core::QMatrix;
        use fusor_gguf::GgmlType;
        let (k, n, m) = (4096usize, 4096usize, 256usize);
        let block_bytes = 144usize; // Q4K native
        let blocks = (k / 256) * n;
        let bytes = vec![0x11u8; blocks * block_bytes];
        let act = make(m * k, 0.013, 0.5);
        let launches = {
            let w =
                QMatrix::from_parts(&device, &bytes, [n, k].into(), GgmlType::Q4K).unwrap();
            let a = Tensor::from_slice(&device, [m, k], &act);
            count_of(&a.q_mat_mul(&w).sum(1))
        };
        let t = time_it(|| {
            let w =
                QMatrix::from_parts(&device, &bytes, [n, k].into(), GgmlType::Q4K).unwrap();
            let a = Tensor::from_slice(&device, [m, k], &act);
            let y = a.q_mat_mul(&w).sum(1);
            block_on(y.as_slice::<1, f32>()).unwrap();
            launches
        });
        row("qmatmul_q4k_256x4096x4096", &t);
    }

    // ---- 8. quantized matvec, Q4K weights, M=1: the LLM decode shape. ----
    {
        use fusor_core::QMatrix;
        use fusor_gguf::GgmlType;
        let (k, n) = (4096usize, 4096usize);
        let block_bytes = 144usize; // Q4K native
        let blocks = (k / 256) * n;
        let bytes = vec![0x11u8; blocks * block_bytes];
        let act = make(k, 0.013, 0.5);
        let launches = {
            let w =
                QMatrix::from_parts(&device, &bytes, [n, k].into(), GgmlType::Q4K).unwrap();
            let a = Tensor::from_slice(&device, [1, k], &act);
            count_of(&a.q_mat_mul(&w).sum(1))
        };
        let t = time_it(|| {
            let w =
                QMatrix::from_parts(&device, &bytes, [n, k].into(), GgmlType::Q4K).unwrap();
            let a = Tensor::from_slice(&device, [1, k], &act);
            let y = a.q_mat_mul(&w).sum(1);
            block_on(y.as_slice::<1, f32>()).unwrap();
            launches
        });
        row("qmatmul_q4k_1x4096x4096", &t);
    }
}
