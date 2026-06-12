// A/B microbench: conv-as-im2col matmul with the gather materialization
// (the old pipeline) vs the implicit-GEMM unflatten (the new pipeline).
//
// Holding the flat [M, K] view tensor across materialization trips the
// resolver's live-reference guard, which declines the unflatten — giving
// exactly the old gather + matmul pipeline in the same binary.
//
// Run with:
//   cargo run --package fusor-core --example bench_conv_implicit_gemm --release

use std::time::{Duration, Instant};

use fusor_core::{Device, StrideSpec, Tensor};

const WARMUP: usize = 5;
const MEASURED: usize = 30;

struct ConvCase {
    name: &'static str,
    b: usize,
    c: usize,
    h: usize,
    w: usize,
    n: usize,
    kh: usize,
    kw: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(async {
        let device = Device::new().await?;
        println!("bench_conv_implicit_gemm (warmup {WARMUP}, measured {MEASURED})");
        println!();

        let cases = [
            // Coop-tile-unaligned M: both variants run the generic reduce.
            ConvCase {
                name: "small_unaligned",
                b: 2,
                c: 8,
                h: 16,
                w: 16,
                n: 16,
                kh: 3,
                kw: 3,
            },
            ConvCase {
                name: "mid_unaligned_n256",
                b: 1,
                c: 128,
                h: 35,
                w: 35,
                n: 256,
                kh: 3,
                kw: 3,
            },
            ConvCase {
                name: "mid_unaligned_n128",
                b: 1,
                c: 128,
                h: 35,
                w: 35,
                n: 128,
                kh: 3,
                kw: 3,
            },
            ConvCase {
                name: "mid_unaligned_n64",
                b: 1,
                c: 128,
                h: 35,
                w: 35,
                n: 64,
                kh: 3,
                kw: 3,
            },
            // Coop-tile-aligned M/K/N: the matmul runs the hardware kernel.
            ConvCase {
                name: "large_aligned",
                b: 2,
                c: 64,
                h: 34,
                w: 34,
                n: 128,
                kh: 3,
                kw: 3,
            },
            ConvCase {
                name: "vision_aligned",
                b: 1,
                c: 256,
                h: 66,
                w: 66,
                n: 256,
                kh: 3,
                kw: 3,
            },
        ];

        for case in &cases {
            bench_case(&device, case);
        }
        Ok(())
    })
}

fn bench_case(device: &Device, case: &ConvCase) {
    let &ConvCase {
        name,
        b,
        c,
        h,
        w,
        n,
        kh,
        kw,
    } = case;
    let (oh, ow) = (h - kh + 1, w - kw + 1);
    let (m, k) = (b * oh * ow, c * kh * kw);

    let input_host: Vec<f32> = (0..b * c * h * w).map(|i| (i % 13) as f32 * 0.1).collect();
    let weight_host: Vec<f32> = (0..n * k).map(|i| (i % 7) as f32 * 0.01).collect();
    let input = Tensor::from_slice(device, [b, c, h, w], &input_host);
    let weight = Tensor::from_slice(device, [n, k], &weight_host);
    input.materialize_sync();
    weight.materialize_sync();

    let build = |hold_flat: bool| -> (Tensor, Option<Tensor>) {
        let windows = input.restride([
            StrideSpec::dim(0, b),
            StrideSpec::dim_with(2, oh, 1),
            StrideSpec::dim_with(3, ow, 1),
            StrideSpec::dim(1, c),
            StrideSpec::dim(2, kh),
            StrideSpec::dim(3, kw),
        ]);
        let a = windows.reshape([m, k]);
        let b_mat = weight.restride([StrideSpec::dim(1, k), StrideSpec::dim(0, n)]);
        let out = a.mat_mul(&b_mat);
        (out, hold_flat.then_some(a))
    };

    let run = |hold_flat: bool| -> (Vec<Duration>, usize) {
        let mut kernels = 0;
        for _ in 0..WARMUP {
            let (out, held) = build(hold_flat);
            kernels = out.count_kernels_to_resolve();
            device.poll_wait();
            drop(held);
            drop(out);
        }
        let mut samples = Vec::with_capacity(MEASURED);
        for _ in 0..MEASURED {
            let (out, held) = build(hold_flat);
            let start = Instant::now();
            let _ = out.count_kernels_to_resolve();
            device.poll_wait();
            samples.push(start.elapsed());
            drop(held);
            drop(out);
        }
        samples.sort_unstable();
        (samples, kernels)
    };

    let (gather, gather_kernels) = run(true);
    let (implicit, implicit_kernels) = run(false);

    let stats = |samples: &[Duration]| {
        let mean = samples.iter().sum::<Duration>() / samples.len() as u32;
        let p50 = samples[samples.len() / 2];
        (
            mean.as_secs_f64() * 1000.0,
            p50.as_secs_f64() * 1000.0,
            samples[0].as_secs_f64() * 1000.0,
        )
    };
    let (g_mean, g_p50, g_min) = stats(&gather);
    let (i_mean, i_p50, i_min) = stats(&implicit);

    println!("{name}: conv {b}x{c}x{h}x{w} k{kh}x{kw} -> matmul {m}x{k} @ {k}x{n}");
    println!("  gather+matmul ({gather_kernels} dispatches): mean {g_mean:.3} ms, p50 {g_p50:.3} ms, min {g_min:.3} ms");
    println!("  implicit-GEMM ({implicit_kernels} dispatches): mean {i_mean:.3} ms, p50 {i_p50:.3} ms, min {i_min:.3} ms");
    println!("  speedup (p50): {:.2}x", g_p50 / i_p50);
    println!();
}
