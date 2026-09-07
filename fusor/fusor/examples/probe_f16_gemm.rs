//! Probe: which plan a dense contraction at a large M extracts to.
//! `probe_f16_gemm <f16|f32|cast> [m] [k] [n]`
use fusor::{Device, Tensor};
use half::f16;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("f16");
    let dim = |i: usize, d: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    let (m, k, n) = (dim(1, 1944), dim(2, 1280), dim(3, 3420));
    let device = pollster::block_on(Device::gpu()).expect("gpu");
    let xs: Vec<f32> = (0..m * k)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let ws: Vec<f32> = (0..n * k)
        .map(|i| ((i % 89) as f32 - 44.0) * 0.01)
        .collect();
    // Host reference: every row when the product is small, else row 0.
    let ref_rows = if m * n * k <= 400_000_000 { m } else { 1 };
    let act_of = |i: usize| -> f32 {
        if mode == "model2" {
            // silu(a) * b with b = a reversed, as the model form below.
            let a = xs[i];
            let b = xs[xs.len() - 1 - i];
            a / (1.0 + (-a).exp()) * b
        } else {
            xs[i]
        }
    };
    let reference: Vec<f32> = (0..ref_rows * n)
        .map(|ij| {
            let (i, j) = (ij / n, ij % n);
            (0..k).map(|kk| act_of(i * k + kk) * ws[j * k + kk]).sum()
        })
        .collect();
    let mut out = Vec::new();
    for iter in 0..3 {
        let t = std::time::Instant::now();
        out = match mode {
            "f16" => {
                let x = Tensor::<2, f16>::from_slice(
                    &device,
                    [m, k],
                    &xs.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>(),
                );
                let w = Tensor::<2, f16>::from_slice(
                    &device,
                    [n, k],
                    &ws.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>(),
                );
                x.matmul_t(&w).cast::<f32>().to_flat()
            }
            "f32" => {
                let x = Tensor::<2, f32>::from_slice(&device, [m, k], &xs);
                let w = Tensor::<2, f32>::from_slice(&device, [n, k], &ws);
                x.matmul_t(&w).to_flat()
            }
            "f16acc" => {
                let x = Tensor::<2, f16>::from_slice(
                    &device,
                    [m, k],
                    &xs.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>(),
                );
                let w = Tensor::<2, f16>::from_slice(
                    &device,
                    [n, k],
                    &ws.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>(),
                );
                let y = x
                    .as_dyn()
                    .matmul_t_acc(w.as_dyn(), fusor::Dtype::F16)
                    .expect("matmul");
                Tensor::<2, f32>::from_dyn(y.cast(fusor::Dtype::F32).expect("cast")).to_flat()
            }
            "cast" => {
                let x = Tensor::<2, f32>::from_slice(&device, [m, k], &xs);
                let w = Tensor::<2, f16>::from_slice(
                    &device,
                    [n, k],
                    &ws.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>(),
                );
                x.cast::<f16>().matmul_t(&w).cast::<f32>().to_flat()
            }
            "model2" => {
                // The down projection's form: silu(gate) * up from two
                // contraction-shaped inputs, cast, contracted, cast back.
                let a = Tensor::<3, f32>::from_slice(&device, [1, m, k], &xs);
                let b_data: Vec<f32> = xs.iter().rev().copied().collect();
                let b = Tensor::<3, f32>::from_slice(&device, [1, m, k], &b_data);
                let w = Tensor::<2, f16>::from_slice(
                    &device,
                    [n, k],
                    &ws.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>(),
                );
                let bias = Tensor::<1, f32>::zeros(&device, [n]);
                let act = a.silu().mul(&b).cast::<f16>().as_dyn().clone();
                let y = fusor::quantized::contract_rows(&act, w.as_dyn(), w.extent(0))
                    .expect("contract");
                let y = Tensor::<3, f32>::from_dyn(y.cast(fusor::Dtype::F32).expect("cast"));
                let y: Tensor<3, f32> = y.add_(&bias);
                y.to_flat()
            }
            _ => {
                // The model's form: a rank-3 f32 activation, cast, folded to
                // rows, contracted against an f16 leaf, cast back, plus a bias.
                let x = Tensor::<3, f32>::from_slice(&device, [1, m, k], &xs);
                let w = Tensor::<2, f16>::from_slice(
                    &device,
                    [n, k],
                    &ws.iter().map(|v| f16::from_f32(*v)).collect::<Vec<_>>(),
                );
                let bias = Tensor::<1, f32>::zeros(&device, [n]);
                let act = x.cast::<f16>().as_dyn().clone();
                let y = fusor::quantized::contract_rows(&act, w.as_dyn(), w.extent(0))
                    .expect("contract");
                let y = Tensor::<3, f32>::from_dyn(y.cast(fusor::Dtype::F32).expect("cast"));
                let y: Tensor<3, f32> = y.add_(&bias);
                y.to_flat()
            }
        };
        let elapsed = t.elapsed();
        println!("  iter {iter}: {elapsed:?}");
    }
    let max_err = reference
        .iter()
        .zip(&out[..reference.len()])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let bad = reference
        .iter()
        .zip(&out[..reference.len()])
        .filter(|(a, b)| (*a - *b).abs() > 0.05)
        .count();
    println!("  reference rows {ref_rows}: {bad} elements off by more than 0.05");
    println!(
        "{mode} [{m},{k}]x[{n},{k}]^T: {} elements, row-0 max abs err {max_err:.4}",
        out.len()
    );
}
