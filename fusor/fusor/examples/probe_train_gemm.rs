//! Probe: the training step's contraction shapes on the general Session path.
//! `probe_train_gemm <fwd|dw|dx> [m] [k] [n]`
use fusor::{Device, Tensor};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("dw");
    let dim = |i: usize, d: usize| args.get(i).and_then(|v| v.parse().ok()).unwrap_or(d);
    // dw: out[m,n] = x[k,m]^T @ g[k,n]; fwd: out[m,n] = x[m,k] @ w[k,n];
    // dx: out[m,n] = g[m,k] @ w[n,k]^T.
    let (m, k, n) = (dim(1, 192), dim(2, 1024), dim(3, 96));
    let device = pollster::block_on(Device::gpu()).expect("gpu");
    let data = |len: usize, p: usize| -> Vec<f32> {
        (0..len)
            .map(|i| ((i % p) as f32 - p as f32 / 2.0) * 0.01)
            .collect()
    };
    for iter in 0..6 {
        let t = std::time::Instant::now();
        let out = match mode {
            "dw" => {
                let x = Tensor::<2, f32>::from_slice(&device, [k, m], &data(k * m, 97));
                let g = Tensor::<2, f32>::from_slice(&device, [k, n], &data(k * n, 89));
                x.t().matmul(&g).to_flat()
            }
            "fwd" => {
                let x = Tensor::<2, f32>::from_slice(&device, [m, k], &data(m * k, 97));
                let w = Tensor::<2, f32>::from_slice(&device, [k, n], &data(k * n, 89));
                x.matmul(&w).to_flat()
            }
            "dx" => {
                let g = Tensor::<2, f32>::from_slice(&device, [m, k], &data(m * k, 97));
                let w = Tensor::<2, f32>::from_slice(&device, [n, k], &data(n * k, 89));
                g.matmul_t(&w).to_flat()
            }
            "pair" => {
                // Two products of one input, both read back: a group of
                // independent contractions. Checked against the host.
                let xs = data(m * k, 97);
                // The second product is wider, so its tile geometry and block
                // differ from the first's and one member runs widened.
                let n2 = std::env::var("PAIR_N2")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(n);
                let (w1, w2) = (data(k * n, 89), data(k * n2, 83));
                let x = Tensor::<2, f32>::from_slice(&device, [m, k], &xs);
                let a = x.matmul(&Tensor::<2, f32>::from_slice(&device, [k, n], &w1));
                let b = x.matmul(&Tensor::<2, f32>::from_slice(&device, [k, n2], &w2));
                device
                    .session()
                    .resolve(&[a.as_dyn().clone(), b.as_dyn().clone()])
                    .unwrap();
                let (ga, gb) = (a.to_flat(), b.to_flat());
                let host = |w: &[f32], n: usize| -> Vec<f32> {
                    (0..m * n)
                        .map(|ij| {
                            (0..k)
                                .map(|kk| xs[ij / n * k + kk] * w[kk * n + ij % n])
                                .sum()
                        })
                        .collect()
                };
                let err = |g: &[f32], h: &[f32]| {
                    g.iter()
                        .zip(h)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max)
                };
                eprintln!(
                    "pair max_err a={} b={}",
                    err(&ga, &host(&w1, n)),
                    err(&gb, &host(&w2, n2))
                );
                ga
            }
            "one" => {
                // One product, checked against the host.
                let (xs, ws) = (data(m * k, 97), data(k * n, 89));
                let x = Tensor::<2, f32>::from_slice(&device, [m, k], &xs);
                let g = x
                    .matmul(&Tensor::<2, f32>::from_slice(&device, [k, n], &ws))
                    .to_flat();
                let err = (0..m * n)
                    .map(|ij| {
                        let h: f32 = (0..k)
                            .map(|kk| xs[ij / n * k + kk] * ws[kk * n + ij % n])
                            .sum();
                        (g[ij] - h).abs()
                    })
                    .fold(0f32, f32::max);
                eprintln!("one max_err={err}");
                g
            }
            _ => panic!("mode: fwd|dw|dx|pair|one"),
        };
        eprintln!(
            "iter={iter} ms={:.3} out0={}",
            t.elapsed().as_secs_f64() * 1e3,
            out[0]
        );
    }
}
