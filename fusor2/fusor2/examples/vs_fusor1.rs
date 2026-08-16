//! fusor2 half of the fusor-vs-fusor2 comparison.
//!
//! Same workloads, same shapes, same protocol as
//! `fusor-ml/core/examples/vs_fusor2.rs`. Prints one TSV row per workload:
//!
//!     workload  cold_ms  min_ms  median_ms  launches
//!
//! Protocol per iteration: fresh `Graph`, upload the leaves, build the
//! expression, force resolve + wait + readback. Upload is inside the timed
//! region on both sides, matching `fusor-ml/fusor/benches/fused.rs`'s own
//! methodology.

use fusor2::tensor::Dyn as Tensor;
use fusor2::{Graph, Session};
use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::launch::MaskKind;
use fusor2_ir::shape::Dim;
use std::time::{Duration, Instant};

const WARMUP: usize = 3;
const ITERS: usize = 10;

fn dims(shape: &[u64]) -> Vec<Dim> {
    shape.iter().map(|n| Dim::Const(*n)).collect()
}

fn bytes_of(data: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(data.len() * 4);
    for v in data {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

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
    launches: u64,
}

/// Run `f` once cold, then WARMUP + ITERS times, timing each.
fn time_it(session: &Session, mut f: impl FnMut() -> Result<(), String>) -> Timing {
    let l0 = session.launch_count();
    let t = Instant::now();
    f().expect("cold iteration failed");
    let cold = t.elapsed();
    let launches = session.launch_count() - l0;

    for _ in 0..WARMUP {
        f().expect("warmup failed");
    }
    let mut samples = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t = Instant::now();
        f().expect("timed iteration failed");
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

/// A 20-op elementwise chain.
fn chain(x: &Tensor, y: &Tensor) -> Result<Tensor, String> {
    let mut z = x.add(y).map_err(|e| e.to_string())?;
    for _ in 0..5 {
        z = z
            .mul(y)
            .map_err(|e| e.to_string())?
            .tanh()
            .map_err(|e| e.to_string())?
            .add(x)
            .map_err(|e| e.to_string())?
            .abs()
            .map_err(|e| e.to_string())?;
    }
    Ok(z)
}

fn main() {
    let device = if std::env::var("FUSOR2_DEVICE").as_deref() == Ok("cpu") {
        fusor2::session::Backend::cpu().expect("cpu backend")
    } else {
        match fusor2::session::Backend::gpu_blocking() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("no gpu: {e}");
                return;
            }
        }
    };
    let session = Session::new(device).expect("session");
    println!("# fusor2 on {}", session.device().name());
    println!("workload\tcold_ms\tmin_ms\tmedian_ms\tlaunches");

    // Upload + readback floor with no compute; every row below is this plus its kernel.
    {
        let n = 2048usize;
        let x = bytes_of(&make(n * n, 0.013, 0.9));
        let sh = dims(&[n as u64, n as u64]);
        let t = time_it(&session, || {
            let g = Graph::new(&session);
            let x = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &x).map_err(|e| e.to_string())?;
            let z = x.sum(1).map_err(|e| e.to_string())?;
            z.to_vec_f32().map_err(|e| e.to_string())?;
            Ok(())
        });
        row("passthrough_2048", &t);
    }

    {
        let n = 2048usize;
        let a = bytes_of(&make(n * n, 0.013, 0.5));
        let b = bytes_of(&make(n * n, 0.017, 0.5));
        let sh = dims(&[n as u64, n as u64]);
        let t = time_it(&session, || {
            let g = Graph::new(&session);
            let a = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &a).map_err(|e| e.to_string())?;
            let b = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &b).map_err(|e| e.to_string())?;
            let c = a
                .matmul(&b)
                .map_err(|e| e.to_string())?
                .sum(1)
                .map_err(|e| e.to_string())?;
            c.to_vec_f32().map_err(|e| e.to_string())?;
            Ok(())
        });
        row("matmul_2048", &t);
    }

    {
        let n = 2048usize;
        let a = bytes_of(&make(n * n, 0.013, 0.5));
        let b = bytes_of(&make(n * n, 0.017, 0.5));
        let bias = bytes_of(&make(n * n, 0.023, 0.1));
        let sh = dims(&[n as u64, n as u64]);
        let t = time_it(&session, || {
            let g = Graph::new(&session);
            let a = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &a).map_err(|e| e.to_string())?;
            let b = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &b).map_err(|e| e.to_string())?;
            let bs =
                Tensor::from_slice(g.handle(), Dtype::F32, &sh, &bias).map_err(|e| e.to_string())?;
            let c = a
                .matmul(&b)
                .map_err(|e| e.to_string())?
                .add(&bs)
                .map_err(|e| e.to_string())?
                .tanh()
                .map_err(|e| e.to_string())?
                .sum(1)
                .map_err(|e| e.to_string())?;
            c.to_vec_f32().map_err(|e| e.to_string())?;
            Ok(())
        });
        row("matmul_epilogue_2048", &t);
    }

    {
        let n = 2048usize;
        let x = bytes_of(&make(n * n, 0.013, 0.9));
        let y = bytes_of(&make(n * n, 0.017, 0.9));
        let sh = dims(&[n as u64, n as u64]);
        let t = time_it(&session, || {
            let g = Graph::new(&session);
            let x = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &x).map_err(|e| e.to_string())?;
            let y = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &y).map_err(|e| e.to_string())?;
            let z = chain(&x, &y)?.sum(1).map_err(|e| e.to_string())?;
            z.to_vec_f32().map_err(|e| e.to_string())?;
            Ok(())
        });
        row("elementwise_chain_2048", &t);
    }

    {
        let n = 2048usize;
        let x = bytes_of(&make(n * n, 0.013, 2.0));
        let sh = dims(&[n as u64, n as u64]);
        let t = time_it(&session, || {
            let g = Graph::new(&session);
            let x = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &x).map_err(|e| e.to_string())?;
            let z = x
                .softmax_last_dim()
                .map_err(|e| e.to_string())?
                .sum(1)
                .map_err(|e| e.to_string())?;
            z.to_vec_f32().map_err(|e| e.to_string())?;
            Ok(())
        });
        row("softmax_2048", &t);
    }

    {
        let n = 2048usize;
        let x = bytes_of(&make(n * n, 0.013, 1.0));
        let w = bytes_of(&make(n, 0.031, 1.0));
        let sh = dims(&[n as u64, n as u64]);
        let wsh = dims(&[n as u64]);
        let t = time_it(&session, || {
            let g = Graph::new(&session);
            let x = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &x).map_err(|e| e.to_string())?;
            let w =
                Tensor::from_slice(g.handle(), Dtype::F32, &wsh, &w).map_err(|e| e.to_string())?;
            let z = x
                .rms_norm(&w, 1e-5)
                .map_err(|e| e.to_string())?
                .sum(1)
                .map_err(|e| e.to_string())?;
            z.to_vec_f32().map_err(|e| e.to_string())?;
            Ok(())
        });
        row("rms_norm_2048", &t);
    }

    {
        let shape = [1u64, 8, 1024, 64];
        let numel: usize = shape.iter().product::<u64>() as usize;
        let q = bytes_of(&make(numel, 0.013, 0.1));
        let k = bytes_of(&make(numel, 0.017, 0.7));
        let v = bytes_of(&make(numel, 0.019, 1.3));
        let sh = dims(&shape);
        let scale = 1.0f32 / (64f32).sqrt();
        let t = time_it(&session, || {
            let g = Graph::new(&session);
            let q = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &q).map_err(|e| e.to_string())?;
            let k = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &k).map_err(|e| e.to_string())?;
            let v = Tensor::from_slice(g.handle(), Dtype::F32, &sh, &v).map_err(|e| e.to_string())?;
            let o = fusor2::composite::attention(&q, &k, &v, MaskKind::None, Some(scale))
                .map_err(|e| e.to_string())?
                .sum(3)
                .map_err(|e| e.to_string())?;
            o.to_vec_f32().map_err(|e| e.to_string())?;
            Ok(())
        });
        row("attention_1x8x1024x64", &t);
    }

    {
        use fusor2_ir::dtype::{QFmt, QLayout};
        let fmt = QFmt::Q4K;
        let be = fmt.block_elements() as u64;
        let (k, n, m) = (be * 16, 4096u64, 256u64);
        let blocks = (k / be) * n;
        let bytes = vec![0x11u8; (blocks * u64::from(fmt.block_bytes(QLayout::Native))) as usize];
        let act = bytes_of(&make((m * k) as usize, 0.013, 0.5));
        let t = time_it(&session, || {
            let g = Graph::new(&session);
            let w = g
                .quantized(fmt, QLayout::Native, [Dim::Const(n), Dim::Const(k)], &bytes)
                .map_err(|e| e.to_string())?;
            let a = Tensor::from_slice(
                g.handle(),
                Dtype::F32,
                &dims(&[m, k]),
                &act,
            )
            .map_err(|e| e.to_string())?;
            let y = a.matmul_t(&w).map_err(|e| e.to_string())?;
            let s = y.sum(1).map_err(|e| e.to_string())?;
            s.to_vec_f32().map_err(|e| e.to_string())?;
            Ok(())
        });
        row("qmatmul_q4k_256x4096x4096", &t);
    }

    // Quantized matvec at M=1, the LLM decode shape.
    {
        use fusor2_ir::dtype::{QFmt, QLayout};
        let fmt = QFmt::Q4K;
        let be = fmt.block_elements() as u64;
        let (k, n, m) = (be * 16, 4096u64, 1u64);
        let blocks = (k / be) * n;
        let bytes = vec![0x11u8; (blocks * u64::from(fmt.block_bytes(QLayout::Native))) as usize];
        let act = bytes_of(&make((m * k) as usize, 0.013, 0.5));
        let t = time_it(&session, || {
            let g = Graph::new(&session);
            let w = g
                .quantized(fmt, QLayout::Native, [Dim::Const(n), Dim::Const(k)], &bytes)
                .map_err(|e| e.to_string())?;
            let a = Tensor::from_slice(
                g.handle(),
                Dtype::F32,
                &dims(&[m, k]),
                &act,
            )
            .map_err(|e| e.to_string())?;
            let y = a.matmul_t(&w).map_err(|e| e.to_string())?;
            let s = y.sum(1).map_err(|e| e.to_string())?;
            s.to_vec_f32().map_err(|e| e.to_string())?;
            Ok(())
        });
        row("qmatmul_q4k_1x4096x4096", &t);
    }
}
