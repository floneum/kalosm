//! Part 5: cross-backend differential. Same data, same call, CPU vs GPU vs a
//! host reference. No probe touches the e-graph; `q/k/v` are never read back
//! before the attention is built.

use fusor2::composite::attention;
use fusor2::{Session, };
use fusor2::tensor::Dyn as Tensor;
use fusor2_conformance::harness::{dims, is_gpu, sessions};
use fusor2_ir::ir::launch::MaskKind;

fn data(shape: &[u64], seed: u32) -> Vec<f32> {
    let n: usize = shape.iter().product::<u64>() as usize;
    (0..n)
        .map(|i| {
            (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32) / 500.0 - 1.0
        })
        .collect()
}

fn upload(g: &fusor2::graph::GraphRef, shape: &[u64], d: &[f32]) -> Tensor {
    fusor2_conformance::harness::from_f32(g, &dims(shape), d).unwrap()
}

fn host(qd: &[f32], kd: &[f32], vd: &[f32], b: usize, h: usize, l: usize, kk: usize, d: usize) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let mut out = vec![0.0f32; b * h * l * d];
    for bi in 0..b {
        for hi in 0..h {
            for qi in 0..l {
                let mut s = vec![0.0f32; kk];
                for ki in 0..kk {
                    let mut acc = 0.0f32;
                    for dd in 0..d {
                        acc += qd[((bi * h + hi) * l + qi) * d + dd]
                            * kd[((bi * h + hi) * kk + ki) * d + dd];
                    }
                    s[ki] = acc * scale;
                }
                let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for x in s.iter_mut() {
                    *x = (*x - m).exp();
                    sum += *x;
                }
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for ki in 0..kk {
                        acc += (s[ki] / sum) * vd[((bi * h + hi) * kk + ki) * d + dd];
                    }
                    out[((bi * h + hi) * l + qi) * d + dd] = acc;
                }
            }
        }
    }
    out
}

fn once(session: &Session, b: u64, h: u64, lq: u64, lk: u64, dh: u64) -> Option<Vec<f32>> {
    let gr = fusor2::graph::Graph::new(session);
    let g = gr.handle();
    let q = upload(&g, &[b, h, lq, dh], &data(&[b, h, lq, dh], 1));
    let k = upload(&g, &[b, h, lk, dh], &data(&[b, h, lk, dh], 2));
    let v = upload(&g, &[b, h, lk, dh], &data(&[b, h, lk, dh], 3));
    let out = attention(&q, &k, &v, MaskKind::None, None).ok()?;
    match out.to_vec_f32() {
        Ok(x) => Some(x),
        Err(e) => {
            println!("    resolve error: {e}");
            None
        }
    }
}

fn maxdiff(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut w = 0.0f32;
    let mut at = 0;
    for i in 0..a.len().min(b.len()) {
        let d = (a[i] - b[i]).abs();
        if d > w {
            w = d;
            at = i;
        }
    }
    (w, at)
}

fn main() {
    let ss = sessions();
    let cpu = ss.iter().find(|s| !is_gpu(s));
    let gpu = ss.iter().find(|s| is_gpu(s));
    println!("{:>26}  {:>12}  {:>12}  {:>12}", "shape", "cpu-vs-host", "gpu-vs-host", "gpu-vs-cpu");
    for (b, h, lq, lk, dh) in [
        (1u64, 1u64, 8u64, 8u64, 4u64),
        (1, 1, 64, 64, 4),
        (1, 2, 64, 64, 4),
        (1, 8, 64, 64, 4),
        (1, 8, 128, 128, 4),
        (1, 1, 128, 128, 4),
        (1, 1, 128, 128, 16),
        (2, 2, 8, 8, 32),
        (1, 4, 32, 32, 32),
    ] {
        let hd = host(
            &data(&[b, h, lq, dh], 1),
            &data(&[b, h, lk, dh], 2),
            &data(&[b, h, lk, dh], 3),
            b as usize,
            h as usize,
            lq as usize,
            lk as usize,
            dh as usize,
        );
        let c = cpu.and_then(|s| once(s, b, h, lq, lk, dh));
        let gp = gpu.and_then(|s| once(s, b, h, lq, lk, dh));
        let f = |x: &Option<Vec<f32>>, y: &[f32]| match x {
            Some(v) => format!("{:.2e}", maxdiff(v, y).0),
            None => "ERR".into(),
        };
        let gc = match (&c, &gp) {
            (Some(a), Some(bb)) => format!("{:.2e}", maxdiff(a, bb).0),
            _ => "ERR".into(),
        };
        println!(
            "{:>26}  {:>12}  {:>12}  {:>12}",
            format!("[{b},{h},{lq},{lk}]x{dh}"),
            f(&c, &hd),
            f(&gp, &hd),
            gc
        );
    }
}
