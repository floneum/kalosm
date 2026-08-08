//! Builds `attention(q,k,v)` and reads it back through the public API, with no
//! e-graph access, and compares against a host reference.

use fusor2::composite::attention::attention;
use fusor2::{Session, Tensor};
use fusor2_conformance::harness::{dims, is_gpu, sessions};
use fusor2_ir::ir::level1::MaskKind;

fn upload(g: &fusor2::graph::GraphRef, shape: &[u64], seed: u32) -> Tensor {
    let n: usize = shape.iter().product::<u64>() as usize;
    let data: Vec<f32> = (0..n)
        .map(|i| {
            (((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32) / 500.0 - 1.0
        })
        .collect();
    fusor2_conformance::harness::from_f32(g, &dims(shape), &data).unwrap()
}

fn host(
    qd: &[f32],
    kd: &[f32],
    vd: &[f32],
    b: usize,
    h: usize,
    l: usize,
    kk: usize,
    d: usize,
) -> Vec<f32> {
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

fn run(session: &Session, b: u64, h: u64, lq: u64, lk: u64, dh: u64) {
    let backend = if is_gpu(session) { "gpu" } else { "cpu" };
    let gr = fusor2::graph::Graph::new(session);
    let g = gr.handle();
    let q = upload(&g, &[b, h, lq, dh], 1);
    let k = upload(&g, &[b, h, lk, dh], 2);
    let v = upload(&g, &[b, h, lk, dh], 3);
    let qd = q.to_vec_f32().unwrap();
    let kd = k.to_vec_f32().unwrap();
    let vd = v.to_vec_f32().unwrap();
    let out = match attention(&q, &k, &v, MaskKind::None, None) {
        Ok(o) => o,
        Err(e) => {
            println!("[{backend}] B{b} H{h} Lq{lq} Lk{lk} Dh{dh}: BUILD ERR {e}");
            return;
        }
    };
    match out.to_vec_f32() {
        Ok(got) => {
            let e = host(
                &qd, &kd, &vd, b as usize, h as usize, lq as usize, lk as usize, dh as usize,
            );
            let mut worst = 0.0f32;
            let mut at = 0;
            for i in 0..e.len() {
                let d = (got[i] - e[i]).abs();
                if d > worst {
                    worst = d;
                    at = i;
                }
            }
            let verdict = if worst > 1e-3 { "  <<< WRONG" } else { "" };
            println!(
                "[{backend}] B{b} H{h} Lq{lq} Lk{lk} Dh{dh}: max|err| = {worst:.3e} at {at} (got {} want {}){verdict}",
                got[at], e[at]
            );
        }
        Err(err) => println!(
            "[{backend}] B{b} H{h} Lq{lq} Lk{lk} Dh{dh}: RESOLVE ERROR: {err}   <<< COMPILE FAILURE"
        ),
    }
}

fn main() {
    for s in sessions() {
        for (b, h, lq, lk) in [(2u64, 2u64, 8u64, 8u64), (1, 8, 128, 128)] {
            for dh in [4u64, 16, 32, 48, 64, 96, 128] {
                run(&s, b, h, lq, lk, dh);
            }
        }
    }
}
