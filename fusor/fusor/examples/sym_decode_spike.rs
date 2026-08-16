//! Spike: a decode-shaped step against a fixed-capacity KV leaf with a
//! symbolic current length. Checks, on the real GPU:
//!   1. numeric correctness at several bindings,
//!   2. whether the second..Nth step re-saturates / re-extracts
//!      (FUSOR_RESOLVE_PROFILE=1),
//!   3. wall time per step.
//!
//! Run: FUSOR_RESOLVE_PROFILE=1 cargo run --release -p fusor --example sym_decode_spike

use fusor::composite::attention_masked;
use fusor::device::Device;
use fusor::tensor::Dyn as Tensor;
use fusor::{Dim, Dtype};
use fusor_ir::ir::launch::MaskKind;
use fusor_ir::shape::StrideSpec;

const CAP: u64 = 64; // capacity
const HKV: u64 = 2;
const HQ: u64 = 4;
const DH: u64 = 8;

fn narrow_sym(t: &Tensor, axis: usize, len: Dim) -> fusor::Result<Tensor> {
    let shape = t.shape();
    let specs: Vec<StrideSpec> = shape
        .iter()
        .copied()
        .enumerate()
        .map(|(i, d)| {
            if i == axis {
                StrideSpec::dim(i as u32, len)
            } else {
                StrideSpec::dim(i as u32, d)
            }
        })
        .collect();
    t.restride(&specs)
}

fn main() -> fusor::Result<()> {
    let device = Device::gpu_blocking()?;
    let graph = device.graph().clone();
    let session = device.session();

    // Fixed-capacity K / V leaves.
    let kshape = [
        Dim::Const(1),
        Dim::Const(HKV),
        Dim::Const(CAP),
        Dim::Const(DH),
    ];
    let k_cache = graph.leaf("k_cache", &kshape, Dtype::F32)?;
    let v_cache = graph.leaf("v_cache", &kshape, Dtype::F32)?;
    let zeros = vec![0u8; (HKV * CAP * DH * 4) as usize];
    k_cache.set_bytes(zeros.clone())?;
    v_cache.set_bytes(zeros)?;

    // The step inputs: one new k/v row and one query, plus the write position.
    let knew = graph.leaf(
        "k_new",
        &[
            Dim::Const(1),
            Dim::Const(HKV),
            Dim::Const(1),
            Dim::Const(DH),
        ],
        Dtype::F32,
    )?;
    let vnew = graph.leaf(
        "v_new",
        &[
            Dim::Const(1),
            Dim::Const(HKV),
            Dim::Const(1),
            Dim::Const(DH),
        ],
        Dtype::F32,
    )?;
    let q = graph.leaf(
        "q",
        &[Dim::Const(1), Dim::Const(HQ), Dim::Const(1), Dim::Const(DH)],
        Dtype::F32,
    )?;
    let pos = graph.leaf("pos", &[Dim::Const(1)], Dtype::U32)?;

    // The step graph: scatter the new row in at `pos`, attend over the first
    // `total` (symbolic) rows.
    let total = graph.sym("kv_total");
    let k_out = k_cache.scatter_set(2, &pos, &knew, true)?;
    let v_out = v_cache.scatter_set(2, &pos, &vnew, true)?;
    let k_att = narrow_sym(&k_out, 2, total)?;
    let v_att = narrow_sym(&v_out, 2, total)?;
    let o = attention_masked(&q, &k_att, &v_att, MaskKind::None, None, Some(1.0))?;

    // Reference on the host.
    let mut k_host = vec![0f32; (HKV * CAP * DH) as usize];
    let mut v_host = vec![0f32; (HKV * CAP * DH) as usize];

    let steps = 6u64;
    for step in 0..steps {
        // New random-ish rows for this step.
        let kv_row: Vec<f32> = (0..HKV * DH)
            .map(|i| ((i as f32) * 0.37 + step as f32 * 0.11).sin())
            .collect();
        let v_row: Vec<f32> = (0..HKV * DH)
            .map(|i| ((i as f32) * 0.13 - step as f32 * 0.07).cos())
            .collect();
        let q_row: Vec<f32> = (0..HQ * DH)
            .map(|i| ((i as f32) * 0.05 + step as f32 * 0.29).sin())
            .collect();
        knew.set_bytes(kv_row.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        vnew.set_bytes(v_row.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        q.set_bytes(q_row.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        pos.set_bytes((step as u32).to_le_bytes().to_vec())?;
        graph.bind("kv_total", step + 1);

        let t0 = std::time::Instant::now();
        session.resolve(&[o.clone(), k_out.clone(), v_out.clone()])?;
        let out = o.to_vec_f32()?;
        let dt = t0.elapsed();

        // Host reference.
        for h in 0..HKV as usize {
            for d in 0..DH as usize {
                k_host[h * (CAP * DH) as usize + step as usize * DH as usize + d] =
                    kv_row[h * DH as usize + d];
                v_host[h * (CAP * DH) as usize + step as usize * DH as usize + d] =
                    v_row[h * DH as usize + d];
            }
        }
        let groups = (HQ / HKV) as usize;
        let mut expect = vec![0f32; (HQ * DH) as usize];
        for hq in 0..HQ as usize {
            let hkv = hq / groups;
            let qv = &q_row[hq * DH as usize..(hq + 1) * DH as usize];
            let len = (step + 1) as usize;
            let mut scores = vec![0f32; len];
            for j in 0..len {
                let kv = &k_host[hkv * (CAP * DH) as usize + j * DH as usize..];
                scores[j] = qv.iter().zip(&kv[..DH as usize]).map(|(a, b)| a * b).sum();
            }
            let m = scores.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
            let denom: f32 = exps.iter().sum();
            for d in 0..DH as usize {
                let mut acc = 0f32;
                for j in 0..len {
                    let vv = v_host[hkv * (CAP * DH) as usize + j * DH as usize + d];
                    acc += exps[j] / denom * vv;
                }
                expect[hq * DH as usize + d] = acc;
            }
        }
        let max_err = out
            .iter()
            .zip(&expect)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!(
            "step {step}: len {} max_err {max_err:.2e} wall {:?}",
            step + 1,
            dt
        );
        assert!(
            max_err < 1e-4,
            "step {step} diverged: {out:?} vs {expect:?}"
        );

        // Feed the written caches back in as the next step's leaves.
        k_cache.adopt_buffer(&k_out)?;
        v_cache.adopt_buffer(&v_out)?;
        o.clear_device_buf();
        k_out.clear_device_buf();
        v_out.clear_device_buf();
    }
    println!("spike ok");
    Ok(())
}
