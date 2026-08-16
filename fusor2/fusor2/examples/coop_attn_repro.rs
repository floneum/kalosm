//! Minimal repro for the fuzzed coop-attention miscompiles: run attention at
//! odd/small shapes with the member sweep armed and report how many class
//! members computed wrong values. Exit code 1 when any did.
//!
//! Run: FUSOR2_VERIFY_MEMBERS=1 cargo run --release -p fusor2 --example coop_attn_repro

use fusor2::tensor::Dyn;
use fusor2::{Device, Dtype};

fn tensor(graph: &fusor2::graph::GraphRef, dims: &[u64], seed: u64) -> Dyn {
    let n: u64 = dims.iter().product();
    // Deterministic, mildly irregular values; nothing special about the form.
    let data: Vec<f32> = (0..n)
        .map(|i| {
            let x = ((i ^ seed) as u32).wrapping_mul(2654435761) as f32 / u32::MAX as f32;
            x * 2.0 - 1.0
        })
        .collect();
    let shape: Vec<fusor2::Dim> = dims.iter().map(|&d| fusor2::Dim::Const(d)).collect();
    Dyn::from_slice(graph, Dtype::F32, &shape, bytemuck::cast_slice(&data)).expect("leaf")
}

fn main() {
    if std::env::var_os("FUSOR2_VERIFY_MEMBERS").is_none() {
        // The sweep is the whole point; refuse to silently test nothing.
        unsafe { std::env::set_var("FUSOR2_VERIFY_MEMBERS", "1") };
    }
    let device = Device::gpu_blocking().expect("gpu");
    let graph = device.graph().handle();

    // The suite's ATTN_SPEC regime: tiny lengths and head dims, where a
    // whole problem fits inside one partial coop tile.
    let cases: &[(u64, u64, u64, u64, u64)] = &[
        // (batch, heads, q_len, kv_len, head_dim)
        (1, 1, 3, 5, 4),
        (1, 1, 7, 11, 6),
        (1, 1, 3, 5, 6),
        (1, 1, 3, 5, 8),
        (1, 1, 3, 6, 4),
        (1, 1, 3, 8, 4),
        (1, 1, 3, 9, 4),
        // Exact-fill shapes: the reader's logical element count equals a
        // padded extent (q * kv == 64 against 64-wide coop padding), so the
        // readback's exact-run parse and the padded parse are indistinguishable
        // from (shape, strides) alone. These are the residual 5 full-suite
        // miscompiles, reproduced locally.
        (1, 1, 8, 8, 8),
        (1, 1, 4, 16, 4),
        (1, 1, 16, 4, 4),
        (1, 1, 2, 32, 8),
        (1, 1, 4, 4, 4),
    ];

    for &(b, h, q_len, kv_len, hd) in cases {
        let q = tensor(&graph, &[b, h, q_len, hd], 1);
        let k = tensor(&graph, &[b, h, kv_len, hd], 2);
        let v = tensor(&graph, &[b, h, kv_len, hd], 3);
        for (name, mask) in [
            ("none", fusor2::cache::MaskKind::None),
            ("causal", fusor2::cache::MaskKind::Causal),
        ] {
            let before = fusor2::session::wrong_member_count();
            let o = fusor2::composite::attention(&q, &k, &v, mask, None)
                .expect("attention build");
            let host = o.to_vec_f32().expect("resolve");
            let wrong = fusor2::session::wrong_member_count() - before;
            println!(
                "attention b{b} h{h} q{q_len} kv{kv_len} d{hd} mask={name}: wrong_members={wrong} (out[0]={:.4})",
                host.first().copied().unwrap_or(f32::NAN)
            );
        }
    }
    let total = fusor2::session::wrong_member_count();
    println!("TOTAL_WRONG_MEMBERS: {total}");
    std::process::exit(if total > 0 { 1 } else { 0 });
}
