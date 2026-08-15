//! GPU kernel-span bench for the 8B-decode quantized matvec shapes.
//!
//! Measures the qgemv launch's own GPU timestamp span — not wall clock — for
//! each decode-plan matmul shape, Q4K and Q6K, and prints effective GB/s
//! computed from the quantized byte volume the kernel actually reads (the
//! stored block stream; the f32 activation row is <0.1% of it and excluded on
//! both sides of the comparison).
//!
//! The reference column comes from the companion example in the fusor-ml
//! workspace: `fusor-ml/core/examples/qgemv_bench.rs` (run with
//! `FUSOR_TRACE_GPU_KERNELS=1`). Both examples build bit-identical weights
//! and activations, so the printed `y[0..2]` must agree across the two.
//!
//! Run: FUSOR2_TUNE_CACHE=/tmp/qgemv_bench_tune.json \
//!      cargo run --release -p fusor2 --example qgemv_bench
//!
//! Protocol per shape: one cold resolve (saturate + extract + race), then the
//! per-dispatch timestamp path is armed on the launcher and every steady-state
//! replay reports each launch's GPU span; the table keeps the element-wise
//! minimum across iterations (a slow sample is contention, a fast one is the
//! kernel).

use fusor2::device::Device;
use fusor2::quantized::QMatrix;
use fusor2::session::Backend;
use fusor2::{Dim, Dtype};
use fusor2_ir::dtype::{QFmt, QLayout};

/// One Q4_K native block: 256 elements in 144 bytes.
/// d = 0.125 (f16 0x3000), dmin = 0, scales/mins mild, quants a ramp.
fn q4k_bytes(rows: u64, cols: u64) -> Vec<u8> {
    let blocks = rows * cols / 256;
    let mut template = [0u8; 144];
    template[0..2].copy_from_slice(&0x3000u16.to_le_bytes()); // d
    template[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin
    for i in 0..12 {
        template[4 + i] = 0x11; // small packed 6-bit scales
    }
    for i in 0..128 {
        template[16 + i] = (i as u8).wrapping_mul(37);
    }
    let mut out = Vec::with_capacity((blocks * 144) as usize);
    for _ in 0..blocks {
        out.extend_from_slice(&template);
    }
    out
}

/// One Q6_K native block: 256 elements in 210 bytes.
/// ql[128] | qh[64] | scales[16] | d(f16).
fn q6k_bytes(rows: u64, cols: u64) -> Vec<u8> {
    let blocks = rows * cols / 256;
    let mut template = [0u8; 210];
    for i in 0..128 {
        template[i] = (i as u8).wrapping_mul(29);
    }
    for i in 0..64 {
        template[128 + i] = (i as u8).wrapping_mul(53);
    }
    for i in 0..16 {
        template[192 + i] = 3; // small signed scales
    }
    template[208..210].copy_from_slice(&0x3000u16.to_le_bytes()); // d
    let mut out = Vec::with_capacity((blocks * 210) as usize);
    for _ in 0..blocks {
        out.extend_from_slice(&template);
    }
    out
}

fn main() -> fusor2::Result<()> {
    // The decode plan's matmuls: W [rows, cols], y[1, rows] = x[1, cols] * W^T.
    let shapes: &[(&str, u64, u64, QFmt)] = &[
        ("attn_4096x4096_q4k", 4096, 4096, QFmt::Q4K),
        ("gateup_14336x4096_q4k", 14336, 4096, QFmt::Q4K),
        ("down_4096x14336_q4k", 4096, 14336, QFmt::Q4K),
        ("attn_4096x4096_q6k", 4096, 4096, QFmt::Q6K),
        ("gateup_14336x4096_q6k", 14336, 4096, QFmt::Q6K),
        ("down_4096x14336_q6k", 4096, 14336, QFmt::Q6K),
    ];
    println!("shape\tlaunches\tmin_us\tmedian_us\tGB/s\tspans_us");
    for (name, rows, cols, fmt) in shapes {
        let device = Device::gpu_blocking()?;
        let graph = device.graph().clone();
        let bytes = match fmt {
            QFmt::Q6K => q6k_bytes(*rows, *cols),
            _ => q4k_bytes(*rows, *cols),
        };
        let w = QMatrix::from_raw_bytes(
            &graph,
            *fmt,
            QLayout::Native,
            [Dim::Const(*rows), Dim::Const(*cols)],
            &bytes,
        )?;
        let x = graph.leaf("x", &[Dim::Const(1), Dim::Const(*cols)], Dtype::F32)?;
        // Same activation formula as the fusor-ml companion, element for
        // element, so the printed y values must agree across implementations.
        let xdata: Vec<f32> = (0..*cols).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();
        x.set_bytes(xdata.iter().flat_map(|f| f.to_le_bytes()).collect())?;

        let y = w.q_mat_mul(&x)?;
        let session = device.session();
        // Cold resolve: saturate + extract + (race, when the cache is cold).
        let v = y.to_vec_f32()?;
        eprintln!("[bench] {name} y[0..2] = {:?}", &v[..2.min(v.len())]);

        // Steady state on the per-dispatch timestamp path. Fresh bytes each
        // iteration so the leaf really rebinds and the resolve is a replay,
        // not a memoized readback.
        let Backend::Gpu(target) = session.device() else {
            unreachable!("gpu_blocking returned a non-gpu device");
        };
        let mut xbytes: Vec<u8> = xdata.iter().flat_map(|f| f.to_le_bytes()).collect();
        target.launcher().set_tuning(true);
        let iters = 50u32;
        let mut spans: Option<Vec<f64>> = None;
        let mut totals: Vec<f64> = Vec::with_capacity(iters as usize);
        for i in 0..iters {
            xbytes[0] = i as u8;
            x.set_bytes(xbytes.clone())?;
            // A resolved root's device buffer is the value cache; drop it so
            // the resolve re-dispatches instead of early-returning (the same
            // invalidation the KV cache does after adoption).
            y.clear_device_buf();
            session.resolve(std::slice::from_ref(&y))?;
            session.wait()?;
            if let Some(us) = target.launcher().take_last_profile() {
                totals.push(us.iter().sum());
                // Element-wise minimum: a slow sample is contention, a fast
                // one is the kernel.
                spans = Some(match spans {
                    Some(prev) if prev.len() == us.len() => {
                        prev.iter().zip(&us).map(|(a, b)| a.min(*b)).collect()
                    }
                    _ => us,
                });
            }
        }
        target.launcher().set_tuning(false);
        let spans = spans.expect("no gpu timestamps — adapter has no kernel timer");
        let kernel_us: f64 = spans.iter().sum();
        // The median shows what the online explorer's substitutions cost; the
        // min is the incumbent (or the best explored variant) alone.
        totals.sort_by(f64::total_cmp);
        let median_us = totals[totals.len() / 2];
        let qbytes = *rows * *cols * u64::from(fmt.block_bytes(QLayout::Native))
            / u64::from(fmt.block_elements());
        println!(
            "{name}\t{}\t{kernel_us:.1}\t{median_us:.1}\t{:.1}\t{:?}",
            spans.len(),
            qbytes as f64 / kernel_us / 1e3,
            spans.iter().map(|s| (s * 10.0).round() / 10.0).collect::<Vec<_>>(),
        );
    }
    Ok(())
}
