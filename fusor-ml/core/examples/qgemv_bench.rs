//! Reference column for fusor2's `qgemv_bench`: the same 8B-decode quantized
//! matvec shapes through fusor-ml's qgemv, timed by GPU kernel timestamps.
//!
//! Weights and activations are bit-identical to the fusor2 side
//! (`fusor2-verified/fusor2/examples/qgemv_bench.rs`), so the printed
//! `y[0..2]` must agree across the two.
//!
//! Run: FUSOR_TRACE_GPU_KERNELS=1 \
//!      cargo run --release -p fusor-core --example qgemv_bench
//!
//! GB/s is computed from the quantized byte volume as stored on the GPU:
//! Q4K keeps its native 144 B/block layout (Metal has shader f16), Q6K is
//! always re-packed to the 212 B/block f32-scale layout
//! (`qmatrix_storage_layout_for_parts`, core/src/quantized/mod.rs).
//!
//! Replayed resolves skip profiling in fusor-ml, so every iteration drains
//! `take_kernel_profiles` and the table keeps the fastest profiled sample;
//! when only the cold resolve is profiled, that is the sample reported.

use fusor_core::{Device, QMatrix, Tensor};
use fusor_gguf::GgmlType;
use pollster::block_on;

/// One Q4_K native block: 256 elements in 144 bytes. Identical to fusor2 side.
fn q4k_bytes(rows: u64, cols: u64) -> Vec<u8> {
    let blocks = rows * cols / 256;
    let mut template = [0u8; 144];
    template[0..2].copy_from_slice(&0x3000u16.to_le_bytes()); // d
    template[2..4].copy_from_slice(&0x0000u16.to_le_bytes()); // dmin
    for i in 0..12 {
        template[4 + i] = 0x11;
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

/// One Q6_K native block: 256 elements in 210 bytes. Identical to fusor2 side.
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
        template[192 + i] = 3;
    }
    template[208..210].copy_from_slice(&0x3000u16.to_le_bytes()); // d
    let mut out = Vec::with_capacity((blocks * 210) as usize);
    for _ in 0..blocks {
        out.extend_from_slice(&template);
    }
    out
}

fn main() {
    if std::env::var_os("FUSOR_TRACE_GPU_KERNELS").is_none() {
        eprintln!("run with FUSOR_TRACE_GPU_KERNELS=1 — there is no kernel span without it");
        std::process::exit(2);
    }
    // W [rows, cols], y[1, rows] = x[1, cols] * W^T; stored GPU bytes/block
    // per `qmatrix_storage_layout_for_parts`.
    let shapes: &[(&str, u64, u64, GgmlType, u64)] = &[
        ("attn_4096x4096_q4k", 4096, 4096, GgmlType::Q4K, 144),
        ("gateup_14336x4096_q4k", 14336, 4096, GgmlType::Q4K, 144),
        ("down_4096x14336_q4k", 4096, 14336, GgmlType::Q4K, 144),
        ("attn_4096x4096_q6k", 4096, 4096, GgmlType::Q6K, 212),
        ("gateup_14336x4096_q6k", 14336, 4096, GgmlType::Q6K, 212),
        ("down_4096x14336_q6k", 4096, 14336, GgmlType::Q6K, 212),
    ];
    println!("shape\tkernels\tkernel_us\tGB/s\tsamples\tkernel");
    for (name, rows, cols, ty, block_bytes) in shapes {
        let bytes = match ty {
            GgmlType::Q6K => q6k_bytes(*rows, *cols),
            _ => q4k_bytes(*rows, *cols),
        };
        let xdata: Vec<f32> = (0..*cols).map(|i| ((i % 97) as f32) * 0.01 - 0.5).collect();

        let mut best_us = f64::INFINITY;
        let mut samples = 0usize;
        let mut kernels = 0usize;
        let mut kernel_name = String::new();
        // Replayed resolves skip profiling, so one device yields exactly two
        // profiled samples (the plain resolve and the recording one). A fresh
        // device gets a fresh flush-plan cache; cycle devices for more
        // samples. Kernel recompiles are host-side and never in the GPU span.
        for round in 0..4u32 {
            let device = block_on(Device::new()).expect("gpu");
            let w = QMatrix::from_parts(
                &device,
                &bytes,
                vec![*rows as usize, *cols as usize].into_boxed_slice(),
                *ty,
            )
            .expect("qmatrix");
            let mut fold = |profiles: Vec<fusor_core::KernelProfile>| {
                for p in profiles {
                    if p.unmeasured_kernels > 0 {
                        continue; // a partially sampled resolve is not a span
                    }
                    samples += 1;
                    kernels = p.kernels;
                    let us = p.accounted_ms * 1000.0;
                    if us < best_us {
                        best_us = us;
                        if let Some(row) = p.top_names.first() {
                            kernel_name = row.name.clone();
                        }
                    }
                }
            };
            let mut xi = xdata.clone();
            for i in 0..8u32 {
                if i > 0 {
                    xi[0] = (i % 8) as f32 * 0.01;
                }
                let x = Tensor::from_slice::<f32>(&device, [1, *cols as usize], &xi);
                let y = x.q_mat_mul(&w);
                let v = block_on(y.as_slice::<2, f32>()).expect("readback");
                if round == 0 && i == 0 {
                    eprintln!("[bench] {name} y[0..2] = [{:?}, {:?}]", v[[0, 0]], v[[0, 1]]);
                }
                fold(device.take_kernel_profiles());
            }
        }
        assert!(samples > 0, "no profiled resolve for {name}");
        let qbytes = *rows * *cols / 256 * block_bytes;
        println!(
            "{name}\t{kernels}\t{best_us:.1}\t{:.1}\t{samples}\t{kernel_name}",
            qbytes as f64 / best_us / 1e3,
        );
    }
}
