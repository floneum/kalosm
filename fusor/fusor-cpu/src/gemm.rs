//! Optimized dense contractions for the CPU backend.

use fusor_ir::Result;
use fusor_ir::error::Error;

use crate::emit::RawBuf;

#[derive(Clone, Debug)]
pub struct ContractSpec {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub batch: u32,
    pub out: usize,
    pub a: usize,
    pub a_strides: [u32; 3],
    pub bias: Option<(usize, [u32; 3])>,
    pub b: usize,
    pub b_strides: [u32; 3],
}

impl ContractSpec {
    pub fn parse(name: &str) -> Option<Self> {
        let (encoded, gelu) = if let Some(encoded) = name.strip_prefix("cpu_contract_blas:") {
            (encoded, false)
        } else {
            (name.strip_prefix("cpu_contract_gelu_blas:")?, true)
        };
        let values: Vec<u32> = encoded
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        if values.len() != if gelu { 17 } else { 13 } {
            return None;
        }
        let (bias, b_offset) = if gelu {
            (
                Some((values[9] as usize, [values[10], values[11], values[12]])),
                13,
            )
        } else {
            (None, 9)
        };
        Some(Self {
            m: values[0],
            n: values[1],
            k: values[2],
            batch: values[3],
            out: values[4] as usize,
            a: values[5] as usize,
            a_strides: [values[6], values[7], values[8]],
            bias,
            b: values[b_offset] as usize,
            b_strides: [
                values[b_offset + 1],
                values[b_offset + 2],
                values[b_offset + 3],
            ],
        })
    }
}

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

#[cfg(target_os = "macos")]
pub(crate) fn run(spec: &ContractSpec, bufs: &[RawBuf]) -> Result<()> {
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    let a = bufs
        .get(spec.a)
        .ok_or_else(|| Error::Device("GEMM A binding is missing".into()))?;
    let b = bufs
        .get(spec.b)
        .ok_or_else(|| Error::Device("GEMM B binding is missing".into()))?;
    let out = bufs
        .get(spec.out)
        .ok_or_else(|| Error::Device("GEMM output binding is missing".into()))?;
    let mut materialized = Vec::new();
    let (a_ptr, a_strides) = if let Some((bias_binding, bias_strides)) = spec.bias {
        let bias = bufs
            .get(bias_binding)
            .ok_or_else(|| Error::Device("GEMM bias binding is missing".into()))?;
        materialized.resize(
            spec.batch as usize * spec.m as usize * spec.k as usize,
            0.0f32,
        );
        if spec.a_strides[2] != 1 || bias_strides[2] != 1 {
            return Err(Error::Device(
                "the Cranelift GELU prepass requires unit-stride contraction rows".into(),
            ));
        }
        let a_len = a.bytes / std::mem::size_of::<f32>();
        let bias_len = bias.bytes / std::mem::size_of::<f32>();
        let depth = spec.k as usize;
        for batch in 0..spec.batch as usize {
            for row in 0..spec.m as usize {
                let ai = batch * spec.a_strides[0] as usize + row * spec.a_strides[1] as usize;
                let bi = batch * bias_strides[0] as usize + row * bias_strides[1] as usize;
                if ai + depth > a_len || bi + depth > bias_len {
                    return Err(Error::Device(
                        "a fused-GELU row exceeds one of its bindings".into(),
                    ));
                }
                let a = unsafe { std::slice::from_raw_parts((a.ptr as *const f32).add(ai), depth) };
                let bias =
                    unsafe { std::slice::from_raw_parts((bias.ptr as *const f32).add(bi), depth) };
                let out = &mut materialized[(batch * spec.m as usize + row) * depth..][..depth];
                crate::jit::gelu_dense(a, bias, out).map_err(Error::Device)?;
            }
        }
        (materialized.as_ptr(), [spec.m * spec.k, spec.k, 1])
    } else {
        (a.ptr as *const f32, spec.a_strides)
    };
    let broadcast_a = a_strides[1] == 0 && a_strides[2] == 1;
    let (trans_a, lda) = if a_strides[2] == 1 {
        (NO_TRANS, a_strides[1].max(spec.k))
    } else {
        (TRANS, a_strides[2])
    };
    let broadcast_b = spec.b_strides[2] == 0 && spec.b_strides[1] == 1;
    let (trans_b, ldb) = if broadcast_b {
        (NO_TRANS, 1)
    } else if spec.b_strides[2] == 1 {
        (NO_TRANS, spec.b_strides[1])
    } else {
        (TRANS, spec.b_strides[2])
    };

    for batch in 0..spec.batch {
        let a_offset = batch as usize * a_strides[0] as usize;
        let b_offset = batch as usize * spec.b_strides[0] as usize;
        let c_offset = batch as usize * spec.m as usize * spec.n as usize;
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                trans_a,
                trans_b,
                if broadcast_a { 1 } else { spec.m as i32 },
                if broadcast_b { 1 } else { spec.n as i32 },
                spec.k as i32,
                1.0,
                a_ptr.add(a_offset),
                lda as i32,
                (b.ptr as *const f32).add(b_offset),
                ldb as i32,
                0.0,
                (out.ptr as *mut f32).add(c_offset),
                spec.n as i32,
            );
            let out_ptr = (out.ptr as *mut f32).add(c_offset);
            if broadcast_b {
                let rows = if broadcast_a { 1 } else { spec.m as usize };
                for row in 0..rows {
                    let value = *out_ptr.add(row * spec.n as usize);
                    std::slice::from_raw_parts_mut(
                        out_ptr.add(row * spec.n as usize),
                        spec.n as usize,
                    )
                    .fill(value);
                }
            }
            if broadcast_a {
                let first = (out.ptr as *const f32).add(c_offset);
                for row in 1..spec.m as usize {
                    std::ptr::copy_nonoverlapping(
                        first,
                        (out.ptr as *mut f32).add(c_offset + row * spec.n as usize),
                        spec.n as usize,
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub(crate) fn run(_spec: &ContractSpec, _bufs: &[RawBuf]) -> Result<()> {
    Err(Error::Device(
        "the optimized GEMM path currently requires Accelerate".into(),
    ))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::jit::gelu_dense;

    #[test]
    fn batched_wide_gemm_matches_reference() {
        let (batch, m, n, k) = (2usize, 3usize, 65usize, 5usize);
        let a: Vec<f32> = (0..batch * k)
            .map(|i| (i as f32 % 11.0) * 0.1 - 0.5)
            .collect();
        let b: Vec<f32> = (0..batch * k)
            .map(|i| (i as f32 % 17.0) * 0.05 - 0.4)
            .collect();
        let mut out = vec![0.0f32; batch * m * n];
        let raw = |values: &[f32]| RawBuf {
            ptr: values.as_ptr() as *mut u8,
            bytes: std::mem::size_of_val(values),
        };
        let out_bytes = std::mem::size_of_val(out.as_slice());
        let bufs = [
            raw(&a),
            raw(&b),
            RawBuf {
                ptr: out.as_mut_ptr().cast(),
                bytes: out_bytes,
            },
        ];
        run(
            &ContractSpec {
                m: m as u32,
                n: n as u32,
                k: k as u32,
                batch: batch as u32,
                out: 2,
                a: 0,
                a_strides: [k as u32, 0, 1],
                bias: None,
                b: 1,
                b_strides: [k as u32, 1, 0],
            },
            &bufs,
        )
        .unwrap();
        for bi in 0..batch {
            for row in 0..m {
                for col in 0..n {
                    let want: f32 = (0..k)
                        .map(|depth| a[bi * k + depth] * b[bi * k + depth])
                        .sum();
                    let got = out[(bi * m + row) * n + col];
                    assert!((got - want).abs() < 1e-5, "[{bi},{row},{col}]");
                }
            }
        }
    }

    #[test]
    fn cranelift_gelu_tracks_the_tanh_definition() {
        let values: Vec<f32> = (-100..=100).map(|i| i as f32 / 10.0).collect();
        let bias = vec![0.0; values.len()];
        let mut out = vec![0.0; values.len()];
        gelu_dense(&values, &bias, &mut out).unwrap();
        for (&x, &got) in values.iter().zip(&out) {
            let inner = 0.797_884_56 * (x + 0.044_715 * x * x * x);
            let expected = 0.5 * x * (1.0 + inner.tanh());
            assert!((got - expected).abs() < 2.0e-4, "x={x}");
        }
    }
}
