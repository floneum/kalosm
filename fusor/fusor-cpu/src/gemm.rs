//! Dense contractions for the CPU backend: Accelerate on macOS, a portable
//! SIMD-friendly kernel everywhere else (including wasm32), with a fused
//! elementwise transform of either operand or of the result evaluated a slice
//! at a time around the product.

use std::cell::RefCell;

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::scalar::ScalarExpr;

use crate::emit::RawBuf;
use crate::slices::{SliceOp, decode, transform};

/// MACs below which the portable kernel beats a BLAS call.
#[cfg(all(target_os = "macos", not(feature = "wasm-emit")))]
const SMALL: usize = 1 << 14;

/// `expr` as a transform of its one argument, as kernel-name text; empty for
/// the identity, None when it is not such a function.
pub(crate) fn encode(expr: &ScalarExpr) -> Option<String> {
    crate::slices::encode(expr, 1)
}

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
    /// Fused transforms of A's elements, B's elements and the result.
    pub pre_a: Vec<SliceOp>,
    pub pre_b: Vec<SliceOp>,
    pub post: Vec<SliceOp>,
}

impl ContractSpec {
    pub fn parse(name: &str) -> Option<Self> {
        let (encoded, gelu) = if let Some(encoded) = name.strip_prefix("cpu_contract_blas:") {
            (encoded, false)
        } else {
            (name.strip_prefix("cpu_contract_gelu_blas:")?, true)
        };
        let mut parts = encoded.split(';');
        let values: Vec<u32> = parts
            .next()?
            .split(',')
            .map(str::parse)
            .collect::<std::result::Result<_, _>>()
            .ok()?;
        if values.len() != if gelu { 17 } else { 13 } {
            return None;
        }
        let mut transform = || decode(parts.next().unwrap_or(""));
        let (pre_a, pre_b, post) = (transform()?, transform()?, transform()?);
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
            pre_a,
            pre_b,
            post,
        })
    }
}

thread_local! {
    /// Materialized operands of a launch with fused input transforms.
    static SIDES: RefCell<[Vec<f32>; 2]> = const { RefCell::new([Vec::new(), Vec::new()]) };
}

/// The largest element index a `[batch, rows, cols]` walk with `strides` reads.
fn reach(strides: [u32; 3], batch: u32, rows: u32, cols: u32) -> usize {
    (batch as usize - 1) * strides[0] as usize
        + (rows as usize - 1) * strides[1] as usize
        + (cols as usize - 1) * strides[2] as usize
}

pub(crate) fn run(spec: &ContractSpec, bufs: &[RawBuf]) -> Result<()> {
    let missing = |what: &str| Error::Device(format!("GEMM {what} binding is missing"));
    let a = bufs.get(spec.a).ok_or_else(|| missing("A"))?;
    let b = bufs.get(spec.b).ok_or_else(|| missing("B"))?;
    let out = bufs.get(spec.out).ok_or_else(|| missing("output"))?;
    let (batch, m, n, k) = (
        spec.batch as usize,
        spec.m as usize,
        spec.n as usize,
        spec.k as usize,
    );
    if out.bytes / 4 < batch * m * n
        || a.bytes / 4 <= reach(spec.a_strides, spec.batch, spec.m, spec.k)
        || b.bytes / 4 <= reach(spec.b_strides, spec.batch, spec.k, spec.n)
    {
        return Err(Error::Device("a GEMM operand exceeds its binding".into()));
    }
    let products =
        |a_ptr: *const f32, a_strides: [u32; 3], b_ptr: *const f32, b_strides: [u32; 3]| {
            for index in 0..batch {
                // SAFETY: the reach of every operand was checked against its
                // binding above; a materialized side is exactly its dense size.
                unsafe {
                    sgemm(
                        [m, n, k],
                        a_ptr.add(index * a_strides[0] as usize),
                        [a_strides[1] as usize, a_strides[2] as usize],
                        b_ptr.add(index * b_strides[0] as usize),
                        [b_strides[1] as usize, b_strides[2] as usize],
                        (out.ptr as *mut f32).add(index * m * n),
                    );
                }
            }
            if !spec.post.is_empty() {
                // SAFETY: the output holds `batch * m * n` elements (checked above).
                transform(&spec.post, unsafe {
                    std::slice::from_raw_parts_mut(out.ptr as *mut f32, batch * m * n)
                });
            }
        };
    if spec.bias.is_none() && spec.pre_a.is_empty() && spec.pre_b.is_empty() {
        products(
            a.ptr as *const f32,
            spec.a_strides,
            b.ptr as *const f32,
            spec.b_strides,
        );
        return Ok(());
    }
    SIDES.with(|sides| {
        let mut sides = sides.borrow_mut();
        let [side_a, side_b] = &mut *sides;
        // A side with a transform is gathered dense, transformed, and read
        // from there.
        let (a_ptr, a_strides) = if let Some((bias_binding, bias_strides)) = spec.bias {
            let bias = bufs.get(bias_binding).ok_or_else(|| missing("bias"))?;
            gelu_side(spec, a, bias, bias_strides, side_a)?;
            (side_a.as_ptr(), [spec.m * spec.k, spec.k, 1])
        } else if spec.pre_a.is_empty() {
            (a.ptr as *const f32, spec.a_strides)
        } else {
            dense(a.ptr as *const f32, spec.a_strides, [batch, m, k], side_a);
            transform(&spec.pre_a, side_a);
            (side_a.as_ptr(), [spec.m * spec.k, spec.k, 1])
        };
        let (b_ptr, b_strides) = if spec.pre_b.is_empty() {
            (b.ptr as *const f32, spec.b_strides)
        } else {
            dense(b.ptr as *const f32, spec.b_strides, [batch, k, n], side_b);
            transform(&spec.pre_b, side_b);
            (side_b.as_ptr(), [spec.k * spec.n, spec.n, 1])
        };
        products(a_ptr, a_strides, b_ptr, b_strides);
        Ok(())
    })
}

/// Gather a strided `[batch, rows, cols]` operand into `out`, dense.
fn dense(ptr: *const f32, strides: [u32; 3], [batch, rows, cols]: [usize; 3], out: &mut Vec<f32>) {
    let [by_batch, by_row, by_col] = strides.map(|s| s as usize);
    out.clear();
    out.reserve(batch * rows * cols);
    // A block already dense in memory is one copy.
    let whole = (cols == 1 || by_col == 1) && (rows == 1 || by_row == cols);
    for b in 0..batch {
        // SAFETY: the caller checked the operand's reach.
        unsafe {
            if whole {
                out.extend_from_slice(std::slice::from_raw_parts(
                    ptr.add(b * by_batch),
                    rows * cols,
                ));
                continue;
            }
            for r in 0..rows {
                let base = b * by_batch + r * by_row;
                if by_col == 1 {
                    out.extend_from_slice(std::slice::from_raw_parts(ptr.add(base), cols));
                } else {
                    out.extend((0..cols).map(|c| *ptr.add(base + c * by_col)));
                }
            }
        }
    }
}

#[cfg(target_os = "macos")]
fn gelu_side(
    spec: &ContractSpec,
    a: &RawBuf,
    bias: &RawBuf,
    bias_strides: [u32; 3],
    out: &mut Vec<f32>,
) -> Result<()> {
    if spec.a_strides[2] != 1 || bias_strides[2] != 1 {
        return Err(Error::Device(
            "the Cranelift GELU prepass requires unit-stride contraction rows".into(),
        ));
    }
    let depth = spec.k as usize;
    out.clear();
    out.resize(spec.batch as usize * spec.m as usize * depth, 0.0);
    let a_len = a.bytes / std::mem::size_of::<f32>();
    let bias_len = bias.bytes / std::mem::size_of::<f32>();
    for batch in 0..spec.batch as usize {
        for row in 0..spec.m as usize {
            let ai = batch * spec.a_strides[0] as usize + row * spec.a_strides[1] as usize;
            let bi = batch * bias_strides[0] as usize + row * bias_strides[1] as usize;
            if ai + depth > a_len || bi + depth > bias_len {
                return Err(Error::Device(
                    "a fused-GELU row exceeds one of its bindings".into(),
                ));
            }
            // SAFETY: both rows were bounds-checked just above.
            let a = unsafe { std::slice::from_raw_parts((a.ptr as *const f32).add(ai), depth) };
            let bias =
                unsafe { std::slice::from_raw_parts((bias.ptr as *const f32).add(bi), depth) };
            let out = &mut out[(batch * spec.m as usize + row) * depth..][..depth];
            crate::jit::gelu_dense(a, bias, out).map_err(Error::Device)?;
        }
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn gelu_side(
    _: &ContractSpec,
    _: &RawBuf,
    _: &RawBuf,
    _: [u32; 3],
    _: &mut Vec<f32>,
) -> Result<()> {
    Err(Error::Device(
        "the fused-GELU contraction is lowered on macOS only".into(),
    ))
}

/// `c[i, j] = sum_k a[i * a[0] + k * a[1]] * b[k * b[0] + j * b[1]]`, `c` dense.
///
/// # Safety
/// Every index the strides reach must be in bounds of its operand, and `c`
/// must hold `m * n` elements.
#[cfg(all(target_os = "macos", not(feature = "wasm-emit")))]
unsafe fn sgemm(
    [m, n, k]: [usize; 3],
    a: *const f32,
    a_strides: [usize; 2],
    b: *const f32,
    b_strides: [usize; 2],
    c: *mut f32,
) {
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
    // A small product costs less than the call into Accelerate, and so does
    // one whose left operand turns out mostly zeros.
    if m * n * k <= SMALL {
        return unsafe { sgemm_portable([m, n, k], a, a_strides, b, b_strides, c) };
    }
    if a_strides[1] == 1 && b_strides[1] == 1 && m * k <= 1 << 14 {
        let out = unsafe { std::slice::from_raw_parts_mut(c, m * n) };
        if unsafe { sparse_rows([m, n, k], a, a_strides[0], b, b_strides[0], out, m * k / 8) } {
            return;
        }
    }
    const ROW_MAJOR: i32 = 101;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;
    let broadcast_a = a_strides[0] == 0 && a_strides[1] == 1;
    let (trans_a, lda) = if a_strides[1] == 1 {
        (NO_TRANS, a_strides[0].max(k))
    } else {
        (TRANS, a_strides[1])
    };
    let broadcast_b = b_strides[1] == 0 && b_strides[0] == 1;
    let (trans_b, ldb) = if broadcast_b {
        (NO_TRANS, 1)
    } else if b_strides[1] == 1 {
        (NO_TRANS, b_strides[0])
    } else {
        (TRANS, b_strides[1])
    };
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            trans_a,
            trans_b,
            if broadcast_a { 1 } else { m as i32 },
            if broadcast_b { 1 } else { n as i32 },
            k as i32,
            1.0,
            a,
            lda as i32,
            b,
            ldb as i32,
            0.0,
            c,
            n as i32,
        );
        if broadcast_b {
            let rows = if broadcast_a { 1 } else { m };
            for row in 0..rows {
                let value = *c.add(row * n);
                std::slice::from_raw_parts_mut(c.add(row * n), n).fill(value);
            }
        }
        if broadcast_a {
            for row in 1..m {
                std::ptr::copy_nonoverlapping(c, c.add(row * n), n);
            }
        }
    }
}

/// The portable kernel: every stride pattern is correct, and the common ones
/// (a contiguous dot, a contiguous axpy over the output or over B's rows) run
/// as loops the compiler vectorizes.
#[cfg(any(not(target_os = "macos"), feature = "wasm-emit"))]
unsafe fn sgemm(
    [m, n, k]: [usize; 3],
    a: *const f32,
    a_strides: [usize; 2],
    b: *const f32,
    b_strides: [usize; 2],
    c: *mut f32,
) {
    unsafe { sgemm_portable([m, n, k], a, a_strides, b, b_strides, c) }
}

unsafe fn sgemm_portable(
    [m, n, k]: [usize; 3],
    a: *const f32,
    [ai, ak]: [usize; 2],
    b: *const f32,
    [bk, bj]: [usize; 2],
    c: *mut f32,
) {
    unsafe {
        let out = std::slice::from_raw_parts_mut(c, m * n);
        if n == 1 && ak == 1 && bk == 1 {
            matvec(a, ai, std::slice::from_raw_parts(b, k), out);
        } else if n == 1 && ai == 1 {
            // A read down its columns: one axpy over the outputs per k.
            out.fill(0.0);
            for kk in 0..k {
                let scale = *b.add(kk * bk);
                if scale != 0.0 {
                    axpy(scale, std::slice::from_raw_parts(a.add(kk * ak), m), out);
                }
            }
        } else if bj == 1 && ak == 1 {
            sparse_rows([m, n, k], a, ai, b, bk, out, usize::MAX);
        } else if bj == 1 {
            out.fill(0.0);
            for i in 0..m {
                let row = &mut out[i * n..(i + 1) * n];
                for kk in 0..k {
                    let scale = *a.add(i * ai + kk * ak);
                    if scale != 0.0 {
                        axpy(scale, std::slice::from_raw_parts(b.add(kk * bk), n), row);
                    }
                }
            }
        } else {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0f32;
                    for kk in 0..k {
                        sum += *a.add(i * ai + kk * ak) * *b.add(kk * bk + j * bj);
                    }
                    out[i * n + j] = sum;
                }
            }
        }
    }
}

/// Whether every value is (positive or negative) zero.
fn zeros<const N: usize>(values: &[f32; N]) -> bool {
    values.iter().fold(0, |bits, v| bits | v.to_bits()) << 1 == 0
}

/// `out[i, ..] = sum_k a[i * by_row + k] * b[k * by_k ..][..n]`, one axpy per
/// nonzero of `a` (zero multipliers are skipped, as the reference GEMM does,
/// and runs of them 64 and 16 at a time). Gives up, returning false, once
/// more than `budget` nonzeros are met: the operand is not sparse.
///
/// # Safety
/// `a` holds `m` rows of `k` at `by_row`, `b` holds `k` rows of `n` at `by_k`.
unsafe fn sparse_rows(
    [m, n, k]: [usize; 3],
    a: *const f32,
    by_row: usize,
    b: *const f32,
    by_k: usize,
    out: &mut [f32],
    budget: usize,
) -> bool {
    let mut met = 0;
    for i in 0..m {
        let row = &mut out[i * n..(i + 1) * n];
        row.fill(0.0);
        let mut kk = 0;
        while kk < k {
            unsafe {
                let at = a.add(i * by_row + kk);
                if kk + 64 <= k && zeros(&*(at as *const [f32; 64])) {
                    kk += 64;
                    continue;
                }
                if kk + 16 <= k && zeros(&*(at as *const [f32; 16])) {
                    kk += 16;
                    continue;
                }
                if *at != 0.0 {
                    met += 1;
                    if met > budget {
                        return false;
                    }
                    axpy(*at, std::slice::from_raw_parts(b.add(kk * by_k), n), row);
                }
            }
            kk += 1;
        }
    }
    true
}

/// `out[i] = a[i * stride ..][..x.len()] . x`, eight rows at a time so their
/// sums advance in parallel instead of each waiting on its own adds.
///
/// # Safety
/// Every row must be in bounds of `a`.
unsafe fn matvec(a: *const f32, stride: usize, x: &[f32], out: &mut [f32]) {
    const ROWS: usize = 8;
    const LANES: usize = 8;
    let k = x.len();
    let whole = k / LANES * LANES;
    let (groups, rest) = out.as_chunks_mut::<ROWS>();
    let done = groups.len() * ROWS;
    for (g, group) in groups.iter_mut().enumerate() {
        let rows: [*const f32; ROWS] =
            std::array::from_fn(|r| unsafe { a.add((g * ROWS + r) * stride) });
        let mut acc = [[0f32; LANES]; ROWS];
        // SAFETY: every chunk read is `at + LANES <= k`, within `x` and every row.
        #[cfg(target_arch = "aarch64")]
        unsafe {
            // Fused multiply-adds, two four-lane sums per row.
            use core::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};
            let mut sums = [[vdupq_n_f32(0.); 2]; ROWS];
            for at in (0..whole).step_by(LANES) {
                let (x0, x1) = (
                    vld1q_f32(x.as_ptr().add(at)),
                    vld1q_f32(x.as_ptr().add(at + 4)),
                );
                for r in 0..ROWS {
                    sums[r][0] = vfmaq_f32(sums[r][0], vld1q_f32(rows[r].add(at)), x0);
                    sums[r][1] = vfmaq_f32(sums[r][1], vld1q_f32(rows[r].add(at + 4)), x1);
                }
            }
            for r in 0..ROWS {
                vst1q_f32(acc[r].as_mut_ptr(), sums[r][0]);
                vst1q_f32(acc[r].as_mut_ptr().add(4), sums[r][1]);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for at in (0..whole).step_by(LANES) {
            let xs = unsafe { &*(x.as_ptr().add(at) as *const [f32; LANES]) };
            for r in 0..ROWS {
                let ws = unsafe { &*(rows[r].add(at) as *const [f32; LANES]) };
                for l in 0..LANES {
                    acc[r][l] += ws[l] * xs[l];
                }
            }
        }
        for r in 0..ROWS {
            let sums = &acc[r];
            let mut sum = ((sums[0] + sums[4]) + (sums[2] + sums[6]))
                + ((sums[1] + sums[5]) + (sums[3] + sums[7]));
            for (t, x) in x.iter().enumerate().skip(whole) {
                sum += unsafe { *rows[r].add(t) } * x;
            }
            group[r] = sum;
        }
    }
    for (i, out) in rest.iter_mut().enumerate() {
        *out = dot(
            unsafe { std::slice::from_raw_parts(a.add((done + i) * stride), k) },
            x,
        );
    }
}

/// Thirty-two independent partial sums: the loop vectorizes and no sum
/// waits on the previous add.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 32;
    let mut acc = [0f32; LANES];
    let ((wide_a, rest_a), (wide_b, rest_b)) = (a.as_chunks::<LANES>(), b.as_chunks::<LANES>());
    let tail: f32 = rest_a.iter().zip(rest_b).map(|(x, y)| x * y).sum();
    for (x, y) in wide_a.iter().zip(wide_b) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    // Fold the partial sums pairwise: a serial sum would wait on every add.
    let mut width = LANES / 2;
    while width > 0 {
        for i in 0..width {
            acc[i] += acc[i + width];
        }
        width /= 2;
    }
    acc[0] + tail
}

fn axpy(scale: f32, x: &[f32], y: &mut [f32]) {
    for (y, x) in y.iter_mut().zip(x) {
        *y += scale * *x;
    }
}

#[cfg(test)]
mod portable_tests {
    use super::*;

    #[test]
    #[ignore]
    fn time_a_sparse_row_times_matrix() {
        let w: Vec<f32> = (0..832 * 129).map(|i| (i % 17) as f32 * 0.01).collect();
        let mut x = vec![0f32; 2 * 832];
        let mut out = vec![0f32; 2 * 129];
        for nonzeros in [0usize, 6] {
            for i in 0..nonzeros {
                x[i * 131] = 1.;
                x[832 + i * 97] = -1.;
            }
            let raw = |p: *const f32, n: usize| RawBuf {
                ptr: p as *mut u8,
                bytes: n * 4,
            };
            let bufs = [
                raw(out.as_mut_ptr(), 258),
                raw(x.as_ptr(), x.len()),
                raw(w.as_ptr(), w.len()),
            ];
            let spec =
                ContractSpec::parse("cpu_contract_blas:1,129,832,2,0,1,832,0,1,2,0,129,1").unwrap();
            let start = std::time::Instant::now();
            for _ in 0..1_000_000 {
                run(&spec, &bufs).unwrap();
                std::hint::black_box(&out);
            }
            println!(
                "sparse 2x[1,832]@[832,129], {nonzeros} nonzeros each: {:.0} ns",
                start.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    /// `cargo test -p fusor-cpu --release --lib -- --ignored --nocapture time_`
    #[test]
    #[ignore]
    fn time_a_small_matvec() {
        let w: Vec<f32> = (0..32 * 128).map(|i| (i % 17) as f32 * 0.01).collect();
        let x: Vec<f32> = (0..128).map(|i| (i % 5) as f32 - 2.).collect();
        let mut out = vec![0f32; 32];
        let raw = |p: *const f32, n: usize| RawBuf {
            ptr: p as *mut u8,
            bytes: n * 4,
        };
        let bufs = [
            raw(out.as_mut_ptr(), 32),
            raw(w.as_ptr(), w.len()),
            raw(x.as_ptr(), 128),
        ];
        for transforms in ["", ";;x0.c0.b7;"] {
            let spec = ContractSpec::parse(&format!(
                "cpu_contract_blas:32,1,128,1,0,1,0,128,1,2,0,1,0{transforms}"
            ))
            .unwrap();
            let start = std::time::Instant::now();
            for _ in 0..1_000_000 {
                run(&spec, &bufs).unwrap();
                std::hint::black_box(&out);
            }
            println!(
                "matvec 32x128 {transforms:?}: {:.0} ns",
                start.elapsed().as_secs_f64() * 1e3
            );
        }
    }

    #[test]
    fn portable_kernel_matches_the_definition_for_every_stride_pattern() {
        let (m, n, k) = (5usize, 3usize, 7usize);
        let data: Vec<f32> = (0..512)
            .map(|i| ((i * 37) % 23) as f32 * 0.1 - 1.0)
            .collect();
        // (a strides, b strides, n): dense, transposed A, transposed B,
        // broadcast A rows, a matrix-vector product both ways.
        for (a_strides, b_strides, n) in [
            ([k, 1], [n, 1], n),
            ([1, m], [n, 1], n),
            ([k, 1], [1, k], n),
            ([0, 1], [n, 1], n),
            ([k, 1], [1, 0], 1),
            ([1, m], [1, 0], 1),
        ] {
            let mut got = vec![0f32; m * n];
            unsafe {
                sgemm_portable(
                    [m, n, k],
                    data.as_ptr(),
                    a_strides,
                    data[200..].as_ptr(),
                    b_strides,
                    got.as_mut_ptr(),
                );
            }
            for i in 0..m {
                for j in 0..n {
                    let want: f32 = (0..k)
                        .map(|kk| {
                            data[i * a_strides[0] + kk * a_strides[1]]
                                * data[200 + kk * b_strides[0] + j * b_strides[1]]
                        })
                        .sum();
                    assert!(
                        (got[i * n + j] - want).abs() < 1e-4,
                        "{a_strides:?} {b_strides:?} [{i},{j}]"
                    );
                }
            }
        }
    }
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
                pre_a: Vec::new(),
                pre_b: Vec::new(),
                post: Vec::new(),
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
            let inner = 0.797_884_6 * (x + 0.044_715 * x * x * x);
            let expected = 0.5 * x * (1.0 + inner.tanh());
            assert!((got - expected).abs() < 2.0e-4, "x={x}");
        }
    }
}
