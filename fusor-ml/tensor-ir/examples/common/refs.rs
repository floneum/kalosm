//! CPU reference implementations and workload fixtures shared between
//! `runtime_benchmarks` and `runtime_beam_search`. Included via `#[path]`
//! so no Cargo.toml changes are required.

use tensor_ir::*;

pub fn cpu_matmul_reference(m: u32, n: u32, k: u32, a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0f32; (m * n) as usize];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for inner in 0..k {
                acc += a[(row * k + inner) as usize] * b[(inner * n + col) as usize];
            }
            output[(row * n + col) as usize] = acc;
        }
    }
    output
}

pub fn cpu_reduce_sum_reference(rows: u32, cols: u32, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows as usize];
    for (row, out_row) in out.iter_mut().enumerate() {
        let base = row * cols as usize;
        let mut acc = 0.0f32;
        for col in 0..cols as usize {
            acc += x[base + col];
        }
        *out_row = acc;
    }
    out
}

pub fn cpu_softmax_reference(rows: u32, cols: u32, x: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; (rows * cols) as usize];
    for row in 0..rows as usize {
        let base = row * cols as usize;
        let mut row_max = f32::NEG_INFINITY;
        for col in 0..cols as usize {
            row_max = row_max.max(x[base + col]);
        }
        let mut denom = 0.0f32;
        for col in 0..cols as usize {
            let e = (x[base + col] - row_max).exp();
            out[base + col] = e;
            denom += e;
        }
        for col in 0..cols as usize {
            out[base + col] /= denom;
        }
    }
    out
}

pub fn cpu_attention_reference(seq: u32, d: u32, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let mut scores = vec![0.0f32; (seq * seq) as usize];
    for i in 0..seq as usize {
        for j in 0..seq as usize {
            let mut acc = 0.0f32;
            for inner in 0..d as usize {
                acc += q[i * d as usize + inner] * k[j * d as usize + inner];
            }
            scores[i * seq as usize + j] = acc;
        }
    }
    let probs = cpu_softmax_reference(seq, seq, &scores);
    let mut out = vec![0.0f32; (seq * d) as usize];
    for i in 0..seq as usize {
        for col in 0..d as usize {
            let mut acc = 0.0f32;
            for inner in 0..seq as usize {
                acc += probs[i * seq as usize + inner] * v[inner * d as usize + col];
            }
            out[i * d as usize + col] = acc;
        }
    }
    out
}

#[allow(
    dead_code,
    reason = "shared example fixture is only consumed by runtime_beam_search"
)]
pub struct Workload {
    pub name: &'static str,
    pub expr: egg::RecExpr<TensorIr>,
    pub inputs: Vec<Vec<f32>>,
    pub expected: Vec<f32>,
}

#[allow(
    dead_code,
    reason = "shared example fixture is only consumed by runtime_beam_search"
)]
pub fn build_workload(kind: &str, m: u32, n: u32, k: u32) -> Result<Workload, String> {
    match kind {
        "matmul" => {
            let mut builder = IrBuilder::new();
            let a = builder.input(0, Shape(vec![Dim::Const(m), Dim::Const(k)]), DType::F32);
            let b = builder.input(1, Shape(vec![Dim::Const(k), Dim::Const(n)]), DType::F32);
            let arg0 = builder.scalar_arg(0);
            let arg1 = builder.scalar_arg(1);
            let body = builder.bin_op(BinaryOp::Mul, arg0, arg1);
            let _ = builder.contraction(
                Shape(vec![Dim::Const(m), Dim::Const(n), Dim::Const(k)]),
                &[
                    (
                        a,
                        Strides(vec![Dim::Const(k), Dim::Const(0), Dim::Const(1)]),
                    ),
                    (
                        b,
                        Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(n)]),
                    ),
                ],
                body,
                &[(2, ReduceOp::Add)],
            );
            let input_a: Vec<f32> = (0..m * k).map(|i| (i % 7) as f32 * 0.1).collect();
            let input_b: Vec<f32> = (0..k * n).map(|i| (i % 5) as f32 * 0.1).collect();
            let expected = cpu_matmul_reference(m, n, k, &input_a, &input_b);
            Ok(Workload {
                name: "matmul",
                expr: builder.expr,
                inputs: vec![input_a, input_b],
                expected,
            })
        }
        "reduce_sum" => {
            let (rows, cols) = (m, k);
            let mut builder = IrBuilder::new();
            let x = builder.input(
                0,
                Shape(vec![Dim::Const(rows), Dim::Const(cols)]),
                DType::F32,
            );
            let _ = builder.reduce(x, 1, ReduceOp::Add);
            let input: Vec<f32> = (0..rows * cols).map(|i| (i % 11) as f32 * 0.1).collect();
            let expected = cpu_reduce_sum_reference(rows, cols, &input);
            Ok(Workload {
                name: "reduce_sum",
                expr: builder.expr,
                inputs: vec![input],
                expected,
            })
        }
        "softmax" => {
            let (rows, cols) = (m, k);
            let mut builder = IrBuilder::new();
            let shape = Shape(vec![Dim::Const(rows), Dim::Const(cols)]);
            let x = builder.input(0, shape.clone(), DType::F32);
            let _ = builder.softmax(x, shape, 1);
            let input: Vec<f32> = (0..rows * cols)
                .map(|i| ((i % 13) as f32 - 6.0) * 0.1)
                .collect();
            let expected = cpu_softmax_reference(rows, cols, &input);
            Ok(Workload {
                name: "softmax",
                expr: builder.expr,
                inputs: vec![input],
                expected,
            })
        }
        "attention" => {
            let (seq, d) = (m, k);
            let mut builder = IrBuilder::new();
            let q = builder.input(0, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
            let kk = builder.input(1, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
            let v = builder.input(2, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
            let qk_tile = Shape(vec![Dim::Const(seq), Dim::Const(seq), Dim::Const(d)]);
            let q_r = builder.restride(
                q,
                qk_tile.clone(),
                Strides(vec![Dim::Const(d), Dim::Const(0), Dim::Const(1)]),
            );
            let k_r = builder.restride(
                kk,
                qk_tile.clone(),
                Strides(vec![Dim::Const(0), Dim::Const(d), Dim::Const(1)]),
            );
            let arg0 = builder.scalar_arg(0);
            let arg1 = builder.scalar_arg(1);
            let mul_body = builder.bin_op(BinaryOp::Mul, arg0, arg1);
            let qk_mul = builder.elementwise(qk_tile, &[q_r, k_r], mul_body);
            let scores = builder.reduce(qk_mul, 2, ReduceOp::Add);
            let scores_shape = Shape(vec![Dim::Const(seq), Dim::Const(seq)]);
            let probs = builder.softmax(scores, scores_shape, 1);
            let pv_tile = Shape(vec![Dim::Const(seq), Dim::Const(d), Dim::Const(seq)]);
            let p_r = builder.restride(
                probs,
                pv_tile.clone(),
                Strides(vec![Dim::Const(seq), Dim::Const(0), Dim::Const(1)]),
            );
            let v_r = builder.restride(
                v,
                pv_tile.clone(),
                Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(d)]),
            );
            let arg0 = builder.scalar_arg(0);
            let arg1 = builder.scalar_arg(1);
            let mul_body = builder.bin_op(BinaryOp::Mul, arg0, arg1);
            let pv_mul = builder.elementwise(pv_tile, &[p_r, v_r], mul_body);
            let _ = builder.reduce(pv_mul, 2, ReduceOp::Add);
            let q_input: Vec<f32> = (0..seq * d).map(|i| (i % 11) as f32 * 0.1).collect();
            let k_input: Vec<f32> = (0..seq * d).map(|i| (i % 13) as f32 * 0.1).collect();
            let v_input: Vec<f32> = (0..seq * d).map(|i| (i % 17) as f32 * 0.1).collect();
            let expected = cpu_attention_reference(seq, d, &q_input, &k_input, &v_input);
            Ok(Workload {
                name: "attention",
                expr: builder.expr,
                inputs: vec![q_input, k_input, v_input],
                expected,
            })
        }
        other => Err(format!(
            "unknown workload '{other}'. valid: matmul, reduce_sum, softmax, attention"
        )),
    }
}
