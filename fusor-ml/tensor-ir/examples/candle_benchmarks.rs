//! Head-to-head benchmark: tensor_ir (wgpu → Metal) vs candle-core (Metal
//! direct) on sgemv, sgemm, flash attention, softmax, elementwise add, and
//! reduce sum.
//!
//! Both sides use the same inputs and validate outputs against CPU references.
//! Tensor IR reports both GPU timestamp kernel time and host wall-clock time;
//! Candle reports host wall-clock time through `device.synchronize`. Tensor IR
//! also reports an amortized batched host time that shares one command
//! submission/synchronization across repeated executions.
//!
//! Run with:
//!   cargo run -r -p tensor_ir --features "runtime candle" \
//!       --example candle_benchmarks

#[cfg(not(all(feature = "runtime", feature = "candle")))]
fn main() {
    eprintln!(
        "Enable both features: cargo run -r -p tensor_ir --features \"runtime candle\" --example candle_benchmarks"
    );
    std::process::exit(1);
}

#[cfg(all(feature = "runtime", feature = "candle"))]
#[path = "common/refs.rs"]
mod common_refs;

#[cfg(all(feature = "runtime", feature = "candle"))]
use common_refs::{
    cpu_attention_reference, cpu_matmul_reference, cpu_reduce_sum_reference, cpu_softmax_reference,
};

#[cfg(all(feature = "runtime", feature = "candle"))]
use candle_core::{Device, Tensor};
#[cfg(all(feature = "runtime", feature = "candle"))]
use egg::Language;
#[cfg(all(feature = "runtime", feature = "candle"))]
use std::time::Instant;
#[cfg(all(feature = "runtime", feature = "candle"))]
use tensor_ir::*;

#[cfg(all(feature = "runtime", feature = "candle"))]
#[derive(Debug, Clone, Copy)]
struct RunCfg {
    warmup: u32,
    timing: u32,
    tir_batch: u32,
}

#[cfg(all(feature = "runtime", feature = "candle"))]
#[derive(Debug, Clone)]
struct Stats {
    median: f64,
}

#[cfg(all(feature = "runtime", feature = "candle"))]
struct AttentionDispatchCost;

#[cfg(all(feature = "runtime", feature = "candle"))]
impl egg::CostFunction<TensorIr> for AttentionDispatchCost {
    type Cost = f64;

    fn cost<C>(&mut self, enode: &TensorIr, mut costs: C) -> Self::Cost
    where
        C: FnMut(egg::Id) -> Self::Cost,
    {
        let children = enode.children().iter().map(|id| costs(*id)).sum::<f64>();
        let node = match enode {
            TensorIr::Dispatch(DispatchNode::Dispatch { .. }) => 1.0,
            TensorIr::Dispatch(_) => 50.0,
            TensorIr::Simd(_) => 10.0,
            TensorIr::BinOp(_, _) | TensorIr::UnOp(_, _) | TensorIr::TernOp(_, _) => 20.0,
            TensorIr::Const(_) | TensorIr::Nil | TensorIr::Cons(_) => 0.1,
            TensorIr::HighLevel(_) => 1_000.0,
        };
        node + children
    }
}

#[cfg(all(feature = "runtime", feature = "candle"))]
impl Stats {
    fn from(mut samples: Vec<f64>) -> Self {
        assert!(!samples.is_empty());
        samples.sort_by(f64::total_cmp);
        Self {
            median: samples[samples.len() / 2],
        }
    }
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn time_runs<F: FnMut()>(cfg: RunCfg, mut f: F) -> Stats {
    for _ in 0..cfg.warmup {
        f();
    }
    let mut samples = Vec::with_capacity(cfg.timing as usize);
    for _ in 0..cfg.timing {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_secs_f64() * 1e6);
    }
    Stats::from(samples)
}

/// GPU-only timing for a tensor_ir program via `ctx.benchmark`. Prepares the
/// shader + buffers once, then uses wgpu timestamp queries to measure kernel
/// execution time across `cfg.timing` runs.
#[cfg(all(feature = "runtime", feature = "candle"))]
fn time_tir_gpu(
    ctx: &GpuContext,
    program: &DispatchProgram,
    inputs: &[&[f32]],
    cfg: RunCfg,
) -> Result<Stats, String> {
    let result = ctx.benchmark(
        program,
        inputs,
        &ShapeParams::default(),
        ProgramBenchmarkConfig {
            warmup_runs: cfg.warmup,
            timing_runs: cfg.timing,
        },
    )?;
    Ok(Stats {
        median: result.median_gpu_us,
    })
}

/// Host wall-clock timing for a prepared tensor_ir program. This includes
/// command encoding, submission, and synchronization, but not readback.
#[cfg(all(feature = "runtime", feature = "candle"))]
fn time_tir_host(
    ctx: &GpuContext,
    program: &DispatchProgram,
    inputs: &[&[f32]],
    cfg: RunCfg,
) -> Result<Stats, String> {
    let result = ctx.benchmark_host(
        program,
        inputs,
        &ShapeParams::default(),
        ProgramBenchmarkConfig {
            warmup_runs: cfg.warmup,
            timing_runs: cfg.timing,
        },
    )?;
    Ok(Stats {
        median: result.median_host_us,
    })
}

/// Amortized host wall-clock timing for a prepared tensor_ir program. Each
/// sample batches `cfg.tir_batch` executions into one command submission and
/// divides elapsed time by the batch size.
#[cfg(all(feature = "runtime", feature = "candle"))]
fn time_tir_host_batched(
    ctx: &GpuContext,
    program: &DispatchProgram,
    inputs: &[&[f32]],
    cfg: RunCfg,
) -> Result<Stats, String> {
    let result = ctx.benchmark_host_batched(
        program,
        inputs,
        &ShapeParams::default(),
        ProgramBenchmarkConfig {
            warmup_runs: cfg.warmup,
            timing_runs: cfg.timing,
        },
        cfg.tir_batch,
    )?;
    Ok(Stats {
        median: result.median_host_us,
    })
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn max_abs_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch {} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn validate_err(
    op: &str,
    implementation: &str,
    out: &[f32],
    reference: &[f32],
) -> Result<f32, String> {
    if out.len() < reference.len() {
        return Err(format!(
            "{op}: {implementation} produced {} values, expected at least {}",
            out.len(),
            reference.len()
        ));
    }
    let err = max_abs_err(&out[..reference.len()], reference);
    if err >= 1e-3 {
        return Err(format!("{op}: {implementation} max_err={err:.3e}"));
    }
    Ok(err)
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn seq_f32(len: usize, modulus: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % modulus) as f32) / modulus as f32)
        .collect()
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn signed_seq_f32(len: usize, modulus: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % modulus) as f32 - (modulus as f32 / 2.0)) / modulus as f32)
        .collect()
}

/// Quick tuning benchmark on a candidate. Returns median GPU µs so a single
/// timestamp outlier can't flip the selection order.
#[cfg(all(feature = "runtime", feature = "candle"))]
fn quick_time_us(ctx: &GpuContext, program: &DispatchProgram, inputs: &[&[f32]]) -> Option<f64> {
    use std::panic::{self, AssertUnwindSafe};
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        ctx.benchmark(
            program,
            inputs,
            &ShapeParams::default(),
            ProgramBenchmarkConfig {
                warmup_runs: 3,
                timing_runs: 15,
            },
        )
    }));
    if let Ok(Ok(r)) = result {
        Some(r.median_gpu_us)
    } else {
        None
    }
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn program_shape_summary(program: &DispatchProgram) -> String {
    let dispatches = program
        .dispatches
        .iter()
        .map(|d| {
            format!(
                "wg={} sg={} out={} tg={}B",
                d.workgroups,
                d.simdgroups,
                d.outputs.len(),
                d.threadgroup_bytes()
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    format!(
        "peak_tg={}B [{dispatches}]",
        program.peak_threadgroup_bytes()
    )
}

/// Try direct pipeline lowering, then beam-search candidates; benchmark each
/// valid candidate and keep the fastest. The cost model used by
/// `beam_extract_valid_candidates` is a heuristic; measuring on-device kernel
/// time directly lets us pick the one that actually runs the fastest.
#[cfg(all(feature = "runtime", feature = "candle"))]
fn compile_tensor_ir(
    ctx: &GpuContext,
    expr: &TensorExprProgram,
    inputs: &[&[f32]],
    expected: &[f32],
) -> Result<(DispatchProgram, Vec<f32>), String> {
    use std::panic::{self, AssertUnwindSafe};

    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let mut best: Option<(f64, DispatchProgram, Vec<f32>)> = None;
    let tolerance_ok = |out: &[f32]| -> bool {
        out.len() >= expected.len()
            && max_abs_err(&out[..expected.len()], expected).is_finite()
            && max_abs_err(&out[..expected.len()], expected) < 1e-2
    };
    let debug_candidates = std::env::var_os("TIR_DUMP_CANDIDATES").is_some();
    let mut consider = |label: &str, program: DispatchProgram, ctx: &GpuContext| {
        let out = panic::catch_unwind(AssertUnwindSafe(|| {
            ctx.execute(&program, inputs, &ShapeParams::default())
        }));
        let Ok(out) = out else { return };
        if !tolerance_ok(&out) {
            return;
        }
        let Some(us) = quick_time_us(ctx, &program, inputs) else {
            return;
        };
        if debug_candidates {
            eprintln!("{label:>8} {us:8.2}us {}", program_shape_summary(&program));
        }
        match &best {
            Some((best_us, _, _)) if *best_us <= us => {}
            _ => best = Some((us, program, out)),
        }
    };

    // Candidate 1: direct pipeline lowering (semantically correct by
    // construction, often fast for simple ops).
    let fast = panic::catch_unwind(AssertUnwindSafe(|| {
        StagedPipeline::default()
            .build(expr)
            .map(SimdProgram::into_dispatch_program)
    }));
    if let Ok(Ok(program)) = fast {
        consider("direct", program, ctx);
    }

    // Candidates 2…N: beam-search extractions from the saturated e-graph.
    // Bump beam_width + candidate limit: every extra candidate costs only a
    // quick-benchmark (3 warmup + 15 timed submits) to evaluate, and the wider
    // survey often surfaces register-blocked / tiled variants the default
    // beam misses.
    let beam_result = panic::catch_unwind(AssertUnwindSafe(|| {
        let recexpr = tensor_expr_to_recexpr(expr)?;
        let mut egraph = TensorEGraph::default();
        let root = egraph.add_expr(&recexpr);
        egraph.rebuild();
        let runner = RunnerConfig {
            iter_limit: 20,
            node_limit: 200_000,
            time_limit_secs: 60,
            device: DeviceProfile::default(),
            lowering: LoweringOptions::default(),
        };
        let egraph = saturate_phases(egraph, Phase::all(), &runner);
        let beam_cfg = BeamConfig {
            beam_width: 128,
            ..BeamConfig::default()
        };
        let candidates = beam_extract_valid_candidates(
            &egraph,
            root,
            &beam_cfg,
            &DeviceProfile::default(),
            &LoweringOptions::default(),
            128,
        );
        Ok::<_, String>((egraph, candidates))
    }));
    if let Ok(Ok((egraph, candidates))) = beam_result {
        for (_, extracted) in candidates {
            let program_opt = panic::catch_unwind(AssertUnwindSafe(|| {
                build_dispatch_program_from_extracted(
                    &extracted,
                    egraph.clone(),
                    &DeviceProfile::default(),
                    &LoweringOptions::default(),
                )
            }));
            let Ok(program) = program_opt else { continue };
            if program.dispatches.is_empty() {
                continue;
            }
            consider("beam", program, ctx);
        }
    }

    panic::set_hook(hook);
    best.map(|(_, p, out)| (p, out))
        .ok_or_else(|| "no valid tensor_ir candidate".to_string())
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn build_sgemm_expr(m: u32, n: u32, k: u32) -> TensorExprProgram {
    let mut b = TensorExprBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(m), Dim::Const(k)]), DType::F32);
    let rhs = b.input(1, Shape(vec![Dim::Const(k), Dim::Const(n)]), DType::F32);
    let a0 = b.scalar_arg(0);
    let a1 = b.scalar_arg(1);
    let body = b.scalar_binop(BinaryOp::Mul, [a0, a1]);
    let root = b.contraction(
        Shape(vec![Dim::Const(m), Dim::Const(n), Dim::Const(k)]),
        &[
            (
                a,
                Strides(vec![Dim::Const(k), Dim::Const(0), Dim::Const(1)]),
            ),
            (
                rhs,
                Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(n)]),
            ),
        ],
        body,
        &[(2, ReduceOp::Add)],
    );
    b.build(root).expect("valid sgemm expr")
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn build_sgemv_expr(m: u32, k: u32) -> TensorExprProgram {
    let mut b = TensorExprBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(m), Dim::Const(k)]), DType::F32);
    let x = b.input(1, Shape(vec![Dim::Const(k), Dim::Const(1)]), DType::F32);
    let a0 = b.scalar_arg(0);
    let a1 = b.scalar_arg(1);
    let body = b.scalar_binop(BinaryOp::Mul, [a0, a1]);
    let root = b.contraction(
        Shape(vec![Dim::Const(m), Dim::Const(1), Dim::Const(k)]),
        &[
            (
                a,
                Strides(vec![Dim::Const(k), Dim::Const(0), Dim::Const(1)]),
            ),
            (
                x,
                Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(1)]),
            ),
        ],
        body,
        &[(2, ReduceOp::Add)],
    );
    b.build(root).expect("valid sgemv expr")
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn build_add_expr(rows: u32, cols: u32) -> TensorExprProgram {
    let mut b = TensorExprBuilder::new();
    let shape = Shape(vec![Dim::Const(rows), Dim::Const(cols)]);
    let a = b.input(0, shape.clone(), DType::F32);
    let r = b.input(1, shape.clone(), DType::F32);
    let a0 = b.scalar_arg(0);
    let a1 = b.scalar_arg(1);
    let body = b.scalar_binop(BinaryOp::Add, [a0, a1]);
    let root = b.elementwise(shape, &[a, r], body);
    b.build(root).expect("valid add expr")
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn build_reduce_sum_expr(rows: u32, cols: u32) -> TensorExprProgram {
    let mut b = TensorExprBuilder::new();
    let x = b.input(
        0,
        Shape(vec![Dim::Const(rows), Dim::Const(cols)]),
        DType::F32,
    );
    let root = b.reduce(x, 1, ReduceOp::Add);
    b.build(root).expect("valid reduce_sum expr")
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn build_attention_expr(seq: u32, d: u32) -> TensorExprProgram {
    let mut b = TensorExprBuilder::new();
    let q = b.input(0, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
    let k = b.input(1, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
    let v = b.input(2, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);

    let qk_tile = Shape(vec![Dim::Const(seq), Dim::Const(seq), Dim::Const(d)]);
    let q_r = b.restride(
        q,
        qk_tile.clone(),
        Strides(vec![Dim::Const(d), Dim::Const(0), Dim::Const(1)]),
    );
    let k_r = b.restride(
        k,
        qk_tile.clone(),
        Strides(vec![Dim::Const(0), Dim::Const(d), Dim::Const(1)]),
    );
    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let mul_body = b.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let qk_mul = b.elementwise(qk_tile, &[q_r, k_r], mul_body);
    let scores = b.reduce(qk_mul, 2, ReduceOp::Add);

    let scores_shape = Shape(vec![Dim::Const(seq), Dim::Const(seq)]);
    let probs = b.softmax(scores, scores_shape, 1);

    let pv_tile = Shape(vec![Dim::Const(seq), Dim::Const(d), Dim::Const(seq)]);
    let p_r = b.restride(
        probs,
        pv_tile.clone(),
        Strides(vec![Dim::Const(seq), Dim::Const(0), Dim::Const(1)]),
    );
    let v_r = b.restride(
        v,
        pv_tile.clone(),
        Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(d)]),
    );
    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let mul_body = b.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let pv_mul = b.elementwise(pv_tile, &[p_r, v_r], mul_body);
    let root = b.reduce(pv_mul, 2, ReduceOp::Add);

    b.build(root).expect("valid attention expr")
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn candle_sync(device: &Device) {
    device.synchronize().expect("candle synchronize");
}

#[cfg(all(feature = "runtime", feature = "candle"))]
#[derive(Debug)]
struct Row {
    op: &'static str,
    shape: String,
    tir_gpu: Stats,
    tir_host: Stats,
    tir_batched: Stats,
    candle_host: Stats,
    tir_err: f32,
    candle_err: f32,
    dispatches: usize,
}

#[cfg(all(feature = "runtime", feature = "candle"))]
struct TirBenchResult {
    gpu: Stats,
    host: Stats,
    batched: Stats,
    err: f32,
    dispatches: usize,
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn sum_stats(stats: &[Stats]) -> Stats {
    Stats {
        median: stats.iter().map(|s| s.median).sum(),
    }
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn transpose_row_major(rows: u32, cols: u32, data: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; (rows * cols) as usize];
    for row in 0..rows as usize {
        for col in 0..cols as usize {
            out[col * rows as usize + row] = data[row * cols as usize + col];
        }
    }
    out
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn print_header() {
    println!(
        "{:<12} {:<18} {:>12} {:>12} {:>12} {:>12} {:>10} {:>10} {:>10} {:>10} {:>5}",
        "op",
        "shape",
        "tir_gpu_med",
        "tir_host_med",
        "tir_batch",
        "can_host_med",
        "sync_spd",
        "batch_spd",
        "tir_err",
        "can_err",
        "disp"
    );
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn print_row(r: &Row) {
    let sync_speedup = if r.tir_host.median > 0.0 {
        r.candle_host.median / r.tir_host.median
    } else {
        f64::NAN
    };
    let batch_speedup = if r.tir_batched.median > 0.0 {
        r.candle_host.median / r.tir_batched.median
    } else {
        f64::NAN
    };
    println!(
        "{:<12} {:<18} {:>12.2} {:>12.2} {:>12.2} {:>12.2} {:>9.2}x {:>9.2}x {:>10.2e} {:>10.2e} {:>5}",
        r.op,
        r.shape,
        r.tir_gpu.median,
        r.tir_host.median,
        r.tir_batched.median,
        r.candle_host.median,
        sync_speedup,
        batch_speedup,
        r.tir_err,
        r.candle_err,
        r.dispatches
    );
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn run_sgemm(
    ctx: &GpuContext,
    device: &Device,
    cfg: RunCfg,
    m: u32,
    n: u32,
    k: u32,
) -> Result<Row, String> {
    let expr = build_sgemm_expr(m, n, k);
    let a = seq_f32((m * k) as usize, 17);
    let b_data = seq_f32((k * n) as usize, 31);
    let inputs: Vec<&[f32]> = vec![&a, &b_data];
    let reference = cpu_matmul_reference(m, n, k, &a, &b_data);
    let (program, tir_out) = compile_tensor_ir(ctx, &expr, &inputs, &reference)?;
    let tir_err = validate_err("sgemm", "tensor_ir", &tir_out, &reference)?;

    let ta = Tensor::from_slice(&a, (m as usize, k as usize), device).map_err(|e| e.to_string())?;
    let tb =
        Tensor::from_slice(&b_data, (k as usize, n as usize), device).map_err(|e| e.to_string())?;
    let candle_out = ta
        .matmul(&tb)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?;
    let candle_err = validate_err("sgemm", "candle", &candle_out, &reference)?;

    let tir_gpu_stats = time_tir_gpu(ctx, &program, &inputs, cfg)?;
    let tir_host_stats = time_tir_host(ctx, &program, &inputs, cfg)?;
    let tir_batched_stats = time_tir_host_batched(ctx, &program, &inputs, cfg)?;
    let candle_stats = time_runs(cfg, || {
        let c = ta.matmul(&tb).expect("candle matmul");
        drop(c);
        candle_sync(device);
    });

    Ok(Row {
        op: "sgemm",
        shape: format!("{m}x{n}x{k}"),
        tir_gpu: tir_gpu_stats,
        tir_host: tir_host_stats,
        tir_batched: tir_batched_stats,
        candle_host: candle_stats,
        tir_err,
        candle_err,
        dispatches: program.dispatches.len(),
    })
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn run_sgemv(
    ctx: &GpuContext,
    device: &Device,
    cfg: RunCfg,
    m: u32,
    k: u32,
) -> Result<Row, String> {
    let expr = build_sgemv_expr(m, k);
    let a = seq_f32((m * k) as usize, 17);
    let x = seq_f32(k as usize, 31);
    let inputs: Vec<&[f32]> = vec![&a, &x];
    let reference = cpu_matmul_reference(m, 1, k, &a, &x);
    let (program, tir_out) = compile_tensor_ir(ctx, &expr, &inputs, &reference)?;
    let tir_err = validate_err("sgemv", "tensor_ir", &tir_out, &reference)?;

    let ta = Tensor::from_slice(&a, (m as usize, k as usize), device).map_err(|e| e.to_string())?;
    let tx = Tensor::from_slice(&x, (k as usize, 1usize), device).map_err(|e| e.to_string())?;
    let candle_out = ta
        .matmul(&tx)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?;
    let candle_err = validate_err("sgemv", "candle", &candle_out, &reference)?;

    let tir_gpu_stats = time_tir_gpu(ctx, &program, &inputs, cfg)?;
    let tir_host_stats = time_tir_host(ctx, &program, &inputs, cfg)?;
    let tir_batched_stats = time_tir_host_batched(ctx, &program, &inputs, cfg)?;
    let candle_stats = time_runs(cfg, || {
        let c = ta.matmul(&tx).expect("candle sgemv");
        drop(c);
        candle_sync(device);
    });

    Ok(Row {
        op: "sgemv",
        shape: format!("{m}x{k}"),
        tir_gpu: tir_gpu_stats,
        tir_host: tir_host_stats,
        tir_batched: tir_batched_stats,
        candle_host: candle_stats,
        tir_err,
        candle_err,
        dispatches: program.dispatches.len(),
    })
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn run_softmax(
    ctx: &GpuContext,
    device: &Device,
    cfg: RunCfg,
    rows: u32,
    cols: u32,
) -> Result<Row, String> {
    let input = signed_seq_f32((rows * cols) as usize, 7);
    let inputs: Vec<&[f32]> = vec![&input];
    let reference = cpu_softmax_reference(rows, cols, &input);
    // Softmax is a composite (reduce-max → sub → exp → reduce-sum → div);
    // use the same low-level `IrBuilder` + beam-search path that
    // `runtime_beam_search` used, which finds a valid Dispatch candidate at
    // this shape.
    let program = compile_softmax_via_irbuilder(ctx, rows, cols, &inputs, &reference)?;
    let tir_out = ctx.execute(&program, &inputs, &ShapeParams::default());
    let tir_err = validate_err("softmax", "tensor_ir", &tir_out, &reference)?;

    let tx = Tensor::from_slice(&input, (rows as usize, cols as usize), device)
        .map_err(|e| e.to_string())?;
    let candle_out = candle_softmax(&tx)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?;
    let candle_err = validate_err("softmax", "candle", &candle_out, &reference)?;

    let tir_gpu_stats = time_tir_gpu(ctx, &program, &inputs, cfg)?;
    let tir_host_stats = time_tir_host(ctx, &program, &inputs, cfg)?;
    let tir_batched_stats = time_tir_host_batched(ctx, &program, &inputs, cfg)?;
    let candle_stats = time_runs(cfg, || {
        let o = candle_softmax(&tx).expect("candle softmax");
        drop(o);
        candle_sync(device);
    });

    Ok(Row {
        op: "softmax",
        shape: format!("{rows}x{cols}"),
        tir_gpu: tir_gpu_stats,
        tir_host: tir_host_stats,
        tir_batched: tir_batched_stats,
        candle_host: candle_stats,
        tir_err,
        candle_err,
        dispatches: program.dispatches.len(),
    })
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn compile_softmax_via_irbuilder(
    ctx: &GpuContext,
    rows: u32,
    cols: u32,
    inputs: &[&[f32]],
    expected: &[f32],
) -> Result<DispatchProgram, String> {
    use std::panic::{self, AssertUnwindSafe};

    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<DispatchProgram, String> {
        let make_recexpr = || {
            let mut builder = IrBuilder::new();
            let shape = Shape(vec![Dim::Const(rows), Dim::Const(cols)]);
            let x = builder.input(0, shape.clone(), DType::F32);
            let _ = builder.softmax(x, shape, 1);
            builder.expr
        };

        // Softmax's compound reduction trips every lowering corner case in
        // the rule system; give saturation a big budget so anything lurking
        // can surface, and widen the beam aggressively because the cheapest-
        // cost candidate is usually the slow sequential form.
        let runner = RunnerConfig {
            iter_limit: 40,
            node_limit: 500_000,
            time_limit_secs: 120,
            device: DeviceProfile::default(),
            lowering: LoweringOptions::default(),
        };
        let beam_cfg = BeamConfig {
            beam_width: 512,
            ..BeamConfig::default()
        };

        let mut best: Option<(f64, DispatchProgram)> = None;

        let mut try_candidates_from = |egraph: TensorEGraph, root: egg::Id| {
            let candidates = beam_extract_valid_candidates(
                &egraph,
                root,
                &beam_cfg,
                &DeviceProfile::default(),
                &LoweringOptions::default(),
                256,
            );
            for (_, extracted) in candidates {
                let program_opt = panic::catch_unwind(AssertUnwindSafe(|| {
                    build_dispatch_program_from_extracted(
                        &extracted,
                        egraph.clone(),
                        &DeviceProfile::default(),
                        &LoweringOptions::default(),
                    )
                }));
                let Ok(program) = program_opt else { continue };
                if program.dispatches.is_empty() {
                    continue;
                }
                let out = panic::catch_unwind(AssertUnwindSafe(|| {
                    ctx.execute(&program, inputs, &ShapeParams::default())
                }));
                let Ok(out) = out else { continue };
                if out.len() < expected.len() {
                    continue;
                }
                let err = max_abs_err(&out[..expected.len()], expected);
                if !err.is_finite() || err >= 1e-2 {
                    continue;
                }
                let Some(us) = quick_time_us(ctx, &program, inputs) else {
                    continue;
                };
                match &best {
                    Some((best_us, _)) if *best_us <= us => {}
                    _ => best = Some((us, program)),
                }
            }
        };

        // Pass A: default saturation (all rules).
        let mut egraph = TensorEGraph::default();
        let root = egraph.add_expr(&make_recexpr());
        egraph.rebuild();
        let egraph = saturate_phases(egraph, Phase::all(), &runner);
        try_candidates_from(egraph, root);

        // Pass B: saturation with `recursive-to-dispatch` filtered out.
        // That rule greedily fuses softmax's whole compound expression into
        // one sequential Dispatch whose body top is a Div — hiding the
        // reducing Thetas from `theta-split-cooperative`. Without the rule,
        // the per-stage phase-1 reduce / elementwise lowerings can produce
        // multiple smaller Dispatches that *can* each be split cooperatively.
        let mut alt = TensorEGraph::default();
        let alt_root = alt.add_expr(&make_recexpr());
        alt.rebuild();
        let filtered_rules: Vec<_> = all_rules(&runner)
            .into_iter()
            .filter(|r| !format!("{}", r.name).contains("recursive-to-dispatch"))
            .collect();
        let alt_egraph = egg::Runner::<_, _, ()>::default()
            .with_egraph(alt)
            .with_iter_limit(runner.iter_limit)
            .with_node_limit(runner.node_limit)
            .with_time_limit(std::time::Duration::from_secs(runner.time_limit_secs))
            .with_scheduler(egg::BackoffScheduler::default())
            .run(&filtered_rules)
            .egraph;
        try_candidates_from(alt_egraph, alt_root);

        best.map(|(_, p)| p)
            .ok_or_else(|| "no valid softmax candidate".to_string())
    }));

    panic::set_hook(hook);
    match result {
        Ok(r) => r,
        Err(_) => Err("softmax saturation panicked".to_string()),
    }
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn compile_attention_via_irbuilder(
    ctx: &GpuContext,
    seq: u32,
    d: u32,
    inputs: &[&[f32]],
    expected: &[f32],
) -> Result<DispatchProgram, String> {
    use std::panic::{self, AssertUnwindSafe};

    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let result = panic::catch_unwind(AssertUnwindSafe(|| -> Result<DispatchProgram, String> {
        let mut builder = IrBuilder::new();
        let q = builder.input(0, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
        let k = builder.input(1, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
        let v = builder.input(2, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);

        let qk_tile = Shape(vec![Dim::Const(seq), Dim::Const(seq), Dim::Const(d)]);
        let q_r = builder.restride(
            q,
            qk_tile.clone(),
            Strides(vec![Dim::Const(d), Dim::Const(0), Dim::Const(1)]),
        );
        let k_r = builder.restride(
            k,
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

        let recexpr = builder.expr.clone();
        let runner = RunnerConfig {
            iter_limit: 10,
            node_limit: 100_000,
            time_limit_secs: 30,
            device: DeviceProfile::default(),
            lowering: LoweringOptions::default(),
        };
        let beam_cfg = BeamConfig {
            beam_width: 64,
            ..BeamConfig::default()
        };
        let mut best: Option<(f64, DispatchProgram)> = None;

        let mut try_candidates_from = |egraph: TensorEGraph, root: egg::Id| {
            let mut candidates = beam_extract_candidates(&egraph, root, &beam_cfg, 256);
            let greedy_dispatch =
                egg::Extractor::new(&egraph, AttentionDispatchCost).find_best(root);
            if !candidates
                .iter()
                .any(|(_, expr)| expr == &greedy_dispatch.1)
            {
                candidates.push(greedy_dispatch);
            }

            for (_, extracted) in candidates {
                let program_opt = panic::catch_unwind(AssertUnwindSafe(|| {
                    build_dispatch_program_from_extracted(
                        &extracted,
                        egraph.clone(),
                        &DeviceProfile::default(),
                        &LoweringOptions::default(),
                    )
                }));
                let Ok(program) = program_opt else { continue };
                if program.dispatches.is_empty() {
                    continue;
                }
                let out = panic::catch_unwind(AssertUnwindSafe(|| {
                    ctx.execute(&program, inputs, &ShapeParams::default())
                }));
                let Ok(out) = out else { continue };
                if out.len() < expected.len() {
                    continue;
                }
                let err = max_abs_err(&out[..expected.len()], expected);
                if !err.is_finite() || err >= 1e-2 {
                    continue;
                }
                let Some(us) = quick_time_us(ctx, &program, inputs) else {
                    continue;
                };
                match &best {
                    Some((best_us, _)) if *best_us <= us => {}
                    _ => best = Some((us, program)),
                }
            }
        };

        let mut egraph = TensorEGraph::default();
        let root = egraph.add_expr(&recexpr);
        egraph.rebuild();
        let egraph = saturate(egraph, &runner);
        try_candidates_from(egraph, root);

        let mut alt = TensorEGraph::default();
        let alt_root = alt.add_expr(&recexpr);
        alt.rebuild();
        let filtered_rules: Vec<_> = all_rules(&runner)
            .into_iter()
            .filter(|r| !format!("{}", r.name).contains("recursive-to-dispatch"))
            .collect();
        let alt_egraph = egg::Runner::<_, _, ()>::default()
            .with_egraph(alt)
            .with_iter_limit(runner.iter_limit)
            .with_node_limit(runner.node_limit)
            .with_time_limit(std::time::Duration::from_secs(runner.time_limit_secs))
            .with_scheduler(egg::BackoffScheduler::default())
            .run(&filtered_rules)
            .egraph;
        try_candidates_from(alt_egraph, alt_root);

        best.map(|(_, p)| p)
            .ok_or_else(|| "no valid attention candidate".to_string())
    }));

    panic::set_hook(hook);
    match result {
        Ok(r) => r,
        Err(_) => Err("attention saturation panicked".to_string()),
    }
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn candle_softmax(x: &Tensor) -> candle_core::Result<Tensor> {
    // softmax(x, axis=-1) = e^(x - max) / sum(e^(x - max))
    let m = x.max_keepdim(1)?;
    let shifted = x.broadcast_sub(&m)?;
    let e = shifted.exp()?;
    let s = e.sum_keepdim(1)?;
    e.broadcast_div(&s)
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn candle_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> candle_core::Result<Tensor> {
    let kt = k.t()?;
    let scores = q.matmul(&kt)?;
    let probs = candle_softmax(&scores)?;
    probs.matmul(v)
}

#[cfg(all(feature = "runtime", feature = "candle"))]
/// Fallback path when the full attention expression does not extract as one
/// runnable program: lower the three primitive stages independently and sum
/// their timings.
fn run_flash_attention_staged_tir(
    ctx: &GpuContext,
    cfg: RunCfg,
    seq: u32,
    d: u32,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    reference: &[f32],
) -> Result<TirBenchResult, String> {
    let kt = transpose_row_major(seq, d, k);
    let scores_ref = cpu_matmul_reference(seq, seq, d, q, &kt);
    let probs_ref = cpu_softmax_reference(seq, seq, &scores_ref);

    let qk_expr = build_sgemm_expr(seq, seq, d);
    let qk_inputs: Vec<&[f32]> = vec![q, &kt];
    let (qk_program, qk_out) = compile_tensor_ir(ctx, &qk_expr, &qk_inputs, &scores_ref)?;
    let _ = validate_err("flash_attn/qk", "tensor_ir", &qk_out, &scores_ref)?;
    let scores = qk_out[..scores_ref.len()].to_vec();

    let softmax_inputs: Vec<&[f32]> = vec![&scores_ref];
    let softmax_program =
        compile_softmax_via_irbuilder(ctx, seq, seq, &softmax_inputs, &probs_ref)?;
    let probs_out = ctx.execute(&softmax_program, &[&scores], &ShapeParams::default());
    let _ = validate_err("flash_attn/softmax", "tensor_ir", &probs_out, &probs_ref)?;
    let probs = probs_out[..probs_ref.len()].to_vec();

    let pv_expr = build_sgemm_expr(seq, d, seq);
    let pv_inputs: Vec<&[f32]> = vec![&probs_ref, v];
    let (pv_program, _) = compile_tensor_ir(ctx, &pv_expr, &pv_inputs, reference)?;
    let tir_out = ctx.execute(&pv_program, &[&probs, v], &ShapeParams::default());
    let err = validate_err("flash_attn", "tensor_ir", &tir_out, reference)?;

    let qk_gpu = time_tir_gpu(ctx, &qk_program, &qk_inputs, cfg)?;
    let softmax_gpu = time_tir_gpu(ctx, &softmax_program, &softmax_inputs, cfg)?;
    let pv_gpu = time_tir_gpu(ctx, &pv_program, &pv_inputs, cfg)?;

    let qk_host = time_tir_host(ctx, &qk_program, &qk_inputs, cfg)?;
    let softmax_host = time_tir_host(ctx, &softmax_program, &softmax_inputs, cfg)?;
    let pv_host = time_tir_host(ctx, &pv_program, &pv_inputs, cfg)?;

    let qk_batched = time_tir_host_batched(ctx, &qk_program, &qk_inputs, cfg)?;
    let softmax_batched = time_tir_host_batched(ctx, &softmax_program, &softmax_inputs, cfg)?;
    let pv_batched = time_tir_host_batched(ctx, &pv_program, &pv_inputs, cfg)?;

    Ok(TirBenchResult {
        gpu: sum_stats(&[qk_gpu, softmax_gpu, pv_gpu]),
        host: sum_stats(&[qk_host, softmax_host, pv_host]),
        batched: sum_stats(&[qk_batched, softmax_batched, pv_batched]),
        err,
        dispatches: qk_program.dispatches.len()
            + softmax_program.dispatches.len()
            + pv_program.dispatches.len(),
    })
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn run_flash_attention(
    ctx: &GpuContext,
    device: &Device,
    cfg: RunCfg,
    seq: u32,
    d: u32,
) -> Result<Row, String> {
    let expr = build_attention_expr(seq, d);
    let q = seq_f32((seq * d) as usize, 11);
    let k = seq_f32((seq * d) as usize, 13);
    let v = seq_f32((seq * d) as usize, 17);
    let inputs: Vec<&[f32]> = vec![&q, &k, &v];
    let reference = cpu_attention_reference(seq, d, &q, &k, &v);
    let mut tir_options = Vec::new();

    if let Ok((program, tir_out)) = compile_tensor_ir(ctx, &expr, &inputs, &reference) {
        let err = validate_err("flash_attn", "tensor_ir", &tir_out, &reference)?;
        tir_options.push(TirBenchResult {
            gpu: time_tir_gpu(ctx, &program, &inputs, cfg)?,
            host: time_tir_host(ctx, &program, &inputs, cfg)?,
            batched: time_tir_host_batched(ctx, &program, &inputs, cfg)?,
            err,
            dispatches: program.dispatches.len(),
        });
    }

    if let Ok(staged) = run_flash_attention_staged_tir(ctx, cfg, seq, d, &q, &k, &v, &reference) {
        tir_options.push(staged);
    }

    if tir_options.is_empty()
        && let Ok(program) = compile_attention_via_irbuilder(ctx, seq, d, &inputs, &reference)
    {
        let tir_out = ctx.execute(&program, &inputs, &ShapeParams::default());
        let err = validate_err("flash_attn", "tensor_ir", &tir_out, &reference)?;
        tir_options.push(TirBenchResult {
            gpu: time_tir_gpu(ctx, &program, &inputs, cfg)?,
            host: time_tir_host(ctx, &program, &inputs, cfg)?,
            batched: time_tir_host_batched(ctx, &program, &inputs, cfg)?,
            err,
            dispatches: program.dispatches.len(),
        });
    }

    let tir = tir_options
        .into_iter()
        .min_by(|lhs, rhs| {
            lhs.gpu
                .median
                .partial_cmp(&rhs.gpu.median)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| "no valid flash attention tensor_ir candidate".to_string())?;

    let tq =
        Tensor::from_slice(&q, (seq as usize, d as usize), device).map_err(|e| e.to_string())?;
    let tk =
        Tensor::from_slice(&k, (seq as usize, d as usize), device).map_err(|e| e.to_string())?;
    let tv =
        Tensor::from_slice(&v, (seq as usize, d as usize), device).map_err(|e| e.to_string())?;
    let candle_out = candle_attention(&tq, &tk, &tv)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?;
    let candle_err = validate_err("flash_attn", "candle", &candle_out, &reference)?;

    let candle_stats = time_runs(cfg, || {
        let o = candle_attention(&tq, &tk, &tv).expect("candle attention");
        drop(o);
        candle_sync(device);
    });

    Ok(Row {
        op: "flash_attn",
        shape: format!("{seq}x{d}"),
        tir_gpu: tir.gpu,
        tir_host: tir.host,
        tir_batched: tir.batched,
        candle_host: candle_stats,
        tir_err: tir.err,
        candle_err,
        dispatches: tir.dispatches,
    })
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn run_elementwise_add(
    ctx: &GpuContext,
    device: &Device,
    cfg: RunCfg,
    rows: u32,
    cols: u32,
) -> Result<Row, String> {
    let expr = build_add_expr(rows, cols);
    let a = seq_f32((rows * cols) as usize, 13);
    let b = seq_f32((rows * cols) as usize, 29);
    let inputs: Vec<&[f32]> = vec![&a, &b];
    let reference: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let (program, tir_out) = compile_tensor_ir(ctx, &expr, &inputs, &reference)?;
    let tir_err = validate_err("elementwise", "tensor_ir", &tir_out, &reference)?;

    let ta = Tensor::from_slice(&a, (rows as usize, cols as usize), device)
        .map_err(|e| e.to_string())?;
    let tb = Tensor::from_slice(&b, (rows as usize, cols as usize), device)
        .map_err(|e| e.to_string())?;
    let candle_out = ta
        .add(&tb)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?;
    let candle_err = validate_err("elementwise", "candle", &candle_out, &reference)?;

    let tir_gpu_stats = time_tir_gpu(ctx, &program, &inputs, cfg)?;
    let tir_host_stats = time_tir_host(ctx, &program, &inputs, cfg)?;
    let tir_batched_stats = time_tir_host_batched(ctx, &program, &inputs, cfg)?;
    let candle_stats = time_runs(cfg, || {
        let c = ta.add(&tb).expect("candle add");
        drop(c);
        candle_sync(device);
    });

    Ok(Row {
        op: "elementwise",
        shape: format!("{rows}x{cols}"),
        tir_gpu: tir_gpu_stats,
        tir_host: tir_host_stats,
        tir_batched: tir_batched_stats,
        candle_host: candle_stats,
        tir_err,
        candle_err,
        dispatches: program.dispatches.len(),
    })
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn run_reduce_sum(
    ctx: &GpuContext,
    device: &Device,
    cfg: RunCfg,
    rows: u32,
    cols: u32,
) -> Result<Row, String> {
    let expr = build_reduce_sum_expr(rows, cols);
    let input = seq_f32((rows * cols) as usize, 19);
    let inputs: Vec<&[f32]> = vec![&input];
    let reference = cpu_reduce_sum_reference(rows, cols, &input);
    let (program, tir_out) = compile_tensor_ir(ctx, &expr, &inputs, &reference)?;
    let tir_err = validate_err("reduce_sum", "tensor_ir", &tir_out, &reference)?;

    let tx = Tensor::from_slice(&input, (rows as usize, cols as usize), device)
        .map_err(|e| e.to_string())?;
    let candle_out = tx
        .sum(1)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string())?;
    let candle_err = validate_err("reduce_sum", "candle", &candle_out, &reference)?;

    let tir_gpu_stats = time_tir_gpu(ctx, &program, &inputs, cfg)?;
    let tir_host_stats = time_tir_host(ctx, &program, &inputs, cfg)?;
    let tir_batched_stats = time_tir_host_batched(ctx, &program, &inputs, cfg)?;
    let candle_stats = time_runs(cfg, || {
        let o = tx.sum(1).expect("candle reduce_sum");
        drop(o);
        candle_sync(device);
    });

    Ok(Row {
        op: "reduce_sum",
        shape: format!("{rows}x{cols}"),
        tir_gpu: tir_gpu_stats,
        tir_host: tir_host_stats,
        tir_batched: tir_batched_stats,
        candle_host: candle_stats,
        tir_err,
        candle_err,
        dispatches: program.dispatches.len(),
    })
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn parse_arg<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> T {
    args.get(index)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(all(feature = "runtime", feature = "candle"))]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let cfg = RunCfg {
        warmup: parse_arg(&args, 1, 3u32),
        timing: parse_arg(&args, 2, 20u32),
        tir_batch: parse_arg(&args, 3, 64u32),
    };

    let ctx = GpuContext::new();
    let device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("candle Metal device unavailable: {e}");
            eprintln!("candle_benchmarks requires a Metal-capable GPU.");
            std::process::exit(1);
        }
    };

    println!("=== Tensor IR vs candle-core ===");
    println!(
        "tensor_ir adapter: {} ({:?})",
        ctx.adapter_info().name,
        ctx.adapter_info().backend
    );
    println!("candle device:     {:?}", device.location());
    println!(
        "warmup={} timing={} tir_batch={} | tir_gpu=GPU timestamp; tir_host/candle=sync host; tir_batch=amortized host",
        cfg.warmup, cfg.timing, cfg.tir_batch
    );
    println!();
    print_header();

    let runs: Vec<Result<Row, String>> = vec![
        run_sgemv(&ctx, &device, cfg, 128, 256),
        run_sgemm(&ctx, &device, cfg, 256, 256, 256),
        run_flash_attention(&ctx, &device, cfg, 32, 32),
        run_softmax(&ctx, &device, cfg, 1024, 1024),
        run_elementwise_add(&ctx, &device, cfg, 1024, 1024),
        run_reduce_sum(&ctx, &device, cfg, 256, 256),
    ];

    for r in runs {
        match r {
            Ok(row) => print_row(&row),
            Err(err) => println!("FAILED: {err}"),
        }
    }
}
