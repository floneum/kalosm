#[cfg(not(feature = "runtime"))]
fn main() {
    eprintln!(
        "Enable the runtime feature: cargo run -r -p tensor_ir --features runtime --example runtime_benchmarks"
    );
    std::process::exit(1);
}

#[cfg(feature = "runtime")]
use std::panic::{self, AssertUnwindSafe};

#[cfg(feature = "runtime")]
use tensor_ir::*;

#[cfg(feature = "runtime")]
#[path = "common/refs.rs"]
mod common_refs;
#[cfg(feature = "runtime")]
use common_refs::{
    cpu_attention_reference, cpu_matmul_reference, cpu_reduce_sum_reference, cpu_softmax_reference,
};

#[cfg(feature = "runtime")]
fn catch_quiet_unwind<T>(f: impl FnOnce() -> T) -> std::thread::Result<T> {
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(AssertUnwindSafe(f));
    panic::set_hook(hook);
    result
}

#[cfg(feature = "runtime")]
#[derive(Clone)]
struct BenchmarkCase {
    name: &'static str,
    expr: TensorExprProgram,
    inputs: Vec<Vec<f32>>,
    expected: Option<Vec<f32>>,
}

#[cfg(feature = "runtime")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warmup_runs = parse_arg(&args, 1, 3u32);
    let timing_runs = parse_arg(&args, 2, 10u32);
    let tuning_candidate_limit = parse_arg(&args, 3, 8usize);
    let tuning_warmup_runs = parse_arg(&args, 4, 1u32);
    let tuning_timing_runs = parse_arg(&args, 5, 2u32);

    let ctx = GpuContext::new();
    println!("=== Tensor IR GPU Benchmarks ===");
    println!(
        "adapter: {} ({:?})",
        ctx.adapter_info().name,
        ctx.adapter_info().backend
    );
    println!("warmup_runs: {warmup_runs}");
    println!("timing_runs: {timing_runs}");
    println!("tuning_candidate_limit: {tuning_candidate_limit}");
    println!("tuning_warmup_runs: {tuning_warmup_runs}");
    println!("tuning_timing_runs: {tuning_timing_runs}");
    println!();

    let pipeline = StagedPipeline::default();
    let cases = benchmark_cases();

    for case in cases {
        let program = match build_dispatch_program(
            &pipeline,
            &ctx,
            &case,
            tuning_candidate_limit,
            tuning_warmup_runs,
            tuning_timing_runs,
        ) {
            Ok(simd) => simd,
            Err(err) => {
                println!("{:<20} lowering failed: {}", case.name, err);
                continue;
            }
        };

        let inputs: Vec<&[f32]> = case.inputs.iter().map(Vec::as_slice).collect();
        match ctx.benchmark(
            &program,
            &inputs,
            ProgramBenchmarkConfig {
                warmup_runs,
                timing_runs,
            },
        ) {
            Ok(result) => println!(
                "{:<20} median_gpu_us={:>10.2} min={:>10.2} max={:>10.2} dispatches={} simdgroups={} outputs={} tg_buffers={}",
                case.name,
                result.median_gpu_us,
                result.min_gpu_us,
                result.max_gpu_us,
                program.dispatches.len(),
                program
                    .dispatches
                    .first()
                    .map(|d| d.simdgroups)
                    .unwrap_or(0),
                program
                    .dispatches
                    .first()
                    .map(|d| d.outputs.len())
                    .unwrap_or(0),
                program
                    .dispatches
                    .first()
                    .map(|d| d.tg_buffers.len())
                    .unwrap_or(0),
            ),
            Err(err) => println!("{:<20} benchmark failed: {}", case.name, err),
        }
    }
}

#[cfg(feature = "runtime")]
fn build_dispatch_program(
    pipeline: &StagedPipeline,
    ctx: &GpuContext,
    case: &BenchmarkCase,
    tuning_candidate_limit: usize,
    tuning_warmup_runs: u32,
    tuning_timing_runs: u32,
) -> Result<DispatchProgram, String> {
    let Some(expected) = case.expected.as_ref() else {
        let kernel = pipeline.lower(&case.expr)?;
        return Ok(pipeline.compile(kernel)?.into_dispatch_program());
    };

    let recexpr = tensor_expr_to_recexpr(&case.expr)?;
    let mut egraph = TensorEGraph::default();
    let root = egraph.add_expr(&recexpr);
    egraph.rebuild();

    let runner = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: tensor_ir::LoweringOptions::default(),
    };
    let egraph = saturate(egraph, &runner);

    let inputs: Vec<&[f32]> = case.inputs.iter().map(Vec::as_slice).collect();
    let target_valid_candidates = tuning_candidate_limit.max(1);
    let search_limit = target_valid_candidates.saturating_mul(8);
    let candidates = beam_extract_valid_candidates(
        &egraph,
        root,
        &BeamConfig::default(),
        &DeviceProfile::default(),
        &tensor_ir::LoweringOptions::default(),
        search_limit,
    );
    let mut best: Option<(f64, u32, usize, usize, DispatchProgram)> = None;
    let tuning_config = ProgramBenchmarkConfig {
        warmup_runs: tuning_warmup_runs.max(1),
        timing_runs: tuning_timing_runs.max(1),
    };
    let total_candidates = candidates.len();
    let mut valid_candidates = 0usize;
    println!(
        "{:<20} search candidates={} target_valid={}",
        case.name, total_candidates, target_valid_candidates
    );

    for (index, (_cost, expr)) in candidates.into_iter().enumerate() {
        if index == 0 || (index + 1) == total_candidates || (index + 1) % 8 == 0 {
            println!(
                "{:<20} scanning candidate {}/{}",
                case.name,
                index + 1,
                total_candidates
            );
        }
        let program = build_dispatch_program_from_extracted(
            &expr,
            egraph.clone(),
            &DeviceProfile::default(),
            &tensor_ir::LoweringOptions::default(),
        );
        let Some(dispatch) = program.dispatches.first() else {
            continue;
        };
        let gpu_result = match catch_quiet_unwind(|| ctx.execute(&program, &inputs)) {
            Ok(result) => result,
            Err(_) => continue,
        };
        let max_err = gpu_result[..expected.len()]
            .iter()
            .zip(expected)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f32, f32::max);
        if max_err >= 1e-3 {
            continue;
        }
        valid_candidates += 1;
        if valid_candidates == 1
            || valid_candidates == target_valid_candidates
            || valid_candidates.is_multiple_of(8)
        {
            println!(
                "{:<20} valid candidate {}/{}",
                case.name, valid_candidates, target_valid_candidates
            );
        }
        let result = match catch_quiet_unwind(|| ctx.benchmark(&program, &inputs, tuning_config)) {
            Ok(Ok(result)) => result,
            Ok(Err(_)) | Err(_) => continue,
        };

        let physical_workgroups = dispatch.workgroups / dispatch.simdgroups.max(1);
        let score = result.median_gpu_us;

        match &best {
            Some((best_score, best_workgroups, best_tg_buffers, best_outputs, _))
                if score > *best_score
                    || (score == *best_score
                        && (physical_workgroups > *best_workgroups
                            || (physical_workgroups == *best_workgroups
                                && dispatch.tg_buffers.len() < *best_tg_buffers)
                            || (physical_workgroups == *best_workgroups
                                && dispatch.tg_buffers.len() == *best_tg_buffers
                                && dispatch.outputs.len() <= *best_outputs))) => {}
            _ => {
                best = Some((
                    score,
                    physical_workgroups,
                    dispatch.tg_buffers.len(),
                    dispatch.outputs.len(),
                    program,
                ))
            }
        }

        if valid_candidates >= target_valid_candidates {
            break;
        }
    }

    best.map(|(_, _, _, _, program)| program)
        .ok_or_else(|| format!("no valid {} candidates", case.name))
}

#[cfg(feature = "runtime")]
fn parse_arg<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> T {
    args.get(index)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "runtime")]
fn benchmark_cases() -> Vec<BenchmarkCase> {
    vec![
        build_matmul_case(256, 256, 256),
        build_reduce_sum_case(2048, 1024),
        build_reduce_max_case(2048, 1024),
        build_elementwise_add_case(1024, 1024),
        build_elementwise_mul_case(1024, 1024),
        build_relu_case(1024, 1024),
        build_softmax_case(1024, 1024),
        build_attention_case(32, 32),
    ]
}

#[cfg(feature = "runtime")]
fn build_matmul_case(m: u32, n: u32, k: u32) -> BenchmarkCase {
    let mut builder = TensorExprBuilder::new();
    let a = builder.input(0, Shape(vec![Dim::Lit(m), Dim::Lit(k)]), DType::F32);
    let b = builder.input(1, Shape(vec![Dim::Lit(k), Dim::Lit(n)]), DType::F32);
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let root = builder.contraction(
        Shape(vec![Dim::Lit(m), Dim::Lit(n), Dim::Lit(k)]),
        &[
            (a, Strides(vec![i64::from(k), 0, 1])),
            (b, Strides(vec![0, 1, i64::from(n)])),
        ],
        body,
        &[(2, ReduceOp::Add)],
    );
    let input_a = seq_f32((m * k) as usize, 17);
    let input_b = seq_f32((k * n) as usize, 31);
    BenchmarkCase {
        name: "matmul",
        expr: builder.build(root).expect("valid matmul benchmark"),
        inputs: vec![input_a.clone(), input_b.clone()],
        expected: Some(cpu_matmul_reference(m, n, k, &input_a, &input_b)),
    }
}

#[cfg(feature = "runtime")]
fn build_reduce_sum_case(rows: u32, cols: u32) -> BenchmarkCase {
    let mut builder = TensorExprBuilder::new();
    let x = builder.input(0, Shape(vec![Dim::Lit(rows), Dim::Lit(cols)]), DType::F32);
    let root = builder.reduce(x, 1, ReduceOp::Add);
    let input = seq_f32((rows * cols) as usize, 19);
    BenchmarkCase {
        name: "reduce_sum",
        expr: builder.build(root).expect("valid reduce benchmark"),
        expected: Some(cpu_reduce_sum_reference(rows, cols, &input)),
        inputs: vec![input],
    }
}

#[cfg(feature = "runtime")]
fn build_reduce_max_case(rows: u32, cols: u32) -> BenchmarkCase {
    let mut builder = TensorExprBuilder::new();
    let x = builder.input(0, Shape(vec![Dim::Lit(rows), Dim::Lit(cols)]), DType::F32);
    let root = builder.reduce(x, 1, ReduceOp::Max);
    BenchmarkCase {
        name: "reduce_max",
        expr: builder.build(root).expect("valid reduce benchmark"),
        inputs: vec![signed_seq_f32((rows * cols) as usize, 23)],
        expected: None,
    }
}

#[cfg(feature = "runtime")]
fn build_elementwise_add_case(rows: u32, cols: u32) -> BenchmarkCase {
    let mut builder = TensorExprBuilder::new();
    let shape = Shape(vec![Dim::Lit(rows), Dim::Lit(cols)]);
    let a = builder.input(0, shape.clone(), DType::F32);
    let b = builder.input(1, shape.clone(), DType::F32);
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.scalar_binop(BinaryOp::Add, [arg0, arg1]);
    let root = builder.elementwise(shape, &[a, b], body);
    BenchmarkCase {
        name: "elementwise_add",
        expr: builder.build(root).expect("valid add benchmark"),
        inputs: vec![
            seq_f32((rows * cols) as usize, 13),
            seq_f32((rows * cols) as usize, 29),
        ],
        expected: None,
    }
}

#[cfg(feature = "runtime")]
fn build_elementwise_mul_case(rows: u32, cols: u32) -> BenchmarkCase {
    let mut builder = TensorExprBuilder::new();
    let shape = Shape(vec![Dim::Lit(rows), Dim::Lit(cols)]);
    let a = builder.input(0, shape.clone(), DType::F32);
    let b = builder.input(1, shape.clone(), DType::F32);
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let root = builder.elementwise(shape, &[a, b], body);
    BenchmarkCase {
        name: "elementwise_mul",
        expr: builder.build(root).expect("valid mul benchmark"),
        inputs: vec![
            seq_f32((rows * cols) as usize, 37),
            seq_f32((rows * cols) as usize, 41),
        ],
        expected: None,
    }
}

#[cfg(feature = "runtime")]
fn build_relu_case(rows: u32, cols: u32) -> BenchmarkCase {
    let mut builder = TensorExprBuilder::new();
    let shape = Shape(vec![Dim::Lit(rows), Dim::Lit(cols)]);
    let x = builder.input(0, shape.clone(), DType::F32);
    let arg0 = builder.scalar_arg(0);
    let zero = builder.scalar_f32(0.0);
    let body = builder.scalar_binop(BinaryOp::Max, [arg0, zero]);
    let root = builder.elementwise(shape, &[x], body);
    BenchmarkCase {
        name: "relu",
        expr: builder.build(root).expect("valid relu benchmark"),
        inputs: vec![signed_seq_f32((rows * cols) as usize, 11)],
        expected: None,
    }
}

#[cfg(feature = "runtime")]
fn build_softmax_case(rows: u32, cols: u32) -> BenchmarkCase {
    let mut builder = TensorExprBuilder::new();
    let shape = Shape(vec![Dim::Lit(rows), Dim::Lit(cols)]);
    let x = builder.input(0, shape.clone(), DType::F32);
    let root = builder.softmax(x, shape, 1);
    let input = signed_seq_f32((rows * cols) as usize, 7);
    BenchmarkCase {
        name: "softmax",
        expr: builder.build(root).expect("valid softmax benchmark"),
        expected: Some(cpu_softmax_reference(rows, cols, &input)),
        inputs: vec![input],
    }
}

#[cfg(feature = "runtime")]
fn build_attention_case(seq: u32, d: u32) -> BenchmarkCase {
    // softmax(Q · Kᵀ, axis=1) · V, expressed as two matmul decompositions with
    // an intermediate softmax — mirrors examples/egraph_visualize.rs:385.
    let mut builder = TensorExprBuilder::new();
    let q = builder.input(0, Shape(vec![Dim::Lit(seq), Dim::Lit(d)]), DType::F32);
    let k = builder.input(1, Shape(vec![Dim::Lit(seq), Dim::Lit(d)]), DType::F32);
    let v = builder.input(2, Shape(vec![Dim::Lit(seq), Dim::Lit(d)]), DType::F32);

    // Scores = Q · Kᵀ over tile [seq, seq, d], reducing the d axis.
    let qk_tile = Shape(vec![Dim::Lit(seq), Dim::Lit(seq), Dim::Lit(d)]);
    let q_r = builder.restride(q, qk_tile.clone(), Strides(vec![i64::from(d), 0, 1]));
    let k_r = builder.restride(k, qk_tile.clone(), Strides(vec![0, i64::from(d), 1]));
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let mul_body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let qk_mul = builder.elementwise(qk_tile, &[q_r, k_r], mul_body);
    let scores = builder.reduce(qk_mul, 2, ReduceOp::Add);

    // Probs = softmax(Scores, axis=1) over [seq, seq].
    let scores_shape = Shape(vec![Dim::Lit(seq), Dim::Lit(seq)]);
    let probs = builder.softmax(scores, scores_shape, 1);

    // Output = Probs · V over tile [seq, d, seq], reducing the inner seq axis.
    let pv_tile = Shape(vec![Dim::Lit(seq), Dim::Lit(d), Dim::Lit(seq)]);
    let p_r = builder.restride(probs, pv_tile.clone(), Strides(vec![i64::from(seq), 0, 1]));
    let v_r = builder.restride(v, pv_tile.clone(), Strides(vec![0, 1, i64::from(d)]));
    let arg0 = builder.scalar_arg(0);
    let arg1 = builder.scalar_arg(1);
    let mul_body = builder.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let pv_mul = builder.elementwise(pv_tile, &[p_r, v_r], mul_body);
    let root = builder.reduce(pv_mul, 2, ReduceOp::Add);

    let q_input = seq_f32((seq * d) as usize, 11);
    let k_input = seq_f32((seq * d) as usize, 13);
    let v_input = seq_f32((seq * d) as usize, 17);
    let expected = cpu_attention_reference(seq, d, &q_input, &k_input, &v_input);
    BenchmarkCase {
        name: "attention",
        expr: builder.build(root).expect("valid attention benchmark"),
        inputs: vec![q_input, k_input, v_input],
        expected: Some(expected),
    }
}

#[cfg(feature = "runtime")]
fn seq_f32(len: usize, modulus: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % modulus) as f32) / modulus as f32)
        .collect()
}

#[cfg(feature = "runtime")]
fn signed_seq_f32(len: usize, modulus: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i % modulus) as f32 - (modulus as f32 / 2.0)) / modulus as f32)
        .collect()
}
