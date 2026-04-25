//! Runtime fuzz checks for executable rewrite candidates.

use std::panic::{self, AssertUnwindSafe};

use crate::extractor::BeamConfig;
use crate::language::TensorIr;
use crate::rules::{self, Phase, RunnerConfig, SaturationReport};
use crate::skeleton::{beam_extract_valid_candidates, build_dispatch_program_from_extracted};
use crate::types::{DType, DeviceProfile, Dim, LoweringOptions, Shape, ShapeParams};

#[derive(Clone, Copy, Debug)]
struct MatmulCase {
    m: u32,
    n: u32,
    k: u32,
    seed: u32,
}

fn fuzz_data(len: u32, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let mixed = index
                .wrapping_mul(1_664_525)
                .wrapping_add(seed.wrapping_mul(1_013_904_223))
                .rotate_left((seed % 23) + 1);
            let signed = (mixed % 2048) as i32 - 1024;
            signed as f32 / 1024.0
        })
        .collect()
}

fn matmul_reference(m: u32, n: u32, k: u32, a: &[f32], b: &[f32]) -> Vec<f32> {
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

fn max_abs_err(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn matmul_candidates(
    case: MatmulCase,
    candidate_limit: usize,
) -> (
    crate::TensorEGraph,
    SaturationReport,
    Vec<(f64, egg::RecExpr<TensorIr>)>,
) {
    let mut builder = crate::IrBuilder::new();
    let lhs = builder.input(
        0,
        Shape(vec![Dim::Const(case.m), Dim::Const(case.k)]),
        DType::F32,
    );
    let rhs = builder.input(
        1,
        Shape(vec![Dim::Const(case.k), Dim::Const(case.n)]),
        DType::F32,
    );
    super::build_binary_mul_add_contraction_ir(&mut builder, lhs, rhs, case.m, case.n, case.k);

    let mut egraph = crate::TensorEGraph::default();
    let root = egraph.add_expr(&builder.expr);
    egraph.rebuild();
    let runner = RunnerConfig {
        iter_limit: 30,
        node_limit: 100_000,
        time_limit_secs: 60,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let (egraph, saturation) = rules::saturate_phases_reported(egraph, Phase::all(), &runner);
    let beam = BeamConfig {
        beam_width: 24,
        ..BeamConfig::default()
    };
    let candidates = beam_extract_valid_candidates(
        &egraph,
        root,
        &beam,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        candidate_limit,
    );
    (egraph, saturation, candidates)
}

#[test]
fn fuzz_all_runnable_matmul_transformations_match_original_implementation() {
    std::thread::Builder::new()
        .name("runtime-fuzz-matmul-candidates".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(run_matmul_candidate_fuzz)
        .expect("spawn runtime fuzz thread")
        .join()
        .expect("runtime fuzz thread panicked");
}

fn run_matmul_candidate_fuzz() {
    const CASES: &[MatmulCase] = &[
        MatmulCase {
            m: 8,
            n: 8,
            k: 8,
            seed: 0x0000_10ad,
        },
        MatmulCase {
            m: 16,
            n: 16,
            k: 16,
            seed: 0x0000_b17e,
        },
        MatmulCase {
            m: 32,
            n: 16,
            k: 16,
            seed: 0x0000_f00d,
        },
        MatmulCase {
            m: 8,
            n: 8,
            k: 7,
            seed: 0x0000_0007,
        },
        MatmulCase {
            m: 8,
            n: 8,
            k: 13,
            seed: 0x0000_0013,
        },
        MatmulCase {
            m: 8,
            n: 8,
            k: 31,
            seed: 0x0000_0031,
        },
        MatmulCase {
            m: 8,
            n: 4,
            k: 257,
            seed: 0x0000_0257,
        },
    ];
    const CANDIDATE_LIMIT: usize = 16;
    const TOLERANCE: f32 = 1e-3;

    let ctx = crate::runtime::GpuContext::new();

    for case in CASES {
        let (egraph, saturation, candidates) = matmul_candidates(*case, CANDIDATE_LIMIT);
        let stop_reasons = saturation
            .phases
            .iter()
            .map(|phase| format!("{:?}: {}", phase.phase, phase.stop_reason))
            .collect::<Vec<_>>()
            .join(", ");
        assert!(
            !candidates.is_empty(),
            "expected at least one runnable candidate for {case:?}; stop_reasons=[{stop_reasons}]"
        );

        let lhs = fuzz_data(case.m * case.k, case.seed ^ 0xa5a5_5a5a);
        let rhs = fuzz_data(case.k * case.n, case.seed ^ 0x5a5a_a5a5);
        let inputs: [&[f32]; 2] = [&lhs, &rhs];
        let expected = matmul_reference(case.m, case.n, case.k, &lhs, &rhs);
        let mut checked = 0usize;

        for (candidate_index, (_cost, expr)) in candidates.iter().enumerate() {
            let program = build_dispatch_program_from_extracted(
                expr,
                egraph.clone(),
                &DeviceProfile::default(),
                &LoweringOptions::default(),
            );
            assert!(
                !program.dispatches.is_empty(),
                "candidate {candidate_index} for {case:?} had no runnable dispatches"
            );
            let gpu_output = panic::catch_unwind(AssertUnwindSafe(|| {
                ctx.execute(&program, &inputs, &ShapeParams::default())
            }))
            .unwrap_or_else(|_| {
                panic!("candidate {candidate_index} for {case:?} panicked while running")
            });

            assert!(
                gpu_output.len() >= expected.len(),
                "candidate {candidate_index} for {case:?} produced {} values, expected at least {}",
                gpu_output.len(),
                expected.len()
            );

            let max_err = max_abs_err(&gpu_output[..expected.len()], &expected);
            assert!(
                max_err.is_finite() && max_err < TOLERANCE,
                "candidate {candidate_index} for {case:?} differed from original implementation: max_err={max_err}; stop_reasons=[{stop_reasons}]"
            );
            checked += 1;
        }

        assert!(
            checked > 0,
            "expected at least one runnable candidate for {case:?}"
        );
    }
}
