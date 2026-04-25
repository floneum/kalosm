//! Integration tests for the tensor IR.

use crate::analysis::TensorAnalysis;
use crate::builders::IrBuilder;
use crate::extractor::{BeamConfig, beam_extract, beam_extract_candidates, greedy_extract};
use crate::language::*;
use crate::pipeline::{
    StageConfig, StagedPipeline, compile_kernel, lower_tensor_expr, lower_tensor_expr_with_report,
};
use crate::rules::{self, Phase, RunnerConfig};
use crate::stages::TensorExprBuilder;
use crate::types::*;
use egg::Language;

/// Test that we can build and add a simple elementwise expression.
#[test]
fn test_build_elementwise() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(64), Dim::Lit(128)]), DType::F32);
    let arg0 = b.scalar_arg(0);
    let body = b.un_op(UnaryOp::Exp, arg0);
    let _ewise = b.elementwise(Shape(vec![Dim::Lit(64), Dim::Lit(128)]), &[a], body);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _root = egraph.add_expr(&b.expr);
    assert!(egraph.total_size() > 0);
}

/// Test that we can build a matmul expression.
#[test]
fn test_build_matmul() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(64), Dim::Lit(128)]), DType::F32);
    let b_input = b.input(1, Shape(vec![Dim::Lit(128), Dim::Lit(256)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_input, 64, 256, 128);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    // The root should be a Reduce node
    let root_class = &egraph[root];
    assert!(
        root_class
            .iter()
            .any(|n| matches!(n, TensorIr::HighLevel(HighLevelNode::Reduce { .. })))
    );
}

#[test]
fn test_tensor_expr_summary_tracks_lowering_inputs() {
    let mut b = TensorExprBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(64), Dim::Lit(128)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Lit(128), Dim::Lit(256)]), DType::F32);
    let mm = super::build_binary_mul_add_contraction_expr(&mut b, a, b_in, 64, 256, 128);
    let expr = b.build(mm).expect("valid tensor expr");

    let summary = expr.summary().expect("summary");
    assert_eq!(
        summary.output_shape,
        Some(Shape(vec![Dim::Lit(64), Dim::Lit(256)]))
    );
    assert_eq!(summary.input_count, 2);
    assert!(summary.has_reduce);
    assert!(summary.has_elementwise);
}

#[test]
fn test_staged_pipeline_produces_kernel_without_high_level_ops() {
    let mut b = TensorExprBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(64), Dim::Lit(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Lit(64), Dim::Lit(64)]), DType::F32);
    let mm = super::build_binary_mul_add_contraction_expr(&mut b, a, b_in, 64, 64, 64);
    let expr = b.build(mm).expect("valid tensor expr");

    let pipeline = StagedPipeline::default();
    let kernel = lower_tensor_expr(&expr, pipeline.config()).expect("kernel");
    let simd = compile_kernel(kernel.clone()).expect("simd");

    assert!(!simd.dispatch_program().dispatches.is_empty());
    for node in kernel.extracted().as_ref() {
        assert!(
            !matches!(
                node,
                TensorIr::HighLevel(HighLevelNode::Restride { .. })
                    | TensorIr::HighLevel(HighLevelNode::Elementwise { .. })
                    | TensorIr::HighLevel(HighLevelNode::Reduce { .. })
            ),
            "kernel stage should not contain high-level compute nodes: {node:?}"
        );
    }
}

#[test]
fn test_lower_tensor_expr_with_report_records_phase_and_candidate_stats() {
    let mut b = TensorExprBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(128)]), DType::F32);
    let arg = b.scalar_arg(0);
    let body = b.scalar_unop(UnaryOp::Exp, arg);
    let root = b.elementwise(Shape(vec![Dim::Lit(128)]), &[a], body);
    let expr = b.build(root).expect("valid tensor expr");

    let mut config = StageConfig::default();
    config.runner.iter_limit = 5;
    config.runner.node_limit = 20_000;
    config.runner.time_limit_secs = 10;

    let (kernel, report) =
        lower_tensor_expr_with_report(&expr, &config).expect("reported lowering");

    assert!(report.error.is_none());
    assert_eq!(report.input_nodes, expr.nodes().len());
    assert_eq!(
        report.summary.as_ref().map(|summary| summary.input_count),
        Some(1)
    );
    assert_eq!(report.saturation.phases.len(), Phase::all().len());
    assert!(
        report
            .saturation
            .phases
            .iter()
            .all(|phase| phase.nodes_after >= phase.nodes_before)
    );

    let extraction = report.extraction.expect("extraction report");
    assert_eq!(extraction.selected_cost, Some(kernel.cost()));
    assert!(extraction.selected_nodes.is_some());
    assert!(extraction.candidate_validation.raw_candidates > 0);
    assert!(extraction.candidate_validation.returned > 0);
}

#[test]
fn test_scalar_lane_3d_elementwise_lowering() {
    let mut b = TensorExprBuilder::new();
    let a = b.input(
        0,
        Shape(vec![Dim::Lit(3), Dim::Lit(16), Dim::Lit(16)]),
        DType::F32,
    );
    let c = b.input(
        1,
        Shape(vec![Dim::Lit(3), Dim::Lit(16), Dim::Lit(16)]),
        DType::F32,
    );
    let c = b.restride(
        c,
        Shape(vec![Dim::Lit(3), Dim::Lit(16), Dim::Lit(16)]),
        Strides(vec![0, 0, 0]),
    );
    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let body = b.scalar_binop(BinaryOp::Mul, [arg0, arg1]);
    let root = b.elementwise(
        Shape(vec![Dim::Lit(3), Dim::Lit(16), Dim::Lit(16)]),
        &[a, c],
        body,
    );
    let expr = b.build(root).expect("valid tensor expr");
    let mut config = StageConfig::default();
    config.runner.device.simd_width = 1;
    config.runner.device.max_simdgroups = 1;
    lower_tensor_expr(&expr, &config).expect("3d scalar-lane elementwise lowers");
}

#[test]
fn test_lower_tensor_expr_error_carries_partial_report() {
    let mut b = TensorExprBuilder::new();
    let root = b.input(0, Shape(vec![Dim::Lit(4)]), DType::U32);
    let expr = b.build(root).expect("valid tensor expr");

    let err = lower_tensor_expr_with_report(&expr, &StageConfig::default())
        .expect_err("u32 backend unsupported");

    assert!(err.message.contains("supports only f32"));
    assert_eq!(err.report.error.as_deref(), Some(err.message.as_str()));
    assert_eq!(err.report.input_nodes, expr.nodes().len());
    assert!(err.report.saturation.phases.is_empty());
    assert!(err.report.extraction.is_none());
}

#[test]
fn test_lower_tensor_expr_rejects_non_f32_tensor_inputs() {
    let mut b = TensorExprBuilder::new();
    let root = b.input(0, Shape(vec![Dim::Lit(64)]), DType::U32);
    let expr = b.build(root).expect("valid tensor expr");

    let pipeline = StagedPipeline::default();
    let err = lower_tensor_expr(&expr, pipeline.config()).expect_err("u32 backend unsupported");
    assert!(
        err.contains("f32"),
        "error should describe the backend dtype restriction, got: {err}"
    );
}

#[test]
fn test_lower_tensor_expr_errors_when_no_executable_rewrite_candidate_exists() {
    let mut b = TensorExprBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(64)]), DType::F32);
    let arg0 = b.scalar_arg(0);
    let body = b.scalar_unop(UnaryOp::Exp, arg0);
    let ewise = b.elementwise(Shape(vec![Dim::Lit(64)]), &[a], body);
    let expr = b.build(ewise).expect("valid tensor expr");

    let mut config = StageConfig::default();
    config.runner.iter_limit = 0;

    let err = lower_tensor_expr(&expr, &config)
        .expect_err("no lowering phases should produce no executable kernel");
    assert!(
        err.contains("no valid executable kernel candidates"),
        "expected no-candidate error, got: {err}"
    );
}

/// `DeviceProfile::max_threadgroup_bytes` must constrain lowering. The
/// expected behavior is two-layered:
///   1. The tile provider drops configs whose A+B tile bytes exceed the
///      budget (compile-time exploration win).
///   2. If a candidate slips through anyway, the post-extraction hard-reject
///      in `lower_tensor_expr` fires (defense-in-depth).
///
/// Together, the chosen kernel's `peak_threadgroup_bytes` is always within the
/// configured budget, regardless of which layer enforced it.
#[test]
fn test_device_budget_constrains_lowering() {
    let mut b = TensorExprBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(64), Dim::Lit(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Lit(64), Dim::Lit(64)]), DType::F32);
    let mm = super::build_binary_mul_add_contraction_expr(&mut b, a, b_in, 64, 64, 64);
    let expr = b.build(mm).expect("valid tensor expr");

    let pipeline = StagedPipeline::default();

    // Default profile fits and picks a tiled, TG-using kernel.
    let default_kernel =
        lower_tensor_expr(&expr, pipeline.config()).expect("default device must accept kernel");
    let default_program = crate::skeleton::build_dispatch_program_from_extracted(
        default_kernel.extracted(),
        default_kernel.egraph().clone(),
        default_kernel.device(),
        &pipeline.config().runner.lowering,
    );
    let default_tg = default_program.peak_threadgroup_bytes();
    assert!(
        default_tg > 0,
        "default-device matmul should pick a TG-using tile; got {default_tg} bytes"
    );

    // Squeeze budget below any 2D matmul tile (smallest config: 16x16 tile_k=16
    // → 16*16*4 = 1024 bytes per tile, 2 tiles = 2048). At 256 bytes the
    // provider drops every tile config and phase 1 must fall back to the
    // non-TG lowering.
    let mut tight_cfg = pipeline.config().clone();
    tight_cfg.runner.device.max_threadgroup_bytes = 256;
    let tight_kernel =
        lower_tensor_expr(&expr, &tight_cfg).expect("tight device should still produce a kernel");
    let tight_program = crate::skeleton::build_dispatch_program_from_extracted(
        tight_kernel.extracted(),
        tight_kernel.egraph().clone(),
        tight_kernel.device(),
        &tight_cfg.runner.lowering,
    );
    let tight_tg = tight_program.peak_threadgroup_bytes();
    assert!(
        tight_tg <= u64::from(tight_cfg.runner.device.max_threadgroup_bytes),
        "chosen kernel's peak_threadgroup_bytes ({tight_tg}) must respect the budget ({})",
        tight_cfg.runner.device.max_threadgroup_bytes
    );
}

/// Test that we can build a softmax expression.
#[test]
fn test_build_softmax() {
    let mut b = IrBuilder::new();
    let shape = Shape(vec![Dim::Lit(32), Dim::Lit(128)]);
    let x = b.input(0, shape.clone(), DType::F32);
    let _sm = b.softmax(x, shape, 1);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _root = egraph.add_expr(&b.expr);
    assert!(egraph.total_size() > 0);
}

#[test]
fn test_composite_dispatch_analysis_marks_generic_nested_reduce_tree() {
    let rows = 32u32;
    let cols = 64u32;

    let mut b = IrBuilder::new();
    let x = b.input(0, Shape(vec![Dim::Lit(rows), Dim::Lit(cols)]), DType::F32);
    let _expr = super::build_centered_row_sum_ir(&mut b, x, rows, cols);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);
    egraph.rebuild();

    let data = &egraph[egraph.find(root)].data;
    assert!(
        data.composite_dispatch.lowerable,
        "analysis should mark a generic nested reduce/elementwise tree as composite-dispatch lowerable"
    );
}

/// Test dependence analysis on basic expressions.
#[test]
fn test_dep_analysis() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();

    // Index(Lane) should have dep = {Lane}
    let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
        IndexLevel::Lane,
    ))));
    egraph.rebuild();
    assert!(egraph[lane].data.dep.contains_lane());
    assert!(!egraph[lane].data.dep.contains_simdgroup());

    // Index(Workgroup) should have dep = {Workgroup}
    let wg = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
        IndexLevel::Workgroup,
    ))));
    egraph.rebuild();
    assert!(egraph[wg].data.dep.contains_workgroup());

    // Const should have dep = {}
    let lit = egraph.add(TensorIr::Const(ScalarValue::U32(42)));
    egraph.rebuild();
    assert_eq!(egraph[lit].data.dep, DepSet::EMPTY);

    // Op(add, [lane, wg]) should have dep = {Lane, Workgroup}
    let add = egraph.add(TensorIr::BinOp(BinaryOp::Add, [lane, wg]));
    egraph.rebuild();
    assert!(egraph[add].data.dep.contains_lane());
    assert!(egraph[add].data.dep.contains_workgroup());
    assert!(!egraph[add].data.dep.contains_simdgroup());
}

/// Test shape propagation through analysis.
#[test]
fn test_shape_propagation() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();

    let input = egraph.add(TensorIr::HighLevel(HighLevelNode::Input {
        id: 0,
        shape: Shape(vec![Dim::Lit(64), Dim::Lit(128)]),
        dtype: DType::F32,
    }));
    egraph.rebuild();

    let data = &egraph[input].data;
    assert_eq!(data.shape, Some(Shape(vec![Dim::Lit(64), Dim::Lit(128)])));
    assert_eq!(data.dtype, Some(DType::F32));
}

/// Test constant folding in analysis.
#[test]
fn test_constant_folding() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();

    let a = egraph.add(TensorIr::Const(ScalarValue::I32(3)));
    let b = egraph.add(TensorIr::Const(ScalarValue::I32(4)));
    let add = egraph.add(TensorIr::BinOp(BinaryOp::Add, [a, b]));
    egraph.rebuild();

    assert_eq!(egraph[add].data.constant, Some(ScalarValue::I32(7)));
}

/// Test greedy extraction round-trip.
#[test]
fn test_greedy_extraction() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(32)]), DType::F32);
    let arg0 = b.scalar_arg(0);
    let body = b.un_op(UnaryOp::Exp, arg0);
    let _ewise = b.elementwise(Shape(vec![Dim::Lit(32)]), &[a], body);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);
    egraph.rebuild();

    let (cost, extracted) = greedy_extract(&egraph, root);
    assert!(cost < f64::INFINITY);
    assert!(!extracted.as_ref().is_empty());
}

/// Test beam extraction with multiple candidates.
#[test]
fn test_beam_extraction() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(32)]), DType::F32);
    let arg0 = b.scalar_arg(0);
    let body = b.un_op(UnaryOp::Exp, arg0);
    let _ewise = b.elementwise(Shape(vec![Dim::Lit(32)]), &[a], body);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);
    egraph.rebuild();

    let config = BeamConfig {
        beam_width: 4,
        ..Default::default()
    };
    let (cost, _extracted) = beam_extract(&egraph, root, &config);
    assert!(cost < f64::INFINITY);
}

#[test]
fn test_beam_extract_candidates_order_and_uniqueness() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(64), Dim::Lit(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Lit(64), Dim::Lit(64)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, 64, 64, 64);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    let beam_cfg = BeamConfig {
        beam_width: 8,
        ..Default::default()
    };
    let (best_cost, best_expr) = beam_extract(&egraph, root, &beam_cfg);
    let candidates = beam_extract_candidates(&egraph, root, &beam_cfg, 6);

    assert!(!candidates.is_empty(), "expected at least one candidate");
    assert_eq!(candidates[0].0, best_cost);
    assert_eq!(candidates[0].1, best_expr);

    let mut seen = std::collections::HashSet::new();
    let mut prev_cost = f64::NEG_INFINITY;
    for (cost, expr) in &candidates {
        assert!(
            *cost >= prev_cost,
            "candidate costs should be nondecreasing"
        );
        prev_cost = *cost;
        assert!(
            seen.insert(format!("{expr:?}")),
            "candidate extractions should be unique"
        );
    }
}

#[test]
fn test_beam_extract_candidates_are_acyclic_for_tiled_kernels() {
    let cases = [(64, 64, 64), (128, 1, 256)];
    for (m, n, k) in cases {
        let mut b = IrBuilder::new();
        let a = b.input(0, Shape(vec![Dim::Lit(m), Dim::Lit(k)]), DType::F32);
        let b_in = b.input(1, Shape(vec![Dim::Lit(k), Dim::Lit(n)]), DType::F32);
        let _kernel = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, m, n, k);

        let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
        let root = egraph.add_expr(&b.expr);

        let config = RunnerConfig {
            iter_limit: 10,
            node_limit: 50_000,
            time_limit_secs: 30,
            device: DeviceProfile::default(),
            lowering: LoweringOptions::default_const(),
        };
        let egraph =
            rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

        let beam_cfg = BeamConfig {
            beam_width: 8,
            ..Default::default()
        };
        let candidates = beam_extract_candidates(&egraph, root, &beam_cfg, 6);
        assert!(
            !candidates.is_empty(),
            "expected candidates for shape ({m}, {n}, {k})"
        );

        for (_, expr) in candidates {
            for (idx, node) in expr.as_ref().iter().enumerate() {
                for child in node.children() {
                    assert!(
                        usize::from(*child) < idx,
                        "RecExpr child indices must point backward: idx={idx}, child={}, node={node:?}",
                        usize::from(*child)
                    );
                }
            }
        }
    }
}

#[test]
fn test_beam_extract_candidates_find_deep_alternative() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();

    let one = egraph.add(TensorIr::Const(ScalarValue::U32(1)));
    let two = egraph.add(TensorIr::Const(ScalarValue::U32(2)));
    let three = egraph.add(TensorIr::Const(ScalarValue::U32(3)));
    let four = egraph.add(TensorIr::Const(ScalarValue::U32(4)));
    let five = egraph.add(TensorIr::Const(ScalarValue::U32(5)));
    let deep_alt = egraph.add(TensorIr::BinOp(BinaryOp::Add, [one, two]));
    egraph.union(three, deep_alt);
    egraph.rebuild();

    let inner = egraph.find(three);
    let sum = egraph.add(TensorIr::BinOp(BinaryOp::Add, [four, inner]));
    let root = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [sum, five]));
    egraph.rebuild();

    let beam_cfg = BeamConfig {
        beam_width: 4,
        ..Default::default()
    };
    let candidates = beam_extract_candidates(&egraph, root, &beam_cfg, 4);

    assert!(
        candidates.len() >= 2,
        "expected the beam search to surface the deeper alternative"
    );
    assert_eq!(candidates[0].1.as_ref().len(), 5);

    let add_count = candidates[1]
        .1
        .as_ref()
        .iter()
        .filter(|node| matches!(node, TensorIr::BinOp(name, _) if *name == BinaryOp::Add))
        .count();
    assert_eq!(
        add_count, 2,
        "second candidate should differ below the root's immediate children"
    );
}

#[test]
fn test_beam_extract_accounts_for_dispatch_execution_multiplicity() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();

    let zero = egraph.add(TensorIr::Const(ScalarValue::U32(0)));
    let one = egraph.add(TensorIr::Const(ScalarValue::U32(1)));
    let two = egraph.add(TensorIr::Const(ScalarValue::U32(2)));
    let three = egraph.add(TensorIr::Const(ScalarValue::U32(3)));
    let four = egraph.add(TensorIr::Const(ScalarValue::U32(4)));
    let five = egraph.add(TensorIr::Const(ScalarValue::U32(5)));
    let six = egraph.add(TensorIr::Const(ScalarValue::U32(6)));
    let seven = egraph.add(TensorIr::Const(ScalarValue::U32(7)));

    let cheap_body = egraph.add(TensorIr::BinOp(BinaryOp::Add, [one, two]));
    let deep_0 = egraph.add(TensorIr::BinOp(BinaryOp::Add, [cheap_body, three]));
    let deep_1 = egraph.add(TensorIr::BinOp(BinaryOp::Add, [deep_0, four]));
    let deep_2 = egraph.add(TensorIr::BinOp(BinaryOp::Add, [deep_1, five]));
    let deep_3 = egraph.add(TensorIr::BinOp(BinaryOp::Add, [deep_2, six]));
    let expensive_body = egraph.add(TensorIr::BinOp(BinaryOp::Add, [deep_3, seven]));

    let slow_children = add_list(&mut egraph, &[expensive_body, zero]);
    let slow_dispatch = egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: 1,
        num_inputs: 0,
        children_list: slow_children,
    }));
    let fast_children = add_list(&mut egraph, &[cheap_body, zero]);
    let fast_dispatch = egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: 128,
        num_inputs: 0,
        children_list: fast_children,
    }));

    egraph.union(slow_dispatch, fast_dispatch);
    egraph.rebuild();

    let beam_cfg = BeamConfig {
        beam_width: 4,
        ..Default::default()
    };
    let (_cost, extracted) = beam_extract(&egraph, egraph.find(slow_dispatch), &beam_cfg);

    let TensorIr::Dispatch(DispatchNode::Dispatch { workgroups, .. }) =
        extracted.as_ref().last().expect("dispatch root")
    else {
        panic!(
            "expected dispatch root, got {:?}",
            extracted.as_ref().last()
        );
    };
    assert_eq!(
        *workgroups, 1,
        "beam extractor should prefer the dispatch with less repeated body work"
    );
}
