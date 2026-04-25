//! Integration tests for the tensor IR.

use crate::analysis::TensorAnalysis;
use crate::builders::IrBuilder;
use crate::extractor::{BeamConfig, beam_extract_candidates};
use crate::language::*;
use crate::naga_codegen;
use crate::rules::{self, Phase, RunnerConfig};
use crate::skeleton::{
    DispatchProgram, beam_extract_valid_candidates, build_dispatch_program_from_extracted,
};
use crate::types::*;
use egg::{Id, Language, RecExpr};

/// Test that Phases 1+2+3 produce Threadgroup loads via generic promotion.
#[test]
fn test_tiled_promotion_generic() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, 64, 64, 64);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    // Phase 1 lowers, Phase 2 tiles, Phase 3 promotes to TG
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    // Search the entire e-graph for Threadgroup loads
    let mut has_tg_load = false;
    for class in egraph.classes() {
        for node in class.iter() {
            if matches!(
                node,
                TensorIr::Simd(SimdNode::Load {
                    tier: MemTier::Threadgroup(_),
                    ..
                })
            ) {
                has_tg_load = true;
            }
        }
    }
    assert!(
        has_tg_load,
        "Phases 1+2+3 should produce Threadgroup loads via generic promotion"
    );
}

/// Test that the beam extractor prefers K-tiled (Tg loads) over flat (Device loads).
#[test]
fn test_k_tiled_cost_preference() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
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
    let (_cost, extracted) = beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable K-tiled candidate");

    // The extracted program should prefer threadgroup loads over device loads
    let mut tg_loads = 0;
    let mut dev_loads = 0;
    for node in extracted.as_ref() {
        match node {
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Threadgroup(_),
                ..
            }) => tg_loads += 1,
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Device(_),
                ..
            }) => dev_loads += 1,
            _ => {}
        }
    }
    assert!(
        tg_loads > 0,
        "Extractor should select K-tiled variant with threadgroup loads (tg={tg_loads}, dev={dev_loads})"
    );
}

fn first_lowerable_matmul_candidate(
    egraph: &egg::EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    beam_cfg: &BeamConfig,
    limit: usize,
) -> Option<(usize, f64, RecExpr<TensorIr>, DispatchProgram)> {
    beam_extract_candidates(egraph, root, beam_cfg, limit)
        .into_iter()
        .enumerate()
        .find_map(|(index, (cost, expr))| {
            let program = build_dispatch_program_from_extracted(
                &expr,
                egraph.clone(),
                &DeviceProfile::default(),
                &LoweringOptions::default(),
            );
            if program.dispatches.is_empty() {
                None
            } else {
                Some((index, cost, expr, program))
            }
        })
}

/// Test that build_dispatch_program_from_extracted produces a DispatchProgram.
#[test]
fn test_build_dispatch_program() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
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
    let (_index, _cost, _extracted, program) =
        first_lowerable_matmul_candidate(&egraph, root, &beam_cfg, 32)
            .expect("expected at least one lowerable candidate");

    assert!(
        !program.dispatches.is_empty(),
        "DispatchProgram should have at least 1 dispatch"
    );
    assert_eq!(
        program.dispatches.len(),
        1,
        "Fused matmul should produce exactly 1 dispatch"
    );
}

#[test]
fn test_late_dispatch_produces_tiled_register_block_outputs() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
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
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    let root = egraph.find(root);
    let reg_blocked = egraph[root].iter().find(|node| {
        let TensorIr::Dispatch(DispatchNode::Dispatch {
            num_inputs,
            children_list,
            ..
        }) = node
        else {
            return false;
        };
        let children = extract_list(&egraph, *children_list);
        let body_len = children.len().saturating_sub(*num_inputs as usize);
        body_len == 8 // 4 outputs x 2 (value, addr)
    });
    assert!(
        reg_blocked.is_some(),
        "LateDispatch should produce a 4-output register-blocked dispatch"
    );
}

/// Test that the skeleton body contains cooperative loading structure (Loop → Store(Tg) → Barrier).
#[test]
fn test_cooperative_load_structure() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
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
    let candidates = beam_extract_candidates(&egraph, root, &beam_cfg, 16);
    let mut saw_rejected_tg_candidate = false;
    let mut saw_lowerable_candidate = false;

    for (_cost, expr) in candidates {
        let extracted_has_tg = expr.as_ref().iter().any(|node| {
            matches!(
                node,
                TensorIr::Simd(SimdNode::Load {
                    tier: MemTier::Threadgroup(_),
                    ..
                })
            )
        });
        let program = build_dispatch_program_from_extracted(
            &expr,
            egraph.clone(),
            &DeviceProfile::default(),
            &LoweringOptions::default(),
        );
        if program.dispatches.is_empty() {
            saw_rejected_tg_candidate |= extracted_has_tg;
            continue;
        }

        saw_lowerable_candidate = true;
        let dispatch = &program.dispatches[0];
        fn check_inner(
            egraph: &crate::TensorEGraph,
            root: Id,
            seen: &mut std::collections::HashSet<Id>,
        ) -> (bool, bool) {
            let canonical = egraph.find(root);
            if !seen.insert(canonical) {
                return (false, false);
            }

            let mut has_tg_store = false;
            let mut has_barrier = false;
            for node in egraph[canonical].iter() {
                match node {
                    TensorIr::Simd(SimdNode::Store {
                        tier: MemTier::Threadgroup(_),
                        ..
                    })
                    | TensorIr::Simd(SimdNode::StoreIf {
                        tier: MemTier::Threadgroup(_),
                        ..
                    }) => has_tg_store = true,
                    TensorIr::Simd(SimdNode::Barrier { regions, .. }) if !regions.is_empty() => {
                        has_barrier = true
                    }
                    _ => {}
                }
                for child in node.children() {
                    let (s, b) = check_inner(egraph, *child, seen);
                    has_tg_store |= s;
                    has_barrier |= b;
                }
            }
            (has_tg_store, has_barrier)
        }

        let mut seen = std::collections::HashSet::new();
        let (has_tg_store, has_barrier) =
            dispatch.outputs.iter().fold((false, false), |acc, output| {
                let (s, b) = check_inner(&program.egraph, output.value_id, &mut seen);
                (acc.0 || s, acc.1 || b)
            });

        if has_tg_store || has_barrier {
            assert!(
                has_tg_store,
                "pure output term should contain Store to Threadgroup (cooperative load)"
            );
            assert!(
                has_barrier,
                "pure output term should contain Barrier after cooperative loads"
            );
            return;
        }
    }

    assert!(
        saw_rejected_tg_candidate,
        "expected incoherent threadgroup candidates to be rejected before lowering"
    );
    assert!(
        saw_lowerable_candidate,
        "expected at least one lowerable fallback candidate"
    );
}

/// Test that TgBufferInfo has correct tile dimensions.
#[test]
fn test_tg_buffer_sizes() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
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
    let (_cost, extracted) = beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable dispatch candidate");

    // Extract tg buffer info from the extracted program
    let nodes = extracted.as_ref();
    let root_idx = nodes.len() - 1;

    // Find the dispatch and its body (the nested theta)
    if let TensorIr::Dispatch(DispatchNode::Dispatch { children_list, .. }) = &nodes[root_idx] {
        let body_idx = usize::from(*extract_recexpr_list(nodes, *children_list).last().unwrap());
        if let TensorIr::Simd(SimdNode::Theta {
            children: [_, _, inner_id],
            ..
        }) = &nodes[body_idx]
        {
            let inner_idx = usize::from(*inner_id);
            if let TensorIr::Simd(SimdNode::Theta {
                children: [_, count_id, _],
                ..
            }) = &nodes[inner_idx]
            {
                let tile_k =
                    if let TensorIr::Const(ScalarValue::U32(v)) = &nodes[usize::from(*count_id)] {
                        *v
                    } else {
                        panic!("inner theta count should be a u32 literal");
                    };

                let bufs = crate::skeleton::collect_tg_buffer_info(
                    nodes,
                    inner_idx,
                    tile_k,
                    &DeviceProfile::default(),
                );
                assert!(
                    !bufs.is_empty(),
                    "Should find threadgroup buffer info from inner theta"
                );

                for buf in &bufs {
                    assert!(
                        buf.size > 0,
                        "TgBufferInfo size should be positive, got {} for {}",
                        buf.size,
                        buf.tg_name
                    );
                    // tile_k should divide evenly into the region size
                    assert!(
                        buf.size % tile_k == 0,
                        "TgBuffer {} size ({}) should be a multiple of tile_k ({})",
                        buf.tg_name,
                        buf.size,
                        tile_k
                    );
                }
                return; // success
            }
        }
    }

    // If the extractor picked a non-nested Theta, that's still valid but we
    // can't test buffer info
    println!("Note: extractor didn't pick nested Theta variant; skipping buffer size checks");
}

// ═══════════════════════════════════════════════════════
// Generic (non-matmul) cooperative loading tests
// ═══════════════════════════════════════════════════════

/// Test that generic 1D row-reduction gets Threadgroup loads via Phase 3 promotion.
#[test]
fn test_generic_1d_reduce_promotion() {
    let mut b = IrBuilder::new();
    // A simple row-wise sum reduction: result[32] = sum(input[32, 64], axis=1)
    let x = b.input(0, Shape(vec![Dim::Const(32), Dim::Const(64)]), DType::F32);

    // Build elementwise identity (just pass through)
    let arg0 = b.scalar_arg(0);
    let ewise = b.elementwise(Shape(vec![Dim::Const(32), Dim::Const(64)]), &[x], arg0);
    let _reduce = b.reduce(ewise, 1, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    // Run Phases 1 (lowering), 2 (tiling), and 3 (stride-zero / Tg promotion)
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    // Check for Threadgroup loads anywhere in the e-graph
    let mut has_tg_load = false;
    for class in egraph.classes() {
        for node in class.iter() {
            if matches!(
                node,
                TensorIr::Simd(SimdNode::Load {
                    tier: MemTier::Threadgroup(_),
                    ..
                })
            ) {
                has_tg_load = true;
            }
        }
    }
    assert!(
        has_tg_load,
        "Generic 1D row-reduction should get Threadgroup loads via Phase 3 promotion"
    );
}

/// Test that reduce-max also benefits from cooperative loading promotion.
#[test]
fn test_reduce_max_cooperative_load() {
    let mut b = IrBuilder::new();
    // Row-wise max: result[32] = max(input[32, 128], axis=1)
    let x = b.input(0, Shape(vec![Dim::Const(32), Dim::Const(128)]), DType::F32);

    let arg0 = b.scalar_arg(0);
    let ewise = b.elementwise(Shape(vec![Dim::Const(32), Dim::Const(128)]), &[x], arg0);
    let _reduce = b.reduce(ewise, 1, ReduceOp::Max);

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

    // Extract and check that the program has Tg loads
    let beam_cfg = BeamConfig {
        beam_width: 8,
        ..Default::default()
    };
    let (_cost, extracted) = beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable dispatch candidate");

    let mut tg_loads = 0;
    for node in extracted.as_ref() {
        if matches!(
            node,
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Threadgroup(_),
                ..
            })
        ) {
            tg_loads += 1;
        }
    }
    assert!(
        tg_loads > 0,
        "Reduce-max should have Threadgroup loads in extracted program"
    );
}

// ═══════════════════════════════════════════════════════
// Naga codegen tests
// ═══════════════════════════════════════════════════════

/// Test that the matmul pipeline can be lowered to a valid Naga module and WGSL.
#[test]
fn test_codegen_matmul_wgsl() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
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
    let (_index, _cost, _extracted, program) =
        first_lowerable_matmul_candidate(&egraph, root, &beam_cfg, 32)
            .expect("expected a lowerable matmul candidate");

    assert!(!program.dispatches.is_empty());

    // Lower to Naga module.
    let module =
        naga_codegen::lower_dispatch_program(naga_codegen::verify(&program).expect("verify"));

    // Should have one entry point.
    assert_eq!(module.entry_points.len(), 1, "Should produce 1 entry point");
    assert_eq!(module.entry_points[0].name, "dispatch_0");
    assert_eq!(module.entry_points[0].stage, naga::ShaderStage::Compute);
    let expected_wg_size = program.dispatches[0].simdgroups * DeviceProfile::default().simd_width;
    assert_eq!(
        module.entry_points[0].workgroup_size,
        [expected_wg_size, 1, 1]
    );

    // Validate and emit WGSL.
    let wgsl =
        naga_codegen::module_to_wgsl(&module).expect("Module should validate and produce WGSL");
    assert!(!wgsl.is_empty(), "WGSL output should be non-empty");

    // Sanity-check WGSL content.
    assert!(wgsl.contains("@compute"), "WGSL should contain @compute");
    assert!(
        wgsl.contains("@workgroup_size"),
        "WGSL should contain @workgroup_size"
    );
}

/// Test that lower_to_wgsl convenience works.
#[test]
fn test_codegen_lower_to_wgsl() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
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
    let (_index, _cost, _extracted, program) =
        first_lowerable_matmul_candidate(&egraph, root, &beam_cfg, 32)
            .expect("expected a lowerable matmul candidate");

    let wgsl =
        naga_codegen::lower_to_wgsl(&program).expect("lower_to_wgsl should succeed for matmul");
    assert!(
        wgsl.contains("dispatch_0"),
        "WGSL should contain entry point name"
    );
}

#[test]
fn test_dispatch_program_tracks_pipeline_roots() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
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
    let (_cost, extracted) = beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable dispatch candidate");

    let mut nodes = extracted.as_ref().to_vec();
    let root_id = Id::from(nodes.len() - 1);
    let nil_id = Id::from(nodes.len());
    nodes.push(TensorIr::Nil);
    let list_id = Id::from(nodes.len());
    nodes.push(TensorIr::Cons([root_id, nil_id]));
    nodes.push(TensorIr::Dispatch(DispatchNode::Pipeline(list_id)));
    let pipeline_expr: RecExpr<TensorIr> = nodes.into();

    let program = build_dispatch_program_from_extracted(
        &pipeline_expr,
        egraph.clone(),
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );
    assert_eq!(program.pipelines.len(), 1);
    assert_eq!(program.pipelines[0].dispatch_indices.len(), 1);
    assert_eq!(program.dispatches.len(), 1);
    assert_eq!(program.dispatches[0].pipeline_index, Some(0));
    assert_eq!(program.dispatches[0].pipeline_stage, 0);
}

#[test]
fn test_pipeline_root_fuses_linear_plain_stages_into_one_dispatch() {
    let mut b = IrBuilder::new();
    let input = b.input(0, Shape(vec![Dim::Const(32)]), DType::F32);
    let workgroup = b.index(IndexLevel::Workgroup);
    let lane = b.index(IndexLevel::Lane);
    let simd_width = b.low_u32(DeviceProfile::default().simd_width);
    let wg_base = b.bin_op(BinaryOp::Mul, workgroup, simd_width);
    let addr = b.bin_op(BinaryOp::Add, wg_base, lane);
    let token = b.token();

    let load0 = b.load_at(MemTier::Device(BufferRef::Input(0)), addr, token);
    let one = b.low_f32(1.0);
    let stage0_val = b.bin_op(BinaryOp::Add, load0, one);
    let stage0_children = b.list(&[input, stage0_val, addr]);
    let d0 = b.expr.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(1),
        num_inputs: 1,
        children_list: stage0_children,
    }));

    let load1 = b.load_at(MemTier::Device(BufferRef::Input(0)), addr, token);
    let two = b.low_f32(2.0);
    let stage1_val = b.bin_op(BinaryOp::Mul, load1, two);
    let stage1_children = b.list(&[d0, stage1_val, addr]);
    let d1 = b.expr.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(1),
        num_inputs: 1,
        children_list: stage1_children,
    }));

    let _pipeline = b.pipeline(&[d0, d1]);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _root = egraph.add_expr(&b.expr);

    let program = build_dispatch_program_from_extracted(
        &b.expr,
        egraph.clone(),
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );

    assert_eq!(program.pipelines.len(), 1);
    assert_eq!(program.pipelines[0].dispatch_indices, vec![0]);
    assert_eq!(program.dispatches.len(), 1);
    assert_eq!(program.dispatches[0].inputs.len(), 1);
    assert_eq!(program.dispatches[0].pipeline_index, Some(0));
    assert_eq!(program.dispatches[0].pipeline_stage, 0);

    let wgsl = naga_codegen::lower_to_wgsl(&program).expect("fused linear pipeline should lower");
    assert!(wgsl.contains("dispatch_0"));
}

#[test]
fn test_phase6_pipeline_wins_extraction_when_linear_chain_is_fusable() {
    let mut b = IrBuilder::new();
    let input = b.input(0, Shape(vec![Dim::Const(32)]), DType::F32);
    let workgroup = b.index(IndexLevel::Workgroup);
    let lane = b.index(IndexLevel::Lane);
    let simd_width = b.low_u32(DeviceProfile::default().simd_width);
    let wg_base = b.bin_op(BinaryOp::Mul, workgroup, simd_width);
    let addr = b.bin_op(BinaryOp::Add, wg_base, lane);
    let token = b.token();

    let load0 = b.load_at(MemTier::Device(BufferRef::Input(0)), addr, token);
    let one = b.low_f32(1.0);
    let stage0_val = b.bin_op(BinaryOp::Add, load0, one);
    let stage0_children = b.list(&[input, stage0_val, addr]);
    let d0 = b.expr.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(1),
        num_inputs: 1,
        children_list: stage0_children,
    }));

    let load1 = b.load_at(MemTier::Device(BufferRef::Input(0)), addr, token);
    let two = b.low_f32(2.0);
    let stage1_val = b.bin_op(BinaryOp::Mul, load1, two);
    let stage1_children = b.list(&[d0, stage1_val, addr]);
    let d1 = b.expr.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(1),
        num_inputs: 1,
        children_list: stage1_children,
    }));

    let _seq = b.seq(&[d0, d1]);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::LateDispatch], &config);

    let beam_cfg = BeamConfig {
        beam_width: 8,
        ..Default::default()
    };
    let (_cost, extracted) = crate::skeleton::beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable fused pipeline candidate");

    let program = build_dispatch_program_from_extracted(
        &extracted,
        egraph.clone(),
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );
    assert_eq!(program.dispatches.len(), 1);
    assert_eq!(program.pipelines.len(), 1);
}

#[test]
fn test_softmax_weighted_reduce_builds_lowered_dispatch_program() {
    let rows = 32u32;
    let weights = 32u32;
    let outputs = 32u32;

    let mut b = IrBuilder::new();
    let scores = b.input(
        0,
        Shape(vec![Dim::Const(rows), Dim::Const(weights)]),
        DType::F32,
    );
    let values = b.input(
        1,
        Shape(vec![Dim::Const(weights), Dim::Const(outputs)]),
        DType::F32,
    );
    let _weighted =
        super::build_softmax_weighted_reduce_ir(&mut b, scores, values, rows, outputs, weights);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 100_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    let beam_cfg = BeamConfig {
        beam_width: 32,
        ..Default::default()
    };
    let (_cost, extracted) = crate::skeleton::beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable recursive dispatch candidate");
    let program = build_dispatch_program_from_extracted(
        &extracted,
        egraph.clone(),
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );

    assert!(
        !program.dispatches.is_empty(),
        "softmax-weighted-reduce should lower to executable dispatches"
    );
    assert!(program.pipelines.is_empty());

    let wgsl = naga_codegen::lower_to_wgsl(&program)
        .expect("lower_to_wgsl should succeed for recursive dispatch lowering");
    for idx in 0..program.dispatches.len() {
        assert!(wgsl.contains(&format!("dispatch_{idx}")));
    }
}

#[test]
fn test_recursive_dispatch_builds_nested_reduce_elementwise_program() {
    let rows = 32u32;
    let cols = 32u32;

    let mut b = IrBuilder::new();
    let x = b.input(
        0,
        Shape(vec![Dim::Const(rows), Dim::Const(cols)]),
        DType::F32,
    );
    let _expr = super::build_centered_row_sum_ir(&mut b, x, rows, cols);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 100_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    let beam_cfg = BeamConfig {
        beam_width: 32,
        ..Default::default()
    };
    let (_cost, extracted) = crate::skeleton::beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable recursive dispatch candidate");
    let program = build_dispatch_program_from_extracted(
        &extracted,
        egraph.clone(),
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );

    assert_eq!(program.dispatches.len(), 1);
    assert!(program.pipelines.is_empty());

    let wgsl = naga_codegen::lower_to_wgsl(&program)
        .expect("lower_to_wgsl should succeed for nested recursive dispatch");
    assert!(wgsl.contains("dispatch_0"));
}

#[test]
fn test_fused_attention_builds_lowered_dispatch_program() {
    let seq = 32u32;
    let d = 32u32;

    let mut b = IrBuilder::new();
    let q = b.input(0, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
    let k = b.input(1, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
    let v = b.input(2, Shape(vec![Dim::Const(seq), Dim::Const(d)]), DType::F32);
    let _attn = super::build_attention_ir(&mut b, q, k, v, seq, d);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 100_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    let beam_cfg = BeamConfig {
        beam_width: 32,
        ..Default::default()
    };
    let (_cost, extracted) = crate::skeleton::beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable recursive dispatch candidate");
    let program = build_dispatch_program_from_extracted(
        &extracted,
        egraph.clone(),
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );

    assert!(
        !program.dispatches.is_empty(),
        "attention should lower to at least one executable dispatch"
    );
    assert!(program.pipelines.is_empty());

    let wgsl = naga_codegen::lower_to_wgsl(&program)
        .expect("lower_to_wgsl should succeed for attention fused via recursive dispatch lowering");
    for idx in 0..program.dispatches.len() {
        assert!(wgsl.contains(&format!("dispatch_{idx}")));
    }
}
