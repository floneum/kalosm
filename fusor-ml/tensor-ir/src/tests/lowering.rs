//! Integration tests for the tensor IR.

use crate::analysis::TensorAnalysis;
use crate::builders::IrBuilder;
use crate::extractor::{BeamConfig, beam_extract, greedy_extract};
use crate::language::*;
use crate::rules::{self, Phase, RunnerConfig};
use crate::types::*;

fn has_binop(
    egraph: &egg::EGraph<TensorIr, TensorAnalysis>,
    id: egg::Id,
    op: BinaryOp,
    lhs: egg::Id,
    rhs: egg::Id,
) -> bool {
    let lhs = egraph.find(lhs);
    let rhs = egraph.find(rhs);
    egraph[egraph.find(id)].iter().any(|node| {
        matches!(
            node,
            TensorIr::BinOp(name, [a, b])
                if *name == op && egraph.find(*a) == lhs && egraph.find(*b) == rhs
        )
    })
}

fn has_right_rotation(
    egraph: &egg::EGraph<TensorIr, TensorAnalysis>,
    root: egg::Id,
    op: BinaryOp,
    a: egg::Id,
    b: egg::Id,
    c: egg::Id,
) -> bool {
    let a = egraph.find(a);
    egraph[egraph.find(root)].iter().any(|node| {
        let TensorIr::BinOp(name, [lhs, rhs]) = node else {
            return false;
        };
        *name == op && egraph.find(*lhs) == a && has_binop(egraph, *rhs, op, b, c)
    })
}

#[test]
fn test_binary_op_associativity_metadata() {
    for op in [
        BinaryOp::Add,
        BinaryOp::Mul,
        BinaryOp::Max,
        BinaryOp::Min,
        BinaryOp::And,
        BinaryOp::Or,
        BinaryOp::Xor,
    ] {
        assert!(op.is_associative(), "{op} should be associative");
    }

    for op in [
        BinaryOp::Sub,
        BinaryOp::Div,
        BinaryOp::Mod,
        BinaryOp::Pow,
        BinaryOp::Shl,
        BinaryOp::Shr,
        BinaryOp::Eq,
        BinaryOp::Neq,
        BinaryOp::Lt,
        BinaryOp::Le,
        BinaryOp::Gt,
        BinaryOp::Ge,
    ] {
        assert!(!op.is_associative(), "{op} should not be associative");
    }

    assert_eq!(ReduceOp::from_bin_op(BinaryOp::Add), Some(ReduceOp::Add));
    assert_eq!(ReduceOp::from_bin_op(BinaryOp::Mul), Some(ReduceOp::Mul));
    assert_eq!(ReduceOp::from_bin_op(BinaryOp::Max), Some(ReduceOp::Max));
    assert_eq!(ReduceOp::from_bin_op(BinaryOp::Min), Some(ReduceOp::Min));
    assert_eq!(ReduceOp::from_bin_op(BinaryOp::And), Some(ReduceOp::And));
    assert_eq!(ReduceOp::from_bin_op(BinaryOp::Or), Some(ReduceOp::Or));
    assert_eq!(ReduceOp::from_bin_op(BinaryOp::Xor), Some(ReduceOp::Xor));
    assert_eq!(
        ReduceOp::And.identity(DType::Bool),
        Some(ScalarValue::Bool(true))
    );
    assert_eq!(
        ReduceOp::Or.identity(DType::Bool),
        Some(ScalarValue::Bool(false))
    );
    assert_eq!(ReduceOp::Xor.identity(DType::Bool), None);
    assert_eq!(
        ReduceOp::And.identity(DType::U32),
        Some(ScalarValue::U32(u32::MAX))
    );
    assert_eq!(
        ReduceOp::Xor.identity(DType::U32),
        Some(ScalarValue::U32(0))
    );
}

#[test]
fn test_phase1_associative_binop_rotate_right_add() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let a = egraph.add(TensorIr::Const(ScalarValue::U32(2)));
    let b = egraph.add(TensorIr::Const(ScalarValue::U32(3)));
    let c = egraph.add(TensorIr::Const(ScalarValue::U32(4)));
    let lhs = egraph.add(TensorIr::BinOp(BinaryOp::Add, [a, b]));
    let root = egraph.add(TensorIr::BinOp(BinaryOp::Add, [lhs, c]));

    let config = RunnerConfig {
        iter_limit: 4,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    assert!(
        has_right_rotation(&egraph, root, BinaryOp::Add, a, b, c),
        "associativity should add a + (b + c)"
    );
}

#[test]
fn test_phase1_associative_binop_rotate_right_and() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let a = egraph.add(TensorIr::Const(ScalarValue::Bool(true)));
    let b = egraph.add(TensorIr::Const(ScalarValue::Bool(false)));
    let c = egraph.add(TensorIr::Const(ScalarValue::Bool(true)));
    let lhs = egraph.add(TensorIr::BinOp(BinaryOp::And, [a, b]));
    let root = egraph.add(TensorIr::BinOp(BinaryOp::And, [lhs, c]));

    let config = RunnerConfig {
        iter_limit: 4,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    assert!(
        has_right_rotation(&egraph, root, BinaryOp::And, a, b, c),
        "associativity should add a & (b & c)"
    );
}

#[test]
fn test_phase1_associative_binop_does_not_rotate_sub() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let a = egraph.add(TensorIr::Const(ScalarValue::U32(8)));
    let b = egraph.add(TensorIr::Const(ScalarValue::U32(3)));
    let c = egraph.add(TensorIr::Const(ScalarValue::U32(1)));
    let lhs = egraph.add(TensorIr::BinOp(BinaryOp::Sub, [a, b]));
    let root = egraph.add(TensorIr::BinOp(BinaryOp::Sub, [lhs, c]));

    let config = RunnerConfig {
        iter_limit: 4,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    assert!(
        !has_right_rotation(&egraph, root, BinaryOp::Sub, a, b, c),
        "non-associative sub should not add a - (b - c)"
    );
}

/// Test that Phase 1 lowering produces Dispatch nodes.
#[test]
fn test_phase1_elementwise_lowering() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(1024)]), DType::F32);
    let arg0 = b.scalar_arg(0);
    let body = b.un_op(UnaryOp::Exp, arg0);
    let _ewise = b.elementwise(Shape(vec![Dim::Const(1024)]), &[a], body);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    // Run only phase 1 rules
    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    // The root class should now also contain a Dispatch node
    let root = egraph.find(root);
    let has_dispatch = egraph[root]
        .iter()
        .any(|n| matches!(n, TensorIr::Dispatch(DispatchNode::Dispatch { .. })));
    assert!(
        has_dispatch,
        "Phase 1 lowering should produce a Dispatch node in the root e-class"
    );
}

#[test]
fn test_phase1_elementwise_register_blocking() {
    let mut b = IrBuilder::new();
    let a = b.input(
        0,
        Shape(vec![Dim::Const(1024), Dim::Const(1024)]),
        DType::F32,
    );
    let arg0 = b.scalar_arg(0);
    let body = b.un_op(UnaryOp::Exp, arg0);
    let _ewise = b.elementwise(Shape(vec![Dim::Const(1024), Dim::Const(1024)]), &[a], body);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 20_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    // Phase 1 now produces only plain, unblocked Dispatches. The
    // register-blocked variants are emitted by the late-dispatch
    // `register_blocking` rewrite, so we run both phases here.
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    let root = egraph.find(root);
    let has_dispatch = egraph[root]
        .iter()
        .any(|n| matches!(n, TensorIr::Dispatch(DispatchNode::Dispatch { .. })));
    assert!(
        has_dispatch,
        "aligned elementwise should lower to at least one Dispatch"
    );

    // After late dispatch, one of the Dispatches should have 4 output
    // pairs (register blocked) — currently produced by the
    // `register_blocking` rewrites under LateDispatch.
    let reg_blocked = egraph[root].iter().any(|n| {
        let TensorIr::Dispatch(DispatchNode::Dispatch {
            num_inputs,
            children_list,
            ..
        }) = n
        else {
            return false;
        };
        let children = extract_list(&egraph, *children_list);
        let body_len = children.len().saturating_sub(*num_inputs as usize);
        body_len == 8
    });
    assert!(
        reg_blocked,
        "aligned elementwise lowering should produce a 4-output (register-blocked) dispatch via LateDispatch"
    );
}

/// Test Theta tiling in Phase 2.
#[test]
fn test_phase2_theta_tiling() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();

    let init = egraph.add(TensorIr::Const(ScalarValue::F32(
        ordered_float::OrderedFloat(0.0),
    )));
    let count = egraph.add(TensorIr::Const(ScalarValue::U32(64)));
    let acc = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(0))));
    let one = egraph.add(TensorIr::Const(ScalarValue::F32(
        ordered_float::OrderedFloat(1.0),
    )));
    let update = egraph.add(TensorIr::BinOp(BinaryOp::Add, [acc, one]));
    let theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
        children: [init, count, update],
    }));

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::LateDispatch], &config);

    // Should now have nested Thetas (tiled versions)
    let theta_class = egraph.find(theta);
    let num_thetas = egraph[theta_class]
        .iter()
        .filter(|n| matches!(n, TensorIr::Simd(SimdNode::Theta { .. })))
        .count();
    assert!(
        num_thetas > 1,
        "Tiling should produce additional Theta nodes. Got {num_thetas}"
    );
}

/// Test tiled load promotion only fires inside a tile-local Theta update.
#[test]
fn test_phase3_thread_uniform_promotion() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();

    let init = egraph.add(TensorIr::Const(ScalarValue::F32(
        ordered_float::OrderedFloat(0.0),
    )));
    let count = egraph.add(TensorIr::Const(ScalarValue::U32(16)));
    let acc = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(0))));

    let outer = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::iter(1))));
    let inner = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::iter(0))));
    let tile = egraph.add(TensorIr::Const(ScalarValue::U32(16)));
    let outer_tile = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [outer, tile]));
    let addr = egraph.add(TensorIr::BinOp(BinaryOp::Add, [outer_tile, inner]));
    let token = egraph.add(TensorIr::Dispatch(DispatchNode::Token));
    let load = egraph.add(TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Device(BufferRef::Input(0)),
        children: [addr, token],
    }));

    // Also build a lane-dependent load — it is not tile-local and should not
    // be promoted.
    let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
        IndexLevel::Lane,
    ))));
    let lane_token = egraph.add(TensorIr::Dispatch(DispatchNode::Token));
    let lane_load = egraph.add(TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Device(BufferRef::Input(1)),
        children: [lane, lane_token],
    }));

    let sum = egraph.add(TensorIr::BinOp(BinaryOp::Add, [acc, load]));
    let update = egraph.add(TensorIr::BinOp(BinaryOp::Add, [sum, lane_load]));
    let _theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
        children: [init, count, update],
    }));

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::LateDispatch], &config);

    // Tile-local load should be promoted to threadgroup.
    let load_class = egraph.find(load);
    let has_tg = egraph[load_class].iter().any(|n| {
        matches!(
            n,
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Threadgroup(_),
                ..
            })
        )
    });
    assert!(
        has_tg,
        "tile-local load inside Theta should be promoted to threadgroup"
    );

    // Lane-dependent load should not be promoted.
    let lane_class = egraph.find(lane_load);
    let lane_has_tg = egraph[lane_class].iter().any(|n| {
        matches!(
            n,
            TensorIr::Simd(SimdNode::Load {
                tier: MemTier::Threadgroup(_),
                ..
            })
        )
    });
    assert!(
        !lane_has_tg,
        "lane-dependent load should not be promoted to threadgroup"
    );
}

/// Test the simdgroup-broadcast rule fires on a TG load whose address
/// partitions the simdgroup into buckets (fewer unique values than
/// `simd_width`). Constructs `addr = (lane / 4) * 8 + 1` so the load has
/// 8 unique values across 32 lanes, and expects a `Shuffle` to be unioned
/// into the load's eclass by `phase5::simdgroup_broadcast`.
#[test]
fn test_phase5_simdgroup_broadcast() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();

    let lane = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(
        IndexLevel::Lane,
    ))));
    let four = egraph.add(TensorIr::Const(ScalarValue::U32(4)));
    let eight = egraph.add(TensorIr::Const(ScalarValue::U32(8)));
    let one = egraph.add(TensorIr::Const(ScalarValue::U32(1)));
    let bucket = egraph.add(TensorIr::BinOp(BinaryOp::Div, [lane, four]));
    let scaled = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [bucket, eight]));
    let addr = egraph.add(TensorIr::BinOp(BinaryOp::Add, [scaled, one]));
    let state = egraph.add(TensorIr::Dispatch(DispatchNode::Token));
    let buf = BufferRef::Input(0);
    let load = egraph.add(TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Device(buf).to_threadgroup(),
        children: [addr, state],
    }));

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::LateDispatch], &config);

    let load_class = egraph.find(load);
    let shuffle = egraph[load_class]
        .iter()
        .find_map(|n| match n {
            TensorIr::Simd(SimdNode::Shuffle([_, src])) => Some(*src),
            _ => None,
        })
        .expect("simdgroup-broadcast should union a Shuffle into the TG load's eclass");
    // Shuffle source should be lane-dependent (not a constant 0 — that's the
    // fully-invariant case handled by broadcast_as_shuffle).
    let src_is_const_zero = matches!(&egraph[shuffle].data.constant, Some(ScalarValue::U32(0)));
    assert!(
        !src_is_const_zero,
        "shuffle source should be lane-dependent (bucket representative), not constant 0"
    );
}

/// Test ReduceSimd → shuffle tree (Rule 11).
#[test]
fn test_phase4_shuffle_tree() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();

    let value = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(0))));
    let rsimd = egraph.add(TensorIr::Simd(SimdNode::ReduceSimd {
        op: ReduceOp::Add,
        src: value,
    }));

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::LateDispatch], &config);

    // The ReduceSimd class should now have BinOp nodes (shuffle tree)
    let rsimd_class = egraph.find(rsimd);
    let has_low_op = egraph[rsimd_class]
        .iter()
        .any(|n| matches!(n, TensorIr::BinOp(..)));
    assert!(has_low_op, "ReduceSimd should be expanded to shuffle tree");
}

/// Stride-table regression: a 3-operand GEMM-family contraction
/// `out[m,n] = Σ_k A[m,k] * B[k,n] * w[k]` exercises the same recognizer
/// as matmul but with `num_inputs == 3` and a non-`mul(arg0,arg1)` body.
/// The axes classify as `[Output, Output, Reduced]` from the operand
/// stride table, so the tiled variant fires regardless of body or the
/// specific inputs.
#[test]
fn test_stride_table_three_operand_gemm_family() {
    let mut b = IrBuilder::new();
    let m = 32u32;
    let n = 32u32;
    let k = 32u32;

    // Inputs. A: [M, K]; B: [K, N]; w: [K].
    let a_in = b.input(0, Shape(vec![Dim::Const(m), Dim::Const(k)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(k), Dim::Const(n)]), DType::F32);
    let w_in = b.input(2, Shape(vec![Dim::Const(k)]), DType::F32);

    // Broadcast each to [M, N, K] via Restride with stride-0 slots on
    // the axes this operand doesn't depend on.
    let a_restrided = b.restride(
        a_in,
        Shape(vec![Dim::Const(m), Dim::Const(n), Dim::Const(k)]),
        Strides(vec![Dim::Const(k), Dim::Const(0), Dim::Const(1)]),
    );
    let b_restrided = b.restride(
        b_in,
        Shape(vec![Dim::Const(m), Dim::Const(n), Dim::Const(k)]),
        Strides(vec![Dim::Const(0), Dim::Const(1), Dim::Const(n)]),
    );
    let w_restrided = b.restride(
        w_in,
        Shape(vec![Dim::Const(m), Dim::Const(n), Dim::Const(k)]),
        Strides(vec![Dim::Const(0), Dim::Const(0), Dim::Const(1)]),
    );

    // Body: arg0 * arg1 * arg2
    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let arg2 = b.scalar_arg(2);
    let ab = b.bin_op(BinaryOp::Mul, arg0, arg1);
    let body = b.bin_op(BinaryOp::Mul, ab, arg2);

    let ewise = b.elementwise(
        Shape(vec![Dim::Const(m), Dim::Const(n), Dim::Const(k)]),
        &[a_restrided, b_restrided, w_restrided],
        body,
    );
    let root_expr = b.reduce(ewise, 2, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 50_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    let root = egraph.find(root);
    let has_dispatch = egraph[root].iter().any(|n| {
        matches!(
            n,
            TensorIr::Dispatch(DispatchNode::Dispatch { num_inputs: 3, .. })
        )
    });
    assert!(
        has_dispatch,
        "Stride-table recognizer should produce a 3-input Dispatch for \
         the 3-operand GEMM-family contraction (matmul+weighted-k); this \
         validates that the refactor isn't matmul-shape-gated."
    );
    // Silence unused warnings for builder-local ids.
    let _ = root_expr;
}

/// Batched contraction regression: `[B,M,K] x [B,K,N] -> [B,M,N]`.
///
/// The arbitrary-rank planner should flatten `(B,M)` into logical `M`,
/// keep `N` as the last non-reduced axis, and still emit the fast-path
/// register-blocked dispatch family.
#[test]
fn test_batched_contraction_lowering() {
    let mut b = IrBuilder::new();
    let batch = 2u32;
    let m = 32u32;
    let n = 32u32;
    let k = 32u32;

    let a_in = b.input(
        0,
        Shape(vec![Dim::Const(batch), Dim::Const(m), Dim::Const(k)]),
        DType::F32,
    );
    let b_in = b.input(
        1,
        Shape(vec![Dim::Const(batch), Dim::Const(k), Dim::Const(n)]),
        DType::F32,
    );

    let a_restrided = b.restride(
        a_in,
        Shape(vec![
            Dim::Const(batch),
            Dim::Const(m),
            Dim::Const(n),
            Dim::Const(k),
        ]),
        Strides(vec![
            Dim::Const(m * k),
            Dim::Const(k),
            Dim::Const(0),
            Dim::Const(1),
        ]),
    );
    let b_restrided = b.restride(
        b_in,
        Shape(vec![
            Dim::Const(batch),
            Dim::Const(m),
            Dim::Const(n),
            Dim::Const(k),
        ]),
        Strides(vec![
            Dim::Const(k * n),
            Dim::Const(0),
            Dim::Const(1),
            Dim::Const(n),
        ]),
    );

    let lhs = b.scalar_arg(0);
    let rhs = b.scalar_arg(1);
    let body = b.bin_op(BinaryOp::Mul, lhs, rhs);
    let ewise = b.elementwise(
        Shape(vec![
            Dim::Const(batch),
            Dim::Const(m),
            Dim::Const(n),
            Dim::Const(k),
        ]),
        &[a_restrided, b_restrided],
        body,
    );
    let _root_expr = b.reduce(ewise, 3, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 50_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    // Register blocking is now encoded as >1 output pairs in the
    // dispatch's children_list. It's produced by the phase-2
    // `register-block-{2,4}` rewrite on top of phase-1 lowering, so
    // this test runs both phases.
    let root = egraph.find(root);
    let has_blocked_dispatch = egraph[root].iter().any(|n| {
        let TensorIr::Dispatch(DispatchNode::Dispatch {
            num_inputs: 2,
            children_list,
            ..
        }) = n
        else {
            return false;
        };
        let children = crate::language::extract_list(&egraph, *children_list);
        let body_len = children.len().saturating_sub(2);
        body_len > 2 // >1 (value, addr) pair = register-blocked
    });
    assert!(
        has_blocked_dispatch,
        "batched contraction should lower through the grouped contraction fast path"
    );
}

/// Multi-axis reduction regression: `Σ_{k0,k1} A[m,k0,k1] * B[k0,k1,n]`.
///
/// This exercises the reduce-axis remapping logic: the outer `Reduce`
/// observes the inner-reduced rank, so its axis id must be translated back
/// into the original elementwise index space before classification.
#[test]
fn test_multi_axis_contraction_lowering() {
    let mut b = IrBuilder::new();
    let m = 32u32;
    let n = 32u32;
    let k0 = 4u32;
    let k1 = 8u32;

    let a_in = b.input(
        0,
        Shape(vec![Dim::Const(m), Dim::Const(k0), Dim::Const(k1)]),
        DType::F32,
    );
    let b_in = b.input(
        1,
        Shape(vec![Dim::Const(k0), Dim::Const(k1), Dim::Const(n)]),
        DType::F32,
    );

    let a_restrided = b.restride(
        a_in,
        Shape(vec![
            Dim::Const(m),
            Dim::Const(n),
            Dim::Const(k0),
            Dim::Const(k1),
        ]),
        Strides(vec![
            Dim::Const(k0 * k1),
            Dim::Const(0),
            Dim::Const(k1),
            Dim::Const(1),
        ]),
    );
    let b_restrided = b.restride(
        b_in,
        Shape(vec![
            Dim::Const(m),
            Dim::Const(n),
            Dim::Const(k0),
            Dim::Const(k1),
        ]),
        Strides(vec![
            Dim::Const(0),
            Dim::Const(1),
            Dim::Const(k1 * n),
            Dim::Const(n),
        ]),
    );

    let lhs = b.scalar_arg(0);
    let rhs = b.scalar_arg(1);
    let body = b.bin_op(BinaryOp::Mul, lhs, rhs);
    let ewise = b.elementwise(
        Shape(vec![
            Dim::Const(m),
            Dim::Const(n),
            Dim::Const(k0),
            Dim::Const(k1),
        ]),
        &[a_restrided, b_restrided],
        body,
    );
    let reduce_k1 = b.reduce(ewise, 3, ReduceOp::Add);
    let _root_expr = b.reduce(reduce_k1, 2, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 50_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    let root = egraph.find(root);
    let has_fused_dispatch = egraph[root].iter().any(|n| {
        matches!(
            n,
            TensorIr::Dispatch(DispatchNode::Dispatch { num_inputs: 2, .. })
        )
    });
    assert!(
        has_fused_dispatch,
        "multi-axis contraction should lower as one fused 2-input dispatch"
    );
}

/// With templates removed, mixed-op nested reductions (e.g. outer Max
/// over inner Add) lower through `recursive-to-dispatch` as a
/// Dispatch with a nested Theta body. The outer Theta's role-tag no
/// longer gates anything — phase-2/4 rules read the body structurally
/// and decline to tile composite bodies on their own merits. This test
/// verifies that a composite-body fused Dispatch is produced and
/// *not* rejected.
#[test]
fn test_multi_axis_contraction_mixed_reduce_ops_fallback() {
    let mut b = IrBuilder::new();
    let m = 32u32;
    let n = 32u32;
    let k0 = 4u32;
    let k1 = 8u32;

    let a_in = b.input(
        0,
        Shape(vec![Dim::Const(m), Dim::Const(k0), Dim::Const(k1)]),
        DType::F32,
    );
    let b_in = b.input(
        1,
        Shape(vec![Dim::Const(k0), Dim::Const(k1), Dim::Const(n)]),
        DType::F32,
    );

    let a_restrided = b.restride(
        a_in,
        Shape(vec![
            Dim::Const(m),
            Dim::Const(n),
            Dim::Const(k0),
            Dim::Const(k1),
        ]),
        Strides(vec![
            Dim::Const(k0 * k1),
            Dim::Const(0),
            Dim::Const(k1),
            Dim::Const(1),
        ]),
    );
    let b_restrided = b.restride(
        b_in,
        Shape(vec![
            Dim::Const(m),
            Dim::Const(n),
            Dim::Const(k0),
            Dim::Const(k1),
        ]),
        Strides(vec![
            Dim::Const(0),
            Dim::Const(1),
            Dim::Const(k1 * n),
            Dim::Const(n),
        ]),
    );

    let lhs = b.scalar_arg(0);
    let rhs = b.scalar_arg(1);
    let body = b.bin_op(BinaryOp::Mul, lhs, rhs);
    let ewise = b.elementwise(
        Shape(vec![
            Dim::Const(m),
            Dim::Const(n),
            Dim::Const(k0),
            Dim::Const(k1),
        ]),
        &[a_restrided, b_restrided],
        body,
    );
    let reduce_k1 = b.reduce(ewise, 3, ReduceOp::Add);
    let _root_expr = b.reduce(reduce_k1, 2, ReduceOp::Max);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 50_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    let root = egraph.find(root);
    let has_fused_dispatch = egraph[root].iter().any(|n| {
        matches!(
            n,
            TensorIr::Dispatch(DispatchNode::Dispatch { num_inputs: 2, .. })
        )
    });
    assert!(
        has_fused_dispatch,
        "mixed-op nested reductions should lower to a composite-body 2-input Dispatch via recursive-to-dispatch"
    );
}

/// End-to-end test: build matmul, saturate, extract.
#[test]
fn test_matmul_end_to_end() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(32)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(32), Dim::Const(64)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, 64, 64, 32);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    // Run all phases
    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    // Extract best program
    let (cost, extracted) = greedy_extract(&egraph, root);
    assert!(cost < f64::INFINITY, "Should extract a valid program");
    assert!(
        !extracted.as_ref().is_empty(),
        "Extracted program should be non-empty"
    );
}

/// Test that register blocking produces a Dispatch with 4 (value, addr)
/// output pairs. Register blocking is encoded structurally in the
/// children list, not via reg_m/reg_n tags. Run phases 1 + 2 so the
/// phase-2 `register-block-N` rule fires on phase-1 output.
#[test]
fn test_register_blocking_dispatch() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, 64, 64, 64);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 50_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    let root = egraph.find(root);
    let reg_blocked = egraph[root].iter().find(|n| {
        let TensorIr::Dispatch(DispatchNode::Dispatch {
            num_inputs,
            children_list,
            ..
        }) = n
        else {
            return false;
        };
        let children = extract_list(&egraph, *children_list);
        let body_len = children.len().saturating_sub(*num_inputs as usize);
        body_len == 8 // 4 outputs × 2 (value, addr)
    });
    assert!(
        reg_blocked.is_some(),
        "Phase 1 should produce a 4-output register-blocked dispatch"
    );
}

// ═══════════════════════════════════════════════════════
// Tiled matmul tests
// ═══════════════════════════════════════════════════════

/// Test that Phase 1 fused lowering produces a tiled Dispatch for matmul.
#[test]
fn test_fused_matmul_lowering() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, 64, 64, 64);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 50_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    // The root e-class should contain a tiled Dispatch.
    // For 64×64 matmul: total_elements = 4096, workgroups = 4096/32 = 128.
    let root = egraph.find(root);
    let tiled_dispatch = egraph[root].iter().find(|n| {
        if let TensorIr::Dispatch(DispatchNode::Dispatch { workgroups, .. }) = n {
            workgroups.as_const().is_some_and(|w| w <= 128)
        } else {
            false
        }
    });
    assert!(
        tiled_dispatch.is_some(),
        "Fused matmul lowering should produce a tiled Dispatch in root class"
    );
}

/// Test that Phase 1+2 produces nested Thetas (K-loop tiling) for matmul.
#[test]
fn test_tiled_matmul_k_tiling() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, 64, 64, 64);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    // Count total Theta nodes in the e-graph
    let mut theta_count = 0;
    for class in egraph.classes() {
        for node in class.iter() {
            if matches!(node, TensorIr::Simd(SimdNode::Theta { .. })) {
                theta_count += 1;
            }
        }
    }

    // Phase 2 should have created tiled (nested) Thetas.
    // At minimum we expect the original Theta plus tiled versions.
    assert!(
        theta_count >= 2,
        "K-loop tiling should produce nested Thetas, got {theta_count}"
    );
}

/// Test that the cost model prefers tiled dispatch over naive dispatch.
#[test]
fn test_cost_model_prefers_tiled() {
    use crate::extractor::SyntheticCostModel;

    let model = SyntheticCostModel::default();
    let device = DeviceProfile::default();
    // Cost model reads `egraph[children_list].data.dtype_bytes` for the
    // register-pressure term; build an egraph with a stand-in eclass so the
    // index is valid.
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let placeholder = egraph.add(TensorIr::Nil);

    // Tiled dispatch: 16 workgroups
    let tiled = TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(16),
        num_inputs: 2,
        children_list: placeholder,
    });

    // Naive dispatch: 8192 workgroups
    let naive = TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(8192),
        num_inputs: 2,
        children_list: placeholder,
    });

    // Seq of 2 dispatches
    let seq = TensorIr::Dispatch(DispatchNode::Seq(placeholder));
    let pipeline = TensorIr::Dispatch(DispatchNode::Pipeline(placeholder));

    let tiled_cost = model.node_cost(&tiled, &egraph, &device);
    let naive_cost = model.node_cost(&naive, &egraph, &device);
    let seq_cost = model.node_cost(&seq, &egraph, &device);
    let pipeline_cost = model.node_cost(&pipeline, &egraph, &device);

    assert!(
        tiled_cost < naive_cost,
        "Tiled (16 wg, cost={tiled_cost}) should be cheaper than naive (8192 wg, cost={naive_cost})"
    );
    assert!(
        seq_cost > tiled_cost,
        "Seq (cost={seq_cost}) should be more expensive than single dispatch (cost={tiled_cost})"
    );
    assert!(
        pipeline_cost < seq_cost,
        "Pipeline (cost={pipeline_cost}) should be cheaper than Seq (cost={seq_cost})"
    );
}

#[test]
fn test_phase6_seq_lowers_to_pipeline() {
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let placeholder = egraph.add(TensorIr::Nil);
    let d0 = egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(8),
        num_inputs: 0,
        children_list: placeholder,
    }));
    let d1 = egraph.add(TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(8),
        num_inputs: 0,
        children_list: placeholder,
    }));
    let list = add_list(&mut egraph, &[d0, d1]);
    let seq = egraph.add(TensorIr::Dispatch(DispatchNode::Seq(list)));

    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::LateDispatch], &config);

    let seq_class = egraph.find(seq);
    let has_pipeline = egraph[seq_class]
        .iter()
        .any(|n| matches!(n, TensorIr::Dispatch(DispatchNode::Pipeline(_))));
    assert!(
        has_pipeline,
        "Phase 6 should introduce a Pipeline node for a Seq of dispatches"
    );
}

/// Cost model should prefer a register-blocked Dispatch (fewer
/// workgroups × more outputs per lane) over the scalar form covering
/// the same output footprint. Register blocking is encoded purely
/// structurally as additional `(value, addr)` pairs in the children
/// list; the cost model reads `num_outputs = (children.len() -
/// num_inputs) / 2`.
#[test]
fn test_cost_model_prefers_register_blocking_when_it_reduces_workgroups() {
    use crate::extractor::SyntheticCostModel;

    let model = SyntheticCostModel::default();
    let device = DeviceProfile::default();
    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let dummy_val = egraph.add(TensorIr::Const(ScalarValue::F32(
        ordered_float::OrderedFloat(0.0),
    )));
    let dummy_addr = egraph.add(TensorIr::Const(ScalarValue::U32(0)));

    // Scalar dispatch: 32_768 workgroups × 1 output pair.
    let scalar_children = add_list(&mut egraph, &[dummy_val, dummy_val, dummy_val, dummy_addr]);
    let scalar = TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(32_768),
        num_inputs: 2,
        children_list: scalar_children,
    });

    // Register-blocked dispatch: 8_192 workgroups × 4 output pairs,
    // covering the same total footprint (32_768 outputs).
    let blocked_children = add_list(
        &mut egraph,
        &[
            dummy_val, dummy_val, dummy_val, dummy_addr, dummy_val, dummy_addr, dummy_val,
            dummy_addr, dummy_val, dummy_addr,
        ],
    );
    let blocked = TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups: Dim::Const(8_192),
        num_inputs: 2,
        children_list: blocked_children,
    });

    let scalar_cost = model.node_cost(&scalar, &egraph, &device);
    let blocked_cost = model.node_cost(&blocked, &egraph, &device);

    assert!(
        blocked_cost < scalar_cost,
        "register-blocked dispatch (cost={blocked_cost}) should be cheaper than scalar (cost={scalar_cost}) when it reduces workgroup launches for the same footprint"
    );
}

/// End-to-end test: full pipeline extracts a fused dispatch (no Seq) for matmul.
#[test]
fn test_tiled_matmul_end_to_end() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Const(64), Dim::Const(64)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, 64, 64, 64);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    // Run all phases
    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    // Extract best program
    let beam_cfg = BeamConfig {
        beam_width: 8,
        ..Default::default()
    };
    let (cost, extracted) = beam_extract(&egraph, root, &beam_cfg);
    assert!(cost < f64::INFINITY, "Should extract a valid program");

    // Check the extracted program contains a Dispatch but NOT a Seq at the root
    let root_node = extracted.as_ref().last().unwrap();
    assert!(
        !matches!(root_node, TensorIr::Dispatch(DispatchNode::Seq(_))),
        "Extracted matmul should be a single fused Dispatch, not a Seq. Got: {root_node:?}"
    );
}

/// Test as_3d_lit helper.
#[test]
fn test_shape_as_3d_lit() {
    let s = Shape(vec![Dim::Const(64), Dim::Const(32), Dim::Const(16)]);
    assert_eq!(s.as_3d_lit(), Some((64, 32, 16)));

    let s2 = Shape(vec![Dim::Const(64), Dim::Const(32)]);
    assert_eq!(s2.as_3d_lit(), None);

    let s3 = Shape(vec![Dim::Const(64), Dim::Symbol(0), Dim::Const(16)]);
    assert_eq!(s3.as_3d_lit(), None);
}

#[test]
fn test_symbolic_shape_numel_and_row_major_strides() {
    let shape = Shape(vec![Dim::Symbol(0), Dim::Symbol(1), Dim::Symbol(2)]);
    let strides = Strides::row_major_for_shape(&shape);
    let params = ShapeParams::from([2, 3, 4]);

    assert_eq!(shape.numel().as_const(), None);
    assert_eq!(shape.numel().eval_u32(&params), Some(24));
    assert_eq!(
        strides
            .0
            .iter()
            .map(|stride| stride.eval_u32(&params).unwrap())
            .collect::<Vec<_>>(),
        vec![12, 4, 1]
    );
}

#[test]
fn test_phase1_symbolic_elementwise_uses_algebraic_workgroups() {
    let shape = Shape(vec![Dim::Symbol(0), Dim::Symbol(1)]);
    let mut b = IrBuilder::new();
    let input = b.input(0, shape.clone(), DType::F32);
    let body = b.scalar_arg(0);
    let _ewise = b.elementwise(shape, &[input], body);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);
    let config = RunnerConfig {
        iter_limit: 5,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);
    let params = ShapeParams::from([7, 9]);

    let has_symbolic_dispatch = egraph[egraph.find(root)].iter().any(|node| {
        matches!(
            node,
            TensorIr::Dispatch(DispatchNode::Dispatch { workgroups, .. })
                if workgroups.as_const().is_none() && workgroups.eval_u32(&params) == Some(2)
        )
    });
    assert!(
        has_symbolic_dispatch,
        "symbolic [M,N] elementwise lowering should keep workgroups algebraic"
    );
}

#[test]
fn test_symbolic_elementwise_codegen_reads_shape_params() {
    let shape = Shape(vec![Dim::Symbol(0), Dim::Symbol(1)]);
    let mut b = crate::TensorExprBuilder::new();
    let input = b.input(0, shape.clone(), DType::F32);
    let body = b.scalar_arg(0);
    let root = b.elementwise(shape, &[input], body);
    let expr = b.build(root).expect("symbolic tensor expr should validate");

    let mut config = crate::StageConfig::default();
    config.runner.iter_limit = 6;
    config.runner.node_limit = 20_000;
    config.candidate_limit = Some(1);

    let kernel = crate::lower_tensor_expr(&expr, &config)
        .expect("symbolic elementwise expression should lower");
    let wgsl = crate::lower_to_wgsl(&kernel).expect("symbolic program should lower to WGSL");
    let params = ShapeParams::from([8, 9]);

    assert!(wgsl.contains("shape_params"));
    assert!(
        !wgsl.contains("Symbol"),
        "WGSL should load runtime dimensions, not print symbolic placeholders"
    );
    assert!(
        kernel.extracted().as_ref().iter().any(|node| matches!(
            node,
            TensorIr::Effect(EffectNode::Dispatch { workgroups, .. })
                if workgroups.as_const().is_none() && workgroups.eval_u32(&params) == Some(3)
        )),
        "compiled dispatch should retain algebraic ceildiv(M*N, simd_width) workgroups"
    );
}

// ═══════════════════════════════════════════════════════
// Theta-merge (post-lowering reduce fusion) tests
// ═══════════════════════════════════════════════════════

/// Two independent reductions (Max, Add) over the same input and axis
/// should, after phase-1 lowering + phase-2 theta-merge, produce a
/// single Dispatch with `reg_n=2` carrying a `Theta { RunningReduction,
/// Pack(init_max, init_add), count, Pack(update_max, update_add) }`.
#[test]
fn test_phase2_theta_merge_independent_reduces() {
    // reduce_dim=32 hits the simple-reduce path in reduce_lowering
    // (cooperative kicks in only when reduce_dim > simd_width).
    let shape = Shape(vec![Dim::Const(128), Dim::Const(32)]);
    let mut b = IrBuilder::new();
    let a = b.input(0, shape.clone(), DType::F32);
    let _rmax = b.reduce(a, 1, ReduceOp::Max);
    let mut b2 = IrBuilder::new();
    let a2 = b2.input(0, shape, DType::F32);
    let _radd = b2.reduce(a2, 1, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _ = egraph.add_expr(&b.expr);
    let _ = egraph.add_expr(&b2.expr);

    let config = RunnerConfig {
        iter_limit: 6,
        node_limit: 30_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    // theta-merge now encodes two outputs structurally via two
    // `(value, addr)` pairs in the children_list.
    let merged = egraph.classes().find(|class| {
        class.iter().any(|node| {
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
            body_len == 4 // 2 outputs × 2 (value, addr)
        })
    });
    assert!(
        merged.is_some(),
        "theta-merge should emit a 2-output Dispatch when two compatible reductions coexist"
    );

    // The merged e-class (or the canonical eclass it unioned with) must
    // now contain a `Theta { RunningReduction }` somewhere reachable
    // via the Dispatch's children.
    let has_running_reduction = egraph.classes().any(|class| {
        class
            .iter()
            .any(|node| matches!(node, TensorIr::Simd(SimdNode::Theta { .. })))
    });
    assert!(
        has_running_reduction,
        "theta-merge should introduce a RunningReduction Theta into the e-graph"
    );
}

/// `exp(a - b)` must acquire an equivalent `Mul(Exp(a), Exp(Neg(b)))`
/// form in the e-graph after phase-1 saturation. This is the scalar
/// equivalence that (with a future factor-out-constant-bcast partner)
/// decouples `Σ exp(x - bcast(max))` into independent reductions.
#[test]
fn test_phase1_exp_sub_split() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Const(64)]), DType::F32);
    let bcast = b.input(1, Shape(vec![Dim::Const(64)]), DType::F32);
    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let sub = b.bin_op(BinaryOp::Sub, arg0, arg1);
    let body = b.un_op(UnaryOp::Exp, sub);
    let _ewise = b.elementwise(Shape(vec![Dim::Const(64)]), &[a, bcast], body);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _ = egraph.add_expr(&b.expr);
    let config = RunnerConfig {
        iter_limit: 4,
        node_limit: 10_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    // Any e-class that contained `exp(sub(...))` should now also contain
    // a `Mul(Exp(_), Exp(Neg(_)))` — two UnOp(Exp) nodes and a Mul
    // linking them.
    let has_product_form = egraph.classes().any(|class| {
        class.iter().any(|node| {
            let TensorIr::BinOp(BinaryOp::Mul, [lhs, rhs]) = node else {
                return false;
            };
            let lhs_is_exp = egraph[*lhs]
                .iter()
                .any(|n| matches!(n, TensorIr::UnOp(UnaryOp::Exp, _)));
            let rhs_is_exp_neg = egraph[*rhs].iter().any(|n| {
                let TensorIr::UnOp(UnaryOp::Exp, inner) = n else {
                    return false;
                };
                egraph[*inner]
                    .iter()
                    .any(|m| matches!(m, TensorIr::UnOp(UnaryOp::Neg, _)))
            });
            lhs_is_exp && rhs_is_exp_neg
        })
    });
    assert!(
        has_product_form,
        "exp-sub-split should introduce `Mul(Exp(a), Exp(Neg(b)))` into the e-graph"
    );
}

/// `Reduce(a, Add, Elementwise([x, bcast_c], Mul(P0, P1)))` should
/// acquire an equivalent `Elementwise([Reduce(a, Add, x), c], Mul(P0,
/// P1))` — `c` factored out of the sum since it's axis-invariant.
#[test]
fn test_phase1_factor_reduce_mul_bcast() {
    let shape = Shape(vec![Dim::Const(32), Dim::Const(64)]);
    let mut b = IrBuilder::new();
    let x = b.input(0, shape.clone(), DType::F32);
    let c = b.input(1, Shape(vec![Dim::Const(32)]), DType::F32);
    // bcast c over axis 1: stride [1, 0] so axis-1 accesses all the
    // same scalar.
    let c_bcast = b.restride(
        c,
        shape.clone(),
        Strides(vec![Dim::Const(1), Dim::Const(0)]),
    );
    let p0 = b.scalar_arg(0);
    let p1 = b.scalar_arg(1);
    let body = b.bin_op(BinaryOp::Mul, p0, p1);
    let ewise = b.elementwise(shape, &[x, c_bcast], body);
    let _reduced = b.reduce(ewise, 1, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _ = egraph.add_expr(&b.expr);
    let config = RunnerConfig {
        iter_limit: 4,
        node_limit: 20_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    // After saturation, some e-class should contain an Elementwise
    // whose first input (ewise_children[0]) e-class contains an inner
    // `Reduce(axis=1, Add, _)` — that's the factored form.
    let found_factored = egraph.classes().any(|class| {
        class.iter().any(|node| {
            let TensorIr::HighLevel(HighLevelNode::Elementwise {
                num_inputs,
                children_list,
                ..
            }) = node
            else {
                return false;
            };
            if *num_inputs != 2 {
                return false;
            }
            let children = extract_list(&egraph, *children_list);
            egraph[children[0]].iter().any(|n| {
                matches!(
                    n,
                    TensorIr::HighLevel(HighLevelNode::Reduce {
                        axis: 1,
                        op: ReduceOp::Add,
                        ..
                    })
                )
            })
        })
    });
    assert!(
        found_factored,
        "factor-reduce-mul-bcast should introduce an Elementwise wrapping an inner Reduce"
    );
}

/// End-to-end: the softmax-denominator pattern as built by the
/// high-level softmax IR (shift+exp+sum, two elementwises fed into a
/// reduce) should, after phase-1 + phase-2 saturation, produce a
/// `Theta { RunningReduction }` in the e-graph. The chain of rewrites
/// is: `ewise-fuse` (merge Sub/Exp ewises) → `exp-sub-split` (Exp(Sub)
/// → Mul(Exp, Exp(Neg))) → `factor-reduce-mul-bcast` (factor the
/// exp-of-max out of the sum) → two independent reduces (Max over x,
/// Add over exp(x)) → `theta-merge-reduction`.
#[test]
fn test_e2e_softmax_denominator_decouples_to_running_reduction() {
    let shape = Shape(vec![Dim::Const(128), Dim::Const(32)]);
    let axis = 1;
    let mut b = IrBuilder::new();
    let x = b.input(0, shape.clone(), DType::F32);
    // max = Reduce(a, Max, x)
    let max = b.reduce(x, axis, ReduceOp::Max);
    // bcast max over axis
    let mut bcast_strides = vec![Dim::Const(1); shape.rank()];
    bcast_strides[axis as usize] = Dim::Const(0);
    let max_bcast = b.restride(max, shape.clone(), Strides(bcast_strides));
    // shifted = Elementwise([x, max_bcast], Sub(P0, P1))
    let p0 = b.scalar_arg(0);
    let p1 = b.scalar_arg(1);
    let sub = b.bin_op(BinaryOp::Sub, p0, p1);
    let shifted = b.elementwise(shape.clone(), &[x, max_bcast], sub);
    // exp_val = Elementwise([shifted], Exp(P0))
    let p0b = b.scalar_arg(0);
    let exp_body = b.un_op(UnaryOp::Exp, p0b);
    let exp_val = b.elementwise(shape, &[shifted], exp_body);
    // sum = Reduce(a, Add, exp_val) — the denominator.
    let _sum = b.reduce(exp_val, axis, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _ = egraph.add_expr(&b.expr);
    let config = RunnerConfig {
        iter_limit: 8,
        node_limit: 200_000,
        time_limit_secs: 15,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    let has_running_reduction = egraph.classes().any(|class| {
        class
            .iter()
            .any(|node| matches!(node, TensorIr::Simd(SimdNode::Theta { .. })))
    });
    assert!(
        has_running_reduction,
        "softmax-denominator chain should produce a RunningReduction Theta after phase-1+2"
    );
}

// ═══════════════════════════════════════════════════════
// Cooperative loading / K-tiled tests
// ═══════════════════════════════════════════════════════
