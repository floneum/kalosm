use std::sync::Arc;

use fusor_ir::device::{Caps, CoopKind, DeviceKind, Limits, SubgroupWidths};
use fusor_ir::dtype::Dtype;
use fusor_ir::egraph::{EGraph, Id, RuleFn};
use fusor_ir::ir::Op;
use fusor_ir::ir::logical::{BufferId, EinSpec, Label, LeafKind, Logical};
use fusor_ir::shape::Dim;

fn caps() -> Caps {
    Caps {
        kind: DeviceKind::Cpu,
        name: "candidate construction test".into(),
        limits: Limits {
            max_compute_invocations_per_workgroup: 1,
            ..Limits::default()
        },
        subgroups: None,
        f16: false,
        bf16: false,
        coop: Default::default(),
        atomic_f32: false,
        workgroup_alias: false,
        mixed_precision_coop_store: false,
        pipeline_cache: false,
        timestamp_query: false,
        simd_widths: Default::default(),
        threads: 1,
    }
}

/// A GPU with default limits and fixed 32-wide subgroups.
fn gpu_caps() -> Caps {
    let mut caps = caps();
    caps.kind = DeviceKind::Gpu;
    caps.limits = Limits::default();
    caps.subgroups = Some(SubgroupWidths { min: 32, max: 32 });
    caps
}

/// [`gpu_caps`] plus an 8x8x8 cooperative matrix over `operand`.
fn coop_caps(operand: Dtype) -> Caps {
    let mut caps = gpu_caps();
    caps.coop.push(CoopKind {
        operand,
        acc: Dtype::F32,
        m: 8,
        n: 8,
        k: 8,
    });
    caps
}

fn planner_graph() -> (Arc<fusor_tile::Planner>, EGraph) {
    let planner = Arc::new(fusor_tile::Planner::new());
    (
        planner.clone(),
        EGraph::new(fusor_ir::CoreSemantics::new(planner)),
    )
}

fn graph() -> EGraph {
    planner_graph().1
}

fn consts<const N: usize>(shape: [u64; N]) -> [Dim; N] {
    shape.map(Dim::Const)
}

/// An f32 buffer leaf.
fn leaf(graph: &mut EGraph, name: u32, shape: &[Dim]) -> Id {
    graph
        .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
            name: BufferId(name),
            dtype: Dtype::F32,
            shape: shape.iter().copied().collect(),
        })))
        .unwrap()
}

/// An f32 contraction of `a` and `b` over the `[a, b, out]` label lists.
fn contract(graph: &mut EGraph, a: Id, b: Id, [la, lb, lo]: [&[u8]; 3]) -> Id {
    let labels = |l: &[u8]| l.iter().copied().map(Label).collect();
    graph
        .add(Op::Logical(Logical::Contract {
            spec: EinSpec {
                a: labels(la),
                b: labels(lb),
                out: labels(lo),
            },
            acc: Dtype::F32,
            a,
            b,
            outs: 1,
        }))
        .unwrap()
}

/// `a[m, k] x b[k, n]`.
const MATMUL: [&[u8]; 3] = [&[0, 1], &[1, 2], &[0, 2]];

/// Run one rule on `id` the way the saturation driver does.
fn apply(graph: &mut EGraph, caps: &Caps, id: Id, rule: RuleFn) -> Option<Id> {
    let node = graph.node(id).clone();
    let facts = graph.facts_view(id, caps);
    rule(&mut graph.builder(caps), id, &node, &facts)
}

/// `rule` declines on `id` without minting anything: an inapplicable rule
/// must not leave an invalid alternative behind.
fn assert_declines(graph: &mut EGraph, caps: &Caps, id: Id, rule: RuleFn) {
    let before = graph.len();
    assert!(apply(graph, caps, id, rule).is_none());
    assert_eq!(graph.len(), before);
}

#[test]
fn shared_matmul_ancestors_are_charged_once_with_every_branch() {
    use fusor_ir::dtype::{QFmt, QLayout};
    use fusor_ir::egraph::{Saturate, SaturationBudget};
    use fusor_ir::extract::Extractor;
    use fusor_ir::ir::launch::Launch;
    use fusor_ir::scalar::{BinOp, ScalarExpr};

    let caps = gpu_caps();
    let (planner, mut graph) = planner_graph();
    let mut x = leaf(&mut graph, 0, &[Dim::ONE, Dim::Const(4096)]);
    let mut matmuls = Vec::new();
    let mut layers = Vec::new();
    for layer in 0..32 {
        let mut branches = Vec::new();
        for branch in 0..4 {
            let weight = graph
                .add(Op::Logical(Logical::Leaf(LeafKind::Quantized {
                    name: BufferId(1 + layer * 4 + branch),
                    fmt: QFmt::Q4K,
                    layout: QLayout::Native,
                    shape: [Dim::Const(4096); 2].into_iter().collect(),
                })))
                .unwrap();
            let matmul = contract(&mut graph, x, weight, [&[0, 1], &[2, 1], &[0, 2]]);
            branches.push(matmul);
            matmuls.push(matmul);
        }
        let expr = (1..4).fold(ScalarExpr::arg(0, Dtype::F32), |a, i| {
            ScalarExpr::bin(BinOp::Add, a, ScalarExpr::arg(i, Dtype::F32))
        });
        x = graph
            .add(Op::Logical(Logical::Map {
                expr,
                ins: branches.into_iter().collect(),
                outs: 1,
            }))
            .unwrap();
        layers.push(x);
    }
    graph.add_root(x);
    let rules: Vec<_> = fusor_ir::CORE_RULES
        .iter()
        .chain(fusor_tile::SCHED_RULES)
        .copied()
        .collect();
    fusor_ir::Driver::new()
        .saturate(
            &mut graph,
            &caps,
            &rules,
            SaturationBudget {
                max_applications: 0,
                ..Default::default()
            },
        )
        .unwrap();
    let cost = fusor_cost::Roofline::new(fusor_cost::facts::seed_facts(&caps));
    let extractor = fusor_cost::LocalSearch::new(planner.clone(), caps);
    let bounds = extractor.lower_bound(&graph, &cost);
    let selected = extractor.seed(&graph, &[x], &bounds, &cost).unwrap();
    for matmul in matmuls {
        let member = selected.selected(graph.class_of(matmul)).unwrap();
        assert!(matches!(
            graph.node(member).op,
            Op::Launch(Launch::Contract { .. })
        ));
    }
    let realize = |root| {
        fusor_cost::realize::realize(&graph, &[root], &selected, &cost, planner.as_ref()).unwrap()
    };
    let first = realize(layers[0]);
    let all = realize(x);
    assert_eq!(first.components.len(), 5);
    assert_eq!(all.components.len(), 32 * 5);
    let first_cost = fusor_cost::realize::exact_cost(&first, &selected, &cost);
    let all_cost = fusor_cost::realize::exact_cost(&all, &selected, &cost);
    assert_eq!(all_cost.0, first_cost.0 * 32);
}

#[test]
fn repeated_matmul_costs_do_not_degrade_to_free_arithmetic() {
    use fusor_ir::cost::CostModel;
    use fusor_ir::extract::Extractor;
    use fusor_ir::ir::launch::{
        AccessPlan, ContractSide, Family, IndexSpace, Launch, Operand, ScheduleDomain,
    };
    use fusor_ir::scalar::ScalarExpr;
    use fusor_ir::shape::Layout;

    let caps = coop_caps(Dtype::F32);
    let (planner, mut graph) = planner_graph();
    let domain = fusor_tile::domains::coop::legal(
        Dim::Const(128),
        Dim::Const(128),
        Dim::Const(4096),
        Dtype::F32,
        Dtype::F32,
        &caps,
    );
    let side = |graph: &mut EGraph, name, shape: [Dim; 2]| {
        ContractSide::one(
            ScalarExpr::arg(0, Dtype::F32),
            Operand {
                src: leaf(graph, name, &shape),
                layout: Layout::contiguous(&shape),
                access: AccessPlan::Alias,
            },
        )
    };
    let mut matmuls = Vec::new();
    for i in 0..4096 {
        let a = side(&mut graph, 2 * i, consts([128, 4096]));
        let b = side(&mut graph, 2 * i + 1, consts([4096, 128]));
        matmuls.push(
            graph
                .builder(&caps)
                .add_launch(Launch::Contract {
                    output: IndexSpace::new(consts([128, 128])),
                    m: Dim::Const(128),
                    n: Dim::Const(128),
                    k: Dim::Const(4096),
                    batch: Dim::ONE,
                    family: Family::Coop,
                    post: ScalarExpr::arg(0, Dtype::F32),
                    acc: Dtype::F32,
                    a,
                    b,
                    sched: ScheduleDomain::Coop(domain.clone().into()),
                })
                .unwrap(),
        );
    }
    let cost = fusor_cost::Roofline::new(fusor_cost::facts::seed_facts(&caps));
    let bounds = fusor_cost::LocalSearch::new(planner, caps).lower_bound(&graph, &cost);
    assert!(bounds[matmuls[0].index()].0 > cost.facts().launch_ps);
    for id in matmuls.iter().skip(1) {
        assert_eq!(bounds[id.index()], bounds[matmuls[0].index()]);
    }
}

#[test]
fn symbolic_contractions_expose_their_reduction_and_address_maps() {
    use fusor_ir::ir::launch::{AccessPlan, Launch};
    let caps = caps();
    let mut graph = graph();
    let length = Dim::Sym(graph.fresh_sym());
    for varying_k in [false, true] {
        let (n, k) = if varying_k {
            (Dim::Const(7), length)
        } else {
            (length, Dim::Const(19))
        };
        let a = leaf(&mut graph, 0, &[Dim::Const(2), Dim::Const(3), k]);
        let b = leaf(&mut graph, 1, &[Dim::Const(2), n, k]);
        let id = contract(&mut graph, a, b, [&[0, 1, 3], &[0, 2, 3], &[0, 1, 2]]);
        apply(
            &mut graph,
            &caps,
            id,
            fusor_ir::rules::lower_floor::lower_contract_generic,
        )
        .expect("a dynamic contraction must remain visible as a reduction");
        let folded = graph
            .members(graph.class_of(id))
            .into_iter()
            .find(|member| matches!(graph.node(*member).op, Op::Launch(Launch::Fold { .. })))
            .unwrap();
        let Op::Launch(Launch::Fold { ops, .. }) = &graph.node(folded).op else {
            panic!("the contraction floor must expose a Fold");
        };
        for len in [1, 17, 65] {
            let (n, k) = if varying_k { (7, len) } else { (len, 19) };
            for batch in 0..2 {
                for row in 0..3 {
                    for col in 0..n {
                        for reduced in 0..k {
                            for (side, expected) in [
                                (0, (batch * 3 + row) * k + reduced),
                                (1, (batch * n + col) * k + reduced),
                            ] {
                                let op = &ops[side];
                                assert!(matches!(op.access, AccessPlan::Alias));
                                let address: u64 = [batch, row, col, reduced]
                                    .into_iter()
                                    .zip(op.layout.strides())
                                    .map(|(i, stride)| {
                                        i * stride.evaluate(&mut |_| Some(len)).unwrap()
                                    })
                                    .sum();
                                assert_eq!(address, expected);
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn contraction_rule_does_not_construct_an_empty_schedule() {
    let mut graph = graph();
    let a = leaf(&mut graph, 0, &consts([9, 1]));
    let b = leaf(&mut graph, 1, &consts([1, 1]));
    let id = contract(&mut graph, a, b, MATMUL);
    assert_declines(
        &mut graph,
        &caps(),
        id,
        fusor_tile::rules::contract::lower_generic,
    );
}

#[test]
fn cooperative_domains_contain_only_supported_scratch_combinations() {
    use fusor_ir::ir::kernel::{ArenaPlanner, ScalarElement};
    use fusor_ir::ir::launch::coop_tiles;
    let planner = fusor_tile::Planner::new();
    for limit in [2048, 16384, 32768] {
        for dtype in [Dtype::F32, Dtype::F16] {
            let mut caps = coop_caps(dtype);
            caps.limits.max_compute_workgroup_storage_size = limit;
            caps.f16 = true;
            let domain = fusor_tile::domains::coop::legal(
                Dim::Const(32),
                Dim::Const(32),
                Dim::Const(256),
                dtype,
                Dtype::F32,
                &caps,
            );
            assert!(!domain.is_empty(), "{dtype:?}, {limit}");
            for point in &domain.schedules {
                assert!(point.geom.legal(32, 256));
                assert!((1..=2).contains(&point.staging));
                let element = if dtype == Dtype::F16 {
                    ScalarElement::F16
                } else {
                    ScalarElement::F32
                };
                let tiles = coop_tiles(point.geom, element, point.staging);
                // Independent extent calculation: two stacked operand tiles
                // plus enough room for a scalar accumulator copy.
                let g = point.geom;
                let depth = u64::from(point.staging);
                let expected =
                    depth * u64::from(g.bm * g.bk + g.bk * (g.bn / g.n_passes)) * dtype.byte_size()
                        + u64::from(g.bm * (g.bn / g.n_passes)) * 4;
                let actual = planner.workgroup_bytes(&tiles, &caps).unwrap();
                assert!(u64::from(actual) >= expected);
                assert!(actual <= limit, "{point:?}: {actual} > {limit}");
            }
        }
    }
}

#[test]
fn promotion_constructs_a_domain_for_the_promoted_carrier() {
    use fusor_ir::carrier::Carrier;
    use fusor_ir::ir::launch::{
        AccessPlan, FoldDomain, FoldStrat, IndexSpace, Launch, Operand, ScheduleDomain,
    };
    use fusor_ir::scalar::{BinOp, ScalarExpr};
    use fusor_ir::shape::Layout;
    let mut caps = caps();
    caps.limits = Limits::default();
    let mut graph = graph();
    let dims = consts([64, 32]);
    let src = leaf(&mut graph, 0, &dims);
    let id = graph
        .add(Op::Launch(Launch::Fold {
            space: IndexSpace::new(dims),
            axis: 1,
            vec_axes: Default::default(),
            carrier: Carrier::binop(
                BinOp::Add,
                Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
                Dtype::F32,
            ),
            acc: Dtype::F32,
            post: [ScalarExpr::arg(0, Dtype::F32)].into_iter().collect(),
            ops: vec![Operand {
                src,
                layout: Layout::contiguous(&dims),
                access: AccessPlan::Alias,
            }],
            sched: ScheduleDomain::Fold(
                FoldDomain {
                    strategies: [
                        FoldStrat::WgTree { lane_group: 256 },
                        FoldStrat::WgTree { lane_group: 1 },
                    ]
                    .into_iter()
                    .collect(),
                }
                .into(),
            ),
        }))
        .unwrap();
    apply(&mut graph, &caps, id, fusor_ir::rules::promote::promote)
        .expect("row-per-lane promotion is supported");
    let promoted = graph
        .members(graph.class_of(id))
        .into_iter()
        .find_map(|id| match &graph.node(id).op {
            Op::Launch(Launch::Fold {
                vec_axes,
                carrier,
                sched,
                ..
            }) if !vec_axes.is_empty() => Some((carrier, sched)),
            _ => None,
        })
        .unwrap();
    assert_eq!(promoted.0.lanes(), Some(64));
    let ScheduleDomain::Fold(domain) = promoted.1 else {
        panic!("expected fold domain")
    };
    assert_eq!(
        domain.strategies.as_slice(),
        &[FoldStrat::WgTree { lane_group: 1 }]
    );
}

#[test]
fn contraction_alternatives_preserve_the_logical_output_shape() {
    use fusor_ir::ir::launch::Launch;
    use fusor_ir::rules::split_k::split_k;
    use fusor_ir::scalar::{BinOp, ScalarExpr};
    use fusor_tile::rules::contract::{lower_coop, lower_generic, lower_sgemm, lower_sgemv};
    let caps = coop_caps(Dtype::F32);
    for (batch, m, n) in [
        (&[2, 3][..], &[4][..], &[6][..]),
        (&[1, 1][..], &[4][..], &[6][..]),
        (&[2, 1][..], &[3, 4][..], &[5, 6][..]),
    ] {
        let mut graph = graph();
        let shape =
            |parts: &[&[u64]]| -> Vec<Dim> { parts.concat().into_iter().map(Dim::Const).collect() };
        let a = leaf(&mut graph, 0, &shape(&[batch, m, &[256]]));
        let b = leaf(&mut graph, 1, &shape(&[batch, &[256], n]));
        let batch_end = batch.len() as u8;
        let m_end = batch_end + m.len() as u8;
        let n_end = m_end + n.len() as u8;
        let a_labels: Vec<u8> = (0..m_end).chain([n_end]).collect();
        let b_labels: Vec<u8> = (0..batch_end).chain([n_end]).chain(m_end..n_end).collect();
        let out_labels: Vec<u8> = (0..n_end).collect();
        let id = contract(&mut graph, a, b, [&a_labels, &b_labels, &out_labels]);
        let shape = graph.facts(id).shape.clone();
        for rule in [lower_generic, lower_sgemm, lower_sgemv, lower_coop] {
            let variant = apply(&mut graph, &caps, id, rule).unwrap();
            assert_eq!(graph.facts(variant).shape, shape);
            let variant_node = graph.node(variant).clone();
            if matches!(variant_node.op, Op::Launch(Launch::Contract { .. })) {
                let split = apply(&mut graph, &caps, variant, split_k).unwrap();
                assert_eq!(graph.facts(split).shape, shape);
                for left in [true, false] {
                    let mut indexed = variant_node.op.clone();
                    let Op::Launch(Launch::Contract { a, b, .. }) = &mut indexed else {
                        unreachable!()
                    };
                    let side = if left { a } else { b };
                    side.pre = ScalarExpr::bin(
                        BinOp::Add,
                        side.pre.clone(),
                        ScalarExpr::cast(
                            Dtype::F32,
                            ScalarExpr::index_of((side.primary().layout.rank() - 1) as u32),
                        ),
                    );
                    let indexed = graph.add(indexed).unwrap();
                    assert_declines(&mut graph, &caps, indexed, split_k);
                }
            }
        }
        assert_graph_invariants(&graph, &caps);
    }
}

#[test]
fn symbolic_contraction_groups_keep_their_bound_extents() {
    use fusor_ir::ir::launch::Launch;
    use fusor_ir::shape::SymId;
    let s = Dim::Sym(SymId(0));
    let t = Dim::Sym(SymId(1));
    let mut caps = gpu_caps();
    caps.subgroups = None;
    for (rows, expected) in [
        (vec![s], 5),
        (vec![s, Dim::ONE], 5),
        (vec![s, Dim::Const(2)], 10),
        (vec![Dim::Const(2), s], 10),
        (vec![s, t], 15),
    ] {
        let mut graph = graph();
        let a_shape: Vec<Dim> = rows.iter().copied().chain([Dim::Const(8)]).collect();
        let a = leaf(&mut graph, 0, &a_shape);
        let b = leaf(&mut graph, 1, &consts([8, 4]));
        let k = rows.len() as u8;
        let a_labels: Vec<u8> = (0..=k).collect();
        let out_labels: Vec<u8> = (0..k).chain([k + 1]).collect();
        let id = contract(&mut graph, a, b, [&a_labels, &[k, k + 1], &out_labels]);
        let variant = apply(
            &mut graph,
            &caps,
            id,
            fusor_tile::rules::contract::lower_sgemm,
        )
        .unwrap();
        let Op::Launch(Launch::Contract { m, .. }) = &graph.node(variant).op else {
            unreachable!()
        };
        assert_eq!(
            m.evaluate(&mut |sym| [5, 3].get(sym.0 as usize).copied()),
            Some(expected)
        );
        assert_eq!(graph.facts(variant).shape, graph.facts(id).shape);
        assert_declines(
            &mut graph,
            &caps,
            variant,
            fusor_ir::rules::specialize::specialize_dim,
        );
    }
}

#[test]
fn grouped_padding_symbols_reach_the_uniform_plan() {
    use fusor_cost::{Roofline, extract::LocalSearch, realize::NodeCache};
    use fusor_ir::extract::Extraction;
    use fusor_ir::ir::launch::Launch;
    use fusor_ir::shape::SymId;
    let caps = coop_caps(Dtype::F32);
    let (planner, mut graph) = planner_graph();
    let batch = Dim::Sym(SymId(0));
    let a = leaf(
        &mut graph,
        0,
        &[Dim::Const(2), batch, Dim::Const(3), Dim::Const(8)],
    );
    let b = leaf(
        &mut graph,
        1,
        &[Dim::Const(2), batch, Dim::Const(8), Dim::Const(5)],
    );
    let id = contract(
        &mut graph,
        a,
        b,
        [&[0, 1, 2, 3], &[0, 1, 3, 4], &[0, 1, 2, 4]],
    );
    let variant = apply(
        &mut graph,
        &caps,
        id,
        fusor_tile::rules::contract::lower_coop,
    )
    .unwrap();
    let Op::Launch(Launch::Contract { sched, .. }) = &graph.node(variant).op else {
        unreachable!()
    };
    let mut extraction = Extraction::default();
    for node in [a, b, variant] {
        extraction.sigma.insert(graph.class_of(node), node);
    }
    extraction.m.grow(graph.len());
    extraction.m.insert(variant.index());
    extraction.theta.insert(variant, sched.point(0).unwrap());
    let facts = fusor_cost::facts::seed_facts(&caps);
    let plan = LocalSearch::new(planner, caps)
        .replan(
            &graph,
            &[variant],
            &mut extraction,
            &Roofline::new(facts.clone()),
            &mut NodeCache::new(graph.len()),
        )
        .unwrap();
    let buffer = plan
        .buffers
        .iter()
        .find(|buffer| buffer.value == variant)
        .unwrap();
    let Dim::Sym(stride) = buffer.layout.strides()[0] else {
        panic!("batch stride must depend on its symbolic inner extent")
    };
    assert!(
        plan.symbols.contains(&stride),
        "the padded batch stride needs a uniform slot"
    );
    let dense = fusor_ir::shape::Layout::contiguous(&graph.facts(variant).shape);
    let Dim::Sym(logical_stride) = dense.strides()[0] else {
        unreachable!()
    };
    assert!(
        plan.symbols.contains(&logical_stride),
        "repadding needs the logical batch stride too"
    );
    for extent in [3, 5] {
        let bind = &mut |symbol| (symbol == SymId(0)).then_some(extent);
        let inner_stride = buffer.layout.strides()[1].evaluate(bind).unwrap();
        assert_eq!(Dim::Sym(stride).evaluate(bind), Some(extent * inner_stride));
        assert_eq!(
            buffer.elements.evaluate(bind),
            Some(2 * extent * inner_stride)
        );
    }
    let mut buffers = plan.buffers.clone();
    buffers
        .iter_mut()
        .find(|buffer| buffer.value == variant)
        .unwrap()
        .layout = dense;
    assert_ne!(
        plan.hash,
        fusor_cost::plan::plan_hash(
            &graph,
            &plan.extraction,
            &plan.launches,
            &buffers,
            &plan.symbols,
            &facts,
        )
    );
}

#[test]
fn fold_split_declines_unbound_carrier_slots_and_changed_coordinates_before_minting() {
    use fusor_ir::carrier::{Carrier, oracle};
    use fusor_ir::scalar::{BinOp, ScalarExpr};

    let sum = Carrier::binop(BinOp::Add, fusor_ir::dtype::Splat::F32(0.0), Dtype::F32);
    for (carrier, should_split) in [
        (oracle::welford(Dtype::F32), false),
        (
            sum.clone()
                .with_lift([ScalarExpr::cast(Dtype::F32, ScalarExpr::index_of(1))]),
            false,
        ),
        (sum, true),
    ] {
        let caps = caps();
        let mut graph = graph();
        let input = leaf(&mut graph, 0, &consts([4, 996]));
        let fold = |carrier| {
            Op::Logical(Logical::Fold {
                carrier,
                axis: 1,
                acc: Dtype::F32,
                ins: smallvec::smallvec![input],
            })
        };
        if carrier.width() > 1 {
            let before = graph.len();
            assert!(graph.add(fold(carrier.as_merge())).is_err());
            assert_eq!(
                graph.len(),
                before,
                "unbound lift operands must never enter the graph"
            );
        }
        let id = graph.add(fold(carrier)).unwrap();
        let before = graph.len();
        let variant = apply(&mut graph, &caps, id, fusor_ir::rules::algebra::strip);
        assert_eq!(variant.is_some(), should_split);
        if !should_split {
            assert_eq!(
                graph.len(),
                before,
                "declined splits must not leave invalid orphan folds"
            );
        }
        assert_graph_invariants(&graph, &caps);
    }
}

fn assert_graph_invariants(graph: &EGraph, caps: &Caps) {
    use fusor_ir::ir::VerifyCtx;
    for index in 0..graph.len() {
        let id = Id(index as u32);
        let node = graph.node(id);
        if let Op::Launch(fusor_ir::ir::launch::Launch::Contract {
            output,
            m,
            n,
            batch,
            ..
        }) = &node.op
        {
            assert_eq!(
                output.iterations(),
                Some(m.as_const().unwrap() * n.as_const().unwrap() * batch.as_const().unwrap())
            );
        }
        let operands: Vec<_> = node
            .children
            .iter()
            .map(|id| graph.facts(*id).clone())
            .collect();
        graph
            .semantics()
            .verify(&VerifyCtx {
                node,
                id,
                operands: &operands,
                result: graph.facts(id),
                caps,
            })
            .unwrap();
        if let Op::Union(a, b) = node.op {
            assert_eq!(
                graph.facts(a),
                graph.facts(b),
                "union {id} changes the value's type"
            );
        }
    }
}

#[test]
fn matvec_serial_cost_tracks_the_parallel_reduction() {
    use fusor_ir::ir::launch::{SchedPoint, SgemvParams};

    let caps = gpu_caps();
    for (k, cols, expected_steps) in [
        (1, 1, 1),
        (65, 1, 2),
        (896, 1, 14),
        (896, 4, 28),
        (14336, 1, 224),
        (14336, 4, 448),
    ] {
        let mut graph = graph();
        let a = leaf(&mut graph, 0, &consts([1, k]));
        let b = leaf(&mut graph, 1, &consts([k, 4096]));
        let id = contract(&mut graph, a, b, MATMUL);
        let launch = apply(
            &mut graph,
            &caps,
            id,
            fusor_tile::rules::contract::lower_sgemv,
        )
        .unwrap();
        let point = SchedPoint::Sgemv(SgemvParams {
            vector: 32,
            subgroups: 2,
            cols,
            parts: 1,
            gap: 0,
        });
        assert_eq!(
            fusor_cost::realize::node_serial_steps(&graph.node(launch).op, Some(point), &caps),
            (0, expected_steps),
            "k={k}, cols={cols}"
        );
    }
}

#[test]
fn symbolic_map_fusion_preserves_the_producer_address() {
    use fusor_ir::ir::launch::{AccessPlan, IndexSpace, Launch, Operand, ScheduleDomain};
    use fusor_ir::scalar::{BinOp, ScalarExpr};
    use fusor_ir::shape::{Layout, SymId};

    let s = Dim::Sym(SymId(0));
    for (shape, strides, should_fuse) in [
        (vec![s, Dim::Const(2)], vec![Dim::Const(2), Dim::ONE], true),
        (vec![s, Dim::Const(2)], vec![Dim::ONE, s], false),
        (
            vec![s, Dim::Const(2)],
            vec![Dim::Const(2), Dim::Const(0)],
            false,
        ),
        (vec![Dim::Const(2), s], vec![s, Dim::ONE], true),
        (vec![Dim::Const(2), s], vec![Dim::ONE, Dim::Const(2)], false),
        (
            vec![s, Dim::ONE, Dim::Const(2)],
            vec![Dim::Const(2), Dim::Const(7), Dim::ONE],
            true,
        ),
    ] {
        let caps = caps();
        let mut graph = graph();
        let input = leaf(&mut graph, 0, &shape);
        let arg = ScalarExpr::arg(0, Dtype::F32);
        let mut map = |body, src, layout| {
            graph
                .builder(&caps)
                .add_launch(Launch::Map {
                    space: IndexSpace::new(shape.iter().copied()),
                    body,
                    ops: vec![Operand {
                        src,
                        layout,
                        access: AccessPlan::Alias,
                    }],
                    sched: ScheduleDomain::Point,
                })
                .unwrap()
        };
        let square = ScalarExpr::bin(BinOp::Mul, arg.clone(), arg.clone());
        let producer = map(square, input, Layout::contiguous(&shape));
        let view = Layout::from_parts(Dim::Const(0), &shape, &strides).unwrap();
        let reader = map(arg, producer, view);
        let before = graph.len();
        let fused = apply(
            &mut graph,
            &caps,
            reader,
            fusor_ir::rules::fusion::map_into_map,
        );
        assert_eq!(
            fused.is_some(),
            should_fuse,
            "shape={shape:?}, strides={strides:?}"
        );
        if !should_fuse {
            assert_eq!(
                graph.len(),
                before,
                "an invalid fusion must never be minted"
            );
        }
    }
}
