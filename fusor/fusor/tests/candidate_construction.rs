use std::sync::Arc;

use fusor_ir::device::{Caps, DeviceKind, Limits};
use fusor_ir::dtype::Dtype;
use fusor_ir::egraph::EGraph;
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

#[test]
fn contraction_rule_does_not_construct_an_empty_schedule() {
    let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(Arc::new(
        fusor_tile::Planner::new(),
    )));
    let mut leaf = |name, shape: [u64; 2]| {
        graph
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(name),
                dtype: Dtype::F32,
                shape: shape.map(Dim::Const).into_iter().collect(),
            })))
            .unwrap()
    };
    let a = leaf(0, [9, 1]);
    let b = leaf(1, [1, 1]);
    let id = graph
        .add(Op::Logical(Logical::Contract {
            spec: EinSpec {
                a: [Label(0), Label(1)].into_iter().collect(),
                b: [Label(1), Label(2)].into_iter().collect(),
                out: [Label(0), Label(2)].into_iter().collect(),
            },
            a,
            b,
            acc: Dtype::F32,
            outs: 1,
        }))
        .unwrap();
    let caps = caps();
    let facts = graph.facts_view(id, &caps);
    let node = graph.node(id).clone();
    let before = graph.len();
    let generated =
        fusor_tile::rules::contract::lower_generic(&mut graph.builder(&caps), id, &node, &facts);
    assert!(generated.is_none());
    assert_eq!(
        before,
        graph.len(),
        "an inapplicable rule must not mint an invalid alternative"
    );
}

#[test]
fn cooperative_domains_contain_only_supported_scratch_combinations() {
    use fusor_ir::device::{CoopKind, SubgroupWidths};
    use fusor_ir::ir::kernel::{ArenaPlanner, ScalarElement};
    use fusor_ir::ir::launch::coop_tiles;
    let planner = fusor_tile::Planner::new();
    for limit in [2048, 16384, 32768] {
        for dtype in [Dtype::F32, Dtype::F16] {
            let mut caps = caps();
            caps.kind = DeviceKind::Gpu;
            caps.limits = Limits {
                max_compute_workgroup_storage_size: limit,
                ..Limits::default()
            };
            caps.subgroups = Some(SubgroupWidths { min: 32, max: 32 });
            caps.f16 = true;
            caps.coop.push(CoopKind {
                operand: dtype,
                acc: Dtype::F32,
                m: 8,
                n: 8,
                k: 8,
            });
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
    let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(Arc::new(
        fusor_tile::Planner::new(),
    )));
    let dims = [Dim::Const(64), Dim::Const(32)];
    let src = graph
        .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
            name: BufferId(0),
            dtype: Dtype::F32,
            shape: dims.into_iter().collect(),
        })))
        .unwrap();
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
            sched: ScheduleDomain::Fold(FoldDomain {
                strategies: [
                    FoldStrat::WgTree { lane_group: 256 },
                    FoldStrat::WgTree { lane_group: 1 },
                ]
                .into_iter()
                .collect(),
            }),
        }))
        .unwrap();
    let node = graph.node(id).clone();
    let facts = graph.facts_view(id, &caps);
    fusor_ir::rules::promote::promote(&mut graph.builder(&caps), id, &node, &facts)
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
    let mut caps = caps();
    caps.kind = DeviceKind::Gpu;
    caps.limits = Limits::default();
    let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(Arc::new(
        fusor_tile::Planner::new(),
    )));
    let mut leaf = |name, shape: [u64; 4]| {
        graph
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(name),
                dtype: Dtype::F32,
                shape: shape.map(Dim::Const).into_iter().collect(),
            })))
            .unwrap()
    };
    let a = leaf(0, [2, 3, 4, 5]);
    let b = leaf(1, [2, 3, 5, 6]);
    let id = graph
        .add(Op::Logical(Logical::Contract {
            spec: EinSpec {
                a: [0, 1, 2, 3].map(Label).into_iter().collect(),
                b: [0, 1, 3, 4].map(Label).into_iter().collect(),
                out: [0, 1, 2, 4].map(Label).into_iter().collect(),
            },
            a,
            b,
            acc: Dtype::F32,
            outs: 1,
        }))
        .unwrap();
    let node = graph.node(id).clone();
    let facts = graph.facts_view(id, &caps);
    let shape = facts.own().shape.clone();
    for rule in [
        fusor_tile::rules::contract::lower_generic,
        fusor_tile::rules::contract::lower_sgemm,
        fusor_tile::rules::contract::lower_sgemv,
    ] {
        let variant = rule(&mut graph.builder(&caps), id, &node, &facts).unwrap();
        assert_eq!(graph.facts(variant).shape, shape);
        assert_graph_invariants(&graph, &caps);
    }
}

fn assert_graph_invariants(graph: &EGraph, caps: &Caps) {
    use fusor_ir::egraph::Id;
    use fusor_ir::ir::{OpDefRegistry, VerifyCtx};
    let registry = OpDefRegistry::new();
    for index in 0..graph.len() {
        let id = Id(index as u32);
        let node = graph.node(id);
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
                registry: &registry,
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
