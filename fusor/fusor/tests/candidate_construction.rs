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
fn shared_matmul_ancestors_are_charged_once_with_every_branch() {
    use fusor_ir::device::SubgroupWidths;
    use fusor_ir::dtype::{QFmt, QLayout};
    use fusor_ir::egraph::{Saturate, SaturationBudget};
    use fusor_ir::extract::Extractor;
    use fusor_ir::ir::launch::Launch;
    use fusor_ir::scalar::{BinOp, ScalarExpr};

    let mut caps = caps();
    caps.kind = DeviceKind::Gpu;
    caps.limits = Limits::default();
    caps.subgroups = Some(SubgroupWidths { min: 32, max: 32 });
    let planner = Arc::new(fusor_tile::Planner::new());
    let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(planner.clone()));
    let mut x = graph
        .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
            name: BufferId(0),
            dtype: Dtype::F32,
            shape: [Dim::ONE, Dim::Const(4096)].into_iter().collect(),
        })))
        .unwrap();
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
            let matmul = graph
                .add(Op::Logical(Logical::Contract {
                    spec: EinSpec {
                        a: [Label(0), Label(1)].into_iter().collect(),
                        b: [Label(2), Label(1)].into_iter().collect(),
                        out: [Label(0), Label(2)].into_iter().collect(),
                    },
                    acc: Dtype::F32,
                    a: x,
                    b: weight,
                    outs: 1,
                }))
                .unwrap();
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
    use fusor_ir::device::{CoopKind, SubgroupWidths};
    use fusor_ir::extract::Extractor;
    use fusor_ir::ir::launch::{
        AccessPlan, ContractSide, Family, IndexSpace, Launch, Operand, ScheduleDomain,
    };
    use fusor_ir::scalar::ScalarExpr;
    use fusor_ir::shape::Layout;

    let mut caps = caps();
    caps.kind = DeviceKind::Gpu;
    caps.limits = Limits::default();
    caps.subgroups = Some(SubgroupWidths { min: 32, max: 32 });
    caps.coop.push(CoopKind {
        operand: Dtype::F32,
        acc: Dtype::F32,
        m: 8,
        n: 8,
        k: 8,
    });
    let planner = Arc::new(fusor_tile::Planner::new());
    let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(planner.clone()));
    let domain = fusor_tile::domains::coop::legal(
        Dim::Const(128),
        Dim::Const(128),
        Dim::Const(4096),
        Dtype::F32,
        Dtype::F32,
        &caps,
    );
    let side = |graph: &mut EGraph, name, shape: [Dim; 2]| {
        let src = graph
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(name),
                dtype: Dtype::F32,
                shape: shape.into_iter().collect(),
            })))
            .unwrap();
        ContractSide::one(
            ScalarExpr::arg(0, Dtype::F32),
            Operand {
                src,
                layout: Layout::contiguous(&shape),
                access: AccessPlan::Alias,
            },
        )
    };
    let mut matmuls = Vec::new();
    for i in 0..4096 {
        let a = side(&mut graph, 2 * i, [Dim::Const(128), Dim::Const(4096)]);
        let b = side(&mut graph, 2 * i + 1, [Dim::Const(4096), Dim::Const(128)]);
        matmuls.push(
            graph
                .builder(&caps)
                .add_launch(Launch::Contract {
                    output: IndexSpace::new([Dim::Const(128), Dim::Const(128)]),
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
    let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(Arc::new(
        fusor_tile::Planner::new(),
    )));
    let length = graph.fresh_sym();
    for varying_k in [false, true] {
        let n = if varying_k {
            Dim::Const(7)
        } else {
            Dim::Sym(length)
        };
        let k = if varying_k {
            Dim::Sym(length)
        } else {
            Dim::Const(19)
        };
        let mut leaf = |name, shape: [Dim; 3]| {
            graph
                .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                    name: BufferId(name),
                    dtype: Dtype::F32,
                    shape: shape.into_iter().collect(),
                })))
                .unwrap()
        };
        let a = leaf(0, [Dim::Const(2), Dim::Const(3), k]);
        let b = leaf(1, [Dim::Const(2), n, k]);
        let id = graph
            .add(Op::Logical(Logical::Contract {
                spec: EinSpec {
                    a: [Label(0), Label(1), Label(3)].into_iter().collect(),
                    b: [Label(0), Label(2), Label(3)].into_iter().collect(),
                    out: [Label(0), Label(1), Label(2)].into_iter().collect(),
                },
                a,
                b,
                acc: Dtype::F32,
                outs: 1,
            }))
            .unwrap();
        let node = graph.node(id).clone();
        let facts = graph.facts_view(id, &caps);
        fusor_ir::rules::lower_floor::lower_contract_generic(
            &mut graph.builder(&caps),
            id,
            &node,
            &facts,
        )
        .expect("a dynamic contraction must remain visible as a reduction");
        let folded = graph.members(graph.class_of(id)).into_iter().find(|member|
            matches!(graph.node(*member).op, Op::Launch(Launch::Fold { .. }))
        ).unwrap();
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
    use fusor_ir::device::{CoopKind, SubgroupWidths};
    use fusor_ir::ir::launch::Launch;
    use fusor_ir::scalar::{BinOp, ScalarExpr};
    let mut caps = caps();
    caps.kind = DeviceKind::Gpu;
    caps.limits = Limits::default();
    caps.subgroups = Some(SubgroupWidths { min: 32, max: 32 });
    caps.coop.push(CoopKind {
        operand: Dtype::F32,
        acc: Dtype::F32,
        m: 8,
        n: 8,
        k: 8,
    });
    for (batch, m, n) in [
        (&[2, 3][..], &[4][..], &[6][..]),
        (&[1, 1][..], &[4][..], &[6][..]),
        (&[2, 1][..], &[3, 4][..], &[5, 6][..]),
    ] {
        let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(Arc::new(
            fusor_tile::Planner::new(),
        )));
        let mut leaf = |name, shape: Vec<u64>| {
            graph
                .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                    name: BufferId(name),
                    dtype: Dtype::F32,
                    shape: shape.into_iter().map(Dim::Const).collect(),
                })))
                .unwrap()
        };
        let a = leaf(0, batch.iter().chain(m).copied().chain([256]).collect());
        let b = leaf(
            1,
            batch
                .iter()
                .copied()
                .chain([256])
                .chain(n.iter().copied())
                .collect(),
        );
        let batch_end = batch.len() as u8;
        let m_end = batch_end + m.len() as u8;
        let n_end = m_end + n.len() as u8;
        let id = graph
            .add(Op::Logical(Logical::Contract {
                spec: EinSpec {
                    a: (0..m_end).chain([n_end]).map(Label).collect(),
                    b: (0..batch_end)
                        .chain([n_end])
                        .chain(m_end..n_end)
                        .map(Label)
                        .collect(),
                    out: (0..n_end).map(Label).collect(),
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
            fusor_tile::rules::contract::lower_coop,
        ] {
            let variant = rule(&mut graph.builder(&caps), id, &node, &facts).unwrap();
            assert_eq!(graph.facts(variant).shape, shape);
            let variant_node = graph.node(variant).clone();
            if matches!(variant_node.op, Op::Launch(Launch::Contract { .. })) {
                let facts = graph.facts_view(variant, &caps);
                let split = fusor_ir::rules::split_k::split_k(
                    &mut graph.builder(&caps),
                    variant,
                    &variant_node,
                    &facts,
                )
                .unwrap();
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
                    let node = graph.node(indexed).clone();
                    let facts = graph.facts_view(indexed, &caps);
                    let before = graph.len();
                    assert!(
                        fusor_ir::rules::split_k::split_k(
                            &mut graph.builder(&caps),
                            indexed,
                            &node,
                            &facts,
                        )
                        .is_none()
                    );
                    assert_eq!(graph.len(), before);
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
    let mut caps = caps();
    caps.kind = DeviceKind::Gpu;
    caps.limits = Limits::default();
    for (rows, expected) in [
        (vec![s], 5),
        (vec![s, Dim::ONE], 5),
        (vec![s, Dim::Const(2)], 10),
        (vec![Dim::Const(2), s], 10),
        (vec![s, t], 15),
    ] {
        let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(Arc::new(
            fusor_tile::Planner::new(),
        )));
        let mut leaf = |name, shape: Vec<Dim>| {
            graph
                .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                    name: BufferId(name),
                    dtype: Dtype::F32,
                    shape: shape.into_iter().collect(),
                })))
                .unwrap()
        };
        let a = leaf(0, rows.iter().copied().chain([Dim::Const(8)]).collect());
        let b = leaf(1, vec![Dim::Const(8), Dim::Const(4)]);
        let k = rows.len() as u8;
        let id = graph
            .add(Op::Logical(Logical::Contract {
                spec: EinSpec {
                    a: (0..=k).map(Label).collect(),
                    b: [Label(k), Label(k + 1)].into_iter().collect(),
                    out: (0..k).chain([k + 1]).map(Label).collect(),
                },
                a,
                b,
                acc: Dtype::F32,
                outs: 1,
            }))
            .unwrap();
        let node = graph.node(id).clone();
        let facts = graph.facts_view(id, &caps);
        let variant =
            fusor_tile::rules::contract::lower_sgemm(&mut graph.builder(&caps), id, &node, &facts)
                .unwrap();
        let node = graph.node(variant).clone();
        let Op::Launch(Launch::Contract { m, .. }) = &node.op else {
            unreachable!()
        };
        assert_eq!(
            m.evaluate(&mut |sym| [5, 3].get(sym.0 as usize).copied()),
            Some(expected)
        );
        assert_eq!(graph.facts(variant).shape, facts.own().shape);
        let facts = graph.facts_view(variant, &caps);
        let before = graph.len();
        assert!(
            fusor_ir::rules::specialize::specialize_dim(
                &mut graph.builder(&caps),
                variant,
                &node,
                &facts,
            )
            .is_none()
        );
        assert_eq!(graph.len(), before);
    }
}

#[test]
fn grouped_padding_symbols_reach_the_uniform_plan() {
    use fusor_cost::{Roofline, extract::LocalSearch, realize::NodeCache};
    use fusor_ir::device::{CoopKind, SubgroupWidths};
    use fusor_ir::extract::Extraction;
    use fusor_ir::ir::launch::Launch;
    use fusor_ir::shape::SymId;
    let mut caps = caps();
    caps.kind = DeviceKind::Gpu;
    caps.limits = Limits::default();
    caps.subgroups = Some(SubgroupWidths { min: 32, max: 32 });
    caps.coop.push(CoopKind {
        operand: Dtype::F32,
        acc: Dtype::F32,
        m: 8,
        n: 8,
        k: 8,
    });
    let planner = Arc::new(fusor_tile::Planner::new());
    let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(planner.clone()));
    let batch = Dim::Sym(SymId(0));
    let mut leaf = |name, m, n| {
        graph
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(name),
                dtype: Dtype::F32,
                shape: [Dim::Const(2), batch, Dim::Const(m), Dim::Const(n)]
                    .into_iter()
                    .collect(),
            })))
            .unwrap()
    };
    let a = leaf(0, 3, 8);
    let b = leaf(1, 8, 5);
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
    let variant =
        fusor_tile::rules::contract::lower_coop(&mut graph.builder(&caps), id, &node, &facts)
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
        let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(Arc::new(
            fusor_tile::Planner::new(),
        )));
        let input = graph
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(0),
                dtype: Dtype::F32,
                shape: [Dim::Const(4), Dim::Const(996)].into_iter().collect(),
            })))
            .unwrap();
        if carrier.width() > 1 {
            let before = graph.len();
            assert!(
                graph
                    .add(Op::Logical(Logical::Fold {
                        carrier: carrier.as_merge(),
                        axis: 1,
                        acc: Dtype::F32,
                        ins: smallvec::smallvec![input],
                    }))
                    .is_err()
            );
            assert_eq!(
                graph.len(),
                before,
                "unbound lift operands must never enter the graph"
            );
        }
        let id = graph
            .add(Op::Logical(Logical::Fold {
                carrier,
                axis: 1,
                acc: Dtype::F32,
                ins: smallvec::smallvec![input],
            }))
            .unwrap();
        let node = graph.node(id).clone();
        let facts = graph.facts_view(id, &caps);
        let before = graph.len();
        let variant = fusor_ir::rules::algebra::strip(&mut graph.builder(&caps), id, &node, &facts);
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
    use fusor_ir::egraph::Id;
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
    use fusor_ir::device::SubgroupWidths;
    use fusor_ir::ir::launch::{SchedPoint, SgemvParams};

    let mut caps = caps();
    caps.kind = DeviceKind::Gpu;
    caps.limits = Limits::default();
    caps.subgroups = Some(SubgroupWidths { min: 32, max: 32 });
    for (k, cols, expected_steps) in [
        (1, 1, 1),
        (65, 1, 2),
        (896, 1, 14),
        (896, 4, 28),
        (14336, 1, 224),
        (14336, 4, 448),
    ] {
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
        let a = leaf(0, [1, k]);
        let b = leaf(1, [k, 4096]);
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
        let node = graph.node(id).clone();
        let facts = graph.facts_view(id, &caps);
        let launch =
            fusor_tile::rules::contract::lower_sgemv(&mut graph.builder(&caps), id, &node, &facts)
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
        let mut graph = EGraph::new(fusor_ir::CoreSemantics::new(Arc::new(
            fusor_tile::Planner::new(),
        )));
        let input = graph
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(0),
                dtype: Dtype::F32,
                shape: shape.iter().copied().collect(),
            })))
            .unwrap();
        let arg = ScalarExpr::arg(0, Dtype::F32);
        let producer = graph
            .builder(&caps)
            .add_launch(Launch::Map {
                space: IndexSpace::new(shape.iter().copied()),
                body: ScalarExpr::bin(BinOp::Mul, arg.clone(), arg.clone()),
                ops: vec![Operand {
                    src: input,
                    layout: Layout::contiguous(&shape),
                    access: AccessPlan::Alias,
                }],
                sched: ScheduleDomain::Point,
            })
            .unwrap();
        let reader = graph
            .builder(&caps)
            .add_launch(Launch::Map {
                space: IndexSpace::new(shape.iter().copied()),
                body: arg,
                ops: vec![Operand {
                    src: producer,
                    layout: Layout::from_parts(Dim::Const(0), &shape, &strides).unwrap(),
                    access: AccessPlan::Alias,
                }],
                sched: ScheduleDomain::Point,
            })
            .unwrap();
        let node = graph.node(reader).clone();
        let facts = graph.facts_view(reader, &caps);
        let before = graph.len();
        let fused =
            fusor_ir::rules::fusion::map_into_map(&mut graph.builder(&caps), reader, &node, &facts);
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
