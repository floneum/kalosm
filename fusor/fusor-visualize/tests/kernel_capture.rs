use fusor_ir::{
    dtype::NumericContract,
    ir::kernel::*,
    scalar::{BinOp, CmpOp},
};
use fusor_visualize::{
    AccessKind, Action, GpuConfig, Simulator, demo,
    kernel::{CaptureOptions, capture},
};
use std::sync::Arc;

fn lit(v: u32) -> TileExpr {
    TileExpr::new(
        TileExprKind::Literal(TileLiteral::U32(v)),
        ScalarElement::U32.element(),
    )
}
fn lane() -> TileExpr {
    TileExpr::new(
        TileExprKind::Builtin(Builtin::Lane),
        ScalarElement::U32.element(),
    )
}
fn bin(op: BinOp, a: TileExpr, b: TileExpr) -> TileExpr {
    TileExpr::new(
        TileExprKind::Binary {
            op,
            left: a,
            right: b,
            numeric: NumericContract::STRICT,
        },
        ScalarElement::U32.element(),
    )
}
fn local_value(local: &Local) -> TileExpr {
    TileExpr::new(TileExprKind::LoadLocal(local.clone()), local.element)
}
fn load(ir: &KernelIr, index: TileExpr, mask: TileExpr) -> TileExpr {
    let buffer = ir.buffers[0].clone();
    TileExpr::new(
        TileExprKind::Load {
            src: Source::Storage(StorageView {
                offset: 0,
                layout: buffer.layout.clone(),
                buffer,
            }),
            addr: Box::new(Addr::Linear(index)),
            mask,
            fill: lit(0),
        },
        ScalarElement::U32.element(),
    )
}
fn reads(trace: &fusor_visualize::Trace) -> Vec<Vec<u64>> {
    trace
        .events
        .iter()
        .filter_map(|event| match &event.action {
            Action::Memory {
                access: AccessKind::Read,
                lanes,
                ..
            } => Some(lanes.iter().map(|l| l.address).collect()),
            _ => None,
        })
        .collect()
}

#[test]
fn quantized_decoders_trace_packed_words_in_both_layouts() {
    use fusor_ir::dtype::{QFmt, QLayout};
    for (fmt, layout, stride, offsets) in [
        (QFmt::Q4K, QLayout::Native, 144, vec![0, 4, 8, 12, 143]),
        (
            QFmt::Q4K,
            QLayout::F32Scales,
            148,
            vec![0, 4, 8, 12, 16, 147],
        ),
        (QFmt::Q6K, QLayout::Native, 210, vec![127, 191, 207, 208]),
        (QFmt::Q6K, QLayout::F32Scales, 212, vec![127, 191, 207, 208]),
    ] {
        let mut ir = demo::kernel("coalesced", 8).unwrap();
        let mask = TileExpr::new(
            TileExprKind::Compare {
                op: CmpOp::Lt,
                left: lane(),
                right: lit(3),
            },
            ScalarElement::Bool.element(),
        );
        let value = TileExpr::new(
            TileExprKind::Load {
                src: Source::Quantized(QuantizedView {
                    fmt,
                    layout,
                    data: StorageView {
                        buffer: ir.buffers[0].clone(),
                        offset: 7,
                        layout: TileLayout::contiguous(MemoryLevel::Storage, &[8, 256]),
                    },
                }),
                addr: Box::new(Addr::Rc2 {
                    row: lane(),
                    col: lit(255),
                }),
                mask,
                fill: TileExpr::new(
                    TileExprKind::Literal(TileLiteral::F32(0)),
                    ScalarElement::F32.element(),
                ),
            },
            ScalarElement::F32.element(),
        );
        ir.body = vec![Stmt::StoreLocal {
            dst: Arc::new(LocalDecl::new(ScalarElement::F32.element())),
            value,
        }];
        let trace = capture(&ir, CaptureOptions::default()).unwrap();
        for lane in 0..3u64 {
            let actual: std::collections::BTreeSet<_> = trace
                .events
                .iter()
                .flat_map(|e| match &e.action {
                    Action::Memory { lanes, .. } => lanes
                        .iter()
                        .filter(|l| u64::from(l.lane) == lane)
                        .map(|l| l.address)
                        .collect::<Vec<_>>(),
                    _ => vec![],
                })
                .collect();
            let expected = offsets
                .iter()
                .map(|off| ((lane * stride + off) / 4) * 4 + 28)
                .collect();
            assert_eq!(actual, expected, "{fmt:?}/{layout:?}, lane {lane}");
        }
        for e in &trace.events {
            if let Action::Memory { lanes, .. } = &e.action {
                assert!(lanes.iter().all(|l| l.lane < 3));
            }
        }
    }
}

#[test]
fn vector_arithmetic_preserves_scalar_reads_but_cannot_invent_addresses() {
    let mut ir = demo::kernel("coalesced", 4).unwrap();
    let vec = TileExpr::new(
        TileExprKind::Vec {
            lanes: 2,
            scalar: ScalarElement::U32,
            parts: vec![
                load(&ir, lane(), lit(1)),
                load(&ir, bin(BinOp::Add, lane(), lit(8)), lit(1)),
            ],
        },
        ElementType::Vector {
            scalar: ScalarElement::U32,
            lanes: 2,
        },
    );
    let component = TileExpr::new(
        TileExprKind::VecComponent {
            vector: vec,
            component: 0,
        },
        ScalarElement::U32.element(),
    );
    ir.body = vec![Stmt::StoreLocal {
        dst: Arc::new(LocalDecl::new(ScalarElement::U32.element())),
        value: component.clone(),
    }];
    let trace = capture(&ir, CaptureOptions::default()).unwrap();
    assert_eq!(reads(&trace), vec![vec![0, 4, 8, 12], vec![32, 36, 40, 44]]);
    ir.body = vec![Stmt::StoreLocal {
        dst: Arc::new(LocalDecl::new(ScalarElement::U32.element())),
        value: load(&ir, component, lit(1)),
    }];
    assert!(
        capture(&ir, CaptureOptions::default())
            .unwrap_err()
            .0
            .contains("unknown")
    );
}

#[test]
fn all_demos_capture_and_run_at_different_subgroup_sizes_without_a_gpu() {
    for name in demo::NAMES {
        for width in [16, 32, 64] {
            let trace = capture(
                &demo::kernel(name, 64).unwrap(),
                CaptureOptions {
                    subgroup_size: width,
                    ..CaptureOptions::default()
                },
            )
            .unwrap();
            let mut sim = Simulator::new(
                trace,
                GpuConfig {
                    subgroup_size: width,
                    ..GpuConfig::default()
                },
            )
            .unwrap();
            while sim.step().is_some() {}
            if *name == "padded" {
                assert_eq!(sim.totals().bank_conflict_steps, 0);
            }
        }
    }
}

#[test]
fn masked_lanes_never_request_memory_even_with_invalid_inactive_addresses() {
    let mut ir = demo::kernel("coalesced", 8).unwrap();
    let mask = TileExpr::new(
        TileExprKind::Compare {
            op: CmpOp::Lt,
            left: lane(),
            right: lit(3),
        },
        ScalarElement::Bool.element(),
    );
    let index = bin(BinOp::Div, lit(12), bin(BinOp::Sub, lit(3), lane()));
    ir.body = vec![Stmt::StoreLocal {
        dst: Arc::new(LocalDecl::new(ScalarElement::U32.element())),
        value: load(&ir, index, mask),
    }];
    let trace = capture(&ir, CaptureOptions::default()).unwrap();
    assert_eq!(reads(&trace), vec![vec![16, 24, 48]]);
}

#[test]
fn counted_loops_update_indices_and_accumulators_simultaneously() {
    let mut ir = demo::kernel("coalesced", 1).unwrap();
    let elem = ScalarElement::U32.element();
    let a = Arc::new(LocalDecl::new(elem));
    let b = Arc::new(LocalDecl::new(elem));
    let i = Arc::new(LocalDecl::new(elem));
    let sink = Arc::new(LocalDecl::new(elem));
    ir.body = vec![Stmt::Loop {
        count: Some(lit(3)),
        index: Some(i.clone()),
        accumulators: vec![
            Accumulator {
                local: a.clone(),
                init: lit(1),
                update: local_value(&b),
            },
            Accumulator {
                local: b.clone(),
                init: lit(2),
                update: local_value(&a),
            },
        ],
        body: vec![Stmt::StoreLocal {
            dst: sink,
            value: load(
                &ir,
                bin(BinOp::Add, local_value(&a), local_value(&i)),
                lit(1),
            ),
        }],
    }];
    let trace = capture(&ir, CaptureOptions::default()).unwrap();
    assert_eq!(reads(&trace), vec![vec![4], vec![12], vec![12]]);
}

#[test]
fn divergent_control_reconverges_and_returned_lanes_stay_inactive() {
    let mut ir = demo::kernel("coalesced", 4).unwrap();
    let body = ir.body.clone();
    ir.body = vec![Stmt::If {
        condition: TileExpr::new(
            TileExprKind::Compare {
                op: CmpOp::Lt,
                left: lane(),
                right: lit(2),
            },
            ScalarElement::Bool.element(),
        ),
        accept: vec![Stmt::Return],
        reject: body.clone(),
    }];
    ir.body.extend(body);
    let trace = capture(&ir, CaptureOptions::default()).unwrap();
    assert_eq!(reads(&trace), vec![vec![8, 12], vec![8, 12]]);
    ir.body.push(Stmt::Barrier);
    assert!(
        capture(&ir, CaptureOptions::default())
            .unwrap_err()
            .0
            .contains("divergent")
    );
}

#[test]
fn data_dependent_indices_fail_instead_of_becoming_zero() {
    let mut ir = demo::kernel("coalesced", 4).unwrap();
    let gathered = load(&ir, load(&ir, lane(), lit(1)), lit(1));
    ir.body = vec![Stmt::StoreLocal {
        dst: Arc::new(LocalDecl::new(ScalarElement::U32.element())),
        value: gathered,
    }];
    assert!(
        capture(&ir, CaptureOptions::default())
            .unwrap_err()
            .0
            .contains("unknown")
    );
}

#[test]
fn concrete_input_bytes_resolve_indirect_addresses_and_writes_invalidate_them() {
    let mut ir = demo::kernel("coalesced", 4).unwrap();
    let gathered = load(&ir, load(&ir, lane(), lit(1)), lit(1));
    ir.body = vec![Stmt::StoreLocal {
        dst: Arc::new(LocalDecl::new(ScalarElement::U32.element())),
        value: gathered,
    }];
    let options = CaptureOptions {
        known_bindings: std::collections::BTreeMap::from([(
            1,
            [3u32, 2, 1, 0]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect(),
        )]),
        ..CaptureOptions::default()
    };
    assert_eq!(
        reads(&capture(&ir, options.clone()).unwrap()),
        vec![vec![0, 4, 8, 12], vec![12, 8, 4, 0]]
    );
    let buffer = ir.buffers[0].clone();
    ir.body.insert(
        0,
        Stmt::Store {
            dst: StorageView {
                layout: buffer.layout.clone(),
                buffer,
                offset: 0,
            },
            addr: Addr::Linear(lane()),
            value: lit(1),
            mask: lit(1),
        },
    );
    assert!(capture(&ir, options).unwrap_err().0.contains("unknown"));
}

#[test]
fn empty_loops_are_budgeted_and_invalid_workgroups_are_rejected() {
    let mut ir = demo::kernel("coalesced", 4).unwrap();
    ir.body = vec![Stmt::Loop {
        count: Some(lit(u32::MAX)),
        index: None,
        accumulators: vec![],
        body: vec![],
    }];
    assert!(
        capture(
            &ir,
            CaptureOptions {
                max_actions: 10,
                ..CaptureOptions::default()
            }
        )
        .unwrap_err()
        .0
        .contains("budget")
    );
    assert!(
        capture(
            &ir,
            CaptureOptions {
                workgroups: vec![[2, 0, 0]],
                ..CaptureOptions::default()
            }
        )
        .is_err()
    );
}

#[test]
fn storage_offset_and_rank_two_non_contiguous_layout_are_applied() {
    let mut ir = demo::kernel("coalesced", 4).unwrap();
    let buffer = ir.buffers[0].clone();
    let view = StorageView {
        buffer,
        offset: 7,
        layout: TileLayout {
            extents: vec![4, 4].into(),
            indexing: fusor_ir::shape::MultiFlattenMap::affine(&[4, 4], &[1, 8]),
            level: MemoryLevel::Storage,
        },
    };
    let value = TileExpr::new(
        TileExprKind::Load {
            src: Source::Storage(view),
            addr: Box::new(Addr::Rc2 {
                row: lane(),
                col: lit(2),
            }),
            mask: lit(1),
            fill: lit(0),
        },
        ScalarElement::U32.element(),
    );
    ir.body = vec![Stmt::StoreLocal {
        dst: Arc::new(LocalDecl::new(ScalarElement::U32.element())),
        value,
    }];
    assert_eq!(
        reads(&capture(&ir, CaptureOptions::default()).unwrap()),
        vec![vec![92, 96, 100, 104]]
    );
}

#[test]
fn unsupported_collectives_are_rejected_even_in_unreachable_branches() {
    let mut ir = demo::kernel("coalesced", 4).unwrap();
    let tile = Arc::new(TileDecl::new(
        ScalarElement::U32.element(),
        TileLayout::contiguous(MemoryLevel::Workgroup, &[4]),
        "x",
    ));
    ir.body = vec![Stmt::If {
        condition: lit(0),
        accept: vec![Stmt::FillTile {
            dst: tile,
            value: lit(1),
            bounds: [None, None],
        }],
        reject: vec![],
    }];
    assert!(
        capture(&ir, CaptureOptions::default())
            .unwrap_err()
            .0
            .contains("backend-specific")
    );
}
