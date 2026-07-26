use super::*;
use crate::ScalarElement;

fn f32() -> ElementType {
    ScalarElement::F32.element()
}

fn u32() -> ElementType {
    ScalarElement::U32.element()
}

#[test]
fn op_enum_is_source_tile_program_only() {
    let ir = tile::build(|phase| {
        let x = phase.storage_read(f32(), Shape::new([1, 8]));
        let y = phase.storage_write(f32(), Shape::new([1, 8]));
        phase.program_grid(8, [1, 1, 1], |program| {
            let lane = program.lane();
            let mask = lane.lt(8);
            let value = program.load(x.at((0, &lane)), mask.clone(), 0.0);
            program.store(y.at((0, lane)), value, mask);
        });
    });

    assert!(!ir.body.is_empty());
}

#[test]
fn tile_source_softmax_lowers_to_naga() {
    const ROWS: u32 = 4;
    const COLS: u32 = 100;
    const BLOCK: u32 = 128;
    let ir = tile::build(|phase| {
        let x = phase.storage_read(f32(), Shape::new([ROWS, COLS]));
        let y = phase.storage_write(f32(), Shape::new([ROWS, COLS]));
        phase.program_grid(BLOCK, [1, ROWS, 1], |program| {
            let row = program.program_id(WorkgroupAxis::Y);
            let col = program.lane();
            let mask = col.lt(COLS);
            let values = program.load(x.at((&row, &col)), mask.clone(), -3.40282e38);
            let max = program.reduce_max(values.clone());
            let exp = (values - max).exp();
            let sum = program.reduce_sum(exp.clone());
            program.store(y.at((row, col)), exp / sum, mask);
        });
    });

    let lowered = lower_or_fail(&ir, "tile softmax");
    let mut wgsl = String::new();
    let mut writer =
        naga::back::wgsl::Writer::new(&mut wgsl, naga::back::wgsl::WriterFlags::empty());
    writer
        .write(lowered.module(), lowered.info())
        .expect("WGSL serialization should succeed");
    naga::front::wgsl::parse_str(&wgsl).expect("WGSL should parse after serialization");
}

#[test]
fn subgroup_reduce_records_wgsl_extension_requirement() {
    let ir = tile::build(|phase| {
        let subgroup = crate::SubgroupToken::new_unchecked();
        let x = phase.storage_read(f32(), Shape::new([32]));
        let y = phase.storage_write(f32(), Shape::new([32]));
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            let mask = lane.clone().lt(32u32);
            let value = program.load(x.at(lane.clone()), mask.clone(), 0.0);
            let max = subgroup.subgroup_reduce_max(program, value);
            program.store(y.at(lane), max, mask);
        });
    });

    let lowered = lower_or_fail(&ir, "subgroup reduce");
    assert_eq!(lowered.wgsl_extension_prelude(), "enable subgroups;\n\n");
}

#[test]
fn subgroup_builtins_use_subgroup_capability_and_wgsl_extension() {
    let ir = tile::build(|phase| {
        let subgroup = crate::SubgroupToken::new_unchecked();
        let y = phase.storage_write(u32(), Shape::new([32]));
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            let value = subgroup.subgroup_lane(program) + subgroup.subgroup_size(program);
            program.store(y.at(lane), value, true);
        });
    });

    let lowered = lower_or_fail(&ir, "subgroup builtins");
    assert_eq!(lowered.wgsl_extension_prelude(), "enable subgroups;\n\n");
}

#[test]
fn kernel_builder_unused_storage_still_lowers_as_binding() {
    let layout = Layout::contiguous(MemoryLevel::Storage, Shape::new([8]));
    let mut kb = KernelBuilder::<&'static str>::new();
    let _unused = kb.read(f32(), KernelTensorRef::new("unused", layout.clone()));
    let x = kb.read(f32(), KernelTensorRef::new("input", layout.clone()));
    let y = kb.write(f32(), KernelTensorRef::new("output", layout));

    kb.program().program_grid(8, [1, 1, 1], |program| {
        let lane = program.lane();
        let mask = lane.clone().lt(8u32);
        let value = program.load(x.at(lane.clone()), mask.clone(), 0.0);
        program.store(y.at(lane), value, mask);
    });

    let (ir, bindings) = kb.finish();
    assert_eq!(bindings, ["unused", "input", "output"]);

    let lowered = lower_or_fail(&ir, "unused KernelBuilder storage");
    let mut storage_bindings: Vec<_> = lowered
        .module()
        .global_variables
        .iter()
        .filter_map(|(_, global)| match global.space {
            naga::AddressSpace::Storage { .. } => Some(
                global
                    .binding
                    .as_ref()
                    .expect("storage global has resource binding")
                    .binding,
            ),
            _ => None,
        })
        .collect();
    storage_bindings.sort_unstable();

    assert_eq!(storage_bindings, [0, 1, 2]);
}

#[test]
fn lowered_naga_uses_anonymous_ir_objects_except_entry_point() {
    let ir = tile::build(|phase| {
        let x = phase.storage_read(f32(), Shape::new([1, 8]));
        let y = phase.storage_write(f32(), Shape::new([1, 8]));
        phase.program_grid(8, [1, 1, 1], |program| {
            let lane = program.lane();
            let mask = lane.lt(8);
            let value = program.load(x.at((0, &lane)), mask.clone(), 0.0);
            program.store(y.at((0, lane)), value, mask);
        });
    });
    let lowered = lower_or_fail(&ir, "tile");
    let module = lowered.module();

    assert!(module.types.iter().all(|(_, ty)| ty.name.is_none()));
    assert!(module
        .global_variables
        .iter()
        .all(|(_, global)| global.name.is_none()));
    for entry in &module.entry_points {
        assert_eq!(entry.name, "main");
        assert!(entry.function.name.is_none());
        assert!(entry
            .function
            .arguments
            .iter()
            .all(|arg| arg.name.is_none()));
        assert!(entry
            .function
            .local_variables
            .iter()
            .all(|(_, local)| local.name.is_none()));
    }
}

#[test]
fn generic_vector_load_and_dot_lower_to_naga() {
    let ir = tile::build(|phase| {
        let x = phase.storage_read(ElementType::vector(ScalarElement::F32, 2), Shape::new([16]));
        let y = phase.storage_write(f32(), Shape::new([16]));
        phase.program_grid(16, [1, 1, 1], |program| {
            let lane = program.lane();
            let mask = lane.clone().lt(16);
            let value = program.load(x.at(lane.clone()), mask.clone(), TileLiteral::f32(0.0));
            let dot = program.vector_dot(value.clone(), value);
            program.store(y.at(lane), dot, mask);
        });
    });

    lower_or_fail(&ir, "generic vec2 dot");
}

#[test]
fn vector_load_casts_mask_fill_before_splatting() {
    let ir = tile::build(|phase| {
        let element = ElementType::vector(ScalarElement::F16, 4);
        let x = phase.storage_read(element, Shape::new([16]));
        let y = phase.storage_write(element, Shape::new([16]));
        phase.program_grid(16, [1, 1, 1], |program| {
            let lane = program.lane();
            let mask = lane.clone().lt(16);
            let value = program.load(x.at(lane.clone()), mask.clone(), 0.0);
            program.store(y.at(lane), value, mask);
        });
    });

    lower_or_fail(&ir, "f16 vec4 load with f32 fill");
}

#[test]
fn if_branches_do_not_share_branch_local_expression_memos() {
    let local = std::rc::Rc::new(LocalDecl {
        element: ElementType::F32,
    });
    let one = Expr::new(ExprKind::Literal(TileLiteral::f32(1.0)), ElementType::F32);
    let two = Expr::new(ExprKind::Literal(TileLiteral::f32(2.0)), ElementType::F32);
    let shared = Expr::new(
        ExprKind::Shared(Expr::new(
            ExprKind::Binary {
                op: TileBinaryOp::Add,
                left: Box::new(one),
                right: Box::new(two),
            },
            ElementType::F32,
        )),
        ElementType::F32,
    );
    let ir = KernelIr {
        buffers: Vec::new(),
        grid: [1, 1, 1],
        block: 1,
        byte_arena: false,
        body: vec![Stmt::If {
            condition: Expr::new(
                ExprKind::Literal(TileLiteral::Bool(true)),
                ElementType::Bool,
            ),
            accept: vec![Stmt::StoreLocal {
                dst: local.clone(),
                value: shared.clone(),
            }],
            reject: vec![Stmt::StoreLocal {
                dst: local,
                value: shared,
            }],
        }],
    };

    lower_or_fail(&ir, "if branch shared expression memo");
}

#[test]
fn typed_coop_accumulator_records_scalar_role_and_shape() {
    // A coop accumulator is a `CoopMatrix { role: C, .. }`-typed private local;
    // the StoreLocal reaching it carries that element type. The IR is a tree, so
    // we inspect the emitted body directly (there is no `locals()` side-table).
    let ir = tile::build(|phase| {
        let coop = crate::CoopMatrixToken::new_unchecked();
        phase.program_grid(32, [1, 1, 1], |program| {
            let acc = coop.alloc_coop_acc(program, ScalarElement::F32, 8, 8);
            let zero = coop.coop_zero(program, ScalarElement::F32, 8, 8);
            coop.coop_store_local(program, &acc, zero);
        });
    });

    let Some(crate::ir::Stmt::StoreLocal { dst, .. }) = ir.body.first() else {
        panic!("expected a coop StoreLocal as the first statement");
    };
    assert_eq!(
        dst.element,
        ElementType::coop_matrix(ScalarElement::F32, CoopMatrixRole::C, 8, 8)
    );
}

#[test]
fn cooperative_load_store_layout_flags_use_transposed_internal_layout() {
    fn collect_coop_store_row_major(block: &naga::Block, out: &mut Vec<bool>) {
        for stmt in block.iter() {
            match stmt {
                naga::Statement::CooperativeStore { data, .. } => out.push(data.row_major),
                naga::Statement::Block(inner) => collect_coop_store_row_major(inner, out),
                naga::Statement::If { accept, reject, .. } => {
                    collect_coop_store_row_major(accept, out);
                    collect_coop_store_row_major(reject, out);
                }
                naga::Statement::Switch { cases, .. } => {
                    for case in cases {
                        collect_coop_store_row_major(&case.body, out);
                    }
                }
                naga::Statement::Loop {
                    body, continuing, ..
                } => {
                    collect_coop_store_row_major(body, out);
                    collect_coop_store_row_major(continuing, out);
                }
                _ => {}
            }
        }
    }

    fn lowered_coop_layout_flags(output_layout: Layout) -> (Vec<bool>, Vec<bool>) {
        let ir = tile::build(|phase| {
            let coop = crate::CoopMatrixToken::new_unchecked();
            let y = phase.storage_write_with_layout_offset(f32(), output_layout, 0);
            let a_tile = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
            let b_tile = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
            phase.program_grid(32, [1, 1, 1], |program| {
                let acc = coop.alloc_coop_acc(program, ScalarElement::F32, 8, 8);
                let zero = coop.coop_zero(program, ScalarElement::F32, 8, 8);
                coop.coop_store_local(program, &acc, zero);

                let a = coop.coop_load_a(program, &a_tile, 0u32, 0u32, ScalarElement::F32, 8, 8);
                let b = coop.coop_load_b(program, &b_tile, 0u32, 0u32, ScalarElement::F32, 8, 8);
                let c = coop.coop_load_local(program, &acc);
                let mma = coop.coop_mma(program, a, b, c);
                coop.coop_store_local(program, &acc, mma);
                coop.coop_store(program, &acc, &y, 0u32, 0u32);
            });
        });

        let lowered = lower_or_fail(&ir, "coop layout flags");
        let function = &lowered.module().entry_points[0].function;
        let load_flags = function
            .expressions
            .iter()
            .filter_map(|(_, expr)| match expr {
                naga::Expression::CooperativeLoad { data, .. } => Some(data.row_major),
                _ => None,
            })
            .collect();
        let mut store_flags = Vec::new();
        collect_coop_store_row_major(&function.body, &mut store_flags);
        (load_flags, store_flags)
    }

    let row_major = Layout::contiguous(MemoryLevel::Storage, Shape::new([8, 8]));
    let col_major = Layout::strided(MemoryLevel::Storage, Shape::new([8, 8]), &[1, 8]);

    let (loads, stores) = lowered_coop_layout_flags(row_major);
    assert_eq!(loads, [false, false]);
    assert_eq!(stores, [false]);

    let (loads, stores) = lowered_coop_layout_flags(col_major);
    assert_eq!(loads, [false, false]);
    assert_eq!(stores, [true]);
}

#[test]
fn general_group_reduce_lowers_without_subgroup_intrinsics() {
    // A general combine cannot use subgroup collectives (they are
    // per-operator), so `group_reduce_with` stages through workgroup memory.
    // It must therefore lower on devices with no subgroup support at all.
    let ir = tile::build(|phase| {
        let x = phase.storage_read(f32(), Shape::new([64]));
        let y = phase.storage_write(f32(), Shape::new([64]));
        phase.program_grid(64, [1, 1, 1], |program| {
            let lane = program.lane();
            let mask = lane.clone().lt(64u32);
            let value = program.load(x.at(lane.clone()), mask.clone(), 0.0);
            // An arbitrary associative body, not one of the closed ops.
            let combined = program.group_reduce_with(16, value, |program, a, b| {
                let scaled = a.clone() * b.clone();
                program.bind(a + b + scaled)
            });
            program.store(y.at(lane), combined, mask);
        });
    });

    let lowered = lower_or_fail(&ir, "general group reduce");
    assert_eq!(
        lowered.wgsl_extension_prelude(),
        "",
        "a general combine must not require the subgroups extension"
    );
}

#[test]
fn joint_carrier_group_reduce_stages_every_slot() {
    // A two-slot carrier whose second slot reads the first on both sides —
    // the shape that makes online softmax a carrier rather than two
    // independent reductions. Each slot needs its own staging array, so the
    // lowered program must hold two block-sized workgroup allocations.
    let ir = tile::build(|phase| {
        let x = phase.storage_read(f32(), Shape::new([64]));
        let y = phase.storage_write(f32(), Shape::new([64]));
        phase.program_grid(64, [1, 1, 1], |program| {
            let lane = program.lane();
            let mask = lane.clone().lt(64u32);
            let value = program.load(x.at(lane.clone()), mask.clone(), 0.0);
            let one = program.bind(value.clone() * 0.0 + 1.0);
            let combined = program.group_reduce_with_vec(
                16,
                vec![value, one],
                |program, acc, incoming| {
                    let mut acc = acc.into_iter();
                    let (m, l) = (acc.next().unwrap(), acc.next().unwrap());
                    let mut incoming = incoming.into_iter();
                    let (m2, l2) = (incoming.next().unwrap(), incoming.next().unwrap());
                    let joined = program.bind(m.clone().max(m2.clone()));
                    let scaled = l * (m - joined.clone()).exp() + l2 * (m2 - joined.clone()).exp();
                    vec![joined, program.bind(scaled)]
                },
            );
            let mut combined = combined.into_iter();
            let (m, l) = (combined.next().unwrap(), combined.next().unwrap());
            program.store(y.at(lane), m + l.log(), mask);
        });
    });

    let lowered = lower_or_fail(&ir, "joint carrier group reduce");
    assert_eq!(
        lowered.wgsl_extension_prelude(),
        "",
        "a joint carrier must not require the subgroups extension"
    );
}
