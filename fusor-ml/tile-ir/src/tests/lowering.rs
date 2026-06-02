use super::*;
use crate::ScalarElement;

fn f32() -> ElementType {
    ScalarElement::F32.element()
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

    let _ = ir.body();
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
            let values = program.load(x.at((&row, &col)), mask.clone(), f32::MIN);
            let max = program.reduce_max(values.clone());
            let exp = (values - max).exp();
            let sum = program.reduce_sum(exp.clone());
            program.store(y.at((row, col)), exp / sum, mask);
        });
    });

    lower_or_fail(&ir, "tile softmax");
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
        phase.program_grid(32, [1, 1, 1], |program| {
            let acc = program.alloc_coop_acc(ScalarElement::F32, 8, 8);
            let zero = program.coop_zero(ScalarElement::F32, 8, 8);
            program.store_local_coop(&acc, zero);
        });
    });

    let Some(crate::Stmt::StoreLocal { dst, .. }) = ir.body().first() else {
        panic!("expected a coop StoreLocal as the first statement");
    };
    assert_eq!(
        dst.element,
        ElementType::coop_matrix(ScalarElement::F32, CoopMatrixRole::C, 8, 8)
    );
}
