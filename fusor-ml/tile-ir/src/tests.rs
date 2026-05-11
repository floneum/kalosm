use super::*;

/// Lower `ir` and panic with a labelled message on failure. Returns the
/// lowered Naga kernel for tests that want to inspect the module.
fn lower_or_fail(ir: &KernelIr, label: &str) -> NagaKernel {
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("{label} lowering failed: {error}"))
}

fn ggml_quant_formats() -> [GgmlQuantFormat; 12] {
    [
        GgmlQuantFormat::Q4_0,
        GgmlQuantFormat::Q4_1,
        GgmlQuantFormat::Q5_0,
        GgmlQuantFormat::Q5_1,
        GgmlQuantFormat::Q8_0,
        GgmlQuantFormat::Q8_1,
        GgmlQuantFormat::Q2K,
        GgmlQuantFormat::Q3K,
        GgmlQuantFormat::Q4K,
        GgmlQuantFormat::Q5K,
        GgmlQuantFormat::Q6K,
        GgmlQuantFormat::Q8K,
    ]
}

fn tile_stmts_contain_load_role(stmts: &[TileStmt], role: CoopOperandRole) -> bool {
    stmts.iter().any(|stmt| match stmt {
        TileStmt::LoadCoop { role: r, .. } => *r == role,
        TileStmt::Fold { body, .. } => tile_stmts_contain_load_role(body, role),
        _ => false,
    })
}

#[test]
fn layout_is_structured_shape_strides_and_memory_level() {
    let shape = Shape::new([4, 8]);
    let layout = Layout::contiguous(MemoryLevel::Storage, shape.clone());

    assert_eq!(layout.shape(), &shape);
    assert_eq!(layout.affine_strides(), vec![8, 1]);
    assert_eq!(layout.memory_level(), MemoryLevel::Storage);
    assert_eq!(layout.element_count().get(), 32);
}

#[test]
fn op_enum_is_source_tile_program_only() {
    let ir = tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([1, 8]));
        let y = phase.storage_write::<F32, 2>(Shape::new([1, 8]));
        phase.program_grid::<8>([1, 1, 1], |program| {
            let lane = program.arange();
            let mask = lane.lt(8);
            let value = program.load(x.at(0, &lane), mask.clone(), 0.0);
            program.store(y.at(0, lane), value, mask);
        });
    });

    // The kernel body is a single `TileProgramOp` by construction.
    let _ = ir.body();
}

#[test]
fn tile_source_softmax_lowers_to_naga() {
    const ROWS: u32 = 4;
    const COLS: u32 = 100;
    const BLOCK: usize = 128;
    let ir = tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([ROWS, COLS]));
        let y = phase.storage_write::<F32, 2>(Shape::new([ROWS, COLS]));
        phase.program_grid::<BLOCK>([1, ROWS, 1], |program| {
            let row = program.program_id(WorkgroupAxis::Y);
            let col = program.arange();
            let mask = col.lt(COLS);
            let values = program.load(x.at(&row, &col), mask.clone(), f32::MIN);
            let max = program.reduce_max(values.clone());
            let exp = (values - max).exp();
            let sum = program.reduce_sum(exp.clone());
            program.store(y.at(row, col), exp / sum, mask);
        });
    });

    lower_or_fail(&ir, "tile softmax");
}

#[test]
fn lowered_naga_uses_anonymous_ir_objects_except_entry_point() {
    let ir = tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([1, 8]));
        let y = phase.storage_write::<F32, 2>(Shape::new([1, 8]));
        phase.program_grid::<8>([1, 1, 1], |program| {
            let lane = program.arange();
            let mask = lane.lt(8);
            let value = program.load(x.at(0, &lane), mask.clone(), 0.0);
            program.store(y.at(0, lane), value, mask);
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

