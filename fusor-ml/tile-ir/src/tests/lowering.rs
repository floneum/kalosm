use super::*;

#[test]
fn op_enum_is_source_tile_program_only() {
    let ir = tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([1, 8]));
        let y = phase.storage_write::<F32, 2>(Shape::new([1, 8]));
        phase.program_grid::<8>([1, 1, 1], |program| {
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
    const BLOCK: usize = 128;
    let ir = tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([ROWS, COLS]));
        let y = phase.storage_write::<F32, 2>(Shape::new([ROWS, COLS]));
        phase.program_grid::<BLOCK>([1, ROWS, 1], |program| {
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
fn lowered_naga_uses_anonymous_ir_objects_except_entry_point() {
    let ir = tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([1, 8]));
        let y = phase.storage_write::<F32, 2>(Shape::new([1, 8]));
        phase.program_grid::<8>([1, 1, 1], |program| {
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
        let x = phase.storage_read::<Vector<F32, 2>, 1>(Shape::new([16]));
        let y = phase.storage_write::<F32, 1>(Shape::new([16]));
        phase.program_grid::<16>([1, 1, 1], |program| {
            let lane = program.lane();
            let mask = lane.clone().lt(16);
            let value = program.load_vector::<F32, 2>(
                x.at(lane.clone()),
                mask.clone(),
                TileLiteral::f32(0.0),
            );
            let dot = program.vector_dot::<F32, 2>(value.clone(), value);
            program.store_linear(y.at(lane), dot, mask);
        });
    });

    lower_or_fail(&ir, "generic vec2 dot");
}

#[test]
fn typed_coop_accumulator_records_scalar_role_and_shape() {
    let ir = tile::build(|phase| {
        phase.program_grid::<32>([1, 1, 1], |program| {
            let acc = program.alloc_coop_acc_typed::<F32, 8, 8>();
            program.zero_coop_acc(&acc);
        });
    });

    assert_eq!(
        ir.locals()[0].element,
        ElementType::coop_matrix(ScalarElement::F32, CoopMatrixRole::C, 8, 8)
    );
}
