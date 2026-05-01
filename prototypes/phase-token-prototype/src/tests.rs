use super::*;

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

fn assert_only_tile_programs(ir: &KernelIr) {
    assert!(
        ir.body()
            .ops()
            .iter()
            .all(|op| matches!(op, Op::TileProgram(_))),
        "source IR should contain only tile programs"
    );
}

#[test]
fn layout_is_structured_shape_strides_and_memory_level() {
    let shape = Shape::new([4, 8]);
    let layout = Layout::contiguous(MemoryLevel::Storage, shape.clone());

    assert_eq!(layout.shape(), &shape);
    assert_eq!(layout.strides().values(), &[8, 1]);
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

    let [Op::TileProgram(_)] = ir.body().ops() else {
        panic!("expected exactly one tile program");
    };
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

    assert_only_tile_programs(&ir);
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("tile softmax lowering failed: {error}"));
}

#[test]
fn tile_source_dense_matmul_lowers_to_naga() {
    let ir = tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([8, 33]));
        let b = phase.storage_read::<F32, 2>(Shape::new([33, 5]));
        let y = phase.storage_write::<F32, 2>(Shape::new([8, 5]));
        phase.matmul::<64>(&a, &b, &y);
    });

    assert_only_tile_programs(&ir);
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("dense matmul lowering failed: {error}"));
}

#[test]
fn tile_source_qmatmul_and_qgemv_lower_all_supported_ggml_formats() {
    for format in ggml_quant_formats() {
        let k = format.block_elements();
        let ir = tile::build(|phase| {
            let a = phase.storage_read::<F32, 2>(Shape::new([3, k]));
            let b = phase.quantized_matrix(format, k, 7);
            let y = phase.storage_write::<F32, 2>(Shape::new([3, 7]));
            phase.qmatmul::<8, 4, 8>(&a, &b, &y, 4);
        });
        assert_only_tile_programs(&ir);
        ir.lower_to_naga()
            .unwrap_or_else(|error| panic!("{format:?} tile qmatmul lowering failed: {error}"));

        let ir = tile::build(|phase| {
            let a = phase.storage_read::<F32, 2>(Shape::new([1, k]));
            let b = phase.quantized_matrix(format, k, 7);
            let y = phase.storage_write::<F32, 2>(Shape::new([1, 7]));
            phase.qgemv::<4, 64>(&a, &b, &y, 4, 3);
        });
        assert_only_tile_programs(&ir);
        ir.lower_to_naga()
            .unwrap_or_else(|error| panic!("{format:?} tile qgemv lowering failed: {error}"));
    }
}

#[test]
fn qdequantize_lowers_large_embedding_table_as_tile_program() {
    let k = 3584;
    let n = 152064;
    let total = k * n;
    let ir = tile::build(|phase| {
        let b = phase.quantized_matrix(GgmlQuantFormat::Q4K, k, n);
        let y = phase.storage_write::<F32, 1>(Shape::new([total]));
        phase.qdequantize(&b, &y, 65_535);
    });

    let [Op::TileProgram(program)] = ir.body().ops() else {
        panic!("qdequantize should expand to a tile program");
    };
    assert_eq!(program.block, 256);
    let [store] = program.stores.as_slice() else {
        panic!("qdequantize should emit one tile store");
    };
    assert!(matches!(store.value, TileExpr::QuantizedLoad(_)));
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("large Q4K qdequantize lowering failed: {error}"));
}
