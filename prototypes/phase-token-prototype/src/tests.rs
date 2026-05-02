use super::*;
use crate::tile::Tile;

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

fn tile_stmts_contain_load_role(stmts: &[TileStmt], role: CoopOperandRole) -> bool {
    stmts.iter().any(|stmt| match stmt {
        TileStmt::LoadCoop { role: r, .. } => *r == role,
        TileStmt::WhileTrue { body, .. } => tile_stmts_contain_load_role(body, role),
        _ => false,
    })
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
fn coop_qmatmul_q8_0_lowers_through_subgroup_dsl() {
    // Exercises the cooperative-matrix DSL: workgroup tile copies, coop MMA,
    // structured K loop, and cooperative store. Picks the BM=BN=64, BK=32
    // shape that the deleted accelerator's `coop8` path used.
    const M: u32 = 64;
    const K: u32 = 64;
    const N: u32 = 64;
    let ir = tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([M, K]));
        let b = phase.quantized_matrix(GgmlQuantFormat::Q8_0, K, N);
        let y = phase.storage_write::<F32, 2>(Shape::new([M, N]));
        phase.qmatmul::<64, 64, 32>(&a, &b, &y, 4);
    });

    let [Op::TileProgram(program)] = ir.body().ops() else {
        panic!("expected one tile program");
    };
    assert_eq!(program.block, 128, "BM*BN=64*64 → 4 subgroups → 128 lanes");
    assert!(
        program
            .body
            .iter()
            .any(|stmt| !matches!(stmt, TileStmt::Store(_))),
        "coop qmatmul must emit a subgroup-collective body"
    );
    assert!(
        program
            .body
            .iter()
            .all(|stmt| !matches!(stmt, TileStmt::Store(_))),
        "coop qmatmul stores via cooperative_store, not per-lane stores"
    );
    assert!(
        tile_stmts_contain_load_role(&program.body, CoopOperandRole::A)
            && tile_stmts_contain_load_role(&program.body, CoopOperandRole::B),
        "coop qmatmul should express A and B fragment roles on LoadCoop statements"
    );
    assert!(
        !ir.coop_accs().is_empty(),
        "coop qmatmul must declare cooperative-matrix accumulators"
    );
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("coop qmatmul lowering failed: {error}"));
}

#[test]
fn primitive_qgemm_q8_0_in_dsl() {
    // Scalar qgemm expressed in primitives — one output cell per subgroup.
    // The K-axis dot product is tiled with the same vectorized dequant + dot4
    // primitives the qgemv kernel uses. Each subgroup's lanes stride along K
    // by VALUES_PER_LANE; the partial sum is collapsed with subgroup_reduce_sum.
    //
    // Throughput is lower than the cooperative-matrix accelerator path
    // (1 cell/subgroup vs. an 8x8 fragment's 64 cells), but every operation is
    // user-visible and there is no hidden TileProgramAccelerator.
    const VALUES_PER_LANE: u32 = 8;
    const SUBGROUP_SIZE: u32 = 32;
    const SUBGROUPS_X: u32 = 4;
    const SUBGROUPS_Y: u32 = 1;
    const SUBGROUPS: u32 = SUBGROUPS_X * SUBGROUPS_Y;
    const BLOCK: usize = (SUBGROUPS * SUBGROUP_SIZE) as usize;
    const COLS_PER_WORKGROUP: u32 = SUBGROUPS_X;
    const ROWS_PER_WORKGROUP: u32 = SUBGROUPS_Y;
    const K_PER_ITER: u32 = SUBGROUP_SIZE * VALUES_PER_LANE;
    const M: u32 = 16;
    const K: u32 = 1024;
    const N: u32 = 64;
    const K_ITERATIONS: u32 = K / K_PER_ITER;

    let ir = tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([M, K]));
        let b = phase.quantized_matrix(GgmlQuantFormat::Q8_0, K, N);
        let y = phase.storage_write::<F32, 2>(Shape::new([M, N]));
        phase.program_grid::<BLOCK>(
            [N / COLS_PER_WORKGROUP, M / ROWS_PER_WORKGROUP, 1],
            |program| {
                let sg = program.subgroup_id();
                let row_in_wg = sg.clone() / SUBGROUPS_X;
                let col_in_wg = sg % SUBGROUPS_X;
                let row = program.program_id(WorkgroupAxis::Y) * ROWS_PER_WORKGROUP + row_in_wg;
                let col = program.program_id(WorkgroupAxis::X) * COLS_PER_WORKGROUP + col_in_wg;
                let lane = program.subgroup_lane();
                let k_base = program.loop_index() * K_PER_ITER + lane.clone() * VALUES_PER_LANE;
                let mask = row.lt(M).and(col.lt(N)).and(k_base.lt(K));

                let bs = program.load_quantized_block::<8>(&b, &k_base, &col, mask.clone(), 0.0);
                let a0 = program.load(a.at(&row, k_base.clone() + 0u32), mask.clone(), 0.0);
                let a1 = program.load(a.at(&row, k_base.clone() + 1u32), mask.clone(), 0.0);
                let a2 = program.load(a.at(&row, k_base.clone() + 2u32), mask.clone(), 0.0);
                let a3 = program.load(a.at(&row, k_base.clone() + 3u32), mask.clone(), 0.0);
                let a4 = program.load(a.at(&row, k_base.clone() + 4u32), mask.clone(), 0.0);
                let a5 = program.load(a.at(&row, k_base.clone() + 5u32), mask.clone(), 0.0);
                let a6 = program.load(a.at(&row, k_base.clone() + 6u32), mask.clone(), 0.0);
                let a7 = program.load(a.at(&row, k_base.clone() + 7u32), mask.clone(), 0.0);

                let [b0, b1, b2, b3, b4, b5, b6, b7] = bs;
                let dot_lo = program.dot4([a0, a1, a2, a3], [b0, b1, b2, b3]);
                let dot_hi = program.dot4([a4, a5, a6, a7], [b4, b5, b6, b7]);
                let body = dot_lo + dot_hi;

                let partial = program.loop_fold(
                    TileReduceOp::Sum,
                    K_ITERATIONS,
                    body,
                    TileLiteral::F32(F32Bits::new(0.0)),
                );
                let sum = program.subgroup_reduce_sum(partial);
                let store_mask = lane.eq(0).and(row.lt(M)).and(col.lt(N));
                program.store(y.at(row, col), sum, store_mask);
            },
        );
    });

    assert_only_tile_programs(&ir);
    assert!(
        ir.tiles().is_empty(),
        "scalar primitive qgemm should not need workgroup scratch"
    );
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("primitive qgemm lowering failed: {error}"));
}

#[test]
fn primitive_qgemv_q8_0_matches_accelerator_structure() {
    // Express the Q8_0 qgemv kernel using only DSL primitives:
    //   * subgroup_id / subgroup_lane         (column-group + K-stride lanes)
    //   * load_quantized_block::<8>           (vectorized 8-wide dequant w/ shared scale)
    //   * loaded scalars composed through dot4 (vec4 fused multiply-add)
    //   * loop_fold over K                    (per-lane K accumulator)
    //   * subgroup_reduce_sum                 (cross-lane reduction, no scratch tile)
    //
    // Same instruction-level building blocks the accelerator emits — the only
    // structural difference is each subgroup owns one column instead of four.
    const VALUES_PER_LANE: u32 = 8;
    const SUBGROUP_SIZE: u32 = 32;
    const SUBGROUPS: u32 = 4;
    const BLOCK: usize = (SUBGROUPS * SUBGROUP_SIZE) as usize;
    const COLS_PER_WORKGROUP: u32 = SUBGROUPS;
    const K_PER_ITER: u32 = SUBGROUP_SIZE * VALUES_PER_LANE;
    const K: u32 = 1024;
    const N: u32 = 64;
    const K_ITERATIONS: u32 = K / K_PER_ITER;

    let ir = tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([1, K]));
        let b = phase.quantized_matrix(GgmlQuantFormat::Q8_0, K, N);
        let y = phase.storage_write::<F32, 2>(Shape::new([1, N]));
        phase.program_grid::<BLOCK>([N / COLS_PER_WORKGROUP, 1, 1], |program| {
            let col_group_base = program.program_id(WorkgroupAxis::X) * COLS_PER_WORKGROUP;
            let col = col_group_base + program.subgroup_id();
            let lane = program.subgroup_lane();
            let k_base = program.loop_index() * K_PER_ITER + lane.clone() * VALUES_PER_LANE;
            let mask = col.lt(N).and(k_base.lt(K));

            let bs = program.load_quantized_block::<8>(&b, &k_base, &col, mask.clone(), 0.0);
            let a0 = program.load(a.at(0, k_base.clone() + 0u32), mask.clone(), 0.0);
            let a1 = program.load(a.at(0, k_base.clone() + 1u32), mask.clone(), 0.0);
            let a2 = program.load(a.at(0, k_base.clone() + 2u32), mask.clone(), 0.0);
            let a3 = program.load(a.at(0, k_base.clone() + 3u32), mask.clone(), 0.0);
            let a4 = program.load(a.at(0, k_base.clone() + 4u32), mask.clone(), 0.0);
            let a5 = program.load(a.at(0, k_base.clone() + 5u32), mask.clone(), 0.0);
            let a6 = program.load(a.at(0, k_base.clone() + 6u32), mask.clone(), 0.0);
            let a7 = program.load(a.at(0, k_base.clone() + 7u32), mask.clone(), 0.0);

            let [b0, b1, b2, b3, b4, b5, b6, b7] = bs;
            let dot_lo = program.dot4([a0, a1, a2, a3], [b0, b1, b2, b3]);
            let dot_hi = program.dot4([a4, a5, a6, a7], [b4, b5, b6, b7]);
            let body = dot_lo + dot_hi;

            let partial = program.loop_fold(
                TileReduceOp::Sum,
                K_ITERATIONS,
                body,
                TileLiteral::F32(F32Bits::new(0.0)),
            );
            let sum = program.subgroup_reduce_sum(partial);
            let store_mask = lane.eq(0).and(col.lt(N));
            program.store(y.at(0, col), sum, store_mask);
        });
    });

    assert_only_tile_programs(&ir);
    assert!(
        ir.tiles().is_empty(),
        "primitive qgemv should not need workgroup scratch"
    );
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("primitive qgemv lowering failed: {error}"));
}

#[test]
fn quantized_block_lane_primitive_lowers_to_naga() {
    // Q8_0 dequant primitive — 8 dequantized values per call sharing the
    // block scale lookup. Mirrors `dequantize_q8_0_values8` in the accelerator
    // path, but invoked from user DSL code.
    const K: u32 = 256;
    const N: u32 = 8;
    const SUBGROUPS: u32 = 4;
    const SUBGROUP_SIZE: u32 = 32;
    const BLOCK: usize = (SUBGROUPS * SUBGROUP_SIZE) as usize;

    let ir = tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([1, K]));
        let b = phase.quantized_matrix(GgmlQuantFormat::Q8_0, K, N);
        let y = phase.storage_write::<F32, 2>(Shape::new([1, N]));
        phase.program_grid::<BLOCK>([N / SUBGROUPS, 1, 1], |program| {
            let col = program.program_id(WorkgroupAxis::X) * SUBGROUPS + program.subgroup_id();
            let lane = program.subgroup_lane();
            let k_iterations = K / (SUBGROUP_SIZE * 8);
            let k_base = program.loop_index() * (SUBGROUP_SIZE * 8) + lane.clone() * 8;
            let mask = col.lt(N).and(k_base.lt(K));
            let bs = program.load_quantized_block::<8>(&b, &k_base, &col, mask.clone(), 0.0);
            let mut acc: Option<Tile<BLOCK>> = None;
            for (i, b_val) in bs.into_iter().enumerate() {
                let a_idx = k_base.clone() + i as u32;
                let a_val = program.load(a.at(0, a_idx), mask.clone(), 0.0);
                let term = a_val * b_val;
                acc = Some(match acc {
                    Some(prev) => prev + term,
                    None => term,
                });
            }
            let body = acc.expect("at least one lane");
            let partial = program.loop_fold(
                TileReduceOp::Sum,
                k_iterations,
                body,
                TileLiteral::F32(F32Bits::new(0.0)),
            );
            let sum = program.subgroup_reduce_sum(partial);
            let store_mask = lane.eq(0).and(col.lt(N));
            program.store(y.at(0, col), sum, store_mask);
        });
    });

    assert_only_tile_programs(&ir);
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("quantized block lane lowering failed: {error}"));
}

#[test]
fn subgroup_index_primitives_lower_to_naga() {
    // Each subgroup owns one column; subgroup_lane strides over K. Mirrors the
    // qgemv accelerator's subgroup partitioning, expressed in primitives only.
    const K: u32 = 256;
    const N: u32 = 64;
    const SUBGROUPS: u32 = 4; // workgroup = 4 subgroups
    const SUBGROUP_SIZE: u32 = 32;
    const BLOCK: usize = (SUBGROUPS * SUBGROUP_SIZE) as usize;

    let ir = tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([1, K]));
        let b = phase.quantized_matrix(GgmlQuantFormat::Q8_0, K, N);
        let y = phase.storage_write::<F32, 2>(Shape::new([1, N]));
        phase.program_grid::<BLOCK>([N / SUBGROUPS, 1, 1], |program| {
            let col = program.program_id(WorkgroupAxis::X) * SUBGROUPS + program.subgroup_id();
            let lane = program.subgroup_lane();
            let k_iterations = K / SUBGROUP_SIZE;
            let k_index = program.loop_index() * SUBGROUP_SIZE + lane.clone();
            let mask = col.lt(N).and(k_index.lt(K));
            let a_value = program.load(a.at(0, &k_index), mask.clone(), 0.0);
            let b_value = program.load_quantized(&b, &k_index, &col, mask.clone(), 0.0);
            let partial = program.loop_fold(
                TileReduceOp::Sum,
                k_iterations,
                a_value * b_value,
                TileLiteral::F32(F32Bits::new(0.0)),
            );
            let sum = program.subgroup_reduce_sum(partial);
            let store_mask = lane.eq(0).and(col.lt(N));
            program.store(y.at(0, col), sum, store_mask);
        });
    });

    assert_only_tile_programs(&ir);
    assert!(
        ir.tiles().is_empty(),
        "subgroup-partitioned qgemv should not allocate scratch tiles"
    );
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("subgroup-partitioned qgemv lowering failed: {error}"));
}

#[test]
fn subgroup_reduce_qgemv_primitive_lowers_to_naga() {
    // A non-cheating qgemv: every piece is a primitive on `TileBlock`, with the
    // K-axis reduction produced by `subgroup_reduce_sum` instead of the
    // workgroup-scratch tree used by `group_reduce_sum`. One workgroup ==
    // one subgroup == one output column; lanes stride along K.
    const K: u32 = 256;
    const N: u32 = 64;
    const BLOCK: usize = 32; // subgroup size

    let ir = tile::build(|phase| {
        let a = phase.storage_read::<F32, 2>(Shape::new([1, K]));
        let b = phase.quantized_matrix(GgmlQuantFormat::Q8_0, K, N);
        let y = phase.storage_write::<F32, 2>(Shape::new([1, N]));
        let k_iterations = K / BLOCK as u32;
        phase.program_grid::<BLOCK>([N, 1, 1], |program| {
            let col = program.program_id(WorkgroupAxis::X);
            let k_lane = program.arange();
            let k_index = program.loop_index() * BLOCK as u32 + k_lane.clone();
            let mask = col.lt(N).and(k_index.lt(K));
            let a_value = program.load(a.at(0, &k_index), mask.clone(), 0.0);
            let b_value = program.load_quantized(&b, &k_index, &col, mask.clone(), 0.0);
            let partial = program.loop_fold(
                TileReduceOp::Sum,
                k_iterations,
                a_value * b_value,
                TileLiteral::F32(F32Bits::new(0.0)),
            );
            let sum = program.subgroup_reduce_sum(partial);
            let store_mask = k_lane.eq(0).and(col.lt(N));
            program.store(y.at(0, col), sum, store_mask);
        });
    });

    assert_only_tile_programs(&ir);

    // The body should not allocate a workgroup-scratch tile for the reduction.
    assert!(
        ir.tiles().is_empty(),
        "subgroup reduce should not allocate a workgroup scratch tile"
    );

    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("primitive subgroup-reduce qgemv lowering failed: {error}"));
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
    let [TileStmt::Store(store)] = program.body.as_slice() else {
        panic!("qdequantize should emit one tile store");
    };
    assert!(matches!(store.value, TileExpr::QuantizedLoad(_)));
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("large Q4K qdequantize lowering failed: {error}"));
}
