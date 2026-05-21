use fusor_tile_ir::{
    tile, F32Bits, GgmlQuantFormat, KernelBuilder, KernelTensorRef, Layout, MemoryLevel,
    NagaKernel, Shape, F16, F32,
};
use fusor_tile_ir_kernels::{
    batched_gemv_with_epilogues, batched_matmul_with_epilogues, flash_attention,
    linear_storage_layout, qdequantize, qgemv_q4k_paired, qgemv_with_epilogue,
    qgemv_workgroup_with_epilogue, qmatmul_with_epilogue, qmatmul_workgroup_with_epilogues,
    quantized_matrix, rms_norm_vec4, try_batched_coop_matmul, DenseMatmulEpilogues,
    DenseMatmulShape, FlashAttentionDims, FlashAttentionMeta, PairedEpilogue, Q4KPairedGgml,
    QmatmulEpilogues, RmsNormVec4, RmsNormVec4Meta, TensorMeta, UnaryEpilogue,
};

fn lower_or_fail(ir: &fusor_tile_ir::KernelIr, label: &str) -> NagaKernel {
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("{label} lowering failed: {error}"))
}

#[test]
fn streaming_flash_attention_regression_shape_lowers_to_naga() {
    let layout = linear_storage_layout();
    let mut kb = KernelBuilder::<()>::new();
    flash_attention::<F32, 32, ()>(
        &mut kb,
        KernelTensorRef::new((), layout.clone()),
        KernelTensorRef::new((), layout.clone()),
        KernelTensorRef::new((), layout.clone()),
        None,
        KernelTensorRef::new((), layout),
        FlashAttentionMeta {
            dims: FlashAttentionDims {
                batch: 1,
                num_heads: 32,
                num_kv_heads: 8,
                q_seq_len: 48,
                kv_seq_len: 48,
                head_dim: 128,
            },
            scale: F32Bits::new(1.0 / 128.0f32.sqrt()),
            q_meta: TensorMeta::new(vec![196_608, 6_144, 128, 1], 0),
            k_meta: TensorMeta::new(vec![49_152, 6_144, 128, 1], 0),
            v_meta: TensorMeta::new(vec![49_152, 6_144, 128, 1], 0),
            mask_meta: None,
            output_meta: TensorMeta::new(vec![196_608, 6_144, 128, 1], 0),
            dispatch_size: [16, 1536, 1],
        },
    )
    .expect("streaming flash attention should build");
    let (ir, _) = kb.finish();

    lower_or_fail(&ir, "streaming flash attention");
}

#[test]
fn rms_norm_vec4_minimal_lowers() {
    let layout = Layout::strided(MemoryLevel::Storage, Shape::new([1]), &[1]);
    let mut kb = KernelBuilder::<()>::new();
    let input = KernelTensorRef::with_offset((), layout.clone(), 0);
    let weight = KernelTensorRef::with_offset((), layout.clone(), 0);
    let output = KernelTensorRef::with_offset((), layout.clone(), 0);
    let meta = RmsNormVec4Meta {
        cols: 4,
        cols_vec: 1,
        eps: F32Bits::new(1e-5),
        input_offset_vec: 0,
        input_row_stride_vec: 1,
        residual_offset_vec: None,
        residual_row_stride_vec: 0,
        weight_offset_vec: 0,
        bias_offset_vec: None,
        output_offset_vec: 0,
        output_row_stride_vec: 1,
    };
    rms_norm_vec4(
        &mut kb,
        RmsNormVec4 {
            input,
            residual: None,
            weight,
            bias: None,
            output,
            meta,
            rows: 1,
        },
    )
    .unwrap();
    let (ir, _) = kb.finish();
    lower_or_fail(&ir, "rms_norm_vec4");
}

fn qgemv_ir(format: GgmlQuantFormat, rows: u32, cols: u32) -> fusor_tile_ir::KernelIr {
    tile::build(|program| {
        let a = program.storage_read::<F32, 2>(Shape::new([1, rows]));
        let b = quantized_matrix(program, format, rows, cols);
        let y = program.storage_write::<F32, 2>(Shape::new([1, cols]));
        qgemv_with_epilogue::<4, 64>(program, &a, &b, &y, 1, Option::<&UnaryEpilogue>::None);
    })
}

#[test]
fn generic_q8_qgemv_lowers() {
    let ir = qgemv_ir(GgmlQuantFormat::Q8_0, 256, 1024);
    lower_or_fail(&ir, "q8_0 qgemv");
}

#[test]
fn q4k_ggml_qgemv_lowers() {
    let ir = qgemv_ir(GgmlQuantFormat::Q4K, 4096, 8192);
    lower_or_fail(&ir, "q4k ggml qgemv");
}

#[test]
fn q6k_ggml_qgemv_lowers() {
    let ir = qgemv_ir(GgmlQuantFormat::Q6K, 4096, 8192);
    lower_or_fail(&ir, "q6k ggml qgemv");
}

#[test]
fn q4k_paired_epilogue_lowers() {
    let ir = tile::build(|program| {
        let rows = 4096;
        let pair_cols = 4096;
        let a = program.storage_read::<F32, 2>(Shape::new([1, rows]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q4K, rows, pair_cols * 2);
        let y = program.storage_write::<F32, 2>(Shape::new([1, pair_cols]));
        let epilogue =
            PairedEpilogue::with_extras("mul", 0, |tiles| tiles[0].clone() * tiles[1].clone());
        qgemv_q4k_paired(
            program,
            Q4KPairedGgml {
                a: &a,
                b: &b,
                y: &y,
                pair_cols,
                m_rows: 1,
                workgroups_x: 1,
                epilogue: &epilogue,
                extras: &[],
            },
        );
    });
    lower_or_fail(&ir, "q4k paired qgemv");
}

#[test]
fn scalar_qmatmul_lowers() {
    let ir = tile::build(|program| {
        let a = program.storage_read::<F32, 2>(Shape::new([8, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 16);
        let y = program.storage_write::<F32, 2>(Shape::new([8, 16]));
        qmatmul_with_epilogue::<8, 4, 8>(program, &a, &b, &y, 4, &QmatmulEpilogues::empty());
    });
    lower_or_fail(&ir, "scalar qmatmul");
}

#[test]
fn cooperative_qmatmul_lowers() {
    let ir = tile::build(|program| {
        let a = program.storage_read::<F32, 2>(Shape::new([64, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 64);
        let y = program.storage_write::<F32, 2>(Shape::new([64, 64]));
        qmatmul_with_epilogue::<64, 64, 32>(program, &a, &b, &y, 4, &QmatmulEpilogues::empty());
    });
    lower_or_fail(&ir, "cooperative qmatmul");
}

#[test]
fn batched_dense_f32_matmul_lowers() {
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 3,
            m: 8,
            k: 256,
            n: 4,
        };
        let a = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.m, shape.k]));
        let b = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.k, shape.n]));
        let y = program.storage_write::<F32, 2>(Shape::new([shape.batch * shape.m, shape.n]));
        batched_matmul_with_epilogues::<F32, 32, 32, 8>(
            program,
            &a,
            &b,
            &y,
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
        );
    });
    lower_or_fail(&ir, "batched dense f32 matmul");
}

#[test]
fn batched_dense_f32_gemv_lowers() {
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 3,
            m: 5,
            k: 256,
            n: 1,
        };
        let a = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.m, shape.k]));
        let b = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.k, shape.n]));
        let y = program.storage_write::<F32, 2>(Shape::new([shape.batch * shape.m, shape.n]));
        batched_gemv_with_epilogues::<F32, 4, 8, 128>(
            program,
            &a,
            &b,
            &y,
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
        );
    });
    lower_or_fail(&ir, "batched dense f32 gemv");
}

#[test]
fn batched_dense_f16_matmul_lowers() {
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 2,
            m: 8,
            k: 128,
            n: 4,
        };
        let a = program
            .storage_read::<fusor_tile_ir::F16, 2>(Shape::new([shape.batch * shape.m, shape.k]));
        let b = program
            .storage_read::<fusor_tile_ir::F16, 2>(Shape::new([shape.batch * shape.k, shape.n]));
        let y = program
            .storage_write::<fusor_tile_ir::F16, 2>(Shape::new([shape.batch * shape.m, shape.n]));
        batched_matmul_with_epilogues::<fusor_tile_ir::F16, 32, 32, 8>(
            program,
            &a,
            &b,
            &y,
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
        );
    });
    lower_or_fail(&ir, "batched dense f16 matmul");
}

#[test]
fn batched_dense_f16_gemv_lowers() {
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 2,
            m: 5,
            k: 128,
            n: 1,
        };
        let a = program
            .storage_read::<fusor_tile_ir::F16, 2>(Shape::new([shape.batch * shape.m, shape.k]));
        let b = program
            .storage_read::<fusor_tile_ir::F16, 2>(Shape::new([shape.batch * shape.k, shape.n]));
        let y = program
            .storage_write::<fusor_tile_ir::F16, 2>(Shape::new([shape.batch * shape.m, shape.n]));
        batched_gemv_with_epilogues::<fusor_tile_ir::F16, 4, 8, 128>(
            program,
            &a,
            &b,
            &y,
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
        );
    });
    lower_or_fail(&ir, "batched dense f16 gemv");
}

#[test]
fn cooperative_dense_f32_matmul_lowers() {
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 2,
            m: 64,
            k: 256,
            n: 64,
        };
        let a = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.m, shape.k]));
        let b = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.k, shape.n]));
        let y = program.storage_write::<F32, 2>(Shape::new([shape.batch * shape.m, shape.n]));
        assert!(try_batched_coop_matmul::<F32, 64, 64, 16>(
            program,
            &a,
            &b,
            &y,
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
        ));
    });
    lower_or_fail(&ir, "cooperative dense f32 matmul");
}

#[test]
fn cooperative_dense_f16_matmul_lowers() {
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 2,
            m: 64,
            k: 256,
            n: 64,
        };
        let a = program.storage_read::<F16, 2>(Shape::new([shape.batch * shape.m, shape.k]));
        let b = program.storage_read::<F16, 2>(Shape::new([shape.batch * shape.k, shape.n]));
        let y = program.storage_write::<F16, 2>(Shape::new([shape.batch * shape.m, shape.n]));
        assert!(try_batched_coop_matmul::<F16, 64, 64, 16>(
            program,
            &a,
            &b,
            &y,
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
        ));
    });
    lower_or_fail(&ir, "cooperative dense f16 matmul");
}

#[test]
fn cooperative_dense_f32_matmul_128x128_lowers() {
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 1,
            m: 128,
            k: 256,
            n: 128,
        };
        let a = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.m, shape.k]));
        let b = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.k, shape.n]));
        let y = program.storage_write::<F32, 2>(Shape::new([shape.batch * shape.m, shape.n]));
        assert!(try_batched_coop_matmul::<F32, 128, 128, 16>(
            program,
            &a,
            &b,
            &y,
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
        ));
    });
    lower_or_fail(&ir, "cooperative dense f32 128x128 matmul");
}

#[test]
fn cooperative_dense_f32_matmul_128x64_lowers() {
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 1,
            m: 128,
            k: 256,
            n: 64,
        };
        let a = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.m, shape.k]));
        let b = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.k, shape.n]));
        let y = program.storage_write::<F32, 2>(Shape::new([shape.batch * shape.m, shape.n]));
        assert!(try_batched_coop_matmul::<F32, 128, 64, 16>(
            program,
            &a,
            &b,
            &y,
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
        ));
    });
    lower_or_fail(&ir, "cooperative dense f32 128x64 matmul");
}

#[test]
fn cooperative_dense_f32_matmul_128x256_npass_lowers() {
    // Exercises the BK=16, N_PASSES=4 variant that mirrors coop_gemm.rs on
    // main: per-pass B/acc footprint with double-buffered K-pair iteration.
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 1,
            m: 128,
            k: 256,
            n: 256,
        };
        let a = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.m, shape.k]));
        let b = program.storage_read::<F32, 2>(Shape::new([shape.batch * shape.k, shape.n]));
        let y = program.storage_write::<F32, 2>(Shape::new([shape.batch * shape.m, shape.n]));
        assert!(try_batched_coop_matmul::<F32, 128, 256, 16>(
            program,
            &a,
            &b,
            &y,
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
        ));
    });
    lower_or_fail(&ir, "cooperative dense f32 128x256 N_PASSES=4 matmul");
}

#[test]
fn qdequantize_lowers() {
    let ir = tile::build(|program| {
        let b = quantized_matrix(program, GgmlQuantFormat::Q4K, 256, 4);
        let y = program.storage_write::<F32, 1>(Shape::new([1024]));
        qdequantize(program, &b, &y, 1);
    });
    lower_or_fail(&ir, "qdequantize");
}

/// Regression for the fallback branch in `qmatmul_tile_with_epilogue`. When
/// `BM*BN*BK != 256` (true for every caller in core's `quantized/matmul`),
/// the fallback used to drop the epilogue. With a non-identity `post`
/// epilogue (here: `tanh`), the lowered Naga module must contain a `Tanh`
/// math call somewhere in the function body.
fn qmatmul_epilogue_fallback_ir(post: Option<&UnaryEpilogue>) -> fusor_tile_ir::KernelIr {
    tile::build(|program| {
        // `m = 2` skips the `m == 1` qgemv branch.
        // BM*BN*BK = 64*64*32 = 131072 != 256 — forces the fallback path.
        let a = program.storage_read::<F32, 2>(Shape::new([2, 64]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 64, 64);
        let y = program.storage_write::<F32, 2>(Shape::new([2, 64]));
        let epilogues = QmatmulEpilogues {
            pre: None,
            pre_with_extras: None,
            pre_extra_inputs: &[],
            post,
            post_with_extras: None,
            post_extra_inputs: &[],
            post_acc_init_col_vector: None,
        };
        qmatmul_with_epilogue::<64, 64, 32>(program, &a, &b, &y, 4, &epilogues);
    })
}

fn module_uses_tanh(module: &naga::Module) -> bool {
    module.functions.iter().any(|(_, function)| {
        function.expressions.iter().any(|(_, expr)| {
            matches!(
                expr,
                naga::Expression::Math {
                    fun: naga::MathFunction::Tanh,
                    ..
                }
            )
        })
    }) || module.entry_points.iter().any(|entry| {
        entry.function.expressions.iter().any(|(_, expr)| {
            matches!(
                expr,
                naga::Expression::Math {
                    fun: naga::MathFunction::Tanh,
                    ..
                }
            )
        })
    })
}

#[test]
fn workgroup_qmatmul_lowers_without_subgroups() {
    let ir = tile::build(|program| {
        let a = program.storage_read::<F32, 2>(Shape::new([32, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 32);
        let y = program.storage_write::<F32, 2>(Shape::new([32, 32]));
        qmatmul_workgroup_with_epilogues::<32, 32, 8>(
            program,
            &a,
            &b,
            &y,
            &QmatmulEpilogues::empty(),
            65_535,
        );
    });
    let lowered = lower_or_fail(&ir, "workgroup qmatmul");
    assert!(
        !module_uses_subgroup(lowered.module()),
        "workgroup qmatmul emitted subgroup ops"
    );
}

#[test]
fn workgroup_qgemv_lowers_without_subgroups() {
    let ir = tile::build(|program| {
        let a = program.storage_read::<F32, 2>(Shape::new([1, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q4K, 256, 128);
        let y = program.storage_write::<F32, 2>(Shape::new([1, 128]));
        qgemv_workgroup_with_epilogue::<64, 8>(
            program,
            &a,
            &b,
            &y,
            &QmatmulEpilogues::empty(),
            65_535,
        );
    });
    let lowered = lower_or_fail(&ir, "workgroup qgemv");
    assert!(
        !module_uses_subgroup(lowered.module()),
        "workgroup qgemv emitted subgroup ops"
    );
}

fn module_uses_subgroup(module: &naga::Module) -> bool {
    let uses_in = |expressions: &naga::Arena<naga::Expression>| {
        expressions.iter().any(|(_, expr)| {
            matches!(
                expr,
                naga::Expression::SubgroupOperationResult { .. }
                    | naga::Expression::SubgroupBallotResult
            )
        })
    };
    module
        .functions
        .iter()
        .any(|(_, f)| uses_in(&f.expressions))
        || module
            .entry_points
            .iter()
            .any(|entry| uses_in(&entry.function.expressions))
}

#[test]
fn qmatmul_fallback_preserves_post_epilogue() {
    let tanh = UnaryEpilogue::new("test_tanh", |tile| tile.tanh());
    let with_post = qmatmul_epilogue_fallback_ir(Some(&tanh));
    let lowered = lower_or_fail(&with_post, "qmatmul fallback with tanh post");
    assert!(
        module_uses_tanh(lowered.module()),
        "fallback path dropped the post epilogue: lowered Naga module contains no Tanh call"
    );

    // Control: same shape, no epilogue. The lowered module must NOT contain
    // a Tanh call, ruling out a false-positive from unrelated kernel math.
    let without = qmatmul_epilogue_fallback_ir(None);
    let lowered = lower_or_fail(&without, "qmatmul fallback no epilogue");
    assert!(
        !module_uses_tanh(lowered.module()),
        "control case unexpectedly contains a Tanh call — test is not specific enough"
    );
}
