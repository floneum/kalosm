use fusor_tile_ir::{tile, GgmlQuantFormat, NagaKernel, ScalarElement, Shape};
use fusor_tile_ir_kernels::{
    qgemv_with_epilogue, qgemv_workgroup_f16_with_epilogue, qgemv_workgroup_with_epilogue,
    qmatmul_with_epilogue, qmatmul_workgroup_f16_with_epilogues, qmatmul_workgroup_with_epilogues,
    quantized_matrix, try_batched_coop_matmul, DenseCoopMatmulConfig, DenseCoopMatmulTile,
    DenseMatmulEpilogues, DenseMatmulShape, DenseMatmulTensors, QmatmulEpilogues, SubgroupConfig,
    UnaryEpilogue, UnaryEpilogueWithExtras,
};

fn lower_or_fail(ir: &fusor_tile_ir::KernelIr, label: &str) -> NagaKernel {
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("{label} lowering failed: {error}"))
}

fn subgroup_token() -> fusor_tile_ir::SubgroupToken {
    fusor_tile_ir::SubgroupToken::new_unchecked()
}

fn subgroup_config(size: u32) -> SubgroupConfig {
    SubgroupConfig::fixed(subgroup_token(), size)
}

fn coop_token() -> fusor_tile_ir::CoopMatrixToken {
    fusor_tile_ir::CoopMatrixToken::new_unchecked()
}

fn qgemv_ir_with_subgroup_size(
    format: GgmlQuantFormat,
    rows: u32,
    cols: u32,
    subgroup_size: u32,
) -> fusor_tile_ir::KernelIr {
    tile::build(|program| {
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([1, rows]));
        let b = quantized_matrix(program, format, rows, cols);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([1, cols]));
        qgemv_with_epilogue(
            program,
            &a,
            &b,
            &y,
            1,
            subgroup_config(subgroup_size),
            Option::<&UnaryEpilogue>::None,
        );
    })
}

fn qgemv_ir(format: GgmlQuantFormat, rows: u32, cols: u32) -> fusor_tile_ir::KernelIr {
    qgemv_ir_with_subgroup_size(format, rows, cols, 32)
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
fn q4k_ggml_qgemv_tail_columns_lower() {
    let ir = qgemv_ir(GgmlQuantFormat::Q4K, 4096, 8193);
    lower_or_fail(&ir, "q4k ggml qgemv tail columns");
}

#[test]
fn q4k_ggml_qgemv_tail_columns_lower_with_64_lane_subgroups() {
    let ir = qgemv_ir_with_subgroup_size(GgmlQuantFormat::Q4K, 4096, 8193, 64);
    lower_or_fail(&ir, "q4k ggml qgemv tail columns subgroup64");
}

#[test]
fn q4k_mid_qgemv_with_three_cols_per_subgroup_lowers() {
    let ir = qgemv_ir(GgmlQuantFormat::Q4K, 4096, 5120);
    lower_or_fail(&ir, "q4k mid qgemv 4x3");
}

#[test]
fn q4k_native_ggml_qgemv_lowers() {
    let ir = qgemv_ir(GgmlQuantFormat::Q4KNative, 4096, 8192);
    lower_or_fail(&ir, "q4k native ggml qgemv");
}

#[test]
fn q6k_ggml_qgemv_lowers() {
    let ir = qgemv_ir(GgmlQuantFormat::Q6K, 4096, 8192);
    lower_or_fail(&ir, "q6k ggml qgemv");
}

#[test]
fn scalar_qmatmul_lowers() {
    let ir = tile::build(|program| {
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([8, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 16);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([8, 16]));
        qmatmul_with_epilogue(
            program,
            &a,
            &b,
            &y,
            &QmatmulEpilogues::empty(),
            coop_token(),
            subgroup_config(32),
            8,
            4,
            8,
        );
    });
    lower_or_fail(&ir, "scalar qmatmul");
}

#[test]
fn cooperative_qmatmul_lowers() {
    let ir = tile::build(|program| {
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([64, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 64);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([64, 64]));
        qmatmul_with_epilogue(
            program,
            &a,
            &b,
            &y,
            &QmatmulEpilogues::empty(),
            coop_token(),
            subgroup_config(32),
            64,
            64,
            32,
        );
    });
    lower_or_fail(&ir, "cooperative qmatmul");
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
        let a = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.m, shape.k]),
        );
        let b = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.k, shape.n]),
        );
        let y = program.storage_write(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.m, shape.n]),
        );
        assert!(try_batched_coop_matmul(
            program,
            DenseMatmulTensors {
                a: &a,
                b: &b,
                y: &y,
            },
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
            DenseCoopMatmulConfig {
                coop: coop_token(),
                subgroups: subgroup_config(32),
                tile: DenseCoopMatmulTile {
                    bm: 64,
                    bn: 64,
                    bk: 16,
                },
            },
        ));
    });
    lower_or_fail(&ir, "cooperative dense f32 matmul");
}

#[test]
fn cooperative_dense_f32_matmul_with_pre_and_post_epilogues_lowers() {
    let pre = UnaryEpilogue::new("test_scale", |tile| tile * tile::Tile::f32(0.5));
    let post = UnaryEpilogue::new("test_tanh", |tile| tile.tanh());
    let ir = tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 1,
            m: 61,
            k: 63,
            n: 59,
        };
        let a = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.m, shape.k]),
        );
        let b = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.k, shape.n]),
        );
        let y = program.storage_write(
            ScalarElement::F32.element(),
            // Cooperative stores cover the whole selected output tile.
            Shape::new([64, 64]),
        );
        assert!(try_batched_coop_matmul(
            program,
            DenseMatmulTensors {
                a: &a,
                b: &b,
                y: &y,
            },
            shape,
            &DenseMatmulEpilogues {
                pre_a: Some(&pre),
                pre_b: None,
                post: Some(&post),
            },
            65_535,
            DenseCoopMatmulConfig {
                coop: coop_token(),
                subgroups: subgroup_config(32),
                tile: DenseCoopMatmulTile {
                    bm: 64,
                    bn: 64,
                    bk: 16,
                },
            },
        ));
    });
    lower_or_fail(&ir, "cooperative dense f32 matmul with epilogues");
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
        let a = program.storage_read(
            ScalarElement::F16.element(),
            Shape::new([shape.batch * shape.m, shape.k]),
        );
        let b = program.storage_read(
            ScalarElement::F16.element(),
            Shape::new([shape.batch * shape.k, shape.n]),
        );
        let y = program.storage_write(
            ScalarElement::F16.element(),
            Shape::new([shape.batch * shape.m, shape.n]),
        );
        assert!(try_batched_coop_matmul(
            program,
            DenseMatmulTensors {
                a: &a,
                b: &b,
                y: &y,
            },
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
            DenseCoopMatmulConfig {
                coop: coop_token(),
                subgroups: subgroup_config(32),
                tile: DenseCoopMatmulTile {
                    bm: 64,
                    bn: 64,
                    bk: 16,
                },
            },
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
        let a = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.m, shape.k]),
        );
        let b = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.k, shape.n]),
        );
        let y = program.storage_write(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.m, shape.n]),
        );
        assert!(try_batched_coop_matmul(
            program,
            DenseMatmulTensors {
                a: &a,
                b: &b,
                y: &y,
            },
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
            DenseCoopMatmulConfig {
                coop: coop_token(),
                subgroups: subgroup_config(32),
                tile: DenseCoopMatmulTile {
                    bm: 128,
                    bn: 128,
                    bk: 16,
                },
            },
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
        let a = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.m, shape.k]),
        );
        let b = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.k, shape.n]),
        );
        let y = program.storage_write(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.m, shape.n]),
        );
        assert!(try_batched_coop_matmul(
            program,
            DenseMatmulTensors {
                a: &a,
                b: &b,
                y: &y,
            },
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
            DenseCoopMatmulConfig {
                coop: coop_token(),
                subgroups: subgroup_config(32),
                tile: DenseCoopMatmulTile {
                    bm: 128,
                    bn: 64,
                    bk: 16,
                },
            },
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
        let a = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.m, shape.k]),
        );
        let b = program.storage_read(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.k, shape.n]),
        );
        let y = program.storage_write(
            ScalarElement::F32.element(),
            Shape::new([shape.batch * shape.m, shape.n]),
        );
        assert!(try_batched_coop_matmul(
            program,
            DenseMatmulTensors {
                a: &a,
                b: &b,
                y: &y,
            },
            shape,
            &DenseMatmulEpilogues::empty(),
            65_535,
            DenseCoopMatmulConfig {
                coop: coop_token(),
                subgroups: subgroup_config(32),
                tile: DenseCoopMatmulTile {
                    bm: 128,
                    bn: 256,
                    bk: 16,
                },
            },
        ));
    });
    lower_or_fail(&ir, "cooperative dense f32 128x256 N_PASSES=4 matmul");
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
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([2, 64]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 64, 64);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([2, 64]));
        let epilogues = QmatmulEpilogues {
            pre: None,
            pre_with_extras: None,
            pre_extra_inputs: &[],
            post,
            post_with_extras: None,
            post_extra_inputs: &[],
            post_accumulator_offsets: &[],
            post_acc_init_col_vector: None,
        };
        qmatmul_with_epilogue(
            program,
            &a,
            &b,
            &y,
            &epilogues,
            coop_token(),
            subgroup_config(32),
            64,
            64,
            32,
        );
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
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([32, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q8_0, 256, 32);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([32, 32]));
        qmatmul_workgroup_with_epilogues(program, &a, &b, &y, &QmatmulEpilogues::empty(), 65_535);
    });
    let lowered = lower_or_fail(&ir, "workgroup qmatmul");
    assert!(
        !module_uses_subgroup(lowered.module()),
        "workgroup qmatmul emitted subgroup ops"
    );
}

#[test]
fn f16_staged_workgroup_qmatmul_lowers_without_subgroups() {
    let ir = tile::build(|program| {
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([32, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q4KNative, 256, 32);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([32, 32]));
        qmatmul_workgroup_f16_with_epilogues(
            program,
            &a,
            &b,
            &y,
            &QmatmulEpilogues::empty(),
            65_535,
        );
    });
    let lowered = lower_or_fail(&ir, "f16 staged workgroup qmatmul");
    assert!(
        module_uses_f16(lowered.module()),
        "f16 staged workgroup qmatmul did not allocate f16 scratch"
    );
    assert!(
        !module_uses_subgroup(lowered.module()),
        "f16 staged workgroup qmatmul emitted subgroup ops"
    );
}

#[test]
fn workgroup_qgemv_lowers_without_subgroups() {
    let ir = tile::build(|program| {
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([1, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q4K, 256, 128);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([1, 128]));
        qgemv_workgroup_with_epilogue(program, &a, &b, &y, &QmatmulEpilogues::empty(), 65_535);
    });
    let lowered = lower_or_fail(&ir, "workgroup qgemv");
    assert!(
        !module_uses_subgroup(lowered.module()),
        "workgroup qgemv emitted subgroup ops"
    );
}

#[test]
fn f16_staged_workgroup_qgemv_lowers_without_subgroups() {
    let ir = tile::build(|program| {
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([1, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q4KNative, 256, 128);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([1, 128]));
        qgemv_workgroup_f16_with_epilogue(program, &a, &b, &y, &QmatmulEpilogues::empty(), 65_535);
    });
    let lowered = lower_or_fail(&ir, "f16 staged workgroup qgemv");
    assert!(
        module_uses_f16(lowered.module()),
        "f16 staged workgroup qgemv did not allocate f16 scratch"
    );
    assert!(
        !module_uses_subgroup(lowered.module()),
        "f16 staged workgroup qgemv emitted subgroup ops"
    );
}

#[test]
fn q4k_native_workgroup_qgemv_lowers_without_subgroups() {
    let ir = tile::build(|program| {
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([1, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q4KNative, 256, 128);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([1, 128]));
        qgemv_workgroup_with_epilogue(program, &a, &b, &y, &QmatmulEpilogues::empty(), 65_535);
    });
    let lowered = lower_or_fail(&ir, "q4k native workgroup qgemv");
    assert!(
        !module_uses_subgroup(lowered.module()),
        "native workgroup qgemv emitted subgroup ops"
    );
}

#[test]
fn workgroup_qgemv_accumulator_offsets_lower_without_subgroups() {
    let post = UnaryEpilogueWithExtras::new_with_value_arity("paired_product", 2, 0, |values| {
        values[0].clone() * values[1].clone()
    });
    let offsets = [0, 64];
    let epilogues = QmatmulEpilogues {
        post_with_extras: Some(&post),
        post_accumulator_offsets: &offsets,
        ..QmatmulEpilogues::empty()
    };
    let ir = tile::build(|program| {
        let a = program.storage_read(ScalarElement::F32.element(), Shape::new([1, 256]));
        let b = quantized_matrix(program, GgmlQuantFormat::Q4K, 256, 128);
        let y = program.storage_write(ScalarElement::F32.element(), Shape::new([1, 64]));
        qgemv_workgroup_with_epilogue(program, &a, &b, &y, &epilogues, 65_535);
    });
    let lowered = lower_or_fail(&ir, "workgroup qgemv accumulator offsets");
    assert!(
        !module_uses_subgroup(lowered.module()),
        "workgroup qgemv accumulator offsets emitted subgroup ops"
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

fn module_uses_f16(module: &naga::Module) -> bool {
    module.types.iter().any(|(_, ty)| {
        matches!(
            ty.inner,
            naga::TypeInner::Scalar(naga::Scalar {
                kind: naga::ScalarKind::Float,
                width: 2,
            })
        )
    })
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
