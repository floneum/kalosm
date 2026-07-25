//! Machine-pinned IR goldens for the workgroup-tiled quantized matmul family.
//!
//! The family is emitted by one register-tile template (`qmatmul_workgroup`),
//! and runs where the subgroup paths can't: adapters without
//! `Features::SUBGROUP`, plus every f16-activation quantized matmul. That
//! makes it hard to cover with the local GPU suites, so each config's tile IR
//! and lowered Naga module are hashed and pinned here. The digests were
//! captured from the hand-written kernels the template replaced, which the
//! template reproduced bit-for-bit across the full
//! shape x format x storage/staging x epilogue cross product (336 configs);
//! this file keeps the union of one-axis sweeps around the two base shapes.
//!
//! An intentional codegen change re-captures the goldens from the failure
//! output.

use fusor_tile_ir::{tile, ElementType, GgmlQuantFormat, ScalarElement, Shape};
use fusor_tile_ir_kernels::{
    qmatmul_workgroup_with_epilogues, quantized_matrix, QmatmulEpilogues, QmatmulExtra,
    UnaryEpilogue, UnaryEpilogueWithExtras,
};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Epilogue {
    None,
    Post,
    PostColumn,
    PostPointwise,
    Pre,
    PrePointwise,
    PairedOffsets,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    m: u32,
    k: u32,
    n: u32,
    format: GgmlQuantFormat,
    storage: ScalarElement,
    staging: ScalarElement,
    epilogue: Epilogue,
}

impl Case {
    fn label(&self) -> String {
        format!(
            "m={} k={} n={} {:?} storage={:?} staging={:?} {:?}",
            self.m, self.k, self.n, self.format, self.storage, self.staging, self.epilogue
        )
    }

    /// Matrix columns backing `n` output columns: the paired-offset epilogue
    /// reads two accumulators per output column.
    fn matrix_cols(&self) -> u32 {
        match self.epilogue {
            Epilogue::PairedOffsets => self.n * 2,
            _ => self.n,
        }
    }
}

fn build_ir(case: Case) -> fusor_tile_ir::KernelIr {
    tile::build(move |program| {
        let a = program.storage_read(case.storage.element(), Shape::new([case.m, case.k]));
        let b = quantized_matrix(program, case.format, case.k, case.matrix_cols());
        let column = program.storage_read(ElementType::F32, Shape::new([case.n]));
        let pointwise = program.storage_read(ElementType::F32, Shape::new([case.m, case.n]));
        let pre_pointwise = program.storage_read(ElementType::F32, Shape::new([case.m, case.k]));
        let y = program.storage_write(case.storage.element(), Shape::new([case.m, case.n]));

        let post = UnaryEpilogue::new("test_post", |tile| tile.tanh());
        let pre = UnaryEpilogue::new("test_pre", |tile| tile.silu());
        let post_extras = UnaryEpilogueWithExtras::new("test_post_extras", 1, |tiles| {
            tiles[0].clone() * tiles[1].clone()
        });
        let pre_extras = UnaryEpilogueWithExtras::new("test_pre_extras", 1, |tiles| {
            tiles[0].clone() + tiles[1].clone()
        });
        let paired =
            UnaryEpilogueWithExtras::new_with_value_arity("test_paired_product", 2, 0, |values| {
                values[0].clone().silu() * values[1].clone()
            });
        let column_extra = [QmatmulExtra::Column(&column)];
        let pointwise_extra = [QmatmulExtra::Pointwise(&pointwise)];
        let pre_pointwise_extra = [QmatmulExtra::Pointwise(&pre_pointwise)];
        let offsets = [0, case.matrix_cols() / 2];

        let epilogues = match case.epilogue {
            Epilogue::None => QmatmulEpilogues::empty(),
            Epilogue::Post => QmatmulEpilogues::post(&post),
            Epilogue::PostColumn => QmatmulEpilogues {
                post_with_extras: Some(&post_extras),
                post_extra_inputs: &column_extra,
                ..QmatmulEpilogues::empty()
            },
            Epilogue::PostPointwise => QmatmulEpilogues {
                post_with_extras: Some(&post_extras),
                post_extra_inputs: &pointwise_extra,
                ..QmatmulEpilogues::empty()
            },
            Epilogue::Pre => QmatmulEpilogues::pre(&pre),
            Epilogue::PrePointwise => QmatmulEpilogues {
                pre_with_extras: Some(&pre_extras),
                pre_extra_inputs: &pre_pointwise_extra,
                ..QmatmulEpilogues::empty()
            },
            Epilogue::PairedOffsets => QmatmulEpilogues {
                post_with_extras: Some(&paired),
                post_accumulator_offsets: &offsets,
                ..QmatmulEpilogues::empty()
            },
        };

        qmatmul_workgroup_with_epilogues(program, &a, &b, &y, case.staging, &epilogues, 65_535);
    })
}

/// Union of one-axis sweeps around the two base shapes: every shape, quant
/// format, storage/staging pair and epilogue form appears, without paying the
/// full cross product's dump-formatting cost.
fn cases() -> Vec<Case> {
    const BASE: [(u32, u32, u32); 2] = [(32, 256, 32), (1, 256, 128)];
    let mut cases: Vec<Case> = Vec::new();
    let mut push = |case: Case| {
        if !cases.iter().any(|seen| seen.label() == case.label()) {
            cases.push(case);
        }
    };
    let base_case = |(m, k, n): (u32, u32, u32)| Case {
        m,
        k,
        n,
        format: GgmlQuantFormat::Q4K,
        storage: ScalarElement::F32,
        staging: ScalarElement::F32,
        epilogue: Epilogue::None,
    };
    // Aligned and ragged shapes for both geometries: the tiled family covers
    // 32x32 output tiles over an 8-deep K chunk, the single-row family 1x64.
    for shape in [
        (32, 256, 32),
        (33, 260, 33),
        (96, 512, 160),
        (1, 256, 128),
        (1, 260, 130),
        (1, 512, 64),
    ] {
        push(base_case(shape));
    }
    for shape in BASE {
        for format in [
            GgmlQuantFormat::Q8_0,
            GgmlQuantFormat::Q4K,
            GgmlQuantFormat::Q4KNative,
            GgmlQuantFormat::Q6K,
        ] {
            push(Case {
                format,
                ..base_case(shape)
            });
        }
        for (storage, staging) in [
            (ScalarElement::F32, ScalarElement::F32),
            (ScalarElement::F32, ScalarElement::F16),
            (ScalarElement::F16, ScalarElement::F16),
        ] {
            push(Case {
                storage,
                staging,
                ..base_case(shape)
            });
            // The paired-accumulator epilogue is a single-row form, and f16
            // storage only ever runs without epilogues.
            if storage == ScalarElement::F16 {
                continue;
            }
            for epilogue in [
                Epilogue::Post,
                Epilogue::PostColumn,
                Epilogue::PostPointwise,
                Epilogue::Pre,
                Epilogue::PrePointwise,
                Epilogue::PairedOffsets,
            ] {
                if epilogue == Epilogue::PairedOffsets && shape.0 != 1 {
                    continue;
                }
                push(Case {
                    storage,
                    staging,
                    epilogue,
                    ..base_case(shape)
                });
            }
        }
    }
    cases
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// `Expr`'s cached `hash` mixes the `Rc` address of every referenced local, so
/// it differs between two builds of the same program. Everything else in the
/// dump is structural; local identity is pinned by the lowered Naga module,
/// whose handles are arena indices.
fn structural_dump(ir: &fusor_tile_ir::KernelIr) -> String {
    format!("{ir:#?}")
        .lines()
        .filter(|line| !line.trim_start().starts_with("hash: "))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn workgroup_qmatmul_ir_matches_golden() {
    let measured = cases()
        .into_iter()
        .map(|case| {
            let ir = build_ir(case);
            let ir_digest = fnv1a(structural_dump(&ir).as_bytes());
            let lowered = ir
                .lower_to_naga()
                .unwrap_or_else(|error| panic!("lowering failed for {}: {error}", case.label()));
            let naga_digest = fnv1a(format!("{:#?}", lowered.module()).as_bytes());
            format!(
                "{} ir {ir_digest:#018x} naga {naga_digest:#018x}",
                case.label()
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let golden = include_str!("goldens/qmatmul_workgroup_ir.txt");
    assert!(
        golden.trim() == measured.trim(),
        "workgroup qmatmul IR golden mismatch; measured values:\n{measured}"
    );
}
