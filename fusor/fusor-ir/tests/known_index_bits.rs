use fusor_ir::dtype::NumericContract;
use fusor_ir::ir::kernel::{
    Builtin, ScalarElement, TileBinaryOp as Op, TileCompareOp, TileExpr, TileExprKind as K,
    TileLiteral, simplify_index,
};

fn lit(n: u32) -> TileExpr {
    TileExpr::new(
        K::Literal(TileLiteral::U32(n)),
        ScalarElement::U32.element(),
    )
}
fn bin(op: Op, a: TileExpr, b: TileExpr) -> TileExpr {
    TileExpr::new(
        K::Binary {
            op,
            left: a,
            right: b,
            numeric: NumericContract::RELAXED,
        },
        ScalarElement::U32.element(),
    )
}
fn lane() -> TileExpr {
    TileExpr::new(
        K::Builtin(Builtin::SubgroupLane),
        ScalarElement::U32.element(),
    )
}
fn eval(e: &TileExpr, value: u32) -> u32 {
    match e.kind() {
        K::Literal(TileLiteral::U32(n)) => *n,
        K::Builtin(Builtin::SubgroupId) => value.rotate_left(13),
        K::Builtin(Builtin::Lane) => value.reverse_bits(),
        K::Builtin(_) => value,
        K::Binary {
            op, left, right, ..
        } => {
            let (a, b) = (eval(left, value), eval(right, value));
            match op {
                Op::Add => a.wrapping_add(b),
                Op::Sub => a.wrapping_sub(b),
                Op::Mul => a.wrapping_mul(b),
                Op::Div => a / b,
                Op::Rem => a % b,
                Op::BitAnd => a & b,
                Op::BitOr => a | b,
                Op::BitXor => a ^ b,
                Op::Shl => a.wrapping_shl(b),
                Op::Shr => a.wrapping_shr(b),
                Op::Min => a.min(b),
                Op::Max => a.max(b),
                _ => panic!("unexpected binary"),
            }
        }
        _ => panic!("unexpected expression"),
    }
}

#[test]
fn holes_in_strided_lane_addresses_make_decisions_constant() {
    // Two disjoint runs of index bits, with a gap that alignment alone misses.
    let index = bin(
        Op::Add,
        bin(Op::Mul, bin(Op::Div, lane(), lit(4)), lit(128)),
        bin(Op::Mul, bin(Op::Rem, lane(), lit(4)), lit(8)),
    );
    let decision = bin(Op::BitAnd, bin(Op::Shr, index, lit(5)), lit(1));
    assert_eq!(simplify_index(&decision), lit(0));
    let condition = TileExpr::new(
        K::Compare {
            op: TileCompareOp::Eq,
            left: decision.clone(),
            right: lit(0),
        },
        ScalarElement::Bool.element(),
    );
    let selected = TileExpr::new(
        K::Select {
            condition,
            accept: lit(7),
            reject: lit(9),
        },
        ScalarElement::U32.element(),
    );
    assert_eq!(simplify_index(&selected), lit(7));
    for value in 0..4096 {
        assert_eq!(eval(&decision, value), 0);
    }
}

#[test]
fn index_simplification_preserves_carries_and_unsigned_wrap() {
    let mut seed = 0x9e3779b9u32;
    for shift in 0..32 {
        for mask in [1, 3, 15, 255, 0x55555555, 0x80000001, u32::MAX] {
            let a = bin(Op::Shl, bin(Op::BitAnd, lane(), lit(mask)), lit(shift));
            for offset in [0, 1, 31, 128, u32::MAX] {
                let input = bin(Op::Add, a.clone(), lit(offset));
                for shifted in [
                    bin(Op::Shr, input.clone(), lit(shift)),
                    bin(Op::Div, input, lit(1u32 << shift)),
                ] {
                    let input = bin(Op::BitAnd, shifted, lit(mask));
                    let result = simplify_index(&input);
                    for _ in 0..32 {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        assert_eq!(
                            eval(&input, seed),
                            eval(&result, seed),
                            "shift={shift} mask={mask:x} offset={offset}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn masked_offsets_share_across_aligned_rows_and_loop_steps() {
    let row = TileExpr::new(
        K::Builtin(Builtin::SubgroupId),
        ScalarElement::U32.element(),
    );
    let step = TileExpr::new(K::Builtin(Builtin::Lane), ScalarElement::U32.element());
    let offset = bin(Op::Mul, lane(), lit(8));
    for mask in [1, 24, 152, 255] {
        let expected = simplify_index(&bin(Op::BitAnd, offset.clone(), lit(mask)));
        for column in 0..4 {
            let address = bin(
                Op::Add,
                bin(
                    Op::Add,
                    bin(Op::Mul, step.clone(), lit(1024)),
                    offset.clone(),
                ),
                bin(Op::Mul, bin(Op::Add, row.clone(), lit(column)), lit(14336)),
            );
            let masked = bin(Op::BitAnd, address, lit(mask));
            let result = simplify_index(&masked);
            assert_eq!(result, expected, "column={column} mask={mask}");
            let mut seed = u32::MAX;
            for _ in 0..1024 {
                assert_eq!(eval(&masked, seed), eval(&result, seed));
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            }
        }
    }
}

#[test]
fn masked_sums_keep_lower_bits_that_carry_into_sparse_masks() {
    let mut seed = u32::MAX;
    for shift in 0..32 {
        for mask in [0, 1, 0x80, 0xa4, 255, 0x80000000, 0x80000001, u32::MAX] {
            for offset in [0, 1, 127, 255, u32::MAX] {
                let sum = bin(
                    Op::Add,
                    bin(
                        Op::Add,
                        bin(Op::Shl, lane(), lit(shift)),
                        bin(Op::BitAnd, lane(), lit(127)),
                    ),
                    lit(offset),
                );
                let original = bin(Op::BitAnd, sum, lit(mask));
                let simplified = simplify_index(&original);
                for _ in 0..128 {
                    assert_eq!(
                        eval(&original, seed),
                        eval(&simplified, seed),
                        "shift={shift} mask={mask:x} offset={offset} value={seed:x}"
                    );
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                }
            }
        }
    }
}
