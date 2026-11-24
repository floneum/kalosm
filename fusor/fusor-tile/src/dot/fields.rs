//! Pair exact unsigned bit-field conversions through the standard two-lane
//! unpack operation. No storage format, tensor shape or device name is involved.
use super::{K, Kernel, NumericContract, Op, ScalarElement, Term, TileExpr};
use fusor_ir::ir::kernel::{TileLiteral, TileUnaryOp};
use rustc_hash::FxHashMap;

fn literal(e: &TileExpr) -> Option<u32> {
    if let K::Literal(TileLiteral::U32(n)) = e.kind() {
        Some(*n)
    } else {
        None
    }
}

/// `(word >> shift) & mask`, allowing nested constant shifts and masks.
fn field(e: &TileExpr) -> (TileExpr, u32, u32) {
    field_bounded(e, 32).unwrap_or_else(|| (e.clone(), 0, u32::MAX))
}

fn field_bounded(e: &TileExpr, remaining: usize) -> Option<(TileExpr, u32, u32)> {
    let remaining = remaining.checked_sub(1)?;
    if let K::Binary {
        op, left, right, ..
    } = e.kind()
    {
        match (op, literal(right)) {
            (Op::BitAnd, Some(mask)) => {
                let (word, shift, inner) = field_bounded(left, remaining)?;
                return Some((word, shift, inner & mask));
            }
            (Op::Shr, Some(n)) if n < 32 => {
                let (word, shift, mask) = field_bounded(left, remaining)?;
                if shift + n < 32 {
                    return Some((word, shift + n, mask >> n));
                }
            }
            _ => {}
        }
    }
    Some((e.clone(), 0, u32::MAX))
}

struct Packed {
    value: TileExpr,
    mask: u32,
    component: Option<usize>,
}

/// Lift compatible bit operations into two independent 16-bit fields. Every
/// intermediate occupies at most ten bits, so left shifts cannot carry into
/// the other field. Literals broadcast; mixed component assignments decline.
fn packed(
    b: &Kernel,
    e: &TileExpr,
    numeric: NumericContract,
    budget: &mut usize,
) -> Option<Packed> {
    *budget = budget.checked_sub(1)?;
    let both = |n| b.u32(n | (n << 16));
    if let Some(n) = literal(e).filter(|n| *n <= 1023) {
        return Some(Packed {
            value: both(n),
            mask: n,
            component: None,
        });
    }
    let combined = (|| {
        let K::Binary {
            op, left, right, ..
        } = e.kind()
        else {
            return None;
        };
        match op {
            Op::BitOr | Op::BitXor | Op::BitAnd => {
                let a = packed(b, left, numeric, budget)?;
                let c = packed(b, right, numeric, budget)?;
                if a.component.is_some() && c.component.is_some() && a.component != c.component {
                    return None;
                }
                let mask = if *op == Op::BitAnd {
                    a.mask & c.mask
                } else {
                    a.mask | c.mask
                };
                Some(Packed {
                    value: b.binary(*op, a.value, c.value, numeric),
                    mask,
                    component: a.component.or(c.component),
                })
            }
            Op::Shl | Op::Shr => {
                let shift = literal(right).filter(|s| *s < 10)?;
                let a = packed(b, left, numeric, budget)?;
                let mask = if *op == Op::Shl {
                    a.mask << shift
                } else {
                    a.mask >> shift
                };
                if mask > 1023 {
                    return None;
                }
                let value = b.binary(*op, a.value, b.u32(shift), numeric);
                let value = if *op == Op::Shr {
                    b.binary(Op::BitAnd, value, both(mask), numeric)
                } else {
                    value
                };
                Some(Packed {
                    value,
                    mask,
                    component: a.component,
                })
            }
            _ => None,
        }
    })();
    let result = combined.or_else(|| {
        let (word, shift, mask) = field(e);
        if mask == 0 || mask > 1023 || shift % 16 + (32 - mask.leading_zeros()) > 16 {
            return None;
        }
        let word = if shift.is_multiple_of(16) {
            word
        } else {
            b.binary(Op::Shr, word, b.u32(shift % 16), numeric)
        };
        Some(Packed {
            value: b.binary(Op::BitAnd, word, both(mask), numeric),
            mask,
            component: Some((shift / 16) as usize),
        })
    })?;
    // Both source components must reach the same normal form even when one
    // has an implicit high-word mask and the other has an explicit mask.
    let (word, shift, mask) = field(&result.value);
    let word = if shift == 0 {
        word
    } else {
        b.binary(Op::Shr, word, b.u32(shift), numeric)
    };
    let value = if mask == u32::MAX {
        word
    } else {
        b.binary(Op::BitAnd, word, b.u32(mask), numeric)
    };
    Some(Packed { value, ..result })
}

pub(super) fn pack_casts(b: &Kernel, terms: &mut [Term], numeric: NumericContract) -> bool {
    let f32_type = ScalarElement::F32.element();
    let mut groups = FxHashMap::<(TileExpr, u32), [Vec<TileExpr>; 2]>::default();
    for value in terms.iter().flat_map(|t| &t.factors) {
        let K::Cast { value: integer, to } = value.kind() else {
            continue;
        };
        if *to != f32_type || integer.element() != ScalarElement::U32.element() {
            continue;
        }
        let Some(p) = packed(b, integer, numeric, &mut 128) else {
            continue;
        };
        let Some(component) = p.component else {
            continue;
        };
        if p.mask == 0 {
            continue;
        }
        groups.entry((p.value, p.mask)).or_default()[component].push(value.clone());
    }
    let mut replacements = FxHashMap::default();
    for ((mantissas, mask), casts) in groups {
        if casts.iter().any(Vec::is_empty) {
            continue;
        }
        // A normal half has ten mantissa bits. Every admitted integer plus
        // the next power of two is represented exactly, including zero.
        let bits = 32 - mask.leading_zeros();
        let mantissas = b.binary(Op::Shl, mantissas, b.u32(10 - bits), numeric);
        let exponent = (15 + bits) << 10;
        let encoded = b.binary(
            Op::BitOr,
            mantissas,
            b.u32(exponent | (exponent << 16)),
            numeric,
        );
        let values = b.unary(TileUnaryOp::Unpack2x16Float, encoded, numeric);
        for (component, aliases) in casts.into_iter().enumerate() {
            let value = TileExpr::new(
                K::VecComponent {
                    vector: values.clone(),
                    component: component as u32,
                },
                f32_type,
            );
            // Keep the bias scalar so relaxed dot algebra can combine its
            // correction with the other shared offsets in the whole window.
            let value = b.binary(Op::Sub, value, b.f32((1u32 << bits) as f32), numeric);
            for alias in aliases {
                replacements.insert(alias, value.clone());
            }
        }
    }
    for factor in terms.iter_mut().flat_map(|t| &mut t.factors) {
        if let Some(replacement) = replacements.get(factor) {
            *factor = replacement.clone();
        }
    }
    !replacements.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn integer(e: &TileExpr, word: &dyn Fn(u64) -> u32) -> u32 {
        match e.kind() {
            K::Literal(TileLiteral::U32(n)) => *n,
            K::LoadLocal(local) => word(local.id()),
            K::Binary {
                op, left, right, ..
            } => {
                let (a, b) = (integer(left, word), integer(right, word));
                match op {
                    Op::BitAnd => a & b,
                    Op::BitOr => a | b,
                    Op::Shr => a >> b,
                    Op::Shl => a << b,
                    _ => panic!("integer op"),
                }
            }
            _ => panic!("integer expression"),
        }
    }
    fn floats(e: &TileExpr, word: &dyn Fn(u64) -> u32) -> Vec<f32> {
        match e.kind() {
            K::Literal(TileLiteral::F32(n)) => vec![f32::from_bits(*n)],
            K::Cast { value, .. } => vec![integer(value, word) as f32],
            K::Unary {
                op: TileUnaryOp::Unpack2x16Float,
                value,
                ..
            } => {
                let bits = integer(value, word);
                vec![
                    half::f16::from_bits(bits as u16).to_f32(),
                    half::f16::from_bits((bits >> 16) as u16).to_f32(),
                ]
            }
            K::Vec { parts, .. } => parts.iter().map(|p| floats(p, word)[0]).collect(),
            K::VecComponent { vector, component } => {
                vec![floats(vector, word)[*component as usize]]
            }
            K::Binary {
                op, left, right, ..
            } => floats(left, word)
                .into_iter()
                .zip(floats(right, word))
                .map(|(a, b)| match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    _ => panic!("float op"),
                })
                .collect(),
            _ => panic!("float expression"),
        }
    }

    #[test]
    fn packed_integer_fields_convert_exactly_without_format_metadata() {
        let b = Kernel::new();
        let word = b.load_local(b.local(ScalarElement::U32.element()));
        let numeric = NumericContract::RELAXED;
        for mask in [1u32, 7, 15, 31, 255, 1023] {
            for shift in 0..=16 - (32 - mask.leading_zeros()) {
                let original: Vec<_> = [shift, shift + 16]
                    .into_iter()
                    .map(|shift| {
                        b.cast(
                            b.binary(
                                Op::BitAnd,
                                b.binary(Op::Shr, word.clone(), b.u32(shift), numeric),
                                b.u32(mask),
                                numeric,
                            ),
                            ScalarElement::F32.element(),
                        )
                    })
                    .collect();
                let mut terms: Vec<_> = original
                    .iter()
                    .map(|e| Term {
                        negative: false,
                        factors: vec![e.clone()],
                    })
                    .collect();
                pack_casts(&b, &mut terms, numeric);
                assert_ne!(original[0], terms[0].factors[0]);
                let mut seed = u32::MAX;
                for _ in 0..1024 {
                    for (before, after) in original.iter().zip(&terms) {
                        assert_eq!(
                            floats(before, &|_| seed)[0].to_bits(),
                            floats(&after.factors[0], &|_| seed)[0].to_bits()
                        );
                    }
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                }
            }
        }
    }

    #[test]
    fn composed_bit_planes_pair_without_storage_metadata() {
        let b = Kernel::new();
        let first = b.local(ScalarElement::U32.element());
        let second = b.local(ScalarElement::U32.element());
        let numeric = NumericContract::RELAXED;
        for (low_bits, high_bits) in [(1, 1), (4, 2), (3, 5), (5, 5)] {
            for shift in 0..=16 - low_bits.max(high_bits) {
                let original: Vec<_> = [shift, shift + 16]
                    .into_iter()
                    .map(|shift| {
                        let field = |local, mask| {
                            b.binary(
                                Op::BitAnd,
                                b.binary(Op::Shr, b.load_local(local), b.u32(shift), numeric),
                                b.u32(mask),
                                numeric,
                            )
                        };
                        let lo = field(first.clone(), (1 << low_bits) - 1);
                        let hi = b.binary(
                            Op::Shl,
                            field(second.clone(), (1 << high_bits) - 1),
                            b.u32(low_bits),
                            numeric,
                        );
                        b.cast(
                            b.binary(Op::BitOr, lo, hi, numeric),
                            ScalarElement::F32.element(),
                        )
                    })
                    .collect();
                let mut terms: Vec<_> = original
                    .iter()
                    .map(|e| Term {
                        negative: false,
                        factors: vec![e.clone()],
                    })
                    .collect();
                assert!(pack_casts(&b, &mut terms, numeric));
                let mut seed = 0x9e3779b9u32;
                for _ in 0..1024 {
                    let a = seed;
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    let c = seed;
                    let values = |id| {
                        if id == first.id() {
                            a
                        } else {
                            assert_eq!(id, second.id());
                            c
                        }
                    };
                    for (before, after) in original.iter().zip(&terms) {
                        assert_eq!(
                            floats(before, &values)[0].to_bits(),
                            floats(&after.factors[0], &values)[0].to_bits()
                        );
                    }
                }
            }
        }
    }
}
