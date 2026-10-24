//! Bit facts for wrapping unsigned index arithmetic. Unknown bits stay unknown;
//! no bounds on lane IDs, loop variables or memory contents are assumed.
use super::{ScalarElement, TileBinaryOp as Op, TileExpr, TileExprKind as K, TileLiteral, lit_u32};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Default)]
pub(super) struct Bits {
    pub zero: u32,
    pub one: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carries_are_sound_for_every_four_bit_abstract_input() {
        let mut states = Vec::new();
        for mut ternary in 0..81 {
            let mut bits = Bits { zero: !15, one: 0 };
            for bit in 0..4 {
                match ternary % 3 {
                    0 => bits.zero |= 1 << bit,
                    1 => bits.one |= 1 << bit,
                    _ => {}
                }
                ternary /= 3;
            }
            let values: Vec<u32> = (0..16)
                .filter(|v| v & bits.zero == 0 && v & bits.one == bits.one)
                .collect();
            states.push((bits, values));
        }
        for (a, av) in &states {
            for (b, bv) in &states {
                for negate in [false, true] {
                    let result = if negate {
                        a.add(b.not()).add(Bits::constant(1))
                    } else {
                        a.add(*b)
                    };
                    assert_eq!(result.zero & result.one, 0);
                    for x in av {
                        for y in bv {
                            let value = if negate {
                                x.wrapping_sub(*y)
                            } else {
                                x.wrapping_add(*y)
                            };
                            assert_eq!(value & result.zero, 0);
                            assert_eq!(value & result.one, result.one);
                        }
                    }
                }
            }
        }
    }
}

impl Bits {
    fn constant(n: u32) -> Self {
        Self { zero: !n, one: n }
    }
    fn shl(self, n: u32) -> Self {
        Self {
            zero: (self.zero << n) | ((1u32 << n) - 1),
            one: self.one << n,
        }
    }
    fn shr(self, n: u32) -> Self {
        Self {
            zero: (self.zero >> n) | !(u32::MAX >> n),
            one: self.one >> n,
        }
    }
    fn and(self, b: Self) -> Self {
        Self {
            zero: self.zero | b.zero,
            one: self.one & b.one,
        }
    }
    fn or(self, b: Self) -> Self {
        Self {
            zero: self.zero & b.zero,
            one: self.one | b.one,
        }
    }
    fn not(self) -> Self {
        Self {
            zero: self.one,
            one: self.zero,
        }
    }
    fn add(self, b: Self) -> Self {
        // Carry into each bit is monotone in all lower input bits. Evaluate
        // both extremes to find carries that are always zero or always one.
        let minimum = self.one.wrapping_add(b.one);
        let maximum = (!self.zero).wrapping_add(!b.zero);
        let carry_one = minimum ^ self.one ^ b.one;
        let carry_possible = maximum ^ !self.zero ^ !b.zero;
        let known = (self.zero | self.one) & (b.zero | b.one) & (carry_one | !carry_possible);
        Self {
            zero: !minimum & known,
            one: minimum & known,
        }
    }
}

#[derive(Default)]
pub(super) struct Analysis(FxHashMap<TileExpr, Bits>);

impl Analysis {
    pub(super) fn get(&mut self, e: &TileExpr) -> Bits {
        if let Some(bits) = self.0.get(e) {
            return *bits;
        }
        let bits = if e.element() != ScalarElement::U32.element() {
            Bits::default()
        } else {
            match e.kind() {
                K::Literal(TileLiteral::U32(n)) => Bits::constant(*n),
                K::Binary {
                    op, left, right, ..
                } => {
                    let a = self.get(left);
                    let b = self.get(right);
                    match op {
                        Op::Add => a.add(b),
                        Op::Sub => a.add(b.not()).add(Bits::constant(1)),
                        Op::BitAnd => a.and(b),
                        Op::BitOr => a.or(b),
                        Op::BitXor => Self::xor(a, b),
                        Op::Shl => lit_u32(right)
                            .filter(|n| *n < 32)
                            .map_or(Bits::default(), |n| a.shl(n)),
                        Op::Shr => lit_u32(right)
                            .filter(|n| *n < 32)
                            .map_or(Bits::default(), |n| a.shr(n)),
                        Op::Div => lit_u32(right)
                            .filter(|n| n.is_power_of_two())
                            .map_or(Bits::default(), |n| a.shr(n.trailing_zeros())),
                        Op::Rem => lit_u32(right)
                            .filter(|n| n.is_power_of_two())
                            .map_or(Bits::default(), |n| a.and(Bits::constant(n - 1))),
                        Op::Mul => match (lit_u32(left), lit_u32(right)) {
                            (Some(n), _) => Self::mul(b, n),
                            (_, Some(n)) => Self::mul(a, n),
                            _ => Bits::default(),
                        },
                        _ => Bits::default(),
                    }
                }
                K::Select { accept, reject, .. } => {
                    let a = self.get(accept);
                    let b = self.get(reject);
                    Bits {
                        zero: a.zero & b.zero,
                        one: a.one & b.one,
                    }
                }
                _ => Bits::default(),
            }
        };
        self.0.insert(e.clone(), bits);
        bits
    }

    pub(super) fn compare(
        &mut self,
        op: super::TileCompareOp,
        left: &TileExpr,
        right: &TileExpr,
    ) -> Option<bool> {
        use super::TileCompareOp as C;
        if left.element() != ScalarElement::U32.element() || right.element() != left.element() {
            return None;
        }
        let (a, b) = (self.get(left), self.get(right));
        let lt = |a: Bits, b: Bits| {
            if !a.zero < b.one {
                Some(true)
            } else if a.one >= !b.zero {
                Some(false)
            } else {
                None
            }
        };
        let equal = || {
            if (a.one & b.zero) | (b.one & a.zero) != 0 {
                Some(false)
            } else if (a.zero | a.one) == u32::MAX && (b.zero | b.one) == u32::MAX {
                Some(a.one == b.one)
            } else {
                None
            }
        };
        match op {
            C::Lt => lt(a, b),
            C::Ge => lt(a, b).map(|x| !x),
            C::Gt => lt(b, a),
            C::Le => lt(b, a).map(|x| !x),
            C::Eq => equal(),
            C::Ne => equal().map(|x| !x),
        }
    }

    fn xor(a: Bits, b: Bits) -> Bits {
        Bits {
            zero: (a.zero & b.zero) | (a.one & b.one),
            one: (a.zero & b.one) | (a.one & b.zero),
        }
    }
    fn mul(a: Bits, n: u32) -> Bits {
        if a.zero | a.one == u32::MAX {
            return Bits::constant(a.one.wrapping_mul(n));
        }
        if n.count_ones() <= 4 {
            let mut sum = Bits::constant(0);
            for shift in 0..32 {
                if n & (1 << shift) != 0 {
                    sum = sum.add(a.shl(shift));
                }
            }
            sum
        } else {
            let count = (a.zero.trailing_ones() + n.trailing_zeros()).min(32);
            Bits {
                zero: u32::MAX.checked_shr(32 - count).unwrap_or(0),
                one: 0,
            }
        }
    }
}
