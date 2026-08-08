//! Scalar dtypes, quantized block formats, numeric contracts, persistence.

use crate::scalar::Lit;

/// Element type of an L0/L1 value. Two widenings past the reference's
/// `{f32, f16, u32}`: `I32` (float->int casts, `round`, sort-key scatter)
/// and `BF16` (free — shares the `widen-compute` rule with F16). No `Bool`:
/// comparisons return 1.0/0.0 in the operand dtype, exactly as today.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    U32,
    I32,
    /// A block-quantized weight format. Only ever the dtype of a
    /// `LeafKind::Quantized` leaf or an `L0::Dequant` input.
    Q(QFmt),
}

impl Dtype {
    /// Bytes one dense element occupies; quantized formats report 0.
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q(_) => 0,
        }
    }

    /// Accumulator width available in this dtype, in bits.
    pub const fn accum_bits(self) -> u8 {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 32,
            Self::F16 => 16,
            Self::BF16 => 16,
            Self::Q(_) => 0,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F16 | Self::BF16)
    }

    pub const fn is_int(self) -> bool {
        matches!(self, Self::U32 | Self::I32)
    }

    pub const fn is_quantized(self) -> bool {
        matches!(self, Self::Q(_))
    }

    /// What a storage-only narrow float widens to for compute — the type
    /// side of the `widen-compute` lowering rule.
    pub const fn compute_dtype(self) -> Self {
        match self {
            Self::F16 | Self::BF16 => Self::F32,
            other => other,
        }
    }
}

/// The six GGUF block formats fusor2 ingests end to end, on both backends.
/// The twelve the reference names but cannot ingest do not appear; adding
/// one is a `BlockSpec` row plus a `BlockProgram`, never a kernel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum QFmt {
    Q4_0,
    Q5_0,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl QFmt {
    /// Every ingestible format, in a fixed table-driving order.
    pub const ALL: [QFmt; 6] = [
        QFmt::Q4_0,
        QFmt::Q5_0,
        QFmt::Q8_0,
        QFmt::Q4K,
        QFmt::Q5K,
        QFmt::Q6K,
    ];

    pub const fn block_elements(self) -> u32 {
        match self {
            Self::Q4_0 | Self::Q5_0 | Self::Q8_0 => 32,
            Self::Q4K | Self::Q5K | Self::Q6K => 256,
        }
    }

    pub const fn block_bytes(self, layout: QLayout) -> u32 {
        match (self, layout) {
            (Self::Q4_0, QLayout::Native) => 18,
            (Self::Q4_0, QLayout::F32Scales) => 20,
            (Self::Q5_0, QLayout::Native) => 22,
            (Self::Q5_0, QLayout::F32Scales) => 24,
            (Self::Q8_0, QLayout::Native) => 34,
            (Self::Q8_0, QLayout::F32Scales) => 36,
            (Self::Q4K, QLayout::Native) => 144,
            (Self::Q4K, QLayout::F32Scales) => 148,
            (Self::Q5K, QLayout::Native) => 176,
            (Self::Q5K, QLayout::F32Scales) => 180,
            (Self::Q6K, QLayout::Native) => 210,
            (Self::Q6K, QLayout::F32Scales) => 212,
        }
    }
}

/// On-device byte layout of a quantized matrix. Both are legal inputs
/// *everywhere*; moving between them is the priced `qrepack` rewrite, not a
/// decision frozen at upload.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QLayout {
    Native,
    F32Scales,
}

/// Rounding mode carried on `ScalarKind::Round`. A real primitive, so the
/// trainer's 14-chained-comparison `round_small` deletes;
/// `HalfAwayFromZero` is what MSQ1 export idempotence depends on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RoundMode {
    HalfToEven,
    HalfAwayFromZero,
    Floor,
    Ceil,
    Trunc,
}

/// What a value's numerics permit. **Monotone**: no rewrite may lower
/// `min_accum_bits` or `min_operand_bits`, nor enable `reassoc`/`contract`
/// where a value forbids it. This makes `fold_split` sound, survives to WGSL
/// as an emitter obligation against Metal fast math, and stops an epilogue
/// fusion from narrowing a cooperative kernel's f32 accumulator.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NumericContract {
    pub min_accum_bits: u8,
    /// Narrowest **operand** a rewrite may substitute for this value's
    /// inputs, in bits. `32` — the default — means the operands stay f32.
    ///
    /// A separate axis from `min_accum_bits` and from `reassoc`/`contract`
    /// because it is a separate kind of permission: reassociating a sum,
    /// contracting a `mul`+`add` into an FMA and widening an accumulator are
    /// all exact-to-rounding, while re-encoding an operand onto a coarser
    /// grid is not. Without it, `reassoc` was being read as licence for the
    /// int8 activation path, whose error is ~1.7% at Q8_0 —
    /// four orders of magnitude past anything a rounding permission can
    /// authorize, and independent of accumulation order.
    pub min_operand_bits: u8,
    pub reassoc: bool,
    pub contract: bool,
}

impl NumericContract {
    /// f32 accumulation, f32 operands, reassociation and contraction
    /// allowed. The default every value carries.
    pub const RELAXED: Self = Self {
        min_accum_bits: 32,
        min_operand_bits: 32,
        reassoc: true,
        contract: true,
    };

    /// f32 accumulation, f32 operands, no reassociation, no contraction. QAT
    /// fake-quant rounding and the MSQ1 export path carry this.
    pub const STRICT: Self = Self {
        min_accum_bits: 32,
        min_operand_bits: 32,
        reassoc: false,
        contract: false,
    };

    /// [`Self::RELAXED`] plus permission to re-encode operands onto an 8-bit
    /// grid: what would license an int8-activation dot.
    ///
    /// Weaker than `RELAXED` — `RELAXED.allows(RELAXED_OPERANDS)` — so it
    /// cannot be reached by [`Self::meet`] from values that do not already
    /// carry it. That is the point: quantizing the activations is an opt-in
    /// with a measurable accuracy cost, not something a fusion can decide.
    /// **Nothing in the tree mints it**, so the int8 path is offered nowhere.
    /// This is the licence kept for it: when an int8 dot returns it must
    /// return as one mechanism — a `Dot4I8`-shaped `ScalarKind` over packed
    /// operands inside an ordinary `KContract` — guarded by this contract,
    /// never as a second contraction node with its own lowerings.
    pub const RELAXED_OPERANDS: Self = Self {
        min_operand_bits: 8,
        ..Self::RELAXED
    };

    /// True when `self` permits everything `other` requires.
    pub const fn allows(self, other: Self) -> bool {
        self.min_accum_bits >= other.min_accum_bits
            && self.min_operand_bits >= other.min_operand_bits
            && (other.reassoc || !self.reassoc)
            && (other.contract || !self.contract)
    }

    /// The strongest contract weaker than both. Monotone by construction.
    pub const fn meet(self, other: Self) -> Self {
        Self {
            min_accum_bits: if self.min_accum_bits > other.min_accum_bits {
                self.min_accum_bits
            } else {
                other.min_accum_bits
            },
            min_operand_bits: if self.min_operand_bits > other.min_operand_bits {
                self.min_operand_bits
            } else {
                other.min_operand_bits
            },
            reassoc: self.reassoc && other.reassoc,
            contract: self.contract && other.contract,
        }
    }

}

/// How long a value lives. Lets a quantized repack amortize against a
/// weight's lifetime, and tells the extractor what it may recompute.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Persistence {
    Step,
    Persistent,
}

/// A typed constant. `PartialEq`/`Hash` are bitwise so the e-graph memo is
/// exact: `-0.0` and `0.0` are distinct, `NaN` hash-conses with itself.
#[derive(Copy, Clone, Debug)]
pub enum Splat {
    F32(f32),
    F16(u16),
    BF16(u16),
    U32(u32),
    I32(i32),
}

impl Splat {
    pub const fn dtype(self) -> Dtype {
        match self {
            Self::F32(_) => Dtype::F32,
            Self::F16(_) => Dtype::F16,
            Self::BF16(_) => Dtype::BF16,
            Self::U32(_) => Dtype::U32,
            Self::I32(_) => Dtype::I32,
        }
    }

    pub const fn bits(self) -> u32 {
        match self {
            Self::F32(v) => v.to_bits(),
            Self::F16(v) | Self::BF16(v) => v as u32,
            Self::U32(v) => v,
            Self::I32(v) => v as u32,
        }
    }
}

impl PartialEq for Splat {
    fn eq(&self, other: &Self) -> bool {
        self.dtype() == other.dtype() && self.bits() == other.bits()
    }
}
impl Eq for Splat {}
impl std::hash::Hash for Splat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.dtype().hash(state);
        self.bits().hash(state);
    }
}

impl From<Splat> for Lit {
    fn from(s: Splat) -> Self {
        Lit(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The operand-precision axis sits in the lattice exactly where the other
    /// three do: `meet` keeps the stricter side, `allows` reads "self is at
    /// least as strict as other", and f32 operands are the default.
    #[test]
    fn narrow_operands_are_a_strict_relaxation() {
        assert_eq!(NumericContract::RELAXED.min_operand_bits, 32);
        assert_eq!(NumericContract::STRICT.min_operand_bits, 32);
        assert_eq!(NumericContract::RELAXED_OPERANDS.min_operand_bits, 8);

        // One direction only: RELAXED satisfies everything RELAXED_OPERANDS
        // asks for, never the reverse. That asymmetry is the whole guard on
        // the int8 activation dot.
        assert!(NumericContract::RELAXED.allows(NumericContract::RELAXED_OPERANDS));
        assert!(!NumericContract::RELAXED_OPERANDS.allows(NumericContract::RELAXED));
    }

    /// The licence cannot be manufactured by propagation: one unlicensed
    /// input takes it away, so no fusion can decide to quantize a value's
    /// operands on its behalf.
    #[test]
    fn meet_cannot_reach_the_licence() {
        let m = NumericContract::RELAXED_OPERANDS.meet(NumericContract::RELAXED);
        assert_eq!(m.min_operand_bits, 32);
        assert_eq!(m, NumericContract::RELAXED);
        assert_eq!(
            NumericContract::RELAXED_OPERANDS
                .meet(NumericContract::RELAXED_OPERANDS)
                .min_operand_bits,
            8
        );
        // Narrow operands say nothing about rounding: a licensed value that
        // also forbids reassociation still forbids it.
        let strict_narrow = NumericContract::RELAXED_OPERANDS.meet(NumericContract::STRICT);
        assert!(!strict_narrow.reassoc);
        assert_eq!(strict_narrow.min_operand_bits, 32);
    }

    /// Adding the axis moved nothing that was already there.
    #[test]
    fn the_rounding_axes_are_unmoved() {
        assert!(NumericContract::STRICT.allows(NumericContract::RELAXED));
        assert!(!NumericContract::RELAXED.allows(NumericContract::STRICT));
        assert_eq!(
            NumericContract::RELAXED.meet(NumericContract::STRICT),
            NumericContract::STRICT
        );
    }
}
