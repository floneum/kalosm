//! Scalar dtypes, quantized block formats, numeric contracts, persistence.

use crate::scalar::Lit;

/// Element type of a Logical/Launch value. No `Bool`: comparisons return
/// 1.0/0.0 in the operand dtype.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Dtype {
    F32,
    F16,
    BF16,
    U32,
    I32,
    /// A block-quantized weight format: a quantized leaf or a `Dequant` input.
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

    /// What a storage-only narrow float widens to for compute.
    pub const fn compute_dtype(self) -> Self {
        match self {
            Self::F16 | Self::BF16 => Self::F32,
            other => other,
        }
    }
}

/// The GGUF block formats fusor ingests on both backends.
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

/// On-device byte layout of a quantized matrix; both are legal everywhere.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum QLayout {
    Native,
    F32Scales,
}

/// Rounding mode carried on `ScalarKind::Round`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RoundMode {
    HalfToEven,
    HalfAwayFromZero,
    Floor,
    Ceil,
    Trunc,
}

/// What a value's numerics permit. Monotone: no rewrite may lower the bit
/// floors or enable `reassoc`/`contract` where a value forbids it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NumericContract {
    pub min_accum_bits: u8,
    /// Narrowest operand a rewrite may substitute for this value's inputs,
    /// in bits; `32` keeps operands f32.
    pub min_operand_bits: u8,
    pub reassoc: bool,
    pub contract: bool,
}

impl NumericContract {
    /// f32 accumulation and operands, reassociation and contraction allowed.
    pub const RELAXED: Self = Self {
        min_accum_bits: 32,
        min_operand_bits: 32,
        reassoc: true,
        contract: true,
    };

    /// f32 accumulation and operands, no reassociation or contraction.
    pub const STRICT: Self = Self {
        min_accum_bits: 32,
        min_operand_bits: 32,
        reassoc: false,
        contract: false,
    };

    /// [`Self::RELAXED`] plus 8-bit operand re-encoding; unreachable by
    /// [`Self::meet`] from values that do not already carry it.
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

/// How long a value lives.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Persistence {
    Step,
    Persistent,
}

/// A typed constant, compared and hashed bitwise so the memo is exact.
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

    /// The value as `f64`, exact for every variant.
    pub fn to_f64(self) -> f64 {
        match self {
            Self::F32(v) => f64::from(v),
            Self::F16(b) => half::f16::from_bits(b).to_f64(),
            Self::BF16(b) => half::bf16::from_bits(b).to_f64(),
            Self::U32(v) => f64::from(v),
            Self::I32(v) => f64::from(v),
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
