use super::ElementType;

/// Floating point literal stored by bits so IR equality remains exact.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct F32Bits(pub u32);

impl F32Bits {
    /// Store an f32 by raw bits.
    pub fn new(value: f32) -> Self {
        Self(value.to_bits())
    }

    /// Recover the f32 value.
    pub fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// A typed scalar literal stored by bits so IR equality remains exact. Vector
/// constants are not literals — they are built by composing scalar literals
/// (e.g. `Expr::ComposeVector`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileLiteral {
    /// 32-bit floating point literal.
    F32(F32Bits),
    /// 16-bit floating point literal, stored as IEEE bits.
    F16(u16),
    /// 32-bit unsigned integer literal.
    U32(u32),
    /// Boolean literal.
    Bool(bool),
}

impl TileLiteral {
    /// Element type represented by this literal.
    pub const fn element(self) -> ElementType {
        match self {
            Self::F32(_) => ElementType::F32,
            Self::F16(_) => ElementType::F16,
            Self::U32(_) => ElementType::U32,
            Self::Bool(_) => ElementType::Bool,
        }
    }

    /// `TileLiteral::F32(F32Bits::new(value))` — the float-literal shape
    /// every kernel and test builds. Inlined so callers don't repeat the
    /// `F32Bits::new` wrapping each time.
    pub fn f32(value: f32) -> Self {
        Self::F32(F32Bits::new(value))
    }
}

/// Unary operation over a tile expression.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileUnaryOp {
    /// Exponential.
    Exp,
    /// Base-2 exponential.
    Exp2,
    /// Natural logarithm.
    Log,
    /// Base-2 logarithm.
    Log2,
    /// Square root.
    Sqrt,
    /// Reciprocal square root.
    InverseSqrt,
    /// Sine.
    Sin,
    /// Cosine.
    Cos,
    /// Tangent.
    Tan,
    /// Hyperbolic tangent.
    Tanh,
    /// Arc-sine.
    Asin,
    /// Arc-cosine.
    Acos,
    /// Arc-tangent.
    Atan,
    /// Hyperbolic sine.
    Sinh,
    /// Hyperbolic cosine.
    Cosh,
    /// Inverse hyperbolic sine.
    Asinh,
    /// Inverse hyperbolic cosine.
    Acosh,
    /// Inverse hyperbolic tangent.
    Atanh,
    /// Absolute value.
    Abs,
    /// Arithmetic negation.
    Neg,
}

/// Binary operation over two tile expressions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileBinaryOp {
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Remainder.
    Rem,
    /// Power.
    Pow,
    /// Minimum.
    Min,
    /// Maximum.
    Max,
    /// Bitwise and.
    BitAnd,
    /// Bitwise or.
    BitOr,
    /// Bitwise xor.
    BitXor,
    /// Logical and.
    LogicalAnd,
    /// Logical or.
    LogicalOr,
}

/// Cross-lane reduction operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileReduceOp {
    /// Sum.
    Sum,
    /// Product.
    Product,
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}

impl TileReduceOp {
    /// The binary operator that combines two values under this reduction.
    /// Used both by the kernel-builder when desugaring loop folds and by the
    /// lowerer when emitting cross-lane reduce trees.
    pub const fn binary(self) -> TileBinaryOp {
        match self {
            Self::Sum => TileBinaryOp::Add,
            Self::Product => TileBinaryOp::Mul,
            Self::Max => TileBinaryOp::Max,
            Self::Min => TileBinaryOp::Min,
        }
    }
}

/// Comparison operation over tile expressions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileCompareOp {
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
}
