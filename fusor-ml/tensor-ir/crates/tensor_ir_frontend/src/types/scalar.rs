//! Scalar value, dtype, and operator enums.

use std::fmt;
use std::str::FromStr;

use ordered_float::OrderedFloat;

/// Element data type for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    F16,
    F32,
    U32,
    I32,
    Bool,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F16 => write!(f, "f16"),
            Self::F32 => write!(f, "f32"),
            Self::U32 => write!(f, "u32"),
            Self::I32 => write!(f, "i32"),
            Self::Bool => write!(f, "bool"),
        }
    }
}

impl FromStr for DType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            "u32" => Ok(Self::U32),
            "i32" => Ok(Self::I32),
            "bool" => Ok(Self::Bool),
            _ => Err(format!("unknown dtype: {s}")),
        }
    }
}

impl DType {
    /// Storage size in bytes for one scalar of this dtype.
    #[must_use]
    pub const fn byte_size(self) -> u32 {
        match self {
            Self::F16 => 2,
            // Bool is promoted to a 32-bit word in WGSL/MSL storage.
            Self::F32 | Self::U32 | Self::I32 | Self::Bool => 4,
        }
    }
}

/// Scalar literal value.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScalarValue {
    F16(OrderedFloat<f32>),
    F32(OrderedFloat<f32>),
    I32(i32),
    U32(u32),
    Bool(bool),
}

impl fmt::Display for ScalarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F16(v) => write!(f, "{v}h"),
            Self::F32(v) => write!(f, "{v}f"),
            Self::I32(v) => write!(f, "{v}i"),
            Self::U32(v) => write!(f, "{v}u"),
            Self::Bool(v) => write!(f, "{v}"),
        }
    }
}

impl FromStr for ScalarValue {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        if s == "true" {
            return Ok(Self::Bool(true));
        }
        if s == "false" {
            return Ok(Self::Bool(false));
        }
        if let Some(rest) = s.strip_suffix('f') {
            return rest
                .parse::<f32>()
                .map(|v| Self::F32(OrderedFloat(v)))
                .map_err(|e| format!("bad f32: {e}"));
        }
        if let Some(rest) = s.strip_suffix('h') {
            return rest
                .parse::<f32>()
                .map(|v| Self::F16(OrderedFloat(v)))
                .map_err(|e| format!("bad f16: {e}"));
        }
        if let Some(rest) = s.strip_suffix('u') {
            return rest
                .parse::<u32>()
                .map(ScalarValue::U32)
                .map_err(|e| format!("bad u32: {e}"));
        }
        if let Some(rest) = s.strip_suffix('i') {
            return rest
                .parse::<i32>()
                .map(ScalarValue::I32)
                .map_err(|e| format!("bad i32: {e}"));
        }
        Err(format!("bad scalar: {s}"))
    }
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Max,
    Min,
    Pow,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for BinaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Mod => "mod",
            Self::Max => "max",
            Self::Min => "min",
            Self::Pow => "pow",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::Eq => "eq",
            Self::Neq => "neq",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        };
        write!(f, "{s}")
    }
}

impl BinaryOp {
    #[must_use]
    pub const fn is_commutative(self) -> bool {
        matches!(
            self,
            Self::Add
                | Self::Mul
                | Self::Max
                | Self::Min
                | Self::And
                | Self::Or
                | Self::Xor
                | Self::Eq
                | Self::Neq
        )
    }
}

impl FromStr for BinaryOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "add" => Ok(Self::Add),
            "sub" => Ok(Self::Sub),
            "mul" => Ok(Self::Mul),
            "div" => Ok(Self::Div),
            "mod" => Ok(Self::Mod),
            "max" => Ok(Self::Max),
            "min" => Ok(Self::Min),
            "pow" => Ok(Self::Pow),
            "and" => Ok(Self::And),
            "or" => Ok(Self::Or),
            "xor" => Ok(Self::Xor),
            "shl" => Ok(Self::Shl),
            "shr" => Ok(Self::Shr),
            "eq" => Ok(Self::Eq),
            "neq" => Ok(Self::Neq),
            "lt" => Ok(Self::Lt),
            "le" => Ok(Self::Le),
            "gt" => Ok(Self::Gt),
            "ge" => Ok(Self::Ge),
            _ => Err(format!("unknown binary op: {s}")),
        }
    }
}

/// A unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnaryOp {
    Neg,
    Not,
    Exp,
    Exp2,
    Log,
    Log2,
    Sin,
    Cos,
    Tan,
    Tanh,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Abs,
    Sqrt,
    CastF32,
    CastF16,
    CastI32,
    CastU32,
    CastBool,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Neg => "neg",
            Self::Not => "not",
            Self::Exp => "exp",
            Self::Exp2 => "exp2",
            Self::Log => "log",
            Self::Log2 => "log2",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tan => "tan",
            Self::Tanh => "tanh",
            Self::Asin => "asin",
            Self::Acos => "acos",
            Self::Atan => "atan",
            Self::Sinh => "sinh",
            Self::Cosh => "cosh",
            Self::Asinh => "asinh",
            Self::Acosh => "acosh",
            Self::Atanh => "atanh",
            Self::Abs => "abs",
            Self::Sqrt => "sqrt",
            Self::CastF32 => "cast_f32",
            Self::CastF16 => "cast_f16",
            Self::CastI32 => "cast_i32",
            Self::CastU32 => "cast_u32",
            Self::CastBool => "cast_bool",
        };
        write!(f, "{s}")
    }
}

impl FromStr for UnaryOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "neg" => Ok(Self::Neg),
            "not" => Ok(Self::Not),
            "exp" => Ok(Self::Exp),
            "exp2" => Ok(Self::Exp2),
            "log" => Ok(Self::Log),
            "log2" => Ok(Self::Log2),
            "sin" => Ok(Self::Sin),
            "cos" => Ok(Self::Cos),
            "tan" => Ok(Self::Tan),
            "tanh" => Ok(Self::Tanh),
            "asin" => Ok(Self::Asin),
            "acos" => Ok(Self::Acos),
            "atan" => Ok(Self::Atan),
            "sinh" => Ok(Self::Sinh),
            "cosh" => Ok(Self::Cosh),
            "asinh" => Ok(Self::Asinh),
            "acosh" => Ok(Self::Acosh),
            "atanh" => Ok(Self::Atanh),
            "abs" => Ok(Self::Abs),
            "sqrt" => Ok(Self::Sqrt),
            "cast_f32" => Ok(Self::CastF32),
            "cast_f16" => Ok(Self::CastF16),
            "cast_i32" => Ok(Self::CastI32),
            "cast_u32" => Ok(Self::CastU32),
            "cast_bool" => Ok(Self::CastBool),
            _ => Err(format!("unknown unary op: {s}")),
        }
    }
}

/// A ternary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TernaryOp {
    Fma,
    Select,
}

impl fmt::Display for TernaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Fma => "fma",
            Self::Select => "select",
        };
        write!(f, "{s}")
    }
}

impl FromStr for TernaryOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "fma" => Ok(Self::Fma),
            "select" => Ok(Self::Select),
            _ => Err(format!("unknown ternary op: {s}")),
        }
    }
}

/// Associative reduction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ReduceOp {
    Add,
    Mul,
    Max,
    Min,
}

impl ReduceOp {
    #[must_use]
    pub fn identity_f16(&self) -> f32 {
        match self {
            Self::Add => 0.0,
            Self::Mul => 1.0,
            Self::Max => -65_504.0,
            Self::Min => 65_504.0,
        }
    }

    #[must_use]
    pub fn identity_f32(&self) -> f32 {
        match self {
            Self::Add => 0.0,
            Self::Mul => 1.0,
            // Naga/WGPU validation rejects literal infinities in shader IR.
            // Use the largest finite sentinels instead.
            Self::Max => -f32::MAX,
            Self::Min => f32::MAX,
        }
    }

    #[must_use]
    pub const fn identity_u32(&self) -> u32 {
        match self {
            Self::Add => 0,
            Self::Mul => 1,
            Self::Max => u32::MIN,
            Self::Min => u32::MAX,
        }
    }

    #[must_use]
    pub const fn identity_i32(&self) -> i32 {
        match self {
            Self::Add => 0,
            Self::Mul => 1,
            Self::Max => i32::MIN,
            Self::Min => i32::MAX,
        }
    }

    #[must_use]
    pub const fn bin_op(&self) -> BinaryOp {
        match self {
            Self::Add => BinaryOp::Add,
            Self::Mul => BinaryOp::Mul,
            Self::Max => BinaryOp::Max,
            Self::Min => BinaryOp::Min,
        }
    }
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add => write!(f, "add"),
            Self::Mul => write!(f, "mul"),
            Self::Max => write!(f, "max"),
            Self::Min => write!(f, "min"),
        }
    }
}

impl FromStr for ReduceOp {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "add" => Ok(Self::Add),
            "mul" => Ok(Self::Mul),
            "max" => Ok(Self::Max),
            "min" => Ok(Self::Min),
            _ => Err(format!("unknown reduce op: {s}")),
        }
    }
}
