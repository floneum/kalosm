#![allow(unused_imports)]
use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitXor, Div, Mul, Rem, Sub};

use crate::ir::{
    BlockDequantId, BufferAccess, BufferDecl, BufferRef, CoopFragmentId,
    CoopOperandRole, DynamicOffset, F32Bits, F32Vec4, Im2ColNhwcMap, KernelIr, Layout, LocalDecl,
    LocalRef, MemoryLevel, Numeric, Op,
    QuantizedVecDotKind, Shape, StorageIndexMap, StorageView, TileBinaryOp, TileCompareOp,
    TileDecl, TileExpr, TileIndexExpr, TileIndexedStoreStmt, TileLevel, TileLinearLoadExpr,
    TileLiteral, TileLoadExpr, TileMaskExpr, TileOrigin, TileProgramOp, TileQuantizedLoadExpr,
    TileReduceOp, TileRef, TileScalarExpr, TileStmt, TileStoreStmt, TileUnaryOp, TileVec4LoadExpr,
    WorkgroupAxis, WorkgroupOffset, F32, U32,
};
use crate::quantized::{GgmlQuantFormat, QuantizedMatrix};
use super::*;

/// Handle to an 8x8 cooperative-matrix accumulator local.
#[derive(Copy, Clone)]
pub struct CoopAcc {
    pub(super) local: LocalRef,
}

/// Handle to a cooperatively-loaded 8x8 fragment SSA value. Reusable across
/// any number of `coop_mma` calls in the same scope without re-loading.
#[derive(Copy, Clone)]
pub struct CoopFragment {
    pub(super) id: CoopFragmentId,
    pub(super) role: CoopOperandRole,
}

/// Handle to a bound subexpression. Each call to `get()` returns a fresh
/// `Tile` that lowers to a load from the private local that backs the binding.
/// Allocating the local and emitting the binding store happens at the call
/// site of `bind`.
#[derive(Clone, Copy)]
pub struct Bound<const BLOCK: usize> {
    pub(super) local: LocalRef,
    pub(super) _block: PhantomData<[(); BLOCK]>,
}

/// Iterator description passed to `TileBlock::fold`. Currently only counted
/// ranges are supported; future variants (chunks, strided, zip) compose into
/// the same `Fold` shape.
#[derive(Clone)]
pub struct FoldIter {
    pub(crate) iter: crate::ir::TileIter,
}

/// Construct a counted `0..count` iterator for `TileBlock::fold`.
pub fn range<const BLOCK: usize>(count: Tile<BLOCK>) -> FoldIter {
    FoldIter {
        iter: crate::ir::TileIter::Range {
            count: Box::new(count.expr),
        },
    }
}

impl<const BLOCK: usize> Bound<BLOCK> {
    pub fn get(&self) -> Tile<BLOCK> {
        Tile {
            expr: TileExpr::LoadLocal(self.local),
        }
    }
}

pub struct Address<T, const N: usize> {
    pub(super) view: StorageView,
    pub(super) row: TileIndexExpr,
    pub(super) col: TileIndexExpr,
    pub(super) _ty: PhantomData<T>,
}

pub struct LinearAddress<T, const N: usize> {
    pub(super) view: StorageView,
    pub(super) index: TileIndexExpr,
    pub(super) _ty: PhantomData<T>,
}

pub struct Local<T, const N: usize> {
    pub(super) local: LocalRef,
    pub(super) _ty: PhantomData<(T, [(); N])>,
}

pub struct ErasedAddress<const N: usize> {
    pub(super) view: StorageView,
    pub(super) row: TileIndexExpr,
    pub(super) col: TileIndexExpr,
}

#[derive(Clone)]
pub struct LaneTile2d<const ROWS: usize, const COLS: usize, const N: usize> {
    pub(super) row: Range<N>,
    pub(super) col: Range<N>,
}

impl<const ROWS: usize, const COLS: usize, const N: usize> LaneTile2d<ROWS, COLS, N> {
    pub fn row(&self) -> Range<N> {
        self.row.clone()
    }

    pub fn col(&self) -> Range<N> {
        self.col.clone()
    }
}

pub trait IntoIndex<const N: usize> {
    fn into_index(self) -> TileIndexExpr;
}

#[derive(Clone)]
pub struct ScalarIndex {
    pub(super) expr: TileIndexExpr,
}

#[derive(Clone)]
pub struct Range<const N: usize> {
    pub(super) expr: TileIndexExpr,
}

impl<const N: usize> IntoIndex<N> for ScalarIndex {
    fn into_index(self) -> TileIndexExpr {
        self.expr
    }
}

impl<const N: usize> IntoIndex<N> for &ScalarIndex {
    fn into_index(self) -> TileIndexExpr {
        self.expr.clone()
    }
}

impl<const N: usize> IntoIndex<N> for Range<N> {
    fn into_index(self) -> TileIndexExpr {
        self.expr
    }
}

impl<const N: usize> IntoIndex<N> for &Range<N> {
    fn into_index(self) -> TileIndexExpr {
        self.expr.clone()
    }
}

impl<const N: usize> IntoIndex<N> for u32 {
    fn into_index(self) -> TileIndexExpr {
        TileIndexExpr::Literal(self)
    }
}

impl<const N: usize> IntoIndex<N> for Tile<N> {
    fn into_index(self) -> TileIndexExpr {
        TileIndexExpr::Value(Box::new(self.expr))
    }
}

impl<const N: usize> IntoIndex<N> for &Tile<N> {
    fn into_index(self) -> TileIndexExpr {
        TileIndexExpr::Value(Box::new(self.expr.clone()))
    }
}

pub(super) fn index_compare<const N: usize>(left: TileIndexExpr, op: TileCompareOp, value: u32) -> Mask<N> {
    Mask {
        expr: TileMaskExpr::Compare {
            op,
            left,
            right: TileIndexExpr::Literal(value),
        },
    }
}

macro_rules! range_compare_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $name(&self, value: u32) -> Mask<N> {
                index_compare(self.expr.clone(), TileCompareOp::$op, value)
            }
        )+
    };
}

macro_rules! scalar_index_compare_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $name<const N: usize>(&self, value: u32) -> Mask<N> {
                index_compare(self.expr.clone(), TileCompareOp::$op, value)
            }
        )+
    };
}

impl<const N: usize> Range<N> {
    range_compare_methods!(lt => Lt, le => Le, gt => Gt, ge => Ge, eq => Eq);
}

impl ScalarIndex {
    scalar_index_compare_methods!(lt => Lt, le => Le, gt => Gt, ge => Ge, eq => Eq);
}

macro_rules! impl_index_u32_ops {
    (generic($($generics:tt)+), $ty:ty, $out:ty, $ctor:ident, $div_msg:literal, $mod_msg:literal) => {
        impl_index_u32_ops!(@impl [impl<$($generics)+>] $ty, $out, $ctor, $div_msg, $mod_msg);
    };
    ($ty:ty, $out:ty, $ctor:ident, $div_msg:literal, $mod_msg:literal) => {
        impl_index_u32_ops!(@impl [impl] $ty, $out, $ctor, $div_msg, $mod_msg);
    };
    (@impl [$($impl_head:tt)*] $ty:ty, $out:ty, $ctor:ident, $div_msg:literal, $mod_msg:literal) => {
        $($impl_head)* Add<u32> for $ty {
            type Output = $out;

            fn add(self, rhs: u32) -> Self::Output {
                $ctor {
                    expr: TileIndexExpr::Add(
                        Box::new(self.expr),
                        Box::new(TileIndexExpr::Literal(rhs)),
                    ),
                }
            }
        }

        $($impl_head)* Mul<u32> for $ty {
            type Output = $out;

            fn mul(self, rhs: u32) -> Self::Output {
                $ctor {
                    expr: TileIndexExpr::Mul(Box::new(self.expr), rhs),
                }
            }
        }

        $($impl_head)* Div<u32> for $ty {
            type Output = $out;

            fn div(self, rhs: u32) -> Self::Output {
                assert!(rhs > 0, $div_msg);
                $ctor {
                    expr: TileIndexExpr::Div(Box::new(self.expr), rhs),
                }
            }
        }

        $($impl_head)* BitAnd<u32> for $ty {
            type Output = $out;

            fn bitand(self, rhs: u32) -> Self::Output {
                $ctor {
                    expr: TileIndexExpr::Value(Box::new(TileExpr::Binary {
                        op: TileBinaryOp::BitAnd,
                        left: Box::new(TileExpr::Index(self.expr)),
                        right: Box::new(TileExpr::Literal(TileLiteral::U32(rhs))),
                    })),
                }
            }
        }

        $($impl_head)* BitXor<u32> for $ty {
            type Output = $out;

            fn bitxor(self, rhs: u32) -> Self::Output {
                $ctor {
                    expr: TileIndexExpr::Value(Box::new(TileExpr::Binary {
                        op: TileBinaryOp::BitXor,
                        left: Box::new(TileExpr::Index(self.expr)),
                        right: Box::new(TileExpr::Literal(TileLiteral::U32(rhs))),
                    })),
                }
            }
        }

        $($impl_head)* Rem<u32> for $ty {
            type Output = $out;

            fn rem(self, rhs: u32) -> Self::Output {
                assert!(rhs > 0, $mod_msg);
                $ctor {
                    expr: TileIndexExpr::Mod(Box::new(self.expr), rhs),
                }
            }
        }
    };
}

impl_index_u32_ops!(
    ScalarIndex,
    ScalarIndex,
    ScalarIndex,
    "scalar index divisor must be non-zero",
    "scalar index modulus must be non-zero"
);
impl_index_u32_ops!(
    generic(const N: usize),
    Range<N>,
    Range<N>,
    Range,
    "tile index divisor must be non-zero",
    "tile index modulus must be non-zero"
);

impl Add<ScalarIndex> for ScalarIndex {
    type Output = ScalarIndex;

    fn add(self, rhs: ScalarIndex) -> Self::Output {
        ScalarIndex {
            expr: TileIndexExpr::Add(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl<const N: usize> Add<Range<N>> for ScalarIndex {
    type Output = Range<N>;

    fn add(self, rhs: Range<N>) -> Self::Output {
        Range {
            expr: TileIndexExpr::Add(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl<const N: usize> Add<ScalarIndex> for Range<N> {
    type Output = Range<N>;

    fn add(self, rhs: ScalarIndex) -> Self::Output {
        Range {
            expr: TileIndexExpr::Add(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

#[derive(Clone)]
pub struct Mask<const N: usize> {
    pub(super) expr: TileMaskExpr,
}

impl<const N: usize> Mask<N> {
    pub fn all() -> Self {
        Self {
            expr: TileMaskExpr::True,
        }
    }

    pub fn and(self, rhs: Self) -> Self {
        Self {
            expr: TileMaskExpr::And(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

#[derive(Clone)]
pub struct Scalar {
    pub(super) expr: TileScalarExpr,
}

impl Scalar {
    pub fn literal(value: f32) -> Self {
        Self {
            expr: TileScalarExpr::Literal(TileLiteral::F32(F32Bits::new(value))),
        }
    }
}

#[derive(Clone)]
pub struct Tile<const N: usize> {
    pub(super) expr: TileExpr,
}

macro_rules! tile_unary_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $name(self) -> Self {
                self.unary(TileUnaryOp::$op)
            }
        )+
    };
}

macro_rules! tile_compare_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $name(self, rhs: Self) -> Self {
                Self::compare_bool(TileCompareOp::$op, self, rhs)
            }
        )+
    };
}

macro_rules! tile_binary_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            pub fn $name(self, rhs: Self) -> Self {
                self.binary(TileBinaryOp::$op, rhs)
            }
        )+
    };
}

impl<const N: usize> Tile<N> {
    pub fn literal(value: TileLiteral) -> Self {
        Self {
            expr: TileExpr::Literal(value),
        }
    }

    pub fn from_index(index: impl IntoIndex<N>) -> Self {
        Self {
            expr: TileExpr::Index(index.into_index()),
        }
    }

    pub fn unary(self, op: TileUnaryOp) -> Self {
        Self {
            expr: TileExpr::Unary {
                op,
                value: Box::new(self.expr),
            },
        }
    }

    tile_unary_methods!(exp => Exp, inverse_sqrt => InverseSqrt, exp2 => Exp2, tanh => Tanh, neg_unary => Neg);

    /// Sigmoid activation: `1 / (1 + exp(-x))`.
    pub fn sigmoid(self) -> Self {
        let one = Tile::literal(TileLiteral::F32(F32Bits::new(1.0)));
        one.clone() / (one + self.neg_unary().exp())
    }

    /// SiLU (a.k.a. swish) activation: `x * sigmoid(x)`.
    pub fn silu(self) -> Self {
        self.clone() * self.sigmoid()
    }

    /// GELU activation, tanh approximation:
    /// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`.
    pub fn gelu(self) -> Self {
        let half = Tile::literal(TileLiteral::F32(F32Bits::new(0.5)));
        let one = Tile::literal(TileLiteral::F32(F32Bits::new(1.0)));
        let coeff = Tile::literal(TileLiteral::F32(F32Bits::new(0.044_715)));
        let sqrt_2_over_pi = Tile::literal(TileLiteral::F32(F32Bits::new(0.797_884_56)));
        let x = self;
        let x_cubed = x.clone() * x.clone() * x.clone();
        let inner = sqrt_2_over_pi * (x.clone() + coeff * x_cubed);
        half * x * (one + inner.tanh())
    }

    /// ReLU activation: `max(x, 0)`.
    pub fn relu(self) -> Self {
        let zero = Tile::literal(TileLiteral::F32(F32Bits::new(0.0)));
        let condition = Tile::compare_bool(TileCompareOp::Gt, self.clone(), zero.clone());
        Tile::select(condition, self, zero)
    }

    pub fn cast(self, to: crate::ElementType) -> Self {
        Self {
            expr: TileExpr::Cast {
                value: Box::new(self.expr),
                to,
            },
        }
    }

    pub fn bitcast(self, to: crate::ElementType) -> Self {
        Self {
            expr: TileExpr::Bitcast {
                value: Box::new(self.expr),
                to,
            },
        }
    }

    pub fn select(condition: Self, accept: Self, reject: Self) -> Self {
        Self {
            expr: TileExpr::Select {
                condition: Box::new(condition.expr),
                accept: Box::new(accept.expr),
                reject: Box::new(reject.expr),
            },
        }
    }

    pub fn compare(op: TileCompareOp, left: Self, right: Self, output: crate::ElementType) -> Self {
        Self {
            expr: TileExpr::Compare {
                op,
                left: Box::new(left.expr),
                right: Box::new(right.expr),
                output,
            },
        }
    }

    pub fn compare_bool(op: TileCompareOp, left: Self, right: Self) -> Self {
        Self::compare(op, left, right, crate::ElementType::Bool)
    }

    tile_compare_methods!(lt => Lt, le => Le, gt => Gt, ge => Ge, eq => Eq, ne => Ne);

    pub fn binary(self, op: TileBinaryOp, rhs: Self) -> Self {
        Tile {
            expr: TileExpr::Binary {
                op,
                left: Box::new(self.expr),
                right: Box::new(rhs.expr),
            },
        }
    }

    tile_binary_methods!(
        max => Max,
        min => Min,
        bit_and => BitAnd,
        bit_or => BitOr,
        bit_xor => BitXor,
        and => LogicalAnd,
        or => LogicalOr,
    );
}

impl<const N: usize> From<Scalar> for Tile<N> {
    fn from(value: Scalar) -> Self {
        Self {
            expr: TileExpr::Scalar(value.expr),
        }
    }
}

macro_rules! impl_tile_binary {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<const N: usize> $trait for Tile<N> {
            type Output = Tile<N>;

            fn $method(self, rhs: Self) -> Self::Output {
                self.binary($op, rhs)
            }
        }

        impl<const N: usize> $trait<Scalar> for Tile<N> {
            type Output = Tile<N>;

            fn $method(self, rhs: Scalar) -> Self::Output {
                Tile {
                    expr: TileExpr::Binary {
                        op: $op,
                        left: Box::new(self.expr),
                        right: Box::new(TileExpr::Scalar(rhs.expr)),
                    },
                }
            }
        }
    };
}

impl_tile_binary!(Add, add, TileBinaryOp::Add);
impl_tile_binary!(Sub, sub, TileBinaryOp::Sub);
impl_tile_binary!(Mul, mul, TileBinaryOp::Mul);
impl_tile_binary!(Div, div, TileBinaryOp::Div);
impl_tile_binary!(Rem, rem, TileBinaryOp::Rem);
