use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Sub};

use crate::ir::{
    Bool, CoopFragmentId, CoopOperandRole, ElementType, Expr, LocalRef, Numeric, StorageView,
    TileBinaryOp, TileCompareOp, TileIndexedStoreStmt, TileLinearLoadExpr, TileLiteral,
    TileLoadExpr, TileRef, TileStoreStmt, TileUnaryOp, F16, F32, U32,
};

#[derive(Copy, Clone)]
pub struct CoopAcc<T, const ROWS: usize, const COLS: usize> {
    pub(super) local: LocalRef,
    pub(super) _ty: PhantomData<T>,
}

#[derive(Copy, Clone)]
pub struct CoopFragment<T, const ROWS: usize, const COLS: usize> {
    pub(super) id: CoopFragmentId,
    pub(super) role: CoopOperandRole,
    pub(super) _ty: PhantomData<T>,
}

#[derive(Clone)]
pub struct FoldIter {
    pub(crate) count: Box<Expr>,
}

pub fn range(count: impl Into<Tile<U32>>) -> FoldIter {
    FoldIter {
        count: Box::new(count.into().expr),
    }
}

pub struct Address<T, const R: usize> {
    pub(super) view: StorageView,
    pub(super) indices: [Box<Expr>; R],
    pub(super) _ty: PhantomData<T>,
}

impl<T, const R: usize> Address<T, R> {
    pub(super) fn load_expr(self, mask: Expr, fill: TileLiteral) -> Expr {
        let mut indices = self.indices.into_iter();
        match R {
            1 => {
                let fill = match self.view.buffer.element {
                    ElementType::Vector { scalar, lanes } => {
                        assert!((2..=4).contains(&lanes));
                        assert_eq!(fill.element(), scalar.element());
                        Expr::ComposeVector {
                            scalar,
                            lanes,
                            values: (0..lanes).map(|_| Expr::Literal(fill)).collect(),
                        }
                    }
                    element => {
                        assert_eq!(fill.element(), element);
                        Expr::Literal(fill)
                    }
                };
                Expr::LoadLinear(TileLinearLoadExpr {
                    src: self.view,
                    index: indices.next().expect("rank-1 address has an index"),
                    mask: Box::new(mask),
                    fill: Box::new(fill),
                })
            }
            2 => Expr::Load(TileLoadExpr {
                src: crate::ir::LoadSource::Storage(self.view),
                row: indices.next().expect("rank-2 address has a row"),
                col: indices.next().expect("rank-2 address has a column"),
                mask: Box::new(mask),
                fill: Box::new(Expr::Literal(fill)),
            }),
            _ => panic!("tile storage I/O supports rank-1 and rank-2 addresses"),
        }
    }

    pub(super) fn store_stmt(self, value: Expr, mask: Expr) -> crate::ir::TileStmt {
        let mut indices = self.indices.into_iter();
        match R {
            1 => crate::ir::TileStmt::StoreIndexed(TileIndexedStoreStmt {
                dst: self.view,
                index: indices.next().expect("rank-1 address has an index"),
                value,
                mask: Box::new(mask),
            }),
            2 => crate::ir::TileStmt::Store(TileStoreStmt {
                dst: self.view,
                row: indices.next().expect("rank-2 address has a row"),
                col: indices.next().expect("rank-2 address has a column"),
                value,
                mask: Box::new(mask),
            }),
            _ => panic!("tile storage I/O supports rank-1 and rank-2 addresses"),
        }
    }
}

pub struct Local<T> {
    pub(super) local: LocalRef,
    pub(super) _ty: PhantomData<T>,
}

pub struct Workgroup<T> {
    pub(super) tile: TileRef,
    pub(super) _ty: PhantomData<T>,
}

impl<T> Copy for Workgroup<T> {}

impl<T> Clone for Workgroup<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> From<Workgroup<T>> for TileRef {
    fn from(value: Workgroup<T>) -> Self {
        value.tile
    }
}

pub(super) fn boxed_u32_literal(value: u32) -> Box<Expr> {
    Box::new(Expr::Literal(TileLiteral::U32(value)))
}

pub(super) fn boxed_index(value: impl Into<Tile<U32>>) -> Box<Expr> {
    Box::new(value.into().expr)
}

pub type Mask = Tile<Bool>;

pub struct Tile<T: Numeric = F32> {
    pub(super) expr: Expr,
    pub(super) _ty: PhantomData<T>,
}

impl<T: Numeric> Clone for Tile<T> {
    fn clone(&self) -> Self {
        Self {
            expr: self.expr.clone(),
            _ty: PhantomData,
        }
    }
}

impl<T: Numeric> Tile<T> {
    pub(super) fn from_expr(expr: Expr) -> Self {
        debug_assert_eq!(expr.element(), T::ELEMENT, "typed tile element mismatch");
        Self {
            expr,
            _ty: PhantomData,
        }
    }

    pub fn literal(value: impl Into<TileLiteral>) -> Self {
        Self::from_expr(Expr::Literal(value.into()))
    }

    pub fn signature_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        self.expr.hash(&mut hasher);
        hasher.finish()
    }

    pub fn unary(self, op: TileUnaryOp) -> Self {
        Self::from_expr(Expr::Unary {
            op,
            value: Box::new(self.expr),
        })
    }

    pub fn cast<To: Numeric>(self) -> Tile<To> {
        Tile::from_expr(Expr::Cast {
            value: Box::new(self.expr),
            to: To::ELEMENT,
        })
    }

    pub fn bitcast<To: Numeric>(self) -> Tile<To> {
        Tile::from_expr(Expr::Bitcast {
            value: Box::new(self.expr),
            to: To::ELEMENT,
        })
    }

    pub fn select(condition: Mask, accept: Self, reject: Self) -> Self {
        Self::from_expr(Expr::Select {
            condition: Box::new(condition.expr),
            accept: Box::new(accept.expr),
            reject: Box::new(reject.expr),
        })
    }

    pub fn compare_bool(op: TileCompareOp, left: Self, right: Self) -> Mask {
        Tile::<Bool>::from_expr(Expr::Compare {
            op,
            left: Box::new(left.expr),
            right: Box::new(right.expr),
        })
    }

    pub fn lt(&self, rhs: impl Into<Tile<T>>) -> Mask {
        Self::compare_bool(TileCompareOp::Lt, self.clone(), rhs.into())
    }

    pub fn le(&self, rhs: impl Into<Tile<T>>) -> Mask {
        Self::compare_bool(TileCompareOp::Le, self.clone(), rhs.into())
    }

    pub fn gt(&self, rhs: impl Into<Tile<T>>) -> Mask {
        Self::compare_bool(TileCompareOp::Gt, self.clone(), rhs.into())
    }

    pub fn ge(&self, rhs: impl Into<Tile<T>>) -> Mask {
        Self::compare_bool(TileCompareOp::Ge, self.clone(), rhs.into())
    }

    pub fn eq(&self, rhs: impl Into<Tile<T>>) -> Mask {
        Self::compare_bool(TileCompareOp::Eq, self.clone(), rhs.into())
    }

    pub fn ne(&self, rhs: impl Into<Tile<T>>) -> Mask {
        Self::compare_bool(TileCompareOp::Ne, self.clone(), rhs.into())
    }

    pub fn binary(self, op: TileBinaryOp, rhs: Self) -> Self {
        Self::from_expr(Expr::Binary {
            op,
            left: Box::new(self.expr),
            right: Box::new(rhs.expr),
        })
    }

    pub fn max(self, rhs: impl Into<Tile<T>>) -> Self {
        self.binary(TileBinaryOp::Max, rhs.into())
    }

    pub fn min(self, rhs: impl Into<Tile<T>>) -> Self {
        self.binary(TileBinaryOp::Min, rhs.into())
    }
}

impl Tile<F32> {
    pub fn exp(self) -> Self {
        self.unary(TileUnaryOp::Exp)
    }
    pub fn exp2(self) -> Self {
        self.unary(TileUnaryOp::Exp2)
    }
    pub fn tanh(self) -> Self {
        self.unary(TileUnaryOp::Tanh)
    }
    pub fn inverse_sqrt(self) -> Self {
        self.unary(TileUnaryOp::InverseSqrt)
    }
    pub fn neg_unary(self) -> Self {
        self.unary(TileUnaryOp::Neg)
    }
    pub fn sigmoid(self) -> Self {
        let one = Self::literal(1.0);
        one.clone() / (one + self.neg_unary().exp())
    }
    pub fn silu(self) -> Self {
        self.clone() * self.sigmoid()
    }
    pub fn gelu(self) -> Self {
        let half = Self::literal(0.5);
        let one = Self::literal(1.0);
        let coeff = Self::literal(0.044_715);
        let sqrt_2_over_pi = Self::literal(0.797_884_6);
        let x = self;
        let x_cubed = x.clone() * x.clone() * x.clone();
        let inner = sqrt_2_over_pi * (x.clone() + coeff * x_cubed);
        half * x * (one + inner.tanh())
    }
    pub fn relu(self) -> Self {
        let zero = Self::literal(0.0);
        let condition = self.gt(zero.clone());
        Self::select(condition, self, zero)
    }
}

impl Tile<F16> {
    pub fn literal_bits(value: u16) -> Self {
        Self::from_expr(Expr::Literal(TileLiteral::F16(value)))
    }
}

impl Tile<U32> {
    pub fn from_index(index: impl Into<Tile<U32>>) -> Self {
        index.into()
    }
    pub fn bit_and(self, rhs: impl Into<Tile<U32>>) -> Self {
        self.binary(TileBinaryOp::BitAnd, rhs.into())
    }
    pub fn bit_or(self, rhs: impl Into<Tile<U32>>) -> Self {
        self.binary(TileBinaryOp::BitOr, rhs.into())
    }
    pub fn bit_xor(self, rhs: impl Into<Tile<U32>>) -> Self {
        self.binary(TileBinaryOp::BitXor, rhs.into())
    }
}

impl Tile<Bool> {
    pub fn all() -> Self {
        Self::literal(TileLiteral::Bool(true))
    }
    pub fn and(self, rhs: impl Into<Tile<Bool>>) -> Self {
        self.binary(TileBinaryOp::LogicalAnd, rhs.into())
    }
    pub fn or(self, rhs: impl Into<Tile<Bool>>) -> Self {
        self.binary(TileBinaryOp::LogicalOr, rhs.into())
    }
}

impl From<TileLiteral> for Tile {
    fn from(value: TileLiteral) -> Self {
        Self::literal(value)
    }
}

impl From<f32> for Tile<F32> {
    fn from(value: f32) -> Self {
        Self::literal(TileLiteral::f32(value))
    }
}

impl From<u32> for Tile<U32> {
    fn from(value: u32) -> Self {
        Self::literal(TileLiteral::U32(value))
    }
}

impl From<i32> for Tile<U32> {
    fn from(value: i32) -> Self {
        assert!(
            value >= 0,
            "negative integer literal cannot become a U32 tile"
        );
        Self::literal(TileLiteral::U32(value as u32))
    }
}

impl From<&u32> for Tile<U32> {
    fn from(value: &u32) -> Self {
        Self::literal(TileLiteral::U32(*value))
    }
}

impl From<usize> for Tile<U32> {
    fn from(value: usize) -> Self {
        Self::literal(TileLiteral::U32(value as u32))
    }
}

impl From<bool> for Tile<Bool> {
    fn from(value: bool) -> Self {
        Self::literal(TileLiteral::Bool(value))
    }
}

impl<T: Numeric> From<&Tile<T>> for Tile<T> {
    fn from(value: &Tile<T>) -> Self {
        value.clone()
    }
}

macro_rules! impl_tile_binary {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<T, Rhs> $trait<Rhs> for Tile<T>
        where
            T: Numeric,
            Rhs: Into<Tile<T>>,
        {
            type Output = Tile<T>;
            fn $method(self, rhs: Rhs) -> Self::Output {
                self.binary($op, rhs.into())
            }
        }
    };
}

impl_tile_binary!(Add, add, TileBinaryOp::Add);
impl_tile_binary!(Sub, sub, TileBinaryOp::Sub);
impl_tile_binary!(Mul, mul, TileBinaryOp::Mul);
impl_tile_binary!(Div, div, TileBinaryOp::Div);
impl_tile_binary!(Rem, rem, TileBinaryOp::Rem);
impl_tile_binary!(BitAnd, bitand, TileBinaryOp::BitAnd);
impl_tile_binary!(BitOr, bitor, TileBinaryOp::BitOr);
impl_tile_binary!(BitXor, bitxor, TileBinaryOp::BitXor);
