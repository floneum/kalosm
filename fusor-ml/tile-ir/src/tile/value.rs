use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Sub};

use crate::ir::{
    CoopFragmentId, CoopOperandRole, Expr, LocalRef, StorageView, TileBinaryOp, TileCompareOp,
    TileIndexedStoreStmt, TileLinearLoadExpr, TileLiteral, TileLoadExpr, TileStoreStmt,
    TileUnaryOp,
};

/// Handle to a cooperative-matrix accumulator local.
#[derive(Copy, Clone)]
pub struct CoopAcc<T, const ROWS: usize, const COLS: usize> {
    pub(super) local: LocalRef,
    pub(super) _ty: PhantomData<T>,
}

/// Handle to a cooperatively-loaded fragment SSA value.
#[derive(Copy, Clone)]
pub struct CoopFragment<T, const ROWS: usize, const COLS: usize> {
    pub(super) id: CoopFragmentId,
    pub(super) role: CoopOperandRole,
    pub(super) _ty: PhantomData<T>,
}

/// Iterator description passed to `TileBlock::fold`.
#[derive(Clone)]
pub struct FoldIter {
    pub(crate) count: Box<Expr>,
}

/// Construct a counted `0..count` iterator for `TileBlock::fold`.
pub fn range(count: Tile) -> FoldIter {
    FoldIter {
        count: Box::new(count.expr),
    }
}

/// Storage address for a rank-1 or rank-2 view.
pub struct Address<T, const R: usize> {
    pub(super) view: StorageView,
    pub(super) indices: [Box<Expr>; R],
    pub(super) _ty: PhantomData<T>,
}

impl<T, const R: usize> Address<T, R> {
    pub(super) fn load_expr(self, mask: Box<Expr>, fill: TileLiteral) -> Expr {
        let mut indices = self.indices.into_iter();
        match R {
            1 => {
                let fill = match self.view.buffer.element {
                    crate::ElementType::Vector { scalar, lanes } => {
                        assert!(
                            (2..=4).contains(&lanes),
                            "vector load supports 2, 3, or 4 lanes"
                        );
                        assert_eq!(
                            fill.element(),
                            scalar.element(),
                            "vector load fill scalar type mismatch"
                        );
                        Expr::ComposeVector {
                            scalar,
                            lanes,
                            values: (0..lanes).map(|_| Expr::Literal(fill)).collect(),
                        }
                    }
                    element => {
                        assert_eq!(fill.element(), element, "linear load fill type mismatch");
                        Expr::Literal(fill)
                    }
                };
                Expr::LoadLinear(TileLinearLoadExpr {
                    src: self.view,
                    index: indices.next().expect("rank-1 address has an index"),
                    mask,
                    fill: Box::new(fill),
                })
            }
            2 => Expr::Load(TileLoadExpr {
                src: crate::ir::LoadSource::Storage(self.view),
                row: indices.next().expect("rank-2 address has a row"),
                col: indices.next().expect("rank-2 address has a column"),
                mask,
                fill: Box::new(Expr::Literal(fill)),
            }),
            _ => panic!("tile storage I/O supports rank-1 and rank-2 addresses"),
        }
    }

    pub(super) fn store_stmt(self, value: Expr, mask: Box<Expr>) -> crate::ir::TileStmt {
        let mut indices = self.indices.into_iter();
        match R {
            1 => crate::ir::TileStmt::StoreIndexed(TileIndexedStoreStmt {
                dst: self.view,
                index: indices.next().expect("rank-1 address has an index"),
                value,
                mask,
            }),
            2 => crate::ir::TileStmt::Store(TileStoreStmt {
                dst: self.view,
                row: indices.next().expect("rank-2 address has a row"),
                col: indices.next().expect("rank-2 address has a column"),
                value,
                mask,
            }),
            _ => panic!("tile storage I/O supports rank-1 and rank-2 addresses"),
        }
    }
}

/// Private local handle.
pub struct Local<T> {
    pub(super) local: LocalRef,
    pub(super) _ty: PhantomData<T>,
}

pub(super) fn boxed_u32_literal(value: u32) -> Box<Expr> {
    Box::new(Expr::Literal(TileLiteral::U32(value)))
}

pub(super) fn boxed_index(value: impl Into<Tile>) -> Box<Expr> {
    Box::new(value.into().expr)
}

/// Per-lane boolean mask.
#[derive(Clone)]
pub struct Mask {
    pub(super) expr: Box<Expr>,
}

impl Mask {
    /// Mask that accepts every lane.
    pub fn all() -> Self {
        Self {
            expr: Box::new(Expr::Literal(TileLiteral::Bool(true))),
        }
    }

    /// Logical conjunction of two masks.
    pub fn and(self, rhs: impl Into<Mask>) -> Self {
        let rhs = rhs.into();
        Self {
            expr: Box::new(Expr::Binary {
                op: TileBinaryOp::LogicalAnd,
                left: self.expr,
                right: rhs.expr,
            }),
        }
    }

    /// Logical disjunction of two masks.
    pub fn or(self, rhs: impl Into<Mask>) -> Self {
        let rhs = rhs.into();
        Self {
            expr: Box::new(Expr::Binary {
                op: TileBinaryOp::LogicalOr,
                left: self.expr,
                right: rhs.expr,
            }),
        }
    }

    /// Compare this mask with another bool expression.
    pub fn eq(&self, rhs: impl Into<Tile>) -> Self {
        Tile::from(self).eq(rhs)
    }

    /// Compare this mask with another bool expression.
    pub fn ne(&self, rhs: impl Into<Tile>) -> Self {
        Tile::from(self).ne(rhs)
    }
}

impl From<Tile> for Mask {
    fn from(value: Tile) -> Self {
        Self {
            expr: Box::new(value.expr),
        }
    }
}

impl From<&Tile> for Mask {
    fn from(value: &Tile) -> Self {
        Self {
            expr: Box::new(value.expr.clone()),
        }
    }
}

impl From<Mask> for Tile {
    fn from(value: Mask) -> Self {
        Self { expr: *value.expr }
    }
}

impl From<&Mask> for Tile {
    fn from(value: &Mask) -> Self {
        Self {
            expr: (*value.expr).clone(),
        }
    }
}

/// Per-lane tile expression.
#[derive(Clone)]
pub struct Tile {
    pub(super) expr: Expr,
}

impl From<TileLiteral> for Tile {
    fn from(value: TileLiteral) -> Self {
        Self::literal(value)
    }
}

impl From<f32> for Tile {
    fn from(value: f32) -> Self {
        Self::literal(TileLiteral::f32(value))
    }
}

impl From<u32> for Tile {
    fn from(value: u32) -> Self {
        Self::literal(TileLiteral::U32(value))
    }
}

impl From<&u32> for Tile {
    fn from(value: &u32) -> Self {
        Self::literal(TileLiteral::U32(*value))
    }
}

impl From<bool> for Tile {
    fn from(value: bool) -> Self {
        Self::literal(TileLiteral::Bool(value))
    }
}

impl From<&Tile> for Tile {
    fn from(value: &Tile) -> Self {
        value.clone()
    }
}

macro_rules! tile_unary_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Apply unary `", stringify!($op), "`.")]
            pub fn $name(self) -> Self {
                self.unary(TileUnaryOp::$op)
            }
        )+
    };
}

macro_rules! tile_compare_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Compare two tile expressions with `", stringify!($op), "`.")]
            pub fn $name(&self, rhs: impl Into<Tile>) -> Mask {
                Mask::from(Self::compare_bool(TileCompareOp::$op, self.clone(), rhs.into()))
            }
        )+
    };
}

macro_rules! tile_binary_methods {
    ($($name:ident => $op:ident),+ $(,)?) => {
        $(
            #[doc = concat!("Apply binary `", stringify!($op), "`.")]
            pub fn $name(self, rhs: impl Into<Tile>) -> Self {
                self.binary(TileBinaryOp::$op, rhs.into())
            }
        )+
    };
}

impl Tile {
    /// Build a tile literal.
    pub fn literal(value: impl Into<TileLiteral>) -> Self {
        Self {
            expr: Expr::Literal(value.into()),
        }
    }

    /// Build a tile from an index expression.
    pub fn from_index(index: impl Into<Tile>) -> Self {
        index.into()
    }

    /// Stable structural hash of this tile's expression tree.
    pub fn signature_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.expr.hash(&mut hasher);
        hasher.finish()
    }

    /// Apply an arbitrary unary operation.
    pub fn unary(self, op: TileUnaryOp) -> Self {
        Self {
            expr: Expr::Unary {
                op,
                value: Box::new(self.expr),
            },
        }
    }

    tile_unary_methods!(exp => Exp, inverse_sqrt => InverseSqrt, exp2 => Exp2, tanh => Tanh, neg_unary => Neg);

    /// Sigmoid activation: `1 / (1 + exp(-x))`.
    pub fn sigmoid(self) -> Self {
        let one = Tile::literal(1.0);
        one.clone() / (one + self.neg_unary().exp())
    }

    /// SiLU (a.k.a. swish) activation: `x * sigmoid(x)`.
    pub fn silu(self) -> Self {
        self.clone() * self.sigmoid()
    }

    /// GELU activation, tanh approximation.
    pub fn gelu(self) -> Self {
        let half = Tile::literal(0.5);
        let one = Tile::literal(1.0);
        let coeff = Tile::literal(0.044_715);
        let sqrt_2_over_pi = Tile::literal(0.797_884_6);
        let x = self;
        let x_cubed = x.clone() * x.clone() * x.clone();
        let inner = sqrt_2_over_pi * (x.clone() + coeff * x_cubed);
        half * x * (one + inner.tanh())
    }

    /// ReLU activation: `max(x, 0)`.
    pub fn relu(self) -> Self {
        let zero = Tile::literal(0.0);
        let condition = Tile::compare_bool(TileCompareOp::Gt, self.clone(), zero.clone());
        Tile::select(condition, self, zero)
    }

    /// Cast to another element type.
    pub fn cast(self, to: crate::ElementType) -> Self {
        Self {
            expr: Expr::Cast {
                value: Box::new(self.expr),
                to,
            },
        }
    }

    /// Bitcast to another element type.
    pub fn bitcast(self, to: crate::ElementType) -> Self {
        Self {
            expr: Expr::Bitcast {
                value: Box::new(self.expr),
                to,
            },
        }
    }

    /// Select between two tile values.
    pub fn select(condition: impl Into<Tile>, accept: Self, reject: Self) -> Self {
        let condition = condition.into();
        Self {
            expr: Expr::Select {
                condition: Box::new(condition.expr),
                accept: Box::new(accept.expr),
                reject: Box::new(reject.expr),
            },
        }
    }

    /// Compare two tiles producing a `Bool`-typed tile, then optionally
    /// broadcast `1`/`0` of `output`'s element type.
    pub fn compare(op: TileCompareOp, left: Self, right: Self, output: crate::ElementType) -> Self {
        let condition = Self::compare_bool(op, left, right);
        if output == crate::ElementType::Bool {
            condition
        } else {
            let one = Tile::literal(1.0).cast(output);
            let zero = Tile::literal(0.0).cast(output);
            Self::select(condition, one, zero)
        }
    }

    /// Compare two tiles and produce a bool tile.
    pub fn compare_bool(op: TileCompareOp, left: Self, right: Self) -> Self {
        Self {
            expr: Expr::Compare {
                op,
                left: Box::new(left.expr),
                right: Box::new(right.expr),
            },
        }
    }

    tile_compare_methods!(lt => Lt, le => Le, gt => Gt, ge => Ge, eq => Eq, ne => Ne);

    /// Apply an arbitrary binary operation.
    pub fn binary(self, op: TileBinaryOp, rhs: Self) -> Self {
        Tile {
            expr: Expr::Binary {
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

macro_rules! impl_tile_binary {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<Rhs> $trait<Rhs> for Tile
        where
            Rhs: Into<Tile>,
        {
            type Output = Tile;

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
