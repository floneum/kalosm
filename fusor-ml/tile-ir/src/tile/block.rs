use std::marker::PhantomData;

use super::value::boxed_u32_literal;
use super::*;
use crate::ir::{
    Builtin, Expr, FloatElement, LoadSource, Numeric, ScalarMarker, TileIndexedStoreStmt,
    TileLinearLoadExpr, TileLiteral, TileLoadExpr, TileRef, TileStmt, TileStoreStmt, TileUnaryOp,
    Vector, WorkgroupAxis, U32,
};
use crate::quantized::QuantizedMatrix;

/// Builder for the body of one tile-program workgroup.
///
/// `BLOCK` is the number of invocations in the workgroup. Values returned by
/// methods on `TileBlock` are symbolic tile expressions, not host values.
pub struct TileBlock<'a, const BLOCK: usize> {
    pub(super) program: &'a mut Program,
    pub(super) grid: [u32; 3],
    pub(super) body: Vec<TileStmt>,
    /// Stack of nested statement builders. The innermost frame collects
    /// statements emitted inside `while_true`/`fold` closures; popped into
    /// the enclosing loop's body on closure exit.
    pub(super) stmt_stack: Vec<Vec<TileStmt>>,
}

/// Wrap a `Builtin` as a `ScalarIndex` (u32-typed scalar). All `program.*_id`
/// / `subgroup_*` getters are one-line wrappers over this.
fn builtin_index(builtin: Builtin) -> ScalarIndex {
    ScalarIndex {
        expr: Box::new(Expr::Builtin(builtin)),
    }
}

/// Boxed `f32` fill literal used by every `load*` entry point's `fill` field.
pub(super) fn f32_fill(value: f32) -> Box<Expr> {
    Box::new(Expr::Literal(TileLiteral::f32(value)))
}

/// Unwrap an iterable of `Tile`s into the parallel `Expr` vector. Used by
/// the quantized-dot entry points that pack tile arrays into
/// `PackedActivations` variants and by `sum`/`fold` that move the underlying
/// expressions out of their typed wrappers.
pub(super) fn tiles_to_exprs<const BLOCK: usize>(
    values: impl IntoIterator<Item = Tile<BLOCK>>,
) -> Vec<Expr> {
    values.into_iter().map(|t| t.expr).collect()
}

impl<const BLOCK: usize> TileBlock<'_, BLOCK> {
    /// Workgroup id component for `axis`.
    pub fn program_id(&self, axis: WorkgroupAxis) -> ScalarIndex {
        builtin_index(Builtin::ProgramId(axis))
    }

    /// Subgroup id within the workgroup.
    pub fn subgroup_id(&self) -> ScalarIndex {
        builtin_index(Builtin::SubgroupId)
    }

    /// Lane id within the subgroup.
    pub fn subgroup_lane(&self) -> ScalarIndex {
        builtin_index(Builtin::SubgroupLane)
    }

    /// Runtime subgroup size.
    pub fn subgroup_size(&self) -> ScalarIndex {
        builtin_index(Builtin::SubgroupSize)
    }

    /// Number of subgroups in the workgroup.
    pub fn num_subgroups(&self) -> ScalarIndex {
        builtin_index(Builtin::NumSubgroups)
    }

    /// Dispatch grid for this tile program.
    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    /// This invocation's lane index within the workgroup (`0..BLOCK`).
    /// Lowers to `@builtin(local_invocation_index)`.
    pub fn lane(&self) -> Range<BLOCK> {
        Range {
            expr: Box::new(Expr::Builtin(Builtin::Lane)),
        }
    }

    /// Interpret lanes as row-major coordinates for a tile with `dims`.
    pub fn lane_tiles<const DIMS: usize>(&self, dims: &[u32; DIMS]) -> [Range<BLOCK>; DIMS] {
        assert!(
            DIMS > 0,
            "lane tile coordinates require at least one dimension"
        );
        let lane_count = dims.iter().try_fold(1usize, |product, &dim| {
            assert!(dim > 0, "lane tile dimensions must be non-zero");
            product.checked_mul(dim as usize)
        });
        assert_eq!(
            lane_count,
            Some(BLOCK),
            "lane tile shape must match the tile program block size"
        );

        let lane = self.lane();
        std::array::from_fn(|axis| {
            let stride = dims[axis + 1..].iter().product::<u32>();
            let coord = if stride == 1 {
                lane.clone()
            } else {
                lane.clone() / stride
            };
            coord % dims[axis]
        })
    }

    /// Load a rank-2 storage element with a masked fill value.
    pub fn load<T>(&self, address: Address<T, BLOCK>, mask: Mask<BLOCK>, fill: f32) -> Tile<BLOCK> {
        Tile {
            expr: Expr::Load(TileLoadExpr {
                src: LoadSource::Storage(address.view),
                row: address.row,
                col: address.col,
                mask: mask.expr,
                fill: f32_fill(fill),
            }),
        }
    }

    /// Load a rank-2 storage element with a masked fill literal.
    pub fn load_literal<T>(
        &self,
        address: Address<T, BLOCK>,
        mask: Mask<BLOCK>,
        fill: TileLiteral,
    ) -> Tile<BLOCK> {
        Tile {
            expr: Expr::Load(TileLoadExpr {
                src: LoadSource::Storage(address.view),
                row: address.row,
                col: address.col,
                mask: mask.expr,
                fill: Box::new(Expr::Literal(fill)),
            }),
        }
    }

    /// Load a rank-1 storage element with a masked fill literal.
    pub fn load_linear<T: Numeric>(
        &self,
        address: LinearAddress<T, BLOCK>,
        mask: Mask<BLOCK>,
        fill: TileLiteral,
    ) -> Tile<BLOCK> {
        Tile {
            expr: Expr::LoadLinear(TileLinearLoadExpr {
                src: address.view,
                index: address.index,
                mask: mask.expr,
                fill: Box::new(Expr::Literal(fill)),
            }),
        }
    }

    /// Load a vector from a vector-typed rank-1 storage view.
    ///
    /// ```
    /// use fusor_tile_ir::{tile, Shape, TileLiteral, Vector, F32};
    ///
    /// let ir = tile::build(|program| {
    ///     let x = program.storage_read::<Vector<F32, 2>, 1>(Shape::new([16]));
    ///     program.program_grid::<16>([1, 1, 1], |block| {
    ///         let lane = block.lane();
    ///         let mask = lane.clone().lt(16);
    ///         let _value = block.load_vector::<F32, 2>(x.at(lane), mask, TileLiteral::f32(0.0));
    ///     });
    /// });
    /// # let _ = ir;
    /// ```
    pub fn load_vector<T: ScalarMarker, const LANES: usize>(
        &self,
        address: LinearAddress<Vector<T, LANES>, BLOCK>,
        mask: Mask<BLOCK>,
        fill: TileLiteral,
    ) -> Tile<BLOCK> {
        let scalar = Tile::literal(fill);
        let fill_vector = self.vector_splat::<T, LANES>(scalar).expr;
        Tile {
            expr: Expr::LoadLinear(TileLinearLoadExpr {
                src: address.view,
                index: address.index,
                mask: mask.expr,
                fill: Box::new(fill_vector),
            }),
        }
    }

    /// Load and dequantize one scalar from a quantized matrix.
    pub fn load_quantized(
        &self,
        matrix: &QuantizedMatrix,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        Tile {
            expr: Expr::Load(TileLoadExpr {
                src: crate::ir::LoadSource::Quantized(matrix.clone()),
                row: row.into_index(),
                col: col.into_index(),
                mask: mask.expr,
                fill: f32_fill(fill),
            }),
        }
    }

    /// Load N consecutive dequantized values from one column of a packed
    /// quantized matrix. The lowerer emits a format-specific helper when one
    /// exists, otherwise it lowers the same block-shaped request as N scalar
    /// dequantizations. Each lane is bound to a private local that subsequent
    /// references load. `k_base` must be aligned to N so the values cover one
    /// scale block.
    pub fn load_quantized_block<const N: usize>(
        &mut self,
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> [Tile<BLOCK>; N] {
        assert!(
            N == 8 || N == 16,
            "load_quantized_block currently supports N == 8 or N == 16"
        );
        let id = self.program.next_block_dequant_id();
        let k_base = k_base.into_index();
        let col = col.into_index();
        let mask_expr = mask.expr;
        let fill_expr: Box<Expr> = f32_fill(fill);
        std::array::from_fn(|lane| Tile {
            expr: Expr::QuantizedBlockLane {
                id,
                src: matrix.clone(),
                k_base: k_base.clone(),
                col: col.clone(),
                mask: mask_expr.clone(),
                fill: fill_expr.clone(),
                block_n: N as u32,
                lane: lane as u32,
            },
        })
    }

    /// Bind a subexpression to a private local so subsequent references reuse
    /// the value without re-emitting its computation. Pushes a
    /// `TileStmt::StoreLocal` at the call site; subsequent `Bound::get()` calls
    /// return tiles that lower to a load of the bound local.
    pub fn bind(&mut self, value: Tile<BLOCK>) -> Bound<BLOCK> {
        let element = value.expr.element();
        let local = self.program.alloc_local_element(element);
        self.push_stmt(TileStmt::StoreLocal {
            dst: local,
            value: value.expr,
        });
        Bound {
            local,
            _block: PhantomData,
        }
    }

    /// Dot product between two float vector tile expressions.
    pub fn vector_dot<T: FloatElement, const LANES: usize>(
        &self,
        left: Tile<BLOCK>,
        right: Tile<BLOCK>,
    ) -> Tile<BLOCK> {
        assert!(
            (2..=4).contains(&LANES),
            "vector_dot supports 2, 3, or 4 lanes"
        );
        Tile {
            expr: Expr::VectorDot {
                scalar: T::SCALAR,
                lanes: LANES as u32,
                left: Box::new(left.expr),
                right: Box::new(right.expr),
            },
        }
    }

    /// Build a vector by repeating one scalar tile expression.
    pub fn vector_splat<T: ScalarMarker, const LANES: usize>(
        &self,
        value: Tile<BLOCK>,
    ) -> Tile<BLOCK> {
        assert!(
            (2..=4).contains(&LANES),
            "vector_splat supports 2, 3, or 4 lanes"
        );
        assert_eq!(
            value.expr.element(),
            T::SCALAR.element(),
            "vector_splat scalar type mismatch"
        );
        Tile {
            expr: Expr::ComposeVector {
                scalar: T::SCALAR,
                lanes: LANES as u32,
                values: (0..LANES).map(|_| value.expr.clone()).collect(),
            },
        }
    }

    /// Pack scalars into a vector.
    pub fn compose_vector<T: ScalarMarker, const LANES: usize>(
        &self,
        values: [Tile<BLOCK>; LANES],
    ) -> Tile<BLOCK> {
        assert!(
            (2..=4).contains(&LANES),
            "compose_vector supports 2, 3, or 4 lanes"
        );
        let values = values
            .into_iter()
            .map(|value| {
                assert_eq!(
                    value.expr.element(),
                    T::SCALAR.element(),
                    "compose_vector scalar type mismatch"
                );
                value.expr
            })
            .collect();
        Tile {
            expr: Expr::ComposeVector {
                scalar: T::SCALAR,
                lanes: LANES as u32,
                values,
            },
        }
    }

    /// Create a tile literal.
    pub fn literal(&self, value: TileLiteral) -> Tile<BLOCK> {
        Tile {
            expr: Expr::Literal(value),
        }
    }

    /// Create an f32 tile literal.
    pub fn f32(&self, value: f32) -> Tile<BLOCK> {
        self.literal(TileLiteral::f32(value))
    }

    /// Create a u32 tile literal.
    pub fn u32(&self, value: u32) -> Tile<BLOCK> {
        self.literal(TileLiteral::U32(value))
    }

    /// Create a bool tile literal.
    pub fn bool(&self, value: bool) -> Tile<BLOCK> {
        self.literal(TileLiteral::Bool(value))
    }

    /// Convert an index expression into a u32 tile expression.
    pub fn index(&self, value: impl IntoIndex<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: *value.into_index(),
        }
    }

    /// Exponential of a tile expression.
    pub fn exp(&self, value: Tile<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: Expr::Unary {
                op: TileUnaryOp::Exp,
                value: Box::new(value.expr),
            },
        }
    }

    /// Emit a workgroup memory barrier.
    pub fn workgroup_barrier(&mut self) {
        self.push_stmt(TileStmt::Barrier);
    }

    /// Allocate a private per-invocation local.
    pub fn private<T: Numeric>(&mut self) -> Local<T, BLOCK> {
        Local {
            local: self.program.alloc_local::<T>(),
            _ty: PhantomData,
        }
    }

    /// Load a private local.
    pub fn load_local<T>(&self, local: &Local<T, BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: Expr::LoadLocal(local.local),
        }
    }

    /// Store a private local.
    pub fn store_local<T>(&mut self, local: &Local<T, BLOCK>, value: Tile<BLOCK>) {
        self.push_stmt(TileStmt::StoreLocal {
            dst: local.local,
            value: value.expr,
        });
    }

    /// Load a value from workgroup tile storage.
    pub fn load_workgroup(&self, tile: TileRef, index: impl IntoIndex<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: Expr::LoadWorkgroup {
                src: tile,
                index: index.into_index(),
            },
        }
    }

    /// Store a value into workgroup tile storage.
    pub fn store_workgroup(
        &mut self,
        tile: TileRef,
        index: impl IntoIndex<BLOCK>,
        value: Tile<BLOCK>,
    ) {
        self.push_stmt(TileStmt::StoreWorkgroup {
            dst: tile,
            index: index.into_index(),
            value: value.expr,
        });
    }

    /// Emit `if condition { ... }`.
    pub fn if_then(&mut self, condition: Tile<BLOCK>, accept: impl FnOnce(&mut Self)) {
        self.if_else(condition, accept, |_| {});
    }

    /// Emit `if condition { ... } else { ... }`.
    pub fn if_else(
        &mut self,
        condition: Tile<BLOCK>,
        accept: impl FnOnce(&mut Self),
        reject: impl FnOnce(&mut Self),
    ) {
        self.stmt_stack.push(Vec::new());
        accept(self);
        let accept = self.stmt_stack.pop().expect("if accept frame missing");
        self.stmt_stack.push(Vec::new());
        reject(self);
        let reject = self.stmt_stack.pop().expect("if reject frame missing");
        self.push_stmt(TileStmt::If {
            condition: condition.expr,
            accept,
            reject,
        });
    }

    /// Emit an unstructured loop.
    pub fn loop_forever(&mut self, body: impl FnOnce(&mut Self)) {
        self.stmt_stack.push(Vec::new());
        body(self);
        let body = self.stmt_stack.pop().expect("loop frame missing");
        self.push_stmt(TileStmt::Loop { body });
    }

    /// Break out of the innermost loop.
    pub fn break_loop(&mut self) {
        self.push_stmt(TileStmt::Break);
    }

    /// `if condition { break }` — the dominant kernel-side pattern for
    /// counted/conditional `loop_forever` exits. Bare `break_loop` remains for
    /// the unconditional case and for compound break bodies.
    pub fn break_if(&mut self, condition: Tile<BLOCK>) {
        self.if_then(condition, |program| program.break_loop());
    }

    /// Return from the kernel.
    pub fn return_(&mut self) {
        self.push_stmt(TileStmt::Return);
    }

    /// Emit a counted `while true` loop. The body closure receives a
    /// `ScalarIndex` bound to this loop's iteration counter; nested loops each
    /// own their own index. Desugars into a `TileStmt::Fold` over a counted
    /// range with no accumulators — the AST has no dedicated counted loop
    /// variant.
    pub fn while_true<F>(&mut self, max_iterations: u32, body: F)
    where
        F: FnOnce(&mut Self, ScalarIndex),
    {
        assert!(
            max_iterations > 0,
            "while_true max_iterations must be non-zero"
        );
        let iter_var_local = self.program.alloc_local::<U32>();
        self.stmt_stack.push(Vec::new());
        let iter_index = ScalarIndex {
            expr: Box::new(Expr::LoadLocal(iter_var_local)),
        };
        body(self, iter_index);
        let stmts = self.stmt_stack.pop().expect("while_true frame missing");
        self.push_stmt(TileStmt::Fold {
            count: boxed_u32_literal(max_iterations),
            iter_var: iter_var_local.id,
            body: stmts,
            accumulators: Vec::new(),
        });
    }

    pub(super) fn push_stmt(&mut self, stmt: TileStmt) {
        if let Some(frame) = self.stmt_stack.last_mut() {
            frame.push(stmt);
        } else {
            self.body.push(stmt);
        }
    }

    /// Store to a rank-2 storage address.
    pub fn store<T>(&mut self, address: Address<T, BLOCK>, value: Tile<BLOCK>, mask: Mask<BLOCK>) {
        self.push_stmt(TileStmt::Store(TileStoreStmt {
            dst: address.view,
            row: address.row,
            col: address.col,
            value: value.expr,
            mask: mask.expr,
        }));
    }

    /// Store to a rank-1 storage address.
    pub fn store_linear<T: Numeric>(
        &mut self,
        address: LinearAddress<T, BLOCK>,
        value: Tile<BLOCK>,
        mask: Mask<BLOCK>,
    ) {
        self.push_stmt(TileStmt::StoreIndexed(TileIndexedStoreStmt {
            dst: address.view,
            index: address.index,
            value: value.expr,
            mask: mask.expr,
        }));
    }
}
