use std::marker::PhantomData;

use super::value::boxed_u32_literal;
use super::*;
use crate::ir::{
    Builtin, Expr, FloatElement, Numeric, ScalarMarker, TileLiteral, TileLoadExpr, TileRef,
    TileStmt, TileUnaryOp, WorkgroupAxis, U32,
};
use crate::quantized::QuantizedMatrix;

/// Builder for the body of one tile-program workgroup.
///
/// `BLOCK` is the number of invocations in the workgroup. Values returned by
/// methods on `TileBlock` are symbolic tile expressions, not host values.
pub struct TileBlock<'a> {
    pub(super) program: &'a mut Program,
    pub(super) grid: [u32; 3],
    pub(super) block: usize,
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
pub(super) fn tiles_to_exprs(values: impl IntoIterator<Item = Tile>) -> Vec<Expr> {
    values.into_iter().map(|t| t.expr).collect()
}

impl TileBlock<'_> {
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

    /// Number of invocations in this workgroup.
    pub fn block_size(&self) -> usize {
        self.block
    }

    /// This invocation's lane index within the workgroup (`0..BLOCK`).
    /// Lowers to `@builtin(local_invocation_index)`.
    pub fn lane(&self) -> Range {
        Range {
            expr: Box::new(Expr::Builtin(Builtin::Lane)),
        }
    }

    /// Interpret lanes as row-major coordinates for a tile with `dims`.
    pub fn lane_tiles<const DIMS: usize>(&self, dims: &[u32; DIMS]) -> [Range; DIMS] {
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
            Some(self.block),
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

    /// Load from rank-1, rank-2, or vector rank-1 storage with a masked fill.
    ///
    /// ```
    /// use fusor_tile_ir::{tile, Shape, Vector, F32};
    ///
    /// let ir = tile::build(|program| {
    ///     let x = program.storage_read::<Vector<F32, 2>, 1>(Shape::new([16]));
    ///     program.program_grid::<16>([1, 1, 1], |block| {
    ///         let lane = block.lane();
    ///         let mask = lane.clone().lt(16);
    ///         let _value = block.load(x.at(lane), mask, 0.0);
    ///     });
    /// });
    /// # let _ = ir;
    /// ```
    pub fn load<A>(&self, address: A, mask: Mask, fill: impl IntoTileLiteral) -> Tile
    where
        A: TileLoadAddress,
    {
        Tile {
            expr: address.load_expr(mask.expr, fill.into_tile_literal()),
        }
    }

    /// Load and dequantize one scalar from a quantized matrix.
    pub fn load_quantized(
        &self,
        matrix: &QuantizedMatrix,
        row: impl IntoIndex,
        col: impl IntoIndex,
        mask: Mask,
        fill: f32,
    ) -> Tile {
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
        k_base: impl IntoIndex,
        col: impl IntoIndex,
        mask: Mask,
        fill: f32,
    ) -> [Tile; N] {
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
    pub fn bind(&mut self, value: Tile) -> Bound {
        let element = value.expr.element();
        let local = self.program.alloc_local_element(element);
        self.push_stmt(TileStmt::StoreLocal {
            dst: local,
            value: value.expr,
        });
        Bound { local }
    }

    /// Dot product between two float vector tile expressions.
    pub fn vector_dot<T: FloatElement, const LANES: usize>(&self, left: Tile, right: Tile) -> Tile {
        validate_vector_lanes(LANES, "vector_dot");
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
    pub fn vector_splat<T: ScalarMarker, const LANES: usize>(&self, value: Tile) -> Tile {
        validate_vector_lanes(LANES, "vector_splat");
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
        values: [Tile; LANES],
    ) -> Tile {
        validate_vector_lanes(LANES, "compose_vector");
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
    pub fn literal(&self, value: TileLiteral) -> Tile {
        Tile {
            expr: Expr::Literal(value),
        }
    }

    /// Create an f32 tile literal.
    pub fn f32(&self, value: f32) -> Tile {
        self.literal(TileLiteral::f32(value))
    }

    /// Create a u32 tile literal.
    pub fn u32(&self, value: u32) -> Tile {
        self.literal(TileLiteral::U32(value))
    }

    /// Create a bool tile literal.
    pub fn bool(&self, value: bool) -> Tile {
        self.literal(TileLiteral::Bool(value))
    }

    /// Convert an index expression into a u32 tile expression.
    pub fn index(&self, value: impl IntoIndex) -> Tile {
        Tile {
            expr: *value.into_index(),
        }
    }

    /// Exponential of a tile expression.
    pub fn exp(&self, value: Tile) -> Tile {
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
    pub fn private<T: Numeric>(&mut self) -> Local<T> {
        Local {
            local: self.program.alloc_local::<T>(),
            _ty: PhantomData,
        }
    }

    /// Load a private local.
    pub fn load_local<T>(&self, local: &Local<T>) -> Tile {
        Tile {
            expr: Expr::LoadLocal(local.local),
        }
    }

    /// Store a private local.
    pub fn store_local<T>(&mut self, local: &Local<T>, value: Tile) {
        self.push_stmt(TileStmt::StoreLocal {
            dst: local.local,
            value: value.expr,
        });
    }

    /// Load a value from workgroup tile storage.
    pub fn load_workgroup(&self, tile: TileRef, index: impl IntoIndex) -> Tile {
        Tile {
            expr: Expr::LoadWorkgroup {
                src: tile,
                index: index.into_index(),
            },
        }
    }

    /// Store a value into workgroup tile storage.
    pub fn store_workgroup(&mut self, tile: TileRef, index: impl IntoIndex, value: Tile) {
        self.push_stmt(TileStmt::StoreWorkgroup {
            dst: tile,
            index: index.into_index(),
            value: value.expr,
        });
    }

    /// Emit `if condition { ... }`.
    pub fn if_then(&mut self, condition: Tile, accept: impl FnOnce(&mut Self)) {
        self.if_else(condition, accept, |_| {});
    }

    /// Emit `if condition { ... } else { ... }`.
    pub fn if_else(
        &mut self,
        condition: Tile,
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
    pub fn break_if(&mut self, condition: Tile) {
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

    /// Store to a rank-1 or rank-2 storage address.
    pub fn store<A>(&mut self, address: A, value: Tile, mask: Mask)
    where
        A: TileStoreAddress,
    {
        self.push_stmt(address.store_stmt(value.expr, mask.expr));
    }
}

fn validate_vector_lanes(lanes: usize, op: &str) {
    assert!((2..=4).contains(&lanes), "{op} supports 2, 3, or 4 lanes");
}
