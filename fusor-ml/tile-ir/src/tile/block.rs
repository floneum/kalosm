
use std::marker::PhantomData;

use crate::ir::{
    Builtin, CoopOperandRole, DotK, ElementType, Expr, F32Bits, F32Vec4, Layout, LocalRef,
    MemoryLevel, Numeric, PackedActivations, Shape, TileBinaryOp, TileIndexedStoreStmt,
    LoadSource, TileLinearLoadExpr, TileLiteral, TileLoadExpr, TileReduceOp, TileRef,
    TileStmt, TileStoreStmt, TileUnaryOp, WorkgroupAxis, F32, U32,
};
use crate::quantized::QuantizedMatrix;
use super::*;

macro_rules! tile_reduce_entrypoints {
    ($(($reduce:ident, $loop_reduce:ident, $group_reduce:ident, $subgroup_reduce:ident, $op:ident)),+ $(,)?) => {
        $(
            pub fn $reduce(&mut self, value: Tile<BLOCK>) -> Scalar {
                self.reduce(TileReduceOp::$op, value)
            }

            pub fn $loop_reduce(&mut self, iterations: u32, value: Tile<BLOCK>) -> Scalar {
                self.loop_reduce(TileReduceOp::$op, iterations, value)
            }

            pub fn $group_reduce<const GROUP: usize>(&mut self, value: Tile<BLOCK>) -> Tile<BLOCK> {
                self.group_reduce::<GROUP>(TileReduceOp::$op, value)
            }

            pub fn $subgroup_reduce(&self, value: Tile<BLOCK>) -> Tile<BLOCK> {
                self.subgroup_reduce(TileReduceOp::$op, value)
            }
        )+
    };
}

macro_rules! quantized_vec_dot_entrypoint {
    ($name:ident, $packing:ident, [$($n:literal),+ $(,)?], $msg:literal) => {
        pub fn $name<const N: usize>(
            &self,
            a: [Tile<BLOCK>; N],
            matrix: &QuantizedMatrix,
            k_base: impl IntoIndex<BLOCK>,
            col: impl IntoIndex<BLOCK>,
            mask: Mask<BLOCK>,
            fill: f32,
        ) -> Tile<BLOCK> {
            assert!($(N == $n)||+, $msg);
            Tile {
                expr: Expr::QuantizedDot {
                    src: matrix.clone(),
                    activations: PackedActivations::$packing(
                        a.into_iter().map(|value| value.expr).collect(),
                    ),
                    k: DotK::Base(k_base.into_index()),
                    col: col.into_index(),
                    mask: mask.expr,
                    fill: Box::new(Expr::Literal(TileLiteral::F32(F32Bits::new(fill)))),
                    block_n: N as u32,
                },
            }
        }
    };
}

pub struct TileBlock<'a, const BLOCK: usize> {
    pub(super) program: &'a mut Program,
    pub(super) grid: [u32; 3],
    pub(super) body: Vec<TileStmt>,
    /// Stack of nested statement builders. The innermost frame collects
    /// statements emitted inside `while_true`/`fold` closures; popped into
    /// the enclosing loop's body on closure exit.
    pub(super) stmt_stack: Vec<Vec<TileStmt>>,
}


impl<const BLOCK: usize> TileBlock<'_, BLOCK> {
    pub fn program_id(&self, axis: WorkgroupAxis) -> ScalarIndex {
        ScalarIndex {
            expr: Box::new(Expr::Builtin(Builtin::ProgramId(axis))),
        }
    }

    pub fn subgroup_id(&self) -> ScalarIndex {
        ScalarIndex {
            expr: Box::new(Expr::Builtin(Builtin::SubgroupId)),
        }
    }

    pub fn subgroup_lane(&self) -> ScalarIndex {
        ScalarIndex {
            expr: Box::new(Expr::Builtin(Builtin::SubgroupLane)),
        }
    }

    pub fn subgroup_size(&self) -> ScalarIndex {
        ScalarIndex {
            expr: Box::new(Expr::Builtin(Builtin::SubgroupSize)),
        }
    }

    pub fn num_subgroups(&self) -> ScalarIndex {
        ScalarIndex {
            expr: Box::new(Expr::Builtin(Builtin::NumSubgroups)),
        }
    }

    pub fn grid(&self) -> [u32; 3] {
        self.grid
    }

    pub fn arange(&self) -> Range<BLOCK> {
        Range {
            expr: Box::new(Expr::Builtin(Builtin::Lane)),
        }
    }

    pub fn lane_tile_2d<const ROWS: usize, const COLS: usize>(
        &self,
    ) -> LaneTile2d<ROWS, COLS, BLOCK> {
        assert!(
            ROWS > 0 && COLS > 0 && ROWS * COLS == BLOCK,
            "2D lane tile shape must match the tile program block size"
        );
        let lane = self.arange();
        LaneTile2d {
            row: lane.clone() / COLS as u32,
            col: lane % COLS as u32,
        }
    }

    pub fn loop_index(&self) -> ScalarIndex {
        ScalarIndex {
            expr: Box::new(Expr::Builtin(Builtin::LoopIndex)),
        }
    }

    pub fn load<T>(&self, address: Address<T, BLOCK>, mask: Mask<BLOCK>, fill: f32) -> Tile<BLOCK> {
        Tile {
            expr: Expr::Load(TileLoadExpr {
                src: LoadSource::Storage(address.view),
                row: address.row,
                col: address.col,
                mask: mask.expr,
                fill: Box::new(Expr::Literal(TileLiteral::F32(F32Bits::new(fill)))),
            }),
        }
    }

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

    /// Load a `vec4<f32>` from an `F32Vec4`-typed storage view. Routes through
    /// the generic `LoadLinear` with a vec4-splat fill expression — the AST
    /// has no dedicated `LoadVec4` variant and no vec4 literal kind.
    pub fn load_vec4(
        &self,
        address: LinearAddress<F32Vec4, BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        let scalar = Box::new(Expr::Literal(TileLiteral::F32(F32Bits::new(fill))));
        Tile {
            expr: Expr::LoadLinear(TileLinearLoadExpr {
                src: address.view,
                index: address.index,
                mask: mask.expr,
                fill: Box::new(Expr::Compose4 {
                    values: [scalar.clone(), scalar.clone(), scalar.clone(), scalar],
                }),
            }),
        }
    }

    pub fn load_erased(
        &self,
        address: ErasedAddress<BLOCK>,
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
                fill: Box::new(Expr::Literal(TileLiteral::F32(F32Bits::new(fill)))),
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
        let fill_expr: Box<Expr> = Box::new(Expr::Literal(TileLiteral::F32(F32Bits::new(fill))));
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

    /// Sum a flat value list. Constructs a balanced binary tree of
    /// `Expr::Binary(Add, ...)` so the lowerer's recursion depth is
    /// `O(log N)` instead of `O(N)` — the AST has no flat-sum variant.
    pub fn sum(&self, values: impl IntoIterator<Item = Tile<BLOCK>>) -> Tile<BLOCK> {
        let mut exprs: Vec<Expr> = values.into_iter().map(|t| t.expr).collect();
        if exprs.is_empty() {
            return Tile {
                expr: Expr::Literal(TileLiteral::F32(F32Bits::new(0.0))),
            };
        }
        while exprs.len() > 1 {
            let mut next = Vec::with_capacity(exprs.len().div_ceil(2));
            let mut iter = exprs.into_iter();
            while let Some(left) = iter.next() {
                match iter.next() {
                    Some(right) => next.push(Expr::Binary {
                        op: TileBinaryOp::Add,
                        left: Box::new(left),
                        right: Box::new(right),
                    }),
                    None => next.push(left),
                }
            }
            exprs = next;
        }
        Tile {
            expr: exprs.pop().expect("at least one element"),
        }
    }

    /// Iterator-based, state-carrying fold. Iterates over `iter`, threading
    /// `N` `F32` accumulators through the body. Each iteration the body sees
    /// the current iter element and the current accumulator values; it
    /// returns the new accumulator values, which become the values for the
    /// next iteration. After the loop, returns the N final accumulator
    /// values.
    ///
    /// Implementation: allocates a U32 local for the iterator value plus one
    /// F32 local per accumulator, pushes a `TileStmt::Fold` carrying body
    /// statements emitted by the closure plus per-accumulator `init`/`update`
    /// expressions, and returns Tiles that read the accumulator locals
    /// post-loop. Currently the accumulator element type is fixed to `F32`.
    pub fn fold<const N: usize, F>(
        &mut self,
        iter: super::FoldIter,
        initial: [Tile<BLOCK>; N],
        body: F,
    ) -> [Tile<BLOCK>; N]
    where
        F: FnOnce(&mut Self, ScalarIndex, [Tile<BLOCK>; N]) -> [Tile<BLOCK>; N],
    {
        assert!(N > 0, "fold must have at least one accumulator");
        let initial_exprs: Vec<Expr> = initial.into_iter().map(|t| t.expr).collect();

        // Allocate the iterator-value local and one local per accumulator.
        let iter_var_local = self.program.alloc_local::<U32>();
        let acc_locals: [LocalRef; N] =
            std::array::from_fn(|_| self.program.alloc_local::<F32>());

        // Build the body in a fresh stmt frame so we can capture exactly the
        // statements emitted by the closure.
        self.stmt_stack.push(Vec::new());
        let iter_element = ScalarIndex {
            expr: Box::new(Expr::LoadLocal(iter_var_local)),
        };
        let acc_tiles: [Tile<BLOCK>; N] = std::array::from_fn(|i| Tile {
            expr: Expr::LoadLocal(acc_locals[i]),
        });
        let new_state = body(self, iter_element, acc_tiles);
        let body_stmts = self.stmt_stack.pop().expect("fold body frame missing");

        let accumulators: Vec<crate::ir::FoldAccumulator> = new_state
            .into_iter()
            .enumerate()
            .map(|(i, new)| crate::ir::FoldAccumulator {
                name: acc_locals[i].id,
                element: acc_locals[i].element,
                init: initial_exprs[i].clone(),
                update: new.expr,
            })
            .collect();

        self.push_stmt(TileStmt::Fold {
            count: iter.count,
            iter_var: iter_var_local.id,
            body: body_stmts,
            accumulators,
        });

        std::array::from_fn(|i| Tile {
            expr: Expr::LoadLocal(acc_locals[i]),
        })
    }

    /// Run one K-loop with N parallel reductions. The body closure runs once
    /// at IR-build time and produces N tile expressions that all share the
    /// same loop scope; the lowerer materializes a single Naga loop with N
    /// accumulator locals so common subexpressions across the N outputs are
    /// emitted only once per iteration (when bound via `pin`).
    ///
    /// Implementation: desugars into a `TileStmt::Fold` over a counted range
    /// with one `FoldAccumulator` per lane. Each accumulator's `update` is
    /// `binary(reduce_op_to_binary(op), LoadLocal(acc), body[lane])`. After
    /// the statement the per-accumulator locals hold the final reduced
    /// values.
    pub fn loop_fold_n<const N: usize, F>(
        &mut self,
        op: TileReduceOp,
        iterations: u32,
        initials: [TileLiteral; N],
        body: F,
    ) -> [Tile<BLOCK>; N]
    where
        F: FnOnce(&mut Self) -> [Tile<BLOCK>; N],
    {
        assert!(iterations > 0, "loop_fold_n iterations must be non-zero");
        assert!(N > 0, "loop_fold_n must have at least one accumulator");

        // Allocate one local per accumulator, typed by the initial literal.
        let acc_locals: [LocalRef; N] = std::array::from_fn(|i| {
            self.program.alloc_local_element(initials[i].element())
        });
        // Allocate the iter_var local (unused by the body — bodies use
        // `Builtin::LoopIndex` to address the iteration index — but
        // `TileStmt::Fold` requires a name).
        let iter_var_local = self.program.alloc_local::<U32>();

        // Build the body in a fresh stmt frame so we can capture exactly the
        // statements emitted by the closure (e.g. `let` bindings of shared
        // subexpressions). The body closure references the lane accumulators
        // and yields the per-iteration values to combine into them.
        self.stmt_stack.push(Vec::new());
        let bodies = body(self);
        let body_stmts = self.stmt_stack.pop().expect("loop_fold_n body frame missing");

        let binary_op = op.binary();
        let accumulators: Vec<crate::ir::FoldAccumulator> = bodies
            .into_iter()
            .enumerate()
            .map(|(i, lane_value)| crate::ir::FoldAccumulator {
                name: acc_locals[i].id,
                element: acc_locals[i].element,
                init: Expr::Literal(initials[i]),
                update: Expr::Binary {
                    op: binary_op,
                    left: Box::new(Expr::LoadLocal(acc_locals[i])),
                    right: Box::new(lane_value.expr),
                },
            })
            .collect();

        self.push_stmt(TileStmt::Fold {
            count: Box::new(Expr::Literal(TileLiteral::U32(iterations))),
            iter_var: iter_var_local.id,
            body: body_stmts,
            accumulators,
        });

        std::array::from_fn(|i| Tile {
            expr: Expr::LoadLocal(acc_locals[i]),
        })
    }

    /// Fused 4-way dot product: `a[0]*b[0] + .. + a[3]*b[3]` in a single
    /// `Math::Dot` over `vec4<f32>` operands. Lowers to the same instruction
    /// sequence the qgemv accelerator emits.
    pub fn dot4(&self, a: [Tile<BLOCK>; 4], b: [Tile<BLOCK>; 4]) -> Tile<BLOCK> {
        let left = self.compose4(a);
        let right = self.compose4(b);
        self.vec4_dot(left, right)
    }

    pub fn vec4_dot(&self, left: Tile<BLOCK>, right: Tile<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: Expr::Vec4Dot {
                left: Box::new(left.expr),
                right: Box::new(right.expr),
            },
        }
    }

    /// `vec4<f32>(value, value, value, value)`. Lowers to a `Compose4` over
    /// four references to the same value — Naga's downstream optimizer
    /// collapses identical SSA reads.
    pub fn vec4_splat(&self, value: Tile<BLOCK>) -> Tile<BLOCK> {
        let v = value.expr;
        Tile {
            expr: Expr::Compose4 {
                values: [
                    Box::new(v.clone()),
                    Box::new(v.clone()),
                    Box::new(v.clone()),
                    Box::new(v),
                ],
            },
        }
    }

    /// Pack four scalars into a `vec4<f32>`.
    pub fn compose4(&self, values: [Tile<BLOCK>; 4]) -> Tile<BLOCK> {
        let [v0, v1, v2, v3] = values;
        Tile {
            expr: Expr::Compose4 {
                values: [
                    Box::new(v0.expr),
                    Box::new(v1.expr),
                    Box::new(v2.expr),
                    Box::new(v3.expr),
                ],
            },
        }
    }

    /// 8-wide quantized dot with raw `f32` activations (Q8_0 / Q6K). Routes
    /// through the generic `QuantizedDot` with `F32` activations, a flat
    /// `Base` K coordinate, and `block_n = 8`.
    pub fn quantized_q8_0_dot8(
        &self,
        a: [Tile<BLOCK>; 8],
        matrix: &QuantizedMatrix,
        k_base: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        Tile {
            expr: Expr::QuantizedDot {
                src: matrix.clone(),
                activations: PackedActivations::F32(
                    a.into_iter().map(|value| value.expr).collect(),
                ),
                k: DotK::Base(k_base.into_index()),
                col: col.into_index(),
                mask: mask.expr,
                fill: Box::new(Expr::Literal(TileLiteral::F32(F32Bits::new(fill)))),
                block_n: 8,
            },
        }
    }

    quantized_vec_dot_entrypoint!(
        quantized_q8_activation_dot,
        Q8,
        [8, 16],
        "q8 activation dot currently supports N == 8 or N == 16"
    );

    quantized_vec_dot_entrypoint!(
        quantized_q4k_f32_dot,
        F32,
        [8, 16, 32],
        "q4k f32 dot currently supports N == 8, N == 16, or N == 32"
    );

    #[allow(clippy::too_many_arguments)]
    pub fn quantized_q4k_ggml_dot(
        &self,
        a_low: [Tile<BLOCK>; 16],
        a_high: [Tile<BLOCK>; 16],
        sums: [Tile<BLOCK>; 4],
        matrix: &QuantizedMatrix,
        block: impl IntoIndex<BLOCK>,
        iq: impl IntoIndex<BLOCK>,
        ir: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        Tile {
            expr: Expr::QuantizedDot {
                src: matrix.clone(),
                activations: PackedActivations::Q4KGgml {
                    low: a_low.into_iter().map(|value| value.expr).collect(),
                    high: a_high.into_iter().map(|value| value.expr).collect(),
                    sums: sums.into_iter().map(|value| value.expr).collect(),
                },
                k: DotK::Block {
                    block: block.into_index(),
                    c0: iq.into_index(),
                    c1: ir.into_index(),
                },
                col: col.into_index(),
                mask: mask.expr,
                fill: Box::new(Expr::Literal(TileLiteral::F32(F32Bits::new(fill)))),
                block_n: 32,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn quantized_q6k_ggml_dot(
        &self,
        a: [Tile<BLOCK>; 16],
        matrix: &QuantizedMatrix,
        block: impl IntoIndex<BLOCK>,
        ip: impl IntoIndex<BLOCK>,
        il: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
        mask: Mask<BLOCK>,
        fill: f32,
    ) -> Tile<BLOCK> {
        Tile {
            expr: Expr::QuantizedDot {
                src: matrix.clone(),
                activations: PackedActivations::F32(
                    a.into_iter().map(|value| value.expr).collect(),
                ),
                k: DotK::Block {
                    block: block.into_index(),
                    c0: ip.into_index(),
                    c1: il.into_index(),
                },
                col: col.into_index(),
                mask: mask.expr,
                fill: Box::new(Expr::Literal(TileLiteral::F32(F32Bits::new(fill)))),
                block_n: 16,
            },
        }
    }

    pub fn literal(&self, value: TileLiteral) -> Tile<BLOCK> {
        Tile {
            expr: Expr::Literal(value),
        }
    }

    pub fn f32(&self, value: f32) -> Tile<BLOCK> {
        self.literal(TileLiteral::F32(F32Bits::new(value)))
    }

    pub fn u32(&self, value: u32) -> Tile<BLOCK> {
        self.literal(TileLiteral::U32(value))
    }

    pub fn bool(&self, value: bool) -> Tile<BLOCK> {
        self.literal(TileLiteral::Bool(value))
    }

    pub fn index(&self, value: impl IntoIndex<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: *value.into_index(),
        }
    }

    pub fn exp(&self, value: Tile<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: Expr::Unary {
                op: TileUnaryOp::Exp,
                value: Box::new(value.expr),
            },
        }
    }

    tile_reduce_entrypoints!(
        (
            reduce_sum,
            loop_reduce_sum,
            group_reduce_sum,
            subgroup_reduce_sum,
            Sum
        ),
        (
            reduce_max,
            loop_reduce_max,
            group_reduce_max,
            subgroup_reduce_max,
            Max
        ),
        (
            reduce_min,
            loop_reduce_min,
            group_reduce_min,
            subgroup_reduce_min,
            Min
        ),
    );

    /// Per-lane scalar fold over `iterations` iterations. Desugars into a
    /// single-accumulator `TileStmt::Fold`; the AST has no dedicated
    /// loop-fold expression.
    pub fn loop_fold(
        &mut self,
        op: TileReduceOp,
        iterations: u32,
        value: Tile<BLOCK>,
        initial: TileLiteral,
    ) -> Tile<BLOCK> {
        assert!(iterations > 0, "loop fold iterations must be non-zero");
        let element = initial.element();
        let acc_local = self.program.alloc_local_element(element);
        let iter_var_local = self.program.alloc_local::<U32>();
        self.push_stmt(TileStmt::Fold {
            count: Box::new(Expr::Literal(TileLiteral::U32(iterations))),
            iter_var: iter_var_local.id,
            body: Vec::new(),
            accumulators: vec![crate::ir::FoldAccumulator {
                name: acc_local.id,
                element,
                init: Expr::Literal(initial),
                update: Expr::Binary {
                    op: op.binary(),
                    left: Box::new(Expr::LoadLocal(acc_local)),
                    right: Box::new(value.expr),
                },
            }],
        });
        Tile {
            expr: Expr::LoadLocal(acc_local),
        }
    }

    fn subgroup_reduce(&self, op: TileReduceOp, value: Tile<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: Expr::SubgroupReduce {
                op,
                value: Box::new(value.expr),
            },
        }
    }

    fn group_reduce<const GROUP: usize>(
        &mut self,
        op: TileReduceOp,
        value: Tile<BLOCK>,
    ) -> Tile<BLOCK> {
        assert!(
            GROUP > 0 && GROUP <= BLOCK && GROUP.is_power_of_two() && BLOCK.is_multiple_of(GROUP),
            "tile group reduction size must be a power-of-two divisor of the block"
        );
        let scratch = self.program.alloc_tile::<F32>(Layout::contiguous(
            MemoryLevel::Workgroup,
            Shape::new([BLOCK as u32]),
        ));
        Tile {
            expr: Expr::Reduce {
                op,
                iterations: 1,
                value: Box::new(value.expr),
                scratch,
                group_size: GROUP as u32,
            },
        }
    }

    fn reduce(&mut self, op: TileReduceOp, value: Tile<BLOCK>) -> Scalar {
        self.loop_reduce(op, 1, value)
    }

    fn loop_reduce(&mut self, op: TileReduceOp, iterations: u32, value: Tile<BLOCK>) -> Scalar {
        assert!(iterations > 0, "loop reduce iterations must be non-zero");
        let scratch = self.program.alloc_tile::<F32>(Layout::contiguous(
            MemoryLevel::Workgroup,
            Shape::new([BLOCK as u32]),
        ));
        Scalar {
            expr: Expr::Reduce {
                op,
                iterations,
                value: Box::new(value.expr),
                scratch,
                group_size: BLOCK as u32,
            },
        }
    }

    /// Allocate an 8x8 f32 cooperative-matrix accumulator local. Returned
    /// handle is consumed by `zero_coop_acc`, `mma_from_tiles`, and
    /// `coop_store`.
    pub fn alloc_coop_acc(&mut self) -> CoopAcc {
        let local = self
            .program
            .alloc_local_element(ElementType::CoopMatrixF32 { rows: 8, cols: 8 });
        CoopAcc { local }
    }

    pub fn zero_coop_acc(&mut self, acc: &CoopAcc) {
        self.push_stmt(TileStmt::ZeroCoopAcc { acc: acc.local });
    }

    /// Stage a workgroup-tile region of dense `src` into the workgroup-tile
    /// `dst`. Used for the A operand in qmatmul. The lowerer emits a flat
    /// per-invocation loop.
    pub fn copy_storage_to_tile(
        &mut self,
        dst_tile: TileRef,
        src: &Storage<F32, 2>,
        row_offset: impl IntoIndex<BLOCK>,
        col_offset: impl IntoIndex<BLOCK>,
    ) {
        self.push_stmt(TileStmt::CopyToWorkgroupTile {
            dst: dst_tile,
            src: crate::ir::CopySource::Storage(src.view.clone()),
            row_offset: row_offset.into_index(),
            col_offset: col_offset.into_index(),
        });
    }

    /// Stage a workgroup-tile region of quantized `src` into the f32
    /// workgroup-tile `dst`, dequantizing on the fly. Used for the B operand
    /// in qmatmul.
    pub fn copy_quant_to_tile(
        &mut self,
        dst_tile: TileRef,
        src: &QuantizedMatrix,
        row_offset: impl IntoIndex<BLOCK>,
        col_offset: impl IntoIndex<BLOCK>,
    ) {
        self.push_stmt(TileStmt::CopyToWorkgroupTile {
            dst: dst_tile,
            src: crate::ir::CopySource::Quantized(src.clone()),
            row_offset: row_offset.into_index(),
            col_offset: col_offset.into_index(),
        });
    }

    pub fn workgroup_barrier(&mut self) {
        self.push_stmt(TileStmt::Barrier);
    }

    pub fn private<T: Numeric>(&mut self) -> Local<T, BLOCK> {
        Local {
            local: self.program.alloc_local::<T>(),
            _ty: PhantomData,
        }
    }

    pub fn load_local<T>(&self, local: &Local<T, BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: Expr::LoadLocal(local.local),
        }
    }

    pub fn store_local<T>(&mut self, local: &Local<T, BLOCK>, value: Tile<BLOCK>) {
        self.push_stmt(TileStmt::StoreLocal {
            dst: local.local,
            value: value.expr,
        });
    }

    pub fn load_workgroup(&self, tile: TileRef, index: impl IntoIndex<BLOCK>) -> Tile<BLOCK> {
        Tile {
            expr: Expr::LoadWorkgroup {
                src: tile,
                index: index.into_index(),
            },
        }
    }

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

    pub fn if_then(&mut self, condition: Tile<BLOCK>, accept: impl FnOnce(&mut Self)) {
        self.if_else(condition, accept, |_| {});
    }

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

    pub fn loop_forever(&mut self, body: impl FnOnce(&mut Self)) {
        self.stmt_stack.push(Vec::new());
        body(self);
        let body = self.stmt_stack.pop().expect("loop frame missing");
        self.push_stmt(TileStmt::Loop { body });
    }

    pub fn break_loop(&mut self) {
        self.push_stmt(TileStmt::Break);
    }

    pub fn return_(&mut self) {
        self.push_stmt(TileStmt::Return);
    }

    /// `acc += coop_load_a(a_tile, ar, ak) * coop_load_b(b_tile, bk, bc)`.
    /// Convenience wrapper that emits `coop_load_a`, `coop_load_b`, then
    /// `coop_mma`. For MMAs that share an A or B operand across the inner row ×
    /// col grid, prefer the explicit calls so fragment handles can be reused.
    pub fn mma_from_tiles(
        &mut self,
        acc: &CoopAcc,
        a_tile: TileRef,
        a_row: impl IntoIndex<BLOCK>,
        a_col: impl IntoIndex<BLOCK>,
        b_tile: TileRef,
        b_row: impl IntoIndex<BLOCK>,
        b_col: impl IntoIndex<BLOCK>,
    ) {
        let a = self.coop_load_a(a_tile, a_row, a_col);
        let b = self.coop_load_b(b_tile, b_row, b_col);
        self.coop_mma(acc, &a, &b);
    }

    /// Cooperatively load an 8x8 A fragment from a workgroup tile. The
    /// returned handle's SSA value is bound at the load site and reused
    /// wherever the handle is consumed by `coop_mma` in the same scope.
    pub fn coop_load_a(
        &mut self,
        tile: TileRef,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
    ) -> CoopFragment {
        self.coop_load(CoopOperandRole::A, tile, row, col)
    }

    /// Cooperatively load an 8x8 B fragment.
    pub fn coop_load_b(
        &mut self,
        tile: TileRef,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
    ) -> CoopFragment {
        self.coop_load(CoopOperandRole::B, tile, row, col)
    }

    fn coop_load(
        &mut self,
        role: CoopOperandRole,
        tile: TileRef,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
    ) -> CoopFragment {
        let id = self.program.next_coop_fragment_id();
        self.push_stmt(TileStmt::LoadCoop {
            id,
            role,
            tile,
            row: row.into_index(),
            col: col.into_index(),
        });
        CoopFragment { id, role }
    }

    /// `acc += a * b` where `a`/`b` are fragments previously loaded via
    /// `coop_load_a`/`coop_load_b`.
    pub fn coop_mma(&mut self, acc: &CoopAcc, a: &CoopFragment, b: &CoopFragment) {
        assert_eq!(
            a.role,
            CoopOperandRole::A,
            "coop_mma A operand must be an A-role fragment"
        );
        assert_eq!(
            b.role,
            CoopOperandRole::B,
            "coop_mma B operand must be a B-role fragment"
        );
        self.push_stmt(TileStmt::Mma {
            acc: acc.local,
            a: a.id,
            b: b.id,
        });
    }

    /// Cooperatively store `acc` to `dst` at (row, col).
    pub fn coop_store(
        &mut self,
        acc: &CoopAcc,
        dst: &Storage<F32, 2>,
        row: impl IntoIndex<BLOCK>,
        col: impl IntoIndex<BLOCK>,
    ) {
        self.push_stmt(TileStmt::StoreCoopAcc {
            acc: acc.local,
            dst: dst.view.clone(),
            row: row.into_index(),
            col: col.into_index(),
        });
    }

    /// Emit a counted `while true` loop where `program.loop_index()` resolves
    /// to the current iteration. Desugars into a `TileStmt::Fold` over a
    /// counted range with no accumulators — the AST has no dedicated counted
    /// loop variant.
    pub fn while_true<F: FnOnce(&mut Self)>(&mut self, max_iterations: u32, body: F) {
        assert!(
            max_iterations > 0,
            "while_true max_iterations must be non-zero"
        );
        let iter_var_local = self.program.alloc_local::<U32>();
        self.stmt_stack.push(Vec::new());
        body(self);
        let stmts = self.stmt_stack.pop().expect("while_true frame missing");
        self.push_stmt(TileStmt::Fold {
            count: Box::new(Expr::Literal(TileLiteral::U32(max_iterations))),
            iter_var: iter_var_local.id,
            body: stmts,
            accumulators: Vec::new(),
        });
    }

    fn push_stmt(&mut self, stmt: TileStmt) {
        if let Some(frame) = self.stmt_stack.last_mut() {
            frame.push(stmt);
        } else {
            self.body.push(stmt);
        }
    }

    pub fn store<T>(&mut self, address: Address<T, BLOCK>, value: Tile<BLOCK>, mask: Mask<BLOCK>) {
        self.push_stmt(TileStmt::Store(TileStoreStmt {
            dst: address.view,
            row: address.row,
            col: address.col,
            value: value.expr,
            mask: mask.expr,
        }));
    }

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

    pub fn store_erased(
        &mut self,
        address: ErasedAddress<BLOCK>,
        value: Tile<BLOCK>,
        mask: Mask<BLOCK>,
    ) {
        self.push_stmt(TileStmt::Store(TileStoreStmt {
            dst: address.view,
            row: address.row,
            col: address.col,
            value: value.expr,
            mask: mask.expr,
        }));
    }
}

