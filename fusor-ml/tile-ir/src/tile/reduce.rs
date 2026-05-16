use super::block::tiles_to_exprs;
use super::value::boxed_u32_literal;
use super::*;
use crate::ir::{
    Expr, Layout, LocalRef, MemoryLevel, Shape, TileBinaryOp, TileLiteral, TileReduceOp, TileStmt,
    F32, U32,
};

macro_rules! tile_reduce_entrypoints {
    ($(($reduce:ident, $loop_reduce:ident, $group_reduce:ident, $subgroup_reduce:ident, $op:ident)),+ $(,)?) => {
        $(
            #[doc = concat!("Reduce `value` across the whole workgroup with `", stringify!($op), "`.")]
            pub fn $reduce(&mut self, value: Tile) -> Tile {
                self.reduce(TileReduceOp::$op, value)
            }

            #[doc = concat!("Loop-reduce across `iterations` using `", stringify!($op), "`.")]
            #[doc = ""]
            #[doc = "The body closure receives a `Tile` bound to the current"]
            #[doc = "iteration; the returned tile is accumulated per lane and then"]
            #[doc = "cross-lane reduced."]
            pub fn $loop_reduce<F>(&mut self, iterations: u32, body: F) -> Tile
            where
                F: FnOnce(&mut Self, Tile) -> Tile,
            {
                self.loop_reduce(TileReduceOp::$op, iterations, body)
            }

            #[doc = concat!("Reduce `value` across a fixed group using `", stringify!($op), "`.")]
            pub fn $group_reduce<const GROUP: usize>(&mut self, value: Tile) -> Tile {
                self.group_reduce::<GROUP>(TileReduceOp::$op, value)
            }

            #[doc = concat!("Reduce `value` across the current subgroup using `", stringify!($op), "`.")]
            pub fn $subgroup_reduce(&self, value: Tile) -> Tile {
                self.subgroup_reduce(TileReduceOp::$op, value)
            }
        )+
    };
}

impl TileBlock<'_> {
    /// Sum a flat value list. Constructs a balanced binary tree of
    /// `Expr::Binary(Add, ...)` so the lowerer's recursion depth is
    /// `O(log N)` instead of `O(N)` — the AST has no flat-sum variant.
    pub fn sum(&self, values: impl IntoIterator<Item = Tile>) -> Tile {
        let mut exprs = tiles_to_exprs(values);
        if exprs.is_empty() {
            return Tile {
                expr: Expr::Literal(TileLiteral::f32(0.0)),
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
    pub fn fold<const N: usize, F>(
        &mut self,
        iter: super::FoldIter,
        initial: [Tile; N],
        body: F,
    ) -> [Tile; N]
    where
        F: FnOnce(&mut Self, Tile, [Tile; N]) -> [Tile; N],
    {
        assert!(N > 0, "fold must have at least one accumulator");
        let initial_exprs = tiles_to_exprs(initial);

        let iter_var_local = self.program.alloc_local::<U32>();
        let acc_locals: [LocalRef; N] = std::array::from_fn(|_| self.program.alloc_local::<F32>());

        self.stmt_stack.push(Vec::new());
        let iter_element = Tile {
            expr: Expr::LoadLocal(iter_var_local),
        };
        let acc_tiles: [Tile; N] = std::array::from_fn(|i| Tile {
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
    /// emitted only once per iteration. The closure receives a `Tile`
    /// bound to this loop's iteration counter.
    pub fn loop_fold_n<const N: usize, F>(
        &mut self,
        op: TileReduceOp,
        iterations: u32,
        initials: [TileLiteral; N],
        body: F,
    ) -> [Tile; N]
    where
        F: FnOnce(&mut Self, Tile) -> [Tile; N],
    {
        assert!(iterations > 0, "loop_fold_n iterations must be non-zero");
        assert!(N > 0, "loop_fold_n must have at least one accumulator");

        let acc_locals: [LocalRef; N] =
            std::array::from_fn(|i| self.program.alloc_local_element(initials[i].element()));
        let iter_var_local = self.program.alloc_local::<U32>();

        self.stmt_stack.push(Vec::new());
        let iter_index = Tile {
            expr: Expr::LoadLocal(iter_var_local),
        };
        let bodies = body(self, iter_index);
        let body_stmts = self
            .stmt_stack
            .pop()
            .expect("loop_fold_n body frame missing");

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
            count: boxed_u32_literal(iterations),
            iter_var: iter_var_local.id,
            body: body_stmts,
            accumulators,
        });

        std::array::from_fn(|i| Tile {
            expr: Expr::LoadLocal(acc_locals[i]),
        })
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

    /// Per-lane scalar fold over `iterations` iterations. The body closure
    /// receives a `Tile` bound to this loop's iteration counter and
    /// returns the per-iteration value to accumulate. Desugars into a
    /// single-accumulator `TileStmt::Fold`; the AST has no dedicated loop-fold
    /// expression.
    pub fn loop_fold<F>(
        &mut self,
        op: TileReduceOp,
        iterations: u32,
        initial: TileLiteral,
        body: F,
    ) -> Tile
    where
        F: FnOnce(&mut Self, Tile) -> Tile,
    {
        assert!(iterations > 0, "loop fold iterations must be non-zero");
        let element = initial.element();
        let acc_local = self.program.alloc_local_element(element);
        let iter_var_local = self.program.alloc_local::<U32>();
        self.stmt_stack.push(Vec::new());
        let iter_index = Tile {
            expr: Expr::LoadLocal(iter_var_local),
        };
        let value = body(self, iter_index);
        let body_stmts = self.stmt_stack.pop().expect("loop_fold body frame missing");
        self.push_stmt(TileStmt::Fold {
            count: boxed_u32_literal(iterations),
            iter_var: iter_var_local.id,
            body: body_stmts,
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

    fn subgroup_reduce(&self, op: TileReduceOp, value: Tile) -> Tile {
        Tile {
            expr: Expr::SubgroupReduce {
                op,
                value: Box::new(value.expr),
            },
        }
    }

    fn group_reduce<const GROUP: usize>(&mut self, op: TileReduceOp, value: Tile) -> Tile {
        let block = self.block_size();
        assert!(
            GROUP > 0 && GROUP <= block && GROUP.is_power_of_two() && block.is_multiple_of(GROUP),
            "tile group reduction size must be a power-of-two divisor of the block"
        );
        let scratch = self.program.alloc_tile::<F32>(Layout::contiguous(
            MemoryLevel::Workgroup,
            Shape::new([block as u32]),
        ));
        Tile {
            expr: Expr::Reduce {
                op,
                iterations: 1,
                iter_var: None,
                value: Box::new(value.expr),
                scratch,
                group_size: GROUP as u32,
            },
        }
    }

    fn reduce(&mut self, op: TileReduceOp, value: Tile) -> Tile {
        let block = self.block_size();
        let scratch = self.program.alloc_tile::<F32>(Layout::contiguous(
            MemoryLevel::Workgroup,
            Shape::new([block as u32]),
        ));
        Tile {
            expr: Expr::Reduce {
                op,
                iterations: 1,
                iter_var: None,
                value: Box::new(value.expr),
                scratch,
                group_size: block as u32,
            },
        }
    }

    fn loop_reduce<F>(&mut self, op: TileReduceOp, iterations: u32, body: F) -> Tile
    where
        F: FnOnce(&mut Self, Tile) -> Tile,
    {
        assert!(iterations > 0, "loop reduce iterations must be non-zero");
        let block = self.block_size();
        let scratch = self.program.alloc_tile::<F32>(Layout::contiguous(
            MemoryLevel::Workgroup,
            Shape::new([block as u32]),
        ));
        let iter_var_local = self.program.alloc_local::<U32>();
        let iter_index = Tile {
            expr: Expr::LoadLocal(iter_var_local),
        };
        // Push a stmt frame to catch any statements the body tries to emit.
        // `Expr::Reduce` is a pure expression — its synthesized loop has no
        // IR-level body, so emitted statements would silently leak to the
        // surrounding scope (outside the lowerer's internal loop). Use
        // `loop_fold` for stateful loop bodies.
        self.stmt_stack.push(Vec::new());
        let value = body(self, iter_index);
        let leaked = self
            .stmt_stack
            .pop()
            .expect("loop_reduce body frame missing");
        assert!(
            leaked.is_empty(),
            "loop_reduce body must be a pure expression; use loop_fold for stateful loop bodies"
        );
        Tile {
            expr: Expr::Reduce {
                op,
                iterations,
                iter_var: Some(iter_var_local.id),
                value: Box::new(value.expr),
                scratch,
                group_size: block as u32,
            },
        }
    }
}
