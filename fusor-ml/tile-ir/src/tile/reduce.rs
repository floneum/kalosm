use super::block::load_local_expr;
use super::value::{zero_expr, FoldIter};
use super::{Tile, TileBlock};
use crate::ir::{
    Accumulator, ElementType, Expr, ExprKind, Layout, MemoryLevel, ReduceKind, Shape, TileBinaryOp,
    TileReduceOp,
};

macro_rules! tile_reduce_entrypoints {
    ($(($reduce:ident, $loop_reduce:ident, $group_reduce:ident, $op:ident)),+ $(,)?) => {
        $(
            /// Cross-lane reduction across the whole workgroup.
            pub fn $reduce(&mut self, value: Tile) -> Tile {
                self.reduce(TileReduceOp::$op, value)
            }
            /// Per-lane accumulation across `iterations`, then a workgroup tree.
            pub fn $loop_reduce<F>(&mut self, iterations: u32, body: F) -> Tile
            where
                F: FnOnce(&mut Self, Tile) -> Tile,
            {
                self.loop_reduce(TileReduceOp::$op, iterations, body)
            }
            /// Cross-lane reduction over contiguous-lane groups of `group_size`.
            pub fn $group_reduce(&mut self, group_size: u32, value: Tile) -> Tile {
                self.group_reduce(TileReduceOp::$op, group_size, value)
            }
        )+
    };
}

impl TileBlock<'_> {
    tile_reduce_entrypoints!(
        (reduce_sum, loop_reduce_sum, group_reduce_sum, Sum),
        (reduce_max, loop_reduce_max, group_reduce_max, Max),
        (reduce_min, loop_reduce_min, group_reduce_min, Min),
    );

    /// Pairwise tree-sum of a set of values into a single value (no cross-lane
    /// communication). Returns the zero of `element` for an empty input.
    pub fn sum(&self, values: impl IntoIterator<Item = Tile>, element: ElementType) -> Tile {
        let mut exprs: Vec<Expr> = values.into_iter().map(Tile::into_expr).collect();
        if exprs.is_empty() {
            return Tile::from_expr(zero_expr(element));
        }
        while exprs.len() > 1 {
            let mut next = Vec::with_capacity(exprs.len().div_ceil(2));
            let mut iter = exprs.into_iter();
            while let Some(left) = iter.next() {
                match iter.next() {
                    Some(right) => {
                        let ty = left.element();
                        next.push(Expr::new(
                            ExprKind::Binary {
                                op: TileBinaryOp::Add,
                                left: Box::new(left),
                                right: Box::new(right),
                            },
                            ty,
                        ));
                    }
                    None => next.push(left),
                }
            }
            exprs = next;
        }
        Tile::from_expr(exprs.pop().expect("at least one element"))
    }

    /// Counted loop carrying `N` f32 accumulators. The body returns the new
    /// accumulator values each iteration; the loop returns the final values.
    pub fn fold<const N: usize, F>(
        &mut self,
        iter: FoldIter,
        initial: [Tile; N],
        body: F,
    ) -> [Tile; N]
    where
        F: FnOnce(&mut Self, Tile, [Tile; N]) -> [Tile; N],
    {
        assert!(N > 0);
        let result = self.fold_vec(iter, initial.into_iter().collect(), |slf, idx, accs| {
            let accs: [Tile; N] = accs
                .try_into()
                .unwrap_or_else(|_| panic!("fold accumulator arity mismatch"));
            body(slf, idx, accs).into_iter().collect()
        });
        result
            .try_into()
            .unwrap_or_else(|_| panic!("fold returned wrong arity"))
    }

    /// Runtime-sized [`fold`](Self::fold). The body must return a Vec the same
    /// length as `initial`.
    pub fn fold_vec<F>(&mut self, iter: FoldIter, initial: Vec<Tile>, body: F) -> Vec<Tile>
    where
        F: FnOnce(&mut Self, Tile, Vec<Tile>) -> Vec<Tile>,
    {
        let n = initial.len();
        assert!(n > 0, "fold needs at least one accumulator");
        let acc_locals: Vec<_> = initial
            .iter()
            .map(|t| self.program.alloc_local(t.element()))
            .collect();
        let index = self.program.alloc_local(ElementType::U32);

        self.open_frame();
        let acc_tiles: Vec<Tile> = acc_locals
            .iter()
            .map(|l| load_local_expr(l.decl()))
            .collect();
        let new_state = body(self, load_local_expr(index.decl()), acc_tiles);
        assert_eq!(new_state.len(), n, "fold body returned wrong arity");
        let body_stmts = self.close_frame();

        let accumulators: Vec<Accumulator> = acc_locals
            .iter()
            .zip(initial)
            .zip(new_state)
            .map(|((local, init), update)| Accumulator {
                local: local.decl().clone(),
                init: init.into_expr(),
                update: update.into_expr(),
            })
            .collect();

        self.push_counted_loop(
            iter.count_expr(),
            Some(index.decl().clone()),
            accumulators,
            body_stmts,
        );

        acc_locals
            .into_iter()
            .map(|l| load_local_expr(l.decl()))
            .collect()
    }

    // ---- internals -------------------------------------------------------

    pub(crate) fn subgroup_reduce(&self, op: TileReduceOp, value: Tile) -> Tile {
        let ty = value.element();
        Tile::new(
            ExprKind::Reduce {
                op,
                kind: ReduceKind::Subgroup,
                value: Box::new(value.into_expr()),
            },
            ty,
        )
    }

    /// Cross-lane reduction over contiguous-lane groups of `group_size`.
    pub fn group_reduce(&mut self, op: TileReduceOp, group_size: u32, value: Tile) -> Tile {
        let block = self.block_size();
        assert!(group_size > 0 && group_size <= block && group_size.is_power_of_two());
        let scratch = self.alloc_reduce_scratch(value.element());
        let ty = value.element();
        Tile::new(
            ExprKind::Reduce {
                op,
                kind: ReduceKind::Workgroup {
                    scratch: scratch.decl().clone(),
                    group_size,
                },
                value: Box::new(value.into_expr()),
            },
            ty,
        )
    }

    /// Whole-workgroup reduction built from subgroup collectives: one
    /// per-subgroup reduce, the per-subgroup partials staged through a
    /// `num_subgroups`-sized workgroup array, and a serial fold of the
    /// partials on every lane. Two barriers total, versus one per tree level
    /// in [`Self::group_reduce`]. The caller owns the device gating (fixed
    /// `subgroup_size` dividing the workgroup size) via `SubgroupToken`.
    pub(crate) fn workgroup_reduce_via_subgroups(
        &mut self,
        op: TileReduceOp,
        subgroup_size: u32,
        value: Tile,
    ) -> Tile {
        let block = self.block_size();
        assert!(
            subgroup_size > 0 && block.is_multiple_of(subgroup_size),
            "workgroup_reduce_via_subgroups requires a fixed subgroup size dividing the block"
        );
        let element = value.element();
        let partial = self.subgroup_reduce(op, value);
        let num_subgroups = block / subgroup_size;
        if num_subgroups == 1 {
            return self.bind(partial);
        }
        let partial = self.bind(partial);
        let scratch = self.program.alloc_tile(
            element,
            Layout::contiguous(MemoryLevel::Workgroup, Shape::new([num_subgroups])),
        );
        // Barrier before seeding the scratch: a previous reduction through
        // the same call site (a reduce inside a loop) may still have lanes
        // reading the prior partials.
        self.workgroup_barrier();
        let subgroup_lane = self.subgroup_lane();
        let subgroup_id = self.subgroup_id();
        self.if_then(subgroup_lane.eq(0u32), |program| {
            program.store_workgroup(&scratch, subgroup_id, partial);
        });
        self.workgroup_barrier();
        let mut combined = self.load_workgroup(&scratch, 0u32);
        for index in 1..num_subgroups {
            let next = self.load_workgroup(&scratch, index);
            combined = combined.binary(op.binary(), next);
        }
        self.bind(combined)
    }

    fn reduce(&mut self, op: TileReduceOp, value: Tile) -> Tile {
        let block = self.block_size();
        self.group_reduce(op, block, value)
    }

    fn loop_reduce<F>(&mut self, op: TileReduceOp, iterations: u32, body: F) -> Tile
    where
        F: FnOnce(&mut Self, Tile) -> Tile,
    {
        assert!(iterations > 0);
        let block = self.block_size();
        let index = self.program.alloc_local(ElementType::U32);
        self.open_frame();
        let value = body(self, load_local_expr(index.decl()));
        let leaked = self.close_frame();
        assert!(
            leaked.is_empty(),
            "loop_reduce body must be side-effect-free"
        );
        let scratch = self.alloc_reduce_scratch(value.element());
        let ty = value.element();
        Tile::new(
            ExprKind::Reduce {
                op,
                kind: ReduceKind::Loop {
                    iterations,
                    index: index.decl().clone(),
                    scratch: scratch.decl().clone(),
                    group_size: block,
                },
                value: Box::new(value.into_expr()),
            },
            ty,
        )
    }

    fn alloc_reduce_scratch(&mut self, element: ElementType) -> super::value::WorkgroupTile {
        let block = self.block_size();
        self.program.alloc_tile(
            element,
            Layout::contiguous(MemoryLevel::Workgroup, Shape::new([block])),
        )
    }
}

impl FoldIter {
    pub(super) fn count_expr(self) -> Expr {
        *self.count
    }
}
