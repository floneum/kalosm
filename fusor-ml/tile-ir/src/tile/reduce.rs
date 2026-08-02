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

    /// Whole-workgroup reduction built from subgroup collectives — the
    /// `group_size == block` case of [`Self::group_reduce_via_subgroups`].
    pub(crate) fn workgroup_reduce_via_subgroups(
        &mut self,
        op: TileReduceOp,
        subgroup_size: u32,
        value: Tile,
    ) -> Tile {
        let block = self.block_size();
        self.group_reduce_via_subgroups(op, subgroup_size, block, value)
    }

    /// Cross-lane reduction over lane groups of `group_size`, built from
    /// subgroup collectives: one per-subgroup reduce, the per-subgroup
    /// partials staged through a `num_subgroups`-sized workgroup array, and a
    /// serial fold of the `group_size / subgroup_size` partials that belong to
    /// the lane's own group. Two barriers total whatever the group size,
    /// versus one per tree level in [`Self::group_reduce`] — and none at all
    /// when a group is exactly one subgroup. The caller owns the device gating
    /// (a fixed `subgroup_size`) via `SubgroupToken`.
    ///
    /// Groups are subgroup-aligned by construction: group `g` owns subgroup
    /// ids `[g * s, (g + 1) * s)`. Callers packing several rows into one
    /// workgroup must derive the row a lane serves from `subgroup_id` the same
    /// way, because the mapping from `local_invocation_index` onto subgroups
    /// is implementation defined.
    pub(crate) fn group_reduce_via_subgroups(
        &mut self,
        op: TileReduceOp,
        subgroup_size: u32,
        group_size: u32,
        value: Tile,
    ) -> Tile {
        let block = self.block_size();
        assert!(
            subgroup_size > 0
                && group_size.is_multiple_of(subgroup_size)
                && block.is_multiple_of(group_size),
            "group_reduce_via_subgroups requires subgroup-aligned groups tiling the block"
        );
        let element = value.element();
        let partial = self.subgroup_reduce(op, value);
        let per_group = group_size / subgroup_size;
        if per_group == 1 {
            return self.bind(partial);
        }
        let partial = self.bind(partial);
        let num_subgroups = block / subgroup_size;
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
            program.store_workgroup(&scratch, subgroup_id.clone(), partial);
        });
        self.workgroup_barrier();
        // The whole block is one group: every lane folds the array from slot
        // zero, so the base index stays a literal.
        let base = (per_group != num_subgroups).then(|| {
            let group_base = subgroup_id / per_group * per_group;
            self.bind(group_base)
        });
        let mut combined = match &base {
            Some(base) => self.load_workgroup(&scratch, base.clone()),
            None => self.load_workgroup(&scratch, 0u32),
        };
        for index in 1..per_group {
            let next = match &base {
                Some(base) => self.load_workgroup(&scratch, base.clone() + index),
                None => self.load_workgroup(&scratch, index),
            };
            combined = combined.binary(op.binary(), next);
        }
        self.bind(combined)
    }

    /// Cross-lane reduction over lane groups of `group_size` under a combine
    /// the caller supplies, rather than one of the closed [`TileReduceOp`]s.
    ///
    /// Subgroup collectives are per-operator, so they cannot serve a general
    /// monoid. This stages every lane's partial through a block-sized
    /// workgroup array and has each lane fold its own group's slice, which
    /// needs no intrinsic and leaves every lane holding the combined value —
    /// the same broadcast contract as [`Self::group_reduce`].
    ///
    /// `combine` must be associative over the staged values; it is invoked
    /// `group_size - 1` times per lane in a fixed left-to-right order, so a
    /// non-associative body silently produces an order-dependent result.
    pub fn group_reduce_with<F>(&mut self, group_size: u32, value: Tile, mut combine: F) -> Tile
    where
        F: FnMut(&mut Self, Tile, Tile) -> Tile,
    {
        let mut combined = self.group_reduce_with_vec(
            group_size,
            vec![value],
            |program, mut acc, mut incoming| {
                let acc = acc.pop().expect("one slot");
                let incoming = incoming.pop().expect("one slot");
                vec![combine(program, acc, incoming)]
            },
        );
        combined.pop().expect("one slot")
    }

    /// Cross-lane reduction of an `N`-slot carrier under a joint combine —
    /// the tuple-valued [`Self::group_reduce_with`].
    ///
    /// A joint carrier cannot be reduced slot by slot: every outgoing slot may
    /// read every incoming one, which is exactly what makes online softmax a
    /// carrier rather than three independent reductions (its normalizer is
    /// rescaled by a factor derived from *both* sides' maxima). So each slot
    /// gets its own block-sized staging array and the whole tuple folds in one
    /// pass, leaving every lane holding the combined carrier.
    ///
    /// `combine` must be associative over the staged carriers; it is invoked
    /// `group_size - 1` times per lane in a fixed left-to-right order.
    pub fn group_reduce_with_vec<F>(
        &mut self,
        group_size: u32,
        values: Vec<Tile>,
        mut combine: F,
    ) -> Vec<Tile>
    where
        F: FnMut(&mut Self, Vec<Tile>, Vec<Tile>) -> Vec<Tile>,
    {
        let block = self.block_size();
        assert!(!values.is_empty(), "a carrier needs at least one slot");
        assert!(
            group_size > 0 && group_size <= block && block.is_multiple_of(group_size),
            "group_reduce_with requires lane groups tiling the block"
        );
        if group_size == 1 {
            return values.into_iter().map(|value| self.bind(value)).collect();
        }
        let width = values.len();
        let scratch: Vec<_> = values
            .iter()
            .map(|value| {
                self.program.alloc_tile(
                    value.element(),
                    Layout::contiguous(MemoryLevel::Workgroup, Shape::new([block])),
                )
            })
            .collect();
        let lane = self.lane();
        // Barrier before seeding: an earlier reduction through this same call
        // site (a reduce inside a loop) may still have lanes reading the
        // previous round's partials. Mirrors `group_reduce_via_subgroups`.
        self.workgroup_barrier();
        for (slot, value) in scratch.iter().zip(values) {
            self.store_workgroup(slot, lane.clone(), value);
        }
        self.workgroup_barrier();
        let base = self.bind(lane.clone() / group_size * group_size);
        if group_size.is_power_of_two() {
            // Halving tree: `log2(group_size)` inlined copies of the combine
            // instead of `group_size - 1`. That matters here in a way it does
            // not for a closed operator — a carrier's combine can be a dozen
            // transcendental ops per slot, and unrolling it 255 deep inside a
            // streaming loop produces a shader no driver compiles usefully.
            //
            // Each round's write is guarded but its barrier is not, so the
            // barrier stays workgroup-uniform.
            let local = self.bind(lane % group_size);
            let mut stride = group_size / 2;
            while stride >= 1 {
                let scratch = &scratch;
                let (base, local) = (base.clone(), local.clone());
                self.if_then(local.clone().lt(stride), |program| {
                    let target = program.bind(base + local);
                    // Bind before combining and again before storing: the
                    // results go back into the very arrays the other slots'
                    // bodies read, so an unbound load would be re-evaluated
                    // *after* a sibling slot had overwritten it.
                    let acc: Vec<Tile> = scratch
                        .iter()
                        .map(|slot| {
                            let value = program.load_workgroup(slot, target.clone());
                            program.bind(value)
                        })
                        .collect();
                    let incoming: Vec<Tile> = scratch
                        .iter()
                        .map(|slot| {
                            let value = program.load_workgroup(slot, target.clone() + stride);
                            program.bind(value)
                        })
                        .collect();
                    let next = combine(program, acc, incoming);
                    assert_eq!(next.len(), width, "joint combine changed the carrier width");
                    let next: Vec<Tile> =
                        next.into_iter().map(|value| program.bind(value)).collect();
                    for (slot, value) in scratch.iter().zip(next) {
                        program.store_workgroup(slot, target.clone(), value);
                    }
                });
                self.workgroup_barrier();
                stride /= 2;
            }
        } else {
            // No halving tree without a power-of-two group: every lane folds
            // its own group's slice left to right.
            let mut combined: Vec<Tile> = scratch
                .iter()
                .map(|slot| self.load_workgroup(slot, base.clone()))
                .collect();
            for index in 1..group_size {
                let next: Vec<Tile> = scratch
                    .iter()
                    .map(|slot| self.load_workgroup(slot, base.clone() + index))
                    .collect();
                combined = combine(self, combined, next);
                assert_eq!(combined.len(), width, "joint combine changed the carrier width");
            }
            return combined.into_iter().map(|value| self.bind(value)).collect();
        }
        // Every lane leaves holding the group's combined carrier — the same
        // broadcast contract as `group_reduce`.
        scratch
            .iter()
            .map(|slot| {
                let value = self.load_workgroup(slot, base.clone());
                self.bind(value)
            })
            .collect()
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
