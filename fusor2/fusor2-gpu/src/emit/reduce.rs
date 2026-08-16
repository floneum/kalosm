//! Cross-lane reductions: subgroup collectives, shared-memory trees, and the
//! loop-then-tree hybrid. The strategy is a parameter on the node.

use fusor2_ir::ir::kernel::{
    Builtin, ElementType, ReduceKind, ScalarElement, Tile, TileExpr, TileExprKind, TileReduceOp,
};
use fusor2_ir::target::EmitError;
use naga::{
    Barrier, BinaryOperator, Block, CollectiveOperation, Expression, Handle, MathFunction, Span,
    Statement, SubgroupOperation,
};

use super::expr::{binary_math, binary_operator};
use super::{Emitter, ScratchKind};

impl Emitter<'_> {
    pub(crate) fn reduce(
        &mut self,
        op: TileReduceOp,
        kind: &ReduceKind,
        value: &TileExpr,
        out: &mut Block,
    ) -> Result<Handle<Expression>, EmitError> {
        match kind {
            ReduceKind::Subgroup => {
                let element = value.element();
                let v = self.expr(value, out)?;
                self.subgroup_reduce(out, v, op, element)
            }
            ReduceKind::Workgroup {
                scratch,
                group_size,
            } => {
                let element = value.element();
                let v = self.expr(value, out)?;
                if self.upgrades_tree(*group_size, element) {
                    self.collective_tree_reduce(out, scratch, v, op, element)
                } else {
                    self.tree_reduce(out, scratch, v, op, *group_size)
                }
            }
            ReduceKind::Loop {
                iterations,
                index,
                scratch,
                group_size,
            } => {
                let element = value.element();
                let acc = self.loop_reduce(out, op, value, *iterations, index)?;
                if self.upgrades_tree(*group_size, element) {
                    self.collective_tree_reduce(out, scratch, acc, op, element)
                } else {
                    self.tree_reduce(out, scratch, acc, op, *group_size)
                }
            }
        }
    }

    /// The emit-side of [`crate::emit::collective_tree`]: same predicate, this
    /// kernel's block and this device's fixed width.
    fn upgrades_tree(&self, group_size: u32, element: ElementType) -> bool {
        let width = self
            .caps
            .subgroups
            .filter(|s| s.is_fixed())
            .map(|s| s.assumed());
        crate::emit::collective_tree(width, self.workgroup_invocations, group_size, element)
    }

    /// `Subgroup` — one collective, rejecting the operand shapes a collective
    /// cannot take.
    fn subgroup_reduce(
        &mut self,
        out: &mut Block,
        value: Handle<Expression>,
        op: TileReduceOp,
        element: ElementType,
    ) -> Result<Handle<Expression>, EmitError> {
        let subgroup_op = match op {
            TileReduceOp::Sum => SubgroupOperation::Add,
            TileReduceOp::Product => SubgroupOperation::Mul,
            TileReduceOp::Max => SubgroupOperation::Max,
            TileReduceOp::Min => SubgroupOperation::Min,
        };
        let scalar = match element {
            ElementType::Scalar(ScalarElement::Bool) => {
                return Err(EmitError::Unsupported(
                    "subgroup reduce on bool is undefined".into(),
                ));
            }
            ElementType::Scalar(s) => s,
            ElementType::Vector { .. } => {
                return Err(EmitError::Unsupported(
                    "subgroup reduce on a vector is unsupported".into(),
                ));
            }
            ElementType::CoopMatrix { .. } => {
                return Err(EmitError::Unsupported(
                    "subgroup reduce on a cooperative fragment is unsupported".into(),
                ));
            }
        };
        let ty = self.element_type(ElementType::Scalar(scalar))?;
        let result = self.append(Expression::SubgroupOperationResult { ty });
        out.push(
            Statement::SubgroupCollectiveOperation {
                op: subgroup_op,
                collective_op: CollectiveOperation::Reduce,
                argument: value,
                result,
            },
            Span::default(),
        );
        Ok(result)
    }

    /// Per-lane accumulation into a spill local seeded with the reduce
    /// identity, then the tree.
    fn loop_reduce(
        &mut self,
        out: &mut Block,
        op: TileReduceOp,
        value: &TileExpr,
        iterations: u32,
        index: &fusor2_ir::ir::kernel::Local,
    ) -> Result<Handle<Expression>, EmitError> {
        let element = value.element();
        let depth = self.depth;
        let acc = self.scratch_local(ScratchKind::Spill, element, depth)?;
        let identity = self.reduce_identity(op, element)?;
        self.store_local(out, acc, identity);

        let iter_local = self.private_local(index)?;
        let count = self.u32_lit(iterations);
        let value = value.clone();
        self.dynamic_loop(out, count, move |em, loop_body, loop_index| {
            em.store_local(loop_body, iter_local, loop_index);
            let v = em.expr(&value, loop_body)?;
            let current = em.load_local_handle(loop_body, acc);
            let combined = em.combine(loop_body, op, current, v);
            em.store_local(loop_body, acc, combined);
            Ok(())
        })?;
        Ok(self.load_local_handle(out, acc))
    }

    /// `Workgroup` — **barrier, seed `scratch[lane]`, barrier, then the halving
    /// tree with a barrier after each stride**.
    ///
    /// The leading barrier is load-bearing: when the scratch tile is reused
    /// inside one kernel, a lane could otherwise overwrite `scratch[lane]`
    /// while another lane still reads the previous reduction's value.
    fn tree_reduce(
        &mut self,
        out: &mut Block,
        scratch: &Tile,
        value: Handle<Expression>,
        op: TileReduceOp,
        group_size: u32,
    ) -> Result<Handle<Expression>, EmitError> {
        let block = self.workgroup_invocations;
        if group_size == 0
            || !group_size.is_power_of_two()
            || group_size > block
            || !block.is_multiple_of(group_size)
        {
            return Err(EmitError::Unsupported(format!(
                "tree reduce needs a power-of-two group size dividing the block, got \
                 {group_size} with block {block}"
            )));
        }

        let lane = self.lane();
        let lane_ptr = self.tile_dynamic_pointer(out, scratch, lane)?;
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        self.store_tile_value(out, scratch, lane_ptr, value)?;
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let (compare_index, result_index) = if group_size == block {
            let zero = self.u32_lit(0);
            (lane, zero)
        } else {
            let group_offset = self.mod_literal_u32(out, lane, group_size);
            let group_base = self.bin(out, BinaryOperator::Subtract, lane, group_offset);
            (group_offset, group_base)
        };

        let mut stride = group_size / 2;
        while stride > 0 {
            let limit = self.u32_lit(stride);
            let participates = self.bin(out, BinaryOperator::Less, compare_index, limit);
            let scratch_c = scratch.clone();
            let (accept, ()) = self.nested(move |em, accept| {
                let rhs_index = em.add_literal_u32(accept, lane, stride);
                let lhs_ptr = em.tile_dynamic_pointer(accept, &scratch_c, lane)?;
                let rhs_ptr = em.tile_dynamic_pointer(accept, &scratch_c, rhs_index)?;
                let lhs = em.load_tile_value(accept, &scratch_c, lhs_ptr)?;
                let rhs = em.load_tile_value(accept, &scratch_c, rhs_ptr)?;
                let reduced = em.combine(accept, op, lhs, rhs);
                em.store_tile_value(accept, &scratch_c, lhs_ptr, reduced)
            })?;
            out.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            out.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        let result_ptr = self.tile_dynamic_pointer(out, scratch, result_index)?;
        self.load_tile_value(out, scratch, result_ptr)
    }

    /// The whole-block tree on a fixed-subgroup-width device: one collective
    /// per subgroup, the per-subgroup partials staged through the first
    /// `block/width` scratch slots, and a serial fold every lane performs in
    /// the same order — two barriers total against the tree's
    /// `2 + log2(block)`.
    ///
    /// Every lane folds the identical slots in the identical order, so all
    /// lanes hold the same total. When one subgroup covers the block the
    /// collective alone is the reduction: no scratch, no barriers.
    fn collective_tree_reduce(
        &mut self,
        out: &mut Block,
        scratch: &Tile,
        value: Handle<Expression>,
        op: TileReduceOp,
        element: ElementType,
    ) -> Result<Handle<Expression>, EmitError> {
        let width = self.caps.subgroup_width();
        let block = self.workgroup_invocations;
        let sub = self.subgroup_reduce(out, value, op, element)?;
        if block == width {
            return Ok(sub);
        }
        let nsub = block / width;
        // The leading barrier is load-bearing for the same reason as the
        // tree's: when the scratch tile is reused inside one kernel, a leader
        // could otherwise overwrite its slot while another lane still reads
        // the previous reduction's partials.
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        let u32e = ElementType::Scalar(ScalarElement::U32);
        let sid_e = TileExpr::new(TileExprKind::Builtin(Builtin::SubgroupId), u32e);
        let sid = self.expr(&sid_e, out)?;
        let slane_e = TileExpr::new(TileExprKind::Builtin(Builtin::SubgroupLane), u32e);
        let slane = self.expr(&slane_e, out)?;
        let zero = self.u32_lit(0);
        let leader = self.bin(out, BinaryOperator::Equal, slane, zero);
        let scratch_c = scratch.clone();
        let (accept, ()) = self.nested(move |em, accept| {
            let ptr = em.tile_dynamic_pointer(accept, &scratch_c, sid)?;
            em.store_tile_value(accept, &scratch_c, ptr, sub)
        })?;
        out.push(
            Statement::If {
                condition: leader,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        let first = self.u32_lit(0);
        let ptr = self.tile_dynamic_pointer(out, scratch, first)?;
        let mut total = self.load_tile_value(out, scratch, ptr)?;
        for i in 1..nsub {
            let idx = self.u32_lit(i);
            let ptr = self.tile_dynamic_pointer(out, scratch, idx)?;
            let v = self.load_tile_value(out, scratch, ptr)?;
            total = self.combine(out, op, total, v);
        }
        Ok(total)
    }

    /// The N-ary reduction: an explicit log-tree over `lanes * block`
    /// scratch, evaluating the carrier's `merge` at every level.
    ///
    /// `Subgroup` is refused: there is no hardware collective for a
    /// multi-lane merge. `Loop` is refused too: per-lane accumulation needs
    /// the carrier's identities, which live on `Launch::Fold`, so the
    /// lowering builds that with `Stmt::Loop` and closes with the tree.
    ///
    /// Every `merge` expression reads only its formals, so all `lanes` merges
    /// are evaluated before any is written back and no level can read a slot its
    /// sibling has already overwritten.
    pub(crate) fn reduce_n(
        &mut self,
        kind: &ReduceKind,
        values: &[TileExpr],
        merge: &fusor2_ir::ir::kernel::MergeBody,
        scratch: &[Tile],
        outs: &[fusor2_ir::ir::kernel::Local],
        out: &mut Block,
    ) -> Result<(), EmitError> {
        let group_size = match kind {
            ReduceKind::Workgroup { group_size, .. } => *group_size,
            ReduceKind::Subgroup => {
                return Err(EmitError::Unsupported(
                    "a multi-lane merge has no subgroup collective: one value, one operator is \
                     all the hardware offers"
                        .into(),
                ));
            }
            ReduceKind::Loop { .. } => {
                return Err(EmitError::Unsupported(
                    "a multi-lane loop reduction seeds from the carrier's identities, which the \
                     lowering carries: build the per-lane loop with Stmt::Loop and close with \
                     ReduceKind::Workgroup"
                        .into(),
                ));
            }
        };
        let n = values.len();
        let block = self.workgroup_invocations;
        if group_size == 0
            || !group_size.is_power_of_two()
            || group_size > block
            || !block.is_multiple_of(group_size)
        {
            return Err(EmitError::Unsupported(format!(
                "tree reduce needs a power-of-two group size dividing the block, got \
                 {group_size} with block {block}"
            )));
        }

        let staged: Vec<Handle<Expression>> = values
            .iter()
            .map(|v| self.expr(v, out))
            .collect::<Result<_, _>>()?;
        let lane = self.lane();
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        for (tile, value) in scratch.iter().zip(&staged) {
            let ptr = self.tile_dynamic_pointer(out, tile, lane)?;
            self.store_tile_value(out, tile, ptr, *value)?;
        }
        out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let (compare_index, result_index) = if group_size == block {
            let zero = self.u32_lit(0);
            (lane, zero)
        } else {
            let group_offset = self.mod_literal_u32(out, lane, group_size);
            let group_base = self.bin(out, BinaryOperator::Subtract, lane, group_offset);
            (group_offset, group_base)
        };

        let mut stride = group_size / 2;
        while stride > 0 {
            let limit = self.u32_lit(stride);
            let participates = self.bin(out, BinaryOperator::Less, compare_index, limit);
            let tiles: Vec<Tile> = scratch.to_vec();
            let merge = merge.clone();
            let (accept, ()) = self.nested(move |em, accept| {
                let rhs_index = em.add_literal_u32(accept, lane, stride);
                // Both partials into the formals first: a merge reads only its
                // formals, so nothing below can observe a half-written level.
                for (i, tile) in tiles.iter().enumerate() {
                    let lhs_ptr = em.tile_dynamic_pointer(accept, tile, lane)?;
                    let value = em.load_tile_value(accept, tile, lhs_ptr)?;
                    let local = em.private_local(&merge.lhs[i])?;
                    em.store_local(accept, local, value);
                    let rhs_ptr = em.tile_dynamic_pointer(accept, tile, rhs_index)?;
                    let value = em.load_tile_value(accept, tile, rhs_ptr)?;
                    let local = em.private_local(&merge.rhs[i])?;
                    em.store_local(accept, local, value);
                }
                let merged: Vec<Handle<Expression>> = merge
                    .body
                    .iter()
                    .map(|e| em.expr(e, accept))
                    .collect::<Result<_, _>>()?;
                for (tile, value) in tiles.iter().zip(merged) {
                    let ptr = em.tile_dynamic_pointer(accept, tile, lane)?;
                    em.store_tile_value(accept, tile, ptr, value)?;
                }
                Ok(())
            })?;
            out.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            out.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        for i in 0..n {
            let ptr = self.tile_dynamic_pointer(out, &scratch[i], result_index)?;
            let value = self.load_tile_value(out, &scratch[i], ptr)?;
            let local = self.private_local(&outs[i])?;
            self.store_local(out, local, value);
        }
        Ok(())
    }

    /// The binary the reduction folds with.
    pub(crate) fn combine(
        &mut self,
        body: &mut Block,
        op: TileReduceOp,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        let binop = op.binary();
        match binary_operator(binop) {
            Some(naga_op) => self.bin(body, naga_op, left, right),
            None => {
                let fun: MathFunction =
                    binary_math(binop).expect("min/max are the only math reductions");
                self.math2(body, fun, left, right)
            }
        }
    }
}
