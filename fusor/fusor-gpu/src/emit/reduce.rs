//! Cross-lane reductions: subgroup collectives and shared-memory trees.

use fusor_ir::ir::kernel::{
    Builtin, ElementType, ReduceKind, ScalarElement, Tile, TileExpr, TileExprKind, TileReduceOp,
};
use fusor_ir::target::EmitError;
use naga::{
    Barrier, BinaryOperator, Block, CollectiveOperation, Expression, Handle, MathFunction, Span,
    Statement, SubgroupOperation,
};

use super::Emitter;
use super::expr::{binary_math, binary_operator};

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
        let plan = crate::reduction::CollectivePlan::new(
            self.workgroup_invocations,
            self.caps.subgroup_width(),
        )
        .expect("collective upgrade predicate checked geometry");
        plan.emit(
            &mut NagaCollective {
                emitter: self,
                out,
                scratch,
                op,
                element,
            },
            value,
        )
    }

    /// The N-ary reduction: an explicit log-tree over `lanes * block`
    /// scratch, evaluating the carrier's `merge` at every level.
    ///
    /// `Subgroup` is refused: there is no hardware collective for a
    /// multi-lane merge. Per-lane accumulation is lowered with `Stmt::Loop`
    /// before this collective.
    ///
    /// Every `merge` expression reads only its formals, so all `lanes` merges
    /// are evaluated before any is written back and no level can read a slot its
    /// sibling has already overwritten.
    pub(crate) fn reduce_n(
        &mut self,
        kind: &ReduceKind,
        values: &[TileExpr],
        merge: &fusor_ir::ir::kernel::MergeBody,
        scratch: &[Tile],
        outs: &[fusor_ir::ir::kernel::Local],
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

/// Naga addressing adapter for the same collective recipe used by the logical
/// accessor prototype. The instruction and barrier order is unchanged.
struct NagaCollective<'a, 'caps> {
    emitter: &'a mut Emitter<'caps>,
    out: &'a mut Block,
    scratch: &'a Tile,
    op: TileReduceOp,
    element: ElementType,
}
impl crate::reduction::CollectiveEmitter for NagaCollective<'_, '_> {
    type Value = Handle<Expression>;
    type Error = EmitError;
    fn subgroup(&mut self, value: Self::Value) -> Result<Self::Value, EmitError> {
        self.emitter
            .subgroup_reduce(self.out, value, self.op, self.element)
    }
    fn barrier(&mut self) {
        self.out.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
    }
    fn store_leader(&mut self, value: Self::Value) -> Result<(), EmitError> {
        let u32e = ElementType::Scalar(ScalarElement::U32);
        let sid_e = TileExpr::new(TileExprKind::Builtin(Builtin::SubgroupId), u32e);
        let sid = self.emitter.expr(&sid_e, self.out)?;
        let lane_e = TileExpr::new(TileExprKind::Builtin(Builtin::SubgroupLane), u32e);
        let lane = self.emitter.expr(&lane_e, self.out)?;
        let zero = self.emitter.u32_lit(0);
        let leader = self
            .emitter
            .bin(self.out, BinaryOperator::Equal, lane, zero);
        let scratch = self.scratch.clone();
        let (accept, ()) = self.emitter.nested(move |em, accept| {
            let ptr = em.tile_dynamic_pointer(accept, &scratch, sid)?;
            em.store_tile_value(accept, &scratch, ptr, value)
        })?;
        self.out.push(
            Statement::If {
                condition: leader,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        Ok(())
    }
    fn load_partial(&mut self, index: u32) -> Result<Self::Value, EmitError> {
        let index = self.emitter.u32_lit(index);
        let ptr = self
            .emitter
            .tile_dynamic_pointer(self.out, self.scratch, index)?;
        self.emitter.load_tile_value(self.out, self.scratch, ptr)
    }
    fn combine(&mut self, left: Self::Value, right: Self::Value) -> Self::Value {
        self.emitter.combine(self.out, self.op, left, right)
    }
}
