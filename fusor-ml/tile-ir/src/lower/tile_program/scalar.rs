use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn lower_tile_scalar_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        expr: &TileScalarExpr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match expr {
            TileScalarExpr::Literal(value) => {
                Ok(expressions.append(Self::tile_literal(*value), Span::default()))
            }
            TileScalarExpr::Reduce {
                op,
                value,
                scratch: scratch_tile,
            } => {
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                self.lower_tile_reduce_value(
                    expressions,
                    body,
                    value,
                    *scratch_tile,
                    *op,
                    self.workgroup_invocations,
                )
            }
            TileScalarExpr::LoopReduce {
                op,
                iterations,
                value,
                scratch: scratch_tile,
            } => {
                let value = self.lower_tile_loop_reduce_value(
                    expressions,
                    scratch,
                    body,
                    value,
                    *iterations,
                    *op,
                    spill_depth,
                )?;
                self.lower_tile_reduce_value(
                    expressions,
                    body,
                    value,
                    *scratch_tile,
                    *op,
                    self.workgroup_invocations,
                )
            }
        }
    }

    pub(in crate::lower) fn lower_tile_loop_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        value: &TileExpr,
        iterations: u32,
        op: TileReduceOp,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let element = self.tile_expr_element(value)?;
        self.lower_tile_loop_accumulate_value(
            expressions,
            scratch,
            body,
            value,
            iterations,
            op,
            element,
            Self::tile_reduce_identity(op, element),
            0,
            spill_depth,
        )
    }

    pub(in crate::lower) fn lower_tile_loop_fold_value(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        value: &TileExpr,
        iterations: u32,
        op: TileReduceOp,
        initial: TileLiteral,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let element = initial.element();
        self.lower_tile_loop_accumulate_value(
            expressions,
            scratch,
            body,
            value,
            iterations,
            op,
            element,
            Self::tile_literal(initial),
            spill_depth,
            spill_depth,
        )
    }

    pub(in crate::lower) fn lower_tile_loop_accumulate_value(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        value: &TileExpr,
        iterations: u32,
        op: TileReduceOp,
        element: ElementType,
        initial: Expression,
        acc_spill_depth: usize,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let acc = self.tile_expr_spill_local(scratch, element, acc_spill_depth)?;
        let acc_ptr = expressions.append(Expression::LocalVariable(acc), Span::default());
        let initial = expressions.append(initial, Span::default());
        body.push(
            Statement::Store {
                pointer: acc_ptr,
                value: initial,
            },
            Span::default(),
        );

        self.emit_counted_loop(
            expressions,
            scratch,
            body,
            iterations,
            |expressions, loop_body, _| {
                // Cache entries reference values scoped to the outer block. Snapshot
                // expression-handle caches, but drop q8 activation locals because the
                // loop body may overwrite the shared scratch slots.
                let saved = self.snapshot_tile_loop_caches();
                let value = self.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    loop_body,
                    value,
                    spill_depth + 1,
                )?;
                self.restore_tile_loop_caches(saved);
                let acc =
                    expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
                loop_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, acc)),
                    Span::default(),
                );
                let reduced = self.emit_tile_expr(
                    expressions,
                    loop_body,
                    Self::tile_reduce_expression(op, acc, value),
                );
                loop_body.push(
                    Statement::Store {
                        pointer: acc_ptr,
                        value: reduced,
                    },
                    Span::default(),
                );
                Ok(())
            },
        )?;

        let value = expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        Ok(value)
    }

    pub(in crate::lower) fn lower_tile_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        scratch_tile: TileRef,
        op: TileReduceOp,
        group_size: u32,
    ) -> Result<Handle<Expression>, LowerError> {
        if group_size == 0
            || !group_size.is_power_of_two()
            || group_size > self.workgroup_invocations
            || self.workgroup_invocations % group_size != 0
        {
            return Err(LowerError::UnsupportedOperation(
                "tile reduce requires a power-of-two group size that divides the block",
            ));
        }

        let lane = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let (lane_ptr, lane_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, lane)?;
        Self::push_emits(body, lane_ptr_emits);
        body.push(
            Statement::Store {
                pointer: lane_ptr,
                value,
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let (compare_index, result_index) = if group_size == self.workgroup_invocations {
            let zero =
                expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
            (lane, zero)
        } else {
            let mut index_emits = Vec::new();
            let group_offset =
                self.mod_literal_u32_emitted(expressions, lane, group_size, &mut index_emits);
            let group_base = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: lane,
                    right: group_offset,
                },
                Span::default(),
            );
            index_emits.push(Self::single_expression_range(expressions, group_base));
            Self::push_emits(body, index_emits);
            (group_offset, group_base)
        };

        let mut stride = group_size / 2;
        while stride > 0 {
            let limit =
                expressions.append(Expression::Literal(Literal::U32(stride)), Span::default());
            let participates = self.emit_tile_expr(
                expressions,
                body,
                Expression::Binary {
                    op: BinaryOperator::Less,
                    left: compare_index,
                    right: limit,
                },
            );
            let accept =
                self.lower_tile_reduce_step(expressions, scratch_tile, lane, stride, op)?;
            body.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            body.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        let (result_ptr, result_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, result_index)?;
        Self::push_emits(body, result_ptr_emits);
        let result = expressions.append(
            Expression::Load {
                pointer: result_ptr,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, result)),
            Span::default(),
        );
        Ok(result)
    }

    pub(in crate::lower) fn lower_tile_reduce_step(
        &self,
        expressions: &mut Arena<Expression>,
        scratch_tile: TileRef,
        lane: Handle<Expression>,
        stride: u32,
        op: TileReduceOp,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        let rhs_index = self.add_literal_u32(expressions, lane, stride);
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, rhs_index)),
            Span::default(),
        );
        let (lhs_ptr, lhs_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, lane)?;
        let (rhs_ptr, rhs_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, rhs_index)?;
        Self::push_emits(&mut body, lhs_ptr_emits);
        Self::push_emits(&mut body, rhs_ptr_emits);
        let lhs = expressions.append(Expression::Load { pointer: lhs_ptr }, Span::default());
        let rhs = expressions.append(Expression::Load { pointer: rhs_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::range_from(expressions, lhs, rhs)),
            Span::default(),
        );
        let reduced = self.emit_tile_expr(
            expressions,
            &mut body,
            Self::tile_reduce_expression(op, lhs, rhs),
        );
        body.push(
            Statement::Store {
                pointer: lhs_ptr,
                value: reduced,
            },
            Span::default(),
        );
        Ok(body)
    }

    pub(in crate::lower) fn lower_tile_loop_fold_group_output(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        group: LoopFoldGroupId,
        lane: u32,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        if let Some(values) = self.loop_fold_group_cache.borrow().get(&group).cloned() {
            return Ok(values.get(lane as usize).copied().ok_or(
                LowerError::UnsupportedOperation("fold group lane out of range"),
            )?);
        }

        let g = self
            .ir
            .loop_fold_groups
            .get(group.index())
            .ok_or(LowerError::UnsupportedOperation("unknown fold group"))?
            .clone();
        let n = g.bodies.len();
        if g.initials.len() != n {
            return Err(LowerError::UnsupportedOperation(
                "fold group initial count mismatch",
            ));
        }
        let offset = self.fold_group_offsets.get(group.index()).copied().ok_or(
            LowerError::UnsupportedOperation("fold group offset missing"),
        )?;
        if offset + n > self.fold_accumulator_locals.len() {
            return Err(LowerError::UnsupportedOperation(
                "fold group accumulator pool exhausted",
            ));
        }
        let acc_locals: Vec<_> = (0..n)
            .map(|i| self.fold_accumulator_locals[offset + i])
            .collect();

        // Initialize accumulators.
        for (i, local) in acc_locals.iter().enumerate() {
            let init = expressions.append(Self::tile_literal(g.initials[i]), Span::default());
            let ptr = expressions.append(Expression::LocalVariable(*local), Span::default());
            body.push(
                Statement::Store {
                    pointer: ptr,
                    value: init,
                },
                Span::default(),
            );
        }

        self.emit_counted_loop(
            expressions,
            scratch,
            body,
            g.iterations,
            |expressions, loop_body, _| {
                // Snapshot caches that reference handles in the outer block; the loop
                // body emits a fresh scope. Q8 activation packs use shared scratch
                // locals, so discard that cache around loops instead of restoring it.
                let saved = self.snapshot_tile_loop_caches();

                // Lower each body[i] within the same loop body, accumulate into acc_locals[i].
                for (i, body_expr) in g.bodies.iter().enumerate() {
                    let value = self.lower_tile_expr_lane(
                        expressions,
                        scratch,
                        loop_body,
                        body_expr,
                        spill_depth + 1,
                    )?;
                    let acc_ptr = expressions
                        .append(Expression::LocalVariable(acc_locals[i]), Span::default());
                    let acc_load =
                        expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
                    loop_body.push(
                        Statement::Emit(Self::single_expression_range(expressions, acc_load)),
                        Span::default(),
                    );
                    let reduced = self.emit_tile_expr(
                        expressions,
                        loop_body,
                        Self::tile_reduce_expression(g.op, acc_load, value),
                    );
                    loop_body.push(
                        Statement::Store {
                            pointer: acc_ptr,
                            value: reduced,
                        },
                        Span::default(),
                    );
                }

                self.restore_tile_loop_caches(saved);
                Ok(())
            },
        )?;

        // Materialize per-accumulator final loads in the outer block; cache them.
        let mut handles = Vec::with_capacity(n);
        for local in &acc_locals {
            let ptr = expressions.append(Expression::LocalVariable(*local), Span::default());
            let value = expressions.append(Expression::Load { pointer: ptr }, Span::default());
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            handles.push(value);
        }
        self.loop_fold_group_cache
            .borrow_mut()
            .insert(group, handles.clone());
        Ok(handles[lane as usize])
    }

    pub(in crate::lower) fn lower_tile_subgroup_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        op: TileReduceOp,
        element: ElementType,
    ) -> Result<Handle<Expression>, LowerError> {
        let subgroup_op = match op {
            TileReduceOp::Sum => SubgroupOperation::Add,
            TileReduceOp::Product => SubgroupOperation::Mul,
            TileReduceOp::Max => SubgroupOperation::Max,
            TileReduceOp::Min => SubgroupOperation::Min,
        };
        let result_ty = match element {
            ElementType::F32 => self.f32_ty,
            ElementType::F16 => self.f16_ty.ok_or(LowerError::UnsupportedOperation(
                "subgroup reduce on f16 requires f16 capability",
            ))?,
            ElementType::U32 => self.u32_ty,
            ElementType::F32Vec4 => {
                return Err(LowerError::UnsupportedOperation(
                    "subgroup reduce on vec4 values is not supported",
                ));
            }
            ElementType::Bool => {
                return Err(LowerError::UnsupportedOperation(
                    "subgroup reduce on bool values is not supported",
                ));
            }
        };
        let result = expressions.append(
            Expression::SubgroupOperationResult { ty: result_ty },
            Span::default(),
        );
        body.push(
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

}
