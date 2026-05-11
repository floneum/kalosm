use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn lower_tile_loop_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        value: &Expr,
        iterations: u32,
        op: TileReduceOp,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let element = value.element();
        let acc = self.tile_expr_spill_local(scratch, element, 0)?;
        let acc_ptr = expressions.append(Expression::LocalVariable(acc), Span::default());
        let initial = expressions.append(
            Self::tile_reduce_identity(op, element),
            Span::default(),
        );
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
        let lane_ptr = self.tile_dynamic_pointer(expressions, scratch_tile, lane, body)?;
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
            let group_offset =
                self.mod_literal_u32_emitted(expressions, lane, group_size, body);
            let group_base = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: lane,
                    right: group_offset,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, group_base)),
                Span::default(),
            );
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

        let result_ptr =
            self.tile_dynamic_pointer(expressions, scratch_tile, result_index, body)?;
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
        let lhs_ptr = self.tile_dynamic_pointer(expressions, scratch_tile, lane, &mut body)?;
        let rhs_ptr =
            self.tile_dynamic_pointer(expressions, scratch_tile, rhs_index, &mut body)?;
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
            ElementType::CoopMatrixF32 { .. } => {
                return Err(LowerError::UnsupportedOperation(
                    "subgroup reduce on cooperative-matrix values is not supported",
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
