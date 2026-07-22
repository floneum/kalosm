use super::*;
use crate::ir::ReduceKind;

const WGSL_SAFE_F32_MAX: f32 = 3.40282e38;

struct LoopReduceValue<'a> {
    value: &'a Expr,
    iterations: u32,
    index: &'a Local,
    op: TileReduceOp,
    spill_depth: usize,
}

impl<'a> Lowerer<'a> {
    /// Lower an `ExprKind::Reduce`, dispatching on `ReduceKind`:
    /// - `Subgroup` → a `subgroupAdd`/`subgroupMax`/... collective.
    /// - `Workgroup { scratch, group_size }` → cross-lane shared-memory tree.
    /// - `Loop { iterations, index, scratch, group_size }` → per-lane
    ///   accumulation across `iterations` loop iterations, then the tree.
    pub(in crate::lower) fn lower_reduce(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: TileReduceOp,
        kind: &ReduceKind,
        value: &Expr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match kind {
            ReduceKind::Subgroup => {
                let element = value.element();
                let value = self.lower_expr_lane(expressions, body, value, spill_depth)?;
                self.lower_subgroup_reduce_value(expressions, body, value, op, element)
            }
            ReduceKind::Workgroup {
                scratch,
                group_size,
            } => {
                let value = self.lower_expr_lane(expressions, body, value, spill_depth)?;
                self.lower_reduce_value(expressions, body, scratch, value, op, *group_size)
            }
            ReduceKind::Loop {
                iterations,
                index,
                scratch,
                group_size,
            } => {
                let value = self.lower_loop_reduce_value(
                    expressions,
                    body,
                    LoopReduceValue {
                        value,
                        iterations: *iterations,
                        index,
                        op,
                        spill_depth,
                    },
                )?;
                self.lower_reduce_value(expressions, body, scratch, value, op, *group_size)
            }
        }
    }

    fn lower_loop_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        reduce: LoopReduceValue<'_>,
    ) -> Result<Handle<Expression>, LowerError> {
        let LoopReduceValue {
            value,
            iterations,
            index,
            op,
            spill_depth,
        } = reduce;
        let element = value.element();
        let acc = self.scratch_local(ScratchKind::Spill, element, 0)?;
        let initial = expressions.append(Self::tile_reduce_identity(op, element), Span::default());
        self.store_local(expressions, body, acc, initial);

        let iter_var_local = self.private_local(index)?;
        self.emit_counted_loop(
            expressions,
            body,
            iterations,
            |expressions, loop_body, loop_index| {
                self.store_local(expressions, loop_body, iter_var_local, loop_index);
                let saved = self.snapshot_loop_caches();
                let value = self.lower_expr_lane(expressions, loop_body, value, spill_depth + 1)?;
                self.restore_loop_caches(saved);
                let acc_ptr = self.local_var(expressions, acc);
                let acc_value = Self::emit_load(expressions, loop_body, acc_ptr);
                let reduced = self.emit(
                    expressions,
                    loop_body,
                    Self::tile_reduce_expression(op, acc_value, value),
                );
                self.store_local(expressions, loop_body, acc, reduced);
                Ok(())
            },
        )?;

        let acc_ptr = self.local_var(expressions, acc);
        Ok(Self::emit_load(expressions, body, acc_ptr))
    }

    fn lower_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        scratch_tile: &Tile,
        value: Handle<Expression>,
        op: TileReduceOp,
        group_size: u32,
    ) -> Result<Handle<Expression>, LowerError> {
        if group_size == 0
            || !group_size.is_power_of_two()
            || group_size > self.workgroup_invocations
            || !self.workgroup_invocations.is_multiple_of(group_size)
        {
            return Err(LowerError::UnsupportedOperation(
                "tile reduce requires a power-of-two group size that divides the block",
            ));
        }

        let lane = Self::function_arg(expressions, LOCAL_INVOCATION_INDEX_ARG);
        let lane_ptr = self.tile_dynamic_pointer(expressions, scratch_tile, lane, body)?;
        // Barrier *before* seeding the scratch: when this reduction reuses
        // workgroup memory that a previous reduction/staging in the same kernel
        // already touched (e.g. fused decode kernels), a lane could otherwise
        // overwrite `scratch[lane]` while another lane is still reading the
        // prior value, producing a non-deterministic reduction. Without
        // subgroups every reduction goes through this path, so the hazard only
        // shows up on devices that disable subgroups (the web build).
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        self.store_tile_value(expressions, body, scratch_tile, lane_ptr, value);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let (compare_index, result_index) = if group_size == self.workgroup_invocations {
            let zero = self.u32(expressions, 0);
            (lane, zero)
        } else {
            let group_offset = self.mod_literal_u32_emitted(expressions, lane, group_size, body);
            let group_base = self.emit(
                expressions,
                body,
                Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: lane,
                    right: group_offset,
                },
            );
            (group_offset, group_base)
        };

        let mut stride = group_size / 2;
        while stride > 0 {
            let limit = self.u32(expressions, stride);
            let participates = self.emit(
                expressions,
                body,
                Expression::Binary {
                    op: BinaryOperator::Less,
                    left: compare_index,
                    right: limit,
                },
            );
            let accept = self.lower_reduce_step(expressions, scratch_tile, lane, stride, op)?;
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
        Ok(self.load_tile_value(expressions, body, scratch_tile, result_ptr))
    }

    fn lower_reduce_step(
        &self,
        expressions: &mut Arena<Expression>,
        scratch_tile: &Tile,
        lane: Handle<Expression>,
        stride: u32,
        op: TileReduceOp,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        let rhs_index = self.add_literal_u32_emitted(expressions, lane, stride, &mut body);
        let lhs_ptr = self.tile_dynamic_pointer(expressions, scratch_tile, lane, &mut body)?;
        let rhs_ptr = self.tile_dynamic_pointer(expressions, scratch_tile, rhs_index, &mut body)?;
        let lhs = self.load_tile_value(expressions, &mut body, scratch_tile, lhs_ptr);
        let rhs = self.load_tile_value(expressions, &mut body, scratch_tile, rhs_ptr);
        let reduced = self.emit(
            expressions,
            &mut body,
            Self::tile_reduce_expression(op, lhs, rhs),
        );
        self.store_tile_value(expressions, &mut body, scratch_tile, lhs_ptr, reduced);
        Ok(body)
    }

    fn lower_subgroup_reduce_value(
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
            ElementType::F16 => self.element_type(ElementType::F16).map_err(|_| {
                LowerError::UnsupportedOperation("subgroup reduce on f16 requires f16 capability")
            })?,
            ElementType::U32 => self.u32_ty,
            ElementType::Vector { .. } => {
                return Err(LowerError::UnsupportedOperation(
                    "subgroup reduce on vector values is not supported",
                ));
            }
            ElementType::Bool => {
                return Err(LowerError::UnsupportedOperation(
                    "subgroup reduce on bool values is not supported",
                ));
            }
            ElementType::CoopMatrix { .. } => {
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

    pub(in crate::lower) fn tile_reduce_identity(
        op: TileReduceOp,
        element: ElementType,
    ) -> Expression {
        let (f32_value, f16_value, u32_value, bool_value) = match op {
            TileReduceOp::Sum => (0.0_f32, 0.0_f32, 0_u32, false),
            TileReduceOp::Product => (1.0_f32, 1.0_f32, 1_u32, true),
            TileReduceOp::Max => (-WGSL_SAFE_F32_MAX, -65504.0, 0_u32, false),
            TileReduceOp::Min => (WGSL_SAFE_F32_MAX, 65504.0, u32::MAX, true),
        };
        match element {
            ElementType::F32 => Expression::Literal(Literal::F32(f32_value)),
            ElementType::F16 => Expression::Literal(Literal::F16(half::f16::from_f32(f16_value))),
            ElementType::U32 => Expression::Literal(Literal::U32(u32_value)),
            ElementType::Bool => Expression::Literal(Literal::Bool(bool_value)),
            ElementType::Vector { .. } => panic!("vector reductions are not supported"),
            ElementType::CoopMatrix { .. } => {
                panic!("cooperative-matrix reductions are not supported")
            }
        }
    }

    pub(in crate::lower) fn tile_reduce_expression(
        op: TileReduceOp,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Expression {
        Self::tile_binary_expression(op.binary(), left, right)
    }
}
