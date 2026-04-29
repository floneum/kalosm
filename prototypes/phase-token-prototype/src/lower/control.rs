use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_store_tile(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        src: TileRef,
        dst: &StorageView,
    ) -> Result<Statement, LowerError> {
        let src_layout = self.tile_layout(src)?;
        let dst_layout = self.storage_layout(dst)?;
        if src_layout.shape() != dst_layout.shape() {
            return Err(LowerError::UnsupportedOperation("store shape mismatch"));
        }

        let mut body = Block::new();
        let (flat, flat_emit) = self.load_u32_local(expressions, index_local);
        body.push(Statement::Emit(flat_emit), Span::default());

        let (src_index, src_index_emits) =
            self.index_from_flat(expressions, flat, src_layout, src_layout, &[])?;
        let (dst_index, dst_index_emits) = self.index_from_flat(
            expressions,
            flat,
            src_layout,
            dst_layout,
            &dst.dynamic_offsets,
        )?;
        Self::push_emits(&mut body, src_index_emits);
        Self::push_emits(&mut body, dst_index_emits);

        let (src_pointer, src_pointer_emits) =
            self.tile_dynamic_pointer(expressions, src, src_index)?;
        let (dst_pointer, dst_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst_index)?;
        Self::push_emits(&mut body, src_pointer_emits);
        Self::push_emits(&mut body, dst_pointer_emits);

        let value = expressions.append(
            Expression::Load {
                pointer: src_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: dst_pointer,
                value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(expressions, index_local, src_layout.element_count(), body))
    }

    pub(super) fn counted_loop(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        end: u32,
        body: Block,
    ) -> Statement {
        let init = self.store_u32_literal(expressions, index_local, 0);
        let (done, done_emit) = Self::u32_done_condition(expressions, index_local, end);
        let mut loop_body = Block::new();
        loop_body.push(Statement::Emit(done_emit), Span::default());
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        loop_body.push(Statement::Block(body), Span::default());
        loop_body.push(
            self.increment_u32_local(expressions, index_local, 1),
            Span::default(),
        );

        Statement::Block(Block::from_vec(vec![
            init,
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
        ]))
    }

    pub(super) fn store_u32_literal(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        value: u32,
    ) -> Statement {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let value = expressions.append(Expression::Literal(Literal::U32(value)), Span::default());
        Statement::Store { pointer, value }
    }

    pub(super) fn increment_u32_local(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        amount: u32,
    ) -> Statement {
        let amount = expressions.append(Expression::Literal(Literal::U32(amount)), Span::default());
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let next = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: current,
                right: amount,
            },
            Span::default(),
        );
        Statement::Block(Block::from_vec(vec![
            Statement::Emit(Self::range_from(expressions, current, next)),
            Statement::Store {
                pointer,
                value: next,
            },
        ]))
    }

    pub(super) fn load_u32_local(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
    ) -> (Handle<Expression>, Range<Expression>) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let value = expressions.append(Expression::Load { pointer }, Span::default());
        (value, Self::single_expression_range(expressions, value))
    }

    pub(super) fn subgroup_column_base(
        &self,
        expressions: &mut Arena<Expression>,
        subgroup_cols: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        self.subgroup_base(expressions, subgroup_cols, emits)
    }

    pub(super) fn subgroup_partition_bases(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        partition: CoopPartition,
        subgroup_rows: u32,
        subgroup_cols: u32,
    ) -> (Option<Handle<Expression>>, Option<Handle<Expression>>) {
        match partition {
            CoopPartition::Single => (None, None),
            CoopPartition::Rows => {
                let mut emits = Vec::new();
                let row_base = self.subgroup_base(expressions, subgroup_rows, &mut emits);
                Self::push_emits(body, emits);
                (Some(row_base), None)
            }
            CoopPartition::Columns => {
                let mut emits = Vec::new();
                let col_base = self.subgroup_column_base(expressions, subgroup_cols, &mut emits);
                Self::push_emits(body, emits);
                (None, Some(col_base))
            }
            CoopPartition::InterleavedGrid {
                row_groups: _,
                col_groups,
            } => {
                let mut row_emits = Vec::new();
                let row_base = self.subgroup_grid_base(
                    expressions,
                    COOP_MATRIX_DIM,
                    col_groups,
                    true,
                    &mut row_emits,
                );
                Self::push_emits(body, row_emits);
                let mut col_emits = Vec::new();
                let col_base = self.subgroup_grid_base(
                    expressions,
                    COOP_MATRIX_DIM,
                    col_groups,
                    false,
                    &mut col_emits,
                );
                Self::push_emits(body, col_emits);
                (Some(row_base), Some(col_base))
            }
        }
    }

    pub(super) fn coop_tile_offset(partition: CoopPartition, row_axis: bool, tile: u32) -> u32 {
        let stride_groups = match partition {
            CoopPartition::InterleavedGrid {
                row_groups,
                col_groups,
            } => {
                if row_axis {
                    row_groups
                } else {
                    col_groups
                }
            }
            _ => 1,
        };
        tile * COOP_MATRIX_DIM * stride_groups
    }

    pub(super) fn subgroup_grid_base(
        &self,
        expressions: &mut Arena<Expression>,
        extent_per_subgroup: u32,
        col_groups: u32,
        row_axis: bool,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if self.coop_subgroups <= 1 {
            return expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        }
        let subgroup_id = expressions.append(
            Expression::FunctionArgument(SUBGROUP_ID_ARG),
            Span::default(),
        );
        let group = if row_axis {
            self.div_literal_u32_emitted(expressions, subgroup_id, col_groups, emits)
        } else {
            self.mod_literal_u32_emitted(expressions, subgroup_id, col_groups, emits)
        };
        self.mul_literal_u32_emitted(expressions, group, extent_per_subgroup, emits)
    }

    pub(super) fn subgroup_base(
        &self,
        expressions: &mut Arena<Expression>,
        extent_per_subgroup: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if self.coop_subgroups <= 1 {
            return expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        }
        let subgroup_id = expressions.append(
            Expression::FunctionArgument(SUBGROUP_ID_ARG),
            Span::default(),
        );
        self.mul_literal_u32_emitted(expressions, subgroup_id, extent_per_subgroup, emits)
    }

    pub(super) fn current_loop_index(&self) -> Handle<LocalVariable> {
        self.loop_index_local
            .expect("scratch locals must be created before lowering storage offsets")
    }

    pub(super) fn u32_done_condition(
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        end: u32,
    ) -> (Handle<Expression>, Range<Expression>) {
        let end = expressions.append(Expression::Literal(Literal::U32(end)), Span::default());
        let pointer = expressions.append(Expression::LocalVariable(index_local), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let condition = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: current,
                right: end,
            },
            Span::default(),
        );

        (condition, Self::range_from(expressions, current, condition))
    }

    pub(super) fn push_emits(body: &mut Block, emits: Vec<Range<Expression>>) {
        for emit in emits {
            body.push(Statement::Emit(emit), Span::default());
        }
    }

    pub(super) fn lower_workgroup_tile_op(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        tile: TileRef,
        tile_body: impl FnOnce(
            &Self,
            &mut Arena<Expression>,
            Handle<LocalVariable>,
        ) -> Result<Block, LowerError>,
    ) -> Result<Statement, LowerError> {
        let layout = self.tile_layout(tile)?;
        let body = tile_body(self, expressions, tile_index)?;
        Ok(self.distributed_index_loop(expressions, tile_index, layout.element_count(), body))
    }

    pub(super) fn distributed_index_loop(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        end: std::num::NonZeroU32,
        body: Block,
    ) -> Statement {
        let init_index = self.init_tile_index(expressions, index_local);
        if end.get() == self.workgroup_invocations {
            return Statement::Block(Block::from_vec(vec![init_index, Statement::Block(body)]));
        }
        let mut loop_body = Block::new();
        let (done, done_emit) = Self::tile_done_condition(expressions, index_local, end);
        loop_body.push(Statement::Emit(done_emit), Span::default());
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        loop_body.push(Statement::Block(body), Span::default());
        loop_body.push(
            self.advance_tile_index(expressions, index_local),
            Span::default(),
        );

        Statement::Block(Block::from_vec(vec![
            init_index,
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
        ]))
    }

    pub(super) fn init_tile_index(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
    ) -> Statement {
        self.init_tile_index_with_offset(expressions, tile_index, 0)
    }

    pub(super) fn init_tile_index_with_offset(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        chunk: u32,
    ) -> Statement {
        let pointer = expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let lane = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let value = if chunk == 0 {
            lane
        } else {
            self.add_literal_u32(expressions, lane, chunk * self.workgroup_invocations)
        };
        Statement::Store { pointer, value }
    }

    pub(super) fn advance_tile_index(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
    ) -> Statement {
        let workgroup_size = expressions.append(
            Expression::Literal(Literal::U32(self.workgroup_invocations)),
            Span::default(),
        );
        let pointer = expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let next = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: current,
                right: workgroup_size,
            },
            Span::default(),
        );

        Statement::Block(Block::from_vec(vec![
            Statement::Emit(Self::range_from(expressions, current, next)),
            Statement::Store {
                pointer,
                value: next,
            },
        ]))
    }

    pub(super) fn tile_done_condition(
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        element_count: std::num::NonZeroU32,
    ) -> (Handle<Expression>, Range<Expression>) {
        let element_count = expressions.append(
            Expression::Literal(Literal::U32(element_count.get())),
            Span::default(),
        );
        let pointer = expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let condition = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: current,
                right: element_count,
            },
            Span::default(),
        );

        (condition, Self::range_from(expressions, current, condition))
    }
}
