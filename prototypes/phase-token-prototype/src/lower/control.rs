use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn push_emits(body: &mut Block, emits: Vec<Range<Expression>>) {
        for emit in emits {
            body.push(Statement::Emit(emit), Span::default());
        }
    }

    pub(super) fn single_expression_range(
        _expressions: &Arena<Expression>,
        value: Handle<Expression>,
    ) -> Range<Expression> {
        Range::new_from_bounds(value, value)
    }

    pub(super) fn range_from(
        _expressions: &Arena<Expression>,
        first: Handle<Expression>,
        second: Handle<Expression>,
    ) -> Range<Expression> {
        Range::new_from_bounds(first, second)
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

    pub(super) fn increment_u32_local_by_expr(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        amount: Handle<Expression>,
        multiplier: u32,
    ) -> Statement {
        let mut emits = Vec::new();
        let step = self.mul_literal_u32_emitted(expressions, amount, multiplier, &mut emits);
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let next = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: current,
                right: step,
            },
            Span::default(),
        );
        let mut body = Block::new();
        Self::push_emits(&mut body, emits);
        body.push(
            Statement::Emit(Self::range_from(expressions, current, next)),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer,
                value: next,
            },
            Span::default(),
        );
        Statement::Block(body)
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

    pub(super) fn current_loop_index(&self) -> Handle<LocalVariable> {
        self.loop_index_local
            .expect("scratch locals must be created before lowering storage offsets")
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
                let col_base = self.subgroup_base(expressions, subgroup_cols, &mut emits);
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

    fn subgroup_grid_base(
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

    fn subgroup_base(
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

    pub(super) fn bin_lit_u32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = expressions.append(Expression::Literal(Literal::U32(right)), Span::default());
        let value = expressions.append(Expression::Binary { op, left, right }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        value
    }
}
