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
