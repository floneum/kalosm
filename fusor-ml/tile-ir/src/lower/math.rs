use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn storage_tile_and_offset(
        &self,
        tile: TileRef,
    ) -> Result<(TileRef, u32), LowerError> {
        let decl = self
            .ir
            .tiles()
            .get(tile.id.index())
            .ok_or(LowerError::UnknownTile(tile.id))?;
        if decl.element != tile.element {
            return Err(LowerError::TileElementMismatch {
                tile: tile.id,
                declared: decl.element,
                used: tile.element,
            });
        }
        Ok((tile, 0))
    }

    pub(super) fn add_literal_u32(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            return value;
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value + literal)),
                Span::default(),
            );
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: value,
                right: literal,
            },
            Span::default(),
        )
    }

    pub(super) fn add_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 0 {
            return value;
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value + literal)),
                Span::default(),
            );
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: value,
                right: literal,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    pub(super) fn mul_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value * literal)),
                Span::default(),
            );
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Multiply,
                left: value,
                right: literal,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    pub(super) fn div_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value / literal)),
                Span::default(),
            );
        }
        let value = if literal.is_power_of_two() {
            let shift = expressions.append(
                Expression::Literal(Literal::U32(literal.trailing_zeros())),
                Span::default(),
            );
            expressions.append(
                Expression::Binary {
                    op: BinaryOperator::ShiftRight,
                    left: value,
                    right: shift,
                },
                Span::default(),
            )
        } else {
            let literal =
                expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
            expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Divide,
                    left: value,
                    right: literal,
                },
                Span::default(),
            )
        };
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    pub(super) fn mod_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 1 {
            return expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        }
        if let Some(value) = Self::u32_literal(expressions, value) {
            return expressions.append(
                Expression::Literal(Literal::U32(value % literal)),
                Span::default(),
            );
        }
        let value = if literal.is_power_of_two() {
            let mask = expressions.append(
                Expression::Literal(Literal::U32(literal - 1)),
                Span::default(),
            );
            expressions.append(
                Expression::Binary {
                    op: BinaryOperator::And,
                    left: value,
                    right: mask,
                },
                Span::default(),
            )
        } else {
            let literal =
                expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
            expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Modulo,
                    left: value,
                    right: literal,
                },
                Span::default(),
            )
        };
        emits.push(Self::single_expression_range(expressions, value));
        value
    }
}
