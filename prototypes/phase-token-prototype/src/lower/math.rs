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

        match decl.origin {
            TileOrigin::Allocation => Ok((tile, 0)),
            TileOrigin::View { source, mapping } => {
                let (root, base_offset) = self.storage_tile_and_offset(source)?;
                let source_layout = self.tile_layout(source)?;
                let local_offset = match mapping {
                    ViewMapping::Partition { origin, .. } => {
                        Self::linear_index_prefix(source_layout, &origin)?
                    }
                };
                Ok((
                    root,
                    base_offset.checked_add(local_offset).ok_or(
                        LowerError::UnsupportedOperation("tile view offset overflow"),
                    )?,
                ))
            }
        }
    }

    pub(super) fn matrix_shape(layout: &Layout) -> Result<[u32; 2], LowerError> {
        if layout.shape().rank() != 2 {
            return Err(LowerError::UnsupportedOperation("non-matrix mma"));
        }
        Ok([
            layout.shape().dims()[0].get(),
            layout.shape().dims()[1].get(),
        ])
    }

    pub(super) fn linear_index_prefix(layout: &Layout, coords: &[u32]) -> Result<u32, LowerError> {
        let rank = layout.strides().rank();
        if coords.len() > rank && coords[rank..].iter().any(|coord| *coord != 0) {
            return Err(LowerError::UnsupportedOperation("layout rank mismatch"));
        }
        Ok(coords
            .iter()
            .take(rank)
            .zip(layout.strides().values())
            .map(|(coord, stride)| coord * stride)
            .sum())
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

    pub(super) fn mul_literal_u32(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
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
        expressions.append(
            Expression::Binary {
                op: BinaryOperator::Multiply,
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

    pub(super) fn single_expression_range(
        expressions: &Arena<Expression>,
        handle: Handle<Expression>,
    ) -> Range<Expression> {
        Self::range_from(expressions, handle, handle)
    }

    pub(super) fn range_from(
        expressions: &Arena<Expression>,
        first: Handle<Expression>,
        last: Handle<Expression>,
    ) -> Range<Expression> {
        Range::from_index_range(first.index() as u32..last.index() as u32 + 1, expressions)
    }
}
