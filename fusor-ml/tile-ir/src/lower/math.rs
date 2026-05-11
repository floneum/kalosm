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

    pub(super) fn add_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        body: &mut Block,
    ) -> Handle<Expression> {
        if literal == 0 {
            return value;
        }
        if let Some(folded) = Self::u32_literal(expressions, value) {
            return self.u32(expressions,folded + literal);
        }
        let rhs = self.u32(expressions,literal);
        self.emit(expressions, body, Expression::Binary {
            op: BinaryOperator::Add,
            left: value,
            right: rhs,
        })
    }

    pub(super) fn mul_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        body: &mut Block,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        if let Some(folded) = Self::u32_literal(expressions, value) {
            return self.u32(expressions,folded * literal);
        }
        let rhs = self.u32(expressions,literal);
        self.emit(expressions, body, Expression::Binary {
            op: BinaryOperator::Multiply,
            left: value,
            right: rhs,
        })
    }

    pub(super) fn div_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        body: &mut Block,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        if let Some(folded) = Self::u32_literal(expressions, value) {
            return self.u32(expressions,folded / literal);
        }
        let (op, rhs) = if literal.is_power_of_two() {
            (
                BinaryOperator::ShiftRight,
                self.u32(expressions,literal.trailing_zeros()),
            )
        } else {
            (BinaryOperator::Divide, self.u32(expressions,literal))
        };
        self.emit(expressions, body, Expression::Binary { op, left: value, right: rhs })
    }

    pub(super) fn mod_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        body: &mut Block,
    ) -> Handle<Expression> {
        if literal == 1 {
            return self.u32(expressions,0);
        }
        if let Some(folded) = Self::u32_literal(expressions, value) {
            return self.u32(expressions,folded % literal);
        }
        let (op, rhs) = if literal.is_power_of_two() {
            (BinaryOperator::And, self.u32(expressions,literal - 1))
        } else {
            (BinaryOperator::Modulo, self.u32(expressions,literal))
        };
        self.emit(expressions, body, Expression::Binary { op, left: value, right: rhs })
    }

}
