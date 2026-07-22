use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn tile_literal(value: TileLiteral) -> Expression {
        match value {
            TileLiteral::F32(value) => Expression::Literal(Literal::F32(value.get())),
            TileLiteral::F16(value) => {
                Expression::Literal(Literal::F16(half::f16::from_bits(value)))
            }
            TileLiteral::U32(value) => Expression::Literal(Literal::U32(value)),
            TileLiteral::Bool(value) => Expression::Literal(Literal::Bool(value)),
        }
    }

    pub(in crate::lower) fn zero_literal(element: ElementType) -> Expression {
        Self::tile_reduce_identity(TileReduceOp::Sum, element)
    }

    pub(in crate::lower) fn element_scalar(element: ElementType) -> Scalar {
        match element {
            ElementType::F32 => Scalar::F32,
            ElementType::F16 => Scalar {
                kind: ScalarKind::Float,
                width: 2,
            },
            ElementType::U32 => Scalar::U32,
            ElementType::Bool => Scalar::BOOL,
            ElementType::Vector { scalar, .. } | ElementType::CoopMatrix { scalar, .. } => {
                Self::scalar_type_inner(scalar).expect("scalar element is supported")
            }
        }
    }

    pub(in crate::lower) fn cast_tile_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        source: ElementType,
        target: ElementType,
    ) -> Handle<Expression> {
        if source == target {
            return value;
        }
        let scalar = Self::element_scalar(target);
        self.cast_as(expressions, body, value, scalar.kind, Some(scalar.width))
    }

    /// Element type a tile's backing array is emitted with: its region's
    /// canonical element, which differs from the tile's own element only
    /// when the region is shared across types.
    pub(in crate::lower) fn tile_emitted_element(&self, tile: &Tile) -> ElementType {
        match self.tile_arena.assignment.get(&super::super::tile_key(tile)) {
            Some(super::super::arena::Placement::Region { index }) => {
                self.tile_arena.regions[*index].canonical
            }
            _ => tile.element,
        }
    }

    /// Load one element through a tile pointer, bitcasting from the
    /// region's canonical type back to the tile's element when they differ.
    pub(in crate::lower) fn load_tile_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        tile: &Tile,
        pointer: Handle<Expression>,
    ) -> Handle<Expression> {
        let value = Self::emit_load(expressions, body, pointer);
        if self.tile_emitted_element(tile) == tile.element {
            return value;
        }
        let scalar = Self::element_scalar(tile.element);
        self.cast_as(expressions, body, value, scalar.kind, None)
    }

    /// Bitcast a tile-element value to the region's canonical type (when
    /// they differ) and store it through the tile pointer.
    pub(in crate::lower) fn store_tile_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        tile: &Tile,
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let emitted = self.tile_emitted_element(tile);
        let value = if emitted == tile.element {
            value
        } else {
            let scalar = Self::element_scalar(emitted);
            self.cast_as(expressions, body, value, scalar.kind, None)
        };
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    pub(in crate::lower) fn condition_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        element: ElementType,
    ) -> Handle<Expression> {
        if element == ElementType::Bool {
            return value;
        }
        let zero = expressions.append(Self::zero_literal(element), Span::default());
        self.emit(
            expressions,
            body,
            Expression::Binary {
                op: BinaryOperator::NotEqual,
                left: value,
                right: zero,
            },
        )
    }

    pub(in crate::lower) fn tile_unary_math(op: TileUnaryOp) -> Option<MathFunction> {
        Some(match op {
            TileUnaryOp::Exp => MathFunction::Exp,
            TileUnaryOp::Exp2 => MathFunction::Exp2,
            TileUnaryOp::Log => MathFunction::Log,
            TileUnaryOp::Log2 => MathFunction::Log2,
            TileUnaryOp::Sqrt => MathFunction::Sqrt,
            TileUnaryOp::InverseSqrt => MathFunction::InverseSqrt,
            TileUnaryOp::Sin => MathFunction::Sin,
            TileUnaryOp::Cos => MathFunction::Cos,
            TileUnaryOp::Tan => MathFunction::Tan,
            TileUnaryOp::Tanh => MathFunction::Tanh,
            TileUnaryOp::Asin => MathFunction::Asin,
            TileUnaryOp::Acos => MathFunction::Acos,
            TileUnaryOp::Atan => MathFunction::Atan,
            TileUnaryOp::Sinh => MathFunction::Sinh,
            TileUnaryOp::Cosh => MathFunction::Cosh,
            TileUnaryOp::Asinh => MathFunction::Asinh,
            TileUnaryOp::Acosh => MathFunction::Acosh,
            TileUnaryOp::Atanh => MathFunction::Atanh,
            TileUnaryOp::Abs => MathFunction::Abs,
            TileUnaryOp::Unpack2x16Float => MathFunction::Unpack2x16float,
            TileUnaryOp::Neg => return None,
        })
    }

    pub(in crate::lower) fn tile_binary_expression(
        op: TileBinaryOp,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Expression {
        let naga_op = match op {
            TileBinaryOp::Add => BinaryOperator::Add,
            TileBinaryOp::Sub => BinaryOperator::Subtract,
            TileBinaryOp::Mul => BinaryOperator::Multiply,
            TileBinaryOp::Div => BinaryOperator::Divide,
            TileBinaryOp::Rem => BinaryOperator::Modulo,
            TileBinaryOp::BitAnd => BinaryOperator::And,
            TileBinaryOp::BitOr => BinaryOperator::InclusiveOr,
            TileBinaryOp::BitXor => BinaryOperator::ExclusiveOr,
            TileBinaryOp::Shr => BinaryOperator::ShiftRight,
            TileBinaryOp::Shl => BinaryOperator::ShiftLeft,
            TileBinaryOp::LogicalAnd => BinaryOperator::LogicalAnd,
            TileBinaryOp::LogicalOr => BinaryOperator::LogicalOr,
            TileBinaryOp::Pow | TileBinaryOp::Min | TileBinaryOp::Max => {
                let fun = match op {
                    TileBinaryOp::Pow => MathFunction::Pow,
                    TileBinaryOp::Min => MathFunction::Min,
                    TileBinaryOp::Max => MathFunction::Max,
                    _ => unreachable!(),
                };
                return Expression::Math {
                    fun,
                    arg: left,
                    arg1: Some(right),
                    arg2: None,
                    arg3: None,
                };
            }
        };
        Expression::Binary {
            op: naga_op,
            left,
            right,
        }
    }

    pub(in crate::lower) fn tile_compare_binary(op: TileCompareOp) -> BinaryOperator {
        match op {
            TileCompareOp::Lt => BinaryOperator::Less,
            TileCompareOp::Le => BinaryOperator::LessEqual,
            TileCompareOp::Gt => BinaryOperator::Greater,
            TileCompareOp::Ge => BinaryOperator::GreaterEqual,
            TileCompareOp::Eq => BinaryOperator::Equal,
            TileCompareOp::Ne => BinaryOperator::NotEqual,
        }
    }
}
