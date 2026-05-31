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
