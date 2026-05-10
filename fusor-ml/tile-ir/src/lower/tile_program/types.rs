use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn tile_reduce_identity(op: TileReduceOp, element: ElementType) -> Expression {
        match op {
            TileReduceOp::Sum => Self::zero_literal(element),
            TileReduceOp::Product => Self::one_literal(element),
            TileReduceOp::Max => match element {
                ElementType::F32 => Expression::Literal(Literal::F32(f32::MIN)),
                ElementType::F16 => {
                    Expression::Literal(Literal::F16(half::f16::from_f32(-65504.0)))
                }
                ElementType::U32 => Expression::Literal(Literal::U32(0)),
                ElementType::F32Vec4 => panic!("vec4 reductions are not supported"),
                ElementType::Bool => panic!("bool reductions are not supported"),
                ElementType::CoopMatrixF32 { .. } => {
                    panic!("cooperative-matrix reductions are not supported")
                }
            },
            TileReduceOp::Min => match element {
                ElementType::F32 => Expression::Literal(Literal::F32(f32::MAX)),
                ElementType::F16 => Expression::Literal(Literal::F16(half::f16::from_f32(65504.0))),
                ElementType::U32 => Expression::Literal(Literal::U32(u32::MAX)),
                ElementType::F32Vec4 => panic!("vec4 reductions are not supported"),
                ElementType::Bool => panic!("bool reductions are not supported"),
                ElementType::CoopMatrixF32 { .. } => {
                    panic!("cooperative-matrix reductions are not supported")
                }
            },
        }
    }

    pub(in crate::lower) fn tile_reduce_expression(
        op: TileReduceOp,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Expression {
        match op {
            TileReduceOp::Sum => Expression::Binary {
                op: BinaryOperator::Add,
                left,
                right,
            },
            TileReduceOp::Product => Expression::Binary {
                op: BinaryOperator::Multiply,
                left,
                right,
            },
            TileReduceOp::Max => Expression::Math {
                fun: MathFunction::Max,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
            TileReduceOp::Min => Expression::Math {
                fun: MathFunction::Min,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
        }
    }

    pub(in crate::lower) fn tile_expr_element(&self, expr: &Expr) -> Result<ElementType, LowerError> {
        match expr {
            Expr::Load(load) => Ok(load.src.buffer.element),
            Expr::LoadLinear(load) => Ok(load.src.buffer.element),
            Expr::LoadVec4(_) => Ok(ElementType::F32Vec4),
            Expr::LoadWorkgroup { src, .. } => Ok(src.element),
            Expr::LoadLocal(local) => Ok(local.element),
            Expr::QuantizedLoad(_) | Expr::Full(_) => Ok(ElementType::F32),
            Expr::Literal(value) => Ok(value.element()),
            Expr::Builtin(_) => Ok(ElementType::U32),
            Expr::Reduce { scratch, .. } => Ok(scratch.element),
            Expr::LoopReduce { scratch, .. } => Ok(scratch.element),
            Expr::Unary { value, .. } | Expr::Binary { left: value, .. } => {
                self.tile_expr_element(value)
            }
            Expr::Sum { values } => values
                .first()
                .map(|value| self.tile_expr_element(value))
                .unwrap_or(Ok(ElementType::F32)),
            Expr::Cast { to, .. } => Ok(*to),
            Expr::Bitcast { to, .. } => Ok(*to),
            Expr::Select { accept, .. } => self.tile_expr_element(accept),
            Expr::Compare { output, .. } => Ok(*output),
            Expr::LoopFold { initial, .. } => Ok(initial.element()),
            Expr::GroupReduce { scratch, .. } => Ok(scratch.element),
            Expr::SubgroupReduce { value, .. } => self.tile_expr_element(value),
            Expr::QuantizedBlockLane { .. } => Ok(ElementType::F32),
            Expr::Vec4Dot { .. }
            | Expr::QuantizedQ8_0Dot8 { .. }
            | Expr::QuantizedVecDot { .. }
            | Expr::QuantizedQ4KGgmlDot { .. }
            | Expr::QuantizedQ6KGgmlDot { .. } => Ok(ElementType::F32),
            Expr::Vec4Splat { .. } | Expr::Compose4 { .. } => Ok(ElementType::F32Vec4),
        }
    }

    pub(in crate::lower) fn element_scratch_index(element: ElementType) -> usize {
        match element {
            ElementType::F32 => 0,
            ElementType::F16 => 1,
            ElementType::U32 => 2,
            ElementType::F32Vec4 => 3,
            ElementType::Bool => 4,
            ElementType::CoopMatrixF32 { .. } => {
                panic!("cooperative-matrix scratch is not supported")
            }
        }
    }

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
        match element {
            ElementType::F32 => Expression::Literal(Literal::F32(0.0)),
            ElementType::F16 => Expression::Literal(Literal::F16(half::f16::from_f32(0.0))),
            ElementType::U32 => Expression::Literal(Literal::U32(0)),
            ElementType::F32Vec4 => panic!("vec4 literal requires composition"),
            ElementType::Bool => Expression::Literal(Literal::Bool(false)),
            ElementType::CoopMatrixF32 { .. } => panic!("cooperative-matrix has no scalar literal"),
        }
    }

    pub(in crate::lower) fn one_literal(element: ElementType) -> Expression {
        match element {
            ElementType::F32 => Expression::Literal(Literal::F32(1.0)),
            ElementType::F16 => Expression::Literal(Literal::F16(half::f16::from_f32(1.0))),
            ElementType::U32 => Expression::Literal(Literal::U32(1)),
            ElementType::F32Vec4 => panic!("vec4 literal requires composition"),
            ElementType::Bool => Expression::Literal(Literal::Bool(true)),
            ElementType::CoopMatrixF32 { .. } => panic!("cooperative-matrix has no scalar literal"),
        }
    }

    pub(in crate::lower) fn element_scalar(element: ElementType) -> Scalar {
        match element {
            ElementType::F32 => Scalar::F32,
            ElementType::F16 => Scalar {
                kind: ScalarKind::Float,
                width: 2,
            },
            ElementType::U32 => Scalar::U32,
            ElementType::F32Vec4 => Scalar::F32,
            ElementType::Bool => Scalar::BOOL,
            ElementType::CoopMatrixF32 { .. } => Scalar::F32,
        }
    }

    pub(in crate::lower) fn vec4_splat_literal(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: f32,
    ) -> Handle<Expression> {
        let value = expressions.append(Expression::Literal(Literal::F32(value)), Span::default());
        self.emit_tile_expr(
            expressions,
            body,
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: vec![value, value, value, value],
            },
        )
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
        self.emit_tile_expr(
            expressions,
            body,
            Expression::As {
                expr: value,
                kind: scalar.kind,
                convert: Some(scalar.width),
            },
        )
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
        self.emit_tile_expr(
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
        match op {
            TileBinaryOp::Add => Expression::Binary {
                op: BinaryOperator::Add,
                left,
                right,
            },
            TileBinaryOp::Sub => Expression::Binary {
                op: BinaryOperator::Subtract,
                left,
                right,
            },
            TileBinaryOp::Mul => Expression::Binary {
                op: BinaryOperator::Multiply,
                left,
                right,
            },
            TileBinaryOp::Div => Expression::Binary {
                op: BinaryOperator::Divide,
                left,
                right,
            },
            TileBinaryOp::Rem => Expression::Binary {
                op: BinaryOperator::Modulo,
                left,
                right,
            },
            TileBinaryOp::Pow => Expression::Math {
                fun: MathFunction::Pow,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
            TileBinaryOp::Min => Expression::Math {
                fun: MathFunction::Min,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
            TileBinaryOp::Max => Expression::Math {
                fun: MathFunction::Max,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
            TileBinaryOp::BitAnd => Expression::Binary {
                op: BinaryOperator::And,
                left,
                right,
            },
            TileBinaryOp::BitOr => Expression::Binary {
                op: BinaryOperator::InclusiveOr,
                left,
                right,
            },
            TileBinaryOp::BitXor => Expression::Binary {
                op: BinaryOperator::ExclusiveOr,
                left,
                right,
            },
            TileBinaryOp::LogicalAnd => Expression::Binary {
                op: BinaryOperator::LogicalAnd,
                left,
                right,
            },
            TileBinaryOp::LogicalOr => Expression::Binary {
                op: BinaryOperator::LogicalOr,
                left,
                right,
            },
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

    pub(in crate::lower) fn emit_tile_expr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expr: Expression,
    ) -> Handle<Expression> {
        let handle = expressions.append(expr, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, handle)),
            Span::default(),
        );
        handle
    }
}
