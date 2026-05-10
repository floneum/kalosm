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
            },
            TileReduceOp::Min => match element {
                ElementType::F32 => Expression::Literal(Literal::F32(f32::MAX)),
                ElementType::F16 => Expression::Literal(Literal::F16(half::f16::from_f32(65504.0))),
                ElementType::U32 => Expression::Literal(Literal::U32(u32::MAX)),
                ElementType::F32Vec4 => panic!("vec4 reductions are not supported"),
                ElementType::Bool => panic!("bool reductions are not supported"),
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

    pub(in crate::lower) fn lower_tile_index_exprs<const N: usize>(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        exprs: [&TileIndexExpr; N],
        spill_depth: usize,
    ) -> Result<[Handle<Expression>; N], LowerError> {
        let mut out = [None; N];
        for (slot, expr) in out.iter_mut().zip(exprs.iter()) {
            *slot =
                Some(self.lower_tile_index_expr(expressions, scratch, body, expr, spill_depth)?);
        }
        Ok(out.map(Option::unwrap))
    }

    pub(in crate::lower) fn lower_tile_index_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        expr: &TileIndexExpr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        Ok(match expr {
            TileIndexExpr::Lane => expressions.append(
                Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
                Span::default(),
            ),
            TileIndexExpr::LoopIndex => {
                let pointer = expressions.append(
                    Expression::LocalVariable(self.current_loop_index()),
                    Span::default(),
                );
                let value = expressions.append(Expression::Load { pointer }, Span::default());
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, value)),
                    Span::default(),
                );
                value
            }
            TileIndexExpr::ProgramId(axis) => {
                let wg = expressions.append(
                    Expression::FunctionArgument(WORKGROUP_ID_ARG),
                    Span::default(),
                );
                self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::AccessIndex {
                        base: wg,
                        index: axis.index(),
                    },
                )
            }
            TileIndexExpr::SubgroupId => expressions.append(
                Expression::FunctionArgument(SUBGROUP_ID_ARG),
                Span::default(),
            ),
            TileIndexExpr::SubgroupLane => expressions.append(
                Expression::FunctionArgument(SUBGROUP_INVOCATION_ID_ARG),
                Span::default(),
            ),
            TileIndexExpr::SubgroupSize => expressions.append(
                Expression::FunctionArgument(SUBGROUP_SIZE_ARG),
                Span::default(),
            ),
            TileIndexExpr::NumSubgroups => expressions.append(
                Expression::FunctionArgument(NUM_SUBGROUPS_ARG),
                Span::default(),
            ),
            TileIndexExpr::Literal(value) => {
                expressions.append(Expression::Literal(Literal::U32(*value)), Span::default())
            }
            TileIndexExpr::Add(left, right) => {
                let left =
                    self.lower_tile_index_expr(expressions, scratch, body, left, spill_depth)?;
                let right =
                    self.lower_tile_index_expr(expressions, scratch, body, right, spill_depth)?;
                self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Binary {
                        op: BinaryOperator::Add,
                        left,
                        right,
                    },
                )
            }
            TileIndexExpr::Mul(value, literal) => {
                let value =
                    self.lower_tile_index_expr(expressions, scratch, body, value, spill_depth)?;
                let rhs = expressions
                    .append(Expression::Literal(Literal::U32(*literal)), Span::default());
                self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Binary {
                        op: BinaryOperator::Multiply,
                        left: value,
                        right: rhs,
                    },
                )
            }
            TileIndexExpr::Div(value, literal) => {
                let value =
                    self.lower_tile_index_expr(expressions, scratch, body, value, spill_depth)?;
                let rhs = expressions
                    .append(Expression::Literal(Literal::U32(*literal)), Span::default());
                self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Binary {
                        op: BinaryOperator::Divide,
                        left: value,
                        right: rhs,
                    },
                )
            }
            TileIndexExpr::Mod(value, literal) => {
                let value =
                    self.lower_tile_index_expr(expressions, scratch, body, value, spill_depth)?;
                let rhs = expressions
                    .append(Expression::Literal(Literal::U32(*literal)), Span::default());
                self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Binary {
                        op: BinaryOperator::Modulo,
                        left: value,
                        right: rhs,
                    },
                )
            }
            TileIndexExpr::Value(value) => {
                self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?
            }
        })
    }

    pub(in crate::lower) fn lower_tile_mask_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        expr: &TileMaskExpr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        Ok(match expr {
            TileMaskExpr::True => {
                expressions.append(Expression::Literal(Literal::Bool(true)), Span::default())
            }
            TileMaskExpr::Compare { op, left, right } => {
                let left =
                    self.lower_tile_index_expr(expressions, scratch, body, left, spill_depth)?;
                let right =
                    self.lower_tile_index_expr(expressions, scratch, body, right, spill_depth)?;
                let op = match op {
                    TileCompareOp::Lt => BinaryOperator::Less,
                    TileCompareOp::Le => BinaryOperator::LessEqual,
                    TileCompareOp::Gt => BinaryOperator::Greater,
                    TileCompareOp::Ge => BinaryOperator::GreaterEqual,
                    TileCompareOp::Eq => BinaryOperator::Equal,
                    TileCompareOp::Ne => BinaryOperator::NotEqual,
                };
                self.emit_tile_expr(expressions, body, Expression::Binary { op, left, right })
            }
            TileMaskExpr::And(left, right) => {
                let left =
                    self.lower_tile_mask_expr(expressions, scratch, body, left, spill_depth)?;
                let right =
                    self.lower_tile_mask_expr(expressions, scratch, body, right, spill_depth)?;
                self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Binary {
                        op: BinaryOperator::LogicalAnd,
                        left,
                        right,
                    },
                )
            }
        })
    }

    pub(in crate::lower) fn tile_expr_element(&self, expr: &TileExpr) -> Result<ElementType, LowerError> {
        match expr {
            TileExpr::Load(load) => Ok(load.src.buffer.element),
            TileExpr::LoadLinear(load) => Ok(load.src.buffer.element),
            TileExpr::LoadVec4(_) => Ok(ElementType::F32Vec4),
            TileExpr::LoadWorkgroup { src, .. } => Ok(src.element),
            TileExpr::LoadLocal(local) => Ok(local.element),
            TileExpr::QuantizedLoad(_) | TileExpr::Full(_) => Ok(ElementType::F32),
            TileExpr::Literal(value) => Ok(value.element()),
            TileExpr::Index(_) => Ok(ElementType::U32),
            TileExpr::Scalar(expr) => self.tile_scalar_expr_element(expr),
            TileExpr::Unary { value, .. } | TileExpr::Binary { left: value, .. } => {
                self.tile_expr_element(value)
            }
            TileExpr::Sum { values } => values
                .first()
                .map(|value| self.tile_expr_element(value))
                .unwrap_or(Ok(ElementType::F32)),
            TileExpr::Cast { to, .. } => Ok(*to),
            TileExpr::Bitcast { to, .. } => Ok(*to),
            TileExpr::Select { accept, .. } => self.tile_expr_element(accept),
            TileExpr::Compare { output, .. } => Ok(*output),
            TileExpr::LoopFold { initial, .. } => Ok(initial.element()),
            TileExpr::GroupReduce { scratch, .. } => Ok(scratch.element),
            TileExpr::SubgroupReduce { value, .. } => self.tile_expr_element(value),
            TileExpr::QuantizedBlockLane { .. } => Ok(ElementType::F32),
            TileExpr::Dot4 { .. }
            | TileExpr::Vec4Dot { .. }
            | TileExpr::QuantizedQ8_0Dot8 { .. }
            | TileExpr::QuantizedVecDot { .. }
            | TileExpr::QuantizedQ4KGgmlDot { .. }
            | TileExpr::QuantizedQ6KGgmlDot { .. } => Ok(ElementType::F32),
            TileExpr::Vec4Splat { .. } => Ok(ElementType::F32Vec4),
            TileExpr::PinnedRef { id } => self
                .ir
                .pinned_values
                .get(id.index())
                .map(|value| self.tile_expr_element(value).unwrap_or(ElementType::F32))
                .ok_or(LowerError::UnsupportedOperation("unknown pin id")),
            TileExpr::LoopFoldGroupOutput { group, lane } => {
                let g = self
                    .ir
                    .loop_fold_groups
                    .get(group.index())
                    .ok_or(LowerError::UnsupportedOperation("unknown fold group"))?;
                Ok(g.initials
                    .get(*lane as usize)
                    .map(|init| init.element())
                    .unwrap_or(ElementType::F32))
            }
        }
    }

    pub(in crate::lower) fn tile_scalar_expr_element(&self, expr: &TileScalarExpr) -> Result<ElementType, LowerError> {
        match expr {
            TileScalarExpr::Reduce { scratch, .. } | TileScalarExpr::LoopReduce { scratch, .. } => {
                Ok(scratch.element)
            }
            TileScalarExpr::Literal(value) => Ok(value.element()),
        }
    }

    pub(in crate::lower) fn element_scratch_index(element: ElementType) -> usize {
        match element {
            ElementType::F32 => 0,
            ElementType::F16 => 1,
            ElementType::U32 => 2,
            ElementType::F32Vec4 => 3,
            ElementType::Bool => 4,
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
        }
    }

    pub(in crate::lower) fn one_literal(element: ElementType) -> Expression {
        match element {
            ElementType::F32 => Expression::Literal(Literal::F32(1.0)),
            ElementType::F16 => Expression::Literal(Literal::F16(half::f16::from_f32(1.0))),
            ElementType::U32 => Expression::Literal(Literal::U32(1)),
            ElementType::F32Vec4 => panic!("vec4 literal requires composition"),
            ElementType::Bool => Expression::Literal(Literal::Bool(true)),
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
