use super::*;
use crate::ir::Builtin;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn lower_tile_expr_lane(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        expr: &Expr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match expr {
            Expr::Load(load) => {
                self.lower_tile_load_expr(expressions, scratch, body, load, spill_depth)
            }
            Expr::LoadLinear(load) => {
                self.lower_tile_linear_load_expr(expressions, scratch, body, load, spill_depth)
            }
            Expr::LoadWorkgroup { src, index } => {
                let index =
                    self.lower_tile_expr_lane(expressions, scratch, body, index, spill_depth)?;
                let (ptr, emits) = self.tile_dynamic_pointer(expressions, *src, index)?;
                Self::push_emits(body, emits);
                Ok(Self::emit_load(expressions, body, ptr))
            }
            Expr::LoadLocal(local) => {
                let local = self.private_local(*local)?;
                let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
                Ok(Self::emit_load(expressions, body, pointer))
            }
            Expr::QuantizedLoad(load) => {
                self.lower_tile_quantized_load_expr(expressions, scratch, body, load, spill_depth)
            }
            Expr::Literal(value) => {
                Ok(expressions.append(Self::tile_literal(*value), Span::default()))
            }
            Expr::Builtin(builtin) => Ok(self.lower_builtin(expressions, body, *builtin)),
            Expr::Reduce {
                op,
                iterations,
                value,
                scratch: scratch_tile,
            } => {
                let value = if *iterations == 1 {
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?
                } else {
                    self.lower_tile_loop_reduce_value(
                        expressions,
                        scratch,
                        body,
                        value,
                        *iterations,
                        *op,
                        spill_depth,
                    )?
                };
                self.lower_tile_reduce_value(
                    expressions,
                    body,
                    value,
                    *scratch_tile,
                    *op,
                    self.workgroup_invocations,
                )
            }
            Expr::Unary { op, value } => {
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                let expr = match Self::tile_unary_math(*op) {
                    Some(fun) => Expression::Math {
                        fun,
                        arg: value,
                        arg1: None,
                        arg2: None,
                        arg3: None,
                    },
                    None => match op {
                        TileUnaryOp::Neg => Expression::Unary {
                            op: naga::UnaryOperator::Negate,
                            expr: value,
                        },
                        _ => unreachable!(),
                    },
                };
                Ok(self.emit_tile_expr(expressions, body, expr))
            }
            Expr::Binary { op, left, right } => {
                let left =
                    self.lower_tile_expr_lane(expressions, scratch, body, left, spill_depth + 1)?;
                let right =
                    self.lower_tile_expr_lane(expressions, scratch, body, right, spill_depth + 1)?;
                let expr = Self::tile_binary_expression(*op, left, right);
                Ok(self.emit_tile_expr(expressions, body, expr))
            }
            Expr::Cast { value, to } => {
                let source = self.tile_expr_element(value)?;
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                Ok(self.cast_tile_value(expressions, body, value, source, *to))
            }
            Expr::Bitcast { value, to } => {
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                let scalar = Self::element_scalar(*to);
                Ok(self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::As {
                        expr: value,
                        kind: scalar.kind,
                        convert: None,
                    },
                ))
            }
            Expr::Select {
                condition,
                accept,
                reject,
            } => {
                let condition_ty = self.tile_expr_element(condition)?;
                let condition = self.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    body,
                    condition,
                    spill_depth + 1,
                )?;
                let condition = self.condition_value(expressions, body, condition, condition_ty);
                let accept =
                    self.lower_tile_expr_lane(expressions, scratch, body, accept, spill_depth + 1)?;
                let reject =
                    self.lower_tile_expr_lane(expressions, scratch, body, reject, spill_depth + 1)?;
                Ok(self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Select {
                        condition,
                        accept,
                        reject,
                    },
                ))
            }
            Expr::Compare {
                op,
                left,
                right,
                output,
            } => {
                let left =
                    self.lower_tile_expr_lane(expressions, scratch, body, left, spill_depth + 1)?;
                let right =
                    self.lower_tile_expr_lane(expressions, scratch, body, right, spill_depth + 1)?;
                let condition = self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Binary {
                        op: Self::tile_compare_binary(*op),
                        left,
                        right,
                    },
                );
                if *output == ElementType::Bool {
                    return Ok(condition);
                }
                let one = expressions.append(Self::one_literal(*output), Span::default());
                let zero = expressions.append(Self::zero_literal(*output), Span::default());
                Ok(self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Select {
                        condition,
                        accept: one,
                        reject: zero,
                    },
                ))
            }
            Expr::GroupReduce {
                op,
                value,
                scratch: scratch_tile,
                group_size,
            } => {
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                self.lower_tile_reduce_value(
                    expressions,
                    body,
                    value,
                    *scratch_tile,
                    *op,
                    *group_size,
                )
            }
            Expr::SubgroupReduce { op, value } => {
                let element = self.tile_expr_element(value)?;
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                self.lower_tile_subgroup_reduce_value(expressions, body, value, *op, element)
            }
            Expr::QuantizedBlockLane {
                id,
                src,
                k_base,
                col,
                mask,
                fill,
                block_n,
                lane,
            } => self.lower_tile_quantized_block_lane(
                expressions,
                scratch,
                body,
                *id,
                src,
                k_base,
                col,
                mask,
                *fill,
                *block_n,
                *lane,
                spill_depth,
            ),
            Expr::Compose4 { values } => {
                let mut handles = Vec::with_capacity(4);
                for value in values.iter() {
                    handles.push(self.lower_tile_expr_lane(
                        expressions,
                        scratch,
                        body,
                        value,
                        spill_depth + 1,
                    )?);
                }
                Ok(self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Compose {
                        ty: self.f32_vec4_ty,
                        components: handles,
                    },
                ))
            }
            Expr::Vec4Dot { left, right } => {
                let left =
                    self.lower_tile_expr_lane(expressions, scratch, body, left, spill_depth + 1)?;
                let right =
                    self.lower_tile_expr_lane(expressions, scratch, body, right, spill_depth + 1)?;
                let dot = expressions.append(
                    Expression::Math {
                        fun: MathFunction::Dot,
                        arg: left,
                        arg1: Some(right),
                        arg2: None,
                        arg3: None,
                    },
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, dot)),
                    Span::default(),
                );
                Ok(dot)
            }
            Expr::QuantizedDot {
                src,
                activations,
                k,
                col,
                mask,
                fill,
                block_n,
            } => self.lower_tile_quantized_dot_expr(
                expressions,
                scratch,
                body,
                src,
                activations,
                k,
                col,
                mask,
                *fill,
                *block_n,
                spill_depth,
            ),
        }
    }

    pub(in crate::lower) fn lower_builtin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        builtin: Builtin,
    ) -> Handle<Expression> {
        match builtin {
            Builtin::Lane => expressions.append(
                Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
                Span::default(),
            ),
            Builtin::LoopIndex => {
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
            Builtin::ProgramId(axis) => {
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
            Builtin::SubgroupId => expressions.append(
                Expression::FunctionArgument(SUBGROUP_ID_ARG),
                Span::default(),
            ),
            Builtin::SubgroupLane => expressions.append(
                Expression::FunctionArgument(SUBGROUP_INVOCATION_ID_ARG),
                Span::default(),
            ),
            Builtin::SubgroupSize => expressions.append(
                Expression::FunctionArgument(SUBGROUP_SIZE_ARG),
                Span::default(),
            ),
            Builtin::NumSubgroups => expressions.append(
                Expression::FunctionArgument(NUM_SUBGROUPS_ARG),
                Span::default(),
            ),
        }
    }
}
