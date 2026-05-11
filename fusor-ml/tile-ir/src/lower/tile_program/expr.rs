use super::*;
use crate::ir::Builtin;

impl<'a> Lowerer<'a> {
    /// Top-level entry point for lowering an `Expr` tree. External callers
    /// (statement lowering, fragment loads, fold init, etc.) all enter at
    /// `spill_depth = 0`; the recursive arms inside `lower_tile_expr_lane`
    /// pass through their own `spill_depth` (sometimes incremented to limit
    /// register pressure on nested binary ops).
    pub(in crate::lower) fn lower_tile_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        expr: &Expr,
    ) -> Result<Handle<Expression>, LowerError> {
        self.lower_tile_expr_lane(expressions, scratch, body, expr, 0)
    }

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
                let ptr = self.tile_dynamic_pointer(expressions, *src, index, body)?;
                Ok(Self::emit_load(expressions, body, ptr))
            }
            Expr::LoadLocal(local) => {
                let local = self.private_local(*local)?;
                Ok(self.load_local(expressions, body, local))
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
                group_size,
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
                    *group_size,
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
                Ok(self.emit(expressions, body, expr))
            }
            Expr::Binary { op, left, right } => {
                let left =
                    self.lower_tile_expr_lane(expressions, scratch, body, left, spill_depth + 1)?;
                let right =
                    self.lower_tile_expr_lane(expressions, scratch, body, right, spill_depth + 1)?;
                let expr = Self::tile_binary_expression(*op, left, right);
                Ok(self.emit(expressions, body, expr))
            }
            Expr::Cast { value, to } => {
                let source = value.element();
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                Ok(self.cast_tile_value(expressions, body, value, source, *to))
            }
            Expr::Bitcast { value, to } => {
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                let scalar = Self::element_scalar(*to);
                Ok(self.cast_as(expressions, body, value, scalar.kind, None))
            }
            Expr::Select {
                condition,
                accept,
                reject,
            } => {
                let condition_ty = condition.element();
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
                Ok(self.emit(
                    expressions,
                    body,
                    Expression::Select {
                        condition,
                        accept,
                        reject,
                    },
                ))
            }
            Expr::Compare { op, left, right } => {
                let left =
                    self.lower_tile_expr_lane(expressions, scratch, body, left, spill_depth + 1)?;
                let right =
                    self.lower_tile_expr_lane(expressions, scratch, body, right, spill_depth + 1)?;
                Ok(self.emit(
                    expressions,
                    body,
                    Expression::Binary {
                        op: Self::tile_compare_binary(*op),
                        left,
                        right,
                    },
                ))
            }
            Expr::SubgroupReduce { op, value } => {
                let element = value.element();
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
                fill,
                *block_n,
                *lane,
                spill_depth,
            ),
            Expr::Compose4 { values } => {
                let handles = values
                    .iter()
                    .map(|value| {
                        self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth + 1)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let handles: [_; 4] = handles.try_into().expect("Compose4 carries exactly 4");
                Ok(self.compose_f32_vec4(expressions, body, handles))
            }
            Expr::Vec4Dot { left, right } => {
                let left =
                    self.lower_tile_expr_lane(expressions, scratch, body, left, spill_depth + 1)?;
                let right =
                    self.lower_tile_expr_lane(expressions, scratch, body, right, spill_depth + 1)?;
                Ok(self.dot_f32_vec4(expressions, body, left, right))
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
                fill,
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
            Builtin::Lane => Self::function_arg(expressions, LOCAL_INVOCATION_INDEX_ARG),
            Builtin::SubgroupId => Self::function_arg(expressions, SUBGROUP_ID_ARG),
            Builtin::SubgroupLane => Self::function_arg(expressions, SUBGROUP_INVOCATION_ID_ARG),
            Builtin::SubgroupSize => Self::function_arg(expressions, SUBGROUP_SIZE_ARG),
            Builtin::NumSubgroups => Self::function_arg(expressions, NUM_SUBGROUPS_ARG),
            Builtin::LoopIndex => self.load_local(expressions, body, self.current_loop_index()),
            Builtin::ProgramId(axis) => {
                let wg = Self::function_arg(expressions, WORKGROUP_ID_ARG);
                self.emit(
                    expressions,
                    body,
                    Expression::AccessIndex {
                        base: wg,
                        index: axis.index(),
                    },
                )
            }
        }
    }

    pub(in crate::lower) fn function_arg(
        expressions: &mut Arena<Expression>,
        arg: u32,
    ) -> Handle<Expression> {
        expressions.append(Expression::FunctionArgument(arg), Span::default())
    }
}
