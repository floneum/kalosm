use super::*;
use crate::ir::Builtin;
use crate::lower::quantized::dot_lowering::{
    DotLoweringCtx, Q4KGgmlDotLowering, Q6KGgmlDotLowering, Q8_0Dot8Lowering, QuantizedDotLowering,
    VecDotLowering,
};

pub(in crate::lower) fn builtin_to_index_expr(builtin: Builtin) -> TileIndexExpr {
    match builtin {
        Builtin::Lane => TileIndexExpr::Lane,
        Builtin::LoopIndex => TileIndexExpr::LoopIndex,
        Builtin::ProgramId(axis) => TileIndexExpr::ProgramId(axis),
        Builtin::SubgroupId => TileIndexExpr::SubgroupId,
        Builtin::SubgroupLane => TileIndexExpr::SubgroupLane,
        Builtin::SubgroupSize => TileIndexExpr::SubgroupSize,
        Builtin::NumSubgroups => TileIndexExpr::NumSubgroups,
    }
}

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn lower_tile_expr_lane(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        expr: &TileExpr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match expr {
            TileExpr::Load(load) => {
                self.lower_tile_load_expr(expressions, scratch, body, load, spill_depth)
            }
            TileExpr::LoadLinear(load) => {
                self.lower_tile_linear_load_expr(expressions, scratch, body, load, spill_depth)
            }
            TileExpr::LoadVec4(load) => {
                self.lower_tile_vec4_load_expr(expressions, scratch, body, load, spill_depth)
            }
            TileExpr::LoadWorkgroup { src, index } => {
                let index =
                    self.lower_tile_index_expr(expressions, scratch, body, index, spill_depth)?;
                let (ptr, emits) = self.tile_dynamic_pointer(expressions, *src, index)?;
                Self::push_emits(body, emits);
                Ok(Self::emit_load(expressions, body, ptr))
            }
            TileExpr::LoadLocal(local) => {
                let local = self.private_local(*local)?;
                let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
                Ok(Self::emit_load(expressions, body, pointer))
            }
            TileExpr::QuantizedLoad(load) => {
                self.lower_tile_quantized_load_expr(expressions, scratch, body, load, spill_depth)
            }
            TileExpr::Full(value) => Ok(expressions.append(
                Expression::Literal(Literal::F32(value.get())),
                Span::default(),
            )),
            TileExpr::Literal(value) => {
                Ok(expressions.append(Self::tile_literal(*value), Span::default()))
            }
            TileExpr::Index(index) => {
                self.lower_tile_index_expr(expressions, scratch, body, index, spill_depth)
            }
            TileExpr::Builtin(builtin) => {
                let index = builtin_to_index_expr(*builtin);
                self.lower_tile_index_expr(expressions, scratch, body, &index, spill_depth)
            }
            TileExpr::Scalar(expr) => {
                self.lower_tile_scalar_expr(expressions, scratch, body, expr, spill_depth)
            }
            TileExpr::Unary { op, value } => {
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
            TileExpr::Binary { op, left, right } => {
                let left =
                    self.lower_tile_expr_lane(expressions, scratch, body, left, spill_depth + 1)?;
                let right =
                    self.lower_tile_expr_lane(expressions, scratch, body, right, spill_depth + 1)?;
                let expr = Self::tile_binary_expression(*op, left, right);
                Ok(self.emit_tile_expr(expressions, body, expr))
            }
            TileExpr::Sum { values } => {
                let mut iter = values.iter();
                let Some(first) = iter.next() else {
                    return Ok(
                        expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default())
                    );
                };
                let mut sum =
                    self.lower_tile_expr_lane(expressions, scratch, body, first, spill_depth + 1)?;
                for value in iter {
                    let rhs = self.lower_tile_expr_lane(
                        expressions,
                        scratch,
                        body,
                        value,
                        spill_depth + 1,
                    )?;
                    sum = self.emit_tile_expr(
                        expressions,
                        body,
                        Expression::Binary {
                            op: BinaryOperator::Add,
                            left: sum,
                            right: rhs,
                        },
                    );
                }
                Ok(sum)
            }
            TileExpr::Cast { value, to } => {
                let source = self.tile_expr_element(value)?;
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                Ok(self.cast_tile_value(expressions, body, value, source, *to))
            }
            TileExpr::Bitcast { value, to } => {
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
            TileExpr::Select {
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
            TileExpr::Compare {
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
            TileExpr::LoopFold {
                op,
                iterations,
                value,
                initial,
            } => self.lower_tile_loop_fold_value(
                expressions,
                scratch,
                body,
                value,
                *iterations,
                *op,
                *initial,
                spill_depth,
            ),
            TileExpr::GroupReduce {
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
            TileExpr::SubgroupReduce { op, value } => {
                let element = self.tile_expr_element(value)?;
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                self.lower_tile_subgroup_reduce_value(expressions, body, value, *op, element)
            }
            TileExpr::QuantizedBlockLane {
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
            TileExpr::Compose4 { values } => {
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
            TileExpr::Vec4Dot { left, right } => {
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
            TileExpr::Vec4Splat { value } => {
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                Ok(self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Compose {
                        ty: self.f32_vec4_ty,
                        components: vec![value, value, value, value],
                    },
                ))
            }
            TileExpr::QuantizedQ8_0Dot8 {
                a,
                src,
                k_base,
                col,
                mask,
                fill,
            } => Q8_0Dot8Lowering { a, k_base }.lower(
                self,
                DotLoweringCtx {
                    expressions,
                    scratch,
                    body,
                    spill_depth,
                    src,
                    col,
                    mask,
                    fill: *fill,
                },
            ),
            TileExpr::QuantizedVecDot {
                kind,
                a,
                src,
                k_base,
                col,
                mask,
                fill,
                block_n,
            } => VecDotLowering {
                kind: *kind,
                a,
                k_base,
                block_n: *block_n,
            }
            .lower(
                self,
                DotLoweringCtx {
                    expressions,
                    scratch,
                    body,
                    spill_depth,
                    src,
                    col,
                    mask,
                    fill: *fill,
                },
            ),
            TileExpr::QuantizedQ4KGgmlDot {
                a_low,
                a_high,
                sums,
                src,
                block,
                iq,
                ir,
                col,
                mask,
                fill,
            } => Q4KGgmlDotLowering {
                a_low,
                a_high,
                sums,
                block,
                iq,
                ir,
            }
            .lower(
                self,
                DotLoweringCtx {
                    expressions,
                    scratch,
                    body,
                    spill_depth,
                    src,
                    col,
                    mask,
                    fill: *fill,
                },
            ),
            TileExpr::QuantizedQ6KGgmlDot {
                a,
                src,
                block,
                ip,
                il,
                col,
                mask,
                fill,
            } => Q6KGgmlDotLowering { a, block, ip, il }.lower(
                self,
                DotLoweringCtx {
                    expressions,
                    scratch,
                    body,
                    spill_depth,
                    src,
                    col,
                    mask,
                    fill: *fill,
                },
            ),
        }
    }

}
