use super::*;
use crate::ir::Builtin;

impl<'a> Lowerer<'a> {
    /// Top-level entry point for lowering an `Expr` tree. External callers all
    /// enter at `spill_depth = 0`; the recursive arms pass through their own
    /// `spill_depth` (sometimes incremented to limit register pressure on
    /// nested binary ops).
    pub(in crate::lower) fn lower_expr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expr: &Expr,
    ) -> Result<Handle<Expression>, LowerError> {
        self.lower_expr_lane(expressions, body, expr, 0)
    }

    pub(in crate::lower) fn lower_exprs_lane(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        exprs: &[Expr],
        spill_depth: usize,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        exprs
            .iter()
            .map(|expr| self.lower_expr_lane(expressions, body, expr, spill_depth))
            .collect()
    }

    pub(in crate::lower) fn lower_expr_lane(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expr: &Expr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match expr.kind() {
            ExprKind::Load {
                src,
                addr,
                mask,
                fill,
            } => self.lower_load_expr(expressions, body, src, addr, mask, fill, spill_depth),
            ExprKind::LoadTile { tile, index } => {
                let index = self.lower_expr_lane(expressions, body, index, spill_depth)?;
                let ptr = self.tile_dynamic_pointer(expressions, tile, index, body)?;
                Ok(Self::emit_load(expressions, body, ptr))
            }
            ExprKind::LoadLocal(local) => {
                // Coop accumulators chain through the acc-value SSA memo: a live
                // memo entry is reused without emitting a Load, so a sequence of
                // `StoreLocal(acc, CoopMma{c: LoadLocal(acc)})` becomes 1 Load +
                // N CooperativeMultiplyAdd + 1 Store per accumulator.
                if matches!(local.element, ElementType::CoopMatrix { .. }) {
                    if let Some(value) = self.coop_acc_value(local) {
                        return Ok(value);
                    }
                }
                let handle = self.private_local(local)?;
                Ok(self.load_local(expressions, body, handle))
            }
            ExprKind::Literal(value) => {
                Ok(expressions.append(Self::tile_literal(*value), Span::default()))
            }
            ExprKind::Builtin(builtin) => Ok(self.lower_builtin(expressions, body, *builtin)),
            ExprKind::Reduce { op, kind, value } => {
                self.lower_reduce(expressions, body, *op, kind, value, spill_depth)
            }
            ExprKind::Unary { op, value } => {
                let value = self.lower_expr_lane(expressions, body, value, spill_depth)?;
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
            ExprKind::Binary { op, left, right } => {
                let left = self.lower_expr_lane(expressions, body, left, spill_depth + 1)?;
                let right = self.lower_expr_lane(expressions, body, right, spill_depth + 1)?;
                let expr = Self::tile_binary_expression(*op, left, right);
                Ok(self.emit(expressions, body, expr))
            }
            ExprKind::Cast { value, to } => {
                // A cast to a cooperative-matrix type is the coop accumulator
                // zero-init (`coop_zero`): there is no scalar→fragment cast, it
                // lowers to `Expression::ZeroValue` (the pre-rewrite
                // `ZeroCoopAcc`). Matching master, this is appended as a baked
                // constant expression, not wrapped in an `Emit`.
                if matches!(to, ElementType::CoopMatrix { .. }) {
                    let ty = self.element_type(*to)?;
                    return Ok(expressions.append(Expression::ZeroValue(ty), Span::default()));
                }
                let source = value.element();
                let value = self.lower_expr_lane(expressions, body, value, spill_depth)?;
                Ok(self.cast_tile_value(expressions, body, value, source, *to))
            }
            ExprKind::Bitcast { value, to } => {
                let value = self.lower_expr_lane(expressions, body, value, spill_depth)?;
                let scalar = Self::element_scalar(*to);
                Ok(self.cast_as(expressions, body, value, scalar.kind, None))
            }
            ExprKind::Select {
                condition,
                accept,
                reject,
            } => {
                let condition_ty = condition.element();
                let condition =
                    self.lower_expr_lane(expressions, body, condition, spill_depth + 1)?;
                let condition = self.condition_value(expressions, body, condition, condition_ty);
                let accept = self.lower_expr_lane(expressions, body, accept, spill_depth + 1)?;
                let reject = self.lower_expr_lane(expressions, body, reject, spill_depth + 1)?;
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
            ExprKind::Compare { op, left, right } => {
                let left = self.lower_expr_lane(expressions, body, left, spill_depth + 1)?;
                let right = self.lower_expr_lane(expressions, body, right, spill_depth + 1)?;
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
            ExprKind::Vec {
                scalar,
                lanes,
                parts,
            } => {
                let handles = parts
                    .iter()
                    .map(|value| self.lower_expr_lane(expressions, body, value, spill_depth + 1))
                    .collect::<Result<Vec<_>, _>>()?;
                let ty = self.vector_type_handle(*scalar, *lanes)?;
                Ok(self.emit(
                    expressions,
                    body,
                    Expression::Compose {
                        ty,
                        components: handles,
                    },
                ))
            }
            ExprKind::Dot { left, right } => {
                let (scalar, lanes) = match left.element() {
                    ElementType::Vector { scalar, lanes } => (scalar, lanes),
                    _ => {
                        return Err(LowerError::UnsupportedOperation(
                            "vector dot requires a vector operand",
                        ));
                    }
                };
                if !matches!(scalar, ScalarElement::F32 | ScalarElement::F16) {
                    return Err(LowerError::UnsupportedOperation(
                        "vector dot requires a floating-point vector",
                    ));
                }
                self.vector_type_handle(scalar, lanes)?;
                let left = self.lower_expr_lane(expressions, body, left, spill_depth + 1)?;
                let right = self.lower_expr_lane(expressions, body, right, spill_depth + 1)?;
                Ok(self.math2(expressions, body, MathFunction::Dot, left, right))
            }
            ExprKind::CoopLoad {
                role,
                scalar,
                rows,
                cols,
                src,
            } => {
                // A loaded fragment is shared across every MMA in its row/column
                // (the same `Rc` is cloned into each `CoopMma`). Memoize the
                // emission on the node identity so the fragment is loaded once
                // per loop iteration, not once per MMA (revives the pre-rewrite
                // coop fragment cache — fragments dedup by node identity).
                if let Some(handle) = self.expr_memo.borrow().get(&expr.as_ptr()).copied() {
                    return Ok(handle);
                }
                let handle =
                    self.lower_coop_load(expressions, body, *role, *scalar, *rows, *cols, src)?;
                self.expr_memo.borrow_mut().insert(expr.as_ptr(), handle);
                Ok(handle)
            }
            ExprKind::CoopMma { a, b, c } => self.lower_coop_mma(expressions, body, a, b, c),
            ExprKind::Dequantize { .. } => {
                // A bare `Dequantize` projects lane 0; in practice it is always
                // wrapped in `Shared` and projected by `LaneOf`.
                let handles = self.lower_dequantize(expressions, body, expr, spill_depth)?;
                Ok(handles[0])
            }
            ExprKind::LaneOf { block, lane } => {
                let handles = self.lower_lane_of_block(expressions, body, block, spill_depth)?;
                handles
                    .get(*lane as usize)
                    .copied()
                    .ok_or(LowerError::UnsupportedOperation(
                        "quantized block lane out of range",
                    ))
            }
            ExprKind::QuantizedDot { .. } => {
                self.lower_quantized_dot(expressions, body, expr, spill_depth)
            }
            ExprKind::Shared(inner) => {
                self.lower_shared(expressions, body, expr, inner, spill_depth)
            }
        }
    }

    /// Resolve the per-lane handles of a `LaneOf`'s `block` operand. The block
    /// is typically a `Shared(Dequantize)`; emit-once memoization keys on the
    /// shared node's `Rc::as_ptr`.
    fn lower_lane_of_block(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        block: &Expr,
        spill_depth: usize,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        match block.kind() {
            ExprKind::Shared(inner) => {
                self.lower_shared_lanes(expressions, body, block, inner, spill_depth)
            }
            ExprKind::Dequantize { .. } => {
                self.lower_dequantize(expressions, body, block, spill_depth)
            }
            _ => Err(LowerError::UnsupportedOperation(
                "LaneOf expects a Dequantize block",
            )),
        }
    }

    /// Lower a `Shared(inner)` value node, emit-once memoized on the shared
    /// node's identity. Used for coop fragments and any structurally-shared
    /// subtree.
    fn lower_shared(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        shared: &Expr,
        inner: &Expr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        if let Some(handle) = self.expr_memo.borrow().get(&shared.as_ptr()).copied() {
            return Ok(handle);
        }
        let handle = self.lower_expr_lane(expressions, body, inner, spill_depth)?;
        self.expr_memo.borrow_mut().insert(shared.as_ptr(), handle);
        Ok(handle)
    }

    /// Lower a `Shared(Dequantize)` to its N lane handles, emit-once memoized on
    /// the shared node's identity (the `dequant_memo`). `LaneOf` returns
    /// `handles[lane]`.
    fn lower_shared_lanes(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        shared: &Expr,
        inner: &Expr,
        spill_depth: usize,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        if let Some(handles) = self.dequant_memo.borrow().get(&shared.as_ptr()).cloned() {
            return Ok(handles);
        }
        let handles = self.lower_dequantize(expressions, body, inner, spill_depth)?;
        self.dequant_memo
            .borrow_mut()
            .insert(shared.as_ptr(), handles.clone());
        Ok(handles)
    }

    pub(in crate::lower) fn lower_builtin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        builtin: Builtin,
    ) -> Handle<Expression> {
        match builtin {
            Builtin::Lane => Self::function_arg(expressions, LOCAL_INVOCATION_INDEX_ARG),
            Builtin::SubgroupId => Self::function_arg(
                expressions,
                self.subgroup_id_arg
                    .expect("subgroup_id argument should be declared"),
            ),
            Builtin::SubgroupLane => Self::function_arg(
                expressions,
                self.subgroup_invocation_id_arg
                    .expect("subgroup_invocation_id argument should be declared"),
            ),
            Builtin::SubgroupSize => Self::function_arg(
                expressions,
                self.subgroup_size_arg
                    .expect("subgroup_size argument should be declared"),
            ),
            Builtin::NumSubgroups => Self::function_arg(
                expressions,
                self.num_subgroups_arg
                    .expect("num_subgroups argument should be declared"),
            ),
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
