use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_tile_program(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &TileProgramOp,
    ) -> Result<Statement, LowerError> {
        if op.block == 0 || op.block != self.workgroup_invocations {
            return Err(LowerError::UnsupportedOperation(
                "tile program block must match workgroup size",
            ));
        }

        let mut body = Block::new();
        for stmt in &op.body {
            match stmt {
                TileStmt::Store(store) => {
                    self.lower_tile_store_stmt(expressions, scratch, &mut body, store)?;
                }
                _ => self.lower_tile_stmt(expressions, scratch, &mut body, stmt)?,
            }
        }
        Ok(Statement::Block(body))
    }

    fn lower_tile_store_stmt(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        store: &TileStoreStmt,
    ) -> Result<(), LowerError> {
        self.block_dequant_cache.borrow_mut().clear();
        self.pin_cache.borrow_mut().clear();
        // loop_fold_group_cache is intentionally NOT cleared here — group
        // outputs survive across stores because the K loop is emitted into
        // the outer body block, dominated by every store that follows.
        let value = self.lower_tile_expr_lane(expressions, scratch, body, &store.value, 0)?;
        let mask = self.lower_tile_mask_expr(expressions, scratch, body, &store.mask, 0)?;
        let mut accept = Block::new();
        let row = self.lower_tile_index_expr(expressions, scratch, &mut accept, &store.row, 0)?;
        let col = self.lower_tile_index_expr(expressions, scratch, &mut accept, &store.col, 0)?;
        let (dst_index, dst_index_emits) =
            self.storage_index_from_coords(expressions, &store.dst, &[row, col])?;
        let (dst_ptr, dst_ptr_emits) =
            self.storage_dynamic_pointer(expressions, &store.dst, dst_index)?;
        Self::push_emits(&mut accept, dst_index_emits);
        Self::push_emits(&mut accept, dst_ptr_emits);
        accept.push(
            Statement::Store {
                pointer: dst_ptr,
                value,
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: mask,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        Ok(())
    }

    fn lower_tile_expr_lane(
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
                let left_ty = self.tile_expr_element(left)?;
                let left =
                    self.lower_tile_expr_lane(expressions, scratch, body, left, spill_depth + 1)?;
                let spill = self.tile_expr_spill_local(scratch, left_ty, spill_depth)?;
                let spill_ptr =
                    expressions.append(Expression::LocalVariable(spill), Span::default());
                body.push(
                    Statement::Store {
                        pointer: spill_ptr,
                        value: left,
                    },
                    Span::default(),
                );
                let right =
                    self.lower_tile_expr_lane(expressions, scratch, body, right, spill_depth + 1)?;
                let left =
                    expressions.append(Expression::Load { pointer: spill_ptr }, Span::default());
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, left)),
                    Span::default(),
                );
                let expr = Self::tile_binary_expression(*op, left, right);
                Ok(self.emit_tile_expr(expressions, body, expr))
            }
            TileExpr::Cast { value, to } => {
                let source = self.tile_expr_element(value)?;
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                Ok(self.cast_tile_value(expressions, body, value, source, *to))
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
                let condition = self.numeric_not_zero(expressions, body, condition, condition_ty);
                let accept_ty = self.tile_expr_element(accept)?;
                let accept =
                    self.lower_tile_expr_lane(expressions, scratch, body, accept, spill_depth + 1)?;
                let spill = self.tile_expr_spill_local(scratch, accept_ty, spill_depth)?;
                let spill_ptr =
                    expressions.append(Expression::LocalVariable(spill), Span::default());
                body.push(
                    Statement::Store {
                        pointer: spill_ptr,
                        value: accept,
                    },
                    Span::default(),
                );
                let reject =
                    self.lower_tile_expr_lane(expressions, scratch, body, reject, spill_depth + 1)?;
                let accept =
                    expressions.append(Expression::Load { pointer: spill_ptr }, Span::default());
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, accept)),
                    Span::default(),
                );
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
                let left_ty = self.tile_expr_element(left)?;
                let left =
                    self.lower_tile_expr_lane(expressions, scratch, body, left, spill_depth + 1)?;
                let spill = self.tile_expr_spill_local(scratch, left_ty, spill_depth)?;
                let spill_ptr =
                    expressions.append(Expression::LocalVariable(spill), Span::default());
                body.push(
                    Statement::Store {
                        pointer: spill_ptr,
                        value: left,
                    },
                    Span::default(),
                );
                let right =
                    self.lower_tile_expr_lane(expressions, scratch, body, right, spill_depth + 1)?;
                let left =
                    expressions.append(Expression::Load { pointer: spill_ptr }, Span::default());
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, left)),
                    Span::default(),
                );
                let condition = self.emit_tile_expr(
                    expressions,
                    body,
                    Expression::Binary {
                        op: Self::tile_compare_binary(*op),
                        left,
                        right,
                    },
                );
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
                self.lower_tile_group_reduce_value(
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
            TileExpr::PinnedRef { id } => {
                if let Some(handle) = self.pin_cache.borrow().get(id).copied() {
                    return Ok(handle);
                }
                let value_expr = self
                    .ir
                    .pinned_values
                    .get(id.index())
                    .ok_or(LowerError::UnsupportedOperation("unknown pin id"))?
                    .clone();
                // Lower the bound value once into the current block; cache the
                // SSA handle. naga's dominator-based SSA validates re-use from
                // any nested block so a single handle suffices.
                let value = self.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    body,
                    &value_expr,
                    spill_depth,
                )?;
                self.pin_cache.borrow_mut().insert(*id, value);
                Ok(value)
            }
            TileExpr::LoopFoldGroupOutput { group, lane } => self
                .lower_tile_loop_fold_group_output(
                    expressions,
                    scratch,
                    body,
                    *group,
                    *lane,
                    spill_depth,
                ),
            TileExpr::Dot4 { a, b } => {
                let mut a_handles = Vec::with_capacity(4);
                let mut b_handles = Vec::with_capacity(4);
                for i in 0..4 {
                    a_handles.push(self.lower_tile_expr_lane(
                        expressions,
                        scratch,
                        body,
                        &a[i],
                        spill_depth + 1,
                    )?);
                }
                for i in 0..4 {
                    b_handles.push(self.lower_tile_expr_lane(
                        expressions,
                        scratch,
                        body,
                        &b[i],
                        spill_depth + 1,
                    )?);
                }
                let a_vec = expressions.append(
                    Expression::Compose {
                        ty: self.f32_vec4_ty,
                        components: a_handles,
                    },
                    Span::default(),
                );
                let b_vec = expressions.append(
                    Expression::Compose {
                        ty: self.f32_vec4_ty,
                        components: b_handles,
                    },
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, a_vec)),
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, b_vec)),
                    Span::default(),
                );
                let dot = expressions.append(
                    Expression::Math {
                        fun: MathFunction::Dot,
                        arg: a_vec,
                        arg1: Some(b_vec),
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
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_tile_quantized_block_lane(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        id: BlockDequantId,
        src: &QuantizedMatrix,
        k_base: &TileIndexExpr,
        col: &TileIndexExpr,
        mask: &TileMaskExpr,
        fill: F32Bits,
        block_n: u32,
        lane: u32,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        if lane >= block_n {
            return Err(LowerError::UnsupportedOperation(
                "quantized block lane out of range",
            ));
        }
        if let Some(values) = self.block_dequant_cache.borrow().get(&id).cloned() {
            return Ok(values[lane as usize]);
        }

        // First lane request: emit the shared dequant helper into a masked
        // block. Cache the resulting per-lane handles for siblings.
        let tmp_locals: Vec<_> = (0..block_n)
            .map(|i| {
                self.block_dequant_value_local(scratch, i)
                    .ok_or(LowerError::UnsupportedOperation(
                        "quantized block lane exceeds available scratch locals",
                    ))
            })
            .collect::<Result<_, _>>()?;
        let fill_value = expressions.append(
            Expression::Literal(Literal::F32(fill.get())),
            Span::default(),
        );
        for local in &tmp_locals {
            let ptr = expressions.append(Expression::LocalVariable(*local), Span::default());
            body.push(
                Statement::Store {
                    pointer: ptr,
                    value: fill_value,
                },
                Span::default(),
            );
        }

        let mask_handle =
            self.lower_tile_mask_expr(expressions, scratch, body, mask, spill_depth)?;
        let mut accept = Block::new();
        let k_base_handle =
            self.lower_tile_index_expr(expressions, scratch, &mut accept, k_base, spill_depth)?;
        let col_handle =
            self.lower_tile_index_expr(expressions, scratch, &mut accept, col, spill_depth)?;
        let (values, value_emits) = match (src.format, block_n) {
            (GgmlQuantFormat::Q8_0, 8) => {
                self.dequantize_q8_0_values8(expressions, src, k_base_handle, col_handle)?
            }
            (GgmlQuantFormat::Q4K, 8) => {
                self.dequantize_q4k_values8(expressions, src, k_base_handle, col_handle)?
            }
            (GgmlQuantFormat::Q6K, 8) => {
                self.dequantize_q6k_values8(expressions, src, k_base_handle, col_handle)?
            }
            (GgmlQuantFormat::Q5_0, 16) => {
                self.dequantize_q5_0_values16(expressions, src, k_base_handle, col_handle)?
            }
            _ => {
                return Err(LowerError::UnsupportedOperation(
                    "quantized block dequant only supports Q8_0/Q4K/Q6K x8 and Q5_0 x16",
                ));
            }
        };
        Self::push_emits(&mut accept, value_emits);
        for (local, value) in tmp_locals.iter().zip(values.iter()) {
            let ptr = expressions.append(Expression::LocalVariable(*local), Span::default());
            accept.push(
                Statement::Store {
                    pointer: ptr,
                    value: *value,
                },
                Span::default(),
            );
        }
        body.push(
            Statement::If {
                condition: mask_handle,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        // Materialize the locals into SSA loads we hand back per lane.
        let mut handles = Vec::with_capacity(block_n as usize);
        for local in &tmp_locals {
            let ptr = expressions.append(Expression::LocalVariable(*local), Span::default());
            let value = expressions.append(Expression::Load { pointer: ptr }, Span::default());
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            handles.push(value);
        }
        self.block_dequant_cache
            .borrow_mut()
            .insert(id, handles.clone());
        Ok(handles[lane as usize])
    }

    fn block_dequant_value_local(
        &self,
        scratch: ScratchLocals,
        index: u32,
    ) -> Option<Handle<LocalVariable>> {
        scratch.block_dequant.get(index as usize).copied()
    }

    fn tile_expr_spill_local(
        &self,
        scratch: ScratchLocals,
        element: ElementType,
        depth: usize,
    ) -> Result<Handle<LocalVariable>, LowerError> {
        scratch
            .spills
            .get(Self::element_scratch_index(element))
            .and_then(|spills| spills.get(depth))
            .copied()
            .ok_or(LowerError::UnsupportedOperation(
                "tile expression nesting is too deep",
            ))
    }

    fn tile_value_local(
        scratch: ScratchLocals,
        element: ElementType,
    ) -> Result<Handle<LocalVariable>, LowerError> {
        scratch
            .values
            .get(Self::element_scratch_index(element))
            .copied()
            .ok_or(LowerError::UnsupportedOperation(
                "unsupported tile value type",
            ))
    }

    fn lower_tile_load_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        load: &TileLoadExpr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let element = load.src.buffer.element;
        let tmp = Self::tile_value_local(scratch, element)?;
        let tmp_ptr = expressions.append(Expression::LocalVariable(tmp), Span::default());
        let fill_source = load.fill.element();
        let fill = expressions.append(Self::tile_literal(load.fill), Span::default());
        let fill = self.cast_tile_value(expressions, body, fill, fill_source, element);
        body.push(
            Statement::Store {
                pointer: tmp_ptr,
                value: fill,
            },
            Span::default(),
        );

        let mask =
            self.lower_tile_mask_expr(expressions, scratch, body, &load.mask, spill_depth)?;
        let mut accept = Block::new();
        let row =
            self.lower_tile_index_expr(expressions, scratch, &mut accept, &load.row, spill_depth)?;
        let col =
            self.lower_tile_index_expr(expressions, scratch, &mut accept, &load.col, spill_depth)?;
        let (src_index, src_index_emits) =
            self.storage_index_from_coords(expressions, &load.src, &[row, col])?;
        let (src_ptr, src_ptr_emits) =
            self.storage_dynamic_pointer(expressions, &load.src, src_index)?;
        Self::push_emits(&mut accept, src_index_emits);
        Self::push_emits(&mut accept, src_ptr_emits);
        let value = expressions.append(Expression::Load { pointer: src_ptr }, Span::default());
        accept.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        accept.push(
            Statement::Store {
                pointer: tmp_ptr,
                value,
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: mask,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        let loaded = expressions.append(Expression::Load { pointer: tmp_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, loaded)),
            Span::default(),
        );
        Ok(loaded)
    }

    fn lower_tile_quantized_load_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        load: &TileQuantizedLoadExpr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let tmp = Self::tile_value_local(scratch, ElementType::F32)?;
        let tmp_ptr = expressions.append(Expression::LocalVariable(tmp), Span::default());
        let fill = expressions.append(
            Expression::Literal(Literal::F32(load.fill.get())),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: tmp_ptr,
                value: fill,
            },
            Span::default(),
        );

        let mask =
            self.lower_tile_mask_expr(expressions, scratch, body, &load.mask, spill_depth)?;
        let mut accept = Block::new();
        let row =
            self.lower_tile_index_expr(expressions, scratch, &mut accept, &load.row, spill_depth)?;
        let col =
            self.lower_tile_index_expr(expressions, scratch, &mut accept, &load.col, spill_depth)?;
        let (value, value_emits) = self.dequantize_qvalue(expressions, &load.src, row, col)?;
        Self::push_emits(&mut accept, value_emits);
        accept.push(
            Statement::Store {
                pointer: tmp_ptr,
                value,
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: mask,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        let loaded = expressions.append(Expression::Load { pointer: tmp_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, loaded)),
            Span::default(),
        );
        Ok(loaded)
    }

    fn lower_tile_scalar_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        expr: &TileScalarExpr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match expr {
            TileScalarExpr::Literal(value) => {
                Ok(expressions.append(Self::tile_literal(*value), Span::default()))
            }
            TileScalarExpr::Reduce {
                op,
                value,
                scratch: scratch_tile,
            } => {
                let value =
                    self.lower_tile_expr_lane(expressions, scratch, body, value, spill_depth)?;
                self.lower_tile_reduce_value(expressions, body, value, *scratch_tile, *op)
            }
            TileScalarExpr::LoopReduce {
                op,
                iterations,
                value,
                scratch: scratch_tile,
            } => {
                let value = self.lower_tile_loop_reduce_value(
                    expressions,
                    scratch,
                    body,
                    value,
                    *iterations,
                    *op,
                    spill_depth,
                )?;
                self.lower_tile_reduce_value(expressions, body, value, *scratch_tile, *op)
            }
        }
    }

    fn lower_tile_loop_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        value: &TileExpr,
        iterations: u32,
        op: TileReduceOp,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let element = self.tile_expr_element(value)?;
        let acc = self.tile_expr_spill_local(scratch, element, 0)?;
        let acc_ptr = expressions.append(Expression::LocalVariable(acc), Span::default());
        let identity = expressions.append(Self::tile_reduce_identity(op, element), Span::default());
        body.push(
            Statement::Store {
                pointer: acc_ptr,
                value: identity,
            },
            Span::default(),
        );

        let loop_ptr = expressions.append(
            Expression::LocalVariable(scratch.loop_index),
            Span::default(),
        );
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        body.push(
            Statement::Store {
                pointer: loop_ptr,
                value: zero,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let loop_index =
            expressions.append(Expression::Load { pointer: loop_ptr }, Span::default());
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, loop_index)),
            Span::default(),
        );
        let done = self.bin_lit_u32(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            loop_index,
            iterations,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        // Cache entries reference handles emitted into the outer block; they
        // become out of scope inside this loop body. Snapshot, clear for the
        // body, and restore on exit so callers see consistent state.
        let saved_dequant: Vec<_> = self.block_dequant_cache.borrow_mut().drain().collect();
        let saved_pin: Vec<_> = self.pin_cache.borrow_mut().drain().collect();
        let value = self.lower_tile_expr_lane(
            expressions,
            scratch,
            &mut loop_body,
            value,
            spill_depth + 1,
        )?;
        {
            let mut cache = self.block_dequant_cache.borrow_mut();
            cache.clear();
            for (k, v) in saved_dequant {
                cache.insert(k, v);
            }
        }
        {
            let mut cache = self.pin_cache.borrow_mut();
            cache.clear();
            for (k, v) in saved_pin {
                cache.insert(k, v);
            }
        }
        let acc = expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, acc)),
            Span::default(),
        );
        let reduced = self.emit_tile_expr(
            expressions,
            &mut loop_body,
            Self::tile_reduce_expression(op, acc, value),
        );
        loop_body.push(
            Statement::Store {
                pointer: acc_ptr,
                value: reduced,
            },
            Span::default(),
        );
        loop_body.push(
            self.increment_u32_local(expressions, scratch.loop_index, 1),
            Span::default(),
        );
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        let value = expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        Ok(value)
    }

    fn lower_tile_loop_fold_value(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        value: &TileExpr,
        iterations: u32,
        op: TileReduceOp,
        initial: TileLiteral,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let element = initial.element();
        let acc = self.tile_expr_spill_local(scratch, element, spill_depth)?;
        let acc_ptr = expressions.append(Expression::LocalVariable(acc), Span::default());
        let initial = expressions.append(Self::tile_literal(initial), Span::default());
        body.push(
            Statement::Store {
                pointer: acc_ptr,
                value: initial,
            },
            Span::default(),
        );

        let loop_ptr = expressions.append(
            Expression::LocalVariable(scratch.loop_index),
            Span::default(),
        );
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        body.push(
            Statement::Store {
                pointer: loop_ptr,
                value: zero,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let loop_index =
            expressions.append(Expression::Load { pointer: loop_ptr }, Span::default());
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, loop_index)),
            Span::default(),
        );
        let done = self.bin_lit_u32(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            loop_index,
            iterations,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        // Cache entries reference handles emitted into the outer block; they
        // become out of scope inside this loop body. Snapshot, clear for the
        // body, and restore on exit so callers see consistent state.
        let saved_dequant: Vec<_> = self.block_dequant_cache.borrow_mut().drain().collect();
        let saved_pin: Vec<_> = self.pin_cache.borrow_mut().drain().collect();
        let value = self.lower_tile_expr_lane(
            expressions,
            scratch,
            &mut loop_body,
            value,
            spill_depth + 1,
        )?;
        {
            let mut cache = self.block_dequant_cache.borrow_mut();
            cache.clear();
            for (k, v) in saved_dequant {
                cache.insert(k, v);
            }
        }
        {
            let mut cache = self.pin_cache.borrow_mut();
            cache.clear();
            for (k, v) in saved_pin {
                cache.insert(k, v);
            }
        }
        let acc = expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, acc)),
            Span::default(),
        );
        let reduced = self.emit_tile_expr(
            expressions,
            &mut loop_body,
            Self::tile_reduce_expression(op, acc, value),
        );
        loop_body.push(
            Statement::Store {
                pointer: acc_ptr,
                value: reduced,
            },
            Span::default(),
        );
        loop_body.push(
            self.increment_u32_local(expressions, scratch.loop_index, 1),
            Span::default(),
        );
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        let value = expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        Ok(value)
    }

    fn lower_tile_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        scratch_tile: TileRef,
        op: TileReduceOp,
    ) -> Result<Handle<Expression>, LowerError> {
        let lane = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let (lane_ptr, lane_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, lane)?;
        Self::push_emits(body, lane_ptr_emits);
        body.push(
            Statement::Store {
                pointer: lane_ptr,
                value,
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let mut stride = self.workgroup_invocations / 2;
        while stride > 0 {
            let lane = expressions.append(
                Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
                Span::default(),
            );
            let limit =
                expressions.append(Expression::Literal(Literal::U32(stride)), Span::default());
            let participates = self.emit_tile_expr(
                expressions,
                body,
                Expression::Binary {
                    op: BinaryOperator::Less,
                    left: lane,
                    right: limit,
                },
            );
            let accept =
                self.lower_tile_reduce_step(expressions, scratch_tile, lane, stride, op)?;
            body.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            body.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        let (result_ptr, result_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, zero)?;
        Self::push_emits(body, result_ptr_emits);
        let result = expressions.append(
            Expression::Load {
                pointer: result_ptr,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, result)),
            Span::default(),
        );
        Ok(result)
    }

    fn lower_tile_reduce_step(
        &self,
        expressions: &mut Arena<Expression>,
        scratch_tile: TileRef,
        lane: Handle<Expression>,
        stride: u32,
        op: TileReduceOp,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        let rhs_index = self.add_literal_u32(expressions, lane, stride);
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, rhs_index)),
            Span::default(),
        );
        let (lhs_ptr, lhs_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, lane)?;
        let (rhs_ptr, rhs_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, rhs_index)?;
        Self::push_emits(&mut body, lhs_ptr_emits);
        Self::push_emits(&mut body, rhs_ptr_emits);
        let lhs = expressions.append(Expression::Load { pointer: lhs_ptr }, Span::default());
        let rhs = expressions.append(Expression::Load { pointer: rhs_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::range_from(expressions, lhs, rhs)),
            Span::default(),
        );
        let reduced = self.emit_tile_expr(
            expressions,
            &mut body,
            Self::tile_reduce_expression(op, lhs, rhs),
        );
        body.push(
            Statement::Store {
                pointer: lhs_ptr,
                value: reduced,
            },
            Span::default(),
        );
        Ok(body)
    }

    fn lower_tile_group_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        scratch_tile: TileRef,
        op: TileReduceOp,
        group_size: u32,
    ) -> Result<Handle<Expression>, LowerError> {
        if group_size == 0
            || !group_size.is_power_of_two()
            || group_size > self.workgroup_invocations
            || self.workgroup_invocations % group_size != 0
        {
            return Err(LowerError::UnsupportedOperation(
                "tile group reduce requires a power-of-two group size that divides the block",
            ));
        }

        let lane = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let (lane_ptr, lane_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, lane)?;
        Self::push_emits(body, lane_ptr_emits);
        body.push(
            Statement::Store {
                pointer: lane_ptr,
                value,
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let mut index_emits = Vec::new();
        let group_offset =
            self.mod_literal_u32_emitted(expressions, lane, group_size, &mut index_emits);
        let group_base = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Subtract,
                left: lane,
                right: group_offset,
            },
            Span::default(),
        );
        index_emits.push(Self::single_expression_range(expressions, group_base));
        Self::push_emits(body, index_emits);

        let mut stride = group_size / 2;
        while stride > 0 {
            let limit =
                expressions.append(Expression::Literal(Literal::U32(stride)), Span::default());
            let participates = self.emit_tile_expr(
                expressions,
                body,
                Expression::Binary {
                    op: BinaryOperator::Less,
                    left: group_offset,
                    right: limit,
                },
            );
            let accept =
                self.lower_tile_reduce_step(expressions, scratch_tile, lane, stride, op)?;
            body.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            body.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        let (result_ptr, result_ptr_emits) =
            self.tile_dynamic_pointer(expressions, scratch_tile, group_base)?;
        Self::push_emits(body, result_ptr_emits);
        let result = expressions.append(
            Expression::Load {
                pointer: result_ptr,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, result)),
            Span::default(),
        );
        Ok(result)
    }

    fn lower_tile_loop_fold_group_output(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        group: LoopFoldGroupId,
        lane: u32,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        if let Some(values) = self.loop_fold_group_cache.borrow().get(&group).cloned() {
            return Ok(values.get(lane as usize).copied().ok_or(
                LowerError::UnsupportedOperation("fold group lane out of range"),
            )?);
        }

        let g = self
            .ir
            .loop_fold_groups
            .get(group.index())
            .ok_or(LowerError::UnsupportedOperation("unknown fold group"))?
            .clone();
        let n = g.bodies.len();
        if g.initials.len() != n {
            return Err(LowerError::UnsupportedOperation(
                "fold group initial count mismatch",
            ));
        }
        let offset = self.fold_group_offsets.get(group.index()).copied().ok_or(
            LowerError::UnsupportedOperation("fold group offset missing"),
        )?;
        if offset + n > self.fold_accumulator_locals.len() {
            return Err(LowerError::UnsupportedOperation(
                "fold group accumulator pool exhausted",
            ));
        }
        let acc_locals: Vec<_> = (0..n)
            .map(|i| self.fold_accumulator_locals[offset + i])
            .collect();

        // Initialize accumulators.
        for (i, local) in acc_locals.iter().enumerate() {
            let init = expressions.append(Self::tile_literal(g.initials[i]), Span::default());
            let ptr = expressions.append(Expression::LocalVariable(*local), Span::default());
            body.push(
                Statement::Store {
                    pointer: ptr,
                    value: init,
                },
                Span::default(),
            );
        }

        // Initialize loop index.
        let loop_ptr = expressions.append(
            Expression::LocalVariable(scratch.loop_index),
            Span::default(),
        );
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        body.push(
            Statement::Store {
                pointer: loop_ptr,
                value: zero,
            },
            Span::default(),
        );

        // Build loop body.
        let mut loop_body = Block::new();
        let loop_index =
            expressions.append(Expression::Load { pointer: loop_ptr }, Span::default());
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, loop_index)),
            Span::default(),
        );
        let done = self.bin_lit_u32(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            loop_index,
            g.iterations,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        // Snapshot caches that reference handles in the outer block; the loop
        // body emits a fresh scope. Restore on exit so callers see consistent
        // state.
        let saved_dequant: Vec<_> = self.block_dequant_cache.borrow_mut().drain().collect();
        let saved_pin: Vec<_> = self.pin_cache.borrow_mut().drain().collect();

        // Lower each body[i] within the same loop body, accumulate into acc_locals[i].
        for (i, body_expr) in g.bodies.iter().enumerate() {
            let value = self.lower_tile_expr_lane(
                expressions,
                scratch,
                &mut loop_body,
                body_expr,
                spill_depth + 1,
            )?;
            let acc_ptr =
                expressions.append(Expression::LocalVariable(acc_locals[i]), Span::default());
            let acc_load =
                expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
            loop_body.push(
                Statement::Emit(Self::single_expression_range(expressions, acc_load)),
                Span::default(),
            );
            let reduced = self.emit_tile_expr(
                expressions,
                &mut loop_body,
                Self::tile_reduce_expression(g.op, acc_load, value),
            );
            loop_body.push(
                Statement::Store {
                    pointer: acc_ptr,
                    value: reduced,
                },
                Span::default(),
            );
        }

        // Restore caches now that we're leaving the loop body scope.
        {
            let mut cache = self.block_dequant_cache.borrow_mut();
            cache.clear();
            for (k, v) in saved_dequant {
                cache.insert(k, v);
            }
        }
        {
            let mut cache = self.pin_cache.borrow_mut();
            cache.clear();
            for (k, v) in saved_pin {
                cache.insert(k, v);
            }
        }

        loop_body.push(
            self.increment_u32_local(expressions, scratch.loop_index, 1),
            Span::default(),
        );
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        // Materialize per-accumulator final loads in the outer block; cache them.
        let mut handles = Vec::with_capacity(n);
        for local in &acc_locals {
            let ptr = expressions.append(Expression::LocalVariable(*local), Span::default());
            let value = expressions.append(Expression::Load { pointer: ptr }, Span::default());
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            handles.push(value);
        }
        self.loop_fold_group_cache
            .borrow_mut()
            .insert(group, handles.clone());
        Ok(handles[lane as usize])
    }

    fn lower_tile_subgroup_reduce_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        op: TileReduceOp,
        element: ElementType,
    ) -> Result<Handle<Expression>, LowerError> {
        let subgroup_op = match op {
            TileReduceOp::Sum => SubgroupOperation::Add,
            TileReduceOp::Product => SubgroupOperation::Mul,
            TileReduceOp::Max => SubgroupOperation::Max,
            TileReduceOp::Min => SubgroupOperation::Min,
        };
        let result_ty = match element {
            ElementType::F32 => self.f32_ty,
            ElementType::F16 => self.f16_ty.ok_or(LowerError::UnsupportedOperation(
                "subgroup reduce on f16 requires f16 capability",
            ))?,
            ElementType::U32 => self.u32_ty,
        };
        let result = expressions.append(
            Expression::SubgroupOperationResult { ty: result_ty },
            Span::default(),
        );
        body.push(
            Statement::SubgroupCollectiveOperation {
                op: subgroup_op,
                collective_op: CollectiveOperation::Reduce,
                argument: value,
                result,
            },
            Span::default(),
        );
        Ok(result)
    }

    fn tile_reduce_identity(op: TileReduceOp, element: ElementType) -> Expression {
        match op {
            TileReduceOp::Sum => Self::zero_literal(element),
            TileReduceOp::Product => Self::one_literal(element),
            TileReduceOp::Max => match element {
                ElementType::F32 => Expression::Literal(Literal::F32(f32::MIN)),
                ElementType::F16 => {
                    Expression::Literal(Literal::F16(half::f16::from_f32(-65504.0)))
                }
                ElementType::U32 => Expression::Literal(Literal::U32(0)),
            },
            TileReduceOp::Min => match element {
                ElementType::F32 => Expression::Literal(Literal::F32(f32::MAX)),
                ElementType::F16 => Expression::Literal(Literal::F16(half::f16::from_f32(65504.0))),
                ElementType::U32 => Expression::Literal(Literal::U32(u32::MAX)),
            },
        }
    }

    fn tile_reduce_expression(
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

    pub(super) fn lower_tile_index_expr(
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

    fn lower_tile_mask_expr(
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

    fn tile_expr_element(&self, expr: &TileExpr) -> Result<ElementType, LowerError> {
        match expr {
            TileExpr::Load(load) => Ok(load.src.buffer.element),
            TileExpr::QuantizedLoad(_) | TileExpr::Full(_) => Ok(ElementType::F32),
            TileExpr::Literal(value) => Ok(value.element()),
            TileExpr::Index(_) => Ok(ElementType::U32),
            TileExpr::Scalar(expr) => self.tile_scalar_expr_element(expr),
            TileExpr::Unary { value, .. } | TileExpr::Binary { left: value, .. } => {
                self.tile_expr_element(value)
            }
            TileExpr::Cast { to, .. } => Ok(*to),
            TileExpr::Select { accept, .. } => self.tile_expr_element(accept),
            TileExpr::Compare { output, .. } => Ok(*output),
            TileExpr::LoopFold { initial, .. } => Ok(initial.element()),
            TileExpr::GroupReduce { scratch, .. } => Ok(scratch.element),
            TileExpr::SubgroupReduce { value, .. } => self.tile_expr_element(value),
            TileExpr::QuantizedBlockLane { .. } => Ok(ElementType::F32),
            TileExpr::Dot4 { .. } => Ok(ElementType::F32),
            TileExpr::PinnedRef { .. } => Ok(ElementType::F32),
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

    fn tile_scalar_expr_element(&self, expr: &TileScalarExpr) -> Result<ElementType, LowerError> {
        match expr {
            TileScalarExpr::Reduce { scratch, .. } | TileScalarExpr::LoopReduce { scratch, .. } => {
                Ok(scratch.element)
            }
            TileScalarExpr::Literal(value) => Ok(value.element()),
        }
    }

    fn element_scratch_index(element: ElementType) -> usize {
        match element {
            ElementType::F32 => 0,
            ElementType::F16 => 1,
            ElementType::U32 => 2,
        }
    }

    fn tile_literal(value: TileLiteral) -> Expression {
        match value {
            TileLiteral::F32(value) => Expression::Literal(Literal::F32(value.get())),
            TileLiteral::F16(value) => {
                Expression::Literal(Literal::F16(half::f16::from_bits(value)))
            }
            TileLiteral::U32(value) => Expression::Literal(Literal::U32(value)),
        }
    }

    fn zero_literal(element: ElementType) -> Expression {
        match element {
            ElementType::F32 => Expression::Literal(Literal::F32(0.0)),
            ElementType::F16 => Expression::Literal(Literal::F16(half::f16::from_f32(0.0))),
            ElementType::U32 => Expression::Literal(Literal::U32(0)),
        }
    }

    fn one_literal(element: ElementType) -> Expression {
        match element {
            ElementType::F32 => Expression::Literal(Literal::F32(1.0)),
            ElementType::F16 => Expression::Literal(Literal::F16(half::f16::from_f32(1.0))),
            ElementType::U32 => Expression::Literal(Literal::U32(1)),
        }
    }

    fn element_scalar(element: ElementType) -> Scalar {
        match element {
            ElementType::F32 => Scalar::F32,
            ElementType::F16 => Scalar {
                kind: ScalarKind::Float,
                width: 2,
            },
            ElementType::U32 => Scalar::U32,
        }
    }

    fn cast_tile_value(
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

    fn numeric_not_zero(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        element: ElementType,
    ) -> Handle<Expression> {
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

    fn tile_unary_math(op: TileUnaryOp) -> Option<MathFunction> {
        Some(match op {
            TileUnaryOp::Exp => MathFunction::Exp,
            TileUnaryOp::Exp2 => MathFunction::Exp2,
            TileUnaryOp::Log => MathFunction::Log,
            TileUnaryOp::Log2 => MathFunction::Log2,
            TileUnaryOp::Sqrt => MathFunction::Sqrt,
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

    fn tile_binary_expression(
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
        }
    }

    fn tile_compare_binary(op: TileCompareOp) -> BinaryOperator {
        match op {
            TileCompareOp::Lt => BinaryOperator::Less,
            TileCompareOp::Le => BinaryOperator::LessEqual,
            TileCompareOp::Gt => BinaryOperator::Greater,
            TileCompareOp::Ge => BinaryOperator::GreaterEqual,
            TileCompareOp::Eq => BinaryOperator::Equal,
        }
    }

    fn emit_tile_expr(
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
