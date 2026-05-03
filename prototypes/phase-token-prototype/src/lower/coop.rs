use super::*;
use naga::{Barrier, CooperativeData, CooperativeRole, CooperativeSize};

const COOP_SIZE: CooperativeSize = CooperativeSize::Eight;

impl<'a> Lowerer<'a> {
    /// Lower non-store tile statements. WhileTrue emits a Naga `loop` with an
    /// explicit break guard; coop ops emit cooperative-matrix Loads/MMA/Store.
    pub(super) fn lower_tile_stmt_body(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        stmts: &[TileStmt],
    ) -> Result<(), LowerError> {
        for stmt in stmts {
            self.lower_tile_stmt(expressions, scratch, body, stmt)?;
        }
        Ok(())
    }

    pub(super) fn lower_tile_stmt(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        stmt: &TileStmt,
    ) -> Result<(), LowerError> {
        match stmt {
            TileStmt::Store(_) => Err(LowerError::UnsupportedOperation(
                "store statements must be lowered by lower_tile_program",
            )),
            TileStmt::Barrier => {
                body.push(
                    Statement::ControlBarrier(Barrier::WORK_GROUP),
                    Span::default(),
                );
                Ok(())
            }
            TileStmt::ZeroCoopAcc { id } => {
                let local = self
                    .coop_acc_locals
                    .get(id.index())
                    .copied()
                    .ok_or(LowerError::UnsupportedOperation("unknown coop acc id"))?;
                let ty = self.coop_c_ty.ok_or(LowerError::UnsupportedOperation(
                    "coop C type missing — tile program uses coop statements without cooperative-matrix support",
                ))?;
                let zero = expressions.append(Expression::ZeroValue(ty), Span::default());
                let ptr = expressions.append(Expression::LocalVariable(local), Span::default());
                body.push(
                    Statement::Store {
                        pointer: ptr,
                        value: zero,
                    },
                    Span::default(),
                );
                Ok(())
            }
            TileStmt::WhileTrue {
                max_iterations,
                body: inner,
            } => self.lower_tile_while_true(expressions, scratch, body, *max_iterations, inner),
            TileStmt::CopyToWorkgroupTile {
                dst,
                src,
                row_offset,
                col_offset,
            } => self.lower_copy_to_tile(
                expressions,
                scratch,
                body,
                *dst,
                src,
                row_offset,
                col_offset,
            ),
            TileStmt::CopyQuantToWorkgroupTile {
                dst,
                src,
                row_offset,
                col_offset,
            } => self.lower_copy_quant_to_tile(
                expressions,
                scratch,
                body,
                *dst,
                src,
                row_offset,
                col_offset,
            ),
            TileStmt::StoreCoopAcc { acc, dst, row, col } => {
                self.lower_store_coop_acc(expressions, scratch, body, *acc, dst, row, col)
            }
            TileStmt::LoadCoop {
                id,
                role,
                tile,
                row,
                col,
            } => self.lower_load_coop_fragment(
                expressions,
                scratch,
                body,
                *id,
                *tile,
                row,
                col,
                match role {
                    CoopOperandRole::A => CooperativeRole::A,
                    CoopOperandRole::B => CooperativeRole::B,
                },
            ),
            TileStmt::Mma { acc, a, b } => self.lower_coop_mma(expressions, body, *acc, *a, *b),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_load_coop_fragment(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        id: CoopFragmentId,
        tile: TileRef,
        row: &TileIndexExpr,
        col: &TileIndexExpr,
        role: CooperativeRole,
    ) -> Result<(), LowerError> {
        let layout = self.tile_layout(tile)?;
        let stride_u = Self::row_major_tile_stride(layout)?;
        let row_h = self.lower_tile_index_expr(expressions, scratch, body, row, 0)?;
        let col_h = self.lower_tile_index_expr(expressions, scratch, body, col, 0)?;
        let mut emits = Vec::new();
        let index = self.tile_matrix_index_inline(expressions, &mut emits, row_h, col_h, stride_u);
        let (ptr, ptr_emits) = self.tile_dynamic_pointer(expressions, tile, index)?;
        emits.extend(ptr_emits);
        Self::push_emits(body, emits);
        let stride =
            expressions.append(Expression::Literal(Literal::U32(stride_u)), Span::default());
        let frag = expressions.append(
            Expression::CooperativeLoad {
                columns: COOP_SIZE,
                rows: COOP_SIZE,
                role,
                data: CooperativeData {
                    pointer: ptr,
                    stride,
                    row_major: false,
                },
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, frag)),
            Span::default(),
        );
        self.coop_fragment_cache.borrow_mut().insert(id, frag);
        Ok(())
    }

    fn lower_coop_mma(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        acc_id: CoopAccId,
        a_id: CoopFragmentId,
        b_id: CoopFragmentId,
    ) -> Result<(), LowerError> {
        let acc_local = self
            .coop_acc_locals
            .get(acc_id.index())
            .copied()
            .ok_or(LowerError::UnsupportedOperation("unknown coop acc"))?;
        let a = self
            .coop_fragment_cache
            .borrow()
            .get(&a_id)
            .copied()
            .ok_or(LowerError::UnsupportedOperation(
                "coop_mma A fragment not loaded in current scope",
            ))?;
        let b = self
            .coop_fragment_cache
            .borrow()
            .get(&b_id)
            .copied()
            .ok_or(LowerError::UnsupportedOperation(
                "coop_mma B fragment not loaded in current scope",
            ))?;
        // Get the current SSA value of this accumulator: cache hit → reuse;
        // cache miss → emit one Load from the accumulator local. Subsequent
        // MMAs in this scope chain through SSA without touching the local.
        let c = match self.coop_acc_value_cache.borrow().get(&acc_id).copied() {
            Some(value) => value,
            None => {
                let acc_ptr =
                    expressions.append(Expression::LocalVariable(acc_local), Span::default());
                let load =
                    expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, load)),
                    Span::default(),
                );
                load
            }
        };
        let next = expressions.append(
            Expression::CooperativeMultiplyAdd { a, b, c },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, next)),
            Span::default(),
        );
        self.coop_acc_value_cache.borrow_mut().insert(acc_id, next);
        Ok(())
    }

    /// Flush every cached accumulator SSA back to its local. Called at the
    /// end of any scope where the cache must not leak (loop body iteration
    /// boundary, before reads of the local, end of program body, etc.).
    fn flush_coop_acc_cache(&self, expressions: &mut Arena<Expression>, body: &mut Block) {
        let drained: Vec<_> = self.coop_acc_value_cache.borrow_mut().drain().collect();
        for (acc_id, value) in drained {
            let acc_local = match self.coop_acc_locals.get(acc_id.index()).copied() {
                Some(l) => l,
                None => continue,
            };
            let acc_ptr = expressions.append(Expression::LocalVariable(acc_local), Span::default());
            body.push(
                Statement::Store {
                    pointer: acc_ptr,
                    value,
                },
                Span::default(),
            );
        }
    }

    fn lower_tile_while_true(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        max_iterations: u32,
        inner: &[TileStmt],
    ) -> Result<(), LowerError> {
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
            max_iterations,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        // Cache entries reference SSA handles emitted into the outer block;
        // they are out of scope inside this loop body. Snapshot, clear for the
        // body, restore on exit. The accumulator-value cache must be flushed
        // at iteration boundaries too — its SSA chain only carries within one
        // iteration (loop-carry goes through the accumulator local).
        let saved_frag: Vec<_> = self.coop_fragment_cache.borrow_mut().drain().collect();
        let saved_acc: Vec<_> = self.coop_acc_value_cache.borrow_mut().drain().collect();
        self.lower_tile_stmt_body(expressions, scratch, &mut loop_body, inner)?;
        self.flush_coop_acc_cache(expressions, &mut loop_body);
        {
            let mut cache = self.coop_fragment_cache.borrow_mut();
            cache.clear();
            for (k, v) in saved_frag {
                cache.insert(k, v);
            }
        }
        {
            let mut cache = self.coop_acc_value_cache.borrow_mut();
            cache.clear();
            for (k, v) in saved_acc {
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
        Ok(())
    }

    fn lower_copy_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        dst: TileRef,
        src: &StorageView,
        row_offset: &TileIndexExpr,
        col_offset: &TileIndexExpr,
    ) -> Result<(), LowerError> {
        let layout = self.tile_layout(dst)?;
        let [rows, cols] = Self::tile_shape(layout)?;
        let total = rows
            .checked_mul(cols)
            .ok_or(LowerError::UnsupportedOperation(
                "workgroup tile size overflow",
            ))?;
        let stride = Self::row_major_tile_stride(layout)?;
        let local = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let row_base = self.lower_tile_index_expr(expressions, scratch, body, row_offset, 0)?;
        let col_base = self.lower_tile_index_expr(expressions, scratch, body, col_offset, 0)?;

        let passes = total.div_ceil(self.workgroup_invocations);
        for pass in 0..passes {
            let full_pass = (pass + 1) * self.workgroup_invocations <= total;
            let mut guard_emits = Vec::new();
            let flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut guard_emits,
            );
            let condition = (!full_pass).then(|| {
                self.cmp_lit(
                    expressions,
                    &mut guard_emits,
                    BinaryOperator::Less,
                    flat,
                    total,
                )
            });
            let mut emits = Vec::new();
            let local_row = self.div_literal_u32_emitted(expressions, flat, cols, &mut emits);
            let local_col = self.mod_literal_u32_emitted(expressions, flat, cols, &mut emits);
            let global_row = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                row_base,
                local_row,
            );
            let global_col = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                col_base,
                local_col,
            );
            let tile_index = self.tile_matrix_index_inline(
                expressions,
                &mut emits,
                local_row,
                local_col,
                stride,
            );
            let (tile_ptr, tile_ptr_emits) =
                self.tile_dynamic_pointer(expressions, dst, tile_index)?;
            emits.extend(tile_ptr_emits);
            let (storage_index, storage_emits) =
                self.storage_index_from_coords(expressions, src, &[global_row, global_col])?;
            emits.extend(storage_emits);
            let (storage_ptr, storage_ptr_emits) =
                self.storage_dynamic_pointer(expressions, src, storage_index)?;
            emits.extend(storage_ptr_emits);
            let value = expressions.append(
                Expression::Load {
                    pointer: storage_ptr,
                },
                Span::default(),
            );
            emits.push(Self::single_expression_range(expressions, value));

            let mut accept = Block::new();
            Self::push_emits(&mut accept, emits);
            accept.push(
                Statement::Store {
                    pointer: tile_ptr,
                    value,
                },
                Span::default(),
            );

            Self::push_guarded_or_full_block(body, guard_emits, condition, accept);
        }
        Ok(())
    }

    fn lower_copy_quant_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        dst: TileRef,
        src: &QuantizedMatrix,
        row_offset: &TileIndexExpr,
        col_offset: &TileIndexExpr,
    ) -> Result<(), LowerError> {
        let layout = self.tile_layout(dst)?;
        let [rows, cols] = Self::tile_shape(layout)?;
        let stride = Self::row_major_tile_stride(layout)?;
        // We dispatch into format-specific N-wide vectorized helpers when N
        // divides the tile-row dimension. Otherwise we fall back to one
        // dequant per invocation per pass.
        let n = match src.format {
            GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q4K | GgmlQuantFormat::Q6K => 8,
            GgmlQuantFormat::Q5_0 => 16,
            _ => 0,
        };
        let local = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let row_base = self.lower_tile_index_expr(expressions, scratch, body, row_offset, 0)?;
        let col_base = self.lower_tile_index_expr(expressions, scratch, body, col_offset, 0)?;

        if n > 0 && rows.is_multiple_of(n) {
            let groups_per_col = rows / n;
            let total = groups_per_col * cols;
            let passes = total.div_ceil(self.workgroup_invocations);
            for pass in 0..passes {
                let full_pass = (pass + 1) * self.workgroup_invocations <= total;
                let mut guard_emits = Vec::new();
                let flat = self.add_literal_u32_emitted(
                    expressions,
                    local,
                    pass * self.workgroup_invocations,
                    &mut guard_emits,
                );
                let condition = (!full_pass).then(|| {
                    self.cmp_lit(
                        expressions,
                        &mut guard_emits,
                        BinaryOperator::Less,
                        flat,
                        total,
                    )
                });
                let mut emits = Vec::new();
                let local_k_group =
                    self.div_literal_u32_emitted(expressions, flat, cols, &mut emits);
                let local_col = self.mod_literal_u32_emitted(expressions, flat, cols, &mut emits);
                let local_k_base =
                    self.mul_literal_u32_emitted(expressions, local_k_group, n, &mut emits);
                let global_k_base = self.bin(
                    expressions,
                    &mut emits,
                    BinaryOperator::Add,
                    row_base,
                    local_k_base,
                );
                let global_col = self.bin(
                    expressions,
                    &mut emits,
                    BinaryOperator::Add,
                    col_base,
                    local_col,
                );
                let mut tile_ptrs = Vec::with_capacity(n as usize);
                for lane in 0..n {
                    let local_k =
                        self.add_literal_u32_emitted(expressions, local_k_base, lane, &mut emits);
                    let tile_index = self.tile_matrix_index_inline(
                        expressions,
                        &mut emits,
                        local_k,
                        local_col,
                        stride,
                    );
                    let (ptr, ptr_emits) =
                        self.tile_dynamic_pointer(expressions, dst, tile_index)?;
                    emits.extend(ptr_emits);
                    tile_ptrs.push(ptr);
                }
                let mut accept = Block::new();
                Self::push_emits(&mut accept, emits);
                let (values, value_emits) = match (src.format, n) {
                    (GgmlQuantFormat::Q8_0, 8) => {
                        self.dequantize_q8_0_values8(expressions, src, global_k_base, global_col)?
                    }
                    (GgmlQuantFormat::Q4K, 8) => {
                        self.dequantize_q4k_values8(expressions, src, global_k_base, global_col)?
                    }
                    (GgmlQuantFormat::Q6K, 8) => {
                        self.dequantize_q6k_values8(expressions, src, global_k_base, global_col)?
                    }
                    (GgmlQuantFormat::Q5_0, 16) => {
                        self.dequantize_q5_0_values16(expressions, src, global_k_base, global_col)?
                    }
                    _ => unreachable!(),
                };
                Self::push_emits(&mut accept, value_emits);
                for (ptr, value) in tile_ptrs.into_iter().zip(values.into_iter()) {
                    accept.push(
                        Statement::Store {
                            pointer: ptr,
                            value,
                        },
                        Span::default(),
                    );
                }
                Self::push_guarded_or_full_block(body, guard_emits, condition, accept);
            }
            return Ok(());
        }

        // Scalar fallback: one element per invocation per pass.
        let total = rows * cols;
        let passes = total.div_ceil(self.workgroup_invocations);
        for pass in 0..passes {
            let full_pass = (pass + 1) * self.workgroup_invocations <= total;
            let mut guard_emits = Vec::new();
            let flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut guard_emits,
            );
            let condition = (!full_pass).then(|| {
                self.cmp_lit(
                    expressions,
                    &mut guard_emits,
                    BinaryOperator::Less,
                    flat,
                    total,
                )
            });
            let mut emits = Vec::new();
            let local_row = self.div_literal_u32_emitted(expressions, flat, cols, &mut emits);
            let local_col = self.mod_literal_u32_emitted(expressions, flat, cols, &mut emits);
            let global_row = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                row_base,
                local_row,
            );
            let global_col = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                col_base,
                local_col,
            );
            let tile_index = self.tile_matrix_index_inline(
                expressions,
                &mut emits,
                local_row,
                local_col,
                stride,
            );
            let (tile_ptr, tile_ptr_emits) =
                self.tile_dynamic_pointer(expressions, dst, tile_index)?;
            emits.extend(tile_ptr_emits);
            let (value, value_emits) =
                self.dequantize_qvalue(expressions, src, global_row, global_col)?;
            emits.extend(value_emits);
            let mut accept = Block::new();
            Self::push_emits(&mut accept, emits);
            accept.push(
                Statement::Store {
                    pointer: tile_ptr,
                    value,
                },
                Span::default(),
            );
            Self::push_guarded_or_full_block(body, guard_emits, condition, accept);
        }
        Ok(())
    }

    fn lower_store_coop_acc(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        acc_id: CoopAccId,
        dst: &StorageView,
        row: &TileIndexExpr,
        col: &TileIndexExpr,
    ) -> Result<(), LowerError> {
        // Flush any pending acc SSA so the Load below sees the current value.
        self.flush_coop_acc_cache(expressions, body);
        let acc_local = self
            .coop_acc_locals
            .get(acc_id.index())
            .copied()
            .ok_or(LowerError::UnsupportedOperation("unknown coop acc"))?;
        let (stride_u, row_major) = Self::cooperative_store_layout(&dst.layout)?;
        let row_h = self.lower_tile_index_expr(expressions, scratch, body, row, 0)?;
        let col_h = self.lower_tile_index_expr(expressions, scratch, body, col, 0)?;
        let (storage_index, storage_emits) =
            self.storage_index_from_coords(expressions, dst, &[row_h, col_h])?;
        Self::push_emits(body, storage_emits);
        let (storage_ptr, ptr_emits) =
            self.storage_dynamic_pointer(expressions, dst, storage_index)?;
        Self::push_emits(body, ptr_emits);

        let stride =
            expressions.append(Expression::Literal(Literal::U32(stride_u)), Span::default());
        let acc_ptr = expressions.append(Expression::LocalVariable(acc_local), Span::default());
        let acc_value = expressions.append(Expression::Load { pointer: acc_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, acc_value)),
            Span::default(),
        );
        body.push(
            Statement::CooperativeStore {
                target: acc_value,
                data: CooperativeData {
                    pointer: storage_ptr,
                    stride,
                    row_major,
                },
            },
            Span::default(),
        );
        Ok(())
    }

    fn tile_shape(layout: &Layout) -> Result<[u32; 2], LowerError> {
        if layout.shape().rank() != 2 {
            return Err(LowerError::UnsupportedOperation(
                "workgroup tile must be rank-2",
            ));
        }
        Ok([
            layout.shape().dims()[0].get(),
            layout.shape().dims()[1].get(),
        ])
    }

    fn row_major_tile_stride(layout: &Layout) -> Result<u32, LowerError> {
        if layout.strides().rank() != 2 || layout.strides().values()[1] != 1 {
            return Err(LowerError::UnsupportedOperation(
                "workgroup tile must be row-major",
            ));
        }
        Ok(layout.strides().values()[0])
    }

    fn cooperative_store_layout(layout: &Layout) -> Result<(u32, bool), LowerError> {
        if layout.shape().rank() != 2 || layout.strides().rank() != 2 {
            return Err(LowerError::UnsupportedOperation(
                "cooperative store requires a rank-2 output view",
            ));
        }
        let strides = layout.strides().values();
        if strides[1] == 1 {
            Ok((strides[0], false))
        } else if strides[0] == 1 {
            Ok((strides[1], true))
        } else {
            Err(LowerError::UnsupportedOperation(
                "cooperative store requires row-major or column-major output strides",
            ))
        }
    }

    fn tile_matrix_index_inline(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        row: Handle<Expression>,
        col: Handle<Expression>,
        stride: u32,
    ) -> Handle<Expression> {
        let row_offset = self.mul_literal_u32_emitted(expressions, row, stride, emits);
        self.bin(expressions, emits, BinaryOperator::Add, row_offset, col)
    }
}
