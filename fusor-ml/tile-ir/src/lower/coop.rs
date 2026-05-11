use super::*;
use naga::{Barrier, CooperativeData, CooperativeRole, CooperativeSize};

const COOP_SIZE: CooperativeSize = CooperativeSize::Eight;

impl<'a> Lowerer<'a> {
    /// Lower non-store tile statements. Coop ops emit cooperative-matrix
    /// Loads/MMA/Store.
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
            TileStmt::Store(store) => self.lower_tile_store_stmt(expressions, scratch, body, store),
            TileStmt::StoreIndexed(store) => self.lower_tile_indexed_store_stmt(
                expressions,
                scratch,
                body,
                &store.dst,
                &store.index,
                &store.value,
                &store.mask,
            ),
            TileStmt::StoreLocal { dst, value } => {
                let value_ty = value.element();
                if value_ty != dst.element {
                    return Err(LowerError::LocalElementMismatch {
                        local: dst.id,
                        declared: dst.element,
                        used: value_ty,
                    });
                }
                let value = self.lower_tile_expr(expressions, scratch, body, value)?;
                let local = self.private_local(*dst)?;
                self.store_local(expressions, body, local, value);
                Ok(())
            }
            TileStmt::StoreWorkgroup { dst, index, value } => {
                let value_ty = value.element();
                if value_ty != dst.element {
                    return Err(LowerError::TileElementMismatch {
                        tile: dst.id,
                        declared: dst.element,
                        used: value_ty,
                    });
                }
                let value = self.lower_tile_expr(expressions, scratch, body, value)?;
                let index = self.lower_tile_expr(expressions, scratch, body, index)?;
                let pointer = self.tile_dynamic_pointer(expressions, *dst, index, body)?;
                body.push(Statement::Store { pointer, value }, Span::default());
                Ok(())
            }
            TileStmt::If {
                condition,
                accept,
                reject,
            } => {
                let condition_ty = condition.element();
                let condition =
                    self.lower_tile_expr(expressions, scratch, body, condition)?;
                let condition = self.condition_value(expressions, body, condition, condition_ty);
                let mut accept_block = Block::new();
                self.lower_tile_stmt_body(expressions, scratch, &mut accept_block, accept)?;
                let mut reject_block = Block::new();
                self.lower_tile_stmt_body(expressions, scratch, &mut reject_block, reject)?;
                body.push(
                    Statement::If {
                        condition,
                        accept: accept_block,
                        reject: reject_block,
                    },
                    Span::default(),
                );
                Ok(())
            }
            TileStmt::Loop { body: inner } => {
                self.flush_coop_acc_cache(expressions, body);
                let mut loop_body = Block::new();
                self.lower_tile_stmt_body(expressions, scratch, &mut loop_body, inner)?;
                self.flush_coop_acc_cache(expressions, &mut loop_body);
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
            TileStmt::Break => {
                body.push(Statement::Break, Span::default());
                Ok(())
            }
            TileStmt::Return => {
                body.push(Statement::Return { value: None }, Span::default());
                Ok(())
            }
            TileStmt::Barrier => {
                body.push(
                    Statement::ControlBarrier(Barrier::WORK_GROUP),
                    Span::default(),
                );
                Ok(())
            }
            TileStmt::ZeroCoopAcc { acc } => {
                let local = self.private_local(*acc)?;
                let ty = self.coop_c_ty.ok_or(LowerError::UnsupportedOperation(
                    "coop C type missing — tile program uses coop statements without cooperative-matrix support",
                ))?;
                let zero = expressions.append(Expression::ZeroValue(ty), Span::default());
                self.store_local(expressions, body, local, zero);
                Ok(())
            }
            TileStmt::CopyToWorkgroupTile {
                dst,
                src,
                row_offset,
                col_offset,
            } => match src {
                CopySource::Storage(view) => self.lower_copy_to_tile(
                    expressions,
                    scratch,
                    body,
                    *dst,
                    view,
                    row_offset,
                    col_offset,
                ),
                CopySource::Quantized(matrix) => self.lower_copy_quant_to_tile(
                    expressions,
                    scratch,
                    body,
                    *dst,
                    matrix,
                    row_offset,
                    col_offset,
                ),
            },
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
            TileStmt::Fold {
                count,
                iter_var,
                body: fold_body,
                accumulators,
            } => self.lower_tile_fold_stmt(
                expressions,
                scratch,
                body,
                count,
                *iter_var,
                fold_body,
                accumulators,
            ),
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
        row: &Expr,
        col: &Expr,
        role: CooperativeRole,
    ) -> Result<(), LowerError> {
        let layout = self.tile_layout(tile)?;
        let stride_u = Self::row_major_tile_stride(layout)?;
        let row_h = self.lower_tile_expr(expressions, scratch, body, row)?;
        let col_h = self.lower_tile_expr(expressions, scratch, body, col)?;
        let index = self.tile_matrix_index_inline(expressions, body, row_h, col_h, stride_u);
        let ptr = self.tile_dynamic_pointer(expressions, tile, index, body)?;
        let stride = self.u32(expressions, stride_u);
        let frag = self.emit(
            expressions,
            body,
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
        );
        self.coop_fragment_cache.borrow_mut().insert(id, frag);
        Ok(())
    }

    fn lower_coop_mma(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        acc: LocalRef,
        a_id: CoopFragmentId,
        b_id: CoopFragmentId,
    ) -> Result<(), LowerError> {
        let acc_local = self.private_local(acc)?;
        let a = self.coop_fragment_handle(a_id, "A")?;
        let b = self.coop_fragment_handle(b_id, "B")?;
        // Get the current SSA value of this accumulator: cache hit → reuse;
        // cache miss → emit one Load from the accumulator local. Subsequent
        // MMAs in this scope chain through SSA without touching the local.
        let c = match self.coop_acc_value_cache.borrow().get(&acc.id).copied() {
            Some(value) => value,
            None => {
                let acc_ptr =
                    self.local_var(expressions, acc_local);
                Self::emit_load(expressions, body, acc_ptr)
            }
        };
        let next = self.emit(expressions, body, Expression::CooperativeMultiplyAdd { a, b, c });
        self.coop_acc_value_cache.borrow_mut().insert(acc.id, next);
        Ok(())
    }

    /// Look up a previously-loaded coop fragment by id. Both operands of an
    /// MMA need this lookup; `role` is interpolated into the error message
    /// when the fragment isn't in the cache.
    fn coop_fragment_handle(
        &self,
        id: CoopFragmentId,
        role: &'static str,
    ) -> Result<Handle<Expression>, LowerError> {
        self.coop_fragment_cache
            .borrow()
            .get(&id)
            .copied()
            .ok_or(LowerError::UnsupportedOperation(match role {
                "A" => "coop_mma A fragment not loaded in current scope",
                "B" => "coop_mma B fragment not loaded in current scope",
                _ => "coop_mma fragment not loaded in current scope",
            }))
    }

    /// Flush every cached accumulator SSA back to its local. Called at the
    /// end of any scope where the cache must not leak (loop body iteration
    /// boundary, before reads of the local, end of program body, etc.).
    pub(super) fn flush_coop_acc_cache(&self, expressions: &mut Arena<Expression>, body: &mut Block) {
        let drained: Vec<_> = self.coop_acc_value_cache.borrow_mut().drain().collect();
        for (local_id, value) in drained {
            let acc_local = match self
                .private_locals
                .get(local_id.index())
                .copied()
                .flatten()
            {
                Some(l) => l,
                None => continue,
            };
            self.store_local(expressions, body, acc_local, value);
        }
    }

    fn lower_copy_passes(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<Expression>,
        total: u32,
        mut build_accept: impl FnMut(&mut Arena<Expression>, Handle<Expression>) -> Result<Block, LowerError>,
    ) -> Result<(), LowerError> {
        let passes = total.div_ceil(self.workgroup_invocations);
        for pass in 0..passes {
            let full_pass = (pass + 1) * self.workgroup_invocations <= total;
            let mut guard_block = Block::new();
            let flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut guard_block,
            );
            let condition = (!full_pass).then(|| {
                self.cmp_lit(
                    expressions,
                    &mut guard_block,
                    BinaryOperator::Less,
                    flat,
                    total,
                )
            });
            let accept = build_accept(expressions, flat)?;
            Self::push_guarded_or_full_block(body, guard_block, condition, accept);
        }
        Ok(())
    }

    fn lower_copy_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        dst: TileRef,
        src: &StorageView,
        row_offset: &Expr,
        col_offset: &Expr,
    ) -> Result<(), LowerError> {
        let layout = self.tile_layout(dst)?;
        let [rows, cols] = Self::tile_shape(layout)?;
        let total = rows
            .checked_mul(cols)
            .ok_or(LowerError::UnsupportedOperation(
                "workgroup tile size overflow",
            ))?;
        let stride = Self::row_major_tile_stride(layout)?;
        let local = Self::function_arg(expressions, LOCAL_INVOCATION_INDEX_ARG);
        let row_base = self.lower_tile_expr(expressions, scratch, body, row_offset)?;
        let col_base = self.lower_tile_expr(expressions, scratch, body, col_offset)?;

        self.lower_copy_passes(expressions, body, local, total, |expressions, flat| {
            let mut accept = Block::new();
            let CopyLaneCoords { global_row, global_col, tile_ptr } = self
                .copy_lane_pointer_and_globals(
                    expressions,
                    &mut accept,
                    flat,
                    dst,
                    cols,
                    stride,
                    row_base,
                    col_base,
                )?;
            let storage_index = self.storage_index_from_coords(
                expressions,
                src,
                &[global_row, global_col],
                &mut accept,
            )?;
            let storage_ptr =
                self.storage_dynamic_pointer(expressions, src, storage_index, &mut accept)?;
            let value = Self::emit_load(expressions, &mut accept, storage_ptr);
            accept.push(
                Statement::Store {
                    pointer: tile_ptr,
                    value,
                },
                Span::default(),
            );

            Ok(accept)
        })
    }

    fn lower_copy_quant_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        dst: TileRef,
        src: &QuantizedMatrix,
        row_offset: &Expr,
        col_offset: &Expr,
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
        let local = Self::function_arg(expressions, LOCAL_INVOCATION_INDEX_ARG);
        let row_base = self.lower_tile_expr(expressions, scratch, body, row_offset)?;
        let col_base = self.lower_tile_expr(expressions, scratch, body, col_offset)?;

        if n > 0 && rows.is_multiple_of(n) {
            let groups_per_col = rows / n;
            let total = groups_per_col * cols;
            self.lower_copy_passes(expressions, body, local, total, |expressions, flat| {
                let mut accept = Block::new();
                let local_k_group =
                    self.div_literal_u32_emitted(expressions, flat, cols, &mut accept);
                let local_col = self.mod_literal_u32_emitted(expressions, flat, cols, &mut accept);
                let local_k_base =
                    self.mul_literal_u32_emitted(expressions, local_k_group, n, &mut accept);
                let global_k_base = self.bin(
                    expressions,
                    &mut accept,
                    BinaryOperator::Add,
                    row_base,
                    local_k_base,
                );
                let global_col = self.bin(
                    expressions,
                    &mut accept,
                    BinaryOperator::Add,
                    col_base,
                    local_col,
                );
                let mut tile_ptrs = Vec::with_capacity(n as usize);
                for lane in 0..n {
                    let local_k =
                        self.add_literal_u32_emitted(expressions, local_k_base, lane, &mut accept);
                    let tile_index = self.tile_matrix_index_inline(
                        expressions,
                        &mut accept,
                        local_k,
                        local_col,
                        stride,
                    );
                    let ptr =
                        self.tile_dynamic_pointer(expressions, dst, tile_index, &mut accept)?;
                    tile_ptrs.push(ptr);
                }
                let values = match (src.format, n) {
                    (GgmlQuantFormat::Q8_0, 8) => self.dequantize_q8_0_values8(
                        expressions,
                        src,
                        global_k_base,
                        global_col,
                        &mut accept,
                    )?,
                    (GgmlQuantFormat::Q4K, 8) => self.dequantize_q4k_values8(
                        expressions,
                        src,
                        global_k_base,
                        global_col,
                        &mut accept,
                    )?,
                    (GgmlQuantFormat::Q6K, 8) => self.dequantize_q6k_values8(
                        expressions,
                        src,
                        global_k_base,
                        global_col,
                        &mut accept,
                    )?,
                    (GgmlQuantFormat::Q5_0, 16) => self.dequantize_q5_0_values16(
                        expressions,
                        src,
                        global_k_base,
                        global_col,
                        &mut accept,
                    )?,
                    _ => unreachable!(),
                };
                for (ptr, value) in tile_ptrs.into_iter().zip(values) {
                    accept.push(
                        Statement::Store {
                            pointer: ptr,
                            value,
                        },
                        Span::default(),
                    );
                }
                Ok(accept)
            })?;
            return Ok(());
        }

        // Scalar fallback: one element per invocation per pass.
        let total = rows * cols;
        self.lower_copy_passes(expressions, body, local, total, |expressions, flat| {
            let mut accept = Block::new();
            let CopyLaneCoords { global_row, global_col, tile_ptr } = self
                .copy_lane_pointer_and_globals(
                    expressions,
                    &mut accept,
                    flat,
                    dst,
                    cols,
                    stride,
                    row_base,
                    col_base,
                )?;
            let value =
                self.dequantize_qvalue(expressions, src, global_row, global_col, &mut accept)?;
            accept.push(
                Statement::Store {
                    pointer: tile_ptr,
                    value,
                },
                Span::default(),
            );
            Ok(accept)
        })
    }

    fn lower_store_coop_acc(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        acc: LocalRef,
        dst: &StorageView,
        row: &Expr,
        col: &Expr,
    ) -> Result<(), LowerError> {
        // Flush any pending acc SSA so the Load below sees the current value.
        self.flush_coop_acc_cache(expressions, body);
        let acc_local = self.private_local(acc)?;
        let (stride_u, row_major) = Self::cooperative_store_layout(&dst.layout)?;
        let row_h = self.lower_tile_expr(expressions, scratch, body, row)?;
        let col_h = self.lower_tile_expr(expressions, scratch, body, col)?;
        let storage_index =
            self.storage_index_from_coords(expressions, dst, &[row_h, col_h], body)?;
        let storage_ptr = self.storage_dynamic_pointer(expressions, dst, storage_index, body)?;

        let stride = self.u32(expressions, stride_u);
        let acc_ptr = self.local_var(expressions, acc_local);
        let acc_value = Self::emit_load(expressions, body, acc_ptr);
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
        body: &mut Block,
        row: Handle<Expression>,
        col: Handle<Expression>,
        stride: u32,
    ) -> Handle<Expression> {
        let row_offset = self.mul_literal_u32_emitted(expressions, row, stride, body);
        self.add(expressions, body, row_offset, col)
    }

    /// Resolve a flat invocation index into the destination tile pointer plus
    /// the source's global (row, col). Shared by `lower_copy_to_tile` and the
    /// scalar fallback in `lower_copy_quant_to_tile`.
    #[allow(clippy::too_many_arguments)]
    fn copy_lane_pointer_and_globals(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        flat: Handle<Expression>,
        dst: TileRef,
        cols: u32,
        stride: u32,
        row_base: Handle<Expression>,
        col_base: Handle<Expression>,
    ) -> Result<CopyLaneCoords, LowerError> {
        let local_row = self.div_literal_u32_emitted(expressions, flat, cols, body);
        let local_col = self.mod_literal_u32_emitted(expressions, flat, cols, body);
        let global_row = self.add(expressions, body, row_base, local_row);
        let global_col = self.add(expressions, body, col_base, local_col);
        let tile_index =
            self.tile_matrix_index_inline(expressions, body, local_row, local_col, stride);
        let tile_ptr = self.tile_dynamic_pointer(expressions, dst, tile_index, body)?;
        Ok(CopyLaneCoords { global_row, global_col, tile_ptr })
    }
}

/// One copy lane's resolved global source (row, col) and destination tile
/// pointer. Returned by `Lowerer::copy_lane_pointer_and_globals`.
pub(super) struct CopyLaneCoords {
    pub global_row: Handle<Expression>,
    pub global_col: Handle<Expression>,
    pub tile_ptr: Handle<Expression>,
}
