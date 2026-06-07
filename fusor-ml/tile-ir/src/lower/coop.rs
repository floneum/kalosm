use super::*;
use crate::ir::{Addr, CoopSrc};
use naga::{Barrier, CooperativeData, CooperativeRole};

impl<'a> Lowerer<'a> {
    pub(super) fn lower_stmt_body(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        stmts: &[Stmt],
    ) -> Result<(), LowerError> {
        for stmt in stmts {
            self.lower_stmt(expressions, body, stmt)?;
        }
        Ok(())
    }

    pub(super) fn lower_stmt(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        stmt: &Stmt,
    ) -> Result<(), LowerError> {
        match stmt {
            Stmt::Store {
                dst,
                addr,
                value,
                mask,
            } => self.lower_store_stmt(expressions, body, dst, addr, value, mask),
            Stmt::StoreLocal { dst, value } => {
                self.lower_store_local(expressions, body, dst, value)
            }
            Stmt::StoreTile { dst, index, value } => {
                let value = self.lower_expr(expressions, body, value)?;
                let index = self.lower_expr(expressions, body, index)?;
                let pointer = self.tile_dynamic_pointer(expressions, dst, index, body)?;
                body.push(Statement::Store { pointer, value }, Span::default());
                Ok(())
            }
            Stmt::FillTile { dst, value } => self.lower_fill_tile(expressions, body, dst, value),
            Stmt::CoopStore { acc, dst, addr } => {
                self.lower_store_coop_acc(expressions, body, acc, dst, addr)
            }
            Stmt::If {
                condition,
                accept,
                reject,
            } => {
                let condition_ty = condition.element();
                let condition = self.lower_expr(expressions, body, condition)?;
                let condition = self.condition_value(expressions, body, condition, condition_ty);
                let accept_block = self.lower_branch_block(expressions, accept)?;
                let reject_block = self.lower_branch_block(expressions, reject)?;
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
            Stmt::Loop {
                count: Some(count),
                index,
                accumulators,
                body: loop_body,
            } => self.lower_counted_loop(
                expressions,
                body,
                count,
                index.as_ref(),
                accumulators,
                loop_body,
            ),
            Stmt::Loop {
                count: None,
                body: inner,
                ..
            } => {
                self.flush_coop_acc_cache(expressions, body);
                let mut loop_body = Block::new();
                self.lower_stmt_body(expressions, &mut loop_body, inner)?;
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
            Stmt::Break => {
                body.push(Statement::Break, Span::default());
                Ok(())
            }
            Stmt::Return => {
                body.push(Statement::Return { value: None }, Span::default());
                Ok(())
            }
            Stmt::Barrier => {
                body.push(
                    Statement::ControlBarrier(Barrier::WORK_GROUP),
                    Span::default(),
                );
                Ok(())
            }
        }
    }

    /// Lower `Stmt::StoreLocal`. Coop accumulators participate in the SSA
    /// acc-value memo: an MMA-valued store updates the memo and defers the
    /// store to the next flush (giving 1 Load + N MMA + 1 Store per iteration);
    /// any other coop-valued store (zero/set) writes immediately and clears the
    /// memo so the next MMA reloads once.
    fn lower_store_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        dst: &Local,
        value: &Expr,
    ) -> Result<(), LowerError> {
        let is_coop = matches!(dst.element, ElementType::CoopMatrix { .. });
        if is_coop && matches!(value.kind(), ExprKind::CoopMma { .. }) {
            let next = self.lower_expr(expressions, body, value)?;
            self.coop_acc_value_cache
                .borrow_mut()
                .insert(local_key_decl(dst), next);
            return Ok(());
        }
        let value = self.lower_expr(expressions, body, value)?;
        let local = self.private_local(dst)?;
        if is_coop {
            self.coop_acc_value_cache
                .borrow_mut()
                .remove(&local_key_decl(dst));
        }
        self.store_local(expressions, body, local, value);
        Ok(())
    }

    /// Lower an `ExprKind::CoopLoad` to a cooperative-matrix load expression.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::lower) fn lower_coop_load(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
        src: &CoopSrc,
    ) -> Result<Handle<Expression>, LowerError> {
        let role = match role {
            CoopMatrixRole::A => CooperativeRole::A,
            CoopMatrixRole::B => CooperativeRole::B,
            CoopMatrixRole::C => CooperativeRole::C,
        };
        match src {
            CoopSrc::TileRegion { tile, row, col } => {
                let layout = self.tile_layout(tile);
                let stride_u = Self::row_major_tile_stride(layout)?;
                let _ = scalar;
                let row_h = self.lower_expr(expressions, body, row)?;
                let col_h = self.lower_expr(expressions, body, col)?;
                let index =
                    self.tile_matrix_index_inline(expressions, body, row_h, col_h, stride_u);
                let ptr = self.tile_dynamic_pointer(expressions, tile, index, body)?;
                let stride = self.u32(expressions, stride_u);
                Ok(self.emit(
                    expressions,
                    body,
                    Expression::CooperativeLoad {
                        columns: Self::cooperative_size(cols)?,
                        rows: Self::cooperative_size(rows)?,
                        role,
                        data: CooperativeData {
                            pointer: ptr,
                            stride,
                            // Metal's simdgroup matrix orientation makes
                            // row-major A/B fragments multiply as B * A.
                            // Keep Fusor's logical A * B by holding coop
                            // fragments transposed internally.
                            row_major: false,
                        },
                    },
                ))
            }
            CoopSrc::BroadcastCol { src, col } => {
                let layout = self.storage_layout(src);
                if layout.shape().rank() != 1 {
                    return Err(LowerError::UnsupportedOperation(
                        "coop broadcast load expects rank-1 storage",
                    ));
                }
                let col_h = self.lower_expr(expressions, body, col)?;
                let ptr = self.storage_dynamic_pointer(expressions, src, col_h, body)?;
                let stride = self.u32(expressions, 0);
                Ok(self.emit(
                    expressions,
                    body,
                    Expression::CooperativeLoad {
                        columns: Self::cooperative_size(cols)?,
                        rows: Self::cooperative_size(rows)?,
                        role,
                        data: CooperativeData {
                            pointer: ptr,
                            stride,
                            // C-broadcast participates in the same transposed
                            // accumulator representation as A/B fragments.
                            row_major: false,
                        },
                    },
                ))
            }
        }
    }

    /// Lower an `ExprKind::CoopMma`. When `c` is a `LoadLocal(acc)` of a coop
    /// accumulator with a live memo entry, the cached SSA value is reused so no
    /// extra `Load` is emitted; otherwise `c` lowers normally (a single Load).
    pub(in crate::lower) fn lower_coop_mma(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        a: &Expr,
        b: &Expr,
        c: &Expr,
    ) -> Result<Handle<Expression>, LowerError> {
        let a = self.lower_expr(expressions, body, a)?;
        let b = self.lower_expr(expressions, body, b)?;
        let c = self.lower_expr(expressions, body, c)?;
        Ok(self.emit(
            expressions,
            body,
            Expression::CooperativeMultiplyAdd { a, b, c },
        ))
    }

    /// The current SSA value cached for a coop accumulator local, if any.
    pub(in crate::lower) fn coop_acc_value(&self, local: &Local) -> Option<Handle<Expression>> {
        self.coop_acc_value_cache
            .borrow()
            .get(&local_key_decl(local))
            .copied()
    }

    /// Flush every cached accumulator SSA back to its local. Called at the end
    /// of any scope where the cache must not leak.
    pub(super) fn flush_coop_acc_cache(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
    ) {
        let drained: Vec<_> = self.coop_acc_value_cache.borrow_mut().drain().collect();
        for (decl_ptr, value) in drained {
            let acc_local = match self.locals.borrow().get(&(decl_ptr as *const ())).copied() {
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
        mut build_accept: impl FnMut(
            &mut Arena<Expression>,
            Handle<Expression>,
        ) -> Result<Block, LowerError>,
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

    /// Lower a `Stmt::FillTile`. `value` is the per-element source — a masked
    /// `Load`, dense or quantized. The dense and quant cases share one emitter;
    /// each recognizes the contiguous vec4 fast path internally.
    fn lower_fill_tile(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        dst: &Tile,
        value: &Expr,
    ) -> Result<(), LowerError> {
        let ExprKind::Load {
            src,
            addr,
            mask,
            fill,
        } = value.kind()
        else {
            return Err(LowerError::UnsupportedOperation(
                "FillTile value must be a Load",
            ));
        };
        let Addr::Rc2 { row, col } = addr else {
            return Err(LowerError::UnsupportedOperation(
                "FillTile value must be a rank-2 Load",
            ));
        };
        match src {
            Source::Storage(view) => {
                self.lower_copy_to_tile(expressions, body, dst, view, row, col, mask, fill)
            }
            Source::Quantized(matrix) => {
                self.lower_copy_quant_to_tile(expressions, body, dst, matrix, row, col)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn lower_copy_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        dst: &Tile,
        src: &StorageView,
        row_offset: &Expr,
        col_offset: &Expr,
        _mask: &Expr,
        _fill: &Expr,
    ) -> Result<(), LowerError> {
        let layout = self.tile_layout(dst);
        let [rows, cols] = Self::tile_shape(layout)?;
        let stride = Self::row_major_tile_stride(layout)?;
        let local = Self::function_arg(expressions, LOCAL_INVOCATION_INDEX_ARG);
        let row_base = self.lower_expr(expressions, body, row_offset)?;
        let col_base = self.lower_expr(expressions, body, col_offset)?;

        const VEC: u32 = 4;
        if cols.is_multiple_of(VEC) && Self::storage_has_unit_inner_stride(&src.layout) {
            let groups_per_row = cols / VEC;
            let total_groups =
                rows.checked_mul(groups_per_row)
                    .ok_or(LowerError::UnsupportedOperation(
                        "workgroup tile size overflow",
                    ))?;
            return self.lower_copy_passes(
                expressions,
                body,
                local,
                total_groups,
                |expressions, flat| {
                    let mut accept = Block::new();
                    let local_row = self.div_literal_u32_emitted(
                        expressions,
                        flat,
                        groups_per_row,
                        &mut accept,
                    );
                    let local_col_group = self.mod_literal_u32_emitted(
                        expressions,
                        flat,
                        groups_per_row,
                        &mut accept,
                    );
                    let local_col_base = self.mul_literal_u32_emitted(
                        expressions,
                        local_col_group,
                        VEC,
                        &mut accept,
                    );
                    let global_row = self.add(expressions, &mut accept, row_base, local_row);
                    let global_col_base =
                        self.add(expressions, &mut accept, col_base, local_col_base);
                    let storage_index_base = self.storage_index_from_coords(
                        expressions,
                        src,
                        &[global_row, global_col_base],
                        &mut accept,
                    )?;
                    let tile_index_base = self.tile_matrix_index_inline(
                        expressions,
                        &mut accept,
                        local_row,
                        local_col_base,
                        stride,
                    );
                    let mut values = [None; VEC as usize];
                    for i in 0..VEC {
                        let storage_index = self.add_literal_u32_emitted(
                            expressions,
                            storage_index_base,
                            i,
                            &mut accept,
                        );
                        let storage_ptr = self.storage_dynamic_pointer(
                            expressions,
                            src,
                            storage_index,
                            &mut accept,
                        )?;
                        values[i as usize] =
                            Some(Self::emit_load(expressions, &mut accept, storage_ptr));
                    }
                    for i in 0..VEC {
                        let tile_index = self.add_literal_u32_emitted(
                            expressions,
                            tile_index_base,
                            i,
                            &mut accept,
                        );
                        let tile_ptr =
                            self.tile_dynamic_pointer(expressions, dst, tile_index, &mut accept)?;
                        accept.push(
                            Statement::Store {
                                pointer: tile_ptr,
                                value: values[i as usize].expect("loaded above"),
                            },
                            Span::default(),
                        );
                    }
                    Ok(accept)
                },
            );
        }

        let total = rows
            .checked_mul(cols)
            .ok_or(LowerError::UnsupportedOperation(
                "workgroup tile size overflow",
            ))?;
        let lane_layout = CopyLaneLayout {
            cols,
            stride,
            row_base,
            col_base,
        };

        self.lower_copy_passes(expressions, body, local, total, |expressions, flat| {
            let mut accept = Block::new();
            let CopyLaneCoords {
                global_row,
                global_col,
                tile_ptr,
            } = self.copy_lane_pointer_and_globals(
                expressions,
                &mut accept,
                flat,
                dst,
                lane_layout,
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

    /// True if the storage view's innermost axis is contiguous (stride 1).
    fn storage_has_unit_inner_stride(layout: &Layout) -> bool {
        if !layout.is_affine() {
            return false;
        }
        let strides = layout.affine_strides();
        strides.last().copied() == Some(1)
    }

    fn lower_copy_quant_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        dst: &Tile,
        src: &QuantizedMatrix,
        row_offset: &Expr,
        col_offset: &Expr,
    ) -> Result<(), LowerError> {
        if !matches!(dst.element, ElementType::F32 | ElementType::F16) {
            return Err(LowerError::UnsupportedOperation(
                "quantized tile copy requires an f32/f16 tile",
            ));
        }
        let layout = self.tile_layout(dst);
        let [rows, cols] = Self::tile_shape(layout)?;
        let stride = Self::row_major_tile_stride(layout)?;
        let n = match src.format {
            GgmlQuantFormat::Q8_0
            | GgmlQuantFormat::Q8_0Native
            | GgmlQuantFormat::Q4K
            | GgmlQuantFormat::Q4KNative
            | GgmlQuantFormat::Q6K
            | GgmlQuantFormat::Q6KNative => 8,
            GgmlQuantFormat::Q5_0 | GgmlQuantFormat::Q5_0Native => 16,
            _ => 0,
        };
        let local = Self::function_arg(expressions, LOCAL_INVOCATION_INDEX_ARG);
        let row_base = self.lower_expr(expressions, body, row_offset)?;
        let col_base = self.lower_expr(expressions, body, col_offset)?;

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
                let last_lane =
                    self.add_literal_u32_emitted(expressions, global_k_base, n - 1, &mut accept);
                let row_ok = self.cmp_lit(
                    expressions,
                    &mut accept,
                    BinaryOperator::Less,
                    last_lane,
                    src.rows,
                );
                let col_ok = self.cmp_lit(
                    expressions,
                    &mut accept,
                    BinaryOperator::Less,
                    global_col,
                    src.cols,
                );
                let in_bounds = self.bin(
                    expressions,
                    &mut accept,
                    BinaryOperator::LogicalAnd,
                    row_ok,
                    col_ok,
                );

                let mut in_bounds_body = Block::new();
                let values = match (src.format, n) {
                    (GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q8_0Native, 8) => self
                        .dequantize_q8_0_values8(
                            expressions,
                            src,
                            global_k_base,
                            global_col,
                            &mut in_bounds_body,
                        )?,
                    (GgmlQuantFormat::Q4K | GgmlQuantFormat::Q4KNative, 8) => self
                        .dequantize_q4k_values8(
                            expressions,
                            src,
                            global_k_base,
                            global_col,
                            &mut in_bounds_body,
                        )?,
                    (GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative, 8) => self
                        .dequantize_q6k_values8(
                            expressions,
                            src,
                            global_k_base,
                            global_col,
                            &mut in_bounds_body,
                        )?,
                    (GgmlQuantFormat::Q5_0 | GgmlQuantFormat::Q5_0Native, 16) => self
                        .dequantize_q5_0_values16(
                            expressions,
                            src,
                            global_k_base,
                            global_col,
                            &mut in_bounds_body,
                        )?,
                    _ => unreachable!(),
                };
                for (ptr, value) in tile_ptrs.iter().copied().zip(values) {
                    let value = self.cast_tile_value(
                        expressions,
                        &mut in_bounds_body,
                        value,
                        ElementType::F32,
                        dst.element,
                    );
                    in_bounds_body.push(
                        Statement::Store {
                            pointer: ptr,
                            value,
                        },
                        Span::default(),
                    );
                }

                let zero_f32 = self.f32(expressions, 0.0);
                let zero = self.cast_tile_value(
                    expressions,
                    &mut accept,
                    zero_f32,
                    ElementType::F32,
                    dst.element,
                );
                let mut out_of_bounds_body = Block::new();
                for ptr in tile_ptrs {
                    out_of_bounds_body.push(
                        Statement::Store {
                            pointer: ptr,
                            value: zero,
                        },
                        Span::default(),
                    );
                }
                accept.push(
                    Statement::If {
                        condition: in_bounds,
                        accept: in_bounds_body,
                        reject: out_of_bounds_body,
                    },
                    Span::default(),
                );
                Ok(accept)
            })?;
            return Ok(());
        }

        let total = rows * cols;
        let lane_layout = CopyLaneLayout {
            cols,
            stride,
            row_base,
            col_base,
        };
        self.lower_copy_passes(expressions, body, local, total, |expressions, flat| {
            let mut accept = Block::new();
            let CopyLaneCoords {
                global_row,
                global_col,
                tile_ptr,
            } = self.copy_lane_pointer_and_globals(
                expressions,
                &mut accept,
                flat,
                dst,
                lane_layout,
            )?;
            let row_ok = self.cmp_lit(
                expressions,
                &mut accept,
                BinaryOperator::Less,
                global_row,
                src.rows,
            );
            let col_ok = self.cmp_lit(
                expressions,
                &mut accept,
                BinaryOperator::Less,
                global_col,
                src.cols,
            );
            let in_bounds = self.bin(
                expressions,
                &mut accept,
                BinaryOperator::LogicalAnd,
                row_ok,
                col_ok,
            );
            let mut in_bounds_body = Block::new();
            let value = self.dequantize_qvalue(
                expressions,
                src,
                global_row,
                global_col,
                &mut in_bounds_body,
            )?;
            let value = self.cast_tile_value(
                expressions,
                &mut in_bounds_body,
                value,
                ElementType::F32,
                dst.element,
            );
            in_bounds_body.push(
                Statement::Store {
                    pointer: tile_ptr,
                    value,
                },
                Span::default(),
            );
            let zero_f32 = self.f32(expressions, 0.0);
            let zero = self.cast_tile_value(
                expressions,
                &mut accept,
                zero_f32,
                ElementType::F32,
                dst.element,
            );
            let mut out_of_bounds_body = Block::new();
            out_of_bounds_body.push(
                Statement::Store {
                    pointer: tile_ptr,
                    value: zero,
                },
                Span::default(),
            );
            accept.push(
                Statement::If {
                    condition: in_bounds,
                    accept: in_bounds_body,
                    reject: out_of_bounds_body,
                },
                Span::default(),
            );
            Ok(accept)
        })
    }

    /// Lower `Stmt::CoopStore` to a `Statement::CooperativeStore`. Never routed
    /// through the per-lane `Store` path.
    fn lower_store_coop_acc(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        acc: &Local,
        dst: &StorageView,
        addr: &Addr,
    ) -> Result<(), LowerError> {
        // Flush any pending acc SSA so the Load below sees the current value.
        self.flush_coop_acc_cache(expressions, body);
        let acc_local = self.private_local(acc)?;
        let (stride_u, row_major) = Self::cooperative_store_layout(&dst.layout)?;
        let (row, col) = match addr {
            Addr::Rc2 { row, col } => (row, col),
            Addr::Linear(_) => {
                return Err(LowerError::UnsupportedOperation(
                    "cooperative store requires a rank-2 address",
                ));
            }
        };
        let row_h = self.lower_expr(expressions, body, row)?;
        let col_h = self.lower_expr(expressions, body, col)?;
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
                    // Accumulators are transposed internally; invert the
                    // destination layout flag to write logical row/col order.
                    row_major: !row_major,
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
        if !layout.is_affine() {
            return Err(LowerError::UnsupportedOperation(
                "workgroup tile must be row-major",
            ));
        }
        let strides = layout.affine_strides();
        if strides.len() != 2 || strides[1] != 1 {
            return Err(LowerError::UnsupportedOperation(
                "workgroup tile must be row-major",
            ));
        }
        Ok(strides[0])
    }

    fn cooperative_store_layout(layout: &Layout) -> Result<(u32, bool), LowerError> {
        if !layout.is_affine() || layout.shape().rank() != 2 {
            return Err(LowerError::UnsupportedOperation(
                "cooperative store requires a rank-2 output view",
            ));
        }
        let strides = layout.affine_strides();
        if strides[1] == 1 {
            Ok((strides[0], true))
        } else if strides[0] == 1 {
            Ok((strides[1], false))
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

    fn copy_lane_pointer_and_globals(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        flat: Handle<Expression>,
        dst: &Tile,
        layout: CopyLaneLayout,
    ) -> Result<CopyLaneCoords, LowerError> {
        let CopyLaneLayout {
            cols,
            stride,
            row_base,
            col_base,
        } = layout;
        let local_row = self.div_literal_u32_emitted(expressions, flat, cols, body);
        let local_col = self.mod_literal_u32_emitted(expressions, flat, cols, body);
        let global_row = self.add(expressions, body, row_base, local_row);
        let global_col = self.add(expressions, body, col_base, local_col);
        let tile_index =
            self.tile_matrix_index_inline(expressions, body, local_row, local_col, stride);
        let tile_ptr = self.tile_dynamic_pointer(expressions, dst, tile_index, body)?;
        Ok(CopyLaneCoords {
            global_row,
            global_col,
            tile_ptr,
        })
    }
}

/// `Rc::as_ptr` key for a `Local` decl, typed as `*const LocalDecl` for the
/// coop acc-value memo.
fn local_key_decl(local: &Local) -> *const LocalDecl {
    std::rc::Rc::as_ptr(local)
}

/// One copy lane's resolved global source (row, col) and destination tile
/// pointer.
struct CopyLaneCoords {
    global_row: Handle<Expression>,
    global_col: Handle<Expression>,
    tile_ptr: Handle<Expression>,
}

#[derive(Clone, Copy)]
struct CopyLaneLayout {
    cols: u32,
    stride: u32,
    row_base: Handle<Expression>,
    col_base: Handle<Expression>,
}
