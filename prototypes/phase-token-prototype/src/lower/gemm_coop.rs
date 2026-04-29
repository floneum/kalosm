use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_shared_gemm_loop_to_storage_coop8(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a_load: &crate::CooperativeLoadOp,
        b_load: &crate::CooperativeLoadOp,
        op: &GemmDescriptor,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let _coop_c_ty = self.coop_f32_c_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix C type was not allocated",
        ))?;

        if a_load.dst != op.a || b_load.dst != op.b {
            return Err(LowerError::UnsupportedOperation(
                "shared cooperative gemm load mismatch",
            ));
        }

        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        let (subgroup_rows, subgroup_cols, partition) = if self.coop_subgroups > 1 {
            if m == 64 && n == 64 && self.coop_subgroups == 4 {
                (
                    32,
                    32,
                    CoopPartition::InterleavedGrid {
                        row_groups: 2,
                        col_groups: 2,
                    },
                )
            } else if n >= m && n % self.coop_subgroups == 0 {
                (m, n / self.coop_subgroups, CoopPartition::Columns)
            } else if m % self.coop_subgroups == 0 {
                (m / self.coop_subgroups, n, CoopPartition::Rows)
            } else if n % self.coop_subgroups == 0 {
                (m, n / self.coop_subgroups, CoopPartition::Columns)
            } else {
                return Err(LowerError::UnsupportedOperation(
                    "cooperative matrix multi-subgroup tile width mismatch",
                ));
            }
        } else {
            (m, n, CoopPartition::Single)
        };
        if m == 0
            || n == 0
            || m % 8 != 0
            || n % 8 != 0
            || subgroup_rows % 8 != 0
            || subgroup_cols % 8 != 0
            || (subgroup_rows / 8) * (subgroup_cols / 8) > scratch.coop_accs.len() as u32
            || m_acc != m
            || n_acc != n
            || m_dst != m
            || n_dst != n
            || k_a != k_b
            || k_a % 8 != 0
            || a_layout.memory_level() != MemoryLevel::Workgroup
            || b_layout.memory_level() != MemoryLevel::Workgroup
            || acc_layout.memory_level() != MemoryLevel::Private
        {
            return Err(LowerError::UnsupportedOperation(
                "shared cooperative matrix lowering requires workgroup A/B tiles, a private accumulator, 8x8 fragments, and K divisible by 8",
            ));
        }
        if outer_iterations == 0 {
            return Err(LowerError::UnsupportedOperation(
                "cooperative matrix gemm loop iteration count must be non-zero",
            ));
        }

        let tile_rows = subgroup_rows / 8;
        let tile_cols = subgroup_cols / 8;
        let fragment_count = (tile_rows * tile_cols) as usize;
        let acc_locals = scratch.coop_accs[..fragment_count]
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or(LowerError::UnsupportedOperation(
                "cooperative matrix accumulator locals were not allocated",
            ))?;

        let a_stride = Self::row_major_matrix_leading_stride(a_layout)?;
        let b_stride = Self::row_major_matrix_leading_stride(b_layout)?;
        let dst_stride = Self::row_major_matrix_leading_stride(dst_layout)?;
        let a_stride =
            expressions.append(Expression::Literal(Literal::U32(a_stride)), Span::default());
        let b_stride =
            expressions.append(Expression::Literal(Literal::U32(b_stride)), Span::default());
        let dst_stride = expressions.append(
            Expression::Literal(Literal::U32(dst_stride)),
            Span::default(),
        );

        let mut body = Block::new();
        let mut acc_pointers = Vec::with_capacity(acc_locals.len());
        for local in acc_locals {
            let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
            acc_pointers.push(pointer);
        }
        let active_subgroup = if self.workgroup_invocations > self.coop_subgroups * 32 {
            let subgroup_id = expressions.append(
                Expression::FunctionArgument(SUBGROUP_ID_ARG),
                Span::default(),
            );
            let subgroup_limit = expressions.append(
                Expression::Literal(Literal::U32(self.coop_subgroups)),
                Span::default(),
            );
            let active = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Less,
                    left: subgroup_id,
                    right: subgroup_limit,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::range_from(expressions, subgroup_id, active)),
                Span::default(),
            );
            Some(active)
        } else {
            None
        };

        let mut inner_body = Block::new();
        let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        inner_body.push(Statement::Emit(k_emit), Span::default());
        let mut base_k_emits = Vec::new();
        let base_k = self.mul_literal_u32_emitted(expressions, k_chunk, 8, &mut base_k_emits);
        Self::push_emits(&mut inner_body, base_k_emits);
        self.append_shared_coop_k_chunk(
            expressions,
            &mut inner_body,
            op.a,
            op.b,
            &acc_pointers,
            tile_rows,
            tile_cols,
            base_k,
            a_stride,
            b_stride,
            subgroup_cols,
            subgroup_rows,
            partition,
        )?;

        let mut outer_body = Block::new();
        outer_body.push(
            self.lower_cooperative_load(expressions, scratch.tile_index, a_load.dst, &a_load.src)?,
            Span::default(),
        );
        outer_body.push(
            self.lower_cooperative_load(expressions, scratch.tile_index, b_load.dst, &b_load.src)?,
            Span::default(),
        );
        outer_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        let mma_loop = self.counted_loop(expressions, scratch.mma_k, k_a / 8, inner_body);
        if let Some(active_subgroup) = active_subgroup {
            outer_body.push(
                Statement::If {
                    condition: active_subgroup,
                    accept: Block::from_vec(vec![mma_loop]),
                    reject: Block::new(),
                },
                Span::default(),
            );
        } else {
            outer_body.push(mma_loop, Span::default());
        }
        outer_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations,
                outer_body,
            ),
            Span::default(),
        );

        let (subgroup_row_base, subgroup_col_base) = self.subgroup_partition_bases(
            expressions,
            &mut body,
            partition,
            subgroup_rows,
            subgroup_cols,
        );
        let mut store_body = Block::new();
        for row_tile in 0..tile_rows {
            for col_tile in 0..tile_cols {
                let acc_index = (row_tile * tile_cols + col_tile) as usize;
                let acc_value = expressions.append(
                    Expression::Load {
                        pointer: acc_pointers[acc_index],
                    },
                    Span::default(),
                );
                store_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, acc_value)),
                    Span::default(),
                );
                let row = if let Some(subgroup_row_base) = subgroup_row_base {
                    let mut row_emits = Vec::new();
                    let row = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_row_base,
                        Self::coop_tile_offset(partition, true, row_tile),
                        &mut row_emits,
                    );
                    Self::push_emits(&mut store_body, row_emits);
                    row
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, true, row_tile,
                        ))),
                        Span::default(),
                    )
                };
                let col = if let Some(subgroup_col_base) = subgroup_col_base {
                    let mut col_emits = Vec::new();
                    let col = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_col_base,
                        Self::coop_tile_offset(partition, false, col_tile),
                        &mut col_emits,
                    );
                    Self::push_emits(&mut store_body, col_emits);
                    col
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, false, col_tile,
                        ))),
                        Span::default(),
                    )
                };
                let (dst_index, dst_index_emits) =
                    self.storage_index_from_coords(expressions, dst, &[row, col])?;
                Self::push_emits(&mut store_body, dst_index_emits);
                let (dst_pointer, dst_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, dst, dst_index)?;
                Self::push_emits(&mut store_body, dst_pointer_emits);
                store_body.push(
                    Statement::CooperativeStore {
                        target: acc_value,
                        data: CooperativeData {
                            pointer: dst_pointer,
                            stride: dst_stride,
                            row_major: false,
                        },
                    },
                    Span::default(),
                );
            }
        }
        if let Some(active_subgroup) = active_subgroup {
            body.push(
                Statement::If {
                    condition: active_subgroup,
                    accept: store_body,
                    reject: Block::new(),
                },
                Span::default(),
            );
        } else {
            body.push(Statement::Block(store_body), Span::default());
        }

        Ok(Statement::Block(body))
    }

    pub(super) fn lower_storage_gemm_loop_to_storage_coop8(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        a: &StorageView,
        b: &StorageView,
        dst: &StorageView,
        outer_iterations: u32,
    ) -> Result<Statement, LowerError> {
        let _coop_a_ty = self.coop_f32_a_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix A type was not allocated",
        ))?;
        let _coop_b_ty = self.coop_f32_b_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix B type was not allocated",
        ))?;
        let _coop_c_ty = self.coop_f32_c_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix C type was not allocated",
        ))?;

        let a_layout = self.storage_layout(a)?;
        let b_layout = self.storage_layout(b)?;
        let dst_layout = self.storage_layout(dst)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_dst, n_dst] = Self::matrix_shape(dst_layout)?;
        let (subgroup_rows, subgroup_cols, partition) = if self.coop_subgroups > 1 {
            if m == 64 && n == 64 && self.coop_subgroups == 4 {
                (
                    32,
                    32,
                    CoopPartition::InterleavedGrid {
                        row_groups: 2,
                        col_groups: 2,
                    },
                )
            } else if n >= m && n % self.coop_subgroups == 0 {
                (m, n / self.coop_subgroups, CoopPartition::Columns)
            } else if m % self.coop_subgroups == 0 {
                (m / self.coop_subgroups, n, CoopPartition::Rows)
            } else if n % self.coop_subgroups == 0 {
                (m, n / self.coop_subgroups, CoopPartition::Columns)
            } else {
                return Err(LowerError::UnsupportedOperation(
                    "cooperative matrix multi-subgroup tile shape mismatch",
                ));
            }
        } else {
            (m, n, CoopPartition::Single)
        };
        if m == 0
            || n == 0
            || m % COOP_MATRIX_DIM != 0
            || n % COOP_MATRIX_DIM != 0
            || subgroup_rows % COOP_MATRIX_DIM != 0
            || subgroup_cols % COOP_MATRIX_DIM != 0
            || (subgroup_rows / COOP_MATRIX_DIM) * (subgroup_cols / COOP_MATRIX_DIM)
                > scratch.coop_accs.len() as u32
            || m_dst != m
            || n_dst != n
            || k_a != k_b
            || k_a % COOP_MATRIX_DIM != 0
        {
            return Err(LowerError::UnsupportedOperation(
                "cooperative matrix lowering requires output tiles made of at most sixteen cooperative fragments and compatible K",
            ));
        }
        if outer_iterations == 0 {
            return Err(LowerError::UnsupportedOperation(
                "cooperative matrix gemm loop iteration count must be non-zero",
            ));
        }

        let tile_rows = subgroup_rows / COOP_MATRIX_DIM;
        let tile_cols = subgroup_cols / COOP_MATRIX_DIM;
        let fragment_count = (tile_rows * tile_cols) as usize;
        let acc_locals = scratch.coop_accs[..fragment_count]
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or(LowerError::UnsupportedOperation(
                "cooperative matrix accumulator locals were not allocated",
            ))?;

        let a_stride = Self::row_major_matrix_leading_stride(a_layout)?;
        let b_stride = Self::row_major_matrix_leading_stride(b_layout)?;
        let dst_stride = Self::row_major_matrix_leading_stride(dst_layout)?;
        let a_stride =
            expressions.append(Expression::Literal(Literal::U32(a_stride)), Span::default());
        let b_stride =
            expressions.append(Expression::Literal(Literal::U32(b_stride)), Span::default());
        let dst_stride = expressions.append(
            Expression::Literal(Literal::U32(dst_stride)),
            Span::default(),
        );

        let mut body = Block::new();
        let mut acc_pointers = Vec::with_capacity(acc_locals.len());
        for local in acc_locals {
            let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
            acc_pointers.push(pointer);
        }

        let k_chunks = k_a / COOP_MATRIX_DIM;
        let outer_unroll = COOP_MATRIX_OUTER_UNROLL.min(outer_iterations).max(1);
        let outer_unroll = (1..=outer_unroll)
            .rev()
            .find(|unroll| outer_iterations % unroll == 0)
            .unwrap_or(1);
        let (a_linear_base, a_linear_base_emits) = if PREFER_LINEAR_BASE_HOIST {
            self.storage_linear_base_without_loop_offsets(expressions, a)?
        } else {
            (None, Vec::new())
        };
        Self::push_emits(&mut body, a_linear_base_emits);
        let (b_linear_base, b_linear_base_emits) = if PREFER_LINEAR_BASE_HOIST {
            self.storage_linear_base_without_loop_offsets(expressions, b)?
        } else {
            (None, Vec::new())
        };
        Self::push_emits(&mut body, b_linear_base_emits);
        let (dst_linear_base, dst_linear_base_emits) = if PREFER_LINEAR_BASE_HOIST {
            self.storage_linear_base_without_loop_offsets(expressions, dst)?
        } else {
            (None, Vec::new())
        };
        Self::push_emits(&mut body, dst_linear_base_emits);
        let mut outer_body = Block::new();
        let (loop_index, loop_emit) = self.load_u32_local(expressions, scratch.loop_index);
        outer_body.push(Statement::Emit(loop_emit), Span::default());
        let mut loop_base_emits = Vec::new();
        let loop_k_base = self.mul_literal_u32_emitted(
            expressions,
            loop_index,
            k_a * outer_unroll,
            &mut loop_base_emits,
        );
        Self::push_emits(&mut outer_body, loop_base_emits);
        if k_chunks <= 4 {
            for outer_chunk in 0..outer_unroll {
                for k_chunk in 0..k_chunks {
                    let mut base_k_emits = Vec::new();
                    let base_k = self.add_literal_u32_emitted(
                        expressions,
                        loop_k_base,
                        outer_chunk * k_a + k_chunk * COOP_MATRIX_DIM,
                        &mut base_k_emits,
                    );
                    Self::push_emits(&mut outer_body, base_k_emits);
                    self.append_coop_k_chunk(
                        expressions,
                        &mut outer_body,
                        a,
                        b,
                        &acc_pointers,
                        a_linear_base,
                        b_linear_base,
                        tile_rows,
                        tile_cols,
                        base_k,
                        a_stride,
                        b_stride,
                        subgroup_cols,
                        subgroup_rows,
                        partition,
                        true,
                    )?;
                }
            }
        } else {
            let mut inner_body = Block::new();
            let (k_chunk, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
            inner_body.push(Statement::Emit(k_emit), Span::default());
            let mut base_k_emits = Vec::new();
            let inner_k = self.mul_literal_u32_emitted(
                expressions,
                k_chunk,
                COOP_MATRIX_DIM,
                &mut base_k_emits,
            );
            let base_k = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: loop_k_base,
                    right: inner_k,
                },
                Span::default(),
            );
            base_k_emits.push(Self::single_expression_range(expressions, base_k));
            Self::push_emits(&mut inner_body, base_k_emits);
            self.append_coop_k_chunk(
                expressions,
                &mut inner_body,
                a,
                b,
                &acc_pointers,
                a_linear_base,
                b_linear_base,
                tile_rows,
                tile_cols,
                base_k,
                a_stride,
                b_stride,
                subgroup_cols,
                subgroup_rows,
                partition,
                true,
            )?;
            outer_body.push(
                self.counted_loop(expressions, scratch.mma_k, k_chunks, inner_body),
                Span::default(),
            );
        }
        body.push(
            self.counted_loop(
                expressions,
                scratch.loop_index,
                outer_iterations / outer_unroll,
                outer_body,
            ),
            Span::default(),
        );

        let (subgroup_row_base, subgroup_col_base) = self.subgroup_partition_bases(
            expressions,
            &mut body,
            partition,
            subgroup_rows,
            subgroup_cols,
        );
        for row_tile in 0..tile_rows {
            for col_tile in 0..tile_cols {
                let acc_index = (row_tile * tile_cols + col_tile) as usize;
                let acc_value = expressions.append(
                    Expression::Load {
                        pointer: acc_pointers[acc_index],
                    },
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, acc_value)),
                    Span::default(),
                );
                let row = if let Some(subgroup_row_base) = subgroup_row_base {
                    let mut row_emits = Vec::new();
                    let row = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_row_base,
                        Self::coop_tile_offset(partition, true, row_tile),
                        &mut row_emits,
                    );
                    Self::push_emits(&mut body, row_emits);
                    row
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, true, row_tile,
                        ))),
                        Span::default(),
                    )
                };
                let col = if let Some(subgroup_col_base) = subgroup_col_base {
                    let mut col_emits = Vec::new();
                    let col = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_col_base,
                        Self::coop_tile_offset(partition, false, col_tile),
                        &mut col_emits,
                    );
                    Self::push_emits(&mut body, col_emits);
                    col
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, false, col_tile,
                        ))),
                        Span::default(),
                    )
                };
                let (dst_index, dst_index_emits) = if PREFER_LINEAR_BASE_HOIST {
                    let (dst_index, mut dst_index_emits) =
                        self.layout_index_expr(expressions, dst_layout, &[row, col])?;
                    let dst_index = self.add_optional_base_u32_emitted(
                        expressions,
                        dst_index,
                        dst_linear_base,
                        &mut dst_index_emits,
                    );
                    (dst_index, dst_index_emits)
                } else {
                    self.storage_index_from_coords(expressions, dst, &[row, col])?
                };
                Self::push_emits(&mut body, dst_index_emits);
                let (dst_pointer, dst_pointer_emits) =
                    self.storage_dynamic_pointer(expressions, dst, dst_index)?;
                Self::push_emits(&mut body, dst_pointer_emits);
                body.push(
                    Statement::CooperativeStore {
                        target: acc_value,
                        data: CooperativeData {
                            pointer: dst_pointer,
                            stride: dst_stride,
                            row_major: false,
                        },
                    },
                    Span::default(),
                );
            }
        }

        Ok(Statement::Block(body))
    }

    pub(super) fn append_shared_coop_k_chunk(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        a: TileRef,
        b: TileRef,
        acc_pointers: &[Handle<Expression>],
        tile_rows: u32,
        tile_cols: u32,
        base_k: Handle<Expression>,
        a_stride: Handle<Expression>,
        b_stride: Handle<Expression>,
        subgroup_cols: u32,
        subgroup_rows: u32,
        partition: CoopPartition,
    ) -> Result<(), LowerError> {
        let a_layout = self.tile_layout(a)?;
        let b_layout = self.tile_layout(b)?;
        let (subgroup_row_base, subgroup_col_base) = self.subgroup_partition_bases(
            expressions,
            body,
            partition,
            subgroup_rows,
            subgroup_cols,
        );

        let mut a_fragments = Vec::with_capacity(tile_rows as usize);
        for row_tile in 0..tile_rows {
            let row = if let Some(subgroup_row_base) = subgroup_row_base {
                let mut row_emits = Vec::new();
                let row = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_row_base,
                    Self::coop_tile_offset(partition, true, row_tile),
                    &mut row_emits,
                );
                Self::push_emits(body, row_emits);
                row
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, true, row_tile,
                    ))),
                    Span::default(),
                )
            };
            let (a_index, a_index_emits) =
                self.layout_index_expr(expressions, a_layout, &[row, base_k])?;
            Self::push_emits(body, a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.tile_dynamic_pointer(expressions, a, a_index)?;
            Self::push_emits(body, a_pointer_emits);
            let a_value = expressions.append(
                Expression::CooperativeLoad {
                    columns: CooperativeSize::Eight,
                    rows: CooperativeSize::Eight,
                    role: CooperativeRole::A,
                    data: CooperativeData {
                        pointer: a_pointer,
                        stride: a_stride,
                        row_major: false,
                    },
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            a_fragments.push(a_value);
        }

        let mut b_fragments = Vec::with_capacity(tile_cols as usize);
        for col_tile in 0..tile_cols {
            let col = if let Some(subgroup_col_base) = subgroup_col_base {
                let mut col_emits = Vec::new();
                let col = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_col_base,
                    Self::coop_tile_offset(partition, false, col_tile),
                    &mut col_emits,
                );
                Self::push_emits(body, col_emits);
                col
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, false, col_tile,
                    ))),
                    Span::default(),
                )
            };
            let (b_index, b_index_emits) =
                self.layout_index_expr(expressions, b_layout, &[base_k, col])?;
            Self::push_emits(body, b_index_emits);
            let (b_pointer, b_pointer_emits) =
                self.tile_dynamic_pointer(expressions, b, b_index)?;
            Self::push_emits(body, b_pointer_emits);
            let b_value = expressions.append(
                Expression::CooperativeLoad {
                    columns: CooperativeSize::Eight,
                    rows: CooperativeSize::Eight,
                    role: CooperativeRole::B,
                    data: CooperativeData {
                        pointer: b_pointer,
                        stride: b_stride,
                        row_major: false,
                    },
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, b_value)),
                Span::default(),
            );
            b_fragments.push(b_value);
        }

        for row_tile in 0..tile_rows {
            for col_tile in 0..tile_cols {
                let acc_index = (row_tile * tile_cols + col_tile) as usize;
                let acc_pointer = acc_pointers[acc_index];
                let acc_value = expressions.append(
                    Expression::Load {
                        pointer: acc_pointer,
                    },
                    Span::default(),
                );
                let next_acc = expressions.append(
                    Expression::CooperativeMultiplyAdd {
                        a: a_fragments[row_tile as usize],
                        b: b_fragments[col_tile as usize],
                        c: acc_value,
                    },
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, acc_value)),
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, next_acc)),
                    Span::default(),
                );
                body.push(
                    Statement::Store {
                        pointer: acc_pointer,
                        value: next_acc,
                    },
                    Span::default(),
                );
            }
        }

        Ok(())
    }

    pub(super) fn append_coop_k_chunk(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        a: &StorageView,
        b: &StorageView,
        acc_pointers: &[Handle<Expression>],
        a_linear_base: Option<Handle<Expression>>,
        b_linear_base: Option<Handle<Expression>>,
        tile_rows: u32,
        tile_cols: u32,
        base_k: Handle<Expression>,
        a_stride: Handle<Expression>,
        b_stride: Handle<Expression>,
        subgroup_cols: u32,
        subgroup_rows: u32,
        partition: CoopPartition,
        _skip_loop_offsets: bool,
    ) -> Result<(), LowerError> {
        let (subgroup_row_base, subgroup_col_base) = self.subgroup_partition_bases(
            expressions,
            body,
            partition,
            subgroup_rows,
            subgroup_cols,
        );
        let mut a_fragments = Vec::with_capacity(tile_rows as usize);
        for row_tile in 0..tile_rows {
            let row = if let Some(subgroup_row_base) = subgroup_row_base {
                let mut row_emits = Vec::new();
                let row = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_row_base,
                    Self::coop_tile_offset(partition, true, row_tile),
                    &mut row_emits,
                );
                Self::push_emits(body, row_emits);
                row
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, true, row_tile,
                    ))),
                    Span::default(),
                )
            };
            let (a_index, a_index_emits) = if PREFER_LINEAR_BASE_HOIST {
                let (a_index, mut a_index_emits) =
                    self.layout_index_expr(expressions, self.storage_layout(a)?, &[row, base_k])?;
                let a_index = self.add_optional_base_u32_emitted(
                    expressions,
                    a_index,
                    a_linear_base,
                    &mut a_index_emits,
                );
                (a_index, a_index_emits)
            } else {
                self.storage_index_from_coords_without_loop_offsets(expressions, a, &[row, base_k])?
            };
            Self::push_emits(body, a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.storage_dynamic_pointer(expressions, a, a_index)?;
            Self::push_emits(body, a_pointer_emits);
            let a_value = expressions.append(
                Expression::CooperativeLoad {
                    columns: COOP_MATRIX_SIZE,
                    rows: COOP_MATRIX_SIZE,
                    role: CooperativeRole::A,
                    data: CooperativeData {
                        pointer: a_pointer,
                        stride: a_stride,
                        row_major: false,
                    },
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_value)),
                Span::default(),
            );
            a_fragments.push(a_value);
        }

        let mut b_fragments = Vec::with_capacity(tile_cols as usize);
        for col_tile in 0..tile_cols {
            let col = if let Some(subgroup_col_base) = subgroup_col_base {
                let mut col_emits = Vec::new();
                let col = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_col_base,
                    Self::coop_tile_offset(partition, false, col_tile),
                    &mut col_emits,
                );
                Self::push_emits(body, col_emits);
                col
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, false, col_tile,
                    ))),
                    Span::default(),
                )
            };
            let (b_index, b_index_emits) = if PREFER_LINEAR_BASE_HOIST {
                let (b_index, mut b_index_emits) =
                    self.layout_index_expr(expressions, self.storage_layout(b)?, &[base_k, col])?;
                let b_index = self.add_optional_base_u32_emitted(
                    expressions,
                    b_index,
                    b_linear_base,
                    &mut b_index_emits,
                );
                (b_index, b_index_emits)
            } else {
                self.storage_index_from_coords_without_loop_offsets(expressions, b, &[base_k, col])?
            };
            Self::push_emits(body, b_index_emits);
            let (b_pointer, b_pointer_emits) =
                self.storage_dynamic_pointer(expressions, b, b_index)?;
            Self::push_emits(body, b_pointer_emits);
            let b_value = expressions.append(
                Expression::CooperativeLoad {
                    columns: COOP_MATRIX_SIZE,
                    rows: COOP_MATRIX_SIZE,
                    role: CooperativeRole::B,
                    data: CooperativeData {
                        pointer: b_pointer,
                        stride: b_stride,
                        row_major: false,
                    },
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, b_value)),
                Span::default(),
            );
            b_fragments.push(b_value);
        }

        for row_tile in 0..tile_rows {
            for col_tile in 0..tile_cols {
                let acc_index = (row_tile * tile_cols + col_tile) as usize;
                let acc_pointer = acc_pointers[acc_index];
                let acc_value = expressions.append(
                    Expression::Load {
                        pointer: acc_pointer,
                    },
                    Span::default(),
                );
                let next_acc = expressions.append(
                    Expression::CooperativeMultiplyAdd {
                        a: a_fragments[row_tile as usize],
                        b: b_fragments[col_tile as usize],
                        c: acc_value,
                    },
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, acc_value)),
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, next_acc)),
                    Span::default(),
                );
                body.push(
                    Statement::Store {
                        pointer: acc_pointer,
                        value: next_acc,
                    },
                    Span::default(),
                );
            }
        }

        Ok(())
    }
}
