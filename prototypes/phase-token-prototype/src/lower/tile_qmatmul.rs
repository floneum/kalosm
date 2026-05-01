use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_tile_program_accelerator(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        accelerator: &TileProgramAccelerator,
    ) -> Result<Statement, LowerError> {
        match accelerator {
            TileProgramAccelerator::QMatmul(op) => {
                self.lower_tile_qmatmul_coop(expressions, scratch, op)
            }
            TileProgramAccelerator::QGemv(op) => {
                self.lower_tile_qgemv_subgroup(expressions, scratch, op)
            }
        }
    }

    fn lower_tile_qgemv_subgroup(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &TileQGemvProgramOp,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(&op.a)?;
        let y_layout = self.storage_layout(&op.y)?;
        let [m, k_size] = Self::matrix_shape(a_layout)?;
        let [y_m, n] = Self::matrix_shape(y_layout)?;
        if m != 1 || y_m != 1 || k_size != op.b.rows || n != op.b.cols {
            return Err(LowerError::UnsupportedOperation("qgemv shape mismatch"));
        }
        if !matches!(
            op.b.format,
            GgmlQuantFormat::Q4K | GgmlQuantFormat::Q5_0 | GgmlQuantFormat::Q8_0
        ) {
            return Err(LowerError::UnsupportedOperation(
                "subgroup qgemv currently supports Q4K, Q5_0, and Q8_0",
            ));
        }

        let values_per_lane = match op.b.format {
            GgmlQuantFormat::Q5_0 => 16,
            _ => 8,
        };
        let output_cols_per_subgroup = op.b.format.qgemv_cols_per_subgroup();
        let mut body = Block::new();
        let wg = expressions.append(
            Expression::FunctionArgument(WORKGROUP_ID_ARG),
            Span::default(),
        );
        let wg_x = expressions.append(
            Expression::AccessIndex { base: wg, index: 0 },
            Span::default(),
        );
        let wg_y = expressions.append(
            Expression::AccessIndex { base: wg, index: 1 },
            Span::default(),
        );
        let subgroup_id = expressions.append(
            Expression::FunctionArgument(SUBGROUP_ID_ARG),
            Span::default(),
        );
        let subgroup_lane = expressions.append(
            Expression::FunctionArgument(SUBGROUP_INVOCATION_ID_ARG),
            Span::default(),
        );
        let subgroup_size = expressions.append(
            Expression::FunctionArgument(SUBGROUP_SIZE_ARG),
            Span::default(),
        );
        let num_subgroups = expressions.append(
            Expression::FunctionArgument(NUM_SUBGROUPS_ARG),
            Span::default(),
        );

        let mut emits = Vec::new();
        let wg_y_offset =
            self.mul_literal_u32_emitted(expressions, wg_y, op.workgroups_x, &mut emits);
        let n_group = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            wg_x,
            wg_y_offset,
        );
        let cols_per_workgroup = self.mul_literal_u32_emitted(
            expressions,
            num_subgroups,
            output_cols_per_subgroup,
            &mut emits,
        );
        let col_group_base = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Multiply,
            n_group,
            cols_per_workgroup,
        );
        let subgroup_col_base = self.mul_literal_u32_emitted(
            expressions,
            subgroup_id,
            output_cols_per_subgroup,
            &mut emits,
        );
        let col0 = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_group_base,
            subgroup_col_base,
        );
        Self::push_emits(&mut body, emits);

        let mut cols = Vec::with_capacity(output_cols_per_subgroup as usize);
        let mut col_in_bounds = Vec::with_capacity(output_cols_per_subgroup as usize);
        for offset in 0..output_cols_per_subgroup {
            let col = if offset == 0 {
                col0
            } else {
                self.add_literal_u32(expressions, col0, offset)
            };
            if offset != 0 {
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, col)),
                    Span::default(),
                );
            }
            let in_bounds = self.bin_lit_u32(expressions, &mut body, BinaryOperator::Less, col, n);
            cols.push(col);
            col_in_bounds.push(in_bounds);
        }

        let sum_locals = &scratch.spills[0][..output_cols_per_subgroup as usize];
        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        for sum in sum_locals {
            let pointer = expressions.append(Expression::LocalVariable(*sum), Span::default());
            body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
        }

        let k_ptr = expressions.append(
            Expression::LocalVariable(scratch.loop_index),
            Span::default(),
        );
        let mut initial_k_emits = Vec::new();
        let initial_k = self.mul_literal_u32_emitted(
            expressions,
            subgroup_lane,
            values_per_lane,
            &mut initial_k_emits,
        );
        Self::push_emits(&mut body, initial_k_emits);
        body.push(
            Statement::Store {
                pointer: k_ptr,
                value: initial_k,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let (k, k_emit) = self.load_u32_local(expressions, scratch.loop_index);
        loop_body.push(Statement::Emit(k_emit), Span::default());
        let k_done = self.bin_lit_u32(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            k,
            k_size,
        );
        loop_body.push(
            Statement::If {
                condition: k_done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let row_zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        let mut a_scalars = Vec::with_capacity(values_per_lane as usize);
        for lane in 0..values_per_lane {
            let mut a_emits = Vec::new();
            let a_k = self.add_literal_u32_emitted(expressions, k, lane, &mut a_emits);
            let (a_index, a_index_emits) =
                self.storage_index_from_coords(expressions, &op.a, &[row_zero, a_k])?;
            let (a_ptr, a_ptr_emits) = self.storage_dynamic_pointer(expressions, &op.a, a_index)?;
            a_emits.extend(a_index_emits);
            a_emits.extend(a_ptr_emits);
            Self::push_emits(&mut loop_body, a_emits);
            let a = expressions.append(Expression::Load { pointer: a_ptr }, Span::default());
            loop_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a)),
                Span::default(),
            );
            a_scalars.push(a);
        }

        let chunk_count = (values_per_lane / 4) as usize;
        let mut a_vecs = Vec::with_capacity(chunk_count);
        for chunk in 0..chunk_count {
            let a_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components: a_scalars[(chunk * 4)..(chunk * 4 + 4)].to_vec(),
                },
                Span::default(),
            );
            loop_body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_vec)),
                Span::default(),
            );
            a_vecs.push(a_vec);
        }

        for ((col, in_bounds), sum_local) in cols
            .iter()
            .copied()
            .zip(col_in_bounds.iter().copied())
            .zip(sum_locals.iter().copied())
        {
            let mut fma_body = Block::new();
            let (b_values, b_emits) = match op.b.format {
                GgmlQuantFormat::Q4K => self.dequantize_q4k_values8(expressions, &op.b, k, col)?,
                GgmlQuantFormat::Q5_0 => {
                    self.dequantize_q5_0_values16(expressions, &op.b, k, col)?
                }
                GgmlQuantFormat::Q8_0 => {
                    self.dequantize_q8_0_values8(expressions, &op.b, k, col)?
                }
                _ => unreachable!(),
            };
            Self::push_emits(&mut fma_body, b_emits);

            let mut dots = Vec::with_capacity(chunk_count);
            for (chunk, a_vec) in a_vecs.iter().copied().enumerate() {
                let b_vec = expressions.append(
                    Expression::Compose {
                        ty: self.f32_vec4_ty,
                        components: b_values[(chunk * 4)..(chunk * 4 + 4)].to_vec(),
                    },
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
                fma_body.push(
                    Statement::Emit(Self::range_from(expressions, b_vec, dot)),
                    Span::default(),
                );
                dots.push(dot);
            }
            let mut dot = dots[0];
            let mut dot_emit_start = None;
            for next in dots.into_iter().skip(1) {
                dot = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Add,
                        left: dot,
                        right: next,
                    },
                    Span::default(),
                );
                dot_emit_start.get_or_insert(dot);
            }
            if let Some(start) = dot_emit_start {
                fma_body.push(
                    Statement::Emit(Self::range_from(expressions, start, dot)),
                    Span::default(),
                );
            } else {
                fma_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, dot)),
                    Span::default(),
                );
            }

            let sum_ptr = expressions.append(Expression::LocalVariable(sum_local), Span::default());
            let sum = expressions.append(Expression::Load { pointer: sum_ptr }, Span::default());
            let value = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: sum,
                    right: dot,
                },
                Span::default(),
            );
            fma_body.push(
                Statement::Emit(Self::range_from(expressions, sum, value)),
                Span::default(),
            );
            fma_body.push(
                Statement::Store {
                    pointer: sum_ptr,
                    value,
                },
                Span::default(),
            );
            loop_body.push(
                Statement::If {
                    condition: in_bounds,
                    accept: fma_body,
                    reject: Block::new(),
                },
                Span::default(),
            );
        }

        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::from_vec(vec![self.increment_u32_local_by_expr(
                    expressions,
                    scratch.loop_index,
                    subgroup_size,
                    values_per_lane,
                )]),
                break_if: None,
            },
            Span::default(),
        );

        let lane_is_first = self.bin_lit_u32(
            expressions,
            &mut body,
            BinaryOperator::Equal,
            subgroup_lane,
            0,
        );
        for ((col, in_bounds), sum_local) in cols
            .into_iter()
            .zip(col_in_bounds.into_iter())
            .zip(sum_locals.iter().copied())
        {
            let sum_ptr = expressions.append(Expression::LocalVariable(sum_local), Span::default());
            let sum = expressions.append(Expression::Load { pointer: sum_ptr }, Span::default());
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum)),
                Span::default(),
            );
            let reduced = expressions.append(
                Expression::SubgroupOperationResult { ty: self.f32_ty },
                Span::default(),
            );
            body.push(
                Statement::SubgroupCollectiveOperation {
                    op: SubgroupOperation::Add,
                    collective_op: CollectiveOperation::Reduce,
                    argument: sum,
                    result: reduced,
                },
                Span::default(),
            );
            let should_store = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::LogicalAnd,
                    left: in_bounds,
                    right: lane_is_first,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, should_store)),
                Span::default(),
            );
            let (y_index, y_emits) =
                self.storage_index_from_coords(expressions, &op.y, &[row_zero, col])?;
            let (y_ptr, y_ptr_emits) = self.storage_dynamic_pointer(expressions, &op.y, y_index)?;
            Self::push_emits(&mut body, y_emits);
            Self::push_emits(&mut body, y_ptr_emits);
            body.push(
                Statement::If {
                    condition: should_store,
                    accept: Block::from_vec(vec![Statement::Store {
                        pointer: y_ptr,
                        value: reduced,
                    }]),
                    reject: Block::new(),
                },
                Span::default(),
            );
        }

        Ok(Statement::Block(body))
    }

    fn lower_tile_qmatmul_coop(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &TileQMatmulProgramOp,
    ) -> Result<Statement, LowerError> {
        let _coop_a_ty = self.coop_f32_a_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix A type was not allocated",
        ))?;
        let _coop_b_ty = self.coop_f32_b_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix B type was not allocated",
        ))?;
        let coop_c_ty = self.coop_f32_c_ty.ok_or(LowerError::UnsupportedOperation(
            "cooperative matrix C type was not allocated",
        ))?;

        let a_layout = self.storage_layout(&op.a)?;
        let y_layout = self.storage_layout(&op.y)?;
        let a_tile_layout = self.tile_layout(op.a_tile)?;
        let b_tile_layout = self.tile_layout(op.b_tile)?;
        let [m, k_size] = Self::matrix_shape(a_layout)?;
        let [y_m, n] = Self::matrix_shape(y_layout)?;
        let [a_tile_m, a_tile_k] = Self::matrix_shape(a_tile_layout)?;
        let [b_tile_k, b_tile_n] = Self::matrix_shape(b_tile_layout)?;
        if m != y_m
            || k_size != op.b.rows
            || n != op.b.cols
            || a_tile_m != op.tile_m
            || a_tile_k != op.tile_k
            || b_tile_k != op.tile_k
            || b_tile_n != op.tile_n
            || a_tile_layout.memory_level() != MemoryLevel::Workgroup
            || b_tile_layout.memory_level() != MemoryLevel::Workgroup
            || !Self::is_row_major_storage_matrix(a_tile_layout)
            || !Self::is_row_major_storage_matrix(b_tile_layout)
            || !m.is_multiple_of(op.tile_m)
            || !n.is_multiple_of(op.tile_n)
            || !k_size.is_multiple_of(op.tile_k)
            || !op.tile_m.is_multiple_of(COOP_MATRIX_DIM)
            || !op.tile_n.is_multiple_of(COOP_MATRIX_DIM)
            || !op.tile_k.is_multiple_of(COOP_MATRIX_DIM)
        {
            return Err(LowerError::UnsupportedOperation(
                "cooperative qmatmul requires exact row-major workgroup tiles and divisible matrix dimensions",
            ));
        }

        let (subgroup_rows, subgroup_cols, partition) =
            self.qmatmul_coop_partition(op)
                .ok_or(LowerError::UnsupportedOperation(
                    "cooperative qmatmul tile partition mismatch",
                ))?;
        let tile_rows = subgroup_rows / COOP_MATRIX_DIM;
        let tile_cols = subgroup_cols / COOP_MATRIX_DIM;
        let fragment_count = (tile_rows * tile_cols) as usize;
        if fragment_count == 0 || fragment_count > scratch.coop_accs.len() {
            return Err(LowerError::UnsupportedOperation(
                "cooperative qmatmul fragment count exceeds available accumulator locals",
            ));
        }
        let acc_locals = scratch.coop_accs[..fragment_count]
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or(LowerError::UnsupportedOperation(
                "cooperative qmatmul accumulator locals were not allocated",
            ))?;

        let a_stride = expressions.append(
            Expression::Literal(Literal::U32(Self::row_major_matrix_leading_stride(
                a_tile_layout,
            )?)),
            Span::default(),
        );
        let b_stride = expressions.append(
            Expression::Literal(Literal::U32(Self::row_major_matrix_leading_stride(
                b_tile_layout,
            )?)),
            Span::default(),
        );
        let (y_leading_stride, y_row_major) = Self::cooperative_matrix_store_layout(y_layout)?;
        let y_stride = expressions.append(
            Expression::Literal(Literal::U32(y_leading_stride)),
            Span::default(),
        );

        let mut body = Block::new();
        let local = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let wg = expressions.append(
            Expression::FunctionArgument(WORKGROUP_ID_ARG),
            Span::default(),
        );
        let tile_col = expressions.append(
            Expression::AccessIndex { base: wg, index: 0 },
            Span::default(),
        );
        let tile_row = expressions.append(
            Expression::AccessIndex { base: wg, index: 1 },
            Span::default(),
        );
        let mut tile_emits = Vec::new();
        let row_base =
            self.mul_literal_u32_emitted(expressions, tile_row, op.tile_m, &mut tile_emits);
        let col_base =
            self.mul_literal_u32_emitted(expressions, tile_col, op.tile_n, &mut tile_emits);
        Self::push_emits(&mut body, tile_emits);

        let mut acc_pointers = Vec::with_capacity(acc_locals.len());
        let zero_acc = expressions.append(Expression::ZeroValue(coop_c_ty), Span::default());
        for local in acc_locals {
            let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
            body.push(
                Statement::Store {
                    pointer,
                    value: zero_acc,
                },
                Span::default(),
            );
            acc_pointers.push(pointer);
        }

        let k_ptr = expressions.append(
            Expression::LocalVariable(scratch.loop_index),
            Span::default(),
        );
        let zero_u = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        body.push(
            Statement::Store {
                pointer: k_ptr,
                value: zero_u,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let (k_start, k_emit) = self.load_u32_local(expressions, scratch.loop_index);
        loop_body.push(Statement::Emit(k_emit), Span::default());
        let k_done = self.bin_lit_u32(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            k_start,
            k_size,
        );
        loop_body.push(
            Statement::If {
                condition: k_done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        self.append_qmatmul_b_tile_loads(
            expressions,
            &mut loop_body,
            op,
            local,
            col_base,
            k_start,
        )?;
        self.append_qmatmul_a_tile_loads(
            expressions,
            &mut loop_body,
            op,
            local,
            row_base,
            k_start,
        )?;
        loop_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        for kk in (0..op.tile_k).step_by(COOP_MATRIX_DIM as usize) {
            let base_k = expressions.append(Expression::Literal(Literal::U32(kk)), Span::default());
            self.append_shared_coop_k_chunk(
                expressions,
                &mut loop_body,
                op.a_tile,
                op.b_tile,
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
        }
        loop_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::from_vec(vec![self.increment_u32_local(
                    expressions,
                    scratch.loop_index,
                    op.tile_k,
                )]),
                break_if: None,
            },
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

                let local_row = if let Some(subgroup_row_base) = subgroup_row_base {
                    let mut emits = Vec::new();
                    let row = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_row_base,
                        Self::coop_tile_offset(partition, true, row_tile),
                        &mut emits,
                    );
                    Self::push_emits(&mut store_body, emits);
                    row
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, true, row_tile,
                        ))),
                        Span::default(),
                    )
                };
                let local_col = if let Some(subgroup_col_base) = subgroup_col_base {
                    let mut emits = Vec::new();
                    let col = self.add_literal_u32_emitted(
                        expressions,
                        subgroup_col_base,
                        Self::coop_tile_offset(partition, false, col_tile),
                        &mut emits,
                    );
                    Self::push_emits(&mut store_body, emits);
                    col
                } else {
                    expressions.append(
                        Expression::Literal(Literal::U32(Self::coop_tile_offset(
                            partition, false, col_tile,
                        ))),
                        Span::default(),
                    )
                };

                let mut output_emits = Vec::new();
                let row = self.bin(
                    expressions,
                    &mut output_emits,
                    BinaryOperator::Add,
                    row_base,
                    local_row,
                );
                let col = self.bin(
                    expressions,
                    &mut output_emits,
                    BinaryOperator::Add,
                    col_base,
                    local_col,
                );
                Self::push_emits(&mut store_body, output_emits);
                let (y_index, y_emits) =
                    self.storage_index_from_coords(expressions, &op.y, &[row, col])?;
                Self::push_emits(&mut store_body, y_emits);
                let (y_ptr, y_ptr_emits) =
                    self.storage_dynamic_pointer(expressions, &op.y, y_index)?;
                Self::push_emits(&mut store_body, y_ptr_emits);
                store_body.push(
                    Statement::CooperativeStore {
                        target: acc_value,
                        data: CooperativeData {
                            pointer: y_ptr,
                            stride: y_stride,
                            row_major: y_row_major,
                        },
                    },
                    Span::default(),
                );
            }
        }
        body.push(Statement::Block(store_body), Span::default());
        Ok(Statement::Block(body))
    }

    fn append_qmatmul_a_tile_loads(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: &TileQMatmulProgramOp,
        local: Handle<Expression>,
        row_base: Handle<Expression>,
        k_start: Handle<Expression>,
    ) -> Result<(), LowerError> {
        const VALUES_PER_LOAD: u32 = 8;
        let groups_per_row = op.tile_k / VALUES_PER_LOAD;
        let total = op.tile_m * groups_per_row;
        let passes = total.div_ceil(self.workgroup_invocations);
        let tile_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.a_tile)?)?;
        for pass in 0..passes {
            let mut condition_emits = Vec::new();
            let flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut condition_emits,
            );
            let lane_active = self.cmp_lit(
                expressions,
                &mut condition_emits,
                BinaryOperator::Less,
                flat,
                total,
            );
            let mut emits = Vec::new();
            let local_row =
                self.div_literal_u32_emitted(expressions, flat, groups_per_row, &mut emits);
            let local_k_group =
                self.mod_literal_u32_emitted(expressions, flat, groups_per_row, &mut emits);
            let local_k_base = self.mul_literal_u32_emitted(
                expressions,
                local_k_group,
                VALUES_PER_LOAD,
                &mut emits,
            );
            let row = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                row_base,
                local_row,
            );
            let k = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                k_start,
                local_k_base,
            );
            let mut a_tile_ptrs = Vec::with_capacity(VALUES_PER_LOAD as usize);
            for lane in 0..VALUES_PER_LOAD {
                let local_k_lane =
                    self.add_literal_u32_emitted(expressions, local_k_base, lane, &mut emits);
                let a_tile_index = self.tile_matrix_index(
                    expressions,
                    &mut emits,
                    local_row,
                    local_k_lane,
                    tile_stride,
                );
                let (a_tile_ptr, a_tile_ptr_emits) =
                    self.tile_dynamic_pointer(expressions, op.a_tile, a_tile_index)?;
                emits.extend(a_tile_ptr_emits);
                a_tile_ptrs.push(a_tile_ptr);
            }

            let mut lane_body = Block::new();
            if (pass + 1) * self.workgroup_invocations <= total {
                condition_emits.extend(emits);
                Self::push_emits(&mut lane_body, condition_emits);
            } else {
                Self::push_emits(body, condition_emits);
                Self::push_emits(&mut lane_body, emits);
            }
            for (lane, a_tile_ptr) in a_tile_ptrs.into_iter().enumerate() {
                let mut lane_emits = Vec::new();
                let k_lane =
                    self.add_literal_u32_emitted(expressions, k, lane as u32, &mut lane_emits);
                let (a_index, a_index_emits) =
                    self.storage_index_from_coords(expressions, &op.a, &[row, k_lane])?;
                let (a_ptr, a_ptr_emits) =
                    self.storage_dynamic_pointer(expressions, &op.a, a_index)?;
                lane_emits.extend(a_index_emits);
                lane_emits.extend(a_ptr_emits);
                Self::push_emits(&mut lane_body, lane_emits);
                let a_value =
                    expressions.append(Expression::Load { pointer: a_ptr }, Span::default());
                lane_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, a_value)),
                    Span::default(),
                );
                lane_body.push(
                    Statement::Store {
                        pointer: a_tile_ptr,
                        value: a_value,
                    },
                    Span::default(),
                );
            }
            if (pass + 1) * self.workgroup_invocations <= total {
                body.push(Statement::Block(lane_body), Span::default());
            } else {
                body.push(
                    Statement::If {
                        condition: lane_active,
                        accept: lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            }
        }
        Ok(())
    }

    fn append_qmatmul_b_tile_loads(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: &TileQMatmulProgramOp,
        local: Handle<Expression>,
        col_base: Handle<Expression>,
        k_start: Handle<Expression>,
    ) -> Result<(), LowerError> {
        if op.b.format == GgmlQuantFormat::Q8_0 {
            return self.append_qmatmul_b_tile_loads_q8_0x8(
                expressions,
                body,
                op,
                local,
                col_base,
                k_start,
            );
        }
        if op.b.format == GgmlQuantFormat::Q5_0 {
            return self.append_qmatmul_b_tile_loads_q5_0x16(
                expressions,
                body,
                op,
                local,
                col_base,
                k_start,
            );
        }
        if matches!(op.b.format, GgmlQuantFormat::Q4K | GgmlQuantFormat::Q6K) {
            return self.append_qmatmul_b_tile_loads_values8(
                expressions,
                body,
                op,
                local,
                col_base,
                k_start,
            );
        }

        let total = op.tile_k * op.tile_n;
        let passes = total.div_ceil(self.workgroup_invocations);
        let tile_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.b_tile)?)?;
        for pass in 0..passes {
            let mut condition_emits = Vec::new();
            let flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut condition_emits,
            );
            let lane_active = self.cmp_lit(
                expressions,
                &mut condition_emits,
                BinaryOperator::Less,
                flat,
                total,
            );
            let mut emits = Vec::new();
            let local_k = self.div_literal_u32_emitted(expressions, flat, op.tile_n, &mut emits);
            let local_col = self.mod_literal_u32_emitted(expressions, flat, op.tile_n, &mut emits);
            let k = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                k_start,
                local_k,
            );
            let col = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                col_base,
                local_col,
            );
            let b_tile_index =
                self.tile_matrix_index(expressions, &mut emits, local_k, local_col, tile_stride);
            let (b_tile_ptr, b_tile_ptr_emits) =
                self.tile_dynamic_pointer(expressions, op.b_tile, b_tile_index)?;
            emits.extend(b_tile_ptr_emits);

            let mut lane_body = Block::new();
            if (pass + 1) * self.workgroup_invocations <= total {
                condition_emits.extend(emits);
                Self::push_emits(&mut lane_body, condition_emits);
            } else {
                Self::push_emits(body, condition_emits);
                Self::push_emits(&mut lane_body, emits);
            }
            let (b_value, b_emits) = self.dequantize_qvalue(expressions, &op.b, k, col)?;
            Self::push_emits(&mut lane_body, b_emits);
            lane_body.push(
                Statement::Store {
                    pointer: b_tile_ptr,
                    value: b_value,
                },
                Span::default(),
            );
            if (pass + 1) * self.workgroup_invocations <= total {
                body.push(Statement::Block(lane_body), Span::default());
            } else {
                body.push(
                    Statement::If {
                        condition: lane_active,
                        accept: lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            }
        }
        Ok(())
    }

    fn append_qmatmul_b_tile_loads_q8_0x8(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: &TileQMatmulProgramOp,
        local: Handle<Expression>,
        col_base: Handle<Expression>,
        k_start: Handle<Expression>,
    ) -> Result<(), LowerError> {
        const VALUES_PER_LOAD: u32 = 8;
        let groups_per_col = op.tile_k / VALUES_PER_LOAD;
        let total = groups_per_col * op.tile_n;
        let passes = total.div_ceil(self.workgroup_invocations);
        let tile_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.b_tile)?)?;
        for pass in 0..passes {
            let mut condition_emits = Vec::new();
            let flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut condition_emits,
            );
            let lane_active = self.cmp_lit(
                expressions,
                &mut condition_emits,
                BinaryOperator::Less,
                flat,
                total,
            );
            let mut emits = Vec::new();
            let local_k_group =
                self.div_literal_u32_emitted(expressions, flat, op.tile_n, &mut emits);
            let local_col = self.mod_literal_u32_emitted(expressions, flat, op.tile_n, &mut emits);
            let local_k_base = self.mul_literal_u32_emitted(
                expressions,
                local_k_group,
                VALUES_PER_LOAD,
                &mut emits,
            );
            let k_base = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                k_start,
                local_k_base,
            );
            let col = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                col_base,
                local_col,
            );
            let mut b_tile_ptrs = Vec::with_capacity(VALUES_PER_LOAD as usize);
            for lane in 0..VALUES_PER_LOAD {
                let local_k =
                    self.add_literal_u32_emitted(expressions, local_k_base, lane, &mut emits);
                let b_tile_index = self.tile_matrix_index(
                    expressions,
                    &mut emits,
                    local_k,
                    local_col,
                    tile_stride,
                );
                let (b_tile_ptr, b_tile_ptr_emits) =
                    self.tile_dynamic_pointer(expressions, op.b_tile, b_tile_index)?;
                emits.extend(b_tile_ptr_emits);
                b_tile_ptrs.push(b_tile_ptr);
            }

            let mut lane_body = Block::new();
            if (pass + 1) * self.workgroup_invocations <= total {
                condition_emits.extend(emits);
                Self::push_emits(&mut lane_body, condition_emits);
            } else {
                Self::push_emits(body, condition_emits);
                Self::push_emits(&mut lane_body, emits);
            }
            let (values, value_emits) =
                self.dequantize_q8_0_values8(expressions, &op.b, k_base, col)?;
            Self::push_emits(&mut lane_body, value_emits);
            for (pointer, value) in b_tile_ptrs.into_iter().zip(values) {
                lane_body.push(Statement::Store { pointer, value }, Span::default());
            }
            if (pass + 1) * self.workgroup_invocations <= total {
                body.push(Statement::Block(lane_body), Span::default());
            } else {
                body.push(
                    Statement::If {
                        condition: lane_active,
                        accept: lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            }
        }
        Ok(())
    }

    fn append_qmatmul_b_tile_loads_q5_0x16(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: &TileQMatmulProgramOp,
        local: Handle<Expression>,
        col_base: Handle<Expression>,
        k_start: Handle<Expression>,
    ) -> Result<(), LowerError> {
        const VALUES_PER_LOAD: u32 = 16;
        let groups_per_col = op.tile_k / VALUES_PER_LOAD;
        let total = groups_per_col * op.tile_n;
        let passes = total.div_ceil(self.workgroup_invocations);
        let tile_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.b_tile)?)?;
        for pass in 0..passes {
            let mut condition_emits = Vec::new();
            let flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut condition_emits,
            );
            let lane_active = self.cmp_lit(
                expressions,
                &mut condition_emits,
                BinaryOperator::Less,
                flat,
                total,
            );
            let mut emits = Vec::new();
            let local_k_group =
                self.div_literal_u32_emitted(expressions, flat, op.tile_n, &mut emits);
            let local_col = self.mod_literal_u32_emitted(expressions, flat, op.tile_n, &mut emits);
            let local_k_base = self.mul_literal_u32_emitted(
                expressions,
                local_k_group,
                VALUES_PER_LOAD,
                &mut emits,
            );
            let k_base = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                k_start,
                local_k_base,
            );
            let col = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                col_base,
                local_col,
            );
            let mut b_tile_ptrs = Vec::with_capacity(VALUES_PER_LOAD as usize);
            for lane in 0..VALUES_PER_LOAD {
                let local_k =
                    self.add_literal_u32_emitted(expressions, local_k_base, lane, &mut emits);
                let b_tile_index = self.tile_matrix_index(
                    expressions,
                    &mut emits,
                    local_k,
                    local_col,
                    tile_stride,
                );
                let (b_tile_ptr, b_tile_ptr_emits) =
                    self.tile_dynamic_pointer(expressions, op.b_tile, b_tile_index)?;
                emits.extend(b_tile_ptr_emits);
                b_tile_ptrs.push(b_tile_ptr);
            }

            let mut lane_body = Block::new();
            if (pass + 1) * self.workgroup_invocations <= total {
                condition_emits.extend(emits);
                Self::push_emits(&mut lane_body, condition_emits);
            } else {
                Self::push_emits(body, condition_emits);
                Self::push_emits(&mut lane_body, emits);
            }
            let (values, value_emits) =
                self.dequantize_q5_0_values16(expressions, &op.b, k_base, col)?;
            Self::push_emits(&mut lane_body, value_emits);
            for (pointer, value) in b_tile_ptrs.into_iter().zip(values) {
                lane_body.push(Statement::Store { pointer, value }, Span::default());
            }
            if (pass + 1) * self.workgroup_invocations <= total {
                body.push(Statement::Block(lane_body), Span::default());
            } else {
                body.push(
                    Statement::If {
                        condition: lane_active,
                        accept: lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            }
        }
        Ok(())
    }

    fn append_qmatmul_b_tile_loads_values8(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: &TileQMatmulProgramOp,
        local: Handle<Expression>,
        col_base: Handle<Expression>,
        k_start: Handle<Expression>,
    ) -> Result<(), LowerError> {
        const VALUES_PER_LOAD: u32 = 8;
        let groups_per_col = op.tile_k / VALUES_PER_LOAD;
        let total = groups_per_col * op.tile_n;
        let passes = total.div_ceil(self.workgroup_invocations);
        let tile_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.b_tile)?)?;
        for pass in 0..passes {
            let mut condition_emits = Vec::new();
            let flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut condition_emits,
            );
            let lane_active = self.cmp_lit(
                expressions,
                &mut condition_emits,
                BinaryOperator::Less,
                flat,
                total,
            );
            let mut emits = Vec::new();
            let local_k_group =
                self.div_literal_u32_emitted(expressions, flat, op.tile_n, &mut emits);
            let local_col = self.mod_literal_u32_emitted(expressions, flat, op.tile_n, &mut emits);
            let local_k_base = self.mul_literal_u32_emitted(
                expressions,
                local_k_group,
                VALUES_PER_LOAD,
                &mut emits,
            );
            let k_base = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                k_start,
                local_k_base,
            );
            let col = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                col_base,
                local_col,
            );
            let mut b_tile_ptrs = Vec::with_capacity(VALUES_PER_LOAD as usize);
            for lane in 0..VALUES_PER_LOAD {
                let local_k =
                    self.add_literal_u32_emitted(expressions, local_k_base, lane, &mut emits);
                let b_tile_index = self.tile_matrix_index(
                    expressions,
                    &mut emits,
                    local_k,
                    local_col,
                    tile_stride,
                );
                let (b_tile_ptr, b_tile_ptr_emits) =
                    self.tile_dynamic_pointer(expressions, op.b_tile, b_tile_index)?;
                emits.extend(b_tile_ptr_emits);
                b_tile_ptrs.push(b_tile_ptr);
            }

            let mut lane_body = Block::new();
            if (pass + 1) * self.workgroup_invocations <= total {
                condition_emits.extend(emits);
                Self::push_emits(&mut lane_body, condition_emits);
            } else {
                Self::push_emits(body, condition_emits);
                Self::push_emits(&mut lane_body, emits);
            }
            let (values, value_emits) = match op.b.format {
                GgmlQuantFormat::Q4K => {
                    self.dequantize_q4k_values8(expressions, &op.b, k_base, col)?
                }
                GgmlQuantFormat::Q6K => {
                    self.dequantize_q6k_values8(expressions, &op.b, k_base, col)?
                }
                _ => unreachable!(),
            };
            Self::push_emits(&mut lane_body, value_emits);
            for (pointer, value) in b_tile_ptrs.into_iter().zip(values) {
                lane_body.push(Statement::Store { pointer, value }, Span::default());
            }
            if (pass + 1) * self.workgroup_invocations <= total {
                body.push(Statement::Block(lane_body), Span::default());
            } else {
                body.push(
                    Statement::If {
                        condition: lane_active,
                        accept: lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn append_shared_coop_k_chunk(
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
        let a_stride_u32 = Self::row_major_matrix_leading_stride(a_layout)?;
        let b_stride_u32 = Self::row_major_matrix_leading_stride(b_layout)?;
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
                let mut emits = Vec::new();
                let row = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_row_base,
                    Self::coop_tile_offset(partition, true, row_tile),
                    &mut emits,
                );
                Self::push_emits(body, emits);
                row
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, true, row_tile,
                    ))),
                    Span::default(),
                )
            };
            let mut a_index_emits = Vec::new();
            let a_index =
                self.tile_matrix_index(expressions, &mut a_index_emits, row, base_k, a_stride_u32);
            Self::push_emits(body, a_index_emits);
            let (a_pointer, a_pointer_emits) =
                self.tile_dynamic_pointer(expressions, a, a_index)?;
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
                let mut emits = Vec::new();
                let col = self.add_literal_u32_emitted(
                    expressions,
                    subgroup_col_base,
                    Self::coop_tile_offset(partition, false, col_tile),
                    &mut emits,
                );
                Self::push_emits(body, emits);
                col
            } else {
                expressions.append(
                    Expression::Literal(Literal::U32(Self::coop_tile_offset(
                        partition, false, col_tile,
                    ))),
                    Span::default(),
                )
            };
            let mut b_index_emits = Vec::new();
            let b_index =
                self.tile_matrix_index(expressions, &mut b_index_emits, base_k, col, b_stride_u32);
            Self::push_emits(body, b_index_emits);
            let (b_pointer, b_pointer_emits) =
                self.tile_dynamic_pointer(expressions, b, b_index)?;
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

    fn qmatmul_coop_partition(
        &self,
        op: &TileQMatmulProgramOp,
    ) -> Option<(u32, u32, CoopPartition)> {
        let m = op.tile_m;
        let n = op.tile_n;
        if self.coop_subgroups <= 1 {
            return Some((m, n, CoopPartition::Single));
        }
        if m == 64 && n == 64 && self.coop_subgroups == 4 {
            return Some((
                32,
                32,
                CoopPartition::InterleavedGrid {
                    row_groups: 2,
                    col_groups: 2,
                },
            ));
        }
        if m == 128 && n == 64 && self.coop_subgroups == 8 {
            return Some((
                32,
                32,
                CoopPartition::InterleavedGrid {
                    row_groups: 4,
                    col_groups: 2,
                },
            ));
        }
        if m == 128 && n == 128 && self.coop_subgroups == 16 {
            return Some((
                32,
                32,
                CoopPartition::InterleavedGrid {
                    row_groups: 4,
                    col_groups: 4,
                },
            ));
        }
        if m == 128 && n == 32 && self.coop_subgroups == 8 {
            return Some((
                32,
                16,
                CoopPartition::InterleavedGrid {
                    row_groups: 4,
                    col_groups: 2,
                },
            ));
        }
        if m.is_multiple_of(32) && n.is_multiple_of(32) {
            let row_groups = m / 32;
            let col_groups = n / 32;
            if row_groups
                .checked_mul(col_groups)
                .is_some_and(|subgroups| subgroups == self.coop_subgroups)
            {
                return Some((
                    32,
                    32,
                    CoopPartition::InterleavedGrid {
                        row_groups,
                        col_groups,
                    },
                ));
            }
        }
        if n >= m && n.is_multiple_of(self.coop_subgroups) {
            return Some((m, n / self.coop_subgroups, CoopPartition::Columns));
        }
        if m.is_multiple_of(self.coop_subgroups) {
            return Some((m / self.coop_subgroups, n, CoopPartition::Rows));
        }
        if n.is_multiple_of(self.coop_subgroups) {
            return Some((m, n / self.coop_subgroups, CoopPartition::Columns));
        }
        None
    }

    fn tile_matrix_index(
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
