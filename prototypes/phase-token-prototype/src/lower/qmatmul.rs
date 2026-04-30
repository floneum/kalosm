use super::*;

type DequantizedValues4 = (Vec<Handle<Expression>>, Vec<Range<Expression>>);

impl<'a> Lowerer<'a> {
    pub(super) fn lower_qdequantize(
        &self,
        expressions: &mut Arena<Expression>,
        _scratch: ScratchLocals,
        op: &QDequantizeOp,
    ) -> Result<Statement, LowerError> {
        self.storage_layout(&op.b.data)?;
        let output_layout = self.storage_layout(&op.y)?;
        let total =
            op.b.rows
                .checked_mul(op.b.cols)
                .ok_or(LowerError::UnsupportedOperation(
                    "qdequantize output element count overflow",
                ))?;
        if output_layout.element_count().get() != total || !output_layout.is_row_major() {
            return Err(LowerError::UnsupportedOperation(
                "qdequantize output must be row-major dense storage",
            ));
        }

        let mut body = Block::new();
        let local = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
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
        let mut emits = Vec::new();
        let wg_y_offset =
            self.mul_literal_u32_emitted(expressions, wg_y, op.workgroups_x, &mut emits);
        let linear_wg = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            wg_x,
            wg_y_offset,
        );
        let base = self.mul_literal_u32_emitted(
            expressions,
            linear_wg,
            self.workgroup_invocations,
            &mut emits,
        );
        let flat = self.bin(expressions, &mut emits, BinaryOperator::Add, base, local);
        Self::push_emits(&mut body, emits);

        let done = self.bin_lit_u32(
            expressions,
            &mut body,
            BinaryOperator::GreaterEqual,
            flat,
            total,
        );
        body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Return { value: None }]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let mut emits = Vec::new();
        let dense_row = self.div_literal_u32_emitted(expressions, flat, op.b.rows, &mut emits);
        let dense_col = self.mod_literal_u32_emitted(expressions, flat, op.b.rows, &mut emits);
        Self::push_emits(&mut body, emits);

        let (value, value_emits) =
            self.dequantize_qvalue(expressions, &op.b, dense_col, dense_row)?;
        Self::push_emits(&mut body, value_emits);
        let (y_ptr, y_ptr_emits) = self.storage_dynamic_pointer(expressions, &op.y, flat)?;
        Self::push_emits(&mut body, y_ptr_emits);
        body.push(
            Statement::Store {
                pointer: y_ptr,
                value,
            },
            Span::default(),
        );
        Ok(Statement::Block(body))
    }

    pub(super) fn lower_qmatmul(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &QMatMulOp,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.storage_layout(&op.a)?;
        let y_layout = self.storage_layout(&op.y)?;
        self.storage_layout(&op.b.data)?;
        let [m, k_size] = Self::matrix_shape(a_layout)?;
        let [y_m, y_n] = Self::matrix_shape(y_layout)?;
        if k_size != op.b.rows || m != y_m || op.b.cols != y_n {
            return Err(LowerError::UnsupportedOperation("qmatmul shape mismatch"));
        }
        if m == 1 && op.use_qgemv {
            return self.lower_qmatmul_qgemv(expressions, scratch, op, k_size, y_n);
        }
        if self.can_lower_qmatmul_tiled_coop(op, m, k_size, y_n)? {
            return self.lower_qmatmul_tiled_coop(expressions, scratch, op, m, k_size, y_n);
        }
        if self.can_lower_qmatmul_tiled(op)? {
            return self.lower_qmatmul_tiled(expressions, scratch, op, m, k_size, y_n);
        }

        let mut body = Block::new();
        let local = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        let wg = expressions.append(
            Expression::FunctionArgument(WORKGROUP_ID_ARG),
            Span::default(),
        );
        let row = expressions.append(
            Expression::AccessIndex { base: wg, index: 1 },
            Span::default(),
        );
        let col_group = expressions.append(
            Expression::AccessIndex { base: wg, index: 0 },
            Span::default(),
        );
        let mut emits = Vec::new();
        let col_base = self.mul_literal_u32_emitted(
            expressions,
            col_group,
            self.workgroup_invocations,
            &mut emits,
        );
        let col = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_base,
            local,
        );
        Self::push_emits(&mut body, emits);

        let col_done = self.bin_lit_u32(
            expressions,
            &mut body,
            BinaryOperator::GreaterEqual,
            col,
            op.b.cols,
        );
        body.push(
            Statement::If {
                condition: col_done,
                accept: Block::from_vec(vec![Statement::Return { value: None }]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let sum_ptr =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        body.push(
            Statement::Store {
                pointer: sum_ptr,
                value: zero,
            },
            Span::default(),
        );
        let k_ptr = expressions.append(Expression::LocalVariable(scratch.mma_k), Span::default());
        let zero_u = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        body.push(
            Statement::Store {
                pointer: k_ptr,
                value: zero_u,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
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
        for lane in 0..op.vector_width {
            let mut lane_emits = Vec::new();
            let k_lane = self.add_literal_u32_emitted(expressions, k, lane, &mut lane_emits);
            Self::push_emits(&mut loop_body, lane_emits);
            let lane_block = self.lower_qmatmul_lane(expressions, scratch, op, row, col, k_lane)?;
            if lane != 0 && k_size % op.vector_width != 0 {
                let guard = self.bin_lit_u32(
                    expressions,
                    &mut loop_body,
                    BinaryOperator::Less,
                    k_lane,
                    k_size,
                );
                loop_body.push(
                    Statement::If {
                        condition: guard,
                        accept: lane_block,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            } else {
                loop_body.push(Statement::Block(lane_block), Span::default());
            }
        }
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::from_vec(vec![self.increment_u32_local(
                    expressions,
                    scratch.mma_k,
                    op.vector_width,
                )]),
                break_if: None,
            },
            Span::default(),
        );

        let (y_index, y_emits) = self.storage_index_from_coords(expressions, &op.y, &[row, col])?;
        let (y_ptr, y_ptr_emits) = self.storage_dynamic_pointer(expressions, &op.y, y_index)?;
        Self::push_emits(&mut body, y_emits);
        Self::push_emits(&mut body, y_ptr_emits);
        let sum = expressions.append(Expression::Load { pointer: sum_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, sum)),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: y_ptr,
                value: sum,
            },
            Span::default(),
        );
        Ok(Statement::Block(body))
    }

    fn lower_qmatmul_qgemv(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &QMatMulOp,
        k_size: u32,
        n: u32,
    ) -> Result<Statement, LowerError> {
        const VALUES_PER_LANE: u32 = 8;

        let output_cols_per_subgroup = match op.b.format {
            GgmlQuantFormat::Q2K
            | GgmlQuantFormat::Q3K
            | GgmlQuantFormat::Q4K
            | GgmlQuantFormat::Q5K
            | GgmlQuantFormat::Q6K
            | GgmlQuantFormat::Q8K => 2,
            GgmlQuantFormat::Q4_0
            | GgmlQuantFormat::Q4_1
            | GgmlQuantFormat::Q5_0
            | GgmlQuantFormat::Q5_1
            | GgmlQuantFormat::Q8_0
            | GgmlQuantFormat::Q8_1 => 4,
        };

        let mut body = Block::new();
        let wg = expressions.append(
            Expression::FunctionArgument(WORKGROUP_ID_ARG),
            Span::default(),
        );
        let n_group = expressions.append(
            Expression::AccessIndex { base: wg, index: 0 },
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
        let cols_per_workgroup = self.mul_literal_u32_emitted(
            expressions,
            num_subgroups,
            output_cols_per_subgroup,
            &mut emits,
        );
        let col_base = self.bin(
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
            col_base,
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

        let sum_locals = (0..output_cols_per_subgroup)
            .map(|index| self.gemm_sum_local(scratch, index))
            .collect::<Result<Vec<_>, _>>()?;
        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        for sum in &sum_locals {
            let pointer = expressions.append(Expression::LocalVariable(*sum), Span::default());
            body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
        }
        let k_ptr = expressions.append(Expression::LocalVariable(scratch.mma_k), Span::default());
        let mut initial_k_emits = Vec::new();
        let initial_k = self.mul_literal_u32_emitted(
            expressions,
            subgroup_lane,
            VALUES_PER_LANE,
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
        let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
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

        let row_zero = self.u32(expressions, 0);
        let mut a_vecs = Vec::with_capacity(2);
        for chunk in 0..2 {
            let mut a_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let mut a_emits = Vec::new();
                let a_k =
                    self.add_literal_u32_emitted(expressions, k, chunk * 4 + lane, &mut a_emits);
                let (a_index, a_index_emits) =
                    self.storage_index_from_coords(expressions, &op.a, &[row_zero, a_k])?;
                let (a_ptr, a_ptr_emits) =
                    self.storage_dynamic_pointer(expressions, &op.a, a_index)?;
                a_emits.extend(a_index_emits);
                a_emits.extend(a_ptr_emits);
                Self::push_emits(&mut loop_body, a_emits);
                let a = expressions.append(Expression::Load { pointer: a_ptr }, Span::default());
                loop_body.push(
                    Statement::Emit(Self::single_expression_range(expressions, a)),
                    Span::default(),
                );
                a_values.push(a);
            }
            let a_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components: a_values,
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
            let (b_values, b_emits) = self.dequantize_qvalues8(expressions, &op.b, k, col)?;
            Self::push_emits(&mut fma_body, b_emits);
            let mut dots = Vec::with_capacity(2);
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
            let dot = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: dots[0],
                    right: dots[1],
                },
                Span::default(),
            );
            fma_body.push(
                Statement::Emit(Self::single_expression_range(expressions, dot)),
                Span::default(),
            );
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
                continuing: self.increment_u32_local_by_expr(
                    expressions,
                    scratch.mma_k,
                    subgroup_size,
                    VALUES_PER_LANE,
                ),
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
        for ((col, in_bounds), sum_local) in cols.into_iter().zip(col_in_bounds).zip(sum_locals) {
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
            let row_zero = self.u32(expressions, 0);
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

    fn can_lower_qmatmul_tiled(&self, op: &QMatMulOp) -> Result<bool, LowerError> {
        let a_tile = self.tile_layout(op.a_tile)?;
        let b_tile = self.tile_layout(op.b_tile)?;
        Ok(a_tile.memory_level() == MemoryLevel::Workgroup
            && b_tile.memory_level() == MemoryLevel::Workgroup
            && a_tile.shape().rank() == 2
            && b_tile.shape().rank() == 2
            && a_tile.shape().dims()[0].get() == op.tile_m
            && a_tile.shape().dims()[1].get() == op.tile_k
            && b_tile.shape().dims()[0].get() == op.tile_k
            && b_tile.shape().dims()[1].get() == op.tile_n
            && op.tile_n.is_multiple_of(8)
            && op.tile_k.is_multiple_of(4)
            && op.tile_m * (op.tile_n / 8) <= self.workgroup_invocations
            && op.tile_m == 32
            && op.tile_n == 64
            && op.tile_k == 32)
    }

    fn can_lower_qmatmul_tiled_coop(
        &self,
        op: &QMatMulOp,
        m: u32,
        k_size: u32,
        n: u32,
    ) -> Result<bool, LowerError> {
        let a_tile = self.tile_layout(op.a_tile)?;
        let b_tile = self.tile_layout(op.b_tile)?;
        let y_layout = self.storage_layout(&op.y)?;
        Ok(self.coop_f32_a_ty.is_some()
            && self.coop_f32_b_ty.is_some()
            && self.coop_f32_c_ty.is_some()
            && a_tile.memory_level() == MemoryLevel::Workgroup
            && b_tile.memory_level() == MemoryLevel::Workgroup
            && Self::can_cooperative_store_matrix(y_layout)
            && a_tile.shape().rank() == 2
            && b_tile.shape().rank() == 2
            && a_tile.shape().dims()[0].get() == op.tile_m
            && a_tile.shape().dims()[1].get() == op.tile_k
            && b_tile.shape().dims()[0].get() == op.tile_k
            && b_tile.shape().dims()[1].get() == op.tile_n
            && (op.tile_m == 32 || op.tile_m == 64 || op.tile_m == 128)
            && (op.tile_n == 32 || op.tile_n == 64 || op.tile_n == 128)
            && (op.tile_k == 16 || op.tile_k == 32)
            && op.tile_m.is_multiple_of(COOP_MATRIX_DIM)
            && op.tile_n.is_multiple_of(COOP_MATRIX_DIM)
            && op.tile_k.is_multiple_of(COOP_MATRIX_DIM)
            && m.is_multiple_of(op.tile_m * self.qmatmul_coop_row_panels(op))
            && n.is_multiple_of(op.tile_n)
            && k_size.is_multiple_of(op.tile_k)
            && self
                .qmatmul_coop_partition(op)
                .map(|(rows, cols, _)| {
                    rows.is_multiple_of(COOP_MATRIX_DIM)
                        && cols.is_multiple_of(COOP_MATRIX_DIM)
                        && (rows / COOP_MATRIX_DIM)
                            * (cols / COOP_MATRIX_DIM)
                            * self.qmatmul_coop_row_panels(op)
                            <= 16
                })
                .unwrap_or(false))
    }

    fn can_cooperative_store_matrix(layout: &Layout) -> bool {
        Self::cooperative_matrix_store_layout(layout).is_ok()
    }

    fn should_load_qmatmul_a_tile_col_coalesced(layout: &Layout) -> bool {
        layout.shape().rank() == 2
            && layout.strides().rank() == 2
            && layout.strides().values()[0] < layout.strides().values()[1]
    }

    fn cooperative_matrix_store_layout(layout: &Layout) -> Result<(u32, bool), LowerError> {
        if layout.shape().rank() != 2 || layout.strides().rank() != 2 {
            return Err(LowerError::UnsupportedOperation(
                "qmatmul cooperative store requires a rank-2 output view",
            ));
        }

        let strides = layout.strides().values();
        if strides[1] == 1 {
            Ok((strides[0], false))
        } else if strides[0] == 1 {
            Ok((strides[1], true))
        } else {
            Err(LowerError::UnsupportedOperation(
                "qmatmul cooperative store requires row-major or column-major output strides",
            ))
        }
    }

    fn qmatmul_coop_row_panels(&self, op: &QMatMulOp) -> u32 {
        let _ = op;
        1
    }

    fn qmatmul_coop_partition(&self, op: &QMatMulOp) -> Option<(u32, u32, CoopPartition)> {
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

    fn lower_qmatmul_tiled_coop(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &QMatMulOp,
        m: u32,
        k_size: u32,
        n: u32,
    ) -> Result<Statement, LowerError> {
        let (subgroup_rows, subgroup_cols, partition) =
            self.qmatmul_coop_partition(op)
                .ok_or(LowerError::UnsupportedOperation(
                    "qmatmul cooperative tile partition mismatch",
                ))?;
        let row_panels = self.qmatmul_coop_row_panels(op);
        let tile_rows = subgroup_rows / COOP_MATRIX_DIM;
        let tile_cols = subgroup_cols / COOP_MATRIX_DIM;
        let fragment_count = (tile_rows * tile_cols) as usize;
        let acc_count = fragment_count * row_panels as usize;
        let acc_locals = scratch.coop_accs[..acc_count]
            .iter()
            .copied()
            .collect::<Option<Vec<_>>>()
            .ok_or(LowerError::UnsupportedOperation(
                "qmatmul cooperative accumulator locals were not allocated",
            ))?;

        let a_layout = self.tile_layout(op.a_tile)?;
        let b_layout = self.tile_layout(op.b_tile)?;
        let y_layout = self.storage_layout(&op.y)?;
        let (y_leading_stride, y_row_major) = Self::cooperative_matrix_store_layout(y_layout)?;
        let a_stride = expressions.append(
            Expression::Literal(Literal::U32(Self::row_major_matrix_leading_stride(
                a_layout,
            )?)),
            Span::default(),
        );
        let b_stride = expressions.append(
            Expression::Literal(Literal::U32(Self::row_major_matrix_leading_stride(
                b_layout,
            )?)),
            Span::default(),
        );
        let y_stride = expressions.append(
            Expression::Literal(Literal::U32(y_leading_stride)),
            Span::default(),
        );
        if !Self::is_row_major_storage_matrix(a_layout)
            || !Self::is_row_major_storage_matrix(b_layout)
        {
            return Err(LowerError::UnsupportedOperation(
                "qmatmul cooperative lowering requires row-major workgroup tiles",
            ));
        }

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
        let row_base = self.mul_literal_u32_emitted(
            expressions,
            tile_row,
            op.tile_m * row_panels,
            &mut tile_emits,
        );
        let col_base =
            self.mul_literal_u32_emitted(expressions, tile_col, op.tile_n, &mut tile_emits);
        Self::push_emits(&mut body, tile_emits);

        let mut acc_pointers = Vec::with_capacity(acc_locals.len());
        for local in acc_locals {
            let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
            acc_pointers.push(pointer);
        }

        let k_ptr = expressions.append(Expression::LocalVariable(scratch.mma_k), Span::default());
        let zero_u = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        body.push(
            Statement::Store {
                pointer: k_ptr,
                value: zero_u,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let (k_start, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
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
            k_size,
            true,
        )?;
        for panel in 0..row_panels {
            let mut panel_emits = Vec::new();
            let panel_row_base = self.add_literal_u32_emitted(
                expressions,
                row_base,
                panel * op.tile_m,
                &mut panel_emits,
            );
            Self::push_emits(&mut loop_body, panel_emits);
            self.append_qmatmul_a_tile_loads(
                expressions,
                &mut loop_body,
                op,
                local,
                panel_row_base,
                k_start,
                k_size,
                m,
                true,
            )?;
            loop_body.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            let acc_start = panel as usize * fragment_count;
            let acc_end = acc_start + fragment_count;
            for kk in (0..op.tile_k).step_by(COOP_MATRIX_DIM as usize) {
                let base_k =
                    expressions.append(Expression::Literal(Literal::U32(kk)), Span::default());
                self.append_shared_coop_k_chunk(
                    expressions,
                    &mut loop_body,
                    op.a_tile,
                    op.b_tile,
                    &acc_pointers[acc_start..acc_end],
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
        }
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::from_vec(vec![self.increment_u32_local(
                    expressions,
                    scratch.mma_k,
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
        for panel in 0..row_panels {
            let mut panel_emits = Vec::new();
            let panel_row_base = self.add_literal_u32_emitted(
                expressions,
                row_base,
                panel * op.tile_m,
                &mut panel_emits,
            );
            Self::push_emits(&mut store_body, panel_emits);
            let acc_base = panel as usize * fragment_count;
            for row_tile in 0..tile_rows {
                for col_tile in 0..tile_cols {
                    let acc_index = acc_base + (row_tile * tile_cols + col_tile) as usize;
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
                    let local_col = if let Some(subgroup_col_base) = subgroup_col_base {
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
                    let mut output_emits = Vec::new();
                    let row = self.bin(
                        expressions,
                        &mut output_emits,
                        BinaryOperator::Add,
                        panel_row_base,
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
        }
        body.push(Statement::Block(store_body), Span::default());

        let _ = n;
        Ok(Statement::Block(body))
    }

    fn lower_qmatmul_tiled(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &QMatMulOp,
        m: u32,
        k_size: u32,
        n: u32,
    ) -> Result<Statement, LowerError> {
        const OUTPUT_COLS_PER_LANE: u32 = 8;

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

        let mut emits = Vec::new();
        let output_col_groups = op.tile_n / OUTPUT_COLS_PER_LANE;
        let output_lane_count = op.tile_m * output_col_groups;
        let active_lane = self.cmp_lit(
            expressions,
            &mut emits,
            BinaryOperator::Less,
            local,
            output_lane_count,
        );
        let output_local_row =
            self.div_literal_u32_emitted(expressions, local, output_col_groups, &mut emits);
        let output_col_group =
            self.mod_literal_u32_emitted(expressions, local, output_col_groups, &mut emits);
        let row_base = self.mul_literal_u32_emitted(expressions, tile_row, op.tile_m, &mut emits);
        let col_base = self.mul_literal_u32_emitted(expressions, tile_col, op.tile_n, &mut emits);
        let row = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            row_base,
            output_local_row,
        );
        let local_col0 = self.mul_literal_u32_emitted(
            expressions,
            output_col_group,
            OUTPUT_COLS_PER_LANE,
            &mut emits,
        );
        let col0 = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_base,
            local_col0,
        );
        Self::push_emits(&mut body, emits);

        let row_in_bounds = self.bin_lit_u32(expressions, &mut body, BinaryOperator::Less, row, m);
        let active_output_row = expressions.append(
            Expression::Binary {
                op: BinaryOperator::LogicalAnd,
                left: active_lane,
                right: row_in_bounds,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(
                expressions,
                active_output_row,
            )),
            Span::default(),
        );
        let mut cols = Vec::with_capacity(OUTPUT_COLS_PER_LANE as usize);
        let mut output_in_bounds = Vec::with_capacity(OUTPUT_COLS_PER_LANE as usize);
        for offset in 0..OUTPUT_COLS_PER_LANE {
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
            let col_in_bounds =
                self.bin_lit_u32(expressions, &mut body, BinaryOperator::Less, col, n);
            let in_bounds = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::LogicalAnd,
                    left: active_output_row,
                    right: col_in_bounds,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, in_bounds)),
                Span::default(),
            );
            cols.push(col);
            output_in_bounds.push(in_bounds);
        }

        let sum_locals = (0..OUTPUT_COLS_PER_LANE)
            .map(|index| self.gemm_sum_local(scratch, index))
            .collect::<Result<Vec<_>, _>>()?;
        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let mut init_body = Block::new();
        for sum in &sum_locals {
            let pointer = expressions.append(Expression::LocalVariable(*sum), Span::default());
            init_body.push(
                Statement::Store {
                    pointer,
                    value: zero,
                },
                Span::default(),
            );
        }
        body.push(
            Statement::If {
                condition: active_lane,
                accept: init_body,
                reject: Block::new(),
            },
            Span::default(),
        );
        let k_ptr = expressions.append(Expression::LocalVariable(scratch.mma_k), Span::default());
        let zero_u = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        body.push(
            Statement::Store {
                pointer: k_ptr,
                value: zero_u,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let (k_start, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
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

        self.append_qmatmul_tile_loads(
            expressions,
            &mut loop_body,
            op,
            local,
            row_base,
            col_base,
            k_start,
            k_size,
            m,
        )?;
        loop_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        loop_body.push(
            Statement::If {
                condition: active_lane,
                accept: self.qmatmul_tiled_accumulate(
                    expressions,
                    scratch,
                    op,
                    output_local_row,
                    local_col0,
                )?,
                reject: Block::new(),
            },
            Span::default(),
        );
        loop_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::from_vec(vec![self.increment_u32_local(
                    expressions,
                    scratch.mma_k,
                    op.tile_k,
                )]),
                break_if: None,
            },
            Span::default(),
        );

        for ((sum_local, col), in_bounds) in sum_locals.into_iter().zip(cols).zip(output_in_bounds)
        {
            let (y_index, y_emits) =
                self.storage_index_from_coords(expressions, &op.y, &[row, col])?;
            let (y_ptr, y_ptr_emits) = self.storage_dynamic_pointer(expressions, &op.y, y_index)?;
            Self::push_emits(&mut body, y_emits);
            Self::push_emits(&mut body, y_ptr_emits);
            let sum_ptr = expressions.append(Expression::LocalVariable(sum_local), Span::default());
            let sum = expressions.append(Expression::Load { pointer: sum_ptr }, Span::default());
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, sum)),
                Span::default(),
            );
            body.push(
                Statement::If {
                    condition: in_bounds,
                    accept: Block::from_vec(vec![Statement::Store {
                        pointer: y_ptr,
                        value: sum,
                    }]),
                    reject: Block::new(),
                },
                Span::default(),
            );
        }

        Ok(Statement::Block(body))
    }

    #[allow(clippy::too_many_arguments)]
    fn append_qmatmul_a_tile_loads(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: &QMatMulOp,
        local: Handle<Expression>,
        row_base: Handle<Expression>,
        k_start: Handle<Expression>,
        k_size: u32,
        m: u32,
        bounds_known: bool,
    ) -> Result<(), LowerError> {
        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let a_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.a_tile)?)?;
        if Self::should_load_qmatmul_a_tile_col_coalesced(self.storage_layout(&op.a)?) {
            return self.append_qmatmul_a_tile_loads_col_coalesced(
                expressions,
                body,
                op,
                local,
                row_base,
                k_start,
                k_size,
                m,
                bounds_known,
            );
        }

        const VALUES_PER_LOAD: u32 = 8;
        let groups_per_row = op.tile_k / VALUES_PER_LOAD;
        let a_vector_loads = groups_per_row * op.tile_m;
        let a_load_passes = a_vector_loads.div_ceil(self.workgroup_invocations);
        for pass in 0..a_load_passes {
            let mut emits = Vec::new();
            let a_flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut emits,
            );
            let a_lane_active = self.cmp_lit(
                expressions,
                &mut emits,
                BinaryOperator::Less,
                a_flat,
                a_vector_loads,
            );
            Self::push_emits(body, emits);

            let mut a_lane_body = Block::new();
            let mut emits = Vec::new();
            let a_local_row =
                self.div_literal_u32_emitted(expressions, a_flat, groups_per_row, &mut emits);
            let a_col_group =
                self.mod_literal_u32_emitted(expressions, a_flat, groups_per_row, &mut emits);
            let a_local_col_base =
                self.mul_literal_u32_emitted(expressions, a_col_group, VALUES_PER_LOAD, &mut emits);
            let a_row = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                row_base,
                a_local_row,
            );
            let a_k = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                k_start,
                a_local_col_base,
            );
            let a_load_in_bounds = if bounds_known {
                None
            } else {
                let a_row_in_bounds =
                    self.cmp_lit(expressions, &mut emits, BinaryOperator::Less, a_row, m);
                let a_k_in_bounds =
                    self.cmp_lit(expressions, &mut emits, BinaryOperator::Less, a_k, k_size);
                Some(self.bin(
                    expressions,
                    &mut emits,
                    BinaryOperator::LogicalAnd,
                    a_row_in_bounds,
                    a_k_in_bounds,
                ))
            };
            let mut a_tile_ptrs = Vec::with_capacity(VALUES_PER_LOAD as usize);
            for lane in 0..VALUES_PER_LOAD {
                let a_local_col =
                    self.add_literal_u32_emitted(expressions, a_local_col_base, lane, &mut emits);
                let a_tile_index = self.tile_matrix_index(
                    expressions,
                    &mut emits,
                    a_local_row,
                    a_local_col,
                    a_stride,
                );
                let (a_tile_ptr, a_tile_ptr_emits) =
                    self.tile_dynamic_pointer(expressions, op.a_tile, a_tile_index)?;
                emits.extend(a_tile_ptr_emits);
                a_tile_ptrs.push(a_tile_ptr);
            }
            Self::push_emits(&mut a_lane_body, emits);

            let mut a_accept = Block::new();
            for (lane, a_tile_ptr) in a_tile_ptrs.iter().copied().enumerate() {
                let mut lane_emits = Vec::new();
                let a_k_lane =
                    self.add_literal_u32_emitted(expressions, a_k, lane as u32, &mut lane_emits);
                let (a_index, a_index_emits) =
                    self.storage_index_from_coords(expressions, &op.a, &[a_row, a_k_lane])?;
                lane_emits.extend(a_index_emits);
                let (a_ptr, a_ptr_emits) =
                    self.storage_dynamic_pointer(expressions, &op.a, a_index)?;
                lane_emits.extend(a_ptr_emits);
                Self::push_emits(&mut a_accept, lane_emits);
                let a_value =
                    expressions.append(Expression::Load { pointer: a_ptr }, Span::default());
                a_accept.push(
                    Statement::Emit(Self::single_expression_range(expressions, a_value)),
                    Span::default(),
                );
                a_accept.push(
                    Statement::Store {
                        pointer: a_tile_ptr,
                        value: a_value,
                    },
                    Span::default(),
                );
            }
            if let Some(a_load_in_bounds) = a_load_in_bounds {
                let a_reject = Block::from_vec(
                    a_tile_ptrs
                        .iter()
                        .copied()
                        .map(|pointer| Statement::Store {
                            pointer,
                            value: zero,
                        })
                        .collect(),
                );
                a_lane_body.push(
                    Statement::If {
                        condition: a_load_in_bounds,
                        accept: a_accept,
                        reject: a_reject,
                    },
                    Span::default(),
                );
            } else {
                a_lane_body.push(Statement::Block(a_accept), Span::default());
            }
            if (pass + 1) * self.workgroup_invocations <= a_vector_loads {
                body.push(Statement::Block(a_lane_body), Span::default());
            } else {
                body.push(
                    Statement::If {
                        condition: a_lane_active,
                        accept: a_lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn append_qmatmul_a_tile_loads_col_coalesced(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: &QMatMulOp,
        local: Handle<Expression>,
        row_base: Handle<Expression>,
        k_start: Handle<Expression>,
        k_size: u32,
        m: u32,
        bounds_known: bool,
    ) -> Result<(), LowerError> {
        const ROWS_PER_LOAD: u32 = 8;

        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        let a_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.a_tile)?)?;
        let groups_per_col = op.tile_m / ROWS_PER_LOAD;
        let a_vector_loads = groups_per_col * op.tile_k;
        let a_load_passes = a_vector_loads.div_ceil(self.workgroup_invocations);
        for pass in 0..a_load_passes {
            let mut emits = Vec::new();
            let a_flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut emits,
            );
            let a_lane_active = self.cmp_lit(
                expressions,
                &mut emits,
                BinaryOperator::Less,
                a_flat,
                a_vector_loads,
            );
            Self::push_emits(body, emits);

            let mut a_lane_body = Block::new();
            let mut emits = Vec::new();
            let a_local_col =
                self.div_literal_u32_emitted(expressions, a_flat, groups_per_col, &mut emits);
            let a_row_group =
                self.mod_literal_u32_emitted(expressions, a_flat, groups_per_col, &mut emits);
            let a_local_row_base =
                self.mul_literal_u32_emitted(expressions, a_row_group, ROWS_PER_LOAD, &mut emits);
            let a_row_base = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                row_base,
                a_local_row_base,
            );
            let a_k = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::Add,
                k_start,
                a_local_col,
            );
            let mut a_tile_ptrs = Vec::with_capacity(ROWS_PER_LOAD as usize);
            for lane in 0..ROWS_PER_LOAD {
                let a_local_row =
                    self.add_literal_u32_emitted(expressions, a_local_row_base, lane, &mut emits);
                let a_tile_index = self.tile_matrix_index(
                    expressions,
                    &mut emits,
                    a_local_row,
                    a_local_col,
                    a_stride,
                );
                let (a_tile_ptr, a_tile_ptr_emits) =
                    self.tile_dynamic_pointer(expressions, op.a_tile, a_tile_index)?;
                emits.extend(a_tile_ptr_emits);
                a_tile_ptrs.push(a_tile_ptr);
            }
            Self::push_emits(&mut a_lane_body, emits);

            let mut a_accept = Block::new();
            for (lane, a_tile_ptr) in a_tile_ptrs.iter().copied().enumerate() {
                let mut lane_emits = Vec::new();
                let a_row = self.add_literal_u32_emitted(
                    expressions,
                    a_row_base,
                    lane as u32,
                    &mut lane_emits,
                );

                let mut load_store = Block::new();
                let (a_index, a_index_emits) =
                    self.storage_index_from_coords(expressions, &op.a, &[a_row, a_k])?;
                let (a_ptr, a_ptr_emits) =
                    self.storage_dynamic_pointer(expressions, &op.a, a_index)?;
                Self::push_emits(&mut load_store, a_index_emits);
                Self::push_emits(&mut load_store, a_ptr_emits);
                let a_value =
                    expressions.append(Expression::Load { pointer: a_ptr }, Span::default());
                load_store.push(
                    Statement::Emit(Self::single_expression_range(expressions, a_value)),
                    Span::default(),
                );
                load_store.push(
                    Statement::Store {
                        pointer: a_tile_ptr,
                        value: a_value,
                    },
                    Span::default(),
                );

                if bounds_known {
                    Self::push_emits(&mut a_accept, lane_emits);
                    a_accept.push(Statement::Block(load_store), Span::default());
                } else {
                    let a_row_in_bounds =
                        self.cmp_lit(expressions, &mut lane_emits, BinaryOperator::Less, a_row, m);
                    let a_k_in_bounds = self.cmp_lit(
                        expressions,
                        &mut lane_emits,
                        BinaryOperator::Less,
                        a_k,
                        k_size,
                    );
                    let a_load_in_bounds = self.bin(
                        expressions,
                        &mut lane_emits,
                        BinaryOperator::LogicalAnd,
                        a_row_in_bounds,
                        a_k_in_bounds,
                    );
                    Self::push_emits(&mut a_accept, lane_emits);
                    a_accept.push(
                        Statement::If {
                            condition: a_load_in_bounds,
                            accept: load_store,
                            reject: Block::from_vec(vec![Statement::Store {
                                pointer: a_tile_ptr,
                                value: zero,
                            }]),
                        },
                        Span::default(),
                    );
                }
            }
            a_lane_body.push(Statement::Block(a_accept), Span::default());
            if (pass + 1) * self.workgroup_invocations <= a_vector_loads {
                body.push(Statement::Block(a_lane_body), Span::default());
            } else {
                body.push(
                    Statement::If {
                        condition: a_lane_active,
                        accept: a_lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn append_qmatmul_b_tile_loads(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: &QMatMulOp,
        local: Handle<Expression>,
        col_base: Handle<Expression>,
        k_start: Handle<Expression>,
        k_size: u32,
        bounds_known: bool,
    ) -> Result<(), LowerError> {
        let b_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.b_tile)?)?;
        let values_per_load = match op.b.format {
            GgmlQuantFormat::Q4_0
            | GgmlQuantFormat::Q4_1
            | GgmlQuantFormat::Q5_0
            | GgmlQuantFormat::Q5_1
            | GgmlQuantFormat::Q8_0
            | GgmlQuantFormat::Q8_1 => 8,
            GgmlQuantFormat::Q2K
            | GgmlQuantFormat::Q3K
            | GgmlQuantFormat::Q4K
            | GgmlQuantFormat::Q5K
            | GgmlQuantFormat::Q6K
            | GgmlQuantFormat::Q8K => 16,
        };
        let b_vector_loads = (op.tile_k / values_per_load) * op.tile_n;
        let b_load_passes = b_vector_loads.div_ceil(self.workgroup_invocations);
        for pass in 0..b_load_passes {
            let mut b_lane_emits = Vec::new();
            let b_flat = self.add_literal_u32_emitted(
                expressions,
                local,
                pass * self.workgroup_invocations,
                &mut b_lane_emits,
            );
            let b_lane_active = self.cmp_lit(
                expressions,
                &mut b_lane_emits,
                BinaryOperator::Less,
                b_flat,
                b_vector_loads,
            );
            Self::push_emits(body, b_lane_emits);

            let mut b_lane_body = Block::new();
            let mut b_emits = Vec::new();
            let b_vec_group =
                self.div_literal_u32_emitted(expressions, b_flat, op.tile_n, &mut b_emits);
            let b_vec_col =
                self.mod_literal_u32_emitted(expressions, b_flat, op.tile_n, &mut b_emits);
            let b_k_base = self.mul_literal_u32_emitted(
                expressions,
                b_vec_group,
                values_per_load,
                &mut b_emits,
            );
            let b_k_base = self.bin(
                expressions,
                &mut b_emits,
                BinaryOperator::Add,
                k_start,
                b_k_base,
            );
            let b_vec_col_global = self.bin(
                expressions,
                &mut b_emits,
                BinaryOperator::Add,
                col_base,
                b_vec_col,
            );
            let b_vec_in_bounds = if bounds_known {
                None
            } else {
                let b_vec_k_in_bounds = self.cmp_lit(
                    expressions,
                    &mut b_emits,
                    BinaryOperator::Less,
                    b_k_base,
                    k_size,
                );
                let b_vec_col_in_bounds = self.cmp_lit(
                    expressions,
                    &mut b_emits,
                    BinaryOperator::Less,
                    b_vec_col_global,
                    op.b.cols,
                );
                Some(self.bin(
                    expressions,
                    &mut b_emits,
                    BinaryOperator::LogicalAnd,
                    b_vec_k_in_bounds,
                    b_vec_col_in_bounds,
                ))
            };

            let mut b_tile_ptrs = Vec::with_capacity(values_per_load as usize);
            for lane in 0..values_per_load {
                let tile_row =
                    self.add_literal_u32_emitted(expressions, b_k_base, lane, &mut b_emits);
                let tile_row = self.bin(
                    expressions,
                    &mut b_emits,
                    BinaryOperator::Subtract,
                    tile_row,
                    k_start,
                );
                let b_tile_index = self.tile_matrix_index(
                    expressions,
                    &mut b_emits,
                    tile_row,
                    b_vec_col,
                    b_stride,
                );
                let (b_tile_ptr, b_tile_ptr_emits) =
                    self.tile_dynamic_pointer(expressions, op.b_tile, b_tile_index)?;
                b_emits.extend(b_tile_ptr_emits);
                b_tile_ptrs.push(b_tile_ptr);
            }
            Self::push_emits(&mut b_lane_body, b_emits);

            let mut b_accept = Block::new();
            let (b_values, b_value_emits) = if values_per_load == 16 {
                self.dequantize_qvalues16(expressions, &op.b, b_k_base, b_vec_col_global)?
            } else if values_per_load == 8 {
                self.dequantize_qvalues8(expressions, &op.b, b_k_base, b_vec_col_global)?
            } else {
                self.dequantize_qvalues4(expressions, &op.b, b_k_base, b_vec_col_global)?
            };
            Self::push_emits(&mut b_accept, b_value_emits);
            for (pointer, value) in b_tile_ptrs.iter().copied().zip(b_values) {
                b_accept.push(Statement::Store { pointer, value }, Span::default());
            }

            if let Some(b_vec_in_bounds) = b_vec_in_bounds {
                let zero =
                    expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
                let b_reject = Block::from_vec(
                    b_tile_ptrs
                        .iter()
                        .copied()
                        .map(|pointer| Statement::Store {
                            pointer,
                            value: zero,
                        })
                        .collect(),
                );
                b_lane_body.push(
                    Statement::If {
                        condition: b_vec_in_bounds,
                        accept: b_accept,
                        reject: b_reject,
                    },
                    Span::default(),
                );
            } else {
                b_lane_body.push(Statement::Block(b_accept), Span::default());
            }
            if (pass + 1) * self.workgroup_invocations <= b_vector_loads {
                body.push(Statement::Block(b_lane_body), Span::default());
            } else {
                body.push(
                    Statement::If {
                        condition: b_lane_active,
                        accept: b_lane_body,
                        reject: Block::new(),
                    },
                    Span::default(),
                );
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn append_qmatmul_tile_loads(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: &QMatMulOp,
        local: Handle<Expression>,
        row_base: Handle<Expression>,
        col_base: Handle<Expression>,
        k_start: Handle<Expression>,
        k_size: u32,
        m: u32,
    ) -> Result<(), LowerError> {
        self.append_qmatmul_a_tile_loads(
            expressions,
            body,
            op,
            local,
            row_base,
            k_start,
            k_size,
            m,
            false,
        )?;
        self.append_qmatmul_b_tile_loads(
            expressions,
            body,
            op,
            local,
            col_base,
            k_start,
            k_size,
            false,
        )
    }

    fn qmatmul_tiled_accumulate(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &QMatMulOp,
        local_row: Handle<Expression>,
        local_col0: Handle<Expression>,
    ) -> Result<Block, LowerError> {
        const OUTPUT_COLS_PER_LANE: u32 = 8;
        let a_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.a_tile)?)?;
        let b_stride = Self::row_major_matrix_leading_stride(self.tile_layout(op.b_tile)?)?;

        let mut body = Block::new();
        let sum_locals = (0..OUTPUT_COLS_PER_LANE)
            .map(|index| self.gemm_sum_local(scratch, index))
            .collect::<Result<Vec<_>, _>>()?;
        let mut col_emits = Vec::new();
        let local_cols = (0..OUTPUT_COLS_PER_LANE)
            .map(|offset| {
                if offset == 0 {
                    local_col0
                } else {
                    self.add_literal_u32_emitted(expressions, local_col0, offset, &mut col_emits)
                }
            })
            .collect::<Vec<_>>();
        Self::push_emits(&mut body, col_emits);

        for kk_base in (0..op.tile_k).step_by(4) {
            let mut emits = Vec::new();
            let mut a_ptrs = Vec::with_capacity(4);
            let mut b_ptrs_by_col: Vec<Vec<Handle<Expression>>> = (0..OUTPUT_COLS_PER_LANE)
                .map(|_| Vec::with_capacity(4))
                .collect();
            for lane in 0..4 {
                let kk = kk_base + lane;
                let kk_expr = self.u32(expressions, kk);
                let a_index =
                    self.tile_matrix_index(expressions, &mut emits, local_row, kk_expr, a_stride);
                let (a_ptr, a_ptr_emits) =
                    self.tile_dynamic_pointer(expressions, op.a_tile, a_index)?;
                emits.extend(a_ptr_emits);
                a_ptrs.push(a_ptr);

                for (col_offset, local_col) in local_cols.iter().copied().enumerate() {
                    let kk_expr = self.u32(expressions, kk);
                    let b_index = self.tile_matrix_index(
                        expressions,
                        &mut emits,
                        kk_expr,
                        local_col,
                        b_stride,
                    );
                    let (b_ptr, b_ptr_emits) =
                        self.tile_dynamic_pointer(expressions, op.b_tile, b_index)?;
                    emits.extend(b_ptr_emits);
                    b_ptrs_by_col[col_offset].push(b_ptr);
                }
            }
            Self::push_emits(&mut body, emits);

            let mut a_values = Vec::with_capacity(4);
            for a_ptr in a_ptrs {
                let a = expressions.append(Expression::Load { pointer: a_ptr }, Span::default());
                body.push(
                    Statement::Emit(Self::single_expression_range(expressions, a)),
                    Span::default(),
                );
                a_values.push(a);
            }
            let a_vec = expressions.append(
                Expression::Compose {
                    ty: self.f32_vec4_ty,
                    components: a_values,
                },
                Span::default(),
            );
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, a_vec)),
                Span::default(),
            );

            for (sum_local, b_ptrs) in sum_locals.iter().copied().zip(b_ptrs_by_col) {
                let mut b_values = Vec::with_capacity(4);
                for b_ptr in b_ptrs {
                    let b =
                        expressions.append(Expression::Load { pointer: b_ptr }, Span::default());
                    body.push(
                        Statement::Emit(Self::single_expression_range(expressions, b)),
                        Span::default(),
                    );
                    b_values.push(b);
                }
                let b_vec = expressions.append(
                    Expression::Compose {
                        ty: self.f32_vec4_ty,
                        components: b_values,
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
                body.push(
                    Statement::Emit(Self::range_from(expressions, b_vec, dot)),
                    Span::default(),
                );
                let sum_ptr =
                    expressions.append(Expression::LocalVariable(sum_local), Span::default());
                let sum =
                    expressions.append(Expression::Load { pointer: sum_ptr }, Span::default());
                let value = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Add,
                        left: sum,
                        right: dot,
                    },
                    Span::default(),
                );
                body.push(
                    Statement::Emit(Self::range_from(expressions, sum, value)),
                    Span::default(),
                );
                body.push(
                    Statement::Store {
                        pointer: sum_ptr,
                        value,
                    },
                    Span::default(),
                );
            }
        }
        Ok(body)
    }

    fn lower_qmatmul_lane(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &QMatMulOp,
        row: Handle<Expression>,
        col: Handle<Expression>,
        k: Handle<Expression>,
    ) -> Result<Block, LowerError> {
        let (a_index, a_emits) = self.storage_index_from_coords(expressions, &op.a, &[row, k])?;
        let (a_ptr, a_ptr_emits) = self.storage_dynamic_pointer(expressions, &op.a, a_index)?;
        let mut body = Block::new();
        Self::push_emits(&mut body, a_emits);
        Self::push_emits(&mut body, a_ptr_emits);
        let a = expressions.append(Expression::Load { pointer: a_ptr }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, a)),
            Span::default(),
        );
        let (b, b_emits) = self.dequantize_qvalue(expressions, &op.b, k, col)?;
        Self::push_emits(&mut body, b_emits);
        let sum_ptr =
            expressions.append(Expression::LocalVariable(scratch.mma_sum), Span::default());
        let sum = expressions.append(Expression::Load { pointer: sum_ptr }, Span::default());
        let value = expressions.append(
            Expression::Math {
                fun: MathFunction::Fma,
                arg: a,
                arg1: Some(b),
                arg2: Some(sum),
                arg3: None,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::range_from(expressions, sum, value)),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: sum_ptr,
                value,
            },
            Span::default(),
        );
        Ok(body)
    }

    fn dequantize_qvalue(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let mut emits = Vec::new();
        let block_elems = matrix.format.block_elements();
        let block_words = matrix.format.block_words();
        let block = self.div_literal_u32_emitted(expressions, k, block_elems, &mut emits);
        let q = self.and_lit(expressions, &mut emits, k, block_elems - 1);
        let col_block =
            self.mul_literal_u32_emitted(expressions, col, matrix.rows / block_elems, &mut emits);
        let block_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_block,
            block,
        );
        let base = self.mul_literal_u32_emitted(expressions, block_index, block_words, &mut emits);
        let value = match matrix.format {
            GgmlQuantFormat::Q4_0 => self.dequant_q4_0(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q4_1 => self.dequant_q4_1(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q5_0 => self.dequant_q5_0(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q5_1 => self.dequant_q5_1(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q8_0 => self.dequant_q8_0(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q8_1 => self.dequant_q8_1(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q2K => self.dequant_q2k(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q3K => self.dequant_q3k(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q4K => self.dequant_q4k(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q5K => self.dequant_q5k(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q6K => self.dequant_q6k(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q8K => self.dequant_q8k(expressions, matrix, base, q, &mut emits)?,
        };
        Ok((value, emits))
    }

    fn dequantize_qvalues4(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<DequantizedValues4, LowerError> {
        let mut emits = Vec::new();
        let block_elems = matrix.format.block_elements();
        let block_words = matrix.format.block_words();
        let block = self.div_literal_u32_emitted(expressions, k_base, block_elems, &mut emits);
        let q = self.and_lit(expressions, &mut emits, k_base, block_elems - 1);
        let col_block =
            self.mul_literal_u32_emitted(expressions, col, matrix.rows / block_elems, &mut emits);
        let block_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_block,
            block,
        );
        let base = self.mul_literal_u32_emitted(expressions, block_index, block_words, &mut emits);
        let values = match matrix.format {
            GgmlQuantFormat::Q4_0 => {
                self.dequant_q4_0x4(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q4_1 => {
                self.dequant_q4_1x4(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q5_0 => {
                self.dequant_q5_0x4(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q5_1 => {
                self.dequant_q5_1x4(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q8_0 => {
                self.dequant_q8_0x4(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q8_1 => {
                self.dequant_q8_1x4(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q2K => self.dequant_q2kx4(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q3K => self.dequant_q3kx4(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q4K => {
                self.dequant_k_nibblex4(expressions, matrix, base, q, &mut emits, false)?
            }
            GgmlQuantFormat::Q5K => {
                self.dequant_k_nibblex4(expressions, matrix, base, q, &mut emits, true)?
            }
            GgmlQuantFormat::Q6K => self.dequant_q6kx4(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q8K => self.dequant_q8kx4(expressions, matrix, base, q, &mut emits)?,
        };
        Ok((values, emits))
    }

    fn dequantize_qvalues8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<DequantizedValues4, LowerError> {
        let mut emits = Vec::new();
        let block_elems = matrix.format.block_elements();
        let block_words = matrix.format.block_words();
        let block = self.div_literal_u32_emitted(expressions, k_base, block_elems, &mut emits);
        let q = self.and_lit(expressions, &mut emits, k_base, block_elems - 1);
        let col_block =
            self.mul_literal_u32_emitted(expressions, col, matrix.rows / block_elems, &mut emits);
        let block_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_block,
            block,
        );
        let base = self.mul_literal_u32_emitted(expressions, block_index, block_words, &mut emits);
        let values = match matrix.format {
            GgmlQuantFormat::Q4_0 => {
                self.dequant_q4_0x8(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q4_1 => {
                self.dequant_q4_1x8(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q5_0 => {
                self.dequant_q5_0x8(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q5_1 => {
                self.dequant_q5_1x8(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q8_0 => {
                self.dequant_q8_0x8(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q8_1 => {
                self.dequant_q8_1x8(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q2K => self.dequant_q2kx8(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q3K => self.dequant_q3kx8(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q4K => {
                self.dequant_k_nibblex8(expressions, matrix, base, q, &mut emits, false)?
            }
            GgmlQuantFormat::Q5K => {
                self.dequant_k_nibblex8(expressions, matrix, base, q, &mut emits, true)?
            }
            GgmlQuantFormat::Q6K => self.dequant_q6kx8(expressions, matrix, base, q, &mut emits)?,
            GgmlQuantFormat::Q8K => self.dequant_q8kx8(expressions, matrix, base, q, &mut emits)?,
        };
        Ok((values, emits))
    }

    fn dequantize_qvalues16(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<DequantizedValues4, LowerError> {
        let mut emits = Vec::new();
        let block_elems = matrix.format.block_elements();
        let block_words = matrix.format.block_words();
        let block = self.div_literal_u32_emitted(expressions, k_base, block_elems, &mut emits);
        let q = self.and_lit(expressions, &mut emits, k_base, block_elems - 1);
        let col_block =
            self.mul_literal_u32_emitted(expressions, col, matrix.rows / block_elems, &mut emits);
        let block_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_block,
            block,
        );
        let base = self.mul_literal_u32_emitted(expressions, block_index, block_words, &mut emits);
        let values = match matrix.format {
            GgmlQuantFormat::Q2K => {
                self.dequant_q2kx16(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q3K => {
                self.dequant_q3kx16(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q4K => {
                self.dequant_k_nibblex16(expressions, matrix, base, q, &mut emits, false)?
            }
            GgmlQuantFormat::Q5K => {
                self.dequant_k_nibblex16(expressions, matrix, base, q, &mut emits, true)?
            }
            GgmlQuantFormat::Q6K => {
                self.dequant_q6kx16(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q8K => {
                self.dequant_q8kx16(expressions, matrix, base, q, &mut emits)?
            }
            GgmlQuantFormat::Q4_0
            | GgmlQuantFormat::Q4_1
            | GgmlQuantFormat::Q5_0
            | GgmlQuantFormat::Q5_1
            | GgmlQuantFormat::Q8_0
            | GgmlQuantFormat::Q8_1 => {
                return Err(LowerError::UnsupportedOperation(
                    "x16 dequant only supports K qmatmul block formats",
                ));
            }
        };
        Ok((values, emits))
    }

    fn dequant_q4_0x4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 1);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let bytes = self.unpack4x_u8(e, emits, word);
        let mask = self.u32(e, 0x0f);
        let mask = self.splat4(e, emits, mask);
        let low = self.bin(e, emits, BinaryOperator::And, bytes, mask);
        let shift = self.u32(e, 4);
        let shift = self.splat4(e, emits, shift);
        let high_q = self.shr(e, emits, bytes, shift);
        let quant = self.select(e, emits, high, high_q, low);
        let quant_f = self.as_f32(e, emits, quant);
        let center = self.f32(e, 8.0);
        let center = self.splat4(e, emits, center);
        let centered = self.sub(e, emits, quant_f, center);
        let scale = self.splat4(e, emits, scale);
        let result = self.mul(e, emits, centered, scale);
        Ok((0..4)
            .map(|lane| self.vec4_component(e, emits, result, lane))
            .collect())
    }

    fn dequant_q4_0x8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 1);
        let word0 = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let word1_off = self.add_lit(e, emits, word_off, 1);
        let word1 = self.load_word_dynamic(e, matrix, base, word1_off, emits)?;
        let scale = self.splat4(e, emits, scale);
        let center = self.f32(e, 8.0);
        let center = self.splat4(e, emits, center);
        let mut values = Vec::with_capacity(8);
        for word in [word0, word1] {
            let bytes = self.unpack4x_u8(e, emits, word);
            let mask = self.u32(e, 0x0f);
            let mask = self.splat4(e, emits, mask);
            let low = self.bin(e, emits, BinaryOperator::And, bytes, mask);
            let shift = self.u32(e, 4);
            let shift = self.splat4(e, emits, shift);
            let high_q = self.shr(e, emits, bytes, shift);
            let quant = self.select(e, emits, high, high_q, low);
            let quant_f = self.as_f32(e, emits, quant);
            let centered = self.sub(e, emits, quant_f, center);
            let result = self.mul(e, emits, centered, scale);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q5_0x4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let qh = self.load_word(e, matrix, base, 1, emits)?;
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 2);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let high_index = self.add_lit(e, emits, q_local, 16);
        let hi_bit_base = self.select(e, emits, high, high_index, q_local);
        let bytes = self.unpack4x_u8(e, emits, word);
        let mask = self.u32(e, 0x0f);
        let mask = self.splat4(e, emits, mask);
        let low = self.bin(e, emits, BinaryOperator::And, bytes, mask);
        let shift = self.u32(e, 4);
        let shift = self.splat4(e, emits, shift);
        let high4 = self.shr(e, emits, bytes, shift);
        let low4 = self.select(e, emits, high, high4, low);
        let offsets = self.u32_vec4(e, emits, [0, 1, 2, 3]);
        let hi_base = self.splat4(e, emits, hi_bit_base);
        let hi_bit_index = self.bin(e, emits, BinaryOperator::Add, hi_base, offsets);
        let qh_vec = self.splat4(e, emits, qh);
        let shifted_qh = self.shr(e, emits, qh_vec, hi_bit_index);
        let one = self.u32(e, 1);
        let one = self.splat4(e, emits, one);
        let hi_bit = self.bin(e, emits, BinaryOperator::And, shifted_qh, one);
        let four = self.u32(e, 4);
        let four = self.splat4(e, emits, four);
        let hi_bit = self.bin(e, emits, BinaryOperator::ShiftLeft, hi_bit, four);
        let quant = self.bin(e, emits, BinaryOperator::InclusiveOr, low4, hi_bit);
        let quant_f = self.as_f32(e, emits, quant);
        let center = self.f32(e, 16.0);
        let center = self.splat4(e, emits, center);
        let centered = self.sub(e, emits, quant_f, center);
        let scale = self.splat4(e, emits, scale);
        let result = self.mul(e, emits, centered, scale);
        Ok((0..4)
            .map(|lane| self.vec4_component(e, emits, result, lane))
            .collect())
    }

    fn dequant_q5_0x8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let qh = self.load_word(e, matrix, base, 1, emits)?;
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 2);
        let word0 = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let word1_off = self.add_lit(e, emits, word_off, 1);
        let word1 = self.load_word_dynamic(e, matrix, base, word1_off, emits)?;
        let high_index = self.add_lit(e, emits, q_local, 16);
        let hi_bit_base = self.select(e, emits, high, high_index, q_local);
        let scale = self.splat4(e, emits, scale);
        let center = self.f32(e, 16.0);
        let center = self.splat4(e, emits, center);
        let qh_vec = self.splat4(e, emits, qh);
        let mut values = Vec::with_capacity(8);
        for (word, offset_base) in [(word0, [0, 1, 2, 3]), (word1, [4, 5, 6, 7])] {
            let bytes = self.unpack4x_u8(e, emits, word);
            let mask = self.u32(e, 0x0f);
            let mask = self.splat4(e, emits, mask);
            let low = self.bin(e, emits, BinaryOperator::And, bytes, mask);
            let shift = self.u32(e, 4);
            let shift = self.splat4(e, emits, shift);
            let high4 = self.shr(e, emits, bytes, shift);
            let low4 = self.select(e, emits, high, high4, low);
            let offsets = self.u32_vec4(e, emits, offset_base);
            let hi_base = self.splat4(e, emits, hi_bit_base);
            let hi_bit_index = self.bin(e, emits, BinaryOperator::Add, hi_base, offsets);
            let shifted_qh = self.shr(e, emits, qh_vec, hi_bit_index);
            let one = self.u32(e, 1);
            let one = self.splat4(e, emits, one);
            let hi_bit = self.bin(e, emits, BinaryOperator::And, shifted_qh, one);
            let four = self.u32(e, 4);
            let four = self.splat4(e, emits, four);
            let hi_bit = self.bin(e, emits, BinaryOperator::ShiftLeft, hi_bit, four);
            let quant = self.bin(e, emits, BinaryOperator::InclusiveOr, low4, hi_bit);
            let quant_f = self.as_f32(e, emits, quant);
            let centered = self.sub(e, emits, quant_f, center);
            let result = self.mul(e, emits, centered, scale);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q8_0x4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let q_word = self.shr_lit(e, emits, q, 2);
        let word_off = self.add_lit(e, emits, q_word, 1);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let signed = self.unpack4x_i8(e, emits, word);
        let signed = self.as_f32(e, emits, signed);
        let scale = self.splat4(e, emits, scale);
        let result = self.mul(e, emits, signed, scale);
        Ok((0..4)
            .map(|lane| self.vec4_component(e, emits, result, lane))
            .collect())
    }

    fn dequant_q8_0x8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let q_word = self.shr_lit(e, emits, q, 2);
        let word_off = self.add_lit(e, emits, q_word, 1);
        let word0 = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let word1_off = self.add_lit(e, emits, word_off, 1);
        let word1 = self.load_word_dynamic(e, matrix, base, word1_off, emits)?;
        let scale = self.splat4(e, emits, scale);
        let mut values = Vec::with_capacity(8);
        for word in [word0, word1] {
            let signed = self.unpack4x_i8(e, emits, word);
            let signed = self.as_f32(e, emits, signed);
            let result = self.mul(e, emits, signed, scale);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q4_1x4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q4_1x(e, matrix, base, q, emits, 4)
    }

    fn dequant_q4_1x8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q4_1x(e, matrix, base, q, emits, 8)
    }

    fn dequant_q4_1x(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        count: u32,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let min_word = self.load_word(e, matrix, base, 1, emits)?;
        let min = self.bitcast_f32(e, emits, min_word);
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 2);
        let scale = self.splat4(e, emits, scale);
        let min = self.splat4(e, emits, min);
        let mut values = Vec::with_capacity(count as usize);
        for word_index in 0..(count / 4) {
            let off = if word_index == 0 {
                word_off
            } else {
                self.add_lit(e, emits, word_off, word_index)
            };
            let word = self.load_word_dynamic(e, matrix, base, off, emits)?;
            let bytes = self.unpack4x_u8(e, emits, word);
            let mask = self.u32(e, 0x0f);
            let mask = self.splat4(e, emits, mask);
            let low = self.bin(e, emits, BinaryOperator::And, bytes, mask);
            let shift = self.u32(e, 4);
            let shift = self.splat4(e, emits, shift);
            let high_q = self.shr(e, emits, bytes, shift);
            let quant = self.select(e, emits, high, high_q, low);
            let quant_f = self.as_f32(e, emits, quant);
            let scaled = self.mul(e, emits, quant_f, scale);
            let result = self.bin(e, emits, BinaryOperator::Add, scaled, min);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q5_1x4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q5_1x(e, matrix, base, q, emits, 4)
    }

    fn dequant_q5_1x8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q5_1x(e, matrix, base, q, emits, 8)
    }

    fn dequant_q5_1x(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        count: u32,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let min_word = self.load_word(e, matrix, base, 1, emits)?;
        let min = self.bitcast_f32(e, emits, min_word);
        let qh = self.load_word(e, matrix, base, 2, emits)?;
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 3);
        let high_index = self.add_lit(e, emits, q_local, 16);
        let hi_bit_base = self.select(e, emits, high, high_index, q_local);
        let scale = self.splat4(e, emits, scale);
        let min = self.splat4(e, emits, min);
        let qh_vec = self.splat4(e, emits, qh);
        let mut values = Vec::with_capacity(count as usize);
        for word_index in 0..(count / 4) {
            let off = if word_index == 0 {
                word_off
            } else {
                self.add_lit(e, emits, word_off, word_index)
            };
            let word = self.load_word_dynamic(e, matrix, base, off, emits)?;
            let bytes = self.unpack4x_u8(e, emits, word);
            let mask = self.u32(e, 0x0f);
            let mask = self.splat4(e, emits, mask);
            let low = self.bin(e, emits, BinaryOperator::And, bytes, mask);
            let shift = self.u32(e, 4);
            let shift = self.splat4(e, emits, shift);
            let high4 = self.shr(e, emits, bytes, shift);
            let low4 = self.select(e, emits, high, high4, low);
            let offsets = self.u32_vec4(
                e,
                emits,
                [
                    word_index * 4,
                    word_index * 4 + 1,
                    word_index * 4 + 2,
                    word_index * 4 + 3,
                ],
            );
            let hi_base = self.splat4(e, emits, hi_bit_base);
            let hi_bit_index = self.bin(e, emits, BinaryOperator::Add, hi_base, offsets);
            let shifted_qh = self.shr(e, emits, qh_vec, hi_bit_index);
            let one = self.u32(e, 1);
            let one = self.splat4(e, emits, one);
            let hi_bit = self.bin(e, emits, BinaryOperator::And, shifted_qh, one);
            let four = self.u32(e, 4);
            let four = self.splat4(e, emits, four);
            let hi_bit = self.bin(e, emits, BinaryOperator::ShiftLeft, hi_bit, four);
            let quant = self.bin(e, emits, BinaryOperator::InclusiveOr, low4, hi_bit);
            let quant_f = self.as_f32(e, emits, quant);
            let scaled = self.mul(e, emits, quant_f, scale);
            let result = self.bin(e, emits, BinaryOperator::Add, scaled, min);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q8_1x4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q8_1x(e, matrix, base, q, emits, 4)
    }

    fn dequant_q8_1x8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q8_1x(e, matrix, base, q, emits, 8)
    }

    fn dequant_q8_1x(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        count: u32,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let q_word = self.shr_lit(e, emits, q, 2);
        let word_off = self.add_lit(e, emits, q_word, 2);
        let scale = self.splat4(e, emits, scale);
        let mut values = Vec::with_capacity(count as usize);
        for word_index in 0..(count / 4) {
            let off = if word_index == 0 {
                word_off
            } else {
                self.add_lit(e, emits, word_off, word_index)
            };
            let word = self.load_word_dynamic(e, matrix, base, off, emits)?;
            let signed = self.unpack4x_i8(e, emits, word);
            let signed = self.as_f32(e, emits, signed);
            let result = self.mul(e, emits, signed, scale);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q2kx4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q2kx(e, matrix, base, q, emits, 4)
    }

    fn dequant_q2kx8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q2kx(e, matrix, base, q, emits, 8)
    }

    fn dequant_q2kx16(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q2kx(e, matrix, base, q, emits, 16)
    }

    fn dequant_q2kx(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        count: u32,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 20, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let dmin_word = self.load_word(e, matrix, base, 21, emits)?;
        let dmin = self.bitcast_f32(e, emits, dmin_word);
        let group = self.shr_lit(e, emits, q, 4);
        let scale_word = self.shr_lit(e, emits, group, 2);
        let scale_word = self.load_word_dynamic(e, matrix, base, scale_word, emits)?;
        let scale_lane = self.and_lit(e, emits, group, 3);
        let scale_byte = self.byte_at(e, emits, scale_word, scale_lane);
        let scale_quant = self.and_lit(e, emits, scale_byte, 0x0f);
        let scale_quant_f = self.as_f32(e, emits, scale_quant);
        let scale = self.mul(e, emits, scale_quant_f, d);
        let min_quant = self.shr_lit(e, emits, scale_byte, 4);
        let min_quant_f = self.as_f32(e, emits, min_quant);
        let min = self.mul(e, emits, min_quant_f, dmin);
        let q_local = self.and_lit(e, emits, q, 15);
        let chunk = self.shr_lit(e, emits, group, 3);
        let group_in_chunk = self.and_lit(e, emits, group, 7);
        let pair = self.and_lit(e, emits, group_in_chunk, 1);
        let byte_base = self.shl_lit(e, emits, chunk, 5);
        let pair_offset = self.shl_lit(e, emits, pair, 4);
        let byte_base = self.bin(e, emits, BinaryOperator::Add, byte_base, pair_offset);
        let byte_base = self.bin(e, emits, BinaryOperator::Add, byte_base, q_local);
        let word_off = self.shr_lit(e, emits, byte_base, 2);
        let word_off = self.add_lit(e, emits, word_off, 4);
        let shift = self.shr_lit(e, emits, group_in_chunk, 1);
        let shift = self.shl_lit(e, emits, shift, 1);
        let shift = self.splat4(e, emits, shift);
        let scale = self.splat4(e, emits, scale);
        let min = self.splat4(e, emits, min);
        let mask = self.u32(e, 3);
        let mask = self.splat4(e, emits, mask);
        let mut values = Vec::with_capacity(count as usize);
        for word_index in 0..(count / 4) {
            let off = if word_index == 0 {
                word_off
            } else {
                self.add_lit(e, emits, word_off, word_index)
            };
            let word = self.load_word_dynamic(e, matrix, base, off, emits)?;
            let bytes = self.unpack4x_u8(e, emits, word);
            let shifted = self.shr(e, emits, bytes, shift);
            let quant = self.bin(e, emits, BinaryOperator::And, shifted, mask);
            let quant_f = self.as_f32(e, emits, quant);
            let scaled = self.mul(e, emits, quant_f, scale);
            let result = self.sub(e, emits, scaled, min);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q3kx4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q3kx(e, matrix, base, q, emits, 4)
    }

    fn dequant_q3kx8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q3kx(e, matrix, base, q, emits, 8)
    }

    fn dequant_q3kx16(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q3kx(e, matrix, base, q, emits, 16)
    }

    fn dequant_q3kx(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        count: u32,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 27, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let group = self.shr_lit(e, emits, q, 4);
        let scale_quant = self.q3k_scale(e, matrix, base, group, emits)?;
        let scale_f = self.as_f32(e, emits, scale_quant);
        let center = self.f32(e, 32.0);
        let scale_f = self.sub(e, emits, scale_f, center);
        let scale = self.mul(e, emits, scale_f, d);
        let q_local = self.and_lit(e, emits, q, 15);
        let chunk = self.shr_lit(e, emits, group, 3);
        let group_in_chunk = self.and_lit(e, emits, group, 7);
        let pair = self.and_lit(e, emits, group_in_chunk, 1);
        let byte_base = self.shl_lit(e, emits, chunk, 5);
        let pair_offset = self.shl_lit(e, emits, pair, 4);
        let byte_base = self.bin(e, emits, BinaryOperator::Add, byte_base, pair_offset);
        let byte_base = self.bin(e, emits, BinaryOperator::Add, byte_base, q_local);
        let word_off = self.shr_lit(e, emits, byte_base, 2);
        let word_off = self.add_lit(e, emits, word_off, 8);
        let hmask_base = self.bin(e, emits, BinaryOperator::Add, pair_offset, q_local);
        let hmask_word_off = self.shr_lit(e, emits, hmask_base, 2);
        let hmask_bit_pair = self.shr_lit(e, emits, group_in_chunk, 1);
        let chunk_mask_base = self.shl_lit(e, emits, chunk, 2);
        let hmask_bit = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            chunk_mask_base,
            hmask_bit_pair,
        );
        let one = self.u32(e, 1);
        let hmask = self.bin(e, emits, BinaryOperator::ShiftLeft, one, hmask_bit);
        let hmask = self.splat4(e, emits, hmask);
        let shift = self.shl_lit(e, emits, hmask_bit_pair, 1);
        let shift = self.splat4(e, emits, shift);
        let scale = self.splat4(e, emits, scale);
        let mask = self.u32(e, 3);
        let mask = self.splat4(e, emits, mask);
        let zero_u = self.u32(e, 0);
        let zero_u = self.splat4(e, emits, zero_u);
        let zero = self.f32(e, 0.0);
        let zero = self.splat4(e, emits, zero);
        let four = self.f32(e, 4.0);
        let four = self.splat4(e, emits, four);
        let mut values = Vec::with_capacity(count as usize);
        for word_index in 0..(count / 4) {
            let q_off = if word_index == 0 {
                word_off
            } else {
                self.add_lit(e, emits, word_off, word_index)
            };
            let h_off = if word_index == 0 {
                hmask_word_off
            } else {
                self.add_lit(e, emits, hmask_word_off, word_index)
            };
            let word = self.load_word_dynamic(e, matrix, base, q_off, emits)?;
            let hword = self.load_word_dynamic(e, matrix, base, h_off, emits)?;
            let bytes = self.unpack4x_u8(e, emits, word);
            let shifted = self.shr(e, emits, bytes, shift);
            let low = self.bin(e, emits, BinaryOperator::And, shifted, mask);
            let low_f = self.as_f32(e, emits, low);
            let hbytes = self.unpack4x_u8(e, emits, hword);
            let high_bits = self.bin(e, emits, BinaryOperator::And, hbytes, hmask);
            let high_set = self.bin(e, emits, BinaryOperator::NotEqual, high_bits, zero_u);
            let penalty = self.select(e, emits, high_set, zero, four);
            let centered = self.sub(e, emits, low_f, penalty);
            let result = self.mul(e, emits, centered, scale);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q8kx4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q8kx(e, matrix, base, q, emits, 4)
    }

    fn dequant_q8kx8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q8kx(e, matrix, base, q, emits, 8)
    }

    fn dequant_q8kx16(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        self.dequant_q8kx(e, matrix, base, q, emits, 16)
    }

    fn dequant_q8kx(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        count: u32,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let q_word = self.shr_lit(e, emits, q, 2);
        let word_off = self.add_lit(e, emits, q_word, 1);
        let scale = self.splat4(e, emits, scale);
        let mut values = Vec::with_capacity(count as usize);
        for word_index in 0..(count / 4) {
            let off = if word_index == 0 {
                word_off
            } else {
                self.add_lit(e, emits, word_off, word_index)
            };
            let word = self.load_word_dynamic(e, matrix, base, off, emits)?;
            let signed = self.unpack4x_i8(e, emits, word);
            let signed = self.as_f32(e, emits, signed);
            let result = self.mul(e, emits, signed, scale);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_k_nibblex4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        q5: bool,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 0, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let dmin_word = self.load_word(e, matrix, base, 1, emits)?;
        let dmin = self.bitcast_f32(e, emits, dmin_word);
        let group = self.shr_lit(e, emits, q, 5);
        let (scale_byte, min_byte) = self.k_scale_pair(e, matrix, base, group, emits)?;
        let scale_f = self.as_f32(e, emits, scale_byte);
        let scale = self.mul(e, emits, scale_f, d);
        let min_f = self.as_f32(e, emits, min_byte);
        let min = self.mul(e, emits, min_f, dmin);
        let in_group = self.and_lit(e, emits, q, 31);
        let group_pair = self.shr_lit(e, emits, group, 1);
        let group_pair_offset = self.shl_lit(e, emits, group_pair, 5);
        let byte_index = self.bin(e, emits, BinaryOperator::Add, group_pair_offset, in_group);
        let data_base = if q5 { 13 } else { 5 };
        let data_word = self.shr_lit(e, emits, byte_index, 2);
        let data_off = self.add_lit(e, emits, data_word, data_base);
        let word = self.load_word_dynamic(e, matrix, base, data_off, emits)?;
        let group_low = self.and_lit(e, emits, group, 1);
        let high = self.cmp_lit(e, emits, BinaryOperator::NotEqual, group_low, 0);
        let qh_word = if q5 {
            let qh_byte_index = self.and_lit(e, emits, q, 31);
            let qh_word = self.shr_lit(e, emits, qh_byte_index, 2);
            let qh_off = self.add_lit(e, emits, qh_word, 5);
            Some(self.load_word_dynamic(e, matrix, base, qh_off, emits)?)
        } else {
            None
        };
        let qh_bit_index = q5.then(|| self.shr_lit(e, emits, q, 5));

        let bytes = self.unpack4x_u8(e, emits, word);
        let shift = self.u32(e, 4);
        let shift = self.splat4(e, emits, shift);
        let byte_hi = self.shr(e, emits, bytes, shift);
        let mask = self.u32(e, 0x0f);
        let mask = self.splat4(e, emits, mask);
        let byte_lo = self.bin(e, emits, BinaryOperator::And, bytes, mask);
        let mut quant = self.select(e, emits, high, byte_hi, byte_lo);
        if let (Some(qh_word), Some(qh_bit_index)) = (qh_word, qh_bit_index) {
            let qh_bytes = self.unpack4x_u8(e, emits, qh_word);
            let qh_bit_index = self.splat4(e, emits, qh_bit_index);
            let shifted_qh = self.shr(e, emits, qh_bytes, qh_bit_index);
            let one = self.u32(e, 1);
            let one = self.splat4(e, emits, one);
            let bit = self.bin(e, emits, BinaryOperator::And, shifted_qh, one);
            let four = self.u32(e, 4);
            let four = self.splat4(e, emits, four);
            let bit = self.bin(e, emits, BinaryOperator::ShiftLeft, bit, four);
            quant = self.bin(e, emits, BinaryOperator::InclusiveOr, quant, bit);
        }
        let quant_f = self.as_f32(e, emits, quant);
        let scale = self.splat4(e, emits, scale);
        let scaled = self.mul(e, emits, quant_f, scale);
        let min = self.splat4(e, emits, min);
        let result = self.sub(e, emits, scaled, min);
        Ok((0..4)
            .map(|lane| self.vec4_component(e, emits, result, lane))
            .collect())
    }

    fn dequant_q6kx4(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 52, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let chunk = self.shr_lit(e, emits, q, 7);
        let local = self.and_lit(e, emits, q, 127);
        let high_byte_index = self.and_lit(e, emits, local, 31);
        let low_group = self.shr_lit(e, emits, local, 5);
        let chunk_low_base = self.shl_lit(e, emits, chunk, 6);
        let low_group_parity = self.and_lit(e, emits, low_group, 1);
        let low_group_offset = self.shl_lit(e, emits, low_group_parity, 5);
        let low_nibble_shift = self.shr_lit(e, emits, low_group, 1);
        let low_nibble_shift = self.shl_lit(e, emits, low_nibble_shift, 2);
        let low_nibble_shift = self.splat4(e, emits, low_nibble_shift);
        let local_low_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_byte_index,
            low_group_offset,
        );
        let lower_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            chunk_low_base,
            local_low_index,
        );
        let low_word_off = self.shr_lit(e, emits, lower_index, 2);
        let low_word = self.load_word_dynamic(e, matrix, base, low_word_off, emits)?;
        let high_chunk_base = self.shl_lit(e, emits, chunk, 5);
        let high_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_chunk_base,
            high_byte_index,
        );
        let high_word_base = self.shr_lit(e, emits, high_index, 2);
        let high_word_off = self.add_lit(e, emits, high_word_base, 32);
        let high_word = self.load_word_dynamic(e, matrix, base, high_word_off, emits)?;
        let high_shift = self.shl_lit(e, emits, low_group, 1);
        let scale_chunk_base = self.shl_lit(e, emits, chunk, 3);
        let high_byte_half = self.shr_lit(e, emits, high_byte_index, 4);
        let low_group_scale = self.shl_lit(e, emits, low_group, 1);
        let local_scale_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_byte_half,
            low_group_scale,
        );
        let scale_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            scale_chunk_base,
            local_scale_index,
        );
        let scale_word_base = self.shr_lit(e, emits, scale_index, 2);
        let scale_word_off = self.add_lit(e, emits, scale_word_base, 48);
        let scale_word = self.load_word_dynamic(e, matrix, base, scale_word_off, emits)?;
        let scale_lane = self.and_lit(e, emits, scale_index, 3);
        let scale_byte = self.byte_at(e, emits, scale_word, scale_lane);
        let scale = self.signed_byte_f32(e, emits, scale_byte);
        let scale = self.mul(e, emits, scale, d);
        let low_bytes = self.unpack4x_u8(e, emits, low_word);
        let shift4 = self.u32(e, 4);
        let shift4 = self.splat4(e, emits, shift4);
        let mask4 = self.u32(e, 0x0f);
        let mask4 = self.splat4(e, emits, mask4);
        let low_shifted = self.shr(e, emits, low_bytes, low_nibble_shift);
        let low4 = self.bin(e, emits, BinaryOperator::And, low_shifted, mask4);

        let high_bytes = self.unpack4x_u8(e, emits, high_word);
        let high_shift = self.splat4(e, emits, high_shift);
        let high_shifted = self.shr(e, emits, high_bytes, high_shift);
        let mask2 = self.u32(e, 3);
        let mask2 = self.splat4(e, emits, mask2);
        let high2 = self.bin(e, emits, BinaryOperator::And, high_shifted, mask2);
        let high2 = self.bin(e, emits, BinaryOperator::ShiftLeft, high2, shift4);
        let quant = self.bin(e, emits, BinaryOperator::InclusiveOr, low4, high2);
        let quant_f = self.as_f32(e, emits, quant);
        let center = self.f32(e, 32.0);
        let center = self.splat4(e, emits, center);
        let centered = self.sub(e, emits, quant_f, center);
        let scale = self.splat4(e, emits, scale);
        let result = self.mul(e, emits, centered, scale);
        Ok((0..4)
            .map(|lane| self.vec4_component(e, emits, result, lane))
            .collect())
    }

    fn dequant_k_nibblex8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        q5: bool,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 0, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let dmin_word = self.load_word(e, matrix, base, 1, emits)?;
        let dmin = self.bitcast_f32(e, emits, dmin_word);
        let group = self.shr_lit(e, emits, q, 5);
        let (scale_byte, min_byte) = self.k_scale_pair(e, matrix, base, group, emits)?;
        let scale_f = self.as_f32(e, emits, scale_byte);
        let scale = self.mul(e, emits, scale_f, d);
        let scale = self.splat4(e, emits, scale);
        let min_f = self.as_f32(e, emits, min_byte);
        let min = self.mul(e, emits, min_f, dmin);
        let min = self.splat4(e, emits, min);

        let in_group = self.and_lit(e, emits, q, 31);
        let group_pair = self.shr_lit(e, emits, group, 1);
        let group_pair_offset = self.shl_lit(e, emits, group_pair, 5);
        let byte_index = self.bin(e, emits, BinaryOperator::Add, group_pair_offset, in_group);
        let data_base = if q5 { 13 } else { 5 };
        let data_word = self.shr_lit(e, emits, byte_index, 2);
        let data_off = self.add_lit(e, emits, data_word, data_base);
        let word0 = self.load_word_dynamic(e, matrix, base, data_off, emits)?;
        let word1_off = self.add_lit(e, emits, data_off, 1);
        let word1 = self.load_word_dynamic(e, matrix, base, word1_off, emits)?;

        let group_low = self.and_lit(e, emits, group, 1);
        let high = self.cmp_lit(e, emits, BinaryOperator::NotEqual, group_low, 0);
        let qh_words = if q5 {
            let qh_byte_index = self.and_lit(e, emits, q, 31);
            let qh_word = self.shr_lit(e, emits, qh_byte_index, 2);
            let qh_off = self.add_lit(e, emits, qh_word, 5);
            let qh0 = self.load_word_dynamic(e, matrix, base, qh_off, emits)?;
            let qh1_off = self.add_lit(e, emits, qh_off, 1);
            let qh1 = self.load_word_dynamic(e, matrix, base, qh1_off, emits)?;
            Some([qh0, qh1])
        } else {
            None
        };
        let qh_bit_index = q5.then(|| self.shr_lit(e, emits, q, 5));

        let mut values = Vec::with_capacity(8);
        for (word_index, word) in [word0, word1].into_iter().enumerate() {
            let bytes = self.unpack4x_u8(e, emits, word);
            let shift = self.u32(e, 4);
            let shift = self.splat4(e, emits, shift);
            let byte_hi = self.shr(e, emits, bytes, shift);
            let mask = self.u32(e, 0x0f);
            let mask = self.splat4(e, emits, mask);
            let byte_lo = self.bin(e, emits, BinaryOperator::And, bytes, mask);
            let mut quant = self.select(e, emits, high, byte_hi, byte_lo);
            if let (Some(qh_words), Some(qh_bit_index)) = (qh_words, qh_bit_index) {
                let qh_bytes = self.unpack4x_u8(e, emits, qh_words[word_index]);
                let qh_bit_index = self.splat4(e, emits, qh_bit_index);
                let shifted_qh = self.shr(e, emits, qh_bytes, qh_bit_index);
                let one = self.u32(e, 1);
                let one = self.splat4(e, emits, one);
                let bit = self.bin(e, emits, BinaryOperator::And, shifted_qh, one);
                let four = self.u32(e, 4);
                let four = self.splat4(e, emits, four);
                let bit = self.bin(e, emits, BinaryOperator::ShiftLeft, bit, four);
                quant = self.bin(e, emits, BinaryOperator::InclusiveOr, quant, bit);
            }
            let quant_f = self.as_f32(e, emits, quant);
            let scaled = self.mul(e, emits, quant_f, scale);
            let result = self.sub(e, emits, scaled, min);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q6kx8(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 52, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let chunk = self.shr_lit(e, emits, q, 7);
        let local = self.and_lit(e, emits, q, 127);
        let high_byte_index = self.and_lit(e, emits, local, 31);
        let low_group = self.shr_lit(e, emits, local, 5);
        let chunk_low_base = self.shl_lit(e, emits, chunk, 6);
        let low_group_parity = self.and_lit(e, emits, low_group, 1);
        let low_group_offset = self.shl_lit(e, emits, low_group_parity, 5);
        let low_nibble_shift = self.shr_lit(e, emits, low_group, 1);
        let low_nibble_shift = self.shl_lit(e, emits, low_nibble_shift, 2);
        let low_nibble_shift = self.splat4(e, emits, low_nibble_shift);
        let local_low_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_byte_index,
            low_group_offset,
        );
        let lower_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            chunk_low_base,
            local_low_index,
        );
        let low_word_off = self.shr_lit(e, emits, lower_index, 2);
        let low_word0 = self.load_word_dynamic(e, matrix, base, low_word_off, emits)?;
        let low_word1_off = self.add_lit(e, emits, low_word_off, 1);
        let low_word1 = self.load_word_dynamic(e, matrix, base, low_word1_off, emits)?;
        let high_chunk_base = self.shl_lit(e, emits, chunk, 5);
        let high_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_chunk_base,
            high_byte_index,
        );
        let high_word_base = self.shr_lit(e, emits, high_index, 2);
        let high_word_off = self.add_lit(e, emits, high_word_base, 32);
        let high_word0 = self.load_word_dynamic(e, matrix, base, high_word_off, emits)?;
        let high_word1_off = self.add_lit(e, emits, high_word_off, 1);
        let high_word1 = self.load_word_dynamic(e, matrix, base, high_word1_off, emits)?;
        let high_shift = self.shl_lit(e, emits, low_group, 1);
        let high_shift = self.splat4(e, emits, high_shift);

        let scale_chunk_base = self.shl_lit(e, emits, chunk, 3);
        let high_byte_half = self.shr_lit(e, emits, high_byte_index, 4);
        let low_group_scale = self.shl_lit(e, emits, low_group, 1);
        let local_scale_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_byte_half,
            low_group_scale,
        );
        let scale_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            scale_chunk_base,
            local_scale_index,
        );
        let scale_word_base = self.shr_lit(e, emits, scale_index, 2);
        let scale_word_off = self.add_lit(e, emits, scale_word_base, 48);
        let scale_word = self.load_word_dynamic(e, matrix, base, scale_word_off, emits)?;
        let scale_lane = self.and_lit(e, emits, scale_index, 3);
        let scale_byte = self.byte_at(e, emits, scale_word, scale_lane);
        let scale = self.signed_byte_f32(e, emits, scale_byte);
        let scale = self.mul(e, emits, scale, d);
        let scale = self.splat4(e, emits, scale);

        let shift4 = self.u32(e, 4);
        let shift4 = self.splat4(e, emits, shift4);
        let mask4 = self.u32(e, 0x0f);
        let mask4 = self.splat4(e, emits, mask4);
        let mask2 = self.u32(e, 3);
        let mask2 = self.splat4(e, emits, mask2);
        let center = self.f32(e, 32.0);
        let center = self.splat4(e, emits, center);

        let mut values = Vec::with_capacity(8);
        for (low_word, high_word) in [(low_word0, high_word0), (low_word1, high_word1)] {
            let low_bytes = self.unpack4x_u8(e, emits, low_word);
            let low_shifted = self.shr(e, emits, low_bytes, low_nibble_shift);
            let low4 = self.bin(e, emits, BinaryOperator::And, low_shifted, mask4);

            let high_bytes = self.unpack4x_u8(e, emits, high_word);
            let high_shifted = self.shr(e, emits, high_bytes, high_shift);
            let high2 = self.bin(e, emits, BinaryOperator::And, high_shifted, mask2);
            let high2 = self.bin(e, emits, BinaryOperator::ShiftLeft, high2, shift4);
            let quant = self.bin(e, emits, BinaryOperator::InclusiveOr, low4, high2);
            let quant_f = self.as_f32(e, emits, quant);
            let centered = self.sub(e, emits, quant_f, center);
            let result = self.mul(e, emits, centered, scale);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_k_nibblex16(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        q5: bool,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 0, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let dmin_word = self.load_word(e, matrix, base, 1, emits)?;
        let dmin = self.bitcast_f32(e, emits, dmin_word);
        let group = self.shr_lit(e, emits, q, 5);
        let (scale_byte, min_byte) = self.k_scale_pair(e, matrix, base, group, emits)?;
        let scale_f = self.as_f32(e, emits, scale_byte);
        let scale = self.mul(e, emits, scale_f, d);
        let scale = self.splat4(e, emits, scale);
        let min_f = self.as_f32(e, emits, min_byte);
        let min = self.mul(e, emits, min_f, dmin);
        let min = self.splat4(e, emits, min);

        let in_group = self.and_lit(e, emits, q, 31);
        let group_pair = self.shr_lit(e, emits, group, 1);
        let group_pair_offset = self.shl_lit(e, emits, group_pair, 5);
        let byte_index = self.bin(e, emits, BinaryOperator::Add, group_pair_offset, in_group);
        let data_base = if q5 { 13 } else { 5 };
        let data_word = self.shr_lit(e, emits, byte_index, 2);
        let data_off = self.add_lit(e, emits, data_word, data_base);
        let data_words = (0..4)
            .map(|offset| {
                let word_off = self.add_lit(e, emits, data_off, offset);
                self.load_word_dynamic(e, matrix, base, word_off, emits)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let group_low = self.and_lit(e, emits, group, 1);
        let high = self.cmp_lit(e, emits, BinaryOperator::NotEqual, group_low, 0);
        let qh_words = if q5 {
            let qh_byte_index = self.and_lit(e, emits, q, 31);
            let qh_word = self.shr_lit(e, emits, qh_byte_index, 2);
            let qh_off = self.add_lit(e, emits, qh_word, 5);
            Some(
                (0..4)
                    .map(|offset| {
                        let word_off = self.add_lit(e, emits, qh_off, offset);
                        self.load_word_dynamic(e, matrix, base, word_off, emits)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        let qh_bit_index = q5.then(|| self.shr_lit(e, emits, q, 5));

        let mut values = Vec::with_capacity(16);
        for (word_index, word) in data_words.into_iter().enumerate() {
            let bytes = self.unpack4x_u8(e, emits, word);
            let shift = self.u32(e, 4);
            let shift = self.splat4(e, emits, shift);
            let byte_hi = self.shr(e, emits, bytes, shift);
            let mask = self.u32(e, 0x0f);
            let mask = self.splat4(e, emits, mask);
            let byte_lo = self.bin(e, emits, BinaryOperator::And, bytes, mask);
            let mut quant = self.select(e, emits, high, byte_hi, byte_lo);
            if let (Some(qh_words), Some(qh_bit_index)) = (qh_words.as_ref(), qh_bit_index) {
                let qh_bytes = self.unpack4x_u8(e, emits, qh_words[word_index]);
                let qh_bit_index = self.splat4(e, emits, qh_bit_index);
                let shifted_qh = self.shr(e, emits, qh_bytes, qh_bit_index);
                let one = self.u32(e, 1);
                let one = self.splat4(e, emits, one);
                let bit = self.bin(e, emits, BinaryOperator::And, shifted_qh, one);
                let four = self.u32(e, 4);
                let four = self.splat4(e, emits, four);
                let bit = self.bin(e, emits, BinaryOperator::ShiftLeft, bit, four);
                quant = self.bin(e, emits, BinaryOperator::InclusiveOr, quant, bit);
            }
            let quant_f = self.as_f32(e, emits, quant);
            let scaled = self.mul(e, emits, quant_f, scale);
            let result = self.sub(e, emits, scaled, min);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q6kx16(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 52, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let chunk = self.shr_lit(e, emits, q, 7);
        let local = self.and_lit(e, emits, q, 127);
        let high_byte_index = self.and_lit(e, emits, local, 31);
        let low_group = self.shr_lit(e, emits, local, 5);
        let chunk_low_base = self.shl_lit(e, emits, chunk, 6);
        let low_group_parity = self.and_lit(e, emits, low_group, 1);
        let low_group_offset = self.shl_lit(e, emits, low_group_parity, 5);
        let low_nibble_shift = self.shr_lit(e, emits, low_group, 1);
        let low_nibble_shift = self.shl_lit(e, emits, low_nibble_shift, 2);
        let low_nibble_shift = self.splat4(e, emits, low_nibble_shift);
        let local_low_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_byte_index,
            low_group_offset,
        );
        let lower_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            chunk_low_base,
            local_low_index,
        );
        let low_word_off = self.shr_lit(e, emits, lower_index, 2);
        let low_words = (0..4)
            .map(|offset| {
                let word_off = self.add_lit(e, emits, low_word_off, offset);
                self.load_word_dynamic(e, matrix, base, word_off, emits)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let high_chunk_base = self.shl_lit(e, emits, chunk, 5);
        let high_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_chunk_base,
            high_byte_index,
        );
        let high_word_base = self.shr_lit(e, emits, high_index, 2);
        let high_word_off = self.add_lit(e, emits, high_word_base, 32);
        let high_words = (0..4)
            .map(|offset| {
                let word_off = self.add_lit(e, emits, high_word_off, offset);
                self.load_word_dynamic(e, matrix, base, word_off, emits)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let high_shift = self.shl_lit(e, emits, low_group, 1);
        let high_shift = self.splat4(e, emits, high_shift);

        let scale_chunk_base = self.shl_lit(e, emits, chunk, 3);
        let high_byte_half = self.shr_lit(e, emits, high_byte_index, 4);
        let low_group_scale = self.shl_lit(e, emits, low_group, 1);
        let local_scale_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_byte_half,
            low_group_scale,
        );
        let scale_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            scale_chunk_base,
            local_scale_index,
        );
        let scale_word_base = self.shr_lit(e, emits, scale_index, 2);
        let scale_word_off = self.add_lit(e, emits, scale_word_base, 48);
        let scale_word = self.load_word_dynamic(e, matrix, base, scale_word_off, emits)?;
        let scale_lane = self.and_lit(e, emits, scale_index, 3);
        let scale_byte = self.byte_at(e, emits, scale_word, scale_lane);
        let scale = self.signed_byte_f32(e, emits, scale_byte);
        let scale = self.mul(e, emits, scale, d);
        let scale = self.splat4(e, emits, scale);

        let shift4 = self.u32(e, 4);
        let shift4 = self.splat4(e, emits, shift4);
        let mask4 = self.u32(e, 0x0f);
        let mask4 = self.splat4(e, emits, mask4);
        let mask2 = self.u32(e, 3);
        let mask2 = self.splat4(e, emits, mask2);
        let center = self.f32(e, 32.0);
        let center = self.splat4(e, emits, center);

        let mut values = Vec::with_capacity(16);
        for (low_word, high_word) in low_words.into_iter().zip(high_words) {
            let low_bytes = self.unpack4x_u8(e, emits, low_word);
            let low_shifted = self.shr(e, emits, low_bytes, low_nibble_shift);
            let low4 = self.bin(e, emits, BinaryOperator::And, low_shifted, mask4);

            let high_bytes = self.unpack4x_u8(e, emits, high_word);
            let high_shifted = self.shr(e, emits, high_bytes, high_shift);
            let high2 = self.bin(e, emits, BinaryOperator::And, high_shifted, mask2);
            let high2 = self.bin(e, emits, BinaryOperator::ShiftLeft, high2, shift4);
            let quant = self.bin(e, emits, BinaryOperator::InclusiveOr, low4, high2);
            let quant_f = self.as_f32(e, emits, quant);
            let centered = self.sub(e, emits, quant_f, center);
            let result = self.mul(e, emits, centered, scale);
            values.extend((0..4).map(|lane| self.vec4_component(e, emits, result, lane)));
        }
        Ok(values)
    }

    fn dequant_q4_0(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 1);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, q_local, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let low = self.and_lit(e, emits, byte, 0x0f);
        let high_q = self.shr_lit(e, emits, byte, 4);
        let quant = self.select(e, emits, high, high_q, low);
        let quant_f = self.as_f32(e, emits, quant);
        let center = self.f32(e, 8.0);
        let centered = self.sub(e, emits, quant_f, center);
        Ok(self.mul(e, emits, centered, scale))
    }

    fn dequant_q5_0(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let qh = self.load_word(e, matrix, base, 1, emits)?;
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 2);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, q_local, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let low = self.and_lit(e, emits, byte, 0x0f);
        let high4 = self.shr_lit(e, emits, byte, 4);
        let low4 = self.select(e, emits, high, high4, low);
        let high_index = self.add_lit(e, emits, q_local, 16);
        let hi_bit_index = self.select(e, emits, high, high_index, q_local);
        let shifted_qh = self.shr(e, emits, qh, hi_bit_index);
        let hi_bit_low = self.and_lit(e, emits, shifted_qh, 1);
        let hi_bit = self.shl_lit(e, emits, hi_bit_low, 4);
        let quant = self.bin(e, emits, BinaryOperator::InclusiveOr, low4, hi_bit);
        let quant_f = self.as_f32(e, emits, quant);
        let center = self.f32(e, 16.0);
        let centered = self.sub(e, emits, quant_f, center);
        Ok(self.mul(e, emits, centered, scale))
    }

    fn dequant_q8_0(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let q_word = self.shr_lit(e, emits, q, 2);
        let word_off = self.add_lit(e, emits, q_word, 1);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, q, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let signed = self.signed_byte_f32(e, emits, byte);
        Ok(self.mul(e, emits, signed, scale))
    }

    fn dequant_q4_1(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let min_word = self.load_word(e, matrix, base, 1, emits)?;
        let min = self.bitcast_f32(e, emits, min_word);
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 2);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, q_local, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let low = self.and_lit(e, emits, byte, 0x0f);
        let high_q = self.shr_lit(e, emits, byte, 4);
        let quant = self.select(e, emits, high, high_q, low);
        let quant_f = self.as_f32(e, emits, quant);
        let scaled = self.mul(e, emits, quant_f, scale);
        Ok(self.bin(e, emits, BinaryOperator::Add, scaled, min))
    }

    fn dequant_q5_1(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let min_word = self.load_word(e, matrix, base, 1, emits)?;
        let min = self.bitcast_f32(e, emits, min_word);
        let qh = self.load_word(e, matrix, base, 2, emits)?;
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, 3);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, q_local, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let low = self.and_lit(e, emits, byte, 0x0f);
        let high4 = self.shr_lit(e, emits, byte, 4);
        let low4 = self.select(e, emits, high, high4, low);
        let high_index = self.add_lit(e, emits, q_local, 16);
        let hi_bit_index = self.select(e, emits, high, high_index, q_local);
        let shifted_qh = self.shr(e, emits, qh, hi_bit_index);
        let hi_bit_low = self.and_lit(e, emits, shifted_qh, 1);
        let hi_bit = self.shl_lit(e, emits, hi_bit_low, 4);
        let quant = self.bin(e, emits, BinaryOperator::InclusiveOr, low4, hi_bit);
        let quant_f = self.as_f32(e, emits, quant);
        let scaled = self.mul(e, emits, quant_f, scale);
        Ok(self.bin(e, emits, BinaryOperator::Add, scaled, min))
    }

    fn dequant_q8_1(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let q_word = self.shr_lit(e, emits, q, 2);
        let word_off = self.add_lit(e, emits, q_word, 2);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, q, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let signed = self.signed_byte_f32(e, emits, byte);
        Ok(self.mul(e, emits, signed, scale))
    }

    fn dequant_q2k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 20, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let dmin_word = self.load_word(e, matrix, base, 21, emits)?;
        let dmin = self.bitcast_f32(e, emits, dmin_word);
        let group = self.shr_lit(e, emits, q, 4);
        let scale_word_off = self.shr_lit(e, emits, group, 2);
        let scale_word = self.load_word_dynamic(e, matrix, base, scale_word_off, emits)?;
        let scale_lane = self.and_lit(e, emits, group, 3);
        let scale_byte = self.byte_at(e, emits, scale_word, scale_lane);
        let scale_quant = self.and_lit(e, emits, scale_byte, 0x0f);
        let scale_quant_f = self.as_f32(e, emits, scale_quant);
        let scale = self.mul(e, emits, scale_quant_f, d);
        let min_quant = self.shr_lit(e, emits, scale_byte, 4);
        let min_quant_f = self.as_f32(e, emits, min_quant);
        let min = self.mul(e, emits, min_quant_f, dmin);
        let q_local = self.and_lit(e, emits, q, 15);
        let chunk = self.shr_lit(e, emits, group, 3);
        let group_in_chunk = self.and_lit(e, emits, group, 7);
        let pair = self.and_lit(e, emits, group_in_chunk, 1);
        let byte_base = self.shl_lit(e, emits, chunk, 5);
        let pair_offset = self.shl_lit(e, emits, pair, 4);
        let byte_base = self.bin(e, emits, BinaryOperator::Add, byte_base, pair_offset);
        let byte_index = self.bin(e, emits, BinaryOperator::Add, byte_base, q_local);
        let word_off = self.shr_lit(e, emits, byte_index, 2);
        let word_off = self.add_lit(e, emits, word_off, 4);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, byte_index, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let shift = self.shr_lit(e, emits, group_in_chunk, 1);
        let shift = self.shl_lit(e, emits, shift, 1);
        let shifted = self.shr(e, emits, byte, shift);
        let quant = self.and_lit(e, emits, shifted, 3);
        let quant_f = self.as_f32(e, emits, quant);
        let scaled = self.mul(e, emits, quant_f, scale);
        Ok(self.sub(e, emits, scaled, min))
    }

    fn dequant_q3k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 27, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let group = self.shr_lit(e, emits, q, 4);
        let scale_quant = self.q3k_scale(e, matrix, base, group, emits)?;
        let scale_quant_f = self.as_f32(e, emits, scale_quant);
        let center = self.f32(e, 32.0);
        let scale_quant_f = self.sub(e, emits, scale_quant_f, center);
        let scale = self.mul(e, emits, scale_quant_f, d);
        let q_local = self.and_lit(e, emits, q, 15);
        let chunk = self.shr_lit(e, emits, group, 3);
        let group_in_chunk = self.and_lit(e, emits, group, 7);
        let pair = self.and_lit(e, emits, group_in_chunk, 1);
        let byte_base = self.shl_lit(e, emits, chunk, 5);
        let pair_offset = self.shl_lit(e, emits, pair, 4);
        let byte_base = self.bin(e, emits, BinaryOperator::Add, byte_base, pair_offset);
        let byte_index = self.bin(e, emits, BinaryOperator::Add, byte_base, q_local);
        let word_off = self.shr_lit(e, emits, byte_index, 2);
        let word_off = self.add_lit(e, emits, word_off, 8);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, byte_index, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let shift = self.shr_lit(e, emits, group_in_chunk, 1);
        let shift = self.shl_lit(e, emits, shift, 1);
        let shifted = self.shr(e, emits, byte, shift);
        let quant = self.and_lit(e, emits, shifted, 3);
        let quant_f = self.as_f32(e, emits, quant);
        let hmask_index = self.bin(e, emits, BinaryOperator::Add, pair_offset, q_local);
        let hmask_word_off = self.shr_lit(e, emits, hmask_index, 2);
        let hword = self.load_word_dynamic(e, matrix, base, hmask_word_off, emits)?;
        let hmask_lane = self.and_lit(e, emits, hmask_index, 3);
        let hbyte = self.byte_at(e, emits, hword, hmask_lane);
        let hmask_bit_pair = self.shr_lit(e, emits, group_in_chunk, 1);
        let chunk_mask_base = self.shl_lit(e, emits, chunk, 2);
        let hmask_bit = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            chunk_mask_base,
            hmask_bit_pair,
        );
        let one = self.u32(e, 1);
        let hmask = self.bin(e, emits, BinaryOperator::ShiftLeft, one, hmask_bit);
        let high = self.bin(e, emits, BinaryOperator::And, hbyte, hmask);
        let high_set = self.cmp_lit(e, emits, BinaryOperator::NotEqual, high, 0);
        let zero = self.f32(e, 0.0);
        let four = self.f32(e, 4.0);
        let penalty = self.select(e, emits, high_set, zero, four);
        let centered = self.sub(e, emits, quant_f, penalty);
        Ok(self.mul(e, emits, centered, scale))
    }

    fn dequant_q8k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let q_word = self.shr_lit(e, emits, q, 2);
        let word_off = self.add_lit(e, emits, q_word, 1);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, q, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let signed = self.signed_byte_f32(e, emits, byte);
        Ok(self.mul(e, emits, signed, scale))
    }

    fn dequant_q4k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        self.dequant_k_nibble(e, matrix, base, q, emits, false)
    }

    fn dequant_q5k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        self.dequant_k_nibble(e, matrix, base, q, emits, true)
    }

    fn dequant_k_nibble(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
        q5: bool,
    ) -> Result<Handle<Expression>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 0, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let dmin_word = self.load_word(e, matrix, base, 1, emits)?;
        let dmin = self.bitcast_f32(e, emits, dmin_word);
        let group = self.shr_lit(e, emits, q, 5);
        let scale_byte = self.k_scale(e, matrix, base, group, false, emits)?;
        let scale_f = self.as_f32(e, emits, scale_byte);
        let scale = self.mul(e, emits, scale_f, d);
        let min_byte = self.k_scale(e, matrix, base, group, true, emits)?;
        let min_f = self.as_f32(e, emits, min_byte);
        let min = self.mul(e, emits, min_f, dmin);
        let in_group = self.and_lit(e, emits, q, 31);
        let group_pair = self.shr_lit(e, emits, group, 1);
        let group_pair_offset = self.shl_lit(e, emits, group_pair, 5);
        let byte_index = self.bin(e, emits, BinaryOperator::Add, group_pair_offset, in_group);
        let data_base = if q5 { 13 } else { 5 };
        let data_word = self.shr_lit(e, emits, byte_index, 2);
        let data_off = self.add_lit(e, emits, data_word, data_base);
        let word = self.load_word_dynamic(e, matrix, base, data_off, emits)?;
        let byte_lane = self.and_lit(e, emits, byte_index, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let group_low = self.and_lit(e, emits, group, 1);
        let high = self.cmp_lit(e, emits, BinaryOperator::NotEqual, group_low, 0);
        let byte_hi = self.shr_lit(e, emits, byte, 4);
        let byte_lo = self.and_lit(e, emits, byte, 0x0f);
        let mut quant = self.select(e, emits, high, byte_hi, byte_lo);
        if q5 {
            let qh_byte_index = self.and_lit(e, emits, q, 31);
            let qh_word = self.shr_lit(e, emits, qh_byte_index, 2);
            let qh_off = self.add_lit(e, emits, qh_word, 5);
            let qh = self.load_word_dynamic(e, matrix, base, qh_off, emits)?;
            let qh_lane = self.and_lit(e, emits, qh_byte_index, 3);
            let qh_byte = self.byte_at(e, emits, qh, qh_lane);
            let qh_bit_index = self.shr_lit(e, emits, q, 5);
            let shifted_qh = self.shr(e, emits, qh_byte, qh_bit_index);
            let bit = self.and_lit(e, emits, shifted_qh, 1);
            let bit = self.shl_lit(e, emits, bit, 4);
            quant = self.bin(e, emits, BinaryOperator::InclusiveOr, quant, bit);
        }
        let quant_f = self.as_f32(e, emits, quant);
        let scaled = self.mul(e, emits, quant_f, scale);
        Ok(self.sub(e, emits, scaled, min))
    }

    fn dequant_q6k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 52, emits)?;
        let d = self.bitcast_f32(e, emits, d_word);
        let chunk = self.shr_lit(e, emits, q, 7);
        let local = self.and_lit(e, emits, q, 127);
        let high_byte_index = self.and_lit(e, emits, local, 31);
        let low_group = self.shr_lit(e, emits, local, 5);
        let chunk_low_base = self.shl_lit(e, emits, chunk, 6);
        let low_group_parity = self.and_lit(e, emits, low_group, 1);
        let low_group_offset = self.shl_lit(e, emits, low_group_parity, 5);
        let local_low_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_byte_index,
            low_group_offset,
        );
        let lower_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            chunk_low_base,
            local_low_index,
        );
        let low_word_off = self.shr_lit(e, emits, lower_index, 2);
        let low_word = self.load_word_dynamic(e, matrix, base, low_word_off, emits)?;
        let low_lane = self.and_lit(e, emits, lower_index, 3);
        let low_byte = self.byte_at(e, emits, low_word, low_lane);
        let low_nibble_shift = self.shr_lit(e, emits, low_group, 1);
        let low_nibble_shift = self.shl_lit(e, emits, low_nibble_shift, 2);
        let low_shifted = self.shr(e, emits, low_byte, low_nibble_shift);
        let low4 = self.and_lit(e, emits, low_shifted, 0x0f);
        let high_chunk_base = self.shl_lit(e, emits, chunk, 5);
        let high_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_chunk_base,
            high_byte_index,
        );
        let high_word_base = self.shr_lit(e, emits, high_index, 2);
        let high_word_off = self.add_lit(e, emits, high_word_base, 32);
        let high_word = self.load_word_dynamic(e, matrix, base, high_word_off, emits)?;
        let high_lane = self.and_lit(e, emits, high_index, 3);
        let high_byte = self.byte_at(e, emits, high_word, high_lane);
        let high_shift = self.shl_lit(e, emits, low_group, 1);
        let high_shifted = self.shr(e, emits, high_byte, high_shift);
        let high2 = self.and_lit(e, emits, high_shifted, 3);
        let high2 = self.shl_lit(e, emits, high2, 4);
        let quant = self.bin(e, emits, BinaryOperator::InclusiveOr, low4, high2);
        let scale_chunk_base = self.shl_lit(e, emits, chunk, 3);
        let high_byte_half = self.shr_lit(e, emits, high_byte_index, 4);
        let low_group_scale = self.shl_lit(e, emits, low_group, 1);
        let local_scale_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            high_byte_half,
            low_group_scale,
        );
        let scale_index = self.bin(
            e,
            emits,
            BinaryOperator::Add,
            scale_chunk_base,
            local_scale_index,
        );
        let scale_word_base = self.shr_lit(e, emits, scale_index, 2);
        let scale_word_off = self.add_lit(e, emits, scale_word_base, 48);
        let scale_word = self.load_word_dynamic(e, matrix, base, scale_word_off, emits)?;
        let scale_lane = self.and_lit(e, emits, scale_index, 3);
        let scale_byte = self.byte_at(e, emits, scale_word, scale_lane);
        let scale = self.signed_byte_f32(e, emits, scale_byte);
        let scale = self.mul(e, emits, scale, d);
        let quant_f = self.as_f32(e, emits, quant);
        let center = self.f32(e, 32.0);
        let centered = self.sub(e, emits, quant_f, center);
        Ok(self.mul(e, emits, centered, scale))
    }

    fn load_word(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        offset: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let index = self.add_lit(e, emits, base, offset);
        self.load_word_at(e, matrix, index, emits)
    }

    fn tile_matrix_index(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        row: Handle<Expression>,
        col: Handle<Expression>,
        stride: u32,
    ) -> Handle<Expression> {
        let row_offset = self.mul_literal_u32_emitted(e, row, stride, emits);
        self.bin(e, emits, BinaryOperator::Add, row_offset, col)
    }

    fn load_word_dynamic(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        offset: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let index = self.bin(e, emits, BinaryOperator::Add, base, offset);
        self.load_word_at(e, matrix, index, emits)
    }

    fn load_word_at(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        index: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let (ptr, ptr_emits) = self.storage_dynamic_pointer(e, &matrix.data, index)?;
        emits.extend(ptr_emits);
        Ok(self.emit(e, emits, Expression::Load { pointer: ptr }))
    }

    fn k_scale(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        group: Handle<Expression>,
        min: bool,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, group, 4);
        let lane = self.and_lit(e, emits, group, 3);
        let low_word = self.load_word(e, matrix, base, if min { 3 } else { 2 }, emits)?;
        let low_byte = self.byte_at(e, emits, low_word, lane);
        let low_scale = self.and_lit(e, emits, low_byte, 0x3f);

        let extra_word = self.load_word(e, matrix, base, 4, emits)?;
        let extra_byte = self.byte_at(e, emits, extra_word, lane);
        let lsb = if min {
            let shifted = self.shr_lit(e, emits, extra_byte, 4);
            self.and_lit(e, emits, shifted, 0x0f)
        } else {
            self.and_lit(e, emits, extra_byte, 0x0f)
        };
        let msb_bits = self.and_lit(e, emits, low_byte, 0xc0);
        let msb = self.shr_lit(e, emits, msb_bits, 2);
        let high_scale = self.bin(e, emits, BinaryOperator::InclusiveOr, lsb, msb);
        Ok(self.select(e, emits, high, high_scale, low_scale))
    }

    fn k_scale_pair(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        group: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<(Handle<Expression>, Handle<Expression>), LowerError> {
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, group, 4);
        let lane = self.and_lit(e, emits, group, 3);

        let scale_word = self.load_word(e, matrix, base, 2, emits)?;
        let scale_low_byte = self.byte_at(e, emits, scale_word, lane);
        let scale_low = self.and_lit(e, emits, scale_low_byte, 0x3f);

        let min_word = self.load_word(e, matrix, base, 3, emits)?;
        let min_low_byte = self.byte_at(e, emits, min_word, lane);
        let min_low = self.and_lit(e, emits, min_low_byte, 0x3f);

        let extra_word = self.load_word(e, matrix, base, 4, emits)?;
        let extra_byte = self.byte_at(e, emits, extra_word, lane);

        let scale_lsb = self.and_lit(e, emits, extra_byte, 0x0f);
        let scale_msb_bits = self.and_lit(e, emits, scale_low_byte, 0xc0);
        let scale_msb = self.shr_lit(e, emits, scale_msb_bits, 2);
        let scale_high = self.bin(e, emits, BinaryOperator::InclusiveOr, scale_lsb, scale_msb);

        let min_lsb = self.shr_lit(e, emits, extra_byte, 4);
        let min_lsb = self.and_lit(e, emits, min_lsb, 0x0f);
        let min_msb_bits = self.and_lit(e, emits, min_low_byte, 0xc0);
        let min_msb = self.shr_lit(e, emits, min_msb_bits, 2);
        let min_high = self.bin(e, emits, BinaryOperator::InclusiveOr, min_lsb, min_msb);

        Ok((
            self.select(e, emits, high, scale_high, scale_low),
            self.select(e, emits, high, min_high, min_low),
        ))
    }

    fn q3k_scale(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        group: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let s0 = self.load_word(e, matrix, base, 24, emits)?;
        let s1 = self.load_word(e, matrix, base, 25, emits)?;
        let s2 = self.load_word(e, matrix, base, 26, emits)?;
        let lane = self.and_lit(e, emits, group, 3);
        let group_word_bit = self.and_lit(e, emits, group, 4);
        let zero = self.u32(e, 0);
        let use_s1 = self.bin(e, emits, BinaryOperator::NotEqual, group_word_bit, zero);
        let scale_word = self.select(e, emits, use_s1, s1, s0);
        let scale_byte = self.byte_at(e, emits, scale_word, lane);
        let low_nibble = self.and_lit(e, emits, scale_byte, 0x0f);
        let high_nibble = self.shr_lit(e, emits, scale_byte, 4);
        let use_high_nibble = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, group, 8);
        let low = self.select(e, emits, use_high_nibble, high_nibble, low_nibble);
        let extra_byte = self.byte_at(e, emits, s2, lane);
        let high_shift = self.shr_lit(e, emits, group, 2);
        let high_shift = self.shl_lit(e, emits, high_shift, 1);
        let high = self.shr(e, emits, extra_byte, high_shift);
        let high = self.and_lit(e, emits, high, 3);
        let high = self.shl_lit(e, emits, high, 4);
        Ok(self.bin(e, emits, BinaryOperator::InclusiveOr, low, high))
    }

    fn unpack4x_u8(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        word: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::Math {
                fun: MathFunction::Unpack4xU8,
                arg: word,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        )
    }

    fn unpack4x_i8(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        word: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::Math {
                fun: MathFunction::Unpack4xI8,
                arg: word,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        )
    }

    fn splat4(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::Splat {
                size: VectorSize::Quad,
                value,
            },
        )
    }

    fn u32_vec4(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        values: [u32; 4],
    ) -> Handle<Expression> {
        let components = values.into_iter().map(|value| self.u32(e, value)).collect();
        self.emit(
            e,
            emits,
            Expression::Compose {
                ty: self.u32_vec4_ty,
                components,
            },
        )
    }

    fn vec4_component(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        vector: Handle<Expression>,
        index: u32,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::AccessIndex {
                base: vector,
                index,
            },
        )
    }

    fn byte_at(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        word: Handle<Expression>,
        byte: Handle<Expression>,
    ) -> Handle<Expression> {
        let shift = self.shl_lit(e, emits, byte, 3);
        let shifted = self.shr(e, emits, word, shift);
        self.and_lit(e, emits, shifted, 0xff)
    }

    fn signed_byte_f32(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        byte: Handle<Expression>,
    ) -> Handle<Expression> {
        let bias = self.u32(e, 128);
        let biased = self.bin(e, emits, BinaryOperator::ExclusiveOr, byte, bias);
        let as_i32 = self.emit(
            e,
            emits,
            Expression::As {
                expr: biased,
                kind: ScalarKind::Sint,
                convert: Some(4),
            },
        );
        let offset = e.append(Expression::Literal(Literal::I32(128)), Span::default());
        let signed = self.emit(
            e,
            emits,
            Expression::Binary {
                op: BinaryOperator::Subtract,
                left: as_i32,
                right: offset,
            },
        );
        self.emit(
            e,
            emits,
            Expression::As {
                expr: signed,
                kind: ScalarKind::Float,
                convert: Some(4),
            },
        )
    }

    fn bin_lit_u32(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        let value = e.append(Expression::Binary { op, left, right }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(e, value)),
            Span::default(),
        );
        value
    }

    fn increment_u32_local_by_expr(
        &self,
        e: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        step_source: Handle<Expression>,
        step_multiplier: u32,
    ) -> Block {
        let mut body = Block::new();
        let (current, current_emit) = self.load_u32_local(e, local);
        body.push(Statement::Emit(current_emit), Span::default());
        let mut emits = Vec::new();
        let step = self.mul_literal_u32_emitted(e, step_source, step_multiplier, &mut emits);
        let next = self.bin(e, &mut emits, BinaryOperator::Add, current, step);
        Self::push_emits(&mut body, emits);
        let pointer = e.append(Expression::LocalVariable(local), Span::default());
        body.push(
            Statement::Store {
                pointer,
                value: next,
            },
            Span::default(),
        );
        body
    }

    fn emit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        expr: Expression,
    ) -> Handle<Expression> {
        let value = e.append(expr, Span::default());
        emits.push(Self::single_expression_range(e, value));
        value
    }

    fn bin(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(e, emits, Expression::Binary { op, left, right })
    }

    fn cmp_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, emits, op, left, right)
    }

    fn select(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::Select {
                condition,
                accept,
                reject,
            },
        )
    }

    fn shr(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, emits, BinaryOperator::ShiftRight, left, right)
    }

    fn shr_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.shr(e, emits, left, right)
    }

    fn shl_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, emits, BinaryOperator::ShiftLeft, left, right)
    }

    fn and_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, emits, BinaryOperator::And, left, right)
    }

    fn add_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        self.add_literal_u32_emitted(e, left, right, emits)
    }

    fn sub(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, emits, BinaryOperator::Subtract, left, right)
    }

    fn mul(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, emits, BinaryOperator::Multiply, left, right)
    }

    fn as_f32(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::As {
                expr: value,
                kind: ScalarKind::Float,
                convert: Some(4),
            },
        )
    }

    fn bitcast_f32(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::As {
                expr: value,
                kind: ScalarKind::Float,
                convert: None,
            },
        )
    }

    fn u32(&self, e: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        e.append(Expression::Literal(Literal::U32(value)), Span::default())
    }

    fn f32(&self, e: &mut Arena<Expression>, value: f32) -> Handle<Expression> {
        e.append(Expression::Literal(Literal::F32(value)), Span::default())
    }
}
