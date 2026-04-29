use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_block(
        &self,
        ir_block: &crate::Block,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();

        let ops = ir_block.ops();
        let mut op_index = 0;
        while op_index < ops.len() {
            if let Some((statement, consumed)) =
                self.try_lower_fused_gemm_store(ops, op_index, expressions, scratch)?
            {
                body.push(statement, Span::default());
                op_index += consumed;
                continue;
            }

            let op = &ops[op_index];
            match op {
                Op::Block(op) => {
                    body.push(
                        Statement::Block(self.lower_block(&op.body, expressions, scratch)?),
                        Span::default(),
                    );
                }
                Op::FillTile(op) => match self.tile_layout(op.dst)?.memory_level() {
                    MemoryLevel::Workgroup => {
                        body.push(
                            self.store_zero_to_tile(expressions, scratch.tile_index, op.dst)?,
                            Span::default(),
                        );
                    }
                    MemoryLevel::Private => {
                        body.push(
                            self.fill_private_tile(expressions, scratch.linear_index, op.dst)?,
                            Span::default(),
                        );
                    }
                    memory => return Err(LowerError::UnsupportedMemoryLevel(memory)),
                },
                Op::CooperativeLoad(op) => {
                    body.push(
                        self.lower_cooperative_load(
                            expressions,
                            scratch.tile_index,
                            op.dst,
                            &op.src,
                        )?,
                        Span::default(),
                    );
                }
                Op::Barrier(op) => {
                    let barrier = match op.scope {
                        BarrierScope::Workgroup => Barrier::WORK_GROUP,
                    };
                    body.push(Statement::ControlBarrier(barrier), Span::default());
                }
                Op::Partition(op) => {
                    for binding in &op.bindings {
                        self.tile_layout(binding.source)?;
                        self.tile_layout(binding.view)?;
                    }
                    body.push(
                        Statement::Block(self.lower_block(&op.body, expressions, scratch)?),
                        Span::default(),
                    );
                }
                Op::Gemm(op) => {
                    body.push(self.lower_gemm(expressions, scratch, op)?, Span::default());
                }
                Op::Gemv(op) => {
                    body.push(self.lower_gemv(expressions, scratch, op)?, Span::default());
                }
                Op::Mma(op) => {
                    body.push(self.lower_mma(expressions, scratch, op)?, Span::default());
                }
                Op::StoreTile(op) => {
                    body.push(
                        self.lower_store_tile(expressions, scratch.store_index, op.src, &op.dst)?,
                        Span::default(),
                    );
                }
                Op::Loop(op) => {
                    let crate::LoopKind::RangeStep { iterations, .. } = op.kind;
                    let loop_body = self.lower_block(&op.body, expressions, scratch)?;
                    body.push(
                        self.counted_loop(expressions, scratch.loop_index, iterations, loop_body),
                        Span::default(),
                    );
                }
            }
            op_index += 1;
        }

        Ok(body)
    }

    pub(super) fn try_lower_fused_gemm_store(
        &self,
        ops: &[Op],
        op_index: usize,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Option<(Statement, usize)>, LowerError> {
        let Some(Op::FillTile(fill)) = ops.get(op_index) else {
            return Ok(None);
        };
        let Some(Op::Loop(loop_op)) = ops.get(op_index + 1) else {
            return Ok(None);
        };
        let Some(Op::StoreTile(store)) = ops.get(op_index + 2) else {
            return Ok(None);
        };
        if fill.value != crate::FillValue::Zero || store.src != fill.dst {
            return Ok(None);
        }
        let crate::LoopKind::RangeStep { iterations, .. } = loop_op.kind;

        if let Some(statement) = self.try_lower_shared_gemm_store(
            &loop_op.body,
            fill.dst,
            &store.dst,
            iterations,
            expressions,
            scratch,
        )? {
            return Ok(Some((statement, 3)));
        }

        if let Some(statement) = self.try_lower_direct_gemm_store(
            &loop_op.body,
            fill.dst,
            &store.dst,
            iterations,
            expressions,
            scratch,
        )? {
            return Ok(Some((statement, 3)));
        }

        if iterations != 1 {
            return Ok(None);
        }

        let mut body = Block::new();
        let mut fused_gemm = false;
        for op in loop_op.body.ops() {
            match op {
                Op::CooperativeLoad(op) => {
                    body.push(
                        self.lower_cooperative_load(
                            expressions,
                            scratch.tile_index,
                            op.dst,
                            &op.src,
                        )?,
                        Span::default(),
                    );
                }
                Op::Barrier(op) => {
                    if fused_gemm {
                        continue;
                    }
                    let barrier = match op.scope {
                        BarrierScope::Workgroup => Barrier::WORK_GROUP,
                    };
                    body.push(Statement::ControlBarrier(barrier), Span::default());
                }
                Op::Gemm(op) if op.acc == fill.dst && !fused_gemm => {
                    body.push(
                        self.lower_gemm_to_storage(expressions, scratch, op, &store.dst)?,
                        Span::default(),
                    );
                    fused_gemm = true;
                }
                _ => return Ok(None),
            }
        }

        if fused_gemm {
            Ok(Some((Statement::Block(body), 3)))
        } else {
            Ok(None)
        }
    }

    pub(super) fn try_lower_shared_gemm_store(
        &self,
        loop_body: &crate::Block,
        acc: TileRef,
        dst: &StorageView,
        iterations: u32,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Option<Statement>, LowerError> {
        if !PREFER_SHARED_GEMM {
            return Ok(None);
        }
        let mut loads = Vec::new();
        let mut gemm = None;
        for op in loop_body.ops() {
            match op {
                Op::CooperativeLoad(op) => loads.push(op),
                Op::Barrier(_) => {}
                Op::Gemm(op) if op.acc == acc && gemm.is_none() => gemm = Some(op),
                _ => return Ok(None),
            }
        }

        let Some(gemm) = gemm else {
            return Ok(None);
        };
        let Some(a_load) = loads.iter().find(|load| load.dst == gemm.a) else {
            return Ok(None);
        };
        let Some(b_load) = loads.iter().find(|load| load.dst == gemm.b) else {
            return Ok(None);
        };

        self.lower_shared_gemm_loop_to_storage_4col(
            expressions,
            scratch,
            a_load,
            b_load,
            gemm,
            dst,
            iterations,
        )
    }

    pub(super) fn try_lower_direct_gemm_store(
        &self,
        loop_body: &crate::Block,
        acc: TileRef,
        dst: &StorageView,
        iterations: u32,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Option<Statement>, LowerError> {
        let mut loads = Vec::new();
        let mut gemm = None;
        for op in loop_body.ops() {
            match op {
                Op::CooperativeLoad(op) => loads.push(op),
                Op::Barrier(_) => {}
                Op::Gemm(op) if op.acc == acc && gemm.is_none() => gemm = Some(op),
                _ => return Ok(None),
            }
        }

        let Some(gemm) = gemm else {
            return Ok(None);
        };
        let Some(a_load) = loads.iter().find(|load| load.dst == gemm.a) else {
            return Ok(None);
        };
        let Some(b_load) = loads.iter().find(|load| load.dst == gemm.b) else {
            return Ok(None);
        };

        if PREFER_COOP_MATRIX_GEMM && PREFER_SHARED_COOP_GEMM {
            let a_layout = self.tile_layout(gemm.a)?;
            let b_layout = self.tile_layout(gemm.b)?;
            let acc_layout = self.tile_layout(gemm.acc)?;
            let dst_layout = self.storage_layout(dst)?;
            if Self::can_lower_shared_gemm_coop8(
                a_layout, b_layout, acc_layout, dst_layout, iterations,
            ) {
                return Ok(Some(self.lower_shared_gemm_loop_to_storage_coop8(
                    expressions,
                    scratch,
                    a_load,
                    b_load,
                    gemm,
                    dst,
                    iterations,
                )?));
            }
        }

        Ok(Some(self.lower_storage_gemm_loop_to_storage(
            expressions,
            scratch,
            &a_load.src,
            &b_load.src,
            gemm,
            dst,
            iterations,
        )?))
    }

    pub(super) fn fill_private_tile(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        tile: TileRef,
    ) -> Result<Statement, LowerError> {
        let layout = self.tile_layout(tile)?;
        let mut body = Block::new();
        let (index, index_emit) = self.load_u32_local(expressions, index_local);
        let (pointer, pointer_emits) = self.tile_dynamic_pointer(expressions, tile, index)?;
        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        body.push(Statement::Emit(index_emit), Span::default());
        for emit in pointer_emits {
            body.push(Statement::Emit(emit), Span::default());
        }
        body.push(
            Statement::Store {
                pointer,
                value: zero,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(expressions, index_local, layout.element_count(), body))
    }
}
