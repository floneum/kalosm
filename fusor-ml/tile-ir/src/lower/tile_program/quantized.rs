use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn lower_tile_quantized_q8_0_dot8_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        a: &[Box<TileExpr>; 8],
        src: &QuantizedMatrix,
        k_base: &TileIndexExpr,
        col: &TileIndexExpr,
        mask: &TileMaskExpr,
        fill: F32Bits,
        format: GgmlQuantFormat,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let a_handles =
            self.lower_tile_exprs_lane(expressions, scratch, body, a, spill_depth + 1)?;
        let a_handles: [Handle<Expression>; 8] = a_handles
            .try_into()
            .map_err(|_| LowerError::UnsupportedOperation("q8_0 dot8 expected 8 A values"))?;

        self.lower_masked_quantized_col_value(
            expressions,
            scratch,
            body,
            k_base,
            col,
            mask,
            fill,
            spill_depth,
            |expressions, k_base, col| match format {
                GgmlQuantFormat::Q8_0 => {
                    self.dequantize_q8_0_dot8(expressions, src, k_base, col, &a_handles)
                }
                GgmlQuantFormat::Q6K => {
                    self.dequantize_q6k_dot8(expressions, src, k_base, col, &a_handles)
                }
                _ => Err(LowerError::UnsupportedOperation(
                    "unsupported quantized f32 dot8 format",
                )),
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::lower) fn lower_tile_quantized_vec_dot_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        kind: QuantizedVecDotKind,
        a: &[Box<TileExpr>],
        src: &QuantizedMatrix,
        k_base: &TileIndexExpr,
        col: &TileIndexExpr,
        mask: &TileMaskExpr,
        fill: F32Bits,
        block_n: u32,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let a_handles =
            self.lower_tile_exprs_lane(expressions, scratch, body, a, spill_depth + 1)?;
        match kind {
            QuantizedVecDotKind::Q8Activation => {
                let a_packs =
                    self.cached_q8_activation_packs(expressions, scratch, body, &a_handles)?;
                self.lower_masked_quantized_col_value(
                    expressions,
                    scratch,
                    body,
                    k_base,
                    col,
                    mask,
                    fill,
                    spill_depth,
                    |expressions, k_base, col| match (src.format, block_n) {
                        (GgmlQuantFormat::Q4K, 8 | 16) => {
                            self.q4k_q8_activation_dot(expressions, src, k_base, col, &a_packs)
                        }
                        (GgmlQuantFormat::Q6K, 8 | 16) => {
                            self.q6k_q8_activation_dot(expressions, src, k_base, col, &a_packs)
                        }
                        _ => Err(LowerError::UnsupportedOperation(
                            "q8 activation dot only supports Q4K/Q6K dot8/dot16",
                        )),
                    },
                )
            }
            QuantizedVecDotKind::Q4KF32 => self.lower_masked_quantized_col_value(
                expressions,
                scratch,
                body,
                k_base,
                col,
                mask,
                fill,
                spill_depth,
                |expressions, k_base, col| match (src.format, block_n) {
                    (GgmlQuantFormat::Q4K, 8 | 16 | 32) => {
                        self.q4k_f32_dot(expressions, src, k_base, col, &a_handles)
                    }
                    _ => Err(LowerError::UnsupportedOperation(
                        "q4k f32 dot only supports Q4K dot8/dot16/dot32",
                    )),
                },
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::lower) fn lower_tile_quantized_q4k_ggml_dot_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        a_low: &[Box<TileExpr>],
        a_high: &[Box<TileExpr>],
        sums: &[Box<TileExpr>],
        src: &QuantizedMatrix,
        block: &TileIndexExpr,
        iq: &TileIndexExpr,
        ir: &TileIndexExpr,
        col: &TileIndexExpr,
        mask: &TileMaskExpr,
        fill: F32Bits,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        if a_low.len() != 16 || a_high.len() != 16 || sums.len() != 4 {
            return Err(LowerError::UnsupportedOperation(
                "q4k ggml dot requires 16 low activations, 16 high activations, and 4 sums",
            ));
        }

        let a_low_handles =
            self.lower_tile_exprs_lane(expressions, scratch, body, a_low, spill_depth + 1)?;
        let a_high_handles =
            self.lower_tile_exprs_lane(expressions, scratch, body, a_high, spill_depth + 1)?;
        let sum_handles =
            self.lower_tile_exprs_lane(expressions, scratch, body, sums, spill_depth + 1)?;

        self.lower_masked_f32_value(
            expressions,
            scratch,
            body,
            mask,
            spill_depth,
            fill,
            |expressions, block_body| {
                let [block_h, iq_h, ir_h, col_h] = self.lower_tile_index_exprs(
                    expressions,
                    scratch,
                    block_body,
                    [block, iq, ir, col],
                    spill_depth,
                )?;
                self.q4k_ggml_dot(
                    expressions,
                    src,
                    block_h,
                    iq_h,
                    ir_h,
                    col_h,
                    &a_low_handles,
                    &a_high_handles,
                    &sum_handles,
                )
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::lower) fn lower_tile_quantized_q6k_ggml_dot_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        a: &[Box<TileExpr>],
        src: &QuantizedMatrix,
        block: &TileIndexExpr,
        ip: &TileIndexExpr,
        il: &TileIndexExpr,
        col: &TileIndexExpr,
        mask: &TileMaskExpr,
        fill: F32Bits,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        if a.len() != 16 {
            return Err(LowerError::UnsupportedOperation(
                "q6k ggml dot requires 16 activations",
            ));
        }

        let a_handles =
            self.lower_tile_exprs_lane(expressions, scratch, body, a, spill_depth + 1)?;

        self.lower_masked_f32_value(
            expressions,
            scratch,
            body,
            mask,
            spill_depth,
            fill,
            |expressions, block_body| {
                let [block_h, ip_h, il_h, col_h] = self.lower_tile_index_exprs(
                    expressions,
                    scratch,
                    block_body,
                    [block, ip, il, col],
                    spill_depth,
                )?;
                self.q6k_ggml_dot(expressions, src, block_h, ip_h, il_h, col_h, &a_handles)
            },
        )
    }

    pub(in crate::lower) fn lower_tile_exprs_lane(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        exprs: &[Box<TileExpr>],
        spill_depth: usize,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        exprs
            .iter()
            .map(|expr| self.lower_tile_expr_lane(expressions, scratch, body, expr, spill_depth))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::lower) fn lower_masked_quantized_col_value(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        k_base: &TileIndexExpr,
        col: &TileIndexExpr,
        mask: &TileMaskExpr,
        fill: F32Bits,
        spill_depth: usize,
        lower_value: impl FnOnce(
            &mut Arena<Expression>,
            Handle<Expression>,
            Handle<Expression>,
        )
            -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        self.lower_masked_f32_value(
            expressions,
            scratch,
            body,
            mask,
            spill_depth,
            fill,
            |expressions, block| {
                let k_base =
                    self.lower_tile_index_expr(expressions, scratch, block, k_base, spill_depth)?;
                let col =
                    self.lower_tile_index_expr(expressions, scratch, block, col, spill_depth)?;
                lower_value(expressions, k_base, col)
            },
        )
    }

    pub(in crate::lower) fn dequantize_quantized_block_values(
        &self,
        expressions: &mut Arena<Expression>,
        src: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        block_n: u32,
    ) -> Result<(Vec<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        match (src.format, block_n) {
            (GgmlQuantFormat::Q8_0, 8) => {
                self.dequantize_q8_0_values8(expressions, src, k_base, col)
            }
            (GgmlQuantFormat::Q4K, 8) => self.dequantize_q4k_values8(expressions, src, k_base, col),
            (GgmlQuantFormat::Q6K, 8) => self.dequantize_q6k_values8(expressions, src, k_base, col),
            (GgmlQuantFormat::Q6K, 16) => {
                self.dequantize_q6k_values16(expressions, src, k_base, col)
            }
            (GgmlQuantFormat::Q5_0, 16) => {
                self.dequantize_q5_0_values16(expressions, src, k_base, col)
            }
            (_, 8 | 16) => self.dequantize_qvalues(expressions, src, k_base, col, block_n),
            _ => Err(LowerError::UnsupportedOperation(
                "quantized block dequant only supports 8-wide or 16-wide blocks",
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(in crate::lower) fn lower_tile_quantized_block_lane(
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

        if matches!(mask, TileMaskExpr::True) {
            let k_base_handle =
                self.lower_tile_index_expr(expressions, scratch, body, k_base, spill_depth)?;
            let col_handle =
                self.lower_tile_index_expr(expressions, scratch, body, col, spill_depth)?;
            let (values, value_emits) = self.dequantize_quantized_block_values(
                expressions,
                src,
                k_base_handle,
                col_handle,
                block_n,
            )?;
            Self::push_emits(body, value_emits);
            self.block_dequant_cache
                .borrow_mut()
                .insert(id, values.clone());
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
        let (values, value_emits) = self.dequantize_quantized_block_values(
            expressions,
            src,
            k_base_handle,
            col_handle,
            block_n,
        )?;
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

    pub(in crate::lower) fn block_dequant_value_local(
        &self,
        scratch: ScratchLocals,
        index: u32,
    ) -> Option<Handle<LocalVariable>> {
        scratch.block_dequant.get(index as usize).copied()
    }

    pub(in crate::lower) fn tile_expr_spill_local(
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

    pub(in crate::lower) fn tile_value_local(
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

    pub(in crate::lower) fn lower_masked_f32_value(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        mask: &TileMaskExpr,
        spill_depth: usize,
        fill: F32Bits,
        lower_value: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
        )
            -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        if matches!(mask, TileMaskExpr::True) {
            let (value, emits) = lower_value(expressions, body)?;
            Self::push_emits(body, emits);
            return Ok(value);
        }

        let fill = expressions.append(
            Expression::Literal(Literal::F32(fill.get())),
            Span::default(),
        );
        self.lower_masked_value_to_local(
            expressions,
            scratch,
            body,
            mask,
            spill_depth,
            ElementType::F32,
            fill,
            |expressions, accept| {
                let (value, emits) = lower_value(expressions, accept)?;
                Self::push_emits(accept, emits);
                Ok(value)
            },
        )
    }

    pub(in crate::lower) fn lower_masked_value_to_local(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        mask: &TileMaskExpr,
        spill_depth: usize,
        element: ElementType,
        fill: Handle<Expression>,
        lower_accept_value: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
        ) -> Result<Handle<Expression>, LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        let tmp = Self::tile_value_local(scratch, element)?;
        let tmp_ptr = expressions.append(Expression::LocalVariable(tmp), Span::default());
        body.push(
            Statement::Store {
                pointer: tmp_ptr,
                value: fill,
            },
            Span::default(),
        );

        let mask = self.lower_tile_mask_expr(expressions, scratch, body, mask, spill_depth)?;
        let mut accept = Block::new();
        let value = lower_accept_value(expressions, &mut accept)?;
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
}
