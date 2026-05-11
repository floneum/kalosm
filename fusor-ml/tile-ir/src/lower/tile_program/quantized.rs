use super::*;

impl<'a> Lowerer<'a> {
    /// Lower a unified `Expr::QuantizedDot`. Activations and K coordinate are
    /// materialised once, then the format-specific helper is selected by the
    /// `(format, activations, k, block_n)` tuple. Unsupported combinations
    /// return `LowerError::UnsupportedOperation` with the same messages the
    /// pre-merge helpers used.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::lower) fn lower_tile_quantized_dot_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        src: &QuantizedMatrix,
        activations: &PackedActivations,
        k: &DotK,
        col: &Expr,
        mask: &Expr,
        fill: &Expr,
        block_n: u32,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match (activations, k) {
            (PackedActivations::F32(a), DotK::Base(k_base)) => {
                let a_handles =
                    self.lower_tile_exprs_lane(expressions, scratch, body, a, spill_depth + 1)?;
                self.lower_masked_quantized_col_value(
                    expressions,
                    scratch,
                    body,
                    k_base,
                    col,
                    mask,
                    fill,
                    spill_depth,
                    |expressions, k_base, col, body| match (src.format, block_n) {
                        (GgmlQuantFormat::Q8_0, 8) => {
                            let a8: [Handle<Expression>; 8] =
                                a_handles.as_slice().try_into().map_err(|_| {
                                    LowerError::UnsupportedOperation(
                                        "f32 activation dot only supports dot8",
                                    )
                                })?;
                            self.dequantize_q8_0_dot8(expressions, src, k_base, col, &a8, body)
                        }
                        (GgmlQuantFormat::Q6K, 8) => {
                            let a8: [Handle<Expression>; 8] =
                                a_handles.as_slice().try_into().map_err(|_| {
                                    LowerError::UnsupportedOperation(
                                        "f32 activation dot only supports dot8",
                                    )
                                })?;
                            self.dequantize_q6k_dot8(expressions, src, k_base, col, &a8, body)
                        }
                        (GgmlQuantFormat::Q4K, 8 | 16 | 32) => {
                            self.q4k_f32_dot(expressions, src, k_base, col, &a_handles, body)
                        }
                        _ => Err(LowerError::UnsupportedOperation(
                            "f32 activation dot only supports Q8_0/Q6K dot8 or Q4K dot8/dot16/dot32",
                        )),
                    },
                )
            }
            (PackedActivations::Q8(a), DotK::Base(k_base)) => {
                let a_handles =
                    self.lower_tile_exprs_lane(expressions, scratch, body, a, spill_depth + 1)?;
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
                    |expressions, k_base, col, body| match (src.format, block_n) {
                        (GgmlQuantFormat::Q4K, 8 | 16) => self.q4k_q8_activation_dot(
                            expressions,
                            src,
                            k_base,
                            col,
                            &a_packs,
                            body,
                        ),
                        (GgmlQuantFormat::Q6K, 8 | 16) => self.q6k_q8_activation_dot(
                            expressions,
                            src,
                            k_base,
                            col,
                            &a_packs,
                            body,
                        ),
                        _ => Err(LowerError::UnsupportedOperation(
                            "q8 activation dot only supports Q4K/Q6K dot8/dot16",
                        )),
                    },
                )
            }
            (
                PackedActivations::Q4KGgml { low, high, sums },
                DotK::Block { block, c0: iq, c1: ir },
            ) => {
                if low.len() != 16 || high.len() != 16 || sums.len() != 4 {
                    return Err(LowerError::UnsupportedOperation(
                        "q4k ggml dot requires 16 low activations, 16 high activations, and 4 sums",
                    ));
                }
                if !matches!(src.format, GgmlQuantFormat::Q4K) {
                    return Err(LowerError::UnsupportedOperation(
                        "q4k ggml dot only supports the Q4K format",
                    ));
                }

                let low_handles =
                    self.lower_tile_exprs_lane(expressions, scratch, body, low, spill_depth + 1)?;
                let high_handles =
                    self.lower_tile_exprs_lane(expressions, scratch, body, high, spill_depth + 1)?;
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
                        let block_h = self.lower_tile_expr_lane(
                            expressions,
                            scratch,
                            block_body,
                            block,
                            spill_depth,
                        )?;
                        let iq_h = self
                            .lower_tile_expr_lane(expressions, scratch, block_body, iq, spill_depth)?;
                        let ir_h = self
                            .lower_tile_expr_lane(expressions, scratch, block_body, ir, spill_depth)?;
                        let col_h = self
                            .lower_tile_expr_lane(expressions, scratch, block_body, col, spill_depth)?;
                        self.q4k_ggml_dot(
                            expressions,
                            src,
                            block_h,
                            iq_h,
                            ir_h,
                            col_h,
                            &low_handles,
                            &high_handles,
                            &sum_handles,
                            block_body,
                        )
                    },
                )
            }
            (
                PackedActivations::F32(a),
                DotK::Block { block, c0: ip, c1: il },
            ) => {
                if a.len() != 16 {
                    return Err(LowerError::UnsupportedOperation(
                        "q6k ggml dot requires 16 activations",
                    ));
                }
                if !matches!(src.format, GgmlQuantFormat::Q6K) {
                    return Err(LowerError::UnsupportedOperation(
                        "q6k ggml dot only supports the Q6K format",
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
                        let block_h = self.lower_tile_expr_lane(
                            expressions,
                            scratch,
                            block_body,
                            block,
                            spill_depth,
                        )?;
                        let ip_h = self
                            .lower_tile_expr_lane(expressions, scratch, block_body, ip, spill_depth)?;
                        let il_h = self
                            .lower_tile_expr_lane(expressions, scratch, block_body, il, spill_depth)?;
                        let col_h = self
                            .lower_tile_expr_lane(expressions, scratch, block_body, col, spill_depth)?;
                        self.q6k_ggml_dot(
                            expressions,
                            src,
                            block_h,
                            ip_h,
                            il_h,
                            col_h,
                            &a_handles,
                            block_body,
                        )
                    },
                )
            }
            (PackedActivations::Q8(_), DotK::Block { .. }) => Err(LowerError::UnsupportedOperation(
                "q8 activation dot does not support block-shaped K coordinates",
            )),
            (PackedActivations::Q4KGgml { .. }, DotK::Base(_)) => {
                Err(LowerError::UnsupportedOperation(
                    "q4k ggml dot requires a block-shaped K coordinate",
                ))
            }
        }
    }

    pub(in crate::lower) fn lower_tile_exprs_lane(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        exprs: &[Expr],
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
        k_base: &Expr,
        col: &Expr,
        mask: &Expr,
        fill: &Expr,
        spill_depth: usize,
        lower_value: impl FnOnce(
            &mut Arena<Expression>,
            Handle<Expression>,
            Handle<Expression>,
            &mut Block,
        ) -> Result<Handle<Expression>, LowerError>,
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
                    self.lower_tile_expr_lane(expressions, scratch, block, k_base, spill_depth)?;
                let col =
                    self.lower_tile_expr_lane(expressions, scratch, block, col, spill_depth)?;
                lower_value(expressions, k_base, col, block)
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
        body: &mut Block,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        match (src.format, block_n) {
            (GgmlQuantFormat::Q8_0, 8) => {
                self.dequantize_q8_0_values8(expressions, src, k_base, col, body)
            }
            (GgmlQuantFormat::Q4K, 8) => {
                self.dequantize_q4k_values8(expressions, src, k_base, col, body)
            }
            (GgmlQuantFormat::Q6K, 8) => {
                self.dequantize_q6k_values8(expressions, src, k_base, col, body)
            }
            (GgmlQuantFormat::Q6K, 16) => {
                self.dequantize_q6k_values16(expressions, src, k_base, col, body)
            }
            (GgmlQuantFormat::Q5_0, 16) => {
                self.dequantize_q5_0_values16(expressions, src, k_base, col, body)
            }
            (_, 8 | 16) => self.dequantize_qvalues(expressions, src, k_base, col, block_n, body),
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
        k_base: &Expr,
        col: &Expr,
        mask: &Expr,
        fill: &Expr,
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

        if mask.is_constant_true() {
            let k_base_handle =
                self.lower_tile_expr_lane(expressions, scratch, body, k_base, spill_depth)?;
            let col_handle =
                self.lower_tile_expr_lane(expressions, scratch, body, col, spill_depth)?;
            let values = self.dequantize_quantized_block_values(
                expressions,
                src,
                k_base_handle,
                col_handle,
                block_n,
                body,
            )?;
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
        let fill_source = self.tile_expr_element(fill)?;
        let fill_value =
            self.lower_tile_expr_lane(expressions, scratch, body, fill, spill_depth)?;
        let fill_value =
            self.cast_tile_value(expressions, body, fill_value, fill_source, ElementType::F32);
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
            self.lower_tile_expr_lane(expressions, scratch, body, mask, spill_depth)?;
        let mut accept = Block::new();
        let k_base_handle =
            self.lower_tile_expr_lane(expressions, scratch, &mut accept, k_base, spill_depth)?;
        let col_handle =
            self.lower_tile_expr_lane(expressions, scratch, &mut accept, col, spill_depth)?;
        let values = self.dequantize_quantized_block_values(
            expressions,
            src,
            k_base_handle,
            col_handle,
            block_n,
            &mut accept,
        )?;
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
        mask: &Expr,
        spill_depth: usize,
        fill: &Expr,
        lower_value: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
        ) -> Result<Handle<Expression>, LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        if mask.is_constant_true() {
            return lower_value(expressions, body);
        }

        let fill_source = self.tile_expr_element(fill)?;
        let fill_handle =
            self.lower_tile_expr_lane(expressions, scratch, body, fill, spill_depth)?;
        let fill_handle =
            self.cast_tile_value(expressions, body, fill_handle, fill_source, ElementType::F32);
        self.lower_masked_value_to_local(
            expressions,
            scratch,
            body,
            mask,
            spill_depth,
            ElementType::F32,
            fill_handle,
            lower_value,
        )
    }

    pub(in crate::lower) fn lower_masked_value_to_local(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        mask: &Expr,
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

        let mask = self.lower_tile_expr_lane(expressions, scratch, body, mask, spill_depth)?;
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
