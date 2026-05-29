use super::*;
use crate::lower::quantized::GgmlBlockCoords;

impl<'a> Lowerer<'a> {
    /// Lower a unified `Expr::QuantizedDot`. Activations and K coordinate are
    /// materialised once, then the format-specific helper is selected by the
    /// `(format, activations, k, block_n)` tuple. Unsupported combinations
    /// return `LowerError::UnsupportedOperation` with the same messages the
    /// pre-merge helpers used.
    pub(in crate::lower) fn lower_tile_quantized_dot_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        request: QuantizedDotLowering<'_>,
    ) -> Result<Handle<Expression>, LowerError> {
        let QuantizedDotLowering {
            src,
            activations,
            k,
            col,
            masked,
            block_n,
        } = request;
        match (activations, k) {
            (PackedActivations::F32(a), DotK::Base(k_base)) => {
                let a_handles = self.lower_tile_exprs_lane(
                    expressions,
                    scratch,
                    body,
                    a,
                    masked.spill_depth + 1,
                )?;
                self.lower_masked_quantized_col_value(
                    expressions,
                    scratch,
                    body,
                    MaskedQuantizedCol {
                        k_base,
                        col,
                        masked,
                    },
                    |expressions, k_base, col, body| match (src.format, block_n) {
                        (GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q8_0Native, 8) => {
                            let a8: [Handle<Expression>; 8] =
                                a_handles.as_slice().try_into().map_err(|_| {
                                    LowerError::UnsupportedOperation(
                                        "f32 activation dot only supports dot8",
                                    )
                                })?;
                            self.dequantize_q8_0_dot8(expressions, src, k_base, col, &a8, body)
                        }
                        (GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative, 8) => {
                            let a8: [Handle<Expression>; 8] =
                                a_handles.as_slice().try_into().map_err(|_| {
                                    LowerError::UnsupportedOperation(
                                        "f32 activation dot only supports dot8",
                                    )
                                })?;
                            self.dequantize_q6k_dot8(expressions, src, k_base, col, &a8, body)
                        }
                        (GgmlQuantFormat::Q4K | GgmlQuantFormat::Q4KNative, 8 | 16 | 32) => {
                            self.q4k_f32_dot(expressions, src, k_base, col, &a_handles, body)
                        }
                        _ => Err(LowerError::UnsupportedOperation(
                            "f32 activation dot only supports Q8_0/Q6K dot8 or Q4K dot8/dot16/dot32",
                        )),
                    },
                )
            }
            (PackedActivations::Q8(a), DotK::Base(k_base)) => {
                let a_handles = self.lower_tile_exprs_lane(
                    expressions,
                    scratch,
                    body,
                    a,
                    masked.spill_depth + 1,
                )?;
                let a_packs =
                    self.cached_q8_activation_packs(expressions, scratch, body, &a_handles)?;
                self.lower_masked_quantized_col_value(
                    expressions,
                    scratch,
                    body,
                    MaskedQuantizedCol {
                        k_base,
                        col,
                        masked,
                    },
                    |expressions, k_base, col, body| match (src.format, block_n) {
                        (GgmlQuantFormat::Q4K | GgmlQuantFormat::Q4KNative, 8 | 16) => self
                            .q4k_q8_activation_dot(expressions, src, k_base, col, &a_packs, body),
                        (GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative, 8 | 16) => self
                            .q6k_q8_activation_dot(expressions, src, k_base, col, &a_packs, body),
                        _ => Err(LowerError::UnsupportedOperation(
                            "q8 activation dot only supports Q4K/Q6K dot8/dot16",
                        )),
                    },
                )
            }
            (
                PackedActivations::Q4KGgml { low, high, sums },
                DotK::Block {
                    block,
                    c0: iq,
                    c1: ir,
                },
            ) => {
                if low.len() != 16 || high.len() != 16 || sums.len() != 4 {
                    return Err(LowerError::UnsupportedOperation(
                        "q4k ggml dot requires 16 low activations, 16 high activations, and 4 sums",
                    ));
                }
                if !src.format.is_q4k_family() {
                    return Err(LowerError::UnsupportedOperation(
                        "q4k ggml dot only supports Q4K formats",
                    ));
                }

                let low_handles = self.lower_tile_exprs_lane(
                    expressions,
                    scratch,
                    body,
                    low,
                    masked.spill_depth + 1,
                )?;
                let high_handles = self.lower_tile_exprs_lane(
                    expressions,
                    scratch,
                    body,
                    high,
                    masked.spill_depth + 1,
                )?;
                let sum_handles = self.lower_tile_exprs_lane(
                    expressions,
                    scratch,
                    body,
                    sums,
                    masked.spill_depth + 1,
                )?;

                self.lower_masked_f32_value(
                    expressions,
                    scratch,
                    body,
                    masked,
                    |expressions, block_body| {
                        let coords = self.lower_ggml_block_coords(
                            expressions,
                            scratch,
                            block_body,
                            GgmlBlockCoordExprs {
                                block,
                                c0: iq,
                                c1: ir,
                                col,
                                spill_depth: masked.spill_depth,
                            },
                        )?;
                        self.q4k_ggml_dot(
                            expressions,
                            src,
                            coords,
                            crate::lower::quantized::Q4KGgmlActivationHandles {
                                low: &low_handles,
                                high: &high_handles,
                                sums: &sum_handles,
                            },
                            block_body,
                        )
                    },
                )
            }
            (
                PackedActivations::F32(a),
                DotK::Block {
                    block,
                    c0: ip,
                    c1: il,
                },
            ) => {
                if a.len() != 16 {
                    return Err(LowerError::UnsupportedOperation(
                        "q6k ggml dot requires 16 activations",
                    ));
                }
                if !src.format.is_q6k_family() {
                    return Err(LowerError::UnsupportedOperation(
                        "q6k ggml dot only supports Q6K formats",
                    ));
                }

                let a_handles = self.lower_tile_exprs_lane(
                    expressions,
                    scratch,
                    body,
                    a,
                    masked.spill_depth + 1,
                )?;

                self.lower_masked_f32_value(
                    expressions,
                    scratch,
                    body,
                    masked,
                    |expressions, block_body| {
                        let coords = self.lower_ggml_block_coords(
                            expressions,
                            scratch,
                            block_body,
                            GgmlBlockCoordExprs {
                                block,
                                c0: ip,
                                c1: il,
                                col,
                                spill_depth: masked.spill_depth,
                            },
                        )?;
                        self.q6k_ggml_dot(expressions, src, coords, &a_handles, block_body)
                    },
                )
            }
            (PackedActivations::Q8(_), DotK::Block { .. }) => {
                Err(LowerError::UnsupportedOperation(
                    "q8 activation dot does not support block-shaped K coordinates",
                ))
            }
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

    /// Lower the four `(block, c0, c1, col)` index expressions used by Q4K and
    /// Q6K ggml dot helpers.
    pub(in crate::lower) fn lower_ggml_block_coords(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        coords: GgmlBlockCoordExprs<'_>,
    ) -> Result<GgmlBlockCoords, LowerError> {
        Ok(GgmlBlockCoords {
            block: self.lower_tile_expr_lane(
                expressions,
                scratch,
                body,
                coords.block,
                coords.spill_depth,
            )?,
            c0: self.lower_tile_expr_lane(
                expressions,
                scratch,
                body,
                coords.c0,
                coords.spill_depth,
            )?,
            c1: self.lower_tile_expr_lane(
                expressions,
                scratch,
                body,
                coords.c1,
                coords.spill_depth,
            )?,
            col: self.lower_tile_expr_lane(
                expressions,
                scratch,
                body,
                coords.col,
                coords.spill_depth,
            )?,
        })
    }

    pub(in crate::lower) fn lower_masked_quantized_col_value(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        request: MaskedQuantizedCol<'_>,
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
            request.masked,
            |expressions, block| {
                let k_base = self.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    block,
                    request.k_base,
                    request.masked.spill_depth,
                )?;
                let col = self.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    block,
                    request.col,
                    request.masked.spill_depth,
                )?;
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
            (GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q8_0Native, 8) => {
                self.dequantize_q8_0_values8(expressions, src, k_base, col, body)
            }
            (GgmlQuantFormat::Q4K | GgmlQuantFormat::Q4KNative, 8) => {
                self.dequantize_q4k_values8(expressions, src, k_base, col, body)
            }
            (GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative, 8) => {
                self.dequantize_q6k_values8(expressions, src, k_base, col, body)
            }
            (GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative, 16) => {
                self.dequantize_q6k_values16(expressions, src, k_base, col, body)
            }
            (GgmlQuantFormat::Q5_0 | GgmlQuantFormat::Q5_0Native, 16) => {
                self.dequantize_q5_0_values16(expressions, src, k_base, col, body)
            }
            (_, 8 | 16) => self.dequantize_qvalues(expressions, src, k_base, col, block_n, body),
            _ => Err(LowerError::UnsupportedOperation(
                "quantized block dequant only supports 8-wide or 16-wide blocks",
            )),
        }
    }

    pub(in crate::lower) fn lower_tile_quantized_block_lane(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        request: QuantizedBlockLaneLowering<'_>,
    ) -> Result<Handle<Expression>, LowerError> {
        let QuantizedBlockLaneLowering {
            id,
            src,
            k_base,
            col,
            masked,
            block_n,
            lane,
        } = request;
        let MaskedF32Value {
            mask,
            fill,
            spill_depth,
        } = masked;
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
        let fill_source = fill.element();
        let fill_value =
            self.lower_tile_expr_lane(expressions, scratch, body, fill, spill_depth)?;
        let fill_value =
            self.cast_tile_value(expressions, body, fill_value, fill_source, ElementType::F32);
        for local in &tmp_locals {
            self.store_local(expressions, body, *local, fill_value);
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
            self.store_local(expressions, &mut accept, *local, *value);
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
        let handles: Vec<_> = tmp_locals
            .iter()
            .map(|local| self.load_local(expressions, body, *local))
            .collect();
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
            .get(Self::element_scratch_index(element)?)
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
            .get(Self::element_scratch_index(element)?)
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
        masked: MaskedF32Value<'_>,
        lower_value: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
        ) -> Result<Handle<Expression>, LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        if masked.mask.is_constant_true() {
            return lower_value(expressions, body);
        }

        let fill_source = masked.fill.element();
        let fill_handle =
            self.lower_tile_expr_lane(expressions, scratch, body, masked.fill, masked.spill_depth)?;
        let fill_handle = self.cast_tile_value(
            expressions,
            body,
            fill_handle,
            fill_source,
            ElementType::F32,
        );
        self.lower_masked_value_to_local(
            expressions,
            scratch,
            body,
            MaskedLocalValue {
                mask: masked.mask,
                element: ElementType::F32,
                fill: fill_handle,
                spill_depth: masked.spill_depth,
            },
            lower_value,
        )
    }

    pub(in crate::lower) fn lower_masked_value_to_local(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        masked: MaskedLocalValue<'_>,
        lower_accept_value: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
        ) -> Result<Handle<Expression>, LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        let tmp = Self::tile_value_local(scratch, masked.element)?;
        let tmp_ptr = self.local_var(expressions, tmp);
        body.push(
            Statement::Store {
                pointer: tmp_ptr,
                value: masked.fill,
            },
            Span::default(),
        );

        let mask =
            self.lower_tile_expr_lane(expressions, scratch, body, masked.mask, masked.spill_depth)?;
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
        let loaded = Self::emit_load(expressions, body, tmp_ptr);
        Ok(loaded)
    }
}
