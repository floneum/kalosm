use super::*;

impl<'a> Lowerer<'a> {
    /// Lower an `ExprKind::Dequantize`, producing its `lanes` f32 handles. The
    /// `lanes` width carries the caller's `values_per_lane` tiling choice. The
    /// emit-once memoization (so all `LaneOf` projections share one helper
    /// emission) is handled by the `Shared` wrapper above this; this method
    /// always emits the dequant helper once.
    pub(in crate::lower) fn lower_dequantize(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        dequant: &Expr,
        outer_spill: usize,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let ExprKind::Dequantize {
            src,
            k_base,
            col,
            mask,
            fill,
            lanes,
        } = dequant.kind()
        else {
            return Err(LowerError::UnsupportedOperation(
                "expected a Dequantize node",
            ));
        };
        let block_n = *lanes;
        let spill_depth = outer_spill;

        if mask.is_constant_true() {
            let k_base_handle = self.lower_expr_lane(expressions, body, k_base, spill_depth)?;
            let col_handle = self.lower_expr_lane(expressions, body, col, spill_depth)?;
            return self.dequantize_quantized_block_values(
                expressions,
                src,
                k_base_handle,
                col_handle,
                block_n,
                body,
            );
        }

        // Masked: fill all N lane locals, then overwrite under the mask.
        let tmp_locals: Vec<_> = (0..block_n)
            .map(|i| self.scratch_f32(ScratchKind::BlockDequant, i))
            .collect();
        let fill_source = fill.element();
        let fill_value = self.lower_expr_lane(expressions, body, fill, spill_depth)?;
        let fill_value =
            self.cast_tile_value(expressions, body, fill_value, fill_source, ElementType::F32);
        for local in &tmp_locals {
            self.store_local(expressions, body, *local, fill_value);
        }

        let mask_handle = self.lower_expr_lane(expressions, body, mask, spill_depth)?;
        let mut accept = Block::new();
        let k_base_handle = self.lower_expr_lane(expressions, &mut accept, k_base, spill_depth)?;
        let col_handle = self.lower_expr_lane(expressions, &mut accept, col, spill_depth)?;
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
        Ok(tmp_locals
            .iter()
            .map(|local| self.load_local(expressions, body, *local))
            .collect())
    }

    /// Lower an `ExprKind::QuantizedDot` — a fused per-column quantized dot. The
    /// activations are materialised once (and Q8-packed once, outside the column
    /// mask), then the masked column emits the format-specific fused dot:
    /// `QuantActivation::F32` decodes the weights to f32 (Q8_0/Q6K dot8, Q4K
    /// dot8/16/32), `Q8` keeps them quantized and emits `Dot4I8Packed`.
    pub(in crate::lower) fn lower_quantized_dot(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expr: &Expr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        let ExprKind::QuantizedDot {
            src,
            packing,
            activations,
            k_base,
            col,
            mask,
            fill,
        } = expr.kind()
        else {
            return Err(LowerError::UnsupportedOperation(
                "expected a QuantizedDot node",
            ));
        };
        let block_n = activations.len();
        let a_handles = self.lower_exprs_lane(expressions, body, activations, spill_depth + 1)?;

        // Q8 packs the activations once, outside the per-column mask.
        let a_packs = match packing {
            QuantActivation::Q8 => {
                Some(self.cached_q8_activation_packs(expressions, body, &a_handles)?)
            }
            QuantActivation::F32 => None,
        };

        self.lower_masked_f32_value(
            expressions,
            body,
            MaskedF32Value {
                mask,
                fill,
                spill_depth,
            },
            |expressions, block| {
                let k = self.lower_expr_lane(expressions, block, k_base, spill_depth)?;
                let c = self.lower_expr_lane(expressions, block, col, spill_depth)?;
                match packing {
                    QuantActivation::F32 => match (src.format, block_n) {
                        (GgmlQuantFormat::Q4_0 | GgmlQuantFormat::Q4_0Native, 8 | 16 | 32) => {
                            self.q4_0_f32_dot(expressions, src, k, c, &a_handles, block)
                        }
                        (GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q8_0Native, 8) => {
                            let a8 = Self::expect_dot8(&a_handles)?;
                            self.dequantize_q8_0_dot8(expressions, src, k, c, &a8, block)
                        }
                        (GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative, 8) => {
                            let a8 = Self::expect_dot8(&a_handles)?;
                            self.dequantize_q6k_dot8(expressions, src, k, c, &a8, block)
                        }
                        (GgmlQuantFormat::Q4K | GgmlQuantFormat::Q4KNative, 8 | 16 | 32) => {
                            self.q4k_f32_dot(expressions, src, k, c, &a_handles, block)
                        }
                        _ => Err(LowerError::UnsupportedOperation(
                            "f32 activation dot only supports Q4_0 dot8/16, Q8_0/Q6K dot8, or Q4K dot8/16/32",
                        )),
                    },
                    QuantActivation::Q8 => {
                        let a_packs = a_packs.as_ref().expect("q8 packs materialised above");
                        match src.format {
                            GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative => {
                                self.q6k_q8_activation_dot(expressions, src, k, c, a_packs, block)
                            }
                            _ => Err(LowerError::UnsupportedOperation(
                                "q8 activation dot only supports Q6K",
                            )),
                        }
                    }
                }
            },
        )
    }

    fn expect_dot8(
        a_handles: &[Handle<Expression>],
    ) -> Result<[Handle<Expression>; 8], LowerError> {
        a_handles
            .try_into()
            .map_err(|_| LowerError::UnsupportedOperation("f32 activation dot only supports dot8"))
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

    pub(in crate::lower) fn lower_masked_f32_value(
        &self,
        expressions: &mut Arena<Expression>,
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
            self.lower_expr_lane(expressions, body, masked.fill, masked.spill_depth)?;
        let fill_handle = self.cast_tile_value(
            expressions,
            body,
            fill_handle,
            fill_source,
            ElementType::F32,
        );
        self.lower_masked_value_to_local(
            expressions,
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
        body: &mut Block,
        masked: MaskedLocalValue<'_>,
        lower_accept_value: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
        ) -> Result<Handle<Expression>, LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        let tmp = self.scratch_local(ScratchKind::Value, masked.element, 0)?;
        let tmp_ptr = self.local_var(expressions, tmp);
        body.push(
            Statement::Store {
                pointer: tmp_ptr,
                value: masked.fill,
            },
            Span::default(),
        );

        let mask = self.lower_expr_lane(expressions, body, masked.mask, masked.spill_depth)?;
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
