use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn dequantize_qvalue(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let block_elems = matrix.format.block_elements();
        let block_words = matrix.format.block_words();
        let block = self.div_literal_u32_emitted(expressions, k, block_elems, body);
        let q = self.and_lit(expressions, body, k, block_elems - 1);
        let base = self.quantized_block_base(expressions, matrix, block, col, block_words, body);
        let value = if let Some(spec) = AffineDequantSpec::for_format(matrix.format) {
            self.dequant_affine(expressions, matrix, base, q, spec, body)?
        } else {
            match matrix.format {
                GgmlQuantFormat::Q2K => self.dequant_q2k(expressions, matrix, base, q, body)?,
                GgmlQuantFormat::Q3K => self.dequant_q3k(expressions, matrix, base, q, body)?,
                GgmlQuantFormat::Q4K | GgmlQuantFormat::Q4KNative => {
                    self.dequant_q4k(expressions, matrix, base, q, body)?
                }
                GgmlQuantFormat::Q5K | GgmlQuantFormat::Q5KNative => {
                    self.dequant_q5k(expressions, matrix, base, q, body)?
                }
                GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative => {
                    self.dequant_q6k(expressions, matrix, base, q, body)?
                }
                GgmlQuantFormat::Q8K => self.dequant_q8k(expressions, matrix, base, q, body)?,
                GgmlQuantFormat::Q4_0
                | GgmlQuantFormat::Q4_0Native
                | GgmlQuantFormat::Q4_1
                | GgmlQuantFormat::Q5_0
                | GgmlQuantFormat::Q5_0Native
                | GgmlQuantFormat::Q5_1
                | GgmlQuantFormat::Q8_0
                | GgmlQuantFormat::Q8_0Native
                | GgmlQuantFormat::Q8_1 => unreachable!("affine formats handled above"),
            }
        };
        Ok(value)
    }

    pub(in crate::lower) fn dequantize_qvalues(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        n: u32,
        body: &mut Block,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let mut values = Vec::with_capacity(n as usize);
        for lane in 0..n {
            let k = self.add_literal_u32_emitted(expressions, k_base, lane, body);
            let value = self.dequantize_qvalue(expressions, matrix, k, col, body)?;
            values.push(value);
        }
        Ok(values)
    }

    pub(in crate::lower) fn dequantize_q8_0_values8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        if !matrix.format.is_q8_0_family() {
            return Err(LowerError::UnsupportedOperation(
                "q8_0 vector dequantizer only supports Q8_0 formats",
            ));
        }

        let parts = self.q8_0_block_parts8(expressions, matrix, k_base, col, body)?;
        let mut values = Vec::with_capacity(8);
        for signed in self.q8_0_components8(expressions, body, &parts) {
            values.push(self.mul(expressions, body, signed, parts.scale));
        }
        Ok(values)
    }

    pub(in crate::lower) fn dequantize_q8_0_dot8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &[Handle<Expression>; 8],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if !matrix.format.is_q8_0_family() {
            return Err(LowerError::UnsupportedOperation(
                "q8_0 dot8 only supports Q8_0 formats",
            ));
        }

        let parts = self.q8_0_block_parts8(expressions, matrix, k_base, col, body)?;
        let q_components = self.q8_0_components8(expressions, body, &parts);
        let sum = self.dot_vec4_chunks(expressions, body, a, &q_components);
        Ok(self.mul(expressions, body, sum, parts.scale))
    }

    pub(in crate::lower) fn q8_0_block_parts8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Q8_0BlockParts, LowerError> {
        let block = self.div_literal_u32_emitted(expressions, k_base, 32, body);
        let q = self.and_lit(expressions, body, k_base, 31);
        let base = self.quantized_block_base(
            expressions,
            matrix,
            block,
            col,
            matrix.format.block_words(),
            body,
        );
        let scale = self.load_affine_scale_f32(expressions, matrix, base, 0, body)?;
        let q_word = self.shr_lit(expressions, body, q, 2);
        let data_byte = self.shl_lit(expressions, body, q_word, 2);
        let data_byte = self.add_lit(
            expressions,
            body,
            data_byte,
            self.q8_0_data_byte_offset(matrix.format)?,
        );
        let word0 = self.load_word_at_block_dynamic_byte_offset(
            expressions,
            matrix,
            base,
            data_byte,
            body,
        )?;
        let word1_byte = self.add_lit(expressions, body, data_byte, 4);
        let word1 = self.load_word_at_block_dynamic_byte_offset(
            expressions,
            matrix,
            base,
            word1_byte,
            body,
        )?;
        Ok(Q8_0BlockParts {
            scale,
            words: [word0, word1],
        })
    }

    pub(in crate::lower) fn q8_0_components8(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        parts: &Q8_0BlockParts,
    ) -> [Handle<Expression>; 8] {
        std::array::from_fn(|lane| {
            let byte_lane = self.u32(expressions, (lane % 4) as u32);
            let word = parts.words[usize::from(lane >= 4)];
            let byte = self.byte_at(expressions, body, word, byte_lane);
            self.signed_byte_f32(expressions, body, byte)
        })
    }

    pub(in crate::lower) fn dequantize_q4k_values8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        if !matrix.format.is_q4k_family() {
            return Err(LowerError::UnsupportedOperation(
                "q4k vector dequantizer only supports Q4K formats",
            ));
        }

        let block = self.div_literal_u32_emitted(expressions, k_base, 256, body);
        let q_base = self.and_lit(expressions, body, k_base, 255);
        let base = self.quantized_block_base(
            expressions,
            matrix,
            block,
            col,
            matrix.format.block_words(),
            body,
        );

        let (d, dmin) = self.q4k_load_d_dmin(expressions, matrix, base, body)?;
        let group = self.shr_lit(expressions, body, q_base, 5);
        let (scale_byte, min_byte) =
            self.q4k_scale_min_bytes(expressions, matrix, base, group, body)?;
        let scale_f = self.as_f32(expressions, body, scale_byte);
        let scale = self.mul(expressions, body, scale_f, d);
        let min_f = self.as_f32(expressions, body, min_byte);
        let min = self.mul(expressions, body, min_f, dmin);

        let in_group = self.and_lit(expressions, body, q_base, 31);
        let group_pair = self.shr_lit(expressions, body, group, 1);
        let group_pair_offset = self.shl_lit(expressions, body, group_pair, 5);
        let byte_index = self.bin(
            expressions,
            body,
            BinaryOperator::Add,
            group_pair_offset,
            in_group,
        );
        let data_word = self.shr_lit(expressions, body, byte_index, 2);
        let data_base = self.q4k_data_word_offset(matrix.format)?;
        let word0_off = self.add_lit(expressions, body, data_word, data_base);
        let word1_off = self.add_lit(expressions, body, data_word, data_base + 1);
        let word0 = self.load_word_dynamic(expressions, matrix, base, word0_off, body)?;
        let word1 = self.load_word_dynamic(expressions, matrix, base, word1_off, body)?;
        let group_low = self.and_lit(expressions, body, group, 1);
        let high = self.cmp_lit(expressions, body, BinaryOperator::NotEqual, group_low, 0);

        let mut values = Vec::with_capacity(8);
        for lane in 0..8 {
            let byte_lane = self.u32(expressions, (lane % 4) as u32);
            let word = if lane < 4 { word0 } else { word1 };
            let byte = self.byte_at(expressions, body, word, byte_lane);
            let byte_hi = self.shr_lit(expressions, body, byte, 4);
            let byte_lo = self.and_lit(expressions, body, byte, 0x0f);
            let quant = self.select(expressions, body, high, byte_hi, byte_lo);
            let quant_f = self.as_f32(expressions, body, quant);
            let scaled = self.mul(expressions, body, quant_f, scale);
            values.push(self.sub(expressions, body, scaled, min));
        }
        Ok(values)
    }

    pub(in crate::lower) fn dequantize_q5_0_values16(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        if !matrix.format.is_q5_0_family() {
            return Err(LowerError::UnsupportedOperation(
                "q5_0 vector dequantizer only supports Q5_0 formats",
            ));
        }

        let block = self.div_literal_u32_emitted(expressions, k_base, 32, body);
        let q_base = self.and_lit(expressions, body, k_base, 31);
        let base = self.quantized_block_base(
            expressions,
            matrix,
            block,
            col,
            matrix.format.block_words(),
            body,
        );
        let scale = self.load_affine_scale_f32(expressions, matrix, base, 0, body)?;
        let qh = self.load_word_at_block_byte_offset(
            expressions,
            matrix,
            base,
            self.q5_0_high_byte_offset(matrix.format)?,
            body,
        )?;
        let high = self.cmp_lit(expressions, body, BinaryOperator::GreaterEqual, q_base, 16);
        let sixteen = self.u32(expressions, 16);
        let zero = self.u32(expressions, 0);
        let high_base = self.select(expressions, body, high, sixteen, zero);
        let words = [
            self.load_word_at_block_byte_offset(
                expressions,
                matrix,
                base,
                self.q5_0_data_byte_offset(matrix.format)?,
                body,
            )?,
            self.load_word_at_block_byte_offset(
                expressions,
                matrix,
                base,
                self.q5_0_data_byte_offset(matrix.format)? + 4,
                body,
            )?,
            self.load_word_at_block_byte_offset(
                expressions,
                matrix,
                base,
                self.q5_0_data_byte_offset(matrix.format)? + 8,
                body,
            )?,
            self.load_word_at_block_byte_offset(
                expressions,
                matrix,
                base,
                self.q5_0_data_byte_offset(matrix.format)? + 12,
                body,
            )?,
        ];

        let mut values = Vec::with_capacity(16);
        for lane in 0..16 {
            let byte_lane = self.u32(expressions, (lane % 4) as u32);
            let byte = self.byte_at(expressions, body, words[lane / 4], byte_lane);
            let low = self.and_lit(expressions, body, byte, 0x0f);
            let high4 = self.shr_lit(expressions, body, byte, 4);
            let low4 = self.select(expressions, body, high, high4, low);
            let lane_index =
                self.add_literal_u32_emitted(expressions, high_base, lane as u32, body);
            let shifted_qh = self.shr(expressions, body, qh, lane_index);
            let hi_bit_low = self.and_lit(expressions, body, shifted_qh, 1);
            let hi_bit = self.shl_lit(expressions, body, hi_bit_low, 4);
            let quant = self.bin(expressions, body, BinaryOperator::InclusiveOr, low4, hi_bit);
            let quant_f = self.as_f32(expressions, body, quant);
            let center = self.f32(expressions, 16.0);
            let centered = self.sub(expressions, body, quant_f, center);
            values.push(self.mul(expressions, body, centered, scale));
        }
        Ok(values)
    }

    pub(in crate::lower) fn q6k_block_parts(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Q6KBlockParts, LowerError> {
        let block = self.div_literal_u32_emitted(expressions, k_base, 256, body);
        let q_base = self.and_lit(expressions, body, k_base, 255);
        let base = self.quantized_block_base(
            expressions,
            matrix,
            block,
            col,
            matrix.format.block_words(),
            body,
        );

        let d = self.q6k_load_d(expressions, matrix, base, body)?;
        let chunk = self.shr_lit(expressions, body, q_base, 7);
        let local = self.and_lit(expressions, body, q_base, 127);
        let high_byte_index = self.and_lit(expressions, body, local, 31);
        let low_group = self.shr_lit(expressions, body, local, 5);

        let chunk_low_base = self.shl_lit(expressions, body, chunk, 6);
        let low_group_parity = self.and_lit(expressions, body, low_group, 1);
        let low_group_offset = self.shl_lit(expressions, body, low_group_parity, 5);
        let local_low_index = self.bin(
            expressions,
            body,
            BinaryOperator::Add,
            high_byte_index,
            low_group_offset,
        );
        let lower_index = self.bin(
            expressions,
            body,
            BinaryOperator::Add,
            chunk_low_base,
            local_low_index,
        );
        let low_word_base = self.shr_lit(expressions, body, lower_index, 2);
        let low_words =
            self.load_word_pair_dynamic(expressions, matrix, base, low_word_base, 0, body)?;
        let low_shift = self.shr_lit(expressions, body, low_group, 1);
        let low_shift = self.shl_lit(expressions, body, low_shift, 2);

        let high_chunk_base = self.shl_lit(expressions, body, chunk, 5);
        let high_index = self.bin(
            expressions,
            body,
            BinaryOperator::Add,
            high_chunk_base,
            high_byte_index,
        );
        let high_word_base = self.shr_lit(expressions, body, high_index, 2);
        let high_words =
            self.load_word_pair_dynamic(expressions, matrix, base, high_word_base, 32, body)?;
        let high_shift = self.shl_lit(expressions, body, low_group, 1);

        let scale_chunk_base = self.shl_lit(expressions, body, chunk, 3);
        let high_byte_half = self.shr_lit(expressions, body, high_byte_index, 4);
        let low_group_scale = self.shl_lit(expressions, body, low_group, 1);
        let local_scale_index = self.bin(
            expressions,
            body,
            BinaryOperator::Add,
            high_byte_half,
            low_group_scale,
        );
        let scale_index = self.bin(
            expressions,
            body,
            BinaryOperator::Add,
            scale_chunk_base,
            local_scale_index,
        );
        let scale_byte =
            self.load_byte_dynamic(expressions, matrix, base, scale_index, 192, body)?;
        let scale = self.signed_byte_f32(expressions, body, scale_byte);
        let scale = self.mul(expressions, body, scale, d);

        Ok(Q6KBlockParts {
            low_words,
            high_words,
            low_shift,
            high_shift,
            scale,
        })
    }

    pub(in crate::lower) fn q6k_centered_component(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        parts: &Q6KBlockParts,
        lane: usize,
    ) -> Handle<Expression> {
        let quant = self.q6k_quant_component(expressions, body, parts, lane);
        self.center_quant_by_32(expressions, body, quant)
    }

    pub(in crate::lower) fn q6k_quant_component(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        parts: &Q6KBlockParts,
        lane: usize,
    ) -> Handle<Expression> {
        let byte_lane = self.u32(expressions, (lane % 4) as u32);
        let low_word = parts.low_words[usize::from(lane >= 4)];
        let low_byte = self.byte_at(expressions, body, low_word, byte_lane);
        let low_shifted = self.shr(expressions, body, low_byte, parts.low_shift);
        let low4 = self.and_lit(expressions, body, low_shifted, 0x0f);

        let high_word = parts.high_words[usize::from(lane >= 4)];
        let high_byte = self.byte_at(expressions, body, high_word, byte_lane);
        let high_shifted = self.shr(expressions, body, high_byte, parts.high_shift);
        let high2 = self.and_lit(expressions, body, high_shifted, 3);
        let high2 = self.shl_lit(expressions, body, high2, 4);
        self.or(expressions, body, low4, high2)
    }

    pub(in crate::lower) fn dequantize_q6k_values8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        if !matrix.format.is_q6k_family() {
            return Err(LowerError::UnsupportedOperation(
                "q6k vector dequantizer only supports Q6K formats",
            ));
        }

        let parts = self.q6k_block_parts(expressions, matrix, k_base, col, body)?;

        let mut values = Vec::with_capacity(8);
        for lane in 0..8 {
            let centered = self.q6k_centered_component(expressions, body, &parts, lane);
            values.push(self.mul(expressions, body, centered, parts.scale));
        }
        Ok(values)
    }

    pub(in crate::lower) fn dequantize_q6k_values16(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Vec<Handle<Expression>>, LowerError> {
        let mut values = self.dequantize_q6k_values8(expressions, matrix, k_base, col, body)?;
        let k_base_hi = self.add_lit(expressions, body, k_base, 8);
        let hi_values = self.dequantize_q6k_values8(expressions, matrix, k_base_hi, col, body)?;
        values.extend(hi_values);
        Ok(values)
    }

    pub(in crate::lower) fn dequantize_q6k_dot8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &[Handle<Expression>; 8],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if !matrix.format.is_q6k_family() {
            return Err(LowerError::UnsupportedOperation(
                "q6k dot8 only supports Q6K formats",
            ));
        }

        let parts = self.q6k_block_parts(expressions, matrix, k_base, col, body)?;

        let mut q_components = Vec::with_capacity(8);
        for lane in 0..8 {
            q_components.push(self.q6k_centered_component(expressions, body, &parts, lane));
        }

        let sum = self.dot_vec4_chunks(expressions, body, a, &q_components);
        Ok(self.mul(expressions, body, sum, parts.scale))
    }

    pub(in crate::lower) fn q4k_q8_activation_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &Q8ActivationPacks,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        self.q8_activation_pack_pair_dot(expressions, body, k_base, a, |s, e, b, k, off| {
            s.q4k_q8_activation_dot8(e, matrix, QuantDotCoords { k_base: k, col }, a, off, b)
        })
    }

    /// Sum a per-pair dot product over every 2-pack chunk of the activation
    /// stream. Shared by Q4K and Q6K x Q8 activation dot helpers.
    pub(in crate::lower) fn q8_activation_pack_pair_dot(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        k_base: Handle<Expression>,
        a: &Q8ActivationPacks,
        mut chunk: impl FnMut(
            &Self,
            &mut Arena<Expression>,
            &mut Block,
            Handle<Expression>,
            usize,
        ) -> Result<Handle<Expression>, LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        if a.len == 0 || !a.len.is_multiple_of(2) {
            return Err(LowerError::UnsupportedOperation(
                "q8 activation dot requires an even number of activation packs",
            ));
        }

        let mut total = self.f32(expressions, 0.0);
        for pack_offset in (0..a.len).step_by(2) {
            let k = self.add_lit(expressions, body, k_base, (pack_offset * 4) as u32);
            let chunk = chunk(self, expressions, body, k, pack_offset)?;
            total = self.add(expressions, body, total, chunk);
        }
        Ok(total)
    }

    pub(in crate::lower) fn q4k_q8_activation_dot8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        coords: QuantDotCoords,
        a: &Q8ActivationPacks,
        pack_offset: usize,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if !matrix.format.is_q4k_family() || a.len < pack_offset + 2 {
            return Err(LowerError::UnsupportedOperation(
                "q4k x q8 activation dot requires a Q4K format and two activation packs",
            ));
        }

        let block = self.q4k_quant_packs8(expressions, matrix, coords.k_base, coords.col, body)?;
        let total = self.q8_activation_packs_dot(
            expressions,
            body,
            a,
            pack_offset,
            Q8ActivationDotRhs {
                scale: block.scale,
                packs: block.data,
                min: Some(block.min),
            },
        );
        Ok(total)
    }

    pub(in crate::lower) fn q4k_f32_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &[Handle<Expression>],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if !matrix.format.is_q4k_family() || a.is_empty() || !a.len().is_multiple_of(8) {
            return Err(LowerError::UnsupportedOperation(
                "q4k f32 dot requires a Q4K format and a multiple of 8 activation values",
            ));
        }
        if a.len() == 32 {
            return self.q4k_f32_dot_exact::<32, 8>(
                expressions,
                matrix,
                QuantDotCoords { k_base, col },
                a,
                true,
                body,
            );
        }
        if a.len() == 16 {
            return self.q4k_f32_dot_exact::<16, 4>(
                expressions,
                matrix,
                QuantDotCoords { k_base, col },
                a,
                false,
                body,
            );
        }

        let mut total = self.f32(expressions, 0.0);
        for pack_offset in (0..a.len()).step_by(8) {
            let k = self.add_lit(expressions, body, k_base, pack_offset as u32);
            let block = self.q4k_quant_values::<8, 2>(expressions, matrix, k, col, false, body)?;
            let chunk = self.q4k_f32_weighted_sum(
                expressions,
                body,
                block.scale,
                block.min,
                &block.data,
                &a[pack_offset..pack_offset + 8],
            );
            total = self.add(expressions, body, total, chunk);
        }
        Ok(total)
    }

    pub(in crate::lower) fn q4k_f32_dot_exact<const N: usize, const WORDS: usize>(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        coords: QuantDotCoords,
        a: &[Handle<Expression>],
        whole_group_pair: bool,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        debug_assert_eq!(WORDS * 4, N);
        debug_assert_eq!(a.len(), N);

        let block = self.q4k_quant_values::<N, WORDS>(
            expressions,
            matrix,
            coords.k_base,
            coords.col,
            whole_group_pair,
            body,
        )?;

        let total =
            self.q4k_f32_weighted_sum(expressions, body, block.scale, block.min, &block.data, a);
        Ok(total)
    }

    pub(in crate::lower) fn q4k_f32_weighted_sum(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        scale: Handle<Expression>,
        min: Handle<Expression>,
        quants: &[Handle<Expression>],
        a: &[Handle<Expression>],
    ) -> Handle<Expression> {
        let weighted_sum = self.dot_quant_vec4_chunks(expressions, body, a, quants);
        let activation_sum = self.sum_values(expressions, body, a);
        let scaled = self.mul(expressions, body, weighted_sum, scale);
        let min_term = self.mul(expressions, body, activation_sum, min);
        self.sub(expressions, body, scaled, min_term)
    }

    pub(in crate::lower) fn sum_values(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        values: &[Handle<Expression>],
    ) -> Handle<Expression> {
        let mut total = self.f32(expressions, 0.0);
        for value in values {
            total = self.add(expressions, body, total, *value);
        }
        total
    }

    pub(in crate::lower) fn dot_quant_vec4_chunks(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: &[Handle<Expression>],
        right_quants: &[Handle<Expression>],
    ) -> Handle<Expression> {
        debug_assert_eq!(left.len(), right_quants.len());
        debug_assert!(!left.is_empty());
        debug_assert_eq!(left.len() % 4, 0);

        let mut total = self.f32(expressions, 0.0);
        for (left_chunk, right_chunk) in left.chunks_exact(4).zip(right_quants.chunks_exact(4)) {
            let right_chunk = right_chunk
                .iter()
                .map(|quant| self.as_f32(expressions, body, *quant))
                .collect::<Vec<_>>();
            let dot = self.dot_vec4(expressions, body, left_chunk, &right_chunk);
            total = self.add(expressions, body, total, dot);
        }
        total
    }

    pub(in crate::lower) fn dot_vec4_chunks(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: &[Handle<Expression>],
        right: &[Handle<Expression>],
    ) -> Handle<Expression> {
        debug_assert_eq!(left.len(), right.len());
        debug_assert!(!left.is_empty());
        debug_assert_eq!(left.len() % 4, 0);

        let mut chunks = left.chunks_exact(4).zip(right.chunks_exact(4));
        let (left_chunk, right_chunk) = chunks
            .next()
            .expect("dot_vec4_chunks requires at least one vec4");
        let mut total = self.dot_vec4(expressions, body, left_chunk, right_chunk);
        for (left_chunk, right_chunk) in chunks {
            let dot = self.dot_vec4(expressions, body, left_chunk, right_chunk);
            total = self.add(expressions, body, total, dot);
        }
        total
    }

    pub(in crate::lower) fn dot_vec4(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: &[Handle<Expression>],
        right: &[Handle<Expression>],
    ) -> Handle<Expression> {
        let left = self.compose_f32_vec4(expressions, body, left.try_into().unwrap());
        let right = self.compose_f32_vec4(expressions, body, right.try_into().unwrap());
        self.dot_f32_vec4(expressions, body, left, right)
    }
}
