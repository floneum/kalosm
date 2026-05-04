use super::*;

impl<'a> Lowerer<'a> {
    fn quantized_block_base(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        block: Handle<Expression>,
        col: Handle<Expression>,
        block_words: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        let blocks_per_col = matrix.rows / matrix.format.block_elements();
        let col_block = self.mul_literal_u32_emitted(e, col, blocks_per_col, emits);
        let block_index = self.bin(e, emits, BinaryOperator::Add, col_block, block);
        self.mul_literal_u32_emitted(e, block_index, block_words, emits)
    }

    pub(super) fn dequantize_qvalue(
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
        let base =
            self.quantized_block_base(expressions, matrix, block, col, block_words, &mut emits);
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

    pub(super) fn dequantize_qvalues(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        n: u32,
    ) -> Result<(Vec<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        let mut emits = Vec::new();
        let mut values = Vec::with_capacity(n as usize);
        for lane in 0..n {
            let k = self.add_literal_u32_emitted(expressions, k_base, lane, &mut emits);
            let (value, mut value_emits) = self.dequantize_qvalue(expressions, matrix, k, col)?;
            emits.append(&mut value_emits);
            values.push(value);
        }
        Ok((values, emits))
    }

    pub(super) fn dequantize_q8_0_values8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<(Vec<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q8_0 {
            return Err(LowerError::UnsupportedOperation(
                "q8_0 vector dequantizer only supports Q8_0",
            ));
        }

        let mut emits = Vec::new();
        let block = self.div_literal_u32_emitted(expressions, k_base, 32, &mut emits);
        let q = self.and_lit(expressions, &mut emits, k_base, 31);
        let col_block =
            self.mul_literal_u32_emitted(expressions, col, matrix.rows / 32, &mut emits);
        let block_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_block,
            block,
        );
        let base = self.mul_literal_u32_emitted(expressions, block_index, 9, &mut emits);
        let scale_word = self.load_word(expressions, matrix, base, 0, &mut emits)?;
        let scale = self.bitcast_f32(expressions, &mut emits, scale_word);
        let q_word = self.shr_lit(expressions, &mut emits, q, 2);
        let word0_off = self.add_lit(expressions, &mut emits, q_word, 1);
        let word1_off = self.add_lit(expressions, &mut emits, q_word, 2);
        let word0 = self.load_word_dynamic(expressions, matrix, base, word0_off, &mut emits)?;
        let word1 = self.load_word_dynamic(expressions, matrix, base, word1_off, &mut emits)?;

        let mut values = Vec::with_capacity(8);
        for lane in 0..8 {
            let byte_lane = expressions.append(
                Expression::Literal(Literal::U32((lane % 4) as u32)),
                Span::default(),
            );
            let word = if lane < 4 { word0 } else { word1 };
            let byte = self.byte_at(expressions, &mut emits, word, byte_lane);
            let signed = self.signed_byte_f32(expressions, &mut emits, byte);
            values.push(self.mul(expressions, &mut emits, signed, scale));
        }
        Ok((values, emits))
    }

    pub(super) fn dequantize_q8_0_dot8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &[Handle<Expression>; 8],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q8_0 {
            return Err(LowerError::UnsupportedOperation(
                "q8_0 dot8 only supports Q8_0",
            ));
        }

        let mut emits = Vec::new();
        let block = self.div_literal_u32_emitted(expressions, k_base, 32, &mut emits);
        let q = self.and_lit(expressions, &mut emits, k_base, 31);
        let col_block =
            self.mul_literal_u32_emitted(expressions, col, matrix.rows / 32, &mut emits);
        let block_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_block,
            block,
        );
        let base = self.mul_literal_u32_emitted(expressions, block_index, 9, &mut emits);
        let scale_word = self.load_word(expressions, matrix, base, 0, &mut emits)?;
        let scale = self.bitcast_f32(expressions, &mut emits, scale_word);
        let q_word = self.shr_lit(expressions, &mut emits, q, 2);
        let word0_off = self.add_lit(expressions, &mut emits, q_word, 1);
        let word1_off = self.add_lit(expressions, &mut emits, q_word, 2);
        let word0 = self.load_word_dynamic(expressions, matrix, base, word0_off, &mut emits)?;
        let word1 = self.load_word_dynamic(expressions, matrix, base, word1_off, &mut emits)?;

        let mut q_components = Vec::with_capacity(8);
        for lane in 0..8 {
            let byte_lane = expressions.append(
                Expression::Literal(Literal::U32((lane % 4) as u32)),
                Span::default(),
            );
            let word = if lane < 4 { word0 } else { word1 };
            let byte = self.byte_at(expressions, &mut emits, word, byte_lane);
            q_components.push(self.signed_byte_f32(expressions, &mut emits, byte));
        }
        let a0 = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: a[..4].to_vec(),
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, a0));
        let q0 = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: q_components[..4].to_vec(),
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, q0));
        let a1 = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: a[4..].to_vec(),
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, a1));
        let q1 = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: q_components[4..].to_vec(),
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, q1));
        let dot0 = expressions.append(
            Expression::Math {
                fun: MathFunction::Dot,
                arg: a0,
                arg1: Some(q0),
                arg2: None,
                arg3: None,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, dot0));
        let dot1 = expressions.append(
            Expression::Math {
                fun: MathFunction::Dot,
                arg: a1,
                arg1: Some(q1),
                arg2: None,
                arg3: None,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, dot1));
        let sum = self.bin(expressions, &mut emits, BinaryOperator::Add, dot0, dot1);
        Ok((self.mul(expressions, &mut emits, sum, scale), emits))
    }

    pub(super) fn dequantize_q4k_values8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<(Vec<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q4K {
            return Err(LowerError::UnsupportedOperation(
                "q4k vector dequantizer only supports Q4K",
            ));
        }

        let mut emits = Vec::new();
        let block = self.div_literal_u32_emitted(expressions, k_base, 256, &mut emits);
        let q_base = self.and_lit(expressions, &mut emits, k_base, 255);
        let base = self.quantized_block_base(expressions, matrix, block, col, 37, &mut emits);

        let d_word = self.load_word(expressions, matrix, base, 0, &mut emits)?;
        let d = self.bitcast_f32(expressions, &mut emits, d_word);
        let dmin_word = self.load_word(expressions, matrix, base, 1, &mut emits)?;
        let dmin = self.bitcast_f32(expressions, &mut emits, dmin_word);
        let group = self.shr_lit(expressions, &mut emits, q_base, 5);
        let scale_byte = self.k_scale(expressions, matrix, base, group, false, &mut emits)?;
        let scale_f = self.as_f32(expressions, &mut emits, scale_byte);
        let scale = self.mul(expressions, &mut emits, scale_f, d);
        let min_byte = self.k_scale(expressions, matrix, base, group, true, &mut emits)?;
        let min_f = self.as_f32(expressions, &mut emits, min_byte);
        let min = self.mul(expressions, &mut emits, min_f, dmin);

        let in_group = self.and_lit(expressions, &mut emits, q_base, 31);
        let group_pair = self.shr_lit(expressions, &mut emits, group, 1);
        let group_pair_offset = self.shl_lit(expressions, &mut emits, group_pair, 5);
        let byte_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            group_pair_offset,
            in_group,
        );
        let data_word = self.shr_lit(expressions, &mut emits, byte_index, 2);
        let word0_off = self.add_lit(expressions, &mut emits, data_word, 5);
        let word1_off = self.add_lit(expressions, &mut emits, data_word, 6);
        let word0 = self.load_word_dynamic(expressions, matrix, base, word0_off, &mut emits)?;
        let word1 = self.load_word_dynamic(expressions, matrix, base, word1_off, &mut emits)?;
        let group_low = self.and_lit(expressions, &mut emits, group, 1);
        let high = self.cmp_lit(
            expressions,
            &mut emits,
            BinaryOperator::NotEqual,
            group_low,
            0,
        );

        let mut values = Vec::with_capacity(8);
        for lane in 0..8 {
            let byte_lane = expressions.append(
                Expression::Literal(Literal::U32((lane % 4) as u32)),
                Span::default(),
            );
            let word = if lane < 4 { word0 } else { word1 };
            let byte = self.byte_at(expressions, &mut emits, word, byte_lane);
            let byte_hi = self.shr_lit(expressions, &mut emits, byte, 4);
            let byte_lo = self.and_lit(expressions, &mut emits, byte, 0x0f);
            let quant = self.select(expressions, &mut emits, high, byte_hi, byte_lo);
            let quant_f = self.as_f32(expressions, &mut emits, quant);
            let scaled = self.mul(expressions, &mut emits, quant_f, scale);
            values.push(self.sub(expressions, &mut emits, scaled, min));
        }
        Ok((values, emits))
    }

    pub(super) fn dequantize_q5_0_values16(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<(Vec<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q5_0 {
            return Err(LowerError::UnsupportedOperation(
                "q5_0 vector dequantizer only supports Q5_0",
            ));
        }

        let mut emits = Vec::new();
        let block = self.div_literal_u32_emitted(expressions, k_base, 32, &mut emits);
        let q_base = self.and_lit(expressions, &mut emits, k_base, 31);
        let col_block =
            self.mul_literal_u32_emitted(expressions, col, matrix.rows / 32, &mut emits);
        let block_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            col_block,
            block,
        );
        let base = self.mul_literal_u32_emitted(expressions, block_index, 6, &mut emits);
        let scale_word = self.load_word(expressions, matrix, base, 0, &mut emits)?;
        let scale = self.bitcast_f32(expressions, &mut emits, scale_word);
        let qh = self.load_word(expressions, matrix, base, 1, &mut emits)?;
        let high = self.cmp_lit(
            expressions,
            &mut emits,
            BinaryOperator::GreaterEqual,
            q_base,
            16,
        );
        let sixteen = expressions.append(Expression::Literal(Literal::U32(16)), Span::default());
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        let high_base = self.select(expressions, &mut emits, high, sixteen, zero);
        let words = [
            self.load_word(expressions, matrix, base, 2, &mut emits)?,
            self.load_word(expressions, matrix, base, 3, &mut emits)?,
            self.load_word(expressions, matrix, base, 4, &mut emits)?,
            self.load_word(expressions, matrix, base, 5, &mut emits)?,
        ];

        let mut values = Vec::with_capacity(16);
        for lane in 0..16 {
            let byte_lane = expressions.append(
                Expression::Literal(Literal::U32((lane % 4) as u32)),
                Span::default(),
            );
            let byte = self.byte_at(expressions, &mut emits, words[lane / 4], byte_lane);
            let low = self.and_lit(expressions, &mut emits, byte, 0x0f);
            let high4 = self.shr_lit(expressions, &mut emits, byte, 4);
            let low4 = self.select(expressions, &mut emits, high, high4, low);
            let lane_index = self.add_literal_u32(expressions, high_base, lane as u32);
            let shifted_qh = self.shr(expressions, &mut emits, qh, lane_index);
            let hi_bit_low = self.and_lit(expressions, &mut emits, shifted_qh, 1);
            let hi_bit = self.shl_lit(expressions, &mut emits, hi_bit_low, 4);
            let quant = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::InclusiveOr,
                low4,
                hi_bit,
            );
            let quant_f = self.as_f32(expressions, &mut emits, quant);
            let center = self.f32(expressions, 16.0);
            let centered = self.sub(expressions, &mut emits, quant_f, center);
            values.push(self.mul(expressions, &mut emits, centered, scale));
        }
        Ok((values, emits))
    }

    pub(super) fn dequantize_q6k_values8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<(Vec<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q6K {
            return Err(LowerError::UnsupportedOperation(
                "q6k vector dequantizer only supports Q6K",
            ));
        }

        let mut emits = Vec::new();
        let block = self.div_literal_u32_emitted(expressions, k_base, 256, &mut emits);
        let q_base = self.and_lit(expressions, &mut emits, k_base, 255);
        let base = self.quantized_block_base(expressions, matrix, block, col, 53, &mut emits);

        let d_word = self.load_word(expressions, matrix, base, 52, &mut emits)?;
        let d = self.bitcast_f32(expressions, &mut emits, d_word);
        let chunk = self.shr_lit(expressions, &mut emits, q_base, 7);
        let local = self.and_lit(expressions, &mut emits, q_base, 127);
        let high_byte_index = self.and_lit(expressions, &mut emits, local, 31);
        let low_group = self.shr_lit(expressions, &mut emits, local, 5);

        let chunk_low_base = self.shl_lit(expressions, &mut emits, chunk, 6);
        let low_group_parity = self.and_lit(expressions, &mut emits, low_group, 1);
        let low_group_offset = self.shl_lit(expressions, &mut emits, low_group_parity, 5);
        let local_low_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            high_byte_index,
            low_group_offset,
        );
        let lower_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            chunk_low_base,
            local_low_index,
        );
        let low_word_base = self.shr_lit(expressions, &mut emits, lower_index, 2);
        let low_word1_off = self.add_lit(expressions, &mut emits, low_word_base, 1);
        let low_word0 =
            self.load_word_dynamic(expressions, matrix, base, low_word_base, &mut emits)?;
        let low_word1 =
            self.load_word_dynamic(expressions, matrix, base, low_word1_off, &mut emits)?;
        let low_shift = self.shr_lit(expressions, &mut emits, low_group, 1);
        let low_shift = self.shl_lit(expressions, &mut emits, low_shift, 2);

        let high_chunk_base = self.shl_lit(expressions, &mut emits, chunk, 5);
        let high_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            high_chunk_base,
            high_byte_index,
        );
        let high_word_base = self.shr_lit(expressions, &mut emits, high_index, 2);
        let high_word0_off = self.add_lit(expressions, &mut emits, high_word_base, 32);
        let high_word1_off = self.add_lit(expressions, &mut emits, high_word_base, 33);
        let high_word0 =
            self.load_word_dynamic(expressions, matrix, base, high_word0_off, &mut emits)?;
        let high_word1 =
            self.load_word_dynamic(expressions, matrix, base, high_word1_off, &mut emits)?;
        let high_shift = self.shl_lit(expressions, &mut emits, low_group, 1);

        let scale_chunk_base = self.shl_lit(expressions, &mut emits, chunk, 3);
        let high_byte_half = self.shr_lit(expressions, &mut emits, high_byte_index, 4);
        let low_group_scale = self.shl_lit(expressions, &mut emits, low_group, 1);
        let local_scale_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            high_byte_half,
            low_group_scale,
        );
        let scale_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            scale_chunk_base,
            local_scale_index,
        );
        let scale_word_base = self.shr_lit(expressions, &mut emits, scale_index, 2);
        let scale_word_off = self.add_lit(expressions, &mut emits, scale_word_base, 48);
        let scale_word =
            self.load_word_dynamic(expressions, matrix, base, scale_word_off, &mut emits)?;
        let scale_lane = self.and_lit(expressions, &mut emits, scale_index, 3);
        let scale_byte = self.byte_at(expressions, &mut emits, scale_word, scale_lane);
        let scale = self.signed_byte_f32(expressions, &mut emits, scale_byte);
        let scale = self.mul(expressions, &mut emits, scale, d);

        let mut values = Vec::with_capacity(8);
        for lane in 0..8 {
            let byte_lane = expressions.append(
                Expression::Literal(Literal::U32((lane % 4) as u32)),
                Span::default(),
            );
            let low_word = if lane < 4 { low_word0 } else { low_word1 };
            let low_byte = self.byte_at(expressions, &mut emits, low_word, byte_lane);
            let low_shifted = self.shr(expressions, &mut emits, low_byte, low_shift);
            let low4 = self.and_lit(expressions, &mut emits, low_shifted, 0x0f);

            let high_word = if lane < 4 { high_word0 } else { high_word1 };
            let high_byte = self.byte_at(expressions, &mut emits, high_word, byte_lane);
            let high_shifted = self.shr(expressions, &mut emits, high_byte, high_shift);
            let high2 = self.and_lit(expressions, &mut emits, high_shifted, 3);
            let high2 = self.shl_lit(expressions, &mut emits, high2, 4);
            let quant = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::InclusiveOr,
                low4,
                high2,
            );
            let quant_f = self.as_f32(expressions, &mut emits, quant);
            let center = self.f32(expressions, 32.0);
            let centered = self.sub(expressions, &mut emits, quant_f, center);
            values.push(self.mul(expressions, &mut emits, centered, scale));
        }
        Ok((values, emits))
    }

    pub(super) fn dequantize_q6k_values16(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Result<(Vec<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        let (mut values, mut emits) =
            self.dequantize_q6k_values8(expressions, matrix, k_base, col)?;
        let k_base_hi = self.add_lit(expressions, &mut emits, k_base, 8);
        let (hi_values, hi_emits) =
            self.dequantize_q6k_values8(expressions, matrix, k_base_hi, col)?;
        emits.extend(hi_emits);
        values.extend(hi_values);
        Ok((values, emits))
    }

    pub(super) fn q4k_q8_activation_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &Q8ActivationPacks,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if a.len == 0 || !a.len.is_multiple_of(2) {
            return Err(LowerError::UnsupportedOperation(
                "q4k x q8 activation dot requires an even number of activation packs",
            ));
        }

        let mut emits = Vec::new();
        let mut total = self.f32(expressions, 0.0);
        for pack_offset in (0..a.len).step_by(2) {
            let k = self.add_lit(expressions, &mut emits, k_base, (pack_offset * 4) as u32);
            let (chunk, chunk_emits) =
                self.q4k_q8_activation_dot8(expressions, matrix, k, col, a, pack_offset)?;
            emits.extend(chunk_emits);
            total = self.bin(expressions, &mut emits, BinaryOperator::Add, total, chunk);
        }
        Ok((total, emits))
    }

    fn q4k_q8_activation_dot8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &Q8ActivationPacks,
        pack_offset: usize,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q4K || a.len < pack_offset + 2 {
            return Err(LowerError::UnsupportedOperation(
                "q4k x q8 activation dot requires Q4K and two activation packs",
            ));
        }

        let mut emits = Vec::new();
        let (b_scale, b_min, b_packs) =
            self.q4k_quant_packs8(expressions, matrix, k_base, col, &mut emits)?;
        let mut total = self.f32(expressions, 0.0);
        for i in 0..2 {
            let a_pack_index = pack_offset + i;
            let a_pack = self.load_local(expressions, &mut emits, a.packs[a_pack_index]);
            let dot = self.dot4_i8_packed(expressions, &mut emits, a_pack, b_packs[i]);
            let scaled_dot = self.mul(expressions, &mut emits, dot, b_scale);
            let a_sum_i32 = self.load_local(expressions, &mut emits, a.sums_i32[a_pack_index]);
            let a_sum = self.as_f32(expressions, &mut emits, a_sum_i32);
            let min_term = self.mul(expressions, &mut emits, a_sum, b_min);
            let unscaled = self.sub(expressions, &mut emits, scaled_dot, min_term);
            let a_scale = self.load_local(expressions, &mut emits, a.scales[a_pack_index]);
            let chunk = self.mul(expressions, &mut emits, unscaled, a_scale);
            total = self.bin(expressions, &mut emits, BinaryOperator::Add, total, chunk);
        }
        Ok((total, emits))
    }

    pub(super) fn q6k_q8_activation_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &Q8ActivationPacks,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q6K {
            return Err(LowerError::UnsupportedOperation(
                "q6k x q8 activation dot only supports Q6K",
            ));
        }

        if a.len == 0 || !a.len.is_multiple_of(2) {
            return Err(LowerError::UnsupportedOperation(
                "q6k x q8 activation dot requires an even number of activation packs",
            ));
        }

        let mut emits = Vec::new();
        let mut total = self.f32(expressions, 0.0);
        for pack_offset in (0..a.len).step_by(2) {
            let k = self.add_lit(expressions, &mut emits, k_base, (pack_offset * 4) as u32);
            let (chunk, chunk_emits) =
                self.q6k_q8_activation_dot8(expressions, matrix, k, col, a, pack_offset)?;
            emits.extend(chunk_emits);
            total = self.bin(expressions, &mut emits, BinaryOperator::Add, total, chunk);
        }
        Ok((total, emits))
    }

    fn q6k_q8_activation_dot8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &Q8ActivationPacks,
        pack_offset: usize,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if a.len < pack_offset + 2 {
            return Err(LowerError::UnsupportedOperation(
                "q6k x q8 activation dot8 requires two activation packs",
            ));
        }

        let mut emits = Vec::new();
        let (b_scale, b_packs) =
            self.q6k_quant_packs8(expressions, matrix, k_base, col, &mut emits)?;
        let mut total = self.f32(expressions, 0.0);
        for i in 0..2 {
            let a_pack = pack_offset + i;
            let a_pack_value = self.load_local(expressions, &mut emits, a.packs[a_pack]);
            let dot = self.dot4_i8_packed(expressions, &mut emits, a_pack_value, b_packs[i]);
            let scaled = self.mul(expressions, &mut emits, dot, b_scale);
            let a_scale = self.load_local(expressions, &mut emits, a.scales[a_pack]);
            let chunk = self.mul(expressions, &mut emits, scaled, a_scale);
            total = self.bin(expressions, &mut emits, BinaryOperator::Add, total, chunk);
        }
        Ok((total, emits))
    }

    pub(super) fn cached_q8_activation_packs(
        &self,
        e: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        a: &[Handle<Expression>],
    ) -> Result<Q8ActivationPacks, LowerError> {
        let key = a.to_vec();
        if let Some(packs) = self.q8_activation_pack_cache.borrow().get(&key).cloned() {
            return Ok(packs);
        }

        let mut emits = Vec::new();
        let values = self.q8_activation_pack_values(e, a, &mut emits)?;
        let packs = Self::q8_activation_pack_locals(scratch, values.packs.len())?;
        self.q8_activation_pack_cache.borrow_mut().clear();
        Self::push_emits(body, emits);
        Self::store_q8_activation_pack_values(e, body, &packs, values);
        self.q8_activation_pack_cache
            .borrow_mut()
            .insert(key, packs.clone());
        Ok(packs)
    }

    pub(super) fn q8_activation_pack_values(
        &self,
        e: &mut Arena<Expression>,
        a: &[Handle<Expression>],
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Q8ActivationPackValues, LowerError> {
        if a.is_empty() || !a.len().is_multiple_of(4) {
            return Err(LowerError::UnsupportedOperation(
                "q8 activation packing requires a non-empty multiple of 4",
            ));
        }

        let qmax = self.f32(e, 127.0);

        let mut scales = Vec::with_capacity(a.len() / 4);
        let mut sums_i32 = Vec::with_capacity(a.len() / 4);
        let mut packs = Vec::with_capacity(a.len() / 4);
        for chunk in a.chunks(4) {
            let mut max_abs = self.f32(e, 0.0);
            for value in chunk {
                let abs = self.math1(e, emits, MathFunction::Abs, *value);
                max_abs = self.math2(e, emits, MathFunction::Max, max_abs, abs);
            }
            let epsilon = self.f32(e, 1.0e-8);
            max_abs = self.math2(e, emits, MathFunction::Max, max_abs, epsilon);
            let inv_scale = self.div(e, emits, qmax, max_abs);
            let scale = self.div(e, emits, max_abs, qmax);
            let mut sum_i32 = self.i32(e, 0);
            let mut packed_values = Vec::with_capacity(4);
            for (lane, value) in chunk.iter().enumerate() {
                let scaled = self.mul(e, emits, *value, inv_scale);
                let rounded = self.math1(e, emits, MathFunction::Round, scaled);
                let lo = self.f32(e, -127.0);
                let hi = self.f32(e, 127.0);
                let clamped = self.math2(e, emits, MathFunction::Min, rounded, hi);
                let clamped = self.math2(e, emits, MathFunction::Max, clamped, lo);
                let q_i32 = self.as_i32(e, emits, clamped);
                sum_i32 = self.bin(e, emits, BinaryOperator::Add, sum_i32, q_i32);
                debug_assert!(lane < 4);
                packed_values.push(q_i32);
            }
            scales.push(scale);
            sums_i32.push(sum_i32);
            packs.push(self.pack_i8x4(e, emits, packed_values)?);
        }

        Ok(Q8ActivationPackValues {
            scales,
            packs,
            sums_i32,
        })
    }

    fn q8_activation_pack_locals(
        scratch: ScratchLocals,
        len: usize,
    ) -> Result<Q8ActivationPacks, LowerError> {
        if len > scratch.q8_activation_packs.len() {
            return Err(LowerError::UnsupportedOperation(
                "q8 activation packing supports at most four packs",
            ));
        }

        Ok(Q8ActivationPacks {
            len,
            scales: scratch.q8_activation_scales,
            packs: scratch.q8_activation_packs,
            sums_i32: scratch.q8_activation_sums_i32,
        })
    }

    fn store_q8_activation_pack_values(
        e: &mut Arena<Expression>,
        body: &mut Block,
        locals: &Q8ActivationPacks,
        values: Q8ActivationPackValues,
    ) {
        debug_assert_eq!(locals.len, values.scales.len());
        debug_assert_eq!(locals.len, values.packs.len());
        debug_assert_eq!(locals.len, values.sums_i32.len());

        for i in 0..locals.len {
            let scale_ptr = e.append(Expression::LocalVariable(locals.scales[i]), Span::default());
            body.push(
                Statement::Store {
                    pointer: scale_ptr,
                    value: values.scales[i],
                },
                Span::default(),
            );

            let pack_ptr = e.append(Expression::LocalVariable(locals.packs[i]), Span::default());
            body.push(
                Statement::Store {
                    pointer: pack_ptr,
                    value: values.packs[i],
                },
                Span::default(),
            );

            let sum_ptr = e.append(
                Expression::LocalVariable(locals.sums_i32[i]),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: sum_ptr,
                    value: values.sums_i32[i],
                },
                Span::default(),
            );
        }
    }

    fn q4k_quant_packs8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<
        (
            Handle<Expression>,
            Handle<Expression>,
            [Handle<Expression>; 2],
        ),
        LowerError,
    > {
        let block = self.div_literal_u32_emitted(expressions, k_base, 256, emits);
        let q_base = self.and_lit(expressions, emits, k_base, 255);
        let base = self.quantized_block_base(expressions, matrix, block, col, 37, emits);

        let d_word = self.load_word(expressions, matrix, base, 0, emits)?;
        let d = self.bitcast_f32(expressions, emits, d_word);
        let dmin_word = self.load_word(expressions, matrix, base, 1, emits)?;
        let dmin = self.bitcast_f32(expressions, emits, dmin_word);
        let group = self.shr_lit(expressions, emits, q_base, 5);
        let scale_byte = self.k_scale(expressions, matrix, base, group, false, emits)?;
        let scale_f = self.as_f32(expressions, emits, scale_byte);
        let scale = self.mul(expressions, emits, scale_f, d);
        let min_byte = self.k_scale(expressions, matrix, base, group, true, emits)?;
        let min_f = self.as_f32(expressions, emits, min_byte);
        let min = self.mul(expressions, emits, min_f, dmin);

        let in_group = self.and_lit(expressions, emits, q_base, 31);
        let group_pair = self.shr_lit(expressions, emits, group, 1);
        let group_pair_offset = self.shl_lit(expressions, emits, group_pair, 5);
        let byte_index = self.bin(
            expressions,
            emits,
            BinaryOperator::Add,
            group_pair_offset,
            in_group,
        );
        let data_word = self.shr_lit(expressions, emits, byte_index, 2);
        let word0_off = self.add_lit(expressions, emits, data_word, 5);
        let word1_off = self.add_lit(expressions, emits, data_word, 6);
        let word0 = self.load_word_dynamic(expressions, matrix, base, word0_off, emits)?;
        let word1 = self.load_word_dynamic(expressions, matrix, base, word1_off, emits)?;
        let group_low = self.and_lit(expressions, emits, group, 1);
        let high = self.cmp_lit(expressions, emits, BinaryOperator::NotEqual, group_low, 0);

        let packs = std::array::from_fn(|chunk| {
            let mut packed_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let source_lane = chunk * 4 + lane;
                let byte_lane = expressions.append(
                    Expression::Literal(Literal::U32((source_lane % 4) as u32)),
                    Span::default(),
                );
                let word = if source_lane < 4 { word0 } else { word1 };
                let byte = self.byte_at(expressions, emits, word, byte_lane);
                let byte_hi = self.shr_lit(expressions, emits, byte, 4);
                let byte_lo = self.and_lit(expressions, emits, byte, 0x0f);
                let quant = self.select(expressions, emits, high, byte_hi, byte_lo);
                packed_values.push(self.as_i32(expressions, emits, quant));
            }
            self.pack_i8x4(expressions, emits, packed_values)
                .expect("q4k packs exactly four i8 values")
        });
        Ok((scale, min, packs))
    }

    fn q6k_quant_packs8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<(Handle<Expression>, [Handle<Expression>; 2]), LowerError> {
        let block = self.div_literal_u32_emitted(expressions, k_base, 256, emits);
        let q_base = self.and_lit(expressions, emits, k_base, 255);
        let base = self.quantized_block_base(expressions, matrix, block, col, 53, emits);

        let d_word = self.load_word(expressions, matrix, base, 52, emits)?;
        let d = self.bitcast_f32(expressions, emits, d_word);
        let chunk = self.shr_lit(expressions, emits, q_base, 7);
        let local = self.and_lit(expressions, emits, q_base, 127);
        let high_byte_index = self.and_lit(expressions, emits, local, 31);
        let low_group = self.shr_lit(expressions, emits, local, 5);

        let chunk_low_base = self.shl_lit(expressions, emits, chunk, 6);
        let low_group_parity = self.and_lit(expressions, emits, low_group, 1);
        let low_group_offset = self.shl_lit(expressions, emits, low_group_parity, 5);
        let local_low_index = self.bin(
            expressions,
            emits,
            BinaryOperator::Add,
            high_byte_index,
            low_group_offset,
        );
        let lower_index = self.bin(
            expressions,
            emits,
            BinaryOperator::Add,
            chunk_low_base,
            local_low_index,
        );
        let low_word_base = self.shr_lit(expressions, emits, lower_index, 2);
        let low_word1_off = self.add_lit(expressions, emits, low_word_base, 1);
        let low_word0 = self.load_word_dynamic(expressions, matrix, base, low_word_base, emits)?;
        let low_word1 = self.load_word_dynamic(expressions, matrix, base, low_word1_off, emits)?;
        let low_shift = self.shr_lit(expressions, emits, low_group, 1);
        let low_shift = self.shl_lit(expressions, emits, low_shift, 2);

        let high_chunk_base = self.shl_lit(expressions, emits, chunk, 5);
        let high_index = self.bin(
            expressions,
            emits,
            BinaryOperator::Add,
            high_chunk_base,
            high_byte_index,
        );
        let high_word_base = self.shr_lit(expressions, emits, high_index, 2);
        let high_word0_off = self.add_lit(expressions, emits, high_word_base, 32);
        let high_word1_off = self.add_lit(expressions, emits, high_word_base, 33);
        let high_word0 =
            self.load_word_dynamic(expressions, matrix, base, high_word0_off, emits)?;
        let high_word1 =
            self.load_word_dynamic(expressions, matrix, base, high_word1_off, emits)?;
        let high_shift = self.shl_lit(expressions, emits, low_group, 1);

        let scale_chunk_base = self.shl_lit(expressions, emits, chunk, 3);
        let high_byte_half = self.shr_lit(expressions, emits, high_byte_index, 4);
        let low_group_scale = self.shl_lit(expressions, emits, low_group, 1);
        let local_scale_index = self.bin(
            expressions,
            emits,
            BinaryOperator::Add,
            high_byte_half,
            low_group_scale,
        );
        let scale_index = self.bin(
            expressions,
            emits,
            BinaryOperator::Add,
            scale_chunk_base,
            local_scale_index,
        );
        let scale_word_base = self.shr_lit(expressions, emits, scale_index, 2);
        let scale_word_off = self.add_lit(expressions, emits, scale_word_base, 48);
        let scale_word =
            self.load_word_dynamic(expressions, matrix, base, scale_word_off, emits)?;
        let scale_lane = self.and_lit(expressions, emits, scale_index, 3);
        let scale_byte = self.byte_at(expressions, emits, scale_word, scale_lane);
        let scale = self.signed_byte_f32(expressions, emits, scale_byte);
        let scale = self.mul(expressions, emits, scale, d);

        let packs = std::array::from_fn(|chunk| {
            let mut packed_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let source_lane = chunk * 4 + lane;
                let byte_lane = expressions.append(
                    Expression::Literal(Literal::U32((source_lane % 4) as u32)),
                    Span::default(),
                );
                let low_word = if source_lane < 4 {
                    low_word0
                } else {
                    low_word1
                };
                let low_byte = self.byte_at(expressions, emits, low_word, byte_lane);
                let low_shifted = self.shr(expressions, emits, low_byte, low_shift);
                let low4 = self.and_lit(expressions, emits, low_shifted, 0x0f);

                let high_word = if source_lane < 4 {
                    high_word0
                } else {
                    high_word1
                };
                let high_byte = self.byte_at(expressions, emits, high_word, byte_lane);
                let high_shifted = self.shr(expressions, emits, high_byte, high_shift);
                let high2 = self.and_lit(expressions, emits, high_shifted, 3);
                let high2 = self.shl_lit(expressions, emits, high2, 4);
                let quant = self.bin(expressions, emits, BinaryOperator::InclusiveOr, low4, high2);
                let quant_i32 = self.as_i32(expressions, emits, quant);
                let center = self.i32(expressions, 32);
                let centered = self.bin(
                    expressions,
                    emits,
                    BinaryOperator::Subtract,
                    quant_i32,
                    center,
                );
                packed_values.push(centered);
            }
            self.pack_i8x4(expressions, emits, packed_values)
                .expect("q6k packs exactly four i8 values")
        });
        Ok((scale, packs))
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

    fn load_local(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = e.append(Expression::LocalVariable(local), Span::default());
        self.emit(e, emits, Expression::Load { pointer })
    }

    pub(super) fn bin(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(e, emits, Expression::Binary { op, left, right })
    }

    pub(super) fn cmp_lit(
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

    fn div(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, emits, BinaryOperator::Divide, left, right)
    }

    fn math1(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        fun: MathFunction,
        arg: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::Math {
                fun,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        )
    }

    fn math2(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::Math {
                fun,
                arg,
                arg1: Some(arg1),
                arg2: None,
                arg3: None,
            },
        )
    }

    fn dot4_i8_packed(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        let dot = self.emit(
            e,
            emits,
            Expression::Math {
                fun: MathFunction::Dot4I8Packed,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
        );
        self.as_f32(e, emits, dot)
    }

    fn pack_i8x4(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        values: Vec<Handle<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let components: [Handle<Expression>; 4] = values
            .try_into()
            .map_err(|_| LowerError::UnsupportedOperation("pack_i8x4 requires 4 values"))?;
        let vec = e.append(
            Expression::Compose {
                ty: self.i32_vec4_ty,
                components: components.to_vec(),
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(e, vec));
        Ok(self.emit(
            e,
            emits,
            Expression::Math {
                fun: MathFunction::Pack4xI8Clamp,
                arg: vec,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        ))
    }

    fn as_i32(
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
                kind: ScalarKind::Sint,
                convert: Some(4),
            },
        )
    }

    pub(super) fn as_f32(
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

    fn i32(&self, e: &mut Arena<Expression>, value: i32) -> Handle<Expression> {
        e.append(Expression::Literal(Literal::I32(value)), Span::default())
    }

    fn f32(&self, e: &mut Arena<Expression>, value: f32) -> Handle<Expression> {
        e.append(Expression::Literal(Literal::F32(value)), Span::default())
    }
}
