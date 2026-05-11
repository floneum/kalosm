use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn quantized_block_base(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        block: Handle<Expression>,
        col: Handle<Expression>,
        block_words: u32,
        body: &mut Block,
    ) -> Handle<Expression> {
        let blocks_per_col = matrix.rows / matrix.format.block_elements();
        let col_block = self.mul_literal_u32_emitted(e, col, blocks_per_col, body);
        let block_index = self.bin(e, body, BinaryOperator::Add, col_block, block);
        self.mul_literal_u32_emitted(e, block_index, block_words, body)
    }

    pub(in crate::lower) fn load_word(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        offset: u32,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let index = self.add_lit(e, body, base, offset);
        self.load_word_at(e, matrix, index, body)
    }

    pub(in crate::lower) fn load_word_dynamic(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        offset: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let index = self.bin(e, body, BinaryOperator::Add, base, offset);
        self.load_word_at(e, matrix, index, body)
    }

    pub(in crate::lower) fn load_word_pair_dynamic(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        word_base: Handle<Expression>,
        first_offset: u32,
        body: &mut Block,
    ) -> Result<[Handle<Expression>; 2], LowerError> {
        let word0_off = self.add_lit(e, body, word_base, first_offset);
        let word1_off = self.add_lit(e, body, word_base, first_offset + 1);
        Ok([
            self.load_word_dynamic(e, matrix, base, word0_off, body)?,
            self.load_word_dynamic(e, matrix, base, word1_off, body)?,
        ])
    }

    pub(in crate::lower) fn load_word_at(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        index: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let ptr = self.storage_dynamic_pointer(e, &matrix.data, index, body)?;
        Ok(self.emit(e, body, Expression::Load { pointer: ptr }))
    }

    pub(in crate::lower) fn k_scale(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        group: Handle<Expression>,
        min: bool,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let high = self.cmp_lit(e, body, BinaryOperator::GreaterEqual, group, 4);
        let lane = self.and_lit(e, body, group, 3);
        let low_word = self.load_word(e, matrix, base, if min { 3 } else { 2 }, body)?;
        let low_byte = self.byte_at(e, body, low_word, lane);
        let low_scale = self.and_lit(e, body, low_byte, 0x3f);

        let extra_word = self.load_word(e, matrix, base, 4, body)?;
        let extra_byte = self.byte_at(e, body, extra_word, lane);
        let lsb = if min {
            let shifted = self.shr_lit(e, body, extra_byte, 4);
            self.and_lit(e, body, shifted, 0x0f)
        } else {
            self.and_lit(e, body, extra_byte, 0x0f)
        };
        let msb_bits = self.and_lit(e, body, low_byte, 0xc0);
        let msb = self.shr_lit(e, body, msb_bits, 2);
        let high_scale = self.bin(e, body, BinaryOperator::InclusiveOr, lsb, msb);
        Ok(self.select(e, body, high, high_scale, low_scale))
    }

    pub(in crate::lower) fn q4k_scale_min_bytes(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        group: Handle<Expression>,
        body: &mut Block,
    ) -> Result<(Handle<Expression>, Handle<Expression>), LowerError> {
        let high = self.cmp_lit(e, body, BinaryOperator::GreaterEqual, group, 4);
        let lane = self.and_lit(e, body, group, 3);

        let scale_word = self.load_word(e, matrix, base, 2, body)?;
        let min_word = self.load_word(e, matrix, base, 3, body)?;
        let extra_word = self.load_word(e, matrix, base, 4, body)?;

        let scale_low_byte = self.byte_at(e, body, scale_word, lane);
        let min_low_byte = self.byte_at(e, body, min_word, lane);
        let extra_byte = self.byte_at(e, body, extra_word, lane);

        let scale_low = self.and_lit(e, body, scale_low_byte, 0x3f);
        let scale_lsb = self.and_lit(e, body, extra_byte, 0x0f);
        let scale_msb_bits = self.and_lit(e, body, scale_low_byte, 0xc0);
        let scale_msb = self.shr_lit(e, body, scale_msb_bits, 2);
        let scale_high = self.bin(e, body, BinaryOperator::InclusiveOr, scale_lsb, scale_msb);
        let scale = self.select(e, body, high, scale_high, scale_low);

        let min_low = self.and_lit(e, body, min_low_byte, 0x3f);
        let min_lsb_shifted = self.shr_lit(e, body, extra_byte, 4);
        let min_lsb = self.and_lit(e, body, min_lsb_shifted, 0x0f);
        let min_msb_bits = self.and_lit(e, body, min_low_byte, 0xc0);
        let min_msb = self.shr_lit(e, body, min_msb_bits, 2);
        let min_high = self.bin(e, body, BinaryOperator::InclusiveOr, min_lsb, min_msb);
        let min = self.select(e, body, high, min_high, min_low);

        Ok((scale, min))
    }

    pub(in crate::lower) fn q3k_scale(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        group: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let s0 = self.load_word(e, matrix, base, 24, body)?;
        let s1 = self.load_word(e, matrix, base, 25, body)?;
        let s2 = self.load_word(e, matrix, base, 26, body)?;
        let lane = self.and_lit(e, body, group, 3);
        let group_word_bit = self.and_lit(e, body, group, 4);
        let zero = self.u32(e, 0);
        let use_s1 = self.bin(e, body, BinaryOperator::NotEqual, group_word_bit, zero);
        let scale_word = self.select(e, body, use_s1, s1, s0);
        let scale_byte = self.byte_at(e, body, scale_word, lane);
        let low_nibble = self.and_lit(e, body, scale_byte, 0x0f);
        let high_nibble = self.shr_lit(e, body, scale_byte, 4);
        let use_high_nibble = self.cmp_lit(e, body, BinaryOperator::GreaterEqual, group, 8);
        let low = self.select(e, body, use_high_nibble, high_nibble, low_nibble);
        let extra_byte = self.byte_at(e, body, s2, lane);
        let high_shift = self.shr_lit(e, body, group, 2);
        let high_shift = self.shl_lit(e, body, high_shift, 1);
        let high = self.shr(e, body, extra_byte, high_shift);
        let high = self.and_lit(e, body, high, 3);
        let high = self.shl_lit(e, body, high, 4);
        Ok(self.bin(e, body, BinaryOperator::InclusiveOr, low, high))
    }

    pub(in crate::lower) fn byte_at(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        word: Handle<Expression>,
        byte: Handle<Expression>,
    ) -> Handle<Expression> {
        let shift = self.shl_lit(e, body, byte, 3);
        let shifted = self.shr(e, body, word, shift);
        self.and_lit(e, body, shifted, 0xff)
    }

    pub(in crate::lower) fn signed_byte_f32(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        byte: Handle<Expression>,
    ) -> Handle<Expression> {
        let bias = self.u32(e, 128);
        let biased = self.bin(e, body, BinaryOperator::ExclusiveOr, byte, bias);
        let as_i32 = self.as_i32(e, body, biased);
        let offset = self.i32(e, 128);
        let signed = self.sub(e, body, as_i32, offset);
        self.as_f32(e, body, signed)
    }

    pub(in crate::lower) fn emit(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        expr: Expression,
    ) -> Handle<Expression> {
        let value = e.append(expr, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(e, value)),
            Span::default(),
        );
        value
    }

    pub(in crate::lower) fn load_local(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = e.append(Expression::LocalVariable(local), Span::default());
        self.emit(e, body, Expression::Load { pointer })
    }

    pub(in crate::lower) fn bin(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(e, body, Expression::Binary { op, left, right })
    }

    pub(in crate::lower) fn cmp_lit(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, body, op, left, right)
    }

    pub(in crate::lower) fn select(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            body,
            Expression::Select {
                condition,
                accept,
                reject,
            },
        )
    }

    pub(in crate::lower) fn shr(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, body, BinaryOperator::ShiftRight, left, right)
    }

    pub(in crate::lower) fn shr_lit(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, body, BinaryOperator::ShiftRight, left, right)
    }

    pub(in crate::lower) fn shl_lit(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, body, BinaryOperator::ShiftLeft, left, right)
    }

    pub(in crate::lower) fn and_lit(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, body, BinaryOperator::And, left, right)
    }

    pub(in crate::lower) fn add_lit(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        self.add_literal_u32_emitted(e, left, right, body)
    }

    pub(in crate::lower) fn sub(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, body, BinaryOperator::Subtract, left, right)
    }

    pub(in crate::lower) fn mul(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, body, BinaryOperator::Multiply, left, right)
    }

    pub(in crate::lower) fn div(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, body, BinaryOperator::Divide, left, right)
    }

    pub(in crate::lower) fn math1(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        fun: MathFunction,
        arg: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            body,
            Expression::Math {
                fun,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(in crate::lower) fn math2(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            body,
            Expression::Math {
                fun,
                arg,
                arg1: Some(arg1),
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(in crate::lower) fn dot4_i8_packed(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        let dot = self.math2(e, body, MathFunction::Dot4I8Packed, left, right);
        self.as_f32(e, body, dot)
    }

    pub(in crate::lower) fn compose_f32_vec4(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        components: [Handle<Expression>; 4],
    ) -> Handle<Expression> {
        let vec = e.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: components.to_vec(),
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(e, vec)),
            Span::default(),
        );
        vec
    }

    pub(in crate::lower) fn dot_f32_vec4(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.math2(e, body, MathFunction::Dot, left, right)
    }

    pub(in crate::lower) fn vec4_component(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        vector: Handle<Expression>,
        index: u32,
    ) -> Handle<Expression> {
        self.emit(
            e,
            body,
            Expression::AccessIndex {
                base: vector,
                index,
            },
        )
    }

    pub(in crate::lower) fn u8_lane_f32(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        word: Handle<Expression>,
        lane: u32,
    ) -> Handle<Expression> {
        let shifted = if lane == 0 {
            word
        } else {
            self.shr_lit(e, body, word, lane * 8)
        };
        let byte = self.and_lit(e, body, shifted, 0xff);
        self.as_f32(e, body, byte)
    }

    pub(in crate::lower) fn center_q6k_quant(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        quant: Handle<Expression>,
    ) -> Handle<Expression> {
        let quant = self.as_f32(e, body, quant);
        let center = self.f32(e, 32.0);
        self.sub(e, body, quant, center)
    }

    pub(in crate::lower) fn q4k_ggml_accumulate_word_scalar(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        word: Handle<Expression>,
        activations: &[Handle<Expression>],
        pair: usize,
        sums: &mut [Handle<Expression>; 4],
    ) {
        let high_word = self.shr_lit(e, body, word, 16);
        let entries = [
            (word, 0usize, pair * 4, 0x000f_u32),
            (word, 1usize, pair * 4 + 1, 0x0f00_u32),
            (word, 2usize, pair * 4 + 8, 0x00f0_u32),
            (word, 3usize, pair * 4 + 9, 0xf000_u32),
            (high_word, 0usize, pair * 4 + 2, 0x000f_u32),
            (high_word, 1usize, pair * 4 + 3, 0x0f00_u32),
            (high_word, 2usize, pair * 4 + 10, 0x00f0_u32),
            (high_word, 3usize, pair * 4 + 11, 0xf000_u32),
        ];

        for (source, sum_index, activation_index, mask) in entries {
            let masked = self.and_lit(e, body, source, mask);
            let quant = self.as_f32(e, body, masked);
            let term = self.mul(e, body, activations[activation_index], quant);
            sums[sum_index] = self.bin(e, body, BinaryOperator::Add, sums[sum_index], term);
        }
    }

    pub(in crate::lower) fn q4k_ggml_accumulate_word_vector(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        word: Handle<Expression>,
        activations: &[Handle<Expression>],
        pair: usize,
        sums: &mut [Handle<Expression>; 4],
    ) {
        let high_word = self.shr_lit(e, body, word, 16);
        for (source, activation_base) in [(word, pair * 4), (high_word, pair * 4 + 2)] {
            let q0 = self.and_lit(e, body, source, 0x000f);
            let q0 = self.as_f32(e, body, q0);
            let q1 = self.and_lit(e, body, source, 0x0f00);
            let q1 = self.as_f32(e, body, q1);
            let q2 = self.and_lit(e, body, source, 0x00f0);
            let q2 = self.as_f32(e, body, q2);
            let q3 = self.and_lit(e, body, source, 0xf000);
            let q3 = self.as_f32(e, body, q3);
            let quant_vec = self.compose_f32_vec4(e, body, [q0, q1, q2, q3]);
            let activation_vec = self.compose_f32_vec4(
                e,
                body,
                [
                    activations[activation_base],
                    activations[activation_base + 1],
                    activations[activation_base + 8],
                    activations[activation_base + 9],
                ],
            );
            let terms = self.mul(e, body, activation_vec, quant_vec);
            for (sum_index, sum) in sums.iter_mut().enumerate() {
                let term = self.vec4_component(e, body, terms, sum_index as u32);
                *sum = self.bin(e, body, BinaryOperator::Add, *sum, term);
            }
        }
    }

    pub(in crate::lower) fn pack_i8x4(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
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
        body.push(
            Statement::Emit(Self::single_expression_range(e, vec)),
            Span::default(),
        );
        Ok(self.math1(e, body, MathFunction::Pack4xI8Clamp, vec))
    }

    pub(in crate::lower) fn as_i32(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            body,
            Expression::As {
                expr: value,
                kind: ScalarKind::Sint,
                convert: Some(4),
            },
        )
    }

    pub(in crate::lower) fn as_f32(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            body,
            Expression::As {
                expr: value,
                kind: ScalarKind::Float,
                convert: Some(4),
            },
        )
    }

    pub(in crate::lower) fn bitcast_f32(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            body,
            Expression::As {
                expr: value,
                kind: ScalarKind::Float,
                convert: None,
            },
        )
    }

    pub(in crate::lower) fn u32(&self, e: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        e.append(Expression::Literal(Literal::U32(value)), Span::default())
    }

    pub(in crate::lower) fn i32(&self, e: &mut Arena<Expression>, value: i32) -> Handle<Expression> {
        e.append(Expression::Literal(Literal::I32(value)), Span::default())
    }

    pub(in crate::lower) fn f32(&self, e: &mut Arena<Expression>, value: f32) -> Handle<Expression> {
        e.append(Expression::Literal(Literal::F32(value)), Span::default())
    }
}
