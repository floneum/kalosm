use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn quantized_block_base(
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

    pub(in crate::lower) fn load_word(
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

    pub(in crate::lower) fn load_word_dynamic(
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

    pub(in crate::lower) fn load_word_pair_dynamic(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        word_base: Handle<Expression>,
        first_offset: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<[Handle<Expression>; 2], LowerError> {
        let word0_off = self.add_lit(e, emits, word_base, first_offset);
        let word1_off = self.add_lit(e, emits, word_base, first_offset + 1);
        Ok([
            self.load_word_dynamic(e, matrix, base, word0_off, emits)?,
            self.load_word_dynamic(e, matrix, base, word1_off, emits)?,
        ])
    }

    pub(in crate::lower) fn load_word_at(
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

    pub(in crate::lower) fn k_scale(
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

    pub(in crate::lower) fn q4k_scale_min_bytes(
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
        let min_word = self.load_word(e, matrix, base, 3, emits)?;
        let extra_word = self.load_word(e, matrix, base, 4, emits)?;

        let scale_low_byte = self.byte_at(e, emits, scale_word, lane);
        let min_low_byte = self.byte_at(e, emits, min_word, lane);
        let extra_byte = self.byte_at(e, emits, extra_word, lane);

        let scale_low = self.and_lit(e, emits, scale_low_byte, 0x3f);
        let scale_lsb = self.and_lit(e, emits, extra_byte, 0x0f);
        let scale_msb_bits = self.and_lit(e, emits, scale_low_byte, 0xc0);
        let scale_msb = self.shr_lit(e, emits, scale_msb_bits, 2);
        let scale_high = self.bin(e, emits, BinaryOperator::InclusiveOr, scale_lsb, scale_msb);
        let scale = self.select(e, emits, high, scale_high, scale_low);

        let min_low = self.and_lit(e, emits, min_low_byte, 0x3f);
        let min_lsb_shifted = self.shr_lit(e, emits, extra_byte, 4);
        let min_lsb = self.and_lit(e, emits, min_lsb_shifted, 0x0f);
        let min_msb_bits = self.and_lit(e, emits, min_low_byte, 0xc0);
        let min_msb = self.shr_lit(e, emits, min_msb_bits, 2);
        let min_high = self.bin(e, emits, BinaryOperator::InclusiveOr, min_lsb, min_msb);
        let min = self.select(e, emits, high, min_high, min_low);

        Ok((scale, min))
    }

    pub(in crate::lower) fn q3k_scale(
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

    pub(in crate::lower) fn byte_at(
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

    pub(in crate::lower) fn signed_byte_f32(
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

    pub(in crate::lower) fn emit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        expr: Expression,
    ) -> Handle<Expression> {
        let value = e.append(expr, Span::default());
        emits.push(Self::single_expression_range(e, value));
        value
    }

    pub(in crate::lower) fn load_local(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = e.append(Expression::LocalVariable(local), Span::default());
        self.emit(e, emits, Expression::Load { pointer })
    }

    pub(in crate::lower) fn bin(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(e, emits, Expression::Binary { op, left, right })
    }

    pub(in crate::lower) fn cmp_lit(
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

    pub(in crate::lower) fn select(
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

    pub(in crate::lower) fn shr(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, emits, BinaryOperator::ShiftRight, left, right)
    }

    pub(in crate::lower) fn shr_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, emits, BinaryOperator::ShiftRight, left, right)
    }

    pub(in crate::lower) fn shl_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, emits, BinaryOperator::ShiftLeft, left, right)
    }

    pub(in crate::lower) fn and_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = self.u32(e, right);
        self.bin(e, emits, BinaryOperator::And, left, right)
    }

    pub(in crate::lower) fn add_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        self.add_literal_u32_emitted(e, left, right, emits)
    }

    pub(in crate::lower) fn sub(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, emits, BinaryOperator::Subtract, left, right)
    }

    pub(in crate::lower) fn mul(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, emits, BinaryOperator::Multiply, left, right)
    }

    pub(in crate::lower) fn div(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(e, emits, BinaryOperator::Divide, left, right)
    }

    pub(in crate::lower) fn math1(
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

    pub(in crate::lower) fn math2(
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

    pub(in crate::lower) fn dot4_i8_packed(
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

    pub(in crate::lower) fn compose_f32_vec4(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        components: [Handle<Expression>; 4],
    ) -> Handle<Expression> {
        let vec = e.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: components.to_vec(),
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(e, vec));
        vec
    }

    pub(in crate::lower) fn dot_f32_vec4(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            e,
            emits,
            Expression::Math {
                fun: MathFunction::Dot,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(in crate::lower) fn vec4_component(
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

    pub(in crate::lower) fn u8_lane_f32(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        word: Handle<Expression>,
        lane: u32,
    ) -> Handle<Expression> {
        let shifted = if lane == 0 {
            word
        } else {
            self.shr_lit(e, emits, word, lane * 8)
        };
        let byte = self.and_lit(e, emits, shifted, 0xff);
        self.as_f32(e, emits, byte)
    }

    pub(in crate::lower) fn center_q6k_quant(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        quant: Handle<Expression>,
    ) -> Handle<Expression> {
        let quant = self.as_f32(e, emits, quant);
        let center = self.f32(e, 32.0);
        self.sub(e, emits, quant, center)
    }

    pub(in crate::lower) fn q4k_ggml_accumulate_word_scalar(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        word: Handle<Expression>,
        activations: &[Handle<Expression>],
        pair: usize,
        sums: &mut [Handle<Expression>; 4],
    ) {
        let high_word = self.shr_lit(e, emits, word, 16);
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
            let masked = self.and_lit(e, emits, source, mask);
            let quant = self.as_f32(e, emits, masked);
            let term = self.mul(e, emits, activations[activation_index], quant);
            sums[sum_index] = self.bin(e, emits, BinaryOperator::Add, sums[sum_index], term);
        }
    }

    pub(in crate::lower) fn q4k_ggml_accumulate_word_vector(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        word: Handle<Expression>,
        activations: &[Handle<Expression>],
        pair: usize,
        sums: &mut [Handle<Expression>; 4],
    ) {
        let high_word = self.shr_lit(e, emits, word, 16);
        for (source, activation_base) in [(word, pair * 4), (high_word, pair * 4 + 2)] {
            let q0 = self.and_lit(e, emits, source, 0x000f);
            let q0 = self.as_f32(e, emits, q0);
            let q1 = self.and_lit(e, emits, source, 0x0f00);
            let q1 = self.as_f32(e, emits, q1);
            let q2 = self.and_lit(e, emits, source, 0x00f0);
            let q2 = self.as_f32(e, emits, q2);
            let q3 = self.and_lit(e, emits, source, 0xf000);
            let q3 = self.as_f32(e, emits, q3);
            let quant_vec = self.compose_f32_vec4(e, emits, [q0, q1, q2, q3]);
            let activation_vec = self.compose_f32_vec4(
                e,
                emits,
                [
                    activations[activation_base],
                    activations[activation_base + 1],
                    activations[activation_base + 8],
                    activations[activation_base + 9],
                ],
            );
            let terms = self.mul(e, emits, activation_vec, quant_vec);
            for (sum_index, sum) in sums.iter_mut().enumerate() {
                let term = self.vec4_component(e, emits, terms, sum_index as u32);
                *sum = self.bin(e, emits, BinaryOperator::Add, *sum, term);
            }
        }
    }

    pub(in crate::lower) fn pack_i8x4(
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

    pub(in crate::lower) fn as_i32(
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

    pub(in crate::lower) fn as_f32(
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

    pub(in crate::lower) fn bitcast_f32(
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
