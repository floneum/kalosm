use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn q4k_ggml_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        block: Handle<Expression>,
        iq: Handle<Expression>,
        ir: Handle<Expression>,
        col: Handle<Expression>,
        a_low: &[Handle<Expression>],
        a_high: &[Handle<Expression>],
        sums: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q4K
            || a_low.len() != 16
            || a_high.len() != 16
            || sums.len() != 4
        {
            return Err(LowerError::UnsupportedOperation(
                "q4k ggml dot requires Q4K, 16 low/high activations, and 4 sums",
            ));
        }

        let mut emits = Vec::new();
        let base = self.quantized_block_base(expressions, matrix, block, col, 37, &mut emits);
        let d_word = self.load_word(expressions, matrix, base, 0, &mut emits)?;
        let d = self.bitcast_f32(expressions, &mut emits, d_word);
        let dmin_word = self.load_word(expressions, matrix, base, 1, &mut emits)?;
        let dmin = self.bitcast_f32(expressions, &mut emits, dmin_word);

        let scale_shift = self.shl_lit(expressions, &mut emits, iq, 4);
        let sc0 = self.load_word(expressions, matrix, base, 2, &mut emits)?;
        let sc1 = self.load_word(expressions, matrix, base, 3, &mut emits)?;
        let sc2 = self.load_word(expressions, matrix, base, 4, &mut emits)?;
        let sc0 = self.shr(expressions, &mut emits, sc0, scale_shift);
        let sc1 = self.shr(expressions, &mut emits, sc1, scale_shift);
        let sc2 = self.shr(expressions, &mut emits, sc2, scale_shift);

        let first_two = self.and_lit(expressions, &mut emits, sc0, 0x3f3f);
        let second_two = self.and_lit(expressions, &mut emits, sc1, 0x3f3f);
        let third_low = self.and_lit(expressions, &mut emits, sc2, 0x0f0f);
        let third_high = self.and_lit(expressions, &mut emits, sc0, 0xc0c0);
        let third_high = self.shr_lit(expressions, &mut emits, third_high, 2);
        let third_two = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::InclusiveOr,
            third_low,
            third_high,
        );
        let fourth_low = self.shr_lit(expressions, &mut emits, sc2, 4);
        let fourth_low = self.and_lit(expressions, &mut emits, fourth_low, 0x0f0f);
        let fourth_high = self.and_lit(expressions, &mut emits, sc1, 0xc0c0);
        let fourth_high = self.shr_lit(expressions, &mut emits, fourth_high, 2);
        let fourth_two = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::InclusiveOr,
            fourth_low,
            fourth_high,
        );

        let odd_scales = [
            self.u8_lane_f32(expressions, &mut emits, first_two, 0),
            self.u8_lane_f32(expressions, &mut emits, first_two, 1),
            self.u8_lane_f32(expressions, &mut emits, third_two, 0),
            self.u8_lane_f32(expressions, &mut emits, third_two, 1),
        ];
        let even_scales = [
            self.u8_lane_f32(expressions, &mut emits, second_two, 0),
            self.u8_lane_f32(expressions, &mut emits, second_two, 1),
            self.u8_lane_f32(expressions, &mut emits, fourth_two, 0),
            self.u8_lane_f32(expressions, &mut emits, fourth_two, 1),
        ];

        let iq_words = self.shl_lit(expressions, &mut emits, iq, 3);
        let ir_words = self.shl_lit(expressions, &mut emits, ir, 1);
        let data_offset = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            iq_words,
            ir_words,
        );

        let mut first_sums = [self.f32(expressions, 0.0); 4];
        let mut second_sums = [self.f32(expressions, 0.0); 4];
        let use_vector_accumulate = matrix.rows <= 4096 && matrix.cols >= 8192;
        for j in 0..2 {
            let word_off = self.add_lit(expressions, &mut emits, data_offset, 5 + j as u32);
            let word = self.load_word_dynamic(expressions, matrix, base, word_off, &mut emits)?;
            if use_vector_accumulate {
                self.q4k_ggml_accumulate_word_vector(
                    expressions,
                    &mut emits,
                    word,
                    a_low,
                    j,
                    &mut first_sums,
                );
            } else {
                self.q4k_ggml_accumulate_word_scalar(
                    expressions,
                    &mut emits,
                    word,
                    a_low,
                    j,
                    &mut first_sums,
                );
            }

            let word_off = self.add_lit(expressions, &mut emits, data_offset, 21 + j as u32);
            let word = self.load_word_dynamic(expressions, matrix, base, word_off, &mut emits)?;
            if use_vector_accumulate {
                self.q4k_ggml_accumulate_word_vector(
                    expressions,
                    &mut emits,
                    word,
                    a_high,
                    j,
                    &mut second_sums,
                );
            } else {
                self.q4k_ggml_accumulate_word_scalar(
                    expressions,
                    &mut emits,
                    word,
                    a_high,
                    j,
                    &mut second_sums,
                );
            }
        }

        let inv_256 = self.f32(expressions, 1.0 / 256.0);
        let inv_16 = self.f32(expressions, 1.0 / 16.0);
        let one = self.f32(expressions, 1.0);
        let small_shift_sums = self.compose_f32_vec4(
            expressions,
            &mut emits,
            [first_sums[0], first_sums[2], second_sums[0], second_sums[2]],
        );
        let large_shift_sums = self.compose_f32_vec4(
            expressions,
            &mut emits,
            [first_sums[1], first_sums[3], second_sums[1], second_sums[3]],
        );
        let inv_256_vec = self.compose_f32_vec4(
            expressions,
            &mut emits,
            [inv_256, inv_256, inv_256, inv_256],
        );
        let large_shift_sums = self.mul(expressions, &mut emits, large_shift_sums, inv_256_vec);
        let combined = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            small_shift_sums,
            large_shift_sums,
        );
        let odd_scales = self.compose_f32_vec4(expressions, &mut emits, odd_scales);
        let weighted = self.mul(expressions, &mut emits, combined, odd_scales);
        let shift4 = self.compose_f32_vec4(expressions, &mut emits, [one, inv_16, one, inv_16]);
        let scaled_dot = self.dot_f32_vec4(expressions, &mut emits, weighted, shift4);
        let scaled_dot = self.mul(expressions, &mut emits, d, scaled_dot);

        let sum_vec = self.compose_f32_vec4(
            expressions,
            &mut emits,
            [sums[0], sums[1], sums[2], sums[3]],
        );
        let even_scales = self.compose_f32_vec4(expressions, &mut emits, even_scales);
        let min_dot = self.dot_f32_vec4(expressions, &mut emits, sum_vec, even_scales);
        let min_dot = self.mul(expressions, &mut emits, dmin, min_dot);
        let total = self.sub(expressions, &mut emits, scaled_dot, min_dot);
        Ok((total, emits))
    }

    pub(in crate::lower) fn q6k_ggml_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        block: Handle<Expression>,
        ip: Handle<Expression>,
        il: Handle<Expression>,
        col: Handle<Expression>,
        a: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q6K || a.len() != 16 {
            return Err(LowerError::UnsupportedOperation(
                "q6k ggml dot requires Q6K and 16 activations",
            ));
        }

        let mut emits = Vec::new();
        let base = self.quantized_block_base(expressions, matrix, block, col, 53, &mut emits);
        let d_word = self.load_word(expressions, matrix, base, 52, &mut emits)?;
        let d = self.bitcast_f32(expressions, &mut emits, d_word);

        let l0 = self.shl_lit(expressions, &mut emits, il, 2);
        let low_base = self.shl_lit(expressions, &mut emits, ip, 6);
        let low_byte_offset = self.bin(expressions, &mut emits, BinaryOperator::Add, low_base, l0);
        let low_word_offset = self.shr_lit(expressions, &mut emits, low_byte_offset, 2);
        let q1_word =
            self.load_word_dynamic(expressions, matrix, base, low_word_offset, &mut emits)?;
        let q2_word_offset = self.add_lit(expressions, &mut emits, low_word_offset, 8);
        let q2_word =
            self.load_word_dynamic(expressions, matrix, base, q2_word_offset, &mut emits)?;

        let high_base = self.shl_lit(expressions, &mut emits, ip, 5);
        let high_byte_offset =
            self.bin(expressions, &mut emits, BinaryOperator::Add, high_base, l0);
        let high_word_offset = self.shr_lit(expressions, &mut emits, high_byte_offset, 2);
        let high_word_offset = self.add_lit(expressions, &mut emits, high_word_offset, 32);
        let qh_word =
            self.load_word_dynamic(expressions, matrix, base, high_word_offset, &mut emits)?;

        let scale_base = self.shl_lit(expressions, &mut emits, ip, 3);
        let scale_low = self.shr_lit(expressions, &mut emits, il, 2);
        let scale_index = self.bin(
            expressions,
            &mut emits,
            BinaryOperator::Add,
            scale_base,
            scale_low,
        );
        let scale_word0_offset = self.shr_lit(expressions, &mut emits, scale_index, 2);
        let scale_word0_offset = self.add_lit(expressions, &mut emits, scale_word0_offset, 48);
        let scale_word1_offset = self.add_lit(expressions, &mut emits, scale_word0_offset, 1);
        let scale_word0 =
            self.load_word_dynamic(expressions, matrix, base, scale_word0_offset, &mut emits)?;
        let scale_word1 =
            self.load_word_dynamic(expressions, matrix, base, scale_word1_offset, &mut emits)?;
        let scale_lane0 = self.and_lit(expressions, &mut emits, scale_index, 3);
        let scale_lane1 = self.add_lit(expressions, &mut emits, scale_lane0, 2);
        let scales = [
            self.byte_at(expressions, &mut emits, scale_word0, scale_lane0),
            self.byte_at(expressions, &mut emits, scale_word0, scale_lane1),
            self.byte_at(expressions, &mut emits, scale_word1, scale_lane0),
            self.byte_at(expressions, &mut emits, scale_word1, scale_lane1),
        ]
        .map(|byte| self.signed_byte_f32(expressions, &mut emits, byte));

        let mut sums = [self.f32(expressions, 0.0); 4];
        for l in 0..4 {
            let lane =
                expressions.append(Expression::Literal(Literal::U32(l as u32)), Span::default());
            let q1_byte = self.byte_at(expressions, &mut emits, q1_word, lane);
            let q2_byte = self.byte_at(expressions, &mut emits, q2_word, lane);
            let qh_byte = self.byte_at(expressions, &mut emits, qh_word, lane);

            let q0_low = self.and_lit(expressions, &mut emits, q1_byte, 0x0f);
            let q0_high = self.and_lit(expressions, &mut emits, qh_byte, 0x03);
            let q0_high = self.shl_lit(expressions, &mut emits, q0_high, 4);
            let q0 = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::InclusiveOr,
                q0_low,
                q0_high,
            );
            let q0 = self.center_q6k_quant(expressions, &mut emits, q0);

            let q1_low = self.and_lit(expressions, &mut emits, q2_byte, 0x0f);
            let q1_high = self.and_lit(expressions, &mut emits, qh_byte, 0x0c);
            let q1_high = self.shl_lit(expressions, &mut emits, q1_high, 2);
            let q1 = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::InclusiveOr,
                q1_low,
                q1_high,
            );
            let q1 = self.center_q6k_quant(expressions, &mut emits, q1);

            let q2_low = self.shr_lit(expressions, &mut emits, q1_byte, 4);
            let q2_high = self.and_lit(expressions, &mut emits, qh_byte, 0x30);
            let q2 = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::InclusiveOr,
                q2_low,
                q2_high,
            );
            let q2 = self.center_q6k_quant(expressions, &mut emits, q2);

            let q3_low = self.shr_lit(expressions, &mut emits, q2_byte, 4);
            let q3_high = self.and_lit(expressions, &mut emits, qh_byte, 0xc0);
            let q3_high = self.shr_lit(expressions, &mut emits, q3_high, 2);
            let q3 = self.bin(
                expressions,
                &mut emits,
                BinaryOperator::InclusiveOr,
                q3_low,
                q3_high,
            );
            let q3 = self.center_q6k_quant(expressions, &mut emits, q3);

            for (sum, activation, quant) in [
                (0, a[4 * l], q0),
                (1, a[4 * l + 1], q1),
                (2, a[4 * l + 2], q2),
                (3, a[4 * l + 3], q3),
            ] {
                let weighted = self.mul(expressions, &mut emits, activation, quant);
                sums[sum] = self.bin(
                    expressions,
                    &mut emits,
                    BinaryOperator::Add,
                    sums[sum],
                    weighted,
                );
            }
        }

        let sum_vec = self.compose_f32_vec4(expressions, &mut emits, sums);
        let scale_vec = self.compose_f32_vec4(expressions, &mut emits, scales);
        let weighted = self.dot_f32_vec4(expressions, &mut emits, sum_vec, scale_vec);
        let total = self.mul(expressions, &mut emits, d, weighted);
        Ok((total, emits))
    }

    pub(in crate::lower) fn q6k_q8_activation_dot(
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

    pub(in crate::lower) fn q6k_q8_activation_dot8(
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
        let total = self.q8_activation_packs_dot(
            expressions,
            &mut emits,
            a,
            pack_offset,
            b_scale,
            b_packs,
            None,
        );
        Ok((total, emits))
    }

    pub(in crate::lower) fn q8_activation_packs_dot(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        a: &Q8ActivationPacks,
        pack_offset: usize,
        b_scale: Handle<Expression>,
        b_packs: [Handle<Expression>; 2],
        b_min: Option<Handle<Expression>>,
    ) -> Handle<Expression> {
        let mut total = self.f32(expressions, 0.0);
        for (i, b_pack) in b_packs.into_iter().enumerate() {
            let a_pack_index = pack_offset + i;
            let a_pack = self.load_local(expressions, emits, a.packs[a_pack_index]);
            let dot = self.dot4_i8_packed(expressions, emits, a_pack, b_pack);
            let scaled = self.mul(expressions, emits, dot, b_scale);
            let unscaled = if let Some(b_min) = b_min {
                let a_sum_i32 = self.load_local(expressions, emits, a.sums_i32[a_pack_index]);
                let a_sum = self.as_f32(expressions, emits, a_sum_i32);
                let min_term = self.mul(expressions, emits, a_sum, b_min);
                self.sub(expressions, emits, scaled, min_term)
            } else {
                scaled
            };
            let a_scale = self.load_local(expressions, emits, a.scales[a_pack_index]);
            let chunk = self.mul(expressions, emits, unscaled, a_scale);
            total = self.bin(expressions, emits, BinaryOperator::Add, total, chunk);
        }
        total
    }

    pub(in crate::lower) fn cached_q8_activation_packs(
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

    pub(in crate::lower) fn q8_activation_pack_values(
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

    pub(in crate::lower) fn q8_activation_pack_locals(
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

    pub(in crate::lower) fn store_q8_activation_pack_values(
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

    pub(in crate::lower) fn q4k_quant_packs8(
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
        let parts = self.q4k_block_parts(expressions, matrix, k_base, col, emits)?;
        let (words, nibble_shift) =
            self.q4k_quant_words::<2>(expressions, matrix, &parts, false, emits)?;

        let packs = std::array::from_fn(|chunk| {
            let mut packed_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let source_lane = chunk * 4 + lane;
                let byte_lane = expressions.append(
                    Expression::Literal(Literal::U32((source_lane % 4) as u32)),
                    Span::default(),
                );
                let byte = self.byte_at(expressions, emits, words[source_lane / 4], byte_lane);
                let shifted = self.shr(expressions, emits, byte, nibble_shift);
                let quant = self.and_lit(expressions, emits, shifted, 0x0f);
                packed_values.push(self.as_i32(expressions, emits, quant));
            }
            self.pack_i8x4(expressions, emits, packed_values)
                .expect("q4k packs exactly four i8 values")
        });
        Ok((parts.scale, parts.min, packs))
    }

    pub(in crate::lower) fn q4k_quant_values<const N: usize, const WORDS: usize>(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        whole_group_pair: bool,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<
        (
            Handle<Expression>,
            Handle<Expression>,
            [Handle<Expression>; N],
        ),
        LowerError,
    > {
        debug_assert_eq!(WORDS * 4, N);
        let block = self.div_literal_u32_emitted(expressions, k_base, 256, emits);
        let q_base = self.and_lit(expressions, emits, k_base, 255);
        let parts =
            self.q4k_block_parts_from_block(expressions, matrix, block, q_base, col, emits)?;
        let (words, nibble_shift) =
            self.q4k_quant_words::<WORDS>(expressions, matrix, &parts, whole_group_pair, emits)?;

        let quants = std::array::from_fn(|source_lane| {
            let byte_lane = expressions.append(
                Expression::Literal(Literal::U32((source_lane % 4) as u32)),
                Span::default(),
            );
            let byte = self.byte_at(expressions, emits, words[source_lane / 4], byte_lane);
            let shifted = self.shr(expressions, emits, byte, nibble_shift);
            self.and_lit(expressions, emits, shifted, 0x0f)
        });

        Ok((parts.scale, parts.min, quants))
    }

    pub(in crate::lower) fn q4k_block_parts(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Q4KBlockParts, LowerError> {
        let block = self.div_literal_u32_emitted(expressions, k_base, 256, emits);
        let q_base = self.and_lit(expressions, emits, k_base, 255);
        self.q4k_block_parts_from_block(expressions, matrix, block, q_base, col, emits)
    }

    pub(in crate::lower) fn q4k_block_parts_from_block(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        block: Handle<Expression>,
        q_base: Handle<Expression>,
        col: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Q4KBlockParts, LowerError> {
        let base = self.quantized_block_base(expressions, matrix, block, col, 37, emits);
        let d_word = self.load_word(expressions, matrix, base, 0, emits)?;
        let d = self.bitcast_f32(expressions, emits, d_word);
        let dmin_word = self.load_word(expressions, matrix, base, 1, emits)?;
        let dmin = self.bitcast_f32(expressions, emits, dmin_word);
        let group = self.shr_lit(expressions, emits, q_base, 5);
        let (scale_byte, min_byte) =
            self.q4k_scale_min_bytes(expressions, matrix, base, group, emits)?;
        let scale_f = self.as_f32(expressions, emits, scale_byte);
        let scale = self.mul(expressions, emits, scale_f, d);
        let min_f = self.as_f32(expressions, emits, min_byte);
        let min = self.mul(expressions, emits, min_f, dmin);

        Ok(Q4KBlockParts {
            base,
            q_base,
            group,
            scale,
            min,
        })
    }

    pub(in crate::lower) fn q4k_quant_words<const WORDS: usize>(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        parts: &Q4KBlockParts,
        whole_group_pair: bool,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<([Handle<Expression>; WORDS], Handle<Expression>), LowerError> {
        let data_word = if whole_group_pair {
            let group_pair = self.shr_lit(expressions, emits, parts.group, 1);
            self.shl_lit(expressions, emits, group_pair, 3)
        } else {
            let in_group = self.and_lit(expressions, emits, parts.q_base, 31);
            let group_pair = self.shr_lit(expressions, emits, parts.group, 1);
            let group_pair_offset = self.shl_lit(expressions, emits, group_pair, 5);
            let byte_index = self.bin(
                expressions,
                emits,
                BinaryOperator::Add,
                group_pair_offset,
                in_group,
            );
            self.shr_lit(expressions, emits, byte_index, 2)
        };

        let mut offsets = Vec::with_capacity(WORDS);
        for word in 0..WORDS {
            offsets.push(self.add_lit(expressions, emits, data_word, 5 + word as u32));
        }

        let mut words = Vec::with_capacity(WORDS);
        for offset in offsets {
            words.push(self.load_word_dynamic(expressions, matrix, parts.base, offset, emits)?);
        }
        let words = words.try_into().ok().expect("q4k word count mismatch");
        let group_low = self.and_lit(expressions, emits, parts.group, 1);
        let nibble_shift = self.shl_lit(expressions, emits, group_low, 2);
        Ok((words, nibble_shift))
    }

    pub(in crate::lower) fn q6k_quant_packs8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<(Handle<Expression>, [Handle<Expression>; 2]), LowerError> {
        let parts = self.q6k_block_parts(expressions, matrix, k_base, col, emits)?;

        let packs = std::array::from_fn(|chunk| {
            let mut packed_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let source_lane = chunk * 4 + lane;
                let quant = self.q6k_quant_component(expressions, emits, &parts, source_lane);
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
        Ok((parts.scale, packs))
    }
}
