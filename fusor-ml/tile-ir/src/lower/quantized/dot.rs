use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn q4k_ggml_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        coords: GgmlBlockCoords,
        a_low: &[Handle<Expression>],
        a_high: &[Handle<Expression>],
        sums: &[Handle<Expression>],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if matrix.format != GgmlQuantFormat::Q4K
            || a_low.len() != 16
            || a_high.len() != 16
            || sums.len() != 4
        {
            return Err(LowerError::UnsupportedOperation(
                "q4k ggml dot requires Q4K, 16 low/high activations, and 4 sums",
            ));
        }
        let GgmlBlockCoords { block, c0: iq, c1: ir, col } = coords;

        let base = self.quantized_block_base(expressions, matrix, block, col, 37, body);
        let d_word = self.load_word(expressions, matrix, base, 0, body)?;
        let d = self.bitcast_f32(expressions, body, d_word);
        let dmin_word = self.load_word(expressions, matrix, base, 1, body)?;
        let dmin = self.bitcast_f32(expressions, body, dmin_word);

        let scale_shift = self.shl_lit(expressions, body, iq, 4);
        let sc0 = self.load_word(expressions, matrix, base, 2, body)?;
        let sc1 = self.load_word(expressions, matrix, base, 3, body)?;
        let sc2 = self.load_word(expressions, matrix, base, 4, body)?;
        let sc0 = self.shr(expressions, body, sc0, scale_shift);
        let sc1 = self.shr(expressions, body, sc1, scale_shift);
        let sc2 = self.shr(expressions, body, sc2, scale_shift);

        let first_two = self.and_lit(expressions, body, sc0, 0x3f3f);
        let second_two = self.and_lit(expressions, body, sc1, 0x3f3f);
        let third_low = self.and_lit(expressions, body, sc2, 0x0f0f);
        let third_high = self.and_lit(expressions, body, sc0, 0xc0c0);
        let third_high = self.shr_lit(expressions, body, third_high, 2);
        let third_two = self.bin(
            expressions,
            body,
            BinaryOperator::InclusiveOr,
            third_low,
            third_high,
        );
        let fourth_low = self.shr_lit(expressions, body, sc2, 4);
        let fourth_low = self.and_lit(expressions, body, fourth_low, 0x0f0f);
        let fourth_high = self.and_lit(expressions, body, sc1, 0xc0c0);
        let fourth_high = self.shr_lit(expressions, body, fourth_high, 2);
        let fourth_two = self.bin(
            expressions,
            body,
            BinaryOperator::InclusiveOr,
            fourth_low,
            fourth_high,
        );

        let odd_scales = [
            self.u8_lane_f32(expressions, body, first_two, 0),
            self.u8_lane_f32(expressions, body, first_two, 1),
            self.u8_lane_f32(expressions, body, third_two, 0),
            self.u8_lane_f32(expressions, body, third_two, 1),
        ];
        let even_scales = [
            self.u8_lane_f32(expressions, body, second_two, 0),
            self.u8_lane_f32(expressions, body, second_two, 1),
            self.u8_lane_f32(expressions, body, fourth_two, 0),
            self.u8_lane_f32(expressions, body, fourth_two, 1),
        ];

        let iq_words = self.shl_lit(expressions, body, iq, 3);
        let ir_words = self.shl_lit(expressions, body, ir, 1);
        let data_offset = self.bin(
            expressions,
            body,
            BinaryOperator::Add,
            iq_words,
            ir_words,
        );

        let mut first_sums = [self.f32(expressions, 0.0); 4];
        let mut second_sums = [self.f32(expressions, 0.0); 4];
        #[allow(clippy::type_complexity)]
        let accumulate: fn(
            &Self,
            &mut Arena<Expression>,
            &mut Block,
            Handle<Expression>,
            &[Handle<Expression>],
            usize,
            &mut [Handle<Expression>; 4],
        ) = if matrix.rows <= 4096 && matrix.cols >= 8192 {
            Self::q4k_ggml_accumulate_word_vector
        } else {
            Self::q4k_ggml_accumulate_word_scalar
        };
        for j in 0..2 {
            let word_off = self.add_lit(expressions, body, data_offset, 5 + j as u32);
            let word = self.load_word_dynamic(expressions, matrix, base, word_off, body)?;
            accumulate(self, expressions, body, word, a_low, j, &mut first_sums);

            let word_off = self.add_lit(expressions, body, data_offset, 21 + j as u32);
            let word = self.load_word_dynamic(expressions, matrix, base, word_off, body)?;
            accumulate(self, expressions, body, word, a_high, j, &mut second_sums);
        }

        let inv_256 = self.f32(expressions, 1.0 / 256.0);
        let inv_16 = self.f32(expressions, 1.0 / 16.0);
        let one = self.f32(expressions, 1.0);
        let small_shift_sums = self.compose_f32_vec4(
            expressions,
            body,
            [first_sums[0], first_sums[2], second_sums[0], second_sums[2]],
        );
        let large_shift_sums = self.compose_f32_vec4(
            expressions,
            body,
            [first_sums[1], first_sums[3], second_sums[1], second_sums[3]],
        );
        let inv_256_vec = self.compose_f32_vec4(
            expressions,
            body,
            [inv_256, inv_256, inv_256, inv_256],
        );
        let large_shift_sums = self.mul(expressions, body, large_shift_sums, inv_256_vec);
        let combined = self.bin(
            expressions,
            body,
            BinaryOperator::Add,
            small_shift_sums,
            large_shift_sums,
        );
        let odd_scales = self.compose_f32_vec4(expressions, body, odd_scales);
        let weighted = self.mul(expressions, body, combined, odd_scales);
        let shift4 = self.compose_f32_vec4(expressions, body, [one, inv_16, one, inv_16]);
        let scaled_dot = self.dot_f32_vec4(expressions, body, weighted, shift4);
        let scaled_dot = self.mul(expressions, body, d, scaled_dot);

        let sum_vec = self.compose_f32_vec4(
            expressions,
            body,
            [sums[0], sums[1], sums[2], sums[3]],
        );
        let even_scales = self.compose_f32_vec4(expressions, body, even_scales);
        let min_dot = self.dot_f32_vec4(expressions, body, sum_vec, even_scales);
        let min_dot = self.mul(expressions, body, dmin, min_dot);
        let total = self.sub(expressions, body, scaled_dot, min_dot);
        Ok(total)
    }

    pub(in crate::lower) fn q6k_ggml_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        coords: GgmlBlockCoords,
        a: &[Handle<Expression>],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if matrix.format != GgmlQuantFormat::Q6K || a.len() != 16 {
            return Err(LowerError::UnsupportedOperation(
                "q6k ggml dot requires Q6K and 16 activations",
            ));
        }
        let GgmlBlockCoords { block, c0: ip, c1: il, col } = coords;

        let base = self.quantized_block_base(expressions, matrix, block, col, 53, body);
        let d_word = self.load_word(expressions, matrix, base, 52, body)?;
        let d = self.bitcast_f32(expressions, body, d_word);

        let l0 = self.shl_lit(expressions, body, il, 2);
        let low_base = self.shl_lit(expressions, body, ip, 6);
        let low_byte_offset = self.bin(expressions, body, BinaryOperator::Add, low_base, l0);
        let low_word_offset = self.shr_lit(expressions, body, low_byte_offset, 2);
        let q1_word =
            self.load_word_dynamic(expressions, matrix, base, low_word_offset, body)?;
        let q2_word_offset = self.add_lit(expressions, body, low_word_offset, 8);
        let q2_word =
            self.load_word_dynamic(expressions, matrix, base, q2_word_offset, body)?;

        let high_base = self.shl_lit(expressions, body, ip, 5);
        let high_byte_offset =
            self.bin(expressions, body, BinaryOperator::Add, high_base, l0);
        let high_word_offset = self.shr_lit(expressions, body, high_byte_offset, 2);
        let high_word_offset = self.add_lit(expressions, body, high_word_offset, 32);
        let qh_word =
            self.load_word_dynamic(expressions, matrix, base, high_word_offset, body)?;

        let scale_base = self.shl_lit(expressions, body, ip, 3);
        let scale_low = self.shr_lit(expressions, body, il, 2);
        let scale_index = self.bin(
            expressions,
            body,
            BinaryOperator::Add,
            scale_base,
            scale_low,
        );
        let scale_word0_offset = self.shr_lit(expressions, body, scale_index, 2);
        let scale_word0_offset = self.add_lit(expressions, body, scale_word0_offset, 48);
        let scale_word1_offset = self.add_lit(expressions, body, scale_word0_offset, 1);
        let scale_word0 =
            self.load_word_dynamic(expressions, matrix, base, scale_word0_offset, body)?;
        let scale_word1 =
            self.load_word_dynamic(expressions, matrix, base, scale_word1_offset, body)?;
        let scale_lane0 = self.and_lit(expressions, body, scale_index, 3);
        let scale_lane1 = self.add_lit(expressions, body, scale_lane0, 2);
        let scales = [
            self.byte_at(expressions, body, scale_word0, scale_lane0),
            self.byte_at(expressions, body, scale_word0, scale_lane1),
            self.byte_at(expressions, body, scale_word1, scale_lane0),
            self.byte_at(expressions, body, scale_word1, scale_lane1),
        ]
        .map(|byte| self.signed_byte_f32(expressions, body, byte));

        let mut sums = [self.f32(expressions, 0.0); 4];
        for l in 0..4 {
            let lane = self.u32(expressions, l as u32);
            let q1_byte = self.byte_at(expressions, body, q1_word, lane);
            let q2_byte = self.byte_at(expressions, body, q2_word, lane);
            let qh_byte = self.byte_at(expressions, body, qh_word, lane);

            let q0_low = self.and_lit(expressions, body, q1_byte, 0x0f);
            let q0_high = self.and_lit(expressions, body, qh_byte, 0x03);
            let q0_high = self.shl_lit(expressions, body, q0_high, 4);
            let q0 = self.bin(
                expressions,
                body,
                BinaryOperator::InclusiveOr,
                q0_low,
                q0_high,
            );
            let q0 = self.center_q6k_quant(expressions, body, q0);

            let q1_low = self.and_lit(expressions, body, q2_byte, 0x0f);
            let q1_high = self.and_lit(expressions, body, qh_byte, 0x0c);
            let q1_high = self.shl_lit(expressions, body, q1_high, 2);
            let q1 = self.bin(
                expressions,
                body,
                BinaryOperator::InclusiveOr,
                q1_low,
                q1_high,
            );
            let q1 = self.center_q6k_quant(expressions, body, q1);

            let q2_low = self.shr_lit(expressions, body, q1_byte, 4);
            let q2_high = self.and_lit(expressions, body, qh_byte, 0x30);
            let q2 = self.bin(
                expressions,
                body,
                BinaryOperator::InclusiveOr,
                q2_low,
                q2_high,
            );
            let q2 = self.center_q6k_quant(expressions, body, q2);

            let q3_low = self.shr_lit(expressions, body, q2_byte, 4);
            let q3_high = self.and_lit(expressions, body, qh_byte, 0xc0);
            let q3_high = self.shr_lit(expressions, body, q3_high, 2);
            let q3 = self.bin(
                expressions,
                body,
                BinaryOperator::InclusiveOr,
                q3_low,
                q3_high,
            );
            let q3 = self.center_q6k_quant(expressions, body, q3);

            for (sum, activation, quant) in [
                (0, a[4 * l], q0),
                (1, a[4 * l + 1], q1),
                (2, a[4 * l + 2], q2),
                (3, a[4 * l + 3], q3),
            ] {
                let weighted = self.mul(expressions, body, activation, quant);
                sums[sum] = self.bin(
                    expressions,
                    body,
                    BinaryOperator::Add,
                    sums[sum],
                    weighted,
                );
            }
        }

        let sum_vec = self.compose_f32_vec4(expressions, body, sums);
        let scale_vec = self.compose_f32_vec4(expressions, body, scales);
        let weighted = self.dot_f32_vec4(expressions, body, sum_vec, scale_vec);
        let total = self.mul(expressions, body, d, weighted);
        Ok(total)
    }

    pub(in crate::lower) fn q6k_q8_activation_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &Q8ActivationPacks,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if matrix.format != GgmlQuantFormat::Q6K {
            return Err(LowerError::UnsupportedOperation(
                "q6k x q8 activation dot only supports Q6K",
            ));
        }

        self.q8_activation_pack_pair_dot(expressions, body, k_base, a, |s, e, b, k, off| {
            s.q6k_q8_activation_dot8(e, matrix, k, col, a, off, b)
        })
    }

    pub(in crate::lower) fn q6k_q8_activation_dot8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &Q8ActivationPacks,
        pack_offset: usize,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if a.len < pack_offset + 2 {
            return Err(LowerError::UnsupportedOperation(
                "q6k x q8 activation dot8 requires two activation packs",
            ));
        }

        let (b_scale, b_packs) =
            self.q6k_quant_packs8(expressions, matrix, k_base, col, body)?;
        let total = self.q8_activation_packs_dot(
            expressions,
            body,
            a,
            pack_offset,
            b_scale,
            b_packs,
            None,
        );
        Ok(total)
    }

    pub(in crate::lower) fn q8_activation_packs_dot(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        a: &Q8ActivationPacks,
        pack_offset: usize,
        b_scale: Handle<Expression>,
        b_packs: [Handle<Expression>; 2],
        b_min: Option<Handle<Expression>>,
    ) -> Handle<Expression> {
        let mut total = self.f32(expressions, 0.0);
        for (i, b_pack) in b_packs.into_iter().enumerate() {
            let a_pack_index = pack_offset + i;
            let a_pack = self.load_local(expressions, body, a.packs[a_pack_index]);
            let dot = self.dot4_i8_packed(expressions, body, a_pack, b_pack);
            let scaled = self.mul(expressions, body, dot, b_scale);
            let unscaled = if let Some(b_min) = b_min {
                let a_sum_i32 = self.load_local(expressions, body, a.sums_i32[a_pack_index]);
                let a_sum = self.as_f32(expressions, body, a_sum_i32);
                let min_term = self.mul(expressions, body, a_sum, b_min);
                self.sub(expressions, body, scaled, min_term)
            } else {
                scaled
            };
            let a_scale = self.load_local(expressions, body, a.scales[a_pack_index]);
            let chunk = self.mul(expressions, body, unscaled, a_scale);
            total = self.bin(expressions, body, BinaryOperator::Add, total, chunk);
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

        let values = self.q8_activation_pack_values(e, a, body)?;
        let packs = Self::q8_activation_pack_locals(scratch, values.packs.len())?;
        self.q8_activation_pack_cache.borrow_mut().clear();
        self.store_q8_activation_pack_values(e, body, &packs, values);
        self.q8_activation_pack_cache
            .borrow_mut()
            .insert(key, packs.clone());
        Ok(packs)
    }

    pub(in crate::lower) fn q8_activation_pack_values(
        &self,
        e: &mut Arena<Expression>,
        a: &[Handle<Expression>],
        body: &mut Block,
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
                let abs = self.math1(e, body, MathFunction::Abs, *value);
                max_abs = self.math2(e, body, MathFunction::Max, max_abs, abs);
            }
            let epsilon = self.f32(e, 1.0e-8);
            max_abs = self.math2(e, body, MathFunction::Max, max_abs, epsilon);
            let inv_scale = self.div(e, body, qmax, max_abs);
            let scale = self.div(e, body, max_abs, qmax);
            let mut sum_i32 = self.i32(e, 0);
            let mut packed_values = Vec::with_capacity(4);
            for (lane, value) in chunk.iter().enumerate() {
                let scaled = self.mul(e, body, *value, inv_scale);
                let rounded = self.math1(e, body, MathFunction::Round, scaled);
                let lo = self.f32(e, -127.0);
                let hi = self.f32(e, 127.0);
                let clamped = self.math2(e, body, MathFunction::Min, rounded, hi);
                let clamped = self.math2(e, body, MathFunction::Max, clamped, lo);
                let q_i32 = self.as_i32(e, body, clamped);
                sum_i32 = self.bin(e, body, BinaryOperator::Add, sum_i32, q_i32);
                debug_assert!(lane < 4);
                packed_values.push(q_i32);
            }
            scales.push(scale);
            sums_i32.push(sum_i32);
            packs.push(self.pack_i8x4(e, body, packed_values)?);
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
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        locals: &Q8ActivationPacks,
        values: Q8ActivationPackValues,
    ) {
        debug_assert_eq!(locals.len, values.scales.len());
        debug_assert_eq!(locals.len, values.packs.len());
        debug_assert_eq!(locals.len, values.sums_i32.len());

        for i in 0..locals.len {
            self.store_local(e, body, locals.scales[i], values.scales[i]);
            self.store_local(e, body, locals.packs[i], values.packs[i]);
            self.store_local(e, body, locals.sums_i32[i], values.sums_i32[i]);
        }
    }

    pub(in crate::lower) fn q4k_quant_packs8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Q4KQuantBlock<2>, LowerError> {
        let parts = self.q4k_block_parts(expressions, matrix, k_base, col, body)?;
        let (words, nibble_shift) =
            self.q4k_quant_words::<2>(expressions, matrix, &parts, false, body)?;

        let data = std::array::from_fn(|chunk| {
            let mut packed_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let source_lane = chunk * 4 + lane;
                let byte_lane = self.u32(expressions, (source_lane % 4) as u32);
                let byte = self.byte_at(expressions, body, words[source_lane / 4], byte_lane);
                let shifted = self.shr(expressions, body, byte, nibble_shift);
                let quant = self.and_lit(expressions, body, shifted, 0x0f);
                packed_values.push(self.as_i32(expressions, body, quant));
            }
            self.pack_i8x4(expressions, body, packed_values)
                .expect("q4k packs exactly four i8 values")
        });
        Ok(Q4KQuantBlock { scale: parts.scale, min: parts.min, data })
    }

    pub(in crate::lower) fn q4k_quant_values<const N: usize, const WORDS: usize>(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        whole_group_pair: bool,
        body: &mut Block,
    ) -> Result<Q4KQuantBlock<N>, LowerError> {
        debug_assert_eq!(WORDS * 4, N);
        let block = self.div_literal_u32_emitted(expressions, k_base, 256, body);
        let q_base = self.and_lit(expressions, body, k_base, 255);
        let parts =
            self.q4k_block_parts_from_block(expressions, matrix, block, q_base, col, body)?;
        let (words, nibble_shift) =
            self.q4k_quant_words::<WORDS>(expressions, matrix, &parts, whole_group_pair, body)?;

        let data = std::array::from_fn(|source_lane| {
            let byte_lane = self.u32(expressions, (source_lane % 4) as u32);
            let byte = self.byte_at(expressions, body, words[source_lane / 4], byte_lane);
            let shifted = self.shr(expressions, body, byte, nibble_shift);
            self.and_lit(expressions, body, shifted, 0x0f)
        });

        Ok(Q4KQuantBlock { scale: parts.scale, min: parts.min, data })
    }

    pub(in crate::lower) fn q4k_block_parts(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Q4KBlockParts, LowerError> {
        let block = self.div_literal_u32_emitted(expressions, k_base, 256, body);
        let q_base = self.and_lit(expressions, body, k_base, 255);
        self.q4k_block_parts_from_block(expressions, matrix, block, q_base, col, body)
    }

    pub(in crate::lower) fn q4k_block_parts_from_block(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        block: Handle<Expression>,
        q_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Q4KBlockParts, LowerError> {
        let base = self.quantized_block_base(expressions, matrix, block, col, 37, body);
        let d_word = self.load_word(expressions, matrix, base, 0, body)?;
        let d = self.bitcast_f32(expressions, body, d_word);
        let dmin_word = self.load_word(expressions, matrix, base, 1, body)?;
        let dmin = self.bitcast_f32(expressions, body, dmin_word);
        let group = self.shr_lit(expressions, body, q_base, 5);
        let (scale_byte, min_byte) =
            self.q4k_scale_min_bytes(expressions, matrix, base, group, body)?;
        let scale_f = self.as_f32(expressions, body, scale_byte);
        let scale = self.mul(expressions, body, scale_f, d);
        let min_f = self.as_f32(expressions, body, min_byte);
        let min = self.mul(expressions, body, min_f, dmin);

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
        body: &mut Block,
    ) -> Result<([Handle<Expression>; WORDS], Handle<Expression>), LowerError> {
        let data_word = if whole_group_pair {
            let group_pair = self.shr_lit(expressions, body, parts.group, 1);
            self.shl_lit(expressions, body, group_pair, 3)
        } else {
            let in_group = self.and_lit(expressions, body, parts.q_base, 31);
            let group_pair = self.shr_lit(expressions, body, parts.group, 1);
            let group_pair_offset = self.shl_lit(expressions, body, group_pair, 5);
            let byte_index = self.bin(
                expressions,
                body,
                BinaryOperator::Add,
                group_pair_offset,
                in_group,
            );
            self.shr_lit(expressions, body, byte_index, 2)
        };

        let mut offsets = Vec::with_capacity(WORDS);
        for word in 0..WORDS {
            offsets.push(self.add_lit(expressions, body, data_word, 5 + word as u32));
        }

        let mut words = Vec::with_capacity(WORDS);
        for offset in offsets {
            words.push(self.load_word_dynamic(expressions, matrix, parts.base, offset, body)?);
        }
        let words = words.try_into().expect("q4k word count mismatch");
        let group_low = self.and_lit(expressions, body, parts.group, 1);
        let nibble_shift = self.shl_lit(expressions, body, group_low, 2);
        Ok((words, nibble_shift))
    }

    pub(in crate::lower) fn q6k_quant_packs8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        body: &mut Block,
    ) -> Result<(Handle<Expression>, [Handle<Expression>; 2]), LowerError> {
        let parts = self.q6k_block_parts(expressions, matrix, k_base, col, body)?;

        let packs = std::array::from_fn(|chunk| {
            let mut packed_values = Vec::with_capacity(4);
            for lane in 0..4 {
                let source_lane = chunk * 4 + lane;
                let quant = self.q6k_quant_component(expressions, body, &parts, source_lane);
                let quant_i32 = self.as_i32(expressions, body, quant);
                let center = self.i32(expressions, 32);
                let centered = self.bin(
                    expressions,
                    body,
                    BinaryOperator::Subtract,
                    quant_i32,
                    center,
                );
                packed_values.push(centered);
            }
            self.pack_i8x4(expressions, body, packed_values)
                .expect("q6k packs exactly four i8 values")
        });
        Ok((parts.scale, packs))
    }
}
