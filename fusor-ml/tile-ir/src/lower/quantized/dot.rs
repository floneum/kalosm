use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn q6k_q8_activation_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &Q8ActivationPacks,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if !matrix.format.is_q6k_family() {
            return Err(LowerError::UnsupportedOperation(
                "q6k x q8 activation dot only supports Q6K formats",
            ));
        }

        self.q8_activation_pack_pair_dot(expressions, body, k_base, a, |s, e, b, k, off| {
            s.q6k_q8_activation_dot8(e, matrix, QuantDotCoords { k_base: k, col }, a, off, b)
        })
    }

    pub(in crate::lower) fn q6k_q8_activation_dot8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        coords: QuantDotCoords,
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
            self.q6k_quant_packs8(expressions, matrix, coords.k_base, coords.col, body)?;
        let total = self.q8_activation_packs_dot(
            expressions,
            body,
            a,
            pack_offset,
            Q8ActivationDotRhs {
                scale: b_scale,
                packs: b_packs,
                min: None,
            },
        );
        Ok(total)
    }

    pub(in crate::lower) fn q8_activation_packs_dot(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        a: &Q8ActivationPacks,
        pack_offset: usize,
        rhs: Q8ActivationDotRhs,
    ) -> Handle<Expression> {
        let mut total = self.f32(expressions, 0.0);
        for (i, b_pack) in rhs.packs.into_iter().enumerate() {
            let a_pack_index = pack_offset + i;
            let a_pack = self.load_local(expressions, body, a.packs[a_pack_index]);
            let dot = self.dot4_i8_packed(expressions, body, a_pack, b_pack);
            let scaled = self.mul(expressions, body, dot, rhs.scale);
            let unscaled = if let Some(b_min) = rhs.min {
                let a_sum_i32 = self.load_local(expressions, body, a.sums_i32[a_pack_index]);
                let a_sum = self.as_f32(expressions, body, a_sum_i32);
                let min_term = self.mul(expressions, body, a_sum, b_min);
                self.sub(expressions, body, scaled, min_term)
            } else {
                scaled
            };
            let a_scale = self.load_local(expressions, body, a.scales[a_pack_index]);
            let chunk = self.mul(expressions, body, unscaled, a_scale);
            total = self.add(expressions, body, total, chunk);
        }
        total
    }

    pub(in crate::lower) fn cached_q8_activation_packs(
        &self,
        e: &mut Arena<Expression>,
        body: &mut Block,
        a: &[Handle<Expression>],
    ) -> Result<Q8ActivationPacks, LowerError> {
        let key = a.to_vec();
        if let Some(packs) = self.q8_activation_pack_cache.borrow().get(&key).cloned() {
            return Ok(packs);
        }

        let values = self.q8_activation_pack_values(e, a, body)?;
        let packs = self.q8_activation_pack_locals(values.packs.len())?;
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
                sum_i32 = self.add(e, body, sum_i32, q_i32);
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
        &self,
        len: usize,
    ) -> Result<Q8ActivationPacks, LowerError> {
        if len > 4 {
            return Err(LowerError::UnsupportedOperation(
                "q8 activation packing supports at most four packs",
            ));
        }
        let scales = std::array::from_fn(|i| self.scratch_f32(ScratchKind::Q8Scale, i as u32));
        let packs = std::array::from_fn(|i| self.scratch_u32(ScratchKind::Q8Pack, i as u32));
        let sums_i32 = std::array::from_fn(|i| self.scratch_i32(ScratchKind::Q8Sum, i as u32));
        Ok(Q8ActivationPacks {
            len,
            scales,
            packs,
            sums_i32,
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

        Ok(Q4KQuantBlock {
            scale: parts.scale,
            min: parts.min,
            data,
        })
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
        let base = self.quantized_block_base(
            expressions,
            matrix,
            block,
            col,
            matrix.format.block_words(),
            body,
        );
        let (d, dmin) = self.load_k_d_dmin(expressions, matrix, base, body)?;
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

        let data_base = self.q4k_data_word_offset(matrix.format)?;
        let mut offsets = Vec::with_capacity(WORDS);
        for word in 0..WORDS {
            offsets.push(self.add_lit(expressions, body, data_word, data_base + word as u32));
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
