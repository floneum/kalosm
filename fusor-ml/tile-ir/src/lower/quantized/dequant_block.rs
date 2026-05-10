use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn dequant_affine(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        spec: AffineDequantSpec,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        match spec {
            AffineDequantSpec::Q8 { data_offset } => {
                self.dequant_q8_scaled(e, matrix, base, q, data_offset, emits)
            }
            AffineDequantSpec::Centered { nibble, center } => {
                let scale_word = self.load_word(e, matrix, base, 0, emits)?;
                let scale = self.bitcast_f32(e, emits, scale_word);
                let quant = self.affine_nibble(e, matrix, base, q, nibble, emits)?;
                let quant_f = self.as_f32(e, emits, quant);
                let center = self.f32(e, center);
                let centered = self.sub(e, emits, quant_f, center);
                Ok(self.mul(e, emits, centered, scale))
            }
            AffineDequantSpec::ScaleMin { nibble } => {
                let scale_word = self.load_word(e, matrix, base, 0, emits)?;
                let scale = self.bitcast_f32(e, emits, scale_word);
                let min_word = self.load_word(e, matrix, base, 1, emits)?;
                let min = self.bitcast_f32(e, emits, min_word);
                let quant = self.affine_nibble(e, matrix, base, q, nibble, emits)?;
                let quant_f = self.as_f32(e, emits, quant);
                let scaled = self.mul(e, emits, quant_f, scale);
                Ok(self.bin(e, emits, BinaryOperator::Add, scaled, min))
            }
        }
    }

    pub(in crate::lower) fn affine_nibble(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        nibble: AffineNibble,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        match nibble {
            AffineNibble::Q4 { data_offset } => {
                self.q4_block_nibble(e, matrix, base, q, data_offset, emits)
            }
            AffineNibble::Q5 {
                high_offset,
                data_offset,
            } => self.q5_block_nibble(e, matrix, base, q, high_offset, data_offset, emits),
        }
    }

    pub(in crate::lower) fn q4_block_nibble(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        data_offset: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, data_offset);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, q_local, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let low = self.and_lit(e, emits, byte, 0x0f);
        let high_q = self.shr_lit(e, emits, byte, 4);
        Ok(self.select(e, emits, high, high_q, low))
    }

    pub(in crate::lower) fn q5_block_nibble(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        high_offset: u32,
        data_offset: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let qh = self.load_word(e, matrix, base, high_offset, emits)?;
        let high = self.cmp_lit(e, emits, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, emits, q, 15);
        let q_word = self.shr_lit(e, emits, q_local, 2);
        let word_off = self.add_lit(e, emits, q_word, data_offset);
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
        Ok(self.bin(e, emits, BinaryOperator::InclusiveOr, low4, hi_bit))
    }

    pub(in crate::lower) fn dequant_q23k_quant_f(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        group: Handle<Expression>,
        data_base: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<
        (
            Handle<Expression>,
            Handle<Expression>,
            Handle<Expression>,
            Handle<Expression>,
            Handle<Expression>,
        ),
        LowerError,
    > {
        let q_local = self.and_lit(e, emits, q, 15);
        let chunk = self.shr_lit(e, emits, group, 3);
        let group_in_chunk = self.and_lit(e, emits, group, 7);
        let pair = self.and_lit(e, emits, group_in_chunk, 1);
        let byte_base = self.shl_lit(e, emits, chunk, 5);
        let pair_offset = self.shl_lit(e, emits, pair, 4);
        let byte_base = self.bin(e, emits, BinaryOperator::Add, byte_base, pair_offset);
        let byte_index = self.bin(e, emits, BinaryOperator::Add, byte_base, q_local);
        let word_off = self.shr_lit(e, emits, byte_index, 2);
        let word_off = self.add_lit(e, emits, word_off, data_base);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, byte_index, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let shift = self.shr_lit(e, emits, group_in_chunk, 1);
        let shift = self.shl_lit(e, emits, shift, 1);
        let shifted = self.shr(e, emits, byte, shift);
        let quant = self.and_lit(e, emits, shifted, 3);
        let quant_f = self.as_f32(e, emits, quant);
        Ok((group_in_chunk, chunk, pair_offset, q_local, quant_f))
    }

    pub(in crate::lower) fn dequant_q2k(
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
        let (_, _, _, _, quant_f) =
            self.dequant_q23k_quant_f(e, matrix, base, q, group, 4, emits)?;
        let scaled = self.mul(e, emits, quant_f, scale);
        Ok(self.sub(e, emits, scaled, min))
    }

    pub(in crate::lower) fn dequant_q3k(
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
        let (group_in_chunk, chunk, pair_offset, q_local, quant_f) =
            self.dequant_q23k_quant_f(e, matrix, base, q, group, 8, emits)?;
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

    pub(in crate::lower) fn dequant_q8k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        self.dequant_q8_scaled(e, matrix, base, q, 1, emits)
    }

    pub(in crate::lower) fn dequant_q8_scaled(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        data_offset: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(e, emits, scale_word);
        let q_word = self.shr_lit(e, emits, q, 2);
        let word_off = self.add_lit(e, emits, q_word, data_offset);
        let word = self.load_word_dynamic(e, matrix, base, word_off, emits)?;
        let byte_lane = self.and_lit(e, emits, q, 3);
        let byte = self.byte_at(e, emits, word, byte_lane);
        let signed = self.signed_byte_f32(e, emits, byte);
        Ok(self.mul(e, emits, signed, scale))
    }

    pub(in crate::lower) fn dequant_q4k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        self.dequant_k_nibble(e, matrix, base, q, emits, false)
    }

    pub(in crate::lower) fn dequant_q5k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        self.dequant_k_nibble(e, matrix, base, q, emits, true)
    }

    pub(in crate::lower) fn dequant_k_nibble(
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

    pub(in crate::lower) fn dequant_q6k(
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
}
