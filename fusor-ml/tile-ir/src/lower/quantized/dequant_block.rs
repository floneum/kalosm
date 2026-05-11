use super::*;

/// Intermediate values produced by `dequant_q23k_quant_f`. The 2-bit dequant
/// shares enough address arithmetic between Q2K and Q3K to be worth lifting,
/// but Q3K reuses the offsets to compute its high-bit mask while Q2K only
/// keeps the dequantized value.
pub(in crate::lower) struct Q23KQuantParts {
    pub group_in_chunk: Handle<Expression>,
    pub chunk: Handle<Expression>,
    pub pair_offset: Handle<Expression>,
    pub q_local: Handle<Expression>,
    pub quant_f: Handle<Expression>,
}

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn dequant_affine(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        spec: AffineDequantSpec,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        match spec {
            AffineDequantSpec::Q8 { data_offset } => {
                self.dequant_q8_scaled(e, matrix, base, q, data_offset, body)
            }
            AffineDequantSpec::Centered { nibble, center } => {
                let scale_word = self.load_word(e, matrix, base, 0, body)?;
                let scale = self.bitcast_f32(e, body, scale_word);
                let quant = self.affine_nibble(e, matrix, base, q, nibble, body)?;
                let quant_f = self.as_f32(e, body, quant);
                let center = self.f32(e, center);
                let centered = self.sub(e, body, quant_f, center);
                Ok(self.mul(e, body, centered, scale))
            }
            AffineDequantSpec::ScaleMin { nibble } => {
                let scale_word = self.load_word(e, matrix, base, 0, body)?;
                let scale = self.bitcast_f32(e, body, scale_word);
                let min_word = self.load_word(e, matrix, base, 1, body)?;
                let min = self.bitcast_f32(e, body, min_word);
                let quant = self.affine_nibble(e, matrix, base, q, nibble, body)?;
                let quant_f = self.as_f32(e, body, quant);
                let scaled = self.mul(e, body, quant_f, scale);
                Ok(self.bin(e, body, BinaryOperator::Add, scaled, min))
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
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        match nibble {
            AffineNibble::Q4 { data_offset } => {
                self.q4_block_nibble(e, matrix, base, q, data_offset, body)
            }
            AffineNibble::Q5 {
                high_offset,
                data_offset,
            } => self.q5_block_nibble(e, matrix, base, q, high_offset, data_offset, body),
        }
    }

    pub(in crate::lower) fn q4_block_nibble(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        data_offset: u32,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let high = self.cmp_lit(e, body, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, body, q, 15);
        let q_word = self.shr_lit(e, body, q_local, 2);
        let word_off = self.add_lit(e, body, q_word, data_offset);
        let word = self.load_word_dynamic(e, matrix, base, word_off, body)?;
        let byte_lane = self.and_lit(e, body, q_local, 3);
        let byte = self.byte_at(e, body, word, byte_lane);
        let low = self.and_lit(e, body, byte, 0x0f);
        let high_q = self.shr_lit(e, body, byte, 4);
        Ok(self.select(e, body, high, high_q, low))
    }

    pub(in crate::lower) fn q5_block_nibble(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        high_offset: u32,
        data_offset: u32,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let qh = self.load_word(e, matrix, base, high_offset, body)?;
        let high = self.cmp_lit(e, body, BinaryOperator::GreaterEqual, q, 16);
        let q_local = self.and_lit(e, body, q, 15);
        let q_word = self.shr_lit(e, body, q_local, 2);
        let word_off = self.add_lit(e, body, q_word, data_offset);
        let word = self.load_word_dynamic(e, matrix, base, word_off, body)?;
        let byte_lane = self.and_lit(e, body, q_local, 3);
        let byte = self.byte_at(e, body, word, byte_lane);
        let low = self.and_lit(e, body, byte, 0x0f);
        let high4 = self.shr_lit(e, body, byte, 4);
        let low4 = self.select(e, body, high, high4, low);
        let high_index = self.add_lit(e, body, q_local, 16);
        let hi_bit_index = self.select(e, body, high, high_index, q_local);
        let shifted_qh = self.shr(e, body, qh, hi_bit_index);
        let hi_bit_low = self.and_lit(e, body, shifted_qh, 1);
        let hi_bit = self.shl_lit(e, body, hi_bit_low, 4);
        Ok(self.bin(e, body, BinaryOperator::InclusiveOr, low4, hi_bit))
    }

    pub(in crate::lower) fn dequant_q23k_quant_f(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        group: Handle<Expression>,
        data_base: u32,
        body: &mut Block,
    ) -> Result<Q23KQuantParts, LowerError> {
        let q_local = self.and_lit(e, body, q, 15);
        let chunk = self.shr_lit(e, body, group, 3);
        let group_in_chunk = self.and_lit(e, body, group, 7);
        let pair = self.and_lit(e, body, group_in_chunk, 1);
        let byte_base = self.shl_lit(e, body, chunk, 5);
        let pair_offset = self.shl_lit(e, body, pair, 4);
        let byte_base = self.bin(e, body, BinaryOperator::Add, byte_base, pair_offset);
        let byte_index = self.bin(e, body, BinaryOperator::Add, byte_base, q_local);
        let word_off = self.shr_lit(e, body, byte_index, 2);
        let word_off = self.add_lit(e, body, word_off, data_base);
        let word = self.load_word_dynamic(e, matrix, base, word_off, body)?;
        let byte_lane = self.and_lit(e, body, byte_index, 3);
        let byte = self.byte_at(e, body, word, byte_lane);
        let shift = self.shr_lit(e, body, group_in_chunk, 1);
        let shift = self.shl_lit(e, body, shift, 1);
        let shifted = self.shr(e, body, byte, shift);
        let quant = self.and_lit(e, body, shifted, 3);
        let quant_f = self.as_f32(e, body, quant);
        Ok(Q23KQuantParts { group_in_chunk, chunk, pair_offset, q_local, quant_f })
    }

    pub(in crate::lower) fn dequant_q2k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 20, body)?;
        let d = self.bitcast_f32(e, body, d_word);
        let dmin_word = self.load_word(e, matrix, base, 21, body)?;
        let dmin = self.bitcast_f32(e, body, dmin_word);
        let group = self.shr_lit(e, body, q, 4);
        let scale_word_off = self.shr_lit(e, body, group, 2);
        let scale_word = self.load_word_dynamic(e, matrix, base, scale_word_off, body)?;
        let scale_lane = self.and_lit(e, body, group, 3);
        let scale_byte = self.byte_at(e, body, scale_word, scale_lane);
        let scale_quant = self.and_lit(e, body, scale_byte, 0x0f);
        let scale_quant_f = self.as_f32(e, body, scale_quant);
        let scale = self.mul(e, body, scale_quant_f, d);
        let min_quant = self.shr_lit(e, body, scale_byte, 4);
        let min_quant_f = self.as_f32(e, body, min_quant);
        let min = self.mul(e, body, min_quant_f, dmin);
        let parts = self.dequant_q23k_quant_f(e, matrix, base, q, group, 4, body)?;
        let scaled = self.mul(e, body, parts.quant_f, scale);
        Ok(self.sub(e, body, scaled, min))
    }

    pub(in crate::lower) fn dequant_q3k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 27, body)?;
        let d = self.bitcast_f32(e, body, d_word);
        let group = self.shr_lit(e, body, q, 4);
        let scale_quant = self.q3k_scale(e, matrix, base, group, body)?;
        let scale_quant_f = self.as_f32(e, body, scale_quant);
        let center = self.f32(e, 32.0);
        let scale_quant_f = self.sub(e, body, scale_quant_f, center);
        let scale = self.mul(e, body, scale_quant_f, d);
        let parts = self.dequant_q23k_quant_f(e, matrix, base, q, group, 8, body)?;
        let hmask_index = self.bin(e, body, BinaryOperator::Add, parts.pair_offset, parts.q_local);
        let hmask_word_off = self.shr_lit(e, body, hmask_index, 2);
        let hword = self.load_word_dynamic(e, matrix, base, hmask_word_off, body)?;
        let hmask_lane = self.and_lit(e, body, hmask_index, 3);
        let hbyte = self.byte_at(e, body, hword, hmask_lane);
        let hmask_bit_pair = self.shr_lit(e, body, parts.group_in_chunk, 1);
        let chunk_mask_base = self.shl_lit(e, body, parts.chunk, 2);
        let hmask_bit = self.bin(
            e,
            body,
            BinaryOperator::Add,
            chunk_mask_base,
            hmask_bit_pair,
        );
        let one = self.u32(e, 1);
        let hmask = self.bin(e, body, BinaryOperator::ShiftLeft, one, hmask_bit);
        let high = self.bin(e, body, BinaryOperator::And, hbyte, hmask);
        let high_set = self.cmp_lit(e, body, BinaryOperator::NotEqual, high, 0);
        let zero = self.f32(e, 0.0);
        let four = self.f32(e, 4.0);
        let penalty = self.select(e, body, high_set, zero, four);
        let centered = self.sub(e, body, parts.quant_f, penalty);
        Ok(self.mul(e, body, centered, scale))
    }

    pub(in crate::lower) fn dequant_q8k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        self.dequant_q8_scaled(e, matrix, base, q, 1, body)
    }

    pub(in crate::lower) fn dequant_q8_scaled(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        data_offset: u32,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let scale_word = self.load_word(e, matrix, base, 0, body)?;
        let scale = self.bitcast_f32(e, body, scale_word);
        let q_word = self.shr_lit(e, body, q, 2);
        let word_off = self.add_lit(e, body, q_word, data_offset);
        let word = self.load_word_dynamic(e, matrix, base, word_off, body)?;
        let byte_lane = self.and_lit(e, body, q, 3);
        let byte = self.byte_at(e, body, word, byte_lane);
        let signed = self.signed_byte_f32(e, body, byte);
        Ok(self.mul(e, body, signed, scale))
    }

    pub(in crate::lower) fn dequant_q4k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        self.dequant_k_nibble(e, matrix, base, q, body, false)
    }

    pub(in crate::lower) fn dequant_q5k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        self.dequant_k_nibble(e, matrix, base, q, body, true)
    }

    pub(in crate::lower) fn dequant_k_nibble(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        body: &mut Block,
        q5: bool,
    ) -> Result<Handle<Expression>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 0, body)?;
        let d = self.bitcast_f32(e, body, d_word);
        let dmin_word = self.load_word(e, matrix, base, 1, body)?;
        let dmin = self.bitcast_f32(e, body, dmin_word);
        let group = self.shr_lit(e, body, q, 5);
        let scale_byte = self.k_scale(e, matrix, base, group, false, body)?;
        let scale_f = self.as_f32(e, body, scale_byte);
        let scale = self.mul(e, body, scale_f, d);
        let min_byte = self.k_scale(e, matrix, base, group, true, body)?;
        let min_f = self.as_f32(e, body, min_byte);
        let min = self.mul(e, body, min_f, dmin);
        let in_group = self.and_lit(e, body, q, 31);
        let group_pair = self.shr_lit(e, body, group, 1);
        let group_pair_offset = self.shl_lit(e, body, group_pair, 5);
        let byte_index = self.bin(e, body, BinaryOperator::Add, group_pair_offset, in_group);
        let data_base = if q5 { 13 } else { 5 };
        let data_word = self.shr_lit(e, body, byte_index, 2);
        let data_off = self.add_lit(e, body, data_word, data_base);
        let word = self.load_word_dynamic(e, matrix, base, data_off, body)?;
        let byte_lane = self.and_lit(e, body, byte_index, 3);
        let byte = self.byte_at(e, body, word, byte_lane);
        let group_low = self.and_lit(e, body, group, 1);
        let high = self.cmp_lit(e, body, BinaryOperator::NotEqual, group_low, 0);
        let byte_hi = self.shr_lit(e, body, byte, 4);
        let byte_lo = self.and_lit(e, body, byte, 0x0f);
        let mut quant = self.select(e, body, high, byte_hi, byte_lo);
        if q5 {
            let qh_byte_index = self.and_lit(e, body, q, 31);
            let qh_word = self.shr_lit(e, body, qh_byte_index, 2);
            let qh_off = self.add_lit(e, body, qh_word, 5);
            let qh = self.load_word_dynamic(e, matrix, base, qh_off, body)?;
            let qh_lane = self.and_lit(e, body, qh_byte_index, 3);
            let qh_byte = self.byte_at(e, body, qh, qh_lane);
            let qh_bit_index = self.shr_lit(e, body, q, 5);
            let shifted_qh = self.shr(e, body, qh_byte, qh_bit_index);
            let bit = self.and_lit(e, body, shifted_qh, 1);
            let bit = self.shl_lit(e, body, bit, 4);
            quant = self.bin(e, body, BinaryOperator::InclusiveOr, quant, bit);
        }
        let quant_f = self.as_f32(e, body, quant);
        let scaled = self.mul(e, body, quant_f, scale);
        Ok(self.sub(e, body, scaled, min))
    }

    pub(in crate::lower) fn dequant_q6k(
        &self,
        e: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        base: Handle<Expression>,
        q: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let d_word = self.load_word(e, matrix, base, 52, body)?;
        let d = self.bitcast_f32(e, body, d_word);
        let chunk = self.shr_lit(e, body, q, 7);
        let local = self.and_lit(e, body, q, 127);
        let high_byte_index = self.and_lit(e, body, local, 31);
        let low_group = self.shr_lit(e, body, local, 5);
        let chunk_low_base = self.shl_lit(e, body, chunk, 6);
        let low_group_parity = self.and_lit(e, body, low_group, 1);
        let low_group_offset = self.shl_lit(e, body, low_group_parity, 5);
        let local_low_index = self.bin(
            e,
            body,
            BinaryOperator::Add,
            high_byte_index,
            low_group_offset,
        );
        let lower_index = self.bin(
            e,
            body,
            BinaryOperator::Add,
            chunk_low_base,
            local_low_index,
        );
        let low_word_off = self.shr_lit(e, body, lower_index, 2);
        let low_word = self.load_word_dynamic(e, matrix, base, low_word_off, body)?;
        let low_lane = self.and_lit(e, body, lower_index, 3);
        let low_byte = self.byte_at(e, body, low_word, low_lane);
        let low_nibble_shift = self.shr_lit(e, body, low_group, 1);
        let low_nibble_shift = self.shl_lit(e, body, low_nibble_shift, 2);
        let low_shifted = self.shr(e, body, low_byte, low_nibble_shift);
        let low4 = self.and_lit(e, body, low_shifted, 0x0f);
        let high_chunk_base = self.shl_lit(e, body, chunk, 5);
        let high_index = self.bin(
            e,
            body,
            BinaryOperator::Add,
            high_chunk_base,
            high_byte_index,
        );
        let high_word_base = self.shr_lit(e, body, high_index, 2);
        let high_word_off = self.add_lit(e, body, high_word_base, 32);
        let high_word = self.load_word_dynamic(e, matrix, base, high_word_off, body)?;
        let high_lane = self.and_lit(e, body, high_index, 3);
        let high_byte = self.byte_at(e, body, high_word, high_lane);
        let high_shift = self.shl_lit(e, body, low_group, 1);
        let high_shifted = self.shr(e, body, high_byte, high_shift);
        let high2 = self.and_lit(e, body, high_shifted, 3);
        let high2 = self.shl_lit(e, body, high2, 4);
        let quant = self.bin(e, body, BinaryOperator::InclusiveOr, low4, high2);
        let scale_chunk_base = self.shl_lit(e, body, chunk, 3);
        let high_byte_half = self.shr_lit(e, body, high_byte_index, 4);
        let low_group_scale = self.shl_lit(e, body, low_group, 1);
        let local_scale_index = self.bin(
            e,
            body,
            BinaryOperator::Add,
            high_byte_half,
            low_group_scale,
        );
        let scale_index = self.bin(
            e,
            body,
            BinaryOperator::Add,
            scale_chunk_base,
            local_scale_index,
        );
        let scale_word_base = self.shr_lit(e, body, scale_index, 2);
        let scale_word_off = self.add_lit(e, body, scale_word_base, 48);
        let scale_word = self.load_word_dynamic(e, matrix, base, scale_word_off, body)?;
        let scale_lane = self.and_lit(e, body, scale_index, 3);
        let scale_byte = self.byte_at(e, body, scale_word, scale_lane);
        let scale = self.signed_byte_f32(e, body, scale_byte);
        let scale = self.mul(e, body, scale, d);
        let quant_f = self.as_f32(e, body, quant);
        let center = self.f32(e, 32.0);
        let centered = self.sub(e, body, quant_f, center);
        Ok(self.mul(e, body, centered, scale))
    }
}
