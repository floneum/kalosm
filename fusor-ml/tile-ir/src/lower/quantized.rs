use super::*;

struct Q6KBlockParts {
    low_words: [Handle<Expression>; 2],
    high_words: [Handle<Expression>; 2],
    low_shift: Handle<Expression>,
    high_shift: Handle<Expression>,
    scale: Handle<Expression>,
}

struct Q4KBlockParts {
    base: Handle<Expression>,
    q_base: Handle<Expression>,
    group: Handle<Expression>,
    scale: Handle<Expression>,
    min: Handle<Expression>,
}

struct Q8_0BlockParts {
    scale: Handle<Expression>,
    words: [Handle<Expression>; 2],
}

/// Quant-byte extraction layout for the affine GGML formats.
///
/// `Q4` packs two 4-bit nibbles per byte; `Q5` adds an extra high-bit
/// register (`high_offset`) so the upper bit of each 5-bit value can be
/// reconstructed alongside the low nibble at `data_offset`.
#[derive(Clone, Copy)]
enum AffineNibble {
    Q4 { data_offset: u32 },
    Q5 { high_offset: u32, data_offset: u32 },
}

/// The simple affine GGML quant family: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`,
/// `Q8_0`, `Q8_1`. Each block stores a single scale (and optionally a min)
/// and dequantizes via one of three affine forms:
///
/// - `Centered`: `(quant − center) · scale` — used by Q4_0/Q5_0.
/// - `ScaleMin`: `quant · scale + min` — used by Q4_1/Q5_1.
/// - `Q8`: `signed_byte · scale` — used by Q8_0/Q8_1.
///
/// The richer K-quants (Q2K…Q8K) carry per-block group scales and live in
/// their own dedicated lowering paths.
#[derive(Clone, Copy)]
enum AffineDequantSpec {
    Centered { nibble: AffineNibble, center: f32 },
    ScaleMin { nibble: AffineNibble },
    Q8 { data_offset: u32 },
}

impl AffineDequantSpec {
    fn for_format(format: GgmlQuantFormat) -> Option<Self> {
        Some(match format {
            GgmlQuantFormat::Q4_0 => Self::Centered {
                nibble: AffineNibble::Q4 { data_offset: 1 },
                center: 8.0,
            },
            GgmlQuantFormat::Q5_0 => Self::Centered {
                nibble: AffineNibble::Q5 {
                    high_offset: 1,
                    data_offset: 2,
                },
                center: 16.0,
            },
            GgmlQuantFormat::Q8_0 => Self::Q8 { data_offset: 1 },
            GgmlQuantFormat::Q4_1 => Self::ScaleMin {
                nibble: AffineNibble::Q4 { data_offset: 2 },
            },
            GgmlQuantFormat::Q5_1 => Self::ScaleMin {
                nibble: AffineNibble::Q5 {
                    high_offset: 2,
                    data_offset: 3,
                },
            },
            GgmlQuantFormat::Q8_1 => Self::Q8 { data_offset: 2 },
            GgmlQuantFormat::Q2K
            | GgmlQuantFormat::Q3K
            | GgmlQuantFormat::Q4K
            | GgmlQuantFormat::Q5K
            | GgmlQuantFormat::Q6K
            | GgmlQuantFormat::Q8K => return None,
        })
    }
}

macro_rules! expr_binary_helper {
    ($name:ident, $op:ident) => {
        fn $name(
            &self,
            e: &mut Arena<Expression>,
            emits: &mut Vec<Range<Expression>>,
            left: Handle<Expression>,
            right: Handle<Expression>,
        ) -> Handle<Expression> {
            self.bin(e, emits, BinaryOperator::$op, left, right)
        }
    };
}

macro_rules! expr_literal_binary_helper {
    ($name:ident, $op:ident) => {
        fn $name(
            &self,
            e: &mut Arena<Expression>,
            emits: &mut Vec<Range<Expression>>,
            left: Handle<Expression>,
            right: u32,
        ) -> Handle<Expression> {
            let right = self.u32(e, right);
            self.bin(e, emits, BinaryOperator::$op, left, right)
        }
    };
}

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
        let value = if let Some(spec) = AffineDequantSpec::for_format(matrix.format) {
            self.dequant_affine(expressions, matrix, base, q, spec, &mut emits)?
        } else {
            match matrix.format {
                GgmlQuantFormat::Q2K => self.dequant_q2k(expressions, matrix, base, q, &mut emits)?,
                GgmlQuantFormat::Q3K => self.dequant_q3k(expressions, matrix, base, q, &mut emits)?,
                GgmlQuantFormat::Q4K => self.dequant_q4k(expressions, matrix, base, q, &mut emits)?,
                GgmlQuantFormat::Q5K => self.dequant_q5k(expressions, matrix, base, q, &mut emits)?,
                GgmlQuantFormat::Q6K => self.dequant_q6k(expressions, matrix, base, q, &mut emits)?,
                GgmlQuantFormat::Q8K => self.dequant_q8k(expressions, matrix, base, q, &mut emits)?,
                GgmlQuantFormat::Q4_0
                | GgmlQuantFormat::Q4_1
                | GgmlQuantFormat::Q5_0
                | GgmlQuantFormat::Q5_1
                | GgmlQuantFormat::Q8_0
                | GgmlQuantFormat::Q8_1 => unreachable!("legacy formats handled above"),
            }
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
        let parts = self.q8_0_block_parts8(expressions, matrix, k_base, col, &mut emits)?;
        let mut values = Vec::with_capacity(8);
        for signed in self.q8_0_components8(expressions, &mut emits, &parts) {
            values.push(self.mul(expressions, &mut emits, signed, parts.scale));
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
        let parts = self.q8_0_block_parts8(expressions, matrix, k_base, col, &mut emits)?;
        let q_components = self.q8_0_components8(expressions, &mut emits, &parts);
        let sum = self.dot_vec4_chunks(expressions, &mut emits, a, &q_components);
        Ok((self.mul(expressions, &mut emits, sum, parts.scale), emits))
    }

    fn q8_0_block_parts8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Q8_0BlockParts, LowerError> {
        let block = self.div_literal_u32_emitted(expressions, k_base, 32, emits);
        let q = self.and_lit(expressions, emits, k_base, 31);
        let col_block = self.mul_literal_u32_emitted(expressions, col, matrix.rows / 32, emits);
        let block_index = self.bin(expressions, emits, BinaryOperator::Add, col_block, block);
        let base = self.mul_literal_u32_emitted(expressions, block_index, 9, emits);
        let scale_word = self.load_word(expressions, matrix, base, 0, emits)?;
        let scale = self.bitcast_f32(expressions, emits, scale_word);
        let q_word = self.shr_lit(expressions, emits, q, 2);
        let word0_off = self.add_lit(expressions, emits, q_word, 1);
        let word1_off = self.add_lit(expressions, emits, q_word, 2);
        let word0 = self.load_word_dynamic(expressions, matrix, base, word0_off, emits)?;
        let word1 = self.load_word_dynamic(expressions, matrix, base, word1_off, emits)?;
        Ok(Q8_0BlockParts {
            scale,
            words: [word0, word1],
        })
    }

    fn q8_0_components8(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        parts: &Q8_0BlockParts,
    ) -> [Handle<Expression>; 8] {
        std::array::from_fn(|lane| {
            let byte_lane = expressions.append(
                Expression::Literal(Literal::U32((lane % 4) as u32)),
                Span::default(),
            );
            let word = parts.words[usize::from(lane >= 4)];
            let byte = self.byte_at(expressions, emits, word, byte_lane);
            self.signed_byte_f32(expressions, emits, byte)
        })
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

    fn q6k_block_parts(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Q6KBlockParts, LowerError> {
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
        let low_words =
            self.load_word_pair_dynamic(expressions, matrix, base, low_word_base, 0, emits)?;
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
        let high_words =
            self.load_word_pair_dynamic(expressions, matrix, base, high_word_base, 32, emits)?;
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

        Ok(Q6KBlockParts {
            low_words,
            high_words,
            low_shift,
            high_shift,
            scale,
        })
    }

    fn q6k_centered_component(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        parts: &Q6KBlockParts,
        lane: usize,
    ) -> Handle<Expression> {
        let quant = self.q6k_quant_component(expressions, emits, parts, lane);
        self.center_q6k_quant(expressions, emits, quant)
    }

    fn q6k_quant_component(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        parts: &Q6KBlockParts,
        lane: usize,
    ) -> Handle<Expression> {
        let byte_lane = expressions.append(
            Expression::Literal(Literal::U32((lane % 4) as u32)),
            Span::default(),
        );
        let low_word = parts.low_words[usize::from(lane >= 4)];
        let low_byte = self.byte_at(expressions, emits, low_word, byte_lane);
        let low_shifted = self.shr(expressions, emits, low_byte, parts.low_shift);
        let low4 = self.and_lit(expressions, emits, low_shifted, 0x0f);

        let high_word = parts.high_words[usize::from(lane >= 4)];
        let high_byte = self.byte_at(expressions, emits, high_word, byte_lane);
        let high_shifted = self.shr(expressions, emits, high_byte, parts.high_shift);
        let high2 = self.and_lit(expressions, emits, high_shifted, 3);
        let high2 = self.shl_lit(expressions, emits, high2, 4);
        self.bin(expressions, emits, BinaryOperator::InclusiveOr, low4, high2)
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
        let parts = self.q6k_block_parts(expressions, matrix, k_base, col, &mut emits)?;

        let mut values = Vec::with_capacity(8);
        for lane in 0..8 {
            let centered = self.q6k_centered_component(expressions, &mut emits, &parts, lane);
            values.push(self.mul(expressions, &mut emits, centered, parts.scale));
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

    pub(super) fn dequantize_q6k_dot8(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &[Handle<Expression>; 8],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q6K {
            return Err(LowerError::UnsupportedOperation(
                "q6k dot8 only supports Q6K",
            ));
        }

        let mut emits = Vec::new();
        let parts = self.q6k_block_parts(expressions, matrix, k_base, col, &mut emits)?;

        let mut q_components = Vec::with_capacity(8);
        for lane in 0..8 {
            q_components.push(self.q6k_centered_component(expressions, &mut emits, &parts, lane));
        }

        let sum = self.dot_vec4_chunks(expressions, &mut emits, a, &q_components);
        Ok((self.mul(expressions, &mut emits, sum, parts.scale), emits))
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
        let total = self.q8_activation_packs_dot(
            expressions,
            &mut emits,
            a,
            pack_offset,
            b_scale,
            b_packs,
            Some(b_min),
        );
        Ok((total, emits))
    }

    pub(super) fn q4k_f32_dot(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if matrix.format != GgmlQuantFormat::Q4K || a.is_empty() || !a.len().is_multiple_of(8) {
            return Err(LowerError::UnsupportedOperation(
                "q4k f32 dot requires Q4K and a multiple of 8 activation values",
            ));
        }
        if a.len() == 32 {
            return self.q4k_f32_dot_exact::<32, 8>(
                expressions,
                matrix,
                k_base,
                col,
                a,
                true,
            );
        }
        if a.len() == 16 {
            return self.q4k_f32_dot_exact::<16, 4>(
                expressions,
                matrix,
                k_base,
                col,
                a,
                false,
            );
        }

        let mut emits = Vec::new();
        let mut total = self.f32(expressions, 0.0);
        for pack_offset in (0..a.len()).step_by(8) {
            let k = self.add_lit(expressions, &mut emits, k_base, pack_offset as u32);
            let (scale, min, quants) =
                self.q4k_quant_values::<8, 2>(expressions, matrix, k, col, false, &mut emits)?;
            let mut weighted_sum = self.f32(expressions, 0.0);
            let mut activation_sum = self.f32(expressions, 0.0);
            for lane in 0..8 {
                let q = self.as_f32(expressions, &mut emits, quants[lane]);
                let activation = a[pack_offset + lane];
                let weighted = self.mul(expressions, &mut emits, activation, q);
                weighted_sum = self.bin(
                    expressions,
                    &mut emits,
                    BinaryOperator::Add,
                    weighted_sum,
                    weighted,
                );
                activation_sum = self.bin(
                    expressions,
                    &mut emits,
                    BinaryOperator::Add,
                    activation_sum,
                    activation,
                );
            }
            let scaled = self.mul(expressions, &mut emits, weighted_sum, scale);
            let min_term = self.mul(expressions, &mut emits, activation_sum, min);
            let chunk = self.sub(expressions, &mut emits, scaled, min_term);
            total = self.bin(expressions, &mut emits, BinaryOperator::Add, total, chunk);
        }
        Ok((total, emits))
    }

    fn q4k_f32_dot_exact<const N: usize, const WORDS: usize>(
        &self,
        expressions: &mut Arena<Expression>,
        matrix: &QuantizedMatrix,
        k_base: Handle<Expression>,
        col: Handle<Expression>,
        a: &[Handle<Expression>],
        whole_group_pair: bool,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        debug_assert_eq!(WORDS * 4, N);
        debug_assert_eq!(a.len(), N);

        let mut emits = Vec::new();
        let (scale, min, quants) =
            self.q4k_quant_values::<N, WORDS>(
                expressions,
                matrix,
                k_base,
                col,
                whole_group_pair,
                &mut emits,
            )?;

        let total = self.q4k_f32_weighted_sum(expressions, &mut emits, scale, min, &quants, a);
        Ok((total, emits))
    }

    fn q4k_f32_weighted_sum(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        scale: Handle<Expression>,
        min: Handle<Expression>,
        quants: &[Handle<Expression>],
        a: &[Handle<Expression>],
    ) -> Handle<Expression> {
        let weighted_sum = self.dot_quant_vec4_chunks(expressions, emits, a, quants);
        let activation_sum = self.sum_values(expressions, emits, a);
        let scaled = self.mul(expressions, emits, weighted_sum, scale);
        let min_term = self.mul(expressions, emits, activation_sum, min);
        self.sub(expressions, emits, scaled, min_term)
    }

    fn sum_values(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        values: &[Handle<Expression>],
    ) -> Handle<Expression> {
        let mut total = self.f32(expressions, 0.0);
        for value in values {
            total = self.bin(expressions, emits, BinaryOperator::Add, total, *value);
        }
        total
    }

    fn dot_quant_vec4_chunks(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
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
                .map(|quant| self.as_f32(expressions, emits, *quant))
                .collect::<Vec<_>>();
            let dot = self.dot_vec4(expressions, emits, left_chunk, &right_chunk);
            total = self.bin(expressions, emits, BinaryOperator::Add, total, dot);
        }
        total
    }

    fn dot_vec4_chunks(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
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
        let mut total = self.dot_vec4(expressions, emits, left_chunk, right_chunk);
        for (left_chunk, right_chunk) in chunks {
            let dot = self.dot_vec4(expressions, emits, left_chunk, right_chunk);
            total = self.bin(expressions, emits, BinaryOperator::Add, total, dot);
        }
        total
    }

    fn dot_vec4(
        &self,
        expressions: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: &[Handle<Expression>],
        right: &[Handle<Expression>],
    ) -> Handle<Expression> {
        debug_assert_eq!(left.len(), 4);
        debug_assert_eq!(right.len(), 4);

        let left = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: left.to_vec(),
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, left));
        let right = expressions.append(
            Expression::Compose {
                ty: self.f32_vec4_ty,
                components: right.to_vec(),
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, right));
        let dot = expressions.append(
            Expression::Math {
                fun: MathFunction::Dot,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, dot));
        dot
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn q4k_ggml_dot(
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

    pub(super) fn q6k_ggml_dot(
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

    fn q8_activation_packs_dot(
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

    fn q4k_quant_values<const N: usize, const WORDS: usize>(
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

    fn q4k_block_parts(
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

    fn q4k_block_parts_from_block(
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

    fn q4k_quant_words<const WORDS: usize>(
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

    fn q6k_quant_packs8(
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

    fn dequant_affine(
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

    fn affine_nibble(
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

    fn q4_block_nibble(
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

    fn q5_block_nibble(
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
        self.dequant_q8_scaled(e, matrix, base, q, 1, emits)
    }

    fn dequant_q8_scaled(
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

    fn load_word_pair_dynamic(
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

    fn q4k_scale_min_bytes(
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

    expr_binary_helper!(shr, ShiftRight);
    expr_literal_binary_helper!(shr_lit, ShiftRight);
    expr_literal_binary_helper!(shl_lit, ShiftLeft);
    expr_literal_binary_helper!(and_lit, And);

    fn add_lit(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        self.add_literal_u32_emitted(e, left, right, emits)
    }

    expr_binary_helper!(sub, Subtract);
    expr_binary_helper!(mul, Multiply);
    expr_binary_helper!(div, Divide);

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

    fn compose_f32_vec4(
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

    fn dot_f32_vec4(
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

    fn vec4_component(
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

    fn u8_lane_f32(
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

    fn center_q6k_quant(
        &self,
        e: &mut Arena<Expression>,
        emits: &mut Vec<Range<Expression>>,
        quant: Handle<Expression>,
    ) -> Handle<Expression> {
        let quant = self.as_f32(e, emits, quant);
        let center = self.f32(e, 32.0);
        self.sub(e, emits, quant, center)
    }

    fn q4k_ggml_accumulate_word_scalar(
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

    fn q4k_ggml_accumulate_word_vector(
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
