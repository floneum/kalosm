use super::*;

mod dequant_block;
mod dot;
pub(super) mod dot_lowering;
mod helpers;
mod values;

pub(super) struct Q6KBlockParts {
    low_words: [Handle<Expression>; 2],
    high_words: [Handle<Expression>; 2],
    low_shift: Handle<Expression>,
    high_shift: Handle<Expression>,
    scale: Handle<Expression>,
}

pub(super) struct Q4KBlockParts {
    base: Handle<Expression>,
    q_base: Handle<Expression>,
    group: Handle<Expression>,
    scale: Handle<Expression>,
    min: Handle<Expression>,
}

pub(super) struct Q8_0BlockParts {
    scale: Handle<Expression>,
    words: [Handle<Expression>; 2],
}

/// Quant-byte extraction layout for the affine GGML formats.
///
/// `Q4` packs two 4-bit nibbles per byte; `Q5` adds an extra high-bit
/// register (`high_offset`) so the upper bit of each 5-bit value can be
/// reconstructed alongside the low nibble at `data_offset`.
#[derive(Clone, Copy)]
pub(super) enum AffineNibble {
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
pub(super) enum AffineDequantSpec {
    Centered { nibble: AffineNibble, center: f32 },
    ScaleMin { nibble: AffineNibble },
    Q8 { data_offset: u32 },
}

impl AffineDequantSpec {
    pub(super) fn for_format(format: GgmlQuantFormat) -> Option<Self> {
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
