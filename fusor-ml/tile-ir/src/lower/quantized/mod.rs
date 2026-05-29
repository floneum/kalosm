use super::*;

mod dequant_block;
mod dot;
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

/// `(scale, min)` factor pair plus per-quantization-block decoded data.
/// `data` is `[Handle<Expression>; N]` where `N` is either the per-quad pack
/// count (`2`) or the dequantized lane count (`8`/`16`/`32`).
pub(in crate::lower) struct Q4KQuantBlock<const N: usize> {
    pub scale: Handle<Expression>,
    pub min: Handle<Expression>,
    pub data: [Handle<Expression>; N],
}

/// Activation handles consumed by the Q4K ggml dot lowering: 16 low-nibble
/// loads, 16 high-nibble loads, and 4 per-pair sums. Bundled to keep
/// `q4k_ggml_dot` from carrying three parallel slice arguments.
pub(in crate::lower) struct Q4KGgmlActivationHandles<'a> {
    pub low: &'a [Handle<Expression>],
    pub high: &'a [Handle<Expression>],
    pub sums: &'a [Handle<Expression>],
}

/// Resolved `(block, c0, c1, col)` coordinates for a Q4K/Q6K ggml dot helper.
/// Q4K uses `(iq, ir)` for `(c0, c1)`; Q6K uses `(ip, il)`.
pub(in crate::lower) struct GgmlBlockCoords {
    pub block: Handle<Expression>,
    pub c0: Handle<Expression>,
    pub c1: Handle<Expression>,
    pub col: Handle<Expression>,
}

#[derive(Clone, Copy)]
pub(in crate::lower) struct QuantDotCoords {
    pub k_base: Handle<Expression>,
    pub col: Handle<Expression>,
}

pub(in crate::lower) struct Q8ActivationDotRhs {
    pub scale: Handle<Expression>,
    pub packs: [Handle<Expression>; 2],
    pub min: Option<Handle<Expression>>,
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
            GgmlQuantFormat::Q4_0 | GgmlQuantFormat::Q4_0Native => Self::Centered {
                nibble: AffineNibble::Q4 { data_offset: 1 },
                center: 8.0,
            },
            GgmlQuantFormat::Q5_0 | GgmlQuantFormat::Q5_0Native => Self::Centered {
                nibble: AffineNibble::Q5 {
                    high_offset: 1,
                    data_offset: 2,
                },
                center: 16.0,
            },
            GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q8_0Native => Self::Q8 { data_offset: 1 },
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
            | GgmlQuantFormat::Q4KNative
            | GgmlQuantFormat::Q5K
            | GgmlQuantFormat::Q5KNative
            | GgmlQuantFormat::Q6K
            | GgmlQuantFormat::Q6KNative
            | GgmlQuantFormat::Q8K => return None,
        })
    }
}
