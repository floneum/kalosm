//! The quantized format table. Twelve rows: six formats x two layouts.
//!
//! Everything here is a table lookup or `const fn` arithmetic. No function in
//! this module selects a layout, and none mentions a device, a capability or a
//! kernel — `QLayout` is an operand attribute the extractor prices, not a
//! property this crate decides.

use fusor2_ir::Result;
use fusor2_ir::dtype::{QFmt, QLayout};
use fusor2_ir::ir::level2::{StorageView, TileExpr};

use crate::decode;
use crate::decode_k;

/// A format's decode program. Not one `ScalarExpr`: Q6K's 210-byte
/// non-word-aligned block with per-super-block group scales is not a
/// per-element formula. Returns the one decoded element as a scalar f32.
pub type BlockEmitFn = fn(&BlockDecodeArgs<'_>) -> Result<TileExpr>;

/// Inputs a [`BlockEmitFn`] decodes from.
///
/// A decode yields exactly one element. Every consumer — both emitters' single
/// quantized load and the cooperative staging fill — asks for one, and a
/// contraction over a block-quantized operand reaches its elements through the
/// staging fill's own lane arithmetic, never through a width carried in here.
#[derive(Clone, Debug)]
pub struct BlockDecodeArgs<'a> {
    pub src: &'a StorageView,
    pub layout: QLayout,
    pub k_base: TileExpr,
    pub col: TileExpr,
    pub mask: TileExpr,
    pub fill: TileExpr,
}

/// A decode program plus its identity. Equality and hashing are by `name`,
/// never by function-pointer address (codegen units may merge functions).
#[derive(Copy, Clone, Debug)]
pub struct BlockProgram {
    pub name: &'static str,
    pub emit: BlockEmitFn,
}

impl PartialEq for BlockProgram {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}
impl Eq for BlockProgram {}
impl std::hash::Hash for BlockProgram {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

/// One row of the quantized format table. Formats are data.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockSpec {
    pub fmt: QFmt,
    pub elements: u16,
    pub bytes: u16,
    pub layout: QLayout,
    pub decode: BlockProgram,
}

/// Every decodable `(format, layout)` pair, in `QFmt::ALL` order with
/// `Native` first in each pair. Index arithmetic is `fmt as usize * 2 +
/// layout as usize`; [`block_spec`] relies on it.
pub static BLOCK_SPECS: &[BlockSpec] = &[
    row(QFmt::Q4_0, QLayout::Native, decode::DECODE_Q4_0_NATIVE),
    row(QFmt::Q4_0, QLayout::F32Scales, decode::DECODE_Q4_0_F32),
    row(QFmt::Q5_0, QLayout::Native, decode::DECODE_Q5_0_NATIVE),
    row(QFmt::Q5_0, QLayout::F32Scales, decode::DECODE_Q5_0_F32),
    row(QFmt::Q8_0, QLayout::Native, decode::DECODE_Q8_0_NATIVE),
    row(QFmt::Q8_0, QLayout::F32Scales, decode::DECODE_Q8_0_F32),
    row(QFmt::Q4K, QLayout::Native, decode_k::DECODE_Q4K_NATIVE),
    row(QFmt::Q4K, QLayout::F32Scales, decode_k::DECODE_Q4K_F32),
    row(QFmt::Q5K, QLayout::Native, decode_k::DECODE_Q5K_NATIVE),
    row(QFmt::Q5K, QLayout::F32Scales, decode_k::DECODE_Q5K_F32),
    row(QFmt::Q6K, QLayout::Native, decode_k::DECODE_Q6K_NATIVE),
    row(QFmt::Q6K, QLayout::F32Scales, decode_k::DECODE_Q6K_F32),
];

/// One table row. `elements` and `bytes` are taken from `QFmt` so the
/// normative values live in exactly one place.
const fn row(fmt: QFmt, layout: QLayout, decode: BlockProgram) -> BlockSpec {
    BlockSpec {
        fmt,
        elements: fmt.block_elements() as u16,
        bytes: fmt.block_bytes(layout) as u16,
        layout,
        decode,
    }
}

/// The row for one `(format, layout)`. Total: the table is dense over
/// `QFmt::ALL x {Native, F32Scales}` and a unit test proves it.
pub fn block_spec(fmt: QFmt, layout: QLayout) -> &'static BlockSpec {
    let index = fmt as usize * 2 + layout as usize;
    debug_assert!(index < BLOCK_SPECS.len());
    &BLOCK_SPECS[index]
}

/// Byte offsets of the fields every decode program reads. The fields tile the
/// block exactly: no gaps, no overlap, widths summing to `block_bytes`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockFields {
    /// Super-block scale (`d`).
    pub scale: u32,
    /// Super-block minimum (`dmin`), for the K-quants that carry one.
    pub min: Option<u32>,
    /// `(offset, len)` of the packed per-group scale bytes.
    pub group_scales: Option<(u32, u32)>,
    /// High-bit plane, for the formats whose quant does not fit a nibble.
    pub qh: Option<u32>,
    /// Low-bit plane — always present, always the largest field.
    pub ql: u32,
    /// `true` iff `scale`/`min` are f16, which is exactly `layout == Native`.
    pub scale_is_f16: bool,
}

/// Field offsets for one `(format, layout)`. Native is the raw GGUF block
/// verbatim; `F32Scales` widens every f16 scale to f32 and shifts the rest.
pub const fn block_fields(fmt: QFmt, layout: QLayout) -> BlockFields {
    let f16 = matches!(layout, QLayout::Native);
    // How far the fields after one scale move when that scale widens.
    let bump = if f16 { 0 } else { 2 };
    match fmt {
        QFmt::Q4_0 => BlockFields {
            scale: 0,
            min: None,
            group_scales: None,
            qh: None,
            ql: 2 + bump,
            scale_is_f16: f16,
        },
        QFmt::Q5_0 => BlockFields {
            scale: 0,
            min: None,
            group_scales: None,
            qh: Some(2 + bump),
            ql: 6 + bump,
            scale_is_f16: f16,
        },
        QFmt::Q8_0 => BlockFields {
            scale: 0,
            min: None,
            group_scales: None,
            qh: None,
            ql: 2 + bump,
            scale_is_f16: f16,
        },
        QFmt::Q4K => BlockFields {
            scale: 0,
            min: Some(2 + bump),
            group_scales: Some((4 + 2 * bump, 12)),
            qh: None,
            ql: 16 + 2 * bump,
            scale_is_f16: f16,
        },
        QFmt::Q5K => BlockFields {
            scale: 0,
            min: Some(2 + bump),
            group_scales: Some((4 + 2 * bump, 12)),
            qh: Some(16 + 2 * bump),
            ql: 48 + 2 * bump,
            scale_is_f16: f16,
        },
        // Q6K puts its super-block scale last, which is why its native block
        // is 210 bytes and not word-aligned: every field read goes through a
        // byte-addressed load.
        QFmt::Q6K => BlockFields {
            scale: 208,
            min: None,
            group_scales: Some((192, 16)),
            qh: Some(128),
            ql: 0,
            scale_is_f16: f16,
        },
    }
}

/// Width of the `ql` plane in bytes.
pub const fn ql_width(fmt: QFmt) -> u32 {
    match fmt {
        QFmt::Q4_0 | QFmt::Q5_0 => 16,
        QFmt::Q8_0 => 32,
        QFmt::Q4K | QFmt::Q5K | QFmt::Q6K => 128,
    }
}

/// Width of the `qh` plane in bytes, or 0 when the format has none.
pub const fn qh_width(fmt: QFmt) -> u32 {
    match fmt {
        QFmt::Q5_0 => 4,
        QFmt::Q5K => 32,
        QFmt::Q6K => 64,
        _ => 0,
    }
}

/// Whether a block's stride is a whole number of u32 words.
pub const fn word_aligned(fmt: QFmt, layout: QLayout) -> bool {
    fmt.block_bytes(layout).is_multiple_of(4)
}

// ---------------------------------------------------------------------------
// Scalar reference decoder
// ---------------------------------------------------------------------------

/// Read the `scale`-shaped field at `offset` out of a raw block.
fn read_scale(block: &[u8], offset: u32, is_f16: bool) -> f32 {
    let o = offset as usize;
    if is_f16 {
        half::f16::from_le_bytes([block[o], block[o + 1]]).to_f32()
    } else {
        f32::from_le_bytes([block[o], block[o + 1], block[o + 2], block[o + 3]])
    }
}

/// Unpack the 12 packed-scale bytes of a Q4K/Q5K block into eight 6-bit group
/// scales and eight 6-bit group offsets.
///
/// A six-bit mask on words 0 and 1 gives groups 0-3; the low nibbles of word 2
/// supply scale bits 0-3 of groups 4-7 and its high nibbles the offset bits 0-3,
/// with bits 4-5 coming from the top two bits of words 0 and 1.
pub fn unpack_k4_scales_offsets(packed: &[u8; 12]) -> ([u8; 8], [u8; 8]) {
    const SIX_BITS: u32 = 0b0011_1111_0011_1111_0011_1111_0011_1111;
    const MSB_TWO: u32 = 0b1100_0000_1100_0000_1100_0000_1100_0000;
    const MSB_SCALES: u32 = 0b0000_1111_0000_1111_0000_1111_0000_1111;
    const MSB_OFFSET: u32 = 0b1111_0000_1111_0000_1111_0000_1111_0000;

    let w0 = u32::from_le_bytes([packed[0], packed[1], packed[2], packed[3]]);
    let w1 = u32::from_le_bytes([packed[4], packed[5], packed[6], packed[7]]);
    let w2 = u32::from_le_bytes([packed[8], packed[9], packed[10], packed[11]]);

    let first_scales = w0 & SIX_BITS;
    let first_offsets = w1 & SIX_BITS;
    let second_scales = ((w0 & MSB_TWO) >> 2) | (w2 & MSB_SCALES);
    let second_offsets = ((w1 & MSB_TWO) >> 2) | ((w2 & MSB_OFFSET) >> 4);

    let mut scales = [0u8; 8];
    let mut offsets = [0u8; 8];
    scales[..4].copy_from_slice(&first_scales.to_le_bytes());
    scales[4..].copy_from_slice(&second_scales.to_le_bytes());
    offsets[..4].copy_from_slice(&first_offsets.to_le_bytes());
    offsets[4..].copy_from_slice(&second_offsets.to_le_bytes());
    (scales, offsets)
}

/// The scalar reference decoder: one raw block in, `fmt.block_elements()` f32
/// out. This is the oracle the emitted decode programs, the CPU emitter's
/// correctness tests and conformance all compare against.
///
/// Panics if `block` is shorter than `fmt.block_bytes(layout)` or `out` is
/// shorter than `fmt.block_elements()`.
pub fn cpu_dequantize_block(fmt: QFmt, layout: QLayout, block: &[u8], out: &mut [f32]) {
    let fields = block_fields(fmt, layout);
    let bytes = fmt.block_bytes(layout) as usize;
    let elements = fmt.block_elements() as usize;
    assert!(
        block.len() >= bytes,
        "block is {} bytes, {fmt:?}/{layout:?} needs {bytes}",
        block.len()
    );
    assert!(
        out.len() >= elements,
        "output is {} lanes, {fmt:?} decodes {elements}",
        out.len()
    );
    let ql = fields.ql as usize;
    match fmt {
        QFmt::Q4_0 => {
            let scale = read_scale(block, fields.scale, fields.scale_is_f16);
            for i in 0..16 {
                let byte = block[ql + i];
                out[i] = ((byte & 0x0f) as i32 - 8) as f32 * scale;
                out[i + 16] = ((byte >> 4) as i32 - 8) as f32 * scale;
            }
        }
        QFmt::Q5_0 => {
            let scale = read_scale(block, fields.scale, fields.scale_is_f16);
            let h = fields.qh.unwrap() as usize;
            let high = u32::from_le_bytes([block[h], block[h + 1], block[h + 2], block[h + 3]]);
            for i in 0..16 {
                let byte = block[ql + i];
                let low_bit = ((high >> i) as u8 & 1) << 4;
                let high_bit = ((high >> (i + 16)) as u8 & 1) << 4;
                out[i] = (((byte & 0x0f) | low_bit) as i32 - 16) as f32 * scale;
                out[i + 16] = (((byte >> 4) | high_bit) as i32 - 16) as f32 * scale;
            }
        }
        QFmt::Q8_0 => {
            let scale = read_scale(block, fields.scale, fields.scale_is_f16);
            for (i, slot) in out[..32].iter_mut().enumerate() {
                *slot = block[ql + i] as i8 as f32 * scale;
            }
        }
        QFmt::Q4K => {
            let d = read_scale(block, fields.scale, fields.scale_is_f16);
            let dmin = read_scale(block, fields.min.unwrap(), fields.scale_is_f16);
            let (gs, len) = fields.group_scales.unwrap();
            let packed: [u8; 12] = block[gs as usize..gs as usize + len as usize]
                .try_into()
                .expect("group scales are 12 bytes");
            let (scales, offsets) = unpack_k4_scales_offsets(&packed);
            for (e, slot) in out[..256].iter_mut().enumerate() {
                let group = e >> 5;
                let byte = block[ql + (group >> 1) * 32 + (e & 31)];
                let q = if group & 1 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                *slot = scales[group] as f32 * d * q as f32 - offsets[group] as f32 * dmin;
            }
        }
        QFmt::Q5K => {
            let d = read_scale(block, fields.scale, fields.scale_is_f16);
            let dmin = read_scale(block, fields.min.unwrap(), fields.scale_is_f16);
            let (gs, len) = fields.group_scales.unwrap();
            let packed: [u8; 12] = block[gs as usize..gs as usize + len as usize]
                .try_into()
                .expect("group scales are 12 bytes");
            let (scales, offsets) = unpack_k4_scales_offsets(&packed);
            let qh = fields.qh.unwrap() as usize;
            for (e, slot) in out[..256].iter_mut().enumerate() {
                let group = e >> 5;
                let byte = block[ql + (group >> 1) * 32 + (e & 31)];
                let nibble = if group & 1 == 0 {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let bit = (block[qh + (e & 31)] >> group) & 1;
                let q = nibble + 16 * bit;
                *slot = scales[group] as f32 * d * q as f32 - offsets[group] as f32 * dmin;
            }
        }
        QFmt::Q6K => {
            let d = read_scale(block, fields.scale, fields.scale_is_f16);
            let (gs, _) = fields.group_scales.unwrap();
            let qh = fields.qh.unwrap() as usize;
            for (e, slot) in out[..256].iter_mut().enumerate() {
                let chunk = e >> 7;
                let local = e & 127;
                let hb = local & 31;
                let low_group = local >> 5;
                let low_byte = block[ql + chunk * 64 + (low_group & 1) * 32 + hb];
                let low4 = (low_byte >> ((low_group >> 1) * 4)) & 0x0f;
                let high_byte = block[qh + chunk * 32 + hb];
                let high2 = (high_byte >> (low_group * 2)) & 0x03;
                let q = ((high2 << 4) | low4) as i32 - 32;
                let scale_index = chunk * 8 + (hb >> 4) + low_group * 2;
                // Signed: a negative group scale flips the sign of the lane.
                let group_scale = block[gs as usize + scale_index] as i8;
                *slot = d * group_scale as f32 * q as f32;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashSet;

    #[test]
    fn block_spec_table_is_total_and_matches_contract() {
        let mut names = FxHashSet::default();
        for fmt in QFmt::ALL {
            for layout in [QLayout::Native, QLayout::F32Scales] {
                let spec = block_spec(fmt, layout);
                assert_eq!(spec.fmt, fmt);
                assert_eq!(spec.layout, layout);
                assert_eq!(spec.bytes as u32, fmt.block_bytes(layout));
                assert_eq!(spec.elements as u32, fmt.block_elements());
                assert!(
                    names.insert(spec.decode.name),
                    "duplicate block program name {}",
                    spec.decode.name
                );
            }
        }
        assert_eq!(BLOCK_SPECS.len(), 12);
        assert_eq!(names.len(), 12);
    }

    #[test]
    fn block_fields_land_inside_the_block() {
        /// Width of `scale` (and of `min`, when present) in bytes.
        const fn scale_width(f: &BlockFields) -> u32 {
            if f.scale_is_f16 { 2 } else { 4 }
        }
        for fmt in QFmt::ALL {
            for layout in [QLayout::Native, QLayout::F32Scales] {
                let f = block_fields(fmt, layout);
                let bytes = fmt.block_bytes(layout);
                let mut spans: Vec<(u32, u32)> = vec![(f.scale, scale_width(&f))];
                if let Some(min) = f.min {
                    spans.push((min, scale_width(&f)));
                }
                if let Some((off, len)) = f.group_scales {
                    spans.push((off, len));
                }
                if let Some(qh) = f.qh {
                    spans.push((qh, qh_width(fmt)));
                }
                spans.push((f.ql, ql_width(fmt)));

                let total: u32 = spans.iter().map(|(_, w)| *w).sum();
                assert_eq!(total, bytes, "{fmt:?}/{layout:?} fields do not tile");

                spans.sort_unstable();
                let mut cursor = 0;
                for (off, width) in spans {
                    assert_eq!(off, cursor, "{fmt:?}/{layout:?} gap or overlap at {off}");
                    assert!(off + width <= bytes, "{fmt:?}/{layout:?} field overruns");
                    cursor = off + width;
                }
                assert_eq!(cursor, bytes);
                assert_eq!(f.scale_is_f16, layout == QLayout::Native);
            }
        }
    }

    #[test]
    fn storage_size_and_alignment_are_derived() {
        assert!(!word_aligned(QFmt::Q6K, QLayout::Native));
        assert!(word_aligned(QFmt::Q6K, QLayout::F32Scales));
        assert!(!word_aligned(QFmt::Q4_0, QLayout::Native));
        assert!(word_aligned(QFmt::Q4_0, QLayout::F32Scales));
        assert!(!word_aligned(QFmt::Q5_0, QLayout::Native));
        assert!(!word_aligned(QFmt::Q8_0, QLayout::Native));
        assert!(word_aligned(QFmt::Q4K, QLayout::Native));
    }

    #[test]
    fn cpu_dequantize_matches_ggml_vectors() {
        // Q4_0: scale 0.5, nibbles 0..=15 in both halves. llama.cpp's
        // split-half order puts the low nibble of byte i at lane i and the
        // high nibble at lane i + 16.
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        for i in 0..16 {
            block[2 + i] = (i as u8) | ((i as u8) << 4);
        }
        let mut out = [0.0f32; 32];
        cpu_dequantize_block(QFmt::Q4_0, QLayout::Native, &block, &mut out);
        for (i, pair) in out[..16].iter().zip(&out[16..]).enumerate() {
            assert_eq!(*pair.0, (i as f32 - 8.0) * 0.5);
            assert_eq!(*pair.1, (i as f32 - 8.0) * 0.5);
        }

        // Q5_0: every fifth bit set, every nibble zero => q = 16, centered 0.
        let mut block = vec![0u8; 22];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[2..6].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        let mut out = [0.0f32; 32];
        cpu_dequantize_block(QFmt::Q5_0, QLayout::Native, &block, &mut out);
        for v in out {
            assert_eq!(v, 0.0);
        }
        // ... and with no fifth bit, nibble 15 centers to -1.
        block[2..6].copy_from_slice(&0u32.to_le_bytes());
        for i in 0..16 {
            block[6 + i] = 0xFF;
        }
        cpu_dequantize_block(QFmt::Q5_0, QLayout::Native, &block, &mut out);
        for v in out {
            assert_eq!(v, -1.0);
        }

        // Q8_0: signed bytes times the scale.
        let mut block = vec![0u8; 34];
        block[0..2].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        for i in 0..32 {
            block[2 + i] = (i as i32 - 16) as i8 as u8;
        }
        let mut out = [0.0f32; 32];
        cpu_dequantize_block(QFmt::Q8_0, QLayout::Native, &block, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, (i as f32 - 16.0) * 0.25);
        }

        // Q4K: groups 0..4 scale 1, offset 0; every quant nibble 3.
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        for g in 0..4 {
            block[4 + g] = 1;
            block[8 + g] = 0;
        }
        for i in 0..128 {
            block[16 + i] = 0x33;
        }
        let mut out = [0.0f32; 256];
        cpu_dequantize_block(QFmt::Q4K, QLayout::Native, &block, &mut out);
        for (e, v) in out[..128].iter().enumerate() {
            assert_eq!(*v, 6.0, "lane {e}");
        }

        // Q5K: same scale setup, plus one high bit per lane in group 0.
        let mut block = vec![0u8; 176];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[4] = 1;
        for i in 0..32 {
            block[16 + i] = 0x01;
        }
        for i in 0..128 {
            block[48 + i] = 0x02;
        }
        let mut out = [0.0f32; 256];
        cpu_dequantize_block(QFmt::Q5K, QLayout::Native, &block, &mut out);
        for (e, v) in out[..32].iter().enumerate() {
            assert_eq!(*v, 18.0, "lane {e}");
        }

        // Q6K: the group scales are signed i8 and must be sign-extended.
        let mut block = vec![0u8; 210];
        for i in 0..16 {
            block[192 + i] = 1u8;
        }
        block[192] = (-3i8) as u8;
        block[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        let mut out = [0.0f32; 256];
        cpu_dequantize_block(QFmt::Q6K, QLayout::Native, &block, &mut out);
        // Group 0 covers lanes 0..16 of chunk 0; quant 0 centers to -32.
        for (e, v) in out[..16].iter().enumerate() {
            assert_eq!(*v, 96.0, "lane {e}");
        }
        for (e, v) in out[16..32].iter().enumerate() {
            assert_eq!(*v, -32.0, "lane {}", e + 16);
        }
    }

    #[test]
    fn k4_scale_unpack_round_trips_the_reference_bit_pattern() {
        // Groups 0..4 read six bits straight out of words 0/1; groups 4..8
        // take bits 4-5 from the top of words 0/1 and bits 0-3 from word 2.
        let mut packed = [0u8; 12];
        packed[0] = 0b1100_0001;
        packed[4] = 0b0100_0010;
        packed[8] = 0b0101_1010;
        let (scales, offsets) = unpack_k4_scales_offsets(&packed);
        assert_eq!(scales[0], 1);
        assert_eq!(offsets[0], 2);
        assert_eq!(scales[4], 0b11_1010);
        assert_eq!(offsets[4], 0b01_0101);
    }
}
