//! Block decode programs for the 256-element K-quants.
//!
//! These are exactly why `decode` is a `BlockProgram` and not a `ScalarExpr`:
//! Q6K's 210-byte non-word-aligned block with per-super-block group scales is
//! not a per-element formula, which is why the reference needs
//! `Q4KBlockParts` / `Q6KBlockParts`.
//!
//! Owned by W11.

use fusor2_ir::Result;
use fusor2_ir::dtype::{QFmt, QLayout};
use fusor2_ir::ir::level2::{TileCompareOp, TileExpr};

use crate::blocks::{BlockDecodeArgs, BlockFields, BlockProgram, block_fields};
use crate::decode::{
    add, and_lit, block_base_and_q, cmp, expect_layout, f32_lit, finish, load_block_byte,
    load_scale_f32, mul, or, sel, shl_lit, shr, shr_lit, signed_byte_f32, sub, u32_lit, u32_to_f32,
};

/// Six-bit scale and six-bit offset of one 32-element group, read straight out
/// of the 12 packed bytes.
///
/// Groups 0-3 take six bits from bytes `0..4` (scale) and `4..8` (offset).
/// Groups 4-7 take bits 4-5 from the top two bits of those same bytes and bits
/// 0-3 from the low / high nibble of bytes `8..12`. This is
/// `unpack_k4_scales_offsets` evaluated at one dynamic group index, which is
/// three byte loads instead of twelve.
fn k4_group_scale_min(
    args: &BlockDecodeArgs<'_>,
    base: &TileExpr,
    fields: &BlockFields,
    group: &TileExpr,
) -> (TileExpr, TileExpr) {
    let (gs, _) = fields.group_scales.expect("K4 carries group scales");
    let lane = and_lit(group.clone(), 3);
    let high = cmp(TileCompareOp::Ge, group.clone(), u32_lit(4));

    let scale_byte = load_block_byte(args, base, gs, Some(lane.clone()));
    let min_byte = load_block_byte(args, base, gs + 4, Some(lane.clone()));
    let extra_byte = load_block_byte(args, base, gs + 8, Some(lane));

    let scale_low = and_lit(scale_byte.clone(), 0x3f);
    let scale_high = or(
        and_lit(extra_byte.clone(), 0x0f),
        shr_lit(and_lit(scale_byte, 0xc0), 2),
    );
    let scale = sel(high.clone(), scale_high, scale_low);

    let min_low = and_lit(min_byte.clone(), 0x3f);
    let min_high = or(
        and_lit(shr_lit(extra_byte, 4), 0x0f),
        shr_lit(and_lit(min_byte, 0xc0), 2),
    );
    let min = sel(high, min_high, min_low);

    (scale, min)
}

/// The whole-word form of the same unpack: three packed words in, eight group
/// scales and eight group offsets out.
///
/// Ported from the reference's `first_scales_min_k4` / `second_scales_min_k4`.
/// Emitters that decode a whole super-block at once want this; the per-lane
/// programs below use the group-indexed form instead.
pub fn unpack_k4_scales_offsets(packed: [TileExpr; 3]) -> ([TileExpr; 8], [TileExpr; 8]) {
    const SIX_BITS: u32 = 0b0011_1111_0011_1111_0011_1111_0011_1111;
    const MSB_TWO: u32 = 0b1100_0000_1100_0000_1100_0000_1100_0000;
    const MSB_SCALES: u32 = 0b0000_1111_0000_1111_0000_1111_0000_1111;
    const MSB_OFFSET: u32 = 0b1111_0000_1111_0000_1111_0000_1111_0000;

    let [w0, w1, w2] = packed;
    let first_scales = and_lit(w0.clone(), SIX_BITS);
    let first_offsets = and_lit(w1.clone(), SIX_BITS);
    let second_scales = or(
        shr_lit(and_lit(w0, MSB_TWO), 2),
        and_lit(w2.clone(), MSB_SCALES),
    );
    let second_offsets = or(
        shr_lit(and_lit(w1, MSB_TWO), 2),
        shr_lit(and_lit(w2, MSB_OFFSET), 4),
    );

    let byte_of = |word: &TileExpr, i: u32| and_lit(shr_lit(word.clone(), i * 8), 0xff);
    let scales = std::array::from_fn(|g| {
        if g < 4 {
            byte_of(&first_scales, g as u32)
        } else {
            byte_of(&second_scales, g as u32 - 4)
        }
    });
    let offsets = std::array::from_fn(|g| {
        if g < 4 {
            byte_of(&first_offsets, g as u32)
        } else {
            byte_of(&second_offsets, g as u32 - 4)
        }
    });
    (scales, offsets)
}

/// `scales[g] * d * q - offsets[g] * dmin`, shared by Q4K and Q5K.
fn k4_lane(
    args: &BlockDecodeArgs<'_>,
    fields: &BlockFields,
    base: &TileExpr,
    q: &TileExpr,
    with_high_bit: bool,
) -> TileExpr {
    let d = load_scale_f32(args, base, fields.scale, fields.scale_is_f16);
    let dmin = load_scale_f32(
        args,
        base,
        fields.min.expect("K4 carries dmin"),
        fields.scale_is_f16,
    );
    let group = shr_lit(q.clone(), 5);
    let (group_scale, group_min) = k4_group_scale_min(args, base, fields, &group);

    // The 128 quant bytes are indexed by 64-element chunk: group pair `g/2`
    // supplies the byte, group parity picks the nibble.
    let in_group = and_lit(q.clone(), 31);
    let byte_index = add(shl_lit(shr_lit(group.clone(), 1), 5), in_group.clone());
    let byte = load_block_byte(args, base, fields.ql, Some(byte_index));
    let take_high = cmp(TileCompareOp::Ne, and_lit(group.clone(), 1), u32_lit(0));
    let nibble = sel(take_high, shr_lit(byte.clone(), 4), and_lit(byte, 0x0f));

    let quant = if with_high_bit {
        let qh_off = fields.qh.expect("Q5K carries a qh plane");
        let qh_byte = load_block_byte(args, base, qh_off, Some(in_group));
        let bit = and_lit(shr(qh_byte, group), 1);
        or(nibble, shl_lit(bit, 4))
    } else {
        nibble
    };

    let scale = mul(u32_to_f32(group_scale), d);
    let offset = mul(u32_to_f32(group_min), dmin);
    sub(mul(u32_to_f32(quant), scale), offset)
}

fn decode_k4(
    args: &BlockDecodeArgs<'_>,
    fmt: QFmt,
    want: QLayout,
    name: &'static str,
) -> Result<TileExpr> {
    expect_layout(args, want, name)?;
    let fields = block_fields(fmt, want);
    let with_high_bit = matches!(fmt, QFmt::Q5K);
    let (base, q) = block_base_and_q(args, fmt);
    Ok(finish(
        args,
        k4_lane(args, &fields, &base, &q, with_high_bit),
    ))
}

/// `scale * scales_i8[k/16] * (((qh_bits << 4) | ql_nibble) - 32)`.
///
/// Q6K's 128-element chunk addressing: within a chunk, `low_group = (k & 127)
/// >> 5` selects both which half of the 64-byte low plane supplies the byte
/// (`low_group & 1`), which nibble of it (`low_group >> 1`), and which 2-bit
/// field of the high plane (`low_group * 2`).
fn q6k_lane(
    args: &BlockDecodeArgs<'_>,
    fields: &BlockFields,
    base: &TileExpr,
    q: &TileExpr,
) -> TileExpr {
    let (gs, _) = fields.group_scales.expect("Q6K carries group scales");
    let qh_off = fields.qh.expect("Q6K carries a qh plane");
    let d = load_scale_f32(args, base, fields.scale, fields.scale_is_f16);

    let chunk = shr_lit(q.clone(), 7);
    let local = and_lit(q.clone(), 127);
    let hb = and_lit(local.clone(), 31);
    let low_group = shr_lit(local, 5);

    let low_index = add(
        shl_lit(chunk.clone(), 6),
        add(shl_lit(and_lit(low_group.clone(), 1), 5), hb.clone()),
    );
    let low_byte = load_block_byte(args, base, fields.ql, Some(low_index));
    let low4 = and_lit(
        shr(low_byte, shl_lit(shr_lit(low_group.clone(), 1), 2)),
        0x0f,
    );

    let high_index = add(shl_lit(chunk.clone(), 5), hb.clone());
    let high_byte = load_block_byte(args, base, qh_off, Some(high_index));
    let high2 = shl_lit(and_lit(shr(high_byte, shl_lit(low_group.clone(), 1)), 3), 4);

    let quant = or(low4, high2);

    let scale_index = add(
        shl_lit(chunk, 3),
        add(shr_lit(hb, 4), shl_lit(low_group, 1)),
    );
    let scale_byte = load_block_byte(args, base, gs, Some(scale_index));
    // The Q6K group scales are signed i8; a negative one flips the lane.
    let group_scale = signed_byte_f32(scale_byte);

    let centered = sub(u32_to_f32(quant), f32_lit(32.0));
    mul(centered, mul(group_scale, d))
}

fn decode_q6k(args: &BlockDecodeArgs<'_>, want: QLayout, name: &'static str) -> Result<TileExpr> {
    expect_layout(args, want, name)?;
    let fields = block_fields(QFmt::Q6K, want);
    let (base, q) = block_base_and_q(args, QFmt::Q6K);
    Ok(finish(args, q6k_lane(args, &fields, &base, &q)))
}

// ---------------------------------------------------------------------------
// The six programs
// ---------------------------------------------------------------------------

/// Q4K, raw GGUF bytes: f16 `d`, f16 `dmin`, 12 packed group scales, 128
/// nibble bytes.
pub fn decode_q4k_native(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_k4(args, QFmt::Q4K, QLayout::Native, "decode_q4k_native")
}

/// Q4K with `d`/`dmin` widened to f32.
pub fn decode_q4k_f32(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_k4(args, QFmt::Q4K, QLayout::F32Scales, "decode_q4k_f32")
}

/// Q5K, raw GGUF bytes: Q4K plus a 32-byte high-bit plane.
pub fn decode_q5k_native(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_k4(args, QFmt::Q5K, QLayout::Native, "decode_q5k_native")
}

/// Q5K with `d`/`dmin` widened to f32.
pub fn decode_q5k_f32(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_k4(args, QFmt::Q5K, QLayout::F32Scales, "decode_q5k_f32")
}

/// Q6K, raw GGUF bytes: the 210-byte block that is not word-aligned.
pub fn decode_q6k_native(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_q6k(args, QLayout::Native, "decode_q6k_native")
}

/// Q6K with the super-block scale widened to f32; 212 bytes, word-aligned.
pub fn decode_q6k_f32(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_q6k(args, QLayout::F32Scales, "decode_q6k_f32")
}

pub const DECODE_Q4K_NATIVE: BlockProgram = BlockProgram {
    name: "decode_q4k_native",
    emit: decode_q4k_native,
};
pub const DECODE_Q4K_F32: BlockProgram = BlockProgram {
    name: "decode_q4k_f32",
    emit: decode_q4k_f32,
};
pub const DECODE_Q5K_NATIVE: BlockProgram = BlockProgram {
    name: "decode_q5k_native",
    emit: decode_q5k_native,
};
pub const DECODE_Q5K_F32: BlockProgram = BlockProgram {
    name: "decode_q5k_f32",
    emit: decode_q5k_f32,
};
pub const DECODE_Q6K_NATIVE: BlockProgram = BlockProgram {
    name: "decode_q6k_native",
    emit: decode_q6k_native,
};
pub const DECODE_Q6K_F32: BlockProgram = BlockProgram {
    name: "decode_q6k_f32",
    emit: decode_q6k_f32,
};
