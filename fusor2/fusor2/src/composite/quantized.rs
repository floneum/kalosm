//! `L0::Dequant`'s definitional expansion: a `Restride` over the block stream
//! read as `u32` words, plus a `Map` of unpack bit-arithmetic.
//!
//! A block decode is bit arithmetic over a strided view of a byte buffer.
//! Spelled that way it is two core nodes, and every mechanism that already
//! exists — fusion, the contraction lowering, the cooperative staging fill,
//! the cost model — consumes it with no quantization-aware code. `Dequant` was
//! the one op in the tree with no `defn`, against the discipline
//! [`crate::composite::macro_op`] states; this is that `defn`.
//!
//! **One fact bounds what is reachable, and it is about addresses, not about
//! arithmetic.**
//!
//! The block stream is bound as a rank-1 `U32` leaf, so every field a decode
//! reads has to be a whole `u32` word of it — which makes the block *stride*
//! the gate. [`QLayout::F32Scales`] is universally word-aligned:
//! 20/24/36/148/180/212 bytes. Of the native strides only Q4K's 144 and Q5K's
//! 176 are; Q4_0's 18, Q5_0's 22, Q8_0's 34 and Q6K's 210 each leave two bytes
//! over, so their blocks walk in and out of word phase and a `Restride` cannot
//! address them. Those four keep their `BlockProgram` at `Native`, and
//! `qrepack` — already priced — is the bridge. No byte-addressed load and no
//! format-specific alignment code appears here.
//!
//! An f16 scale is **not** part of that gate, though it used to be believed to
//! be: see [`f16_lane`], which decodes a half out of a word in pure
//! `ScalarExpr` arithmetic. The half-word offset it needs is [`scale_field`],
//! the 2-aligned counterpart of [`field_word`].
//!
//! No format constant is written here. Block strides come from
//! `QFmt::block_bytes`, field offsets from `fusor2_gguf::blocks::block_fields`
//! — the crate that parses the files and is where format knowledge stops.
//!
//! Owned by W13.

use fusor2_autograd::tape::TapeExt;
use fusor2_ir::Result;
use fusor2_ir::autograd::{Tape, Val};
use fusor2_ir::dtype::{Dtype, QFmt, QLayout, Splat};
use fusor2_ir::egraph::Id;
use fusor2_ir::scalar::{BinOp, CmpOp, ScalarExpr, UnOp};
use fusor2_ir::shape::{Dim, StrideSpec};

use crate::composite::{index_leaf, index_run};
use crate::quantized::QMatrix;

/// Words one block occupies, or `None` when its stride is not a whole number
/// of `u32` words.
///
/// **No format knowledge**: it is `bytes % 4`.
fn block_words(fmt: QFmt, layout: QLayout) -> Option<u64> {
    let bytes = u64::from(fmt.block_bytes(layout));
    bytes.is_multiple_of(4).then(|| bytes / 4)
}

/// A byte offset inside a block as a word offset, or `None` if it straddles.
fn field_word(byte_offset: u32) -> Option<u64> {
    byte_offset.is_multiple_of(4).then(|| u64::from(byte_offset) / 4)
}

/// A scale-shaped field as `(word, lane)`: the `u32` word that holds it, and —
/// when the layout stores it as an f16 — which 16-bit lane of that word it is.
/// `None` for the lane means the field *is* the whole word, an f32.
///
/// **No format knowledge**: it is `offset / 4` and `(offset % 4) / 2`. An f16
/// field only has to be 2-aligned, which is why this is not [`field_word`];
/// that difference is the whole reason [`QLayout::Native`] used to be
/// unreachable at the offset level, on top of the arithmetic reason
/// [`f16_lane`] solves.
fn scale_field(byte_offset: u32, is_f16: bool) -> Option<(u64, Option<u32>)> {
    if is_f16 {
        byte_offset
            .is_multiple_of(2)
            .then(|| (u64::from(byte_offset) / 4, Some((byte_offset % 4) / 2)))
    } else {
        field_word(byte_offset).map(|w| (w, None))
    }
}

/// The `[rows, blocks]` prefix every view of the block stream shares: a row is
/// `blocks * bw` words, a block is `bw` words, and `word` is the field's
/// offset inside the block. Both axes read input dim 0 — the block stream is
/// one rank-1 `U32` leaf, and `restride_layout` composes multipliers against
/// its unit stride.
fn block_prefix(rows: u64, blocks: u64, bw: u64, word: u64) -> [StrideSpec; 2] {
    [
        StrideSpec::dim_with(0, Dim::Const(rows), (blocks * bw) as u32),
        StrideSpec::dim_with(0, Dim::Const(blocks), bw as u32).with_offset(Dim::Const(word)),
    ]
}

/// A `U32` literal.
fn u(v: u32) -> ScalarExpr {
    ScalarExpr::lit(Splat::U32(v))
}

/// An `F32` literal.
fn f(v: f32) -> ScalarExpr {
    ScalarExpr::lit(Splat::F32(v))
}

/// `(word >> shift) & mask` — one packed field out of a `u32`.
fn field(word: ScalarExpr, shift: ScalarExpr, mask: u32) -> ScalarExpr {
    ScalarExpr::bin(
        BinOp::BitAnd,
        ScalarExpr::bin(BinOp::Shr, word, shift),
        u(mask),
    )
}

/// `(q - bias) * bitcast<f32>(scale_word)` — the tail every block format
/// shares once its quant is an unsigned integer in a register.
fn scaled(q: ScalarExpr, bias: f32, scale_word: ScalarExpr) -> ScalarExpr {
    ScalarExpr::bin(
        BinOp::Mul,
        ScalarExpr::bin(BinOp::Sub, ScalarExpr::cast(Dtype::F32, q), f(bias)),
        ScalarExpr::bitcast(Dtype::F32, scale_word),
    )
}


/// One IEEE-754 binary16 lane of `word`, as an `f32`, in pure scalar
/// arithmetic.
///
/// This is what the module note used to say did not exist: "f16 scales have no
/// `Map` spelling", which kept every [`QLayout::Native`] row on a
/// `BlockProgram`. The obstacle was never expressiveness — it was reaching for
/// the wrong primitive. `Unpack2x16Float` yields a two-lane *vector*, and
/// picking a lane needs `VecComponent`, which exists only at L2; `ScalarExpr`
/// is scalar by construction and has no projection.
///
/// But a half is just bits, and the decode is arithmetic the vocabulary already
/// has. For a normal half `(-1)^s * 2^(e-15) * (1 + m/1024)`:
///
/// ```text
///   h = (word >> 16*lane) & 0xFFFF
///   s = (h >> 15) & 1        e = (h >> 10) & 0x1F        m = h & 0x3FF
///   value = (1 - 2s) * exp2(e - 15) * (1 + m * (1/1024))
/// ```
///
/// `Shr`/`BitAnd` are the mask-and-shift the algebra rules already read as
/// their `Div`/`Rem` twins, and `Exp2` supplies the exponent — so this is one
/// `Map`, visible to constant folding and to fusion, with no intrinsic and no
/// new node.
///
/// **The `e == 0` branch is not optional, and assuming it was is a bug this
/// nearly shipped with.** The normal formula at `h == 0` yields
/// `2^-15 * (1 + 0)` = 3.05e-5, not zero — and an all-zero block is an ordinary
/// thing for a quantized file to contain. IEEE spells that row differently:
/// `e == 0` means `2^-14 * (m/1024)` with no implicit leading one, which gives
/// exactly `0.0` at `m == 0` and the correct value for every subnormal. So the
/// magnitude is a `select` on `e == 0` over the two rows.
///
/// Checked exhaustively against `f16::from_bits` over all 63,488 finite halves:
/// zero mismatches. Inf and NaN (`e == 31`) are not representable scales and are
/// not special-cased; a file carrying one is already broken.
fn f16_lane(word: ScalarExpr, lane: u32) -> ScalarExpr {
    let half = field(word, u(16 * lane), 0xFFFF);
    let sign = ScalarExpr::bin(BinOp::BitAnd, ScalarExpr::bin(BinOp::Shr, half.clone(), u(15)), u(1));
    let exp = ScalarExpr::bin(BinOp::BitAnd, ScalarExpr::bin(BinOp::Shr, half.clone(), u(10)), u(0x1F));
    let mant = ScalarExpr::bin(BinOp::BitAnd, half, u(0x3FF));

    // (1 - 2s): +1.0 when the sign bit is clear, -1.0 when set.
    let signum = ScalarExpr::bin(
        BinOp::Sub,
        f(1.0),
        ScalarExpr::bin(BinOp::Mul, f(2.0), ScalarExpr::cast(Dtype::F32, sign)),
    );
    let exp_f = ScalarExpr::cast(Dtype::F32, exp.clone());
    let mant_f = ScalarExpr::bin(
        BinOp::Mul,
        ScalarExpr::cast(Dtype::F32, mant),
        f(1.0 / 1024.0),
    );
    // Normal: 2^(e-15) * (1 + m/1024). The bias is subtracted in f32 so it
    // cannot underflow an unsigned integer at e < 15.
    let normal = ScalarExpr::bin(
        BinOp::Mul,
        ScalarExpr::un(UnOp::Exp2, ScalarExpr::bin(BinOp::Sub, exp_f, f(15.0))),
        ScalarExpr::bin(BinOp::Add, f(1.0), mant_f.clone()),
    );
    // Subnormal (and zero): 2^-14 * (m/1024), no implicit leading one.
    let subnormal = ScalarExpr::bin(BinOp::Mul, f(2.0f32.powi(-14)), mant_f);
    let magnitude = ScalarExpr::select(
        ScalarExpr::cmp(CmpOp::Eq, exp, u(0)),
        subnormal,
        normal,
    );
    ScalarExpr::bin(BinOp::Mul, signum, magnitude)
}

/// [`scaled`] against an f16 scale packed in `lane` of `scale_word`.
fn scaled_f16(q: ScalarExpr, bias: f32, scale_word: ScalarExpr, lane: u32) -> ScalarExpr {
    ScalarExpr::bin(
        BinOp::Mul,
        ScalarExpr::bin(BinOp::Sub, ScalarExpr::cast(Dtype::F32, q), f(bias)),
        f16_lane(scale_word, lane),
    )
}

/// The definitional expansion of `Dequant(q)`, or `None` when this
/// `(fmt, layout)` is not spelled as bit arithmetic yet.
///
/// `None` is what makes this additive: the class stays exactly as it is, and
/// the pairs that still need a `BlockProgram` — the four `Native` rows whose
/// block stride is not a whole number of words — keep it.
pub(crate) fn dequant_defn(q: &QMatrix) -> Result<Option<Id>> {
    let (Dim::Const(rows), Dim::Const(cols)) = (q.rows, q.cols) else {
        return Ok(None);
    };
    let (fmt, layout) = (q.fmt, q.layout);
    // No layout guard: an f16 scale decodes through `f16_lane`, and what is
    // left of the old restriction is exactly the alignment test below — which
    // is a property of the block stride, not of the layout name.
    let Some(bw) = block_words(fmt, layout) else {
        return Ok(None);
    };
    let elements = u64::from(fmt.block_elements());
    if elements == 0 || cols % elements != 0 {
        return Ok(None);
    }
    let blocks = cols / elements;

    let graph = q.tensor.graph();
    // An adjoint-minted quantized value has no host bytes, so it has no defn.
    let Some(bytes) = graph.leaf_bytes(q.tensor.id()) else {
        return Ok(None);
    };
    if !bytes.len().is_multiple_of(4) {
        return Ok(None);
    }
    // The same block stream bound a second time as a rank-1 `U32` leaf.
    // `constant_leaf` is content-addressed, so repeated `dequantize` calls on
    // one weight — and two weights holding the same bytes — share one node.
    let words = graph.constant_leaf(
        Dtype::U32,
        &[Dim::Const(bytes.len() as u64 / 4)],
        bytes,
    )?;

    let fields = fusor2_gguf::blocks::block_fields(fmt, layout);
    let (Some((scale_word, scale_lane)), Some(ql_word)) = (
        scale_field(fields.scale, fields.scale_is_f16),
        field_word(fields.ql),
    ) else {
        return Ok(None);
    };
    // The block's own scale, whichever width this layout stores it in. Both
    // arms read the same word operand; only the decode of it differs.
    let scale_of = move |word: ScalarExpr, lane: Option<u32>| match lane {
        Some(l) => f16_lane(word, l),
        None => ScalarExpr::bitcast(Dtype::F32, word),
    };
    // `scaled` / `scaled_f16`, chosen by the layout: the `(q - bias) * d` tail
    // the three 32-element formats share.
    let scaled_by = move |q: ScalarExpr, bias: f32, word: ScalarExpr| match scale_lane {
        Some(l) => scaled_f16(q, bias, word, l),
        None => scaled(q, bias, word),
    };

    let scale_specs = |axes: &[StrideSpec]| -> Vec<StrideSpec> {
        let mut v = block_prefix(rows, blocks, bw, scale_word).to_vec();
        v.extend_from_slice(axes);
        v
    };

    let built = match fmt {
        // 36 bytes = 9 words: an f32 scale at word 0, then 32 quant bytes in
        // words 1..9. Element `e` is byte `e`, so the lane decomposes as
        // `e = w * 4 + b` over `[8 words, 4 bytes]`.
        QFmt::Q8_0 => {
            let inner = [
                StrideSpec::dim_with(0, Dim::Const(8), 1),
                StrideSpec::broadcast(Dim::Const(4)),
            ];
            let mut quant = block_prefix(rows, blocks, bw, ql_word).to_vec();
            quant.extend_from_slice(&inner);
            let scale = scale_specs(&[
                StrideSpec::broadcast(Dim::Const(8)),
                StrideSpec::broadcast(Dim::Const(4)),
            ]);
            // The byte selector as an *index leaf*, not `IndexOf`:
            // `covers_for_substitution` refuses to substitute an
            // `IndexOf`-reading body into any nest whose space is not exactly
            // its own, which is the gate this decode must clear to reach a
            // contraction's staging fill.
            let shift = index_leaf(graph, &[0, 8, 16, 24])?;
            let shift_specs = [
                StrideSpec::broadcast(Dim::Const(rows)),
                StrideSpec::broadcast(Dim::Const(blocks)),
                StrideSpec::broadcast(Dim::Const(8)),
                StrideSpec::dim(0, Dim::Const(4)),
            ];
            // `^ 0x80` then `- 128.0` sign-extends the byte with no integer
            // cast, so the body stays a `U32 -> F32` chain.
            let signed = ScalarExpr::bin(
                BinOp::BitXor,
                field(ScalarExpr::arg(0, Dtype::U32), ScalarExpr::arg(2, Dtype::U32), 0xFF),
                u(0x80),
            );
            let body = scaled_by(signed, 128.0, ScalarExpr::arg(1, Dtype::U32));
            graph.build(|t| {
                let quant = t.restride(&quant, words)?;
                let scale = t.restride(&scale, words)?;
                let shift = t.restride(&shift_specs, shift)?;
                t.map(body, &[quant, scale, shift])
            })?
        }

        // 20 bytes = 5 words: an f32 scale at word 0, then 16 nibble-packed
        // bytes in words 1..5. Element `e = nib * 16 + w * 4 + b`, so the low
        // nibble of byte `i` is element `i` and the high nibble is `i + 16`.
        QFmt::Q4_0 => {
            let inner = [
                StrideSpec::broadcast(Dim::Const(2)),
                StrideSpec::dim_with(0, Dim::Const(4), 1),
                StrideSpec::broadcast(Dim::Const(4)),
            ];
            let mut quant = block_prefix(rows, blocks, bw, ql_word).to_vec();
            quant.extend_from_slice(&inner);
            let scale = scale_specs(&[
                StrideSpec::broadcast(Dim::Const(2)),
                StrideSpec::broadcast(Dim::Const(4)),
                StrideSpec::broadcast(Dim::Const(4)),
            ]);
            // `8 * b + 4 * nib`, indexed `nib * 4 + b`.
            let shift = index_leaf(graph, &NIBBLE_SHIFTS)?;
            let shift_specs = [
                StrideSpec::broadcast(Dim::Const(rows)),
                StrideSpec::broadcast(Dim::Const(blocks)),
                StrideSpec::dim_with(0, Dim::Const(2), 4),
                StrideSpec::broadcast(Dim::Const(4)),
                StrideSpec::dim(0, Dim::Const(4)),
            ];
            let nibble = field(
                ScalarExpr::arg(0, Dtype::U32),
                ScalarExpr::arg(2, Dtype::U32),
                0xF,
            );
            let body = scaled_by(nibble, 8.0, ScalarExpr::arg(1, Dtype::U32));
            graph.build(|t| {
                let quant = t.restride(&quant, words)?;
                let scale = t.restride(&scale, words)?;
                let shift = t.restride(&shift_specs, shift)?;
                t.map(body, &[quant, scale, shift])
            })?
        }

        // 24 bytes = 6 words: an f32 scale at word 0, a 32-bit high-bit plane
        // at word 1, then 16 nibble-packed bytes in words 2..6. The fifth bit
        // of element `e` is bit `e` of the plane.
        QFmt::Q5_0 => {
            let Some(qh_word) = fields.qh.and_then(field_word) else {
                return Ok(None);
            };
            let inner = [
                StrideSpec::broadcast(Dim::Const(2)),
                StrideSpec::dim_with(0, Dim::Const(4), 1),
                StrideSpec::broadcast(Dim::Const(4)),
            ];
            let mut quant = block_prefix(rows, blocks, bw, ql_word).to_vec();
            quant.extend_from_slice(&inner);
            let broadcast_inner = [
                StrideSpec::broadcast(Dim::Const(2)),
                StrideSpec::broadcast(Dim::Const(4)),
                StrideSpec::broadcast(Dim::Const(4)),
            ];
            let mut high = block_prefix(rows, blocks, bw, qh_word).to_vec();
            high.extend_from_slice(&broadcast_inner);
            let scale = scale_specs(&broadcast_inner);
            let shift = index_leaf(graph, &NIBBLE_SHIFTS)?;
            let shift_specs = [
                StrideSpec::broadcast(Dim::Const(rows)),
                StrideSpec::broadcast(Dim::Const(blocks)),
                StrideSpec::dim_with(0, Dim::Const(2), 4),
                StrideSpec::broadcast(Dim::Const(4)),
                StrideSpec::dim(0, Dim::Const(4)),
            ];
            // The bit index *is* the element index, `nib * 16 + w * 4 + b`.
            let bit_index = index_run(graph, 0, 32)?;
            let bit_specs = [
                StrideSpec::broadcast(Dim::Const(rows)),
                StrideSpec::broadcast(Dim::Const(blocks)),
                StrideSpec::dim_with(0, Dim::Const(2), 16),
                StrideSpec::dim_with(0, Dim::Const(4), 4),
                StrideSpec::dim(0, Dim::Const(4)),
            ];
            let nibble = field(
                ScalarExpr::arg(0, Dtype::U32),
                ScalarExpr::arg(3, Dtype::U32),
                0xF,
            );
            let bit = field(
                ScalarExpr::arg(1, Dtype::U32),
                ScalarExpr::arg(4, Dtype::U32),
                1,
            );
            let quant_value = ScalarExpr::bin(
                BinOp::Add,
                nibble,
                ScalarExpr::bin(BinOp::Mul, bit, u(16)),
            );
            let body = scaled_by(quant_value, 16.0, ScalarExpr::arg(2, Dtype::U32));
            graph.build(|t| {
                let quant = t.restride(&quant, words)?;
                let high = t.restride(&high, words)?;
                let scale = t.restride(&scale, words)?;
                let shift = t.restride(&shift_specs, shift)?;
                let bit_index = t.restride(&bit_specs, bit_index)?;
                t.map(body, &[quant, high, scale, shift, bit_index])
            })?
        }

        // 148 bytes = 37 words: an f32 `d` at word 0, an f32 `dmin` at word 1,
        // the twelve packed six-bit group scale/min bytes in words 2..5, then
        // 128 nibble-packed quant bytes in words 5..37.
        //
        // Element `e = gp * 64 + parity * 32 + w * 4 + b`, so the group index
        // is `g = gp * 2 + parity` and the quant byte is `gp * 32 + w * 4 + b`
        // — one byte serving group `2 * gp` in its low nibble and `2 * gp + 1`
        // in its high one, which is exactly `decode_k::k4_lane`'s
        // `byte_index`. The reference's two branches, `lane = g & 3` and
        // `high = g >= 4`, are then functions of `(gp, parity)` alone, so they
        // become two eight-entry index leaves — the same device the nibble
        // shift already uses, and for the same reason: an `IndexOf`-reading
        // body is refused by `covers_for_substitution`.
        //
        // Q5K is 180 bytes = 45 words: the identical field set with a 32-byte
        // high-bit plane inserted ahead of the quants. Element `e`'s fifth bit
        // is bit `g` of plane byte `in_group = w * 4 + b`, so its shift inside
        // the word is `8 * b + g` — the plane word depends on `w` alone, and
        // the shift is one more index leaf over `(gp, parity, b)`. Nothing else
        // about the block differs, which is why it shares this arm rather than
        // copying it.
        QFmt::Q4K | QFmt::Q5K => {
            let (Some((min_word, min_lane)), Some(gs_word)) = (
                fields
                    .min
                    .and_then(|at| scale_field(at, fields.scale_is_f16)),
                fields.group_scales.map(|(at, _)| at).and_then(field_word),
            ) else {
                return Ok(None);
            };
            let fifth_bit = matches!(fmt, QFmt::Q5K);
            let qh_word = fields.qh.and_then(field_word);
            if fifth_bit && qh_word.is_none() {
                return Ok(None);
            }
            // Every operand is the same rank-6 `[rows, blocks, 4, 2, 8, 4]`.
            let elem_bcast = || {
                [
                    StrideSpec::broadcast(Dim::Const(4)),
                    StrideSpec::broadcast(Dim::Const(2)),
                    StrideSpec::broadcast(Dim::Const(8)),
                    StrideSpec::broadcast(Dim::Const(4)),
                ]
            };
            let word_at = |w: u64| {
                let mut v = block_prefix(rows, blocks, bw, w).to_vec();
                v.extend_from_slice(&elem_bcast());
                v
            };
            // `gp` steps a whole 32-byte chunk (8 words); `w` steps one word.
            let mut quant_specs = block_prefix(rows, blocks, bw, ql_word).to_vec();
            quant_specs.extend_from_slice(&[
                StrideSpec::dim_with(0, Dim::Const(4), 8),
                StrideSpec::broadcast(Dim::Const(2)),
                StrideSpec::dim_with(0, Dim::Const(8), 1),
                StrideSpec::broadcast(Dim::Const(4)),
            ]);
            // `8 * b + 4 * parity`: where the nibble sits inside its word.
            let nib = index_leaf(graph, &NIBBLE_SHIFTS)?;
            let nib_specs = [
                StrideSpec::broadcast(Dim::Const(rows)),
                StrideSpec::broadcast(Dim::Const(blocks)),
                StrideSpec::broadcast(Dim::Const(4)),
                StrideSpec::dim_with(0, Dim::Const(2), 4),
                StrideSpec::broadcast(Dim::Const(8)),
                StrideSpec::dim(0, Dim::Const(4)),
            ];
            // `8 * (g & 3)` and `g >= 4`, tabulated over `(gp, parity)`.
            let lane = index_leaf(graph, &[0, 8, 16, 24, 0, 8, 16, 24])?;
            let high = index_leaf(graph, &[0, 0, 0, 0, 1, 1, 1, 1])?;
            let group_specs = [
                StrideSpec::broadcast(Dim::Const(rows)),
                StrideSpec::broadcast(Dim::Const(blocks)),
                StrideSpec::dim_with(0, Dim::Const(4), 2),
                StrideSpec::dim(0, Dim::Const(2)),
                StrideSpec::broadcast(Dim::Const(8)),
                StrideSpec::broadcast(Dim::Const(4)),
            ];

            // The fifth-bit plane and its shift, appended after the nine
            // operands Q4K needs so the shared body's arg numbering is stable.
            let mut extra: Vec<(Vec<StrideSpec>, Id)> = Vec::new();
            if let Some(qh_word) = qh_word.filter(|_| fifth_bit) {
                // Plane byte `w * 4 + b` is byte `b` of plane word `w`.
                let mut plane = block_prefix(rows, blocks, bw, qh_word).to_vec();
                plane.extend_from_slice(&[
                    StrideSpec::broadcast(Dim::Const(4)),
                    StrideSpec::broadcast(Dim::Const(2)),
                    StrideSpec::dim_with(0, Dim::Const(8), 1),
                    StrideSpec::broadcast(Dim::Const(4)),
                ]);
                extra.push((plane, words));
                // `8 * b + g` for `g = 2 * gp + parity`, tabulated over
                // `(gp, parity, b)` — an index leaf for the same reason the
                // nibble shift is one.
                let table: Vec<u32> = (0..4 * 2 * 4)
                    .map(|i| {
                        let (gp, parity, b) = (i / 8, (i / 4) % 2, i % 4);
                        8 * b + 2 * gp + parity
                    })
                    .collect();
                let bit_specs = vec![
                    StrideSpec::broadcast(Dim::Const(rows)),
                    StrideSpec::broadcast(Dim::Const(blocks)),
                    StrideSpec::dim_with(0, Dim::Const(4), 8),
                    StrideSpec::dim_with(0, Dim::Const(2), 4),
                    StrideSpec::broadcast(Dim::Const(8)),
                    StrideSpec::dim(0, Dim::Const(4)),
                ];
                extra.push((bit_specs, index_leaf(graph, &table)?));
            }

            let arg = |i: u32| ScalarExpr::arg(i, Dtype::U32);
            let byte = |w: u32| field(arg(w), arg(7), 0xFF);
            let (scale_byte, min_byte, extra_byte) = (byte(2), byte(3), byte(4));
            let pick = |low: ScalarExpr, hi: ScalarExpr| {
                ScalarExpr::select(ScalarExpr::cmp(CmpOp::Ne, arg(8), u(0)), hi, low)
            };
            // Op-for-op `decode_k::k4_group_scale_min`.
            let gs = pick(
                ScalarExpr::bin(BinOp::BitAnd, scale_byte.clone(), u(0x3F)),
                ScalarExpr::bin(
                    BinOp::BitOr,
                    ScalarExpr::bin(BinOp::BitAnd, extra_byte.clone(), u(0x0F)),
                    ScalarExpr::bin(
                        BinOp::Shr,
                        ScalarExpr::bin(BinOp::BitAnd, scale_byte, u(0xC0)),
                        u(2),
                    ),
                ),
            );
            let gm = pick(
                ScalarExpr::bin(BinOp::BitAnd, min_byte.clone(), u(0x3F)),
                ScalarExpr::bin(
                    BinOp::BitOr,
                    ScalarExpr::bin(
                        BinOp::BitAnd,
                        ScalarExpr::bin(BinOp::Shr, extra_byte, u(4)),
                        u(0x0F),
                    ),
                    ScalarExpr::bin(
                        BinOp::Shr,
                        ScalarExpr::bin(BinOp::BitAnd, min_byte, u(0xC0)),
                        u(2),
                    ),
                ),
            );
            let nibble = field(arg(5), arg(6), 0xF);
            // `k4_lane`'s `or(nibble, shl_lit(bit, 4))`, op for op.
            let quant = if fifth_bit {
                ScalarExpr::bin(
                    BinOp::BitOr,
                    nibble,
                    ScalarExpr::bin(BinOp::Shl, field(arg(9), arg(10), 1), u(4)),
                )
            } else {
                nibble
            };
            // At `Native` both scales are f16 lanes of the block's *first*
            // word — `d` in lane 0, `dmin` in lane 1 — so operands 0 and 1 are
            // the same restride and only the lane differs.
            let d = scale_of(arg(0), scale_lane);
            let dmin = scale_of(arg(1), min_lane);
            // `k4_lane`'s exact operation order — `q * (gs * d) - (gm * dmin)`,
            // not a reassociated equivalent — so this is bit-identical to
            // `cpu_dequantize_block` rather than merely close.
            let body = ScalarExpr::bin(
                BinOp::Sub,
                ScalarExpr::bin(
                    BinOp::Mul,
                    ScalarExpr::cast(Dtype::F32, quant),
                    ScalarExpr::bin(BinOp::Mul, ScalarExpr::cast(Dtype::F32, gs), d),
                ),
                ScalarExpr::bin(BinOp::Mul, ScalarExpr::cast(Dtype::F32, gm), dmin),
            );
            graph.build(|t| {
                let mut ops = vec![
                    t.restride(&word_at(scale_word), words)?,
                    t.restride(&word_at(min_word), words)?,
                    t.restride(&word_at(gs_word), words)?,
                    t.restride(&word_at(gs_word + 1), words)?,
                    t.restride(&word_at(gs_word + 2), words)?,
                    t.restride(&quant_specs, words)?,
                    t.restride(&nib_specs, nib)?,
                    t.restride(&group_specs, lane)?,
                    t.restride(&group_specs, high)?,
                ];
                for (specs, id) in &extra {
                    ops.push(t.restride(specs, *id)?);
                }
                t.map(body, &ops)
            })?
        }

        // 212 bytes = 53 words: 128 low-plane bytes in words 0..32, the 64
        // high-bit bytes in words 32..48, sixteen *signed* i8 group scales in
        // words 48..52, then the f32 super-block scale last, at word 52.
        //
        // Q6K is the one format whose scale sits at the end, which is exactly
        // why its native block is 210 bytes and not word-aligned. That is not
        // a problem to solve here: a `Map` only ever sees `F32Scales` (f16
        // scales have no lane projection at L0), and `F32Scales` widens the
        // trailing scale to four bytes, making the block 53 whole words. The
        // 210-byte stride is `qrepack`'s side of the bridge, and `qrepack` is
        // already priced.
        //
        // The reference walks `e = chunk * 128 + group * 32 + hb` with two
        // terms that are *not* affine in `(chunk, group, hb)`: `group & 1`
        // picks the 32-byte half of the low plane, and `hb >> 4` picks which
        // of the two group scales in that half applies. Splitting the two
        // indices at exactly those bits — `group = gh * 2 + gl` and
        // `hb = wh * 16 + wl * 4 + b` — makes every address a plain stride
        // over the rank-6 nest `[chunk 2, gh 2, gl 2, wh 2, wl 4, b 4]`,
        // whose 256 lanes flatten back to `e` in order:
        //
        //   ql word = 16 * chunk + 8 * gl + 4 * wh + wl
        //   qh word =  8 * chunk           + 4 * wh + wl
        //   gs word =  2 * chunk + gh    (byte `8 * chunk + 4 * gh + 2 * gl + wh`,
        //                                 and `2 * gl + wh < 4` never carries)
        //
        // so no byte-addressed load and no divmod appear anywhere.
        QFmt::Q6K => {
            let (Some(qh_word), Some(gs_word)) = (
                fields.qh.and_then(field_word),
                fields.group_scales.map(|(at, _)| at).and_then(field_word),
            ) else {
                return Ok(None);
            };
            /// Extents of `[chunk, gh, gl, wh, wl, b]`.
            const AXES: [u64; 6] = [2, 2, 2, 2, 4, 4];
            // A stride of 0 is a broadcast: that axis does not move this
            // operand.
            let elem = |strides: [u32; 6]| -> [StrideSpec; 6] {
                std::array::from_fn(|i| {
                    if strides[i] == 0 {
                        StrideSpec::broadcast(Dim::Const(AXES[i]))
                    } else {
                        StrideSpec::dim_with(0, Dim::Const(AXES[i]), strides[i])
                    }
                })
            };
            let word_at = |w: u64, strides: [u32; 6]| {
                let mut v = block_prefix(rows, blocks, bw, w).to_vec();
                v.extend_from_slice(&elem(strides));
                v
            };
            // An index leaf is one small buffer shared by every block, so the
            // two block axes broadcast over it.
            let table_at = |strides: [u32; 6]| {
                let mut v = vec![
                    StrideSpec::broadcast(Dim::Const(rows)),
                    StrideSpec::broadcast(Dim::Const(blocks)),
                ];
                v.extend_from_slice(&elem(strides));
                v
            };

            // `8 * (2 * gl + wh)`: which byte of the group-scale word this
            // lane's signed scale is.
            let gs_shift = index_leaf(graph, &[0, 8, 16, 24])?;
            // `8 * b + 4 * gh`: where the low nibble sits inside its word.
            let nib_shift = index_leaf(graph, &NIBBLE_SHIFTS)?;
            // `8 * b + 4 * gh + 2 * gl`: where the two high bits sit inside
            // theirs — the byte selector and `2 * group` in one table.
            let high_shift = index_leaf(
                graph,
                &[
                    0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30,
                ],
            )?;

            let arg = |i: u32| ScalarExpr::arg(i, Dtype::U32);
            let low4 = field(arg(2), arg(5), 0x0F);
            let high2 = field(arg(3), arg(6), 0x03);
            let quant = ScalarExpr::bin(
                BinOp::BitOr,
                ScalarExpr::bin(BinOp::Shl, high2, u(4)),
                low4,
            );
            // The group scale is signed. `^ 0x80` then `- 128.0` sign-extends
            // the byte with no integer cast, the same way Q8_0's payload does.
            let group_scale = ScalarExpr::bin(
                BinOp::Sub,
                ScalarExpr::cast(
                    Dtype::F32,
                    ScalarExpr::bin(BinOp::BitXor, field(arg(1), arg(4), 0xFF), u(0x80)),
                ),
                f(128.0),
            );
            let d = scale_of(arg(0), scale_lane);
            // `d * group_scale * q`'s exact association, as
            // `cpu_dequantize_block` writes it, so this is bit-identical to
            // the oracle rather than merely close.
            let body = ScalarExpr::bin(
                BinOp::Mul,
                ScalarExpr::bin(BinOp::Mul, d, group_scale),
                ScalarExpr::bin(BinOp::Sub, ScalarExpr::cast(Dtype::F32, quant), f(32.0)),
            );
            graph.build(|t| {
                let ops = [
                    t.restride(&word_at(scale_word, [0, 0, 0, 0, 0, 0]), words)?,
                    t.restride(&word_at(gs_word, [2, 1, 0, 0, 0, 0]), words)?,
                    t.restride(&word_at(ql_word, [16, 0, 8, 4, 1, 0]), words)?,
                    t.restride(&word_at(qh_word, [8, 0, 0, 4, 1, 0]), words)?,
                    t.restride(&table_at([0, 0, 2, 1, 0, 0]), gs_shift)?,
                    t.restride(&table_at([0, 4, 0, 0, 0, 1]), nib_shift)?,
                    t.restride(&table_at([0, 8, 4, 0, 0, 1]), high_shift)?,
                ];
                t.map(body, &ops)
            })?
        }
    };

    // The `Map` is a block-shaped nest; the value is `[rows, cols]`.
    let flat: Val = graph.build(|t| t.reshape(built, &[Dim::Const(rows), Dim::Const(cols)]))?;
    Ok(Some(flat))
}

/// `8 * b + 4 * nib` for `nib in 0..2`, `b in 0..4`: where the nibble of
/// element `nib * 16 + w * 4 + b` sits inside its word. Shared by every
/// nibble-packed 32-element format.
const NIBBLE_SHIFTS: [u32; 8] = [0, 8, 16, 24, 4, 12, 20, 28];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;
    use crate::session::{Device, Session};
    use half::f16;

    /// [`f16_lane`] against `f16::from_bits`, for **every** finite half, in
    /// both lanes, through the real evaluator.
    ///
    /// This is the claim the whole `Native` expansion rests on, so it is
    /// checked exhaustively rather than on a sample: the decode is 63,488
    /// separate pieces of arithmetic, and the subnormal row — which the normal
    /// formula gets wrong at every one of its 2,048 inputs, `0.0` included —
    /// is only 3% of them. A random-scale test would miss it about 97% of the
    /// time.
    ///
    /// `e == 31` is Inf/NaN: not a representable scale, and NaN is unequal to
    /// itself, so those 2,048 patterns are excluded rather than asserted on.
    #[test]
    fn f16_lane_decodes_every_finite_half_exactly() {
        let session = Session::new(Device::cpu().unwrap()).unwrap();
        let graph = Graph::new(&session);
        let graph = graph.handle();
        let bits: Vec<u16> = (0..=u16::MAX).filter(|b| (b >> 10) & 0x1F != 31).collect();
        assert_eq!(bits.len(), 63_488);
        for lane in [0u32, 1] {
            // The half under test in `lane`, the other lane left at zero.
            let words: Vec<u8> = bits
                .iter()
                .flat_map(|b| (u32::from(*b) << (16 * lane)).to_le_bytes())
                .collect();
            let leaf = graph
                .constant_leaf(Dtype::U32, &[Dim::Const(bits.len() as u64)], words)
                .unwrap();
            let body = f16_lane(ScalarExpr::arg(0, Dtype::U32), lane);
            let id = graph.build(|t| t.map(body, &[leaf])).unwrap();
            let got = graph.tensor(id).to_vec_f32().unwrap();
            assert_eq!(got.len(), bits.len());
            for (g, b) in got.iter().zip(&bits) {
                // Exact: a half is exactly representable in f32, and the
                // arithmetic here is a power of two times a sum of exact
                // dyadic rationals.
                assert_eq!(*g, f16::from_bits(*b).to_f32(), "lane {lane} bits {b:#06x}");
            }
        }
    }
}
