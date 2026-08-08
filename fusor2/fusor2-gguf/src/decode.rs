//! Block decode programs for the 32-element formats, in both storage layouts.
//!
//! A decode emits an L2 snippet. Native-layout f16 scales are read through
//! `Unpack2x16Float`, so no `ScalarElement::F16` appears in an emitted term
//! and `SHADER_F16` is not required.
//!
//! `BlockDecodeArgs` carries no row extent: the flat element index a program
//! decodes is `k_base + col`, with the row stride folded into `col` (typically
//! `row_index * elements_per_row`) and `k_base` the element offset inside the
//! row. Block base and intra-block index come from that flat index alone.
//!
//! `src` is a plain `u32` storage view. Native blocks are not word-aligned
//! (18, 22, 34 and 210 bytes), so every field read goes through
//! [`load_block_byte`], which loads the containing word and shifts the byte
//! out.

use fusor2_ir::Result;
use fusor2_ir::dtype::{NumericContract, QFmt, QLayout};
use fusor2_ir::ir::level2::{
    Addr, ElementType, ScalarElement, Source, TileBinaryOp, TileCompareOp, TileExpr, TileExprKind,
    TileLiteral, TileUnaryOp,
};

use crate::blocks::{BlockDecodeArgs, BlockFields, BlockProgram, block_fields};

pub(crate) const U32: ElementType = ElementType::Scalar(ScalarElement::U32);
pub(crate) const I32: ElementType = ElementType::Scalar(ScalarElement::I32);
pub(crate) const F32: ElementType = ElementType::Scalar(ScalarElement::F32);
pub(crate) const BOOL: ElementType = ElementType::Scalar(ScalarElement::Bool);

/// Decode arithmetic is ordinary f32: reassociation and contraction are both
/// permitted, and the accumulator is f32-wide.
const RELAXED: NumericContract = NumericContract::RELAXED;

pub(crate) fn u32_lit(v: u32) -> TileExpr {
    TileExpr::new(TileExprKind::Literal(TileLiteral::U32(v)), U32)
}

pub(crate) fn f32_lit(v: f32) -> TileExpr {
    TileExpr::new(TileExprKind::Literal(TileLiteral::F32(v.to_bits())), F32)
}

/// `op(a, b)` typed as `a`.
pub(crate) fn bin(op: TileBinaryOp, left: TileExpr, right: TileExpr) -> TileExpr {
    let ty = left.element();
    TileExpr::new(
        TileExprKind::Binary {
            op,
            left,
            right,
            numeric: RELAXED,
        },
        ty,
    )
}

pub(crate) fn add(a: TileExpr, b: TileExpr) -> TileExpr {
    bin(TileBinaryOp::Add, a, b)
}
pub(crate) fn sub(a: TileExpr, b: TileExpr) -> TileExpr {
    bin(TileBinaryOp::Sub, a, b)
}
pub(crate) fn mul(a: TileExpr, b: TileExpr) -> TileExpr {
    bin(TileBinaryOp::Mul, a, b)
}
pub(crate) fn and_lit(a: TileExpr, m: u32) -> TileExpr {
    bin(TileBinaryOp::BitAnd, a, u32_lit(m))
}
pub(crate) fn or(a: TileExpr, b: TileExpr) -> TileExpr {
    bin(TileBinaryOp::BitOr, a, b)
}
pub(crate) fn shr(a: TileExpr, b: TileExpr) -> TileExpr {
    bin(TileBinaryOp::Shr, a, b)
}
pub(crate) fn shr_lit(a: TileExpr, n: u32) -> TileExpr {
    shr(a, u32_lit(n))
}
pub(crate) fn shl(a: TileExpr, b: TileExpr) -> TileExpr {
    bin(TileBinaryOp::Shl, a, b)
}
pub(crate) fn shl_lit(a: TileExpr, n: u32) -> TileExpr {
    shl(a, u32_lit(n))
}
pub(crate) fn mul_lit(a: TileExpr, n: u32) -> TileExpr {
    mul(a, u32_lit(n))
}
pub(crate) fn add_lit(a: TileExpr, n: u32) -> TileExpr {
    if n == 0 { a } else { add(a, u32_lit(n)) }
}

pub(crate) fn cmp(op: TileCompareOp, left: TileExpr, right: TileExpr) -> TileExpr {
    TileExpr::new(TileExprKind::Compare { op, left, right }, BOOL)
}

pub(crate) fn sel(condition: TileExpr, accept: TileExpr, reject: TileExpr) -> TileExpr {
    let ty = accept.element();
    TileExpr::new(
        TileExprKind::Select {
            condition,
            accept,
            reject,
        },
        ty,
    )
}

pub(crate) fn cast(value: TileExpr, to: ElementType) -> TileExpr {
    TileExpr::new(TileExprKind::Cast { value, to }, to)
}

pub(crate) fn bitcast(value: TileExpr, to: ElementType) -> TileExpr {
    TileExpr::new(TileExprKind::Bitcast { value, to }, to)
}

/// Numeric widening of a u32 quant to f32.
pub(crate) fn u32_to_f32(v: TileExpr) -> TileExpr {
    cast(v, F32)
}

/// `bitcast_i32(byte << 24) >> 24` then widen: an arithmetic shift recovers
/// the sign of a byte that arrived in the low bits of a u32.
pub(crate) fn signed_byte_f32(byte: TileExpr) -> TileExpr {
    let shifted = shl_lit(byte, 24);
    let signed = bitcast(shifted, I32);
    let sar = bin(TileBinaryOp::Shr, signed, u32_lit(24));
    cast(sar, F32)
}

/// Byte base of the block holding the addressed element, and that element's
/// index inside the block.
pub(crate) fn block_base_and_q(args: &BlockDecodeArgs<'_>, fmt: QFmt) -> (TileExpr, TileExpr) {
    let elements = fmt.block_elements();
    debug_assert!(elements.is_power_of_two());
    let bytes = fmt.block_bytes(args.layout);
    let flat = add(args.k_base.clone(), args.col.clone());
    let block = shr_lit(flat.clone(), elements.trailing_zeros());
    let base = mul_lit(block, bytes);
    let q = and_lit(flat, elements - 1);
    (base, q)
}

/// Load the u32 word at `word_index` out of the quantized storage view.
pub(crate) fn load_word(args: &BlockDecodeArgs<'_>, word_index: TileExpr) -> TileExpr {
    TileExpr::new(
        TileExprKind::Load {
            src: Source::Storage(args.src.clone()),
            addr: Box::new(Addr::Linear(word_index)),
            mask: args.mask.clone(),
            fill: u32_lit(0),
        },
        U32,
    )
}

/// Load one byte of a block: `(base + byte_offset + dynamic) >> 2` selects the
/// word, `((base + byte_offset + dynamic) & 3) * 8` the shift. Native blocks
/// are not word-aligned, so the containing word is the only thing the storage
/// view can address.
pub(crate) fn load_block_byte(
    args: &BlockDecodeArgs<'_>,
    base: &TileExpr,
    byte_offset: u32,
    dynamic_index: Option<TileExpr>,
) -> TileExpr {
    let local = match dynamic_index {
        Some(i) => add_lit(i, byte_offset),
        None => u32_lit(byte_offset),
    };
    let global = add(base.clone(), local);
    let word = load_word(args, shr_lit(global.clone(), 2));
    let lane = and_lit(global, 3);
    let shift = shl_lit(lane, 3);
    and_lit(shr(word, shift), 0xff)
}

/// Read a scale-shaped field as f32.
///
/// The f16 path loads the containing word, applies `Unpack2x16Float` and picks
/// the half dynamically: a Native block base is not word-aligned, so which
/// half holds the scale depends on the block index. No f16 element type is
/// constructed.
pub(crate) fn load_scale_f32(
    args: &BlockDecodeArgs<'_>,
    base: &TileExpr,
    offset: u32,
    scale_is_f16: bool,
) -> TileExpr {
    let global = add_lit(base.clone(), offset);
    let word = load_word(args, shr_lit(global.clone(), 2));
    if !scale_is_f16 {
        return bitcast(word, F32);
    }
    let pair = TileExpr::new(
        TileExprKind::Unary {
            op: TileUnaryOp::Unpack2x16Float,
            value: word,
            numeric: RELAXED,
        },
        ElementType::Vector {
            scalar: ScalarElement::F32,
            lanes: 2,
        },
    );
    let low = TileExpr::new(
        TileExprKind::VecComponent {
            vector: pair.clone(),
            component: 0,
        },
        F32,
    );
    let high = TileExpr::new(
        TileExprKind::VecComponent {
            vector: pair,
            component: 1,
        },
        F32,
    );
    let half = and_lit(shr_lit(global, 1), 1);
    sel(cmp(TileCompareOp::Ne, half, u32_lit(0)), high, low)
}

/// Apply the load's `mask`/`fill` to the decoded element.
///
/// The result is a scalar, never a one-lane vector: WGSL has vec2/vec3/vec4
/// only.
pub(crate) fn finish(args: &BlockDecodeArgs<'_>, value: TileExpr) -> TileExpr {
    if args.mask.is_constant_true() {
        return value;
    }
    let fill = if args.fill.element() == F32 {
        args.fill.clone()
    } else {
        f32_lit(0.0)
    };
    sel(args.mask.clone(), value, fill)
}

/// Reject a program invoked with the wrong `layout`, so a table row can never
/// silently decode the other layout's byte offsets.
pub(crate) fn expect_layout(
    args: &BlockDecodeArgs<'_>,
    want: QLayout,
    name: &'static str,
) -> Result<()> {
    if args.layout == want {
        Ok(())
    } else {
        Err(fusor2_ir::error::Error::Dtype(format!(
            "{name} decodes {want:?} blocks, got {:?}",
            args.layout
        )))
    }
}

/// `q & 15` addresses the byte; `q < 16` takes the low nibble and `q >= 16`
/// the high nibble: the nibble order is split-half, not adjacent pairs.
pub(crate) fn nibble_q4(
    args: &BlockDecodeArgs<'_>,
    base: &TileExpr,
    fields: &BlockFields,
    q: &TileExpr,
) -> TileExpr {
    let q_local = and_lit(q.clone(), 15);
    let byte = load_block_byte(args, base, fields.ql, Some(q_local));
    let low = and_lit(byte.clone(), 0x0f);
    let high = shr_lit(byte, 4);
    let take_high = cmp(TileCompareOp::Ge, q.clone(), u32_lit(16));
    sel(take_high, high, low)
}

/// Q5_0's fifth bit lives in the 32-bit `qh` plane: bit `q_local` for the low
/// half, bit `q_local + 16` for the high half.
pub(crate) fn nibble_q5(
    args: &BlockDecodeArgs<'_>,
    base: &TileExpr,
    fields: &BlockFields,
    q: &TileExpr,
) -> TileExpr {
    let qh_off = fields.qh.expect("Q5_0 carries a qh plane");
    let q_local = and_lit(q.clone(), 15);
    let take_high = cmp(TileCompareOp::Ge, q.clone(), u32_lit(16));
    let low4 = nibble_q4(args, base, fields, q);

    // Read the qh plane a byte at a time: the block base need not be
    // word-aligned, so a whole-word load would straddle the boundary.
    let bit_index = sel(take_high, add_lit(q_local.clone(), 16), q_local);
    let qh_byte = load_block_byte(args, base, qh_off, Some(shr_lit(bit_index.clone(), 3)));
    let bit = and_lit(shr(qh_byte, and_lit(bit_index, 7)), 1);
    or(low4, shl_lit(bit, 4))
}

/// `(q - center) * scale`.
fn centered(quant: TileExpr, center: f32, scale: TileExpr) -> TileExpr {
    mul(sub(u32_to_f32(quant), f32_lit(center)), scale)
}

fn decode_affine(
    args: &BlockDecodeArgs<'_>,
    fmt: QFmt,
    want: QLayout,
    name: &'static str,
) -> Result<TileExpr> {
    expect_layout(args, want, name)?;
    let fields = block_fields(fmt, want);
    let (base, q) = block_base_and_q(args, fmt);
    let scale = load_scale_f32(args, &base, fields.scale, fields.scale_is_f16);
    let value = match fmt {
        QFmt::Q4_0 => centered(nibble_q4(args, &base, &fields, &q), 8.0, scale),
        QFmt::Q5_0 => centered(nibble_q5(args, &base, &fields, &q), 16.0, scale),
        QFmt::Q8_0 => {
            let byte = load_block_byte(args, &base, fields.ql, Some(q));
            mul(signed_byte_f32(byte), scale)
        }
        other => {
            return Err(fusor2_ir::error::Error::Dtype(format!(
                "{other:?} is not an affine-family format"
            )));
        }
    };
    Ok(finish(args, value))
}

/// Q4_0, raw GGUF bytes: f16 scale, 16 nibble-pair bytes, 18 bytes total.
pub fn decode_q4_0_native(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_affine(args, QFmt::Q4_0, QLayout::Native, "decode_q4_0_native")
}

/// Q4_0 with the scale widened to f32; 20 bytes, word-aligned.
pub fn decode_q4_0_f32(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_affine(args, QFmt::Q4_0, QLayout::F32Scales, "decode_q4_0_f32")
}

/// Q5_0, raw GGUF bytes: f16 scale, 4-byte high-bit plane, 16 nibble bytes.
pub fn decode_q5_0_native(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_affine(args, QFmt::Q5_0, QLayout::Native, "decode_q5_0_native")
}

/// Q5_0 with the scale widened to f32.
pub fn decode_q5_0_f32(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_affine(args, QFmt::Q5_0, QLayout::F32Scales, "decode_q5_0_f32")
}

/// Q8_0, raw GGUF bytes: f16 scale then 32 signed bytes.
pub fn decode_q8_0_native(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_affine(args, QFmt::Q8_0, QLayout::Native, "decode_q8_0_native")
}

/// Q8_0 with the scale widened to f32.
pub fn decode_q8_0_f32(args: &BlockDecodeArgs<'_>) -> Result<TileExpr> {
    decode_affine(args, QFmt::Q8_0, QLayout::F32Scales, "decode_q8_0_f32")
}

pub const DECODE_Q4_0_NATIVE: BlockProgram = BlockProgram {
    name: "decode_q4_0_native",
    emit: decode_q4_0_native,
};
pub const DECODE_Q4_0_F32: BlockProgram = BlockProgram {
    name: "decode_q4_0_f32",
    emit: decode_q4_0_f32,
};
pub const DECODE_Q5_0_NATIVE: BlockProgram = BlockProgram {
    name: "decode_q5_0_native",
    emit: decode_q5_0_native,
};
pub const DECODE_Q5_0_F32: BlockProgram = BlockProgram {
    name: "decode_q5_0_f32",
    emit: decode_q5_0_f32,
};
pub const DECODE_Q8_0_NATIVE: BlockProgram = BlockProgram {
    name: "decode_q8_0_native",
    emit: decode_q8_0_native,
};
pub const DECODE_Q8_0_F32: BlockProgram = BlockProgram {
    name: "decode_q8_0_f32",
    emit: decode_q8_0_f32,
};

// A pure-Rust evaluator over the node set these programs build, used to check
// them against the scalar decoder.

#[cfg(test)]
pub(crate) mod interp {
    use super::*;
    use fusor2_ir::ir::level2::{BufferAccess, BufferDecl, MemoryLevel, StorageView, TileLayout};
    use std::sync::Arc;

    /// A value the evaluator can hold. Vectors are always f32 here, which is
    /// the only vector these programs build.
    #[derive(Clone, Debug, PartialEq)]
    pub enum V {
        U(u32),
        I(i32),
        F(f32),
        B(bool),
        Vf(Vec<f32>),
    }

    impl V {
        /// Only `VecComponent` reads this; `Unpack2x16Float` is the one vector
        /// a decode program builds.
        fn f32s(&self) -> &[f32] {
            match self {
                V::Vf(v) => v,
                other => panic!("expected an f32 vector, got {other:?}"),
            }
        }
        fn u(&self) -> u32 {
            match self {
                V::U(v) => *v,
                other => panic!("expected u32, got {other:?}"),
            }
        }
        pub fn f(&self) -> f32 {
            match self {
                V::F(v) => *v,
                other => panic!("expected f32, got {other:?}"),
            }
        }
        fn b(&self) -> bool {
            match self {
                V::B(v) => *v,
                other => panic!("expected bool, got {other:?}"),
            }
        }
    }

    /// A `u32` storage view over `words` words, which is how a quantized
    /// matrix is bound.
    pub fn storage_view(words: usize) -> StorageView {
        let layout = TileLayout::contiguous(MemoryLevel::Storage, &[words as u32]);
        let buffer = Arc::new(BufferDecl {
            binding: 1,
            element: U32,
            layout: layout.clone(),
            access: BufferAccess::Read,
        });
        StorageView {
            buffer,
            offset: 0,
            layout,
        }
    }

    /// Pad `bytes` up to a whole number of words and reinterpret little-endian.
    pub fn to_words(bytes: &[u8]) -> Vec<u32> {
        let mut padded = bytes.to_vec();
        while !padded.len().is_multiple_of(4) {
            padded.push(0);
        }
        padded
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    pub fn eval(expr: &TileExpr, words: &[u32]) -> V {
        match expr.kind() {
            TileExprKind::Literal(lit) => match lit {
                TileLiteral::U32(v) => V::U(*v),
                TileLiteral::I32(v) => V::I(*v),
                TileLiteral::F32(bits) => V::F(f32::from_bits(*bits)),
                TileLiteral::Bool(v) => V::B(*v),
                other => panic!("literal {other:?} is not built by a decode program"),
            },
            TileExprKind::Load {
                addr, mask, fill, ..
            } => {
                if !eval(mask, words).b() {
                    return eval(fill, words);
                }
                let Addr::Linear(index) = addr.as_ref() else {
                    panic!("decode programs only emit linear addresses");
                };
                let i = eval(index, words).u() as usize;
                V::U(words.get(i).copied().unwrap_or(0))
            }
            TileExprKind::Unary { op, value, .. } => {
                assert_eq!(*op, TileUnaryOp::Unpack2x16Float, "unexpected unary {op:?}");
                let w = eval(value, words).u();
                V::Vf(vec![
                    half::f16::from_bits(w as u16).to_f32(),
                    half::f16::from_bits((w >> 16) as u16).to_f32(),
                ])
            }
            TileExprKind::Binary {
                op, left, right, ..
            } => {
                let l = eval(left, words);
                let r = eval(right, words);
                binary(*op, l, r)
            }
            TileExprKind::Compare { op, left, right } => {
                let (l, r) = (eval(left, words), eval(right, words));
                let ord = match (&l, &r) {
                    (V::U(a), V::U(b)) => a.partial_cmp(b),
                    (V::I(a), V::I(b)) => a.partial_cmp(b),
                    (V::F(a), V::F(b)) => a.partial_cmp(b),
                    _ => panic!("mismatched compare operands {l:?} {r:?}"),
                };
                let Some(ord) = ord else { return V::B(false) };
                V::B(match op {
                    TileCompareOp::Lt => ord.is_lt(),
                    TileCompareOp::Le => ord.is_le(),
                    TileCompareOp::Gt => ord.is_gt(),
                    TileCompareOp::Ge => ord.is_ge(),
                    TileCompareOp::Eq => ord.is_eq(),
                    TileCompareOp::Ne => !ord.is_eq(),
                })
            }
            TileExprKind::Cast { value, to } => {
                let v = eval(value, words);
                match (v, *to) {
                    (V::U(a), ElementType::Scalar(ScalarElement::F32)) => V::F(a as f32),
                    (V::I(a), ElementType::Scalar(ScalarElement::F32)) => V::F(a as f32),
                    (V::F(a), ElementType::Scalar(ScalarElement::F32)) => V::F(a),
                    (v, to) => panic!("unsupported cast {v:?} -> {to:?}"),
                }
            }
            TileExprKind::Bitcast { value, to } => {
                let v = eval(value, words);
                match (v, *to) {
                    (V::U(a), ElementType::Scalar(ScalarElement::I32)) => V::I(a as i32),
                    (V::U(a), ElementType::Scalar(ScalarElement::F32)) => V::F(f32::from_bits(a)),
                    (V::I(a), ElementType::Scalar(ScalarElement::U32)) => V::U(a as u32),
                    (v, to) => panic!("unsupported bitcast {v:?} -> {to:?}"),
                }
            }
            TileExprKind::Select {
                condition,
                accept,
                reject,
            } => {
                if eval(condition, words).b() {
                    eval(accept, words)
                } else {
                    eval(reject, words)
                }
            }
            TileExprKind::Vec { parts, .. } => {
                V::Vf(parts.iter().map(|p| eval(p, words).f()).collect())
            }
            TileExprKind::VecComponent { vector, component } => {
                V::F(eval(vector, words).f32s()[*component as usize])
            }
            other => panic!("node {other:?} is not built by a decode program"),
        }
    }

    fn binary(op: TileBinaryOp, l: V, r: V) -> V {
        use fusor2_ir::scalar::BinOp as B;
        match (&l, &r) {
            (V::U(a), V::U(b)) => {
                let (a, b) = (*a, *b);
                V::U(match op {
                    B::Add => a.wrapping_add(b),
                    B::Sub => a.wrapping_sub(b),
                    B::Mul => a.wrapping_mul(b),
                    B::Div => a / b,
                    B::Rem => a % b,
                    B::Min => a.min(b),
                    B::Max => a.max(b),
                    B::BitAnd => a & b,
                    B::BitOr => a | b,
                    B::BitXor => a ^ b,
                    B::Shr => a >> b,
                    B::Shl => a << b,
                    other => panic!("unsupported u32 binary {other:?}"),
                })
            }
            // An i32 left operand makes `Shr` arithmetic, which is what
            // recovers a signed byte's sign.
            (V::I(a), V::U(b)) if matches!(op, B::Shr | B::Shl) => {
                let (a, b) = (*a, *b);
                V::I(if matches!(op, B::Shr) { a >> b } else { a << b })
            }
            (V::I(a), V::I(b)) => {
                let (a, b) = (*a, *b);
                V::I(match op {
                    B::Add => a.wrapping_add(b),
                    B::Sub => a.wrapping_sub(b),
                    B::Mul => a.wrapping_mul(b),
                    B::BitAnd => a & b,
                    B::BitOr => a | b,
                    other => panic!("unsupported i32 binary {other:?}"),
                })
            }
            (V::F(a), V::F(b)) => {
                let (a, b) = (*a, *b);
                V::F(match op {
                    B::Add => a + b,
                    B::Sub => a - b,
                    B::Mul => a * b,
                    B::Div => a / b,
                    B::Min => a.min(b),
                    B::Max => a.max(b),
                    B::Pow => a.powf(b),
                    other => panic!("unsupported f32 binary {other:?}"),
                })
            }
            _ => panic!("mismatched binary operands {l:?} {r:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::interp::{eval, storage_view, to_words};
    use super::*;
    use crate::blocks::{BLOCK_SPECS, cpu_dequantize_block};
    use crate::repack::tests::lcg_blocks;
    use fusor2_ir::ir::level2::TileLiteral;

    fn mask(v: bool) -> TileExpr {
        TileExpr::new(TileExprKind::Literal(TileLiteral::Bool(v)), BOOL)
    }

    /// Every one of the twelve rows must agree with the scalar decoder on
    /// every element of every block.
    #[test]
    fn decode_programs_agree_with_cpu_dequantize() {
        for spec in BLOCK_SPECS {
            let fmt = spec.fmt;
            let layout = spec.layout;
            let stride = fmt.block_bytes(layout) as usize;
            let elements = fmt.block_elements() as usize;
            let raw = lcg_blocks(fmt, layout, 4, 0xd0_0000 + fmt as u64 * 2 + layout as u64);
            let words = to_words(&raw);
            let view = storage_view(words.len());

            let mut expected = vec![0.0f32; elements];
            for block in 0..4usize {
                cpu_dequantize_block(
                    fmt,
                    layout,
                    &raw[block * stride..(block + 1) * stride],
                    &mut expected,
                );
                for element in 0..elements {
                    let args = BlockDecodeArgs {
                        src: &view,
                        layout,
                        k_base: u32_lit((block * elements + element) as u32),
                        col: u32_lit(0),
                        mask: mask(true),
                        fill: f32_lit(0.0),
                    };
                    let program = (spec.decode.emit)(&args).unwrap();
                    assert_eq!(
                        program.element(),
                        F32,
                        "{} must return a scalar f32",
                        spec.decode.name
                    );
                    let want = expected[element];
                    let have = eval(&program, &words).f();
                    assert!(
                        (want - have).abs() <= 1e-6 * want.abs().max(1.0),
                        "{} block {block} element {element}: expected {want}, got {have}",
                        spec.decode.name,
                    );
                }
            }
        }
    }

    #[test]
    fn decode_respects_mask_and_fill() {
        for spec in BLOCK_SPECS {
            let raw = lcg_blocks(spec.fmt, spec.layout, 1, 99);
            let words = to_words(&raw);
            let view = storage_view(words.len());
            let args = BlockDecodeArgs {
                src: &view,
                layout: spec.layout,
                k_base: u32_lit(0),
                col: u32_lit(0),
                mask: mask(false),
                fill: f32_lit(-7.5),
            };
            let program = (spec.decode.emit)(&args).unwrap();
            let got = eval(&program, &words);
            assert_eq!(
                got.f(),
                -7.5,
                "{} ignored its mask",
                spec.decode.name
            );
        }
    }

    /// A row invoked with the other layout's byte offsets is an error, not a
    /// silently wrong decode.
    #[test]
    fn a_program_refuses_the_other_layout() {
        let raw = lcg_blocks(QFmt::Q4_0, QLayout::Native, 1, 1);
        let words = to_words(&raw);
        let view = storage_view(words.len());
        let args = BlockDecodeArgs {
            src: &view,
            layout: QLayout::F32Scales,
            k_base: u32_lit(0),
            col: u32_lit(0),
            mask: mask(true),
            fill: f32_lit(0.0),
        };
        assert!(decode_q4_0_native(&args).is_err());
        assert!(decode_q4_0_f32(&args).is_ok());
    }

    /// No emitted node may carry an f16 element type; a native-layout scale is
    /// read through `Unpack2x16Float` instead.
    #[test]
    fn no_f16_element_appears_in_an_emitted_program() {
        fn walk(e: &TileExpr, seen: &mut usize) {
            assert!(
                !e.element().uses_f16(),
                "f16 element at node {:?}",
                e.kind()
            );
            *seen += 1;
            match e.kind() {
                TileExprKind::Load {
                    addr, mask, fill, ..
                } => {
                    let Addr::Linear(i) = addr.as_ref() else {
                        unreachable!()
                    };
                    walk(i, seen);
                    walk(mask, seen);
                    walk(fill, seen);
                }
                TileExprKind::Unary { value, .. }
                | TileExprKind::Cast { value, .. }
                | TileExprKind::Bitcast { value, .. } => walk(value, seen),
                TileExprKind::Binary { left, right, .. }
                | TileExprKind::Compare { left, right, .. } => {
                    walk(left, seen);
                    walk(right, seen);
                }
                TileExprKind::Select {
                    condition,
                    accept,
                    reject,
                } => {
                    walk(condition, seen);
                    walk(accept, seen);
                    walk(reject, seen);
                }
                TileExprKind::Vec { parts, .. } => parts.iter().for_each(|p| walk(p, seen)),
                TileExprKind::VecComponent { vector, .. } => walk(vector, seen),
                _ => {}
            }
        }

        let view = storage_view(1024);
        for spec in BLOCK_SPECS {
            let args = BlockDecodeArgs {
                src: &view,
                layout: spec.layout,
                k_base: u32_lit(0),
                col: u32_lit(0),
                mask: mask(true),
                fill: f32_lit(0.0),
            };
            let program = (spec.decode.emit)(&args).unwrap();
            let mut seen = 0;
            walk(&program, &mut seen);
            assert!(
                seen > 8,
                "{} emitted a suspiciously small program",
                spec.decode.name
            );
        }
    }
}
