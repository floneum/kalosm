//! Host-side repack between the two on-device storage layouts.
//!
//! Both layouts are legal inputs everywhere, and moving between them is a
//! priced rewrite (`qrepack`, amortized over `Persistence::Persistent`).
//! `repack` moves bytes between two layouts the caller names; it chooses
//! nothing.

use fusor2_ir::Result;
use fusor2_ir::dtype::{QFmt, QLayout};
use fusor2_ir::error::Error;

use crate::blocks::{block_fields, qh_width, ql_width};

/// Byte-exact conversion between the two layouts, field by field.
///
/// `Native -> F32Scales` widens each f16 scale (and `min`, where the format
/// carries one) to f32 and copies the remaining planes to their new offsets.
/// `F32Scales -> Native` narrows, losslessly, since the f32 came from an f16.
/// `from == to` is a memcpy.
///
/// Appends to `dst`; the caller may reuse a buffer across weights.
pub fn repack(fmt: QFmt, from: QLayout, to: QLayout, src: &[u8], dst: &mut Vec<u8>) -> Result<()> {
    let src_stride = fmt.block_bytes(from) as usize;
    let dst_stride = fmt.block_bytes(to) as usize;
    if !src.len().is_multiple_of(src_stride) {
        return Err(Error::Shape(format!(
            "{} bytes is not a whole number of {fmt:?}/{from:?} blocks ({src_stride} bytes each)",
            src.len()
        )));
    }
    let blocks = src.len() / src_stride;

    if from == to {
        dst.extend_from_slice(src);
        return Ok(());
    }

    let sf = block_fields(fmt, from);
    let df = block_fields(fmt, to);
    let planes: [(u32, u32, u32); 2] = [
        (sf.ql, df.ql, ql_width(fmt)),
        (
            sf.qh.unwrap_or(0),
            df.qh.unwrap_or(0),
            if sf.qh.is_some() { qh_width(fmt) } else { 0 },
        ),
    ];

    dst.reserve(blocks * dst_stride);
    for b in 0..blocks {
        let s = &src[b * src_stride..(b + 1) * src_stride];
        let start = dst.len();
        dst.resize(start + dst_stride, 0);
        let d = &mut dst[start..start + dst_stride];

        write_scale(
            d,
            df.scale,
            df.scale_is_f16,
            read_scale(s, sf.scale, sf.scale_is_f16),
        );
        if let (Some(smin), Some(dmin)) = (sf.min, df.min) {
            write_scale(
                d,
                dmin,
                df.scale_is_f16,
                read_scale(s, smin, sf.scale_is_f16),
            );
        }
        if let (Some((so, len)), Some((do_, _))) = (sf.group_scales, df.group_scales) {
            copy_plane(s, so, d, do_, len);
        }
        for (so, dof, len) in planes {
            copy_plane(s, so, d, dof, len);
        }
    }
    Ok(())
}

/// Bytes [`repack`] appends for `blocks` input blocks.
pub const fn repack_len(fmt: QFmt, to: QLayout, blocks: usize) -> usize {
    blocks * fmt.block_bytes(to) as usize
}

/// Output length of a [`repack`] whose input is `src_len` bytes, so a caller
/// can size a buffer first. Returns 0 when `src_len` is not a whole number of
/// blocks — [`repack`] rejects that case.
pub const fn repacked_len(fmt: QFmt, from: QLayout, to: QLayout, src_len: usize) -> usize {
    let stride = fmt.block_bytes(from) as usize;
    if !src_len.is_multiple_of(stride) {
        return 0;
    }
    repack_len(fmt, to, src_len / stride)
}

fn read_scale(block: &[u8], offset: u32, is_f16: bool) -> f32 {
    let o = offset as usize;
    if is_f16 {
        half::f16::from_le_bytes([block[o], block[o + 1]]).to_f32()
    } else {
        f32::from_le_bytes([block[o], block[o + 1], block[o + 2], block[o + 3]])
    }
}

fn write_scale(block: &mut [u8], offset: u32, is_f16: bool, value: f32) {
    let o = offset as usize;
    if is_f16 {
        block[o..o + 2].copy_from_slice(&half::f16::from_f32(value).to_le_bytes());
    } else {
        block[o..o + 4].copy_from_slice(&value.to_le_bytes());
    }
}

fn copy_plane(src: &[u8], src_off: u32, dst: &mut [u8], dst_off: u32, len: u32) {
    if len == 0 {
        return;
    }
    let (s, d, n) = (src_off as usize, dst_off as usize, len as usize);
    dst[d..d + n].copy_from_slice(&src[s..s + n]);
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::blocks::cpu_dequantize_block;

    /// Deterministic pseudorandom block bytes. A fixed-seed LCG keeps the
    /// fixtures identical across runs and platforms.
    pub(crate) fn lcg_blocks(fmt: QFmt, layout: QLayout, blocks: usize, seed: u64) -> Vec<u8> {
        let stride = fmt.block_bytes(layout) as usize;
        let fields = block_fields(fmt, layout);
        let mut state = seed;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut out = vec![0u8; blocks * stride];
        for b in 0..blocks {
            let block = &mut out[b * stride..(b + 1) * stride];
            for byte in block.iter_mut() {
                *byte = next() as u8;
            }
            // Scales must round-trip through f16 exactly, so seed them from a
            // real f16 bit pattern with the NaN/inf exponent excluded.
            let mut finite_f16 = || {
                let bits = (next() as u16) & 0x7BFF;
                half::f16::from_bits(bits)
            };
            write_scale(
                block,
                fields.scale,
                fields.scale_is_f16,
                finite_f16().to_f32(),
            );
            if let Some(min) = fields.min {
                write_scale(block, min, fields.scale_is_f16, finite_f16().to_f32());
            }
        }
        out
    }

    #[test]
    fn repack_round_trips_bit_exactly() {
        for fmt in QFmt::ALL {
            let native = lcg_blocks(fmt, QLayout::Native, 4, 0x5eed_0000 + fmt as u64);
            let mut wide = Vec::new();
            repack(fmt, QLayout::Native, QLayout::F32Scales, &native, &mut wide).unwrap();
            assert_eq!(wide.len(), 4 * fmt.block_bytes(QLayout::F32Scales) as usize);
            assert_eq!(wide.len(), repack_len(fmt, QLayout::F32Scales, 4));
            assert_eq!(
                wide.len(),
                repacked_len(fmt, QLayout::Native, QLayout::F32Scales, native.len())
            );

            let mut back = Vec::new();
            repack(fmt, QLayout::F32Scales, QLayout::Native, &wide, &mut back).unwrap();
            assert_eq!(back, native, "{fmt:?} did not round trip bit-exactly");
        }

        // Q6K grows exactly two bytes per block.
        let grew = fusor2_ir::dtype::QFmt::Q6K;
        assert_eq!(
            grew.block_bytes(QLayout::F32Scales) - grew.block_bytes(QLayout::Native),
            2
        );
        let pct = 2.0f64 / 210.0 * 100.0;
        assert!((pct - 0.952).abs() < 0.001, "Q6K grows {pct}%");
    }

    #[test]
    fn same_layout_is_a_memcpy_and_ragged_input_errors() {
        let native = lcg_blocks(QFmt::Q4K, QLayout::Native, 3, 7);
        let mut out = Vec::new();
        repack(
            QFmt::Q4K,
            QLayout::Native,
            QLayout::Native,
            &native,
            &mut out,
        )
        .unwrap();
        assert_eq!(out, native);

        let mut out = Vec::new();
        let err = repack(
            QFmt::Q4K,
            QLayout::Native,
            QLayout::F32Scales,
            &native[..native.len() - 1],
            &mut out,
        )
        .unwrap_err();
        assert!(matches!(err, fusor2_ir::error::Error::Shape(_)));
    }

    #[test]
    fn repack_preserves_dequantized_values() {
        for fmt in QFmt::ALL {
            let elements = fmt.block_elements() as usize;
            let native = lcg_blocks(fmt, QLayout::Native, 4, 0xa11ce + fmt as u64);
            let mut wide = Vec::new();
            repack(fmt, QLayout::Native, QLayout::F32Scales, &native, &mut wide).unwrap();

            let ns = fmt.block_bytes(QLayout::Native) as usize;
            let ws = fmt.block_bytes(QLayout::F32Scales) as usize;
            let mut a = vec![0.0f32; elements];
            let mut b = vec![0.0f32; elements];
            for i in 0..4 {
                cpu_dequantize_block(fmt, QLayout::Native, &native[i * ns..(i + 1) * ns], &mut a);
                cpu_dequantize_block(fmt, QLayout::F32Scales, &wide[i * ws..(i + 1) * ws], &mut b);
                for lane in 0..elements {
                    assert_eq!(
                        a[lane].to_bits(),
                        b[lane].to_bits(),
                        "{fmt:?} block {i} lane {lane}: f16->f32 widening is exact"
                    );
                }
            }
        }
    }
}
