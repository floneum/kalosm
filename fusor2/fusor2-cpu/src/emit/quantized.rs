//! Quantized decode on CPU, running the same `BlockProgram` the GPU emitter
//! runs — the decode tables are shared, only the emitter differs.
//!
//! **There is no per-format code in this file.** `Dequantize`, `LaneOf` and
//! `QuantizedDot` are rewritten at *compile* time into ordinary `TileExpr`s by
//! calling W11's `BlockProgram::emit`, and the resulting tree is compiled by
//! the same tape builder as everything else. That is what makes a lazy
//! per-element dequantize inside a fused expression a *fusion alternative*
//! rather than a materialization: the decode nodes simply become part of the
//! consumer's tape.
//!
//! Both activation packings are expressed:
//!
//! * `QAct::F32` decodes a block into registers and `mul_add`s — the shape of
//!   the reference's `process_row_simd_tiled`.
//! * `QAct::Q8Dp4a` first rounds the activations through an int8 grid with a
//!   per-block scale before the same accumulation, which is the numerical
//!   content of the DP4a path.
//!
//! Owned by W10.

use fusor2_ir::dtype::{BlockDecodeArgs, QAct, QLayout};
use fusor2_ir::ir::level2::{
    ElementType, QuantizedView, ScalarElement, TileExpr, TileExprKind, TileLiteral,
};
use fusor2_ir::scalar::{BinOp, UnOp};
use fusor2_ir::target::EmitError;

use fusor2_ir::dtype::NumericContract;

fn f32_ty() -> ElementType {
    ElementType::Scalar(ScalarElement::F32)
}

fn litf(v: f32) -> TileExpr {
    TileExpr::new(
        TileExprKind::Literal(TileLiteral::F32(v.to_bits())),
        f32_ty(),
    )
}

fn bin(op: BinOp, a: TileExpr, b: TileExpr) -> TileExpr {
    TileExpr::new(
        TileExprKind::Binary {
            op,
            left: a,
            right: b,
            numeric: NumericContract::RELAXED,
        },
        f32_ty(),
    )
}

fn un(op: UnOp, x: TileExpr) -> TileExpr {
    TileExpr::new(
        TileExprKind::Unary {
            op,
            value: x,
            numeric: NumericContract::RELAXED,
        },
        f32_ty(),
    )
}

/// Look up the shared decode program and run it, yielding the `lanes`-wide
/// decoded block as an ordinary expression.
///
/// Interning of the per-launch scratch a decode may want (`Q8Scale`, `Q8Pack`,
/// `Q8Sum`, `BlockDequant`, keyed by `(kind, element, depth)`) is the caller's
/// job and happens in [`crate::emit`]'s tile table; nothing here is
/// format-specific.
pub fn expand_dequantize(
    src: &QuantizedView,
    k_base: &TileExpr,
    col: &TileExpr,
    mask: &TileExpr,
    fill: &TileExpr,
    lanes: u32,
) -> Result<TileExpr, EmitError> {
    if fusor2_gguf::BLOCK_SPECS.is_empty() {
        return Err(EmitError::MissingCapability(
            "fusor2-gguf BLOCK_SPECS is empty: no quantized decode program is registered",
        ));
    }
    let spec = fusor2_gguf::block_spec(src.fmt, src.layout);
    let args = BlockDecodeArgs {
        src: &src.data,
        layout: src.layout,
        k_base: k_base.clone(),
        col: col.clone(),
        mask: mask.clone(),
        fill: fill.clone(),
        lanes,
    };
    (spec.decode.emit)(&args).map_err(|e| EmitError::Unsupported(e.to_string()))
}

/// Project lane `lane` out of a decoded block.
///
/// A one-lane decode has no vector to project out of — the program returns
/// the scalar directly — so lane 0 of a scalar is that scalar.
pub fn lane_of(block: TileExpr, lane: u32) -> TileExpr {
    if lane == 0 && block.element() == f32_ty() {
        return block;
    }
    TileExpr::new(TileExprKind::LaneOf { block, lane }, f32_ty())
}

/// Rewrite a fused quantized dot into decode-then-accumulate.
///
/// The block scale is decoded **once** for the whole block (the decode program
/// returns every lane of one block from one expression), which is exactly the
/// property `Dequantize + Dot` cannot have: it would re-decode per lane.
pub fn expand_quantized_dot(
    src: &QuantizedView,
    packing: QAct,
    activations: &[TileExpr],
    k_base: &TileExpr,
    col: &TileExpr,
    mask: &TileExpr,
    fill: &TileExpr,
) -> Result<TileExpr, EmitError> {
    let lanes = activations.len() as u32;
    if lanes == 0 {
        return Ok(litf(0.0));
    }
    let block = expand_dequantize(src, k_base, col, mask, fill, lanes)?;

    let acts: Vec<TileExpr> = match packing {
        QAct::F32 => activations.to_vec(),
        QAct::Q8Dp4a => {
            // Activations meet still-quantized weights through an int8 grid.
            // The scale is `max|a| / 127`, shared by the whole block, so the
            // rounding error is the DP4a path's, not the f32 path's.
            let mut amax = un(UnOp::Abs, activations[0].clone());
            for a in &activations[1..] {
                amax = bin(BinOp::Max, amax, un(UnOp::Abs, a.clone()));
            }
            let scale = bin(
                BinOp::Max,
                bin(BinOp::Div, amax, litf(127.0)),
                litf(f32::MIN_POSITIVE),
            );
            activations
                .iter()
                .map(|a| {
                    let q = TileExpr::new(
                        TileExprKind::Round {
                            mode: fusor2_ir::dtype::RoundMode::HalfAwayFromZero,
                            value: bin(BinOp::Div, a.clone(), scale.clone()),
                        },
                        f32_ty(),
                    );
                    let q = bin(BinOp::Max, bin(BinOp::Min, q, litf(127.0)), litf(-128.0));
                    bin(BinOp::Mul, q, scale.clone())
                })
                .collect()
        }
    };

    let mut acc = bin(BinOp::Mul, acts[0].clone(), lane_of(block.clone(), 0));
    for (i, a) in acts.iter().enumerate().skip(1) {
        acc = bin(
            BinOp::Add,
            acc,
            bin(BinOp::Mul, a.clone(), lane_of(block.clone(), i as u32)),
        );
    }
    Ok(acc)
}

/// Bytes one quantized block occupies, so the tape can address it.
pub fn block_bytes(view: &QuantizedView) -> u32 {
    view.fmt.block_bytes(view.layout)
}

/// Elements one block decodes to.
pub fn block_elements(view: &QuantizedView) -> u32 {
    view.fmt.block_elements()
}

/// Which activation packings a format admits, straight off the shared table.
pub fn activations(view: &QuantizedView) -> &'static [QAct] {
    if fusor2_gguf::BLOCK_SPECS.is_empty() {
        return &[QAct::F32];
    }
    fusor2_gguf::block_spec(view.fmt, view.layout).activation
}

/// Both layouts are legal inputs everywhere; moving between them is the priced
/// `qrepack` rewrite, never a decision frozen at upload.
pub const LEGAL_LAYOUTS: [QLayout; 2] = [QLayout::Native, QLayout::F32Scales];

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::dtype::QFmt;
    use fusor2_ir::ir::level2::{
        BufferAccess, BufferDecl, MemoryLevel, StorageView, TileLayout,
    };
    use std::sync::Arc;

    fn view(fmt: QFmt, layout: QLayout) -> QuantizedView {
        let buffer = Arc::new(BufferDecl {
            binding: 1,
            element: ElementType::Scalar(ScalarElement::U32),
            layout: TileLayout::contiguous(MemoryLevel::Storage, &[1024]),
            access: BufferAccess::Read,
        });
        QuantizedView {
            data: StorageView {
                buffer,
                offset: 0,
                layout: TileLayout::contiguous(MemoryLevel::Storage, &[1024]),
            },
            fmt,
            layout,
            rows: 8,
            cols: 256,
        }
    }

    #[test]
    fn block_geometry_matches_the_shared_table() {
        for fmt in QFmt::ALL {
            for layout in LEGAL_LAYOUTS {
                let v = view(fmt, layout);
                assert_eq!(block_bytes(&v), fmt.block_bytes(layout));
                assert_eq!(block_elements(&v), fmt.block_elements());
            }
        }
    }

    #[test]
    fn a_missing_decode_table_is_an_error_not_a_panic() {
        // W11 has not populated `BLOCK_SPECS` yet; the emitter must report
        // that as a capability gap rather than unwinding through `todo!()`.
        if !fusor2_gguf::BLOCK_SPECS.is_empty() {
            return;
        }
        let v = view(QFmt::Q4_0, QLayout::Native);
        let zero = litf(0.0);
        let t = TileExpr::new(
            TileExprKind::Literal(TileLiteral::Bool(true)),
            ElementType::Scalar(ScalarElement::Bool),
        );
        let err = expand_dequantize(&v, &zero, &zero, &t, &zero, 32).unwrap_err();
        assert!(matches!(err, EmitError::MissingCapability(_)));
    }
}
