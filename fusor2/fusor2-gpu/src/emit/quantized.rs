//! Quantized loads: running a format's `BlockProgram` for a `Source::Quantized`.
//!
//! Formats are **data**: the per-format decode is the `BlockProgram` that
//! `fusor2-gguf` supplies as a `BlockEmitFn`, invoked here with a
//! `BlockDecodeArgs`. Adding Q4_1 is a table row over there, not a kernel here.
//!
//! Owned by W8.

use fusor2_gguf::blocks::{BlockDecodeArgs, BlockProgram};
use fusor2_ir::ir::level2::{
    ElementType, QuantizedView, ScalarElement, TileExpr, TileExprKind, TileLiteral,
};
use fusor2_ir::target::EmitError;
use naga::{Block, Expression, Handle};

use super::Emitter;

fn f32_element() -> ElementType {
    ElementType::Scalar(ScalarElement::F32)
}

impl Emitter<'_> {
    /// The decode program for one `(format, layout)` pair. Formats are data:
    /// the row lives in `fusor2-gguf`, never here.
    fn block_program(&self, view: &QuantizedView) -> BlockProgram {
        fusor2_gguf::block_spec(view.fmt, view.layout).decode
    }

    /// One decoded element, addressed by L2 expressions.
    pub(crate) fn decode_one(
        &mut self,
        out: &mut Block,
        src: &QuantizedView,
        row: &TileExpr,
        col: &TileExpr,
    ) -> Result<Handle<Expression>, EmitError> {
        let program = self.block_program(src);
        let args = BlockDecodeArgs {
            src: &src.data,
            layout: src.layout,
            k_base: row.clone(),
            col: col.clone(),
            mask: TileExpr::new(
                TileExprKind::Literal(TileLiteral::Bool(true)),
                ElementType::Scalar(ScalarElement::Bool),
            ),
            fill: TileExpr::new(
                TileExprKind::Literal(TileLiteral::F32(0f32.to_bits())),
                f32_element(),
            ),
        };
        let decoded = (program.emit)(&args)
            .map_err(|e| EmitError::Unsupported(format!("{} decode: {e}", program.name)))?;
        if decoded.element() != f32_element() {
            return Err(EmitError::Unsupported(format!(
                "{} decode returned {:?}, expected a scalar f32",
                program.name,
                decoded.element()
            )));
        }
        self.expr(&decoded, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::emit_module;
    use crate::emit::testkit::{self, *};
    use fusor2_ir::dtype::{QFmt, QLayout};
    use fusor2_ir::ir::level2::{Addr, KernelIr, Source, Stmt, StorageView};

    const LANES: u32 = 8;
    const COLS: u32 = 4;

    fn qview(data: &fusor2_ir::ir::level2::Buffer) -> QuantizedView {
        QuantizedView {
            data: StorageView {
                buffer: data.clone(),
                offset: 0,
                layout: fusor2_ir::ir::level2::TileLayout::contiguous(
                    fusor2_ir::ir::level2::MemoryLevel::Storage,
                    &[COLS * LANES],
                ),
            },
            fmt: QFmt::Q8_0,
            layout: QLayout::Native,
        }
    }

    /// One real `Q8_0` native block: an f16 scale then 32 signed bytes, which
    /// is exactly what `fusor2_gguf::block_spec(Q8_0, Native).decode` reads.
    ///
    /// The scale is `0.125` — a power of two, so it is exact in binary16 and
    /// exact again as an f32 product — and quant `i` is `i + 1`, so element
    /// `i` decodes to `0.125 * (i + 1)` with no rounding anywhere.
    fn weight_bytes() -> Vec<u8> {
        let mut out = vec![0u8; (COLS * LANES) as usize * 4];
        // binary16 0.125: sign 0, exponent 12 (= -3 + 15), mantissa 0.
        out[0..2].copy_from_slice(&0x3000u16.to_le_bytes());
        for i in 0..32u8 {
            out[2 + i as usize] = i + 1;
        }
        out
    }

    #[test]
    fn masked_quantized_load_lowers() {
        let uni = testkit::buffer(0, u32e(), 4, false);
        let data = testkit::buffer(1, u32e(), COLS * LANES, false);
        let dst = testkit::buffer(2, f32e(), COLS, true);
        let dv = view(&dst, &[COLS]);
        let value = TileExpr::new(
            TileExprKind::Load {
                src: Source::Quantized(qview(&data)),
                addr: Box::new(Addr::Rc2 {
                    row: lit_u32(0),
                    col: lane(),
                }),
                mask: TileExpr::new(
                    TileExprKind::Compare {
                        op: fusor2_ir::scalar::CmpOp::Lt,
                        left: lane(),
                        right: lit_u32(2),
                    },
                    boole(),
                ),
                fill: lit_f32(-1.0),
            },
            f32_element(),
        );
        let ir = KernelIr {
            buffers: vec![uni, data, dst],
            grid: [1, 1, 1],
            block: COLS,
            body: vec![Stmt::Store {
                dst: dv,
                addr: Addr::Linear(lane()),
                value,
                mask: tru(),
            }],
            byte_arena: None,
            name: "masked_q",
        };
        let emitted = emit_module(&ir, &caps(false, true), &no_plan()).expect("emit");
        let Some(gpu) = gpu() else {
            eprintln!("no wgpu adapter; skipping the numeric half");
            return;
        };
        let inputs = vec![uniforms(), weight_bytes(), bytes_of(&[0.0; COLS as usize])];
        let out = f32s(&run_emitted(&gpu, &ir, emitted, &inputs, 2));
        assert!((out[0] - 0.125).abs() < 1e-6, "got {out:?}");
        assert!((out[1] - 0.25).abs() < 1e-6, "got {out:?}");
        assert_eq!(out[2], -1.0);
        assert_eq!(out[3], -1.0);
    }
}
