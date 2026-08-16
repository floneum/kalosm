//! Quantized loads: running a format's `BlockProgram` for a `Source::Quantized`.
//!
//! Formats are **data**: the per-format decode is the `BlockProgram` that
//! `fusor-gguf` supplies as a `BlockEmitFn`, invoked here with a
//! `BlockDecodeArgs`. Adding Q4_1 is a table row over there, not a kernel here.

use fusor_gguf::blocks::{BlockDecodeArgs, BlockProgram};
use fusor_ir::ir::kernel::{
    ElementType, QuantizedView, ScalarElement, TileExpr, TileExprKind, TileLiteral,
};
use fusor_ir::target::EmitError;
use naga::{Block, Expression, Handle};

use super::Emitter;

fn f32_element() -> ElementType {
    ElementType::Scalar(ScalarElement::F32)
}

impl Emitter<'_> {
    /// The decode program for one `(format, layout)` pair. Formats are data:
    /// the row lives in `fusor-gguf`, never here.
    fn block_program(&self, view: &QuantizedView) -> BlockProgram {
        fusor_gguf::block_spec(view.fmt, view.layout).decode
    }

    /// One decoded element, addressed by Kernel expressions.
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
        // Aligned-window algebra over the decode's index arithmetic. A
        // lane's consecutive elements differ only in a small literal added
        // to an aligned base, so after the rewrite their word and scale
        // subexpressions are *structurally equal* and the emitter's memo
        // shares the loads — one word read per window, one scale decode per
        // group, with the block format never appearing in the rule.
        let decoded = fusor_ir::ir::kernel::simplify_index(&decoded);
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
