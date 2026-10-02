//! Quantized loads: running a format's `BlockProgram` for a `Source::Quantized`.
//! Formats are data supplied by `fusor-gguf`.

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
    /// The decode program for one `(format, layout)` pair.
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
        // Aligned-window algebra: a lane's consecutive elements differ by a small
        // literal on an aligned base, so their word and scale loads hash-cons to one
        // per window.
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
