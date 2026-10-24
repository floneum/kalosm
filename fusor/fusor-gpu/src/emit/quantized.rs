//! Quantized loads use the same decode expressions available to compiler passes.
use fusor_ir::ir::kernel::{Addr, QuantizedView, ScalarElement, TileExpr};
use fusor_ir::target::EmitError;
use fusor_tile::build::Kernel;
use naga::{Block, Expression, Handle};

use super::Emitter;

impl Emitter<'_> {
    pub(crate) fn decode_one(
        &mut self,
        out: &mut Block,
        src: &QuantizedView,
        row: &TileExpr,
        col: &TileExpr,
    ) -> Result<Handle<Expression>, EmitError> {
        let b = Kernel::new();
        let decoded = fusor_gguf::blocks::decode_load(
            src,
            &Addr::Linear(b.add(row.clone(), col.clone())),
            b.bool(true),
            b.f32(0.),
        )
        .map_err(|e| EmitError::Unsupported(format!("quantized decode: {e}")))?;
        if decoded.element() != ScalarElement::F32.element() {
            return Err(EmitError::Unsupported(
                "quantized decode must produce scalar f32".into(),
            ));
        }
        self.expr(&decoded, out)
    }
}
