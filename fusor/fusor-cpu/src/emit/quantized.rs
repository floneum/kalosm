//! Quantized decode on CPU: a `Source::Quantized` load becomes ordinary
//! `TileExpr`s via the shared `BlockProgram::emit`, so the decode fuses into
//! the consumer's tape rather than materializing.

use fusor_gguf::blocks::BlockDecodeArgs;
use fusor_ir::ir::kernel::{ElementType, QuantizedView, ScalarElement, TileExpr};
use fusor_ir::target::EmitError;

fn f32_ty() -> ElementType {
    ElementType::Scalar(ScalarElement::F32)
}

/// Run the shared decode program for the element at `(k_base, col)`.
pub(crate) fn expand_dequantize(
    src: &QuantizedView,
    k_base: &TileExpr,
    col: &TileExpr,
    mask: &TileExpr,
    fill: &TileExpr,
) -> Result<TileExpr, EmitError> {
    let spec = fusor_gguf::block_spec(src.fmt, src.layout);
    let args = BlockDecodeArgs {
        src: &src.data,
        layout: src.layout,
        k_base: k_base.clone(),
        col: col.clone(),
        mask: mask.clone(),
        fill: fill.clone(),
    };
    let decoded = (spec.decode.emit)(&args).map_err(|e| EmitError::Unsupported(e.to_string()))?;
    if decoded.element() != f32_ty() {
        return Err(EmitError::Unsupported(format!(
            "{} decode returned {:?}, expected a scalar f32",
            spec.decode.name,
            decoded.element()
        )));
    }
    Ok(decoded)
}
