//! Quantized decode on CPU, running the same `BlockProgram` the GPU emitter
//! runs — the decode tables are shared, only the emitter differs.
//!
//! A `Source::Quantized` load is rewritten at *compile* time into ordinary
//! `TileExpr`s by calling `BlockProgram::emit`, and the resulting tree is
//! compiled by the same tape builder as everything else. That is what makes a
//! lazy per-element dequantize inside a fused expression a *fusion alternative*
//! rather than a materialization: the decode nodes simply become part of the
//! consumer's tape.

use fusor2_gguf::blocks::BlockDecodeArgs;
use fusor2_ir::ir::kernel::{ElementType, QuantizedView, ScalarElement, TileExpr};
use fusor2_ir::target::EmitError;

fn f32_ty() -> ElementType {
    ElementType::Scalar(ScalarElement::F32)
}

/// Look up the shared decode program and run it for the single element at
/// `(k_base, col)`, yielding an ordinary scalar expression.
pub(crate) fn expand_dequantize(
    src: &QuantizedView,
    k_base: &TileExpr,
    col: &TileExpr,
    mask: &TileExpr,
    fill: &TileExpr,
) -> Result<TileExpr, EmitError> {
    let spec = fusor2_gguf::block_spec(src.fmt, src.layout);
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
