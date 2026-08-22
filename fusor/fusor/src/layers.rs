//! Parameterized layers. Each is a thin struct over a few `Tensor` parameters
//! and a `forward`; none of them owns a kernel.

pub(crate) mod conv;
pub(crate) mod embedding;
pub(crate) mod layer_norm;
pub(crate) mod linear;
pub(crate) mod rms_norm;

pub use conv::ConvNd;
pub use embedding::Embedding;
pub use layer_norm::{LayerNorm, LayerNormNd};
pub use linear::Linear;
pub use rms_norm::RmsNorm;

use fusor_gguf::VarBuilder;
use fusor_ir::dtype::Dtype;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::shape::Dim;

use crate::graph::GraphRef;
use crate::tensor::Tensor;
use crate::tensor::typed::Element;
use crate::{Error, Result};

/// A loaded parameter as a const-rank value of the layer's element type.
///
/// [`load_dense`] always hands back `F32` — that is what the reference's
/// `dequantize()` does for every parameter — so a `Linear<f16>` needs one cast
/// on the way in. Writing it here means the five layer types do not each
/// repeat it, and it means `Linear::<f16>::load` reads an f32 checkpoint,
/// which is the case that actually occurs.
pub(crate) fn as_typed<const R: usize, T: Element>(
    t: Tensor,
    what: &str,
) -> Result<crate::Tensor<R, T>> {
    let t = if t.dtype() == T::DTYPE {
        t
    } else {
        t.cast(T::DTYPE)?
    };
    crate::Tensor::<R, T>::try_from_dyn(t).map_err(|e| Error::Shape(format!("{what}: {e}")))
}

/// One GGUF tensor as a dense `F32` value in `graph`.
///
/// `fusor_gguf` reverses a tensor's extents at read, so `raw.shape` is
/// already row-major: a `[out, in]` weight is `[out, in]` here, matching the
/// reference's `weight.shape()[0] == out_features`.
///
/// A block-quantized entry becomes a `Leaf(Quantized)` plus one `Logical::Dequant`
/// — the decode is a device-side block program, never a host loop. `F16` and
/// `BF16` entries are cast, which is what the reference's `dequantize()` does
/// for every parameter it hands a layer.
pub(crate) fn load_dense(vb: &VarBuilder, graph: &GraphRef, name: &str) -> Result<Tensor> {
    let raw = vb.get_raw(name)?;
    let shape: Vec<Dim> = raw.shape.iter().map(|d| Dim::Const(*d)).collect();

    let Dtype::Q(fmt) = raw.fmt else {
        let dense = Tensor::from_slice(graph, raw.fmt, &shape, &raw.bytes)?;
        return match raw.fmt {
            Dtype::F32 => Ok(dense),
            _ => dense.cast(Dtype::F32),
        };
    };

    // A quantized leaf is `[rows, cols]`: the block stream runs along the
    // inner extent, so that is the one the format's block size has to divide.
    let [rows, cols] = match shape.as_slice() {
        [cols] => [Dim::Const(1), *cols],
        [rows, cols] => [*rows, *cols],
        other => {
            return Err(Error::Shape(format!(
                "{name} has shape {other:?}; a block-quantized weight is rank 1 or 2"
            )));
        }
    };
    let elements = fmt.block_elements() as u64;
    if let Dim::Const(c) = cols
        && c % elements != 0
    {
        return Err(Error::Shape(format!(
            "{name} is {fmt:?}, whose inner extent must be a multiple of {elements}, got {c}"
        )));
    }
    if let (Dim::Const(r), Dim::Const(c)) = (rows, cols) {
        let want = r * (c / elements) * fmt.block_bytes(raw.layout) as u64;
        if raw.bytes.len() as u64 != want {
            return Err(Error::Shape(format!(
                "{name} is {fmt:?}/{:?} [{r}, {c}], which is {want} bytes of blocks, got {}",
                raw.layout,
                raw.bytes.len()
            )));
        }
    }
    let leaf = Tensor::emit(
        graph,
        Logical::Leaf(LeafKind::Quantized {
            name: graph.fresh_buffer_id(),
            fmt,
            layout: raw.layout,
            shape: [rows, cols].into_iter().collect(),
        }),
    )?;
    graph.set_leaf_bytes(leaf.id(), raw.bytes.to_vec());
    Tensor::emit(
        graph,
        Logical::Dequant {
            fmt,
            layout: raw.layout,
            x: leaf.id(),
        },
    )
}

/// [`load_dense`], or `None` when the key is absent. A missing key is the
/// only thing swallowed: a present-but-unreadable entry still errors.
pub(crate) fn load_optional(
    vb: &VarBuilder,
    graph: &GraphRef,
    name: &str,
) -> Result<Option<Tensor>> {
    if !vb.contains_key(name) {
        return Ok(None);
    }
    load_dense(vb, graph, name).map(Some)
}

/// A normalization weight or bias as a rank-1 value.
///
/// GGUF writers disagree about whether a norm vector is `[n]` or `[1, n]`;
/// the reference squeezes the degenerate axis rather than refusing, so this
/// does too.
pub(crate) fn as_vector(t: Tensor, name: &str) -> Result<Tensor> {
    match t.shape().as_slice() {
        [_] => Ok(t),
        [a, _] if a.known_eq(Dim::Const(1)) => t.squeeze(0),
        [_, b] if b.known_eq(Dim::Const(1)) => t.squeeze(1),
        other => Err(Error::Shape(format!(
            "{name} must be a vector or a squeezable vector, got {other:?}"
        ))),
    }
}
