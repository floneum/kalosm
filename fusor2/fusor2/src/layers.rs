//! Parameterized layers. Each is a thin struct over a few `Tensor` parameters
//! and a `forward`; none of them owns a kernel.

pub mod conv;
pub mod embedding;
pub mod layer_norm;
pub mod linear;
pub mod rms_norm;

pub use conv::ConvNd;
pub use embedding::Embedding;
pub use layer_norm::{LayerNorm, LayerNormNd};
pub use linear::Linear;
pub use rms_norm::RmsNorm;


use fusor2_gguf::VarBuilder;
use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::level0::{L0, LeafKind};
use fusor2_ir::shape::Dim;

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
    crate::Tensor::<R, T>::try_from_dyn(t)
        .map_err(|e| Error::Shape(format!("{what}: {e}")))
}

/// One GGUF tensor as a dense `F32` value in `graph`.
///
/// `fusor2_gguf` reverses a tensor's extents at read, so `raw.shape` is
/// already row-major: a `[out, in]` weight is `[out, in]` here, matching the
/// reference's `weight.shape()[0] == out_features`.
///
/// A block-quantized entry becomes a `Leaf(Quantized)` plus one `L0::Dequant`
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
        L0::Leaf(LeafKind::Quantized {
            name: graph.fresh_buffer_id(),
            fmt,
            layout: raw.layout,
            shape: [rows, cols].into_iter().collect(),
        }),
    )?;
    graph.set_leaf_bytes(leaf.id(), raw.bytes.to_vec());
    Tensor::emit(
        graph,
        L0::Dequant {
            fmt,
            layout: raw.layout,
            x: leaf.id(),
        },
    )
}

/// A const-rank leaf, for the layer tests. `T::DTYPE` is the leaf's dtype, so
/// a `Tensor<2, f16>` leaf is an f16 one.
#[cfg(test)]
pub(crate) fn test_leaf<const R: usize, T: Element>(
    g: &crate::graph::Graph,
    shape: &[u64],
) -> crate::Tensor<R, T> {
    let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
    crate::Tensor::from_dyn(g.leaf("t", &dims, T::DTYPE).expect("leaf"))
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use fusor2_gguf::parse::{GgmlType, GgufMetadata, GgufTensor, GgufVersion};
    use fusor2_ir::shape::Dim;

    use crate::graph::Graph;
    use crate::session::{Backend, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().expect("cpu device")).expect("session"))
    }

    /// A synthetic GGUF file. `parse::fixture` is `#[cfg(test)]`-gated inside
    /// its own crate, so this rebuilds it from the public writer: shapes are
    /// row-major here and `GgufMetadata::write` reverses them onto the wire.
    fn gguf(tensors: &[(&str, GgmlType, &[u64], Vec<u8>)]) -> VarBuilder {
        let mut infos = Vec::new();
        let mut offset = 0u64;
        for (name, ty, shape, _) in tensors {
            let shape: smallvec::SmallVec<[u64; 4]> = shape.iter().copied().collect();
            let bytes = GgufTensor::byte_size(*ty, &shape).expect("whole blocks");
            infos.push(GgufTensor {
                name: (*name).to_string(),
                ty: *ty,
                shape,
                offset,
                bytes,
            });
            offset += bytes;
        }
        let meta = GgufMetadata {
            version: GgufVersion::V3,
            entries: Vec::new(),
            tensors: infos,
            tensor_data_offset: 0,
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        meta.write(
            &mut buf,
            tensors.iter().map(|(n, _, _, d)| (*n, d.as_slice())),
        )
        .expect("write");
        VarBuilder::new(Arc::new(
            fusor2_gguf::parse::Gguf::from_bytes(buf.into_inner()).expect("gguf"),
        ))
    }

    fn model() -> VarBuilder {
        gguf(&[
            (
                "blk.0.attn_q.weight",
                GgmlType::Q4_0,
                &[2, 32],
                (0..36u8).collect(),
            ),
            ("blk.0.attn_q.bias", GgmlType::F32, &[2], vec![0u8; 8]),
            ("norm.weight", GgmlType::F16, &[1, 4], vec![0u8; 8]),
        ])
    }

    #[test]
    fn a_gguf_matrix_keeps_its_row_major_shape() {
        let g = graph();
        let vb = model().pp("blk").pp(0);
        let w = load_dense(&vb, g.handle(), "attn_q.weight").unwrap();
        // `[out, in]`, not the `[in, out]` the file writes on the wire.
        assert_eq!(&w.shape()[..], &[Dim::Const(2), Dim::Const(32)]);
        assert_eq!(w.dtype(), Dtype::F32);
    }

    #[test]
    fn a_missing_key_is_none_and_a_present_one_is_some() {
        let g = graph();
        let vb = model().pp("blk").pp(0);
        assert!(
            load_optional(&vb, g.handle(), "attn_q.bias")
                .unwrap()
                .is_some()
        );
        assert!(load_optional(&vb, g.handle(), "nope").unwrap().is_none());
    }

    #[test]
    fn a_half_precision_vector_arrives_as_f32() {
        let g = graph();
        let vb = model();
        let w = load_dense(&vb, g.handle(), "norm.weight").unwrap();
        assert_eq!(w.dtype(), Dtype::F32);
        let v = as_vector(w, "norm.weight").unwrap();
        assert_eq!(&v.shape()[..], &[Dim::Const(4)]);
    }

    #[test]
    fn two_quantized_leaves_do_not_hash_cons_together() {
        let g = graph();
        let vb = model().pp("blk").pp(0);
        let a = load_dense(&vb, g.handle(), "attn_q.weight").unwrap();
        let b = load_dense(&vb, g.handle(), "attn_q.weight").unwrap();
        assert_ne!(a.id(), b.id(), "each load owns its own buffer");
    }
}
