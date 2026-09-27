//! Inference-time caches: KV, attention masks and rope tables.

pub(crate) mod kv;
pub(crate) mod mask;
pub(crate) mod rope;

pub use kv::{KvCache, TensorCache};
pub use mask::{AttentionMask, MaskCache};
pub use rope::RopeCache;

/// The mask attribute [`AttentionMask::Structural`] carries and
/// [`crate::composite::attention`] consumes, re-exported so a model crate
/// never has to name the IR crate.
pub use fusor_ir::ir::launch::MaskKind;

use fusor_ir::dtype::Dtype;
use fusor_ir::shape::Dim;

use crate::graph::Graph;
use crate::tensor::typed::Element;
use crate::{Result, Tensor};

/// A host table computed in `f32` and cast once to `T` on its way in: `-inf`
/// and f64-accumulated angles are exact at every float width.
pub(crate) fn f32_table<T: Element>(
    graph: &Graph,
    shape: [Dim; 2],
    data: &[f32],
) -> Result<Tensor<2, T>> {
    let dense = graph.tensor(Dtype::F32, &shape, bytemuck::cast_slice(data))?;
    Tensor::try_from_dyn(dense.into_dtype(T::DTYPE)?)
}
