//! Composite operations expanded directly into the logical graph.

pub(crate) mod activations;
pub(crate) mod attention;
pub(crate) mod conv;
pub(crate) mod loss;
pub(crate) mod normalization;
pub(crate) mod pool;
pub(crate) mod quantized;
pub(crate) mod rope;
pub(crate) mod upsample;

pub use attention::{
    attention, attention_causal, attention_grads, attention_lse, attention_masked,
    attention_with_lse,
};
pub use conv::{conv, grouped_conv, pad_with_zeros};
pub use loss::{binary_cross_entropy_with_logits, distillation_loss, mse, softmax_cross_entropy};
pub use pool::{PoolSize, pool, pool_avg, pool_max, pool_min};
pub use rope::{
    base_inverse_frequency, rope, rope_interleaved, rope_interleaved_pair,
    rope_interleaved_pair_with_position, rope_interleaved_with_position, rope_pair,
    rope_pair_with_position, rope_with_position, rotate_half,
};
pub use upsample::{upsample_bilinear, upsample_nearest, upsample_nearest2d};

use fusor_autograd::tape::GraphTape;
use fusor_ir::egraph::Id;
use fusor_ir::shape::Dim;
use fusor_ir::{Error, Result};

use crate::graph::GraphRef;
use crate::tensor::Tensor;

/// Which reduction a pool performs over each window.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum PoolReduce {
    /// Maximum pooling.
    Max,
    /// Minimum pooling.
    Min,
    /// Average pooling.
    Mean,
}

/// Build a composite from logical operations.
pub(crate) fn core_op(
    graph: &GraphRef,
    build: impl FnOnce(&mut GraphTape<'_>) -> Result<Id>,
) -> Result<Tensor> {
    let id = graph.build(build)?;
    Ok(graph.tensor(id))
}

/// A const extent, or an error. Used only where an algorithm genuinely needs
/// the integer (a window size, a kernel extent).
pub(crate) fn const_dim(d: Dim, what: &str) -> Result<u64> {
    d.as_const()
        .ok_or_else(|| Error::Shape(format!("{what} needs a decidable extent, got {d}")))
}

/// A rank-1 `u32` index leaf holding `values`: one small buffer uploaded
/// once, fed to scatter and gather as a real index tensor.
pub(crate) fn index_leaf(graph: &GraphRef, values: &[u32]) -> Result<Id> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for v in values {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    graph.constant_leaf(
        fusor_ir::dtype::Dtype::U32,
        &[Dim::Const(values.len() as u64)],
        bytes,
    )
}

/// `index_leaf` over the run `start .. start + len`.
pub(crate) fn index_run(graph: &GraphRef, start: u64, len: u64) -> Result<Id> {
    let values: Vec<u32> = (0..len)
        .map(|i| {
            u32::try_from(start + i)
                .map_err(|_| Error::Shape(format!("index {} exceeds a u32", start + i)))
        })
        .collect::<Result<_>>()?;
    index_leaf(graph, &values)
}
