//! `fusor2` — the user-facing facade.
//!
//! One crate root, one [`Tensor`], one [`Device`].
//!
//! [`Tensor<R, T>`](Tensor) carries its rank and its dtype in the type and
//! panics on a mismatch: a forward pass chains thirty ops per expression and a
//! rank error in any of them is a bug in the model, not a runtime condition a
//! caller can act on. It has **no** `B: Fusion<R, D>` parameter, no
//! `OUT_RANK`/`DIFF`/`MaxRank` witness traits, and no rank ceiling of 21 — the
//! e-graph decides materialization, so there is nothing for a backing
//! parameter to say.
//!
//! Underneath it, and reachable by [`Tensor::into_dyn`] / [`Tensor::as_dyn`],
//! is [`tensor::Dyn`]: the same node with runtime rank and dtype, returning
//! `Result` from every op. That is the layer for code where a shape is data —
//! a GGUF loader, a pass over a heterogeneous list — and it is deliberately
//! *not* the headline type. `Tensor` is a `repr(transparent)` newtype over it,
//! so moving between the two is free.
//!
//! [`Device`] is what a constructor takes and what [`Tensor::device`] hands
//! back — one type, so `Tensor::zeros(&x.device(), shape)` compiles. The
//! backend selector it is built from is [`session::Backend`], and a
//! [`Session`] is built from that; neither is something a model names.
//!
//! [`autograd`] is the differentiable const-rank tensor on top of all of it,
//! with the tape, `with_backwards` and the gradient map.
//!
//! Every op is a thin builder that mints L0 nodes. **Macro ops union the sugar
//! node with its `defn` expansion in the same call**, so recognition ordering,
//! sole-consumer gates and the five destructive recognizers evaporate — there
//! was never anything to recognize. That is also why there is no `*_fused`
//! family on this surface: `x.rms_norm(w, eps)` and a hypothetical
//! `rms_norm_fused` would mint the same node, and how many kernels it launches
//! is the extractor's answer, not the caller's.

pub mod autograd;
mod broadcast;
pub mod cache;
pub mod composite;
pub mod device;
pub mod graph;
pub mod layers;
pub(crate) mod ops;
pub mod optim;
pub mod quantized;
pub mod sampling;
pub mod session;
pub mod tensor;

/// The intended public surface, restated as `use` lines, so it cannot
/// silently drift. Test-only: it defines no public item.
#[cfg(test)]
mod api_surface;

/// The trainer's API surface, restated so a regression is a compile error
/// here. Test-only: it defines no public item.
#[cfg(test)]
mod trainer_surface;

// --- the root -------------------------------------------------------------
//
// One naming, unconditionally. There is no feature that swaps what `Tensor`
// or `Device` means: the const-rank tensor and the device that builds it are
// the API, and the runtime-rank pair keeps its module path for the code that
// genuinely needs it.

pub use device::Device;
pub use tensor::typed::{Axis, Element, Minus1, Minus2, Tensor, cat, stack};
/// The reference's axis-selector spelling: `t.sum::<2>(D::Minus1)`. Same
/// items as the root `Minus1`/`Minus2`; kept so an axis argument written
/// against the old API compiles unchanged.
///
/// The reference also named the selector *trait* `Dim`; here that name is the
/// extent enum, and [`Axis`] is the trait. Old generic bounds port with one
/// line: `use fusor2::Axis as Dim;`.
#[allow(non_snake_case)]
pub mod D {
    pub use crate::tensor::typed::{Minus1, Minus2};
}
/// The reference's name for [`tensor::typed::Element`], kept at the root it
/// had there.
pub use tensor::typed::SimdElement;

pub use graph::Graph;
pub use quantized::QMatrix;
pub use session::Session;
/// Readback: `pollster::block_on(t.as_slice())?.to_vec()`.
pub use tensor::readback::ToVec;

pub use fusor2_gguf::{ShardedVarBuilder, VarBuilder};
pub use fusor2_ir::dtype::Dtype;
pub use fusor2_ir::shape::Dim;
pub use fusor2_ir::{Error, Result};

// Reference-spelling re-exports: names the old API had at its root, pointing
// at the same items they resolve to today. Paths only — no second
// implementation exists behind any of these.
pub use cache::{MaskKind, RopeCache};
pub use composite::rope::base_inverse_frequency;
