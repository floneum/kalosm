//! `fusor2` — the user-facing facade.
//!
//! One crate root, one [`Tensor`], one [`Device`].
//!
//! [`Tensor<R, T>`](Tensor) carries its rank and its dtype in the type and
//! panics on a mismatch: a rank error is a bug in the model, not a runtime
//! condition a caller can act on.
//!
//! Underneath it, and reachable by [`Tensor::into_dyn`] / [`Tensor::as_dyn`],
//! is [`tensor::Dyn`]: the same node with runtime rank and dtype, returning
//! `Result` from every op. That is the layer for code where a shape is data —
//! a GGUF loader, a pass over a heterogeneous list. `Tensor` is a
//! `repr(transparent)` newtype over it, so moving between the two is free.
//!
//! [`Device`] is what a constructor takes and what [`Tensor::device`] hands
//! back — one type, so `Tensor::zeros(&x.device(), shape)` compiles. The
//! backend selector it is built from is [`session::Backend`], and a
//! [`Session`] is built from that; neither is something a model names.
//!
//! [`autograd`] is the differentiable const-rank tensor on top of all of it,
//! with the tape, `with_backwards` and the gradient map.
//!
//! Every op is a thin builder that mints Logical nodes. Macro ops union the
//! sugar node with its `defn` expansion in the same call; how many kernels an
//! op launches is the extractor's answer, not the caller's.

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

pub use device::Device;
pub use tensor::typed::{Axis, Element, Minus1, Minus2, Tensor, cat, stack};

pub use graph::Graph;
pub use quantized::QMatrix;
pub use session::Session;
/// Readback: `pollster::block_on(t.as_slice())?.to_vec()`.
pub use tensor::readback::ToVec;

pub use fusor2_gguf::{ShardedVarBuilder, VarBuilder};
pub use fusor2_ir::dtype::Dtype;
pub use fusor2_ir::shape::Dim;
pub use fusor2_ir::{Error, Result};
