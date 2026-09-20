//! `fusor` — the user-facing facade.
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
//! Operations build Logical nodes. Composite operations expand into the same
//! graph, where extraction chooses fusion and materialization together.

#![warn(missing_docs, unreachable_pub)]

#[cfg(not(any(feature = "cpu", feature = "gpu")))]
compile_error!("fusor: enable at least one backend feature, `cpu` or `gpu`");

pub mod autograd;
mod broadcast;
pub mod cache;
pub mod composite;
pub mod device;
pub mod graph;
pub mod layers;
pub(crate) mod ops;
pub mod optim;
#[cfg(feature = "gpu")]
pub mod program;
pub mod quantized;
pub mod sampling;
pub mod session;
pub mod tensor;

pub use device::Device;
pub use tensor::typed::{Axis, Element, Minus1, Minus2, Tensor, cat, stack};

pub use graph::Graph;
pub use quantized::QMatrix;
pub use session::Session;
/// Readback: `pollster::block_on(t.as_slice())?.to_vec()`.
pub use tensor::readback::ToVec;

pub use fusor_gguf::{ShardedVarBuilder, VarBuilder};
pub use fusor_ir::dtype::Dtype;
pub use fusor_ir::shape::Dim;
pub use fusor_ir::{Error, Result};
