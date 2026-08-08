//! `fusor2` — the user-facing facade.
//!
//! Two views of one tensor. [`tensor::Tensor`] has runtime rank and dtype and
//! returns `Result` from every op.
//! [`tensor::typed::Tensor<R, T>`](tensor::typed::Tensor) tracks rank and dtype
//! in the type and panics on a mismatch. Neither carries a backing-store
//! parameter or a rank ceiling: the e-graph decides materialization.
//!
//! [`autograd`] is the differentiable const-rank tensor on top of that, with
//! the tape, `with_backwards` and the gradient map.
//!
//! Every op is a thin builder that mints L0 nodes. Macro ops union the sugar
//! node with its `defn` expansion in the same call, so there is no recognition
//! pass and no recognition ordering.
//!
//! # The `typed-api` feature and the root `Tensor`
//!
//! Without the feature, `fusor2::Tensor` is [`tensor::Tensor`] and
//! `fusor2::Device` is [`session::Device`]; with it, they are
//! [`tensor::typed::Tensor`] and [`device::Device`]. Both views are always
//! compiled and always reachable by their module paths; the feature moves
//! nothing but those two re-exports.
//!
//! The feature is not additive, so `--all-features` is not a valid
//! configuration of this workspace: turning it on stops every in-workspace
//! caller that spells the runtime-rank root from compiling. Enable it from a
//! consumer that wants the const-rank root and leave it off everywhere else.

pub mod autograd;
pub mod broadcast;
pub mod cache;
pub mod composite;
pub mod device;
pub mod graph;
pub mod layers;
pub mod ops;
pub mod optim;
pub mod quantized;
pub mod sampling;
pub mod session;
pub mod tensor;

/// The trainer's API surface, restated so a regression is a compile error
/// here. Test-only: it defines no public item.
#[cfg(test)]
mod trainer_surface;

pub use graph::{Gradients, Graph};
pub use quantized::QMatrix;
pub use session::Session;
pub use tensor::readback::{TensorSlice, ToVec};
pub use tensor::typed::Typed;

/// The two root namings `typed-api` chooses between, each written once.
///
/// Both submodules are compiled in both configurations and only one is
/// re-exported, so the cfg can hide an un-performed re-export but never a
/// missing item. `#[allow(unused_imports)]` silences the arm that is not
/// re-exported.
mod root {
    /// The crate root without `typed-api`: runtime rank, `Result`-returning.
    #[allow(unused_imports)]
    pub mod dynamic {
        pub use crate::session::Device;
        pub use crate::tensor::Tensor;
    }

    /// The crate root with `typed-api`: const rank, panic-on-error. This is
    /// the set `betlang-train` resolves `use fusor::{Device, Tensor, cat}`
    /// against, and `trainer_surface` type-checks against it directly.
    #[allow(unused_imports)]
    pub mod typed {
        pub use crate::device::Device;
        pub use crate::tensor::typed::{Tensor, cat, stack};
    }
}

#[cfg(not(feature = "typed-api"))]
pub use root::dynamic::{Device, Tensor};

#[cfg(feature = "typed-api")]
pub use root::typed::{Device, Tensor, cat, stack};

/// Pins what the root names: a type identity that compiles only while each
/// root name resolves to the item named on the left. Never read; only the
/// coercion is the point.
#[allow(dead_code)]
#[cfg(not(feature = "typed-api"))]
const DYNAMIC_ROOT_IS_UNCHANGED: fn(tensor::Tensor, session::Device) -> (Tensor, Device) =
    |t, d| (t, d);

#[allow(dead_code)]
#[cfg(feature = "typed-api")]
const TYPED_ROOT_IS_THE_CONST_RANK_PAIR: fn(
    tensor::typed::Tensor<2, f32>,
    device::Device,
) -> (Tensor<2, f32>, Device) = |t, d| (t, d);

pub use fusor2_gguf::{ShardedVarBuilder, VarBuilder};
pub use fusor2_ir::dtype::{Dtype, Persistence, QFmt, QLayout, RoundMode};
pub use fusor2_ir::shape::{Dim, SymId};
pub use fusor2_ir::{Error, Result};
