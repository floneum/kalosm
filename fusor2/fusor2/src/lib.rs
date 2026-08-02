//! `fusor2` — the user-facing facade.
//!
//! Two views of one tensor. [`tensor::Tensor`] has runtime rank and dtype and
//! returns `Result` from every op: the layer to use when a shape is data.
//! [`tensor::typed::Tensor<R, T>`](tensor::typed::Tensor) tracks rank and
//! dtype in the type and panics on a mismatch: the layer a model is written
//! in, where a rank error is a bug and not a runtime condition. Neither has a
//! `B: Fusion<R, D>` parameter, no `OUT_RANK`/`DIFF`/`MaxRank` witness traits,
//! and no rank ceiling of 21 — the e-graph decides materialization, so there
//! is nothing for a backing parameter to say.
//!
//! [`autograd`] is the differentiable const-rank tensor on top of that, with
//! the tape, `with_backwards` and the gradient map.
//!
//! Every op is a thin builder that mints L0 nodes. **Macro ops union the sugar
//! node with its `defn` expansion in the same call**, so recognition ordering,
//! sole-consumer gates and the five destructive recognizers evaporate — there
//! was never anything to recognize.
//!
//! # The `typed-api` feature and the root `Tensor`
//!
//! `Tensor` and `Device` mean different things to the two views, and a crate
//! root has one of each name:
//!
//! * `fusor2::Tensor` is [`tensor::Tensor`] and `fusor2::Device` is
//!   [`session::Device`] — the default, and what every crate in this
//!   workspace compiles against.
//! * with `features = ["typed-api"]`, `fusor2::Tensor` is
//!   [`tensor::typed::Tensor`] and `fusor2::Device` is [`device::Device`].
//!
//! Both views are *always* compiled and always reachable by their module
//! paths; the feature moves nothing but two re-exports. It exists because
//! `Device::cpu()` genuinely has two signatures — `Result<Device>` for a
//! caller that falls back to another backend, `Device` for a model that would
//! only `unwrap` it — and no single item is both. The same holds for `Tensor`:
//! the runtime-rank one returns `Result` from `abs()`, the const-rank one
//! returns a tensor, and a zero-argument method cannot dispatch on that.
//!
//! **`typed-api` is not additive, and `--all-features` is not a valid
//! configuration of this workspace.** Turning it on changes what
//! `fusor2::Tensor` and `fusor2::Device` name, so every in-workspace caller
//! that spells the runtime-rank root stops compiling — `fusor2-conformance`,
//! and this crate's own `tests/lowering_regressions.rs`. Enable it from a
//! consumer that wants the const-rank root — `betlang-train` does — and leave
//! it off everywhere else. `cargo build --workspace --all-targets`,
//! `cargo test --workspace` and the conformance binary all take the default.
//!
//! Two things keep that from being a blind spot:
//!
//! * `root` below names *both* root sets in a module that is compiled in both
//!   configurations, so deleting or renaming an item either one exports is a
//!   default-build error, not a `typed-api`-only one. The cfg can hide an
//!   un-performed re-export, never a missing item.
//! * `cargo test -p fusor2 --features typed-api --lib` resolves the const-rank
//!   root for real and passes, and `trainer_surface` — which restates
//!   `betlang-train`'s whole surface — imports through that root when the
//!   feature is on. Only the `tests/` target is dark under the feature, and it
//!   is dark because it spells the *other* root.

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
/// Both submodules are compiled in **both** configurations and only one is
/// re-exported, so deleting or renaming an item either arm names is a compile
/// error under the default features too — the cfg cannot hide a missing item,
/// only an un-performed re-export.
///
/// `#[allow(unused_imports)]` silences the arm that is not re-exported. It
/// costs nothing: a path that fails to resolve is a hard error, not a lint.
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

/// Routing the re-exports through `root` must not change what the root
/// *names*. Every other crate in this workspace compiles against the default
/// root, so a slip here would be a silent, workspace-wide type change rather
/// than an error in this file. These coercions are type identities: they
/// compile only while each root name is the very item it was before.
// Never read: the value is irrelevant and only the coercion is the point, so
// `dead_code` has nothing useful to say. The check itself still runs — a type
// mismatch here is an error, not a lint.
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
