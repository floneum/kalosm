//! Pluggable binder kinds — the abstract interface used by `binding.rs`
//! (shift/subst) and `naga_codegen` (frame-stack lookup).
//!
//! `BinderKind` tags the kind of binder a `VarRef::Bound` is scoped to.
//! `BinderInfo` describes, for a single IR node, which kind of binder it is
//! and which of its children lie inside the newly introduced scope.
//! `HasBinder` is the trait `binding.rs` queries to stay agnostic of any
//! particular IR variant.
//!
//! Adding a new binder kind is a three-step change: add a variant here, add
//! the matching `BinderInfo` in `impl HasBinder for TensorIr`, and add a
//! codegen frame in `naga_codegen::BinderFrame`. No edits to `binding.rs`.
//!
//! Slot numbering is per-kind: the constants in [`slots`] spell out what a
//! given `slot: u8` means for each `BinderKind`. De Bruijn depths are
//! per-kind as well — a Theta substitution never disturbs a Dispatch-bound
//! reference and vice versa, because shift/subst filter on `kind`.

use std::fmt;
use std::str::FromStr;

/// Which kind of binder a `VarRef::Bound` refers to.
///
/// Depths are counted independently per kind: `Bound { kind: Theta, depth: 0 }`
/// names the innermost enclosing `Theta`, regardless of how many `Dispatch`
/// frames lie between the reference and the binder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BinderKind {
    /// `SimdNode::Theta` — a functional fixpoint loop. Two slots: iter, acc.
    Theta,
    /// `DispatchNode::Dispatch` — binds the three GPU thread-index levels
    /// (lane, simdgroup, workgroup) for the enclosed kernel body.
    Dispatch,
}

impl fmt::Display for BinderKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Theta => write!(f, "theta"),
            Self::Dispatch => write!(f, "dispatch"),
        }
    }
}

impl FromStr for BinderKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "theta" => Ok(Self::Theta),
            "dispatch" => Ok(Self::Dispatch),
            _ => Err(format!("unknown binder kind: {s}")),
        }
    }
}

/// Static description of a binder node.
///
/// `body_mask` is positional against `Language::children()` for the node —
/// bit `i` set ⇒ child `i` is inside the new scope (its De Bruijn cutoff is
/// incremented by 1 when shift/subst crosses into it). Bits outside the
/// child count are ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BinderInfo {
    pub kind: BinderKind,
    pub body_mask: u16,
}

/// Trait implemented by the IR enum. Returns `Some(..)` iff this node
/// introduces a new binding scope.
pub trait HasBinder {
    fn binder_info(&self) -> Option<BinderInfo>;
}

/// Per-kind slot numbering.
///
/// `VarRef::Bound { kind, slot, depth }` — `slot` is a small u8 index whose
/// meaning is fixed per kind by the constants below. Use these names instead
/// of literal integers at call sites.
pub mod slots {
    pub const THETA_ITER: u8 = 0;
    pub const THETA_ACC: u8 = 1;

    pub const DISPATCH_LANE: u8 = 0;
    pub const DISPATCH_SIMDGROUP: u8 = 1;
    pub const DISPATCH_WORKGROUP: u8 = 2;
}
