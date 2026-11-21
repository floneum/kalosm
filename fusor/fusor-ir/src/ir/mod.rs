//! Levels, the node type, op tags, op semantics.

pub mod kernel;
pub mod launch;
pub mod logical;
pub mod visit;

use crate::egraph::Id;
use crate::error::Result;
use crate::facts::{ValueFacts, Work};
use crate::ir::launch::{Effect, Launch};
use crate::ir::logical::Logical;
use smallvec::SmallVec;
use std::fmt;

/// The three descending abstraction levels. Nothing skips a level.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Level {
    Logical,
    Launch,
    Kernel,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Logical => "logical",
            Self::Launch => "launch",
            Self::Kernel => "kernel",
        })
    }
}

/// A node's operator. `Union` is a node above both operands, so every
/// alternative stays alive without a rebuild and the graph stays acyclic.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    Logical(Logical),
    Launch(Launch),
    Union(Id, Id),
}

impl Op {
    /// Level this operator belongs to; `Union` inherits its operands'.
    pub fn level(&self) -> Option<Level> {
        match self {
            Self::Logical(_) => Some(Level::Logical),
            Self::Launch(_) => Some(Level::Launch),
            Self::Union(..) => None,
        }
    }

    /// O(1) dispatch tag. Rules filter on this before any matching.
    pub fn tag(&self) -> OpTag {
        match self {
            Self::Logical(o) => o.tag(),
            Self::Launch(o) => o.tag(),
            Self::Union(..) => OpTag::Union,
        }
    }
}

/// Children of one node, inline up to 4.
pub type Children = SmallVec<[Id; 4]>;

/// One hash-consed e-graph node; `children` are strictly smaller ids.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    pub op: Op,
    pub level: Level,
    pub children: Children,
}

/// Flat O(1) dispatch tag for the rule table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpTag {
    // Logical
    Leaf,
    Map,
    Fold,
    Contract,
    Restride,
    Window,
    Gather,
    Scatter,
    Dequant,
    Project,
    // Launch
    LaunchMap,
    LaunchFold,
    LaunchStreamFold,
    LaunchContract,
    LaunchGather,
    LaunchScatter,
    LaunchSlab,
    LaunchGroup,
    // structural
    Union,
}

impl OpTag {
    pub const fn level(self) -> Option<Level> {
        match self {
            Self::Leaf
            | Self::Map
            | Self::Fold
            | Self::Contract
            | Self::Restride
            | Self::Window
            | Self::Gather
            | Self::Scatter
            | Self::Dequant
            | Self::Project => Some(Level::Logical),
            Self::Union => None,
            _ => Some(Level::Launch),
        }
    }
}

/// Read-only context handed to a level verifier.
pub struct VerifyCtx<'a> {
    pub node: &'a Node,
    pub id: Id,
    pub operands: &'a [ValueFacts],
    pub result: &'a ValueFacts,
    pub caps: &'a crate::device::Caps,
}

/// Type inference, cost accounting, verification and effects for one
/// operator; implemented once by [`crate::CoreSemantics`].
pub trait Semantics: Send + Sync {
    /// Operand ids of `op`, in the order every other method expects.
    fn children(&self, op: &Op) -> Children;

    /// Total shape/dtype/numeric inference. Never panics.
    fn infer(&self, op: &Op, ins: &[ValueFacts]) -> Result<ValueFacts>;

    /// Work at these shapes.
    fn work(&self, op: &Op, ins: &[ValueFacts], out: &ValueFacts) -> Work;

    /// Level-local verification (`verify_l0` / `verify_launch`).
    fn verify(&self, cx: &VerifyCtx<'_>) -> Result<()>;

    /// Purity. An `InPlace` node is pinned in the materialized set.
    fn effect(&self, op: &Op) -> Effect;
}
