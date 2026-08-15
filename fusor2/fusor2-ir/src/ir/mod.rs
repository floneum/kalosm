//! Levels, the node type, op tags, op semantics and the open op registry.

pub mod level0;
pub mod level1;
pub mod level2;

use crate::egraph::Id;
use crate::error::Result;
use crate::facts::{ValueFacts, Work};
use crate::ir::level0::L0;
use crate::ir::level1::{Effect, L1};
use smallvec::SmallVec;
use std::fmt;

/// The three descending abstraction levels. Nothing skips a level.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Level {
    L0,
    L1,
    L2,
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::L0 => "l0",
            Self::L1 => "l1",
            Self::L2 => "l2",
        })
    }
}

/// A node's operator. `Union` is a *node*, not a union-find edge: it keeps
/// every alternative alive simultaneously without a rebuild, and is
/// allocated at an id strictly greater than both operands, so acyclicity is
/// a property of the id allocator.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    L0(L0),
    L1(L1),
    Union(Id, Id),
}

impl Op {
    /// Level this operator belongs to; `Union` inherits its operands'.
    pub fn level(&self) -> Option<Level> {
        match self {
            Self::L0(_) => Some(Level::L0),
            Self::L1(_) => Some(Level::L1),
            Self::Union(..) => None,
        }
    }

    /// O(1) dispatch tag. Rules filter on this before any matching.
    pub fn tag(&self) -> OpTag {
        match self {
            Self::L0(o) => o.tag(),
            Self::L1(o) => o.tag(),
            Self::Union(..) => OpTag::Union,
        }
    }
}

/// Children of one node, inline up to 4.
pub type Children = SmallVec<[Id; 4]>;

/// One hash-consed e-graph node. **Acyclicity is structural, not checked**:
/// `children` may only contain ids strictly smaller than the node's own,
/// and the only id allocator is `EGraph::add`. There is no union-find, no
/// `rebuild()`, no congruence closure and no cycle probe.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Node {
    pub op: Op,
    pub level: Level,
    pub children: Children,
}

/// Flat O(1) dispatch tag for the rule table.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpTag {
    // L0
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
    // L1
    KMap,
    KFold,
    KContract,
    KGather,
    KScatter,
    KRegion,
    Ext,
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
            | Self::Project => Some(Level::L0),
            Self::Union => None,
            _ => Some(Level::L1),
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
    pub registry: &'a OpDefRegistry,
}

/// Type inference, cost accounting, verification and effects for one
/// operator. One implementation ([`crate::CoreSemantics`]) covers the
/// closed `L0`/`L1` enums plus the open [`OpDefRegistry`]. **Object-safe**:
/// the e-graph stores it as `Arc<dyn Semantics>`, so `fusor2-ir` is the
/// only crate that has to know how the closed enums infer.
pub trait Semantics: Send + Sync {
    /// Operand ids of `op`, in the order every other method expects.
    fn children(&self, op: &Op) -> Children;

    /// Total shape/dtype/numeric inference. Never panics.
    fn infer(&self, op: &Op, ins: &[ValueFacts]) -> Result<ValueFacts>;

    /// Work at these shapes. Must vary with shape — [`Self::verify`]
    /// rejects a registration whose `work` is constant.
    fn work(&self, op: &Op, ins: &[ValueFacts], out: &ValueFacts) -> Work;

    /// Level-local verification (`verify_l0` / `verify_l1`).
    fn verify(&self, cx: &VerifyCtx<'_>) -> Result<()>;

    /// Purity. An `InPlace` node is pinned in the materialized set.
    fn effect(&self, op: &Op) -> Effect;
}

/// Identity of an entry in the open op registry.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OpDefId(pub u32);

/// Identity of an attribute blob in the side table. Attributes live outside
/// [`Op`] so `Op` stays `Hash + Eq` and the hash-cons memo is exact.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AttrId(pub u32);

/// The single escape hatch for ops outside the closed enums. Top-k and the
/// two samplers enter this way: inference-only, no adjoint, one declared
/// cost row. Adding one changes no core file.
#[derive(Clone)]
pub struct OpDef {
    pub name: &'static str,
    pub tag: OpTag,
    pub verify: fn(&VerifyCtx<'_>) -> Result<()>,
    pub infer: fn(&[ValueFacts]) -> Result<ValueFacts>,
    pub work: fn(&[ValueFacts], &ValueFacts) -> Work,
    pub adjoint: Option<crate::autograd::AdjointKind>,
    /// Empty means "cannot run on any target"; `verify_plan` rejects
    /// selecting such a node.
    pub lower_per_target: &'static [(&'static str, level2::LowerFn)],
    pub effect: Effect,
}

impl fmt::Debug for OpDef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OpDef").field("name", &self.name).finish()
    }
}

/// The open registry. Registration order is id order and must be stable
/// across processes (plan hashes read it).
#[derive(Default, Clone, Debug)]
pub struct OpDefRegistry {
    defs: Vec<OpDef>,
}

impl OpDefRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, def: OpDef) -> OpDefId {
        let id = OpDefId(self.defs.len() as u32);
        self.defs.push(def);
        id
    }

    pub fn get(&self, id: OpDefId) -> Option<&OpDef> {
        self.defs.get(id.0 as usize)
    }

    pub fn by_name(&self, name: &str) -> Option<(OpDefId, &OpDef)> {
        self.defs
            .iter()
            .position(|d| d.name == name)
            .map(|i| (OpDefId(i as u32), &self.defs[i]))
    }

    pub fn iter(&self) -> impl Iterator<Item = (OpDefId, &OpDef)> {
        self.defs
            .iter()
            .enumerate()
            .map(|(i, d)| (OpDefId(i as u32), d))
    }
}
