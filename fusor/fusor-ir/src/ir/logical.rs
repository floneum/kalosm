//! Logical `tensor` — ten nodes of whole-tensor algebra: no index space, no
//! loop, no device.

use crate::carrier::Carrier;
use crate::dtype::{Dtype, QFmt, QLayout, Splat};
use crate::egraph::Id;
use crate::ir::OpTag;
use crate::scalar::ScalarExpr;
use crate::shape::{BoundsProof, Dim, SlidingWindow, StrideSpec, SymId};
use smallvec::SmallVec;

/// The ten Logical nodes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Logical {
    Leaf(LeafKind),

    /// Elementwise map; every operand has the output shape. `outs > 1` is a
    /// tuple read back through [`Logical::Project`].
    Map {
        expr: ScalarExpr,
        ins: SmallVec<[Id; 4]>,
        outs: u8,
    },

    /// Reduce `axis` with a [`Carrier`], an N-slot accumulator with its own
    /// identities, lift and merge; the lift reads `ins` as `Arg(0..n)`.
    Fold {
        carrier: Carrier,
        axis: u32,
        acc: Dtype,
        ins: SmallVec<[Id; 4]>,
    },

    /// Einstein-summation contraction; every matmul form is an [`EinSpec`].
    Contract {
        spec: EinSpec,
        acc: Dtype,
        a: Id,
        b: Id,
        outs: u8,
    },

    /// The one view primitive; all ~22 view ops lower to it.
    Restride {
        specs: SmallVec<[StrideSpec; 6]>,
        bounds: BoundsProof,
        x: Id,
    },

    /// Sliding windows, kept apart from [`Logical::Restride`] for its adjoint.
    Window {
        specs: SmallVec<[SlidingWindow; 3]>,
        x: Id,
    },

    /// Gather rows along `axis`.
    Gather {
        axis: u32,
        x: Id,
        idx: Id,
    },

    /// Scatter into `base`. `unique` is caller-proved index uniqueness, which
    /// `Set` requires; `Add` accumulates duplicates.
    Scatter {
        axis: u32,
        combine: ScatterCombine,
        base: Id,
        idx: Id,
        upd: Id,
        unique: bool,
    },

    Dequant {
        fmt: QFmt,
        layout: QLayout,
        x: Id,
    },

    /// Read one result out of a tuple-producing node.
    Project {
        slot: u8,
        x: Id,
    },
}

impl Logical {
    pub const fn tag(&self) -> OpTag {
        match self {
            Self::Leaf(_) => OpTag::Leaf,
            Self::Map { .. } => OpTag::Map,
            Self::Fold { .. } => OpTag::Fold,
            Self::Contract { .. } => OpTag::Contract,
            Self::Restride { .. } => OpTag::Restride,
            Self::Window { .. } => OpTag::Window,
            Self::Gather { .. } => OpTag::Gather,
            Self::Scatter { .. } => OpTag::Scatter,
            Self::Dequant { .. } => OpTag::Dequant,
            Self::Project { .. } => OpTag::Project,
        }
    }
}

/// What a leaf is. `Param` infers persistence; `Uniform` is a runtime scalar
/// from binding 0 that never enters a kernel key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LeafKind {
    Buffer {
        name: BufferId,
        dtype: Dtype,
        shape: SmallVec<[Dim; 6]>,
    },
    Param {
        name: BufferId,
        dtype: Dtype,
        shape: SmallVec<[Dim; 6]>,
    },
    Const {
        value: Splat,
        shape: SmallVec<[Dim; 6]>,
    },
    Uniform {
        sym: SymId,
        dtype: Dtype,
    },
    Quantized {
        name: BufferId,
        fmt: QFmt,
        layout: QLayout,
        shape: SmallVec<[Dim; 2]>,
    },
}

/// Stable name of an externally-supplied buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub u32);

/// How an extremum reduction splits its gradient among tied elements; read
/// only by `fold_adjoint`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TiePolicy {
    SplitEvenly,
    FirstWins,
}

/// How colliding scatter writes combine.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScatterCombine {
    Set,
    Add,
}

/// One index label in an [`EinSpec`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Label(pub u8);

/// Index labels for a contraction: a label in a and b but not out is summed;
/// one in all three is a batch axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EinSpec {
    pub a: SmallVec<[Label; 6]>,
    pub b: SmallVec<[Label; 6]>,
    pub out: SmallVec<[Label; 6]>,
}

impl EinSpec {
    /// The spec for `d/da`: `grad x b -> a`.
    pub fn d_lhs(&self) -> Self {
        Self {
            a: self.out.clone(),
            b: self.b.clone(),
            out: self.a.clone(),
        }
    }

    /// The spec for `d/db`: `a x grad -> b`.
    pub fn d_rhs(&self) -> Self {
        Self {
            a: self.a.clone(),
            b: self.out.clone(),
            out: self.b.clone(),
        }
    }
}
