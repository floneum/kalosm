//! L0 `tensor` — ten nodes of whole-tensor algebra. No index space, no loop,
//! no device. Only L0 can express adjoint generation, contraction
//! reassociation, the fold-splitting law, and gradient checkpointing.

use crate::carrier::Carrier;
use crate::dtype::{Dtype, QFmt, QLayout, Splat};
use crate::egraph::Id;
use crate::ir::OpTag;
use crate::scalar::ScalarExpr;
use crate::shape::{BoundsProof, Dim, SlidingWindow, StrideSpec, SymId};
use smallvec::SmallVec;

/// The ten L0 nodes. Every elementwise unary, comparison, and activation
/// is one `Map` with a different [`ScalarExpr`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum L0 {
    Leaf(LeafKind),

    /// Elementwise map. **No implicit broadcasting**: every operand has the
    /// output shape; the frontend emits `Restride { multiplier: 0 }`.
    /// `outs > 1` produces a tuple read back through [`L0::Project`].
    Map {
        expr: ScalarExpr,
        ins: SmallVec<[Id; 4]>,
        outs: u8,
    },

    /// Reduce `axis` with a [`Carrier`] — an N-slot accumulator carrying its
    /// own identities, lift and merge. There is no named combine: `Add`,
    /// `Mul`, `Max` and `Min` are [`Carrier::binop`] values, and Welford
    /// `(n, mean, m2)`, online softmax `(max, sum)` and attention's `sum p*v`
    /// are carriers a *rule* constructs. `ins` is the operand list the lift
    /// reads as `Arg(0..n)`.
    Fold {
        carrier: Carrier,
        axis: u32,
        acc: Dtype,
        ins: SmallVec<[Id; 4]>,
    },

    /// Einstein-summation contraction. `matmul`, `mat_mul_transposed_rhs`
    /// and every batched form are one node with a different [`EinSpec`] —
    /// transposed-rhs is a spec, not an op.
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

    /// Sliding windows. Survives as a core op rather than collapsing into
    /// [`L0::Restride`] because its adjoint is decided by two integers.
    Window {
        specs: SmallVec<[SlidingWindow; 3]>,
        x: Id,
    },

    /// Gather rows along `axis`. `index_select`, `embedding`,
    /// `gather_last` and `i()` are this.
    Gather { axis: u32, x: Id, idx: Id },

    /// Scatter into `base`. `cat`/`stack`/`pad_axis`/`repeat`/
    /// `slice_assign` are `Scatter{Set}` into a const leaf; the adjoint of
    /// [`L0::Gather`] is `Scatter{Add}`. `unique` is caller-proved index
    /// uniqueness: `verify_l0` rejects `Set` without it, while `Add` is
    /// always legal and duplicates accumulate (normative).
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
    Project { slot: u8, x: Id },
}

impl L0 {
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

/// What a leaf is. `Param` is distinguished from `Buffer` so `Persistence`
/// is inferred, not annotated; `Uniform` is a runtime scalar read from
/// binding 0 (learning rate, bias correction, loss scale, clip norm) and
/// never enters a kernel key.
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

/// How an extremum reduction splits its gradient among tied elements.
/// Carried on [`Carrier::tie`] and read only by `fold_adjoint`: an autograd
/// attribute, never a compiler decision.
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

/// Index labels for a contraction. `verify_l0` requires every label to
/// appear in >= 2 of {a, b, out} and contracted extents to agree. A label
/// in a and b but not out is summed; one in all three is a batch axis.
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
