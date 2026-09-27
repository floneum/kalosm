//! Every level-generic rewrite rule, and the one table the driver is handed.
//!
//! Guards may read only [`crate::egraph::Facts`] — legality, never
//! profitability.
//!
//! Rule order carries no semantics; the fixed order below exists only for
//! reproducibility.

pub mod absorb_view;
pub mod algebra;
pub mod fusion;
pub mod group;
pub mod layout;
pub mod lower_floor;
pub mod promote;
pub mod rebase;
pub mod scatter_fold;
pub mod sink;
pub mod slab;
pub mod specialize;
pub mod split_k;
pub mod stream;
pub mod tuple;

use crate::carrier::Carrier;
use crate::dtype::Dtype;
use crate::egraph::{Builder, Id, Rule, ViewSpine};
use crate::ir::Op;
use crate::ir::launch::{AccessPlan, IndexSpace, Launch, Operand, ScheduleDomain};
use crate::ir::logical::Logical;
use crate::scalar::ScalarExpr;
use crate::shape::{Dim, Dims, Layout, StrideSpec};
use smallvec::SmallVec;

/// Position of a rule in whatever slice was handed to the driver. `RuleId`
/// is positional, not global: a target concatenates [`CORE_RULES`] with its
/// own `Target::rules()` and the driver indexes the concatenation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuleId(pub u16);

/// Every rule `fusor-ir` owns, in a fixed order. A target's `rules()` is
/// concatenated onto this; nothing here mentions lane or subgroup geometry.
pub static CORE_RULES: &[Rule] = &[
    // Logical algebra
    algebra::STRIP,
    algebra::RECOGNIZE_CONTRACT,
    algebra::CONTRACT_REASSOC,
    algebra::CONST_FOLD_MAP,
    algebra::IDENTITY_ELIM,
    algebra::WIDEN_STORE_CAST,
    algebra::UNIT_FOLD_COLLAPSE,
    // Launch fusion
    fusion::ABSORB,
    fusion::MAP_INTO_CONTRACT,
    fusion::MAP_INTO_MAP,
    fusion::FOLD_POST_EPILOGUE,
    slab::FORM_SLAB_MAP,
    slab::FORM_SLAB_FOLD,
    stream::STREAM_FOLD,
    // Launch fold algebra — the carrier laws. `HOIST` and `RETARGET` are two
    // entries sharing one dependence query: the driver's fired set is per
    // `(RuleId, Id)`, so one merged rule could fire at most once per node
    // and the second answer would be unreachable.
    promote::PROMOTE,
    rebase::HOIST,
    rebase::RETARGET,
    tuple::TUPLE,
    tuple::TUPLE_SIBLING,
    // Launch sinking
    sink::SINK_EPILOGUE,
    sink::FOLD_VIEWS_INTO_INDEX,
    sink::FOLD_VIEWS_INTO_FOLD_INDEX,
    // Launch operand access
    layout::OPERAND_ALIAS,
    layout::OPERAND_GATHER,
    layout::OPERAND_PACK,
    layout::OPERAND_UNFLATTEN,
    // the M0 correctness floor
    lower_floor::LOWER_MAP,
    lower_floor::LOWER_FOLD,
    lower_floor::LOWER_CONTRACT_GENERIC,
    lower_floor::LOWER_RESTRIDE,
    lower_floor::LOWER_WINDOW,
    lower_floor::LOWER_GATHER,
    lower_floor::LOWER_SCATTER,
    lower_floor::LOWER_DEQUANT,
    lower_floor::LOWER_PROJECT,
    // shape specialization
    specialize::SPECIALIZE_DIM,
    split_k::SPLIT_K,
    absorb_view::ABSORB_VIEW_INTO_CONTRACT,
    absorb_view::ABSORB_BROADCAST_INTO_MAP,
    absorb_view::ABSORB_BROADCAST_INTO_FOLD,
    group::FORM_GROUP_MAP,
    group::FORM_GROUP_FOLD,
    group::FORM_GROUP_CONTRACT,
    group::FORM_GROUP_SLAB,
    scatter_fold::SCATTER_ADD_AS_FOLD,
];

/// Bytes of private accumulator one invocation may hold: 256 (64 `f32`
/// lanes, the widest shipped tile) for a carrier `PROMOTE` widens, 1024 for
/// a `TUPLE` joint. `Caps` has no portable register-budget fact, so both are
/// conservative policy; over budget a rule declines rather than minting a
/// node no backend can lower.
pub(crate) fn private_acc_bytes(_caps: &crate::device::Caps, promoted: bool) -> u64 {
    if promoted { 256 } else { 1024 }
}

/// An operand read straight out of its producer's dense row-major layout.
pub(crate) fn alias_operand_of(src: Id, shape: &[Dim]) -> Operand {
    Operand {
        src,
        layout: Layout::contiguous(shape),
        access: AccessPlan::Alias,
    }
}

/// The identity scalar body, `Arg(0)`.
pub(crate) fn ident_expr(dtype: Dtype) -> ScalarExpr {
    ScalarExpr::arg(0, dtype)
}

/// The access predicate `map_into_fold` and `map_into_contract` guard on:
/// `Alias`, `Unflatten` and `Gather` are legal in any space; a `Pack` is
/// legal only when the packed layout has the consuming space's rank.
pub(crate) fn access_legal_in(a: &AccessPlan, space: &IndexSpace) -> bool {
    match a {
        AccessPlan::Alias | AccessPlan::Unflatten(_) | AccessPlan::Gather => true,
        AccessPlan::Pack { into } => into.rank() == space.rank(),
    }
}

/// The elementwise producer shape a fusion rule inlines.
///
/// Equality in this e-graph is **not congruent**, so an `Logical::Map` and the
/// `Launch::Map` it was lowered to are one class but the consuming Launch node's
/// operand still names whichever id the frontend built. Both spellings
/// denote the same value, so both are inlinable; this normalizes them.
pub(crate) struct MapView {
    pub space: IndexSpace,
    pub body: ScalarExpr,
    pub ops: Vec<Operand>,
}

/// Read `id` as an elementwise producer, in either spelling.
pub(crate) fn map_view(b: &Builder<'_>, id: Id) -> Option<MapView> {
    match b.node(id).op.clone() {
        Op::Launch(Launch::Map {
            space, body, ops, ..
        }) => Some(MapView { space, body, ops }),
        Op::Logical(Logical::Map { expr, ins, outs: 1 }) => {
            let space = IndexSpace::new(b.facts_of(id).shape.iter().copied());
            let ops = ins
                .iter()
                .map(|&s| alias_operand_of(s, &b.facts_of(s).shape))
                .collect();
            Some(MapView {
                space,
                body: expr,
                ops,
            })
        }
        _ => None,
    }
}

/// Splice `inner`'s body in for `Arg(slot)` of a reader of `ops`: the
/// reader's other operands keep their order and low indices, the producer's
/// are appended by the caller. Returns the retained operands and the
/// substitution renumbering the reader's `Arg`s onto them.
pub(crate) fn splice_args(
    b: &Builder<'_>,
    ops: &[Operand],
    slot: usize,
    inner: &MapView,
) -> (Vec<Operand>, Vec<ScalarExpr>) {
    let base = ops.len() - 1;
    let body = shift_args(&inner.body, base as u32, &operand_dtypes(b, &inner.ops));
    let args = operand_dtypes(b, ops)
        .iter()
        .enumerate()
        .map(|(j, d)| match j.cmp(&slot) {
            std::cmp::Ordering::Equal => body.clone(),
            std::cmp::Ordering::Less => ScalarExpr::arg(j as u32, *d),
            std::cmp::Ordering::Greater => ScalarExpr::arg(j as u32 - 1, *d),
        })
        .collect();
    let mut retained: Vec<Operand> = Vec::with_capacity(base + inner.ops.len());
    retained.extend(
        ops.iter()
            .enumerate()
            .filter(|(j, _)| *j != slot)
            .map(|(_, o)| o.clone()),
    );
    (retained, args)
}

/// A reduction nest, normalized out of whichever spelling the operand named
/// as [`map_view`] normalizes a map: a `Logical::Fold` reads as the
/// `Launch::Fold` fields `lower_fold` would mint, its lift as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FoldView {
    /// The id the operand named.
    pub id: Id,
    pub space: IndexSpace,
    pub axis: u32,
    pub vec_axes: SmallVec<[u32; 2]>,
    pub carrier: Carrier,
    pub acc: Dtype,
    pub post: SmallVec<[ScalarExpr; 4]>,
    pub ops: Vec<Operand>,
    pub sched: ScheduleDomain,
}

impl FoldView {
    /// The domain this nest's own expressions are written against.
    pub(crate) fn iter_space(&self) -> IndexSpace {
        self.space.iterated(&self.vec_axes)
    }

    /// The reduced axis's index in [`Self::iter_space`]: `vec_axes` is the
    /// contiguous block immediately before `axis`.
    pub(crate) fn reduced_iter_axis(&self) -> Option<u32> {
        self.axis
            .checked_sub(u32::try_from(self.vec_axes.len()).ok()?)
    }

    /// The output dims before the carrier axis.
    pub(crate) fn base_dims(&self) -> Dims {
        self.space.fold_out_dims(self.axis, &self.vec_axes)
    }
}

/// Read `id` as a reduction nest, in either spelling.
pub(crate) fn fold_view(b: &Builder<'_>, id: Id) -> Option<FoldView> {
    match b.node(id).op.clone() {
        Op::Launch(Launch::Fold {
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            ops,
            sched,
        }) => Some(FoldView {
            id,
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            ops,
            sched,
        }),
        Op::Logical(Logical::Fold {
            carrier,
            axis,
            acc,
            ins,
        }) => {
            let in_shape = b.facts_of(*ins.first()?).shape.clone();
            Some(FoldView {
                id,
                space: IndexSpace::new(in_shape.iter().copied()),
                axis,
                vec_axes: SmallVec::new(),
                post: (0..carrier.width())
                    .map(|i| ScalarExpr::arg(i as u32, acc))
                    .collect(),
                carrier,
                acc,
                ops: ins
                    .iter()
                    .map(|x| alias_operand_of(*x, &in_shape))
                    .collect(),
                sched: ScheduleDomain::Point,
            })
        }
        _ => None,
    }
}

/// Re-apply a chain of pure views over a rewritten base, innermost first.
/// Rebuilding the nodes rather than composing their specs keeps every relative
/// stride exactly as it was written.
pub(crate) fn rebuild_spine(b: &mut Builder<'_>, spine: &ViewSpine, base: Id) -> Option<Id> {
    let mut cur = base;
    for &v in &spine.views {
        let Op::Logical(Logical::Restride { specs, bounds, .. }) = b.node(v).op.clone() else {
            return None;
        };
        cur = b
            .add_logical(Logical::Restride {
                specs,
                bounds,
                x: cur,
            })
            .ok()?;
    }
    Some(cur)
}

/// Renumber `Arg(i)` to `Arg(i + by)` throughout `e`, given each argument's
/// element type.
pub(crate) fn shift_args(e: &ScalarExpr, by: u32, arg_dtypes: &[Dtype]) -> ScalarExpr {
    let args: Vec<ScalarExpr> = arg_dtypes
        .iter()
        .enumerate()
        .map(|(i, d)| ScalarExpr::arg(i as u32 + by, *d))
        .collect();
    e.compose(&args)
}

/// Element type each operand of `ops` presents to a scalar body.
pub(crate) fn operand_dtypes(b: &Builder<'_>, ops: &[Operand]) -> Vec<Dtype> {
    ops.iter().map(|o| b.facts_of(o.src).dtype).collect()
}

/// Apply a relative restride spec vector to a dense row-major input shape.
///
/// A symbolic offset or stride composes into a derived symbol
/// (`Dim + Dim`, `Dim * Dim`) the backends evaluate at dispatch. Returns
/// `None` only when a stride or offset is genuinely opaque (a row-major
/// stride past a symbolic axis, or overflow): a view of those cannot be
/// stated as one layout, and a rule must decline rather than invent one.
pub(crate) fn composed_layout(specs: &[StrideSpec], in_shape: &[Dim]) -> Option<Layout> {
    use crate::shape::OPAQUE_SYM;
    let in_strides = Layout::row_major_strides(in_shape);
    let opaque = |d: Dim| matches!(d, Dim::Sym(s) if s == OPAQUE_SYM);
    let mut shape: Vec<Dim> = Vec::with_capacity(specs.len());
    let mut strides: Vec<Dim> = Vec::with_capacity(specs.len());
    let mut offset = Dim::Const(0);
    for s in specs {
        shape.push(s.size);
        // A broadcast axis at offset 0 names no input axis at all: a scalar
        // widened to a vector has none to name.
        if s.multiplier == 0 && s.offset.known_eq(Dim::Const(0)) {
            strides.push(Dim::Const(0));
            continue;
        }
        let base = *in_strides.get(s.input_dim as usize)?;
        // Accumulated for every spec, including a stride-0 one: an axis being
        // broadcast says nothing about where in the input it starts.
        if !s.offset.known_eq(Dim::Const(0)) {
            offset = offset + s.offset * base;
            if opaque(offset) {
                return None;
            }
        }
        if s.multiplier == 0 {
            strides.push(Dim::Const(0));
            continue;
        }
        let stride = base * Dim::Const(u64::from(s.multiplier));
        if opaque(stride) {
            return None;
        }
        strides.push(stride);
    }
    Layout::from_parts(offset, &shape, &strides).ok()
}

/// The plain affine layout a whole view spine denotes over its base, or
/// `None` when the composition is not expressible as one stride vector.
///
/// Composing a chain substitutes each stage's strides into the next, so a
/// narrow → reshape → transpose spine collapses to
/// `offset + Σ stride_j · i_j` over the base buffer. Everything must be
/// const-decidable and every stage's bounds proof [`BoundsProof::Static`]:
/// a `RuntimeMask` view masks reads the composed layout could not, and
/// dropping a mask is a wrong value.
///
/// A spec that walks past its input axis's extent (an axis-merging reshape)
/// is only affine when the stage it reads is dense row-major from that axis
/// inward, so the composition declines elsewhere.
pub(crate) fn composed_spine_layout(
    b: &Builder<'_>,
    spine: &crate::egraph::ViewSpine,
) -> Option<Layout> {
    use crate::ir::logical::Logical;
    let base_shape = b.facts_of(spine.base).shape.clone();
    let mut shape: Vec<u64> = base_shape
        .iter()
        .map(|d| d.as_const())
        .collect::<Option<_>>()?;
    let mut strides: Vec<u64> = Layout::row_major_strides(&base_shape)
        .iter()
        .map(|d| d.as_const())
        .collect::<Option<_>>()?;
    let mut offset: u64 = 0;
    for view in &spine.views {
        let Op::Logical(Logical::Restride { specs, bounds, .. }) = &b.node(*view).op else {
            return None;
        };
        if *bounds != crate::shape::BoundsProof::Static {
            return None;
        }
        // Whether the *current* stage is one dense row-major block, which is
        // the only stage an axis-overrunning spec addresses correctly: there
        // the stage is a flat window and `k · multiplier · in_stride` is flat
        // addressing, whatever axis boundaries the walk crosses.
        let dense = {
            let mut want = 1u64;
            let mut ok = true;
            for i in (0..shape.len()).rev() {
                // An extent-1 axis is unobservable whatever its stride says.
                if shape[i] <= 1 {
                    continue;
                }
                if strides[i] != want {
                    ok = false;
                    break;
                }
                want = want.saturating_mul(shape[i]);
            }
            ok
        };
        let mut nshape: Vec<u64> = Vec::with_capacity(specs.len());
        let mut nstrides: Vec<u64> = Vec::with_capacity(specs.len());
        for s in specs {
            let idim = s.input_dim as usize;
            let in_ext = *shape.get(idim)?;
            let in_stride = *strides.get(idim)?;
            let size = s.size.as_const()?;
            let off = s.offset.as_const()?;
            offset = offset.checked_add(off.checked_mul(in_stride)?)?;
            if s.multiplier == 0 {
                nshape.push(size);
                nstrides.push(0);
                continue;
            }
            let span = u64::from(s.multiplier)
                .checked_mul(size.saturating_sub(1))?
                .checked_add(off)?;
            if span >= in_ext.max(1) && !dense {
                return None;
            }
            nshape.push(size);
            nstrides.push(in_stride.checked_mul(u64::from(s.multiplier))?);
        }
        shape = nshape;
        strides = nstrides;
    }
    let shape: Vec<Dim> = shape.into_iter().map(Dim::Const).collect();
    let strides: Vec<Dim> = strides.into_iter().map(Dim::Const).collect();
    Layout::from_parts(Dim::Const(offset), &shape, &strides).ok()
}

/// Whether a spec vector is the identity view of `in_shape`.
pub(crate) fn is_identity_specs(specs: &[StrideSpec], in_shape: &[Dim]) -> bool {
    specs.len() == in_shape.len()
        && specs.iter().enumerate().all(|(i, s)| {
            s.multiplier == 1
                && s.input_dim as usize == i
                && s.offset.known_eq(Dim::Const(0))
                && s.size.known_eq(in_shape[i])
        })
}
