//! Every level-generic rewrite rule, and the one table the driver is handed.
//!
//! Guards may read only [`crate::egraph::Facts`] — legality, never
//! profitability.
//!
//! Rule order carries no semantics; the fixed order below exists only for
//! reproducibility.

pub mod algebra;
pub mod fusion;
pub mod layout;
pub mod lower_floor;
pub mod promote;
pub mod rebase;
pub mod sink;
pub mod specialize;
pub mod tuple;

use crate::dtype::Dtype;
use crate::egraph::{Builder, Id, Rule};
use crate::ir::Op;
use crate::ir::logical::Logical;
use crate::ir::launch::{AccessPlan, IndexSpace, Launch, Operand};
use crate::scalar::ScalarExpr;
use crate::shape::{Dim, Layout, StrideSpec};

/// Position of a rule in whatever slice was handed to the driver. `RuleId`
/// is positional, not global: a target concatenates [`CORE_RULES`] with its
/// own `Target::rules()` and the driver indexes the concatenation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RuleId(pub u16);

/// Every rule `fusor2-ir` owns, in a fixed order. A target's `rules()` is
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
    fusion::FORM_KREGION,
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
];

/// Look a core rule up by the name its `rule!` declaration stringified.
pub fn rule_id(name: &str) -> Option<RuleId> {
    CORE_RULES
        .iter()
        .position(|r| r.name == name)
        .map(|i| RuleId(i as u16))
}

/// The core rule at `id`. Panics when `id` is out of range, which can only
/// happen if a caller mixes ids minted against a different slice.
pub fn rule(id: RuleId) -> &'static Rule {
    &CORE_RULES[id.0 as usize]
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
        Op::Logical(Logical::Map { expr, ins, outs }) if outs == 1 => {
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
/// Returns `None` when a stride or offset is not decidable.
pub(crate) fn composed_layout(specs: &[StrideSpec], in_shape: &[Dim]) -> Option<Layout> {
    let in_strides = Layout::row_major_strides(in_shape);
    let mut shape: Vec<Dim> = Vec::with_capacity(specs.len());
    let mut strides: Vec<Dim> = Vec::with_capacity(specs.len());
    let mut offset: u64 = 0;
    for s in specs {
        shape.push(s.size);
        let base = in_strides.get(s.input_dim as usize)?.as_const()?;
        // Accumulated for every spec, including a stride-0 one: an axis being
        // broadcast says nothing about where in the input it starts.
        offset = offset.checked_add(s.offset.as_const()?.checked_mul(base)?)?;
        if s.multiplier == 0 {
            strides.push(Dim::Const(0));
            continue;
        }
        strides.push(Dim::Const(base.checked_mul(u64::from(s.multiplier))?));
    }
    Layout::from_parts(Dim::Const(offset), &shape, &strides).ok()
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
