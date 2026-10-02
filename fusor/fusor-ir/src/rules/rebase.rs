//! HOIST and RETARGET: one dependence query (is this operand invariant along the
//! reduction axis?) answered two ways. HOIST applies a [`crate::carrier::HOM_TABLE`] row
//! when the operand is not a fold over the same axis; RETARGET carries the reference
//! alongside when it is. Two rules, since the fired set is per `(RuleId, Id)`.

use crate::carrier::{
    Carrier, HOM_TABLE, HomRow, HomShape, RETARGET_TABLE, RetargetRow, SlotTy, commute_canon,
    is_total_on, map_args,
};
use crate::dtype::{Dtype, Splat};
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, IndexSpace, Launch, Operand};
use crate::ir::logical::{Logical, TiePolicy};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::{
    access_legal_in, alias_operand_of, composed_layout, fold_view, map_view, splice_args,
};
use crate::scalar::{BinOp, CmpOp, ScalarExpr, ScalarKind, UnOp};
use crate::shape::{Dim, Layout};
use smallvec::{SmallVec, smallvec};

rule!(
    HOIST,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = hoist,
);

rule!(
    RETARGET,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = retarget,
);

/// Whether the read `o` performs lands on the same element for every value of `axis`;
/// `None` when undecidable, and every caller declines.
fn invariant_along(o: &Operand, space: &IndexSpace, axis: u32) -> Option<bool> {
    let a = axis as usize;
    if a >= space.rank() {
        return None;
    }
    if !matches!(o.access, AccessPlan::Unflatten(_))
        && o.layout.rank() == space.rank()
        && o.layout
            .shape()
            .iter()
            .zip(&space.dims)
            .all(|(x, d)| x.known_eq(*d))
    {
        return match o.layout.strides()[a].as_const() {
            Some(0) => Some(true),
            Some(_) => Some(false),
            // A symbolic stride over a non-unit axis is undecidable.
            None => space.dims[a].known_eq(Dim::ONE).then_some(true),
        };
    }
    o.varies_along(space, axis).map(|v| !v)
}

/// The read an operand edge performs, with a single-node pure view spine composed into
/// the layout (the floor spells a broadcast as a `Restride`), plus the id it names.
pub(crate) fn effective(b: &Builder<'_>, o: &Operand, space: &IndexSpace) -> (Operand, Id) {
    let plain = || (o.clone(), o.src);
    if !matches!(o.access, AccessPlan::Alias) || !o.layout.is_contiguous() {
        return plain();
    }
    let spine = b.trace_pure_views(o.src);
    if spine.views.len() != 1 {
        return plain();
    }
    let Op::Logical(Logical::Restride { specs, .. }) = b.node(spine.views[0]).op.clone() else {
        return plain();
    };
    if specs.len() != space.rank()
        || !specs
            .iter()
            .zip(&space.dims)
            .all(|(s, d)| s.size.known_eq(*d))
    {
        return plain();
    }
    let base_shape = b.facts_of(spine.base).shape.clone();
    match composed_layout(&specs, &base_shape) {
        Some(layout) => (
            Operand {
                src: spine.base,
                layout,
                access: AccessPlan::Alias,
            },
            spine.base,
        ),
        None => plain(),
    }
}

/// Whether two edges read the same elements of the same value.
fn same_read(a: &Operand, b: &Operand) -> bool {
    if a.src != b.src || std::mem::discriminant(&a.access) != std::mem::discriminant(&b.access) {
        return false;
    }
    if a.layout == b.layout {
        return true;
    }
    match (a.address_map(), b.address_map()) {
        (Some(x), Some(y)) => x == y,
        _ => false,
    }
}

/// One step of an `accum`-endomorphism surround on the path from a lift's root down to
/// a folded subterm.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Peel {
    /// `y |-> y * s`, an `(R, +)`-endomorphism.
    Mul(ScalarExpr),
    /// `y |-> y / s`.
    Div(ScalarExpr),
    /// `y |-> y + s`, a `max`/`min`-endomorphism (translation), never an additive one.
    Add(ScalarExpr),
    /// `y |-> -y`.
    Neg,
}

impl Peel {
    /// The monoid this step acts through; steps commute only within one action.
    const fn action(&self) -> BinOp {
        match self {
            Self::Mul(_) | Self::Div(_) | Self::Neg => BinOp::Mul,
            Self::Add(_) => BinOp::Add,
        }
    }

    /// Apply this step, dropping a multiplication by one and an addition of zero.
    fn apply(&self, y: ScalarExpr) -> ScalarExpr {
        match self {
            Self::Mul(s) => {
                if is_lit_value(&y, 1.0) {
                    s.clone()
                } else if is_lit_value(s, 1.0) {
                    y
                } else {
                    ScalarExpr::bin(BinOp::Mul, y, s.clone())
                }
            }
            Self::Div(s) => {
                if is_lit_value(s, 1.0) {
                    y
                } else {
                    ScalarExpr::bin(BinOp::Div, y, s.clone())
                }
            }
            Self::Add(s) => {
                if is_lit_value(&y, 0.0) {
                    s.clone()
                } else if is_lit_value(s, 0.0) {
                    y
                } else {
                    ScalarExpr::bin(BinOp::Add, y, s.clone())
                }
            }
            Self::Neg => ScalarExpr::un(UnOp::Neg, y),
        }
    }
}

/// Which peel steps are endomorphisms of `accum`.
const fn peel_legal_in(accum: BinOp, p: &Peel) -> bool {
    matches!(
        (accum, p),
        (BinOp::Add, Peel::Mul(_) | Peel::Div(_) | Peel::Neg)
            | (BinOp::Max | BinOp::Min, Peel::Add(_))
    )
}

/// Peel the `accum`-linear surround off `e` down to the first subterm `hit` accepts,
/// `peels[0]` outermost. Every sibling must be free of the subterm.
fn linear_factor(
    e: &ScalarExpr,
    accum: BinOp,
    hit: &dyn Fn(&ScalarExpr) -> bool,
) -> Option<(Vec<Peel>, ScalarExpr)> {
    if hit(e) {
        return Some((Vec::new(), e.clone()));
    }
    let try_side = |child: &ScalarExpr, sibling: Option<&ScalarExpr>, step: Peel| {
        if !peel_legal_in(accum, &step) {
            return None;
        }
        if sibling.is_some_and(|s| contains(s, hit)) {
            return None;
        }
        let (mut rest, inner) = linear_factor(child, accum, hit)?;
        rest.insert(0, step);
        Some((rest, inner))
    };
    match e.kind() {
        ScalarKind::Bin {
            op: BinOp::Mul,
            a,
            b,
        } => try_side(a, Some(b), Peel::Mul(b.clone()))
            .or_else(|| try_side(b, Some(a), Peel::Mul(a.clone()))),
        ScalarKind::Bin {
            op: BinOp::Div,
            a,
            b,
        } => try_side(a, Some(b), Peel::Div(b.clone())),
        ScalarKind::Bin {
            op: BinOp::Add,
            a,
            b,
        } => try_side(a, Some(b), Peel::Add(b.clone()))
            .or_else(|| try_side(b, Some(a), Peel::Add(a.clone()))),
        ScalarKind::Un { op: UnOp::Neg, x } => try_side(x, None, Peel::Neg),
        _ => None,
    }
}

/// Rebuild `L(seed)` from a peel chain, outermost applied last.
fn apply_peels(peels: &[Peel], seed: ScalarExpr) -> ScalarExpr {
    peels.iter().rev().fold(seed, |acc, p| p.apply(acc))
}

fn contains(e: &ScalarExpr, pred: &dyn Fn(&ScalarExpr) -> bool) -> bool {
    let mut found = false;
    e.walk(&mut |e| found = found || pred(e));
    found
}

fn reads_arg(e: &ScalarExpr, i: u32) -> bool {
    contains(e, &|x| matches!(x.kind(), ScalarKind::Arg(j) if *j == i))
}

fn is_lit_value(e: &ScalarExpr, v: f64) -> bool {
    matches!(e.kind(), ScalarKind::Lit(l) if l.0.to_f64() == v)
}

/// Equal modulo commutation.
fn expr_eq(a: &ScalarExpr, b: &ScalarExpr) -> bool {
    a == b || commute_canon(a) == commute_canon(b)
}

/// Drop operand edges no lift reads, renumbering the rest; `None` when all are read.
/// An unread edge would be charged traffic the kernel never performs.
fn prune_operands(
    lifts: &[ScalarExpr],
    ops: &[Operand],
) -> Option<(SmallVec<[ScalarExpr; 4]>, Vec<Operand>)> {
    let mut used: Vec<u32> = Vec::new();
    for l in lifts {
        l.collect_args(&mut used);
    }
    used.retain(|i| (*i as usize) < ops.len());
    used.sort_unstable();
    if used.len() == ops.len() || used.is_empty() {
        return None;
    }
    let renumber = |i: u32| used.iter().position(|&u| u == i).map_or(i, |p| p as u32);
    Some((
        lifts.iter().map(|l| map_args(l, &renumber)).collect(),
        used.iter().map(|&i| ops[i as usize].clone()).collect(),
    ))
}

/// The binop a single-slot carrier accumulates with, modulo commutation. A `Vector`
/// slot is admitted: promotion changes width, not algebra.
fn single_slot_accum(c: &Carrier) -> Option<BinOp> {
    (c.width() == 1).then(|| slot_accum(c, 0)).flatten()
}

/// The binop slot `k` accumulates with, when its merge is the self-contained
/// `merge[k] = op(Arg(k), Arg(w + k))` and nothing else.
fn slot_accum(c: &Carrier, k: usize) -> Option<BinOp> {
    let w = c.width();
    let ScalarKind::Bin { op, a, b } = c.merge[k].kind() else {
        return None;
    };
    let (lhs, rhs) = (ScalarKind::Arg(k as u32), ScalarKind::Arg((w + k) as u32));
    let forward = a.kind() == &lhs && b.kind() == &rhs;
    let swapped = a.kind() == &rhs && b.kind() == &lhs;
    (forward || (swapped && op.is_commutative()))
        .then_some(*op)
        .filter(|op| Carrier::binop_identity(*op, c.identity[k].dtype()) == Some(c.identity[k]))
}

/// One matched application of a [`HomRow`]'s `h` inside a lift.
struct HMatch {
    row: &'static HomRow,
    /// The invariant side, over the fold's `Arg` numbering; `None` for a unary row.
    c: Option<ScalarExpr>,
    /// The subterm `h` was applied to.
    inner: ScalarExpr,
}

impl HMatch {
    /// Re-apply `h` to `y`, with the invariant side renumbered by `remap`.
    fn apply(&self, y: ScalarExpr, remap: &dyn Fn(&ScalarExpr) -> ScalarExpr) -> ScalarExpr {
        match (self.row.h, &self.c) {
            (HomShape::MulByLit, Some(c)) => ScalarExpr::bin(BinOp::Mul, y, remap(c)),
            (HomShape::DivByLit, Some(c)) => ScalarExpr::bin(BinOp::Div, y, remap(c)),
            (HomShape::AddInvariant, Some(c)) => ScalarExpr::bin(BinOp::Add, y, remap(c)),
            (HomShape::TotalMonotone(op) | HomShape::TotalAntitone(op), _) => ScalarExpr::un(op, y),
            _ => y,
        }
    }
}

/// Recognize one application of `row`'s `h` at the root of `e`.
fn match_h(
    e: &ScalarExpr,
    row: &'static HomRow,
    invariant: &dyn Fn(&ScalarExpr) -> bool,
) -> Option<HMatch> {
    let mk = |c: Option<ScalarExpr>, inner: &ScalarExpr| {
        Some(HMatch {
            row,
            c,
            inner: inner.clone(),
        })
    };
    match (row.h, e.kind()) {
        (
            HomShape::MulByLit,
            ScalarKind::Bin {
                op: BinOp::Mul,
                a,
                b,
            },
        ) => {
            for (c, inner) in [(a, b), (b, a)] {
                if admissible_scale(c, row, invariant) && !invariant(inner) {
                    return mk(Some(c.clone()), inner);
                }
            }
            None
        }
        (
            HomShape::DivByLit,
            ScalarKind::Bin {
                op: BinOp::Div,
                a,
                b,
            },
        ) => (admissible_scale(b, row, invariant) && !invariant(a))
            .then(|| mk(Some(b.clone()), a))
            .flatten(),
        (
            HomShape::AddInvariant,
            ScalarKind::Bin {
                op: BinOp::Add,
                a,
                b,
            },
        ) => {
            for (c, inner) in [(a, b), (b, a)] {
                if invariant(c) && !invariant(inner) && !is_lit_value(c, 0.0) {
                    return mk(Some(c.clone()), inner);
                }
            }
            None
        }
        (
            HomShape::TotalMonotone(op) | HomShape::TotalAntitone(op),
            ScalarKind::Un { op: got, x },
        ) if *got == op => {
            // A partial unary could turn a number into a NaN.
            is_total_on(op, x.dtype()).then(|| mk(None, x)).flatten()
        }
        _ => None,
    }
}

/// A literal that scales: neither zero (not invertible) nor one (a no-op).
fn is_scaling_lit(e: &ScalarExpr) -> bool {
    matches!(e.kind(), ScalarKind::Lit(l)
        if l.0.to_f64() != 0.0 && l.0.to_f64() != 1.0)
}

/// Whether this row's identity depends on the factor's sign: a negative scale swaps an
/// extremum, so those rows admit only a literal.
const fn sign_sensitive(row: &HomRow) -> bool {
    matches!(row.h, HomShape::MulByLit | HomShape::DivByLit)
        && matches!(row.from, BinOp::Max | BinOp::Min)
}

/// Whether `c` may be peeled out as this row's factor; a sign-sensitive row demands a
/// positive literal.
fn admissible_scale(c: &ScalarExpr, row: &HomRow, invariant: &dyn Fn(&ScalarExpr) -> bool) -> bool {
    if is_lit_value(c, 1.0) || is_lit_value(c, 0.0) {
        return false;
    }
    if sign_sensitive(row) {
        return is_scaling_lit(c) && matches!(c.kind(), ScalarKind::Lit(l) if l.0.to_f64() > 0.0);
    }
    invariant(c)
}

/// The monoid `h` acts through, or `None` (e.g. `exp`) when `h` admits no surround and
/// fires only at the root.
const fn h_action(h: HomShape) -> Option<BinOp> {
    match h {
        HomShape::MulByLit | HomShape::DivByLit | HomShape::TotalAntitone(UnOp::Neg) => {
            Some(BinOp::Mul)
        }
        HomShape::AddInvariant => Some(BinOp::Add),
        _ => None,
    }
}

/// The homomorphism theorem both ways: a lift `L(h(inner))` becomes `h` applied outside
/// `Fold{row.from}(L(inner))`, and a `post` `h(Arg(0))` moves into the lift. Both stay
/// live; each peel strictly shrinks the lift.
pub fn hoist(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Fold {
        carrier, acc, post, ..
    }) = &node.op
    else {
        return None;
    };
    if acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }
    // One slot only: a multi-slot merge couples its slots.
    if carrier.width() != 1 || post.len() != 1 {
        return None;
    }
    let accum = single_slot_accum(carrier)?;
    let outward = hoist_outward(b, id, node, f, accum);
    hoist_inward(b, id, node, f, accum).or(outward)
}

fn hoist_outward(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
    accum: BinOp,
) -> Option<Id> {
    let Op::Launch(
        op @ Launch::Fold {
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            ops,
            ..
        },
    ) = &node.op
    else {
        return None;
    };
    // The peeled factor is applied once outside the reduced and promoted axes.
    let outside: SmallVec<[u32; 4]> = std::iter::once(*axis)
        .chain(vec_axes.iter().copied())
        .collect();
    let reads: Vec<(Operand, Id)> = ops.iter().map(|o| effective(b, o, space)).collect();
    let inv: Vec<bool> = reads
        .iter()
        .map(|(o, _)| {
            outside
                .iter()
                .all(|a| invariant_along(o, space, *a) == Some(true))
        })
        .collect();
    let is_invariant = |e: &ScalarExpr| -> bool {
        if e.reads_index_of() {
            return false;
        }
        let mut used = Vec::new();
        e.collect_args(&mut used);
        used.iter()
            .all(|&i| inv.get(i as usize).copied() == Some(true))
    };

    // Peel greedily, recording each `h` to re-apply outside, outermost first.
    let mut lift = carrier.lift[0].clone();
    let mut cur = accum;
    let mut peeled: Vec<HMatch> = Vec::new();
    while let Some((peels, m)) = HOM_TABLE
        .iter()
        .filter(|row| row.to == cur)
        .filter(|row| row.exact_in_float || f.own().numeric.reassoc)
        .find_map(|row| {
            let (peels, matched) =
                linear_factor(&lift, cur, &|e| match_h(e, row, &is_invariant).is_some())?;
            // `L` must be an endomorphism of both monoids and commute with `h`;
            // an empty surround is `L = id`.
            if !peels.is_empty() {
                let action = h_action(row.h)?;
                if !peels
                    .iter()
                    .all(|p| peel_legal_in(row.from, p) && p.action() == action)
                {
                    return None;
                }
            }
            let m = match_h(&matched, row, &is_invariant)?;
            if m.c.as_ref().is_some_and(ScalarExpr::reads_index_of) {
                return None;
            }
            Some((peels, m))
        })
    {
        lift = apply_peels(&peels, m.inner.clone());
        cur = m.row.from;
        peeled.push(m);
    }
    if peeled.is_empty() {
        return None;
    }

    // Decide everything that can decline before minting anything.
    let base = if cur == accum {
        carrier.clone()
    } else {
        rebind_accum(carrier, cur, *acc)?
    };
    let (inner_lift, inner_ops) = match prune_operands(&[lift.clone()], ops) {
        Some((l, o)) => (l, o),
        None => (smallvec![lift], ops.clone()),
    };
    let inner_carrier = base.with_lift(inner_lift);

    // The outer map reads the fold (slot 0) plus the invariant operands the peeled
    // factors name, re-viewed at the fold's output space.
    let out_shape: Vec<Dim> = f.own().shape.to_vec();
    let mut projected: Vec<Operand> = Vec::new();
    let mut slot_of: Vec<(u32, u32)> = Vec::new();
    for m in &peeled {
        let Some(c) = &m.c else { continue };
        let mut used = Vec::new();
        c.collect_args(&mut used);
        for i in used {
            if slot_of.iter().any(|(j, _)| *j == i) {
                continue;
            }
            slot_of.push((i, projected.len() as u32 + 1));
            projected.push(project_operand(
                &reads[i as usize].0,
                space,
                &outside,
                carrier,
            )?);
        }
    }
    let remap = |e: &ScalarExpr| -> ScalarExpr {
        map_args(e, &|i| {
            slot_of
                .iter()
                .find(|(j, _)| *j == i)
                .map_or(i, |(_, slot)| *slot)
        })
    };
    let mut body = ScalarExpr::arg(0, *acc);
    for m in peeled.iter().rev() {
        body = m.apply(body, &remap);
    }
    let body = post[0].compose(&[body]);
    // A casting `post` would make the union a lie.
    if body.dtype() != f.own().dtype {
        return None;
    }

    // The inner fold: `cur` accumulation, identity `post`, no edge for the factor.
    let inner = b
        .add_launch(with_fold(op, inner_carrier, *acc, Some(inner_ops)))
        .ok()?;
    let mut outer_ops = vec![alias_operand_of(inner, &out_shape)];
    outer_ops.extend(projected);
    let outer = crate::rules::lower_floor::floor_map(
        b,
        IndexSpace::new(out_shape.iter().copied()),
        body,
        outer_ops,
    )?;
    b.union(id, outer).ok()
}

/// `h(Fold{from}(x)) == Fold{to}(Map{h}(x))` left to right: a closed `h` in `post`
/// moves into the lift.
fn hoist_inward(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
    accum: BinOp,
) -> Option<Id> {
    let Op::Launch(
        op @ Launch::Fold {
            carrier, acc, post, ..
        },
    ) = &node.op
    else {
        return None;
    };
    let closed = |e: &ScalarExpr| -> bool {
        let mut used = Vec::new();
        e.collect_args(&mut used);
        used.is_empty() && !e.reads_index_of()
    };
    let (row, m) = HOM_TABLE
        .iter()
        .filter(|row| row.from == accum && row.to != accum)
        .filter(|row| row.exact_in_float || f.own().numeric.reassoc)
        .find_map(|row| {
            let m = match_h(&post[0], row, &closed)?;
            (m.inner.kind() == &ScalarKind::Arg(0)).then_some((row, m))
        })?;
    let pushed = rebind_accum(carrier, row.to, *acc)?
        .with_lift([m.apply(carrier.lift[0].clone(), &|c| c.clone())]);
    let alt = b.add_launch(with_fold(op, pushed, *acc, None)).ok()?;
    b.union(id, alt).ok()
}

/// `fold` accumulating through `carrier` with an identity `post`, reading `ops` if given.
fn with_fold(fold: &Launch, carrier: Carrier, acc: Dtype, ops: Option<Vec<Operand>>) -> Launch {
    let mut out = fold.clone();
    if let Launch::Fold {
        carrier: c,
        post,
        ops: o,
        ..
    } = &mut out
    {
        *c = carrier;
        *post = smallvec![ScalarExpr::arg(0, acc)];
        if let Some(ops) = ops {
            *o = ops;
        }
    }
    out
}

/// The same single-slot carrier accumulating with `op`, keeping slot shape and tie.
fn rebind_accum(c: &Carrier, op: BinOp, acc: Dtype) -> Option<Carrier> {
    Some(Carrier {
        slots: c.slots.clone(),
        identity: smallvec![Carrier::binop_identity(op, acc)?],
        lift: c.lift.clone(),
        merge: smallvec![ScalarExpr::bin(
            op,
            ScalarExpr::arg(0, acc),
            ScalarExpr::arg(1, acc)
        )],
        associative: op.is_associative(),
        tie: matches!(op, BinOp::Max | BinOp::Min).then(|| c.tie.unwrap_or(TiePolicy::SplitEvenly)),
    })
}

/// An operand restated as a plain strided read over `space`; an `Unflatten` with one
/// sub-axis per axis is a stride vector.
pub(crate) fn as_alias_over(o: &Operand, space: &IndexSpace) -> Option<Operand> {
    if matches!(o.access, AccessPlan::Alias) && o.layout.rank() == space.rank() {
        return Some(o.clone());
    }
    let AccessPlan::Unflatten(map) = &o.access else {
        return None;
    };
    if map.rank() != space.rank() || !map.is_affine() {
        return None;
    }
    let mut shape: Vec<Dim> = Vec::with_capacity(space.rank());
    let mut strides: Vec<Dim> = Vec::with_capacity(space.rank());
    for (g, d) in map.groups.iter().zip(&space.dims) {
        let s = g.sub_axes[0];
        if u64::from(s.extent) != d.as_const()? {
            return None;
        }
        shape.push(*d);
        strides.push(Dim::Const(u64::from(s.stride)));
    }
    Some(Operand {
        src: o.src,
        layout: Layout::from_parts(o.layout.offset(), &shape, &strides).ok()?,
        access: AccessPlan::Alias,
    })
}

/// A read over the full `space` restated over the iteration space; `None` when it
/// varies along a promoted axis.
pub(crate) fn on_iter_space(o: &Operand, space: &IndexSpace, vec_axes: &[u32]) -> Option<Operand> {
    if vec_axes.is_empty() {
        return Some(o.clone());
    }
    let o = as_alias_over(o, space)?;
    if vec_axes.iter().any(|a| {
        o.layout
            .strides()
            .get(*a as usize)
            .is_some_and(|s| s.as_const() != Some(0))
    }) {
        return None;
    }
    drop_axes(&o, vec_axes, None)
}

/// An alias with the axes in `drop` removed, then a stride-0 `lanes` axis appended.
fn drop_axes(o: &Operand, drop: &[u32], lanes: Option<Dim>) -> Option<Operand> {
    let (mut shape, mut strides): (Vec<Dim>, Vec<Dim>) = o
        .layout
        .shape()
        .iter()
        .zip(o.layout.strides())
        .enumerate()
        .filter(|(i, _)| !drop.contains(&(*i as u32)))
        .map(|(_, (d, s))| (*d, *s))
        .unzip();
    if let Some(lanes) = lanes {
        shape.push(lanes);
        strides.push(Dim::Const(0));
    }
    Some(Operand {
        src: o.src,
        layout: Layout::from_parts(o.layout.offset(), &shape, &strides).ok()?,
        access: AccessPlan::Alias,
    })
}

fn project_operand(
    o: &Operand,
    space: &IndexSpace,
    drop: &[u32],
    carrier: &Carrier,
) -> Option<Operand> {
    // Alias only: restating an affine `Unflatten` here miscomputes; fix such an edge
    // at its mint.
    if !matches!(o.access, AccessPlan::Alias) {
        return None;
    }
    drop_axes(&as_alias_over(o, space)?, drop, carrier.out_dim()?)
}

/// Hole indices no operand can occupy, for reading a row's action back out.
const HOLE_D: u32 = u32::MAX - 1;
const HOLE_V: u32 = u32::MAX;

/// Read `T` out of a [`RetargetRow`] by applying it to two holes: the monoid it acts
/// through and `h(delta)` over [`HOLE_D`]. Declines anything but `v (+) f(delta)`.
fn row_action(row: &RetargetRow, dtype: Dtype) -> Option<(BinOp, ScalarExpr)> {
    let d = ScalarExpr::arg(HOLE_D, dtype);
    let v = ScalarExpr::arg(HOLE_V, dtype);
    let t = (row.retarget)(&d, &v, dtype);
    let ScalarKind::Bin { op, a, b } = t.kind() else {
        return None;
    };
    for (side, other) in [(a, b), (b, a)] {
        if side.kind() == &ScalarKind::Arg(HOLE_V) && !reads_arg(other, HOLE_V) {
            return Some((*op, other.clone()));
        }
    }
    None
}

/// Whether `e` is `template[HOLE_D := (u - Arg(ref_arg))]` for one shared `u`.
fn match_shift(
    e: &ScalarExpr,
    template: &ScalarExpr,
    ref_arg: u32,
    bound: &mut Option<ScalarExpr>,
) -> bool {
    if template.kind() == &ScalarKind::Arg(HOLE_D) {
        let ScalarKind::Bin {
            op: BinOp::Sub,
            a,
            b,
        } = e.kind()
        else {
            return false;
        };
        if b.kind() != &ScalarKind::Arg(ref_arg) {
            return false;
        }
        return match bound {
            Some(prev) => expr_eq(prev, a),
            None => {
                *bound = Some(a.clone());
                true
            }
        };
    }
    match (e.kind(), template.kind()) {
        (ScalarKind::Un { op: o1, x: x1 }, ScalarKind::Un { op: o2, x: x2 }) if o1 == o2 => {
            match_shift(x1, x2, ref_arg, bound)
        }
        (
            ScalarKind::Bin {
                op: o1,
                a: a1,
                b: b1,
            },
            ScalarKind::Bin {
                op: o2,
                a: a2,
                b: b2,
            },
        ) if o1 == o2 => {
            let mut probe = bound.clone();
            if match_shift(a1, a2, ref_arg, &mut probe) && match_shift(b1, b2, ref_arg, &mut probe)
            {
                *bound = probe;
                return true;
            }
            if !o1.is_commutative() {
                return false;
            }
            let mut probe = bound.clone();
            if match_shift(a1, b2, ref_arg, &mut probe) && match_shift(b1, a2, ref_arg, &mut probe)
            {
                *bound = probe;
                return true;
            }
            false
        }
        _ => expr_eq(e, template),
    }
}

/// The reference fold an invariant operand names, when it is one.
struct RefFold {
    id: Id,
    carrier: Carrier,
    /// The reference's element expression over the reading fold's operands.
    lift: ScalarExpr,
}

/// A reduction-carried dependence on another reduction over the same axis, discharged
/// by carrying the reference alongside and rescaling. Fires only where the program
/// already computes the reference, and redirects only this reader.
pub fn retarget(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        ..
    }) = &node.op
    else {
        return None;
    };
    // Retargeting reassociates and inserts a rounding step per merge.
    if !f.own().numeric.reassoc || acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }
    if post.len() != carrier.width() || carrier.width() == 0 {
        return None;
    }

    // The reference is a nest over the iteration space; compare operands after
    // projecting onto it.
    let iter = space.iterated(vec_axes);
    let iter_axis = *axis - vec_axes.len() as u32;
    let reads: Vec<(Operand, Id)> = ops.iter().map(|o| effective(b, o, space)).collect();
    let proj: Vec<(Operand, Id)> = reads
        .iter()
        .map(|(o, i)| {
            (
                on_iter_space(o, space, vec_axes).unwrap_or_else(|| o.clone()),
                *i,
            )
        })
        .collect();
    for r in 0..ops.len() {
        // A read varying along a promoted axis cannot name such a reference.
        if on_iter_space(&reads[r].0, space, vec_axes).is_none() {
            continue;
        }
        if invariant_along(&proj[r].0, &iter, iter_axis) != Some(true) {
            continue;
        }
        let Some(reference) = reference_fold(b, reads[r].1, &iter, iter_axis, &proj) else {
            continue;
        };
        // Acyclicity: no kept operand may be the reference.
        if (0..ops.len()).any(|i| i != r && b.class_members(reads[i].1).contains(&reference.id)) {
            continue;
        }
        if let Some(hit) = mint_retarget(b, id, node, r, &reference) {
            return Some(hit);
        }
    }
    None
}

/// Whether `src` is a fold over the same axis of the same reads.
fn reference_fold(
    b: &Builder<'_>,
    src: Id,
    space: &IndexSpace,
    axis: u32,
    reader: &[(Operand, Id)],
) -> Option<RefFold> {
    for cand in b.class_members(src) {
        // A `post` hides `rho` from readers.
        let Some(v) = fold_view(b, cand).filter(|v| {
            v.vec_axes.is_empty() && v.post.len() == 1 && v.post[0].kind() == &ScalarKind::Arg(0)
        }) else {
            continue;
        };
        if v.axis != axis || v.carrier.width() != 1 {
            continue;
        }
        if v.carrier.slots[0] != SlotTy::Scalar || !v.carrier.associative {
            continue;
        }
        if v.space.dims != space.dims {
            continue;
        }
        let Some(lift) = common_basis(b, &v, reader) else {
            continue;
        };
        return Some(RefFold {
            id: cand,
            lift,
            carrier: v.carrier,
        });
    }
    None
}

/// Producers `common_basis` substitutes before giving up; bounds work only.
const MAX_EXPANSIONS: usize = 8;

/// The reference's lift rewritten over the reader's operands, substituting elementwise
/// producers as fusion does; `None` without a common basis.
fn common_basis(
    b: &Builder<'_>,
    v: &crate::rules::FoldView,
    reader: &[(Operand, Id)],
) -> Option<ScalarExpr> {
    let mut lift = v.carrier.lift[0].clone();
    let mut ops = v.ops.clone();
    for _ in 0..MAX_EXPANSIONS {
        let place = |o: &Operand| -> Option<usize> {
            let (eff, _) = effective(b, o, &v.space);
            reader.iter().position(|(x, _)| same_read(x, &eff))
        };
        if let Some(remap) = ops.iter().map(place).collect::<Option<Vec<_>>>() {
            return Some(map_args(&lift, &|i| {
                remap.get(i as usize).map_or(i, |p| *p as u32)
            }));
        }
        // Substitute one unplaced producer, exactly as `splice` does.
        let slot = ops.iter().position(|o| place(o).is_none())?;
        if !matches!(ops[slot].access, AccessPlan::Alias) {
            return None;
        }
        let inner = map_view(b, ops[slot].src)?;
        if !v.space.covers(&inner.space)
            || !inner
                .ops
                .iter()
                .all(|o| access_legal_in(&o.access, &v.space))
        {
            return None;
        }
        let (mut next, args) = splice_args(b, &ops, slot, &inner);
        lift = lift.compose(&args);
        next.extend(inner.ops.iter().cloned());
        ops = next;
    }
    None
}

fn float_magnitude_bits(value: ScalarExpr) -> ScalarExpr {
    ScalarExpr::bin(
        BinOp::BitAnd,
        ScalarExpr::bitcast(Dtype::U32, ScalarExpr::cast(Dtype::F32, value)),
        ScalarExpr::lit(Splat::U32(0x7fffffff)),
    )
}

fn mint_retarget(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    r: usize,
    reference: &RefFold,
) -> Option<Id> {
    let Op::Launch(
        op @ Launch::Fold {
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            ops,
            sched,
        },
    ) = &node.op
    else {
        return None;
    };
    let w = carrier.width();
    let ref_arg = r as u32;

    for row in RETARGET_TABLE {
        // Every slot must accumulate with the row's binop.
        if (0..w).any(|k| slot_accum(carrier, k) != Some(row.accum)) {
            continue;
        }
        let stat = (row.stat)(*acc);
        if stat.width() != 1
            || stat.slots != reference.carrier.slots
            || stat.identity != reference.carrier.identity
            || !expr_eq(&stat.merge[0], &reference.carrier.merge[0])
        {
            continue;
        }
        let Some((action, template)) = row_action(row, *acc) else {
            continue;
        };
        let Some(seed) = Carrier::binop_identity(action, *acc) else {
            continue;
        };

        // One row must cover every slot after peeling: one rescale factor for all.
        let mut bound: Option<ScalarExpr> = None;
        let mut lifts: SmallVec<[ScalarExpr; 4]> = SmallVec::new();
        let mut ok = true;
        for k in 0..w {
            let hit = |e: &ScalarExpr| {
                let mut probe = None;
                match_shift(e, &template, ref_arg, &mut probe)
            };
            let Some((peels, matched)) = linear_factor(&carrier.lift[k], row.accum, &hit) else {
                ok = false;
                break;
            };
            // `T` and every peeled `L` must act through one monoid to commute.
            if !peels.iter().all(|p| p.action() == action)
                || !match_shift(&matched, &template, ref_arg, &mut bound)
            {
                ok = false;
                break;
            }
            // `h(e) = id`, so an element enters as `L(identity)`.
            lifts.push(apply_peels(&peels, ScalarExpr::lit(seed)));
        }
        if !ok || lifts.iter().any(|e| reads_arg(e, ref_arg)) {
            continue;
        }
        // The reference must be exactly what this fold subtracts.
        let u = bound?;
        if !expr_eq(&u, &reference.lift) {
            continue;
        }
        let invalid = match row.accum {
            BinOp::Add => ScalarExpr::cast(
                *acc,
                ScalarExpr::bitcast(Dtype::F32, ScalarExpr::lit(Splat::U32(0x7fc00000))),
            ),
            BinOp::Max => ScalarExpr::lit(Carrier::binop_identity(BinOp::Max, *acc)?),
            _ => continue,
        };
        let nan_input = ScalarExpr::cmp(
            CmpOp::Gt,
            float_magnitude_bits(u.clone()),
            ScalarExpr::lit(Splat::U32(0x7f800000)),
        );
        for lift in &mut lifts {
            *lift = ScalarExpr::select(nan_input.clone(), invalid.clone(), lift.clone());
        }

        // The joint fold reads the reference's own inputs: acyclic by construction.
        let drop = |i: u32| if i > ref_arg { i - 1 } else { i };
        let new_ops: Vec<Operand> = ops
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != r)
            .map(|(_, o)| o.clone())
            .collect();
        if new_ops.is_empty() {
            continue;
        }
        // Every joint lift, so an edge only the reference read is pruned once.
        let all: SmallVec<[ScalarExpr; 4]> = std::iter::once(map_args(&reference.lift, &drop))
            .chain(lifts.iter().map(|e| map_args(e, &drop)))
            .collect();
        let (all, new_ops) = match prune_operands(&all, &new_ops) {
            Some(pruned) => pruned,
            None => (all, new_ops),
        };
        let stat = Carrier {
            lift: smallvec![all[0].clone()],
            ..stat
        };
        let body = Carrier {
            slots: carrier.slots.clone(),
            identity: carrier.identity.clone(),
            lift: all[1..].iter().cloned().collect(),
            merge: carrier.merge.clone(),
            associative: carrier.associative,
            tie: carrier.tie,
        };
        let joint = Carrier::retarget(&stat, row, &body, 0)?;
        // Slot ranges count lanes: a `Vector` slot is many.
        let (lanes, body_lanes) = (joint.lanes()?, carrier.lanes()?);
        let sched = sched.with_fold_carrier(lanes, acc.byte_size(), b.caps())?;
        let Some(body_axis) = carrier.out_dim() else {
            continue;
        };
        // Free dims: `space` minus the reduced and promoted axes.
        let free = space.fold_out_dims(*axis, vec_axes);
        // Check both readbacks before minting so a decline leaves no orphan.
        if !view_expressible(&free, lanes, 1, body_lanes, body_axis)
            || !view_expressible(&free, lanes, 0, 1, None)
        {
            continue;
        }

        // A nonempty row with a nonfinite reference keeps the shifted result.
        let nonempty = match space.dims[*axis as usize] {
            Dim::Const(n) => ScalarExpr::lit(Splat::U32(u32::from(n != 0))),
            Dim::Sym(crate::shape::OPAQUE_SYM) => continue,
            Dim::Sym(sym) => ScalarExpr::cmp(
                CmpOp::Ne,
                ScalarExpr::uniform(sym, Dtype::U32),
                ScalarExpr::lit(Splat::U32(0)),
            ),
        };
        let invalid_reference = ScalarExpr::bin(
            BinOp::BitAnd,
            nonempty,
            ScalarExpr::cmp(
                CmpOp::Ge,
                float_magnitude_bits(ScalarExpr::arg(0, *acc)),
                ScalarExpr::lit(Splat::U32(0x7f800000)),
            ),
        );
        let finalized: Vec<_> = (0..w)
            .map(|i| {
                ScalarExpr::select(
                    invalid_reference.clone(),
                    invalid.clone(),
                    ScalarExpr::arg(i as u32 + 1, *acc),
                )
            })
            .collect();
        let joint_post: SmallVec<[ScalarExpr; 4]> = std::iter::once(ScalarExpr::arg(0, *acc))
            .chain(post.iter().map(|e| e.compose(&finalized)))
            .collect();

        let mut joint_fold = op.clone();
        if let Launch::Fold {
            carrier,
            post,
            ops,
            sched: s,
            ..
        } = &mut joint_fold
        {
            (*carrier, *post, *ops, *s) = (joint, joint_post, new_ops, sched);
        }
        let joint_id = b.add_launch(joint_fold).ok()?;

        let body_view = slot_view(b, joint_id, &free, lanes, 1, body_lanes, body_axis)?;
        let ref_view = slot_view(b, joint_id, &free, lanes, 0, 1, None)?;
        // Redirect only this reader; slot 0 cannot replace a shared reference.
        let _ = ref_view;
        return b.union(id, body_view).ok();
    }
    None
}

/// Whether lanes `[off, off + len)` of a joint fold's carrier axis read back as a
/// strided view.
fn view_expressible(free: &[Dim], lanes: u64, off: u64, len: u64, want_axis: Option<Dim>) -> bool {
    slot_layout(free, lanes, off, len, want_axis).is_some()
}

/// The `(shape, layout)` a slot readback reads through, if the range fits.
fn slot_layout(
    free: &[Dim],
    lanes: u64,
    off: u64,
    len: u64,
    want_axis: Option<Dim>,
) -> Option<(Vec<Dim>, Layout)> {
    if off.checked_add(len)? > lanes {
        return None;
    }
    let mut joint_shape: Vec<Dim> = free.to_vec();
    joint_shape.push(Dim::Const(lanes));
    let strides = Layout::row_major_strides(&joint_shape);

    let mut shape: Vec<Dim> = free.to_vec();
    let mut view: Vec<Dim> = strides[..free.len()].to_vec();
    match want_axis {
        Some(d) if d.known_eq(Dim::Const(len)) => {
            shape.push(d);
            view.push(Dim::Const(1));
        }
        Some(_) => return None,
        None if len == 1 => {}
        None => return None,
    }
    let layout = Layout::from_parts(Dim::Const(off), &shape, &view).ok()?;
    Some((shape, layout))
}

fn slot_view(
    b: &mut Builder<'_>,
    joint: Id,
    free: &[Dim],
    lanes: u64,
    off: u64,
    len: u64,
    want_axis: Option<Dim>,
) -> Option<Id> {
    let (shape, layout) = slot_layout(free, lanes, off, len, want_axis)?;
    let dtype = b.facts_of(joint).dtype;
    crate::rules::lower_floor::floor_alias_map(b, joint, layout, &shape, dtype)
}
