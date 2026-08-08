//! HOIST and RETARGET, two rules sharing one dependence query: is an operand of
//! a reduction nest invariant along the reduced axis? The query is decided on
//! [`Operand::address_map`], with a single pure view collapsed into the layout
//! first, not on a syntactic broadcast test.
//!
//! HOIST applies when the invariant operand is not derived from a fold over the
//! same axis. For `h` a monoid homomorphism from `(+, e+)` to `(x, ex)`,
//! `h(Fold{+, a}(x)) == Fold{x, a}(Map{h}(x))`. Both directions are minted and
//! cost decides; the pairs are data in [`crate::carrier::HOM_TABLE`].
//!
//! RETARGET applies when it is: a reduction-carried dependence on another
//! reduction over the same axis is discharged by carrying the reference
//! alongside and rescaling, per [`crate::carrier::RETARGET_TABLE`] and
//! [`crate::carrier::Carrier::retarget`].
//!
//! They are two separate [`Rule`](crate::egraph::Rule) entries because the
//! driver's fired set is per `(RuleId, Id)`: one merged rule could fire at most
//! once per node, making the answer it declined unreachable.

use crate::carrier::{
    Carrier, HOM_TABLE, HomRow, HomShape, RETARGET_TABLE, RetargetRow, SlotTy, is_total_on,
    map_args, probes_for,
};
use crate::dtype::Dtype;
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::level0::{L0, TiePolicy};
use crate::ir::level1::{AccessPlan, IndexSpace, L1, Operand};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::{
    access_legal_in, alias_operand_of, composed_layout, map_view, operand_dtypes, shift_args,
};
use crate::scalar::{BinOp, ScalarExpr, ScalarKind, UnOp, splat_f32};
use crate::shape::{Dim, Layout};
use smallvec::{SmallVec, smallvec};

rule!(
    HOIST,
    level = Level::L1,
    head = OpTag::KFold,
    tag = RuleTag::Additive,
    apply = hoist,
);

rule!(
    RETARGET,
    level = Level::L1,
    head = OpTag::KFold,
    tag = RuleTag::Additive,
    apply = retarget,
);

/// The flat-index window `[lo, hi)` that `axis` owns in a row-major walk of
/// `space`. `None` when a dim below `axis` is symbolic or the product
/// overflows.
fn axis_window(space: &IndexSpace, axis: u32) -> Option<(u64, u64)> {
    let a = axis as usize;
    let divisor = space
        .dims
        .get(a + 1..)?
        .iter()
        .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))?;
    let modulus = space.dims.get(a)?.as_const()?;
    Some((divisor, divisor.checked_mul(modulus)?))
}

/// Does the read `o` performs land on the same element for every value of
/// `axis`? `Some(true)` proves invariance, `Some(false)` variance, `None`
/// undecidable, and every caller declines on `None`.
///
/// A layout stated axis-for-axis against the space answers from its own stride
/// along `axis`, which works under a `Dim::Sym` extent. Otherwise the divmod
/// form answers: no [`AddressTerm`](crate::ir::level1::AddressTerm) overlapping
/// the axis's `(divisor, modulus)` window may carry a nonzero stride.
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
            // A symbolic stride over a non-unit axis is a real read at one
            // binding and not at another.
            None => space.dims[a].known_eq(Dim::ONE).then_some(true),
        };
    }
    let (lo, hi) = axis_window(space, axis)?;
    let map = o.address_map()?;
    Some(!map.terms.iter().any(|t| {
        let t_lo = u64::from(t.divisor);
        let t_hi = t_lo.saturating_mul(u64::from(t.modulus));
        t.stride != 0 && t_lo < hi && lo < t_hi
    }))
}

/// The read an operand edge actually performs, with one pure view collapsed
/// into the layout, plus the id that read ultimately names.
///
/// The lowering floor spells a broadcast as a `Restride` node and gives the
/// consuming edge a dense layout, so composing the view's spec vector into the
/// layout is what makes the dependence query see the read. Only a single-node
/// spine is collapsed: composing a multi-node spec vector is decidable only
/// once every extent is known.
pub(crate) fn effective(b: &Builder<'_>, o: &Operand, space: &IndexSpace) -> (Operand, Id) {
    let plain = || (o.clone(), o.src);
    if !matches!(o.access, AccessPlan::Alias) || !o.layout.is_contiguous() {
        return plain();
    }
    let spine = b.trace_pure_views(o.src);
    if spine.views.len() != 1 {
        return plain();
    }
    let Op::L0(L0::Restride { specs, .. }) = b.node(spine.views[0]).op.clone() else {
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

/// Two edges that read the same elements of the same value. Compared on `src`
/// plus the index map, so two different-but-equal spellings still match.
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

/// One step of an `accum`-endomorphism surround, as it appears on the path from
/// a lift's root down to a folded subterm. The subterm may sit anywhere on such
/// a path, not only at the lift's root.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Peel {
    /// `y |-> y * s`, an `(R, +)`-endomorphism.
    Mul(ScalarExpr),
    /// `y |-> y / s`.
    Div(ScalarExpr),
    /// `y |-> y + s`, an `(R, max)`- and `(R, min)`-endomorphism (translation),
    /// and never an additive one.
    Add(ScalarExpr),
    /// `y |-> -y`.
    Neg,
}

impl Peel {
    /// The monoid this step acts through: `Mul` for a scalar multiplication,
    /// `Add` for a translation. Two steps commute iff their actions agree.
    const fn action(&self) -> BinOp {
        match self {
            Self::Mul(_) | Self::Div(_) | Self::Neg => BinOp::Mul,
            Self::Add(_) => BinOp::Add,
        }
    }

    /// Apply this step, dropping a multiplication by one and an addition of
    /// zero. A retargeted body lift is this chain applied to the action's
    /// identity, so an un-simplified `1 * v` costs a multiply per element per
    /// lane.
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

/// Peel the `accum`-linear surround off `e` on the path to the first subterm
/// `hit` accepts. `peels[0]` is the outermost step.
///
/// The sibling of every step must be free of the subterm, and
/// `|peels| + |inner| < |e|` strictly.
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

fn contains<P: Fn(&ScalarExpr) -> bool + ?Sized>(e: &ScalarExpr, pred: &P) -> bool {
    if pred(e) {
        return true;
    }
    let mut found = false;
    e.for_each_child(|c| found = found || contains(c, pred));
    found
}

fn reads_arg(e: &ScalarExpr, i: u32) -> bool {
    contains(e, &|x| matches!(x.kind(), ScalarKind::Arg(j) if *j == i))
}

fn reads_any_index(e: &ScalarExpr) -> bool {
    contains(e, &|x| matches!(x.kind(), ScalarKind::IndexOf(_)))
}

fn arg_indices(e: &ScalarExpr, out: &mut Vec<u32>) {
    if let ScalarKind::Arg(i) = e.kind()
        && !out.contains(i)
    {
        out.push(*i);
    }
    e.for_each_child(|c| arg_indices(c, out));
}

fn is_lit_value(e: &ScalarExpr, v: f32) -> bool {
    matches!(e.kind(), ScalarKind::Lit(l) if splat_f32(l.0) == v)
}

/// Sort every commutative binop's children into a fixed order, so a guard
/// spelled `Add(Arg(k), Arg(w + k))` still matches `Add(Arg(w + k), Arg(k))`.
///
/// `ScalarExpr` does not canonicalize on construction and the e-graph
/// canonicalizes only `Op::Union` children.
fn canon(e: &ScalarExpr) -> ScalarExpr {
    match e.kind() {
        ScalarKind::Bin { op, a, b } => {
            let (a, b) = (canon(a), canon(b));
            if op.is_commutative() && b.structural_hash() < a.structural_hash() {
                ScalarExpr::bin(*op, b, a)
            } else {
                ScalarExpr::bin(*op, a, b)
            }
        }
        _ => e.map_children(|c| canon(c)),
    }
}

fn expr_eq(a: &ScalarExpr, b: &ScalarExpr) -> bool {
    a == b || canon(a) == canon(b)
}

/// Drop operand edges no lift reads any more, renumbering what is left.
/// `None` when every operand is still read.
///
/// An operand list that kept a discharged edge would price the hoisted form at
/// the unhoisted form's bandwidth, and the law would never be selected.
fn prune_operands(
    lifts: &[ScalarExpr],
    ops: &[Operand],
) -> Option<(SmallVec<[ScalarExpr; 4]>, Vec<Operand>)> {
    let mut used: Vec<u32> = Vec::new();
    for l in lifts {
        arg_indices(l, &mut used);
    }
    used.retain(|i| (*i as usize) < ops.len());
    used.sort_unstable();
    if used.len() == ops.len() || used.is_empty() {
        return None;
    }
    let renumber = |i: u32| {
        used.iter()
            .position(|&u| u == i)
            .map_or(i, |p| p as u32)
    };
    Some((
        lifts.iter().map(|l| map_args(l, &renumber)).collect(),
        used.iter().map(|&i| ops[i as usize].clone()).collect(),
    ))
}

/// The binop a single-slot carrier accumulates with, modulo commutation.
///
/// Unlike [`Carrier::kind`] this admits a `Vector` slot: promotion changes a
/// slot's width, never its algebra.
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
    (forward || (swapped && op.is_commutative())).then_some(*op)
}

/// One matched application of a [`HomRow`]'s `h` inside a lift.
struct HMatch {
    row: &'static HomRow,
    /// The invariant side, over the *fold's* `Arg` numbering. `None` for a
    /// unary row.
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
            (HomShape::TotalMonotone(op) | HomShape::TotalAntitone(op), _) => {
                ScalarExpr::un(op, y)
            }
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
            // A monotone row over a unary that is partial on the operand dtype
            // can turn a number into a NaN.
            is_total_on(op, x.dtype()).then(|| mk(None, x)).flatten()
        }
        _ => None,
    }
}

/// A literal that scales: not zero (which is not invertible) and not one (which
/// would make the rewrite a no-op that still costs a node).
fn is_scaling_lit(e: &ScalarExpr) -> bool {
    matches!(e.kind(), ScalarKind::Lit(l)
        if splat_f32(l.0) != 0.0 && splat_f32(l.0) != 1.0)
}

/// Does the identity this row states depend on the sign of its factor?
///
/// Scaling an extremum by a negative number swaps the extremum, and an
/// axis-invariant layout proves invariance rather than positivity, so those
/// rows admit only a `Lit`, whose sign is decidable. Scaling an additive fold
/// is sign-blind — `sum(x * c) == sum(x) * c` holds for every `c` — so any
/// axis-invariant expression is admissible there.
const fn sign_sensitive(row: &HomRow) -> bool {
    matches!(row.h, HomShape::MulByLit | HomShape::DivByLit)
        && matches!(row.from, BinOp::Max | BinOp::Min)
}

/// May `c` be peeled out as this row's factor?
///
/// The sign-sensitive branch demands a positive literal, not merely a literal:
/// `max(x * -2) == min(x) * -2`, not `max(x) * -2`. [`HOM_TABLE`] holds no
/// extremum scaling row, so the branch guards rows added later.
fn admissible_scale(
    c: &ScalarExpr,
    row: &HomRow,
    invariant: &dyn Fn(&ScalarExpr) -> bool,
) -> bool {
    // A no-op factor is still a node the outer map has to evaluate.
    if is_lit_value(c, 1.0) || is_lit_value(c, 0.0) {
        return false;
    }
    if sign_sensitive(row) {
        return is_scaling_lit(c) && matches!(c.kind(), ScalarKind::Lit(l) if splat_f32(l.0) > 0.0);
    }
    invariant(c)
}

/// The monoid `h` acts through, for commutation purposes. `None` means `h` is
/// neither a scalar multiplication nor a translation (`exp` is that case) and
/// admits no surround; the row is still usable at the root.
const fn h_action(h: HomShape) -> Option<BinOp> {
    match h {
        HomShape::MulByLit | HomShape::DivByLit | HomShape::TotalAntitone(UnOp::Neg) => {
            Some(BinOp::Mul)
        }
        HomShape::AddInvariant => Some(BinOp::Add),
        _ => None,
    }
}

/// Outward: a lift of the form `L(h(inner))`, with `h` applied anywhere on an
/// `accum`-linear path, becomes `Fold{row.from}(L(inner))` with `h` applied
/// once outside.
///
/// Inward: a `post` of the form `h(Arg(0))` moves into the lift and the
/// accumulator becomes `row.to`. Both alternatives stay live and the cost model
/// decides. Every hoistable factor is peeled in one firing, and `|lift|`
/// strictly decreases at each step.
pub fn hoist(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(L1::KFold {
        carrier, acc, post, ..
    }) = &node.op
    else {
        return None;
    };
    if acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }
    // One slot. A multi-slot carrier couples its slots through `merge`, so
    // changing one slot's monoid changes what every sibling reading it computes.
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
    let Op::L1(L1::KFold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    // Every axis the peeled factor ends up outside: the reduced one, and every
    // axis promoted into the accumulator's data space, since a factor that
    // varies across lanes cannot be applied once at the end.
    let outside: SmallVec<[u32; 4]> =
        std::iter::once(*axis).chain(vec_axes.iter().copied()).collect();
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
        if reads_any_index(e) {
            return false;
        }
        let mut used = Vec::new();
        arg_indices(e, &mut used);
        used.iter()
            .all(|&i| inv.get(i as usize).copied() == Some(true))
    };

    // Each step narrows the lift and records the `h` to re-apply outside,
    // outermost first.
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
            // `L` has to be an endomorphism of both monoids and commute with
            // `h`: `Fold{to}(L(h(x))) = Fold{to}(h(L(x))) = h(Fold{from}(L(x)))`
            // needs each equality in turn. An empty surround is `L = id`.
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
            if m.c.as_ref().is_some_and(|c| reads_any_index(c)) {
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

    // Everything that can still decline is decided before anything is minted,
    // so a decline leaves no orphan node behind.
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
    if !inner_carrier.identity_closed(probes_for(*acc)) {
        return None;
    }

    // The outer map reads the fold plus whichever invariant operands the peeled
    // factors name, each re-viewed at the fold's own output space. Slot 0 is
    // reserved for the fold itself.
    let out_shape: Vec<Dim> = f.own().shape.to_vec();
    let mut projected: Vec<Operand> = Vec::new();
    let mut slot_of: Vec<(u32, u32)> = Vec::new();
    for m in &peeled {
        let Some(c) = &m.c else { continue };
        let mut used = Vec::new();
        arg_indices(c, &mut used);
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
    // A nest's dtype is its accumulator's whatever its `post` says, so a `post`
    // that casts would make the map and the fold two different values.
    if body.dtype() != f.own().dtype {
        return None;
    }

    // The inner fold: same slot shape, `cur` as the accumulation, an identity
    // `post` since the epilogue moves outside after `h`, and no edge for the
    // factor that just left.
    let inner = b
        .add_l1(L1::KFold {
            space: space.clone(),
            axis: *axis,
            vec_axes: vec_axes.clone(),
            carrier: inner_carrier,
            acc: *acc,
            post: smallvec![ScalarExpr::arg(0, *acc)],
            ops: inner_ops,
            sched: sched.clone(),
        })
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

/// `h(Fold{from}(x)) == Fold{to}(Map{h}(x))`, read left to right: a closed `h`
/// sitting in `post` moves into the lift. `h` must be closed because a `post`
/// reads accumulator slots, not operands.
fn hoist_inward(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
    accum: BinOp,
) -> Option<Id> {
    let Op::L1(L1::KFold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let closed = |e: &ScalarExpr| -> bool {
        let mut used = Vec::new();
        arg_indices(e, &mut used);
        used.is_empty() && !reads_any_index(e)
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
    if !pushed.identity_closed(probes_for(*acc)) {
        return None;
    }
    let alt = b
        .add_l1(L1::KFold {
            space: space.clone(),
            axis: *axis,
            vec_axes: vec_axes.clone(),
            carrier: pushed,
            acc: *acc,
            post: smallvec![ScalarExpr::arg(0, *acc)],
            ops: ops.clone(),
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// The same single-slot carrier accumulating with `op` instead: new identity,
/// new merge, the slot shape and the tie policy carried through.
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
        tie: matches!(op, BinOp::Max | BinOp::Min)
            .then(|| c.tie.unwrap_or(TiePolicy::SplitEvenly)),
    })
}

/// An operand restated as a plain strided read over `space`.
///
/// An `Unflatten` map with one sub-axis per logical axis is a stride vector:
/// the two spellings denote the same read, so both are accepted. A map with
/// several sub-axes per axis is a genuine divmod decomposition and declines.
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

/// A read stated over the full `space`, restated over the **iteration** space.
///
/// `None` when the operand varies along a promoted axis: such a value is not a
/// function of the iteration coordinate, so no nest over the iteration space
/// produces it.
pub(crate) fn on_iter_space(
    o: &Operand,
    space: &IndexSpace,
    vec_axes: &[u32],
) -> Option<Operand> {
    if vec_axes.is_empty() {
        return Some(o.clone());
    }
    let o = as_alias_over(o, space)?;
    let mut shape: Vec<Dim> = Vec::new();
    let mut strides: Vec<Dim> = Vec::new();
    for i in 0..space.rank() {
        if vec_axes.contains(&(i as u32)) {
            if o.layout.strides()[i].as_const() != Some(0) {
                return None;
            }
            continue;
        }
        shape.push(o.layout.shape()[i]);
        strides.push(o.layout.strides()[i]);
    }
    Some(Operand {
        src: o.src,
        layout: Layout::from_parts(o.layout.offset(), &shape, &strides).ok()?,
        access: AccessPlan::Alias,
    })
}

/// Re-view an operand stated against the fold's `space` at the fold's *output*
/// space: drop every axis in `drop` — the reduced axis and any promoted ones —
/// and append a stride-0 axis for the carrier lanes the output carries.
///
/// Only the rank-aligned case is expressible; anything else declines.
fn project_operand(
    o: &Operand,
    space: &IndexSpace,
    drop: &[u32],
    carrier: &Carrier,
) -> Option<Operand> {
    // Alias only. Accepting an affine `Unflatten` here projects wrongly and
    // miscomputes the GPU top-p, min-p and mirostat sampling paths.
    if !matches!(o.access, AccessPlan::Alias) {
        return None;
    }
    let o = &as_alias_over(o, space)?;
    let mut shape: Vec<Dim> = Vec::new();
    let mut strides: Vec<Dim> = Vec::new();
    for i in 0..space.rank() {
        if drop.contains(&(i as u32)) {
            continue;
        }
        shape.push(o.layout.shape()[i]);
        strides.push(o.layout.strides()[i]);
    }
    if let Some(lanes) = carrier.out_dim()? {
        shape.push(lanes);
        strides.push(Dim::Const(0));
    }
    Some(Operand {
        src: o.src,
        layout: Layout::from_parts(o.layout.offset(), &shape, &strides).ok()?,
        access: AccessPlan::Alias,
    })
}

/// Hole indices no operand can occupy, used to read a table row's own action
/// back out of it.
const HOLE_D: u32 = u32::MAX - 1;
const HOLE_V: u32 = u32::MAX;

/// Read `T` out of a [`RetargetRow`]: apply the row's own `retarget` to two
/// holes and classify the result.
///
/// Returns the monoid `T` acts through and `h(delta)` as a template over
/// [`HOLE_D`]. A row whose `T` is not `v (+) f(delta)` for a single binop
/// declines.
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

/// Does `e` equal `template[HOLE_D := (u - Arg(ref_arg))]`, for one `u` shared
/// by every call? Compared modulo commutation.
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
    /// The reference's element expression, renumbered onto the reading fold's
    /// operand list.
    lift: ScalarExpr,
}

/// A reduction-carried dependence on another reduction over the same axis is
/// discharged by carrying the reference alongside and rescaling.
///
/// The match is an operand with no stride over the reduced axis whose value is
/// a fold over that axis of reads this fold already performs, and whose element
/// expression is exactly the `u` this fold subtracts. One [`RETARGET_TABLE`]
/// row covers every retargeted slot's lift after [`linear_factor`] peels each
/// surround. Two unions are written: to the body slot view and the reference.
pub fn retarget(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(L1::KFold {
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

    // The reference is a nest over the iteration space, one rank shorter than
    // the space this node states its operand maps against, so the two operand
    // lists are comparable only after projection. With no promoted axis
    // `iter == space`, `iter_axis == axis` and `proj == reads`.
    let iter = IndexSpace::new(
        space
            .dims
            .iter()
            .enumerate()
            .filter(|(i, _)| !vec_axes.contains(&(*i as u32)))
            .map(|(_, d)| *d),
    );
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
        // An operand that varies along a promoted axis is left in its own space
        // by the projection above, where it matches nothing.
        if on_iter_space(&reads[r].0, space, vec_axes).is_none() {
            continue;
        }
        if invariant_along(&proj[r].0, &iter, iter_axis) != Some(true) {
            continue;
        }
        let Some(reference) = reference_fold(b, reads[r].1, &iter, iter_axis, &proj) else {
            continue;
        };
        // Acyclicity: the reference is reached from an operand, so its id is
        // already below this node's. What has to be checked is that no operand
        // the joint fold keeps *is* the reference, since the second union puts
        // the reference's class above the joint node.
        if (0..ops.len()).any(|i| i != r && class_members(b, reads[i].1).contains(&reference.id)) {
            continue;
        }
        if let Some(hit) = mint_retarget(b, id, node, r, &reference) {
            return Some(hit);
        }
    }
    None
}

/// The non-`Union` members reachable downward from `id`. A `Builder` cannot see
/// a class's root, so this sees exactly the alternatives an operand edge names.
fn class_members(b: &Builder<'_>, id: Id) -> SmallVec<[Id; 8]> {
    let mut out: SmallVec<[Id; 8]> = SmallVec::new();
    let mut stack = vec![id];
    while let Some(cur) = stack.pop() {
        if out.contains(&cur) {
            continue;
        }
        match &b.node(cur).op {
            Op::Union(x, y) => {
                stack.push(*x);
                stack.push(*y);
            }
            _ => out.push(cur),
        }
    }
    out
}

/// A reduction nest in either spelling — the `L0::Fold` the frontend built or
/// the `L1::KFold` it was lowered to.
///
/// Equality in this e-graph is not congruent, so a consuming edge names
/// whichever id it was handed while both denote the same value. Matching only
/// the L1 spelling would miss an edge naming the L0 one.
struct FoldView {
    space: IndexSpace,
    axis: u32,
    carrier: Carrier,
    ops: Vec<Operand>,
}

fn fold_view(b: &Builder<'_>, id: Id) -> Option<FoldView> {
    match b.node(id).op.clone() {
        Op::L1(L1::KFold {
            space,
            axis,
            vec_axes,
            carrier,
            post,
            ops,
            ..
        }) => {
            // A `post` means a reader sees `post(rho)` and not `rho`.
            (vec_axes.is_empty() && post.len() == 1 && post[0].kind() == &ScalarKind::Arg(0))
                .then_some(FoldView {
                    space,
                    axis,
                    carrier,
                    ops,
                })
        }
        Op::L0(L0::Fold {
            carrier, axis, ins, ..
        }) => {
            let space = IndexSpace::new(b.facts_of(*ins.first()?).shape.iter().copied());
            let ops = ins
                .iter()
                .map(|&s| alias_operand_of(s, &b.facts_of(s).shape))
                .collect();
            Some(FoldView {
                space,
                axis,
                carrier,
                ops,
            })
        }
        _ => None,
    }
}

/// Is `src` a fold over the same axis of the same reads? Decided by
/// address-map and id comparison, never by naming a producer.
fn reference_fold(
    b: &Builder<'_>,
    src: Id,
    space: &IndexSpace,
    axis: u32,
    reader: &[(Operand, Id)],
) -> Option<RefFold> {
    for cand in class_members(b, src) {
        let Some(v) = fold_view(b, cand) else { continue };
        if v.axis != axis || v.carrier.width() != 1 {
            continue;
        }
        if v.carrier.slots[0] != SlotTy::Scalar || !v.carrier.associative {
            continue;
        }
        if v.space.dims.len() != space.dims.len()
            || !v
                .space
                .dims
                .iter()
                .zip(&space.dims)
                .all(|(a, c)| a.known_eq(*c))
        {
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

/// How many producers `common_basis` will substitute before giving up. Each
/// step consumes one operand edge of a finite chain, so this bounds work only.
const MAX_EXPANSIONS: usize = 8;

/// The reference's element expression, rewritten over the **reading** fold's
/// operand list, or `None` when the two folds have no common basis.
///
/// Every read the reference performs has to be a read this fold already
/// performs, decided on address maps. The two nests absorb independently, so
/// the reference's expression is first brought down to the reader's basis by
/// the same substitution the fusion law performs. A producer is admitted only
/// if it is an elementwise value at a covered index space.
fn common_basis(
    b: &Builder<'_>,
    v: &FoldView,
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
        // Substitute one unplaced producer, as `splice` does.
        let slot = ops.iter().position(|o| place(o).is_none())?;
        if !matches!(ops[slot].access, AccessPlan::Alias) {
            return None;
        }
        let inner = map_view(b, ops[slot].src)?;
        if !v.space.covers(&inner.space)
            || !inner.ops.iter().all(|o| access_legal_in(&o.access, &v.space))
        {
            return None;
        }
        let base = ops.len() - 1;
        let body = shift_args(&inner.body, base as u32, &operand_dtypes(b, &inner.ops));
        let args: Vec<ScalarExpr> = operand_dtypes(b, &ops)
            .iter()
            .enumerate()
            .map(|(j, d)| match j.cmp(&slot) {
                std::cmp::Ordering::Equal => body.clone(),
                std::cmp::Ordering::Less => ScalarExpr::arg(j as u32, *d),
                std::cmp::Ordering::Greater => ScalarExpr::arg(j as u32 - 1, *d),
            })
            .collect();
        lift = lift.compose(&args);
        let mut next: Vec<Operand> = ops
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != slot)
            .map(|(_, o)| o.clone())
            .collect();
        next.extend(inner.ops.iter().cloned());
        ops = next;
    }
    None
}

fn mint_retarget(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    r: usize,
    reference: &RefFold,
) -> Option<Id> {
    let Op::L1(L1::KFold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let w = carrier.width();
    let ref_arg = r as u32;

    for row in RETARGET_TABLE {
        // The row's accumulation must be this carrier's, in every slot: the
        // mint *replaces* each slot's merge with the retargeted form, so a slot
        // that merges any other way is not this row's module.
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

        // One row must cover every slot simultaneously, after each slot's
        // linear surround is peeled: a scalar running sum and a vector
        // accumulator get the same factor or the law does not apply.
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
            // `commutes(T, L)`: `T` and every peeled `L` act through the same
            // monoid, or they do not commute and the rescale is wrong.
            if !peels.iter().all(|p| p.action() == action)
                || !match_shift(&matched, &template, ref_arg, &mut bound)
            {
                ok = false;
                break;
            }
            // `h(e) = id`, so an element enters as `L(identity)` and the first
            // element needs no special case.
            lifts.push(apply_peels(&peels, ScalarExpr::lit(seed)));
        }
        if !ok {
            continue;
        }
        // The reference must be exactly what this fold subtracts, or the
        // one-pass form seeds the accumulator with a value the two-pass form
        // never referenced.
        let u = bound?;
        if !expr_eq(&u, &reference.lift) {
            continue;
        }

        // Discharge the feedback operand: the joint fold reads the reference's
        // own inputs, which is what makes the result acyclic by construction.
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
        // Every lift of the joint node, in slot order, so an edge that only the
        // discharged reference read is pruned once for all of them.
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
        if !joint.identity_closed(probes_for(*acc)) {
            continue;
        }
        // Slot ranges are counted in lanes, not slots: a `Vector` slot is as
        // many lanes as it has positions, and the reference is always the one
        // lane the row's stat carrier declares.
        let (lanes, body_lanes) = (joint.lanes()?, carrier.lanes()?);
        let Some(body_axis) = carrier.out_dim() else {
            continue;
        };
        // The output's free dims: `space` minus the reduced axis AND minus
        // every promoted axis, since a promoted extent lives in the carrier's
        // lanes rather than in the write map.
        let free: Vec<Dim> = space
            .dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != *axis as usize && !vec_axes.contains(&(*i as u32)))
            .map(|(_, d)| *d)
            .collect();
        // Both readbacks are checked expressible before the joint node exists.
        if !view_expressible(&free, lanes, 1, body_lanes, body_axis)
            || !view_expressible(&free, lanes, 0, 1, None)
        {
            continue;
        }

        // Slot 0 is the reference; the rest are this fold's, so every `post`
        // shifts by one and the reference's is the identity.
        let joint_post: SmallVec<[ScalarExpr; 4]> = std::iter::once(ScalarExpr::arg(0, *acc))
            .chain(post.iter().map(|e| map_args(e, &|i| i + 1)))
            .collect();

        let joint_id = b
            .add_l1(L1::KFold {
                space: space.clone(),
                axis: *axis,
                vec_axes: vec_axes.clone(),
                carrier: joint,
                acc: *acc,
                post: joint_post,
                ops: new_ops,
                sched: sched.clone(),
            })
            .ok()?;

        let body_view = slot_view(b, joint_id, &free, lanes, 1, body_lanes, body_axis)?;
        let ref_view = slot_view(b, joint_id, &free, lanes, 0, 1, None)?;
        let hit = b.union(id, body_view).ok()?;
        // The second union stops extraction running the reference pass again.
        b.union(reference.id, ref_view).ok()?;
        return Some(hit);
    }
    None
}

/// Can lanes `[off, off + len)` of a joint fold's trailing carrier axis be read
/// back as an ordinary strided view? Minted as one `KMap { body: Arg(0) }`,
/// since the offset is not spellable in a single relative spec vector.
fn view_expressible(
    free: &[Dim],
    lanes: u64,
    off: u64,
    len: u64,
    want_axis: Option<Dim>,
) -> bool {
    slot_layout(free, lanes, off, len, want_axis).is_some()
}

/// The `(shape, layout)` a slot readback reads through, or `None` when the
/// range and the target's own carrier axis disagree.
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

#[cfg(test)]
mod tests {
    use crate::dtype::Splat;
    use super::*;
    use crate::carrier::eval;
    use crate::dtype::RoundMode;
    use crate::egraph::{EGraph, SaturationReport, Saturate, SaturationBudget};
    use crate::ir::level1::ScheduleDomain;
    use crate::rules::CORE_RULES;
    use crate::rules::test_support as ts;
    use crate::saturate::CoreSaturate;
    use crate::shape::{StrideSpec, SymId};

    const F: Dtype = Dtype::F32;

    fn a(i: u32) -> ScalarExpr {
        ScalarExpr::arg(i, F)
    }
    fn lit(v: f32) -> ScalarExpr {
        ScalarExpr::lit(Splat::F32(v))
    }
    fn bin(op: BinOp, x: ScalarExpr, y: ScalarExpr) -> ScalarExpr {
        ScalarExpr::bin(op, x, y)
    }

    fn saturate(g: &mut EGraph) -> SaturationReport {
        let caps = ts::caps();
        CoreSaturate
            .saturate(g, &caps, CORE_RULES, SaturationBudget::default())
            .unwrap()
    }

    fn fired(r: &SaturationReport, name: &str) -> u32 {
        r.fired
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, c)| *c)
            .unwrap_or(0)
    }

    /// Every `KFold` reachable from `id`'s class, including the ones minted
    /// under it.
    fn folds_in(g: &EGraph, id: Id) -> Vec<(Id, Carrier, Vec<Operand>)> {
        let mut out = Vec::new();
        let mut seen: Vec<Id> = Vec::new();
        let mut stack: Vec<Id> = g.class_ids(g.class_of(id)).into_vec();
        while let Some(cur) = stack.pop() {
            if seen.contains(&cur) {
                continue;
            }
            seen.push(cur);
            if let Op::L1(L1::KFold { carrier, ops, .. }) = &g.node(cur).op {
                out.push((cur, carrier.clone(), ops.clone()));
            }
            for c in g.node(cur).children.clone() {
                stack.extend(g.class_ids(g.class_of(c)));
            }
        }
        out
    }

    fn maps_in(g: &EGraph, id: Id) -> Vec<(Id, ScalarExpr, Vec<Operand>)> {
        let mut out = Vec::new();
        for m in g.class_ids(g.class_of(id)) {
            if let Op::L1(L1::KMap { body, ops, .. }) = &g.node(m).op {
                out.push((m, body.clone(), ops.clone()));
            }
        }
        out
    }

    /// An independent host run of a nest: absorb every element of the reduced
    /// axis with the carrier, then apply `post`.
    fn run_fold(c: &Carrier, post: &[ScalarExpr], rows: &[Vec<f32>]) -> Vec<f32> {
        let acc = rows.iter().fold(c.identity_f32(), |acc, r| {
            c.absorb(&acc, r).expect("evaluator covers this carrier")
        });
        post.iter()
            .map(|p| eval(p, &acc).expect("evaluator covers this post"))
            .collect()
    }

    fn space(dims: &[Dim]) -> IndexSpace {
        IndexSpace::new(dims.iter().copied())
    }

    /// The query is answered on the read, not on the spelling: a broadcast in a
    /// `Restride` node leaves the consuming edge with a dense layout, and only
    /// collapsing the view exposes the stride-0 axis.
    #[test]
    fn the_dependence_query_sees_the_read_not_the_spelling() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let row = ts::buffer(&mut g, F, &dims[..1]);
        let bcast = ts::restride(
            &mut g,
            &[
                StrideSpec::dim(0, Dim::Const(4)),
                StrideSpec::broadcast(Dim::Const(8)),
            ],
            row,
        );
        let caps = ts::caps();
        let b = g.builder(&caps);
        let sp = space(&dims);

        let dense = alias_operand_of(x, &dims);
        assert_eq!(invariant_along(&dense, &sp, 1), Some(false));
        assert_eq!(invariant_along(&dense, &sp, 0), Some(false));

        // The edge as the floor writes it: dense over the reading space.
        let edge = alias_operand_of(bcast, &dims);
        assert_eq!(
            invariant_along(&edge, &sp, 1),
            Some(false),
            "the spelling hides the broadcast"
        );
        let (eff, base) = effective(&b, &edge, &sp);
        assert_eq!(base, row);
        assert_eq!(eff.layout.strides(), &[Dim::Const(1), Dim::Const(0)]);
        assert_eq!(invariant_along(&eff, &sp, 1), Some(true));
        assert_eq!(invariant_along(&eff, &sp, 0), Some(false));
    }

    /// The divmod form and the rank-aligned form must agree wherever both
    /// apply, and the rank-aligned one must still answer where the divmod form
    /// cannot: a symbolic extent has no divisor.
    #[test]
    fn the_query_answers_under_a_symbolic_extent() {
        let sym = Dim::Sym(SymId(7));
        let sp = space(&[Dim::Const(4), sym]);
        assert_eq!(axis_window(&sp, 0), None, "no divisor past a Sym");

        let bcast = Operand {
            src: Id(0),
            layout: Layout::from_parts(Dim::Const(0), &[Dim::Const(4), sym], &[
                Dim::Const(1),
                Dim::Const(0),
            ])
            .unwrap(),
            access: AccessPlan::Alias,
        };
        assert_eq!(invariant_along(&bcast, &sp, 1), Some(true));

        // A transposed const-shaped read: only the divmod form can answer, and
        // it does.
        let sp2 = space(&[Dim::Const(4), Dim::Const(8)]);
        let transposed = Operand {
            src: Id(0),
            layout: Layout::from_parts(Dim::Const(0), &[Dim::Const(8), Dim::Const(4)], &[
                Dim::Const(1),
                Dim::Const(8),
            ])
            .unwrap(),
            access: AccessPlan::Alias,
        };
        assert_eq!(invariant_along(&transposed, &sp2, 1), Some(false));
    }

    /// `sum_j (x_j * c) == c * sum_j x_j` on a saturated graph: the fold loses
    /// the scale and a map outside gains it.
    #[test]
    fn hoist_fires_on_a_saturated_scaled_reduction() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let scaled = ts::map(&mut g, bin(BinOp::Mul, a(0), lit(0.125)), &[x]);
        let s = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Add, F),
            1,
            F,
            scaled,
        );
        g.add_root(s);
        let report = saturate(&mut g);
        assert!(report.saturated, "{report:?}");
        assert!(fired(&report, "HOIST") > 0, "HOIST never fired: {report:?}");

        let outer = maps_in(&g, s)
            .into_iter()
            .find(|(_, body, _)| {
                matches!(body.kind(), ScalarKind::Bin { op: BinOp::Mul, .. })
            })
            .expect("a map applying the scale outside the fold");
        let Op::L1(L1::KFold { carrier, ops, .. }) = g.node(outer.2[0].src).op.clone() else {
            panic!("the hoisted map must read a fold, got {:?}", g.node(outer.2[0].src).op)
        };
        assert_eq!(carrier.lift[0], a(0), "the scale is still in the lift");
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].src, x, "the fold reads the buffer directly");
    }

    /// The hoisted program computes the same number as the unhoisted one, both
    /// run against a hand-written reference.
    #[test]
    fn hoisting_a_scale_out_of_a_sum_is_numerically_the_same() {
        let xs = [1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5];
        let c = 0.125f32;
        let unhoisted = Carrier::binop(BinOp::Add, Splat::F32(0.0), F)
            .with_lift([bin(BinOp::Mul, a(0), lit(c))]);
        let hoisted = Carrier::binop(BinOp::Add, Splat::F32(0.0), F);

        let rows: Vec<Vec<f32>> = xs.iter().map(|x| vec![*x]).collect();
        let want: f32 = xs.iter().map(|x| x * c).sum();
        let got_unhoisted = run_fold(&unhoisted, &[a(0)], &rows)[0];
        let inner = run_fold(&hoisted, &[a(0)], &rows)[0];
        let got_hoisted = eval(&bin(BinOp::Mul, a(0), lit(c)), &[inner]).unwrap();
        assert!((got_unhoisted - want).abs() < 1e-5);
        assert!((got_hoisted - want).abs() < 1e-5, "{got_hoisted} vs {want}");
    }

    /// `max_j(x_j + bias) == max_j(x_j) + bias` is bit-exact under
    /// round-to-nearest, so the row carries `exact_in_float: true` and fires on
    /// a value whose `NumericContract` forbids reassociation.
    #[test]
    fn hoist_fires_under_strict_on_an_extremum_with_an_invariant_bias() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let bias = ts::buffer(&mut g, F, &dims[..1]);
        let bb = ts::restride(
            &mut g,
            &[
                StrideSpec::dim(0, Dim::Const(4)),
                StrideSpec::broadcast(Dim::Const(8)),
            ],
            bias,
        );
        // `infer` meets a rounding body with `NumericContract::STRICT`, so
        // `reassoc` is false downstream.
        let q = ts::map(&mut g, ScalarExpr::round(RoundMode::HalfAwayFromZero, a(0)), &[x]);
        let shifted = ts::map(&mut g, bin(BinOp::Add, a(0), a(1)), &[q, bb]);
        let mx = ts::fold(&mut g, ts::binop_carrier(BinOp::Max, F), 1, F, shifted);
        g.add_root(mx);
        assert!(
            !g.facts(mx).numeric.reassoc,
            "precondition: the value must forbid reassociation"
        );

        let report = saturate(&mut g);
        assert!(fired(&report, "HOIST") > 0, "HOIST declined under STRICT: {report:?}");

        let outer = maps_in(&g, mx)
            .into_iter()
            .find(|(_, body, ops)| {
                matches!(body.kind(), ScalarKind::Bin { op: BinOp::Add, .. }) && ops.len() == 2
            })
            .expect("a map adding the bias outside the fold");
        // The bias left the reduction: it is read once per output row, through
        // its own rank-1 layout, not once per reduced element.
        let biased = outer.2.iter().find(|o| o.src == bias).expect("the bias operand");
        assert_eq!(biased.layout.shape(), &[Dim::Const(4)]);
        let Op::L1(L1::KFold { carrier, ops, .. }) = g.node(outer.2[0].src).op.clone() else {
            panic!("the hoisted map must read a fold")
        };
        assert_eq!(carrier.kind(), Some(BinOp::Max));
        assert!(
            !contains(&carrier.lift[0], &|e| matches!(
                e.kind(),
                ScalarKind::Bin { op: BinOp::Add, .. }
            )),
            "the bias is still inside the reduction: {:?}",
            carrier.lift[0]
        );
        assert!(ops.iter().all(|o| o.src != bias));
    }

    /// The same identity, run: `max(x + b) == max(x) + b`, exactly.
    #[test]
    fn hoisting_a_bias_out_of_an_extremum_is_exact() {
        let xs = [1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5];
        let b = 0.3f32;
        let rows: Vec<Vec<f32>> = xs.iter().map(|x| vec![*x, b]).collect();
        let unhoisted = Carrier::binop(BinOp::Max, Splat::F32(f32::NEG_INFINITY), F)
            .with_lift([bin(BinOp::Add, a(0), a(1))]);
        let hoisted = Carrier::binop(BinOp::Max, Splat::F32(f32::NEG_INFINITY), F);
        let inner = run_fold(&hoisted, &[a(0)], &rows)[0];
        let got = eval(&bin(BinOp::Add, a(0), a(1)), &[inner, b]).unwrap();
        let want = run_fold(&unhoisted, &[a(0)], &rows)[0];
        assert_eq!(got, want, "the extremum rows are bit-exact or they are wrong");
        assert_eq!(want, xs.iter().copied().fold(f32::NEG_INFINITY, f32::max) + b);
    }

    /// `min_j(-x_j) == -max_j(x_j)`: the antitone row, which also hands the
    /// `Min` adjoint its tie handling for free.
    #[test]
    fn hoist_turns_a_negated_minimum_into_a_maximum() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let neg = ts::map(&mut g, ScalarExpr::un(UnOp::Neg, a(0)), &[x]);
        let mn = ts::fold(&mut g, ts::binop_carrier(BinOp::Min, F), 1, F, neg);
        g.add_root(mn);
        let report = saturate(&mut g);
        assert!(fired(&report, "HOIST") > 0, "{report:?}");

        let outer = maps_in(&g, mn)
            .into_iter()
            .find(|(_, body, _)| matches!(body.kind(), ScalarKind::Un { op: UnOp::Neg, .. }))
            .expect("a negation outside the fold");
        let Op::L1(L1::KFold { carrier, ops, .. }) = g.node(outer.2[0].src).op.clone() else {
            panic!("the hoisted map must read a fold")
        };
        assert_eq!(carrier.kind(), Some(BinOp::Max), "Min under Neg is Max");
        assert_eq!(carrier.identity[0], Splat::F32(f32::NEG_INFINITY));
        assert_eq!(carrier.lift[0], a(0));
        assert_eq!(ops[0].src, x);

        let xs = [1.5f32, -3.0, 7.25, 0.0];
        let rows: Vec<Vec<f32>> = xs.iter().map(|v| vec![*v]).collect();
        let hoisted = -run_fold(&carrier, &[a(0)], &rows)[0];
        let direct = xs.iter().map(|v| -v).fold(f32::INFINITY, f32::min);
        assert_eq!(hoisted, direct);
    }

    /// `ValueFacts` carries no sign lattice, so `max(sqrt x) = sqrt(max x)` and
    /// `log(prod x) = sum(log x)` must not fire: both are partial.
    #[test]
    fn partial_unaries_never_hoist() {
        for (un, op) in [(UnOp::Sqrt, BinOp::Max), (UnOp::Log, BinOp::Mul)] {
            let mut g = ts::graph();
            let dims = [Dim::Const(4), Dim::Const(8)];
            let x = ts::buffer(&mut g, F, &dims);
            let inner = ts::map(&mut g, ScalarExpr::un(un, a(0)), &[x]);
            let f = ts::fold(&mut g, ts::binop_carrier(op, F), 1, F, inner);
            g.add_root(f);
            let report = saturate(&mut g);
            assert_eq!(
                fired(&report, "HOIST"),
                0,
                "{un:?} is partial on f32 and must not hoist out of {op:?}"
            );
        }
        assert!(!is_total_on(UnOp::Sqrt, F) && !is_total_on(UnOp::Log, F));
        assert!(
            !HOM_TABLE.iter().any(|r| matches!(
                r.h,
                HomShape::TotalMonotone(UnOp::Log) | HomShape::TotalAntitone(UnOp::Log)
            )),
            "log(prod x) = sum(log x) is unsound over sign and is not a row"
        );
    }

    /// The factor is decided by the dependence query, not by being a literal:
    /// a per-row denominator held in a tensor is invariant along the reduced
    /// axis, and `sum(x / n) == sum(x) / n` for every `n`, sign included.
    #[test]
    fn hoist_peels_a_factor_that_is_data_rather_than_a_literal() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let n = ts::buffer(&mut g, F, &dims[..1]);
        let bn = ts::restride(
            &mut g,
            &[
                StrideSpec::dim(0, Dim::Const(4)),
                StrideSpec::broadcast(Dim::Const(8)),
            ],
            n,
        );
        let ratio = ts::map(&mut g, bin(BinOp::Div, a(0), a(1)), &[x, bn]);
        let s = ts::fold(&mut g, ts::binop_carrier(BinOp::Add, F), 1, F, ratio);
        g.add_root(s);
        let report = saturate(&mut g);
        assert!(fired(&report, "HOIST") > 0, "HOIST never fired: {report:?}");

        let outer = maps_in(&g, s)
            .into_iter()
            .find(|(_, body, ops)| {
                matches!(body.kind(), ScalarKind::Bin { op: BinOp::Div, .. }) && ops.len() == 2
            })
            .expect("a map dividing by the count outside the fold");
        // The denominator is read once per output row, through its own rank-1
        // layout, instead of once per reduced element.
        let den = outer.2.iter().find(|o| o.src == n).expect("the count operand");
        assert_eq!(den.layout.shape(), &[Dim::Const(4)]);
        let Op::L1(L1::KFold { carrier, ops, .. }) = g.node(outer.2[0].src).op.clone() else {
            panic!("the hoisted map must read a fold")
        };
        assert_eq!(carrier.lift[0], a(0), "the divide is still in the lift");
        assert_eq!(ops.len(), 1, "the discharged edge is still read: {ops:?}");
        assert_eq!(ops[0].src, x);

        let xs = [1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5];
        let count = 5.0f32;
        let rows: Vec<Vec<f32>> = xs.iter().map(|v| vec![*v]).collect();
        let inner = run_fold(&carrier, &[a(0)], &rows)[0];
        let got = eval(&bin(BinOp::Div, a(0), a(1)), &[inner, count]).unwrap();
        let want: f32 = xs.iter().map(|v| v / count).sum();
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }

    /// An additive fold is sign-blind, so any axis-invariant expression may be
    /// peeled; `max(x * c) == max(x) * c` is false for `c < 0`, so a row over
    /// `Max`/`Min` admits only a `Lit`. `HOM_TABLE` holds no extremum scaling
    /// row, so this pins the guard rather than a firing.
    #[test]
    fn a_scaling_row_over_an_extremum_admits_only_a_literal() {
        let always = |_: &ScalarExpr| true;
        for h in [HomShape::MulByLit, HomShape::DivByLit] {
            let additive = HomRow { h, from: BinOp::Add, to: BinOp::Add, exact_in_float: false };
            let extremum = HomRow { h, from: BinOp::Max, to: BinOp::Max, exact_in_float: true };
            assert!(!sign_sensitive(&additive));
            assert!(sign_sensitive(&extremum));
            // A runtime value is invariant but of unknown sign.
            assert!(admissible_scale(&a(1), &additive, &always));
            assert!(!admissible_scale(&a(1), &extremum, &always));
            // A literal is admissible to both, and a no-op factor to neither.
            assert!(admissible_scale(&lit(2.0), &additive, &always));
            assert!(admissible_scale(&lit(2.0), &extremum, &always));
            // A negative literal is admissible only to the additive row:
            // `max(x * -2) == min(x) * -2`, which is a different row.
            assert!(admissible_scale(&lit(-2.0), &additive, &always));
            assert!(!admissible_scale(&lit(-2.0), &extremum, &always));
            for degenerate in [lit(1.0), lit(0.0)] {
                assert!(!admissible_scale(&degenerate, &additive, &always));
                assert!(!admissible_scale(&degenerate, &extremum, &always));
            }
        }
        // A factor that varies along the axis is inadmissible to every row.
        let never = |_: &ScalarExpr| false;
        let additive = HomRow {
            h: HomShape::MulByLit,
            from: BinOp::Add,
            to: BinOp::Add,
            exact_in_float: false,
        };
        assert!(!admissible_scale(&a(1), &additive, &never));
    }

    /// Every hoistable factor leaves in one firing, and `|lift|` strictly
    /// decreases at each step.
    #[test]
    fn hoist_peels_every_factor_in_one_firing() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let s = ts::kfold(
            &mut g,
            &dims,
            1,
            ts::binop_carrier(BinOp::Add, F)
                .with_lift([bin(BinOp::Div, bin(BinOp::Mul, a(0), lit(0.5)), lit(4.0))]),
            F,
            a(0),
            vec![alias_operand_of(x, &dims)],
        );
        g.add_root(s);
        let report = saturate(&mut g);
        assert!(fired(&report, "HOIST") > 0, "{report:?}");

        // Both factors leave together in the order they were peeled.
        let outer = maps_in(&g, s)
            .into_iter()
            .find(|(_, body, _)| {
                [0.5f32, 4.0].iter().all(|v| contains(body, &|e| is_lit_value(e, *v)))
            })
            .expect("a map applying both factors outside the fold");
        let Op::L1(L1::KFold { carrier, .. }) = g.node(outer.2[0].src).op.clone() else {
            panic!("the hoisted map must read a fold")
        };
        assert_eq!(carrier.lift[0], a(0), "a factor is still in the lift");

        let xs = [1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5];
        let rows: Vec<Vec<f32>> = xs.iter().map(|v| vec![*v]).collect();
        let inner = run_fold(&carrier, &[a(0)], &rows)[0];
        let got = eval(&outer.1, &[inner]).unwrap();
        let want: f32 = xs.iter().map(|v| v * 0.5 / 4.0).sum();
        assert!((got - want).abs() < 1e-5, "{got} vs {want}");
    }

    /// `prod_j exp(x_j) == exp(sum_j x_j)` replaces `n - 1` exponentials and
    /// `n - 1` multiplies with adds, and keeps the product in range.
    #[test]
    fn hoist_turns_a_product_of_exponentials_into_one_exponential() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let p = ts::kfold(
            &mut g,
            &dims,
            1,
            ts::binop_carrier(BinOp::Mul, F).with_lift([ScalarExpr::un(UnOp::Exp, a(0))]),
            F,
            a(0),
            vec![alias_operand_of(x, &dims)],
        );
        g.add_root(p);
        let report = saturate(&mut g);
        assert!(fired(&report, "HOIST") > 0, "{report:?}");

        let outer = maps_in(&g, p)
            .into_iter()
            .find(|(_, body, _)| matches!(body.kind(), ScalarKind::Un { op: UnOp::Exp, .. }))
            .expect("one exponential outside the fold");
        let Op::L1(L1::KFold { carrier, .. }) = g.node(outer.2[0].src).op.clone() else {
            panic!("the hoisted map must read a fold")
        };
        assert_eq!(carrier.kind(), Some(BinOp::Add), "the product became a sum");
        assert_eq!(carrier.identity[0], Splat::F32(0.0));
        assert_eq!(carrier.lift[0], a(0), "the exponential left the lift");

        // Every partial product leaves the representable range; the sum does
        // not, and the answer is an ordinary number.
        let xs = [-100.0f32, -100.0, 250.0];
        let rows: Vec<Vec<f32>> = xs.iter().map(|v| vec![*v]).collect();
        let via_sum = eval(&outer.1, &[run_fold(&carrier, &[a(0)], &rows)[0]]).unwrap();
        let via_product: f32 = xs.iter().map(|v| v.exp()).product();
        assert!(!via_product.is_finite(), "the product form survives: {via_product}");
        let want = 50.0f32.exp();
        assert!(
            via_sum.is_finite() && (via_sum / want - 1.0).abs() < 1e-5,
            "{via_sum} vs {want}"
        );
    }

    /// The dependence query is what stops the law: a `bias` that varies along
    /// the reduced axis is not invariant, and `max(x + b_j) != max(x) + b_j`.
    #[test]
    fn a_factor_that_varies_along_the_axis_does_not_hoist() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let bias = ts::buffer(&mut g, F, &dims);
        let shifted = ts::map(&mut g, bin(BinOp::Add, a(0), a(1)), &[x, bias]);
        let mx = ts::fold(&mut g, ts::binop_carrier(BinOp::Max, F), 1, F, shifted);
        g.add_root(mx);
        let report = saturate(&mut g);
        assert_eq!(fired(&report, "HOIST"), 0, "{report:?}");
    }

    /// Both directions are minted. `exp(sum x) == prod(exp x)` read left to
    /// right pushes `h` into the lift and turns the monoid from `(R,+)` into
    /// `(R,x)`; the cost model decides which runs.
    #[test]
    fn hoist_mints_the_inward_direction_too() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let e = ts::kfold(
            &mut g,
            &dims,
            1,
            ts::binop_carrier(BinOp::Add, F),
            F,
            ScalarExpr::un(UnOp::Exp, a(0)),
            vec![alias_operand_of(x, &dims)],
        );
        g.add_root(e);
        let report = saturate(&mut g);
        assert!(fired(&report, "HOIST") > 0, "{report:?}");

        let pushed = folds_in(&g, e)
            .into_iter()
            .find(|(_, c, _)| c.kind() == Some(BinOp::Mul))
            .expect("a product fold over exp(x)");
        assert_eq!(pushed.1.identity[0], Splat::F32(1.0));
        assert!(matches!(
            pushed.1.lift[0].kind(),
            ScalarKind::Un { op: UnOp::Exp, .. }
        ));

        let xs = [0.5f32, -1.25, 2.0, 0.75];
        let rows: Vec<Vec<f32>> = xs.iter().map(|v| vec![*v]).collect();
        let via_product = run_fold(&pushed.1, &[a(0)], &rows)[0];
        let via_sum = xs.iter().sum::<f32>().exp();
        assert!((via_product - via_sum).abs() < 1e-4, "{via_product} vs {via_sum}");
    }

    /// The two-pass shift chain: a running reference over an axis, then a
    /// second fold over the same axis whose body subtracts it.
    fn shift_chain(g: &mut EGraph, dims: &[Dim], body: ScalarExpr, extra: &[Id]) -> (Id, Id) {
        let x = ts::buffer(g, F, dims);
        let m = ts::fold(g, ts::binop_carrier(BinOp::Max, F), 1, F, x);
        let bm = ts::restride(
            g,
            &[StrideSpec::dim(0, dims[0]), StrideSpec::broadcast(dims[1])],
            m,
        );
        let mut ins: Vec<Id> = extra.to_vec();
        ins.push(x);
        ins.push(bm);
        let p = ts::map(g, body, &ins);
        let l = ts::fold(g, ts::binop_carrier(BinOp::Add, F), 1, F, p);
        g.add_root(l);
        (m, l)
    }

    /// The joint carrier a firing produced, in its unpromoted spelling.
    /// `PROMOTE` also fires on a retargeted fold and mints a `Vector`-slotted
    /// alternative beside it.
    fn retargeted(g: &EGraph, id: Id) -> Option<(Id, Carrier, Vec<Operand>)> {
        folds_in(g, id)
            .into_iter()
            .find(|(_, c, _)| c.width() == 2 && c.slots.iter().all(|s| *s == SlotTy::Scalar))
    }

    /// The derived carrier is compared term for term with the
    /// `shift_stabilized_sum` oracle, including the `safe_delta` guard.
    #[test]
    fn retarget_derives_the_shift_carrier_on_a_saturated_graph() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let (m, l) = shift_chain(
            &mut g,
            &dims,
            ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1))),
            &[],
        );
        // The two laws must not cost this chain its saturation.
        let mut baseline = ts::graph();
        shift_chain(
            &mut baseline,
            &dims,
            ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1))),
            &[],
        );
        let without: Vec<crate::egraph::Rule> = CORE_RULES
            .iter()
            .filter(|r| r.name != "HOIST" && r.name != "RETARGET")
            .copied()
            .collect();
        let caps = ts::caps();
        let base = CoreSaturate
            .saturate(&mut baseline, &caps, &without, SaturationBudget::default())
            .unwrap();

        let report = saturate(&mut g);
        assert!(fired(&report, "RETARGET") > 0, "RETARGET never fired: {report:?}");
        // The slot view under the joint fold puts one more level below every
        // retargeted chain, so the chain needs one extra round to saturate.
        assert!(report.truncated.is_empty(), "{report:?}");
        assert!(base.saturated, "precondition: the chain saturates without the laws");
        let mut deep = ts::graph();
        shift_chain(
            &mut deep,
            &dims,
            ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1))),
            &[],
        );
        let deep_report = CoreSaturate
            .saturate(
                &mut deep,
                &caps,
                CORE_RULES,
                SaturationBudget {
                    max_rounds: 8,
                    ..SaturationBudget::default()
                },
            )
            .unwrap();
        assert!(
            deep_report.saturated,
            "the chain does not saturate even at eight rounds: {deep_report:?}"
        );

        let (joint, carrier, ops) = retargeted(&g, l).expect("a two-slot fold");
        let oracle = crate::carrier::oracle::shift_stabilized_sum(UnOp::Exp, F);
        assert_eq!(carrier.slots, oracle.slots);
        assert_eq!(carrier.identity, oracle.identity);
        assert_eq!(
            carrier.merge, oracle.merge,
            "the derived merge differs from the oracle"
        );
        assert_eq!(carrier.lift[1], lit(1.0), "h(e) = id, so an element enters as 1");
        // The joint fold reads the source, not the reference's output.
        assert_eq!(ops.len(), 1);
        assert_eq!(
            g.facts(ops[0].src).shape.as_slice(),
            &dims,
            "the joint fold reads the source directly"
        );

        // This fold's class holds the body slot view, and the reference's class
        // holds the reference slot view of the same node.
        let reaches = |from: Id| -> bool {
            let mut stack = g.class_ids(g.class_of(from));
            let mut seen: Vec<Id> = Vec::new();
            while let Some(cur) = stack.pop() {
                if cur == joint {
                    return true;
                }
                if seen.contains(&cur) {
                    continue;
                }
                seen.push(cur);
                // A slot readback is one `KMap { Arg(0) }` over the joint fold,
                // so walking only views stops at the readback.
                let view = match &g.node(cur).op {
                    Op::Union(..) => true,
                    Op::L0(L0::Restride { .. }) => true,
                    Op::L1(L1::KMap { body, ops, .. }) => {
                        body.kind() == &ScalarKind::Arg(0) && ops.len() == 1
                    }
                    _ => false,
                };
                if view {
                    for c in g.node(cur).children.clone() {
                        stack.extend(g.class_ids(g.class_of(c)));
                    }
                }
            }
            false
        };
        assert!(reaches(l), "the body slot view is not in the reader's class");
        assert!(
            reaches(m),
            "the reference is left to run a second time: no slot-0 union"
        );
    }

    /// One pass equals two passes, and stays finite where the naive form
    /// overflows.
    #[test]
    fn the_derived_shift_carrier_is_one_pass_equal_to_two_pass() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let (_, l) = shift_chain(
            &mut g,
            &dims,
            ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1))),
            &[],
        );
        saturate(&mut g);
        let (_, c, _) = retargeted(&g, l).expect("a two-slot fold");

        for xs in [
            vec![1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5],
            vec![900.0, 901.0, 899.5],
            vec![-4.0],
        ] {
            let rows: Vec<Vec<f32>> = xs.iter().map(|v| vec![*v]).collect();
            let got = run_fold(&c, &[a(0), a(1)], &rows);
            let want_max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let want_sum: f32 = xs.iter().map(|v| (v - want_max).exp()).sum();
            assert!((got[0] - want_max).abs() < 1e-6, "{got:?} vs {want_max}");
            assert!(
                (got[1] - want_sum).abs() < 1e-5 * want_sum.abs().max(1.0),
                "{got:?} vs {want_sum}"
            );
            assert!(got[1].is_finite());
        }
        // The naive form overflows on the same input.
        let naive: f32 = [900.0f32, 901.0, 899.5].iter().map(|v| v.exp()).sum();
        assert!(naive.is_infinite());
    }

    /// Any split equals one sequential pass — the property a blocked schedule
    /// and a tree reduction both depend on — and merging two identity lanes
    /// gives the identity, which is the NaN `safe_delta` exists for.
    #[test]
    fn the_derived_carrier_splits_anywhere_and_absorbs_its_identity() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let (_, l) = shift_chain(
            &mut g,
            &dims,
            ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1))),
            &[],
        );
        saturate(&mut g);
        let (_, c, _) = retargeted(&g, l).expect("a two-slot fold");

        let xs = [1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5, 4.25, -0.5];
        let run = |seg: &[f32]| -> Vec<f32> {
            seg.iter().fold(c.identity_f32(), |acc, v| {
                c.absorb(&acc, &[*v]).unwrap()
            })
        };
        let whole = run(&xs);
        for cut in 0..=xs.len() {
            let joined = c.eval_merge(&run(&xs[..cut]), &run(&xs[cut..])).unwrap();
            assert!(
                joined
                    .iter()
                    .zip(&whole)
                    .all(|(x, y)| (x - y).abs() <= 1e-4 * y.abs().max(1.0)),
                "cut {cut}: {joined:?} vs {whole:?}"
            );
        }
        let ident = c.identity_f32();
        let merged = c.eval_merge(&ident, &ident).unwrap();
        assert!(merged.iter().all(|v| !v.is_nan()), "identity merge is NaN: {merged:?}");
        assert_eq!(merged, ident);
        assert!(c.identity_closed(probes_for(F)));
    }

    /// A one-pass weighted log-sum-exp, `sum_c w_c * exp(x_c - lse(x))`. The
    /// weight sits on a multiplicative path outside the shifted term, so a
    /// root-only peel would match nothing.
    #[test]
    fn retarget_fires_on_a_weighted_log_sum_exp() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let w = ts::buffer(&mut g, F, &dims);
        let (_, l) = shift_chain(
            &mut g,
            &dims,
            bin(
                BinOp::Mul,
                a(0),
                ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(1), a(2))),
            ),
            &[w],
        );
        let report = saturate(&mut g);
        assert!(fired(&report, "RETARGET") > 0, "{report:?}");

        let (_, c, ops) = retargeted(&g, l).expect("a two-slot fold");
        assert_eq!(ops.len(), 2, "the weight and the source, not the reference");
        // `L(1) = w`: the surround applied to the action's identity, so the
        // element enters as its own weight with no multiply by one.
        assert_eq!(c.lift[1], a(0));

        let xs = [1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5];
        let ws = [0.25f32, 1.0, -0.5, 2.0, 0.125, 3.0];
        let rows: Vec<Vec<f32>> = ws.iter().zip(&xs).map(|(w, x)| vec![*w, *x]).collect();
        let got = run_fold(&c, &[a(0), a(1)], &rows);
        let want_max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let want: f32 = ws
            .iter()
            .zip(&xs)
            .map(|(w, x)| w * (x - want_max).exp())
            .sum();
        assert!((got[0] - want_max).abs() < 1e-6, "{got:?}");
        assert!((got[1] - want).abs() < 1e-5, "{got:?} vs {want}");
    }

    /// The same law over `(R, max)` instead of `(R, +)`:
    /// `max_j(w_j + x_j - max_k x_k)` fires at the `max-plus` row with no
    /// exponential anywhere.
    #[test]
    fn retarget_fires_at_the_max_plus_row_on_a_tropical_reduction() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let w = ts::buffer(&mut g, F, &dims);
        let x = ts::buffer(&mut g, F, &dims);
        let m = ts::fold(&mut g, ts::binop_carrier(BinOp::Max, F), 1, F, x);
        let bm = ts::restride(
            &mut g,
            &[StrideSpec::dim(0, dims[0]), StrideSpec::broadcast(dims[1])],
            m,
        );
        let p = ts::map(
            &mut g,
            bin(BinOp::Add, a(0), bin(BinOp::Sub, a(1), a(2))),
            &[w, x, bm],
        );
        let r = ts::fold(&mut g, ts::binop_carrier(BinOp::Max, F), 1, F, p);
        g.add_root(r);
        let report = saturate(&mut g);
        assert!(fired(&report, "RETARGET") > 0, "{report:?}");

        let (_, c, _) = retargeted(&g, r).expect("a two-slot fold");
        assert!(
            !contains(&c.merge[1], &|e| matches!(
                e.kind(),
                ScalarKind::Un { op: UnOp::Exp, .. }
            )),
            "the max-plus row must not carry an exponential: {:?}",
            c.merge[1]
        );
        assert_eq!(c.lift[1], a(0));

        let xs = [1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5];
        let ws = [0.25f32, 1.0, -0.5, 2.0, 0.125, 3.0];
        let rows: Vec<Vec<f32>> = ws.iter().zip(&xs).map(|(w, x)| vec![*w, *x]).collect();
        let got = run_fold(&c, &[a(0), a(1)], &rows);
        let want_max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let want = ws
            .iter()
            .zip(&xs)
            .map(|(w, x)| w + x - want_max)
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((got[0] - want_max).abs() < 1e-6, "{got:?}");
        assert!((got[1] - want).abs() < 1e-6, "{got:?} vs {want}");
        // The identity merge for this row is `max(-inf + 0, -inf + 0)`, a NaN
        // without the delta guard.
        let ident = c.identity_f32();
        assert_eq!(c.eval_merge(&ident, &ident).unwrap(), ident);
    }

    /// A reduction over a `Dim::Sym` axis still retargets, where the
    /// strip-mining law declines for lack of a blocking factor.
    #[test]
    fn retarget_fires_over_a_symbolic_reduction_length() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Sym(SymId(3))];
        let (_, l) = shift_chain(
            &mut g,
            &dims,
            ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1))),
            &[],
        );
        let report = saturate(&mut g);
        assert!(fired(&report, "RETARGET") > 0, "{report:?}");
        assert_eq!(fired(&report, "STRIP"), 0, "a Sym extent has no block factor");

        let (joint, c, _) = retargeted(&g, l).expect("a two-slot fold");
        let Op::L1(L1::KFold { space, .. }) = &g.node(joint).op else {
            panic!()
        };
        assert_eq!(space.dims[1], Dim::Sym(SymId(3)), "the axis stayed symbolic");

        // Equal to the two-pass form at three bindings of the same plan.
        for n in [1usize, 5, 33] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.37 - 4.0).collect();
            let rows: Vec<Vec<f32>> = xs.iter().map(|v| vec![*v]).collect();
            let got = run_fold(&c, &[a(0), a(1)], &rows);
            let want_max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let want: f32 = xs.iter().map(|v| (v - want_max).exp()).sum();
            assert!((got[0] - want_max).abs() < 1e-6, "n={n}: {got:?}");
            assert!((got[1] - want).abs() < 1e-5, "n={n}: {got:?} vs {want}");
        }
    }

    /// When the value being reduced is itself computed, the reference fold
    /// pulls that producer into its own lift and stops naming the intermediate,
    /// so the two element expressions are compared on the basis
    /// `common_basis` establishes.
    #[test]
    fn retarget_fires_when_the_reference_absorbed_its_own_producer() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        // One producer read by both nests: the reference reduces it and the
        // body subtracts the reference from it.
        let scaled = ts::map(&mut g, bin(BinOp::Mul, a(0), lit(0.5)), &[x]);
        let m = ts::fold(&mut g, ts::binop_carrier(BinOp::Max, F), 1, F, scaled);
        let bm = ts::restride(
            &mut g,
            &[StrideSpec::dim(0, dims[0]), StrideSpec::broadcast(dims[1])],
            m,
        );
        let p = ts::map(
            &mut g,
            ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1))),
            &[scaled, bm],
        );
        let l = ts::fold(&mut g, ts::binop_carrier(BinOp::Add, F), 1, F, p);
        g.add_root(l);
        let report = saturate(&mut g);
        assert!(fired(&report, "RETARGET") > 0, "{report:?}");

        let (_, c, ops) = retargeted(&g, l).expect("a two-slot fold");
        // The joint fold reads the source, with the shared producer in its own
        // lift: one traversal, one buffer, no intermediate.
        assert_eq!(ops.len(), 1);
        assert_eq!(ops[0].src, x);
        assert_eq!(c.lift[0], bin(BinOp::Mul, a(0), lit(0.5)));

        let xs = [1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5];
        let rows: Vec<Vec<f32>> = xs.iter().map(|v| vec![*v]).collect();
        let got = run_fold(&c, &[a(0), a(1)], &rows);
        let scaled_xs: Vec<f32> = xs.iter().map(|v| v * 0.5).collect();
        let want_max = scaled_xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let want: f32 = scaled_xs.iter().map(|v| (v - want_max).exp()).sum();
        assert!((got[0] - want_max).abs() < 1e-6, "{got:?}");
        assert!((got[1] - want).abs() < 1e-5, "{got:?} vs {want}");
    }

    /// A reduction whose accumulator already has more than one slot, built
    /// directly because no single rule in this file mints one.
    fn two_slot_shift(
        g: &mut EGraph,
        dims: &[Dim],
        lifts: [ScalarExpr; 2],
        merges: [ScalarExpr; 2],
    ) -> (Id, Id, Id) {
        let x = ts::buffer(g, F, dims);
        let w = ts::buffer(g, F, dims);
        let m = ts::fold(g, ts::binop_carrier(BinOp::Max, F), 1, F, x);
        let bm = ts::restride(
            g,
            &[StrideSpec::dim(0, dims[0]), StrideSpec::broadcast(dims[1])],
            m,
        );
        let carrier = Carrier {
            slots: smallvec![SlotTy::Scalar, SlotTy::Scalar],
            identity: smallvec![Splat::F32(0.0), Splat::F32(0.0)],
            lift: lifts.into_iter().collect(),
            merge: merges.into_iter().collect(),
            associative: true,
            tie: None,
        };
        let node = g
            .add(Op::L1(L1::KFold {
                space: space(dims),
                axis: 1,
                vec_axes: SmallVec::new(),
                carrier,
                acc: F,
                post: smallvec![a(0), a(1)],
                ops: vec![
                    alias_operand_of(x, dims),
                    alias_operand_of(bm, dims),
                    alias_operand_of(w, dims),
                ],
                sched: ScheduleDomain::Point,
            }))
            .unwrap();
        g.add_root(node);
        (m, node, x)
    }

    /// One row covers every retargeted slot at once, so a plain running sum and
    /// a weighted one are rescaled by the same factor with the same expression.
    ///
    /// The program is a self-normalized importance-sampling estimator: the
    /// partition function `sum exp(s)` and the estimate `sum v * exp(s)` over
    /// one axis.
    #[test]
    fn one_row_retargets_every_slot_of_a_multi_slot_accumulator() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let h = ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1)));
        let (m, node, x) = two_slot_shift(
            &mut g,
            &dims,
            [h.clone(), bin(BinOp::Mul, a(2), h)],
            [bin(BinOp::Add, a(0), a(2)), bin(BinOp::Add, a(1), a(3))],
        );
        let report = saturate(&mut g);
        assert!(fired(&report, "RETARGET") > 0, "{report:?}");

        let (_, c, ops) = folds_in(&g, node)
            .into_iter()
            .find(|(_, c, _)| c.width() == 3)
            .expect("a three-slot fold");
        assert_eq!(ops.len(), 2, "the reference edge is discharged");
        assert!(ops.iter().any(|o| o.src == x));
        // Slot 1 and slot 2 differ in what they carry, never in how they are
        // rescaled.
        let w = c.width();
        let rescale = |k: usize| -> ScalarExpr {
            map_args(&c.merge[k], &|i| match i {
                _ if i == k as u32 => HOLE_D,
                _ if i == (w + k) as u32 => HOLE_V,
                _ => i,
            })
        };
        assert_eq!(
            rescale(1),
            rescale(2),
            "slot 1 and slot 2 are rescaled differently: {:?} vs {:?}",
            c.merge[1],
            c.merge[2]
        );
        // The rescale carries the delta guard, without which merging two
        // identity lanes is `0 * exp((-inf) - (-inf))`, a NaN.
        assert!(
            contains(&c.merge[1], &|e| matches!(e.kind(), ScalarKind::Select { .. })),
            "no delta guard in the rescale: {:?}",
            c.merge[1]
        );
        let ident = c.identity_f32();
        assert_eq!(c.eval_merge(&ident, &ident).unwrap(), ident);
        assert!(m.0 < node.0, "the reference is below its reader");

        let xs = [1.5f32, -3.0, 7.25, 0.0, 2.0, -11.5];
        let vs = [0.25f32, 1.0, -0.5, 2.0, 0.125, 3.0];
        let rows: Vec<Vec<f32>> = xs.iter().zip(&vs).map(|(x, v)| vec![*x, *v]).collect();
        let got = run_fold(&c, &[a(0), a(1), a(2)], &rows);
        let want_max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let want_z: f32 = xs.iter().map(|v| (v - want_max).exp()).sum();
        let want_e: f32 = xs
            .iter()
            .zip(&vs)
            .map(|(x, v)| v * (x - want_max).exp())
            .sum();
        assert!((got[0] - want_max).abs() < 1e-6, "{got:?}");
        assert!((got[1] - want_z).abs() < 1e-5, "{got:?} vs {want_z}");
        assert!((got[2] - want_e).abs() < 1e-5, "{got:?} vs {want_e}");
    }

    /// When one row cannot cover both slots the law declines rather than
    /// rescaling half an accumulator. Slot 1 here accumulates with `Max` while
    /// slot 0 accumulates with `Add`, so no single row's `T` is an
    /// endomorphism of both.
    #[test]
    fn a_slot_no_row_covers_stops_the_whole_firing() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let h = ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1)));
        let (_, node, _) = two_slot_shift(
            &mut g,
            &dims,
            [h.clone(), bin(BinOp::Mul, a(2), h)],
            [
                bin(BinOp::Add, a(0), a(2)),
                bin(BinOp::Max, a(1), a(3)),
            ],
        );
        let report = saturate(&mut g);
        assert_eq!(fired(&report, "RETARGET"), 0, "{report:?}");
        assert!(
            folds_in(&g, node)
                .into_iter()
                .all(|(_, c, _)| c.width() < 3),
            "a slot was retargeted by a row that does not cover it"
        );
    }

    /// When the shift comes from a supplied buffer rather than a fold over this
    /// axis, the structural condition fails and the rule declines.
    #[test]
    fn a_supplied_reference_is_not_a_carried_dependence() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let x = ts::buffer(&mut g, F, &dims);
        let lse = ts::buffer(&mut g, F, &dims[..1]);
        let bl = ts::restride(
            &mut g,
            &[StrideSpec::dim(0, dims[0]), StrideSpec::broadcast(dims[1])],
            lse,
        );
        let p = ts::map(
            &mut g,
            ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(0), a(1))),
            &[x, bl],
        );
        let l = ts::fold(&mut g, ts::binop_carrier(BinOp::Add, F), 1, F, p);
        g.add_root(l);
        let report = saturate(&mut g);
        assert_eq!(fired(&report, "RETARGET"), 0, "{report:?}");
        assert!(retargeted(&g, l).is_none());
    }

    /// The reference has to be exactly what the body subtracts. Here the fold
    /// subtracts `max(x)` from `x + y`, so the one-pass form would seed the
    /// accumulator with a value the two-pass form never referenced.
    #[test]
    fn a_reference_over_a_different_element_expression_declines() {
        let mut g = ts::graph();
        let dims = [Dim::Const(4), Dim::Const(8)];
        let y = ts::buffer(&mut g, F, &dims);
        let (_, l) = shift_chain(
            &mut g,
            &dims,
            ScalarExpr::un(
                UnOp::Exp,
                bin(BinOp::Sub, bin(BinOp::Add, a(1), a(0)), a(2)),
            ),
            &[y],
        );
        let report = saturate(&mut g);
        assert_eq!(fired(&report, "RETARGET"), 0, "{report:?}");
        assert!(retargeted(&g, l).is_none());
    }

    /// Every guard compares modulo commutation, since a commutative binop's
    /// children may arrive in either order.
    #[test]
    fn every_guard_matches_modulo_commutation() {
        // The accumulation read.
        let swapped = Carrier {
            merge: smallvec![bin(BinOp::Add, a(1), a(0))],
            ..Carrier::binop(BinOp::Add, Splat::F32(0.0), F)
        };
        assert_eq!(single_slot_accum(&swapped), Some(BinOp::Add));
        let noncommutative = Carrier {
            merge: smallvec![bin(BinOp::Sub, a(1), a(0))],
            ..Carrier::binop(BinOp::Add, Splat::F32(0.0), F)
        };
        assert_eq!(single_slot_accum(&noncommutative), None, "Sub does not commute");

        // The shift template, whose `u - rho` does not commute but sits under
        // commutative parents.
        let (action, template) = row_action(&RETARGET_TABLE[0], F).unwrap();
        assert_eq!(action, BinOp::Mul);
        let shifted = ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(5), a(2)));
        let mut bound = None;
        assert!(match_shift(&shifted, &template, 2, &mut bound));
        assert_eq!(bound, Some(a(5)));
        // The reversed subtraction is a different program.
        let reversed = ScalarExpr::un(UnOp::Exp, bin(BinOp::Sub, a(2), a(5)));
        let mut bound = None;
        assert!(!match_shift(&reversed, &template, 2, &mut bound));

        // A commutative parent in either order finds the same peel.
        for e in [
            bin(BinOp::Mul, a(0), shifted.clone()),
            bin(BinOp::Mul, shifted.clone(), a(0)),
        ] {
            let hit = |x: &ScalarExpr| {
                let mut probe = None;
                match_shift(x, &template, 2, &mut probe)
            };
            let (peels, matched) = linear_factor(&e, BinOp::Add, &hit).expect("a peel");
            assert_eq!(peels, vec![Peel::Mul(a(0))]);
            assert_eq!(matched, shifted);
        }
    }

    /// Every `RETARGET_TABLE` row's action is read out of the row. A row whose
    /// `T` is not `v (+) f(delta)` for one binop declines.
    ///
    /// The action and the accumulation are different: a shift row's `T` acts on
    /// the module by multiplication while the module accumulates by addition.
    #[test]
    fn every_retarget_row_declares_its_own_action() {
        for row in RETARGET_TABLE {
            let (action, _) = row_action(row, F)
                .unwrap_or_else(|| panic!("{} has an unreadable action", row.name));
            // `T(0) = id`: the element enters the retargeted lift as
            // `L(identity)` and the first element needs no special case.
            let at_zero = (row.retarget)(&lit(0.0), &a(0), F);
            for v in [-2.5f32, 0.0, 1.0, 7.25] {
                assert_eq!(
                    eval(&at_zero, &[v]),
                    Some(v),
                    "{}: T(0) must be the identity",
                    row.name
                );
            }
            // `T` composes with the module's accumulation as the action says.
            assert!(
                matches!(action, BinOp::Mul | BinOp::Add),
                "{}: unclassifiable action {action:?}",
                row.name
            );
            // The row's own reference carrier is identity-closed.
            let stat = (row.stat)(F);
            assert!(stat.identity_closed(probes_for(F)), "{}", row.name);
        }
    }

    /// The set of rows this rule can reach is exactly the set with a firing
    /// test above.
    #[test]
    fn the_reachable_retarget_rows_are_the_tested_ones() {
        let names: Vec<&str> = RETARGET_TABLE.iter().map(|r| r.name).collect();
        assert!(names.contains(&"shift-exp"), "{names:?}");
        assert!(names.contains(&"max-plus"), "{names:?}");
        // The three approximate-exponential rows are the same law at a
        // different accuracy contract; the emitter chooses between them.
        for row in RETARGET_TABLE {
            assert!(row_action(row, F).is_some(), "{} is unreadable", row.name);
        }
    }
}
