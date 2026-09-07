//! HOIST and RETARGET — one dependence query, answered two ways.
//!
//! The query is asked of every operand of a reduction nest: is this operand
//! invariant along the reduction axis? It is decided on the read the edge
//! actually performs — [`Operand::address_map`], with a single pure view
//! collapsed into the layout first.
//!
//! * **HOIST** applies when the invariant operand is *not* derived from a
//!   fold over the same axis. If `h` is a monoid homomorphism from `(+, e+)` to
//!   `(x, ex)` then `h(Fold{+, a}(x)) == Fold{x, a}(Map{h}(x))`. Both
//!   directions are minted and cost decides. The pairs live in
//!   [`crate::carrier::HOM_TABLE`].
//! * **RETARGET** applies when it *is*: a reduction-carried dependence on
//!   another reduction over the same axis is discharged by carrying the
//!   reference alongside and rescaling, per
//!   [`crate::carrier::RETARGET_TABLE`] and [`crate::carrier::Carrier::retarget`].
//!
//! Neither rule names a producer, an op, a frontend chain or an algorithm;
//! they match the shape of the feedback.
//!
//! They are two [`Rule`](crate::egraph::Rule) entries sharing one helper: the
//! driver's fired set is per `(RuleId, Id)`, so a single merged rule could
//! fire at most once per node.

use crate::carrier::{
    Carrier, HOM_TABLE, HomRow, HomShape, RETARGET_TABLE, RetargetRow, SlotTy, is_total_on,
    map_args, probes_for,
};
use crate::dtype::{Dtype, Splat};
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, IndexSpace, Launch, Operand};
use crate::ir::logical::{Logical, TiePolicy};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::{
    access_legal_in, alias_operand_of, composed_layout, map_view, operand_dtypes, shift_args,
};
use crate::scalar::{BinOp, ScalarExpr, ScalarKind, UnOp};
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

/// Does the read `o` performs land on the *same element* for every value of
/// `axis`?
///
/// `Some(true)` proves invariance, `Some(false)` proves variance, `None` means
/// undecidable and every caller declines rather than guessing.
///
/// When the operand's layout is stated axis-for-axis against the space, its
/// own stride along `axis` answers directly, even under a `Dim::Sym` extent.
/// Otherwise the divmod form: no
/// [`AddressTerm`](crate::ir::launch::AddressTerm) overlapping the axis's
/// `(divisor, modulus)` window may carry a nonzero stride.
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
            // binding and not at another; refuse.
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
/// consuming edge a dense layout over the reading space, so the view's spec
/// vector must be composed into the layout for the dependence query to see
/// the read rather than the spelling.
///
/// Only a single-node spine is collapsed: composing a multi-node spec vector
/// is decidable only once every extent is known.
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

/// Two edges that read the same elements of the same value, compared on `src`
/// plus the index map.
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
/// a lift's root down to a folded subterm.
///
/// The folded subterm may sit anywhere on such a path, not only at the lift's
/// root.
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
    /// `Add` for a translation. Two multiplications commute, two translations
    /// commute, a multiplication and a translation do not.
    const fn action(&self) -> BinOp {
        match self {
            Self::Mul(_) | Self::Div(_) | Self::Neg => BinOp::Mul,
            Self::Add(_) => BinOp::Add,
        }
    }

    /// Apply this step, dropping a multiplication by one and an addition of
    /// zero. A retargeted body lift is this chain applied to the action's
    /// identity, so an un-simplified `1 * v` would cost one multiply per
    /// element per lane.
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
/// `|peels| + |inner| < |e|` strictly, so the laws that call this are
/// well-founded.
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

fn children_of(e: &ScalarExpr) -> SmallVec<[ScalarExpr; 3]> {
    use ScalarKind as K;
    match e.kind() {
        K::Un { x, .. }
        | K::Cast { x, .. }
        | K::Bitcast { x, .. }
        | K::Round { x, .. }
        | K::Splat { x, .. } => smallvec![x.clone()],
        K::Bin { a, b, .. } | K::Cmp { a, b, .. } | K::Dot { a, b } => {
            smallvec![a.clone(), b.clone()]
        }
        K::Select { c, t, f } => smallvec![c.clone(), t.clone(), f.clone()],
        _ => SmallVec::new(),
    }
}

fn contains(e: &ScalarExpr, pred: &dyn Fn(&ScalarExpr) -> bool) -> bool {
    pred(e) || children_of(e).iter().any(|c| contains(c, pred))
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
    for c in children_of(e) {
        arg_indices(&c, out);
    }
}

fn is_lit_value(e: &ScalarExpr, v: f32) -> bool {
    matches!(e.kind(), ScalarKind::Lit(l) if splat_f32(l.0) == v)
}

fn splat_f32(s: Splat) -> f32 {
    match s {
        Splat::F32(v) => v,
        Splat::F16(b) => half::f16::from_bits(b).to_f32(),
        Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
        Splat::U32(v) => v as f32,
        Splat::I32(v) => v as f32,
    }
}

/// Sort every commutative binop's children into a fixed order, so a guard
/// spelled `Add(Arg(k), Arg(w + k))` still matches `Add(Arg(w + k), Arg(k))`.
///
/// `ScalarExpr` does not canonicalize on construction and the e-graph
/// canonicalizes only `Op::Union` children.
fn canon(e: &ScalarExpr) -> ScalarExpr {
    use ScalarKind as K;
    match e.kind() {
        K::Un { op, x } => ScalarExpr::un(*op, canon(x)),
        K::Bin { op, a, b } => {
            let (a, b) = (canon(a), canon(b));
            if op.is_commutative() && b.structural_hash() < a.structural_hash() {
                ScalarExpr::bin(*op, b, a)
            } else {
                ScalarExpr::bin(*op, a, b)
            }
        }
        K::Cmp { op, a, b } => ScalarExpr::cmp(*op, canon(a), canon(b)),
        K::Select { c, t, f } => ScalarExpr::select(canon(c), canon(t), canon(f)),
        K::Cast { to, x } => ScalarExpr::cast(*to, canon(x)),
        K::Bitcast { to, x } => ScalarExpr::bitcast(*to, canon(x)),
        K::Round { mode, x } => ScalarExpr::round(*mode, canon(x)),
        _ => e.clone(),
    }
}

fn expr_eq(a: &ScalarExpr, b: &ScalarExpr) -> bool {
    a == b || canon(a) == canon(b)
}

/// Drop operand edges no lift reads any more, renumbering what is left.
/// `None` when every operand is still read.
///
/// An edge the body never reads is traffic the cost model charges and the
/// kernel never performs; keeping it would price the hoisted form at the
/// unhoisted form's bandwidth.
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
    let renumber = |i: u32| used.iter().position(|&u| u == i).map_or(i, |p| p as u32);
    Some((
        lifts.iter().map(|l| map_args(l, &renumber)).collect(),
        used.iter().map(|&i| ops[i as usize].clone()).collect(),
    ))
}

/// The binop a single-slot carrier accumulates with, modulo commutation.
///
/// Admits a `Vector` slot: promotion changes a slot's width, never its
/// algebra.
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
            // A monotone row over a unary that is partial on the operand
            // dtype can turn a number into a NaN.
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
/// axis-invariant layout proves invariance, not positivity; those rows admit
/// a `Lit`, whose sign is decidable, and nothing else.
///
/// Scaling an additive fold is sign-blind — `sum(x * c) == sum(x) * c` holds
/// for every `c` — so any axis-invariant expression is admissible.
const fn sign_sensitive(row: &HomRow) -> bool {
    matches!(row.h, HomShape::MulByLit | HomShape::DivByLit)
        && matches!(row.from, BinOp::Max | BinOp::Min)
}

/// May `c` be peeled out as this row's factor?
///
/// The sign-sensitive branch demands a **positive** literal:
/// `max(x * -2) == min(x) * -2`, not `max(x) * -2`.
fn admissible_scale(c: &ScalarExpr, row: &HomRow, invariant: &dyn Fn(&ScalarExpr) -> bool) -> bool {
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
/// admits no surround at all; with no surround there is nothing to commute
/// with, and the row still fires at the root.
const fn h_action(h: HomShape) -> Option<BinOp> {
    match h {
        HomShape::MulByLit | HomShape::DivByLit | HomShape::TotalAntitone(UnOp::Neg) => {
            Some(BinOp::Mul)
        }
        HomShape::AddInvariant => Some(BinOp::Add),
        _ => None,
    }
}

/// The homomorphism theorem, read outward and inward.
///
/// Outward: a lift of the form `L(h(inner))` — `h` applied anywhere on an
/// `accum`-linear path — becomes `Fold{row.from}(L(inner))` with `h` applied
/// once, outside.
///
/// Inward: a `post` of the form `h(Arg(0))` moves into the lift and the
/// accumulator becomes `row.to`. Both alternatives stay live; cost decides.
///
/// Greedy: every hoistable factor is peeled in one firing, and `|lift|`
/// strictly decreases at each step, so the law is well-founded.
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
    // One slot only: a multi-slot carrier couples its slots through `merge`,
    // so changing one slot's monoid changes what every sibling reads.
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
    let Op::Launch(Launch::Fold {
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
    // The peeled factor ends up outside the reduced axis and every promoted
    // axis: a factor that varies across lanes cannot be applied once at the end.
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
        if reads_any_index(e) {
            return false;
        }
        let mut used = Vec::new();
        arg_indices(e, &mut used);
        used.iter()
            .all(|&i| inv.get(i as usize).copied() == Some(true))
    };

    // Peel greedily. Each step narrows the lift and records the `h` that has to
    // be re-applied outside, outermost first.
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
            // needs each equality in turn. An empty surround is `L = id` and
            // asks nothing of the row's action.
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
            if m.c.as_ref().is_some_and(reads_any_index) {
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
    // so a declined firing leaves no orphan.
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
    // A `post` that casts would make the map and the fold two different values
    // and the union a lie.
    if body.dtype() != f.own().dtype {
        return None;
    }

    // The inner fold: same slot shape, `cur` as the accumulation, an identity
    // `post`, and no edge for the factor that just left.
    let inner = b
        .add_launch(Launch::Fold {
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
/// sitting in `post` moves into the lift. Minted so both directions compete;
/// `h` must be closed because a `post` reads accumulator slots, not operands.
fn hoist_inward(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
    accum: BinOp,
) -> Option<Id> {
    let Op::Launch(Launch::Fold {
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
        .add_launch(Launch::Fold {
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
        tie: matches!(op, BinOp::Max | BinOp::Min).then(|| c.tie.unwrap_or(TiePolicy::SplitEvenly)),
    })
}

/// An operand restated as a plain strided read over `space`.
///
/// An `Unflatten` map with one sub-axis per logical axis is a stride vector —
/// the two spellings denote the same read. A map with several sub-axes per
/// axis is a genuine divmod decomposition and is left alone.
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

/// A read stated over the full `space`, restated over the iteration space.
///
/// `None` when the operand varies along a promoted axis: such a value is not
/// a function of the iteration coordinate, so no nest over the iteration
/// space can produce it.
pub(crate) fn on_iter_space(o: &Operand, space: &IndexSpace, vec_axes: &[u32]) -> Option<Operand> {
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

fn project_operand(
    o: &Operand,
    space: &IndexSpace,
    drop: &[u32],
    carrier: &Carrier,
) -> Option<Operand> {
    // Alias only: restating an affine `Unflatten` here lets the rule fire on
    // nests it must decline (three GPU sampling cases — top-p, min-p,
    // mirostat — computed wrong values). If an edge arrives here as
    // `Unflatten`, the fix is at the mint.
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
/// [`HOLE_D`]. A row whose `T` is not `v (+) f(delta)` for a single binop is
/// declined rather than guessed at.
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
/// The rule names no producer. What it matches is an operand with no stride
/// over the reduced axis whose value is a fold over that same axis of reads
/// this fold already performs, and whose element expression is exactly the
/// `u` this fold subtracts. One [`RETARGET_TABLE`] row must cover every
/// retargeted slot's lift after `linear_factor` peels each slot's surround.
///
/// The law never invents a reference: it fires only where the source program
/// already computed one.
///
/// Two unions are written: this fold to the body slot view, and the reference
/// fold to the reference slot view. Without the second, extraction keeps
/// computing the reference separately.
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

    // The reference is a nest over the ITERATION space, one rank shorter than
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
        // An operand that varies along a promoted axis cannot name a
        // reference over the iteration space.
        if on_iter_space(&reads[r].0, space, vec_axes).is_none() {
            continue;
        }
        if invariant_along(&proj[r].0, &iter, iter_axis) != Some(true) {
            continue;
        }
        let Some(reference) = reference_fold(b, reads[r].1, &iter, iter_axis, &proj) else {
            continue;
        };
        // Acyclicity: no operand the joint fold keeps may be the reference,
        // since the second union puts the reference's class above the joint
        // node.
        if (0..ops.len()).any(|i| i != r && class_members(b, reads[i].1).contains(&reference.id)) {
            continue;
        }
        if let Some(hit) = mint_retarget(b, id, node, r, &reference) {
            return Some(hit);
        }
    }
    None
}

/// The non-`Union` members reachable downward from `id`. A `Builder` cannot
/// see a class's root, so this sees exactly the alternatives an operand edge
/// already names.
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

/// A reduction nest in either spelling — the `Logical::Fold` the frontend
/// built or the `Launch::Fold` it was lowered to.
///
/// Equality in this e-graph is not congruent, so a consuming edge names
/// whichever id it was handed while both denote the same value.
struct FoldView {
    space: IndexSpace,
    axis: u32,
    carrier: Carrier,
    ops: Vec<Operand>,
}

fn fold_view(b: &Builder<'_>, id: Id) -> Option<FoldView> {
    match b.node(id).op.clone() {
        Op::Launch(Launch::Fold {
            space,
            axis,
            vec_axes,
            carrier,
            post,
            ops,
            ..
        }) => {
            // A `post` means a reader sees `post(rho)` and not `rho`; there is
            // nothing to redirect it to.
            (vec_axes.is_empty() && post.len() == 1 && post[0].kind() == &ScalarKind::Arg(0))
                .then_some(FoldView {
                    space,
                    axis,
                    carrier,
                    ops,
                })
        }
        Op::Logical(Logical::Fold {
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

/// Is `src` a fold over the same axis of the same reads?
///
/// Pure address-map and id comparison; no producer is named.
fn reference_fold(
    b: &Builder<'_>,
    src: Id,
    space: &IndexSpace,
    axis: u32,
    reader: &[(Operand, Id)],
) -> Option<RefFold> {
    for cand in class_members(b, src) {
        let Some(v) = fold_view(b, cand) else {
            continue;
        };
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

/// How many producers `common_basis` will substitute before giving up. This
/// bounds work rather than deciding anything.
const MAX_EXPANSIONS: usize = 8;

/// The reference's element expression, rewritten over the reading fold's
/// operand list, or `None` when the two folds have no common basis.
///
/// The two nests are absorbed independently, so the reference's expression is
/// brought down to the reader's basis by the same substitution the fusion law
/// performs, and the two are compared there. A producer is admitted only if
/// it is an elementwise value at a covered index space.
fn common_basis(b: &Builder<'_>, v: &FoldView, reader: &[(Operand, Id)]) -> Option<ScalarExpr> {
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
    let Op::Launch(Launch::Fold {
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
        // mint replaces each slot's merge with the retargeted form.
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
        // linear surround is peeled: every slot gets the same rescale factor
        // or the law does not apply.
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
        // many lanes as it has positions.
        let (lanes, body_lanes) = (joint.lanes()?, carrier.lanes()?);
        let Some(body_axis) = carrier.out_dim() else {
            continue;
        };
        // The output's free dims: `space` minus the reduced axis and every
        // promoted axis, since a promoted extent lives in the carrier's lanes.
        let free: Vec<Dim> = space
            .dims
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != *axis as usize && !vec_axes.contains(&(*i as u32)))
            .map(|(_, d)| *d)
            .collect();
        // Both readbacks are checked expressible before the joint node
        // exists, so a decline costs no orphan.
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
            .add_launch(Launch::Fold {
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
        // The second union stops extraction running the reference pass a
        // second time.
        b.union(reference.id, ref_view).ok()?;
        return Some(hit);
    }
    None
}

/// Read lanes `[off, off + len)` of a joint fold's trailing carrier axis back
/// as an ordinary strided view.
///
/// Minted as the one `Map { body: Arg(0) }` a view lowers to: the offset is
/// not spellable in a single relative spec vector.
fn view_expressible(free: &[Dim], lanes: u64, off: u64, len: u64, want_axis: Option<Dim>) -> bool {
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
