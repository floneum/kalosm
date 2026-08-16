//! The generic fold algebra: an N-slot accumulator with a lift and an
//! associative merge, both ordinary [`ScalarExpr`]s.
//!
//! A carrier is data: slot shapes, per-slot identities, and two expressions.
//! Online softmax, Welford, log-sum-exp, split-K and plain reductions are all
//! values a rewrite rule constructs; `Add`, `Mul`, `Max` and `Min` are
//! [`Carrier::binop`] values.
//!
//! # The laws that produce the interesting carriers
//!
//! **Tupling** ([`Carrier::tuple`]). Two folds over the same axis of the same
//! input are one fold over the concatenated accumulator, with structurally
//! identical slots deduplicated as canonicalization. Exactly value-preserving:
//! every slot folds in precisely the order it folded alone, so this needs no
//! `reassoc` permission.
//!
//! **Promotion** ([`Carrier::promote`]). A free axis of the nest moves into the
//! accumulator's data space: `Scalar -> Vector(d)`. Register tiling, the CPU
//! lane tile and flash's output accumulator are one law.
//!
//! **Retargeting** ([`Carrier::retarget`]). A reduction-carried dependence on
//! another reduction over the same axis is discharged by carrying the reference
//! alongside and rescaling by `T(rho_s - rho)`. At `h = exp` and a scalar
//! module this is online softmax; at a `Vector(Dh)` module it is flash's output
//! accumulator, with the *same* expression, because `(R^Dh, +)` is a monoid.
//!
//! Every carrier, however minted, owes [`Carrier::identity_closed`]:
//! `merge(identity, identity) == identity`. A rescale spelled without
//! [`Carrier::safe_delta`] computes `0 * exp((-inf) - (-inf)) = NaN`, and
//! every workgroup-tree and subgroup schedule merges padded identity lanes.

use crate::dtype::{Dtype, Splat};
use crate::ir::logical::TiePolicy;
use crate::scalar::{BinOp, CmpOp, ScalarExpr, ScalarKind, UnOp};
use crate::shape::Dim;
use smallvec::{SmallVec, smallvec};

/// The shape of one accumulator slot.
///
/// [`SlotTy::Vector`] carries one value per position of a dim appended to the
/// output shape — attention's `sum p*v` accumulator, which has to be rescaled
/// by the same factor as the running sum and therefore has to be a slot of the
/// *same* carrier rather than a separate fold.
///
/// A `Vector` extent should be [`Dim::Const`]: a symbolic private-array extent
/// is allocatable on neither backend. [`Carrier::lanes`] returns `None` on a
/// symbolic extent and every guard that reads it declines.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SlotTy {
    Scalar,
    Vector(Dim),
}

impl SlotTy {
    pub fn lanes(&self) -> Option<u64> {
        match self {
            Self::Scalar => Some(1),
            Self::Vector(d) => d.as_const(),
        }
    }
}

/// A fold algebra: per-slot identities, a lift from an element into the
/// accumulator, and an associative merge of two accumulators.
///
/// * `lift[k]` is an expression over `Arg(0..n_ops)` — the fold's **operands**.
///   This is the one place an element expression lives; `Fold` carries no
///   separate `pre`.
/// * `merge[k]` is an expression over `Arg(0..w)` (the left accumulator) and
///   `Arg(w..2w)` (the right one), `w = slots.len()`. Cross-*slot* reads are
///   legal and required — flash's `l` and `o` both read `m`. A `Vector` slot's
///   merge is positionwise: no merge may read another position.
///
/// The element-absorption form used by a sequential inner loop is
/// `merge(acc, lift(x))`; a tree reduction uses `merge` directly on partial
/// accumulators.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Carrier {
    pub slots: SmallVec<[SlotTy; 4]>,
    pub identity: SmallVec<[Splat; 4]>,
    pub lift: SmallVec<[ScalarExpr; 4]>,
    pub merge: SmallVec<[ScalarExpr; 4]>,
    /// Declared associativity. A non-associative carrier is legal but may not
    /// be split or tree-reduced; `fold_split` and every collective strategy
    /// guard on it.
    pub associative: bool,
    /// How an extremum reduction splits its gradient among tied elements. Read
    /// **only** by `fold_adjoint`: an autograd attribute, never a compiler
    /// decision.
    pub tie: Option<TiePolicy>,
}

/// The result of [`Carrier::tuple`]: the joint carrier plus, for each side,
/// where its slot `k` ended up after deduplication.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tupled {
    pub carrier: Carrier,
    pub lhs: SmallVec<[u8; 4]>,
    pub rhs: SmallVec<[u8; 4]>,
}

/// How the right-hand carrier's **operand** indices map onto the unified
/// operand list a tupled fold reads. `lift` is renumbered through this before
/// the join; `merge` is renumbered by slot position and never touches it.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ArgRemap {
    pub map: SmallVec<[u32; 4]>,
}

impl ArgRemap {
    /// The identity on `n` operands — both folds already read the same list.
    pub fn identity(n: usize) -> Self {
        Self {
            map: (0..n as u32).collect(),
        }
    }
    /// Append `other`'s operands after `n_self` of ours.
    pub fn shifted(n_self: usize, n_other: usize) -> Self {
        Self {
            map: (0..n_other as u32).map(|i| i + n_self as u32).collect(),
        }
    }
    fn at(&self, i: u32) -> u32 {
        self.map.get(i as usize).copied().unwrap_or(i)
    }
}

impl Carrier {
    /// Slot count.
    pub fn width(&self) -> usize {
        self.slots.len()
    }

    /// Sum of slot lanes — the extent of the carrier axis appended to the
    /// output shape. `None` if any `Vector` extent is symbolic.
    pub fn lanes(&self) -> Option<u64> {
        self.slots
            .iter()
            .try_fold(0u64, |a, s| a.checked_add(s.lanes()?))
    }

    /// Lane offset of slot `i` in the appended carrier axis.
    pub fn slot_offset(&self, i: usize) -> Option<u64> {
        self.slots
            .get(..i)?
            .iter()
            .try_fold(0u64, |a, s| a.checked_add(s.lanes()?))
    }

    /// The dim a fold appends to its output shape: `None` for a single scalar
    /// slot (nothing appended), `Some(d)` otherwise. The outer `None` means
    /// undecidable — a multi-slot carrier with a symbolic `Vector` extent.
    pub fn out_dim(&self) -> Option<Option<Dim>> {
        match self.slots.as_slice() {
            [SlotTy::Scalar] => Some(None),
            [SlotTy::Vector(d)] => Some(Some(*d)),
            _ => self.lanes().map(|n| Some(Dim::Const(n))),
        }
    }

    /// Single-slot binop recognition — the only thing Kernel's hardware fast
    /// path reads. `Some(op)` exactly when this carrier is one scalar slot
    /// merged by `op`, whatever its lift does, so a fold with a fused `pre`
    /// still emits `subgroupAdd`.
    pub fn kind(&self) -> Option<BinOp> {
        if self.slots.len() != 1 || !matches!(self.slots[0], SlotTy::Scalar) {
            return None;
        }
        let ScalarKind::Bin { op, a, b } = self.merge[0].kind() else {
            return None;
        };
        let (a, b) = (a.kind(), b.kind());
        let forward = matches!((a, b), (ScalarKind::Arg(0), ScalarKind::Arg(1)));
        let swapped = matches!((a, b), (ScalarKind::Arg(1), ScalarKind::Arg(0)));
        (forward || (swapped && op.is_commutative())).then_some(*op)
    }

    /// The single-slot case: a plain binary reduction.
    ///
    /// An extremum defaults to `TiePolicy::SplitEvenly` so that every spelling
    /// of `max` in the frontend hash-conses to one node; `with_tie` overrides
    /// it where parity with a reference trainer demands `FirstWins`.
    pub fn binop(op: BinOp, identity: Splat, dtype: Dtype) -> Self {
        Self {
            slots: smallvec![SlotTy::Scalar],
            identity: smallvec![identity],
            lift: smallvec![ScalarExpr::arg(0, dtype)],
            merge: smallvec![ScalarExpr::bin(
                op,
                ScalarExpr::arg(0, dtype),
                ScalarExpr::arg(1, dtype),
            )],
            associative: op.is_associative(),
            tie: matches!(op, BinOp::Max | BinOp::Min).then_some(TiePolicy::SplitEvenly),
        }
    }

    /// The per-dtype identity of a scalar binop reduction, or `None` for a
    /// quantized dtype (a quantized value is never an accumulator) or an op
    /// with no identity in the vocabulary.
    pub fn binop_identity(op: BinOp, dtype: Dtype) -> Option<Splat> {
        let of = |f: f32, u: u32, i: i32| -> Option<Splat> {
            Some(match dtype {
                Dtype::F32 => Splat::F32(f),
                Dtype::F16 => Splat::F16(half::f16::from_f32(f).to_bits()),
                Dtype::BF16 => Splat::BF16(half::bf16::from_f32(f).to_bits()),
                Dtype::U32 => Splat::U32(u),
                Dtype::I32 => Splat::I32(i),
                Dtype::Q(_) => return None,
            })
        };
        match op {
            BinOp::Add => of(0.0, 0, 0),
            BinOp::Mul => of(1.0, 1, 1),
            BinOp::Max => of(f32::NEG_INFINITY, 0, i32::MIN),
            BinOp::Min => of(f32::INFINITY, u32::MAX, i32::MAX),
            _ => None,
        }
    }

    /// Replace the lift: an element expression over the fold's operands, one
    /// per slot.
    #[must_use]
    pub fn with_lift(mut self, lift: impl IntoIterator<Item = ScalarExpr>) -> Self {
        self.lift = lift.into_iter().collect();
        self
    }

    /// Declare the tie policy an extremum reduction's adjoint reads.
    #[must_use]
    pub fn with_tie(mut self, tie: TiePolicy) -> Self {
        self.tie = Some(tie);
        self
    }

    /// Does any `lift`, `merge` expression read `IndexOf(axis)`? The guard that
    /// correctly refuses to promote an axis a positional term depends on.
    pub fn reads_index_of(&self, axis: u32) -> bool {
        self.lift
            .iter()
            .chain(&self.merge)
            .any(|e| reads_index_of(e, axis))
    }

    /// The tupling law, with slot deduplication as canonicalization.
    ///
    /// Two folds over the same axis of the same input are one fold over the
    /// concatenated accumulator. Slots equal in `(SlotTy, identity, lift,
    /// merge modulo slot renumbering and modulo commutation)` collapse to one,
    /// so joining `(m, l)` with `(m, o)` yields three slots.
    ///
    /// `remap` renumbers `other`'s lift onto the unified operand list. Merge
    /// expressions are renumbered by slot position and never see it.
    ///
    /// Deduplication is restricted to slots whose merge reads only their own
    /// position: such a slot's value is a function of its own history alone, so
    /// two structurally identical ones are equal at every point. A slot whose
    /// merge reads a *sibling* is left alone.
    pub fn tuple(&self, other: &Carrier, remap: &ArgRemap) -> Tupled {
        let ns = self.width();
        let other_lift: SmallVec<[ScalarExpr; 4]> = other
            .lift
            .iter()
            .map(|e| map_args(e, &|i| remap.at(i)))
            .collect();

        // Where each of `other`'s slots lands. `None` until decided.
        let mut rhs: SmallVec<[u8; 4]> = SmallVec::new();
        let mut extra: Vec<usize> = Vec::new(); // other-slot indices that survive
        for j in 0..other.width() {
            let sig = self_contained_signature(other, j, &other_lift[j]);
            let hit = sig.as_ref().and_then(|s| {
                (0..ns).find(|&k| self_contained_signature(self, k, &self.lift[k]).as_ref() == Some(s))
            });
            match hit {
                Some(k) => rhs.push(k as u8),
                None => {
                    rhs.push((ns + extra.len()) as u8);
                    extra.push(j);
                }
            }
        }
        let w = ns + extra.len();
        let lhs: SmallVec<[u8; 4]> = (0..ns as u8).collect();

        // Self's `a` block is already right; its `b` block moves to `w..w+ns`.
        let mut merge: SmallVec<[ScalarExpr; 4]> = self
            .merge
            .iter()
            .map(|e| {
                map_args(e, &|i| {
                    if (i as usize) < ns {
                        i
                    } else {
                        i - ns as u32 + w as u32
                    }
                })
            })
            .collect();
        // Other's slot `j` reads its own numbering; renumber through `rhs`.
        let no = other.width();
        for &j in &extra {
            merge.push(map_args(&other.merge[j], &|i| {
                if (i as usize) < no {
                    u32::from(rhs[i as usize])
                } else {
                    w as u32 + u32::from(rhs[i as usize - no])
                }
            }));
        }

        let mut slots = self.slots.clone();
        let mut identity = self.identity.clone();
        let mut lift = self.lift.clone();
        for &j in &extra {
            slots.push(other.slots[j]);
            identity.push(other.identity[j]);
            lift.push(other_lift[j].clone());
        }

        Tupled {
            carrier: Carrier {
                slots,
                identity,
                lift,
                merge,
                associative: self.associative && other.associative,
                tie: self.tie.or(other.tie),
            },
            lhs,
            rhs,
        }
    }

    /// The same algebra reading partial accumulators instead of elements:
    /// `lift[k] = Arg(k)`.
    ///
    /// The outer level of a split must use this: reusing the inner carrier
    /// applies `lift` to a partial max and silently computes a wrong value.
    /// The resulting fold takes ONE operand carrying the inner fold's
    /// trailing carrier axis, never `width` operands.
    pub fn as_merge(&self) -> Carrier {
        Carrier {
            slots: self.slots.clone(),
            identity: self.identity.clone(),
            lift: (0..self.width())
                .map(|k| ScalarExpr::arg(k as u32, self.identity[k].dtype()))
                .collect(),
            merge: self.merge.clone(),
            associative: self.associative,
            tie: self.tie,
        }
    }

    /// Promotion: every `Scalar` slot becomes `Vector(extent)`; an existing
    /// `Vector(d)` becomes `Vector(d * extent)`, row-major over the promoted
    /// axes. Repeated promotion coalesces, so `TM x TN` register tiling is two
    /// firings.
    ///
    /// `None` when `extent` is not [`Dim::Const`]: a symbolic private-array
    /// extent is allocatable on neither backend. The caller checks the
    /// positionwise condition with [`Carrier::reads_index_of`].
    pub fn promote(&self, extent: Dim) -> Option<Carrier> {
        let e = extent.as_const()?;
        let slots = self
            .slots
            .iter()
            .map(|s| match s {
                SlotTy::Scalar => Some(SlotTy::Vector(Dim::Const(e))),
                SlotTy::Vector(d) => Some(SlotTy::Vector(Dim::Const(d.as_const()?.checked_mul(e)?))),
            })
            .collect::<Option<_>>()?;
        Some(Carrier {
            slots,
            ..self.clone()
        })
    }

    /// `Delta = select(a == b, identity, a - b)`.
    ///
    /// Without it `merge(identity, identity)` on a shifted carrier is
    /// `0 * exp((-inf) - (-inf)) = NaN`, and merging `(-inf, NaN)` against a
    /// real partial propagates it. Every workgroup-tree and subgroup schedule
    /// merges padded identity lanes, and a fully-masked causal row hits it too.
    pub fn safe_delta(a: ScalarExpr, b: ScalarExpr, e: Splat) -> ScalarExpr {
        ScalarExpr::select(
            ScalarExpr::cmp(CmpOp::Eq, a.clone(), b.clone()),
            ScalarExpr::lit(e),
            ScalarExpr::bin(BinOp::Sub, a, b),
        )
    }

    /// Retargeting: carry the reference `rho` alongside the body and
    /// rescale by `T(rho_s - rho)`.
    ///
    /// ```text
    /// slots    = stat.slots ++ body.slots
    /// identity = stat.identity ++ body.identity
    /// lift     = stat.lift    ++ body.lift
    /// merge    = ( stat.merge,
    ///              T(D_a).V_a  (+)  T(D_b).V_b ),  D_s = safe_delta(rho_s, rho)
    /// ```
    ///
    /// `ref_slot` names the slot of `stat` holding `rho`. One table row covers
    /// every retargeted slot: a `Vector(Dh)` slot gets the same factor as a
    /// `Scalar` one. `None` when `ref_slot` is out of range or the row's
    /// accumulation binop has no identity in `dtype`.
    ///
    /// The caller supplies `body.lift` already written at `rho := u`, which is
    /// legal because `h(e) = id`: an element enters as `h(u - u) . w = w`, so
    /// the first element needs no special case.
    pub fn retarget(
        stat: &Carrier,
        row: &RetargetRow,
        body: &Carrier,
        ref_slot: usize,
    ) -> Option<Carrier> {
        if ref_slot >= stat.width() || body.width() == 0 {
            return None;
        }
        let ns = stat.width();
        let nb = body.width();
        let w = ns + nb;
        let dtype = stat.identity[ref_slot].dtype();
        let e = Self::binop_identity(BinOp::Add, dtype)?;

        let up = |e: &ScalarExpr, n: usize, base: usize| {
            map_args(e, &|i| {
                if (i as usize) < n {
                    i + base as u32
                } else {
                    i - n as u32 + w as u32 + base as u32
                }
            })
        };

        let stat_merge: SmallVec<[ScalarExpr; 4]> =
            stat.merge.iter().map(|m| up(m, ns, 0)).collect();
        let rho = stat_merge[ref_slot].clone();
        let d_a = Self::safe_delta(ScalarExpr::arg(ref_slot as u32, dtype), rho.clone(), e);
        let d_b = Self::safe_delta(
            ScalarExpr::arg((w + ref_slot) as u32, dtype),
            rho,
            e,
        );

        let mut merge = stat_merge;
        for k in 0..nb {
            let v_a = ScalarExpr::arg((ns + k) as u32, dtype);
            let v_b = ScalarExpr::arg((w + ns + k) as u32, dtype);
            merge.push(ScalarExpr::bin(
                row.accum,
                (row.retarget)(&d_a, &v_a, dtype),
                (row.retarget)(&d_b, &v_b, dtype),
            ));
        }

        Some(Carrier {
            slots: stat.slots.iter().chain(&body.slots).copied().collect(),
            identity: stat
                .identity
                .iter()
                .chain(&body.identity)
                .copied()
                .collect(),
            lift: stat.lift.iter().chain(&body.lift).cloned().collect(),
            merge,
            associative: stat.associative && body.associative,
            tie: stat.tie.or(body.tie),
        })
    }

    /// The carrier obligation:
    ///
    /// * `merge(identity, identity) == identity`;
    /// * `merge(identity, lift(x)) == lift(x)` over the probes;
    /// * `merge` is associative when `associative` is declared.
    ///
    /// An expression the host evaluator does not cover reports "unknown", and
    /// unknown passes.
    pub fn identity_closed(&self, probes: &[f32]) -> bool {
        let w = self.width();
        if self.identity.len() != w || self.lift.len() != w || self.merge.len() != w || w == 0 {
            return false;
        }
        let ident: Vec<f32> = self.identity.iter().map(splat_f32).collect();

        let Some(ii) = self.eval_merge(&ident, &ident) else {
            return true;
        };
        if !same(&ii, &ident) {
            return false;
        }

        for &x in probes {
            let Some(l) = self.eval_lift(&[x]) else {
                continue;
            };
            let Some(m) = self.eval_merge(&ident, &l) else {
                continue;
            };
            if !same(&m, &l) {
                return false;
            }
        }

        if self.associative && probes.len() >= 3 {
            let (a, b, c) = (
                self.eval_lift(&[probes[0]]),
                self.eval_lift(&[probes[1]]),
                self.eval_lift(&[probes[2]]),
            );
            if let (Some(a), Some(b), Some(c)) = (a, b, c)
                && let (Some(ab), Some(bc)) = (self.eval_merge(&a, &b), self.eval_merge(&b, &c))
                && let (Some(l), Some(r)) =
                    (self.eval_merge(&ab, &c), self.eval_merge(&a, &bc))
                && !close(&l, &r)
            {
                return false;
            }
        }
        true
    }

    /// The `(slot, position)` each accumulator lane belongs to, in lane
    /// order. A `Scalar` slot is one lane; a `Vector(d)` slot is `d`.
    ///
    /// This is the coordinate system Kernel reduces in: `Stmt::Reduce` carries
    /// one value, one `merge` expression and one output `Local` per lane.
    pub fn lane_slots(&self) -> Option<Vec<(usize, u64)>> {
        let mut out = Vec::new();
        for (k, s) in self.slots.iter().enumerate() {
            for p in 0..s.lanes()? {
                out.push((k, p));
            }
        }
        Some(out)
    }

    /// Per-lane identities, in lane order.
    pub fn identity_lanes(&self) -> Option<Vec<Splat>> {
        Some(
            self.lane_slots()?
                .into_iter()
                .map(|(k, _)| self.identity[k])
                .collect(),
        )
    }

    /// `merge`, expanded from one expression per **slot** to one per **lane**.
    ///
    /// In the result, lane `i` of the left accumulator is `Arg(i)` and lane `i`
    /// of the right is `Arg(lanes + i)`, so a lowering evaluates each expression
    /// against `lhs_loads ++ rhs_loads` with no further renumbering.
    ///
    /// A `Vector` slot's merge is positionwise: at position `p` a read of
    /// another `Vector` slot resolves to that slot's position `p`, and a read
    /// of a `Scalar` slot to its single lane. `None` when a `Vector` extent is
    /// symbolic, when an `Arg` is out of range, or when two `Vector` slots
    /// that read each other disagree in extent, because clamping a position
    /// would silently compute the wrong element.
    pub fn merge_lanes(&self) -> Option<Vec<ScalarExpr>> {
        let w = self.width();
        let lanes = self.lane_slots()?;
        let total = lanes.len();
        let widths: Vec<u64> = self.slots.iter().map(|s| s.lanes()).collect::<Option<_>>()?;
        let bases: Vec<u64> = (0..w).map(|k| self.slot_offset(k)).collect::<Option<_>>()?;

        let mut out = Vec::with_capacity(total);
        for &(k, p) in &lanes {
            // Refuse before rewriting: an out-of-range `Arg`, or a cross-slot
            // read whose extent does not match this slot's.
            let bad = std::cell::Cell::new(false);
            let resolve = |a: u32| -> u32 {
                let (j, right) = if (a as usize) < w {
                    (a as usize, false)
                } else {
                    (a as usize - w, true)
                };
                if j >= w || a as usize >= 2 * w {
                    bad.set(true);
                    return 0;
                }
                if widths[j] != 1 && widths[j] != widths[k] {
                    bad.set(true);
                    return 0;
                }
                let pos = if widths[j] == 1 { 0 } else { p };
                let lane = (bases[j] + pos) as u32;
                if right { total as u32 + lane } else { lane }
            };
            let e = map_args(&self.merge[k], &resolve);
            if bad.get() {
                return None;
            }
            out.push(e);
        }
        Some(out)
    }

    /// One expression per **slot**, reading `Arg(0..width)`, expanded to one per
    /// **lane**, reading `Arg(0..lanes)`.
    ///
    /// This is [`Carrier::merge_lanes`]'s resolution over a single accumulator
    /// instead of a pair: `post[k]` is written against slot values, while a
    /// lowering holds one register per lane. At lane `(k, p)` a read of
    /// another `Vector` slot resolves to that slot's position `p` and a read
    /// of a `Scalar` slot to its single lane. `None` on the same
    /// disagreements `merge_lanes` refuses.
    pub fn expand_lanes(&self, per_slot: &[ScalarExpr]) -> Option<Vec<ScalarExpr>> {
        let w = self.width();
        if per_slot.len() != w {
            return None;
        }
        let lanes = self.lane_slots()?;
        let widths: Vec<u64> = self.slots.iter().map(|s| s.lanes()).collect::<Option<_>>()?;
        let bases: Vec<u64> = (0..w).map(|k| self.slot_offset(k)).collect::<Option<_>>()?;

        let mut out = Vec::with_capacity(lanes.len());
        for &(k, p) in &lanes {
            let bad = std::cell::Cell::new(false);
            let resolve = |a: u32| -> u32 {
                let j = a as usize;
                if j >= w || (widths[j] != 1 && widths[j] != widths[k]) {
                    bad.set(true);
                    return 0;
                }
                let pos = if widths[j] == 1 { 0 } else { p };
                (bases[j] + pos) as u32
            };
            let e = map_args(&per_slot[k], &resolve);
            if bad.get() {
                return None;
            }
            out.push(e);
        }
        Some(out)
    }

    /// Host evaluation of `lift` over one element's operand values.
    pub fn eval_lift(&self, args: &[f32]) -> Option<Vec<f32>> {
        self.lift.iter().map(|e| eval(e, args)).collect()
    }

    /// Host evaluation of `merge` over two accumulators.
    pub fn eval_merge(&self, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
        let args: Vec<f32> = a.iter().chain(b).copied().collect();
        self.merge.iter().map(|e| eval(e, &args)).collect()
    }

    /// Absorb one element the way a sequential inner loop does.
    pub fn absorb(&self, acc: &[f32], args: &[f32]) -> Option<Vec<f32>> {
        let l = self.eval_lift(args)?;
        self.eval_merge(acc, &l)
    }

    /// The identity, as host floats.
    pub fn identity_f32(&self) -> Vec<f32> {
        self.identity.iter().map(splat_f32).collect()
    }
}

/// The probe set every carrier is checked against. Float-shaped; integer
/// accumulators use [`INT_PROBES`], whose `Max` identity is `0`.
pub const PROBES: [f32; 6] = [-3.5, -1.0, 0.0, 0.5, 2.25, 900.0];
/// Probes for an integer accumulator, where `Max`'s identity is `0` and a
/// negative probe would (correctly) fail `merge(identity, lift(x)) == lift(x)`.
pub const INT_PROBES: [f32; 5] = [0.0, 1.0, 2.0, 7.0, 13.0];

/// The probe set appropriate to an accumulator dtype.
pub fn probes_for(d: Dtype) -> &'static [f32] {
    match d {
        Dtype::U32 | Dtype::I32 => &INT_PROBES,
        _ => &PROBES,
    }
}

/// The syntactic shape of a homomorphism `h` in [`HOM_TABLE`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum HomShape {
    /// `(* c)`, `c` a positive `Lit` invariant along the reduced axis.
    MulByLit,
    /// `(/ c)`, `c` a nonzero `Lit` invariant along the reduced axis.
    DivByLit,
    /// `(+ c)`, `c` invariant along the reduced axis.
    AddInvariant,
    /// A unary **total on the operand dtype** and monotonically increasing.
    TotalMonotone(UnOp),
    /// A unary total on the operand dtype and monotonically decreasing.
    TotalAntitone(UnOp),
}

/// One row of the homomorphism theorem: `h(Fold{from}(x)) == Fold{to}(Map{h}(x))`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct HomRow {
    pub h: HomShape,
    pub from: BinOp,
    pub to: BinOp,
    /// Bit-exact under round-to-nearest, so the row fires with no reassoc
    /// permission, even where `NumericContract::STRICT` holds.
    pub exact_in_float: bool,
}

/// The homomorphism rows.
///
/// Absent, because `ValueFacts` carries no sign or range lattice:
/// `Log : Mul -> Add` (false whenever any `x_i <= 0`), and the general
/// monotone/antitone rows over partial unaries (`Sqrt`, `Log`, `Log2`,
/// `Asin`, `Acos`, `Atanh`) — a row over a partial unary can turn a number
/// into a NaN. `Neg` is the only total unary in the vocabulary today.
pub const HOM_TABLE: &[HomRow] = &[
    HomRow { h: HomShape::MulByLit, from: BinOp::Add, to: BinOp::Add, exact_in_float: false },
    HomRow { h: HomShape::DivByLit, from: BinOp::Add, to: BinOp::Add, exact_in_float: false },
    HomRow { h: HomShape::AddInvariant, from: BinOp::Max, to: BinOp::Max, exact_in_float: true },
    HomRow { h: HomShape::AddInvariant, from: BinOp::Min, to: BinOp::Min, exact_in_float: true },
    HomRow { h: HomShape::TotalAntitone(UnOp::Neg), from: BinOp::Max, to: BinOp::Min, exact_in_float: true },
    HomRow { h: HomShape::TotalAntitone(UnOp::Neg), from: BinOp::Min, to: BinOp::Max, exact_in_float: true },
    HomRow { h: HomShape::TotalMonotone(UnOp::Exp), from: BinOp::Add, to: BinOp::Mul, exact_in_float: false },
];

/// A unary total on `d`: defined for every value of the dtype, so a monotone
/// row over it can never turn a number into a NaN.
pub const fn is_total_on(op: UnOp, d: Dtype) -> bool {
    match op {
        UnOp::Neg | UnOp::Abs => true,
        UnOp::Exp | UnOp::Exp2 | UnOp::ApproximateExp | UnOp::LessApproximateExp => {
            matches!(d, Dtype::F32 | Dtype::F16 | Dtype::BF16)
        }
        _ => false,
    }
}

/// One row of the retargeting law: how `T(delta)` acts on one slot of the
/// module, and which binop accumulates it.
///
/// `retarget` takes one call per slot and the same expression serves a
/// `Scalar` and a `Vector` slot.
#[derive(Copy, Clone)]
pub struct RetargetRow {
    pub name: &'static str,
    /// The reference statistic's own carrier, at an accumulator dtype.
    pub stat: fn(Dtype) -> Carrier,
    /// `T(delta)` applied to one slot.
    pub retarget: fn(&ScalarExpr, &ScalarExpr, Dtype) -> ScalarExpr,
    /// The binop that accumulates the module. A row may not retarget a carrier
    /// whose accumulation binop differs.
    pub accum: BinOp,
}

impl std::fmt::Debug for RetargetRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetargetRow")
            .field("name", &self.name)
            .field("accum", &self.accum)
            .finish()
    }
}

fn running_max(d: Dtype) -> Carrier {
    Carrier::binop(
        BinOp::Max,
        Carrier::binop_identity(BinOp::Max, d).unwrap_or(Splat::F32(f32::NEG_INFINITY)),
        d,
    )
}

/// `T(delta) v = v * h(delta)` — the shift rows' action.
macro_rules! shift_row {
    ($name:literal, $h:expr) => {
        RetargetRow {
            name: $name,
            stat: running_max,
            retarget: |delta, v, _d| {
                ScalarExpr::bin(BinOp::Mul, v.clone(), ScalarExpr::un($h, delta.clone()))
            },
            accum: BinOp::Add,
        }
    };
}

/// The retargeting rows.
///
/// The four shift rows differ only in which exponential the emitter is
/// permitted to use — at `h = exp` the derived carrier is online softmax.
/// `max-plus` is the same law over the `(R, max)` monoid.
///
/// The raw-moment row (whose `T` reads the two counts `n_a`, `n_b`, not the
/// delta alone) and the Goertzel rotation row (whose `T` mixes two slots)
/// need a wider `retarget` signature than one-slot-at-a-time, so they are
/// absent.
pub const RETARGET_TABLE: &[RetargetRow] = &[
    shift_row!("shift-exp", UnOp::Exp),
    shift_row!("shift-exp2", UnOp::Exp2),
    shift_row!("shift-exp-approx", UnOp::ApproximateExp),
    shift_row!("shift-exp-less-approx", UnOp::LessApproximateExp),
    RetargetRow {
        name: "max-plus",
        stat: running_max,
        retarget: |delta, v, _d| ScalarExpr::bin(BinOp::Add, v.clone(), delta.clone()),
        accum: BinOp::Max,
    },
];

/// Rewrite every `Arg(i)` in `e` to `Arg(f(i))`, leaving all other nodes alone.
pub fn map_args(e: &ScalarExpr, f: &dyn Fn(u32) -> u32) -> ScalarExpr {
    use ScalarKind as K;
    match e.kind() {
        K::Arg(i) => ScalarExpr::arg(f(*i), e.dtype()),
        K::Un { op, x } => ScalarExpr::un(*op, map_args(x, f)),
        K::Bin { op, a, b } => ScalarExpr::bin(*op, map_args(a, f), map_args(b, f)),
        K::Cmp { op, a, b } => ScalarExpr::cmp(*op, map_args(a, f), map_args(b, f)),
        K::Select { c, t, f: fe } => {
            ScalarExpr::select(map_args(c, f), map_args(t, f), map_args(fe, f))
        }
        K::Cast { to, x } => ScalarExpr::cast(*to, map_args(x, f)),
        K::Bitcast { to, x } => ScalarExpr::bitcast(*to, map_args(x, f)),
        K::Round { mode, x } => ScalarExpr::round(*mode, map_args(x, f)),
        _ => e.clone(),
    }
}

/// Rewrite every `Arg` leaf's declared dtype, leaving indices alone.
///
/// The floor lowering reads a fold's operands at the **operand** dtype and
/// accumulates at `acc`, so a carrier's `lift` — the one expression that touches
/// elements — is retyped on the way into Launch while `merge`, which reads
/// accumulators, rides through untouched.
pub fn retype_args(e: &ScalarExpr, dtype: Dtype) -> ScalarExpr {
    use ScalarKind as K;
    match e.kind() {
        K::Arg(i) => ScalarExpr::arg(*i, dtype),
        K::Un { op, x } => ScalarExpr::un(*op, retype_args(x, dtype)),
        K::Bin { op, a, b } => {
            ScalarExpr::bin(*op, retype_args(a, dtype), retype_args(b, dtype))
        }
        K::Cmp { op, a, b } => {
            ScalarExpr::cmp(*op, retype_args(a, dtype), retype_args(b, dtype))
        }
        K::Select { c, t, f } => ScalarExpr::select(
            retype_args(c, dtype),
            retype_args(t, dtype),
            retype_args(f, dtype),
        ),
        K::Cast { to, x } => ScalarExpr::cast(*to, retype_args(x, dtype)),
        K::Bitcast { to, x } => ScalarExpr::bitcast(*to, retype_args(x, dtype)),
        K::Round { mode, x } => ScalarExpr::round(*mode, retype_args(x, dtype)),
        _ => e.clone(),
    }
}

fn reads_index_of(e: &ScalarExpr, axis: u32) -> bool {
    use ScalarKind as K;
    match e.kind() {
        K::IndexOf(a) => *a == axis,
        K::Un { x, .. } | K::Cast { x, .. } | K::Bitcast { x, .. } | K::Round { x, .. } => {
            reads_index_of(x, axis)
        }
        K::Bin { a, b, .. } | K::Cmp { a, b, .. } | K::Dot { a, b } => {
            reads_index_of(a, axis) || reads_index_of(b, axis)
        }
        K::Select { c, t, f } => {
            reads_index_of(c, axis) || reads_index_of(t, axis) || reads_index_of(f, axis)
        }
        K::Splat { x, .. } => reads_index_of(x, axis),
        _ => false,
    }
}

/// The signature deduplication compares slots on, or `None` when the slot's
/// merge reads a sibling and is therefore not a function of its own history.
fn self_contained_signature(
    c: &Carrier,
    k: usize,
    lift: &ScalarExpr,
) -> Option<(SlotTy, Splat, ScalarExpr, ScalarExpr)> {
    let w = c.width();
    let foreign = std::cell::Cell::new(false);
    let canon = map_args(&c.merge[k], &|i| {
        if i as usize != k && i as usize != w + k {
            foreign.set(true);
        }
        if i as usize == k { 0 } else { 1 }
    });
    if foreign.get() {
        return None;
    }
    Some((c.slots[k], c.identity[k], lift.clone(), commute_canon(&canon)))
}

/// Sort a commutative binop's children into a canonical order so that
/// `Add(a, b)` and `Add(b, a)` compare equal. `ScalarExpr` does not canonicalize
/// on construction, so a guard spelled `merge[k] == Add(Arg(k), Arg(n+k))`
/// would otherwise silently stop firing on half the graphs.
fn commute_canon(e: &ScalarExpr) -> ScalarExpr {
    use ScalarKind as K;
    match e.kind() {
        K::Un { op, x } => ScalarExpr::un(*op, commute_canon(x)),
        K::Bin { op, a, b } => {
            let (a, b) = (commute_canon(a), commute_canon(b));
            if op.is_commutative() && b.structural_hash() < a.structural_hash() {
                ScalarExpr::bin(*op, b, a)
            } else {
                ScalarExpr::bin(*op, a, b)
            }
        }
        K::Cmp { op, a, b } => ScalarExpr::cmp(*op, commute_canon(a), commute_canon(b)),
        K::Select { c, t, f } => {
            ScalarExpr::select(commute_canon(c), commute_canon(t), commute_canon(f))
        }
        K::Cast { to, x } => ScalarExpr::cast(*to, commute_canon(x)),
        K::Bitcast { to, x } => ScalarExpr::bitcast(*to, commute_canon(x)),
        K::Round { mode, x } => ScalarExpr::round(*mode, commute_canon(x)),
        _ => e.clone(),
    }
}

fn splat_f32(s: &Splat) -> f32 {
    match *s {
        Splat::F32(v) => v,
        Splat::F16(b) => half::f16::from_bits(b).to_f32(),
        Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
        Splat::U32(v) => v as f32,
        Splat::I32(v) => v as f32,
    }
}

fn same(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x == y || (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-6 * x.abs().max(1.0))
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| {
            (x.is_nan() && y.is_nan())
                || (x.is_infinite() && y.is_infinite() && x.signum() == y.signum())
                || (x - y).abs() <= 1e-3 * x.abs().max(1.0)
        })
}

/// A host evaluator over f32, enough to run a carrier's expressions. `None`
/// means "this node is outside the evaluator", which every caller treats as
/// unknown rather than as a failure.
pub fn eval(e: &ScalarExpr, args: &[f32]) -> Option<f32> {
    use ScalarKind as K;
    Some(match e.kind() {
        K::Arg(i) => *args.get(*i as usize)?,
        K::Lit(l) => splat_f32(&l.0),
        K::Un { op, x } => {
            let v = eval(x, args)?;
            match op {
                UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp => v.exp(),
                UnOp::Exp2 => v.exp2(),
                UnOp::Log => v.ln(),
                UnOp::Log2 => v.log2(),
                UnOp::Sqrt => v.sqrt(),
                UnOp::InverseSqrt => 1.0 / v.sqrt(),
                UnOp::Neg => -v,
                UnOp::Abs => v.abs(),
                UnOp::Sin => v.sin(),
                UnOp::Cos => v.cos(),
                UnOp::Tanh => v.tanh(),
                _ => return None,
            }
        }
        K::Bin { op, a, b } => {
            let (x, y) = (eval(a, args)?, eval(b, args)?);
            match op {
                BinOp::Add => x + y,
                BinOp::Sub => x - y,
                BinOp::Mul => x * y,
                BinOp::Div => x / y,
                BinOp::Max => x.max(y),
                BinOp::Min => x.min(y),
                BinOp::Pow => x.powf(y),
                _ => return None,
            }
        }
        K::Cmp { op, a, b } => {
            let (x, y) = (eval(a, args)?, eval(b, args)?);
            let t = match op {
                CmpOp::Lt => x < y,
                CmpOp::Le => x <= y,
                CmpOp::Gt => x > y,
                CmpOp::Ge => x >= y,
                CmpOp::Eq => x == y,
                CmpOp::Ne => x != y,
            };
            if t { 1.0 } else { 0.0 }
        }
        K::Select { c, t, f } => {
            if eval(c, args)? != 0.0 {
                eval(t, args)?
            } else {
                eval(f, args)?
            }
        }
        K::Cast { x, .. } => eval(x, args)?,
        _ => return None,
    })
}

#[doc(hidden)]
pub mod oracle {
    //! Two hand-written algorithms kept as test fixtures: the carriers the
    //! laws derive must match them term for term.
    //!
    //! Nothing in the compiler may call these. They are `pub` only so that
    //! `fusor2-conformance` can run the same two carriers on real hardware.

    use super::*;

    /// `(running max, sum of h(element - running max))` in one pass — online
    /// softmax at `h = exp`. Spelled with [`Carrier::safe_delta`], because the
    /// unguarded form computes `0 * exp((-inf) - (-inf)) = NaN` at
    /// `merge(identity, identity)`.
    pub fn shift_stabilized_sum(h: UnOp, dtype: Dtype) -> Carrier {
        let e = Carrier::binop_identity(BinOp::Add, dtype).unwrap();
        let (m_a, l_a) = (ScalarExpr::arg(0, dtype), ScalarExpr::arg(1, dtype));
        let (m_b, l_b) = (ScalarExpr::arg(2, dtype), ScalarExpr::arg(3, dtype));
        let m = ScalarExpr::bin(BinOp::Max, m_a.clone(), m_b.clone());
        let rescale = |m_side: ScalarExpr, l_side: ScalarExpr| {
            ScalarExpr::bin(
                BinOp::Mul,
                l_side,
                ScalarExpr::un(h, Carrier::safe_delta(m_side, m.clone(), e)),
            )
        };
        Carrier {
            slots: smallvec![SlotTy::Scalar, SlotTy::Scalar],
            identity: smallvec![
                Carrier::binop_identity(BinOp::Max, dtype).unwrap(),
                e
            ],
            lift: smallvec![
                ScalarExpr::arg(0, dtype),
                ScalarExpr::lit(one(dtype))
            ],
            merge: smallvec![
                m.clone(),
                ScalarExpr::bin(BinOp::Add, rescale(m_a, l_a), rescale(m_b, l_b))
            ],
            associative: true,
            tie: None,
        }
    }

    /// `(n, mean, m2)` — the numerically stable variance accumulator.
    pub fn welford(dtype: Dtype) -> Carrier {
        let (n_a, mean_a, m2_a) = (
            ScalarExpr::arg(0, dtype),
            ScalarExpr::arg(1, dtype),
            ScalarExpr::arg(2, dtype),
        );
        let (n_b, mean_b, m2_b) = (
            ScalarExpr::arg(3, dtype),
            ScalarExpr::arg(4, dtype),
            ScalarExpr::arg(5, dtype),
        );
        let n = ScalarExpr::bin(BinOp::Add, n_a.clone(), n_b.clone());
        let delta = ScalarExpr::bin(BinOp::Sub, mean_b, mean_a.clone());
        let safe_n = ScalarExpr::select(
            ScalarExpr::cmp(CmpOp::Eq, n.clone(), ScalarExpr::lit(zero(dtype))),
            ScalarExpr::lit(one(dtype)),
            n.clone(),
        );
        let mean = ScalarExpr::bin(
            BinOp::Add,
            mean_a,
            ScalarExpr::bin(
                BinOp::Div,
                ScalarExpr::bin(BinOp::Mul, delta.clone(), n_b.clone()),
                safe_n.clone(),
            ),
        );
        let m2 = ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::bin(BinOp::Add, m2_a, m2_b),
            ScalarExpr::bin(
                BinOp::Div,
                ScalarExpr::bin(
                    BinOp::Mul,
                    ScalarExpr::bin(BinOp::Mul, delta.clone(), delta),
                    ScalarExpr::bin(BinOp::Mul, n_a, n_b),
                ),
                safe_n,
            ),
        );
        Carrier {
            slots: smallvec![SlotTy::Scalar, SlotTy::Scalar, SlotTy::Scalar],
            identity: smallvec![zero(dtype), zero(dtype), zero(dtype)],
            lift: smallvec![
                ScalarExpr::lit(one(dtype)),
                ScalarExpr::arg(0, dtype),
                ScalarExpr::lit(zero(dtype)),
            ],
            merge: smallvec![n, mean, m2],
            associative: true,
            tie: None,
        }
    }

    fn splat(d: Dtype, v: f32) -> Splat {
        match d {
            Dtype::F16 => Splat::F16(half::f16::from_f32(v).to_bits()),
            Dtype::BF16 => Splat::BF16(half::bf16::from_f32(v).to_bits()),
            Dtype::U32 => Splat::U32(v.max(0.0) as u32),
            Dtype::I32 => Splat::I32(v as i32),
            _ => Splat::F32(v),
        }
    }
    pub fn zero(d: Dtype) -> Splat {
        splat(d, 0.0)
    }
    pub fn one(d: Dtype) -> Splat {
        splat(d, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(c: &Carrier, xs: &[f32]) -> Vec<f32> {
        xs.iter().fold(c.identity_f32(), |acc, &x| {
            c.absorb(&acc, &[x]).expect("evaluator covers this carrier")
        })
    }

    fn merge(c: &Carrier, a: &[f32], b: &[f32]) -> Vec<f32> {
        c.eval_merge(a, b).expect("evaluator covers this carrier")
    }

    const XS: [f32; 8] = [1.5, -3.0, 7.25, 0.0, 7.25, -11.5, 2.0, 4.75];

    fn sum() -> Carrier {
        Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32)
    }
    fn max() -> Carrier {
        Carrier::binop(BinOp::Max, Splat::F32(f32::NEG_INFINITY), Dtype::F32)
    }

    #[test]
    fn a_binop_carrier_is_a_plain_reduction() {
        assert!((run(&sum(), &XS)[0] - XS.iter().sum::<f32>()).abs() < 1e-5);
        assert_eq!(run(&max(), &XS)[0], 7.25);
    }

    /// The one thing Kernel's hardware fast path reads. A fused `lift` must not
    /// change the answer, or every existing single-slot fold loses its
    /// collective.
    #[test]
    fn binop_recognition_survives_a_fused_lift() {
        assert_eq!(sum().kind(), Some(BinOp::Add));
        let fused = sum().with_lift([ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::arg(1, Dtype::F32),
        )]);
        assert_eq!(fused.kind(), Some(BinOp::Add));
        assert_eq!(oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32).kind(), None);
        assert_eq!(oracle::welford(Dtype::F32).kind(), None);
    }

    /// `merge(a, b)` written the other way round is still the same reduction —
    /// otherwise a canonicalizing builder would silently lose the fast path.
    #[test]
    fn binop_recognition_is_modulo_commutation() {
        let a = ScalarExpr::arg(0, Dtype::F32);
        let b = ScalarExpr::arg(1, Dtype::F32);
        let swapped = Carrier {
            merge: smallvec![ScalarExpr::bin(BinOp::Add, b.clone(), a.clone())],
            ..sum()
        };
        assert_eq!(swapped.kind(), Some(BinOp::Add));
        let sub = Carrier {
            merge: smallvec![ScalarExpr::bin(BinOp::Sub, b, a)],
            ..sum()
        };
        assert_eq!(sub.kind(), None, "Sub does not commute");
    }

    #[test]
    fn shift_stabilized_sum_equals_the_two_pass_result() {
        let c = oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
        let got = run(&c, &XS);
        let want_max = XS.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let want_sum: f32 = XS.iter().map(|x| (x - want_max).exp()).sum();
        assert!((got[0] - want_max).abs() < 1e-6, "max: {got:?}");
        assert!((got[1] - want_sum).abs() < 1e-5, "sum: {got:?} want {want_sum}");
    }

    #[test]
    fn shift_stabilized_sum_survives_what_a_naive_sum_does_not() {
        let big = [900.0f32, 901.0, 899.5];
        let naive: f32 = big.iter().map(|x| x.exp()).sum();
        assert!(naive.is_infinite(), "precondition: naive exp overflows");
        let got = run(&oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32), &big);
        assert!(got[1].is_finite() && got[1] > 1.0, "{got:?}");
    }

    /// Merging two identity lanes must give the identity. The unguarded
    /// rescale gives `0 * exp((-inf) - (-inf)) = NaN`, and every
    /// workgroup-tree schedule merges padded lanes.
    #[test]
    fn every_carrier_is_identity_closed() {
        let carriers = [
            sum(),
            max(),
            Carrier::binop(BinOp::Mul, Splat::F32(1.0), Dtype::F32),
            Carrier::binop(BinOp::Min, Splat::F32(f32::INFINITY), Dtype::F32),
            oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32),
            oracle::welford(Dtype::F32),
        ];
        for c in &carriers {
            assert!(c.identity_closed(&PROBES), "not identity-closed: {c:?}");
            let i = c.identity_f32();
            assert!(
                merge(c, &i, &i).iter().all(|v| !v.is_nan()),
                "merge(identity, identity) produced a NaN"
            );
        }
        for row in RETARGET_TABLE {
            let stat = (row.stat)(Dtype::F32);
            assert!(stat.identity_closed(&PROBES), "{}: stat not closed", row.name);
        }
    }

    /// Negative control: the unguarded spelling is the bug the `safe_delta`
    /// guard exists for, and it is a NaN, not a rounding wobble.
    #[test]
    fn an_unguarded_rescale_is_not_identity_closed() {
        let d = Dtype::F32;
        let (m_a, l_a) = (ScalarExpr::arg(0, d), ScalarExpr::arg(1, d));
        let (m_b, l_b) = (ScalarExpr::arg(2, d), ScalarExpr::arg(3, d));
        let m = ScalarExpr::bin(BinOp::Max, m_a.clone(), m_b.clone());
        let raw = |ms: ScalarExpr, ls: ScalarExpr| {
            ScalarExpr::bin(
                BinOp::Mul,
                ls,
                ScalarExpr::un(UnOp::Exp, ScalarExpr::bin(BinOp::Sub, ms, m.clone())),
            )
        };
        let bad = Carrier {
            slots: smallvec![SlotTy::Scalar, SlotTy::Scalar],
            identity: smallvec![Splat::F32(f32::NEG_INFINITY), Splat::F32(0.0)],
            lift: smallvec![ScalarExpr::arg(0, d), ScalarExpr::lit(Splat::F32(1.0))],
            merge: smallvec![
                m.clone(),
                ScalarExpr::bin(BinOp::Add, raw(m_a, l_a), raw(m_b, l_b))
            ],
            associative: true,
            tie: None,
        };
        let i = bad.identity_f32();
        assert!(merge(&bad, &i, &i)[1].is_nan(), "precondition: unguarded is NaN");
        assert!(!bad.identity_closed(&PROBES));
    }

    #[test]
    fn the_interesting_carriers_are_associative() {
        for c in [
            oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32),
            oracle::welford(Dtype::F32),
        ] {
            let (l, m, r) = (&XS[..3], &XS[3..5], &XS[5..]);
            let (a, b, d) = (run(&c, l), run(&c, m), run(&c, r));
            let left = merge(&c, &merge(&c, &a, &b), &d);
            let right = merge(&c, &a, &merge(&c, &b, &d));
            assert!(close(&left, &right), "{left:?} vs {right:?}");
            // Splitting anywhere equals one sequential pass — the property
            // `fold_split` actually depends on.
            assert!(close(&left, &run(&c, &XS)), "{left:?} vs whole");
        }
    }

    /// Every split point, including the empty prefix and the empty suffix,
    /// where a padded lane group lands.
    #[test]
    fn any_split_equals_one_sequential_pass() {
        for c in [
            sum(),
            max(),
            oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32),
            oracle::welford(Dtype::F32),
        ] {
            let whole = run(&c, &XS);
            for cut in 0..=XS.len() {
                let joined = merge(&c, &run(&c, &XS[..cut]), &run(&c, &XS[cut..]));
                assert!(close(&joined, &whole), "cut {cut}: {joined:?} vs {whole:?}");
            }
        }
    }

    #[test]
    fn welford_matches_a_two_pass_variance() {
        let got = run(&oracle::welford(Dtype::F32), &XS);
        let n = XS.len() as f32;
        let mean = XS.iter().sum::<f32>() / n;
        let m2: f32 = XS.iter().map(|x| (x - mean) * (x - mean)).sum();
        assert!((got[0] - n).abs() < 1e-6, "count: {got:?}");
        assert!((got[1] - mean).abs() < 1e-5, "mean: {got:?}");
        assert!((got[2] - m2).abs() < 1e-3, "m2: {got:?}");
    }

    #[test]
    fn tupling_two_carriers_runs_both_in_one_pass() {
        let both = sum().tuple(&max(), &ArgRemap::identity(1));
        assert_eq!(both.carrier.width(), 2);
        assert_eq!(&both.lhs[..], &[0]);
        assert_eq!(&both.rhs[..], &[1]);
        let got = run(&both.carrier, &XS);
        assert!((got[0] - run(&sum(), &XS)[0]).abs() < 1e-5);
        assert_eq!(got[1], run(&max(), &XS)[0]);
    }

    #[test]
    fn tupling_a_multi_slot_carrier_renumbers_correctly() {
        let stabilized = oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
        let both = stabilized.tuple(&sum(), &ArgRemap::identity(1));
        assert_eq!(both.carrier.width(), 3);
        let got = run(&both.carrier, &XS);
        let want_pair = run(&stabilized, &XS);
        assert!((got[0] - want_pair[0]).abs() < 1e-6, "{got:?}");
        assert!((got[1] - want_pair[1]).abs() < 1e-5, "{got:?}");
        assert!((got[2] - XS.iter().sum::<f32>()).abs() < 1e-5, "{got:?}");
    }

    /// Joining two carriers that share a slot must give three slots, not
    /// four. If dedup is wrong the flash carrier gets two maxes that drift
    /// apart under the rescale.
    #[test]
    fn tupling_deduplicates_a_shared_slot() {
        let ml = oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
        let joined = ml.tuple(&ml, &ArgRemap::identity(1));
        assert_eq!(joined.carrier.width(), 3, "the two maxes must collapse");
        assert_eq!(&joined.rhs[..], &[0, 2]);

        let got = run(&joined.carrier, &XS);
        let want = run(&ml, &XS);
        assert!((got[0] - want[0]).abs() < 1e-6, "{got:?}");
        assert!((got[1] - want[1]).abs() < 1e-5, "{got:?}");
        assert!((got[2] - want[1]).abs() < 1e-5, "second sum drifted: {got:?}");
        assert!(joined.carrier.identity_closed(&PROBES));
    }

    #[test]
    fn dedup_leaves_distinct_slots_alone() {
        let joined = sum().tuple(&max(), &ArgRemap::identity(1));
        assert_eq!(joined.carrier.width(), 2);
        let joined = sum().tuple(&sum(), &ArgRemap::identity(1));
        assert_eq!(joined.carrier.width(), 1, "two identical sums are one sum");
    }

    /// `as_merge` reads partial accumulators. Feeding it the same partials a
    /// blocked fold produces must equal one sequential pass — the test that
    /// catches an outer level applying `lift` to a partial max.
    #[test]
    fn as_merge_joins_partials_of_a_multi_slot_carrier() {
        let c = oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
        let outer = c.as_merge();
        assert_eq!(outer.merge, c.merge);
        let blocks: Vec<Vec<f32>> = XS.chunks(3).map(|b| run(&c, b)).collect();
        let joined = blocks.iter().fold(outer.identity_f32(), |acc, b| {
            outer.absorb(&acc, b).unwrap()
        });
        assert!(close(&joined, &run(&c, &XS)), "{joined:?}");

        // The wrong spelling — reusing the inner carrier at the outer level —
        // must actually disagree, or this test proves nothing.
        let wrong = c.absorb(&c.identity_f32(), &blocks[0]).unwrap();
        assert!(!close(&wrong, &blocks[0]));
    }

    #[test]
    fn promote_widens_every_slot_and_coalesces() {
        let c = oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
        let p = c.promote(Dim::Const(64)).unwrap();
        assert_eq!(&p.slots[..], &[SlotTy::Vector(Dim::Const(64)); 2]);
        assert_eq!(p.lanes(), Some(128));
        assert_eq!(p.slot_offset(1), Some(64));
        // Two firings are TM x TN.
        let q = p.promote(Dim::Const(4)).unwrap();
        assert_eq!(&q.slots[..], &[SlotTy::Vector(Dim::Const(256)); 2]);
        // Merge and identity ride through untouched: promotion changes the
        // footprint, never the algebra.
        assert_eq!(q.merge, c.merge);
        assert_eq!(q.identity, c.identity);
        assert!(c.promote(Dim::Sym(crate::shape::SymId(0))).is_none());
    }

    #[test]
    fn out_dim_matches_the_shipped_shape_convention() {
        assert_eq!(sum().out_dim(), Some(None));
        assert_eq!(
            oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32).out_dim(),
            Some(Some(Dim::Const(2)))
        );
        assert_eq!(oracle::welford(Dtype::F32).out_dim(), Some(Some(Dim::Const(3))));
        assert_eq!(
            sum().promote(Dim::Const(64)).unwrap().out_dim(),
            Some(Some(Dim::Const(64)))
        );
    }

    /// Retargeting a plain `Add` body against a running max at the
    /// `shift-exp` row reproduces the online-softmax carrier — term for term
    /// against the oracle, including the `safe_delta` guard.
    #[test]
    fn retarget_derives_the_shift_stabilized_carrier() {
        let d = Dtype::F32;
        let row = &RETARGET_TABLE[0];
        assert_eq!(row.name, "shift-exp");
        // The body enters at `rho := u`, so `h(u - u) . w = w = 1`.
        let body = Carrier::binop(BinOp::Add, Splat::F32(0.0), d)
            .with_lift([ScalarExpr::lit(Splat::F32(1.0))]);
        let got = Carrier::retarget(&(row.stat)(d), row, &body, 0).unwrap();
        let want = oracle::shift_stabilized_sum(UnOp::Exp, d);
        assert_eq!(got.slots, want.slots);
        assert_eq!(got.identity, want.identity);
        assert_eq!(got.lift, want.lift);
        assert_eq!(got.merge, want.merge, "derived merge differs from the oracle");
    }

    /// The same law over `(R, max)` instead of `(R, +)`: `max_i(x_i + w_i)`
    /// computed against a running max.
    #[test]
    fn retarget_at_the_max_plus_row_computes_a_tropical_reduction() {
        let d = Dtype::F32;
        let row = RETARGET_TABLE.iter().find(|r| r.name == "max-plus").unwrap();
        let body = Carrier::binop(BinOp::Max, Splat::F32(f32::NEG_INFINITY), d)
            .with_lift([ScalarExpr::lit(Splat::F32(0.0))]);
        let c = Carrier::retarget(&(row.stat)(d), row, &body, 0).unwrap();
        assert!(c.identity_closed(&PROBES), "{c:?}");
        let got = run(&c, &XS);
        let want = XS.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!((got[0] - want).abs() < 1e-6, "{got:?}");
        // Slot 1 carries `max_i(x_i) - rho`, which is 0 at the running max.
        assert!((got[1] - 0.0).abs() < 1e-6, "{got:?}");
    }

    #[test]
    fn hom_table_omits_the_unsound_rows() {
        assert!(
            !HOM_TABLE
                .iter()
                .any(|r| matches!(r.h, HomShape::TotalMonotone(UnOp::Log) | HomShape::TotalAntitone(UnOp::Log))),
            "log(prod x) = sum(log x) is unsound over sign"
        );
        for r in HOM_TABLE {
            if let HomShape::TotalMonotone(u) | HomShape::TotalAntitone(u) = r.h {
                assert!(
                    is_total_on(u, Dtype::F32),
                    "{u:?} is partial on f32; a monotone row over it can mint a NaN"
                );
            }
        }
        assert!(!is_total_on(UnOp::Sqrt, Dtype::F32));
        assert!(!is_total_on(UnOp::Log, Dtype::F32));
    }

    #[test]
    fn integer_carriers_are_closed_at_their_own_probes() {
        for d in [Dtype::U32, Dtype::I32] {
            for op in [BinOp::Add, BinOp::Mul, BinOp::Max, BinOp::Min] {
                let c = Carrier::binop(op, Carrier::binop_identity(op, d).unwrap(), d);
                assert!(c.identity_closed(probes_for(d)), "{op:?} in {d:?}");
            }
        }
        assert!(Carrier::binop_identity(BinOp::Add, Dtype::Q(crate::dtype::QFmt::Q4K)).is_none());
    }

    /// A single scalar slot expands to itself: `Arg(0)`/`Arg(1)` are already
    /// lane 0 of each side.
    #[test]
    fn a_scalar_carrier_expands_to_its_own_merge() {
        let c = sum();
        let lanes = c.merge_lanes().unwrap();
        assert_eq!(lanes.len(), 1);
        assert_eq!(lanes[0], c.merge[0]);
        assert_eq!(c.lane_slots().unwrap(), vec![(0, 0)]);
        assert_eq!(c.identity_lanes().unwrap().len(), 1);
    }

    /// Two scalar slots: the right accumulator's `Arg(w + j)` becomes
    /// `Arg(lanes + j)`, which at `w == lanes` is the identity renumbering.
    #[test]
    fn two_scalar_slots_renumber_to_two_lanes() {
        let c = oracle::shift_stabilized_sum(UnOp::Exp, Dtype::F32);
        let lanes = c.merge_lanes().unwrap();
        assert_eq!(lanes.len(), 2);
        assert_eq!(lanes, c.merge.to_vec());
        assert_eq!(c.lane_slots().unwrap(), vec![(0, 0), (1, 0)]);
    }

    /// A `Vector(d)` slot is `d` lanes whose merge reads its own position,
    /// and a `Scalar` sibling read resolves to that sibling's single lane.
    #[test]
    fn a_vector_slot_expands_positionwise_and_reads_a_scalar_sibling_at_lane_zero() {
        let d = Dtype::F32;
        // slots: [Scalar m, Vector(3) o]; o's merge reads m (Arg(0)/Arg(2)).
        let c = Carrier {
            slots: smallvec![SlotTy::Scalar, SlotTy::Vector(Dim::Const(3))],
            identity: smallvec![Splat::F32(0.0), Splat::F32(0.0)],
            lift: smallvec![ScalarExpr::arg(0, d), ScalarExpr::arg(0, d)],
            merge: smallvec![
                ScalarExpr::bin(BinOp::Add, ScalarExpr::arg(0, d), ScalarExpr::arg(2, d)),
                ScalarExpr::bin(
                    BinOp::Add,
                    ScalarExpr::bin(BinOp::Add, ScalarExpr::arg(1, d), ScalarExpr::arg(3, d)),
                    ScalarExpr::arg(0, d),
                ),
            ],
            associative: true,
            tie: None,
        };
        assert_eq!(c.lanes(), Some(4));
        assert_eq!(
            c.lane_slots().unwrap(),
            vec![(0, 0), (1, 0), (1, 1), (1, 2)]
        );
        let lanes = c.merge_lanes().unwrap();
        assert_eq!(lanes.len(), 4);
        // Lane 0 is m: left lane 0 + right lane 4.
        assert_eq!(
            lanes[0],
            ScalarExpr::bin(BinOp::Add, ScalarExpr::arg(0, d), ScalarExpr::arg(4, d))
        );
        // Lane 1+p is o at position p: left lane 1+p, right lane 5+p, plus the
        // scalar m at lane 0 — the same expression at every position.
        for p in 0..3u32 {
            assert_eq!(
                lanes[1 + p as usize],
                ScalarExpr::bin(
                    BinOp::Add,
                    ScalarExpr::bin(
                        BinOp::Add,
                        ScalarExpr::arg(1 + p, d),
                        ScalarExpr::arg(5 + p, d)
                    ),
                    ScalarExpr::arg(0, d),
                )
            );
        }
        // Identities follow the slot, one per lane.
        assert_eq!(c.identity_lanes().unwrap().len(), 4);
    }

    /// Two `Vector` slots that read each other must agree in extent: clamping a
    /// position would silently merge the wrong element.
    #[test]
    fn a_cross_slot_read_between_unequal_vector_slots_declines() {
        let d = Dtype::F32;
        let c = Carrier {
            slots: smallvec![SlotTy::Vector(Dim::Const(2)), SlotTy::Vector(Dim::Const(3))],
            identity: smallvec![Splat::F32(0.0), Splat::F32(0.0)],
            lift: smallvec![ScalarExpr::arg(0, d), ScalarExpr::arg(0, d)],
            merge: smallvec![
                ScalarExpr::bin(BinOp::Add, ScalarExpr::arg(0, d), ScalarExpr::arg(2, d)),
                ScalarExpr::bin(BinOp::Add, ScalarExpr::arg(1, d), ScalarExpr::arg(0, d)),
            ],
            associative: true,
            tie: None,
        };
        assert_eq!(c.merge_lanes(), None);
        // A symbolic extent declines for the same reason `lanes()` does.
        let sym = Carrier {
            slots: smallvec![SlotTy::Vector(Dim::Sym(crate::shape::SymId(0)))],
            ..sum()
        };
        assert_eq!(sym.merge_lanes(), None);
    }

    /// The floor lowering retypes a lift's `Arg` leaves without touching its
    /// shape. Replacing the lift instead — a per-slot `Arg(0)` — is right only
    /// for a binop carrier and folds Welford's constants over the data.
    #[test]
    fn retyping_a_lift_preserves_its_shape() {
        let c = oracle::welford(Dtype::F32);
        let retyped: Vec<ScalarExpr> = c
            .lift
            .iter()
            .map(|e| retype_args(e, Dtype::F16))
            .collect();
        assert_eq!(retyped.len(), 3);
        assert!(matches!(retyped[0].kind(), ScalarKind::Lit(_)));
        assert!(matches!(retyped[1].kind(), ScalarKind::Arg(0)));
        assert_eq!(retyped[1].dtype(), Dtype::F16);
        assert!(matches!(retyped[2].kind(), ScalarKind::Lit(_)));
    }
}
