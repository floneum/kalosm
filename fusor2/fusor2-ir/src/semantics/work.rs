//! `work()` rows for every op. The verifier rejects a registration whose work
//! does not vary with shape. `index_ops` is exactly the term the view-fold-vs-gather
//! tradeoff needs.
//!
//! Symbolic dims price as `1`. A `Sym` extent is bound at dispatch, so a
//! shape-family plan is costed at its smallest legal binding and the specialised
//! variant — which knows the real extent — is the one that can out-price it.

use crate::contract_spec;
use crate::facts::{ValueFacts, Work};
use crate::ir::Op;
use crate::ir::level0::L0;
use crate::ir::level1::{Family, L1, SchedPoint, ScheduleDomain};
use crate::ir::level2::ScalarElement;
use crate::scalar::{BinOp, ScalarExpr, ScalarKind};
use crate::shape::Dim;
use rustc_hash::FxHashSet;

/// Work one op performs at these shapes.
///
/// `L1::Ext` has no row here — its cost lives in the open registry, which
/// only [`crate::CoreSemantics`] holds. This function reports the honest
/// index-op floor for it; [`work_of_with`] uses the registered row.
pub fn work_of(op: &Op, ins: &[ValueFacts], out: &ValueFacts) -> Work {
    match op {
        Op::L0(o) => work_l0(o, ins, out),
        Op::L1(o) => work_l1(o, ins, out),
        // A union node is a choice, not a computation.
        Op::Union(..) => Work::default(),
    }
}

/// [`work_of`] with the extension registry, so `L1::Ext` reports its own row.
pub fn work_of_with(
    op: &Op,
    ins: &[ValueFacts],
    out: &ValueFacts,
    registry: &crate::ir::OpDefRegistry,
) -> Work {
    if let Op::L1(L1::Ext { def, .. }) = op
        && let Some(d) = registry.get(*def)
    {
        return (d.work)(ins, out);
    }
    work_of(op, ins, out)
}

// ---------------------------------------------------------------------------
// Scalar expression cost
// ---------------------------------------------------------------------------

/// `(arith, transcendental, index)` operation counts of one evaluation of
/// `e`. The tree is hash-consed, so a shared subexpression is counted **once**
/// — which is what a structurally-CSE'd emitter actually issues.
///
/// `UnOp::is_transcendental()` and `BinOp::Pow` are transcendental;
/// `IndexOf` is an index op; everything else is arithmetic.
/// `(arith, transcendental, index)` of evaluating every slot's lift once.
/// Shared subexpressions across slots are counted once, which is what a
/// structurally-CSE'd emitter actually issues.
pub fn carrier_lift_cost(c: &crate::carrier::Carrier) -> (u64, u64, u64) {
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    let mut acc = (0u64, 0u64, 0u64);
    for e in &c.lift {
        count(e, &mut seen, &mut acc);
    }
    acc
}

pub fn scalar_expr_cost(e: &ScalarExpr) -> (u64, u64, u64) {
    let mut seen: FxHashSet<u64> = FxHashSet::default();
    let mut acc = (0u64, 0u64, 0u64);
    count(e, &mut seen, &mut acc);
    acc
}

fn count(e: &ScalarExpr, seen: &mut FxHashSet<u64>, acc: &mut (u64, u64, u64)) {
    if !seen.insert(e.structural_hash()) {
        return;
    }
    match e.kind() {
        ScalarKind::Arg(_) | ScalarKind::Lit(_) | ScalarKind::Uniform(_) => {}
        ScalarKind::IndexOf(_) => acc.2 += 1,
        ScalarKind::Un { op, x } => {
            if op.is_transcendental() {
                acc.1 += 1;
            } else {
                acc.0 += 1;
            }
            count(x, seen, acc);
        }
        ScalarKind::Bin { op, a, b } => {
            if matches!(op, BinOp::Pow) {
                acc.1 += 1;
            } else {
                acc.0 += 1;
            }
            count(a, seen, acc);
            count(b, seen, acc);
        }
        ScalarKind::Cmp { a, b, .. } | ScalarKind::Dot { a, b } => {
            acc.0 += 1;
            count(a, seen, acc);
            count(b, seen, acc);
        }
        ScalarKind::Select { c, t, f } => {
            acc.0 += 1;
            count(c, seen, acc);
            count(t, seen, acc);
            count(f, seen, acc);
        }
        ScalarKind::Cast { x, .. }
        | ScalarKind::Bitcast { x, .. }
        | ScalarKind::Round { x, .. }
        | ScalarKind::Splat { x, .. } => {
            acc.0 += 1;
            count(x, seen, acc);
        }
    }
}

// ---------------------------------------------------------------------------
// L0 rows
// ---------------------------------------------------------------------------

pub fn work_l0(op: &L0, ins: &[ValueFacts], out: &ValueFacts) -> Work {
    let e = elements(out);
    match op {
        // The two documented constant-work exemptions: a leaf reads a buffer
        // the plan already accounts for, and a projection is a relabelling.
        L0::Leaf(_) | L0::Project { .. } => Work::default(),

        L0::Map { expr, .. } => {
            let (arith, trans, index) = scalar_expr_cost(expr);
            Work {
                macs: e.saturating_mul(arith),
                transcendentals: e.saturating_mul(trans),
                index_ops: e.saturating_mul(index),
                wg_bytes: 0,
            }
        }

        // One merge per slot per element, plus the lift. `width` is what the
        // deleted `Combine::arity` reported; it is now a property of the
        // carrier's own slot list rather than of a name.
        L0::Fold { carrier, .. } => {
            let width = carrier.width() as u64;
            let ein = ins.first().map_or(0, elements);
            let (lift_a, lift_t, lift_i) = carrier_lift_cost(carrier);
            Work {
                macs: ein
                    .saturating_mul(width.saturating_add(lift_a))
                    .saturating_add(e.saturating_mul(width)),
                transcendentals: ein.saturating_mul(lift_t),
                index_ops: ein.saturating_mul(lift_i),
                wg_bytes: 0,
            }
        }

        L0::Contract { spec, .. } => {
            let macs = match (ins.first(), ins.get(1)) {
                (Some(a), Some(b)) => contract_spec::extents(spec, &a.shape, &b.shape)
                    .and_then(|ext| contract_spec::mnkb(spec, &ext))
                    .map(|[m, n, k, batch]| {
                        priced(batch)
                            .saturating_mul(priced(m))
                            .saturating_mul(priced(n))
                            .saturating_mul(priced(k))
                    })
                    .unwrap_or(0),
                _ => 0,
            };
            Work {
                macs,
                ..Work::default()
            }
        }

        L0::Restride { .. } | L0::Window { .. } => Work {
            index_ops: e,
            ..Work::default()
        },

        L0::Gather { .. } => Work {
            index_ops: e.saturating_mul(2),
            ..Work::default()
        },

        L0::Scatter { .. } => Work {
            index_ops: ins.get(2).map_or(0, elements).saturating_mul(2),
            ..Work::default()
        },

        L0::Dequant { fmt, .. } => Work {
            index_ops: e.saturating_mul(quant_decode_ops(*fmt)),
            ..Work::default()
        },
    }
}

// ---------------------------------------------------------------------------
// L1 rows
// ---------------------------------------------------------------------------

pub fn work_l1(op: &L1, ins: &[ValueFacts], out: &ValueFacts) -> Work {
    let e = elements(out);
    match op {
        L1::KMap { body, ops, .. } => {
            let (arith, trans, index) = scalar_expr_cost(body);
            let decode: u64 = ins
                .iter()
                .map(|f| decode_ops_of(f.dtype))
                .fold(0, u64::saturating_add);
            Work {
                macs: e.saturating_mul(arith),
                transcendentals: e.saturating_mul(trans),
                index_ops: e
                    .saturating_mul(index)
                    .saturating_add(operand_index_ops(ops, e))
                    .saturating_add(e.saturating_mul(decode)),
                wg_bytes: 0,
            }
        }

        // The carrier's `lift` is what the deleted `pre` was, and `width`
        // merges run per element instead of one named combine's arity. A
        // promoted axis leaves the iteration domain and reappears as carrier
        // lanes, so the per-element merge count rises with `lanes()`: the same
        // total work, priced where it actually happens.
        L1::KFold {
            carrier,
            post,
            ops,
            space,
            vec_axes,
            ..
        } => {
            let lanes = carrier.lanes().unwrap_or(carrier.width() as u64);
            // A promoted axis leaves the iteration domain and reappears as
            // carrier lanes, so its extent is already counted in `lanes`.
            // Multiplying the full `space` by `lanes` again charges a nest that
            // merges each element into its own lane exactly once at `lanes`
            // times its true cost, which is why a promoted `KFold` was never
            // selected. Filtering `vec_axes` out is the correct row and is a
            // no-op on every unpromoted node.
            //
            // A wrong value in `normalization::composed_backward_saturates
            // [gpu]` (a layer_norm adjoint reading -2.0558887 where every
            // entry must be zero) that surfaces with this row correct is
            // **not** a defect in the promoted path. Measured:
            //
            //   - CPU selects a promoted nest here (vec_axes=[0], axis=1),
            //     lowers it, and is CORRECT.
            //   - GPU never lowers a promoted nest in this case at all, yet
            //     changes answer. Dumping every lowered node with its schedule
            //     point shows an identical node set both ways; the *only*
            //     difference is that one `KContract` moves from
            //     `Sgemm{bm:16,bn:32,bk:8,tm:2,tn:2}` to
            //     `Coop{bm:16,bn:16,bk:8,n_passes:1,subgroups:1,rg:1,cg:1}`.
            //   - Denying `Caps::coop_supported` with this row corrected makes
            //     the case pass.
            //
            // So the wrong value is a defect in the GPU cooperative-matrix
            // contraction at that geometry, which correcting
            // this row merely perturbs extraction into selecting. Pricing
            // promotion wrongly to avoid it would be
            // a cost model encoding a legality decision.
            let ein = space
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| !vec_axes.contains(&(*i as u32)))
                .map(|(_, d)| priced(*d))
                .fold(1u64, |a, b| a.saturating_mul(b));
            let (lift_a, lift_t, lift_i) = carrier_lift_cost(carrier);
            let (post_a, post_t, post_i) = post
                .iter()
                .map(scalar_expr_cost)
                .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
            // The inline decode of a quantized operand, once per iterated
            // element — the same schedule-independent floor `KMap` and
            // `KContract` already price. Without it a fold reading a
            // `Dtype::Q(_)` operand decoded for free, and extraction
            // preferred the per-element-decode fold over a contraction
            // family whose lane window amortizes the decode.
            let decode: u64 = ins
                .iter()
                .map(|f| decode_ops_of(f.dtype))
                .fold(0, u64::saturating_add);
            Work {
                macs: ein
                    .saturating_mul(lanes.saturating_add(lift_a))
                    .saturating_add(e.saturating_mul(lanes.saturating_add(post_a))),
                transcendentals: ein
                    .saturating_mul(lift_t)
                    .saturating_add(e.saturating_mul(post_t)),
                index_ops: ein
                    .saturating_mul(lift_i)
                    .saturating_add(e.saturating_mul(post_i))
                    .saturating_add(operand_index_ops(ops, ein))
                    .saturating_add(ein.saturating_mul(decode)),
                wg_bytes: 0,
            }
        }

        L1::KContract {
            m,
            n,
            k,
            batch,
            family,
            a,
            b: rhs,
            post,
            acc,
            sched,
            ..
        } => {
            let (b, m, n, k) = (priced(*batch), priced(*m), priced(*n), priced(*k));
            let mut w = Work {
                macs: b.saturating_mul(m).saturating_mul(n).saturating_mul(k),
                ..Work::default()
            };
            // A side's `pre` runs once per loaded element of *that side*.
            // Operand index arithmetic is deliberately not priced here: it
            // never was, and a contraction's traffic term dominates it. What
            // multiplies with a multi-operand side is the *bytes*, which
            // `fusor2_cost::realize` counts per operand.
            w = w.add(epilogue_work(&a.pre, b.saturating_mul(m).saturating_mul(k)));
            w = w.add(epilogue_work(&rhs.pre, b.saturating_mul(k).saturating_mul(n)));
            // The staged decode of a quantized operand, once per element —
            // the schedule-independent floor. The per-tile re-execution is
            // schedule knowledge and lives in `fusor2_cost::realize`.
            let (a_elems, b_elems) = (
                b.saturating_mul(m).saturating_mul(k),
                b.saturating_mul(k).saturating_mul(n),
            );
            // Decode arithmetic is shifts and masks on the scalar ALU —
            // `index_ops` is the field priced at that rate; `macs` would run
            // it at the MMA rate and make it invisible.
            for (i, f) in ins.iter().enumerate() {
                let elems = if i < a.len() { a_elems } else { b_elems };
                w.index_ops = w
                    .index_ops
                    .saturating_add(elems.saturating_mul(decode_ops_of(f.dtype)));
            }
            w = w.add(epilogue_work(post, b.saturating_mul(m).saturating_mul(n)));
            if *family == Family::Coop {
                w.wg_bytes = coop_staged_bytes(sched, element_of(*acc));
            }
            w
        }

        L1::KGather { ops, .. } => Work {
            index_ops: e.saturating_mul(2).saturating_add(operand_index_ops(ops, e)),
            ..Work::default()
        },

        L1::KScatter { ops, .. } => {
            let upd = ins.last().map_or(e, elements);
            Work {
                index_ops: upd
                    .saturating_mul(2)
                    .saturating_add(operand_index_ops(ops, upd)),
                ..Work::default()
            }
        }

        // A region's true work is the sum of its members'. `ins` carries only
        // their *facts*, so this row is the shape-varying floor every member
        // pays to land its output; `fusor2-cost` sums the exact rows on the
        // realized DAG, where it has the member nodes ([`sum_work`]).
        L1::KRegion { .. } => ins.iter().fold(Work::default(), |acc, f| {
            acc.add(Work {
                index_ops: elements(f),
                ..Work::default()
            })
        }),

        // Honest floor without the registry; see [`work_of_with`].
        L1::Ext { ops, .. } => Work {
            index_ops: e.saturating_add(operand_index_ops(ops, e)),
            ..Work::default()
        },
    }
}

/// Arithmetic a backend's block-decode program spends per decoded element.
///
/// The decode is invisible to the IR on two paths — `Source::Quantized` in a
/// contraction's staging fill, and the identity `KMap` a materializing
/// `L0::Dequant` lowers to, where the format program rides in the operand
/// read — so it has to be priced from the format alone. Counts are the
/// per-element share of each format's unpack: shift/mask the quant, decode
/// the block scale (and minimum, and 6-bit group scales for the K formats),
/// one fma. Underpricing these was what made decode-in-the-fill and
/// decode-once indistinguishable to the extractor at every M.
pub fn quant_decode_ops(fmt: crate::dtype::QFmt) -> u64 {
    use crate::dtype::QFmt;
    match fmt {
        QFmt::Q8_0 => 4,
        QFmt::Q4_0 => 6,
        QFmt::Q5_0 => 8,
        QFmt::Q4K => 10,
        QFmt::Q5K => 12,
        QFmt::Q6K => 12,
    }
}

/// [`quant_decode_ops`] for a dtype, zero when dense.
pub fn decode_ops_of(d: crate::dtype::Dtype) -> u64 {
    match d {
        crate::dtype::Dtype::Q(fmt) => quant_decode_ops(fmt),
        _ => 0,
    }
}

pub fn epilogue_work(expr: &ScalarExpr, iterations: u64) -> Work {
    let (arith, trans, index) = scalar_expr_cost(expr);
    Work {
        macs: iterations.saturating_mul(arith),
        transcendentals: iterations.saturating_mul(trans),
        index_ops: iterations.saturating_mul(index),
        wg_bytes: 0,
    }
}

fn operand_index_ops(ops: &[crate::ir::level1::Operand], iterations: u64) -> u64 {
    ops.iter().fold(0u64, |acc, o| {
        acc.saturating_add(iterations.saturating_mul(o.access.index_ops()))
    })
}

/// Bytes staged through workgroup memory by a cooperative geometry. This is
/// the **traffic** term (`score_fs`'s T2), not a legality footprint: the
/// allocation `verify_l1` admits against comes from the injected
/// `ArenaPlanner` and nowhere else.
fn coop_staged_bytes(sched: &ScheduleDomain, elem: ScalarElement) -> u64 {
    if !matches!(sched, ScheduleDomain::Coop(_)) {
        return 0;
    }
    let Some(SchedPoint::Coop { geom, staging, .. }) = sched.point(0) else {
        return 0;
    };
    crate::verify_l1::coop_tiles(geom, elem, staging)
        .decls
        .iter()
        .map(|t| t.layout.element_count() * t.element.byte_size())
        .sum()
}

fn element_of(d: crate::dtype::Dtype) -> ScalarElement {
    match d {
        crate::dtype::Dtype::F16 => ScalarElement::F16,
        crate::dtype::Dtype::BF16 => ScalarElement::BF16,
        crate::dtype::Dtype::U32 => ScalarElement::U32,
        crate::dtype::Dtype::I32 => ScalarElement::I32,
        _ => ScalarElement::F32,
    }
}

/// Element count, symbolic dims priced as 1.
fn elements(f: &ValueFacts) -> u64 {
    f.shape
        .iter()
        .map(|d| priced(*d))
        .fold(1u64, |a, b| a.saturating_mul(b))
}

/// A symbolic dim prices as 1.
fn priced(d: Dim) -> u64 {
    d.as_const().unwrap_or(1)
}

/// True when `work` varies across two distinct shape bindings — the check
/// [`crate::ir::Semantics::verify`] applies to an `OpDef` registration.
pub fn work_is_shape_sensitive(
    work: fn(&[ValueFacts], &ValueFacts) -> Work,
    small: (&[ValueFacts], &ValueFacts),
    large: (&[ValueFacts], &ValueFacts),
) -> bool {
    work(small.0, small.1) != work(large.0, large.1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::{Dtype, Splat};
    use crate::egraph::Id;
    use crate::carrier::{ArgRemap, Carrier};
    use crate::ir::level0::{EinSpec, Label};
    use crate::ir::level1::{AccessPlan, IndexSpace, Operand};
    use crate::scalar::UnOp;
    use crate::shape::Layout;
    use smallvec::smallvec;

    fn f32s(shape: &[u64]) -> ValueFacts {
        ValueFacts::new(Dtype::F32, shape.iter().map(|&d| Dim::Const(d)))
    }
    fn operand(src: u32) -> Operand {
        Operand {
            src: Id(src),
            layout: Layout::contiguous(&[Dim::Const(1)]),
            access: AccessPlan::Alias,
        }
    }

    #[test]
    fn scalar_cost_classifies_ops() {
        let x = ScalarExpr::arg(0, Dtype::F32);
        // exp is transcendental, add is arith, IndexOf is an index op.
        let e = ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::un(UnOp::Exp, x.clone()),
            ScalarExpr::cast(Dtype::F32, ScalarExpr::index_of(0)),
        );
        let (arith, trans, index) = scalar_expr_cost(&e);
        assert_eq!(trans, 1); // exp
        assert_eq!(index, 1); // IndexOf
        assert_eq!(arith, 2); // add + cast

        // Pow counts as transcendental.
        let p = ScalarExpr::bin(BinOp::Pow, x.clone(), ScalarExpr::lit(Splat::F32(3.0)));
        assert_eq!(scalar_expr_cost(&p), (0, 1, 0));

        // Abs/Neg are arith, not transcendental.
        assert_eq!(scalar_expr_cost(&ScalarExpr::un(UnOp::Abs, x.clone())).0, 1);

        // A shared subtree is counted once.
        let shared = ScalarExpr::un(UnOp::Exp, x);
        let twice = ScalarExpr::bin(BinOp::Mul, shared.clone(), shared);
        assert_eq!(scalar_expr_cost(&twice), (1, 1, 0));
    }

    #[test]
    fn map_work_scales_with_elements() {
        let expr = ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32));
        let op = L0::Map {
            expr,
            ins: smallvec![Id(0)],
            outs: 1,
        };
        let out = f32s(&[4, 8]);
        let w = work_l0(&op, std::slice::from_ref(&out), &out);
        assert_eq!(w.transcendentals, 32);
        assert_eq!(w.macs, 0);
    }

    // ---- Test 3 ----------------------------------------------------------

    #[test]
    fn contract_macs_are_batch_m_n_k() {
        let spec = EinSpec {
            a: smallvec![Label(b'b'), Label(b'i'), Label(b'k')],
            b: smallvec![Label(b'b'), Label(b'j'), Label(b'k')],
            out: smallvec![Label(b'b'), Label(b'i'), Label(b'j')],
        };
        let op = L0::Contract {
            spec,
            acc: Dtype::F32,
            a: Id(0),
            b: Id(1),
            outs: 1,
        };
        let w = work_l0(
            &op,
            &[f32s(&[2, 3, 4]), f32s(&[2, 5, 4])],
            &f32s(&[2, 3, 5]),
        );
        assert_eq!(w.macs, 120);
    }

    #[test]
    fn fold_and_view_rows() {
        let sum = Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32);
        let fold = L0::Fold {
            carrier: sum.clone(),
            axis: 0,
            acc: Dtype::F32,
            ins: smallvec![Id(0)],
        };
        let w = work_l0(&fold, &[f32s(&[8, 4])], &f32s(&[4]));
        assert_eq!(w.macs, 32 + 4);

        // A two-slot carrier does twice the merge work, and the row says so
        // without knowing what either slot means.
        let two = L0::Fold {
            carrier: sum
                .tuple(
                    &Carrier::binop(BinOp::Max, Splat::F32(f32::NEG_INFINITY), Dtype::F32),
                    &ArgRemap::identity(1),
                )
                .carrier,
            axis: 0,
            acc: Dtype::F32,
            ins: smallvec![Id(0)],
        };
        assert_eq!(
            work_l0(&two, &[f32s(&[8, 4])], &f32s(&[4, 2])).macs,
            2 * 32 + 2 * 8
        );

        let restride = L0::Restride {
            specs: smallvec![crate::shape::StrideSpec::dim(0, Dim::Const(4))],
            bounds: crate::shape::BoundsProof::Static,
            x: Id(0),
        };
        assert_eq!(work_l0(&restride, &[f32s(&[4])], &f32s(&[4])).index_ops, 4);

        let gather = L0::Gather {
            axis: 0,
            x: Id(0),
            idx: Id(1),
        };
        assert_eq!(
            work_l0(&gather, &[f32s(&[16, 3]), f32s(&[4])], &f32s(&[4, 3])).index_ops,
            24
        );

        let dequant = L0::Dequant {
            fmt: crate::dtype::QFmt::Q4K,
            layout: crate::dtype::QLayout::Native,
            x: Id(0),
        };
        let w = work_l0(&dequant, &[f32s(&[512])], &f32s(&[512]));
        // Decode arithmetic is scalar-ALU shift/mask work: it prices as
        // `index_ops` (`quant_decode_ops` per element), never as MACs — in
        // `macs` a coop launch ran it at the MMA rate, i.e. for free.
        assert_eq!(w.macs, 0);
        assert_eq!(w.index_ops, 512 * quant_decode_ops(crate::dtype::QFmt::Q4K));
    }

    #[test]
    fn kmap_adds_operand_index_ops() {
        let ops = vec![
            Operand {
                src: Id(0),
                layout: Layout::contiguous(&[Dim::Const(4)]),
                access: AccessPlan::Gather,
            },
            operand(1),
        ];
        let op = L1::KMap {
            space: IndexSpace::new([Dim::Const(4)]),
            body: ScalarExpr::arg(0, Dtype::F32),
            ops,
            sched: ScheduleDomain::Point,
        };
        let out = f32s(&[4]);
        // Gather costs 1 index op per element; Alias costs 0.
        assert_eq!(work_l1(&op, &[out.clone(), out.clone()], &out).index_ops, 4);
    }

    #[test]
    fn union_and_leaf_are_free() {
        assert_eq!(
            work_of(&Op::Union(Id(0), Id(1)), &[], &f32s(&[4])),
            Work::default()
        );
        assert_eq!(
            work_of(
                &Op::L0(L0::Project { slot: 0, x: Id(0) }),
                &[f32s(&[4])],
                &f32s(&[4])
            ),
            Work::default()
        );
    }
}
