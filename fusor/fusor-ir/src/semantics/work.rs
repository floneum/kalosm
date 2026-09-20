//! Shape-dependent work rows; `index_ops` prices view-fold versus gather.
//!
//! Symbolic dims price as `1`. A `Sym` extent is bound at dispatch, so a
//! shape-family plan is costed at its smallest legal binding and the specialised
//! variant — which knows the real extent — is the one that can out-price it.

use crate::contract_spec;
use crate::facts::{ValueFacts, Work};
use crate::ir::Op;
use crate::ir::kernel::ScalarElement;
use crate::ir::launch::{Family, Launch, SchedPoint, ScheduleDomain};
use crate::ir::logical::Logical;
use crate::scalar::{BinOp, ScalarExpr, ScalarKind};
use crate::shape::Dim;
use rustc_hash::FxHashSet;

/// Work one op performs at these shapes.
pub fn work_of(op: &Op, ins: &[ValueFacts], out: &ValueFacts) -> Work {
    match op {
        Op::Logical(o) => work_l0(o, ins, out),
        Op::Launch(o) => work_l1(o, ins, out),
        // A union node is a choice, not a computation.
        Op::Union(..) => Work::default(),
    }
}

/// `(arith, transcendental, index)` of evaluating every slot's lift once.
/// Shared subexpressions across slots are counted once, matching what a
/// structurally-CSE'd emitter issues. `UnOp::is_transcendental()` and
/// `BinOp::Pow` are transcendental; `IndexOf` is an index op; everything else
/// is arithmetic.
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

pub fn work_l0(op: &Logical, ins: &[ValueFacts], out: &ValueFacts) -> Work {
    let e = elements(out);
    match op {
        // The two documented constant-work exemptions: a leaf reads a buffer
        // the plan already accounts for, and a projection is a relabelling.
        Logical::Leaf(_) | Logical::Project { .. } => Work::default(),

        Logical::Map { expr, .. } => {
            let (arith, trans, index) = scalar_expr_cost(expr);
            Work {
                macs: e.saturating_mul(arith),
                transcendentals: e.saturating_mul(trans),
                index_ops: e.saturating_mul(index),
                wg_bytes: 0,
            }
        }

        // One merge per slot per element, plus the lift.
        Logical::Fold { carrier, .. } => {
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

        Logical::Contract { spec, .. } => {
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

        Logical::Restride { .. } | Logical::Window { .. } => Work {
            index_ops: e,
            ..Work::default()
        },

        Logical::Gather { .. } => Work {
            index_ops: e.saturating_mul(2),
            ..Work::default()
        },

        Logical::Scatter { .. } => Work {
            index_ops: ins.get(2).map_or(0, elements).saturating_mul(2),
            ..Work::default()
        },

        Logical::Dequant { fmt, .. } => Work {
            index_ops: e.saturating_mul(quant_decode_ops(*fmt)),
            ..Work::default()
        },
    }
}

pub fn work_l1(op: &Launch, ins: &[ValueFacts], out: &ValueFacts) -> Work {
    let e = elements(out);
    match op {
        Launch::Map { body, ops, .. } => {
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

        // A promoted axis leaves the iteration domain and reappears as
        // carrier lanes, so the per-element merge count rises with `lanes()`.
        Launch::Fold {
            carrier,
            post,
            ops,
            space,
            vec_axes,
            ..
        } => {
            let lanes = carrier.lanes().unwrap_or(carrier.width() as u64);
            // A promoted axis's extent is already counted in `lanes`, so
            // `vec_axes` must be filtered out of the iterated space or the
            // nest is charged `lanes` times its true cost. The filter is a
            // no-op on every unpromoted node.
            //
            // A wrong value in `normalization::composed_backward_saturates
            // [gpu]` that surfaces with this row correct is a defect in the
            // GPU cooperative-matrix contraction at
            // `Coop{bm:16,bn:16,bk:8,n_passes:1,subgroups:1,rg:1,cg:1}`,
            // which this row merely perturbs extraction into selecting —
            // denying `Caps::coop_supported` makes the case pass.
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
            // element — the same schedule-independent floor `Map` and
            // `Contract` price.
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

        Launch::Contract {
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
            // A side's `pre` runs once per loaded element of that side.
            // Operand index arithmetic is not priced here: a contraction's
            // traffic term dominates it, and `fusor_cost::realize` counts
            // the bytes per operand.
            w = w.add(epilogue_work(&a.pre, b.saturating_mul(m).saturating_mul(k)));
            w = w.add(epilogue_work(
                &rhs.pre,
                b.saturating_mul(k).saturating_mul(n),
            ));
            // The staged decode of a quantized operand, once per element —
            // the schedule-independent floor. The per-tile re-execution is
            // schedule knowledge and lives in `fusor_cost::realize`.
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

        Launch::Gather { ops, .. } => Work {
            index_ops: e
                .saturating_mul(2)
                .saturating_add(operand_index_ops(ops, e)),
            ..Work::default()
        },

        Launch::Scatter { ops, .. } => {
            let upd = ins.last().map_or(e, elements);
            Work {
                index_ops: upd
                    .saturating_mul(2)
                    .saturating_add(operand_index_ops(ops, upd)),
                ..Work::default()
            }
        }

        // Members carry their own work; sequencing adds none.
        Launch::Slab { .. } | Launch::Group { .. } => Work::default(),
    }
}

/// Arithmetic a backend's block-decode program spends per decoded element.
///
/// The decode is invisible to the IR on two paths — `Source::Quantized` in a
/// contraction's staging fill, and the identity `Map` a materializing
/// `Logical::Dequant` lowers to, where the format program rides in the operand
/// read — so it has to be priced from the format alone. Counts are the
/// per-element share of each format's unpack: shift/mask the quant, decode
/// the block scale (and minimum, and 6-bit group scales for the K formats),
/// one fma.
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

/// Index-op equivalents of one scalar load from cache: the load/store port
/// issues at a fraction of the ALU rate and the value is not there for the
/// next instruction. Tiled contractions stage their loads and do not pay
/// this per MAC; a map or a fold does, once per operand per iteration.
const LOAD_INDEX_OPS: u64 = 4;

/// Index-op equivalents of one integer divide or modulo: a u32 division is
/// a multi-instruction sequence, not an ALU slot.
const DIVMOD_INDEX_OPS: u64 = 8;

fn operand_index_ops(ops: &[crate::ir::launch::Operand], iterations: u64) -> u64 {
    ops.iter().fold(0u64, |acc, o| {
        let per = o
            .access
            .index_ops()
            .saturating_mul(DIVMOD_INDEX_OPS)
            .saturating_add(LOAD_INDEX_OPS);
        acc.saturating_add(iterations.saturating_mul(per))
    })
}

/// Bytes staged through workgroup memory by a cooperative geometry. This is
/// the **traffic** term (`score_fs`'s T2), not a legality footprint: the
/// allocation `verify_launch` admits against comes from the injected
/// `ArenaPlanner` and nowhere else.
fn coop_staged_bytes(sched: &ScheduleDomain, elem: ScalarElement) -> u64 {
    if !matches!(sched, ScheduleDomain::Coop(_)) {
        return 0;
    }
    let Some(SchedPoint::Coop { geom, staging, .. }) = sched.point(0) else {
        return 0;
    };
    crate::ir::launch::coop_tiles(geom, elem, staging)
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
