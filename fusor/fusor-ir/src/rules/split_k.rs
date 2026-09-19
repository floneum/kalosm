//! `SPLIT_K`: a contraction over a long reduced axis as a batched
//! contraction over chunks of it — one batch element per chunk, so the grid
//! is `splits` times wider and each workgroup's k loop `splits` times
//! shorter — and a fold summing the chunks, with the epilogue moved after
//! the sum. A weight gradient is a few output tiles reducing over every
//! token: unsplit it is a handful of workgroups each walking the whole
//! axis. Both spellings stay live; cost decides.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, ContractSide, IndexSpace, Launch, Operand, ScheduleDomain};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::ident_expr;
use crate::scalar::BinOp;
use crate::shape::{Dim, Layout};
use smallvec::SmallVec;

rule!(
    SPLIT_K,
    level = Level::Launch,
    head = OpTag::LaunchContract,
    tag = RuleTag::Additive,
    apply = split_k,
);

/// Shortest reduced axis worth splitting, and the shortest chunk: a chunk
/// still has to feed a tile's k loop a few times.
const MIN_K: u64 = 256;
const MIN_CHUNK: u64 = 32;
const SPLITS: [u64; 4] = [4, 8, 16, 32];

pub fn split_k(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Contract {
        m,
        n,
        k,
        batch,
        family,
        post,
        acc,
        a,
        b: rhs,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let kc = k.as_const()?;
    if kc < MIN_K || std::env::var_os("FUSOR_NO_SPLIT_K").is_some() {
        return None;
    }
    let (mc, nc, batch_c) = (m.as_const()?, n.as_const()?, batch.as_const()?);
    let log = std::env::var_os("FUSOR_SPLIT_LOG").is_some();
    if log {
        eprintln!(
            "SPLITK {id}: m={mc} n={nc} k={kc} b={batch_c} a={:?}/{:?} {:?} b={:?}/{:?} {:?}",
            a.primary().layout.shape(),
            a.primary().layout.strides(),
            a.primary().access,
            rhs.primary().layout.shape(),
            rhs.primary().layout.strides(),
            rhs.primary().access
        );
    }
    // `a` is `[batch.., m.., k]` and `b` is `[batch.., k, n..]`, k a single
    // axis on each: the split threads a chunk axis in front of the row
    // group and a chunk stride through k.
    let a_k = k_axis(a.primary().layout.shape(), batch_c, mc, kc, false);
    let b_k = k_axis(rhs.primary().layout.shape(), batch_c, nc, kc, true);
    if log {
        eprintln!(
            "  axes a={:?} b={:?}",
            a_k.as_ref().map(|k| (k.axis, k.batch_axes)),
            b_k.as_ref().map(|k| (k.axis, k.batch_axes))
        );
    }
    let (a_k, b_k) = (a_k?, b_k?);
    // An `Alias` or a `Pack` reads the operand's own layout; a gather or an
    // unflatten carries its own map, which the chunk stride cannot thread.
    if a.ops
        .iter()
        .chain(rhs.ops.iter())
        .any(|o| !matches!(o.access, AccessPlan::Alias | AccessPlan::Pack { .. }))
    {
        return None;
    }
    let mut last = None;
    for s in SPLITS {
        if !kc.is_multiple_of(s) || kc / s < MIN_CHUNK {
            continue;
        }
        let chunk = Dim::Const(kc / s);
        let side =
            |side: &ContractSide, k_axis: usize, batch_axes: usize| -> Option<ContractSide> {
                let ops: Option<SmallVec<[Operand; 2]>> = side
                    .ops
                    .iter()
                    .map(|o| {
                        let layout = &o.layout;
                        let shape = layout.shape();
                        let strides = layout.strides();
                        let k_stride = strides[k_axis];
                        let mut new_shape: SmallVec<[Dim; 6]> = SmallVec::new();
                        let mut new_strides: SmallVec<[Dim; 6]> = SmallVec::new();
                        for (i, (d, st)) in shape.iter().zip(strides).enumerate() {
                            if i == batch_axes {
                                new_shape.push(Dim::Const(s));
                                new_strides.push(k_stride * chunk);
                            }
                            if i == k_axis {
                                new_shape.push(chunk);
                            } else {
                                new_shape.push(*d);
                            }
                            new_strides.push(*st);
                        }
                        if batch_axes == shape.len() {
                            new_shape.push(Dim::Const(s));
                            new_strides.push(k_stride * chunk);
                        }
                        let access = match &o.access {
                            AccessPlan::Pack { .. } => AccessPlan::Pack {
                                into: Layout::contiguous(&new_shape),
                            },
                            _ => AccessPlan::Alias,
                        };
                        Some(Operand {
                            src: o.src,
                            layout: Layout::from_parts(layout.offset(), &new_shape, &new_strides)
                                .ok()?,
                            access,
                        })
                    })
                    .collect();
                Some(ContractSide {
                    pre: side.pre.clone(),
                    ops: ops?,
                })
            };
        let a2 = side(a, a_k.axis, a_k.batch_axes)?;
        let b2 = side(rhs, b_k.axis, b_k.batch_axes)?;
        let partials = b
            .add_launch(Launch::Contract {
                m: *m,
                n: *n,
                k: chunk,
                batch: *batch * Dim::Const(s),
                family: *family,
                post: ident_expr(*acc),
                acc: *acc,
                a: a2,
                b: b2,
                sched: sched.clone(),
            })
            .ok()?;
        // The partials land as `[batch * s, m, n]`; the sum walks them as
        // `[batch, s, m, n]` (or `[s, m, n]` without a batch), so the fold's
        // output is the contraction's own shape.
        let mut space: SmallVec<[Dim; 6]> = SmallVec::new();
        let axis = if batch_c == 1 {
            space.push(Dim::Const(s));
            0u32
        } else {
            space.push(*batch);
            space.push(Dim::Const(s));
            1u32
        };
        space.push(*m);
        space.push(*n);
        let sum = b
            .add_launch(Launch::Fold {
                space: IndexSpace::new(space.iter().copied()),
                axis,
                vec_axes: SmallVec::new(),
                carrier: crate::carrier::Carrier::binop(
                    BinOp::Add,
                    crate::carrier::Carrier::binop_identity(BinOp::Add, *acc)?,
                    *acc,
                )
                .with_lift([ident_expr(*acc)]),
                acc: *acc,
                post: smallvec::smallvec![post.clone()],
                ops: vec![crate::rules::alias_operand_of(partials, &space)],
                sched: ScheduleDomain::Point,
            })
            .ok()?;
        last = Some(b.union(id, sum).ok()?);
    }
    last
}

struct KAxis {
    axis: usize,
    batch_axes: usize,
}

/// The position of the single k axis in a side's `[batch.., x.., k]` (a) or
/// `[batch.., k, x..]` (b) layout, and how many batch axes precede it.
fn k_axis(shape: &[Dim], batch: u64, x: u64, k: u64, k_first: bool) -> Option<KAxis> {
    let consts: Vec<u64> = shape.iter().map(|d| d.as_const()).collect::<Option<_>>()?;
    let mut i = 0;
    let mut acc = 1u64;
    while i < consts.len() && acc < batch {
        acc *= consts[i];
        i += 1;
    }
    if acc != batch {
        return None;
    }
    let batch_axes = i;
    if k_first {
        if consts.get(i) != Some(&k) {
            return None;
        }
        let rest: u64 = consts[i + 1..].iter().product();
        (rest == x).then_some(KAxis {
            axis: i,
            batch_axes,
        })
    } else {
        let last = consts.len().checked_sub(1)?;
        if consts[last] != k || last < i {
            return None;
        }
        let mid: u64 = consts[i..last].iter().product();
        (mid == x).then_some(KAxis {
            axis: last,
            batch_axes,
        })
    }
}
