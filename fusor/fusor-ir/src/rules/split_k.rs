//! `SPLIT_K`: a contraction over a long reduced axis as a batched
//! contraction over chunks of it — one batch element per chunk, so the grid
//! is `splits` times wider and each workgroup's k loop `splits` times
//! shorter — and a fold summing the chunks, with the epilogue moved after
//! the sum. A weight gradient is a few output tiles reducing over every
//! token: unsplit it is a handful of workgroups each walking the whole
//! axis. Both spellings stay live; cost decides.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, ContractSide, Launch, Operand, ScheduleDomain};
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
        output,
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
    // Splitting changes operand coordinates, including the reduced-axis origin.
    if kc < MIN_K || a.pre.reads_index_of() || rhs.pre.reads_index_of() {
        return None;
    }
    let (mc, nc, batch_c) = (m.as_const()?, n.as_const()?, batch.as_const()?);
    let mut batch_axes = 0;
    let mut batch_elements = 1u64;
    while batch_elements < batch_c {
        batch_elements = batch_elements.checked_mul(output.dims.get(batch_axes)?.as_const()?)?;
        batch_axes += 1;
    }
    if batch_elements != batch_c {
        return None;
    }
    // `a` is `[batch.., m.., k]` and `b` is `[batch.., k, n..]`, k a single
    // axis on each: the split threads a chunk axis in front of the row
    // group and a chunk stride through k.
    let a_k = k_axis(a.primary().layout.shape(), batch_c, mc, kc, false);
    let b_k = k_axis(rhs.primary().layout.shape(), batch_c, nc, kc, true);
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
        let mut partial_output = output.clone();
        partial_output.dims.insert(batch_axes, Dim::Const(s));
        let partials = b
            .add_launch(Launch::Contract {
                output: partial_output.clone(),
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
        let sum = b
            .add_launch(Launch::Fold {
                space: partial_output.clone(),
                axis: batch_axes as u32,
                vec_axes: SmallVec::new(),
                carrier: crate::carrier::Carrier::binop(
                    BinOp::Add,
                    crate::carrier::Carrier::binop_identity(BinOp::Add, *acc)?,
                    *acc,
                )
                .with_lift([ident_expr(*acc)]),
                acc: *acc,
                post: smallvec::smallvec![post.clone()],
                ops: vec![crate::rules::alias_operand_of(
                    partials,
                    &partial_output.dims,
                )],
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
        while consts.get(i) == Some(&1) {
            i += 1;
        }
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
