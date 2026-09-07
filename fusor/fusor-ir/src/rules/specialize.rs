//! R7 — `specialize_dim` substitutes a symbolic extent for the concrete one
//! its operands already carry. Priced by compile amortization: on first
//! sighting of a shape family the generic symbolic variant wins outright.
//! After a binding recurs, this variant wins where specialization pays.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::Launch;
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::shape::Dim;

rule!(
    SPECIALIZE_DIM,
    level = Level::Launch,
    head = OpTag::LaunchContract,
    tag = RuleTag::Additive,
    apply = specialize_dim,
);

/// Mint the variant in which a `Dim::Sym` on the node is replaced by the
/// `Dim::Const` an operand's own layout already proves it to be.
///
/// Legality-only substitution: the symbol and the constant denote the same
/// extent. Whether specializing pays is decided by the pricing crate, and both
/// variants stay live either way.
pub fn specialize_dim(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
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
    // Every operand of a side agrees on shape — they differ only in buffer,
    // stride and access — so the decided extent is readable off either one.
    let a_shape = a.primary().layout.shape();
    let b_shape = rhs.primary().layout.shape();
    // `a` is `[batch?, m, k]` and `b` is `[batch?, k, n]`, so each field has
    // exactly one place to read a decided extent from.
    let from_end = |shape: &[Dim], back: usize| -> Option<Dim> {
        shape.len().checked_sub(back).map(|i| shape[i])
    };
    let bound = |field: Dim, candidate: Option<Dim>| -> Option<Dim> {
        match (field, candidate) {
            (Dim::Sym(_), Some(c @ Dim::Const(_))) => Some(c),
            _ => None,
        }
    };

    let new_k = bound(*k, from_end(a_shape, 1));
    let new_m = bound(*m, from_end(a_shape, 2));
    let new_n = bound(*n, from_end(b_shape, 1));
    let new_batch = bound(*batch, from_end(a_shape, 3));
    if new_k.is_none() && new_m.is_none() && new_n.is_none() && new_batch.is_none() {
        return None;
    }

    let specialized = b
        .add_launch(Launch::Contract {
            m: new_m.unwrap_or(*m),
            n: new_n.unwrap_or(*n),
            k: new_k.unwrap_or(*k),
            batch: new_batch.unwrap_or(*batch),
            family: *family,
            post: post.clone(),
            acc: *acc,
            a: a.clone(),
            b: rhs.clone(),
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, specialized).ok()
}
