//! R7 — `specialize_dim` substitutes a symbolic extent for the concrete one
//! its operands already carry. Priced by compile amortization: on first
//! sighting of a shape family the generic symbolic variant wins outright, so
//! nothing compiles per length bucket. After a binding recurs, this variant
//! wins where specialization pays.
//!
//! The trainer's ten sequence buckets, its `tiles = slots.div_ceil(64) + 1024`
//! padding and `--bench`'s single-bucket filter all become unnecessary.
//!
//! Owned by W2.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::level1::L1;
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::shape::Dim;

rule!(
    SPECIALIZE_DIM,
    level = Level::L1,
    head = OpTag::KContract,
    tag = RuleTag::Additive,
    apply = specialize_dim,
);

/// Mint the variant in which a `Dim::Sym` on the node is replaced by the
/// `Dim::Const` an operand's own layout already proves it to be.
///
/// This is a legality-only substitution: the symbol and the constant denote
/// the same extent, because the operand layout the nest reads is where that
/// extent comes from. Whether specializing pays is compile amortization, in
/// the pricing crate, and both variants stay live either way.
pub fn specialize_dim(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::L1(L1::KContract {
        m,
        n,
        k,
        batch,
        family,
        pre_a,
        pre_b,
        post,
        acc,
        a,
        b: rhs,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let a_shape = a.layout.shape();
    let b_shape = rhs.layout.shape();
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
        .add_l1(L1::KContract {
            m: new_m.unwrap_or(*m),
            n: new_n.unwrap_or(*n),
            k: new_k.unwrap_or(*k),
            batch: new_batch.unwrap_or(*batch),
            family: *family,
            pre_a: pre_a.clone(),
            pre_b: pre_b.clone(),
            post: post.clone(),
            acc: *acc,
            a: a.clone(),
            b: rhs.clone(),
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, specialized).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::rules::test_support as ts;
    use crate::rules::{alias_operand_of, ident_expr};
    use crate::ir::level1::{Family, ScheduleDomain};

    #[test]
    fn specialize_dim_binds_a_symbolic_row_count() {
        let mut g = ts::graph();
        let rows = g.fresh_sym();
        let a_shape = [Dim::Const(4), Dim::Const(8)];
        let b_shape = [Dim::Const(8), Dim::Const(2)];
        let a = ts::buffer(&mut g, Dtype::F32, &a_shape);
        let bb = ts::buffer(&mut g, Dtype::F32, &b_shape);
        let c = g
            .add(Op::L1(L1::KContract {
                m: Dim::Sym(rows),
                n: Dim::Const(2),
                k: Dim::Const(8),
                batch: Dim::ONE,
                family: Family::Sgemm,
                pre_a: ident_expr(Dtype::F32),
                pre_b: ident_expr(Dtype::F32),
                post: ident_expr(Dtype::F32),
                acc: Dtype::F32,
                a: alias_operand_of(a, &a_shape),
                b: alias_operand_of(bb, &b_shape),
                sched: ScheduleDomain::Point,
            }))
            .unwrap();

        let caps = ts::caps();
        let node = g.node(c).clone();
        let facts = g.facts_view(c, &caps);
        let mut b = g.builder(&caps);
        assert!(specialize_dim(&mut b, c, &node, &facts).is_some());

        let members = g.chain(c);
        assert_eq!(members.len(), 2);
        let alt = members.into_iter().find(|&i| i != c).unwrap();
        let Op::L1(L1::KContract { m, .. }) = &g.node(alt).op else {
            panic!()
        };
        assert_eq!(*m, Dim::Const(4));
        // The generic variant survives, so a fresh binding still has a home.
        assert!(matches!(
            &g.node(c).op,
            Op::L1(L1::KContract { m: Dim::Sym(_), .. })
        ));
    }

    #[test]
    fn specialize_dim_declines_when_nothing_is_symbolic() {
        let mut g = ts::graph();
        let a_shape = [Dim::Const(4), Dim::Const(8)];
        let b_shape = [Dim::Const(8), Dim::Const(2)];
        let a = ts::buffer(&mut g, Dtype::F32, &a_shape);
        let bb = ts::buffer(&mut g, Dtype::F32, &b_shape);
        let c = ts::kcontract(
            &mut g,
            Dim::Const(4),
            Dim::Const(2),
            Dim::Const(8),
            ident_expr(Dtype::F32),
            alias_operand_of(a, &a_shape),
            alias_operand_of(bb, &b_shape),
        );
        let caps = ts::caps();
        let node = g.node(c).clone();
        let facts = g.facts_view(c, &caps);
        let mut b = g.builder(&caps);
        assert!(specialize_dim(&mut b, c, &node, &facts).is_none());
    }
}
