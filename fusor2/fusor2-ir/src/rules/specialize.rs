//! `specialize_dim` substitutes a symbolic extent for the concrete one its
//! operands already carry. Priced by compile amortization: on first sighting
//! of a shape family the generic symbolic variant wins outright, so nothing
//! compiles per length bucket. After a binding recurs, this variant wins
//! where specialization pays.

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
        .add_l1(L1::KContract {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::rules::test_support as ts;
    use crate::rules::{alias_operand_of, ident_expr};
    use crate::ir::level1::{ContractSide, Family, ScheduleDomain};

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
                post: ident_expr(Dtype::F32),
                acc: Dtype::F32,
                a: ContractSide::one(ident_expr(Dtype::F32), alias_operand_of(a, &a_shape)),
                b: ContractSide::one(ident_expr(Dtype::F32), alias_operand_of(bb, &b_shape)),
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
