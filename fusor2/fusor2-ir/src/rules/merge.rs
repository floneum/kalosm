//! Horizontal merging. [`crate::ir::level1::KMerged`]'s constructor already
//! refuses a segment carrying an epilogue, so the illegal state is
//! unrepresentable; the un-merged epilogue-carrying contraction stays a live
//! alternative in the same chain, and merging and epilogue fusion compete on
//! price instead of one vetoing the other.
//!
//! The reference's `fusion_toposort`, its wave-generation dependency tracker
//! and `merge_profile`'s `None`-on-epilogue veto are all deleted.
//!
//! Owned by W2.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::level1::{
    Family, KMerged, L1, MapDomain, MergeKey, MergeSegment, ScheduleDomain, WaveCat,
};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::is_ident;
use crate::shape::Dim;
use smallvec::SmallVec;

rule!(
    MERGE_CONTRACT_WAVE,
    level = Level::L1,
    head = OpTag::KContract,
    tag = RuleTag::Additive,
    apply = merge_contract_wave,
);

rule!(
    MERGE_ROW_WAVE,
    level = Level::L1,
    head = OpTag::KFold,
    tag = RuleTag::Additive,
    apply = merge_row_wave,
);

rule!(
    MERGE_REGION_WAVE,
    level = Level::L1,
    head = OpTag::KRegion,
    tag = RuleTag::Additive,
    apply = merge_region_wave,
);

/// The wave key and epilogue flag of one candidate node, or `None` when the
/// node is not a mergeable shape at all.
fn segment_of(b: &Builder<'_>, id: Id) -> Option<MergeSegment> {
    match &b.node(id).op {
        Op::L1(L1::KContract {
            m,
            n,
            k,
            batch,
            family,
            post,
            acc,
            ..
        }) => Some(MergeSegment {
            id,
            key: MergeKey {
                m: *m,
                n: *n,
                k: *k,
                batch: *batch,
                splits: 1,
                dtype: *acc,
                family: *family,
            },
            has_epilogue: !is_ident(post),
        }),
        Op::L1(L1::KFold {
            space,
            axis,
            acc,
            post,
            carrier,
            ..
        }) => {
            // A merged wave writes one value per row; a multi-slot carrier
            // writes a lane group, which no `MergeKey` describes.
            if carrier.width() != 1 {
                return None;
            }
            let k = *space.dims.get(*axis as usize)?;
            let rows = space
                .dims
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != *axis as usize)
                .try_fold(1u64, |acc, (_, d)| acc.checked_mul(d.as_const()?))
                .map_or(Dim::ONE, Dim::Const);
            Some(MergeSegment {
                id,
                key: MergeKey {
                    m: rows,
                    n: Dim::ONE,
                    k,
                    batch: Dim::ONE,
                    splits: 1,
                    dtype: *acc,
                    family: Family::GenericFold,
                },
                has_epilogue: post.iter().any(|p| !is_ident(p)),
            })
        }
        _ => None,
    }
}

/// Union a wave over `segs` into `id`'s class.
///
/// **A wave of one is refused.** `KMerged { segments: [x] }` names `x` as a
/// child, so unioning it into `x`'s own class produces a member whose operand
/// resolves back to the class it is a member of. `realize::walk` reports any
/// selection that reaches one as "selection is cyclic through %N", and the
/// class's `lower_bound` becomes a self-referential recurrence that burns the
/// whole Kleene iteration. It fired on **every** `KContract` and **every**
/// `KFold`, so every graph holding a reduction carried one.
///
/// Nothing is lost: a one-segment wave denotes exactly the segment, and a
/// genuine multi-segment wave still comes from [`merge_region_wave`], which
/// reads sibling members off a `KRegion` and never names its own class.
///
/// The wave is minted **with its schedule domain**, derived from the segments'
/// shared index space and the device. Every segment shares the `MergeKey`, so
/// the first segment's own value is that space; `verify_l1` recomputes the
/// same domain from the merged node's inferred facts, which are that same
/// value, so the two cannot drift.
fn mint(b: &mut Builder<'_>, id: Id, cat: WaveCat, segs: Vec<MergeSegment>) -> Option<Id> {
    if segs.len() < 2 {
        return None;
    }
    let sched = linear_domain_of(b, segs.first()?.id);
    let merged = KMerged::new(cat, segs, sched).ok()?;
    let wave = b.add_l1(L1::KMerged(merged)).ok()?;
    b.union(id, wave).ok()
}

/// The linear schedule domain of a composite whose value is `landed`'s.
pub fn linear_domain_of(b: &Builder<'_>, landed: Id) -> ScheduleDomain {
    ScheduleDomain::Map(MapDomain::linear_over(
        b.caps(),
        &b.facts_of(landed).shape,
    ))
}

/// The merged-wave spelling of a dense contraction.
///
/// Reader-rooted like every other rule here: a node can only name itself and
/// its children. A lone contraction therefore has no sibling to merge with
/// and [`mint`] refuses the wave of one; the multi-segment case is
/// [`merge_region_wave`]. A segment carrying an epilogue is refused by the
/// constructor, not by this rule.
pub fn merge_contract_wave(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    if !matches!(node.op, Op::L1(L1::KContract { .. })) {
        return None;
    }
    let seg = segment_of(b, id)?;
    let cat = if seg.key.splits > 1 {
        WaveCat::MatmulSplitK
    } else {
        WaveCat::Matmul
    };
    mint(b, id, cat, vec![seg])
}

/// The row-program counterpart: same wave discipline over a `KFold`.
pub fn merge_row_wave(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    if !matches!(node.op, Op::L1(L1::KFold { .. })) {
        return None;
    }
    let seg = segment_of(b, id)?;
    mint(b, id, WaveCat::Row, vec![seg])
}

/// A region already names several sibling members, so this is the one place
/// a genuine multi-segment wave is expressible without a graph-global rule
/// form: every member that is a mergeable shape and shares one key becomes a
/// segment.
pub fn merge_region_wave(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::L1(L1::KRegion { members, .. }) = &node.op else {
        return None;
    };
    let members: SmallVec<[Id; 8]> = members.clone();
    let segs: Vec<MergeSegment> = members.iter().filter_map(|&m| segment_of(b, m)).collect();
    if segs.len() < 2 {
        return None;
    }
    mint(b, id, WaveCat::Region, segs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::rules::test_support as ts;
    use crate::rules::{alias_operand_of, ident_expr};
    use crate::scalar::{ScalarExpr, UnOp};

    fn fire(g: &mut crate::egraph::EGraph, id: Id, r: &crate::egraph::Rule) -> Option<Id> {
        let caps = ts::caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (r.apply)(&mut b, id, &node, &facts)
    }

    /// A region over `members`, carrying the domain `verify_l1` will
    /// recompute from the node's own facts.
    fn region(g: &mut crate::egraph::EGraph, members: &[Id], live_out: u32) -> Id {
        let sched = ScheduleDomain::Map(MapDomain::linear_over(
            &ts::caps(),
            &g.facts(members[live_out as usize]).shape,
        ));
        g.add(Op::L1(L1::KRegion {
            members: members.iter().copied().collect(),
            live_outs: smallvec::smallvec![live_out],
            sched,
        }))
        .unwrap()
    }

    fn contraction(g: &mut crate::egraph::EGraph, post: ScalarExpr) -> Id {
        let a = ts::buffer(g, Dtype::F32, &[Dim::Const(4), Dim::Const(8)]);
        let bb = ts::buffer(g, Dtype::F32, &[Dim::Const(8), Dim::Const(2)]);
        ts::kcontract(
            g,
            Dim::Const(4),
            Dim::Const(2),
            Dim::Const(8),
            post,
            alias_operand_of(a, &[Dim::Const(4), Dim::Const(8)]),
            alias_operand_of(bb, &[Dim::Const(8), Dim::Const(2)]),
        )
    }

    /// Test 7. A merged body with per-segment epilogue identities is
    /// unbuildable, and the epilogue-carrying contraction stays in the chain.
    #[test]
    fn kmerged_rejects_epilogue_segments() {
        let key = MergeKey {
            m: Dim::Const(4),
            n: Dim::Const(2),
            k: Dim::Const(8),
            batch: Dim::ONE,
            splits: 1,
            dtype: Dtype::F32,
            family: Family::Sgemm,
        };
        let err = KMerged::new(
            WaveCat::Matmul,
            [MergeSegment {
                id: Id(0),
                key,
                has_epilogue: true,
            }],
            ScheduleDomain::Map(MapDomain::linear(&ts::caps(), 8)),
        )
        .unwrap_err();
        assert!(matches!(err, crate::error::Error::Legality(_)));

        let mut g = ts::graph();
        let with_epi = contraction(
            &mut g,
            ScalarExpr::un(UnOp::Tanh, ScalarExpr::arg(0, Dtype::F32)),
        );
        assert!(fire(&mut g, with_epi, &MERGE_CONTRACT_WAVE).is_none());
        let chain = g.chain(with_epi);
        assert_eq!(chain, vec![with_epi]);
    }

    /// CHANGED ASSERTION — this used to assert the rule mints a
    /// `KMerged { segments: [plain] }` and unions it into `plain`'s class.
    /// That is exactly the self-referential member: the wave's only child is
    /// `plain`, and after the union `class_of(plain)` is the wave's own
    /// class, so `realize::walk` reports any selection reaching it as
    /// "selection is cyclic through %N". It fired on every `KContract` and
    /// every `KFold` in every graph. A wave of one denotes exactly its
    /// segment, so refusing it loses no alternative.
    #[test]
    fn a_lone_contraction_gets_no_wave_of_one() {
        let mut g = ts::graph();
        let plain = contraction(&mut g, ident_expr(Dtype::F32));
        assert!(fire(&mut g, plain, &MERGE_CONTRACT_WAVE).is_none());
        assert_eq!(g.chain(plain), vec![plain]);
    }

    /// The invariant the refusal exists for: no class member may name its
    /// own class as an operand.
    #[test]
    fn no_class_member_names_its_own_class() {
        let mut g = ts::graph();
        let c1 = contraction(&mut g, ident_expr(Dtype::F32));
        let c2 = contraction(&mut g, ident_expr(Dtype::F32));
        let region = region(&mut g, &[c1, c2], 0);
        fire(&mut g, c1, &MERGE_CONTRACT_WAVE);
        fire(&mut g, c2, &MERGE_CONTRACT_WAVE);
        fire(&mut g, region, &MERGE_REGION_WAVE);
        for i in 0..g.len() {
            let id = Id(i as u32);
            // A `Union` names both its operands' classes by construction and
            // is never a selectable member; only the members matter.
            if matches!(g.node(id).op, Op::Union(..)) {
                continue;
            }
            let class = g.class_of(id);
            for child in g.node(id).children.iter() {
                assert_ne!(
                    g.class_of(*child),
                    class,
                    "{id} is a member of its own operand's class"
                );
            }
        }
    }

    /// A minted wave carries the domain its segments' shared index space
    /// implies, and it is a domain with something in it to decide. Before
    /// this, `L1::schedule()` returned `None` for both composite forms and
    /// extraction handed the lowering `SchedPoint::Point` unconditionally —
    /// the one node family the architecture calls its own fusion primitive
    /// was the one whose geometry was not a selection.
    #[test]
    fn a_minted_wave_carries_a_searchable_domain() {
        let mut g = ts::graph();
        let a = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(128), Dim::Const(64)]);
        let bb = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(64), Dim::Const(64)]);
        let big = |g: &mut crate::egraph::EGraph| {
            ts::kcontract(
                g,
                Dim::Const(128),
                Dim::Const(64),
                Dim::Const(64),
                ident_expr(Dtype::F32),
                alias_operand_of(a, &[Dim::Const(128), Dim::Const(64)]),
                alias_operand_of(bb, &[Dim::Const(64), Dim::Const(64)]),
            )
        };
        let c1 = big(&mut g);
        let c2 = big(&mut g);
        let region = region(&mut g, &[c1, c2], 0);
        assert!(fire(&mut g, region, &MERGE_REGION_WAVE).is_some());
        let alt = g.chain(region).into_iter().find(|&i| i != region).unwrap();
        let Op::L1(op @ L1::KMerged(_)) = &g.node(alt).op else {
            panic!("no wave")
        };
        let sched = op.schedule().expect("a wave declares its schedule space");
        assert_eq!(
            *sched,
            ScheduleDomain::Map(MapDomain::linear_over(&ts::caps(), &g.facts(c1).shape)),
            "the domain is derived from the segments' shared index space"
        );
        assert!(
            sched.len() > 1,
            "a schedule domain of one is a node that opted out of the search"
        );
        // And the region it was minted from declares the same space.
        assert_eq!(g.node(region).op, Op::L1(L1::KRegion {
            members: smallvec::smallvec![c1, c2],
            live_outs: smallvec::smallvec![0],
            sched: sched.clone(),
        }));
    }

    #[test]
    fn merge_region_wave_needs_two_matching_segments() {
        let mut g = ts::graph();
        let c1 = contraction(&mut g, ident_expr(Dtype::F32));
        let c2 = contraction(&mut g, ident_expr(Dtype::F32));
        let region = region(&mut g, &[c1, c2], 0);
        assert!(fire(&mut g, region, &MERGE_REGION_WAVE).is_some());
        let alt = g.chain(region).into_iter().find(|&i| i != region).unwrap();
        let Op::L1(L1::KMerged(w)) = &g.node(alt).op else {
            panic!()
        };
        assert_eq!(w.segments().len(), 2);
        assert_eq!(w.category(), WaveCat::Region);
    }
}
