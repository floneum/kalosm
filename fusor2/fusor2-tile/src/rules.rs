//! The lowering rules that consult a schedule-domain generator: the four
//! order-free contraction family rules plus `tile_contract` and
//! `unfuse_coop_epilogue`, the four `Scatter` lowerings, the two gather
//! lowerings.
//!
//! All four contraction families coexist in one chain and compete on cost.
//! Padded MACs enter the issue term, so a badly padded coop tile loses to
//! sgemv on cost.

pub mod contract;
pub mod gather;
pub mod scatter;

use fusor2_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor2_ir::ir::level1::{L1, Operand, ScheduleDomain};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;

use crate::domains::{DomainCtx, default_planner, fold_domain_for, map_domain};

rule!(
    TILE_FOLD,
    level = Level::L1,
    head = OpTag::KFold,
    tag = RuleTag::Additive,
    apply = tile_fold,
);

rule!(
    TILE_GATHER,
    level = Level::L1,
    head = OpTag::KGather,
    tag = RuleTag::Additive,
    apply = tile_gather,
);

rule!(
    TILE_SCATTER,
    level = Level::L1,
    head = OpTag::KScatter,
    tag = RuleTag::Additive,
    apply = tile_scatter,
);

/// Attach the complete legal reduction domain to a `KFold` that arrived
/// carrying [`ScheduleDomain::Point`]. Without it the fold reaches extraction
/// with no schedule decision, leaving the reduction strategy and the lane-group
/// width to the emitter's default.
///
/// The domain is generated for this carrier's lane count, so a wide accumulator
/// is filtered by workgroup storage rather than crashed at `verify_plan`. An
/// empty domain means the rule does not apply.
pub fn tile_fold(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(l1) = &node.op else { return None };
    let L1::KFold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        sched: ScheduleDomain::Point,
        ..
    } = l1
    else {
        return None;
    };
    // Neither backend lowers a promoted nest whose reduced axis is not last:
    // their address arithmetic reads one output row as `vec_extent *
    // axis_extent` consecutive elements.
    if !vec_axes.is_empty() && *axis as usize + 1 != space.rank() {
        return None;
    }
    let k = *space.dims.get(*axis as usize)?;
    // A symbolic `Vector` slot extent is allocatable on neither backend; the
    // rule declines rather than guessing a footprint.
    let lanes = carrier.lanes()?;
    let dom = fold_domain_for(
        k,
        lanes,
        acc.byte_size(),
        &DomainCtx::new(f.caps(), default_planner()),
    );
    if dom.strategies.is_empty() {
        return None;
    }

    let mut rebuilt = l1.clone();
    if let L1::KFold { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Fold(dom);
    }
    let new = b.add_l1(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// The accesses of a node's operand list, as the map-domain generator reads
/// them: a per-lane gather has no vector load to widen into, so it forbids a
/// vectorized tiling.
fn accesses(ops: &[Operand]) -> Vec<fusor2_ir::ir::level1::AccessPlan> {
    ops.iter().map(|o| o.access.clone()).collect()
}

/// Attach the elementwise tiling domain to a floor-lowered `KGather`, without
/// touching `mode`. `gather::GATHER_*` mint a mode and a domain together, so
/// without this rule a plan that wants the floor's mode gets no domain at all.
///
/// There is no `TILE_MAP` beside this: a `KMap` domain minted as an additive
/// alternative spends the extractor's move budget on `RESELECT` between nodes
/// differing only in a field `RESCHEDULE` already ranges over, so it is
/// attached where the node is minted (`lower_floor.rs`).
pub fn tile_gather(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(l1) = &node.op else { return None };
    let L1::KGather {
        space,
        ops,
        sched: ScheduleDomain::Point,
        ..
    } = l1
    else {
        return None;
    };
    let dom = map_domain(
        &space.dims,
        &accesses(ops),
        &DomainCtx::new(f.caps(), default_planner()),
    );
    if dom.tilings.len() <= 1 {
        return None;
    }
    let mut rebuilt = l1.clone();
    if let L1::KGather { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Map(dom);
    }
    let new = b.add_l1(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// Attach the elementwise tiling domain to a floor-lowered `KScatter`, without
/// touching `mode`. `scatter::SCATTER_*` mint one of the four modes together
/// with a domain; the four lowerings coexist and compete on cost.
pub fn tile_scatter(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(l1) = &node.op else { return None };
    let L1::KScatter {
        space,
        ops,
        sched: ScheduleDomain::Point,
        ..
    } = l1
    else {
        return None;
    };
    let dom = map_domain(
        &space.dims,
        &accesses(ops),
        &DomainCtx::new(f.caps(), default_planner()),
    );
    if dom.tilings.len() <= 1 {
        return None;
    }
    let mut rebuilt = l1.clone();
    if let L1::KScatter { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Map(dom);
    }
    let new = b.add_l1(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// Every rule `fusor2-tile` owns, in a fixed declaration order. Order carries
/// no semantics; it exists only so a run is reproducible.
pub static TILE_RULES: &[Rule] = &[
    contract::TILE_CONTRACT,
    TILE_FOLD,
    TILE_GATHER,
    TILE_SCATTER,
    contract::LOWER_COOP,
    contract::LOWER_SGEMM,
    contract::LOWER_SGEMV,
    contract::LOWER_GENERIC,
    contract::UNFUSE_COOP_EPILOGUE,
    scatter::SCATTER_ATOMIC,
    scatter::SCATTER_SORT_SEGMENT,
    scatter::SCATTER_WG_PRIVATE_MERGE,
    scatter::SCATTER_ONE_HOT_CONTRACT,
    gather::GATHER_ROW_PER_GROUP,
    gather::GATHER_VECTORIZED,
];

/// Alias for [`TILE_RULES`].
pub static SCHED_RULES: &[Rule] = TILE_RULES;

#[cfg(test)]
pub(crate) mod testing {
    //! A graph fixture over the real [`CoreSemantics`], so rule tests exercise
    //! the real rule bodies against the pipeline's inference and verification.

    use fusor2_ir::device::Caps;
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::ir::level0::{BufferId, EinSpec, L0, LeafKind, ScatterCombine};
    use fusor2_ir::ir::level1::{ContractSide, Family, L1, Operand, ScheduleDomain};
    use fusor2_ir::ir::Op;
    use fusor2_ir::scalar::ScalarExpr;
    use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
    use fusor2_ir::shape::{Dim, Layout};
    use smallvec::SmallVec;
    use std::sync::Arc;

    /// A graph plus the caps its rules see.
    pub struct Fixture {
        pub graph: EGraph,
        pub caps: Caps,
        next_buffer: u32,
    }

    impl Fixture {
        pub fn new(caps: Caps) -> Self {
            Self {
                graph: EGraph::new(CoreSemantics::new(Arc::new(SumArenaPlanner))),
                caps,
                next_buffer: 0,
            }
        }

        fn name(&mut self) -> BufferId {
            let id = BufferId(self.next_buffer);
            self.next_buffer += 1;
            id
        }

        pub fn buffer(&mut self, dtype: Dtype, shape: &[u64]) -> Id {
            let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
            self.buffer_dims(dtype, &dims)
        }

        pub fn buffer_dims(&mut self, dtype: Dtype, shape: &[Dim]) -> Id {
            let name = self.name();
            self.graph
                .add(Op::L0(L0::Leaf(LeafKind::Buffer {
                    name,
                    dtype,
                    shape: shape.iter().copied().collect(),
                })))
                .expect("leaf")
        }

        pub fn contract(&mut self, spec: EinSpec, acc: Dtype, a: Id, b: Id) -> Id {
            self.graph
                .add(Op::L0(L0::Contract {
                    spec,
                    acc,
                    a,
                    b,
                    outs: 1,
                }))
                .expect("contract")
        }

        pub fn gather(&mut self, axis: u32, x: Id, idx: Id) -> Id {
            self.graph
                .add(Op::L0(L0::Gather { axis, x, idx }))
                .expect("gather")
        }

        pub fn scatter(
            &mut self,
            axis: u32,
            combine: ScatterCombine,
            base: Id,
            idx: Id,
            upd: Id,
        ) -> Id {
            self.graph
                .add(Op::L0(L0::Scatter {
                    axis,
                    combine,
                    base,
                    idx,
                    upd,
                    unique: true,
                }))
                .expect("scatter")
        }

        /// A `KContract` that arrived carrying `ScheduleDomain::Point`,
        /// which is exactly what `tile_contract` upgrades.
        pub fn point_contract(
            &mut self,
            family: Family,
            dtype: Dtype,
            a: Id,
            b: Id,
            extent: u64,
        ) -> Id {
            let dim = Dim::Const(extent);
            let layout = Layout::contiguous(&[dim, dim]);
            let operand = |src| Operand {
                src,
                layout: layout.clone(),
                access: fusor2_ir::ir::level1::AccessPlan::Alias,
            };
            self.graph
                .add(Op::L1(L1::KContract {
                    m: dim,
                    n: dim,
                    k: dim,
                    batch: Dim::Const(1),
                    family,
                    post: ScalarExpr::arg(0, dtype),
                    acc: dtype,
                    a: ContractSide::one(ScalarExpr::arg(0, dtype), operand(a)),
                    b: ContractSide::one(ScalarExpr::arg(0, dtype), operand(b)),
                    sched: ScheduleDomain::Point,
                }))
                .expect("point contract")
        }

        /// A `KFold` that arrived carrying [`ScheduleDomain::Point`], which
        /// is what the floor lowering mints and what `TILE_FOLD` upgrades.
        pub fn point_fold(
            &mut self,
            x: Id,
            shape: &[u64],
            axis: u32,
            carrier: fusor2_ir::carrier::Carrier,
        ) -> Id {
            self.promoted_fold(x, shape, axis, &[], carrier)
        }

        /// The same node with `vec_axes` spelled out: a promoted nest is
        /// `space = free.. ++ vec.. ++ [reduced]`.
        pub fn promoted_fold(
            &mut self,
            x: Id,
            shape: &[u64],
            axis: u32,
            vec_axes: &[u32],
            carrier: fusor2_ir::carrier::Carrier,
        ) -> Id {
            let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
            let post: SmallVec<[ScalarExpr; 4]> = (0..carrier.width())
                .map(|i| ScalarExpr::arg(i as u32, Dtype::F32))
                .collect();
            self.graph
                .add(Op::L1(L1::KFold {
                    space: fusor2_ir::ir::level1::IndexSpace::new(dims.iter().copied()),
                    axis,
                    vec_axes: vec_axes.iter().copied().collect(),
                    carrier,
                    acc: Dtype::F32,
                    post,
                    ops: vec![Operand {
                        src: x,
                        layout: Layout::contiguous(&dims),
                        access: fusor2_ir::ir::level1::AccessPlan::Alias,
                    }],
                    sched: ScheduleDomain::Point,
                }))
                .expect("point fold")
        }

        /// A `KGather` at [`ScheduleDomain::Point`] carrying the floor's mode.
        pub fn point_gather(
            &mut self,
            x: Id,
            idx: Id,
            out: &[u64],
            mode: fusor2_ir::ir::level1::GatherMode,
        ) -> Id {
            let dims: Vec<Dim> = out.iter().map(|d| Dim::Const(*d)).collect();
            let alias = |src, layout| Operand {
                src,
                layout,
                access: fusor2_ir::ir::level1::AccessPlan::Alias,
            };
            self.graph
                .add(Op::L1(L1::KGather {
                    space: fusor2_ir::ir::level1::IndexSpace::new(dims.iter().copied()),
                    axis: 0,
                    mode,
                    ops: vec![
                        alias(x, Layout::contiguous(&dims)),
                        alias(idx, Layout::contiguous(&dims[..1])),
                    ],
                    sched: ScheduleDomain::Point,
                }))
                .expect("point gather")
        }

        /// A `KScatter` at [`ScheduleDomain::Point`] carrying the floor's mode.
        pub fn point_scatter(
            &mut self,
            base: Id,
            idx: Id,
            upd: Id,
            space: &[u64],
            mode: fusor2_ir::ir::level1::ScatterMode,
            combine: ScatterCombine,
        ) -> Id {
            let dims: Vec<Dim> = space.iter().map(|d| Dim::Const(*d)).collect();
            let alias = |src, layout| Operand {
                src,
                layout,
                access: fusor2_ir::ir::level1::AccessPlan::Alias,
            };
            self.graph
                .add(Op::L1(L1::KScatter {
                    space: fusor2_ir::ir::level1::IndexSpace::new(dims.iter().copied()),
                    axis: 0,
                    mode,
                    combine,
                    ops: vec![
                        alias(base, Layout::contiguous(&dims)),
                        alias(idx, Layout::contiguous(&dims[..1])),
                        alias(upd, Layout::contiguous(&dims)),
                    ],
                    sched: ScheduleDomain::Point,
                }))
                .expect("point scatter")
        }

        /// Clone a `KContract` with a different post epilogue and union it
        /// into the same chain.
        pub fn with_post(&mut self, id: Id, post: ScalarExpr) -> Id {
            let Op::L1(l1) = &self.graph.node(id).op else {
                panic!("not an L1 node");
            };
            let mut rebuilt = l1.clone();
            if let L1::KContract { post: p, .. } = &mut rebuilt {
                *p = post;
            }
            let new = self.graph.add(Op::L1(rebuilt)).expect("post variant");
            self.graph.union(id, new).expect("union");
            new
        }

        pub fn chain(&self, id: Id) -> Vec<Id> {
            self.graph.chain(id)
        }

        /// Apply one rule to one node and return what it minted.
        pub fn apply_one(&mut self, rule: &fusor2_ir::egraph::Rule, id: Id) -> Option<Id> {
            let node = self.graph.node(id).clone();
            let facts = self.graph.facts_view(id, &self.caps);
            let mut b = self.graph.builder(&self.caps);
            (rule.apply)(&mut b, id, &node, &facts)
        }

        /// Apply every matching rule to every chain member, to a fixpoint in
        /// node count. A miniature saturation driver.
        pub fn apply_all(&mut self, rules: &[fusor2_ir::egraph::Rule], root: Id) {
            for _ in 0..6 {
                let before = self.graph.len();
                for id in self.graph.chain(root) {
                    let node = self.graph.node(id).clone();
                    let tag = node.op.tag();
                    for rule in rules {
                        if rule.head != tag {
                            continue;
                        }
                        let facts = self.graph.facts_view(id, &self.caps);
                        let mut b = self.graph.builder(&self.caps);
                        (rule.apply)(&mut b, id, &node, &facts);
                    }
                }
                if self.graph.len() == before {
                    return;
                }
            }
        }
    }

    /// The L1 op a chain member holds, if it is one.
    pub fn l1_of(fx: &Fixture, id: Id) -> Option<L1> {
        match &fx.graph.node(id).op {
            Op::L1(l1) => Some(l1.clone()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_names_are_unique_and_stable() {
        let names: Vec<&'static str> = TILE_RULES.iter().map(|r| r.name).collect();
        assert_eq!(names.len(), 15);

        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate rule name");

        assert_eq!(
            sorted,
            [
                "GATHER_ROW_PER_GROUP",
                "GATHER_VECTORIZED",
                "LOWER_COOP",
                "LOWER_GENERIC",
                "LOWER_SGEMM",
                "LOWER_SGEMV",
                "SCATTER_ATOMIC",
                "SCATTER_ONE_HOT_CONTRACT",
                "SCATTER_SORT_SEGMENT",
                "SCATTER_WG_PRIVATE_MERGE",
                "TILE_CONTRACT",
                "TILE_FOLD",
                "TILE_GATHER",
                "TILE_SCATTER",
                "UNFUSE_COOP_EPILOGUE",
            ]
        );
    }

    #[test]
    fn sched_rules_is_the_tile_table() {
        assert_eq!(SCHED_RULES.len(), TILE_RULES.len());
        assert!(std::ptr::eq(SCHED_RULES, TILE_RULES));
    }

    /// On the trainer's conv-backward contraction shapes every contraction
    /// chain and every `Scatter{Add}` chain keeps at least four alternatives,
    /// which makes the choice a cost decision rather than a routing decision.
    ///
    /// The shapes are the im2col forms of the wordseq student's three conv
    /// layers at batch 128 and the modal 768-unit bucket: `dWeight` is
    /// `A^T @ grad` over `[rows, in_ch*kernel] x [rows, out_ch]`, and `dInput`
    /// is `grad @ W^T`.
    #[test]
    fn trainer_conv_backward_chains_keep_four_alternatives() {
        use crate::domains::testing::apple_caps;
        use crate::rules::testing::{Fixture, l1_of};
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::ir::level0::{EinSpec, Label, ScatterCombine};
        use smallvec::smallvec;

        let spec = EinSpec {
            a: smallvec![Label(b'm'), Label(b'k')],
            b: smallvec![Label(b'k'), Label(b'n')],
            out: smallvec![Label(b'm'), Label(b'n')],
        };
        // (m, k, n) for conv0 7/24->64, conv1 5/64->128, conv2 3/128->128,
        // each in both backward directions, plus the 256->96->48 head.
        let shapes: [(u64, u64, u64); 8] = [
            (168, 98_304, 64),
            (98_304, 64, 168),
            (320, 24_576, 128),
            (24_576, 128, 320),
            (384, 12_288, 128),
            (12_288, 128, 384),
            (256, 128, 96),
            (96, 128, 48),
        ];
        for (m, k, n) in shapes {
            let mut fx = Fixture::new(apple_caps());
            let a = fx.buffer(Dtype::F32, &[m, k]);
            let b = fx.buffer(Dtype::F32, &[k, n]);
            let c = fx.contract(spec.clone(), Dtype::F32, a, b);
            fx.apply_all(TILE_RULES, c);
            let members = fx
                .chain(c)
                .into_iter()
                .filter(|id| l1_of(&fx, *id).is_some())
                .count();
            assert!(members >= 4, "{m}x{k}x{n} left only {members} alternatives");
        }

        // The embedding-gradient scatter: 1024 hash bins, 24 units wide.
        let mut fx = Fixture::new(apple_caps());
        let base = fx.buffer(Dtype::F32, &[1024, 24]);
        let idx = fx.buffer(Dtype::U32, &[128 * 768 * 3]);
        let upd = fx.buffer(Dtype::F32, &[128 * 768 * 3, 24]);
        let s = fx.scatter(0, ScatterCombine::Add, base, idx, upd);
        fx.apply_all(TILE_RULES, s);
        let members = fx
            .chain(s)
            .into_iter()
            .filter(|id| l1_of(&fx, *id).is_some())
            .count();
        assert!(members >= 4, "the scatter chain left only {members}");
    }

    /// A floor-lowered `KFold` arrives with no schedule decision to make;
    /// after the table runs the chain holds one carrying a real reduction
    /// domain, for the widest carrier that still fits.
    ///
    /// The 3-lane leg is a `Vector` carrier slot with empty `vec_axes`, which
    /// `check_vec_axes` admits but both `lower_*_carrier` refuse; `tile_fold`
    /// prices a domain for it anyway.
    #[test]
    fn tile_fold_upgrades_a_point_scheduled_reduction() {
        use crate::domains::testing::apple_caps;
        use crate::rules::testing::{Fixture, l1_of};
        use fusor2_ir::carrier::Carrier;
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::scalar::BinOp;
        use fusor2_ir::shape::Dim;

        let add = Carrier::binop(
            BinOp::Add,
            Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
            Dtype::F32,
        );
        for (lanes, carrier) in [(1u64, add.clone()), (3, add.promote(Dim::Const(3)).unwrap())] {
            let mut fx = Fixture::new(apple_caps());
            let x = fx.buffer(Dtype::F32, &[64, 1024]);
            let fold = fx.point_fold(x, &[64, 1024], 1, carrier);
            fx.apply_all(TILE_RULES, fold);
            let domains: Vec<usize> = fx
                .chain(fold)
                .into_iter()
                .filter_map(|m| match l1_of(&fx, m) {
                    Some(L1::KFold { sched, .. }) => Some(sched.len()),
                    _ => None,
                })
                .collect();
            assert!(
                domains.iter().any(|n| *n > 1),
                "{lanes}-lane carrier kept only {domains:?} schedule points"
            );
        }
    }

    /// `space = [rows, dh, k]` with `vec_axes = [1]` and the reduced axis last
    /// is what PROMOTE mints and what both backends lower per promoted
    /// position, so it must be scheduled here rather than reaching extraction
    /// on `ScheduleDomain::Point`. This asserts on the rule, not the table.
    #[test]
    fn tile_fold_schedules_a_promoted_nest() {
        use crate::domains::testing::apple_caps;
        use crate::rules::testing::{Fixture, l1_of};
        use fusor2_ir::carrier::Carrier;
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::scalar::BinOp;
        use fusor2_ir::shape::Dim;

        const DH: u64 = 8;
        let promoted = Carrier::binop(
            BinOp::Add,
            Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
            Dtype::F32,
        )
        .promote(Dim::Const(DH))
        .unwrap();
        assert_eq!(promoted.lanes(), Some(DH));

        let mut fx = Fixture::new(apple_caps());
        let shape = [64, DH, 1024];
        let x = fx.buffer(Dtype::F32, &shape);
        let fold = fx.promoted_fold(x, &shape, 2, &[1], promoted);
        let new = fx
            .apply_one(&TILE_FOLD, fold)
            .expect("TILE_FOLD declined a promoted nest whose reduced axis is last");
        match l1_of(&fx, new) {
            Some(L1::KFold {
                vec_axes,
                sched: ScheduleDomain::Fold(d),
                ..
            }) => {
                assert_eq!(vec_axes.as_slice(), [1], "the promotion was not carried");
                assert!(
                    d.strategies.len() > 1,
                    "a promoted nest kept {} strategies, which is not a decision",
                    d.strategies.len()
                );
            }
            other => panic!("TILE_FOLD minted {other:?}"),
        }
    }

    /// A promoted nest whose reduced axis is not last is the one form neither
    /// backend lowers, because the address arithmetic reads one output row as
    /// `vec_extent * axis_extent` consecutive elements.
    #[test]
    fn tile_fold_declines_a_promoted_nest_whose_axis_is_not_last() {
        use crate::domains::testing::apple_caps;
        use crate::rules::testing::Fixture;
        use fusor2_ir::carrier::Carrier;
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::scalar::BinOp;
        use fusor2_ir::shape::Dim;

        const DH: u64 = 8;
        let promoted = Carrier::binop(
            BinOp::Add,
            Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
            Dtype::F32,
        )
        .promote(Dim::Const(DH))
        .unwrap();

        let mut fx = Fixture::new(apple_caps());
        // `vec_axes = [1]` is still the contiguous block immediately before
        // the reduced axis 2 — `verify_l1` admits this node — but axis 2 is
        // not the last of a rank-4 space.
        let shape = [64, DH, 1024, 5];
        let x = fx.buffer(Dtype::F32, &shape);
        let fold = fx.promoted_fold(x, &shape, 2, &[1], promoted);
        assert!(
            fx.apply_one(&TILE_FOLD, fold).is_none(),
            "a domain was minted for a nest neither backend lowers"
        );
    }

    /// A `Vector(128)` f32 accumulator wants 128 KiB of scratch against
    /// Apple's 32 KiB at every lane group, so no cross-lane close is offered.
    #[test]
    fn tile_fold_offers_only_row_per_lane_for_a_carrier_too_wide_to_close() {
        use crate::domains::testing::apple_caps;
        use crate::rules::testing::{Fixture, l1_of};
        use fusor2_ir::carrier::Carrier;
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::scalar::BinOp;
        use fusor2_ir::shape::Dim;

        let wide = Carrier::binop(
            BinOp::Add,
            Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
            Dtype::F32,
        )
        .promote(Dim::Const(128))
        .unwrap();
        let mut fx = Fixture::new(apple_caps());
        let x = fx.buffer(Dtype::F32, &[64, 1024]);
        let fold = fx.point_fold(x, &[64, 1024], 1, wide);
        fx.apply_all(TILE_RULES, fold);
        // A 128-lane carrier cannot close across lanes on this device, but it
        // can still run row-per-lane, so TILE_FOLD *does* attach a domain —
        // every point of which must be a one-lane group.
        assert!(
            fx.chain(fold)
                .into_iter()
                .filter_map(|m| l1_of(&fx, m))
                .all(|m| match m {
                    L1::KFold { sched, .. } => sched
                        .iter()
                        .all(|p| !matches!(p, fusor2_ir::ir::level1::SchedPoint::Fold(s)
                            if s.lane_group(apple_caps().subgroup_width()) > 1)),
                    _ => true,
                }),
            "a cross-lane close was offered for a carrier too wide to stage"
        );
    }

    /// Gather and scatter reach the floor at [`ScheduleDomain::Point`]. Each
    /// must leave a chain member carrying a real domain **at the floor's own
    /// mode**, so the mode and the schedule stay independent decisions.
    #[test]
    fn the_floor_scheduled_node_kinds_all_gain_a_domain() {
        use crate::domains::testing::apple_caps;
        use crate::rules::testing::{Fixture, l1_of};
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::ir::level0::ScatterCombine;
        use fusor2_ir::ir::level1::{GatherMode, ScatterMode};

        // KGather, at the floor's own mode.
        let mut fx = Fixture::new(apple_caps());
        let x = fx.buffer(Dtype::F32, &[512, 128]);
        let idx = fx.buffer(Dtype::U32, &[64]);
        let g = fx.point_gather(x, idx, &[64, 128], GatherMode::RowPerGroup);
        fx.apply_all(TILE_RULES, g);
        assert!(
            fx.chain(g).into_iter().filter_map(|i| l1_of(&fx, i)).any(|l| matches!(
                l,
                L1::KGather { mode: GatherMode::RowPerGroup, sched: ScheduleDomain::Map(d), .. }
                    if d.tilings.len() > 1
            )),
            "a floor-lowered KGather kept no schedule decision at its own mode"
        );

        // KScatter, at the floor's own mode.
        let mut fx = Fixture::new(apple_caps());
        let base = fx.buffer(Dtype::F32, &[1024, 24]);
        let idx = fx.buffer(Dtype::U32, &[256]);
        let upd = fx.buffer(Dtype::F32, &[256, 24]);
        let s = fx.point_scatter(base, idx, upd, &[256, 24], ScatterMode::Atomic, ScatterCombine::Add);
        fx.apply_all(TILE_RULES, s);
        assert!(
            fx.chain(s).into_iter().filter_map(|i| l1_of(&fx, i)).any(|l| matches!(
                l,
                L1::KScatter { mode: ScatterMode::Atomic, sched: ScheduleDomain::Map(d), .. }
                    if d.tilings.len() > 1
            )),
            "a floor-lowered KScatter kept no schedule decision at its own mode"
        );
    }

    /// A domain of one point is not a decision, so the rule declines rather
    /// than growing the class with a node that offers the same schedule. A
    /// rank-1 space has no tileable dim (the innermost is excluded) and a
    /// device with one SIMD width multiplies nothing.
    #[test]
    fn a_single_point_domain_is_not_minted() {
        use crate::domains::testing::baseline_caps;
        use crate::rules::testing::{Fixture, l1_of};
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::ir::level1::GatherMode;

        let mut fx = Fixture::new(baseline_caps());
        let x = fx.buffer(Dtype::F32, &[512]);
        let idx = fx.buffer(Dtype::U32, &[64]);
        let g = fx.point_gather(x, idx, &[64], GatherMode::RowPerGroup);
        fx.apply_all(&[TILE_GATHER], g);
        assert!(
            fx.chain(g)
                .into_iter()
                .filter_map(|i| l1_of(&fx, i))
                .all(|l| matches!(l, L1::KGather { sched: ScheduleDomain::Point, .. })),
            "a one-point domain was minted anyway"
        );
    }

    /// Every minted domain round-trips: `point(i)` resolves for every index
    /// the domain reports, which is what `verify_plan` reads.
    #[test]
    fn every_minted_map_domain_resolves_every_index() {
        use crate::domains::testing::apple_caps;
        use crate::rules::testing::{Fixture, l1_of};
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::ir::level1::GatherMode;

        let mut fx = Fixture::new(apple_caps());
        let x = fx.buffer(Dtype::F32, &[512, 64, 256]);
        let idx = fx.buffer(Dtype::U32, &[16]);
        let g = fx.point_gather(idx, x, &[16, 64, 256], GatherMode::RowPerGroup);
        fx.apply_all(&[TILE_GATHER], g);
        let mut seen = 0;
        for l in fx.chain(g).into_iter().filter_map(|i| l1_of(&fx, i)) {
            if let L1::KGather { sched, .. } = l
                && !matches!(sched, ScheduleDomain::Point)
            {
                seen += 1;
                for i in 0..sched.len() {
                    assert!(sched.point(i).is_some(), "index {i} does not resolve");
                }
            }
        }
        assert!(seen > 0);
    }

    #[test]
    fn every_rule_declares_a_head_at_its_own_level() {
        for rule in TILE_RULES {
            assert_eq!(
                rule.head.level(),
                Some(rule.level),
                "{} heads a {:?} op at {:?}",
                rule.name,
                rule.head.level(),
                rule.level
            );
        }
    }
}
