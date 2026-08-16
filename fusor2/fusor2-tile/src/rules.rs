//! The lowering rules that consult a schedule-domain generator: the
//! order-free contraction family rules plus `unfuse_coop_epilogue`, the
//! four `Scatter` lowerings, the two gather lowerings.
//!
//! All the families coexist in one chain and compete on cost.

pub mod contract;
pub mod gather;
pub mod scatter;

use fusor2_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor2_ir::ir::launch::{Launch, Operand, ScheduleDomain};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;

use crate::domains::{DomainCtx, default_planner, fold_domain_for, map_domain};

rule!(
    TILE_FOLD,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = tile_fold,
);

rule!(
    TILE_GATHER,
    level = Level::Launch,
    head = OpTag::LaunchGather,
    tag = RuleTag::Additive,
    apply = tile_gather,
);

rule!(
    TILE_SCATTER,
    level = Level::Launch,
    head = OpTag::LaunchScatter,
    tag = RuleTag::Additive,
    apply = tile_scatter,
);

/// Attach the complete legal reduction domain to a `Fold` that arrived
/// carrying [`ScheduleDomain::Point`].
///
/// The floor lowering (`fusor2-ir`) cannot generate one: schedule domains are
/// filtered by the exact arena footprint, which lives here.
///
/// The domain is generated for this carrier's lane count, so a wide
/// accumulator is filtered by workgroup storage rather than admitted and
/// crashed at `verify_plan`. An empty domain means the rule does not apply,
/// never that the node is broken.
///
/// Promoted folds included: `space = free.. ++ vec.. ++ [reduced]` is a fold
/// like any other here; both backends lower it per promoted position.
pub fn tile_fold(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(l1) = &node.op else { return None };
    let Launch::Fold {
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
    // the address arithmetic reads one output row as
    // `vec_extent * axis_extent` consecutive elements. Pricing a schedule
    // point for it would make extraction prefer a plan that fails at
    // lowering.
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
    if let Launch::Fold { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Fold(dom);
    }
    let new = b.add_launch(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// The accesses of a node's operand list, as the map-domain generator reads
/// them: a per-lane gather has no vector load to widen into, so it forbids a
/// vectorized tiling. Legality, not preference.
fn accesses(ops: &[Operand]) -> Vec<fusor2_ir::ir::launch::AccessPlan> {
    ops.iter().map(|o| o.access.clone()).collect()
}

/// Attach the elementwise tiling domain to a floor-lowered `Gather`,
/// without touching `mode`.
///
/// `gather::GATHER_*` mint a mode and a domain together; splitting them makes
/// both late decisions.
///
/// There is deliberately no `TILE_MAP` beside this: a `Map` domain minted as
/// an additive alternative measurably regresses extraction, and has to be
/// attached where the node is minted (`lower_floor.rs`) so it replaces
/// `ScheduleDomain::Point` instead of competing with it.
pub fn tile_gather(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(l1) = &node.op else { return None };
    let Launch::Gather {
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
    if let Launch::Gather { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Map(dom);
    }
    let new = b.add_launch(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// Attach the elementwise tiling domain to a floor-lowered `Scatter`,
/// without touching `mode`.
///
/// Same split as [`tile_gather`]: this only stops the floor's mode from
/// being the one alternative with no schedule.
pub fn tile_scatter(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(l1) = &node.op else { return None };
    let Launch::Scatter {
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
    if let Launch::Scatter { sched, .. } = &mut rebuilt {
        *sched = ScheduleDomain::Map(dom);
    }
    let new = b.add_launch(rebuilt).ok()?;
    b.union(id, new).ok()?;
    Some(new)
}

/// Every rule `fusor2-tile` owns, in a fixed declaration order. Order carries
/// no semantics; it exists only so a run is reproducible.
pub static TILE_RULES: &[Rule] = &[
    TILE_FOLD,
    // `Map` is deliberately absent — see the note above `tile_gather`.
    TILE_GATHER,
    TILE_SCATTER,
    contract::LOWER_COOP,
    contract::LOWER_SGEMM,
    contract::LOWER_SGEMV,
    contract::LOWER_GENERIC,
    contract::UNFUSE_COOP_EPILOGUE,
    // scatter: two coexisting lowerings
    scatter::SCATTER_ATOMIC,
    scatter::SCATTER_SORT_SEGMENT,
    // gather: two coexisting lowerings
    gather::GATHER_ROW_PER_GROUP,
    gather::GATHER_QUANTIZED_ROWS,
];

/// The name `fusor2-tile`'s rule table has always been exported under.
pub static SCHED_RULES: &[Rule] = TILE_RULES;

#[cfg(test)]
pub(crate) mod testing {
    //! A minimal [`Semantics`] and a graph fixture, so rule tests
    //! exercise the real rule bodies against a real [`EGraph`] without
    //! depending on `CoreSemantics`. Inference here is only as
    //! precise as the guards under test read.

    use fusor2_ir::Result;
    use fusor2_ir::device::Caps;
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::facts::{ValueFacts, Work};
    use fusor2_ir::ir::logical::{
        BufferId, EinSpec, Logical, Label, LeafKind, ScatterCombine,
    };
    use fusor2_ir::ir::launch::{
        Effect, Launch, Operand, ScheduleDomain,
    };
    use fusor2_ir::ir::{Children, Op, Semantics, VerifyCtx};
    use fusor2_ir::scalar::ScalarExpr;
    use fusor2_ir::shape::{Dim, Layout};
    use smallvec::{SmallVec, smallvec};
    use std::sync::Arc;

    pub struct TestSemantics;

    fn ops_children(ops: &[Operand]) -> Children {
        ops.iter().map(|o| o.src).collect()
    }

    impl Semantics for TestSemantics {
        fn children(&self, op: &Op) -> Children {
            match op {
                Op::Union(a, b) => smallvec![*a, *b],
                Op::Logical(l0) => match l0 {
                    Logical::Leaf(_) => Children::new(),
                    Logical::Map { ins, .. } | Logical::Fold { ins, .. } => {
                        ins.iter().copied().collect()
                    }
                    Logical::Restride { x, .. }
                    | Logical::Window { x, .. }
                    | Logical::Dequant { x, .. }
                    | Logical::Project { x, .. } => smallvec![*x],
                    Logical::Contract { a, b, .. } => smallvec![*a, *b],
                    Logical::Gather { x, idx, .. } => smallvec![*x, *idx],
                    Logical::Scatter {
                        base, idx, upd, ..
                    } => smallvec![*base, *idx, *upd],
                },
                Op::Launch(l1) => match l1 {
                    Launch::Map { ops, .. }
                    | Launch::Fold { ops, .. }
                    | Launch::Gather { ops, .. }
                    | Launch::Scatter { ops, .. }
                    | Launch::Ext { ops, .. } => ops_children(ops),
                    Launch::Contract { a, b, .. } => {
                        a.ops.iter().chain(b.ops.iter()).map(|o| o.src).collect()
                    }
                    Launch::Region { members, .. } => members.iter().copied().collect(),
                },
            }
        }

        fn infer(&self, op: &Op, ins: &[ValueFacts]) -> Result<ValueFacts> {
            let first = || ins.first().cloned().unwrap_or(ValueFacts::new(Dtype::F32, []));
            Ok(match op {
                Op::Union(..) => first(),
                Op::Logical(l0) => match l0 {
                    Logical::Leaf(LeafKind::Buffer { dtype, shape, .. })
                    | Logical::Leaf(LeafKind::Param { dtype, shape, .. }) => {
                        ValueFacts::new(*dtype, shape.iter().copied())
                    }
                    Logical::Leaf(LeafKind::Const { value, shape }) => {
                        ValueFacts::new(value.dtype(), shape.iter().copied())
                    }
                    Logical::Leaf(LeafKind::Uniform { dtype, .. }) => ValueFacts::new(*dtype, []),
                    Logical::Leaf(LeafKind::Quantized { fmt, shape, .. }) => {
                        ValueFacts::new(Dtype::Q(*fmt), shape.iter().copied())
                    }
                    Logical::Map { expr, .. } => {
                        ValueFacts::new(expr.dtype(), first().shape.iter().copied())
                    }
                    Logical::Fold { acc, axis, .. } => {
                        let mut shape = first().shape;
                        if (*axis as usize) < shape.len() {
                            shape.remove(*axis as usize);
                        }
                        ValueFacts::new(*acc, shape)
                    }
                    Logical::Contract { spec, acc, .. } => {
                        ValueFacts::new(*acc, out_shape(spec, ins))
                    }
                    Logical::Restride { specs, .. } => {
                        ValueFacts::new(first().dtype, specs.iter().map(|s| s.size))
                    }
                    Logical::Window { .. } | Logical::Project { .. } => first(),
                    Logical::Gather { axis, .. } => {
                        let x = first();
                        let mut shape = x.shape.clone();
                        if let (Some(slot), Some(idx)) =
                            (shape.get_mut(*axis as usize), ins.get(1))
                            && let Some(n) = idx.shape.first()
                        {
                            *slot = *n;
                        }
                        ValueFacts::new(dense(x.dtype), shape)
                    }
                    Logical::Scatter { .. } => first(),
                    Logical::Dequant { .. } => {
                        ValueFacts::new(Dtype::F32, first().shape.iter().copied())
                    }
                },
                Op::Launch(l1) => match l1 {
                    Launch::Map { body, space, .. } => {
                        ValueFacts::new(body.dtype(), space.dims.iter().copied())
                    }
                    Launch::Fold {
                        acc,
                        space,
                        axis,
                        carrier,
                        vec_axes,
                        ..
                    } => {
                        let mut dims: fusor2_ir::shape::Dims = space
                            .dims
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| {
                                *i != *axis as usize && !vec_axes.contains(&(*i as u32))
                            })
                            .map(|(_, d)| *d)
                            .collect();
                        if let Some(Some(d)) = carrier.out_dim() {
                            dims.push(d);
                        }
                        ValueFacts::new(*acc, dims)
                    }
                    Launch::Contract {
                        m, n, batch, post, ..
                    } => ValueFacts::new(post.dtype(), [*batch, *m, *n]),
                    Launch::Gather { space, .. } => {
                        ValueFacts::new(dense(first().dtype), space.dims.iter().copied())
                    }
                    Launch::Scatter { .. } => first(),
                    Launch::Region { .. } | Launch::Ext { .. } => first(),
                },
            })
        }

        fn work(&self, _op: &Op, ins: &[ValueFacts], out: &ValueFacts) -> Work {
            let elems = out.elements().unwrap_or(1);
            let reads: u64 = ins.iter().map(|i| i.elements().unwrap_or(1)).sum();
            Work {
                macs: elems,
                transcendentals: 0,
                index_ops: reads,
                wg_bytes: 0,
            }
        }

        fn verify(&self, _cx: &VerifyCtx<'_>) -> Result<()> {
            Ok(())
        }

        fn effect(&self, op: &Op) -> Effect {
            match op {
                Op::Launch(Launch::Scatter { mode, .. })
                    if *mode == fusor2_ir::ir::launch::ScatterMode::Atomic =>
                {
                    Effect::InPlace(fusor2_ir::ir::launch::BufferRole(0))
                }
                _ => Effect::Pure,
            }
        }
    }

    /// A quantized value decodes into f32 wherever a dense consumer reads
    /// it.
    fn dense(d: Dtype) -> Dtype {
        if d.is_quantized() { Dtype::F32 } else { d }
    }

    fn out_shape(spec: &EinSpec, ins: &[ValueFacts]) -> SmallVec<[Dim; 6]> {
        let lookup = |label: Label| -> Dim {
            if let Some(i) = spec.a.iter().position(|l| *l == label)
                && let Some(d) = ins.first().and_then(|f| f.shape.get(i))
            {
                return *d;
            }
            if let Some(i) = spec.b.iter().position(|l| *l == label)
                && let Some(d) = ins.get(1).and_then(|f| f.shape.get(i))
            {
                return *d;
            }
            Dim::Const(1)
        };
        spec.out.iter().copied().map(lookup).collect()
    }

    /// A graph plus the caps its rules see.
    pub struct Fixture {
        pub graph: EGraph,
        pub caps: Caps,
        next_buffer: u32,
    }

    impl Fixture {
        pub fn new(caps: Caps) -> Self {
            Self {
                graph: EGraph::new(Arc::new(TestSemantics)),
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
                .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                    name,
                    dtype,
                    shape: shape.iter().copied().collect(),
                })))
                .expect("leaf")
        }

        pub fn contract(&mut self, spec: EinSpec, acc: Dtype, a: Id, b: Id) -> Id {
            self.graph
                .add(Op::Logical(Logical::Contract {
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
                .add(Op::Logical(Logical::Gather { axis, x, idx }))
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
                .add(Op::Logical(Logical::Scatter {
                    axis,
                    combine,
                    base,
                    idx,
                    upd,
                    unique: true,
                }))
                .expect("scatter")
        }

        /// A `Fold` that arrived carrying [`ScheduleDomain::Point`], which
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
        /// `space = free.. ++ vec.. ++ [reduced]`, which is what PROMOTE mints
        /// and what both backends' per-position operand addressing reads.
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
                .add(Op::Launch(Launch::Fold {
                    space: fusor2_ir::ir::launch::IndexSpace::new(dims.iter().copied()),
                    axis,
                    vec_axes: vec_axes.iter().copied().collect(),
                    carrier,
                    acc: Dtype::F32,
                    post,
                    ops: vec![Operand {
                        src: x,
                        layout: Layout::contiguous(&dims),
                        access: fusor2_ir::ir::launch::AccessPlan::Alias,
                    }],
                    sched: ScheduleDomain::Point,
                }))
                .expect("point fold")
        }

        /// A `Gather` at [`ScheduleDomain::Point`] carrying the floor's mode.
        pub fn point_gather(
            &mut self,
            x: Id,
            idx: Id,
            out: &[u64],
            mode: fusor2_ir::ir::launch::GatherMode,
        ) -> Id {
            let dims: Vec<Dim> = out.iter().map(|d| Dim::Const(*d)).collect();
            let alias = |src, layout| Operand {
                src,
                layout,
                access: fusor2_ir::ir::launch::AccessPlan::Alias,
            };
            self.graph
                .add(Op::Launch(Launch::Gather {
                    space: fusor2_ir::ir::launch::IndexSpace::new(dims.iter().copied()),
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

        /// A `Scatter` at [`ScheduleDomain::Point`] carrying the floor's mode.
        pub fn point_scatter(
            &mut self,
            base: Id,
            idx: Id,
            upd: Id,
            space: &[u64],
            mode: fusor2_ir::ir::launch::ScatterMode,
            combine: ScatterCombine,
        ) -> Id {
            let dims: Vec<Dim> = space.iter().map(|d| Dim::Const(*d)).collect();
            let alias = |src, layout| Operand {
                src,
                layout,
                access: fusor2_ir::ir::launch::AccessPlan::Alias,
            };
            self.graph
                .add(Op::Launch(Launch::Scatter {
                    space: fusor2_ir::ir::launch::IndexSpace::new(dims.iter().copied()),
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

        /// Clone a `Contract` with a different post epilogue and union it
        /// into the same chain.
        pub fn with_post(&mut self, id: Id, post: ScalarExpr) -> Id {
            let Op::Launch(l1) = &self.graph.node(id).op else {
                panic!("not an Launch node");
            };
            let mut rebuilt = l1.clone();
            if let Launch::Contract { post: p, .. } = &mut rebuilt {
                *p = post;
            }
            let new = self.graph.add(Op::Launch(rebuilt)).expect("post variant");
            self.graph.union(id, new).expect("union");
            new
        }

        pub fn chain(&self, id: Id) -> Vec<Id> {
            self.graph.chain(id)
        }

        /// Apply **one** rule to **one** node and return what it minted, so a
        /// test can distinguish "the rule fired" from "some rule in the table
        /// happened to leave a member that matches".
        pub fn apply_one(&mut self, rule: &fusor2_ir::egraph::Rule, id: Id) -> Option<Id> {
            let node = self.graph.node(id).clone();
            let facts = self.graph.facts_view(id, &self.caps);
            let mut b = self.graph.builder(&self.caps);
            (rule.apply)(&mut b, id, &node, &facts)
        }

        /// Apply every matching rule to every chain member, to a fixpoint
        /// in node count. A miniature saturation driver — enough to let a
        /// rule that consumes another rule's output fire.
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

    /// The Launch op a chain member holds, if it is one.
    pub fn l1_of(fx: &Fixture, id: Id) -> Option<Launch> {
        match &fx.graph.node(id).op {
            Op::Launch(l1) => Some(l1.clone()),
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
        assert_eq!(names.len(), 12);

        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate rule name");

        assert_eq!(
            sorted,
            [
                "GATHER_QUANTIZED_ROWS",
                "GATHER_ROW_PER_GROUP",
                "LOWER_COOP",
                "LOWER_GENERIC",
                "LOWER_SGEMM",
                "LOWER_SGEMV",
                "SCATTER_ATOMIC",
                "SCATTER_SORT_SEGMENT",
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

    /// Applied to the trainer's conv-backward contraction shapes, every
    /// contraction chain keeps at least four members, which makes the choice
    /// a cost decision rather than a routing decision.
    ///
    /// The shapes are the im2col forms of the wordseq student's three conv
    /// layers at batch 128 and the modal 768-unit bucket: `dWeight` is
    /// `A^T @ grad` over `[rows, in_ch*kernel] x [rows, out_ch]`, and
    /// `dInput` is `grad @ W^T`.
    #[test]
    fn trainer_conv_backward_chains_keep_four_alternatives() {
        use crate::domains::testing::apple_caps;
        use crate::rules::testing::{Fixture, l1_of};
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::ir::logical::{EinSpec, Label, ScatterCombine};
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
        assert!(members >= 2, "the scatter chain left only {members}");
    }

    /// A floor-lowered `Fold` arrives with no schedule decision to make.
    /// After the table runs, the chain holds one whose domain is a real
    /// reduction domain — and the widest carrier that still fits keeps one.
    ///
    /// The 3-lane leg is a `Vector` carrier slot with empty `vec_axes`, which
    /// `check_vec_axes` admits but neither `lower_*_carrier` lowers; if a
    /// mirror precondition is ever added to `tile_fold`, change this fixture
    /// to a genuine promoted nest, not the rule.
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
                    Some(Launch::Fold { sched, .. }) => Some(sched.len()),
                    _ => None,
                })
                .collect();
            assert!(
                domains.iter().any(|n| *n > 1),
                "{lanes}-lane carrier kept only {domains:?} schedule points"
            );
        }
    }

    /// The promoted nest reaches extraction with a schedule decision:
    /// `space = [rows, dh, k]` with `vec_axes = [1]` and the reduced axis
    /// last is what PROMOTE mints and what both backends lower per promoted
    /// position.
    ///
    /// Asserts on the rule, not on the table, so it fails if `tile_fold`
    /// declines further down at `carrier.lanes()` or on an empty domain.
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
            Some(Launch::Fold {
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
    /// backend lowers; pricing a schedule point for it would make extraction
    /// prefer a plan that fails at lowering.
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
        // the reduced axis 2, but axis 2 is not the last of a rank-4 space.
        let shape = [64, DH, 1024, 5];
        let x = fx.buffer(Dtype::F32, &shape);
        let fold = fx.promoted_fold(x, &shape, 2, &[1], promoted);
        assert!(
            fx.apply_one(&TILE_FOLD, fold).is_none(),
            "a domain was minted for a nest neither backend lowers"
        );
    }

    /// The footprint clause: a `Vector(128)` f32 accumulator wants 128 KiB of
    /// scratch against Apple's 32 KiB at every lane group.
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
        // can still run row-per-lane, so TILE_FOLD does attach a domain —
        // every point of which must be a one-lane group.
        assert!(
            fx.chain(fold)
                .into_iter()
                .filter_map(|m| l1_of(&fx, m))
                .all(|m| match m {
                    Launch::Fold { sched, .. } => sched
                        .iter()
                        .all(|p| !matches!(p, fusor2_ir::ir::launch::SchedPoint::Fold(s)
                            if s.lane_group(apple_caps().subgroup_width()) > 1)),
                    _ => true,
                }),
            "a cross-lane close was offered for a carrier too wide to stage"
        );
    }

    /// Gather and scatter reach the floor at [`ScheduleDomain::Point`]. Each
    /// must leave a chain member carrying a real domain at the floor's own
    /// mode.
    #[test]
    fn the_floor_scheduled_node_kinds_all_gain_a_domain() {
        use crate::domains::testing::apple_caps;
        use crate::rules::testing::{Fixture, l1_of};
        use fusor2_ir::dtype::Dtype;
        use fusor2_ir::ir::logical::ScatterCombine;
        use fusor2_ir::ir::launch::{GatherMode, ScatterMode};

        // Gather, at the floor's own mode.
        let mut fx = Fixture::new(apple_caps());
        let x = fx.buffer(Dtype::F32, &[512, 128]);
        let idx = fx.buffer(Dtype::U32, &[64]);
        let g = fx.point_gather(x, idx, &[64, 128], GatherMode::RowPerGroup);
        fx.apply_all(TILE_RULES, g);
        assert!(
            fx.chain(g).into_iter().filter_map(|i| l1_of(&fx, i)).any(|l| matches!(
                l,
                Launch::Gather { mode: GatherMode::RowPerGroup, sched: ScheduleDomain::Map(d), .. }
                    if d.tilings.len() > 1
            )),
            "a floor-lowered Gather kept no schedule decision at its own mode"
        );

        // Scatter, at the floor's own mode.
        let mut fx = Fixture::new(apple_caps());
        let base = fx.buffer(Dtype::F32, &[1024, 24]);
        let idx = fx.buffer(Dtype::U32, &[256]);
        let upd = fx.buffer(Dtype::F32, &[256, 24]);
        let s = fx.point_scatter(base, idx, upd, &[256, 24], ScatterMode::Atomic, ScatterCombine::Add);
        fx.apply_all(TILE_RULES, s);
        assert!(
            fx.chain(s).into_iter().filter_map(|i| l1_of(&fx, i)).any(|l| matches!(
                l,
                Launch::Scatter { mode: ScatterMode::Atomic, sched: ScheduleDomain::Map(d), .. }
                    if d.tilings.len() > 1
            )),
            "a floor-lowered Scatter kept no schedule decision at its own mode"
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
        use fusor2_ir::ir::launch::GatherMode;

        let mut fx = Fixture::new(baseline_caps());
        let x = fx.buffer(Dtype::F32, &[512]);
        let idx = fx.buffer(Dtype::U32, &[64]);
        let g = fx.point_gather(x, idx, &[64], GatherMode::RowPerGroup);
        fx.apply_all(&[TILE_GATHER], g);
        assert!(
            fx.chain(g)
                .into_iter()
                .filter_map(|i| l1_of(&fx, i))
                .all(|l| matches!(l, Launch::Gather { sched: ScheduleDomain::Point, .. })),
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
        use fusor2_ir::ir::launch::GatherMode;

        let mut fx = Fixture::new(apple_caps());
        let x = fx.buffer(Dtype::F32, &[512, 64, 256]);
        let idx = fx.buffer(Dtype::U32, &[16]);
        let g = fx.point_gather(idx, x, &[16, 64, 256], GatherMode::RowPerGroup);
        fx.apply_all(&[TILE_GATHER], g);
        let mut seen = 0;
        for l in fx.chain(g).into_iter().filter_map(|i| l1_of(&fx, i)) {
            if let Launch::Gather { sched, .. } = l
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
