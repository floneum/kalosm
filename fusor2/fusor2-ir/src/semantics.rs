//! [`CoreSemantics`]: the single [`Semantics`] implementation covering the
//! closed `L0`/`L1` enums plus the open [`OpDefRegistry`]. Total inference,
//! work rows, effects and the two level verifiers hang off this type.

pub mod children;
pub mod infer_l0;
pub mod infer_l1;
pub mod work;

use crate::device::Caps;
use crate::error::{Error, Result};
use crate::facts::{ValueFacts, Work};
use crate::ir::level0::ScatterCombine;
use crate::ir::level1::{BufferRole, Effect, L1, ScatterMode};
use crate::ir::level2::ArenaPlanner;
use crate::ir::{Children, Op, OpDefRegistry, Semantics, VerifyCtx};
use std::sync::Arc;

/// The core semantics. Holds the [`ArenaPlanner`] because `verify_l1` admits a
/// geometry against the exact `arena_plan` bytes, the same memoized function
/// the L2 emitter lays out with.
pub struct CoreSemantics {
    planner: Arc<dyn ArenaPlanner>,
    registry: OpDefRegistry,
}

impl CoreSemantics {
    /// Build the shared semantics object the e-graph is constructed with.
    /// Returns `Arc<dyn Semantics>` because the e-graph holds only the trait
    /// object.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(planner: Arc<dyn ArenaPlanner>) -> Arc<dyn Semantics> {
        Arc::new(Self {
            planner,
            registry: OpDefRegistry::new(),
        })
    }

    /// Same, with a pre-populated extension registry.
    pub fn with_registry(
        planner: Arc<dyn ArenaPlanner>,
        registry: OpDefRegistry,
    ) -> Arc<dyn Semantics> {
        Arc::new(Self { planner, registry })
    }

    pub fn planner(&self) -> &Arc<dyn ArenaPlanner> {
        &self.planner
    }

    pub fn registry(&self) -> &OpDefRegistry {
        &self.registry
    }
}

impl Semantics for CoreSemantics {
    fn children(&self, op: &Op) -> Children {
        children::children_of(op)
    }

    fn infer(&self, op: &Op, ins: &[ValueFacts]) -> Result<ValueFacts> {
        match op {
            Op::L0(o) => infer_l0::infer_l0(o, ins),
            Op::L1(o) => infer_l1::infer_l1_with(o, ins, &self.registry),
            // A union stands for alternatives that infer identically by
            // construction; pass the first through.
            Op::Union(..) => ins
                .first()
                .cloned()
                .ok_or_else(|| Error::Shape("a Union node needs its alternatives' facts".into())),
        }
    }

    fn work(&self, op: &Op, ins: &[ValueFacts], out: &ValueFacts) -> Work {
        work::work_of_with(op, ins, out, &self.registry)
    }

    fn verify(&self, cx: &VerifyCtx<'_>) -> Result<()> {
        match cx.node.op {
            Op::L0(_) => crate::verify_l0::verify_l0(cx),
            Op::L1(_) => crate::verify_l1::verify_l1(cx, self.planner.as_ref()),
            // A union carries no semantics of its own; its operands are
            // verified as their own nodes.
            Op::Union(..) => Ok(()),
        }
    }

    fn effect(&self, op: &Op) -> Effect {
        effect_of(op)
    }
}

/// Purity of one operator.
///
/// `KScatter` writing through operand 0 with atomics or a `Set` combine mutates
/// state and is pinned in the materialized set; inlining a two-consumer atomic
/// scatter into both consumers applies the atomics twice. Everything else is
/// pure: an L0 node describes a value, not a write.
pub fn effect_of(op: &Op) -> Effect {
    match op {
        Op::L1(L1::KScatter { mode, combine, .. })
            if matches!(mode, ScatterMode::Atomic) || matches!(combine, ScatterCombine::Set) =>
        {
            Effect::InPlace(BufferRole(0))
        }
        Op::L0(_) | Op::L1(_) | Op::Union(..) => Effect::Pure,
    }
}

/// A trivially-correct [`ArenaPlanner`] for callers that need a
/// [`CoreSemantics`] without `fusor2-tile`'s planner: `fusor2-ir`'s own tests
/// and the CPU target, which has no workgroup memory, so a tile set's exact
/// footprint is the sum of its declared bytes.
pub struct SumArenaPlanner;

impl ArenaPlanner for SumArenaPlanner {
    fn arena_plan(
        &self,
        _ir: &crate::ir::level2::KernelIr,
        _caps: &Caps,
    ) -> Result<crate::ir::level2::ArenaPlan> {
        Ok(crate::ir::level2::ArenaPlan {
            mode: crate::ir::level2::ArenaMode::Regions,
            total_bytes: 0,
            placements: Default::default(),
            barriers_inserted: Default::default(),
        })
    }

    fn workgroup_bytes(&self, tiles: &crate::ir::level2::Tiles, _caps: &Caps) -> Result<u32> {
        Ok(tiles
            .decls
            .iter()
            .map(|t| (t.layout.element_count() * t.element.byte_size()) as u32)
            .sum())
    }

    fn barrier_suggestions(
        &self,
        _ir: &crate::ir::level2::KernelIr,
    ) -> Vec<crate::ir::level2::BarrierSuggestion> {
        Vec::new()
    }

    fn verify_arena(
        &self,
        _ir: &crate::ir::level2::KernelIr,
        _plan: &crate::ir::level2::ArenaPlan,
    ) -> Result<()> {
        Ok(())
    }

    fn verify_uniformity(&self, _ir: &crate::ir::level2::KernelIr) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{DeviceKind, Limits, SubgroupWidths};
    use crate::dtype::{Dtype, QFmt, QLayout, Splat};
    use crate::egraph::Id;
    use crate::carrier::{Carrier, SlotTy};
    use crate::ir::level0::{BufferId, EinSpec, L0, Label, LeafKind, TiePolicy};
    use crate::ir::level1::{
        AccessPlan, ContractSide, Family, IndexSpace, MapDomain, Operand, ScheduleDomain,
    };
    use crate::ir::{Level, Node, OpTag};
    use crate::scalar::{BinOp, CmpOp, ScalarExpr, UnOp};
    use crate::shape::{BoundsProof, Dim, Layout, SlidingWindow, StrideSpec, SymId};
    use smallvec::{SmallVec, smallvec};

    fn caps() -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "fuzz".into(),
            limits: Limits::default(),
            subgroups: Some(SubgroupWidths { min: 32, max: 32 }),
            f16: true,
            bf16: true,
            coop: Default::default(),
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: Default::default(),
            threads: 1,
        }
    }

    fn semantics() -> Arc<dyn Semantics> {
        CoreSemantics::new(Arc::new(SumArenaPlanner))
    }

    #[test]
    fn dispatch_routes_to_the_right_level() {
        let sem = semantics();
        let leaf = Op::L0(L0::Leaf(LeafKind::Param {
            name: BufferId(0),
            dtype: Dtype::F32,
            shape: smallvec![Dim::Const(8)],
        }));
        let facts = sem.infer(&leaf, &[]).unwrap();
        assert_eq!(facts.persistence, crate::dtype::Persistence::Persistent);
        assert_eq!(sem.effect(&leaf), Effect::Pure);
        assert!(sem.children(&leaf).is_empty());

        let node = Node {
            children: Children::new(),
            op: leaf,
            level: Level::L0,
        };
        let caps = caps();
        let registry = OpDefRegistry::new();
        let cx = VerifyCtx {
            node: &node,
            id: Id(0),
            operands: &[],
            result: &facts,
            caps: &caps,
            registry: &registry,
        };
        sem.verify(&cx).unwrap();
    }

    #[test]
    fn effect_table() {
        let scatter = |mode, combine| {
            Op::L1(L1::KScatter {
                space: IndexSpace::new([Dim::Const(4)]),
                axis: 0,
                mode,
                combine,
                ops: vec![],
                sched: ScheduleDomain::Point,
            })
        };
        assert_eq!(
            effect_of(&scatter(ScatterMode::Atomic, ScatterCombine::Add)),
            Effect::InPlace(BufferRole(0))
        );
        assert_eq!(
            effect_of(&scatter(ScatterMode::SortSegment, ScatterCombine::Set)),
            Effect::InPlace(BufferRole(0))
        );
        assert_eq!(
            effect_of(&scatter(ScatterMode::SortSegment, ScatterCombine::Add)),
            Effect::Pure
        );
        assert_eq!(
            effect_of(&Op::L1(L1::KMap {
                space: IndexSpace::new([Dim::Const(4)]),
                body: ScalarExpr::arg(0, Dtype::F32),
                ops: vec![],
                sched: ScheduleDomain::Point,
            })),
            Effect::Pure
        );
        assert_eq!(effect_of(&Op::Union(Id(0), Id(1))), Effect::Pure);
    }

    /// xorshift64*, so the fuzz corpus is deterministic and dependency-free.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            if n == 0 { 0 } else { self.next() % n }
        }
    }

    fn rand_dim(rng: &mut Rng) -> Dim {
        match rng.below(6) {
            0 => Dim::Sym(SymId(rng.below(3) as u32)),
            1 => Dim::Const(0),
            2 => Dim::Const(1),
            n => Dim::Const(n),
        }
    }

    fn rand_dtype(rng: &mut Rng) -> Dtype {
        match rng.below(6) {
            0 => Dtype::F16,
            1 => Dtype::BF16,
            2 => Dtype::U32,
            3 => Dtype::I32,
            4 => Dtype::Q(QFmt::Q4K),
            _ => Dtype::F32,
        }
    }

    fn rand_facts(rng: &mut Rng) -> ValueFacts {
        let rank = rng.below(5) as usize;
        let shape: SmallVec<[Dim; 6]> = (0..rank).map(|_| rand_dim(rng)).collect();
        let mut f = ValueFacts::new(rand_dtype(rng), shape);
        f.outs = rng.below(3) as u8;
        f
    }

    fn rand_expr(rng: &mut Rng, depth: u32) -> ScalarExpr {
        if depth == 0 {
            return match rng.below(4) {
                0 => ScalarExpr::lit(Splat::F32(1.5)),
                1 => ScalarExpr::uniform(SymId(0), Dtype::F32),
                2 => ScalarExpr::index_of(rng.below(4) as u32),
                _ => ScalarExpr::arg(rng.below(4) as u32, rand_dtype(rng)),
            };
        }
        match rng.below(6) {
            0 => ScalarExpr::un(UnOp::Exp, rand_expr(rng, depth - 1)),
            1 => ScalarExpr::un(UnOp::Abs, rand_expr(rng, depth - 1)),
            2 => ScalarExpr::bin(
                BinOp::Pow,
                rand_expr(rng, depth - 1),
                rand_expr(rng, depth - 1),
            ),
            3 => ScalarExpr::cmp(
                CmpOp::Lt,
                rand_expr(rng, depth - 1),
                rand_expr(rng, depth - 1),
            ),
            4 => ScalarExpr::select(
                rand_expr(rng, depth - 1),
                rand_expr(rng, depth - 1),
                rand_expr(rng, depth - 1),
            ),
            _ => ScalarExpr::cast(rand_dtype(rng), rand_expr(rng, depth - 1)),
        }
    }

    fn rand_operand(rng: &mut Rng) -> Operand {
        let rank = rng.below(4) as usize;
        let shape: Vec<Dim> = (0..rank).map(|_| rand_dim(rng)).collect();
        Operand {
            src: Id(rng.below(4) as u32),
            layout: Layout::contiguous(&shape),
            access: match rng.below(4) {
                0 => AccessPlan::Gather,
                1 => AccessPlan::Pack {
                    into: Layout::contiguous(&shape),
                },
                2 => AccessPlan::Unflatten(crate::shape::MultiFlattenMap::affine(&[2, 2], &[2, 1])),
                _ => AccessPlan::Alias,
            },
        }
    }

    fn rand_l0(rng: &mut Rng) -> L0 {
        match rng.below(10) {
            0 => L0::Leaf(match rng.below(5) {
                0 => LeafKind::Buffer {
                    name: BufferId(0),
                    dtype: rand_dtype(rng),
                    shape: smallvec![rand_dim(rng)],
                },
                1 => LeafKind::Param {
                    name: BufferId(1),
                    dtype: rand_dtype(rng),
                    shape: smallvec![rand_dim(rng)],
                },
                2 => LeafKind::Const {
                    value: Splat::F32(0.0),
                    shape: smallvec![rand_dim(rng)],
                },
                3 => LeafKind::Uniform {
                    sym: SymId(0),
                    dtype: Dtype::F32,
                },
                _ => LeafKind::Quantized {
                    name: BufferId(2),
                    fmt: QFmt::Q6K,
                    layout: QLayout::Native,
                    shape: smallvec![rand_dim(rng), rand_dim(rng)],
                },
            }),
            1 => L0::Map {
                expr: rand_expr(rng, 3),
                ins: (0..rng.below(4)).map(|i| Id(i as u32)).collect(),
                outs: rng.below(3) as u8,
            },
            2 => L0::Fold {
                carrier: rand_carrier(rng),
                axis: rng.below(5) as u32,
                acc: rand_dtype(rng),
                ins: smallvec![Id(0)],
            },
            3 => {
                let pick = |rng: &mut Rng, n: u64| -> SmallVec<[Label; 6]> {
                    (0..n).map(|_| Label(rng.below(4) as u8)).collect()
                };
                let (na, nb, no) = (rng.below(4), rng.below(4), rng.below(4));
                L0::Contract {
                    spec: EinSpec {
                        a: pick(rng, na),
                        b: pick(rng, nb),
                        out: pick(rng, no),
                    },
                    acc: rand_dtype(rng),
                    a: Id(0),
                    b: Id(1),
                    outs: rng.below(3) as u8,
                }
            }
            4 => L0::Restride {
                specs: (0..rng.below(4))
                    .map(|_| StrideSpec {
                        input_dim: rng.below(6) as u32,
                        multiplier: rng.below(3) as u32,
                        size: rand_dim(rng),
                        offset: rand_dim(rng),
                    })
                    .collect(),
                bounds: if rng.below(2) == 0 {
                    BoundsProof::Static
                } else {
                    BoundsProof::RuntimeMask
                },
                x: Id(0),
            },
            5 => L0::Window {
                specs: (0..rng.below(3))
                    .map(|_| {
                        SlidingWindow::new(
                            rng.below(5) as u32,
                            rng.below(5) as u32,
                            rng.below(5) as u32,
                        )
                    })
                    .collect(),
                x: Id(0),
            },
            6 => L0::Gather {
                axis: rng.below(5) as u32,
                x: Id(0),
                idx: Id(1),
            },
            7 => L0::Scatter {
                axis: rng.below(5) as u32,
                combine: if rng.below(2) == 0 {
                    ScatterCombine::Set
                } else {
                    ScatterCombine::Add
                },
                base: Id(0),
                idx: Id(1),
                upd: Id(2),
                unique: rng.below(2) == 0,
            },
            8 => L0::Dequant {
                fmt: QFmt::ALL[rng.below(6) as usize],
                layout: QLayout::Native,
                x: Id(0),
            },
            _ => L0::Project {
                slot: rng.below(4) as u8,
                x: Id(0),
            },
        }
    }

    /// A random carrier, including ill-typed ones: verification must not panic
    /// on any node a rule could construct.
    fn rand_carrier(rng: &mut Rng) -> Carrier {
        let d = rand_dtype(rng);
        let op = match rng.below(4) {
            0 => BinOp::Add,
            1 => BinOp::Mul,
            2 => BinOp::Max,
            _ => BinOp::Min,
        };
        let base = Carrier::binop(
            op,
            Carrier::binop_identity(op, d).unwrap_or(crate::dtype::Splat::F32(0.0)),
            d,
        );
        match rng.below(4) {
            0 => base,
            1 => base.with_tie(TiePolicy::FirstWins),
            2 => Carrier {
                slots: smallvec![SlotTy::Vector(rand_dim(rng))],
                ..base
            },
            _ => {
                let other = Carrier::binop(
                    BinOp::Max,
                    Carrier::binop_identity(BinOp::Max, d).unwrap_or(crate::dtype::Splat::F32(0.0)),
                    d,
                );
                base.tuple(&other, &crate::carrier::ArgRemap::identity(1)).carrier
            }
        }
    }

    fn rand_l1(rng: &mut Rng) -> L1 {
        let space = IndexSpace::new((0..rng.below(4)).map(|_| rand_dim(rng)).collect::<Vec<_>>());
        let ops: Vec<Operand> = (0..rng.below(4)).map(|_| rand_operand(rng)).collect();
        match rng.below(7) {
            0 => L1::KMap {
                space,
                body: rand_expr(rng, 2),
                ops,
                sched: ScheduleDomain::Point,
            },
            1 => L1::KFold {
                space,
                axis: rng.below(5) as u32,
                vec_axes: smallvec![],
                carrier: rand_carrier(rng),
                acc: rand_dtype(rng),
                post: smallvec![rand_expr(rng, 2)],
                ops,
                sched: ScheduleDomain::Point,
            },
            2 => L1::KContract {
                m: rand_dim(rng),
                n: rand_dim(rng),
                k: rand_dim(rng),
                batch: rand_dim(rng),
                family: Family::Coop,
                post: rand_expr(rng, 2),
                acc: rand_dtype(rng),
                a: ContractSide::one(rand_expr(rng, 2), rand_operand(rng)),
                b: ContractSide::one(rand_expr(rng, 2), rand_operand(rng)),
                sched: ScheduleDomain::Coop(Default::default()),
            },
            3 => L1::KGather {
                space,
                axis: rng.below(4) as u32,
                mode: crate::ir::level1::GatherMode::RowPerGroup,
                ops,
                sched: ScheduleDomain::Point,
            },
            4 => L1::KScatter {
                space,
                axis: rng.below(4) as u32,
                mode: ScatterMode::Atomic,
                combine: ScatterCombine::Add,
                ops,
                sched: ScheduleDomain::Point,
            },
            5 => L1::KRegion {
                members: (0..rng.below(4)).map(|i| Id(i as u32)).collect(),
                live_outs: (0..rng.below(3)).map(|i| i as u32).collect(),
                sched: ScheduleDomain::Map(MapDomain::linear(&caps(), rng.below(4096) as u64)),
            },
            _ => L1::Ext {
                def: crate::ir::OpDefId(rng.below(3) as u32),
                ops,
                attrs: crate::ir::AttrId(0),
            },
        }
    }

    #[test]
    fn infer_and_verify_are_total_over_10_000_random_ops() {
        let sem = semantics();
        let caps = caps();
        let registry = OpDefRegistry::new();
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

        for i in 0..10_000u32 {
            let op = if i % 2 == 0 {
                Op::L0(rand_l0(&mut rng))
            } else {
                Op::L1(rand_l1(&mut rng))
            };
            let n_ins = rng.below(4) as usize;
            let ins: Vec<ValueFacts> = (0..n_ins).map(|_| rand_facts(&mut rng)).collect();

            // Inference is total: `Ok` or a typed `Error`, never a panic.
            let inferred = sem.infer(&op, &ins);

            let level = op.level().unwrap_or(Level::L0);
            let node = Node {
                children: children::children_of(&op),
                op,
                level,
            };

            // Verification and work accounting are total at both the inferred
            // facts and a wrong set.
            let wrong = rand_facts(&mut rng);
            for result in [inferred.as_ref().ok(), Some(&wrong)].into_iter().flatten() {
                let cx = VerifyCtx {
                    node: &node,
                    id: Id(i),
                    operands: &ins,
                    result,
                    caps: &caps,
                    registry: &registry,
                };
                let _ = sem.verify(&cx);
                let _ = sem.work(&node.op, &ins, result);
            }
            let _ = sem.effect(&node.op);
        }
    }

    #[test]
    fn the_fuzz_corpus_reaches_almost_every_tag() {
        let mut rng = Rng(7);
        let mut seen: Vec<OpTag> = Vec::new();
        for _ in 0..2_000 {
            let op = if rng.below(2) == 0 {
                Op::L0(rand_l0(&mut rng))
            } else {
                Op::L1(rand_l1(&mut rng))
            };
            let tag = op.tag();
            if !seen.contains(&tag) {
                seen.push(tag);
            }
            let _ = children::children_of(&op);
        }
        // 10 L0 tags + 7 of the 8 L1 tags (KMerged needs its constructor,
        // which is exercised in `verify_l1`'s tests).
        assert!(seen.len() >= 17, "only saw {} tags", seen.len());
    }
}
