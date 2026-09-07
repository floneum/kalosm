//! [`CoreSemantics`]: the single [`Semantics`] implementation covering the
//! closed `Logical`/`Launch` enums plus the open [`OpDefRegistry`]. Total inference,
//! work rows, effects and the two level verifiers hang off this type.

pub mod children;
pub mod infer_launch;
pub mod infer_logical;
pub mod work;

use crate::device::Caps;
use crate::error::{Error, Result};
use crate::facts::{ValueFacts, Work};
use crate::ir::kernel::ArenaPlanner;
use crate::ir::launch::{BufferRole, Effect, Launch, ScatterMode};
use crate::ir::logical::ScatterCombine;
use crate::ir::{Children, Level, Op, OpDefRegistry, Semantics, VerifyCtx};
use std::sync::Arc;

/// The core semantics. Holds the [`ArenaPlanner`] because `verify_launch` admits
/// a geometry against the *exact* `arena_plan` bytes — the same pure
/// memoized function the Kernel emitter lays out with, so there is no Launch/Kernel
/// admission mismatch.
pub struct CoreSemantics {
    planner: Arc<dyn ArenaPlanner>,
    registry: OpDefRegistry,
}

impl CoreSemantics {
    /// Build the shared semantics object the e-graph is constructed with.
    /// Returns `Arc<dyn Semantics>`: the e-graph only ever holds the trait object.
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
            Op::Logical(o) => infer_logical::infer_logical(o, ins),
            Op::Launch(o) => infer_launch::infer_launch_with(o, ins, &self.registry),
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
            Op::Logical(_) => crate::verify_l0::verify_l0(cx),
            Op::Launch(_) => crate::verify_launch::verify_launch(cx, self.planner.as_ref()),
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
/// `Scatter` writing through operand 0 with atomics or a `Set` combine
/// mutates state and is therefore **pinned in the materialized set**:
/// without that, toggling a two-consumer atomic scatter out of `M` inlines it
/// into both consumers' kernels and the atomics apply twice, doubling the
/// embedding gradient. Everything else is pure — a Logical node describes a
/// value, not a write.
pub fn effect_of(op: &Op) -> Effect {
    match op {
        Op::Launch(Launch::Scatter { mode, combine, .. })
            if matches!(mode, ScatterMode::Atomic) || matches!(combine, ScatterCombine::Set) =>
        {
            Effect::InPlace(BufferRole(0))
        }
        Op::Logical(_) | Op::Launch(_) | Op::Union(..) => Effect::Pure,
    }
}

/// Level of an operator, for callers that build a [`crate::ir::Node`] by hand.
/// `Union` inherits its operands' level, which the e-graph resolves; here it
/// defaults to `Logical`.
pub fn level_of(op: &Op) -> Level {
    op.level().unwrap_or(Level::Logical)
}

/// A trivially-correct [`ArenaPlanner`] for callers that need a
/// [`CoreSemantics`] before `fusor-tile`'s planner exists — notably
/// `fusor-ir`'s own tests and the CPU target, which has no workgroup memory
/// at all, so the exact footprint of any tile set really is the sum of its
/// declared bytes.
pub struct SumArenaPlanner;

impl ArenaPlanner for SumArenaPlanner {
    fn arena_plan(
        &self,
        _ir: &crate::ir::kernel::KernelIr,
        _caps: &Caps,
    ) -> Result<crate::ir::kernel::ArenaPlan> {
        Ok(crate::ir::kernel::ArenaPlan {
            mode: crate::ir::kernel::ArenaMode::Regions,
            total_bytes: 0,
            placements: Default::default(),
            barriers_inserted: Default::default(),
        })
    }

    fn workgroup_bytes(&self, tiles: &crate::ir::kernel::Tiles, _caps: &Caps) -> Result<u32> {
        Ok(tiles
            .decls
            .iter()
            .map(|t| (t.layout.element_count() * t.element.byte_size()) as u32)
            .sum())
    }

    fn barrier_suggestions(
        &self,
        _ir: &crate::ir::kernel::KernelIr,
    ) -> Vec<crate::ir::kernel::BarrierSuggestion> {
        Vec::new()
    }

    fn verify_arena(
        &self,
        _ir: &crate::ir::kernel::KernelIr,
        _plan: &crate::ir::kernel::ArenaPlan,
    ) -> Result<()> {
        Ok(())
    }

    fn verify_uniformity(&self, _ir: &crate::ir::kernel::KernelIr) -> Result<()> {
        Ok(())
    }
}
