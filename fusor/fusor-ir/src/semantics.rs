//! [`CoreSemantics`]: the single [`Semantics`] implementation covering the
//! closed `Logical`/`Launch` enums. Total inference,
//! work rows, effects and the two level verifiers hang off this type.

pub mod children;
pub mod infer_launch;
pub mod infer_logical;
pub mod work;

use crate::error::{Error, Result};
use crate::facts::{ValueFacts, Work};
use crate::ir::kernel::ArenaPlanner;
use crate::ir::launch::{BufferRole, Effect, Launch, ScatterMode};
use crate::ir::logical::ScatterCombine;
use crate::ir::{Children, Level, Op, Semantics, VerifyCtx};
use std::sync::Arc;

/// The core semantics. Holds the [`ArenaPlanner`] because `verify_launch` admits
/// a geometry against the *exact* `arena_plan` bytes — the same pure
/// memoized function the Kernel emitter lays out with, so there is no Launch/Kernel
/// admission mismatch.
pub struct CoreSemantics {
    planner: Arc<dyn ArenaPlanner>,
}

impl CoreSemantics {
    /// Build the shared semantics object the e-graph is constructed with.
    /// Returns `Arc<dyn Semantics>`: the e-graph only ever holds the trait object.
    #[allow(clippy::new_ret_no_self)]
    pub fn new(planner: Arc<dyn ArenaPlanner>) -> Arc<dyn Semantics> {
        Arc::new(Self { planner })
    }

    pub fn planner(&self) -> &Arc<dyn ArenaPlanner> {
        &self.planner
    }
}

impl Semantics for CoreSemantics {
    fn children(&self, op: &Op) -> Children {
        children::children_of(op)
    }

    fn infer(&self, op: &Op, ins: &[ValueFacts]) -> Result<ValueFacts> {
        match op {
            Op::Logical(o) => infer_logical::infer_logical(o, ins),
            Op::Launch(o) => infer_launch::infer_launch(o, ins),
            // A union stands for alternatives that infer identically by
            // construction; pass the first through.
            Op::Union(..) => ins
                .first()
                .cloned()
                .ok_or_else(|| Error::Shape("a Union node needs its alternatives' facts".into())),
        }
    }

    fn work(&self, op: &Op, ins: &[ValueFacts], out: &ValueFacts) -> Work {
        work::work_of(op, ins, out)
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
