//! [`CoreSemantics`]: the single [`Semantics`] implementation over the
//! closed `Logical`/`Launch` enums.

pub mod children;
pub mod infer_launch;
pub mod infer_logical;
pub mod work;

use crate::error::{Error, Result};
use crate::facts::{ValueFacts, Work};
use crate::ir::kernel::ArenaPlanner;
use crate::ir::launch::{BufferRole, Effect, Launch, ScatterMode};
use crate::ir::logical::ScatterCombine;
use crate::ir::{Children, Op, Semantics, VerifyCtx};
use std::sync::Arc;

/// The core semantics. Holds the [`ArenaPlanner`] so `verify_launch` admits
/// against the exact bytes the Kernel emitter lays out.
pub struct CoreSemantics {
    planner: Arc<dyn ArenaPlanner>,
}

impl CoreSemantics {
    /// Build the shared semantics object the e-graph is constructed with.
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
            // Alternatives infer identically; pass the first through.
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
            Op::Union(..) => Ok(()),
        }
    }

    fn effect(&self, op: &Op) -> Effect {
        effect_of(op)
    }
}

/// Purity of one operator. An atomic or `Set` scatter writes through operand
/// 0 and is pinned materialized, so its writes never apply twice.
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
