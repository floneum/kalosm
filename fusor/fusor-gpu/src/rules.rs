//! GPU-exclusive lowering rules: the ones that mention lane or subgroup
//! geometry. Logical rules are inherited from `fusor-ir`, schedule-domain
//! rules from `fusor-tile`.
//!
//! Every guard reads [`Facts`] alone — device capabilities, shapes and dtypes.
//! A rule that would not pay still fires; `fusor-cost` rejects it on
//! realized-DAG cost.

use fusor_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor_ir::ir::launch::{Launch, ScatterMode};
use fusor_ir::ir::{Level, Node, Op, OpTag};
use fusor_ir::rule;

/// Rules only this backend contributes.
pub static GPU_RULES: &[Rule] = &[GPU_SCATTER_ATOMIC];

rule!(
    GPU_SCATTER_ATOMIC,
    level = Level::Launch,
    head = OpTag::LaunchScatter,
    tag = RuleTag::Additive,
    apply = gpu_scatter_atomic,
);

/// Mint `Scatter{Atomic}`.
///
/// The only legality question is whether the device has `atomicAdd` on f32 in
/// storage.
fn gpu_scatter_atomic(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.caps().atomic_f32 {
        return None;
    }
    let Op::Launch(Launch::Scatter {
        space,
        axis,
        mode,
        combine,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    if *mode == ScatterMode::Atomic {
        return None;
    }
    if *combine != fusor_ir::ir::logical::ScatterCombine::Add {
        return None;
    }
    let alt = b
        .add_launch(Launch::Scatter {
            space: space.clone(),
            axis: *axis,
            mode: ScatterMode::Atomic,
            combine: *combine,
            ops: ops.clone(),
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, alt).ok()
}
