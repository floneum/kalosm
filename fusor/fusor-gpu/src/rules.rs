//! GPU-exclusive lowering rules: the ones that mention lane or subgroup
//! geometry. Guards read [`Facts`] alone; `fusor-cost` rejects what won't pay.

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
/// The only legality question is f32 `atomicAdd` in storage.
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
