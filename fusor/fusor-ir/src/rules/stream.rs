use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::Launch;
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;

rule!(
    STREAM_FOLD,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = stream_fold
);

pub fn stream_fold(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(fold @ Launch::Fold { ops, .. }) = &node.op else {
        return None;
    };
    if !f.own().numeric.reassoc {
        return None;
    }
    for (slot, read) in ops.iter().enumerate() {
        for source in b.class_members(read.src) {
            if b.class_of(source) == b.class_of(id) {
                continue;
            }
            let Op::Launch(producer @ Launch::Fold { .. }) = &b.node(source).op else {
                continue;
            };
            let Some(op) = Launch::stream_fold(producer.clone(), fold.clone(), slot as u32) else {
                continue;
            };
            let inputs: rustc_hash::FxHashSet<_> = crate::semantics::children::children_launch(&op)
                .into_iter()
                .map(|x| b.class_of(x))
                .collect();
            if inputs.len() + 2 > b.caps().limits.max_storage_buffers_per_shader_stage as usize {
                continue;
            }
            let Ok(candidate) = b.add_launch(op) else {
                continue;
            };
            return b.union(id, candidate).ok();
        }
    }
    None
}
