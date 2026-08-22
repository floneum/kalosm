//! CPU-specific graph rewrites.
//!
//! Scheduling and access selection belong to the native emitter now. The only
//! graph rewrite needed by this backend widens narrow storage values to f32
//! compute and narrows once at the output boundary.

use fusor_ir::dtype::Dtype;
use fusor_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor_ir::ir::launch::{Launch, Operand, ScheduleDomain};
use fusor_ir::ir::{Level, Node, Op, OpTag};
use fusor_ir::rule;
use fusor_ir::scalar::ScalarExpr;

rule!(
    WIDEN_COMPUTE,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = widen_compute,
);

/// Every rule this backend contributes; the order carries no semantics.
pub static CPU_RULES: &[Rule] = &[WIDEN_COMPUTE];

fn map_parts(node: &Node) -> Option<(&Vec<Operand>, &ScheduleDomain, &ScalarExpr)> {
    match &node.op {
        Op::Launch(Launch::Map {
            ops, sched, body, ..
        }) => Some((ops, sched, body)),
        _ => None,
    }
}

fn rebuild(
    node: &Node,
    ops: Vec<Operand>,
    sched: ScheduleDomain,
    body: ScalarExpr,
) -> Option<Launch> {
    match &node.op {
        Op::Launch(Launch::Map { space, .. }) => Some(Launch::Map {
            space: space.clone(),
            body,
            ops,
            sched,
        }),
        _ => None,
    }
}

pub(crate) fn widen_compute(
    builder: &mut Builder<'_>,
    id: Id,
    node: &Node,
    facts: &Facts<'_>,
) -> Option<Id> {
    let (ops, sched, body) = map_parts(node)?;
    let narrow = |dtype: Dtype| matches!(dtype, Dtype::F16 | Dtype::BF16);
    let output = facts.own().dtype;
    if !narrow(output) && !(0..ops.len()).any(|index| facts.dtype(index).is_some_and(narrow)) {
        return None;
    }

    let inputs = (0..ops.len())
        .map(|index| {
            let dtype = facts.dtype(index).unwrap_or(Dtype::F32);
            let arg = ScalarExpr::arg(index as u32, dtype);
            if narrow(dtype) {
                ScalarExpr::cast(Dtype::F32, arg)
            } else {
                arg
            }
        })
        .collect::<Vec<_>>();
    let body = body.compose(&inputs);
    let body = if narrow(output) {
        ScalarExpr::cast(output, body)
    } else {
        body
    };
    let alternative = rebuild(node, ops.clone(), sched.clone(), body)?;
    let alternative = builder.add_launch(alternative).ok()?;
    builder.union(id, alternative).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_stable_and_named() {
        let names = CPU_RULES.iter().map(|rule| rule.name).collect::<Vec<_>>();
        assert_eq!(names, vec!["WIDEN_COMPUTE"]);
        assert!(CPU_RULES.iter().all(|rule| rule.level == Level::Launch));
    }

    #[test]
    fn parallelism_has_no_hardcoded_work_threshold() {
        for source in [
            include_str!("launch.rs"),
            include_str!("pool.rs"),
            include_str!("emit.rs"),
            include_str!("target.rs"),
        ] {
            assert!(!source.contains("const PARALLEL_THRESHOLD"));
        }
    }
}
