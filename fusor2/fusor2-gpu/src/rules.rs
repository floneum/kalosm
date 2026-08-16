//! GPU-exclusive lowering rules: the ones that mention lane or subgroup
//! geometry. Logical rules are inherited from `fusor2-ir`, schedule-domain
//! rules from `fusor2-tile`.
//!
//! Every guard reads [`Facts`] alone — device capabilities, shapes and dtypes.
//! A rule that would not pay still fires; `fusor2-cost` rejects it on
//! realized-DAG cost.

use fusor2_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor2_ir::ir::launch::{FoldDomain, FoldStrat, Launch, ScheduleDomain, ScatterMode};
use fusor2_ir::ir::{Level, Node, Op, OpTag};
use fusor2_ir::rule;

/// Rules only this backend contributes.
pub static GPU_RULES: &[Rule] = &[
    GPU_FOLD_SUBGROUP,
    GPU_COOP_STAGE_VIA_TILE,
    GPU_SCATTER_ATOMIC,
];


rule!(
    GPU_FOLD_SUBGROUP,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = gpu_fold_subgroup,
);

rule!(
    GPU_COOP_STAGE_VIA_TILE,
    level = Level::Launch,
    head = OpTag::LaunchContract,
    tag = RuleTag::Additive,
    apply = gpu_coop_stage_via_tile,
);

rule!(
    GPU_SCATTER_ATOMIC,
    level = Level::Launch,
    head = OpTag::LaunchScatter,
    tag = RuleTag::Additive,
    apply = gpu_scatter_atomic,
);

/// Mint `FoldStrat::Subgroup` into a `Fold`'s domain.
///
/// Legality is a *fixed* subgroup width: a ranged width makes every
/// subgroup-size-aware body treat the device as variable.
fn gpu_fold_subgroup(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.caps().subgroups.is_some_and(|s| s.is_fixed()) {
        return None;
    }
    let Op::Launch(Launch::Fold { sched, .. }) = &node.op else {
        return None;
    };
    let ScheduleDomain::Fold(domain) = sched else {
        return None;
    };
    if domain.strategies.contains(&FoldStrat::Subgroup) {
        return None;
    }
    let mut strategies = domain.strategies.clone();
    strategies.push(FoldStrat::Subgroup);
    let Op::Launch(Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        ..
    }) = &node.op
    else {
        return None;
    };
    let alt = b
        .add_launch(Launch::Fold {
            space: space.clone(),
            axis: *axis,
            vec_axes: vec_axes.clone(),
            carrier: carrier.clone(),
            acc: *acc,
            post: post.clone(),
            ops: ops.clone(),
            sched: ScheduleDomain::Fold(FoldDomain { strategies }),
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// Mint the staged-store cooperative alternative.
///
/// Without the fork's mixed-precision cooperative store, an f32-accumulated
/// kernel writing f16 output must stage through a workgroup tile and cast per
/// lane.
fn gpu_coop_stage_via_tile(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if f.caps().mixed_precision_coop_store {
        return None;
    }
    let Op::Launch(Launch::Contract {
        m,
        n,
        k,
        batch,
        family,
        post,
        acc,
        a,
        b: rhs,
        sched,
    }) = &node.op
    else {
        return None;
    };
    if *family != fusor2_ir::ir::launch::Family::Coop {
        return None;
    }
    // Only a narrowing accumulator needs the staging tile.
    if *acc == f.own().dtype {
        return None;
    }
    let ScheduleDomain::Coop(domain) = sched else {
        return None;
    };
    // The staged form wants a single-buffered tile budget: staging depth 1
    // frees the bytes the cast tile costs.
    if domain.staging.as_slice() == [1] {
        return None;
    }
    let mut alt_domain = domain.clone();
    alt_domain.staging = smallvec::smallvec![1];
    let alt = b
        .add_launch(Launch::Contract {
            m: *m,
            n: *n,
            k: *k,
            batch: *batch,
            family: fusor2_ir::ir::launch::Family::Coop,
            post: post.clone(),
            acc: *acc,
            a: a.clone(),
            b: rhs.clone(),
            sched: ScheduleDomain::Coop(alt_domain),
        })
        .ok()?;
    b.union(id, alt).ok()
}

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
    if *combine != fusor2_ir::ir::logical::ScatterCombine::Add {
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

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::device::{Caps, DeviceKind, Limits, SubgroupWidths};

    fn caps() -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "test".into(),
            limits: Limits::default(),
            subgroups: Some(SubgroupWidths { min: 32, max: 32 }),
            f16: true,
            bf16: false,
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

    #[test]
    fn every_gpu_rule_is_additive_and_head_filtered() {
        assert_eq!(GPU_RULES.len(), 3);
        for r in GPU_RULES {
            assert_eq!(r.tag, RuleTag::Additive, "{}", r.name);
            assert_eq!(r.level, Level::Launch, "{}", r.name);
            assert_ne!(r.head, OpTag::Union, "{}", r.name);
        }
    }

    #[test]
    fn rule_names_are_the_constant_identifiers() {
        let names: Vec<_> = GPU_RULES.iter().map(|r| r.name).collect();
        assert!(names.contains(&"GPU_FOLD_SUBGROUP"));
        assert!(names.contains(&"GPU_COOP_STAGE_VIA_TILE"));
        assert!(names.contains(&"GPU_SCATTER_ATOMIC"));
    }

    #[test]
    fn heads_dispatch_to_the_right_op_family() {
        let by = |n: &str| GPU_RULES.iter().find(|r| r.name == n).unwrap().head;
        assert_eq!(by("GPU_FOLD_SUBGROUP"), OpTag::LaunchFold);
        assert_eq!(by("GPU_SCATTER_ATOMIC"), OpTag::LaunchScatter);
        assert_eq!(by("GPU_COOP_STAGE_VIA_TILE"), OpTag::LaunchContract);
    }

    #[test]
    fn a_variable_subgroup_width_blocks_the_subgroup_fold() {
        let mut c = caps();
        c.subgroups = Some(SubgroupWidths { min: 8, max: 32 });
        assert!(!c.subgroups.unwrap().is_fixed());
        c.subgroups = None;
        assert!(!c.subgroups.is_some_and(|s| s.is_fixed()));
    }

    /// `Facts` has no accessor a profitability judgement could use.
    #[test]
    fn facts_exposes_no_profitability_signal() {
        fn _assert_guard_shape(f: &Facts<'_>) {
            let _ = f.caps();
            let _ = f.level();
            let _ = f.own();
            let _ = f.operands();
        }
    }
}
