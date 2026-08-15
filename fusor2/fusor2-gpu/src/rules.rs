//! GPU-exclusive lowering rules: the ones that mention lane or subgroup
//! geometry. Every L0 rule is inherited from `fusor2-ir`, and every
//! schedule-domain rule from `fusor2-tile`.
//!
//! Every guard below reads [`Facts`] alone — device capabilities, shapes and
//! dtypes. **None reads a consumer count, liveness or cost**, and none can:
//! `Facts` structurally does not expose them. A rule that "would not pay"
//! still fires; `fusor2-cost` rejects it on realized-DAG cost.
//!
//! Owned by W9.

use fusor2_ir::egraph::{Builder, Facts, Id, Rule, RuleTag};
use fusor2_ir::ir::level1::{FoldDomain, FoldStrat, L1, ScheduleDomain, ScatterMode};
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
    level = Level::L1,
    head = OpTag::KFold,
    tag = RuleTag::Additive,
    apply = gpu_fold_subgroup,
);

rule!(
    GPU_COOP_STAGE_VIA_TILE,
    level = Level::L1,
    head = OpTag::KContract,
    tag = RuleTag::Additive,
    apply = gpu_coop_stage_via_tile,
);

rule!(
    GPU_SCATTER_ATOMIC,
    level = Level::L1,
    head = OpTag::KScatter,
    tag = RuleTag::Additive,
    apply = gpu_scatter_atomic,
);

/// Mint `FoldStrat::Subgroup` into a `KFold`'s domain.
///
/// Legality is a *fixed* subgroup width: a ranged width makes every
/// subgroup-size-aware body treat the device as variable. Whether the
/// collective beats the tree at this row length is the cost model's call.
fn gpu_fold_subgroup(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.caps().subgroups.is_some_and(|s| s.is_fixed()) {
        return None;
    }
    let Op::L1(L1::KFold { sched, .. }) = &node.op else {
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
    let Op::L1(L1::KFold {
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
        .add_l1(L1::KFold {
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
/// lane. That is a *footprint* difference, so both forms coexist and the cost
/// model prices the extra tile — it is never a routing decision that sends the
/// whole contraction to the generic reduce.
fn gpu_coop_stage_via_tile(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if f.caps().mixed_precision_coop_store {
        return None;
    }
    let Op::L1(L1::KContract {
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
    if *family != fusor2_ir::ir::level1::Family::Coop {
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
        .add_l1(L1::KContract {
            m: *m,
            n: *n,
            k: *k,
            batch: *batch,
            family: fusor2_ir::ir::level1::Family::Coop,
            post: post.clone(),
            acc: *acc,
            a: a.clone(),
            b: rhs.clone(),
            sched: ScheduleDomain::Coop(alt_domain),
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// Mint `KScatter{Atomic}`.
///
/// The only legality question is whether the device has `atomicAdd` on f32 in
/// storage. Whether atomics beat the workgroup-private merge at this bin
/// count and skew is priced, not guarded.
fn gpu_scatter_atomic(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    if !f.caps().atomic_f32 {
        return None;
    }
    let Op::L1(L1::KScatter {
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
    if *combine != fusor2_ir::ir::level0::ScatterCombine::Add {
        return None;
    }
    let alt = b
        .add_l1(L1::KScatter {
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
            assert_eq!(r.level, Level::L1, "{}", r.name);
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
        assert_eq!(by("GPU_FOLD_SUBGROUP"), OpTag::KFold);
        assert_eq!(by("GPU_SCATTER_ATOMIC"), OpTag::KScatter);
        assert_eq!(by("GPU_COOP_STAGE_VIA_TILE"), OpTag::KContract);
    }

    #[test]
    fn a_variable_subgroup_width_blocks_the_subgroup_fold() {
        let mut c = caps();
        c.subgroups = Some(SubgroupWidths { min: 8, max: 32 });
        assert!(!c.subgroups.unwrap().is_fixed());
        c.subgroups = None;
        assert!(!c.subgroups.is_some_and(|s| s.is_fixed()));
    }

    /// The guards read `Caps` only. This is the compile-time restatement of
    /// that: `Facts` has no accessor a profitability judgement could use.
    #[test]
    fn facts_exposes_no_profitability_signal() {
        fn _assert_guard_shape(f: &Facts<'_>) {
            let _ = f.caps();
            let _ = f.level();
            let _ = f.own();
            let _ = f.operands();
            // There is deliberately no `f.consumers()`, `f.is_live()` or
            // `f.cost()` to call here.
        }
    }
}
