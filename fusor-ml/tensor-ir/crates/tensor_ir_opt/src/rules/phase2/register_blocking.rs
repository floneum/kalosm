//! Generic register-blocking rewrite: take a single-output `Dispatch` and
//! union it with a multi-output variant where each lane produces
//! `blocking_factor` outputs.
//!
//! Structurally: the input `Dispatch`'s value and output address both
//! reference `VarRef::thread(Workgroup)`. We substitute `workgroup` with
//! `blocking_factor * workgroup + i` for `i in 0..blocking_factor` to
//! create `blocking_factor` distinct `(value, addr)` pairs, then divide
//! the workgroup count by the factor. Every output pair then describes a
//! distinct output element under the same launch configuration. If
//! substitution does not actually produce distinct output addresses, the
//! rule skips the blocked variant.
//!
//! This replaces the template-emitted `reg_m * reg_n` output fan-out
//! with a local rewrite. There's no `reg_m` / `reg_n` tag — the
//! "register-blocked" shape IS a Dispatch whose children list carries
//! more than one `(value, addr)` pair.

use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::binding;
use crate::language::{DispatchNode, TensorIr, extract_list, try_add_value_addr_dispatch};
use crate::rules::RunnerConfig;
use crate::types::{BinaryOp, BinderKind, DeviceProfile, ScalarValue, slots};

/// Build a register-blocking rewrite for the given blocking factor.
///
/// Device-feasibility is inlined as a literal: the combined per-lane
/// register footprint must fit inside `device.max_registers_per_lane`
/// assuming 4-byte scalars.
pub(super) fn build(factor: u32, device: &DeviceProfile) -> Rewrite<TensorIr, TensorAnalysis> {
    let budget_slots = device.max_registers_per_lane;
    let device_ok = factor <= budget_slots.max(1);
    Rewrite::new(
        format!("register-block-{factor}"),
        SimpleEclassSearcher::new(move |egraph, eclass| {
            if !device_ok {
                return false;
            }
            egraph[eclass]
                .iter()
                .any(|node| dispatch_is_blockable(egraph, node, factor))
        }),
        crate::applier::AdaptedApplier(RegisterBlockingApplier { factor }),
    )
    .unwrap()
}

fn dispatch_is_blockable(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
    factor: u32,
) -> bool {
    let TensorIr::Dispatch(DispatchNode::Dispatch {
        workgroups,
        num_inputs,
        children_list,
    }) = node
    else {
        return false;
    };
    if factor <= 1 || !workgroups.is_multiple_of(factor) {
        return false;
    }
    let children = extract_list(egraph, *children_list);
    let body_len = children.len().saturating_sub(*num_inputs as usize);
    // Only fire on scalar dispatches (one (value, addr) pair). Multi-
    // output dispatches are already blocked.
    body_len == 2
}

struct RegisterBlockingApplier {
    factor: u32,
}

impl crate::applier::TypedApplier for RegisterBlockingApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        let blockable: Vec<_> = egraph[eclass]
            .iter()
            .filter(|n| dispatch_is_blockable(egraph, n, self.factor))
            .cloned()
            .collect();

        let mut results = Vec::new();
        for node in blockable {
            let TensorIr::Dispatch(DispatchNode::Dispatch {
                workgroups,
                num_inputs,
                children_list,
            }) = node
            else {
                continue;
            };

            let children = extract_list(egraph, children_list);
            let num_inputs_us = num_inputs as usize;
            if children.len() != num_inputs_us + 2 {
                continue;
            }
            let inputs = children[..num_inputs_us].to_vec();
            let orig_value = children[num_inputs_us];
            let orig_addr = children[num_inputs_us + 1];

            let new_workgroups = workgroups / self.factor;
            let factor_lit = egraph.add(TensorIr::Const(ScalarValue::U32(self.factor)));

            let mut new_children = inputs;
            for i in 0..self.factor {
                // Build the replacement workgroup index: new_wg * factor + i.
                let wg_var = egraph.add(TensorIr::Simd(crate::language::SimdNode::Var(
                    crate::types::VarRef::thread(crate::types::IndexLevel::Workgroup),
                )));
                let scaled = egraph.add(TensorIr::BinOp(BinaryOp::Mul, [wg_var, factor_lit]));
                let offset = egraph.add(TensorIr::Const(ScalarValue::U32(i)));
                let replacement = egraph.add(TensorIr::BinOp(BinaryOp::Add, [scaled, offset]));

                // Substitute workgroup refs in both value and addr. The new
                // ref is itself a workgroup-bound expression, so the
                // substitution is idempotent-safe: subsequent i's see the
                // original children (we call `subst` on the originals each
                // iteration, not on the already-rewritten form).
                let new_value = binding::subst(
                    egraph,
                    orig_value,
                    BinderKind::Dispatch,
                    slots::DISPATCH_WORKGROUP,
                    0,
                    replacement,
                );
                let new_addr = binding::subst(
                    egraph,
                    orig_addr,
                    BinderKind::Dispatch,
                    slots::DISPATCH_WORKGROUP,
                    0,
                    replacement,
                );
                new_children.push(new_value);
                new_children.push(new_addr);
            }

            let Some(blocked) =
                try_add_value_addr_dispatch(egraph, new_workgroups, num_inputs, &new_children)
            else {
                continue;
            };
            egraph.union(eclass, blocked);
            results.push(blocked);
        }

        results
    }
}

#[must_use]
pub fn all(config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    // The device register budget gates each factor in `build`, so wider
    // variants are available on GPUs with enough per-lane register headroom
    // without being forced onto smaller devices.
    const BLOCKING_FACTORS: &[u32] = &[2, 4, 8, 16];
    BLOCKING_FACTORS
        .iter()
        .copied()
        .map(|f| build(f, &config.device))
        .collect()
}
