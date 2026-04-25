//! Post-lowering merge of two independent same-iteration reductions.
//!
//! Input: two e-classes each containing
//!     `Dispatch { workgroups=W, num_inputs=N, reg_m=1, reg_n=1,
//!                 children = [inputs.., theta, out_addr] }`
//! where `theta = Theta { role: Reduction, count=K, init, update }`,
//! the two Dispatches share `(W, N, inputs, K)`, and neither `update`
//! body references the other's `theta`.
//!
//! Output: one merged Dispatch with `reg_n=2` (signalling a 2-output
//! layout — the `(val, addr)` pair count that `dispatch_body_pairs`
//! reads) whose single `Theta { role: RunningReduction }` carries a
//! `Pack(init₁, init₂)` accumulator and `Pack(update₁, update₂)`
//! update, and whose two output pairs are
//! `(Extract(0, θ), addr₁), (Extract(1, θ), addr₂)`.
//!
//! Each original Dispatch e-class is unioned with the merged Dispatch.
//! Semantically: "running the merged dispatch accomplishes the
//! original's work (and more)"; the extractor then picks the merged
//! form iff it's cheaper than running both originals.
//!
//! Why post-lowering:
//! - `count`/`init`/`update` are concrete e-graph nodes so matching
//!   doesn't require re-deriving address math from HighLevel.
//! - Independence is trivial by construction — `reduce_lowering`
//!   never introduces cross-reduce references in a `Theta`'s body, so
//!   we don't need the BFS subtree check a HighLevel merge would.
//! - Compatible with `exp_algebra` equivalences firing first at
//!   HighLevel: those rewrites compose to normalize `Σ exp(x -
//!   bcast(max))` into `(Σ exp(x)) / exp(max)`, after which the two
//!   independent reductions lower separately and this rule merges
//!   them.
//!
//! Today the rule handles exactly pair-merge; extending to N-way is a
//! follow-up iteration (successive pair-merges compose, so N-way
//! isn't strictly needed, just cheaper).

use egg::{EGraph, Id, Rewrite};

use crate::analysis::TensorAnalysis;
use crate::applier::SimpleEclassSearcher;
use crate::language::{
    DispatchNode, SimdNode, TensorIr, add_list, extract_list, try_add_value_addr_dispatch,
};
use crate::rules::RunnerConfig;
pub(super) fn build(_config: &RunnerConfig) -> Rewrite<TensorIr, TensorAnalysis> {
    Rewrite::new(
        "theta-merge-reduction",
        SimpleEclassSearcher::new(|egraph, eclass| {
            egraph[eclass]
                .iter()
                .any(|node| dispatch_is_merge_candidate(egraph, node))
        }),
        crate::applier::AdaptedApplier(ThetaMergeApplier),
    )
    .unwrap()
}

/// A Dispatch is a merge candidate when it carries a scalar-init
/// `Theta { role: Reduction }` as its single output value. The
/// partner search then looks for another such Dispatch with matching
/// shape fields elsewhere in the e-graph.
fn dispatch_is_merge_candidate(egraph: &EGraph<TensorIr, TensorAnalysis>, node: &TensorIr) -> bool {
    let Some(d) = DispatchView::from(node, egraph) else {
        return false;
    };
    // Only pair-merge scalar dispatches: exactly one (value, addr) output
    // pair. Multi-output dispatches already carry a Pack value that this
    // rule doesn't know how to extend.
    if d.num_outputs != 1 {
        return false;
    }
    reduction_theta_in_eclass(egraph, d.output_pairs[0].0).is_some()
}

/// Cheap view over a Dispatch's fields.
struct DispatchView {
    workgroups: u32,
    num_inputs: u32,
    inputs: Vec<Id>,
    output_pairs: Vec<(Id, Id)>,
    num_outputs: usize,
}

impl DispatchView {
    fn from(node: &TensorIr, egraph: &EGraph<TensorIr, TensorAnalysis>) -> Option<Self> {
        let TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups,
            num_inputs,
            children_list,
            ..
        }) = node
        else {
            return None;
        };
        let children = extract_list(egraph, *children_list);
        // Children layout is `[inputs (num_inputs), output_pairs
        // (2*num_outputs)]`. Derive num_outputs structurally.
        let body_len = children.len().saturating_sub(*num_inputs as usize);
        if !body_len.is_multiple_of(2) {
            return None;
        }
        let num_outputs = body_len / 2;
        let inputs: Vec<Id> = children[..*num_inputs as usize].to_vec();
        let output_pairs: Vec<(Id, Id)> = (0..num_outputs)
            .map(|i| {
                let base = *num_inputs as usize + i * 2;
                (children[base], children[base + 1])
            })
            .collect();
        Some(Self {
            workgroups: *workgroups,
            num_inputs: *num_inputs,
            inputs,
            output_pairs,
            num_outputs,
        })
    }
}

/// Find a scalar, non-composite `Theta` node in the given eclass,
/// returning its `(init, count, update)` children. Candidate shape is:
/// - `init` is a scalar value (analysis flag `contains_pack == false`:
///   not already a running-reduction tuple),
/// - `update` has no nested Theta (`contains_theta == false`: not a
///   composite-body reduction whose iteration space we'd duplicate).
///
/// Both predicates are O(1) lookups on analysis data that propagates
/// bottom-up.
fn reduction_theta_in_eclass(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    eclass: Id,
) -> Option<(Id, Id, Id)> {
    let canonical = egraph.find(eclass);
    egraph[canonical].iter().find_map(|node| {
        let TensorIr::Simd(SimdNode::Theta {
            children: [init, count, update],
        }) = node
        else {
            return None;
        };
        if egraph[*init].data.contains_pack {
            return None;
        }
        if egraph[*update].data.contains_theta {
            return None;
        }
        Some((*init, *count, *update))
    })
}

/// Canonical-id equality across the e-graph.
fn classes_match(egraph: &EGraph<TensorIr, TensorAnalysis>, a: Id, b: Id) -> bool {
    egraph.find(a) == egraph.find(b)
}

fn input_lists_match(egraph: &EGraph<TensorIr, TensorAnalysis>, a: &[Id], b: &[Id]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| classes_match(egraph, *x, *y))
}

struct ThetaMergeApplier;

impl crate::applier::TypedApplier for ThetaMergeApplier {
    fn apply(&self, egraph: &mut EGraph<TensorIr, TensorAnalysis>, eclass: Id) -> Vec<Id> {
        // Pull the primary Dispatch's fields.
        let Some((primary_view, primary_theta)) = primary_fields(egraph, eclass) else {
            return vec![];
        };

        // Search the e-graph for a partner Dispatch in a different e-class.
        let primary_canonical = egraph.find(eclass);
        let partner = find_partner(egraph, primary_canonical, &primary_view);
        let Some((partner_eclass, partner_view, partner_theta)) = partner else {
            return vec![];
        };

        // Build the merged form.
        let Some(merged_dispatch) = build_merged_dispatch(
            egraph,
            &primary_view,
            primary_theta,
            &partner_view,
            partner_theta,
        ) else {
            return vec![];
        };

        egraph.union(eclass, merged_dispatch);
        egraph.union(partner_eclass, merged_dispatch);
        vec![merged_dispatch]
    }
}

/// Pull the primary's Dispatch fields + its `Theta { Reduction }`'s
/// `(init, count, update)`. Returns `None` if the e-class doesn't hold a
/// merge-candidate Dispatch (shouldn't normally happen since the
/// searcher only matches candidates, but belt-and-suspenders).
fn primary_fields(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    eclass: Id,
) -> Option<(DispatchView, (Id, Id, Id))> {
    let canonical = egraph.find(eclass);
    for node in egraph[canonical].iter() {
        let Some(view) = DispatchView::from(node, egraph) else {
            continue;
        };
        if view.num_outputs != 1 {
            continue;
        }
        if let Some(theta) = reduction_theta_in_eclass(egraph, view.output_pairs[0].0) {
            return Some((view, theta));
        }
    }
    None
}

/// Scan the e-graph for another e-class containing a compatible
/// single-reduction Dispatch. Returns the partner eclass id, its view,
/// and its theta children. Fires only when `count` matches structurally
/// — same canonical e-class id.
fn find_partner(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    primary_canonical: Id,
    primary: &DispatchView,
) -> Option<(Id, DispatchView, (Id, Id, Id))> {
    let primary_count = reduction_theta_in_eclass(egraph, primary.output_pairs[0].0)?.1;
    for eclass_id in egraph.classes().map(|c| c.id) {
        if egraph.find(eclass_id) == primary_canonical {
            continue;
        }
        for node in egraph[eclass_id].iter() {
            let Some(view) = DispatchView::from(node, egraph) else {
                continue;
            };
            if view.num_outputs != 1
                || view.workgroups != primary.workgroups
                || view.num_inputs != primary.num_inputs
            {
                continue;
            }
            if !input_lists_match(egraph, &primary.inputs, &view.inputs) {
                continue;
            }
            let Some(partner_theta) = reduction_theta_in_eclass(egraph, view.output_pairs[0].0)
            else {
                continue;
            };
            if !classes_match(egraph, partner_theta.1, primary_count) {
                continue;
            }
            return Some((egraph.find(eclass_id), view, partner_theta));
        }
    }
    None
}

fn build_merged_dispatch(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    primary: &DispatchView,
    (init1, count1, update1): (Id, Id, Id),
    partner: &DispatchView,
    (init2, _count2, update2): (Id, Id, Id),
) -> Option<Id> {
    // Pack(init1, init2) / Pack(update1, update2).
    let init_list = add_list(egraph, &[init1, init2]);
    let init_pack = egraph.add(TensorIr::Dispatch(DispatchNode::Pack {
        children_list: init_list,
    }));
    let update_list = add_list(egraph, &[update1, update2]);
    let update_pack = egraph.add(TensorIr::Dispatch(DispatchNode::Pack {
        children_list: update_list,
    }));

    let merged_theta = egraph.add(TensorIr::Simd(SimdNode::Theta {
        children: [init_pack, count1, update_pack],
    }));

    let slot0 = egraph.add(TensorIr::Dispatch(DispatchNode::Extract {
        index: 0,
        tuple: merged_theta,
    }));
    let slot1 = egraph.add(TensorIr::Dispatch(DispatchNode::Extract {
        index: 1,
        tuple: merged_theta,
    }));

    // Dispatch children: [shared inputs..., slot0, addr0, slot1, addr1].
    let mut children = primary.inputs.clone();
    children.push(slot0);
    children.push(primary.output_pairs[0].1);
    children.push(slot1);
    children.push(partner.output_pairs[0].1);
    // Output arity = 2 is encoded structurally: two (value, addr) pairs
    // after the input list. Readers that need the count derive it from
    // `(children.len() - num_inputs) / 2`.
    try_add_value_addr_dispatch(egraph, primary.workgroups, primary.num_inputs, &children)
}
