//! Beam search extractor with synthetic cost model.
//!
//! Replaces egg's built-in greedy/ILP extraction with a beam search
//! that keeps the top-k candidate programs at each expansion step.

use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Language, RecExpr};

use crate::analysis::TensorAnalysis;
use crate::language::{DispatchNode, HighLevelNode, SimdNode, TensorIr};
use crate::types::{BinderKind, DeviceProfile, Dim, MemTier, ScalarValue, VarRef};

fn dim_cost_value(dim: &Dim) -> f64 {
    f64::from(dim.as_const().unwrap_or(1024))
}

/// Synthetic cost model for scoring candidate programs.
#[derive(Debug, Clone)]
pub struct SyntheticCostModel {
    /// Cost of a device memory load.
    pub device_load_cost: f64,
    /// Cost of a threadgroup memory load.
    pub threadgroup_load_cost: f64,
    /// Cost of an arithmetic operation.
    pub arithmetic_cost: f64,
    /// Cost of a shuffle operation.
    pub shuffle_cost: f64,
    /// Fixed penalty per barrier.
    pub barrier_cost: f64,
    /// Quadratic penalty as estimated threadgroup memory usage approaches
    /// `DeviceProfile::max_threadgroup_bytes`. Currently applied indirectly
    /// (favoring threadgroup loads over device loads); a precise per-Dispatch
    /// budget check follows in Stage 1 once the skeleton's tile-shape
    /// derivation is exposed via Analysis.
    pub threadgroup_pressure_penalty: f64,
    /// Quadratic penalty for `Dispatch` nodes whose register block
    /// (`reg_m * reg_n * dtype_bytes`) exceeds
    /// `DeviceProfile::max_registers_per_lane * 4` (rough bytes-per-register).
    pub register_pressure_penalty: f64,
    /// Bonus subtracted from the cost of `Load`/`Store` against threadgroup
    /// memory. This is the *positive* signal that lets a footprint-aware cost
    /// model reward kernels that actually use threadgroup memory, replacing
    /// the matmul-shape filter in `select_kernel_candidate`.
    pub threadgroup_use_bonus: f64,
}

impl Default for SyntheticCostModel {
    fn default() -> Self {
        Self {
            device_load_cost: 100.0,
            threadgroup_load_cost: 5.0,
            arithmetic_cost: 1.0,
            shuffle_cost: 2.0,
            barrier_cost: 50.0,
            threadgroup_pressure_penalty: 10.0,
            register_pressure_penalty: 5.0,
            threadgroup_use_bonus: 1.0,
        }
    }
}

impl SyntheticCostModel {
    /// Score a single e-node against a target `DeviceProfile`.
    ///
    /// `egraph` is supplied so dispatch-level nodes can consult per-eclass
    /// analysis facts (notably `dtype_bytes`) when computing register-block
    /// footprint without having to re-walk subtrees.
    #[must_use]
    pub fn node_cost(
        &self,
        node: &TensorIr,
        egraph: &EGraph<TensorIr, TensorAnalysis>,
        device: &DeviceProfile,
    ) -> f64 {
        match node {
            // High-level nodes have high cost to prefer lowered versions
            TensorIr::HighLevel(
                HighLevelNode::Elementwise { .. } | HighLevelNode::Reduce { .. },
            ) => 1000.0,

            // Scalar expressions and structural nodes.
            TensorIr::BinOp(..) | TensorIr::UnOp(..) | TensorIr::TernOp(..) => self.arithmetic_cost,

            // Dispatch-level: scale with total output coverage, plus a small
            // workgroup-count term so kernels that cover the same output
            // footprint with fewer virtual workgroups get credit, plus a
            // device-aware register-pressure penalty when the per-lane
            // register block exceeds the target occupancy budget. Output
            // arity is derived from the children-list shape instead of
            // a `reg_m * reg_n` tag, and the "composite" bias is derived
            // from whether the value subtree contains nested Thetas.
            TensorIr::Dispatch(DispatchNode::Dispatch {
                workgroups,
                num_inputs,
                children_list,
            }) => {
                let list_data = &egraph[*children_list].data;
                let children = crate::language::extract_list(egraph, *children_list);
                let num_outputs = children.len().saturating_sub(*num_inputs as usize) / 2;
                let num_outputs_u32 = u32::try_from(num_outputs).unwrap_or(1).max(1);
                let workgroups_cost = dim_cost_value(workgroups);
                let covered_outputs = workgroups_cost * f64::from(num_outputs_u32);
                // "Composite" = value subtree contains a nested Theta chain.
                // Checked via analysis data on any value child (all value
                // children share the same structural property for a
                // well-formed dispatch).
                let is_composite = children
                    .iter()
                    .skip(*num_inputs as usize)
                    .step_by(2)
                    .any(|value| egraph[*value].data.contains_theta);
                let dispatch_bias = if is_composite { 6.0 } else { 10.0 };
                let base = 0.001f64.mul_add(
                    workgroups_cost,
                    0.004f64.mul_add(covered_outputs, dispatch_bias),
                );

                // Register-pressure penalty: rough bytes-per-lane estimate.
                let dtype_bytes = list_data.dtype_bytes.unwrap_or(4).max(1);
                let reg_block_bytes =
                    u64::from(num_outputs_u32).saturating_mul(u64::from(dtype_bytes));
                let budget_bytes = u64::from(device.max_registers_per_lane).saturating_mul(4);
                let reg_penalty = if reg_block_bytes > budget_bytes && budget_bytes > 0 {
                    #[allow(clippy::cast_precision_loss)]
                    let overflow_ratio =
                        (reg_block_bytes - budget_bytes) as f64 / budget_bytes as f64;
                    self.register_pressure_penalty * overflow_ratio * overflow_ratio
                } else {
                    0.0
                };

                // Shape-aware penalties. `DispatchShapeFacts` is populated
                // by analysis and read here so the extractor compares
                // dispatch variants (plain vs cooperative vs tiled) with
                // the same cost scale. When shape is unknown (e.g. on
                // in-flight extractions before facts propagate), fall back
                // to zero so nothing shifts for unshaped dispatches.
                let shape_penalty =
                    egraph[*children_list]
                        .data
                        .dispatch_shape
                        .map_or(0.0, |facts| {
                            let mut penalty = 0.0;
                            // Cooperative dispatches pay for the shuffle tree
                            // plus the simd_width-wide workgroup multiplier,
                            // which is only worthwhile when per-lane work is
                            // large. Penalty shrinks as `tile_k` grows.
                            if facts.cooperative {
                                let k = facts.tile_k.unwrap_or(1).max(1);
                                let sw = device.simd_width.max(1);
                                if k < sw {
                                    // For small reductions the cooperative
                                    // overhead dominates; scale penalty by
                                    // unused-lane ratio.
                                    #[allow(clippy::cast_precision_loss)]
                                    let unused_ratio = f64::from(sw - k) / f64::from(sw);
                                    penalty += self.barrier_cost * unused_ratio;
                                }
                            }
                            // Threadgroup memory pressure: soft quadratic
                            // penalty as usage approaches the device budget.
                            if facts.tg_bytes > 0 && device.max_threadgroup_bytes > 0 {
                                #[allow(clippy::cast_precision_loss)]
                                let utilization = f64::from(facts.tg_bytes)
                                    / f64::from(device.max_threadgroup_bytes);
                                if utilization > 1.0 {
                                    let overflow = utilization - 1.0;
                                    penalty +=
                                        self.threadgroup_pressure_penalty * overflow * overflow;
                                }
                            }
                            penalty
                        });

                base + reg_penalty + shape_penalty
            }
            // Strongly penalize Seq to prefer fused single dispatch or a
            // Pipeline candidate over multiple independent dispatches.
            TensorIr::Dispatch(DispatchNode::Seq(_)) => 100.0,
            TensorIr::Dispatch(DispatchNode::Pipeline(_)) => 60.0,
            TensorIr::Simd(SimdNode::Load { tier, .. }) => match tier {
                MemTier::Device(_) => self.device_load_cost,
                // Bonus on top of the base cost — kernels that use TG memory
                // beat plain device-load kernels even at equal arithmetic.
                MemTier::Threadgroup(_) => self.threadgroup_load_cost - self.threadgroup_use_bonus,
            },
            TensorIr::Simd(SimdNode::Shuffle(_)) => self.shuffle_cost,
            TensorIr::Simd(SimdNode::ReduceSimd { .. }) => self.shuffle_cost * 5.0, // log2(32) shuffles
            TensorIr::Simd(SimdNode::Theta { .. }) => 0.5, // small overhead; don't over-penalize nested loops
            TensorIr::Simd(SimdNode::Store { tier, .. } | SimdNode::StoreIf { tier, .. }) => {
                match tier {
                    MemTier::Device(_) => self.device_load_cost,
                    MemTier::Threadgroup(_) => {
                        self.threadgroup_load_cost - self.threadgroup_use_bonus
                    }
                }
            }
            TensorIr::Simd(SimdNode::Barrier { .. }) => self.barrier_cost,
            TensorIr::HighLevel(HighLevelNode::Input { .. } | HighLevelNode::Restride { .. })
            | TensorIr::Const(_)
            | TensorIr::ShapeParam(_)
            | TensorIr::HighLevel(
                HighLevelNode::Param(_)
                | HighLevelNode::Index(_)
                | HighLevelNode::IndexedParam { .. },
            )
            | TensorIr::Nil
            | TensorIr::Cons(_)
            | TensorIr::Dispatch(
                DispatchNode::Token | DispatchNode::Pack { .. } | DispatchNode::Extract { .. },
            )
            | TensorIr::Simd(SimdNode::Var(_)) => 0.0,
        }
    }
}

/// Configuration for beam search extraction.
#[derive(Debug, Clone)]
pub struct BeamConfig {
    /// Number of candidates to keep at each step.
    pub beam_width: usize,
    /// Cost model for scoring.
    pub cost_model: SyntheticCostModel,
    /// Hardware/runtime parameters consulted by the cost model when
    /// evaluating dispatch-level pressure penalties.
    pub device: DeviceProfile,
}

impl Default for BeamConfig {
    fn default() -> Self {
        Self {
            beam_width: 8,
            cost_model: SyntheticCostModel::default(),
            device: DeviceProfile::default(),
        }
    }
}

/// Extract the best program from the e-graph using beam search.
///
/// # Panics
///
/// Panics if beam search produces no valid extraction candidates.
#[must_use]
pub fn beam_extract(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    config: &BeamConfig,
) -> (f64, RecExpr<TensorIr>) {
    beam_extract_candidates(egraph, root, config, 1)
        .into_iter()
        .next()
        .expect("beam search found no valid extraction")
}

/// Extract multiple candidate programs from the e-graph.
///
/// Search is guided by a demand-weighted relaxation over partial extraction
/// states. Repeated uses of the same lowered subtree therefore contribute
/// proportionally during search instead of being counted only once per
/// selected e-class. The returned candidates are complete extracted
/// programs sorted by the same demand-weighted synthetic cost.
pub fn beam_extract_candidates(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    config: &BeamConfig,
    limit: usize,
) -> Vec<(f64, RecExpr<TensorIr>)> {
    if limit == 0 {
        return Vec::new();
    }

    let root = egraph.find(root);
    let result_width = config.beam_width.max(1);
    let frontier_cap = 128usize.max(result_width);
    let frontier_width = result_width
        .saturating_mul(4)
        .clamp(result_width, frontier_cap);
    let per_class_cap = 256usize.max(frontier_width);
    let per_class_width = frontier_width
        .saturating_mul(4)
        .clamp(frontier_width, per_class_cap);
    let mut class_info = HashMap::new();
    let mut visiting = HashSet::new();
    let root_lower_bound = collect_class_info(
        egraph,
        root,
        &config.cost_model,
        &config.device,
        per_class_width,
        &mut class_info,
        &mut visiting,
    );

    if !root_lower_bound.is_finite() {
        return Vec::new();
    }

    let Some(initial_state) = summarize_state(egraph, root, &class_info, &HashMap::new()) else {
        return Vec::new();
    };
    let mut frontier = vec![BeamState {
        selections: HashMap::new(),
        pending: initial_state.pending,
        pending_demand: initial_state.pending_demand,
        total_cost: initial_state.total_cost,
    }];
    let mut results = Vec::new();
    let mut seen_exprs: HashSet<Vec<TensorIr>> = HashSet::new();

    while !frontier.is_empty() {
        frontier.sort_by(compare_states);
        frontier.truncate(frontier_width);

        let mut next_frontier: HashMap<Vec<(usize, usize)>, BeamState> = HashMap::new();

        for state in frontier {
            let Some(class_id) = select_next_class(&state.pending) else {
                if let Some(expr) =
                    build_recexpr_from_choices(egraph, root, &class_info, &state.selections)
                    && recexpr_has_valid_var_scopes(&expr)
                {
                    let signature = expr.as_ref().to_vec();
                    if seen_exprs.insert(signature) {
                        results.push((state.total_cost, expr));
                    }
                }
                continue;
            };

            let Some(info) = class_info.get(&class_id) else {
                continue;
            };

            for (choice_idx, choice) in info.choices.iter().enumerate() {
                let Some(next_state) = expand_state(
                    egraph,
                    root,
                    &class_info,
                    &state,
                    class_id,
                    choice_idx,
                    choice,
                ) else {
                    continue;
                };

                if next_state.pending.is_empty() {
                    if let Some(expr) = build_recexpr_from_choices(
                        egraph,
                        root,
                        &class_info,
                        &next_state.selections,
                    ) && recexpr_has_valid_var_scopes(&expr)
                    {
                        let signature = expr.as_ref().to_vec();
                        if seen_exprs.insert(signature) {
                            results.push((next_state.total_cost, expr));
                        }
                    }
                    continue;
                }

                insert_frontier_state(&mut next_frontier, next_state);
            }
        }

        let mut next: Vec<_> = next_frontier.into_values().collect();
        next.sort_by(compare_states);
        next.truncate(frontier_width);
        frontier = next;
    }

    results.sort_by(|a, b| cmp_f64(a.0, b.0).then_with(|| a.1.as_ref().cmp(b.1.as_ref())));
    results.truncate(limit);
    results
}

#[derive(Debug, Clone)]
struct ChoiceInfo {
    node: TensorIr,
    local_cost: f64,
    estimated_total_cost: f64,
    child_edges: Vec<(Id, f64)>,
}

#[derive(Debug, Clone)]
struct ClassInfo {
    lower_bound: f64,
    choices: Vec<ChoiceInfo>,
}

#[derive(Debug, Clone)]
struct BeamState {
    selections: HashMap<Id, usize>,
    pending: Vec<Id>,
    pending_demand: HashMap<Id, f64>,
    total_cost: f64,
}

#[derive(Debug)]
struct StateSummary {
    total_cost: f64,
    pending: Vec<Id>,
    pending_demand: HashMap<Id, f64>,
}

fn collect_class_info(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    class_id: Id,
    cost_model: &SyntheticCostModel,
    device: &DeviceProfile,
    choice_limit: usize,
    info_map: &mut HashMap<Id, ClassInfo>,
    visiting: &mut HashSet<Id>,
) -> f64 {
    let class_id = egraph.find(class_id);

    if let Some(info) = info_map.get(&class_id) {
        return info.lower_bound;
    }
    if !visiting.insert(class_id) {
        return f64::INFINITY;
    }

    let mut choices = Vec::new();
    for node in egraph[class_id].iter() {
        if has_invalid_direct_self_ref(egraph, class_id, node)
            || has_invalid_threadgroup_address(egraph, node)
        {
            continue;
        }

        let local_cost = cost_model.node_cost(node, egraph, device);
        let mut estimated_total_cost = local_cost;
        let mut valid = true;
        let child_edges = weighted_child_edges(egraph, node);

        for (child_class, multiplier) in &child_edges {
            let child_class = egraph.find(*child_class);
            if child_class == class_id {
                if matches!(node, TensorIr::Simd(SimdNode::Theta { .. })) {
                    continue;
                }
                valid = false;
                break;
            }

            let child_cost = collect_class_info(
                egraph,
                child_class,
                cost_model,
                device,
                choice_limit,
                info_map,
                visiting,
            );
            if !child_cost.is_finite() {
                valid = false;
                break;
            }
            estimated_total_cost += multiplier * child_cost;
        }

        if valid {
            choices.push(ChoiceInfo {
                node: node.clone(),
                local_cost,
                estimated_total_cost,
                child_edges,
            });
        }
    }

    visiting.remove(&class_id);

    choices.sort_by(|a, b| {
        cmp_f64(a.estimated_total_cost, b.estimated_total_cost)
            .then_with(|| cmp_f64(a.local_cost, b.local_cost))
            .then_with(|| a.node.cmp(&b.node))
    });
    choices.truncate(choice_limit.max(1));

    let lower_bound = choices
        .first()
        .map_or(f64::INFINITY, |choice| choice.estimated_total_cost);
    info_map.insert(
        class_id,
        ClassInfo {
            lower_bound,
            choices,
        },
    );
    lower_bound
}

fn has_invalid_direct_self_ref(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    class_id: Id,
    node: &TensorIr,
) -> bool {
    node.children()
        .iter()
        .any(|child| egraph.find(*child) == class_id)
}

fn has_invalid_threadgroup_address(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
) -> bool {
    let TensorIr::Simd(SimdNode::Load {
        tier: MemTier::Threadgroup(_),
        children,
    }) = node
    else {
        return false;
    };

    let addr_data = &egraph[children[0]].data;
    addr_data.var_dep.contains(&VarRef::iter(1))
}

fn expand_state(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    class_info: &HashMap<Id, ClassInfo>,
    state: &BeamState,
    class_id: Id,
    choice_idx: usize,
    choice: &ChoiceInfo,
) -> Option<BeamState> {
    let _ = class_info.get(&class_id)?;
    let mut selections = state.selections.clone();
    selections.insert(class_id, choice_idx);

    if introduces_cycle(egraph, class_info, &selections, class_id, &choice.node) {
        return None;
    }

    let summary = summarize_state(egraph, root, class_info, &selections)?;
    Some(BeamState {
        selections,
        pending: summary.pending,
        pending_demand: summary.pending_demand,
        total_cost: summary.total_cost,
    })
}

fn summarize_state(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    class_info: &HashMap<Id, ClassInfo>,
    selections: &HashMap<Id, usize>,
) -> Option<StateSummary> {
    const MAX_DEMAND: f64 = 1_000_000_000.0;

    let mut total_cost = 0.0;
    let mut pending_demand = HashMap::new();
    let mut stack: Vec<(Id, f64)> = vec![(egraph.find(root), 1.0)];

    while let Some((class_id, demand)) = stack.pop() {
        let class_id = egraph.find(class_id);
        let demand = demand.min(MAX_DEMAND);
        if demand <= 0.0 {
            continue;
        }

        let info = class_info.get(&class_id)?;
        if let Some(&choice_idx) = selections.get(&class_id) {
            let choice = info.choices.get(choice_idx)?;
            total_cost += demand * choice.local_cost;
            for (child, edge_multiplier) in &choice.child_edges {
                let next_demand = (demand * *edge_multiplier).min(MAX_DEMAND);
                if next_demand > 0.0 {
                    stack.push((egraph.find(*child), next_demand));
                }
            }
        } else {
            total_cost += demand * info.lower_bound;
            *pending_demand.entry(class_id).or_insert(0.0) += demand;
        }
    }

    let mut pending: Vec<_> = pending_demand.keys().copied().collect();
    pending.sort_by(|lhs, rhs| {
        let lhs_priority = pending_demand.get(lhs).copied().unwrap_or(0.0)
            * class_info
                .get(lhs)
                .map_or(f64::INFINITY, |info| info.lower_bound);
        let rhs_priority = pending_demand.get(rhs).copied().unwrap_or(0.0)
            * class_info
                .get(rhs)
                .map_or(f64::INFINITY, |info| info.lower_bound);
        cmp_f64(rhs_priority, lhs_priority).then_with(|| usize::from(*lhs).cmp(&usize::from(*rhs)))
    });

    Some(StateSummary {
        total_cost,
        pending,
        pending_demand,
    })
}

fn introduces_cycle(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    class_info: &HashMap<Id, ClassInfo>,
    selections: &HashMap<Id, usize>,
    class_id: Id,
    node: &TensorIr,
) -> bool {
    for child in node.children() {
        let child_class = egraph.find(*child);
        if child_class == class_id {
            return true;
        }

        if selected_reaches(egraph, class_info, selections, child_class, class_id) {
            return true;
        }
    }

    false
}

fn selected_reaches(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    class_info: &HashMap<Id, ClassInfo>,
    selections: &HashMap<Id, usize>,
    start: Id,
    target: Id,
) -> bool {
    let start = egraph.find(start);
    let target = egraph.find(target);
    if start == target {
        return true;
    }

    let mut stack = vec![start];
    let mut visited = HashSet::new();

    while let Some(current) = stack.pop() {
        let current = egraph.find(current);
        if !visited.insert(current) {
            continue;
        }
        if current == target {
            return true;
        }

        let Some(&choice_idx) = selections.get(&current) else {
            continue;
        };
        let Some(info) = class_info.get(&current) else {
            continue;
        };
        let Some(choice) = info.choices.get(choice_idx) else {
            continue;
        };

        for child in choice.node.children() {
            let child_class = egraph.find(*child);
            stack.push(child_class);
        }
    }

    false
}

fn build_recexpr_from_choices(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
    class_info: &HashMap<Id, ClassInfo>,
    selections: &HashMap<Id, usize>,
) -> Option<RecExpr<TensorIr>> {
    let mut expr = RecExpr::default();
    let mut built = HashMap::new();
    let mut visiting = HashSet::new();
    let root = egraph.find(root);
    let mut stack = vec![(root, false)];

    while let Some((class_id, expanded)) = stack.pop() {
        let class_id = egraph.find(class_id);
        if built.contains_key(&class_id) {
            continue;
        }

        if expanded {
            let choice_idx = *selections.get(&class_id)?;
            let info = class_info.get(&class_id)?;
            let mut node = info.choices.get(choice_idx)?.node.clone();
            for child in node.children_mut() {
                let child_class = egraph.find(*child);
                if child_class == class_id {
                    return None;
                }
                *child = *built.get(&child_class)?;
            }
            let new_id = expr.add(node);
            built.insert(class_id, new_id);
            visiting.remove(&class_id);
            continue;
        }

        if !visiting.insert(class_id) {
            return None;
        }

        let choice_idx = *selections.get(&class_id)?;
        let info = class_info.get(&class_id)?;
        let node = &info.choices.get(choice_idx)?.node;
        stack.push((class_id, true));
        for child in node.children().iter().rev() {
            let child_class = egraph.find(*child);
            if child_class == class_id {
                return None;
            }
            if !built.contains_key(&child_class) {
                stack.push((child_class, false));
            }
        }
    }

    built.get(&root)?;
    Some(expr)
}

fn select_next_class(pending: &[Id]) -> Option<Id> {
    pending.first().copied()
}

fn insert_frontier_state(frontier: &mut HashMap<Vec<(usize, usize)>, BeamState>, state: BeamState) {
    let signature = selection_signature(&state.selections);
    match frontier.get(&signature) {
        Some(existing)
            if compare_states(existing, &state).is_lt()
                || compare_states(existing, &state).is_eq() => {}
        _ => {
            frontier.insert(signature, state);
        }
    }
}

fn selection_signature(selections: &HashMap<Id, usize>) -> Vec<(usize, usize)> {
    let mut entries: Vec<_> = selections
        .iter()
        .map(|(class_id, choice_idx)| (usize::from(*class_id), *choice_idx))
        .collect();
    entries.sort_unstable();
    entries
}

fn compare_states(lhs: &BeamState, rhs: &BeamState) -> std::cmp::Ordering {
    cmp_f64(lhs.total_cost, rhs.total_cost)
        .then_with(|| lhs.pending.len().cmp(&rhs.pending.len()))
        .then_with(|| cmp_f64(sum_pending_demand(lhs), sum_pending_demand(rhs)))
        .then_with(|| {
            selection_signature(&lhs.selections).cmp(&selection_signature(&rhs.selections))
        })
}

fn sum_pending_demand(state: &BeamState) -> f64 {
    state.pending_demand.values().copied().sum()
}

fn cmp_f64(lhs: f64, rhs: f64) -> std::cmp::Ordering {
    lhs.partial_cmp(&rhs).unwrap_or(std::cmp::Ordering::Equal)
}

fn weighted_child_edges(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    node: &TensorIr,
) -> Vec<(Id, f64)> {
    const STRUCTURAL_EDGE_WEIGHT: f64 = 1.0e-6;

    match node {
        TensorIr::HighLevel(HighLevelNode::Elementwise { children_list, .. })
        | TensorIr::Dispatch(DispatchNode::Pack { children_list }) => {
            let mut edges = vec![(*children_list, STRUCTURAL_EDGE_WEIGHT)];
            edges.extend(
                crate::language::extract_list(egraph, *children_list)
                    .into_iter()
                    .map(|child| (child, 1.0)),
            );
            edges
        }
        TensorIr::Dispatch(DispatchNode::Seq(list_id) | DispatchNode::Pipeline(list_id)) => {
            let mut edges = vec![(*list_id, STRUCTURAL_EDGE_WEIGHT)];
            edges.extend(
                crate::language::extract_list(egraph, *list_id)
                    .into_iter()
                    .map(|child| (child, 1.0)),
            );
            edges
        }
        TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups,
            num_inputs,
            children_list,
        }) => {
            let children = crate::language::extract_list(egraph, *children_list);
            let num_inputs_usize = usize::try_from(*num_inputs).expect("dispatch input count fits");
            let input_count = num_inputs_usize.min(children.len());
            // Output pairs laid out as (value, addr) after inputs. Each
            // dispatch invocation produces `num_outputs` values, which is
            // the per-lane register block size.
            let num_outputs = children.len().saturating_sub(input_count) / 2;
            let num_outputs_u32 = u32::try_from(num_outputs).unwrap_or(1).max(1);
            let body_multiplier = dim_cost_value(workgroups) * f64::from(num_outputs_u32);

            let mut edges = Vec::with_capacity(children.len() + 1);
            edges.push((*children_list, STRUCTURAL_EDGE_WEIGHT));
            for child in children.iter().take(input_count) {
                edges.push((*child, 1.0));
            }
            for child in &children[input_count..] {
                edges.push((*child, body_multiplier));
            }
            edges
        }
        TensorIr::Simd(SimdNode::Theta {
            children: [init, count, update],
            ..
        }) => {
            let trip_count = match &egraph[*count].data.constant {
                Some(ScalarValue::U32(v)) => *v,
                Some(ScalarValue::I32(v)) if *v > 0 => (*v).cast_unsigned(),
                _ => 1,
            };
            vec![
                (*init, 1.0),
                (*count, 1.0),
                (*update, f64::from(trip_count.max(1))),
            ]
        }
        _ => node.children().iter().map(|child| (*child, 1.0)).collect(),
    }
}

fn recexpr_has_valid_var_scopes(expr: &RecExpr<TensorIr>) -> bool {
    let nodes = expr.as_ref();
    if nodes.is_empty() {
        return true;
    }

    for (idx, node) in nodes.iter().enumerate() {
        if node
            .children()
            .iter()
            .any(|child| usize::from(*child) >= idx)
        {
            return false;
        }
    }

    // Walk top-down tracking the current Theta nesting depth. Every
    // Theta-bound `VarRef::Bound { kind: Theta, depth, .. }` must satisfy
    // `depth < binder_depth`. Dispatch-bound refs are kernel-scope and always
    // considered valid here.
    let mut stack: Vec<(usize, u32)> = vec![(nodes.len() - 1, 0)];
    while let Some((idx, binder_depth)) = stack.pop() {
        match &nodes[idx] {
            TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Theta,
                depth,
                ..
            })) => {
                if *depth >= binder_depth {
                    return false;
                }
            }
            TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Dispatch,
                ..
            })) => {}
            TensorIr::Simd(SimdNode::Theta {
                children: [init, count, update],
                ..
            }) => {
                stack.push((usize::from(*init), binder_depth));
                stack.push((usize::from(*count), binder_depth));
                stack.push((usize::from(*update), binder_depth + 1));
            }
            node => {
                for child in node.children() {
                    stack.push((usize::from(*child), binder_depth));
                }
            }
        }
    }

    true
}

/// Simple greedy extractor for cases where beam search is overkill.
#[must_use]
pub fn greedy_extract(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    root: Id,
) -> (f64, RecExpr<TensorIr>) {
    beam_extract(
        egraph,
        root,
        &BeamConfig {
            beam_width: 1,
            ..Default::default()
        },
    )
}
