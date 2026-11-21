use crate::interpreter::{BindingMemory, Limits, MemoryValue, Simulator};
use crate::trace::ScheduleTrace;
use crate::{FAILURE_VERSION, FailureRecord};
use fusor_cost::{LocalSearch, Roofline, facts};
use fusor_ir::carrier::Carrier;
use fusor_ir::device::{Caps, DeviceKind, Limits as DeviceLimits};
use fusor_ir::dtype::{Dtype, Splat};
use fusor_ir::egraph::{EGraph, Saturate, SaturationBudget};
use fusor_ir::extract::{ExtractBudget, Extractor, Plan};
use fusor_ir::ir::Op;
use fusor_ir::ir::kernel::KernelIr;
use fusor_ir::ir::launch::SchedPoint;
use fusor_ir::ir::logical::{BufferId, EinSpec, Label, LeafKind, Logical};
use fusor_ir::scalar::{BinOp, ScalarExpr};
use fusor_ir::shape::{BoundsProof, Dim, StrideSpec};
use fusor_ir::target::LowerCtx;
use fusor_ir::{CORE_RULES, CoreSemantics, Driver};
use fusor_tile::Planner;
use serde::{Deserialize, Serialize};
use smallvec::smallvec;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::Arc;

const CAPS_PROFILE: &str = "gpu-baseline-no-subgroup-coop-f16-v1";
const MAX_PLANS_PER_CASE: usize = 64;
const MAX_EXECUTED_PLANS_PER_CASE: usize = 16;
const MAX_RECIPE_DIMENSION: u32 = 65;
const MAX_RECIPE_ELEMENTS: u64 = 8192;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum RecipeKind {
    TailMap,
    Sum,
    Matmul,
    MapChain,
    TransposeMap,
    MatmulRowSum,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Recipe {
    pub kind: RecipeKind,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub depth: u32,
    pub input_seed: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateSelection {
    pub launch: usize,
    pub label: String,
}

#[derive(Clone, Debug, Default)]
pub struct CampaignResult {
    pub cases: u64,
    pub distinct_shapes: u64,
    pub enumerated_candidates: u64,
    pub executed_plans: u64,
    pub unsupported_plans: u64,
    pub execution_errors: u64,
    pub candidate_errors: u64,
    pub dispatches: u64,
    pub steps: u64,
    pub first_failure: Option<FailureRecord>,
}

pub fn campaign(seed: u64, iterations: u32) -> Result<CampaignResult, String> {
    campaign_with_limits(seed, iterations, Limits::default())
}

pub fn campaign_with_limits(
    seed: u64,
    iterations: u32,
    limits: Limits,
) -> Result<CampaignResult, String> {
    campaign_with_fault_hook(seed, iterations, limits, no_output_fault)
}

type OutputFault = fn(&mut [f32]);

fn no_output_fault(_: &mut [f32]) {}

#[cfg(test)]
fn injected_output_discrepancy(output: &mut [f32]) {
    if let Some(value) = output.first_mut() {
        *value += 1.0;
    }
}

fn campaign_with_fault_hook(
    seed: u64,
    iterations: u32,
    limits: Limits,
    output_fault: OutputFault,
) -> Result<CampaignResult, String> {
    if iterations == 0 {
        return Err("iterations must be greater than zero".into());
    }
    let mut total = CampaignResult::default();
    let mut shapes = HashSet::new();
    for iteration in 0..iterations {
        let recipe = Recipe::generated(seed, iteration);
        shapes.insert(shape_identity(&recipe));
        total.distinct_shapes = shapes.len() as u64;
        let inputs = generate_inputs(&recipe);
        let prepared = match prepare(&recipe) {
            Ok(prepared) => prepared,
            Err(error) => {
                total.cases += 1;
                total.first_failure = Some(failure_record(
                    "planning",
                    recipe,
                    inputs,
                    "base".into(),
                    Vec::new(),
                    Vec::new(),
                    error,
                ));
                break;
            }
        };
        total.cases += 1;
        let reference = reference(&recipe, &inputs);
        let mut queue = VecDeque::from([PlanWork {
            plan: prepared.plan.clone(),
            selections: Vec::new(),
            next_launch: 0,
            candidate: "base".into(),
        }]);
        let mut visited = HashSet::new();
        let mut executed_plans = 0usize;
        while let Some(work) = queue.pop_front() {
            if !visited.insert(work.plan.hash) {
                continue;
            }
            if executed_plans >= MAX_EXECUTED_PLANS_PER_CASE {
                break;
            }
            executed_plans += 1;
            total.executed_plans += 1;
            let execution = execute_plan_with_limits(
                &prepared.graph,
                &prepared.roots,
                &prepared.caps,
                &work.plan,
                &inputs,
                limits,
                seed ^ u64::from(iteration).rotate_left(17) ^ work.plan.hash.0 as u64,
                None,
                output_fault,
            );
            total.dispatches += execution.dispatches;
            total.steps += execution.steps;
            match execution.result {
                Ok(actual) => {
                    if let Some(detail) = compare_results(&actual, &reference) {
                        total.first_failure = Some(failure_record(
                            "differential",
                            recipe.clone(),
                            inputs.clone(),
                            work.candidate.clone(),
                            work.selections.clone(),
                            execution.traces,
                            detail,
                        ));
                        break;
                    }
                }
                Err((error, traces)) => {
                    let kind = classify_sim_error(&error);
                    total.execution_errors += 1;
                    if kind == "unsupported-ir" {
                        total.unsupported_plans += 1;
                    }
                    total.first_failure = Some(failure_record(
                        &kind,
                        recipe.clone(),
                        inputs.clone(),
                        work.candidate.clone(),
                        work.selections.clone(),
                        traces,
                        error,
                    ));
                    break;
                }
            }
            if total.first_failure.is_some() {
                break;
            }

            for launch in work.next_launch..work.plan.launches.len() {
                let results = prepared.search.test_launch_variant_results(
                    &prepared.graph,
                    &prepared.roots,
                    &work.plan,
                    launch,
                    &prepared.cost,
                );
                for (label, candidate_result) in results {
                    total.enumerated_candidates += 1;
                    let mut selections = work.selections.clone();
                    selections.push(CandidateSelection {
                        launch,
                        label: label.clone(),
                    });
                    match candidate_result {
                        Ok(plan) => {
                            if queue.len() < MAX_PLANS_PER_CASE {
                                queue.push_back(PlanWork {
                                    plan,
                                    selections,
                                    next_launch: launch + 1,
                                    candidate: format!("{} -> {}", work.candidate, label),
                                });
                            }
                        }
                        Err(error) => {
                            total.candidate_errors += 1;
                            total.first_failure = Some(failure_record(
                                "candidate-replan",
                                recipe.clone(),
                                inputs.clone(),
                                format!("{} -> {}", work.candidate, label),
                                selections,
                                Vec::new(),
                                error.to_string(),
                            ));
                            break;
                        }
                    }
                }
                if total.first_failure.is_some() {
                    break;
                }
            }
            if total.first_failure.is_some() {
                break;
            }
        }
        if total.first_failure.is_some() {
            break;
        }
    }
    if let Some(failure) = &mut total.first_failure {
        failure.limits = limits;
    }
    validate_campaign_coverage(total)
}

fn validate_campaign_coverage(total: CampaignResult) -> Result<CampaignResult, String> {
    if total.first_failure.is_none()
        && (total.cases == 0
            || total.enumerated_candidates == 0
            || total.executed_plans == 0
            || total.dispatches == 0
            || total.steps == 0)
    {
        return Err(format!(
            "coverage incomplete: cases={}, enumerated_candidates={}, executed_plans={}, dispatches={}, steps={}",
            total.cases,
            total.enumerated_candidates,
            total.executed_plans,
            total.dispatches,
            total.steps
        ));
    }
    Ok(total)
}

pub fn replay_failure(record: &FailureRecord) -> Result<String, String> {
    replay_failure_with_fault(record, no_output_fault)
}

fn replay_failure_with_fault(
    record: &FailureRecord,
    output_fault: OutputFault,
) -> Result<String, String> {
    if record.version != FAILURE_VERSION {
        return Err(format!(
            "unsupported failure record version {}",
            record.version
        ));
    }
    if record.caps_profile != CAPS_PROFILE {
        return Err(format!("caps profile mismatch: {}", record.caps_profile));
    }
    validate_inputs(&record.recipe, &record.inputs)?;
    if record.kind == "candidate-replan" {
        if !record.trace.is_empty() {
            return Err("candidate failure trace has trailing entries".into());
        }
        if record.selections.is_empty() {
            return Err("candidate failure has no selected candidate".into());
        }
    }
    if record.kind == "planning" && (!record.trace.is_empty() || !record.selections.is_empty()) {
        return Err("planning failure has unused selections or traces".into());
    }
    let mut prepared = match replay_preparation(record, prepare(&record.recipe))? {
        ReplayPreparation::Ready(prepared) => prepared,
        ReplayPreparation::PlanningFailure(error) => {
            return Ok(format!("reproduced planning failure: {error}"));
        }
    };
    let (plan, candidate_error) = apply_selections(&mut prepared, &record.selections)?;
    if record.kind == "candidate-replan" {
        return reproduce_candidate_failure(record, candidate_error);
    }
    if candidate_error.is_some() {
        return Err("recorded candidate selection now fails during replan".into());
    }
    let plan = plan.ok_or_else(|| "replay selection did not produce a plan".to_owned())?;
    let execution = execute_plan_with_limits(
        &prepared.graph,
        &prepared.roots,
        &prepared.caps,
        &plan,
        &record.inputs,
        record.limits,
        0,
        Some(&record.trace),
        output_fault,
    );
    match execution.result {
        Err((error, _)) => {
            let kind = classify_sim_error(&error);
            if kind != record.kind {
                return Err(format!(
                    "failure category changed: expected {}, got {kind}: {error}",
                    record.kind
                ));
            }
            Ok(format!("reproduced {kind}: {error}"))
        }
        Ok(actual) => {
            let reference = reference(&record.recipe, &record.inputs);
            if let Some(detail) = compare_results(&actual, &reference) {
                if record.kind == "differential" {
                    Ok(format!("reproduced differential failure: {detail}"))
                } else {
                    Err(format!(
                        "failure did not reproduce; differential check: {detail}"
                    ))
                }
            } else {
                Err("recorded failure no longer reproduces".into())
            }
        }
    }
}

fn reproduce_candidate_failure(
    record: &FailureRecord,
    candidate_error: Option<String>,
) -> Result<String, String> {
    match candidate_error {
        Some(error) if error == record.detail => {
            Ok(format!("reproduced candidate replan failure: {error}"))
        }
        Some(error) => Err(format!(
            "candidate replan failure changed: expected {}, got {error}",
            record.detail
        )),
        None => Err("recorded candidate replan failure did not reproduce".into()),
    }
}

enum ReplayPreparation {
    Ready(Box<Prepared>),
    PlanningFailure(String),
}

fn replay_preparation(
    record: &FailureRecord,
    prepared: Result<Prepared, String>,
) -> Result<ReplayPreparation, String> {
    match prepared {
        Ok(_) if record.kind == "planning" => {
            Err("recorded planning failure no longer reproduces".into())
        }
        Ok(prepared) => Ok(ReplayPreparation::Ready(Box::new(prepared))),
        Err(error) if record.kind == "planning" && error == record.detail => {
            Ok(ReplayPreparation::PlanningFailure(error))
        }
        Err(error) if record.kind == "planning" => Err(format!(
            "planning failure changed: expected {}, got {error}",
            record.detail
        )),
        Err(error) => Err(format!(
            "unexpected planning failure during replay: {error}"
        )),
    }
}

pub fn minimize_failure(record: &FailureRecord) -> Result<FailureRecord, String> {
    minimize_failure_with_fault(record, no_output_fault)
}

fn minimize_failure_with_fault(
    record: &FailureRecord,
    output_fault: OutputFault,
) -> Result<FailureRecord, String> {
    replay_failure_with_fault(record, output_fault)?;
    let current = minimize_failure_search(record, |recipe| {
        first_failure_for_recipe(recipe, record.limits, output_fault).map(|mut failure| {
            failure.limits = record.limits;
            failure
        })
    });
    replay_failure_with_fault(&current, output_fault)?;
    Ok(current)
}

fn minimize_failure_search(
    record: &FailureRecord,
    mut failure_for_recipe: impl FnMut(&Recipe) -> Option<FailureRecord>,
) -> FailureRecord {
    let mut current = record.clone();
    loop {
        let mut improved = false;
        for candidate in current.recipe.shrinks() {
            let Some(failure) = failure_for_recipe(&candidate) else {
                continue;
            };
            if same_failure_fingerprint(&failure, record)
                && recipe_complexity(&failure.recipe) < recipe_complexity(&current.recipe)
            {
                current = failure;
                improved = true;
                break;
            }
        }
        if !improved {
            break;
        }
    }
    current
}

fn same_failure_fingerprint(candidate: &FailureRecord, original: &FailureRecord) -> bool {
    if candidate.kind != original.kind
        || candidate.recipe.kind != original.recipe.kind
        || candidate.candidate != original.candidate
        || candidate.selections != original.selections
    {
        return false;
    }
    match candidate.kind.as_str() {
        "differential" => true,
        "planning" | "candidate-replan" => candidate.detail == original.detail,
        "resource-limit" => same_fingerprint(
            limit_fingerprint(&candidate.detail),
            limit_fingerprint(&original.detail),
        ),
        "data-race" => {
            same_fingerprint(race_space(&candidate.detail), race_space(&original.detail))
        }
        "out-of-bounds" | "uninitialized-read" => same_fingerprint(
            error_space(&candidate.detail),
            error_space(&original.detail),
        ),
        _ => candidate.detail == original.detail,
    }
}

fn same_fingerprint<T: PartialEq>(candidate: Option<T>, original: Option<T>) -> bool {
    candidate.is_some() && candidate == original
}

fn limit_fingerprint(detail: &str) -> Option<&'static str> {
    if detail.starts_with("simulator step limit") {
        Some("steps")
    } else if detail.starts_with("simulator memory ") {
        Some("memory")
    } else if detail.starts_with("simulator invocations ") {
        Some("invocations")
    } else {
        None
    }
}

fn race_space(detail: &str) -> Option<&str> {
    let start = detail.find("unordered ")? + "unordered ".len();
    let end = detail[start..].find(" conflict")? + start;
    Some(&detail[start..end])
}

fn error_space(detail: &str) -> Option<&str> {
    if let Some(rest) = detail.strip_prefix("uninitialized ") {
        return rest.split(" read at index").next();
    }
    detail.split(" index ").next()
}

fn recipe_complexity(recipe: &Recipe) -> u64 {
    u64::from(recipe.m) + u64::from(recipe.n) + u64::from(recipe.k) + u64::from(recipe.depth)
}

fn shape_identity(recipe: &Recipe) -> (RecipeKind, u32, u32, u32) {
    match recipe.kind {
        RecipeKind::TailMap | RecipeKind::Sum => (recipe.kind, recipe.m, 0, 0),
        RecipeKind::MapChain => (recipe.kind, recipe.m, recipe.depth, 0),
        RecipeKind::TransposeMap => (recipe.kind, recipe.m, recipe.n, 0),
        RecipeKind::Matmul | RecipeKind::MatmulRowSum => {
            (recipe.kind, recipe.m, recipe.n, recipe.k)
        }
    }
}

fn first_failure_for_recipe(
    recipe: &Recipe,
    limits: Limits,
    output_fault: OutputFault,
) -> Option<FailureRecord> {
    let inputs = generate_inputs(recipe);
    let prepared = match prepare(recipe) {
        Ok(prepared) => prepared,
        Err(error) => {
            return Some(failure_record(
                "planning",
                recipe.clone(),
                inputs,
                "base".into(),
                Vec::new(),
                Vec::new(),
                error,
            ));
        }
    };
    let expected = reference(recipe, &inputs);
    let base = execute_plan_with_limits(
        &prepared.graph,
        &prepared.roots,
        &prepared.caps,
        &prepared.plan,
        &inputs,
        limits,
        recipe.input_seed,
        None,
        output_fault,
    );
    match base.result {
        Err((error, traces)) => {
            return Some(failure_record(
                &classify_sim_error(&error),
                recipe.clone(),
                inputs,
                "base".into(),
                Vec::new(),
                traces,
                error,
            ));
        }
        Ok(actual) if compare_results(&actual, &expected).is_some() => {
            return Some(failure_record(
                "differential",
                recipe.clone(),
                inputs,
                "base".into(),
                Vec::new(),
                base.traces,
                compare_results(&actual, &expected).unwrap(),
            ));
        }
        Ok(_) => {}
    }
    for launch in 0..prepared.plan.launches.len() {
        for (label, result) in prepared.search.test_launch_variant_results(
            &prepared.graph,
            &prepared.roots,
            &prepared.plan,
            launch,
            &prepared.cost,
        ) {
            let selection = vec![CandidateSelection {
                launch,
                label: label.clone(),
            }];
            match result {
                Err(error) => {
                    return Some(failure_record(
                        "candidate-replan",
                        recipe.clone(),
                        inputs,
                        label,
                        selection,
                        Vec::new(),
                        error.to_string(),
                    ));
                }
                Ok(plan) => {
                    let execution = execute_plan_with_limits(
                        &prepared.graph,
                        &prepared.roots,
                        &prepared.caps,
                        &plan,
                        &inputs,
                        limits,
                        recipe.input_seed,
                        None,
                        output_fault,
                    );
                    match execution.result {
                        Err((error, traces)) => {
                            return Some(failure_record(
                                &classify_sim_error(&error),
                                recipe.clone(),
                                inputs,
                                label,
                                selection,
                                traces,
                                error,
                            ));
                        }
                        Ok(actual) => {
                            if let Some(detail) = compare_results(&actual, &expected) {
                                return Some(failure_record(
                                    "differential",
                                    recipe.clone(),
                                    inputs,
                                    label,
                                    selection,
                                    execution.traces,
                                    detail,
                                ));
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

impl Recipe {
    fn generated(seed: u64, iteration: u32) -> Self {
        let mut rng =
            CampaignRng::new(seed ^ u64::from(iteration).wrapping_mul(0x9e3779b97f4a7c15));
        let odd = |value: u64| (3 + (value % 5) * 2) as u32;
        let kind = match iteration % 6 {
            0 => RecipeKind::TailMap,
            1 => RecipeKind::Sum,
            2 => RecipeKind::Matmul,
            3 => RecipeKind::MapChain,
            4 => RecipeKind::TransposeMap,
            5 => RecipeKind::MatmulRowSum,
            _ => unreachable!(),
        };
        let m = if matches!(kind, RecipeKind::TailMap | RecipeKind::Sum) {
            [31, 32, 33, 63, 64, 65][((iteration / 6) % 6) as usize]
        } else {
            odd(rng.next())
        };
        Self {
            kind,
            m,
            n: odd(rng.next()),
            k: odd(rng.next()),
            depth: 2 + (rng.next() % 3) as u32,
            input_seed: rng.next(),
        }
    }

    fn input_lengths(&self) -> Vec<usize> {
        match self.kind {
            RecipeKind::TailMap | RecipeKind::Sum | RecipeKind::MapChain => vec![self.m as usize],
            RecipeKind::TransposeMap => vec![(self.m * self.n) as usize],
            RecipeKind::Matmul | RecipeKind::MatmulRowSum => {
                vec![(self.m * self.k) as usize, (self.k * self.n) as usize]
            }
        }
    }

    fn shrinks(&self) -> Vec<Self> {
        let mut shrinks = Vec::new();
        let mut shrink = self.clone();
        if shrink.m > 1 {
            shrink.m = (shrink.m / 2).max(1);
            shrinks.push(shrink);
        }
        let mut shrink = self.clone();
        if shrink.n > 1 {
            shrink.n = (shrink.n / 2).max(1);
            shrinks.push(shrink);
        }
        let mut shrink = self.clone();
        if shrink.k > 1 {
            shrink.k = (shrink.k / 2).max(1);
            shrinks.push(shrink);
        }
        let mut shrink = self.clone();
        if shrink.depth > 1 {
            shrink.depth -= 1;
            shrinks.push(shrink);
        }
        shrinks
    }
}

struct Prepared {
    graph: EGraph,
    roots: Vec<fusor_ir::egraph::Id>,
    plan: Plan,
    caps: Caps,
    search: LocalSearch,
    cost: Roofline,
}

struct PlanWork {
    plan: Plan,
    selections: Vec<CandidateSelection>,
    next_launch: usize,
    candidate: String,
}

fn prepare(recipe: &Recipe) -> Result<Prepared, String> {
    validate_recipe(recipe)?;
    let caps = baseline_caps();
    let planner = Arc::new(Planner::new());
    let mut graph = EGraph::new(CoreSemantics::new(planner.clone()));
    let mut dependency_roots = Vec::new();
    let root = match recipe.kind {
        RecipeKind::TailMap | RecipeKind::MapChain | RecipeKind::Sum => {
            let shape = [Dim::Const(u64::from(recipe.m))];
            let input = add_buffer(&mut graph, 0, &shape)?;
            match recipe.kind {
                RecipeKind::TailMap => add_map(&mut graph, input, false)?,
                RecipeKind::MapChain => {
                    let mut value = input;
                    for _ in 0..recipe.depth {
                        value = add_map(&mut graph, value, true)?;
                    }
                    value
                }
                RecipeKind::Sum => add_sum(&mut graph, input)?,
                _ => unreachable!(),
            }
        }
        RecipeKind::TransposeMap => {
            let shape = [
                Dim::Const(u64::from(recipe.m)),
                Dim::Const(u64::from(recipe.n)),
            ];
            let input = add_buffer(&mut graph, 0, &shape)?;
            let transposed = graph
                .add(Op::Logical(Logical::Restride {
                    specs: smallvec![
                        StrideSpec::dim(1, Dim::Const(u64::from(recipe.n))),
                        StrideSpec::dim(0, Dim::Const(u64::from(recipe.m))),
                    ],
                    bounds: BoundsProof::Static,
                    x: input,
                }))
                .map_err(|error| error.to_string())?;
            add_map(&mut graph, transposed, false)?
        }
        RecipeKind::Matmul | RecipeKind::MatmulRowSum => {
            let contraction = add_matmul(&mut graph, recipe)?;
            if recipe.kind == RecipeKind::Matmul {
                contraction
            } else {
                dependency_roots.push(contraction);
                add_sum_axis(&mut graph, contraction, 0)?
            }
        }
    };
    graph.add_root(root);
    for dependency in &dependency_roots {
        graph.add_root(*dependency);
    }
    let mut rules = Vec::new();
    rules.extend_from_slice(CORE_RULES);
    rules.extend_from_slice(fusor_tile::SCHED_RULES);
    rules.extend_from_slice(fusor_gpu::GPU_RULES);
    let report = Driver::new()
        .saturate(&mut graph, &caps, &rules, SaturationBudget::default())
        .map_err(|error| error.to_string())?;
    if !report.saturated {
        return Err(format!(
            "planner saturation truncated at nodes {:?}",
            report.truncated
        ));
    }
    let cost = Roofline::new(facts::seed_facts(&caps));
    let search = LocalSearch::new(planner, caps.clone());
    let roots = std::iter::once(root)
        .chain(dependency_roots)
        .collect::<Vec<_>>();
    let plan = search
        .extract(&graph, &roots, &cost, ExtractBudget::default())
        .map_err(|error| error.to_string())?;
    if plan.launches.is_empty() {
        return Err("extraction produced no GPU dispatches".into());
    }
    Ok(Prepared {
        graph,
        roots,
        plan,
        caps,
        search,
        cost,
    })
}

fn validate_recipe(recipe: &Recipe) -> Result<(), String> {
    if recipe.m == 0 || recipe.n == 0 || recipe.k == 0 || recipe.depth == 0 {
        return Err("recipe dimensions and depth must be positive".into());
    }
    for (name, value) in [("m", recipe.m), ("n", recipe.n), ("k", recipe.k)] {
        if value > MAX_RECIPE_DIMENSION {
            return Err(format!(
                "recipe dimension {name}={value} exceeds {MAX_RECIPE_DIMENSION}"
            ));
        }
    }
    if recipe.depth > 8 {
        return Err(format!("recipe depth {} exceeds 8", recipe.depth));
    }
    let product = |a: u32, b: u32| {
        u64::from(a)
            .checked_mul(u64::from(b))
            .ok_or_else(|| "recipe element count overflow".to_owned())
    };
    let elements = match recipe.kind {
        RecipeKind::TailMap | RecipeKind::Sum | RecipeKind::MapChain => u64::from(recipe.m),
        RecipeKind::TransposeMap => product(recipe.m, recipe.n)?,
        RecipeKind::Matmul | RecipeKind::MatmulRowSum => product(recipe.m, recipe.k)?
            .checked_add(product(recipe.k, recipe.n)?)
            .and_then(|count| count.checked_add(product(recipe.m, recipe.n).ok()?))
            .ok_or_else(|| "recipe total element count overflow".to_owned())?,
    };
    if elements > MAX_RECIPE_ELEMENTS {
        return Err(format!(
            "recipe uses {elements} elements, exceeds {MAX_RECIPE_ELEMENTS}"
        ));
    }
    Ok(())
}

fn add_sum(
    graph: &mut EGraph,
    input: fusor_ir::egraph::Id,
) -> Result<fusor_ir::egraph::Id, String> {
    add_sum_axis(graph, input, 0)
}

fn add_sum_axis(
    graph: &mut EGraph,
    input: fusor_ir::egraph::Id,
    axis: u32,
) -> Result<fusor_ir::egraph::Id, String> {
    graph
        .add(Op::Logical(Logical::Fold {
            carrier: Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32),
            axis,
            acc: Dtype::F32,
            ins: smallvec![input],
        }))
        .map_err(|error| error.to_string())
}

fn add_matmul(graph: &mut EGraph, recipe: &Recipe) -> Result<fusor_ir::egraph::Id, String> {
    let a_shape = [
        Dim::Const(u64::from(recipe.m)),
        Dim::Const(u64::from(recipe.k)),
    ];
    let b_shape = [
        Dim::Const(u64::from(recipe.k)),
        Dim::Const(u64::from(recipe.n)),
    ];
    let a = add_buffer(graph, 0, &a_shape)?;
    let b = add_buffer(graph, 1, &b_shape)?;
    graph
        .add(Op::Logical(Logical::Contract {
            spec: EinSpec {
                a: smallvec![Label(0), Label(1)],
                b: smallvec![Label(1), Label(2)],
                out: smallvec![Label(0), Label(2)],
            },
            acc: Dtype::F32,
            a,
            b,
            outs: 1,
        }))
        .map_err(|error| error.to_string())
}

fn add_buffer(
    graph: &mut EGraph,
    name: u32,
    shape: &[Dim],
) -> Result<fusor_ir::egraph::Id, String> {
    graph
        .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
            name: BufferId(name),
            dtype: Dtype::F32,
            shape: shape.iter().copied().collect(),
        })))
        .map_err(|error| error.to_string())
}

fn add_map(
    graph: &mut EGraph,
    input: fusor_ir::egraph::Id,
    chain: bool,
) -> Result<fusor_ir::egraph::Id, String> {
    let expr = if chain {
        ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::bin(
                BinOp::Mul,
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::lit(Splat::F32(1.125)),
            ),
            ScalarExpr::lit(Splat::F32(0.25)),
        )
    } else {
        ScalarExpr::bin(
            BinOp::Add,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::lit(Splat::F32(0.5)),
        )
    };
    graph
        .add(Op::Logical(Logical::Map {
            expr,
            ins: smallvec![input],
            outs: 1,
        }))
        .map_err(|error| error.to_string())
}

fn baseline_caps() -> Caps {
    Caps {
        kind: DeviceKind::Gpu,
        name: CAPS_PROFILE.into(),
        limits: DeviceLimits::default(),
        subgroups: None,
        f16: false,
        bf16: false,
        coop: Default::default(),
        atomic_f32: false,
        workgroup_alias: false,
        mixed_precision_coop_store: false,
        pipeline_cache: false,
        timestamp_query: false,
        simd_widths: Default::default(),
        threads: 1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum MemoryKey {
    Value(u32),
    Arena,
}

struct PlanExecution {
    result: Result<Vec<f32>, (String, Vec<ScheduleTrace>)>,
    traces: Vec<ScheduleTrace>,
    steps: u64,
    dispatches: u64,
}

#[cfg(test)]
fn execute_plan(
    graph: &EGraph,
    roots: &[fusor_ir::egraph::Id],
    caps: &Caps,
    plan: &Plan,
    inputs: &[Vec<f32>],
    seed: u64,
    replay_traces: Option<&[ScheduleTrace]>,
) -> PlanExecution {
    execute_plan_with_limits(
        graph,
        roots,
        caps,
        plan,
        inputs,
        Limits::default(),
        seed,
        replay_traces,
        no_output_fault,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_plan_with_limits(
    graph: &EGraph,
    roots: &[fusor_ir::egraph::Id],
    caps: &Caps,
    plan: &Plan,
    inputs: &[Vec<f32>],
    limits: Limits,
    seed: u64,
    replay_traces: Option<&[ScheduleTrace]>,
    output_fault: OutputFault,
) -> PlanExecution {
    let mut memories = match allocate_memories(graph, plan, inputs) {
        Ok(memories) => memories,
        Err(error) => return failed_execution(error, Vec::new(), 0, 0),
    };
    let mut traces = Vec::new();
    let mut steps = 0;
    for (dispatch_index, dispatch) in plan.launches.iter().enumerate() {
        let node = graph.node(dispatch.root);
        let theta = plan
            .extraction
            .theta
            .get(&dispatch.root)
            .copied()
            .unwrap_or(SchedPoint::Point);
        let empty_dims = Vec::new();
        let context = LowerCtx {
            plan,
            launch: dispatch,
            graph,
            symbols: &plan.symbols,
            dim_bindings: &empty_dims,
        };
        let kernel = match fusor_gpu::lower_kernel_ir(caps, node, theta, &context) {
            Ok(kernel) => kernel,
            Err(error) => {
                return failed_execution(error.to_string(), traces, steps, dispatch_index as u64);
            }
        };
        let mut bindings = match dispatch_bindings(graph, plan, dispatch, &kernel, &memories) {
            Ok(bindings) => bindings,
            Err(error) => return failed_execution(error, traces, steps, dispatch_index as u64),
        };
        let simulator = Simulator::new(limits);
        let run = match replay_traces.and_then(|traces| traces.get(dispatch_index)) {
            Some(trace) => simulator.replay(&kernel, &mut bindings, trace),
            None if replay_traces.is_some() => {
                return failed_execution(
                    "replay trace count is shorter than plan dispatch count".into(),
                    traces,
                    steps,
                    dispatch_index as u64,
                );
            }
            None => simulator.run(
                &kernel,
                &mut bindings,
                seed.wrapping_add(dispatch_index as u64 * 0x9e3779b97f4a7c15),
            ),
        };
        match run {
            Ok(result) => {
                steps += result.steps;
                traces.push(result.trace);
            }
            Err(error) => {
                if let Some(trace) = error.schedule_trace() {
                    traces.push(trace.clone());
                }
                if replay_traces.is_some_and(|expected| expected.len() > dispatch_index + 1) {
                    return failed_execution(
                        "replay trace count has trailing entries after failed dispatch".into(),
                        traces,
                        steps,
                        dispatch_index as u64 + 1,
                    );
                }
                let result = Err((error.to_string(), traces.clone()));
                return PlanExecution {
                    result,
                    traces,
                    steps,
                    dispatches: dispatch_index as u64 + 1,
                };
            }
        }
        for binding in &dispatch.bindings {
            let key = memory_key(graph, binding.value, binding.arena);
            if let Some(memory) = bindings.get(&binding.binding) {
                memories.insert(key, memory.clone());
            }
        }
    }
    if replay_traces.is_some_and(|traces| traces.len() != plan.launches.len()) {
        return failed_execution(
            "replay trace count has trailing entries".into(),
            traces,
            steps,
            plan.launches.len() as u64,
        );
    }
    let mut output = match read_output(
        graph,
        roots[0],
        plan,
        &memories,
        recipe_output_len(inputs, plan, roots[0], graph),
    ) {
        Ok(output) => output,
        Err(error) => return failed_execution(error, traces, steps, plan.launches.len() as u64),
    };
    output_fault(&mut output);
    PlanExecution {
        result: Ok(output),
        traces,
        steps,
        dispatches: plan.launches.len() as u64,
    }
}

fn failed_execution(
    error: String,
    traces: Vec<ScheduleTrace>,
    steps: u64,
    dispatches: u64,
) -> PlanExecution {
    PlanExecution {
        result: Err((error, traces.clone())),
        traces,
        steps,
        dispatches,
    }
}

fn allocate_memories(
    graph: &EGraph,
    plan: &Plan,
    inputs: &[Vec<f32>],
) -> Result<HashMap<MemoryKey, BindingMemory>, String> {
    let mut memories = HashMap::new();
    for buffer in &plan.buffers {
        let key = memory_key(graph, buffer.value, buffer.arena.is_some());
        if memories.contains_key(&key) {
            continue;
        }
        let elements = if buffer.arena.is_some() {
            usize::try_from(plan.arena_bytes.div_ceil(4).max(1))
                .map_err(|error| error.to_string())?
        } else {
            usize::try_from(
                buffer
                    .elements
                    .as_const()
                    .ok_or("simulator requires bounded constant buffers")?,
            )
            .map_err(|error| error.to_string())?
        };
        let id = match key {
            MemoryKey::Value(value) => u64::from(value) + 1,
            MemoryKey::Arena => u64::MAX - 1,
        };
        memories.insert(key, BindingMemory::uninitialized(id, elements));
    }
    for dispatch in &plan.launches {
        for binding in &dispatch.bindings {
            let key = memory_key(graph, binding.value, binding.arena);
            if memories.contains_key(&key) {
                continue;
            }
            let elements = if let Some(buffer) = plan
                .buffers
                .iter()
                .find(|buffer| graph.class_of(buffer.value) == graph.class_of(binding.value))
            {
                usize::try_from(
                    buffer
                        .elements
                        .as_const()
                        .ok_or("symbolic binding is unsupported")?,
                )
                .map_err(|error| error.to_string())?
            } else if let Some(name) = logical_buffer_name(graph, binding.value) {
                inputs
                    .get(name as usize)
                    .ok_or("input binding has no recipe data")?
                    .len()
            } else {
                return Err(format!(
                    "binding {} value {:?} has no allocation plan",
                    binding.binding, binding.value
                ));
            };
            let id = match key {
                MemoryKey::Value(value) => u64::from(value) + 1,
                MemoryKey::Arena => u64::MAX - 1,
            };
            memories.insert(key, BindingMemory::uninitialized(id, elements));
        }
    }
    for buffer in &plan.buffers {
        if let Some(name) = logical_buffer_name(graph, buffer.value) {
            let Some(values) = inputs.get(name as usize) else {
                continue;
            };
            let key = memory_key(graph, buffer.value, buffer.arena.is_some());
            let memory = memories
                .get_mut(&key)
                .ok_or("input buffer allocation missing")?;
            let start = buffer.arena.unwrap_or(0).div_ceil(4) as usize
                + usize::try_from(buffer.layout.offset().as_const().unwrap_or(0))
                    .map_err(|error| error.to_string())?;
            for (offset, value) in values.iter().enumerate() {
                let cell = memory
                    .cells
                    .get_mut(start + offset)
                    .ok_or("input buffer exceeds plan allocation")?;
                *cell = Some(MemoryValue::F32(*value));
            }
        }
    }
    for dispatch in &plan.launches {
        for binding in &dispatch.bindings {
            if let Some(name) = logical_buffer_name(graph, binding.value)
                && let Some(values) = inputs.get(name as usize)
            {
                let key = memory_key(graph, binding.value, binding.arena);
                let memory = memories
                    .get_mut(&key)
                    .ok_or("input binding allocation missing")?;
                let offset = plan
                    .buffers
                    .iter()
                    .find(|buffer| graph.class_of(buffer.value) == graph.class_of(binding.value))
                    .map(|buffer| {
                        buffer.arena.unwrap_or(0).div_ceil(4) as usize
                            + buffer.layout.offset().as_const().unwrap_or(0) as usize
                    })
                    .unwrap_or(0);
                for (index, value) in values.iter().enumerate() {
                    let cell = memory
                        .cells
                        .get_mut(offset + index)
                        .ok_or("input binding exceeds simulator allocation")?;
                    *cell = Some(MemoryValue::F32(*value));
                }
            }
        }
    }
    Ok(memories)
}

fn logical_buffer_name(graph: &EGraph, id: fusor_ir::egraph::Id) -> Option<u32> {
    graph
        .class_ids(graph.class_of(id))
        .into_iter()
        .find_map(|member| match &graph.node(member).op {
            Op::Logical(Logical::Leaf(
                LeafKind::Buffer { name, .. } | LeafKind::Param { name, .. },
            )) => Some(name.0),
            _ => None,
        })
}

fn memory_key(graph: &EGraph, value: fusor_ir::egraph::Id, arena: bool) -> MemoryKey {
    if arena {
        MemoryKey::Arena
    } else {
        MemoryKey::Value(graph.class_of(value).0.0)
    }
}

fn dispatch_bindings(
    graph: &EGraph,
    _plan: &Plan,
    dispatch: &fusor_ir::extract::Dispatch,
    kernel: &KernelIr,
    memories: &HashMap<MemoryKey, BindingMemory>,
) -> Result<BTreeMap<u32, BindingMemory>, String> {
    let mut bindings = BTreeMap::new();
    let mut uniform = BindingMemory::uninitialized(0, 1);
    uniform.cells[0] = Some(MemoryValue::U32(0));
    bindings.insert(0, uniform);
    for buffer in &kernel.buffers {
        if buffer.binding == 0 {
            continue;
        }
        let plan_binding = dispatch
            .bindings
            .iter()
            .find(|binding| binding.binding == buffer.binding)
            .ok_or_else(|| {
                format!(
                    "KernelIr binding {} has no dispatch binding",
                    buffer.binding
                )
            })?;
        let key = memory_key(graph, plan_binding.value, plan_binding.arena);
        let memory = memories
            .get(&key)
            .ok_or_else(|| format!("binding {} has no simulator allocation", buffer.binding))?;
        bindings.insert(buffer.binding, memory.clone());
    }
    Ok(bindings)
}

fn recipe_output_len(
    inputs: &[Vec<f32>],
    plan: &Plan,
    root: fusor_ir::egraph::Id,
    graph: &EGraph,
) -> usize {
    if let Some(buffer) = plan
        .buffers
        .iter()
        .find(|buffer| graph.class_of(buffer.value) == graph.class_of(root))
    {
        return buffer
            .layout
            .shape()
            .iter()
            .try_fold(1usize, |count, dim| {
                count.checked_mul(usize::try_from(dim.as_const()?).ok()?)
            })
            .unwrap_or(1);
    }
    inputs.first().map_or(1, Vec::len)
}

fn read_output(
    graph: &EGraph,
    root: fusor_ir::egraph::Id,
    plan: &Plan,
    memories: &HashMap<MemoryKey, BindingMemory>,
    count: usize,
) -> Result<Vec<f32>, String> {
    let buffer = plan
        .buffers
        .iter()
        .find(|buffer| graph.class_of(buffer.value) == graph.class_of(root))
        .ok_or("extracted output has no plan buffer")?;
    let key = memory_key(graph, buffer.value, buffer.arena.is_some());
    let memory = memories
        .get(&key)
        .ok_or("extracted output allocation missing")?;
    let shape = buffer
        .layout
        .shape()
        .iter()
        .map(|dim| {
            usize::try_from(dim.as_const().ok_or("symbolic output shape")?)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let strides = buffer
        .layout
        .strides()
        .iter()
        .map(|dim| {
            usize::try_from(dim.as_const().ok_or("symbolic output stride")?)
                .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, String>>()?;
    if shape.len() != strides.len() {
        return Err("output shape/stride rank mismatch".into());
    }
    let logical_count = shape
        .iter()
        .try_fold(1usize, |count, extent| count.checked_mul(*extent))
        .ok_or("output element count overflow")?;
    if logical_count != count {
        return Err(format!(
            "logical output length {logical_count} differs from expected {count}"
        ));
    }
    let start = buffer.arena.unwrap_or(0).div_ceil(4) as usize
        + usize::try_from(
            buffer
                .layout
                .offset()
                .as_const()
                .ok_or("symbolic output offset")?,
        )
        .map_err(|error| error.to_string())?;
    (0..logical_count)
        .map(|flat| {
            let mut remainder = flat;
            let mut offset = 0usize;
            for axis in (0..shape.len()).rev() {
                let coordinate = remainder % shape[axis];
                remainder /= shape[axis];
                offset = offset
                    .checked_add(
                        coordinate
                            .checked_mul(strides[axis])
                            .ok_or("output stride overflow")?,
                    )
                    .ok_or("output offset overflow")?;
            }
            let physical = start.checked_add(offset).ok_or("output range overflow")?;
            match memory
                .cells
                .get(physical)
                .ok_or("output range exceeds simulator allocation")?
            {
                Some(MemoryValue::F32(value)) => Ok(*value),
                Some(other) => Err(format!("output element {flat} is not f32: {other:?}")),
                None => Err(format!("output element {flat} is uninitialized")),
            }
        })
        .collect()
}

fn reference(recipe: &Recipe, inputs: &[Vec<f32>]) -> Vec<f32> {
    match recipe.kind {
        RecipeKind::TailMap => inputs[0].iter().map(|value| value + 0.5).collect(),
        RecipeKind::Sum => vec![inputs[0].iter().copied().sum()],
        RecipeKind::MapChain => {
            let mut values = inputs[0].clone();
            for _ in 0..recipe.depth.max(1) {
                values = values
                    .into_iter()
                    .map(|value| value * 1.125 + 0.25)
                    .collect();
            }
            values
        }
        RecipeKind::Matmul => {
            let (m, n, k) = (recipe.m as usize, recipe.n as usize, recipe.k as usize);
            let mut result = vec![0.0; m * n];
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0f32;
                    for inner in 0..k {
                        sum += inputs[0][row * k + inner] * inputs[1][inner * n + col];
                    }
                    result[row * n + col] = sum;
                }
            }
            result
        }
        RecipeKind::TransposeMap => {
            let (m, n) = (recipe.m as usize, recipe.n as usize);
            (0..n)
                .flat_map(|row| (0..m).map(move |col| inputs[0][col * n + row] + 0.5))
                .collect()
        }
        RecipeKind::MatmulRowSum => {
            let (m, n, k) = (recipe.m as usize, recipe.n as usize, recipe.k as usize);
            let mut result = vec![0.0; n];
            for row in 0..m {
                for col in 0..n {
                    let mut dot = 0.0f32;
                    for inner in 0..k {
                        dot += inputs[0][row * k + inner] * inputs[1][inner * n + col];
                    }
                    result[col] += dot;
                }
            }
            result
        }
    }
}

fn compare_results(actual: &[f32], expected: &[f32]) -> Option<String> {
    if actual.len() != expected.len() {
        return Some(format!(
            "output length {} differs from reference {}",
            actual.len(),
            expected.len()
        ));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let same_nonfinite = (actual.is_nan() && expected.is_nan())
            || (actual.is_infinite()
                && expected.is_infinite()
                && actual.is_sign_positive() == expected.is_sign_positive());
        let tolerance = 1e-4 + 1e-4 * expected.abs();
        if !same_nonfinite
            && (!actual.is_finite()
                || !expected.is_finite()
                || (actual - expected).abs() > tolerance)
        {
            return Some(format!(
                "element {index}: simulator={actual:?}, reference={expected:?}, tolerance={tolerance}"
            ));
        }
    }
    None
}

fn generate_inputs(recipe: &Recipe) -> Vec<Vec<f32>> {
    let mut rng = CampaignRng::new(recipe.input_seed);
    recipe
        .input_lengths()
        .into_iter()
        .map(|length| {
            (0..length)
                .map(|_| (rng.next() % 2001) as i32 as f32 / 250.0 - 4.0)
                .collect()
        })
        .collect()
}

fn validate_inputs(recipe: &Recipe, inputs: &[Vec<f32>]) -> Result<(), String> {
    validate_recipe(recipe)?;
    let expected = recipe.input_lengths();
    if inputs.len() != expected.len()
        || inputs
            .iter()
            .zip(&expected)
            .any(|(input, length)| input.len() != *length)
    {
        return Err("failure input lengths do not match recipe".into());
    }
    Ok(())
}

fn apply_selections(
    prepared: &mut Prepared,
    selections: &[CandidateSelection],
) -> Result<(Option<Plan>, Option<String>), String> {
    let mut plan = prepared.plan.clone();
    for (index, selection) in selections.iter().enumerate() {
        let results = prepared.search.test_launch_variant_results(
            &prepared.graph,
            &prepared.roots,
            &plan,
            selection.launch,
            &prepared.cost,
        );
        let Some((_, result)) = results
            .into_iter()
            .find(|(label, _)| label == &selection.label)
        else {
            return Err(format!(
                "candidate label {:?} is not available for launch {}",
                selection.label, selection.launch
            ));
        };
        match result {
            Ok(next) => plan = next,
            Err(error) => {
                let error = candidate_selection_error(index, selections.len(), error.to_string())?;
                return Ok((None, Some(error)));
            }
        }
    }
    Ok((Some(plan), None))
}

fn candidate_selection_error(
    failed_at: usize,
    selection_count: usize,
    error: String,
) -> Result<String, String> {
    if failed_at + 1 != selection_count {
        return Err("candidate failure has trailing selections".into());
    }
    Ok(error)
}

fn failure_record(
    kind: &str,
    recipe: Recipe,
    inputs: Vec<Vec<f32>>,
    candidate: String,
    selections: Vec<CandidateSelection>,
    trace: Vec<ScheduleTrace>,
    detail: String,
) -> FailureRecord {
    FailureRecord {
        version: FAILURE_VERSION,
        kind: kind.into(),
        recipe,
        inputs,
        caps_profile: CAPS_PROFILE.into(),
        limits: Limits::default(),
        candidate,
        selections,
        trace,
        detail,
    }
}

fn classify_sim_error(error: &str) -> String {
    let lower = error.to_ascii_lowercase();
    if lower.contains("unsupported kernelir") {
        "unsupported-ir".into()
    } else if lower.contains("out of length") || lower.contains("outside length") {
        "out-of-bounds".into()
    } else if lower.contains("uninitialized") {
        "uninitialized-read".into()
    } else if lower.contains("unordered") || lower.contains("conflict") {
        "data-race".into()
    } else if lower.contains("barrier") || lower.contains("deadlock") {
        "barrier-divergence".into()
    } else if lower.contains("limit") {
        "resource-limit".into()
    } else if lower.contains("trace") {
        "trace-invalid".into()
    } else {
        "simulator-error".into()
    }
}

struct CampaignRng(u64);

impl CampaignRng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe(kind: RecipeKind) -> Recipe {
        Recipe {
            kind,
            m: 7,
            n: 3,
            k: 5,
            depth: 2,
            input_seed: 19,
        }
    }

    fn check_lowered_recipe(kind: RecipeKind) {
        let recipe = recipe(kind);
        let inputs = generate_inputs(&recipe);
        let prepared = prepare(&recipe).unwrap();
        let execution = execute_plan(
            &prepared.graph,
            &prepared.roots,
            &prepared.caps,
            &prepared.plan,
            &inputs,
            23,
            None,
        );
        let actual = execution
            .result
            .unwrap_or_else(|(error, _)| panic!("{kind:?}: {error}"));
        assert!(
            compare_results(&actual, &reference(&recipe, &inputs)).is_none(),
            "{kind:?}: {actual:?}"
        );
        assert!(execution.dispatches > 0);
        assert!(execution.steps > 0);
    }

    #[test]
    fn real_lowered_tail_map_matches_independent_reference() {
        check_lowered_recipe(RecipeKind::TailMap);
    }

    #[test]
    fn real_lowered_sum_matches_independent_reference() {
        check_lowered_recipe(RecipeKind::Sum);
    }

    #[test]
    fn real_lowered_matmul_matches_independent_reference() {
        check_lowered_recipe(RecipeKind::Matmul);
    }

    #[test]
    fn real_lowered_transpose_view_matches_independent_reference() {
        check_lowered_recipe(RecipeKind::TransposeMap);
    }

    #[test]
    fn matmul_then_row_sum_is_dependent_and_multi_dispatch() {
        let recipe = recipe(RecipeKind::MatmulRowSum);
        let inputs = generate_inputs(&recipe);
        let prepared = prepare(&recipe).unwrap();
        assert!(
            prepared.plan.launches.len() >= 2,
            "plan must dispatch matmul and row sum separately"
        );
        let execution = execute_plan(
            &prepared.graph,
            &prepared.roots,
            &prepared.caps,
            &prepared.plan,
            &inputs,
            23,
            None,
        );
        let actual = execution
            .result
            .unwrap_or_else(|(error, _)| panic!("MatmulRowSum: {error}"));
        assert!(
            compare_results(&actual, &reference(&recipe, &inputs)).is_none(),
            "{actual:?}"
        );
        assert!(execution.dispatches >= 2);
    }

    #[test]
    fn map_and_sum_cover_workgroup_boundary_tails() {
        for kind in [RecipeKind::TailMap, RecipeKind::Sum] {
            for length in [31, 32, 33, 63, 64, 65] {
                let mut recipe = recipe(kind);
                recipe.m = length;
                let inputs = generate_inputs(&recipe);
                let prepared = prepare(&recipe).unwrap();
                let execution = execute_plan(
                    &prepared.graph,
                    &prepared.roots,
                    &prepared.caps,
                    &prepared.plan,
                    &inputs,
                    u64::from(length),
                    None,
                );
                let actual = execution
                    .result
                    .unwrap_or_else(|(error, _)| panic!("{kind:?} {length}: {error}"));
                assert!(
                    compare_results(&actual, &reference(&recipe, &inputs)).is_none(),
                    "{kind:?} {length}: {actual:?}"
                );
            }
        }
    }

    #[test]
    fn map_chain_campaign_has_candidate_and_execution_coverage() {
        let result = campaign(0, 4).expect("bounded campaign should reach all recipe families");
        assert!(result.first_failure.is_none());
        assert!(result.enumerated_candidates > 0);
        assert!(result.executed_plans > 0);
        assert!(result.dispatches > 0);
        assert!(result.steps > 0);
    }

    #[test]
    fn campaign_failure_json_replays_and_minimizes_with_recorded_step_limit() {
        let limits = Limits {
            max_steps: 1,
            ..Limits::default()
        };
        let record = campaign_with_limits(7, 1, limits)
            .unwrap()
            .first_failure
            .expect("one-step budget must produce a simulator failure");
        assert_eq!(record.limits.max_steps, 1);
        assert_eq!(record.kind, "resource-limit");
        let json = serde_json::to_vec(&record).unwrap();
        let decoded: FailureRecord = serde_json::from_slice(&json).unwrap();
        assert!(replay_failure(&decoded).unwrap().contains("reproduced"));
        let minimized = minimize_failure(&decoded).unwrap();
        assert_eq!(minimized.limits.max_steps, 1);
        assert!(replay_failure(&minimized).unwrap().contains("reproduced"));
    }

    #[test]
    fn injected_discrepancy_saves_replays_and_minimizes_real_lowered_output() {
        let record = campaign_with_fault_hook(7, 1, Limits::default(), injected_output_discrepancy)
            .unwrap()
            .first_failure
            .expect("test discrepancy must be detected by host reference");
        assert_eq!(record.kind, "differential");
        assert!(!record.trace.is_empty());
        let record: FailureRecord =
            serde_json::from_slice(&serde_json::to_vec(&record).unwrap()).unwrap();
        assert!(
            replay_failure_with_fault(&record, injected_output_discrepancy)
                .unwrap()
                .contains("reproduced differential failure")
        );
        let minimized = minimize_failure_with_fault(&record, injected_output_discrepancy).unwrap();
        assert!(recipe_complexity(&minimized.recipe) < recipe_complexity(&record.recipe));
        assert!(same_failure_fingerprint(&minimized, &record));
        let minimized: FailureRecord =
            serde_json::from_slice(&serde_json::to_vec(&minimized).unwrap()).unwrap();
        assert!(
            replay_failure_with_fault(&minimized, injected_output_discrepancy)
                .unwrap()
                .contains("reproduced differential failure")
        );
    }

    #[test]
    fn invocation_limit_campaign_failure_replays_and_rejects_extra_traces() {
        let limits = Limits {
            max_invocations: 1,
            ..Limits::default()
        };
        let record = campaign_with_limits(1, 1, limits)
            .unwrap()
            .first_failure
            .expect("single-invocation limit must fail a generated GPU launch");
        assert_eq!(record.kind, "resource-limit");
        assert_eq!(record.limits, limits);
        assert_eq!(record.trace.len(), 1);
        assert!(record.trace[0].choices.is_empty());
        assert!(replay_failure(&record).unwrap().contains("reproduced"));
        let mut extra = record.clone();
        extra.trace[0].choices.push(crate::trace::Choice {
            workgroup: 0,
            boundary: "statement".into(),
            runnable: vec![0],
            runnable_digest: 0,
            lane: 0,
        });
        assert!(replay_failure(&extra).is_err());
    }

    #[test]
    fn failed_dispatch_replay_rejects_unused_later_dispatch_trace() {
        let mut recipe = recipe(RecipeKind::MatmulRowSum);
        recipe.m = 3;
        recipe.n = 3;
        recipe.k = 3;
        let limits = Limits {
            max_steps: 1,
            ..Limits::default()
        };
        let mut record = first_failure_for_recipe(&recipe, limits, no_output_fault)
            .expect("low step limit should fail a real lowered dispatch");
        assert_eq!(record.kind, "resource-limit");
        assert!(record.trace.len() < prepare(&recipe).unwrap().plan.launches.len());
        let later = record.trace[0].clone();
        record.trace.push(later);
        assert!(
            replay_failure(&record)
                .unwrap_err()
                .contains("trailing entries after failed dispatch")
        );
    }

    #[test]
    fn malformed_serialized_recipe_bounds_are_rejected_before_input_validation() {
        let mut recipe = recipe(RecipeKind::Matmul);
        recipe.m = u32::MAX;
        assert!(validate_recipe(&recipe).is_err());
        let record = failure_record(
            "resource-limit",
            recipe,
            Vec::new(),
            "base".into(),
            Vec::new(),
            Vec::new(),
            "simulator invocation limit".into(),
        );
        assert!(
            replay_failure(&record)
                .unwrap_err()
                .contains("recipe dimension")
        );
    }

    #[test]
    fn failure_fingerprint_keeps_candidate_and_error_subtype() {
        use crate::interpreter::SimError;

        let mut recipe = recipe(RecipeKind::TailMap);
        let mut original = failure_record(
            "resource-limit",
            recipe.clone(),
            generate_inputs(&recipe),
            "base".into(),
            Vec::new(),
            Vec::new(),
            SimError::StepLimit { limit: 1 }.to_string(),
        );
        let mut candidate = original.clone();
        candidate.detail = SimError::MemoryLimit {
            required: 2,
            limit: 1,
        }
        .to_string();
        assert!(!same_failure_fingerprint(&candidate, &original));
        candidate = original.clone();
        candidate.kind = "data-race".into();
        candidate.detail = SimError::Race {
            space: "storage".into(),
            index: 2,
            first: "lane 0 write".into(),
            second: "lane 1 write".into(),
        }
        .to_string();
        original.kind = "data-race".into();
        original.detail = SimError::Race {
            space: "workgroup".into(),
            index: 2,
            first: "lane 0 write".into(),
            second: "lane 1 write".into(),
        }
        .to_string();
        assert!(!same_failure_fingerprint(&candidate, &original));

        original.kind = "differential".into();
        original.detail = "output 0 differed".into();
        candidate = original.clone();
        candidate.detail = "output 1 differed".into();
        assert!(same_failure_fingerprint(&candidate, &original));
        candidate.candidate = "different candidate".into();
        assert!(!same_failure_fingerprint(&candidate, &original));

        original.kind = "planning".into();
        original.detail = "first planning error".into();
        candidate = original.clone();
        candidate.detail = "different planning error".into();
        assert!(!same_failure_fingerprint(&candidate, &original));

        recipe.kind = RecipeKind::Sum;
        candidate = original.clone();
        candidate.recipe = recipe;
        assert!(!same_failure_fingerprint(&candidate, &original));
    }

    #[test]
    fn shape_identity_counts_only_recipe_relevant_dimensions() {
        let mut tail = recipe(RecipeKind::TailMap);
        let same_tail_shape = Recipe {
            n: 99,
            k: 101,
            depth: 12,
            ..tail.clone()
        };
        assert_eq!(shape_identity(&tail), shape_identity(&same_tail_shape));
        tail.kind = RecipeKind::MapChain;
        let mut deeper_chain = tail.clone();
        deeper_chain.depth += 1;
        assert_ne!(shape_identity(&tail), shape_identity(&deeper_chain));
    }

    #[test]
    fn recorded_planning_failure_is_verified() {
        let recipe = recipe(RecipeKind::Sum);
        let detail = "injected planner failure".to_owned();
        let original = failure_record(
            "planning",
            recipe.clone(),
            generate_inputs(&recipe),
            "base".into(),
            Vec::new(),
            Vec::new(),
            detail.clone(),
        );
        assert!(matches!(
            replay_preparation(&original, Err(detail.clone())).unwrap(),
            ReplayPreparation::PlanningFailure(error) if error == detail
        ));
        assert!(replay_preparation(&original, Err("changed planner failure".into())).is_err());
        assert!(replay_preparation(&original, prepare(&recipe)).is_err());
    }

    #[test]
    fn failure_shrinking_is_iterative_monotone_and_refreshes_traces() {
        let mut original_recipe = recipe(RecipeKind::TailMap);
        original_recipe.m = 17;
        original_recipe.n = 11;
        original_recipe.k = 7;
        original_recipe.depth = 5;
        let original = failure_record(
            "differential",
            original_recipe.clone(),
            generate_inputs(&original_recipe),
            "base".into(),
            Vec::new(),
            vec![ScheduleTrace {
                version: crate::trace::TRACE_VERSION,
                program_digest: 0,
                input_digest: 0,
                seed: u64::from(original_recipe.m),
                choices: Vec::new(),
            }],
            "injected differential failure".into(),
        );
        let minimized = minimize_failure_search(&original, |candidate| {
            Some(failure_record(
                "differential",
                candidate.clone(),
                generate_inputs(candidate),
                "base".into(),
                Vec::new(),
                vec![ScheduleTrace {
                    version: crate::trace::TRACE_VERSION,
                    program_digest: 0,
                    input_digest: 0,
                    seed: u64::from(candidate.m),
                    choices: Vec::new(),
                }],
                "injected differential failure".into(),
            ))
        });
        assert!(recipe_complexity(&minimized.recipe) < recipe_complexity(&original.recipe));
        assert_eq!(
            (
                minimized.recipe.m,
                minimized.recipe.n,
                minimized.recipe.k,
                minimized.recipe.depth
            ),
            (1, 1, 1, 1)
        );
        assert_eq!(minimized.trace[0].seed, u64::from(minimized.recipe.m));
        assert_ne!(minimized.trace[0], original.trace[0]);
    }

    #[test]
    fn zero_coverage_does_not_discard_planning_failure() {
        let recipe = recipe(RecipeKind::TailMap);
        let failure = failure_record(
            "planning",
            recipe.clone(),
            generate_inputs(&recipe),
            "base".into(),
            Vec::new(),
            Vec::new(),
            "injected planning failure".into(),
        );
        let result = CampaignResult {
            first_failure: Some(failure),
            ..CampaignResult::default()
        };
        assert!(
            validate_campaign_coverage(result)
                .unwrap()
                .first_failure
                .is_some()
        );
        assert!(validate_campaign_coverage(CampaignResult::default()).is_err());
    }

    #[test]
    fn candidate_failure_replay_rejects_unused_trace_and_selection_tail() {
        let recipe = recipe(RecipeKind::TailMap);
        let mut record = failure_record(
            "candidate-replan",
            recipe.clone(),
            generate_inputs(&recipe),
            "candidate".into(),
            vec![CandidateSelection {
                launch: 0,
                label: "candidate".into(),
            }],
            vec![ScheduleTrace {
                version: crate::trace::TRACE_VERSION,
                program_digest: 0,
                input_digest: 0,
                seed: 0,
                choices: Vec::new(),
            }],
            "injected candidate failure".into(),
        );
        assert_eq!(
            replay_failure(&record).unwrap_err(),
            "candidate failure trace has trailing entries"
        );

        record.trace.clear();
        record.selections.clear();
        assert_eq!(
            replay_failure(&record).unwrap_err(),
            "candidate failure has no selected candidate"
        );
        assert_eq!(
            candidate_selection_error(0, 2, "failure".into()).unwrap_err(),
            "candidate failure has trailing selections"
        );
        assert_eq!(
            candidate_selection_error(0, 1, "failure".into()).unwrap(),
            "failure"
        );
        assert!(
            reproduce_candidate_failure(&record, Some("injected candidate failure".into()))
                .unwrap()
                .contains("reproduced candidate replan failure")
        );
        assert!(reproduce_candidate_failure(&record, Some("changed failure".into())).is_err());
        assert!(reproduce_candidate_failure(&record, None).is_err());
    }
}
