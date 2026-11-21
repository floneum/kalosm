use super::{
    CAPS_PROFILE, CandidateSelection, PlanExecution, Recipe, RecipeKind, add_buffer,
    apply_selections, compare_results, execute_plan_with_limits, generate_inputs, no_output_fault,
    prepare, reference, validate_recipe,
};
use crate::interpreter::Limits;
use crate::trace::{ScheduleTrace, stable_digest};
use fusor_cost::replay::{ReplayCache, l0_term_hash};
use fusor_ir::cost::CostModel;
use fusor_ir::egraph::{ClassId, Id};
use fusor_ir::extract::{Extractor, Plan, ReplayKey};
use fusor_ir::ir::Op;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::shape::Dim;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const CACHE_RECORD_VERSION: u32 = 1;
const CACHE_CAMPAIGN_VERSION: u32 = 1;
const MAX_CACHE_ITERATIONS: u32 = 512;
const MAX_CACHE_FILLERS: u32 = 16;
const MAX_EXTRA_FILLERS: u8 = 3;
const FILLER_BUFFER_BASE: u32 = 100_000;
const TRANSPOSE_DIMS: [u32; 7] = [2, 3, 7, 15, 31, 32, 33];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CacheReplayKey {
    pub l0_term: u64,
    pub device: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct CacheSigmaPair {
    pub class: u32,
    pub member: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct CacheScheduleSelection {
    pub node: u32,
    pub point: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "outcome", content = "value", rename_all = "snake_case")]
pub enum CacheExecutionOutcome {
    OutputBits(Vec<u32>),
    Error(String),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CacheExecutionRecord {
    pub dispatches: u64,
    pub steps: u64,
    pub traces: Vec<ScheduleTrace>,
    pub outcome: CacheExecutionOutcome,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CacheObservation {
    pub cache_reused: bool,
    pub cache_hit: bool,
    pub fallback_calls: u32,
    pub returned_sigma: Vec<CacheSigmaPair>,
    pub invalid_selections: Vec<CacheSigmaPair>,
    pub verify_error: Option<String>,
    pub source: CacheExecutionRecord,
    pub direct: CacheExecutionRecord,
    pub replan_error: Option<String>,
    pub replan: Option<CacheExecutionRecord>,
    pub numerical_errors: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CacheRecord {
    pub version: u32,
    pub caps_profile: String,
    pub recipe: Recipe,
    pub inputs: Vec<Vec<f32>>,
    pub limits: Limits,
    pub schedule_seed: u64,
    pub selected_candidate: CandidateSelection,
    pub source_base_graph_len: u32,
    pub source_graph_len: u32,
    pub target_base_graph_len: u32,
    pub target_graph_len: u32,
    pub forwarding_new_ids: Vec<u32>,
    pub filler_count: u32,
    pub extra_filler_count: u8,
    pub replay_key: CacheReplayKey,
    pub source_sigma: Vec<CacheSigmaPair>,
    pub materialization_ids: Vec<u32>,
    pub schedule_selections: Vec<CacheScheduleSelection>,
    pub stale_key_collision: CacheSigmaPair,
    pub filler_buffer_id: u32,
    pub input_config_identity: u64,
    pub expected: CacheObservation,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CacheCampaignResult {
    pub version: u32,
    pub seed: u64,
    pub iterations: u32,
    pub cases: u64,
    pub distinct_shapes: u64,
    pub cache_hits: u64,
    pub fallback_extractions: u64,
    pub invalid_selections: u64,
    pub verify_errors: u64,
    pub replan_errors: u64,
    pub numerical_errors: u64,
    pub execution_errors: u64,
    pub dispatches: u64,
    pub first_failure: Option<CacheRecord>,
}

struct BuiltCase {
    source: super::Prepared,
    retained: Plan,
    target: super::Prepared,
    selected_candidate: CandidateSelection,
    source_base_graph_len: usize,
    forwarding_new_ids: Vec<u32>,
    filler_count: u32,
    replay_key: CacheReplayKey,
    source_sigma: Vec<CacheSigmaPair>,
    materialization_ids: Vec<u32>,
    schedule_selections: Vec<CacheScheduleSelection>,
    stale_key_collision: CacheSigmaPair,
    filler_buffer_id: u32,
}

pub fn cache_campaign(seed: u64, iterations: u32) -> Result<CacheCampaignResult, String> {
    if iterations == 0 || iterations > MAX_CACHE_ITERATIONS {
        return Err(format!(
            "cache campaign iterations must be between 1 and {MAX_CACHE_ITERATIONS}"
        ));
    }
    let mut result = CacheCampaignResult {
        version: CACHE_CAMPAIGN_VERSION,
        seed,
        iterations,
        cases: 0,
        distinct_shapes: 0,
        cache_hits: 0,
        fallback_extractions: 0,
        invalid_selections: 0,
        verify_errors: 0,
        replan_errors: 0,
        numerical_errors: 0,
        execution_errors: 0,
        dispatches: 0,
        first_failure: None,
    };
    let mut shapes = HashSet::new();
    for iteration in 0..iterations {
        let (recipe, extra_fillers, schedule_seed) = generated_case(seed, iteration);
        shapes.insert((recipe.m, recipe.n));
        let candidate = discover_candidate(&recipe, extra_fillers)?;
        let built = build_case(&recipe, &candidate, extra_fillers)?;
        let record = record_case(
            built,
            recipe,
            extra_fillers,
            schedule_seed,
            Limits::default(),
        )?;
        accumulate(&mut result, &record);
        result.cases += 1;
        result.distinct_shapes = shapes.len() as u64;
        if result.first_failure.is_none() && observation_failed(&record.expected) {
            result.first_failure = Some(record);
        }
    }
    Ok(result)
}

pub fn replay_cache_record(record: &CacheRecord) -> Result<String, String> {
    validate_record(record)?;
    let mut built = build_case(
        &record.recipe,
        &record.selected_candidate,
        record.extra_filler_count,
    )?;
    validate_case_snapshot(record, &built)?;
    let actual = observe(
        &mut built,
        &record.recipe,
        &record.inputs,
        record.limits,
        record.schedule_seed,
        Some(&record.expected),
    )?;
    if actual != record.expected {
        return Err("paired-cache observation changed during replay".into());
    }
    Ok(format!(
        "reproduced cache observation: hit={} fallbacks={} invalid_selections={} verify_error={} replan_error={} numerical_errors={}",
        actual.cache_hit,
        actual.fallback_calls,
        actual.invalid_selections.len(),
        actual.verify_error.is_some(),
        actual.replan_error.is_some(),
        actual.numerical_errors.len()
    ))
}

fn generated_case(seed: u64, iteration: u32) -> (Recipe, u8, u64) {
    let mut rng =
        super::CampaignRng::new(seed ^ u64::from(iteration).wrapping_mul(0xd6e8feb86659fd93));
    let m_index = (rng.next() % TRANSPOSE_DIMS.len() as u64) as usize;
    let n_offset = (rng.next() % (TRANSPOSE_DIMS.len() as u64 - 1)) as usize;
    let n_index = if n_offset >= m_index {
        n_offset + 1
    } else {
        n_offset
    };
    let recipe = Recipe {
        kind: RecipeKind::TransposeMap,
        m: TRANSPOSE_DIMS[m_index],
        n: TRANSPOSE_DIMS[n_index],
        k: 1,
        depth: 1,
        input_seed: rng.next(),
    };
    let extra_fillers = (rng.next() % u64::from(MAX_EXTRA_FILLERS + 1)) as u8;
    let schedule_seed = rng.next();
    (recipe, extra_fillers, schedule_seed)
}

fn discover_candidate(recipe: &Recipe, extra_fillers: u8) -> Result<CandidateSelection, String> {
    let discovery = prepare(recipe)?;
    let mut last_error = "no candidate has a selected view copy".to_owned();
    for launch in 0..discovery.plan.launches.len() {
        for (label, candidate) in discovery.search.test_launch_variant_results(
            &discovery.graph,
            &discovery.roots,
            &discovery.plan,
            launch,
            &discovery.cost,
        ) {
            if candidate.is_err() {
                continue;
            }
            let selection = CandidateSelection { launch, label };
            match build_case(recipe, &selection, extra_fillers) {
                Ok(_) => return Ok(selection),
                Err(error) => last_error = error,
            }
        }
    }
    Err(last_error)
}

fn validate_cache_recipe(recipe: &Recipe) -> Result<(), String> {
    validate_recipe(recipe)?;
    if recipe.kind != RecipeKind::TransposeMap
        || !TRANSPOSE_DIMS.contains(&recipe.m)
        || !TRANSPOSE_DIMS.contains(&recipe.n)
        || recipe.m == recipe.n
        || recipe.k != 1
        || recipe.depth != 1
    {
        return Err("cache recipe is outside the bounded transpose-map subset".into());
    }
    Ok(())
}

fn build_case(
    recipe: &Recipe,
    candidate: &CandidateSelection,
    extra_fillers: u8,
) -> Result<BuiltCase, String> {
    validate_cache_recipe(recipe)?;
    if extra_fillers > MAX_EXTRA_FILLERS {
        return Err("cache extra filler count exceeds 3".into());
    }
    let mut source = prepare(recipe)?;
    let (Some(original), None) = apply_selections(&mut source, std::slice::from_ref(candidate))?
    else {
        return Err("recorded candidate has no realized plan".into());
    };
    let source_base_graph_len = source.graph.len();
    let key = ReplayKey {
        l0_term: l0_term_hash(&source.graph, &source.roots),
        device: source.cost.facts().fingerprint(),
    };
    let mut forwarded_extraction = original.extraction.clone();
    if !fusor_cost::forward::forward_selected_views(
        &mut source.graph,
        &source.caps,
        &source.roots,
        &original,
        &mut forwarded_extraction,
    ) {
        return Err("recorded candidate does not forward a selected view copy".into());
    }
    let forwarding_new_ids = (source_base_graph_len..source.graph.len())
        .map(|id| id as u32)
        .collect::<Vec<_>>();
    if forwarding_new_ids.is_empty() {
        return Err("view forwarding added no graph nodes".into());
    }

    let original_sigma_len = original.extraction.sigma.len();
    let mut retained = original.clone();
    let selected_members = retained
        .extraction
        .sigma
        .values()
        .copied()
        .collect::<Vec<_>>();
    retained.extraction.sigma = selected_members
        .into_iter()
        .map(|member| (source.graph.class_of(member), member))
        .collect();
    if retained.extraction.sigma.len() != original_sigma_len {
        return Err("selected members merged during view forwarding".into());
    }
    retained.extraction.m.grow(source.graph.len());
    let source_sigma = sigma_pairs(retained.extraction.sigma.iter());
    if !source_sigma.iter().any(|pair| {
        pair.class as usize >= source_base_graph_len
            && (pair.member as usize) < source_base_graph_len
    }) {
        return Err("forwarding did not move a selected member's class key".into());
    }
    source
        .search
        .verify_plan(&source.graph, &retained)
        .map_err(|error| format!("retained source plan failed verification: {error}"))?;

    let mut target = prepare(recipe)?;
    let target_base_graph_len = target.graph.len();
    if target_base_graph_len != source_base_graph_len
        || l0_term_hash(&target.graph, &target.roots) != key.l0_term
        || target.cost.facts().fingerprint() != key.device
    {
        return Err("independently rebuilt graph changed the pre-forward cache key".into());
    }
    for pair in &source_sigma {
        let member = Id(pair.member);
        if member.index() >= target_base_graph_len
            || source.graph.node(member).op != target.graph.node(member).op
        {
            return Err("original selected member is not valid in graph B".into());
        }
    }
    let max_id = source_sigma
        .iter()
        .flat_map(|pair| [pair.class, pair.member])
        .max()
        .ok_or("retained source plan has empty sigma")?;
    let minimum_fillers = (max_id as usize + 1).saturating_sub(target_base_graph_len);
    let filler_count = minimum_fillers
        .checked_add(usize::from(extra_fillers))
        .ok_or("cache filler count overflow")?;
    if filler_count > MAX_CACHE_FILLERS as usize {
        return Err(format!(
            "cache filler count {filler_count} exceeds {MAX_CACHE_FILLERS}"
        ));
    }
    for offset in 0..filler_count {
        let previous_len = target.graph.len();
        let name = FILLER_BUFFER_BASE
            .checked_add(offset as u32)
            .ok_or("filler buffer ID overflow")?;
        let filler = add_buffer(&mut target.graph, name, &[Dim::Const(1)])?;
        if filler.index() != previous_len || target.graph.len() != previous_len + 1 {
            return Err("unrelated filler buffer did not append exactly one graph node".into());
        }
    }
    if l0_term_hash(&target.graph, &target.roots) != key.l0_term {
        return Err("unrelated fillers changed the target root-term hash".into());
    }
    let stale_key_collision = source_sigma
        .iter()
        .find_map(|pair| {
            if (pair.class as usize) < target_base_graph_len
                || (pair.member as usize) >= target_base_graph_len
            {
                return None;
            }
            let id = Id(pair.class);
            let Op::Logical(Logical::Leaf(LeafKind::Buffer { name, .. })) =
                &target.graph.node(id).op
            else {
                return None;
            };
            (name.0 >= FILLER_BUFFER_BASE && target.graph.class_of(Id(pair.member)) != ClassId(id))
                .then_some((pair.clone(), name.0))
        })
        .ok_or("graph B has no unrelated filler at a cached class key")?;

    let materialization_ids = original.extraction.m.ones().map(|id| id as u32).collect();
    let mut schedule_selections = original
        .extraction
        .theta
        .iter()
        .map(|(node, point)| CacheScheduleSelection {
            node: node.0,
            point: format!("{point:?}"),
        })
        .collect::<Vec<_>>();
    schedule_selections.sort();
    Ok(BuiltCase {
        source,
        retained,
        target,
        selected_candidate: candidate.clone(),
        source_base_graph_len,
        forwarding_new_ids,
        filler_count: filler_count as u32,
        replay_key: CacheReplayKey {
            l0_term: key.l0_term,
            device: key.device,
        },
        source_sigma,
        materialization_ids,
        schedule_selections,
        stale_key_collision: stale_key_collision.0,
        filler_buffer_id: stale_key_collision.1,
    })
}

fn record_case(
    mut built: BuiltCase,
    recipe: Recipe,
    extra_fillers: u8,
    schedule_seed: u64,
    limits: Limits,
) -> Result<CacheRecord, String> {
    let inputs = generate_inputs(&recipe);
    let expected = observe(&mut built, &recipe, &inputs, limits, schedule_seed, None)?;
    let mut record = CacheRecord {
        version: CACHE_RECORD_VERSION,
        caps_profile: CAPS_PROFILE.to_owned(),
        recipe,
        inputs,
        limits,
        schedule_seed,
        selected_candidate: built.selected_candidate,
        source_base_graph_len: built.source_base_graph_len as u32,
        source_graph_len: built.source.graph.len() as u32,
        target_base_graph_len: (built.target.graph.len() - built.filler_count as usize) as u32,
        target_graph_len: built.target.graph.len() as u32,
        forwarding_new_ids: built.forwarding_new_ids,
        filler_count: built.filler_count,
        extra_filler_count: extra_fillers,
        replay_key: built.replay_key,
        source_sigma: built.source_sigma,
        materialization_ids: built.materialization_ids,
        schedule_selections: built.schedule_selections,
        stale_key_collision: built.stale_key_collision,
        filler_buffer_id: built.filler_buffer_id,
        input_config_identity: 0,
        expected,
    };
    record.input_config_identity = config_identity(&record);
    Ok(record)
}

fn validate_record(record: &CacheRecord) -> Result<(), String> {
    if record.version != CACHE_RECORD_VERSION {
        return Err(format!(
            "unsupported cache record version {}",
            record.version
        ));
    }
    if record.caps_profile != CAPS_PROFILE {
        return Err("cache caps profile mismatch".into());
    }
    if record.limits != Limits::default() {
        return Err("cache record limits differ from bounded baseline".into());
    }
    validate_cache_recipe(&record.recipe)?;
    if record.extra_filler_count > MAX_EXTRA_FILLERS || record.filler_count > MAX_CACHE_FILLERS {
        return Err("cache filler count exceeds bounded subset".into());
    }
    let generated = generate_inputs(&record.recipe);
    if !same_input_bits(&generated, &record.inputs) {
        return Err("cache record inputs differ from generated recipe inputs".into());
    }
    if record.input_config_identity != config_identity(record) {
        return Err("cache input/config identity mismatch".into());
    }
    Ok(())
}

fn validate_case_snapshot(record: &CacheRecord, built: &BuiltCase) -> Result<(), String> {
    let target_base_len = built.target.graph.len() - built.filler_count as usize;
    let checks = [
        (
            record.source_base_graph_len as usize == built.source_base_graph_len,
            "source base graph length",
        ),
        (
            record.source_graph_len as usize == built.source.graph.len(),
            "source graph length",
        ),
        (
            record.target_base_graph_len as usize == target_base_len,
            "target base graph length",
        ),
        (
            record.target_graph_len as usize == built.target.graph.len(),
            "target graph length",
        ),
        (
            record.forwarding_new_ids == built.forwarding_new_ids,
            "forwarding node IDs",
        ),
        (record.filler_count == built.filler_count, "filler count"),
        (record.replay_key == built.replay_key, "ReplayKey"),
        (record.source_sigma == built.source_sigma, "source sigma"),
        (
            record.materialization_ids == built.materialization_ids,
            "materialization IDs",
        ),
        (
            record.schedule_selections == built.schedule_selections,
            "schedule selections",
        ),
        (
            record.stale_key_collision == built.stale_key_collision,
            "stale-key collision",
        ),
        (
            record.filler_buffer_id == built.filler_buffer_id,
            "filler buffer ID",
        ),
    ];
    if let Some((_, field)) = checks.into_iter().find(|(equal, _)| !equal) {
        return Err(format!("cache record {field} changed"));
    }
    Ok(())
}

fn observe(
    built: &mut BuiltCase,
    recipe: &Recipe,
    inputs: &[Vec<f32>],
    limits: Limits,
    schedule_seed: u64,
    replay_expected: Option<&CacheObservation>,
) -> Result<CacheObservation, String> {
    let source = run_stage(
        &built.source,
        &built.retained,
        inputs,
        limits,
        schedule_seed,
        replay_expected.map(|expected| &expected.source),
    )?;
    let mut numerical_errors = Vec::new();
    record_numerical_error(&mut numerical_errors, "source", &source, recipe, inputs);

    let cache = ReplayCache::new();
    let key = ReplayKey {
        l0_term: built.replay_key.l0_term,
        device: built.replay_key.device,
    };
    cache.insert(key, built.retained.clone());
    let mut fallback_calls = 0u32;
    let (cached, cache_reused) = cache
        .get_or_extract(key, &built.target.graph, || {
            fallback_calls += 1;
            Ok(built.target.plan.clone())
        })
        .map_err(|error| format!("ReplayCache lookup failed: {error}"))?;
    let cached = (*cached).clone();
    let returned_sigma = sigma_pairs(cached.extraction.sigma.iter());
    let invalid_selections = returned_sigma
        .iter()
        .filter(|pair| built.target.graph.class_of(Id(pair.member)) != ClassId(Id(pair.class)))
        .cloned()
        .collect::<Vec<_>>();
    let verify_error = built
        .target
        .search
        .verify_plan(&built.target.graph, &cached)
        .err()
        .map(|error| error.to_string());
    let direct = run_stage(
        &built.target,
        &cached,
        inputs,
        limits,
        schedule_seed,
        replay_expected.map(|expected| &expected.direct),
    )?;
    record_numerical_error(&mut numerical_errors, "direct", &direct, recipe, inputs);

    let mut extraction = cached.extraction.clone();
    let (replan_error, replan) = match built.target.search.replan_extraction(
        &built.target.graph,
        &built.target.roots,
        &mut extraction,
        &built.target.cost,
    ) {
        Err(error) => (Some(error.to_string()), None),
        Ok(plan) => {
            let expected = replay_expected.and_then(|observation| observation.replan.as_ref());
            if replay_expected.is_some() && expected.is_none() {
                return Err("cache replay produced an unrecorded replan execution".into());
            }
            let run = run_stage(
                &built.target,
                &plan,
                inputs,
                limits,
                schedule_seed,
                expected,
            )?;
            record_numerical_error(&mut numerical_errors, "replan", &run, recipe, inputs);
            (None, Some(run))
        }
    };
    if let Some(expected) = replay_expected
        && expected.replan.is_some() != replan.is_some()
    {
        return Err("cache replay replan outcome changed".into());
    }
    Ok(CacheObservation {
        cache_reused,
        cache_hit: fallback_calls == 0 && cache_reused,
        fallback_calls,
        returned_sigma,
        invalid_selections,
        verify_error,
        source,
        direct,
        replan_error,
        replan,
        numerical_errors,
    })
}

fn run_stage(
    prepared: &super::Prepared,
    plan: &Plan,
    inputs: &[Vec<f32>],
    limits: Limits,
    schedule_seed: u64,
    replay: Option<&CacheExecutionRecord>,
) -> Result<CacheExecutionRecord, String> {
    let execution = execute_plan_with_limits(
        &prepared.graph,
        &prepared.roots,
        &prepared.caps,
        plan,
        inputs,
        limits,
        schedule_seed,
        replay.map(|expected| expected.traces.as_slice()),
        no_output_fault,
        "base",
    );
    let snapshot = execution_record(execution);
    if let Some(expected) = replay
        && snapshot != *expected
    {
        return Err("cache dispatch trace or execution snapshot changed".into());
    }
    Ok(snapshot)
}

fn execution_record(execution: PlanExecution) -> CacheExecutionRecord {
    let outcome = match execution.result {
        Ok(values) => {
            CacheExecutionOutcome::OutputBits(values.into_iter().map(f32::to_bits).collect())
        }
        Err((detail, _)) => CacheExecutionOutcome::Error(detail),
    };
    CacheExecutionRecord {
        dispatches: execution.dispatches,
        steps: execution.steps,
        traces: execution.traces,
        outcome,
    }
}

fn record_numerical_error(
    errors: &mut Vec<String>,
    stage: &str,
    execution: &CacheExecutionRecord,
    recipe: &Recipe,
    inputs: &[Vec<f32>],
) {
    if let CacheExecutionOutcome::OutputBits(bits) = &execution.outcome {
        let actual = bits.iter().copied().map(f32::from_bits).collect::<Vec<_>>();
        if let Some(detail) = compare_results(&actual, &reference(recipe, inputs)) {
            errors.push(format!("{stage}: {detail}"));
        }
    }
}

fn observation_failed(observation: &CacheObservation) -> bool {
    !observation.invalid_selections.is_empty()
        || observation.verify_error.is_some()
        || observation.replan_error.is_some()
        || !observation.numerical_errors.is_empty()
        || execution_failed(&observation.source)
        || execution_failed(&observation.direct)
        || observation.replan.as_ref().is_some_and(execution_failed)
}

fn execution_failed(execution: &CacheExecutionRecord) -> bool {
    matches!(execution.outcome, CacheExecutionOutcome::Error(_))
}

fn accumulate(result: &mut CacheCampaignResult, record: &CacheRecord) {
    let observation = &record.expected;
    result.cache_hits += u64::from(observation.cache_hit);
    result.fallback_extractions += u64::from(observation.fallback_calls);
    result.invalid_selections += observation.invalid_selections.len() as u64;
    result.verify_errors += u64::from(observation.verify_error.is_some());
    result.replan_errors += u64::from(observation.replan_error.is_some());
    result.numerical_errors += observation.numerical_errors.len() as u64;
    result.execution_errors += u64::from(execution_failed(&observation.source))
        + u64::from(execution_failed(&observation.direct))
        + observation
            .replan
            .as_ref()
            .map_or(0, |execution| u64::from(execution_failed(execution)));
    result.dispatches += observation.source.dispatches
        + observation.direct.dispatches
        + observation
            .replan
            .as_ref()
            .map_or(0, |execution| execution.dispatches);
}

fn same_input_bits(left: &[Vec<f32>], right: &[Vec<f32>]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(a, b)| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        })
}

fn sigma_pairs<'a>(sigma: impl Iterator<Item = (&'a ClassId, &'a Id)>) -> Vec<CacheSigmaPair> {
    let mut pairs = sigma
        .map(|(class, member)| CacheSigmaPair {
            class: class.0.0,
            member: member.0,
        })
        .collect::<Vec<_>>();
    pairs.sort();
    pairs
}

fn config_identity(record: &CacheRecord) -> u64 {
    let mut bytes = Vec::new();
    push_u64(&mut bytes, u64::from(record.version));
    push_str(&mut bytes, &record.caps_profile);
    push_u64(
        &mut bytes,
        match record.recipe.kind {
            RecipeKind::TransposeMap => 1,
            _ => 0,
        },
    );
    for value in [
        record.recipe.m,
        record.recipe.n,
        record.recipe.k,
        record.recipe.depth,
    ] {
        push_u64(&mut bytes, u64::from(value));
    }
    push_u64(&mut bytes, record.recipe.input_seed);
    for input in &record.inputs {
        push_u64(&mut bytes, input.len() as u64);
        for value in input {
            push_u64(&mut bytes, u64::from(value.to_bits()));
        }
    }
    for limit in [
        record.limits.max_steps,
        record.limits.max_memory_bytes,
        record.limits.max_workgroups,
        record.limits.max_invocations,
    ] {
        push_u64(&mut bytes, limit);
    }
    push_u64(&mut bytes, record.schedule_seed);
    push_u64(&mut bytes, record.selected_candidate.launch as u64);
    push_str(&mut bytes, &record.selected_candidate.label);
    for value in [
        record.source_base_graph_len,
        record.source_graph_len,
        record.target_base_graph_len,
        record.target_graph_len,
        record.filler_count,
    ] {
        push_u64(&mut bytes, u64::from(value));
    }
    bytes.push(record.extra_filler_count);
    push_u64(&mut bytes, record.replay_key.l0_term);
    push_u64(&mut bytes, record.replay_key.device);
    push_u64(&mut bytes, record.forwarding_new_ids.len() as u64);
    for id in &record.forwarding_new_ids {
        push_u64(&mut bytes, u64::from(*id));
    }
    push_u64(&mut bytes, record.source_sigma.len() as u64);
    for pair in &record.source_sigma {
        push_u64(&mut bytes, u64::from(pair.class));
        push_u64(&mut bytes, u64::from(pair.member));
    }
    push_u64(&mut bytes, record.materialization_ids.len() as u64);
    for id in &record.materialization_ids {
        push_u64(&mut bytes, u64::from(*id));
    }
    push_u64(&mut bytes, record.schedule_selections.len() as u64);
    for selection in &record.schedule_selections {
        push_u64(&mut bytes, u64::from(selection.node));
        push_str(&mut bytes, &selection.point);
    }
    push_u64(&mut bytes, u64::from(record.stale_key_collision.class));
    push_u64(&mut bytes, u64::from(record.stale_key_collision.member));
    push_u64(&mut bytes, u64::from(record.filler_buffer_id));
    stable_digest(&bytes)
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
fn push_str(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, value.len() as u64);
    bytes.extend_from_slice(value.as_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_record() -> CacheRecord {
        let recipe = Recipe {
            kind: RecipeKind::TransposeMap,
            m: 7,
            n: 3,
            k: 1,
            depth: 1,
            input_seed: 19,
        };
        let extra = 0;
        let candidate = discover_candidate(&recipe, extra).unwrap();
        let built = build_case(&recipe, &candidate, extra).unwrap();
        record_case(built, recipe, extra, 23, Limits::default()).unwrap()
    }

    #[test]
    fn paired_cache_fixture_reports_structural_mismatch_if_present() {
        let record = test_record();
        let recipe = record.recipe.clone();
        let candidate = record.selected_candidate.clone();
        let built = build_case(&recipe, &candidate, record.extra_filler_count).unwrap();
        println!(
            "paired-cache candidate={:?} key=({:016x},{:016x}) source_len={}->{} target_len={}->{} fillers={} forwarding={:?} collision={:?}/buffer:{} source_sigma={:?} returned_sigma={:?} invalid={:?} cache_hit={} fallbacks={} verify_error={:?} replan_error={:?} numerical_errors={:?}",
            record.selected_candidate,
            record.replay_key.l0_term,
            record.replay_key.device,
            record.source_base_graph_len,
            record.source_graph_len,
            record.target_base_graph_len,
            record.target_graph_len,
            record.filler_count,
            record.forwarding_new_ids,
            record.stale_key_collision,
            record.filler_buffer_id,
            record.source_sigma,
            record.expected.returned_sigma,
            record.expected.invalid_selections,
            record.expected.cache_hit,
            record.expected.fallback_calls,
            record.expected.verify_error,
            record.expected.replan_error,
            record.expected.numerical_errors
        );
        println!(
            "target class_of(cache_key)={:?} class_of(selected_member)={:?} key_node={:?}",
            built
                .target
                .graph
                .class_of(Id(record.stale_key_collision.class)),
            built
                .target
                .graph
                .class_of(Id(record.stale_key_collision.member)),
            built
                .target
                .graph
                .node(Id(record.stale_key_collision.class))
                .op
        );
        assert!(record.source_sigma.contains(&record.stale_key_collision));
        assert!(record.filler_buffer_id >= FILLER_BUFFER_BASE);
        assert!(record.expected.cache_hit || record.expected.fallback_calls > 0);
        let computed_invalid = record
            .expected
            .returned_sigma
            .iter()
            .filter(|pair| built.target.graph.class_of(Id(pair.member)) != ClassId(Id(pair.class)))
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(record.expected.invalid_selections, computed_invalid);
        if !record.expected.invalid_selections.is_empty() {
            assert!(
                record.expected.verify_error.is_some() || record.expected.replan_error.is_some()
            );
        }
    }

    #[test]
    fn cache_record_json_roundtrip_replays_exact_observation() {
        let record = test_record();
        let decoded: CacheRecord =
            serde_json::from_slice(&serde_json::to_vec(&record).unwrap()).unwrap();
        assert_eq!(record, decoded);
        assert!(
            replay_cache_record(&decoded)
                .unwrap()
                .contains("reproduced cache observation")
        );
    }

    #[test]
    fn cache_record_rejects_changed_inputs_config_ids_and_trailing_traces() {
        let record = test_record();
        let mut invalid = record.clone();
        invalid.version += 1;
        assert!(replay_cache_record(&invalid).is_err());
        let mut invalid = record.clone();
        invalid.recipe.m += 1;
        assert!(replay_cache_record(&invalid).is_err());
        let mut invalid = record.clone();
        invalid.inputs[0][0] += 1.0;
        assert!(replay_cache_record(&invalid).is_err());
        let mut invalid = record.clone();
        invalid.limits.max_steps -= 1;
        assert!(replay_cache_record(&invalid).is_err());
        let mut invalid = record.clone();
        invalid.replay_key.device ^= 1;
        assert!(replay_cache_record(&invalid).is_err());
        let mut invalid = record.clone();
        invalid.selected_candidate.label.push_str("bad");
        assert!(replay_cache_record(&invalid).is_err());
        let mut invalid = record.clone();
        invalid.source_sigma[0].member ^= 1;
        assert!(replay_cache_record(&invalid).is_err());
        let mut invalid = record.clone();
        invalid.source_sigma[0].class = u32::MAX;
        invalid.source_sigma[0].member = u32::MAX;
        invalid.input_config_identity = config_identity(&invalid);
        assert_eq!(
            replay_cache_record(&invalid).unwrap_err(),
            "cache record source sigma changed"
        );
        let mut invalid = record.clone();
        invalid.filler_count += 1;
        assert!(replay_cache_record(&invalid).is_err());
        let mut invalid = record.clone();
        invalid
            .expected
            .source
            .traces
            .push(invalid.expected.source.traces[0].clone());
        assert!(replay_cache_record(&invalid).is_err());
    }

    #[test]
    fn cache_campaign_runs_bounded_distinct_transpose_pairs() {
        let result = cache_campaign(11, 4).unwrap();
        assert_eq!(result.cases, 4);
        assert!(result.distinct_shapes > 1);
        assert_eq!(
            result.cache_hits + result.fallback_extractions,
            result.cases
        );
        assert!(result.dispatches > 0);
        if let Some(record) = &result.first_failure {
            assert!(replay_cache_record(record).unwrap().contains("reproduced"));
        }
    }

    #[test]
    #[ignore = "requires PR471 ReplayCache member-based recanonicalization"]
    fn paired_cache_replay_preserves_selection_on_equivalent_graph() {
        let record = test_record();
        assert!(record.expected.invalid_selections.is_empty());
        assert!(record.expected.verify_error.is_none());
        assert!(record.expected.replan_error.is_none());
        assert!(record.expected.numerical_errors.is_empty());
        assert!(!execution_failed(&record.expected.source));
        assert!(!execution_failed(&record.expected.direct));
        assert!(
            record
                .expected
                .replan
                .as_ref()
                .is_some_and(|execution| !execution_failed(execution))
        );
        assert!(replay_cache_record(&record).is_ok());
    }
}
