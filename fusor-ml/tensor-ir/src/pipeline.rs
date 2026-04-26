use std::{
    collections::{HashMap, HashSet},
    time::Duration,
};

use egg::{Language, RecExpr};

use crate::TensorEGraph;
use crate::extractor::BeamConfig;
use crate::language::{
    DispatchNode, EffectNode, HighLevelNode, SimdNode, TensorIr, add_list, extract_list,
    extract_recexpr_list,
};
use crate::rules::{RunnerConfig, SaturationReport, saturate_phases, saturate_phases_reported};
use crate::stages::{Phase, TensorExprNode, TensorExprProgram, TensorExprSummary};
use crate::types::{
    BinaryOp, BinderKind, BufferRef, DType, DeviceProfile, LoweringOptions, MemTier, ScalarValue,
    Shape, TensorId, VarRef, slots,
};

pub use tensor_ir_egraph::tensor_expr_to_recexpr;

/// Configuration for the strict staged lowering pipeline.
#[derive(Debug, Clone, Default)]
pub struct StageConfig {
    pub runner: RunnerConfig,
    pub beam: BeamConfig,
    pub candidate_limit: Option<usize>,
}

/// Structured diagnostics for one lowering attempt.
#[derive(Debug, Clone)]
pub struct LoweringReport {
    pub input_nodes: usize,
    pub summary: Option<TensorExprSummary>,
    pub legacy_expr_nodes: Option<usize>,
    pub initial_egraph_nodes: Option<usize>,
    pub initial_egraph_classes: Option<usize>,
    pub saturation: SaturationReport,
    pub extraction: Option<ExtractionReport>,
    pub error: Option<String>,
}

impl LoweringReport {
    #[must_use]
    pub fn new(input_nodes: usize) -> Self {
        Self {
            input_nodes,
            summary: None,
            legacy_expr_nodes: None,
            initial_egraph_nodes: None,
            initial_egraph_classes: None,
            saturation: SaturationReport::default(),
            extraction: None,
            error: None,
        }
    }
}

/// Structured diagnostics for extraction and executable-candidate selection.
#[derive(Debug, Clone)]
pub struct ExtractionReport {
    pub beam_width: usize,
    pub candidate_limit: usize,
    pub candidate_validation: CandidateValidationReport,
    pub selected_candidate_index: Option<usize>,
    pub selected_cost: Option<f64>,
    pub selected_nodes: Option<usize>,
    pub error: Option<String>,
    pub elapsed: Duration,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CandidateValidationReport {
    pub requested_limit: usize,
    pub raw_candidate_limit: usize,
    pub raw_candidates: usize,
    pub invalid_var_scopes: usize,
    pub empty_dispatch_programs: usize,
    pub verification_failures: usize,
    pub accepted_before_limit: usize,
    pub returned: usize,
}

/// Owned executable extraction. The `expr` is the actual effectful program
/// handed to verification/codegen; `eclass_for_node` records which e-class a
/// node came from when it originated in the saturated e-graph.
#[derive(Debug, Clone)]
pub struct ExtractedProgram {
    expr: RecExpr<TensorIr>,
    eclass_for_node: Vec<Option<egg::Id>>,
    cost: f64,
}

impl ExtractedProgram {
    #[must_use]
    pub const fn expr(&self) -> &RecExpr<TensorIr> {
        &self.expr
    }

    #[must_use]
    pub fn eclass_for_node(&self) -> &[Option<egg::Id>] {
        &self.eclass_for_node
    }

    #[must_use]
    pub const fn cost(&self) -> f64 {
        self.cost
    }
}

/// Lowering failure with the diagnostics collected before the failure.
#[derive(Debug, Clone)]
pub struct LoweringError {
    pub message: String,
    pub report: LoweringReport,
}

impl LoweringError {
    #[must_use]
    pub const fn new(message: String, report: LoweringReport) -> Self {
        Self { message, report }
    }
}

impl std::fmt::Display for LoweringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for LoweringError {}

/// Stage 3: explicit kernel IR chosen from the mixed lowering backend.
#[derive(Debug, Clone)]
pub struct KernelProgram {
    root: egg::Id,
    /// Executable effectful program rooted at `EffectNode::Program`.
    extracted: ExtractedProgram,
    egraph: TensorEGraph,
    /// Device profile this kernel was lowered for.
    device: DeviceProfile,
    /// Lowering toggles carried from the runner config.
    lowering: LoweringOptions,
}

impl KernelProgram {
    #[must_use]
    pub const fn root(&self) -> egg::Id {
        self.root
    }

    #[must_use]
    pub const fn extracted(&self) -> &RecExpr<TensorIr> {
        self.extracted.expr()
    }

    #[must_use]
    pub const fn extracted_program(&self) -> &ExtractedProgram {
        &self.extracted
    }

    #[must_use]
    pub const fn egraph(&self) -> &TensorEGraph {
        &self.egraph
    }

    #[must_use]
    pub const fn cost(&self) -> f64 {
        self.extracted.cost()
    }

    #[must_use]
    pub const fn device(&self) -> &DeviceProfile {
        &self.device
    }

    #[must_use]
    pub const fn lowering(&self) -> &LoweringOptions {
        &self.lowering
    }
}

/// End-to-end staged lowering façade.
#[derive(Debug, Clone, Default)]
pub struct StagedPipeline {
    config: StageConfig,
}

impl StagedPipeline {
    #[must_use]
    pub const fn new(config: StageConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub const fn config(&self) -> &StageConfig {
        &self.config
    }

    /// # Errors
    ///
    /// Returns an error if lowering the tensor expression fails.
    pub fn lower(&self, expr: &TensorExprProgram) -> Result<KernelProgram, String> {
        lower_tensor_expr(expr, &self.config)
    }

    /// # Errors
    ///
    /// Returns an error with a partial report if lowering the tensor
    /// expression fails.
    pub fn lower_with_report(
        &self,
        expr: &TensorExprProgram,
    ) -> Result<(KernelProgram, LoweringReport), LoweringError> {
        lower_tensor_expr_with_report(expr, &self.config)
    }

    /// # Errors
    ///
    /// Returns an error if candidate lowering fails before extraction completes.
    pub fn lower_candidates(
        &self,
        expr: &TensorExprProgram,
        limit: usize,
    ) -> Result<Vec<KernelProgram>, String> {
        lower_tensor_expr_candidates(expr, &self.config, limit)
    }
}

/// Lower a tensor expression into a validated kernel-stage program.
///
/// # Errors
///
/// Returns an error if lowering, saturation, extraction, or validation
/// fails — including if the chosen candidate exceeds the configured
/// [`DeviceProfile::max_threadgroup_bytes`].
pub fn lower_tensor_expr(
    expr: &TensorExprProgram,
    config: &StageConfig,
) -> Result<KernelProgram, String> {
    lower_tensor_expr_with_report(expr, config)
        .map(|(kernel, _report)| kernel)
        .map_err(|err| err.message)
}

/// Lower a tensor expression into a validated kernel-stage program and return
/// structured diagnostics for the attempt.
///
/// # Errors
///
/// Returns an error with a partial report if lowering, saturation,
/// extraction, validation, or device-budget enforcement fails.
pub fn lower_tensor_expr_with_report(
    expr: &TensorExprProgram,
    config: &StageConfig,
) -> Result<(KernelProgram, LoweringReport), LoweringError> {
    let mut report = LoweringReport::new(expr.nodes().len());
    let summary = match validate_tensor_expr_lowering_contract(expr) {
        Ok(summary) => {
            report.summary = Some(summary.clone());
            summary
        }
        Err(message) => {
            report.error = Some(message.clone());
            return Err(LoweringError::new(message, report));
        }
    };

    let span = tracing::info_span!(
        "lower_tensor_expr",
        phases = ?Phase::all(),
    );
    let _enter = span.enter();

    let legacy_expr = match tensor_expr_to_recexpr(expr) {
        Ok(expr) => expr,
        Err(message) => {
            report.error = Some(message.clone());
            return Err(LoweringError::new(message, report));
        }
    };
    report.legacy_expr_nodes = Some(legacy_expr.as_ref().len());
    let mut egraph = TensorEGraph::default();
    let root = egraph.add_expr(&legacy_expr);
    egraph.rebuild();
    report.initial_egraph_nodes = Some(egraph.total_size());
    report.initial_egraph_classes = Some(egraph.number_of_classes());

    let (mut egraph, saturation) = saturate_phases_reported(egraph, Phase::all(), &config.runner);
    report.saturation = saturation;
    let effect_root = materialize_effect_programs_in_egraph(&mut egraph, root);
    egraph.rebuild();
    let beam = tuned_beam_config(&summary, config);

    let extract_span = tracing::info_span!(
        "extract",
        beam_width = beam.beam_width,
        nodes = egraph.total_size(),
        classes = egraph.number_of_classes(),
    );
    let (candidate_result, extraction_report) = extract_span.in_scope(|| {
        select_kernel_candidate_with_report(
            &egraph,
            effect_root,
            &beam,
            &config.runner.device,
            config.runner.lowering,
            config
                .candidate_limit
                .unwrap_or_else(|| beam.beam_width.max(16)),
        )
    });
    report.extraction = Some(extraction_report);
    let extracted = match candidate_result {
        Ok(candidate) => candidate,
        Err(message) => {
            report.error = Some(message.clone());
            return Err(LoweringError::new(message, report));
        }
    };
    tracing::info!(
        cost = extracted.cost(),
        extracted_nodes = extracted.expr().as_ref().len(),
        "extraction complete"
    );

    if let Err(message) = validate_kernel_expr(extracted.expr()) {
        report.error = Some(message.clone());
        return Err(LoweringError::new(message, report));
    }

    Ok((
        KernelProgram {
            root: effect_root,
            extracted,
            egraph,
            device: config.runner.device,
            lowering: config.runner.lowering,
        },
        report,
    ))
}

/// Lower a tensor expression into multiple validated kernel candidates.
///
/// # Errors
///
/// Returns an error if lowering or saturation fails before candidate extraction.
pub fn lower_tensor_expr_candidates(
    expr: &TensorExprProgram,
    config: &StageConfig,
    limit: usize,
) -> Result<Vec<KernelProgram>, String> {
    let summary = validate_tensor_expr_lowering_contract(expr)?;

    let legacy_expr = tensor_expr_to_recexpr(expr)?;
    let mut egraph = TensorEGraph::default();
    let root = egraph.add_expr(&legacy_expr);
    egraph.rebuild();

    let mut egraph = saturate_phases(egraph, Phase::all(), &config.runner);
    let effect_root = materialize_effect_programs_in_egraph(&mut egraph, root);
    egraph.rebuild();
    let beam = tuned_beam_config(&summary, config);
    let raw_limit = limit.saturating_mul(64).max(limit);
    let candidates =
        crate::extractor::beam_extract_candidates(&egraph, effect_root, &beam, raw_limit);

    let mut kernels = Vec::new();
    for (cost, extracted) in candidates {
        let effect_program = extracted_program_from_egraph_candidate(&egraph, cost, extracted);
        if validate_kernel_expr(effect_program.expr()).is_err() {
            continue;
        }
        if enforce_effect_threadgroup_budget(&effect_program, &egraph, &config.runner.device)
            .is_err()
        {
            continue;
        }
        kernels.push(KernelProgram {
            root: effect_root,
            extracted: effect_program,
            egraph: egraph.clone(),
            device: config.runner.device,
            lowering: config.runner.lowering,
        });
        if kernels.len() >= limit {
            break;
        }
    }
    Ok(kernels)
}

fn select_kernel_candidate_with_report(
    egraph: &TensorEGraph,
    root: egg::Id,
    beam: &BeamConfig,
    _device: &DeviceProfile,
    _lowering: LoweringOptions,
    candidate_limit: usize,
) -> (Result<ExtractedProgram, String>, ExtractionReport) {
    let start = std::time::Instant::now();
    let raw_candidate_limit = candidate_limit.saturating_mul(64).max(candidate_limit);
    let candidates =
        crate::extractor::beam_extract_candidates(egraph, root, beam, raw_candidate_limit);
    let raw_candidates = candidates.len();
    let candidate_validation = CandidateValidationReport {
        requested_limit: candidate_limit,
        raw_candidate_limit,
        raw_candidates,
        invalid_var_scopes: 0,
        empty_dispatch_programs: 0,
        verification_failures: 0,
        accepted_before_limit: 0,
        returned: 0,
    };
    let mut report = ExtractionReport {
        beam_width: beam.beam_width,
        candidate_limit,
        candidate_validation,
        selected_candidate_index: None,
        selected_cost: None,
        selected_nodes: None,
        error: None,
        elapsed: Duration::default(),
    };
    if candidates.is_empty() {
        let message = "no valid executable kernel candidates after rewrite saturation".to_string();
        report.error = Some(message.clone());
        report.elapsed = start.elapsed();
        return (Err(message), report);
    }

    let mut last_rejection = None;
    for (index, (cost, extracted)) in candidates.into_iter().enumerate() {
        let extracted_nodes = extracted.as_ref().len();
        let effect_program = extracted_program_from_egraph_candidate(egraph, cost, extracted);
        if let Err(error) = validate_kernel_expr(effect_program.expr()) {
            last_rejection = Some(error);
            report.candidate_validation.verification_failures += 1;
            continue;
        }
        if let Err(error) = enforce_effect_threadgroup_budget(&effect_program, egraph, _device) {
            last_rejection = Some(error);
            report.candidate_validation.verification_failures += 1;
            continue;
        }

        report.candidate_validation.accepted_before_limit += 1;
        report.candidate_validation.returned += 1;
        report.selected_candidate_index = Some(index);
        report.selected_cost = Some(cost);
        report.selected_nodes = Some(extracted_nodes);
        report.elapsed = start.elapsed();
        return (Ok(effect_program), report);
    }

    let message = last_rejection.map_or_else(
        || "no valid executable kernel candidates after effect-program validation".to_string(),
        |error| {
            format!(
                "no valid executable kernel candidates after effect-program validation: {error}"
            )
        },
    );
    report.error = Some(message.clone());
    report.elapsed = start.elapsed();
    (Err(message), report)
}

pub(crate) fn enforce_effect_threadgroup_budget(
    extracted: &ExtractedProgram,
    egraph: &TensorEGraph,
    device: &DeviceProfile,
) -> Result<(), String> {
    if device.max_threadgroup_bytes == 0 {
        return Ok(());
    }
    let expr = extracted.expr();
    let nodes = expr.as_ref();
    let Some(root_idx) = nodes.len().checked_sub(1) else {
        return Err("kernel program is empty".into());
    };
    let TensorIr::Effect(EffectNode::Program {
        children: [_, body, _],
    }) = &nodes[root_idx]
    else {
        return Err("kernel root is not an effect program".into());
    };

    let mut dispatch_layouts = Vec::new();
    collect_effect_dispatch_threadgroup_layouts(
        nodes,
        extracted,
        egraph,
        *body,
        device,
        &mut dispatch_layouts,
    )?;
    for (dispatch_index, buffers) in dispatch_layouts.into_iter().enumerate() {
        let bytes = buffers
            .values()
            .fold(0_u64, |acc, layout| acc.saturating_add(layout.bytes()));
        let budget = u64::from(device.max_threadgroup_bytes);
        if bytes > budget {
            return Err(format!(
                "kernel dispatch {dispatch_index} exceeds device threadgroup budget: needs {bytes} bytes, device allows {budget} bytes (DeviceProfile::max_threadgroup_bytes)"
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct ThreadgroupBudgetLayout {
    elements: u32,
    dtype: DType,
}

impl ThreadgroupBudgetLayout {
    fn bytes(self) -> u64 {
        u64::from(self.elements.max(1)).saturating_mul(u64::from(self.dtype.byte_size()))
    }
}

fn collect_effect_dispatch_threadgroup_layouts(
    nodes: &[TensorIr],
    extracted: &ExtractedProgram,
    egraph: &TensorEGraph,
    step: egg::Id,
    device: &DeviceProfile,
    out: &mut Vec<HashMap<BufferRef, ThreadgroupBudgetLayout>>,
) -> Result<(), String> {
    match &nodes[usize::from(step)] {
        TensorIr::Effect(EffectNode::Seq(list_id)) => {
            for child in extract_recexpr_list(nodes, *list_id) {
                collect_effect_dispatch_threadgroup_layouts(
                    nodes, extracted, egraph, child, device, out,
                )?;
            }
        }
        TensorIr::Effect(EffectNode::Dispatch {
            simdgroups,
            children: [_state, body],
            ..
        }) => {
            let mut buffers = HashMap::new();
            collect_threadgroup_layouts_in_effect_chain(
                nodes,
                extracted,
                egraph,
                *body,
                *simdgroups,
                device.simd_width,
                &mut buffers,
            )?;
            out.push(buffers);
        }
        TensorIr::Effect(EffectNode::Token) => {}
        other => return Err(format!("program body contains non-effect node {other:?}")),
    }
    Ok(())
}

fn collect_threadgroup_layouts_in_effect_chain(
    nodes: &[TensorIr],
    extracted: &ExtractedProgram,
    egraph: &TensorEGraph,
    state: egg::Id,
    simdgroups: u32,
    simd_width: u32,
    buffers: &mut HashMap<BufferRef, ThreadgroupBudgetLayout>,
) -> Result<(), String> {
    match &nodes[usize::from(state)] {
        TensorIr::Effect(EffectNode::Token) => Ok(()),
        TensorIr::Effect(EffectNode::Store { tier, children }) => {
            collect_threadgroup_layouts_in_effect_chain(
                nodes,
                extracted,
                egraph,
                children[2],
                simdgroups,
                simd_width,
                buffers,
            )?;
            if let MemTier::Threadgroup(buffer) = tier {
                record_threadgroup_budget_layout(
                    nodes,
                    extracted,
                    egraph,
                    *buffer,
                    children[0],
                    children[1],
                    None,
                    simdgroups,
                    simd_width,
                    buffers,
                );
            } else {
                collect_threadgroup_layouts_in_value_subtree(
                    nodes,
                    extracted,
                    egraph,
                    children[1],
                    simdgroups,
                    simd_width,
                    buffers,
                    &mut HashSet::new(),
                );
            }
            Ok(())
        }
        TensorIr::Effect(EffectNode::StoreIf { tier, children }) => {
            collect_threadgroup_layouts_in_effect_chain(
                nodes,
                extracted,
                egraph,
                children[3],
                simdgroups,
                simd_width,
                buffers,
            )?;
            if let MemTier::Threadgroup(buffer) = tier {
                record_threadgroup_budget_layout(
                    nodes,
                    extracted,
                    egraph,
                    *buffer,
                    children[1],
                    children[2],
                    Some(children[0]),
                    simdgroups,
                    simd_width,
                    buffers,
                );
            } else {
                collect_threadgroup_layouts_in_value_subtree(
                    nodes,
                    extracted,
                    egraph,
                    children[2],
                    simdgroups,
                    simd_width,
                    buffers,
                    &mut HashSet::new(),
                );
            }
            Ok(())
        }
        TensorIr::Effect(EffectNode::Barrier { state, .. }) => {
            collect_threadgroup_layouts_in_effect_chain(
                nodes, extracted, egraph, *state, simdgroups, simd_width, buffers,
            )
        }
        other => Err(format!(
            "dispatch body contains non-effect state node {other:?}"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn collect_threadgroup_layouts_in_value_subtree(
    nodes: &[TensorIr],
    extracted: &ExtractedProgram,
    egraph: &TensorEGraph,
    id: egg::Id,
    simdgroups: u32,
    simd_width: u32,
    buffers: &mut HashMap<BufferRef, ThreadgroupBudgetLayout>,
    seen: &mut HashSet<egg::Id>,
) {
    if !seen.insert(id) {
        return;
    }
    match &nodes[usize::from(id)] {
        TensorIr::Simd(SimdNode::Load {
            tier: MemTier::Threadgroup(buffer),
            children,
        }) => {
            record_threadgroup_budget_load(
                nodes,
                extracted,
                egraph,
                *buffer,
                id,
                children[0],
                simdgroups,
                simd_width,
                buffers,
            );
            collect_threadgroup_layouts_in_value_subtree(
                nodes,
                extracted,
                egraph,
                children[1],
                simdgroups,
                simd_width,
                buffers,
                seen,
            );
        }
        TensorIr::Simd(SimdNode::Store { tier, children }) => {
            collect_threadgroup_layouts_in_value_subtree(
                nodes,
                extracted,
                egraph,
                children[2],
                simdgroups,
                simd_width,
                buffers,
                seen,
            );
            if let MemTier::Threadgroup(buffer) = tier {
                record_threadgroup_budget_layout(
                    nodes,
                    extracted,
                    egraph,
                    *buffer,
                    children[0],
                    children[1],
                    None,
                    simdgroups,
                    simd_width,
                    buffers,
                );
            }
        }
        TensorIr::Simd(SimdNode::StoreIf { tier, children }) => {
            collect_threadgroup_layouts_in_value_subtree(
                nodes,
                extracted,
                egraph,
                children[3],
                simdgroups,
                simd_width,
                buffers,
                seen,
            );
            if let MemTier::Threadgroup(buffer) = tier {
                record_threadgroup_budget_layout(
                    nodes,
                    extracted,
                    egraph,
                    *buffer,
                    children[1],
                    children[2],
                    Some(children[0]),
                    simdgroups,
                    simd_width,
                    buffers,
                );
            }
        }
        TensorIr::Simd(SimdNode::Barrier { state, .. }) => {
            collect_threadgroup_layouts_in_value_subtree(
                nodes, extracted, egraph, *state, simdgroups, simd_width, buffers, seen,
            );
        }
        node => {
            for child in node.children() {
                collect_threadgroup_layouts_in_value_subtree(
                    nodes, extracted, egraph, *child, simdgroups, simd_width, buffers, seen,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn record_threadgroup_budget_load(
    nodes: &[TensorIr],
    extracted: &ExtractedProgram,
    egraph: &TensorEGraph,
    buffer: BufferRef,
    load: egg::Id,
    addr: egg::Id,
    simdgroups: u32,
    simd_width: u32,
    buffers: &mut HashMap<BufferRef, ThreadgroupBudgetLayout>,
) {
    let elements = {
        let mut memo = HashMap::new();
        expr_upper_bound_recexpr(nodes, addr, simdgroups, simd_width, &mut memo)
            .map(|max_addr| max_addr.saturating_add(1))
            .unwrap_or(4096)
            .max(1)
    };
    let dtype = extracted_node_dtype(extracted, egraph, load).unwrap_or(DType::F32);
    buffers
        .entry(buffer)
        .and_modify(|layout| {
            layout.elements = layout.elements.max(elements);
            if layout.dtype == DType::F32 {
                layout.dtype = dtype;
            }
        })
        .or_insert(ThreadgroupBudgetLayout { elements, dtype });
}

#[allow(clippy::too_many_arguments)]
fn record_threadgroup_budget_layout(
    nodes: &[TensorIr],
    extracted: &ExtractedProgram,
    egraph: &TensorEGraph,
    buffer: BufferRef,
    addr: egg::Id,
    value: egg::Id,
    cond: Option<egg::Id>,
    simdgroups: u32,
    simd_width: u32,
    buffers: &mut HashMap<BufferRef, ThreadgroupBudgetLayout>,
) {
    let elements = cond
        .and_then(|cond| guarded_store_extent_recexpr(nodes, addr, cond))
        .or_else(|| {
            let mut memo = HashMap::new();
            expr_upper_bound_recexpr(nodes, addr, simdgroups, simd_width, &mut memo)
                .map(|max_addr| max_addr.saturating_add(1))
        })
        .unwrap_or(4096)
        .max(1);
    let dtype = extracted_node_dtype(extracted, egraph, value).unwrap_or(DType::F32);
    buffers
        .entry(buffer)
        .and_modify(|layout| {
            layout.elements = layout.elements.max(elements);
            if layout.dtype == DType::F32 {
                layout.dtype = dtype;
            }
        })
        .or_insert(ThreadgroupBudgetLayout { elements, dtype });
}

fn extracted_node_dtype(
    extracted: &ExtractedProgram,
    egraph: &TensorEGraph,
    id: egg::Id,
) -> Option<DType> {
    extracted
        .eclass_for_node()
        .get(usize::from(id))
        .copied()
        .flatten()
        .and_then(|eclass| egraph[egraph.find(eclass)].data.dtype)
}

fn guarded_store_extent_recexpr(nodes: &[TensorIr], addr: egg::Id, cond: egg::Id) -> Option<u32> {
    match &nodes[usize::from(cond)] {
        TensorIr::BinOp(BinaryOp::Lt, [lhs, rhs]) if *lhs == addr => const_u32_recexpr(nodes, *rhs),
        _ => None,
    }
}

fn const_u32_recexpr(nodes: &[TensorIr], id: egg::Id) -> Option<u32> {
    match nodes.get(usize::from(id)) {
        Some(TensorIr::Const(ScalarValue::U32(value))) => Some(*value),
        _ => None,
    }
}

fn expr_upper_bound_recexpr(
    nodes: &[TensorIr],
    id: egg::Id,
    simdgroups: u32,
    simd_width: u32,
    memo: &mut HashMap<egg::Id, Option<u32>>,
) -> Option<u32> {
    if let Some(bound) = memo.get(&id) {
        return *bound;
    }
    let bound = match &nodes[usize::from(id)] {
        TensorIr::Const(ScalarValue::U32(value)) => Some(*value),
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_LANE,
            depth: 0,
        })) => Some(simd_width.saturating_sub(1)),
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_SIMDGROUP,
            depth: 0,
        })) => Some(simdgroups.saturating_sub(1)),
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_WORKGROUP,
            depth: 0,
        })) => None,
        TensorIr::BinOp(op, [lhs, rhs]) => {
            let lhs_bound = expr_upper_bound_recexpr(nodes, *lhs, simdgroups, simd_width, memo);
            let rhs_bound = expr_upper_bound_recexpr(nodes, *rhs, simdgroups, simd_width, memo);
            match op {
                BinaryOp::Add => lhs_bound.zip(rhs_bound).map(|(a, b)| a.saturating_add(b)),
                BinaryOp::Sub => lhs_bound,
                BinaryOp::Mul => lhs_bound.zip(rhs_bound).map(|(a, b)| a.saturating_mul(b)),
                BinaryOp::Div => lhs_bound
                    .zip(const_u32_recexpr(nodes, *rhs))
                    .and_then(|(lhs, rhs)| (rhs != 0).then_some(lhs / rhs)),
                BinaryOp::Mod => const_u32_recexpr(nodes, *rhs)
                    .map(|rhs| rhs.saturating_sub(1))
                    .zip(lhs_bound)
                    .map(|(rhs, lhs)| rhs.min(lhs)),
                BinaryOp::Min => lhs_bound.zip(rhs_bound).map(|(a, b)| a.min(b)),
                BinaryOp::Max => lhs_bound.zip(rhs_bound).map(|(a, b)| a.max(b)),
                _ => None,
            }
        }
        _ => None,
    };
    memo.insert(id, bound);
    bound
}

fn tuned_beam_config(summary: &TensorExprSummary, config: &StageConfig) -> BeamConfig {
    let mut beam = config.beam.clone();
    beam.device = config.runner.device;

    if summary.has_reduce && !summary.has_elementwise {
        beam.beam_width = beam.beam_width.max(32);
    }
    if summary.has_reduce && summary.has_elementwise {
        beam.beam_width = beam.beam_width.max(256);
    }

    beam
}

fn validate_tensor_expr_lowering_contract(
    expr: &TensorExprProgram,
) -> Result<TensorExprSummary, String> {
    let summary = expr.summary()?;

    if summary.output_shape.is_none() {
        return Err("backend lowering requires a tensor-shaped output".into());
    }
    if !matches!(summary.dtype, Some(DType::F16 | DType::F32 | DType::U32)) {
        return Err(format!(
            "Naga backend currently supports f16/f32/u32 tensor outputs, found {:?}",
            summary.dtype
        ));
    }
    for (idx, node) in expr.nodes().iter().enumerate() {
        if let TensorExprNode::Input { dtype, .. } = node
            && !matches!(dtype, DType::F16 | DType::F32 | DType::U32)
        {
            return Err(format!(
                "Naga backend currently supports f16/f32/u32 tensor inputs; input node {idx} has {dtype}"
            ));
        }
    }

    Ok(summary)
}

#[derive(Debug, Clone)]
struct MaterializedEffectBody {
    body: egg::Id,
    tensors: Vec<TensorId>,
    outputs: Vec<TensorId>,
}

fn tensor_id_for_eclass(id: egg::Id) -> TensorId {
    TensorId(u32::try_from(usize::from(id)).unwrap_or(u32::MAX))
}

fn add_tensor_marker(egraph: &mut TensorEGraph, tensor: TensorId) -> egg::Id {
    egraph.add(TensorIr::Const(ScalarValue::U32(tensor.0)))
}

/// Materialize executable `EffectNode::Program` alternatives into `egraph` and
/// return the effect-program root used for extraction.
#[doc(hidden)]
pub fn materialize_effect_programs_in_egraph(egraph: &mut TensorEGraph, root: egg::Id) -> egg::Id {
    let root = egraph.find(root);
    let mut memo = HashMap::new();
    let body = materialize_effect_body_for_eclass(egraph, root, &mut memo).unwrap_or_else(|| {
        let token = egraph.add(TensorIr::Effect(EffectNode::Token));
        MaterializedEffectBody {
            body: token,
            tensors: Vec::new(),
            outputs: Vec::new(),
        }
    });

    let mut declared = Vec::new();
    declared.extend(input_buffer_declarations(egraph));
    let mut seen = HashSet::new();
    for tensor in &body.tensors {
        if seen.insert(*tensor) {
            declared.push(add_tensor_marker(egraph, *tensor));
        }
    }
    let buffers = add_list(egraph, &declared);

    let outputs = if body.outputs.is_empty() {
        vec![add_tensor_marker(egraph, tensor_id_for_eclass(root))]
    } else {
        body.outputs
            .iter()
            .map(|tensor| add_tensor_marker(egraph, *tensor))
            .collect()
    };
    let outputs = add_list(egraph, &outputs);

    egraph.add(TensorIr::Effect(EffectNode::Program {
        children: [buffers, body.body, outputs],
    }))
}

fn input_buffer_declarations(egraph: &mut TensorEGraph) -> Vec<egg::Id> {
    let inputs = egraph
        .classes()
        .flat_map(|class| class.iter())
        .filter_map(|node| {
            if matches!(node, TensorIr::HighLevel(HighLevelNode::Input { .. })) {
                Some(node.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    inputs.into_iter().map(|node| egraph.add(node)).collect()
}

fn materialize_effect_body_for_eclass(
    egraph: &mut TensorEGraph,
    root: egg::Id,
    memo: &mut HashMap<egg::Id, Option<MaterializedEffectBody>>,
) -> Option<MaterializedEffectBody> {
    let root = egraph.find(root);
    if let Some(existing) = memo.get(&root) {
        return existing.clone();
    }
    memo.insert(root, None);

    let nodes = egraph[root].iter().cloned().collect::<Vec<_>>();
    let mut variants = Vec::new();
    for node in nodes {
        match node {
            TensorIr::Dispatch(dispatch @ DispatchNode::Dispatch { .. }) => {
                variants.push(materialize_effect_dispatch(egraph, root, &dispatch)?);
            }
            TensorIr::Dispatch(DispatchNode::Seq(list) | DispatchNode::Pipeline(list)) => {
                let children = extract_list(egraph, list);
                if let Some(body) = materialize_effect_sequence(egraph, &children, memo) {
                    variants.push(body);
                }
            }
            _ => {}
        }
    }

    let mut first = None;
    let mut tensors = Vec::new();
    let mut outputs: Option<Vec<TensorId>> = None;
    for variant in variants {
        if let Some(expected) = &outputs
            && *expected != variant.outputs
        {
            continue;
        }
        if let Some(base) = first {
            egraph.union(base, variant.body);
        } else {
            first = Some(variant.body);
            outputs = Some(variant.outputs.clone());
        }
        tensors.extend(variant.tensors);
    }

    let result = first.map(|body| MaterializedEffectBody {
        body: egraph.find(body),
        tensors,
        outputs: outputs.unwrap_or_default(),
    });
    memo.insert(root, result.clone());
    result
}

fn materialize_effect_sequence(
    egraph: &mut TensorEGraph,
    roots: &[egg::Id],
    memo: &mut HashMap<egg::Id, Option<MaterializedEffectBody>>,
) -> Option<MaterializedEffectBody> {
    let mut steps = Vec::new();
    let mut tensors = Vec::new();
    let mut outputs = Vec::new();
    for root in roots {
        let child = materialize_effect_body_for_eclass(egraph, *root, memo)?;
        steps.push(child.body);
        tensors.extend(child.tensors);
        outputs = child.outputs;
    }
    let body = match steps.as_slice() {
        [] => egraph.add(TensorIr::Effect(EffectNode::Token)),
        [single] => *single,
        _ => {
            let list = add_list(egraph, &steps);
            egraph.add(TensorIr::Effect(EffectNode::Seq(list)))
        }
    };
    Some(MaterializedEffectBody {
        body,
        tensors,
        outputs,
    })
}

fn materialize_effect_dispatch(
    egraph: &mut TensorEGraph,
    dispatch_eclass: egg::Id,
    dispatch: &DispatchNode,
) -> Option<MaterializedEffectBody> {
    let DispatchNode::Dispatch {
        workgroups,
        num_inputs,
        children_list,
    } = dispatch
    else {
        return None;
    };

    let children = extract_list(egraph, *children_list);
    let first_output = *num_inputs as usize;
    if children.len() <= first_output || !(children.len() - first_output).is_multiple_of(2) {
        return None;
    }
    let output_count = (children.len() - first_output) / 2;
    let dispatch_dtype_bytes = egraph[egraph.find(*children_list)].data.dtype_bytes;
    if output_count > 1
        && (dispatch_dtype_bytes == Some(DType::F16.byte_size())
            || children[first_output..]
                .chunks_exact(2)
                .any(|pair| egraph[egraph.find(pair[0])].data.dtype == Some(DType::F16)))
    {
        return None;
    }

    let tensor = tensor_id_for_eclass(dispatch_eclass);
    let mut state = egraph.add(TensorIr::Effect(EffectNode::Token));
    for pair in children[first_output..].chunks_exact(2) {
        let value = pair[0];
        let addr = pair[1];
        state = if let Some(elements) = egraph[dispatch_eclass]
            .data
            .shape
            .as_ref()
            .and_then(Shape::static_numel)
        {
            let elements = egraph.add(TensorIr::Const(ScalarValue::U32(elements)));
            let in_bounds = egraph.add(TensorIr::BinOp(BinaryOp::Lt, [addr, elements]));
            egraph.add(TensorIr::Effect(EffectNode::StoreIf {
                tier: MemTier::Device(BufferRef::Tensor(tensor)),
                children: [in_bounds, addr, value, state],
            }))
        } else {
            egraph.add(TensorIr::Effect(EffectNode::Store {
                tier: MemTier::Device(BufferRef::Tensor(tensor)),
                children: [addr, value, state],
            }))
        };
    }

    let dispatch_state = egraph.add(TensorIr::Effect(EffectNode::Token));
    let body = egraph.add(TensorIr::Effect(EffectNode::Dispatch {
        workgroups: workgroups.clone(),
        simdgroups: 1,
        children: [dispatch_state, state],
    }));
    Some(MaterializedEffectBody {
        body,
        tensors: vec![tensor],
        outputs: vec![tensor],
    })
}

fn extracted_program_from_egraph_candidate(
    egraph: &TensorEGraph,
    cost: f64,
    expr: RecExpr<TensorIr>,
) -> ExtractedProgram {
    let eclass_for_node = eclass_map_for_extracted_candidate(egraph, &expr);
    ExtractedProgram {
        expr,
        eclass_for_node,
        cost,
    }
}

fn eclass_map_for_extracted_candidate(
    egraph: &TensorEGraph,
    extracted: &RecExpr<TensorIr>,
) -> Vec<Option<egg::Id>> {
    let nodes = extracted.as_ref();
    let mut eclasses = Vec::with_capacity(nodes.len());
    for (idx, node) in nodes.iter().enumerate() {
        let mut lookup_node = node.clone();
        let mut missing_child = false;
        for child in lookup_node.children_mut() {
            let Some(Some(mapped)) = eclasses.get(usize::from(*child)).copied() else {
                missing_child = true;
                break;
            };
            *child = mapped;
        }
        if missing_child {
            eclasses.push(None);
            continue;
        }
        let found = egraph.lookup(lookup_node).map(|id| egraph.find(id));
        debug_assert!(
            found.is_some(),
            "extracted node {idx} was not found in the saturated e-graph"
        );
        eclasses.push(found);
    }
    eclasses
}

pub(crate) fn validate_kernel_expr(expr: &RecExpr<TensorIr>) -> Result<(), String> {
    let nodes = expr.as_ref();
    let Some(root) = nodes.last() else {
        return Err("kernel program is empty".into());
    };
    if !matches!(root, TensorIr::Effect(EffectNode::Program { .. })) {
        return Err(format!(
            "kernel root must be an effect program, found {root:?}"
        ));
    }
    for (idx, node) in nodes.iter().enumerate() {
        for child in node.children() {
            if usize::from(*child) >= idx {
                return Err(format!(
                    "kernel node {idx} has non-topological child {}",
                    usize::from(*child)
                ));
            }
        }
    }
    validate_program_var_scopes(nodes, nodes.len() - 1)?;
    validate_tuple_extracts(nodes)?;
    validate_effect_program(nodes, nodes.len() - 1)?;
    Ok(())
}

fn validate_tuple_extracts(nodes: &[TensorIr]) -> Result<(), String> {
    let mut memo = HashMap::new();
    for (idx, node) in nodes.iter().enumerate() {
        if let TensorIr::Dispatch(DispatchNode::Extract { index, tuple }) = node
            && let Some(arity) = tuple_arity(nodes, *tuple, &mut memo)
            && (*index as usize) >= arity
        {
            return Err(format!(
                "extract node {idx} reads tuple slot {index}, but tuple arity is {arity}"
            ));
        }
    }
    Ok(())
}

fn tuple_arity(
    nodes: &[TensorIr],
    id: egg::Id,
    memo: &mut HashMap<egg::Id, Option<usize>>,
) -> Option<usize> {
    if let Some(arity) = memo.get(&id) {
        return *arity;
    }
    let arity = match &nodes[usize::from(id)] {
        TensorIr::Dispatch(DispatchNode::Pack { children_list }) => {
            Some(extract_recexpr_list(nodes, *children_list).len())
        }
        TensorIr::Simd(SimdNode::Theta {
            children: [init, ..],
        }) => tuple_arity(nodes, *init, memo),
        TensorIr::Dispatch(DispatchNode::Extract { tuple, .. }) => {
            tuple_arity(nodes, *tuple, memo).and_then(|arity| (arity > 0).then_some(1))
        }
        _ => Some(1),
    };
    memo.insert(id, arity);
    arity
}

#[derive(Debug, Default)]
struct EffectValidation {
    declared_tensors: HashSet<TensorId>,
    stored_tensors: HashSet<TensorId>,
    output_tensors: Vec<TensorId>,
    dispatches: usize,
}

#[derive(Debug, Default)]
struct DispatchEffectValidation {
    threadgroup_stores: HashSet<BufferRef>,
    threadgroup_barriers: HashSet<BufferRef>,
}

fn validate_effect_program(nodes: &[TensorIr], root_idx: usize) -> Result<(), String> {
    let TensorIr::Effect(EffectNode::Program {
        children: [buffers, body, outputs],
    }) = &nodes[root_idx]
    else {
        return Err("kernel root is not an effect program".into());
    };

    let mut ctx = EffectValidation::default();
    for buffer in extract_recexpr_list(nodes, *buffers) {
        if let Some(tensor) = tensor_marker(nodes, buffer) {
            ctx.declared_tensors.insert(tensor);
            continue;
        };
        if matches!(
            nodes.get(usize::from(buffer)),
            Some(TensorIr::HighLevel(HighLevelNode::Input { .. }))
        ) {
            continue;
        }
        return Err(format!(
            "program buffer declaration {buffer:?} is not a tensor id marker or external input"
        ));
    }
    for output in extract_recexpr_list(nodes, *outputs) {
        let Some(tensor) = tensor_marker(nodes, output) else {
            return Err(format!(
                "program output {output:?} is not a tensor id marker"
            ));
        };
        ctx.output_tensors.push(tensor);
    }

    validate_effect_step(nodes, *body, &mut ctx)?;

    if ctx.dispatches == 0 {
        return Err("effect program contains no dispatch".into());
    }
    for output in &ctx.output_tensors {
        if !ctx.declared_tensors.contains(output) {
            return Err(format!(
                "program output {output} is not declared as a buffer"
            ));
        }
        if !ctx.stored_tensors.contains(output) {
            return Err(format!("program output {output} is never stored"));
        }
    }

    Ok(())
}

fn tensor_marker(nodes: &[TensorIr], id: egg::Id) -> Option<TensorId> {
    match nodes.get(usize::from(id)) {
        Some(TensorIr::Const(ScalarValue::U32(value))) => Some(TensorId(*value)),
        _ => None,
    }
}

fn validate_effect_step(
    nodes: &[TensorIr],
    step: egg::Id,
    ctx: &mut EffectValidation,
) -> Result<(), String> {
    match &nodes[usize::from(step)] {
        TensorIr::Effect(EffectNode::Seq(list_id)) => {
            for child in extract_recexpr_list(nodes, *list_id) {
                validate_effect_step(nodes, child, ctx)?;
            }
        }
        TensorIr::Effect(EffectNode::Dispatch {
            simdgroups,
            children: [_state, body],
            ..
        }) => {
            if *simdgroups == 0 {
                return Err("effect dispatch has zero simdgroups".into());
            }
            ctx.dispatches += 1;
            let mut dispatch_ctx = DispatchEffectValidation::default();
            validate_effect_chain(nodes, *body, ctx, &mut dispatch_ctx)?;
        }
        TensorIr::Effect(EffectNode::Token) => {}
        TensorIr::Effect(
            EffectNode::Store { .. } | EffectNode::StoreIf { .. } | EffectNode::Barrier { .. },
        ) => {
            return Err(format!(
                "effect node {:?} appears outside a dispatch body",
                nodes[usize::from(step)]
            ));
        }
        other => return Err(format!("program body contains non-effect node {other:?}")),
    }
    Ok(())
}

fn validate_effect_chain(
    nodes: &[TensorIr],
    state: egg::Id,
    ctx: &mut EffectValidation,
    dispatch_ctx: &mut DispatchEffectValidation,
) -> Result<(), String> {
    match &nodes[usize::from(state)] {
        TensorIr::Effect(EffectNode::Token) => Ok(()),
        TensorIr::Effect(EffectNode::Store { tier, children }) => {
            let [addr, value, previous] = *children;
            validate_effect_chain(nodes, previous, ctx, dispatch_ctx)?;
            validate_value_memory(nodes, addr, ctx, dispatch_ctx)?;
            validate_value_memory(nodes, value, ctx, dispatch_ctx)?;
            validate_store_target(*tier, ctx, dispatch_ctx)
        }
        TensorIr::Effect(EffectNode::StoreIf { tier, children }) => {
            let [cond, addr, value, previous] = *children;
            validate_effect_chain(nodes, previous, ctx, dispatch_ctx)?;
            validate_value_memory(nodes, addr, ctx, dispatch_ctx)?;
            validate_value_memory(nodes, value, ctx, dispatch_ctx)?;
            validate_value_memory(nodes, cond, ctx, dispatch_ctx)?;
            validate_store_target(*tier, ctx, dispatch_ctx)
        }
        TensorIr::Effect(EffectNode::Barrier { regions, state }) => {
            validate_effect_chain(nodes, *state, ctx, dispatch_ctx)?;
            for region in regions {
                if !dispatch_ctx.threadgroup_stores.contains(region) {
                    return Err(format!(
                        "threadgroup barrier references {region} before any store"
                    ));
                }
                dispatch_ctx.threadgroup_barriers.insert(*region);
            }
            Ok(())
        }
        other => Err(format!("dispatch body contains non-state node {other:?}")),
    }
}

fn validate_store_target(
    tier: MemTier,
    ctx: &mut EffectValidation,
    dispatch_ctx: &mut DispatchEffectValidation,
) -> Result<(), String> {
    match tier {
        MemTier::Device(BufferRef::Tensor(tensor)) => {
            if !ctx.declared_tensors.contains(&tensor) {
                return Err(format!("store targets undeclared tensor buffer {tensor}"));
            }
            ctx.stored_tensors.insert(tensor);
            Ok(())
        }
        MemTier::Device(BufferRef::External(index) | BufferRef::Input(index)) => Err(format!(
            "store targets read-only external input buffer {index}"
        )),
        MemTier::Device(BufferRef::Output(index)) => Err(format!(
            "store targets implicit output buffer {index}; use BufferRef::Tensor"
        )),
        MemTier::Threadgroup(buffer) => {
            dispatch_ctx.threadgroup_stores.insert(buffer);
            Ok(())
        }
    }
}

fn validate_value_memory(
    nodes: &[TensorIr],
    root: egg::Id,
    ctx: &EffectValidation,
    dispatch_ctx: &DispatchEffectValidation,
) -> Result<(), String> {
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        let node = &nodes[usize::from(id)];
        match node {
            TensorIr::Simd(SimdNode::Load { tier, .. }) => {
                validate_load_source(*tier, ctx, dispatch_ctx)?;
            }
            TensorIr::Simd(SimdNode::Store { tier, .. })
            | TensorIr::Simd(SimdNode::StoreIf { tier, .. }) => {
                if tier.is_device() {
                    return Err(format!(
                        "value subtree contains legacy device store {tier}; use EffectNode::Store"
                    ));
                }
            }
            _ => {}
        }
        for child in node.children() {
            stack.push(*child);
        }
    }
    Ok(())
}

fn validate_load_source(
    tier: MemTier,
    ctx: &EffectValidation,
    dispatch_ctx: &DispatchEffectValidation,
) -> Result<(), String> {
    match tier {
        MemTier::Device(BufferRef::Tensor(tensor)) => {
            if !ctx.declared_tensors.contains(&tensor) {
                return Err(format!("load reads undeclared tensor buffer {tensor}"));
            }
            if !ctx.stored_tensors.contains(&tensor) {
                return Err(format!(
                    "load reads tensor buffer {tensor} before it is stored"
                ));
            }
            Ok(())
        }
        MemTier::Device(BufferRef::External(_) | BufferRef::Input(_) | BufferRef::Output(_)) => {
            Ok(())
        }
        MemTier::Threadgroup(buffer) => {
            if !dispatch_ctx.threadgroup_stores.is_empty()
                && !dispatch_ctx.threadgroup_barriers.contains(&buffer)
            {
                return Err(format!(
                    "threadgroup load reads {buffer} before a covering barrier"
                ));
            }
            Ok(())
        }
    }
}

fn validate_program_var_scopes(nodes: &[TensorIr], root_idx: usize) -> Result<(), String> {
    let mut stack = vec![(root_idx, 0_u32, 0_u32)];
    while let Some((idx, theta_depth, dispatch_depth)) = stack.pop() {
        match &nodes[idx] {
            TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Theta,
                depth,
                ..
            })) if *depth >= theta_depth => {
                return Err(format!("unbound theta variable at node {idx}"));
            }
            TensorIr::Simd(SimdNode::Var(VarRef::Bound {
                kind: BinderKind::Dispatch,
                depth,
                ..
            })) if *depth >= dispatch_depth => {
                return Err(format!("unbound dispatch variable at node {idx}"));
            }
            TensorIr::Simd(SimdNode::Theta {
                children: [init, count, update],
            }) => {
                stack.push((usize::from(*init), theta_depth, dispatch_depth));
                stack.push((usize::from(*count), theta_depth, dispatch_depth));
                stack.push((usize::from(*update), theta_depth + 1, dispatch_depth));
            }
            TensorIr::Dispatch(DispatchNode::Dispatch { children_list, .. }) => {
                stack.push((usize::from(*children_list), theta_depth, dispatch_depth + 1));
            }
            TensorIr::Effect(EffectNode::Dispatch {
                children: [state, body],
                ..
            }) => {
                stack.push((usize::from(*state), theta_depth, dispatch_depth));
                stack.push((usize::from(*body), theta_depth, dispatch_depth + 1));
            }
            node => {
                for child in node.children() {
                    stack.push((usize::from(*child), theta_depth, dispatch_depth));
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod effect_validation_tests {
    use super::*;
    use crate::types::{Dim, ScalarValue};

    fn list(expr: &mut RecExpr<TensorIr>, items: &[egg::Id]) -> egg::Id {
        let mut curr = expr.add(TensorIr::Nil);
        for &item in items.iter().rev() {
            curr = expr.add(TensorIr::Cons([item, curr]));
        }
        curr
    }

    #[test]
    fn validate_program_rejects_unbound_theta_var() {
        let mut expr = RecExpr::default();
        let tensor = expr.add(TensorIr::Const(ScalarValue::U32(0)));
        let nil = expr.add(TensorIr::Nil);
        let buffers = expr.add(TensorIr::Cons([tensor, nil]));
        let token = expr.add(TensorIr::Effect(EffectNode::Token));
        let value = expr.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(0))));
        let addr = expr.add(TensorIr::Const(ScalarValue::U32(0)));
        let store = expr.add(TensorIr::Effect(EffectNode::Store {
            tier: MemTier::Device(BufferRef::Tensor(TensorId(0))),
            children: [addr, value, token],
        }));
        let dispatch = expr.add(TensorIr::Effect(EffectNode::Dispatch {
            workgroups: Dim::Const(1),
            simdgroups: 1,
            children: [token, store],
        }));
        let outputs = list(&mut expr, &[tensor]);
        expr.add(TensorIr::Effect(EffectNode::Program {
            children: [buffers, dispatch, outputs],
        }));

        let err = validate_kernel_expr(&expr).expect_err("unbound theta var should reject");
        assert!(err.contains("unbound theta variable"));
    }

    #[test]
    fn validate_program_rejects_missing_output_store() {
        let mut expr = RecExpr::default();
        let tensor = expr.add(TensorIr::Const(ScalarValue::U32(0)));
        let buffers = list(&mut expr, &[tensor]);
        let token = expr.add(TensorIr::Effect(EffectNode::Token));
        let dispatch = expr.add(TensorIr::Effect(EffectNode::Dispatch {
            workgroups: Dim::Const(1),
            simdgroups: 1,
            children: [token, token],
        }));
        let outputs = list(&mut expr, &[tensor]);
        expr.add(TensorIr::Effect(EffectNode::Program {
            children: [buffers, dispatch, outputs],
        }));

        let err = validate_kernel_expr(&expr).expect_err("missing output store should reject");
        assert!(err.contains("never stored"));
    }
}
