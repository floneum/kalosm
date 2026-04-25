use std::time::Duration;

use egg::{Language, RecExpr};

use crate::TensorEGraph;
use crate::extractor::BeamConfig;
use crate::language::{DispatchNode, HighLevelNode, TensorIr};
use crate::rules::{RunnerConfig, SaturationReport, saturate_phases, saturate_phases_reported};
use crate::skeleton::{
    CandidateValidationReport, DispatchProgram, build_dispatch_program_from_extracted,
};
use crate::stages::{Phase, TensorExprNode, TensorExprProgram, TensorExprSummary};
use crate::types::{DType, DeviceProfile, LoweringOptions};

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
    pub budget_rejections: usize,
    pub selected_candidate_index: Option<usize>,
    pub selected_cost: Option<f64>,
    pub selected_nodes: Option<usize>,
    pub error: Option<String>,
    pub elapsed: Duration,
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
    extracted: RecExpr<TensorIr>,
    egraph: TensorEGraph,
    cost: f64,
    /// Device profile this kernel was lowered for. Carried so that
    /// downstream stages (`compile_kernel`, codegen) keep using the same
    /// device parameters that lowering optimized against.
    device: DeviceProfile,
    /// Lowering toggles carried from the runner config so the skeleton
    /// stays consistent with the structural choices already made (e.g.
    /// whether to unroll inner loops).
    lowering: LoweringOptions,
}

impl KernelProgram {
    #[must_use]
    pub const fn root(&self) -> egg::Id {
        self.root
    }

    #[must_use]
    pub const fn extracted(&self) -> &RecExpr<TensorIr> {
        &self.extracted
    }

    #[must_use]
    pub const fn egraph(&self) -> &TensorEGraph {
        &self.egraph
    }

    #[must_use]
    pub const fn cost(&self) -> f64 {
        self.cost
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

/// Stage 4: backend-facing SIMD/dispatch skeleton ready for codegen.
#[derive(Debug)]
pub struct SimdProgram {
    kernel: KernelProgram,
    dispatch_program: DispatchProgram,
}

impl SimdProgram {
    #[must_use]
    pub const fn kernel(&self) -> &KernelProgram {
        &self.kernel
    }

    #[must_use]
    pub const fn dispatch_program(&self) -> &DispatchProgram {
        &self.dispatch_program
    }

    #[must_use]
    pub fn into_dispatch_program(self) -> DispatchProgram {
        self.dispatch_program
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

    /// # Errors
    ///
    /// Returns an error if the validated kernel cannot be compiled into a dispatch program.
    pub fn compile(&self, kernel: KernelProgram) -> Result<SimdProgram, String> {
        compile_kernel(kernel)
    }

    /// # Errors
    ///
    /// Returns an error if any staged lowering step fails.
    pub fn build(&self, expr: &TensorExprProgram) -> Result<SimdProgram, String> {
        let kernel = self.lower(expr)?;
        self.compile(kernel)
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

    let (egraph, saturation) = saturate_phases_reported(egraph, Phase::all(), &config.runner);
    report.saturation = saturation;
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
            root,
            &beam,
            &config.runner.device,
            config.runner.lowering,
            config
                .candidate_limit
                .unwrap_or_else(|| beam.beam_width.max(16)),
        )
    });
    report.extraction = Some(extraction_report);
    let (cost, extracted) = match candidate_result {
        Ok(candidate) => candidate,
        Err(message) => {
            report.error = Some(message.clone());
            return Err(LoweringError::new(message, report));
        }
    };
    tracing::info!(
        cost,
        extracted_nodes = extracted.as_ref().len(),
        "extraction complete"
    );

    if let Err(message) = validate_kernel_expr(&extracted) {
        report.error = Some(message.clone());
        return Err(LoweringError::new(message, report));
    }
    if let Err(message) = enforce_device_budget(
        &extracted,
        &egraph,
        &config.runner.device,
        config.runner.lowering,
    ) {
        report.error = Some(message.clone());
        return Err(LoweringError::new(message, report));
    }

    Ok((
        KernelProgram {
            root,
            extracted,
            egraph,
            cost,
            device: config.runner.device,
            lowering: config.runner.lowering,
        },
        report,
    ))
}

/// Build the dispatch program for the extracted kernel and confirm it fits in
/// the target device's threadgroup memory budget. `TgBufferInfo` carries its
/// own `dtype_bytes` so the byte total is exact.
fn enforce_device_budget(
    extracted: &RecExpr<TensorIr>,
    egraph: &TensorEGraph,
    device: &DeviceProfile,
    lowering: LoweringOptions,
) -> Result<(), String> {
    let program =
        build_dispatch_program_from_extracted(extracted, egraph.clone(), device, &lowering);
    let peak = program.peak_threadgroup_bytes();
    let budget = u64::from(device.max_threadgroup_bytes);
    if peak > budget {
        return Err(format!(
            "kernel exceeds device threadgroup budget: needs {peak} bytes, \
             device allows {budget} bytes (DeviceProfile::max_threadgroup_bytes)"
        ));
    }
    Ok(())
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

    let egraph = saturate_phases(egraph, Phase::all(), &config.runner);
    let beam = tuned_beam_config(&summary, config);
    let candidates = crate::skeleton::beam_extract_valid_candidates(
        &egraph,
        root,
        &beam,
        &config.runner.device,
        &config.runner.lowering,
        limit,
    );

    let mut kernels = Vec::new();
    for (cost, extracted) in candidates {
        if validate_kernel_expr(&extracted).is_err() {
            continue;
        }
        if enforce_device_budget(
            &extracted,
            &egraph,
            &config.runner.device,
            config.runner.lowering,
        )
        .is_err()
        {
            continue;
        }
        kernels.push(KernelProgram {
            root,
            extracted,
            egraph: egraph.clone(),
            cost,
            device: config.runner.device,
            lowering: config.runner.lowering,
        });
    }
    Ok(kernels)
}

fn select_kernel_candidate_with_report(
    egraph: &TensorEGraph,
    root: egg::Id,
    beam: &BeamConfig,
    device: &DeviceProfile,
    lowering: LoweringOptions,
    candidate_limit: usize,
) -> (Result<(f64, RecExpr<TensorIr>), String>, ExtractionReport) {
    let start = std::time::Instant::now();
    let (candidates, candidate_validation) =
        crate::skeleton::beam_extract_valid_candidates_with_report(
            egraph,
            root,
            beam,
            device,
            &lowering,
            candidate_limit,
        );
    let mut report = ExtractionReport {
        beam_width: beam.beam_width,
        candidate_limit,
        candidate_validation,
        budget_rejections: 0,
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

    let mut last_budget_error = None;
    for (index, (cost, extracted)) in candidates.into_iter().enumerate() {
        match enforce_device_budget(&extracted, egraph, device, lowering) {
            Ok(()) => {
                report.selected_candidate_index = Some(index);
                report.selected_cost = Some(cost);
                report.selected_nodes = Some(extracted.as_ref().len());
                report.elapsed = start.elapsed();
                return (Ok((cost, extracted)), report);
            }
            Err(message) => {
                report.budget_rejections += 1;
                last_budget_error = Some(message);
            }
        }
    }

    let message = format!(
        "no valid executable kernel candidates fit the device threadgroup budget of {} bytes",
        device.max_threadgroup_bytes
    );
    report.error = Some(last_budget_error.unwrap_or_else(|| message.clone()));
    report.elapsed = start.elapsed();
    (Err(message), report)
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

/// Compile a validated kernel-stage program into a backend dispatch skeleton.
///
/// # Errors
///
/// Returns an error if the extracted kernel produces an invalid dispatch program.
pub fn compile_kernel(kernel: KernelProgram) -> Result<SimdProgram, String> {
    let dispatch_program = build_dispatch_program_from_extracted(
        kernel.extracted(),
        kernel.egraph.clone(),
        &kernel.device,
        &kernel.lowering,
    );
    validate_dispatch_program(&dispatch_program)?;
    crate::verify(&dispatch_program).map_err(|e| format!("dispatch verification failed: {e}"))?;
    Ok(SimdProgram {
        kernel,
        dispatch_program,
    })
}

fn validate_kernel_expr(expr: &RecExpr<TensorIr>) -> Result<(), String> {
    let nodes = expr.as_ref();
    let Some(root) = nodes.last() else {
        return Err("kernel program is empty".into());
    };
    if !matches!(
        root,
        TensorIr::Dispatch(
            DispatchNode::Dispatch { .. } | DispatchNode::Seq(_) | DispatchNode::Pipeline(_)
        )
    ) {
        return Err(format!("kernel root must be a dispatch, found {root:?}"));
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

        if matches!(
            node,
            TensorIr::HighLevel(
                HighLevelNode::Restride { .. }
                    | HighLevelNode::Elementwise { .. }
                    | HighLevelNode::Reduce { .. }
            )
        ) {
            return Err(format!(
                "kernel stage must not contain semantic tensor node {node:?}"
            ));
        }
    }

    Ok(())
}

fn validate_dispatch_program(program: &DispatchProgram) -> Result<(), String> {
    if program.dispatches.is_empty() {
        return Err("compiled SIMD program has no dispatches".into());
    }
    for (idx, dispatch) in program.dispatches.iter().enumerate() {
        if dispatch.outputs.is_empty() {
            return Err(format!("dispatch {idx} has no outputs"));
        }
        if dispatch.simdgroups == 0 {
            return Err(format!("dispatch {idx} has zero simdgroups"));
        }
    }
    Ok(())
}
