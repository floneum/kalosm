#![cfg_attr(
    not(test),
    warn(
        clippy::all,
        clippy::pedantic,
        clippy::nursery,
        clippy::cargo,
        clippy::dbg_macro,
        clippy::todo,
        clippy::unimplemented,
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::unreachable,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::exit,
        clippy::mem_forget,
        clippy::as_underscore,
        clippy::allow_attributes_without_reason,
        clippy::rest_pat_in_fully_bound_structs,
        clippy::str_to_string,
        clippy::verbose_file_reads,
        clippy::lossy_float_literal,
        clippy::float_cmp_const,
        clippy::modulo_arithmetic,
        rust_2018_idioms,
        unreachable_pub,
        trivial_numeric_casts,
        trivial_casts,
        unused_lifetimes,
        unused_qualifications,
        unsafe_code,
        single_use_lifetimes
    )
)]
#![allow(
    clippy::multiple_crate_versions,
    reason = "the facade crate intentionally integrates several external dependency trees"
)]
//! Tensor IR: staged tensor-to-kernel lowering for GPU code generation.
//!
//! The public architecture is split into explicit stages:
//! - `TensorExprProgram`: pure tensor semantics
//! - `KernelProgram`: validated effectful e-graph program form
//!
//! Internally, lowering uses an e-graph representation as a backend
//! implementation detail separate from the public architectural model.

pub mod analysis {
    pub use tensor_ir_egraph::analysis::*;
}
pub mod applier {
    pub use tensor_ir_egraph::applier::*;
}
pub mod binding {
    pub use tensor_ir_egraph::binding::*;
}
pub mod builders {
    pub use tensor_ir_egraph::builders::*;
}
pub mod extractor {
    pub use tensor_ir_egraph::extractor::*;
}
pub mod language {
    pub use tensor_ir_egraph::language::*;
}
pub mod naga_codegen {
    pub use tensor_ir_codegen_naga::*;
}
pub mod pipeline;
pub mod rules {
    use egg::{BackoffScheduler, Runner};

    use crate::TensorAnalysis;
    use crate::language::TensorIr;

    pub use tensor_ir_frontend::Phase;
    pub use tensor_ir_opt::rules::{RunnerConfig, SaturationPhaseReport, SaturationReport};

    #[must_use]
    pub fn all_rules(config: &RunnerConfig) -> Vec<egg::Rewrite<TensorIr, TensorAnalysis>> {
        let mut rules = tensor_ir_opt::rules::all_rules(config);
        rules.extend(tensor_ir_dispatch::state_threading_rules(
            config.device,
            config.lowering,
        ));
        rules
    }

    #[must_use]
    pub fn saturate(
        egraph: egg::EGraph<TensorIr, TensorAnalysis>,
        config: &RunnerConfig,
    ) -> egg::EGraph<TensorIr, TensorAnalysis> {
        let rules = all_rules(config);
        let runner = Runner::default()
            .with_egraph(egraph)
            .with_iter_limit(config.iter_limit)
            .with_node_limit(config.node_limit)
            .with_time_limit(std::time::Duration::from_secs(config.time_limit_secs))
            .with_scheduler(BackoffScheduler::default())
            .run(&rules);

        runner.egraph
    }

    #[must_use]
    pub fn saturate_phases(
        egraph: egg::EGraph<TensorIr, TensorAnalysis>,
        phases: &[Phase],
        config: &RunnerConfig,
    ) -> egg::EGraph<TensorIr, TensorAnalysis> {
        saturate_phases_reported(egraph, phases, config).0
    }

    #[must_use]
    pub fn saturate_phases_reported(
        mut egraph: egg::EGraph<TensorIr, TensorAnalysis>,
        phases: &[Phase],
        config: &RunnerConfig,
    ) -> (egg::EGraph<TensorIr, TensorAnalysis>, SaturationReport) {
        if !phases.contains(&Phase::StateThreading) {
            return tensor_ir_opt::rules::saturate_phases_reported(egraph, phases, config);
        }

        let mut report = SaturationReport::default();
        for phase in phases {
            match phase {
                Phase::StateThreading => {
                    egraph.rebuild();
                    let rules =
                        tensor_ir_dispatch::state_threading_rules(config.device, config.lowering);
                    let nodes_before = egraph.total_size();
                    let classes_before = egraph.number_of_classes();
                    let start = std::time::Instant::now();
                    let runner = Runner::default()
                        .with_egraph(egraph)
                        .with_iter_limit(config.iter_limit)
                        .with_node_limit(config.node_limit)
                        .with_time_limit(std::time::Duration::from_secs(config.time_limit_secs))
                        .with_scheduler(BackoffScheduler::default())
                        .run(&rules);
                    let elapsed = start.elapsed();
                    let iterations = runner.iterations.len();
                    let nodes_after = runner.egraph.total_size();
                    let classes_after = runner.egraph.number_of_classes();
                    let stop_reason = format!("{:?}", runner.stop_reason);
                    report.phases.push(SaturationPhaseReport {
                        phase: *phase,
                        rule_count: rules.len(),
                        iter_limit: config.iter_limit,
                        node_limit: config.node_limit,
                        time_limit_secs: config.time_limit_secs,
                        iterations,
                        nodes_before,
                        nodes_after,
                        classes_before,
                        classes_after,
                        stop_reason,
                        elapsed,
                    });
                    egraph = runner.egraph;
                }
                _ => {
                    let (next, phase_report) = tensor_ir_opt::rules::saturate_phases_reported(
                        egraph,
                        std::slice::from_ref(phase),
                        config,
                    );
                    report.extend(phase_report);
                    egraph = next;
                }
            }
        }
        (egraph, report)
    }
}
#[cfg(feature = "runtime")]
pub mod runtime {
    pub use tensor_ir_runtime_wgpu::*;
}
pub mod stages {
    pub use tensor_ir_frontend::stages::*;
}
pub mod types {
    pub use tensor_ir_frontend::types::*;
}

pub use analysis::{TensorAnalysis, TensorData};
pub use builders::IrBuilder;
pub use extractor::{
    BeamConfig, SyntheticCostModel, beam_extract, beam_extract_candidates, greedy_extract,
};
#[doc(hidden)]
pub use language::{DispatchNode, HighLevelNode, SimdNode, TensorIr};
pub use naga_codegen::{module_to_msl, module_to_wgsl};
pub use pipeline::{
    CandidateValidationReport, ExtractedProgram, ExtractionReport, KernelProgram, LoweringError,
    LoweringReport, StageConfig, StagedPipeline, lower_tensor_expr, lower_tensor_expr_candidates,
    lower_tensor_expr_with_report, tensor_expr_to_recexpr,
};
pub use rules::{
    Phase, RunnerConfig, SaturationPhaseReport, SaturationReport, all_rules, saturate,
    saturate_phases, saturate_phases_reported,
};
#[cfg(feature = "runtime")]
pub use runtime::{GpuBenchmarkResult, GpuContext, ProgramBenchmarkConfig};
pub use stages::{ExprId, TensorExprBuilder, TensorExprNode, TensorExprProgram, TensorExprSummary};
pub use tensor_ir_egraph::{TensorEGraph, TensorRewrite, TensorRunner};
pub use types::*;

/// Witness that a [`KernelProgram`] has passed program verification.
pub struct VerifiedProgram<'a> {
    kernel: &'a KernelProgram,
}

impl<'a> VerifiedProgram<'a> {
    #[must_use]
    pub const fn kernel(&self) -> &'a KernelProgram {
        self.kernel
    }
}

/// Verify an effectful [`KernelProgram`].
///
/// # Errors
///
/// Returns a verification error if the chosen extraction has unbound vars,
/// malformed tuple extracts, or a recursive chosen-node cycle.
pub fn verify_program(kernel: &KernelProgram) -> Result<VerifiedProgram<'_>, String> {
    pipeline::validate_kernel_expr(kernel.extracted())?;
    pipeline::enforce_effect_threadgroup_budget(
        kernel.extracted_program(),
        kernel.egraph(),
        kernel.device(),
    )?;
    Ok(VerifiedProgram { kernel })
}

/// Lower a verified effectful program to Naga.
///
/// # Errors
///
/// Returns a codegen error if the verified effect program is malformed for
/// the Naga backend.
pub fn lower_program(verified: VerifiedProgram<'_>) -> Result<naga::Module, String> {
    tensor_ir_codegen_naga::lower_effect_program(
        verified.kernel.extracted(),
        verified.kernel.device(),
    )
}

/// Verify, lower, and emit WGSL from an effectful [`KernelProgram`].
///
/// # Errors
///
/// Returns an error if verification, Naga validation, or WGSL writing fails.
pub fn lower_to_wgsl(kernel: &KernelProgram) -> Result<String, String> {
    let verified =
        verify_program(kernel).map_err(|error| format!("verification error: {error}"))?;
    let module = lower_program(verified).map_err(|error| format!("codegen error: {error}"))?;
    module_to_wgsl(&module)
}

/// Verify, lower, and emit MSL from an effectful [`KernelProgram`].
///
/// # Errors
///
/// Returns an error if verification, Naga validation, or MSL writing fails.
pub fn lower_to_msl(kernel: &KernelProgram) -> Result<String, String> {
    let verified =
        verify_program(kernel).map_err(|error| format!("verification error: {error}"))?;
    let module = lower_program(verified).map_err(|error| format!("codegen error: {error}"))?;
    module_to_msl(&module)
}

#[cfg(test)]
mod tests;
