//! Rewrite rules for the tensor IR e-graph.
//!
//! Rules are split into two runtime phases:
//! - [`Phase::Lowering`]: High-level tensor ops → generic, unblocked
//!   per-lane Dispatches. Only rules in [`phase1`] run here.
//! - [`Phase::LateDispatch`]: Unified shape saturation. All rules in
//!   [`phase2`]–[`phase6`] run together so the extractor's shape-aware
//!   cost model can compare tile size, register blocking, TG promotion,
//!   cooperative split, shuffle tree, and fusion variants side-by-side.
//!
//! The `phaseN` modules preserve topic-grouping for readability; they
//! are not exposed as standalone `Phase` variants.

pub mod phase1;
pub mod phase2;
pub mod phase3;
pub mod phase4;
pub mod phase5;
pub mod phase6;

use std::time::Duration;

use egg::{BackoffScheduler, Rewrite, Runner};
pub use tensor_ir_frontend::Phase;

use crate::analysis::TensorAnalysis;
use crate::language::TensorIr;
use crate::types::{DeviceProfile, LoweringOptions};

/// Collect all rewrite rules across all phases.
#[must_use]
pub fn all_rules(config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    let mut rules = Vec::new();
    rules.extend(phase1::rules(config));
    if scalar_only(config) {
        return rules;
    }
    rules.extend(phase2::rules(config));
    rules.extend(phase3::rules(config));
    rules.extend(phase4::rules(config));
    rules.extend(phase5::rules(config));
    rules.extend(phase6::rules(config));
    rules
}

fn scalar_only(config: &RunnerConfig) -> bool {
    config.device.simd_width <= 1
        && config.device.max_simdgroups <= 1
        && config.device.max_registers_per_lane <= 1
}

/// Configuration for the equality saturation runner.
#[derive(Debug, Clone)]
pub struct RunnerConfig {
    /// Maximum number of iterations.
    pub iter_limit: usize,
    /// Maximum number of e-nodes.
    pub node_limit: usize,
    /// Maximum time in seconds.
    pub time_limit_secs: u64,
    /// Hardware/runtime parameters used by device-aware rules and the cost
    /// model. Phases that don't need it can ignore it.
    pub device: DeviceProfile,
    /// Structural lowering toggles (e.g. inner-loop unrolling). Customize to
    /// trade perf for IR readability.
    pub lowering: LoweringOptions,
}

/// Per-phase equality-saturation diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaturationPhaseReport {
    pub phase: Phase,
    pub rule_count: usize,
    pub iter_limit: usize,
    pub node_limit: usize,
    pub time_limit_secs: u64,
    pub iterations: usize,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub classes_before: usize,
    pub classes_after: usize,
    pub stop_reason: String,
    pub elapsed: Duration,
}

/// Equality-saturation diagnostics for a multi-phase run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SaturationReport {
    pub phases: Vec<SaturationPhaseReport>,
}

impl SaturationReport {
    #[must_use]
    pub fn total_elapsed(&self) -> Duration {
        self.phases.iter().map(|phase| phase.elapsed).sum()
    }

    pub fn extend(&mut self, other: Self) {
        self.phases.extend(other.phases);
    }
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            iter_limit: 30,
            node_limit: 100_000,
            time_limit_secs: 60,
            device: DeviceProfile::default(),
            lowering: LoweringOptions::default(),
        }
    }
}

/// Run equality saturation with all rules.
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

/// Run equality saturation with only the specified phases.
///
/// Each phase runs its rules in **its own [`Runner`]** against the same
/// (mutated) e-graph — this gives every phase a fresh `BackoffScheduler` ban
/// state and its own slice of the `iter_limit` / `node_limit` budget, while
/// the e-graph itself is preserved across phases so equivalences from earlier
/// phases remain visible to later ones (no rebuild). Per-phase tracing spans
/// surface saturation status, e-graph size, and stop reason — useful when a
/// phase silently clips on the node limit.
#[must_use]
pub fn saturate_phases(
    egraph: egg::EGraph<TensorIr, TensorAnalysis>,
    phases: &[Phase],
    config: &RunnerConfig,
) -> egg::EGraph<TensorIr, TensorAnalysis> {
    saturate_phases_reported(egraph, phases, config).0
}

/// Run equality saturation with only the specified phases and return
/// structured diagnostics for each phase.
#[must_use]
pub fn saturate_phases_reported(
    mut egraph: egg::EGraph<TensorIr, TensorAnalysis>,
    phases: &[Phase],
    config: &RunnerConfig,
) -> (egg::EGraph<TensorIr, TensorAnalysis>, SaturationReport) {
    let mut report = SaturationReport::default();
    for phase in phases {
        egraph.rebuild();
        let rules = phase_rules(*phase, config);
        let rule_count = rules.len();
        let span = tracing::info_span!(
            "saturate_phase",
            phase = ?phase,
            rules = rules.len(),
            iter_limit = config.iter_limit,
            node_limit = config.node_limit,
        );
        let _enter = span.enter();

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

        tracing::info!(
            iterations,
            nodes_before,
            nodes_after,
            classes_before,
            classes_after,
            stop_reason,
            "phase complete"
        );

        report.phases.push(SaturationPhaseReport {
            phase: *phase,
            rule_count,
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
    (egraph, report)
}

fn phase_rules(phase: Phase, config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    match phase {
        Phase::Lowering => phase1::rules(config),
        Phase::LateDispatch if scalar_only(config) => Vec::new(),
        Phase::LateDispatch => {
            // Union of every rewrite that participates in dispatch-shape
            // selection. Running them in one saturation gives the
            // extractor's shape-aware cost model every variant to choose
            // among, instead of committing piecewise across phases 2–6.
            let mut rules = Vec::new();
            rules.extend(phase2::rules(config));
            rules.extend(phase3::rules(config));
            rules.extend(phase4::rules(config));
            rules.extend(phase5::rules(config));
            rules.extend(phase6::rules(config));
            rules
        }
        Phase::StateThreading => Vec::new(),
    }
}
