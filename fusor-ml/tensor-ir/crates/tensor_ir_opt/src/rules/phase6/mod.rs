//! Phase 6: Pipeline fusion.

mod pipeline_fusion;

use egg::Rewrite;

use crate::analysis::TensorAnalysis;
use crate::language::TensorIr;
use crate::rules::RunnerConfig;

#[must_use]
pub fn rules(_config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    vec![pipeline_fusion::build()]
}
