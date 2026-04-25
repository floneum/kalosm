//! Phase 4: Reduction Optimizations (Rules 10-13)

mod reduce_simd_to_shuffle_tree;
mod theta_inner_cooperative;
mod theta_split_cooperative;
mod theta_to_reduce_simd;

use egg::Rewrite;

use crate::analysis::TensorAnalysis;
use crate::language::TensorIr;
use crate::rules::RunnerConfig;

#[must_use]
pub fn rules(config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    vec![
        theta_split_cooperative::build(config),
        theta_inner_cooperative::build(config),
        theta_to_reduce_simd::build(config),
        reduce_simd_to_shuffle_tree::build(config),
    ]
}
