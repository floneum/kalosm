//! Phase 3: Stride-Zero Detection and Threadgroup Promotion
//!
//! Rule (tiled_load_promotion): for outer-K tiled reductions, rewrite a
//!        device Load's address into tile-local form and switch the load to
//!        threadgroup tier.
//! Dependence analysis that feeds this rule lives in `analysis.rs`.
//!
//! Threadgroup promotion is only sound when the address is already tile-local.
//! Phase-1's `fused_reduce_elementwise_lowering` and `tiled_load_promotion`
//! emit correctly-strided TG Loads paired with matching cooperative Stores.

mod tiled_load_promotion;

use egg::Rewrite;

use crate::analysis::TensorAnalysis;
use crate::language::TensorIr;
use crate::rules::RunnerConfig;

#[must_use]
pub fn rules(config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    vec![tiled_load_promotion::build(config)]
}
