//! Phase 2: Theta Splitting / Tiling
//!
//! Rule 4: Tile a Theta
//! Rule 5: Multiple outputs per lane

mod register_blocking;
mod theta_merge;
mod theta_tiling;

use egg::Rewrite;

use crate::analysis::TensorAnalysis;
use crate::language::TensorIr;
use crate::rules::RunnerConfig;

const TILE_SIZES: &[u32] = &[8, 16, 32, 64];

pub fn rules(config: &RunnerConfig) -> Vec<Rewrite<TensorIr, TensorAnalysis>> {
    let mut rules: Vec<_> = TILE_SIZES
        .iter()
        .copied()
        .map(theta_tiling::build)
        .collect();
    rules.push(theta_merge::build(config));
    rules.extend(register_blocking::all(config));
    rules
}
