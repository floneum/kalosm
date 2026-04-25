//! Shared unrolling helpers for lowering passes.
//!
//! Two call-site shapes exist in the crate:
//!
//! - **Direct** (phase1 lowering): builds the loop body as a function of
//!   the iteration index via `egraph.add`, with no `chosen` override and
//!   no `iter(0)` substitution. Use [`unroll_fold_direct`].
//! - **Substituted** (skeleton tiled lowering): has a pre-built body that
//!   references `VarRef::iter(0)`; each step substitutes a fresh literal,
//!   tracked through the `chosen` extraction-override map. Use
//!   [`unroll_fold_substituted`] (with [`k_step_lit`] for the shared
//!   `Const(U32(k_step))` idiom).

use std::collections::HashMap;

use egg::{EGraph, Id};

use crate::add_and_choose;
use crate::analysis::TensorAnalysis;
use crate::language::{SimdNode, TensorIr};
use crate::types::{ScalarValue, VarRef};

/// Fully unroll a fold-style reduction into the e-graph, wrapping the
/// result in a `Theta(init, count, update)` node. The body is called once
/// per iteration with the running accumulator id and returns the new
/// accumulator id. The initial accumulator value is `Var(acc(0))`, so the
/// surrounding `Theta` binds it.
pub fn unroll_fold_direct<F>(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    init: Id,
    count: Id,
    iters: u32,
    mut body: F,
) -> Id
where
    F: FnMut(&mut EGraph<TensorIr, TensorAnalysis>, u32, Id) -> Id,
{
    let acc_var = egraph.add(TensorIr::Simd(SimdNode::Var(VarRef::acc(0))));
    let mut update = acc_var;
    for idx in 0..iters {
        update = body(egraph, idx, update);
    }
    egraph.add(TensorIr::Simd(SimdNode::Theta {
        children: [init, count, update],
    }))
}

/// Fold-style unroll for `chosen`-tracked substitution call sites. Does
/// NOT wrap the result in a `Theta` — callers either span multiple
/// accumulators under one outer `Theta` (multi-output tiled), or compose
/// with surrounding state-threading per step. The body receives the
/// running accumulator id and returns the new accumulator id; the initial
/// value is whatever `seed` the caller provides (typically the
/// already-bound `current_acc`, not a fresh `Var(acc(0))`).
pub fn unroll_fold_substituted<F>(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    iters: u32,
    seed: Id,
    mut body: F,
) -> Id
where
    F: FnMut(&mut EGraph<TensorIr, TensorAnalysis>, &mut HashMap<Id, TensorIr>, u32, Id) -> Id,
{
    let mut acc = seed;
    for idx in 0..iters {
        acc = body(egraph, chosen, idx, acc);
    }
    acc
}

/// Build the `Const(U32(k_step))` literal used as the `iter(0)`
/// replacement in tiled-output unrolling, registered in `chosen` so
/// extraction picks this exact node.
pub fn k_step_lit(
    egraph: &mut EGraph<TensorIr, TensorAnalysis>,
    chosen: &mut HashMap<Id, TensorIr>,
    k_step: u32,
) -> Id {
    add_and_choose(egraph, chosen, TensorIr::Const(ScalarValue::U32(k_step)))
}
