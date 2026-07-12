//! Stage-1 recognition rules: contraction → MatMul/QMatMul and quantized
//! embedding gather → QEmbedding.
//!
//! Each rule is a direct port of the corresponding destructive sweep
//! (`recognize_contractions` / `recognize_embeddings`), reusing the exact
//! matcher and builder functions (`match_contraction`,
//! `Contraction::{to_q_mat_mul, to_mat_mul}`, `try_unflatten_matmul_input`,
//! `QEmbeddingOperation::new`). The difference is purely mechanical: instead
//! of overwriting the root node, the rebuilt operation is added as an
//! alternative e-node in the root's class; the extractor commits it.
//!
//! Rules run two-phase per round — match immutably, then add — so payload
//! borrows never overlap e-graph mutation.

use super::super::ExecutionVariant;
use super::super::recognize::match_contraction;
use super::lang::Prov;
use super::{EGraphDriver, EgRule, RuleCtx};
use crate::nary_wise::NaryExpr;

impl EGraphDriver {
    /// The ingested (identity) payload of a provenance, if it has one.
    pub(super) fn identity_variant(&self, prov: Prov) -> Option<&ExecutionVariant> {
        self.identity_variants[prov.0 as usize].as_ref()
    }

    pub(super) fn prov_count(&self) -> u32 {
        self.identity_payloads.len() as u32
    }
}

/// `Reduce(Sum, last axis, plain input, empty post)` over a two-factor
/// multiply of pure `DimIndex` loads → QMatMul (preferred) or MatMul.
/// Port of `try_recognize_contraction` (recognize.rs).
pub(super) struct RecognizeContraction;

impl EgRule for RecognizeContraction {
    fn apply_round(&self, driver: &mut EGraphDriver, ctx: &RuleCtx<'_>) -> bool {
        let mut pending: Vec<(Prov, ExecutionVariant)> = Vec::new();
        for prov in (0..driver.prov_count()).map(Prov) {
            let Some(ExecutionVariant::Reduce(reduce)) = driver.identity_variant(prov) else {
                continue;
            };
            let Some(value) = reduce.plain_input() else {
                continue;
            };
            let value_prov = driver.prov_of[&value];
            let value_facts = driver.egraph.analysis.facts_of(value_prov);
            // The multiply must be uncached, not externally held, and exist
            // solely for this reduce (consumed elsewhere it has to
            // materialize anyway) — the destructive sweep's exact gates.
            if value_facts.exec.is_none() || value_facts.externally_live {
                continue;
            }
            if value_facts.consumer_count != 1 {
                continue;
            }
            let Some(ExecutionVariant::Elementwise(nary)) = driver.identity_variant(value_prov)
            else {
                continue;
            };
            let Some(contraction) = match_contraction(reduce, nary) else {
                continue;
            };
            if let Some((operation, _activation)) =
                contraction.to_q_mat_mul(|key| ctx.graph.dequantize_variant(key))
            {
                pending.push((prov, ExecutionVariant::QMatMul(Box::new(operation))));
            } else if let Some((mut operation, _inputs)) =
                contraction.to_mat_mul(&ctx.graph.device())
            {
                ctx.resolver
                    .try_unflatten_matmul_input(ctx.graph, &mut operation);
                pending.push((prov, ExecutionVariant::MatMul(operation)));
            }
        }
        let mut changed = false;
        for (prov, variant) in pending {
            changed |= driver.add_alternative(prov, variant);
        }
        changed
    }
}

/// Quantized row gather `Elementwise([table, idx], table[idx[i], j])` →
/// QEmbedding. Port of `try_recognize_q_embedding` (recognize.rs).
pub(super) struct RecognizeQEmbedding;

impl EgRule for RecognizeQEmbedding {
    fn apply_round(&self, driver: &mut EGraphDriver, ctx: &RuleCtx<'_>) -> bool {
        let mut pending: Vec<(Prov, ExecutionVariant)> = Vec::new();
        for prov in (0..driver.prov_count()).map(Prov) {
            let Some(ExecutionVariant::Elementwise(nary)) = driver.identity_variant(prov) else {
                continue;
            };
            if nary.inputs.len() != 2 || nary.shape.len() != 2 {
                continue;
            }
            // Peel the optional cast from the load type to the requested type.
            let gather = match &nary.expression {
                NaryExpr::Op { children, function }
                    if function.op == crate::nary_wise::NaryOp::Cast && children.len() == 1 =>
                {
                    &children[0]
                }
                expr => expr,
            };
            let NaryExpr::IndexedInput {
                input_idx: 0,
                indices,
            } = gather
            else {
                continue;
            };
            let [row, NaryExpr::DimIndex(1)] = indices.as_slice() else {
                continue;
            };
            let NaryExpr::IndexedInput {
                input_idx: 1,
                indices: row_indices,
            } = row
            else {
                continue;
            };
            if row_indices.as_slice() != [NaryExpr::DimIndex(0)] {
                continue;
            }
            let Some(dequantize) = ctx.graph.dequantize_variant(nary.inputs[0]) else {
                continue;
            };
            if crate::quantized::dequantize::quant_format(&dequantize.matrix).is_none() {
                continue;
            }
            if dequantize.matrix.shape().len() != 2 || dequantize.matrix.shape()[1] != nary.shape[1]
            {
                continue;
            }
            let indexes = nary.inputs[1];
            let operation = crate::quantized::embedding::QEmbeddingOperation::new(
                indexes,
                nary.shape[0],
                dequantize.matrix.clone(),
                nary.output_datatype,
            );
            pending.push((prov, ExecutionVariant::QEmbedding(operation)));
        }
        let mut changed = false;
        for (prov, variant) in pending {
            changed |= driver.add_alternative(prov, variant);
        }
        changed
    }
}
