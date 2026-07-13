//! GPU-oriented extraction cost.
//!
//! Dispatches dominate small and medium fusion decisions, materialized bytes
//! are the next-order cost, and approximate arithmetic work breaks remaining
//! ties.  The model is deliberately monotone and deterministic; legality is
//! still owned by the rewrite generators.

use std::ops::{Add, AddAssign};

use super::super::ExecutionVariant;
use super::EGraphDriver;
use super::extract::{ExtractState, Selection};
use super::lang::Prov;
use crate::DataTypeEnum;
use crate::nary_wise::NaryExpr;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct GpuCost {
    pub(super) dispatches: u64,
    pub(super) materialized_bytes: u128,
    pub(super) work: u128,
}

impl Add for GpuCost {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            dispatches: self.dispatches.saturating_add(rhs.dispatches),
            materialized_bytes: self
                .materialized_bytes
                .saturating_add(rhs.materialized_bytes),
            work: self.work.saturating_add(rhs.work),
        }
    }
}

impl AddAssign for GpuCost {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct CostDelta {
    dispatches: i128,
    materialized_bytes: i128,
    work: i128,
}

impl CostDelta {
    #[cfg(test)]
    pub(super) fn improves(self) -> bool {
        self < Self::default()
    }

    /// Fusion rules are one-way structural simplifications. An equal-cost
    /// canonicalization (notably unit-reduce -> elementwise) is useful because
    /// it exposes the next fusion and cannot cycle back to its old form.
    pub(super) fn non_worse(self) -> bool {
        self <= Self::default()
    }
}

impl EGraphDriver {
    pub(super) fn switch_cost_delta(
        &self,
        state: &ExtractState,
        prov: Prov,
        candidate: &ExecutionVariant,
        kills: &[u32],
    ) -> CostDelta {
        let current = self
            .selected_variant(state, prov)
            .map(variant_cost)
            .unwrap_or_default();
        let removed = kills.iter().fold(GpuCost::default(), |mut cost, &dead| {
            if let Some(variant) = self.selected_variant(state, Prov(dead)) {
                cost += variant_cost(variant);
            }
            cost
        });
        let candidate = variant_cost(candidate);
        CostDelta {
            dispatches: i128::from(candidate.dispatches)
                - i128::from(current.dispatches)
                - i128::from(removed.dispatches),
            materialized_bytes: as_i128(candidate.materialized_bytes)
                - as_i128(current.materialized_bytes)
                - as_i128(removed.materialized_bytes),
            work: as_i128(candidate.work) - as_i128(current.work) - as_i128(removed.work),
        }
    }

    pub(super) fn extraction_cost(&self, state: &ExtractState) -> GpuCost {
        let mut cost = GpuCost::default();
        for prov in 0..state.sel.len() as u32 {
            if state.needed[prov as usize]
                && let Some(variant) = self.selected_variant(state, Prov(prov))
            {
                cost += variant_cost(variant);
            }
        }
        cost
    }

    pub(super) fn selection_cost(&self, state: &ExtractState, prov: Prov) -> GpuCost {
        self.selected_variant(state, prov)
            .map(variant_cost)
            .unwrap_or_default()
    }

    fn selected_variant<'a>(
        &'a self,
        state: &'a ExtractState,
        prov: Prov,
    ) -> Option<&'a ExecutionVariant> {
        match &state.sel[prov.0 as usize] {
            Selection::Identity => self.identity_variant(prov),
            Selection::Alt(enode) => enode
                .payload()
                .map(|payload| self.egraph.analysis.payloads.get(payload)),
        }
    }
}

fn as_i128(value: u128) -> i128 {
    value.min(i128::MAX as u128) as i128
}

fn elements(shape: &[usize]) -> u128 {
    shape
        .iter()
        .fold(1u128, |size, &dim| size.saturating_mul(dim as u128))
}

fn bytes(shape: &[usize], datatype: DataTypeEnum) -> u128 {
    elements(shape).saturating_mul(datatype.element_size() as u128)
}

fn expr_work(expression: &NaryExpr) -> u128 {
    match expression {
        NaryExpr::Op { children, .. } => 1 + children.iter().map(expr_work).sum::<u128>(),
        NaryExpr::IndexedInput { indices, .. } => 1 + indices.iter().map(expr_work).sum::<u128>(),
        NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => 1,
    }
}

pub(super) fn variant_cost(variant: &ExecutionVariant) -> GpuCost {
    match variant {
        ExecutionVariant::Tensor(_) => GpuCost::default(),
        ExecutionVariant::QMatrix(operation) => {
            let output = bytes(operation.matrix.shape(), operation.datatype);
            GpuCost {
                dispatches: 1,
                materialized_bytes: output,
                work: elements(operation.matrix.shape())
                    .saturating_mul(1 + operation.post_dequantize.functions.len() as u128),
            }
        }
        ExecutionVariant::Elementwise(operation) => {
            let output_elements = elements(&operation.shape);
            GpuCost {
                dispatches: 1,
                materialized_bytes: bytes(&operation.shape, operation.output_datatype),
                work: output_elements.saturating_mul(expr_work(&operation.expression)),
            }
        }
        ExecutionVariant::Reduce(operation) => {
            let output_shape = operation.out_shape();
            GpuCost {
                dispatches: 1,
                materialized_bytes: bytes(&output_shape, operation.out_datatype()),
                work: elements(&operation.shape).saturating_mul(
                    expr_work(&operation.expression)
                        + 1
                        + operation.post_element_wise.functions.len() as u128,
                ),
            }
        }
        ExecutionVariant::View(operation) => {
            // Some views become zero-dispatch aliases and some materialize a
            // gather. Treat one dispatch as the safe upper bound: folding a
            // view can only make this estimate more accurate, never add work.
            GpuCost {
                dispatches: 1,
                materialized_bytes: bytes(operation.shape(), operation.datatype),
                work: elements(operation.shape()),
            }
        }
        ExecutionVariant::Assign(_) => GpuCost {
            dispatches: 1,
            materialized_bytes: 0,
            work: 1,
        },
        ExecutionVariant::Region(operation) => {
            let output_count = operation.output_count() as u128;
            GpuCost {
                dispatches: 1,
                materialized_bytes: elements(&operation.shape)
                    .saturating_mul(4)
                    .saturating_mul(output_count),
                work: elements(&operation.shape).saturating_mul(operation.statements.len() as u128),
            }
        }
        ExecutionVariant::MatMul(operation) => {
            let batch = elements(operation.a.batch_shape());
            let m = operation.a.rows() as u128;
            let n = operation.b.cols() as u128;
            let k = operation.a.cols() as u128;
            let epilogue = operation.pre_element_wise[0].functions.len()
                + operation.pre_element_wise[1].functions.len()
                + operation.post_element_wise.functions.len();
            GpuCost {
                dispatches: 1,
                materialized_bytes: bytes(&operation.out_shape, operation.datatype),
                work: batch
                    .saturating_mul(m)
                    .saturating_mul(n)
                    .saturating_mul(k.saturating_mul(2).saturating_add(epilogue as u128)),
            }
        }
        ExecutionVariant::QMatMul(operation) => {
            let k = operation.matrix.shape()[1] as u128;
            let n = operation.matrix.shape()[0] as u128;
            let rows = elements(&operation.in_shape[..operation.in_shape.len() - 1]);
            let epilogue = operation
                .pre_element_wise_expr
                .as_ref()
                .map(|epilogue| expr_work(&epilogue.expression))
                .unwrap_or_default()
                + operation
                    .post_element_wise_expr
                    .as_ref()
                    .map(|epilogue| expr_work(&epilogue.expression))
                    .unwrap_or_default();
            GpuCost {
                dispatches: 1,
                materialized_bytes: bytes(&operation.out_shape, operation.input_datatype),
                work: rows
                    .saturating_mul(n)
                    .saturating_mul(k.saturating_mul(2).saturating_add(epilogue)),
            }
        }
        ExecutionVariant::QEmbedding(operation) => GpuCost {
            dispatches: 1,
            materialized_bytes: bytes(&operation.out_shape, operation.datatype),
            work: elements(&operation.out_shape),
        },
        ExecutionVariant::RowProgram(_) => GpuCost {
            dispatches: 1,
            materialized_bytes: 0,
            work: 1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_reduction_dominates_secondary_costs() {
        assert!(
            CostDelta {
                dispatches: -1,
                materialized_bytes: 1_000_000,
                work: 1_000_000,
            }
            .improves()
        );
    }

    #[test]
    fn bytes_then_work_break_dispatch_ties() {
        assert!(
            CostDelta {
                dispatches: 0,
                materialized_bytes: -1,
                work: 1_000_000,
            }
            .improves()
        );
        assert!(
            CostDelta {
                dispatches: 0,
                materialized_bytes: 0,
                work: -1,
            }
            .improves()
        );
        assert!(!CostDelta::default().improves());
        assert!(CostDelta::default().non_worse());
    }
}
