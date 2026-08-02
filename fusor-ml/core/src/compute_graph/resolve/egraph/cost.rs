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
use super::interner::variant_dependencies;
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

/// Roofline constants for the scalarized model, measured on this machine
/// through `kernel_bench`'s anchors (`roof_bw`, `roof_flops`) and the
/// inter-kernel gap of a training step. They convert the three terms to one
/// clock so a dispatch, a byte and a flop can actually outweigh each other.
///
/// Under the lexicographic tuple `dispatches` is effectively infinite: the
/// model will pay unbounded traffic and unbounded arithmetic to remove one
/// launch. On the measured step, dispatches are 0.2% of modeled time.
const DISPATCH_NS: u128 = 1_000;
/// ~340 GB/s achievable (402 MB stream add in ~1.17 ms).
const BYTES_PER_NS: u128 = 340;
/// ~5.5 TFLOP/s achievable (68.7 GFLOP merged matmul in ~12.6 ms).
const FLOPS_PER_NS: u128 = 5_468;

/// The three terms on one clock, scaled by both rates so the comparison
/// stays exact in integers rather than truncating two divisions.
fn scaled_nanos(dispatches: i128, materialized_bytes: i128, work: i128) -> i128 {
    let dispatch_scale = (DISPATCH_NS * BYTES_PER_NS * FLOPS_PER_NS) as i128;
    dispatches.saturating_mul(dispatch_scale)
        + materialized_bytes.saturating_mul(FLOPS_PER_NS as i128)
        + work.saturating_mul(BYTES_PER_NS as i128)
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct CostDelta {
    dispatches: i128,
    materialized_bytes: i128,
    work: i128,
    /// Compare on one clock instead of the lexicographic tuple
    /// (`FUSOR_SPIKE_SCALAR_COST`). Uniform across a run.
    scalar: bool,
}

impl CostDelta {
    /// Fusion rules are one-way structural simplifications. An equal-cost
    /// canonicalization (notably unit-reduce -> elementwise) is useful because
    /// it exposes the next fusion and cannot cycle back to its old form.
    pub(super) fn non_worse(self) -> bool {
        self <= Self::default()
    }

    /// The ordering key: one clock when scalarized, otherwise the
    /// lexicographic tuple this model shipped with.
    fn key(self) -> (i128, i128, i128) {
        if self.scalar {
            (
                scaled_nanos(self.dispatches, self.materialized_bytes, self.work),
                0,
                0,
            )
        } else {
            (self.dispatches, self.materialized_bytes, self.work)
        }
    }
}

// Ordering is by `key`, so equality must be too: `min_by_key` and `non_worse`
// both rely on `Ord` agreeing with `Eq`.
impl PartialEq for CostDelta {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}

impl Eq for CostDelta {}

impl PartialOrd for CostDelta {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CostDelta {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key().cmp(&other.key())
    }
}

impl EGraphDriver {
    /// The write footprint of one variant.
    fn output_bytes(&self, variant: &ExecutionVariant) -> u128 {
        match variant {
            // Leaves cost no dispatch, but reading one still moves its bytes.
            ExecutionVariant::Tensor(data) => bytes(data.info.layout.shape(), data.info.datatype),
            // Attention writes a Q-shaped result; resolve it through Q.
            ExecutionVariant::Attention(operation) => self
                .prov_of
                .get(&operation.q)
                .map(|&prov| self.dependency_bytes(prov))
                .unwrap_or_default(),
            other => variant_cost(other).materialized_bytes,
        }
    }

    /// Bytes a consumer moves to read `prov`'s value: that producer's own
    /// output footprint. Shape and datatype are invariant across a node's
    /// selections, so the identity form answers for every form.
    fn dependency_bytes(&self, prov: Prov) -> u128 {
        self.identity_variant(prov)
            .map(|variant| self.output_bytes(variant))
            .unwrap_or_default()
    }

    /// [`variant_cost`] plus the bytes the variant reads from its inputs.
    ///
    /// The stock model counts output writes only, which makes producer
    /// duplication invisible: inlining a producer deletes one write and adds
    /// a read of each of that producer's own inputs, and only the write was
    /// ever scored. Under `FUSOR_SPIKE_READ_TRAFFIC` the byte term becomes
    /// total traffic, so a fusion that trades one write for two reads is
    /// priced as the loss it is.
    fn traffic_cost(&self, state: &ExtractState, variant: &ExecutionVariant) -> GpuCost {
        let mut cost = variant_cost(variant);
        if !state.read_traffic {
            return cost;
        }
        cost.materialized_bytes = self.output_bytes(variant);
        let reads = variant_dependencies(variant)
            .into_iter()
            .filter_map(|inner| self.prov_of.get(&inner).copied())
            .fold(0u128, |total, prov| {
                total.saturating_add(self.dependency_bytes(prov))
            });
        cost.materialized_bytes = cost.materialized_bytes.saturating_add(reads);
        cost
    }

    pub(super) fn switch_cost_delta(
        &self,
        state: &ExtractState,
        prov: Prov,
        candidate: &ExecutionVariant,
        kills: &[u32],
    ) -> CostDelta {
        let current = self
            .selected_variant(state, prov)
            .map(|variant| self.traffic_cost(state, variant))
            .unwrap_or_default();
        let removed = kills.iter().fold(GpuCost::default(), |mut cost, &dead| {
            if let Some(variant) = self.selected_variant(state, Prov(dead)) {
                cost += self.traffic_cost(state, variant);
            }
            cost
        });
        let candidate = self.traffic_cost(state, candidate);
        CostDelta {
            dispatches: i128::from(candidate.dispatches)
                - i128::from(current.dispatches)
                - i128::from(removed.dispatches),
            materialized_bytes: as_i128(candidate.materialized_bytes)
                - as_i128(current.materialized_bytes)
                - as_i128(removed.materialized_bytes),
            work: as_i128(candidate.work) - as_i128(current.work) - as_i128(removed.work),
            scalar: state.scalar_cost,
        }
    }

    pub(super) fn extraction_cost(&self, state: &ExtractState) -> GpuCost {
        let mut cost = GpuCost::default();
        for prov in 0..state.sel.len() as u32 {
            if state.needed[prov as usize]
                && let Some(variant) = self.selected_variant(state, Prov(prov))
            {
                cost += self.traffic_cost(state, variant);
            }
        }
        cost
    }

    pub(super) fn selection_cost(&self, state: &ExtractState, prov: Prov) -> GpuCost {
        self.selected_variant(state, prov)
            .map(|variant| self.traffic_cost(state, variant))
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
        // A fold writes every output and evaluates its step at every
        // coordinate of the folded index space. Blocking multiplies the
        // dispatch by the number of blocks, which is what makes a split
        // visible to the cost model at all.
        ExecutionVariant::Fold(operation) => {
            let out_shape = operation.out_shape();
            let materialized_bytes = operation
                .outputs
                .iter()
                .map(|output| bytes(&out_shape, output.datatype))
                .fold(0u128, |total, output| total.saturating_add(output));
            GpuCost {
                dispatches: 1,
                materialized_bytes,
                work: elements(&operation.shape).saturating_mul(operation.step_work()),
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
        // A row program's `shape` is its operating footprint; for a reducing
        // program the write is smaller, so this is an upper bound. `work`
        // stays a placeholder — the per-step cost is not derivable here.
        ExecutionVariant::RowProgram(operation) => GpuCost {
            dispatches: 1,
            materialized_bytes: bytes(&operation.shape, operation.output_datatype),
            work: elements(&operation.shape).saturating_mul(operation.work_units() as u128),
        },
        // Attention cannot size its own output: the shape lives on its Q
        // operand, so `traffic_cost` fills the write in from the dependency
        // table. `work` remains a placeholder, so the model under-counts
        // attention compute.
        ExecutionVariant::Attention(_) => GpuCost {
            dispatches: 1,
            materialized_bytes: 0,
            work: 1,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lexicographic(dispatches: i128, materialized_bytes: i128, work: i128) -> CostDelta {
        CostDelta {
            dispatches,
            materialized_bytes,
            work,
            scalar: false,
        }
    }

    fn scalar(dispatches: i128, materialized_bytes: i128, work: i128) -> CostDelta {
        CostDelta {
            dispatches,
            materialized_bytes,
            work,
            scalar: true,
        }
    }

    #[test]
    fn dispatch_reduction_dominates_secondary_costs() {
        assert!(lexicographic(-1, 1_000_000, 1_000_000) < CostDelta::default());
    }

    #[test]
    fn bytes_then_work_break_dispatch_ties() {
        assert!(lexicographic(0, -1, 1_000_000) < CostDelta::default());
        assert!(lexicographic(0, 0, -1) < CostDelta::default());
        assert!(CostDelta::default().non_worse());
        assert!(!lexicographic(0, 0, 1).non_worse());
    }

    #[test]
    fn scalar_cost_lets_traffic_outweigh_a_dispatch() {
        // One saved launch is worth 1 us. 1 MB of extra traffic costs about
        // 3 us at the measured roof, so the trade is a loss — a verdict the
        // lexicographic tuple cannot reach.
        assert!(lexicographic(-1, 1_000_000, 0).non_worse());
        assert!(!scalar(-1, 1_000_000, 0).non_worse());
    }

    #[test]
    fn scalar_cost_still_takes_a_cheap_dispatch_saving() {
        // Same saved launch, but only 100 KB of added traffic: still a win.
        assert!(scalar(-1, 100_000, 0).non_worse());
    }

    #[test]
    fn scalar_cost_prices_recompute_against_a_saved_write() {
        // Dropping a 1 MB write in exchange for recomputing 1M flops wins:
        // the write is ~2.9 us of bandwidth, the arithmetic ~0.2 us.
        assert!(scalar(0, -1_000_000, 1_000_000).non_worse());
        // Ten times the arithmetic for the same saved write does not.
        assert!(!scalar(0, -1_000_000, 20_000_000).non_worse());
    }

    #[test]
    fn ordering_agrees_with_equality_in_both_modes() {
        // `min_by_key` and `non_worse` both require Ord to agree with Eq.
        let a = scalar(1, -(BYTES_PER_NS as i128) * 1_000, 0);
        let b = scalar(0, 0, 0);
        assert_eq!(a == b, a.cmp(&b) == std::cmp::Ordering::Equal);
        let c = lexicographic(0, 0, 5);
        assert_eq!(c == c, c.cmp(&c) == std::cmp::Ordering::Equal);
    }
}
