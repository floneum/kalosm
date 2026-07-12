//! Horizontal merging of independent same-category operations into one
//! dispatch.
//!
//! Dense training tapes are dominated by launch overhead: dozens of tiny
//! elementwise kernels, per-parameter Adam updates, and same-shape row
//! reduces, each ~5-10us of fixed launch cost. This pass walks the already
//! toposorted queue and groups independent operations of the same category
//! (Adam updates / elementwise naries / chunked-map row programs) into one
//! kernel whose body is a build-time-unrolled chain of per-segment regions,
//! each guarded by a uniform linear-workgroup-id range compare.
//!
//! Safety and gating:
//! - Every graph exposes compatible matmul, row and elementwise operations.
//!   Per-operation shape and binding limits decide eligibility.
//!   `FUSOR_DISABLE_HORIZONTAL_FUSION` disables the pass.
//! - Grouping is dependency-sound by a wave discipline: an operation joins
//!   the open wave of its category only if it does not (transitively) depend
//!   on any open-wave member of any category; a dependency on an open wave
//!   flushes that wave (emitting the merged dispatch) first. The emitted
//!   queue is therefore always a valid topological order.
//! - Merged outputs are one fresh buffer per segment (never a concatenated
//!   buffer), so flush-replay slot attribution stays 1 buffer <-> 1 slot.

use super::*;
use crate::mir::operation::hash_mir_value;
use crate::row_program::RowProgramOperation;

const CAT_ROW: usize = 2;
/// Dense matmuls bound for the single-pass cooperative-matrix kernel.
const CAT_MATMUL: usize = 3;
/// Dense matmuls bound for the split-K route (weight-gradient shapes).
/// A separate wave from [`CAT_MATMUL`]: the training tape's split-K matmuls
/// are gradient sinks nothing but the optimizer consumes, so their wave
/// survives the entire backward walk and merges across layers, while the
/// forward/dX matmuls flush at every chain dependency.
const CAT_MATMUL_SPLITK: usize = 4;
/// Elementwise regions: multi-output regions from `fusion_region` and
/// single elementwise operations hosted as one-statement regions.
const CAT_REGION: usize = 0;
const CATEGORY_COUNT: usize = 6;

/// Segments of one merged dispatch, in queue order. All members are
/// mutually independent.
#[derive(Debug)]
pub(super) enum MergedSegments {
    Row(Vec<(NodeIndex, RowProgramOperation)>),
    MatMul(Vec<(NodeIndex, crate::matmul::MatMulOperation)>),
    Region(Vec<(NodeIndex, crate::region::ElementwiseRegionOperation)>),
}

impl MergedSegments {
    pub(super) fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        match self {
            Self::Row(segments) => {
                for (_, op) in segments {
                    op.visit_dependencies(f);
                }
            }
            Self::MatMul(segments) => {
                for (_, op) in segments {
                    op.visit_dependencies(f);
                }
            }
            Self::Region(segments) => {
                for (_, op) in segments {
                    op.visit_dependencies(f);
                }
            }
        }
    }

    fn representative(&self) -> NodeIndex {
        match self {
            Self::Row(segments) => segments[0].0,
            Self::MatMul(segments) => segments[0].0,
            Self::Region(segments) => segments[0].0,
        }
    }
}

/// One mergeable operation, categorized.
enum SegOp {
    Row(RowProgramOperation),
    Region(crate::region::ElementwiseRegionOperation),
    /// Carries the profile key so the flush can partition the wave into
    /// same-profile dispatches.
    MatMul(
        Box<crate::matmul::MatMulOperation>,
        crate::matmul::MatmulMergeKey,
    ),
}

impl SegOp {
    fn category(&self) -> usize {
        match self {
            Self::Row(_) => CAT_ROW,
            Self::Region(_) => CAT_REGION,
            Self::MatMul(_, key) => {
                if key.splits().is_some() {
                    CAT_MATMUL_SPLITK
                } else {
                    CAT_MATMUL
                }
            }
        }
    }

    /// Storage bindings this segment will declare in a merged kernel.
    fn bindings(&self) -> usize {
        match self {
            Self::Row(op) => op.inputs.len() + 1,
            Self::Region(op) => op.binding_count(),
            Self::MatMul(..) => MATMUL_SEGMENT_BINDINGS,
        }
    }
}

/// Bindings per merged-matmul segment: `a`, `b`, and the output (whose
/// allocation carries any split-K scratch).
const MATMUL_SEGMENT_BINDINGS: usize = 3;

pub(super) struct HorizontalMerger {
    enabled: bool,
    device: crate::Device,
    /// Max total storage bindings per merged dispatch.
    budget: usize,
    /// Current open-wave generation per category (starts at 1; 0 = "none").
    open_gen: [u32; CATEGORY_COUNT],
    /// For every processed node: the latest wave generation (per category)
    /// it transitively depends on.
    dep_gen: FxHashMap<NodeIndex, [u32; CATEGORY_COUNT]>,
    /// Wave membership (including trailing views): node -> (category,
    /// generation it was added under).
    member: FxHashMap<NodeIndex, (usize, u32)>,
    waves: [Vec<(NodeIndex, SegOp)>; CATEGORY_COUNT],
    /// Zero-cost view aliases of wave members, deferred with their wave and
    /// emitted right after its merged dispatch.
    trailing: [Vec<(NodeIndex, QueuedOperation)>; CATEGORY_COUNT],
    wave_bindings: [usize; CATEGORY_COUNT],
    /// Distinct input nodes of the open region wave: shared inputs (a
    /// learning-rate tensor read by every optimizer segment) bind once, so
    /// the budget only charges new nodes.
    region_wave_inputs: FxHashSet<NodeIndex>,
    /// `before[k][c]`: open wave `k` must flush before open wave `c`
    /// (members of `c` depend on members of `k`). Kept acyclic.
    before: [[bool; CATEGORY_COUNT]; CATEGORY_COUNT],
}

impl HorizontalMerger {
    pub(super) fn new(enabled: bool, device: &crate::Device) -> Self {
        Self {
            enabled,
            device: device.clone(),
            // Total storage bindings per merged dispatch. The nary budget is
            // inputs-only (it assumes one extra output binding); merged
            // kernels bind everything explicitly, and Metal rejects pipeline
            // layouts at the full 31-buffer limit, so stay at the budget.
            budget: device.nary_direct_input_binding_budget(),
            open_gen: [1; CATEGORY_COUNT],
            dep_gen: FxHashMap::default(),
            member: FxHashMap::default(),
            waves: Default::default(),
            trailing: Default::default(),
            wave_bindings: [0; CATEGORY_COUNT],
            region_wave_inputs: FxHashSet::default(),
            before: [[false; CATEGORY_COUNT]; CATEGORY_COUNT],
        }
    }

    /// Whether wave `from` must (transitively) flush before wave `to`.
    fn reaches(&self, from: usize, to: usize) -> bool {
        if self.before[from][to] {
            return true;
        }
        (0..CATEGORY_COUNT).any(|middle| {
            middle != from && middle != to && self.before[from][middle] && self.before[middle][to]
        })
    }

    fn categorize(&self, node: &ExecutionNode) -> Option<SegOp> {
        match &node.variant {
            // Regions already have a valid single-region queue lowering. The
            // merger may combine independent regions; there is no element cap
            // because multi-output regions have no tiled fallback.
            ExecutionVariant::Region(op) => Some(SegOp::Region(op.clone())),
            ExecutionVariant::Elementwise(op) => {
                // Ops at or above the register-reuse tiled path's engagement
                // element count stay out of merges so that plan never
                // regresses; the bound is derived from the same policy the
                // tiled planner reads, so the two cannot drift apart.
                let elements: usize = op.shape.iter().product();
                let bound = self.device.dispatch_policy().merge_elements_bound();
                (elements < bound && op.inputs.len() + 1 < self.budget).then(|| {
                    SegOp::Region(crate::region::ElementwiseRegionOperation::from_nary(
                        op.clone(),
                        node.inner_idx,
                    ))
                })
            }
            ExecutionVariant::Reduce(op) => {
                self.row_candidate(RowProgramOperation::from_reduce(op))
            }
            ExecutionVariant::GraphOp(op) => {
                let row = op.as_row_program()?.clone();
                self.row_candidate(row)
            }
            ExecutionVariant::MatMul(op) => {
                // Only matmuls bound for the cooperative-matrix kernel
                // produce a profile; generic-path contractions and
                // epilogue-fused matmuls lower standalone.
                if std::env::var_os("FUSOR_TRACE_MATMUL_MERGE").is_some() {
                    eprintln!(
                        "matmul_merge_candidate name={} profile={:?}",
                        op.name(),
                        op.merge_profile(&self.device)
                    );
                }
                let key = op.merge_profile(&self.device)?;
                Some(SegOp::MatMul(Box::new(op.clone()), key))
            }
            _ => None,
        }
    }

    fn row_candidate(&self, row: RowProgramOperation) -> Option<SegOp> {
        (row.dynamic_axis.is_none()
            && row.mergeable_chunked_map()
            && row.inputs.len() + 1 < self.budget)
            .then_some(SegOp::Row(row))
    }

    /// Feed one toposorted node. `lowered` is the node's normal lowering (for
    /// the non-merged path). Emits queue entries into `out`.
    pub(super) fn push(
        &mut self,
        node: &ExecutionNode,
        lowered: Option<QueuedOperation>,
        out: &mut Vec<(NodeIndex, QueuedOperation)>,
    ) {
        if !self.enabled {
            if let Some(op) = lowered {
                out.push((node.inner_idx, op));
            }
            return;
        }

        // Latest wave generation (per category) this node depends on,
        // through direct wave members and transitively via `dep_gen`.
        let mut dep = [0u32; CATEGORY_COUNT];
        let mut visit = |input: NodeIndex| {
            if let Some(&(cat, generation)) = self.member.get(&input) {
                dep[cat] = dep[cat].max(generation);
            }
            if let Some(gens) = self.dep_gen.get(&input) {
                for (slot, generation) in dep.iter_mut().zip(gens) {
                    *slot = (*slot).max(*generation);
                }
            }
        };
        match &node.variant {
            ExecutionVariant::Tensor(_) => {}
            ExecutionVariant::QMatrix(op) => op.visit_dependencies(&mut visit),
            ExecutionVariant::Elementwise(op) => op.visit_dependencies(&mut visit),
            ExecutionVariant::Reduce(op) => op.visit_dependencies(&mut visit),
            ExecutionVariant::View(op) => op.visit_dependencies(&mut visit),
            ExecutionVariant::Assign(op) => op.visit_dependencies(&mut visit),
            ExecutionVariant::Region(op) => op.visit_dependencies(&mut visit),
            ExecutionVariant::MatMul(op) => op.visit_dependencies(&mut visit),
            ExecutionVariant::QMatMul(op) => op.visit_dependencies(&mut visit),
            ExecutionVariant::QEmbedding(op) => op.visit_dependencies(&mut visit),
            ExecutionVariant::GraphOp(op) => op.visit_dependencies(&mut visit),
        }

        match self.categorize(node) {
            Some(seg) => {
                let cat = seg.category();
                // Flush the own-category wave first if this op depends on it
                // (this also flushes the wave's ordered predecessors).
                if dep[cat] == self.open_gen[cat] {
                    self.flush(cat, out);
                }
                // Region segments only pay for input nodes the open wave
                // does not already bind: the builder deduplicates shared
                // read-only inputs (a learning-rate tensor read by every
                // optimizer segment) into one binding.
                let fresh_bindings = |merger: &Self, seg: &SegOp| match seg {
                    SegOp::Region(op) => {
                        op.inputs
                            .iter()
                            .filter(|input| !merger.region_wave_inputs.contains(*input))
                            .collect::<FxHashSet<_>>()
                            .len()
                            + op.output_count()
                    }
                    other => other.bindings(),
                };
                if self.wave_bindings[cat] + fresh_bindings(self, &seg) > self.budget {
                    self.flush(cat, out);
                }
                // Recompute after a possible flush: a cleared wave shares
                // nothing yet.
                let seg_bindings = fresh_bindings(self, &seg);
                if let SegOp::Region(op) = &seg {
                    self.region_wave_inputs.extend(op.inputs.iter().copied());
                }
                // Dependencies on other open waves become flush-order
                // constraints (that wave's merged dispatch is emitted before
                // ours) unless that would create a cycle.
                for (other, &generation) in dep.iter().enumerate() {
                    if other != cat && generation == self.open_gen[other] {
                        if self.reaches(cat, other) {
                            self.flush(other, out);
                        } else {
                            self.before[other][cat] = true;
                        }
                    }
                }
                self.wave_bindings[cat] += seg_bindings;
                self.member
                    .insert(node.inner_idx, (cat, self.open_gen[cat]));
                self.waves[cat].push((node.inner_idx, seg));
                self.dep_gen.insert(node.inner_idx, dep);
            }
            None => {
                self.dep_gen.insert(node.inner_idx, dep);
                let open: Vec<usize> = (0..CATEGORY_COUNT)
                    .filter(|&cat| dep[cat] == self.open_gen[cat])
                    .collect();
                // A zero-cost view of an open-wave value defers with that
                // wave instead of flushing it: it is emitted right after the
                // wave's merged dispatch, so its input is cached in time.
                if matches!(node.variant, ExecutionVariant::View(_))
                    && !open.is_empty()
                    && let Some(op) = lowered
                {
                    // Attach to the wave every other open dependency is
                    // ordered before; bail to flushing if none dominates.
                    if let Some(&last) = open
                        .iter()
                        .find(|&&cat| open.iter().all(|&o| o == cat || self.reaches(o, cat)))
                    {
                        self.member
                            .insert(node.inner_idx, (last, self.open_gen[last]));
                        self.trailing[last].push((node.inner_idx, op));
                        return;
                    }
                    for cat in open {
                        self.flush(cat, out);
                    }
                    out.push((node.inner_idx, op));
                    return;
                }
                for cat in open {
                    self.flush(cat, out);
                }
                if let Some(op) = lowered {
                    out.push((node.inner_idx, op));
                }
            }
        }
    }

    pub(super) fn finish(&mut self, out: &mut Vec<(NodeIndex, QueuedOperation)>) {
        for cat in 0..CATEGORY_COUNT {
            self.flush(cat, out);
        }
    }

    fn flush(&mut self, cat: usize, out: &mut Vec<(NodeIndex, QueuedOperation)>) {
        // Ordered predecessors first. Clear this wave's constraint edges
        // before recursing so a predecessor's own flush can never loop back.
        let predecessors: Vec<usize> = (0..CATEGORY_COUNT)
            .filter(|&other| other != cat && self.before[other][cat])
            .collect();
        for other in 0..CATEGORY_COUNT {
            self.before[cat][other] = false;
            self.before[other][cat] = false;
        }
        for other in predecessors {
            self.flush(other, out);
        }
        let wave = std::mem::take(&mut self.waves[cat]);
        let trailing = std::mem::take(&mut self.trailing[cat]);
        if cat == CAT_REGION {
            self.region_wave_inputs.clear();
        }
        self.wave_bindings[cat] = 0;
        self.open_gen[cat] += 1;
        if wave.is_empty() {
            debug_assert!(trailing.is_empty());
            return;
        }
        if wave.len() == 1 {
            let (node, seg) = wave.into_iter().next().expect("length checked");
            let op: QueuedOperation = match seg {
                SegOp::Row(op) => QueuedOperation::Operation(Arc::new(op)),
                SegOp::MatMul(op, _) => QueuedOperation::Operation(Arc::new(*op)),
                // A lone one-statement region lowers through the standalone
                // elementwise path (which keeps the register-reuse tiled
                // plan); genuine multi-output regions have no standalone
                // lowering and stay merged.
                SegOp::Region(op) => {
                    if op.statements.len() == 1 {
                        let nary = op
                            .into_nary()
                            .expect("single-statement regions always emit their output");
                        QueuedOperation::Operation(Arc::new(nary))
                    } else {
                        QueuedOperation::Merged(MergedSegments::Region(vec![(node, op)]))
                    }
                }
            };
            out.push((node, op));
            out.extend(trailing);
            return;
        }
        if cat == CAT_MATMUL || cat == CAT_MATMUL_SPLITK {
            // Partition into same-profile groups (first-occurrence order):
            // only identical profiles share a guarded dispatch, so the
            // segment bodies differ solely in their storage bindings. All
            // wave members are mutually independent, so any group order is
            // a valid topological order.
            type MatmulGroup = Vec<(NodeIndex, crate::matmul::MatMulOperation)>;
            let mut groups: Vec<(crate::matmul::MatmulMergeKey, MatmulGroup)> = Vec::new();
            for (node, seg) in wave {
                let SegOp::MatMul(op, key) = seg else {
                    unreachable!("wave category mismatch");
                };
                match groups.iter_mut().find(|(existing, _)| *existing == key) {
                    Some((_, group)) => group.push((node, *op)),
                    None => groups.push((key, vec![(node, *op)])),
                }
            }
            for (key, group) in groups {
                if std::env::var_os("FUSOR_TRACE_MATMUL_MERGE").is_some() {
                    eprintln!(
                        "matmul_merge_flush cat={cat} size={} key={key:?}",
                        group.len()
                    );
                }
                if group.len() == 1 {
                    let (node, op) = group.into_iter().next().expect("length checked");
                    out.push((node, QueuedOperation::Operation(Arc::new(op))));
                } else {
                    let merged = MergedSegments::MatMul(group);
                    out.push((merged.representative(), QueuedOperation::Merged(merged)));
                }
            }
            out.extend(trailing);
            return;
        }
        let merged = match cat {
            CAT_REGION => MergedSegments::Region(
                wave.into_iter()
                    .map(|(node, seg)| match seg {
                        SegOp::Region(op) => (node, op),
                        _ => unreachable!("wave category mismatch"),
                    })
                    .collect(),
            ),
            _ => MergedSegments::Row(
                wave.into_iter()
                    .map(|(node, seg)| match seg {
                        SegOp::Row(op) => (node, op),
                        _ => unreachable!("wave category mismatch"),
                    })
                    .collect(),
            ),
        };
        out.push((merged.representative(), QueuedOperation::Merged(merged)));
        out.extend(trailing);
    }
}

/// Hash one merged dispatch's cache-key material: every segment's kernel
/// fields plus every MIR input value layout.
pub(crate) fn hash_merged_segments<O: Operation>(
    state: &mut rustc_hash::FxHasher,
    segments: &[O],
    segment_inputs: &[Vec<MirValue>],
) {
    use std::hash::Hash;
    segments.len().hash(state);
    for (op, inputs) in segments.iter().zip(segment_inputs) {
        op.hash_kernel_fields(state);
        inputs.len().hash(state);
        for input in inputs {
            hash_mir_value(state, input);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::kernel_selection::CooperativeMatrixKind;
    use crate::{Device, Tensor};

    #[test]
    fn standard_graph_merges_independent_qkv_matmuls() {
        pollster::block_on(async {
            const D: usize = 64;

            let Ok(device) = Device::new().await else {
                return;
            };
            let coop_viable = device
                .coop_token(CooperativeMatrixKind::F32F32M8N8K8)
                .is_some()
                && device
                    .subgroup_config()
                    .is_some_and(|config| config.is_fixed());
            if !coop_viable {
                return;
            }

            let input_values = (0..D * D)
                .map(|index| (index % 17) as f32 * 0.125 - 1.0)
                .collect::<Vec<_>>();
            let diagonal = |scale: f32| {
                (0..D * D)
                    .map(|index| {
                        let row = index / D;
                        let column = index % D;
                        if row == column { scale } else { 0.0 }
                    })
                    .collect::<Vec<_>>()
            };
            let input = Tensor::from_slice(&device, [D, D], &input_values);
            let q = input.mat_mul(&Tensor::from_slice(&device, [D, D], &diagonal(1.0)));
            let k = input.mat_mul(&Tensor::from_slice(&device, [D, D], &diagonal(2.0)));
            let v = input.mat_mul(&Tensor::from_slice(&device, [D, D], &diagonal(-0.5)));
            let output = &(&q + &k) + &v;

            assert_eq!(
                output.count_kernels_to_resolve(),
                2,
                "Q/K/V should share one matmul dispatch followed by one nary dispatch",
            );
            let values = output.as_slice::<2, f32>().await.unwrap();
            for row in 0..D {
                for column in 0..D {
                    let expected = input_values[row * D + column] * 2.5;
                    assert!(
                        (values[[row, column]] - expected).abs() < 1e-4,
                        "mismatch at [{row}, {column}]",
                    );
                }
            }
        });
    }
}
