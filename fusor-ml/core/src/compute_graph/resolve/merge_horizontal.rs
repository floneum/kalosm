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
//! - The pass only runs when `optimize_large_graph` took the dense branch
//!   (`has_qmatmul == false`, > 512 nodes) and the
//!   `FUSOR_DISABLE_HORIZONTAL_FUSION` kill switch is unset, so quantized
//!   decode graphs and small dense conformance graphs take byte-identical
//!   paths by construction.
//! - Grouping is dependency-sound by a wave discipline: an operation joins
//!   the open wave of its category only if it does not (transitively) depend
//!   on any open-wave member of any category; a dependency on an open wave
//!   flushes that wave (emitting the merged dispatch) first. The emitted
//!   queue is therefore always a valid topological order.
//! - Merged outputs are one fresh buffer per segment (never a concatenated
//!   buffer), so flush-replay slot attribution stays 1 buffer <-> 1 slot.

use super::*;
use crate::mir::kernel_backend::DirectKernel;
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

/// Elementwise ops at or above this element count may take the register
/// reuse tiled path in the single-op builder; keep them out of merges so
/// that plan never regresses.
const MAX_MERGED_NARY_ELEMENTS: usize = 262_144;

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
    MatMul(crate::matmul::MatMulOperation, crate::matmul::MatmulMergeKey),
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
            // Regions have no standalone lowering; the merger always hosts
            // them (a lone region becomes a single-segment merged dispatch).
            // No element cap: multi-output regions have no tiled fallback.
            ExecutionVariant::Region(op) => Some(SegOp::Region(op.clone())),
            ExecutionVariant::Elementwise(op) => {
                let elements: usize = op.shape.iter().product();
                (elements < MAX_MERGED_NARY_ELEMENTS && op.inputs.len() + 1 < self.budget)
                    .then(|| {
                        SegOp::Region(crate::region::ElementwiseRegionOperation::from_nary(
                            op.clone(),
                            node.inner_idx,
                        ))
                    })
            }
            ExecutionVariant::Reduce(op) => {
                let mut row = RowProgramOperation::from_reduce(op);
                // The merger only runs on the dense large-graph branch, so
                // its hosted row programs always take the tuned codegen.
                row.dense_codegen = true;
                self.row_candidate(row)
            }
            ExecutionVariant::GraphOp(op) => {
                let row = op.as_row_program()?.clone();
                self.row_candidate(row)
            }
            ExecutionVariant::MatMul(op) => {
                // Only dense-tuned matmuls bound for the cooperative-matrix
                // kernel produce a profile; everything else (quantized
                // graphs never run this pass, generic-path contractions,
                // epilogue-fused matmuls) lowers standalone.
                if std::env::var_os("FUSOR_TRACE_MATMUL_MERGE").is_some() {
                    eprintln!(
                        "matmul_merge_candidate name={} dense={} profile={:?}",
                        op.name(),
                        op.dense_codegen,
                        op.merge_profile(&self.device)
                    );
                }
                let key = op.merge_profile(&self.device)?;
                Some(SegOp::MatMul(op.clone(), key))
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
                for other in 0..CATEGORY_COUNT {
                    if other != cat && dep[other] == self.open_gen[other] {
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
                    && lowered.is_some()
                {
                    let op = lowered.expect("checked above");
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
                SegOp::Row(op) => QueuedOperation::Generic(Arc::new(op)),
                SegOp::MatMul(op, _) => QueuedOperation::Generic(Arc::new(op)),
                // A lone one-statement region lowers through the standalone
                // elementwise path (which keeps the register-reuse tiled
                // plan); genuine multi-output regions have no standalone
                // lowering and stay merged.
                SegOp::Region(op) => {
                    if op.statements.len() == 1 {
                        let nary = op
                            .into_nary()
                            .expect("single-statement regions always emit their output");
                        QueuedOperation::Generic(Arc::new(nary))
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
                    Some((_, group)) => group.push((node, op)),
                    None => groups.push((key, vec![(node, op)])),
                }
            }
            for (key, group) in groups {
                if std::env::var_os("FUSOR_TRACE_MATMUL_MERGE").is_some() {
                    eprintln!("matmul_merge_flush cat={cat} size={} key={key:?}", group.len());
                }
                if group.len() == 1 {
                    let (node, op) = group.into_iter().next().expect("length checked");
                    out.push((node, QueuedOperation::Generic(Arc::new(op))));
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

impl MergedSegments {
    /// Segment (node, op) views in queue order.
    fn segment_ops(&self) -> Vec<(NodeIndex, &dyn Operation)> {
        match self {
            Self::Row(segments) => segments
                .iter()
                .map(|(node, op)| (*node, op as &dyn Operation))
                .collect(),
            Self::MatMul(segments) => segments
                .iter()
                .map(|(node, op)| (*node, op as &dyn Operation))
                .collect(),
            Self::Region(_) => {
                unreachable!("region segments are gathered without the Operation trait")
            }
        }
    }
}

/// One entry of the dense three-phase queue, preserving queue order.
enum DenseStep {
    View {
        node: NodeIndex,
        result: TensorData,
        deps: Vec<NodeIndex>,
    },
    CopyAssign {
        node: NodeIndex,
        copies: Vec<CopyBufferRecord>,
        op: QueuedOperation,
    },
    Work(usize),
}

enum DenseWorkKind {
    Generic {
        inputs: Vec<MirValue>,
        workgroup_shape: crate::mir::workgroup_shape::WorkgroupShape,
        resolved: TensorData,
        /// Node whose dead buffer this output claimed, if any.
        claimed_from: Option<NodeIndex>,
    },
    Merged {
        segment_inputs: Vec<Vec<MirValue>>,
        /// One entry per segment output, in segment/statement order (regions
        /// contribute several outputs per segment), with the node whose dead
        /// buffer the output claimed, if any.
        outputs: Vec<(NodeIndex, TensorData, Option<NodeIndex>)>,
    },
}

struct DenseWork {
    node: NodeIndex,
    op: QueuedOperation,
    kind: DenseWorkKind,
    built: std::sync::Mutex<Option<DenseBuilt>>,
}

struct DenseBuilt {
    kernels: Vec<DirectKernel>,
    prepared: Vec<Option<(PreparedDirectDispatch, String)>>,
    /// False when a merged builder declined and the kernels are per-segment
    /// fallbacks (which a flush plan cannot express).
    merged_ok: bool,
}

/// A structural plan-cache key for one horizontally merged dispatch: the
/// wave discriminant plus every segment's own structural kernel key (or the
/// region's kernel fields), so isomorphic waves across resolves and
/// processes share one plan.
fn merged_plan_cache_key(
    merged: &MergedSegments,
    segment_inputs: &[Vec<MirValue>],
) -> crate::mir::kernel_backend::KernelCacheKey {
    struct MergedPlanKernelVariant;
    crate::mir::kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
        use std::hash::Hash;
        crate::mir::kernel_backend::KernelVariantKey::of::<MergedPlanKernelVariant>().hash(state);
        std::mem::discriminant(merged).hash(state);
        segment_inputs.len().hash(state);
        match merged {
            MergedSegments::Region(segments) => {
                for ((_, op), inputs) in segments.iter().zip(segment_inputs) {
                    op.hash_kernel_fields(state);
                    inputs.len().hash(state);
                    for input in inputs {
                        crate::mir::operation::hash_mir_value(state, input);
                    }
                }
            }
            _ => {
                for ((_, op), inputs) in merged.segment_ops().iter().zip(segment_inputs) {
                    op.kernel_cache_key_with_dispatch(
                        crate::mir::kernel_backend::KernelVariantKey::of::<MergedPlanKernelVariant>(),
                        None,
                        [0; 3],
                        inputs,
                    )
                    .hash(state);
                }
            }
        }
    })
}

fn build_dense_work(
    work: &DenseWork,
    graph: &ComputeGraphInner,
    device: &crate::Device,
    plan_cache_enabled: bool,
) -> DenseBuilt {
    let build_timer = std::time::Instant::now();
    let (kernels, merged_ok) = match (&work.op, &work.kind) {
        (
            QueuedOperation::QMatMul(qmatmul),
            DenseWorkKind::Generic {
                inputs,
                workgroup_shape,
                ..
            },
        ) => {
            let build_kernels = || {
                qmatmul
                    .build_direct_kernels(graph, workgroup_shape, inputs)
                    .unwrap_or_else(|error| panic!("{error}"))
                    .into_kernels()
            };
            let kernels = if plan_cache_enabled {
                let kernel_key = structural_kernel_key(qmatmul.as_ref(), inputs, workgroup_shape);
                super::run::resolve_cached_direct_plan(
                    device.kernel_cache(),
                    kernel_key,
                    super::run::direct_plan_binding_buffers(inputs),
                    build_kernels,
                )
            } else {
                build_kernels()
            };
            (kernels, true)
        }
        (QueuedOperation::Generic(operation), DenseWorkKind::Generic { inputs, workgroup_shape, .. }) => {
            let build_kernels = || {
                vec![
                    operation
                        .build_direct_kernel(graph, workgroup_shape, inputs)
                        .unwrap_or_else(|| {
                            panic!(
                                "operation did not provide a direct kernel: {}",
                                operation.name()
                            )
                        }),
                ]
            };
            let kernels = if plan_cache_enabled {
                let kernel_key =
                    structural_kernel_key(operation.as_ref(), inputs, workgroup_shape);
                super::run::resolve_cached_direct_plan(
                    device.kernel_cache(),
                    kernel_key,
                    super::run::direct_plan_binding_buffers(inputs),
                    build_kernels,
                )
            } else {
                build_kernels()
            };
            (kernels, true)
        }
        (QueuedOperation::Merged(merged), DenseWorkKind::Merged { segment_inputs, .. }) => {
            // Merged kernels go through the same plan cache as single ops:
            // buffers are presented flattened in segment order, and the
            // insert path verifies that order matches the kernel's true
            // binding order (folded or deduplicated plans silently skip).
            let expected: Vec<std::sync::Arc<wgpu::Buffer>> = segment_inputs
                .iter()
                .flatten()
                .filter_map(|value| match value {
                    MirValue::Tensor(tensor) => Some(tensor.buffer().clone()),
                    MirValue::QMatrix(matrix) => Some(matrix.buffer().clone()),
                    MirValue::Integer(_) | MirValue::Float(_) => None,
                })
                .collect();
            let plan_key =
                plan_cache_enabled.then(|| merged_plan_cache_key(merged, segment_inputs));
            if let Some(key) = plan_key
                && let Some(kernels) = device
                    .kernel_cache()
                    .direct_plan_cache()
                    .get_many(device.kernel_cache(), key, &[&expected])
            {
                return finish_dense_build(build_timer, kernels, true, device);
            }
            let built = match merged {
                MergedSegments::Row(segments) => {
                    crate::row_program::build_merged_row_program_kernel(
                        graph,
                        &segments.iter().map(|(_, op)| op.clone()).collect::<Vec<_>>(),
                        segment_inputs,
                    )
                }
                MergedSegments::MatMul(segments) => crate::matmul::build_merged_matmul_kernel(
                    graph,
                    &segments.iter().map(|(_, op)| op.clone()).collect::<Vec<_>>(),
                    segment_inputs,
                ),
                MergedSegments::Region(segments) => {
                    crate::nary_direct::build_merged_region_kernel(
                        graph,
                        &segments.iter().map(|(_, op)| op.clone()).collect::<Vec<_>>(),
                        segment_inputs,
                    )
                }
            };
            match built {
                Some(kernel) => {
                    if let Some(key) = plan_key {
                        device.kernel_cache().direct_plan_cache().insert_many(
                            key,
                            std::slice::from_ref(&kernel),
                            &[&expected],
                        );
                    }
                    (vec![kernel], true)
                }
                None if matches!(merged, MergedSegments::Region(_)) => {
                    // Region fallback: one standalone region kernel per
                    // segment. Correct but not plan-expressible; poisoned in
                    // phase 3.
                    let MergedSegments::Region(segments) = merged else {
                        unreachable!("matched above");
                    };
                    let kernels = segments
                        .iter()
                        .zip(segment_inputs)
                        .map(|((_, op), inputs)| {
                            crate::nary_direct::build_merged_region_kernel(
                                graph,
                                std::slice::from_ref(op),
                                std::slice::from_ref(inputs),
                            )
                            .unwrap_or_else(|| {
                                panic!("region fallback did not provide a kernel: {}", op.name())
                            })
                        })
                        .collect();
                    (kernels, false)
                }
                None => {
                    // Fallback: per-segment kernels. Correct but not
                    // plan-expressible; the recording is poisoned in phase 3.
                    let max_subgroup_size = device.max_subgroup_size();
                    let kernels = merged
                        .segment_ops()
                        .into_iter()
                        .zip(segment_inputs)
                        .map(|((_, op), inputs)| {
                            let constraints = op.workgroup_shape_constraints(device);
                            let workgroup_shape = constraints
                                .solve(max_subgroup_size, &device.limits())
                                .unwrap_or_else(|| {
                                    panic!("failed to solve workgroup shape for merged fallback")
                                });
                            op.build_direct_kernel(graph, &workgroup_shape, inputs)
                                .unwrap_or_else(|| {
                                    panic!(
                                        "merged fallback segment did not provide a kernel: {}",
                                        op.name()
                                    )
                                })
                        })
                        .collect();
                    (kernels, false)
                }
            }
        }
        _ => unreachable!("dense work kind matches its queued operation"),
    };
    finish_dense_build(build_timer, kernels, merged_ok, device)
}

/// Prepare dispatches (which also compiles shaders and pipelines, here on
/// the parallel build workers) and assemble the phase-2 result.
fn finish_dense_build(
    build_timer: std::time::Instant,
    kernels: Vec<crate::mir::kernel_backend::DirectKernel>,
    merged_ok: bool,
    device: &crate::Device,
) -> DenseBuilt {
    let prepared = kernels
        .iter()
        .map(|kernel| {
            kernel
                .prepare_dispatch(device.kernel_cache())
                .map(|dispatch| (dispatch, kernel.name().to_string()))
        })
        .collect();
    if std::env::var_os("FUSOR_TRACE_BUILD_TIMES").is_some() {
        let total = build_timer.elapsed();
        if total.as_millis() >= 2 {
            eprintln!(
                "build_time total={total:?} first={}",
                kernels.first().map(|k| k.name()).unwrap_or("")
            );
        }
    }
    DenseBuilt {
        kernels,
        prepared,
        merged_ok,
    }
}

impl Resolver {
    /// Three-phase queue execution for dense large graphs: serial input
    /// gathering and output caching (queue order), parallel kernel building
    /// and dispatch preparation, then serial recording, encoding, and
    /// release accounting in exactly the original queue order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_dense_queue(
        &self,
        graph: &mut ComputeGraphInner,
        device: &crate::Device,
        max_subgroup_size: u32,
        queued_operations: Vec<(NodeIndex, QueuedOperation)>,
        remaining_consumers: &mut FxHashMap<NodeIndex, usize>,
        target_set: &FxHashSet<NodeIndex>,
        ledger: &mut super::alloc_reuse::BufferLedger,
        plan_cache_enabled: bool,
        commands: &mut Vec<CommandRecord>,
        host_profile: &mut ResolveHostProfile,
        host_trace: bool,
        on_dispatch_name: &mut dyn FnMut(&str) -> Option<String>,
    ) {
        // Phase 1: gather inputs, allocate outputs, cache results.
        let gather_start = host_trace.then(Instant::now);
        let mut steps = Vec::with_capacity(queued_operations.len());
        let mut work: Vec<DenseWork> = Vec::new();
        for (node, queued_operation) in queued_operations {
            let view_result = if let Some(node_data) = graph.nodes.nodes.node_weight(node) {
                match &node_data.variant {
                    ComputeGraphNodeVariant::View(view) => graph
                        .get_cached_result(view.input)
                        .and_then(|input| view.try_map_tensor(input)),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(result) = view_result {
                let mut deps = Vec::new();
                graph.visit_dependencies(node, &mut |dep| deps.push(dep));
                graph.set_cached_result(node, result.clone());
                ledger.note_transient(result.buffer());
                ledger.consume(graph, &deps, target_set);
                steps.push(DenseStep::View { node, result, deps });
                continue;
            }
            let slice_copy = graph.nodes.nodes.node_weight(node).and_then(|node_data| {
                let ComputeGraphNodeVariant::Assign(slice_assign) = &node_data.variant else {
                    return None;
                };
                Self::try_prepare_in_place_slice_assign_copy(graph, slice_assign)
            });
            if let Some((output, copies)) = slice_copy {
                graph.set_cached_result(node, output.clone());
                ledger.note_transient(output.buffer());
                for copy in &copies {
                    ledger.note_transient(&copy.source);
                    ledger.note_transient(&copy.destination);
                }
                let mut deps = Vec::new();
                queued_operation.visit_dependencies(&mut |dep| deps.push(dep));
                ledger.consume(graph, &deps, target_set);
                steps.push(DenseStep::CopyAssign {
                    node,
                    copies,
                    op: queued_operation,
                });
                continue;
            }
            match &queued_operation {
                QueuedOperation::Generic(_) | QueuedOperation::QMatMul(_) => {
                    let (mut inputs, output_value) = match &queued_operation {
                        QueuedOperation::Generic(operation) => {
                            let inputs = operation.inputs(graph);
                            let output = operation.output(graph, &inputs);
                            (inputs, output)
                        }
                        QueuedOperation::QMatMul(qmatmul) => {
                            let inputs = qmatmul.inputs(graph);
                            let output = qmatmul.output(graph, &inputs);
                            (inputs, output)
                        }
                        QueuedOperation::Merged(_) => unreachable!("matched above"),
                    };
                    let MirValue::Tensor(mut resolved) = output_value else {
                        panic!("Kernel input value is not a tensor");
                    };
                    // Cache the output before the death accounting: a
                    // source is only releasable once every alive-uncached
                    // descendant (this very operation) is cached.
                    graph.set_cached_result(node, resolved.clone());
                    let mut deps = Vec::new();
                    queued_operation.visit_dependencies(&mut |dep| deps.push(dep));
                    ledger.consume(graph, &deps, target_set);
                    let mut claimed_from = None;
                    if ledger.enabled() {
                        let out_ptr = Arc::as_ptr(resolved.buffer()) as usize;
                        let forbidden: FxHashSet<usize> = inputs
                            .iter()
                            .filter_map(|value| match value {
                                MirValue::Tensor(tensor) => {
                                    let ptr = Arc::as_ptr(tensor.buffer()) as usize;
                                    (ptr != out_ptr).then_some(ptr)
                                }
                                _ => None,
                            })
                            .collect();
                        if let Some(swapped) = ledger.try_claim(node, &resolved, &forbidden) {
                            for value in inputs.iter_mut() {
                                if let MirValue::Tensor(tensor) = value
                                    && Arc::as_ptr(tensor.buffer()) as usize == out_ptr
                                {
                                    *value = swapped.clone().into();
                                }
                            }
                            resolved = swapped;
                            claimed_from = ledger.chosen_source(node);
                            graph.set_cached_result(node, resolved.clone());
                        }
                    }
                    ledger.note_alloc(&resolved);
                    for value in &inputs {
                        if let MirValue::Tensor(tensor) = value {
                            ledger.note_transient(tensor.buffer());
                        }
                    }
                    ledger.note_transient(resolved.buffer());
                    let constraints = match &queued_operation {
                        QueuedOperation::Generic(operation) => {
                            operation.workgroup_shape_constraints(device)
                        }
                        QueuedOperation::QMatMul(qmatmul) => {
                            qmatmul.workgroup_shape_constraints(device)
                        }
                        QueuedOperation::Merged(_) => unreachable!("matched above"),
                    };
                    let workgroup_shape = constraints
                        .solve(max_subgroup_size, &device.limits())
                        .unwrap_or_else(|| {
                            panic!(
                                "Failed to find a valid workgroup shape for constraints {constraints:?}"
                            )
                        });
                    steps.push(DenseStep::Work(work.len()));
                    work.push(DenseWork {
                        node,
                        op: queued_operation,
                        kind: DenseWorkKind::Generic {
                            inputs,
                            workgroup_shape,
                            resolved,
                            claimed_from,
                        },
                        built: std::sync::Mutex::new(None),
                    });
                }
                QueuedOperation::Merged(merged) => {
                    let mut segment_inputs: Vec<Vec<MirValue>> = Vec::new();
                    let mut outputs: Vec<(NodeIndex, TensorData, Option<NodeIndex>)> = Vec::new();
                    if let MergedSegments::Region(segments) = merged {
                        let device = graph.device();
                        // A segment may write an output over one of its own
                        // input buffers only when no other segment of this
                        // dispatch binds that buffer (concurrent workgroups)
                        // — count cached-buffer pointers across the whole
                        // dispatch and require the source to be unique.
                        let mut dispatch_ptr_uses: FxHashMap<usize, u32> = FxHashMap::default();
                        for (_, op) in segments {
                            for idx in &op.inputs {
                                if let Some(cached) = graph.get_cached_result(*idx) {
                                    *dispatch_ptr_uses
                                        .entry(Arc::as_ptr(cached.buffer()) as usize)
                                        .or_insert(0) += 1;
                                }
                            }
                        }
                        // Segments share one unsynchronized dispatch, so a
                        // scratch claim must avoid every segment's reads, not
                        // just the claiming segment's own.
                        let dispatch_reads: FxHashSet<usize> =
                            dispatch_ptr_uses.keys().copied().collect();
                        for (_, op) in segments {
                            let values: Vec<MirValue> = op
                                .inputs
                                .iter()
                                .map(|idx| {
                                    graph
                                        .get_result(*idx)
                                        .expect("region inputs resolve before the region")
                                        .into()
                                })
                                .collect();
                            // Register the gathered input clones before any
                            // claim so the reference accounting that guards
                            // in-place claims sees them.
                            for value in &values {
                                if let MirValue::Tensor(tensor) = value {
                                    ledger.note_transient(tensor.buffer());
                                }
                            }
                            let mut values = values;
                            let reads = op.input_read_summary();
                            let mut slot_claimed = vec![false; op.inputs.len()];
                            // Cache every output before the death accounting:
                            // sources are only releasable once this region
                            // (their last alive-uncached descendant) counts
                            // as cached.
                            let mut fresh_outputs = Vec::new();
                            for statement in &op.statements {
                                let Some(out_node) = statement.output else {
                                    continue;
                                };
                                let output = TensorData::new_for_shape(
                                    &device,
                                    &op.shape,
                                    statement.datatype,
                                );
                                graph.set_cached_result(out_node, output.clone());
                                fresh_outputs.push(output);
                            }
                            {
                                let mut deps = Vec::new();
                                op.visit_dependencies(&mut |dep| deps.push(dep));
                                ledger.consume(graph, &deps, target_set);
                            }
                            let mut fresh_outputs = fresh_outputs.into_iter();
                            for (position, statement) in op.statements.iter().enumerate() {
                                let Some(out_node) = statement.output else {
                                    continue;
                                };
                                let mut output =
                                    fresh_outputs.next().expect("one fresh output per statement");
                                let mut claimed_from = None;
                                // Write in place over an input this statement
                                // is the last reader of: per-thread the load
                                // precedes the store and threads own disjoint
                                // elements, so identity reads stay exact.
                                for (slot, source) in op.inputs.iter().enumerate() {
                                    if slot_claimed[slot]
                                        || !reads[slot].identity_only
                                        || reads[slot].last_reader != Some(position)
                                    {
                                        continue;
                                    }
                                    let unique = graph
                                        .get_cached_result(*source)
                                        .map(|cached| Arc::as_ptr(cached.buffer()) as usize)
                                        .and_then(|ptr| dispatch_ptr_uses.get(&ptr))
                                        == Some(&1);
                                    if !unique {
                                        continue;
                                    }
                                    if let Some(swapped) = ledger.try_claim_in_place(
                                        out_node, &output, *source, graph, target_set,
                                    ) {
                                        output = swapped;
                                        claimed_from = Some(*source);
                                        slot_claimed[slot] = true;
                                        break;
                                    }
                                }
                                if claimed_from.is_none()
                                    && let Some(swapped) =
                                        ledger.try_claim(out_node, &output, &dispatch_reads)
                                {
                                    output = swapped;
                                    claimed_from = ledger.chosen_source(out_node);
                                }
                                if claimed_from.is_some() {
                                    graph.set_cached_result(out_node, output.clone());
                                }
                                ledger.note_alloc(&output);
                                ledger.note_transient(output.buffer());
                                values.push(output.clone().into());
                                outputs.push((out_node, output, claimed_from));
                            }
                            segment_inputs.push(values);
                        }
                    } else {
                        for (seg_node, op) in merged.segment_ops() {
                            let inputs = op.inputs(graph);
                            let MirValue::Tensor(output) = op.output(graph, &inputs) else {
                                panic!("merged segment output is not a tensor");
                            };
                            graph.set_cached_result(seg_node, output.clone());
                            ledger.note_alloc(&output);
                            for value in &inputs {
                                if let MirValue::Tensor(tensor) = value {
                                    ledger.note_transient(tensor.buffer());
                                }
                            }
                            ledger.note_transient(output.buffer());
                            outputs.push((seg_node, output, None));
                            segment_inputs.push(inputs);
                        }
                        let mut deps = Vec::new();
                        queued_operation.visit_dependencies(&mut |dep| deps.push(dep));
                        ledger.consume(graph, &deps, target_set);
                    }
                    steps.push(DenseStep::Work(work.len()));
                    work.push(DenseWork {
                        node,
                        op: queued_operation,
                        kind: DenseWorkKind::Merged {
                            segment_inputs,
                            outputs,
                        },
                        built: std::sync::Mutex::new(None),
                    });
                }
            }
        }
        // Allocation is complete: releases past this point free buffers no
        // claim can use anymore.
        ledger.freeze();
        if let Some(start) = gather_start {
            host_profile.inputs += start.elapsed();
        }

        // Phase 2: build kernels and prepare dispatches in parallel. Builds
        // are pure functions of (operation, layouts, buffers); the shared
        // kernel caches are internally synchronized.
        let build_start = host_trace.then(Instant::now);
        #[cfg(target_arch = "wasm32")]
        for item in &work {
            *item.built.lock().unwrap() =
                Some(build_dense_work(item, graph, device, plan_cache_enabled));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let workers = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(work.len())
                .max(1);
            // Tiny queues build serially: thread spawns cost more than the
            // handful of (usually plan-cached) kernel builds they would
            // parallelize.
            if workers <= 1 || work.len() < 16 {
                for item in &work {
                    *item.built.lock().unwrap() =
                        Some(build_dense_work(item, graph, device, plan_cache_enabled));
                }
            } else {
                let next = std::sync::atomic::AtomicUsize::new(0);
                let graph_ref: &ComputeGraphInner = graph;
                std::thread::scope(|scope| {
                    for _ in 0..workers {
                        scope.spawn(|| {
                            loop {
                                let index =
                                    next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let Some(item) = work.get(index) else { break };
                                let built = build_dense_work(
                                    item,
                                    graph_ref,
                                    device,
                                    plan_cache_enabled,
                                );
                                *item.built.lock().unwrap() = Some(built);
                            }
                        });
                    }
                });
            }
        }
        if let Some(start) = build_start {
            host_profile.build_kernel += start.elapsed();
        }

        // Phase 3: record, encode, and release in queue order.
        let encode_start = host_trace.then(Instant::now);
        for step in steps {
            match step {
                DenseStep::View { node, result, deps } => {
                    if let Some(recorder) = &self.recorder {
                        recorder.borrow_mut().record_view_alias(node, &result, &deps);
                    }
                    Self::release_dead_intermediates_from_graph(
                        graph,
                        &[node],
                        remaining_consumers,
                        target_set,
                        ledger,
                    );
                }
                DenseStep::CopyAssign { node, copies, op } => {
                    if let Some(recorder) = &self.recorder {
                        let output = graph
                            .get_cached_result(node)
                            .expect("copy-assign output cached in phase 1")
                            .clone();
                        recorder.borrow_mut().record_copy_assign(node, &output, &op);
                    }
                    commands.extend(copies.into_iter().map(CommandRecord::CopyBuffer));
                    Self::release_dead_intermediates(
                        graph,
                        &[&op],
                        remaining_consumers,
                        target_set,
                        ledger,
                    );
                }
                DenseStep::Work(index) => {
                    let item = &work[index];
                    let built = item
                        .built
                        .lock()
                        .unwrap()
                        .take()
                        .expect("dense work built in phase 2");
                    if let Some(recorder) = &self.recorder {
                        match (&item.op, &item.kind) {
                            (
                                QueuedOperation::Generic(_),
                                DenseWorkKind::Generic {
                                    resolved,
                                    claimed_from,
                                    ..
                                },
                            ) => {
                                recorder.borrow_mut().record_dispatch(
                                    item.node,
                                    &built.kernels,
                                    resolved,
                                    &item.op,
                                    *claimed_from,
                                );
                            }
                            (
                                QueuedOperation::Merged(merged),
                                DenseWorkKind::Merged { outputs, .. },
                            ) => {
                                if built.merged_ok {
                                    let node_outputs: Vec<(
                                        NodeIndex,
                                        &TensorData,
                                        Option<NodeIndex>,
                                    )> = outputs
                                        .iter()
                                        .map(|(node, output, claimed)| (*node, output, *claimed))
                                        .collect();
                                    recorder.borrow_mut().record_merged_dispatch(
                                        &node_outputs,
                                        &built.kernels,
                                        merged,
                                    );
                                } else {
                                    recorder.borrow_mut().poison();
                                }
                            }
                            (QueuedOperation::QMatMul(_), _) => {
                                // Quantized matmuls are never part of a flush
                                // plan (decode graphs are excluded before
                                // recording arms; this is belt-and-braces).
                                recorder.borrow_mut().poison();
                            }
                            _ => unreachable!("dense work kind matches its queued operation"),
                        }
                    }
                    for prepared in built.prepared {
                        if let Some((dispatch, name)) = prepared {
                            let category = on_dispatch_name(&name);
                            commands.push(CommandRecord::Dispatch(DispatchRecord {
                                dispatch,
                                name,
                                category,
                            }));
                        }
                    }
                    Self::release_dead_intermediates(
                        graph,
                        &[&item.op],
                        remaining_consumers,
                        target_set,
                        ledger,
                    );
                }
            }
        }
        if let Some(start) = encode_start {
            host_profile.prepare_dispatch += start.elapsed();
        }
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
