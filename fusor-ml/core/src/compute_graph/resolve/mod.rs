//! One materialization pipeline for the lazy compute graph.
//!
//! A resolve builds the temporary execution graph, recognizes specialized
//! operations and applies policy-driven fusion through the
//! equality-saturation optimizer (see [`egraph`]), lowers nodes into an
//! operation queue, builds complete kernel plans, and encodes the resulting
//! command records. Flush replay skips deterministic planning but rejoins the
//! same command-record encoder.

use std::sync::Arc;

use web_time::{Duration, Instant};

use crate::{
    DataTypeEnum, Layout,
    mir::{inputs::MirValue, kernel_backend::PreparedDirectDispatch, operation::Operation},
    nary_wise::{ElementwiseOperation, NaryExpr, NaryOp, NaryScalar},
    quantized::matmul::QMatMulOperation,
    tensor::TensorData,
};
use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{ComputeGraphInner, ComputeGraphNode, ComputeGraphNodeVariant, NodeIndex};
use crate::{
    MatMulOperation, ReduceOperation, dequantize::DequantizeOperation,
    quantized::embedding::QEmbeddingOperation, slice_assign::SliceAssignOperation,
    view::ViewOperation,
};

mod alloc_reuse;
mod cluster_match;
mod egraph;
mod execution;
pub(crate) mod flush_replay;
mod fusion_region;
mod fusion_row;
#[cfg(test)]
mod key_goldens;
pub(crate) mod merge_horizontal;
pub(crate) mod plan_cache;
mod queue_executor;
mod recognize;
mod recognize_attention;
mod recognize_cat;
#[cfg(test)]
mod recognize_gates;
mod run;
#[cfg(feature = "graphvis")]
mod visualize;

pub(crate) use egraph::FusionPlanStore;
pub(crate) use plan_cache::structural_kernel_key;

pub(crate) struct ResolverResult {
    pub(crate) data: TensorData,
    pub(crate) total_kernels: usize,
}

#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
struct DispatchRecord {
    dispatch: PreparedDirectDispatch,
    name: String,
    category: Option<String>,
}

#[cfg(not(target_arch = "wasm32"))]
struct DispatchMetadata {
    name: Option<String>,
    category: Option<String>,
}

struct CopyBufferRecord {
    source: Arc<wgpu::Buffer>,
    destination: Arc<wgpu::Buffer>,
    source_offset: u64,
    destination_offset: u64,
    size: u64,
}

enum CommandRecord {
    Dispatch(DispatchRecord),
    CopyBuffer(CopyBufferRecord),
}

enum QueuedOperation {
    Operation(Arc<dyn Operation>),
    /// Independent compatible operations merged into one dispatch (see
    /// `merge_horizontal`).
    Merged(merge_horizontal::MergedSegments),
}

impl QueuedOperation {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        match self {
            Self::Operation(operation) => operation.visit_dependencies(f),
            Self::Merged(merged) => merged.visit_dependencies(f),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Default)]
struct KernelProfileAggregate {
    count: usize,
    total_ns: f64,
    max_ns: f64,
}

#[cfg(not(target_arch = "wasm32"))]
impl KernelProfileAggregate {
    fn record(&mut self, ns: f64) {
        self.count += 1;
        self.total_ns += ns;
        self.max_ns = self.max_ns.max(ns);
    }
}

#[derive(Default)]
struct ResolveHostProfile {
    build_execution_graph: Duration,
    optimize: Duration,
    toposort: Duration,
    queue_lowering: Duration,
    consumer_count: Duration,
    encoder_create: Duration,
    map_layout: Duration,
    inputs: Duration,
    output: Duration,
    workgroup: Duration,
    build_kernel: Duration,
    prepare_dispatch: Duration,
    release: Duration,
    timestamp_setup: Duration,
    encode: Duration,
    submit: Duration,
    profile_readback: Duration,
}

impl ResolveHostProfile {
    fn print(&self, total: Duration, queued_ops: usize, kernels: usize) {
        tracing::info!(
            "resolve_host_profile queued_ops={queued_ops} kernels={kernels} total={total:?} \
build_execution_graph={:?} optimize={:?} toposort={:?} queue_lowering={:?} \
consumer_count={:?} encoder_create={:?} map_layout={:?} inputs={:?} output={:?} \
workgroup={:?} build_kernel={:?} prepare_dispatch={:?} release={:?} \
timestamp_setup={:?} encode={:?} submit={:?} profile_readback={:?}",
            self.build_execution_graph,
            self.optimize,
            self.toposort,
            self.queue_lowering,
            self.consumer_count,
            self.encoder_create,
            self.map_layout,
            self.inputs,
            self.output,
            self.workgroup,
            self.build_kernel,
            self.prepare_dispatch,
            self.release,
            self.timestamp_setup,
            self.encode,
            self.submit,
            self.profile_readback,
        );
    }
}

#[allow(dead_code, reason = "execution-side category labeling for profiling")]
/// What an execution-graph node lowers to. The graph vocabulary (the first
/// six variants) enters verbatim; the region variants exist only here —
/// recognition rebuilds them from composed clusters, and fusion enriches
/// them with epilogues.
#[derive(Debug, Clone)]
pub(crate) enum ExecutionVariant {
    Tensor(crate::tensor::TensorData),
    QMatrix(DequantizeOperation),
    Elementwise(ElementwiseOperation),
    Reduce(ReduceOperation),
    View(ViewOperation),
    Assign(SliceAssignOperation),
    /// Multi-output elementwise region formed by `fusion_region` on the
    /// dense branch; never present in the inner graph.
    Region(crate::region::ElementwiseRegionOperation),
    // Recognized regions.
    MatMul(MatMulOperation),
    QMatMul(Box<QMatMulOperation>),
    QEmbedding(QEmbeddingOperation),
    RowProgram(crate::row_program::RowProgramOperation),
    Attention(crate::flash_attention::FlashAttentionOperation),
}

impl ExecutionVariant {
    /// Dependencies in dependency-slot order: the order the e-graph mirrors
    /// as e-node children. Tensor leaves have none.
    pub(super) fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        match self {
            Self::Tensor(_) => {}
            Self::QMatrix(op) => op.visit_dependencies(f),
            Self::Elementwise(op) => op.visit_dependencies(f),
            Self::Reduce(op) => op.visit_dependencies(f),
            Self::View(op) => op.visit_dependencies(f),
            Self::Assign(op) => op.visit_dependencies(f),
            Self::Region(op) => op.visit_dependencies(f),
            Self::MatMul(op) => op.visit_dependencies(f),
            Self::QMatMul(op) => op.visit_dependencies(f),
            Self::QEmbedding(op) => op.visit_dependencies(f),
            Self::RowProgram(op) => op.visit_dependencies(f),
            Self::Attention(op) => op.visit_dependencies(f),
        }
    }

    /// The same slots as [`Self::visit_dependencies`], in the same order, as
    /// rebindable references.
    pub(super) fn visit_dependencies_mut(&mut self, f: &mut dyn FnMut(&mut NodeIndex)) {
        match self {
            Self::Tensor(_) => {}
            Self::QMatrix(op) => op.visit_dependencies_mut(f),
            Self::Elementwise(op) => op.visit_dependencies_mut(f),
            Self::Reduce(op) => op.visit_dependencies_mut(f),
            Self::View(op) => op.visit_dependencies_mut(f),
            Self::Assign(op) => op.visit_dependencies_mut(f),
            Self::Region(op) => op.visit_dependencies_mut(f),
            Self::MatMul(op) => op.visit_dependencies_mut(f),
            Self::QMatMul(op) => op.visit_dependencies_mut(f),
            Self::QEmbedding(op) => op.visit_dependencies_mut(f),
            Self::RowProgram(op) => op.visit_dependencies_mut(f),
            Self::Attention(op) => op.visit_dependencies_mut(f),
        }
    }
}

impl From<ComputeGraphNodeVariant> for ExecutionVariant {
    fn from(variant: ComputeGraphNodeVariant) -> Self {
        match variant {
            ComputeGraphNodeVariant::Tensor(op) => Self::Tensor(op),
            ComputeGraphNodeVariant::QMatrix(op) => Self::QMatrix(op),
            ComputeGraphNodeVariant::Elementwise(op) => Self::Elementwise(op),
            ComputeGraphNodeVariant::Reduce(op) => Self::Reduce(op),
            ComputeGraphNodeVariant::View(op) => Self::View(op),
            ComputeGraphNodeVariant::Assign(op) => Self::Assign(op),
        }
    }
}

#[derive(Debug, Clone)]
struct ExecutionNode {
    inner_idx: NodeIndex,
    variant: ExecutionVariant,
}

type ExecutionGraph = StableGraph<ExecutionNode, ()>;
type ExecutionNodeIndex = petgraph::graph::NodeIndex;

fn dispatch_category(name: &str) -> String {
    name.split('_').take(2).collect::<Vec<_>>().join("_")
}

fn padded_query_buffer_size(size: u64) -> u64 {
    let align_mask = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT - 1;
    ((size + align_mask) & !align_mask).max(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT)
}

#[cfg(not(target_arch = "wasm32"))]
fn collect_gpu_kernel_profile(
    records: &[DispatchMetadata],
    timestamps: &[u64],
    timestamp_period_ns: f64,
    timestamp_mode: &'static str,
) -> crate::KernelProfile {
    let mut category_profile = FxHashMap::<String, KernelProfileAggregate>::default();
    let mut name_profile = FxHashMap::<String, KernelProfileAggregate>::default();
    let mut accounted_ns = 0.0;

    for (index, record) in records.iter().enumerate() {
        let begin = timestamps.get(index * 2).copied().unwrap_or_default();
        let end = timestamps.get(index * 2 + 1).copied().unwrap_or(begin);
        let ns = end.saturating_sub(begin) as f64 * timestamp_period_ns;
        accounted_ns += ns;
        if let Some(category) = &record.category {
            category_profile
                .entry(category.clone())
                .or_default()
                .record(ns);
        }
        if let Some(name) = &record.name {
            name_profile.entry(name.clone()).or_default().record(ns);
        }
    }

    let span_ns = match (timestamps.first(), timestamps.last()) {
        (Some(first), Some(last)) => last.saturating_sub(*first) as f64 * timestamp_period_ns,
        _ => 0.0,
    };

    let rows = |profile: FxHashMap<String, KernelProfileAggregate>| {
        let mut rows = profile
            .into_iter()
            .map(|(name, aggregate)| crate::KernelProfileRow {
                name,
                count: aggregate.count,
                total_ms: aggregate.total_ns / 1_000_000.0,
                average_us: aggregate.total_ns / aggregate.count as f64 / 1_000.0,
                max_us: aggregate.max_ns / 1_000.0,
            })
            .collect::<Vec<_>>();
        rows.sort_by(|a, b| {
            b.total_ms
                .partial_cmp(&a.total_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        rows
    };
    let categories = rows(category_profile);
    let mut top_names = rows(name_profile);
    top_names.truncate(32);

    let profile = crate::KernelProfile {
        timestamp_mode,
        kernels: records.len(),
        accounted_ms: accounted_ns / 1_000_000.0,
        span_ms: span_ns / 1_000_000.0,
        timestamp_period_ns,
        categories,
        top_names,
    };
    log_gpu_kernel_profile(&profile);
    profile
}

#[cfg(not(target_arch = "wasm32"))]
fn log_gpu_kernel_profile(profile: &crate::KernelProfile) {
    let tuples = |rows: &[crate::KernelProfileRow]| {
        rows.iter()
            .map(|row| {
                (
                    row.name.clone(),
                    row.count,
                    row.total_ms,
                    row.average_us,
                    row.max_us,
                )
            })
            .collect::<Vec<_>>()
    };
    tracing::info!(
        "resolve_gpu_kernel_profile mode={} kernels={} accounted_ms={:.3} span_ms={:.3} timestamp_period_ns={:.3}",
        profile.timestamp_mode,
        profile.kernels,
        profile.accounted_ms,
        profile.span_ms,
        profile.timestamp_period_ns
    );
    let categories = tuples(&profile.categories);
    tracing::info!("resolve_gpu_kernel_categories {categories:?}");
    let names = tuples(&profile.top_names);
    tracing::info!("resolve_gpu_kernel_top_names {names:?}");
}

pub(crate) struct Resolver {
    execution_graph: ExecutionGraph,
    node_mapping: FxHashMap<NodeIndex, ExecutionNodeIndex>,
    targets: Vec<NodeIndex>,
    resolved_set: FxHashSet<NodeIndex>,
    // Materialization-plan recorder, armed on the first occurrence of a
    // structurally cacheable target set. Dense and quantized graphs share it.
    // `RefCell` because some hook sites (`add_physical_dependencies`) only
    // hold `&self`.
    recorder: Option<std::cell::RefCell<flush_replay::PlanRecorder>>,
    // Compatible independent operations may merge into one dispatch.
    /// One semantic e-class may satisfy several lazy graph observations.
    /// Keys are the execution nodes that materialize; values receive the
    /// same allocation without another dispatch.
    shared_outputs: FxHashMap<NodeIndex, Vec<NodeIndex>>,
    /// Wall-clock spent in each optimizer sub-phase of this resolve, for the
    /// host-cost ledger printed under `FUSOR_TRACE_RESOLVE_HOST`.
    optimize_phases: execution::OptimizePhases,
}

impl Resolver {
    pub(crate) fn new(graph: &mut ComputeGraphInner, target: NodeIndex) -> Self {
        Self::new_batch(graph, vec![target])
    }

    pub(crate) fn new_batch(graph: &mut ComputeGraphInner, targets: Vec<NodeIndex>) -> Self {
        let resolved_set = graph
            .nodes
            .nodes
            .node_indices()
            .filter(|&idx| {
                graph
                    .nodes
                    .nodes
                    .node_weight(idx)
                    .map(|n| n.cached.is_some())
                    .unwrap_or(false)
            })
            .collect();
        Self {
            targets,
            execution_graph: Default::default(),
            node_mapping: Default::default(),
            resolved_set,
            recorder: None,
            shared_outputs: Default::default(),
            optimize_phases: Default::default(),
        }
    }

    /// A batch resolver that additionally records a replayable
    /// [`flush_replay::FlushPlan`] of everything it resolves.
    pub(crate) fn new_batch_with_recording(
        graph: &mut ComputeGraphInner,
        targets: Vec<NodeIndex>,
        fingerprint: flush_replay::FlushFingerprint,
    ) -> Self {
        let recorder = flush_replay::PlanRecorder::new(graph, &targets, fingerprint);
        let mut resolver = Self::new_batch(graph, targets);
        resolver.recorder = Some(std::cell::RefCell::new(recorder));
        resolver
    }

    /// The recorded plan, if recording was armed and never poisoned.
    pub(crate) fn take_recorded_plan(&mut self) -> Option<flush_replay::FlushPlan> {
        self.recorder
            .take()
            .and_then(|recorder| recorder.into_inner().finish())
    }
}
