//! One materialization pipeline for the lazy compute graph.
//!
//! A resolve builds the temporary execution graph, recognizes specialized
//! operations and applies policy-driven fusion through the
//! equality-saturation optimizer (see [`egraph`]), lowers nodes into an
//! operation queue, builds complete kernel plans, and encodes the resulting
//! command records. Flush replay skips deterministic planning but rejoins the
//! same command-record encoder.

use std::{str::FromStr, sync::Arc};

use web_time::{Duration, Instant};

use crate::{
    DataTypeEnum, Layout,
    mir::{inputs::MirValue, kernel_backend::PreparedDirectDispatch, operation::Operation},
    nary_wise::{ElementwiseOperation, ExtractedUnaryChain, NaryExpr, NaryOp, NaryScalar},
    quantized::matmul::QMatMulOperation,
    tensor::TensorData,
};
use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    ComputeGraphInner, ComputeGraphNode, ComputeGraphNodeVariant, GraphOperation, NodeIndex,
};
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
mod fold_views;
mod fusion_basic;
mod fusion_matmul;
mod fusion_region;
mod fusion_row;
pub(crate) mod merge_horizontal;
mod plan_cache;
mod queue_executor;
mod recognize;
mod recognize_attention;
mod recognize_cat;
mod run;

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

const LARGE_GRAPH_NARY_FUSION_MIN_LAST_DIM: usize = 512;

const DEFAULT_OPTIMIZE_NODE_LIMIT: usize = 512;

fn optimize_node_limit() -> usize {
    std::env::var("FUSOR_RESOLVE_OPTIMIZE_MAX_NODES")
        .ok()
        .and_then(|value| usize::from_str(&value).ok())
        .unwrap_or(DEFAULT_OPTIMIZE_NODE_LIMIT)
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
    GraphOp(Arc<dyn GraphOperation>),
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
fn print_gpu_kernel_profile(
    records: &[DispatchMetadata],
    timestamps: &[u64],
    timestamp_period_ns: f64,
    timestamp_mode: &str,
) {
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

    let mut categories = category_profile
        .into_iter()
        .map(|(name, aggregate)| {
            (
                name,
                aggregate.count,
                aggregate.total_ns / 1_000_000.0,
                aggregate.total_ns / aggregate.count as f64 / 1_000.0,
                aggregate.max_ns / 1_000.0,
            )
        })
        .collect::<Vec<_>>();
    categories.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut names = name_profile
        .into_iter()
        .map(|(name, aggregate)| {
            (
                name,
                aggregate.count,
                aggregate.total_ns / 1_000_000.0,
                aggregate.total_ns / aggregate.count as f64 / 1_000.0,
                aggregate.max_ns / 1_000.0,
            )
        })
        .collect::<Vec<_>>();
    names.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    names.truncate(32);

    tracing::info!(
        "resolve_gpu_kernel_profile mode={} kernels={} accounted_ms={:.3} span_ms={:.3} timestamp_period_ns={:.3}",
        timestamp_mode,
        records.len(),
        accounted_ns / 1_000_000.0,
        span_ns / 1_000_000.0,
        timestamp_period_ns
    );
    tracing::info!("resolve_gpu_kernel_categories {categories:?}");
    tracing::info!("resolve_gpu_kernel_top_names {names:?}");
}

pub(crate) struct Resolver {
    execution_graph: ExecutionGraph,
    node_mapping: FxHashMap<NodeIndex, ExecutionNodeIndex>,
    targets: Vec<NodeIndex>,
    resolved_set: FxHashSet<NodeIndex>,
    // Materialization-plan recorder, armed only on the second sighting of an
    // isomorphic QMatMul-free target set. `None` on first-seen and quantized
    // resolve paths, so decode resolves never record.
    // `RefCell` because some hook sites (`add_physical_dependencies`) only
    // hold `&self`.
    recorder: Option<std::cell::RefCell<flush_replay::PlanRecorder>>,
    // QMatMul-free graphs may merge independent cooperative matmuls. The
    // large dense profile additionally merges row and elementwise work.
    horizontal_merge: bool,
    horizontal_merge_dense_ops: bool,
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
            horizontal_merge: false,
            horizontal_merge_dense_ops: false,
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
