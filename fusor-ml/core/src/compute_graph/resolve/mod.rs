use std::{
    collections::VecDeque,
    str::FromStr,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    DataTypeEnum, Layout,
    compute_graph::layout_pass::LayoutPass,
    mir::{inputs::MirValue, kernel_backend::PreparedDirectDispatch, operation::Operation},
    nary_direct::eval_nary_expr_on_tiles,
    nary_wise::{ExtractedUnaryChain, NaryExpr, NaryOperation, UnaryFunctionChain},
    quantized::matmul::{ElementwiseEpilogue, QMatMulOperation},
    tensor::TensorData,
};
use fusor_gguf::GgmlType;
use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{ComputeGraphInner, ComputeGraphNode, ComputeGraphNodeVariant, NodeIndex};

mod execution;
mod fusion_basic;
mod fusion_matmul;
mod fusion_paired;
mod run;

pub(crate) struct ResolverResult {
    pub(crate) data: TensorData,
    pub(crate) total_kernels: usize,
}

struct DispatchRecord {
    dispatch: PreparedDirectDispatch,
    name: Option<String>,
    category: Option<String>,
}

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
    Generic(Arc<dyn Operation>),
    QMatMul(Box<QMatMulOperation>),
}

impl QueuedOperation {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        match self {
            Self::Generic(operation) => operation.visit_dependencies(f),
            Self::QMatMul(operation) => operation.visit_dependencies(f),
        }
    }

    fn inputs(&self, graph: &ComputeGraphInner) -> Vec<MirValue> {
        match self {
            Self::Generic(operation) => operation.inputs(graph),
            Self::QMatMul(operation) => operation.inputs(graph),
        }
    }
}

#[derive(Default)]
struct KernelProfileAggregate {
    count: usize,
    total_ns: f64,
    max_ns: f64,
}

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

#[derive(Default)]
struct ResolveHostCategoryProfile {
    count: usize,
    inputs: Duration,
    output: Duration,
    workgroup: Duration,
    build_kernel: Duration,
    prepare_dispatch: Duration,
}

#[derive(Default)]
struct OptimizeProfile {
    iterations: usize,
    changed: usize,
    fuse_naries_count: usize,
    fuse_naries: Duration,
    fuse_reduce_count: usize,
    fuse_reduce: Duration,
    fuse_matmul_count: usize,
    fuse_matmul: Duration,
    fuse_paired_qmatmul_count: usize,
    fuse_paired_qmatmul: Duration,
    fuse_rmsnorm_count: usize,
    fuse_rmsnorm: Duration,
}

impl OptimizeProfile {
    fn print(&self) {
        eprintln!(
            "resolve_optimize_profile iterations={} changed={} \
fuse_naries_count={} fuse_naries={:?} \
fuse_reduce_count={} fuse_reduce={:?} \
fuse_matmul_count={} fuse_matmul={:?} \
fuse_paired_qmatmul_count={} fuse_paired_qmatmul={:?} \
fuse_rmsnorm_count={} fuse_rmsnorm={:?}",
            self.iterations,
            self.changed,
            self.fuse_naries_count,
            self.fuse_naries,
            self.fuse_reduce_count,
            self.fuse_reduce,
            self.fuse_matmul_count,
            self.fuse_matmul,
            self.fuse_paired_qmatmul_count,
            self.fuse_paired_qmatmul,
            self.fuse_rmsnorm_count,
            self.fuse_rmsnorm,
        );
    }
}

const DEFAULT_OPTIMIZE_NODE_LIMIT: usize = 512;

fn optimize_node_limit() -> usize {
    std::env::var("FUSOR_RESOLVE_OPTIMIZE_MAX_NODES")
        .ok()
        .and_then(|value| usize::from_str(&value).ok())
        .unwrap_or(DEFAULT_OPTIMIZE_NODE_LIMIT)
}

impl ResolveHostProfile {
    fn print(&self, total: Duration, queued_ops: usize, kernels: usize) {
        eprintln!(
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

fn print_host_category_profile(profile: FxHashMap<&'static str, ResolveHostCategoryProfile>) {
    let mut profile = profile
        .into_iter()
        .map(|(category, profile)| {
            (
                category,
                profile.count,
                profile.inputs,
                profile.output,
                profile.workgroup,
                profile.build_kernel,
                profile.prepare_dispatch,
            )
        })
        .collect::<Vec<_>>();
    profile.sort_by_key(|entry| std::cmp::Reverse(entry.5));
    eprintln!("resolve_host_category_profile {profile:?}");
}

fn node_category(variant: &ComputeGraphNodeVariant) -> &'static str {
    match variant {
        ComputeGraphNodeVariant::Nary(_) => "nary",
        ComputeGraphNodeVariant::SliceAssign(_) => "slice_assign",
        ComputeGraphNodeVariant::Resize(_) => "resize",
        ComputeGraphNodeVariant::MapLayout(_) => "map_layout",
        ComputeGraphNodeVariant::Dequantize(_) => "dequantize",
        ComputeGraphNodeVariant::QEmbedding(_) => "q_embedding",
        ComputeGraphNodeVariant::MatMul(_) => "matmul",
        ComputeGraphNodeVariant::QMatMul(op) => {
            if op.paired.is_some() {
                "q_mat_paired"
            } else {
                "q_matmul"
            }
        }
        ComputeGraphNodeVariant::Tensor(_) => "tensor",
        ComputeGraphNodeVariant::Reduce(_) => "reduce",
        ComputeGraphNodeVariant::FlashAttention(_) => "flash_attention",
        ComputeGraphNodeVariant::GraphOp(op) => op.category(),
    }
}

fn as_rms_norm(variant: &ComputeGraphNodeVariant) -> Option<&crate::RmsNormOperation> {
    let ComputeGraphNodeVariant::GraphOp(op) = variant else {
        return None;
    };
    op.as_any().downcast_ref::<crate::RmsNormOperation>()
}

#[derive(Debug, Clone)]
struct ExecutionNode {
    inner_idx: NodeIndex,
    variant: ComputeGraphNodeVariant,
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

    eprintln!(
        "resolve_gpu_kernel_profile mode={} kernels={} accounted_ms={:.3} span_ms={:.3} timestamp_period_ns={:.3}",
        timestamp_mode,
        records.len(),
        accounted_ns / 1_000_000.0,
        span_ns / 1_000_000.0,
        timestamp_period_ns
    );
    eprintln!("resolve_gpu_kernel_categories {categories:?}");
    eprintln!("resolve_gpu_kernel_top_names {names:?}");
}

pub(crate) struct Resolver {
    execution_graph: ExecutionGraph,
    node_mapping: FxHashMap<NodeIndex, ExecutionNodeIndex>,
    layout_cache: FxHashMap<NodeIndex, Option<crate::TensorLayoutInfo>>,
    targets: Vec<NodeIndex>,
    resolved_set: FxHashSet<NodeIndex>,
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
            layout_cache: Default::default(),
            resolved_set,
        }
    }
}
