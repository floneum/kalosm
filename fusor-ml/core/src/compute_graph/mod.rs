use std::{any::Any, sync::Arc};

use parking_lot::RwLock;
pub use petgraph::graph::NodeIndex;
use petgraph::prelude::StableGraph;
use petgraph::visit::EdgeRef;
use resolve::Resolver;
use rustc_hash::FxHashMap;
#[cfg(feature = "graphvis")]
use tabbycat::Graph;

mod layout_pass;
mod queue;
mod resolve;
#[cfg(test)]
mod tests;
#[cfg(feature = "graphvis")]
mod visualize;

use crate::{
    DataTypeEnum, Device, FlashAttentionOperation, MatMulOperation, QMatrix, ReduceOperation,
    RmsNormOperation,
    compute_graph::resolve::ResolverResult,
    dequantize::DequantizeOperation,
    map_layout::MapLayoutOperation,
    mir::{inputs::MirValue, operation::Operation},
    nary_wise::NaryOperation,
    quantized::embedding::QEmbeddingOperation,
    quantized::matmul::QMatMulOperation,
    resize::ResizeOperation,
    slice_assign::SliceAssignOperation,
    tensor::{TensorData, TensorLayoutInfo},
    visit_tiled::MaybeQData,
};

#[derive(Clone)]
pub(crate) struct ComputeGraph {
    inner: Arc<RwLock<ComputeGraphInner>>,
}

impl ComputeGraph {
    pub(crate) fn new(device: &Device) -> Self {
        let inner = Arc::new(RwLock::new(ComputeGraphInner::new(device)));
        Self { inner }
    }

    fn with_mut<R, F: FnOnce(&mut ComputeGraphInner) -> R>(&self, f: F) -> R {
        let mut inner = self.inner.write();
        let result = f(&mut inner);
        #[cfg(feature = "extra_assertions")]
        {
            inner.verify_integrity()
        }
        result
    }

    fn create_node(&self, node: ComputeGraphNodeVariant) -> NodeIndex {
        self.with_mut(|inner| inner.create_node(node))
    }

    pub(crate) fn create_nary(&self, op: NaryOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Nary(op))
    }

    pub(crate) fn create_mat_mul(&self, op: MatMulOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::MatMul(op))
    }

    pub(crate) fn create_q_mat_mul(&self, op: QMatMulOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::QMatMul(Box::new(op)))
    }

    pub(crate) fn create_q_embedding(&self, op: QEmbeddingOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::QEmbedding(op))
    }

    pub(crate) fn create_reduce(&self, op: ReduceOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Reduce(op))
    }

    pub(crate) fn create_graph_op(&self, op: Arc<dyn GraphOperation>) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::GraphOp(op))
    }

    pub(crate) fn create_rms_norm(&self, op: RmsNormOperation) -> NodeIndex {
        self.create_graph_op(Arc::new(op))
    }

    pub(crate) fn create_flash_attention(&self, op: FlashAttentionOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::FlashAttention(op))
    }

    pub(crate) fn create_map_layout(&self, op: MapLayoutOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::MapLayout(op))
    }

    pub(crate) fn create_resize(&self, op: ResizeOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Resize(op))
    }

    pub(crate) fn create_slice_assign(&self, op: SliceAssignOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::SliceAssign(op))
    }

    pub(crate) fn create_tensor(&self, op: TensorData) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Tensor(op))
    }

    pub(crate) fn dequantize(&self, matrix: QMatrix, ty: DataTypeEnum) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Dequantize(
            DequantizeOperation::new(matrix, ty),
        ))
    }

    pub(crate) fn resolve(&self, key: NodeIndex) -> ResolverResult {
        if let Some(data) = {
            let inner = self.inner.read();
            inner.get_cached_result(key).cloned()
        } {
            return ResolverResult {
                data,
                total_kernels: 0,
            };
        }

        if let Some(data) = {
            let mut inner = self.inner.write();
            let data = inner.try_resolve_direct_qmatmul(key);
            #[cfg(feature = "extra_assertions")]
            {
                inner.verify_integrity()
            }
            data
        } {
            return data;
        }

        let (data, removed) = {
            let mut inner = self.inner.write();
            let mut removed = Vec::new();
            let mut resolver = Resolver::new(&mut inner, key);
            let data = resolver.run(&mut inner, &mut removed);
            inner.try_auto_flush(&mut removed);
            #[cfg(feature = "extra_assertions")]
            {
                inner.verify_integrity()
            }
            (data, removed)
        };
        // Drop removed nodes now that the resolver has submitted its commands.
        drop(removed);

        data
    }

    /// Resolve multiple targets in a single pass. Since sequential `resolve()`
    /// now reuses kernels across sibling targets (via the
    /// `live_descendant_count` predicate in the freeing path), this is
    /// primarily a hint to share one command-encoder submission.
    pub(crate) fn resolve_batch(&self, keys: &[NodeIndex]) -> usize {
        if keys.is_empty() {
            return 0;
        }
        let trace = std::env::var_os("FUSOR_TRACE_DECODE").is_some()
            || std::env::var_os("FUSOR_TRACE_RESOLVE").is_some();
        let start = trace.then(std::time::Instant::now);
        let total_kernels;
        let removed = {
            let mut inner = self.inner.write();
            let mut removed = Vec::new();
            let mut resolver = Resolver::new_batch(&mut inner, keys.to_vec());
            let result = resolver.run(&mut inner, &mut removed);
            total_kernels = result.total_kernels;
            if let Some(start) = start {
                eprintln!(
                    "resolve_batch keys={} kernels={} elapsed={:?}",
                    keys.len(),
                    result.total_kernels,
                    start.elapsed()
                );
            }
            inner.try_auto_flush(&mut removed);
            #[cfg(feature = "extra_assertions")]
            {
                inner.verify_integrity()
            }
            removed
        };
        drop(removed);
        total_kernels
    }

    #[cfg(feature = "graphvis")]
    pub(crate) fn graphvis(&self, root: NodeIndex) -> Graph {
        self.with_mut(|inner| inner.graphvis(root))
    }

    pub(crate) fn add_reference(&self, key: NodeIndex) {
        self.with_mut(|inner| inner.add_reference(key));
    }

    pub(crate) fn remove_reference(&self, key: NodeIndex) {
        let removed = {
            let mut inner = self.inner.write();
            let mut removed = Vec::new();
            inner.remove_reference(key, &mut removed);
            #[cfg(feature = "extra_assertions")]
            {
                inner.verify_integrity()
            }
            removed
        };
        drop(removed);
    }

    pub(crate) fn detach_cached(&self, keys: &[NodeIndex]) {
        let removed = {
            let mut inner = self.inner.write();
            let mut removed = Vec::new();
            inner.detach_cached(keys, &mut removed);
            #[cfg(feature = "extra_assertions")]
            {
                inner.verify_integrity()
            }
            removed
        };
        drop(removed);
    }

    #[cfg(test)]
    pub(crate) fn set_flush_threshold(&self, threshold: usize) {
        self.inner.write().flush_threshold = threshold;
    }

    #[cfg(test)]
    pub(crate) fn node_count(&self) -> usize {
        self.inner.read().nodes.nodes.node_count()
    }

    #[cfg(test)]
    pub(crate) fn live_descendant_count(&self, key: NodeIndex) -> u32 {
        self.inner
            .read()
            .nodes
            .nodes
            .node_weight(key)
            .map(|n| n.live_descendant_count)
            .unwrap_or(0)
    }

    #[cfg(test)]
    pub(crate) fn is_cached_for_test(&self, key: NodeIndex) -> bool {
        self.inner
            .read()
            .nodes
            .nodes
            .node_weight(key)
            .map(|n| n.cached.is_some())
            .unwrap_or(false)
    }

    #[cfg(test)]
    pub(crate) fn cached_node_count(&self) -> usize {
        let inner = self.inner.read();
        inner
            .nodes
            .nodes
            .node_indices()
            .filter(|idx| {
                inner
                    .nodes
                    .nodes
                    .node_weight(*idx)
                    .map(|n| n.cached.is_some())
                    .unwrap_or(false)
            })
            .count()
    }
}

#[derive(Default)]
pub(crate) struct ComputeGraphNodes {
    pub(crate) nodes: StableGraph<ComputeGraphNode, ()>,
}

pub(crate) struct ComputeGraphNode {
    variant: ComputeGraphNodeVariant,
    reference_count: u32,
    // Number of outgoing edges to children that are currently
    // `alive_uncached()` (see below). Maintained eagerly; lets the resolver
    // free intermediates only when no user-held lazy tensor still needs this
    // node's result to be re-computed. Sequential `resolve()` calls can then
    // reuse shared ancestors instead of recomputing them, matching
    // `resolve_batch`. A descendant that has already been resolved (cached)
    // no longer contributes, so deep chains where only the final tensor is
    // held still free intermediates eagerly during the resolve.
    live_descendant_count: u32,
    cached: Option<TensorData>,
}

impl ComputeGraphNode {
    /// True iff this node is still uncached AND has a path to a user-held
    /// `LazyTensorData` (directly or transitively). Drives counter
    /// propagation: a parent counts this child in its
    /// `live_descendant_count` iff `alive_uncached() == true`.
    fn alive_uncached(&self) -> bool {
        self.cached.is_none() && (self.reference_count > 0 || self.live_descendant_count > 0)
    }

    /// True iff this node's `cached` buffer should be preserved past the
    /// current resolve: either user code holds a `LazyTensorData` for it, or
    /// some still-uncached live descendant will benefit from it on a future
    /// resolve. Independent of this node's own `cached` state.
    fn should_keep_cached(&self) -> bool {
        self.reference_count > 0 || self.live_descendant_count > 0
    }
}

pub(crate) trait GraphOperation: Operation + Send + Sync {
    fn as_any(&self) -> &dyn Any;

    fn category(&self) -> &'static str;

    fn output_layout(
        &self,
        input_layouts: &FxHashMap<NodeIndex, TensorLayoutInfo>,
    ) -> Option<TensorLayoutInfo>;
}

#[derive(Clone, Debug)]
pub(crate) enum ComputeGraphNodeVariant {
    Nary(NaryOperation),
    SliceAssign(SliceAssignOperation),
    Resize(ResizeOperation),
    MapLayout(MapLayoutOperation),
    Dequantize(DequantizeOperation),
    QEmbedding(QEmbeddingOperation),
    MatMul(MatMulOperation),
    QMatMul(Box<QMatMulOperation>),
    Tensor(TensorData),
    Reduce(ReduceOperation),
    FlashAttention(FlashAttentionOperation),
    GraphOp(Arc<dyn GraphOperation>),
}

impl ComputeGraphNodeVariant {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        match &self {
            ComputeGraphNodeVariant::Nary(op) => {
                for input in &op.inputs {
                    f(*input);
                }
            }
            ComputeGraphNodeVariant::MatMul(op) => {
                f(op.first);
                f(op.second);
            }
            ComputeGraphNodeVariant::QMatMul(op) => {
                f(op.input);
                if let Some(epilogue) = &op.pre_element_wise_expr {
                    for extra in &epilogue.extras {
                        f(*extra);
                    }
                }
                if let Some(epilogue) = &op.post_element_wise_expr {
                    for extra in &epilogue.extras {
                        f(*extra);
                    }
                }
                if let Some(paired) = &op.paired {
                    for extra in &paired.extras {
                        f(*extra);
                    }
                }
            }
            ComputeGraphNodeVariant::QEmbedding(op) => {
                f(op.indexes);
            }
            ComputeGraphNodeVariant::Reduce(op) => f(op.value),
            ComputeGraphNodeVariant::FlashAttention(op) => op.visit_dependencies(f),
            ComputeGraphNodeVariant::GraphOp(op) => op.visit_dependencies(f),
            ComputeGraphNodeVariant::MapLayout(op) => f(op.input),
            ComputeGraphNodeVariant::Resize(op) => f(op.input),
            ComputeGraphNodeVariant::SliceAssign(op) => {
                f(op.input);
                f(op.value);
            }
            ComputeGraphNodeVariant::Dequantize(_) => {}
            ComputeGraphNodeVariant::Tensor(_) => {}
        }
    }
}

pub(crate) struct ComputeGraphInner {
    pub(crate) device: crate::WeakDevice,
    pub(crate) nodes: ComputeGraphNodes,
    // Auto-flush all pending lazy outputs once the graph grows past this many
    // nodes. Bounds memory growth on fully-lazy loops (e.g. vision encoders)
    // where the user would otherwise need to sprinkle explicit `resolve()`
    // calls. 0 disables.
    flush_threshold: usize,
}

const DEFAULT_FLUSH_THRESHOLD: usize = 8192;

fn read_flush_threshold() -> usize {
    std::env::var("FUSOR_GRAPH_FLUSH_THRESHOLD")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_FLUSH_THRESHOLD)
}

impl ComputeGraphInner {
    fn new(device: &Device) -> Self {
        Self {
            device: device.downgrade(),
            nodes: ComputeGraphNodes::default(),
            flush_threshold: read_flush_threshold(),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(device: crate::WeakDevice) -> Self {
        Self {
            device,
            nodes: ComputeGraphNodes::default(),
            flush_threshold: 0,
        }
    }

    /// If the graph has grown past the configured threshold, materialize every
    /// pending lazy output (nodes with `reference_count > 0 && cached.is_none()`)
    /// in a single batched resolve. The user has already expressed intent to
    /// consume each of those outputs (via a live `LazyTensorData` handle), so
    /// this never forces work the user didn't ask for — it just compresses the
    /// schedule. Called from the end of `resolve()` / `resolve_batch()`.
    fn try_auto_flush(&mut self, removed: &mut Vec<ComputeGraphNode>) {
        if self.flush_threshold == 0 {
            return;
        }
        if self.nodes.nodes.node_count() < self.flush_threshold {
            return;
        }
        let pending: Vec<NodeIndex> = self
            .nodes
            .nodes
            .node_indices()
            .filter(|&k| {
                let Some(n) = self.nodes.nodes.node_weight(k) else {
                    return false;
                };
                n.reference_count > 0 && n.cached.is_none()
            })
            .collect();
        if pending.is_empty() {
            return;
        }
        let mut resolver = Resolver::new_batch(self, pending);
        let _ = resolver.run(self, removed);
    }

    /// Upgrade the weak device reference to a strong one.
    /// Panics if the device has been dropped (should not happen during normal operation).
    pub(crate) fn device(&self) -> Device {
        self.device
            .upgrade()
            .expect("Device was dropped while ComputeGraph is still in use")
    }

    fn create_node(&mut self, node: ComputeGraphNodeVariant) -> NodeIndex {
        let node = self.nodes.nodes.add_node(ComputeGraphNode {
            variant: node,
            reference_count: 1,
            live_descendant_count: 0,
            cached: None,
        });
        // New node has `reference_count = 1`, so it is alive. Adding edges
        // below propagates that liveness up to each dependency.
        self.add_dependency_edges(node);
        node
    }

    fn add_reference(&mut self, key: NodeIndex) {
        let transitioned_alive = {
            let node = self.nodes.nodes.node_weight_mut(key).unwrap();
            let prev_alive = node.alive_uncached();
            node.reference_count += 1;
            !prev_alive && node.alive_uncached()
        };
        if transitioned_alive {
            self.propagate_alive_change(key, true);
        }
    }

    fn add_dependency_edges(&mut self, key: NodeIndex) {
        let mut dependencies = Vec::new();
        self.visit_dependencies(key, &mut |dep| {
            dependencies.push(dep);
        });
        for dep in dependencies {
            self.add_dependency_edge(dep, key);
        }
    }

    /// Add an edge `from -> to` and maintain `live_descendant_count`.
    pub(crate) fn add_dependency_edge(&mut self, from: NodeIndex, to: NodeIndex) {
        self.nodes.nodes.add_edge(from, to, ());
        let to_alive = self
            .nodes
            .nodes
            .node_weight(to)
            .map(|n| n.alive_uncached())
            .unwrap_or(false);
        if !to_alive {
            return;
        }
        let from_transitioned = {
            let Some(from_node) = self.nodes.nodes.node_weight_mut(from) else {
                return;
            };
            let prev_alive = from_node.alive_uncached();
            from_node.live_descendant_count = from_node
                .live_descendant_count
                .checked_add(1)
                .expect("live_descendant_count overflow");
            !prev_alive && from_node.alive_uncached()
        };
        if from_transitioned {
            self.propagate_alive_change(from, true);
        }
    }

    /// Propagate an `alive_uncached`-state change to all ancestors.
    /// `now_alive = true` if `start` transitioned not-alive_uncached →
    /// alive_uncached, otherwise the reverse.
    fn propagate_alive_change(&mut self, start: NodeIndex, now_alive: bool) {
        let mut stack = vec![start];
        while let Some(child) = stack.pop() {
            let parents: Vec<NodeIndex> = self
                .nodes
                .nodes
                .neighbors_directed(child, petgraph::Direction::Incoming)
                .collect();
            for parent in parents {
                let parent_transitioned = {
                    let Some(parent_node) = self.nodes.nodes.node_weight_mut(parent) else {
                        continue;
                    };
                    let prev_parent_alive = parent_node.alive_uncached();
                    if now_alive {
                        parent_node.live_descendant_count = parent_node
                            .live_descendant_count
                            .checked_add(1)
                            .expect("live_descendant_count overflow");
                    } else {
                        parent_node.live_descendant_count =
                            parent_node.live_descendant_count.saturating_sub(1);
                    }
                    prev_parent_alive != parent_node.alive_uncached()
                };
                if parent_transitioned {
                    stack.push(parent);
                }
            }
        }
    }

    fn visit_dependencies(&self, key: NodeIndex, f: &mut dyn FnMut(NodeIndex)) {
        if let Some(node) = self.nodes.nodes.node_weight(key) {
            node.variant.visit_dependencies(f);
        }
    }

    fn ensure_tensor_cached(&mut self, key: NodeIndex) -> Option<()> {
        if self.get_cached_result(key).is_some() {
            return Some(());
        }

        let data = match self.nodes.nodes.node_weight(key)?.variant.clone() {
            ComputeGraphNodeVariant::Tensor(data) => data,
            _ => return None,
        };
        self.set_cached_result(key, data);
        Some(())
    }

    fn try_submit_direct_qmatmul(
        &mut self,
        operation: &QMatMulOperation,
    ) -> Option<(TensorData, usize)> {
        self.ensure_tensor_cached(operation.input)?;

        let device = self.device();
        let workgroup_shape = crate::mir::workgroup_shape::WorkgroupShape::new(1, 1, 1);
        let inputs = operation.inputs(self);
        let direct_kernel_plan = operation
            .build_direct_kernels(self, &workgroup_shape, &inputs)
            .ok()?;
        let MirValue::Tensor(output) = operation.output(self, &inputs) else {
            return None;
        };

        let mut command_encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("QMatMul Direct Encoder"),
                });
        let total_kernels = direct_kernel_plan.dispatch_count();
        for direct_kernel in direct_kernel_plan.into_kernels() {
            direct_kernel.run(device.kernel_cache(), &mut command_encoder);
        }
        if total_kernels > 0 {
            device.wgpu_queue().submit(Some(command_encoder.finish()));
            device.reset_initialized_buffers();
        }

        Some((output, total_kernels))
    }

    fn try_resolve_direct_qmatmul(&mut self, key: NodeIndex) -> Option<ResolverResult> {
        let operation = match self.nodes.nodes.node_weight(key)?.variant.clone() {
            ComputeGraphNodeVariant::QMatMul(operation) => operation,
            _ => return None,
        };
        let (output, total_kernels) = self.try_submit_direct_qmatmul(&operation)?;
        self.set_cached_result(key, output.clone());
        Some(ResolverResult {
            data: output,
            total_kernels,
        })
    }

    fn remove_reference(&mut self, key: NodeIndex, removed: &mut Vec<ComputeGraphNode>) {
        let transitioned_dead = {
            let node = self.nodes.nodes.node_weight_mut(key).unwrap();
            let prev_alive = node.alive_uncached();
            node.reference_count = node.reference_count.saturating_sub(1);
            prev_alive && !node.alive_uncached()
        };
        if transitioned_dead {
            self.propagate_alive_change(key, false);
        }
        self.check_life(key, removed);
    }

    fn check_life(&mut self, key: NodeIndex, removed: &mut Vec<ComputeGraphNode>) {
        // The node is needed iff it has external references OR some
        // uncached live descendant. `live_descendant_count` is maintained
        // eagerly, so this is O(1).
        match self
            .nodes
            .nodes
            .node_weight(key)
            .map(|n| n.should_keep_cached())
        {
            Some(true) | None => return,
            Some(false) => {}
        }

        let mut dependencies = Vec::new();
        self.visit_dependencies(key, &mut |dependency| {
            dependencies.push(dependency);
        });

        // Not needed — remove it. Per the invariant above, the node's
        // `alive_uncached` was already false (cached.is_some() or
        // ref==luc==0), so its contribution to each parent's
        // `live_descendant_count` is already 0; no further bookkeeping is
        // needed when the edges go away with the node.
        self.remove_key(key, removed);

        for dependency in dependencies {
            self.check_life(dependency, removed);
        }
    }

    fn remove_key(&mut self, key: NodeIndex, removed: &mut Vec<ComputeGraphNode>) {
        // Remove the node from the graph (this also removes all edges)
        if let Some(node) = self.nodes.nodes.remove_node(key) {
            removed.push(node);
        }
    }

    fn detach_cached(&mut self, keys: &[NodeIndex], removed: &mut Vec<ComputeGraphNode>) {
        for &key in keys {
            let Some(data) = self.get_cached_result(key).cloned() else {
                continue;
            };

            let mut dependencies = Vec::new();
            self.visit_dependencies(key, &mut |dependency| {
                dependencies.push(dependency);
            });

            if let Some(node) = self.nodes.nodes.node_weight_mut(key) {
                node.variant = ComputeGraphNodeVariant::Tensor(data.clone());
                node.cached = Some(data);
            }

            // Decoupling `key` from its dependencies: each parent loses `key`
            // as a child. Because we just set `key.cached = Some(...)`, `key`
            // is no longer `alive_uncached`, so its parents' counter already
            // doesn't include it — removing the edges needs no further
            // counter bookkeeping. The dependencies themselves may become
            // collectable, so we still check_life them below.
            let incoming: Vec<petgraph::graph::EdgeIndex> = self
                .nodes
                .nodes
                .edges_directed(key, petgraph::Direction::Incoming)
                .map(|edge| edge.id())
                .collect();
            for edge_id in incoming {
                self.nodes.nodes.remove_edge(edge_id);
            }

            for dependency in dependencies {
                self.check_life(dependency, removed);
            }
        }
    }

    pub(crate) fn get_result_or_qmatrix(&self, key: NodeIndex) -> Option<MaybeQData> {
        let node = self.nodes.nodes.node_weight(key)?;
        if let Some(cached) = &node.cached {
            return Some(cached.clone().into());
        }
        match &node.variant {
            ComputeGraphNodeVariant::Dequantize(op) => Some(op.matrix.clone().into()),
            ComputeGraphNodeVariant::Tensor(op) => Some(op.clone().into()),
            _ => None,
        }
    }

    pub(crate) fn get_result(&self, key: NodeIndex) -> Option<TensorData> {
        self.get_cached_result(key).cloned()
    }

    pub(crate) fn set_cached_result(&mut self, key: NodeIndex, data: TensorData) {
        // Setting `cached` flips `alive_uncached` false: a cached node no
        // longer needs to be recomputed, so its parents can free their own
        // cached buffers once no other uncached descendant remains. Propagate
        // the transition so ancestor counters reflect the new state.
        let transitioned_dead = {
            let node = self.nodes.nodes.node_weight_mut(key).unwrap();
            let prev_alive = node.alive_uncached();
            node.cached = Some(data);
            prev_alive && !node.alive_uncached()
        };
        if transitioned_dead {
            self.propagate_alive_change(key, false);
        }
    }

    pub(crate) fn get_cached_result(&self, key: NodeIndex) -> Option<&TensorData> {
        self.nodes
            .nodes
            .node_weight(key)
            .and_then(|n| n.cached.as_ref())
    }

    pub(crate) fn has_live_reference(&self, key: NodeIndex) -> bool {
        self.nodes
            .nodes
            .node_weight(key)
            .map(|n| n.reference_count > 0)
            .unwrap_or(false)
    }

    /// Returns true if this node's cached buffer would still benefit some
    /// future resolve: either the user holds a `LazyTensorData` for it
    /// directly, or some uncached live descendant will read its cached value
    /// instead of recomputing. Backed by the eagerly-maintained
    /// `live_descendant_count`, so this is O(1).
    pub(crate) fn has_live_lazy_descendant(&self, key: NodeIndex) -> bool {
        self.nodes
            .nodes
            .node_weight(key)
            .map(|n| n.should_keep_cached())
            .unwrap_or(false)
    }

    #[cfg(feature = "extra_assertions")]
    fn contains_key(&self, key: NodeIndex) -> bool {
        self.nodes.nodes.contains_node(key)
    }

    #[cfg(feature = "extra_assertions")]
    fn verify_integrity(&self) {
        // Check that all edges point to existing nodes
        for key in self.nodes.nodes.node_indices() {
            for neighbor in self.nodes.nodes.neighbors(key) {
                assert!(
                    self.nodes.nodes.contains_node(neighbor),
                    "edge points to non-existent node {neighbor:?}"
                );
            }
        }

        // Check that all dependencies of non-cached nodes exist
        for key in self.nodes.nodes.node_indices() {
            let is_cached = self
                .nodes
                .nodes
                .node_weight(key)
                .map(|n| n.cached.is_some())
                .unwrap_or(false);
            if is_cached {
                continue;
            }
            self.visit_dependencies(key, &mut |dependency| {
                assert!(
                    self.contains_key(dependency),
                    "dependency {dependency:?} of {key:?} does not exist"
                );
            });
        }

        // Check that `live_descendant_count` matches the number of outgoing
        // edges to `alive_uncached()` children.
        for key in self.nodes.nodes.node_indices() {
            let expected: u32 = self
                .nodes
                .nodes
                .neighbors_directed(key, petgraph::Direction::Outgoing)
                .filter(|child| {
                    self.nodes
                        .nodes
                        .node_weight(*child)
                        .map(|n| n.alive_uncached())
                        .unwrap_or(false)
                })
                .count()
                .try_into()
                .expect("live_descendant_count exceeds u32");
            let actual = self
                .nodes
                .nodes
                .node_weight(key)
                .map(|n| n.live_descendant_count)
                .unwrap_or(0);
            assert_eq!(
                actual, expected,
                "live_descendant_count mismatch at {key:?}: expected {expected}, got {actual}"
            );
        }
    }
}
