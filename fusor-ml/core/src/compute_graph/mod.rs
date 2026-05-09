use std::sync::Arc;

use parking_lot::RwLock;
pub use petgraph::graph::NodeIndex;
use petgraph::prelude::StableGraph;
use petgraph::visit::EdgeRef;
use resolve::Resolver;
use rustc_hash::FxHashSet;
use tabbycat::Graph;

mod layout_pass;
mod queue;
mod resolve;
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
    quantized::matmul::{QMatMulOperation, QMatMulPairedOperation},
    resize::ResizeOperation,
    slice_assign::SliceAssignOperation,
    tensor::TensorData,
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
        self.create_node(ComputeGraphNodeVariant::QMatMul(op))
    }

    pub(crate) fn create_q_mat_mul_paired(&self, op: QMatMulPairedOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::QMatMulPaired(op))
    }

    pub(crate) fn create_q_embedding(&self, op: QEmbeddingOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::QEmbedding(op))
    }

    pub(crate) fn create_reduce(&self, op: ReduceOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Reduce(op))
    }

    pub(crate) fn create_rms_norm(&self, op: RmsNormOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::RmsNorm(op))
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

    /// Resolve multiple targets in a single pass. All targets share one
    /// execution graph so intermediate nodes can be freed as soon as every
    /// consumer within the batch has been computed, keeping peak GPU memory
    /// much lower than resolving targets one-by-one.
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
            #[cfg(feature = "extra_assertions")]
            {
                inner.verify_integrity()
            }
            removed
        };
        drop(removed);
        total_kernels
    }

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
}

#[derive(Default)]
pub(crate) struct ComputeGraphNodes {
    pub(crate) nodes: StableGraph<ComputeGraphNode, ()>,
}

pub(crate) struct ComputeGraphNode {
    variant: ComputeGraphNodeVariant,
    reference_count: u32,
    cached: Option<TensorData>,
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
    QMatMul(QMatMulOperation),
    QMatMulPaired(QMatMulPairedOperation),
    Tensor(TensorData),
    Reduce(ReduceOperation),
    RmsNorm(RmsNormOperation),
    FlashAttention(FlashAttentionOperation),
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
            }
            ComputeGraphNodeVariant::QMatMulPaired(op) => {
                f(op.input);
            }
            ComputeGraphNodeVariant::QEmbedding(op) => {
                f(op.indexes);
            }
            ComputeGraphNodeVariant::Reduce(op) => f(op.value),
            ComputeGraphNodeVariant::RmsNorm(op) => {
                f(op.input);
                if let Some(residual) = op.residual {
                    f(residual);
                }
                f(op.weight);
                if let Some(bias) = op.bias {
                    f(bias);
                }
            }
            ComputeGraphNodeVariant::FlashAttention(op) => op.visit_dependencies(f),
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
}

impl ComputeGraphInner {
    fn new(device: &Device) -> Self {
        Self {
            device: device.downgrade(),
            nodes: ComputeGraphNodes::default(),
        }
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
            cached: None,
        });
        self.add_dependency_edges(node);
        node
    }

    fn add_reference(&mut self, key: NodeIndex) {
        let node = self.nodes.nodes.node_weight_mut(key).unwrap();

        node.reference_count += 1;
    }

    fn add_dependency_edges(&mut self, key: NodeIndex) {
        let mut dependencies = Vec::new();
        self.visit_dependencies(key, &mut |dep| {
            dependencies.push(dep);
        });
        for dep in dependencies {
            self.nodes.nodes.add_edge(dep, key, ());
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

    fn try_submit_direct_qmatmul(&mut self, operation: &QMatMulOperation) -> Option<TensorData> {
        self.ensure_tensor_cached(operation.input)?;

        let device = self.device();
        let workgroup_shape = crate::mir::workgroup_shape::WorkgroupShape::new(1, 1, 1);
        let inputs = operation.inputs(self);
        let direct_kernel = operation.build_direct_kernel(self, &workgroup_shape, &inputs)?;
        let MirValue::Tensor(output) = operation.output(self, &inputs) else {
            return None;
        };

        let mut command_encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("QMatMul Direct Encoder"),
                });
        direct_kernel.run(&device, &mut command_encoder);
        device.wgpu_queue().submit(Some(command_encoder.finish()));
        device.reset_initialized_buffers();

        Some(output)
    }

    fn try_resolve_direct_qmatmul(&mut self, key: NodeIndex) -> Option<ResolverResult> {
        let operation = match self.nodes.nodes.node_weight(key)?.variant.clone() {
            ComputeGraphNodeVariant::QMatMul(operation) => operation,
            _ => return None,
        };
        let output = self.try_submit_direct_qmatmul(&operation)?;
        self.set_cached_result(key, output.clone());
        Some(ResolverResult {
            data: output,
            total_kernels: 1,
        })
    }

    fn remove_reference(&mut self, key: NodeIndex, removed: &mut Vec<ComputeGraphNode>) {
        let node = self.nodes.nodes.node_weight_mut(key).unwrap();
        node.reference_count = node.reference_count.saturating_sub(1);
        self.check_life(key, removed);
    }

    fn check_life(&mut self, key: NodeIndex, removed: &mut Vec<ComputeGraphNode>) {
        // Check the reference count
        let ref_count = self.nodes.nodes.node_weight(key).map(|n| n.reference_count);
        match ref_count {
            Some(count) if count > 0 => {
                // The node still has references, so it is alive
                return;
            }
            None => {
                // The node is already dead
                return;
            }
            _ => {}
        }

        // Check if any of the nodes that depend on this key are alive
        let dependents: Vec<_> = self
            .nodes
            .nodes
            .neighbors_directed(key, petgraph::Direction::Outgoing)
            .collect();

        for dependant in dependents {
            // Keep dependencies alive while any downstream dependent is materially
            // live, even if intermediate nodes have already been computed.
            if self.has_materially_live_dependant(dependant, &mut FxHashSet::default()) {
                return;
            }
        }

        let mut dependencies = Vec::new();
        self.visit_dependencies(key, &mut |dependency| {
            dependencies.push(dependency);
        });

        // If no other nodes depend on this key and it has zero references, it is dead
        // remove it from the graph
        self.remove_key(key, removed);

        // Then check if any nodes it depends on are alive
        for dependency in dependencies {
            self.check_life(dependency, removed);
        }
    }

    fn has_materially_live_dependant(
        &self,
        key: NodeIndex,
        visited: &mut FxHashSet<NodeIndex>,
    ) -> bool {
        if !visited.insert(key) {
            return false;
        }

        let Some(node) = self.nodes.nodes.node_weight(key) else {
            return false;
        };

        if node.reference_count > 0 {
            return true;
        }

        self.nodes
            .nodes
            .neighbors_directed(key, petgraph::Direction::Outgoing)
            .any(|dependant| self.has_materially_live_dependant(dependant, visited))
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

            let incoming_edges = self
                .nodes
                .nodes
                .edges_directed(key, petgraph::Direction::Incoming)
                .map(|edge| edge.id())
                .collect::<Vec<_>>();
            for edge in incoming_edges {
                self.nodes.nodes.remove_edge(edge);
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
        let node = self.nodes.nodes.node_weight_mut(key).unwrap();
        node.cached = Some(data);
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
            .map(|node| node.reference_count > 0)
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
    }
}
