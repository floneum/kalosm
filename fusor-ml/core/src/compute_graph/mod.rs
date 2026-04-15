use std::sync::Arc;

use parking_lot::RwLock;
pub(crate) use petgraph::graph::NodeIndex;
use petgraph::prelude::StableGraph;
use resolve::Resolver;
use tabbycat::Graph;
use wgpu::CommandEncoderDescriptor;

mod layout_pass;
mod queue;
mod resolve;
mod visualize;

use crate::{
    DataTypeEnum, Device, ElementWiseOperation, MatMulOperation, PairWiseOperation, QMatrix,
    ReduceOperation, composite::where_cond::WhereCondOperation,
    compute_graph::resolve::ResolverResult, dequantize::DequantizeOperation,
    index_select::IndexSelectOperation, map_layout::MapLayoutOperation, mir::operation::Operation,
    nary_wise::NaryOperation, quantized::matmul::QMatMulOperation, resize::ResizeOperation,
    slice_assign::SliceAssignOperation, tensor::TensorData, visit_tiled::MaybeQData,
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

    pub(crate) fn create_element_wise(&self, op: ElementWiseOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::ElementWise(op))
    }

    pub(crate) fn create_pair_wise(&self, op: PairWiseOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::PairWise(op))
    }

    pub(crate) fn create_mat_mul(&self, op: MatMulOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::MatMul(op))
    }

    pub(crate) fn create_q_mat_mul(&self, op: QMatMulOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::QMatMul(op))
    }

    pub(crate) fn create_reduce(&self, op: ReduceOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Reduce(op))
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

    pub(crate) fn create_index_select(&self, op: IndexSelectOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::IndexSelect(op))
    }

    pub(crate) fn create_tensor(&self, op: TensorData) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Tensor(op))
    }

    pub(crate) fn dequantize(&self, matrix: QMatrix, ty: DataTypeEnum) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Dequantize(
            DequantizeOperation::new(matrix, ty),
        ))
    }

    pub(crate) fn create_custom(&self, op: Arc<dyn Operation + Send + Sync>) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Custom(op))
    }

    pub(crate) fn create_where_cond(&self, op: WhereCondOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::WhereCond(op))
    }

    pub(crate) fn resolve(&self, key: NodeIndex, device: &Device) -> ResolverResult {
        let mut encoder = device
            .wgpu_device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("ComputeGraph Encoder"),
            });
        let data = self.with_mut(|inner| {
            let mut resolver = Resolver::new(inner, key, &mut encoder);
            resolver.run(inner)
        });
        device.wgpu_queue().submit(Some(encoder.finish()));
        // Reset the written flag on all buffers
        device.reset_initialized_buffers();

        // Flush the cache to a file
        if let (Some(pipeline_cache), Some(cache_file)) =
            (device.wgpu_cache(), device.wgpu_cache_file())
        {
            let data = pipeline_cache.get_data();
            if let Some(data) = data {
                let temp_file = cache_file.with_extension("temp");
                std::fs::write(&temp_file, &data).unwrap();
                std::fs::rename(&temp_file, cache_file).unwrap();
            }
        }

        data
    }

    pub(crate) fn graphvis(&self, root: NodeIndex) -> Graph {
        self.with_mut(|inner| inner.graphvis(root))
    }

    pub(crate) fn add_reference(&self, key: NodeIndex) {
        self.with_mut(|inner| inner.add_reference(key));
    }

    pub(crate) fn remove_reference(&self, key: NodeIndex) {
        self.with_mut(|inner| inner.remove_reference(key));
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
    ElementWise(ElementWiseOperation),
    PairWise(PairWiseOperation),
    Nary(NaryOperation),
    SliceAssign(SliceAssignOperation),
    Resize(ResizeOperation),
    MapLayout(MapLayoutOperation),
    Dequantize(DequantizeOperation),
    MatMul(MatMulOperation),
    QMatMul(QMatMulOperation),
    Tensor(TensorData),
    Reduce(ReduceOperation),
    IndexSelect(IndexSelectOperation),
    WhereCond(WhereCondOperation),
    Custom(Arc<dyn Operation + Send + Sync>),
}

impl ComputeGraphNodeVariant {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        match &self {
            ComputeGraphNodeVariant::ElementWise(op) => f(op.value),
            ComputeGraphNodeVariant::PairWise(op) => {
                f(op.first);
                f(op.second);
            }
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
            ComputeGraphNodeVariant::Reduce(op) => f(op.value),
            ComputeGraphNodeVariant::MapLayout(op) => f(op.input),
            ComputeGraphNodeVariant::Resize(op) => f(op.input),
            ComputeGraphNodeVariant::SliceAssign(op) => {
                f(op.input);
                f(op.value);
            }
            ComputeGraphNodeVariant::IndexSelect(op) => {
                f(op.input);
                f(op.indexes);
            }
            ComputeGraphNodeVariant::WhereCond(op) => {
                f(op.condition);
                f(op.on_true);
                f(op.on_false);
            }
            ComputeGraphNodeVariant::Dequantize(_) => {}
            ComputeGraphNodeVariant::Tensor(_) => {}
            ComputeGraphNodeVariant::Custom(op) => {
                op.visit_dependencies(f);
            }
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

    fn remove_reference(&mut self, key: NodeIndex) {
        let node = self.nodes.nodes.node_weight_mut(key).unwrap();
        node.reference_count = node.reference_count.saturating_sub(1);
        self.check_life(key);
    }

    fn check_life(&mut self, key: NodeIndex) {
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
            // If the dependant still exists and it hasn't been computed yet
            // keep this node alive
            if let Some(dep_node) = self.nodes.nodes.node_weight(dependant) {
                let computed = dep_node.cached.is_some();
                if !computed {
                    return;
                }
            }
        }

        let mut dependencies = Vec::new();
        self.visit_dependencies(key, &mut |dependency| {
            dependencies.push(dependency);
        });

        // If no other nodes depend on this key and it has zero references, it is dead
        // remove it from the graph
        self.remove_key(key);

        // Then check if any nodes it depends on are alive
        for dependency in dependencies {
            self.check_life(dependency);
        }
    }

    fn remove_key(&mut self, key: NodeIndex) {
        // Remove the node from the graph (this also removes all edges)
        self.nodes.nodes.remove_node(key);
    }

    pub(crate) fn get_result_or_qmatrix(&self, key: NodeIndex) -> Option<MaybeQData> {
        let node = self.nodes.nodes.node_weight(key)?;
        match &node.variant {
            ComputeGraphNodeVariant::Dequantize(op) => Some(op.matrix.clone().into()),
            ComputeGraphNodeVariant::Tensor(op) => Some(op.clone().into()),
            _ => node.cached.as_ref().map(|t| t.clone().into()),
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
