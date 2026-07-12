use std::sync::Arc;

use parking_lot::RwLock;
pub use petgraph::graph::NodeIndex;
use petgraph::prelude::StableGraph;
use resolve::Resolver;
use resolve::flush_replay::{self, ReplayAction};
use rustc_hash::FxHashMap;
#[cfg(feature = "graphvis")]
use tabbycat::Graph;

pub(crate) use resolve::flush_replay::FlushPlanCache;

mod layout_pass;
mod queue;
pub(crate) mod resolve;
#[cfg(test)]
mod tests;
#[cfg(feature = "graphvis")]
mod visualize;

use crate::{
    DataTypeEnum, Device, QMatrix, ReduceOperation, compute_graph::resolve::ResolverResult,
    dequantize::DequantizeOperation, nary_wise::ElementwiseOperation,
    slice_assign::SliceAssignOperation, tensor::TensorData, view::ViewOperation,
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

    pub(crate) fn create_nary(&self, op: ElementwiseOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Elementwise(op))
    }

    pub(crate) fn create_reduce(&self, op: ReduceOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Reduce(op))
    }

    pub(crate) fn create_view(&self, op: ViewOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::View(op))
    }

    /// Clone the view at `key` if that node is an unresolved view. Used to
    /// collapse view chains at construction time. Cached views are excluded:
    /// a resolved view no longer keeps its base alive, so the base node may
    /// already be culled from the graph — and a new view over the cached
    /// buffer is a zero-cost map anyway, while composing past it would force
    /// a recompute from the (possibly released) base.
    pub(crate) fn get_view(&self, key: NodeIndex) -> Option<ViewOperation> {
        let inner = self.inner.read();
        let node = inner.nodes.nodes.node_weight(key)?;
        if node.cached.is_some() {
            return None;
        }
        match &node.variant {
            ComputeGraphNodeVariant::View(op) => Some(op.clone()),
            _ => None,
        }
    }

    pub(crate) fn create_slice_assign(&self, op: SliceAssignOperation) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Assign(op))
    }

    pub(crate) fn create_tensor(&self, op: TensorData) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::Tensor(op))
    }

    pub(crate) fn dequantize(&self, matrix: QMatrix, ty: DataTypeEnum) -> NodeIndex {
        self.create_node(ComputeGraphNodeVariant::QMatrix(DequantizeOperation::new(
            matrix, ty,
        )))
    }

    /// Resolve every pending lazy output now, submitting the work to the
    /// GPU without waiting for it or downloading anything. Keeps the pending
    /// graph small in iteration-heavy workloads like training loops.
    pub(crate) fn flush(&self) {
        let mut removed = Vec::new();
        {
            let mut inner = self.inner.write();
            inner.flush_all_pending(&mut removed);
            inner.prune_deferred_dead(&mut removed);
            #[cfg(feature = "extra_assertions")]
            {
                inner.verify_integrity()
            }
        }
        drop(removed);
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

        let (data, removed) = {
            let mut inner = self.inner.write();
            let mut removed = Vec::new();
            let (data, ()) = inner.resolve_target_with_replay(key, &mut removed, |_, _| ());
            inner.try_auto_flush(&mut removed);
            inner.prune_deferred_dead(&mut removed);
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

    pub(crate) fn resolve_with_tail<T>(
        &self,
        key: NodeIndex,
        tail: impl FnOnce(&TensorData, &mut wgpu::CommandEncoder) -> T,
    ) -> (ResolverResult, T) {
        if let Some(data) = {
            let inner = self.inner.read();
            inner.get_cached_result(key).cloned()
        } {
            let device = data.device().clone();
            let mut command_encoder =
                device
                    .wgpu_device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Resolver Tail Encoder"),
                    });
            let tail_result = tail(&data, &mut command_encoder);
            device.wgpu_queue().submit(Some(command_encoder.finish()));
            device.reset_initialized_buffers();
            return (
                ResolverResult {
                    data,
                    total_kernels: 0,
                },
                tail_result,
            );
        }

        let (data, removed, tail_result) = {
            let mut inner = self.inner.write();
            let mut removed = Vec::new();
            let (data, tail_result) = inner.resolve_target_with_replay(key, &mut removed, tail);
            inner.try_auto_flush(&mut removed);
            inner.prune_deferred_dead(&mut removed);
            #[cfg(feature = "extra_assertions")]
            {
                inner.verify_integrity()
            }
            (data, removed, tail_result)
        };
        drop(removed);

        (data, tail_result)
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
            inner.prune_deferred_dead(&mut removed);
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
    // reuse shared ancestors instead of recomputing them. A descendant that
    // has already been resolved (cached) no longer contributes, so deep chains
    // where only the final tensor is held still free intermediates eagerly
    // during the resolve.
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

/// The graph vocabulary. Exactly three core operations — elementwise
/// visitation, reduction, and zero-dispatch views — over tensor and
/// quantized-matrix leaves, plus the in-place region write (pure data
/// movement, no compute dispatch). Everything else (matmul, attention,
/// normalization, embedding...) is a composition of these that the resolver
/// recognizes into fused execution regions.
#[derive(Clone, Debug)]
pub(crate) enum ComputeGraphNodeVariant {
    Tensor(TensorData),
    QMatrix(DequantizeOperation),
    Elementwise(ElementwiseOperation),
    Reduce(ReduceOperation),
    View(ViewOperation),
    Assign(SliceAssignOperation),
}

impl ComputeGraphNodeVariant {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        match &self {
            ComputeGraphNodeVariant::Elementwise(op) => {
                for input in &op.inputs {
                    f(*input);
                }
            }
            ComputeGraphNodeVariant::Reduce(op) => {
                for input in &op.inputs {
                    f(*input);
                }
            }
            ComputeGraphNodeVariant::View(op) => f(op.input),
            ComputeGraphNodeVariant::Assign(op) => {
                f(op.input);
                f(op.value);
            }
            ComputeGraphNodeVariant::QMatrix(_) => {}
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
    // Incremental pending-sink set: every node with `reference_count > 0 &&
    // cached.is_none()`, tagged with a monotonically increasing insertion
    // sequence number. Replaces the O(all-nodes) scan in `flush_all_pending`
    // and makes sink enumeration deterministic in tape-construction order
    // (StableGraph recycles indices, so `node_indices()` order is not stable
    // across isomorphic steps) — which is what lets flush fingerprints of
    // isomorphic steps collide.
    pending_sinks: FxHashMap<NodeIndex, u64>,
    pending_seq: u64,
    // Nodes whose `should_keep_cached()` flipped false during a resolve
    // (their last alive descendant was cached). They cannot be removed at
    // that point — the in-flight execution still reads their buffers by
    // index — so removal is deferred to `prune_deferred_dead` at the end of
    // the public operation. Without this, every cached-over node lingers as
    // a permanent husk: `check_life` only runs on reference drops, and a
    // dead node's references are already gone.
    deferred_dead: Vec<NodeIndex>,
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
            pending_sinks: FxHashMap::default(),
            pending_seq: 0,
            deferred_dead: Vec::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn new_for_test(device: crate::WeakDevice) -> Self {
        Self {
            device,
            nodes: ComputeGraphNodes::default(),
            flush_threshold: 0,
            pending_sinks: FxHashMap::default(),
            pending_seq: 0,
            deferred_dead: Vec::new(),
        }
    }

    /// If the graph has grown past the configured threshold, materialize every
    /// pending lazy output (nodes with `reference_count > 0 && cached.is_none()`)
    /// in a single batched resolve. The user has already expressed intent to
    /// consume each of those outputs (via a live `LazyTensorData` handle), so
    /// this never forces work the user didn't ask for — it just compresses the
    /// schedule. Called from the end of `resolve()`.
    fn try_auto_flush(&mut self, removed: &mut Vec<ComputeGraphNode>) {
        if self.flush_threshold == 0 {
            return;
        }
        if self.nodes.nodes.node_count() < self.flush_threshold {
            return;
        }
        self.flush_all_pending(removed);
    }

    /// Materialize every pending lazy output in a single batched resolve.
    ///
    /// Consecutive structurally identical pending subgraphs go through the
    /// flush-plan replay cache: the first occurrence records the full resolve
    /// and later occurrences replay it, skipping execution-graph building,
    /// optimization, lowering, and kernel building.
    fn flush_all_pending(&mut self, removed: &mut Vec<ComputeGraphNode>) {
        // Enumerate pending sinks in insertion (tape-construction) order so
        // fingerprints of isomorphic steps are deterministic for every graph.
        let mut pending: Vec<(u64, NodeIndex)> = self
            .pending_sinks
            .iter()
            .filter(|&(&key, _)| {
                self.nodes
                    .nodes
                    .node_weight(key)
                    .map(|n| n.reference_count > 0 && n.cached.is_none())
                    .unwrap_or(false)
            })
            .map(|(&key, &seq)| (seq, key))
            .collect();
        pending.sort_unstable();
        let pending: Vec<NodeIndex> = pending.into_iter().map(|(_, key)| key).collect();
        if pending.is_empty() {
            return;
        }

        match flush_replay::prepare_replay(self, &pending) {
            ReplayAction::Replay { plan, fingerprint } => {
                let _ =
                    flush_replay::execute_replay_with_tail(self, &plan, &fingerprint, |_, _| ());
                return;
            }
            ReplayAction::Record { key, fingerprint } => {
                let mut resolver = Resolver::new_batch_with_recording(self, pending, fingerprint);
                let _ = resolver.run(self, removed);
                if let Some(plan) = resolver.take_recorded_plan() {
                    self.device().flush_plan_cache().insert(key, Arc::new(plan));
                }
                return;
            }
            ReplayAction::Resolve => {}
        }

        let mut resolver = Resolver::new_batch(self, pending);
        let _ = resolver.run(self, removed);
    }

    /// Resolve one target, recording or replaying the same bufferless plan
    /// format used by batched flushes. This is the hot materialization path
    /// for repeated isomorphic inference graphs such as `as_slice()` during
    /// autoregressive generation.
    fn resolve_target_with_replay<T>(
        &mut self,
        target: NodeIndex,
        removed: &mut Vec<ComputeGraphNode>,
        tail: impl FnOnce(&TensorData, &mut wgpu::CommandEncoder) -> T,
    ) -> (ResolverResult, T) {
        match flush_replay::prepare_replay(self, &[target]) {
            ReplayAction::Replay { plan, fingerprint } => {
                flush_replay::execute_replay_with_tail(self, &plan, &fingerprint, tail)
            }
            ReplayAction::Record { key, fingerprint } => {
                let mut resolver =
                    Resolver::new_batch_with_recording(self, vec![target], fingerprint);
                let result = resolver.run_with_tail(self, removed, tail);
                if let Some(plan) = resolver.take_recorded_plan() {
                    self.device().flush_plan_cache().insert(key, Arc::new(plan));
                }
                result
            }
            ReplayAction::Resolve => {
                let mut resolver = Resolver::new(self, target);
                resolver.run_with_tail(self, removed, tail)
            }
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
            live_descendant_count: 0,
            cached: None,
        });
        // New node has `reference_count = 1` and no cached result: pending.
        self.mark_pending(node);
        // New node has `reference_count = 1`, so it is alive. Adding edges
        // below propagates that liveness up to each dependency.
        self.add_dependency_edges(node);
        node
    }

    /// Track `key` in the pending-sink set (referenced and uncached).
    fn mark_pending(&mut self, key: NodeIndex) {
        let seq = self.pending_seq;
        self.pending_seq += 1;
        self.pending_sinks.entry(key).or_insert(seq);
    }

    fn add_reference(&mut self, key: NodeIndex) {
        let (transitioned_alive, now_pending) = {
            let node = self.nodes.nodes.node_weight_mut(key).unwrap();
            let prev_alive = node.alive_uncached();
            node.reference_count += 1;
            (!prev_alive && node.alive_uncached(), node.cached.is_none())
        };
        if now_pending {
            self.mark_pending(key);
        }
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
                let (parent_transitioned, parent_now_dead) = {
                    let Some(parent_node) = self.nodes.nodes.node_weight_mut(parent) else {
                        continue;
                    };
                    let prev_parent_alive = parent_node.alive_uncached();
                    let prev_parent_kept = parent_node.should_keep_cached();
                    if now_alive {
                        parent_node.live_descendant_count = parent_node
                            .live_descendant_count
                            .checked_add(1)
                            .expect("live_descendant_count overflow");
                    } else {
                        parent_node.live_descendant_count =
                            parent_node.live_descendant_count.saturating_sub(1);
                    }
                    (
                        prev_parent_alive != parent_node.alive_uncached(),
                        prev_parent_kept && !parent_node.should_keep_cached(),
                    )
                };
                // A parent whose last live descendant just went away is now
                // unreachable by any future resolve and must eventually be
                // removed. `check_life` cannot run here — during a resolve
                // the execution graph still reads this node — so record it
                // for `prune_deferred_dead`. Note this also catches CACHED
                // parents, which never flip `alive_uncached` and so are
                // invisible to the transition propagation below.
                if parent_now_dead {
                    self.deferred_dead.push(parent);
                }
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

    fn remove_reference(&mut self, key: NodeIndex, removed: &mut Vec<ComputeGraphNode>) {
        let (transitioned_dead, still_referenced) = {
            let node = self.nodes.nodes.node_weight_mut(key).unwrap();
            let prev_alive = node.alive_uncached();
            node.reference_count = node.reference_count.saturating_sub(1);
            (
                prev_alive && !node.alive_uncached(),
                node.reference_count > 0,
            )
        };
        if !still_referenced {
            self.pending_sinks.remove(&key);
        }
        if transitioned_dead {
            self.propagate_alive_change(key, false);
        }
        self.check_life(key, removed);
    }

    fn check_life(&mut self, key: NodeIndex, removed: &mut Vec<ComputeGraphNode>) {
        // Iterative worklist, NOT recursion: teardown cascades one frame per
        // node, and dropping the last handle to a long-lived chain (e.g. an
        // optimizer moment at the end of a training run) must not overflow
        // the stack.
        let mut worklist = vec![key];
        while let Some(key) = worklist.pop() {
            // The node is needed iff it has external references OR some
            // uncached live descendant. `live_descendant_count` is maintained
            // eagerly, so this is O(1).
            match self
                .nodes
                .nodes
                .node_weight(key)
                .map(|n| n.should_keep_cached())
            {
                Some(true) | None => continue,
                Some(false) => {}
            }

            // Not needed — remove it. Per the invariant above, the node's
            // `alive_uncached` was already false (cached.is_some() or
            // ref==luc==0), so its contribution to each parent's
            // `live_descendant_count` is already 0; no further bookkeeping is
            // needed when the edges go away with the node.
            self.visit_dependencies(key, &mut |dependency| {
                worklist.push(dependency);
            });
            self.remove_key(key, removed);
        }
    }

    /// Remove nodes whose liveness died inside a resolve (recorded in
    /// `deferred_dead` by `propagate_alive_change`). Runs at the end of the
    /// public graph operations, once the execution that was still reading
    /// those nodes' buffers has been submitted. Entries removed by an earlier
    /// cascade are skipped by `check_life`'s existence check.
    fn prune_deferred_dead(&mut self, removed: &mut Vec<ComputeGraphNode>) {
        while let Some(key) = self.deferred_dead.pop() {
            self.check_life(key, removed);
        }
    }

    fn remove_key(&mut self, key: NodeIndex, removed: &mut Vec<ComputeGraphNode>) {
        // Remove the node from the graph (this also removes all edges)
        if let Some(node) = self.nodes.nodes.remove_node(key) {
            // A removable node has `reference_count == 0`, so it should
            // already be out of the pending set; defensive removal keeps the
            // set exact even if that invariant ever slips.
            self.pending_sinks.remove(&key);
            removed.push(node);
        }
    }

    pub(crate) fn get_result_or_qmatrix(&self, key: NodeIndex) -> Option<MaybeQData> {
        let node = self.nodes.nodes.node_weight(key)?;
        if let Some(cached) = &node.cached {
            return Some(cached.clone().into());
        }
        match &node.variant {
            ComputeGraphNodeVariant::QMatrix(op) => Some(op.matrix.clone().into()),
            ComputeGraphNodeVariant::Tensor(op) => Some(op.clone().into()),
            _ => None,
        }
    }

    pub(crate) fn get_result(&self, key: NodeIndex) -> Option<TensorData> {
        self.get_cached_result(key).cloned()
    }

    pub(crate) fn set_cached_result(&mut self, key: NodeIndex, data: TensorData) {
        // A cached node is no longer a pending sink.
        self.pending_sinks.remove(&key);
        // Setting `cached` flips `alive_uncached` false: a cached node no
        // longer needs to be recomputed, so its parents can free their own
        // cached buffers once no other uncached descendant remains. Propagate
        // the transition so ancestor counters reflect the new state.
        let (transitioned_dead, now_dead) = {
            let node = self.nodes.nodes.node_weight_mut(key).unwrap();
            let prev_alive = node.alive_uncached();
            node.cached = Some(data);
            (
                prev_alive && !node.alive_uncached(),
                !node.should_keep_cached(),
            )
        };
        if now_dead {
            self.deferred_dead.push(key);
        }
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
        // Dead nodes (no references, no live uncached descendant) are pruned
        // eagerly — by the `check_life` cascade on reference drops and by
        // `prune_deferred_dead` after resolves — so none may survive past the
        // end of a public operation. A node lingering here is a husk: it
        // would accumulate once per training step and make final teardown
        // O(steps).
        assert!(
            self.deferred_dead.is_empty(),
            "deferred dead set not drained"
        );
        for key in self.nodes.nodes.node_indices() {
            let node = self.nodes.nodes.node_weight(key).unwrap();
            assert!(
                node.should_keep_cached(),
                "dead node {key:?} survived pruning"
            );
        }

        // Check that all edges point to existing nodes
        for key in self.nodes.nodes.node_indices() {
            for neighbor in self.nodes.nodes.neighbors(key) {
                assert!(
                    self.nodes.nodes.contains_node(neighbor),
                    "edge points to non-existent node {neighbor:?}"
                );
            }
        }

        // Check that all dependencies of non-cached nodes that could still
        // resolve exist. A dead uncached node (no references and no alive
        // descendants — e.g. an intermediate whose buffer was released after
        // its handle dropped) is unreachable by any future resolve: resolves
        // start from a live handle, and everything alive transitively keeps
        // its dependencies' `live_descendant_count` positive. Its
        // dependencies may therefore be legitimately removed from under it.
        for key in self.nodes.nodes.node_indices() {
            let resolvable = self
                .nodes
                .nodes
                .node_weight(key)
                .map(|n| n.cached.is_none() && n.should_keep_cached())
                .unwrap_or(false);
            if !resolvable {
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

        // Check that the incremental pending-sink set exactly matches the
        // predicate it caches (`reference_count > 0 && cached.is_none()`).
        for key in self.nodes.nodes.node_indices() {
            let pending = self
                .nodes
                .nodes
                .node_weight(key)
                .map(|n| n.reference_count > 0 && n.cached.is_none())
                .unwrap_or(false);
            assert_eq!(
                self.pending_sinks.contains_key(&key),
                pending,
                "pending_sinks mismatch at {key:?}: expected pending={pending}"
            );
        }
        for key in self.pending_sinks.keys() {
            assert!(
                self.nodes.nodes.contains_node(*key),
                "pending_sinks contains removed node {key:?}"
            );
        }
    }
}
