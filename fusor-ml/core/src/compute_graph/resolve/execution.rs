use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) enum OptimizePolicy {
    Standard,
    LargeGraph { optimize_decode_graphs: bool },
}

// This is a separate safety/performance gate from the configurable threshold
// that selects the optimizer profile. Preserve the historical behavior unless
// the dedicated QMatMul fusion override is set.
const STANDARD_QMATMUL_FUSION_NODE_LIMIT: usize = 512;

impl OptimizePolicy {
    pub(super) fn select(
        node_count: usize,
        node_limit: usize,
        optimize_decode_graphs: bool,
    ) -> Self {
        if node_limit != 0 && node_count > node_limit {
            Self::LargeGraph {
                optimize_decode_graphs,
            }
        } else {
            Self::Standard
        }
    }

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Standard => "standard",
            Self::LargeGraph { .. } => "large_graph",
        }
    }

    fn is_large_graph(self) -> bool {
        matches!(self, Self::LargeGraph { .. })
    }

    fn runs_stage2_fusion(self, is_single_token_decode: bool) -> bool {
        match self {
            Self::Standard => true,
            Self::LargeGraph {
                optimize_decode_graphs,
            } => optimize_decode_graphs || !is_single_token_decode,
        }
    }
}

impl Resolver {
    pub(super) fn release_dead_intermediates(
        graph: &mut ComputeGraphInner,
        produced_ops: &[&QueuedOperation],
        remaining_consumers: &mut FxHashMap<NodeIndex, usize>,
        targets: &FxHashSet<NodeIndex>,
        ledger: &mut super::alloc_reuse::BufferLedger,
    ) {
        for op in produced_ops {
            op.visit_dependencies(&mut |dep| {
                if let Some(count) = remaining_consumers.get_mut(&dep) {
                    *count = count.saturating_sub(1);
                    if *count == 0
                        && !targets.contains(&dep)
                        && !graph.has_live_lazy_descendant(dep)
                    {
                        if let Some(cached) = graph.get_cached_result(dep) {
                            ledger.note_released(dep, cached);
                        }
                        // All consumers within this execution have been
                        // processed and no user-held lazy tensor still
                        // transitively depends on `dep` — free the cached
                        // buffer. The descendant check must include
                        // `live_descendant_count`, not just direct
                        // references: clearing `cached` on a node that still
                        // has an alive-uncached descendant flips it back to
                        // alive-uncached without propagating the transition,
                        // undercounting every ancestor's descendant counter.
                        if let Some(node) = graph.nodes.nodes.node_weight_mut(dep) {
                            node.cached = None;
                        }
                    }
                }
            });
        }
    }

    /// Like `release_dead_intermediates` but uses the compute graph's
    /// `visit_dependencies` instead of an Operation's. Used for map-layout
    /// and resize nodes that are resolved immediately without being lowered
    /// to an Operation.
    pub(super) fn release_dead_intermediates_from_graph(
        graph: &mut ComputeGraphInner,
        produced_nodes: &[NodeIndex],
        remaining_consumers: &mut FxHashMap<NodeIndex, usize>,
        targets: &FxHashSet<NodeIndex>,
        ledger: &mut super::alloc_reuse::BufferLedger,
    ) {
        for &produced in produced_nodes {
            let mut deps = Vec::new();
            graph.visit_dependencies(produced, &mut |dep| {
                deps.push(dep);
            });
            for dep in deps {
                if let Some(count) = remaining_consumers.get_mut(&dep) {
                    *count = count.saturating_sub(1);
                    if *count == 0
                        && !targets.contains(&dep)
                        && !graph.has_live_lazy_descendant(dep)
                    {
                        if let Some(cached) = graph.get_cached_result(dep) {
                            ledger.note_released(dep, cached);
                        }
                        if let Some(node) = graph.nodes.nodes.node_weight_mut(dep) {
                            node.cached = None;
                        }
                    }
                }
            }
        }
    }

    pub(super) fn try_prepare_in_place_slice_assign_copy(
        graph: &ComputeGraphInner,
        operation: &crate::slice_assign::SliceAssignOperation,
    ) -> Option<(TensorData, Vec<CopyBufferRecord>)> {
        let input = graph.get_cached_result(operation.input)?;
        let value = graph.get_cached_result(operation.value)?;
        if input.datatype() != value.datatype() || operation.slices.len() != input.layout().rank() {
            return None;
        }

        let output = input.slice(&operation.slices);
        if output.layout().shape() != value.layout().shape()
            || !output.layout().inner_dim_contiguous()
            || !value.layout().inner_dim_contiguous()
        {
            return None;
        }

        let element_size = input.datatype().element_size();
        let shape = value.layout().shape();
        let row_elems = *shape.last()?;
        let copy_size = row_elems.checked_mul(element_size)? as u64;
        if copy_size == 0 || !copy_size.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            return None;
        }

        let outer_rank = shape.len().saturating_sub(1);
        let outer_count = shape[..outer_rank]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
        let source_strides = value.layout().strides();
        let destination_strides = output.layout().strides();
        let source_base = value.layout().offset();
        let destination_base = output.layout().offset();
        let mut copies = Vec::with_capacity(outer_count);

        for linear in 0..outer_count {
            let mut remaining = linear;
            let mut source_element = source_base;
            let mut destination_element = destination_base;
            for dim in (0..outer_rank).rev() {
                let dim_len = shape[dim];
                let index = if dim_len == 0 { 0 } else { remaining % dim_len };
                remaining = remaining.checked_div(dim_len).unwrap_or(0);
                source_element = source_element.checked_add(index * source_strides[dim])?;
                destination_element =
                    destination_element.checked_add(index * destination_strides[dim])?;
            }

            let source_offset = source_element.checked_mul(element_size)? as u64;
            let destination_offset = destination_element.checked_mul(element_size)? as u64;
            if !source_offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
                || !destination_offset.is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
            {
                return None;
            }
            copies.push(CopyBufferRecord {
                source: value.buffer().clone(),
                destination: input.buffer().clone(),
                source_offset,
                destination_offset,
                size: copy_size,
            });
        }

        Some((input.clone(), copies))
    }

    pub(super) fn build_execution_graph(
        &mut self,
        graph: &ComputeGraphInner,
        node: NodeIndex,
    ) -> Option<ExecutionNodeIndex> {
        if self.resolved_set.contains(&node) {
            return None;
        }
        if let Some(&idx) = self.node_mapping.get(&node) {
            return Some(idx);
        }

        let node_data = graph
            .nodes
            .nodes
            .node_weight(node)
            .expect("Node not found in graph");
        let variant = node_data.variant.clone();

        // Add to execution graph
        let exec_idx = self.execution_graph.add_node(ExecutionNode {
            inner_idx: node,
            variant: variant.clone().into(),
        });
        self.node_mapping.insert(node, exec_idx);

        // Find dependencies
        let mut dependencies = Vec::new();
        variant.visit_dependencies(&mut |dependency| {
            dependencies.push(dependency);
        });

        for dependency in dependencies {
            if let Some(dep_exec_idx) = self.build_execution_graph(graph, dependency) {
                self.execution_graph.add_edge(dep_exec_idx, exec_idx, ());
            }
        }

        Some(exec_idx)
    }

    pub(super) fn lower_node(
        &self,
        exec_idx: ExecutionNodeIndex,
        node: &ExecutionNode,
    ) -> Option<QueuedOperation> {
        match &node.variant {
            ExecutionVariant::Elementwise(op) => {
                Some(QueuedOperation::Operation(Arc::new(op.clone())))
            }
            ExecutionVariant::MatMul(op) => Some(QueuedOperation::Operation(Arc::new(op.clone()))),
            ExecutionVariant::Reduce(op) => Some(QueuedOperation::Operation(Arc::new(op.clone()))),
            ExecutionVariant::GraphOp(op) => Some(QueuedOperation::Operation(op.clone())),
            ExecutionVariant::View(op) => Some(QueuedOperation::Operation(Arc::new(op.clone()))),
            ExecutionVariant::Assign(op) => Some(QueuedOperation::Operation(Arc::new(op.clone()))),
            ExecutionVariant::QEmbedding(op) => {
                Some(QueuedOperation::Operation(Arc::new(op.clone())))
            }
            ExecutionVariant::Region(op) => Some(QueuedOperation::Merged(
                merge_horizontal::MergedSegments::Region(vec![(node.inner_idx, op.clone())]),
            )),
            ExecutionVariant::QMatMul(op) => {
                Some(QueuedOperation::Operation(Arc::new(op.as_ref().clone())))
            }
            ExecutionVariant::QMatrix(op) => {
                // Skip materializing the dense tensor when every consumer
                // reads the block-quantized data directly (fused reduces and
                // elementwise expressions decode per element; qmatmul and
                // embedding kernels decode per block).
                if self.qmatrix_consumed_raw(exec_idx, node.inner_idx) {
                    return None;
                }
                Some(QueuedOperation::Operation(Arc::new(op.clone())))
            }
            ExecutionVariant::Tensor(_) => None, // Handled in execution loop
        }
    }

    /// Whether every consumer of a quantized-matrix node reads the raw
    /// blocks (region ops decode per block; expression reads decode per
    /// element) rather than needing the dense tensor. Only custom-indexed
    /// expression reads require the dense form.
    fn qmatrix_consumed_raw(&self, exec_idx: ExecutionNodeIndex, inner: NodeIndex) -> bool {
        let mut any = false;
        for consumer in self
            .execution_graph
            .neighbors_directed(exec_idx, petgraph::Direction::Outgoing)
        {
            any = true;
            let raw = match &self.execution_graph[consumer].variant {
                ExecutionVariant::QMatMul(_) | ExecutionVariant::QEmbedding(_) => true,
                ExecutionVariant::Elementwise(nary) => {
                    !nary.inputs.iter().enumerate().any(|(slot, &input)| {
                        input == inner && nary.expression.uses_custom_indexing_for_input(slot)
                    })
                }
                ExecutionVariant::Reduce(reduce) => {
                    !reduce.inputs.iter().enumerate().any(|(slot, &input)| {
                        input == inner && reduce.expression.uses_custom_indexing_for_input(slot)
                    })
                }
                _ => false,
            };
            if !raw {
                return false;
            }
        }
        any
    }

    // --- Rewrite Engine ---

    pub(super) fn optimize(
        &mut self,
        graph: &mut ComputeGraphInner,
        policy: OptimizePolicy,
    ) -> bool {
        // Rebuild composed contraction / normalization clusters into their
        // specialized operations first, while they are still in the exact
        // canonical form the API emitted (before view folding or fusion
        // disturbs them). This phase is unconditional: decode classification
        // is only reliable after recognition has minted QMatMul nodes.
        self.recognize_via_egraph(graph);
        // The qmatmul scan runs after recognition (which can mint QMatMul
        // nodes) and before row fusion (which never creates or removes
        // them), so the dense gate below is structural and stable.
        let has_qmatmul = self.execution_graph.node_indices().any(|node| {
            matches!(
                self.execution_graph[node].variant,
                ExecutionVariant::QMatMul(_)
            )
        });
        let dense = Self::dense_reduce_fusion_enabled(has_qmatmul);
        let large_dense = policy.is_large_graph() && dense;
        let has_dense_matmul = dense
            && self.execution_graph.node_indices().any(|node| {
                matches!(
                    self.execution_graph[node].variant,
                    ExecutionVariant::MatMul(_)
                )
            });
        // Standard dense graphs may merge independent cooperative matmuls and
        // model graphs may absorb mathematically equivalent interleaved views
        // into row programs. Standalone reduction graphs retain their exact
        // legacy ordering; dense codegen and horizontal row/region merging
        // remain restricted to the established large dense profile.
        self.horizontal_merge =
            dense && std::env::var_os("FUSOR_DISABLE_HORIZONTAL_FUSION").is_none();
        self.horizontal_merge_dense_ops = large_dense;
        self.fuse_row_programs(graph, large_dense || has_dense_matmul);
        self.recognize_assign_chains(graph);

        let has_reduce = self.execution_graph.node_indices().any(|node| {
            matches!(
                self.execution_graph[node].variant,
                ExecutionVariant::Reduce(_)
            )
        });
        let has_matmul = self.execution_graph.node_indices().any(|node| {
            matches!(
                self.execution_graph[node].variant,
                ExecutionVariant::MatMul(_)
            )
        });
        use super::egraph::{CandidateKind, ReduceFusion, Stage2Profile};
        let profile = match policy {
            OptimizePolicy::Standard => Stage2Profile {
                candidates: CandidateKind::General,
                reduce_fusion: if !has_reduce {
                    ReduceFusion::Disabled
                } else if dense {
                    ReduceFusion::Dense
                } else {
                    ReduceFusion::Conservative
                },
                try_matmul_fusion: has_matmul || has_qmatmul,
                allow_qmatmul_elementwise_fusion: self.execution_graph.node_count()
                    <= STANDARD_QMATMUL_FUSION_NODE_LIMIT
                    || std::env::var_os("FUSOR_RESOLVE_QMATMUL_ELEMENTWISE_FUSION").is_some(),
                dense,
                skip_externally_live: self.horizontal_merge_dense_ops,
                enable_dense_codegen: false,
            },
            OptimizePolicy::LargeGraph { .. } if has_qmatmul => Stage2Profile {
                candidates: CandidateKind::LargeQuantized,
                reduce_fusion: ReduceFusion::Disabled,
                try_matmul_fusion: true,
                allow_qmatmul_elementwise_fusion: true,
                dense,
                skip_externally_live: self.horizontal_merge_dense_ops,
                enable_dense_codegen: false,
            },
            OptimizePolicy::LargeGraph { .. } => Stage2Profile {
                candidates: CandidateKind::Dense,
                reduce_fusion: if dense && has_reduce {
                    ReduceFusion::Dense
                } else {
                    ReduceFusion::Disabled
                },
                try_matmul_fusion: true,
                allow_qmatmul_elementwise_fusion: true,
                dense,
                skip_externally_live: self.horizontal_merge_dense_ops,
                enable_dense_codegen: dense,
            },
        };

        let is_single_token_decode = has_qmatmul && self.is_single_token_qmatmul_graph();
        let run_fixpoint = policy.runs_stage2_fusion(is_single_token_decode);
        if run_fixpoint {
            self.fuse_via_egraph(graph, profile.clone());
        }

        // Dense large-graph kernel tuning is opted into per operation, after
        // rewrite has settled: matmuls get the wider divisor-aligned split-K
        // fan-out (with elided K bounds), row programs get axis-sized
        // workgroups, subgroup whole-block reductions, and staged reads.
        // Quantized graphs (`has_qmatmul`) leave the flags unset, so decode
        // kernels are byte-identical to the committed lowering.
        if profile.enable_dense_codegen {
            // Region formation generalizes the sole-consumer nary gate: it
            // fuses externally-live producers into multi-output regions.
            // Regions lower independently; horizontal merging may combine
            // them with additional compatible work but is not required.
            self.form_elementwise_regions(graph);
            self.mark_dense_codegen();
        }
        self.coalesce_equivalent_eclasses(graph);
        run_fixpoint
    }

    /// Set `dense_codegen` on every matmul and row-program operation in the
    /// execution graph after the large dense rewrite profile settles.
    fn mark_dense_codegen(&mut self) {
        let nodes: Vec<ExecutionNodeIndex> = self.execution_graph.node_indices().collect();
        for node in nodes {
            match &mut self.execution_graph[node].variant {
                ExecutionVariant::MatMul(op) => op.dense_codegen = true,
                ExecutionVariant::GraphOp(op) => {
                    if let Some(row) = op.as_row_program()
                        && !row.dense_codegen
                    {
                        let mut row = row.clone();
                        row.dense_codegen = true;
                        *op = std::sync::Arc::new(row);
                    }
                }
                _ => {}
            }
        }
    }

    /// Whether the dense-graph reduce-fusion rules may run: never for
    /// graphs containing QMatMul (decode behavior is frozen), and gated by
    /// a kill-switch env var that is also hashed into the flush-replay
    /// fingerprint.
    pub(super) fn dense_reduce_fusion_enabled(has_qmatmul: bool) -> bool {
        !has_qmatmul && std::env::var_os("FUSOR_RESOLVE_DISABLE_DENSE_REDUCE_FUSION").is_none()
    }

    fn is_single_token_qmatmul_graph(&self) -> bool {
        let mut qmatmul_count = 0usize;
        let mut single_token_count = 0usize;
        for node in self.execution_graph.node_indices() {
            let ExecutionVariant::QMatMul(qmatmul) = &self.execution_graph[node].variant else {
                continue;
            };
            qmatmul_count += 1;
            if qmatmul.in_shape.len() >= 2
                && qmatmul.in_shape[..qmatmul.in_shape.len() - 1]
                    .iter()
                    .product::<usize>()
                    == 1
            {
                single_token_count += 1;
            }
        }
        qmatmul_count >= 16 && single_token_count * 4 >= qmatmul_count * 3
    }

    // Helpers
    pub(super) fn add_physical_dependencies(
        &self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        inputs: &[NodeIndex],
    ) {
        let inner_idx = self.execution_graph[node_idx].inner_idx;
        for &input in inputs {
            graph.add_dependency_edge(input, inner_idx);
            if let Some(recorder) = &self.recorder {
                // These edges are persistent inner-graph side effects of the
                // optimizer; a replayed plan must re-add them, so record them.
                recorder.borrow_mut().record_physical_edge(input, inner_idx);
            }
        }
    }

    pub(super) fn get_input_node_in_exec_graph(
        &self,
        inner_input: NodeIndex,
    ) -> Option<ExecutionNodeIndex> {
        self.node_mapping.get(&inner_input).copied()
    }

    /// Walk through view nodes from `inner` down to the first non-view
    /// node, composing each view's collapsed stage stack. Public tensor ops
    /// collapse into single view nodes at construction, but composed
    /// clusters (attention's attached GQA/transpose views) still layer view
    /// nodes deliberately. Returns the base node and the composed layout
    /// over the base's logical value space; the layout is `None` when
    /// `inner` is not a view (identity). Views that don't collapse or
    /// compose (or carry a fill region) act as chain breaks: the walk stops
    /// without seeing through them.
    pub(super) fn walk_view_chain(&self, mut inner: NodeIndex) -> (NodeIndex, Option<Layout>) {
        let mut composed: Option<Layout> = None;
        loop {
            let Some(exec) = self.get_input_node_in_exec_graph(inner) else {
                return (inner, composed);
            };
            let ExecutionVariant::View(view) = &self.execution_graph[exec].variant else {
                return (inner, composed);
            };
            let Some(collapsed) = view.composed_layout() else {
                return (inner, composed);
            };
            let next = match &composed {
                None => collapsed,
                Some(outer) => match crate::view::compose_layouts(outer, &collapsed) {
                    Some(layout) => layout,
                    None => return (inner, composed),
                },
            };
            composed = Some(next);
            inner = view.input;
        }
    }

    /// The layout a (possibly chained-view) node presents over its base
    /// node's value space when the base materializes at `base_layout`.
    /// `None` when the view does not compose with that layout.
    pub(super) fn apply_view_chain(base_layout: &Layout, chain: &Option<Layout>) -> Option<Layout> {
        match chain {
            None => Some(base_layout.clone()),
            Some(view) => crate::view::compose_layouts(view, base_layout),
        }
    }

    pub(super) fn check_cached(&self, graph: &ComputeGraphInner, inner_idx: NodeIndex) -> bool {
        graph.get_cached_result(inner_idx).is_some()
    }

    pub(super) fn remove_node_if_dead(&mut self, node_idx: ExecutionNodeIndex) {
        if !self.execution_graph.contains_node(node_idx) {
            return;
        }
        let inner_idx = self.execution_graph[node_idx].inner_idx;
        if self.targets.contains(&inner_idx) {
            return;
        }
        if self
            .execution_graph
            .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
            .count()
            == 0
        {
            // Collect incoming neighbors before removing
            let incoming: Vec<_> = self
                .execution_graph
                .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                .collect();
            self.execution_graph.remove_node(node_idx);
            // Recursively check if dependencies are now dead
            for dep in incoming {
                self.remove_node_if_dead(dep);
            }
        }
    }
}

#[cfg(test)]
mod optimizer_policy_tests {
    use super::OptimizePolicy;

    #[test]
    fn configured_node_limit_selects_optimizer_policy() {
        let standard = OptimizePolicy::select(600, 1_024, false);
        assert!(matches!(standard, OptimizePolicy::Standard));

        let large = OptimizePolicy::select(1_025, 1_024, false);
        assert!(matches!(large, OptimizePolicy::LargeGraph { .. }));

        let unlimited = OptimizePolicy::select(10_000, 0, false);
        assert!(matches!(unlimited, OptimizePolicy::Standard));
    }

    #[test]
    fn only_large_decode_policy_can_skip_the_rewrite_fixpoint() {
        let standard = OptimizePolicy::select(10, 512, false);
        assert!(standard.runs_stage2_fusion(true));

        let large = OptimizePolicy::select(513, 512, false);
        assert!(!large.runs_stage2_fusion(true));
        assert!(large.runs_stage2_fusion(false));

        let opted_in = OptimizePolicy::select(513, 512, true);
        assert!(opted_in.runs_stage2_fusion(true));
    }
}
