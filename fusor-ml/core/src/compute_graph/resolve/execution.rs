use super::*;

/// Wall-clock per optimizer sub-phase, one resolve. Keep the sections aligned
/// with the actual work: Stage-1 recognition, imperative row/assign fusion,
/// Stage-2 planning and extraction, region formation, and semantic coalescing.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct OptimizePhases {
    pub(super) recognize: Duration,
    pub(super) row_fusion: Duration,
    pub(super) stage2: Duration,
    pub(super) regions: Duration,
    pub(super) coalesce: Duration,
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

    pub(super) fn optimize(&mut self, graph: &mut ComputeGraphInner) {
        // Rebuild composed contraction / normalization clusters into their
        // specialized operations first, while they are still in the exact
        // canonical form the API emitted (before view folding or fusion
        // disturbs them).
        let phase_start = Instant::now();
        self.recognize_operations(graph);
        self.optimize_phases.recognize += phase_start.elapsed();

        // Every graph uses the full optimizer. Individual generators retain
        // their semantic, device and cost gates; graph size is not one of
        // them. The plan memo keeps repeated-layer discovery proportional to
        // unique local structure rather than total layer count.
        self.horizontal_merge = std::env::var_os("FUSOR_DISABLE_HORIZONTAL_FUSION").is_none();
        let phase_start = Instant::now();
        self.fuse_row_programs(graph);
        self.recognize_assign_chains(graph);
        self.optimize_phases.row_fusion += phase_start.elapsed();

        let phase_start = Instant::now();
        self.fuse_operations(graph);
        self.optimize_phases.stage2 += phase_start.elapsed();

        // Region formation generalizes the sole-consumer nary gate: it fuses
        // externally-live producers into multi-output regions. Codegen then
        // selects tuned kernels from operation shape and device capabilities.
        let phase_start = Instant::now();
        self.form_elementwise_regions(graph);
        self.optimize_phases.regions += phase_start.elapsed();

        let phase_start = Instant::now();
        self.coalesce_equivalent_eclasses(graph);
        self.optimize_phases.coalesce += phase_start.elapsed();
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
