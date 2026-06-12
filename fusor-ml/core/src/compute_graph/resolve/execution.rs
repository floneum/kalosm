use super::*;

impl Resolver {
    pub(super) fn release_dead_intermediates(
        graph: &mut ComputeGraphInner,
        produced_ops: &[&QueuedOperation],
        remaining_consumers: &mut FxHashMap<NodeIndex, usize>,
        targets: &FxHashSet<NodeIndex>,
    ) {
        for op in produced_ops {
            op.visit_dependencies(&mut |dep| {
                if let Some(count) = remaining_consumers.get_mut(&dep) {
                    *count = count.saturating_sub(1);
                    if *count == 0
                        && !targets.contains(&dep)
                        && !graph.has_live_lazy_descendant(dep)
                    {
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
                        && let Some(node) = graph.nodes.nodes.node_weight_mut(dep)
                    {
                        node.cached = None;
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
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ExecutionVariant::MatMul(op) => Some(QueuedOperation::Generic(Arc::new(op.clone()))),
            ExecutionVariant::Reduce(op) => Some(QueuedOperation::Generic(Arc::new(op.clone()))),
            ExecutionVariant::GraphOp(op) => Some(QueuedOperation::Generic(op.clone())),
            ExecutionVariant::View(op) => Some(QueuedOperation::Generic(Arc::new(op.clone()))),
            ExecutionVariant::Assign(op) => Some(QueuedOperation::Generic(Arc::new(op.clone()))),
            ExecutionVariant::QEmbedding(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ExecutionVariant::QMatMul(op) => Some(QueuedOperation::QMatMul(op.clone())),
            ExecutionVariant::QMatrix(op) => {
                // Skip materializing the dense tensor when every consumer
                // reads the block-quantized data directly (fused reduces and
                // elementwise expressions decode per element; qmatmul and
                // embedding kernels decode per block).
                if self.qmatrix_consumed_raw(exec_idx, node.inner_idx) {
                    return None;
                }
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
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
        let profile_enabled = std::env::var_os("FUSOR_TRACE_OPTIMIZE").is_some();
        let mut profile = OptimizeProfile::default();
        // Rebuild composed contraction / normalization clusters into their
        // specialized operations first, while they are still in the exact
        // canonical form the API emitted (before view folding or fusion
        // disturbs them).
        self.recognize_contractions(graph);
        self.recognize_embeddings(graph);
        self.recognize_attention(graph);
        self.fuse_row_programs(graph);
        self.recognize_assign_chains(graph);
        // The current rewrite rules can only start from Nary nodes (nary
        // fusion, post-op reduce/matmul fusion) or MatMul nodes (pre-op
        // unary fusion). Avoid scanning every QMatMul/attention node in
        // decode graphs with hundreds of kernels.
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
        let has_qmatmul = self.execution_graph.node_indices().any(|node| {
            matches!(
                self.execution_graph[node].variant,
                ExecutionVariant::QMatMul(_)
            )
        });
        let allow_qmatmul_elementwise_fusion = self.execution_graph.node_count()
            <= DEFAULT_OPTIMIZE_NODE_LIMIT
            || std::env::var_os("FUSOR_RESOLVE_QMATMUL_ELEMENTWISE_FUSION").is_some();
        let mut worklist: VecDeque<ExecutionNodeIndex> = self
            .execution_graph
            .node_indices()
            .filter(|&node| self.is_optimization_candidate(node))
            .collect();
        let mut in_worklist: FxHashSet<ExecutionNodeIndex> = worklist.iter().copied().collect();

        while let Some(node_idx) = worklist.pop_front() {
            profile.iterations += 1;
            in_worklist.remove(&node_idx);

            if !self.execution_graph.contains_node(node_idx) {
                continue;
            }

            // Edges are dependency -> consumer, and only downstream nodes can
            // become newly fusible from these rewrites.
            let consumers: Vec<_> = self
                .execution_graph
                .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                .collect();

            // 1. Fold view inputs into the nary body so fusion sees through
            //    layout changes
            // 2. Fuse naries together (combine expression trees)
            // 3. Try to fuse resulting nary into specialized ops (reduce, matmul, etc.)
            let changed = self.try_fold_view_inputs(graph, node_idx);

            let start = profile_enabled.then(Instant::now);
            let changed = changed | self.try_fuse_naries(graph, node_idx);
            if let Some(start) = start {
                profile.fuse_naries_count += 1;
                profile.fuse_naries += start.elapsed();
            }

            let changed = if changed {
                true
            } else {
                let start = profile_enabled.then(Instant::now);
                let changed = has_reduce
                    && (self.try_fuse_into_reduce(graph, node_idx)
                        || self.try_fuse_producer_into_reduce(graph, node_idx));
                if let Some(start) = start {
                    profile.fuse_reduce_count += 1;
                    profile.fuse_reduce += start.elapsed();
                }
                changed
            };

            let changed = if changed {
                true
            } else {
                let start = profile_enabled.then(Instant::now);
                let changed = (has_matmul || has_qmatmul)
                    && self.try_fuse_into_matmul(graph, node_idx, allow_qmatmul_elementwise_fusion);
                if let Some(start) = start {
                    profile.fuse_matmul_count += 1;
                    profile.fuse_matmul += start.elapsed();
                }
                changed
            };

            if changed {
                profile.changed += 1;
                // Re-add the current node to worklist if it still exists
                if self.execution_graph.contains_node(node_idx)
                    && self.is_optimization_candidate(node_idx)
                    && in_worklist.insert(node_idx)
                {
                    worklist.push_back(node_idx);
                }

                // Re-add downstream fusion candidates that might now be fusible
                // — both the consumers captured before this rewrite and any it
                // created — descending through view nodes (e.g. the MapLayout
                // broadcast `add_` inserts) that sit between a changed node and
                // the next candidate.
                self.enqueue_downstream_candidates(
                    consumers,
                    Self::is_optimization_candidate,
                    &mut worklist,
                    &mut in_worklist,
                );
                if self.execution_graph.contains_node(node_idx) {
                    let new_consumers: Vec<_> = self
                        .execution_graph
                        .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                        .collect();
                    self.enqueue_downstream_candidates(
                        new_consumers,
                        Self::is_optimization_candidate,
                        &mut worklist,
                        &mut in_worklist,
                    );
                }
            }
        }
        if profile_enabled {
            profile.print();
        }
    }

    pub(super) fn optimize_large_graph(&mut self, graph: &mut ComputeGraphInner) {
        self.recognize_contractions(graph);
        self.recognize_embeddings(graph);
        self.recognize_attention(graph);
        self.fuse_row_programs(graph);
        self.recognize_assign_chains(graph);
        let has_qmatmul = self.execution_graph.node_indices().any(|node| {
            matches!(
                self.execution_graph[node].variant,
                ExecutionVariant::QMatMul(_)
            )
        });
        if !has_qmatmul {
            return;
        }

        let mut worklist = self
            .execution_graph
            .node_indices()
            .filter(|&node| self.is_large_graph_nary_candidate(node))
            .collect::<VecDeque<_>>();
        let mut in_worklist = worklist.iter().copied().collect::<FxHashSet<_>>();

        while let Some(node_idx) = worklist.pop_front() {
            in_worklist.remove(&node_idx);
            if !self.execution_graph.contains_node(node_idx) {
                continue;
            }

            let consumers = self
                .execution_graph
                .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                .collect::<Vec<_>>();
            let mut changed = self.try_fold_view_inputs(graph, node_idx);
            changed |= self.try_fuse_naries(graph, node_idx);
            if !changed && self.execution_graph.contains_node(node_idx) {
                changed = self.try_fuse_into_matmul(graph, node_idx, true);
            }

            if changed {
                if self.execution_graph.contains_node(node_idx)
                    && self.is_large_graph_nary_candidate(node_idx)
                    && in_worklist.insert(node_idx)
                {
                    worklist.push_back(node_idx);
                }
                self.enqueue_downstream_candidates(
                    consumers,
                    Self::is_large_graph_nary_candidate,
                    &mut worklist,
                    &mut in_worklist,
                );
            }
        }
    }

    /// Re-enqueue downstream fusion candidates reachable from `seeds`,
    /// descending through `MapLayout` view nodes. A rewrite (e.g. fusing an
    /// `add` into a qmatmul epilogue) can make a candidate that sits *behind* a
    /// broadcast/narrow view newly fusible; those views are not optimization
    /// candidates themselves, so a plain direct-consumer scan would never reach
    /// the candidate past them.
    fn enqueue_downstream_candidates(
        &self,
        seeds: impl IntoIterator<Item = ExecutionNodeIndex>,
        is_candidate: impl Fn(&Self, ExecutionNodeIndex) -> bool,
        worklist: &mut VecDeque<ExecutionNodeIndex>,
        in_worklist: &mut FxHashSet<ExecutionNodeIndex>,
    ) {
        let mut stack: Vec<ExecutionNodeIndex> = seeds.into_iter().collect();
        let mut visited = FxHashSet::default();
        while let Some(node) = stack.pop() {
            if !self.execution_graph.contains_node(node) || !visited.insert(node) {
                continue;
            }
            if is_candidate(self, node) {
                if in_worklist.insert(node) {
                    worklist.push_back(node);
                }
            } else if matches!(
                self.execution_graph[node].variant,
                ExecutionVariant::View(_)
            ) {
                stack.extend(
                    self.execution_graph
                        .neighbors_directed(node, petgraph::Direction::Outgoing),
                );
            }
        }
    }

    pub(super) fn is_large_graph_nary_candidate(&self, node_idx: ExecutionNodeIndex) -> bool {
        let ExecutionVariant::Elementwise(nary) = &self.execution_graph[node_idx].variant else {
            return false;
        };
        if nary.shape.last().copied().unwrap_or_default() >= LARGE_GRAPH_NARY_FUSION_MIN_LAST_DIM {
            return true;
        }

        nary.inputs.iter().any(|&input| {
            let (base_inner, _) = self.walk_view_chain(input);
            self.get_input_node_in_exec_graph(base_inner)
                .is_some_and(|exec_idx| {
                    matches!(
                        self.execution_graph[exec_idx].variant,
                        ExecutionVariant::QMatMul(_)
                    )
                })
        })
    }

    pub(super) fn is_single_token_qmatmul_graph(&self) -> bool {
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

    pub(super) fn is_optimization_candidate(&self, node_idx: ExecutionNodeIndex) -> bool {
        matches!(
            self.execution_graph[node_idx].variant,
            ExecutionVariant::Elementwise(_)
                | ExecutionVariant::MatMul(_)
                | ExecutionVariant::QMatMul(_)
                | ExecutionVariant::Reduce(_)
        )
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

    pub(super) fn infer_layout_cached(
        &mut self,
        graph: &ComputeGraphInner,
        inner_idx: NodeIndex,
    ) -> Option<crate::TensorLayoutInfo> {
        self.layout_pass.visit(graph, inner_idx);
        self.layout_pass.output_layout.get(&inner_idx).cloned()
    }

    pub(super) fn try_normalize_qmatmul_post_extra(
        &mut self,
        graph: &ComputeGraphInner,
        extra_inner: NodeIndex,
        output_shape: &[usize],
    ) -> Option<NodeIndex> {
        let last_dim = *output_shape.last()?;
        let extra_info = self.infer_layout_cached(graph, extra_inner)?;
        if extra_info.datatype() != DataTypeEnum::F32 || extra_info.layout().shape() != output_shape
        {
            return None;
        }

        let layout = extra_info.layout();
        let is_column_broadcast = layout.offset() == 0
            && layout.strides().last().copied() == Some(1)
            && layout.shape().last().copied() == Some(last_dim)
            && layout.strides()[..layout.strides().len().saturating_sub(1)]
                .iter()
                .all(|stride| *stride == 0);
        if !is_column_broadcast {
            return Some(extra_inner);
        }

        let (base_inner, _) = self.walk_view_chain(extra_inner);
        let base_info = self.infer_layout_cached(graph, base_inner)?;
        let base_layout = base_info.layout();
        if base_info.datatype() == DataTypeEnum::F32
            && base_layout.shape() == [last_dim]
            && base_layout.is_contiguous()
            && base_layout.offset() == 0
        {
            Some(base_inner)
        } else {
            Some(extra_inner)
        }
    }

    pub(super) fn check_cached(&self, graph: &ComputeGraphInner, inner_idx: NodeIndex) -> bool {
        graph.get_cached_result(inner_idx).is_some()
    }

    pub(super) fn remove_node_if_dead(&mut self, node_idx: ExecutionNodeIndex) {
        if !self.execution_graph.contains_node(node_idx) {
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
