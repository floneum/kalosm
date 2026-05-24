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
                    if *count == 0 && !targets.contains(&dep) && !graph.has_live_reference(dep) {
                        // All consumers within this execution have been
                        // processed and no user-held lazy tensor still
                        // transitively depends on `dep` — free the cached
                        // buffer.
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
        if !operation.in_place {
            return None;
        }
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
            variant: variant.clone(),
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

    pub(super) fn lower_node(&self, node: &ExecutionNode) -> Option<QueuedOperation> {
        match &node.variant {
            ComputeGraphNodeVariant::Nary(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ComputeGraphNodeVariant::MatMul(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ComputeGraphNodeVariant::Reduce(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ComputeGraphNodeVariant::FlashAttention(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ComputeGraphNodeVariant::GraphOp(op) => Some(QueuedOperation::Generic(op.clone())),
            ComputeGraphNodeVariant::MapLayout(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ComputeGraphNodeVariant::Resize(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ComputeGraphNodeVariant::SliceAssign(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ComputeGraphNodeVariant::QEmbedding(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ComputeGraphNodeVariant::QMatMul(op) => Some(QueuedOperation::QMatMul(op.clone())),
            ComputeGraphNodeVariant::Dequantize(op) => {
                Some(QueuedOperation::Generic(Arc::new(op.clone())))
            }
            ComputeGraphNodeVariant::Tensor(_) => None, // Handled in execution loop
        }
    }

    // --- Rewrite Engine ---

    pub(super) fn optimize(&mut self, graph: &mut ComputeGraphInner) {
        let profile_enabled = std::env::var_os("FUSOR_TRACE_OPTIMIZE").is_some();
        let mut profile = OptimizeProfile::default();
        // The current rewrite rules can only start from Nary nodes (nary
        // fusion, post-op reduce/matmul fusion) or MatMul nodes (pre-op
        // unary fusion). Avoid scanning every QMatMul/RMS/attention node in
        // decode graphs with hundreds of kernels.
        let has_reduce = self.execution_graph.node_indices().any(|node| {
            matches!(
                self.execution_graph[node].variant,
                ComputeGraphNodeVariant::Reduce(_)
            )
        });
        let has_matmul = self.execution_graph.node_indices().any(|node| {
            matches!(
                self.execution_graph[node].variant,
                ComputeGraphNodeVariant::MatMul(_)
            )
        });
        let has_qmatmul = self.execution_graph.node_indices().any(|node| {
            matches!(
                self.execution_graph[node].variant,
                ComputeGraphNodeVariant::QMatMul(_)
            )
        });
        let has_rmsnorm = self
            .execution_graph
            .node_indices()
            .any(|node| as_rms_norm(&self.execution_graph[node].variant).is_some());
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

            // 1. Fuse naries together (combine expression trees)
            // 2. Try to fuse resulting nary into specialized ops (reduce, matmul, etc.)
            let start = profile_enabled.then(Instant::now);
            // Keep the large-graph fast path to nary fusion by default. The
            // paired qmatmul rewrite still runs in the full optimizer for
            // small graphs, but applying it in the decode-sized fast path can
            // corrupt Llama 3.1 chat output.
            let changed = self.try_fuse_naries(graph, node_idx)
                || (std::env::var_os("FUSOR_RESOLVE_LARGE_GRAPH_PAIRED_QMATMUL").is_some()
                    && self.try_fuse_paired_qmatmul(graph, node_idx));
            if let Some(start) = start {
                profile.fuse_naries_count += 1;
                profile.fuse_naries += start.elapsed();
            }

            let changed = if changed {
                true
            } else {
                let start = profile_enabled.then(Instant::now);
                let changed = has_reduce && self.try_fuse_into_reduce(graph, node_idx);
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

            let changed = if changed {
                true
            } else {
                let start = profile_enabled.then(Instant::now);
                let changed = has_qmatmul && self.try_fuse_paired_qmatmul(graph, node_idx);
                if let Some(start) = start {
                    profile.fuse_paired_qmatmul_count += 1;
                    profile.fuse_paired_qmatmul += start.elapsed();
                }
                changed
            };

            let changed = if changed {
                true
            } else {
                let start = profile_enabled.then(Instant::now);
                let changed = has_rmsnorm && self.try_fuse_into_rmsnorm(graph, node_idx);
                if let Some(start) = start {
                    profile.fuse_rmsnorm_count += 1;
                    profile.fuse_rmsnorm += start.elapsed();
                }
                changed
            };

            if changed {
                profile.changed += 1;
                // Re-add the current node to worklist if it still exists
                if self.execution_graph.contains_node(node_idx)
                    && self.is_optimization_candidate(node_idx)
                    && !in_worklist.contains(&node_idx)
                {
                    worklist.push_back(node_idx);
                    in_worklist.insert(node_idx);
                }

                // Re-add consumers that might be affected by this change.
                for consumer in consumers {
                    if self.execution_graph.contains_node(consumer)
                        && self.is_optimization_candidate(consumer)
                        && !in_worklist.contains(&consumer)
                    {
                        worklist.push_back(consumer);
                        in_worklist.insert(consumer);
                    }
                }

                // Also add new consumers that may have been created.
                if self.execution_graph.contains_node(node_idx) {
                    for consumer in self
                        .execution_graph
                        .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
                    {
                        if self.is_optimization_candidate(consumer)
                            && !in_worklist.contains(&consumer)
                        {
                            worklist.push_back(consumer);
                            in_worklist.insert(consumer);
                        }
                    }
                }
            }
        }
        if profile_enabled {
            profile.print();
        }
    }

    pub(super) fn optimize_large_graph(&mut self, graph: &mut ComputeGraphInner) {
        let has_qmatmul = self.execution_graph.node_indices().any(|node| {
            matches!(
                self.execution_graph[node].variant,
                ComputeGraphNodeVariant::QMatMul(_)
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
            let mut changed = self.try_fuse_naries(graph, node_idx);
            if std::env::var_os("FUSOR_RESOLVE_LARGE_GRAPH_PAIRED_QMATMUL").is_some()
                && self.execution_graph.contains_node(node_idx)
            {
                changed = self.try_fuse_paired_qmatmul(graph, node_idx) || changed;
            }

            if changed {
                if self.execution_graph.contains_node(node_idx)
                    && self.is_large_graph_nary_candidate(node_idx)
                    && !in_worklist.contains(&node_idx)
                {
                    worklist.push_back(node_idx);
                    in_worklist.insert(node_idx);
                }
                for consumer in consumers {
                    if self.execution_graph.contains_node(consumer)
                        && self.is_large_graph_nary_candidate(consumer)
                        && !in_worklist.contains(&consumer)
                    {
                        worklist.push_back(consumer);
                        in_worklist.insert(consumer);
                    }
                }
            }
        }
    }

    pub(super) fn is_large_graph_nary_candidate(&self, node_idx: ExecutionNodeIndex) -> bool {
        let ComputeGraphNodeVariant::Nary(nary) = &self.execution_graph[node_idx].variant else {
            return false;
        };
        nary.shape.last().copied().unwrap_or_default() >= 1024
    }

    pub(super) fn is_single_token_qmatmul_graph(&self) -> bool {
        let mut qmatmul_count = 0usize;
        let mut single_token_count = 0usize;
        for node in self.execution_graph.node_indices() {
            let ComputeGraphNodeVariant::QMatMul(qmatmul) = &self.execution_graph[node].variant
            else {
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
            ComputeGraphNodeVariant::Nary(_)
                | ComputeGraphNodeVariant::MatMul(_)
                | ComputeGraphNodeVariant::QMatMul(_)
        ) || as_rms_norm(&self.execution_graph[node_idx].variant).is_some()
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

    pub(super) fn walk_map_layout_chain(
        &self,
        mut inner: NodeIndex,
    ) -> Option<(NodeIndex, Vec<crate::map_layout::MapLayoutOperation>)> {
        let mut chain = Vec::new();
        loop {
            let Some(exec) = self.get_input_node_in_exec_graph(inner) else {
                if chain.is_empty() {
                    return None;
                }
                chain.reverse();
                return Some((inner, chain));
            };
            let ComputeGraphNodeVariant::MapLayout(map) =
                self.execution_graph[exec].variant.clone()
            else {
                chain.reverse();
                return Some((inner, chain));
            };
            inner = map.input;
            chain.push(map);
        }
    }

    pub(super) fn apply_map_layout_chain(
        base: &Layout,
        chain: &[crate::map_layout::MapLayoutOperation],
    ) -> Layout {
        chain
            .iter()
            .fold(base.clone(), |layout, map| map.map_layout(&layout))
    }

    pub(super) fn infer_layout(
        graph: &ComputeGraphInner,
        inner_idx: NodeIndex,
    ) -> Option<crate::TensorLayoutInfo> {
        let mut pass = LayoutPass::default();
        pass.visit(graph, inner_idx);
        pass.output_layout.remove(&inner_idx)
    }

    pub(super) fn infer_layout_cached(
        &mut self,
        graph: &ComputeGraphInner,
        inner_idx: NodeIndex,
    ) -> Option<crate::TensorLayoutInfo> {
        if let Some(layout) = self.layout_cache.get(&inner_idx) {
            return layout.clone();
        }
        let layout = Self::infer_layout(graph, inner_idx);
        self.layout_cache.insert(inner_idx, layout.clone());
        layout
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

        let (base_inner, _) = self
            .walk_map_layout_chain(extra_inner)
            .unwrap_or((extra_inner, Vec::new()));
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
