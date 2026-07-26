use super::*;

/// Wall-clock per optimizer phase, one resolve.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct OptimizePhases {
    pub(super) recognition: Duration,
    pub(super) extraction: Duration,
    pub(super) physical: Duration,
}

/// The remaining-consumer bookkeeping a release pass decrements. A live
/// resolve counts by inner node index; a replayed plan counts by plan slot.
pub(super) trait ConsumerCounts {
    type Key: Copy;

    /// Drop one consumer of `key`, returning the node whose cached buffer is
    /// now dead: `None` while consumers remain, while the key is untracked,
    /// or when the node is an output of this execution.
    fn consume(&mut self, key: Self::Key) -> Option<NodeIndex>;
}

pub(super) struct NodeConsumers<'a> {
    pub(super) counts: &'a mut FxHashMap<NodeIndex, usize>,
    pub(super) targets: &'a FxHashSet<NodeIndex>,
}

impl ConsumerCounts for NodeConsumers<'_> {
    type Key = NodeIndex;

    fn consume(&mut self, node: NodeIndex) -> Option<NodeIndex> {
        let count = self.counts.get_mut(&node)?;
        *count = count.saturating_sub(1);
        (*count == 0 && !self.targets.contains(&node)).then_some(node)
    }
}

pub(super) struct SlotConsumers<'a> {
    pub(super) slots: &'a [NodeIndex],
    pub(super) counts: &'a mut [u32],
    pub(super) is_target: &'a [bool],
}

impl ConsumerCounts for SlotConsumers<'_> {
    type Key = u32;

    fn consume(&mut self, slot: u32) -> Option<NodeIndex> {
        let slot = slot as usize;
        let count = &mut self.counts[slot];
        *count = count.saturating_sub(1);
        (*count == 0 && !self.is_target[slot]).then(|| self.slots[slot])
    }
}

/// Free the cached buffers of the dependencies `visit` yields. A buffer is
/// released once all consumers within this execution have been processed and
/// no user-held lazy tensor still transitively depends on it. The descendant
/// check must include `live_descendant_count`, not just direct references:
/// clearing `cached` on a node that still has an alive-uncached descendant
/// flips it back to alive-uncached without propagating the transition,
/// undercounting every ancestor's descendant counter. Because
/// `has_live_lazy_descendant` is consulted here, reference-count drift
/// invisible to a replayed plan's structural fingerprint is handled exactly
/// as a full resolve handles it.
pub(super) fn release_consumed<C: ConsumerCounts>(
    graph: &mut ComputeGraphInner,
    counts: &mut C,
    mut ledger: Option<&mut super::alloc_reuse::BufferLedger>,
    visit: impl FnOnce(&mut dyn FnMut(C::Key)),
) {
    visit(&mut |key| {
        let Some(dep) = counts.consume(key) else {
            return;
        };
        if graph.has_live_lazy_descendant(dep) {
            return;
        }
        if let Some(ledger) = ledger.as_deref_mut()
            && let Some(cached) = graph.get_cached_result(dep)
        {
            ledger.note_released(dep, cached);
        }
        if let Some(node) = graph.nodes.nodes.node_weight_mut(dep) {
            node.cached = None;
        }
    });
}

impl Resolver {
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
            // A fold lowers through its reduce form. Multi-slot carriers have
            // no kernel yet, and nothing constructs one, so reaching this with
            // a general fold is a wiring bug rather than a missing feature.
            ExecutionVariant::Fold(op) => {
                let reduce = op.to_reduce().expect(
                    "a multi-slot fold reached lowering; only reduce-form folds are lowerable",
                );
                Some(QueuedOperation::Operation(Arc::new(reduce)))
            }
            ExecutionVariant::RowProgram(op) => {
                Some(QueuedOperation::Operation(Arc::new(op.clone())))
            }
            ExecutionVariant::Attention(op) => {
                Some(QueuedOperation::Operation(Arc::new(op.clone())))
            }
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
        // Every graph uses the full optimizer. Recognition and fusion share
        // one value e-graph; the structural plan memo reuses discovery across
        // repeated layers while preserving allocation-distinct values.
        self.optimize_operations(graph);

        // Region formation generalizes the sole-consumer nary gate: it fuses
        // externally-live producers into multi-output regions. Codegen then
        // selects tuned kernels from operation shape and device capabilities.
        let phase_start = Instant::now();
        self.form_elementwise_regions(graph);
        self.optimize_phases.physical += phase_start.elapsed();
        #[cfg(feature = "graphvis")]
        if let Some(dir) = &graph.device().config().dump_stages {
            super::visualize::dump_stage(
                dir,
                &self.execution_graph,
                super::visualize::Stage::Regions,
            );
        }
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

    /// [`egraph::compose::walk_view_chain`] over the execution graph's
    /// current forms.
    pub(super) fn walk_view_chain(&self, inner: NodeIndex) -> (NodeIndex, Option<Layout>) {
        egraph::compose::walk_view_chain(inner, |inner| {
            let exec = self.get_input_node_in_exec_graph(inner)?;
            let ExecutionVariant::View(view) = &self.execution_graph[exec].variant else {
                return None;
            };
            Some((view.composed_layout()?, view.input))
        })
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
            self.node_mapping.remove(&inner_idx);
            // Recursively check if dependencies are now dead
            for dep in incoming {
                self.remove_node_if_dead(dep);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    #[test]
    fn dead_node_removal_clears_its_inner_mapping() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let input = Tensor::new(&device, &[1.0f32, 2.0, 3.0, 4.0]);
            let intermediate = &input + 1.0;
            let output = &intermediate * 2.0;
            let target = output.data().key;
            let intermediate_inner = intermediate.data().key;

            device.compute_graph().with_mut(|graph| {
                let mut resolver = Resolver::new_batch(graph, vec![target]);
                resolver.build_execution_graph(graph, target);
                let intermediate_exec = resolver.node_mapping[&intermediate_inner];
                let target_exec = resolver.node_mapping[&target];
                let edge = resolver
                    .execution_graph
                    .find_edge(intermediate_exec, target_exec)
                    .expect("intermediate feeds the target");
                resolver.execution_graph.remove_edge(edge);
                resolver.remove_node_if_dead(intermediate_exec);

                assert!(!resolver.execution_graph.contains_node(intermediate_exec));
                assert!(
                    !resolver.node_mapping.contains_key(&intermediate_inner),
                    "removed execution nodes must not remain addressable"
                );
            });
        });
    }
}
