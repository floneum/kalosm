use super::*;

impl Resolver {
    pub(super) fn try_fuse_naries(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        let ComputeGraphNodeVariant::Nary(nary) = node_variant else {
            return false;
        };

        // Collect all fusible nary inputs
        let mut expression = nary.expression.clone();
        let mut all_inputs = nary.inputs.clone();
        let mut fused_execs = Vec::new();

        // Get the max storage buffers limit from GPU
        let max_storage_bindings =
            graph.device().limits().max_storage_buffers_per_shader_stage as usize;

        for (_input_idx, &input_inner) in nary.inputs.iter().enumerate() {
            if self.check_cached(graph, input_inner) {
                continue;
            }
            let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) else {
                continue;
            };
            // Check if the node still exists (it may have been removed during optimization)
            if !self.execution_graph.contains_node(input_exec) {
                continue;
            }
            let ComputeGraphNodeVariant::Nary(input_nary) =
                &self.execution_graph[input_exec].variant
            else {
                continue;
            };

            // Inline: offset input nary's indices to append after current inputs.
            let offset = all_inputs.len();
            let inlined = Self::offset_input_indices(&input_nary.expression, offset);
            // `input_inner` may appear in `all_inputs` at multiple slots —
            // beyond the explicit `input_idx` slot from `nary.inputs`, earlier
            // fusions in this same loop can have inlined chains that
            // re-introduce `input_inner` at later slots. Substitute at every
            // such slot so we don't leave dangling `IndexedInput` references
            // pointing to a now-fused-away node.
            let target_slots: Vec<usize> = all_inputs
                .iter()
                .enumerate()
                .filter_map(|(slot, value)| (*value == input_inner).then_some(slot))
                .collect();
            let mut new_expression = expression.clone();
            let mut success = true;
            for slot in target_slots {
                let (next, s) = Self::substitute_input_in_expr(&new_expression, slot, &inlined);
                new_expression = next;
                success &= s;
            }

            // Only fuse if substitution was successful
            // If not, the expression still references the original input which must remain
            if success {
                // Check if fusing would exceed the GPU's per-stage buffer limit.
                // On Metal, `max_buffers_per_stage` is 31 and covers ALL buffer
                // types (storage + uniform) plus an implicit sizes buffer added
                // by wgpu. Each unique tensor input needs at least 1 storage
                // binding, and there will also be a small number of uniform
                // "info" bindings (for tensor shape/stride metadata), plus the
                // output storage binding, plus the wgpu sizes buffer.
                //
                // We use `max_storage_bindings` (which equals `max_buffers_per_stage`
                // on Metal = 31) as the hard ceiling and reserve slots for:
                //   - 1 output storage binding
                //   - 1 wgpu sizes buffer
                //   - up to `info_headroom` uniform info bindings
                let info_headroom = 4usize;
                let max_fused_inputs = max_storage_bindings.saturating_sub(2 + info_headroom);

                // Count unique inputs after potential merge (duplicates share a binding)
                let unique_inputs: FxHashSet<_> = all_inputs
                    .iter()
                    .chain(input_nary.inputs.iter())
                    .copied()
                    .collect();

                if unique_inputs.len() >= max_fused_inputs {
                    // Skip fusion - would exceed GPU binding limit
                    continue;
                }

                expression = new_expression;
                all_inputs.extend(input_nary.inputs.iter().copied());
                fused_execs.push((input_exec, input_nary.inputs.clone()));
            }
        }

        if fused_execs.is_empty() {
            return false;
        }

        // Deduplicate and remove unused inputs
        let (final_inputs, final_expression) = Self::deduplicate_inputs(all_inputs, expression);

        let new_nary = NaryOperation {
            inputs: final_inputs.clone(),
            expression: final_expression,
            shape: nary.shape.clone(),
            output_datatype: nary.output_datatype,
        };

        self.execution_graph[node_idx].variant = ComputeGraphNodeVariant::Nary(new_nary.clone());

        // Update graph edges
        for (input_exec, new_inputs) in fused_execs {
            if let Some(edge) = self.execution_graph.find_edge(input_exec, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
            for &new_input in &new_inputs {
                if let Some(exec) = self.get_input_node_in_exec_graph(new_input)
                    && self.execution_graph.find_edge(exec, node_idx).is_none()
                {
                    self.execution_graph.add_edge(exec, node_idx, ());
                }
            }
            self.remove_node_if_dead(input_exec);
        }

        self.add_physical_dependencies(graph, node_idx, &new_nary.inputs);
        true
    }

    /// Add offset to all input indices in an expression.
    pub(super) fn offset_input_indices(expr: &NaryExpr, offset: usize) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|c| Self::offset_input_indices(c, offset))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => NaryExpr::IndexedInput {
                input_idx: input_idx + offset,
                indices: indices
                    .iter()
                    .map(|c| Self::offset_input_indices(c, offset))
                    .collect(),
            },
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        }
    }

    /// Substitute IndexedInput(target_idx) with element-wise access with the replacement expression.
    /// Returns (new_expression, success) where success is true if all references to target_idx
    /// were successfully substituted. If false, the input should NOT be removed from the graph.
    pub(super) fn substitute_input_in_expr(
        expr: &NaryExpr,
        target_idx: usize,
        replacement: &NaryExpr,
    ) -> (NaryExpr, bool) {
        /// Helper to extract input_idx from an IndexedInput with element-wise access
        fn get_elementwise_input_idx(expr: &NaryExpr) -> Option<usize> {
            match expr {
                NaryExpr::IndexedInput { input_idx, indices }
                    if NaryExpr::is_elementwise_indices(indices) =>
                {
                    Some(*input_idx)
                }
                _ => None,
            }
        }

        match expr {
            NaryExpr::Op { children, function } => {
                let mut all_success = true;
                let new_children: Vec<_> = children
                    .iter()
                    .map(|c| {
                        let (new_c, success) =
                            Self::substitute_input_in_expr(c, target_idx, replacement);
                        all_success &= success;
                        new_c
                    })
                    .collect();
                (
                    NaryExpr::Op {
                        children: new_children,
                        function: function.clone(),
                    },
                    all_success,
                )
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                if *input_idx == target_idx {
                    // Check if this is element-wise access
                    if NaryExpr::is_elementwise_indices(indices) {
                        // Element-wise can be fully replaced with any expression
                        (replacement.clone(), true)
                    } else {
                        // Custom indexing can only substitute if replacement is also element-wise
                        if let Some(new_idx) = get_elementwise_input_idx(replacement) {
                            let mut all_success = true;
                            let new_indices: Vec<_> = indices
                                .iter()
                                .map(|c| {
                                    let (new_c, success) =
                                        Self::substitute_input_in_expr(c, target_idx, replacement);
                                    all_success &= success;
                                    new_c
                                })
                                .collect();
                            (
                                NaryExpr::IndexedInput {
                                    input_idx: new_idx,
                                    indices: new_indices,
                                },
                                all_success,
                            )
                        } else {
                            // Cannot fuse complex expression into custom indexed input
                            let all_success = false;
                            let new_indices: Vec<_> = indices
                                .iter()
                                .map(|c| {
                                    let (new_c, _) =
                                        Self::substitute_input_in_expr(c, target_idx, replacement);
                                    new_c
                                })
                                .collect();
                            (
                                NaryExpr::IndexedInput {
                                    input_idx: *input_idx,
                                    indices: new_indices,
                                },
                                all_success,
                            )
                        }
                    }
                } else {
                    // Recurse into the index expressions
                    let mut all_success = true;
                    let new_indices: Vec<_> = indices
                        .iter()
                        .map(|c| {
                            let (new_c, s) =
                                Self::substitute_input_in_expr(c, target_idx, replacement);
                            all_success &= s;
                            new_c
                        })
                        .collect();
                    (
                        NaryExpr::IndexedInput {
                            input_idx: *input_idx,
                            indices: new_indices,
                        },
                        all_success,
                    )
                }
            }
            NaryExpr::DimIndex(dim) => (NaryExpr::DimIndex(*dim), true),
            NaryExpr::Scalar(value) => (NaryExpr::Scalar(*value), true),
        }
    }

    /// Remove unused inputs and deduplicate, returning new inputs and remapped expression.
    pub(super) fn deduplicate_inputs(
        inputs: Vec<NodeIndex>,
        expr: NaryExpr,
    ) -> (Vec<NodeIndex>, NaryExpr) {
        // Collect which input indices are actually used
        let mut seen_indices = FxHashSet::default();
        let mut used_indices = Vec::new();
        Self::collect_used_inputs(&expr, &mut seen_indices, &mut used_indices);

        // Build mapping: old index -> new index, and collect only used inputs
        let mut new_inputs = Vec::new();
        let mut old_to_new = FxHashMap::default();

        for old_idx in used_indices {
            let node = inputs[old_idx];
            // Check if this node already exists in new_inputs (deduplication)
            let new_idx = if let Some(existing) = new_inputs.iter().position(|&n| n == node) {
                existing
            } else {
                let idx = new_inputs.len();
                new_inputs.push(node);
                idx
            };
            old_to_new.insert(old_idx, new_idx);
        }

        let new_expr = Self::remap_input_indices(&expr, &old_to_new);
        (new_inputs, new_expr)
    }

    pub(super) fn collect_used_inputs(
        expr: &NaryExpr,
        seen: &mut FxHashSet<usize>,
        used: &mut Vec<usize>,
    ) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    Self::collect_used_inputs(child, seen, used);
                }
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                if seen.insert(*input_idx) {
                    used.push(*input_idx);
                }
                for c in indices {
                    Self::collect_used_inputs(c, seen, used);
                }
            }
            NaryExpr::DimIndex(_) => {}
            NaryExpr::Scalar(_) => {}
        }
    }

    pub(super) fn remap_input_indices(
        expr: &NaryExpr,
        mapping: &FxHashMap<usize, usize>,
    ) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|c| Self::remap_input_indices(c, mapping))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => NaryExpr::IndexedInput {
                input_idx: mapping[input_idx],
                indices: indices
                    .iter()
                    .map(|c| Self::remap_input_indices(c, mapping))
                    .collect(),
            },
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        }
    }

    pub(super) fn replace_inputs_in_expr(
        expr: &NaryExpr,
        replacements: &[Option<NaryExpr>],
    ) -> Option<NaryExpr> {
        match expr {
            NaryExpr::Op { children, function } => Some(NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| Self::replace_inputs_in_expr(child, replacements))
                    .collect::<Option<Vec<_>>>()?,
                function: function.clone(),
            }),
            NaryExpr::IndexedInput { input_idx, indices } => {
                if let Some(replacement) = replacements.get(*input_idx).and_then(|r| r.as_ref()) {
                    if NaryExpr::is_elementwise_indices(indices) {
                        Some(replacement.clone())
                    } else {
                        None
                    }
                } else {
                    Some(NaryExpr::IndexedInput {
                        input_idx: *input_idx,
                        indices: indices
                            .iter()
                            .map(|index| Self::replace_inputs_in_expr(index, replacements))
                            .collect::<Option<Vec<_>>>()?,
                    })
                }
            }
            NaryExpr::DimIndex(dim) => Some(NaryExpr::DimIndex(*dim)),
            NaryExpr::Scalar(value) => Some(NaryExpr::Scalar(*value)),
        }
    }

    pub(super) fn qmatmul_same_base(first: &QMatMulOperation, second: &QMatMulOperation) -> bool {
        first.input_datatype == second.input_datatype
            && first.input == second.input
            && first.matrix == second.matrix
            && first.in_shape == second.in_shape
            && first.out_shape == second.out_shape
            && first.pre_element_wise_expr == second.pre_element_wise_expr
            && first.paired.is_none()
            && second.paired.is_none()
    }

    pub(super) fn qmatmul_output_expr(
        qmatmul: &QMatMulOperation,
        extras: &mut Vec<NodeIndex>,
        rank: usize,
    ) -> Option<NaryExpr> {
        if let Some(epilogue) = &qmatmul.post_element_wise_expr {
            let mut mapping = Vec::with_capacity(1 + epilogue.extras.len());
            mapping.push(0);
            mapping.extend((0..epilogue.extras.len()).map(|i| extras.len() + 1 + i));
            extras.extend(epilogue.extras.iter().copied());
            Some(epilogue.expression.remap_inputs(&mapping))
        } else {
            Some(NaryExpr::input(0, rank))
        }
    }

    /// Try to extract a unary function chain from a node variant.
    /// Only Nary ops with a single input and element-wise access can be converted.
    pub(super) fn try_get_unary_chain(
        variant: &ComputeGraphNodeVariant,
    ) -> Option<ExtractedUnaryChain> {
        match variant {
            ComputeGraphNodeVariant::Nary(nary) => nary.try_extract_unary_chain(),
            _ => None,
        }
    }

    pub(super) fn try_fuse_into_reduce(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        let Some(el_op) = Self::try_get_unary_chain(&node_variant) else {
            return false;
        };

        let input_inner = el_op.value;
        if self.check_cached(graph, input_inner) {
            return false;
        }

        let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner) else {
            return false;
        };

        let input_variant = self.execution_graph[input_exec_idx].variant.clone();
        let ComputeGraphNodeVariant::Reduce(reduce_op) = input_variant else {
            return false;
        };

        let mut new_reduce = reduce_op.clone();
        let mut existing_post = new_reduce.post_element_wise.functions.clone();
        existing_post.extend(el_op.functions.functions.iter().cloned());
        new_reduce.post_element_wise =
            UnaryFunctionChain::new(existing_post, reduce_op.post_element_wise.input_datatype());

        self.execution_graph[node_idx].variant =
            ComputeGraphNodeVariant::Reduce(new_reduce.clone());

        let reduce_input_inner = reduce_op.value;
        if let Some(reduce_input_exec) = self.get_input_node_in_exec_graph(reduce_input_inner) {
            self.execution_graph
                .add_edge(reduce_input_exec, node_idx, ());
        }

        if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
            self.execution_graph.remove_edge(edge);
        }
        self.add_physical_dependencies(graph, node_idx, &[reduce_input_inner]);
        self.remove_node_if_dead(input_exec_idx);
        true
    }

    pub(super) fn try_fuse_into_rmsnorm(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let ComputeGraphNodeVariant::Nary(_) = &self.execution_graph[node_idx].variant else {
            return false;
        };
        let node_variant = self.execution_graph[node_idx].variant.clone();
        let Some(el_op) = Self::try_get_unary_chain(&node_variant) else {
            return false;
        };
        let input_inner = el_op.value;
        if self.check_cached(graph, input_inner) {
            return false;
        }
        let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner) else {
            return false;
        };
        let input_variant = self.execution_graph[input_exec_idx].variant.clone();
        let Some(rms_op) = as_rms_norm(&input_variant) else {
            return false;
        };
        let mut new_rms = rms_op.clone();
        let mut existing = new_rms.post_element_wise.functions.clone();
        existing.extend(el_op.functions.functions.iter().cloned());
        new_rms.post_element_wise =
            UnaryFunctionChain::new(existing, rms_op.post_element_wise.input_datatype());

        self.execution_graph[node_idx].variant =
            ComputeGraphNodeVariant::GraphOp(Arc::new(new_rms.clone()));

        // Re-wire dependency edges: the new RmsNorm node consumes whatever the
        // old one consumed (input, residual?, weight, bias?).
        let mut deps = vec![new_rms.input];
        if let Some(residual) = new_rms.residual {
            deps.push(residual);
        }
        deps.push(new_rms.weight);
        if let Some(bias) = new_rms.bias {
            deps.push(bias);
        }
        for &dep in &deps {
            if let Some(idx) = self.get_input_node_in_exec_graph(dep)
                && self.execution_graph.find_edge(idx, node_idx).is_none()
            {
                self.execution_graph.add_edge(idx, node_idx, ());
            }
        }
        if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
            self.execution_graph.remove_edge(edge);
        }
        self.add_physical_dependencies(graph, node_idx, &deps);
        self.remove_node_if_dead(input_exec_idx);
        true
    }
}
