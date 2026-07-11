use crate::nary_wise::NaryFunction;

use super::*;

impl Resolver {
    pub(super) fn try_fuse_naries(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        allow_indexed_inline: bool,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        let ExecutionVariant::Elementwise(nary) = node_variant else {
            return false;
        };

        // Collect all fusible nary inputs
        let mut expression = nary.expression.clone();
        let mut all_inputs = nary.inputs.clone();
        let mut fused_execs = Vec::new();

        let max_fused_inputs = graph.device().nary_direct_input_binding_budget();

        for &input_inner in nary.inputs.iter() {
            if self.check_cached(graph, input_inner) {
                continue;
            }
            // Dense branch: an externally live producer (pending sink /
            // user-held node) materializes regardless, so inlining it here
            // would duplicate its compute. Region formation fuses it with
            // its consumers instead, emitting it as a region output.
            if self.horizontal_merge_dense_ops
                && graph
                    .nodes
                    .nodes
                    .node_weight(input_inner)
                    .is_some_and(|node| node.reference_count > 0)
            {
                continue;
            }
            let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) else {
                continue;
            };
            // Check if the node still exists (it may have been removed during optimization)
            if !self.execution_graph.contains_node(input_exec) {
                continue;
            }
            // Inlining duplicates the producer's work unless this node is its
            // only consumer: the producer still materializes for everyone
            // else (e.g. the residual stream feeds every later layer — fusing
            // it forward would re-sum the whole prefix per layer). A user-held
            // reference alone doesn't block fusion — only another consumer in
            // this resolve does.
            if self
                .execution_graph
                .neighbors_directed(input_exec, petgraph::Direction::Outgoing)
                .count()
                != 1
            {
                continue;
            }
            let ExecutionVariant::Elementwise(input_nary) =
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
            for slot in &target_slots {
                let (next, s) = Self::substitute_input_in_expr(&new_expression, *slot, &inlined);
                new_expression = next;
                success &= s;
            }
            // Custom-indexed reads (a folded reshape/transpose between the
            // producer and this node) fail plain substitution; on dense
            // graphs, compose the producer expression with the index list
            // instead — but only where each producer element is read once,
            // so the inlined expression is never re-evaluated.
            if !success
                && allow_indexed_inline
                && target_slots.iter().all(|&slot| {
                    super::fold_views::input_reread_factor(&expression, &nary.shape, slot) == 1
                })
            {
                let mut composed = expression.clone();
                success = true;
                for slot in &target_slots {
                    match Self::substitute_input_composed(&composed, *slot, &inlined) {
                        Some(next) => composed = next,
                        None => {
                            success = false;
                            break;
                        }
                    }
                }
                if success {
                    new_expression = composed;
                }
            }

            // Only fuse if substitution was successful
            // If not, the expression still references the original input which must remain
            if success {
                // Count unique inputs after potential merge (duplicates share a binding).
                let unique_inputs: FxHashSet<_> = all_inputs
                    .iter()
                    .chain(input_nary.inputs.iter())
                    .copied()
                    .collect();

                if unique_inputs.len() > max_fused_inputs {
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

        let new_nary = ElementwiseOperation {
            inputs: final_inputs.clone(),
            expression: final_expression,
            shape: nary.shape.clone(),
            output_datatype: nary.output_datatype,
        };

        self.execution_graph[node_idx].variant = ExecutionVariant::Elementwise(new_nary.clone());

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

    /// Substitute every read of input `target_idx` — elementwise *or*
    /// custom-indexed — with `replacement` evaluated at the read's index
    /// expressions. Where [`Self::substitute_input_in_expr`] declines
    /// custom-indexed reads unless the replacement is a bare input, this
    /// composes the replacement expression with the index list instead:
    /// `input_t[i0, i1]` becomes `replacement` with `DimIndex(d)` rewritten
    /// to `i_d`. Returns `None` when the composition is impossible (an index
    /// list shorter than the replacement's rank).
    pub(super) fn substitute_input_composed(
        expr: &NaryExpr,
        target_idx: usize,
        replacement: &NaryExpr,
    ) -> Option<NaryExpr> {
        Some(match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| Self::substitute_input_composed(child, target_idx, replacement))
                    .collect::<Option<Vec<_>>>()?,
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => {
                let indices: Vec<NaryExpr> = indices
                    .iter()
                    .map(|index| Self::substitute_input_composed(index, target_idx, replacement))
                    .collect::<Option<Vec<_>>>()?;
                if *input_idx == target_idx {
                    Self::compose_expr_with_indices(replacement, &indices)?
                } else {
                    NaryExpr::IndexedInput {
                        input_idx: *input_idx,
                        indices,
                    }
                }
            }
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        })
    }

    /// Evaluate `expr` (written in its own index space) at the coordinates
    /// given by `indices`: every `DimIndex(d)` becomes `indices[d]`. `None`
    /// when `expr` references a dimension `indices` does not provide.
    pub(super) fn compose_expr_with_indices(
        expr: &NaryExpr,
        indices: &[NaryExpr],
    ) -> Option<NaryExpr> {
        Some(match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| Self::compose_expr_with_indices(child, indices))
                    .collect::<Option<Vec<_>>>()?,
                function: function.clone(),
            },
            NaryExpr::IndexedInput {
                input_idx,
                indices: inner,
            } => NaryExpr::IndexedInput {
                input_idx: *input_idx,
                indices: inner
                    .iter()
                    .map(|index| Self::compose_expr_with_indices(index, indices))
                    .collect::<Option<Vec<_>>>()?,
            },
            NaryExpr::DimIndex(dim) => indices.get(*dim)?.clone(),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        })
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

        // Build the input-index remap, collecting only used inputs.
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
            && first.post_accumulator_offsets == second.post_accumulator_offsets
    }

    pub(super) fn qmatmul_output_expr(
        qmatmul: &QMatMulOperation,
        extras: &mut Vec<NodeIndex>,
        rank: usize,
    ) -> Option<NaryExpr> {
        if let Some(epilogue) = &qmatmul.post_element_wise_expr {
            let value_arity = qmatmul.post_accumulator_offsets.len().max(1);
            let mut mapping = Vec::with_capacity(value_arity + epilogue.extras.len());
            mapping.extend(0..value_arity);
            mapping.extend((0..epilogue.extras.len()).map(|i| extras.len() + value_arity + i));
            extras.extend(epilogue.extras.iter().copied());
            Some(epilogue.expression.remap_inputs(&mapping))
        } else {
            Some(NaryExpr::input(0, rank))
        }
    }

    /// Try to extract a unary function chain from a node variant.
    /// Only Nary ops with a single input and element-wise access can be converted.
    pub(super) fn try_get_unary_chain(variant: &ExecutionVariant) -> Option<ExtractedUnaryChain> {
        match variant {
            ExecutionVariant::Elementwise(nary) => nary.try_extract_unary_chain(),
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
        let ExecutionVariant::Reduce(reduce_op) = input_variant else {
            return false;
        };

        let mut new_reduce = reduce_op.clone();
        let mut existing_post = new_reduce.post_element_wise.functions.clone();
        existing_post.extend(el_op.functions.functions.iter().cloned());
        new_reduce.post_element_wise =
            UnaryFunctionChain::new(existing_post, reduce_op.post_element_wise.input_datatype());

        self.execution_graph[node_idx].variant = ExecutionVariant::Reduce(new_reduce.clone());

        for &reduce_input_inner in &reduce_op.inputs {
            if let Some(reduce_input_exec) = self.get_input_node_in_exec_graph(reduce_input_inner) {
                self.execution_graph
                    .add_edge(reduce_input_exec, node_idx, ());
            }
        }

        if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
            self.execution_graph.remove_edge(edge);
        }
        self.add_physical_dependencies(graph, node_idx, &reduce_op.inputs);
        self.remove_node_if_dead(input_exec_idx);
        true
    }

    /// Collapse a reduce over a size-1 axis into an elementwise operation
    /// that folds the single element with the initial value — `f(init, x)`,
    /// exactly what the reduce lowering computes for a one-element row (the
    /// float min/max initial values are finite pseudo-identities, so the
    /// fold is kept rather than dropped to stay bitwise-equal). Backward
    /// tapes emit these from `reduce_broadcast_gradient` when a broadcast
    /// axis has extent 1 (weight gradients reshaped through a leading unit
    /// dim). As an elementwise node it then inlines into its consumers via
    /// normal n-ary fusion. Dense (QMatMul-free) graphs only.
    pub(super) fn try_collapse_unit_reduce(
        &mut self,
        _graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let ExecutionVariant::Reduce(reduce) = &self.execution_graph[node_idx].variant else {
            return false;
        };
        if reduce.shape[reduce.axis] != 1
            // Conservative dtype guard: the reduce lowering casts the folded
            // value to the accumulator dtype before the post chain; keep the
            // rewrite to the homogeneous case where no cast can occur.
            || !reduce.post_element_wise.functions.is_empty()
            || reduce.function.datatype() != reduce.out_datatype()
        {
            return false;
        }
        let reduce = reduce.clone();

        // Evaluate the producer expression at the single axis coordinate.
        let mut mapping = Vec::with_capacity(reduce.shape.len());
        let mut out_pos = 0;
        for dim in 0..reduce.shape.len() {
            if dim == reduce.axis {
                mapping.push(NaryExpr::Scalar(crate::nary_wise::NaryScalar::U32(0)));
            } else {
                mapping.push(NaryExpr::DimIndex(out_pos));
                out_pos += 1;
            }
        }
        let Some(expression) = Self::compose_expr_with_indices(&reduce.expression, &mapping) else {
            return false;
        };

        // Value-exactness: the reduce lowering seeds the accumulator with
        // `initial_value` and folds `f(init, x)` even for a single element,
        // and the float min/max identities are finite pseudo-identities
        // (±3.40282e38 / ±65504) — so `f(init, x)` is NOT `x` at the edges
        // (min of +inf clamps, sum of -0.0 is +0.0). Replay the exact fold
        // as a const elementwise op (same TileBinaryOp the reduce fold
        // uses), keeping the collapsed node bitwise-identical to the
        // unfused reduce.
        use crate::reduce::ReduceOp;
        let init = reduce.function.initial_value;
        let fold_op = match reduce.function.op {
            ReduceOp::Sum => NaryOp::AddConst(init),
            ReduceOp::Product => NaryOp::MulConst(init),
            ReduceOp::Max => NaryOp::MaxConst(init),
            ReduceOp::Min => NaryOp::MinConst(init),
        };
        let dtype = reduce.function.datatype();
        let expression = NaryExpr::Op {
            children: vec![expression],
            function: NaryFunction::unary(
                Some(format!("unit_{}", reduce.function.name())),
                fold_op,
                dtype,
                dtype,
            ),
        };

        let new_nary = ElementwiseOperation {
            inputs: reduce.inputs.clone(),
            expression,
            shape: reduce.out_shape().into(),
            output_datatype: reduce.out_datatype(),
        };
        // Dependencies are unchanged: same inputs, same edges.
        self.execution_graph[node_idx].variant = ExecutionVariant::Elementwise(new_nary);
        true
    }

    /// Rewrite `unary_chain(reduce_out[indices])` — a unary chain over a
    /// single custom-indexed read of a reduction (the shape left behind by a
    /// folded `sum_keepdim` unsqueeze or reshape view) — into a reduce of
    /// its own: the read's index expressions substitute into the reduction's
    /// row dimensions, the reduced axis becomes a fresh trailing dimension,
    /// and the chain folds into the post chain. Each output element
    /// recomputes its row independently, so correctness never depends on the
    /// index mapping being a bijection; the numel guard keeps the total fold
    /// work equal to the original reduce. Dense (QMatMul-free) graphs only.
    pub(super) fn try_fuse_unary_into_reduce_indexed(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let ExecutionVariant::Elementwise(nary) = self.execution_graph[node_idx].variant.clone()
        else {
            return false;
        };
        if nary.inputs.len() != 1 {
            return false;
        }
        let Some((functions, indices)) = Self::extract_unary_chain_indexed(&nary) else {
            return false;
        };
        let input_inner = nary.inputs[0];
        if self.check_cached(graph, input_inner) {
            return false;
        }
        let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) else {
            return false;
        };
        let ExecutionVariant::Reduce(reduce) = self.execution_graph[input_exec].variant.clone()
        else {
            return false;
        };
        // No sole-consumer requirement: when several unary chains read the
        // same reduction, rewriting each one strands the source reduce with
        // no consumers, so it dies and the dispatch count still shrinks by
        // one; the duplicated fold work is bounded by the numel guard below.
        // One row recomputed per output element: equal numel keeps the fold
        // work identical to the original reduce (a broadcast-shaped read
        // would multiply it).
        let rows: usize = reduce
            .shape
            .iter()
            .enumerate()
            .filter_map(|(dim, &size)| (dim != reduce.axis).then_some(size))
            .product();
        if nary.shape.iter().product::<usize>() != rows || indices.len() + 1 != reduce.shape.len() {
            return false;
        }
        // The appended chain must consume exactly the reduce's output dtype.
        let mut current = reduce.out_datatype();
        for function in &functions {
            if function.input_types.as_slice() != [current] {
                return false;
            }
            current = function.output_type;
        }
        if current != nary.output_datatype {
            return false;
        }

        // Remap the reduce expression into the node's index space: row dims
        // take the read's index expressions, the reduced axis becomes the
        // fresh trailing dimension.
        let node_rank = nary.shape.len();
        let mut mapping = Vec::with_capacity(reduce.shape.len());
        let mut out_pos = 0;
        for dim in 0..reduce.shape.len() {
            if dim == reduce.axis {
                mapping.push(NaryExpr::DimIndex(node_rank));
            } else {
                mapping.push(indices[out_pos].clone());
                out_pos += 1;
            }
        }
        let Some(expression) = Self::compose_expr_with_indices(&reduce.expression, &mapping) else {
            return false;
        };

        let mut shape: Vec<usize> = nary.shape.to_vec();
        shape.push(reduce.shape[reduce.axis]);
        let mut post = reduce.post_element_wise.functions.clone();
        post.extend(functions);
        let new_reduce = crate::reduce::ReduceOperation {
            inputs: reduce.inputs.clone(),
            expression,
            shape: shape.into(),
            function: reduce.function.clone(),
            post_element_wise: UnaryFunctionChain::new(
                post,
                reduce.post_element_wise.input_datatype(),
            ),
            axis: node_rank,
        };
        self.execution_graph[node_idx].variant = ExecutionVariant::Reduce(new_reduce);

        for &reduce_input in &reduce.inputs {
            if let Some(exec) = self.get_input_node_in_exec_graph(reduce_input)
                && self.execution_graph.find_edge(exec, node_idx).is_none()
            {
                self.execution_graph.add_edge(exec, node_idx, ());
            }
        }
        if let Some(edge) = self.execution_graph.find_edge(input_exec, node_idx) {
            self.execution_graph.remove_edge(edge);
        }
        self.add_physical_dependencies(graph, node_idx, &reduce.inputs);
        self.remove_node_if_dead(input_exec);
        true
    }

    /// Extract a (possibly empty) unary function chain over exactly one
    /// read of input 0 with arbitrary index expressions, innermost function
    /// first. The index expressions must not read any input themselves.
    fn extract_unary_chain_indexed(
        nary: &ElementwiseOperation,
    ) -> Option<(Vec<NaryFunction>, Vec<NaryExpr>)> {
        fn contains_input(expr: &NaryExpr) -> bool {
            match expr {
                NaryExpr::Op { children, .. } => children.iter().any(contains_input),
                NaryExpr::IndexedInput { .. } => true,
                NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => false,
            }
        }
        let mut functions = Vec::new();
        let mut expr = &nary.expression;
        loop {
            match expr {
                NaryExpr::Op { children, function }
                    if children.len() == 1 && function.input_types.len() == 1 =>
                {
                    functions.push(function.clone());
                    expr = &children[0];
                }
                NaryExpr::IndexedInput {
                    input_idx: 0,
                    indices,
                } if !indices.iter().any(contains_input) => {
                    functions.reverse();
                    return Some((functions, indices.clone()));
                }
                _ => return None,
            }
        }
    }

    /// Inline elementwise producers into a reduce's fused expression: the
    /// reduce evaluates the producer at every index-space coordinate, so a
    /// producer consumed only by this reduce never needs to materialize.
    /// Composed contractions that recognition did not claim collapse to a
    /// single map-reduce kernel here, where the tiled lowering can stage
    /// their reused inputs through workgroup memory.
    pub(super) fn try_fuse_producer_into_reduce(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        allow_indexed_inline: bool,
    ) -> bool {
        let ExecutionVariant::Reduce(reduce) = self.execution_graph[node_idx].variant.clone()
        else {
            return false;
        };

        let mut expression = reduce.expression.clone();
        let mut all_inputs = reduce.inputs.clone();
        let mut fused_execs = Vec::new();
        let max_fused_inputs = graph.device().nary_direct_input_binding_budget();

        for &input_inner in reduce.inputs.iter() {
            if self.check_cached(graph, input_inner) {
                continue;
            }
            let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) else {
                continue;
            };
            if !self.execution_graph.contains_node(input_exec) {
                continue;
            }
            // Same sole-consumer rule as nary fusion: inlining a shared
            // producer would re-evaluate it for every other consumer.
            if self
                .execution_graph
                .neighbors_directed(input_exec, petgraph::Direction::Outgoing)
                .count()
                != 1
            {
                continue;
            }
            let ExecutionVariant::Elementwise(input_nary) =
                &self.execution_graph[input_exec].variant
            else {
                continue;
            };
            // The reduce evaluates this input across the full index space;
            // a producer with any other shape reads out of range — unless
            // every read goes through custom index expressions (folded
            // views/gathers, dense graphs only), which define the mapping
            // into the producer's own index space and carry their own
            // bounds selects.
            if input_nary.shape != reduce.shape && !allow_indexed_inline {
                continue;
            }

            let target_slots: Vec<usize> = all_inputs
                .iter()
                .enumerate()
                .filter_map(|(slot, value)| (*value == input_inner).then_some(slot))
                .collect();

            let offset = all_inputs.len();
            let inlined = Self::offset_input_indices(&input_nary.expression, offset);
            let mut new_expression = expression.clone();
            let mut success = input_nary.shape == reduce.shape;
            if success {
                for slot in &target_slots {
                    let (next, s) =
                        Self::substitute_input_in_expr(&new_expression, *slot, &inlined);
                    new_expression = next;
                    success &= s;
                }
            }
            // Custom-indexed reads fail plain substitution; compose the
            // producer expression with the index list instead, where each
            // producer element is read once (the inlined expression is
            // never re-evaluated).
            if !success
                && allow_indexed_inline
                && target_slots.iter().all(|&slot| {
                    super::fold_views::input_reread_factor(&expression, &reduce.shape, slot) == 1
                })
            {
                let mut composed = expression.clone();
                success = true;
                for slot in &target_slots {
                    match Self::substitute_input_composed(&composed, *slot, &inlined) {
                        Some(next) => composed = next,
                        None => {
                            success = false;
                            break;
                        }
                    }
                }
                if success {
                    new_expression = composed;
                }
            }

            if success {
                let unique_inputs: FxHashSet<_> = all_inputs
                    .iter()
                    .chain(input_nary.inputs.iter())
                    .copied()
                    .collect();
                if unique_inputs.len() > max_fused_inputs {
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

        let (final_inputs, final_expression) = Self::deduplicate_inputs(all_inputs, expression);

        let mut new_reduce = reduce.clone();
        new_reduce.inputs = final_inputs;
        new_reduce.expression = final_expression;
        let new_inputs = new_reduce.inputs.clone();
        self.execution_graph[node_idx].variant = ExecutionVariant::Reduce(new_reduce);

        for (input_exec, producer_inputs) in fused_execs {
            if let Some(edge) = self.execution_graph.find_edge(input_exec, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
            for &new_input in &producer_inputs {
                if let Some(exec) = self.get_input_node_in_exec_graph(new_input)
                    && self.execution_graph.find_edge(exec, node_idx).is_none()
                {
                    self.execution_graph.add_edge(exec, node_idx, ());
                }
            }
            self.remove_node_if_dead(input_exec);
        }

        self.add_physical_dependencies(graph, node_idx, &new_inputs);
        true
    }
}
