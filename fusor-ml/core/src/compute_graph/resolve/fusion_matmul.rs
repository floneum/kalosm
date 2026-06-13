use super::*;

impl Resolver {
    pub(super) fn try_fuse_into_matmul(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        allow_qmatmul_elementwise_fusion: bool,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        // Post-op: fuse elementwise after matmul (dense or quantized).
        if let Some(el_op) = Self::try_get_unary_chain(&node_variant) {
            let input_inner = el_op.value;
            if !self.check_cached(graph, input_inner)
                && let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner)
            {
                let input_variant = self.execution_graph[input_exec_idx].variant.clone();
                // An un-flattened operand was chosen for the coop kernel,
                // which hosts no element-wise chains: fusing one here would
                // demote the matmul to the generic divmod-per-load reduce.
                if let ExecutionVariant::MatMul(matmul_op) = input_variant
                    && matmul_op.a.is_plain()
                    && matmul_op.b.is_plain()
                {
                    let mut new_matmul = matmul_op.clone();
                    let mut existing_post = new_matmul.post_element_wise.functions.clone();
                    existing_post.extend(el_op.functions.functions.iter().cloned());
                    new_matmul.post_element_wise = UnaryFunctionChain::new(
                        existing_post,
                        matmul_op.post_element_wise.input_datatype(),
                    );

                    self.execution_graph[node_idx].variant =
                        ExecutionVariant::MatMul(new_matmul.clone());

                    let (first_inner, second_inner) = (matmul_op.first, matmul_op.second);
                    if let Some(idx) = self.get_input_node_in_exec_graph(first_inner) {
                        self.execution_graph.add_edge(idx, node_idx, ());
                    }
                    if let Some(idx) = self.get_input_node_in_exec_graph(second_inner) {
                        self.execution_graph.add_edge(idx, node_idx, ());
                    }
                    if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    self.add_physical_dependencies(graph, node_idx, &[first_inner, second_inner]);
                    self.remove_node_if_dead(input_exec_idx);
                    return true;
                }
            }
        }

        // Post-op (QMatMul): fuse a general element-wise expression after
        // qmatmul. This handles composite expressions like GELU and ordered
        // extra inputs whose layouts match the output visitation shape.
        if allow_qmatmul_elementwise_fusion
            && let ExecutionVariant::Elementwise(nary) = &node_variant
        {
            // Split/gate expressions built from `narrow` views of a qmatmul
            // output (e.g. SwiGLU's gate/up halves) reach the qmatmul through
            // MapLayout chains with distinct last-dimension column offsets.
            // Absorb them into the accumulator-offset post epilogue before the
            // per-input scan below.
            if self.try_fuse_qmatmul_narrow_accumulators(graph, node_idx, nary) {
                return true;
            }
            for (candidate_input_idx, &input_inner) in nary.inputs.iter().enumerate() {
                if self.get_input_node_in_exec_graph(input_inner).is_none() {
                    continue;
                }
                let (qmatmul_inner, map_chain) = self.walk_view_chain(input_inner);
                let Some(qmatmul_exec_idx) = self.get_input_node_in_exec_graph(qmatmul_inner)
                else {
                    continue;
                };
                let ExecutionVariant::QMatMul(qmatmul_op) =
                    self.execution_graph[qmatmul_exec_idx].variant.clone()
                else {
                    continue;
                };
                if map_chain.is_none()
                    && !self.check_cached(graph, input_inner)
                    && qmatmul_op.post_element_wise_expr.is_none()
                    && qmatmul_op.in_shape[..qmatmul_op.in_shape.len() - 1]
                        .iter()
                        .product::<usize>()
                        == 1
                    && let Some((expression, accumulator_offsets, extras)) = self
                        .try_extract_indexed_qmatmul_post_expr(
                            graph,
                            nary,
                            candidate_input_idx,
                            &qmatmul_op.out_shape,
                        )
                {
                    let Some(input_datatype) = nary
                        .expression
                        .elementwise_input_datatype(candidate_input_idx)
                    else {
                        continue;
                    };
                    if input_datatype != crate::DataTypeEnum::F32
                        || nary.output_datatype != crate::DataTypeEnum::F32
                    {
                        continue;
                    }
                    if !qmatmul_op.supports_indexed_post_accumulator_offsets(
                        &graph.device(),
                        &nary.shape,
                        &accumulator_offsets,
                    ) {
                        continue;
                    }

                    let post_element_wise_expr = ElementwiseEpilogue {
                        expression,
                        extras: extras.clone(),
                        input_datatype,
                        output_datatype: nary.output_datatype,
                    };

                    let mut new_q = qmatmul_op.clone();
                    new_q.out_shape = nary.shape.clone();
                    new_q.post_element_wise_expr = Some(post_element_wise_expr);
                    new_q.post_accumulator_offsets = accumulator_offsets.into_boxed_slice();

                    if !new_q.fits_binding_budget(&graph.device()) {
                        continue;
                    }

                    self.commit_qmatmul_post_fusion(graph, node_idx, &nary.inputs, new_q);
                    return true;
                }
                let Some(mapped_layout) =
                    Self::apply_view_chain(&Layout::contiguous(&qmatmul_op.out_shape), &map_chain)
                else {
                    continue;
                };
                if mapped_layout != Layout::contiguous(&nary.shape) {
                    continue;
                }
                if !nary.expression.uses_input(candidate_input_idx)
                    || nary
                        .expression
                        .uses_custom_indexing_for_input(candidate_input_idx)
                {
                    continue;
                };
                let Some(input_datatype) = nary
                    .expression
                    .elementwise_input_datatype(candidate_input_idx)
                else {
                    continue;
                };
                let mut extras = Vec::new();
                let mut replacements = vec![None; nary.inputs.len()];
                let mut valid_expression = true;
                for (input_idx, &nary_input) in nary.inputs.iter().enumerate() {
                    let (base_inner, chain) = self.walk_view_chain(nary_input);
                    let base_qmatmul =
                        self.get_input_node_in_exec_graph(base_inner)
                            .and_then(|exec| match &self.execution_graph[exec].variant {
                                ExecutionVariant::QMatMul(op) => Some(op.clone()),
                                _ => None,
                            });
                    if let Some(base_qmatmul) = base_qmatmul
                        && Self::qmatmul_same_base(&qmatmul_op, &base_qmatmul)
                    {
                        let alias_layout = Self::apply_view_chain(
                            &Layout::contiguous(&base_qmatmul.out_shape),
                            &chain,
                        );
                        if alias_layout == Some(Layout::contiguous(&nary.shape))
                            && !nary.expression.uses_custom_indexing_for_input(input_idx)
                        {
                            replacements[input_idx] = Self::qmatmul_output_expr(
                                &base_qmatmul,
                                &mut extras,
                                nary.shape.len(),
                            );
                            continue;
                        }
                        valid_expression = false;
                        break;
                    }

                    let Some(extra) =
                        self.try_normalize_qmatmul_post_extra(graph, nary_input, &nary.shape)
                    else {
                        valid_expression = false;
                        break;
                    };
                    replacements[input_idx] =
                        Some(NaryExpr::input(extras.len() + 1, nary.shape.len()));
                    extras.push(extra);
                }
                if !valid_expression {
                    continue;
                }
                let Some(expression) =
                    Self::replace_inputs_in_expr(&nary.expression, &replacements)
                else {
                    continue;
                };
                if self.check_cached(graph, input_inner)
                    || input_datatype != crate::DataTypeEnum::F32
                    || nary.output_datatype != crate::DataTypeEnum::F32
                    || !qmatmul_op.supports_elementwise_epilogue_fusion(&graph.device())
                {
                    continue;
                }

                let post_element_wise_expr = ElementwiseEpilogue {
                    expression,
                    extras: extras.clone(),
                    input_datatype: qmatmul_op
                        .post_element_wise_expr
                        .as_ref()
                        .map(|existing| existing.input_datatype)
                        .unwrap_or(input_datatype),
                    output_datatype: nary.output_datatype,
                };

                let mut new_q = qmatmul_op.clone();
                new_q.post_element_wise_expr = Some(post_element_wise_expr);

                if !new_q.fits_binding_budget(&graph.device()) {
                    continue;
                }

                self.commit_qmatmul_post_fusion(graph, node_idx, &nary.inputs, new_q);
                return true;
            }
        }

        // Pre-op (QMatMul): fuse a general element-wise expression upstream
        // of a single-row qmatmul input. For batched/tiled qmatmul, the
        // transformed activation tile is reloaded for each output-column
        // tile, so expensive expressions like GELU would be recomputed many
        // times. Keep those chains materialized once instead.
        if allow_qmatmul_elementwise_fusion
            && let ExecutionVariant::QMatMul(qmatmul_op) = &node_variant
            && qmatmul_op.in_shape[..qmatmul_op.in_shape.len() - 1]
                .iter()
                .product::<usize>()
                == 1
            && qmatmul_op.supports_elementwise_epilogue_fusion(&graph.device())
            && !self.check_cached(graph, qmatmul_op.input)
            && let Some(input_exec) = self.get_input_node_in_exec_graph(qmatmul_op.input)
        {
            let (nary_inner, nary_map_chain) = self.walk_view_chain(qmatmul_op.input);
            let Some(nary_exec) = self.get_input_node_in_exec_graph(nary_inner) else {
                return false;
            };
            let ExecutionVariant::Elementwise(nary) =
                self.execution_graph[nary_exec].variant.clone()
            else {
                return false;
            };
            let mapped_layout =
                Self::apply_view_chain(&Layout::contiguous(&nary.shape), &nary_map_chain);
            if mapped_layout != Some(Layout::contiguous(&qmatmul_op.in_shape)) {
                return false;
            }

            for (candidate_input_idx, &primary_input) in nary.inputs.iter().enumerate() {
                if !nary.expression.uses_input(candidate_input_idx)
                    || nary
                        .expression
                        .uses_custom_indexing_for_input(candidate_input_idx)
                {
                    continue;
                }
                let Some(input_datatype) = nary
                    .expression
                    .elementwise_input_datatype(candidate_input_idx)
                else {
                    continue;
                };
                if input_datatype != crate::DataTypeEnum::F32
                    || nary.output_datatype != crate::DataTypeEnum::F32
                {
                    continue;
                }

                let (primary_inner, primary_chain) = self.walk_view_chain(primary_input);
                let Some(primary_info) = self.infer_layout_cached(graph, primary_inner) else {
                    continue;
                };
                let Some(primary_layout) =
                    Self::apply_view_chain(primary_info.layout(), &primary_chain)
                else {
                    continue;
                };
                if primary_layout != Layout::contiguous(&nary.shape) {
                    continue;
                }

                let mut mapping = vec![usize::MAX; nary.inputs.len()];
                let mut extras = Vec::new();
                let mut valid_expression = true;
                for (input_idx, &nary_input) in nary.inputs.iter().enumerate() {
                    let (base_inner, chain) = self.walk_view_chain(nary_input);
                    if base_inner == primary_inner {
                        let alias_layout = Self::apply_view_chain(primary_info.layout(), &chain);
                        if alias_layout == Some(Layout::contiguous(&nary.shape))
                            && !nary.expression.uses_custom_indexing_for_input(input_idx)
                        {
                            mapping[input_idx] = 0;
                            continue;
                        }
                        valid_expression = false;
                        break;
                    }

                    let Some(extra) =
                        self.try_normalize_qmatmul_post_extra(graph, nary_input, &nary.shape)
                    else {
                        valid_expression = false;
                        break;
                    };
                    mapping[input_idx] = extras.len() + 1;
                    extras.push(extra);
                }
                if !valid_expression {
                    continue;
                }
                let expression = nary.expression.remap_inputs(&mapping);

                let pre_element_wise_expr =
                    if let Some(existing) = &qmatmul_op.pre_element_wise_expr {
                        if existing.input_datatype != nary.output_datatype {
                            continue;
                        }
                        let mut mapping = Vec::with_capacity(1 + existing.extras.len());
                        mapping.push(0);
                        mapping.extend((0..existing.extras.len()).map(|i| i + 1 + extras.len()));
                        let shifted_existing = existing.expression.remap_inputs(&mapping);
                        let (expression, success) =
                            Self::substitute_input_in_expr(&shifted_existing, 0, &expression);
                        if !success {
                            continue;
                        }
                        let mut combined_extras = extras.clone();
                        combined_extras.extend(existing.extras.clone());
                        ElementwiseEpilogue {
                            expression,
                            extras: combined_extras,
                            input_datatype,
                            output_datatype: existing.output_datatype,
                        }
                    } else {
                        ElementwiseEpilogue {
                            expression,
                            extras: extras.clone(),
                            input_datatype,
                            output_datatype: nary.output_datatype,
                        }
                    };

                let mut new_q = qmatmul_op.clone();
                let deps_extras = pre_element_wise_expr.extras.clone();
                new_q.input = primary_inner;
                new_q.pre_element_wise_expr = Some(pre_element_wise_expr);

                if !new_q.fits_binding_budget(&graph.device()) {
                    continue;
                }

                if let Some(edge) = self.execution_graph.find_edge(input_exec, node_idx) {
                    self.execution_graph.remove_edge(edge);
                }
                if let Some(new) = self.get_input_node_in_exec_graph(new_q.input) {
                    self.execution_graph.add_edge(new, node_idx, ());
                }
                for extra in &deps_extras {
                    if let Some(idx) = self.get_input_node_in_exec_graph(*extra)
                        && self.execution_graph.find_edge(idx, node_idx).is_none()
                    {
                        self.execution_graph.add_edge(idx, node_idx, ());
                    }
                }
                self.execution_graph[node_idx].variant = ExecutionVariant::QMatMul(new_q.clone());
                self.remove_node_if_dead(input_exec);
                let mut deps = vec![new_q.input];
                deps.extend(deps_extras);
                self.add_physical_dependencies(graph, node_idx, &deps);
                return true;
            }
        }

        // Pre-op: fuse elementwise before matmul inputs. Skipped for
        // un-flattened operands: pre chains would demote the matmul off the
        // coop kernel they were chosen for.
        if let ExecutionVariant::MatMul(matmul_op) = &node_variant
            && matmul_op.a.is_plain()
            && matmul_op.b.is_plain()
        {
            let mut new_matmul = matmul_op.clone();
            let mut changed = false;

            // Check first input
            if !self.check_cached(graph, matmul_op.first)
                && let Some(first_exec) = self.get_input_node_in_exec_graph(matmul_op.first)
                && let Some(el_op) =
                    Self::try_get_unary_chain(&self.execution_graph[first_exec].variant)
            {
                new_matmul.first = el_op.value;
                let mut functions = el_op.functions.functions.clone();
                functions.extend(new_matmul.pre_element_wise[0].functions.iter().cloned());
                new_matmul.pre_element_wise[0] =
                    UnaryFunctionChain::new(functions, el_op.functions.input_datatype());
                changed = true;
            }

            // Check second input
            if !self.check_cached(graph, matmul_op.second)
                && let Some(second_exec) = self.get_input_node_in_exec_graph(matmul_op.second)
                && let Some(el_op) =
                    Self::try_get_unary_chain(&self.execution_graph[second_exec].variant)
            {
                new_matmul.second = el_op.value;
                let mut functions = el_op.functions.functions.clone();
                functions.extend(new_matmul.pre_element_wise[1].functions.iter().cloned());
                new_matmul.pre_element_wise[1] =
                    UnaryFunctionChain::new(functions, el_op.functions.input_datatype());
                changed = true;
            }

            if changed {
                self.execution_graph[node_idx].variant =
                    ExecutionVariant::MatMul(new_matmul.clone());

                if new_matmul.first != matmul_op.first {
                    let old = self.get_input_node_in_exec_graph(matmul_op.first).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new) = self.get_input_node_in_exec_graph(new_matmul.first) {
                        self.execution_graph.add_edge(new, node_idx, ());
                    }
                    self.remove_node_if_dead(old);
                }
                if new_matmul.second != matmul_op.second {
                    let old = self.get_input_node_in_exec_graph(matmul_op.second).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new) = self.get_input_node_in_exec_graph(new_matmul.second) {
                        self.execution_graph.add_edge(new, node_idx, ());
                    }
                    self.remove_node_if_dead(old);
                }
                self.add_physical_dependencies(
                    graph,
                    node_idx,
                    &[new_matmul.first, new_matmul.second],
                );
                return true;
            }
        }

        false
    }

    fn qmatmul_dependencies(qmatmul: &QMatMulOperation) -> Vec<NodeIndex> {
        let mut deps = vec![qmatmul.input];
        if let Some(pre) = &qmatmul.pre_element_wise_expr {
            deps.extend(pre.extras.iter().copied());
        }
        if let Some(post) = &qmatmul.post_element_wise_expr {
            deps.extend(post.extras.iter().copied());
        }
        deps
    }

    /// Replace the n-ary node at `node_idx` with a fused qmatmul `new_q`,
    /// rewiring the execution-graph edges: drop the edges from the original
    /// n-ary inputs that the fused operation no longer reads, add edges from
    /// every dependency (activation input + epilogue extras), and prune any
    /// inputs that became dead. Shared by every qmatmul post-epilogue path.
    fn commit_qmatmul_post_fusion(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        nary_inputs: &[NodeIndex],
        new_q: Box<QMatMulOperation>,
    ) {
        let deps = Self::qmatmul_dependencies(&new_q);
        self.execution_graph[node_idx].variant = ExecutionVariant::QMatMul(new_q);

        for input in nary_inputs {
            if deps.contains(input) {
                continue;
            }
            if let Some(input_exec) = self.get_input_node_in_exec_graph(*input)
                && let Some(edge) = self.execution_graph.find_edge(input_exec, node_idx)
            {
                self.execution_graph.remove_edge(edge);
            }
        }
        for dep in &deps {
            if let Some(idx) = self.get_input_node_in_exec_graph(*dep)
                && self.execution_graph.find_edge(idx, node_idx).is_none()
            {
                self.execution_graph.add_edge(idx, node_idx, ());
            }
        }
        self.add_physical_dependencies(graph, node_idx, &deps);
        for input in nary_inputs {
            if deps.contains(input) {
                continue;
            }
            if let Some(input_exec) = self.get_input_node_in_exec_graph(*input) {
                self.remove_node_if_dead(input_exec);
            }
        }
    }

    /// Absorb a split/gate n-ary whose inputs are `narrow` (MapLayout) views of
    /// a single-row qmatmul output into that qmatmul's accumulator-offset post
    /// epilogue. Each distinct last-dimension column offset (e.g. the gate half
    /// at 0 and the up half at `pair_len`) becomes one accumulator value, so a
    /// SwiGLU-style `silu(gate) * up` resolves to a single dynamic qmatmul
    /// kernel where the backend supports it. Returns `false` (leaving the nodes
    /// untouched) when the pattern, dtype, layout, accumulator offsets, or
    /// binding budget are unsupported.
    fn try_fuse_qmatmul_narrow_accumulators(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        nary: &ElementwiseOperation,
    ) -> bool {
        if nary.output_datatype != crate::DataTypeEnum::F32 {
            return false;
        }

        // Find the qmatmul reached through a narrow MapLayout view. A direct
        // (chain-less) reference is the indexed-input form handled below.
        let mut base = None;
        for &input in &nary.inputs {
            let (base_inner, chain) = self.walk_view_chain(input);
            if chain.is_none() {
                continue;
            }
            let Some(exec) = self.get_input_node_in_exec_graph(base_inner) else {
                continue;
            };
            if let ExecutionVariant::QMatMul(op) = &self.execution_graph[exec].variant {
                // A qmatmul that already carries a post epilogue isn't a clean
                // accumulator-offset base; leave it to the general scan.
                if op.post_element_wise_expr.is_some() {
                    continue;
                }
                base = Some((base_inner, op.clone()));
                break;
            }
        }
        let Some((qmatmul_inner, qmatmul_op)) = base else {
            return false;
        };
        if self.check_cached(graph, qmatmul_inner) {
            return false;
        }

        let Some((expression, accumulator_offsets, extras)) = self
            .try_extract_mapped_qmatmul_post_expr(
                graph,
                nary,
                qmatmul_inner,
                &qmatmul_op.out_shape,
            )
        else {
            return false;
        };

        if !qmatmul_op.supports_indexed_post_accumulator_offsets(
            &graph.device(),
            &nary.shape,
            &accumulator_offsets,
        ) {
            return false;
        }

        let post_element_wise_expr = ElementwiseEpilogue {
            expression,
            extras,
            input_datatype: crate::DataTypeEnum::F32,
            output_datatype: nary.output_datatype,
        };

        let mut new_q = qmatmul_op;
        new_q.out_shape = nary.shape.clone();
        new_q.post_element_wise_expr = Some(post_element_wise_expr);
        new_q.post_accumulator_offsets = accumulator_offsets.into_boxed_slice();

        if !new_q.fits_binding_budget(&graph.device()) {
            return false;
        }

        self.commit_qmatmul_post_fusion(graph, node_idx, &nary.inputs, new_q);
        true
    }

    /// Build the post epilogue expression, accumulator column offsets, and
    /// extra-tensor dependencies for an n-ary whose inputs are last-dimension
    /// `narrow` views of `qmatmul_inner`. Inputs that view the qmatmul become
    /// accumulator values (indices `0..offsets.len()`, deduplicated by column
    /// offset); every other input becomes a normalized extra tensor (indices
    /// after the accumulators). Returns `None` when an input isn't a clean
    /// last-dimension narrow, uses custom indexing, or can't be normalized.
    fn try_extract_mapped_qmatmul_post_expr(
        &mut self,
        graph: &mut ComputeGraphInner,
        nary: &ElementwiseOperation,
        qmatmul_inner: NodeIndex,
        qmatmul_out_shape: &[usize],
    ) -> Option<(NaryExpr, Vec<u32>, Vec<NodeIndex>)> {
        if nary.shape.len() != qmatmul_out_shape.len() {
            return None;
        }
        // The accumulator-offset epilogue is only lowered by the single-row
        // qgemv path, so every leading dimension must collapse to one row.
        if qmatmul_out_shape[..qmatmul_out_shape.len() - 1]
            .iter()
            .product::<usize>()
            != 1
        {
            return None;
        }
        let output_cols = nary.shape.last().copied()? as u32;
        let matrix_cols = qmatmul_out_shape.last().copied()? as u32;
        // A full-width (or wider) output isn't a split; the general scan owns
        // that case.
        if output_cols >= matrix_cols {
            return None;
        }

        let qmatmul_out_layout = Layout::contiguous(qmatmul_out_shape);
        let rank = nary.shape.len();

        enum MappedInput {
            Accumulator(usize),
            Extra(usize),
        }

        let mut accumulator_offsets = Vec::new();
        let mut accumulator_map = FxHashMap::default();
        let mut extras = Vec::new();
        let mut mapped = Vec::with_capacity(nary.inputs.len());
        for (input_idx, &nary_input) in nary.inputs.iter().enumerate() {
            if !nary.expression.uses_input(input_idx) {
                mapped.push(None);
                continue;
            }
            if nary.expression.uses_custom_indexing_for_input(input_idx) {
                return None;
            }
            let (base_inner, chain) = self.walk_view_chain(nary_input);
            if base_inner == qmatmul_inner {
                let view = Self::apply_view_chain(&qmatmul_out_layout, &chain)?;
                let offset = Self::qmatmul_last_dim_view_offset(&view, &nary.shape, matrix_cols)?;
                let value_idx = *accumulator_map.entry(offset).or_insert_with(|| {
                    let idx = accumulator_offsets.len();
                    accumulator_offsets.push(offset);
                    idx
                });
                mapped.push(Some(MappedInput::Accumulator(value_idx)));
            } else {
                let extra =
                    self.try_normalize_qmatmul_post_extra(graph, nary_input, &nary.shape)?;
                let pos = extras.len();
                extras.push(extra);
                mapped.push(Some(MappedInput::Extra(pos)));
            }
        }

        // Two distinct column offsets are the smallest split worth folding into
        // the accumulator-offset path; a single offset is either the default
        // full-width store or a partial column the qgemv path can't cover.
        if accumulator_offsets.len() < 2 {
            return None;
        }

        let accumulator_count = accumulator_offsets.len();
        let mut replacements = vec![None; nary.inputs.len()];
        for (input_idx, kind) in mapped.into_iter().enumerate() {
            match kind {
                Some(MappedInput::Accumulator(value_idx)) => {
                    replacements[input_idx] = Some(NaryExpr::input(value_idx, rank));
                }
                Some(MappedInput::Extra(pos)) => {
                    replacements[input_idx] = Some(NaryExpr::input(accumulator_count + pos, rank));
                }
                None => {}
            }
        }

        let expression = Self::replace_inputs_in_expr(&nary.expression, &replacements)?;
        Some((expression, accumulator_offsets, extras))
    }

    /// If `view` is a contiguous last-dimension narrow of a single-row qmatmul
    /// output whose shape matches `output_shape`, return its column offset.
    /// Returns `None` for any non-narrow / strided / out-of-range view.
    fn qmatmul_last_dim_view_offset(
        view: &Layout,
        output_shape: &[usize],
        matrix_cols: u32,
    ) -> Option<u32> {
        if view.shape() != output_shape {
            return None;
        }
        if view.strides().last().copied() != Some(1) {
            return None;
        }
        let offset = u32::try_from(view.offset()).ok()?;
        let output_cols = *output_shape.last()? as u32;
        if offset.checked_add(output_cols)? > matrix_cols {
            return None;
        }
        Some(offset)
    }

    fn try_extract_indexed_qmatmul_post_expr(
        &mut self,
        graph: &mut ComputeGraphInner,
        nary: &ElementwiseOperation,
        qmatmul_input_idx: usize,
        qmatmul_out_shape: &[usize],
    ) -> Option<(NaryExpr, Vec<u32>, Vec<NodeIndex>)> {
        if nary.output_datatype != crate::DataTypeEnum::F32
            || nary.shape.len() != qmatmul_out_shape.len()
            || nary.shape.as_ref() == qmatmul_out_shape
        {
            return None;
        }
        let output_cols = nary.shape.last().copied()? as u32;
        let matrix_cols = qmatmul_out_shape.last().copied()? as u32;
        if output_cols >= matrix_cols {
            return None;
        }

        let temp_input_base = nary.inputs.len();
        let mut accumulator_offsets = Vec::new();
        let mut accumulator_map = FxHashMap::default();
        let expression = Self::replace_indexed_qmatmul_accumulators(
            &nary.expression,
            qmatmul_input_idx,
            nary.shape.len(),
            output_cols,
            matrix_cols,
            temp_input_base,
            &mut accumulator_offsets,
            &mut accumulator_map,
        )?;
        if accumulator_offsets.len() < 2 {
            return None;
        }

        let mut replacements = vec![None; nary.inputs.len()];
        let mut extras = Vec::new();
        for (input_idx, &input) in nary.inputs.iter().enumerate() {
            if input_idx == qmatmul_input_idx || !nary.expression.uses_input(input_idx) {
                continue;
            }
            if nary.expression.uses_custom_indexing_for_input(input_idx) {
                return None;
            }
            let extra = self.try_normalize_qmatmul_post_extra(graph, input, &nary.shape)?;
            replacements[input_idx] = Some(NaryExpr::input(
                accumulator_offsets.len() + extras.len(),
                nary.shape.len(),
            ));
            extras.push(extra);
        }

        let expression = Self::replace_inputs_in_expr(&expression, &replacements)?;
        let expression = Self::remap_temp_accumulator_inputs(
            &expression,
            temp_input_base,
            accumulator_offsets.len(),
        );
        Some((expression, accumulator_offsets, extras))
    }

    #[allow(clippy::too_many_arguments)]
    fn replace_indexed_qmatmul_accumulators(
        expr: &NaryExpr,
        qmatmul_input_idx: usize,
        output_rank: usize,
        output_cols: u32,
        matrix_cols: u32,
        temp_input_base: usize,
        accumulator_offsets: &mut Vec<u32>,
        accumulator_map: &mut FxHashMap<u32, usize>,
    ) -> Option<NaryExpr> {
        match expr {
            NaryExpr::Op { children, function } => Some(NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| {
                        Self::replace_indexed_qmatmul_accumulators(
                            child,
                            qmatmul_input_idx,
                            output_rank,
                            output_cols,
                            matrix_cols,
                            temp_input_base,
                            accumulator_offsets,
                            accumulator_map,
                        )
                    })
                    .collect::<Option<Vec<_>>>()?,
                function: function.clone(),
            }),
            NaryExpr::IndexedInput { input_idx, indices } if *input_idx == qmatmul_input_idx => {
                let offset = Self::extract_qmatmul_last_dim_offset(indices, output_rank)?;
                if output_cols
                    .checked_add(offset)
                    .is_none_or(|cols| cols > matrix_cols)
                {
                    return None;
                }
                let value_idx = if let Some(value_idx) = accumulator_map.get(&offset) {
                    *value_idx
                } else {
                    let value_idx = accumulator_offsets.len();
                    accumulator_offsets.push(offset);
                    accumulator_map.insert(offset, value_idx);
                    value_idx
                };
                Some(NaryExpr::input(temp_input_base + value_idx, output_rank))
            }
            NaryExpr::IndexedInput { input_idx, indices } => Some(NaryExpr::IndexedInput {
                input_idx: *input_idx,
                indices: indices
                    .iter()
                    .map(|index| {
                        Self::replace_indexed_qmatmul_accumulators(
                            index,
                            qmatmul_input_idx,
                            output_rank,
                            output_cols,
                            matrix_cols,
                            temp_input_base,
                            accumulator_offsets,
                            accumulator_map,
                        )
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            NaryExpr::DimIndex(dim) => Some(NaryExpr::DimIndex(*dim)),
            NaryExpr::Scalar(value) => Some(NaryExpr::Scalar(*value)),
        }
    }

    fn extract_qmatmul_last_dim_offset(indices: &[NaryExpr], output_rank: usize) -> Option<u32> {
        if indices.len() != output_rank {
            return None;
        }
        for (dim, index) in indices[..output_rank - 1].iter().enumerate() {
            if !matches!(index, NaryExpr::DimIndex(index_dim) if *index_dim == dim) {
                return None;
            }
        }
        Self::extract_dim_plus_u32_offset(&indices[output_rank - 1], output_rank - 1)
    }

    fn extract_dim_plus_u32_offset(expr: &NaryExpr, dim: usize) -> Option<u32> {
        match expr {
            NaryExpr::DimIndex(index_dim) if *index_dim == dim => Some(0),
            NaryExpr::Op { children, function }
                if function.op == NaryOp::Add && children.len() == 2 =>
            {
                Self::extract_dim_plus_u32_offset_pair(&children[0], &children[1], dim).or_else(
                    || Self::extract_dim_plus_u32_offset_pair(&children[1], &children[0], dim),
                )
            }
            NaryExpr::Op { children, function }
                if matches!(function.op, NaryOp::AddConst(NaryScalar::U32(_)))
                    && children.len() == 1 =>
            {
                let NaryOp::AddConst(NaryScalar::U32(offset)) = function.op else {
                    unreachable!();
                };
                matches!(&children[0], NaryExpr::DimIndex(index_dim) if *index_dim == dim)
                    .then_some(offset)
            }
            _ => None,
        }
    }

    fn extract_dim_plus_u32_offset_pair(
        dim_expr: &NaryExpr,
        offset_expr: &NaryExpr,
        dim: usize,
    ) -> Option<u32> {
        let NaryExpr::DimIndex(index_dim) = dim_expr else {
            return None;
        };
        if *index_dim != dim {
            return None;
        }
        let NaryExpr::Scalar(NaryScalar::U32(offset)) = offset_expr else {
            return None;
        };
        Some(*offset)
    }

    fn remap_temp_accumulator_inputs(
        expr: &NaryExpr,
        temp_input_base: usize,
        accumulator_count: usize,
    ) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| {
                        Self::remap_temp_accumulator_inputs(
                            child,
                            temp_input_base,
                            accumulator_count,
                        )
                    })
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => {
                let input_idx =
                    if (temp_input_base..temp_input_base + accumulator_count).contains(input_idx) {
                        input_idx - temp_input_base
                    } else {
                        *input_idx
                    };
                NaryExpr::IndexedInput {
                    input_idx,
                    indices: indices
                        .iter()
                        .map(|index| {
                            Self::remap_temp_accumulator_inputs(
                                index,
                                temp_input_base,
                                accumulator_count,
                            )
                        })
                        .collect(),
                }
            }
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        }
    }
}
