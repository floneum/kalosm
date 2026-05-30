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
                if let ComputeGraphNodeVariant::MatMul(matmul_op) = input_variant {
                    let mut new_matmul = matmul_op.clone();
                    let mut existing_post = new_matmul.post_element_wise.functions.clone();
                    existing_post.extend(el_op.functions.functions.iter().cloned());
                    new_matmul.post_element_wise = UnaryFunctionChain::new(
                        existing_post,
                        matmul_op.post_element_wise.input_datatype(),
                    );

                    self.execution_graph[node_idx].variant =
                        ComputeGraphNodeVariant::MatMul(new_matmul.clone());

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
            && let ComputeGraphNodeVariant::Nary(nary) = &node_variant
            && nary.inputs.len() <= 4
        {
            for (candidate_input_idx, &input_inner) in nary.inputs.iter().enumerate() {
                if self.get_input_node_in_exec_graph(input_inner).is_none() {
                    continue;
                }
                let (qmatmul_inner, map_chain) = self
                    .walk_map_layout_chain(input_inner)
                    .unwrap_or((input_inner, Vec::new()));
                let Some(qmatmul_exec_idx) = self.get_input_node_in_exec_graph(qmatmul_inner)
                else {
                    continue;
                };
                let ComputeGraphNodeVariant::QMatMul(qmatmul_op) =
                    self.execution_graph[qmatmul_exec_idx].variant.clone()
                else {
                    continue;
                };
                let mapped_layout = Self::apply_map_layout_chain(
                    &Layout::contiguous(&qmatmul_op.out_shape),
                    &map_chain,
                );
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
                    let (base_inner, chain) = self
                        .walk_map_layout_chain(nary_input)
                        .unwrap_or((nary_input, Vec::new()));
                    let base_qmatmul =
                        self.get_input_node_in_exec_graph(base_inner)
                            .and_then(|exec| match &self.execution_graph[exec].variant {
                                ComputeGraphNodeVariant::QMatMul(op) => Some(op.clone()),
                                _ => None,
                            });
                    if let Some(base_qmatmul) = base_qmatmul
                        && Self::qmatmul_same_base(&qmatmul_op, &base_qmatmul)
                    {
                        let alias_layout = Self::apply_map_layout_chain(
                            &Layout::contiguous(&base_qmatmul.out_shape),
                            &chain,
                        );
                        if alias_layout == Layout::contiguous(&nary.shape)
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
                let deps_extras = post_element_wise_expr.extras.clone();
                new_q.post_element_wise_expr = Some(post_element_wise_expr);

                self.execution_graph[node_idx].variant = ComputeGraphNodeVariant::QMatMul(new_q);

                let input_inner_of_qmatmul = qmatmul_op.input;
                let mut deps = vec![input_inner_of_qmatmul];
                deps.extend(deps_extras.iter().copied());
                for input in &nary.inputs {
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
                for input in &nary.inputs {
                    if deps.contains(input) {
                        continue;
                    }
                    if let Some(input_exec) = self.get_input_node_in_exec_graph(*input) {
                        self.remove_node_if_dead(input_exec);
                    }
                }
                return true;
            }
        }

        // Pre-op (QMatMul): fuse a general element-wise expression upstream
        // of a single-row qmatmul input. For batched/tiled qmatmul, the
        // transformed activation tile is reloaded for each output-column
        // tile, so expensive expressions like GELU would be recomputed many
        // times. Keep those chains materialized once instead.
        if allow_qmatmul_elementwise_fusion
            && let ComputeGraphNodeVariant::QMatMul(qmatmul_op) = &node_variant
            && qmatmul_op.in_shape[..qmatmul_op.in_shape.len() - 1]
                .iter()
                .product::<usize>()
                == 1
            && !self.check_cached(graph, qmatmul_op.input)
            && let Some(input_exec) = self.get_input_node_in_exec_graph(qmatmul_op.input)
        {
            let (nary_inner, nary_map_chain) = self
                .walk_map_layout_chain(qmatmul_op.input)
                .unwrap_or((qmatmul_op.input, Vec::new()));
            let Some(nary_exec) = self.get_input_node_in_exec_graph(nary_inner) else {
                return false;
            };
            let ComputeGraphNodeVariant::Nary(nary) =
                self.execution_graph[nary_exec].variant.clone()
            else {
                return false;
            };
            if nary.inputs.len() > 4 {
                return false;
            }
            let mapped_layout =
                Self::apply_map_layout_chain(&Layout::contiguous(&nary.shape), &nary_map_chain);
            if mapped_layout != Layout::contiguous(&qmatmul_op.in_shape) {
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

                let (primary_inner, primary_chain) = self
                    .walk_map_layout_chain(primary_input)
                    .unwrap_or((primary_input, Vec::new()));
                let Some(primary_info) = self.infer_layout_cached(graph, primary_inner) else {
                    continue;
                };
                let primary_layout =
                    Self::apply_map_layout_chain(primary_info.layout(), &primary_chain);
                if primary_layout != Layout::contiguous(&nary.shape) {
                    continue;
                }

                let mut mapping = vec![usize::MAX; nary.inputs.len()];
                let mut extras = Vec::new();
                let mut valid_expression = true;
                for (input_idx, &nary_input) in nary.inputs.iter().enumerate() {
                    let (base_inner, chain) = self
                        .walk_map_layout_chain(nary_input)
                        .unwrap_or((nary_input, Vec::new()));
                    if base_inner == primary_inner {
                        let alias_layout =
                            Self::apply_map_layout_chain(primary_info.layout(), &chain);
                        if alias_layout == Layout::contiguous(&nary.shape)
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
                self.execution_graph[node_idx].variant =
                    ComputeGraphNodeVariant::QMatMul(new_q.clone());
                self.remove_node_if_dead(input_exec);
                let mut deps = vec![new_q.input];
                deps.extend(deps_extras);
                self.add_physical_dependencies(graph, node_idx, &deps);
                return true;
            }
        }

        // Pre-op: fuse elementwise before matmul inputs
        if let ComputeGraphNodeVariant::MatMul(matmul_op) = &node_variant {
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
                    ComputeGraphNodeVariant::MatMul(new_matmul.clone());

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
}
