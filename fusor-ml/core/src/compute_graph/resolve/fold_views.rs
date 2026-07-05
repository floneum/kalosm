use crate::view::ViewOperation;

use super::*;

impl Resolver {
    /// Fold view inputs of an n-ary node directly into its expression: each
    /// `IndexedInput` through the view becomes a load of the view's base
    /// node with the view's coordinate map applied (and a bounds-select
    /// around partially-defined stages). This removes the view node from
    /// between the producer and consumer, so the n-ary fusion passes see
    /// through layout changes instead of stopping at them.
    ///
    /// Affine maps always fold — they rewrite to plain index arithmetic.
    /// Maps that need delinearization (divmod address arithmetic, from a
    /// reshape regrouping non-mergeable strides) re-derive coordinates on
    /// every load, so they only fold where each element is loaded once; a
    /// load re-read across unindexed dims (a contraction operand) keeps the
    /// view node and materializes through the gather fallback instead.
    pub(super) fn try_fold_view_inputs(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let ExecutionVariant::Elementwise(nary) = self.execution_graph[node_idx].variant.clone()
        else {
            return false;
        };

        let mut expression = nary.expression.clone();
        let mut inputs = nary.inputs.clone();
        let mut folded = Vec::new();

        for (slot, input_inner) in nary.inputs.iter().copied().enumerate() {
            if self.check_cached(graph, input_inner) {
                continue;
            }
            let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) else {
                continue;
            };
            if !self.execution_graph.contains_node(input_exec) {
                continue;
            }
            let ExecutionVariant::View(view) = &self.execution_graph[input_exec].variant else {
                continue;
            };
            let needs_delinearize = view.stages.iter().any(|stage| {
                crate::view::affine_dim_indices(&stage.layout, &stage.input_shape).is_none()
            });
            if needs_delinearize && input_reread_factor(&expression, &nary.shape, slot) > 1 {
                continue;
            }
            let view = view.clone();
            let Some(rewritten) = Self::rewrite_view_input(&expression, slot, &view) else {
                continue;
            };
            expression = rewritten;
            inputs[slot] = view.input;
            folded.push((input_exec, view.input));
        }
        if folded.is_empty() {
            return false;
        }

        let (final_inputs, final_expression) = Self::deduplicate_inputs(inputs, expression);
        let new_nary = ElementwiseOperation {
            inputs: final_inputs,
            expression: final_expression,
            shape: nary.shape.clone(),
            output_datatype: nary.output_datatype,
        };
        self.execution_graph[node_idx].variant = ExecutionVariant::Elementwise(new_nary.clone());

        for (view_exec, base_inner) in &folded {
            if let Some(edge) = self.execution_graph.find_edge(*view_exec, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
            if let Some(base_exec) = self.get_input_node_in_exec_graph(*base_inner)
                && self
                    .execution_graph
                    .find_edge(base_exec, node_idx)
                    .is_none()
            {
                self.execution_graph.add_edge(base_exec, node_idx, ());
            }
        }
        self.add_physical_dependencies(graph, node_idx, &new_nary.inputs);
        for (view_exec, _) in folded {
            self.remove_node_if_dead(view_exec);
        }
        true
    }

    /// Like [`Self::try_fold_view_inputs`] but for reduce nodes: fold view
    /// inputs directly into the reduce's fused producer expression. Backward
    /// tapes are full of `reshape -> sum` chains (broadcast-gradient
    /// reductions) where the view hides the elementwise producer from
    /// `try_fuse_producer_into_reduce`; folding the view exposes it. Only
    /// called on dense (QMatMul-free) graphs.
    pub(super) fn try_fold_view_inputs_into_reduce(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let ExecutionVariant::Reduce(reduce) = self.execution_graph[node_idx].variant.clone()
        else {
            return false;
        };

        let mut expression = reduce.expression.clone();
        let mut inputs = reduce.inputs.clone();
        let mut folded = Vec::new();

        for (slot, input_inner) in reduce.inputs.iter().copied().enumerate() {
            if self.check_cached(graph, input_inner) {
                continue;
            }
            let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) else {
                continue;
            };
            if !self.execution_graph.contains_node(input_exec) {
                continue;
            }
            let ExecutionVariant::View(view) = &self.execution_graph[input_exec].variant else {
                continue;
            };
            let needs_delinearize = view.stages.iter().any(|stage| {
                crate::view::affine_dim_indices(&stage.layout, &stage.input_shape).is_none()
            });
            if needs_delinearize && input_reread_factor(&expression, &reduce.shape, slot) > 1 {
                continue;
            }
            let view = view.clone();
            let Some(rewritten) = Self::rewrite_view_input(&expression, slot, &view) else {
                continue;
            };
            expression = rewritten;
            inputs[slot] = view.input;
            folded.push((input_exec, view.input));
        }
        if folded.is_empty() {
            return false;
        }

        let (final_inputs, final_expression) = Self::deduplicate_inputs(inputs, expression);
        let mut new_reduce = reduce.clone();
        new_reduce.inputs = final_inputs;
        new_reduce.expression = final_expression;
        let new_inputs = new_reduce.inputs.clone();
        self.execution_graph[node_idx].variant = ExecutionVariant::Reduce(new_reduce);

        for (view_exec, base_inner) in &folded {
            if let Some(edge) = self.execution_graph.find_edge(*view_exec, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
            if let Some(base_exec) = self.get_input_node_in_exec_graph(*base_inner)
                && self
                    .execution_graph
                    .find_edge(base_exec, node_idx)
                    .is_none()
            {
                self.execution_graph.add_edge(base_exec, node_idx, ());
            }
        }
        self.add_physical_dependencies(graph, node_idx, &new_inputs);
        for (view_exec, _) in folded {
            self.remove_node_if_dead(view_exec);
        }
        true
    }

    /// Rewrite every access to input `target_idx` through `view`'s
    /// coordinate map: the original index expressions (the view's output
    /// coordinates) walk down the stage stack to base coordinates, with
    /// fill selects and in-bounds clamps where stages are partially defined
    /// (both select branches evaluate).
    fn rewrite_view_input(
        expr: &NaryExpr,
        target_idx: usize,
        view: &ViewOperation,
    ) -> Option<NaryExpr> {
        Some(match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| Self::rewrite_view_input(child, target_idx, view))
                    .collect::<Option<Vec<_>>>()?,
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => {
                let indices: Vec<NaryExpr> = indices
                    .iter()
                    .map(|index| Self::rewrite_view_input(index, target_idx, view))
                    .collect::<Option<Vec<_>>>()?;
                if *input_idx != target_idx {
                    NaryExpr::IndexedInput {
                        input_idx: *input_idx,
                        indices,
                    }
                } else {
                    view.value_expression(*input_idx, &indices)?.0
                }
            }
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        })
    }
}

/// The worst re-read factor across this slot's loads: the product of
/// index-space dims a load's coordinates never reference — each such dim
/// re-reads the same element once per step.
pub(super) fn input_reread_factor(expr: &NaryExpr, shape: &[usize], slot: usize) -> usize {
    fn collect_dims(expr: &NaryExpr, referenced: &mut [bool]) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    collect_dims(child, referenced);
                }
            }
            NaryExpr::IndexedInput { indices, .. } => {
                for index in indices {
                    collect_dims(index, referenced);
                }
            }
            NaryExpr::DimIndex(dim) => referenced[*dim] = true,
            NaryExpr::Scalar(_) => {}
        }
    }
    fn visit_loads(expr: &NaryExpr, shape: &[usize], slot: usize, worst: &mut usize) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    visit_loads(child, shape, slot, worst);
                }
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                for index in indices {
                    visit_loads(index, shape, slot, worst);
                }
                if *input_idx == slot {
                    let mut referenced = vec![false; shape.len()];
                    for index in indices {
                        collect_dims(index, &mut referenced);
                    }
                    let factor: usize = shape
                        .iter()
                        .zip(&referenced)
                        .filter(|(_, referenced)| !**referenced)
                        .map(|(size, _)| *size)
                        .product();
                    *worst = (*worst).max(factor);
                }
            }
            NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => {}
        }
    }
    let mut worst = 1;
    visit_loads(expr, shape, slot, &mut worst);
    worst
}
