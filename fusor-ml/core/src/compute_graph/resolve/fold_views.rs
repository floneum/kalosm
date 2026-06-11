use crate::view::{AffineIndex, ViewOperation, affine_dim_indices};

use super::*;

impl Resolver {
    /// Fold view inputs of an n-ary node directly into its expression: each
    /// `IndexedInput` through the view becomes an `IndexedInput` of the
    /// view's base node with the view's coordinate mapping applied (and a
    /// bounds-select around partially-defined views). This removes the view
    /// node from between the producer and consumer, so the n-ary fusion
    /// passes see through layout changes instead of stopping at them.
    ///
    /// Only affine views fold (no divmod in the rewritten indices); anything
    /// else stays a view node and materializes through the gather fallback.
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
            let Some(affine) = affine_dim_indices(&view.layout, &view.input_shape) else {
                continue;
            };
            let view = view.clone();
            expression = Self::rewrite_view_input(&expression, slot, &view, &affine);
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

    /// Rewrite every access to input `target_idx` through `view`'s coordinate
    /// mapping. The original index expressions (the view's output
    /// coordinates) feed the affine per-base-dimension indices; partially
    /// defined views wrap the load in `select(in_bounds, load, fill)` with
    /// the load coordinates clamped in-bounds (both select branches
    /// evaluate).
    fn rewrite_view_input(
        expr: &NaryExpr,
        target_idx: usize,
        view: &ViewOperation,
        affine: &[AffineIndex],
    ) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| Self::rewrite_view_input(child, target_idx, view, affine))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => {
                let indices: Vec<NaryExpr> = indices
                    .iter()
                    .map(|index| Self::rewrite_view_input(index, target_idx, view, affine))
                    .collect();
                if *input_idx != target_idx {
                    return NaryExpr::IndexedInput {
                        input_idx: *input_idx,
                        indices,
                    };
                }

                let fully_defined = view.is_fully_defined();
                let base_indices = affine
                    .iter()
                    .zip(&*view.input_shape)
                    .map(|(index, &extent)| {
                        let index = index.to_expr(&indices);
                        if fully_defined || extent == 0 {
                            index
                        } else {
                            NaryExpr::unary_op(
                                index,
                                "clamp_dim",
                                NaryOp::MinConst(NaryScalar::U32(extent as u32 - 1)),
                                DataTypeEnum::U32,
                                DataTypeEnum::U32,
                            )
                        }
                    })
                    .collect();
                let loaded = NaryExpr::IndexedInput {
                    input_idx: *input_idx,
                    indices: base_indices,
                };
                if fully_defined {
                    return loaded;
                }

                let mut condition = NaryExpr::scalar(NaryScalar::U32(1));
                for (dim, (&defined, &size)) in view.defined.iter().zip(view.shape()).enumerate() {
                    if defined >= size {
                        continue;
                    }
                    let lt_defined = NaryExpr::unary_op(
                        indices[dim].clone(),
                        "lt_defined",
                        NaryOp::LessConst(NaryScalar::U32(defined as u32)),
                        DataTypeEnum::U32,
                        DataTypeEnum::U32,
                    );
                    condition = NaryExpr::mul(condition, lt_defined, DataTypeEnum::U32);
                }
                NaryExpr::select(
                    condition,
                    loaded,
                    NaryExpr::scalar(view.fill),
                    DataTypeEnum::U32,
                    view.datatype,
                )
            }
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        }
    }
}
