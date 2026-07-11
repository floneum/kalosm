use crate::nary_wise::NaryFunction;

use super::*;

impl Resolver {
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

    /// Extract a (possibly empty) unary function chain over exactly one
    /// read of input 0 with arbitrary index expressions, innermost function
    /// first. The index expressions must not read any input themselves.
    pub(super) fn extract_unary_chain_indexed(
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
}
