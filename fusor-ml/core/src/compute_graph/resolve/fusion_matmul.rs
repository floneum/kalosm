use super::*;

impl Resolver {
    /// If `view` is a contiguous last-dimension narrow of a single-row qmatmul
    /// output whose shape matches `output_shape`, return its column offset.
    /// Returns `None` for any non-narrow / strided / out-of-range view.
    pub(super) fn qmatmul_last_dim_view_offset(
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn replace_indexed_qmatmul_accumulators(
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

    pub(super) fn remap_temp_accumulator_inputs(
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
