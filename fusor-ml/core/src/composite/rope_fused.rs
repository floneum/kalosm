use std::sync::Arc;

use crate::{
    DataType, Tensor, compute_graph::NodeIndex, nary_wise::NaryExpr, tensor::DataTypeEnum,
};

impl<D: DataType> Tensor<4, D> {
    /// Apply fused interleaved RoPE (rotary position embedding).
    /// This pairs adjacent elements: (0, 1), (2, 3), etc.
    ///
    /// `cos` and `sin` must already be narrowed to the sequence length.
    pub fn rope_fused(&self, cos: &Tensor<2, D>, sin: &Tensor<2, D>) -> Tensor<4, D> {
        self.rope_fused_impl(cos, sin, RopeMode::Interleaved)
    }

    /// Apply fused normal RoPE (rotary position embedding).
    /// This pairs first half with second half: (0, head_dim/2), (1, head_dim/2+1), etc.
    ///
    /// `cos` and `sin` must already be narrowed to the sequence length.
    pub fn rope_normal_fused(&self, cos: &Tensor<2, D>, sin: &Tensor<2, D>) -> Tensor<4, D> {
        self.rope_fused_impl(cos, sin, RopeMode::Normal)
    }

    fn rope_fused_impl(
        &self,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
        mode: RopeMode,
    ) -> Tensor<4, D> {
        let operation = RopeFusedOperation {
            input: self.key(),
            cos: cos.key(),
            sin: sin.key(),
            datatype: self.datatype(),
            shape: (*self.shape()).into(),
            mode,
        }
        .to_nary();

        Tensor::from_parts(self.data().custom(Arc::new(operation)))
    }
}

/// Determines how element pairs are formed for RoPE
#[derive(Debug, Clone, Copy)]
enum RopeMode {
    /// Pairs adjacent elements: (0, 1), (2, 3), etc.
    Interleaved,
    /// Pairs first half with second half: (0, half), (1, half+1), etc.
    Normal,
}

#[derive(Debug, Clone)]
struct RopeFusedOperation {
    input: NodeIndex,
    cos: NodeIndex,
    sin: NodeIndex,
    datatype: DataTypeEnum,
    shape: Box<[usize]>,
    mode: RopeMode,
}

impl RopeFusedOperation {
    fn rank(&self) -> usize {
        self.shape.len()
    }

    fn to_nary(&self) -> crate::nary_wise::NaryOperation {
        crate::nary_wise::NaryOperation {
            inputs: vec![self.input, self.cos, self.sin],
            expression: self.build_expr(),
            shape: self.shape.clone(),
            output_datatype: self.datatype,
        }
    }

    /// Build the RoPE expression: input * cos + neighbor * sin_with_sign
    fn build_expr(&self) -> NaryExpr {
        let rank = self.rank();
        let dim_seq_idx = rank - 2;
        let dim_last_idx = rank - 1;

        // Current input value
        let input_val = NaryExpr::input(0, rank);

        // Build cos/sin index based on mode
        let cos_sin_indices = self.build_cos_sin_indices(dim_seq_idx, dim_last_idx);
        let cos_val = NaryExpr::indexed_input(1, cos_sin_indices.clone());
        let sin_val = NaryExpr::indexed_input(2, cos_sin_indices);

        // Build neighbor access
        let neighbor_last_dim = self.build_neighbor_index_component(dim_last_idx);
        let neighbor_indices = self.build_indices_with_replaced_last(neighbor_last_dim);
        let neighbor_val = NaryExpr::indexed_input(0, neighbor_indices);

        // Build sign selector for sin
        let sign_condition = self.build_sign_condition(dim_last_idx);
        let neg_sin = NaryExpr::neg(sin_val.clone(), self.datatype);
        let sin_with_sign = NaryExpr::select(
            sign_condition,
            sin_val,
            neg_sin,
            DataTypeEnum::U32,
            self.datatype,
        );

        // Final expression: input * cos + neighbor * sin_with_sign
        let input_times_cos = NaryExpr::mul(input_val, cos_val, self.datatype);
        let neighbor_times_sin = NaryExpr::mul(neighbor_val, sin_with_sign, self.datatype);
        NaryExpr::add(input_times_cos, neighbor_times_sin, self.datatype)
    }

    /// Build the index expressions for accessing cos/sin values (returns Vec for indexed_input)
    fn build_cos_sin_indices(&self, dim_seq_idx: usize, dim_last_idx: usize) -> Vec<NaryExpr> {
        let dim_seq = NaryExpr::DimIndex(dim_seq_idx);
        let dim_last = NaryExpr::DimIndex(dim_last_idx);

        let cos_sin_dim = match self.mode {
            // Interleaved: index = dim_last / 2
            RopeMode::Interleaved => {
                NaryExpr::unsupported_unary(dim_last, "div2", DataTypeEnum::U32, DataTypeEnum::U32)
            }
            // Normal: index = dim_last % half
            RopeMode::Normal => NaryExpr::unsupported_unary(
                dim_last,
                "mod_half",
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            ),
        };

        vec![dim_seq, cos_sin_dim]
    }

    /// Build the expression for computing the neighbor's last dimension index
    fn build_neighbor_index_component(&self, dim_last_idx: usize) -> NaryExpr {
        let dim_last = NaryExpr::DimIndex(dim_last_idx);

        match self.mode {
            // Interleaved: neighbor at dim_last ± 1 based on parity
            RopeMode::Interleaved => NaryExpr::unsupported_unary(
                dim_last,
                "neighbor_interleaved_idx",
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            ),
            // Normal: neighbor at dim_last ± half based on position
            RopeMode::Normal => NaryExpr::unsupported_unary(
                dim_last,
                "neighbor_half_idx",
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            ),
        }
    }

    /// Build the condition expression for selecting sin sign
    fn build_sign_condition(&self, dim_last_idx: usize) -> NaryExpr {
        let dim_last = NaryExpr::DimIndex(dim_last_idx);

        match self.mode {
            // Interleaved: sign based on dim_last % 2 (even = -sin, odd = +sin)
            RopeMode::Interleaved => {
                NaryExpr::unsupported_unary(dim_last, "mod2", DataTypeEnum::U32, DataTypeEnum::U32)
            }
            // Normal: sign based on dim_last / half (first half = -sin, second half = +sin)
            RopeMode::Normal => NaryExpr::unsupported_unary(
                dim_last,
                "div_half",
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            ),
        }
    }

    /// Build index expressions that use all current dimensions except replaces the last one
    fn build_indices_with_replaced_last(&self, new_last: NaryExpr) -> Vec<NaryExpr> {
        let rank = self.rank();
        let mut components: Vec<NaryExpr> = (0..rank - 1).map(NaryExpr::DimIndex).collect();
        components.push(new_last);
        components
    }
}

// Fused vs composite comparison tests are in fusor::composite::rope::tests
