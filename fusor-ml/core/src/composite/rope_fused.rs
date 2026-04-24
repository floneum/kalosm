use std::sync::Arc;

use crate::{
    DataType, Tensor, compute_graph::NodeIndex, nary_wise::NaryExpr, tensor::DataTypeEnum,
};

impl<D: DataType> Tensor<4, D> {
    /// Apply fused interleaved RoPE (rotary position embedding).
    /// This pairs adjacent elements: (0, 1), (2, 3), etc.
    pub fn rope_fused(&self, cos: &Tensor<2, D>, sin: &Tensor<2, D>) -> Tensor<4, D> {
        self.rope_fused_impl(cos, sin, RopeMode::Interleaved, None)
    }

    /// Apply fused interleaved RoPE to the first `rotary_dim` elements and pass the rest through.
    pub fn rope_partial_fused(
        &self,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
        rotary_dim: usize,
    ) -> Tensor<4, D> {
        self.rope_fused_impl(cos, sin, RopeMode::Interleaved, Some(rotary_dim))
    }

    /// Apply fused normal RoPE (rotary position embedding).
    /// This pairs first half with second half: (0, head_dim/2), (1, head_dim/2+1), etc.
    pub fn rope_normal_fused(&self, cos: &Tensor<2, D>, sin: &Tensor<2, D>) -> Tensor<4, D> {
        self.rope_fused_impl(cos, sin, RopeMode::Normal, None)
    }

    fn rope_fused_impl(
        &self,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
        mode: RopeMode,
        rotary_dim: Option<usize>,
    ) -> Tensor<4, D> {
        let [_, _, sequence_length, head_dim] = *self.shape();
        let rotary_dim = rotary_dim.unwrap_or(head_dim);
        assert!(rotary_dim <= head_dim);
        assert_eq!(rotary_dim % 2, 0);
        // Narrow cos/sin to the sequence length
        let cos = cos.narrow(0, 0, sequence_length);
        let sin = sin.narrow(0, 0, sequence_length);

        let operation = RopeFusedOperation {
            input: self.key(),
            cos: cos.key(),
            sin: sin.key(),
            datatype: self.datatype(),
            shape: (*self.shape()).into(),
            mode,
            head_dim,
            rotary_dim,
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
    head_dim: usize,
    rotary_dim: usize,
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
        let input_times_cos = NaryExpr::mul(input_val.clone(), cos_val, self.datatype);
        let neighbor_times_sin = NaryExpr::mul(neighbor_val, sin_with_sign, self.datatype);
        let rotated = NaryExpr::add(input_times_cos, neighbor_times_sin, self.datatype);
        if self.rotary_dim == self.head_dim {
            rotated
        } else {
            let in_rotary = NaryExpr::unary_op(
                NaryExpr::DimIndex(dim_last_idx),
                "lt_rotary",
                format!("let output = select(0u, 1u, input < {}u);", self.rotary_dim),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            NaryExpr::select(
                in_rotary,
                rotated,
                input_val,
                DataTypeEnum::U32,
                self.datatype,
            )
        }
    }

    /// Build the index expressions for accessing cos/sin values (returns Vec for indexed_input)
    fn build_cos_sin_indices(&self, dim_seq_idx: usize, dim_last_idx: usize) -> Vec<NaryExpr> {
        let dim_seq = NaryExpr::DimIndex(dim_seq_idx);
        let dim_last = NaryExpr::DimIndex(dim_last_idx);

        let cos_sin_dim = match self.mode {
            // Interleaved: index = dim_last / 2
            RopeMode::Interleaved => {
                if self.rotary_dim == self.head_dim {
                    NaryExpr::unary_op(
                        dim_last,
                        "div2",
                        "let output = input / 2u;",
                        DataTypeEnum::U32,
                        DataTypeEnum::U32,
                    )
                } else {
                    NaryExpr::unary_op(
                        dim_last,
                        "partial_div2",
                        format!("let output = min(input, {}u) / 2u;", self.rotary_dim - 1),
                        DataTypeEnum::U32,
                        DataTypeEnum::U32,
                    )
                }
            }
            // Normal: index = dim_last % half
            RopeMode::Normal => {
                let half = self.head_dim / 2;
                NaryExpr::unary_op(
                    dim_last,
                    "mod_half",
                    format!("let output = input % {}u;", half),
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                )
            }
        };

        vec![dim_seq, cos_sin_dim]
    }

    /// Build the expression for computing the neighbor's last dimension index
    fn build_neighbor_index_component(&self, dim_last_idx: usize) -> NaryExpr {
        let dim_last = NaryExpr::DimIndex(dim_last_idx);

        match self.mode {
            // Interleaved: neighbor at dim_last ± 1 based on parity
            RopeMode::Interleaved => NaryExpr::unary_op(
                dim_last,
                "neighbor_idx",
                "let output = select(input - 1u, input + 1u, (input % 2u) == 0u);",
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            ),
            // Normal: neighbor at dim_last ± half based on position
            RopeMode::Normal => {
                let half = self.head_dim / 2;
                NaryExpr::unary_op(
                    dim_last,
                    "neighbor_idx",
                    format!(
                        "let output = select(input - {}u, input + {}u, input < {}u);",
                        half, half, half
                    ),
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                )
            }
        }
    }

    /// Build the condition expression for selecting sin sign
    fn build_sign_condition(&self, dim_last_idx: usize) -> NaryExpr {
        let dim_last = NaryExpr::DimIndex(dim_last_idx);

        match self.mode {
            // Interleaved: sign based on dim_last % 2 (even = -sin, odd = +sin)
            RopeMode::Interleaved => NaryExpr::unary_op(
                dim_last,
                "mod2",
                "let output = input % 2u;",
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            ),
            // Normal: sign based on dim_last / half (first half = -sin, second half = +sin)
            RopeMode::Normal => {
                let half = self.head_dim / 2;
                NaryExpr::unary_op(
                    dim_last,
                    "div_half",
                    format!("let output = input / {}u;", half),
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                )
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[tokio::test]
    async fn test_rope_fused_interleaved() {
        let device = Device::test_instance();

        let pos_shape = [11, 32]; // seq_len=11, head_dim/2=32 -> head_dim=64
        let cos_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .cos()
                    })
                    .collect()
            })
            .collect();
        let sin_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .sin()
                    })
                    .collect()
            })
            .collect();

        let cos = Tensor::new(&device, &cos_data);
        let sin = Tensor::new(&device, &sin_data);

        // Input: [1, 3, 11, 64]
        let shape = [1, 3, 11, 64];
        let data: Vec<Vec<Vec<Vec<f32>>>> = (0..shape[0])
            .map(|_| {
                (0..shape[1])
                    .map(|_| {
                        (0..shape[2])
                            .map(|_| {
                                (0..shape[3])
                                    .map(|k| (k as f32).sin()) // Some deterministic data
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let x = Tensor::new(&device, &data);

        // Compare interleaved vs fused
        let rope_original = x.rope_interleaved(&cos, &sin);
        let rope_fused = x.rope_fused(&cos, &sin);

        let original = rope_original.as_slice().await.unwrap();
        let fused = rope_fused.as_slice().await.unwrap();

        // Verify
        for b in 0..shape[0] {
            for h in 0..shape[1] {
                for s in 0..shape[2] {
                    for d in 0..shape[3] {
                        let original_val = original[[b, h, s, d]];
                        let fused_val = fused[[b, h, s, d]];
                        let diff = (original_val - fused_val).abs();
                        assert!(
                            diff < 1e-4,
                            "Mismatch at index [{}, {}, {}, {}]: original {} vs fused {}, diff {}",
                            b,
                            h,
                            s,
                            d,
                            original_val,
                            fused_val,
                            diff
                        );
                    }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_rope_partial_fused_interleaved() {
        let device = Device::test_instance();
        let cos = Tensor::new(
            &device,
            &[[1.0, 0.5], [0.75, 0.25], [0.25, 0.125], [0.9, 0.33]],
        );
        let sin = Tensor::new(
            &device,
            &[[0.0, 0.25], [0.5, 0.75], [0.125, 0.875], [0.1, 0.66]],
        );
        let x = Tensor::new(
            &device,
            &[[[
                [1.0, 2.0, 3.0, 4.0, 50.0, 60.0],
                [5.0, 6.0, 7.0, 8.0, 70.0, 80.0],
                [9.0, 10.0, 11.0, 12.0, 90.0, 100.0],
                [13.0, 14.0, 15.0, 16.0, 110.0, 120.0],
            ]]],
        );

        let rotated = x.narrow(3, 0, 4).rope_fused(&cos, &sin);
        let pass = x.narrow(3, 4, 2);
        let expected = Tensor::cat([rotated, pass], 3);
        let fused = x.rope_partial_fused(&cos, &sin, 4);

        let expected = expected.as_slice().await.unwrap();
        let fused = fused.as_slice().await.unwrap();
        for (actual, expected) in fused.as_slice().iter().zip(expected.as_slice()) {
            assert!((actual - expected).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_rope_fused_normal() {
        let device = Device::test_instance();

        let pos_shape = [11, 32]; // seq_len=11, head_dim/2=32 -> head_dim=64
        let cos_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .cos()
                    })
                    .collect()
            })
            .collect();
        let sin_data: Vec<Vec<f32>> = (0..pos_shape[0])
            .map(|i| {
                (0..pos_shape[1])
                    .map(|j| {
                        ((i as f32) / 10000f32.powf((2 * j) as f32 / (pos_shape[1] * 2) as f32))
                            .sin()
                    })
                    .collect()
            })
            .collect();

        let cos = Tensor::new(&device, &cos_data);
        let sin = Tensor::new(&device, &sin_data);

        // Input: [1, 3, 11, 64]
        let shape = [1, 3, 11, 64];
        let data: Vec<Vec<Vec<Vec<f32>>>> = (0..shape[0])
            .map(|_| {
                (0..shape[1])
                    .map(|_| {
                        (0..shape[2])
                            .map(|_| {
                                (0..shape[3])
                                    .map(|k| (k as f32).sin()) // Some deterministic data
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let x = Tensor::new(&device, &data);

        // Compare normal rope vs fused normal
        let rope_original = x.rope(&cos, &sin);
        let rope_fused = x.rope_normal_fused(&cos, &sin);

        let original = rope_original.as_slice().await.unwrap();
        let fused = rope_fused.as_slice().await.unwrap();

        // Verify
        for b in 0..shape[0] {
            for h in 0..shape[1] {
                for s in 0..shape[2] {
                    for d in 0..shape[3] {
                        let original_val = original[[b, h, s, d]];
                        let fused_val = fused[[b, h, s, d]];
                        let diff = (original_val - fused_val).abs();
                        assert!(
                            diff < 1e-4,
                            "Mismatch at index [{}, {}, {}, {}]: original {} vs fused {}, diff {}",
                            b,
                            h,
                            s,
                            d,
                            original_val,
                            fused_val,
                            diff
                        );
                    }
                }
            }
        }
    }
}
