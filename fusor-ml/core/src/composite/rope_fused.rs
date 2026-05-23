use crate::{
    DataType, Layout, Tensor,
    compute_graph::NodeIndex,
    nary_wise::{NaryExpr, NaryOp, NaryScalar},
    tensor::{DataTypeEnum, LazyTensorData, TensorInfo},
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

    /// Apply fused interleaved RoPE to query and key tensors in one kernel.
    ///
    /// The returned tensors are layout views into one concatenated allocation. This keeps the
    /// decode graph to one RoPE kernel per layer while preserving separate q/k tensor shapes.
    pub fn rope_pair_fused(
        &self,
        k: &Tensor<4, D>,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
    ) -> (Tensor<4, D>, Tensor<4, D>) {
        self.rope_pair_fused_impl(k, cos, sin, RopeMode::Interleaved)
    }

    /// Apply fused normal RoPE to query and key tensors in one kernel.
    ///
    /// The returned tensors are layout views into one concatenated allocation. This keeps the
    /// decode graph to one RoPE kernel per layer while preserving separate q/k tensor shapes.
    pub fn rope_normal_pair_fused(
        &self,
        k: &Tensor<4, D>,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
    ) -> (Tensor<4, D>, Tensor<4, D>) {
        self.rope_pair_fused_impl(k, cos, sin, RopeMode::Normal)
    }

    fn rope_fused_impl(
        &self,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
        mode: RopeMode,
    ) -> Tensor<4, D> {
        let [_, _, _, head_dim] = *self.shape();

        let operation = RopeFusedOperation {
            input: self.key(),
            cos: cos.key(),
            sin: sin.key(),
            datatype: self.datatype(),
            shape: (*self.shape()).into(),
            mode,
            head_dim,
        }
        .to_nary();

        Tensor::from_parts(self.data().nary(operation))
    }

    fn rope_pair_fused_impl(
        &self,
        k: &Tensor<4, D>,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
        mode: RopeMode,
    ) -> (Tensor<4, D>, Tensor<4, D>) {
        let q_shape = *self.shape();
        let k_shape = *k.shape();
        assert_eq!(
            q_shape[0], k_shape[0],
            "paired RoPE requires q and k batch dimensions to match"
        );
        assert_eq!(
            q_shape[2], k_shape[2],
            "paired RoPE requires q and k sequence dimensions to match"
        );
        assert_eq!(
            q_shape[3], k_shape[3],
            "paired RoPE requires q and k head dimensions to match"
        );

        let q_elements = q_shape.iter().product::<usize>();
        let k_elements = k_shape.iter().product::<usize>();
        let total_elements = q_elements
            .checked_add(k_elements)
            .expect("paired RoPE output element count overflow");
        assert!(
            total_elements <= u32::MAX as usize,
            "paired RoPE output is too large for nary direct indexing"
        );
        assert!(
            q_elements % k_elements == 0,
            "paired RoPE expects q element count to be a whole multiple of k element count"
        );

        let operation = RopePairFusedOperation {
            q: self.key(),
            k: k.key(),
            cos: cos.key(),
            sin: sin.key(),
            datatype: self.datatype(),
            q_shape,
            k_shape,
            q_elements,
            k_elements,
            total_elements,
            mode,
        }
        .to_nary();

        let device = self.device().clone();
        let key = device.compute_graph().create_nary(operation);
        let combined: Tensor<1, D> = Tensor::from_parts(LazyTensorData::from_parts(
            device,
            TensorInfo::new(vec![total_elements].into_boxed_slice(), self.datatype()),
            key,
        ));

        let q = combined.restride_layout(Layout::from_parts(
            0,
            q_shape.into(),
            row_major_strides(&q_shape).into_boxed_slice(),
        ));
        let k = combined.restride_layout(Layout::from_parts(
            q_elements,
            k_shape.into(),
            row_major_strides(&k_shape).into_boxed_slice(),
        ));

        (q, k)
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
}

#[derive(Debug, Clone)]
struct RopePairFusedOperation {
    q: NodeIndex,
    k: NodeIndex,
    cos: NodeIndex,
    sin: NodeIndex,
    datatype: DataTypeEnum,
    q_shape: [usize; 4],
    k_shape: [usize; 4],
    q_elements: usize,
    k_elements: usize,
    total_elements: usize,
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
        let dim_seq = NaryExpr::DimIndex(dim_seq_idx);
        let dim_last = NaryExpr::DimIndex(dim_last_idx);

        // Current input value
        let input_val = NaryExpr::input(0, rank);

        // Build cos/sin index based on mode
        let cos_sin_indices = build_cos_sin_indices(dim_seq, dim_last.clone(), self.head_dim, self.mode);
        let cos_val = NaryExpr::indexed_input(1, cos_sin_indices.clone());
        let sin_val = NaryExpr::indexed_input(2, cos_sin_indices);

        // Build neighbor access
        let neighbor_last_dim = build_neighbor_index_component(dim_last.clone(), self.head_dim, self.mode);
        let neighbor_indices = self.build_indices_with_replaced_last(neighbor_last_dim);
        let neighbor_val = NaryExpr::indexed_input(0, neighbor_indices);

        // Build sign selector for sin
        let sign_condition = build_sign_condition(dim_last, self.head_dim, self.mode);
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

    /// Build index expressions that use all current dimensions except replaces the last one
    fn build_indices_with_replaced_last(&self, new_last: NaryExpr) -> Vec<NaryExpr> {
        let rank = self.rank();
        let mut components: Vec<NaryExpr> = (0..rank - 1).map(NaryExpr::DimIndex).collect();
        components.push(new_last);
        components
    }
}

/// Build the index expressions for accessing cos/sin values (returns Vec for indexed_input)
fn build_cos_sin_indices(
    dim_seq: NaryExpr,
    dim_last: NaryExpr,
    head_dim: usize,
    mode: RopeMode,
) -> Vec<NaryExpr> {
    let cos_sin_dim = match mode {
        // Interleaved: index = dim_last / 2
        RopeMode::Interleaved => NaryExpr::unary_op(
            dim_last,
            "div2",
            NaryOp::DivConst(NaryScalar::U32(2)),
            DataTypeEnum::U32,
            DataTypeEnum::U32,
        ),
        // Normal: index = dim_last % half
        RopeMode::Normal => {
            let half = head_dim / 2;
            NaryExpr::unary_op(
                dim_last,
                "mod_half",
                NaryOp::RemConst(NaryScalar::U32(half as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )
        }
    };

    vec![dim_seq, cos_sin_dim]
}

/// Build the expression for computing the neighbor's last dimension index
fn build_neighbor_index_component(dim_last: NaryExpr, head_dim: usize, mode: RopeMode) -> NaryExpr {
    match mode {
        // Interleaved: neighbor at dim_last ± 1 based on parity
        RopeMode::Interleaved => {
            let parity = NaryExpr::unary_op(
                dim_last.clone(),
                "mod2",
                NaryOp::RemConst(NaryScalar::U32(2)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            let is_even = NaryExpr::unary_op(
                parity,
                "eq0",
                NaryOp::EqualConst(NaryScalar::U32(0)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            let plus_one = NaryExpr::unary_op(
                dim_last.clone(),
                "add1",
                NaryOp::AddConst(NaryScalar::U32(1)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            let minus_one = NaryExpr::unary_op(
                dim_last,
                "sub1",
                NaryOp::SubConst(NaryScalar::U32(1)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            NaryExpr::select(
                is_even,
                plus_one,
                minus_one,
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )
        }
        // Normal: neighbor at dim_last ± half based on position
        RopeMode::Normal => {
            let half = head_dim / 2;
            let first_half = NaryExpr::unary_op(
                dim_last.clone(),
                "lt_half",
                NaryOp::LessConst(NaryScalar::U32(half as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            let plus_half = NaryExpr::unary_op(
                dim_last.clone(),
                "add_half",
                NaryOp::AddConst(NaryScalar::U32(half as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            let minus_half = NaryExpr::unary_op(
                dim_last,
                "sub_half",
                NaryOp::SubConst(NaryScalar::U32(half as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            NaryExpr::select(
                first_half,
                plus_half,
                minus_half,
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )
        }
    }
}

/// Build the condition expression for selecting sin sign
fn build_sign_condition(dim_last: NaryExpr, head_dim: usize, mode: RopeMode) -> NaryExpr {
    match mode {
        // Interleaved: sign based on dim_last % 2 (even = -sin, odd = +sin)
        RopeMode::Interleaved => NaryExpr::unary_op(
            dim_last,
            "mod2",
            NaryOp::RemConst(NaryScalar::U32(2)),
            DataTypeEnum::U32,
            DataTypeEnum::U32,
        ),
        // Normal: sign based on dim_last / half (first half = -sin, second half = +sin)
        RopeMode::Normal => {
            let half = head_dim / 2;
            NaryExpr::unary_op(
                dim_last,
                "div_half",
                NaryOp::DivConst(NaryScalar::U32(half as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )
        }
    }
}

impl RopePairFusedOperation {
    fn to_nary(&self) -> crate::nary_wise::NaryOperation {
        crate::nary_wise::NaryOperation {
            inputs: vec![self.q, self.k, self.cos, self.sin],
            expression: self.build_expr(),
            shape: vec![self.total_elements].into_boxed_slice(),
            output_datatype: self.datatype,
        }
    }

    fn build_expr(&self) -> NaryExpr {
        let flat = NaryExpr::DimIndex(0);
        let is_q = NaryExpr::unary_op(
            flat.clone(),
            "lt_q_elements",
            NaryOp::LessConst(NaryScalar::U32(self.q_elements as u32)),
            DataTypeEnum::U32,
            DataTypeEnum::U32,
        );

        let q_flat = rem_const(flat.clone(), self.q_elements, "q_flat");
        let k_flat = rem_const(flat, self.k_elements, "k_flat");
        let q = self.build_rope_for_input(0, q_flat, self.q_shape);
        let k = self.build_rope_for_input(1, k_flat, self.k_shape);

        NaryExpr::select(is_q, q, k, DataTypeEnum::U32, self.datatype)
    }

    fn build_rope_for_input(
        &self,
        input_idx: usize,
        flat: NaryExpr,
        shape: [usize; 4],
    ) -> NaryExpr {
        let head_dim = shape[3];
        let indices = row_major_indices_from_flat(flat.clone(), &shape);
        let input_val = NaryExpr::indexed_input(input_idx, indices.clone());
        let dim_seq = indices[2].clone();
        let dim_last = indices[3].clone();

        let cos_sin_indices = build_cos_sin_indices(dim_seq, dim_last.clone(), head_dim, self.mode);
        let cos_val = NaryExpr::indexed_input(2, cos_sin_indices.clone());
        let sin_val = NaryExpr::indexed_input(3, cos_sin_indices);

        let neighbor_last = build_neighbor_index_component(dim_last.clone(), head_dim, self.mode);
        let mut neighbor_indices = indices;
        neighbor_indices[3] = neighbor_last;
        let neighbor_val = NaryExpr::indexed_input(input_idx, neighbor_indices);

        let sign_condition = build_sign_condition(dim_last, head_dim, self.mode);
        let neg_sin = NaryExpr::neg(sin_val.clone(), self.datatype);
        let sin_with_sign = NaryExpr::select(
            sign_condition,
            sin_val,
            neg_sin,
            DataTypeEnum::U32,
            self.datatype,
        );

        let input_times_cos = NaryExpr::mul(input_val, cos_val, self.datatype);
        let neighbor_times_sin = NaryExpr::mul(neighbor_val, sin_with_sign, self.datatype);
        NaryExpr::add(input_times_cos, neighbor_times_sin, self.datatype)
    }

}

fn row_major_strides(shape: &[usize; 4]) -> Vec<usize> {
    let mut stride = 1;
    let mut strides = vec![0; shape.len()];
    for (i, dim) in shape.iter().enumerate().rev() {
        strides[i] = stride;
        stride *= dim;
    }
    strides
}

fn row_major_indices_from_flat(flat: NaryExpr, shape: &[usize; 4]) -> Vec<NaryExpr> {
    let mut indices = Vec::with_capacity(shape.len());
    for axis in 0..shape.len() {
        let divisor = shape[axis + 1..].iter().product::<usize>();
        let quotient = if divisor == 1 {
            flat.clone()
        } else {
            NaryExpr::unary_op(
                flat.clone(),
                "div_stride",
                NaryOp::DivConst(NaryScalar::U32(divisor as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )
        };
        indices.push(rem_const(quotient, shape[axis], "dim_index"));
    }
    indices
}

fn rem_const(value: NaryExpr, modulus: usize, name: &str) -> NaryExpr {
    if modulus == 1 {
        NaryExpr::scalar(NaryScalar::U32(0))
    } else {
        NaryExpr::unary_op(
            value,
            name,
            NaryOp::RemConst(NaryScalar::U32(modulus as u32)),
            DataTypeEnum::U32,
            DataTypeEnum::U32,
        )
    }
}

// Fused vs composite comparison tests are in fusor::composite::rope::tests
