use crate::{
    DataTypeEnum, Layout, Tensor,
    nary_wise::{ElementwiseOperation, NaryExpr, NaryFunction, NaryOp, NaryScalar},
    view::ViewOperation,
};

impl Tensor {
    /// A view layered directly on this tensor's node, without collapsing
    /// into any underlying view chain. Composed-attention clusters use these
    /// so recognition can peel the exact GQA-expand / transpose / mask
    /// layouts back to the original q/k/v/mask nodes.
    fn attached_view(&self, layout: Layout) -> Tensor {
        Tensor::from_parts(self.data().view(ViewOperation::fully_defined(
            self.key(),
            layout,
            self.shape(),
            self.datatype(),
        )))
    }

    /// Causal attention in its composed form: scores at kv positions
    /// beyond the query position are replaced with `-inf` via an
    /// index-comparison select (`kv_pos <= q_pos`), so causality is pure
    /// index arithmetic — no mask tensor. The resolver recognizes the
    /// cluster and routes it to the attention row program, whose axis bound
    /// skips the masked upper-triangle tiles entirely.
    pub fn attention_causal(&self, k: &Self, v: &Self, scale: f32) -> Self {
        assert_eq!(
            self.shape()[2],
            k.shape()[2],
            "causal attention requires q_seq_len == kv_seq_len \
             (self-attention prefill); use an explicit mask otherwise"
        );
        self.compose_attention(k, v, scale, None, true)
    }

    /// Scaled dot-product attention in its composed form:
    /// `softmax(q · kᵀ · scale [+ mask]) · v`, with K/V expanded across query
    /// heads for grouped-query attention. The resolver recognizes the
    /// canonical cluster and routes it to the fused attention row program;
    /// ineligible shapes lower through the recognized matmul + softmax
    /// kernels (the same math).
    pub fn attention(&self, k: &Self, v: &Self, scale: f32, mask: Option<&Tensor>) -> Self {
        self.compose_attention(k, v, scale, mask, false)
    }

    fn compose_attention(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<&Tensor>,
        causal: bool,
    ) -> Self {
        self.assert_rank::<4>();
        k.assert_rank::<4>();
        v.assert_rank::<4>();
        assert_eq!(self.datatype(), k.datatype());
        assert_eq!(self.datatype(), v.datatype());
        if let Some(mask) = mask {
            mask.assert_rank::<2>();
            assert_eq!(self.datatype(), mask.datatype());
        }

        let q_shape = self.shape();
        let k_shape = k.shape();

        let batch = q_shape[0];
        let num_heads = q_shape[1];
        let q_seq_len = q_shape[2];
        let head_dim = q_shape[3];
        let num_kv_heads = k_shape[1];
        let kv_seq_len = k_shape[2];

        assert!(
            num_heads.is_multiple_of(num_kv_heads),
            "Number of Q heads ({}) must be divisible by number of K/V heads ({})",
            num_heads,
            num_kv_heads
        );

        let groups = num_heads / num_kv_heads;
        let expanded_shape = [batch, num_heads, kv_seq_len, head_dim];
        let expand = |tensor: &Tensor| -> Tensor {
            if groups == 1 {
                return tensor.clone();
            }
            // Two attached views: a stride-0 broadcast across the group dim,
            // then a flat reinterpret down to rank 4.
            let grouped = tensor.attached_view(Layout::from_parts(
                0,
                [batch, num_kv_heads, groups, kv_seq_len, head_dim].into(),
                [
                    num_kv_heads * kv_seq_len * head_dim,
                    kv_seq_len * head_dim,
                    0,
                    head_dim,
                    1,
                ]
                .into(),
            ));
            grouped.attached_view(Layout::contiguous(&expanded_shape))
        };
        let (k_expanded, v_expanded) = (expand(k), expand(v));

        let k_t = k_expanded.attached_view(Layout::contiguous(&expanded_shape).transpose(2, 3));
        let scores = match self.datatype() {
            DataTypeEnum::F32 => self.mat_mul(&k_t) * scale,
            DataTypeEnum::F16 => self.mat_mul(&k_t) * half::f16::from_f32(scale),
            DataTypeEnum::U32 => panic!("attention requires f32/f16 tensors"),
        };
        let scores = if causal {
            // Keep kv positions at or before the query position; everything
            // later contributes exp(-inf) = 0 to the softmax.
            let condition = NaryExpr::Op {
                children: vec![NaryExpr::DimIndex(3), NaryExpr::DimIndex(2)],
                function: NaryFunction::binary(
                    Some("causal_bound".to_string()),
                    NaryOp::LessEqual,
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                    DataTypeEnum::U32,
                ),
            };
            let datatype = self.datatype();
            let neg_inf = match datatype {
                DataTypeEnum::F32 => NaryScalar::F32(f32::NEG_INFINITY),
                DataTypeEnum::F16 => NaryScalar::F16(half::f16::NEG_INFINITY),
                DataTypeEnum::U32 => unreachable!("attention requires f32/f16"),
            };
            let expression = NaryExpr::select(
                condition,
                NaryExpr::input(0, 4),
                NaryExpr::scalar(neg_inf),
                DataTypeEnum::U32,
                datatype,
            );
            Tensor::from_parts(scores.data().nary(ElementwiseOperation {
                inputs: vec![scores.key()],
                expression,
                shape: [batch, num_heads, q_seq_len, kv_seq_len].into(),
                output_datatype: datatype,
            }))
        } else if let Some(mask) = mask {
            let mask_shape = mask.shape();
            assert_eq!(
                mask_shape,
                &[q_seq_len, kv_seq_len],
                "attention mask shape {:?} does not match expected [{}, {}]",
                mask_shape,
                q_seq_len,
                kv_seq_len
            );
            let mask_view = mask.attached_view(Layout::from_parts(
                0,
                [batch, num_heads, q_seq_len, kv_seq_len].into(),
                [0, 0, kv_seq_len, 1].into(),
            ));
            scores + mask_view
        } else {
            scores
        };

        let weights = scores.softmax(3);
        weights.mat_mul(&v_expanded)
    }
}
