use crate::{DataType, FloatDataType, StrideSpec, Tensor};

impl<T> Tensor<4, T>
where
    T: DataType + FloatDataType,
{
    pub fn flash_attention(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<&Tensor<2, T>>,
    ) -> Self {
        if let Some(output) = self.try_flash_attention_direct(k, v, scale, mask) {
            return output;
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
        let (k_expanded, v_expanded) = if groups > 1 {
            let k_expanded = k
                .reshape([batch, num_kv_heads, 1, kv_seq_len, head_dim])
                .broadcast_as([batch, num_kv_heads, groups, kv_seq_len, head_dim])
                .reshape([batch, num_heads, kv_seq_len, head_dim]);
            let v_expanded = v
                .reshape([batch, num_kv_heads, 1, kv_seq_len, head_dim])
                .broadcast_as([batch, num_kv_heads, groups, kv_seq_len, head_dim])
                .reshape([batch, num_heads, kv_seq_len, head_dim]);
            (k_expanded, v_expanded)
        } else {
            (k.clone(), v.clone())
        };

        let k_t = k_expanded.restride([
            StrideSpec::dim(0, batch),
            StrideSpec::dim(1, num_heads),
            StrideSpec::dim(3, head_dim),
            StrideSpec::dim(2, kv_seq_len),
        ]);
        let scores = self.mat_mul(&k_t) * T::from_f32(scale);
        let scores = if let Some(mask) = mask {
            let mask_shape = mask.shape();
            assert_eq!(
                *mask_shape,
                [q_seq_len, kv_seq_len],
                "attention mask shape {:?} does not match expected [{}, {}]",
                mask_shape,
                q_seq_len,
                kv_seq_len
            );
            scores
                + mask
                    .reshape([1, 1, q_seq_len, kv_seq_len])
                    .broadcast_as([batch, num_heads, q_seq_len, kv_seq_len])
        } else {
            scores
        };

        let weights = scores.softmax::<3>(3);
        weights.mat_mul(&v_expanded)
    }
}
