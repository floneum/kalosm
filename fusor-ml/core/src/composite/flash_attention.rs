use crate::{DataTypeEnum, StrideSpec, Tensor};

impl Tensor {
    /// Causal flash attention: the kernel applies a strict lower-triangular
    /// causal mask internally, skipping the upper-triangle Q·K work entirely.
    /// `q_seq_len` must equal `kv_seq_len` (prefill self-attention); other
    /// shapes fall back to [`flash_attention`] with an explicit mask.
    pub fn flash_attention_causal(&self, k: &Self, v: &Self, scale: f32) -> Self {
        self.assert_rank::<4>();
        if let Some(output) = self.try_flash_attention_direct_causal(k, v, scale) {
            return output;
        }
        // Fallback: materialize a causal mask and re-enter the masked path.
        let q_shape = self.shape();
        let seq_len = q_shape[2];
        match self.datatype() {
            DataTypeEnum::F32 => {
                let mut data = vec![0.0f32; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        data[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
                let mask = Tensor::from_slice::<f32>(self.device(), [seq_len, seq_len], &data);
                self.flash_attention(k, v, scale, Some(&mask))
            }
            DataTypeEnum::F16 => {
                let mut data = vec![half::f16::from_f32(0.0); seq_len * seq_len];
                let neg_inf = half::f16::from_f32(f32::NEG_INFINITY);
                for i in 0..seq_len {
                    for j in (i + 1)..seq_len {
                        data[i * seq_len + j] = neg_inf;
                    }
                }
                let mask =
                    Tensor::from_slice::<half::f16>(self.device(), [seq_len, seq_len], &data);
                self.flash_attention(k, v, scale, Some(&mask))
            }
            DataTypeEnum::U32 => panic!("flash_attention requires f32/f16 tensors"),
        }
    }

    pub fn flash_attention(&self, k: &Self, v: &Self, scale: f32, mask: Option<&Tensor>) -> Self {
        self.assert_rank::<4>();
        k.assert_rank::<4>();
        v.assert_rank::<4>();
        assert_eq!(self.datatype(), k.datatype());
        assert_eq!(self.datatype(), v.datatype());
        if let Some(mask) = mask {
            mask.assert_rank::<2>();
            assert_eq!(self.datatype(), mask.datatype());
        }
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
        let scores = match self.datatype() {
            DataTypeEnum::F32 => self.mat_mul(&k_t) * scale,
            DataTypeEnum::F16 => self.mat_mul(&k_t) * half::f16::from_f32(scale),
            DataTypeEnum::U32 => panic!("flash_attention requires f32/f16 tensors"),
        };
        let scores = if let Some(mask) = mask {
            let mask_shape = mask.shape();
            assert_eq!(
                mask_shape,
                &[q_seq_len, kv_seq_len],
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

        let weights = scores.softmax(3);
        weights.mat_mul(&v_expanded)
    }
}
