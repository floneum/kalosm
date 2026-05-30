use crate::{DataTypeEnum, Tensor};

impl Tensor {
    pub fn rms_norm_fused(&self, weight: &Tensor, bias: Option<&Tensor>, eps: f32) -> Self {
        let original_datatype = self.datatype();
        assert!(
            matches!(original_datatype, DataTypeEnum::F32 | DataTypeEnum::F16),
            "rms_norm_fused only supports f32/f16 tensors"
        );
        assert_eq!(weight.datatype(), original_datatype);
        if let Some(bias) = bias {
            assert_eq!(bias.datatype(), original_datatype);
        }
        if let Some(output) = self.try_rms_norm_direct(weight, bias, eps) {
            return output;
        }

        let hidden_size = *self.shape().last().unwrap() as f32;
        let last_dim = self.rank() - 1;
        let mut kept_shape = self.shape().to_vec();
        kept_shape[last_dim] = 1;

        let input = self.cast_to(DataTypeEnum::F32);
        let squared = &input * &input;
        let mean_square = (squared.sum(last_dim) / hidden_size)
            .reshape(&kept_shape)
            .broadcast_as(self.shape());
        let rms = (mean_square + eps).sqrt();
        let mut output =
            (&input / &rms) * &weight.cast_to(DataTypeEnum::F32).broadcast_as(self.shape());

        if let Some(bias) = bias {
            output = &output + &bias.cast_to(DataTypeEnum::F32).broadcast_as(self.shape());
        }

        output.cast_to(original_datatype)
    }

    pub fn rms_norm_fused_no_bias(&self, weight: &Tensor, eps: f32) -> Self {
        self.rms_norm_fused(weight, None, eps)
    }

    pub fn rms_norm_residual_fused(
        &self,
        residual: &Self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Self {
        if let Some(output) = self.try_rms_norm_residual_direct(residual, weight, bias, eps) {
            return output;
        }

        (self.clone() + residual.clone()).rms_norm_fused(weight, bias, eps)
    }
}
