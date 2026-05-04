use crate::{CastTensor, DataType, LastRank, Tensor};

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn rms_norm_fused<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self
    where
        T: CastTensor<f32>,
        f32: CastTensor<T>,
        Tensor<R, f32>: LastRank<OUT_RANK, f32>,
    {
        if let Some(output) = self.try_rms_norm_direct(weight, bias, eps) {
            return output;
        }

        let hidden_size = *self.shape().last().unwrap() as f32;
        let mut kept_shape = *self.shape();
        kept_shape[R - 1] = 1;

        let input = self.cast::<f32>();
        let squared = &input * &input;
        let mean_square = (squared.sum::<OUT_RANK>(crate::D::Minus1) / hidden_size)
            .reshape(kept_shape)
            .broadcast_as(*self.shape());
        let rms = (mean_square + eps).sqrt();
        let mut output = (&input / &rms) * &weight.cast::<f32>().broadcast_as(*self.shape());

        if let Some(bias) = bias {
            output = &output + &bias.cast::<f32>().broadcast_as(*self.shape());
        }

        output.cast::<T>()
    }

    pub fn rms_norm_fused_no_bias<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, T>,
        eps: f32,
    ) -> Self
    where
        T: CastTensor<f32>,
        f32: CastTensor<T>,
        Tensor<R, f32>: LastRank<OUT_RANK, f32>,
    {
        self.rms_norm_fused::<W, OUT_RANK>(weight, None, eps)
    }

    pub fn rms_norm_residual_fused<const W: usize, const OUT_RANK: usize>(
        &self,
        residual: &Self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self
    where
        T: CastTensor<f32>,
        f32: CastTensor<T>,
        Tensor<R, f32>: LastRank<OUT_RANK, f32>,
    {
        if let Some(output) = self.try_rms_norm_residual_direct(residual, weight, bias, eps) {
            return output;
        }

        (self.clone() + residual.clone()).rms_norm_fused::<W, OUT_RANK>(weight, bias, eps)
    }
}
