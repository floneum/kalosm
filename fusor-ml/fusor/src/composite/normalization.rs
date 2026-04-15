//! Normalization operations that work on both CPU and GPU backends.

use crate::{
    AddOp, ConcreteTensor, DivOp, ExpOp, FloatOps, MulOp, SimdBinaryOp, SimdElement, SimdUnaryOp,
    SqrtOp, SubOp, Tensor,
};
use fusor_core::{
    DataType, FloatDataType, LastRank as GpuLastRank, NextRankInner as GpuNextRankInner,
};
use fusor_cpu::{LastRank as CpuLastRank, MaxOp, SimdReduceOp, SumOp, TensorBacking};

impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Softmax along a specific axis.
    ///
    /// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x - max(x)))
    ///
    /// The subtraction of max(x) is for numerical stability.
    pub fn softmax<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        MaxOp: SimdReduceOp<D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Sub<Output = D> + std::ops::Div<Output = D>,
        SubOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        match self {
            Tensor::Cpu(_) => self.softmax_cpu_impl(axis),
            Tensor::Gpu(t) => Tensor::Gpu(t.softmax(axis)),
        }
    }

    /// Softmax along the last dimension.
    ///
    /// This is a convenience method equivalent to `softmax(R - 1)`.
    /// For f32 CPU tensors, this uses an optimized fused implementation.
    pub fn softmax_last_dim<const OUT_RANK: usize>(&self) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        MaxOp: SimdReduceOp<D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Sub<Output = D> + std::ops::Div<Output = D>,
        SubOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        self.softmax::<OUT_RANK>(R - 1)
    }

    /// Slow softmax using composite operations (non-fused).
    ///
    /// This is provided for API parity with fusor-core. On CPU, this is the same
    /// as `softmax`. On GPU, fusor-core has an optimized fused kernel.
    pub fn softmax_slow<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        MaxOp: SimdReduceOp<D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Sub<Output = D> + std::ops::Div<Output = D>,
        SubOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        match self {
            Tensor::Cpu(_) => self.softmax_cpu_impl(axis),
            Tensor::Gpu(t) => Tensor::Gpu(t.softmax_slow(axis)),
        }
    }

    /// Slow softmax along the last dimension using composite operations.
    ///
    /// This is provided for API parity with fusor-core.
    pub fn softmax_slow_last_dim<const OUT_RANK: usize>(&self) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        MaxOp: SimdReduceOp<D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Sub<Output = D> + std::ops::Div<Output = D>,
        SubOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        self.softmax_slow::<OUT_RANK>(R - 1)
    }

    /// CPU implementation of softmax
    fn softmax_cpu_impl<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        MaxOp: SimdReduceOp<D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Sub<Output = D> + std::ops::Div<Output = D>,
        SubOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        // Materialize to concrete first since we need it for operations
        let concrete = self.to_concrete();
        let input_shape = concrete.shape();

        // max(x) with keepdim for broadcasting
        let max_val = concrete.max_keepdim::<OUT_RANK>(axis);

        // x - max(x): broadcast max_val to input shape before subtraction
        let max_broadcast = max_val.broadcast_as(input_shape);
        let shifted = (concrete - max_broadcast).to_concrete();

        // exp(x - max(x))
        let exp_val = shifted.exp().to_concrete();

        // sum(exp(...)) with keepdim, broadcast to input shape
        let sum_exp = exp_val.sum_keepdim::<OUT_RANK>(axis);
        let sum_broadcast = sum_exp.broadcast_as(input_shape);

        // exp / sum
        (exp_val / sum_broadcast).to_concrete()
    }

    /// RMS Normalization along the last axis.
    ///
    /// rms_norm(x) = x / sqrt(mean(x^2) + eps) * weight
    ///
    /// Note: This is a simplified implementation that assumes weight has the same
    /// rank as input. For more complex broadcasting, use the GPU's optimized kernels directly.
    pub fn rms_norm<const OUT_RANK: usize, B2>(
        &self,
        weight: &Tensor<R, D, B2>,
        eps: D,
    ) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D> + std::ops::Div<Output = D> + std::ops::Add<Output = D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        B2: TensorBacking<R, Elem = D>,
    {
        let axis = R - 1; // Normalize along last axis

        // Materialize to concrete first since we need it for operations
        let concrete = self.to_concrete();
        let input_shape = concrete.shape();

        // x^2
        let x_sq = concrete.sqr();

        // mean(x^2) with keepdim along last axis
        let mean_sq = x_sq.mean_keepdim::<OUT_RANK>(axis);

        // mean(x^2) + eps - materialize first since add_scalar requires concrete tensor
        let mean_sq_eps = mean_sq.to_concrete().add_scalar(eps);

        // sqrt(mean(x^2) + eps), broadcast to input shape
        let rms = mean_sq_eps.sqrt();
        let rms_broadcast = rms.broadcast_as(input_shape);

        // x / rms
        let normalized = (concrete / rms_broadcast).to_concrete();

        // normalized * weight
        (&normalized * weight).to_concrete()
    }

    /// Layer Normalization along the last axis.
    ///
    /// layer_norm(x) = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
    ///
    /// If remove_mean is false, skips the mean subtraction (becomes RMS-like).
    ///
    /// Note: This is a simplified implementation that assumes weight and bias have
    /// the same rank as input.
    pub fn layer_norm<const OUT_RANK: usize, B2, B3>(
        &self,
        weight: &Tensor<R, D, B2>,
        bias: Option<&Tensor<R, D, B3>>,
        eps: D,
        remove_mean: bool,
    ) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Add<Output = D>
            + std::ops::Sub<Output = D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SubOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        B2: TensorBacking<R, Elem = D>,
        B3: TensorBacking<R, Elem = D>,
    {
        let axis = R - 1;

        // Materialize to concrete first since we need it for operations
        let concrete = self.to_concrete();
        let input_shape = concrete.shape();

        // Optionally subtract mean
        let centered: Tensor<R, D> = if remove_mean {
            let mean = concrete.mean_keepdim::<OUT_RANK>(axis);
            let mean_broadcast = mean.broadcast_as(input_shape);
            (concrete - mean_broadcast).to_concrete()
        } else {
            concrete
        };

        // Compute variance: mean(centered^2)
        let centered_sq = centered.sqr();
        let var = centered_sq.mean_keepdim::<OUT_RANK>(axis);

        // sqrt(var + eps) - materialize first since add_scalar requires concrete tensor
        let var_plus_eps = var.to_concrete().add_scalar(eps);
        let std = var_plus_eps.sqrt();

        // centered / std: broadcast std to input shape
        let std_broadcast = std.broadcast_as(input_shape);
        let normalized: Tensor<R, D> = (&centered / &std_broadcast).to_concrete();

        // normalized * weight
        let scaled: Tensor<R, D> = match (&normalized, weight) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu((a * b).to_concrete()),
            // Use mul_ for broadcasting (weight may be 1D broadcast to R)
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(a.mul_::<R, R>(b)),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        };

        // + bias if present
        if let Some(b) = bias {
            match (&scaled, b) {
                (Tensor::Cpu(a), Tensor::Cpu(c)) => Tensor::Cpu((a + c).to_concrete()),
                // Use add_ for broadcasting (bias may be 1D broadcast to R)
                (Tensor::Gpu(a), Tensor::Gpu(c)) => Tensor::Gpu(a.add_::<R, R>(c)),
                _ => panic!("Cannot mix CPU and GPU tensors"),
            }
        } else {
            scaled
        }
    }

    /// Fused RMSNorm kernel that performs the entire normalization in a single kernel launch (GPU).
    ///
    /// Formula: output = input / sqrt(mean(input^2) + eps) * weight + bias
    ///
    /// On GPU, this is more efficient than the composite implementation which requires multiple
    /// kernel launches. On CPU, this delegates to the composite operations.
    ///
    /// # Type Parameters
    /// * `W` - Rank of the weight/bias tensor (typically 1 for per-feature weights)
    ///
    /// # Arguments
    /// * `weight` - Scale tensor to apply after normalization
    /// * `bias` - Optional bias tensor to add after scaling
    /// * `eps` - Epsilon for numerical stability
    pub fn rms_norm_fused<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, D, ConcreteTensor<D, W>>,
        bias: Option<&Tensor<W, D, ConcreteTensor<D, W>>>,
        eps: f32,
    ) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Add<Output = D>
            + fusor_core::CastTensor<f32>,
        f32: fusor_core::CastTensor<D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<W, D>): fusor_core::MaxRank<R, D>,
    {
        match (self, weight, bias) {
            // GPU path - use the optimized fused kernel
            (Tensor::Gpu(input), Tensor::Gpu(weight), bias_opt) => {
                let gpu_bias = bias_opt.map(|b| match b {
                    Tensor::Gpu(bias) => bias,
                    _ => panic!("Bias must be on GPU when input is on GPU"),
                });
                Tensor::Gpu(input.rms_norm_fused(weight, gpu_bias, eps))
            }
            // CPU path - use composite operations
            (Tensor::Cpu(_), Tensor::Cpu(_), _) => {
                self.rms_norm_fused_cpu_impl::<W, OUT_RANK>(weight, bias, eps)
            }
            _ => panic!("All tensors must be on the same device"),
        }
    }

    /// Fused RMSNorm without bias
    pub fn rms_norm_fused_no_bias<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, D, ConcreteTensor<D, W>>,
        eps: f32,
    ) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Add<Output = D>
            + fusor_core::CastTensor<f32>,
        f32: fusor_core::CastTensor<D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<W, D>): fusor_core::MaxRank<R, D>,
    {
        self.rms_norm_fused::<W, OUT_RANK>(weight, None, eps)
    }

    /// CPU implementation of fused RMS norm using composite operations
    fn rms_norm_fused_cpu_impl<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, D, ConcreteTensor<D, W>>,
        bias: Option<&Tensor<W, D, ConcreteTensor<D, W>>>,
        eps: f32,
    ) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        fusor_core::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <fusor_core::Tensor<R, D> as fusor_core::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = fusor_core::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D> + std::ops::Div<Output = D> + std::ops::Add<Output = D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
    {
        let axis = R - 1; // Normalize along last axis
        let eps_d = D::from_f32(eps);

        // Materialize to concrete first since we need it for operations
        let concrete = self.to_concrete();
        let input_shape = concrete.shape();

        // x^2
        let x_sq = concrete.sqr();

        // mean(x^2) with keepdim along last axis
        let mean_sq = x_sq.mean_keepdim::<OUT_RANK>(axis);

        // mean(x^2) + eps - materialize first since add_scalar requires concrete tensor
        let mean_sq_eps = mean_sq.to_concrete().add_scalar(eps_d);

        // sqrt(mean(x^2) + eps), broadcast to input shape
        let rms = mean_sq_eps.sqrt();
        let rms_broadcast = rms.broadcast_as(input_shape);

        // x / rms
        let normalized: Tensor<R, D> = (concrete / rms_broadcast).to_concrete();
        let weight_broadcast = weight.broadcast_as(input_shape);
        let scaled: Tensor<R, D> = match (&normalized, &weight_broadcast) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu((a * b).to_concrete()),
            _ => unreachable!(),
        };

        // Add bias if present
        if let Some(b) = bias {
            let bias_broadcast = b.broadcast_as(input_shape);
            match (&scaled, &bias_broadcast) {
                (Tensor::Cpu(a), Tensor::Cpu(c)) => Tensor::Cpu((a + c).to_concrete()),
                _ => unreachable!(),
            }
        } else {
            scaled
        }
    }
}

// Specialized f32 implementation with fused softmax
impl<const R: usize, B> Tensor<R, f32, B>
where
    B: TensorBacking<R, Elem = f32>,
    fusor_core::Tensor<R, f32>: fusor_core::LastRankInner,
{
    /// Optimized fused softmax along the last dimension for f32.
    ///
    /// This performs the entire softmax (max, exp, sum, normalize) in a single
    /// pass through memory, which is significantly faster for large tensors.
    pub fn softmax_last_dim_fused<const OUT_RANK: usize>(&self) -> Tensor<R, f32>
    where
        fusor_core::Tensor<R, f32>: fusor_core::LastRank<OUT_RANK, f32>,
    {
        self.dispatch_ref(
            |t| {
                // Make contiguous if needed, then use fused kernel
                let contiguous = t.to_concrete();
                let result = fusor_cpu::softmax_last_dim_fused(contiguous.inner());
                fusor_cpu::Tensor::new(result)
            },
            |t| t.softmax_last_dim::<OUT_RANK>(),
        )
    }
}

#[cfg(test)]
#[allow(clippy::identity_op, clippy::useless_conversion)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_softmax_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([6], &data));

        let result = t.softmax::<0>(0);
        let slice = result.as_slice().await.unwrap();

        // Compute expected softmax
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = data.iter().map(|x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_vals.iter().sum();
        let expected: Vec<f32> = exp_vals.iter().map(|x| x / sum_exp).collect();

        for i in 0..6 {
            assert!(
                (slice[[i]] - expected[i]).abs() < 0.001,
                "Mismatch at index {}: got {}, expected {}",
                i,
                slice[[i]],
                expected[i]
            );
        }

        // Check that softmax sums to 1
        let sum: f32 = (0..6).map(|i| slice[[i]]).sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_softmax_2d_cpu() {
        // 2x3 tensor, softmax along axis 1
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));

        let result = t.softmax::<1>(1);
        let slice = result.as_slice().await.unwrap();

        // Each row should sum to 1
        let sum_row0: f32 = (0..3).map(|j| slice[[0, j]]).sum();
        let sum_row1: f32 = (0..3).map(|j| slice[[1, j]]).sum();
        assert!((sum_row0 - 1.0).abs() < 0.001);
        assert!((sum_row1 - 1.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_rms_norm_cpu() {
        // Simple test: normalize [1, 2, 3] with weight [1, 1, 1]
        let data = [1.0f32, 2.0, 3.0];
        let t: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &data));
        let weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &[1.0, 1.0, 1.0]));

        let result = t.rms_norm::<0, _>(&weight, 1e-5);
        let slice = result.as_slice().await.unwrap();

        // rms = sqrt(mean([1, 4, 9])) = sqrt(14/3) ~ 2.16
        let rms = ((1.0 + 4.0 + 9.0) / 3.0_f32 + 1e-5).sqrt();
        let expected: Vec<f32> = data.iter().map(|x| x / rms).collect();

        for i in 0..3 {
            assert!(
                (slice[[i]] - expected[i]).abs() < 0.01,
                "Mismatch at index {}: got {}, expected {}",
                i,
                slice[[i]],
                expected[i]
            );
        }
    }

    #[tokio::test]
    async fn test_rms_norm_2d_cpu() {
        // 2x3 tensor
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));
        let weight: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice(
            [2, 3],
            &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ));

        let result = t.rms_norm::<1, _>(&weight, 1e-5);
        let slice = result.as_slice().await.unwrap();

        // Row 0: [1, 2, 3] -> rms = sqrt((1+4+9)/3 + eps) = sqrt(14/3 + eps)
        let rms0 = ((1.0 + 4.0 + 9.0) / 3.0_f32 + 1e-5).sqrt();
        assert!((slice[[0, 0]] - 1.0 / rms0).abs() < 0.01);
        assert!((slice[[0, 1]] - 2.0 / rms0).abs() < 0.01);
        assert!((slice[[0, 2]] - 3.0 / rms0).abs() < 0.01);

        // Row 1: [4, 5, 6] -> rms = sqrt((16+25+36)/3 + eps) = sqrt(77/3 + eps)
        let rms1 = ((16.0 + 25.0 + 36.0) / 3.0_f32 + 1e-5).sqrt();
        assert!((slice[[1, 0]] - 4.0 / rms1).abs() < 0.01);
        assert!((slice[[1, 1]] - 5.0 / rms1).abs() < 0.01);
        assert!((slice[[1, 2]] - 6.0 / rms1).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_layer_norm_cpu() {
        // Simple test with remove_mean=false (RMS-like)
        let data = [1.0f32, 2.0, 3.0];
        let t: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &data));
        let weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &[2.0, 2.0, 2.0]));

        let result = t.layer_norm::<0, _, ConcreteTensor<_, _>>(&weight, None, 1e-5, false);
        let slice = result.as_slice().await.unwrap();

        // With remove_mean=false, this should be like rms_norm but with weight=2
        let rms = ((1.0 + 4.0 + 9.0) / 3.0_f32 + 1e-5).sqrt();
        let expected: Vec<f32> = data.iter().map(|x| (x / rms) * 2.0).collect();

        for i in 0..3 {
            assert!(
                (slice[[i]] - expected[i]).abs() < 0.01,
                "Mismatch at index {}: got {}, expected {}",
                i,
                slice[[i]],
                expected[i]
            );
        }
    }

    #[tokio::test]
    async fn test_layer_norm_with_mean_removal() {
        // Test with remove_mean=true (standard layer norm)
        let data = [1.0f32, 2.0, 3.0];
        let t: Tensor<1, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &data));
        let weight: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([3], &[1.0, 1.0, 1.0]));

        let result = t.layer_norm::<0, _, ConcreteTensor<_, _>>(&weight, None, 1e-5, true);
        let slice = result.as_slice().await.unwrap();

        // mean = 2, centered = [-1, 0, 1]
        // var = mean([1, 0, 1]) = 2/3
        // std = sqrt(2/3 + eps)
        let var: f32 = 2.0 / 3.0;
        let std = (var + 1e-5).sqrt();
        let expected: Vec<f32> = vec![(-1.0) / std, 0.0 / std, 1.0 / std];

        for i in 0..3 {
            assert!(
                (slice[[i]] - expected[i]).abs() < 0.01,
                "Mismatch at index {}: got {}, expected {}",
                i,
                slice[[i]],
                expected[i]
            );
        }
    }

    #[tokio::test]
    async fn test_softmax_cpu_vs_gpu() {
        use crate::Device;

        // Create random-ish data similar to attention scores
        let data: Vec<f32> = (0..1 * 8 * 100 * 100)
            .map(|i| (i as f32 * 0.001).sin() * 10.0)
            .collect();

        // CPU version
        let cpu_tensor: Tensor<4, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 8, 100, 100], &data));
        let cpu_result = cpu_tensor.softmax::<3>(3);
        let cpu_slice = cpu_result.as_slice().await.unwrap();

        // GPU version
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let gpu_tensor: Tensor<4, f32> = Tensor::from_slice(&gpu_device, [1, 8, 100, 100], &data);
        let gpu_result = gpu_tensor.softmax::<3>(3);
        let gpu_slice = gpu_result.as_slice().await.unwrap();

        // Compare
        assert_eq!(cpu_slice.shape(), gpu_slice.shape());

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        let mut count = 0;
        for i in 0..cpu_slice.shape()[0] {
            for j in 0..cpu_slice.shape()[1] {
                for k in 0..cpu_slice.shape()[2].min(50) {
                    for l in 0..cpu_slice.shape()[3].min(50) {
                        let cpu_val: f32 = cpu_slice[[i, j, k, l]].into();
                        let gpu_val: f32 = gpu_slice[[i, j, k, l]].into();
                        let diff = (cpu_val - gpu_val).abs();
                        max_diff = max_diff.max(diff);
                        sum_diff += diff;
                        count += 1;
                    }
                }
            }
        }

        eprintln!(
            "Softmax CPU vs GPU: max_diff={}, mean_diff={}",
            max_diff,
            sum_diff / count as f32
        );
        eprintln!(
            "CPU[0,0,0,0..5]: {:?}",
            (0..5)
                .map(|i| cpu_slice[[0, 0, 0, i]])
                .collect::<Vec<f32>>()
        );
        eprintln!(
            "GPU[0,0,0,0..5]: {:?}",
            (0..5)
                .map(|i| gpu_slice[[0, 0, 0, i]])
                .collect::<Vec<f32>>()
        );

        assert!(
            max_diff < 0.001,
            "Softmax CPU and GPU outputs differ too much: max_diff={}",
            max_diff
        );
    }

    #[tokio::test]
    async fn test_layer_norm_cpu_vs_gpu() {
        use crate::Device;

        // Create random-ish data similar to hidden states
        let data: Vec<f32> = (0..1 * 100 * 384)
            .map(|i| (i as f32 * 0.001).sin() * 2.0)
            .collect();
        let weight_data: Vec<f32> = (0..384)
            .map(|i| 0.9 + (i as f32 * 0.001).cos() * 0.2)
            .collect();
        let bias_data: Vec<f32> = (0..384).map(|i| (i as f32 * 0.0001).sin() * 0.1).collect();

        // CPU version
        let cpu_tensor: Tensor<3, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 100, 384], &data));
        let cpu_weight_1d: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([384], &weight_data));
        let cpu_weight_broadcast = cpu_weight_1d.broadcast_as([1, 100, 384]);
        let cpu_bias: Tensor<1, f32> =
            Tensor::Cpu(fusor_cpu::Tensor::from_slice([384], &bias_data));
        let cpu_bias_broadcast = cpu_bias.broadcast_as([1, 100, 384]);
        let cpu_result = cpu_tensor.layer_norm::<2, _, _>(
            &cpu_weight_broadcast,
            Some(&cpu_bias_broadcast),
            1e-5,
            true,
        );
        let cpu_slice = cpu_result.as_slice().await.unwrap();

        // GPU version
        let gpu_device = Device::new().await.expect("GPU required for this test");
        let gpu_tensor: Tensor<3, f32> = Tensor::from_slice(&gpu_device, [1, 100, 384], &data);
        let gpu_weight_1d: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [384], &weight_data);
        let gpu_weight_broadcast = gpu_weight_1d.broadcast_as([1, 100, 384]);
        let gpu_bias: Tensor<1, f32> = Tensor::from_slice(&gpu_device, [384], &bias_data);
        let gpu_bias_broadcast = gpu_bias.broadcast_as([1, 100, 384]);
        let gpu_result = gpu_tensor.layer_norm::<2, _, _>(
            &gpu_weight_broadcast,
            Some(&gpu_bias_broadcast),
            1e-5,
            true,
        );
        let gpu_slice = gpu_result.as_slice().await.unwrap();

        // Compare
        assert_eq!(cpu_slice.shape(), gpu_slice.shape());

        let mut max_diff = 0.0f32;
        let mut sum_diff = 0.0f32;
        let mut count = 0;
        for i in 0..cpu_slice.shape()[0] {
            for j in 0..cpu_slice.shape()[1].min(50) {
                for k in 0..cpu_slice.shape()[2].min(100) {
                    let cpu_val: f32 = cpu_slice[[i, j, k]].into();
                    let gpu_val: f32 = gpu_slice[[i, j, k]].into();
                    let diff = (cpu_val - gpu_val).abs();
                    max_diff = max_diff.max(diff);
                    sum_diff += diff;
                    count += 1;
                }
            }
        }

        eprintln!(
            "LayerNorm CPU vs GPU: max_diff={}, mean_diff={}",
            max_diff,
            sum_diff / count as f32
        );
        eprintln!(
            "CPU[0,0,0..5]: {:?}",
            (0..5).map(|i| cpu_slice[[0, 0, i]]).collect::<Vec<f32>>()
        );
        eprintln!(
            "GPU[0,0,0..5]: {:?}",
            (0..5).map(|i| gpu_slice[[0, 0, i]]).collect::<Vec<f32>>()
        );

        assert!(
            max_diff < 0.01,
            "LayerNorm CPU and GPU outputs differ too much: max_diff={}",
            max_diff
        );
    }
}
