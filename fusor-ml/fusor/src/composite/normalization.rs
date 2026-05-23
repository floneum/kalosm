//! Normalization operations that work on both CPU and GPU backends.

use crate::cpu::{LastRank as CpuLastRank, MaxOp, SimdReduceOp, SumOp, TensorBacking};
use crate::gpu::{
    DataType, FloatDataType, LastRank as GpuLastRank, NextRankInner as GpuNextRankInner,
};
use crate::{
    AddOp, ConcreteTensor, DivOp, ExpOp, FloatOps, MulOp, SimdBinaryOp, SimdElement, SimdUnaryOp,
    SqrtOp, SubOp, Tensor,
};

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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, f32>: GpuLastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: GpuLastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, f32>: GpuLastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
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
    /// This uses the same composite implementation for both CPU and GPU:
    /// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    pub fn softmax_slow<const OUT_RANK: usize>(&self, axis: usize) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, f32>: GpuLastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
        MaxOp: SimdReduceOp<D>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Sub<Output = D> + std::ops::Div<Output = D>,
        SubOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        // Unified implementation using composite ops for both backends
        self.softmax_cpu_impl(axis)
    }

    /// Slow softmax along the last dimension using composite operations.
    ///
    /// This is provided for API parity with fusor-core.
    pub fn softmax_slow_last_dim<const OUT_RANK: usize>(&self) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, f32>: GpuLastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, f32>: GpuLastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Add<Output = D>
            + crate::gpu::CastTensor<f32>,
        f32: crate::gpu::CastTensor<D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        (crate::gpu::Tensor<R, D>, crate::gpu::Tensor<W, D>): crate::gpu::MaxRank<R, D>,
    {
        match (self, weight, bias) {
            // GPU path - use the optimized fused kernel
            (Tensor::Gpu(input), Tensor::Gpu(gpu_weight), bias_opt) => {
                let gpu_bias = bias_opt.map(|b| match b {
                    Tensor::Gpu(bias) => bias,
                    _ => panic!("Bias must be on GPU when input is on GPU"),
                });
                Tensor::Gpu(input.rms_norm_fused::<W, OUT_RANK>(gpu_weight, gpu_bias, eps))
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, f32>: GpuLastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Add<Output = D>
            + crate::gpu::CastTensor<f32>,
        f32: crate::gpu::CastTensor<D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        (crate::gpu::Tensor<R, D>, crate::gpu::Tensor<W, D>): crate::gpu::MaxRank<R, D>,
    {
        self.rms_norm_fused::<W, OUT_RANK>(weight, None, eps)
    }

    /// Fused `(input + residual) -> RMSNorm` kernel for transformer block boundaries.
    pub fn rms_norm_residual_fused<const W: usize, const OUT_RANK: usize, B2>(
        &self,
        residual: &Tensor<R, D, B2>,
        weight: &Tensor<W, D, ConcreteTensor<D, W>>,
        bias: Option<&Tensor<W, D, ConcreteTensor<D, W>>>,
        eps: f32,
    ) -> Tensor<R, D>
    where
        ConcreteTensor<D, R>: CpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        crate::gpu::Tensor<R, f32>: GpuLastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
        SumOp: SimdReduceOp<D>,
        D: std::ops::Mul<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Add<Output = D>
            + crate::gpu::CastTensor<f32>,
        f32: crate::gpu::CastTensor<D>,
        MulOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        AddOp: SimdBinaryOp<D>,
        SqrtOp: SimdUnaryOp<D>,
        B2: TensorBacking<R, Elem = D>,
        (crate::gpu::Tensor<R, D>, crate::gpu::Tensor<W, D>): crate::gpu::MaxRank<R, D>,
    {
        match (self, residual, weight, bias) {
            (Tensor::Gpu(input), Tensor::Gpu(gpu_residual), Tensor::Gpu(gpu_weight), bias_opt) => {
                let gpu_bias = bias_opt.map(|b| match b {
                    Tensor::Gpu(bias) => bias,
                    _ => panic!("Bias must be on GPU when input is on GPU"),
                });
                Tensor::Gpu(input.rms_norm_residual_fused::<W, OUT_RANK>(
                    gpu_residual,
                    gpu_weight,
                    gpu_bias,
                    eps,
                ))
            }
            (Tensor::Cpu(_), Tensor::Cpu(_), Tensor::Cpu(_), _) => {
                let combined = (self + residual).to_concrete();
                combined.rms_norm_fused_cpu_impl::<W, OUT_RANK>(weight, bias, eps)
            }
            _ => panic!("All tensors must be on the same device"),
        }
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
        crate::gpu::Tensor<R, D>: GpuLastRank<OUT_RANK, D>,
        <crate::gpu::Tensor<R, D> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, D>>,
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

// Specialized f32 implementation with fused normalization helpers
impl<const R: usize, B> Tensor<R, f32, B>
where
    B: TensorBacking<R, Elem = f32>,
    crate::gpu::Tensor<R, f32>: crate::gpu::LastRankInner,
{
    /// Optimized fused layer norm along the last dimension for f32.
    ///
    /// `weight` and `bias` are expected to contain exactly one row of parameters
    /// for the last dimension. On CPU this uses the SIMD fused kernel after
    /// reshaping those parameters to `[1, ..., 1, last_dim]` and making them
    /// contiguous. On GPU it falls back to the common composite layer norm
    /// implementation after broadcasting the parameters to the input shape.
    pub fn layer_norm_last_dim_fused<const OUT_RANK: usize, const W: usize, B2, B3>(
        &self,
        weight: &Tensor<W, f32, B2>,
        bias: Option<&Tensor<W, f32, B3>>,
        eps: f32,
    ) -> Tensor<R, f32>
    where
        ConcreteTensor<f32, R>: CpuLastRank<OUT_RANK, f32>,
        crate::gpu::Tensor<R, f32>: GpuLastRank<OUT_RANK, f32>,
        <crate::gpu::Tensor<R, f32> as crate::gpu::LastRankInner>::LastRank:
            GpuNextRankInner<NextRank = crate::gpu::Tensor<R, f32>>,
        SumOp: SimdReduceOp<f32>,
        AddOp: SimdBinaryOp<f32>,
        SubOp: SimdBinaryOp<f32>,
        MulOp: SimdBinaryOp<f32>,
        DivOp: SimdBinaryOp<f32>,
        SqrtOp: SimdUnaryOp<f32>,
        B2: TensorBacking<W, Elem = f32>,
        B3: TensorBacking<W, Elem = f32>,
    {
        let last_dim = self.shape()[R - 1];
        let weight_elements = weight.shape().iter().product::<usize>();
        assert_eq!(
            weight_elements, last_dim,
            "layer_norm_last_dim_fused expects weight to contain exactly the last dimension"
        );
        if let Some(bias) = bias {
            let bias_elements = bias.shape().iter().product::<usize>();
            assert_eq!(
                bias_elements, last_dim,
                "layer_norm_last_dim_fused expects bias to contain exactly the last dimension"
            );
        }

        let mut param_shape = [1; R];
        param_shape[R - 1] = last_dim;

        match (self, weight) {
            (Tensor::Cpu(input), Tensor::Cpu(weight)) => {
                let input = input.as_ref().make_contiguous();
                let weight = weight.as_ref().reshape(param_shape).make_contiguous();
                let bias = match bias {
                    Some(Tensor::Cpu(bias)) => {
                        Some(bias.as_ref().reshape(param_shape).make_contiguous())
                    }
                    Some(Tensor::Gpu(_)) => {
                        panic!("Layer norm requires tensors on the same backend")
                    }
                    None => None,
                };
                let result = crate::cpu::layer_norm_last_dim_fused(
                    input.inner(),
                    weight.inner(),
                    bias.as_ref().map(|bias| bias.inner()),
                    eps,
                );
                Tensor::Cpu(crate::cpu::Tensor::new(result))
            }
            (Tensor::Gpu(_), Tensor::Gpu(_)) => {
                if matches!(bias, Some(Tensor::Cpu(_))) {
                    panic!("Layer norm requires tensors on the same backend");
                }
                let weight_params = weight.reshape(param_shape);
                let weight = weight_params.broadcast_as(self.shape());
                let bias_params = bias.map(|bias| bias.reshape(param_shape));
                let bias = bias_params
                    .as_ref()
                    .map(|bias| bias.broadcast_as(self.shape()));
                self.layer_norm::<OUT_RANK, _, _>(&weight, bias.as_ref(), eps, true)
            }
            _ => panic!("Layer norm requires tensors on the same backend"),
        }
    }

    /// Optimized fused softmax along the last dimension for f32.
    ///
    /// This performs the entire softmax (max, exp, sum, normalize) in a single
    /// pass through memory, which is significantly faster for large tensors.
    pub fn softmax_last_dim_fused<const OUT_RANK: usize>(&self) -> Tensor<R, f32>
    where
        crate::gpu::Tensor<R, f32>: crate::gpu::LastRank<OUT_RANK, f32>,
    {
        self.dispatch_ref(
            |t| {
                // Make contiguous if needed, then use fused kernel
                let contiguous = t.as_ref().make_contiguous();
                let result = crate::cpu::softmax_last_dim_fused(contiguous.inner());
                crate::cpu::Tensor::new(result)
            },
            |t| t.softmax_last_dim::<OUT_RANK>(),
        )
    }
}
