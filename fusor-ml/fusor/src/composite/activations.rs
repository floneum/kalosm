//! Activation functions that work on both CPU and GPU backends.

use crate::gpu::{DataType, FloatDataType};
use crate::{
    AddOp, DivOp, ExpOp, FloatOps, MulOp, NegOp, SimdBinaryOp, SimdElement, SimdUnaryOp, TanhOp,
    Tensor,
};

/// Maximum |x| fed to `tanh` on GPU before WGSL's `(e^x - e^-x) / (e^x + e^-x)`
/// implementation overflows f32. tanh is saturated outside +/-10 anyway.
const TANH_INPUT_CLAMP: f32 = 15.0;
/// Lower clamp on `1 + tanh(x)`; mathematically the value lives in [0, 2] but
/// driver-specific tanh precision can drift slightly below 0.
const ONE_PLUS_TANH_MIN: f32 = 0.0;
/// Upper clamp on `1 + tanh(x)`; see `ONE_PLUS_TANH_MIN`.
const ONE_PLUS_TANH_MAX: f32 = 2.0;

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
{
    /// Rectified Linear Unit activation: relu(x) = max(0, x)
    pub fn relu(&self) -> Self {
        self.max_scalar(D::from_f32(0.0))
    }

    /// Sigmoid activation: sigmoid(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Tensor<R, D>
    where
        D: std::ops::Neg<Output = D>
            + std::ops::Add<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Mul<Output = D>
            + crate::cpu::Scalar,
        AddOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        NegOp: SimdUnaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        let neg_self = -self;
        let one_plus_exp = neg_self.exp() + D::from_f32(1.0);
        (self.ones_like() / one_plus_exp).to_concrete()
    }

    /// Sigmoid Linear Unit activation: silu(x) = x * sigmoid(x)
    pub fn silu(&self) -> Tensor<R, D>
    where
        D: std::ops::Neg<Output = D>
            + std::ops::Add<Output = D>
            + std::ops::Div<Output = D>
            + std::ops::Mul<Output = D>
            + crate::cpu::Scalar,
        AddOp: SimdBinaryOp<D>,
        DivOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        NegOp: SimdUnaryOp<D>,
        ExpOp: SimdUnaryOp<D>,
    {
        let neg_self = -self;
        let exp_neg = neg_self.exp();
        let one_plus_exp = exp_neg + D::from_f32(1.0);
        (self / one_plus_exp).to_concrete()
    }

    /// Gaussian Error Linear Unit activation (approximate).
    ///
    /// Uses the tanh approximation:
    /// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Self
    where
        AddOp: SimdBinaryOp<D>,
        MulOp: SimdBinaryOp<D>,
        TanhOp: SimdUnaryOp<D>,
        D: crate::cpu::Scalar,
    {
        let coeff = D::from_f32((2.0 / std::f32::consts::PI).sqrt());

        // x^2
        let x_squared = self * self;

        // 0.044715 * x^2 + 1.0
        let inner_factor = x_squared * D::from_f32(0.044715) + D::from_f32(1.0);

        // x * (1 + 0.044715 * x^2)
        let inner = self * &inner_factor;

        // sqrt(2/pi) * (x * (1 + 0.044715 * x^2))
        let tanh_input = inner * coeff;

        // WGSL's tanh(x) computes (e^x - e^-x)/(e^x + e^-x); e^x overflows f32
        // for x > ~88, producing NaN on GPU. For |x| > 10, tanh(x) ~= +/-1, so
        // clamping to TANH_INPUT_CLAMP is mathematically inert but prevents NaN.
        let tanh_input = tanh_input.clamp(
            D::from_f32(-TANH_INPUT_CLAMP),
            D::from_f32(TANH_INPUT_CLAMP),
        );
        // Avoid native tanh here: software renderers (WARP) can under-saturate
        // on GELU's negative tail, leaving visible non-zero outputs.
        let tanh_result = tanh_input
            .tanh_exact()
            .clamp(D::from_f32(-1.0), D::from_f32(1.0));

        // 1 + tanh(...) - mathematically in [0, 2]. Clamp defensively against
        // driver-specific tanh precision that can return values slightly outside [-1, 1].
        let one_plus_tanh = (&tanh_result + D::from_f32(1.0)).clamp(
            D::from_f32(ONE_PLUS_TANH_MIN),
            D::from_f32(ONE_PLUS_TANH_MAX),
        );

        // x * (1 + tanh(...))
        let product = self * &one_plus_tanh;

        // 0.5 * x * (1 + tanh(...))
        (product * D::from_f32(0.5)).to_concrete()
    }
}
