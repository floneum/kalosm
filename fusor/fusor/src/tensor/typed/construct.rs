//! Const-rank constructors.
//!
//! Everything here takes a `&Device` and mints a leaf: `Tensor::new(&device, [[1., 2.], [3., 4.]])`,
//! `Tensor::arange(&device, 0., 10.)`, `Tensor::full(&device, [2, 2], 1.5)`.
//! `zeros`, `ones`, `splat` and `from_slice` live in the parent module beside
//! the type.

use fusor_ir::dtype::Dtype;
use fusor_ir::shape::Dim;

use crate::device::Device;
use crate::tensor::Dyn;
use crate::tensor::construction::FromArray;
use crate::tensor::typed::{Element, Tensor, dims_of};

impl<const R: usize, T: Element> Tensor<R, T> {
    /// A value from a nested Rust array: `Tensor::new(&device, [[1., 2.]])`.
    #[track_caller]
    pub fn new<A: FromArray>(device: &Device, data: A) -> Self {
        Self::wrap("Tensor::new", Dyn::new(device.handle(), data))
    }

    /// Every element `value`.
    ///
    /// Takes shape before value, matching [`Tensor::zeros`] and [`Tensor::ones`].
    /// Folded into the kernel; never a buffer.
    #[track_caller]
    pub fn full(device: &Device, shape: [usize; R], value: T) -> Self {
        Self::wrap(
            "Tensor::full",
            Dyn::full(device.handle(), &dims_of(shape), value.splat()),
        )
    }

    /// Host bytes at a **runtime** dtype, at a known rank.
    ///
    /// This is the weight-load constructor. The bytes are interpreted at
    /// `dtype` and then cast to `T` when the two differ, so an f16 checkpoint
    /// loads into an f32 graph.
    ///
    /// The extents are [`Dim`], not `usize`: a symbolic one is legal here.
    #[track_caller]
    pub fn from_raw_bytes(device: &Device, dtype: Dtype, shape: [Dim; R], bytes: &[u8]) -> Self {
        Self::wrap(
            "Tensor::from_raw_bytes",
            Dyn::from_slice(device.handle(), dtype, &shape, bytes).and_then(|t| {
                if dtype == T::DTYPE {
                    Ok(t)
                } else {
                    t.cast(T::DTYPE)
                }
            }),
        )
    }

    /// A step-local input buffer whose contents the caller sets each step.
    ///
    /// One leaf per step-varying input, minted once, then
    /// [`Tensor::set_bytes`]/[`Tensor::set_elements`] per step. The node id
    /// never changes, so one resolved plan survives a whole generation. It is
    /// not registered as a parameter and carries no name.
    ///
    /// The extents are [`Dim`], not `usize`: a step input is exactly where a
    /// symbolic length shows up.
    #[track_caller]
    pub fn leaf(device: &Device, shape: [Dim; R]) -> Self {
        Self::wrap("Tensor::leaf", device.graph().leaf("", &shape, T::DTYPE))
    }

    /// A learnable parameter leaf, named so a checkpoint can find it again.
    #[track_caller]
    pub fn param(device: &Device, name: &str, shape: [usize; R]) -> Self {
        Self::wrap(
            "Tensor::param",
            Dyn::param(device.handle(), name, T::DTYPE, &dims_of(shape)),
        )
    }
}

impl<T: Element> Tensor<1, T> {
    /// `[start, start + 1, .., end)`.
    #[track_caller]
    pub fn arange(device: &Device, start: impl Into<f64>, end: impl Into<f64>) -> Self {
        Self::arange_step(device, start, end, 1.0)
    }

    /// `[start, start + step, .., end)`.
    ///
    /// The sequence is built on the host; the bounds are not kernel literals.
    #[track_caller]
    pub fn arange_step(
        device: &Device,
        start: impl Into<f64>,
        end: impl Into<f64>,
        step: impl Into<f64>,
    ) -> Self {
        Self::wrap(
            "Tensor::arange_step",
            crate::tensor::construction::arange_step(
                device.handle(),
                T::DTYPE,
                start.into(),
                end.into(),
                step.into(),
            ),
        )
    }
}

impl crate::quantized::QMatrix {
    /// The decoded `[rows, cols]` weight.
    #[track_caller]
    pub fn to_tensor(&self) -> Tensor<2, f32> {
        Tensor::<2, f32>::wrap("QMatrix::to_tensor", self.dequantize())
    }

    /// The rows named by `idx`, decoded. `[n]` in, `[n, cols]` out.
    #[track_caller]
    pub fn rows_at(&self, idx: &Tensor<1, u32>) -> Tensor<2, f32> {
        Tensor::<2, f32>::wrap("QMatrix::rows_at", self.index_select_rows(idx.as_dyn()))
    }
}
