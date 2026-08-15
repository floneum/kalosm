//! Const-rank constructors.
//!
//! Everything here takes a `&Device` and mints a leaf: `Tensor::new(&device, [[1., 2.], [3., 4.]])`,
//! `Tensor::arange(&device, 0., 10.)`, `Tensor::full(&device, [2, 2], 1.5)`.
//! `zeros`, `ones`, `splat` and `from_slice` live in the parent module beside
//! the type.

use fusor2_ir::dtype::Dtype;
use fusor2_ir::shape::Dim;

use crate::device::Device;
use crate::tensor::construction::FromArray;
use crate::tensor::typed::{Element, Tensor, dims_of};
use crate::tensor::Dyn;

impl<const R: usize, T: Element> Tensor<R, T> {
    /// A value from a nested Rust array: `Tensor::new(&device, [[1., 2.]])`.
    ///
    /// The reason wrapping a runtime-rank value is spelled
    /// [`Tensor::from_dyn`] rather than `new` is so a model that ports by
    /// changing imports uses this one.
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

    /// [`Tensor::zeros`] against extents that may be symbolic.
    #[track_caller]
    pub fn zeros_dims(device: &Device, shape: [Dim; R]) -> Self {
        Self::wrap(
            "Tensor::zeros_dims",
            Dyn::zeros(device.handle(), T::DTYPE, &shape),
        )
    }

    /// Host bytes at a **runtime** dtype, at a known rank.
    ///
    /// This is the weight-load constructor. A GGUF entry's dtype is data — it
    /// is read from the file — while its rank is program structure the model
    /// knows, which is exactly the pair the const-rank type expresses and the
    /// `Dyn` layer does not. The bytes are interpreted at `dtype` and then
    /// cast to `T` when the two differ, so
    /// `Tensor::<2, f32>::from_raw_bytes(&d, raw.fmt, shape, &raw.bytes)`
    /// loads an f16 checkpoint into an f32 graph without the caller writing
    /// the cast at all 40 sites.
    ///
    /// The extents are [`Dim`], not `usize`: a symbolic one is legal here.
    #[track_caller]
    pub fn from_raw_bytes(
        device: &Device,
        dtype: Dtype,
        shape: [Dim; R],
        bytes: &[u8],
    ) -> Self {
        Self::wrap(
            "Tensor::from_raw_bytes",
            Dyn::from_slice(device.handle(), dtype, &shape, bytes)
                .and_then(|t| if dtype == T::DTYPE { Ok(t) } else { t.cast(T::DTYPE) }),
        )
    }

    /// A step-local input buffer whose contents the caller sets each step.
    ///
    /// The decode loop's spelling: one leaf per step-varying input, minted
    /// once, then [`Tensor::set_bytes`]/[`Tensor::set_elements`] per step. The
    /// node id never changes, which is what lets one resolved plan survive a
    /// whole generation. Unlike [`Tensor::param`] it is not registered as a
    /// parameter and carries no name — `Graph::leaf` discards the one it is
    /// handed.
    ///
    /// The extents are [`Dim`], not `usize`: a step input is exactly where a
    /// symbolic length shows up.
    #[track_caller]
    pub fn leaf(device: &Device, shape: [Dim; R]) -> Self {
        Self::wrap(
            "Tensor::leaf",
            device.graph().leaf("", &shape, T::DTYPE),
        )
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
    /// The bounds are `impl Into<f64>` rather than `T`. `f64` is used because the
    /// sequence is built **on the host** — these are not kernel literals.
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

// ---------------------------------------------------------------------------
// Quantized weights
// ---------------------------------------------------------------------------

impl crate::quantized::QMatrix {
    /// The decoded `[rows, cols]` weight.
    ///
    /// Const-rank because a `QMatrix` *is* rank 2 — the type carried `rows`
    /// and `cols` all along, so returning a runtime-rank value threw away
    /// something already known.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_nested_array_becomes_a_value_of_that_rank() {
        let device = Device::private();
        let a: Tensor<2, f32> = Tensor::new(&device, [[1.0f32, 2.0], [3.0, 4.0]]);
        assert_eq!(a.shape(), [2, 2]);
        assert_eq!(a.to_flat(), vec![1.0, 2.0, 3.0, 4.0]);

        let v: Tensor<1, f32> = Tensor::new(&device, [1.0f32, 2.0, 3.0]);
        assert_eq!(v.shape(), [3]);
    }

    #[test]
    fn full_is_splat_with_the_references_argument_order() {
        let device = Device::private();
        let a = Tensor::<2, f32>::full(&device, [2, 2], 1.5);
        assert_eq!(a.to_flat(), vec![1.5; 4]);
        assert_eq!(a.id(), Tensor::<2, f32>::splat(&device, 1.5, [2, 2]).id());
    }

    #[test]
    fn arange_counts_and_arange_step_strides() {
        let device = Device::private();
        let ids = Tensor::<1, u32>::arange(&device, 0.0, 4.0);
        assert_eq!(ids.to_flat(), vec![0u32, 1, 2, 3]);
        let half = Tensor::<1, f32>::arange_step(&device, 0.5, 3.0, 1.0);
        assert_eq!(half.to_flat(), vec![0.5, 1.5, 2.5]);
    }

    /// The loader constructor: bytes at a dtype read from a file, a rank the
    /// model knows, and a cast when the two dtypes differ.
    #[test]
    fn from_raw_bytes_casts_a_checkpoint_dtype_to_the_graph_dtype() {
        let device = Device::private();
        let halves: Vec<half::f16> = [1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        let bytes: &[u8] = bytemuck::cast_slice(&halves);

        // Same dtype: no cast, and the leaf is the value itself.
        let as_f16 = Tensor::<2, half::f16>::from_raw_bytes(
            &device,
            Dtype::F16,
            [Dim::Const(2), Dim::Const(2)],
            bytes,
        );
        assert_eq!(as_f16.dtype(), Dtype::F16);

        // Different dtype: the cast is written once, here.
        let as_f32 = Tensor::<2, f32>::from_raw_bytes(
            &device,
            Dtype::F16,
            [Dim::Const(2), Dim::Const(2)],
            bytes,
        );
        assert_eq!(as_f32.dtype(), Dtype::F32);
        assert_eq!(as_f32.to_flat(), vec![1.0f32, 2.0, 3.0, 4.0]);
    }
}
