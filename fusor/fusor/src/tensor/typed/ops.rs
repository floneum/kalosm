//! Views, indexing and readback on the const-rank tensor.
//!
//! Every method here wraps the [`crate::tensor::Tensor`] implementation of
//! the same name and does no arithmetic of its own: the typed layer asserts
//! the rank, resolves the axis and panics instead of returning `Result`.
//!
//! Output rank is a const parameter. Axes are `impl Axis<R>` so `Minus1` goes
//! anywhere a `usize` does. Arrays, not slices, where the rank is known, so
//! `repeat([2, 1, 3])` on a rank-3 value cannot be given the wrong length.

use fusor_ir::shape::Dim;

use crate::Result;
use crate::device::ok;
use crate::tensor::typed::{Axis, Element, Tensor};

impl<const R: usize, T: Element> Tensor<R, T> {
    forward! {
        /// Reshape against extents that may still be symbolic, which is how
        /// the decode loop keeps one plan across sequence lengths.
        fn reshape_dims[const O: usize](shape: [Dim; O] => arr) -> Tensor<O, T>;
        /// Fold the last `from_end + 1` axes into one; `O = R - from_end`.
        fn flatten_last_n[const O: usize](from_end: usize => v) -> Tensor<O, T>;
        /// Fold the first `from_start + 1` axes into one; `O = R - from_start`.
        fn flatten_first_n[const O: usize](from_start: usize => v) -> Tensor<O, T>;
        /// Fold axes `from..=to` into one; `O = R - (to - from)`.
        fn flatten[const O: usize](from: impl Axis<R> => ax, to: impl Axis<R> => ax) -> Tensor<O, T>;
        /// Drop several length-1 axes at once; `O = R - DIFF`.
        fn squeeze_dims[const DIFF: usize, const O: usize](axes: [usize; DIFF] => arr) -> Tensor<O, T>;
        /// Insert several length-1 axes at once; `O = R + DIFF`. The positions
        /// are in the *output*.
        fn unsqueeze_dims[const DIFF: usize, const O: usize](axes: [usize; DIFF] => arr) -> Tensor<O, T>;
        /// Tile each axis `repeats[i]` times.
        fn repeat(repeats: [usize; R] => arr) -> Self;
        /// Pad or truncate each axis to `new_shape`, zero-filling any growth.
        fn resize(new_shape: [usize; R] => dims) -> Self;
        /// Zero-pad one axis by `left` before and `right` after.
        fn pad_with_zeros(axis: impl Axis<R> => ax, left: usize => v, right: usize => v) -> Self;
        /// A sliding window over one axis; `O = R + 1`.
        fn windows[const O: usize](axis: impl Axis<R> => ax32, window: u32 => v, step: u32 => v) -> Tensor<O, T>;
        /// Row lookup: `[.., n]` ids against a `[vocab, dim]` table (the
        /// receiver) gives `[.., n, dim]`, so `O = IDS + 1`.
        fn embedding[const IDS: usize, const O: usize](ids: &Tensor<IDS, u32> => t) -> Tensor<O, T>;
        /// Gather along the last axis with a same-rank index value.
        fn gather_last(idx: &Tensor<R, u32> => t) -> Self;
        /// Write `value` into `ranges`, returning the updated value.
        fn slice_assign(ranges: [std::ops::Range<usize>; R] => arr, value: &Self => t) -> Self;
    }
}

/// Converting readbacks, panicking like the rest of the const-rank API.
macro_rules! read_as {
    ($($(#[$m:meta])* $name:ident -> $out:ty;)*) => {
        impl<const R: usize, T: Element> Tensor<R, T> {$(
            $(#[$m])*
            #[track_caller]
            pub fn $name(&self) -> $out {
                ok(stringify!($name), self.as_dyn().$name())
            }
        )*}
    };
}

read_as! {
    /// Read back as `f32`, converting: the readback of a value computed in
    /// f16 or of a quantized model's logits.
    to_vec_f32 -> Vec<f32>;
    /// Read back as `u32`, converting.
    to_vec_u32 -> Vec<u32>;
    /// Read back as `i32`, converting.
    to_vec_i32 -> Vec<i32>;
    /// The raw bytes of the value at its own dtype.
    to_bytes -> Vec<u8>;
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// [`Tensor::to_vec_f32`] behind the future a runtime awaits.
    pub fn to_vec_f32_async(&self) -> impl Future<Output = Result<Vec<f32>>> + 'static {
        let value = self.as_dyn().clone();
        async move { value.to_vec_f32_async().await }
    }

    /// [`Tensor::to_flat`] behind the same future.
    pub fn to_flat_async(&self) -> impl Future<Output = Result<Vec<T>>> + 'static {
        let value = self.as_dyn().clone();
        async move { value.to_flat_async::<T>().await }
    }
}

/// Rank-1 helpers a sampler and a tokenizer reach for.
impl<T: Element> Tensor<1, T> {
    /// The `k` largest values of a rank-1 value and the indices they sat at,
    /// from one kernel.
    #[track_caller]
    pub fn top_k(&self, k: u32) -> (Tensor<1, T>, Tensor<1, u32>) {
        let (values, indices) = ok("top_k", crate::sampling::top_k_pairs(self.as_dyn(), k));
        (
            Self::wrap("top_k values", Ok(values)),
            Self::wrap("top_k indices", Ok(indices)),
        )
    }
}
