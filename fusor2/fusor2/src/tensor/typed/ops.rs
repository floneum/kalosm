//! Views, indexing and readback on the const-rank tensor.
//!
//! Every method here wraps the [`Dyn`] implementation of the same name and
//! does no arithmetic of its own: the typed layer is a `repr(transparent)`
//! newtype and its whole job is to assert the rank, resolve the axis and
//! panic instead of returning `Result`. If a value here disagrees with the
//! `Dyn` one, the bug is in `crate::ops`, not in this file.
//!
//! Output rank is a const parameter. Axes are `impl Axis<R>` so `Minus1` goes
//! anywhere a `usize` does. Arrays, not slices, where the rank is known, so
//! `repeat([2, 1, 3])` on a rank-3 value cannot be given the wrong length.

use fusor2_ir::shape::Dim;

use crate::device::ok;
use crate::tensor::readback::TensorSlice;
use crate::tensor::typed::{Axis, Element, Tensor, dims_of};
use crate::Result;

/// A padding argument for [`Tensor::pad_axis`]: `usize` pads both sides
/// equally, `(left, right)` pads each independently.
pub trait PadWidths {
    fn widths(self) -> (usize, usize);
}
impl PadWidths for usize {
    fn widths(self) -> (usize, usize) {
        (self, self)
    }
}
impl PadWidths for (usize, usize) {
    fn widths(self) -> (usize, usize) {
        self
    }
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Reshape against extents that may still be symbolic.
    ///
    /// The 91 model call sites that spell `reshape_dims(&[b, Dim::Sym(seq),
    /// h, d])` are the decode loop keeping one plan across sequence lengths.
    /// [`Tensor::reshape`] is the all-constant form and
    /// [`Tensor::reshape_extents`] the one with an inferred hole.
    #[track_caller]
    pub fn reshape_dims<const O: usize>(&self, shape: [Dim; O]) -> Tensor<O, T> {
        Self::wrap("reshape_dims", self.as_dyn().reshape_dims(&shape))
    }

    /// Broadcast against extents that may still be symbolic.
    #[track_caller]
    pub fn broadcast_dims<const O: usize>(&self, shape: [Dim; O]) -> Tensor<O, T> {
        Self::wrap("broadcast_dims", self.as_dyn().broadcast_as(&shape))
    }

    /// Fold the last `from_end + 1` axes into one; output rank
    /// `O = R - from_end`.
    #[track_caller]
    pub fn flatten_last_n<const O: usize>(&self, from_end: usize) -> Tensor<O, T> {
        Self::wrap("flatten_last_n", self.as_dyn().flatten_last_n(from_end))
    }

    /// Fold the first `from_start + 1` axes into one; output rank
    /// `O = R - from_start`.
    #[track_caller]
    pub fn flatten_first_n<const O: usize>(&self, from_start: usize) -> Tensor<O, T> {
        Self::wrap("flatten_first_n", self.as_dyn().flatten_first_n(from_start))
    }

    /// Fold axes `from..=to` into one; output rank `O = R - (to - from)`.
    #[track_caller]
    pub fn flatten<const O: usize>(&self, from: impl Axis<R>, to: impl Axis<R>) -> Tensor<O, T> {
        Self::wrap(
            "flatten",
            self.as_dyn().flatten(from.resolve(), to.resolve()),
        )
    }

    /// Drop several length-1 axes at once; output rank `O = R - DIFF`.
    #[track_caller]
    pub fn squeeze_dims<const DIFF: usize, const O: usize>(
        &self,
        axes: [usize; DIFF],
    ) -> Tensor<O, T> {
        Self::wrap("squeeze_dims", self.as_dyn().squeeze_dims(&axes))
    }

    /// Insert several length-1 axes at once; output rank `O = R + DIFF`. The
    /// positions are in the *output*.
    #[track_caller]
    pub fn unsqueeze_dims<const DIFF: usize, const O: usize>(
        &self,
        axes: [usize; DIFF],
    ) -> Tensor<O, T> {
        Self::wrap("unsqueeze_dims", self.as_dyn().unsqueeze_dims(&axes))
    }

    /// Tile each axis `repeats[i]` times. Rank-preserving.
    #[track_caller]
    pub fn repeat(&self, repeats: [usize; R]) -> Self {
        Self::wrap("repeat", self.as_dyn().repeat(&repeats))
    }

    /// Pad or truncate each axis to `new_shape`, zero-filling any growth.
    /// Rank-preserving.
    #[track_caller]
    pub fn resize(&self, new_shape: [usize; R]) -> Self {
        Self::wrap("resize", self.as_dyn().resize(&dims_of(new_shape)))
    }

    /// [`Tensor::resize`] against extents that may be symbolic.
    #[track_caller]
    pub fn resize_dims(&self, new_shape: [Dim; R]) -> Self {
        Self::wrap("resize_dims", self.as_dyn().resize(&new_shape))
    }

    /// Zero-pad one axis: a bare `usize` pads both sides (the reference's
    /// spelling), a `(left, right)` pair pads each independently.
    #[track_caller]
    pub fn pad_axis(&self, axis: impl Axis<R>, padding: impl PadWidths) -> Self {
        Self::wrap(
            "pad_axis",
            self.as_dyn().pad_axis(axis.resolve(), padding.widths()),
        )
    }

    /// [`Tensor::pad_axis`] with the two sides spelled out explicitly.
    #[track_caller]
    pub fn pad_with_zeros(&self, axis: impl Axis<R>, left: usize, right: usize) -> Self {
        Self::wrap(
            "pad_with_zeros",
            self.as_dyn().pad_with_zeros(axis.resolve(), left, right),
        )
    }

    /// A sliding window over one axis; output rank `O = R + 1`.
    #[track_caller]
    pub fn windows<const O: usize>(&self, axis: impl Axis<R>, window: u32, step: u32) -> Tensor<O, T> {
        Self::wrap(
            "windows",
            self.as_dyn().windows(axis.resolve() as u32, window, step),
        )
    }
}

// ---------------------------------------------------------------------------
// Indexing
// ---------------------------------------------------------------------------

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Row lookup: `[.., n]` of ids against a `[vocab, dim]` table gives
    /// `[.., n, dim]`, so `O = IDS + 1`.
    ///
    /// The receiver is the *table*. [`Dyn::embedding`] is the same gather with
    /// the axis named.
    #[track_caller]
    pub fn embedding<const IDS: usize, const O: usize>(
        &self,
        ids: &Tensor<IDS, u32>,
    ) -> Tensor<O, T> {
        Self::wrap("embedding", self.as_dyn().embedding(ids.as_dyn()))
    }

    /// Gather along the last axis with a same-rank index value.
    #[track_caller]
    pub fn gather_last(&self, idx: &Tensor<R, u32>) -> Self {
        Self::wrap("gather_last", self.as_dyn().gather_last(idx.as_dyn()))
    }

    /// Write `value` into `ranges`, returning the updated value.
    #[track_caller]
    pub fn slice_assign(&self, ranges: [std::ops::Range<usize>; R], value: &Self) -> Self {
        Self::wrap(
            "slice_assign",
            self.as_dyn().slice_assign(&ranges, value.as_dyn()),
        )
    }
}

// ---------------------------------------------------------------------------
// Readback
// ---------------------------------------------------------------------------

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Read back as `f32`, converting if the value is not already f32.
    ///
    /// [`Tensor::to_flat`] is same-dtype: it reads `T` out of a `T`. A model
    /// that computes in f16 and reports in f32 wants this one, and so does
    /// every logits readback of a quantized model.
    #[track_caller]
    pub fn to_vec_f32(&self) -> Vec<f32> {
        ok("to_vec_f32", self.as_dyn().to_vec_f32())
    }

    /// Read back as `u32`, converting.
    #[track_caller]
    pub fn to_vec_u32(&self) -> Vec<u32> {
        ok("to_vec_u32", self.as_dyn().to_vec_u32())
    }

    /// Read back as `i32`, converting.
    #[track_caller]
    pub fn to_vec_i32(&self) -> Vec<i32> {
        ok("to_vec_i32", self.as_dyn().to_vec_i32())
    }

    /// The raw bytes of the value at its own dtype.
    #[track_caller]
    pub fn to_bytes(&self) -> Vec<u8> {
        ok("to_bytes", self.as_dyn().to_bytes())
    }

    /// [`Tensor::to_vec_f32`] behind the future a runtime awaits.
    ///
    /// Returns `Result` rather than panicking, unlike the rest of this
    /// surface: an `await` point is exactly where a caller *does* have
    /// somewhere to put the error, and `rbert` threads one.
    pub fn to_vec_f32_async(&self) -> impl Future<Output = Result<Vec<f32>>> + 'static {
        let slice: Result<TensorSlice> = self.as_dyn().as_slice();
        std::future::ready(slice.and_then(|s| s.to_vec_f32()))
    }

    /// [`Tensor::to_flat`] behind the same future.
    pub fn to_flat_async(&self) -> impl Future<Output = Result<Vec<T>>> + 'static {
        let slice: Result<TensorSlice> = self.as_dyn().as_slice();
        std::future::ready(slice.and_then(|s| s.to_flat::<T>()))
    }
}

/// Rank-1 helpers a sampler and a tokenizer reach for.
impl<T: Element> Tensor<1, T> {
    /// The `k` largest values of a rank-1 value and the indices they sat at.
    ///
    /// The pair is the sampler's contract: `top_k_pairs` is one kernel that
    /// produces both, so asking for them separately would resolve twice.
    #[track_caller]
    pub fn top_k(&self, k: u32) -> (Tensor<1, T>, Tensor<1, u32>) {
        let (values, indices) = ok("top_k", crate::sampling::top_k_pairs(self.as_dyn(), k));
        (
            Self::wrap("top_k values", Ok(values)),
            Self::wrap("top_k indices", Ok(indices)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;
    use crate::tensor::typed::Minus1;

    fn rows(device: &Device) -> Tensor<2, f32> {
        Tensor::<2, f32>::from_slice(device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }

    /// The view family lands on the output rank its call site named and gives
    /// the same values the runtime-rank op does.
    #[test]
    fn the_view_family_lands_on_its_output_parameter() {
        let device = Device::private();
        let a = rows(&device);

        let flat: Tensor<1, f32> = a.flatten_last_n(1);
        assert_eq!(flat.shape(), [6]);
        assert_eq!(flat.to_flat(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let cube: Tensor<3, f32> = a.reshape([1, 2, 3]);
        let back: Tensor<2, f32> = cube.flatten_first_n(1);
        assert_eq!(back.shape(), [2, 3]);
        let joined: Tensor<2, f32> = cube.flatten(1usize, 2usize);
        assert_eq!(joined.shape(), [1, 6]);

        let padded: Tensor<4, f32> = cube.unsqueeze_dims::<1, 4>([3]);
        assert_eq!(padded.shape(), [1, 2, 3, 1]);
        let dropped: Tensor<2, f32> = cube.squeeze_dims::<1, 2>([0]);
        assert_eq!(dropped.shape(), [2, 3]);
    }

    /// `reshape_dims` is the symbolic-extent spelling, and it is what keeps
    /// one plan across sequence lengths: a `Dim::Sym` must survive it.
    #[test]
    fn reshape_dims_carries_a_symbolic_extent_through() {
        let device = Device::private();
        let a = rows(&device);
        let [n, m] = a.extents();
        let same: Tensor<3, f32> = a.reshape_dims([Dim::Const(1), n, m]);
        assert_eq!(same.shape(), [1, 2, 3]);
        assert_eq!(same.extents(), [Dim::Const(1), Dim::Const(2), Dim::Const(3)]);
        assert_eq!(a.elem_count(), Some(6));
        assert_eq!(a.extent(Minus1), Dim::Const(3));
    }

    /// Padding, repeating and resizing keep the rank and mean what the
    /// runtime-rank ops mean.
    #[test]
    fn padding_repeating_and_resizing_agree_with_the_dyn_ops() {
        let device = Device::private();
        let a = Tensor::<1, f32>::from_slice(&device, [3], &[1.0, 2.0, 3.0]);

        assert_eq!(
            a.pad_with_zeros(0usize, 1, 2).to_flat(),
            vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
        );
        assert_eq!(
            a.pad_axis(Minus1, (0, 1)).to_flat(),
            vec![1.0, 2.0, 3.0, 0.0]
        );
        assert_eq!(a.repeat([2]).to_flat(), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(a.resize([2]).to_flat(), vec![1.0, 2.0]);
    }

    /// A conversion readback: the value is u32 and the caller wants f32. The
    /// same-dtype `to_flat` cannot express this at all.
    #[test]
    fn a_converting_readback_changes_the_dtype() {
        let device = Device::private();
        let ids = Tensor::<1, u32>::from_slice(&device, [3], &[7, 8, 9]);
        assert_eq!(ids.to_vec_f32(), vec![7.0, 8.0, 9.0]);
        assert_eq!(ids.to_vec_u32(), vec![7, 8, 9]);
        assert_eq!(
            pollster::block_on(ids.to_vec_f32_async()).unwrap(),
            vec![7.0, 8.0, 9.0]
        );
    }

    /// `set_bytes` replaces a leaf in place: the node id is unchanged, which
    /// is the whole reason a decode loop can keep one resolved plan.
    #[test]
    fn set_bytes_replaces_a_leaf_without_changing_the_node() {
        let device = Device::private();
        let ids = Tensor::<1, u32>::from_slice(&device, [2], &[1, 2]);
        let id = ids.id();
        ids.set_elements(&[30u32, 40]);
        assert_eq!(ids.id(), id, "set_bytes must not mint a new node");
        assert_eq!(ids.to_flat(), vec![30, 40]);
    }

    /// The table is the receiver of `embedding`, and the output rank is the
    /// ids' rank plus one.
    #[test]
    fn embedding_gathers_rows_of_the_receiver() {
        let device = Device::private();
        let table = Tensor::<2, f32>::from_slice(&device, [3, 2], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let ids = Tensor::<1, u32>::from_slice(&device, [2], &[2, 0]);
        let got: Tensor<2, f32> = table.embedding(&ids);
        assert_eq!(got.shape(), [2, 2]);
        assert_eq!(got.to_flat(), vec![4.0, 5.0, 0.0, 1.0]);
    }
}
