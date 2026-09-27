//! The op library, as methods.
//!
//! Every method here calls the corresponding free function at the
//! runtime-rank layer; no math is re-implemented.

use crate::cache::MaskKind;
use crate::composite::{PoolReduce, PoolSize, RopeLayout, RopePos, attention, rope, upsample};
use crate::quantized::QMatrix;
use crate::tensor::typed::{Axis, Element, Tensor, narrow_acc};

impl<const R: usize, T: Element> Tensor<R, T> {
    forward! {
        /// Softmax over `axis`. Rank- and dtype-preserving.
        fn softmax(axis: impl Axis<R> => ax32) -> Self;
        /// Softmax over the last axis.
        fn softmax_last_dim() -> Self;
        /// `log(softmax(x))` over `axis`, evaluated stably.
        fn log_softmax(axis: impl Axis<R> => ax32) -> Self;
        /// `x / sqrt(mean(x^2) + eps) * weight` over the last axis.
        fn rms_norm[const W: usize](weight: &Tensor<W, T> => t, eps: f32 => v) -> Self;
        /// [`Tensor::rms_norm`] with no learned scale.
        fn rms_norm_no_weight(eps: f32 => v) -> Self;
        /// `rms_norm(self + residual)` as one node: the add is inside the
        /// norm's expansion, which the residual-norm kernel reads.
        fn rms_norm_residual[const W: usize](
            residual: &Self => t,
            weight: &Tensor<W, T> => t,
            bias: Option<&Tensor<W, T>> => opt,
            eps: f32 => v,
        ) -> Self;
        /// `(x - mean) / sqrt(var + eps) * weight + bias` over the last axis;
        /// `remove_mean == false` is the RMS-like spelling.
        fn layer_norm[const W: usize](
            weight: &Tensor<W, T> => t,
            bias: Option<&Tensor<W, T>> => opt,
            eps: f32 => v,
            remove_mean: bool => v,
        ) -> Self;
        /// Scaled dot-product attention, `self` being the queries. `scale:
        /// None` is `1/sqrt(d)`; grouped-query attention is inferred from the
        /// head counts.
        fn attention(k: &Self => t, v: &Self => t, mask: MaskKind => v, scale: Option<f32> => v)
            -> Self = attention::attention;
        /// Attention with causality encoded structurally: no mask tensor, and
        /// the upper triangle is never computed.
        fn attention_causal(k: &Self => t, v: &Self => t, scale: Option<f32> => v)
            -> Self = attention::attention_causal;
        /// Attention against a materialized additive mask of rank `MR`.
        fn attention_masked[const MR: usize](
            k: &Self => t,
            v: &Self => t,
            mask: MaskKind => v,
            mask_tensor: Option<&Tensor<MR, T>> => opt,
            scale: Option<f32> => v,
        ) -> Self = attention::attention_masked;
    }

    /// Rotary embedding against `[context, head_dim/2]` tables.
    #[track_caller]
    pub fn rope(
        &self,
        cos: &Tensor<2, T>,
        sin: &Tensor<2, T>,
        layout: RopeLayout,
        pos: RopePos<&Tensor<1, u32>>,
    ) -> Self {
        Self::wrap(
            "rope",
            rope::rope(
                self.as_dyn(),
                cos.as_dyn(),
                sin.as_dyn(),
                layout,
                pos.map(Tensor::as_dyn),
            ),
        )
    }

    /// [`Tensor::rope`] on `self` and `k` in one node, handing back two
    /// views of it: q and k share the table read and the rotation.
    #[track_caller]
    pub fn rope_pair(
        &self,
        k: &Self,
        cos: &Tensor<2, T>,
        sin: &Tensor<2, T>,
        layout: RopeLayout,
        pos: RopePos<&Tensor<1, u32>>,
    ) -> (Self, Self) {
        let (q, k) = crate::device::ok(
            "rope_pair",
            rope::rope_pair(
                self.as_dyn(),
                k.as_dyn(),
                cos.as_dyn(),
                sin.as_dyn(),
                layout,
                pos.map(Tensor::as_dyn),
            ),
        );
        (
            Self::wrap("rope_pair q", Ok(q)),
            Self::wrap("rope_pair k", Ok(k)),
        )
    }
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Window the trailing `DIFF` axes and reduce each window. Rank-preserving.
    ///
    /// The reduction is a [`PoolReduce`] value so the node can carry it as an
    /// attribute and its adjoint can read it.
    #[track_caller]
    pub fn pool<const DIFF: usize>(
        &self,
        pools: [impl Into<PoolSize>; DIFF],
        with: PoolReduce,
    ) -> Self {
        let pools: [PoolSize; DIFF] = pools.map(Into::into);
        Self::wrap(
            "pool",
            narrow_acc::<T>(crate::composite::pool::pool(self.as_dyn(), &pools, with)),
        )
    }

    /// Max pooling over the trailing `DIFF` axes.
    #[track_caller]
    pub fn pool_max<const DIFF: usize>(&self, pools: [impl Into<PoolSize>; DIFF]) -> Self {
        self.pool(pools, PoolReduce::Max)
    }

    /// Min pooling over the trailing `DIFF` axes.
    #[track_caller]
    pub fn pool_min<const DIFF: usize>(&self, pools: [impl Into<PoolSize>; DIFF]) -> Self {
        self.pool(pools, PoolReduce::Min)
    }

    /// Average pooling over the trailing `DIFF` axes.
    #[track_caller]
    pub fn pool_avg<const DIFF: usize>(&self, pools: [impl Into<PoolSize>; DIFF]) -> Self {
        self.pool(pools, PoolReduce::Mean)
    }
}

impl<T: Element> Tensor<4, T> {
    /// Nearest-neighbour upsample of a `[B, C, H, W]` value.
    #[track_caller]
    pub fn upsample_nearest2d(&self, scale_h: usize, scale_w: usize) -> Self {
        let (h, w) = (
            u32::try_from(scale_h).expect("upsample scale fits u32"),
            u32::try_from(scale_w).expect("upsample scale fits u32"),
        );
        Self::wrap(
            "upsample_nearest2d",
            upsample::upsample_nearest2d(self.as_dyn(), h, w),
        )
    }

    /// Bilinear resample of a `[B, C, H, W]` value to `[h, w]`.
    #[track_caller]
    pub fn upsample_bilinear(&self, size: [u64; 2], align_corners: bool) -> Self {
        let size = size.map(fusor_ir::shape::Dim::Const);
        Self::wrap(
            "upsample_bilinear",
            upsample::upsample_bilinear(self.as_dyn(), &size, align_corners),
        )
    }
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// `self @ weights^T`, reading the block-quantized weight in place.
    /// The receiver is the activation.
    ///
    /// A rank-1 activation is one matrix row and routes through a `[1, k]`
    /// view, so the output rank matches the input rank.
    #[track_caller]
    pub fn q_mat_mul(&self, weights: &QMatrix) -> Self {
        Self::wrap(
            "q_mat_mul",
            narrow_acc::<T>(weights.q_mat_mul(self.as_dyn())),
        )
    }
}

impl QMatrix {
    /// Row lookup against a block-quantized table: `[.., n]` of ids against a
    /// `[vocab, dim]` matrix gives `[.., n, dim]`, so `O = IDS + 1`.
    ///
    /// The const-rank spelling of [`QMatrix::index_select_rows`], and the
    /// counterpart of [`Tensor::embedding`] for a quantized table. The gather
    /// decodes straight to the requested element type.
    #[track_caller]
    pub fn embedding<const IDS: usize, const O: usize, T: Element>(
        &self,
        ids: &Tensor<IDS, u32>,
    ) -> Tensor<O, T> {
        Tensor::<O, T>::wrap(
            "QMatrix::embedding",
            self.index_select_rows_to(ids.as_dyn(), T::DTYPE),
        )
    }
}
