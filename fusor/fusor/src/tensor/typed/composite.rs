//! The op library, as methods.
//!
//! Every method here calls the corresponding free function at the [`Dyn`]
//! layer; no math is re-implemented.

use crate::cache::MaskKind;
use crate::composite::{PoolReduce, PoolSize, attention, rope, upsample};
use crate::quantized::QMatrix;
use crate::tensor::typed::{Axis, Element, Tensor, narrow_acc};

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Softmax over `axis`. Rank- and dtype-preserving.
    #[track_caller]
    pub fn softmax(&self, axis: impl Axis<R>) -> Self {
        Self::wrap("softmax", self.as_dyn().softmax(axis.resolve() as u32))
    }

    /// Softmax over the last axis.
    #[track_caller]
    pub fn softmax_last_dim(&self) -> Self {
        Self::wrap("softmax_last_dim", self.as_dyn().softmax_last_dim())
    }

    /// `log(softmax(x))` over `axis`, evaluated stably.
    #[track_caller]
    pub fn log_softmax(&self, axis: impl Axis<R>) -> Self {
        Self::wrap(
            "log_softmax",
            self.as_dyn().log_softmax(axis.resolve() as u32),
        )
    }

    /// `x / sqrt(mean(x^2) + eps) * weight` over the last axis.
    #[track_caller]
    pub fn rms_norm<const W: usize>(&self, weight: &Tensor<W, T>, eps: f32) -> Self {
        Self::wrap("rms_norm", self.as_dyn().rms_norm(weight.as_dyn(), eps))
    }

    /// [`Tensor::rms_norm`] with no learned scale.
    #[track_caller]
    pub fn rms_norm_no_weight(&self, eps: f32) -> Self {
        Self::wrap("rms_norm_no_weight", self.as_dyn().rms_norm_no_weight(eps))
    }

    /// `rms_norm(self + residual)` as one node.
    ///
    /// The add is inside the norm's expansion, so this is a different node
    /// than `(x + r).rms_norm(w, eps)`; the residual-norm kernel reads it.
    #[track_caller]
    pub fn rms_norm_residual<const W: usize>(
        &self,
        residual: &Self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
    ) -> Self {
        Self::wrap(
            "rms_norm_residual",
            self.as_dyn().rms_norm_residual(
                residual.as_dyn(),
                weight.as_dyn(),
                bias.map(Tensor::as_dyn),
                eps,
            ),
        )
    }

    /// `(x - mean) / sqrt(var + eps) * weight + bias` over the last axis.
    ///
    /// `remove_mean == false` gives the RMS-like spelling some checkpoints use.
    #[track_caller]
    pub fn layer_norm<const W: usize>(
        &self,
        weight: &Tensor<W, T>,
        bias: Option<&Tensor<W, T>>,
        eps: f32,
        remove_mean: bool,
    ) -> Self {
        Self::wrap(
            "layer_norm",
            self.as_dyn()
                .layer_norm(weight.as_dyn(), bias.map(Tensor::as_dyn), eps, remove_mean),
        )
    }
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Scaled dot-product attention, `self` being the queries.
    ///
    /// `scale: None` means the head dimension's `1/sqrt(d)`. Grouped-query
    /// attention is inferred from the head counts of `self` and `k`.
    #[track_caller]
    pub fn attention(&self, k: &Self, v: &Self, mask: MaskKind, scale: Option<f32>) -> Self {
        Self::wrap(
            "attention",
            attention::attention(self.as_dyn(), k.as_dyn(), v.as_dyn(), mask, scale),
        )
    }

    /// Attention with causality encoded structurally — no mask tensor is
    /// built, so the upper triangle is never computed.
    #[track_caller]
    pub fn attention_causal(&self, k: &Self, v: &Self, scale: Option<f32>) -> Self {
        Self::wrap(
            "attention_causal",
            attention::attention_causal(self.as_dyn(), k.as_dyn(), v.as_dyn(), scale),
        )
    }

    /// Attention against a materialized additive mask.
    ///
    /// The mask's rank is its own parameter: both `[Lq, Lk]` and
    /// `[B, 1, Lq, Lk]` masks are accepted.
    #[track_caller]
    pub fn attention_masked<const MR: usize>(
        &self,
        k: &Self,
        v: &Self,
        mask: MaskKind,
        mask_tensor: Option<&Tensor<MR, T>>,
        scale: Option<f32>,
    ) -> Self {
        Self::wrap(
            "attention_masked",
            attention::attention_masked(
                self.as_dyn(),
                k.as_dyn(),
                v.as_dyn(),
                mask,
                mask_tensor.map(Tensor::as_dyn),
                scale,
            ),
        )
    }
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Rotary embedding pairing `(i, i + Dh/2)` — the "normal" convention.
    #[track_caller]
    pub fn rope(&self, cos: &Tensor<2, T>, sin: &Tensor<2, T>, offset: u64) -> Self {
        Self::wrap(
            "rope",
            rope::rope(self.as_dyn(), cos.as_dyn(), sin.as_dyn(), offset),
        )
    }

    /// Rotary embedding pairing `(2i, 2i + 1)`.
    #[track_caller]
    pub fn rope_interleaved(&self, cos: &Tensor<2, T>, sin: &Tensor<2, T>, offset: u64) -> Self {
        Self::wrap(
            "rope_interleaved",
            rope::rope_interleaved(self.as_dyn(), cos.as_dyn(), sin.as_dyn(), offset),
        )
    }

    /// [`Tensor::rope`] on `self` and `k` in one node, handing back two
    /// views of it. q and k share the table read and the rotation.
    #[track_caller]
    pub fn rope_pair(
        &self,
        k: &Self,
        cos: &Tensor<2, T>,
        sin: &Tensor<2, T>,
        offset: u64,
    ) -> (Self, Self) {
        let (q, k) = crate::device::ok(
            "rope_pair",
            rope::rope_pair(
                self.as_dyn(),
                k.as_dyn(),
                cos.as_dyn(),
                sin.as_dyn(),
                offset,
            ),
        );
        (
            Self::wrap("rope_pair q", Ok(q)),
            Self::wrap("rope_pair k", Ok(k)),
        )
    }

    /// [`Tensor::rope_interleaved`] on `self` and `k` in one node.
    #[track_caller]
    pub fn rope_interleaved_pair(
        &self,
        k: &Self,
        cos: &Tensor<2, T>,
        sin: &Tensor<2, T>,
        offset: u64,
    ) -> (Self, Self) {
        let (q, k) = crate::device::ok(
            "rope_interleaved_pair",
            rope::rope_interleaved_pair(
                self.as_dyn(),
                k.as_dyn(),
                cos.as_dyn(),
                sin.as_dyn(),
                offset,
            ),
        );
        (
            Self::wrap("rope_interleaved_pair q", Ok(q)),
            Self::wrap("rope_interleaved_pair k", Ok(k)),
        )
    }

    /// [`Tensor::rope_pair`] against a device-side position vector.
    #[track_caller]
    pub fn rope_pair_at(
        &self,
        k: &Self,
        cos: &Tensor<2, T>,
        sin: &Tensor<2, T>,
        positions: &Tensor<1, u32>,
    ) -> (Self, Self) {
        let (q, k) = crate::device::ok(
            "rope_pair_at",
            rope::rope_pair_with_position(
                self.as_dyn(),
                k.as_dyn(),
                cos.as_dyn(),
                sin.as_dyn(),
                positions.as_dyn(),
            ),
        );
        (
            Self::wrap("rope_pair_at q", Ok(q)),
            Self::wrap("rope_pair_at k", Ok(k)),
        )
    }

    /// [`Tensor::rope_interleaved_pair`] against a device-side position
    /// vector.
    #[track_caller]
    pub fn rope_interleaved_pair_at(
        &self,
        k: &Self,
        cos: &Tensor<2, T>,
        sin: &Tensor<2, T>,
        positions: &Tensor<1, u32>,
    ) -> (Self, Self) {
        let (q, k) = crate::device::ok(
            "rope_interleaved_pair_at",
            rope::rope_interleaved_pair_with_position(
                self.as_dyn(),
                k.as_dyn(),
                cos.as_dyn(),
                sin.as_dyn(),
                positions.as_dyn(),
            ),
        );
        (
            Self::wrap("rope_interleaved_pair_at q", Ok(q)),
            Self::wrap("rope_interleaved_pair_at k", Ok(k)),
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
