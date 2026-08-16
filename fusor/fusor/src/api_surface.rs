//! The intended public surface, restated as `use` lines.
//!
//! Every item the crate means to export is named here, so deleting or moving
//! one is a compile error in this crate rather than a discovery made by a
//! downstream build.
//!
//! This pins intended names and type identities. It cannot detect an extra
//! reachable export; `unreachable_pub` catches accidental exports behind a
//! private module, while review of this file and `API.md` governs additions to
//! the reachable surface.
//!
//! Test-only. It defines no public item and calls nothing.

#![allow(unused_imports, dead_code)]

// § 1. The root.
use crate::{
    Axis, Device, Dim, Dtype, Element, Error, Graph, Minus1, Minus2, QMatrix, Result, Session,
    ShardedVarBuilder, Tensor, ToVec, VarBuilder, cat, stack,
};

// § 2. Modules.
use crate::autograd::{
    BackwardTarget, GradientSlot, Gradients as AutogradGradients, Graph as AutogradGraph, Parent,
    Tensor as AutogradTensor, cat as autograd_cat,
};
use crate::cache::{AttentionMask, KvCache, MaskCache, MaskKind, RopeCache, TensorCache};
use crate::composite::{
    PoolReduce, PoolSize, attention, attention_causal, attention_grads, attention_lse,
    attention_masked, attention_with_lse, base_inverse_frequency, binary_cross_entropy_with_logits,
    conv, distillation_loss, grouped_conv, mse, pad_with_zeros, pool, pool_avg, pool_max, pool_min,
    rope, rope_interleaved, rope_interleaved_pair, rope_interleaved_pair_with_position,
    rope_interleaved_with_position, rope_pair, rope_pair_with_position, rope_with_position,
    rotate_half, softmax_cross_entropy, upsample_bilinear, upsample_nearest, upsample_nearest2d,
};
use crate::device::{Cpu, Gpu, KernelProfile, KernelProfileRow};
use crate::graph::{Gradients, GraphRef};
use crate::layers::{ConvNd, Embedding, LayerNorm, LayerNormNd, Linear, RmsNorm};
use crate::optim::{AdamW, clip_global_norm, cosine_decay, global_norm};
use crate::quantized::QMatrix as QMatrixByModulePath;
use crate::sampling::{
    GpuSampledToken, Mirostat2Sampler, StandardSamplerParams, sample, top_k_pairs,
};
use crate::session::{Backend, wrong_member_count};
use crate::tensor::{
    Dyn, Extent, FromArray, IndexOp, RoundMode, Scalar, TensorIndex, TensorSlice, arange,
    arange_step,
};

/// The three claims that are about *identity*, not existence, and that a
/// `use` line cannot make. Each is a type coercion: it compiles only while the
/// left name and the right name are the same item.
///
/// Never called. The value is irrelevant; the signature is the assertion.
const ROOT_TENSOR_IS_THE_CONST_RANK_ONE: fn(
    crate::tensor::typed::Tensor<2, f32>,
) -> Tensor<2, f32> = |t| t;
const ROOT_DEVICE_IS_THE_SESSIONS_OWNER: fn(crate::device::Device) -> Device = |d| d;
const DYN_IS_THE_RUNTIME_RANK_TENSOR: fn(crate::tensor::Dyn) -> Dyn = |t| t;

/// Signatures as coercions that compile only when types match the expected shapes,
/// so a changed rank parameter, a changed element parameter or a `Result`
/// creeping back into a forward is a compile error here.
///
/// Never called.
const OPS_ARE_METHODS_ON_THE_CONST_RANK_TENSOR: fn(&Tensor<4>, &Tensor<4>, &Tensor<2>) =
    |q, v, t| {
        let _: Tensor<4> = q.attention_causal(q, v, None);
        let _: Tensor<4> = q.attention_masked::<2>(q, v, MaskKind::None, Some(t), None);
        let _: Tensor<4> = q.rope(t, t, 0);
        let _: (Tensor<4>, Tensor<4>) = q.rope_pair(v, t, t, 0);
        let _: Tensor<4> = q.upsample_nearest2d(2, 2);
        let _: Tensor<4> = q.softmax(Minus1);
        let _: Tensor<4> = q.rms_norm(t, 1e-5);
        let _: Tensor<4> = q.layer_norm(t, None, 1e-5, true);
        let _: Tensor<4> = q.pool_max([2usize]);
        let _: Tensor<2> = t.pad_with_zeros(0usize, 1, 1);
        let _: Tensor<2> = t.repeat([1, 1]);
        let _: Tensor<1> = t.flatten_last_n(1);
        let _: [Dim; 2] = t.extents();
        let _: Option<u64> = t.elem_count();
        let _: Vec<f32> = t.to_vec_f32();
    };

/// Layer and cache parameters, with rank-generic `forward` methods.
const LAYERS_AND_CACHES_ARE_GENERIC: fn() = || {
    fn linear<T: Element, const R: usize>(l: &Linear<T>, x: &Tensor<R, T>) -> Tensor<R, T> {
        l.forward(x)
    }
    fn rms<const N: usize, T: Element, const R: usize>(
        l: &RmsNorm<N, T>,
        x: &Tensor<R, T>,
    ) -> Tensor<R, T> {
        l.forward(x)
    }
    fn norm<const N: usize, T: Element, const R: usize>(
        l: &LayerNorm<N, T>,
        nd: &LayerNormNd<N, T>,
        x: &Tensor<R, T>,
    ) -> (Tensor<R, T>, Tensor<R, T>) {
        (l.forward(x), nd.forward(x))
    }
    fn embed<T: Element>(l: &Embedding<T>, ids: &Tensor<2, u32>) -> Tensor<3, T> {
        l.forward(ids)
    }
    fn conv<const W: usize, T: Element, const R: usize>(
        l: &ConvNd<W, T>,
        x: &Tensor<R, T>,
    ) -> Tensor<R, T> {
        l.forward(x)
    }
    fn kv<const R: usize, T: Element>(
        c: &mut KvCache<R, T>,
        k: &Tensor<R, T>,
    ) -> (Tensor<R, T>, Tensor<R, T>) {
        c.commit();
        c.append(k, k)
    }
    fn cache<const R: usize, T: Element>(c: &mut TensorCache<R, T>) -> Option<Tensor<R, T>> {
        c.keep_last(1)
    }
    fn mask<T: Element, const R: usize>(m: &AttentionMask<T>, s: &Tensor<R, T>) -> Tensor<R, T> {
        m.apply(s)
    }
    fn rope<T: Element>(c: &RopeCache<T>) -> (Tensor<2, T>, Tensor<2, T>) {
        c.slice(0, 1)
    }
    // The bare names still mean the common case: no turbofish, no defaults
    // spelled out.
    let _: fn(&Linear, &Tensor<3>) -> Tensor<3> = linear;
    let _: fn(&RmsNorm, &Tensor<3>) -> Tensor<3> = rms;
    let _: fn(&LayerNorm, &LayerNormNd, &Tensor<3>) -> (Tensor<3>, Tensor<3>) = norm;
    let _: fn(&Embedding, &Tensor<2, u32>) -> Tensor<3> = embed;
    let _: fn(&ConvNd, &Tensor<4>) -> Tensor<4> = conv;
    let _: fn(&mut KvCache, &Tensor<4>) -> (Tensor<4>, Tensor<4>) = kv;
    let _: fn(&mut TensorCache) -> Option<Tensor<4>> = cache;
    let _: fn(&AttentionMask, &Tensor<4>) -> Tensor<4> = mask;
    let _: fn(&RopeCache) -> (Tensor<2>, Tensor<2>) = rope;
};
