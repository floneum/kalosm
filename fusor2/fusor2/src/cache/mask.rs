//! Attention masks. A causal mask is a `MaskKind::Causal` **attribute**, not a
//! tensor: the compiler skips upper-triangle Q.K work without loading
//! anything. `MaskCache` exists only for the genuinely data-dependent kinds.
//!
//! [`MaskCache::get`] returns three kinds of answers depending on the situation:
//!
//! * a square block is [`MaskKind::Causal`];
//! * `q_len == 1` against a cache of `k_len` keys sees every key, so it is
//!   [`MaskKind::None`];
//! * anything else — a chunk of `q_len > 1` queries at an offset into a longer
//!   key axis — is genuinely rectangular and needs a tensor, which is what
//!   [`MaskCache::materialized`] builds and what `entries` memoizes.

use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::launch::MaskKind;
use fusor2_ir::shape::Dim;
use rustc_hash::FxHashMap;

use crate::device::ok;
use crate::graph::Graph;
use crate::tensor::typed::Element;
use crate::{Error, Result, Tensor};

/// A mask as attention consumes it.
///
/// `T` is the element type of the materialized case, defaulting to `f32`.
/// The rank is fixed at 2: a materialized mask is `[Lq, Lk]` and broadcasts
/// over the batch and head axes.
#[derive(Clone)]
pub enum AttentionMask<T: Element = f32> {
    /// Structural; no tensor is materialized.
    Structural(MaskKind),
    /// A real `[Lq, Lk]` additive mask.
    Tensor(Tensor<2, T>),
}

impl<T: Element> AttentionMask<T> {
    /// The tensor this mask carries, if it carries one. A structural mask
    /// has none.
    pub fn tensor(&self) -> Option<&Tensor<2, T>> {
        match self {
            Self::Structural(_) => None,
            Self::Tensor(t) => Some(t),
        }
    }

    /// Which kind this mask is, structural or not — the argument
    /// `Tensor::attention_masked` takes.
    pub fn kind(&self) -> MaskKind {
        match self {
            Self::Structural(k) => *k,
            Self::Tensor(_) => MaskKind::QkMask,
        }
    }

    /// Add the mask to a score block. A structural mask is applied by the
    /// attention kernel from its `MaskKind`, so there is nothing to add.
    #[track_caller]
    pub fn apply<const R: usize>(&self, scores: &Tensor<R, T>) -> Tensor<R, T> {
        match self {
            Self::Structural(_) => scores.clone(),
            Self::Tensor(m) => Tensor::from_dyn(ok(
                "AttentionMask::apply",
                scores.as_dyn().add_(m.as_dyn()),
            )),
        }
    }
}

/// Memoizes the materialized masks only.
pub struct MaskCache<T: Element = f32> {
    /// Keyed by `(q_len, k_len, window)`, with `0` for an unbounded window.
    entries: FxHashMap<(u64, u64, u64), Tensor<2, T>>,
}

impl<T: Element> Default for MaskCache<T> {
    fn default() -> Self {
        Self {
            entries: FxHashMap::default(),
        }
    }
}

impl<T: Element> MaskCache<T> {
    /// Create an empty materialized-mask cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// The mask a `[q_len, k_len]` score block needs.
    ///
    /// Structural whenever the shape alone decides it, which is both cases a
    /// decode loop ever hits. The rectangular case is an error: materializing
    /// it needs a graph, which is [`MaskCache::materialized`]'s entry point.
    pub fn get(&mut self, q_len: Dim, k_len: Dim) -> Result<AttentionMask<T>> {
        Ok(AttentionMask::Structural(structural_kind(q_len, k_len).ok_or_else(
            || {
                Error::Shape(format!(
                    "a [{q_len}, {k_len}] score block is a chunked prefill at an offset; \
                     its mask is a real tensor, build it with MaskCache::materialized"
                ))
            },
        )?))
    }

    /// The additive `[q_len, k_len]` mask, uploaded once per shape.
    ///
    /// `mask[i, j] = 0` when key `j` is visible to query `i` and `-inf`
    /// otherwise. Queries are the **last** `q_len` positions of the key axis,
    /// which is what a chunked prefill against a warm cache means: query `i`
    /// sits at absolute position `i + (k_len - q_len)`.
    ///
    /// `window` bounds how far back a query may look, mirroring the
    /// reference's sliding-window mask; `None` is unbounded history.
    pub fn materialized(
        &mut self,
        graph: &Graph,
        q_len: Dim,
        k_len: Dim,
        window: Option<u64>,
    ) -> Result<AttentionMask<T>> {
        let (Some(q), Some(k)) = (q_len.as_const(), k_len.as_const()) else {
            return Err(Error::Shape(
                "a materialized mask needs concrete extents; a symbolic length is what \
                 MaskKind::Causal exists for"
                    .into(),
            ));
        };
        if q > k {
            return Err(Error::Shape(format!(
                "a [{q}, {k}] score block has more queries than keys"
            )));
        }
        let key = (q, k, window.unwrap_or(0));
        if let Some(t) = self.entries.get(&key) {
            return Ok(AttentionMask::Tensor(t.clone()));
        }
        let offset = k - q;
        let mut data = vec![0.0f32; (q * k) as usize];
        for i in 0..q {
            let pos = i + offset;
            for j in 0..k {
                let too_new = j > pos;
                let too_old = window.is_some_and(|w| j + w <= pos);
                if too_new || too_old {
                    data[(i * k + j) as usize] = f32::NEG_INFINITY;
                }
            }
        }
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let dense = graph.tensor(Dtype::F32, &[Dim::Const(q), Dim::Const(k)], &bytes)?;
        // The triangle is built in f32 on the host — `-inf` is exact in every
        // float width — and cast once per shape, not per step.
        let dense = if T::DTYPE == Dtype::F32 {
            dense
        } else {
            dense.cast(T::DTYPE)?
        };
        let tensor = Tensor::<2, T>::try_from_dyn(dense)?;
        self.entries.insert(key, tensor.clone());
        Ok(AttentionMask::Tensor(tensor))
    }

    /// How many masks have been uploaded.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no masks have been uploaded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Discard every uploaded mask.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

/// The `MaskKind` a `[q_len, k_len]` block needs without loading anything, or
/// `None` when the block is genuinely rectangular.
fn structural_kind(q_len: Dim, k_len: Dim) -> Option<MaskKind> {
    if q_len.known_eq(k_len) {
        return Some(MaskKind::Causal);
    }
    // One query against a warm cache: every cached key precedes it.
    if q_len.as_const() == Some(1) {
        return Some(MaskKind::None);
    }
    None
}
