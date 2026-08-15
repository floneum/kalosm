//! Attention masks. A causal mask is a `MaskKind::Causal` **attribute**, not a
//! tensor: the compiler skips upper-triangle Q.K work without loading
//! anything. `MaskCache` exists only for the genuinely data-dependent kinds.
//!
//! The reference builds a `[n, n]` `-inf` triangle for every distinct sequence
//! length, uploads it, memoizes it, and then *left-pads it with zeros* when
//! decoding at an offset — at which point it stops being causal at all and the
//! `is_strict_causal` fast path turns itself off. Here the same three
//! situations are three answers from [`MaskCache::get`], and two of them
//! upload nothing:
//!
//! * a square block is [`MaskKind::Causal`];
//! * `q_len == 1` against a cache of `k_len` keys sees every key, so it is
//!   [`MaskKind::None`];
//! * anything else — a chunk of `q_len > 1` queries at an offset into a longer
//!   key axis — is genuinely rectangular and needs a tensor, which is what
//!   [`MaskCache::materialized`] builds and what `entries` memoizes.
//!
//! Owned by W13.

use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::level1::MaskKind;
use fusor2_ir::shape::Dim;
use rustc_hash::FxHashMap;

use crate::device::ok;
use crate::graph::Graph;
use crate::tensor::typed::Element;
use crate::{Error, Result, Tensor};

/// A mask as attention consumes it.
///
/// `T` is the element type of the materialized case, defaulting to `f32`, as
/// the reference's `MaskCache<D: SimdElement>` is generic. The rank is fixed
/// at 2: a materialized mask is `[Lq, Lk]` and broadcasts over the batch and
/// head axes, so there is nothing for a rank parameter to vary.
#[derive(Clone)]
pub enum AttentionMask<T: Element = f32> {
    /// Structural; no tensor is materialized.
    Structural(MaskKind),
    /// A real `[Lq, Lk]` additive mask.
    Tensor(Tensor<2, T>),
}

impl<T: Element> AttentionMask<T> {
    /// The tensor this mask carries, if it carries one. A structural mask
    /// deliberately has none.
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
    pub fn new() -> Self {
        Self::default()
    }

    /// The mask a `[q_len, k_len]` score block needs.
    ///
    /// Structural whenever the shape alone decides it, which is both cases a
    /// decode loop ever hits. The rectangular case is reported as an error
    /// rather than guessed at, because materializing it needs a graph:
    /// [`MaskCache::materialized`] is that entry point.
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

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Backend, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().unwrap()).unwrap())
    }

    #[test]
    fn a_square_block_is_causal_and_loads_nothing() {
        let mut c: MaskCache = MaskCache::new();
        let m = c.get(Dim::Const(8), Dim::Const(8)).unwrap();
        assert!(matches!(m, AttentionMask::Structural(MaskKind::Causal)));
        assert!(m.tensor().is_none());
        assert!(c.is_empty(), "a structural mask must not be memoized");

        // The symbolic square case is the whole point: no length buckets.
        let g = graph();
        let s = g.sym("len");
        assert!(matches!(
            c.get(s, s).unwrap(),
            AttentionMask::Structural(MaskKind::Causal)
        ));
    }

    #[test]
    fn one_query_against_a_warm_cache_needs_no_mask() {
        let mut c: MaskCache = MaskCache::new();
        let m = c.get(Dim::Const(1), Dim::Const(37)).unwrap();
        assert!(matches!(m, AttentionMask::Structural(MaskKind::None)));
        assert!(c.is_empty());
    }

    #[test]
    fn a_chunked_prefill_names_the_entry_point_it_needs() {
        let mut c: MaskCache = MaskCache::new();
        assert!(c.get(Dim::Const(2), Dim::Const(5)).is_err());
    }

    #[test]
    fn the_materialized_mask_is_the_offset_triangle() {
        let g = graph();
        let mut c: MaskCache = MaskCache::new();
        // Two queries at positions 3 and 4 of a five-key axis.
        let m = c
            .materialized(&g, Dim::Const(2), Dim::Const(5), None)
            .unwrap();
        let v = m.tensor().unwrap().to_vec_f32();
        let inf = f32::NEG_INFINITY;
        assert_eq!(v, vec![0.0, 0.0, 0.0, 0.0, inf, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(c.len(), 1);

        // Same shape, same tensor: uploaded once.
        let again = c
            .materialized(&g, Dim::Const(2), Dim::Const(5), None)
            .unwrap();
        assert_eq!(again.tensor().unwrap().id(), m.tensor().unwrap().id());
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn a_sliding_window_also_masks_the_past() {
        let g = graph();
        let mut c: MaskCache = MaskCache::new();
        // Square block, window 2: query i sees keys i-1 and i.
        let m = c
            .materialized(&g, Dim::Const(3), Dim::Const(3), Some(2))
            .unwrap();
        let v = m.tensor().unwrap().to_vec_f32();
        let n = f32::NEG_INFINITY;
        assert_eq!(
            v,
            vec![
                0.0, n, n, //
                0.0, 0.0, n, //
                n, 0.0, 0.0,
            ]
        );
        // A window is a different key from the unwindowed mask of the same shape.
        c.materialized(&g, Dim::Const(3), Dim::Const(3), None)
            .unwrap();
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn a_symbolic_length_cannot_be_materialized() {
        let g = graph();
        let mut c: MaskCache = MaskCache::new();
        let s = g.sym("len");
        assert!(c.materialized(&g, Dim::Const(2), s, None).is_err());
    }
}
