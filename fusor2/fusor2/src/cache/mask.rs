//! Attention masks. A causal mask is a `MaskKind::Causal` attribute, not a
//! tensor: the compiler skips upper-triangle Q.K work without loading anything.
//!
//! [`MaskCache::get`] answers structurally in two cases: a square block is
//! [`MaskKind::Causal`], and `q_len == 1` sees every key so it is
//! [`MaskKind::None`]. A chunk of `q_len > 1` queries at an offset into a
//! longer key axis needs a real tensor from [`MaskCache::materialized`], which
//! `entries` memoizes.

use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::level1::MaskKind;
use fusor2_ir::shape::Dim;
use rustc_hash::FxHashMap;

use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::{Error, Result};

/// A mask as attention consumes it.
#[derive(Clone)]
pub enum AttentionMask {
    /// Structural; no tensor is materialized.
    Structural(MaskKind),
    /// A real `[.., Lq, Lk]` additive mask.
    Tensor(Tensor),
}

impl AttentionMask {
    /// The tensor this mask carries; a structural mask has none.
    pub fn tensor(&self) -> Option<&Tensor> {
        match self {
            Self::Structural(_) => None,
            Self::Tensor(t) => Some(t),
        }
    }

    /// `true` when nothing is loaded to apply this mask.
    pub fn is_structural(&self) -> bool {
        matches!(self, Self::Structural(_))
    }

    /// Add the mask to a score block. A structural mask is applied by the
    /// attention kernel from its `MaskKind`, so there is nothing to add.
    pub fn apply(&self, scores: &Tensor) -> Result<Tensor> {
        match self {
            Self::Structural(_) => Ok(scores.clone()),
            Self::Tensor(m) => scores.add_(m),
        }
    }
}

/// Memoizes the materialized masks only.
#[derive(Default)]
pub struct MaskCache {
    /// Keyed by `(q_len, k_len, window)`, with `0` for an unbounded window.
    entries: FxHashMap<(u64, u64, u64), Tensor>,
}

impl MaskCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// The mask a `[q_len, k_len]` score block needs.
    ///
    /// Structural whenever the shape alone decides it. A rectangular block is
    /// an error here; it needs a graph, so use [`MaskCache::materialized`].
    pub fn get(&mut self, q_len: Dim, k_len: Dim) -> Result<AttentionMask> {
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
    /// otherwise. Queries occupy the last `q_len` positions of the key axis:
    /// query `i` sits at absolute position `i + (k_len - q_len)`.
    ///
    /// `window` bounds how far back a query may look; `None` is unbounded
    /// history.
    pub fn materialized(
        &mut self,
        graph: &Graph,
        q_len: Dim,
        k_len: Dim,
        window: Option<u64>,
    ) -> Result<AttentionMask> {
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
        let tensor = graph.tensor(Dtype::F32, &[Dim::Const(q), Dim::Const(k)], &bytes)?;
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
    use crate::session::{Device, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Device::cpu().unwrap()).unwrap())
    }

    #[test]
    fn a_square_block_is_causal_and_loads_nothing() {
        let mut c = MaskCache::new();
        let m = c.get(Dim::Const(8), Dim::Const(8)).unwrap();
        assert!(matches!(m, AttentionMask::Structural(MaskKind::Causal)));
        assert!(m.tensor().is_none());
        assert!(c.is_empty(), "a structural mask must not be memoized");

        let g = graph();
        let s = g.sym("len");
        assert!(matches!(
            c.get(s, s).unwrap(),
            AttentionMask::Structural(MaskKind::Causal)
        ));
    }

    #[test]
    fn one_query_against_a_warm_cache_needs_no_mask() {
        let mut c = MaskCache::new();
        let m = c.get(Dim::Const(1), Dim::Const(37)).unwrap();
        assert!(matches!(m, AttentionMask::Structural(MaskKind::None)));
        assert!(c.is_empty());
    }

    #[test]
    fn a_chunked_prefill_names_the_entry_point_it_needs() {
        let mut c = MaskCache::new();
        assert!(c.get(Dim::Const(2), Dim::Const(5)).is_err());
    }

    #[test]
    fn the_materialized_mask_is_the_offset_triangle() {
        let g = graph();
        let mut c = MaskCache::new();
        // Two queries at positions 3 and 4 of a five-key axis.
        let m = c
            .materialized(&g, Dim::Const(2), Dim::Const(5), None)
            .unwrap();
        let v = m.tensor().unwrap().to_vec_f32().unwrap();
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
        let mut c = MaskCache::new();
        // Square block, window 2: query i sees keys i-1 and i.
        let m = c
            .materialized(&g, Dim::Const(3), Dim::Const(3), Some(2))
            .unwrap();
        let v = m.tensor().unwrap().to_vec_f32().unwrap();
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
        let mut c = MaskCache::new();
        let s = g.sym("len");
        assert!(c.materialized(&g, Dim::Const(2), s, None).is_err());
    }
}
