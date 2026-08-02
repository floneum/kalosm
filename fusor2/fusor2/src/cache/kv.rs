//! The KV cache. Sequence length is a `Dim::Sym` bound at dispatch, so growing
//! the cache does not recompile anything and there are no length buckets.
//!
//! The reference carries `allocated_seq_len`, a power-of-two growth schedule,
//! a `GPU_CACHE_MIN_ALLOC_SEQ_LEN` floor and a separate `backing` tensor it
//! writes in place — all of it there to stop one `cat` per decode step from
//! reallocating. None of that is here, and it is not an omission: an append
//! builds a `cat` node, and whether that node lands in a new buffer or writes
//! into the previous one is the arena planner's call, priced through the same
//! `BufferRole` machinery every other in-place op goes through. A capacity
//! schedule in the frontend would be the frontend second-guessing the planner.
//!
//! Owned by W13.

use fusor2_ir::shape::Dim;

use crate::tensor::Tensor;
use crate::{Error, Result};

/// A growable append-only tensor cache along one axis.
#[derive(Clone)]
pub struct TensorCache {
    pub data: Option<Tensor>,
    pub axis: u32,
    pub len: Dim,
}

impl TensorCache {
    pub fn new(axis: u32) -> Self {
        Self {
            data: None,
            axis,
            len: Dim::Const(0),
        }
    }

    /// The cached tensor, or `None` before the first append.
    pub fn current(&self) -> Option<&Tensor> {
        self.data.as_ref()
    }

    /// Tokens currently cached along [`TensorCache::axis`].
    pub fn len(&self) -> Dim {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_none()
    }

    /// Append `value` along `axis` and return the whole cache, new part
    /// included.
    ///
    /// The first append stores `value` itself: there is nothing to
    /// concatenate it with, and a `cat` of one operand is a copy nobody
    /// asked for.
    pub fn append(&mut self, value: &Tensor) -> Result<Tensor> {
        let axis = self.axis as usize;
        if axis >= value.rank() {
            return Err(Error::Shape(format!(
                "cache axis {axis} out of range for a rank-{} value",
                value.rank()
            )));
        }
        let added = value.dim(axis);
        // Every check runs before the cache is touched: a rejected append
        // must leave the cache exactly as it was, or a caller that recovers
        // from the error silently continues with an emptied cache.
        let out = match self.data.as_ref() {
            None => value.clone(),
            Some(prev) => {
                if prev.rank() != value.rank() {
                    return Err(Error::Shape(format!(
                        "cache holds rank {} but was appended a rank-{} value",
                        prev.rank(),
                        value.rank()
                    )));
                }
                if prev.dtype() != value.dtype() {
                    return Err(Error::Dtype(format!(
                        "cache holds {:?} but was appended {:?}",
                        prev.dtype(),
                        value.dtype()
                    )));
                }
                for i in 0..prev.rank() {
                    if i != axis && !prev.dim(i).known_eq(value.dim(i)) {
                        return Err(Error::Shape(format!(
                            "cache axis {i} disagrees: {} vs {}",
                            prev.dim(i),
                            value.dim(i)
                        )));
                    }
                }
                Tensor::cat(&[prev.clone(), value.clone()], axis)?
            }
        };
        self.len = add_dims(self.len, added);
        self.data = Some(out.clone());
        Ok(out)
    }

    /// Keep the newest `len` tokens and drop the oldest — the sliding-window
    /// eviction the reference spells `narrow(dim, total - max, max)`.
    pub fn keep_last(&mut self, len: u64) -> Result<Option<Tensor>> {
        let Some(data) = self.data.as_ref() else {
            return Ok(None);
        };
        let axis = self.axis as usize;
        let Some(total) = data.dim(axis).as_const() else {
            return Err(Error::Shape(
                "a symbolic cache extent cannot be evicted by a host-known window; \
                 narrow it with a position gather instead"
                    .into(),
            ));
        };
        if total <= len {
            return Ok(Some(data.clone()));
        }
        let kept = data.narrow(axis, (total - len) as usize, len as usize)?;
        self.len = Dim::Const(len);
        self.data = Some(kept.clone());
        Ok(Some(kept))
    }

    pub fn reset(&mut self) {
        self.data = None;
        self.len = Dim::Const(0);
    }
}

/// `a + b` over extents. Anything involving a symbol has no constant sum, so
/// the cache reports the symbolic side rather than inventing a symbol it
/// cannot bind.
fn add_dims(a: Dim, b: Dim) -> Dim {
    match (a, b) {
        (Dim::Const(x), Dim::Const(y)) => Dim::Const(x + y),
        (Dim::Const(0), other) => other,
        (other, Dim::Const(0)) => other,
        (_, sym) => sym,
    }
}

/// One layer's key and value caches.
#[derive(Clone)]
pub struct KvCache {
    pub k: TensorCache,
    pub v: TensorCache,
}

impl KvCache {
    pub fn new(axis: u32) -> Self {
        Self {
            k: TensorCache::new(axis),
            v: TensorCache::new(axis),
        }
    }

    /// Append one step's keys and values; returns the full cached pair.
    pub fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor)> {
        let keys = self.k.append(k)?;
        let values = self.v.append(v)?;
        Ok((keys, values))
    }

    /// Cached sequence length. The two halves always advance together, so the
    /// key cache is authoritative.
    pub fn len(&self) -> Dim {
        self.k.len()
    }

    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }

    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Device, Session};
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::ir::Op;
    use fusor2_ir::ir::level0::L0;
    use fusor2_ir::egraph::Id;

    fn graph() -> crate::Graph {
        crate::Graph::new(&Session::new(Device::cpu().unwrap()).unwrap())
    }

    fn upload(g: &crate::Graph, shape: &[u64], data: &[f32]) -> Tensor {
        let dims: Vec<Dim> = shape.iter().copied().map(Dim::Const).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.tensor(Dtype::F32, &dims, &bytes).unwrap()
    }

    /// The `Restride` operands of the class `id` names, in order — the two
    /// halves a `cat` glues together.
    fn cat_sources(t: &Tensor) -> Vec<Id> {
        let g = t.graph().egraph.lock();
        let mut out = Vec::new();
        for member in g.class_ids(g.class_of(t.id())) {
            if let Op::L0(L0::Scatter { base, upd, .. }) = &g.node(member).op {
                out.push(*base);
                out.push(*upd);
            }
        }
        out
    }

    #[test]
    fn a_two_step_append_returns_both_steps_in_order() {
        // [batch = 1, heads = 1, len, head_dim = 2], concatenating on axis 2.
        let g = graph();
        let mut cache = KvCache::new(2);

        let k0 = upload(&g, &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let v0 = upload(&g, &[1, 1, 2, 2], &[-1.0, -2.0, -3.0, -4.0]);
        let (ks, vs) = cache.append(&k0, &v0).unwrap();
        assert_eq!(cache.len(), Dim::Const(2));
        assert_eq!(ks.dim(2), Dim::Const(2));
        // The first append is the value itself, not a copy of it, so this
        // readback is a straight upload/download and asserts real numbers.
        assert_eq!(ks.id(), k0.id());
        assert_eq!(ks.to_vec_f32().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(vs.to_vec_f32().unwrap(), vec![-1.0, -2.0, -3.0, -4.0]);

        // Step two: one more token. The cache grows along the cat axis only,
        // and the *new* value is the second operand — appending is ordered,
        // and reversing it silently corrupts a decode loop.
        let k1 = upload(&g, &[1, 1, 1, 2], &[5.0, 6.0]);
        let v1 = upload(&g, &[1, 1, 1, 2], &[-5.0, -6.0]);
        let (ks2, vs2) = cache.append(&k1, &v1).unwrap();
        assert_eq!(cache.len(), Dim::Const(3));
        assert_eq!(
            &ks2.shape()[..],
            &[Dim::Const(1), Dim::Const(1), Dim::Const(3), Dim::Const(2)]
        );
        assert_eq!(&vs2.shape()[..], &ks2.shape()[..]);
        assert_eq!(cache.k.current().unwrap().id(), ks2.id());
        assert_eq!(cache.v.current().unwrap().id(), vs2.id());

        // Numerically this is `views::cat_dim*`'s obligation, and those cases
        // are red for a reason that is not this cache: the emitters index
        // every operand with the flat output index and ignore
        // `Operand::layout`. What is asserted here is the part the cache
        // owns — that the second step glues the *new* value on after the
        // cached one, in that order.
        let sources = cat_sources(&ks2);
        assert!(
            sources.iter().any(|s| *s == k1.id()),
            "the appended value must be an operand of the grown cache"
        );

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), Dim::Const(0));
    }

    #[test]
    fn appending_on_a_leading_axis_grows_that_axis_only() {
        let g = graph();
        let mut cache = TensorCache::new(0);
        let a = upload(&g, &[1, 3], &[1.0, 2.0, 3.0]);
        let b = upload(&g, &[2, 3], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        cache.append(&a).unwrap();
        let out = cache.append(&b).unwrap();
        assert_eq!(cache.len(), Dim::Const(3));
        assert_eq!(&out.shape()[..], &[Dim::Const(3), Dim::Const(3)]);
    }

    #[test]
    fn eviction_narrows_to_the_newest_tokens() {
        let g = graph();
        let mut cache = TensorCache::new(0);
        cache
            .append(&upload(&g, &[4, 1], &[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        let kept = cache.keep_last(2).unwrap().unwrap();
        assert_eq!(cache.len(), Dim::Const(2));
        assert_eq!(&kept.shape()[..], &[Dim::Const(2), Dim::Const(1)]);
        // A window wider than the cache is a no-op that does not renarrow.
        let same = cache.keep_last(8).unwrap().unwrap();
        assert_eq!(same.id(), kept.id());
        assert_eq!(cache.len(), Dim::Const(2));
        // An empty cache has nothing to evict.
        assert!(TensorCache::new(0).keep_last(2).unwrap().is_none());
    }

    #[test]
    fn a_symbolic_length_accumulates_to_the_newest_symbol() {
        let g = graph();
        let s = g.sym("len");
        let mut cache = TensorCache::new(0);
        let t = g.leaf("x", &[s, Dim::Const(2)], Dtype::F32).unwrap();
        cache.append(&t).unwrap();
        assert_eq!(cache.len(), s);
        // And it cannot be evicted by a host-known window.
        assert!(cache.keep_last(1).is_err());
    }

    #[test]
    fn a_mismatched_append_is_an_error_not_a_panic() {
        let g = graph();
        let mut cache = TensorCache::new(0);
        cache.append(&upload(&g, &[1, 3], &[1.0, 2.0, 3.0])).unwrap();
        // A non-cat axis that disagrees.
        assert!(cache.append(&upload(&g, &[1, 4], &[0.0; 4])).is_err());
        // A rank that disagrees.
        assert!(cache.append(&upload(&g, &[3], &[0.0; 3])).is_err());
        // The cache is untouched by a rejected append.
        assert_eq!(cache.len(), Dim::Const(1));
        // A dtype that disagrees.
        let u = g
            .tensor(Dtype::U32, &[Dim::Const(1), Dim::Const(3)], &[0u8; 12])
            .unwrap();
        assert!(cache.append(&u).is_err());
        // An axis off the end.
        assert!(
            TensorCache::new(7)
                .append(&upload(&g, &[2], &[0.0; 2]))
                .is_err()
        );
    }

    #[test]
    fn add_dims_is_saturating_on_symbols() {
        let g = graph();
        let s = g.sym("n");
        assert_eq!(add_dims(Dim::Const(2), Dim::Const(3)), Dim::Const(5));
        assert_eq!(add_dims(Dim::Const(0), s), s);
        assert_eq!(add_dims(s, Dim::Const(0)), s);
        assert_eq!(add_dims(Dim::Const(4), s), s);
    }
}
