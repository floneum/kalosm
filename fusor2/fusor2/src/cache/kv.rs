//! The KV cache. Sequence length is a `Dim::Sym` bound at dispatch, so growing
//! the cache does not recompile anything and there are no length buckets.
//!
//! Two modes:
//!
//! * **cat** (the default): an append is a `cat` node and the whole cache is a
//!   fresh value each step. Correct anywhere, but every step's graph differs
//!   from the last, so a decode loop replans per token.
//! * **fixed** ([`TensorCache::fixed`]): a fixed-capacity external leaf, an
//!   append is one `Scatter{Set}` at a device-side write index, and the
//!   readable cache is a `Dim::Sym`-length narrow of the scatter's output.
//!   Every step reuses the *same* nodes — only leaf bytes and the symbol's
//!   binding change — which is what lets a decode loop replay one plan per
//!   token. After the step's resolve, [`TensorCache::commit`] re-points the
//!   leaf at the buffer the scatter produced (no host round trip).

use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::level0::{L0, LeafKind};
use fusor2_ir::shape::{Dim, StrideSpec};
use rustc_hash::FxHashMap;

use crate::device::ok;
use crate::graph::GraphRef;
use crate::tensor::Dyn;
use crate::tensor::typed::Element;
use crate::{Error, Result, Tensor};

/// The fixed-capacity half of a [`TensorCache`].
#[derive(Clone)]
struct FixedState {
    /// Slots along the cache axis the store leaf holds.
    capacity: u64,
    /// Ring window: at most this many newest tokens are kept, written at
    /// `position % window`. Keys carry their rotary phase already and decode
    /// attention is permutation-invariant over keys, so ring order is sound.
    window: Option<u64>,
    /// Tokens appended so far (absolute count, not clamped to the window).
    len: u64,
    /// The capacity leaf. Recreated only on growth.
    store: Option<Dyn>,
    /// This step's scatter output, until [`TensorCache::commit`] adopts it.
    out: Option<Dyn>,
    /// The last append's `(chunk width, scatter output)`, retained past the
    /// commit that cleared `out`. An append of the same width against the same
    /// store rebuilds exactly these nodes, so [`TensorCache::replay_append`]
    /// re-arms them instead — which is what lets a decode loop skip rebuilding
    /// the model graph entirely. Dropped on growth, which is a new store leaf
    /// and therefore new nodes.
    arm: Option<(u64, Dyn)>,
    /// `u32` write-index leaves, one per appended-chunk width, reused across
    /// steps with fresh bytes.
    idx: FxHashMap<u64, Dyn>,
    /// The symbol the readable length is bound to, named once per store.
    sym: Option<fusor2_ir::shape::SymId>,
    /// The symbol's name. A [`KvCache`] hands both halves one name so the
    /// K and V views carry the *same* symbol — attention contracts their
    /// length axes against each other, and two symbols bound to one value
    /// are still two extents to the shape checker.
    sym_name: String,
}

/// A growable append-only tensor cache along one axis.
///
/// `R` is the rank of the values it holds and `T` their element type.
/// Both default to the decode shape — a rank-4 `[batch, heads, seq, dim]` f32
/// cache — so `TensorCache` alone still names it. The axis is a runtime
/// argument, not a const parameter: a cross-attention cache and a self-
/// attention one differ in axis and not in type.
#[derive(Clone)]
pub struct TensorCache<const R: usize = 4, T: Element = f32> {
    pub data: Option<Tensor<R, T>>,
    pub axis: u32,
    pub len: Dim,
    fixed: Option<FixedState>,
}

impl<const R: usize, T: Element> TensorCache<R, T> {
    pub fn new(axis: u32) -> Self {
        Self {
            data: None,
            axis,
            len: Dim::Const(0),
            fixed: None,
        }
    }

    /// A fixed-capacity cache: appends scatter into a persistent device
    /// buffer and the current value is a symbolic-length narrow. `capacity`
    /// is the initial slot count; it grows by doubling (a new shape family)
    /// when exceeded.
    pub fn fixed(axis: u32, capacity: u64) -> Self {
        Self::fixed_named(axis, capacity, fresh_sym_name())
    }

    /// [`TensorCache::fixed`] with a caller-supplied length-symbol name;
    /// two caches sharing one name share one symbol.
    pub fn fixed_named(axis: u32, capacity: u64, sym_name: String) -> Self {
        Self {
            data: None,
            axis,
            len: Dim::Const(0),
            fixed: Some(FixedState {
                capacity: capacity.max(1),
                window: None,
                len: 0,
                store: None,
                out: None,
                arm: None,
                idx: FxHashMap::default(),
                sym: None,
                sym_name,
            }),
        }
    }

    /// A fixed ring of the newest `window` tokens. Decode-shaped use only:
    /// the caller's mask must already treat every present key as visible.
    pub fn fixed_windowed(axis: u32, window: u64) -> Self {
        let mut cache = Self::fixed(axis, window.max(1));
        if let Some(f) = cache.fixed.as_mut() {
            f.window = Some(window.max(1));
        }
        cache
    }

    /// Whether this cache is in fixed (scatter/ring) mode.
    pub fn is_fixed(&self) -> bool {
        self.fixed.is_some()
    }

    /// Whether eviction is already handled by the ring write.
    pub fn is_ring(&self) -> bool {
        self.fixed.as_ref().is_some_and(|f| f.window.is_some())
    }

    /// The scatter output the current step produced, if any — it must be a
    /// root of the step's resolve so [`TensorCache::commit`] can adopt its
    /// buffer.
    ///
    /// Runtime-rank: a resolve batch collects roots of every rank a step
    /// produced, which is the one place this crate genuinely has a
    /// heterogeneous list.
    pub fn pending(&self) -> Option<Dyn> {
        self.fixed.as_ref().and_then(|f| f.out.clone())
    }

    /// Adopt the resolved scatter output into the store leaf and drop the
    /// output's binding so the next step re-dispatches. Call once per step,
    /// after the resolve that included [`TensorCache::pending`].
    #[track_caller]
    pub fn commit(&mut self) {
        let Some(f) = self.fixed.as_mut() else {
            return;
        };
        if let (Some(store), Some(out)) = (f.store.as_ref(), f.out.take()) {
            ok("TensorCache::commit", store.adopt_buffer(&out));
            out.clear_device_buf();
        }
    }

    /// The cached tensor, or `None` before the first append.
    pub fn current(&self) -> Option<&Tensor<R, T>> {
        self.data.as_ref()
    }

    /// The cached value at runtime rank, for the resolve-batch path.
    pub fn current_dyn(&self) -> Option<&Dyn> {
        self.data.as_ref().map(Tensor::as_dyn)
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
    #[track_caller]
    pub fn append(&mut self, value: &Tensor<R, T>) -> Tensor<R, T> {
        Tensor::from_dyn(ok("TensorCache::append", self.append_dyn(value.as_dyn())))
    }

    /// [`TensorCache::append`] at runtime rank. The whole body is here
    /// because a fixed-mode append walks strides and mints leaves, none of
    /// which the rank in the type would make any easier to write.
    pub(crate) fn append_dyn(&mut self, value: &Dyn) -> Result<Dyn> {
        let axis = self.axis as usize;
        if axis >= value.rank() {
            return Err(Error::Shape(format!(
                "cache axis {axis} out of range for a rank-{} value",
                value.rank()
            )));
        }
        if self.fixed.is_some() {
            return self.append_fixed(value);
        }
        let added = value.dim(axis);
        // Every check runs before the cache is touched: a rejected append
        // must leave the cache exactly as it was, or a caller that recovers
        // from the error silently continues with an emptied cache.
        let out = match self.current_dyn() {
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
                Dyn::cat(&[prev.clone(), value.clone()], axis)?
            }
        };
        self.len = add_dims(self.len, added);
        self.data = Some(Tensor::try_from_dyn(out.clone())?);
        Ok(out)
    }

    /// The fixed-mode append: one `Scatter{Set}` into the capacity leaf at a
    /// device-side write index, and the readable cache is a symbolic-length
    /// narrow of the scatter output. Node identity is step-invariant: the
    /// leaves are minted once, only their bytes and the symbol's binding
    /// move.
    fn append_fixed(&mut self, value: &Dyn) -> Result<Dyn> {
        let axis = self.axis as usize;
        let graph = value.graph().clone();
        let added = value
            .dim(axis)
            .as_const()
            .ok_or_else(|| Error::Shape("a fixed cache appends host-known chunks".into()))?;
        let f = self.fixed.as_mut().expect("checked by append");

        // Capacity: a ring never grows; a plain cache doubles (new shape
        // family) and migrates the committed tokens on device.
        if let Some(w) = f.window {
            if added > w {
                return Err(Error::Shape(format!(
                    "an append of {added} exceeds the {w}-token window"
                )));
            }
        } else if f.len + added > f.capacity {
            let needed = f.len + added;
            grow(f, &graph, value, axis, needed)?;
        }

        // The store leaf, minted on first use. No host bytes: wgpu (and the
        // CPU pool) zero-initialize, and nothing past the bound length is
        // ever read.
        if f.store.is_none() {
            let mut shape = value.shape().to_vec();
            shape[axis] = Dim::Const(f.capacity);
            let store = external_leaf(&graph, &shape, value.dtype())?;
            f.sym = Some(graph.named_sym(&f.sym_name));
            f.store = Some(store);
        }
        let store = f.store.clone().expect("minted above");
        let sym = f.sym.expect("minted with the store");

        // Write positions for this chunk.
        let positions: Vec<u32> = (0..added)
            .map(|i| {
                let abs = f.len + i;
                let slot = match f.window {
                    Some(w) => abs % w,
                    None => abs,
                };
                u32::try_from(slot).expect("capacity fits a u32")
            })
            .collect();
        let idx = match f.idx.get(&added) {
            Some(t) => t.clone(),
            None => {
                let t = external_leaf(&graph, &[Dim::Const(added)], Dtype::U32)?;
                f.idx.insert(added, t.clone());
                t
            }
        };
        idx.set_bytes(positions.iter().flat_map(|v| v.to_le_bytes()).collect())?;

        let out = store.scatter_set(axis, &idx, value, true)?;
        f.len += added;
        let total = match f.window {
            Some(w) => f.len.min(w),
            None => f.len,
        };
        graph.bind_dim(sym, total);

        // The readable cache: `[.., total, ..]` of the scatter output.
        let specs: Vec<StrideSpec> = out
            .shape()
            .iter()
            .copied()
            .enumerate()
            .map(|(i, d)| {
                if i == axis {
                    StrideSpec::dim(i as u32, Dim::Sym(sym))
                } else {
                    StrideSpec::dim(i as u32, d)
                }
            })
            .collect();
        let view = out.restride(&specs)?;

        f.arm = Some((added, out.clone()));
        f.out = Some(out);
        self.len = Dim::Sym(sym);
        self.data = Some(Tensor::try_from_dyn(view.clone())?);
        Ok(view)
    }

    /// Whether [`TensorCache::replay_append`] would rebuild the last append's
    /// nodes exactly: same store, same index leaf, same chunk width, and no
    /// growth in between.
    pub fn can_replay(&self, added: u64) -> bool {
        let Some(f) = self.fixed.as_ref() else {
            return false;
        };
        if self.data.is_none() || f.store.is_none() || f.sym.is_none() {
            return false;
        }
        if f.arm.as_ref().is_none_or(|(w, _)| *w != added) {
            return false;
        }
        if !f.idx.contains_key(&added) {
            return false;
        }
        match f.window {
            Some(w) => added <= w,
            None => f.len + added <= f.capacity,
        }
    }

    /// Advance one append *without* touching the graph: an append of the same
    /// width against the same store hash-conses onto the nodes the last one
    /// minted, so the only real work is the write index's bytes and the length
    /// binding. The caller must have checked [`TensorCache::can_replay`].
    ///
    /// This is what makes a decode loop's graph build disappear: the model's
    /// per-step rebuild exists only to re-derive nodes that already exist, and
    /// this re-arms the one piece of per-step state hiding inside them.
    pub fn replay_append(&mut self, added: u64) -> Result<()> {
        if !self.can_replay(added) {
            return Err(Error::Shape(
                "replay_append needs a fixed cache whose last append had the same width and \
                 whose store has not grown since; check can_replay first"
                    .into(),
            ));
        }
        let f = self.fixed.as_mut().expect("checked by can_replay");
        let (_, out) = f.arm.clone().expect("checked by can_replay");
        let sym = f.sym.expect("checked by can_replay");
        let idx = f.idx.get(&added).cloned().expect("checked by can_replay");

        let positions: Vec<u32> = (0..added)
            .map(|i| {
                let abs = f.len + i;
                let slot = match f.window {
                    Some(w) => abs % w,
                    None => abs,
                };
                u32::try_from(slot).expect("capacity fits a u32")
            })
            .collect();
        idx.set_bytes(positions.iter().flat_map(|v| v.to_le_bytes()).collect())?;

        f.len += added;
        let total = match f.window {
            Some(w) => f.len.min(w),
            None => f.len,
        };
        out.graph().bind_dim(sym, total);
        f.out = Some(out);
        self.len = Dim::Sym(sym);
        Ok(())
    }

    /// Keep the newest `len` tokens and drop the oldest — the sliding-window
    /// eviction the reference spells `narrow(dim, total - max, max)`.
    #[track_caller]
    pub fn keep_last(&mut self, len: u64) -> Option<Tensor<R, T>> {
        ok("TensorCache::keep_last", self.keep_last_inner(len))
    }

    fn keep_last_inner(&mut self, len: u64) -> Result<Option<Tensor<R, T>>> {
        let Some(data) = self.current_dyn() else {
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
            return Ok(self.data.clone());
        }
        let kept = Tensor::try_from_dyn(data.narrow(axis, (total - len) as usize, len as usize)?)?;
        self.len = Dim::Const(len);
        self.data = Some(kept.clone());
        Ok(Some(kept))
    }

    pub fn reset(&mut self) {
        self.data = None;
        self.len = Dim::Const(0);
        if let Some(f) = self.fixed.as_mut() {
            // Keep the leaves: stale slots past the bound length are never
            // read, so a cleared cache reuses the same nodes and buffers.
            f.len = 0;
            f.out = None;
        }
    }
}

/// A process-unique name for a cache's length symbol. Growth keeps the
/// name: the family changes through the store leaf's shape, and reusing the
/// symbol keeps every dependent binding coherent.
fn fresh_sym_name() -> String {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    format!(
        "kv_len#{}",
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

/// An external leaf minted directly on the graph handle (the `Graph` facade
/// is not reachable from a tensor).
fn external_leaf(graph: &GraphRef, shape: &[Dim], dtype: Dtype) -> Result<Dyn> {
    let id = graph.add_l0(L0::Leaf(LeafKind::Buffer {
        name: graph.fresh_buffer_id(),
        dtype,
        shape: shape.iter().copied().collect(),
    }))?;
    Ok(graph.tensor(id))
}

/// Double the store until `needed` fits and migrate the committed tokens on
/// device: one `slice_assign` resolve, then the new leaf adopts its buffer.
/// A new capacity is a new leaf shape — deliberately a new shape family.
fn grow(
    f: &mut FixedState,
    graph: &GraphRef,
    like: &Dyn,
    axis: usize,
    needed: u64,
) -> Result<()> {
    let mut capacity = f.capacity.max(1);
    while capacity < needed {
        capacity *= 2;
    }
    if capacity == f.capacity && f.store.is_some() {
        return Ok(());
    }
    let old = f.store.take();
    let mut shape = like.shape().to_vec();
    shape[axis] = Dim::Const(capacity);
    let store = external_leaf(graph, &shape, like.dtype())?;
    if let (Some(old_store), len @ 1..) = (old, f.len) {
        let mut ranges = Vec::with_capacity(shape.len());
        for (i, d) in shape.iter().enumerate() {
            let extent = if i == axis {
                len
            } else {
                d.as_const()
                    .ok_or_else(|| Error::Shape("a cache store has constant extents".into()))?
            };
            ranges.push(0..extent as usize);
        }
        let kept = old_store.narrow(axis, 0, len as usize)?;
        let moved = store.slice_assign(&ranges, &kept)?;
        graph.session().resolve(&[moved.clone()])?;
        store.adopt_buffer(&moved)?;
        moved.clear_device_buf();
    }
    f.sym = Some(graph.named_sym(&f.sym_name));
    f.capacity = capacity;
    f.store = Some(store);
    f.out = None;
    f.arm = None;
    f.idx.clear();
    Ok(())
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
///
/// `R` and `T` are the cached values', and both default to the decode shape
/// the reference pins outright — its `KvCache<D>` holds two
/// `TensorCache<4, D>` — so a bare `KvCache` is a rank-4 f32 pair.
#[derive(Clone)]
pub struct KvCache<const R: usize = 4, T: Element = f32> {
    pub k: TensorCache<R, T>,
    pub v: TensorCache<R, T>,
}

impl<const R: usize, T: Element> KvCache<R, T> {
    pub fn new(axis: u32) -> Self {
        Self {
            k: TensorCache::new(axis),
            v: TensorCache::new(axis),
        }
    }

    /// Fixed-capacity mode: one plan per decode step. See [`TensorCache::fixed`].
    /// Both halves share one length symbol: attention contracts K's and V's
    /// length axes against each other.
    pub fn with_capacity(axis: u32, capacity: u64) -> Self {
        let name = fresh_sym_name();
        Self {
            k: TensorCache::fixed_named(axis, capacity, name.clone()),
            v: TensorCache::fixed_named(axis, capacity, name),
        }
    }

    /// Ring of the newest `window` tokens. See [`TensorCache::fixed_windowed`].
    pub fn windowed(axis: u32, window: u64) -> Self {
        let name = fresh_sym_name();
        let mut k = TensorCache::fixed_named(axis, window.max(1), name.clone());
        let mut v = TensorCache::fixed_named(axis, window.max(1), name);
        if let Some(f) = k.fixed.as_mut() {
            f.window = Some(window.max(1));
        }
        if let Some(f) = v.fixed.as_mut() {
            f.window = Some(window.max(1));
        }
        Self { k, v }
    }

    pub fn is_fixed(&self) -> bool {
        self.k.is_fixed()
    }

    pub fn is_ring(&self) -> bool {
        self.k.is_ring()
    }

    /// Push this step's scatter outputs into a resolve batch.
    ///
    /// Runtime-rank, like [`TensorCache::pending`]: a resolve batch is a
    /// heterogeneous list of roots.
    pub fn pending_into(&self, batch: &mut Vec<Dyn>) {
        if let Some(k) = self.k.pending() {
            batch.push(k);
        }
        if let Some(v) = self.v.pending() {
            batch.push(v);
        }
    }

    /// Adopt both halves' resolved outputs. Call once per step, after the
    /// resolve that included [`KvCache::pending_into`]'s tensors.
    #[track_caller]
    pub fn commit(&mut self) {
        self.k.commit();
        self.v.commit();
    }

    /// Append one step's keys and values; returns the full cached pair.
    #[track_caller]
    pub fn append(&mut self, k: &Tensor<R, T>, v: &Tensor<R, T>) -> (Tensor<R, T>, Tensor<R, T>) {
        (self.k.append(k), self.v.append(v))
    }

    /// Whether [`KvCache::replay_append`] would rebuild the last append's nodes
    /// exactly. See [`TensorCache::can_replay`].
    pub fn can_replay(&self, added: u64) -> bool {
        self.k.can_replay(added) && self.v.can_replay(added)
    }

    /// Advance both halves without touching the graph. See
    /// [`TensorCache::replay_append`]; the caller must have checked
    /// [`KvCache::can_replay`], which covers both halves so neither can half-
    /// advance.
    pub fn replay_append(&mut self, added: u64) -> Result<()> {
        self.k.replay_append(added)?;
        self.v.replay_append(added)
    }

    /// The cached keys, or `None` before the first append — the reference's
    /// accessor spelling for the public `k` field.
    pub fn k(&self) -> Option<&Tensor<R, T>> {
        self.k.current()
    }

    /// The cached values, or `None` before the first append.
    pub fn v(&self) -> Option<&Tensor<R, T>> {
        self.v.current()
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
    use crate::session::{Backend, Session};
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::ir::Op;
    use fusor2_ir::ir::level0::L0;
    use fusor2_ir::egraph::Id;

    fn graph() -> crate::Graph {
        crate::Graph::new(&Session::new(Backend::cpu().unwrap()).unwrap())
    }

    fn upload(g: &crate::Graph, shape: &[u64], data: &[f32]) -> Dyn {
        let dims: Vec<Dim> = shape.iter().copied().map(Dim::Const).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        g.tensor(Dtype::F32, &dims, &bytes).unwrap()
    }

    fn typed<const R: usize>(g: &crate::Graph, shape: &[u64], data: &[f32]) -> Tensor<R, f32> {
        Tensor::from_dyn(upload(g, shape, data))
    }

    /// The `Restride` operands of the class `id` names, in order — the two
    /// halves a `cat` glues together.
    fn cat_sources(t: &Dyn) -> Vec<Id> {
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
        let mut cache: KvCache<4, f32> = KvCache::new(2);

        let k0: Tensor<4, f32> = typed(&g, &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let v0: Tensor<4, f32> = typed(&g, &[1, 1, 2, 2], &[-1.0, -2.0, -3.0, -4.0]);
        let (ks, vs) = cache.append(&k0, &v0);
        assert_eq!(cache.len(), Dim::Const(2));
        assert_eq!(ks.extent(2usize), Dim::Const(2));
        // The first append is the value itself, not a copy of it, so this
        // readback is a straight upload/download and asserts real numbers.
        assert_eq!(ks.id(), k0.id());
        assert_eq!(ks.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(vs.to_vec_f32(), vec![-1.0, -2.0, -3.0, -4.0]);

        // Step two: one more token. The cache grows along the cat axis only,
        // and the *new* value is the second operand — appending is ordered,
        // and reversing it silently corrupts a decode loop.
        let k1: Tensor<4, f32> = typed(&g, &[1, 1, 1, 2], &[5.0, 6.0]);
        let v1: Tensor<4, f32> = typed(&g, &[1, 1, 1, 2], &[-5.0, -6.0]);
        let (ks2, vs2) = cache.append(&k1, &v1);
        assert_eq!(cache.len(), Dim::Const(3));
        assert_eq!(ks2.shape(), [1, 1, 3, 2]);
        assert_eq!(vs2.shape(), ks2.shape());
        assert_eq!(cache.k.current().unwrap().id(), ks2.id());
        assert_eq!(cache.v.current().unwrap().id(), vs2.id());

        // Numerically this is `views::cat_dim*`'s obligation, and those cases
        // are red for a reason that is not this cache: the emitters index
        // every operand with the flat output index and ignore
        // `Operand::layout`. What is asserted here is the part the cache
        // owns — that the second step glues the *new* value on after the
        // cached one, in that order.
        let sources = cat_sources(ks2.as_dyn());
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
        let mut cache: TensorCache<2, f32> = TensorCache::new(0);
        let a: Tensor<2, f32> = typed(&g, &[1, 3], &[1.0, 2.0, 3.0]);
        let b: Tensor<2, f32> = typed(&g, &[2, 3], &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        cache.append(&a);
        let out = cache.append(&b);
        assert_eq!(cache.len(), Dim::Const(3));
        assert_eq!(out.shape(), [3, 3]);
    }

    #[test]
    fn eviction_narrows_to_the_newest_tokens() {
        let g = graph();
        let mut cache: TensorCache<2, f32> = TensorCache::new(0);
        cache.append(&typed(&g, &[4, 1], &[1.0, 2.0, 3.0, 4.0]));
        let kept = cache.keep_last(2).unwrap();
        assert_eq!(cache.len(), Dim::Const(2));
        assert_eq!(kept.shape(), [2, 1]);
        // A window wider than the cache is a no-op that does not renarrow.
        let same = cache.keep_last(8).unwrap();
        assert_eq!(same.id(), kept.id());
        assert_eq!(cache.len(), Dim::Const(2));
        // An empty cache has nothing to evict.
        assert!(TensorCache::<2, f32>::new(0).keep_last(2).is_none());
    }

    #[test]
    fn a_symbolic_length_accumulates_to_the_newest_symbol() {
        let g = graph();
        let s = g.sym("len");
        let mut cache: TensorCache<2, f32> = TensorCache::new(0);
        let t = g.leaf("x", &[s, Dim::Const(2)], Dtype::F32).unwrap();
        cache.append(&Tensor::from_dyn(t));
        assert_eq!(cache.len(), s);
        // And it cannot be evicted by a host-known window.
        assert!(cache.keep_last_inner(1).is_err());
    }

    /// A rejected append reports rather than panicking *and* leaves the cache
    /// exactly as it was. The typed `append` panics on the same conditions;
    /// `append_dyn` is the layer that has an `Error` to put them in, so the
    /// diagnoses are asserted there.
    #[test]
    fn a_mismatched_append_is_an_error_not_a_panic() {
        let g = graph();
        let mut cache: TensorCache<2, f32> = TensorCache::new(0);
        cache.append_dyn(&upload(&g, &[1, 3], &[1.0, 2.0, 3.0])).unwrap();
        // A non-cat axis that disagrees.
        assert!(cache.append_dyn(&upload(&g, &[1, 4], &[0.0; 4])).is_err());
        // A rank that disagrees.
        assert!(cache.append_dyn(&upload(&g, &[3], &[0.0; 3])).is_err());
        // The cache is untouched by a rejected append.
        assert_eq!(cache.len(), Dim::Const(1));
        // A dtype that disagrees.
        let u = g
            .tensor(Dtype::U32, &[Dim::Const(1), Dim::Const(3)], &[0u8; 12])
            .unwrap();
        assert!(cache.append_dyn(&u).is_err());
        // An axis off the end.
        assert!(
            TensorCache::<1, f32>::new(7)
                .append_dyn(&upload(&g, &[2], &[0.0; 2]))
                .is_err()
        );
    }

    /// The rank and element type are parameters now, so a rank-2 f16 cache is
    /// as ordinary as the rank-4 f32 default.
    #[test]
    fn a_cache_carries_its_rank_and_element_type() {
        let g = graph();
        let mut cache: TensorCache<2, half::f16> = TensorCache::new(0);
        let t = g
            .leaf("x", &[Dim::Const(2), Dim::Const(3)], Dtype::F16)
            .unwrap();
        let out = cache.append(&Tensor::from_dyn(t));
        assert_eq!(out.shape(), [2, 3]);
        assert_eq!(out.dtype(), Dtype::F16);
    }

    /// The whole point of the replay: it re-arms *the same nodes* a rebuild
    /// would have hash-consed onto, and advances the same state. If these two
    /// ever diverge, a decode loop reusing a memoized graph reads the wrong
    /// length or writes the wrong slot.
    #[test]
    fn a_replayed_append_matches_the_append_it_stands_in_for() {
        let g = graph();
        let mut replayed: TensorCache<4, f32> = TensorCache::fixed(2, 8);
        let v: Tensor<4, f32> = typed(&g, &[1, 1, 1, 2], &[1.0, 2.0]);
        replayed.append(&v);
        let first_out = replayed.pending().unwrap().id();

        // A second *real* append against the same store, on a copy.
        let mut rebuilt = replayed.clone();
        rebuilt.append(&v);
        let rebuilt_out = rebuilt.pending().unwrap().id();
        assert_eq!(
            rebuilt_out, first_out,
            "one store and one chunk width is one scatter node"
        );

        assert!(replayed.can_replay(1));
        replayed.replay_append(1).unwrap();
        assert_eq!(replayed.pending().unwrap().id(), rebuilt_out);
        assert_eq!(replayed.len(), rebuilt.len());
        let Dim::Sym(sym) = replayed.len() else {
            panic!("a fixed cache reads through its length symbol");
        };
        assert_eq!(
            g.handle().dim_binding(sym),
            Some(2),
            "the replay binds the length the rebuild would have"
        );
        // And the write index the next scatter reads names slot 1, not 0.
        assert_eq!(write_slots(&replayed), vec![1u32]);
    }

    /// The slots the cache's one-wide write index currently names.
    fn write_slots<const R: usize, T: Element>(cache: &TensorCache<R, T>) -> Vec<u32> {
        let idx = &cache.fixed.as_ref().unwrap().idx[&1];
        idx.graph()
            .leaf_bytes(idx.id())
            .unwrap()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }

    /// Everything that would make the rebuilt nodes differ refuses the replay
    /// rather than re-arming a stale scatter.
    #[test]
    fn a_replay_is_refused_whenever_a_rebuild_would_mint_new_nodes() {
        let g = graph();
        // Never armed.
        let mut fresh: TensorCache<4, f32> = TensorCache::fixed(2, 4);
        assert!(!fresh.can_replay(1));
        assert!(fresh.replay_append(1).is_err());

        let v: Tensor<4, f32> = typed(&g, &[1, 1, 1, 2], &[1.0, 2.0]);
        fresh.append(&v);
        assert!(fresh.can_replay(1));
        // A different chunk width is a different scatter.
        assert!(!fresh.can_replay(2));
        // Past capacity is a grown store, which is a new leaf.
        fresh.replay_append(1).unwrap();
        fresh.replay_append(1).unwrap();
        fresh.replay_append(1).unwrap();
        assert_eq!(write_slots(&fresh), vec![3u32], "four tokens, slots 0..3");
        assert!(!fresh.can_replay(1), "the fifth token needs a bigger store");
        // A reset cache has no readable value to advance.
        fresh.reset();
        assert!(!fresh.can_replay(1));
        // A cat-mode cache has no scatter at all.
        let mut cat: TensorCache<4, f32> = TensorCache::new(2);
        cat.append(&v);
        assert!(!cat.can_replay(1));

        // A ring replays forever and wraps, exactly as its appends do.
        let mut ring: TensorCache<4, f32> = TensorCache::fixed_windowed(2, 2);
        ring.append(&v);
        assert!(ring.can_replay(1));
        ring.replay_append(1).unwrap();
        ring.replay_append(1).unwrap();
        assert_eq!(
            write_slots(&ring),
            vec![0u32],
            "the third token of a two-slot ring writes slot 0"
        );
        assert!(!ring.can_replay(3), "an append wider than the window");
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
