//! The KV cache. Sequence length and storage capacity are symbolic extents,
//! so growing storage preserves the cached append graph.
//!
//! Two modes:
//!
//! * **cat** (the default): an append is a `cat` node and the whole cache is a
//!   fresh value each step with a concrete length. Every step's graph differs
//!   from the last; the session's shape families keep that from replanning
//!   per token.
//! * **fixed** ([`TensorCache::fixed`]): a fixed-capacity external leaf, an
//!   append is one `Scatter{Set}` at a device-side write index, and the
//!   readable cache is a `Dim::Sym`-length narrow of the scatter's output.
//!   Every step reuses the *same* nodes — only leaf bytes and the symbol's
//!   binding change — which is what lets a decode loop replay one plan per
//!   token. After the step's resolve, [`TensorCache::commit`] re-points the
//!   leaf at the buffer the scatter produced (no host round trip); a caller
//!   that never commits still sees every append, since appends chain through
//!   the uncommitted output, and [`TensorCache::detach`] commits for it.

use fusor_ir::dtype::Dtype;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::shape::{Dim, StrideSpec};
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
    /// The persistent capacity leaf and its readable prefix view.
    store: Option<(Dyn, Dyn)>,
    /// This step's scatter output, until [`TensorCache::commit`] adopts it.
    out: Option<Dyn>,
    /// The last append's `(chunk width, scatter output, readable view)`, retained past the
    /// commit that cleared `out`. An append of the same width against the same
    /// store rebuilds exactly these nodes, so [`TensorCache::replay_append`]
    /// re-arms them instead.
    arm: Option<(u64, Dyn, Dyn)>,
    /// `u32` write-index leaves, one per appended-chunk width, reused across
    /// steps with fresh bytes.
    idx: FxHashMap<u64, Dyn>,
    /// The symbol the readable length is bound to, named once per store.
    sym: Option<fusor_ir::shape::SymId>,
    /// The symbol's name. A [`KvCache`] hands both halves one name so the
    /// K and V views carry the same symbol — attention contracts their
    /// length axes against each other.
    sym_name: String,
}

/// A growable append-only tensor cache along one axis.
///
/// `R` is the rank of the values it holds and `T` their element type.
/// Both default to the decode shape — a rank-4 `[batch, heads, seq, dim]` f32
/// cache — so `TensorCache` alone still names it.
#[derive(Clone)]
pub struct TensorCache<const R: usize = 4, T: Element = f32> {
    data: Option<Tensor<R, T>>,
    axis: u32,
    len: Dim,
    fixed: Option<FixedState>,
}

impl<const R: usize, T: Element> TensorCache<R, T> {
    /// Create an empty growable cache along `axis`.
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
    /// is the initial slot count; it grows by doubling when exceeded.
    pub fn fixed(axis: u32, capacity: u64) -> Self {
        Self::fixed_named(axis, capacity, fresh_sym_name())
    }

    /// [`TensorCache::fixed`] with a caller-supplied length-symbol name;
    /// two caches sharing one name share one symbol.
    pub(crate) fn fixed_named(axis: u32, capacity: u64, sym_name: String) -> Self {
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
    /// Runtime-rank: a resolve batch is a heterogeneous list of roots.
    pub(crate) fn pending(&self) -> Option<Dyn> {
        self.fixed.as_ref().and_then(|f| f.out.clone())
    }

    /// Adopt the resolved scatter output into the store leaf and drop the
    /// output's binding so the next step re-dispatches. Call once per step,
    /// after the resolve that included the pending scatter output.
    #[track_caller]
    pub fn commit(&mut self) {
        let Some(f) = self.fixed.as_mut() else {
            return;
        };
        if let (Some((store, view)), Some(out)) = (f.store.as_ref(), f.out.take()) {
            ok("TensorCache::commit", store.adopt_buffer(&out));
            out.clear_device_buf();
            view.clear_device_buf();
            self.data = Some(Tensor::from_dyn(view.clone()));
        }
    }

    /// The cached tensor, or `None` before the first append.
    pub fn current(&self) -> Option<&Tensor<R, T>> {
        self.data.as_ref()
    }

    /// Cut the cached value off from the graph that produced it. A fixed
    /// cache commits: its store already holds every append, so this is the
    /// step's [`TensorCache::commit`] (resolving the pending write first if
    /// nothing has yet). A cat-mode cache re-leafs the value through the host.
    #[track_caller]
    pub fn detach(&mut self) {
        if self.fixed.is_some() {
            if let Some(out) = self.pending()
                && out.graph().device_buf(out.id).is_none()
            {
                ok(
                    "TensorCache::detach",
                    out.graph().session().resolve(std::slice::from_ref(&out)),
                );
            }
            self.commit();
            return;
        }
        if let Some(value) = self.data.as_ref().cloned() {
            self.data = Some(value.detach());
        }
    }

    /// Replace the cached value with a device-side leaf holding its buffer:
    /// [`TensorCache::detach`] without the host round trip. The value must
    /// have resolved (see [`Tensor::materialize`]).
    pub fn materialize(&mut self) {
        if let Some(value) = self.data.as_ref().cloned() {
            self.data = Some(value.materialize());
        }
    }

    /// The cached value at runtime rank, for the resolve-batch path.
    pub(crate) fn current_dyn(&self) -> Option<&Dyn> {
        self.data.as_ref().map(Tensor::as_dyn)
    }

    /// Tokens currently cached along the cache axis.
    pub fn len(&self) -> Dim {
        self.len
    }

    /// Whether the cache has no value.
    pub fn is_empty(&self) -> bool {
        self.data.is_none()
    }

    /// Append `value` along `axis` and return the whole cache, new part
    /// included. The first append stores `value` itself.
    #[track_caller]
    pub fn append(&mut self, value: &Tensor<R, T>) -> Tensor<R, T> {
        Tensor::from_dyn(ok("TensorCache::append", self.append_dyn(value.as_dyn())))
    }

    /// [`TensorCache::append`] at runtime rank.
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
        // must leave the cache exactly as it was.
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

        // A ring never grows; a plain cache doubles and migrates its prefix.
        if let Some(w) = f.window {
            if added > w {
                return Err(Error::Shape(format!(
                    "an append of {added} exceeds the {w}-token window"
                )));
            }
        } else {
            reserve(std::iter::once((&mut *f, axis)), added)?;
        }

        // The store leaf, minted on first use. No host bytes: wgpu (and the
        // CPU pool) zero-initialize, and nothing past the bound length is
        // ever read.
        if f.store.is_none() {
            let mut shape = value.shape().to_vec();
            shape[axis] = if f.window.is_some() {
                Dim::Const(f.capacity)
            } else {
                let capacity = graph.named_sym(&format!("{}:capacity", f.sym_name));
                graph.bind_dim(capacity, f.capacity);
                Dim::Sym(capacity)
            };
            let store = external_leaf(&graph, &shape, value.dtype())?;
            let sym = graph.named_sym(&f.sym_name);
            let view = readable(&store, axis, Dim::Sym(sym))?;
            f.sym = Some(sym);
            f.store = Some((store, view));
        }
        let sym = f.sym.expect("minted with the store");
        // An append before the last one's commit writes through that output,
        // so nothing is lost when a caller resolves several appends at once;
        // `commit` then adopts the final output.
        let store = f
            .out
            .clone()
            .unwrap_or_else(|| f.store.as_ref().expect("minted above").0.clone());

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

        let view = readable(&out, axis, Dim::Sym(sym))?;

        f.arm = Some((added, out.clone(), view.clone()));
        f.out = Some(out);
        self.len = Dim::Sym(sym);
        self.data = Some(Tensor::try_from_dyn(view.clone())?);
        Ok(view)
    }

    /// Whether [`TensorCache::replay_append`] would rebuild the last append's
    /// nodes exactly: same store, same index leaf, same chunk width, and
    /// enough storage for the next chunk.
    pub fn can_replay(&self, added: u64) -> bool {
        let Some(f) = self.fixed.as_ref() else {
            return false;
        };
        if f.store.is_none() || f.sym.is_none() {
            return false;
        }
        if f.arm.as_ref().is_none_or(|(w, _, _)| *w != added) {
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

    /// Advance one append without touching the graph: an append of the same
    /// width against the same store hash-conses onto the nodes the last one
    /// minted, so the only real work is the write index's bytes and the length
    /// binding. The caller must have checked [`TensorCache::can_replay`].
    pub fn replay_append(&mut self, added: u64) -> Result<()> {
        if !self.can_replay(added) {
            return Err(Error::Shape(
                "replay_append needs a fixed cache whose last append had the same width and \
                 whose storage has enough room; check can_replay first"
                    .into(),
            ));
        }
        let f = self.fixed.as_mut().expect("checked by can_replay");
        let (_, out, view) = f.arm.clone().expect("checked by can_replay");
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
        view.clear_device_buf();
        self.data = Some(Tensor::try_from_dyn(view)?);
        Ok(())
    }

    /// Keep the first `len` entries of a committed fixed cache, retaining its
    /// storage and append arm. Ring caches cannot discard an absolute suffix.
    pub fn truncate(&mut self, len: u64) -> Result<()> {
        let f = self
            .fixed
            .as_mut()
            .ok_or_else(|| Error::Shape("truncate requires a fixed cache".into()))?;
        if f.window.is_some() || f.out.is_some() {
            return Err(Error::Shape(
                "truncate requires a committed, non-ring cache".into(),
            ));
        }
        if len > f.len {
            return Err(Error::Shape("truncate cannot extend a cache".into()));
        }
        f.len = len;
        if let Some((_, view)) = &f.store {
            let sym = f.sym.expect("store has a length symbol");
            view.graph().bind_dim(sym, len);
            view.clear_device_buf();
        }
        if len == 0 {
            self.data = None;
            self.len = Dim::Const(0);
        }
        Ok(())
    }

    /// Keep the newest `len` tokens and drop the oldest.
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
        let kept =
            Tensor::try_from_dyn(data.narrow(axis, (total - len) as usize, len as usize)?)?;
        self.len = Dim::Const(len);
        self.data = Some(kept.clone());
        Ok(Some(kept))
    }

    /// Clear the readable cache while retaining reusable fixed buffers.
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

/// A process-unique name for a cache's length and capacity symbols.
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
    let id = graph.add_logical(Logical::Leaf(LeafKind::Buffer {
        name: graph.fresh_buffer_id(),
        dtype,
        shape: shape.iter().copied().collect(),
    }))?;
    Ok(graph.tensor(id))
}

fn readable(value: &Dyn, axis: usize, len: Dim) -> Result<Dyn> {
    let specs: Vec<_> = value
        .shape()
        .iter()
        .enumerate()
        .map(|(i, &dim)| StrideSpec::dim(i as u32, if i == axis { len } else { dim }))
        .collect();
    value.restride(&specs)
}

fn reserve<'a>(
    states: impl IntoIterator<Item = (&'a mut FixedState, usize)>,
    additional: u64,
) -> Result<()> {
    struct Growth<'a> {
        state: &'a mut FixedState,
        axis: usize,
        capacity: u64,
        moved: Option<Dyn>,
    }
    let mut growth = Vec::new();
    for (state, axis) in states {
        if state.window.is_some() {
            continue;
        }
        let needed = state
            .len
            .checked_add(additional)
            .ok_or_else(|| Error::Shape("cache length overflows u64".into()))?;
        if needed <= state.capacity {
            continue;
        }
        let mut capacity = state.capacity;
        while capacity < needed {
            capacity = capacity
                .checked_mul(2)
                .ok_or_else(|| Error::Shape("cache capacity overflows u64".into()))?;
        }
        let moved = match state
            .out
            .as_ref()
            .or_else(|| state.store.as_ref().map(|(store, _)| store))
        {
            Some(source) if state.len != 0 => {
                let graph = source.graph();
                let mut shape = source.shape().to_vec();
                shape[axis] = Dim::Const(capacity);
                let store = external_leaf(graph, &shape, source.dtype())?;
                let kept = source.narrow(axis, 0, state.len as usize)?;
                let idx = Dyn::arange(graph, Dtype::U32, 0., state.len as f64)?;
                Some(store.scatter_set(axis, &idx, &kept, true)?)
            }
            _ => None,
        };
        growth.push(Growth {
            state,
            axis,
            capacity,
            moved,
        });
    }
    let roots: Vec<Dyn> = growth.iter().filter_map(|g| g.moved.clone()).collect();
    if let Some(root) = roots.first() {
        root.graph().session().resolve(&roots)?;
    }
    for Growth {
        state,
        axis,
        capacity,
        moved,
    } in growth
    {
        state.capacity = capacity;
        if let Some((store, view)) = &state.store {
            if let Some(moved) = moved {
                let buffer = moved.graph().device_buf(moved.id()).ok_or_else(|| {
                    Error::Plan("cache migration did not produce a buffer".into())
                })?;
                store
                    .graph()
                    .set_device_buf_class(&[store.id()], &buffer, None);
                moved.clear_device_buf();
            } else {
                store.clear_device_buf();
            }
            let Dim::Sym(capacity_sym) = store.dim(axis) else {
                unreachable!("growable stores have symbolic capacity")
            };
            store.graph().bind_dim(capacity_sym, capacity);
            view.clear_device_buf();
        }
        state.out = None;
        if let Some((_, out, view)) = &state.arm {
            out.clear_device_buf();
            view.clear_device_buf();
        }
    }
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
/// `R` and `T` are the cached values'; a bare `KvCache` is a rank-4 f32 pair.
#[derive(Clone)]
pub struct KvCache<const R: usize = 4, T: Element = f32> {
    k: TensorCache<R, T>,
    v: TensorCache<R, T>,
}

impl<const R: usize, T: Element> KvCache<R, T> {
    /// Create an empty growable key/value cache along `axis`.
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

    /// Ring of the newest `window` tokens.
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

    /// Whether appends write into preallocated storage.
    pub fn is_fixed(&self) -> bool {
        self.k.is_fixed()
    }

    /// Whether the cache retains a fixed-size newest-token window.
    pub fn is_ring(&self) -> bool {
        self.k.is_ring()
    }

    /// Reserve space for `additional` tokens in every fixed cache, migrating
    /// existing values in one resolve. Empty stores remain unallocated and
    /// ring windows retain their configured capacity.
    pub fn reserve_all(caches: &mut [Self], additional: u64) -> Result<()> {
        reserve(
            caches
                .iter_mut()
                .flat_map(|cache| [&mut cache.k, &mut cache.v])
                .filter_map(|cache| {
                    cache
                        .fixed
                        .as_mut()
                        .map(|state| (state, cache.axis as usize))
                }),
            additional,
        )?;
        for cache in caches
            .iter_mut()
            .flat_map(|cache| [&mut cache.k, &mut cache.v])
        {
            if let Some(state) = &cache.fixed
                && state.out.is_none()
                && cache.data.is_some()
                && let Some((_, view)) = &state.store
            {
                cache.data = Some(Tensor::try_from_dyn(view.clone())?);
            }
        }
        Ok(())
    }

    /// Push this step's scatter outputs into a resolve batch.
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
        if let Some(added) = k
            .as_dyn()
            .shape()
            .get(self.k.axis as usize)
            .and_then(|d| d.as_const())
        {
            ok(
                "KvCache::append",
                Self::reserve_all(std::slice::from_mut(self), added),
            );
        }
        (self.k.append(k), self.v.append(v))
    }

    /// Keep only the newest `len` entries in both halves.
    pub fn keep_last(&mut self, len: u64) -> Option<(Tensor<R, T>, Tensor<R, T>)> {
        self.k.keep_last(len).zip(self.v.keep_last(len))
    }

    /// Keep the first `len` entries of both committed fixed caches.
    pub fn truncate(&mut self, len: u64) -> Result<()> {
        self.k.truncate(len)?;
        self.v.truncate(len)
    }

    /// Replace both cached values with detached leaves after they resolve.
    pub fn detach(&mut self) {
        self.k.detach();
        self.v.detach();
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

    /// The cached keys, or `None` before the first append.
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

    /// Whether no key/value pair has been appended.
    pub fn is_empty(&self) -> bool {
        self.k.is_empty()
    }

    /// Clear both halves while retaining reusable fixed buffers.
    pub fn reset(&mut self) {
        self.k.reset();
        self.v.reset();
    }
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use crate::Device;

    fn device() -> Device {
        if std::env::var_os("FUSOR_CONFORMANCE_REQUIRE_GPU").is_some() {
            #[cfg(feature = "gpu")]
            return Device::gpu_blocking().expect("GPU backend required");
            #[cfg(not(feature = "gpu"))]
            panic!("GPU feature required");
        }
        Device::try_cpu().unwrap()
    }

    fn commit(device: &Device, caches: &mut [KvCache<3>]) {
        let mut roots = Vec::new();
        for cache in &*caches {
            cache.pending_into(&mut roots);
        }
        device.session().resolve(&roots).unwrap();
        for cache in caches {
            cache.commit();
        }
    }

    fn check(caches: &[KvCache<3>], total: usize) {
        for (layer, cache) in caches.iter().enumerate() {
            let expected: Vec<f32> = (0..2)
                .flat_map(|row| {
                    (0..total).flat_map(move |step| {
                        (0..2).map(move |col| {
                            (layer * 100 + step.min(2) * 10 + row * 2 + col + 1) as f32
                        })
                    })
                })
                .collect();
            assert_eq!(cache.k().unwrap().narrow(1, 0, total).to_flat(), expected);
            assert_eq!(
                cache.v().unwrap().narrow(1, 0, total).to_flat(),
                expected.iter().map(|x| x * 2.).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn growing_fixed_caches_preserves_pending_appends() {
        let nodes = |caches: &[KvCache<3>]| {
            caches
                .iter()
                .flat_map(|cache| [&cache.k, &cache.v])
                .map(|cache| {
                    let state = cache.fixed.as_ref().unwrap();
                    (
                        state.store.as_ref().unwrap().0.id(),
                        state.arm.as_ref().map(|(_, out, _)| out.id()),
                    )
                })
                .collect::<Vec<_>>()
        };
        for reserved in [false, true] {
            let device = device();
            let capacity = if reserved { 1 } else { 2 };
            let mut caches = [
                KvCache::<3>::with_capacity(1, capacity),
                KvCache::with_capacity(1, capacity),
            ];
            if reserved {
                KvCache::reserve_all(&mut caches, 2).unwrap();
                for cache in &caches {
                    assert!(cache.k.fixed.as_ref().unwrap().store.is_none());
                    assert!(cache.v.fixed.as_ref().unwrap().store.is_none());
                }
            }
            let total = if reserved { 8 } else { 4 };
            for step in 0..total {
                for (layer, cache) in caches.iter_mut().enumerate() {
                    if step >= 3 {
                        cache.replay_append(1).unwrap();
                    } else {
                        let base = (layer * 100 + step * 10) as f32;
                        let k = Tensor::from_slice(
                            &device,
                            [2, 1, 2],
                            &[base + 1., base + 2., base + 3., base + 4.],
                        );
                        cache.append(&k, &k.mul_scalar(2.));
                    }
                }
                if step == 1 {
                    if reserved {
                        let before = nodes(&caches);
                        KvCache::reserve_all(&mut caches, 6).unwrap();
                        assert_eq!(nodes(&caches), before);
                        assert!(caches.iter().all(|cache| cache.can_replay(1)));
                        check(&caches, 2);
                    }
                } else {
                    commit(&device, &mut caches);
                    if step == 0 && reserved {
                        assert!(caches.iter().all(|cache| cache.can_replay(1)));
                        let before = nodes(&caches);
                        KvCache::reserve_all(&mut caches, 3).unwrap();
                        assert_eq!(nodes(&caches), before);
                        assert!(caches.iter().all(|cache| cache.can_replay(1)));
                        check(&caches, 1);
                    }
                }
            }
            check(&caches, total);
        }
    }

    #[test]
    fn reserving_a_ring_preserves_its_window() {
        let device = device();
        let mut caches = [KvCache::<3>::windowed(1, 2)];
        for step in 0..3 {
            KvCache::reserve_all(&mut caches, 100).unwrap();
            let value = Tensor::from_slice(&device, [2, 1, 1], &[step as f32, (step + 10) as f32]);
            caches[0].append(&value, &value);
            commit(&device, &mut caches);
        }
        assert_eq!(
            caches[0].k().unwrap().narrow(1, 0, 2).to_flat(),
            [2., 1., 12., 11.]
        );
        assert_eq!(
            caches[0].v().unwrap().narrow(1, 0, 2).to_flat(),
            [2., 1., 12., 11.]
        );
    }

    #[test]
    fn truncating_a_fixed_cache_replays_from_the_retained_prefix() {
        let device = device();
        let mut caches = [KvCache::<3>::with_capacity(1, 32)];
        let input = Tensor::<3>::leaf(&device, [Dim::Const(2), Dim::Const(16), Dim::Const(2)]);
        let values = input.mul_scalar(2.);
        let read = |tensor: &Tensor<3>| {
            tensor
                .as_dyn()
                .to_bytes()
                .unwrap()
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let mut expected = [Vec::new(), Vec::new()];
        for (step, keep) in [5, 13].into_iter().enumerate() {
            let data: Vec<f32> = (0..64).map(|i| (step * 100 + i) as f32).collect();
            input.set_elements(&data);
            if step == 0 {
                caches[0].append(&input, &values);
            } else {
                caches[0].replay_append(16).unwrap();
            }
            assert!(caches[0].truncate(keep).is_err());
            commit(&device, &mut caches);
            read(caches[0].k().unwrap());
            read(caches[0].v().unwrap());
            caches[0].truncate(keep).unwrap();
            for (row, prefix) in expected.iter_mut().enumerate() {
                prefix.extend_from_slice(&data[row * 32..(row + 1) * 32]);
                prefix.truncate(keep as usize * 2);
            }
            let expected: Vec<f32> = expected.iter().flatten().copied().collect();
            assert_eq!(read(caches[0].k().unwrap()), expected);
            assert_eq!(
                read(caches[0].v().unwrap()),
                expected.iter().map(|x| x * 2.).collect::<Vec<_>>()
            );
            assert!(caches[0].can_replay(16));
        }
        caches[0].truncate(0).unwrap();
        assert!(caches[0].is_empty());
        KvCache::reserve_all(&mut caches, 64).unwrap();
        caches[0].replay_append(16).unwrap();
        commit(&device, &mut caches);
        caches[0].truncate(3).unwrap();
        assert_eq!(
            read(caches[0].k().unwrap()),
            (100..106)
                .chain(132..138)
                .map(|x| x as f32)
                .collect::<Vec<_>>()
        );
    }
}
