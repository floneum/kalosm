//! `Graph` and `Gradients`.
//!
//! The backward transform's output is ingested **together with the forward as
//! one graph with one root set**, which is what makes gradient checkpointing
//! the extractor's materialization bit. Nobody writes a checkpointing pass and
//! there is no user annotation.
//!
//! Owned by W13.

use std::sync::Arc;

use fusor2_autograd::custom::{CustomRegistry, straight_through, with_backwards as register_custom};
use fusor2_autograd::tape::{GraphTape, splat_of};
use fusor2_ir::autograd::{AdjointFn, Parent};
use fusor2_ir::dtype::{Dtype, QFmt, QLayout};
use fusor2_ir::egraph::{EGraph, Id};
use fusor2_ir::ir::level0::{BufferId, L0, LeafKind};
use fusor2_ir::ir::{AttrId, Op};
use fusor2_ir::shape::{Dim, SymId};
use fusor2_ir::target::Buf;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::composite::MacroAttr;
use crate::session::Session;
use crate::tensor::Tensor;
use crate::{Error, Result};

/// Shared handle to the graph a [`Tensor`] belongs to.
pub type GraphRef = Arc<GraphInner>;

/// Host bytes waiting to be uploaded for one external leaf, plus the device
/// buffer once one exists. A `Persistent` leaf keeps its buffer across
/// resolves; a `Step` leaf is re-uploaded whenever its bytes change.
#[derive(Default)]
pub(crate) struct LeafStore {
    pub(crate) bytes: FxHashMap<Id, Vec<u8>>,
    pub(crate) device: FxHashMap<Id, Buf>,
    /// The layout the buffer was *written* under, when it is not the value's
    /// own dense shape. A selected `Coop`/`Sgemm` geometry pads the output to
    /// its tile multiple, so a readback that assumed contiguity would hand
    /// back the first row plus padding.
    pub(crate) layout: FxHashMap<Id, fusor2_ir::shape::Layout>,
}

/// Symbolic bindings: extents for `Dim::Sym` and values for `Leaf::Uniform`.
/// Neither ever enters a kernel's identity — both are words in binding 0.
#[derive(Default)]
pub(crate) struct SymbolStore {
    pub(crate) dims: FxHashMap<SymId, u64>,
    pub(crate) scalars: FxHashMap<SymId, f32>,
    pub(crate) named: FxHashMap<String, SymId>,
}

/// The mutable graph state behind a [`GraphRef`].
pub struct GraphInner {
    pub(crate) egraph: Mutex<EGraph>,
    pub(crate) session: Session,
    pub(crate) params: Mutex<FxHashMap<String, Id>>,
    /// The `AttrId` side table. Attributes live outside `Op` so `Op` stays
    /// `Hash + Eq` and the hash-cons memo stays exact.
    pub(crate) attrs: Mutex<Vec<MacroAttr>>,
    pub(crate) leaves: Mutex<LeafStore>,
    pub(crate) symbols: Mutex<SymbolStore>,
    pub(crate) custom: Mutex<CustomRegistry>,
    next_buffer: Mutex<u32>,
    /// Content-addressed names for immutable host-byte leaves. See
    /// [`GraphInner::constant_leaf`].
    constants: Mutex<FxHashMap<ConstKey, Id>>,
    /// Serializes *whole* resolve-and-read sequences against this graph.
    ///
    /// [`crate::session::Session::resolve`] cannot hold [`Self::egraph`] for
    /// its own duration: saturation and extraction need it, but so do
    /// `selected`, `bind_class` and `leaf_buffer` inside the dispatch that
    /// follows, and the mutex is not reentrant. Releasing it between the two
    /// halves is what let two threads interleave — one binding a freshly
    /// allocated output buffer into [`Self::leaves`] while the other's
    /// `read_bytes` found that buffer, saw nothing in flight, and downloaded
    /// a buffer no dispatch had written yet. That returns **zeros**, not an
    /// error. Measured before this existed: 4 bad readbacks in 640 across 8
    /// threads on one `Device::cpu()`.
    ///
    /// A separate lock rather than a reentrant one, because the section that
    /// has to be atomic is larger than any single e-graph operation:
    /// `read_back` holds this across `resolve` *and* `read_bytes`, so a
    /// concurrent resolve cannot land between them either.
    pub(crate) resolve_lock: Mutex<()>,
}

/// The identity of an immutable host-byte leaf: everything a reader of that
/// buffer can observe.
#[derive(PartialEq, Eq, Hash)]
struct ConstKey {
    dtype: Dtype,
    shape: Vec<Dim>,
    bytes: Vec<u8>,
}

impl GraphInner {
    /// Run `f` with exclusive access to the e-graph.
    pub fn with_egraph<T>(&self, f: impl FnOnce(&mut EGraph) -> Result<T>) -> Result<T> {
        let mut g = self.egraph.lock();
        f(&mut g)
    }

    /// Run `f` with a construction tape over the e-graph. Every macro-op
    /// `defn` is built through this, so forward and backward share one node
    /// arena.
    pub fn build<T>(&self, f: impl FnOnce(&mut GraphTape<'_>) -> Result<T>) -> Result<T> {
        let mut g = self.egraph.lock();
        let mut tape = GraphTape::new(&mut g);
        f(&mut tape)
    }

    pub fn add(&self, op: Op) -> Result<Id> {
        self.egraph.lock().add(op)
    }

    pub fn add_l0(&self, op: L0) -> Result<Id> {
        self.egraph.lock().add(Op::L0(op))
    }

    pub fn union(&self, a: Id, b: Id) -> Result<Id> {
        self.egraph.lock().union(a, b)
    }

    pub fn mark_defn(&self, id: Id) {
        self.egraph.lock().mark_defn(id);
    }

    /// Mark `value` straight-through: forward opaque, adjoint the identity
    /// into operand 0, which must be `x`. The `GraphRef`-level spelling of
    /// [`Graph::straight_through`], so an op that builds its own opaque node
    /// can declare it without a `Graph` handle.
    pub fn register_straight_through(&self, value: Id, x: Id) -> Result<()> {
        let mut reg = self.custom.lock();
        straight_through(&mut reg, value, x)?;
        Ok(())
    }

    /// The `GraphRef`-level spelling of [`Graph::with_backwards`].
    pub fn register_backward(
        &self,
        value: Id,
        parents: &[fusor2_ir::autograd::Parent],
        rule: AdjointFn,
    ) -> Result<()> {
        let mut reg = self.custom.lock();
        register_custom(&mut reg, value, parents, rule)?;
        Ok(())
    }

    pub fn facts(&self, id: Id) -> fusor2_ir::facts::ValueFacts {
        self.egraph.lock().facts(id).clone()
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Wrap a node id as a user-facing tensor.
    pub fn tensor(self: &Arc<Self>, id: Id) -> Tensor {
        Tensor {
            id,
            graph: Arc::clone(self),
        }
    }

    /// Intern a macro attribute blob. Equal attributes share an id, so two
    /// identically-configured macro ops hash-cons together.
    pub fn intern_attrs(&self, attrs: MacroAttr) -> AttrId {
        let mut table = self.attrs.lock();
        if let Some(i) = table.iter().position(|a| *a == attrs) {
            return AttrId(i as u32);
        }
        table.push(attrs);
        AttrId((table.len() - 1) as u32)
    }

    pub fn attrs_of(&self, id: AttrId) -> Option<MacroAttr> {
        self.attrs.lock().get(id.0 as usize).cloned()
    }

    /// The one `BufferId` allocator. Every leaf name in a graph comes from
    /// here, so no two leaves can share a name by accident.
    pub(crate) fn fresh_buffer_id(&self) -> BufferId {
        let mut next = self.next_buffer.lock();
        let id = BufferId(*next);
        *next += 1;
        id
    }

    /// An immutable rank-N leaf holding `bytes`, named by its **content**.
    ///
    /// A leaf's hash-cons key is its `LeafKind`, and host bytes live in a side
    /// table that is not part of that key. So a caller that mints two constants
    /// of the same dtype and shape under one name gets **one node**, and the
    /// second `set_leaf_bytes` silently overwrites the first — which is how
    /// rope's permutation vector became its table-expansion vector. Naming the
    /// leaf by its content makes the key exact in both directions: equal
    /// constants still share a node, unequal ones cannot.
    pub(crate) fn constant_leaf(&self, dtype: Dtype, shape: &[Dim], bytes: Vec<u8>) -> Result<Id> {
        let key = ConstKey {
            dtype,
            shape: shape.to_vec(),
            bytes,
        };
        if let Some(id) = self.constants.lock().get(&key).copied() {
            return Ok(id);
        }
        let id = self.add_l0(L0::Leaf(LeafKind::Buffer {
            name: self.fresh_buffer_id(),
            dtype,
            shape: shape.iter().copied().collect(),
        }))?;
        self.set_leaf_bytes(id, key.bytes.clone());
        self.constants.lock().insert(key, id);
        Ok(id)
    }

    /// A fresh symbolic quantity. Allocated by the e-graph so a frontend
    /// symbol can never collide with one a rule mints (`fold_split`'s block
    /// count).
    pub fn fresh_sym(&self) -> SymId {
        self.egraph.lock().fresh_sym()
    }

    /// A named symbol, created on first use. Two calls with the same name
    /// return the same `SymId`, so `graph.sym("seq")` is stable across a
    /// decode loop.
    pub fn named_sym(&self, name: &str) -> SymId {
        if let Some(s) = self.symbols.lock().named.get(name).copied() {
            return s;
        }
        let s = self.fresh_sym();
        self.symbols.lock().named.insert(name.to_string(), s);
        s
    }

    /// Bind a symbolic extent for the next resolve.
    pub fn bind_dim(&self, sym: SymId, value: u64) {
        self.symbols.lock().dims.insert(sym, value);
    }

    pub fn dim_binding(&self, sym: SymId) -> Option<u64> {
        self.symbols.lock().dims.get(&sym).copied()
    }

    /// Write a runtime scalar. A learning rate set here is a word in
    /// binding 0, never a baked literal, so changing it recompiles nothing.
    pub fn set_uniform(&self, sym: SymId, value: f32) {
        self.symbols.lock().scalars.insert(sym, value);
    }

    pub fn uniform_value(&self, sym: SymId) -> Option<f32> {
        self.symbols.lock().scalars.get(&sym).copied()
    }

    /// Every `(sym, value)` runtime scalar declared so far.
    pub fn uniform_scalars(&self) -> Vec<(SymId, f32)> {
        let mut out: Vec<(SymId, f32)> = self
            .symbols
            .lock()
            .scalars
            .iter()
            .map(|(s, v)| (*s, *v))
            .collect();
        out.sort_by_key(|(s, _)| *s);
        out
    }

    /// Every `(sym, extent)` dim binding declared so far.
    pub fn dim_bindings(&self) -> Vec<(SymId, u64)> {
        let mut out: Vec<(SymId, u64)> = self
            .symbols
            .lock()
            .dims
            .iter()
            .map(|(s, v)| (*s, *v))
            .collect();
        out.sort_by_key(|(s, _)| *s);
        out
    }

    /// Attach host bytes to an external leaf.
    pub fn set_leaf_bytes(&self, id: Id, bytes: Vec<u8>) {
        let mut store = self.leaves.lock();
        store.bytes.insert(id, bytes);
        // A changed upload invalidates the device copy, never the plan.
        store.device.remove(&id);
    }

    pub fn leaf_bytes(&self, id: Id) -> Option<Vec<u8>> {
        self.leaves.lock().bytes.get(&id).cloned()
    }

    /// The device buffer backing `id`, once a resolve has produced one.
    pub fn device_buf(&self, id: Id) -> Option<Buf> {
        self.leaves.lock().device.get(&id).cloned()
    }

    pub(crate) fn set_device_buf(&self, id: Id, buf: Buf) {
        let mut store = self.leaves.lock();
        store.device.insert(id, buf);
        store.layout.remove(&id);
    }

    /// Register a buffer whose bytes are laid out as `layout` rather than as
    /// the value's own dense shape.
    pub(crate) fn set_device_buf_with_layout(
        &self,
        id: Id,
        buf: Buf,
        layout: fusor2_ir::shape::Layout,
    ) {
        let mut store = self.leaves.lock();
        store.device.insert(id, buf);
        store.layout.insert(id, layout);
    }

    pub(crate) fn device_layout(&self, id: Id) -> Option<fusor2_ir::shape::Layout> {
        self.leaves.lock().layout.get(&id).cloned()
    }

    /// Resolve `id` and copy its bytes back to the host. One of exactly
    /// three host syncs.
    ///
    /// The guard spans both halves: a concurrent resolve landing between them
    /// could bind a fresh output buffer for a class this read is about to
    /// download, and downloading a buffer whose dispatch has not run yet
    /// returns zeros rather than an error. See [`GraphInner::resolve_lock`].
    pub fn read_back(self: &Arc<Self>, id: Id) -> Result<Vec<u8>> {
        let tensor = self.tensor(id);
        let resolving = self.resolve_lock.lock();
        self.session
            .resolve_locked(&resolving, std::slice::from_ref(&tensor))?;
        self.session.read_bytes_locked(&resolving, self, id)
    }
}

/// A program under construction.
#[derive(Clone)]
pub struct Graph {
    inner: GraphRef,
}

impl Graph {
    pub fn new(session: &Session) -> Self {
        Self {
            inner: Arc::new(GraphInner {
                egraph: Mutex::new(EGraph::new(session.semantics())),
                session: session.clone(),
                params: Mutex::new(FxHashMap::default()),
                attrs: Mutex::new(Vec::new()),
                leaves: Mutex::new(LeafStore::default()),
                symbols: Mutex::new(SymbolStore::default()),
                custom: Mutex::new(CustomRegistry::new()),
                next_buffer: Mutex::new(0),
                constants: Mutex::new(FxHashMap::default()),
                resolve_lock: Mutex::new(()),
            }),
        }
    }

    pub fn handle(&self) -> &GraphRef {
        &self.inner
    }

    pub fn session(&self) -> &Session {
        &self.inner.session
    }

    /// A trainable parameter. `Persistence::Persistent`, so a quantized repack
    /// amortizes against its lifetime and the extractor knows it may not
    /// recompute it.
    pub fn param(&self, name: &str, shape: &[Dim], dtype: Dtype) -> Result<Tensor> {
        let id = self.inner.add_l0(L0::Leaf(LeafKind::Param {
            name: self.inner.fresh_buffer_id(),
            dtype,
            shape: shape.iter().copied().collect(),
        }))?;
        self.inner.params.lock().insert(name.to_string(), id);
        Ok(self.inner.tensor(id))
    }

    /// A step-local input buffer.
    pub fn leaf(&self, name: &str, shape: &[Dim], dtype: Dtype) -> Result<Tensor> {
        let _ = name;
        let id = self.inner.add_l0(L0::Leaf(LeafKind::Buffer {
            name: self.inner.fresh_buffer_id(),
            dtype,
            shape: shape.iter().copied().collect(),
        }))?;
        Ok(self.inner.tensor(id))
    }

    /// A step-local buffer with host contents.
    pub fn constant_from_raw(&self, dtype: Dtype, shape: &[Dim], bytes: &[u8]) -> Result<Tensor> {
        let t = self.leaf("", shape, dtype)?;
        self.inner.set_leaf_bytes(t.id, bytes.to_vec());
        Ok(t)
    }

    /// Upload dense host data. The preserved spelling of the reference's
    /// `Graph::tensor`.
    pub fn tensor(&self, dtype: Dtype, shape: &[Dim], bytes: &[u8]) -> Result<Tensor> {
        self.constant_from_raw(dtype, shape, bytes)
    }

    /// A splat constant. Folded into the kernel; never a buffer.
    pub fn constant(&self, value: f32, shape: &[Dim], dtype: Dtype) -> Result<Tensor> {
        let id = self.inner.add_l0(L0::Leaf(LeafKind::Const {
            value: splat_of(dtype, value)?,
            shape: shape.iter().copied().collect(),
        }))?;
        Ok(self.inner.tensor(id))
    }

    /// A block-quantized weight leaf.
    pub fn quantized(
        &self,
        fmt: QFmt,
        layout: QLayout,
        shape: [Dim; 2],
        bytes: &[u8],
    ) -> Result<Tensor> {
        let id = self.inner.add_l0(L0::Leaf(LeafKind::Quantized {
            name: self.inner.fresh_buffer_id(),
            fmt,
            layout,
            shape: shape.into_iter().collect(),
        }))?;
        self.inner.set_leaf_bytes(id, bytes.to_vec());
        Ok(self.inner.tensor(id))
    }

    /// A runtime scalar read from binding 0. **Not** a `[1]` tensor and not a
    /// literal: `m * lr` built on one of these recompiles nothing when the
    /// learning rate moves.
    pub fn uniform(&self, name: &str) -> Result<Tensor> {
        self.uniform_typed(name, Dtype::F32)
    }

    pub fn uniform_typed(&self, name: &str, dtype: Dtype) -> Result<Tensor> {
        let sym = self.inner.named_sym(name);
        let id = self
            .inner
            .add_l0(L0::Leaf(LeafKind::Uniform { sym, dtype }))?;
        Ok(self.inner.tensor(id))
    }

    /// Write a runtime scalar declared with [`Graph::uniform`].
    pub fn set_uniform(&self, name: &str, value: f32) {
        let sym = self.inner.named_sym(name);
        self.inner.set_uniform(sym, value);
    }

    /// A fresh symbolic dim, bound at dispatch and never at compile.
    pub fn sym(&self, name: &str) -> Dim {
        Dim::Sym(self.inner.named_sym(name))
    }

    /// Bind a symbolic dim's extent for the next resolve.
    pub fn bind(&self, name: &str, value: u64) {
        let sym = self.inner.named_sym(name);
        self.inner.bind_dim(sym, value);
    }

    /// Intern a macro attribute blob.
    pub fn intern_attrs(&self, attrs: MacroAttr) -> AttrId {
        self.inner.intern_attrs(attrs)
    }

    /// Attach a user-supplied backward to `value`, declaring its parents.
    ///
    /// The rule is a bare `fn` and its targets are bare node ids, so a
    /// closure can never close an `Arc` cycle over the graph. A rule that
    /// omits a `Parent { requires_grad: true }` is an error.
    pub fn with_backwards(
        &self,
        value: &Tensor,
        parents: &[Parent],
        rule: AdjointFn,
    ) -> Result<Tensor> {
        let mut reg = self.inner.custom.lock();
        register_custom(&mut reg, value.id, parents, rule)?;
        Ok(value.clone())
    }

    /// Mark `value` straight-through: forward opaque, adjoint the identity
    /// into `x`. This is the whole of QAT fake-quant.
    pub fn straight_through(&self, value: &Tensor, x: &Tensor) -> Result<Tensor> {
        let mut reg = self.inner.custom.lock();
        straight_through(&mut reg, value.id, x.id)?;
        Ok(value.clone())
    }

    /// Differentiate `loss` with respect to every parameter, seeded with ones.
    pub fn backward(&self, loss: &Tensor) -> Result<Gradients> {
        self.owns(loss, "loss")?;
        let seed = self.ones_like(loss)?;
        let wrt = self.parameter_ids();
        self.backward_ids(loss, seed, &wrt)
    }

    /// Differentiate with respect to an explicit set.
    pub fn backward_with(&self, loss: &Tensor, wrt: &[Tensor]) -> Result<Gradients> {
        self.owns(loss, "loss")?;
        for t in wrt {
            self.owns(t, "wrt")?;
        }
        let seed = self.ones_like(loss)?;
        let ids: Vec<Id> = wrt.iter().map(|t| t.id).collect();
        self.backward_ids(loss, seed, &ids)
    }

    /// Differentiate with an explicit seed — the loss-scale entry point.
    pub fn backward_seeded(
        &self,
        loss: &Tensor,
        seed: &Tensor,
        wrt: &[Tensor],
    ) -> Result<Gradients> {
        self.owns(loss, "loss")?;
        self.owns(seed, "seed")?;
        for t in wrt {
            self.owns(t, "wrt")?;
        }
        let ids: Vec<Id> = wrt.iter().map(|t| t.id).collect();
        self.backward_ids(loss, seed.id, &ids)
    }

    /// The subset of `candidates` that `value` actually depends on.
    ///
    /// [`Graph::backward_with`] refuses a `wrt` the loss cannot reach, which
    /// is right when the caller named it: an unreachable `wrt` is a typo or a
    /// stray `detach`. A partial backward driven by [`crate::autograd`] is the
    /// other case — it hands over a whole frontier and expects most of it to
    /// be behind whatever it is descending from — so it filters first rather
    /// than asking for a weaker error.
    ///
    /// Structural, over `children`: the e-graph only ever adds, so the
    /// construction chain is still there to walk, and equal values are
    /// compared by class rather than by id.
    pub fn reachable_from(&self, value: &Tensor, candidates: &[Tensor]) -> Vec<Tensor> {
        let g = self.inner.egraph.lock();
        let mut want: FxHashMap<fusor2_ir::egraph::ClassId, Vec<usize>> = FxHashMap::default();
        for (i, c) in candidates.iter().enumerate() {
            want.entry(g.class_of(c.id)).or_default().push(i);
        }
        let mut hit = vec![false; candidates.len()];
        let mut seen: rustc_hash::FxHashSet<Id> = rustc_hash::FxHashSet::default();
        let mut stack = vec![value.id];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            if let Some(idx) = want.get(&g.class_of(id)) {
                for i in idx {
                    hit[*i] = true;
                }
            }
            for child in g.node(id).children.iter() {
                stack.push(*child);
            }
        }
        candidates
            .iter()
            .zip(hit)
            .filter(|(_, reached)| *reached)
            .map(|(t, _)| t.clone())
            .collect()
    }

    /// Refuse a tensor from another graph.
    ///
    /// An `Id` is an index into *one* e-graph's arena, so a foreign one either
    /// names an unrelated node or is out of range — the latter panicked in
    /// `facts()` before reaching any check.
    fn owns(&self, t: &Tensor, role: &str) -> Result<()> {
        if Arc::ptr_eq(&t.graph, &self.inner) {
            return Ok(());
        }
        Err(Error::Plan(format!(
            "the {role} tensor belongs to a different graph; a backward pass cannot cross graphs"
        )))
    }

    pub fn gradients(&self, loss: &Tensor) -> Result<Gradients> {
        self.backward(loss)
    }

    /// Every parameter leaf, in creation order.
    pub fn parameters(&self) -> Vec<Tensor> {
        self.parameter_ids()
            .into_iter()
            .map(|id| self.inner.tensor(id))
            .collect()
    }

    fn parameter_ids(&self) -> Vec<Id> {
        let mut ids: Vec<Id> = self.inner.params.lock().values().copied().collect();
        ids.sort_unstable();
        ids
    }

    fn ones_like(&self, t: &Tensor) -> Result<Id> {
        let facts = self.inner.facts(t.id);
        self.inner.add_l0(L0::Leaf(LeafKind::Const {
            value: splat_of(facts.dtype, 1.0)?,
            shape: facts.shape.clone(),
        }))
    }

    fn backward_ids(&self, loss: &Tensor, seed: Id, wrt: &[Id]) -> Result<Gradients> {
        if !Arc::ptr_eq(&loss.graph, &self.inner) {
            return Err(Error::Device(
                "backward across two graphs is not a thing".into(),
            ));
        }
        let caps = self.inner.session.caps();
        let custom = self.inner.custom.lock().clone();
        let mut g = self.inner.egraph.lock();
        let grads = fusor2_autograd::backward::backward_into_with(
            &mut g, &caps, loss.id, seed, wrt, &custom,
        )?;
        // Forward and backward are one graph with one root set: that is what
        // makes "save this activation" versus "recompute it" the extractor's
        // materialization bit rather than a pass anybody writes.
        g.add_root(loss.id);
        let mut entries = FxHashMap::default();
        for (primal, grad) in wrt.iter().zip(&grads) {
            if let Some(grad) = grad {
                g.add_root(*grad);
                entries.insert(*primal, *grad);
            }
        }
        Ok(Gradients { entries })
    }
}

/// The gradient of one backward call, keyed by the primal tensor.
#[derive(Clone, Default)]
pub struct Gradients {
    entries: FxHashMap<Id, Id>,
}

impl Gradients {
    pub fn get(&self, of: &Tensor) -> Option<Tensor> {
        let id = self.entries.get(&of.id).copied()?;
        Some(of.graph.tensor(id))
    }

    /// `(primal, gradient)` pairs, in primal id order.
    pub fn pairs(&self) -> Vec<(Id, Id)> {
        let mut out: Vec<(Id, Id)> = self.entries.iter().map(|(a, b)| (*a, *b)).collect();
        out.sort_unstable();
        out
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop the graph aliases, keeping only the ids. The replacement for the
    /// reference's `into_detached`, which existed to sever lazy expression
    /// trees; here there is nothing to sever.
    pub fn into_detached(self) -> FxHashMap<Id, Id> {
        self.entries
    }
}

/// A parent declaration for [`Graph::with_backwards`].
pub fn parent(t: &Tensor, requires_grad: bool) -> Parent {
    Parent {
        value: t.id,
        requires_grad,
    }
}

/// The gradient slot of a tensor: a bare node id, never a handle.
pub fn slot(t: &Tensor) -> fusor2_ir::autograd::GradientSlot {
    fusor2_ir::autograd::GradientSlot(t.id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Device, Session};

    fn graph() -> Graph {
        let session = Session::new(Device::cpu().expect("cpu device")).expect("session");
        Graph::new(&session)
    }

    #[test]
    fn a_parameter_leaf_is_persistent_and_a_buffer_leaf_is_not() {
        let g = graph();
        let w = g.param("w", &[Dim::Const(4)], Dtype::F32).unwrap();
        let x = g.leaf("x", &[Dim::Const(4)], Dtype::F32).unwrap();
        assert_eq!(
            g.handle().facts(w.id()).persistence,
            fusor2_ir::dtype::Persistence::Persistent
        );
        assert_eq!(
            g.handle().facts(x.id()).persistence,
            fusor2_ir::dtype::Persistence::Step
        );
    }

    #[test]
    fn named_symbols_are_stable_and_never_collide_with_minted_ones() {
        let g = graph();
        let a = g.sym("seq");
        let b = g.sym("seq");
        assert_eq!(a, b);
        let fresh = g.handle().fresh_sym();
        assert_ne!(Dim::Sym(fresh), a);
    }

    #[test]
    fn interning_folds_equal_attribute_blobs() {
        let g = graph();
        let one = g.intern_attrs(MacroAttr::Softmax { axis: 1 });
        let same = g.intern_attrs(MacroAttr::Softmax { axis: 1 });
        let other = g.intern_attrs(MacroAttr::Softmax { axis: 2 });
        assert_eq!(one, same);
        assert_ne!(one, other);
    }

    #[test]
    fn a_uniform_is_a_leaf_not_a_one_element_tensor() {
        let g = graph();
        let lr = g.uniform("lr").unwrap();
        g.set_uniform("lr", 3e-4);
        assert_eq!(g.handle().facts(lr.id()).rank(), 0);
        let sym = g.handle().named_sym("lr");
        assert_eq!(g.handle().uniform_value(sym), Some(3e-4));
    }
}
