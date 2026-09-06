//! `Graph` and `Gradients`.
//!
//! The backward transform's output is ingested together with the forward as
//! one graph with one root set, which is what makes gradient checkpointing
//! the extractor's materialization bit.

use std::sync::{Arc, Weak};

use fusor_autograd::custom::{CustomRegistry, with_backwards as register_custom};
use fusor_autograd::tape::{GraphTape, splat_of};
use fusor_ir::autograd::{AdjointFn, Parent};
use fusor_ir::dtype::{Dtype, QFmt, QLayout};
use fusor_ir::egraph::{EGraph, Id};
use fusor_ir::ir::logical::{BufferId, LeafKind, Logical};
use fusor_ir::ir::{AttrId, Op};
use fusor_ir::shape::{Dim, SymId};
use fusor_ir::target::Buf;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::composite::MacroAttr;
use crate::session::Session;
use crate::tensor::Tensor;
use crate::{Error, Result};

/// Shared handle to the graph a [`crate::Tensor`] belongs to.
#[derive(Clone)]
pub struct GraphRef {
    state: Arc<GraphInner>,
}

#[derive(Clone)]
pub(crate) struct WeakGraphRef {
    state: Weak<GraphInner>,
}

impl WeakGraphRef {
    pub(crate) fn strong_count(&self) -> usize {
        self.state.strong_count()
    }

    pub(crate) fn upgrade(&self) -> Option<GraphRef> {
        self.state.upgrade().map(|state| GraphRef { state })
    }
}

/// Host bytes waiting to be uploaded for one external leaf, plus the device
/// buffer once one exists. A `Persistent` leaf keeps its buffer across
/// resolves; a `Step` leaf is re-uploaded whenever its bytes change.
#[derive(Default)]
pub(crate) struct LeafStore {
    pub(crate) bytes: FxHashMap<Id, Arc<Vec<u8>>>,
    /// The device buffer bound to a value, with the layout it was written
    /// under when that is not the value's own dense shape. A selected
    /// `Coop`/`Sgemm` geometry pads the output to its tile multiple, so a
    /// readback that assumed contiguity would hand back the first row plus
    /// padding.
    pub(crate) device: FxHashMap<Id, (Buf, Option<Arc<fusor_ir::shape::Layout>>)>,
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
/// A graph's kernels are keyed by its arena on the target; once the graph is
/// gone nothing can reach them, so they go with it.
impl Drop for GraphInner {
    fn drop(&mut self) {
        let arena = self.egraph.get_mut().arena_id();
        self.session.device().release_arena(arena);
    }
}

pub(crate) struct GraphInner {
    pub(crate) egraph: Mutex<EGraph>,
    pub(crate) session: Session,
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
    /// Source-addressed `U32` respellings of byte leaves. See
    /// [`GraphInner::words_leaf_of`].
    word_leaves: Mutex<FxHashMap<Id, Id>>,
    /// Source-addressed word-aligned respellings of quantized leaves. See
    /// [`GraphInner::repacked_leaf_of`].
    repack_leaves: Mutex<FxHashMap<Id, Option<Id>>>,
    /// First-union roots, so a rebuilt composite keeps one name. See
    /// [`GraphInner::union_stable`].
    union_memo: Mutex<FxHashMap<(Id, Id), Id>>,
    /// Serializes whole resolve-and-read sequences against this graph.
    ///
    /// [`crate::session::Session::resolve`] cannot hold [`Self::egraph`] for
    /// its own duration (the mutex is not reentrant), and releasing it
    /// between resolve and readback lets a thread download an output buffer
    /// whose dispatch has not run yet — which returns zeros, not an error.
    /// `read_back` holds this across `resolve` and `read_bytes`, so a
    /// concurrent resolve cannot land between them.
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

impl GraphRef {
    pub(crate) fn as_ptr(graph: &Self) -> *const GraphInner {
        Arc::as_ptr(&graph.state)
    }

    pub(crate) fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.state, &b.state)
    }

    pub(crate) fn downgrade(graph: &Self) -> WeakGraphRef {
        WeakGraphRef {
            state: Arc::downgrade(&graph.state),
        }
    }

    pub(crate) fn state(&self) -> &GraphInner {
        &self.state
    }

    /// Run `f` with exclusive access to the e-graph.
    pub(crate) fn with_egraph<T>(&self, f: impl FnOnce(&mut EGraph) -> Result<T>) -> Result<T> {
        let mut g = self.state.egraph.lock();
        f(&mut g)
    }

    /// Run `f` with a construction tape over the e-graph. Every macro-op
    /// `defn` is built through this, so forward and backward share one node
    /// arena.
    pub(crate) fn build<T>(&self, f: impl FnOnce(&mut GraphTape<'_>) -> Result<T>) -> Result<T> {
        let mut g = self.state.egraph.lock();
        let mut tape = GraphTape::new(&mut g);
        f(&mut tape)
    }

    pub(crate) fn add_logical(&self, op: Logical) -> Result<Id> {
        self.state.egraph.lock().add(Op::Logical(op))
    }

    /// Union with a stable return: the first call for a pair unions and
    /// memoizes the root it produced; every later call for the
    /// same pair returns that same id, even after saturation has grown the
    /// class and moved its current root.
    ///
    /// This lets a decode loop rebuild the same composite each step and land
    /// on the same node ids: `union` returns the current class root, which
    /// moves every time a rule unions another member in, so a rebuilt
    /// consumer referencing it would miss the hash-cons memo and re-mint the
    /// whole downstream graph. The memoized id is an ordinary member of the
    /// class; selection, facts and readback are all per class.
    pub(crate) fn union_stable(&self, a: Id, b: Id) -> Result<Id> {
        let key = if a.0 <= b.0 { (a, b) } else { (b, a) };
        if let Some(hit) = self.state.union_memo.lock().get(&key).copied() {
            return Ok(hit);
        }
        let root = self.state.egraph.lock().union(a, b)?;
        self.state.union_memo.lock().insert(key, root);
        Ok(root)
    }

    /// The `GraphRef`-level spelling of [`Graph::with_backwards`].
    pub(crate) fn register_backward(
        &self,
        value: Id,
        parents: &[fusor_ir::autograd::Parent],
        rule: AdjointFn,
    ) -> Result<()> {
        let mut reg = self.state.custom.lock();
        register_custom(&mut reg, value, parents, rule)?;
        Ok(())
    }

    pub(crate) fn facts(&self, id: Id) -> fusor_ir::facts::ValueFacts {
        self.state.egraph.lock().facts(id).clone()
    }

    pub(crate) fn session(&self) -> &Session {
        &self.state.session
    }

    /// Wrap a node id as a user-facing tensor.
    pub(crate) fn tensor(&self, id: Id) -> Tensor {
        Tensor {
            id,
            graph: self.clone(),
        }
    }

    /// Intern a macro attribute blob. Equal attributes share an id, so two
    /// identically-configured macro ops hash-cons together.
    pub(crate) fn intern_attrs(&self, attrs: MacroAttr) -> AttrId {
        let mut table = self.state.attrs.lock();
        if let Some(i) = table.iter().position(|a| *a == attrs) {
            return AttrId(i as u32);
        }
        table.push(attrs);
        AttrId((table.len() - 1) as u32)
    }

    /// The one `BufferId` allocator. Every leaf name in a graph comes from
    /// here, so no two leaves can share a name by accident.
    pub(crate) fn fresh_buffer_id(&self) -> BufferId {
        let mut next = self.state.next_buffer.lock();
        let id = BufferId(*next);
        *next += 1;
        id
    }

    /// An immutable rank-N leaf holding `bytes`, named by its content.
    ///
    /// A leaf's hash-cons key is its `LeafKind`, and host bytes live in a side
    /// table that is not part of that key, so two constants of the same dtype
    /// and shape under one name would share a node and the second
    /// `set_leaf_bytes` would silently overwrite the first. Naming the leaf by
    /// its content makes the key exact in both directions.
    pub(crate) fn constant_leaf(&self, dtype: Dtype, shape: &[Dim], bytes: Vec<u8>) -> Result<Id> {
        let key = ConstKey {
            dtype,
            shape: shape.to_vec(),
            bytes,
        };
        if let Some(id) = self.state.constants.lock().get(&key).copied() {
            return Ok(id);
        }
        let id = self.add_logical(Logical::Leaf(LeafKind::Buffer {
            name: self.fresh_buffer_id(),
            dtype,
            shape: shape.iter().copied().collect(),
        }))?;
        self.set_leaf_bytes(id, key.bytes.clone());
        self.state.constants.lock().insert(key, id);
        Ok(id)
    }

    /// The rank-1 `U32` leaf reading `src`'s host bytes as words, minted at
    /// most once per source leaf and sharing the byte allocation with it.
    ///
    /// Keyed by the source id instead of the content: a leaf's bytes are
    /// immutable once set, so the id names the content exactly.
    ///
    /// `None` when `src` has no host bytes or they are not whole words.
    pub(crate) fn words_leaf_of(&self, src: Id) -> Result<Option<Id>> {
        if let Some(id) = self.state.word_leaves.lock().get(&src).copied() {
            return Ok(Some(id));
        }
        let Some(bytes) = self.leaf_bytes_shared(src) else {
            return Ok(None);
        };
        if !bytes.len().is_multiple_of(4) {
            return Ok(None);
        }
        let id = self.add_logical(Logical::Leaf(LeafKind::Buffer {
            name: self.fresh_buffer_id(),
            dtype: Dtype::U32,
            shape: std::iter::once(Dim::Const(bytes.len() as u64 / 4)).collect(),
        }))?;
        self.set_leaf_bytes_shared(id, bytes);
        self.state.word_leaves.lock().insert(src, id);
        Ok(Some(id))
    }

    /// The word-aligned twin of a quantized leaf: the same blocks repacked
    /// `Native -> F32Scales`, minted at most once per source leaf. The leaf
    /// is a separate value with its own buffer, never unioned with its source
    /// (every id in one value class must denote one buffer; see
    /// `lower::PlanLowering::new`'s `slot_of`). The consumer that wants the
    /// choice priced unions the consuming ops instead (`Tensor::contract_2d`),
    /// so extraction selects a layout per contraction and only the selected
    /// leaf ever uploads.
    ///
    /// `None` when the leaf is not quantized, is not `Native`, has no host
    /// bytes, or its native block stride is already a whole number of words.
    pub(crate) fn repacked_leaf_of(&self, src: Id) -> Result<Option<Id>> {
        if let Some(cached) = self.state.repack_leaves.lock().get(&src).copied() {
            return Ok(cached);
        }
        let mint = || -> Result<Option<Id>> {
            let (fmt, shape) = {
                let g = self.state.egraph.lock();
                match &g.node(src).op {
                    Op::Logical(Logical::Leaf(LeafKind::Quantized {
                        fmt,
                        layout: QLayout::Native,
                        shape,
                        ..
                    })) if fmt.block_bytes(QLayout::Native) % 4 != 0 => (*fmt, shape.clone()),
                    _ => return Ok(None),
                }
            };
            let Some(bytes) = self.leaf_bytes_shared(src) else {
                return Ok(None);
            };
            let mut repacked = Vec::new();
            fusor_gguf::repack::repack(
                fmt,
                QLayout::Native,
                QLayout::F32Scales,
                &bytes,
                &mut repacked,
            )?;
            let id = self.add_logical(Logical::Leaf(LeafKind::Quantized {
                name: self.fresh_buffer_id(),
                fmt,
                layout: QLayout::F32Scales,
                shape,
            }))?;
            self.set_leaf_bytes(id, repacked);
            Ok(Some(id))
        };
        let out = mint()?;
        self.state.repack_leaves.lock().insert(src, out);
        Ok(out)
    }

    /// A fresh symbolic quantity. Allocated by the e-graph so a frontend
    /// symbol can never collide with one a rule mints (`fold_split`'s block
    /// count).
    pub(crate) fn fresh_sym(&self) -> SymId {
        self.state.egraph.lock().fresh_sym()
    }

    /// A named symbol, created on first use. Two calls with the same name
    /// return the same `SymId`, so `graph.sym("seq")` is stable across a
    /// decode loop.
    pub(crate) fn named_sym(&self, name: &str) -> SymId {
        if let Some(s) = self.state.symbols.lock().named.get(name).copied() {
            return s;
        }
        let s = self.fresh_sym();
        self.state.symbols.lock().named.insert(name.to_string(), s);
        s
    }

    /// Bind a symbolic extent for the next resolve.
    pub(crate) fn bind_dim(&self, sym: SymId, value: u64) {
        self.state.symbols.lock().dims.insert(sym, value);
    }

    pub(crate) fn dim_binding(&self, sym: SymId) -> Option<u64> {
        self.state.symbols.lock().dims.get(&sym).copied()
    }

    /// Write a runtime scalar. A learning rate set here is a word in
    /// binding 0, never a baked literal, so changing it recompiles nothing.
    pub(crate) fn set_uniform(&self, sym: SymId, value: f32) {
        self.state.symbols.lock().scalars.insert(sym, value);
    }

    pub(crate) fn uniform_value(&self, sym: SymId) -> Option<f32> {
        self.state.symbols.lock().scalars.get(&sym).copied()
    }

    /// Every `(sym, value)` runtime scalar declared so far.
    pub(crate) fn uniform_scalars(&self) -> Vec<(SymId, f32)> {
        let mut out: Vec<(SymId, f32)> = self
            .state
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
    pub(crate) fn dim_bindings(&self) -> Vec<(SymId, u64)> {
        let mut out: Vec<(SymId, u64)> = self
            .state
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
    pub(crate) fn set_leaf_bytes(&self, id: Id, bytes: Vec<u8>) {
        self.set_leaf_bytes_shared(id, Arc::new(bytes));
    }

    /// [`Self::set_leaf_bytes`] without a copy: two leaves may share one
    /// byte stream — a quantized weight and its `U32`-word respelling are
    /// the same allocation.
    pub(crate) fn set_leaf_bytes_shared(&self, id: Id, bytes: Arc<Vec<u8>>) {
        let mut store = self.state.leaves.lock();
        store.bytes.insert(id, bytes);
        // A changed upload invalidates the device copy, never the plan.
        store.device.remove(&id);
    }

    pub(crate) fn leaf_bytes(&self, id: Id) -> Option<Vec<u8>> {
        self.state
            .leaves
            .lock()
            .bytes
            .get(&id)
            .map(|b| b.as_ref().clone())
    }

    /// The shared handle to a leaf's host bytes, no copy.
    pub(crate) fn leaf_bytes_shared(&self, id: Id) -> Option<Arc<Vec<u8>>> {
        self.state.leaves.lock().bytes.get(&id).cloned()
    }

    /// Run `f` over an external leaf's host bytes without copying them.
    ///
    /// `f` runs with the `leaves` mutex held, so it must not re-enter this
    /// graph. The only caller uploads through the target's pool, which locks
    /// nothing of the graph's.
    pub(crate) fn with_leaf_bytes<T>(&self, id: Id, f: impl FnOnce(&[u8]) -> T) -> Option<T> {
        let store = self.state.leaves.lock();
        store.bytes.get(&id).map(|b| f(b.as_slice()))
    }

    /// The device buffer backing `id`, once a resolve has produced one.
    pub(crate) fn device_buf(&self, id: Id) -> Option<Buf> {
        self.state
            .leaves
            .lock()
            .device
            .get(&id)
            .map(|(b, _)| b.clone())
    }

    /// For each of `ids`, whether it is an external leaf the caller must
    /// supply and, if so, the device buffer it already carries. Two lock
    /// acquisitions for the whole set.
    pub(crate) fn external_leaf_buffers(&self, ids: &[Id]) -> Vec<(Id, Option<Buf>)> {
        let external: Vec<Id> = {
            let g = self.state.egraph.lock();
            ids.iter()
                .copied()
                .filter(|id| {
                    matches!(
                        &g.node(*id).op,
                        Op::Logical(Logical::Leaf(
                            LeafKind::Buffer { .. }
                                | LeafKind::Param { .. }
                                | LeafKind::Quantized { .. }
                        ))
                    )
                })
                .collect()
        };
        let store = self.state.leaves.lock();
        external
            .into_iter()
            .map(|id| (id, store.device.get(&id).map(|(b, _)| b.clone())))
            .collect()
    }

    /// Drop the device buffers bound to `id`'s whole class.
    ///
    /// The decode-loop convention: a step's outputs are cleared before the
    /// next resolve so the same (structurally unchanged) graph re-dispatches
    /// instead of short-circuiting on last step's buffers. Clears every class
    /// member because `Session::bind_class` bound every member.
    pub(crate) fn clear_class_device_buf(&self, id: Id) {
        let members = {
            let mut g = self.state.egraph.lock();
            let class = g.class_of(id);
            g.class_ids_cached(class)
        };
        let mut store = self.state.leaves.lock();
        for m in members.iter() {
            store.device.remove(m);
        }
    }

    pub(crate) fn set_device_buf(&self, id: Id, buf: Buf) {
        let mut store = self.state.leaves.lock();
        store.device.insert(id, (buf, None));
    }

    /// Register one buffer under every id of an e-class, in one lock
    /// acquisition.
    pub(crate) fn set_device_buf_class(
        &self,
        ids: &[Id],
        buf: &Buf,
        layout: Option<&Arc<fusor_ir::shape::Layout>>,
    ) {
        let mut store = self.state.leaves.lock();
        for id in ids {
            store.device.insert(*id, (buf.clone(), layout.cloned()));
        }
    }

    /// Register each `(value, buffer, layout)` under every id of the value's
    /// e-class, for the whole batch under one lock apiece.
    pub(crate) fn bind_classes(&self, items: &[(Id, Buf, Option<Arc<fusor_ir::shape::Layout>>)]) {
        let classes: Vec<Arc<[Id]>> = {
            let mut g = self.state.egraph.lock();
            items
                .iter()
                .map(|(id, _, _)| {
                    let class = g.class_of(*id);
                    g.class_ids_cached(class)
                })
                .collect()
        };
        let mut store = self.state.leaves.lock();
        for ((_, buf, layout), members) in items.iter().zip(&classes) {
            for m in members.iter() {
                store.device.insert(*m, (buf.clone(), layout.clone()));
            }
        }
    }

    pub(crate) fn device_layout(&self, id: Id) -> Option<fusor_ir::shape::Layout> {
        self.state
            .leaves
            .lock()
            .device
            .get(&id)
            .and_then(|(_, l)| l.as_ref())
            .map(|l| (**l).clone())
    }

    /// Resolve `id` and copy its bytes back to the host. One of exactly
    /// three host syncs.
    ///
    /// The guard spans both halves: a concurrent resolve landing between them
    /// could bind a fresh output buffer for a class this read is about to
    /// download, and downloading a buffer whose dispatch has not run yet
    /// returns zeros rather than an error. See [`GraphRef::state`].
    pub(crate) fn read_back(&self, id: Id) -> Result<Vec<u8>> {
        let tensor = self.tensor(id);
        let resolving = self.state.resolve_lock.lock();
        self.state
            .session
            .resolve_locked(&resolving, std::slice::from_ref(&tensor))?;
        self.state.session.read_bytes_locked(&resolving, self, id)
    }

    /// [`Self::read_back`], awaited. The graph lock spans the resolve and
    /// the readback plan; the download runs after it, holding its own
    /// handle on the buffer (see `Session::read_plan_locked`).
    pub(crate) async fn read_back_async(&self, id: Id) -> Result<Vec<u8>> {
        let plan = {
            let tensor = self.tensor(id);
            let resolving = self.state.resolve_lock.lock();
            self.state
                .session
                .resolve_locked(&resolving, std::slice::from_ref(&tensor))?;
            self.state.session.read_plan_locked(&resolving, self, id)?
        };
        self.state.session.read_bytes(plan).await
    }
}

/// A program under construction.
#[derive(Clone)]
pub struct Graph {
    inner: GraphRef,
}

impl Graph {
    /// Create an empty graph resolved by `session`.
    pub fn new(session: &Session) -> Self {
        Self {
            inner: GraphRef {
                state: Arc::new(GraphInner {
                    egraph: Mutex::new(EGraph::new(session.semantics())),
                    session: session.clone(),
                    attrs: Mutex::new(Vec::new()),
                    leaves: Mutex::new(LeafStore::default()),
                    symbols: Mutex::new(SymbolStore::default()),
                    custom: Mutex::new(CustomRegistry::new()),
                    next_buffer: Mutex::new(0),
                    constants: Mutex::new(FxHashMap::default()),
                    word_leaves: Mutex::new(FxHashMap::default()),
                    repack_leaves: Mutex::new(FxHashMap::default()),
                    union_memo: Mutex::new(FxHashMap::default()),
                    resolve_lock: Mutex::new(()),
                }),
            },
        }
    }

    /// Borrow the shared graph handle used by runtime-rank operations.
    pub fn handle(&self) -> &GraphRef {
        &self.inner
    }

    /// The `Graph` a handle names.
    pub(crate) fn from_handle(inner: GraphRef) -> Self {
        Self { inner }
    }

    /// The session that resolves this graph.
    pub fn session(&self) -> &Session {
        self.inner.session()
    }

    /// A trainable parameter. `Persistence::Persistent`, so a quantized repack
    /// amortizes against its lifetime and the extractor knows it may not
    /// recompute it.
    pub fn param(&self, _name: &str, shape: &[Dim], dtype: Dtype) -> Result<Tensor> {
        let id = self.inner.add_logical(Logical::Leaf(LeafKind::Param {
            name: self.inner.fresh_buffer_id(),
            dtype,
            shape: shape.iter().copied().collect(),
        }))?;
        Ok(self.inner.tensor(id))
    }

    /// A step-local input buffer.
    pub fn leaf(&self, name: &str, shape: &[Dim], dtype: Dtype) -> Result<Tensor> {
        let _ = name;
        let id = self.inner.add_logical(Logical::Leaf(LeafKind::Buffer {
            name: self.inner.fresh_buffer_id(),
            dtype,
            shape: shape.iter().copied().collect(),
        }))?;
        Ok(self.inner.tensor(id))
    }

    /// A step-local buffer with host contents.
    pub(crate) fn constant_from_raw(
        &self,
        dtype: Dtype,
        shape: &[Dim],
        bytes: &[u8],
    ) -> Result<Tensor> {
        let t = self.leaf("", shape, dtype)?;
        self.inner.set_leaf_bytes(t.id, bytes.to_vec());
        Ok(t)
    }

    /// Upload dense host data.
    pub fn tensor(&self, dtype: Dtype, shape: &[Dim], bytes: &[u8]) -> Result<Tensor> {
        self.constant_from_raw(dtype, shape, bytes)
    }

    /// A block-quantized weight leaf.
    pub fn quantized(
        &self,
        fmt: QFmt,
        layout: QLayout,
        shape: [Dim; 2],
        bytes: &[u8],
    ) -> Result<Tensor> {
        let id = self.inner.add_logical(Logical::Leaf(LeafKind::Quantized {
            name: self.inner.fresh_buffer_id(),
            fmt,
            layout,
            shape: shape.into_iter().collect(),
        }))?;
        self.inner.set_leaf_bytes(id, bytes.to_vec());
        Ok(self.inner.tensor(id))
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
    /// A partial backward driven by [`crate::autograd`] hands over a whole
    /// frontier and expects most of it to be unreachable, so it filters here
    /// first; [`Graph::backward_with`] still errors on an unreachable `wrt`
    /// the caller named.
    ///
    /// Structural, over `children`; equal values are compared by class rather
    /// than by id.
    pub(crate) fn reachable_from(&self, value: &Tensor, candidates: &[Tensor]) -> Vec<Tensor> {
        let g = self.inner.state().egraph.lock();
        let mut want: FxHashMap<fusor_ir::egraph::ClassId, Vec<usize>> = FxHashMap::default();
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

    /// Refuse a tensor from another graph. An `Id` is an index into one
    /// e-graph's arena, so a foreign one either names an unrelated node or is
    /// out of range.
    fn owns(&self, t: &Tensor, role: &str) -> Result<()> {
        if GraphRef::ptr_eq(&t.graph, &self.inner) {
            return Ok(());
        }
        Err(Error::Plan(format!(
            "the {role} tensor belongs to a different graph; a backward pass cannot cross graphs"
        )))
    }

    fn ones_like(&self, t: &Tensor) -> Result<Id> {
        let facts = self.inner.facts(t.id);
        self.inner.add_logical(Logical::Leaf(LeafKind::Const {
            value: splat_of(facts.dtype, 1.0)?,
            shape: facts.shape.clone(),
        }))
    }

    fn backward_ids(&self, loss: &Tensor, seed: Id, wrt: &[Id]) -> Result<Gradients> {
        if !GraphRef::ptr_eq(&loss.graph, &self.inner) {
            return Err(Error::Device(
                "backward across two graphs is not a thing".into(),
            ));
        }
        let caps = self.inner.session().caps();
        let custom = self.inner.state().custom.lock().clone();
        let mut g = self.inner.state().egraph.lock();
        let grads = fusor_autograd::backward::backward_into_with(
            &mut g, &caps, loss.id, seed, wrt, &custom,
        )?;
        // Forward and backward are one graph with one root set, which makes
        // "save this activation" versus "recompute it" the extractor's
        // materialization bit.
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
    /// The gradient of `of`, when the backward reached it.
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

    /// The number of primal/gradient pairs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the backward produced no gradients.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A parent declaration for a custom backward.
pub(crate) fn parent(t: &Tensor, requires_grad: bool) -> Parent {
    Parent {
        value: t.id,
        requires_grad,
    }
}
