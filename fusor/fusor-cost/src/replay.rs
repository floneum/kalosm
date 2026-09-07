//! The replay memo, keyed on the extraction inputs. Validity is "the inputs
//! are identical".

use fixedbitset::FixedBitSet;
use fusor_ir::Result;
use fusor_ir::egraph::{EGraph, Id};
use fusor_ir::extract::{Plan, PlanHash, ReplayKey};
use fusor_ir::ir::Op;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::shape::Dim;
use parking_lot::Mutex;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Entries the cache keeps before evicting the least recently used.
pub const CAPACITY: usize = 64;

/// Bounded per-process memo from extraction inputs to a finished plan.
///
/// [`Self::get_or_extract`] always runs the closure when the key is absent; a
/// caller that suspects an input changed builds a different key.
#[derive(Default)]
pub struct ReplayCache {
    entries: Mutex<Lru>,
}

/// The architecture document's name for the same type.
pub type ReplayMemo = ReplayCache;

#[derive(Default)]
struct Lru {
    /// Most recently used last.
    order: Vec<ReplayKey>,
    plans: Vec<(ReplayKey, Entry)>,
}

/// One key's plan, and whether *that* plan has passed `verify_plan` against
/// the graph term the key names.
struct Entry {
    plan: Arc<Plan>,
    /// The plan hash last verified under this key. Carried per entry so
    /// evicting the plan evicts the record, and a replacement plan never
    /// inherits its predecessor's clearance.
    verified: Option<PlanHash>,
}

impl Lru {
    fn touch(&mut self, key: ReplayKey) {
        if let Some(i) = self.order.iter().position(|k| *k == key) {
            self.order.remove(i);
        }
        self.order.push(key);
    }

    fn get(&mut self, key: ReplayKey) -> Option<Arc<Plan>> {
        let hit = self
            .plans
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, e)| Arc::clone(&e.plan))?;
        self.touch(key);
        Some(hit)
    }

    fn insert(&mut self, key: ReplayKey, plan: Arc<Plan>) {
        match self.plans.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => {
                if slot.1.verified != Some(plan.hash) {
                    slot.1.verified = None;
                }
                slot.1.plan = plan;
            }
            None => self.plans.push((
                key,
                Entry {
                    plan,
                    verified: None,
                },
            )),
        }
        self.touch(key);
        while self.plans.len() > CAPACITY {
            let evict = self.order.remove(0);
            self.plans.retain(|(k, _)| *k != evict);
        }
    }
}

impl ReplayCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: ReplayKey) -> Option<Arc<Plan>> {
        self.entries.lock().get(key)
    }

    pub fn insert(&self, key: ReplayKey, plan: Plan) {
        self.entries.lock().insert(key, Arc::new(plan));
    }

    /// Look up `key`, extracting through `f` on a miss.
    ///
    /// The returned flag is `plan_unchanged`: `true` when the entry was
    /// already present, or when a re-extraction produced the same
    /// [`PlanHash`] — either way nothing recompiles.
    pub fn get_or_extract(
        &self,
        key: ReplayKey,
        f: impl FnOnce() -> Result<Plan>,
    ) -> Result<(Arc<Plan>, bool)> {
        if let Some(hit) = self.get(key) {
            return Ok((hit, true));
        }
        let fresh = f()?;
        let previous = self.newest_hash();
        let unchanged = previous == Some(fresh.hash);
        let plan = Arc::new(fresh);
        self.entries.lock().insert(key, Arc::clone(&plan));
        Ok((plan, unchanged))
    }

    /// Whether `hash` is the plan this key already put through `verify_plan`.
    ///
    /// `verify_plan` is a pure function of the plan and the graph term it was
    /// extracted from, and a [`ReplayKey`] is that term's identity. The plan
    /// hash is carried too, so a replaced entry never inherits the verdict of
    /// the plan it displaced.
    pub fn is_verified(&self, key: ReplayKey, hash: PlanHash) -> bool {
        self.entries
            .lock()
            .plans
            .iter()
            .any(|(k, e)| *k == key && e.verified == Some(hash))
    }

    /// Record that `hash` passed `verify_plan` under `key`. A no-op when the
    /// entry has since been replaced or evicted.
    pub fn mark_verified(&self, key: ReplayKey, hash: PlanHash) {
        let mut entries = self.entries.lock();
        if let Some((_, e)) = entries.plans.iter_mut().find(|(k, _)| *k == key)
            && e.plan.hash == hash
        {
            e.verified = Some(hash);
        }
    }

    pub fn clear(&self) {
        let mut e = self.entries.lock();
        e.plans.clear();
        e.order.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.lock().plans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn newest_hash(&self) -> Option<PlanHash> {
        let e = self.entries.lock();
        let key = *e.order.last()?;
        e.plans
            .iter()
            .find(|(k, _)| *k == key)
            .map(|(_, e)| e.plan.hash)
    }
}

/// Structural fingerprint of the term a plan was extracted from, **with
/// symbols as symbols**: two dispatches of one shape family produce the same
/// value, so the key discriminates on the binding rather than on the shape.
///
/// A leaf contributes its dtype and its shape, and a `Const` its value; it
/// does not contribute its `BufferId`, so a re-upload into a fresh buffer
/// replays. The key must be injective over everything a cached plan's `Id`s
/// refer to: every member of every class the roots reach, *with its id* —
/// ids are meaningful only in the arena they index, and an arena is
/// append-only, so equal ids holding equal nodes is the same term. Nodes the
/// roots never reach are not hashed: a graph that keeps growing elsewhere
/// (another model, a readback's own small term) leaves this plan valid.
pub fn l0_term_hash(graph: &EGraph, roots: &[Id]) -> u64 {
    let mut h = FxHasher::default();
    h.write_usize(roots.len());
    for r in roots {
        h.write_u32(r.0);
    }
    // Every id of every class reachable from the roots, in id order so the
    // hash does not depend on traversal order. A class's members are what
    // the extractor chooses among, so a plan's ids all lie in this set.
    let mut seen = FixedBitSet::with_capacity(graph.len());
    let mut stack: Vec<Id> = roots.to_vec();
    while let Some(id) = stack.pop() {
        if seen.contains(id.index()) {
            continue;
        }
        for m in graph.class_ids(graph.class_of(id)) {
            if seen.put(m.index()) {
                continue;
            }
            stack.extend(graph.node(m).children.iter().copied());
        }
    }
    // `Hash for ScalarExpr` writes a cached digest, so this stays O(nodes).
    for i in seen.ones() {
        let v = Id(i as u32);
        let node = graph.node(v);
        h.write_u32(v.0);
        node.op.tag().hash(&mut h);
        if let Op::Logical(Logical::Leaf(k)) = &node.op {
            hash_leaf(&mut h, k);
        } else {
            node.op.hash(&mut h);
        }
        for c in node.children.iter() {
            h.write_u32(c.0);
        }
    }
    h.finish()
}

/// Everything about a leaf that changes the plan, and nothing that only names
/// a buffer: the uniform's *slot* rather than its bound value, and the
/// buffer's dtype and shape rather than its `BufferId`.
fn hash_leaf<H: Hasher>(h: &mut H, kind: &LeafKind) {
    std::mem::discriminant(kind).hash(h);
    match kind {
        LeafKind::Buffer { dtype, shape, .. } | LeafKind::Param { dtype, shape, .. } => {
            dtype.hash(h);
            hash_shape(h, shape);
        }
        LeafKind::Const { value, shape } => {
            // Folded into the kernel body, so the value is part of the plan.
            value.hash(h);
            hash_shape(h, shape);
        }
        LeafKind::Uniform { sym, dtype } => {
            h.write_u32(sym.0);
            dtype.hash(h);
        }
        LeafKind::Quantized {
            fmt, layout, shape, ..
        } => {
            fmt.hash(h);
            layout.hash(h);
            hash_shape(h, shape);
        }
    }
}

fn hash_shape<H: Hasher>(h: &mut H, shape: &[Dim]) {
    h.write_usize(shape.len());
    for d in shape {
        match d {
            Dim::Const(v) => {
                h.write_u8(0);
                h.write_u64(*v);
            }
            // A symbolic extent stays symbolic: one plan serves the family and
            // `ReplayKey::binding` is what discriminates the dispatch.
            Dim::Sym(s) => {
                h.write_u8(1);
                h.write_u32(s.0);
            }
        }
    }
}

/// Hash of one dim binding — the vector a symbolic plan is dispatched at.
pub fn binding_hash(dims: &[Dim]) -> u64 {
    let mut h = FxHasher::default();
    for d in dims {
        match d {
            Dim::Const(v) => {
                h.write_u8(0);
                h.write_u64(*v);
            }
            Dim::Sym(s) => {
                h.write_u8(1);
                h.write_u32(s.0);
            }
        }
    }
    h.finish()
}
