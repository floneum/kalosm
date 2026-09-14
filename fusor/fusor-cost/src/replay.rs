//! The replay memo, keyed on the extraction inputs. Validity is "the inputs
//! are identical".

use fixedbitset::FixedBitSet;
use fusor_ir::Result;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::extract::{Plan, PlanHash, ReplayKey};
use fusor_ir::ir::Op;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::shape::Dim;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHasher};
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
    ///
    /// `graph` is what the plan has to be a selection *of*. A [`ReplayKey`]
    /// identifies the term under the roots, which is stable, but
    /// [`Extraction::sigma`] is keyed by class **representative**, and a
    /// union anywhere in the graph moves one. A hit whose keys have only
    /// moved is rekeyed; one whose classes have merged is no longer a
    /// selection of this graph and is re-extracted.
    pub fn get_or_extract(
        &self,
        key: ReplayKey,
        graph: &EGraph,
        f: impl FnOnce() -> Result<Plan>,
    ) -> Result<(Arc<Plan>, bool)> {
        if let Some(hit) = self.get(key) {
            match recanonicalize(graph, &hit) {
                Canonical::Current => return Ok((hit, true)),
                Canonical::Moved(plan) => {
                    let plan = Arc::new(*plan);
                    self.entries.lock().insert(key, Arc::clone(&plan));
                    return Ok((plan, true));
                }
                // Fall through to a fresh extraction, which replaces the entry.
                Canonical::Merged => {}
            }
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
            // the dispatch binds it through the uniform block.
            Dim::Sym(s) => {
                h.write_u8(1);
                h.write_u32(s.0);
            }
        }
    }
}

/// What [`ReplayCache::get_or_extract`] found a memoized plan to be, against
/// the graph as it stands now.
enum Canonical {
    /// Every class key is still its class's representative.
    Current,
    /// Representatives moved; here is the same plan rekeyed.
    Moved(Box<Plan>),
    /// Two selected classes have since merged, so the plan names two members
    /// of one class. Not a selection any more.
    Merged,
}

fn recanonicalize(graph: &EGraph, plan: &Plan) -> Canonical {
    match recanonicalize_sigma(&plan.extraction.sigma, |id| graph.class_of(id)) {
        None => Canonical::Current,
        Some(None) => Canonical::Merged,
        Some(Some(sigma)) => {
            let mut rekeyed = plan.clone();
            rekeyed.extraction.sigma = sigma;
            Canonical::Moved(Box::new(rekeyed))
        }
    }
}

/// `sigma` under `class_of` as it stands now.
///
/// `None` when every key is already its class's representative. `Some(None)`
/// when two keys canonicalize together onto different members — the classes
/// merged, and no rekeying makes one selection out of two. Otherwise the
/// rekeyed map.
fn recanonicalize_sigma(
    sigma: &FxHashMap<ClassId, Id>,
    class_of: impl Fn(Id) -> ClassId,
) -> Option<Option<FxHashMap<ClassId, Id>>> {
    if sigma.keys().all(|c| class_of(c.0) == *c) {
        return None;
    }
    let mut out = FxHashMap::with_capacity_and_hasher(sigma.len(), Default::default());
    for (class, member) in sigma {
        if let Some(other) = out.insert(class_of(class.0), *member)
            && other != *member
        {
            return Some(None);
        }
    }
    Some(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sigma(pairs: &[(u32, u32)]) -> FxHashMap<ClassId, Id> {
        pairs
            .iter()
            .map(|(c, m)| (ClassId(Id(*c)), Id(*m)))
            .collect()
    }

    /// The common case: nothing has moved, and the map is left alone.
    #[test]
    fn a_selection_whose_classes_have_not_moved_is_left_alone() {
        let s = sigma(&[(1, 1), (2, 2)]);
        assert!(recanonicalize_sigma(&s, ClassId).is_none());
    }

    /// A union after the plan was memoized moves a class's representative.
    /// The selection is still one member per class; only the key changed.
    #[test]
    fn a_moved_representative_is_rekeyed() {
        // Class {1, 7} is now represented by 7.
        let moved = |id: Id| ClassId(if id.index() == 1 { Id(7) } else { id });
        let s = sigma(&[(1, 1), (2, 2)]);
        let out = recanonicalize_sigma(&s, moved)
            .expect("keys moved")
            .expect("no merge");
        assert_eq!(out.len(), 2);
        assert_eq!(out[&ClassId(Id(7))], Id(1));
        assert_eq!(out[&ClassId(Id(2))], Id(2));
    }

    /// Two classes the plan selected separately have merged. The plan names
    /// two members of what is now one class, which no rekeying repairs — the
    /// caller has to extract again.
    #[test]
    fn merged_classes_cannot_be_rekeyed() {
        let merged = |_: Id| ClassId(Id(9));
        assert_eq!(recanonicalize_sigma(&sigma(&[(1, 1), (2, 2)]), merged), Some(None));
    }

    /// Two keys that merged onto the *same* member are not a conflict: one
    /// class, one selection, said twice.
    #[test]
    fn merged_classes_agreeing_on_a_member_are_rekeyed() {
        let merged = |_: Id| ClassId(Id(9));
        let out = recanonicalize_sigma(&sigma(&[(1, 5), (2, 5)]), merged)
            .expect("keys moved")
            .expect("no conflict");
        assert_eq!(out.len(), 1);
        assert_eq!(out[&ClassId(Id(9))], Id(5));
    }
}
