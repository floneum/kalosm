//! The replay memo, keyed on the extraction inputs. Validity is "the inputs
//! are identical".

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

/// Entries, most recently used last.
#[derive(Default)]
struct Lru(Vec<(ReplayKey, Entry)>);

/// The `(arena, unions)` stamp a replayed plan was last checked against: a
/// hit on the same graph with no union since needs no recanonicalization.
type Checked = Option<(u64, u64)>;

struct Entry {
    plan: Arc<Plan>,
    checked: Checked,
}

impl Lru {
    /// Move `key`'s entry to most recent and return it.
    fn touch(&mut self, key: ReplayKey) -> Option<&mut Entry> {
        let i = self.0.iter().position(|(k, _)| *k == key)?;
        let entry = self.0.remove(i);
        self.0.push(entry);
        self.0.last_mut().map(|(_, e)| e)
    }

    fn get(&mut self, key: ReplayKey) -> Option<(Arc<Plan>, Checked)> {
        self.touch(key).map(|e| (Arc::clone(&e.plan), e.checked))
    }

    fn mark_checked(&mut self, key: ReplayKey, token: (u64, u64)) {
        if let Some((_, e)) = self.0.iter_mut().find(|(k, _)| *k == key) {
            e.checked = Some(token);
        }
    }

    fn insert(&mut self, key: ReplayKey, plan: Arc<Plan>) {
        match self.touch(key) {
            Some(e) => {
                *e = Entry {
                    plan,
                    checked: None,
                }
            }
            None => self.0.push((
                key,
                Entry {
                    plan,
                    checked: None,
                },
            )),
        }
        if self.0.len() > CAPACITY {
            self.0.remove(0);
        }
    }
}

impl ReplayCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: ReplayKey) -> Option<Arc<Plan>> {
        self.entries.lock().get(key).map(|(plan, _)| plan)
    }

    pub fn insert(&self, key: ReplayKey, plan: Plan) {
        self.entries.lock().insert(key, Arc::new(plan));
    }

    /// Look up `key`, extracting through `f` on a miss; the flag is `true`
    /// when nothing recompiles. A hit whose class keys moved under a union is
    /// rekeyed; one whose classes merged is re-extracted.
    pub fn get_or_extract(
        &self,
        key: ReplayKey,
        graph: &EGraph,
        f: impl FnOnce() -> Result<Plan>,
    ) -> Result<(Arc<Plan>, bool)> {
        let hit = self.entries.lock().get(key);
        if let Some((hit, checked)) = hit {
            let token = (graph.arena_id(), graph.union_count());
            if checked == Some(token) {
                return Ok((hit, true));
            }
            match recanonicalize(graph, &hit) {
                Canonical::Current => {
                    self.entries.lock().mark_checked(key, token);
                    return Ok((hit, true));
                }
                Canonical::Moved(plan) => {
                    let plan = Arc::new(*plan);
                    let mut entries = self.entries.lock();
                    entries.insert(key, Arc::clone(&plan));
                    entries.mark_checked(key, token);
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

    pub fn clear(&self) {
        self.entries.lock().0.clear();
    }

    pub fn len(&self) -> usize {
        self.entries.lock().0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn newest_hash(&self) -> Option<PlanHash> {
        self.entries.lock().0.last().map(|(_, e)| e.plan.hash)
    }
}

/// Structural fingerprint of the term under `roots`, symbols as symbols and
/// leaves without their buffer names: every id of every reachable class,
/// with its node, so a plan's ids all lie in what it hashes.
pub fn l0_term_hash(graph: &EGraph, roots: &[Id]) -> u64 {
    let mut h = FxHasher::default();
    h.write_usize(roots.len());
    for r in roots {
        h.write_u32(r.0);
    }
    // In id order, independent of traversal.
    let (_, seen) = crate::realize::reachable_unsorted(graph, roots);
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
/// a buffer.
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
            Dim::Sym(s) => {
                h.write_u8(1);
                h.write_u32(s.0);
            }
        }
    }
}

/// A memoized plan against the graph as it stands now.
enum Canonical {
    /// Every class key is still its class's representative.
    Current,
    /// Representatives moved; here is the same plan rekeyed.
    Moved(Box<Plan>),
    /// Two selected classes have merged: not a selection any more.
    Merged,
}

fn recanonicalize(graph: &EGraph, plan: &Plan) -> Canonical {
    let sigma = &plan.extraction.sigma;
    let len = graph.len();
    // A node this graph never minted means extracting again here.
    let mut moved = false;
    for (class, member) in sigma {
        if class.0.index() >= len || member.index() >= len {
            return Canonical::Merged;
        }
        moved |= graph.class_of(class.0) != *class;
    }
    if !moved {
        return Canonical::Current;
    }
    match recanonicalize_sigma(sigma, |id| graph.class_of(id)) {
        None => Canonical::Current,
        Some(None) => Canonical::Merged,
        Some(Some(sigma)) => {
            let mut rekeyed = plan.clone();
            rekeyed.extraction.sigma = sigma;
            Canonical::Moved(Box::new(rekeyed))
        }
    }
}

/// `sigma` under `class_of` now: `None` when unmoved, `Some(None)` when two
/// keys merged onto different members, else the rekeyed map.
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
        assert_eq!(
            recanonicalize_sigma(&sigma(&[(1, 1), (2, 2)]), merged),
            Some(None)
        );
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
