//! The replay memo, keyed on the *extraction inputs* rather than a structural
//! fingerprint. Validity is "the inputs are identical", not "a fingerprint
//! matches", so a training loop can no longer freeze step 1's decisions
//! forever.
//!
//! This is affordable precisely because the trainer reads nothing back and the
//! host runs several steps ahead: a ~1.4 ms re-extraction never lands on the
//! critical path.
//!
//! Owned by W7.

use fusor2_ir::Result;
use fusor2_ir::egraph::{EGraph, Id};
use fusor2_ir::extract::{Plan, PlanHash, ReplayKey};
use fusor2_ir::ir::Op;
use fusor2_ir::ir::level0::{L0, LeafKind};
use fusor2_ir::shape::Dim;
use parking_lot::Mutex;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Entries the cache keeps before evicting the least recently used.
pub const CAPACITY: usize = 64;

/// Bounded per-process memo from extraction inputs to a finished plan.
///
/// A hit does **not** mean "skip extraction": [`Self::get_or_extract`] always
/// runs the closure when the key is absent, and a caller that suspects an
/// input changed simply builds a different key. Contrast the reference's
/// `flush_replay`, whose validity condition is a structural fingerprint and
/// which therefore freezes every decision a value (rather than a shape) should
/// have driven.
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
    plans: Vec<(ReplayKey, Arc<Plan>)>,
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
            .map(|(_, p)| Arc::clone(p))?;
        self.touch(key);
        Some(hit)
    }

    fn insert(&mut self, key: ReplayKey, plan: Arc<Plan>) {
        match self.plans.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = plan,
            None => self.plans.push((key, plan)),
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

    /// Drop every plan extracted against one device fingerprint. Called when
    /// a driver update or a recalibration moves the rates the plan was
    /// chosen under.
    pub fn invalidate_device(&self, device: u64) {
        let mut e = self.entries.lock();
        e.plans.retain(|(k, _)| k.device != device);
        let live: Vec<ReplayKey> = e.plans.iter().map(|(k, _)| *k).collect();
        e.order.retain(|k| live.contains(k));
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
        e.plans.iter().find(|(k, _)| *k == key).map(|(_, p)| p.hash)
    }
}

/// Structural fingerprint of the graph a plan was extracted from, **with
/// symbols as symbols**: two dispatches of one shape family produce the same
/// value, so the key discriminates on the binding rather than on the shape.
///
/// A leaf contributes its dtype and its shape, and a `Const` its value; what
/// it does **not** contribute is its `BufferId`, so a re-upload into a fresh
/// buffer replays.
///
/// Two regressions this fixes, both of them the same mistake — a key that is
/// not injective over the thing a cached plan's `Id`s refer to, which is
/// silent plan corruption that only `verify_plan` running again after the
/// lookup turned into an error:
///
/// - the leaf arm hashed only `discriminant(kind)`, so two graphs differing
///   *only* in a leaf's extents were one key. `cat_rank1` and `cat_rank2`
///   collided and `cat_rank2` replayed `BufferPlan`s of rank 1 for its
///   rank-2 values;
/// - the walk covered only the root-reachable L0 spine and ignored the root
///   set, so two graphs on one `Session` — and the conformance harness builds
///   every case's graph on one shared `Session` — could share a key while
///   their ids named different nodes.
pub fn l0_term_hash(graph: &EGraph, roots: &[Id]) -> u64 {
    let mut h = FxHasher::default();
    // The roots are an extraction *input*: the same term rooted at one value
    // and at two is two different plans.
    h.write_usize(roots.len());
    for r in roots {
        h.write_u32(r.0);
    }
    // Every id, not only the root-reachable L0 spine. A cached plan's `Id`s
    // are meaningful **only** in the graph it was extracted from, so the key
    // has to determine that graph, and an id is determined by everything
    // allocated before it — including nodes this term never reaches. Walking
    // the spine alone is what let `attention_with_lse` and
    // `welford_agrees_with_the_two_pass_variance` replay a plan from an
    // earlier case on the same `Session`: both pass in isolation and fail in
    // a full run, with `verify_plan` reporting an operand class the stale
    // `sigma` has no entry for.
    //
    // `Hash for ScalarExpr` writes a cached digest, so this stays O(nodes).
    for i in 0..graph.len() {
        let v = Id(i as u32);
        let node = graph.node(v);
        node.op.tag().hash(&mut h);
        if let Op::L0(L0::Leaf(k)) = &node.op {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::LocalSearch;
    use crate::realize::testkit::{TestCost, TestPlanner, chain_graph, test_caps};
    use fusor2_ir::cost::CostModel;
    use fusor2_ir::extract::{ExtractBudget, Extractor};

    fn plan_for() -> Plan {
        let (g, roots) = chain_graph(3);
        let cost = TestCost::default();
        LocalSearch::new(Arc::new(TestPlanner), test_caps())
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap()
    }

    #[test]
    fn replay_reextracts_on_binding_change() {
        let cache = ReplayCache::new();
        let (g, roots) = chain_graph(3);
        let cost = TestCost::default();
        let search = LocalSearch::new(Arc::new(TestPlanner), test_caps());
        let term = l0_term_hash(&g, &roots);

        let short = ReplayKey {
            l0_term: term,
            device: cost.facts().fingerprint(),
            binding: binding_hash(&[Dim::Const(128)]),
        };
        let long = ReplayKey {
            binding: binding_hash(&[Dim::Const(4096)]),
            ..short
        };
        assert_ne!(short, long, "a different binding is a different key");

        let (a, hit_a) = cache
            .get_or_extract(short, || {
                search.extract(&g, &roots, &cost, ExtractBudget::default())
            })
            .unwrap();
        assert!(!hit_a, "first sighting is a miss");

        // Same term, different binding: a miss, and the extraction runs
        // again rather than replaying step one's decisions.
        let (b, unchanged) = cache
            .get_or_extract(long, || {
                search.extract(&g, &roots, &cost, ExtractBudget::default())
            })
            .unwrap();
        assert!(
            unchanged,
            "the term is symbolic, so re-extraction produced the same plan"
        );
        assert_eq!(a.hash, b.hash);
        assert_eq!(cache.len(), 2);

        // Re-asking for the first key is a hit.
        let (_, hit) = cache.get_or_extract(short, || unreachable!()).unwrap();
        assert!(hit);
    }

    /// The key has to be injective over the term. Two graphs whose only
    /// difference is a leaf's rank used to share one key, so the second
    /// replayed the first's plan — with `BufferPlan`s of the wrong rank and
    /// `Id`s that named different nodes.
    #[test]
    fn a_different_leaf_shape_is_a_different_key() {
        use crate::realize::testkit::{buffer, kmap, new_graph};
        let build = |shape: &[Dim]| {
            let mut g = new_graph();
            let leaf = buffer(&mut g, 0, shape);
            let m = kmap(&mut g, leaf, shape, 1);
            g.add_root(m);
            let roots = g.roots().to_vec();
            l0_term_hash(&g, &roots)
        };
        let rank1 = build(&[Dim::Const(12)]);
        let rank2 = build(&[Dim::Const(3), Dim::Const(4)]);
        let wider = build(&[Dim::Const(3), Dim::Const(8)]);
        assert_ne!(rank1, rank2, "rank must reach the key");
        assert_ne!(rank2, wider, "extents must reach the key");
        assert_eq!(rank1, build(&[Dim::Const(12)]), "and it is still a function");

        // A symbolic extent stays symbolic, so one plan still serves the
        // family and `ReplayKey::binding` is what separates the dispatches.
        let sym = build(&[Dim::Sym(fusor2_ir::shape::SymId(3))]);
        assert_eq!(sym, build(&[Dim::Sym(fusor2_ir::shape::SymId(3))]));
        assert_ne!(sym, rank1);
    }

    /// The key must determine the graph, not just the rooted term: a plan's
    /// `Id`s mean nothing anywhere else. Two graphs that share a rooted term
    /// but differ in what else is allocated, or in what is rooted, are two
    /// keys.
    #[test]
    fn the_key_determines_the_graph_not_only_the_rooted_term() {
        use crate::realize::testkit::{buffer, kmap, new_graph};
        let shape = [Dim::Const(8)];

        let mut plain = new_graph();
        let leaf = buffer(&mut plain, 0, &shape);
        let m = kmap(&mut plain, leaf, &shape, 1);
        plain.add_root(m);
        let plain_roots = plain.roots().to_vec();

        // Same rooted term, but an extra value was built first, so every id
        // in the term sits one slot further along.
        let mut shifted = new_graph();
        let _other = buffer(&mut shifted, 9, &[Dim::Const(4)]);
        let leaf2 = buffer(&mut shifted, 0, &shape);
        let m2 = kmap(&mut shifted, leaf2, &shape, 1);
        shifted.add_root(m2);
        let shifted_roots = shifted.roots().to_vec();

        assert_ne!(
            l0_term_hash(&plain, &plain_roots),
            l0_term_hash(&shifted, &shifted_roots),
            "ids shifted, so the cached plan's ids would name other nodes"
        );

        // Same graph, more roots: a different extraction and a different plan.
        let mut two_roots = new_graph();
        let leaf3 = buffer(&mut two_roots, 0, &shape);
        let a = kmap(&mut two_roots, leaf3, &shape, 1);
        let b = kmap(&mut two_roots, leaf3, &shape, 2);
        two_roots.add_root(a);
        let one = l0_term_hash(&two_roots, &two_roots.roots().to_vec());
        two_roots.add_root(b);
        let both = l0_term_hash(&two_roots, &two_roots.roots().to_vec());
        assert_ne!(one, both, "the root set is an extraction input");
    }

    #[test]
    fn invalidate_device_drops_only_that_device() {
        let cache = ReplayCache::new();
        let a = ReplayKey {
            l0_term: 1,
            device: 10,
            binding: 1,
        };
        let b = ReplayKey {
            l0_term: 1,
            device: 20,
            binding: 1,
        };
        cache.insert(a, plan_for());
        cache.insert(b, plan_for());
        assert_eq!(cache.len(), 2);
        cache.invalidate_device(10);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(a).is_none());
        assert!(cache.get(b).is_some());
    }

    #[test]
    fn the_cache_is_bounded() {
        let cache = ReplayCache::new();
        let plan = plan_for();
        for i in 0..(CAPACITY as u64 + 10) {
            cache.insert(
                ReplayKey {
                    l0_term: i,
                    device: 0,
                    binding: 0,
                },
                plan.clone(),
            );
        }
        assert_eq!(cache.len(), CAPACITY);
        assert!(
            cache
                .get(ReplayKey {
                    l0_term: 0,
                    device: 0,
                    binding: 0
                })
                .is_none(),
            "the oldest entry was evicted"
        );
    }
}
