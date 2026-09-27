//! The acyclic, append-only e-graph, the rule language and the saturation
//! driver's contract.

use crate::device::Caps;
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::launch::Launch;
use crate::ir::logical::Logical;
use crate::ir::{Children, Level, Node, Op, OpTag, Semantics};
use crate::shape::SymId;
use fixedbitset::FixedBitSet;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::fmt;
use std::sync::Arc;

/// An e-graph node id. Children hold strictly smaller ids and `union`
/// allocates above both, so acyclicity is a property of the allocator.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Id(pub u32);

impl Id {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// An e-class handle: the id of the topmost `Op::Union` node containing a
/// value, or the value's own id. Equality is not congruent.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassId(pub Id);

/// Hash-cons key: the operator plus its canonicalized children.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeKey {
    pub op: Op,
    pub children: Children,
}

/// The e-graph: one node arena, one memo, one facts table, no union-find.
pub struct EGraph {
    nodes: Vec<Node>,
    facts: Vec<ValueFacts>,
    /// Shared with every [`SaturationDelta`] recorded off this graph; `add`
    /// writes through `Arc::make_mut`.
    memo: Arc<FxHashMap<NodeKey, Id>>,
    parent: Vec<Option<Id>>,
    /// Unions that merged two classes: only a union moves a representative.
    unions: u64,
    roots: Vec<Id>,
    next_sym: u32,
    sem: Arc<dyn Semantics>,
    /// Nodes covered by a completed bounded search and its lowering floor.
    /// A node reached only by a later root remains eligible for search.
    offered: FixedBitSet,
    /// The current dim bindings, a costing hint so a symbolic extent is priced
    /// at its bound value. Set by the session; never read by a rule.
    pub dim_hints: FxHashMap<SymId, u64>,
    /// Root sets whose reachable closure has completed bounded search.
    pub saturated_root_sets: FxHashSet<Vec<Id>>,
    /// The node count as of the last completed saturation; equal to `len()`
    /// means saturation can be skipped. Set by the session.
    pub saturated_at_len: Option<usize>,
    /// Memo for the replay key's root-closure hash, per root set; valid for
    /// the life of the graph since an append cannot change what roots reach.
    pub l0_term_memo: FxHashMap<Vec<Id>, u64>,
    /// Process-unique identity of this arena, for caches keyed on ids
    /// across graphs.
    arena: u64,
    /// `(arena length, class root -> its id set)`; the length is an exact
    /// validity stamp.
    class_ids_memo: (usize, FxHashMap<ClassId, Arc<[Id]>>),
    /// Per node, the nodes that read it as a child, by the id they wrote.
    readers: Vec<SmallVec<[Id; 4]>>,
}

impl EGraph {
    pub fn new(sem: Arc<dyn Semantics>) -> Self {
        Self {
            nodes: Vec::new(),
            readers: Vec::new(),
            facts: Vec::new(),
            memo: Arc::new(FxHashMap::default()),
            parent: Vec::new(),
            unions: 0,
            roots: Vec::new(),
            next_sym: 0,
            sem,
            arena: {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            class_ids_memo: (usize::MAX, FxHashMap::default()),
            offered: FixedBitSet::new(),
            dim_hints: FxHashMap::default(),
            saturated_root_sets: FxHashSet::default(),
            saturated_at_len: None,
            l0_term_memo: FxHashMap::default(),
        }
    }

    /// Whether `id` was covered by an earlier bounded search.
    pub fn is_offered(&self, id: Id) -> bool {
        self.offered.contains(id.index())
    }

    pub fn mark_offered(&mut self, id: Id) {
        self.offered.grow_and_insert(id.index());
    }

    /// Every id of every class the current roots reach, closed under class
    /// membership and children.
    pub fn reachable_from_roots(&self) -> FixedBitSet {
        let mut seen = FixedBitSet::with_capacity(self.nodes.len());
        let mut stack: Vec<Id> = self.roots.clone();
        while let Some(id) = stack.pop() {
            if seen.contains(id.index()) {
                continue;
            }
            for m in self.class_ids(self.class_of(id)) {
                if seen.put(m.index()) {
                    continue;
                }
                stack.extend(self.nodes[m.index()].children.iter().copied());
            }
        }
        seen
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }
    /// This arena's process-unique identity. An `Id` means nothing without it.
    pub fn arena_id(&self) -> u64 {
        self.arena
    }
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
    pub fn semantics(&self) -> &Arc<dyn Semantics> {
        &self.sem
    }
    pub fn node(&self, id: Id) -> &Node {
        &self.nodes[id.index()]
    }
    pub fn facts(&self, id: Id) -> &ValueFacts {
        &self.facts[id.index()]
    }
    pub fn level(&self, id: Id) -> Level {
        self.nodes[id.index()].level
    }
    /// Extraction roots: the loss plus every requested parameter gradient.
    pub fn roots(&self) -> &[Id] {
        &self.roots
    }
    pub fn add_root(&mut self, id: Id) {
        if !self.roots.contains(&id) {
            self.roots.push(id);
        }
    }
    /// Drop the accumulated root set; earlier resolves' roots are already
    /// buffered.
    pub fn clear_roots(&mut self) {
        self.roots.clear();
    }
    pub fn fresh_sym(&mut self) -> SymId {
        let s = SymId(self.next_sym);
        self.next_sym += 1;
        s
    }

    /// Add a node, hash-consing on canonicalized children. Returns the
    /// existing id on a memo hit.
    pub fn add(&mut self, op: Op) -> Result<Id> {
        let mut children = self.sem.children(&op);
        canonicalize(&op, &mut children);
        let next = Id(self.nodes.len() as u32);
        if let Some(bad) = children.iter().find(|c| c.0 >= next.0) {
            return Err(Error::verify_global(
                Level::Logical,
                format!("child {bad} is not strictly smaller than {next}"),
            ));
        }
        let key = NodeKey {
            op: op.clone(),
            children: children.clone(),
        };
        if let Some(&hit) = self.memo.get(&key) {
            return Ok(hit);
        }
        let ins: SmallVec<[ValueFacts; 4]> = children
            .iter()
            .map(|c| self.facts[c.index()].clone())
            .collect();
        let facts = match &op {
            Op::Union(a, _) => self.facts[a.index()].clone(),
            other => self.sem.infer(other, &ins)?,
        };
        let level = match &op {
            Op::Union(a, _) => self.nodes[a.index()].level,
            other => other.level().expect("non-union ops carry a level"),
        };
        self.index_readers_to(next.index());
        for c in &children {
            self.readers[c.index()].push(next);
        }
        self.nodes.push(Node {
            op,
            level,
            children,
        });
        self.readers.push(SmallVec::new());
        self.facts.push(facts);
        self.parent.push(None);
        // Copy-on-write: a no-op clone unless a `SaturationDelta` still holds
        // the table.
        Arc::make_mut(&mut self.memo).insert(key, next);
        Ok(next)
    }

    /// Assert `a` and `b` are equal by allocating a `Union` above the roots
    /// of both chains, which keeps a class complete.
    pub fn union(&mut self, a: Id, b: Id) -> Result<Id> {
        let (ra, rb) = (self.root_of(a), self.root_of(b));
        if ra == rb {
            return Ok(ra);
        }
        let (lo, hi) = if ra.0 < rb.0 { (ra, rb) } else { (rb, ra) };
        let u = self.add(Op::Union(lo, hi))?;
        self.parent[lo.index()] = Some(u);
        self.parent[hi.index()] = Some(u);
        self.unions += 1;
        Ok(u)
    }

    /// See [`EGraph::unions`].
    pub fn union_count(&self) -> u64 {
        self.unions
    }

    pub fn class_of(&self, id: Id) -> ClassId {
        ClassId(self.root_of(id))
    }

    pub fn root_of(&self, id: Id) -> Id {
        let mut cur = id;
        while let Some(next) = self.parent[cur.index()] {
            cur = next;
        }
        cur
    }

    /// Brings the readers index up to `len` nodes (a replayed delta adds
    /// nodes without [`Self::add`]).
    fn index_readers_to(&mut self, len: usize) {
        while self.readers.len() < len {
            let id = Id(self.readers.len() as u32);
            self.readers.push(SmallVec::new());
            for c in self.nodes[id.index()].children.clone().iter() {
                self.readers[c.index()].push(id);
            }
        }
    }

    /// Every non-`Union` node that reads any id of `class`, deduplicated.
    /// The `Union` spine is not a reader: it is the class itself.
    pub fn readers(&self, class: ClassId) -> Vec<Id> {
        let mut out: Vec<Id> = Vec::new();
        let mut seen: FxHashSet<Id> = FxHashSet::default();
        self.any_reader(class, |r| {
            if seen.insert(r) {
                out.push(r);
            }
            false
        });
        out
    }

    /// Whether some non-`Union` reader of `class` satisfies `f`; stops at
    /// the first.
    pub fn any_reader(&self, class: ClassId, mut f: impl FnMut(Id) -> bool) -> bool {
        for id in self.class_ids(class) {
            let Some(readers) = self.readers.get(id.index()) else {
                continue;
            };
            for r in readers {
                if r.index() < self.nodes.len()
                    && !matches!(self.nodes[r.index()].op, Op::Union(..))
                    && f(*r)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Every id `class` holds, `Union` spine included, in walk order.
    pub fn class_ids(&self, class: ClassId) -> Vec<Id> {
        let mut out = Vec::new();
        self.walk_class(class, |id, _| out.push(id));
        out
    }

    /// Visit every id of `class` once, spine first-seen first.
    fn walk_class(&self, class: ClassId, mut f: impl FnMut(Id, bool)) {
        let mut seen: FxHashSet<Id> = FxHashSet::default();
        let mut stack = vec![class.0];
        while let Some(cur) = stack.pop() {
            if !seen.insert(cur) {
                continue;
            }
            match self.nodes[cur.index()].op {
                Op::Union(a, b) => {
                    f(cur, true);
                    stack.push(b);
                    stack.push(a);
                }
                _ => f(cur, false),
            }
        }
    }

    /// [`Self::class_ids`], memoized against the arena length.
    pub fn class_ids_cached(&mut self, class: ClassId) -> Arc<[Id]> {
        if self.class_ids_memo.0 != self.nodes.len() {
            self.class_ids_memo = (self.nodes.len(), FxHashMap::default());
        }
        if let Some(hit) = self.class_ids_memo.1.get(&class) {
            return Arc::clone(hit);
        }
        let ids: Arc<[Id]> = self.class_ids(class).into();
        self.class_ids_memo.1.insert(class, Arc::clone(&ids));
        ids
    }

    /// Every non-`Union` member of an e-class, in creation order.
    pub fn members(&self, class: ClassId) -> Vec<Id> {
        let mut out = Vec::new();
        self.walk_class(class, |id, spine| {
            if !spine {
                out.push(id);
            }
        });
        out
    }

    pub fn builder<'a>(&'a mut self, caps: &'a Caps) -> Builder<'a> {
        Builder { graph: self, caps }
    }

    /// The next symbol this graph will mint; part of a [`SaturationDelta`]'s
    /// validity condition.
    pub fn next_sym_counter(&self) -> u32 {
        self.next_sym
    }

    /// Capture everything saturation may overwrite rather than append.
    pub fn pre_saturation(&self) -> PreSaturation {
        PreSaturation {
            len: self.nodes.len(),
            parent: self.parent.clone(),
            offered: self.offered.ones().map(|i| i as u32).collect(),
            roots: self.roots.clone(),
            next_sym: self.next_sym,
        }
    }

    /// Record everything a saturation appended above `pre`. Saturation is a
    /// pure function of `(graph, caps, rules, budget)`, so replay is exact.
    pub fn record_saturation(&self, pre: PreSaturation) -> SaturationDelta {
        debug_assert!(pre.len <= self.nodes.len());
        SaturationDelta {
            nodes: self.nodes.clone(),
            facts: self.facts.clone(),
            // Kept whole: re-inserting the tail would rehash every `NodeKey`.
            memo: Arc::clone(&self.memo),
            parent: self.parent.clone(),
            offered: self.offered.clone(),
            roots: self.roots.clone(),
            next_sym: self.next_sym,
            pre,
        }
    }

    /// Re-append a recorded saturation, or report `false` when this graph is
    /// not exactly (by value) the one the delta was recorded against.
    pub fn replay_saturation(&mut self, delta: &SaturationDelta) -> bool {
        let pre = &delta.pre;
        if self.nodes.len() != pre.len
            || self.next_sym != pre.next_sym
            || self.roots != pre.roots
            || self.parent != pre.parent
            || self.nodes[..] != delta.nodes[..pre.len]
        {
            return false;
        }
        if !self
            .offered
            .ones()
            .map(|i| i as u32)
            .eq(pre.offered.iter().copied())
        {
            return false;
        }
        self.nodes.extend_from_slice(&delta.nodes[pre.len..]);
        self.facts.extend_from_slice(&delta.facts[pre.len..]);
        self.index_readers_to(self.nodes.len());
        self.memo = Arc::clone(&delta.memo);
        self.parent.clone_from(&delta.parent);
        self.offered.clone_from(&delta.offered);
        self.roots.clone_from(&delta.roots);
        self.next_sym = delta.next_sym;
        true
    }

    /// The read-only legality view of `id`, as handed to a rule.
    pub fn facts_view<'c>(&self, id: Id, caps: &'c Caps) -> Facts<'c> {
        let node = &self.nodes[id.index()];
        Facts {
            caps,
            level: node.level,
            own: self.facts[id.index()].clone(),
            operands: node
                .children
                .iter()
                .map(|c| self.facts[c.index()].clone())
                .collect(),
        }
    }
}

fn canonicalize(op: &Op, children: &mut Children) {
    if let Op::Union(..) = op {
        children.sort_unstable();
    }
}

/// The write side of the e-graph, handed to a rule.
pub struct Builder<'a> {
    graph: &'a mut EGraph,
    caps: &'a Caps,
}

impl<'a> Builder<'a> {
    pub fn caps(&self) -> &Caps {
        self.caps
    }
    /// The class `id` belongs to.
    pub fn class_of(&self, id: Id) -> ClassId {
        self.graph.class_of(id)
    }
    /// Every node in `id`'s class.
    pub fn class_members(&self, id: Id) -> Vec<Id> {
        self.graph.members(self.graph.class_of(id))
    }
    /// Every id `id`'s class holds, spine included.
    pub fn class_ids(&self, id: Id) -> Vec<Id> {
        self.graph.class_ids(self.graph.class_of(id))
    }
    /// The graph's roots: every value a caller reads back.
    pub fn roots(&self) -> &[Id] {
        self.graph.roots()
    }
    pub fn len(&self) -> usize {
        self.graph.len()
    }
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }
    pub fn arena_id(&self) -> u64 {
        self.graph.arena_id()
    }
    /// Every node reading `id`'s class.
    pub fn readers_of(&self, id: Id) -> Vec<Id> {
        self.graph.readers(self.graph.class_of(id))
    }
    pub fn node(&self, id: Id) -> &Node {
        self.graph.node(id)
    }
    pub fn facts_of(&self, id: Id) -> &ValueFacts {
        self.graph.facts(id)
    }
    pub fn add_logical(&mut self, op: Logical) -> Result<Id> {
        self.graph.add(Op::Logical(op))
    }
    pub fn add_launch(&mut self, op: Launch) -> Result<Id> {
        self.graph.add(Op::Launch(op))
    }
    pub fn add(&mut self, op: Op) -> Result<Id> {
        self.graph.add(op)
    }
    pub fn union(&mut self, a: Id, b: Id) -> Result<Id> {
        self.graph.union(a, b)
    }
    /// Mint a fresh symbolic dim (`fold_split`'s block count).
    pub fn fresh_sym(&mut self) -> SymId {
        self.graph.fresh_sym()
    }
    /// Walk a chain of pure `Restride` views down to their base.
    pub fn trace_pure_views(&self, mut v: Id) -> ViewSpine {
        let mut views: SmallVec<[Id; 4]> = SmallVec::new();
        while let Op::Logical(Logical::Restride { x, .. }) = &self.graph.node(v).op {
            views.push(v);
            v = *x;
        }
        views.reverse();
        ViewSpine { base: v, views }
    }
}

/// A chain of pure views over one base value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewSpine {
    pub base: Id,
    pub views: SmallVec<[Id; 4]>,
}

impl ViewSpine {
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }
}

/// The read-only facts a rule's guards see; borrows only [`Caps`], so a rule
/// can hold it across a `&mut Builder` call.
pub struct Facts<'a> {
    caps: &'a Caps,
    level: Level,
    own: ValueFacts,
    operands: SmallVec<[ValueFacts; 4]>,
}

impl<'a> Facts<'a> {
    pub fn caps(&self) -> &'a Caps {
        self.caps
    }
    pub fn level(&self) -> Level {
        self.level
    }
    pub fn own(&self) -> &ValueFacts {
        &self.own
    }
    pub fn operand(&self, slot: usize) -> Option<&ValueFacts> {
        self.operands.get(slot)
    }
    pub fn operands(&self) -> &[ValueFacts] {
        &self.operands
    }
    pub fn numeric(&self, slot: usize) -> Option<crate::dtype::NumericContract> {
        self.operands.get(slot).map(|f| f.numeric)
    }
    pub fn dim(&self, slot: usize, axis: usize) -> Option<crate::shape::Dim> {
        self.operands.get(slot)?.shape.get(axis).copied()
    }
    pub fn dtype(&self, slot: usize) -> Option<crate::dtype::Dtype> {
        self.operands.get(slot).map(|f| f.dtype)
    }
}

/// Whether a rule adds an alternative or is guaranteed to descend a level;
/// on budget exhaustion only `StrictlyLowering` rules run.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuleTag {
    Additive,
    StrictlyLowering,
}

/// A rewrite rule's body: `Some(id)` reports the id unioned into the chain,
/// `None` that it did not apply.
pub type RuleFn = fn(&mut Builder<'_>, Id, &Node, &Facts<'_>) -> Option<Id>;

/// One rewrite rule, offered every node whose tag is one of `heads`.
#[derive(Copy, Clone)]
pub struct Rule {
    pub name: &'static str,
    pub level: Level,
    pub heads: &'static [OpTag],
    pub tag: RuleTag,
    pub apply: RuleFn,
}

impl fmt::Debug for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rule")
            .field("name", &self.name)
            .field("level", &self.level)
            .field("heads", &self.heads)
            .field("tag", &self.tag)
            .finish()
    }
}

/// Saturation limits. Exhausting any degrades to
/// [`RuleTag::StrictlyLowering`]. Counts, never a clock, so plans are
/// deterministic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SaturationBudget {
    /// Retain existing nodes and allow `node_slope * new_nodes + node_slack`
    /// for nodes reached by this search that have not been offered before.
    pub node_slope: u32,
    pub node_slack: u32,
    pub max_rounds: u32,
    /// Rule bodies invoked.
    pub max_applications: u32,
    /// Raise the application limit to at least this many invocations per
    /// newly offered reachable node. Zero keeps `max_applications` fixed.
    pub application_slope: u32,
}

impl Default for SaturationBudget {
    /// The shipped budget. 10 rounds clears attention (the deepest chain,
    /// 9 rounds); changing any term moves every golden plan.
    fn default() -> Self {
        Self {
            node_slope: 8,
            node_slack: 4096,
            max_rounds: 10,
            max_applications: 200_000,
            application_slope: 0,
        }
    }
}

/// What saturation did. Truncation is never silent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SaturationReport {
    pub initial_nodes: usize,
    pub final_nodes: usize,
    pub rounds: u32,
    /// Wall time. Observability only — nothing in the driver reads it.
    pub micros: u64,
    /// Rule bodies invoked, against `SaturationBudget::max_applications`.
    pub applications: u32,
    pub saturated: bool,
    /// Chains that stopped receiving additive alternatives at a budget.
    pub truncated: Vec<Id>,
    pub fired: Vec<(&'static str, u32)>,
}

/// The overwritable part of a graph's state immediately before saturation:
/// the exact condition a [`SaturationDelta`]'s replay is valid under.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreSaturation {
    len: usize,
    parent: Vec<Option<Id>>,
    offered: Vec<u32>,
    roots: Vec<Id>,
    next_sym: u32,
}

impl PreSaturation {
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Everything one saturation appended to a graph, replayable onto any graph
/// in the identical pre-state; the caller guarantees caps, rules and budget.
#[derive(Clone, Debug)]
pub struct SaturationDelta {
    pre: PreSaturation,
    /// The whole post-saturation state; `nodes[..pre.len]` doubles as the
    /// validity condition.
    nodes: Vec<Node>,
    facts: Vec<ValueFacts>,
    memo: Arc<FxHashMap<NodeKey, Id>>,
    parent: Vec<Option<Id>>,
    offered: FixedBitSet,
    roots: Vec<Id>,
    next_sym: u32,
}

impl SaturationDelta {
    /// Nodes the graph held before saturation.
    pub fn prefix(&self) -> usize {
        self.pre.len
    }
    /// Nodes saturation appended.
    pub fn added(&self) -> usize {
        self.nodes.len() - self.pre.len
    }
    /// O(1) rejection before comparing node lists.
    pub fn could_apply_to(&self, graph: &EGraph) -> bool {
        graph.len() == self.pre.len && graph.next_sym_counter() == self.pre.next_sym
    }
}

/// The saturation driver; targets contribute rules, never a driver.
pub trait Saturate: Send + Sync {
    fn saturate(
        &self,
        graph: &mut EGraph,
        caps: &Caps,
        rules: &[Rule],
        budget: SaturationBudget,
    ) -> Result<SaturationReport>;
}
