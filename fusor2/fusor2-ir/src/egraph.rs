//! The acyclic, append-only e-graph, the rule language and the saturation
//! driver's contract.

use crate::device::Caps;
use crate::error::{Error, Result};
use crate::facts::ValueFacts;
use crate::ir::level0::L0;
use crate::ir::level1::L1;
use crate::ir::{Children, Level, Node, Op, OpTag, Semantics};
use crate::shape::SymId;
use fixedbitset::FixedBitSet;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// An e-graph node id. Ids are dense and monotone: `children` may only hold
/// ids strictly smaller than the node's own, and `union(a, b)` allocates an
/// id greater than both — so acyclicity is a property of this allocator and
/// no rule author can violate it.
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
/// value, or the value's own id when it has no alternatives.
///
/// There is no `UnionFind`, no rank, no path compression and no
/// `rebuild()`. A class's identity is
/// never a merge artifact, so max-rank can never stop preserving global
/// acyclicity. The price, paid deliberately: **equality is not congruent** —
/// unioning `a` and `b` does not union `f(a)` and `f(b)`. Alternatives are
/// minted by rules *at the consumer*, and patterns may match a spine
/// ([`Builder::trace_pure_views`]), which is what makes a consumer-rooted
/// rewrite like `sink_epilogue` expressible with no multi-root rule form.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassId(pub Id);

/// Hash of a node's hash-cons key — the operator plus its canonicalized
/// children. Commutative ops sort children by [`Id`] at construction, so
/// associativity and commutativity are a canonical form, not a rule family.
/// The key itself is never stored: the arena node *is* the key, so a
/// candidate under this hash is confirmed against `nodes[id]` directly.
fn key_hash(op: &Op, children: &Children) -> u64 {
    let mut h = FxHasher::default();
    op.hash(&mut h);
    children.hash(&mut h);
    h.finish()
}

/// The e-graph: one node arena, one memo, one facts table, no union-find.
///
/// Nodes and facts are held behind [`Arc`] so the saturation driver can pin
/// one node's `&Node`/`&ValueFacts` across the `&mut Builder` it hands to a
/// rule — a refcount bump per visit, never a deep clone.
pub struct EGraph {
    nodes: Vec<Arc<Node>>,
    facts: Vec<Arc<ValueFacts>>,
    /// [`key_hash`] to the ids carrying it; holds no `Op` payloads at all.
    /// Shared with every [`SaturationDelta`] recorded off this graph, so a
    /// replay is a refcount bump rather than a copy of the whole table.
    /// `add` is the only writer and takes it back by `Arc::make_mut`, so a
    /// graph that keeps growing after a replay still hash-conses against a
    /// fully populated memo — copy-on-write, not a lazy rebuild.
    memo: Arc<FxHashMap<u64, SmallVec<[Id; 1]>>>,
    parent: Vec<Option<Id>>,
    defns: FixedBitSet,
    roots: Vec<Id>,
    next_sym: u32,
    sem: Arc<dyn Semantics>,
}

impl EGraph {
    pub fn new(sem: Arc<dyn Semantics>) -> Self {
        Self {
            nodes: Vec::new(),
            facts: Vec::new(),
            memo: Arc::new(FxHashMap::default()),
            parent: Vec::new(),
            defns: FixedBitSet::new(),
            roots: Vec::new(),
            next_sym: 0,
            sem,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
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
    /// The arena's shared handle to `id`'s node, for pinning a `&Node`
    /// across a later `&mut` borrow of the graph. A refcount bump; the
    /// pointee is immutable, `add` being the arena's only writer.
    pub fn node_arc(&self, id: Id) -> Arc<Node> {
        Arc::clone(&self.nodes[id.index()])
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
    /// Mark an id as a macro op's definitional expansion. Sugar and its
    /// `defn` are unioned at construction, so there is nothing to
    /// recognize; a `defn` node is never evicted.
    pub fn mark_defn(&mut self, id: Id) {
        self.defns.grow(self.nodes.len());
        self.defns.insert(id.index());
    }
    pub fn is_defn(&self, id: Id) -> bool {
        self.defns.contains(id.index())
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
                Level::L0,
                format!("child {bad} is not strictly smaller than {next}"),
            ));
        }
        let hash = key_hash(&op, &children);
        if let Some(bucket) = self.memo.get(&hash) {
            for &hit in bucket {
                let n = &self.nodes[hit.index()];
                if n.op == op && n.children == children {
                    return Ok(hit);
                }
            }
        }
        let (facts, level) = match &op {
            Op::Union(a, _) => (
                Arc::clone(&self.facts[a.index()]),
                self.nodes[a.index()].level,
            ),
            other => {
                let ins: SmallVec<[ValueFacts; 4]> = children
                    .iter()
                    .map(|c| (*self.facts[c.index()]).clone())
                    .collect();
                let facts = Arc::new(self.sem.infer(other, &ins)?);
                (facts, other.level().expect("non-union ops carry a level"))
            }
        };
        self.nodes.push(Arc::new(Node {
            op,
            level,
            children,
        }));
        self.facts.push(facts);
        self.parent.push(None);
        // Copy-on-write: a no-op clone unless a `SaturationDelta` still holds
        // the table, which is exactly the case where the copy is required for
        // the delta to stay a faithful recording.
        Arc::make_mut(&mut self.memo)
            .entry(hash)
            .or_default()
            .push(next);
        Ok(next)
    }

    /// Assert `a` and `b` are equal by allocating a `Union` above the
    /// *roots* of both chains. Rooting keeps a class complete: unioning `a`
    /// with `b` and later `a` with `d` must leave one class `{a, b, d}`.
    pub fn union(&mut self, a: Id, b: Id) -> Result<Id> {
        let (ra, rb) = (self.root_of(a), self.root_of(b));
        if ra == rb {
            return Ok(ra);
        }
        let (lo, hi) = if ra.0 < rb.0 { (ra, rb) } else { (rb, ra) };
        let u = self.add(Op::Union(lo, hi))?;
        self.parent[lo.index()] = Some(u);
        self.parent[hi.index()] = Some(u);
        Ok(u)
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

    pub fn chain(&self, id: Id) -> Vec<Id> {
        self.members(self.class_of(id)).into_vec()
    }

    /// Every id that resolves to `class`, **including the `Union` spine**.
    ///
    /// [`members`](Self::members) answers "what may `sigma` select", so it
    /// drops the `Union` nodes. A `Union` id is still a name a caller holds:
    /// `macro_op` returns the id `union(defn, sugar)` produced, so the
    /// `Tensor` the user reads back *is* the spine node. Anything keyed on
    /// "this value" rather than "this candidate" has to use this.
    pub fn class_ids(&self, class: ClassId) -> SmallVec<[Id; 8]> {
        let mut out: SmallVec<[Id; 8]> = SmallVec::new();
        let mut stack: SmallVec<[Id; 8]> = SmallVec::new();
        stack.push(class.0);
        while let Some(cur) = stack.pop() {
            if out.contains(&cur) {
                continue;
            }
            out.push(cur);
            if let Op::Union(a, b) = self.nodes[cur.index()].op {
                stack.push(b);
                stack.push(a);
            }
        }
        out
    }

    /// Every non-`Union` member of an e-class, in creation order.
    pub fn members(&self, class: ClassId) -> SmallVec<[Id; 8]> {
        let mut out: SmallVec<[Id; 8]> = SmallVec::new();
        let mut stack: SmallVec<[Id; 8]> = SmallVec::new();
        stack.push(class.0);
        while let Some(cur) = stack.pop() {
            match self.nodes[cur.index()].op {
                Op::Union(a, b) => {
                    stack.push(b);
                    stack.push(a);
                }
                _ => {
                    if !out.contains(&cur) {
                        out.push(cur);
                    }
                }
            }
        }
        out
    }

    pub fn builder<'a>(&'a mut self, caps: &'a Caps) -> Builder<'a> {
        Builder { graph: self, caps }
    }

    /// The next symbol this graph will mint. Part of a
    /// [`SaturationDelta`]'s validity condition: two graphs with identical
    /// nodes but a different `next_sym` saturate to different `fold_split`
    /// block symbols.
    pub fn next_sym_counter(&self) -> u32 {
        self.next_sym
    }

    /// Capture everything saturation may **overwrite** rather than append.
    ///
    /// `nodes` and `facts` are push-only — `add` is the sole allocator and
    /// nothing in this module ever assigns into either — so they are still
    /// readable after saturation and are captured at record time instead.
    /// `parent`, `defns`, `roots` and `next_sym` are not, so they are
    /// captured here.
    pub fn pre_saturation(&self) -> PreSaturation {
        PreSaturation {
            len: self.nodes.len(),
            parent: self.parent.clone(),
            defns: self.defns.ones().map(|i| i as u32).collect(),
            roots: self.roots.clone(),
            next_sym: self.next_sym,
        }
    }

    /// Record everything a saturation appended above `pre`.
    ///
    /// Saturation is a pure function of `(graph, caps, rules, budget)` —
    /// [`SaturationBudget`]'s doc says so and
    /// `saturate::tests::saturation_is_deterministic_under_any_wall_time`
    /// proves it — so a graph in exactly the state `pre` describes saturates
    /// to exactly these nodes at exactly these ids. Replaying the recording
    /// is therefore the same graph, not an approximation of one.
    pub fn record_saturation(&self, pre: PreSaturation) -> SaturationDelta {
        debug_assert!(pre.len <= self.nodes.len());
        SaturationDelta {
            nodes: self.nodes.clone(),
            facts: self.facts.clone(),
            // Kept whole rather than as the appended entries alone —
            // re-inserting the tail would rehash every key, and the tail
            // is the overwhelming majority of the table. Sharing it is free:
            // the next `add` on this graph is what pays for a private copy,
            // and only if there ever is one.
            memo: Arc::clone(&self.memo),
            parent: self.parent.clone(),
            defns: self.defns.clone(),
            roots: self.roots.clone(),
            next_sym: self.next_sym,
            pre,
        }
    }

    /// Re-append a recorded saturation, or report `false` when this graph is
    /// not the one the delta was recorded against.
    ///
    /// The validity check is **exact, not a fingerprint**: every pre-existing
    /// node, every parent link, the `defn` set, the root set and the symbol
    /// counter are compared by value. There is no collision to reason about —
    /// either the graph is bit-for-bit the term the recording was taken from
    /// and the replay is that same pure function's answer, or nothing is
    /// touched and the caller saturates for real. Rejection is cheap: `len`
    /// and `next_sym` reject a mismatch before any node is looked at.
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
        if !self.defns.ones().map(|i| i as u32).eq(pre.defns.iter().copied()) {
            return false;
        }
        // The prefix is already equal by the check above, so only the tail is
        // copied. `memo` is not copied at all: the recording's table is the
        // post-saturation table by construction, and `add` is its only reader
        // and writer, so the graph adopts it and `Arc::make_mut` forks it if
        // and only if this graph is later grown.
        self.nodes.extend_from_slice(&delta.nodes[pre.len..]);
        self.facts.extend_from_slice(&delta.facts[pre.len..]);
        self.memo = Arc::clone(&delta.memo);
        self.parent.clone_from(&delta.parent);
        self.defns.clone_from(&delta.defns);
        self.roots.clone_from(&delta.roots);
        self.next_sym = delta.next_sym;
        true
    }

    /// The read-only legality view of `id`, as handed to a rule. The
    /// returned [`Facts`] borrows **only** `caps`, never the graph, so a
    /// driver can build it and then hand out a `&mut Builder` over the same
    /// graph — that is what makes the four-argument [`RuleFn`] sound. It
    /// shares the arena's facts rather than cloning them.
    pub fn facts_view<'c>(&self, id: Id, caps: &'c Caps) -> Facts<'c> {
        let node = &self.nodes[id.index()];
        Facts {
            caps,
            level: node.level,
            own: Arc::clone(&self.facts[id.index()]),
            operands: node
                .children
                .iter()
                .map(|c| Arc::clone(&self.facts[c.index()]))
                .collect(),
        }
    }
}

fn canonicalize(op: &Op, children: &mut Children) {
    if let Op::Union(..) = op {
        children.sort_unstable();
    }
}

/// The write side of the e-graph, handed to a rule. Deliberately exposes
/// **no** consumer counts, no liveness, no cost and no extraction state.
/// Guards written against a `Builder` and a [`Facts`] can only encode
/// legality; profitability lives in the cost model or nowhere. That
/// restriction is enforced by this API surface, not by convention.
pub struct Builder<'a> {
    graph: &'a mut EGraph,
    caps: &'a Caps,
}

impl<'a> Builder<'a> {
    pub fn caps(&self) -> &Caps {
        self.caps
    }
    pub fn node(&self, id: Id) -> &Node {
        self.graph.node(id)
    }
    pub fn facts_of(&self, id: Id) -> &ValueFacts {
        self.graph.facts(id)
    }
    pub fn level_of(&self, id: Id) -> Level {
        self.graph.level(id)
    }
    pub fn add_l0(&mut self, op: L0) -> Result<Id> {
        self.graph.add(Op::L0(op))
    }
    pub fn add_l1(&mut self, op: L1) -> Result<Id> {
        self.graph.add(Op::L1(op))
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
    pub fn mark_defn(&mut self, id: Id) {
        self.graph.mark_defn(id);
    }

    /// Walk a chain of pure `Restride` views down to their base. This is what
    /// makes `sink_epilogue` a single-rooted rule.
    pub fn trace_pure_views(&self, mut v: Id) -> ViewSpine {
        let mut views: SmallVec<[Id; 4]> = SmallVec::new();
        loop {
            match &self.graph.node(v).op {
                Op::L0(L0::Restride { x, .. }) => {
                    views.push(v);
                    v = *x;
                }
                _ => break,
            }
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

/// The read-only capability token a rule's guards see. Borrows only
/// [`Caps`], never the graph, so a rule can hold it across a `&mut Builder`
/// call. It exposes types, shapes, numerics and device caps and
/// **structurally does not expose** consumer counts, liveness, cost or
/// extraction state.
pub struct Facts<'a> {
    caps: &'a Caps,
    level: Level,
    own: Arc<ValueFacts>,
    operands: SmallVec<[Arc<ValueFacts>; 4]>,
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
        self.operands.get(slot).map(|f| &**f)
    }
    pub fn operands(&self) -> &[Arc<ValueFacts>] {
        &self.operands
    }
    pub fn dtype(&self, slot: usize) -> Option<crate::dtype::Dtype> {
        self.operands.get(slot).map(|f| f.dtype)
    }
}

/// Whether a rule adds an alternative or is guaranteed to descend a level.
/// On budget exhaustion the driver offers only `StrictlyLowering` rules, so
/// every chain still reaches a runnable plan — a degraded-but-valid plan,
/// never a hard error.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuleTag {
    Additive,
    StrictlyLowering,
}

/// A rewrite rule's body. The driver pins the node's [`Arc`] and builds
/// [`Facts`] before calling, so the four parameters do not alias. `Some(id)`
/// reports
/// the id the rule unioned into the chain; `None` means it did not apply.
pub type RuleFn = fn(&mut Builder<'_>, Id, &Node, &Facts<'_>) -> Option<Id>;

/// One rewrite rule. **Rule order carries no semantics**; the fixed order
/// in a `RULES: &[Rule]` exists only for reproducibility.
#[derive(Copy, Clone)]
pub struct Rule {
    pub name: &'static str,
    pub level: Level,
    pub head: OpTag,
    pub tag: RuleTag,
    pub apply: RuleFn,
}

impl fmt::Debug for Rule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rule")
            .field("name", &self.name)
            .field("level", &self.level)
            .field("head", &self.head)
            .field("tag", &self.tag)
            .finish()
    }
}

/// Saturation limits. Exhausting any of them degrades to
/// [`RuleTag::StrictlyLowering`]; it is never an error.
///
/// **Every term is a count, never a clock.** A wall-clock cutoff makes the
/// set of alternatives — and therefore the extracted plan, and therefore the
/// `PlanHash` the cross-process cache is keyed on — depend on how loaded the
/// machine is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SaturationBudget {
    /// `MAX_NODES = node_slope * initial + node_slack`.
    pub node_slope: u32,
    pub node_slack: u32,
    pub max_rounds: u32,
    /// Rule bodies invoked. The `(rule, node)` fired set already bounds this
    /// at `rules * nodes`; this is the term that keeps a pathological graph's
    /// compile time bounded without reading a clock.
    pub max_applications: u32,
}

impl Default for SaturationBudget {
    /// `8 * initial + 4096` nodes, 10 rounds, 200k rule applications.
    ///
    /// 200k is ~40x the largest graph in the conformance suite and still
    /// bounds a 3,000-node step graph's saturation, so in practice the node
    /// ceiling and the round count are what bind.
    ///
    /// A round count bounds chain *depth*, and the deepest chain in the suite
    /// is attention: first saturation is at 9 rounds (CPU, non-causal), 8
    /// (CPU, causal), 7 (GPU, non-causal) and 6 (GPU, causal). The chain
    /// converges rather than diverges — CPU non-causal reports 472 nodes at 7
    /// rounds, 484 at 8 and a fixpoint at 9 — so 10 is the tightest value that
    /// clears all four with a round of headroom.
    ///
    /// Raising this moves extraction across the whole suite and `PlanHash` is
    /// a golden, so it is not a knob to turn without re-running both
    /// `fusor2-conformance` and `cargo test --workspace`.
    fn default() -> Self {
        Self {
            node_slope: 8,
            node_slack: 4096,
            max_rounds: 10,
            max_applications: 200_000,
        }
    }
}

/// What saturation did. Truncation is never silent.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SaturationReport {
    pub initial_nodes: usize,
    pub final_nodes: usize,
    pub rounds: u32,
    /// Wall time. **Observability only** — nothing in the driver reads it,
    /// so two runs of one graph report different micros and the same graph.
    pub micros: u64,
    /// Rule bodies invoked, against `SaturationBudget::max_applications`.
    pub applications: u32,
    pub saturated: bool,
    /// Chains that stopped receiving additive alternatives because a budget
    /// was hit. Reported to conformance.
    pub truncated: Vec<Id>,
    pub fired: Vec<(&'static str, u32)>,
}

/// The overwritable part of a graph's state immediately before saturation.
/// Captured by [`EGraph::pre_saturation`]; carried inside a
/// [`SaturationDelta`] as the exact condition its replay is valid under.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreSaturation {
    len: usize,
    parent: Vec<Option<Id>>,
    defns: Vec<u32>,
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
/// in the identical pre-state.
///
/// Holds no [`Caps`], no rule table and no [`SaturationBudget`]: it is a
/// recording of an *outcome*, and the caller is what guarantees the other
/// three inputs are the ones that produced it. A `Session` fixes its device,
/// its rules and the default budget for its whole life, which is exactly that
/// guarantee.
#[derive(Clone, Debug)]
pub struct SaturationDelta {
    pre: PreSaturation,
    /// The whole post-saturation state. `nodes[..pre.len]` doubles as the
    /// recording's validity condition — `nodes` is append-only, so those
    /// entries are exactly the term saturation was handed.
    nodes: Vec<Arc<Node>>,
    facts: Vec<Arc<ValueFacts>>,
    memo: Arc<FxHashMap<u64, SmallVec<[Id; 1]>>>,
    parent: Vec<Option<Id>>,
    defns: FixedBitSet,
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
    /// O(1) rejection, so a memo scan does not compare node lists it is
    /// already known to differ from.
    pub fn could_apply_to(&self, graph: &EGraph) -> bool {
        graph.len() == self.pre.len && graph.next_sym_counter() == self.pre.next_sym
    }
}

/// The saturation driver. Object-safe. Implemented once in `fusor2-ir`;
/// targets contribute rules, never a driver.
pub trait Saturate: Send + Sync {
    fn saturate(
        &self,
        graph: &mut EGraph,
        caps: &Caps,
        rules: &[Rule],
        budget: SaturationBudget,
    ) -> Result<SaturationReport>;
}
