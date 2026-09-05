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
/// Equality is not congruent — unioning `a` and `b` does not union `f(a)`
/// and `f(b)`. Alternatives are minted by rules at the consumer, and
/// patterns may match a spine ([`Builder::trace_pure_views`]).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassId(pub Id);

/// Hash-cons key: the operator plus its canonicalized children. Commutative
/// ops sort children by [`Id`] at construction, so associativity and
/// commutativity are a canonical form, not a rule family.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NodeKey {
    pub op: Op,
    pub children: Children,
}

/// The e-graph: one node arena, one memo, one facts table, no union-find.
pub struct EGraph {
    nodes: Vec<Node>,
    facts: Vec<ValueFacts>,
    /// Shared with every [`SaturationDelta`] recorded off this graph. `add`
    /// is the only writer and takes it back by `Arc::make_mut`, so a graph
    /// that keeps growing after a replay still hash-conses against a fully
    /// populated memo.
    memo: Arc<FxHashMap<NodeKey, Id>>,
    parent: Vec<Option<Id>>,
    defns: FixedBitSet,
    roots: Vec<Id>,
    next_sym: u32,
    sem: Arc<dyn Semantics>,
    /// Nodes below this index have already had every rule offered to them in
    /// a fully-saturated earlier pass. A rule's applicability is a function
    /// of `(node, child facts, caps)` — all immutable once minted — so
    /// re-offering below the frontier can only re-mint. Advanced by the
    /// saturation driver only when a pass finishes unbudgeted.
    pub saturation_frontier: usize,
    /// The node count as of the last completed saturation on this graph.
    /// `add` is the only structural mutation, so `saturated_at_len ==
    /// Some(len())` means the graph is exactly the one saturation last ran
    /// on — a decode step that rebuilt only memo hits skips saturation. The
    /// session owns setting it.
    pub saturated_at_len: Option<usize>,
    /// Memo for the replay key's whole-graph term hash: `(roots, len) ->
    /// hash`. Valid for the same reason as `saturated_at_len`.
    pub l0_term_memo: Option<(Vec<Id>, usize, u64)>,
    /// Process-unique identity of this arena. An [`Id`] names a node only
    /// together with the graph it indexes, so anything that caches per-node
    /// work keyed on ids across graphs has to carry this.
    arena: u64,
    /// `(arena length, class root -> its id set)`. A class's ids change only
    /// when a node is appended, so the arena length is an exact validity
    /// stamp.
    class_ids_memo: (usize, FxHashMap<ClassId, Arc<[Id]>>),
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
            arena: {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            class_ids_memo: (usize::MAX, FxHashMap::default()),
            saturation_frontier: 0,
            saturated_at_len: None,
            l0_term_memo: None,
        }
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
    /// Drop the accumulated root set. A resolve plans for the values that
    /// call requested; roots left over from earlier resolves are already
    /// buffered (or are views over something buffered) and re-verifying them
    /// against a plan that deliberately does not cover them is a false
    /// failure.
    pub fn clear_roots(&mut self) {
        self.roots.clear();
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
        self.nodes.push(Node {
            op,
            level,
            children,
        });
        self.facts.push(facts);
        self.parent.push(None);
        // Copy-on-write: a no-op clone unless a `SaturationDelta` still holds
        // the table.
        Arc::make_mut(&mut self.memo).insert(key, next);
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
        self.members(self.class_of(id))
    }

    /// Every id that resolves to `class`, including the `Union` spine.
    ///
    /// A `Union` id is still a name a caller holds: `macro_op` returns the id
    /// `union(defn, sugar)` produced, so the `Tensor` the user reads back is
    /// the spine node. Anything keyed on "this value" rather than "this
    /// candidate" has to use this.
    pub fn class_ids(&self, class: ClassId) -> Vec<Id> {
        let mut out = Vec::new();
        // The spine is a DAG; a set membership test keeps this linear.
        let mut seen: FxHashSet<Id> = FxHashSet::default();
        let mut stack = vec![class.0];
        while let Some(cur) = stack.pop() {
            if !seen.insert(cur) {
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
        // The `Union` spine is a DAG, so shared spine nodes are marked seen
        // to avoid re-descending their subtrees.
        let mut seen: FxHashSet<Id> = FxHashSet::default();
        let mut stack = vec![class.0];
        while let Some(cur) = stack.pop() {
            if !seen.insert(cur) {
                continue;
            }
            match self.nodes[cur.index()].op {
                Op::Union(a, b) => {
                    stack.push(b);
                    stack.push(a);
                }
                _ => out.push(cur),
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

    /// Capture everything saturation may overwrite rather than append.
    ///
    /// `nodes` and `facts` are push-only, so they are captured at record time
    /// instead; `parent`, `defns`, `roots` and `next_sym` are captured here.
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
    /// Saturation is a pure function of `(graph, caps, rules, budget)`, so a
    /// graph in exactly the state `pre` describes saturates to exactly these
    /// nodes at exactly these ids: replaying the recording is the same graph.
    pub fn record_saturation(&self, pre: PreSaturation) -> SaturationDelta {
        debug_assert!(pre.len <= self.nodes.len());
        SaturationDelta {
            nodes: self.nodes.clone(),
            facts: self.facts.clone(),
            // Kept whole: re-inserting the tail would rehash every `NodeKey`.
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
    /// The validity check is exact, not a fingerprint: every pre-existing
    /// node, every parent link, the `defn` set, the root set and the symbol
    /// counter are compared by value. `len` and `next_sym` reject a mismatch
    /// before any node is looked at.
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
            .defns
            .ones()
            .map(|i| i as u32)
            .eq(pre.defns.iter().copied())
        {
            return false;
        }
        // The prefix is already equal, so only the tail is copied. The
        // recording's memo is the post-saturation table by construction; the
        // graph adopts it and `Arc::make_mut` forks it if later grown.
        self.nodes.extend_from_slice(&delta.nodes[pre.len..]);
        self.facts.extend_from_slice(&delta.facts[pre.len..]);
        self.memo = Arc::clone(&delta.memo);
        self.parent.clone_from(&delta.parent);
        self.defns.clone_from(&delta.defns);
        self.roots.clone_from(&delta.roots);
        self.next_sym = delta.next_sym;
        self.saturation_frontier = self.nodes.len();
        true
    }

    /// The read-only legality view of `id`, as handed to a rule. The
    /// returned [`Facts`] borrows only `caps`, never the graph, so a driver
    /// can build it and then hand out a `&mut Builder` over the same graph.
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

/// The write side of the e-graph, handed to a rule. Exposes no consumer
/// counts, liveness, cost or extraction state: guards can only encode
/// legality; profitability lives in the cost model.
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
    pub fn mark_defn(&mut self, id: Id) {
        self.graph.mark_defn(id);
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

/// The read-only capability token a rule's guards see. Borrows only
/// [`Caps`], never the graph, so a rule can hold it across a `&mut Builder`
/// call. Exposes types, shapes, numerics and device caps; never consumer
/// counts, liveness, cost or extraction state.
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

/// Whether a rule adds an alternative or is guaranteed to descend a level.
/// On budget exhaustion the driver offers only `StrictlyLowering` rules, so
/// every chain still reaches a runnable plan — a degraded-but-valid plan,
/// never a hard error.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuleTag {
    Additive,
    StrictlyLowering,
}

/// A rewrite rule's body. The driver clones the node and builds [`Facts`]
/// before calling, so the four parameters do not alias. `Some(id)` reports
/// the id the rule unioned into the chain; `None` means it did not apply.
pub type RuleFn = fn(&mut Builder<'_>, Id, &Node, &Facts<'_>) -> Option<Id>;

/// One rewrite rule. Rule order carries no semantics; the fixed order in a
/// `RULES: &[Rule]` exists only for reproducibility.
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
/// Every term is a count, never a clock: a wall-clock cutoff makes the set
/// of alternatives — and therefore the extracted plan and its `PlanHash` —
/// depend on how loaded the machine was.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SaturationBudget {
    /// `MAX_NODES = node_slope * initial + node_slack`.
    pub node_slope: u32,
    pub node_slack: u32,
    pub max_rounds: u32,
    /// Rule bodies invoked; keeps a pathological graph's compile time bounded
    /// without reading a clock.
    pub max_applications: u32,
}

impl Default for SaturationBudget {
    /// The shipped budget: `8 * initial + 4096` nodes, 10 rounds, 200k rule
    /// applications.
    ///
    /// A round count bounds chain depth; the deepest chain in the suite is
    /// attention, whose slowest member first saturates at 9 rounds, so 10 is
    /// the tightest value that clears all four variants with headroom.
    ///
    /// Raising any of these moves extraction across the whole suite and
    /// `PlanHash` is a golden, so it is not a knob to turn without re-running
    /// both `fusor-conformance` and `cargo test --workspace`.
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
    /// Wall time. Observability only — nothing in the driver reads it.
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
/// recording of an outcome, and the caller guarantees the other three inputs
/// are the ones that produced it (a `Session` fixes all three for its life).
#[derive(Clone, Debug)]
pub struct SaturationDelta {
    pre: PreSaturation,
    /// The whole post-saturation state. `nodes[..pre.len]` doubles as the
    /// recording's validity condition — `nodes` is append-only, so those
    /// entries are exactly the term saturation was handed.
    nodes: Vec<Node>,
    facts: Vec<ValueFacts>,
    memo: Arc<FxHashMap<NodeKey, Id>>,
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

/// The saturation driver. Object-safe. Implemented once in `fusor-ir`;
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
