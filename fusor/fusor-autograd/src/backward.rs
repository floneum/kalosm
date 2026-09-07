//! The reverse-mode transform itself: a reverse walk over the primal with a
//! pending-children counter, dispatching each node through [`crate::ADJOINTS`]
//! and accumulating into a per-value gradient slot.
//!
//! The walk needs the primal's topology, which [`Tape`] does not expose — a
//! tape only writes. [`Reverse`] carries a snapshot taken with
//! [`Reverse::over`]; [`backward_into`] is the one-call form.

use crate::adjoints::adjoint_of;
use crate::custom::CustomRegistry;
use crate::structural::structural_adjoint;
use crate::tape::GraphTape;
use fusor_ir::autograd::{Adjoint, AdjointKind, Autograd, Grads, Tape, Val};
use fusor_ir::device::Caps;
use fusor_ir::dtype::{Dtype, NumericContract};
use fusor_ir::egraph::{EGraph, Id};
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::ir::{Children, Node, Op};
use fusor_ir::{Error, Result};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::Arc;

/// A read-only snapshot of the primal graph's structure.
///
/// Ids are dense and monotone and the graph is append-only, so a snapshot
/// taken before the backward is still valid while the backward appends: the
/// nodes it describes never move and never change.
#[derive(Clone, Debug, Default)]
pub struct Topology {
    nodes: Vec<Node>,
    numeric: Vec<NumericContract>,
    is_param: Vec<bool>,
    dtype: Vec<Dtype>,
}
impl Topology {
    pub fn of(graph: &EGraph) -> Self {
        let n = graph.len();
        let mut nodes = Vec::with_capacity(n);
        let mut numeric = Vec::with_capacity(n);
        let mut dtype = Vec::with_capacity(n);
        let mut is_param = vec![false; n];
        for (i, slot) in is_param.iter_mut().enumerate() {
            let id = Id(i as u32);
            let node = graph.node(id);
            let external = matches!(
                &node.op,
                Op::Logical(Logical::Leaf(
                    LeafKind::Param { .. } | LeafKind::Buffer { .. }
                ))
            );
            // Only a float leaf is differentiable. An index buffer is `U32`
            // and `Gather`'s adjoint correctly hands it `None`; marking it
            // requires-grad would starve it and turn every embedding
            // backward into an error.
            *slot = external && graph.facts(id).dtype.is_float();
            nodes.push(node.clone());
            numeric.push(graph.facts(id).numeric);
            dtype.push(graph.facts(id).dtype);
        }
        Self {
            nodes,
            numeric,
            is_param,
            dtype,
        }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn node(&self, id: Val) -> &Node {
        &self.nodes[id.index()]
    }

    pub fn numeric(&self, id: Val) -> NumericContract {
        self.numeric[id.index()]
    }

    pub fn dtype(&self, id: Val) -> Dtype {
        self.dtype[id.index()]
    }

    /// An externally supplied leaf is where `requires_grad` originates;
    /// everything else is derived, never annotated.
    ///
    /// A float `Buffer` or `Param` qualifies; a `Leaf::Const` does not. An
    /// integer leaf is an index, never a differentiable value.
    pub fn is_param(&self, id: Val) -> bool {
        self.is_param[id.index()]
    }

    /// Operands the adjoint walk descends into.
    ///
    /// A `Union` in the forward means a macro op's `defn` was unioned with
    /// its sugar at construction. Autograd runs pre-saturation, so it
    /// descends into operand 0 only — the adjoint is taken once, over one
    /// member of the class.
    pub fn operands(&self, id: Val) -> Children {
        let node = &self.nodes[id.index()];
        match node.op {
            Op::Union(..) => node.children.iter().take(1).copied().collect(),
            _ => node.children.clone(),
        }
    }
}
/// The shipped autograd.
#[derive(Default, Debug, Clone)]
pub struct Reverse {
    topo: Option<Arc<Topology>>,
    custom: Option<Arc<CustomRegistry>>,
}

impl Reverse {
    /// A `Reverse` with no topology. [`Autograd::backward`] on it reports
    /// that it needs one; use [`Reverse::over`] or [`backward_into`].
    pub const fn new() -> Self {
        Self {
            topo: None,
            custom: None,
        }
    }

    /// Snapshot `graph`'s structure so the walk can descend it.
    pub fn over(graph: &EGraph) -> Self {
        Self {
            topo: Some(Arc::new(Topology::of(graph))),
            custom: None,
        }
    }
}

impl Autograd for Reverse {
    fn adjoints(&self) -> &'static [Adjoint] {
        crate::adjoints::ADJOINTS
    }

    fn backward(
        &self,
        tape: &mut dyn Tape,
        root: Val,
        seed: Val,
        wrt: &[Val],
    ) -> Result<Vec<Option<Val>>> {
        let topo = self.topo.as_deref().ok_or_else(|| {
            Error::Plan(
                "Reverse needs the primal topology: `Tape` exposes no children, \
                 so build it with Reverse::over(&graph) or call backward_into"
                    .into(),
            )
        })?;
        walk(topo, self.custom.as_deref(), tape, root, seed, wrt)
    }
}

/// Build the backward for `root` into the same graph the forward lives in,
/// and return one gradient per entry of `wrt`.
///
/// Every returned entry is `Some`: a `wrt` that receives no gradient is an
/// `Err` naming it, never a `None` the caller has to interpret. The `Option`
/// survives only because [`Autograd::backward`] is declared with it.
///
/// The caller then calls `graph.add_root(g)` for every produced gradient:
/// forward and backward are one graph with one root set, which is what makes
/// "save this activation" versus "recompute it" the extractor's
/// materialization bit rather than a checkpointing pass anybody writes.
pub fn backward_into(
    graph: &mut EGraph,
    caps: &Caps,
    root: Id,
    seed: Id,
    wrt: &[Id],
) -> Result<Vec<Option<Id>>> {
    backward_into_with(graph, caps, root, seed, wrt, &CustomRegistry::default())
}

/// [`backward_into`] with a registry of user-supplied backwards.
pub fn backward_into_with(
    graph: &mut EGraph,
    _caps: &Caps,
    root: Id,
    seed: Id,
    wrt: &[Id],
    custom: &CustomRegistry,
) -> Result<Vec<Option<Id>>> {
    let topo = Topology::of(graph);
    let mut tape = GraphTape::new(graph);
    walk(&topo, Some(custom), &mut tape, root, seed, wrt)
}

fn walk(
    topo: &Topology,
    custom: Option<&CustomRegistry>,
    tape: &mut dyn Tape,
    root: Val,
    seed: Val,
    wrt: &[Val],
) -> Result<Vec<Option<Val>>> {
    let n = topo.len();
    if root.index() >= n {
        return Err(Error::Plan(format!(
            "backward root {root} is not in the graph"
        )));
    }
    if wrt.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Reachability from the root.
    let mut reach = vec![false; n];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if reach[id.index()] {
            continue;
        }
        reach[id.index()] = true;
        for c in topo.operands(id) {
            stack.push(c);
        }
    }

    // 2. `requires_grad` is derived, not annotated: a node requires grad iff
    //    it is a `Param` leaf, it is named in `wrt`, or any operand does.
    //    Children are strictly smaller, so one ascending pass is a fixpoint.
    let mut needs = vec![false; n];
    for w in wrt {
        if w.index() < n
            && reach[w.index()]
            && !matches!(
                &topo.node(*w).op,
                Op::Logical(Logical::Leaf(
                    LeafKind::Const { .. } | LeafKind::Uniform { .. }
                ))
            )
        {
            needs[w.index()] = true;
        }
    }
    for i in 0..n {
        if !reach[i] {
            continue;
        }
        let id = Id(i as u32);
        if needs[i] || topo.is_param(id) || topo.operands(id).iter().any(|c| needs[c.index()]) {
            needs[i] = true;
        }
    }
    if !needs[root.index()] {
        // Nothing on the tape from any `wrt` to the root. That is never a
        // silent empty answer: every requested value is reported by name.
        return Err(
            first_missing(topo, &reach, &needs, wrt, &FxHashMap::default()).unwrap_or_else(|| {
                Error::Plan(format!(
                    "backward from {root} reached no requires-grad value"
                ))
            }),
        );
    }

    // 3. One pending counter per requires-grad edge. A node fires exactly
    //    once, with the fully accumulated adjoint.
    let mut pending = vec![0u32; n];
    for i in 0..n {
        if !reach[i] || !needs[i] {
            continue;
        }
        for c in topo.operands(Id(i as u32)) {
            if needs[c.index()] {
                pending[c.index()] += 1;
            }
        }
    }

    // 4. FIFO worklist, seeded at the root. FIFO plus operand-slot order is
    //    what makes the emitted node ids identical run to run.
    let mut grads: FxHashMap<Id, Val> = FxHashMap::default();
    grads.insert(root, seed);
    let mut queue: VecDeque<Id> = VecDeque::new();
    queue.push_back(root);

    while let Some(id) = queue.pop_front() {
        let grad = *grads
            .get(&id)
            .ok_or_else(|| Error::Plan(format!("node {id} fired without an adjoint")))?;
        let operands = topo.operands(id);
        let targets = adjoint_of_node(topo, custom, tape, id, grad, &operands)?;

        for (slot, child) in operands.iter().copied().enumerate() {
            if !needs[child.index()] {
                continue;
            }
            if let Some(g) = targets.get(slot).copied().flatten() {
                let merged = match grads.get(&child).copied() {
                    Some(prev) => tape.accumulate(prev, g)?,
                    None => g,
                };
                grads.insert(child, merged);
            }
            pending[child.index()] -= 1;
            if pending[child.index()] == 0 && grads.contains_key(&child) {
                queue.push_back(child);
            }
        }
    }

    // 5. Every requested value must have received a gradient, and a missing
    //    one is reported by name: a `None` cannot distinguish "not on the
    //    tape" from "a rule dropped this operand".
    if let Some(e) = first_missing(topo, &reach, &needs, wrt, &grads) {
        return Err(e);
    }

    // Every other reachable requires-grad node must have one too: a rule
    // that omits a requires-grad parent starves its whole subgraph.
    for i in 0..n {
        let id = Id(i as u32);
        if reach[i] && needs[i] && !grads.contains_key(&id) {
            return Err(Error::Plan(format!("adjoint starved node {id}")));
        }
    }

    Ok(wrt.iter().map(|v| grads.get(v).copied()).collect())
}

/// The first requested value that received no gradient, as the error naming
/// it and saying why. `None` when every entry of `wrt` has one.
fn first_missing(
    topo: &Topology,
    reach: &[bool],
    needs: &[bool],
    wrt: &[Val],
    grads: &FxHashMap<Id, Val>,
) -> Option<Error> {
    let w = *wrt.iter().find(|w| !grads.contains_key(w))?;
    Some(Error::Plan(format!(
        "no gradient for {w}: {}",
        why_no_gradient(topo, reach, needs, w)
    )))
}

/// Why `w` has no gradient, in the caller's terms.
fn why_no_gradient(topo: &Topology, reach: &[bool], needs: &[bool], w: Val) -> String {
    if w.index() >= topo.len() {
        return "it is not a value in this graph".into();
    }
    if !reach[w.index()] {
        return "it does not reach the loss — nothing on the tape connects the two, \
                which is what detach() does and what an unused tensor looks like"
            .into();
    }
    if let Op::Logical(Logical::Leaf(LeafKind::Const { .. } | LeafKind::Uniform { .. })) =
        &topo.node(w).op
    {
        return "it is a constant leaf; only Param and Buffer leaves carry a gradient".into();
    }
    if !topo.dtype(w).is_float() {
        return format!(
            "it has dtype {:?}, which is an index or a mask, not a differentiable value",
            topo.dtype(w)
        );
    }
    if !needs[w.index()] {
        return "it was not seeded as requires-grad".into();
    }
    "it is on the tape and requires grad, but no adjoint delivered one: an adjoint \
     rule on one of its consumers omitted this operand"
        .into()
}

fn adjoint_of_node(
    topo: &Topology,
    custom: Option<&CustomRegistry>,
    tape: &mut dyn Tape,
    id: Val,
    grad: Val,
    operands: &[Val],
) -> Result<Grads> {
    let node = topo.node(id);

    if let Some(entry) = custom.and_then(|c| c.get(id)) {
        return entry.invoke(tape, node, grad, operands, id);
    }

    match &node.op {
        // The sugar and its `defn` are one class; the adjoint is taken once,
        // over operand 0.
        Op::Union(..) => Ok(smallvec::smallvec![Some(grad)]),

        Op::Launch(_) => Err(Error::Plan(format!(
            "autograd is a Logical -> Logical transform and runs before saturation, \
             but {id} is already at Launch"
        ))),

        Op::Logical(l0) => match l0 {
            // Terminates. A `Param` leaf's entry in `grads` is the answer.
            Logical::Leaf(_) => Ok(Grads::new()),

            // Routes the gradient into the tuple slot it read.
            Logical::Project { .. } => Ok(smallvec::smallvec![Some(grad)]),

            // A `Dequant`'s input is a quantized leaf, which is never
            // trainable: `q_mat_mul`'s gradient goes to the activation only
            // and QAT keeps a separate f32 master.
            Logical::Dequant { .. } => Err(Error::Plan(format!(
                "{id}: quantized weights are not trainable; QAT keeps an f32 master"
            ))),

            other => {
                let tag = other.tag();
                let kind = adjoint_of(tag)
                    .map(|a| a.kind)
                    .ok_or_else(|| Error::Plan(format!("no adjoint registered for {tag:?}")))?;
                match kind {
                    AdjointKind::Analytic(f) => f(tape, node, grad, operands, id),
                    AdjointKind::Structural => structural_adjoint(tape, node, grad, operands, id),
                }
            }
        },
    }
}
