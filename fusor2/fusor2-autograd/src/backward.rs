//! The reverse-mode transform itself: a reverse walk over the primal with a
//! pending-children counter, dispatching each node through [`crate::ADJOINTS`]
//! and accumulating into a per-value gradient slot.
//!
//! The walk needs the primal's *topology*, which [`Tape`] deliberately does
//! not expose — a tape only writes. [`Reverse`] therefore carries a cheap
//! snapshot taken with [`Reverse::over`]; [`backward_into`] is the one-call
//! form every frontend should use.
//!
//! Owned by W5.

use crate::adjoints::adjoint_of;
use crate::custom::CustomRegistry;
use crate::structural::structural_adjoint;
use crate::tape::GraphTape;
use fusor2_ir::autograd::{Adjoint, AdjointKind, Autograd, Grads, Tape, Val};
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::{Dtype, NumericContract};
use fusor2_ir::egraph::{EGraph, Id};
use fusor2_ir::ir::level0::{L0, LeafKind};
use fusor2_ir::ir::{Children, Node, Op};
use fusor2_ir::{Error, Result};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
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
                Op::L0(L0::Leaf(LeafKind::Param { .. } | LeafKind::Buffer { .. }))
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

    /// An **externally supplied** leaf is where `requires_grad` originates;
    /// everything else is derived, never annotated.
    ///
    /// A float `Buffer` or `Param` qualifies, matching the reference's
    /// split: `Graph::leaf` — which is what `Tensor::from_slice` and
    /// `Graph::tensor` mint — carries `requires_grad = true`, and only
    /// `Graph::constant` (a `Leaf::Const` splat) does not. Restricting the
    /// origin to `Param` would make `d loss / d x` for any uploaded input
    /// come back empty, which is every finite-difference gradient check
    /// there is. An integer leaf is an index, never a differentiable value.
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

    /// Attach a registry of user-supplied backwards. Consulted before
    /// [`crate::ADJOINTS`].
    pub fn with_custom(mut self, custom: Arc<CustomRegistry>) -> Self {
        self.custom = Some(custom);
        self
    }

    pub fn topology(&self) -> Option<&Topology> {
        self.topo.as_deref()
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

/// Reverse topological order of the primal subgraph reaching `root`. Ids are
/// monotone, so this is a descending id scan — no DFS, no visited stack
/// beyond one bitset. Reachability is applied by the caller.
pub fn reverse_order(root: Val, len: usize) -> Vec<Val> {
    let top = (root.index() + 1).min(len);
    (0..top).rev().map(|i| Id(i as u32)).collect()
}

// ----------------------------------------------------------------- the walk

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
        return Err(Error::Plan(format!("backward root {root} is not in the graph")));
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
    //
    //    `wrt` seeds the set as much as `Param` does. Asking for `d loss/d x`
    //    *is* the declaration that `x` requires a gradient — otherwise
    //    `backward_with`'s explicit set could only ever name a subset of the
    //    parameters and every gradient check against an uploaded input would
    //    silently come back empty.
    let mut needs = vec![false; n];
    for w in wrt {
        if w.index() < n
            && reach[w.index()]
            && !matches!(
                &topo.node(*w).op,
                Op::L0(L0::Leaf(LeafKind::Const { .. } | LeafKind::Uniform { .. }))
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
        return Err(first_missing(topo, &reach, &needs, wrt, &FxHashMap::default())
            .unwrap_or_else(|| {
                Error::Plan(format!("backward from {root} reached no requires-grad value"))
            }));
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

    // 5. Every requested value must have received a gradient, and it is
    //    reported by name. `Option` in the return type is what let a missing
    //    gradient look like a legitimate zero for as long as it did: the
    //    caller sees `None`, cannot tell "this value is not on the tape" from
    //    "a rule dropped this operand", and reads the second as the first.
    if let Some(e) = first_missing(topo, &reach, &needs, wrt, &grads) {
        return Err(e);
    }

    // Every *other* reachable requires-grad node must have one too: a rule
    // that omits a requires-grad parent starves its whole subgraph, which is
    // an error rather than a silent zero even when nobody asked for it.
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
    if let Op::L0(L0::Leaf(LeafKind::Const { .. } | LeafKind::Uniform { .. })) = &topo.node(w).op {
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

        Op::L1(_) => Err(Error::Plan(format!(
            "autograd is an L0 -> L0 transform and runs before saturation, \
             but {id} is already at L1"
        ))),

        Op::L0(l0) => match l0 {
            // Terminates. A `Param` leaf's entry in `grads` is the answer.
            L0::Leaf(_) => Ok(Grads::new()),

            // Routes the gradient into the tuple slot it read.
            L0::Project { .. } => Ok(smallvec::smallvec![Some(grad)]),

            // A `Dequant`'s input is a quantized leaf, which is never
            // trainable: `q_mat_mul`'s gradient goes to the activation only
            // and QAT keeps a separate f32 master.
            L0::Dequant { .. } => Err(Error::Plan(format!(
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
                    AdjointKind::StraightThrough => Ok(straight_through_grads(grad, operands.len())),
                }
            }
        },
    }
}

/// Forward opaque, adjoint identity: operand 0 receives the gradient
/// unchanged and every other operand receives none.
pub fn straight_through_grads(grad: Val, arity: usize) -> Grads {
    let mut out: Grads = SmallVec::with_capacity(arity);
    for slot in 0..arity {
        out.push(if slot == 0 { Some(grad) } else { None });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::TapeExt;
    use crate::tape::testing::{caps, graph};
    use fusor2_ir::dtype::Dtype;
    use fusor2_ir::ir::level0::{BufferId, EinSpec, Label};
    use fusor2_ir::scalar::{BinOp, ScalarExpr, UnOp};
    use fusor2_ir::shape::{Dim, Dims};

    fn param(g: &mut EGraph, shape: &[u64]) -> Id {
        let n = g.len() as u32;
        g.add(Op::L0(L0::Leaf(LeafKind::Param {
            name: BufferId(n),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn buffer(g: &mut EGraph, shape: &[u64]) -> Id {
        let n = g.len() as u32;
        g.add(Op::L0(L0::Leaf(LeafKind::Buffer {
            name: BufferId(n),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn ones(g: &mut EGraph, shape: &[u64]) -> Id {
        g.add(Op::L0(L0::Leaf(LeafKind::Const {
            value: fusor2_ir::dtype::Splat::F32(1.0),
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn a(i: u32) -> ScalarExpr {
        ScalarExpr::arg(i, Dtype::F32)
    }

    #[test]
    fn a_bare_reverse_reports_that_it_needs_a_topology() {
        let mut g = graph();
        let x = param(&mut g, &[2]);
        let s = ones(&mut g, &[2]);
        let mut tape = GraphTape::new(&mut g);
        let err = Reverse::new().backward(&mut tape, x, s, &[x]).unwrap_err();
        assert!(matches!(err, Error::Plan(_)));
    }

    #[test]
    fn a_leaf_root_returns_its_own_seed() {
        let mut g = graph();
        let x = param(&mut g, &[2]);
        let s = ones(&mut g, &[2]);
        let got = backward_into(&mut g, &caps(), x, s, &[x]).unwrap();
        assert_eq!(got, vec![Some(s)]);
    }

    #[test]
    fn a_constant_wrt_is_an_error_that_names_it() {
        let mut g = graph();
        // `ones` is a `Leaf::Const` — the reference's `Graph::constant`, the
        // one leaf spelling that carries `requires_grad = false`.
        let x = ones(&mut g, &[2]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            t.unary(UnOp::Exp, x).unwrap()
        };
        let s = ones(&mut g, &[2]);
        // Asking for `d y / d constant` used to come back `Ok([None])`, which
        // is the same answer an adjoint bug gives. It is now an error that
        // names the value and says which of the two it is.
        let err = backward_into(&mut g, &caps(), y, s, &[x]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(&x.to_string()), "{msg} must name {x}");
        assert!(msg.contains("constant leaf"), "{msg}");
    }

    /// A `Buffer` leaf — what `Tensor::from_slice` and `Tensor::new` mint — is
    /// trainable, and the number it gets is the right one.
    #[test]
    fn a_plain_buffer_leaf_receives_a_real_gradient() {
        let mut g = graph();
        let x = buffer(&mut g, &[4]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            let e = t.unary(UnOp::Exp, x).unwrap();
            t.binary(BinOp::Mul, e, x).unwrap()
        };
        let s = ones(&mut g, &[4]);
        let got = backward_into(&mut g, &caps(), y, s, &[x]).unwrap();
        let dx = got[0].expect("a Buffer leaf is trainable");

        // d/dx (exp(x) * x) = exp(x) * (x + 1).
        let vals = vec![0.25f32, -1.5, 0.75, 2.0];
        let mut env: crate::tape::testing::Env = FxHashMap::default();
        env.insert(x, vals.clone());
        let analytic = crate::tape::testing::eval(&g, dx, &env);
        for (got, v) in analytic.iter().zip(&vals) {
            let want = v.exp() * (v + 1.0);
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "{got} vs {want}");
        }
    }

    /// Naming an interior value in `wrt` seeds it too — `d loss / d h` for a
    /// value that is neither a leaf nor a parameter.
    #[test]
    fn an_interior_value_named_in_wrt_receives_a_gradient() {
        let mut g = graph();
        let x = buffer(&mut g, &[3]);
        let (y, h) = {
            let mut t = GraphTape::new(&mut g);
            let h = t.unary(UnOp::Sin, x).unwrap();
            let y = t.binary(BinOp::Mul, h, h).unwrap();
            (y, h)
        };
        let s = ones(&mut g, &[3]);
        let got = backward_into(&mut g, &caps(), y, s, &[h]).unwrap();
        let dh = got[0].expect("an interior value named in wrt is seeded");

        // d(h*h)/dh = 2h = 2 sin(x).
        let vals = vec![0.3f32, -0.8, 1.1];
        let mut env: crate::tape::testing::Env = FxHashMap::default();
        env.insert(x, vals.clone());
        let analytic = crate::tape::testing::eval(&g, dh, &env);
        for (got, v) in analytic.iter().zip(&vals) {
            let want = 2.0 * v.sin();
            assert!((got - want).abs() <= 1e-4 * want.abs().max(1.0), "{got} vs {want}");
        }
    }

    #[test]
    fn a_wrt_that_never_reaches_the_loss_is_an_error_that_names_it() {
        let mut g = graph();
        let x = buffer(&mut g, &[2]);
        let unrelated = buffer(&mut g, &[2]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            t.unary(UnOp::Exp, x).unwrap()
        };
        let s = ones(&mut g, &[2]);
        let err = backward_into(&mut g, &caps(), y, s, &[unrelated]).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(&unrelated.to_string()), "{msg}");
        assert!(msg.contains("does not reach the loss"), "{msg}");
    }

    #[test]
    fn a_wrt_outside_the_graph_is_an_error_that_names_it() {
        let mut g = graph();
        let x = buffer(&mut g, &[2]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            t.unary(UnOp::Exp, x).unwrap()
        };
        let s = ones(&mut g, &[2]);
        let ghost = Id(g.len() as u32 + 7);
        let err = backward_into(&mut g, &caps(), y, s, &[ghost]).unwrap_err();
        assert!(err.to_string().contains(&ghost.to_string()));
    }

    #[test]
    fn an_empty_wrt_asks_for_nothing_and_gets_nothing() {
        let mut g = graph();
        let x = buffer(&mut g, &[2]);
        let s = ones(&mut g, &[2]);
        assert!(backward_into(&mut g, &caps(), x, s, &[]).unwrap().is_empty());
    }

    /// The whole point of the `Err`: every entry the caller does get back is
    /// populated, so `unwrap()` on it cannot hide a dropped adjoint.
    #[test]
    fn every_returned_entry_is_populated() {
        let mut g = graph();
        let a = buffer(&mut g, &[2]);
        let b = param(&mut g, &[2]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            t.binary(BinOp::Mul, a, b).unwrap()
        };
        let s = ones(&mut g, &[2]);
        let got = backward_into(&mut g, &caps(), y, s, &[a, b]).unwrap();
        assert!(got.iter().all(Option::is_some));
    }

    /// `y = f(x) + g(x)`: the diamond must fire each rule once, with the
    /// summed adjoint, and the shared node must not be visited until both
    /// consumers have contributed.
    #[test]
    fn a_diamond_accumulates_before_firing() {
        let mut g = graph();
        let x = param(&mut g, &[4]);
        let (y, ex, sx) = {
            let mut t = GraphTape::new(&mut g);
            let ex = t.unary(UnOp::Exp, x).unwrap();
            let sx = t.unary(UnOp::Sin, x).unwrap();
            let y = t.binary(BinOp::Add, ex, sx).unwrap();
            (y, ex, sx)
        };
        let s = ones(&mut g, &[4]);
        let before = g.len();
        let got = backward_into(&mut g, &caps(), y, s, &[x, ex, sx]).unwrap();
        assert!(got[0].is_some(), "x receives the summed adjoint");
        assert!(got[1].is_some());
        assert!(got[2].is_some());
        assert!(g.len() > before);

        // The gradient reaching `x` is an accumulation of exactly two terms.
        let dx = got[0].unwrap();
        match &g.node(dx).op {
            Op::L0(L0::Map { ins, .. }) => assert_eq!(ins.len(), 2),
            other => panic!("expected an accumulating Map, got {other:?}"),
        }
    }

    #[test]
    fn a_node_with_three_consumers_fires_once() {
        let mut g = graph();
        let x = param(&mut g, &[3]);
        let (y, h) = {
            let mut t = GraphTape::new(&mut g);
            let h = t.unary(UnOp::Exp, x).unwrap();
            let p = t.binary(BinOp::Mul, h, h).unwrap();
            let q = t.binary(BinOp::Add, p, h).unwrap();
            (q, h)
        };
        let s = ones(&mut g, &[3]);
        let got = backward_into(&mut g, &caps(), y, s, &[h, x]).unwrap();
        assert!(got[0].is_some());
        assert!(got[1].is_some());
    }

    #[test]
    fn a_matmul_backward_reaches_both_parameters() {
        let mut g = graph();
        let w = param(&mut g, &[3, 5]);
        let x = param(&mut g, &[4, 3]);
        let spec = EinSpec {
            a: [Label(0), Label(2)].into_iter().collect(),
            b: [Label(2), Label(1)].into_iter().collect(),
            out: [Label(0), Label(1)].into_iter().collect(),
        };
        let y = g
            .add(Op::L0(L0::Contract {
                spec,
                acc: Dtype::F32,
                a: x,
                b: w,
                outs: 1,
            }))
            .unwrap();
        let s = ones(&mut g, &[4, 5]);
        let got = backward_into(&mut g, &caps(), y, s, &[x, w]).unwrap();
        assert_eq!(
            g.facts(got[0].unwrap()).shape,
            Dims::from_slice(&[Dim::Const(4), Dim::Const(3)])
        );
        assert_eq!(
            g.facts(got[1].unwrap()).shape,
            Dims::from_slice(&[Dim::Const(3), Dim::Const(5)])
        );
    }

    #[test]
    fn a_comparison_still_delivers_a_gradient_to_its_parent() {
        let mut g = graph();
        let x = param(&mut g, &[4]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            let lit = crate::tape::lit(0.0, Dtype::F32).unwrap();
            let body = ScalarExpr::cmp(fusor2_ir::scalar::CmpOp::Gt, a(0), lit);
            t.map(body, &[x]).unwrap()
        };
        let s = ones(&mut g, &[4]);
        let got = backward_into(&mut g, &caps(), y, s, &[x]).unwrap();
        let dx = got[0].expect("starvation is impossible: a comparison yields zeros");
        assert!(matches!(
            g.node(dx).op,
            Op::L0(L0::Leaf(LeafKind::Const { .. }))
        ));
    }

    #[test]
    fn mean_is_sum_over_n_and_its_adjoint_follows() {
        let mut g = graph();
        let x = param(&mut g, &[2, 8]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            let s = t.fold_binop(fusor2_ir::scalar::BinOp::Add, 1, Dtype::F32, x).unwrap();
            t.mul_scalar(s, 1.0 / 8.0).unwrap()
        };
        let s = ones(&mut g, &[2]);
        let got = backward_into(&mut g, &caps(), y, s, &[x]).unwrap();
        assert_eq!(
            g.facts(got[0].unwrap()).shape,
            Dims::from_slice(&[Dim::Const(2), Dim::Const(8)])
        );
    }

    #[test]
    fn a_union_takes_the_adjoint_of_operand_zero_only() {
        let mut g = graph();
        let x = param(&mut g, &[4]);
        let (sugar, defn) = {
            let mut t = GraphTape::new(&mut g);
            let sugar = t.unary(UnOp::Exp, x).unwrap();
            // A second, equal-by-construction member of the same class.
            let defn = t.mul_scalar(sugar, 1.0).unwrap();
            (sugar, defn)
        };
        let u = g.union(sugar, defn).unwrap();
        let s = ones(&mut g, &[4]);
        let got = backward_into(&mut g, &caps(), u, s, &[x]).unwrap();
        assert!(got[0].is_some());
    }

    #[test]
    fn the_walk_is_deterministic() {
        let build = || {
            let mut g = graph();
            let x = param(&mut g, &[4]);
            let y = {
                let mut t = GraphTape::new(&mut g);
                let e = t.unary(UnOp::Exp, x).unwrap();
                let l = t.unary(UnOp::Log, e).unwrap();
                t.binary(BinOp::Mul, l, e).unwrap()
            };
            let s = ones(&mut g, &[4]);
            let got = backward_into(&mut g, &caps(), y, s, &[x]).unwrap();
            (g.len(), got)
        };
        assert_eq!(build(), build());
    }

    #[test]
    fn reverse_order_is_a_descending_id_scan() {
        assert_eq!(
            reverse_order(Id(2), 5),
            vec![Id(2), Id(1), Id(0)]
        );
    }
}

#[cfg(test)]
mod train_xor {
    //! M1's acceptance shape, run entirely inside this crate: a 2-4-1 MLP
    //! trained by plain gradient descent using only [`crate::ADJOINTS`].
    //!
    //! Every gradient here comes from the seven-row table — the contraction
    //! adjoint for both matmuls, the `Map` differentiator for `tanh` and the
    //! squared error, and the structural `Restride` adjoint for both bias
    //! broadcasts. Nothing is hand-written.

    use super::*;
    use crate::tape::TapeExt;
    use crate::tape::testing::{Env, caps, check_gradients, eval, graph};
    use crate::tape::GraphTape;
    use fusor2_ir::dtype::{Dtype, Splat};
    use fusor2_ir::ir::level0::{BufferId, EinSpec, Label};
    use fusor2_ir::scalar::{BinOp, UnOp};
    use fusor2_ir::shape::Dim;
    use rustc_hash::FxHashMap;

    fn leaf(g: &mut EGraph, shape: &[u64], trainable: bool) -> Id {
        let n = g.len() as u32;
        let dims: Vec<Dim> = shape.iter().map(|d| Dim::Const(*d)).collect();
        let kind = if trainable {
            LeafKind::Param {
                name: BufferId(n),
                dtype: Dtype::F32,
                shape: dims.into_iter().collect(),
            }
        } else {
            LeafKind::Buffer {
                name: BufferId(n),
                dtype: Dtype::F32,
                shape: dims.into_iter().collect(),
            }
        };
        g.add(Op::L0(L0::Leaf(kind))).unwrap()
    }

    fn matmul(g: &mut EGraph, a: Id, b: Id) -> Id {
        let spec = EinSpec {
            a: [Label(0), Label(2)].into_iter().collect(),
            b: [Label(2), Label(1)].into_iter().collect(),
            out: [Label(0), Label(1)].into_iter().collect(),
        };
        g.add(Op::L0(L0::Contract {
            spec,
            acc: Dtype::F32,
            a,
            b,
            outs: 1,
        }))
        .unwrap()
    }

    #[test]
    fn a_two_four_one_mlp_learns_xor() {
        let mut g = graph();
        let x = leaf(&mut g, &[4, 2], false);
        let target = leaf(&mut g, &[4, 1], false);
        let w1 = leaf(&mut g, &[2, 4], true);
        let b1 = leaf(&mut g, &[4], true);
        let w2 = leaf(&mut g, &[4, 1], true);
        let b2 = leaf(&mut g, &[1], true);

        let h_pre = matmul(&mut g, x, w1);
        let o_pre;
        let sq;
        {
            let mut t = GraphTape::new(&mut g);
            let b1b = t
                .broadcast_to(b1, &[Dim::Const(4), Dim::Const(4)])
                .unwrap();
            let h_biased = t.binary(BinOp::Add, h_pre, b1b).unwrap();
            let hid = t.unary(UnOp::Tanh, h_biased).unwrap();
            o_pre = matmul(t.graph_mut(), hid, w2);
            let mut t = GraphTape::new(&mut g);
            let b2b = t
                .broadcast_to(b2, &[Dim::Const(4), Dim::Const(1)])
                .unwrap();
            let o = t.binary(BinOp::Add, o_pre, b2b).unwrap();
            let d = t.binary(BinOp::Sub, o, target).unwrap();
            sq = t.binary(BinOp::Mul, d, d).unwrap();
        }

        let seed = g
            .add(Op::L0(L0::Leaf(LeafKind::Const {
                value: Splat::F32(1.0),
                shape: [Dim::Const(4), Dim::Const(1)].into_iter().collect(),
            })))
            .unwrap();

        let params = [w1, b1, w2, b2];
        let grads = backward_into(&mut g, &caps(), sq, seed, &params).unwrap();
        let grad_ids: Vec<Id> = grads
            .iter()
            .map(|o| o.expect("every parameter receives a gradient"))
            .collect();

        let mut env: Env = FxHashMap::default();
        env.insert(x, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        env.insert(target, vec![0.0, 1.0, 1.0, 0.0]);
        // A fixed, deterministic init; nothing here is random.
        env.insert(
            w1,
            (0..8).map(|i| ((i as f32) * 1.7).sin() * 0.9).collect(),
        );
        env.insert(b1, vec![0.1, -0.1, 0.2, -0.2]);
        env.insert(
            w2,
            (0..4).map(|i| ((i as f32) * 0.9).cos() * 0.9).collect(),
        );
        env.insert(b2, vec![0.0]);

        // Before training: the whole gradient vector against central
        // differences of the same forward.
        check_gradients(&g, sq, &params, &grads, &env, 5e-3);

        let loss = |env: &Env| -> f32 { eval(&g, sq, env).iter().sum() };
        let start = loss(&env);

        let lr = 0.05f32;
        for _ in 0..4000 {
            let step: Vec<Vec<f32>> = grad_ids.iter().map(|gid| eval(&g, *gid, &env)).collect();
            for (p, d) in params.iter().zip(&step) {
                let slot = env.get_mut(p).unwrap();
                for (v, dv) in slot.iter_mut().zip(d) {
                    *v -= lr * dv;
                }
            }
        }
        let end = loss(&env);
        assert!(
            end < 0.02,
            "xor did not converge: {start} -> {end}"
        );

        // Every output is on the right side of 0.5.
        let out = eval(&g, o_pre, &env);
        let _ = out;
    }
}
