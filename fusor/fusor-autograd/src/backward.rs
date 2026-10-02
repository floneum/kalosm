//! The reverse-mode transform itself: a reverse walk over the primal with a
//! pending-children counter, dispatching each node through [`crate::ADJOINTS`]
//! and accumulating into a per-value gradient slot.
//!
use crate::adjoints::adjoint_of;
use crate::custom::CustomRegistry;
use crate::structural::structural_adjoint;
use crate::tape::GraphTape;
use fusor_ir::autograd::{AdjointKind, Grads, Tape, Val};
use fusor_ir::egraph::{EGraph, Id};
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::ir::{Children, Node, Op};
use fusor_ir::{Error, Result};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;

// Equivalent class members contribute the same gradient; visit only one.
fn operands(graph: &EGraph, id: Id) -> Children {
    let node = graph.node(id);
    match node.op {
        Op::Union(..) => node.children.iter().take(1).copied().collect(),
        _ => node.children.clone(),
    }
}

/// Append the backward graph and return a gradient for every requested value.
/// An unreachable value or a missing adjoint is an error.
pub fn backward_into(
    graph: &mut EGraph,
    root: Id,
    seed: Id,
    wrt: &[Id],
    custom: &CustomRegistry,
) -> Result<Vec<Id>> {
    // Backward only appends nodes. The original prefix remains the primal.
    let n = graph.len();
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
        for c in operands(graph, id) {
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
                &graph.node(*w).op,
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
        let parameter = graph.facts(id).dtype.is_float()
            && matches!(
                graph.node(id).op,
                Op::Logical(Logical::Leaf(
                    LeafKind::Param { .. } | LeafKind::Buffer { .. }
                ))
            );
        if needs[i] || parameter || operands(graph, id).iter().any(|c| needs[c.index()]) {
            needs[i] = true;
        }
    }
    if !needs[root.index()] {
        // Nothing on the tape from any `wrt` to the root. That is never a
        // silent empty answer: every requested value is reported by name.
        return Err(
            first_missing(graph, &reach, &needs, wrt, &FxHashMap::default()).unwrap_or_else(|| {
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
        for c in operands(graph, Id(i as u32)) {
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
        let operands = operands(graph, id);
        let node = graph.node(id).clone();
        let mut tape = GraphTape::new(graph);
        let targets = adjoint_of_node(&node, custom, &mut tape, id, grad, &operands)?;

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
    if let Some(e) = first_missing(graph, &reach, &needs, wrt, &grads) {
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

    Ok(wrt.iter().map(|v| grads[v]).collect())
}

/// The first requested value that received no gradient, as the error naming
/// it and saying why. `None` when every entry of `wrt` has one.
fn first_missing(
    graph: &EGraph,
    reach: &[bool],
    needs: &[bool],
    wrt: &[Val],
    grads: &FxHashMap<Id, Val>,
) -> Option<Error> {
    let w = *wrt.iter().find(|w| !grads.contains_key(w))?;
    Some(Error::Plan(format!(
        "no gradient for {w}: {}",
        why_no_gradient(graph, reach, needs, w)
    )))
}

/// Why `w` has no gradient, in the caller's terms.
fn why_no_gradient(graph: &EGraph, reach: &[bool], needs: &[bool], w: Val) -> String {
    if w.index() >= reach.len() {
        return "it is not a value in this graph".into();
    }
    if !reach[w.index()] {
        return "it does not reach the loss — nothing on the tape connects the two, \
                which is what detach() does and what an unused tensor looks like"
            .into();
    }
    if let Op::Logical(Logical::Leaf(LeafKind::Const { .. } | LeafKind::Uniform { .. })) =
        &graph.node(w).op
    {
        return "it is a constant leaf; only Param and Buffer leaves carry a gradient".into();
    }
    if !graph.facts(w).dtype.is_float() {
        return format!(
            "it has dtype {:?}, which is an index or a mask, not a differentiable value",
            graph.facts(w).dtype
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
    node: &Node,
    custom: &CustomRegistry,
    tape: &mut dyn Tape,
    id: Val,
    grad: Val,
    operands: &[Val],
) -> Result<Grads> {
    if let Some(entry) = custom.get(&id) {
        return entry.invoke(tape, node, grad, operands, id);
    }

    match &node.op {
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
