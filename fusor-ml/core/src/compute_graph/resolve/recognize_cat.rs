//! Recognize composed slice-assign chains (`Tensor::cat`, sequential
//! `slice_assign` calls) while the graph is still in canonical form — before
//! view folding smears the narrow offsets into index arithmetic — and rewrite
//! each chain into a single elementwise kernel over the destination index
//! space.
//!
//! Every branch expression lifts into its region's select arm with chunk
//! coordinates rewritten to destination coordinates. Reads through `narrow`
//! views compose at the integer level (`AffineIndex`), so an aligned
//! split + op + cat cancels exactly: `(c - slice_start) + narrow_start = c`.
//! When all arms end up structurally identical and the slices tile the
//! destination, the selects disappear entirely and the chain becomes the op
//! applied to the larger tensor.

use std::ops::Range;

use crate::slice_assign::{slice_assign_expression, slice_region_condition};
use crate::view::{AffineIndex, affine_dim_indices};

use super::*;

/// One matched slice-assign link: an `Elementwise` node whose expression is
/// exactly `slice_assign_expression(slices)` over `[destination, value]`.
struct AssignLink {
    exec: ExecutionNodeIndex,
    inner: NodeIndex,
    destination: NodeIndex,
    value: NodeIndex,
    slices: Box<[Range<usize>]>,
    shape: Box<[usize]>,
    datatype: DataTypeEnum,
}

/// Inputs of the rewritten kernel, deduplicated eagerly so that identical
/// branches produce identical arm expressions (slot-for-slot), which is what
/// the equal-arms collapse compares.
#[derive(Default)]
struct LiftState {
    inputs: Vec<NodeIndex>,
    slots: FxHashMap<NodeIndex, usize>,
    /// Whether any branch expression was inlined or any view was folded —
    /// when nothing lifts, the rewrite would only replicate what n-ary
    /// fusion already does, so the chain is left alone.
    lifted: bool,
}

impl LiftState {
    fn slot(&mut self, inner: NodeIndex) -> usize {
        if let Some(&slot) = self.slots.get(&inner) {
            return slot;
        }
        let slot = self.inputs.len();
        self.inputs.push(inner);
        self.slots.insert(inner, slot);
        slot
    }
}

/// How a branch input slot is rewritten during chunk-space composition.
enum Rep {
    /// Producer expression inlined at element-wise accesses.
    Inline(NaryExpr),
    /// Opaque input, remapped to a slot in the combined input list.
    Slot(usize),
}

fn apply_reps(expr: &NaryExpr, reps: &[Rep]) -> NaryExpr {
    match expr {
        NaryExpr::Op { children, function } => NaryExpr::Op {
            children: children.iter().map(|c| apply_reps(c, reps)).collect(),
            function: function.clone(),
        },
        NaryExpr::IndexedInput { input_idx, indices } => {
            let indices: Vec<NaryExpr> = indices.iter().map(|c| apply_reps(c, reps)).collect();
            match &reps[*input_idx] {
                // Inline replacements are only built for slots accessed
                // element-wise, where the whole load is the producer's value.
                Rep::Inline(expr) => expr.clone(),
                Rep::Slot(slot) => NaryExpr::IndexedInput {
                    input_idx: *slot,
                    indices,
                },
            }
        }
        NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
        NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
    }
}

/// Recover the slice ranges from a candidate slice-assign expression, then
/// verify the match by regenerating the canonical expression and comparing.
pub(super) fn match_slice_assign(nary: &ElementwiseOperation) -> Option<Box<[Range<usize>]>> {
    if nary.inputs.len() != 2 {
        return None;
    }
    let rank = nary.shape.len();
    if rank == 0 {
        return None;
    }
    let NaryExpr::Op { children, function } = &nary.expression else {
        return None;
    };
    if !matches!(function.op, NaryOp::Select) || children.len() != 3 {
        return None;
    }
    let mut starts = vec![None; rank];
    let mut ends = vec![None; rank];
    if !collect_region_bounds(&children[0], &mut starts, &mut ends) {
        return None;
    }
    let slices: Box<[Range<usize>]> = starts
        .iter()
        .zip(&ends)
        .map(|(&start, &end)| Some(start? as usize..end? as usize))
        .collect::<Option<_>>()?;
    (slice_assign_expression(&slices, nary.output_datatype) == nary.expression).then_some(slices)
}

fn collect_region_bounds(
    expr: &NaryExpr,
    starts: &mut [Option<u32>],
    ends: &mut [Option<u32>],
) -> bool {
    match expr {
        NaryExpr::Scalar(NaryScalar::U32(1)) => true,
        NaryExpr::Op { children, function } => match function.op {
            NaryOp::Mul if children.len() == 2 => {
                collect_region_bounds(&children[0], starts, ends)
                    && collect_region_bounds(&children[1], starts, ends)
            }
            NaryOp::GreaterEqualConst(NaryScalar::U32(value)) if children.len() == 1 => {
                let NaryExpr::DimIndex(dim) = children[0] else {
                    return false;
                };
                match starts.get_mut(dim) {
                    Some(slot) => {
                        *slot = Some(value);
                        true
                    }
                    None => false,
                }
            }
            NaryOp::LessConst(NaryScalar::U32(value)) if children.len() == 1 => {
                let NaryExpr::DimIndex(dim) = children[0] else {
                    return false;
                };
                match ends.get_mut(dim) {
                    Some(slot) => {
                        *slot = Some(value);
                        true
                    }
                    None => false,
                }
            }
            _ => false,
        },
        _ => false,
    }
}

/// Whether the chain's slices cover every destination coordinate: all dims
/// full except at most one, whose ranges merge to the full extent.
fn slices_tile(out_shape: &[usize], chain: &[&AssignLink]) -> bool {
    let mut partial_dims: Vec<usize> = Vec::new();
    for (dim, &extent) in out_shape.iter().enumerate() {
        if chain.iter().any(|link| link.slices[dim] != (0..extent)) {
            partial_dims.push(dim);
        }
    }
    match partial_dims.as_slice() {
        [] => true,
        [dim] => {
            let mut ranges: Vec<Range<usize>> =
                chain.iter().map(|link| link.slices[*dim].clone()).collect();
            ranges.sort_by_key(|range| range.start);
            let mut covered = 0;
            for range in ranges {
                if range.start > covered {
                    return false;
                }
                covered = covered.max(range.end);
            }
            covered >= out_shape[*dim]
        }
        _ => false,
    }
}

impl Resolver {
    pub(super) fn recognize_assign_chains(&mut self, graph: &mut ComputeGraphInner) {
        let mut links: FxHashMap<NodeIndex, AssignLink> = FxHashMap::default();
        for exec in self.execution_graph.node_indices() {
            let node = &self.execution_graph[exec];
            let ExecutionVariant::Elementwise(nary) = &node.variant else {
                continue;
            };
            let Some(slices) = match_slice_assign(nary) else {
                continue;
            };
            links.insert(
                node.inner_idx,
                AssignLink {
                    exec,
                    inner: node.inner_idx,
                    destination: nary.inputs[0],
                    value: nary.inputs[1],
                    slices,
                    shape: nary.shape.clone(),
                    datatype: nary.output_datatype,
                },
            );
        }
        if links.is_empty() {
            return;
        }

        // A link is interior when its sole consumer is the next link of the
        // same chain, reading it as the destination. Everything else is a
        // chain tail (its consumers are ordinary readers of the cat result).
        let tails: Vec<NodeIndex> = links
            .values()
            .filter(|link| !self.is_interior_link(&links, link))
            .map(|link| link.inner)
            .collect();
        for tail in tails {
            self.rewrite_assign_chain(graph, &links, tail);
        }
    }

    fn is_interior_link(
        &self,
        links: &FxHashMap<NodeIndex, AssignLink>,
        link: &AssignLink,
    ) -> bool {
        let mut consumers = self
            .execution_graph
            .neighbors_directed(link.exec, petgraph::Direction::Outgoing);
        let (Some(consumer), None) = (consumers.next(), consumers.next()) else {
            return false;
        };
        let consumer = &self.execution_graph[consumer];
        links.get(&consumer.inner_idx).is_some_and(|next| {
            next.destination == link.inner
                && next.shape == link.shape
                && next.datatype == link.datatype
        })
    }

    fn rewrite_assign_chain(
        &mut self,
        graph: &mut ComputeGraphInner,
        links: &FxHashMap<NodeIndex, AssignLink>,
        tail_inner: NodeIndex,
    ) {
        let tail = &links[&tail_inner];
        if !self.execution_graph.contains_node(tail.exec) {
            return;
        }

        // Walk destinations back to the chain base (base → tail order after
        // the reverse).
        let mut chain: Vec<&AssignLink> = vec![tail];
        loop {
            let cur = chain.last().unwrap();
            let Some(prev) = links.get(&cur.destination) else {
                break;
            };
            if !self.execution_graph.contains_node(prev.exec)
                || prev.shape != tail.shape
                || prev.datatype != tail.datatype
                || self
                    .execution_graph
                    .neighbors_directed(prev.exec, petgraph::Direction::Outgoing)
                    .count()
                    != 1
            {
                break;
            }
            chain.push(prev);
        }
        chain.reverse();
        let base_inner = chain[0].destination;

        let out_shape = tail.shape.clone();
        let rank = out_shape.len();
        let mut state = LiftState::default();
        let base_slot = state.slot(base_inner);
        let mut arms = Vec::with_capacity(chain.len());
        for link in &chain {
            let condition = slice_region_condition(&link.slices);
            let arm = self.lift_branch(graph, link, &condition, &out_shape, &mut state);
            arms.push((condition, arm));
        }
        if !state.lifted {
            return;
        }

        let collapsed =
            arms.windows(2).all(|pair| pair[0].1 == pair[1].1) && slices_tile(&out_shape, &chain);
        let expression = if collapsed {
            arms.swap_remove(0).1
        } else {
            let mut expression = NaryExpr::input(base_slot, rank);
            for (condition, arm) in arms {
                expression =
                    NaryExpr::select(condition, arm, expression, DataTypeEnum::U32, tail.datatype);
            }
            expression
        };

        let (final_inputs, final_expression) = Self::deduplicate_inputs(state.inputs, expression);
        if final_inputs.len() > graph.device().nary_direct_input_binding_budget() {
            return;
        }

        let new_nary = ElementwiseOperation {
            inputs: final_inputs.clone(),
            expression: final_expression,
            shape: out_shape,
            output_datatype: tail.datatype,
        };
        let tail_exec = tail.exec;
        self.execution_graph[tail_exec].variant = ExecutionVariant::Elementwise(new_nary);

        let incoming: Vec<ExecutionNodeIndex> = self
            .execution_graph
            .neighbors_directed(tail_exec, petgraph::Direction::Incoming)
            .collect();
        for &source in &incoming {
            if let Some(edge) = self.execution_graph.find_edge(source, tail_exec) {
                self.execution_graph.remove_edge(edge);
            }
        }
        for &input in &final_inputs {
            if let Some(exec) = self.get_input_node_in_exec_graph(input)
                && self.execution_graph.contains_node(exec)
                && self.execution_graph.find_edge(exec, tail_exec).is_none()
            {
                self.execution_graph.add_edge(exec, tail_exec, ());
            }
        }
        self.add_physical_dependencies(graph, tail_exec, &final_inputs);
        for source in incoming {
            self.remove_node_if_dead(source);
        }
    }

    /// Build one region's arm: compose the branch's elementwise cluster in
    /// chunk coordinate space, then rewrite it into destination coordinates.
    fn lift_branch(
        &self,
        graph: &ComputeGraphInner,
        link: &AssignLink,
        condition: &NaryExpr,
        out_shape: &[usize],
        state: &mut LiftState,
    ) -> NaryExpr {
        let chunk_shape: Box<[usize]> = link
            .slices
            .iter()
            .map(|slice| slice.end - slice.start)
            .collect();
        let chunk_expr = self.compose_chunk_expr(graph, link.value, &chunk_shape, state);
        self.lift_expr(&chunk_expr, link, condition, out_shape, state, false)
    }

    /// Inline the branch's elementwise producers within the shared chunk
    /// index space. Producers stay opaque (a plain load) when they are
    /// cached, shared, shaped differently, accessed with custom indices, or
    /// not elementwise.
    fn compose_chunk_expr(
        &self,
        graph: &ComputeGraphInner,
        inner: NodeIndex,
        chunk_shape: &[usize],
        state: &mut LiftState,
    ) -> NaryExpr {
        let rank = chunk_shape.len();
        let inlinable = !self.check_cached(graph, inner)
            && self
                .get_input_node_in_exec_graph(inner)
                .filter(|&exec| self.execution_graph.contains_node(exec))
                .is_some_and(|exec| {
                    matches!(
                        &self.execution_graph[exec].variant,
                        ExecutionVariant::Elementwise(nary) if *nary.shape == *chunk_shape
                    ) && self
                        .execution_graph
                        .neighbors_directed(exec, petgraph::Direction::Outgoing)
                        .count()
                        == 1
                });
        if !inlinable {
            return NaryExpr::input(state.slot(inner), rank);
        }
        let exec = self.get_input_node_in_exec_graph(inner).unwrap();
        let ExecutionVariant::Elementwise(nary) = self.execution_graph[exec].variant.clone() else {
            unreachable!("inlinable check matched an elementwise variant");
        };

        let reps: Vec<Rep> = nary
            .inputs
            .iter()
            .enumerate()
            .map(|(slot, &input)| {
                if nary.expression.uses_custom_indexing_for_input(slot) {
                    Rep::Slot(state.slot(input))
                } else {
                    Rep::Inline(self.compose_chunk_expr(graph, input, chunk_shape, state))
                }
            })
            .collect();
        state.lifted = true;
        apply_reps(&nary.expression, &reps)
    }

    /// Rewrite a chunk-space expression into destination coordinates.
    /// `DimIndex(d)` in value position becomes the unguarded shifted
    /// coordinate (wrapping arithmetic in dead lanes is discarded by the
    /// region select); in index position it becomes the guarded form so
    /// every load stays in bounds on both sides of the select. Element-wise
    /// loads through affine views fold at the integer level when the shifted
    /// map stays in bounds for every destination coordinate — that is where
    /// aligned narrow offsets cancel.
    fn lift_expr(
        &self,
        expr: &NaryExpr,
        link: &AssignLink,
        condition: &NaryExpr,
        out_shape: &[usize],
        state: &mut LiftState,
        index_position: bool,
    ) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|c| self.lift_expr(c, link, condition, out_shape, state, index_position))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => {
                if NaryExpr::is_elementwise_indices(indices)
                    && let Some(folded) =
                        self.try_fold_shifted_view(state.inputs[*input_idx], link, out_shape, state)
                {
                    state.lifted = true;
                    return folded;
                }
                NaryExpr::IndexedInput {
                    input_idx: *input_idx,
                    indices: indices
                        .iter()
                        .map(|c| self.lift_expr(c, link, condition, out_shape, state, true))
                        .collect(),
                }
            }
            NaryExpr::DimIndex(dim) => {
                if index_position {
                    self.guarded_coord(*dim, link, condition, out_shape)
                } else {
                    Self::unguarded_coord(*dim, link)
                }
            }
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        }
    }

    /// The chunk coordinate `c_d - slice_start` as a raw value. Out-of-region
    /// lanes wrap, which only feeds dead select arms.
    fn unguarded_coord(dim: usize, link: &AssignLink) -> NaryExpr {
        let start = link.slices[dim].start;
        if start == 0 {
            return NaryExpr::DimIndex(dim);
        }
        NaryExpr::unary_op(
            NaryExpr::DimIndex(dim),
            "slice_offset",
            NaryOp::SubConst(NaryScalar::U32(start as u32)),
            DataTypeEnum::U32,
            DataTypeEnum::U32,
        )
    }

    /// The chunk coordinate clamped to 0 outside the region, so loads through
    /// it stay in bounds even in dead lanes.
    fn guarded_coord(
        &self,
        dim: usize,
        link: &AssignLink,
        condition: &NaryExpr,
        out_shape: &[usize],
    ) -> NaryExpr {
        let slice = &link.slices[dim];
        if slice.start == 0 && slice.end == out_shape[dim] {
            return NaryExpr::DimIndex(dim);
        }
        NaryExpr::select(
            condition.clone(),
            Self::unguarded_coord(dim, link),
            NaryExpr::scalar(NaryScalar::U32(0)),
            DataTypeEnum::U32,
            DataTypeEnum::U32,
        )
    }

    /// Fold an element-wise load through a fully-defined affine view by
    /// composing the chunk shift into the view's affine map:
    /// `constant' = constant - Σ coeff·slice_start`. Only fires when the
    /// shifted map provably stays inside the base for every destination
    /// coordinate; aligned narrows compose to the identity and read the base
    /// at the bare destination coordinates.
    fn try_fold_shifted_view(
        &self,
        view_inner: NodeIndex,
        link: &AssignLink,
        out_shape: &[usize],
        state: &mut LiftState,
    ) -> Option<NaryExpr> {
        let exec = self.get_input_node_in_exec_graph(view_inner)?;
        if !self.execution_graph.contains_node(exec) {
            return None;
        }
        let ExecutionVariant::View(view) = &self.execution_graph[exec].variant else {
            return None;
        };
        if view.shape().len() != link.slices.len() {
            return None;
        }
        for (extent, slice) in view.shape().iter().zip(&*link.slices) {
            if *extent != slice.end - slice.start {
                return None;
            }
        }
        let base_shape = &view.stages[0].input_shape;
        let collapsed = view.composed_layout()?;
        let affine = affine_dim_indices(&collapsed, base_shape)?;

        let mut shifted = Vec::with_capacity(affine.len());
        for (index, &extent) in affine.iter().zip(&**base_shape) {
            let mut constant = index.constant as i64;
            let mut max_offset = 0i64;
            for &(dim, coefficient) in &index.terms {
                constant -= coefficient as i64 * link.slices[dim].start as i64;
                max_offset += coefficient as i64 * (out_shape[dim] as i64 - 1);
            }
            if constant < 0 || constant + max_offset >= extent as i64 {
                return None;
            }
            shifted.push(AffineIndex {
                constant: constant as u32,
                terms: index.terms.clone(),
            });
        }

        let base_slot = state.slot(view.input);
        let coords: Vec<NaryExpr> = (0..out_shape.len()).map(NaryExpr::DimIndex).collect();
        Some(NaryExpr::IndexedInput {
            input_idx: base_slot,
            indices: shifted.iter().map(|index| index.to_expr(&coords)).collect(),
        })
    }
}
