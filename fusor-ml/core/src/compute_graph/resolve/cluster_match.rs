//! Shared expression/cluster matchers used by recognition and fusion passes.

use crate::nary_wise::{NaryFunction, NaryOp};
use crate::reduce::ReduceOp;

use super::*;

/// The keepdim-broadcast view layout a reduced tensor presents over its base:
/// row-major strides of the reduced shape with a `0` inserted at `axis`.
pub(super) fn keepdim_broadcast_layout(full_shape: &[usize], axis: usize) -> Layout {
    let reduced: Vec<usize> = full_shape
        .iter()
        .enumerate()
        .filter_map(|(dim, &size)| (dim != axis).then_some(size))
        .collect();
    let reduced_strides = Layout::continuous_strides(&reduced);
    let mut strides = Vec::with_capacity(full_shape.len());
    let mut reduced_dim = 0;
    for dim in 0..full_shape.len() {
        if dim == axis {
            strides.push(0);
        } else {
            strides.push(reduced_strides[reduced_dim]);
            reduced_dim += 1;
        }
    }
    Layout::from_parts(0, full_shape.into(), strides.into())
}

/// Layout equality modulo degenerate dimensions: strides of size-1 dims never
/// affect addressing, and view composition leaves arbitrary values there.
pub(super) fn layout_matches(actual: Option<&Layout>, expected: &Layout) -> bool {
    let Some(actual) = actual else {
        return false;
    };
    actual.shape() == expected.shape()
        && actual.offset() == expected.offset()
        && actual
            .shape()
            .iter()
            .zip(actual.strides())
            .zip(expected.strides())
            .all(|((&size, &actual), &expected)| size <= 1 || actual == expected)
}

/// Match a single-input n-ary whose body is `f(input)` with full-rank
/// elementwise indices, returning the function and the input node.
pub(super) fn unary_elementwise(nary: &ElementwiseOperation) -> Option<(NaryFunction, NodeIndex)> {
    let NaryExpr::Op { children, function } = &nary.expression else {
        return None;
    };
    let [
        NaryExpr::IndexedInput {
            input_idx: 0,
            indices,
        },
    ] = children.as_slice()
    else {
        return None;
    };
    (indices.len() == nary.shape.len()
        && NaryExpr::is_elementwise_indices(indices)
        && nary.inputs.len() == 1)
        .then(|| (function.clone(), nary.inputs[0]))
}

/// Match a two-input n-ary whose body is `op(input0, input1)` with full-rank
/// elementwise indices on both sides.
pub(super) fn binary_elementwise(
    nary: &ElementwiseOperation,
) -> Option<(NaryOp, NodeIndex, NodeIndex)> {
    let NaryExpr::Op { children, function } = &nary.expression else {
        return None;
    };
    let [
        NaryExpr::IndexedInput {
            input_idx: 0,
            indices: lhs,
        },
        NaryExpr::IndexedInput {
            input_idx: 1,
            indices: rhs,
        },
    ] = children.as_slice()
    else {
        return None;
    };
    (lhs.len() == nary.shape.len()
        && rhs.len() == nary.shape.len()
        && NaryExpr::is_elementwise_indices(lhs)
        && NaryExpr::is_elementwise_indices(rhs)
        && nary.inputs.len() == 2)
        .then(|| (function.op, nary.inputs[0], nary.inputs[1]))
}

/// The composed softmax cluster matched without rewriting it.
pub(super) struct SoftmaxCluster {
    pub(super) input: NodeIndex,
    pub(super) shape: Box<[usize]>,
    pub(super) axis: usize,
}

impl Resolver {
    pub(super) fn inner_nary(&self, inner: NodeIndex) -> Option<&ElementwiseOperation> {
        let exec = self.get_input_node_in_exec_graph(inner)?;
        match &self.execution_graph[exec].variant {
            ExecutionVariant::Elementwise(nary) => Some(nary),
            _ => None,
        }
    }

    pub(super) fn consumer_count_of(&self, inner: NodeIndex) -> Option<usize> {
        let exec = self.get_input_node_in_exec_graph(inner)?;
        Some(
            self.execution_graph
                .neighbors_directed(exec, petgraph::Direction::Outgoing)
                .count(),
        )
    }

    /// An intermediate node consumed by the recognized cluster alone:
    /// exactly `expected` exec-graph consumers and no user-held reference or
    /// cached value.
    pub(super) fn exclusively_consumed(
        &self,
        graph: &ComputeGraphInner,
        inner: NodeIndex,
        expected: usize,
    ) -> bool {
        !graph.has_live_reference(inner)
            && graph.get_cached_result(inner).is_none()
            && self.consumer_count_of(inner) == Some(expected)
    }

    /// A bare reduce of `op` with no fused chains: returns (axis, value).
    pub(super) fn match_reduce(
        &self,
        inner: NodeIndex,
        op: ReduceOp,
    ) -> Option<(usize, NodeIndex)> {
        let exec = self.get_input_node_in_exec_graph(inner)?;
        let ExecutionVariant::Reduce(reduce) = &self.execution_graph[exec].variant else {
            return None;
        };
        let value = reduce.plain_input()?;
        (reduce.function.op == op && reduce.post_element_wise.functions.is_empty())
            .then_some((reduce.axis, value))
    }

    /// A unary elementwise n-ary matching `accept`, returning its input.
    pub(super) fn match_unary(
        &self,
        inner: NodeIndex,
        accept: impl Fn(&NaryFunction) -> bool,
    ) -> Option<NodeIndex> {
        let nary = self.inner_nary(inner)?;
        let (function, input) = unary_elementwise(nary)?;
        accept(&function).then_some(input)
    }

    /// Match the composed softmax cluster rooted at `probs_inner` (see
    /// `Tensor::softmax`): `div(exp(sub(x, bcast(max(x, axis)))),
    /// bcast(sum(exp, axis)))`, with every intermediate consumed exclusively
    /// by the cluster. Used by attention recognition to see through the
    /// probabilities without the cluster having been rewritten.
    pub(super) fn match_softmax_cluster(
        &self,
        graph: &ComputeGraphInner,
        probs_inner: NodeIndex,
    ) -> Option<SoftmaxCluster> {
        let div = self.inner_nary(probs_inner)?;
        let (div_op, exp_inner, sum_view_inner) = binary_elementwise(div)?;
        if div_op != NaryOp::Div {
            return None;
        }
        let shape = div.shape.clone();

        let shifted_inner = self.match_unary(exp_inner, |function| function.op == NaryOp::Exp)?;
        let (sub_op, x_inner, max_view_inner) = self
            .inner_nary(shifted_inner)
            .and_then(binary_elementwise)?;
        if sub_op != NaryOp::Sub {
            return None;
        }

        // Denominator: bcast(sum(exp, axis)) reading the same exp node.
        let (sum_base, sum_layout) = self.walk_view_chain(sum_view_inner);
        let (axis, sum_value) = self.match_reduce(sum_base, ReduceOp::Sum)?;
        if sum_value != exp_inner
            || !layout_matches(sum_layout.as_ref(), &keepdim_broadcast_layout(&shape, axis))
        {
            return None;
        }

        // Shift: bcast(max(x, axis)) along the same axis, over the same x.
        let (max_base, max_layout) = self.walk_view_chain(max_view_inner);
        let (max_axis, max_value) = self.match_reduce(max_base, ReduceOp::Max)?;
        if max_axis != axis
            || max_value != x_inner
            || !layout_matches(max_layout.as_ref(), &keepdim_broadcast_layout(&shape, axis))
        {
            return None;
        }

        // The whole cluster must exist solely for this softmax.
        (self.exclusively_consumed(graph, exp_inner, 2)
            && self.exclusively_consumed(graph, shifted_inner, 1)
            && self.exclusively_consumed(graph, sum_view_inner, 1)
            && self.exclusively_consumed(graph, max_view_inner, 1)
            && self.exclusively_consumed(graph, sum_base, 1)
            && self.exclusively_consumed(graph, max_base, 1))
        .then_some(SoftmaxCluster {
            input: x_inner,
            shape,
            axis,
        })
    }
}
