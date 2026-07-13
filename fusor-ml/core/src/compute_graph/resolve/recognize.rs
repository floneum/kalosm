//! Contraction matchers over the composed 3-op graph.
//!
//! The tensor API expresses contractions as `Elementwise(Mul) + Reduce(Sum)`
//! over a shared index space (see `Tensor::mat_mul` / `Tensor::q_mat_mul`).
//! The matchers and builders are consumed by a linear recognition sweep over
//! the identity forms emitted by the tensor interface. Keeping recognition
//! outside the equality-saturation pass avoids rebuilding the same large
//! decode e-graph for every generated token.
//! Anything they do not recognize lowers through the generic elementwise +
//! reduce kernels — slower, but correct.

use crate::{
    MatMulOperation, ReduceOperation, quantized::matmul::QMatMulOperation, reduce::ReduceOp,
};

use super::*;

/// The canonical contraction cluster: `Reduce(Sum, axis = rank-1)` over a
/// two-factor multiply where each factor is a bare indexed load with pure
/// `DimIndex` coordinates.
pub(crate) struct Contraction {
    /// The two factors: (inner node, output dims indexed).
    factors: [(NodeIndex, Vec<usize>); 2],
    shape: Box<[usize]>,
    datatype: DataTypeEnum,
}

fn pure_dim_indices(indices: &[NaryExpr]) -> Option<Vec<usize>> {
    indices
        .iter()
        .map(|index| match index {
            NaryExpr::DimIndex(dim) => Some(*dim),
            _ => None,
        })
        .collect()
}

/// Match the canonical multiply + sum-reduce pair.
pub(crate) fn match_contraction(
    reduce: &ReduceOperation,
    nary: &ElementwiseOperation,
) -> Option<Contraction> {
    if reduce.function.op != ReduceOp::Sum
        || reduce.plain_input().is_none()
        || !reduce.post_element_wise.functions.is_empty()
        || reduce.axis + 1 != nary.shape.len()
    {
        return None;
    }

    let NaryExpr::Op { children, function } = &nary.expression else {
        return None;
    };
    if function.op != crate::nary_wise::NaryOp::Mul || children.len() != 2 {
        return None;
    }
    let factor = |expr: &NaryExpr| -> Option<(NodeIndex, Vec<usize>)> {
        let NaryExpr::IndexedInput { input_idx, indices } = expr else {
            return None;
        };
        Some((nary.inputs[*input_idx], pure_dim_indices(indices)?))
    };

    Some(Contraction {
        factors: [factor(&children[0])?, factor(&children[1])?],
        shape: nary.shape.clone(),
        datatype: nary.output_datatype,
    })
}

impl Contraction {
    /// `input [.., K] × QMatrix [N, K] → [.., N]`: index space `[.., N, K]`,
    /// activation indices `[0..r-2, k]` (skipping the `N` dim), matrix
    /// indices `[n, k]`. Returns the rebuilt operation and its activation
    /// input node.
    pub(crate) fn to_q_mat_mul(
        &self,
        dequantize_for: impl Fn(NodeIndex) -> Option<DequantizeOperation>,
    ) -> Option<(QMatMulOperation, NodeIndex)> {
        let rank = self.shape.len();
        if rank < 2 {
            return None;
        }
        let (k_dim, n_dim) = (rank - 1, rank - 2);
        let expected_activation: Vec<usize> = (0..rank - 2).chain(std::iter::once(k_dim)).collect();

        for (activation, matrix) in [
            (&self.factors[0], &self.factors[1]),
            (&self.factors[1], &self.factors[0]),
        ] {
            let Some(matrix_op) = dequantize_for(matrix.0) else {
                continue;
            };
            if matrix.1 != [n_dim, k_dim]
                || activation.1 != expected_activation
                || matrix_op.datatype != self.datatype
                || matrix_op.matrix.shape() != [self.shape[n_dim], self.shape[k_dim]]
            {
                continue;
            }

            let in_shape: Vec<usize> = self
                .shape
                .iter()
                .enumerate()
                .filter_map(|(dim, &size)| (dim != n_dim).then_some(size))
                .collect();
            let operation = QMatMulOperation::new(
                self.datatype,
                &in_shape,
                activation.0,
                matrix_op.matrix.clone(),
            );
            return Some((operation, activation.0));
        }
        None
    }

    /// `a [batch.., M, K] × b [batch.., K, N] → [batch.., M, N]`: index space
    /// `[batch.., M, N, K]`, `a` indices `[batch.., m, k]`, `b` indices
    /// `[batch.., k, n]`. Returns the rebuilt operation and its two inputs.
    pub(crate) fn to_mat_mul(
        &self,
        device: &crate::Device,
    ) -> Option<(MatMulOperation, [NodeIndex; 2])> {
        let rank = self.shape.len();
        if rank < 3 {
            return None;
        }
        let batch = rank - 3;
        let (m_dim, n_dim, k_dim) = (batch, batch + 1, batch + 2);

        let expected_a: Vec<usize> = (0..batch).chain([m_dim, k_dim]).collect();
        let expected_b: Vec<usize> = (0..batch).chain([k_dim, n_dim]).collect();
        let (a, b) = if self.factors[0].1 == expected_a && self.factors[1].1 == expected_b {
            (&self.factors[0], &self.factors[1])
        } else if self.factors[1].1 == expected_a && self.factors[0].1 == expected_b {
            (&self.factors[1], &self.factors[0])
        } else {
            return None;
        };

        let batch_shape = &self.shape[..batch];
        let first_shape: Vec<usize> = batch_shape
            .iter()
            .chain([&self.shape[m_dim], &self.shape[k_dim]])
            .copied()
            .collect();
        let second_shape: Vec<usize> = batch_shape
            .iter()
            .chain([&self.shape[k_dim], &self.shape[n_dim]])
            .copied()
            .collect();

        let operation = MatMulOperation::new(
            self.datatype,
            a.0,
            b.0,
            &first_shape,
            &second_shape,
            None,
            device,
        );
        Some((operation, [a.0, b.0]))
    }
}

impl ComputeGraphInner {
    pub(crate) fn dequantize_variant(&self, key: NodeIndex) -> Option<DequantizeOperation> {
        match &self.nodes.nodes.node_weight(key)?.variant {
            ComputeGraphNodeVariant::QMatrix(op) => Some(op.clone()),
            _ => None,
        }
    }
}

impl Resolver {
    /// Recognize canonical contraction clusters with one linear graph sweep.
    /// This is equivalent to the former native-egg recognition rule, but the
    /// API emits an exact canonical form so saturation adds no value here.
    pub(super) fn recognize_contractions(&mut self, graph: &mut ComputeGraphInner) {
        let reduces: Vec<ExecutionNodeIndex> = self
            .execution_graph
            .node_indices()
            .filter(|&node| {
                matches!(
                    self.execution_graph[node].variant,
                    ExecutionVariant::Reduce(_)
                )
            })
            .collect();
        for node in reduces {
            self.try_recognize_contraction(graph, node);
        }
    }

    fn try_recognize_contraction(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let ExecutionVariant::Reduce(reduce) = &self.execution_graph[node_idx].variant else {
            return false;
        };
        let Some(value) = reduce.plain_input() else {
            return false;
        };
        if self.check_cached(graph, value) || graph.has_live_reference(value) {
            return false;
        }
        let Some(nary_exec) = self.get_input_node_in_exec_graph(value) else {
            return false;
        };
        let ExecutionVariant::Elementwise(nary) = &self.execution_graph[nary_exec].variant else {
            return false;
        };
        if self
            .execution_graph
            .neighbors_directed(nary_exec, petgraph::Direction::Outgoing)
            .count()
            != 1
        {
            return false;
        }
        let Some(contraction) = match_contraction(reduce, nary) else {
            return false;
        };

        if let Some((operation, activation)) =
            contraction.to_q_mat_mul(|key| graph.dequantize_variant(key))
        {
            self.commit_recognized(
                graph,
                node_idx,
                &[activation],
                ExecutionVariant::QMatMul(Box::new(operation)),
            );
            return true;
        }
        if let Some((mut operation, _)) = contraction.to_mat_mul(&graph.device()) {
            let device = graph.device();
            try_unflatten_matmul_input_with(
                &mut operation,
                &device,
                |node| self.check_cached(graph, node),
                |node| graph.has_live_reference(node),
                |node| match &graph.nodes.nodes.node_weight(node)?.variant {
                    ComputeGraphNodeVariant::View(view) => Some(view.clone()),
                    _ => None,
                },
            );
            let inputs = [operation.first, operation.second];
            self.commit_recognized(
                graph,
                node_idx,
                &inputs,
                ExecutionVariant::MatMul(operation),
            );
            return true;
        }
        false
    }

    /// Recognize quantized row gathers with one linear graph sweep.
    pub(super) fn recognize_embeddings(&mut self, graph: &mut ComputeGraphInner) {
        let candidates: Vec<ExecutionNodeIndex> = self
            .execution_graph
            .node_indices()
            .filter(|&node| {
                matches!(
                    self.execution_graph[node].variant,
                    ExecutionVariant::Elementwise(_)
                )
            })
            .collect();
        for node in candidates {
            if self.execution_graph.contains_node(node) {
                self.try_recognize_q_embedding(graph, node);
            }
        }
    }

    fn try_recognize_q_embedding(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let ExecutionVariant::Elementwise(nary) = &self.execution_graph[node_idx].variant else {
            return false;
        };
        if nary.inputs.len() != 2 || nary.shape.len() != 2 {
            return false;
        }
        let gather = match &nary.expression {
            NaryExpr::Op { children, function }
                if function.op == crate::nary_wise::NaryOp::Cast && children.len() == 1 =>
            {
                &children[0]
            }
            expr => expr,
        };
        let NaryExpr::IndexedInput {
            input_idx: 0,
            indices,
        } = gather
        else {
            return false;
        };
        let [row, NaryExpr::DimIndex(1)] = indices.as_slice() else {
            return false;
        };
        let NaryExpr::IndexedInput {
            input_idx: 1,
            indices: row_indices,
        } = row
        else {
            return false;
        };
        if row_indices.as_slice() != [NaryExpr::DimIndex(0)] {
            return false;
        }

        let Some(dequantize) = graph.dequantize_variant(nary.inputs[0]) else {
            return false;
        };
        if crate::quantized::dequantize::quant_format(&dequantize.matrix).is_none()
            || dequantize.matrix.shape().len() != 2
            || dequantize.matrix.shape()[1] != nary.shape[1]
        {
            return false;
        }

        let indexes = nary.inputs[1];
        let operation = QEmbeddingOperation::new(
            indexes,
            nary.shape[0],
            dequantize.matrix.clone(),
            nary.output_datatype,
        );
        self.commit_recognized(
            graph,
            node_idx,
            &[indexes],
            ExecutionVariant::QEmbedding(operation),
        );
        true
    }

    /// Replace a recognized cluster's root with the rebuilt operation: drop
    /// every edge from the cluster's intermediates, wire the operation's
    /// dependencies directly, and let the now-unconsumed intermediates fall
    /// out of the execution graph.
    pub(super) fn commit_recognized(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        dependencies: &[NodeIndex],
        variant: ExecutionVariant,
    ) {
        self.execution_graph[node_idx].variant = variant;

        let previous: Vec<ExecutionNodeIndex> = self
            .execution_graph
            .neighbors_directed(node_idx, petgraph::Direction::Incoming)
            .collect();
        for &prev in &previous {
            if let Some(edge) = self.execution_graph.find_edge(prev, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
        }
        for &dependency in dependencies {
            if let Some(exec) = self.get_input_node_in_exec_graph(dependency)
                && self.execution_graph.find_edge(exec, node_idx).is_none()
            {
                self.execution_graph.add_edge(exec, node_idx, ());
            }
        }
        self.add_physical_dependencies(graph, node_idx, dependencies);
        for prev in previous {
            self.remove_node_if_dead(prev);
        }
    }
}

/// Read a recognized matmul's A operand through its un-flattened producer.
/// Native egg appliers own their context and cannot borrow the live compute
/// graph, so every graph observation is supplied explicitly.
pub(super) fn try_unflatten_matmul_input_with(
    operation: &mut crate::MatMulOperation,
    device: &crate::Device,
    check_cached: impl Fn(NodeIndex) -> bool,
    has_live_reference: impl Fn(NodeIndex) -> bool,
    view_for: impl Fn(NodeIndex) -> Option<crate::view::ViewOperation>,
) {
    if !operation.a.is_plain() || operation.a.batch_dims != 0 {
        return;
    }
    // Only the cooperative-matrix kernel reads an un-flattened operand
    // faster than gather-then-matmul: its tile staging amortizes the
    // per-load coordinate decomposition. The generic reduce re-derives
    // coordinates for every load and measures slower than the gather at
    // every meaningful size, so anything bound for it keeps the
    // materialized matrix.
    if !operation.hardware_matmul_statically_viable(device) {
        return;
    }
    let (m, k) = (operation.a.rows(), operation.a.cols());
    // An already-materialized (or externally held) operand is cheaper to
    // read flat than to re-derive coordinates for.
    if check_cached(operation.first) || has_live_reference(operation.first) {
        return;
    }
    let Some(view) = view_for(operation.first) else {
        return;
    };
    // The stack must be an affine relayout under a flat [M, K]
    // reinterpret, both pure relayouts (no fill regions).
    let [windowed, flat] = view.stages.as_slice() else {
        return;
    };
    if !windowed.is_fully_defined()
        || !flat.is_fully_defined()
        || !flat.layout.is_contiguous()
        || flat.layout.offset() != 0
        || flat.layout.shape() != [m, k]
    {
        return;
    }
    // The kernels substitute the windowed map as affine per-dim index
    // arithmetic; validate it here so the lowering can rely on it.
    if crate::view::affine_dim_indices(&windowed.layout, &windowed.input_shape).is_none() {
        return;
    }
    let operand_shape = windowed.shape();
    // The producer's dims must split cleanly into an `M` prefix and a
    // `K` suffix for the per-side flat-coordinate decomposition.
    let mut product = 1usize;
    let mut k_start = operand_shape.len();
    while k_start > 0 && product < k {
        k_start -= 1;
        let Some(next) = product.checked_mul(operand_shape[k_start]) else {
            return;
        };
        product = next;
    }
    if product != k
        || k_start == 0
        || k_start == operand_shape.len()
        || operand_shape[..k_start].iter().product::<usize>() != m
    {
        return;
    }
    // The flat row/column coordinates decompose with u32 arithmetic.
    let probe = NaryExpr::DimIndex(0);
    if crate::view::row_major_indices_from_flat(probe.clone(), &operand_shape[..k_start]).is_none()
        || crate::view::row_major_indices_from_flat(probe, &operand_shape[k_start..]).is_none()
    {
        return;
    }
    operation.first = view.input;
    operation.a = crate::matmul::MatrixOperand {
        shape: operand_shape.into(),
        batch_dims: 0,
        row_dims: k_start,
        base_map: Some(crate::matmul::OperandBaseMap {
            layout: windowed.layout.clone(),
            base_shape: windowed.input_shape.clone(),
        }),
    };
}
