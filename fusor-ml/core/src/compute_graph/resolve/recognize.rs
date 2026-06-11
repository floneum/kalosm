//! Pattern recognition over the composed 3-op graph.
//!
//! The tensor API expresses contractions as `Elementwise(Mul) + Reduce(Sum)`
//! over a shared index space (see `Tensor::mat_mul` / `Tensor::q_mat_mul`).
//! This pass runs before view folding and n-ary fusion, while the composed
//! cluster is still in the exact canonical form the API emitted, and rebuilds
//! the specialized operation so the existing kernel paths (and their epilogue
//! fusion) take over. Anything it does not recognize lowers through the
//! generic elementwise + reduce kernels — slower, but correct.

use crate::{
    MatMulOperation, ReduceOperation, dequantize::DequantizeOperation,
    quantized::matmul::QMatMulOperation, reduce::ReduceOp,
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

    /// Recognize a composed contraction rooted at `key` directly on the inner
    /// graph (the single-target fast path, no resolver involved). The multiply
    /// must feed only this reduce.
    pub(crate) fn match_direct_qmatmul(&self, key: NodeIndex) -> Option<QMatMulOperation> {
        let ComputeGraphNodeVariant::Reduce(reduce) = &self.nodes.nodes.node_weight(key)?.variant
        else {
            return None;
        };
        let value = reduce.plain_input()?;
        if self.get_cached_result(value).is_some() || self.has_live_reference(value) {
            return None;
        }
        let ComputeGraphNodeVariant::Elementwise(nary) =
            &self.nodes.nodes.node_weight(value)?.variant
        else {
            return None;
        };
        if self
            .nodes
            .nodes
            .neighbors_directed(value, petgraph::Direction::Outgoing)
            .count()
            != 1
        {
            return None;
        }
        let contraction = match_contraction(reduce, nary)?;
        let (operation, _) = contraction.to_q_mat_mul(|key| self.dequantize_variant(key))?;
        Some(operation)
    }
}

impl Resolver {
    /// Recognize composed contraction clusters in the execution graph and
    /// rebuild them as `MatMul` / `QMatMul` nodes. Runs as a linear sweep
    /// before any other rewrite, while the clusters are still in the exact
    /// form the API emitted.
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
        // The multiply must exist solely for this reduce: consumed elsewhere
        // it has to materialize anyway.
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

        let _ = nary_exec;
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
        if let Some((operation, inputs)) = contraction.to_mat_mul(&graph.device()) {
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

    /// Sweep elementwise nodes for quantized embedding gathers.
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
            if !self.execution_graph.contains_node(node) {
                continue;
            }
            self.try_recognize_q_embedding(graph, node);
        }
    }

    /// Quantized row gather: `Elementwise([table, idx], table[idx[i], j])`
    /// over a `[count, hidden]` space (see `QMatrix::index_select_rows_to`).
    /// Rebuilds the block-amortized embedding kernel; dense-storage tables
    /// stay on the generic elementwise path, which reads them directly.
    pub(super) fn try_recognize_q_embedding(
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
        // Peel the optional cast from the load type to the requested type.
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
        if crate::quantized::dequantize::quant_format(&dequantize.matrix).is_none() {
            return false;
        }
        if dequantize.matrix.shape().len() != 2 || dequantize.matrix.shape()[1] != nary.shape[1] {
            return false;
        }

        let indexes = nary.inputs[1];
        let operation = crate::quantized::embedding::QEmbeddingOperation::new(
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
