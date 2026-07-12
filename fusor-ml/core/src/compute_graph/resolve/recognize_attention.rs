//! Recognition of composed attention clusters.
//!
//! Runs third, after contractions and normalizations: by then the canonical
//! cluster from `Tensor::flash_attention` has collapsed to
//! `MatMul(Softmax(scale·MatMul(q, kᵀ) [+ mask]), v)` with the GQA-expand /
//! transpose / mask broadcast views still attached to the original
//! q/k/v/mask nodes. Recognition rebuilds the attention row program when its
//! gates pass; otherwise the cluster runs through the recognized matmul +
//! softmax kernels.

use crate::{
    MatMulOperation,
    nary_wise::{NaryOp, NaryScalar},
    view::ViewOperation,
};

use super::cluster_match::{binary_elementwise, layout_matches, unary_elementwise};
use super::*;

struct MatchedAttention {
    q: NodeIndex,
    k: NodeIndex,
    v: NodeIndex,
    mask: Option<NodeIndex>,
    q_shape: Vec<usize>,
    k_shape: Vec<usize>,
    v_shape: Vec<usize>,
    mask_shape: Option<Vec<usize>>,
    scale: f32,
    datatype: DataTypeEnum,
    causal: bool,
}

/// Match `select(kv_pos <= q_pos, scores, -inf)` over a single input,
/// returning the scores node.
fn match_causal_select(nary: &ElementwiseOperation) -> Option<NodeIndex> {
    let NaryExpr::Op { children, function } = &nary.expression else {
        return None;
    };
    if function.op != NaryOp::Select || nary.inputs.len() != 1 {
        return None;
    }
    let [condition, on_true, on_false] = children.as_slice() else {
        return None;
    };
    let NaryExpr::Op {
        children: bound,
        function: compare,
    } = condition
    else {
        return None;
    };
    if compare.op != NaryOp::LessEqual
        || bound.as_slice() != [NaryExpr::DimIndex(3), NaryExpr::DimIndex(2)]
    {
        return None;
    }
    let NaryExpr::IndexedInput {
        input_idx: 0,
        indices,
    } = on_true
    else {
        return None;
    };
    if indices.len() != nary.shape.len() || !NaryExpr::is_elementwise_indices(indices) {
        return None;
    }
    let negative_infinity = match on_false {
        NaryExpr::Scalar(crate::nary_wise::NaryScalar::F32(value)) => *value == f32::NEG_INFINITY,
        NaryExpr::Scalar(crate::nary_wise::NaryScalar::F16(value)) => {
            *value == half::f16::NEG_INFINITY
        }
        _ => false,
    };
    negative_infinity.then_some(nary.inputs[0])
}

impl Resolver {
    pub(super) fn recognize_attention(&mut self, graph: &mut ComputeGraphInner) {
        let candidates: Vec<ExecutionNodeIndex> = self
            .execution_graph
            .node_indices()
            .filter(|&node| {
                matches!(
                    self.execution_graph[node].variant,
                    ExecutionVariant::MatMul(_)
                )
            })
            .collect();
        for node in candidates {
            if !self.execution_graph.contains_node(node) {
                continue;
            }
            self.try_recognize_attention(graph, node);
        }
    }

    fn inner_view(&self, inner: NodeIndex) -> Option<&ViewOperation> {
        let exec = self.get_input_node_in_exec_graph(inner)?;
        match &self.execution_graph[exec].variant {
            ExecutionVariant::View(view) => Some(view),
            _ => None,
        }
    }

    fn inner_matmul(&self, inner: NodeIndex) -> Option<&MatMulOperation> {
        let exec = self.get_input_node_in_exec_graph(inner)?;
        match &self.execution_graph[exec].variant {
            ExecutionVariant::MatMul(matmul) => Some(matmul),
            _ => None,
        }
    }

    fn try_recognize_attention(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let Some(matched) = self.match_attention(graph, node_idx) else {
            return false;
        };
        let mut dependencies = vec![matched.q, matched.k, matched.v];
        if let Some(mask) = matched.mask {
            dependencies.push(mask);
        }
        // The recognized cluster lowers through the generic attention row
        // program; shapes it cannot host stay composed (matmul + softmax
        // row program + matmul).
        let Some(operation) = crate::row_program::attention_row_program(
            &graph.device(),
            crate::row_program::AttentionInputs {
                q: matched.q,
                k: matched.k,
                v: matched.v,
                mask: matched.mask,
                q_shape: &matched.q_shape,
                k_shape: &matched.k_shape,
                v_shape: &matched.v_shape,
                mask_shape: matched.mask_shape.as_deref(),
                scale: matched.scale,
                input_dtype: matched.datatype,
                causal: matched.causal,
            },
        ) else {
            return false;
        };
        self.commit_recognized(
            graph,
            node_idx,
            &dependencies,
            ExecutionVariant::RowProgram(operation),
        );
        true
    }

    fn match_attention(
        &self,
        graph: &ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> Option<MatchedAttention> {
        let ExecutionVariant::MatMul(out) = &self.execution_graph[node_idx].variant else {
            return None;
        };
        if !out.pre_element_wise[0].functions.is_empty()
            || !out.pre_element_wise[1].functions.is_empty()
            || !out.post_element_wise.functions.is_empty()
            || !out.a.is_plain()
            || !out.b.is_plain()
        {
            return None;
        }
        let probs_inner = out.first;
        let v_eff_inner = out.second;
        let v_eff_shape = out.b.shape.clone();
        let datatype = out.datatype;

        // probs = softmax(scores) along the last axis of a rank-4 space —
        // matched in its composed form (the compiler no longer rewrites
        // standalone softmax into a named operation).
        let softmax = self.match_softmax_cluster(graph, probs_inner)?;
        let shape = softmax.shape.to_vec();
        if shape.len() != 4 || softmax.axis != 3 {
            return None;
        }
        let (batch, num_heads, q_seq_len, kv_seq_len) = (shape[0], shape[1], shape[2], shape[3]);
        let mut scores_inner = softmax.input;
        // The composed softmax reads its input twice (the shift and the max
        // reduction), so the scores node has two in-cluster consumers; every
        // node below it is read once.
        let expected_consumers = |node: NodeIndex| if node == softmax.input { 2 } else { 1 };

        // Causal masking: select(kv_pos <= q_pos, scaled, -inf) — pure index
        // arithmetic emitted by `flash_attention_causal`.
        let mut causal = false;
        if let Some(scaled) = self.inner_nary(scores_inner).and_then(match_causal_select) {
            if !self.exclusively_consumed(graph, scores_inner, expected_consumers(scores_inner)) {
                return None;
            }
            causal = true;
            scores_inner = scaled;
        }

        // Optional additive mask: add(scaled, bcast(mask)) where the mask
        // broadcast reads a rank-2 [q_seq, kv_seq] base.
        let mut mask = None;
        let mut mask_shape: Option<Vec<usize>> = None;
        if !causal
            && let Some((NaryOp::Add, lhs, rhs)) =
                self.inner_nary(scores_inner).and_then(binary_elementwise)
        {
            let mut matched_mask = None;
            for (scaled_side, mask_side) in [(lhs, rhs), (rhs, lhs)] {
                let Some(view) = self.inner_view(mask_side) else {
                    continue;
                };
                let Some(stage) = view.plain().filter(|stage| stage.is_fully_defined()) else {
                    continue;
                };
                let expected =
                    Layout::from_parts(0, shape.clone().into(), [0, 0, kv_seq_len, 1].into());
                if !layout_matches(Some(&stage.layout), &expected)
                    || stage.input_shape.as_ref() != [q_seq_len, kv_seq_len]
                {
                    continue;
                }
                if !self.exclusively_consumed(graph, mask_side, 1) {
                    continue;
                }
                matched_mask = Some((scaled_side, view.input, stage.input_shape.to_vec()));
                break;
            }
            let (scaled_side, mask_base, base_shape) = matched_mask?;
            if !self.exclusively_consumed(graph, scores_inner, expected_consumers(scores_inner)) {
                return None;
            }
            mask = Some(mask_base);
            mask_shape = Some(base_shape);
            scores_inner = scaled_side;
        }

        // scaled = qk · scale
        let scale = {
            let nary = self.inner_nary(scores_inner)?;
            let (function, _) = unary_elementwise(nary)?;
            match function.op {
                NaryOp::MulConst(NaryScalar::F32(scale)) => scale,
                NaryOp::MulConst(NaryScalar::F16(scale)) => scale.to_f32(),
                _ => return None,
            }
        };
        let qk_inner = self.match_unary(scores_inner, |function| {
            matches!(function.op, NaryOp::MulConst(_))
        })?;
        if !self.exclusively_consumed(graph, scores_inner, expected_consumers(scores_inner)) {
            return None;
        }

        // qk = MatMul(q, kᵀ-view)
        let qk = self.inner_matmul(qk_inner)?;
        if !qk.pre_element_wise[0].functions.is_empty()
            || !qk.pre_element_wise[1].functions.is_empty()
            || !qk.post_element_wise.functions.is_empty()
            || !qk.a.is_plain()
            || !qk.b.is_plain()
        {
            return None;
        }
        let q = qk.first;
        let q_shape = qk.a.shape.to_vec();
        let kt_inner = qk.second;
        if q_shape.len() != 4
            || q_shape[..3] != [batch, num_heads, q_seq_len]
            || !self.exclusively_consumed(graph, qk_inner, 1)
        {
            return None;
        }
        let head_dim = q_shape[3];
        let expanded_shape = [batch, num_heads, kv_seq_len, head_dim];

        // kᵀ: a transpose view attached to the (possibly GQA-expanded) k.
        let kt = self.inner_view(kt_inner)?;
        let kt_stage = kt.plain().filter(|stage| stage.is_fully_defined())?;
        let expected_kt = Layout::contiguous(&expanded_shape).transpose(2, 3);
        if !layout_matches(Some(&kt_stage.layout), &expected_kt)
            || kt_stage.input_shape.as_ref() != expanded_shape
            || !self.exclusively_consumed(graph, kt_inner, 1)
        {
            return None;
        }
        let (k, k_shape) = self.peel_gqa_expand(graph, kt.input, &expanded_shape)?;
        let (v, v_shape) = if v_eff_shape.as_ref() == expanded_shape {
            self.peel_gqa_expand(graph, v_eff_inner, &expanded_shape)?
        } else {
            return None;
        };
        if k_shape != v_shape {
            return None;
        }

        // The intermediate cluster must exist solely for this attention.
        if !self.exclusively_consumed(graph, probs_inner, 1) {
            return None;
        }

        Some(MatchedAttention {
            q,
            k,
            v,
            mask,
            q_shape,
            k_shape,
            v_shape,
            mask_shape,
            scale,
            datatype,
            causal,
        })
    }

    /// Peel the canonical GQA expansion (flat-reinterpret view over a
    /// stride-0 group broadcast) back to the original K/V node. Any other
    /// node — including arbitrary user views like KV-cache slices — is the
    /// tensor itself with the full expanded shape (groups == 1).
    fn peel_gqa_expand(
        &self,
        graph: &ComputeGraphInner,
        inner: NodeIndex,
        expanded_shape: &[usize; 4],
    ) -> Option<(NodeIndex, Vec<usize>)> {
        let [batch, num_heads, kv_seq_len, head_dim] = *expanded_shape;
        let unexpanded = Some((inner, expanded_shape.to_vec()));

        let Some(reinterpret_view) = self.inner_view(inner) else {
            return unexpanded;
        };
        let Some(reinterpret) = reinterpret_view
            .plain()
            .filter(|stage| stage.is_fully_defined())
        else {
            return unexpanded;
        };
        // The flat reinterpret: contiguous rank-4 over a rank-5 grouped space.
        if reinterpret.input_shape.len() != 5
            || !layout_matches(
                Some(&reinterpret.layout),
                &Layout::contiguous(expanded_shape),
            )
        {
            return unexpanded;
        }
        let [b, num_kv_heads, groups, s, d] = *reinterpret.input_shape.as_ref() else {
            return unexpanded;
        };
        if b != batch
            || s != kv_seq_len
            || d != head_dim
            || groups <= 1
            || num_kv_heads * groups != num_heads
        {
            return unexpanded;
        }
        let Some(broadcast_view) = self.inner_view(reinterpret_view.input) else {
            return unexpanded;
        };
        let Some(broadcast) = broadcast_view
            .plain()
            .filter(|stage| stage.is_fully_defined())
        else {
            return unexpanded;
        };
        let expected_broadcast = Layout::from_parts(
            0,
            [b, num_kv_heads, groups, s, d].into(),
            [num_kv_heads * s * d, s * d, 0, d, 1].into(),
        );
        if !layout_matches(Some(&broadcast.layout), &expected_broadcast)
            || broadcast.input_shape.as_ref() != [b, num_kv_heads, s, d]
            || !self.exclusively_consumed(graph, inner, 1)
            || !self.exclusively_consumed(graph, reinterpret_view.input, 1)
        {
            return unexpanded;
        }
        Some((broadcast_view.input, vec![b, num_kv_heads, s, d]))
    }
}
