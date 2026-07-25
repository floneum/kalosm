//! Recognition of composed attention clusters.
//!
//! Runs third, after contractions and normalizations: by then the canonical
//! cluster from `Tensor::attention` has collapsed to
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

use super::cluster_match::{
    binary_elementwise, keepdim_broadcast_layout, layout_matches, unary_elementwise,
};
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
    let masked_score = match on_false {
        NaryExpr::Scalar(crate::nary_wise::NaryScalar::F32(value)) => {
            *value == crate::composite::attention::MASKED_SCORE_F32
        }
        NaryExpr::Scalar(crate::nary_wise::NaryScalar::F16(value)) => {
            *value == crate::composite::attention::MASKED_SCORE_F16
        }
        _ => false,
    };
    masked_score.then_some(nary.inputs[0])
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
        // The paired KV-side contraction pattern roots at a slice-assign
        // chain (composed as elementwise region-selects); claim it before
        // the single-contraction scans so the halves are not recognized
        // separately.
        let assign_candidates: Vec<ExecutionNodeIndex> = self
            .execution_graph
            .node_indices()
            .filter(|&node| {
                matches!(
                    self.execution_graph[node].variant,
                    ExecutionVariant::Elementwise(_)
                )
            })
            .collect();
        for node in assign_candidates {
            if !self.execution_graph.contains_node(node) {
                continue;
            }
            self.try_recognize_attention_grad_pair(graph, node);
        }
        for node in candidates {
            if !self.execution_graph.contains_node(node) {
                continue;
            }
            if self.try_recognize_attention(graph, node) {
                continue;
            }
            self.try_recognize_attention_grad(graph, node);
        }
        let lse_candidates: Vec<ExecutionNodeIndex> = self
            .execution_graph
            .node_indices()
            .filter(|&node| {
                matches!(
                    self.execution_graph[node].variant,
                    ExecutionVariant::Elementwise(_)
                )
            })
            .collect();
        for node in lse_candidates {
            if !self.execution_graph.contains_node(node) {
                continue;
            }
            self.try_recognize_score_lse(graph, node);
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
        let inputs = crate::row_program::AttentionInputs {
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
        };
        // Multi-row shapes need cross-row K/V reuse: the fused flash kernel
        // claims them whenever the device and shape qualify. Everything else
        // — decode's single row, ragged or oversized extents, non-f32 —
        // lowers through the generic attention row program as before.
        if matched.q_shape[2] > 1
            && let Some(operation) = crate::flash_attention::FlashAttentionOperation::try_new_output(
                &graph.device(),
                &inputs,
            )
        {
            self.commit_recognized(
                graph,
                node_idx,
                &dependencies,
                ExecutionVariant::Attention(operation),
            );
            return true;
        }
        let Some(operation) = crate::row_program::attention_row_program(&graph.device(), inputs)
        else {
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
        // arithmetic emitted by `attention_causal`.
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

/// One matched scaled-masked score cluster
/// (`scale·q·kᵀ [+ mask | causal-select]`).
struct MatchedScores {
    q: NodeIndex,
    k: NodeIndex,
    mask: Option<NodeIndex>,
    causal: bool,
    scale: f32,
    /// `[batch, heads, q_len, kv_len]`.
    shape: [usize; 4],
    head_dim: usize,
    kv_heads: usize,
    datatype: DataTypeEnum,
}

impl MatchedScores {
    fn dims(&self) -> [usize; 6] {
        let [batch, heads, q_len, kv_len] = self.shape;
        [batch, heads, self.kv_heads, q_len, kv_len, self.head_dim]
    }
}

/// A probability-jacobian factor: `p ∘ (grad_o·vᵀ − bcast(dsum)) · scale`.
struct MatchedDs {
    scores: MatchedScores,
    lse: NodeIndex,
    dsum: NodeIndex,
    grad_o: NodeIndex,
    v: NodeIndex,
}

impl Resolver {
    /// Match the canonical score cluster rooted at `root_inner`. No
    /// consumption constraints: patterns share these intermediates by
    /// construction, and whichever nodes end up unconsumed after their
    /// roots are rewritten cascade away.
    fn match_score_cluster(
        &self,
        graph: &ComputeGraphInner,
        root_inner: NodeIndex,
    ) -> Option<MatchedScores> {
        let root = self.inner_nary(root_inner)?;
        let shape: [usize; 4] = root.shape.as_ref().try_into().ok()?;
        let [batch, num_heads, q_seq_len, kv_seq_len] = shape;
        let datatype = root.output_datatype;
        let mut causal = false;
        let mut mask = None;
        let mut scores_inner = root_inner;
        if let Some(scaled) = self.inner_nary(scores_inner).and_then(match_causal_select) {
            causal = true;
            scores_inner = scaled;
        } else if let Some((NaryOp::Add, lhs, rhs)) =
            self.inner_nary(scores_inner).and_then(binary_elementwise)
        {
            let mut matched = None;
            for (scaled_side, mask_side) in [(lhs, rhs), (rhs, lhs)] {
                let Some(view) = self.inner_view(mask_side) else {
                    continue;
                };
                let Some(stage) = view.plain().filter(|stage| stage.is_fully_defined()) else {
                    continue;
                };
                let expected =
                    Layout::from_parts(0, shape.to_vec().into(), [0, 0, kv_seq_len, 1].into());
                if !layout_matches(Some(&stage.layout), &expected)
                    || stage.input_shape.as_ref() != [q_seq_len, kv_seq_len]
                {
                    continue;
                }
                matched = Some((scaled_side, view.input));
                break;
            }
            let (scaled_side, mask_node) = matched?;
            mask = Some(mask_node);
            scores_inner = scaled_side;
        }
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
        if q_shape.len() != 4 || q_shape[..3] != [batch, num_heads, q_seq_len] {
            return None;
        }
        let head_dim = q_shape[3];
        let expanded_shape = [batch, num_heads, kv_seq_len, head_dim];
        let kt = self.inner_view(qk.second)?;
        let kt_stage = kt.plain().filter(|stage| stage.is_fully_defined())?;
        let expected_kt = Layout::contiguous(&expanded_shape).transpose(2, 3);
        if !layout_matches(Some(&kt_stage.layout), &expected_kt)
            || kt_stage.input_shape.as_ref() != expanded_shape
        {
            return None;
        }
        let (k, k_shape) = self.peel_gqa_expand(graph, kt.input, &expanded_shape)?;
        Some(MatchedScores {
            q,
            k,
            mask,
            causal,
            scale,
            shape,
            head_dim,
            kv_heads: k_shape[1],
            datatype,
        })
    }

    /// Peel a keepdim row statistic broadcast back over `shape`, returning
    /// the rank-3 base node.
    fn peel_row_broadcast(&self, view_inner: NodeIndex, shape: &[usize; 4]) -> Option<NodeIndex> {
        let (base, layout) = self.walk_view_chain(view_inner);
        if base == view_inner {
            return None;
        }
        layout_matches(layout.as_ref(), &keepdim_broadcast_layout(shape, 3)).then_some(base)
    }

    /// Probabilities recomputed from a row statistic:
    /// `exp(scores − bcast(rowstat))`. Returns the score cluster and the
    /// statistic node.
    fn match_prob_cluster(
        &self,
        graph: &ComputeGraphInner,
        prob_inner: NodeIndex,
    ) -> Option<(MatchedScores, NodeIndex)> {
        let shifted_inner = self.match_unary(prob_inner, |function| function.op == NaryOp::Exp)?;
        let (sub_op, scores_root, stat_view) = self
            .inner_nary(shifted_inner)
            .and_then(binary_elementwise)?;
        if sub_op != NaryOp::Sub {
            return None;
        }
        let scores = self.match_score_cluster(graph, scores_root)?;
        let stat = self.peel_row_broadcast(stat_view, &scores.shape)?;
        Some((scores, stat))
    }

    /// Match `p ∘ (grad_o·vᵀ − bcast(dsum)) · scale` rooted at `ds_inner`.
    fn match_ds_cluster(
        &self,
        graph: &ComputeGraphInner,
        ds_inner: NodeIndex,
    ) -> Option<MatchedDs> {
        let ds_scale = {
            let nary = self.inner_nary(ds_inner)?;
            let (function, _) = unary_elementwise(nary)?;
            match function.op {
                NaryOp::MulConst(NaryScalar::F32(scale)) => scale,
                NaryOp::MulConst(NaryScalar::F16(scale)) => scale.to_f32(),
                _ => return None,
            }
        };
        let mul_inner = self.match_unary(ds_inner, |function| {
            matches!(function.op, NaryOp::MulConst(_))
        })?;
        let (mul_op, lhs, rhs) = self.inner_nary(mul_inner).and_then(binary_elementwise)?;
        if mul_op != NaryOp::Mul {
            return None;
        }
        for (p_side, sub_side) in [(lhs, rhs), (rhs, lhs)] {
            let Some((scores, lse)) = self.match_prob_cluster(graph, p_side) else {
                continue;
            };
            if scores.scale != ds_scale {
                continue;
            }
            let Some((NaryOp::Sub, dp_root, dsum_view)) =
                self.inner_nary(sub_side).and_then(binary_elementwise)
            else {
                continue;
            };
            let Some(dsum) = self.peel_row_broadcast(dsum_view, &scores.shape) else {
                continue;
            };
            let Some(dp) = self.inner_matmul(dp_root) else {
                continue;
            };
            if !dp.pre_element_wise[0].functions.is_empty()
                || !dp.pre_element_wise[1].functions.is_empty()
                || !dp.post_element_wise.functions.is_empty()
                || !dp.a.is_plain()
                || !dp.b.is_plain()
            {
                continue;
            }
            let [batch, heads, _q_len, kv_len] = scores.shape;
            let expanded = [batch, heads, kv_len, scores.head_dim];
            let Some(vt) = self.inner_view(dp.second) else {
                continue;
            };
            let Some(vt_stage) = vt.plain().filter(|stage| stage.is_fully_defined()) else {
                continue;
            };
            let expected = Layout::contiguous(&expanded).transpose(2, 3);
            if !layout_matches(Some(&vt_stage.layout), &expected)
                || vt_stage.input_shape.as_ref() != expanded
            {
                continue;
            }
            return Some(MatchedDs {
                scores,
                lse,
                dsum,
                grad_o: dp.first,
                v: vt.input,
            });
        }
        None
    }

    /// A transpose view over the score space `[b, h, q, kv] → [b, h, kv, q]`,
    /// returning the viewed node.
    fn peel_score_transpose(&self, inner: NodeIndex, shape: &[usize; 4]) -> Option<NodeIndex> {
        let view = self.inner_view(inner)?;
        let stage = view.plain().filter(|stage| stage.is_fully_defined())?;
        let expected = Layout::contiguous(shape).transpose(2, 3);
        (layout_matches(Some(&stage.layout), &expected) && stage.input_shape.as_ref() == shape)
            .then_some(view.input)
    }

    /// Recognize the probability-contraction patterns rooted at a matmul:
    /// `ds·k`, `dsᵀ·q`, and `pᵀ·x` all stream tile recomputation from the
    /// row statistics instead of materializing the probability matrices.
    fn try_recognize_attention_grad(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        use crate::flash_attention::{AttentionKernel, AttentionPatternNodes};
        let ExecutionVariant::MatMul(out) = &self.execution_graph[node_idx].variant else {
            return false;
        };
        if !out.pre_element_wise[0].functions.is_empty()
            || !out.pre_element_wise[1].functions.is_empty()
            || !out.post_element_wise.functions.is_empty()
            || !out.a.is_plain()
            || !out.b.is_plain()
        {
            return false;
        }
        let (a_inner, b_inner) = (out.first, out.second);
        let b_shape = out.b.shape.to_vec();
        let datatype = out.datatype;

        // dq-shaped: MatMul(ds, k).
        if let Some(ds) = self.match_ds_cluster(graph, a_inner)
            && b_inner == ds.scores.k
            && datatype == ds.scores.datatype
        {
            let nodes = AttentionPatternNodes {
                q: ds.scores.q,
                k: ds.scores.k,
                v: Some(ds.v),
                grad_o: Some(ds.grad_o),
                lse: Some(ds.lse),
                dsum: Some(ds.dsum),
                mask: ds.scores.mask,
            };
            if let Some(operation) = crate::flash_attention::FlashAttentionOperation::try_new(
                &graph.device(),
                AttentionKernel::GradQ,
                nodes,
                ds.scores.dims(),
                ds.scores.scale,
                ds.scores.causal,
                datatype,
            ) {
                let dependencies = grad_dependencies(&operation);
                self.commit_recognized(
                    graph,
                    node_idx,
                    &dependencies,
                    ExecutionVariant::Attention(operation),
                );
                return true;
            }
        }

        // Transposed-operand shapes: MatMul(dsᵀ, q) and MatMul(pᵀ, x).
        let Some(ds_shape) = self.transposed_operand_shape(a_inner) else {
            return false;
        };
        let Some(src) = self.peel_score_transpose(a_inner, &ds_shape) else {
            return false;
        };
        if let Some(ds) = self.match_ds_cluster(graph, src)
            && ds.scores.shape == ds_shape
            && b_inner == ds.scores.q
            && datatype == ds.scores.datatype
        {
            let nodes = AttentionPatternNodes {
                q: ds.scores.q,
                k: ds.scores.k,
                v: Some(ds.v),
                grad_o: Some(ds.grad_o),
                lse: Some(ds.lse),
                dsum: Some(ds.dsum),
                mask: ds.scores.mask,
            };
            if let Some(operation) = crate::flash_attention::FlashAttentionOperation::try_new(
                &graph.device(),
                AttentionKernel::GradK,
                nodes,
                ds.scores.dims(),
                ds.scores.scale,
                ds.scores.causal,
                datatype,
            ) {
                let dependencies = grad_dependencies(&operation);
                self.commit_recognized(
                    graph,
                    node_idx,
                    &dependencies,
                    ExecutionVariant::Attention(operation),
                );
                return true;
            }
        }
        if let Some((scores, lse)) = self.match_prob_cluster(graph, src)
            && scores.shape == ds_shape
            && datatype == scores.datatype
            && b_shape.len() == 4
            && b_shape[3] == scores.head_dim
        {
            let nodes = AttentionPatternNodes {
                q: scores.q,
                k: scores.k,
                v: None,
                grad_o: Some(b_inner),
                lse: Some(lse),
                dsum: None,
                mask: scores.mask,
            };
            if let Some(operation) = crate::flash_attention::FlashAttentionOperation::try_new(
                &graph.device(),
                AttentionKernel::GradV,
                nodes,
                scores.dims(),
                scores.scale,
                scores.causal,
                datatype,
            ) {
                let dependencies = grad_dependencies(&operation);
                self.commit_recognized(
                    graph,
                    node_idx,
                    &dependencies,
                    ExecutionVariant::Attention(operation),
                );
                return true;
            }
        }
        false
    }

    /// The `[b, h, q, kv]` score space a transposed matmul operand views,
    /// derived from the view node's stage.
    fn transposed_operand_shape(&self, inner: NodeIndex) -> Option<[usize; 4]> {
        let view = self.inner_view(inner)?;
        let stage = view.plain().filter(|stage| stage.is_fully_defined())?;
        stage.input_shape.as_ref().try_into().ok()
    }

    /// Recognize the row log-sum-exp pattern rooted at an elementwise add:
    /// `max(s, 3) + ln Σ exp(s − bcast(max))` streams the scores without
    /// materializing them.
    fn try_recognize_score_lse(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        use crate::flash_attention::{AttentionKernel, AttentionPatternNodes};
        let ExecutionVariant::Elementwise(add) = &self.execution_graph[node_idx].variant else {
            return false;
        };
        let Some((NaryOp::Add, lhs, rhs)) = binary_elementwise(add) else {
            return false;
        };
        let datatype = add.output_datatype;
        for (m_side, log_side) in [(lhs, rhs), (rhs, lhs)] {
            let Some((max_axis, max_value)) =
                self.match_reduce(m_side, crate::reduce::ReduceOp::Max)
            else {
                continue;
            };
            if max_axis != 3 {
                continue;
            }
            let Some(sum_inner) = self.match_unary(log_side, |function| function.op == NaryOp::Log)
            else {
                continue;
            };
            let Some((sum_axis, exp_inner)) =
                self.match_reduce(sum_inner, crate::reduce::ReduceOp::Sum)
            else {
                continue;
            };
            if sum_axis != 3 {
                continue;
            }
            let Some(shifted) = self.match_unary(exp_inner, |function| function.op == NaryOp::Exp)
            else {
                continue;
            };
            let Some((NaryOp::Sub, scores_root, max_view)) =
                self.inner_nary(shifted).and_then(binary_elementwise)
            else {
                continue;
            };
            if scores_root != max_value {
                continue;
            }
            let Some(scores) = self.match_score_cluster(graph, scores_root) else {
                continue;
            };
            let Some(m_base) = self.peel_row_broadcast(max_view, &scores.shape) else {
                continue;
            };
            if m_base != m_side || datatype != scores.datatype {
                continue;
            }
            let nodes = AttentionPatternNodes {
                q: scores.q,
                k: scores.k,
                v: None,
                grad_o: None,
                lse: None,
                dsum: None,
                mask: scores.mask,
            };
            if let Some(operation) = crate::flash_attention::FlashAttentionOperation::try_new(
                &graph.device(),
                AttentionKernel::LogSumExp,
                nodes,
                scores.dims(),
                scores.scale,
                scores.causal,
                datatype,
            ) {
                let dependencies = grad_dependencies(&operation);
                self.commit_recognized(
                    graph,
                    node_idx,
                    &dependencies,
                    ExecutionVariant::Attention(operation),
                );
                return true;
            }
        }
        false
    }
}

/// The operation's dependencies in `visit_dependencies` order.
fn grad_dependencies(
    operation: &crate::flash_attention::FlashAttentionOperation,
) -> Vec<NodeIndex> {
    use crate::mir::operation::Operation;
    let mut dependencies = Vec::new();
    operation.visit_dependencies(&mut |node| dependencies.push(node));
    dependencies
}

impl Resolver {
    /// Match the dk-shaped contraction `dsᵀ · q` at a matmul node.
    fn match_grad_k_root(&self, graph: &ComputeGraphInner, inner: NodeIndex) -> Option<MatchedDs> {
        let matmul = self.inner_matmul(inner)?;
        if !matmul.pre_element_wise[0].functions.is_empty()
            || !matmul.pre_element_wise[1].functions.is_empty()
            || !matmul.post_element_wise.functions.is_empty()
            || !matmul.a.is_plain()
            || !matmul.b.is_plain()
        {
            return None;
        }
        let (a_inner, b_inner, datatype) = (matmul.first, matmul.second, matmul.datatype);
        let shape = self.transposed_operand_shape(a_inner)?;
        let src = self.peel_score_transpose(a_inner, &shape)?;
        let ds = self.match_ds_cluster(graph, src)?;
        (ds.scores.shape == shape && b_inner == ds.scores.q && datatype == ds.scores.datatype)
            .then_some(ds)
    }

    /// Match the dv-shaped contraction `pᵀ · x` at a matmul node, returning
    /// the score cluster, the row statistic, and the free operand.
    fn match_grad_v_root(
        &self,
        graph: &ComputeGraphInner,
        inner: NodeIndex,
    ) -> Option<(MatchedScores, NodeIndex, NodeIndex)> {
        let matmul = self.inner_matmul(inner)?;
        if !matmul.pre_element_wise[0].functions.is_empty()
            || !matmul.pre_element_wise[1].functions.is_empty()
            || !matmul.post_element_wise.functions.is_empty()
            || !matmul.a.is_plain()
            || !matmul.b.is_plain()
        {
            return None;
        }
        let (a_inner, b_inner, datatype) = (matmul.first, matmul.second, matmul.datatype);
        let b_shape = matmul.b.shape.to_vec();
        let shape = self.transposed_operand_shape(a_inner)?;
        let src = self.peel_score_transpose(a_inner, &shape)?;
        let (scores, lse) = self.match_prob_cluster(graph, src)?;
        (scores.shape == shape
            && datatype == scores.datatype
            && b_shape.len() == 4
            && b_shape[3] == scores.head_dim)
            .then_some((scores, lse, b_inner))
    }

    /// Recognize both KV-side contractions landing in one combined tensor
    /// (`dk` rows then `dv` rows along the sequence axis via a slice-assign
    /// chain over a zero base): the paired streaming kernel computes both in
    /// one dispatch, sharing the probability recomputation. Composed
    /// slice-assigns are elementwise region-selects over
    /// `[destination, value]` (see `recognize_cat`).
    fn try_recognize_attention_grad_pair(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        use crate::flash_attention::{AttentionKernel, AttentionPatternNodes};
        struct Link {
            destination: NodeIndex,
            value: NodeIndex,
            slices: Box<[std::ops::Range<usize>]>,
        }
        let link_of = |resolver: &Self, inner: NodeIndex| -> Option<Link> {
            let nary = resolver.inner_nary(inner)?;
            let slices = super::recognize_cat::match_slice_assign(nary)?;
            Some(Link {
                destination: nary.inputs[0],
                value: nary.inputs[1],
                slices,
            })
        };
        let outer_inner = match &self.execution_graph[node_idx].variant {
            ExecutionVariant::Elementwise(_) => self.execution_graph[node_idx].inner_idx,
            _ => return false,
        };
        let Some(outer) = link_of(self, outer_inner) else {
            return false;
        };
        let Some(inner) = link_of(self, outer.destination) else {
            return false;
        };
        let ds = match self.match_grad_k_root(graph, inner.value) {
            Some(ds) => ds,
            None => return false,
        };
        let (scores_v, lse_v, x) = match self.match_grad_v_root(graph, outer.value) {
            Some(matched) => matched,
            None => return false,
        };
        // Both halves must come from the same cluster identity.
        if scores_v.q != ds.scores.q
            || scores_v.k != ds.scores.k
            || scores_v.mask != ds.scores.mask
            || scores_v.causal != ds.scores.causal
            || scores_v.scale != ds.scores.scale
            || lse_v != ds.lse
            || x != ds.grad_o
        {
            return false;
        }
        // The chain must be exactly `zeros -> assign dk half -> assign dv
        // half` with the halves in kernel order (dk rows first).
        let [batch, heads, _q_len, kv_len] = ds.scores.shape;
        let head_dim = ds.scores.head_dim;
        let expect_inner: Box<[std::ops::Range<usize>]> =
            [0..batch, 0..heads, 0..kv_len, 0..head_dim].into();
        let expect_outer: Box<[std::ops::Range<usize>]> =
            [0..batch, 0..heads, kv_len..2 * kv_len, 0..head_dim].into();
        if inner.slices != expect_inner || outer.slices != expect_outer {
            return false;
        }
        // Intermediates die once the root is rewired: the inner assign must
        // feed only the outer, and the zero base only the inner.
        if !self.exclusively_consumed(graph, outer.destination, 1)
            || !self.exclusively_consumed(graph, inner.destination, 1)
        {
            return false;
        }
        let nodes = AttentionPatternNodes {
            q: ds.scores.q,
            k: ds.scores.k,
            v: Some(ds.v),
            grad_o: Some(ds.grad_o),
            lse: Some(ds.lse),
            dsum: Some(ds.dsum),
            mask: ds.scores.mask,
        };
        if let Some(operation) = crate::flash_attention::FlashAttentionOperation::try_new(
            &graph.device(),
            AttentionKernel::GradKV,
            nodes,
            ds.scores.dims(),
            ds.scores.scale,
            ds.scores.causal,
            ds.scores.datatype,
        ) {
            let dependencies = grad_dependencies(&operation);
            self.commit_recognized(
                graph,
                node_idx,
                &dependencies,
                ExecutionVariant::Attention(operation),
            );
            return true;
        }
        false
    }
}
