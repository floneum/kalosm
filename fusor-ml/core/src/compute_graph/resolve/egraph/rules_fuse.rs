//! Stage-2 fusion generators: the per-node fusion rules (view folding, nary
//! inlining, the reduce-fusion family, matmul/qmatmul epilogues), consulted
//! by the extraction worklist with live consumer counts.
//!
//! Each generator is a pure function from the node's current form (and the
//! evolving selection state, read through [`FusionView`]) to a better form;
//! the extractor commits it as a switch and cascades the kills. Gates are
//! legality and profitability conditions: binding budgets, dtype and device
//! capabilities, sole-consumer duplication checks against live counts.

use std::cell::RefCell;

use rustc_hash::FxHashSet;

use super::super::{ExecutionVariant, Resolver, fold_views::input_reread_factor};
use super::EGraphDriver;
use super::extract::{ExtractState, Selection};
use super::lang::Prov;
use crate::compute_graph::layout_pass::LayoutPass;
use crate::compute_graph::{ComputeGraphInner, NodeIndex};
use crate::nary_wise::{ElementwiseOperation, NaryExpr, NaryFunction, UnaryFunctionChain};
use crate::{DataTypeEnum, Layout};

/// Which nodes seed and re-enter the fusion worklist. Transcribes
/// `CandidateProfile` (execution.rs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::compute_graph::resolve) enum CandidateKind {
    /// Elementwise | MatMul | QMatMul | Reduce (`is_optimization_candidate`).
    General,
    /// Elementwise | Reduce (`is_dense_graph_candidate`).
    Dense,
    /// Large quantized graphs (`is_large_graph_nary_candidate`).
    LargeQuantized,
}

/// Which reduce-fusion generators run (`ReduceFusionProfile`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::compute_graph::resolve) enum ReduceFusion {
    Disabled,
    Conservative,
    Dense,
}

/// Stage-2 fusion policy, selected per graph by `optimize()`: which nodes
/// seed the worklist, which rule families run, and the decode-protecting
/// gates (reduce fusion stays off for quantized graphs; qmatmul epilogue
/// fusion is node-count-bounded on the standard profile).
#[derive(Debug, Clone)]
pub(in crate::compute_graph::resolve) struct Stage2Profile {
    pub(in crate::compute_graph::resolve) candidates: CandidateKind,
    pub(in crate::compute_graph::resolve) reduce_fusion: ReduceFusion,
    pub(in crate::compute_graph::resolve) try_matmul_fusion: bool,
    pub(in crate::compute_graph::resolve) allow_qmatmul_elementwise_fusion: bool,
    /// `dense` in the fixpoint: allow indexed inlining in nary/reduce fusion.
    pub(in crate::compute_graph::resolve) dense: bool,
    /// The dense-region branch: externally-live producers are not inlined
    /// (region formation emits them as region outputs instead).
    pub(in crate::compute_graph::resolve) skip_externally_live: bool,
    /// Dense large-graph kernel tuning (region formation +
    /// `mark_dense_codegen`) runs after fusion settles.
    pub(in crate::compute_graph::resolve) enable_dense_codegen: bool,
}

pub(super) struct Stage2Ctx<'a> {
    pub(super) graph: &'a ComputeGraphInner,
    pub(super) profile: Stage2Profile,
    /// Memoized layout inference over the (immutable) inner graph, used by
    /// qmatmul extra normalization. Fresh per stage; recomputation is
    /// correctness-neutral.
    pub(super) layouts: RefCell<LayoutPass>,
}

/// Read-only view of the evolving optimization state, mirroring exactly what
/// the destructive fixpoint reads from the execution graph mid-rewrite.
pub(super) struct FusionView<'a> {
    driver: &'a EGraphDriver,
    state: &'a ExtractState,
    ctx: &'a Stage2Ctx<'a>,
}

impl<'a> FusionView<'a> {
    pub(super) fn new(
        driver: &'a EGraphDriver,
        state: &'a ExtractState,
        ctx: &'a Stage2Ctx<'a>,
    ) -> Self {
        Self { driver, state, ctx }
    }

    /// The current form of an inner node: its selection's payload. `None`
    /// mirrors every case where the destructive code bails — not in the
    /// execution graph (cached boundary / tensor input handled separately),
    /// or already killed.
    pub(super) fn variant_of(&self, inner: NodeIndex) -> Option<&ExecutionVariant> {
        let prov = *self.driver.prov_of.get(&inner)?;
        let facts = self.driver.egraph.analysis.facts_of(prov);
        facts.exec?;
        if !self.state.needed[prov.0 as usize] {
            return None;
        }
        match &self.state.sel[prov.0 as usize] {
            Selection::Identity => self.driver.identity_variant(prov),
            Selection::Alt(enode) => enode
                .payload()
                .map(|payload| self.driver.egraph.analysis.payloads.get(payload)),
        }
    }

    /// `check_cached` equivalent: the node was cached when the resolve
    /// started (ingested as an opaque boundary).
    pub(super) fn is_cached(&self, inner: NodeIndex) -> bool {
        self.driver
            .prov_of
            .get(&inner)
            .is_some_and(|&prov| self.driver.egraph.analysis.facts_of(prov).exec.is_none())
    }

    /// Live consumer count (the destructive `neighbors_directed(..).count()`).
    pub(super) fn consumer_count(&self, inner: NodeIndex) -> u32 {
        self.driver
            .prov_of
            .get(&inner)
            .map(|&prov| self.state.reads[prov.0 as usize])
            .unwrap_or(0)
    }

    fn externally_live(&self, inner: NodeIndex) -> bool {
        self.ctx
            .graph
            .nodes
            .nodes
            .node_weight(inner)
            .is_some_and(|node| node.reference_count > 0)
    }

    pub(super) fn device(&self) -> crate::Device {
        self.ctx.graph.device()
    }

    /// The profile's `allow_qmatmul_elementwise_fusion` flag, exposed for the
    /// matmul-family generators in `rules_fuse_matmul.rs`.
    pub(super) fn allow_qmatmul_elementwise_fusion(&self) -> bool {
        self.ctx.profile.allow_qmatmul_elementwise_fusion
    }

    /// Transcription of `Resolver::walk_view_chain` over current selections.
    pub(super) fn walk_view_chain(&self, mut inner: NodeIndex) -> (NodeIndex, Option<Layout>) {
        let mut composed: Option<Layout> = None;
        loop {
            let Some(ExecutionVariant::View(view)) = self.variant_of(inner) else {
                return (inner, composed);
            };
            let Some(collapsed) = view.composed_layout() else {
                return (inner, composed);
            };
            let next = match &composed {
                None => collapsed,
                Some(outer) => match crate::view::compose_layouts(outer, &collapsed) {
                    Some(layout) => layout,
                    None => return (inner, composed),
                },
            };
            composed = Some(next);
            inner = view.input;
        }
    }

    pub(super) fn layout_of(&self, inner: NodeIndex) -> Option<crate::TensorLayoutInfo> {
        let mut layouts = self.ctx.layouts.borrow_mut();
        layouts.visit(self.ctx.graph, inner);
        layouts.output_layout.get(&inner).cloned()
    }

    /// Transcription of `Resolver::try_normalize_qmatmul_post_extra`.
    pub(super) fn normalize_qmatmul_post_extra(
        &self,
        extra_inner: NodeIndex,
        output_shape: &[usize],
    ) -> Option<NodeIndex> {
        let last_dim = *output_shape.last()?;
        let extra_info = self.layout_of(extra_inner)?;
        if extra_info.datatype() != DataTypeEnum::F32 || extra_info.layout().shape() != output_shape
        {
            return None;
        }
        let layout = extra_info.layout();
        let is_column_broadcast = layout.offset() == 0
            && layout.strides().last().copied() == Some(1)
            && layout.shape().last().copied() == Some(last_dim)
            && layout.strides()[..layout.strides().len().saturating_sub(1)]
                .iter()
                .all(|stride| *stride == 0);
        if !is_column_broadcast {
            return Some(extra_inner);
        }
        let (base_inner, _) = self.walk_view_chain(extra_inner);
        let base_info = self.layout_of(base_inner)?;
        let base_layout = base_info.layout();
        if base_info.datatype() == DataTypeEnum::F32
            && base_layout.shape() == [last_dim]
            && base_layout.is_contiguous()
            && base_layout.offset() == 0
        {
            Some(base_inner)
        } else {
            Some(extra_inner)
        }
    }

    /// `CandidateProfile::matches` transcription over current selections.
    pub(super) fn is_seed_candidate(&self, prov: Prov) -> bool {
        let facts = self.driver.egraph.analysis.facts_of(prov);
        if facts.exec.is_none() || !self.state.needed[prov.0 as usize] {
            return false;
        }
        let Some(variant) = self.variant_of(facts.inner) else {
            return false;
        };
        match self.ctx.profile.candidates {
            CandidateKind::General => matches!(
                variant,
                ExecutionVariant::Elementwise(_)
                    | ExecutionVariant::MatMul(_)
                    | ExecutionVariant::QMatMul(_)
                    | ExecutionVariant::Reduce(_)
            ),
            CandidateKind::Dense => matches!(
                variant,
                ExecutionVariant::Elementwise(_) | ExecutionVariant::Reduce(_)
            ),
            CandidateKind::LargeQuantized => {
                let ExecutionVariant::Elementwise(nary) = variant else {
                    return false;
                };
                if nary.shape.last().copied().unwrap_or_default()
                    >= super::super::LARGE_GRAPH_NARY_FUSION_MIN_LAST_DIM
                {
                    return true;
                }
                nary.inputs.iter().any(|&input| {
                    let (base_inner, _) = self.walk_view_chain(input);
                    matches!(
                        self.variant_of(base_inner),
                        Some(ExecutionVariant::QMatMul(_))
                    )
                })
            }
        }
    }

    /// `enqueue_downstream_candidates` transcription: enqueue candidates
    /// reachable from `seeds`, descending through view nodes.
    pub(super) fn enqueue_downstream(
        &self,
        state: &ExtractState,
        seeds: impl IntoIterator<Item = u32>,
        worklist: &mut std::collections::VecDeque<u32>,
        queued: &mut [bool],
    ) {
        let mut stack: Vec<u32> = seeds.into_iter().collect();
        let mut visited = FxHashSet::default();
        while let Some(prov) = stack.pop() {
            if !state.needed[prov as usize] || !visited.insert(prov) {
                continue;
            }
            if self.is_seed_candidate(Prov(prov)) {
                if !queued[prov as usize] {
                    queued[prov as usize] = true;
                    worklist.push_back(prov);
                }
            } else {
                let facts = &self.driver.egraph.analysis.facts[prov as usize];
                if matches!(
                    self.variant_of(facts.inner),
                    Some(ExecutionVariant::View(_))
                ) {
                    stack.extend(state.consumers[prov as usize].iter().copied());
                }
            }
        }
    }

    /// The per-pop generator pipeline in the fixpoint's attempt order: fold
    /// views, fuse naries, reduce-fusion family, matmul fusion. First
    /// success wins; the extractor commits it and re-enqueues, so composed
    /// sequences (fold-then-fuse in one destructive pop) play out across
    /// consecutive pops with identical results.
    pub(super) fn generate(&self, prov: Prov) -> Option<ExecutionVariant> {
        let facts = self.driver.egraph.analysis.facts_of(prov);
        let current = self.variant_of(facts.inner)?.clone();
        match &current {
            ExecutionVariant::Elementwise(nary) => {
                if let Some(folded) = self.gen_fold_views_elementwise(nary) {
                    return Some(folded);
                }
                if let Some(fused) = self.gen_fuse_naries(nary) {
                    return Some(fused);
                }
            }
            ExecutionVariant::Reduce(_) => {}
            _ => {}
        }
        if let Some(reduced) = self.gen_fuse_reduce(&current) {
            return Some(reduced);
        }
        if self.ctx.profile.try_matmul_fusion
            && let Some(matmul) = self.gen_fuse_into_matmul(&current)
        {
            return Some(matmul);
        }
        None
    }

    /// Transcription of `try_fold_view_inputs` (fold_views.rs:19-91).
    fn gen_fold_views_elementwise(&self, nary: &ElementwiseOperation) -> Option<ExecutionVariant> {
        let (final_inputs, final_expression) =
            self.fold_view_inputs(&nary.inputs, &nary.expression, &nary.shape)?;
        Some(ExecutionVariant::Elementwise(ElementwiseOperation {
            inputs: final_inputs,
            expression: final_expression,
            shape: nary.shape.clone(),
            output_datatype: nary.output_datatype,
        }))
    }

    /// Shared body of the two view-folding rewrites (elementwise + reduce).
    fn fold_view_inputs(
        &self,
        inputs: &[NodeIndex],
        expression: &NaryExpr,
        shape: &[usize],
    ) -> Option<(Vec<NodeIndex>, NaryExpr)> {
        let mut expression = expression.clone();
        let mut inputs = inputs.to_vec();
        let mut folded = false;
        for (slot, input_inner) in inputs.clone().into_iter().enumerate() {
            if self.is_cached(input_inner) {
                continue;
            }
            let Some(ExecutionVariant::View(view)) = self.variant_of(input_inner) else {
                continue;
            };
            let needs_delinearize = view.stages.iter().any(|stage| {
                crate::view::affine_dim_indices(&stage.layout, &stage.input_shape).is_none()
            });
            if needs_delinearize && input_reread_factor(&expression, shape, slot) > 1 {
                continue;
            }
            let view = view.clone();
            let Some(rewritten) = Resolver::rewrite_view_input(&expression, slot, &view) else {
                continue;
            };
            expression = rewritten;
            inputs[slot] = view.input;
            folded = true;
        }
        if !folded {
            return None;
        }
        Some(Resolver::deduplicate_inputs(inputs, expression))
    }

    /// Transcription of `try_fuse_naries` (fusion_basic.rs:6-170).
    fn gen_fuse_naries(&self, nary: &ElementwiseOperation) -> Option<ExecutionVariant> {
        let allow_indexed_inline = self.ctx.profile.dense;
        let mut expression = nary.expression.clone();
        let mut all_inputs = nary.inputs.clone();
        let mut fused_any = false;
        let max_fused_inputs = self.device().nary_direct_input_binding_budget();

        for &input_inner in nary.inputs.iter() {
            if self.is_cached(input_inner) {
                continue;
            }
            // Dense branch: an externally live producer (pending sink /
            // user-held node) materializes regardless, so inlining it here
            // would duplicate its compute. Region formation fuses it with
            // its consumers instead, emitting it as a region output.
            if self.ctx.profile.skip_externally_live && self.externally_live(input_inner) {
                continue;
            }
            // Inlining duplicates the producer's work unless this node is
            // its only consumer. A user-held reference alone doesn't block
            // fusion — only another consumer in this resolve does.
            if self.consumer_count(input_inner) != 1 {
                continue;
            }
            let Some(ExecutionVariant::Elementwise(input_nary)) = self.variant_of(input_inner)
            else {
                continue;
            };
            let offset = all_inputs.len();
            let inlined = Resolver::offset_input_indices(&input_nary.expression, offset);
            let target_slots: Vec<usize> = all_inputs
                .iter()
                .enumerate()
                .filter_map(|(slot, value)| (*value == input_inner).then_some(slot))
                .collect();
            let mut new_expression = expression.clone();
            let mut success = true;
            for slot in &target_slots {
                let (next, s) =
                    Resolver::substitute_input_in_expr(&new_expression, *slot, &inlined);
                new_expression = next;
                success &= s;
            }
            if !success
                && allow_indexed_inline
                && target_slots
                    .iter()
                    .all(|&slot| input_reread_factor(&expression, &nary.shape, slot) == 1)
            {
                let mut composed = expression.clone();
                success = true;
                for slot in &target_slots {
                    match Resolver::substitute_input_composed(&composed, *slot, &inlined) {
                        Some(next) => composed = next,
                        None => {
                            success = false;
                            break;
                        }
                    }
                }
                if success {
                    new_expression = composed;
                }
            }

            if success {
                let unique_inputs: FxHashSet<_> = all_inputs
                    .iter()
                    .chain(input_nary.inputs.iter())
                    .copied()
                    .collect();
                if unique_inputs.len() > max_fused_inputs {
                    continue;
                }
                expression = new_expression;
                all_inputs.extend(input_nary.inputs.iter().copied());
                fused_any = true;
            }
        }
        if !fused_any {
            return None;
        }
        let (final_inputs, final_expression) = Resolver::deduplicate_inputs(all_inputs, expression);
        Some(ExecutionVariant::Elementwise(ElementwiseOperation {
            inputs: final_inputs,
            expression: final_expression,
            shape: nary.shape.clone(),
            output_datatype: nary.output_datatype,
        }))
    }

    /// Transcription of `try_fuse_reduce` (execution.rs:558-578).
    fn gen_fuse_reduce(&self, current: &ExecutionVariant) -> Option<ExecutionVariant> {
        match self.ctx.profile.reduce_fusion {
            ReduceFusion::Disabled => None,
            ReduceFusion::Conservative => self
                .gen_unary_into_reduce(current)
                .or_else(|| self.gen_producer_into_reduce(current, false)),
            ReduceFusion::Dense => self
                .gen_collapse_unit_reduce(current)
                .or_else(|| self.gen_fold_views_into_reduce(current))
                .or_else(|| self.gen_unary_into_reduce(current))
                .or_else(|| self.gen_indexed_unary_into_reduce(current))
                .or_else(|| self.gen_producer_into_reduce(current, true)),
        }
    }

    /// Transcription of `try_fuse_into_reduce` (fusion_basic.rs:525-571).
    fn gen_unary_into_reduce(&self, current: &ExecutionVariant) -> Option<ExecutionVariant> {
        let el_op = Resolver::try_get_unary_chain(current)?;
        let input_inner = el_op.value;
        if self.is_cached(input_inner) {
            return None;
        }
        let ExecutionVariant::Reduce(reduce_op) = self.variant_of(input_inner)? else {
            return None;
        };
        let mut new_reduce = reduce_op.clone();
        let mut existing_post = new_reduce.post_element_wise.functions.clone();
        existing_post.extend(el_op.functions.functions.iter().cloned());
        new_reduce.post_element_wise =
            UnaryFunctionChain::new(existing_post, reduce_op.post_element_wise.input_datatype());
        Some(ExecutionVariant::Reduce(new_reduce))
    }

    /// Transcription of `try_collapse_unit_reduce` (fusion_basic.rs:582-652).
    fn gen_collapse_unit_reduce(&self, current: &ExecutionVariant) -> Option<ExecutionVariant> {
        let ExecutionVariant::Reduce(reduce) = current else {
            return None;
        };
        if reduce.shape[reduce.axis] != 1
            || !reduce.post_element_wise.functions.is_empty()
            || reduce.function.datatype() != reduce.out_datatype()
        {
            return None;
        }
        let mut mapping = Vec::with_capacity(reduce.shape.len());
        let mut out_pos = 0;
        for dim in 0..reduce.shape.len() {
            if dim == reduce.axis {
                mapping.push(NaryExpr::Scalar(crate::nary_wise::NaryScalar::U32(0)));
            } else {
                mapping.push(NaryExpr::DimIndex(out_pos));
                out_pos += 1;
            }
        }
        let expression = Resolver::compose_expr_with_indices(&reduce.expression, &mapping)?;
        use crate::reduce::ReduceOp;
        let init = reduce.function.initial_value;
        let fold_op = match reduce.function.op {
            ReduceOp::Sum => crate::nary_wise::NaryOp::AddConst(init),
            ReduceOp::Product => crate::nary_wise::NaryOp::MulConst(init),
            ReduceOp::Max => crate::nary_wise::NaryOp::MaxConst(init),
            ReduceOp::Min => crate::nary_wise::NaryOp::MinConst(init),
        };
        let dtype = reduce.function.datatype();
        let expression = NaryExpr::Op {
            children: vec![expression],
            function: NaryFunction::unary(
                Some(format!("unit_{}", reduce.function.name())),
                fold_op,
                dtype,
                dtype,
            ),
        };
        Some(ExecutionVariant::Elementwise(ElementwiseOperation {
            inputs: reduce.inputs.clone(),
            expression,
            shape: reduce.out_shape().into(),
            output_datatype: reduce.out_datatype(),
        }))
    }

    /// Transcription of `try_fold_view_inputs_into_reduce`
    /// (fold_views.rs:99-169).
    fn gen_fold_views_into_reduce(&self, current: &ExecutionVariant) -> Option<ExecutionVariant> {
        let ExecutionVariant::Reduce(reduce) = current else {
            return None;
        };
        let (final_inputs, final_expression) =
            self.fold_view_inputs(&reduce.inputs, &reduce.expression, &reduce.shape)?;
        let mut new_reduce = reduce.clone();
        new_reduce.inputs = final_inputs;
        new_reduce.expression = final_expression;
        Some(ExecutionVariant::Reduce(new_reduce))
    }

    /// Transcription of `try_fuse_unary_into_reduce_indexed`
    /// (fusion_basic.rs:663-765).
    fn gen_indexed_unary_into_reduce(
        &self,
        current: &ExecutionVariant,
    ) -> Option<ExecutionVariant> {
        let ExecutionVariant::Elementwise(nary) = current else {
            return None;
        };
        if nary.inputs.len() != 1 {
            return None;
        }
        let (functions, indices) = Resolver::extract_unary_chain_indexed(nary)?;
        let input_inner = nary.inputs[0];
        if self.is_cached(input_inner) {
            return None;
        }
        let ExecutionVariant::Reduce(reduce) = self.variant_of(input_inner)? else {
            return None;
        };
        let rows: usize = reduce
            .shape
            .iter()
            .enumerate()
            .filter_map(|(dim, &size)| (dim != reduce.axis).then_some(size))
            .product();
        if nary.shape.iter().product::<usize>() != rows || indices.len() + 1 != reduce.shape.len() {
            return None;
        }
        let mut current_dtype = reduce.out_datatype();
        for function in &functions {
            if function.input_types.as_slice() != [current_dtype] {
                return None;
            }
            current_dtype = function.output_type;
        }
        if current_dtype != nary.output_datatype {
            return None;
        }
        let node_rank = nary.shape.len();
        let mut mapping = Vec::with_capacity(reduce.shape.len());
        let mut out_pos = 0;
        for dim in 0..reduce.shape.len() {
            if dim == reduce.axis {
                mapping.push(NaryExpr::DimIndex(node_rank));
            } else {
                mapping.push(indices[out_pos].clone());
                out_pos += 1;
            }
        }
        let expression = Resolver::compose_expr_with_indices(&reduce.expression, &mapping)?;
        let mut shape: Vec<usize> = nary.shape.to_vec();
        shape.push(reduce.shape[reduce.axis]);
        let mut post = reduce.post_element_wise.functions.clone();
        post.extend(functions);
        Some(ExecutionVariant::Reduce(crate::reduce::ReduceOperation {
            inputs: reduce.inputs.clone(),
            expression,
            shape: shape.into(),
            function: reduce.function.clone(),
            post_element_wise: UnaryFunctionChain::new(
                post,
                reduce.post_element_wise.input_datatype(),
            ),
            axis: node_rank,
        }))
    }

    /// Transcription of `try_fuse_producer_into_reduce`
    /// (fusion_basic.rs:808-947).
    fn gen_producer_into_reduce(
        &self,
        current: &ExecutionVariant,
        allow_indexed_inline: bool,
    ) -> Option<ExecutionVariant> {
        let ExecutionVariant::Reduce(reduce) = current else {
            return None;
        };
        let mut expression = reduce.expression.clone();
        let mut all_inputs = reduce.inputs.clone();
        let mut fused_any = false;
        let max_fused_inputs = self.device().nary_direct_input_binding_budget();

        for &input_inner in reduce.inputs.iter() {
            if self.is_cached(input_inner) {
                continue;
            }
            if self.consumer_count(input_inner) != 1 {
                continue;
            }
            let Some(ExecutionVariant::Elementwise(input_nary)) = self.variant_of(input_inner)
            else {
                continue;
            };
            if input_nary.shape != reduce.shape && !allow_indexed_inline {
                continue;
            }
            let target_slots: Vec<usize> = all_inputs
                .iter()
                .enumerate()
                .filter_map(|(slot, value)| (*value == input_inner).then_some(slot))
                .collect();
            let offset = all_inputs.len();
            let inlined = Resolver::offset_input_indices(&input_nary.expression, offset);
            let mut new_expression = expression.clone();
            let mut success = input_nary.shape == reduce.shape;
            if success {
                for slot in &target_slots {
                    let (next, s) =
                        Resolver::substitute_input_in_expr(&new_expression, *slot, &inlined);
                    new_expression = next;
                    success &= s;
                }
            }
            if !success
                && allow_indexed_inline
                && target_slots
                    .iter()
                    .all(|&slot| input_reread_factor(&expression, &reduce.shape, slot) == 1)
            {
                let mut composed = expression.clone();
                success = true;
                for slot in &target_slots {
                    match Resolver::substitute_input_composed(&composed, *slot, &inlined) {
                        Some(next) => composed = next,
                        None => {
                            success = false;
                            break;
                        }
                    }
                }
                if success {
                    new_expression = composed;
                }
            }
            if success {
                let unique_inputs: FxHashSet<_> = all_inputs
                    .iter()
                    .chain(input_nary.inputs.iter())
                    .copied()
                    .collect();
                if unique_inputs.len() > max_fused_inputs {
                    continue;
                }
                expression = new_expression;
                all_inputs.extend(input_nary.inputs.iter().copied());
                fused_any = true;
            }
        }
        if !fused_any {
            return None;
        }
        let (final_inputs, final_expression) = Resolver::deduplicate_inputs(all_inputs, expression);
        let mut new_reduce = reduce.clone();
        new_reduce.inputs = final_inputs;
        new_reduce.expression = final_expression;
        Some(ExecutionVariant::Reduce(new_reduce))
    }

    /// Transcription of `try_fuse_into_matmul` (fusion_matmul.rs). Filled in
    /// by the matmul-family port; see `rules_fuse_matmul.rs`.
    fn gen_fuse_into_matmul(&self, current: &ExecutionVariant) -> Option<ExecutionVariant> {
        self.gen_matmul_family(current)
    }
}
