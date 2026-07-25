//! Fusion generators: per-node view folding, nary
//! inlining, the reduce-fusion family, matmul/qmatmul epilogues), consulted
//! by the extraction worklist with live consumer counts.
//!
//! Each generator is a pure function from the node's current form (and the
//! evolving selection state, read through [`FusionView`]) to a legal
//! alternative. The extractor compares alternatives with the GPU cost model,
//! commits the winner, and cascades the kills. Gates cover binding budgets,
//! dtype/device capabilities and duplication checks against live counts.

use std::cell::RefCell;

use rustc_hash::FxHashSet;

use super::super::ExecutionVariant;
use super::EGraphDriver;
use super::compose;
use super::extract::{ExtractState, Selection};
use super::lang::Prov;
use crate::compute_graph::layout_pass::LayoutPass;
use crate::compute_graph::{ComputeGraphInner, NodeIndex};
use crate::nary_wise::{ElementwiseOperation, NaryExpr, NaryFunction, UnaryFunctionChain};
use crate::{DataTypeEnum, Layout};

/// Where the two producer-inlining rewrites disagree; everything else about
/// them is shared.
struct InlineGate {
    /// A producer that materializes anyway is left alone: inlining it into an
    /// elementwise consumer duplicates its compute. Reduce consumers never see
    /// one, because region formation claims it first.
    skip_externally_live: bool,
    /// Substitute directly only when the producer spans the consumer's index
    /// space. A reduce's index space includes the reduced axis, so a
    /// differently-shaped producer must come in through the composed path.
    require_same_index_space: bool,
}

pub(super) struct FusionCtx<'a> {
    pub(super) graph: &'a ComputeGraphInner,
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
    ctx: &'a FusionCtx<'a>,
}

impl<'a> FusionView<'a> {
    pub(super) fn new(
        driver: &'a EGraphDriver,
        state: &'a ExtractState,
        ctx: &'a FusionCtx<'a>,
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

    /// [`compose::walk_view_chain`] over current selections.
    pub(super) fn walk_view_chain(&self, inner: NodeIndex) -> (NodeIndex, Option<Layout>) {
        compose::walk_view_chain(inner, |inner| {
            let ExecutionVariant::View(view) = self.variant_of(inner)? else {
                return None;
            };
            Some((view.composed_layout()?, view.input))
        })
    }

    pub(super) fn layout_of(&self, inner: NodeIndex) -> Option<crate::TensorLayoutInfo> {
        let mut layouts = self.ctx.layouts.borrow_mut();
        layouts.visit(self.ctx.graph, inner);
        layouts.output_layout.get(&inner).cloned()
    }

    /// Look through a qmatmul epilogue operand's view chain to the
    /// contiguous f32 producer the epilogue can index directly.
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

    /// Whether the selected operation can participate in any fusion family.
    pub(super) fn is_seed_candidate(&self, prov: Prov) -> bool {
        let facts = self.driver.egraph.analysis.facts_of(prov);
        if facts.exec.is_none() || !self.state.needed[prov.0 as usize] {
            return false;
        }
        let Some(variant) = self.variant_of(facts.inner) else {
            return false;
        };
        matches!(
            variant,
            ExecutionVariant::Elementwise(_)
                | ExecutionVariant::MatMul(_)
                | ExecutionVariant::QMatMul(_)
                | ExecutionVariant::Reduce(_)
        )
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

    /// All immediately legal alternatives in deterministic tie-break order:
    /// fold views, fuse naries, reduce-fusion family, then matmul fusion.
    /// Extraction compares their GPU costs and re-enqueues the winner, so
    /// chained rewrites happen on later pops after that form becomes current.
    pub(super) fn generate_candidates(&self, prov: Prov) -> Vec<ExecutionVariant> {
        let facts = self.driver.egraph.analysis.facts_of(prov);
        let Some(current) = self.variant_of(facts.inner).cloned() else {
            return Vec::new();
        };
        let mut candidates = Vec::new();
        match &current {
            ExecutionVariant::Elementwise(nary) => {
                if let Some(folded) = self.gen_fold_views_elementwise(nary) {
                    candidates.push(folded);
                }
                if let Some(fused) = self.gen_fuse_naries(nary) {
                    candidates.push(fused);
                }
            }
            ExecutionVariant::Reduce(_) => {}
            _ => {}
        }
        candidates.extend(self.gen_fuse_reduce_candidates(&current));
        if let Some(matmul) = self.gen_fuse_into_matmul(&current) {
            candidates.push(matmul);
        }
        candidates
    }

    /// Fold view producers of this nary's inputs into its index expressions.
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
            if needs_delinearize && compose::input_reread_factor(&expression, shape, slot) > 1 {
                continue;
            }
            let view = view.clone();
            let Some(rewritten) = compose::rewrite_view_input(&expression, slot, &view) else {
                continue;
            };
            expression = rewritten;
            inputs[slot] = view.input;
            folded = true;
        }
        if !folded {
            return None;
        }
        Some(compose::deduplicate_inputs(inputs, expression))
    }

    /// Inline every sole-consumed elementwise producer into this nary,
    /// within the direct-input binding budget.
    fn gen_fuse_naries(&self, nary: &ElementwiseOperation) -> Option<ExecutionVariant> {
        let (final_inputs, final_expression) = self.inline_producers(
            &nary.inputs,
            &nary.expression,
            &nary.shape,
            InlineGate {
                skip_externally_live: true,
                require_same_index_space: false,
            },
        )?;
        Some(ExecutionVariant::Elementwise(ElementwiseOperation {
            inputs: final_inputs,
            expression: final_expression,
            shape: nary.shape.clone(),
            output_datatype: nary.output_datatype,
        }))
    }

    /// Shared body of the two producer-inlining rewrites: substitute every
    /// sole-consumed elementwise producer into `expression`, directly where
    /// the read is element-wise and by composing the producer with the read's
    /// coordinates otherwise. `None` when nothing inlined.
    fn inline_producers(
        &self,
        inputs: &[NodeIndex],
        expression: &NaryExpr,
        shape: &[usize],
        gate: InlineGate,
    ) -> Option<(Vec<NodeIndex>, NaryExpr)> {
        let mut expression = expression.clone();
        let mut all_inputs = inputs.to_vec();
        let mut fused_any = false;
        let max_fused_inputs = self.device().nary_direct_input_binding_budget();

        for &input_inner in inputs.iter() {
            if self.is_cached(input_inner) {
                continue;
            }
            // An externally live producer materializes regardless, so
            // inlining it here would duplicate its compute. Region formation
            // fuses it with consumers and emits it as another output.
            if gate.skip_externally_live && self.externally_live(input_inner) {
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
            let target_slots: Vec<usize> = all_inputs
                .iter()
                .enumerate()
                .filter_map(|(slot, value)| (*value == input_inner).then_some(slot))
                .collect();
            let offset = all_inputs.len();
            let inlined = compose::offset_input_indices(&input_nary.expression, offset);
            let mut new_expression = expression.clone();
            let mut success = !gate.require_same_index_space || input_nary.shape.as_ref() == shape;
            if success {
                for slot in &target_slots {
                    let (next, s) =
                        compose::substitute_input_in_expr(&new_expression, *slot, &inlined);
                    new_expression = next;
                    success &= s;
                }
            }
            if !success
                && target_slots
                    .iter()
                    .all(|&slot| compose::input_reread_factor(&expression, shape, slot) == 1)
            {
                let mut composed = expression.clone();
                success = true;
                for slot in &target_slots {
                    match compose::substitute_input_composed(&composed, *slot, &inlined) {
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
        Some(compose::deduplicate_inputs(all_inputs, expression))
    }

    /// All legal reduce-fusion alternatives. Extraction's cost and
    /// duplication constraints decide whether any candidate commits.
    fn gen_fuse_reduce_candidates(&self, current: &ExecutionVariant) -> Vec<ExecutionVariant> {
        let mut candidates = Vec::new();
        candidates.extend(self.gen_collapse_unit_reduce(current));
        candidates.extend(self.gen_fold_views_into_reduce(current));
        candidates.extend(self.gen_unary_into_reduce(current));
        candidates.extend(self.gen_indexed_unary_into_reduce(current));
        candidates.extend(self.gen_producer_into_reduce(current));
        candidates
    }

    /// Append a consumer's unary chain to its reduce producer's epilogue.
    fn gen_unary_into_reduce(&self, current: &ExecutionVariant) -> Option<ExecutionVariant> {
        let el_op = compose::try_get_unary_chain(current)?;
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

    /// Rewrite a reduce over a unit axis as the equivalent elementwise.
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
        let expression = compose::compose_expr_with_indices(&reduce.expression, &mapping)?;
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

    /// Fold view producers of this reduce's inputs into its index
    /// expressions.
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

    /// Inline a reduce producer read through an index expression into the
    /// consuming nary, turning it into a reduce over the outer axis.
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
        let (functions, indices) = compose::extract_unary_chain_indexed(nary)?;
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
        let expression = compose::compose_expr_with_indices(&reduce.expression, &mapping)?;
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

    /// Inline a sole-consumed elementwise producer into this reduce's
    /// expression.
    fn gen_producer_into_reduce(&self, current: &ExecutionVariant) -> Option<ExecutionVariant> {
        let ExecutionVariant::Reduce(reduce) = current else {
            return None;
        };
        let (final_inputs, final_expression) = self.inline_producers(
            &reduce.inputs,
            &reduce.expression,
            &reduce.shape,
            InlineGate {
                skip_externally_live: false,
                require_same_index_space: true,
            },
        )?;
        let mut new_reduce = reduce.clone();
        new_reduce.inputs = final_inputs;
        new_reduce.expression = final_expression;
        Some(ExecutionVariant::Reduce(new_reduce))
    }

    /// The matmul/qmatmul epilogue family; see `rules_fuse_matmul.rs`.
    fn gen_fuse_into_matmul(&self, current: &ExecutionVariant) -> Option<ExecutionVariant> {
        self.gen_matmul_family(current)
    }
}
