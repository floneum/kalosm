//! Stage-2 matmul/qmatmul fusion generators: pure transcriptions of
//! `try_fuse_into_matmul` and its graph-reading helpers (fusion_matmul.rs),
//! consulted by the extraction worklist.
//!
//! Each branch reproduces the destructive body with the graph reads replaced
//! by [`FusionView`] calls and the commit tails removed
//! (`commit_qmatmul_post_fusion`, edge surgery, `add_physical_dependencies`,
//! dead-node pruning) — the extractor's switch/kill machinery is the commit.
//! Where the original commits a variant to the rewritten node and returns
//! `true`, the generator returns `Some(variant)`. Gate order, early-return
//! structure, budget checks, and builder calls are transcribed verbatim from
//! the destructive originals — those stay untouched until deletion so the
//! parity baseline holds. Pure expression helpers
//! (`replace_indexed_qmatmul_accumulators`, `remap_temp_accumulator_inputs`,
//! `qmatmul_last_dim_view_offset`) are called on `Resolver` directly; only
//! the helpers that read graph state (`try_extract_indexed_qmatmul_post_expr`,
//! `try_extract_mapped_qmatmul_post_expr`,
//! `try_fuse_qmatmul_narrow_accumulators`) are re-transcribed here.

use rustc_hash::FxHashMap;

use super::super::{ExecutionVariant, Resolver};
use super::rules_fuse::FusionView;
use crate::Layout;
use crate::compute_graph::NodeIndex;
use crate::nary_wise::{ElementwiseOperation, NaryExpr, UnaryFunctionChain};
use crate::quantized::matmul::ElementwiseEpilogue;

impl FusionView<'_> {
    /// Transcription of `try_fuse_into_matmul` (fusion_matmul.rs:4-471):
    /// dense matmul post unary chains, qmatmul narrow-accumulator, indexed
    /// post, general elementwise post, qmatmul pre epilogues, and dense
    /// matmul pre unary chains, in the original attempt order. Returns the
    /// new variant for the node being rewritten (first success wins).
    pub(super) fn gen_matmul_family(&self, current: &ExecutionVariant) -> Option<ExecutionVariant> {
        // Post-op: fuse elementwise after matmul (dense or quantized).
        if let Some(el_op) = Resolver::try_get_unary_chain(current) {
            let input_inner = el_op.value;
            if !self.is_cached(input_inner)
                && let Some(input_variant) = self.variant_of(input_inner)
            {
                // Dtype-preserving unary chains are hosted after the
                // cooperative store, independently of how A/B are mapped.
                // Unsupported chains still lower through the generic fused
                // reduction.
                if let ExecutionVariant::MatMul(matmul_op) = input_variant {
                    let mut new_matmul = matmul_op.clone();
                    let mut existing_post = new_matmul.post_element_wise.functions.clone();
                    existing_post.extend(el_op.functions.functions.iter().cloned());
                    new_matmul.post_element_wise = UnaryFunctionChain::new(
                        existing_post,
                        matmul_op.post_element_wise.input_datatype(),
                    );
                    return Some(ExecutionVariant::MatMul(new_matmul));
                }
            }
        }

        // Post-op (QMatMul): fuse a general element-wise expression after
        // qmatmul. This handles composite expressions like GELU and ordered
        // extra inputs whose layouts match the output visitation shape.
        if let ExecutionVariant::Elementwise(nary) = current {
            // Split/gate expressions built from `narrow` views of a qmatmul
            // output (e.g. SwiGLU's gate/up halves) reach the qmatmul through
            // MapLayout chains with distinct last-dimension column offsets.
            // Absorb them into the accumulator-offset post epilogue before the
            // per-input scan below.
            if let Some(fused) = self.gen_fuse_qmatmul_narrow_accumulators(nary) {
                return Some(fused);
            }
            for (candidate_input_idx, &input_inner) in nary.inputs.iter().enumerate() {
                if self.variant_of(input_inner).is_none() {
                    continue;
                }
                let (qmatmul_inner, map_chain) = self.walk_view_chain(input_inner);
                let Some(ExecutionVariant::QMatMul(qmatmul_op)) = self.variant_of(qmatmul_inner)
                else {
                    continue;
                };
                let qmatmul_op = qmatmul_op.clone();
                if map_chain.is_none()
                    && !self.is_cached(input_inner)
                    && qmatmul_op.post_element_wise_expr.is_none()
                    && qmatmul_op.in_shape[..qmatmul_op.in_shape.len() - 1]
                        .iter()
                        .product::<usize>()
                        == 1
                    && let Some((expression, accumulator_offsets, extras)) = self
                        .try_extract_indexed_qmatmul_post_expr(
                            nary,
                            candidate_input_idx,
                            &qmatmul_op.out_shape,
                        )
                {
                    let Some(input_datatype) = nary
                        .expression
                        .elementwise_input_datatype(candidate_input_idx)
                    else {
                        continue;
                    };
                    if input_datatype != crate::DataTypeEnum::F32
                        || nary.output_datatype != crate::DataTypeEnum::F32
                    {
                        continue;
                    }
                    if !qmatmul_op.supports_indexed_post_accumulator_offsets(
                        &self.device(),
                        &nary.shape,
                        &accumulator_offsets,
                    ) {
                        continue;
                    }

                    let post_element_wise_expr = ElementwiseEpilogue {
                        expression,
                        extras: extras.clone(),
                        input_datatype,
                        output_datatype: nary.output_datatype,
                    };

                    let mut new_q = qmatmul_op.clone();
                    new_q.out_shape = nary.shape.clone();
                    new_q.post_element_wise_expr = Some(post_element_wise_expr);
                    new_q.post_accumulator_offsets = accumulator_offsets.into_boxed_slice();

                    if !new_q.fits_binding_budget(&self.device()) {
                        continue;
                    }

                    return Some(ExecutionVariant::QMatMul(new_q));
                }
                let Some(mapped_layout) = Resolver::apply_view_chain(
                    &Layout::contiguous(&qmatmul_op.out_shape),
                    &map_chain,
                ) else {
                    continue;
                };
                if mapped_layout != Layout::contiguous(&nary.shape) {
                    continue;
                }
                if !nary.expression.uses_input(candidate_input_idx)
                    || nary
                        .expression
                        .uses_custom_indexing_for_input(candidate_input_idx)
                {
                    continue;
                };
                let Some(input_datatype) = nary
                    .expression
                    .elementwise_input_datatype(candidate_input_idx)
                else {
                    continue;
                };
                let mut extras = Vec::new();
                let mut replacements = vec![None; nary.inputs.len()];
                let mut valid_expression = true;
                for (input_idx, &nary_input) in nary.inputs.iter().enumerate() {
                    let (base_inner, chain) = self.walk_view_chain(nary_input);
                    let base_qmatmul = match self.variant_of(base_inner) {
                        Some(ExecutionVariant::QMatMul(op)) => Some(op.clone()),
                        _ => None,
                    };
                    if let Some(base_qmatmul) = base_qmatmul
                        && Resolver::qmatmul_same_base(&qmatmul_op, &base_qmatmul)
                    {
                        let alias_layout = Resolver::apply_view_chain(
                            &Layout::contiguous(&base_qmatmul.out_shape),
                            &chain,
                        );
                        if alias_layout == Some(Layout::contiguous(&nary.shape))
                            && !nary.expression.uses_custom_indexing_for_input(input_idx)
                        {
                            replacements[input_idx] = Resolver::qmatmul_output_expr(
                                &base_qmatmul,
                                &mut extras,
                                nary.shape.len(),
                            );
                            continue;
                        }
                        valid_expression = false;
                        break;
                    }

                    let Some(extra) = self.normalize_qmatmul_post_extra(nary_input, &nary.shape)
                    else {
                        valid_expression = false;
                        break;
                    };
                    replacements[input_idx] =
                        Some(NaryExpr::input(extras.len() + 1, nary.shape.len()));
                    extras.push(extra);
                }
                if !valid_expression {
                    continue;
                }
                let Some(expression) =
                    Resolver::replace_inputs_in_expr(&nary.expression, &replacements)
                else {
                    continue;
                };
                if self.is_cached(input_inner)
                    || input_datatype != crate::DataTypeEnum::F32
                    || nary.output_datatype != crate::DataTypeEnum::F32
                    || !qmatmul_op.supports_elementwise_epilogue_fusion(&self.device())
                {
                    continue;
                }

                let post_element_wise_expr = ElementwiseEpilogue {
                    expression,
                    extras: extras.clone(),
                    input_datatype: qmatmul_op
                        .post_element_wise_expr
                        .as_ref()
                        .map(|existing| existing.input_datatype)
                        .unwrap_or(input_datatype),
                    output_datatype: nary.output_datatype,
                };

                let mut new_q = qmatmul_op.clone();
                new_q.post_element_wise_expr = Some(post_element_wise_expr);

                if !new_q.fits_binding_budget(&self.device()) {
                    continue;
                }

                return Some(ExecutionVariant::QMatMul(new_q));
            }
        }

        // Pre-op (QMatMul): fuse a general element-wise expression upstream
        // of a single-row qmatmul input. For batched/tiled qmatmul, the
        // transformed activation tile is reloaded for each output-column
        // tile, so expensive expressions like GELU would be recomputed many
        // times. Keep those chains materialized once instead.
        if let ExecutionVariant::QMatMul(qmatmul_op) = current
            && qmatmul_op.in_shape[..qmatmul_op.in_shape.len() - 1]
                .iter()
                .product::<usize>()
                == 1
            && qmatmul_op.supports_elementwise_epilogue_fusion(&self.device())
            && !self.is_cached(qmatmul_op.input)
            && self.variant_of(qmatmul_op.input).is_some()
        {
            let (nary_inner, nary_map_chain) = self.walk_view_chain(qmatmul_op.input);
            let Some(ExecutionVariant::Elementwise(nary)) = self.variant_of(nary_inner) else {
                return None;
            };
            let nary = nary.clone();
            let mapped_layout =
                Resolver::apply_view_chain(&Layout::contiguous(&nary.shape), &nary_map_chain);
            if mapped_layout != Some(Layout::contiguous(&qmatmul_op.in_shape)) {
                return None;
            }

            for (candidate_input_idx, &primary_input) in nary.inputs.iter().enumerate() {
                if !nary.expression.uses_input(candidate_input_idx)
                    || nary
                        .expression
                        .uses_custom_indexing_for_input(candidate_input_idx)
                {
                    continue;
                }
                let Some(input_datatype) = nary
                    .expression
                    .elementwise_input_datatype(candidate_input_idx)
                else {
                    continue;
                };
                if input_datatype != crate::DataTypeEnum::F32
                    || nary.output_datatype != crate::DataTypeEnum::F32
                {
                    continue;
                }

                let (primary_inner, primary_chain) = self.walk_view_chain(primary_input);
                let Some(primary_info) = self.layout_of(primary_inner) else {
                    continue;
                };
                let Some(primary_layout) =
                    Resolver::apply_view_chain(primary_info.layout(), &primary_chain)
                else {
                    continue;
                };
                if primary_layout != Layout::contiguous(&nary.shape) {
                    continue;
                }

                let mut mapping = vec![usize::MAX; nary.inputs.len()];
                let mut extras = Vec::new();
                let mut valid_expression = true;
                for (input_idx, &nary_input) in nary.inputs.iter().enumerate() {
                    let (base_inner, chain) = self.walk_view_chain(nary_input);
                    if base_inner == primary_inner {
                        let alias_layout =
                            Resolver::apply_view_chain(primary_info.layout(), &chain);
                        if alias_layout == Some(Layout::contiguous(&nary.shape))
                            && !nary.expression.uses_custom_indexing_for_input(input_idx)
                        {
                            mapping[input_idx] = 0;
                            continue;
                        }
                        valid_expression = false;
                        break;
                    }

                    let Some(extra) = self.normalize_qmatmul_post_extra(nary_input, &nary.shape)
                    else {
                        valid_expression = false;
                        break;
                    };
                    mapping[input_idx] = extras.len() + 1;
                    extras.push(extra);
                }
                if !valid_expression {
                    continue;
                }
                let expression = nary.expression.remap_inputs(&mapping);

                let pre_element_wise_expr =
                    if let Some(existing) = &qmatmul_op.pre_element_wise_expr {
                        if existing.input_datatype != nary.output_datatype {
                            continue;
                        }
                        let mut mapping = Vec::with_capacity(1 + existing.extras.len());
                        mapping.push(0);
                        mapping.extend((0..existing.extras.len()).map(|i| i + 1 + extras.len()));
                        let shifted_existing = existing.expression.remap_inputs(&mapping);
                        let (expression, success) =
                            Resolver::substitute_input_in_expr(&shifted_existing, 0, &expression);
                        if !success {
                            continue;
                        }
                        let mut combined_extras = extras.clone();
                        combined_extras.extend(existing.extras.clone());
                        ElementwiseEpilogue {
                            expression,
                            extras: combined_extras,
                            input_datatype,
                            output_datatype: existing.output_datatype,
                        }
                    } else {
                        ElementwiseEpilogue {
                            expression,
                            extras: extras.clone(),
                            input_datatype,
                            output_datatype: nary.output_datatype,
                        }
                    };

                let mut new_q = qmatmul_op.clone();
                new_q.input = primary_inner;
                new_q.pre_element_wise_expr = Some(pre_element_wise_expr);

                if !new_q.fits_binding_budget(&self.device()) {
                    continue;
                }

                return Some(ExecutionVariant::QMatMul(new_q));
            }
        }

        // Pre-op: fuse elementwise before plain matmul inputs. Cooperative
        // matmuls apply dtype-preserving chains while staging A/B; other
        // chains lower through the generic fused reduction. Un-flattened
        // operands remain excluded because their producer mapping is already
        // being absorbed by cooperative staging.
        if let ExecutionVariant::MatMul(matmul_op) = current
            && matmul_op.a.is_plain()
            && matmul_op.b.is_plain()
        {
            let mut new_matmul = matmul_op.clone();
            let mut changed = false;

            // Check first input
            if !self.is_cached(matmul_op.first)
                && let Some(first_variant) = self.variant_of(matmul_op.first)
                && let Some(el_op) = Resolver::try_get_unary_chain(first_variant)
            {
                new_matmul.first = el_op.value;
                let mut functions = el_op.functions.functions.clone();
                functions.extend(new_matmul.pre_element_wise[0].functions.iter().cloned());
                new_matmul.pre_element_wise[0] =
                    UnaryFunctionChain::new(functions, el_op.functions.input_datatype());
                changed = true;
            }

            // Check second input
            if !self.is_cached(matmul_op.second)
                && let Some(second_variant) = self.variant_of(matmul_op.second)
                && let Some(el_op) = Resolver::try_get_unary_chain(second_variant)
            {
                new_matmul.second = el_op.value;
                let mut functions = el_op.functions.functions.clone();
                functions.extend(new_matmul.pre_element_wise[1].functions.iter().cloned());
                new_matmul.pre_element_wise[1] =
                    UnaryFunctionChain::new(functions, el_op.functions.input_datatype());
                changed = true;
            }

            if changed {
                return Some(ExecutionVariant::MatMul(new_matmul));
            }
        }

        None
    }

    /// Transcription of `try_fuse_qmatmul_narrow_accumulators`
    /// (fusion_matmul.rs:535-610). Absorb a split/gate n-ary whose inputs are
    /// `narrow` (MapLayout) views of a single-row qmatmul output into that
    /// qmatmul's accumulator-offset post epilogue. Each distinct
    /// last-dimension column offset (e.g. the gate half at 0 and the up half
    /// at `pair_len`) becomes one accumulator value, so a SwiGLU-style
    /// `silu(gate) * up` resolves to a single dynamic qmatmul kernel where
    /// the backend supports it. Returns `None` when the pattern, dtype,
    /// layout, accumulator offsets, or binding budget are unsupported.
    fn gen_fuse_qmatmul_narrow_accumulators(
        &self,
        nary: &ElementwiseOperation,
    ) -> Option<ExecutionVariant> {
        if nary.output_datatype != crate::DataTypeEnum::F32 {
            return None;
        }

        // Find the qmatmul reached through a narrow MapLayout view. A direct
        // (chain-less) reference is the indexed-input form handled below.
        let mut base = None;
        for &input in &nary.inputs {
            let (base_inner, chain) = self.walk_view_chain(input);
            if chain.is_none() {
                continue;
            }
            if let Some(ExecutionVariant::QMatMul(op)) = self.variant_of(base_inner) {
                // A qmatmul that already carries a post epilogue isn't a clean
                // accumulator-offset base; leave it to the general scan.
                if op.post_element_wise_expr.is_some() {
                    continue;
                }
                base = Some((base_inner, op.clone()));
                break;
            }
        }
        let Some((qmatmul_inner, qmatmul_op)) = base else {
            return None;
        };
        if self.is_cached(qmatmul_inner) {
            return None;
        }

        let Some((expression, accumulator_offsets, extras)) =
            self.try_extract_mapped_qmatmul_post_expr(nary, qmatmul_inner, &qmatmul_op.out_shape)
        else {
            return None;
        };

        if !qmatmul_op.supports_indexed_post_accumulator_offsets(
            &self.device(),
            &nary.shape,
            &accumulator_offsets,
        ) {
            return None;
        }

        let post_element_wise_expr = ElementwiseEpilogue {
            expression,
            extras,
            input_datatype: crate::DataTypeEnum::F32,
            output_datatype: nary.output_datatype,
        };

        let mut new_q = qmatmul_op;
        new_q.out_shape = nary.shape.clone();
        new_q.post_element_wise_expr = Some(post_element_wise_expr);
        new_q.post_accumulator_offsets = accumulator_offsets.into_boxed_slice();

        if !new_q.fits_binding_budget(&self.device()) {
            return None;
        }

        Some(ExecutionVariant::QMatMul(new_q))
    }

    /// Transcription of `try_extract_mapped_qmatmul_post_expr`
    /// (fusion_matmul.rs:619-708). Build the post epilogue expression,
    /// accumulator column offsets, and extra-tensor dependencies for an n-ary
    /// whose inputs are last-dimension `narrow` views of `qmatmul_inner`.
    /// Inputs that view the qmatmul become accumulator values (indices
    /// `0..offsets.len()`, deduplicated by column offset); every other input
    /// becomes a normalized extra tensor (indices after the accumulators).
    /// Returns `None` when an input isn't a clean last-dimension narrow, uses
    /// custom indexing, or can't be normalized.
    fn try_extract_mapped_qmatmul_post_expr(
        &self,
        nary: &ElementwiseOperation,
        qmatmul_inner: NodeIndex,
        qmatmul_out_shape: &[usize],
    ) -> Option<(NaryExpr, Vec<u32>, Vec<NodeIndex>)> {
        if nary.shape.len() != qmatmul_out_shape.len() {
            return None;
        }
        // The accumulator-offset epilogue is only lowered by the single-row
        // qgemv path, so every leading dimension must collapse to one row.
        if qmatmul_out_shape[..qmatmul_out_shape.len() - 1]
            .iter()
            .product::<usize>()
            != 1
        {
            return None;
        }
        let output_cols = nary.shape.last().copied()? as u32;
        let matrix_cols = qmatmul_out_shape.last().copied()? as u32;
        // A full-width (or wider) output isn't a split; the general scan owns
        // that case.
        if output_cols >= matrix_cols {
            return None;
        }

        let qmatmul_out_layout = Layout::contiguous(qmatmul_out_shape);
        let rank = nary.shape.len();

        enum MappedInput {
            Accumulator(usize),
            Extra(usize),
        }

        let mut accumulator_offsets = Vec::new();
        let mut accumulator_map = FxHashMap::default();
        let mut extras = Vec::new();
        let mut mapped = Vec::with_capacity(nary.inputs.len());
        for (input_idx, &nary_input) in nary.inputs.iter().enumerate() {
            if !nary.expression.uses_input(input_idx) {
                mapped.push(None);
                continue;
            }
            if nary.expression.uses_custom_indexing_for_input(input_idx) {
                return None;
            }
            let (base_inner, chain) = self.walk_view_chain(nary_input);
            if base_inner == qmatmul_inner {
                let view = Resolver::apply_view_chain(&qmatmul_out_layout, &chain)?;
                let offset =
                    Resolver::qmatmul_last_dim_view_offset(&view, &nary.shape, matrix_cols)?;
                let value_idx = *accumulator_map.entry(offset).or_insert_with(|| {
                    let idx = accumulator_offsets.len();
                    accumulator_offsets.push(offset);
                    idx
                });
                mapped.push(Some(MappedInput::Accumulator(value_idx)));
            } else {
                let extra = self.normalize_qmatmul_post_extra(nary_input, &nary.shape)?;
                let pos = extras.len();
                extras.push(extra);
                mapped.push(Some(MappedInput::Extra(pos)));
            }
        }

        // Two distinct column offsets are the smallest split worth folding into
        // the accumulator-offset path; a single offset is either the default
        // full-width store or a partial column the qgemv path can't cover.
        if accumulator_offsets.len() < 2 {
            return None;
        }

        let accumulator_count = accumulator_offsets.len();
        let mut replacements = vec![None; nary.inputs.len()];
        for (input_idx, kind) in mapped.into_iter().enumerate() {
            match kind {
                Some(MappedInput::Accumulator(value_idx)) => {
                    replacements[input_idx] = Some(NaryExpr::input(value_idx, rank));
                }
                Some(MappedInput::Extra(pos)) => {
                    replacements[input_idx] = Some(NaryExpr::input(accumulator_count + pos, rank));
                }
                None => {}
            }
        }

        let expression = Resolver::replace_inputs_in_expr(&nary.expression, &replacements)?;
        Some((expression, accumulator_offsets, extras))
    }

    /// Transcription of `try_extract_indexed_qmatmul_post_expr`
    /// (fusion_matmul.rs:732-792).
    fn try_extract_indexed_qmatmul_post_expr(
        &self,
        nary: &ElementwiseOperation,
        qmatmul_input_idx: usize,
        qmatmul_out_shape: &[usize],
    ) -> Option<(NaryExpr, Vec<u32>, Vec<NodeIndex>)> {
        if nary.output_datatype != crate::DataTypeEnum::F32
            || nary.shape.len() != qmatmul_out_shape.len()
            || nary.shape.as_ref() == qmatmul_out_shape
        {
            return None;
        }
        let output_cols = nary.shape.last().copied()? as u32;
        let matrix_cols = qmatmul_out_shape.last().copied()? as u32;
        if output_cols >= matrix_cols {
            return None;
        }

        let temp_input_base = nary.inputs.len();
        let mut accumulator_offsets = Vec::new();
        let mut accumulator_map = FxHashMap::default();
        let expression = Resolver::replace_indexed_qmatmul_accumulators(
            &nary.expression,
            qmatmul_input_idx,
            nary.shape.len(),
            output_cols,
            matrix_cols,
            temp_input_base,
            &mut accumulator_offsets,
            &mut accumulator_map,
        )?;
        if accumulator_offsets.len() < 2 {
            return None;
        }

        let mut replacements = vec![None; nary.inputs.len()];
        let mut extras = Vec::new();
        for (input_idx, &input) in nary.inputs.iter().enumerate() {
            if input_idx == qmatmul_input_idx || !nary.expression.uses_input(input_idx) {
                continue;
            }
            if nary.expression.uses_custom_indexing_for_input(input_idx) {
                return None;
            }
            let extra = self.normalize_qmatmul_post_extra(input, &nary.shape)?;
            replacements[input_idx] = Some(NaryExpr::input(
                accumulator_offsets.len() + extras.len(),
                nary.shape.len(),
            ));
            extras.push(extra);
        }

        let expression = Resolver::replace_inputs_in_expr(&expression, &replacements)?;
        let expression = Resolver::remap_temp_accumulator_inputs(
            &expression,
            temp_input_base,
            accumulator_offsets.len(),
        );
        Some((expression, accumulator_offsets, extras))
    }
}
