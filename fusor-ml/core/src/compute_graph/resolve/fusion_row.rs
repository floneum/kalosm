//! Row-program fusion: collapse clusters of same-axis reductions and
//! elementwise expressions into one [`RowProgramOperation`].
//!
//! Unlike recognition, nothing here matches a named operation. The pass
//! walks upward from an elementwise root, converting everything it can into
//! expressions over a shared slot space: full-shape elementwise producers
//! inline, reductions over a single common axis become per-row scalar
//! phases (their keepdim-broadcast reads turn into scalar references), and
//! anything else becomes an external input. Composed softmax and RMS norm
//! collapse to one kernel this way — and so does any other normalization-
//! shaped cluster the tensor API emits.

use petgraph::algo::toposort;

use crate::{
    nary_wise::NaryFunction,
    row_program::{RowOutput, RowProgramOperation, RowReduce, RowStep},
};

use super::cluster_match::{keepdim_broadcast_layout, layout_matches, unary_elementwise};
use super::*;

/// Scalar slot base while a cluster is under construction; remapped to
/// `externals.len() + phase` on commit.
const SCALAR_SLOT_BASE: usize = usize::MAX / 2;

struct ClusterBuilder<'a> {
    shape: Box<[usize]>,
    axis: Option<usize>,
    externals: Vec<NodeIndex>,
    /// `(scalar chain base node, phase)` — the base node keys deduplication
    /// when the same reduction is read more than once.
    phases: Vec<(NodeIndex, RowReduce)>,
    /// Inner nodes absorbed into the cluster: they must be consumed only by
    /// other members (or the root) and die on commit.
    members: Vec<NodeIndex>,
    /// Memoized full-shape expressions for absorbed elementwise nodes.
    full_exprs: FxHashMap<NodeIndex, NaryExpr>,
    /// The closed absorbable set: nodes outside it read as externals.
    allowed: &'a FxHashSet<NodeIndex>,
}

struct BuilderSnapshot {
    externals: usize,
    phases: usize,
    members: usize,
    full_exprs: FxHashMap<NodeIndex, NaryExpr>,
}

impl ClusterBuilder<'_> {
    fn snapshot(&self) -> BuilderSnapshot {
        BuilderSnapshot {
            externals: self.externals.len(),
            phases: self.phases.len(),
            members: self.members.len(),
            full_exprs: self.full_exprs.clone(),
        }
    }

    fn restore(&mut self, snapshot: BuilderSnapshot) {
        self.externals.truncate(snapshot.externals);
        self.phases.truncate(snapshot.phases);
        self.members.truncate(snapshot.members);
        self.full_exprs = snapshot.full_exprs;
    }

    fn external_slot(&mut self, inner: NodeIndex) -> usize {
        if let Some(slot) = self.externals.iter().position(|&node| node == inner) {
            slot
        } else {
            self.externals.push(inner);
            self.externals.len() - 1
        }
    }

    /// Member insertion is idempotent: one node can be reached through
    /// several operand walks, but is absorbed and removed exactly once.
    fn add_member(&mut self, node: NodeIndex) {
        if !self.members.contains(&node) {
            self.members.push(node);
        }
    }

    fn add_members(&mut self, nodes: impl IntoIterator<Item = NodeIndex>) {
        for node in nodes {
            self.add_member(node);
        }
    }
}

/// Rewrite the slot references of one absorbed node's expression into the
/// cluster slot space: external slots keep their index lists, inlined
/// expressions require plain elementwise reads.
enum SlotRewrite {
    External(usize),
    Inline(NaryExpr),
}

fn rewrite_slots(expr: &NaryExpr, rewrites: &[SlotRewrite]) -> Option<NaryExpr> {
    match expr {
        NaryExpr::Op { children, function } => Some(NaryExpr::Op {
            children: children
                .iter()
                .map(|child| rewrite_slots(child, rewrites))
                .collect::<Option<Vec<_>>>()?,
            function: function.clone(),
        }),
        NaryExpr::IndexedInput { input_idx, indices } => {
            let indices = indices
                .iter()
                .map(|index| rewrite_slots(index, rewrites))
                .collect::<Option<Vec<_>>>()?;
            match &rewrites[*input_idx] {
                SlotRewrite::External(slot) => Some(NaryExpr::IndexedInput {
                    input_idx: *slot,
                    indices,
                }),
                SlotRewrite::Inline(replacement) => {
                    NaryExpr::is_elementwise_indices(&indices).then(|| replacement.clone())
                }
            }
        }
        NaryExpr::DimIndex(dim) => Some(NaryExpr::DimIndex(*dim)),
        NaryExpr::Scalar(value) => Some(NaryExpr::Scalar(*value)),
    }
}

/// Remap construction-time scalar slots to their final indices.
fn finalize_slots(expr: &NaryExpr, external_count: usize) -> NaryExpr {
    egraph::compose::map_loads(expr, &mut |input_idx, _, indices| {
        let input_idx = if input_idx >= SCALAR_SLOT_BASE {
            external_count + (input_idx - SCALAR_SLOT_BASE)
        } else {
            input_idx
        };
        NaryExpr::IndexedInput { input_idx, indices }
    })
}

impl Resolver {
    /// Fuse row-program clusters. Candidate roots are discovered from the
    /// reductions outward — a reduce's scalar flows through its keepdim
    /// broadcast into some elementwise consumer, and the root is the last
    /// single-consumer elementwise below it — so graphs full of elementwise
    /// chains with no reductions (most of a decode graph) cost nothing.
    pub(super) fn fuse_row_programs(&mut self, graph: &mut ComputeGraphInner) {
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
        if reduces.is_empty() {
            return;
        }

        let mut roots = Vec::new();
        let mut rootless = 0usize;
        let reduce_count = reduces.len();
        for reduce in reduces {
            if let Some(root) = self.row_cluster_root(reduce) {
                if !roots.contains(&root) {
                    roots.push(root);
                }
            } else {
                rootless += 1;
            }
        }
        if graph.device().config().trace_row_fusion {
            eprintln!(
                "row_fusion: {reduce_count} reduces, {} roots, {rootless} rootless",
                roots.len()
            );
        }
        // Outermost roots first so each committed cluster is maximal: a
        // root's cluster can contain another candidate root (softmax's two
        // reductions converge on one div).
        let Ok(order) = toposort(&self.execution_graph, None) else {
            return;
        };
        let rank: FxHashMap<ExecutionNodeIndex, usize> = order
            .into_iter()
            .enumerate()
            .map(|(rank, node)| (node, rank))
            .collect();
        roots.sort_by_key(|root| std::cmp::Reverse(rank.get(root).copied().unwrap_or(0)));
        for root in roots {
            if !self.execution_graph.contains_node(root) {
                continue;
            }
            self.try_fuse_row_program(graph, root);
        }
    }

    /// Walk downstream from a reduce to the cluster's natural root: through
    /// any reduced-shape unary chain and keepdim views to the first
    /// full-shape elementwise consumer, then down while that node feeds
    /// exactly one full-shape elementwise.
    fn row_cluster_root(&self, reduce: ExecutionNodeIndex) -> Option<ExecutionNodeIndex> {
        let ExecutionVariant::Reduce(op) = &self.execution_graph[reduce].variant else {
            return None;
        };
        let full_shape = op.shape.clone();

        // Phase 1: descend through scalar-side nodes (views and elementwise
        // ops over fewer elements than the full shape) to a full-shape
        // elementwise consumer.
        let mut node = reduce;
        let mut current = loop {
            let mut consumers = self
                .execution_graph
                .neighbors_directed(node, petgraph::Direction::Outgoing);
            let first = consumers.next()?;
            if consumers.next().is_some() {
                return None;
            }
            match &self.execution_graph[first].variant {
                ExecutionVariant::View(_) => node = first,
                ExecutionVariant::Elementwise(nary) if nary.shape == full_shape => break first,
                ExecutionVariant::Elementwise(_) => node = first,
                _ => return None,
            }
        };

        // Phase 2: descend while the node feeds exactly one full-shape
        // elementwise (other consumers may be reductions inside the cluster).
        loop {
            let mut next = None;
            for consumer in self
                .execution_graph
                .neighbors_directed(current, petgraph::Direction::Outgoing)
            {
                if let ExecutionVariant::Elementwise(nary) = &self.execution_graph[consumer].variant
                    && nary.shape == full_shape
                {
                    if next.is_some() {
                        return Some(current);
                    }
                    next = Some(consumer);
                }
            }
            match next {
                Some(consumer) => current = consumer,
                None => return Some(current),
            }
        }
    }

    fn try_fuse_row_program(
        &mut self,
        graph: &mut ComputeGraphInner,
        root_idx: ExecutionNodeIndex,
    ) -> bool {
        let ExecutionVariant::Elementwise(root) = self.execution_graph[root_idx].variant.clone()
        else {
            return false;
        };

        // Greedy absorption could claim a node another branch still needs
        // (a residual stream read by the next layer, say). Compute the
        // *closed* absorbable set first: collect everything reachable
        // through eligible nodes, then iteratively drop anything consumed
        // outside the set — what remains can fuse without anything
        // escaping. Dropped nodes read as external inputs and materialize
        // once, exactly as they must.
        let mut allowed = self.collect_row_cluster(graph, root_idx, &root, None);
        loop {
            let allowed_execs: FxHashSet<ExecutionNodeIndex> = allowed
                .iter()
                .filter_map(|&inner| self.get_input_node_in_exec_graph(inner))
                .collect();
            let mut violators = Vec::new();
            for &member in &allowed {
                let Some(exec) = self.get_input_node_in_exec_graph(member) else {
                    violators.push(member);
                    continue;
                };
                if self
                    .execution_graph
                    .neighbors_directed(exec, petgraph::Direction::Outgoing)
                    .any(|consumer| consumer != root_idx && !allowed_execs.contains(&consumer))
                {
                    violators.push(member);
                }
            }
            if violators.is_empty() {
                break;
            }
            for violator in violators {
                allowed.remove(&violator);
            }
            // Re-walk: regions only reachable through a dropped node fall
            // out of the set too.
            allowed = self.collect_row_cluster(graph, root_idx, &root, Some(&allowed));
        }
        if allowed.is_empty() {
            return false;
        }
        self.build_row_cluster(graph, root_idx, &root, &allowed)
    }

    /// Collect the nodes the absorption walk could claim: full-shape
    /// elementwise producers and keepdim-broadcast scalar reads of same-axis
    /// reductions, recursively. `within` restricts the walk to an existing
    /// candidate set.
    fn collect_row_cluster(
        &self,
        graph: &ComputeGraphInner,
        _root_idx: ExecutionNodeIndex,
        root: &ElementwiseOperation,
        within: Option<&FxHashSet<NodeIndex>>,
    ) -> FxHashSet<NodeIndex> {
        let mut out = FxHashSet::default();
        let mut axis = None;
        for &input in &root.inputs {
            self.collect_operand(graph, &root.shape, &mut axis, input, within, &mut out);
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_operand(
        &self,
        graph: &ComputeGraphInner,
        shape: &[usize],
        axis: &mut Option<usize>,
        inner: NodeIndex,
        within: Option<&FxHashSet<NodeIndex>>,
        out: &mut FxHashSet<NodeIndex>,
    ) {
        let eligible = |node: NodeIndex| {
            within.is_none_or(|allowed| allowed.contains(&node)) && !out.contains(&node)
        };
        if out.contains(&inner) {
            return;
        }
        if !eligible(inner) && !out.contains(&inner) && within.is_some() {
            return;
        }

        // Scalar path: views → unary chain → same-axis reduce.
        'scalar: {
            let mut nodes = Vec::new();
            let mut node = inner;
            let mut layout: Option<Layout> = None;
            loop {
                let Some(exec) = self.get_input_node_in_exec_graph(node) else {
                    break 'scalar;
                };
                let ExecutionVariant::View(view) = &self.execution_graph[exec].variant else {
                    break;
                };
                if !within.is_none_or(|allowed| allowed.contains(&node)) {
                    break 'scalar;
                }
                let Some(collapsed) = view.composed_layout() else {
                    break 'scalar;
                };
                layout = Some(match &layout {
                    None => collapsed,
                    Some(outer) => match crate::view::compose_layouts(outer, &collapsed) {
                        Some(layout) => layout,
                        None => break 'scalar,
                    },
                });
                nodes.push(node);
                node = view.input;
            }
            let Some(mut layout) = layout else {
                break 'scalar;
            };
            let reduce = loop {
                if self.check_cached(graph, node)
                    || !within.is_none_or(|allowed| allowed.contains(&node))
                {
                    break 'scalar;
                }
                let Some(exec) = self.get_input_node_in_exec_graph(node) else {
                    break 'scalar;
                };
                match &self.execution_graph[exec].variant {
                    ExecutionVariant::Reduce(reduce) => break reduce.clone(),
                    ExecutionVariant::Elementwise(nary) => {
                        if unary_elementwise(nary).is_none() {
                            break 'scalar;
                        }
                        nodes.push(node);
                        let (_, input) = unary_elementwise(nary).unwrap();
                        node = input;
                    }
                    // A pure view *between* unary chain links (the
                    // `sum_keepdim` unsqueeze under `div_scalar`/`sqrt` in
                    // layer norm) composes into the running layout; unaries
                    // are pointwise, so their position relative to pure
                    // layout stages cannot change per-row values.
                    ExecutionVariant::View(view) => {
                        let Some(collapsed) = view.composed_layout() else {
                            break 'scalar;
                        };
                        layout = match crate::view::compose_layouts(&layout, &collapsed) {
                            Some(layout) => layout,
                            None => break 'scalar,
                        };
                        nodes.push(node);
                        node = view.input;
                    }
                    _ => break 'scalar,
                }
            };
            let Some(value) = reduce.plain_input() else {
                break 'scalar;
            };
            if reduce.shape.as_ref() != shape
                || axis.is_some_and(|existing| existing != reduce.axis)
                || !layout_matches(Some(&layout), &keepdim_broadcast_layout(shape, reduce.axis))
            {
                break 'scalar;
            }
            *axis = Some(reduce.axis);
            out.extend(nodes);
            out.insert(node);
            self.collect_operand(graph, shape, axis, value, within, out);
            return;
        }

        // Full-shape elementwise producer.
        if self.check_cached(graph, inner) {
            return;
        }
        let Some(exec) = self.get_input_node_in_exec_graph(inner) else {
            return;
        };
        let ExecutionVariant::Elementwise(nary) = &self.execution_graph[exec].variant else {
            return;
        };
        if nary.shape.as_ref() != shape {
            return;
        }
        let inputs = nary.inputs.clone();
        out.insert(inner);
        for input in inputs {
            self.collect_operand(graph, shape, axis, input, within, out);
        }
    }

    fn build_row_cluster(
        &mut self,
        graph: &mut ComputeGraphInner,
        root_idx: ExecutionNodeIndex,
        root: &ElementwiseOperation,
        allowed: &FxHashSet<NodeIndex>,
    ) -> bool {
        let mut builder = ClusterBuilder {
            shape: root.shape.clone(),
            axis: None,
            externals: Vec::new(),
            phases: Vec::new(),
            members: Vec::new(),
            full_exprs: FxHashMap::default(),
            allowed,
        };
        let mut rewrites = Vec::with_capacity(root.inputs.len());
        for &input in &root.inputs {
            let Some(rewrite) = self.absorb_operand(graph, &mut builder, input) else {
                return false;
            };
            rewrites.push(rewrite);
        }
        let Some(output_expr) = rewrite_slots(&root.expression, &rewrites) else {
            return false;
        };

        let trace = graph.device().config().trace_row_fusion;
        // Fusing pays for itself only when at least one reduction folds in.
        if builder.phases.is_empty() {
            return false;
        }
        let axis = builder.axis.expect("phases imply a chosen axis");

        // The fixpoint guarantees closure; verify before rewriting anyway.
        let member_execs: FxHashSet<ExecutionNodeIndex> = builder
            .members
            .iter()
            .filter_map(|&inner| self.get_input_node_in_exec_graph(inner))
            .collect();
        if member_execs.len() != builder.members.len() {
            return false;
        }
        for &exec in &member_execs {
            for consumer in self
                .execution_graph
                .neighbors_directed(exec, petgraph::Direction::Outgoing)
            {
                if consumer != root_idx && !member_execs.contains(&consumer) {
                    return false;
                }
            }
        }

        let external_count = builder.externals.len();
        let mut steps: Vec<RowStep> = builder
            .phases
            .into_iter()
            .map(|(_, mut phase)| {
                phase.expression = finalize_slots(&phase.expression, external_count);
                RowStep::Reduce(phase)
            })
            .collect();
        steps.push(RowStep::Output(RowOutput::Map(finalize_slots(
            &output_expr,
            external_count,
        ))));
        let operation = RowProgramOperation {
            inputs: builder.externals.clone(),
            shape: builder.shape,
            axis,
            steps,
            output_datatype: root.output_datatype,
            dynamic_axis: None,
        };
        let externals = builder.externals;
        if trace {
            eprintln!(
                "row_fusion: committed root {:?} shape {:?} phases {} externals {}",
                self.execution_graph[root_idx].inner_idx,
                operation.shape,
                operation.phase_count(),
                externals.len()
            );
        }
        self.commit_recognized(
            graph,
            root_idx,
            &externals,
            ExecutionVariant::RowProgram(operation),
        );
        true
    }

    /// Convert one operand read into the cluster slot space: a scalar
    /// reference when it is a keepdim-broadcast of a same-axis reduction, an
    /// inlined expression when it is an absorbable full-shape elementwise
    /// node, and an external slot otherwise.
    fn absorb_operand(
        &self,
        graph: &ComputeGraphInner,
        builder: &mut ClusterBuilder<'_>,
        inner: NodeIndex,
    ) -> Option<SlotRewrite> {
        if !builder.allowed.contains(&inner) {
            return Some(SlotRewrite::External(builder.external_slot(inner)));
        }
        let snapshot = builder.snapshot();
        if let Some(expr) = self.try_absorb_scalar(graph, builder, inner) {
            return Some(SlotRewrite::Inline(expr));
        }
        builder.restore(snapshot);

        let snapshot = builder.snapshot();
        if let Some(expr) = self.try_absorb_full(graph, builder, inner) {
            return Some(SlotRewrite::Inline(expr));
        }
        builder.restore(snapshot);

        Some(SlotRewrite::External(builder.external_slot(inner)))
    }

    /// Inline a full-shape elementwise producer consumed by the cluster.
    fn try_absorb_full(
        &self,
        graph: &ComputeGraphInner,
        builder: &mut ClusterBuilder<'_>,
        inner: NodeIndex,
    ) -> Option<NaryExpr> {
        if let Some(expr) = builder.full_exprs.get(&inner) {
            return Some(expr.clone());
        }
        // A cached intermediate is a hard boundary; a user-held reference
        // alone is not — the inner-graph node outlives this resolve, so a
        // later resolve of that handle simply recomputes it (the same rule
        // n-ary fusion applies).
        if self.check_cached(graph, inner) {
            return None;
        }
        let exec = self.get_input_node_in_exec_graph(inner)?;
        let ExecutionVariant::Elementwise(nary) = &self.execution_graph[exec].variant else {
            return None;
        };
        if nary.shape != builder.shape {
            return None;
        }
        let nary = nary.clone();

        let mut rewrites = Vec::with_capacity(nary.inputs.len());
        for &input in &nary.inputs {
            rewrites.push(self.absorb_operand(graph, builder, input)?);
        }
        let expr = rewrite_slots(&nary.expression, &rewrites)?;
        builder.add_member(inner);
        builder.full_exprs.insert(inner, expr.clone());
        Some(expr)
    }

    /// Absorb a keepdim-broadcast read of a same-axis reduction as a per-row
    /// scalar phase: views compose down to the broadcast layout, an optional
    /// unary chain below them becomes the phase's post chain, and the
    /// reduction's producer is absorbed like any other full-shape value.
    fn try_absorb_scalar(
        &self,
        graph: &ComputeGraphInner,
        builder: &mut ClusterBuilder<'_>,
        inner: NodeIndex,
    ) -> Option<NaryExpr> {
        // Walk the view chain, collecting the member nodes.
        let mut views = Vec::new();
        let mut layout: Option<Layout> = None;
        let mut node = inner;
        loop {
            let exec = self.get_input_node_in_exec_graph(node)?;
            let ExecutionVariant::View(view) = &self.execution_graph[exec].variant else {
                break;
            };
            let collapsed = view.composed_layout()?;
            layout = Some(match &layout {
                None => collapsed,
                Some(outer) => crate::view::compose_layouts(outer, &collapsed)?,
            });
            views.push(node);
            node = view.input;
        }
        let layout = layout?;

        // An optional unary chain on the reduced value (mean scaling, eps,
        // rsqrt...) folds into the phase's post chain, innermost first.
        // Pure views interleaved with the chain (the `sum_keepdim`
        // unsqueeze) compose into the layout; they are tracked separately
        // from `chain_nodes` so phase deduplication stays keyed on real
        // chain nodes.
        let mut layout = layout;
        let mut chain_nodes = Vec::new();
        let mut sandwich_views = Vec::new();
        let mut chain: Vec<NaryFunction> = Vec::new();
        let reduce = loop {
            if self.check_cached(graph, node) {
                return None;
            }
            let exec = self.get_input_node_in_exec_graph(node)?;
            match &self.execution_graph[exec].variant {
                ExecutionVariant::Reduce(reduce) => break reduce.clone(),
                ExecutionVariant::Elementwise(nary) => {
                    let (function, input) = unary_elementwise(nary)?;
                    chain.push(function);
                    chain_nodes.push(node);
                    node = input;
                }
                ExecutionVariant::View(view) => {
                    let collapsed = view.composed_layout()?;
                    layout = crate::view::compose_layouts(&layout, &collapsed)?;
                    sandwich_views.push(node);
                    node = view.input;
                }
                _ => return None,
            }
        };
        chain.reverse();

        let value = reduce.plain_input()?;
        let axis = reduce.axis;
        if reduce.shape != builder.shape {
            return None;
        }
        if let Some(existing) = builder.axis
            && existing != axis
        {
            return None;
        }
        if !layout_matches(
            Some(&layout),
            &keepdim_broadcast_layout(&builder.shape, axis),
        ) {
            return None;
        }

        // The same reduction read again reuses its phase.
        let scalar_ref =
            |phase: usize, rank: usize| NaryExpr::input(SCALAR_SLOT_BASE + phase, rank);
        if let Some(phase) = builder
            .phases
            .iter()
            .position(|(base, _)| *base == chain_nodes.first().copied().unwrap_or(node))
        {
            builder.add_members(views);
            builder.add_members(sandwich_views);
            return Some(scalar_ref(phase, builder.shape.len()));
        }

        builder.axis = Some(axis);
        let rewrite = self.absorb_operand(graph, builder, value)?;
        let expression = rewrite_slots(&NaryExpr::input(0, builder.shape.len()), &[rewrite])?;

        let mut post = reduce.post_element_wise.functions.clone();
        post.extend(chain);
        let post_chain = crate::nary_wise::UnaryFunctionChain::new(
            post,
            reduce.post_element_wise.input_datatype(),
        );

        let phase_key = chain_nodes.first().copied().unwrap_or(node);
        builder.add_members(views);
        builder.add_members(sandwich_views);
        builder.add_members(chain_nodes);
        builder.add_member(node);
        let phase_index = builder.phases.len();
        builder.phases.push((
            phase_key,
            RowReduce {
                expression,
                function: reduce.function.clone(),
                post_chain,
            },
        ));
        Some(scalar_ref(phase_index, builder.shape.len()))
    }
}
