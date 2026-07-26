//! Sinking unary chains that sit behind pure views into their producer's
//! epilogue.
//!
//! A matmul's post epilogue runs over the workgroup's output tile before the
//! store, so an activation fused there costs nothing: no second kernel, no
//! second trip through memory for an activation-sized tensor. The e-graph
//! generator already fuses `unary(matmul)`, but only when the unary chain
//! reads the matmul *directly* — and it rarely does. Every convolution
//! reassembles its `(rows, out_channels)` matmul into
//! `(batch, out_channels, ...spatial)` first, so the activation the caller
//! writes lands behind a reshape and a permute and the fusion never fires.
//!
//! Pure views only relabel coordinates, and a unary function commutes with
//! relabelling: `f(view(x)) == view(f(x))`. So the chain can move to the
//! matmul and the views stay exactly where they are. This runs as a
//! recognizer rather than a generator because it rewrites *two* nodes — the
//! matmul gains the epilogue and the elementwise node collapses into an
//! observation of its own input — and a generator may only return a new
//! variant for the node it was asked about.

use super::{ExecutionNodeIndex, ExecutionVariant, Resolver};
use crate::compute_graph::{ComputeGraphInner, NodeIndex};
use crate::nary_wise::UnaryFunctionChain;

impl Resolver {
    /// Move unary chains that read a matmul through pure views into that
    /// matmul's post epilogue.
    pub(super) fn sink_unary_chains_into_matmuls(&mut self, graph: &mut ComputeGraphInner) {
        let before = self.sunk_chains;
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
            self.try_sink_unary_chain(graph, node);
        }
        if graph.device().config().trace_resolve && self.sunk_chains != before {
            tracing::info!("sink_views chains={}", self.sunk_chains - before);
        }
    }

    fn try_sink_unary_chain(&mut self, graph: &mut ComputeGraphInner, node: ExecutionNodeIndex) {
        if !self.execution_graph.contains_node(node) {
            return;
        }
        let ExecutionVariant::Elementwise(nary) = &self.execution_graph[node].variant else {
            return;
        };
        let output_datatype = nary.output_datatype;
        let Some(chain) = nary.try_extract_unary_chain() else {
            return;
        };
        if chain.functions.functions.is_empty() {
            return;
        }
        // A chain whose own value is observed elsewhere still collapses: the
        // views below it produce the same buffer, and the node stays readable
        // as an observation of its input. What must not be observed is the
        // matmul's *un-activated* output, so every hop from the matmul up to
        // this node has to feed only the next hop.
        let Some(producer) = self.private_view_chain_to_matmul(graph, chain.value, node) else {
            return;
        };
        if self.check_cached(graph, self.execution_graph[producer].inner_idx) {
            return;
        }
        let ExecutionVariant::MatMul(matmul) = &self.execution_graph[producer].variant else {
            return;
        };
        // Only dtype-preserving chains belong after the cooperative store.
        if matmul.datatype != output_datatype {
            return;
        }
        let mut fused = matmul.clone();
        let mut functions = fused.post_element_wise.functions.clone();
        functions.extend(chain.functions.functions.iter().cloned());
        fused.post_element_wise =
            UnaryFunctionChain::new(functions, fused.post_element_wise.input_datatype());
        self.execution_graph[producer].variant = ExecutionVariant::MatMul(fused);
        self.alias_to_input(graph, node, chain.value);
        self.sunk_chains += 1;
    }

    /// The matmul at the base of a chain of pure views reaching `input`, when
    /// every node on that chain — the matmul included — is read by exactly one
    /// consumer, ending at `consumer`. `None` when anything else observes an
    /// intermediate value, when a hop is not a layout-only view, or when the
    /// base is not a matmul.
    fn private_view_chain_to_matmul(
        &self,
        graph: &ComputeGraphInner,
        input: NodeIndex,
        consumer: ExecutionNodeIndex,
    ) -> Option<ExecutionNodeIndex> {
        let mut current = self.get_input_node_in_exec_graph(input)?;
        let mut reader = consumer;
        loop {
            if self
                .execution_graph
                .neighbors_directed(current, petgraph::Direction::Outgoing)
                .any(|other| other != reader)
            {
                return None;
            }
            // Every node up to (not including) the chain itself takes the
            // activation into its value, so none of them may be observed from
            // outside this resolve: a handle the caller still holds, or a
            // value this resolve was asked to produce.
            let inner = self.execution_graph[current].inner_idx;
            if self.targets.contains(&inner)
                || graph
                    .nodes
                    .nodes
                    .node_weight(inner)
                    .is_some_and(|node| node.reference_count > 0)
            {
                return None;
            }
            match &self.execution_graph[current].variant {
                ExecutionVariant::MatMul(_) => return Some(current),
                ExecutionVariant::View(view) => {
                    // Layout-only: a stage stack that composes to one layout
                    // relabels coordinates and nothing more.
                    view.composed_layout()?;
                    let next = self.get_input_node_in_exec_graph(view.input)?;
                    reader = current;
                    current = next;
                }
                _ => return None,
            }
        }
    }

    /// Collapse `node` into an observation of `input`: its value is now
    /// exactly what `input` produces. Consumers read `input`'s execution node,
    /// and `node`'s own index stays resolvable through `shared_outputs`.
    fn alias_to_input(
        &mut self,
        graph: &mut ComputeGraphInner,
        node: ExecutionNodeIndex,
        input: NodeIndex,
    ) {
        let Some(representative) = self.get_input_node_in_exec_graph(input) else {
            return;
        };
        let inner = self.execution_graph[node].inner_idx;
        let consumers: Vec<ExecutionNodeIndex> = self
            .execution_graph
            .neighbors_directed(node, petgraph::Direction::Outgoing)
            .collect();
        for consumer in consumers {
            if consumer != representative
                && self
                    .execution_graph
                    .find_edge(representative, consumer)
                    .is_none()
            {
                self.execution_graph.add_edge(representative, consumer, ());
            }
        }
        self.execution_graph.remove_node(node);
        self.node_mapping.remove(&inner);
        self.shared_outputs.entry(input).or_default().push(inner);
        graph.add_dependency_edge(input, inner);
        if let Some(recorder) = &self.recorder {
            recorder.borrow_mut().record_physical_edge(input, inner);
        }
    }
}
