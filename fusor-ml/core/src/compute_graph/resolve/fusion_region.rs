//! Multi-output region formation.
//!
//! Runs after e-graph fusion for every resolve. Grows regions from each
//! unclaimed elementwise sink by absorbing elementwise producers whose
//! *every* consumer already sits in the region — the generalization of the
//! sole-consumer rule: an externally live value (flush target / user-held
//! node) no longer blocks fusion, it is emitted as one of the region's
//! outputs while later statements read it from a register.
//!
//! Contracting a region is acyclic by construction: an absorbed producer has
//! no out-edges leaving the member set, so redirecting its remaining
//! (incoming) edges to the region node cannot create a cycle.

use super::*;
use crate::region::{ElementwiseRegionOperation, RegionStatement};

/// Where a member's input slot points after region rewriting.
enum RegionSlot {
    /// Deduplicated external input slot.
    External(usize),
    /// Another member's register value: the final `IndexedInput` slot
    /// (`inputs.len() + statement position`).
    Register(usize),
}

/// Why producers were rejected at region-growth fixpoints, accumulated per
/// resolve and reported under `FUSOR_TRACE_RESOLVE`. Only the final
/// (fixpoint-reaching) probe of each region contributes, so the counts
/// describe the frontier that actually blocked growth, not transient
/// re-scans.
#[derive(Debug, Default, Clone, Copy)]
struct RegionRejects {
    non_elementwise: u32,
    shape_mismatch: u32,
    producer_cached: u32,
    outside_consumer: u32,
    custom_indexed_read: u32,
    binding_budget: u32,
}

impl RegionRejects {
    fn add(&mut self, other: &Self) {
        self.non_elementwise += other.non_elementwise;
        self.shape_mismatch += other.shape_mismatch;
        self.producer_cached += other.producer_cached;
        self.outside_consumer += other.outside_consumer;
        self.custom_indexed_read += other.custom_indexed_read;
        self.binding_budget += other.binding_budget;
    }
}

impl Resolver {
    pub(super) fn form_elementwise_regions(&mut self, graph: &mut ComputeGraphInner) {
        let budget = graph.device().nary_direct_input_binding_budget();
        let Ok(order) = toposort(&self.execution_graph, None) else {
            return;
        };
        let position: FxHashMap<ExecutionNodeIndex, usize> = order
            .iter()
            .enumerate()
            .map(|(pos, &node)| (node, pos))
            .collect();
        let mut claimed: FxHashSet<ExecutionNodeIndex> = FxHashSet::default();
        let trace = std::env::var_os("FUSOR_TRACE_RESOLVE").is_some();
        let mut rejects = RegionRejects::default();
        let mut regions_formed = 0usize;
        let mut statements_fused = 0usize;

        for &sink in order.iter().rev() {
            if claimed.contains(&sink) || !self.execution_graph.contains_node(sink) {
                continue;
            }
            let ExecutionVariant::Elementwise(sink_op) = &self.execution_graph[sink].variant else {
                continue;
            };
            if self.check_cached(graph, self.execution_graph[sink].inner_idx) {
                continue;
            }
            let shape = sink_op.shape.clone();

            // Fixpoint absorption.
            let mut member_set: FxHashSet<ExecutionNodeIndex> = FxHashSet::default();
            member_set.insert(sink);
            loop {
                let mut probe = RegionRejects::default();
                let candidate = self.find_absorbable_producer(
                    graph,
                    &member_set,
                    &claimed,
                    &shape,
                    budget,
                    &position,
                    position[&sink],
                    &mut probe,
                );
                match candidate {
                    Some(producer) => {
                        member_set.insert(producer);
                    }
                    None => {
                        // The fixpoint probe: these producers are what
                        // actually blocked further growth.
                        rejects.add(&probe);
                        break;
                    }
                }
            }
            if member_set.len() == 1 {
                continue;
            }

            regions_formed += 1;
            statements_fused += member_set.len();
            // Topological member order = statement order.
            let mut members: Vec<ExecutionNodeIndex> = member_set.iter().copied().collect();
            members.sort_by_key(|node| position[node]);
            self.finalize_region(graph, sink, &members, &member_set, shape);
            claimed.extend(member_set);
        }

        if trace {
            tracing::info!(
                "region_fusion regions={regions_formed} statements={statements_fused} rejects={rejects:?}"
            );
        }
    }

    /// One producer of the current member set that passes every absorption
    /// gate, or `None` at fixpoint.
    #[allow(clippy::too_many_arguments)]
    fn find_absorbable_producer(
        &self,
        graph: &ComputeGraphInner,
        member_set: &FxHashSet<ExecutionNodeIndex>,
        claimed: &FxHashSet<ExecutionNodeIndex>,
        shape: &[usize],
        budget: usize,
        position: &FxHashMap<ExecutionNodeIndex, usize>,
        sink_position: usize,
        rejects: &mut RegionRejects,
    ) -> Option<ExecutionNodeIndex> {
        for &member in member_set {
            for producer in self
                .execution_graph
                .neighbors_directed(member, petgraph::Direction::Incoming)
            {
                if member_set.contains(&producer) || claimed.contains(&producer) {
                    continue;
                }
                let producer_inner = self.execution_graph[producer].inner_idx;
                let ExecutionVariant::Elementwise(producer_op) =
                    &self.execution_graph[producer].variant
                else {
                    rejects.non_elementwise += 1;
                    continue;
                };
                if producer_op.shape.as_ref() != shape {
                    rejects.shape_mismatch += 1;
                    continue;
                }
                if self.check_cached(graph, producer_inner) {
                    rejects.producer_cached += 1;
                    continue;
                }
                // Outside consumers no longer block absorption: the
                // producer's value becomes a region output (the binding
                // budget below already accounts for it). The one hazard is
                // ordering: a consumer scheduled before the region's last
                // member could need the value before the region runs — and
                // could, transitively, feed the region, a cycle. Every
                // member feeds the sink, so the sink carries the region's
                // maximum topological position; consumers strictly after it
                // are provably downstream and safe.
                if self
                    .execution_graph
                    .neighbors_directed(producer, petgraph::Direction::Outgoing)
                    .any(|consumer| {
                        !member_set.contains(&consumer)
                            && position
                                .get(&consumer)
                                .is_none_or(|&pos| pos <= sink_position)
                    })
                {
                    rejects.outside_consumer += 1;
                    continue;
                }
                // Every member reads the producer elementwise: register
                // values have no coordinates to custom-index with.
                let read_elementwise = member_set.iter().all(|&reader| {
                    let ExecutionVariant::Elementwise(reader_op) =
                        &self.execution_graph[reader].variant
                    else {
                        return true;
                    };
                    reader_op
                        .inputs
                        .iter()
                        .enumerate()
                        .filter(|(_, input)| **input == producer_inner)
                        .all(|(slot, _)| !reader_op.expression.uses_custom_indexing_for_input(slot))
                });
                if !read_elementwise {
                    rejects.custom_indexed_read += 1;
                    continue;
                }
                // Binding budget: distinct external inputs + live outputs of
                // the grown region must fit one dispatch.
                if self.region_binding_count(graph, member_set, Some(producer)) > budget {
                    rejects.binding_budget += 1;
                    continue;
                }
                return Some(producer);
            }
        }
        None
    }

    /// Distinct external inputs + emitted outputs for `member_set`
    /// (optionally grown by `extra`).
    fn region_binding_count(
        &self,
        graph: &ComputeGraphInner,
        member_set: &FxHashSet<ExecutionNodeIndex>,
        extra: Option<ExecutionNodeIndex>,
    ) -> usize {
        let mut inner_members: FxHashSet<NodeIndex> = FxHashSet::default();
        let all = member_set.iter().copied().chain(extra);
        for member in all.clone() {
            inner_members.insert(self.execution_graph[member].inner_idx);
        }
        let mut external: FxHashSet<NodeIndex> = FxHashSet::default();
        let mut outputs = 0usize;
        for member in all {
            let node = &self.execution_graph[member];
            let ExecutionVariant::Elementwise(op) = &node.variant else {
                continue;
            };
            for input in &op.inputs {
                if !inner_members.contains(input) {
                    external.insert(*input);
                }
            }
            if self.region_member_is_live(graph, member, member_set, extra) {
                outputs += 1;
            }
        }
        external.len() + outputs
    }

    /// Whether a member's value must be written out: it is externally live
    /// (user-held / pending sink) or it is the region sink (consumers
    /// outside the region read it).
    fn region_member_is_live(
        &self,
        graph: &ComputeGraphInner,
        member: ExecutionNodeIndex,
        member_set: &FxHashSet<ExecutionNodeIndex>,
        extra: Option<ExecutionNodeIndex>,
    ) -> bool {
        let inner = self.execution_graph[member].inner_idx;
        if graph
            .nodes
            .nodes
            .node_weight(inner)
            .is_some_and(|node| node.reference_count > 0)
        {
            return true;
        }
        self.execution_graph
            .neighbors_directed(member, petgraph::Direction::Outgoing)
            .any(|consumer| !member_set.contains(&consumer) && Some(consumer) != extra)
    }

    fn finalize_region(
        &mut self,
        graph: &mut ComputeGraphInner,
        sink: ExecutionNodeIndex,
        members: &[ExecutionNodeIndex],
        member_set: &FxHashSet<ExecutionNodeIndex>,
        shape: Box<[usize]>,
    ) {
        // Statement position per member (topological order).
        let statement_pos: FxHashMap<ExecutionNodeIndex, usize> = members
            .iter()
            .enumerate()
            .map(|(pos, &member)| (member, pos))
            .collect();
        let inner_of: FxHashMap<NodeIndex, ExecutionNodeIndex> = members
            .iter()
            .map(|&member| (self.execution_graph[member].inner_idx, member))
            .collect();

        // Deduplicated external inputs; slots assigned in first-use order.
        let mut inputs: Vec<NodeIndex> = Vec::new();
        let mut input_slot: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut statements = Vec::with_capacity(members.len());
        for &member in members {
            let ExecutionVariant::Elementwise(op) = self.execution_graph[member].variant.clone()
            else {
                unreachable!("region members are elementwise");
            };
            // Map each member input slot to its region slot. Register slots
            // are provisional (`inputs.len()` is not final until all members
            // are processed), so store statement positions and fix up below.
            let slot_map: Vec<RegionSlot> = op
                .inputs
                .iter()
                .map(|input| match inner_of.get(input) {
                    Some(&producer) => RegionSlot::Register(statement_pos[&producer]),
                    None => {
                        let next = inputs.len();
                        let slot = *input_slot.entry(*input).or_insert_with(|| {
                            inputs.push(*input);
                            next
                        });
                        RegionSlot::External(slot)
                    }
                })
                .collect();
            let member_inner = self.execution_graph[member].inner_idx;
            let live = member == sink
                || graph
                    .nodes
                    .nodes
                    .node_weight(member_inner)
                    .is_some_and(|node| node.reference_count > 0)
                || self
                    .execution_graph
                    .neighbors_directed(member, petgraph::Direction::Outgoing)
                    .any(|consumer| !member_set.contains(&consumer));
            statements.push((
                op.expression,
                op.output_datatype,
                slot_map,
                member_inner,
                live,
            ));
        }
        let input_count = inputs.len();
        let statements: Vec<RegionStatement> = statements
            .into_iter()
            .map(
                |(expression, datatype, slot_map, member_inner, live)| RegionStatement {
                    expression: Self::remap_region_expr(&expression, &slot_map, input_count),
                    datatype,
                    output: live.then_some(member_inner),
                },
            )
            .collect();

        let op = ElementwiseRegionOperation {
            inputs,
            statements,
            shape,
        };

        // Rewrite the graph: the sink node becomes the region; absorbed
        // members redirect their incoming external edges to the sink and
        // disappear. `node_mapping` keeps absorbed inner nodes reachable
        // (they resolve to the region node, which caches their outputs).
        let external_inputs = op.inputs.clone();
        for &member in members {
            if member == sink {
                continue;
            }
            let incoming: Vec<ExecutionNodeIndex> = self
                .execution_graph
                .neighbors_directed(member, petgraph::Direction::Incoming)
                .filter(|producer| !member_set.contains(producer))
                .collect();
            for producer in incoming {
                if !self
                    .execution_graph
                    .neighbors_directed(sink, petgraph::Direction::Incoming)
                    .any(|existing| existing == producer)
                {
                    self.execution_graph.add_edge(producer, sink, ());
                }
            }
            // A member absorbed as a region output keeps its downstream
            // ordering: its consumers now read the region's cached result.
            let outgoing: Vec<ExecutionNodeIndex> = self
                .execution_graph
                .neighbors_directed(member, petgraph::Direction::Outgoing)
                .filter(|consumer| !member_set.contains(consumer))
                .collect();
            for consumer in outgoing {
                if !self
                    .execution_graph
                    .neighbors_directed(sink, petgraph::Direction::Outgoing)
                    .any(|existing| existing == consumer)
                {
                    self.execution_graph.add_edge(sink, consumer, ());
                }
            }
            let member_inner = self.execution_graph[member].inner_idx;
            self.execution_graph.remove_node(member);
            self.node_mapping.insert(member_inner, sink);
        }
        self.execution_graph[sink].variant = ExecutionVariant::Region(op);
        self.add_physical_dependencies(graph, sink, &external_inputs);
    }

    /// Rewrite a member expression into region slots: external reads keep
    /// their (recursively remapped) index expressions; register reads drop
    /// their identity indices (guaranteed by the absorption gate) and point
    /// past the input slots into the statement `extras`.
    fn remap_region_expr(expr: &NaryExpr, slot_map: &[RegionSlot], input_count: usize) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| Self::remap_region_expr(child, slot_map, input_count))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => match &slot_map[*input_idx] {
                RegionSlot::Register(statement) => NaryExpr::IndexedInput {
                    input_idx: input_count + statement,
                    indices: Vec::new(),
                },
                RegionSlot::External(slot) => NaryExpr::IndexedInput {
                    input_idx: *slot,
                    indices: indices
                        .iter()
                        .map(|index| Self::remap_region_expr(index, slot_map, input_count))
                        .collect(),
                },
            },
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        }
    }
}
