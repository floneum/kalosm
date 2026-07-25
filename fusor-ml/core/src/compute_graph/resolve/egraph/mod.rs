//! Value e-graph optimizer for operation fusion.
//!
//! Rewrites are strictly additive: a fusion generator's alternative e-node is
//! unioned into the root's class and nothing is ever removed. A GPU-oriented
//! extractor chooses among those alternatives, and one physical planner
//! applies the chosen terms back onto the execution graph.
//!
//! Pure e-node identity is semantic: operator payload plus child e-classes.
//! Allocation-backed leaves use allocation identity. An allocation-independent
//! structural interner lets isomorphic repeated layers share rewrite templates
//! without ever sharing their values. Multiple
//! execution-graph observations of one value e-class are materialized once
//! and cached under every observed `NodeIndex`.

mod analysis;
mod apply;
pub(super) mod compose;
mod cost;
mod extract;
mod ingest;
mod interner;
mod lang;
mod rules_fuse;
mod rules_fuse_matmul;
mod structural_memo;
pub(crate) use structural_memo::FusionPlanStore;

use egg::{EGraph, Id};
use rustc_hash::FxHashMap;

use self::analysis::FusorAnalysis;
use self::lang::FusorLang;
use super::{ExecutionVariant, Resolver};
use crate::compute_graph::{ComputeGraphInner, NodeIndex};

/// Owns the resolve's value e-graph plus the provenance bookkeeping
/// connecting it to the resolver's execution graph.
pub(super) struct EGraphDriver {
    egraph: EGraph<FusorLang, FusorAnalysis>,
    /// Provenance -> e-class id (as returned at add time; canonicalize with
    /// `egraph.find` after unions).
    class_of: Vec<Id>,
    identity_enodes: Vec<FusorLang>,
    identity_variants: Vec<Option<ExecutionVariant>>,
    /// Inner-graph node -> provenance.
    prov_of: FxHashMap<NodeIndex, lang::Prov>,
    provs_of_class: FxHashMap<Id, Vec<lang::Prov>>,
}

impl EGraphDriver {
    /// The API-emitted operation for one observation, before any egg
    /// alternative is selected.
    fn identity_variant(&self, prov: lang::Prov) -> Option<&ExecutionVariant> {
        self.identity_variants[prov.0 as usize].as_ref()
    }

    fn refresh_prov_classes(&mut self) {
        self.provs_of_class.clear();
        for (index, &class) in self.class_of.iter().enumerate() {
            self.provs_of_class
                .entry(self.egraph.find(class))
                .or_default()
                .push(lang::Prov(index as u32));
        }
    }
    /// Fusion mint: skips semantic payload dedup and can reuse a structural
    /// spec learned from an isomorphic earlier occurrence.
    fn mint_alternative_unique(
        &mut self,
        root: lang::Prov,
        variant: ExecutionVariant,
        known_spec: Option<interner::SpecId>,
    ) -> FusorLang {
        let children: Vec<Id> = interner::variant_dependencies(&variant)
            .into_iter()
            .map(|dep| {
                self.class_for(dep)
                    .expect("alternative dependencies must already be ingested")
            })
            .collect();
        let enode = ingest::enode_for(
            &mut self.egraph.analysis,
            &variant,
            root,
            children,
            false,
            known_spec,
        );
        let id = self.egraph.add(enode.clone());
        let root_id = self.class_of[root.0 as usize];
        self.egraph.union(root_id, id);
        enode
    }
}

impl Resolver {
    /// Recognize specialized operations, extend them with explicit cluster
    /// builders, and extract fusion alternatives through one value e-graph.
    /// Allocation-independent structural templates make repeated-layer
    /// fusion planning proportional to unique local structure.
    pub(super) fn optimize_operations(&mut self, graph: &mut ComputeGraphInner) {
        let device = graph.device();
        let config = device.config();
        let recognized = !config
            .spike_no_recognition
            .is_some_and(|budget| self.execution_graph.node_count() <= budget);
        let recognition_start = std::time::Instant::now();
        if recognized {
            self.recognize_all(graph);
        }
        let recognition = recognition_start.elapsed();
        self.optimize_phases.recognition += recognition;
        #[cfg(feature = "graphvis")]
        if let Some(dir) = &config.dump_stages {
            super::visualize::dump_stage(
                dir,
                &self.execution_graph,
                super::visualize::Stage::Recognized,
            );
        }

        let extraction_start = std::time::Instant::now();
        let mut driver = EGraphDriver::ingest(self, graph);
        if config.spike_hoisting {
            let analysis = &driver.egraph.analysis;
            tracing::info!(
                "hoisting_spike_ingest recognized={} exec_nodes={} recognition_us={} ingest_us={} provs={} enodes={} classes={} payloads={} specs={}",
                recognized,
                self.execution_graph.node_count(),
                recognition.as_micros(),
                extraction_start.elapsed().as_micros(),
                analysis.facts.len(),
                driver.egraph.total_number_of_nodes(),
                driver.egraph.number_of_classes(),
                analysis.payloads.payload_count(),
                analysis.payloads.spec_count(),
            );
        }
        let extraction = {
            let ctx = rules_fuse::FusionCtx {
                graph,
                layouts: std::cell::RefCell::new(Default::default()),
            };
            driver.extract_with_fusion(&ctx)
        };
        driver.egraph.rebuild();
        driver.refresh_prov_classes();
        self.apply_egraph_deltas(graph, &driver, &extraction);
        self.coalesce_equivalent_eclasses(graph, &driver);
        self.optimize_phases.extraction += extraction_start.elapsed();
        #[cfg(feature = "graphvis")]
        if let Some(dir) = &config.dump_stages {
            super::visualize::dump_stage(
                dir,
                &self.execution_graph,
                super::visualize::Stage::Extracted,
            );
        }
    }

    /// Every matcher that claims a composed cluster, in nesting order:
    /// attention reads recognized contractions, and the assign chains close
    /// over whatever the earlier passes left.
    pub(super) fn recognize_all(&mut self, graph: &mut ComputeGraphInner) {
        self.recognize_contractions(graph);
        self.recognize_embeddings(graph);
        self.recognize_attention(graph);
        self.fuse_row_programs(graph);
        self.recognize_assign_chains(graph);
    }

    /// Collapse execution nodes that ingestion places in the same semantic
    /// e-class. The representative performs the work; every other inner
    /// `NodeIndex` remains an observation of that result in `shared_outputs`.
    fn coalesce_equivalent_eclasses(
        &mut self,
        graph: &mut ComputeGraphInner,
        driver: &EGraphDriver,
    ) {
        let groups: Vec<Vec<lang::Prov>> = driver.provs_of_class.values().cloned().collect();
        // Removing a duplicate can leave its dependencies dead. Do not prune
        // those dependencies until every e-class from this ingestion snapshot
        // has been coalesced: a dead dependency may itself be a duplicate in a
        // later group, whose snapshotted `facts.exec` must remain valid long
        // enough to register all of its shared output observations.
        let mut potentially_dead = Vec::new();
        for group in groups {
            let executions: Vec<_> = group
                .into_iter()
                .filter_map(|prov| {
                    let facts = driver.egraph.analysis.facts_of(prov);
                    let exec = facts.exec?;
                    let variant = &self.execution_graph.node_weight(exec)?.variant;
                    matches!(
                        variant,
                        ExecutionVariant::Elementwise(_)
                            | ExecutionVariant::Reduce(_)
                            | ExecutionVariant::View(_)
                            | ExecutionVariant::MatMul(_)
                            | ExecutionVariant::QMatMul(_)
                            | ExecutionVariant::QEmbedding(_)
                            | ExecutionVariant::RowProgram(_)
                    )
                    .then_some(exec)
                })
                .collect();
            let Some((&representative, duplicates)) = executions.split_first() else {
                continue;
            };
            let representative_inner = self.execution_graph[representative].inner_idx;
            for &duplicate in duplicates {
                if !self.execution_graph.contains_node(duplicate) {
                    continue;
                }
                let duplicate_inner = self.execution_graph[duplicate].inner_idx;
                let consumers: Vec<_> = self
                    .execution_graph
                    .neighbors_directed(duplicate, petgraph::Direction::Outgoing)
                    .collect();
                let dependencies: Vec<_> = self
                    .execution_graph
                    .neighbors_directed(duplicate, petgraph::Direction::Incoming)
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
                self.execution_graph.remove_node(duplicate);
                self.node_mapping.remove(&duplicate_inner);
                self.shared_outputs
                    .entry(representative_inner)
                    .or_default()
                    .push(duplicate_inner);
                graph.add_dependency_edge(representative_inner, duplicate_inner);
                if let Some(recorder) = &self.recorder {
                    recorder
                        .borrow_mut()
                        .record_physical_edge(representative_inner, duplicate_inner);
                }
                potentially_dead.extend(dependencies);
            }
        }
        for dependency in potentially_dead {
            self.remove_node_if_dead(dependency);
        }
    }
}
