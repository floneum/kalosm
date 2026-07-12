//! Equality-saturation optimizer replacing the recognition sweeps and the
//! destructive rewrite fixpoint.
//!
//! The pipeline runs two saturation stages over per-stage e-graphs:
//! - Stage 1 (always): recognition rules (contraction, quantized embedding,
//!   attention) — the original composed forms persist in the e-graph, so
//!   recognition no longer depends on running before other rewrites.
//! - Stage 2 (policy gated): fusion rules (view folding, nary/reduce fusion,
//!   matmul and qmatmul epilogues), registered per stage-2 profile.
//!
//! Rules are strictly additive: appliers union an alternative e-node into the
//! root's class and never remove anything. Today's destructive greedy
//! decisions are reproduced by the mimic-greedy extractor, and the chosen
//! terms are applied back onto the execution graph as in-place deltas.
//!
//! Pure e-node identity is semantic: operator payload plus child e-classes.
//! Allocation-backed leaves use allocation identity. Multiple execution-graph
//! observations of one e-class are materialized once and cached under every
//! observed `NodeIndex`.

mod analysis;
mod apply;
mod extract;
mod ingest;
mod interner;
mod lang;
mod rules_fuse;
mod rules_fuse_matmul;
mod rules_recognize;

use egg::{EGraph, Id};
use rustc_hash::FxHashMap;

use self::analysis::FusorAnalysis;
use self::lang::{FusorLang, PayloadId};
use super::{ExecutionVariant, Resolver};
use crate::compute_graph::{ComputeGraphInner, NodeIndex};

pub(super) use self::rules_fuse::{CandidateKind, ReduceFusion, Stage2Profile};

/// Owns one stage's e-graph plus the provenance bookkeeping connecting it to
/// the resolver's execution graph.
pub(super) struct EGraphDriver {
    egraph: EGraph<FusorLang, FusorAnalysis>,
    /// Provenance -> e-class id (as returned at add time; canonicalize with
    /// `egraph.find` after unions).
    class_of: Vec<Id>,
    /// Provenance -> the ingested identity e-node's payload id (`None` for
    /// tensor/boundary leaves). Distinguishes the identity form from
    /// rule-minted alternatives during extraction.
    identity_payloads: Vec<Option<PayloadId>>,
    identity_enodes: Vec<FusorLang>,
    identity_variants: Vec<Option<ExecutionVariant>>,
    /// Inner-graph node -> provenance.
    prov_of: FxHashMap<NodeIndex, lang::Prov>,
    provs_of_class: FxHashMap<Id, Vec<lang::Prov>>,
}

/// One rewrite rule: a programmatic searcher + applier pair run as whole
/// rounds over the e-graph. Rules are strictly additive — they may only add
/// alternative e-nodes and union them into existing classes. Saturation is
/// detected through the payload interner: re-applying a rule re-interns the
/// identical payload, hash-conses to the identical e-node, and adds nothing.
trait EgRule {
    /// One deterministic round; returns true if the e-graph grew.
    fn apply_round(&self, driver: &mut EGraphDriver, ctx: &RuleCtx<'_>) -> bool;
}

/// Read-only context rules match against. The inner graph and resolver are
/// immutable for the whole stage; rules mutate only the e-graph.
struct RuleCtx<'a> {
    graph: &'a ComputeGraphInner,
    resolver: &'a Resolver,
}

impl EGraphDriver {
    fn refresh_prov_classes(&mut self) {
        self.provs_of_class.clear();
        for (index, &class) in self.class_of.iter().enumerate() {
            self.provs_of_class
                .entry(self.egraph.find(class))
                .or_default()
                .push(lang::Prov(index as u32));
        }
    }
    /// Add `variant` as an alternative form of `root`'s execution node.
    /// Children are derived from the payload's own dependency list so the
    /// payload/children lockstep has one owner. Returns false when the
    /// identical alternative was already present (idempotent re-application).
    fn add_alternative(&mut self, root: lang::Prov, variant: ExecutionVariant) -> bool {
        let before = self.egraph.total_number_of_nodes();
        self.mint_alternative(root, variant);
        self.egraph.total_number_of_nodes() > before
    }

    /// Like [`Self::add_alternative`], returning the minted e-node (which the
    /// Stage-2 extractor commits as a switch).
    fn mint_alternative(&mut self, root: lang::Prov, variant: ExecutionVariant) -> FusorLang {
        self.mint_alternative_impl(root, variant, true)
    }

    /// Stage-2 mint: skips interner dedup (see `PayloadTable::push_unique`).
    fn mint_alternative_unique(
        &mut self,
        root: lang::Prov,
        variant: ExecutionVariant,
    ) -> FusorLang {
        self.mint_alternative_impl(root, variant, false)
    }

    fn mint_alternative_impl(
        &mut self,
        root: lang::Prov,
        variant: ExecutionVariant,
        dedup: bool,
    ) -> FusorLang {
        let children: Vec<Id> = interner::variant_dependencies(&variant)
            .into_iter()
            .map(|dep| {
                self.class_for(dep)
                    .expect("alternative dependencies must already be ingested")
            })
            .collect();
        let enode = ingest::enode_for(&mut self.egraph.analysis, &variant, root, children, dedup);
        let id = self.egraph.add(enode.clone());
        let root_id = self.class_of[root.0 as usize];
        self.egraph.union(root_id, id);
        enode
    }

    /// Run rule rounds to saturation (bounded by `iter_limit`). Deterministic:
    /// rules run in slice order, each round iterates provenances in order,
    /// and no wall-clock limit exists.
    fn saturate(&mut self, rules: &[&dyn EgRule], ctx: &RuleCtx<'_>, iter_limit: usize) {
        for _ in 0..iter_limit {
            let mut changed = false;
            for rule in rules {
                changed |= rule.apply_round(self, ctx);
            }
            self.egraph.rebuild();
            self.refresh_prov_classes();
            if !changed {
                return;
            }
        }
        tracing::warn!("egraph saturation hit its iteration limit ({iter_limit})");
    }
}

impl Resolver {
    /// Stage 1: recognition via equality saturation. Always runs (the
    /// decode-policy contract: recognition happens even when rewriting is
    /// skipped). Replaces the `recognize_contractions` /
    /// `recognize_embeddings` / `recognize_attention` sweeps.
    pub(super) fn recognize_via_egraph(&mut self, graph: &mut ComputeGraphInner) {
        let mut driver = EGraphDriver::ingest(self, graph);
        {
            let ctx = RuleCtx {
                graph,
                resolver: self,
            };
            let rules: [&dyn EgRule; 2] = [
                &rules_recognize::RecognizeContraction,
                &rules_recognize::RecognizeQEmbedding,
            ];
            // Recognition needs one round plus one for chained recognitions
            // (attention consumes contraction's output); the extra headroom
            // is saturation-detection slack, not a behavioral knob.
            driver.saturate(&rules, &ctx, 4);
        }
        let extraction = driver.extract();
        self.apply_egraph_deltas(graph, &driver, &extraction);
        // Attention recognition consumes the committed MatMul plus the
        // composed softmax cluster on the execution graph, exactly as the
        // destructive pipeline ordered it. Its ordering after contraction
        // recognition is structural here (it runs on the extracted graph),
        // not a phase-fragility: the matcher stays imperative in v1.
        self.recognize_attention(graph);
    }
}

impl Resolver {
    /// Stage 2: fusion via the extraction worklist consulting the pure
    /// generator transcriptions. Replaces `run_rewrite_fixpoint`; the same
    /// policy gates (including the large-decode skip) select whether this
    /// runs at all, upstream in `optimize`.
    pub(super) fn fuse_via_egraph(
        &mut self,
        graph: &mut ComputeGraphInner,
        profile: rules_fuse::Stage2Profile,
    ) {
        let mut driver = EGraphDriver::ingest(self, graph);
        let extraction = {
            let ctx = rules_fuse::Stage2Ctx {
                graph,
                profile,
                layouts: std::cell::RefCell::new(Default::default()),
            };
            driver.extract_with_fusion(&ctx)
        };
        self.apply_egraph_deltas(graph, &driver, &extraction);
    }

    /// Collapse execution nodes that ingestion places in the same semantic
    /// e-class. The representative performs the work; every other inner
    /// `NodeIndex` remains an observation of that result in `shared_outputs`.
    pub(super) fn coalesce_equivalent_eclasses(&mut self, graph: &mut ComputeGraphInner) {
        let driver = EGraphDriver::ingest(self, graph);
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
                            | ExecutionVariant::GraphOp(_)
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
