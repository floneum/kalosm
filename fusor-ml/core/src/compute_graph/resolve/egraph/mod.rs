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
//! Every e-node is salted with the provenance of the execution node it
//! denotes, so distinct graph nodes never share an e-class: kernel counts are
//! byte-identical to the destructive optimizer (no accidental CSE). Dropping
//! the salt is the future opt-in CSE switch.

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
    /// Inner-graph node -> provenance.
    prov_of: FxHashMap<NodeIndex, lang::Prov>,
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
}
