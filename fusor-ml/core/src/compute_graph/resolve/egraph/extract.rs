//! Cost-guided extraction: choose one legal form per execution node,
//! minimizing estimated GPU runtime.
//!
//! Selection state remains per observation so liveness constraints are
//! explicit, while equivalent observations may reference the same e-class.
//! Every observation starts at its identity selection.
//! Two candidate sources drive switches:
//! - **Pre-generated alternatives** from saturation already in the node's
//!   class.
//! - **Fusion generators**: legal alternatives are planned once
//!   per allocation-independent local window and reused by repeated layers.
//!   Successful generations are recorded into the value e-graph and
//!   committed as switches.
//!
//! Consumer counts are multisets over the current selection, initialized
//! from the identity selections (one entry per read occurrence). A switch's
//! kill set is the transitive closure of producers whose counts drop to
//! zero; targets are never killed (they materialize regardless, even when a
//! consumer also inlined their expression).
//!
//! The cost tuple is lexicographic: dispatch count, materialized bytes, then
//! estimated arithmetic work. Determinism comes from provenance-order
//! worklists, fixed rule order and payload-id tie breaks; no decision consults
//! hash-map iteration order.

use egg::Language;

use super::super::Resolver;
use super::EGraphDriver;
use super::interner::variant_dependencies;
use super::lang::{FusorLang, Prov};
use super::rules_fuse::{FusionCtx, FusionView};
use super::structural_memo::{FusionPlanMemo, PlanLookup};

/// What extraction chose for one execution node.
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Selection {
    Identity,
    Alt(FusorLang),
}

pub(super) struct Extraction {
    /// Indexed by `Prov`.
    pub(super) sel: Vec<Selection>,
    /// Selection already materialized in the execution graph when this
    /// extraction began. Only changes from this baseline need applying.
    baseline: Vec<Selection>,
    /// Indexed by `Prov`; false = killed (no longer materialized).
    pub(super) needed: Vec<bool>,
}

impl Extraction {
    /// The non-identity selections of needed nodes, in provenance order.
    pub(super) fn deltas(&self) -> impl Iterator<Item = (Prov, &FusorLang)> {
        self.sel
            .iter()
            .enumerate()
            .filter(|(prov, _)| self.needed[*prov])
            .filter(|(prov, selection)| **selection != self.baseline[*prov])
            .filter_map(|(prov, selection)| match selection {
                Selection::Identity => None,
                Selection::Alt(enode) => Some((Prov(prov as u32), enode)),
            })
    }
}

/// Stable tie-break between equal-cost recognition alternatives.
#[cfg(test)]
fn kind_rank(enode: &FusorLang) -> u32 {
    match enode {
        FusorLang::RowProgram(_, _, _) => 0,
        FusorLang::QMatMul(_, _, _) => 1,
        FusorLang::QEmbedding(_, _, _) => 2,
        FusorLang::MatMul(_, _, _) => 3,
        FusorLang::Region(_, _, _) => 4,
        FusorLang::Reduce(_, _, _) => 5,
        FusorLang::Elementwise(_, _, _) => 6,
        FusorLang::View(_, _, _) => 7,
        FusorLang::Assign(_, _, _) => 8,
        FusorLang::TensorLeaf(..) | FusorLang::Boundary(..) | FusorLang::QMatrixLeaf(..) => 9,
    }
}

/// Live extraction state, indexed by `Prov`.
pub(super) struct ExtractState {
    pub(super) sel: Vec<Selection>,
    pub(super) needed: Vec<bool>,
    /// Multiset consumer count under the current selection.
    pub(super) reads: Vec<u32>,
    /// Reverse index: consumers (by prov) of each node under the current
    /// selection, one entry per read occurrence.
    pub(super) consumers: Vec<Vec<u32>>,
}

impl ExtractState {
    #[cfg(test)]
    pub(super) fn new(driver: &EGraphDriver) -> Self {
        let count = driver.egraph.analysis.facts.len();
        let mut state = ExtractState {
            sel: vec![Selection::Identity; count],
            needed: vec![true; count],
            reads: vec![0; count],
            consumers: vec![Vec::new(); count],
        };
        for prov in 0..count as u32 {
            for child in state.selected_child_provs(driver, Prov(prov)) {
                state.reads[child as usize] += 1;
                state.consumers[child as usize].push(prov);
            }
        }
        state
    }

    /// Reconstruct selection/liveness from the execution graph after the
    /// recognition builders have committed their chosen structural forms.
    /// New row-program/assign alternatives are inserted into the same value
    /// e-graph; removed cluster intermediates stay unneeded.
    pub(super) fn from_execution(driver: &mut EGraphDriver, resolver: &Resolver) -> Self {
        let count = driver.egraph.analysis.facts.len();
        let mut state = ExtractState {
            sel: vec![Selection::Identity; count],
            needed: vec![false; count],
            reads: vec![0; count],
            consumers: vec![Vec::new(); count],
        };

        for index in 0..count {
            let prov = Prov(index as u32);
            let facts = driver.egraph.analysis.facts_of(prov);
            let Some(&exec) = resolver.node_mapping.get(&facts.inner) else {
                // Cached boundaries are not execution nodes but remain
                // available as leaves. Removed cluster intermediates have an
                // original exec index and stay dead.
                state.needed[index] = facts.exec.is_none();
                continue;
            };
            if !resolver.execution_graph.contains_node(exec) {
                continue;
            }
            let variant = resolver.execution_graph[exec].variant.clone();
            let enode = driver.ensure_current_variant(prov, variant);
            state.sel[index] = if driver.is_identity(&enode) {
                Selection::Identity
            } else {
                Selection::Alt(enode)
            };
            state.needed[index] = true;
        }

        // Adding the current variants can merge equivalent root classes.
        // Canonicalize before translating their child classes back to
        // provenances for read and liveness accounting.
        driver.egraph.rebuild();
        driver.refresh_prov_classes();

        for prov in 0..count as u32 {
            if !state.needed[prov as usize] {
                continue;
            }
            for child in state.selected_child_provs(driver, Prov(prov)) {
                state.reads[child as usize] += 1;
                state.consumers[child as usize].push(prov);
            }
        }
        state
    }

    /// Child provenances of the current selection, one entry per read
    /// occurrence.
    pub(super) fn selected_child_provs(&self, driver: &EGraphDriver, prov: Prov) -> Vec<u32> {
        self.selected_enode(driver, prov)
            .children()
            .iter()
            .map(|&child| driver.prov_of_class(child, &self.needed).0)
            .collect()
    }

    pub(super) fn selected_enode<'d>(
        &'d self,
        driver: &'d EGraphDriver,
        prov: Prov,
    ) -> &'d FusorLang {
        match &self.sel[prov.0 as usize] {
            Selection::Identity => driver.identity_enode(prov),
            Selection::Alt(enode) => enode,
        }
    }

    /// The transitive kill set of switching `prov` to `candidate`. Targets
    /// and leaves are never killed (they stay needed; a target producer that
    /// stops being read still materializes, duplicating compute exactly as
    /// the destructive optimizer does).
    fn kills(&self, driver: &EGraphDriver, prov: Prov, candidate: &FusorLang) -> Vec<u32> {
        self.kills_from_child_provs(
            driver,
            prov,
            candidate
                .children()
                .iter()
                .map(|&child| driver.prov_of_class(child, &self.needed).0),
        )
    }

    fn kills_for_variant(
        &self,
        driver: &EGraphDriver,
        prov: Prov,
        candidate: &super::super::ExecutionVariant,
    ) -> Vec<u32> {
        self.kills_from_child_provs(
            driver,
            prov,
            variant_dependencies(candidate)
                .into_iter()
                .filter_map(|inner| driver.prov_of.get(&inner).map(|prov| prov.0)),
        )
    }

    /// A rewrite that inlines a materializing producer but cannot kill it
    /// (another consumer or a target still needs it) duplicates GPU work.
    /// Treat that as a hard extraction constraint rather than hoping an
    /// approximate arithmetic tie-break notices it.
    fn variant_duplicates_required_producer(
        &self,
        driver: &EGraphDriver,
        prov: Prov,
        candidate: &super::super::ExecutionVariant,
        kills: &[u32],
    ) -> bool {
        self.duplicates_required_producer(
            driver,
            prov,
            variant_dependencies(candidate)
                .into_iter()
                .filter_map(|inner| driver.prov_of.get(&inner).map(|prov| prov.0)),
            kills,
        )
    }

    fn duplicates_required_producer(
        &self,
        driver: &EGraphDriver,
        prov: Prov,
        candidate_children: impl IntoIterator<Item = u32>,
        kills: &[u32],
    ) -> bool {
        let candidate_children: rustc_hash::FxHashSet<u32> =
            candidate_children.into_iter().collect();
        self.selected_child_provs(driver, prov)
            .into_iter()
            .any(|child| {
                !candidate_children.contains(&child)
                    && !kills.contains(&child)
                    && self.needed[child as usize]
                    && driver.selection_cost(self, Prov(child)).dispatches > 0
            })
    }

    fn kills_from_child_provs(
        &self,
        driver: &EGraphDriver,
        prov: Prov,
        candidate_children: impl IntoIterator<Item = u32>,
    ) -> Vec<u32> {
        let mut overlay: rustc_hash::FxHashMap<u32, i64> = Default::default();
        for child in self.selected_child_provs(driver, prov) {
            *overlay.entry(child).or_default() -= 1;
        }
        for child in candidate_children {
            *overlay.entry(child).or_default() += 1;
        }
        let mut kills = Vec::new();
        let mut frontier: Vec<u32> = overlay
            .iter()
            .filter(|&(&p, &delta)| {
                delta < 0 && (self.reads[p as usize] as i64 + delta) <= 0 && self.needed[p as usize]
            })
            .map(|(&p, _)| p)
            .collect();
        frontier.sort_unstable();
        while let Some(dead) = frontier.pop() {
            if kills.contains(&dead) {
                continue;
            }
            let facts = &driver.egraph.analysis.facts[dead as usize];
            if facts.is_target || facts.exec.is_none() {
                // Unkillable: targets materialize regardless; leaves have no
                // dispatch to save.
                continue;
            }
            kills.push(dead);
            for child in self.selected_child_provs(driver, Prov(dead)) {
                let entry = overlay.entry(child).or_default();
                *entry -= 1;
                if (self.reads[child as usize] as i64 + *entry) <= 0
                    && self.needed[child as usize]
                    && !kills.contains(&child)
                {
                    frontier.push(child);
                }
            }
        }
        kills.sort_unstable();
        kills
    }

    /// Commit the switch. Returns provs whose situation changed: the
    /// switched node, everything whose read count changed, and the killed
    /// nodes' surviving children.
    fn commit(
        &mut self,
        driver: &EGraphDriver,
        prov: Prov,
        candidate: FusorLang,
        kills: &[u32],
    ) -> Vec<u32> {
        let mut touched = vec![prov.0];
        for child in self.selected_child_provs(driver, prov) {
            self.reads[child as usize] -= 1;
            remove_one(&mut self.consumers[child as usize], prov.0);
            touched.push(child);
        }
        for &child in candidate.children() {
            let child_prov = driver.prov_of_class(child, &self.needed).0;
            self.reads[child_prov as usize] += 1;
            self.consumers[child_prov as usize].push(prov.0);
            touched.push(child_prov);
        }
        self.sel[prov.0 as usize] = Selection::Alt(candidate);
        for &dead in kills {
            if !self.needed[dead as usize] {
                continue;
            }
            self.needed[dead as usize] = false;
            for child in self.selected_child_provs(driver, Prov(dead)) {
                self.reads[child as usize] = self.reads[child as usize].saturating_sub(1);
                remove_one(&mut self.consumers[child as usize], dead);
                touched.push(child);
            }
        }
        touched
    }
}

fn remove_one(consumers: &mut Vec<u32>, value: u32) {
    if let Some(position) = consumers.iter().position(|&c| c == value) {
        consumers.swap_remove(position);
    }
}

impl EGraphDriver {
    /// Extraction over pre-generated alternatives only.
    #[cfg(test)]
    pub(super) fn extract(&self) -> Extraction {
        // Recognition never mutates the e-graph during extraction, but the
        // shared loop wants `&mut self`; recognition alternatives are all
        // already present, so the generator hook is simply absent.
        let mut state = ExtractState::new(self);
        let baseline = state.sel.clone();
        self.run_alternative_switches(&mut state);
        Extraction {
            sel: state.sel,
            baseline,
            needed: state.needed,
        }
    }

    pub(super) fn prov_of_class(&self, id: egg::Id, needed: &[bool]) -> Prov {
        let provenances = &self.provs_of_class[&self.egraph.find(id)];
        provenances
            .iter()
            .copied()
            .find(|prov| needed[prov.0 as usize])
            .expect("a selected e-class child must retain a needed provenance")
    }

    /// The identity e-node of a provenance: the one ingested for the
    /// execution node itself (unique per class by construction).
    pub(super) fn identity_enode(&self, prov: Prov) -> &FusorLang {
        &self.identity_enodes[prov.0 as usize]
    }

    /// Whether an e-node is the ingested identity form (rather than a
    /// rule-minted alternative).
    pub(super) fn is_identity(&self, node: &FusorLang) -> bool {
        self.identity_payloads[node.prov().0 as usize] == node.payload()
    }

    /// Cost-guided worklist over pre-generated class alternatives.
    #[cfg(test)]
    fn run_alternative_switches(&self, state: &mut ExtractState) {
        let mut worklist: std::collections::VecDeque<u32> = (0..state.sel.len() as u32).collect();
        let mut queued = vec![true; state.sel.len()];
        while let Some(prov) = worklist.pop_front() {
            queued[prov as usize] = false;
            if !state.needed[prov as usize] {
                continue;
            }
            if self.egraph.analysis.facts[prov as usize].exec.is_none() {
                continue;
            }
            if let Some(touched) = self.try_alternative_switch(state, Prov(prov)) {
                for t in touched {
                    if !queued[t as usize] {
                        queued[t as usize] = true;
                        worklist.push_back(t);
                    }
                }
            }
        }
    }

    /// Try to switch `prov` to a better pre-generated alternative from its
    /// class. Returns the touched set on success.
    #[cfg(test)]
    fn try_alternative_switch(&self, state: &mut ExtractState, prov: Prov) -> Option<Vec<u32>> {
        let id = self.egraph.find(self.class_of[prov.0 as usize]);
        let candidates: Vec<&FusorLang> = self.egraph[id]
            .nodes
            .iter()
            .filter(|node| !self.is_identity(node))
            .collect();
        let (candidate, kills, _) = candidates
            .into_iter()
            .filter_map(|candidate| {
                let payload = candidate.payload()?;
                let variant = self.egraph.analysis.payloads.get(payload);
                let kills = state.kills(self, prov, candidate);
                let delta = self.switch_cost_delta(state, prov, variant, &kills);
                delta.improves().then_some((candidate, kills, delta))
            })
            .min_by_key(|(candidate, _, delta)| {
                (*delta, kind_rank(candidate), candidate.payload())
            })?;
        Some(state.commit(self, prov, candidate.clone(), &kills))
    }

    /// Fusion extraction worklist, seeded with every
    /// fusion-eligible node in provenance order; after each committed
    /// switch, everything whose situation changed — the node itself, old and
    /// new producers, killed nodes' children, and consumers reachable
    /// through views — re-enters the worklist. Counts only decrease and the
    /// generators' gates are antitone in them, so the loop converges to the
    /// greatest fixpoint regardless of order: maximal legal fusion.
    pub(super) fn extract_with_fusion(
        &mut self,
        resolver: &Resolver,
        ctx: &FusionCtx<'_>,
    ) -> Extraction {
        let mut state = ExtractState::from_execution(self, resolver);
        let baseline = state.sel.clone();
        let mut plans = FusionPlanMemo::default();
        let count = state.sel.len() as u32;
        let mut worklist: std::collections::VecDeque<u32> = (0..count)
            .filter(|&prov| {
                let view = FusionView::new(self, &state, ctx);
                view.is_seed_candidate(Prov(prov))
            })
            .collect();
        let mut queued = vec![false; count as usize];
        for &prov in &worklist {
            queued[prov as usize] = true;
        }

        while let Some(prov) = worklist.pop_front() {
            queued[prov as usize] = false;
            if !state.needed[prov as usize] {
                continue;
            }
            if self.egraph.analysis.facts[prov as usize].exec.is_none() {
                continue;
            }
            let pre_consumers: Vec<u32> = state.consumers[prov as usize].clone();

            let (generated, plan_root, known_spec) = {
                let view = FusionView::new(self, &state, ctx);
                let instance = plans.capture(self, &state, &view, Prov(prov));
                match plans.lookup(&instance, &view) {
                    PlanLookup::Hit(result) => {
                        // The window horizon must cover everything a
                        // generator observes; this tripwire proves it by
                        // regenerating and comparing on every hit.
                        if std::env::var_os("FUSOR_VERIFY_PLAN_SHARING").is_some() {
                            let fresh = view
                                .generate_candidates(Prov(prov))
                                .into_iter()
                                .enumerate()
                                .filter_map(|(order, variant)| {
                                    let kills = state.kills_for_variant(self, Prov(prov), &variant);
                                    if state.variant_duplicates_required_producer(
                                        self,
                                        Prov(prov),
                                        &variant,
                                        &kills,
                                    ) {
                                        return None;
                                    }
                                    let delta = self.switch_cost_delta(
                                        &state,
                                        Prov(prov),
                                        &variant,
                                        &kills,
                                    );
                                    delta.non_worse().then_some((delta, order, variant))
                                })
                                .min_by_key(|(delta, order, _)| (*delta, *order))
                                .map(|(_, _, variant)| variant);
                            match (&result, &fresh) {
                                (None, None) => {}
                                (Some(shared), Some(generated)) => {
                                    assert!(
                                        super::interner::planning_payload_eq(shared, generated),
                                        "shared plan diverges from regeneration: window                                          horizon misses generator input (prov {prov})"
                                    );
                                    assert_eq!(
                                        super::interner::variant_dependencies(shared),
                                        super::interner::variant_dependencies(generated),
                                        "shared plan dependencies diverge (prov {prov})"
                                    );
                                }
                                _ => panic!(
                                    "shared plan presence diverges from regeneration                                      (prov {prov}, shared={}, fresh={})",
                                    result.is_some(),
                                    fresh.is_some()
                                ),
                            }
                        }
                        let spec = plans.known_spec(&instance);
                        (result, instance.root, spec)
                    }
                    PlanLookup::Miss => {
                        let result = view
                            .generate_candidates(Prov(prov))
                            .into_iter()
                            .enumerate()
                            .filter_map(|(order, variant)| {
                                let kills = state.kills_for_variant(self, Prov(prov), &variant);
                                if state.variant_duplicates_required_producer(
                                    self,
                                    Prov(prov),
                                    &variant,
                                    &kills,
                                ) {
                                    return None;
                                }
                                let delta =
                                    self.switch_cost_delta(&state, Prov(prov), &variant, &kills);
                                delta.non_worse().then_some((delta, order, variant))
                            })
                            .min_by_key(|(delta, order, _)| (*delta, *order))
                            .map(|(_, _, variant)| variant);
                        plans.record(&instance, &view, result.as_ref());
                        (result, instance.root, None)
                    }
                }
            };
            let Some(variant) = generated else {
                continue;
            };
            let planned_kills = state.kills_for_variant(self, Prov(prov), &variant);
            if state.variant_duplicates_required_producer(
                self,
                Prov(prov),
                &variant,
                &planned_kills,
            ) || !self
                .switch_cost_delta(&state, Prov(prov), &variant, &planned_kills)
                .non_worse()
            {
                continue;
            }
            // Record the fused form as an alternative of this node's class,
            // then commit the switch with live counts.
            let enode = self.mint_alternative_unique(Prov(prov), variant, known_spec);
            let actual_spec = self
                .egraph
                .analysis
                .payloads
                .spec_of(enode.payload().expect("fused alternative has payload"));
            plans.record_spec(plan_root, actual_spec);
            let kills = state.kills(self, Prov(prov), &enode);
            debug_assert_eq!(planned_kills, kills);
            let mut touched = state.commit(self, Prov(prov), enode, &kills);
            touched.extend(kills.iter().copied());

            let view = FusionView::new(self, &state, ctx);
            // The node itself (chains continue on the fused form), consumers
            // from before and after the switch, and everything the kill
            // cascade touched: a dead producer's siblings can become
            // sole-consumed and newly fusible.
            let new_consumers = state.consumers[prov as usize].clone();
            let seeds = std::iter::once(prov)
                .chain(pre_consumers)
                .chain(new_consumers)
                .chain(touched);
            view.enqueue_downstream(&state, seeds, &mut worklist, &mut queued);
        }
        if std::env::var_os("FUSOR_TRACE_RESOLVE_HOST").is_some() {
            let sharing = plans.stats();
            let cost = self.extraction_cost(&state);
            tracing::info!(
                "resolve_egg_plans windows={} unique={} hits={} misses={} templates={} negative={} payloads={} specs={} dispatches={} bytes={} work={}",
                sharing.windows,
                sharing.unique_windows,
                sharing.hits,
                sharing.misses,
                sharing.templates,
                sharing.negative_templates,
                self.egraph.analysis.payloads.payload_count(),
                self.egraph.analysis.payloads.spec_count(),
                cost.dispatches,
                cost.materialized_bytes,
                cost.work,
            );
        }
        Extraction {
            sel: state.sel,
            baseline,
            needed: state.needed,
        }
    }
}
