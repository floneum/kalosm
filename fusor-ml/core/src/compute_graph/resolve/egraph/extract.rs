//! Cost-guided extraction: choose one legal form per execution node,
//! minimizing estimated GPU runtime.
//!
//! Selection state remains per observation so liveness constraints are
//! explicit, while equivalent observations may reference the same e-class.
//! Every observation starts at its identity selection. Fusion generators
//! drive every switch: legal alternatives are planned once per
//! allocation-independent local window and reused by repeated layers.
//! Successful generations are recorded into the value e-graph and committed
//! as switches.
//!
//! Consumer counts are multisets over the current selection, initialized
//! from the identity selections (one entry per read occurrence). A switch's
//! kill set is the transitive closure of producers whose counts drop to
//! zero; targets are never killed (they materialize regardless, even when a
//! consumer also inlined their expression).
//!
//! The cost tuple is lexicographic: dispatch count, materialized bytes, then
//! estimated arithmetic work. Determinism comes from provenance-order
//! worklists, fixed generator order and generation-order tie breaks; no
//! decision consults hash-map iteration order.

use egg::Language;

use super::EGraphDriver;
use super::interner::variant_dependencies;
use super::lang::{FusorLang, Prov};
use super::rules_fuse::{FusionCtx, FusionView};
use super::structural_memo::{FusionPlanMemo, PlanLookup};

/// What extraction chose for one execution node.
#[derive(Debug, Clone)]
pub(super) enum Selection {
    Identity,
    Alt(FusorLang),
}

pub(super) struct Extraction {
    /// Indexed by `Prov`.
    pub(super) sel: Vec<Selection>,
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
            .filter_map(|(prov, selection)| match selection {
                Selection::Identity => None,
                Selection::Alt(enode) => Some((Prov(prov as u32), enode)),
            })
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
    /// Every ingested observation starts needed and at its identity form.
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

    /// Child provenances of the current selection, one entry per read
    /// occurrence.
    pub(super) fn selected_child_provs(&self, driver: &EGraphDriver, prov: Prov) -> Vec<u32> {
        self.selected_enode(driver, prov)
            .children()
            .iter()
            .map(|&child| driver.prov_of_class(child, &self.needed).0)
            .collect()
    }

    /// Child provenances of a node the cascade is killing. A child class
    /// whose observations are all dead was killed earlier in the same
    /// cascade — a chain-folding generator kills a producer and that
    /// producer's own inputs — and its counts no longer reach any live
    /// selection, so it drops out instead of resolving.
    fn killed_child_provs(&self, driver: &EGraphDriver, prov: Prov) -> Vec<u32> {
        self.selected_enode(driver, prov)
            .children()
            .iter()
            .filter_map(|&child| driver.live_prov_of_class(child, &self.needed))
            .map(|prov| prov.0)
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
            for child in self.killed_child_provs(driver, Prov(dead)) {
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
    pub(super) fn prov_of_class(&self, id: egg::Id, needed: &[bool]) -> Prov {
        self.live_prov_of_class(id, needed)
            .expect("a selected e-class child must retain a needed provenance")
    }

    /// The live provenance of `id`, or `None` once every observation in the
    /// class has been killed.
    pub(super) fn live_prov_of_class(&self, id: egg::Id, needed: &[bool]) -> Option<Prov> {
        self.provs_of_class[&self.egraph.find(id)]
            .iter()
            .copied()
            .find(|prov| needed[prov.0 as usize])
    }

    /// The identity e-node of a provenance: the one ingested for the
    /// execution node itself (unique per class by construction).
    pub(super) fn identity_enode(&self, prov: Prov) -> &FusorLang {
        &self.identity_enodes[prov.0 as usize]
    }

    /// The cheapest legal generator candidate for `prov` under live counts.
    fn best_fusion_candidate(
        &self,
        state: &ExtractState,
        view: &FusionView<'_>,
        prov: Prov,
    ) -> Option<super::super::ExecutionVariant> {
        view.generate_candidates(prov)
            .into_iter()
            .enumerate()
            .filter_map(|(order, variant)| {
                let kills = state.kills_for_variant(self, prov, &variant);
                if state.variant_duplicates_required_producer(self, prov, &variant, &kills) {
                    return None;
                }
                let delta = self.switch_cost_delta(state, prov, &variant, &kills);
                delta.non_worse().then_some((delta, order, variant))
            })
            .min_by_key(|(delta, order, _)| (*delta, *order))
            .map(|(_, _, variant)| variant)
    }

    fn verify_shared_plan(
        &self,
        state: &ExtractState,
        view: &FusionView<'_>,
        prov: Prov,
        shared: Option<&super::super::ExecutionVariant>,
    ) {
        let fresh = self.best_fusion_candidate(state, view, prov);
        match (shared, &fresh) {
            (None, None) => {}
            (Some(shared), Some(generated)) => {
                assert!(
                    super::interner::planning_payload_eq(shared, generated),
                    "shared plan diverges from regeneration: window horizon misses generator input (prov {})",
                    prov.0
                );
                assert_eq!(
                    super::interner::variant_dependencies(shared),
                    super::interner::variant_dependencies(generated),
                    "shared plan dependencies diverge (prov {})",
                    prov.0
                );
            }
            _ => panic!(
                "shared plan presence diverges from regeneration (prov {}, shared={}, fresh={})",
                prov.0,
                shared.is_some(),
                fresh.is_some()
            ),
        }
    }

    /// Fusion extraction worklist, seeded with every
    /// fusion-eligible node in provenance order; after each committed
    /// switch, everything whose situation changed — the node itself, old and
    /// new producers, killed nodes' children, and consumers reachable
    /// through views — re-enters the worklist. Counts only decrease and the
    /// generators' gates are antitone in them, so the loop converges to the
    /// greatest fixpoint regardless of order: maximal legal fusion.
    pub(super) fn extract_with_fusion(&mut self, ctx: &FusionCtx<'_>) -> Extraction {
        let mut state = ExtractState::new(self);
        let device = ctx.graph.device();
        let mut plans = FusionPlanMemo::for_config(device.config());
        // The window horizon must cover everything a generator observes;
        // this tripwire proves it by regenerating and comparing on every
        // hit, per-resolve and device-store alike.
        let verify_sharing = device.config().verify_plan_sharing;
        let store = device.fusion_plan_store();
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
                        if verify_sharing {
                            self.verify_shared_plan(&state, &view, Prov(prov), result.as_ref());
                        }
                        let spec = plans.known_spec(&instance);
                        (result, instance.root, spec)
                    }
                    PlanLookup::Miss => {
                        let key = plans.window_key(instance.root, self);
                        let result = match store.instantiate(key, &instance, &view) {
                            Some(result) => {
                                plans.note_store_hit();
                                if verify_sharing {
                                    self.verify_shared_plan(
                                        &state,
                                        &view,
                                        Prov(prov),
                                        result.as_ref(),
                                    );
                                }
                                plans.record(&instance, &view, result.as_ref());
                                result
                            }
                            None => {
                                plans.note_store_miss();
                                let result = self.best_fusion_candidate(&state, &view, Prov(prov));
                                if let Some(decision) = plans.record(&instance, &view, result.as_ref())
                                {
                                    store.record(key, decision);
                                }
                                result
                            }
                        };
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
        if device.config().spike_hoisting {
            let sharing = plans.stats();
            tracing::info!(
                "hoisting_spike_windows stub_depth={} windows={} unique={} hits={} misses={} store_hits={} store_misses={} capture_us={} capture_ns_per_window={}",
                plans.stub_depth(),
                sharing.windows,
                sharing.unique_windows,
                sharing.hits,
                sharing.misses,
                sharing.store_hits,
                sharing.store_misses,
                plans.capture_time().as_micros(),
                plans.capture_time().as_nanos() / sharing.windows.max(1) as u128,
            );
        }
        if device.config().trace_resolve_host {
            let sharing = plans.stats();
            let cost = self.extraction_cost(&state);
            tracing::info!(
                "resolve_egg_plans windows={} unique={} hits={} misses={} store_hits={} store_misses={} templates={} negative={} payloads={} specs={} dispatches={} bytes={} work={}",
                sharing.windows,
                sharing.unique_windows,
                sharing.hits,
                sharing.misses,
                sharing.store_hits,
                sharing.store_misses,
                sharing.templates,
                sharing.negative_templates,
                self.egraph.analysis.payloads.payload_count(),
                self.egraph.analysis.payloads.spec_count(),
                cost.dispatches,
                cost.materialized_bytes,
                cost.work,
            );
            if sharing.unshareable > 0 {
                tracing::info!(
                    "resolve_egg_plans_unshareable rewrites={}",
                    sharing.unshareable
                );
            }
        }
        Extraction {
            sel: state.sel,
            needed: state.needed,
        }
    }
}
