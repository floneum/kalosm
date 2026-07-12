//! Greedy extraction: choose one form per execution node, maximizing fusion
//! under the profile's legality gates.
//!
//! Selection state remains per observation so liveness constraints are
//! explicit, while equivalent observations may reference the same e-class.
//! Every observation starts at its identity selection.
//! Two candidate sources drive switches:
//! - **Pre-generated alternatives** (Stage-1 recognition rules) already in
//!   the node's class, ranked by [`kind_rank`].
//! - **Fusion generators** (Stage 2): per-node fusion rules consulted with
//!   the *live* consumer counts. Successful generations are recorded into
//!   the e-graph as alternatives and committed as switches.
//!
//! Consumer counts are multisets over the current selection, initialized
//! from the identity selections (one entry per read occurrence). A switch's
//! kill set is the transitive closure of producers whose counts drop to
//! zero; targets are never killed (they materialize regardless, even when a
//! consumer also inlined their expression).
//!
//! Determinism: the worklist is seeded in provenance order, candidates are
//! ranked by (kind rank, payload id), generators run in a fixed per-node
//! order, and no tie-break ever consults e-graph insertion order or map
//! iteration order. The generators' gates are antitone in the (only ever
//! decreasing) consumer counts, so the worklist converges to a unique
//! greatest fixpoint regardless of processing order.

use egg::Language;

use super::EGraphDriver;
use super::lang::{FusorLang, Prov};
use super::rules_fuse::{FusionView, Stage2Ctx};

/// What extraction chose for one execution node.
#[derive(Debug, Clone, PartialEq)]
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

/// Alternative-kind priority: lower ranks win. Mirrors the destructive
/// optimizer's commitment order — a recognized attention program supersedes
/// the matmul it was recognized from, which supersedes the composed form.
fn kind_rank(enode: &FusorLang) -> u32 {
    match enode {
        FusorLang::GraphOp(_, _, _) => 0,
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
    fn new(driver: &EGraphDriver) -> Self {
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
            .map(|&child| driver.prov_of_class(child).0)
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
        let mut overlay: rustc_hash::FxHashMap<u32, i64> = Default::default();
        for child in self.selected_child_provs(driver, prov) {
            *overlay.entry(child).or_default() -= 1;
        }
        for &child in candidate.children() {
            let child_prov = driver.prov_of_class(child).0;
            *overlay.entry(child_prov).or_default() += 1;
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
            let child_prov = driver.prov_of_class(child).0;
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
    /// Stage-1 extraction: pre-generated alternatives only.
    pub(super) fn extract(&self) -> Extraction {
        // Stage 1 never mutates the e-graph during extraction, but the
        // shared loop wants `&mut self`; recognition alternatives are all
        // already present, so the generator hook is simply absent.
        let mut state = ExtractState::new(self);
        self.run_alternative_switches(&mut state);
        Extraction {
            sel: state.sel,
            needed: state.needed,
        }
    }

    pub(super) fn prov_of_class(&self, id: egg::Id) -> Prov {
        self.provs_of_class[&self.egraph.find(id)][0]
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

    /// Worklist over pre-generated class alternatives (Stage 1).
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
    fn try_alternative_switch(&self, state: &mut ExtractState, prov: Prov) -> Option<Vec<u32>> {
        let id = self.egraph.find(self.class_of[prov.0 as usize]);
        // Deterministic candidate order: kind rank, then payload id.
        let mut candidates: Vec<&FusorLang> = self.egraph[id]
            .nodes
            .iter()
            .filter(|node| !self.is_identity(node))
            .collect();
        candidates.sort_by_key(|node| (kind_rank(node), node.payload()));
        let current_rank = match &state.sel[prov.0 as usize] {
            Selection::Identity => u32::MAX,
            Selection::Alt(enode) => kind_rank(enode),
        };
        let candidate = candidates
            .first()
            .filter(|candidate| kind_rank(candidate) < current_rank)?;
        let kills = state.kills(self, prov, candidate);
        Some(state.commit(self, prov, (*candidate).clone(), &kills))
    }

    /// Stage-2 extraction: the fusion-generator worklist. Seeded with every
    /// profile-eligible node in provenance order; after each committed
    /// switch, everything whose situation changed — the node itself, old and
    /// new producers, killed nodes' children, and consumers reachable
    /// through views — re-enters the worklist. Counts only decrease and the
    /// generators' gates are antitone in them, so the loop converges to the
    /// greatest fixpoint regardless of order: maximal fusion under the
    /// profile's legality gates.
    pub(super) fn extract_with_fusion(&mut self, ctx: &Stage2Ctx<'_>) -> Extraction {
        let mut state = ExtractState::new(self);
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

            let generated = {
                let view = FusionView::new(self, &state, ctx);
                view.generate(Prov(prov))
            };
            let Some(variant) = generated else {
                continue;
            };
            // Record the fused form as an alternative of this node's class,
            // then commit the switch with live counts.
            let enode = self.mint_alternative_unique(Prov(prov), variant);
            let kills = state.kills(self, Prov(prov), &enode);
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
        Extraction {
            sel: state.sel,
            needed: state.needed,
        }
    }
}
