//! [`LocalSearch`] — the shipped [`Extractor`].
//!
//! It computes an admissible lower bound, seeds `sigma = argmin lb` with
//! `m = roots u {shared} u {index-space mismatch} u {InPlace}`, runs a local
//! search over `RESELECT`, `FLIP` and `RESCHEDULE` plus [`co_select`], and
//! asserts `verify_plan` on the winner.
//!
//! Every decision path iterates classes and nodes in ascending id order, with
//! no RNG and no hash-map iteration order, so extraction is a pure function of
//! the graph.
//!
//! A rule that fuses `F` values into one node yields that node plus `F` slot
//! views of it, each in a different e-class. Adopting one view alone costs
//! more than adopting none, so the single-move climb cannot walk into the
//! joint; [`co_select`] adopts them together.

use crate::lower_bound::argmin_member;
use crate::moves::{self, SchedCache};
use crate::plan::derive_plan;
use crate::realize::{self, RealizeScratch, Realized};
use fixedbitset::FixedBitSet;
use fusor2_ir::Result;
use fusor2_ir::cost::{CostModel, Picoseconds, ShapeStats};
use fusor2_ir::device::Caps;
use fusor2_ir::egraph::{ClassId, EGraph, Id};
use fusor2_ir::extract::{ExtractBudget, Extraction, Extractor, Launch, Plan};
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::ir::Op;
use fusor2_ir::ir::OpDefRegistry;
use fusor2_ir::error::Error;
use fusor2_ir::ir::level1::{Effect, L1, SchedPoint, ScheduleDomain};
use fusor2_ir::ir::level2::ArenaPlanner;
use fusor2_ir::shape::Dim;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;
use std::time::Instant;

/// The shipped extraction. Deterministic: ties break by node id, then by
/// [`fusor2_ir::extract::Move`] discriminant order, which is the order
/// [`moves::frontier`] emits them in.
pub struct LocalSearch {
    arena: Arc<dyn ArenaPlanner>,
    caps: Caps,
    registry: Option<OpDefRegistry>,
    /// Bounded per-extractor record of which dim bindings each plan has been
    /// seen at. On first sighting the generic symbolic variant wins outright
    /// — nothing compiles per length bucket.
    stats: Mutex<ShapeStats>,
}

/// What the search actually did. Exposed so conformance can assert the
/// budget was honoured and best-so-far never regressed.
#[derive(Clone, Debug, Default)]
pub struct SearchTrace {
    pub moves: u32,
    /// Realizations [`co_select`] spent. Bounded separately from `moves`.
    pub co_moves: u32,
    pub chains: u32,
    pub micros: u64,
    /// Best-so-far after the seed and after every accepted move, from either
    /// pass.
    pub best: Vec<Picoseconds>,
}

impl LocalSearch {
    pub fn new(arena: Arc<dyn ArenaPlanner>, caps: Caps) -> Self {
        Self {
            arena,
            caps,
            registry: None,
            stats: Mutex::new(ShapeStats::new()),
        }
    }

    /// Supply the registry `L1::Ext` nodes were built against, so
    /// `verify_plan`'s sixth clause can check `lower_per_target`.
    pub fn with_registry(mut self, registry: OpDefRegistry) -> Self {
        self.registry = Some(registry);
        self
    }

    pub fn caps(&self) -> &Caps {
        &self.caps
    }

    pub fn arena(&self) -> &Arc<dyn ArenaPlanner> {
        &self.arena
    }

    /// The seed selection, its schedule points and its initial materialized
    /// set.
    pub fn seed(
        &self,
        graph: &EGraph,
        roots: &[Id],
        lb: &[Picoseconds],
        cost: &dyn CostModel,
    ) -> Result<Extraction> {
        let classes = realize::classes(graph);
        let mut cache = RealizeScratch::new(graph.len());
        self.seed_realized(graph, roots, lb, cost, &classes, &mut cache)
            .map(|(ex, _)| ex)
    }

    /// The seed plus the DAG it denotes. The probe realization is reused when
    /// `m_0` added nothing, saving an `O(nodes)` pass.
    fn seed_realized(
        &self,
        graph: &EGraph,
        roots: &[Id],
        lb: &[Picoseconds],
        cost: &dyn CostModel,
        classes: &[ClassId],
        cache: &mut RealizeScratch,
    ) -> Result<(Extraction, Realized)> {
        let mut ex = Extraction {
            sigma: FxHashMap::with_capacity_and_hasher(classes.len(), Default::default()),
            m: FixedBitSet::with_capacity(graph.len()),
            theta: FxHashMap::default(),
        };
        for class in classes {
            ex.sigma.insert(*class, argmin_member(graph, lb, *class, &self.caps));
        }
        seed_theta(graph, &mut ex, cost);

        // Consumer counts and index spaces are independent of `m`.
        pin_inplace(graph, &mut ex);
        for r in roots {
            let sel = realize::select(graph, &ex, *r)?;
            materialize(graph, &mut ex, sel);
        }
        let probe = realize::realize_with(graph, roots, &ex, cost, self.arena.as_ref(), cache)?;
        // The obligation sweep plus the seed's `m_0` clause: every value with
        // more than one realized consumer lands in a buffer. The trail is
        // dropped — the seed is never reverted.
        if repair_trailed(graph, &mut ex, &probe, cost, true).is_empty() {
            return Ok((ex, probe));
        }
        let realized = realize::realize_with(graph, roots, &ex, cost, self.arena.as_ref(), cache)?;
        Ok((ex, realized))
    }

    /// The full run, with an explicit [`ShapeStats`] so a caller driving many
    /// steps shares one. `Extractor::extract` uses the extractor's own.
    pub fn extract_with_stats(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
        stats: &mut ShapeStats,
    ) -> Result<(Plan, SearchTrace)> {
        let started = Instant::now();
        let lb = crate::lower_bound::lower_bound(graph, cost);
        let classes = realize::classes(graph);
        let mut cache = RealizeScratch::new(graph.len());
        let (mut ex, seeded) = self.seed_realized(graph, roots, &lb, cost, &classes, &mut cache)?;
        // The seed is priced as the plan it denotes, the same way every
        // candidate below is, so all moves share one objective.
        let (mut realized, mut best_cost) =
            match self.price(graph, roots, &mut ex, cost, &mut cache) {
                Ok((r, c, _)) => (r, c),
                Err(_) => {
                    let c = realize::exact_cost(&seeded, &ex, cost);
                    (seeded, c)
                }
            };

        let chains = classes.len() as u32;
        // The only stopping condition: a fixed number of realized node visits,
        // never a wall clock, so the winning plan and its `PlanHash` do not
        // depend on machine load.
        let cap = budget.move_cap(graph.len(), chains);
        let mut trace = SearchTrace {
            chains,
            best: vec![best_cost],
            ..SearchTrace::default()
        };
        let readers = readers_by_producer(graph, &classes, &self.caps);

        // [`co_select`] from the seed, restricted to joint producers. A sweep
        // over all producers would reach class members unequal to their
        // siblings and put conformance cases on wrong values.
        //
        // Entering a joint is one-way in both directions, so the seeded
        // adoption can leave the climb in a worse basin. When the seeded sweep
        // changes anything, the descent runs twice — once from each state —
        // and the cheaper plan wins; a tie keeps the un-seeded one.
        let joints = joint_producers(graph, &readers);
        let plain = (ex.clone(), realized.clone(), best_cost);
        let mut seeded_best: Vec<Picoseconds> = Vec::new();
        let mut seed_moves = 0u32;
        while seed_moves < cap
            && co_select_over(
                &joints,
                graph,
                roots,
                cost,
                &readers,
                self.arena.as_ref(),
                &mut ex,
                &mut realized,
                &mut best_cost,
                &mut cache,
                &mut seeded_best,
                &mut seed_moves,
                cap,
            )?
        {}

        let descend = |ex: &mut Extraction,
                           realized: &mut Realized,
                           best_cost: &mut Picoseconds,
                           cache: &mut RealizeScratch,
                           best: &mut Vec<Picoseconds>,
                           moves: &mut u32,
                           co_moves: &mut u32|
         -> Result<()> {
            let mut sched = SchedCache::new();
            'search: loop {
                let mut improved = false;
                for mv in moves::frontier(graph, ex, &classes, budget) {
                    if *moves >= cap {
                        break 'search;
                    }
                    let options = moves::candidates(graph, ex, realized, mv, &lb, &mut sched, cost);
                    for candidate in options {
                        if *moves >= cap {
                            break 'search;
                        }
                        *moves += 1;
                        let Some(undo) = moves::apply(graph, ex, candidate) else {
                            continue;
                        };
                        let attempt = price(graph, roots, ex, cost, self.arena.as_ref(), cache);
                        match attempt {
                            // Strict improvements only; a tie keeps the earlier
                            // (smaller-id) state, which keeps the search
                            // reproducible.
                            Ok((r, c, _)) if c < *best_cost => {
                                *best_cost = c;
                                *realized = r;
                                best.push(c);
                                improved = true;
                                break;
                            }
                            // The move is undone after the obligations it
                            // implied, so a rejected candidate leaves no trace.
                            Ok((_, _, trail)) => {
                                unrepair(ex, trail);
                                moves::undo(ex, undo);
                            }
                            Err(trail) => {
                                unrepair(ex, trail);
                                moves::undo(ex, undo);
                            }
                        }
                    }
                }
                if !improved {
                    break;
                }
            }

            // The compound move the single-move climb above cannot make.
            // Sweeps until a sweep improves nothing. It counts against its own
            // budget, because the climb normally spends every move `cap`
            // allows; `cap` is reused as the bound, so one descent is at worst
            // two searches of the sanctioned size.
            while *co_moves < cap
                && co_select(
                    graph,
                    roots,
                    cost,
                    &readers,
                    self.arena.as_ref(),
                    ex,
                    realized,
                    best_cost,
                    cache,
                    best,
                    co_moves,
                    cap,
                )?
            {}

            // The winner is the live state: every rejected move was undone and
            // every accepted one strictly improved, so `realized` and
            // `best_cost` already describe best-so-far, up to the invariants a
            // sequence of independent moves does not preserve.
            if repair(graph, ex, realized, cost) {
                *realized =
                    realize::realize_with(graph, roots, ex, cost, self.arena.as_ref(), cache)?;
                *best_cost = realize::exact_cost(realized, ex, cost);
                best.push(*best_cost);
            }
            Ok(())
        };

        let mut a_best = seeded_best.clone();
        let mut a_moves = 0u32;
        let mut a_co = seed_moves;
        descend(
            &mut ex,
            &mut realized,
            &mut best_cost,
            &mut cache,
            &mut a_best,
            &mut a_moves,
            &mut a_co,
        )?;

        if seeded_best.is_empty() {
            // The seeded sweep changed nothing, so the two starts are the same
            // state and a second descent would repeat the first move for move.
            trace.moves = a_moves;
            trace.co_moves = a_co;
            trace.best.extend(a_best);
        } else {
            let (mut b_ex, mut b_realized, mut b_cost) = plain;
            let mut b_best: Vec<Picoseconds> = Vec::new();
            let mut b_moves = 0u32;
            let mut b_co = 0u32;
            descend(
                &mut b_ex,
                &mut b_realized,
                &mut b_cost,
                &mut cache,
                &mut b_best,
                &mut b_moves,
                &mut b_co,
            )?;
            trace.moves = a_moves.max(b_moves);
            trace.co_moves = a_co.max(b_co);
            if b_cost <= best_cost {
                ex = b_ex;
                realized = b_realized;
                best_cost = b_cost;
                trace.best.extend(b_best);
            } else {
                trace.best.extend(a_best);
            }
        }

        let plan = derive_plan(graph, &ex, &realized, cost.facts(), best_cost)?;
        crate::verify_plan::verify_plan_with(
            graph,
            &plan,
            self.arena.as_ref(),
            &self.caps,
            self.registry.as_ref(),
        )?;

        stats.observe(plan.hash, &binding_of(graph, &realized));
        trace.micros = started.elapsed().as_micros() as u64;
        Ok((plan, trace))
    }

    /// Realize, [`repair`], re-realize: the cost of the plan this state
    /// denotes, which is the only number an accept test may compare.
    ///
    /// A move can put a producer across a structural cut, obliging a buffer,
    /// so `repair` runs per candidate. The returned [`RepairTrail`] is what a
    /// rejecting caller reverts, including on the error arm. When the trail is
    /// empty the second realization is skipped.
    fn price(
        &self,
        graph: &EGraph,
        roots: &[Id],
        ex: &mut Extraction,
        cost: &dyn CostModel,
        cache: &mut RealizeScratch,
    ) -> std::result::Result<(Realized, Picoseconds, RepairTrail), RepairTrail> {
        price(graph, roots, ex, cost, self.arena.as_ref(), cache)
    }

    /// The plan a given extraction denotes: realize, repair, re-realize,
    /// derive, verify. No search, so a candidate is priced and built by the
    /// same path `extract` returns its winner through.
    pub fn replan(
        &self,
        graph: &EGraph,
        roots: &[Id],
        ex: &mut Extraction,
        cost: &dyn CostModel,
    ) -> Result<Plan> {
        let mut cache = RealizeScratch::new(graph.len());
        let (realized, exact, _) = self
            .price(graph, roots, ex, cost, &mut cache)
            .map_err(|_| Error::Plan("autotune candidate does not realize".into()))?;
        let plan = derive_plan(graph, ex, &realized, cost.facts(), exact)?;
        crate::verify_plan::verify_plan_with(
            graph,
            &plan,
            self.arena.as_ref(),
            &self.caps,
            self.registry.as_ref(),
        )?;
        Ok(plan)
    }

    /// The same, reporting what the search did.
    pub fn extract_traced(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
    ) -> Result<(Plan, SearchTrace)> {
        let mut stats = self.stats.lock();
        self.extract_with_stats(graph, roots, cost, budget, &mut stats)
    }
}

impl Extractor for LocalSearch {
    fn lower_bound(&self, graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds> {
        crate::lower_bound::lower_bound(graph, cost)
    }

    fn extract(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
    ) -> Result<Plan> {
        self.extract_traced(graph, roots, cost, budget)
            .map(|(p, _)| p)
    }

    fn verify_plan(&self, graph: &EGraph, plan: &Plan) -> Result<()> {
        crate::verify_plan::verify_plan_with(
            graph,
            plan,
            self.arena.as_ref(),
            &self.caps,
            self.registry.as_ref(),
        )
    }

    fn launch_variants(
        &self,
        graph: &EGraph,
        roots: &[Id],
        base: &Plan,
        launch_ix: usize,
        cost: &dyn CostModel,
        min_macs: u64,
    ) -> Vec<(String, Plan)> {
        let Some(launch) = base.launches.get(launch_ix) else {
            return Vec::new();
        };
        let root = launch.root;
        if launch_work(graph, base, launch_ix) < min_macs {
            return Vec::new();
        }
        // Timing a plan re-runs it. An in-place node makes that destructive,
        // so an impure plan is never tuned.
        if base.launches.iter().any(|l| {
            l.members
                .iter()
                .any(|m| graph.semantics().effect(&graph.node(*m).op) != Effect::Pure)
        }) {
            return Vec::new();
        }

        let class = graph.class_of(root);
        let here = base.extraction.theta.get(&root).copied();
        // Round-robin across members: one point per member per round, so every
        // member's first-choice geometry races before any member's second.
        // Member-major enumeration would let a wide class's first members
        // spend the whole of `TUNE_MAX_VARIANTS`.
        let per_member: Vec<(Id, SmallVec<[SchedPoint; 8]>)> = graph
            .members(class)
            .into_iter()
            .filter_map(|member| {
                let Op::L1(l1) = &graph.node(member).op else {
                    return None;
                };
                let domain = l1.schedule()?;
                Some((member, sample_points(domain)))
            })
            .collect();
        let rounds = per_member.iter().map(|(_, p)| p.len()).max().unwrap_or(0);
        let mut fair: Vec<(Id, SchedPoint)> = Vec::new();
        for r in 0..rounds {
            for (member, points) in &per_member {
                if let Some(theta) = points.get(r) {
                    fair.push((*member, *theta));
                }
            }
        }
        let mut out: Vec<(String, Plan)> = Vec::new();
        {
            for (member, theta) in fair {
                if out.len() >= TUNE_MAX_VARIANTS {
                    return out;
                }
                if member == root && Some(theta) == here {
                    continue;
                }
                let mut ex = base.extraction.clone();
                if member != root
                    && moves::apply(
                        graph,
                        &mut ex,
                        moves::Candidate::Select {
                            class,
                            node: member,
                        },
                    )
                    .is_none()
                {
                    continue;
                }
                moves::apply(
                    graph,
                    &mut ex,
                    moves::Candidate::Schedule { node: member, theta },
                );
                let Ok(plan) = self.replan(graph, roots, &mut ex, cost) else {
                    continue;
                };
                // A candidate may change the dispatch count: selecting a member
                // whose operand must materialize adds that producer's launch.
                // Such candidates are compared on the whole-plan clock, since
                // the race refuses to file per-launch spans for them.
                out.push((variant_signature(graph, member, theta), plan));
            }
        }
        out
    }
}

/// For each producer class, every `(reading class, member)` pair that reads
/// it, both ascending. A function of the graph alone, so it is built once per
/// extraction. `O(nodes * fanin)`.
fn readers_by_producer(
    graph: &EGraph,
    classes: &[ClassId],
    caps: &Caps,
) -> FxHashMap<ClassId, Vec<(ClassId, Id)>> {
    let mut out: FxHashMap<ClassId, Vec<(ClassId, Id)>> = FxHashMap::default();
    for c in classes {
        for m in realize::selectable(graph, *c, caps) {
            let mut producers: SmallVec<[ClassId; 4]> = SmallVec::new();
            for ch in graph.node(m).children.iter() {
                let p = graph.class_of(*ch);
                // A member reading one producer twice proposes one move.
                if p != *c && !producers.contains(&p) {
                    producers.push(p);
                }
            }
            for p in producers {
                out.entry(p).or_default().push((*c, m));
            }
        }
    }
    // Ascending by class then member, so the sweep below does not depend on
    // hash order.
    for v in out.values_mut() {
        v.sort_unstable();
    }
    out
}

/// One co-selection sweep. For each producer class, adopt together every class
/// that holds a selectable member reading it; keep on a strict improvement in
/// exact global cost, revert through [`moves::undo`] otherwise. A rule fusing
/// `F` values into one node yields `F` slot views in distinct e-classes, and
/// adopting one alone is strictly worse than adopting none, so the single-move
/// climb cannot reach the joint. Costs one realization per producer class with
/// two or more reading classes, sharing `cap` with the frontier search.
#[allow(clippy::too_many_arguments)]
fn co_select(
    graph: &EGraph,
    roots: &[Id],
    cost: &dyn CostModel,
    readers: &FxHashMap<ClassId, Vec<(ClassId, Id)>>,
    arena: &dyn ArenaPlanner,
    ex: &mut Extraction,
    realized: &mut Realized,
    best_cost: &mut Picoseconds,
    cache: &mut RealizeScratch,
    best: &mut Vec<Picoseconds>,
    moves: &mut u32,
    cap: u32,
) -> Result<bool> {
    let mut producers: Vec<ClassId> = readers.keys().copied().collect();
    producers.sort_unstable();
    co_select_over(
        &producers, graph, roots, cost, readers, arena, ex, realized, best_cost, cache, best,
        moves, cap,
    )
}

/// The producer classes holding a multi-slot carrier — a node fusing several
/// values into one, whose readers are slot views. Ascending.
fn joint_producers(
    graph: &EGraph,
    readers: &FxHashMap<ClassId, Vec<(ClassId, Id)>>,
) -> Vec<ClassId> {
    let mut out: Vec<ClassId> = readers
        .keys()
        .copied()
        .filter(|p| {
            graph.members(*p).iter().any(|m| {
                matches!(&graph.node(*m).op, Op::L1(L1::KFold { carrier, .. })
                    if carrier.width() > 1)
            })
        })
        .collect();
    out.sort_unstable();
    out
}

/// [`co_select`] over a stated set of producer classes, in the order given.
#[allow(clippy::too_many_arguments)]
fn co_select_over(
    producers: &[ClassId],
    graph: &EGraph,
    roots: &[Id],
    cost: &dyn CostModel,
    readers: &FxHashMap<ClassId, Vec<(ClassId, Id)>>,
    arena: &dyn ArenaPlanner,
    ex: &mut Extraction,
    realized: &mut Realized,
    best_cost: &mut Picoseconds,
    cache: &mut RealizeScratch,
    best: &mut Vec<Picoseconds>,
    moves: &mut u32,
    cap: u32,
) -> Result<bool> {
    let mut improved = false;
    for p in producers.iter().copied() {
        if *moves >= cap {
            break;
        }
        // The smallest-id member of each reading class that is not already
        // the selected one. `readers[p]` is sorted, so the first entry per
        // class is that member.
        let mut proposal: Vec<(ClassId, Id)> = Vec::new();
        for (c, m) in &readers[&p] {
            if proposal.last().is_some_and(|(last, _)| last == c) {
                continue;
            }
            if ex.sigma.get(c).copied() != Some(*m) {
                proposal.push((*c, *m));
            }
        }
        if proposal.len() < 2 {
            continue;
        }
        let mut undos = Vec::with_capacity(proposal.len());
        for (c, m) in &proposal {
            if let Some(u) = moves::apply(
                graph,
                ex,
                crate::moves::Candidate::Select {
                    class: *c,
                    node: *m,
                },
            ) {
                undos.push(u);
            }
        }
        if undos.is_empty() {
            continue;
        }
        // Adopting `F` slot views makes the joint an `F`-consumer node, and a
        // node outside `M` is inlined into every consumer, so an
        // unmaterialized joint is recomputed once per slot. The seed's
        // `consumers > 1` clause runs before these consumers exist, so the
        // adoption is also offered with the joint materialized. Both states are
        // priced and exact global cost decides.
        let producer = ex.sigma.get(&p).copied().filter(|n| {
            !ex.is_materialized(*n) && realize::leaf_role(graph, *n) == realize::LeafRole::NotLeaf
        });
        let mut kept = false;
        for flip in [false, true] {
            let m_undo = if flip {
                let Some(pn) = producer else { continue };
                match moves::apply(
                    graph,
                    ex,
                    crate::moves::Candidate::Materialize { node: pn, on: true },
                ) {
                    Some(u) => Some(u),
                    None => continue,
                }
            } else {
                None
            };
            *moves += 1;
            match price(graph, roots, ex, cost, arena, cache) {
                // Strict improvements only: a tie keeps the state the search
                // was already in.
                Ok((r, c, _)) if c < *best_cost => {
                    *best_cost = c;
                    *realized = r;
                    best.push(c);
                    improved = true;
                    kept = true;
                    break;
                }
                Ok((_, _, trail)) | Err(trail) => {
                    unrepair(ex, trail);
                    if let Some(u) = m_undo {
                        moves::undo(ex, u);
                    }
                }
            }
            if *moves >= cap {
                break;
            }
        }
        if !kept {
            for u in undos.into_iter().rev() {
                moves::undo(ex, u);
            }
        }
    }
    Ok(improved)
}

fn materialize(graph: &EGraph, ex: &mut Extraction, id: Id) {
    if realize::leaf_role(graph, id) != realize::LeafRole::NotLeaf {
        return;
    }
    if ex.m.len() <= id.index() {
        ex.m.grow(id.index() + 1);
    }
    ex.m.insert(id.index());
}

fn pin_inplace(graph: &EGraph, ex: &mut Extraction) {
    let mut selected: Vec<Id> = ex.sigma.values().copied().collect();
    selected.sort_unstable();
    for id in selected {
        if graph.semantics().effect(&graph.node(id).op) != Effect::Pure {
            materialize(graph, ex, id);
        }
    }
}

/// The starting point of every selected schedule domain: the cheapest by
/// `node_math`, ties by domain index. The full domain stays reachable through
/// `RESCHEDULE`.
///
/// Fill-only, so this is safe to re-run after the search: a point
/// `RESCHEDULE` already chose is kept, and only a `theta` outside its node's
/// domain is replaced.
fn seed_theta(graph: &EGraph, ex: &mut Extraction, cost: &dyn CostModel) {
    let mut trail = RepairTrail::default();
    seed_theta_trailed(graph, ex, cost, &mut trail);
}

/// [`seed_theta`], recording every entry it wrote and the value it replaced.
fn seed_theta_trailed(
    graph: &EGraph,
    ex: &mut Extraction,
    cost: &dyn CostModel,
    trail: &mut RepairTrail,
) {
    let caps = &cost.facts().caps;
    let mut selected: Vec<Id> = ex.sigma.values().copied().collect();
    selected.sort_unstable();
    selected.dedup();
    for id in selected {
        let node = graph.node(id);
        let Op::L1(l1) = &node.op else { continue };
        let Some(domain) = l1.schedule() else {
            continue;
        };
        if matches!(domain, ScheduleDomain::Point) {
            let prev = ex
                .theta
                .insert(id, fusor2_ir::ir::level1::SchedPoint::Point);
            if prev != Some(fusor2_ir::ir::level1::SchedPoint::Point) {
                trail.theta.push((id, prev));
            }
            continue;
        }
        if let Some(current) = ex.theta.get(&id).copied()
            && domain.iter().any(|p| p == current)
            && realize::point_is_legal(graph, id, current, caps)
        {
            continue;
        }
        let scan = realize::DomainScan::new(graph, id, cost);
        // Only points this device can run: `has_legal_point` gates the node
        // and passes as soon as one point fits, so without this clause the
        // cheapest point can win the seed with a footprint over the cap.
        // If nothing is legal, fall back the way `legal_members` does and let
        // `verify_plan` report it rather than leaving `theta` unset.
        let mut best: Option<(Picoseconds, usize, _)> = None;
        let any_legal = domain
            .iter()
            .any(|t| realize::point_is_legal(graph, id, t, caps));
        for (i, theta) in domain.iter().enumerate() {
            if any_legal && !realize::point_is_legal(graph, id, theta, caps) {
                continue;
            }
            let s = scan.price(cost, Some(theta));
            if best.as_ref().is_none_or(|(b, _, _)| s < *b) {
                best = Some((s, i, theta));
            }
        }
        if let Some((_, _, theta)) = best {
            let prev = ex.theta.insert(id, theta);
            if prev != Some(theta) {
                trail.theta.push((id, prev));
            }
        }
    }
}

/// Re-establish, on the search winner, every invariant a plan needs that a
/// sequence of independent moves does not preserve — `verify_plan`'s clauses
/// 3, 5 and 6: a root and an in-place node land in a buffer, a producer cut
/// from a consumer by structure lands in a buffer, and every selected schedule
/// domain has a point in it.
///
/// One pass is a fixpoint, since `order`, `consumers`, `consumer_nodes` and
/// every index space are functions of `sigma` alone. Returns whether anything
/// changed, in which case the caller re-realizes and re-prices.
fn repair(
    graph: &EGraph,
    ex: &mut Extraction,
    realized: &Realized,
    cost: &dyn CostModel,
) -> bool {
    !repair_trailed(graph, ex, realized, cost, false).is_empty()
}

/// Realize, [`repair_trailed`], re-realize. See [`LocalSearch::price`].
fn price(
    graph: &EGraph,
    roots: &[Id],
    ex: &mut Extraction,
    cost: &dyn CostModel,
    arena: &dyn ArenaPlanner,
    cache: &mut RealizeScratch,
) -> std::result::Result<(Realized, Picoseconds, RepairTrail), RepairTrail> {
    let first = realize::realize_with(graph, roots, ex, cost, arena, cache)
        .map_err(|_| RepairTrail::default())?;
    let trail = repair_trailed(graph, ex, &first, cost, false);
    if trail.is_empty() {
        let c = realize::exact_cost(&first, ex, cost);
        return Ok((first, c, trail));
    }
    match realize::realize_with(graph, roots, ex, cost, arena, cache) {
        Ok(second) => {
            let c = realize::exact_cost(&second, ex, cost);
            Ok((second, c, trail))
        }
        Err(_) => Err(trail),
    }
}

/// Everything [`repair_trailed`] added to a state, in the order it added it.
/// A rejected candidate reverts the obligations its move implied as well as
/// the move, so a declined state leaves no buffers behind.
#[derive(Clone, Debug, Default)]
struct RepairTrail {
    /// Nodes newly inserted into `m`. Only nodes that were absent before, so
    /// reverting is an unconditional clear.
    m: SmallVec<[Id; 8]>,
    /// `theta` entries written, each with the value it replaced.
    theta: SmallVec<[(Id, Option<fusor2_ir::ir::level1::SchedPoint>); 8]>,
}

impl RepairTrail {
    fn is_empty(&self) -> bool {
        self.m.is_empty() && self.theta.is_empty()
    }
}

/// Undo a [`RepairTrail`], newest entry first.
fn unrepair(ex: &mut Extraction, trail: RepairTrail) {
    for (id, prev) in trail.theta.into_iter().rev() {
        match prev {
            Some(p) => {
                ex.theta.insert(id, p);
            }
            None => {
                ex.theta.remove(&id);
            }
        }
    }
    for id in trail.m.into_iter().rev() {
        if ex.m.len() > id.index() {
            ex.m.remove(id.index());
        }
    }
}

/// [`repair`], recording what it changed so the caller can revert it.
///
/// With `shared` set it also materializes every node with more than one
/// realized consumer: a producer across any launch split has to land in a
/// buffer or its consumer reads what nothing wrote. That is the seed's `m_0`
/// clause; the per-candidate repair must not apply it, since `FLIP` exists to
/// price those states.
fn repair_trailed(
    graph: &EGraph,
    ex: &mut Extraction,
    realized: &Realized,
    cost: &dyn CostModel,
    shared: bool,
) -> RepairTrail {
    let mut trail = RepairTrail::default();
    for r in &realized.roots {
        if !ex.is_materialized(*r) {
            materialize_trailed(graph, ex, *r, &mut trail);
        }
    }
    for v in &realized.order {
        if realize::leaf_role(graph, *v) != realize::LeafRole::NotLeaf || ex.is_materialized(*v) {
            continue;
        }
        let in_place = graph.semantics().effect(&graph.node(*v).op) != Effect::Pure;
        if in_place
            || (shared && realized.consumers.copied(*v).unwrap_or(0) > 1)
            || moves::at_structural_boundary(graph, realized, *v)
        {
            materialize_trailed(graph, ex, *v, &mut trail);
        }
    }
    seed_theta_trailed(graph, ex, cost, &mut trail);
    trail
}

fn materialize_trailed(graph: &EGraph, ex: &mut Extraction, id: Id, trail: &mut RepairTrail) {
    let before = ex.is_materialized(id);
    materialize(graph, ex, id);
    if !before && ex.is_materialized(id) {
        trail.m.push(id);
    }
}

/// The dim binding this run was extracted at: every root's extents, in root
/// order. `Dim::Sym` stays symbolic, so a symbolic plan records one family
/// rather than one bucket per length.
fn binding_of(graph: &EGraph, realized: &Realized) -> Vec<Dim> {
    let mut out = Vec::new();
    for r in &realized.roots {
        out.extend(graph.facts(*r).shape.iter().copied());
    }
    out
}

/// Variants offered per launch. Timing ~10 of them costs ~450 ms of cold time
/// on a 2048-cube matmul.
const TUNE_MAX_VARIANTS: usize = 16;

/// The tile shapes the pin sweep separates on. Coop geometries are generated
/// `bm`-major, so a domain prefix would be six spellings of the same narrowest
/// tile.
const TUNE_GEOMS: [(u32, u32, u32); 6] = [
    (16, 16, 8),
    (32, 32, 8),
    (64, 64, 8),
    (64, 64, 16),
    (128, 64, 8),
    (128, 128, 8),
];

/// The points of one domain worth timing.
///
/// `splits` and `staging` are pinned to the domain's first entry, always `1`
/// and `1`: splits change the dispatch count, and the `staging == 2` footprint
/// filter is loose by nearly 2x.
fn sample_points(domain: &ScheduleDomain) -> SmallVec<[SchedPoint; 8]> {
    let mut out: SmallVec<[SchedPoint; 8]> = SmallVec::new();
    match domain {
        ScheduleDomain::Point => {}
        ScheduleDomain::Coop(d) => {
            let (Some(splits), Some(staging)) = (d.splits.first(), d.staging.first()) else {
                return out;
            };
            for (bm, bn, bk) in TUNE_GEOMS {
                if let Some(geom) = d
                    .geoms
                    .iter()
                    .find(|g| g.bm == bm && g.bn == bn && g.bk == bk)
                {
                    out.push(SchedPoint::Coop {
                        geom: *geom,
                        splits: *splits,
                        staging: *staging,
                    });
                }
            }
        }
        other => {
            // Two points off a non-coop family, enough to tell the family
            // apart from Coop.
            let n = other.len();
            for i in [0usize, n / 2] {
                if let Some(p) = other.point(i)
                    && !out.contains(&p)
                {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// Work a whole launch issues, across every node family, summed over the
/// launch's members rather than its root alone.
///
/// `transcendentals` are weighted by the ratio the roofline's `trans_ps`
/// implies against a MAC, rounded to an integer.
fn launch_work(graph: &EGraph, base: &Plan, launch_ix: usize) -> u64 {
    const TRANS_WEIGHT: u64 = 8;
    let Some(launch) = base.launches.get(launch_ix) else {
        return 0;
    };
    let mut total: u64 = 0;
    for m in &launch.members {
        let node = graph.node(*m);
        let ins: SmallVec<[ValueFacts; 4]> = node
            .children
            .iter()
            .map(|c| graph.facts(*c).clone())
            .collect();
        let w = graph.semantics().work(&node.op, &ins, graph.facts(*m));
        total = total
            .saturating_add(w.macs)
            .saturating_add(w.transcendentals.saturating_mul(TRANS_WEIGHT))
            .saturating_add(w.index_ops);
    }
    total
}

/// A launch's identity across processes, for the persistent tune cache.
///
/// Node `Id`s are graph-allocation order and mean nothing in the next process,
/// so the key is the op family plus the extents and dtypes that decide which
/// schedule wins.
///
/// It keys the whole launch, not its root: `facts.shape` is the output shape,
/// so a fold's reduced extent and the fused body are both invisible to the
/// root. `TuneCache::record` merges by minimum, so a key that cannot tell two
/// kernels apart files the cheaper span under the other one's name.
pub fn launch_signature(graph: &EGraph, launch: &Launch) -> String {
    let root = launch.root;
    let facts = graph.facts(root);
    let extents = |dims: &[Dim]| -> String {
        dims.iter()
            .map(|d| d.as_const().map_or_else(|| "s".to_string(), |v| v.to_string()))
            .collect::<Vec<_>>()
            .join("x")
    };
    let tag = match &graph.node(root).op {
        Op::L1(l1) => format!("{:?}", l1.tag()),
        Op::L0(_) => "L0".to_string(),
        Op::Union(..) => "U".to_string(),
    };
    let extra = match &graph.node(root).op {
        Op::L1(L1::KContract { m, n, k, batch, .. }) => {
            let c = |d: &Dim| d.as_const().map_or(0, |v| v);
            format!("mnkb={},{},{},{}", c(m), c(n), c(k), c(batch))
        }
        // `space` is the iteration domain and carries the reduced extent; the
        // output shape above does not.
        Op::L1(L1::KFold {
            space,
            axis,
            vec_axes,
            carrier,
            ..
        }) => format!(
            "space=[{}] axis={axis} vec={vec_axes:?} slots={}",
            extents(&space.dims),
            carrier.slots.len()
        ),
        _ => String::new(),
    };
    // What the launch computes, one digest per member, sorted because member
    // order is a realization detail. Fusion happens inside a node, so a
    // histogram of member tags would not separate an elementwise chain from a
    // bare reduction; the digest covers scalar bodies and operand accesses.
    let mut body: Vec<String> = launch
        .members
        .iter()
        .map(|m| {
            let op = &graph.node(*m).op;
            format!("{:?}:{:08x}", op.tag(), body_digest(op) as u32)
        })
        .collect();
    body.sort_unstable();
    format!(
        "{tag}|{:?}|[{}]|{extra}|body={}",
        facts.dtype,
        extents(&facts.shape),
        body.join(",")
    )
}

/// A stable digest of what one member computes, for [`launch_signature`].
///
/// Excludes `Operand::src`: it is a graph-allocation `Id` and means nothing in
/// the next process, so hashing it would make every key a cache miss.
/// Everything else that decides the emitted kernel is hashed; scalar bodies
/// contribute [`ScalarExpr::structural_hash`], which is cached and bottom-up,
/// so this is O(operands).
///
/// A symbolic extent hashes its `SymId`, which is allocation order within a
/// session; a program allocating them in a different order takes a cache miss
/// and one tuning pass, never a wrong answer.
fn body_digest(op: &Op) -> u64 {
    use fusor2_ir::ir::level1::Operand;
    use rustc_hash::FxHasher;
    use std::hash::{Hash, Hasher};

    fn operand(o: &Operand, h: &mut FxHasher) {
        o.layout.hash(h);
        o.access.hash(h);
    }

    let mut h = FxHasher::default();
    op.tag().hash(&mut h);
    match op {
        Op::L1(L1::KMap {
            space, body, ops, ..
        }) => {
            space.dims.hash(&mut h);
            body.structural_hash().hash(&mut h);
            for o in ops {
                operand(o, &mut h);
            }
        }
        Op::L1(L1::KFold {
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            ops,
            ..
        }) => {
            space.dims.hash(&mut h);
            axis.hash(&mut h);
            vec_axes.hash(&mut h);
            carrier.hash(&mut h);
            acc.hash(&mut h);
            for p in post {
                p.structural_hash().hash(&mut h);
            }
            for o in ops {
                operand(o, &mut h);
            }
        }
        Op::L1(L1::KContract {
            m,
            n,
            k,
            batch,
            family,
            post,
            acc,
            a,
            b,
            ..
        }) => {
            (m, n, k, batch, family, acc).hash(&mut h);
            for e in [&a.pre, &b.pre, post] {
                e.structural_hash().hash(&mut h);
            }
            // Arity is part of the key: two sides holding the same leading
            // operand differ if one has absorbed a producer.
            (a.len(), b.len()).hash(&mut h);
            for o in a.ops.iter().chain(b.ops.iter()) {
                operand(o, &mut h);
            }
        }
        Op::L1(L1::KGather {
            space,
            axis,
            mode,
            ops,
            ..
        }) => {
            space.dims.hash(&mut h);
            (axis, mode).hash(&mut h);
            for o in ops {
                operand(o, &mut h);
            }
        }
        Op::L1(L1::KScatter {
            space,
            axis,
            mode,
            combine,
            ops,
            ..
        }) => {
            space.dims.hash(&mut h);
            (axis, mode, combine).hash(&mut h);
            for o in ops {
                operand(o, &mut h);
            }
        }
        Op::L1(L1::KRegion { live_outs, .. }) => live_outs.hash(&mut h),
        Op::L1(L1::KMerged(_) | L1::Ext { .. }) | Op::L0(_) | Op::Union(..) => {}
    }
    h.finish()
}

/// One candidate's identity across processes: the member's family and the
/// schedule point. Paired with [`launch_signature`] this is the cache key.
fn variant_signature(graph: &EGraph, member: Id, theta: SchedPoint) -> String {
    let tag = match &graph.node(member).op {
        Op::L1(l1) => format!("{:?}", l1.tag()),
        _ => "?".to_string(),
    };
    format!("{tag}|{theta:?}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::realize::testkit::{
        N, TestCost, TestPlanner, buffer, chain_graph, fork_graph, kmap, new_graph, test_caps,
    };
    use fusor2_ir::egraph::ClassId;

    fn search() -> LocalSearch {
        LocalSearch::new(Arc::new(TestPlanner), test_caps())
    }

    #[test]
    fn lower_bound_is_admissible() {
        // A seeded LCG keeps the shapes reproducible without pulling in `rand`.
        let cost = TestCost::default();
        for seed in 0..30u64 {
            let mut rng = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let mut next = |n: u64| {
                rng = rng
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (rng >> 33) % n
            };
            let mut g = new_graph();
            let shape = [N];
            let mut pool = vec![buffer(&mut g, 0, &shape)];
            let nodes = 3 + next(8) as usize;
            for _ in 0..nodes {
                let src = pool[next(pool.len() as u64) as usize];
                let depth = 1 + next(4) as u32;
                pool.push(kmap(&mut g, src, &shape, depth));
            }
            // A couple of alternatives per graph, so classes are real.
            if pool.len() > 3 {
                let a = pool[1];
                let b = pool[2];
                g.union(a, b).unwrap();
            }
            g.add_root(*pool.last().unwrap());
            let roots = g.roots().to_vec();

            let lb = crate::lower_bound::lower_bound(&g, &cost);
            let plan = search()
                .extract(&g, &roots, &cost, ExtractBudget::default())
                .unwrap();
            let root_class = g.class_of(roots[0]);
            let bound = lb[root_class.0.index()];
            assert!(
                bound <= plan.cost,
                "seed {seed}: lb {bound:?} exceeded the plan cost {:?}",
                plan.cost
            );
        }
    }

    #[test]
    fn extraction_is_deterministic() {
        let (g, roots, _shared) = fork_graph();
        let cost = TestCost::default();
        let first = search()
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        for _ in 0..20 {
            let again = search()
                .extract(&g, &roots, &cost, ExtractBudget::default())
                .unwrap();
            assert_eq!(again.hash, first.hash);
            assert_eq!(again.cost, first.cost);
        }
    }

    #[test]
    fn launch_cut_at_index_space_mismatch() {
        // KMap[1024] feeding KFold[1024 -> 64]: the fold's space does not
        // cover the map's, so the edge is a forced boundary.
        let mut g = new_graph();
        let wide = [Dim::Const(1024)];
        let leaf = buffer(&mut g, 0, &wide);
        let m = kmap(&mut g, leaf, &wide, 1);
        let f = crate::realize::testkit::kfold(&mut g, m, &[Dim::Const(64), Dim::Const(16)], 1);
        g.add_root(f);
        let roots = g.roots().to_vec();
        let cost = TestCost::default();
        let plan = search()
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        assert_eq!(plan.launches.len(), 2, "{:?}", plan.launches);
    }

    #[test]
    fn budget_is_respected() {
        let mut g = new_graph();
        let shape = [Dim::Const(256)];
        let mut cur = buffer(&mut g, 0, &shape);
        // ~3,000 nodes, the trainer's step-graph scale.
        for _ in 0..3_000 {
            cur = kmap(&mut g, cur, &shape, 1);
        }
        g.add_root(cur);
        let roots = g.roots().to_vec();
        let cost = TestCost::default();
        let budget = ExtractBudget::default();
        let (_, trace) = search().extract_traced(&g, &roots, &cost, budget).unwrap();
        assert!(trace.moves <= 64 * trace.chains.max(1), "{trace:?}");
        assert!(
            trace.best.windows(2).all(|w| w[1] <= w[0]),
            "best-so-far regressed: {:?}",
            trace.best
        );
        // The budget is realized node visits, not a wall clock. The 3,000-deep
        // chain is the pathological shape: one launch with 3,000 members and
        // 3,000 singleton classes.
        assert!(
            trace.moves <= budget.move_cap(g.len(), trace.chains),
            "{trace:?} exceeded the work cap"
        );
        assert!(
            u64::from(trace.moves) * g.len() as u64 <= budget.max_move_work,
            "{trace:?} spent more than {} node visits",
            budget.max_move_work
        );
        // `co_select` is bounded separately from the frontier climb, so the
        // same two bounds hold again against its own counter.
        assert!(
            trace.co_moves <= budget.move_cap(g.len(), trace.chains),
            "{trace:?} exceeded the work cap in co-selection"
        );
        assert!(
            u64::from(trace.co_moves) * g.len() as u64 <= budget.max_move_work,
            "{trace:?} spent more than {} node visits in co-selection",
            budget.max_move_work
        );
    }

    /// Every reading class of a producer appears in the index, once per
    /// member, ascending.
    #[test]
    fn readers_by_producer_indexes_every_reading_class() {
        let (g, _roots, shared) = fork_graph();
        let classes = realize::classes(&g);
        let readers = readers_by_producer(&g, &classes, &test_caps());
        let entry = readers
            .get(&g.class_of(shared))
            .expect("the shared map has readers");
        let reading: Vec<ClassId> = entry.iter().map(|(c, _)| *c).collect();
        assert_eq!(reading.len(), 2, "{entry:?}");
        assert!(reading[0] < reading[1], "not ascending: {entry:?}");
        // And a producer never lists its own class, so a sweep can never
        // propose replacing the producer it is co-selecting around.
        for (p, es) in &readers {
            assert!(es.iter().all(|(c, _)| c != p), "{p:?} lists itself");
        }
    }

    /// A sweep that improves nothing leaves the extraction byte-identical:
    /// every speculative `Select` is reverted through `moves::undo`.
    #[test]
    fn co_select_reverts_every_move_it_does_not_keep() {
        let (g, roots, _shared) = fork_graph();
        let cost = TestCost::default();
        let s = search();
        let lb = crate::lower_bound::lower_bound(&g, &cost);
        let classes = realize::classes(&g);
        let mut cache = RealizeScratch::new(g.len());
        let (mut ex, mut realized) = s
            .seed_realized(&g, &roots, &lb, &cost, &classes, &mut cache)
            .unwrap();
        let before = ex.sigma.clone();
        let before_m = ex.m.clone();
        let mut best = realize::exact_cost(&realized, &ex, &cost);
        let readers = readers_by_producer(&g, &classes, &test_caps());
        let mut trail = Vec::new();
        let mut moves_spent = 0u32;
        let improved = co_select(
            &g,
            &roots,
            &cost,
            &readers,
            s.arena().as_ref(),
            &mut ex,
            &mut realized,
            &mut best,
            &mut cache,
            &mut trail,
            &mut moves_spent,
            u32::MAX,
        )
        .unwrap();
        if !improved {
            assert_eq!(ex.sigma, before, "a declined sweep mutated sigma");
            assert_eq!(ex.m, before_m, "a declined sweep mutated M");
            assert!(trail.is_empty());
        }
        // Whichever way it went, every recorded cost is an improvement on the
        // one before it.
        assert!(trail.windows(2).all(|w| w[1] < w[0]), "{trail:?}");
    }

    /// Co-selection is deterministic: it sweeps in ascending class order over
    /// an index sorted the same way.
    #[test]
    fn co_selected_extraction_is_deterministic() {
        let (g, roots, _shared) = fork_graph();
        let cost = TestCost::default();
        let (first, ft) = search()
            .extract_traced(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        for _ in 0..10 {
            let (again, at) = search()
                .extract_traced(&g, &roots, &cost, ExtractBudget::default())
                .unwrap();
            assert_eq!(again.hash, first.hash);
            assert_eq!(again.cost, first.cost);
            assert_eq!(at.co_moves, ft.co_moves);
        }
    }

    #[test]
    fn remat_prices_exactly() {
        // A 4 MiB f32 map with two consumers. Toggling it out of `M` deletes
        // its whole launch and adds its work once per extra consumer:
        // `saved_write + saved_reads - recompute * (consumers - 1)`.
        use crate::moves::{Candidate, apply};
        use crate::realize::testkit::kmap_neg;
        let mut g = new_graph();
        let shape = [Dim::Const(1024 * 1024)]; // 4 MiB of f32
        let leaf = buffer(&mut g, 0, &shape);
        let p = kmap_neg(&mut g, leaf, &shape, 1);
        // Distinct bodies, or hash-consing would make one node of the two.
        let a = kmap_neg(&mut g, p, &shape, 2);
        let b = kmap_neg(&mut g, p, &shape, 3);
        g.add_root(a);
        g.add_root(b);
        let roots = g.roots().to_vec();

        let cost = TestCost::default();
        let arena = TestPlanner;
        let s = search();
        let lb = crate::lower_bound::lower_bound(&g, &cost);
        let mut ex = s.seed(&g, &roots, &lb, &cost).unwrap();
        assert!(ex.is_materialized(p), "two consumers seed it into M");

        let reads_of = |r: &realize::Realized| -> u64 {
            r.components
                .iter()
                .flat_map(|c| c.reads.iter())
                .map(|(b, n)| b * *n as u64)
                .sum()
        };
        let writes_of =
            |r: &realize::Realized| -> u64 { r.components.iter().map(|c| c.writes).sum() };
        let macs_of =
            |r: &realize::Realized| -> u64 { r.components.iter().map(|c| c.work.macs).sum() };

        let held = realize::realize(&g, &roots, &ex, &cost, &arena).unwrap();
        assert_eq!(held.consumers.copied(p), Some(2));
        let held_cost = realize::exact_cost(&held, &ex, &cost);

        apply(&g, &mut ex, Candidate::Materialize { node: p, on: false }).unwrap();
        let inlined = realize::realize(&g, &roots, &ex, &cost, &arena).unwrap();
        let inlined_cost = realize::exact_cost(&inlined, &ex, &cost);

        // saved_write: exactly the value's own bytes, once.
        let value_bytes = realize::bytes_of(g.facts(p));
        assert_eq!(value_bytes, 4 << 20);
        assert_eq!(writes_of(&held) - writes_of(&inlined), value_bytes);

        // saved_reads: one read per consumer that no longer loads it.
        let consumers = 2u64;
        assert_eq!(
            reads_of(&held) - reads_of(&inlined),
            value_bytes * consumers
        );

        // recompute * (consumers - 1): exactly one extra evaluation.
        let node_macs = 1024 * 1024;
        assert_eq!(
            macs_of(&inlined) - macs_of(&held),
            node_macs * (consumers - 1)
        );

        // In picoseconds: the launches and traffic that vanish, minus the
        // recompute, which the roofline's `max` hides under the consumers'
        // own bandwidth and so costs nothing here.
        let launches_saved = (held.components.len() - inlined.components.len()) as u64;
        let saved_write = value_bytes;
        let saved_reads = value_bytes * consumers;
        assert_eq!(
            held_cost.0 - inlined_cost.0,
            launches_saved * cost.facts().launch_ps + cost.traffic(saved_write + saved_reads, 1).0
        );
        assert!(inlined_cost < held_cost, "inlining a 4 MiB map wins");

        // The search prices inlining as a win but may not ship it: a launch is
        // lowered from one node, so an inlined producer no rule folded into
        // its consumer would leave that kernel reading an operand nothing
        // wrote. `realize::needs_own_buffer` states the obligation and
        // `repair` enforces it on the winner, so the plan still buffers `p`.
        let plan = s
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        assert!(
            plan.extraction.is_materialized(p),
            "until a rule folds `p` into both consumers, the plan has to give it a buffer"
        );
    }

    #[test]
    fn inplace_node_pinned() {
        use crate::realize::testkit::kscatter;
        let mut g = new_graph();
        let shape = [N];
        let base = buffer(&mut g, 0, &shape);
        let idx = buffer(&mut g, 1, &shape);
        let upd = buffer(&mut g, 2, &shape);
        let sc = kscatter(&mut g, base, idx, upd, &shape);
        let a = kmap(&mut g, sc, &shape, 1);
        let b = kmap(&mut g, sc, &shape, 2);
        g.add_root(a);
        g.add_root(b);
        let roots = g.roots().to_vec();
        let cost = TestCost::default();
        let plan = search()
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        assert!(
            plan.extraction.is_materialized(sc),
            "an atomic scatter inlined into two consumers applies twice"
        );
    }

    #[test]
    fn the_extractor_stays_object_safe() {
        // The ILP oracle ships behind the same trait.
        let boxed: Box<dyn Extractor> = Box::new(search());
        let (g, roots) = chain_graph(2);
        let cost = TestCost::default();
        let plan = boxed
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        boxed.verify_plan(&g, &plan).unwrap();
        assert!(!boxed.lower_bound(&g, &cost).is_empty());
    }

    #[test]
    fn seed_selects_the_cheapest_class_member() {
        let (g, roots, cheap, _dear, class) = crate::realize::testkit::seeded_graph();
        let cost = TestCost::default();
        let lb = crate::lower_bound::lower_bound(&g, &cost);
        let ex = search().seed(&g, &roots, &lb, &cost).unwrap();
        assert_eq!(ex.selected(ClassId(class.0)), Some(cheap));
    }

    #[test]
    fn every_root_lands_in_a_buffer() {
        let (g, roots) = chain_graph(3);
        let cost = TestCost::default();
        let plan = search()
            .extract(&g, &roots, &cost, ExtractBudget::default())
            .unwrap();
        for r in &roots {
            let sel = realize::select(&g, &plan.extraction, *r).unwrap();
            assert!(plan.extraction.is_materialized(sel));
            assert!(plan.buffers.iter().any(|b| b.value == sel));
        }
    }
}
