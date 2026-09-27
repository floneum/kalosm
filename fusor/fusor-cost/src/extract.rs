//! Deterministic extraction over class selections and schedule points.
//!
//! Ordinary lowerings seed the selection. Single and compound moves compete
//! on the exact cost of the completed dispatch DAG. Composite ownership
//! fixes buffer obligations before either costing or plan construction.

use crate::debug;
use crate::lower_bound::argmin_member;
use crate::moves::{self, SchedCache, Trail};
use crate::nodes::{Mnkb, domain_of, is_composite, is_group, plan_values, resolved_children};
use crate::plan::derive_plan;
use crate::realize::{self, NodeCache, Realized};
use fixedbitset::FixedBitSet;
use fusor_ir::Result;
use fusor_ir::cost::{CostModel, Picoseconds};
use fusor_ir::device::Caps;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::error::Error;
use fusor_ir::extract::{Dispatch, ExtractBudget, Extraction, Extractor, Plan};
use fusor_ir::ir::Op;
use fusor_ir::ir::kernel::ArenaPlanner;
use fusor_ir::ir::launch::{Launch, SchedPoint, ScheduleDomain};
use fusor_ir::shape::Dim;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::sync::Arc;
use web_time::Instant;

/// The shipped extraction. Deterministic: ties break by node id, then by
/// [`fusor_ir::extract::Move`] discriminant order, which is the order
/// `moves::frontier` emits them in.
pub struct LocalSearch {
    arena: Arc<dyn ArenaPlanner>,
    caps: Caps,
}

/// What the search actually did. Exposed so conformance can assert the
/// budget was honoured and best-so-far never regressed.
#[derive(Clone, Debug, Default)]
pub struct SearchTrace {
    pub moves: u32,
    /// Realizations `co_select` spent; bounded separately from `moves`.
    pub co_moves: u32,
    pub chains: u32,
    pub micros: u64,
    /// Best-so-far after the seed and after every accepted move, from either
    /// pass.
    pub best: Vec<Picoseconds>,
}

impl LocalSearch {
    pub fn new(arena: Arc<dyn ArenaPlanner>, caps: Caps) -> Self {
        Self { arena, caps }
    }

    pub fn caps(&self) -> &Caps {
        &self.caps
    }

    pub fn arena(&self) -> &Arc<dyn ArenaPlanner> {
        &self.arena
    }

    fn search<'a>(
        &'a self,
        graph: &'a EGraph,
        roots: &'a [Id],
        cost: &'a dyn CostModel,
    ) -> Search<'a> {
        Search {
            graph,
            roots,
            cost,
            arena: self.arena.as_ref(),
            caps: &self.caps,
        }
    }

    /// Step 2: the seed selection, its schedule points and its initial
    /// materialized set.
    pub fn seed(
        &self,
        graph: &EGraph,
        roots: &[Id],
        lb: &[Picoseconds],
        cost: &dyn CostModel,
    ) -> Result<Extraction> {
        let (classes, mask) = realize::reachable(graph, roots);
        let launches = crate::lower_bound::bounds_scoped(graph, None, &mask).launches;
        let mut cache = NodeCache::new(graph.len());
        self.search(graph, roots, cost)
            .seed_realized(lb, &launches, &classes, &mut cache, None)
            .map(|seeded| seeded.ex)
    }

    /// Extract a plan and report the deterministic search work.
    pub fn extract_traced(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
    ) -> Result<(Plan, SearchTrace)> {
        self.extract_seeded_traced(graph, roots, cost, budget, None)
    }

    fn extract_seeded_traced(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
        seed: Option<&Plan>,
    ) -> Result<(Plan, SearchTrace)> {
        let started = Instant::now();
        let search = self.search(graph, roots, cost);
        // Everything below is scoped to the classes this resolve's roots
        // reach; a long-lived session graph holds every value it ever built.
        let (classes, mask) = realize::reachable(graph, roots);
        let crate::lower_bound::Bounds {
            costs: lb,
            launches,
        } = crate::lower_bound::bounds_scoped(graph, Some(cost), &mask);
        let mut cache = NodeCache::new(graph.len());
        let mut at = search.seed_realized(&lb, &launches, &classes, &mut cache, seed)?;
        // The seed is priced the same way every candidate below is: as the
        // plan it denotes, not as the state the seeding pass left.
        let mut trail = Trail::default();
        match search.price(&mut at.ex, &mut cache, &mut trail) {
            Some((realized, cost)) => {
                at.realized = realized;
                at.cost = cost;
            }
            None => trail.rollback(&mut at.ex, 0),
        }

        let chains = classes.len() as u32;
        // The cap is the only stopping condition. A wall clock here would
        // make the winning plan — and the `PlanHash` the cross-process cache
        // is keyed on — depend on machine load. The work divisor is the
        // scoped node count, since every move re-realizes the DAG under the
        // roots.
        let cap = budget.move_cap(mask.count_ones(..), chains);
        let readers = readers_by_producer(graph, &classes, &self.caps);
        let mut producers: Vec<ClassId> = readers.keys().copied().collect();
        producers.sort_unstable();
        let climb = Climb {
            lb: &lb,
            readers: &readers,
            producers: &producers,
            cap,
        };
        let initial = at.cost;

        // Adopting all views of a multi-slot reduction can improve the plan
        // even when adopting one view alone cannot. Descend from both that
        // joint adoption and the plain seed, keeping the cheaper result.
        let joints = joint_producers(graph, &readers);
        let plain = at.clone();
        let mut a = SearchTrace::default();
        while a.co_moves < cap && search.co_select(&joints, &climb, &mut at, &mut cache, &mut a) {}
        let joint_moved = !a.best.is_empty();
        search.descend(&climb, &mut at, &mut cache, &mut a);

        let mut trace = if joint_moved {
            let mut b_at = plain;
            let mut b = SearchTrace::default();
            search.descend(&climb, &mut b_at, &mut cache, &mut b);
            let (moves, co_moves) = (a.moves.max(b.moves), a.co_moves.max(b.co_moves));
            let mut chosen = if b_at.cost <= at.cost {
                at = b_at;
                b
            } else {
                a
            };
            chosen.moves = moves;
            chosen.co_moves = co_moves;
            chosen
        } else {
            // The seeded sweep changed nothing, so the two starts are the same
            // state and the second descent would repeat the first move for
            // move.
            a
        };
        trace.chains = chains;
        trace.best.insert(0, initial);

        let plan = derive_plan(graph, &at.ex, &at.realized, cost.facts(), at.cost)?;
        debug::dump_plan(graph, &plan, &at.ex, &at.realized, &self.caps, cost);
        #[cfg(feature = "compiler-tests")]
        crate::verify_plan::verify_plan_with(graph, &plan, self.arena.as_ref(), &self.caps)
            .unwrap_or_else(|e| panic!("constructed plan violates a compiler invariant: {e}"));

        trace.micros = started.elapsed().as_micros() as u64;
        Ok((plan, trace))
    }

    /// Complete a selection and construct its plan without searching. This
    /// is the same construction used to price local-search candidates.
    ///
    /// The `cache` is the caller's: `Work` is a property of a graph node and
    /// nothing in it moves with the extraction.
    pub fn replan(
        &self,
        graph: &EGraph,
        roots: &[Id],
        ex: &mut Extraction,
        cost: &dyn CostModel,
        cache: &mut NodeCache,
    ) -> Result<Plan> {
        cache.clear_schedules();
        let mut trail = Trail::default();
        let plan = self
            .search(graph, roots, cost)
            .price(ex, cache, &mut trail)
            .ok_or_else(|| Error::Plan("autotune candidate does not realize".into()))
            .and_then(|(realized, exact)| derive_plan(graph, ex, &realized, cost.facts(), exact));
        if plan.is_err() {
            trail.rollback(ex, 0);
        }
        // Conformance checks the plan independently of the constructor.
        #[cfg(feature = "compiler-tests")]
        if let Ok(plan) = &plan {
            crate::verify_plan::verify_plan_with(graph, plan, self.arena.as_ref(), &self.caps)
                .unwrap_or_else(|e| panic!("constructed plan violates a compiler invariant: {e}"));
        }
        plan
    }
}

/// What every pricing within one extraction shares.
struct Search<'a> {
    graph: &'a EGraph,
    roots: &'a [Id],
    cost: &'a dyn CostModel,
    arena: &'a dyn ArenaPlanner,
    caps: &'a Caps,
}

/// A selection, the DAG it realizes and that DAG's exact cost.
#[derive(Clone)]
struct Incumbent {
    ex: Extraction,
    realized: Realized,
    cost: Picoseconds,
}

/// The fixed inputs of one descent.
struct Climb<'a> {
    lb: &'a [Picoseconds],
    readers: &'a FxHashMap<ClassId, Vec<(ClassId, Id)>>,
    /// Every key of `readers`, ascending.
    producers: &'a [ClassId],
    cap: u32,
}

impl Search<'_> {
    /// Construct the seed's required buffers before partitioning its launches.
    fn seed_realized(
        &self,
        lb: &[Picoseconds],
        launches: &[u32],
        classes: &[ClassId],
        cache: &mut NodeCache,
        seed: Option<&Plan>,
    ) -> Result<Incumbent> {
        let graph = self.graph;
        let mut ex = Extraction {
            sigma: FxHashMap::with_capacity_and_hasher(classes.len(), Default::default()),
            m: FixedBitSet::with_capacity(graph.len()),
            theta: FxHashMap::default(),
        };
        for class in classes {
            // Start from a concrete lowering before introducing composites
            // or decompositions whose dependencies must be priced together.
            let pick = realize::selectable(graph, *class, self.caps)
                .into_iter()
                .filter(|id| !is_composite(graph, *id))
                .min()
                .unwrap_or_else(|| argmin_member(graph, lb, launches, *class, self.caps));
            debug::sigma(*class, pick, "seed");
            ex.sigma.insert(*class, pick);
        }
        let mut reused = FxHashMap::default();
        if let Some(seed) = seed {
            let live: FxHashSet<_> = plan_values(seed)
                .filter(|member| member.index() < graph.len())
                .map(|member| graph.class_of(member))
                .collect();
            for &member in seed.extraction.sigma.values() {
                if member.index() >= graph.len() {
                    reused.clear();
                    break;
                }
                let class = graph.class_of(member);
                if !live.contains(&class) || !ex.sigma.contains_key(&class) {
                    continue;
                }
                if reused
                    .insert(class, member)
                    .is_some_and(|old| old != member)
                {
                    reused.clear();
                    break;
                }
            }
            for (&class, &member) in &reused {
                ex.sigma.insert(class, member);
                if let Some(&theta) = seed.extraction.theta.get(&member) {
                    ex.theta.insert(member, theta);
                }
            }
            // A class's representative changes whenever it gains a variant.
            reused.retain(|class, _| seed.extraction.sigma.contains_key(class));
        }
        loop {
            let attempt = pin_selection(graph, self.roots, &mut ex, &mut Trail::default(), cache)
                .and_then(|selected| {
                    seed_theta_trailed(
                        graph,
                        &mut ex,
                        &selected.order,
                        self.cost,
                        cache,
                        &mut Trail::default(),
                    );
                    ex.m = selected.buffers(graph);
                    selected.realize(graph, &ex, self.cost, self.arena, cache)
                });
            match attempt {
                Ok(realized) => {
                    let cost = realize::exact_cost(&realized, &ex, self.cost);
                    let at = Incumbent { ex, realized, cost };
                    return Ok(self.choose_seed(at, lb, cache, &reused));
                }
                Err(error) => {
                    if seed.is_some() {
                        return self.seed_realized(lb, launches, classes, cache, None);
                    }
                    if !break_selection_cycles(graph, self.roots, &mut ex, lb, launches, self.caps)?
                    {
                        return Err(error);
                    }
                }
            }
        }
    }

    /// One finite sweep over the seed's live classes, producers before
    /// consumers. Every comparison prices the complete selected DAG, so shared
    /// producers execute once and all independent branches contribute.
    fn choose_seed(
        &self,
        mut at: Incumbent,
        order_costs: &[Picoseconds],
        cache: &mut NodeCache,
        reused: &FxHashMap<ClassId, Id>,
    ) -> Incumbent {
        let graph = self.graph;
        let classes: Vec<_> = at
            .realized
            .components
            .iter()
            .filter(|component| {
                reused.get(&graph.class_of(component.root)) != Some(&component.root)
            })
            .map(|component| graph.class_of(component.root))
            .collect();
        let ordinary = |id| matches!(graph.node(id).op, Op::Launch(_)) && !is_composite(graph, id);
        let inputs = |id: Id| -> Vec<ClassId> {
            graph
                .node(id)
                .children
                .iter()
                .map(|child| graph.class_of(*child))
                .collect()
        };
        for class in classes {
            let mut members = realize::selectable(graph, class, self.caps);
            members.sort_by_key(|member| (order_costs[member.index()], *member));
            for candidate in members {
                let Some(index) = at
                    .realized
                    .components
                    .iter()
                    .position(|component| graph.class_of(component.root) == class)
                else {
                    break;
                };
                let component = &at.realized.components[index];
                let current = component.root;
                if candidate == current {
                    continue;
                }
                let same_inputs = component.members.as_slice() == [current]
                    && ordinary(current)
                    && ordinary(candidate)
                    && inputs(current) == inputs(candidate);
                let mut trail = Trail::default();
                trail.select(&mut at.ex, class, candidate);
                // A same-inputs swap is screened on its own launch first.
                if same_inputs {
                    seed_theta_trailed(
                        graph,
                        &mut at.ex,
                        &[candidate],
                        self.cost,
                        cache,
                        &mut trail,
                    );
                    let score = realize::ordinary_component(
                        graph, &at.ex, candidate, self.cost, self.arena, cache,
                    )
                    .map(|replacement| {
                        at.realized
                            .cost_replacing(index, &replacement, &at.ex, self.cost)
                    });
                    if !score.is_ok_and(|score| score < at.cost) {
                        trail.rollback(&mut at.ex, 0);
                        continue;
                    }
                }
                self.improve(&mut at, cache, &mut trail);
            }
        }
        at
    }

    /// Single moves until none improves, then compound co-selections until a
    /// sweep improves nothing.
    fn descend(
        &self,
        climb: &Climb<'_>,
        at: &mut Incumbent,
        cache: &mut NodeCache,
        trace: &mut SearchTrace,
    ) {
        let mut sched = SchedCache::new();
        'search: loop {
            let mut improved = false;
            for mv in moves::frontier(self.graph, &at.realized.order) {
                if trace.moves >= climb.cap {
                    break 'search;
                }
                let options = moves::candidates(
                    self.graph,
                    &at.ex,
                    &at.realized.order,
                    mv,
                    climb.lb,
                    &mut sched,
                    self.cost,
                );
                for candidate in options {
                    if trace.moves >= climb.cap {
                        break 'search;
                    }
                    trace.moves += 1;
                    let mut trail = Trail::default();
                    if trail.apply(&mut at.ex, candidate) && self.improve(at, cache, &mut trail) {
                        trace.best.push(at.cost);
                        improved = true;
                        break;
                    }
                }
            }
            if !improved {
                break;
            }
        }

        // Step 4b: the compound move the single-move climb above cannot
        // make. Its own counter: the climb normally spends every move `cap`
        // allows, so a shared counter would make this pass unreachable.
        while trace.co_moves < climb.cap && self.co_select(climb.producers, climb, at, cache, trace)
        {
        }
    }

    /// One co-selection sweep over `producers`, in the order given. For each
    /// producer class, adopt together every class that holds a selectable
    /// member reading it; keep on a strict improvement in exact global cost,
    /// revert the trial otherwise.
    ///
    /// This pass reaches members the budget otherwise keeps unselected, so it
    /// leans on the e-graph invariant that every member of a class computes
    /// the same value. Do not weaken that guard to buy launches back.
    ///
    /// One realization per producer class that has two or more reading
    /// classes, counted against `cap`, so the whole extraction remains
    /// bounded by [`ExtractBudget`] and stays a pure function of the graph.
    fn co_select(
        &self,
        producers: &[ClassId],
        climb: &Climb<'_>,
        at: &mut Incumbent,
        cache: &mut NodeCache,
        trace: &mut SearchTrace,
    ) -> bool {
        let mut improved = false;
        for p in producers {
            if trace.co_moves >= climb.cap {
                break;
            }
            // The smallest-id member of each reading class that is not already
            // the selected one. `readers[p]` is sorted, so the first entry per
            // class is that member.
            let mut proposal: Vec<(ClassId, Id)> = Vec::new();
            for (c, m) in &climb.readers[p] {
                if proposal.last().is_some_and(|(last, _)| last == c) {
                    continue;
                }
                if at.ex.sigma.get(c).copied() != Some(*m) {
                    proposal.push((*c, *m));
                }
            }
            if proposal.len() < 2 {
                continue;
            }
            let mut trail = Trail::default();
            for (class, node) in &proposal {
                trail.apply(
                    &mut at.ex,
                    moves::Candidate::Select {
                        class: *class,
                        node: *node,
                    },
                );
            }
            if trail.mark() == 0 {
                continue;
            }
            trace.co_moves += 1;
            if self.improve(at, cache, &mut trail) {
                trace.best.push(at.cost);
                improved = true;
            }
        }
        improved
    }

    /// The DAG `ex` denotes once its obligations are recorded on `trail`,
    /// and its exact cost.
    fn price(
        &self,
        ex: &mut Extraction,
        cache: &mut NodeCache,
        trail: &mut Trail,
    ) -> Option<(Realized, Picoseconds)> {
        let selected = pin_selection(self.graph, self.roots, ex, trail, cache).ok()?;
        seed_theta_trailed(self.graph, ex, &selected.order, self.cost, cache, trail);
        trail.buffers(ex, selected.buffers(self.graph));
        let realized = selected
            .realize(self.graph, ex, self.cost, self.arena, cache)
            .ok()?;
        let c = realize::exact_cost(&realized, ex, self.cost);
        Some((realized, c))
    }

    /// Price the trial `trail` made on `at` and keep it on a strict
    /// improvement; otherwise undo it with the obligations it implied, so a
    /// rejected trial leaves no trace. A tie keeps the earlier (smaller-id)
    /// state, which keeps the search reproducible.
    fn improve(&self, at: &mut Incumbent, cache: &mut NodeCache, trail: &mut Trail) -> bool {
        match self.price(&mut at.ex, cache, trail) {
            Some((realized, cost)) if cost < at.cost => {
                at.realized = realized;
                at.cost = cost;
                true
            }
            _ => {
                trail.rollback(&mut at.ex, 0);
                false
            }
        }
    }
}

impl LocalSearch {
    #[allow(clippy::too_many_arguments)]
    fn candidate_plans(
        &self,
        graph: &EGraph,
        roots: &[Id],
        base: &Plan,
        launch_ix: usize,
        cost: &dyn CostModel,
        min_macs: u64,
        points: fn(&ScheduleDomain) -> SmallVec<[SchedPoint; 8]>,
        limit: usize,
    ) -> Vec<(String, Plan)> {
        let Some((root, class)) = launch_target(graph, base, launch_ix, min_macs) else {
            return Vec::new();
        };
        // No purity guard here: whether a plan may be re-run is a property of
        // the caller's use. `Session::autotune` refuses impure plans before
        // probing; the production explorer runs a candidate exactly once,
        // instead of the incumbent, so an impure plan's pure launches stay
        // explorable.
        let fair = fair_points(
            graph,
            class,
            base.extraction.theta.get(&root).copied(),
            root,
            points,
            limit != usize::MAX,
        );
        let mut out: Vec<(String, Plan)> = Vec::new();
        // One cache for the whole sweep: `Work` is a property of the graph
        // alone.
        let mut cache = NodeCache::new(graph.len());
        for (member, theta, label) in fair {
            if out.len() >= limit {
                debug::tune(launch_ix, || format!("cap reached at {}", out.len()));
                return out;
            }
            let mut ex = base.extraction.clone();
            if !apply_variant(&mut Trail::default(), &mut ex, class, root, member, theta) {
                debug::tune(launch_ix, || format!("SELECT-FAIL {member:?} {label}"));
                continue;
            }
            // A candidate may change the dispatch count: selecting a
            // member whose operand must materialize adds that producer's
            // launch, and dropping one removes it. Such candidates are
            // raced like any tile and adopted only on a measured
            // whole-plan win.
            let plan = match self.replan(graph, roots, &mut ex, cost, &mut cache) {
                Ok(plan) => plan,
                Err(e) => {
                    debug::tune(launch_ix, || format!("REPLAN-FAIL {member:?} {label}: {e}"));
                    continue;
                }
            };
            debug::tune(launch_ix, || format!("OFFER {member:?} {label}"));
            out.push((label, plan));
        }
        out
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

    fn extract_seeded(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
        seed: &Plan,
    ) -> Result<Plan> {
        self.extract_seeded_traced(graph, roots, cost, budget, Some(seed))
            .map(|(plan, _)| plan)
    }

    #[cfg(feature = "compiler-tests")]
    fn verify_plan(&self, graph: &EGraph, plan: &Plan) -> Result<()> {
        crate::verify_plan::verify_plan_with(graph, plan, self.arena.as_ref(), &self.caps)
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
        self.candidate_plans(
            graph,
            roots,
            base,
            launch_ix,
            cost,
            min_macs,
            sample_points,
            TUNE_MAX_VARIANTS,
        )
    }

    #[cfg(feature = "compiler-tests")]
    fn test_launch_variants(
        &self,
        graph: &EGraph,
        roots: &[Id],
        base: &Plan,
        launch_ix: usize,
        cost: &dyn CostModel,
    ) -> Vec<(String, Plan)> {
        self.candidate_plans(
            graph,
            roots,
            base,
            launch_ix,
            cost,
            0,
            test_points,
            usize::MAX,
        )
    }

    fn launch_variant_labels(
        &self,
        graph: &EGraph,
        roots: &[Id],
        base: &Plan,
        launch_ix: usize,
        cost: &dyn CostModel,
        min_macs: u64,
    ) -> Vec<(String, Picoseconds)> {
        let Some((root, class)) = launch_target(graph, base, launch_ix, min_macs) else {
            return Vec::new();
        };
        let search = self.search(graph, roots, cost);
        let mut ex = base.extraction.clone();
        let mut trail = Trail::default();
        let mut cache = NodeCache::new(graph.len());
        let here = base.extraction.theta.get(&root).copied();
        let mut labels: Vec<_> = fair_points(graph, class, here, root, sample_points, true)
            .into_iter()
            .take(TUNE_MAX_VARIANTS)
            .map(|(member, theta, label)| {
                let score = if apply_variant(&mut trail, &mut ex, class, root, member, theta) {
                    search
                        .price(&mut ex, &mut cache, &mut trail)
                        .map(|(_, cost)| cost)
                } else {
                    None
                };
                trail.rollback(&mut ex, 0);
                (label, score.unwrap_or(Picoseconds(u64::MAX)))
            })
            .collect();
        labels.sort_by(|(a, ac), (b, bc)| ac.cmp(bc).then_with(|| a.cmp(b)));
        labels
    }

    /// The batch adoption path: resolve each label to its `(member, theta)`
    /// by signature — no replans — apply every selection and schedule move
    /// onto one cloned extraction, and construct the plan once. The per-swap
    /// candidate enumeration is `fair_points`, the same walk
    /// `launch_variants` and `launch_variant_labels` offer from, so a label
    /// either of them names resolves here and no other does.
    fn replan_extraction(
        &self,
        graph: &EGraph,
        roots: &[Id],
        ex: &mut Extraction,
        cost: &dyn CostModel,
    ) -> Result<Plan> {
        self.replan(graph, roots, ex, cost, &mut NodeCache::new(graph.len()))
    }

    fn replan_with_variants(
        &self,
        graph: &EGraph,
        roots: &[Id],
        base: &Plan,
        cost: &dyn CostModel,
        min_macs: u64,
        swaps: &[(usize, String)],
    ) -> Option<Plan> {
        let mut ex = base.extraction.clone();
        let mut trail = Trail::default();
        let mut applied = false;
        for (ix, name) in swaps {
            let Some((root, class)) = launch_target(graph, base, *ix, min_macs) else {
                continue;
            };
            let here = base.extraction.theta.get(&root).copied();
            let Some((member, theta, _)) =
                fair_points(graph, class, here, root, sample_points, true)
                    .into_iter()
                    .find(|(_, _, label)| label == name)
            else {
                continue;
            };
            applied |= apply_variant(&mut trail, &mut ex, class, root, member, theta);
        }
        if !applied {
            return None;
        }
        self.replan_extraction(graph, roots, &mut ex, cost).ok()
    }
}

/// Launch `ix`'s root and its class, when its work reaches `min_macs`.
fn launch_target(graph: &EGraph, base: &Plan, ix: usize, min_macs: u64) -> Option<(Id, ClassId)> {
    let root = base.launches.get(ix)?.root;
    (launch_work(graph, base, ix) >= min_macs).then(|| (root, graph.class_of(root)))
}

/// Select `member` for `class` unless it is the incumbent `root`, then
/// schedule it at `theta`. `false` when the selection does not apply.
fn apply_variant(
    trail: &mut Trail,
    ex: &mut Extraction,
    class: ClassId,
    root: Id,
    member: Id,
    theta: SchedPoint,
) -> bool {
    if member != root
        && !trail.apply(
            ex,
            moves::Candidate::Select {
                class,
                node: member,
            },
        )
    {
        return false;
    }
    trail.schedule(ex, member, theta);
    true
}

/// For each producer class, every `(reading class, member)` pair that reads
/// it, both ascending. Built once per extraction: it is a function of the
/// graph alone, while which of the pairs is a move depends on `sigma` and is
/// decided per sweep.
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
    // Ascending by class then member, so the sweep below is a pure function
    // of the graph and not of hash order.
    for v in out.values_mut() {
        v.sort_unstable();
    }
    out
}

/// The producer classes holding a **multi-slot carrier** — a node that fuses
/// several values into one, so its readers are slot views and adopting one
/// alone is strictly worse than adopting none.
///
/// Ascending, so a sweep over it is a pure function of the graph.
fn joint_producers(
    graph: &EGraph,
    readers: &FxHashMap<ClassId, Vec<(ClassId, Id)>>,
) -> Vec<ClassId> {
    let mut out: Vec<ClassId> = readers
        .keys()
        .copied()
        .filter(|p| {
            graph.members(*p).iter().any(|m| {
                matches!(&graph.node(*m).op, Op::Launch(Launch::Fold { carrier, .. })
                    if carrier.width() > 1)
            })
        })
        .collect();
    out.sort_unstable();
    out
}

/// Re-select, class by class, until the seeded selection is acyclic.
///
/// The seed picks each class's member independently, so two picks can name
/// each other; a move cannot leave this state because it re-prices through
/// `realize_with`, which fails, and the move is unwound. Each round strikes
/// the member that closed the loop off that class's pool and re-runs `argmin`
/// over the rest, so the loop terminates. A class with no candidate left is a
/// real failure and is reported as one.
///
/// Returns whether anything was re-selected, so the caller can tell a repaired
/// seed from a probe that failed for some other reason.
fn break_selection_cycles(
    graph: &EGraph,
    roots: &[Id],
    ex: &mut Extraction,
    lb: &[Picoseconds],
    launches: &[u32],
    caps: &Caps,
) -> Result<bool> {
    let mut banned: FxHashMap<ClassId, FxHashSet<Id>> = FxHashMap::default();
    let mut repaired = false;
    let mut seen_cycles = 0usize;
    while let Some(v) = realize::selection_cycle(graph, ex, roots) {
        let class = graph.class_of(v);
        seen_cycles += 1;
        debug::cycle(graph, ex, v, seen_cycles);
        let out = banned.entry(class).or_default();
        out.insert(v);
        let next =
            crate::lower_bound::argmin_member_excluding(graph, lb, launches, class, caps, out);
        let (class, next) = match next {
            Some(next) => (class, next),
            None => {
                // Every spelling of this class closes the cycle: it runs
                // through a group somewhere on the path, whose bundling of
                // independent launches is what made the order circular.
                // Re-select that group's class without it.
                let Some((gclass, group)) = cycle_group(graph, ex, v) else {
                    return Err(Error::Plan(format!(
                        "selection is cyclic through {v} and class {} has no acyclic member: \
                         every candidate names a class that names it back",
                        class.0
                    )));
                };
                let out = banned.entry(gclass).or_default();
                out.insert(group);
                let Some(next) = crate::lower_bound::argmin_member_excluding(
                    graph, lb, launches, gclass, caps, out,
                ) else {
                    return Err(Error::Plan(format!(
                        "selection is cyclic through {v}; group {group} in class {} has no replacement",
                        gclass.0
                    )));
                };
                (gclass, next)
            }
        };
        debug::sigma(class, next, "break_selection_cycles");
        ex.sigma.insert(class, next);
        repaired = true;
    }
    Ok(repaired)
}

/// A selected group on the selection cycle through `v`, with its class:
/// the walk from `v` back to itself under the current selection, stopping
/// at the first group met.
fn cycle_group(graph: &EGraph, ex: &Extraction, v: Id) -> Option<(ClassId, Id)> {
    let mut stack: Vec<Id> = vec![v];
    let mut seen: FxHashSet<Id> = FxHashSet::default();
    while let Some(x) = stack.pop() {
        for (c, n) in resolved_children(graph, ex, x) {
            let n = n.unwrap_or(c);
            if is_group(graph, n) {
                return Some((graph.class_of(n), n));
            }
            if seen.insert(n) {
                stack.push(n);
            }
        }
    }
    None
}

/// Fill missing schedules and record them for rollback.
fn seed_theta_trailed(
    graph: &EGraph,
    ex: &mut Extraction,
    selected: &[Id],
    cost: &dyn CostModel,
    cache: &mut NodeCache,
    trail: &mut Trail,
) {
    for &id in selected {
        if ex.theta.contains_key(&id) {
            continue;
        }
        if let Some(theta) = cache.seed_schedule(graph, id, cost) {
            trail.schedule(ex, id, theta);
        }
    }
}

fn pin_selection(
    graph: &EGraph,
    roots: &[Id],
    ex: &mut Extraction,
    trail: &mut Trail,
    cache: &mut NodeCache,
) -> Result<realize::Selected> {
    loop {
        let before = trail.mark();
        let selected = realize::Selected::new(graph, ex, roots, cache)?;
        pin_slabs_trailed(graph, ex, &selected.order, trail);
        if !trail.selected_since(before) {
            return Ok(selected);
        }
        selected.recycle(cache);
    }
}

/// Select each composite's concrete middle members and resolve overlapping
/// ownership before constructing its dispatch and buffers.
fn pin_slabs_trailed(graph: &EGraph, ex: &mut Extraction, order: &[Id], trail: &mut Trail) {
    use crate::nodes::composite_members as members_of;
    /// Every node a composite runs: its members and, for member
    /// composites, theirs.
    fn flat_members(graph: &EGraph, id: Id) -> Vec<Id> {
        let mut out = Vec::new();
        let mut stack = vec![id];
        while let Some(x) = stack.pop() {
            if let Some(ms) = members_of(graph, x) {
                for m in ms.iter() {
                    out.push(*m);
                    stack.push(*m);
                }
            }
        }
        out
    }
    // Realized composites only: `sigma` also holds selections for classes
    // nothing reaches any more, and a pin from one of those would reach
    // into live classes. Groups first, then larger slabs: pinning a
    // composite's middle member as its class's selection deselects any
    // smaller slab that ended there.
    // A group holds the roots before its head, so groups go latest head
    // first: each pins the window before it, and the next unpinned head's
    // group is the window before that.
    let root_of = |id: Id| -> u32 {
        let class = graph.class_of(id);
        graph
            .roots()
            .iter()
            .filter(|r| graph.class_of(**r) == class)
            .map(|r| r.0)
            .min()
            .unwrap_or(u32::MAX)
    };
    let mut slabs: Vec<(bool, u32, usize, Id)> = order
        .iter()
        .filter_map(|id| {
            members_of(graph, *id).map(|m| {
                let group = is_group(graph, *id);
                (
                    !group,
                    if group { u32::MAX - root_of(*id) } else { 0 },
                    m.len(),
                    *id,
                )
            })
        })
        .collect();
    slabs.sort_unstable_by_key(|(slab, head, n, id)| (*slab, *head, std::cmp::Reverse(*n), *id));
    slabs.dedup();

    // Pin `id` and its nested composites.
    fn pin(
        graph: &EGraph,
        ex: &mut Extraction,
        trail: &mut Trail,
        owned: &mut rustc_hash::FxHashMap<ClassId, Id>,
        id: Id,
    ) {
        let Some(members) = members_of(graph, id) else {
            return;
        };
        owned.extend(members.iter().map(|m| (graph.class_of(*m), id)));
        let Some((last, middle)) = members.split_last() else {
            return;
        };
        for m in middle {
            let class = graph.class_of(*m);
            trail.select(ex, class, *m);
            if members_of(graph, *m).is_some() {
                pin(graph, ex, trail, owned, *m);
            }
        }
        if members_of(graph, *last).is_some() {
            pin(graph, ex, trail, owned, *last);
        }
    }

    // Two realized composites may not share a member class: each would run
    // its stage, and the realizer would cut both into one launch. The first
    // keeps it; the other takes its longest spelling sharing nothing, else
    // its last member's own launch.
    let mut owned: rustc_hash::FxHashMap<ClassId, Id> = rustc_hash::FxHashMap::default();
    for (_, _, _, id) in slabs {
        let Some(members) = members_of(graph, id) else {
            continue;
        };
        let class = graph.class_of(id);
        if ex.sigma.get(&class).copied() != Some(id) {
            continue;
        }
        // Already pinned through a composite that contains it.
        if owned.contains_key(&class) {
            continue;
        }
        let flat = flat_members(graph, id);
        if flat.iter().any(|m| owned.contains_key(&graph.class_of(*m))) {
            let alt = graph
                .members(class)
                .into_iter()
                .filter(|a| *a != id)
                .filter_map(|a| {
                    members_of(graph, a)
                        .filter(|_| {
                            flat_members(graph, a)
                                .iter()
                                .all(|x| !owned.contains_key(&graph.class_of(*x)))
                        })
                        .map(|ms| (ms.len(), a))
                })
                .max_by_key(|(n, a)| (*n, std::cmp::Reverse(*a)))
                .map(|(_, a)| a);
            let next = alt.or_else(|| members.last().copied());
            if let Some(next) = next {
                trail.select(ex, class, next);
                if members_of(graph, next).is_some() {
                    pin(graph, ex, trail, &mut owned, next);
                }
            }
            continue;
        }
        pin(graph, ex, trail, &mut owned, id);
    }
}

/// Variants offered per launch, shared round-robin across every member of the
/// class, so a family's sample list earns roughly `16 / members` offered
/// points.
const TUNE_MAX_VARIANTS: usize = 16;

/// Coop geometries are generated `bm`-major, so a domain prefix is six
/// spellings of the same narrowest tile; this is a spread over the tile axis.
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
/// Cooperative samples use the first supported depth of each sampled tile.
fn sample_points(domain: &ScheduleDomain) -> SmallVec<[SchedPoint; 8]> {
    let mut out: SmallVec<[SchedPoint; 8]> = SmallVec::new();
    match domain {
        ScheduleDomain::Point => {}
        ScheduleDomain::Coop(d) => {
            for (bm, bn, bk) in TUNE_GEOMS {
                if let Some((index, _)) = d
                    .schedules
                    .iter()
                    .enumerate()
                    .find(|(_, p)| p.geom.bm == bm && p.geom.bn == bn && p.geom.bk == bk)
                {
                    out.push(d.point(index).expect("index belongs to the domain"));
                }
            }
        }
        other => {
            // The sgemv domain is seed-ordered, one cell of each structure in
            // the prefix: multi-column window-16, multi-column window-32,
            // whole-workgroup-per-element, so its front plus one deep probe is
            // the whole spread. Two points off any other family are enough to
            // tell it apart from Coop.
            let front = if matches!(other, ScheduleDomain::Sgemv(_)) {
                5
            } else {
                1
            };
            let n = other.len();
            for i in (0..front).chain([n / 2]) {
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

/// Work a whole launch issues, across every node family.
///
/// Work is summed over the launch's members, not just its root, because a
/// fused region's cost lives in the members. `transcendentals` are weighted
/// by the ratio the roofline's `trans_ps` implies against a MAC, rounded to a
/// small integer so the gate stays a pure function of the graph.
pub fn launch_work(graph: &EGraph, base: &Plan, launch_ix: usize) -> u64 {
    const TRANS_WEIGHT: u64 = 8;
    /// One byte of storage traffic, in mac-equivalents: a launch's time is
    /// `max(math, traffic)`, so a bandwidth-bound launch must count its bytes
    /// to be worth tuning. The weight is the device-class rate ratio, ~6.8T
    /// macs/s against ~250 GB/s, ≈27 macs per byte, rounded up.
    const BYTE_WEIGHT: u64 = 32;
    let Some(launch) = base.launches.get(launch_ix) else {
        return 0;
    };
    // A symbolic extent is priced at its bound value (the graph's dim hints),
    // not a nominal one: a symbolic plan's launches are as much work as a
    // concrete plan's, and deserve the same tuning.
    let extent = |d: Dim| {
        d.evaluate(&mut |s| {
            Some(
                graph
                    .dim_hints
                    .get(&s)
                    .copied()
                    .unwrap_or(realize::SYM_NOMINAL),
            )
        })
        .unwrap_or(realize::SYM_NOMINAL)
    };
    let mut total: u64 = 0;
    for m in &launch.members {
        let w = realize::work_at(graph, *m, extent);
        total = total
            .saturating_add(w.macs)
            .saturating_add(w.transcendentals.saturating_mul(TRANS_WEIGHT))
            .saturating_add(w.index_ops);
    }
    for b in &launch.bindings {
        let mut facts = graph.facts(b.value).clone();
        facts.shape = facts.shape.iter().map(|d| Dim::Const(extent(*d))).collect();
        total = total.saturating_add(realize::bytes_of(&facts).saturating_mul(BYTE_WEIGHT));
    }
    total
}

/// A launch's identity across processes, for the persistent tune cache.
///
/// Node `Id`s are graph-allocation order and mean nothing in the next process,
/// so the cache cannot key on them. This is the op family plus the extents and
/// dtypes that decide which schedule wins.
///
/// It keys the launch, not its root: `facts.shape` is the output shape, so a
/// fold's reduced extent is invisible to it, and the fused body is invisible
/// to it too. `TuneCache::record` merges by minimum, so a key that cannot
/// tell two kernels apart stores the cheaper one's span under the other one's
/// name.
pub fn launch_signature(graph: &EGraph, launch: &Dispatch) -> String {
    let root = launch.root;
    let facts = graph.facts(root);
    let extents = |dims: &[Dim]| -> String {
        dims.iter()
            .map(|d| {
                d.as_const()
                    .map_or_else(|| "s".to_string(), |v| v.to_string())
            })
            .collect::<Vec<_>>()
            .join("x")
    };
    let op = &graph.node(root).op;
    let tag = tag_of(op);
    let extra = match (op, Mnkb::of(op, |d| d.as_const().unwrap_or(0))) {
        (_, Some(c)) => format!("mnkb={},{},{},{}", c.m, c.n, c.k, c.batch),
        // `space` is the *iteration* domain and carries the reduced extent;
        // the output shape above does not.
        (
            Op::Launch(Launch::Fold {
                space,
                axis,
                vec_axes,
                carrier,
                ..
            }),
            _,
        ) => format!(
            "space=[{}] axis={axis} vec={vec_axes:?} slots={}",
            extents(&space.dims),
            carrier.slots.len()
        ),
        _ => String::new(),
    };
    // What the launch computes, one digest per member, sorted: member order
    // is a realization detail. Fusion in this IR happens inside a node, so
    // the digest is over the scalar bodies and operand accesses, which is
    // what differs.
    let mut body: Vec<String> = launch
        .members
        .iter()
        .map(|m| {
            let op = &graph.node(*m).op;
            // `body_digest` excludes operand `src` Ids, but the dtype behind
            // each operand is semantic, process-stable and kernel-deciding —
            // a Q4K and a Q6K matvec share every extent, scalar body and
            // access plan. Operand dtypes are folded in, in child order.
            use std::hash::{Hash, Hasher};
            let mut h = rustc_hash::FxHasher::default();
            h.write_u64(body_digest(op));
            for c in fusor_ir::semantics::children::children_of(op) {
                graph.facts(c).dtype.hash(&mut h);
            }
            format!("{:?}:{:08x}", op.tag(), h.finish() as u32)
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
/// Excludes `Operand::src`: that is a graph-allocation `Id` and means nothing
/// in the next process. Everything else that decides the emitted kernel is
/// hashed, and scalar bodies contribute [`ScalarExpr::structural_hash`], so
/// this is O(operands), not O(expression).
///
/// A symbolic extent hashes its `SymId`, which is allocation order within a
/// session; a program that allocates in a different order takes a cache miss
/// and one tuning pass, never a wrong answer.
fn body_digest(op: &Op) -> u64 {
    use fusor_ir::ir::launch::Operand;
    use fusor_ir::ir::visit::VisitMut;
    use rustc_hash::FxHasher;
    use std::hash::{Hash, Hasher};
    struct Sources;
    impl VisitMut for Sources {
        fn operand(&mut self, operand: &mut Operand) {
            operand.src = Id(0);
        }
    }
    let mut op = crate::plan::without_schedule(op);
    op.visit_mut(&mut Sources);
    let mut h = FxHasher::default();
    op.hash(&mut h);
    h.finish()
}

/// The label a plan's *own* choice at one launch files its observations
/// under: the same `(family, schedule point)` string a raced variant of that
/// launch gets, so production samples of the incumbent and race samples of
/// its challengers land in one field and rank against each other. A launch
/// whose root carries no schedule point is the domain's single point, labeled
/// `base`.
pub fn incumbent_signature(graph: &EGraph, plan: &Plan, launch_ix: usize) -> Option<String> {
    let launch = plan.launches.get(launch_ix)?;
    let root = launch.root;
    Some(match plan.extraction.theta.get(&root) {
        Some(theta) => variant_signature(graph, root, *theta),
        None => format!("{}|base", tag_of(&graph.node(root).op)),
    })
}

/// Every candidate one launch's root class offers, in the order a variant
/// sweep attempts them: `(member, schedule point, label)`.
///
/// Round-robin across members, not member-major: one point per member per
/// round means every member's first-choice geometry races before any
/// member's second.
///
/// One entry per label, not per `(member, theta)`: the tune store files every
/// same-labeled plan into one min-merged record, so racing two members at the
/// same point burns budget on information the store cannot keep apart.
///
/// The incumbent's own point is not a candidate and never appears.
fn fair_points(
    graph: &EGraph,
    class: ClassId,
    here: Option<SchedPoint>,
    root: Id,
    points: fn(&ScheduleDomain) -> SmallVec<[SchedPoint; 8]>,
    deduplicate_labels: bool,
) -> Vec<(Id, SchedPoint, String)> {
    let per_member: Vec<(Id, SmallVec<[SchedPoint; 8]>)> = graph
        .members(class)
        .into_iter()
        .filter_map(|member| Some((member, points(domain_of(graph, member)?))))
        .collect();
    let rounds = per_member.iter().map(|(_, p)| p.len()).max().unwrap_or(0);
    let mut offered: rustc_hash::FxHashSet<String> = rustc_hash::FxHashSet::default();
    let mut fair = Vec::new();
    for r in 0..rounds {
        for (member, points) in &per_member {
            let Some(&theta) = points.get(r) else {
                continue;
            };
            if *member == root && Some(theta) == here {
                continue;
            }
            let label = variant_signature(graph, *member, theta);
            if deduplicate_labels && !offered.insert(label.clone()) {
                continue;
            }
            let label = if deduplicate_labels {
                label
            } else {
                format!("{member}: {label}")
            };
            fair.push((*member, theta, label));
        }
    }
    fair
}

fn variant_signature(graph: &EGraph, member: Id, theta: SchedPoint) -> String {
    let op = &graph.node(member).op;
    let mut q = String::new();
    for child in fusor_ir::semantics::children::children_of(op) {
        if let Some((_, layout)) = realize::quantized_storage(graph, child) {
            q.push_str(&format!("|q={layout:?}"));
        }
    }
    format!("{}{q}|{theta:?}", tag_of(op))
}

/// An op's family as the tune cache files it.
fn tag_of(op: &Op) -> String {
    match op {
        Op::Launch(launch) => format!("{:?}", launch.tag()),
        Op::Logical(_) => "Logical".to_string(),
        Op::Union(..) => "U".to_string(),
    }
}

/// Structural coverage, independent of the tuner. Small domains are
/// exhaustive; large tiling domains exercise both ends and the midpoint.
/// Domain construction tests separately check every geometry's resources.
#[cfg(feature = "compiler-tests")]
fn test_points(domain: &ScheduleDomain) -> SmallVec<[SchedPoint; 8]> {
    let n = domain.len();
    if n <= 32 {
        return domain.iter().collect();
    }
    [0, 1, n / 2, n / 2 + 1, n - 2, n - 1]
        .into_iter()
        .filter_map(|i| domain.point(i))
        .collect()
}
