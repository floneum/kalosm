//! [`LocalSearch`] — the shipped [`Extractor`].
//!
//! 1. Admissible lower bound, bottom-up, O(nodes).
//! 2. Seed `sigma_0 = argmin lb`; realize; `m_0 = roots u {shared} u
//!    {index-space mismatch} u {InPlace}`; `theta_0` from the local ranking.
//! 3. Exact cost on the realized DAG.
//! 3b. `co_select` over the multi-slot carriers only, from the seed.
//!    Speculative: when it changes anything, step 4 runs from both states and
//!    the cheaper plan wins.
//! 4. Local search over `RESELECT`, `FLIP`, `RESCHEDULE`.
//! 4b. `co_select`, the compound move: adopt every reader of one producer
//!    class together.
//! 5. Budget, keeping best-so-far. Fully deterministic.
//! 6. `verify_plan` on the winner — a hard conformance assert, never a
//!    silent fallback.
//!
//! Every decision path iterates classes and nodes in ascending id order.
//! There is no RNG and no hash-map iteration order anywhere in this file.
//!
//! A rule that fuses `F` values into one node hands this file a node plus `F`
//! slot views of it, each in a different e-class. Adopting one view alone is
//! strictly worse than adopting none, so a search that accepts only single
//! strict improvements cannot reach those states. `co_select` closes the
//! gap by adopting every reader of one producer class together.

use crate::lower_bound::argmin_member;
use crate::moves::{self, SchedCache};
use crate::plan::derive_plan;
use crate::realize::{self, NodeCache, Realized};
use fixedbitset::FixedBitSet;
use fusor_ir::Result;
use fusor_ir::cost::{CostModel, Picoseconds, ShapeStats};
use fusor_ir::device::Caps;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::error::Error;
use fusor_ir::extract::{Dispatch, ExtractBudget, Extraction, Extractor, Plan};
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::Op;
use fusor_ir::ir::OpDefRegistry;
use fusor_ir::ir::kernel::ArenaPlanner;
use fusor_ir::ir::launch::{Effect, Launch, SchedPoint, ScheduleDomain};
use fusor_ir::shape::Dim;
use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::sync::Arc;
use std::time::Instant;

/// The shipped extraction. Deterministic: ties break by node id, then by
/// [`fusor_ir::extract::Move`] discriminant order, which is the order
/// `moves::frontier` emits them in.
pub struct LocalSearch {
    arena: Arc<dyn ArenaPlanner>,
    caps: Caps,
    registry: Option<OpDefRegistry>,
    /// Bounded per-extractor record of which dim bindings each plan has been
    /// seen at.
    stats: Mutex<ShapeStats>,
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
        Self {
            arena,
            caps,
            registry: None,
            stats: Mutex::new(ShapeStats::new()),
        }
    }

    /// Supply the registry `Launch::Ext` nodes were built against, so
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
        let launches = crate::lower_bound::launch_bound_scoped(graph, &mask);
        let mut cache = NodeCache::new(graph.len());
        self.seed_realized(graph, roots, lb, &launches, cost, &classes, &mut cache)
            .map(|(ex, _)| ex)
    }

    /// The seed plus the DAG it denotes. The probe realization is reused
    /// when `m_0` turned out to add nothing.
    fn seed_realized(
        &self,
        graph: &EGraph,
        roots: &[Id],
        lb: &[Picoseconds],
        launches: &[u32],
        cost: &dyn CostModel,
        classes: &[ClassId],
        cache: &mut NodeCache,
    ) -> Result<(Extraction, Realized)> {
        let mut ex = Extraction {
            sigma: FxHashMap::with_capacity_and_hasher(classes.len(), Default::default()),
            m: FixedBitSet::with_capacity(graph.len()),
            theta: FxHashMap::default(),
        };
        for class in classes {
            ex.sigma.insert(
                *class,
                argmin_member(graph, lb, launches, *class, &self.caps),
            );
        }
        seed_theta(graph, &mut ex, cost);

        // Consumer counts and index spaces are independent of `m`.
        pin_inplace(graph, &mut ex);
        for r in roots {
            let sel = realize::select(graph, &ex, *r)?;
            materialize(graph, &mut ex, sel);
        }
        // The per-class `argmin_member` above chooses each class without
        // knowing what any other class chose, so the selection can carry a
        // cycle even though the graph is acyclic. The walk `realize_with`
        // performs detects it; only a failed probe pays for the diagnosis.
        let probe = match realize::realize_with(graph, roots, &ex, cost, self.arena.as_ref(), cache)
        {
            Ok(probe) => probe,
            Err(first) => {
                if !break_selection_cycles(graph, roots, &mut ex, lb, launches, &self.caps)? {
                    // Not a cycle. `first` is the real diagnosis.
                    return Err(first);
                }
                seed_theta(graph, &mut ex, cost);
                pin_inplace(graph, &mut ex);
                for r in roots {
                    let sel = realize::select(graph, &ex, *r)?;
                    materialize(graph, &mut ex, sel);
                }
                realize::realize_with(graph, roots, &ex, cost, self.arena.as_ref(), cache)?
            }
        };
        let mut grew = false;
        for v in &probe.order {
            if realize::leaf_role(graph, *v) != realize::LeafRole::NotLeaf {
                continue;
            }
            if ex.is_materialized(*v) {
                continue;
            }
            if probe.consumers.copied(*v).unwrap_or(0) > 1 {
                materialize(graph, &mut ex, *v);
                grew = true;
                continue;
            }
            // A producer across any structural cut has to land in a buffer
            // or its consumer reads what nothing wrote.
            if moves::at_structural_boundary(graph, &probe, *v) {
                materialize(graph, &mut ex, *v);
                grew = true;
            }
        }
        if !grew {
            return Ok((ex, probe));
        }
        let realized = realize::realize_with(graph, roots, &ex, cost, self.arena.as_ref(), cache)?;
        Ok((ex, realized))
    }

    /// The full run, with an explicit [`ShapeStats`] so a caller driving many
    /// steps sees specialization amortize. `Extractor::extract` uses the
    /// extractor's own.
    pub fn extract_with_stats(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
        stats: &mut ShapeStats,
    ) -> Result<(Plan, SearchTrace)> {
        let started = Instant::now();
        // Everything below is scoped to the classes this resolve's roots
        // reach; a long-lived session graph holds every value it ever built.
        let (classes, mask) = realize::reachable(graph, roots);
        let lb = crate::lower_bound::lower_bound_scoped(graph, cost, &mask);
        let launches = crate::lower_bound::launch_bound_scoped(graph, &mask);
        let mut cache = NodeCache::new(graph.len());
        let (mut ex, seeded) =
            self.seed_realized(graph, roots, &lb, &launches, cost, &classes, &mut cache)?;
        // The seed is priced the same way every candidate below is: as the
        // plan it denotes, not as the state the seeding pass left.
        let (mut realized, mut best_cost) =
            match self.price(graph, roots, &mut ex, cost, &mut cache) {
                Ok((r, c, _)) => (r, c),
                Err(_) => {
                    let c = realize::exact_cost(&seeded, &ex, cost);
                    (seeded, c)
                }
            };

        let chains = classes.len() as u32;
        // The cap is the only stopping condition. A wall clock here would
        // make the winning plan — and the `PlanHash` the cross-process cache
        // is keyed on — depend on machine load. The work divisor is the
        // scoped node count, since every move re-realizes the DAG under the
        // roots.
        let cap = budget.move_cap(mask.count_ones(..), chains);
        let mut trace = SearchTrace {
            chains,
            best: vec![best_cost],
            ..SearchTrace::default()
        };
        let readers = readers_by_producer(graph, &classes, &self.caps);

        // Step 4b, from the seed, over the joints only — and then the whole
        // descent twice, keeping the cheaper plan.
        //
        // Restricted to producer classes holding a multi-slot carrier: the
        // only shape where adopting a single reader is provably worse than
        // adopting none. Unrestricted, a seeded sweep reaches class members
        // that are unequal to their siblings.
        //
        // Speculative, because entering a joint is one-way: once every reader
        // of a joint reads it, dropping one reader alone recomputes that
        // slot's nest while the joint still runs, so every single step back
        // out is a cost increase. When the seeded sweep changes anything, the
        // descent runs twice, once from each state, and the cheaper plan
        // wins; a tie keeps the un-seeded one.
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
                       cache: &mut NodeCache,
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

            // Step 4b: the compound move the single-move climb above cannot
            // make. Sweeps until a sweep improves nothing.
            //
            // Its own counter: the climb normally spends every move `cap`
            // allows, so a shared counter would make this pass unreachable.
            // The extraction stays a pure function of the graph.
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

            // The winner is the live state — up to the invariants a sequence
            // of independent moves does not preserve on its own.
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
            // state and the second descent would repeat the first move for
            // move.
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

        probe_dump(graph, &plan, &ex, &realized, &self.caps);
        stats.observe(plan.hash, &binding_of(graph, &realized));
        trace.micros = started.elapsed().as_micros() as u64;
        Ok((plan, trace))
    }

    /// Realize, [`repair`], re-realize: the cost of the plan this state
    /// denotes, which is the only number an accept test may compare.
    ///
    /// A move can put a producer across a structural cut; the buffer that
    /// producer needs is an obligation of the move and must be priced with it.
    ///
    /// The returned [`RepairTrail`] is what a rejecting caller reverts, so the
    /// obligations die with the move that implied them. The error arm carries
    /// one too: a state that fails to realize after repair still has the
    /// repair on it.
    fn price(
        &self,
        graph: &EGraph,
        roots: &[Id],
        ex: &mut Extraction,
        cost: &dyn CostModel,
        cache: &mut NodeCache,
    ) -> std::result::Result<(Realized, Picoseconds, RepairTrail), RepairTrail> {
        price(graph, roots, ex, cost, self.arena.as_ref(), cache)
    }

    /// The plan a given extraction denotes: realize, repair, re-realize,
    /// derive, verify. No search. A candidate is priced and built by exactly
    /// the path `extract` returns its winner through.
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
        let (realized, exact, _) = self
            .price(graph, roots, ex, cost, cache)
            .map_err(|_| Error::Plan("autotune candidate does not realize".into()))?;
        let plan = derive_plan(graph, ex, &realized, cost.facts(), exact)?;
        // A candidate dispatches on the real device against real buffers, so
        // it passes the same verifier the base plan does — or it is not built.
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
        // No purity guard here: whether a plan may be re-run is a property of
        // the caller's use. `Session::autotune` refuses impure plans before
        // probing; the production explorer runs a candidate exactly once,
        // instead of the incumbent, so an impure plan's pure launches stay
        // explorable.

        let class = graph.class_of(root);
        let fair = fair_points(
            graph,
            class,
            base.extraction.theta.get(&root).copied(),
            root,
        );
        let mut out: Vec<(String, Plan)> = Vec::new();
        // One cache for the whole sweep: `Work` is a property of the graph
        // alone.
        let mut cache = NodeCache::new(graph.len());
        // Names every candidate this sweep drops and why.
        let dbg = std::env::var_os("FUSOR_TUNE_DEBUG").is_some();
        {
            for (member, theta, label) in fair {
                if out.len() >= TUNE_MAX_VARIANTS {
                    if dbg {
                        eprintln!("[vdbg] L{launch_ix} cap reached at {}", out.len());
                    }
                    return out;
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
                    if dbg {
                        eprintln!("[vdbg] L{launch_ix} SELECT-FAIL {member:?} {label}",);
                    }
                    continue;
                }
                moves::apply(
                    graph,
                    &mut ex,
                    moves::Candidate::Schedule {
                        node: member,
                        theta,
                    },
                );
                // A candidate may change the dispatch count: selecting a
                // member whose operand must materialize adds that producer's
                // launch, and dropping one removes it. Such candidates are
                // raced like any tile and adopted only on a measured
                // whole-plan win.
                let plan = match self.replan(graph, roots, &mut ex, cost, &mut cache) {
                    Ok(plan) => plan,
                    Err(e) => {
                        if dbg {
                            eprintln!("[vdbg] L{launch_ix} REPLAN-FAIL {member:?} {label}: {e}",);
                        }
                        continue;
                    }
                };
                if dbg {
                    eprintln!("[vdbg] L{launch_ix} OFFER {member:?} {label}");
                }
                out.push((label, plan));
            }
        }
        out
    }

    fn launch_variant_labels(
        &self,
        graph: &EGraph,
        base: &Plan,
        launch_ix: usize,
        min_macs: u64,
    ) -> Vec<String> {
        let Some(launch) = base.launches.get(launch_ix) else {
            return Vec::new();
        };
        if launch_work(graph, base, launch_ix) < min_macs {
            return Vec::new();
        }
        let root = launch.root;
        fair_points(
            graph,
            graph.class_of(root),
            base.extraction.theta.get(&root).copied(),
            root,
        )
        .into_iter()
        .take(TUNE_MAX_VARIANTS)
        .map(|(_, _, label)| label)
        .collect()
    }

    /// The batch adoption path: resolve each label to its `(member, theta)`
    /// by signature — no replans — apply every selection and schedule move
    /// onto one cloned extraction, and replan/verify once. The per-swap
    /// candidate enumeration is `fair_points`, the same walk
    /// `launch_variants` and `launch_variant_labels` offer from, so a label
    /// either of them names resolves here and no other does.
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
        let mut applied = false;
        for (ix, name) in swaps {
            let Some(launch) = base.launches.get(*ix) else {
                continue;
            };
            if launch_work(graph, base, *ix) < min_macs {
                continue;
            }
            let root = launch.root;
            let class = graph.class_of(root);
            let here = base.extraction.theta.get(&root).copied();
            let Some((member, theta, _)) = fair_points(graph, class, here, root)
                .into_iter()
                .find(|(_, _, label)| label == name)
            else {
                continue;
            };
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
                moves::Candidate::Schedule {
                    node: member,
                    theta,
                },
            );
            applied = true;
        }
        if !applied {
            return None;
        }
        self.replan(
            graph,
            roots,
            &mut ex,
            cost,
            &mut NodeCache::new(graph.len()),
        )
        .ok()
    }
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

/// One co-selection sweep. For each producer class, adopt together every
/// class that holds a selectable member reading it; keep on a strict
/// improvement in exact global cost, revert through [`moves::undo`] otherwise.
///
/// This pass reaches members the budget otherwise keeps unselected, so it
/// leans on the e-graph invariant that every member of a class computes the
/// same value. Do not weaken that guard to buy launches back.
///
/// One realization per producer class that has two or more reading classes.
/// Sweeps share `cap` with the frontier search, so the whole extraction
/// remains bounded by [`ExtractBudget`] and stays a pure function of the
/// graph.
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
    cache: &mut NodeCache,
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
    cache: &mut NodeCache,
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
        // The obligation the adoption creates, priced with it: adopting `F`
        // slot views of one joint makes that joint an `F`-consumer node, and
        // an unmaterialized joint is recomputed once per slot. The seed's
        // `{c : consumers(c) > 1}` pass ran before these consumers existed,
        // so the materialize flip is offered alongside the adoption. Both
        // states are offered and the exact global cost still decides.
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

// Dumps every launch of every extracted plan when `FUSOR_DUMP_PLAN` is set,
// so a launch count can be attributed to specific nodes.
fn probe_dump(graph: &EGraph, plan: &Plan, _ex: &Extraction, realized: &Realized, caps: &Caps) {
    if std::env::var_os("FUSOR_DUMP_PLAN").is_none() {
        return;
    }
    eprintln!(
        "PLAN nodes={} classes={} launches={} buffers={}",
        graph.len(),
        realize::classes(graph).len(),
        plan.launches.len(),
        plan.buffers.len()
    );
    for (i, l) in plan.launches.iter().enumerate() {
        let n = graph.node(l.root);
        let facts = graph.facts(l.root);
        eprintln!(
            "  L{i}: root={:?} op={} shape={:?} members={} grid={:?} block={}",
            l.root,
            op_tag(&n.op),
            facts.shape,
            l.members.len(),
            l.grid,
            l.block
        );
        for m in l.members.iter() {
            eprintln!(
                "        member {:?} {} theta={:?} legal={} dom={:?} class_members={:?}",
                m,
                op_tag(&graph.node(*m).op),
                _ex.theta.get(m),
                realize::has_legal_point(graph, *m, caps),
                realize::domain_of(graph, *m).map(|d| d.len()),
                graph.members(graph.class_of(*m))
            );
        }
    }
    let _ = realized;
    // One compact line per launch: the kind, whether the body is a pure
    // identity copy, and the operand sources by node id, so the launch graph
    // can be walked offline.
    if std::env::var_os("FUSOR_DUMP_EDGES").is_some() {
        for (i, l) in plan.launches.iter().enumerate() {
            let n = graph.node(l.root);
            let facts = graph.facts(l.root);
            let ident = match &n.op {
                Op::Launch(fusor_ir::ir::launch::Launch::Map { ops, body, .. }) => {
                    ops.len() == 1
                        && format!("{body:?}").starts_with("ScalarExpr(ScalarNode { kind: Arg(0)")
                }
                _ => false,
            };
            let kind = match &n.op {
                Op::Launch(fusor_ir::ir::launch::Launch::Map { .. }) => "Map",
                Op::Launch(fusor_ir::ir::launch::Launch::Fold { .. }) => "Fold",
                Op::Launch(fusor_ir::ir::launch::Launch::Contract { .. }) => "Contract",
                Op::Launch(fusor_ir::ir::launch::Launch::Gather { .. }) => "Gather",
                Op::Launch(fusor_ir::ir::launch::Launch::Scatter { .. }) => "Scatter",
                Op::Launch(fusor_ir::ir::launch::Launch::Region { .. }) => "Region",
                Op::Launch(fusor_ir::ir::launch::Launch::Ext { .. }) => "Ext",
                Op::Logical(_) => "Logical",
                Op::Union(_, _) => "Union",
            };
            let mut srcs: Vec<u32> = Vec::new();
            for m in l.members.iter() {
                for c in fusor_ir::semantics::children::children_of(&graph.node(*m).op) {
                    srcs.push(c.0);
                }
            }
            let srcs: Vec<u32> = srcs
                .into_iter()
                .map(|s| graph.class_of(fusor_ir::egraph::Id(s)).0.0)
                .collect();
            eprintln!(
                "EDGE {i} kind={kind} ident={ident} class={} shape={:?} srcs={:?}",
                graph.class_of(l.root).0.0,
                facts.shape,
                srcs
            );
        }
    }
    if std::env::var_os("FUSOR_DUMP_CLASSES").is_none() {
        return;
    }
    for c in realize::classes(graph) {
        let members: Vec<Id> = graph.members(c);
        if members.len() < 2 {
            continue;
        }
        eprintln!("  CLASS {c:?} sel={:?}", _ex.sigma.get(&c));
        for m in members {
            eprintln!("      {m:?} {}", op_tag(&graph.node(m).op));
        }
    }
}

fn op_tag(op: &Op) -> String {
    use fusor_ir::ir::launch::Launch;
    match op {
        Op::Launch(Launch::Map {
            space, ops, body, ..
        }) => {
            let srcs: Vec<String> = ops
                .iter()
                .map(|o| {
                    format!(
                        "{}{}@{:?}",
                        o.src,
                        match &o.access {
                            fusor_ir::ir::launch::AccessPlan::Alias => "",
                            fusor_ir::ir::launch::AccessPlan::Gather => ":G",
                            fusor_ir::ir::launch::AccessPlan::Pack { .. } => ":P",
                            fusor_ir::ir::launch::AccessPlan::Unflatten(_) => ":U",
                        },
                        o.layout.offset()
                    )
                })
                .collect();
            let b = format!("{body:?}");
            format!(
                "Map space={:?} ops={} srcs={:?} body={}",
                space.dims,
                ops.len(),
                srcs,
                &b[..b.len().min(120)]
            )
        }
        Op::Launch(Launch::Fold {
            space,
            axis,
            vec_axes,
            carrier,
            post,
            ops,
            ..
        }) => format!(
            "Fold space={:?} axis={axis} vec={vec_axes:?} slots={} post={} ops={}",
            space.dims,
            carrier.slots.len(),
            post.len(),
            ops.len()
        ),
        Op::Launch(Launch::Contract { m, n, k, batch, .. }) => {
            format!("Contract m={m:?} n={n:?} k={k:?} b={batch:?}")
        }
        other => format!("{other:?}").chars().take(160).collect(),
    }
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

/// The frontier-first point of every selected schedule domain: the cheapest
/// by `node_math`, ties by domain index. The full domain stays reachable
/// through `RESCHEDULE`; this only picks where the search starts.
///
/// Fill-only: a point `RESCHEDULE` already chose is kept, so this is safe to
/// re-run after the search. A `theta` that is *not* a member of its node's
/// domain is replaced — that only happens when the node was never scheduled,
/// because every `RESCHEDULE` candidate comes out of the domain itself.
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
    while let Some(v) = realize::selection_cycle(graph, ex, roots) {
        let class = graph.class_of(v);
        let out = banned.entry(class).or_default();
        out.insert(v);
        let Some(next) =
            crate::lower_bound::argmin_member_excluding(graph, lb, launches, class, caps, out)
        else {
            return Err(Error::Plan(format!(
                "selection is cyclic through {v} and class {} has no acyclic member: \
                 every candidate names a class that names it back",
                class.0
            )));
        };
        ex.sigma.insert(class, next);
        repaired = true;
    }
    Ok(repaired)
}

fn seed_theta(graph: &EGraph, ex: &mut Extraction, cost: &dyn CostModel) -> bool {
    let mut trail = RepairTrail::default();
    seed_theta_trailed(graph, ex, cost, &mut trail);
    !trail.theta.is_empty()
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
        let Op::Launch(l1) = &node.op else { continue };
        let Some(domain) = l1.schedule() else {
            continue;
        };
        if matches!(domain, ScheduleDomain::Point) {
            let prev = ex.theta.insert(id, fusor_ir::ir::launch::SchedPoint::Point);
            if prev != Some(fusor_ir::ir::launch::SchedPoint::Point) {
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
        let ins: SmallVec<[ValueFacts; 4]> = node
            .children
            .iter()
            .map(|c| graph.facts(*c).clone())
            .collect();
        let out = graph.facts(id);
        // Only points this device can actually run: `has_legal_point` gates
        // the node, not the point, so without this clause an over-footprint
        // point can win the seed and nothing is obliged to move off it.
        // If nothing is legal, fall back the same way `legal_members` did and
        // let `verify_plan` name it precisely rather than leaving `theta`
        // unset.
        let mut best: Option<(Picoseconds, usize, _)> = None;
        let any_legal = domain
            .iter()
            .any(|t| realize::point_is_legal(graph, id, t, caps));
        for (i, theta) in domain.iter().enumerate() {
            if any_legal && !realize::point_is_legal(graph, id, theta, caps) {
                continue;
            }
            let s = cost.node_math(node, &ins, out, Some(theta));
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
/// sequence of independent moves does not preserve: a root and an in-place
/// node land in a buffer, a producer cut from a consumer by structure lands
/// in a buffer, and every selected schedule domain has a point in it. These
/// are `verify_plan`'s clauses 3, 5 and 6.
///
/// One pass is a fixpoint: `order`, `consumers` and every index space are
/// functions of `sigma` alone, so materializing a node can never create a new
/// obligation.
///
/// Returns whether anything changed, in which case the caller re-realizes and
/// re-prices.
fn repair(graph: &EGraph, ex: &mut Extraction, realized: &Realized, cost: &dyn CostModel) -> bool {
    let mut changed = false;
    for r in &realized.roots {
        if !ex.is_materialized(*r) {
            materialize(graph, ex, *r);
            changed |= ex.is_materialized(*r);
        }
    }
    for v in &realized.order {
        if realize::leaf_role(graph, *v) != realize::LeafRole::NotLeaf || ex.is_materialized(*v) {
            continue;
        }
        let in_place = graph.semantics().effect(&graph.node(*v).op) != Effect::Pure;
        if in_place || moves::at_structural_boundary(graph, realized, *v) {
            materialize(graph, ex, *v);
            changed |= ex.is_materialized(*v);
        }
    }
    changed | seed_theta(graph, ex, cost)
}

/// Realize, [`repair_trailed`], re-realize. See [`LocalSearch::price`].
fn price(
    graph: &EGraph,
    roots: &[Id],
    ex: &mut Extraction,
    cost: &dyn CostModel,
    arena: &dyn ArenaPlanner,
    cache: &mut NodeCache,
) -> std::result::Result<(Realized, Picoseconds, RepairTrail), RepairTrail> {
    let first = realize::realize_with(graph, roots, ex, cost, arena, cache)
        .map_err(|_| RepairTrail::default())?;
    let trail = repair_trailed(graph, ex, &first, cost);
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
///
/// A rejected candidate has to revert the obligations its move implied as
/// well as the move itself.
#[derive(Clone, Debug, Default)]
struct RepairTrail {
    /// Nodes newly inserted into `m`. Only nodes that were absent before, so
    /// reverting is an unconditional clear.
    m: SmallVec<[Id; 8]>,
    /// `theta` entries written, each with the value it replaced.
    theta: SmallVec<[(Id, Option<fusor_ir::ir::launch::SchedPoint>); 8]>,
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
fn repair_trailed(
    graph: &EGraph,
    ex: &mut Extraction,
    realized: &Realized,
    cost: &dyn CostModel,
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
        if in_place || moves::at_structural_boundary(graph, realized, *v) {
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
/// `splits` and `staging` are pinned to the domain's first entry — always
/// `1` and `1`.
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
        ScheduleDomain::Sgemv(_) => {
            // The sgemv domain is seed-ordered, one cell of each structure in
            // the prefix: multi-column window-16, multi-column window-32,
            // whole-workgroup-per-element. The front of the domain plus one
            // deep probe is the whole spread.
            let n = domain.len();
            for i in (0..5).chain([n / 2]) {
                if let Some(p) = domain.point(i)
                    && !out.contains(&p)
                {
                    out.push(p);
                }
            }
        }
        other => {
            // Two points off a non-coop family: enough to tell the family
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
    for b in &launch.bindings {
        total = total.saturating_add(
            crate::realize::bytes_of(graph.facts(b.value)).saturating_mul(BYTE_WEIGHT),
        );
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
    let tag = match &graph.node(root).op {
        Op::Launch(l1) => format!("{:?}", l1.tag()),
        Op::Logical(_) => "Logical".to_string(),
        Op::Union(..) => "U".to_string(),
    };
    let extra = match &graph.node(root).op {
        Op::Launch(Launch::Contract { m, n, k, batch, .. }) => {
            let c = |d: &Dim| d.as_const().map_or(0, |v| v);
            format!("mnkb={},{},{},{}", c(m), c(n), c(k), c(batch))
        }
        // `space` is the *iteration* domain and carries the reduced extent;
        // the output shape above does not.
        Op::Launch(Launch::Fold {
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
    use rustc_hash::FxHasher;
    use std::hash::{Hash, Hasher};

    fn operand(o: &Operand, h: &mut FxHasher) {
        o.layout.hash(h);
        o.access.hash(h);
    }

    let mut h = FxHasher::default();
    op.tag().hash(&mut h);
    match op {
        Op::Launch(Launch::Map {
            space, body, ops, ..
        }) => {
            space.dims.hash(&mut h);
            body.structural_hash().hash(&mut h);
            for o in ops {
                operand(o, &mut h);
            }
        }
        Op::Launch(Launch::Fold {
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
        Op::Launch(Launch::Contract {
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
        Op::Launch(Launch::Gather {
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
        Op::Launch(Launch::Scatter {
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
        Op::Launch(Launch::Region { live_outs, .. }) => live_outs.hash(&mut h),
        Op::Launch(Launch::Ext { .. }) | Op::Logical(_) | Op::Union(..) => {}
    }
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
        None => {
            let tag = match &graph.node(root).op {
                Op::Launch(l1) => format!("{:?}", l1.tag()),
                _ => "?".to_string(),
            };
            format!("{tag}|base")
        }
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
) -> Vec<(Id, SchedPoint, String)> {
    let per_member: Vec<(Id, SmallVec<[SchedPoint; 8]>)> = graph
        .members(class)
        .into_iter()
        .filter_map(|member| {
            let Op::Launch(l1) = &graph.node(member).op else {
                return None;
            };
            let domain = l1.schedule()?;
            Some((member, sample_points(domain)))
        })
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
            if !offered.insert(label.clone()) {
                continue;
            }
            fair.push((*member, theta, label));
        }
    }
    fair
}

fn variant_signature(graph: &EGraph, member: Id, theta: SchedPoint) -> String {
    let tag = match &graph.node(member).op {
        Op::Launch(l1) => format!("{:?}", l1.tag()),
        _ => "?".to_string(),
    };
    let mut q = String::new();
    for child in fusor_ir::semantics::children::children_of(&graph.node(member).op) {
        if !matches!(graph.facts(child).dtype, fusor_ir::dtype::Dtype::Q(_)) {
            continue;
        }
        let layout = graph
            .class_ids(graph.class_of(child))
            .into_iter()
            .find_map(|m| match &graph.node(m).op {
                Op::Logical(fusor_ir::ir::logical::Logical::Leaf(
                    fusor_ir::ir::logical::LeafKind::Quantized { layout, .. },
                )) => Some(*layout),
                _ => None,
            });
        if let Some(layout) = layout {
            q.push_str(&format!("|q={layout:?}"));
        }
    }
    format!("{tag}{q}|{theta:?}")
}
