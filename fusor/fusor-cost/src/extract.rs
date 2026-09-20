//! Deterministic extraction over class selections and schedule points.
//!
//! Lower bounds seed the selection. Single and compound moves compete on
//! the exact cost of the completed dispatch DAG. Composite ownership fixes
//! buffer obligations before either costing or plan construction.

use crate::lower_bound::argmin_member;
use crate::moves::{self, SchedCache, Trail};
use crate::plan::derive_plan;
use crate::realize::{self, NodeCache, Realized};
use fixedbitset::FixedBitSet;
use fusor_ir::Result;
use fusor_ir::cost::{CostModel, Picoseconds};
use fusor_ir::device::Caps;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::error::Error;
use fusor_ir::extract::{Dispatch, ExtractBudget, Extraction, Extractor, Plan};
use fusor_ir::facts::ValueFacts;
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
        self.seed_realized(graph, roots, lb, &launches, cost, &classes, &mut cache)
            .map(|(ex, _)| ex)
    }

    /// Construct the seed's required buffers before partitioning its launches.
    #[allow(clippy::too_many_arguments)]
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
            let pick = argmin_member(graph, lb, launches, *class, &self.caps);
            sigma_debug(*class, pick, "seed");
            ex.sigma.insert(*class, pick);
        }
        loop {
            let attempt =
                pin_selection(graph, roots, &mut ex, &mut Trail::default()).and_then(|selected| {
                    seed_theta(graph, &mut ex, cost);
                    ex.m = selected.buffers(graph);
                    selected.realize(graph, &ex, cost, self.arena.as_ref(), cache)
                });
            match attempt {
                Ok(realized) => return Ok((ex, realized)),
                Err(error) => {
                    if !break_selection_cycles(graph, roots, &mut ex, lb, launches, &self.caps)? {
                        return Err(error);
                    }
                }
            }
        }
    }

    /// Extract a plan and report the deterministic search work.
    pub fn extract_traced(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
    ) -> Result<(Plan, SearchTrace)> {
        let started = Instant::now();
        // Everything below is scoped to the classes this resolve's roots
        // reach; a long-lived session graph holds every value it ever built.
        let (classes, mask) = realize::reachable(graph, roots);
        let crate::lower_bound::Bounds {
            costs: lb,
            launches,
        } = crate::lower_bound::bounds_scoped(graph, Some(cost), &mask);
        let mut cache = NodeCache::new(graph.len());
        let (mut ex, seeded) =
            self.seed_realized(graph, roots, &lb, &launches, cost, &classes, &mut cache)?;
        // The seed is priced the same way every candidate below is: as the
        // plan it denotes, not as the state the seeding pass left.
        let mut trail = Trail::default();
        let (mut realized, mut best_cost) = match price(
            graph,
            roots,
            &mut ex,
            cost,
            self.arena.as_ref(),
            &mut cache,
            &mut trail,
        ) {
            Some(priced) => priced,
            None => {
                trail.rollback(&mut ex, 0);
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

        // Adopting all views of a multi-slot reduction can improve the plan
        // even when adopting one view alone cannot. Descend from both that
        // joint adoption and the plain seed, keeping the cheaper result.
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
                for mv in moves::frontier(graph, ex, &classes) {
                    if *moves >= cap {
                        break 'search;
                    }
                    let options = moves::candidates(graph, ex, mv, &lb, &mut sched, cost);
                    for candidate in options {
                        if *moves >= cap {
                            break 'search;
                        }
                        *moves += 1;
                        let mut trail = Trail::default();
                        if !trail.apply(ex, candidate) {
                            continue;
                        }
                        let attempt = price(
                            graph,
                            roots,
                            ex,
                            cost,
                            self.arena.as_ref(),
                            cache,
                            &mut trail,
                        );
                        match attempt {
                            // Strict improvements only; a tie keeps the earlier
                            // (smaller-id) state, which keeps the search
                            // reproducible.
                            Some((r, c)) if c < *best_cost => {
                                *best_cost = c;
                                *realized = r;
                                best.push(c);
                                improved = true;
                                break;
                            }
                            // The move is undone after the obligations it
                            // implied, so a rejected candidate leaves no trace.
                            _ => trail.rollback(ex, 0),
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
        // Dump the completed plan for diagnostics.
        probe_dump(graph, &plan, &ex, &realized, &self.caps, cost);
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
        let mut trail = Trail::default();
        let (realized, exact) = match price(
            graph,
            roots,
            ex,
            cost,
            self.arena.as_ref(),
            cache,
            &mut trail,
        ) {
            Some(priced) => priced,
            None => {
                trail.rollback(ex, 0);
                return Err(Error::Plan("autotune candidate does not realize".into()));
            }
        };
        let plan = match derive_plan(graph, ex, &realized, cost.facts(), exact) {
            Ok(plan) => plan,
            Err(error) => {
                trail.rollback(ex, 0);
                return Err(error);
            }
        };
        // Conformance checks the plan independently of the constructor.
        #[cfg(feature = "compiler-tests")]
        crate::verify_plan::verify_plan_with(graph, &plan, self.arena.as_ref(), &self.caps)
            .unwrap_or_else(|e| panic!("constructed plan violates a compiler invariant: {e}"));
        Ok(plan)
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
        let fair = fair_points_with(
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
        // Names every candidate this sweep drops and why.
        let dbg = std::env::var_os("FUSOR_TUNE_DEBUG").is_some();
        {
            for (member, theta, label) in fair {
                if out.len() >= limit {
                    if dbg {
                        eprintln!("[vdbg] L{launch_ix} cap reached at {}", out.len());
                    }
                    return out;
                }
                let mut ex = base.extraction.clone();
                let mut trail = Trail::default();
                if member != root
                    && !trail.apply(
                        &mut ex,
                        moves::Candidate::Select {
                            class,
                            node: member,
                        },
                    )
                {
                    if dbg {
                        eprintln!("[vdbg] L{launch_ix} SELECT-FAIL {member:?} {label}",);
                    }
                    continue;
                }
                trail.apply(
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
    /// onto one cloned extraction, and construct the plan once. The per-swap
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
        let mut trail = Trail::default();
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
                && !trail.apply(
                    &mut ex,
                    moves::Candidate::Select {
                        class,
                        node: member,
                    },
                )
            {
                continue;
            }
            trail.apply(
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
/// improvement in exact global cost, revert the trial otherwise.
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
    for p in producers {
        if *moves >= cap {
            break;
        }
        // The smallest-id member of each reading class that is not already
        // the selected one. `readers[p]` is sorted, so the first entry per
        // class is that member.
        let mut proposal: Vec<(ClassId, Id)> = Vec::new();
        for (c, m) in &readers[p] {
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
        let mut trail = Trail::default();
        for (class, node) in &proposal {
            trail.apply(
                ex,
                moves::Candidate::Select {
                    class: *class,
                    node: *node,
                },
            );
        }
        if trail.mark() == 0 {
            continue;
        }
        *moves += 1;
        match price(graph, roots, ex, cost, arena, cache, &mut trail) {
            Some((r, c)) if c < *best_cost => {
                *best_cost = c;
                *realized = r;
                best.push(c);
                improved = true;
            }
            _ => trail.rollback(ex, 0),
        }
    }
    Ok(improved)
}

// Dumps every launch of every extracted plan when `FUSOR_DUMP_PLAN` is set,
// so a launch count can be attributed to specific nodes.
fn probe_dump(
    graph: &EGraph,
    plan: &Plan,
    _ex: &Extraction,
    realized: &Realized,
    caps: &Caps,
    cost: &dyn CostModel,
) {
    if std::env::var_os("FUSOR_DUMP_PLAN").is_none() {
        return;
    }
    let priced = realized.launches(_ex);
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
        let priced_line = priced
            .iter()
            .find(|p| p.root == l.root)
            .map(|p| {
                format!(
                    "cost_us={:.1} reads={:?} writes={} line_bytes={} lanes={}",
                    cost.launch_cost(p).0 as f64 / 1e6,
                    p.reads,
                    p.writes,
                    p.line_bytes,
                    p.resident_lanes
                )
            })
            .unwrap_or_default();
        eprintln!(
            "  L{i}: root={:?} class={} op={} shape={:?} members={} grid={:?} block={} {priced_line}",
            l.root,
            graph.class_of(l.root).0.index(),
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
                realize::composite_bindings_fit(graph, *m, caps),
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
                Op::Launch(fusor_ir::ir::launch::Launch::Slab { .. }) => "Slab",
                Op::Launch(fusor_ir::ir::launch::Launch::Group { .. }) => "Group",
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

pub(crate) fn op_tag(op: &Op) -> String {
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
        if std::env::var_os("FUSOR_CYCLE_LOG").is_some() {
            seen_cycles += 1;
            let show = |i: Id| {
                format!("{:?}", graph.node(i).op)
                    .chars()
                    .take(120)
                    .collect::<String>()
            };
            let kids: Vec<String> = graph
                .node(v)
                .children
                .iter()
                .map(|c| {
                    let cc = graph.class_of(*c);
                    format!("{c}:c{}->{:?}", cc.0.index(), ex.sigma.get(&cc))
                })
                .collect();
            eprintln!(
                "CYCLE {seen_cycles} at {v} (class {}) {}\n   kids {kids:?}",
                class.0.index(),
                show(v)
            );
            // The path back to `v` under the current selection.
            let mut stack: Vec<(Id, Vec<Id>)> = vec![(v, vec![v])];
            let mut seen: FxHashSet<Id> = FxHashSet::default();
            let mut found: Option<Vec<Id>> = None;
            while let Some((x, path)) = stack.pop() {
                let by_id = matches!(
                    graph.node(x).op,
                    Op::Launch(Launch::Slab { .. } | Launch::Group { .. })
                );
                for c in graph.node(x).children.iter() {
                    let n = if by_id {
                        *c
                    } else {
                        ex.selected(graph.class_of(*c)).unwrap_or(*c)
                    };
                    if n == v {
                        let mut p = path.clone();
                        p.push(n);
                        found = Some(p);
                        break;
                    }
                    if seen.insert(n) {
                        let mut p = path.clone();
                        p.push(n);
                        stack.push((n, p));
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            if let Some(path) = found {
                for n in path {
                    eprintln!(
                        "     {n} (class {}) {}",
                        graph.class_of(n).0.index(),
                        show(n)
                    );
                }
            }
            if seen_cycles > 8 {
                panic!("FUSOR_CYCLE_LOG: stopping after {seen_cycles} cycles");
            }
        }
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
        sigma_debug(class, next, "break_selection_cycles");
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
        let by_id = matches!(
            graph.node(x).op,
            Op::Launch(Launch::Slab { .. } | Launch::Group { .. })
        );
        for c in graph.node(x).children.iter() {
            let n = if by_id {
                *c
            } else {
                ex.selected(graph.class_of(*c)).unwrap_or(*c)
            };
            if matches!(graph.node(n).op, Op::Launch(Launch::Group { .. })) {
                return Some((graph.class_of(n), n));
            }
            if seen.insert(n) {
                stack.push(n);
            }
        }
    }
    None
}

fn seed_theta(graph: &EGraph, ex: &mut Extraction, cost: &dyn CostModel) -> bool {
    let mut trail = Trail::default();
    seed_theta_trailed(graph, ex, cost, &mut trail);
    trail.mark() != 0
}

/// Fill missing schedules and record them for rollback.
fn seed_theta_trailed(
    graph: &EGraph,
    ex: &mut Extraction,
    cost: &dyn CostModel,
    trail: &mut Trail,
) {
    let mut scheduled: FxHashSet<Id> = ex.sigma.values().copied().collect();
    let mut pending: Vec<Id> = scheduled.iter().copied().collect();
    // Composites execute their members by id. Their final member shares the
    // composite's class and therefore has no separate sigma entry.
    while let Some(id) = pending.pop() {
        if let Op::Launch(Launch::Slab { members, .. } | Launch::Group { members, .. }) =
            &graph.node(id).op
        {
            for member in members {
                if scheduled.insert(*member) {
                    pending.push(*member);
                }
            }
        }
    }
    let mut selected: Vec<Id> = scheduled.into_iter().collect();
    selected.sort_unstable();
    for id in selected {
        if ex.theta.contains_key(&id) {
            continue;
        }
        let node = graph.node(id);
        let Op::Launch(l1) = &node.op else { continue };
        let Some(domain) = l1.schedule() else {
            continue;
        };
        if matches!(domain, ScheduleDomain::Point) {
            trail.schedule(ex, id, SchedPoint::Point);
            continue;
        }
        let ins: SmallVec<[ValueFacts; 4]> = node
            .children
            .iter()
            .map(|c| graph.facts(*c).clone())
            .collect();
        let out = graph.facts(id);
        if let Some(theta) = domain
            .iter()
            .min_by_key(|theta| cost.node_math(node, &ins, out, Some(*theta)))
        {
            trail.schedule(ex, id, theta);
        }
    }
}

/// `FUSOR_SIGMA_DEBUG=<class id>`: prints every selection change of that
/// class with the site that made it.
pub(crate) fn sigma_debug(class: ClassId, node: Id, site: &str) {
    thread_local! { static WANT: Option<u32> = std::env::var("FUSOR_SIGMA_DEBUG").ok().and_then(|v| v.parse().ok()); }
    if WANT.with(|w| *w == Some(class.0.index() as u32)) {
        eprintln!("[sigma] class {} <- {node} ({site})", class.0.index());
    }
}

fn price(
    graph: &EGraph,
    roots: &[Id],
    ex: &mut Extraction,
    cost: &dyn CostModel,
    arena: &dyn ArenaPlanner,
    cache: &mut NodeCache,
    trail: &mut Trail,
) -> Option<(Realized, Picoseconds)> {
    let selected = pin_selection(graph, roots, ex, trail).ok()?;
    seed_theta_trailed(graph, ex, cost, trail);
    trail.buffers(ex, selected.buffers(graph));
    let realized = selected.realize(graph, ex, cost, arena, cache).ok()?;
    let c = realize::exact_cost(&realized, ex, cost);
    Some((realized, c))
}

fn pin_selection(
    graph: &EGraph,
    roots: &[Id],
    ex: &mut Extraction,
    trail: &mut Trail,
) -> Result<realize::Selected> {
    loop {
        let before = trail.mark();
        let selected = realize::Selected::new(graph, ex, roots)?;
        pin_slabs_trailed(graph, ex, &selected.order, trail);
        if !trail.selected_since(before) {
            return Ok(selected);
        }
    }
}

/// Select each composite's concrete middle members and resolve overlapping
/// ownership before constructing its dispatch and buffers.
fn pin_slabs_trailed(graph: &EGraph, ex: &mut Extraction, order: &[Id], trail: &mut Trail) {
    use fusor_ir::ir::launch::Launch as L;
    fn members_of(graph: &EGraph, id: Id) -> Option<&smallvec::SmallVec<[Id; 8]>> {
        match &graph.node(id).op {
            Op::Launch(L::Slab { members, .. } | L::Group { members, .. }) => Some(members),
            _ => None,
        }
    }
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
                let group = matches!(graph.node(*id).op, Op::Launch(L::Group { .. }));
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
    // A symbolic extent is priced at its bound value (the graph's dim hints),
    // not a nominal one: a symbolic plan's launches are as much work as a
    // concrete plan's, and deserve the same tuning.
    let hinted_facts = |f: &ValueFacts| -> ValueFacts {
        let mut f = f.clone();
        f.shape = f.shape.iter().map(|d| graph.hinted(*d)).collect();
        f
    };
    let hinted_op = |op: &Op| -> Op {
        let mut op = op.clone();
        match &mut op {
            Op::Launch(Launch::Map { space, .. }) | Op::Launch(Launch::Fold { space, .. }) => {
                space.dims = space.dims.iter().map(|d| graph.hinted(*d)).collect();
            }
            Op::Launch(Launch::Contract {
                m,
                n,
                k,
                batch,
                output,
                ..
            }) => {
                *m = graph.hinted(*m);
                *n = graph.hinted(*n);
                *k = graph.hinted(*k);
                *batch = graph.hinted(*batch);
                output.dims = output.dims.iter().map(|d| graph.hinted(*d)).collect();
            }
            _ => {}
        }
        op
    };
    let mut total: u64 = 0;
    for m in &launch.members {
        let node = graph.node(*m);
        let ins: SmallVec<[ValueFacts; 4]> = node
            .children
            .iter()
            .map(|c| hinted_facts(graph.facts(*c)))
            .collect();
        let out = hinted_facts(graph.facts(*m));
        let w = graph.semantics().work(&hinted_op(&node.op), &ins, &out);
        total = total
            .saturating_add(w.macs)
            .saturating_add(w.transcendentals.saturating_mul(TRANS_WEIGHT))
            .saturating_add(w.index_ops);
    }
    for b in &launch.bindings {
        total = total.saturating_add(
            crate::realize::bytes_of(&hinted_facts(graph.facts(b.value)))
                .saturating_mul(BYTE_WEIGHT),
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
            let c = |d: &Dim| d.as_const().unwrap_or(0);
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
    fair_points_with(graph, class, here, root, sample_points, true)
}

fn fair_points_with(
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
        .filter_map(|member| {
            let Op::Launch(l1) = &graph.node(member).op else {
                return None;
            };
            let domain = l1.schedule()?;
            Some((member, points(domain)))
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
