//! Environment-driven diagnostics, parsed once per process: plan dumps,
//! cycle repair, one class's seed and selection trace, tuning drops, and two
//! pricing features to disable for bisecting.

use crate::nodes::{is_view_copy, resolved_children};
use crate::realize::{self, Realized};
use fusor_ir::cost::{CostModel, Picoseconds};
use fusor_ir::device::Caps;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::extract::{Extraction, Plan};
use fusor_ir::ir::Op;
use fusor_ir::ir::launch::{AccessPlan, Launch};
use rustc_hash::FxHashSet;
use std::sync::OnceLock;

pub(crate) struct Flags {
    pub dump_plan: bool,
    pub dump_edges: bool,
    pub dump_classes: bool,
    pub cycle_log: bool,
    /// `FUSOR_SEED_DEBUG` is set, and the class it names.
    pub seed: bool,
    pub seed_class: Option<usize>,
    pub sigma_class: Option<usize>,
    pub tune: bool,
    /// Every slab member in a buffer.
    pub no_private: bool,
    /// `node_math` without the serial-chain floor.
    pub no_seed_floor: bool,
}

pub(crate) fn flags() -> &'static Flags {
    static FLAGS: OnceLock<Flags> = OnceLock::new();
    FLAGS.get_or_init(|| {
        let set = |name| std::env::var_os(name).is_some();
        let class = |name| std::env::var(name).ok().and_then(|v| v.parse().ok());
        Flags {
            dump_plan: set("FUSOR_DUMP_PLAN"),
            dump_edges: set("FUSOR_DUMP_EDGES"),
            dump_classes: set("FUSOR_DUMP_CLASSES"),
            cycle_log: set("FUSOR_CYCLE_LOG"),
            seed: set("FUSOR_SEED_DEBUG"),
            seed_class: class("FUSOR_SEED_DEBUG"),
            sigma_class: class("FUSOR_SIGMA_DEBUG"),
            tune: set("FUSOR_TUNE_DEBUG"),
            no_private: set("FUSOR_NO_PRIVATE"),
            no_seed_floor: set("FUSOR_NO_SEED_FLOOR"),
        }
    })
}

fn show(graph: &EGraph, id: Id, chars: usize) -> String {
    format!("{:?}", graph.node(id).op)
        .chars()
        .take(chars)
        .collect()
}

/// Every selection change of the `FUSOR_SIGMA_DEBUG` class, with its site.
pub(crate) fn sigma(class: ClassId, node: Id, site: &str) {
    if flags().sigma_class == Some(class.0.index()) {
        eprintln!("[sigma] class {} <- {node} ({site})", class.0.index());
    }
}

/// A tuning candidate the variant sweep offered or dropped.
pub(crate) fn tune(launch_ix: usize, what: impl FnOnce() -> String) {
    if flags().tune {
        eprintln!("[vdbg] L{launch_ix} {}", what());
    }
}

/// Every selectable member's seed key for the `FUSOR_SEED_DEBUG` class.
pub(crate) fn seed_members(
    graph: &EGraph,
    lb: &[Picoseconds],
    launches: &[u32],
    class: ClassId,
    caps: &Caps,
) {
    if flags().seed_class != Some(class.0.index()) {
        return;
    }
    for m in realize::selectable(graph, class, caps) {
        let show: String = format!("{:?}", graph.node(m).op)
            .replace("ScalarExpr(ScalarNode { kind: ", "")
            .chars()
            .take(220)
            .collect();
        let excess: Vec<String> = match &graph.node(m).op {
            Op::Launch(Launch::Group { members, .. }) => members
                .iter()
                .map(|x| {
                    let c = graph.class_of(*x);
                    format!(
                        "{x}:c{}:+{}us:best={:?}",
                        c.0.index(),
                        lb[x.index()].0.saturating_sub(lb[c.0.index()].0) / 1_000_000,
                        crate::lower_bound::argmin_member_excluding(
                            graph,
                            lb,
                            launches,
                            c,
                            caps,
                            &Default::default()
                        )
                    )
                })
                .collect(),
            _ => Vec::new(),
        };
        eprintln!(
            "[seed] class {} member {m:?} lb={} launches={} excess={excess:?} op={show}",
            class.0.index(),
            lb[m.index()].0,
            launches[m.index()],
        );
    }
}

pub(crate) fn seed_chosen(class: ClassId, chosen: Id) {
    if flags().seed_class == Some(class.0.index()) {
        eprintln!("[seed] class {} chose {chosen:?}", class.0.index());
    }
}

/// A node whose best math saturated, and the math of the seed class's nodes.
pub(crate) fn math_table(graph: &EGraph, ids: &[Id], math: &[Picoseconds]) {
    let f = flags();
    if !f.seed {
        return;
    }
    for id in ids {
        if math[id.index()].0 >= u64::MAX / 4 {
            eprintln!("[lb] math saturated at {id:?}: {}", show(graph, *id, 200));
        }
    }
    let Some(want) = f.seed_class else {
        return;
    };
    for id in ids {
        if graph.class_of(*id).0.index() == want {
            eprintln!(
                "[math] class {want} node {id} math={} {}",
                math[id.index()].0,
                show(graph, *id, 120)
            );
        }
    }
}

/// The selection cycle through `v` that seed repair is about to break.
pub(crate) fn cycle(graph: &EGraph, ex: &Extraction, v: Id, seen_cycles: usize) {
    if !flags().cycle_log {
        return;
    }
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
        graph.class_of(v).0.index(),
        show(graph, v, 120)
    );
    // The path back to `v` under the current selection.
    let mut stack: Vec<(Id, Vec<Id>)> = vec![(v, vec![v])];
    let mut seen: FxHashSet<Id> = FxHashSet::default();
    let mut found: Option<Vec<Id>> = None;
    'walk: while let Some((x, path)) = stack.pop() {
        for (c, n) in resolved_children(graph, ex, x) {
            let n = n.unwrap_or(c);
            if n == v || seen.insert(n) {
                let mut p = path.clone();
                p.push(n);
                if n == v {
                    found = Some(p);
                    break 'walk;
                }
                stack.push((n, p));
            }
        }
    }
    for n in found.into_iter().flatten() {
        eprintln!(
            "     {n} (class {}) {}",
            graph.class_of(n).0.index(),
            show(graph, n, 120)
        );
    }
    if seen_cycles > 8 {
        panic!("FUSOR_CYCLE_LOG: stopping after {seen_cycles} cycles");
    }
}

/// Every launch of an extracted plan, so a launch count can be attributed
/// to specific nodes; with `FUSOR_DUMP_EDGES` the launch graph, with
/// `FUSOR_DUMP_CLASSES` every multi-member class and its selection.
pub(crate) fn dump_plan(
    graph: &EGraph,
    plan: &Plan,
    ex: &Extraction,
    realized: &Realized,
    caps: &Caps,
    cost: &dyn CostModel,
) {
    let f = flags();
    if !f.dump_plan {
        return;
    }
    let priced = realized.launches(ex);
    eprintln!(
        "PLAN nodes={} classes={} launches={} buffers={}",
        graph.len(),
        realize::classes(graph).len(),
        plan.launches.len(),
        plan.buffers.len()
    );
    for (i, l) in plan.launches.iter().enumerate() {
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
            op_tag(&graph.node(l.root).op),
            graph.facts(l.root).shape,
            l.members.len(),
            l.grid,
            l.block
        );
        for m in l.members.iter() {
            eprintln!(
                "        member {:?} {} theta={:?} legal={} dom={:?} class_members={:?}",
                m,
                op_tag(&graph.node(*m).op),
                ex.theta.get(m),
                realize::composite_bindings_fit(graph, *m, caps),
                crate::nodes::domain_of(graph, *m).map(|d| d.len()),
                graph.members(graph.class_of(*m))
            );
        }
    }
    // One compact line per launch: the kind, whether the body is a pure
    // identity copy, and the operand source classes.
    if f.dump_edges {
        for (i, l) in plan.launches.iter().enumerate() {
            let op = &graph.node(l.root).op;
            // A launch root is never a union, so this is the tune-cache tag.
            let kind = crate::extract::tag_of(op);
            let srcs: Vec<u32> = l
                .members
                .iter()
                .flat_map(|m| fusor_ir::semantics::children::children_of(&graph.node(*m).op))
                .map(|c| graph.class_of(c).0.0)
                .collect();
            eprintln!(
                "EDGE {i} kind={kind} ident={} class={} shape={:?} srcs={:?}",
                is_view_copy(op),
                graph.class_of(l.root).0.0,
                graph.facts(l.root).shape,
                srcs
            );
        }
    }
    if f.dump_classes {
        for c in realize::classes(graph) {
            let members: Vec<Id> = graph.members(c);
            if members.len() < 2 {
                continue;
            }
            eprintln!("  CLASS {c:?} sel={:?}", ex.sigma.get(&c));
            for m in members {
                eprintln!("      {m:?} {}", op_tag(&graph.node(m).op));
            }
        }
    }
}

/// A short readable spelling of one node for diagnostics.
pub(crate) fn op_tag(op: &Op) -> String {
    match op {
        Op::Launch(Launch::Map {
            space, ops, body, ..
        }) => {
            let srcs: Vec<String> = ops
                .iter()
                .map(|o| {
                    let access = match &o.access {
                        AccessPlan::Alias => "",
                        AccessPlan::Gather => ":G",
                        AccessPlan::Pack { .. } => ":P",
                        AccessPlan::Unflatten(_) => ":U",
                    };
                    format!("{}{access}@{:?}", o.src, o.layout.offset())
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
