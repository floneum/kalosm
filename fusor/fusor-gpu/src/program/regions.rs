//! Form jobs with their own ownership domains, then pack independent jobs into
//! dispatches. Linear dependencies fuse only with a logical index proof. Matrix
//! jobs distribute complete tiles. A failed proof costs a dispatch boundary;
//! scheduling never introduces a cross-workgroup barrier inside a kernel.
use super::{
    index::{Bounds, Expr, GROUP, LOCAL},
    plan::Value,
};
use fusor_ir::{
    egraph::Id,
    ir::logical::{LeafKind, Logical},
    semantics::children::children_logical,
};
use rustc_hash::{FxHashMap, FxHashSet};

#[derive(Debug)]
pub(crate) struct Job {
    pub stages: Vec<Id>,
    pub groups: u32,
    pub tiled: bool,
    pub bucketed: bool,
}

#[derive(Debug)]
pub(crate) struct Region {
    pub jobs: Vec<Job>,
}
impl Region {
    pub(crate) fn stages(&self) -> impl Iterator<Item = Id> + '_ {
        self.jobs.iter().flat_map(|j| j.stages.iter().copied())
    }
}

fn reads(
    id: Id,
    at: Expr,
    values: &[Value],
    map: &FxHashMap<Id, usize>,
    out: &mut Vec<(Id, Expr)>,
) {
    let v = &values[map[&id]];
    match &v.op {
        Logical::Leaf(LeafKind::Const { .. }) => (),
        Logical::Restride { x, specs, .. } => {
            let source = &values[map[x]];
            let index = at.restride(&v.shape, &source.shape, specs);
            reads(*x, index, values, map, out);
        }
        Logical::Map { ins, .. } if v.forwarded => {
            for dep in ins {
                reads(*dep, at.clone(), values, map, out);
            }
        }
        _ => out.push((v.id, at)),
    }
}

fn recipes(
    v: &Value,
    values: &[Value],
    map: &FxHashMap<Id, usize>,
    groups: u32,
) -> (Vec<(Id, Expr)>, Bounds) {
    let share = v.len().div_ceil(groups);
    let mut bounds = Bounds::from([
        (GROUP, u64::from(groups - 1)),
        (LOCAL, u64::from(share - 1)),
    ]);
    let at = Expr::sum([Expr::var(GROUP).scale(share as usize), Expr::var(LOCAL)]);
    let mut out = vec![];
    let mut read = |id, at| reads(id, at, values, map, &mut out);
    match &v.op {
        Logical::Map { ins, .. } => {
            for dep in ins {
                read(*dep, at.clone());
            }
        }
        Logical::Fold { axis, ins, .. } => {
            let source = &values[map[&ins[0]]];
            let shape: Vec<_> = source.shape.iter().map(|n| *n as usize).collect();
            bounds.insert(0, u64::from(source.shape[*axis as usize] - 1));
            let index = super::index::reduction_index(&shape, *axis as usize, at, Expr::var(0));
            for dep in ins {
                read(*dep, index.clone());
            }
        }
        Logical::Contract { spec, a, b, .. } => {
            let mut labels = std::collections::BTreeMap::new();
            for (i, label) in spec.out.iter().enumerate() {
                labels.insert(*label, at.clone().coordinate(&v.shape, i));
            }
            let mut next = 0;
            for (dep, axes) in [(*a, &spec.a), (*b, &spec.b)] {
                let source = &values[map[&dep]];
                let index = Expr::sum(axes.iter().enumerate().map(|(axis, label)| {
                    labels
                        .entry(*label)
                        .or_insert_with(|| {
                            let var = next;
                            next += 1;
                            bounds.insert(var, u64::from(source.shape[axis] - 1));
                            Expr::var(var)
                        })
                        .clone()
                        .scale(source.shape[axis + 1..].iter().product::<u32>() as usize)
                }));
                read(dep, index);
            }
        }
        Logical::Gather { axis, x, idx } => {
            let source = &values[map[x]];
            let inner = source.shape[*axis as usize + 1..].iter().product::<u32>() as usize;
            let count = values[map[idx]].len() as usize;
            bounds.insert(0, u64::from(source.shape[*axis as usize] - 1));
            read(*idx, at.clone().div(inner).modulo(count));
            read(
                *x,
                Expr::sum([
                    at.clone()
                        .div(inner * count)
                        .scale(inner * source.shape[*axis as usize] as usize),
                    Expr::var(0).scale(inner),
                    at.modulo(inner),
                ]),
            );
        }
        // Data dependent scatter is deliberately a cut unless its inputs are
        // already global. Full input ranges are a sound over-approximation.
        _ => {
            for (var, dep) in children_logical(&v.op).into_iter().enumerate() {
                bounds.insert(var, u64::from(values[map[&dep]].len() - 1));
                read(dep, Expr::var(var));
            }
        }
    }
    (out, bounds)
}

pub(crate) fn schedule(
    values: &[Value],
    map: &FxHashMap<Id, usize>,
    stages: &[Id],
    groups: u32,
    limit: usize,
) -> Vec<Region> {
    let recipes: FxHashMap<_, _> = stages
        .iter()
        .map(|id| (*id, recipes(&values[map[id]], values, map, groups)))
        .collect();
    let mut done = FxHashSet::default();
    let all: FxHashSet<_> = stages.iter().copied().collect();
    let mut result = vec![];
    while done.len() < stages.len() {
        let mut current = Job {
            stages: vec![],
            groups,
            tiled: false,
            bucketed: false,
        };
        let mut local = FxHashSet::default();
        loop {
            let mut changed = false;
            for id in stages {
                if current.stages.len() >= limit {
                    break;
                }
                if done.contains(id) {
                    continue;
                }
                let (reads, bounds) = &recipes[id];
                if reads
                    .iter()
                    .any(|(dep, _)| all.contains(dep) && !done.contains(dep))
                {
                    continue;
                }
                let value = &values[map[id]];
                let tile_parallel = groups > 1 && matches!(value.op, Logical::Contract { .. });
                let bucket_parallel = groups > 1
                    && if let Logical::Scatter {
                        axis, idx, combine, ..
                    } = &value.op
                    {
                        let count = values[map[idx]].len();
                        *combine == fusor_ir::ir::logical::ScatterCombine::Add
                            && (64..=1024).contains(&count)
                            && value.shape[*axis as usize + 1..].iter().product::<u32>() >= 32
                    } else {
                        false
                    };
                if bucket_parallel {
                    if !current.stages.is_empty() {
                        continue;
                    }
                    let Logical::Scatter { axis, .. } = &value.op else {
                        unreachable!()
                    };
                    let inner = value.shape[*axis as usize + 1..].iter().product::<u32>();
                    current.bucketed = true;
                    current.groups = (value.len() / inner).clamp(1, 256);
                    current.stages.push(*id);
                    done.insert(*id);
                    break;
                }
                if tile_parallel {
                    if !current.stages.is_empty() {
                        continue;
                    }
                    current.tiled = true;
                    // A bounded grid walks all tiles, including noncanonical
                    // Einstein output orderings and portable/native tile tails.
                    current.groups = matrix_groups(value, false);
                    current.stages.push(*id);
                    done.insert(*id);
                    break;
                }
                let safe = groups == 1
                    || reads.iter().all(|(dep, index)| {
                        !local.contains(dep)
                            || index
                                .clone()
                                .div(values[map[dep]].len().div_ceil(groups) as usize)
                                .simplify(bounds)
                                == Expr::var(GROUP)
                    });
                if safe {
                    current.stages.push(*id);
                    local.insert(*id);
                    done.insert(*id);
                    changed = true;
                }
            }
            if !changed || current.tiled || current.bucketed {
                break;
            }
        }
        assert!(
            !current.stages.is_empty(),
            "logical stage graph must be acyclic"
        );
        result.push(current);
    }
    if groups == 1 {
        return result
            .into_iter()
            .map(|job| Region { jobs: vec![job] })
            .collect();
    }
    // Jobs carry their own ownership domains. Only dependency-independent jobs
    // share a dispatch; its boundary is the sole cross-workgroup rendezvous.
    let producer: FxHashMap<_, _> = result
        .iter()
        .enumerate()
        .flat_map(|(i, j)| j.stages.iter().map(move |id| (*id, i)))
        .collect();
    let deps: Vec<FxHashSet<usize>> = result
        .iter()
        .enumerate()
        .map(|(i, j)| {
            j.stages
                .iter()
                .flat_map(|id| recipes[id].0.iter())
                .filter_map(|(dep, _)| producer.get(dep).copied())
                .filter(|dep| *dep != i)
                .collect()
        })
        .collect();
    let mut pending: Vec<_> = result.into_iter().map(Some).collect();
    let mut done = FxHashSet::default();
    let mut regions = vec![];
    while done.len() < pending.len() {
        let mut picked = vec![];
        let mut count = 0;
        let mut groups = 0;
        for (i, job) in pending.iter().enumerate() {
            let Some(job) = job else { continue };
            if deps[i].iter().all(|d| done.contains(d))
                && count + job.stages.len() <= limit
                && groups + job.groups <= 65535
            {
                count += job.stages.len();
                groups += job.groups;
                picked.push(i);
            }
        }
        assert!(!picked.is_empty(), "job dependency graph must be acyclic");
        let jobs = picked
            .iter()
            .map(|i| {
                done.insert(*i);
                pending[*i].take().unwrap()
            })
            .collect();
        regions.push(Region { jobs });
    }
    regions
}

pub(super) fn matrix_groups(value: &Value, cooperative: bool) -> u32 {
    let Logical::Contract { spec, .. } = &value.op else {
        unreachable!()
    };
    let batch: u32 = spec
        .out
        .iter()
        .zip(&value.shape)
        .filter(|(l, _)| spec.a.contains(l) && spec.b.contains(l))
        .map(|(_, d)| *d)
        .product();
    let rows: u32 = spec
        .out
        .iter()
        .zip(&value.shape)
        .filter(|(l, _)| !spec.b.contains(l))
        .map(|(_, d)| *d)
        .product();
    let cols: u32 = spec
        .out
        .iter()
        .zip(&value.shape)
        .filter(|(l, _)| !spec.a.contains(l))
        .map(|(_, d)| *d)
        .product();
    (batch * rows.div_ceil(if cooperative { 32 } else { 16 }) * cols.div_ceil(16)).clamp(1, 256)
}
