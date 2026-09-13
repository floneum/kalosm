use crate::analysis::Analysis;
use crate::graph::{Graph, Id, Op};
use crate::storage::{Allocation, Packing, Slot, allocate};
pub const BLOCK: usize = 64;
pub fn scratch(id: Id) -> Id {
    usize::MAX - id
}
#[derive(Clone, Debug)]
pub struct Config {
    pub use_subgroups: bool,
    pub subgroup_width: Option<u32>,
    pub audit_indices: bool,
    pub legacy: bool,
    pub serial_limit: usize,
    pub forwarding: bool,
    pub shared_bytes: usize,
    pub global_bytes: usize,
    pub packing: Packing,
    pub fuse: bool,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            use_subgroups: true,
            subgroup_width: None,
            audit_indices: false,
            legacy: false,
            serial_limit: 32,
            forwarding: true,
            shared_bytes: 2048,
            global_bytes: usize::MAX,
            packing: Packing::BestFit,
            fuse: true,
        }
    }
}
impl Config {
    pub fn for_device(&self, gpu: &fusor_gpu::GpuDevice) -> Self {
        let mut out = self.clone();
        out.subgroup_width = if self.use_subgroups && !self.legacy {
            gpu.caps()
                .subgroups
                .filter(|s| s.is_fixed())
                .map(|s| s.assumed())
        } else {
            None
        };
        out
    }
    fn collective(&self) -> Option<fusor_gpu::reduction::CollectivePlan> {
        self.subgroup_width
            .and_then(|w| fusor_gpu::reduction::CollectivePlan::new(BLOCK as u32, w))
    }
}
#[derive(Clone, Debug)]
pub struct Region {
    pub members: Vec<Id>,
    pub groups: usize,
    pub shared: Allocation,
    pub exports: Vec<Id>,
    pub traffic_bytes: usize,
    pub score_ns: f64,
}
#[derive(Clone, Debug)]
pub struct Plan {
    pub collective: Option<fusor_gpu::reduction::CollectivePlan>,
    pub index_recipes: usize,
    pub legacy: bool,
    pub serial_limit: usize,
    pub forwarded: Vec<Id>,
    pub regions: Vec<Region>,
    pub global: Allocation,
    pub score_ns: f64,
    pub partitions: usize,
    pub ownership_rejects: usize,
    pub capacity_rejects: usize,
}
pub fn tree_fold(g: &Graph, id: Id, serial_limit: usize) -> bool {
    matches!(g.values[id].op, Op::Reduce(_, x, a) if g.values[x].shape[a] > serial_limit)
}
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn region(
    g: &Graph,
    members: &[Id],
    cfg: &Config,
    rejected: &mut [usize; 2],
    forwarded: &[Id],
    analysis: &Analysis,
) -> Option<Region> {
    let stages: Vec<Id> = g
        .stages()
        .into_iter()
        .filter(|id| !forwarded.contains(id))
        .collect();
    let divisor = members.iter().fold(0, |a, id| gcd(a, g.values[*id].len()));
    let exports: Vec<Id> = members
        .iter()
        .copied()
        .filter(|id| {
            g.roots.iter().any(|r| g.base(*r) == *id)
                || stages
                    .iter()
                    .any(|s| !members.contains(s) && analysis.dependencies[*s].contains(id))
        })
        .collect();
    let mut best: Option<Region> = None;
    let reads = analysis.external_reads(g, members);
    let writes: usize = exports.iter().map(|id| g.values[*id].len()).sum();
    let traffic_bytes = (reads + writes) * 4;
    for groups in (1..=divisor.min(65535)).filter(|x| divisor % x == 0) {
        if !analysis.owned(g, members, groups) {
            rejected[0] += 1;
            continue;
        }
        let mut slots = vec![];
        for (first, id) in members.iter().enumerate() {
            let last = members
                .iter()
                .enumerate()
                .filter(|(_, s)| analysis.dependencies[**s].contains(id))
                .map(|(i, _)| i)
                .max();
            if let Some(last) = last {
                slots.push(Slot {
                    id: *id,
                    len: g.values[*id].len() / groups,
                    first,
                    last,
                });
            }
            if tree_fold(g, *id, cfg.serial_limit) {
                let len = cfg
                    .collective()
                    .map(|p| p.scratch_elements() as usize)
                    .unwrap_or(BLOCK);
                if len > 0 {
                    slots.push(Slot {
                        id: scratch(*id),
                        len,
                        first,
                        last: first,
                    });
                }
            }
        }
        let shared = allocate(slots, cfg.packing);
        if shared.len * 4 > cfg.shared_bytes {
            rejected[1] += 1;
            continue;
        }
        let mut work = 0.0;
        for id in members {
            let share = g.values[*id].len() / groups;
            work += match g.values[*id].op {
                Op::Reduce(_, x, axis) if tree_fold(g, *id, cfg.serial_limit) => {
                    share as f64
                        * ((g.values[x].shape[axis].div_ceil(BLOCK) * 4) as f64
                            + if cfg.legacy { 240.0 } else { 2000.0 })
                }
                Op::Reduce(_, x, axis) => {
                    (share.div_ceil(BLOCK) * g.values[x].shape[axis] * 4) as f64 + 30.0
                }
                _ => (share.div_ceil(BLOCK) * 6) as f64 + 30.0,
            };
        }
        // An explicit illustrative prior, not calibrated timing. Launch,
        // memory traffic and workgroup waves all matter; minimum bytes alone
        // would collapse every graph onto one serial workgroup.
        let score_ns = (if cfg.legacy { 10_000.0 } else { 3000.0 })
            + traffic_bytes as f64 / 100.0
            + work * groups.div_ceil(if cfg.legacy { 32 } else { 256 }) as f64;
        let candidate = Region {
            members: members.to_vec(),
            groups,
            shared,
            exports: exports.clone(),
            traffic_bytes,
            score_ns,
        };
        if best.as_ref().is_none_or(|b| {
            (score_ns, candidate.shared.len, groups) < (b.score_ns, b.shared.len, b.groups)
        }) {
            best = Some(candidate);
        }
    }
    best
}
fn global(g: &Graph, regions: &[Region], packing: Packing, forwarded: &[Id]) -> Allocation {
    let mut slots = vec![];
    for (first, r) in regions.iter().enumerate() {
        for id in &r.exports {
            let root = g.roots.iter().any(|x| g.base(*x) == *id);
            let last = if root {
                regions.len()
            } else {
                regions
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| {
                        r.members
                            .iter()
                            .any(|s| g.stage_dependencies(*s, forwarded).contains(id))
                    })
                    .map(|(i, _)| i)
                    .max()
                    .unwrap_or(first)
            };
            slots.push(Slot {
                id: *id,
                len: g.values[*id].len(),
                first,
                last,
            });
        }
    }
    allocate(slots, packing)
}
pub fn compile(g: &Graph, cfg: &Config) -> Result<Plan, String> {
    let forwarded = if cfg.forwarding {
        g.forwardable()
    } else {
        vec![]
    };
    let stages: Vec<Id> = g
        .stages()
        .into_iter()
        .filter(|id| !forwarded.contains(id))
        .collect();
    let n = stages.len();
    let analysis = Analysis::new(g, &stages, &forwarded, cfg.audit_indices);
    assert!(
        n > 0 && n <= 12,
        "prototype enumerates at most 12 compute stages"
    );
    let mut rejected = [0, 0];
    let mut cache = vec![vec![None; n + 1]; n];
    for start in 0..n {
        for end in start + 1..=n {
            if cfg.fuse || end == start + 1 {
                cache[start][end] = region(
                    g,
                    &stages[start..end],
                    cfg,
                    &mut rejected,
                    &forwarded,
                    &analysis,
                );
            }
        }
    }
    let mut best: Option<Plan> = None;
    let mut partitions = 0;
    for cuts in 0..1usize << (n - 1) {
        if !cfg.fuse && cuts != (1 << (n - 1)) - 1 {
            continue;
        }
        let mut regions = vec![];
        let mut start = 0;
        let mut legal = true;
        for end in 1..=n {
            if end == n || cuts & (1 << (end - 1)) != 0 {
                if let Some(r) = &cache[start][end] {
                    regions.push(r.clone());
                } else {
                    legal = false;
                    break;
                }
                start = end;
            }
        }
        if !legal {
            continue;
        }
        partitions += 1;
        let global = global(g, &regions, cfg.packing, &forwarded);
        if global.len * 4 > cfg.global_bytes {
            continue;
        }
        let score_ns: f64 = regions.iter().map(|r| r.score_ns).sum();
        if best
            .as_ref()
            .is_none_or(|b| (score_ns, global.len) < (b.score_ns, b.global.len))
        {
            best = Some(Plan {
                collective: cfg.collective(),
                index_recipes: analysis.recipes,
                legacy: cfg.legacy,
                serial_limit: cfg.serial_limit,
                forwarded: forwarded.clone(),
                regions,
                global,
                score_ns,
                partitions: 0,
                ownership_rejects: 0,
                capacity_rejects: 0,
            });
        }
    }
    let mut best = best.ok_or("no plan fits this storage budget")?;
    best.partitions = partitions;
    best.ownership_rejects = rejected[0];
    best.capacity_rejects = rejected[1];
    Ok(best)
}
