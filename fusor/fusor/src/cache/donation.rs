//! Runtime proof for consuming a cache store during resolve-and-commit.
//! No generic tensor input is eligible. Snapshot aliases, other requested
//! roots, later readers and non-dense layouts all retain immutable copying.
use crate::graph::GraphRef;
use fusor_ir::egraph::Id;
use fusor_ir::extract::{BindKind, Plan};
use fusor_ir::ir::{Op, launch::Launch, logical::ScatterCombine};
use fusor_ir::shape::Layout;
use fusor_ir::target::Buf;
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

type Pair = (usize, Id, Id, u32);
pub(crate) struct Variant {
    original: u128,
    pairs: Vec<Pair>,
    plan: Arc<Plan>,
}

pub(crate) fn plan(
    graph: &GraphRef,
    original: &Plan,
    supplied: &mut FxHashMap<Id, Buf>,
    wanted: &[Id],
    prepared: Option<&crate::session::prepared::PreparedRun>,
) -> Option<Arc<Plan>> {
    let trace = std::env::var_os("FUSOR_PROFILE_CACHE_DONATION_TRACE").is_some();
    let start = trace.then(std::time::Instant::now);
    let g = graph.state().egraph.lock();
    let mut leaves = graph.state().leaves.lock();
    if leaves.donations.is_empty() {
        return None;
    }
    let selected = |id| original.extraction.selected(g.class_of(id)).unwrap_or(id);
    let wanted_classes: FxHashSet<_> = wanted.iter().map(|id| g.class_of(*id)).collect();
    // Only cache stores can be consumed. Building owner tables for every
    // immutable model weight repeats irrelevant work each token. Keep raw
    // addresses here: cloning a buffer would change the ownership proof.
    let candidates: Vec<_> = leaves
        .donations
        .iter()
        .filter_map(|store| {
            let base = selected(*store);
            (!wanted_classes.contains(&g.class_of(base)))
                .then(|| supplied.get(&base).map(|buf| (base, buf.addr())))
                .flatten()
        })
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let addresses: FxHashSet<_> = candidates.iter().map(|(_, address)| *address).collect();
    // One pass over owners and readers, rather than rescanning the entire
    // graph and launch list for each of a model's 64 cache stores.
    let mut owners = FxHashMap::default();
    for (id, (buf, _)) in &leaves.device {
        if !addresses.contains(&buf.addr()) {
            continue;
        }
        let class = g.class_of(*id);
        let e = owners.entry(buf.addr()).or_insert((0usize, Some(class)));
        e.0 += 1;
        if e.1 != Some(class) {
            e.1 = None;
        }
    }
    let mut supplied_counts = FxHashMap::default();
    for buf in supplied.values() {
        if addresses.contains(&buf.addr()) {
            *supplied_counts.entry(buf.addr()).or_insert(0usize) += 1;
        }
    }
    let mut last_uses_owned = FxHashMap::default();
    let last_uses = if let Some(prepared) = prepared {
        &prepared.last_uses
    } else {
        for (index, launch) in original.launches.iter().enumerate() {
            for b in &launch.bindings {
                last_uses_owned.insert(b.value, index);
            }
        }
        &last_uses_owned
    };
    let outputs_owned;
    let outputs = if let Some(prepared) = prepared {
        &prepared.outputs
    } else {
        outputs_owned = original
            .buffers
            .iter()
            .enumerate()
            .map(|(i, b)| (b.value, i))
            .collect();
        &outputs_owned
    };
    let mut pairs = Vec::new();
    for (base, address) in candidates {
        let buf = &supplied[&base];
        // Count every graph reference, and reject aliases in a different
        // e-class. Detached/materialized snapshots live in such classes.
        let Some(&(graph_refs, owner)) = owners.get(&address) else {
            continue;
        };
        let supplied_refs = supplied_counts[&address];
        // One further reference belongs to the pool. Any unaccounted owner
        // (including another graph) makes this an ordinary copying scatter.
        if owner != Some(g.class_of(base)) || buf.refcount() != graph_refs + supplied_refs + 1 {
            continue;
        }
        let Some(&last) = last_uses.get(&base) else {
            continue;
        };
        let launch = &original.launches[last];
        let Op::Launch(Launch::Scatter {
            space,
            axis,
            combine: ScatterCombine::Set,
            ops,
            ..
        }) = &g.node(launch.root).op
        else {
            continue;
        };
        if ops.len() != 3
            || selected(ops[0].src) != base
            || ops[1]
                .layout
                .shape()
                .iter()
                .try_fold(1u64, |n, d| n.checked_mul(d.as_const()?))
                != Some(1)
            || ops[0].layout != Layout::contiguous(&space.dims)
            || g.facts(base).shape.as_slice() != space.dims.as_slice()
            || *axis as usize >= space.dims.len()
            || ops[1..].iter().any(|o| selected(o.src) == base)
        {
            continue;
        }
        let Some(&output_index) = outputs.get(&launch.root) else {
            continue;
        };
        let output = &original.buffers[output_index];
        if output.arena.is_some()
            || output.layout != Layout::contiguous(&space.dims)
            || !wanted.contains(&launch.root)
        {
            continue;
        }
        let Some(base_binding) = launch.bindings.iter().find(|b| b.value == base) else {
            continue;
        };
        let Some(out_binding) = launch.bindings.iter().find(|b| b.value == launch.root) else {
            continue;
        };
        if base_binding.arena || out_binding.arena {
            continue;
        }
        pairs.push((last, base, launch.root, out_binding.binding));
    }
    drop(g);
    if pairs.is_empty() {
        return None;
    }
    for &(_, base, output, _) in &pairs {
        supplied.insert(output, supplied[&base].clone());
    }
    if let Some(cached) = leaves
        .donation_plans
        .iter()
        .find(|v| v.original == original.hash.0 && v.pairs == pairs)
    {
        if let Some(start) = start {
            eprintln!(
                "CACHE_DONATION stores={} aliases={} proof_us={} cached=true",
                leaves.donations.len(),
                pairs.len(),
                start.elapsed().as_micros()
            );
        }
        return Some(cached.plan.clone());
    }
    let mut plan = original.clone();
    for &(index, base, output, binding) in &pairs {
        for b in &mut plan.launches[index].bindings {
            if b.value == base || b.value == output {
                b.binding = binding;
                b.kind = BindKind::ReadWrite;
            }
        }
    }
    let mut hash = FxHasher::default();
    "exclusive cache donation".hash(&mut hash);
    pairs.hash(&mut hash);
    plan.hash.0 ^= u128::from(hash.finish()) << 64;
    let plan = Arc::new(plan);
    if leaves.donation_plans.len() == 8 {
        leaves.donation_plans.remove(0);
    }
    if let Some(start) = start {
        eprintln!(
            "CACHE_DONATION stores={} aliases={} proof_us={} cached=false",
            leaves.donations.len(),
            pairs.len(),
            start.elapsed().as_micros()
        );
    }
    leaves.donation_plans.push(Variant {
        original: original.hash.0,
        pairs,
        plan: plan.clone(),
    });
    Some(plan)
}
