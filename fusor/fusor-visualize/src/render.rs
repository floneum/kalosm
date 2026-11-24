//! Formatting only: all metrics and hints come from Simulator.
use crate::*;
use std::fmt::Write;

pub fn summary(metrics: &Metrics) -> String {
    format!(
        "{} steps | storage sectors {} (packed lower bound {}) | cache hits/misses {}/{} | evictions {}\nread reuse: same subgroup {} B, other subgroup {} B, other workgroup {} B\nbank conflicts: {} steps, maximum {} words/bank",
        metrics.steps,
        metrics.transactions,
        metrics.ideal_transactions,
        metrics.cache_hits,
        metrics.cache_misses,
        metrics.cache_evictions,
        metrics.same_subgroup_reuse_bytes,
        metrics.cross_subgroup_reuse_bytes,
        metrics.cross_workgroup_reuse_bytes,
        metrics.bank_conflict_steps,
        metrics.max_bank_degree
    )
}

pub fn step(step: &Step, trace: &Trace, config: &GpuConfig) -> String {
    let mut out = String::new();
    let sg = step
        .subgroup
        .map(|s| s.to_string())
        .unwrap_or_else(|| "all".into());
    writeln!(
        out,
        "STEP {:04}  workgroup {:?}  subgroup {sg}  {}",
        step.index + 1,
        step.workgroup,
        clean(&step.source)
    )
    .unwrap();
    match &step.action {
        Action::Memory {
            resource,
            access,
            lanes,
        } => {
            let resource = trace.resources.iter().find(|r| r.id == *resource).unwrap();
            let mut active = std::collections::BTreeSet::new();
            active.extend(lanes.iter().map(|l| l.lane));
            let physical = trace
                .workgroup_size
                .saturating_sub(step.subgroup.unwrap() * config.subgroup_size)
                .min(config.subgroup_size);
            writeln!(
                out,
                "  {access:?} {:?} {} | active {}/{} existing lanes ({} virtual slots)",
                resource.space,
                clean(&resource.name),
                active.len(),
                physical,
                config.subgroup_size
            )
            .unwrap();
            let m = &step.metrics;
            writeln!(
                out,
                "  requested {} B / unique {} B / duplicate read {} B",
                m.requested_bytes, m.unique_bytes, m.within_subgroup_duplicate_bytes
            )
            .unwrap();
            if resource.space == Space::Storage {
                let utilization = if m.transactions == 0 {
                    0.0
                } else {
                    100.0 * m.unique_bytes as f64
                        / (m.transactions * u64::from(config.transaction_bytes)) as f64
                };
                writeln!(
                    out,
                    "  sectors {} / packed lower bound {} | useful bytes {:.1}%",
                    m.transactions, m.ideal_transactions, utilization
                )
                .unwrap();
                writeln!(
                    out,
                    "  cache line hits {} / misses {} / evictions {} / fill {} B",
                    m.cache_hits, m.cache_misses, m.cache_evictions, m.cache_fill_bytes
                )
                .unwrap();
                for line in &step.evicted {
                    writeln!(
                        out,
                        "  evicted resource {} line {}",
                        line.resource, line.line
                    )
                    .unwrap();
                }
            } else {
                writeln!(
                    out,
                    "  bank rounds {} / lower bound {} (duplicate reads broadcast)",
                    m.max_bank_degree, m.ideal_bank_degree
                )
                .unwrap();
                for (bank, words) in step
                    .bank_words
                    .iter()
                    .enumerate()
                    .filter(|(_, w)| !w.is_empty())
                {
                    writeln!(out, "  bank {bank:>2}: {} word(s) {words:?}", words.len()).unwrap();
                }
            }
            for hint in &step.hints {
                writeln!(out, "  -> {hint}").unwrap();
            }
            writeln!(
                out,
                "  lane  byte address  width  {}",
                if resource.space == Space::Storage {
                    "sectors"
                } else {
                    "banks"
                }
            )
            .unwrap();
            for lane in lanes {
                let unit = if resource.space == Space::Storage {
                    config.transaction_bytes
                } else {
                    config.bank_width_bytes
                };
                let first = lane.address / u64::from(unit);
                let last = (lane.address + u64::from(lane.bytes) - 1) / u64::from(unit);
                let mapping = (first..=last)
                    .map(|i| {
                        if resource.space == Space::Storage {
                            i.to_string()
                        } else {
                            (i % u64::from(config.banks)).to_string()
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                writeln!(
                    out,
                    "  {:>4}  {:>12}  {:>5}  {}",
                    lane.lane, lane.address, lane.bytes, mapping
                )
                .unwrap();
            }
        }
        Action::Barrier { storage } => {
            writeln!(
                out,
                "  {} barrier",
                if *storage { "storage" } else { "workgroup" }
            )
            .unwrap();
        }
        Action::Control { label, lanes } => {
            writeln!(out, "  {} | active lanes {lanes:?}", clean(label)).unwrap();
        }
    }
    out
}

pub fn clean(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}
