use std::collections::{BTreeMap, BTreeSet};

use crate::*;

type ByteKey = (u64, [u32; 3], u64);
type Owner = ([u32; 3], u32);

/// The stepping seam. Rewind resets and deterministically replays to a cursor;
/// views never implement the memory model. One simulator owns one cold-cache run.
pub struct Simulator {
    trace: Trace,
    config: GpuConfig,
    cursor: usize,
    event_cursor: usize,
    subgroup_cursor: u32,
    totals: Metrics,
    cache: BTreeMap<(u64, u64), u64>,
    tick: u64,
    reads: BTreeMap<ByteKey, Owner>,
    writes: BTreeMap<ByteKey, (u32, u64)>,
    epochs: BTreeMap<[u32; 3], u64>,
}

impl Simulator {
    pub fn new(trace: Trace, config: GpuConfig) -> Result<Self> {
        config.validate()?;
        require(trace.version == 1, "unsupported trace version (expected 1)")?;
        require(
            (1..=1024).contains(&trace.workgroup_size),
            "workgroup size must be 1..=1024",
        )?;
        if let Some(size) = trace.recorded_subgroup_size {
            require(
                size == config.subgroup_size,
                "IR trace subgroup size differs; re-record the kernel at the requested size",
            )?;
        }
        let mut resources = BTreeMap::new();
        for resource in &trace.resources {
            require(
                resources.insert(resource.id, resource).is_none(),
                "duplicate resource id",
            )?;
        }
        for (i, event) in trace.events.iter().enumerate() {
            require(
                event.source.chars().all(|c| !c.is_control()),
                format!("event {i}: source contains control characters"),
            )?;
            match &event.action {
                Action::Memory {
                    resource, lanes, ..
                } => {
                    let resource = resources
                        .get(resource)
                        .ok_or_else(|| Error(format!("event {i}: unknown resource {resource}")))?;
                    for lane in lanes {
                        require(
                            lane.lane < trace.workgroup_size,
                            format!("event {i}: lane outside workgroup"),
                        )?;
                        require(
                            (1..=256).contains(&lane.bytes),
                            format!("event {i}: access width must be 1..=256 bytes"),
                        )?;
                        let end = lane.address.checked_add(u64::from(lane.bytes));
                        require(
                            end.is_some_and(|end| end <= resource.bytes),
                            format!(
                                "event {i}: lane {} out-of-bounds access at {} + {} (resource {} bytes)",
                                lane.lane, lane.address, lane.bytes, resource.bytes
                            ),
                        )?;
                    }
                }
                Action::Control { lanes, .. } => {
                    require(
                        lanes.iter().all(|lane| *lane < trace.workgroup_size),
                        format!("event {i}: control lane outside workgroup"),
                    )?;
                }
                Action::Barrier { .. } => {}
            }
        }
        Ok(Self {
            trace,
            config,
            cursor: 0,
            event_cursor: 0,
            subgroup_cursor: 0,
            totals: Metrics::default(),
            cache: BTreeMap::new(),
            tick: 0,
            reads: BTreeMap::new(),
            writes: BTreeMap::new(),
            epochs: BTreeMap::new(),
        })
    }

    pub fn trace(&self) -> &Trace {
        &self.trace
    }
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }
    pub fn totals(&self) -> &Metrics {
        &self.totals
    }
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn is_finished(&self) -> bool {
        self.trace
            .events
            .iter()
            .enumerate()
            .skip(self.event_cursor)
            .all(|(i, event)| match &event.action {
                Action::Memory { lanes, .. } => lanes.iter().all(|lane| {
                    i == self.event_cursor
                        && lane.lane / self.config.subgroup_size < self.subgroup_cursor
                }),
                _ => false,
            })
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
        self.event_cursor = 0;
        self.subgroup_cursor = 0;
        self.totals = Metrics::default();
        self.cache.clear();
        self.reads.clear();
        self.writes.clear();
        self.epochs.clear();
        self.tick = 0;
    }

    /// Position immediately after `count` steps, returning the last step there.
    /// A position beyond the end returns the actual final step.
    pub fn seek(&mut self, count: usize) -> Option<Step> {
        self.reset();
        let mut last = None;
        for _ in 0..count {
            let Some(step) = self.step() else { break };
            last = Some(step);
        }
        last
    }

    pub fn step(&mut self) -> Option<Step> {
        loop {
            let event = self.trace.events.get(self.event_cursor)?.clone();
            let event_index = self.event_cursor;
            let (action, subgroup) = match event.action {
                Action::Memory {
                    resource,
                    access,
                    lanes,
                } => {
                    let subgroup = self.subgroup_cursor;
                    let selected: Vec<_> = lanes
                        .into_iter()
                        .filter(|lane| lane.lane / self.config.subgroup_size == subgroup)
                        .collect();
                    self.subgroup_cursor += 1;
                    if self.subgroup_cursor
                        >= self
                            .trace
                            .workgroup_size
                            .div_ceil(self.config.subgroup_size)
                    {
                        self.event_cursor += 1;
                        self.subgroup_cursor = 0;
                    }
                    if selected.is_empty() {
                        continue;
                    }
                    (
                        Action::Memory {
                            resource,
                            access,
                            lanes: selected,
                        },
                        Some(subgroup),
                    )
                }
                action => {
                    self.event_cursor += 1;
                    (action, None)
                }
            };
            let mut step = Step {
                index: self.cursor,
                event_index,
                workgroup: event.workgroup,
                subgroup,
                source: event.source,
                action,
                metrics: Metrics {
                    steps: 1,
                    ..Metrics::default()
                },
                totals: Metrics::default(),
                sectors: vec![],
                bank_words: vec![],
                evicted: vec![],
                hints: vec![],
            };
            self.analyze(&mut step);
            self.cursor += 1;
            self.totals.add(&step.metrics);
            step.totals = self.totals.clone();
            return Some(step);
        }
    }

    fn analyze(&mut self, step: &mut Step) {
        let Action::Memory {
            resource,
            access,
            lanes,
        } = &step.action
        else {
            if matches!(step.action, Action::Barrier { storage: false }) {
                *self.epochs.entry(step.workgroup).or_default() += 1;
            }
            return;
        };
        let resource = self
            .trace
            .resources
            .iter()
            .find(|r| r.id == *resource)
            .unwrap();
        let subgroup = step.subgroup.unwrap();
        let is_read = *access == AccessKind::Read;
        let scope = if resource.space == Space::Workgroup {
            step.workgroup
        } else {
            [0; 3]
        };
        let mut bytes = BTreeSet::new();
        let mut byte_lanes: BTreeMap<u64, BTreeSet<u32>> = BTreeMap::new();
        let mut sectors = BTreeSet::new();
        let mut lines = BTreeSet::new();
        let mut banks = vec![BTreeSet::new(); self.config.banks as usize];
        for lane in lanes {
            step.metrics.requested_bytes += u64::from(lane.bytes);
            for byte in lane.address..lane.address + u64::from(lane.bytes) {
                bytes.insert(byte);
                byte_lanes.entry(byte).or_default().insert(lane.lane);
                sectors.insert(byte / u64::from(self.config.transaction_bytes));
                lines.insert(byte / u64::from(self.config.cache_line_bytes));
                let word = byte / u64::from(self.config.bank_width_bytes);
                banks[(word % u64::from(self.config.banks)) as usize].insert(word);
            }
        }
        step.metrics.unique_bytes = bytes.len() as u64;
        if is_read {
            step.metrics.within_subgroup_duplicate_bytes =
                step.metrics.requested_bytes - step.metrics.unique_bytes;
            for &byte in &bytes {
                let key = (resource.id, scope, byte);
                if let Some((wg, sg)) = self.reads.insert(key, (step.workgroup, subgroup)) {
                    if wg != step.workgroup {
                        step.metrics.cross_workgroup_reuse_bytes += 1;
                    } else if sg != subgroup {
                        step.metrics.cross_subgroup_reuse_bytes += 1;
                    } else {
                        step.metrics.same_subgroup_reuse_bytes += 1;
                    }
                }
            }
        } else {
            for &byte in &bytes {
                self.reads.remove(&(resource.id, scope, byte));
            }
            if *access == AccessKind::Write && byte_lanes.values().any(|ls| ls.len() > 1) {
                step.hints.push(
                    "Overlapping non-atomic writes by multiple lanes; inspect a possible race."
                        .into(),
                );
            }
        }
        if resource.space == Space::Storage {
            step.metrics.transactions = sectors.len() as u64;
            step.metrics.ideal_transactions =
                (bytes.len() as u64).div_ceil(u64::from(self.config.transaction_bytes));
            step.sectors = sectors.into_iter().collect();
            let capacity = self.config.cache_bytes / self.config.cache_line_bytes;
            for line in lines {
                let key = (resource.id, line);
                self.tick += 1;
                if is_read {
                    if self.cache.contains_key(&key) {
                        step.metrics.cache_hits += 1;
                        self.cache.insert(key, self.tick);
                    } else {
                        step.metrics.cache_misses += 1;
                        // Bypassed cache services sectors, not full cache lines.
                        if capacity > 0 {
                            step.metrics.cache_fill_bytes +=
                                u64::from(self.config.cache_line_bytes);
                            if self.cache.len() == capacity as usize {
                                let oldest =
                                    *self.cache.iter().min_by_key(|(_, age)| **age).unwrap().0;
                                self.cache.remove(&oldest);
                                step.evicted.push(CacheLine {
                                    resource: oldest.0,
                                    line: oldest.1,
                                });
                                step.metrics.cache_evictions += 1;
                            }
                            self.cache.insert(key, self.tick);
                        }
                    }
                } else {
                    self.cache.remove(&key);
                }
            }
            if step.metrics.transactions > step.metrics.ideal_transactions {
                step.hints.push(format!("{} sectors vs packed lower bound {}: inspect lane stride/alignment or transpose the tile.", step.metrics.transactions, step.metrics.ideal_transactions));
            }
            if step.metrics.cross_subgroup_reuse_bytes > 0 {
                step.hints.push("Other subgroups read these bytes: consider workgroup staging if reuse repays stores and barriers; cache hits already help.".into());
            }
            if step.metrics.within_subgroup_duplicate_bytes > 0 {
                step.hints.push("Lanes share read bytes: consider one load plus subgroup broadcast; duplicate reads already coalesce.".into());
            }
            if step.metrics.cache_evictions > 0 {
                step.hints.push("The virtual LRU cache evicted lines: compare a smaller tile or a different loop order.".into());
            }
        } else {
            let epoch = *self.epochs.get(&step.workgroup).unwrap_or(&0);
            let mut uninitialized = false;
            let mut unordered = false;
            for byte in bytes {
                let key = (resource.id, scope, byte);
                if is_read {
                    match self.writes.get(&key) {
                        None => uninitialized = true,
                        Some((sg, e)) if *sg != subgroup && *e == epoch => unordered = true,
                        _ => {}
                    }
                } else {
                    self.writes.insert(key, (subgroup, epoch));
                }
            }
            if uninitialized {
                step.hints.push("Workgroup read without an earlier explicit write in this trace (implicit zero initialization is not traced).".into());
            }
            if unordered {
                step.hints.push("Cross-subgroup shared read follows a write without a workgroup barrier; inspect synchronization.".into());
            }
            step.metrics.max_bank_degree = banks.iter().map(|b| b.len() as u64).max().unwrap_or(0);
            step.metrics.ideal_bank_degree = banks
                .iter()
                .map(|b| b.len() as u64)
                .sum::<u64>()
                .div_ceil(u64::from(self.config.banks));
            let excess = step.metrics.max_bank_degree > step.metrics.ideal_bank_degree;
            step.metrics.bank_conflict_steps = u64::from(excess);
            if excess {
                step.hints.push(format!("{} bank rounds vs lower bound {}: try padding or swizzling shared-memory rows.", step.metrics.max_bank_degree, step.metrics.ideal_bank_degree));
            }
            step.bank_words = banks.into_iter().map(|b| b.into_iter().collect()).collect();
        }
    }
}
