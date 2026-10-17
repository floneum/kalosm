use fusor_visualize::*;

fn memory(addresses: impl IntoIterator<Item = (u32, u64, u32)>) -> Event {
    Event {
        workgroup: [0; 3],
        source: "test load".into(),
        action: Action::Memory {
            resource: 1,
            access: AccessKind::Read,
            lanes: addresses
                .into_iter()
                .map(|(lane, address, bytes)| LaneAccess {
                    lane,
                    address,
                    bytes,
                })
                .collect(),
        },
    }
}
fn trace(events: Vec<Event>, space: Space, block: u32) -> Trace {
    Trace {
        version: 1,
        name: "test".into(),
        workgroup_size: block,
        recorded_subgroup_size: None,
        resources: vec![Resource {
            id: 1,
            name: "data".into(),
            space,
            bytes: 1 << 20,
        }],
        events,
        notes: vec![],
    }
}
fn run(trace: Trace, config: GpuConfig) -> (Vec<Step>, Metrics) {
    let mut sim = Simulator::new(trace, config).unwrap();
    let mut steps = vec![];
    while let Some(step) = sim.step() {
        steps.push(step);
    }
    (steps, sim.totals().clone())
}
fn addresses(stride: u64) -> Event {
    memory((0..32).map(|lane| (lane, u64::from(lane) * stride, 4)))
}

#[test]
fn coalescing_distinguishes_stride_alignment_and_broadcast() {
    let (_, contiguous) = run(
        trace(vec![addresses(4)], Space::Storage, 32),
        GpuConfig::default(),
    );
    let (_, strided) = run(
        trace(vec![addresses(128)], Space::Storage, 32),
        GpuConfig::default(),
    );
    let (_, broadcast) = run(
        trace(vec![addresses(0)], Space::Storage, 32),
        GpuConfig::default(),
    );
    let (_, misaligned) = run(
        trace(
            vec![memory(
                (0..32).map(|lane| (lane, u64::from(lane) * 4 + 4, 4)),
            )],
            Space::Storage,
            32,
        ),
        GpuConfig::default(),
    );
    assert_eq!(
        (contiguous.transactions, contiguous.ideal_transactions),
        (4, 4)
    );
    assert_eq!((strided.transactions, strided.ideal_transactions), (32, 4));
    assert_eq!(
        (misaligned.transactions, misaligned.ideal_transactions),
        (5, 4)
    );
    assert_eq!(
        (
            broadcast.transactions,
            broadcast.unique_bytes,
            broadcast.within_subgroup_duplicate_bytes
        ),
        (1, 4, 124)
    );
    assert_eq!(contiguous.cache_misses, 1); // not 32 independent lane misses
}

#[test]
fn wide_access_crossing_a_sector_and_cache_line_counts_both() {
    let (_, m) = run(
        trace(vec![memory([(0, 124, 16)])], Space::Storage, 1),
        GpuConfig::default(),
    );
    assert_eq!((m.transactions, m.cache_misses, m.unique_bytes), (2, 2, 16));
}

#[test]
fn cache_uses_lru_not_fifo_and_resource_identity() {
    let events = [0, 128, 0, 256, 128].map(|a| memory([(0, a, 4)])).to_vec();
    let config = GpuConfig {
        cache_bytes: 256,
        ..GpuConfig::default()
    };
    let (steps, m) = run(trace(events, Space::Storage, 1), config);
    assert_eq!((m.cache_hits, m.cache_misses, m.cache_evictions), (1, 4, 2));
    assert_eq!(steps[3].evicted[0].line, 1);
    let mut t = trace(
        vec![memory([(0, 0, 4)]), memory([(0, 0, 4)])],
        Space::Storage,
        1,
    );
    t.resources.push(Resource {
        id: 2,
        name: "other".into(),
        space: Space::Storage,
        bytes: 128,
    });
    if let Action::Memory { resource, .. } = &mut t.events[1].action {
        *resource = 2;
    }
    let (_, m) = run(t, GpuConfig::default());
    assert_eq!((m.cache_hits, m.cache_misses), (0, 2));
}

#[test]
fn stores_invalidate_read_reuse_and_do_not_allocate() {
    let load = memory([(0, 0, 4)]);
    let mut store = load.clone();
    if let Action::Memory { access, .. } = &mut store.action {
        *access = AccessKind::Write;
    }
    let (_, m) = run(
        trace(vec![load.clone(), store, load], Space::Storage, 1),
        GpuConfig::default(),
    );
    assert_eq!(
        (m.cache_hits, m.cache_misses, m.same_subgroup_reuse_bytes),
        (0, 2, 0)
    );
}

#[test]
fn disabled_cache_has_no_hits_evictions_or_fills() {
    let load = addresses(4);
    let (_, m) = run(
        trace(vec![load.clone(), load], Space::Storage, 32),
        GpuConfig {
            cache_bytes: 0,
            ..GpuConfig::default()
        },
    );
    assert_eq!(
        (
            m.cache_hits,
            m.cache_misses,
            m.cache_evictions,
            m.cache_fill_bytes
        ),
        (0, 2, 0, 0)
    );
}

#[test]
fn trace_can_be_regrouped_and_partial_subgroups_preserve_all_lanes() {
    let t = trace(
        vec![memory((0..35).map(|lane| (lane, u64::from(lane) * 4, 4)))],
        Space::Storage,
        35,
    );
    for width in [1, 16, 32, 64] {
        let (steps, m) = run(
            t.clone(),
            GpuConfig {
                subgroup_size: width,
                ..GpuConfig::default()
            },
        );
        assert_eq!(steps.len(), 35u32.div_ceil(width) as usize);
        assert_eq!(m.requested_bytes, 140);
    }
}

#[test]
fn bank_padding_eliminates_conflicts_and_reads_broadcast() {
    let (_, conflict) = run(
        trace(vec![addresses(128)], Space::Workgroup, 32),
        GpuConfig::default(),
    );
    let (_, padded) = run(
        trace(vec![addresses(132)], Space::Workgroup, 32),
        GpuConfig::default(),
    );
    let (_, broadcast) = run(
        trace(vec![addresses(0)], Space::Workgroup, 32),
        GpuConfig::default(),
    );
    assert_eq!(
        (conflict.max_bank_degree, conflict.bank_conflict_steps),
        (32, 1)
    );
    assert_eq!((padded.max_bank_degree, padded.bank_conflict_steps), (1, 0));
    assert_eq!(broadcast.max_bank_degree, 1);
    assert_eq!(conflict.transactions, 0); // shared banks are not storage sectors
}

#[test]
fn reuse_is_separated_by_subgroup_and_workgroup() {
    let mut other = memory([(0, 0, 4)]);
    other.workgroup = [1, 0, 0];
    let (_, m) = run(
        trace(
            vec![
                memory([(0, 0, 4)]),
                memory([(0, 0, 4)]),
                memory([(32, 0, 4)]),
                other,
            ],
            Space::Storage,
            64,
        ),
        GpuConfig::default(),
    );
    assert_eq!(
        (
            m.same_subgroup_reuse_bytes,
            m.cross_subgroup_reuse_bytes,
            m.cross_workgroup_reuse_bytes
        ),
        (4, 4, 4)
    );
    assert_eq!((m.cache_hits, m.cache_misses), (3, 1));
}

#[test]
fn only_workgroup_barriers_order_shared_writes_across_subgroups() {
    for barrier in [None, Some(true), Some(false)] {
        let mut write = memory([(0, 0, 4)]);
        if let Action::Memory { access, .. } = &mut write.action {
            *access = AccessKind::Write;
        }
        let mut events = vec![write];
        if let Some(storage) = barrier {
            events.push(Event {
                source: "barrier".into(),
                workgroup: [0; 3],
                action: Action::Barrier { storage },
            });
        }
        events.push(memory([(32, 0, 4)]));
        let (steps, _) = run(trace(events, Space::Workgroup, 64), GpuConfig::default());
        assert_eq!(
            steps
                .last()
                .unwrap()
                .hints
                .iter()
                .any(|h| h.contains("without a workgroup barrier")),
            barrier != Some(false)
        );
    }
}

#[test]
fn shared_memory_is_private_to_each_workgroup() {
    let mut write = memory([(0, 0, 4)]);
    if let Action::Memory { access, .. } = &mut write.action {
        *access = AccessKind::Write;
    }
    let mut read = memory([(0, 0, 4)]);
    read.workgroup = [1, 0, 0];
    let (steps, m) = run(
        trace(vec![write, read], Space::Workgroup, 32),
        GpuConfig::default(),
    );
    assert!(
        steps[1]
            .hints
            .iter()
            .any(|h| h.contains("without an earlier explicit write"))
    );
    assert_eq!(m.cross_workgroup_reuse_bytes, 0);
}

#[test]
fn duplicate_non_atomic_writes_are_not_reported_as_broadcasts() {
    let mut event = addresses(0);
    if let Action::Memory { access, .. } = &mut event.action {
        *access = AccessKind::Write;
    }
    let (steps, m) = run(
        trace(vec![event], Space::Workgroup, 32),
        GpuConfig::default(),
    );
    assert!(steps[0].hints.iter().any(|h| h.contains("possible race")));
    assert_eq!(m.within_subgroup_duplicate_bytes, 0);
}

#[test]
fn seek_replays_identical_cache_history_and_empty_memory_events_finish() {
    let mut t = trace(
        vec![addresses(4), addresses(128), addresses(4)],
        Space::Storage,
        32,
    );
    t.events.push(memory([]));
    let mut sim = Simulator::new(t, GpuConfig::default()).unwrap();
    let original = sim.seek(3).unwrap();
    assert!(sim.is_finished());
    let prior = sim.seek(1).unwrap();
    assert_eq!(prior.totals.steps, 1);
    assert!(!sim.is_finished());
    let replay = sim.seek(3).unwrap();
    assert_eq!(original.totals, replay.totals);
    assert_eq!(
        serde_json::to_string(&original).unwrap(),
        serde_json::to_string(&replay).unwrap()
    );
}

#[test]
fn invalid_config_trace_and_overflow_fail_before_stepping() {
    assert!(
        GpuConfig {
            subgroup_size: 0,
            ..GpuConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        GpuConfig {
            cache_line_bytes: 16,
            ..GpuConfig::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        GpuConfig {
            cache_bytes: 1,
            ..GpuConfig::default()
        }
        .validate()
        .is_err()
    );
    for event in [
        memory([(32, 0, 4)]),
        memory([(0, u64::MAX, 4)]),
        memory([(0, 0, 0)]),
        memory([(0, 1 << 20, 4)]),
    ] {
        assert!(
            Simulator::new(trace(vec![event], Space::Storage, 32), GpuConfig::default()).is_err()
        );
    }
    let mut t = trace(vec![], Space::Storage, 32);
    t.recorded_subgroup_size = Some(16);
    assert!(Simulator::new(t, GpuConfig::default()).is_err());
}

#[test]
fn trace_json_roundtrip_and_unknown_fields() {
    let original = trace(vec![addresses(4)], Space::Storage, 32);
    let encoded = serde_json::to_string(&original).unwrap();
    let decoded = serde_json::from_str(&encoded).unwrap();
    assert_eq!(
        run(original, GpuConfig::default()).1,
        run(decoded, GpuConfig::default()).1
    );
    assert!(
        serde_json::from_str::<Trace>(
            &encoded.replace("\"version\":1", "\"version\":1,\"typo\":3")
        )
        .is_err()
    );
}
