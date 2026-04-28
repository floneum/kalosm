use super::*;

#[test]
fn loop_body_must_end_with_sync_witness() {
    let ir = build(|phase| {
        phase.range_step(
            |mut phase, _| {
                let tile = phase.alloc_workgroup::<F32>();
                let pending = phase.cooperative_load(tile);
                let (ready, mut phase) = pending.sync_tile();

                phase.read(&ready);
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    assert_eq!(
        ir.events(),
        &[
            Event::RangeStepStart,
            Event::AllocWorkgroup { tile: TileId(0) },
            Event::CooperativeLoad { tile: TileId(0) },
            Event::WorkgroupBarrier,
            Event::ReadReady { tile: TileId(0) },
            Event::WorkgroupBarrier,
            Event::RangeStepEnd,
            Event::Finish,
        ],
    );
}

#[test]
fn outer_ready_tile_can_be_read_inside_loop_body() {
    let ir = build(|mut phase| {
        let tile = phase.alloc_workgroup::<F32>();
        let pending = phase.cooperative_load(tile);
        let (ready, phase) = pending.sync_tile();

        phase.range_step(
            |mut phase, _| {
                phase.read(&ready);
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    assert_eq!(
        ir.events(),
        &[
            Event::AllocWorkgroup { tile: TileId(0) },
            Event::CooperativeLoad { tile: TileId(0) },
            Event::WorkgroupBarrier,
            Event::RangeStepStart,
            Event::ReadReady { tile: TileId(0) },
            Event::WorkgroupBarrier,
            Event::RangeStepEnd,
            Event::Finish,
        ],
    );
}

#[test]
fn lowers_to_valid_naga_module() {
    let ir = build(|phase| {
        phase.range_step(
            |mut phase, _| {
                let tile = phase.alloc_workgroup::<F32>();
                let pending = phase.cooperative_load(tile);
                let (ready, mut phase) = pending.sync_tile();
                phase.read(&ready);
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    let lowered = ir.lower_to_naga().unwrap();
    assert_eq!(lowered.module().entry_points.len(), 1);
    assert_eq!(lowered.module().global_variables.iter().count(), 1);
}

#[test]
fn lowers_to_naga_module_with_loop_and_barriers() {
    let ir = build(|phase| {
        phase.range_step(
            |mut phase, _| {
                let tile = phase.alloc_workgroup::<F32>();
                let pending = phase.cooperative_load(tile);
                let (ready, mut phase) = pending.sync_tile();
                phase.read(&ready);
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    let lowered = ir.lower_to_naga().unwrap();
    let entry = &lowered.module().entry_points[0];
    assert_eq!(entry.function.body.len(), 2);
}
