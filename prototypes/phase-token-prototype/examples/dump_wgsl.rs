use phase_token_prototype::{build, Shape, F32};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ir = build(|mut phase| {
        let src = phase.storage_tensor::<F32>(Shape::new([32]));
        let dst = phase.storage_tensor::<F32>(Shape::new([32]));
        phase.range_step(
            |mut phase, _| {
                let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
                let pending = phase.cooperative_load(tile, &src);
                let (ready, mut phase) = pending.sync_tile();

                phase.store_ready_to_storage(&ready, &dst);
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    let lowered = ir.lower_to_naga()?;
    let wgsl = naga::back::wgsl::write_string(
        lowered.module(),
        lowered.info(),
        naga::back::wgsl::WriterFlags::empty(),
    )?;

    print!("{wgsl}");
    Ok(())
}
