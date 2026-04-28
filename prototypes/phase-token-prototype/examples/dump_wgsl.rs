use phase_token_prototype::{build, F32};

fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    let lowered = ir.lower_to_naga()?;
    let wgsl = naga::back::wgsl::write_string(
        lowered.module(),
        lowered.info(),
        naga::back::wgsl::WriterFlags::empty(),
    )?;

    print!("{wgsl}");
    Ok(())
}
