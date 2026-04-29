use phase_token_prototype::{
    build,
    kernels::gemm::{self, GemmTilePlan},
    Shape, F32,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ir = build(|mut phase| {
        let a_in = phase.storage_tensor::<F32>(Shape::new([16, 8]));
        let b_in = phase.storage_tensor::<F32>(Shape::new([8, 16]));
        let c_out = phase.storage_tensor::<F32>(Shape::new([16, 16]));
        let mut acc = phase.alloc_fragment::<F32>(Shape::new([16, 16]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;

        phase.range_step(
            |mut phase, _| {
                let a = phase.alloc_workgroup_tile::<F32>(Shape::new([16, 8]));
                let b = phase.alloc_workgroup_tile::<F32>(Shape::new([8, 16]));

                let pending = phase.cooperative_load_pair(a, &a_in, b, &b_in);
                let (a, b, mut phase) = pending.sync_tiles();

                gemm::tiled(
                    &mut phase,
                    &a,
                    &b,
                    &mut acc,
                    GemmTilePlan::portable(16, 16, 8),
                );
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
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
