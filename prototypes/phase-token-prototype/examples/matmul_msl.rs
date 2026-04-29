use phase_token_prototype::{
    build, DynamicOffset, KernelIr, Layout, LoopOffset, MemoryLevel, Shape, Strides, WorkgroupAxis,
    WorkgroupOffset, F32,
};

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;
const BM: usize = 64;
const BN: usize = 64;
const BK: usize = 16;
const SHARED_PAD: usize = 4;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ir = matmul_ir();
    let lowered = ir.lower_to_naga()?;
    let mut options = naga::back::msl::Options::default();
    options.lang_version = (2, 3);
    options.zero_initialize_workgroup_memory = false;
    options.force_loop_bounding = false;
    options.bounds_check_policies = naga::proc::BoundsCheckPolicies {
        index: naga::proc::BoundsCheckPolicy::Unchecked,
        buffer: naga::proc::BoundsCheckPolicy::Unchecked,
        image_load: naga::proc::BoundsCheckPolicy::Unchecked,
        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
    };
    let pipeline_options = naga::back::msl::PipelineOptions {
        entry_point: Some((naga::ShaderStage::Compute, "main".into())),
        allow_and_force_point_size: false,
        vertex_pulling_transform: false,
        vertex_buffer_mappings: Vec::new(),
    };
    let (msl, _) = naga::back::msl::write_string(
        lowered.module(),
        lowered.info(),
        &options,
        &pipeline_options,
    )?;

    print!("{msl}");
    Ok(())
}

fn matmul_ir() -> KernelIr {
    build(|mut phase| {
        let a_full = phase.storage_tensor_read::<F32>(shape([M, K]));
        let b_full = phase.storage_tensor_read::<F32>(shape([K, N]));
        let c_full = phase.storage_tensor::<F32>(shape([M, N]));
        let a_in = a_full.dynamic_tile_2d(
            shape([BM, BK]),
            Some(DynamicOffset::Workgroup(WorkgroupOffset::new(
                WorkgroupAxis::Y,
                BM as u32,
            ))),
            Some(DynamicOffset::Loop(LoopOffset::new(BK as u32))),
        );
        let b_in = b_full.dynamic_tile_2d(
            shape([BK, BN]),
            Some(DynamicOffset::Loop(LoopOffset::new(BK as u32))),
            Some(DynamicOffset::Workgroup(WorkgroupOffset::new(
                WorkgroupAxis::X,
                BN as u32,
            ))),
        );
        let c_out = c_full.workgroup_tile_2d(
            shape([BM, BN]),
            Some(WorkgroupOffset::new(WorkgroupAxis::Y, BM as u32)),
            Some(WorkgroupOffset::new(WorkgroupAxis::X, BN as u32)),
        );
        let mut acc = phase.alloc_fragment::<F32>(shape([BM, BN]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;

        phase.range_step_count(
            (K / BK) as u32,
            |mut phase, _| {
                let a = phase.alloc_tile_with_layout::<F32>(workgroup_layout([BM, BK], BK));
                let b = phase.alloc_tile_with_layout::<F32>(workgroup_layout([BK, BN], BN));
                let pending = phase.cooperative_load_pair(a, &a_in, b, &b_in);
                let (a, b, mut phase) = pending.sync_tiles();

                phase.gemm(&a, &b, &mut acc);
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
        )
    })
}

fn shape<const R: usize>(dims: [usize; R]) -> Shape {
    Shape::new(dims.map(|dim| dim as u32))
}

fn workgroup_layout(dims: [usize; 2], cols: usize) -> Layout {
    Layout::strided(
        MemoryLevel::Workgroup,
        shape(dims),
        Strides::new([(cols + SHARED_PAD) as u32, 1]),
    )
}
