//! Time one fixed-program region shader in isolation: `N` dependent
//! dispatches in one compute pass over a real-size arena.
//! Run: cargo run --release -p fusor-gpu --example kernel_bench -- <file.wgsl> <grid> <arena_bytes> [dispatches]
use fusor_gpu::target::GpuTarget;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let grid: u32 = args[2].parse().unwrap();
    let arena_bytes: u64 = args[3].parse().unwrap();
    let n: u32 = args.get(4).map_or(400, |x| x.parse().unwrap());
    let target = GpuTarget::new_blocking().expect("gpu");
    let device = target.device().device();
    let queue = target.device().queue();
    let source = std::fs::read_to_string(path).unwrap();
    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    let shader = unsafe {
        device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)) },
            wgpu::ShaderRuntimeChecks { force_loop_bounding: false, ..wgpu::ShaderRuntimeChecks::checked() },
        )
    };
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions { zero_initialize_workgroup_memory: false, ..Default::default() },
        cache: None,
    });
    let arena = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: arena_bytes.div_ceil(4) * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Small finite values everywhere: indices read as tiny integers.
    let fill: Vec<u8> = (0..arena_bytes.div_ceil(4))
        .flat_map(|i| ((((i * 2654435761) % 1000) as f32) / 4000.0).to_bits().to_le_bytes())
        .collect();
    queue.write_buffer(&arena, 0, &fill);
    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: arena.as_entire_binding() }],
    });
    let run = |count: u32| {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind, &[]);
            for _ in 0..count {
                pass.dispatch_workgroups(grid, 1, 1);
            }
        }
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    };
    run(50);
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        run(n);
        best = best.min(t.elapsed().as_secs_f64() * 1e6 / n as f64);
    }
    println!("{best:.2} us/dispatch");
}
