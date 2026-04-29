use std::{borrow::Cow, sync::mpsc};

use phase_token_prototype::{
    build,
    kernels::gemm::{self, GemmTilePlan},
    KernelIr, Shape, F32,
};
use wgpu::util::DeviceExt;

const M: usize = 16;
const N: usize = 16;
const K: usize = 8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ir = matmul_ir();
    let lowered = ir.lower_to_naga()?;

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await?;
    if !adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX)
    {
        return Err(format!(
            "adapter {} does not expose EXPERIMENTAL_COOPERATIVE_MATRIX",
            adapter.get_info().name
        )
        .into());
    }
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("phase-token-prototype device"),
            required_features: wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX,
            required_limits: wgpu::Limits::default(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        })
        .await?;

    let a = make_a();
    let b = make_b();
    let expected = cpu_matmul(&a, &b);

    let a_buffer = storage_buffer(&device, "A", &a, wgpu::BufferUsages::empty());
    let b_buffer = storage_buffer(&device, "B", &b, wgpu::BufferUsages::empty());
    let c_buffer = storage_buffer(
        &device,
        "C",
        &vec![0.0_f32; M * N],
        wgpu::BufferUsages::COPY_SRC,
    );
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("C readback"),
        size: byte_len::<f32>(M * N),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = unsafe {
        device.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: Some("lowered matmul"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(lowered.module().clone())),
            },
            wgpu::ShaderRuntimeChecks::unchecked(),
        )
    };
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("matmul buffers"),
        entries: &storage_bindings(3),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("matmul pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: wgpu::PipelineCompilationOptions {
            zero_initialize_workgroup_memory: false,
            ..Default::default()
        },
        cache: None,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: b_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: c_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&c_buffer, 0, &readback, 0, byte_len::<f32>(M * N));
    queue.submit(Some(encoder.finish()));

    let output = read_f32_buffer(&device, &readback)?;
    let max_abs_error = max_abs_error(&output, &expected);
    if max_abs_error > 1.0e-4 {
        let index = first_mismatch(&output, &expected, 1.0e-4).unwrap_or(0);
        return Err(format!(
            "matmul mismatch at {index}: gpu={} cpu={} max_abs_error={max_abs_error}",
            output[index], expected[index]
        )
        .into());
    }

    println!("run_matmul: ok (max_abs_error = {max_abs_error:.6})");
    Ok(())
}

fn matmul_ir() -> KernelIr {
    build(|mut phase| {
        let a_in = phase.storage_tensor::<F32>(shape([M, K]));
        let b_in = phase.storage_tensor::<F32>(shape([K, N]));
        let c_out = phase.storage_tensor::<F32>(shape([M, N]));
        let mut acc = phase.alloc_fragment::<F32>(shape([M, N]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;

        phase.range_step(
            |mut phase, _| {
                let a = phase.alloc_workgroup_tile::<F32>(shape([M, K]));
                let b = phase.alloc_workgroup_tile::<F32>(shape([K, N]));
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
    })
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &'static str,
    data: &[f32],
    extra_usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra_usage,
    })
}

fn storage_bindings(count: u32) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

fn read_f32_buffer(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let slice = buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;

    let mapped = slice.get_mapped_range();
    let output = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    Ok(output)
}

fn make_a() -> Vec<f32> {
    (0..M)
        .flat_map(|row| (0..K).map(move |k| 1.0 + row as f32 * 0.25 + k as f32 * 0.5))
        .collect()
}

fn make_b() -> Vec<f32> {
    (0..K)
        .flat_map(|k| (0..N).map(move |col| 0.5 + k as f32 * 0.75 - col as f32 * 0.125))
        .collect()
}

fn cpu_matmul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut c = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut sum = 0.0;
            for k in 0..K {
                sum += a[row * K + k] * b[k * N + col];
            }
            c[row * N + col] = sum;
        }
    }
    c
}

fn max_abs_error(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f32::max)
}

fn first_mismatch(actual: &[f32], expected: &[f32], tolerance: f32) -> Option<usize> {
    actual
        .iter()
        .zip(expected)
        .position(|(actual, expected)| (actual - expected).abs() > tolerance)
}

fn byte_len<T>(len: usize) -> u64 {
    (len * std::mem::size_of::<T>()) as u64
}

fn shape<const R: usize>(dims: [usize; R]) -> Shape {
    Shape::new(dims.map(|dim| dim as u32))
}
