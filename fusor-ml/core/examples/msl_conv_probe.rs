//! Probe which simdgroup fragment-conversion MSL forms Metal accepts and
//! which produce correct values: candidate codegen for cooperative stores
//! whose accumulator scalar differs from the destination memory scalar.

fn main() {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::PASSTHROUGH_SHADERS,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let errors = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        {
            let errors = errors.clone();
            device.on_uncaptured_error(std::sync::Arc::new(move |error: wgpu::Error| {
                errors.lock().unwrap().push(format!("{error}"));
            }));
        }

        let variants: &[(&str, &str)] = &[
            (
                "loop64",
                "for (int i = 0; i < 64; i++) { cvt.thread_elements()[i] = half(acc.thread_elements()[i]); }",
            ),
            (
                "loop2",
                "cvt.thread_elements()[0] = half(acc.thread_elements()[0]); cvt.thread_elements()[1] = half(acc.thread_elements()[1]);",
            ),
        ];

        for (name, body) in variants {
            let source = format!(
                r#"#include <metal_stdlib>
kernel void main0(device half* out [[buffer(0)]], uint3 tid [[thread_position_in_grid]]) {{
    metal::simdgroup_float8x8 acc = metal::make_filled_simdgroup_matrix<float, 8, 8>(1.25f);
    metal::simdgroup_half8x8 cvt;
    {body}
    metal::simdgroup_store(cvt, out, 8);
}}
"#
            );
            let module = unsafe {
                device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some(name),
                    msl: Some(std::borrow::Cow::Owned(source)),
                    num_workgroups: (32, 1, 1),
                    ..Default::default()
                })
            };
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                ..Default::default()
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main0"),
                compilation_options: Default::default(),
                cache: None,
            });
            let out = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 64 * 2,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: out.as_entire_binding(),
                }],
            });
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: 64 * 2,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out, 0, &staging, 0, 64 * 2);
            queue.submit([encoder.finish()]);
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let mut failed = false;
            if rx.recv().unwrap().is_err() {
                failed = true;
            }
            {
                let mut collected = errors.lock().unwrap();
                if !collected.is_empty() {
                    let first = collected[0].lines().take(14).collect::<Vec<_>>().join(" | ");
                    println!("{name}: FAIL {first}");
                    collected.clear();
                    failed = true;
                }
            }
            if failed {
                continue;
            }
            let view = slice.get_mapped_range();
            let values: Vec<f32> = view
                .chunks_exact(2)
                .map(|pair| half::f16::from_le_bytes([pair[0], pair[1]]).to_f32())
                .collect();
            let correct = values.iter().filter(|&&value| value == 1.25).count();
            println!("{name}: OK {correct}/64 correct, first 4: {:?}", &values[..4]);
        }
    });
}
