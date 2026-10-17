//! Read-only streaming bandwidth with scalar and vec4 loads. This is an
//! observed synthetic rate, not a hardware peak or a model speedup.
//! Usage: streaming_bandwidth [INPUT_MIB=512] [REPEATS=32]
//! Each input element is read once/dispatch. Every subgroup writes a checksum;
//! all checksums are validated for two deterministic input patterns.
use std::{borrow::Cow, error::Error, sync::mpsc};
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn buffer(d: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: false,
    })
}
fn read(d: &wgpu::Device, b: &wgpu::Buffer) -> Result<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    b.slice(..).map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    d.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let bytes = b.slice(..).get_mapped_range().to_vec();
    b.unmap();
    Ok(bytes)
}
fn pipeline(
    d: &wgpu::Device,
    source: &str,
    buffers: &[&wgpu::Buffer],
) -> Result<(wgpu::ComputePipeline, wgpu::BindGroup)> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)?;
    // SAFETY: fixed dimensions below cover each input element exactly once;
    // dispatch counts and buffer extents are checked before creating pipelines.
    // This matches the production trusted-module policy after naga validation.
    let shader = unsafe {
        d.create_shader_module_trusted(
            wgpu::ShaderModuleDescriptor {
                label: Some("streaming bandwidth"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            },
            wgpu::ShaderRuntimeChecks::unchecked(),
        )
    };
    let p = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let entries: Vec<_> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let g = d.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &p.get_bind_group_layout(0),
        entries: &entries,
    });
    Ok((p, g))
}
fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mib = args
        .next()
        .map(|v| v.parse::<u64>())
        .transpose()?
        .unwrap_or(512);
    let repeats = args
        .next()
        .map(|v| v.parse::<u32>())
        .transpose()?
        .unwrap_or(32);
    if args.next().is_some()
        || !(64..=1024).contains(&mib)
        || !mib.is_power_of_two()
        || !(1..=128).contains(&repeats)
    {
        return Err("usage: streaming_bandwidth [INPUT_MIB=512, power of two in 64..1024] [REPEATS=32, 1..128]".into());
    }
    let gpu = pollster::block_on(fusor_gpu::GpuDevice::request(None))?;
    let d = gpu.device();
    let q = gpu.queue();
    let bytes = mib * 1024 * 1024;
    if gpu.caps().subgroup_width() != 32
        || !gpu
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP)
    {
        return Err("this checksum benchmark requires subgroup width 32 and GPU timestamps".into());
    }
    if bytes > d.limits().max_storage_buffer_binding_size || bytes > d.limits().max_buffer_size {
        return Err("requested input exceeds device buffer limit".into());
    }
    println!(
        "adapter={:?}; input_bytes={bytes}; repeats={repeats}; subgroup=32",
        gpu.adapter_info()
    );
    let groups = (bytes / (4 * 256 * 64)) as u32;
    assert!(groups <= d.limits().max_compute_workgroups_per_dimension);
    let src = buffer(d, bytes, wgpu::BufferUsages::STORAGE);
    let dst = buffer(
        d,
        u64::from(groups) * 8 * 4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let checks = buffer(
        d,
        dst.size(),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let cases = [("scalar", false), ("vec4", true)];
    let mut pipes = Vec::new();
    for (_, vector) in cases {
        let (ty, init, body, finish) = if vector {
            (
                "vec4<f32>",
                "vec4<f32>(0.0)",
                "for(var i=0u;i<16u;i++){sum+=src[w.x*4096u+i*256u+lane];}",
                "sum.x+sum.y+sum.z+sum.w",
            )
        } else {
            (
                "f32",
                "0.0",
                "for(var i=0u;i<64u;i++){sum+=src[w.x*16384u+i*256u+lane];}",
                "sum",
            )
        };
        let source = format!(
            "@group(0) @binding(0) var<storage,read> src:array<{ty}>;\n@group(0) @binding(1) var<storage,read_write> dst:array<f32>;\n@compute @workgroup_size(256) fn main(@builtin(workgroup_id) w:vec3<u32>, @builtin(local_invocation_index) lane:u32) {{var sum={init};{body}let total=subgroupAdd({finish});if(lane%32u==0u){{dst[w.x*8u+lane/32u]=total;}}}}\n"
        );
        pipes.push(pipeline(d, &source, &[&src, &dst])?);
    }
    // Vector initialization, folded in Y when it exceeds the X grid limit.
    let init_groups = (bytes / (4 * 4 * 256)) as u32;
    let init_x = init_groups.min(32768);
    let init_y = init_groups / init_x;
    for seed in [17u32, 73] {
        let source = format!(
            "@group(0) @binding(0) var<storage,read_write> src:array<vec4<f32>>;\n@compute @workgroup_size(256) fn main(@builtin(workgroup_id) w:vec3<u32>,@builtin(local_invocation_index) lane:u32){{let i=(w.y*{init_x}u+w.x)*256u+lane;let b=i*4u+{seed}u;src[i]=vec4<f32>(f32(b&255u),f32((b+1u)&255u),f32((b+2u)&255u),f32((b+3u)&255u));}}\n"
        );
        let (init, bind) = pipeline(d, &source, &[&src])?;
        let mut e = d.create_command_encoder(&Default::default());
        {
            let mut pass = e.begin_compute_pass(&Default::default());
            pass.set_pipeline(&init);
            pass.set_bind_group(0, &bind, &[]);
            pass.dispatch_workgroups(init_x, init_y, 1);
        }
        q.submit([e.finish()]);
        for ((name, vector), (p, g)) in cases.iter().zip(&pipes) {
            let mut e = d.create_command_encoder(&Default::default());
            {
                let mut pass = e.begin_compute_pass(&Default::default());
                pass.set_pipeline(p);
                pass.set_bind_group(0, g, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            e.copy_buffer_to_buffer(&dst, 0, &checks, 0, dst.size());
            q.submit([e.finish()]);
            let data = read(d, &checks)?;
            let expected: Vec<f32> = (0..8u32)
                .map(|sg| {
                    (sg * 32..sg * 32 + 32)
                        .map(|lane| {
                            if *vector {
                                (0..4)
                                    .map(|c| ((lane * 4 + c + seed) & 255) as f32 * 16.0)
                                    .sum::<f32>()
                            } else {
                                ((lane + seed) & 255) as f32 * 64.0
                            }
                        })
                        .sum()
                })
                .collect();
            for (i, b) in data.chunks_exact(4).enumerate() {
                let got = f32::from_le_bytes(b.try_into()?);
                if got != expected[i % 8] {
                    return Err(format!(
                        "{name} seed={seed} output={i}: {got} != {}",
                        expected[i % 8]
                    )
                    .into());
                }
            }
            println!(
                "validated case={name} seed={seed} checksums={}",
                data.len() / 4
            );
        }
    }
    let queries = d.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = buffer(
        d,
        256,
        wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
    );
    let stamps = buffer(
        d,
        16,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut samples: [Vec<f64>; 2] = Default::default();
    for round in 0..15 {
        for step in 0..2 {
            let i = if round % 2 == 0 { step } else { 1 - step };
            let (p, g) = &pipes[i];
            let mut e = d.create_command_encoder(&Default::default());
            {
                let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &queries,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(p);
                pass.set_bind_group(0, g, &[]);
                for _ in 0..repeats {
                    pass.dispatch_workgroups(groups, 1, 1);
                }
            }
            q.submit([e.finish()]);
            d.poll(wgpu::PollType::wait_indefinitely())?;
            // Retire the compute submission before resolving Metal timestamps.
            let mut e = d.create_command_encoder(&Default::default());
            e.resolve_query_set(&queries, 0..2, &resolved, 0);
            e.copy_buffer_to_buffer(&resolved, 0, &stamps, 0, 16);
            q.submit([e.finish()]);
            let ts = read(d, &stamps)?;
            let start = u64::from_le_bytes(ts[..8].try_into()?);
            let end = u64::from_le_bytes(ts[8..].try_into()?);
            if end <= start {
                return Err("invalid GPU timestamp interval".into());
            }
            let us = (end - start) as f64 * f64::from(q.get_timestamp_period())
                / 1000.0
                / f64::from(repeats);
            if round >= 3 {
                samples[i].push(us);
            }
        }
    }
    for (i, (name, _)) in cases.iter().enumerate() {
        samples[i].sort_by(f64::total_cmp);
        let s = &samples[i];
        let us = (s[s.len() / 2 - 1] + s[s.len() / 2]) / 2.0;
        println!(
            "STREAM case={name} input_bytes={bytes} output_bytes={} samples={} gpu_us={us:.3} p25_us={:.3} p75_us={:.3} read_GBs={:.3}",
            dst.size(),
            s.len(),
            s[s.len() / 4],
            s[s.len() * 3 / 4],
            bytes as f64 / us / 1000.0
        );
    }
    Ok(())
}
