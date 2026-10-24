//! Bounded WebGPU streaming-read diagnostic with complete integer checksums.
//! Usage: stream_read_lab [MiB per copy=128] [samples=15]. Two copies rotate.
//! This reports a measured reference, not a hardware peak or model throughput.
use std::sync::Arc;

use fusor::device::Device;
use fusor_gpu::launch::{CommandRecord, TimingMode};
use fusor_gpu::pool::{GpuBuffer, TENSOR_USAGE};
use fusor_ir::target::Target;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mib: usize = args.first().map_or(Ok(128), |s| s.parse())?;
    let samples: usize = args.get(1).map_or(Ok(15), |s| s.parse())?;
    assert!(mib.is_power_of_two() && (1..=128).contains(&mib));
    assert!((1..=31).contains(&samples));
    let validate = std::env::var_os("LAB_VALIDATE_ONLY").is_some();
    let dev = Device::gpu_blocking()?;
    let target = dev.backend().gpu_target().unwrap();
    let device = target.device().device();
    let subgroup = target.caps().subgroup_width();
    assert!(target.caps().subgroups.is_some_and(|s| s.is_fixed()));
    let words = mib * 1024 * 1024 / 4;
    let input: Vec<u32> = (0..words as u32)
        .map(|i| i.wrapping_mul(0x9e3779b1) ^ i.rotate_right(7))
        .collect();
    let buffers: Vec<_> = (0..2)
        .map(|_| {
            target
                .pool()
                .create_buffer_init(bytemuck::cast_slice(&input), TENSOR_USAGE)
        })
        .collect::<Result<_, _>>()?;
    let launcher = target.launcher();
    launcher.set_tuning(!validate);
    for block in [64u32, 128, 256] {
        if block > device.limits().max_compute_invocations_per_workgroup
            || !block.is_multiple_of(subgroup)
        {
            continue;
        }
        for vectors in [4u32, 16, 64] {
            let groups = words as u32 / (4 * block * vectors);
            assert!(groups > 0 && groups <= device.limits().max_compute_workgroups_per_dimension);
            let outputs = groups * block / subgroup;
            let output = target
                .pool()
                .alloc_with_usage(u64::from(outputs) * 4, TENSOR_USAGE)?;
            let source = format!(
                r#"
@group(0) @binding(0) var<storage,read> input:array<vec4<u32>>;
@group(0) @binding(1) var<storage,read_write> output:array<u32>;
@compute @workgroup_size({block})
fn main(@builtin(local_invocation_index) lane:u32,
        @builtin(workgroup_id) group:vec3<u32>,
        @builtin(subgroup_invocation_id) slane:u32,
        @builtin(subgroup_id) sg:u32) {{
    var value=vec4<u32>(0u);
    for (var step=0u; step < {vectors}u; step++) {{
        value ^= input[(group.x * {vectors}u + step) * {block}u + lane];
    }}
    let sum=subgroupXor(value.x ^ value.y ^ value.z ^ value.w);
    if (slane == 0u) {{ output[group.x * {subgroups}u + sg] = sum; }}
}}
"#,
                subgroups = block / subgroup
            );
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("streaming checksum"),
                source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
                    naga::front::wgsl::parse_str(&source)?,
                )),
            });
            let pipeline = Arc::new(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label: Some("streaming checksum"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                },
            ));
            let layout = pipeline.get_bind_group_layout(0);
            let commands: Vec<_> = buffers
                .iter()
                .map(|input| {
                    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("streaming checksum"),
                        layout: &layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: GpuBuffer::of(input, "stream input")
                                    .unwrap()
                                    .buffer
                                    .as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: GpuBuffer::of(&output, "stream output")
                                    .unwrap()
                                    .buffer
                                    .as_entire_binding(),
                            },
                        ],
                    });
                    CommandRecord::Dispatch {
                        name: "streaming_checksum",
                        pipeline: pipeline.clone(),
                        bind_group: Arc::new(bind_group),
                        grid: [groups, 1, 1],
                    }
                })
                .collect();
            let mut expected = vec![0u32; outputs as usize];
            for (index, word) in input.iter().enumerate() {
                let vector = index as u32 / 4;
                let group = vector / (block * vectors);
                let sg = (vector % block) / subgroup;
                expected[(group * (block / subgroup) + sg) as usize] ^= word;
            }
            // Check every subgroup checksum, separately for each physical copy.
            for command in &commands {
                launcher.encode_command_records(
                    std::slice::from_ref(command),
                    None,
                    TimingMode::Whole,
                )?;
                launcher.poll_wait()?;
                let bytes = pollster::block_on(target.readback(&output, u64::from(outputs) * 4))?;
                let actual: Vec<_> = bytes
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .collect();
                assert_eq!(actual, expected, "streaming checksum mismatch");
            }
            if validate {
                println!(
                    "STREAM_VALIDATED block={block} vectors={vectors} bytes={} checksums={outputs}",
                    words * 4 * 2
                );
                continue;
            }
            let query = launcher
                .timestamp_query_set(1)
                .ok_or("missing GPU timestamps")?;
            let repeats = 16;
            let records: Vec<_> = (0..repeats)
                .flat_map(|_| commands.iter().cloned())
                .collect();
            let mut times = Vec::new();
            for _ in 0..samples + 3 {
                launcher.encode_command_records(&records, Some(&query), TimingMode::Whole)?;
                launcher.poll_wait()?;
                times.push(
                    launcher.read_timestamps(target.pool(), &query, 1)?[0]
                        / (repeats * commands.len()) as f64,
                );
            }
            let mut measured = times[3..].to_vec();
            measured.sort_by(f64::total_cmp);
            let median = measured[measured.len() / 2];
            println!(
                "STREAM {}",
                serde_json::json!({"block":block,"vectors":vectors,"copies":2,"read_bytes_per_dispatch":words*4,"written_bytes_per_dispatch":outputs*4,"gpu_us":median,"read_gb_s":words as f64*4./median/1000.,"samples_us":&times[3..],"all_checksums_match":true})
            );
        }
    }
    launcher.set_tuning(false);
    Ok(())
}
