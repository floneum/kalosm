//! Bounded experiment for the three shaders captured by visualize_kernels.
//! Usage: inspect_llama_hotspots SHADER_DIR OUTPUT_CSV
//! Synthetic, deterministic inputs; validates all output elements against a
//! CPU reference before reporting repeated-dispatch GPU timestamp medians.
use std::{borrow::Cow, error::Error, fs, path::Path, sync::mpsc};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Case {
    name: &'static str,
    family: &'static str,
    grid: [u32; 3],
}

const CASES: &[Case] = &[
    Case {
        name: "kfold-original",
        family: "kfold",
        grid: [49152, 2, 1],
    },
    Case {
        name: "kfold-guarded",
        family: "kfold",
        grid: [49152, 2, 1],
    },
    Case {
        name: "kfold-per-lane",
        family: "kfold",
        grid: [384, 1, 1],
    },
    Case {
        name: "kgroup-original",
        family: "kgroup",
        grid: [49152, 2, 1],
    },
    Case {
        name: "kgroup-guarded",
        family: "kgroup",
        grid: [49152, 2, 1],
    },
    Case {
        name: "kgroup-per-lane",
        family: "kgroup",
        grid: [384, 1, 1],
    },
    Case {
        name: "kslab-original",
        family: "kslab",
        grid: [20, 1, 1],
    },
    Case {
        name: "kslab-no-dead-stages",
        family: "kslab",
        grid: [20, 1, 1],
    },
    Case {
        name: "kslab-registers",
        family: "kslab",
        grid: [20, 1, 1],
    },
];

fn buffer(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn read(device: &wgpu::Device, buf: &wgpu::Buffer) -> Result<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device.poll(wgpu::PollType::wait_indefinitely())?;
    rx.recv()??;
    let out = buf.slice(..).get_mapped_range().to_vec();
    buf.unmap();
    Ok(out)
}

fn floats(seed: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| (((i * 13 + seed * 7) % 31) as f32 - 15.0) / 16.0)
        .collect()
}

fn bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn expected(family: &str, arena: &[f32], external: &[f32]) -> Vec<(usize, f32)> {
    let mut result = Vec::new();
    let mut fold = |rows: usize, terms: usize, input: usize, output: usize| {
        for row in 0..rows {
            result.push((
                output + row,
                (0..terms).map(|k| arena[input + k * rows + row]).sum(),
            ));
        }
    };
    match family {
        "kfold" => fold(98304, 4, 218112, 3072),
        "kgroup" => {
            fold(65536, 4, 134144, 68608);
            fold(16384, 16, 396288, 920576);
            fold(16384, 16, 658432, 936960);
        }
        "kslab" => {
            for i in 0..5120 {
                let col = i % 128;
                result.push((
                    21888 + i,
                    arena[16768 + i] * arena[64 + col]
                        + (arena[4288 + i] * external[col]) * arena[13568 + col],
                ));
            }
        }
        _ => unreachable!(),
    }
    result
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: inspect_llama_hotspots SHADER_DIR OUTPUT_CSV".into());
    }
    let gpu = pollster::block_on(fusor_gpu::GpuDevice::request(None))?;
    let device = gpu.device();
    let queue = gpu.queue();
    if !gpu.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
        return Err("GPU timestamps unavailable; no timing result will be fabricated".into());
    }
    println!(
        "adapter={:?}; timestamp_period_ns={}",
        gpu.adapter_info(),
        queue.get_timestamp_period()
    );
    let storage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let uniform = buffer(device, 4096, storage);
    let external = buffer(device, 512, storage);
    let arena = buffer(device, 1412096 * 4, storage);
    let output = buffer(
        device,
        arena.size(),
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = buffer(
        device,
        256,
        wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
    );
    let stamps = buffer(
        device,
        16,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let mut pipelines = Vec::new();
    for case in CASES {
        let source = fs::read_to_string(Path::new(&args[0]).join(format!("{}.wgsl", case.name)))?;
        let module =
            naga::front::wgsl::parse_str(&source).map_err(|e| e.emit_to_string(&source))?;
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|e| e.to_string())?;
        // Same trusted-module policy as the production emitter, after naga validation.
        let shader = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some(case.name),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        let slab = case.family == "kslab";
        let entries: Vec<_> = (0..if slab { 3 } else { 2 })
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding == 0 || (slab && binding == 1),
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(case.name),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffers = if slab {
            vec![&uniform, &external, &arena]
        } else {
            vec![&uniform, &arena]
        };
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &entries,
        });
        // Validate both original and candidate against an independent CPU formula.
        for seed in [1, 19, 73] {
            let input = floats(seed, 1412096);
            let ext = floats(seed + 3, 128);
            let expected = expected(case.family, &input, &ext);
            queue.write_buffer(&arena, 0, &bytes(&input));
            queue.write_buffer(&external, 0, &bytes(&ext));
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(case.grid[0], case.grid[1], case.grid[2]);
            }
            encoder.copy_buffer_to_buffer(&arena, 0, &output, 0, arena.size());
            queue.submit([encoder.finish()]);
            let actual = read(device, &output)?;
            let mut expected_arena = input;
            for (at, value) in expected {
                expected_arena[at] = value;
            }
            for (i, (&want, got)) in expected_arena
                .iter()
                .zip(actual.as_chunks::<4>().0.iter())
                .enumerate()
            {
                let got = f32::from_le_bytes(*got);
                if !got.is_finite() || (got - want).abs() > 1e-5 * (1.0 + want.abs()) {
                    return Err(format!(
                        "{} seed {seed} element {i}: got {got}, expected {want}",
                        case.name
                    )
                    .into());
                }
            }
        }
        println!("{}: validated every arena element, 3 inputs", case.name);
        pipelines.push((pipeline, group));
    }
    // Interleave cases and reverse the order each round to reduce drift bias.
    // Inputs and outputs do not overlap in any case. Batches measure warm,
    // repeated dispatches, not end-to-end inference or cold-cache latency.
    let repeats = std::env::var("INSPECT_REPEATS")
        .ok()
        .map(|s| s.parse::<u32>())
        .transpose()?
        .unwrap_or(32);
    if !(1..=512).contains(&repeats) {
        return Err("INSPECT_REPEATS must be 1..=512".into());
    }
    let inside = gpu
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
    println!("timestamp_inside_pass={inside}; repeats={repeats}");
    let mut samples = vec![Vec::new(); CASES.len()];
    let mut raw = String::from(
        "round,kernel,gpu_us_per_dispatch,submit_readback_wall_us,begin_tick,end_tick\n",
    );
    for round in 0..13 {
        let order: Vec<_> = if round % 2 == 0 {
            (0..CASES.len()).collect()
        } else {
            (0..CASES.len()).rev().collect()
        };
        for index in order {
            let (pipeline, group) = &pipelines[index];
            let case = &CASES[index];
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(case.name),
                    timestamp_writes: (!inside).then_some(wgpu::ComputePassTimestampWrites {
                        query_set: &queries,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, group, &[]);
                if inside {
                    pass.write_timestamp(&queries, 0);
                }
                for _ in 0..repeats {
                    pass.dispatch_workgroups(case.grid[0], case.grid[1], case.grid[2]);
                }
                if inside {
                    pass.write_timestamp(&queries, 1);
                }
            }
            let start = std::time::Instant::now();
            queue.submit([encoder.finish()]);
            // Match Launcher::read_timestamps: this Metal backend needs the
            // compute submission retired before resolving its counter samples.
            // Same-encoder resolve experimentally returned stale prior samples.
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let mut resolve = device.create_command_encoder(&Default::default());
            resolve.resolve_query_set(&queries, 0..2, &resolved, 0);
            resolve.copy_buffer_to_buffer(&resolved, 0, &stamps, 0, 16);
            queue.submit([resolve.finish()]);
            let data = read(device, &stamps)?;
            let wall = start.elapsed().as_secs_f64() * 1e6;
            let begin = u64::from_le_bytes(data[..8].try_into().unwrap());
            let end = u64::from_le_bytes(data[8..].try_into().unwrap());
            let gpu_batch_us =
                end.saturating_sub(begin) as f64 * f64::from(queue.get_timestamp_period()) / 1000.0;
            if end <= begin || gpu_batch_us <= 0.0 || gpu_batch_us > wall * 1.1 {
                return Err(format!("{} invalid/stale GPU timestamp: {gpu_batch_us} us exceeds readback wall {wall} us, or is empty",case.name).into());
            }
            let us = gpu_batch_us / f64::from(repeats);
            raw.push_str(&format!(
                "{round},{},{us:.6},{wall:.3},{begin},{end}\n",
                case.name
            ));
            if round >= 2 {
                samples[index].push(us);
            }
        }
    }
    let mut csv = String::from("kernel,median_us,p25_us,p75_us,dispatches_per_sample,samples\n");
    for (case, mut times) in CASES.iter().zip(samples) {
        times.sort_by(f64::total_cmp);
        let line = format!(
            "{},{:.3},{:.3},{:.3},{repeats},{}\n",
            case.name,
            times[times.len() / 2],
            times[times.len() / 4],
            times[times.len() * 3 / 4],
            times.len()
        );
        print!("{line}");
        csv.push_str(&line);
    }
    fs::write(&args[1], csv)?;
    fs::write(format!("{}.samples.csv", args[1]), raw)?;
    Ok(())
}
