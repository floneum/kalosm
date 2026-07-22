//! Raw sweep: every cooperative tile over the training step's hot matmul
//! shapes, dispatched directly (no resolver, no selection). Establishes each
//! shape's best achievable rate with the current kernels against the
//! measured ~8.9 TF/s simdgroup ceiling.

use fusor_core::Device;
use fusor_tile_ir::tile;
use fusor_tile_ir_kernels::{
    coop_tile_entries, DenseCoopMatmulConfig, DenseMatmulShape, DenseMatmulTensors,
    SubgroupConfig,
};

const SHAPES: [(u32, u32, u32); 8] = [
    (16384, 384, 384),
    (16384, 384, 1536),
    (16384, 1536, 384),
    (384, 16384, 1536),
    (1536, 16384, 384),
    (4096, 4096, 4096),
    (16384, 3072, 1536),
    (1000, 1024, 1024),
];

fn main() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let wgpu_device = device.wgpu_device();
        let queue = device.wgpu_queue();
        wgpu_device.on_uncaptured_error(std::sync::Arc::new(|error: wgpu::Error| {
            eprintln!("wgpu error: {error}");
        }));
        let subgroups = SubgroupConfig::fixed(fusor_tile_ir::SubgroupToken::new_unchecked(), 32);
        let coop = fusor_tile_ir::CoopMatrixToken::new_unchecked();

        use wgpu::util::DeviceExt;
        let host = |len: usize| -> Vec<f32> {
            (0..len).map(|i| ((i % 61) as f32) * 0.01 - 0.3).collect()
        };
        // DTYPE=f16 benches the native f16-storage kernels (mixed MMA with
        // f32 accumulation); verification stays on the f32 path.
        let f16_storage = std::env::var("DTYPE").as_deref() == Ok("f16");
        // SWIZZLE=1,2,4,8,16 sweeps traversal-order groups per tile entry
        // (1 = plain row-major decomposition).
        let swizzle_groups: Vec<u32> = std::env::var("SWIZZLE")
            .map(|list| list.split(',').filter_map(|v| v.parse().ok()).collect())
            .unwrap_or_else(|_| vec![fusor_tile_ir_kernels::DEFAULT_SWIZZLE_GROUP_M]);
        let make = |data: &[f32]| {
            if f16_storage {
                let halves: Vec<half::f16> =
                    data.iter().map(|&value| half::f16::from_f32(value)).collect();
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&halves),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
            } else {
                wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
            }
        };
        let check = std::env::var_os("CHECK").is_some();
        let staging_f16 = std::env::var("STAGING").as_deref() == Ok("f16");

        // Optional shape filter for tight alternation windows: SHAPE=MxKxN.
        let shape_filter = std::env::var("SHAPE").ok();
        for (m, k, n) in SHAPES {
            if let Some(filter) = &shape_filter {
                if *filter != format!("{m}x{k}x{n}") {
                    continue;
                }
            }
            let flops = 2.0 * m as f64 * k as f64 * n as f64;
            let a_host = host((m * k) as usize);
            let b_host = host((k * n) as usize);
            let a_buf = make(&a_host);
            let b_buf = make(&b_host);
            println!("--- {m}x{k}x{n} ({:.1} GFLOP)", flops / 1e9);
            // Candidates built up-front, then measured round-robin: each
            // rep times every entry once, and each entry keeps its minimum
            // across reps. Under bursty GPU contention a contended burst
            // poisons single entries per rep instead of whole entries, so
            // per-entry minima converge toward uncontended times.
            struct Candidate {
                label: String,
                pipeline: wgpu::ComputePipeline,
                binds: Vec<wgpu::BindGroup>,
                grid: [u32; 3],
                best: f64,
            }
            let mut candidates: Vec<Candidate> = Vec::new();
            for (entry, swizzle_group_m) in coop_tile_entries()
                .iter()
                .flat_map(|entry| swizzle_groups.iter().map(move |&group| (entry, group)))
            {
                // The single-buffered profile is excluded from selection and
                // miscomputes when driven raw; skip it.
                if entry.single_buffered {
                    continue;
                }
                let tile = entry.tile;
                let (bm, bn) = (tile.bm, tile.bn);
                let label = format!(
                    "{bm}x{bn} rg{} cg{} np{} sw{swizzle_group_m}",
                    entry.row_groups, entry.col_groups, entry.n_passes,
                );
                let m_pad = m.div_ceil(bm) * bm;
                let n_pad = n.div_ceil(bn) * bn;
                let total_tiles = (m_pad / bm) * (n_pad / bn);
                if total_tiles > 65535 {
                    continue;
                }
                let element_size = if f16_storage { 2 } else { 4 };
                let y_bufs: Vec<_> = (0..4)
                    .map(|_| {
                        wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                            label: None,
                            size: (m_pad as u64) * (n_pad as u64) * element_size,
                            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                            mapped_at_creation: false,
                        })
                    })
                    .collect();
                let elem = if f16_storage {
                    fusor_tile_ir::ScalarElement::F16.element()
                } else {
                    fusor_tile_ir::ScalarElement::F32.element()
                };
                let mut ok = true;
                let ir = tile::build(|phase| {
                    let a = phase.storage_read(elem, fusor_tile_ir::Shape::new([m, k]));
                    let b = phase.storage_read(elem, fusor_tile_ir::Shape::new([k, n]));
                    let y = phase.storage_write(elem, fusor_tile_ir::Shape::new([m_pad, n_pad]));
                    ok = fusor_tile_ir_kernels::try_batched_coop_matmul(
                        phase,
                        DenseMatmulTensors {
                            a: &a,
                            b: &b,
                            y: &y,
                        },
                        DenseMatmulShape { batch: 1, m, k, n },
                        &fusor_tile_ir_kernels::DenseMatmulEpilogues::empty(),
                        65535,
                        DenseCoopMatmulConfig {
                            coop,
                            subgroups,
                            tile,
                            staging: staging_f16.then_some(fusor_tile_ir::ScalarElement::F16),
                            swizzle_group_m,
                        },
                    );
                });
                if !ok {
                    continue;
                }
                let grid = ir.grid;
                let Ok(kernel) = ir.lower_to_naga() else {
                    println!("  {label}: lowering failed");
                    continue;
                };
                let module = unsafe {
                    wgpu_device.create_shader_module_trusted(
                        wgpu::ShaderModuleDescriptor {
                            label: None,
                            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
                                kernel.module().clone(),
                            )),
                        },
                        wgpu::ShaderRuntimeChecks::unchecked(),
                    )
                };
                let pipeline =
                    wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: None,
                        module: &module,
                        entry_point: Some("main"),
                        compilation_options: wgpu::PipelineCompilationOptions {
                            zero_initialize_workgroup_memory: false,
                            ..Default::default()
                        },
                        cache: None,
                    });
                let layout = pipeline.get_bind_group_layout(0);
                let binds: Vec<_> = y_bufs
                    .iter()
                    .map(|y| {
                        wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: None,
                            layout: &layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: a_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: b_buf.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: y.as_entire_binding(),
                                },
                            ],
                        })
                    })
                    .collect();
                let run = |iters: u32| {
                    let mut encoder = wgpu_device.create_command_encoder(&Default::default());
                    for i in 0..iters {
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &binds[(i as usize) % binds.len()], &[]);
                        pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
                    }
                    queue.submit([encoder.finish()]);
                };
                run(3);
                device.poll_wait();
                if check && !f16_storage {
                    let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                        label: None,
                        size: y_bufs[2].size(),
                        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });
                    let mut encoder =
                        wgpu_device.create_command_encoder(&Default::default());
                    encoder.copy_buffer_to_buffer(&y_bufs[2], 0, &staging, 0, staging.size());
                    queue.submit([encoder.finish()]);
                    let slice = staging.slice(..);
                    let (tx, rx) = std::sync::mpsc::channel();
                    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                    device.poll_wait();
                    rx.recv().unwrap().unwrap();
                    let view = slice.get_mapped_range();
                    let y_host: &[f32] = bytemuck::cast_slice(&view);
                    let mut worst = 0f64;
                    for sample in 0..997u64 {
                        let i = ((sample * 7919) % m as u64) as usize;
                        let j = ((sample * 104729) % n as u64) as usize;
                        let mut acc = 0f64;
                        for kk in 0..k as usize {
                            acc += a_host[i * k as usize + kk] as f64
                                * b_host[kk * n as usize + j] as f64;
                        }
                        let got = y_host[i * n_pad as usize + j] as f64;
                        let err = (got - acc).abs() / acc.abs().max(1.0);
                        worst = worst.max(err);
                        if err >= 5e-3 && std::env::var_os("LENIENT").is_none() {
                            panic!("{label} mismatch at ({i},{j}): got {got}, want {acc}");
                        }
                    }
                    println!("  {label} verified, worst rel err {worst:.2e}");
                }
                candidates.push(Candidate {
                    label,
                    pipeline,
                    binds,
                    grid,
                    best: f64::MAX,
                });
            }
            let iters: u32 = std::env::var("ITERS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(10);
            let reps: u32 = std::env::var("REPS")
                .ok()
                .and_then(|value| value.parse().ok())
                .unwrap_or(6);
            for _ in 0..reps {
                for candidate in candidates.iter_mut() {
                    let start = std::time::Instant::now();
                    let mut encoder = wgpu_device.create_command_encoder(&Default::default());
                    for i in 0..iters {
                        let mut pass = encoder.begin_compute_pass(&Default::default());
                        pass.set_pipeline(&candidate.pipeline);
                        pass.set_bind_group(
                            0,
                            &candidate.binds[(i as usize) % candidate.binds.len()],
                            &[],
                        );
                        pass.dispatch_workgroups(
                            candidate.grid[0],
                            candidate.grid[1],
                            candidate.grid[2],
                        );
                    }
                    queue.submit([encoder.finish()]);
                    device.poll_wait();
                    candidate.best = candidate
                        .best
                        .min(start.elapsed().as_secs_f64() / iters as f64);
                }
            }
            for candidate in &candidates {
                let best = candidate.best;
                println!(
                    "  {}: {:.3} ms, {:.2} TF/s ({:.0}%)",
                    candidate.label,
                    best * 1e3,
                    flops / best / 1e12,
                    flops / best / 1e12 / 8.86 * 100.0
                );
            }
        }
    });
}
