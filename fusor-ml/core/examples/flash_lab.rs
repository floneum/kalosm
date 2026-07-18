//! Standalone lab for the fused flash-attention kernel: build, dispatch raw,
//! verify sampled rows against a CPU reference, and time at the training
//! shape ([64, 6, 256, 64]). MODE=causal|mask|none selects the masking form.

use fusor_core::Device;
use fusor_tile_ir::tile;
use fusor_tile_ir_kernels::{FlashAttentionShape, SubgroupConfig};

fn main() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let wgpu_device = device.wgpu_device();
        let queue = device.wgpu_queue();

        let (bh, s, d) = (384u32, 256u32, 64u32);
        let scale = (d as f32).powf(-0.5);
        let mode = std::env::var("MODE").unwrap_or_else(|_| "mask".into());
        let causal = mode == "causal";
        let with_mask = mode == "mask";

        let q_host: Vec<f32> = (0..bh * s * d)
            .map(|i| ((i % 37) as f32) * 0.02 - 0.36)
            .collect();
        let k_host: Vec<f32> = (0..bh * s * d)
            .map(|i| ((i % 29) as f32) * 0.02 - 0.28)
            .collect();
        let v_host: Vec<f32> = (0..bh * s * d)
            .map(|i| ((i % 23) as f32) * 0.05 - 0.55)
            .collect();
        let mask_host: Vec<f32> = (0..s * s)
            .map(|i| if i % s > i / s { -1e9 } else { 0.0 })
            .collect();

        use wgpu::util::DeviceExt;
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let make = |data: &[f32]| {
            wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage,
            })
        };
        let q_buf = make(&q_host);
        let k_buf = make(&k_host);
        let v_buf = make(&v_host);
        let mask_buf = make(&mask_host);

        let subgroups = SubgroupConfig::fixed(fusor_tile_ir::SubgroupToken::new_unchecked(), 32);
        let coop = fusor_tile_ir::CoopMatrixToken::new_unchecked();
        let shape = FlashAttentionShape {
            bh,
            q_len: s,
            kv_len: s,
            head_dim: d,
            scale,
            causal,
        };

        let ir = tile::build(|phase| {
            let f32e = fusor_tile_ir::ScalarElement::F32.element();
            let q = phase.storage_read(f32e, fusor_tile_ir::Shape::new([bh * s, d]));
            let k = phase.storage_read(f32e, fusor_tile_ir::Shape::new([bh * s, d]));
            let v = phase.storage_read(f32e, fusor_tile_ir::Shape::new([bh * s, d]));
            let mask = with_mask
                .then(|| phase.storage_read(f32e, fusor_tile_ir::Shape::new([s, s])));
            let o = phase.storage_write(f32e, fusor_tile_ir::Shape::new([bh * s, d]));
            let ok = fusor_tile_ir_kernels::flash_attention_f32(
                phase,
                &q,
                &k,
                &v,
                mask.as_ref(),
                &o,
                shape,
                subgroups,
                coop,
                65535,
            );
            assert!(ok, "flash kernel rejected shape {shape:?}");
        });
        let grid = ir.grid;
        let kernel = ir.lower_to_naga().expect("lowering failed");
        let module = unsafe {
            wgpu_device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("flash_lab"),
                    source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
                        kernel.module().clone(),
                    )),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("flash_lab"),
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

        let o_bufs: Vec<_> = (0..8)
            .map(|_| {
                wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (bh as u64) * (s as u64) * (d as u64) * 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let binds: Vec<_> = o_bufs
            .iter()
            .map(|o_iter| {
                let mut entries = vec![
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: q_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: k_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: v_buf.as_entire_binding(),
                    },
                ];
                let mut next = 3;
                if with_mask {
                    entries.push(wgpu::BindGroupEntry {
                        binding: next,
                        resource: mask_buf.as_entire_binding(),
                    });
                    next += 1;
                }
                entries.push(wgpu::BindGroupEntry {
                    binding: next,
                    resource: o_iter.as_entire_binding(),
                });
                wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &entries,
                })
            })
            .collect();
        let dispatch = |iters: u32| {
            let mut encoder = wgpu_device.create_command_encoder(&Default::default());
            for i in 0..iters {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &binds[(i as usize) % binds.len()], &[]);
                pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
            }
            queue.submit([encoder.finish()]);
        };

        dispatch(3);
        device.poll_wait();

        // Verify sampled rows against a CPU reference.
        let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: o_bufs[0].size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = wgpu_device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&o_bufs[2], 0, &staging, 0, staging.size());
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll_wait();
        rx.recv().unwrap().unwrap();
        {
            let view = slice.get_mapped_range();
            let o_host: &[f32] = bytemuck::cast_slice(&view);
            let mut worst = 0f64;
            for (b, qi) in [
                (0u32, 0u32),
                (0, 31),
                (5, 100),
                (191, 255),
                (383, 17),
                (383, 255),
            ] {
                // Reference row: scores → softmax → ·V in f64.
                let mut scores = vec![0f64; s as usize];
                for c in 0..s {
                    let mut acc = 0f64;
                    for di in 0..d {
                        acc += q_host[((b * s + qi) * d + di) as usize] as f64
                            * k_host[((b * s + c) * d + di) as usize] as f64;
                    }
                    let mut score = acc * scale as f64;
                    if causal && c > qi {
                        score = f64::NEG_INFINITY;
                    }
                    if with_mask {
                        score += mask_host[(qi * s + c) as usize] as f64;
                    }
                    scores[c as usize] = score;
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = scores.iter().map(|v| (v - m).exp()).sum();
                for di in (0..d).step_by(13) {
                    let mut acc = 0f64;
                    for c in 0..s {
                        acc += (scores[c as usize] - m).exp() / sum
                            * v_host[((b * s + c) * d + di) as usize] as f64;
                    }
                    let got = o_host[((b * s + qi) * d + di) as usize] as f64;
                    worst = worst.max((got - acc).abs());
                    assert!(
                        (got - acc).abs() < 1e-3,
                        "mismatch at bh={b} q={qi} d={di}: got {got}, want {acc}"
                    );
                }
            }
            println!("verified ({mode}), worst abs err {worst:.2e}");
        }
        staging.unmap();

        let iters = 40;
        let start = std::time::Instant::now();
        dispatch(iters);
        device.poll_wait();
        let per = start.elapsed().as_secs_f64() / iters as f64;
        let flops = 4.0 * bh as f64 * s as f64 * s as f64 * d as f64
            * if causal { 0.5 } else { 1.0 };
        println!(
            "flash_lab {mode}: {:.3} ms, {:.2} TF/s",
            per * 1e3,
            flops / per / 1e12
        );
    });
}
