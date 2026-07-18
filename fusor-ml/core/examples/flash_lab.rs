//! Standalone lab for the fused flash-attention kernel: build, dispatch raw,
//! verify sampled rows against a CPU reference, and time at the training
//! shape ([64, 6, 256, 64]).
//!
//! Env: MODE=causal|mask|none selects the masking form; LAYOUT=bhsd|bshd
//! selects the physical operand layout (bshd exercises the strided path);
//! GROUPS=n (dividing the head count) exercises grouped-query K/V.

use fusor_core::Device;
use fusor_tile_ir::tile;
use fusor_tile_ir_kernels::{
    FlashAttentionLayouts, FlashAttentionShape, FlashMaskLayout, FlashOperandLayout,
    SubgroupConfig,
};

fn main() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();
        let wgpu_device = device.wgpu_device();
        let queue = device.wgpu_queue();

        let (b_n, h_n, s, d) = (64u32, 6u32, 256u32, 64u32);
        let scale = (d as f32).powf(-0.5);
        let mode = std::env::var("MODE").unwrap_or_else(|_| "mask".into());
        let causal = mode == "causal";
        let with_mask = mode == "mask";
        let layout_kind = std::env::var("LAYOUT").unwrap_or_else(|_| "bhsd".into());
        let groups: u32 = std::env::var("GROUPS")
            .ok()
            .and_then(|g| g.parse().ok())
            .unwrap_or(1);
        assert_eq!(h_n % groups, 0, "GROUPS must divide the head count");
        let hk_n = h_n / groups;

        // Deterministic logical values addressed by [b, h, s, d].
        let logical = |seed: u32, modulus: u32, scale: f32, shift: f32| {
            move |b: u32, h: u32, si: u32, di: u32| {
                let flat = ((b * 64 + h) * s + si) * d + di;
                ((flat.wrapping_mul(seed) % modulus) as f32) * scale - shift
            }
        };
        let q_val = logical(11, 37, 0.02, 0.36);
        let k_val = logical(7, 29, 0.02, 0.28);
        let v_val = logical(13, 23, 0.05, 0.55);
        let mask_val = |qi: u32, c: u32| if c > qi { -1e9f32 } else { 0.0 };

        // Physical layouts: bhsd is contiguous; bshd stores [batch, seq,
        // heads, dim] and views it as [batch, heads, seq, dim] via strides.
        let layout_for = |heads: u32| match layout_kind.as_str() {
            "bhsd" => FlashOperandLayout::contiguous(heads, s, d),
            "bshd" => FlashOperandLayout {
                offset: 0,
                batch_stride: s * heads * d,
                head_stride: d,
                seq_stride: heads * d,
                dim_stride: 1,
            },
            other => panic!("unknown LAYOUT {other}"),
        };
        let q_layout = layout_for(h_n);
        let kv_layout = layout_for(hk_n);
        let o_layout = FlashOperandLayout::contiguous(h_n, s, d);
        let scatter = |heads: u32,
                       layout: &FlashOperandLayout,
                       value: &dyn Fn(u32, u32, u32, u32) -> f32| {
            let mut host = vec![0f32; (b_n * heads * s * d) as usize];
            for b in 0..b_n {
                for h in 0..heads {
                    for si in 0..s {
                        for di in 0..d {
                            let idx = layout.offset
                                + b * layout.batch_stride
                                + h * layout.head_stride
                                + si * layout.seq_stride
                                + di * layout.dim_stride;
                            host[idx as usize] = value(b, h, si, di);
                        }
                    }
                }
            }
            host
        };
        let q_host = scatter(h_n, &q_layout, &q_val);
        let k_host = scatter(hk_n, &kv_layout, &k_val);
        let v_host = scatter(hk_n, &kv_layout, &v_val);
        let mask_layout = FlashMaskLayout {
            offset: 0,
            q_stride: s,
            kv_stride: 1,
        };
        let mask_host: Vec<f32> = (0..s * s).map(|i| mask_val(i / s, i % s)).collect();

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
            batch: b_n,
            heads: h_n,
            kv_groups: groups,
            q_len: s,
            kv_len: s,
            head_dim: d,
            scale,
            causal,
        };
        let layouts = FlashAttentionLayouts {
            q: q_layout,
            k: kv_layout,
            v: kv_layout,
            o: o_layout,
        };

        let ir = tile::build(|phase| {
            let f32e = fusor_tile_ir::ScalarElement::F32.element();
            let q = phase.storage_read(f32e, fusor_tile_ir::Shape::new([b_n * h_n * s * d]));
            let k = phase.storage_read(f32e, fusor_tile_ir::Shape::new([b_n * hk_n * s * d]));
            let v = phase.storage_read(f32e, fusor_tile_ir::Shape::new([b_n * hk_n * s * d]));
            let mask = with_mask
                .then(|| phase.storage_read(f32e, fusor_tile_ir::Shape::new([s * s])));
            let o = phase.storage_write(f32e, fusor_tile_ir::Shape::new([b_n * h_n * s * d]));
            let ok = fusor_tile_ir_kernels::flash_attention_f32(
                phase,
                &q,
                &k,
                &v,
                mask.as_ref().map(|m| (m, mask_layout)),
                &o,
                &layouts,
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
                    size: (b_n as u64) * (h_n as u64) * (s as u64) * (d as u64) * 4,
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
            for (b, h, qi) in [
                (0u32, 0u32, 0u32),
                (0, 0, 31),
                (0, 5, 100),
                (31, 2, 255),
                (63, 4, 17),
                (63, 5, 255),
            ] {
                // Reference row: scores → softmax → ·V in f64.
                let hk = h / groups;
                let mut scores = vec![0f64; s as usize];
                for c in 0..s {
                    let mut acc = 0f64;
                    for di in 0..d {
                        acc += q_val(b, h, qi, di) as f64 * k_val(b, hk, c, di) as f64;
                    }
                    let mut score = acc * scale as f64;
                    if causal && c > qi {
                        score = f64::NEG_INFINITY;
                    }
                    if with_mask {
                        score += mask_val(qi, c) as f64;
                    }
                    scores[c as usize] = score;
                }
                let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = scores.iter().map(|v| (v - m).exp()).sum();
                for di in (0..d).step_by(13) {
                    let mut acc = 0f64;
                    for c in 0..s {
                        acc += (scores[c as usize] - m).exp() / sum * v_val(b, hk, c, di) as f64;
                    }
                    let got = o_host[(((b * h_n + h) * s + qi) * d + di) as usize] as f64;
                    worst = worst.max((got - acc).abs());
                    assert!(
                        (got - acc).abs() < 1e-3,
                        "mismatch at b={b} h={h} q={qi} d={di}: got {got}, want {acc}"
                    );
                }
            }
            println!("verified ({mode}, {layout_kind}, groups={groups}), worst abs err {worst:.2e}");
        }
        staging.unmap();

        let iters = 40;
        let start = std::time::Instant::now();
        dispatch(iters);
        device.poll_wait();
        let per = start.elapsed().as_secs_f64() / iters as f64;
        let flops = 4.0 * (b_n * h_n) as f64 * s as f64 * s as f64 * d as f64
            * if causal { 0.5 } else { 1.0 };
        println!(
            "flash_lab {mode} {layout_kind} g{groups}: {:.3} ms, {:.2} TF/s",
            per * 1e3,
            flops / per / 1e12
        );
    });
}
