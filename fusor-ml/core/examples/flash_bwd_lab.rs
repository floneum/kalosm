//! Standalone lab for the flash-attention backward kernels: run lse →
//! bwd_q / bwd_kv raw (dsum computed on host, as the composed reduce would),
//! verify against a CPU f64 reference, and time each kernel at
//! [8, 3, 256, 64]. MODE=causal|mask|none selects the masking form.

use fusor_core::Device;
use fusor_tile_ir::tile;
use fusor_tile_ir_kernels::{
    FlashAttentionShape, FlashBwdLayouts, FlashMaskLayout, FlashOperandLayout, FlashRowLayout,
    SubgroupConfig,
};

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

const S: u32 = 256;
const D: u32 = 64;

struct Reference {
    lse: Vec<f64>,
    dsum: Vec<f64>,
    dq: Vec<f64>,
    dk: Vec<f64>,
    dv: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    b_n: usize,
    h_n: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    grad_o: &[f32],
    mask: Option<&[f32]>,
    causal: bool,
    scale: f64,
) -> Reference {
    let (s, d) = (S as usize, D as usize);
    let bh = b_n * h_n;
    let mut lse = vec![0f64; bh * s];
    let mut dsum = vec![0f64; bh * s];
    let mut dq = vec![0f64; bh * s * d];
    let mut dk = vec![0f64; bh * s * d];
    let mut dv = vec![0f64; bh * s * d];
    for g in 0..bh {
        let base = g * s * d;
        let mut p = vec![0f64; s * s];
        for qi in 0..s {
            let mut scores = vec![f64::NEG_INFINITY; s];
            for c in 0..s {
                if causal && c > qi {
                    continue;
                }
                let mut acc = 0f64;
                for di in 0..d {
                    acc += q[base + qi * d + di] as f64 * k[base + c * d + di] as f64;
                }
                let mut score = acc * scale;
                if let Some(mask) = mask {
                    score += mask[qi * s + c] as f64;
                }
                scores[c] = score;
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = scores.iter().map(|x| (x - m).exp()).sum();
            lse[g * s + qi] = m + sum.ln();
            for c in 0..s {
                p[qi * s + c] = (scores[c] - m).exp() / sum;
            }
        }
        // O = P·V feeds dsum = rowsum(dO ∘ O).
        for qi in 0..s {
            let mut row_dsum = 0f64;
            for di in 0..d {
                let mut o = 0f64;
                for c in 0..s {
                    o += p[qi * s + c] * v[base + c * d + di] as f64;
                }
                row_dsum += grad_o[base + qi * d + di] as f64 * o;
            }
            dsum[g * s + qi] = row_dsum;
        }
        for qi in 0..s {
            for c in 0..s {
                let mut dp = 0f64;
                for di in 0..d {
                    dp += grad_o[base + qi * d + di] as f64 * v[base + c * d + di] as f64;
                }
                let ds = p[qi * s + c] * (dp - dsum[g * s + qi]) * scale;
                for di in 0..d {
                    dq[base + qi * d + di] += ds * k[base + c * d + di] as f64;
                    dk[base + c * d + di] += ds * q[base + qi * d + di] as f64;
                    dv[base + c * d + di] +=
                        p[qi * s + c] * grad_o[base + qi * d + di] as f64;
                }
            }
        }
    }
    Reference {
        lse,
        dsum,
        dq,
        dk,
        dv,
    }
}

#[allow(non_snake_case)]
fn main() {
    pollster::block_on(async {
        let B = env_u32("B", 8);
        let H = env_u32("H", 3);
        let verify = std::env::var_os("TIME_ONLY").is_none();
        let device = Device::new().await.unwrap();
        let wgpu_device = device.wgpu_device();
        let queue = device.wgpu_queue();

        let scale = (D as f32).powf(-0.5);
        let mode = std::env::var("MODE").unwrap_or_else(|_| "causal".into());
        let causal = mode == "causal";
        let with_mask = mode == "mask";

        let value = |seed: u32| {
            (0..B * H * S * D)
                .map(|i| ((i.wrapping_mul(seed) % 61) as f32) * 0.02 - 0.6)
                .collect::<Vec<f32>>()
        };
        let q_host = value(11);
        let k_host = value(7);
        let v_host = value(13);
        let do_host = value(29);
        let mask_host: Vec<f32> = (0..S * S)
            .map(|i| if i % 5 == 0 { -1.25 } else { 0.125 })
            .collect();

        let reference = verify.then(|| {
            println!("computing CPU reference...");
            cpu_reference(
                B as usize,
                H as usize,
                &q_host,
                &k_host,
                &v_host,
                &do_host,
                with_mask.then_some(mask_host.as_slice()),
                causal,
                scale as f64,
            )
        });
        let dsum_host: Vec<f32> = match &reference {
            Some(reference) => reference.dsum.iter().map(|&x| x as f32).collect(),
            None => vec![0.0; (B * H * S) as usize],
        };

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
        let do_buf = make(&do_host);
        let mask_buf = make(&mask_host);
        let dsum_buf = make(&dsum_host);
        let out_buffer = |elems: u64| {
            wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: elems * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let lse_buf = out_buffer((B * H * S) as u64);
        let dq_buf = out_buffer((B * H * S * D) as u64);
        let dkv_buf = out_buffer((B * H * 2 * S * D) as u64);

        let subgroups = SubgroupConfig::fixed(fusor_tile_ir::SubgroupToken::new_unchecked(), 32);
        let coop = fusor_tile_ir::CoopMatrixToken::new_unchecked();
        let shape = FlashAttentionShape {
            batch: B,
            heads: H,
            kv_groups: 1,
            q_len: S,
            kv_len: S,
            head_dim: D,
            scale,
            causal,
        };
        let operand = FlashOperandLayout::contiguous(H, S, D);
        let row = FlashRowLayout {
            offset: 0,
            batch_stride: H * S,
            head_stride: S,
            seq_stride: 1,
        };
        let mask_layout = FlashMaskLayout {
            offset: 0,
            q_stride: S,
            kv_stride: 1,
        };
        let layouts = FlashBwdLayouts {
            q: operand,
            k: operand,
            v: operand,
            grad_o: operand,
            lse: row,
            dsum: row,
            out: operand,
        };
        let dkv_layouts = FlashBwdLayouts {
            out: FlashOperandLayout::contiguous(H, 2 * S, D),
            ..layouts
        };

        let f32e = fusor_tile_ir::ScalarElement::F32.element();
        let n4 = B * H * S * D;
        let build = |emit: &dyn Fn(&mut fusor_tile_ir::tile::Program) -> bool| {
            let ir = tile::build(|phase| {
                assert!(emit(phase), "kernel rejected shape");
            });
            let grid = ir.grid;
            let kernel = ir.lower_to_naga().expect("lowering failed");
            let module = unsafe {
                wgpu_device.create_shader_module_trusted(
                    wgpu::ShaderModuleDescriptor {
                        label: Some("flash_bwd_lab"),
                        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(
                            kernel.module().clone(),
                        )),
                    },
                    wgpu::ShaderRuntimeChecks::unchecked(),
                )
            };
            let pipeline =
                wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("flash_bwd_lab"),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                    cache: None,
                });
            (pipeline, grid)
        };

        // LSE kernel: q, k, [mask], lse.
        let (lse_pipeline, lse_grid) = build(&|phase| {
            let q = phase.storage_read(f32e, fusor_tile_ir::Shape::new([n4]));
            let k = phase.storage_read(f32e, fusor_tile_ir::Shape::new([n4]));
            let mask =
                with_mask.then(|| phase.storage_read(f32e, fusor_tile_ir::Shape::new([S * S])));
            let lse = phase.storage_write(f32e, fusor_tile_ir::Shape::new([B * H * S]));
            fusor_tile_ir_kernels::flash_lse_f32(
                phase,
                &q,
                &k,
                mask.as_ref().map(|m| (m, mask_layout)),
                &lse,
                operand,
                operand,
                row,
                shape,
                subgroups,
                coop,
                65535,
            )
        });
        // bwd kernels: q, k, v, do, lse, dsum, [mask], out.
        let bwd_build = |kv: bool| {
            build(&|phase| {
                let q = phase.storage_read(f32e, fusor_tile_ir::Shape::new([n4]));
                let k = phase.storage_read(f32e, fusor_tile_ir::Shape::new([n4]));
                let v = phase.storage_read(f32e, fusor_tile_ir::Shape::new([n4]));
                let go = phase.storage_read(f32e, fusor_tile_ir::Shape::new([n4]));
                let lse = phase.storage_read(f32e, fusor_tile_ir::Shape::new([B * H * S]));
                let dsum = phase.storage_read(f32e, fusor_tile_ir::Shape::new([B * H * S]));
                let mask = with_mask
                    .then(|| phase.storage_read(f32e, fusor_tile_ir::Shape::new([S * S])));
                if kv {
                    let dkv = phase.storage_write(f32e, fusor_tile_ir::Shape::new([2 * n4]));
                    fusor_tile_ir_kernels::flash_bwd_kv_f32(
                        phase,
                        &q,
                        &k,
                        &v,
                        &go,
                        &lse,
                        &dsum,
                        mask.as_ref().map(|m| (m, mask_layout)),
                        &dkv,
                        &dkv_layouts,
                        shape,
                        subgroups,
                        coop,
                        65535,
                    )
                } else {
                    let dq = phase.storage_write(f32e, fusor_tile_ir::Shape::new([n4]));
                    fusor_tile_ir_kernels::flash_bwd_q_f32(
                        phase,
                        &q,
                        &k,
                        &v,
                        &go,
                        &lse,
                        &dsum,
                        mask.as_ref().map(|m| (m, mask_layout)),
                        &dq,
                        &layouts,
                        shape,
                        subgroups,
                        coop,
                        65535,
                    )
                }
            })
        };
        let (dq_pipeline, dq_grid) = bwd_build(false);
        let (dkv_pipeline, dkv_grid) = bwd_build(true);

        let bind = |pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]| {
            let layout = pipeline.get_bind_group_layout(0);
            let entries: Vec<_> = buffers
                .iter()
                .enumerate()
                .map(|(i, buffer)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &entries,
            })
        };
        let lse_bind = if with_mask {
            bind(&lse_pipeline, &[&q_buf, &k_buf, &mask_buf, &lse_buf])
        } else {
            bind(&lse_pipeline, &[&q_buf, &k_buf, &lse_buf])
        };
        let dq_bind = if with_mask {
            bind(
                &dq_pipeline,
                &[&q_buf, &k_buf, &v_buf, &do_buf, &lse_buf, &dsum_buf, &mask_buf, &dq_buf],
            )
        } else {
            bind(
                &dq_pipeline,
                &[&q_buf, &k_buf, &v_buf, &do_buf, &lse_buf, &dsum_buf, &dq_buf],
            )
        };
        let dkv_bind = if with_mask {
            bind(
                &dkv_pipeline,
                &[&q_buf, &k_buf, &v_buf, &do_buf, &lse_buf, &dsum_buf, &mask_buf, &dkv_buf],
            )
        } else {
            bind(
                &dkv_pipeline,
                &[&q_buf, &k_buf, &v_buf, &do_buf, &lse_buf, &dsum_buf, &dkv_buf],
            )
        };

        let run = |passes: &[(&wgpu::ComputePipeline, &wgpu::BindGroup, [u32; 3])], iters: u32| {
            let mut encoder = wgpu_device.create_command_encoder(&Default::default());
            for _ in 0..iters {
                for (pipeline, bind_group, grid) in passes {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, *bind_group, &[]);
                    pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
                }
            }
            queue.submit([encoder.finish()]);
        };
        run(
            &[
                (&lse_pipeline, &lse_bind, lse_grid),
                (&dq_pipeline, &dq_bind, dq_grid),
                (&dkv_pipeline, &dkv_bind, dkv_grid),
            ],
            1,
        );
        device.poll_wait();

        // Read back and verify.
        let read = |buffer: &wgpu::Buffer| {
            let staging = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: buffer.size(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = wgpu_device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, staging.size());
            queue.submit([encoder.finish()]);
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
            device.poll_wait();
            rx.recv().unwrap().unwrap();
            let host: Vec<f32> = {
                let view = slice.get_mapped_range();
                bytemuck::cast_slice(&view).to_vec()
            };
            staging.unmap();
            host
        };
        if let Some(reference) = &reference {
        let lse_host = read(&lse_buf);
        let dq_host = read(&dq_buf);
        let dkv_host = read(&dkv_buf);

        let check = |name: &str, got: &[f32], want: &[f64], tolerance: f64| {
            let mut worst = 0f64;
            let step = (want.len() / 3001).max(1);
            for i in (0..want.len()).step_by(step) {
                let err = (got[i] as f64 - want[i]).abs();
                worst = worst.max(err);
                assert!(
                    err < tolerance,
                    "{name} mismatch at {i}: got {}, want {}",
                    got[i],
                    want[i]
                );
            }
            println!("verified {name}, worst abs err {worst:.2e}");
        };
        check("lse", &lse_host, &reference.lse, 1e-3);
        check("dq", &dq_host, &reference.dq, 5e-3);
        let n = (B * H * S * D) as usize;
        let (dk_got, dv_got): (Vec<f32>, Vec<f32>) = {
            // dkv is [B, H, 2S, D]: per (b, h), dk rows then dv rows.
            let (bh, sd) = ((B * H) as usize, (S * D) as usize);
            let mut dk = vec![0f32; n];
            let mut dv = vec![0f32; n];
            for g in 0..bh {
                dk[g * sd..(g + 1) * sd]
                    .copy_from_slice(&dkv_host[g * 2 * sd..g * 2 * sd + sd]);
                dv[g * sd..(g + 1) * sd]
                    .copy_from_slice(&dkv_host[g * 2 * sd + sd..(g + 1) * 2 * sd]);
            }
            (dk, dv)
        };
        check("dk", &dk_got, &reference.dk, 5e-3);
        check("dv", &dv_got, &reference.dv, 5e-3);
        }

        // Time each kernel.
        for (name, pipeline, bind_group, grid) in [
            ("lse", &lse_pipeline, &lse_bind, lse_grid),
            ("bwd_q", &dq_pipeline, &dq_bind, dq_grid),
            ("bwd_kv", &dkv_pipeline, &dkv_bind, dkv_grid),
        ] {
            let iters = 40;
            let start = std::time::Instant::now();
            run(&[(pipeline, bind_group, grid)], iters);
            device.poll_wait();
            let per = start.elapsed().as_secs_f64() / iters as f64;
            println!("flash_bwd_lab {mode} {name}: {:.3} ms", per * 1e3);
        }
    });
}
