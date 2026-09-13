//! Apples-to-apples kernel comparison: one device, one encoder policy, GPU
//! timestamps around warmed batches. No graph building or readback in the span.
use crate::{emit, graph::Graph, plan, run};
use fusor_gpu::GpuDevice;
use std::time::Instant;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Timer {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    staging: wgpu::Buffer,
}
impl Timer {
    fn new(gpu: &GpuDevice) -> Result<Self> {
        if !gpu.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err("benchmark requires GPU timestamps".into());
        }
        let buffer = |usage| {
            gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("prototype benchmark timestamps"),
                size: 256,
                usage,
                mapped_at_creation: false,
            })
        };
        Ok(Self {
            queries: gpu.device().create_query_set(&wgpu::QuerySetDescriptor {
                label: None,
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            }),
            resolved: buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
            staging: buffer(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
        })
    }
    fn sample(
        &self,
        gpu: &GpuDevice,
        iterations: usize,
        encode: impl Fn(&mut wgpu::ComputePass<'_>),
    ) -> Result<(f64, f64)> {
        let start = Instant::now();
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matched benchmark batch"),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                }),
            });
            for _ in 0..iterations {
                encode(&mut pass);
            }
        }
        gpu.queue().submit([encoder.finish()]);
        gpu.device().poll(wgpu::PollType::wait_indefinitely())?;
        let host_us = start.elapsed().as_secs_f64() * 1e6 / iterations as f64;
        // Resolve after GPU completion: Metal boundary timestamps otherwise
        // can race query resolution. This readback is outside both timings.
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        encoder.resolve_query_set(&self.queries, 0..2, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &self.staging, 0, 256);
        gpu.queue().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        self.staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
        gpu.device().poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let bytes = self.staging.slice(..).get_mapped_range();
        let first = u64::from_le_bytes(bytes[0..8].try_into()?);
        let last = u64::from_le_bytes(bytes[8..16].try_into()?);
        drop(bytes);
        self.staging.unmap();
        if first == 0 || last <= first {
            return Err("invalid GPU timestamps".into());
        }
        Ok((
            (last - first) as f64 * gpu.queue().get_timestamp_period() as f64
                / 1000.0
                / iterations as f64,
            host_us,
        ))
    }
}
fn summarize(name: &str, samples: &[(f64, f64)]) -> f64 {
    let mut gpu: Vec<f64> = samples.iter().map(|s| s.0).collect();
    let mut host: Vec<f64> = samples.iter().map(|s| s.1).collect();
    gpu.sort_by(f64::total_cmp);
    host.sort_by(f64::total_cmp);
    let median = |xs: &[f64]| (xs[(xs.len() - 1) / 2] + xs[xs.len() / 2]) / 2.0;
    println!(
        "  {name}: GPU median={:.3} us/step range={:.3}..{:.3}; host median={:.3} us/step",
        median(&gpu),
        gpu[0],
        gpu[gpu.len() - 1],
        median(&host)
    );
    println!(
        "    GPU samples: {}",
        samples
            .iter()
            .map(|s| format!("{:.3}", s.0))
            .collect::<Vec<_>>()
            .join(", ")
    );
    median(&gpu)
}
pub fn compare(g: &Graph, cfg: &plan::Config, iterations: usize) -> Result<()> {
    let device = fusor::Device::gpu_blocking()?.isolated()?;
    let target = device
        .backend()
        .gpu_target()
        .ok_or("GPU backend required")?;
    let gpu = target.device();
    let device_cfg = cfg.for_device(gpu);
    let cfg = &device_cfg;
    println!(
        "\nBENCH {} {:?} GPU={} iterations={iterations} rounds=12 shared_cap={} B",
        g.name,
        g.values[0].shape,
        gpu.adapter_info().name,
        cfg.shared_bytes
    );
    let roots = run::baseline_graph(g, &device)?;
    // A failed capture must release its state before the next compilation.
    assert!(
        target
            .launcher()
            .capture_executable(|| {
                Err(fusor::Error::Plan(
                    "intentional capture cancellation".into(),
                ))
            })
            .is_err()
    );
    let baseline = match target
        .launcher()
        .capture_executable(|| device.session().resolve(&roots))
    {
        Ok(program) => Some(program),
        Err(e) => {
            println!("  Existing Fusor: ERROR {e}");
            None
        }
    };
    device.session().wait()?;
    let baseline_ok = baseline.is_some();
    // Unlike raw command capture, this executable owns the pool leases.
    // Force pool churn and overwrite the new allocations before replaying.
    if let Some(program) = &baseline {
        let mut churn = vec![];
        for _ in 0..3 {
            for held in program.retained_buffers() {
                let held = held.downcast_ref::<fusor_gpu::pool::GpuBuffer>().unwrap();
                let fresh = target.pool().alloc_with_usage(held.size, held.usage)?;
                assert!(
                    !program
                        .retained_buffers()
                        .iter()
                        .any(|b| b.addr() == fresh.addr()),
                    "pool reused a live executable binding"
                );
                if held.usage.contains(wgpu::BufferUsages::COPY_DST) {
                    let raw = fresh.downcast_ref::<fusor_gpu::pool::GpuBuffer>().unwrap();
                    gpu.queue()
                        .write_buffer(&raw.buffer, 0, &vec![0xcd; held.size as usize]);
                }
                churn.push(fresh);
            }
        }
        for buffer in churn {
            target.pool().recycle(buffer);
        }
        println!(
            "  Frozen baseline: {} retained buffers; pool churn PASS",
            program.retained_buffers().len()
        );
    }
    if let Some(program) = &baseline {
        // Built-in cases have exactly one external input. Update the retained
        // input in place, submit without re-resolving, and read both outputs.
        assert_eq!(program.inputs().len(), 1);
        let input = &program.inputs()[0];
        for seed in 0..3 {
            program.write(input, 0, bytemuck::cast_slice(&g.input_data(seed)[0].1))?;
            program.submit();
            device.session().wait()?;
            let got = roots
                .iter()
                .map(|t| t.to_vec_f32())
                .collect::<fusor::Result<Vec<_>>>()?;
            run::compare(g, seed, &got);
        }
        let size = input
            .downcast_ref::<fusor_gpu::pool::GpuBuffer>()
            .unwrap()
            .size;
        assert!(program.write(input, size, &[0; 4]).is_err());
        assert!(program.write(input, 1, &[0; 4]).is_err());
        for held in program.retained_buffers() {
            if held.addr() != input.addr() {
                assert!(program.write(held, 0, &[0; 4]).is_err());
            }
        }
        program.write(input, 0, bytemuck::cast_slice(&g.input_data(0)[0].1))?;
        println!("  Frozen baseline: changing inputs and update bounds PASS (3 inputs)");
        assert!(
            target
                .launcher()
                .capture_executable(|| device.session().resolve(&roots))
                .is_err()
        );
        assert!(target.launcher().capture_executable(|| Ok(())).is_err());
        println!("  Frozen baseline: failed/empty/resolved capture rejection and cleanup PASS");
    }
    let start = Instant::now();
    let p = plan::compile(g, cfg)?;
    let planning_ms = start.elapsed().as_secs_f64() * 1000.0;
    let sources: Vec<String> = p.regions.iter().map(|r| emit::shader(g, &p, r)).collect();
    let executable = run::Executable::build(gpu, g, &p, &sources)?;
    for seed in 0..3 {
        executable.check(gpu, g, &p, seed)?;
    }
    println!(
        "  Prototype: {} dispatches; shared_peak={} B; global={} B; plan={planning_ms:.3} ms; correctness=PASS (3 inputs)",
        p.regions.len(),
        p.regions
            .iter()
            .map(|r| r.shared.len * 4)
            .max()
            .unwrap_or(0),
        p.global.len * 4
    );
    let timer = Timer::new(gpu)?;
    let mut legacy_cfg = cfg.clone();
    legacy_cfg.legacy = true;
    legacy_cfg.forwarding = false;
    legacy_cfg.serial_limit = 8;
    legacy_cfg.subgroup_width = None;
    let legacy_plan = plan::compile(g, &legacy_cfg)?;
    let sources: Vec<String> = legacy_plan
        .regions
        .iter()
        .map(|r| emit::shader(g, &legacy_plan, r))
        .collect();
    let legacy = run::Executable::build(gpu, g, &legacy_plan, &sources)?;
    for seed in 0..3 {
        legacy.check(gpu, g, &legacy_plan, seed)?;
    }
    println!(
        "  Legacy policies: {} dispatches; shared_peak={} B; global={} B; correctness=PASS (3 inputs)",
        legacy_plan.regions.len(),
        legacy_plan
            .regions
            .iter()
            .map(|r| r.shared.len * 4)
            .max()
            .unwrap_or(0),
        legacy_plan.global.len * 4
    );
    let mut portable_cfg = cfg.clone();
    portable_cfg.subgroup_width = None;
    let portable_plan = plan::compile(g, &portable_cfg)?;
    let sources: Vec<String> = portable_plan
        .regions
        .iter()
        .map(|r| emit::shader(g, &portable_plan, r))
        .collect();
    let portable = run::Executable::build(gpu, g, &portable_plan, &sources)?;
    for seed in 0..3 {
        portable.check(gpu, g, &portable_plan, seed)?;
    }
    println!(
        "  Portable tree: {} dispatches; shared_peak={} B; global={} B; correctness=PASS (3 inputs)",
        portable_plan.regions.len(),
        portable_plan
            .regions
            .iter()
            .map(|r| r.shared.len * 4)
            .max()
            .unwrap_or(0),
        portable_plan.global.len * 4
    );
    // Equal inputs, then warm all compiled plans. The baseline graph/pool is
    // retained by the executable; pool churn above cannot invalidate bindings.
    executable.check(gpu, g, &p, 0)?;
    legacy.check(gpu, g, &legacy_plan, 0)?;
    portable.check(gpu, g, &portable_plan, 0)?;
    for _ in 0..5 {
        timer.sample(gpu, iterations, |pass| executable.encode(pass))?;
        timer.sample(gpu, iterations, |pass| legacy.encode(pass))?;
        timer.sample(gpu, iterations, |pass| portable.encode(pass))?;
        if baseline_ok {
            timer.sample(gpu, iterations, |pass| {
                baseline.as_ref().unwrap().encode(pass)
            })?;
        }
    }
    let mut prototype_samples = vec![];
    let mut baseline_samples = vec![];
    let mut legacy_samples = vec![];
    let mut portable_samples = vec![];
    for round in 0..12 {
        for offset in 0..4 {
            let side = (round + offset) % 4;
            if side == 0 {
                prototype_samples
                    .push(timer.sample(gpu, iterations, |pass| executable.encode(pass))?);
            } else if side == 1 && baseline_ok {
                baseline_samples.push(timer.sample(gpu, iterations, |pass| {
                    baseline.as_ref().unwrap().encode(pass)
                })?);
            } else if side == 2 {
                legacy_samples.push(timer.sample(gpu, iterations, |pass| legacy.encode(pass))?);
            } else if side == 3 {
                portable_samples.push(timer.sample(gpu, iterations, |pass| portable.encode(pass))?);
            }
        }
    }
    let proto = summarize("Prototype", &prototype_samples);
    let original = summarize("Legacy policies", &legacy_samples);
    println!("  legacy/optimized GPU ratio={:.3}x", original / proto);
    let tree = summarize("Portable tree", &portable_samples);
    println!("  portable/prototype GPU ratio={:.3}x", tree / proto);
    if baseline_ok {
        let old = summarize("Existing Fusor", &baseline_samples);
        println!(
            "  existing/prototype GPU ratio={:.3}x; baseline dispatches={}",
            old / proto,
            baseline.as_ref().unwrap().dispatches()
        );
        let got = roots
            .iter()
            .map(|t| t.to_vec_f32())
            .collect::<fusor::Result<Vec<_>>>()?;
        run::compare(g, 0, &got);
        println!("  Captured baseline replay correctness: PASS");
    }
    Ok(())
}
