//! Apples-to-apples kernel comparison: one device, one encoder policy, GPU
//! timestamps around warmed batches. No graph building or readback in the span.
use crate::{emit, graph::Graph, plan, run};
use fusor_gpu::{GpuDevice, launch::CommandRecord};
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
fn encode_baseline(records: &[CommandRecord], pass: &mut wgpu::ComputePass<'_>) {
    for record in records {
        match record {
            CommandRecord::Dispatch {
                pipeline,
                bind_group,
                grid,
                ..
            } => {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group.as_ref(), &[]);
                pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
            }
            _ => unreachable!("copy commands rejected before measurement"),
        }
    }
}
fn summarize(name: &str, samples: &[(f64, f64)]) -> f64 {
    let mut gpu: Vec<f64> = samples.iter().map(|s| s.0).collect();
    let mut host: Vec<f64> = samples.iter().map(|s| s.1).collect();
    gpu.sort_by(f64::total_cmp);
    host.sort_by(f64::total_cmp);
    println!(
        "  {name}: GPU median={:.3} us/step range={:.3}..{:.3}; host median={:.3} us/step",
        gpu[gpu.len() / 2],
        gpu[0],
        gpu[gpu.len() - 1],
        host[host.len() / 2]
    );
    println!(
        "    GPU samples: {}",
        samples
            .iter()
            .map(|s| format!("{:.3}", s.0))
            .collect::<Vec<_>>()
            .join(", ")
    );
    gpu[gpu.len() / 2]
}
pub fn compare(g: &Graph, cfg: &plan::Config, iterations: usize) -> Result<()> {
    let device = fusor::Device::gpu_blocking()?.isolated()?;
    let target = device
        .backend()
        .gpu_target()
        .ok_or("GPU backend required")?;
    let gpu = target.device();
    println!(
        "\nBENCH {} {:?} GPU={} iterations={iterations} rounds=9 shared_cap={} B",
        g.name,
        g.values[0].shape,
        gpu.adapter_info().name,
        cfg.shared_bytes
    );
    let roots = run::baseline_graph(g, &device)?;
    target.launcher().begin_prototype_capture();
    let baseline_result = device.session().resolve(&roots);
    device.session().wait()?;
    let records = target.launcher().end_prototype_capture();
    let baseline_ok = match baseline_result {
        Ok(())
            if !records.is_empty()
                && records
                    .iter()
                    .all(|r| matches!(r, CommandRecord::Dispatch { .. })) =>
        {
            true
        }
        Ok(()) => {
            println!("  Existing Fusor: unsupported empty/copy capture");
            false
        }
        Err(e) => {
            println!("  Existing Fusor: ERROR {e}");
            false
        }
    };
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
        "  Original prototype: {} dispatches; shared_peak={} B; global={} B; correctness=PASS (3 inputs)",
        legacy_plan.regions.len(),
        legacy_plan
            .regions
            .iter()
            .map(|r| r.shared.len * 4)
            .max()
            .unwrap_or(0),
        legacy_plan.global.len * 4
    );
    // Equal inputs, then warm both compiled plans. The baseline graph/pool is
    // held untouched until replay ends; all benchmark allocations use WGPU.
    executable.check(gpu, g, &p, 0)?;
    legacy.check(gpu, g, &legacy_plan, 0)?;
    for _ in 0..5 {
        timer.sample(gpu, iterations, |pass| executable.encode(pass))?;
        timer.sample(gpu, iterations, |pass| legacy.encode(pass))?;
        if baseline_ok {
            timer.sample(gpu, iterations, |pass| encode_baseline(&records, pass))?;
        }
    }
    let mut prototype_samples = vec![];
    let mut baseline_samples = vec![];
    let mut legacy_samples = vec![];
    for round in 0..9 {
        for offset in 0..3 {
            let side = if round % 2 == 0 {
                (round + offset) % 3
            } else {
                (round + 3 - offset) % 3
            };
            if side == 0 {
                prototype_samples
                    .push(timer.sample(gpu, iterations, |pass| executable.encode(pass))?);
            } else if side == 1 && baseline_ok {
                baseline_samples
                    .push(timer.sample(gpu, iterations, |pass| encode_baseline(&records, pass))?);
            } else if side == 2 {
                legacy_samples.push(timer.sample(gpu, iterations, |pass| legacy.encode(pass))?);
            }
        }
    }
    let proto = summarize("Prototype", &prototype_samples);
    let original = summarize("Original prototype", &legacy_samples);
    println!("  original/optimized GPU ratio={:.3}x", original / proto);
    if baseline_ok {
        let old = summarize("Existing Fusor", &baseline_samples);
        println!(
            "  existing/prototype GPU ratio={:.3}x; baseline dispatches={}",
            old / proto,
            records.len()
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
