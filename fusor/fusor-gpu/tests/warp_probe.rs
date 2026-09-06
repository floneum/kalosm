//! Device-removal probe.
//!
//! On DX12/WARP the sampling suite loses the device on one kernel shape:
//! a contraction whose activation `pre` reads four n×n matrices, so the
//! dispatch binds the uniforms, one vector, four read-only matrices and the
//! output. Everything else in the suite passes. This test runs that exact
//! dumped kernel, its passing sibling, and a synthetic family that varies
//! one property at a time, each on a fresh device, and reports which ones
//! remove the device. On every other backend all of them pass, which keeps
//! the harness honest.

use std::borrow::Cow;

use fusor_gpu::{GpuDevice, removed_reason};

struct Variant {
    label: String,
    wgsl: String,
    /// One size per binding, in binding order.
    sizes: Vec<u64>,
    grid: u32,
}

/// `srvs` read-only `array<f32>` bindings after the uniforms, one
/// `read_write` output, a per-lane clamped load of every input, and either
/// a subgroup sum or a plain store.
fn synthetic(srvs: usize, subgroup: bool, clamp: bool) -> Variant {
    let mut w = String::from("@group(0) @binding(0)\nvar<storage> global: array<u32>;\n");
    for i in 1..=srvs {
        w.push_str(&format!(
            "@group(0) @binding({i})\nvar<storage> global_{i}: array<f32>;\n"
        ));
    }
    let out = srvs + 1;
    w.push_str(&format!(
        "@group(0) @binding({out})\nvar<storage, read_write> global_{out}: array<f32>;\n\n"
    ));
    w.push_str("@compute @workgroup_size(4, 1, 1)\nfn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {\n");
    w.push_str("    var acc: f32 = 0f;\n    let i: u32 = (wg.x * 4u) + lane;\n");
    for i in 1..=srvs {
        if clamp {
            w.push_str(&format!(
                "    acc = acc + global_{i}[min(i, (arrayLength((&global_{i})) - 1u))];\n"
            ));
        } else {
            w.push_str(&format!("    acc = acc + global_{i}[i];\n"));
        }
    }
    if subgroup {
        w.push_str("    let total: f32 = subgroupAdd(acc);\n");
    } else {
        w.push_str("    let total: f32 = acc;\n");
    }
    w.push_str(&format!(
        "    if (lane == 0u) {{\n        global_{out}[wg.x] = total;\n    }}\n    return;\n}}\n"
    ));
    Variant {
        label: format!("synthetic srvs={srvs} subgroup={subgroup} clamp={clamp}"),
        wgsl: w,
        sizes: vec![8192; srvs + 2],
        grid: 36,
    }
}

fn variants() -> Vec<Variant> {
    let pairs = include_str!("data/warp_sgemv_cols_pairs.wgsl");
    let outer = include_str!("data/warp_sgemv_cols_outer.wgsl");
    let mut v = vec![
        // The passing sibling first: the harness itself must be clean.
        Variant {
            label: "dump outer product (passes on WARP in CI)".into(),
            wgsl: outer.into(),
            sizes: vec![4, 144, 144, 5184],
            grid: 648,
        },
        Variant {
            label: "dump pairs contraction, exact sizes".into(),
            wgsl: pairs.into(),
            sizes: vec![4, 144, 5184, 5184, 5184, 5184, 144],
            grid: 36,
        },
        Variant {
            label: "dump pairs contraction, all bindings 8 KiB".into(),
            wgsl: pairs.into(),
            sizes: vec![8192; 7],
            grid: 36,
        },
        Variant {
            label: "dump pairs contraction, grid 1".into(),
            wgsl: pairs.into(),
            sizes: vec![4, 144, 5184, 5184, 5184, 5184, 144],
            grid: 1,
        },
    ];
    for srvs in 1..=6 {
        v.push(synthetic(srvs, true, true));
    }
    v.push(synthetic(5, false, true));
    v.push(synthetic(5, true, false));
    v
}

/// Run one variant on a fresh device: compile, bind, dispatch, wait, and
/// ask both wgpu's lost callback and the driver whether the device survived.
fn probe(v: &Variant) -> std::result::Result<(), String> {
    let gpu = pollster::block_on(GpuDevice::request(None)).map_err(|e| format!("request: {e}"))?;
    let device = gpu.device();
    let queue = gpu.queue();
    let module = naga::front::wgsl::parse_str(&v.wgsl).map_err(|e| format!("parse: {e}"))?;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&v.label),
        source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(format!("shader: {e}"));
    }
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // Explicit layout, as production binds: an auto layout would drop the
    // uniforms slot a kernel never reads and shift every register.
    let n = v.sizes.len();
    let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = (0..n)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: i + 1 != n,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect();
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &layout_entries,
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&v.label),
        layout: Some(&layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffers: Vec<wgpu::Buffer> = v
        .sizes
        .iter()
        .map(|&size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();
    let entries: Vec<wgpu::BindGroupEntry> = buffers
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: b.as_entire_binding(),
        })
        .collect();
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &entries,
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
        return Err(format!("setup: {e:?}"));
    }
    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(v.grid, 1, 1);
    }
    queue.submit([enc.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("poll: {e}"))?;
    if let Some(reason) = removed_reason(device) {
        return Err(format!("REMOVED: {reason}"));
    }
    if let Some(reason) = gpu.lost().reason() {
        return Err(format!("LOST: {reason}"));
    }
    // The real failure surfaced on the readback allocation; do that too.
    let out = buffers.last().expect("every variant has an output");
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: out.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(out, 0, &staging, 0, out.size());
    queue.submit([enc.finish()]);
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| format!("poll after copy: {e}"))?;
    if let Some(reason) = removed_reason(device) {
        return Err(format!("REMOVED after copy: {reason}"));
    }
    Ok(())
}

#[test]
fn warp_device_survives_every_variant() {
    let mut report = Vec::new();
    let mut failed = false;
    for v in variants() {
        let outcome = match probe(&v) {
            Ok(()) => "ok".to_string(),
            Err(e) => {
                failed = true;
                e
            }
        };
        let line = format!("{:<50} {outcome}", v.label);
        eprintln!("[warp-probe] {line}");
        report.push(line);
    }
    assert!(
        !failed,
        "a variant removed the device:\n{}",
        report.join("\n")
    );
}
