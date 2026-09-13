//! Thin GPU runner and existing-Fusor comparison. No compiler decisions here.
use crate::graph::{Graph, Op, Point, Reduce, View};
use crate::plan::Plan;
use fusor_gpu::GpuDevice;
use std::time::Instant;
use wgpu::util::DeviceExt;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn validate_shader(source: &str) -> Result<()> {
    let module = naga::front::wgsl::parse_str(source).map_err(|e| e.emit_to_string(source))?;
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        if source.contains("// native subgroup collectives") {
            naga::valid::Capabilities::SUBGROUP
        } else {
            naga::valid::Capabilities::empty()
        },
    )
    .validate(&module)?;
    Ok(())
}
pub fn compare(g: &Graph, seed: usize, got: &[Vec<f32>]) -> f32 {
    let expected = g.reference(seed);
    assert_eq!(expected.len(), got.len());
    let mut worst = 0.0f32;
    for (output, (a, b)) in expected.iter().zip(got).enumerate() {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let error = (x - y).abs();
            worst = worst.max(error);
            assert!(
                y.is_finite() && error <= 2e-4 + 2e-4 * x.abs(),
                "{} seed {seed} output {output}[{i}]: expected {x}, got {y}",
                g.name
            );
        }
    }
    worst
}
pub struct Executable {
    pipelines: Vec<wgpu::ComputePipeline>,
    groups: Vec<u32>,
    bind: wgpu::BindGroup,
    input: wgpu::Buffer,
    arena: wgpu::Buffer,
    readback: wgpu::Buffer,
    bytes: u64,
    pub build_ms: f64,
}
impl Executable {
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>) {
        for (pipeline, groups) in self.pipelines.iter().zip(&self.groups) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.bind, &[]);
            pass.dispatch_workgroups(*groups, 1, 1);
        }
    }
    pub fn build(gpu: &GpuDevice, g: &Graph, p: &Plan, sources: &[String]) -> Result<Self> {
        let start = Instant::now();
        let device = gpu.device();
        if let Some(collective) = p.collective {
            assert_eq!(
                gpu.caps()
                    .subgroups
                    .filter(|s| s.is_fixed())
                    .map(|s| s.assumed()),
                Some(collective.width()),
                "compiled subgroup geometry must match the device"
            );
        }
        // Counted u32 loops increment by at most BLOCK. This bound ensures
        // their final increment cannot wrap; the tree loop divides by two.
        assert!(
            g.values
                .iter()
                .all(|v| v.len() <= u32::MAX as usize - crate::plan::BLOCK)
        );
        assert!(p.regions.iter().all(
            |r| r.shared.len * 4 <= device.limits().max_compute_workgroup_storage_size as usize
        ));
        let input_data: Vec<f32> = g.input_data(0).into_iter().flat_map(|(_, x)| x).collect();
        let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("prototype inputs"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let bytes = (p.global.len.max(1) * 4) as u64;
        let arena = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prototype colored arena"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prototype readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let entries = (0..2)
            .map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: i == 0 },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>();
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: arena.as_entire_binding(),
                },
            ],
        });
        let mut pipelines = vec![];
        for source in sources {
            validate_shader(source)?;
            let descriptor = wgpu::ShaderModuleDescriptor {
                label: Some("prototype region"),
                source: wgpu::ShaderSource::Wgsl(source.clone().into()),
            };
            let module = if p.legacy {
                device.create_shader_module(descriptor)
            } else {
                // SAFETY: only emit::shader's finite counted loops reach this
                // private runner, with counter overflow excluded above.
                // Bounds checks remain enabled. The existing compiler also
                // omits loop instrumentation after proving termination.
                unsafe {
                    device.create_shader_module_trusted(
                        descriptor,
                        wgpu::ShaderRuntimeChecks {
                            force_loop_bounding: false,
                            ..wgpu::ShaderRuntimeChecks::checked()
                        },
                    )
                }
            };
            pipelines.push(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("prototype region"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            );
        }
        Ok(Self {
            pipelines,
            groups: p.regions.iter().map(|r| r.groups as u32).collect(),
            bind,
            input,
            arena,
            readback,
            bytes,
            build_ms: start.elapsed().as_secs_f64() * 1000.0,
        })
    }
    fn dispatch(&self, gpu: &GpuDevice, iterations: usize, readback: bool) {
        let mut encoder = gpu.device().create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("prototype batch"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &self.bind, &[]);
            for _ in 0..iterations {
                for (pipeline, groups) in self.pipelines.iter().zip(&self.groups) {
                    pass.set_pipeline(pipeline);
                    pass.dispatch_workgroups(*groups, 1, 1);
                }
            }
        }
        if readback {
            encoder.copy_buffer_to_buffer(&self.arena, 0, &self.readback, 0, self.bytes);
        }
        gpu.queue().submit([encoder.finish()]);
    }
    pub fn check(&self, gpu: &GpuDevice, g: &Graph, p: &Plan, seed: usize) -> Result<f32> {
        let data: Vec<f32> = g
            .input_data(seed)
            .into_iter()
            .flat_map(|(_, x)| x)
            .collect();
        gpu.queue()
            .write_buffer(&self.input, 0, bytemuck::cast_slice(&data));
        // Poison reused allocations to expose reads before writes.
        let poison = vec![f32::NAN; p.global.len.max(1)];
        gpu.queue()
            .write_buffer(&self.arena, 0, bytemuck::cast_slice(&poison));
        self.dispatch(gpu, 1, true);
        let (tx, rx) = std::sync::mpsc::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        gpu.device().poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;
        let raw = self.readback.slice(..).get_mapped_range();
        let flat: &[f32] = bytemuck::cast_slice(&raw);
        let got: Vec<Vec<f32>> = g
            .roots
            .iter()
            .map(|root| {
                (0..g.values[*root].len())
                    .map(|i| {
                        let (base, j) = g.resolve_index(*root, i);
                        flat[p.global.offset(base).unwrap() + j]
                    })
                    .collect()
            })
            .collect();
        let worst = compare(g, seed, &got);
        drop(raw);
        self.readback.unmap();
        Ok(worst)
    }
    pub fn time(&self, gpu: &GpuDevice, iterations: usize) -> Result<f64> {
        self.dispatch(gpu, 3, false);
        gpu.device().poll(wgpu::PollType::wait_indefinitely())?;
        let mut rounds = vec![];
        for _ in 0..5 {
            let start = Instant::now();
            self.dispatch(gpu, iterations, false);
            gpu.device().poll(wgpu::PollType::wait_indefinitely())?;
            rounds.push(start.elapsed().as_secs_f64() * 1e6 / iterations as f64);
        }
        rounds.sort_by(f64::total_cmp);
        Ok(rounds[2])
    }
}

pub struct Baseline {
    pub launches: u64,
    pub first_ms: f64,
    pub error: f32,
}
/// Calls the existing compiler on exactly the same operations and shapes.
/// Only first resolve time and dispatch count are compared: its host/runtime
/// path differs from the prototype's prebuilt command batching.
pub fn baseline(g: &Graph) -> Result<Baseline> {
    let device = fusor::Device::gpu_blocking()?;
    let roots = baseline_graph(g, &device)?;
    let count = device.session().launch_count();
    let start = Instant::now();
    device.session().resolve(&roots)?;
    device.session().wait()?;
    let first_ms = start.elapsed().as_secs_f64() * 1000.0;
    let launches = device.session().launch_count() - count;
    let got = roots
        .iter()
        .map(|t| t.to_vec_f32())
        .collect::<fusor::Result<Vec<_>>>()?;
    Ok(Baseline {
        launches,
        first_ms,
        error: compare(g, 0, &got),
    })
}
pub fn baseline_graph(g: &Graph, device: &fusor::Device) -> Result<Vec<fusor::tensor::Dyn>> {
    use fusor::{Dim, tensor::Dyn};
    let mut values: Vec<Dyn> = vec![];
    let inputs = g.input_data(0);
    for (id, v) in g.values.iter().enumerate() {
        let dims: Vec<Dim> = v.shape.iter().map(|d| Dim::Const(*d as u64)).collect();
        let value = match &v.op {
            Op::Input => Dyn::from_elements(
                device.graph().handle(),
                &dims,
                &inputs.iter().find(|(x, _)| *x == id).unwrap().1,
            )?,
            Op::View(view, x) => match view {
                View::Reshape => values[*x].reshape_dims(&dims)?,
                View::Permute(axes) => values[*x].permute(axes)?,
                View::Broadcast => values[*x].broadcast_as(&dims)?,
            },
            Op::Point(op, xs) => match op {
                Point::Square => values[xs[0]].sqr()?,
                Point::Neg => values[xs[0]].neg()?,
                Point::Sqrt => values[xs[0]].sqrt()?,
                Point::Exp => values[xs[0]].exp()?,
                Point::Add => values[xs[0]].add(&values[xs[1]])?,
                Point::Sub => values[xs[0]].sub(&values[xs[1]])?,
                Point::Div => values[xs[0]].div(&values[xs[1]])?,
            },
            Op::Reduce(kind, x, axis) => match kind {
                Reduce::Sum => values[*x].sum(*axis)?,
                Reduce::Max => values[*x].max(*axis)?,
            },
        };
        values.push(value);
    }
    Ok(g.roots.iter().map(|r| values[*r].clone()).collect())
}
