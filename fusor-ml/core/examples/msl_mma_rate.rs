//! Pure simdgroup-matrix MMA rate: float8x8 vs half8x8 vs mixed operands,
//! register-resident with no memory traffic in the loop. Settles whether
//! f16 raises the matrix-FLOP ceiling on this GPU or only saves bandwidth.

fn main() {
    pollster::block_on(async {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: wgpu::Features::PASSTHROUGH_SHADERS,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();

        const BLOCK: u32 = 128;
        const SGS: u32 = 4;
        const ACCS: u32 = 8;
        const ITERS: u32 = 4096;
        const WGS: u32 = 1024;

        let kernel = |name: &str, ty: &str, ab_ty: &str| {
            format!(
                r#"
#include <metal_stdlib>
using namespace metal;
kernel void {name}(device float* Y [[buffer(0)]],
                 uint wg [[threadgroup_position_in_grid]],
                 uint lid [[thread_index_in_threadgroup]],
                 uint sgid [[simdgroup_index_in_threadgroup]]) {{
  // Seed from memory so the operands are runtime values, and feed an
  // accumulator back into `a` each outer iteration so `a*b` is never
  // loop-invariant — otherwise the compiler folds the MMA chain.
  {ab_ty} seed = {ab_ty}(Y[wg]);
  simdgroup_{ab_ty}8x8 b = make_filled_simdgroup_matrix<{ab_ty}, 8, 8>(seed * {ab_ty}(0.999));
  // Distinct, mutating multiplicands per chain: no product is shared
  // between chains or repeated across iterations, so the compiler can
  // neither hoist a*b nor strength-reduce the chains to matrix adds.
  simdgroup_{ab_ty}8x8 a[{ACCS}];
  simdgroup_{ty}8x8 acc[{ACCS}];
  #pragma clang loop unroll(full)
  for (uint j = 0; j < {ACCS}; j++) {{
    a[j] = make_filled_simdgroup_matrix<{ab_ty}, 8, 8>(seed + {ab_ty}(j) * {ab_ty}(0.001));
    acc[j] = make_filled_simdgroup_matrix<{ty}, 8, 8>({ty}(0.0));
  }}
  for (uint i = 0; i < {ITERS}; i++) {{
    #pragma clang loop unroll(full)
    for (uint j = 0; j < {ACCS}; j++) {{
      simdgroup_multiply_accumulate(acc[j], a[j], b, acc[j]);
    }}
    #pragma clang loop unroll(full)
    for (uint j = 0; j < {ACCS}; j++) {{
      simdgroup_multiply_accumulate(a[j], a[j], b, a[j]);
    }}
  }}
  // Observe every chain so none is dead-code-eliminated.
  threadgroup {ty} scratch[{ACCS} * 64];
  #pragma clang loop unroll(full)
  for (uint j = 0; j < {ACCS}; j++) {{
    simdgroup_store(acc[j], &scratch[j * 64], 8);
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid == 0) {{
    float total = 0.0;
    for (uint j = 0; j < {ACCS}; j++) {{
      total += float(scratch[j * 64]);
    }}
    Y[wg] = total;
  }}
}}
"#
            )
        };

        use wgpu::util::DeviceExt;
        let seed = vec![1.00048828125f32; WGS as usize];
        let out = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&seed),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: out.as_entire_binding(),
            }],
        });

        for (name, ty, ab_ty) in [
            ("mma_f32", "float", "float"),
            ("mma_f16", "half", "half"),
            ("mma_mixed", "float", "half"),
        ] {
            let source = kernel(name, ty, ab_ty);
            let module = unsafe {
                device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some(name),
                    num_workgroups: (BLOCK, 1, 1),
                    msl: Some(std::borrow::Cow::Owned(source)),
                    ..Default::default()
                })
            };
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            });

            let run = |reps: u32| {
                let mut encoder = device.create_command_encoder(&Default::default());
                for _ in 0..reps {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(WGS, 1, 1);
                }
                queue.submit([encoder.finish()]);
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
            };
            run(2);
            let reps = 8;
            let start = std::time::Instant::now();
            run(reps);
            let per = start.elapsed().as_secs_f64() / reps as f64;
            let flops = (WGS as f64) * (SGS as f64) * (ITERS as f64) * (ACCS as f64 * 2.0) * 1024.0;
            println!("{name}: {:.3} ms, {:.2} TF/s", per * 1e3, flops / per / 1e12);
        }
    });
}
