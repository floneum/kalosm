//! Handwritten-MSL GEMM lab: isolates whether the ~3 TF/s ceiling of the
//! IR-generated FMA kernels is codegen or hardware. Raw wgpu passthrough.

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

        let (m, k, n) = (16384u32, 384u32, 1536u32);
        let (bm, bn, bk, tm, tn): (u32, u32, u32, u32, u32) = {
            let g: Vec<u32> = std::env::var("MSL_GEOM")
                .unwrap_or_else(|_| "64,64,8,4,4".into())
                .split(',')
                .filter_map(|v| v.parse().ok())
                .collect();
            (g[0], g[1], g[2], g[3], g[4])
        };
        let block = (bm / tm) * (bn / tn);
        assert!(tm == 4 || tm == 8);
        assert!(tn == 4 || tn == 8);

        let acc_rows = tm;
        let col_vecs = tn / 4;
        // Generate the accumulator update block.
        let mut fma_block = String::new();
        for i in 0..acc_rows {
            let comp = ["x", "y", "z", "w"][(i % 4) as usize];
            let a_vec = if tm == 4 { "a0".to_string() } else { format!("a{}", i / 4) };
            for v in 0..col_vecs {
                fma_block.push_str(&format!(
                    "      acc[{idx}] = metal::fma(float4({a_vec}.{comp}), b{v}, acc[{idx}]);\n",
                    idx = i * col_vecs + v,
                ));
            }
        }
        let mut a_loads = String::from(
            "      float4 a0 = *(threadgroup const float4*)&As[kk][ty*TM];\n",
        );
        if tm == 8 {
            a_loads.push_str(
                "      float4 a1 = *(threadgroup const float4*)&As[kk][ty*TM+4];\n",
            );
        }
        let mut b_loads = String::from(
            "      float4 b0 = *(threadgroup const float4*)&Bs[kk][tx*TN];\n",
        );
        if tn == 8 {
            b_loads.push_str(
                "      float4 b1 = *(threadgroup const float4*)&Bs[kk][tx*TN+4];\n",
            );
        }
        let mut stores = String::new();
        for i in 0..acc_rows {
            for v in 0..col_vecs {
                stores.push_str(&format!(
                    "  *(device float4*)&Y[(row0 + ty*TM + {i}) * N + col0 + tx*TN + {off}] = acc[{idx}];\n",
                    off = v * 4,
                    idx = i * col_vecs + v,
                ));
            }
        }

        let source = format!(
            r#"
#include <metal_stdlib>
using namespace metal;
constant uint K = {k};
constant uint N = {n};
constant uint BM = {bm};
constant uint BN = {bn};
constant uint BK = {bk};
constant uint TM = {tm};
constant uint TN = {tn};
constant uint BLOCK = {block};
kernel void gemm(device const float* A [[buffer(0)]],
                 device const float* B [[buffer(1)]],
                 device float* Y [[buffer(2)]],
                 uint2 wg [[threadgroup_position_in_grid]],
                 uint lid [[thread_index_in_threadgroup]]) {{
  threadgroup float As[BK][BM];
  threadgroup float Bs[BK][BN];
  const uint tx = lid % (BN/TN);
  const uint ty = lid / (BN/TN);
  const uint row0 = wg.y * BM;
  const uint col0 = wg.x * BN;
  float4 acc[{accs}] = {{}};
  for (uint kb = 0; kb < K/BK; kb++) {{
    for (uint i = lid; i < BM*BK; i += BLOCK) {{
      uint r = i % BM;
      uint kk = i / BM;
      As[kk][r] = A[(row0 + r) * K + kb*BK + kk];
    }}
    for (uint i = lid; i < BK*BN; i += BLOCK) {{
      uint kk = i / BN;
      uint c = i % BN;
      Bs[kk][c] = B[(kb*BK + kk) * N + col0 + c];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (uint kk = 0; kk < BK; kk++) {{
{a_loads}{b_loads}{fma_block}    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
{stores}}}
"#,
            accs = acc_rows * col_vecs,
        );

        let source = if std::env::var_os("MSL_PROBE").is_some() {
            format!(
                r#"
#include <metal_stdlib>
using namespace metal;
kernel void gemm(device const float* A [[buffer(0)]],
                 device const float* B [[buffer(1)]],
                 device float* Y [[buffer(2)]],
                 uint2 wg [[threadgroup_position_in_grid]],
                 uint lid [[thread_index_in_threadgroup]]) {{
  if (wg.x == 0 && wg.y == 0) {{
    Y[lid] = A[lid] + B[lid] * 1000.0;
  }}
}}
"#
            )
        } else {
            source
        };
        let module = unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("gemm"),
                num_workgroups: (block, 1, 1),
                msl: Some(std::borrow::Cow::Owned(source)),
                ..Default::default()
            })
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[0, 1, 2].map(|i| wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: i != 2 },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemm"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("gemm"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });

        use wgpu::util::DeviceExt;
        let a_host: Vec<f32> = (0..m * k).map(|i| ((i % 37) as f32) * 0.01 - 0.18).collect();
        let b_host: Vec<f32> = (0..k * n).map(|i| ((i % 29) as f32) * 0.01 - 0.14).collect();
        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
        let a_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&a_host),
            usage,
        });
        let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&b_host),
            usage,
        });
        let y_bufs: Vec<_> = (0..8)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (m as u64) * (n as u64) * 4,
                    usage,
                    mapped_at_creation: false,
                })
            })
            .collect();
        let binds: Vec<_> = y_bufs
            .iter()
            .map(|y| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: a_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: b_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                    ],
                })
            })
            .collect();

        let dispatch = |iters: u32| {
            let mut encoder = device.create_command_encoder(&Default::default());
            for i in 0..iters {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &binds[(i as usize) % binds.len()], &[]);
                pass.dispatch_workgroups(n / bn, m / bm, 1);
            }
            queue.submit([encoder.finish()]);
        };

        dispatch(3);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        // verify
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: y_bufs[0].size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&y_bufs[0], 0, &staging, 0, staging.size());
        queue.submit([encoder.finish()]);
        let slice = staging.slice(..);
        let (tx_ch, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx_ch.send(r).unwrap());
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().unwrap().unwrap();
        {
            let view = slice.get_mapped_range();
            let y_host: &[f32] = bytemuck::cast_slice(&view);
            for (mi, ni) in [(0u32, 0u32), (7, 13), (511, 1023), (16383, 1535), (9000, 777)] {
                let mut acc = 0f64;
                for ki in 0..k {
                    acc += a_host[(mi * k + ki) as usize] as f64
                        * b_host[(ki * n + ni) as usize] as f64;
                }
                let got = y_host[(mi * n + ni) as usize];
                assert!(
                    (got as f64 - acc).abs() < 1e-2 + acc.abs() * 1e-4,
                    "mismatch at [{mi},{ni}]: got {got}, want {acc}"
                );
            }
        }
        staging.unmap();

        let iters = 40;
        let start = std::time::Instant::now();
        dispatch(iters);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let per = start.elapsed().as_secs_f64() / iters as f64;
        let tf = 2.0 * m as f64 * k as f64 * n as f64 / per / 1e12;
        println!(
            "msl_lab ({bm},{bn},{bk},{tm},{tn}): {:.3} ms, {tf:.2} TF/s (verified)",
            per * 1e3
        );
    });
}
