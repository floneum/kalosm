#![cfg(feature = "runtime")]

//! Integration tests for the tensor IR.

use crate::DispatchProgram;
use crate::analysis::TensorAnalysis;
use crate::builders::IrBuilder;
use crate::extractor::{BeamConfig, beam_extract, beam_extract_candidates};
use crate::language::*;
use crate::naga_codegen;
use crate::rules::{self, Phase, RunnerConfig};
use crate::skeleton::{beam_extract_valid_candidates, build_dispatch_program_from_extracted};
use crate::types::*;
use std::mem::size_of;
use std::panic::{self, AssertUnwindSafe};

pub(super) fn request_test_device(adapter: &wgpu::Adapter) -> (wgpu::Device, wgpu::Queue) {
    let supported = adapter.features();
    let mut required = wgpu::Features::empty();
    if supported.contains(wgpu::Features::SUBGROUP) {
        required |= wgpu::Features::SUBGROUP;
    }
    if supported.contains(wgpu::Features::SUBGROUP_BARRIER) {
        required |= wgpu::Features::SUBGROUP_BARRIER;
    }
    if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required |= wgpu::Features::TIMESTAMP_QUERY;
    }

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("tensor_ir test device"),
        required_features: required,
        ..Default::default()
    }))
    .expect("failed to create device")
}

fn max_abs_err(lhs: &[f32], rhs: &[f32]) -> f32 {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max)
}

fn softmax_weighted_reduce_reference(
    scores: &[f32],
    values: &[f32],
    rows: u32,
    weights: u32,
    outputs: u32,
) -> Vec<f32> {
    let mut expected = vec![0.0f32; (rows * outputs) as usize];
    for row in 0..rows {
        let row_start = (row * weights) as usize;
        let row_scores = &scores[row_start..row_start + weights as usize];
        let row_max = row_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = row_scores
            .iter()
            .map(|score| (*score - row_max).exp())
            .sum::<f32>();
        for out in 0..outputs {
            let mut acc = 0.0f32;
            for weight in 0..weights {
                let prob = (row_scores[weight as usize] - row_max).exp() / denom;
                acc += prob * values[(weight * outputs + out) as usize];
            }
            expected[(row * outputs + out) as usize] = acc;
        }
    }
    expected
}

fn attention_reference(q: &[f32], k: &[f32], v: &[f32], seq: u32, d: u32) -> Vec<f32> {
    let mut expected = vec![0.0f32; (seq * d) as usize];
    for row in 0..seq {
        let mut scores = vec![0.0f32; seq as usize];
        for key in 0..seq {
            let mut score = 0.0f32;
            for feature in 0..d {
                score += q[(row * d + feature) as usize] * k[(key * d + feature) as usize];
            }
            scores[key as usize] = score;
        }
        let row_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let denom = scores
            .iter()
            .map(|score| (*score - row_max).exp())
            .sum::<f32>();
        for out in 0..d {
            let mut acc = 0.0f32;
            for key in 0..seq {
                let prob = (scores[key as usize] - row_max).exp() / denom;
                acc += prob * v[(key * d + out) as usize];
            }
            expected[(row * d + out) as usize] = acc;
        }
    }
    expected
}

#[test]
#[cfg(feature = "runtime")]
fn test_gpu_matmul() {
    use wgpu::util::DeviceExt;

    const M: u32 = 64;
    const N: u32 = 64;
    const K: u32 = 64;
    const SIMD_WIDTH: u32 = 32;

    // ── Build IR and optimize ──
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(M), Dim::Lit(K)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Lit(K), Dim::Lit(N)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, M, N, K);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    let beam_cfg = BeamConfig {
        beam_width: 8,
        ..Default::default()
    };

    let a_data: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..K * N).map(|i| (i % 5) as f32 * 0.1).collect();

    let mut cpu_result = vec![0.0f32; (M * N) as usize];
    for m in 0..M {
        for n in 0..N {
            for k in 0..K {
                cpu_result[(m * N + n) as usize] +=
                    a_data[(m * K + k) as usize] * b_data[(k * N + n) as usize];
            }
        }
    }

    let ctx = crate::runtime::GpuContext::new();
    let candidates = beam_extract_candidates(&egraph, root, &beam_cfg, 24);
    let program = candidates
        .into_iter()
        .filter_map(|(_cost, expr)| {
            let program = build_dispatch_program_from_extracted(
                &expr,
                egraph.clone(),
                &DeviceProfile::default(),
                &LoweringOptions::default(),
            );
            let dispatch = program.dispatches.first()?;
            if program.dispatches.len() != 1
                || dispatch.outputs.len() != 1
                || dispatch.tg_buffers.len() != 2
            {
                return None;
            }
            let gpu_result = match panic::catch_unwind(AssertUnwindSafe(|| {
                ctx.execute(&program, &[&a_data, &b_data])
            })) {
                Ok(result) => result,
                Err(_) => return None,
            };
            let max_err = gpu_result[..cpu_result.len()]
                .iter()
                .zip(&cpu_result)
                .map(|(gpu, cpu)| (gpu - cpu).abs())
                .fold(0.0f32, f32::max);
            (max_err < 1e-3).then_some(program)
        })
        .next()
        .expect("expected a runnable matmul candidate");

    println!("=== Matmul dispatch program ===");
    println!("{program}");
    match crate::lower_to_wgsl(&program) {
        Ok(wgsl) => println!("=== WGSL ===\n{wgsl}"),
        Err(e) => println!("WGSL error: {e}"),
    }

    let dispatch = &program.dispatches[0];

    // ── Lower to naga module ──
    let module =
        naga_codegen::lower_dispatch_program(naga_codegen::verify(&program).expect("verify"));

    // ── wgpu setup ──
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no GPU adapter found");
    let (device, queue) = request_test_device(&adapter);

    // ── Shader module (naga IR directly, no WGSL round-trip) ──
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("matmul_test"),
        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
    });

    // ── Input data ──
    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_0"),
        contents: bytemuck::cast_slice(&a_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_1"),
        contents: bytemuck::cast_slice(&b_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // ── Output + staging buffers ──
    let num_inputs = dispatch.inputs.len();
    let output_elems = (dispatch.workgroups * SIMD_WIDTH) as usize * dispatch.outputs.len();
    let output_bytes = (output_elems * size_of::<f32>()) as u64;

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // ── Pipeline + bind group ──
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("matmul_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("dispatch_0"),
        compilation_options: Default::default(),
        cache: None,
    });

    let input_bufs = [&buf_a, &buf_b];
    let mut entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::new();
    for (i, input_buf) in input_bufs.iter().enumerate().take(num_inputs) {
        entries.push(wgpu::BindGroupEntry {
            binding: i as u32,
            resource: input_buf.as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: num_inputs as u32,
        resource: buf_out.as_entire_binding(),
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul_bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    // ── Dispatch ──
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("matmul_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let physical_wgs = dispatch.workgroups / dispatch.simdgroups.max(1);
        pass.dispatch_workgroups(physical_wgs, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_staging, 0, output_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    // ── Readback ──
    let slice = buf_staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("channel closed").expect("map failed");

    let gpu_result: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

    let max_err = gpu_result
        .iter()
        .zip(&cpu_result)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "GPU matmul max error {max_err} exceeds tolerance 1e-3"
    );
}

#[test]
#[cfg(feature = "runtime")]
fn test_runtime_beam_candidate_smoke() {
    const M: u32 = 64;
    const N: u32 = 64;
    const K: u32 = 64;

    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(M), Dim::Lit(K)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Lit(K), Dim::Lit(N)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, M, N, K);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    let beam_cfg = BeamConfig {
        beam_width: 8,
        ..Default::default()
    };
    let candidates = beam_extract_candidates(&egraph, root, &beam_cfg, 32);
    assert!(!candidates.is_empty(), "expected runtime beam candidates");

    let a_data: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
    let b_data: Vec<f32> = (0..K * N).map(|i| (i % 5) as f32 * 0.1).collect();
    let mut cpu_result = vec![0.0f32; (M * N) as usize];
    for m in 0..M {
        for n in 0..N {
            for k in 0..K {
                cpu_result[(m * N + n) as usize] +=
                    a_data[(m * K + k) as usize] * b_data[(k * N + n) as usize];
            }
        }
    }

    let ctx = crate::runtime::GpuContext::new();
    let mut found_valid = false;
    for (_, expr) in candidates {
        let program = build_dispatch_program_from_extracted(
            &expr,
            egraph.clone(),
            &DeviceProfile::default(),
            &LoweringOptions::default(),
        );
        if program.dispatches.len() != 1 {
            continue;
        }

        let wgsl =
            crate::lower_to_wgsl(&program).expect("single-dispatch candidate should lower to WGSL");
        assert!(
            wgsl.contains("@compute"),
            "WGSL should contain compute entry point"
        );

        let gpu_result = ctx.execute(&program, &[&a_data, &b_data]);
        let max_err = gpu_result[..cpu_result.len()]
            .iter()
            .zip(&cpu_result)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f32, f32::max);
        if max_err < 1e-3 {
            found_valid = true;
            break;
        }
    }

    assert!(
        found_valid,
        "expected at least one runtime beam candidate to match CPU reference"
    );
}

#[test]
#[cfg(feature = "runtime")]
fn test_runtime_selected_softmax_weighted_reduce_is_correct() {
    const ROWS: u32 = 32;
    const WEIGHTS: u32 = 32;
    const OUTPUTS: u32 = 32;

    let mut b = IrBuilder::new();
    let scores = b.input(
        0,
        Shape(vec![Dim::Lit(ROWS), Dim::Lit(WEIGHTS)]),
        DType::F32,
    );
    let values = b.input(
        1,
        Shape(vec![Dim::Lit(WEIGHTS), Dim::Lit(OUTPUTS)]),
        DType::F32,
    );
    let _weighted =
        super::build_softmax_weighted_reduce_ir(&mut b, scores, values, ROWS, OUTPUTS, WEIGHTS);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 100_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);
    let beam_cfg = BeamConfig {
        beam_width: 32,
        ..Default::default()
    };
    let (_cost, expr) = beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable weighted-reduce candidate");
    let program = build_dispatch_program_from_extracted(
        &expr,
        egraph.clone(),
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );
    assert!(
        !program.dispatches.is_empty(),
        "selected weighted-reduce program should have runnable dispatches"
    );

    let scores_data: Vec<f32> = (0..ROWS * WEIGHTS)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.125)
        .collect();
    let values_data: Vec<f32> = (0..WEIGHTS * OUTPUTS)
        .map(|i| ((i % 13) as f32 - 6.0) * 0.2)
        .collect();
    let expected =
        softmax_weighted_reduce_reference(&scores_data, &values_data, ROWS, WEIGHTS, OUTPUTS);

    let ctx = crate::runtime::GpuContext::new();
    let gpu_result = ctx.execute(&program, &[&scores_data, &values_data]);
    let max_err = max_abs_err(&gpu_result[..expected.len()], &expected);
    assert!(
        max_err < 1e-4,
        "selected weighted-reduce kernel should match reference, got max_err={max_err}"
    );
}

#[test]
#[cfg(feature = "runtime")]
fn test_runtime_selected_attention_is_correct() {
    const SEQ: u32 = 32;
    const D: u32 = 32;

    let mut b = IrBuilder::new();
    let q = b.input(0, Shape(vec![Dim::Lit(SEQ), Dim::Lit(D)]), DType::F32);
    let k = b.input(1, Shape(vec![Dim::Lit(SEQ), Dim::Lit(D)]), DType::F32);
    let v = b.input(2, Shape(vec![Dim::Lit(SEQ), Dim::Lit(D)]), DType::F32);
    let _attention = super::build_attention_ir(&mut b, q, k, v, SEQ, D);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 100_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);
    let beam_cfg = BeamConfig {
        beam_width: 32,
        ..Default::default()
    };
    let (_cost, expr) = beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        1,
    )
    .into_iter()
    .next()
    .expect("expected a lowerable attention candidate");
    let program = build_dispatch_program_from_extracted(
        &expr,
        egraph.clone(),
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );
    assert!(
        !program.dispatches.is_empty(),
        "selected attention program should have runnable dispatches"
    );

    let q_data: Vec<f32> = (0..SEQ * D)
        .map(|i| ((i % 19) as f32 - 9.0) * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..SEQ * D)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.08)
        .collect();
    let v_data: Vec<f32> = (0..SEQ * D)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.07)
        .collect();
    let expected = attention_reference(&q_data, &k_data, &v_data, SEQ, D);

    let ctx = crate::runtime::GpuContext::new();
    let gpu_result = ctx.execute(&program, &[&q_data, &k_data, &v_data]);
    let max_err = max_abs_err(&gpu_result[..expected.len()], &expected);
    assert!(
        max_err < 1e-4,
        "selected attention kernel should match reference, got max_err={max_err}"
    );
}

#[test]
#[cfg(feature = "runtime")]
fn test_gpu_elementwise_add() {
    use wgpu::util::DeviceExt;

    const ROWS: u32 = 64;
    const COLS: u32 = 64;
    const N: u32 = ROWS * COLS;
    const SIMD_WIDTH: u32 = 32;

    // ── Build IR: C = A + B ──
    let mut b = IrBuilder::new();
    let shape = Shape(vec![Dim::Lit(ROWS), Dim::Lit(COLS)]);
    let a = b.input(0, shape.clone(), DType::F32);
    let b_in = b.input(1, shape.clone(), DType::F32);
    let arg0 = b.scalar_arg(0);
    let arg1 = b.scalar_arg(1);
    let body = b.bin_op(BinaryOp::Add, arg0, arg1);
    let _ewise = b.elementwise(shape, &[a, b_in], body);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    let beam_cfg = BeamConfig {
        beam_width: 8,
        ..Default::default()
    };
    let (_cost, extracted) = beam_extract(&egraph, root, &beam_cfg);
    let program = build_dispatch_program_from_extracted(
        &extracted,
        egraph,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );
    let dispatch = &program.dispatches[0];

    // ── Lower to naga module ──
    let module =
        naga_codegen::lower_dispatch_program(naga_codegen::verify(&program).expect("verify"));

    // ── wgpu setup ──
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no GPU adapter found");
    let (device, queue) = request_test_device(&adapter);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("add_test"),
        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
    });

    // ── Input data ──
    let a_data: Vec<f32> = (0..N).map(|i| i as f32 * 0.01).collect();
    let b_data: Vec<f32> = (0..N).map(|i| (N - i) as f32 * 0.01).collect();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_0"),
        contents: bytemuck::cast_slice(&a_data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buf_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_1"),
        contents: bytemuck::cast_slice(&b_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_elems = (dispatch.workgroups * SIMD_WIDTH) as usize * dispatch.outputs.len();
    let output_bytes = (output_elems * size_of::<f32>()) as u64;

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("add_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("dispatch_0"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("add_bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_b.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buf_out.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("add_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("add_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let physical_wgs = dispatch.workgroups / dispatch.simdgroups.max(1);
        pass.dispatch_workgroups(physical_wgs, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_staging, 0, output_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = buf_staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("channel closed").expect("map failed");

    let gpu_result: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

    // ── CPU reference: a + b ──
    let cpu_result: Vec<f32> = a_data.iter().zip(&b_data).map(|(a, b)| a + b).collect();

    let max_err = gpu_result[..N as usize]
        .iter()
        .zip(&cpu_result)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-5,
        "GPU elementwise add max error {max_err} exceeds tolerance 1e-5"
    );
}

#[test]
#[cfg(feature = "runtime")]
fn test_runtime_blocked_beam_candidate_lowers() {
    const M: u32 = 64;
    const N: u32 = 64;
    const K: u32 = 64;

    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(M), Dim::Lit(K)]), DType::F32);
    let b_in = b.input(1, Shape(vec![Dim::Lit(K), Dim::Lit(N)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut b, a, b_in, M, N, K);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 12,
        node_limit: 60_000,
        time_limit_secs: 45,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering], &config);

    let beam_cfg = BeamConfig {
        beam_width: 8,
        ..Default::default()
    };
    let candidates = beam_extract_candidates(&egraph, root, &beam_cfg, 32);
    let program = candidates
        .into_iter()
        .find_map(|(_cost, expr)| {
            let program = build_dispatch_program_from_extracted(
                &expr,
                egraph.clone(),
                &DeviceProfile::default(),
                &LoweringOptions::default(),
            );
            let dispatch = program.dispatches.first()?;
            (program.dispatches.len() == 1 && dispatch.outputs.len() > 1).then_some(program)
        })
        .expect("expected a runnable blocked beam candidate");

    let wgsl =
        crate::lower_to_wgsl(&program).expect("blocked runtime candidate should lower to WGSL");
    assert!(
        wgsl.contains("@compute"),
        "WGSL should contain compute entry point"
    );
}

#[test]
#[cfg(feature = "runtime")]
fn test_runtime_benchmark_selected_matmul_is_correct() {
    const M: u32 = 256;
    const N: u32 = 256;
    const K: u32 = 256;

    let target_valid_candidates = 8usize;
    let mut legacy_builder = IrBuilder::new();
    let lhs = legacy_builder.input(0, Shape(vec![Dim::Lit(M), Dim::Lit(K)]), DType::F32);
    let rhs = legacy_builder.input(1, Shape(vec![Dim::Lit(K), Dim::Lit(N)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut legacy_builder, lhs, rhs, M, N, K);

    let mut egraph = crate::TensorEGraph::default();
    let root = egraph.add_expr(&legacy_builder.expr);
    egraph.rebuild();
    let runner = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = crate::saturate(egraph, &runner);
    let candidates = beam_extract_candidates(
        &egraph,
        root,
        &BeamConfig::default(),
        target_valid_candidates * 8,
    );

    let input_a: Vec<f32> = (0..M * K).map(|i| (i % 17) as f32 * 0.1).collect();
    let input_b: Vec<f32> = (0..K * N).map(|i| (i % 31) as f32 * 0.1).collect();
    let inputs: Vec<&[f32]> = vec![&input_a, &input_b];
    let ctx = crate::runtime::GpuContext::new();

    let tuning_config = crate::runtime::ProgramBenchmarkConfig {
        warmup_runs: 1,
        timing_runs: 2,
    };

    let mut expected = vec![0.0f32; (M * N) as usize];
    for row in 0..M {
        for col in 0..N {
            let mut acc = 0.0f32;
            for inner in 0..K {
                acc += input_a[(row * K + inner) as usize] * input_b[(inner * N + col) as usize];
            }
            expected[(row * N + col) as usize] = acc;
        }
    }

    let mut selected: Option<(f64, u32, usize, usize, DispatchProgram)> = None;
    let mut valid_candidates = 0usize;
    for (_cost, expr) in candidates {
        let program = build_dispatch_program_from_extracted(
            &expr,
            egraph.clone(),
            &DeviceProfile::default(),
            &LoweringOptions::default(),
        );
        let Some(dispatch) = program.dispatches.first() else {
            continue;
        };
        let gpu_result =
            match panic::catch_unwind(AssertUnwindSafe(|| ctx.execute(&program, &inputs))) {
                Ok(result) => result,
                Err(_) => continue,
            };
        let max_err = gpu_result[..expected.len()]
            .iter()
            .zip(&expected)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f32, f32::max);
        if max_err >= 1e-3 {
            continue;
        }
        valid_candidates += 1;

        let result = match panic::catch_unwind(AssertUnwindSafe(|| {
            ctx.benchmark(&program, &inputs, tuning_config)
        })) {
            Ok(Ok(result)) => result,
            Ok(Err(_)) | Err(_) => continue,
        };
        let physical_workgroups = dispatch.workgroups / dispatch.simdgroups.max(1);
        let tg_buffer_count = dispatch.tg_buffers.len();
        let output_count = dispatch.outputs.len();
        let score = result.median_gpu_us;

        match &selected {
            Some((best_score, best_workgroups, best_tg_buffers, best_outputs, _))
                if score > *best_score
                    || (score == *best_score
                        && (physical_workgroups > *best_workgroups
                            || (physical_workgroups == *best_workgroups
                                && tg_buffer_count < *best_tg_buffers)
                            || (physical_workgroups == *best_workgroups
                                && tg_buffer_count == *best_tg_buffers
                                && output_count <= *best_outputs))) => {}
            _ => {
                selected = Some((
                    score,
                    physical_workgroups,
                    tg_buffer_count,
                    output_count,
                    program,
                ));
            }
        }

        if valid_candidates >= target_valid_candidates {
            break;
        }
    }

    let (_, _, _, _, program) =
        selected.expect("expected runtime benchmark to select a valid matmul candidate");
    let gpu_result = ctx.execute(&program, &[&input_a, &input_b]);

    let max_err = gpu_result[..expected.len()]
        .iter()
        .zip(&expected)
        .map(|(gpu, cpu)| (gpu - cpu).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-3,
        "runtime benchmark selected matmul should match CPU reference, got max_err={max_err}"
    );
}

#[test]
#[cfg(feature = "runtime")]
fn test_runtime_cooperative_scalar_matmul_candidate_is_correct() {
    const M: u32 = 256;
    const N: u32 = 256;
    const K: u32 = 256;

    let mut legacy_builder = IrBuilder::new();
    let lhs = legacy_builder.input(0, Shape(vec![Dim::Lit(M), Dim::Lit(K)]), DType::F32);
    let rhs = legacy_builder.input(1, Shape(vec![Dim::Lit(K), Dim::Lit(N)]), DType::F32);
    let _mm = super::build_binary_mul_add_contraction_ir(&mut legacy_builder, lhs, rhs, M, N, K);

    let mut egraph = crate::TensorEGraph::default();
    let root = egraph.add_expr(&legacy_builder.expr);
    egraph.rebuild();
    let runner = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = crate::saturate(egraph, &runner);

    let candidates = beam_extract_candidates(&egraph, root, &BeamConfig::default(), 24);

    let input_a: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
    let input_b: Vec<f32> = (0..K * N).map(|i| (i % 5) as f32 * 0.1).collect();
    let expected = {
        let mut output = vec![0.0f32; (M * N) as usize];
        for row in 0..M {
            for col in 0..N {
                let mut acc = 0.0f32;
                for inner in 0..K {
                    acc +=
                        input_a[(row * K + inner) as usize] * input_b[(inner * N + col) as usize];
                }
                output[(row * N + col) as usize] = acc;
            }
        }
        output
    };

    let ctx = crate::runtime::GpuContext::new();
    let mut checked = 0usize;
    let mut best_err = f32::INFINITY;
    for (_cost, expr) in candidates {
        let program = build_dispatch_program_from_extracted(
            &expr,
            egraph.clone(),
            &DeviceProfile::default(),
            &LoweringOptions::default(),
        );
        let Some(dispatch) = program.dispatches.first() else {
            continue;
        };
        if program.dispatches.len() != 1
            || dispatch.outputs.len() != 1
            || dispatch.tg_buffers.len() != 2
        {
            continue;
        }

        let gpu_result = match panic::catch_unwind(AssertUnwindSafe(|| {
            ctx.execute(&program, &[&input_a, &input_b])
        })) {
            Ok(result) => result,
            Err(_) => continue,
        };
        let max_err = gpu_result[..expected.len()]
            .iter()
            .zip(&expected)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f32, f32::max);
        best_err = best_err.min(max_err);
        if max_err < 1e-3 {
            checked += 1;
            break;
        }
    }

    assert!(
        checked > 0,
        "expected at least one correct cooperative scalar matmul candidate, best_err={best_err}"
    );
}

#[test]
#[cfg(feature = "runtime")]
fn test_gpu_sum_reduce() {
    use wgpu::util::DeviceExt;

    const ROWS: u32 = 32;
    const COLS: u32 = 64;
    const SIMD_WIDTH: u32 = 32;

    // ── Build IR: out[row] = sum(A[row, :]) ──
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(ROWS), Dim::Lit(COLS)]), DType::F32);
    let _red = b.reduce(a, 1, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate(egraph, &config);

    let beam_cfg = BeamConfig {
        beam_width: 8,
        ..Default::default()
    };
    let (_cost, extracted) = beam_extract(&egraph, root, &beam_cfg);
    let program = build_dispatch_program_from_extracted(
        &extracted,
        egraph,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
    );
    let dispatch = &program.dispatches[0];

    // ── Lower to naga module ──
    let module =
        naga_codegen::lower_dispatch_program(naga_codegen::verify(&program).expect("verify"));

    // ── wgpu setup ──
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("no GPU adapter found");
    let (device, queue) = request_test_device(&adapter);

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("reduce_test"),
        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
    });

    // ── Input data ──
    let a_data: Vec<f32> = (0..ROWS * COLS).map(|i| (i % 13) as f32 * 0.1).collect();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_0"),
        contents: bytemuck::cast_slice(&a_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_elems = (dispatch.workgroups * SIMD_WIDTH) as usize;
    let output_bytes = (output_elems * size_of::<f32>()) as u64;

    let buf_out = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("output"),
        size: output_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let buf_staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging"),
        size: output_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("reduce_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("dispatch_0"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("reduce_bind_group"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buf_a.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buf_out.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("reduce_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reduce_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        let physical_wgs = dispatch.workgroups / dispatch.simdgroups.max(1);
        pass.dispatch_workgroups(physical_wgs, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&buf_out, 0, &buf_staging, 0, output_bytes);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = buf_staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().expect("channel closed").expect("map failed");

    let gpu_result: Vec<f32> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();

    // ── CPU reference: row-wise sum ──
    let mut cpu_result = vec![0.0f32; ROWS as usize];
    for row in 0..ROWS {
        for col in 0..COLS {
            cpu_result[row as usize] += a_data[(row * COLS + col) as usize];
        }
    }

    let max_err = gpu_result[..ROWS as usize]
        .iter()
        .zip(&cpu_result)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 1e-2,
        "GPU sum reduce max error {max_err} exceeds tolerance 1e-2"
    );
}
