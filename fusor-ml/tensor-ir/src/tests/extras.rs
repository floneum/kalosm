//! Integration tests for the tensor IR.

#[cfg(feature = "runtime")]
use super::runtime_gpu::request_test_device;
use crate::analysis::TensorAnalysis;
use crate::builders::IrBuilder;
#[cfg(feature = "runtime")]
use crate::extractor::BeamConfig;
use crate::language::*;
#[cfg(feature = "runtime")]
use crate::naga_codegen;
use crate::rules::{self, Phase, RunnerConfig};
#[cfg(feature = "runtime")]
use crate::skeleton::{beam_extract_valid_candidates, build_dispatch_program_from_extracted};
use crate::types::*;
#[cfg(feature = "runtime")]
use std::mem::size_of;

/// Test that the Phase 4 theta-split cooperative rule fires for large reductions
/// and that the ReduceSimd + shuffle tree is present in the e-graph.
#[test]
fn test_theta_split_cooperative() {
    // Build a simple reduce: out[row] = sum(A[row, :]) with COLS=128
    // Phase 1 creates Theta(init, 128, ...) which is > SIMD_WIDTH=32.
    // Phase 4 should split into Theta(init, 4, ...) + ReduceSimd.
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(128), Dim::Lit(128)]), DType::F32);
    let _red = b.reduce(a, 1, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _root = egraph.add_expr(&b.expr);

    // Run Phase 1 (lowering) + Phase 4 (reduction optimizations)
    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    // Check that ReduceSimd exists somewhere in the e-graph
    let mut has_reduce_simd = false;
    for class in egraph.classes() {
        for node in class.iter() {
            if matches!(node, TensorIr::Simd(SimdNode::ReduceSimd { .. })) {
                has_reduce_simd = true;
            }
        }
    }

    assert!(
        has_reduce_simd,
        "Phase 4 theta-split rule should have generated ReduceSimd node"
    );

    // Also check for Shuffle nodes (from ReduceSimd → shuffle tree expansion)
    let mut has_shuffle = false;
    for class in egraph.classes() {
        for node in class.iter() {
            if matches!(node, TensorIr::Simd(SimdNode::Shuffle(_))) {
                has_shuffle = true;
            }
        }
    }

    assert!(
        has_shuffle,
        "Phase 4 should have expanded ReduceSimd into shuffle tree"
    );
}

#[test]
fn test_theta_split_cooperative_with_tail() {
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(128), Dim::Lit(100)]), DType::F32);
    let _red = b.reduce(a, 1, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 10,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    let has_reduce_simd = egraph.classes().any(
        |class: &egg::EClass<TensorIr, <TensorAnalysis as egg::Analysis<TensorIr>>::Data>| {
            class
                .iter()
                .any(|n| matches!(n, TensorIr::Simd(SimdNode::ReduceSimd { .. })))
        },
    );
    assert!(
        has_reduce_simd,
        "theta-split cooperative reduction should support non-multiple-of-32 tails"
    );

    let has_select_guard = egraph.classes().any(
        |class: &egg::EClass<TensorIr, <TensorAnalysis as egg::Analysis<TensorIr>>::Data>| {
            class
                .iter()
                .any(|n| matches!(n, TensorIr::TernOp(TernaryOp::Select, _)))
        },
    );
    assert!(
        has_select_guard,
        "tail-handling cooperative reduction should guard out-of-bounds lanes with select"
    );
}

/// GPU test: cooperative sum-reduce with shuffle-down using only Phase 1 + Phase 4.
/// This verifies the theta-split rule produces a numerically correct kernel.
#[test]
#[cfg(feature = "runtime")]
fn test_gpu_cooperative_reduce() {
    use wgpu::util::DeviceExt;

    const ROWS: u32 = 128;
    const COLS: u32 = 128;

    // ── Build IR: out[row] = sum(A[row, :]) ──
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(ROWS), Dim::Lit(COLS)]), DType::F32);
    let _red = b.reduce(a, 1, ReduceOp::Add);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    // Run only Phase 1 + Phase 4 (no tiling/stride-zero) to force cooperative path
    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    // Use a cost model that doesn't penalize higher workgroup counts,
    // so the cooperative variant with ReduceSimd is preferred.
    use crate::extractor::SyntheticCostModel;
    let beam_cfg = BeamConfig {
        beam_width: 8,
        cost_model: SyntheticCostModel {
            // Zero workgroup penalty — let the reduction cost dominate
            shuffle_cost: 0.5,
            ..Default::default()
        },
        device: DeviceProfile::default(),
    };
    let candidates = beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        16,
    );
    let program = candidates
        .into_iter()
        .map(|(_cost, extracted)| {
            build_dispatch_program_from_extracted(
                &extracted,
                egraph.clone(),
                &DeviceProfile::default(),
                &LoweringOptions::default(),
            )
        })
        .find(|program| program.dispatches.len() == 1)
        .expect("expected a fused SGEMV dispatch candidate");
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
        label: Some("coop_reduce_test"),
        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
    });

    // ── Input data ──
    let a_data: Vec<f32> = (0..ROWS * COLS).map(|i| (i % 13) as f32 * 0.1).collect();

    let buf_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("input_0"),
        contents: bytemuck::cast_slice(&a_data),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let output_elems = (dispatch.workgroups * 32) as usize;
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
        label: Some("coop_reduce_pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("dispatch_0"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("coop_reduce_bind_group"),
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
        label: Some("coop_reduce_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("coop_reduce_pass"),
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
        max_err < 1e-1,
        "GPU cooperative reduce max error {max_err} exceeds tolerance 1e-1"
    );
}

/// GPU test: SGEMV y[M] = A[M,K] @ x[K] using cooperative shuffle-down reduction.
/// Expressed as matmul(A[M,K], x[K,1]) → y[M,1].
/// Verifies the theta-split cooperative rule works for fused reduce+ewise (matmul) patterns.
#[test]
#[cfg(feature = "runtime")]
fn test_gpu_cooperative_sgemv() {
    const M: u32 = 64;
    const K: u32 = 128;

    // ── Build IR: y[M] = A[M,K] @ x[K,1] via matmul(A, x, M, N=1, K) ──
    // The matmul builder creates Reduce(axis=2, Elementwise([M,1,K], mul(A,x)))
    // Rule 2b fuses this into a single Dispatch with K-loop Theta.
    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(M), Dim::Lit(K)]), DType::F32);
    let x = b.input(1, Shape(vec![Dim::Lit(K), Dim::Lit(1)]), DType::F32);
    let _y = super::build_binary_mul_add_contraction_ir(&mut b, a, x, M, 1, K);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let root = egraph.add_expr(&b.expr);

    // Run all phases to get fused dispatch + cooperative split
    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    let egraph = crate::saturate_phases(egraph, Phase::all(), &config);

    // Verify ReduceSimd was generated
    let has_reduce_simd = egraph.classes().any(
        |class: &egg::EClass<TensorIr, <TensorAnalysis as egg::Analysis<TensorIr>>::Data>| {
            class
                .iter()
                .any(|n| matches!(n, TensorIr::Simd(SimdNode::ReduceSimd { .. })))
        },
    );
    assert!(
        has_reduce_simd,
        "Phase 4 should generate ReduceSimd for SGEMV K-loop"
    );

    // Extract — use low shuffle cost to prefer cooperative variant
    use crate::extractor::SyntheticCostModel;
    let beam_cfg = BeamConfig {
        beam_width: 16,
        cost_model: SyntheticCostModel {
            shuffle_cost: 0.5,
            ..Default::default()
        },
        device: DeviceProfile::default(),
    };
    let candidates = beam_extract_valid_candidates(
        &egraph,
        root,
        &beam_cfg,
        &DeviceProfile::default(),
        &LoweringOptions::default(),
        16,
    );
    let program = candidates
        .into_iter()
        .map(|(_cost, extracted)| {
            build_dispatch_program_from_extracted(
                &extracted,
                egraph.clone(),
                &DeviceProfile::default(),
                &LoweringOptions::default(),
            )
        })
        .find(|program| program.dispatches.len() == 1)
        .expect("expected a fused SGEMV dispatch candidate");

    println!("=== SGEMV dispatch program ===");
    println!("{program}");
    match crate::lower_to_wgsl(&program) {
        Ok(wgsl) => println!("=== WGSL ===\n{wgsl}"),
        Err(e) => println!("WGSL error: {e}"),
    }

    assert_eq!(
        program.dispatches.len(),
        1,
        "SGEMV should produce a single fused dispatch"
    );

    // Use GpuContext runtime for execution
    let ctx = crate::runtime::GpuContext::new();
    let a_data: Vec<f32> = (0..M * K).map(|i| (i % 7) as f32 * 0.1).collect();
    let x_data: Vec<f32> = (0..K).map(|i| (i % 5) as f32 * 0.2).collect();

    let gpu_result = ctx.execute(&program, &[&a_data, &x_data]);

    // ── CPU reference: y[m] = sum_k A[m,k] * x[k] ──
    let mut cpu_result = vec![0.0f32; M as usize];
    for m in 0..M as usize {
        for k in 0..K as usize {
            cpu_result[m] += a_data[m * K as usize + k] * x_data[k];
        }
    }

    let max_err = gpu_result[..M as usize]
        .iter()
        .zip(&cpu_result)
        .map(|(g, c)| (g - c).abs())
        .fold(0.0f32, f32::max);

    println!("SGEMV max_err = {max_err}");
    assert!(
        max_err < 1e-1,
        "GPU SGEMV max error {max_err} exceeds tolerance 1e-1"
    );
}

/// Test that SGEMV vector broadcast gets promoted to threadgroup memory.
/// The vector x in y = A @ x is read identically by all threads and should
/// be cooperatively loaded into threadgroup memory.
#[test]
fn test_sgemv_vector_broadcast_promotion() {
    const M: u32 = 64;
    const K: u32 = 32;

    let mut b = IrBuilder::new();
    let a = b.input(0, Shape(vec![Dim::Lit(M), Dim::Lit(K)]), DType::F32);
    let x = b.input(1, Shape(vec![Dim::Lit(K), Dim::Lit(1)]), DType::F32);
    let _y = super::build_binary_mul_add_contraction_ir(&mut b, a, x, M, 1, K);

    let mut egraph = egg::EGraph::<TensorIr, TensorAnalysis>::default();
    let _root = egraph.add_expr(&b.expr);

    let config = RunnerConfig {
        iter_limit: 10,
        node_limit: 50_000,
        time_limit_secs: 30,
        device: DeviceProfile::default(),
        lowering: LoweringOptions::default(),
    };
    // Phase 1 lowers, Phase 2 tiles, Phase 3 promotes to TG
    let egraph = rules::saturate_phases(egraph, &[Phase::Lowering, Phase::LateDispatch], &config);

    // Check that Phase 2 created nested Thetas
    let has_nested_theta = egraph.classes().any(|class| {
        class.iter().any(|n| {
            if let TensorIr::Simd(SimdNode::Theta {
                children: [_, count, _],
                ..
            }) = n
            {
                if let Some(ScalarValue::U32(v)) = &egraph[*count].data.constant {
                    *v <= 16 && *v > 1
                } else {
                    false
                }
            } else {
                false
            }
        })
    });
    assert!(
        has_nested_theta,
        "Phase 2 should create tiled (nested) Thetas for SGEMV"
    );

    // Check that the e-graph contains a Threadgroup load for input_1
    let has_input1_tg = egraph.classes().any(|class| {
        class.iter().any(|n| {
            matches!(
                n,
                TensorIr::Simd(SimdNode::Load { tier: MemTier::Threadgroup(buf), .. })
                if *buf == BufferRef::Input(1)
            )
        })
    });
    assert!(
        has_input1_tg,
        "Phase 3 should promote input_1 (vector x) to threadgroup memory via broadcast detection"
    );
}
