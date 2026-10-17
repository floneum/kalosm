//! Compare two actual compiler schedules inside one resident pure GEMV graph.
//! Both command lists remain inside their resolve callbacks, before either
//! arena or uniform buffer is recycled. No command or tensor result is cached.
use fusor::device::Device;
use fusor_gpu::{launch::CommandRecord, target::PlanObserver};
use fusor_ir::ir::{
    Op,
    launch::{Launch, SchedPoint, SgemvParams},
};
use std::sync::{Arc, Mutex, OnceLock, Weak};

type ReplaySample = Arc<Mutex<Option<(f64, usize)>>>;

pub fn install(device: &Device, repeats: usize, sink: ReplaySample) {
    let parts: Vec<u32> = std::env::var("LAB_COMPARE_CELL")
        .expect("comparison cell")
        .split(',')
        .map(|v| v.parse().expect("integer schedule field"))
        .collect();
    let [vector, subgroups, cols, parts, gap] = parts[..] else {
        panic!("LAB_COMPARE_CELL requires vector,subgroups,cols,parts,gap");
    };
    let candidate = SgemvParams {
        vector,
        subgroups,
        cols,
        parts,
        gap,
    };
    assert_eq!(vector, 32);
    assert!([1, 2, 4, 8, 16, 32].contains(&subgroups));
    assert!(cols.is_multiple_of(subgroups) && (1..=8).contains(&(cols / subgroups)));
    assert_eq!((parts, gap), (4, 32));
    let target = device.backend().gpu_target().unwrap();
    assert!(
        subgroups * 32
            <= target
                .device()
                .limits_used()
                .max_compute_invocations_per_workgroup
    );
    let dependent = std::env::var_os("LAB_COMPARE_DEPENDENT").is_some();
    let arithmetic = std::env::var("LAB_COMPARE_Q4_VARIANT")
        .ok()
        .map(|v| v.parse::<u8>().expect("Q4 variant integer"));
    assert!(arithmetic.is_none_or(|n| (1..=3).contains(&n)));
    assert!(std::env::var_os("FUSOR_PROFILE_Q4_VARIANT").is_none());
    let owner = Arc::new(OnceLock::<Weak<PlanObserver>>::new());
    let restore = owner.clone();
    let probe = device.clone();
    let round = Arc::new(Mutex::new(0usize));
    let observer: Arc<PlanObserver> = Arc::new(
        move |plan, graph, binds, base_identities, baseline_commands| {
            let target = probe.backend().gpu_target().unwrap();
            let iteration = *round.lock().unwrap();
            let mut variant = plan.clone();
            let mut output_buffers = Vec::new();
            let mut input_buffers = Vec::new();
            for launch in &plan.launches {
                assert!(
                    matches!(
                        &graph.node(launch.root).op,
                        Op::Launch(Launch::Contract { .. })
                    ),
                    "paired geometry permits pure contraction launches only"
                );
                assert!(matches!(
                    plan.extraction.theta.get(&launch.root),
                    Some(SchedPoint::Sgemv(_))
                ));
                variant
                    .extraction
                    .theta
                    .insert(launch.root, SchedPoint::Sgemv(candidate));
                let buffer = plan
                    .buffers
                    .iter()
                    .find(|b| b.value == launch.root)
                    .expect("output plan");
                assert!(buffer.arena.is_none(), "output must be externally bound");
                let bytes = buffer.elements.as_const().expect("fixed output size")
                    * buffer.dtype.byte_size();
                let buf = binds
                    .buffers
                    .get(&launch.root)
                    .expect("bound output")
                    .clone();
                output_buffers.push((buf, bytes));
                if let Op::Launch(Launch::Contract { a, .. }) = &graph.node(launch.root).op {
                    assert_eq!(a.ops.len(), 1, "one direct activation input");
                    let input = binds
                        .buffers
                        .get(&a.ops[0].src)
                        .expect("bound activation input");
                    input_buffers.push(input.clone());
                }
            }
            let baseline_commands = baseline_commands.to_vec();
            let base_identities = base_identities.to_vec();
            let changed = arithmetic.is_some()
                || plan
                    .launches
                    .iter()
                    .any(|l| plan.extraction.theta[&l.root] != SchedPoint::Sgemv(candidate));
            let nested_probe = probe.clone();
            let nested_sink = sink.clone();
            let guards =
                dependent.then(|| dependency_guards(target, &input_buffers, &output_buffers));
            target.set_plan_observer(Some(Arc::new(move |_, _, _, identities, candidate_commands| {
            let target = nested_probe.backend().gpu_target().unwrap();
            let launcher = target.launcher();
            assert_eq!(baseline_commands.len(), candidate_commands.len());
            assert!(!changed || identities != base_identities, "candidate schedule did not change the lowered kernels");
            if iteration == 0 {
                eprintln!("GEOMETRY_VARIANTS baseline={base_identities:?} candidate={identities:?}");
            }
            let lists = [&baseline_commands[..], candidate_commands];
            for commands in lists {
                assert!(commands.iter().all(|c| matches!(c, CommandRecord::Dispatch { name, .. }
                    if name.starts_with("sgemv_packed_q4") || name.starts_with("sgemv_q6_coalesced"))),
                    "both schedules must use the requested packed GEMV path");
            }
            let order = if iteration.is_multiple_of(2) { [0, 1, 1, 0] } else { [1, 0, 0, 1] };
            let mut times: [Vec<f64>; 2] = Default::default();
            let mut reference: Option<Vec<Vec<u8>>> = None;
            let mut checked = 0usize;
            let mut max_error = 0f32;
            launcher.set_tuning(true);
            let query = launcher.timestamp_query_set(1).expect("GPU timestamps");
            for arm in order {
                let commands = lists[arm];
                let mut records = Vec::with_capacity(repeats * commands.len() * if dependent { 2 } else { 1 });
                for _ in 0..repeats {
                    for (index, command) in commands.iter().enumerate() {
                        records.push(command.clone());
                        if let Some(guards) = &guards { records.push(guards[index].clone()); }
                    }
                }
                launcher.encode_command_records(&records, Some(&query), fusor_gpu::launch::TimingMode::Whole)
                    .expect("paired replay");
                launcher.poll_wait().expect("paired completion");
                let us = launcher.read_timestamps(target.pool(), &query, 1).expect("paired timestamps")[0]
                    / (repeats * commands.len()) as f64;
                assert!(us.is_finite() && us > 0., "invalid paired timestamp");
                times[arm].push(us);
                if iteration < 3 {
                    let outputs: Vec<_> = output_buffers.iter()
                        .map(|(buf, bytes)| pollster::block_on(target.readback(buf, *bytes)).expect("paired output"))
                        .collect();
                    if let Some(reference) = &reference {
                        for (a, b) in reference.iter().zip(&outputs) {
                            let (count, error) = compare(a, b);
                            checked += count;
                            max_error = max_error.max(error);
                        }
                    } else {
                        // Validate finiteness even for the first arm.
                        for bytes in &outputs { compare(bytes, bytes); }
                        reference = Some(outputs);
                    }
                }
            }
            launcher.set_tuning(false);
            let a = (times[0][0] + times[0][1]) / 2.;
            let b = (times[1][0] + times[1][1]) / 2.;
            println!("GEOMETRY_PAIR {}", serde_json::json!({
                "iteration":iteration, "baseline_us":a, "candidate_us":b, "speedup":a/b,
                "order":order, "baseline_samples_us":times[0], "candidate_samples_us":times[1],
                "candidate_cell":[vector,subgroups,cols,parts,gap], "checked_values":checked,
                "dependent_input_guard":dependent,
                "q4_variant":arithmetic.map(|n| [0,n]),
                "max_abs_error":max_error, "dispatches_per_arm":candidate_commands.len(),
            }));
            *nested_sink.lock().unwrap() = Some((a * baseline_commands.len() as f64, baseline_commands.len()));
        })));
            // The nested callback holds both live command lists and compares them
            // before the nested resolve can recycle its temporary storage.
            target
                .set_q4_variant(arithmetic)
                .expect("bounded arithmetic selector");
            let result = target.resolve(&variant, graph, binds);
            target
                .set_q4_variant(None)
                .expect("restore arithmetic selector");
            target.set_plan_observer(restore.get().expect("observer owner").upgrade());
            result.expect("candidate lowering/resolve");
            *round.lock().unwrap() += 1;
        },
    );
    assert!(
        owner.set(Arc::downgrade(&observer)).is_ok(),
        "one observer owner"
    );
    target.set_plan_observer(Some(observer));
}

/// Force a dependent activation access after each GEMV. Valid finite results
/// leave x unchanged; invalid results zero it and fail the CPU reference.
/// Both arms include the same extra dispatch in their timing.
fn dependency_guards(
    target: &fusor_gpu::GpuTarget,
    inputs: &[fusor_ir::target::Buf],
    outputs: &[(fusor_ir::target::Buf, u64)],
) -> Vec<CommandRecord> {
    use fusor_gpu::pool::GpuBuffer;
    let source = r#"
@group(0) @binding(0) var<storage,read> result:array<f32>;
@group(0) @binding(1) var<storage,read_write> input:array<f32>;
@compute @workgroup_size(32)
fn main(@builtin(local_invocation_index) lane:u32) {
    if (lane == 0u) {
        let y = result[0];
        if (!(abs(y) <= 1e30)) { input[0] = 0.0; }
        else { input[0] = input[0]; }
    }
}
"#;
    let device = target.device().device();
    let module = naga::front::wgsl::parse_str(source).expect("dependency WGSL");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("geometry dependency guard"),
        source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module)),
    });
    let pipeline = Arc::new(
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("geometry dependency guard"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }),
    );
    inputs
        .iter()
        .zip(outputs)
        .map(|(input, (output, _))| {
            let input = GpuBuffer::of(input, "dependency input").unwrap();
            let output = GpuBuffer::of(output, "dependency output").unwrap();
            let bind_group = Arc::new(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input.buffer.as_entire_binding(),
                    },
                ],
            }));
            CommandRecord::Dispatch {
                name: "geometry_dependency_guard",
                pipeline: pipeline.clone(),
                bind_group,
                grid: [1, 1, 1],
            }
        })
        .collect()
}

fn compare(a: &[u8], b: &[u8]) -> (usize, f32) {
    assert_eq!(a.len(), b.len(), "paired output lengths differ");
    assert!(a.len().is_multiple_of(4));
    let mut max_error = 0f32;
    for (index, (u, v)) in a.chunks_exact(4).zip(b.chunks_exact(4)).enumerate() {
        let u = f32::from_le_bytes(u.try_into().unwrap());
        let v = f32::from_le_bytes(v.try_into().unwrap());
        let error = (u - v).abs();
        assert!(
            u.is_finite() && v.is_finite() && error <= 0.001 + 0.0001 * u.abs().max(v.abs()),
            "paired output mismatch at {index}: {u} != {v}"
        );
        max_error = max_error.max(error);
    }
    (a.len() / 4, max_error)
}

#[cfg(test)]
mod tests {
    use super::compare;
    #[test]
    #[should_panic(expected = "paired output mismatch")]
    fn rejects_nonfinite_outputs() {
        compare(&f32::NAN.to_le_bytes(), &f32::NAN.to_le_bytes());
    }
    #[test]
    #[should_panic(expected = "paired output mismatch")]
    fn rejects_finite_corruption() {
        compare(&1f32.to_le_bytes(), &2f32.to_le_bytes());
    }
}
