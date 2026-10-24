//! Small repeatable quantized matrix experiment: real extraction and lowering,
//! CPU numerical checks, cold versus warm timing, full outputs for A/B comparison.
//! Usage: qmatmul_lab M_CSV N K [FORMAT=q4k] [ITERATIONS=11]
//! Batched replay requires FUSOR_AUTOTUNE_MIN_MACS=18446744073709551615,
//! so speculative tuning submissions cannot contaminate controlled measurements.
//! Set FUSOR_PROFILE_SUBMIT_SIZE=2147483647 for replay timing without CPU
//! encoding gaps between submissions. All QFmt formats and both storage layouts
//! use the same fixture, reference checks, and measurement path.
use fusor::{Dim, Dtype, device::Device, quantized::QMatrix};
use fusor_ir::dtype::{QFmt, QLayout};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::Instant;
#[path = "support/paired_geometry.rs"]
mod paired_geometry;

fn weights(n: usize, k: usize, fmt: QFmt) -> Vec<u8> {
    let size = fmt.block_bytes(QLayout::Native) as usize;
    let elements = fmt.block_elements() as usize;
    let fields = fusor_gguf::blocks::block_fields(fmt, QLayout::Native);
    let mut bytes = vec![0u8; n * k / elements * size];
    for (b, block) in bytes.chunks_exact_mut(size).enumerate() {
        for (i, v) in block.iter_mut().enumerate() {
            *v = (i.wrapping_mul(37) + b.wrapping_mul(17) + b / (k / elements)) as u8;
        }
        for (offset, value) in [(Some(fields.scale), 1. / 64.), (fields.min, 1. / 128.)] {
            if let Some(offset) = offset {
                let offset = offset as usize;
                let value: f32 = value;
                if fields.scale_is_f16 {
                    block[offset..offset + 2]
                        .copy_from_slice(&half::f16::from_f32(value).to_le_bytes());
                } else {
                    block[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
            }
        }
    }
    bytes
}

fn validate(w: &[u8], x: &[f32], y: &[f32], m: usize, n: usize, k: usize, fmt: QFmt) -> f64 {
    assert_eq!(y.len(), m * n);
    assert!(y.iter().all(|v| v.is_finite()), "nonfinite GPU output");
    let columns: Vec<_> = (0..n.min(33))
        .map(|i| i * (n - 1) / (n.min(33) - 1).max(1))
        .collect();
    let size = fmt.block_bytes(QLayout::Native) as usize;
    let elements = fmt.block_elements() as usize;
    let mut max_error = 0f64;
    for col in columns {
        let mut decoded = vec![0f32; k];
        for j in 0..k / elements {
            let off = (col * k / elements + j) * size;
            fusor_gguf::blocks::cpu_dequantize_block(
                fmt,
                QLayout::Native,
                &w[off..off + size],
                &mut decoded[j * elements..(j + 1) * elements],
            );
        }
        for row in 0..m {
            let expected: f64 = decoded
                .iter()
                .zip(&x[row * k..(row + 1) * k])
                .map(|(a, b)| f64::from(*a) * f64::from(*b))
                .sum();
            let error = (f64::from(y[row * n + col]) - expected).abs();
            max_error = max_error.max(error);
            assert!(
                error <= 0.05 + 0.002 * expected.abs(),
                "CPU reference mismatch at [{row},{col}]: {} != {expected}, error={error}",
                y[row * n + col]
            );
        }
    }
    max_error
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let ms: Vec<usize> = args
        .next()
        .ok_or("usage: qmatmul_lab M_CSV N K [FORMAT=q4k] [ITERATIONS]")?
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    let n: usize = args.next().ok_or("N required")?.parse()?;
    let k: usize = args.next().ok_or("K required")?.parse()?;
    let format_name = args.next().unwrap_or_else(|| "q4k".into());
    let fmt = QFmt::ALL
        .into_iter()
        .find(|fmt| format!("{fmt:?}").eq_ignore_ascii_case(&format_name))
        .ok_or("FORMAT must be one of q4_0, q5_0, q8_0, q4k, q5k, q6k")?;
    let elements = fmt.block_elements() as usize;
    let iters: usize = args.next().unwrap_or_else(|| "11".into()).parse()?;
    assert!(
        args.next().is_none()
            && !ms.is_empty()
            && ms.iter().all(|m| (1..=512).contains(m))
            && n > 0
            && n <= 131072
            && k > 0
            && k <= 32768
            && k.is_multiple_of(elements)
            && (3..=101).contains(&iters)
    );
    let repeats: usize = std::env::var("LAB_BATCH_REPEAT")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1);
    assert!((1..=128).contains(&repeats));
    if repeats > 1
        && std::env::var("FUSOR_AUTOTUNE_MIN_MACS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            != Some(u64::MAX)
    {
        return Err(
            "LAB_BATCH_REPEAT requires FUSOR_AUTOTUNE_MIN_MACS=18446744073709551615".into(),
        );
    }
    let step_control = std::env::var_os("LAB_STEP_CONTROL").is_some();
    assert!(
        !step_control || ms.len() == 1,
        "step control requires one M value"
    );
    let wbytes = if let Some(path) = std::env::var_os("LAB_WEIGHT_FILE") {
        let bytes = std::fs::read(path)?;
        assert_eq!(
            bytes.len(),
            n * k / elements * fmt.block_bytes(QLayout::Native) as usize,
            "LAB_WEIGHT_FILE must contain exactly N*K native quantized weights"
        );
        bytes
    } else {
        weights(n, k, fmt)
    };
    let layout = match std::env::var("LAB_QLAYOUT").as_deref() {
        Ok("f32scales") => QLayout::F32Scales,
        Ok("native") | Err(_) => QLayout::Native,
        _ => return Err("LAB_QLAYOUT must be native or f32scales".into()),
    };
    let mut stored = Vec::new();
    fusor_gguf::repack::repack(fmt, QLayout::Native, layout, &wbytes, &mut stored)?;
    let device = Device::gpu_blocking()?;
    let graph = device.graph().clone();
    let w = QMatrix::from_raw_bytes(
        &graph,
        fmt,
        layout,
        [Dim::Const(n as u64), Dim::Const(k as u64)],
        &stored,
    )?;
    let copies: usize = std::env::var("LAB_WEIGHT_COPIES")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(1);
    assert!((1..=64).contains(&copies));
    assert!(
        stored.len() as u64 * copies as u64 <= 5 * 1024 * 1024 * 1024,
        "weight working set exceeds the 5 GiB experiment budget"
    );
    assert!(
        copies == 1 || std::env::var_os("LAB_DEQUANT_CACHE").is_none(),
        "weight rotation currently tests quantized storage only"
    );
    let mut weights = vec![w];
    for _ in 1..copies {
        weights.push(QMatrix::from_raw_bytes(
            &graph,
            fmt,
            layout,
            [Dim::Const(n as u64), Dim::Const(k as u64)],
            &stored,
        )?);
    }
    let w = &weights[0];
    eprintln!(
        "LAB_WORKING_SET copies={copies} bytes={}",
        stored.len() * copies
    );
    let cache_start = Instant::now();
    let dense = std::env::var("LAB_DEQUANT_CACHE")
        .ok()
        .map(|precision| {
            let dtype = match precision.as_str() {
                "f32" => Dtype::F32,
                "f16" => Dtype::F16,
                _ => panic!("LAB_DEQUANT_CACHE must be f32 or f16"),
            };
            w.dequantize()?.cast(dtype)?.materialize()
        })
        .transpose()?;
    if let Some(dense) = &dense {
        eprintln!(
            "LAB_WEIGHT_CACHE dtype={:?} bytes={} setup_ms={:.3}",
            dense.dtype(),
            n as u64 * k as u64 * dense.dtype().byte_size(),
            cache_start.elapsed().as_secs_f64() * 1e3
        );
    }
    // Logical storage volume, not measured DRAM traffic or cache misses.
    let logical_weight_bytes = dense.as_ref().map_or(stored.len() as u64, |dense| {
        n as u64 * k as u64 * dense.dtype().byte_size()
    });
    let target = device.backend().gpu_target().unwrap();
    if let Some(dir) = std::env::var_os("LAB_TRACE_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir)?;
        target.set_kernel_observer(Some(Arc::new(move |ir, identity, uniforms| {
            let options = fusor_visualize::kernel::CaptureOptions {
                max_actions: 2_000_000,
                known_bindings: uniforms.map(|u| std::collections::BTreeMap::from([(0,u.to_bytes())])).unwrap_or_default(),
                ..Default::default()
            };
            let trace = fusor_visualize::kernel::capture(ir,options).expect("requested kernel address capture");
            let name = format!("{identity:032x}");
            let trace_path = dir.join(format!("{name}.trace.json"));
            serde_json::to_writer(std::fs::File::create(&trace_path).unwrap(),&trace).unwrap();
            let mut sim = fusor_visualize::Simulator::new(trace,Default::default()).unwrap();
            while sim.step().is_some() {}
            let summary = serde_json::json!({"name":ir.name,"identity":name,"grid":ir.grid,"workgroups":[[0,0,0]],
                "trace":trace_path,"metrics":sim.totals(),"config":sim.config(),"notes":sim.trace().notes});
            std::fs::write(dir.join(format!("{name}.json")),serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
            eprintln!("LAB_TRACE {summary}");
        })));
    }
    let replay_sample = Arc::new(Mutex::new(None::<(f64, usize)>));
    if std::env::var_os("LAB_COMPARE_CELL").is_some() {
        assert!(ms == [1] && repeats > 1 && dense.is_none());
        assert!(
            stored.len() * copies <= 512 * 1024 * 1024,
            "paired geometry weight limit is 512 MiB"
        );
        paired_geometry::install(&device, repeats, replay_sample.clone());
    } else if repeats > 1 || std::env::var_os("LAB_PLAN").is_some() {
        let probe = device.clone();
        let sink = replay_sample.clone();
        target.set_plan_observer(Some(Arc::new(move |plan, graph, _, _, commands| {
            if std::env::var_os("LAB_PLAN").is_some() {
                for command in commands {
                    match command {
                        fusor_gpu::launch::CommandRecord::Dispatch { name, grid, .. } => {
                            eprintln!("LAB_COMMAND dispatch={name} grid={grid:?}")
                        }
                        fusor_gpu::launch::CommandRecord::CopyBuffer { .. }
                        | fusor_gpu::launch::CommandRecord::Transfers { .. } => {
                            eprintln!("LAB_COMMAND copy_bytes={}", command.copy_bytes().unwrap_or_default())
                        }
                    }
                }
                for launch in &plan.launches {
                    eprintln!(
                        "LAB_LAUNCH root={:?} members={:?}",
                        graph.node(launch.root).op.tag(),
                        launch
                            .members
                            .iter()
                            .map(|id| graph.node(*id).op.tag())
                            .collect::<Vec<_>>()
                    );
                    for id in std::iter::once(launch.root).chain(launch.members.iter().copied()) {
                        if let fusor_ir::ir::Op::Launch(fusor_ir::ir::launch::Launch::Contract {
                            family,
                            m,
                            n,
                            k,
                            a,
                            b,
                            ..
                        }) = &graph.node(id).op
                        {
                            eprintln!(
                                "LAB_SELECTED_CONTRACT family={family:?} m={m:?} n={n:?} k={k:?}"
                            );
                            for operand in a.ops.iter().chain(&b.ops) {
                                for member in graph.class_ids(graph.class_of(operand.src)) {
                                    if let fusor_ir::ir::Op::Logical(fusor_ir::ir::logical::Logical::Leaf(
                                        fusor_ir::ir::logical::LeafKind::Quantized { fmt, layout, .. }
                                    )) = &graph.node(member).op {
                                        eprintln!("LAB_SELECTED_QUANT fmt={fmt:?} layout={layout:?} block_bytes={}", fmt.block_bytes(*layout));
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if repeats == 1 {
                return;
            }
            // This benchmark's graph is a pure matrix expression. Replay while
            // its uniforms and arena are still live, before resolve recycles them.
            let target = probe.backend().gpu_target().unwrap();
            let launcher = target.launcher();
            // The launcher allocates queries only while timing is enabled.
            // Autotuning is disabled above, and the outer resolve has timing off.
            launcher.set_tuning(true);
            let query = launcher
                .timestamp_query_set(1)
                .expect("GPU timestamp query");
            let repeated: Vec<_> = (0..repeats)
                .flat_map(|_| commands.iter().cloned())
                .collect();
            let encode_start = Instant::now();
            launcher
                .encode_command_records(
                    &repeated,
                    Some(&query),
                    fusor_gpu::launch::TimingMode::Whole,
                )
                .expect("replay submission");
            let encode_us = encode_start.elapsed().as_secs_f64() * 1e6;
            launcher.poll_wait().expect("replay completion");
            let us = launcher
                .read_timestamps(target.pool(), &query, 1)
                .expect("retired replay timestamps")[0]
                / repeats as f64;
            let dispatches = commands
                .iter()
                .filter(|command| {
                    matches!(command, fusor_gpu::launch::CommandRecord::Dispatch { .. })
                        && !command.is_empty_dispatch()
                })
                .count();
            if std::env::var_os("LAB_PLAN").is_some() {
                let submissions = (dispatches * repeats).div_ceil(
                    fusor_gpu::launch::dispatches_per_submit(dispatches * repeats, launcher.backend()),
                );
                eprintln!("LAB_REPLAY submissions={submissions} encode_us={encode_us:.3} gpu_span_us={:.3}", us * repeats as f64);
            }
            *sink.lock().unwrap() = Some((us, dispatches));
            launcher.set_tuning(false);
        })));
    }
    for m in ms {
        let x = graph.leaf(
            "lab-input",
            &[Dim::Const(m as u64), Dim::Const(k as u64)],
            Dtype::F32,
        )?;
        let f16 = std::env::var_os("LAB_MATMUL_F16").is_some();
        let y = if let Some(dense) = &dense {
            x.cast(dense.dtype())?.matmul_t(dense)?.cast(Dtype::F32)?
        } else if f16 {
            w.q_mat_mul(&x.cast(Dtype::F16)?)?.cast(Dtype::F32)?
        } else {
            w.q_mat_mul(&x)?
        };
        let mut outputs = vec![y.clone()];
        for weight in &weights[1..] {
            outputs.push(if f16 {
                weight.q_mat_mul(&x.cast(Dtype::F16)?)?.cast(Dtype::F32)?
            } else {
                weight.q_mat_mul(&x)?
            });
        }
        let mut gpu_samples = vec![];
        let mut wall_samples = vec![];
        let mut cold_ms = 0.0;
        let mut max_error = 0f64;
        let mut launches = 0;
        for iteration in 0..iters + 3 {
            if step_control {
                println!(
                    "LAB_CONTROL {}",
                    serde_json::json!({"event": "ready", "iteration": iteration})
                );
                std::io::stdout().flush()?;
                let mut command = String::new();
                if std::io::stdin().read_line(&mut command)? == 0 || command.trim() != "run" {
                    return Err("step control expected run on stdin".into());
                }
            }
            let mut sample_us = None;
            let divisor = if std::env::var_os("LAB_FRACTIONAL_INPUT").is_some() {
                63.37
            } else {
                64.0
            };
            let data: Vec<f32> = (0..m * k)
                .map(|i| (((i * 13 + i / k * 7 + iteration * 3) % 97) as f32 - 48.0) / divisor)
                .collect();
            x.set_bytes(data.iter().flat_map(|v| v.to_le_bytes()).collect())?;
            for output in &outputs {
                output.clear_device_buf();
            }
            target.launcher().set_tuning(iteration > 0 && repeats == 1);
            target.launcher().take_last_profile();
            let start = Instant::now();
            device.session().resolve(&outputs)?;
            device.session().wait()?;
            let wall_us = start.elapsed().as_secs_f64() * 1e6;
            let spans = target.launcher().take_last_profile();
            if iteration == 0 {
                cold_ms = wall_us / 1000.0;
            }
            if iteration >= 3 {
                let (gpu_us, count) = if repeats > 1 {
                    replay_sample
                        .lock()
                        .unwrap()
                        .take()
                        .ok_or("missing replay timestamp")?
                } else {
                    let spans = spans.ok_or("missing GPU timestamps")?;
                    (spans.iter().sum(), spans.len())
                };
                assert!(
                    gpu_us > 0.0 && gpu_us * repeats as f64 <= wall_us * 1.05,
                    "invalid GPU timing"
                );
                launches = count;
                sample_us = Some(gpu_us / copies as f64);
                gpu_samples.push(gpu_us / copies as f64);
                wall_samples.push(wall_us);
            }
            if iteration < 3 {
                let output = y.to_vec_f32()?;
                max_error = max_error.max(validate(&wbytes, &data, &output, m, n, k, fmt));
                for (copy, other) in outputs.iter().enumerate().skip(1) {
                    let actual = other.to_vec_f32()?;
                    max_error = max_error.max(validate(&wbytes, &data, &actual, m, n, k, fmt));
                    // Independent roots can select different legal reduction
                    // layouts. Check every scalar without requiring identical
                    // reassociation; CPU checks above also cover every copy.
                    for (index, (&expected, &value)) in output.iter().zip(&actual).enumerate() {
                        let tolerance = if f16 {
                            0.05 + 0.002 * expected.abs().max(value.abs())
                        } else {
                            0.0002 + 0.000002 * expected.abs().max(value.abs())
                        };
                        assert!(
                            (value - expected).abs() <= tolerance,
                            "weight copy {copy} differs at {index}: {value} != {expected}, tolerance={tolerance}"
                        );
                    }
                }
                if let Some(dir) = std::env::var_os("LAB_OUTPUT_DIR") {
                    let dir = std::path::PathBuf::from(dir);
                    std::fs::create_dir_all(&dir)?;
                    std::fs::write(
                        dir.join(format!("{fmt:?}-{m}-{n}-{k}-{iteration}.f32")),
                        output
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                }
            }
            if step_control {
                println!(
                    "LAB_CONTROL {}",
                    serde_json::json!({"event": "done", "iteration": iteration, "gpu_us": sample_us})
                );
                std::io::stdout().flush()?;
            }
        }
        target.launcher().set_tuning(false);
        gpu_samples.sort_by(f64::total_cmp);
        wall_samples.sort_by(f64::total_cmp);
        let us = gpu_samples[gpu_samples.len() / 2];
        println!(
            "MATMUL m={m} n={n} k={k} fmt={fmt:?} launches={launches} copies={copies} repeats={repeats} gpu_us={us:.3} p25_us={:.3} p75_us={:.3} wall_us={:.3} cold_ms={cold_ms:.3} logical_weight_bytes={logical_weight_bytes} logical_weight_GBs={:.3} GFLOPs={:.3} max_error={max_error:.6}",
            gpu_samples[gpu_samples.len() / 4],
            gpu_samples[gpu_samples.len() * 3 / 4],
            wall_samples[wall_samples.len() / 2],
            logical_weight_bytes as f64 / us / 1000.0,
            2.0 * m as f64 * n as f64 * k as f64 / us / 1000.0
        );
    }
    target.set_plan_observer(None);
    Ok(())
}
