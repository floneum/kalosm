//! Pure decode-attention replay with full CPU checks. Cache construction and
//! mutation finish before the observer is installed; only attention is replayed.
use fusor::{Dim, Dtype, Tensor, cache::KvCache, device::Device};
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let keys: usize = args
        .first()
        .ok_or("usage: attention_lab KEYS [PREFIX]")?
        .parse()?;
    let prefix: usize = args.get(1).map_or(Ok(keys), |s| s.parse())?;
    assert!((1..=32768).contains(&keys) && (1..=keys).contains(&prefix));
    let repeats: usize = std::env::var("LAB_ATTENTION_REPEATS")
        .ok()
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(64);
    assert!((16..=2048).contains(&repeats));
    assert_eq!(
        std::env::var("FUSOR_AUTOTUNE_MIN_MACS").as_deref(),
        Ok("18446744073709551615")
    );
    let dev = Device::gpu_blocking()?;
    let graph = dev.graph();
    let q = graph.leaf(
        "attention-q",
        &[
            Dim::Const(1),
            Dim::Const(32),
            Dim::Const(1),
            Dim::Const(128),
        ],
        Dtype::F32,
    )?;
    let shape = [
        Dim::Const(1),
        Dim::Const(8),
        Dim::Const(keys as u64),
        Dim::Const(128),
    ];
    let k = graph.leaf("attention-k", &shape, Dtype::F32)?;
    let v = graph.leaf("attention-v", &shape, Dtype::F32)?;
    let sample =
        |i: usize, seed: usize| (((i * 37 + i / 128 * 7 + seed * 11) % 251) as f32 - 125.) / 63.37;
    let score_stress = std::env::var_os("LAB_ATTENTION_SCORE_STRESS").is_some();
    let kd: Vec<f32> = (0..8 * keys * 128)
        .map(|i| {
            if i / 128 % keys < prefix {
                if score_stress {
                    let direction = if (i / (keys * 128)).is_multiple_of(2) {
                        1.
                    } else {
                        -1.
                    };
                    direction * ((i / 128 % keys) as f32 * 17. - 1000.) / 128f32.sqrt()
                } else {
                    sample(i, 3)
                }
            } else {
                f32::NAN
            }
        })
        .collect();
    let vd: Vec<f32> = (0..kd.len())
        .map(|i| {
            if i / 128 % keys < prefix {
                sample(i, 7)
            } else {
                f32::NAN
            }
        })
        .collect();
    k.set_bytes(bytemuck::cast_slice(&kd).to_vec())?;
    v.set_bytes(bytemuck::cast_slice(&vd).to_vec())?;
    let mut cache = KvCache::with_capacity(2, keys as u64);
    cache.append(&Tensor::<4>::from_dyn(k), &Tensor::<4>::from_dyn(v));
    cache.detach();
    cache.truncate(prefix as u64)?;
    let y = Tensor::<4>::from_dyn(fusor::composite::attention(
        &q,
        cache.k().unwrap().as_dyn(),
        cache.v().unwrap().as_dyn(),
        fusor_ir::ir::launch::MaskKind::None,
        None,
    )?);
    let target = dev.backend().gpu_target().unwrap();
    if std::env::var_os("LAB_PLAN").is_some() {
        target.set_kernel_observer(Some(Arc::new(|ir, _, _| {
            eprintln!(
                "ATTENTION_KERNEL name={} block={} grid={:?}",
                ir.name, ir.block, ir.grid
            );
        })));
    }
    if let Some(directory) = std::env::var_os("LAB_TRACE_DIR") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory)?;
        target.set_kernel_observer(Some(Arc::new(move |ir, identity, uniforms| {
            if !ir.name.starts_with("decode_attention") { return; }
            let trace = fusor_visualize::kernel::capture(ir, fusor_visualize::kernel::CaptureOptions {
                workgroups: (0..ir.grid[0].min(4)).map(|x| [x, 0, 0]).collect(),
                max_actions: 2_000_000,
                known_bindings: uniforms.map(|u| std::collections::BTreeMap::from([(0, u.to_bytes())])).unwrap_or_default(),
                ..Default::default()
            }).expect("attention address capture");
            let path = directory.join(format!("{identity:032x}.trace.json"));
            serde_json::to_writer(std::fs::File::create(&path).unwrap(), &trace).unwrap();
            let mut sim = fusor_visualize::Simulator::new(trace, Default::default()).unwrap();
            while sim.step().is_some() {}
            let summary = serde_json::json!({"kernel":ir.name,"grid":ir.grid,"sampled_workgroups":4.min(ir.grid[0]),"trace":path,"metrics":sim.totals(),"config":sim.config(),"notes":sim.trace().notes});
            std::fs::write(directory.join(format!("{identity:032x}.json")), serde_json::to_vec_pretty(&summary).unwrap()).unwrap();
            eprintln!("ATTENTION_TRACE {summary}");
        })));
    }
    let times = Arc::new(Mutex::new(Vec::<f64>::new()));
    let sink = times.clone();
    let probe = dev.clone();
    target.set_plan_observer(Some(Arc::new(move |_, _, _, _, commands| {
        let target = probe.backend().gpu_target().unwrap();
        let launcher = target.launcher();
        if sink.lock().unwrap().is_empty() {
            for command in commands {
                if let fusor_gpu::launch::CommandRecord::Dispatch { name, grid, .. } = command {
                    eprintln!("ATTENTION_COMMAND name={name} grid={grid:?}");
                }
            }
            if std::env::var_os("FUSOR_PROFILE_FUSED_DECODE_ATTENTION").is_some() {
                assert!(commands.iter().any(|command| matches!(command, fusor_gpu::launch::CommandRecord::Dispatch {name, ..} if name.starts_with("decode_attention"))), "requested fused attention did not execute");
            }
        }
        launcher.set_tuning(true);
        let query = launcher.timestamp_query_set(1).expect("timestamp queries");
        let records: Vec<_> = (0..repeats).flat_map(|_| commands.iter().cloned()).collect();
        launcher
            .encode_command_records(&records, Some(&query), fusor_gpu::launch::TimingMode::Whole)
            .expect("replay submit");
        launcher.poll_wait().expect("completion");
        let us = launcher
            .read_timestamps(target.pool(), &query, 1)
            .expect("timestamps")[0]
            / repeats as f64;
        assert!(us.is_finite() && us > 0.);
        sink.lock().unwrap().push(us);
        launcher.set_tuning(false);
    })));
    let mut max_error = 0f64;
    for seed in 0..14 {
        let qd: Vec<f32> = (0..32 * 128)
            .map(|i| {
                if score_stress {
                    2f32.powi(seed as i32)
                } else {
                    sample(i, seed)
                }
            })
            .collect();
        q.set_bytes(bytemuck::cast_slice(&qd).to_vec())?;
        y.as_dyn().clear_device_buf();
        dev.session().resolve(std::slice::from_ref(y.as_dyn()))?;
        dev.session().wait()?;
        if seed < 3 {
            let actual = y.as_dyn().to_vec_f32()?;
            for h in 0..32 {
                let mut scores = vec![0f64; prefix];
                for (key, score) in scores.iter_mut().enumerate() {
                    *score = (0..128)
                        .map(|d| {
                            f64::from(qd[h * 128 + d])
                                * f64::from(kd[((h / 4) * keys + key) * 128 + d])
                        })
                        .sum::<f64>()
                        / 128f64.sqrt();
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for score in &mut scores {
                    *score = (*score - max).exp();
                }
                let sum: f64 = scores.iter().sum();
                for d in 0..128 {
                    let expected: f64 = scores
                        .iter()
                        .enumerate()
                        .map(|(key, p)| p * f64::from(vd[((h / 4) * keys + key) * 128 + d]))
                        .sum::<f64>()
                        / sum;
                    let error = (f64::from(actual[h * 128 + d]) - expected).abs();
                    assert!(
                        error <= 0.002 + 0.002 * expected.abs(),
                        "seed={seed} head={h} channel={d} error={error}"
                    );
                    max_error = max_error.max(error);
                }
            }
        }
    }
    target.set_plan_observer(None);
    target.set_kernel_observer(None);
    let mut times = times.lock().unwrap()[3..].to_vec();
    times.sort_by(f64::total_cmp);
    println!(
        "ATTENTION keys={keys} prefix={prefix} repeats={repeats} gpu_us={:.3} p25_us={:.3} p75_us={:.3} max_error={max_error:.7}",
        times[times.len() / 2],
        times[times.len() / 4],
        times[times.len() * 3 / 4]
    );
    Ok(())
}
