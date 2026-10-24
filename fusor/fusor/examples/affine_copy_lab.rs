//! Small diagnostic of concatenating computed, reshaped storage.
use fusor::{Device, Dim, Tensor};
use fusor_ir::ir::{Op, launch::Launch};
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::gpu_blocking()?;
    let target = device.backend().gpu_target().unwrap();
    let sample = Arc::new(Mutex::new(None));
    let sink = sample.clone();
    let probe = device.clone();
    target.set_plan_observer(Some(Arc::new(move |plan, graph, _, _, commands| {
        for launch in &plan.launches {
            let Op::Launch(Launch::Scatter { ops, .. }) = &graph.node(launch.root).op else { continue };
            let cx = fusor_ir::target::LowerCtx { plan, launch, graph, symbols: &plan.symbols, dim_bindings: &[] };
            eprintln!("AFFINE_SCATTER {:?}", launch.bindings);
            for operand in ops {
                let selected = cx.selected(operand.src);
                eprintln!("AFFINE_OPERAND {operand:?} selected={selected:?} facts={:?} storage={:?} op={:?}", graph.facts(selected).shape, plan.buffers.iter().find(|b| b.value == selected), graph.node(selected).op);
            }
        }
        for command in commands {
            if let fusor_gpu::launch::CommandRecord::Dispatch { name, .. } = command {
                eprintln!("AFFINE_DISPATCH {name}");
                if std::env::var_os("LAB_EXPECT_DIRECT_RANGE").is_some() {
                    assert_ne!(*name, "scatter_rows", "consecutive updates still scan indices");
                }
            } else { eprintln!("AFFINE_COPY"); }
        }
        let target = probe.backend().gpu_target().unwrap();
        let launcher = target.launcher();
        let repeats = 128;
        assert!(fusor_gpu::launch::dispatches_per_submit(commands.len()*repeats, launcher.backend()) >= commands.len()*repeats,
            "set FUSOR_PROFILE_SUBMIT_SIZE=2147483647 to avoid host gaps");
        launcher.set_tuning(true);
        let Some(query) = launcher.timestamp_query_set(1) else {
            // Hosted GPUs can validate the graph without exposing timestamps.
            launcher.set_tuning(false);
            return;
        };
        let repeated: Vec<_> = (0..repeats).flat_map(|_| commands.iter().cloned()).collect();
        launcher.encode_command_records(&repeated, Some(&query), fusor_gpu::launch::TimingMode::Whole).unwrap();
        launcher.poll_wait().unwrap();
        *sink.lock().unwrap() = Some(launcher.read_timestamps(target.pool(), &query, 1).unwrap()[0]/repeats as f64);
        launcher.set_tuning(false);
    })));
    for batch in [1usize, 2] {
        let input: Vec<f32> = (0..batch * 40 * 128).map(|i| (i % 251) as f32).collect();
        let a = Tensor::from_slice(&device, [batch, 32 * 128], &input[..batch * 32 * 128]);
        let b = Tensor::from_slice(&device, [batch, 8 * 128], &input[batch * 32 * 128..]);
        let reshape = |heads| {
            [
                Dim::Const(batch as u64).into(),
                Dim::ONE.into(),
                Dim::Const(heads).into(),
                Dim::Const(128).into(),
            ]
        };
        let a = a
            .as_dyn()
            .mul_scalar(2.)?
            .reshape(&reshape(32))?
            .permute(&[0, 2, 1, 3])?;
        let b = b
            .as_dyn()
            .mul_scalar(3.)?
            .reshape(&reshape(8))?
            .permute(&[0, 2, 1, 3])?;
        let joined = fusor::tensor::Dyn::cat(&[a, b], 1)?.mul_scalar(0.5)?;
        let actual = joined.to_vec_f32()?;
        let mut expected = Vec::new();
        for row in 0..batch {
            expected.extend(
                input[row * 32 * 128..(row + 1) * 32 * 128]
                    .iter()
                    .map(|v| v * 2.),
            );
            expected.extend(
                input[batch * 32 * 128 + row * 8 * 128..batch * 32 * 128 + (row + 1) * 8 * 128]
                    .iter()
                    .map(|v| v * 3.),
            );
        }
        for value in &mut expected {
            *value *= 0.5;
        }
        assert_eq!(actual, expected);
        println!(
            "AFFINE_VERIFIED batch={batch} values={} gpu_us={}",
            actual.len(),
            sample
                .lock()
                .unwrap()
                .take()
                .map_or_else(|| "unavailable".into(), |us| format!("{us:.3}"))
        );
    }
    Ok(())
}
