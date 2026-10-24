//! Bounded gather/elementwise replay, with every output checked on the host.
use fusor::{Device, Tensor};
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    let [rows, width] = args.as_slice() else {
        return Err("indexed_map_lab ROWS WIDTH".into());
    };
    let (rows, width) = (*rows, *width);
    assert!((1..=128).contains(&rows) && (1..=1024).contains(&width));
    assert_eq!(
        std::env::var("FUSOR_AUTOTUNE_MIN_MACS").as_deref(),
        Ok("18446744073709551615")
    );
    let device = Device::gpu_blocking()?;
    let input = Tensor::from_slice(&device, [rows, width], &vec![0f32; rows * width]);
    let indices = Tensor::from_slice(&device, [width], &vec![0u32; width]);
    let gathered = input.as_dyn().index_select(1, indices.as_dyn())?;
    let output = gathered
        .mul_scalar(0.5)?
        .sub(&input.as_dyn().mul_scalar(0.25)?)?;
    let target = device.backend().gpu_target().unwrap();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let sink = samples.clone();
    let probe = device.clone();
    target.set_plan_observer(Some(Arc::new(move |_, _, _, _, commands| {
        let target = probe.backend().gpu_target().unwrap();
        let launcher = target.launcher();
        let names: Vec<_> = commands
            .iter()
            .filter_map(|command| match command {
                fusor_gpu::launch::CommandRecord::Dispatch { name, .. } => Some(name.to_string()),
                _ => None,
            })
            .collect();
        if sink.lock().unwrap().is_empty() {
            eprintln!("INDEXED_MAP_KERNELS {names:?}");
        }
        launcher.set_tuning(true);
        let query = launcher.timestamp_query_set(1).expect("timestamp support");
        let repeated: Vec<_> = (0..64).flat_map(|_| commands.iter().cloned()).collect();
        launcher
            .encode_command_records(
                &repeated,
                Some(&query),
                fusor_gpu::launch::TimingMode::Whole,
            )
            .unwrap();
        launcher.poll_wait().unwrap();
        let us = launcher.read_timestamps(target.pool(), &query, 1).unwrap()[0] / 64.;
        assert!(us.is_finite() && us > 0.);
        sink.lock().unwrap().push((us, names.len()));
        launcher.set_tuning(false);
    })));
    let mut maximum = 0f32;
    for seed in 0..14 {
        let values: Vec<_> = (0..rows * width)
            .map(|i| ((i * 17 + seed * 13) % 101) as f32 / 37. - 1.1)
            .collect();
        let picks: Vec<_> = (0..width)
            .map(|i| ((i * 7 + seed) % width) as u32)
            .collect();
        input.set_elements(&values);
        indices.set_elements(&picks);
        output.clear_device_buf();
        device.session().resolve(std::slice::from_ref(&output))?;
        device.session().wait()?;
        // Readback resolves no new expression and stays outside replay timing.
        for (i, actual) in output.to_vec_f32()?.into_iter().enumerate() {
            let expected =
                values[i / width * width + picks[i % width] as usize] * 0.5 - values[i] * 0.25;
            let error = (actual - expected).abs();
            assert!(
                actual.is_finite() && error <= 1e-6 * (1. + expected.abs()),
                "seed={seed} element={i}: {actual} != {expected}"
            );
            maximum = maximum.max(error);
        }
    }
    target.set_plan_observer(None);
    let samples = samples.lock().unwrap();
    assert_eq!(samples.len(), 14);
    let count = samples[0].1;
    assert!(samples.iter().all(|(_, n)| *n == count));
    let mut warm: Vec<_> = samples[3..].iter().map(|(us, _)| *us).collect();
    warm.sort_by(f64::total_cmp);
    println!(
        "INDEXED_MAP rows={rows} width={width} launches={count} gpu_us={:.3} p25_us={:.3} p75_us={:.3} verified_values={} max_error={maximum}",
        warm[warm.len() / 2],
        warm[warm.len() / 4],
        warm[warm.len() * 3 / 4],
        rows * width * 14
    );
    Ok(())
}
