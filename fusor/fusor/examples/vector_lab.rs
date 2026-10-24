//! Isolated RMS normalization: full CPU reference, then one pre-encoded GPU
//! replay span. Usage: vector_lab [WIDTH=4096] [ROWS=1] [REPEATS=1024].
use fusor::{Dim, Dtype, device::Device};
use std::sync::{Arc, Mutex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let width: usize = args.next().unwrap_or_else(|| "4096".into()).parse()?;
    let rows: usize = args.next().unwrap_or_else(|| "1".into()).parse()?;
    let repeats: usize = args.next().unwrap_or_else(|| "1024".into()).parse()?;
    assert!(
        args.next().is_none()
            && (1..=8192).contains(&width)
            && (1..=4).contains(&rows)
            && (2..=4096).contains(&repeats)
    );
    assert_eq!(
        std::env::var("FUSOR_AUTOTUNE_MIN_MACS").ok().as_deref(),
        Some("18446744073709551615")
    );
    let device = Device::gpu_blocking()?;
    let graph = device.graph().clone();
    let x = graph.leaf(
        "vector-lab-input",
        &[Dim::Const(rows as u64), Dim::Const(width as u64)],
        Dtype::F32,
    )?;
    let weights: Vec<f32> = (0..width)
        .map(|i| 0.25 + ((i * 7) % 83) as f32 / 31.7)
        .collect();
    let w = graph.leaf("vector-lab-weight", &[Dim::Const(width as u64)], Dtype::F32)?;
    w.set_bytes(weights.iter().flat_map(|v| v.to_le_bytes()).collect())?;
    let y = x.rms_norm(&w, 1e-5)?;
    let sample = Arc::new(Mutex::new(None));
    let sink = sample.clone();
    let probe = device.clone();
    let target = device.backend().gpu_target().unwrap();
    target.set_kernel_observer(Some(Arc::new(|ir, _, _| {
        eprintln!(
            "VECTOR_KERNEL name={} block={} grid={:?}",
            ir.name, ir.block, ir.grid
        );
    })));
    target.set_plan_observer(Some(Arc::new(move |_, _, _, _, commands| {
        let target = probe.backend().gpu_target().unwrap();
        let launcher = target.launcher();
        let count = commands.iter().filter(|c| !c.is_empty_dispatch()).count();
        let per_submit =
            fusor_gpu::launch::dispatches_per_submit(count * repeats, launcher.backend());
        assert!(
            count > 0 && per_submit >= count * repeats,
            "use FUSOR_PROFILE_SUBMIT_SIZE=2147483647 to avoid host gaps"
        );
        launcher.set_tuning(true);
        let query = launcher.timestamp_query_set(1).unwrap();
        let repeated: Vec<_> = (0..repeats)
            .flat_map(|_| commands.iter().cloned())
            .collect();
        launcher
            .encode_command_records(
                &repeated,
                Some(&query),
                fusor_gpu::launch::TimingMode::Whole,
            )
            .unwrap();
        launcher.poll_wait().unwrap();
        let us = launcher.read_timestamps(target.pool(), &query, 1).unwrap()[0] / repeats as f64;
        *sink.lock().unwrap() = Some(us);
        launcher.set_tuning(false);
    })));
    let mut samples = Vec::new();
    let mut max_error = 0.0f64;
    for iteration in 0..18 {
        let input: Vec<f32> = (0..rows * width)
            .map(|i| (((i * 13 + i / width * 5 + iteration * 3) % 97) as f32 - 48.0) / 23.71)
            .collect();
        x.set_bytes(input.iter().flat_map(|v| v.to_le_bytes()).collect())?;
        y.clear_device_buf();
        device.session().resolve(std::slice::from_ref(&y))?;
        let us = sample.lock().unwrap().take().unwrap();
        assert!(us.is_finite() && us > 0.0);
        if iteration >= 3 {
            samples.push(us);
        } else {
            let actual = y.to_vec_f32()?;
            assert_eq!(actual.len(), input.len());
            for row in 0..rows {
                let values = &input[row * width..(row + 1) * width];
                let mean = values.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / width as f64;
                let inv = 1.0 / (mean + f64::from(1e-5f32)).sqrt();
                for col in 0..width {
                    let expected = f64::from(values[col]) * inv * f64::from(weights[col]);
                    let got = f64::from(actual[row * width + col]);
                    let error = (got - expected).abs();
                    assert!(
                        got.is_finite() && error < 2e-5 + expected.abs() * 2e-5,
                        "RMS mismatch [{row},{col}]: {got} != {expected}"
                    );
                    max_error = max_error.max(error);
                }
            }
        }
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "VECTOR operation=rms width={width} rows={rows} repeats={repeats} gpu_us={:.3} p25_us={:.3} p75_us={:.3} max_error={max_error:.8}",
        samples[7], samples[3], samples[11]
    );
    target.set_plan_observer(None);
    target.set_kernel_observer(None);
    Ok(())
}
