//! Bounded ordinary-plan replay diagnostic, with no model weights.
use fusor::{Tensor, device::Device};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse())
        .collect::<Result<_, _>>()?;
    assert!(args.len() == 3, "replay_plan_lab BRANCHES WIDTH ROUNDS");
    let [branches, width, rounds] = args[..] else {
        unreachable!()
    };
    assert!((1..=1024).contains(&branches));
    assert!((1..=4096).contains(&width));
    assert!((3..=31).contains(&rounds));
    assert!(branches * width * 4 * 3 <= 64 * 1024 * 1024);
    let device = Device::gpu_blocking()?;
    let data: Vec<f32> = (0..width).map(|i| (i % 251) as f32 / 256.).collect();
    let base = Tensor::from_slice(&device, [width], &data);
    let index = Tensor::from_slice(&device, [1], &[0u32]);
    let mut outputs = Vec::new();
    for branch in 0..branches {
        let update = Tensor::from_slice(&device, [1], &[branch as f32 + 0.5]);
        outputs.push(
            base.as_dyn()
                .scatter_set(0, index.as_dyn(), update.as_dyn(), true)?,
        );
    }
    let target = device.backend().gpu_target().unwrap();
    let launches = Arc::new(AtomicUsize::new(0));
    let observed = launches.clone();
    target.set_plan_observer(Some(Arc::new(move |_, _, _, _, commands| {
        observed.store(commands.len(), Ordering::Relaxed);
    })));
    let mut samples = Vec::new();
    for round in 0..rounds + 2 {
        index
            .as_dyn()
            .set_bytes(((round % width) as u32).to_le_bytes().to_vec())?;
        for output in &outputs {
            output.clear_device_buf();
        }
        let start = Instant::now();
        device.session().resolve(&outputs)?;
        let submitted = start.elapsed();
        device.session().wait()?;
        let completed = start.elapsed();
        assert_eq!(launches.load(Ordering::Relaxed), branches);
        if round >= 2 {
            samples.push(serde_json::json!({
                "submit_ms": submitted.as_secs_f64() * 1000.,
                "total_ms": completed.as_secs_f64() * 1000.,
            }));
        }
    }
    target.set_plan_observer(None);
    let picked = (rounds + 1) % width;
    for (branch, output) in outputs.iter().enumerate() {
        let result = output.to_vec_f32()?;
        for (at, (&got, &original)) in result.iter().zip(&data).enumerate() {
            assert_eq!(
                got,
                if at == picked {
                    branch as f32 + 0.5
                } else {
                    original
                }
            );
        }
    }
    assert_eq!(base.as_dyn().to_vec_f32()?, data);
    println!(
        "REPLAY_PLAN {}",
        serde_json::json!({
            "branches": branches, "width": width, "rounds": rounds,
            "launches": launches.load(Ordering::Relaxed), "samples": samples,
            "verified_values": branches * width,
        })
    );
    Ok(())
}
