//! A 1,280-parameter MLP whose complete training step fits one workgroup.
//! `cargo run --release -p fusor --example train_mlp -- [baseline|fused]`
use fusor::{Device, Tensor, program::TrainingProgram, tensor::Dyn};

fn graph(device: &Device) -> fusor::Result<(Vec<Dyn>, Vec<Dyn>, Vec<Dyn>)> {
    let a = Tensor::<2, f32>::from_slice(
        device,
        [32, 32],
        &(0..1024)
            .map(|i| (i % 17) as f32 * 0.005 - 0.04)
            .collect::<Vec<_>>(),
    );
    let b = Tensor::<2, f32>::from_slice(
        device,
        [32, 8],
        &(0..256)
            .map(|i| (i % 13) as f32 * 0.005 - 0.03)
            .collect::<Vec<_>>(),
    );
    let x = Tensor::<2, f32>::from_slice(
        device,
        [4, 32],
        &(0..128)
            .map(|i| (i % 19) as f32 * 0.1 - 0.9)
            .collect::<Vec<_>>(),
    );
    let h = x.matmul(&a);
    let z = h.tanh();
    let y = z.matmul(&b);
    let squared = y.sqr();
    let rows = squared.sum::<1>(1);
    let loss = rows.sum::<0>(0);
    let params = vec![a.as_dyn().clone(), b.as_dyn().clone()];
    let grads = device.graph().backward_with(loss.as_dyn(), &params)?;
    let mut roots = vec![loss.as_dyn().clone()];
    for p in &params {
        roots.push(p.sub(&grads.get(p).unwrap().mul_scalar(0.01f32)?)?);
    }
    let stale = vec![
        h.as_dyn().clone(),
        z.as_dyn().clone(),
        y.as_dyn().clone(),
        squared.as_dyn().clone(),
        rows.as_dyn().clone(),
    ];
    Ok((params, roots, stale))
}

fn main() -> fusor::Result<()> {
    pollster::block_on(async {
        let fused = match std::env::args().nth(1).as_deref().unwrap_or("fused") {
            "fused" => true,
            "baseline" => false,
            _ => {
                return Err(fusor::Error::Plan(
                    "usage: train_mlp [baseline|fused]".into(),
                ));
            }
        };
        let device = Device::gpu().await?;
        let (params, roots, stale) = graph(&device)?;
        let feedback: Vec<_> = params
            .iter()
            .cloned()
            .zip(roots[1..].iter().cloned())
            .collect();
        let mut program = if fused {
            Some(TrainingProgram::compile(&roots, &feedback).await?)
        } else {
            None
        };
        if let Some(p) = &program {
            assert_eq!(p.stats().kernels, 1);
        }
        for round in 0..4 {
            let before = program
                .as_ref()
                .map_or_else(|| device.session().launch_count(), |p| p.dispatch_count());
            let start = std::time::Instant::now();
            for _ in 0..128 {
                if let Some(p) = &mut program {
                    p.run_async().await?;
                } else {
                    for value in stale.iter().chain(&roots) {
                        value.clear_device_buf();
                    }
                    device.session().resolve(&roots)?;
                    for (p, next) in &feedback {
                        p.adopt_buffer(next)?;
                    }
                }
            }
            let bytes = if let Some(p) = &program {
                p.read(&roots[0]).await?
            } else {
                roots[0].to_bytes_async().await?
            };
            let loss = f32::from_le_bytes(bytes.try_into().unwrap());
            let count = program
                .as_ref()
                .map_or_else(|| device.session().launch_count(), |p| p.dispatch_count())
                - before;
            println!(
                "round={round} us/step={:.3} dispatches/step={} loss={loss}",
                start.elapsed().as_secs_f64() * 1e6 / 128.,
                count / 128
            );
            assert!(loss.is_finite());
        }
        Ok(())
    })
}
