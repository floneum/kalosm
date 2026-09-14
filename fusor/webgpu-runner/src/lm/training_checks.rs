//! Browser-only checks against the actual demo model, without rendering or sampling.
#![cfg(feature = "training-checks")]
use super::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = checkTraining)]
pub async fn check_training(mode: &str, steps: u32) -> std::result::Result<String, JsValue> {
    async fn run(mode: &str, steps: u32) -> Result<String> {
        if !matches!(mode, "portable" | "subgroups" | "accelerated") || !(1..=512).contains(&steps)
        {
            return Err(fusor::Error::Plan(
                "expected portable/subgroups/accelerated and 1..=512 steps".into(),
            ));
        }
        let corpus = Corpus::load();
        let mut model = Lm::new(corpus.vocab_size(), 0x51ed_c0de).await?;
        model
            .compile_training(fusor::program::ProgramOptions {
                matrix_acceleration: mode == "accelerated",
                subgroup_acceleration: mode != "portable",
                ..Default::default()
            })
            .await?;
        let first = model.train(&corpus, 1).await?.loss;
        model.train(&corpus, 16).await?;
        let mut times = vec![];
        let mut losses = vec![];
        for _ in 0..5 {
            let start = web_time::Instant::now();
            let result = model.train(&corpus, steps as usize).await?;
            times.push(start.elapsed().as_secs_f64() * 1000.0 / steps as f64);
            if !result.loss.is_finite() {
                return Err(fusor::Error::Plan(
                    "training produced a nonfinite loss".into(),
                ));
            }
            losses.push(result.loss);
        }
        let scored = model.evaluate(&corpus, 1).await?;
        let program = model.compiled.as_ref().unwrap();
        let acceleration = program.acceleration();
        Ok(format!(
            "{{\"mode\":\"{mode}\",\"first_loss\":{first},\"ms_per_step\":{times:?},\"losses\":{losses:?},\"held_out_loss\":{},\"accuracy\":{},\"subgroups\":{},\"matrices\":\"{:?}\",\"fallback\":{},\"dispatches\":{}}}",
            scored.loss,
            scored.accuracy,
            acceleration.subgroups,
            acceleration.matrices,
            program.acceleration_fallback().is_some(),
            program.stats().kernels,
        ))
    }
    run(mode, steps)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen(js_name = checkProgram)]
pub async fn check_program(mode: &str) -> std::result::Result<(), JsValue> {
    async fn run(mode: &str) -> Result<()> {
        use fusor::{
            Device, Tensor,
            program::{ProgramOptions, TrainingProgram},
        };
        let device = Device::gpu().await?;
        let options = ProgramOptions {
            matrix_acceleration: mode == "accelerated",
            subgroup_acceleration: mode != "portable",
            ..Default::default()
        };
        // Check the static matrix candidate before shape-family promotion
        // deliberately moves repeated changing shapes to a generic plan.
        for (m, k, n) in [(256, 256, 256), (37, 17, 35), (17, 257, 19), (544, 33, 256)] {
            let a: Vec<f32> = (0..m * k)
                .map(|i| ((i * 13 % 29) as f32 - 14.) / 16.)
                .collect();
            let b: Vec<f32> = (0..k * n)
                .map(|i| ((i * 7 % 31) as f32 - 15.) / 16.)
                .collect();
            let x = Tensor::<2, f32>::from_slice(&device, [m, k], &a);
            let y = Tensor::<2, f32>::from_slice(&device, [k, n], &b);
            device.session().upload_leaf(x.as_dyn())?;
            device.session().upload_leaf(y.as_dyn())?;
            let product = x.matmul(&y);
            let sums = product.sum::<1>(1);
            let mut program = TrainingProgram::compile_with_options(
                &[product.as_dyn().clone(), sums.as_dyn().clone()],
                &[],
                options,
            )
            .await?;
            program.run_async().await?;
            let bytes = program.read(product.as_dyn()).await?;
            let actual: Vec<_> = bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            let bytes = program.read(sums.as_dyn()).await?;
            let rows: Vec<_> = bytes
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect();
            let ordinary = product.to_vec_f32_async().await?;
            for r in 0..m {
                let mut row = 0.;
                for c in 0..n {
                    let expected: f32 = (0..k).map(|t| a[r * k + t] * b[t * n + c]).sum();
                    if !actual[r * n + c].is_finite() || (actual[r * n + c] - expected).abs() > 1e-5
                    {
                        return Err(fusor::Error::Plan(format!(
                            "matrix {m}x{k}x{n} at ({r},{c}): {} vs {expected}",
                            actual[r * n + c]
                        )));
                    }
                    if !ordinary[r * n + c].is_finite()
                        || (ordinary[r * n + c] - expected).abs() > 1e-5
                    {
                        return Err(fusor::Error::Plan(format!(
                            "Session matrix {m}x{k}x{n} at ({r},{c}): {} vs {expected}",
                            ordinary[r * n + c]
                        )));
                    }
                    row += expected;
                }
                if !rows[r].is_finite() || (rows[r] - row).abs() > 1e-5 {
                    return Err(fusor::Error::Plan(format!(
                        "row reduction {r}: {} vs {row}",
                        rows[r]
                    )));
                }
            }
        }
        // Exercise the ordinary Naga emitter too: requesting subgroups must
        // also produce the browser's required enable declaration on this path.
        let values: Vec<f32> = (0..19 * 97).map(|i| (i % 23) as f32 / 16.).collect();
        let tensor = Tensor::<2, f32>::from_slice(&device, [19, 97], &values);
        let sums = tensor.sum::<1>(1).to_vec_f32_async().await?;
        for (r, got) in sums.iter().enumerate() {
            let expected: f32 = values[r * 97..(r + 1) * 97].iter().sum();
            if (got - expected).abs() > 1e-5 {
                return Err(fusor::Error::Plan(format!(
                    "Session subgroup sum {r}: {got} vs {expected}"
                )));
            }
        }
        Ok(())
    }
    run(mode)
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen(js_name = checkBenchmarks)]
pub async fn check_benchmarks(filter: &str) -> std::result::Result<String, JsValue> {
    let device = fusor::Device::gpu()
        .await
        .map_err(|e| JsValue::from_str(&e.to_string()))?;
    let mut output = String::new();
    let cases: Vec<_> = fusor_conformance::bench::registry::cases()
        .into_iter()
        .filter(|c| c.name().starts_with("webgpu::") && c.name().contains(filter))
        .collect();
    fusor_conformance::bench::registry::run_cases(
        &device,
        fusor_conformance::bench::BenchmarkConfig::default(),
        cases,
        |event| {
            if let fusor_conformance::bench::BenchmarkEvent::Finished(report) = event {
                use std::fmt::Write;
                writeln!(
                    output,
                    "{} {} {}",
                    report.name, report.median_ms, report.iterations
                )
                .unwrap();
            }
        },
    )
    .await
    .map_err(|e| JsValue::from_str(&e.to_string()))?;
    Ok(output)
}

#[wasm_bindgen(js_name = checkConformance)]
pub async fn check_conformance(filter: &str) -> std::result::Result<String, JsValue> {
    use fusor_conformance::harness::{Harness, sessions_async};
    let sessions = sessions_async().await;
    let reports = Harness::with_filter(filter)
        .run_async(&sessions, |_| {})
        .await;
    if reports.is_empty() {
        return Err(JsValue::from_str("no matching cases or GPU device"));
    }
    let mut output = String::new();
    for report in reports {
        use std::fmt::Write;
        writeln!(output, "{} {:?}", report.case, report.outcome).unwrap();
        if report.outcome.is_fail() {
            return Err(JsValue::from_str(&output));
        }
    }
    Ok(output)
}
