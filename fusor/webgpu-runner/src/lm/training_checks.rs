//! Browser-only checks against the actual demo model, without rendering or sampling.
#![cfg(feature = "training-checks")]
use super::*;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = checkTraining)]
pub async fn check_training(steps: u32) -> std::result::Result<String, JsValue> {
    async fn run(steps: u32) -> Result<String> {
        if !(1..=512).contains(&steps) {
            return Err(fusor::Error::Plan("expected 1..=512 steps".into()));
        }
        let corpus = Corpus::benchmark().await.map_err(fusor::Error::Plan)?;
        let mut model = Lm::new(corpus.vocab_size(), 0x51ed_c0de, ModelConfig::TINY).await?;
        let first = model.train(&corpus, 1).await?.loss;
        model.train(&corpus, 16).await?;
        let mut times = vec![];
        let mut losses = vec![];
        let before = model.dispatch_count();
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
        let dispatches = (model.dispatch_count() - before) / (5 * u64::from(steps));
        let scored = model.evaluate(&corpus, 1).await?;
        Ok(format!(
            "{{\"first_loss\":{first},\"ms_per_step\":{times:?},\"losses\":{losses:?},\"held_out_loss\":{},\"accuracy\":{},\"dispatches\":{dispatches}}}",
            scored.loss, scored.accuracy,
        ))
    }
    run(steps)
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
