//! The browser's actual model, without UI dependencies.
//! `cargo run --release -p fusor --example train_small -- [baseline|fused|single|subgroups|portable|compare] [steps]`
#![allow(dead_code)]
#[path = "../../webgpu-runner/src/lm/config.rs"]
mod config;
#[path = "../../webgpu-runner/src/lm/corpus.rs"]
mod corpus;
#[path = "../../webgpu-runner/src/lm/model.rs"]
mod model;
#[path = "../../webgpu-runner/src/lm/rng.rs"]
mod rng;

fn main() -> fusor::Result<()> {
    pollster::block_on(async {
        let args: Vec<_> = std::env::args().collect();
        let mode = args.get(1).map(String::as_str).unwrap_or("baseline");
        let steps = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(32);
        if !matches!(
            mode,
            "baseline" | "fused" | "single" | "subgroups" | "portable" | "compare"
        ) || steps == 0
        {
            return Err(fusor::Error::Plan(
                "usage: train_small [baseline|fused|single|subgroups|portable|compare] [positive steps per round]"
                    .into(),
            ));
        }
        let corpus = corpus::Corpus::benchmark()
            .await
            .map_err(fusor::Error::Plan)?;
        if mode == "compare" {
            return compare(&corpus, steps).await;
        }
        let start = std::time::Instant::now();
        let mut model =
            model::Lm::new(corpus.vocab_size(), 0x51ed_c0de, config::ModelConfig::TINY).await?;
        if mode != "baseline" {
            model
                .compile_training(fusor::program::ProgramOptions {
                    workgroups: if mode == "single" { Some(1) } else { None },
                    matrix_acceleration: !matches!(mode, "portable" | "subgroups"),
                    subgroup_acceleration: mode != "portable",
                    max_region_stages: 1024,
                })
                .await?;
        }
        let first = model.train(&corpus, 1).await?;
        println!(
            "mode={mode} parameters={} tokens/step={} setup_ms={:.3} first_loss={} plan={:?}",
            config::ModelConfig::TINY.parameters(corpus.vocab_size()),
            config::ModelConfig::TINY.tokens(),
            start.elapsed().as_secs_f64() * 1000.,
            first.loss,
            model.program_stats()
        );
        // Warm every replay path before timing. All modes consume identical data.
        model.train(&corpus, 16).await?;
        for round in 0..3 {
            let before = model.dispatch_count();
            let start = std::time::Instant::now();
            let result = model.train(&corpus, steps).await?;
            println!(
                "round={round} ms/step={:.3} dispatches/step={} loss={}",
                start.elapsed().as_secs_f64() * 1000. / steps as f64,
                (model.dispatch_count() - before) / steps as u64,
                result.loss
            );
            assert!(result.loss.is_finite());
        }
        let evaluation = model.evaluate(&corpus, 1).await?;
        println!(
            "held_out_loss={} accuracy={}",
            evaluation.loss, evaluation.accuracy
        );
        Ok(())
    })
}

// An opt-in performance regression, run on an otherwise idle GPU. Alternating
// order limits clock/temperature bias; the fastest warmed windows exclude the
// reference executor's occasional retuning pauses from the claimed speedup.
async fn compare(corpus: &corpus::Corpus, steps: usize) -> fusor::Result<()> {
    let mut models = [
        model::Lm::new(corpus.vocab_size(), 0x51ed_c0de, config::ModelConfig::TINY).await?,
        model::Lm::new(corpus.vocab_size(), 0x51ed_c0de, config::ModelConfig::TINY).await?,
    ];
    models[1].compile_training(Default::default()).await?;
    for model in &mut models {
        model.train(corpus, 17).await?;
    }
    println!(
        "parameters={} tokens/step={} plan={:?}",
        config::ModelConfig::TINY.parameters(corpus.vocab_size()),
        config::ModelConfig::TINY.tokens(),
        models[1].program_stats()
    );
    let mut best = [f64::INFINITY; 2];
    for round in 0..5 {
        let mut losses = [0.; 2];
        for offset in 0..2 {
            let i = (round + offset) % 2;
            let model = &mut models[i];
            let before = model.dispatch_count();
            let start = std::time::Instant::now();
            let result = model.train(corpus, steps).await?;
            let ms = start.elapsed().as_secs_f64() * 1000. / steps as f64;
            best[i] = best[i].min(ms);
            losses[i] = result.loss;
            println!(
                "round={round} mode={} ms/step={ms:.3} dispatches/step={} loss={}",
                ["baseline", "fused"][i],
                (model.dispatch_count() - before) / steps as u64,
                result.loss
            );
        }
        assert!(
            (losses[0] - losses[1]).abs() < 2e-4,
            "training losses: {losses:?}"
        );
    }
    let a = models[0].evaluate(corpus, 1).await?;
    let b = models[1].evaluate(corpus, 1).await?;
    assert!(
        (a.loss - b.loss).abs() < 2e-4,
        "held-out losses: {} vs {}",
        a.loss,
        b.loss
    );
    println!(
        "held_out baseline_loss={} fused_loss={} baseline_accuracy={} fused_accuracy={}",
        a.loss, b.loss, a.accuracy, b.accuracy
    );
    println!(
        "best baseline_ms={:.3} fused_ms={:.3} speedup={:.3}x",
        best[0],
        best[1],
        best[0] / best[1]
    );
    assert!(
        best[1] < best[0],
        "transformer throughput regression: {best:?}"
    );
    Ok(())
}
