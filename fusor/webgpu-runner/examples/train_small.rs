//! The browser's model trained natively, without UI dependencies.
//! `cargo run --release --example train_small -- [steps]`
#![allow(dead_code)]
#[path = "../src/lm/config.rs"]
mod config;
#[path = "../src/lm/corpus.rs"]
mod corpus;
#[path = "../src/lm/model.rs"]
mod model;
#[path = "../src/lm/rng.rs"]
mod rng;
#[path = "../src/lm/tokenizer.rs"]
mod tokenizer;

fn main() -> fusor::Result<()> {
    pollster::block_on(async {
        let steps = std::env::args()
            .nth(1)
            .and_then(|x| x.parse().ok())
            .filter(|s| *s > 0)
            .unwrap_or(32);
        let corpus = corpus::Corpus::benchmark()
            .await
            .map_err(fusor::Error::Plan)?;
        let start = std::time::Instant::now();
        let mut model =
            model::Lm::new(corpus.vocab_size(), 0x51ed_c0de, config::ModelConfig::TINY).await?;
        let first = model.train(&corpus, 1).await?;
        println!(
            "parameters={} tokens/step={} setup_ms={:.3} first_loss={}",
            config::ModelConfig::TINY.parameters(corpus.vocab_size()),
            config::ModelConfig::TINY.tokens(),
            start.elapsed().as_secs_f64() * 1000.,
            first.loss
        );
        // Warm every replay path before timing.
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
