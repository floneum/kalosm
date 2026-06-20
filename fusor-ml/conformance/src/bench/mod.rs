//! Browser-runnable WebGPU benchmarks.
//!
//! The conformance suite collects assertion builders and lets each runner decide
//! how to execute them. Benchmarks follow the same shape, but each case returns
//! timing data instead of a pass/fail tensor comparison.

pub mod registry;
pub mod sweep;
pub mod webgpu;

#[cfg(feature = "burn-bench")]
pub mod burn;

use std::{future::Future, pin::Pin};

use fusor::Device;

#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

pub type BenchmarkError = Box<dyn std::error::Error>;
pub type BenchmarkResult<T> = Result<T, BenchmarkError>;

type CaseFuture<'a> = Pin<Box<dyn Future<Output = BenchmarkResult<BenchmarkReport>> + 'a>>;
type BenchmarkRunner = dyn for<'a> FnOnce(&'a Device, BenchmarkConfig) -> CaseFuture<'a>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkConfig {
    pub warmups: usize,
    pub iterations: usize,
    pub samples: usize,
}

impl BenchmarkConfig {
    pub const fn new(warmups: usize, iterations: usize, samples: usize) -> Self {
        Self {
            warmups,
            iterations,
            samples,
        }
    }

    pub const fn smoke() -> Self {
        Self::new(0, 1, 1)
    }

    fn sanitized(self) -> Self {
        Self {
            warmups: self.warmups,
            iterations: self.iterations.max(1),
            samples: self.samples.max(1),
        }
    }
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self::new(2, 3, 7)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkReport {
    pub name: String,
    pub warmups: usize,
    pub iterations: usize,
    pub samples: usize,
    pub total_iterations: usize,
    pub sample_mean_ms: Vec<f64>,
    pub total_ms: f64,
    pub mean_ms: f64,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub stddev_ms: f64,
    pub detail: String,
}

impl BenchmarkReport {
    pub(crate) fn new(
        name: impl Into<String>,
        config: BenchmarkConfig,
        samples: Vec<Duration>,
        detail: impl Into<String>,
    ) -> Self {
        let config = config.sanitized();
        let mut sample_mean_ms = samples
            .iter()
            .map(|elapsed| elapsed.as_secs_f64() * 1000.0 / config.iterations as f64)
            .collect::<Vec<_>>();
        let total_ms = samples
            .iter()
            .map(|elapsed| elapsed.as_secs_f64() * 1000.0)
            .sum::<f64>();
        let mean_ms = sample_mean_ms.iter().copied().sum::<f64>() / sample_mean_ms.len() as f64;
        sample_mean_ms.sort_by(f64::total_cmp);
        let median_ms = median(&sample_mean_ms);
        let min_ms = sample_mean_ms.first().copied().unwrap_or(0.0);
        let max_ms = sample_mean_ms.last().copied().unwrap_or(0.0);
        let stddev_ms = stddev(&sample_mean_ms, mean_ms);
        let total_iterations = config.iterations * config.samples;
        Self {
            name: name.into(),
            warmups: config.warmups,
            iterations: config.iterations,
            samples: config.samples,
            total_iterations,
            sample_mean_ms,
            total_ms,
            mean_ms,
            median_ms,
            min_ms,
            max_ms,
            stddev_ms,
            detail: detail.into(),
        }
    }
}

fn median(sorted: &[f64]) -> f64 {
    match sorted.len() {
        0 => 0.0,
        len if len % 2 == 1 => sorted[len / 2],
        len => (sorted[len / 2 - 1] + sorted[len / 2]) * 0.5,
    }
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    variance.sqrt()
}

#[derive(Clone, Debug, PartialEq)]
pub enum BenchmarkEvent {
    Started(String),
    Finished(BenchmarkReport),
}

pub struct BenchmarkCase {
    name: String,
    run: Box<BenchmarkRunner>,
}

impl BenchmarkCase {
    pub fn new(
        name: impl Into<String>,
        run: impl for<'a> FnOnce(&'a Device, BenchmarkConfig) -> CaseFuture<'a> + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            run: Box::new(run),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn run(
        self,
        device: &Device,
        config: BenchmarkConfig,
    ) -> BenchmarkResult<BenchmarkReport> {
        (self.run)(device, config.sanitized()).await
    }
}

pub(crate) async fn time_samples<F, Fut>(
    config: BenchmarkConfig,
    mut run_once: F,
) -> BenchmarkResult<Vec<Duration>>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = BenchmarkResult<()>>,
{
    let config = config.sanitized();
    for _ in 0..config.warmups {
        run_once().await?;
    }

    let mut samples = Vec::with_capacity(config.samples);
    for _ in 0..config.samples {
        let started = Instant::now();
        for _ in 0..config.iterations {
            run_once().await?;
        }
        samples.push(started.elapsed());
    }
    Ok(samples)
}
