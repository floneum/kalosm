#![recursion_limit = "256"]

//! Per-dispatch GPU span probe for the steady decode plan.
//!
//! Runs the same model as `infer_local`, warms the decode plan, then arms the
//! launcher's per-dispatch timestamp path and reports the element-wise MIN
//! span per launch across measured tokens (a slow sample is contention, a
//! fast one is the kernel). Run with `FUSOR_DUMP_PLAN=1 2>plan.txt` and join
//! `SPAN i` lines to the dumped decode-plan `Li:` lines by index.

use kalosm_llama::prelude::*;
use kalosm_llama::Device;
use std::io::Write;
use std::path::PathBuf;

const DEFAULT_MODEL: &str = "/Users/evanalmloff/.cache/huggingface/hub/models--lmstudio-community--Meta-Llama-3.1-8B-Instruct-GGUF/snapshots/8601e6db71269a2b12255ebdf09ab75becf22cc8/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";

fn main() {
    pollster::block_on(async {
        let path = std::env::args()
            .nth(1)
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
        assert!(path.exists(), "model file not found: {}", path.display());

        let device = Device::gpu().await.unwrap();
        let fusor::session::Backend::Gpu(target) = device.backend().clone() else {
            unreachable!("gpu device is gpu");
        };

        let model = Llama::builder()
            .with_source(LlamaSource::new(FileSource::Local(path)))
            .with_device(device)
            .build()
            .await
            .unwrap();

        let warm = 6usize;
        let measured = 10usize;
        let prompt = "Once upon a time there was a penguin named Peng.";
        let mut story = model(prompt).with_sampler(
            GenerationParameters::new().with_max_length((warm + measured + 16) as u32),
        );

        let mut spans: Option<Vec<f64>> = None;
        let mut totals: Vec<f64> = Vec::new();
        let mut tokens = 0usize;
        while let Some(token) = story.next().await {
            tokens += 1;
            print!("{token}");
            std::io::stdout().flush().unwrap();
            if tokens == warm {
                target.launcher().set_tuning(true);
            } else if tokens > warm {
                if let Some(us) = target.launcher().take_last_profile() {
                    totals.push(us.iter().sum());
                    spans = Some(match spans {
                        Some(prev) if prev.len() == us.len() => {
                            prev.iter().zip(&us).map(|(a, b)| a.min(*b)).collect()
                        }
                        _ => us,
                    });
                }
                if tokens >= warm + measured {
                    break;
                }
            }
        }
        target.launcher().set_tuning(false);
        println!();
        let Some(spans) = spans else {
            println!("SPAN_TOTAL none (explorer owned the clock)");
            return;
        };
        let total: f64 = spans.iter().sum();
        println!(
            "SPAN_TOTAL launches={} min_total_us={:.1} median_token_us={:.1}",
            spans.len(),
            total,
            {
                let mut t = totals.clone();
                t.sort_by(f64::total_cmp);
                t[t.len() / 2]
            }
        );
        for (i, s) in spans.iter().enumerate() {
            println!("SPAN {i} {s:.2}");
        }
    });
}
