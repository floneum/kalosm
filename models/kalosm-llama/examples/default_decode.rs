//! Bounded end-to-end decode benchmark through the ordinary Llama builder.
//! No kernel-visualize feature or tuning environment is needed.
//! cargo run -j 1 -p kalosm-llama --example default_decode --release -- MODEL.gguf
use kalosm_llama::{prelude::*, Device};
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let path = PathBuf::from(args.next().ok_or_else(|| {
        anyhow::anyhow!("usage: default_decode MODEL.gguf [STEPS=32] [ROUNDS=1] [default|general]")
    })?);
    let steps = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(32);
    let rounds = args
        .next()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let general = match args.next().as_deref() {
        None | Some("default") => false,
        Some("general") => true,
        _ => anyhow::bail!("device must be default or general"),
    };
    anyhow::ensure!(
        args.next().is_none() && path.is_file(),
        "expected an existing GGUF file"
    );
    anyhow::ensure!(
        (2..=64).contains(&steps) && (1..=10).contains(&rounds),
        "steps 2..64, rounds 1..10"
    );
    pollster::block_on(async {
        let mut builder = Llama::builder().with_source(LlamaSource::new(FileSource::Local(path)));
        if general {
            builder = builder.with_device(Device::gpu().await?);
        }
        let model = builder.build().await?;
        let prompt = format!("{}\nSummarize the main point:", "A GPU executes groups of threads. Adjacent memory accesses can share transactions. Shared memory can retain data for reuse across threads. ".repeat(8));
        let prompt_tokens = model.tokenizer().encode(&prompt, false)?.len();
        let mut samples = Vec::new();
        for round in 0..=rounds {
            let mut session = model.new_session()?;
            for step in 0..=steps {
                let text = Arc::new(Mutex::new(String::new()));
                let capture = text.clone();
                let input = if step == 0 {
                    prompt.clone()
                } else {
                    String::new()
                };
                let start = Instant::now();
                model
                    .stream_text_with_callback(
                        &mut session,
                        input.into(),
                        GenerationParameters::new().with_seed(0).with_max_length(1),
                        move |chunk| {
                            capture.lock().unwrap().push_str(&chunk);
                            Ok(())
                        },
                    )
                    .await?;
                let wall_ms = start.elapsed().as_secs_f64() * 1e3;
                if round > 0 && step > 1 {
                    samples.push(wall_ms);
                }
                println!(
                    "{}",
                    serde_json::json!({"prompt": "long", "prompt_tokens": prompt_tokens,
                    "round": round, "step": step, "phase": if step == 0 { "prefill" } else { "decode" },
                    "warmup": round == 0, "wall_ms": wall_ms, "text": *text.lock().unwrap()})
                );
            }
        }
        samples.sort_by(f64::total_cmp);
        let median = (samples[(samples.len() - 1) / 2] + samples[samples.len() / 2]) / 2.;
        eprintln!(
            "{}",
            serde_json::json!({"samples": samples.len(), "median_ms": median, "tokens_per_second": 1000./median})
        );
        Ok(())
    })
}
