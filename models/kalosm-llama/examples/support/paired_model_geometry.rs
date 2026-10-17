//! One loaded model, two short KV histories, matching decode steps in AB/BA
//! order. Each history advances once per step; no stateful plan is replayed.
use fusor_ir::{
    dtype::{Dtype, QFmt},
    ir::{
        Op,
        launch::{Launch, SchedPoint, SgemvParams},
    },
};
use kalosm_llama::{Device, prelude::*};
use serde_json::json;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::Instant,
};

pub async fn run(
    model: &Llama,
    device: &Device,
    cell: &str,
    rounds: usize,
    steps: usize,
    prompt_repeats: usize,
    dispatches: Arc<AtomicUsize>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        rounds == 1 && steps <= 32,
        "paired model geometry permits one warmup and one measured round, with at most 32 decode steps"
    );
    anyhow::ensure!(
        prompt_repeats <= 8,
        "paired model probe limits the prompt to 208 tokens"
    );
    anyhow::ensure!(
        std::env::var("FUSOR_AUTOTUNE_MIN_MACS").as_deref() == Ok("18446744073709551615"),
        "paired model geometry requires disabled online tuning"
    );
    let parts: Vec<u32> = cell.split(',').map(str::parse).collect::<Result<_, _>>()?;
    let [vector, subgroups, cols, parts, gap] = parts[..] else {
        anyhow::bail!("expected vector,subgroups,cols,parts,gap");
    };
    anyhow::ensure!(
        vector == 32
            && parts == 4
            && gap == 32
            && [1, 2, 4, 8, 16].contains(&subgroups)
            && cols.is_multiple_of(subgroups)
            && (1..=8).contains(&(cols / subgroups)),
        "unsupported comparison cell"
    );
    let candidate_cell = SgemvParams {
        vector,
        subgroups,
        cols,
        parts,
        gap,
    };
    let active = Arc::new(AtomicBool::new(false));
    let decoding = Arc::new(AtomicBool::new(false));
    let matched = Arc::new(AtomicUsize::new(0));
    let target = device.backend().gpu_target().expect("WebGPU target");
    let poll_spin = std::env::var("PROFILE_PAIRED_POLL_SPIN_MS")
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()?;
    anyhow::ensure!(
        poll_spin.is_none_or(|n| n <= 64),
        "poll window must be 0..=64 ms"
    );
    let poll_interval = std::env::var("PROFILE_PAIRED_POLL_INTERVAL_US")
        .ok()
        .map(|value| value.parse::<u64>())
        .transpose()?;
    anyhow::ensure!(
        poll_interval.is_none_or(|n| n <= 1000),
        "poll interval must be 0..=1000 us"
    );
    let arithmetic = std::env::var("PROFILE_PAIRED_Q4_VARIANT")
        .ok()
        .map(|value| value.parse::<u8>())
        .transpose()?;
    anyhow::ensure!(
        arithmetic.is_none_or(|n| (1..=3).contains(&n)),
        "Q4 variant must be 1..=3"
    );
    let stream_rest = std::env::var("PROFILE_PAIRED_STREAM_REST")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    anyhow::ensure!(
        stream_rest.is_none_or(|n| (32..=4096).contains(&n)),
        "paired stream rest must be 32..=4096"
    );
    let candidate_active = active.clone();
    let is_decode = decoding.clone();
    let matches = matched.clone();
    target.set_schedule_override(Some(Arc::new(move |plan, graph, binds| {
        if !is_decode.load(Ordering::Relaxed) {
            return None;
        }
        // Both arms clone and scan the same-sized metadata so the diagnostic
        // hook's work is present on both sides of the timing comparison.
        let mut result = plan.clone();
        for launch in &plan.launches {
            let Op::Launch(Launch::Contract { m, n, k, b, .. }) = &graph.node(launch.root).op
            else {
                continue;
            };
            let dim = |d: fusor_ir::shape::Dim| d.evaluate(&mut |s| binds.dims.get(&s).copied());
            if dim(*m) != Some(1)
                || dim(*n).is_none_or(|n| n < 16384)
                || dim(*k).is_none_or(|k| k < 4096)
                || b.ops.len() != 1
                || graph.facts(b.ops[0].src).dtype != Dtype::Q(QFmt::Q4K)
                || !matches!(
                    result.extraction.theta.get(&launch.root),
                    Some(SchedPoint::Sgemv(_))
                )
            {
                continue;
            }
            matches.fetch_add(1, Ordering::Relaxed);
            if candidate_active.load(Ordering::Relaxed) {
                result
                    .extraction
                    .theta
                    .insert(launch.root, SchedPoint::Sgemv(candidate_cell));
            }
        }
        Some(Arc::new(result))
    })));
    let prompt = format!("{}\nSummarize the main point:", "A GPU executes groups of threads. Adjacent memory accesses can share transactions. Shared memory can retain data for reuse across threads. ".repeat(prompt_repeats));
    let prompt_tokens = model.tokenizer().encode(&prompt, false)?.len();
    let mut logit_index = 0usize;
    let capture_logits = std::env::var_os("PROFILE_LOGITS_DIR").is_some();
    for round in 0..=rounds {
        let mut sessions = [model.new_session()?, model.new_session()?];
        for step in 0..=steps {
            let order = if step.is_multiple_of(2) {
                [0, 1]
            } else {
                [1, 0]
            };
            let mut texts: [String; 2] = Default::default();
            let mut times = [0.; 2];
            let mut counts = [0; 2];
            let mut selected = [0; 2];
            let mut indices = [0; 2];
            for arm in order {
                target.set_q4_variant(arithmetic.map(|n| if arm == 0 { 0 } else { n }))?;
                target.set_stream_submit_rest(
                    stream_rest
                        .filter(|_| step > 0)
                        .map(|candidate| if arm == 0 { 32 } else { candidate }),
                )?;
                target.launcher().set_poll_spin_ms(
                    poll_spin.map(|candidate| if arm == 0 { 2 } else { candidate }),
                )?;
                target.launcher().set_poll_interval_us(
                    poll_interval.map(|candidate| if arm == 0 { 0 } else { candidate }),
                )?;
                active.store(arm == 1, Ordering::Relaxed);
                decoding.store(step > 0, Ordering::Relaxed);
                matched.store(0, Ordering::Relaxed);
                dispatches.store(0, Ordering::Relaxed);
                let capture = Arc::new(Mutex::new(String::new()));
                let output = capture.clone();
                let input = if step == 0 {
                    prompt.clone()
                } else {
                    String::new()
                };
                eprintln!("PAIRED_PHASE_BEGIN round={round} step={step} arm={arm}");
                let start = Instant::now();
                model
                    .stream_text_with_callback(
                        &mut sessions[arm],
                        input.into(),
                        GenerationParameters::new().with_seed(0).with_max_length(1),
                        move |text| {
                            output.lock().unwrap().push_str(&text);
                            Ok(())
                        },
                    )
                    .await?;
                times[arm] = start.elapsed().as_secs_f64() * 1e3;
                texts[arm] = capture.lock().unwrap().clone();
                counts[arm] = dispatches.load(Ordering::Relaxed);
                selected[arm] = matched.load(Ordering::Relaxed);
                indices[arm] = logit_index;
                logit_index += 1;
            }
            anyhow::ensure!(
                texts[0] == texts[1],
                "paired generated text differs at round {round} step {step}"
            );
            // The first prefill may initialize shared weight representations
            // that the second history reuses. Only matched steady decode is a
            // timing comparison; every phase still receives full logit checks.
            anyhow::ensure!(
                round == 0 || step <= 1 || counts[0] == counts[1],
                "schedule-only comparison changed steady dispatch count: {counts:?}"
            );
            anyhow::ensure!(
                step == 0 || (selected[0] > 0 && selected[0] == selected[1]),
                "paired geometry did not match equal decode projections"
            );
            println!(
                "MODEL_GEOMETRY_PAIR {}",
                json!({"round":round,"step":step,"order":order,
                "phase":if step==0 {"prefill"} else {"decode"},"prompt_tokens":prompt_tokens,
                "warmup":round==0 || step<=1,"baseline_ms":times[0],"candidate_ms":times[1],
                "speedup":times[0]/times[1],"text":texts[0],"dispatches":counts,"matched":selected,
                "logit_indices":capture_logits.then_some(indices),"candidate_cell":[vector,subgroups,cols,parts,gap],
                "poll_spin_ms":poll_spin.map(|candidate| [2,candidate]),
                "poll_interval_us":poll_interval.map(|candidate| [0,candidate]),
                "q4_variant":arithmetic.map(|n| [0,n]),
                "stream_rest":stream_rest.filter(|_| step > 0).map(|candidate| [32,candidate])})
            );
        }
    }
    target.set_schedule_override(None);
    target.launcher().set_poll_spin_ms(None)?;
    target.launcher().set_poll_interval_us(None)?;
    target.set_q4_variant(None)?;
    target.set_stream_submit_rest(None)?;
    Ok(())
}
