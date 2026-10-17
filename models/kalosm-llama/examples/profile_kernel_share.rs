#![recursion_limit = "256"]
//! Attribute real inference to captured kernel bodies. Run in separate processes
//! with no timing flag, FUSOR_TIME_PLAN=1, and FUSOR_TIME_RANGE=0. The latter logs
//! exact body identities in dispatch order for joining to the launcher's TSPAN.
//! Round zero and the first decode of each fresh KV cache are excluded from the
//! steady decode summary. Fresh-cache prefill can still compile new graphs, so its
//! wall time is not a warm GPU benchmark. Each awaited call requests exactly one
//! generated token, so prefill and decode cannot overlap.
use kalosm_llama::{prelude::*, Device};
use serde_json::json;
use std::{
    path::PathBuf,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Instant,
};
#[path = "support/paired_model_geometry.rs"]
mod paired_model_geometry;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(args.next().ok_or_else(|| {
        anyhow::anyhow!("usage: profile_kernel_share MODEL.gguf [ROUNDS=3] [DECODE_STEPS=6]")
    })?);
    let rounds = args
        .next()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(3);
    let decode_steps = args
        .next()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(6);
    anyhow::ensure!(args.next().is_none(), "too many arguments");
    anyhow::ensure!((1..=10).contains(&rounds), "rounds must be 1..=10");
    anyhow::ensure!(
        (1..=64).contains(&decode_steps),
        "decode steps must be 1..=64"
    );
    anyhow::ensure!(model_path.is_file(), "model file does not exist");
    let prompt_filter = std::env::var("PROFILE_PROMPT").ok();
    let prompt_repeats = std::env::var("PROFILE_PROMPT_REPEATS")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(8);
    anyhow::ensure!(
        (1..=512).contains(&prompt_repeats),
        "PROFILE_PROMPT_REPEATS must be 1..=512"
    );
    anyhow::ensure!(
        prompt_filter
            .as_deref()
            .is_none_or(|p| matches!(p, "short" | "long")),
        "PROFILE_PROMPT must be short or long"
    );
    let detailed = std::env::var_os("FUSOR_TIME_RANGE").is_some();
    let plan_repeats = std::env::var("PROFILE_REPLAY_PLAN")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(0);
    if plan_repeats > 0 {
        anyhow::ensure!(
            std::env::var_os("FUSOR_PROFILE_CACHE_DONATION").is_none(),
            "plan replay requires immutable cache inputs; disable cache donation"
        );
        anyhow::ensure!(
            (2..=32).contains(&plan_repeats),
            "PROFILE_REPLAY_PLAN must be 2..=32"
        );
        anyhow::ensure!(
            std::env::var_os("PROFILE_METADATA").is_none(),
            "plan replay and metadata capture are separate probes"
        );
        anyhow::ensure!(
            std::env::var("FUSOR_AUTOTUNE_MIN_MACS").as_deref() == Ok("18446744073709551615"),
            "plan replay requires disabled online tuning"
        );
        for name in ["FUSOR_PROFILE_SUBMIT_SIZE", "FUSOR_PASS_SIZE"] {
            anyhow::ensure!(
                std::env::var(name).as_deref() == Ok("2147483647"),
                "plan replay requires {name}=2147483647 for one preencoded submission/pass"
            );
        }
    }
    anyhow::ensure!(
        !(detailed && std::env::var_os("FUSOR_TIME_PLAN").is_some()),
        "use only one GPU timing mode per process"
    );
    let mode = if plan_repeats > 0 {
        "replay"
    } else if std::env::var_os("FUSOR_TIME_SAMPLE_STRIDE").is_some() {
        "sampled"
    } else if detailed {
        "dispatch"
    } else if std::env::var_os("FUSOR_TIME_PLAN").is_some() {
        "whole"
    } else {
        "wall"
    };
    pollster::block_on(async {
        let device = if std::env::var_os("PROFILE_DEFAULT_DEVICE").is_some() {
            anyhow::ensure!(
                plan_repeats == 0,
                "default cache donation forbids immutable plan replay"
            );
            Device::gpu_inference().await?
        } else {
            Device::gpu().await?
        };
        eprintln!("EXECUTION_PROFILE {:?}", device.execution_profile());
        let target = device.backend().gpu_target().expect("GPU device");
        let replay_active = Arc::new(AtomicUsize::new(0));
        if plan_repeats > 0 {
            let active = replay_active.clone();
            let probe = device.clone();
            target.set_plan_observer(Some(Arc::new(move |plan, _, _, _, commands| {
                if active.load(Ordering::Relaxed) == 0 { return; }
                let target = probe.backend().gpu_target().unwrap();
                let launcher = target.launcher();
                // The callback runs before resolve releases any root, arena
                // or uniform. Pure forward commands therefore reuse the same
                // inputs and outputs; this is a GPU diagnostic, not decoding
                // additional tokens or an end-to-end speedup measurement.
                launcher.set_tuning(true);
                let query = launcher.timestamp_query_set(1).expect("replay timestamps");
                let replay: Vec<_> = (0..plan_repeats).flat_map(|_| commands.iter().cloned()).collect();
                let start = Instant::now();
                launcher.encode_command_records(&replay, Some(&query), fusor_gpu::launch::TimingMode::Whole).expect("plan replay submission");
                let encode_ms = start.elapsed().as_secs_f64() * 1e3;
                launcher.poll_wait().expect("plan replay completion");
                let gpu_us = launcher.read_timestamps(target.pool(), &query, 1).expect("plan replay timing")[0];
                launcher.set_tuning(false);
                eprintln!("REPLAY_PLAN {}", json!({"plan": format!("{:032x}", plan.hash.0), "commands": commands.len(), "repeats": plan_repeats, "encode_ms": encode_ms, "gpu_ms_per_replay": gpu_us / 1000. / plan_repeats as f64}));
            })));
        }
        if std::env::var_os("PROFILE_METADATA").is_some() {
            // Repeatedly describing a model-scale graph perturbs the gaps
            // between GPU samples. Cache only exact plan/body/grid matches,
            // and recheck every dimension actually read by the description.
            let cached_metadata = std::env::var_os("PROFILE_METADATA_CACHE").is_some();
            struct MetadataKey {
                plan: u128,
                bodies: Vec<(u128, [u32; 3])>,
                dimensions: Vec<(fusor_ir::shape::SymId, Option<u64>)>,
            }
            let descriptions = Mutex::new(Vec::<MetadataKey>::new());
            target.set_plan_observer(Some(Arc::new(
                move |plan, graph, binds, identities, commands| {
                    let mut descriptions = descriptions.lock().unwrap();
                    if cached_metadata {
                        if let Some((metadata_id, _)) =
                            descriptions.iter().enumerate().find(|(_, key)| {
                                key.plan == plan.hash.0
                                    && key.bodies == identities
                                    && key
                                        .dimensions
                                        .iter()
                                        .all(|(s, value)| binds.dims.get(s).copied() == *value)
                            })
                        {
                            eprintln!("RESOLVE_META_REF {}", json!({"metadata_id": metadata_id}));
                            return;
                        }
                    }
                    let used = std::cell::RefCell::new(std::collections::BTreeSet::new());
                    let mut description = fusor_visualize::plan::describe(
                        plan,
                        graph,
                        &|d| {
                            d.evaluate(&mut |s| {
                                used.borrow_mut().insert(s);
                                binds.dims.get(&s).copied()
                            })
                        },
                        identities,
                    );
                    if cached_metadata {
                        description["metadata_id"] = json!(descriptions.len());
                        descriptions.push(MetadataKey {
                            plan: plan.hash.0,
                            bodies: identities.to_vec(),
                            dimensions: used
                                .into_inner()
                                .into_iter()
                                .map(|s| (s, binds.dims.get(&s).copied()))
                                .collect(),
                        });
                    }
                    eprintln!("RESOLVE_META {description}");
                    let copies: Vec<_> = commands
                        .iter()
                        .enumerate()
                        .filter_map(|(i, c)| {
                            c.copy_bytes()
                                .map(|bytes| json!({"command": i, "bytes": bytes}))
                        })
                        .collect();
                    eprintln!(
                        "COMMAND_META {}",
                        json!({"commands": commands.len(), "copies": copies})
                    );
                },
            )));
        }
        let count = Arc::new(AtomicUsize::new(0));
        let observed = count.clone();
        target.set_dispatch_observer(Some(Arc::new(move |body, grid, _| {
            observed.fetch_add(1, Ordering::Relaxed);
            if detailed {
                eprintln!("BODY {body:032x} grid={grid:?}");
            }
        })));
        let model = Llama::builder()
            .with_source(LlamaSource::new(FileSource::Local(model_path)))
            .with_device(device.clone())
            .build()
            .await?;
        if let Ok(cell) = std::env::var("PROFILE_PAIRED_GEOMETRY") {
            anyhow::ensure!(
                mode == "wall" && std::env::var_os("PROFILE_METADATA").is_none(),
                "paired geometry requires clean wall mode"
            );
            paired_model_geometry::run(
                &model,
                &device,
                &cell,
                rounds,
                decode_steps,
                prompt_repeats,
                count.clone(),
            )
            .await?;
            target.set_dispatch_observer(None);
            return Ok(());
        }
        for round in 0..=rounds {
            for (name, prompt) in [
                ("short", "The capital of France is".to_string()),
                ("long", format!("{}\nSummarize the main point:", "A GPU executes groups of threads. Adjacent memory accesses can share transactions. Shared memory can retain data for reuse across threads. ".repeat(prompt_repeats))),
            ] {
                if prompt_filter.as_deref().is_some_and(|v| v != name) { continue; }
                let prompt_tokens = model.tokenizer().encode(&prompt, false)?.len();
                let mut session = model.new_session()?;
                for step in 0..=decode_steps {
                    let phase = if step == 0 { "prefill" } else { "decode" };
                    let key = format!("{name}-r{round}-{phase}-{step}");
                    let input = if step == 0 { prompt.clone() } else { String::new() };
                    count.store(0, Ordering::Relaxed);
                    replay_active.store(usize::from(round > 0 && step == decode_steps), Ordering::Relaxed);
                    eprintln!("PHASE_BEGIN {key}");
                    let generated = Arc::new(Mutex::new(String::new()));
                    let capture = generated.clone();
                    let start = Instant::now();
                    model.stream_text_with_callback(
                        &mut session,
                        input.into(),
                        GenerationParameters::new().with_seed(0).with_max_length(1),
                        move |text| { capture.lock().unwrap().push_str(&text); Ok(()) },
                    ).await?;
                    let wall_ms = start.elapsed().as_secs_f64() * 1e3;
                    let row = json!({"key": key, "mode": mode, "prompt": name,
                        "prompt_tokens": prompt_tokens, "round": round, "phase": phase,
                        "step": step, "warmup": round == 0, "wall_ms": wall_ms,
                        "dispatches": count.load(Ordering::Relaxed), "text": *generated.lock().unwrap()});
                    eprintln!("PHASE_END {row}");
                    println!("{row}");
                }
            }
        }
        target.set_dispatch_observer(None);
        target.set_plan_observer(None);
        Ok(())
    })
}
