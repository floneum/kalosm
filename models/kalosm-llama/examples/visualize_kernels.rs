#![recursion_limit = "256"]
//! Capture every distinct lowered kernel from local Llama inference, then run
//! the virtual GPU on sampled workgroups. Unsupported kernels remain in the
//! inventory with errors. Lowered candidates and submitted launches are separate.
use std::{
    collections::{BTreeMap, BTreeSet},
    hash::{Hash, Hasher},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use fusor_ir::ir::kernel::{KernelIr, Stmt, TileExpr, TileExprKind};
use fusor_visualize::{
    kernel::{capture, CaptureOptions},
    GpuConfig, Simulator,
};
use kalosm_llama::{prelude::*, Device};
use serde_json::{json, Value};

struct Observed {
    ir: KernelIr,
    uniforms: Option<Vec<u8>>,
    observations: usize,
    runs: BTreeSet<String>,
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = PathBuf::from(args.next().ok_or_else(|| {
        anyhow::anyhow!("usage: visualize_kernels MODEL.gguf OUTPUT_DIR [CHUNKS=4] [BODY_PREFIX_CSV] [NEIGHBORS=1]")
    })?);
    let output = PathBuf::from(
        args.next()
            .ok_or_else(|| anyhow::anyhow!("provide a new output directory"))?,
    );
    let tokens = args
        .next()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(4);
    let filters: Vec<String> = args
        .next()
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    anyhow::ensure!(
        filters
            .iter()
            .all(|s| s.len() <= 32 && s.bytes().all(|b| b.is_ascii_hexdigit())),
        "body filters must be hexadecimal prefixes, separated by commas"
    );
    let neighbors = args
        .next()
        .map(|v| v.parse::<u32>())
        .transpose()?
        .unwrap_or(1);
    anyhow::ensure!((1..=32).contains(&neighbors), "neighbors must be 1..=32");
    anyhow::ensure!(args.next().is_none(), "too many arguments");
    anyhow::ensure!(model_path.is_file(), "model file does not exist");
    anyhow::ensure!(tokens > 0 && tokens <= 128, "tokens must be 1..=128");
    // Refuse to mix artifacts from different runs.
    std::fs::create_dir(&output)?;
    std::fs::create_dir(output.join("kernels"))?;
    let inventory = Arc::new(Mutex::new(BTreeMap::<String, Observed>::new()));
    let submissions = Arc::new(Mutex::new(BTreeMap::<String, usize>::new()));
    let label = Arc::new(Mutex::new(String::from("model-load")));
    let mut runs = vec![];
    let mut native_subgroup = 32;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pollster::block_on(async {
            let device = Device::gpu().await?;
            let target = device.backend().gpu_target().expect("GPU device");
            native_subgroup = device.backend().caps().subgroup_width();
            let seen = inventory.clone();
            let phase = label.clone();
            let lower_filters = filters.clone();
            target.set_kernel_observer(Some(Arc::new(move |ir, body_id, uniforms| {
                if !matches_body(body_id, &lower_filters) {
                    return;
                }
                let uniforms = uniforms.map(|u| u.to_bytes());
                let id = variant_id(body_id, ir.grid, uniforms.as_deref());
                let run = phase.lock().unwrap().clone();
                let mut seen = seen.lock().unwrap();
                let entry = seen.entry(id).or_insert_with(|| Observed {
                    ir: ir.clone(),
                    uniforms,
                    observations: 0,
                    runs: BTreeSet::new(),
                });
                entry.observations += 1;
                entry.runs.insert(run);
            })));
            let submitted = submissions.clone();
            let submit_filters = filters.clone();
            target.set_dispatch_observer(Some(Arc::new(move |body_id, grid, uniforms| {
                if matches_body(body_id, &submit_filters) {
                    let bytes = uniforms.to_bytes();
                    let id = variant_id(body_id, grid, Some(&bytes));
                    *submitted.lock().unwrap().entry(id).or_default() += 1;
                }
            })));
            let model = Llama::builder()
                .with_source(LlamaSource::new(FileSource::Local(model_path.clone())))
                .with_device(device.clone())
                .build()
                .await?;
            for (name, prompt) in [
            ("short-prefill-and-decode", "The capital of France is".to_string()),
            ("long-prefill-and-decode", format!("{}\nSummarize the main point:", "A GPU executes groups of threads. Adjacent memory accesses can share transactions. Shared memory can retain data for reuse across threads. ".repeat(8))),
        ] {
            *label.lock().unwrap() = name.into();
            let prompt_tokens = model.tokenizer().encode(&prompt, true)?.len();
            eprintln!("run={name} prompt_tokens={prompt_tokens} requested_chunks={tokens}");
            let sampler = GenerationParameters::new().with_seed(0).with_max_length((prompt_tokens + tokens + 16) as u32);
            let mut stream = model.complete(&prompt).with_sampler(sampler);
            let mut chunks = 0;
            let mut text = String::new();
            while chunks < tokens {
                let Some(chunk) = stream.next().await else { break; };
                text.push_str(&chunk);
                chunks += 1;
            }
            runs.push(json!({"name": name, "prompt_tokens": prompt_tokens, "requested_chunks": tokens,
                "emitted_chunks": chunks, "completed": chunks == tokens, "text": text}));
            eprintln!("run={name} emitted_chunks={chunks} distinct_variants={}", inventory.lock().unwrap().len());
        }
            target.set_kernel_observer(None);
            target.set_dispatch_observer(None);
            Ok::<_, anyhow::Error>(())
        })
    }));
    let inference_error = match result {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(error.to_string()),
        Err(payload) => Some(
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "inference panicked".into()),
        ),
    };
    let config = GpuConfig {
        subgroup_size: native_subgroup.max(1),
        ..GpuConfig::default()
    };
    let mut rows = vec![];
    let inventory = inventory.lock().unwrap();
    for (index, (id, observed)) in inventory.iter().enumerate() {
        let stem = id.clone();
        let ir = &observed.ir;
        let features = features(ir);
        let workgroups = sampled_workgroups(ir.grid, neighbors);
        let known_bindings = observed
            .uniforms
            .clone()
            .map(|bytes| BTreeMap::from([(0, bytes)]))
            .unwrap_or_default();
        let mut row = json!({"id": stem, "name": ir.name, "grid": ir.grid, "block": ir.block,
            "observations": observed.observations, "runs": observed.runs, "features": features,
            "submitted_dispatches": submissions.lock().unwrap().get(id).copied().unwrap_or(0),
            "sampled_workgroups": workgroups, "uniforms_supplied": observed.uniforms.is_some()});
        // Debug IR is useful for unsupported kernels and does not contain weights.
        std::fs::write(
            output.join("kernels").join(format!("{stem}.ir.txt")),
            bounded_debug(ir),
        )?;
        let result = capture(
            ir,
            CaptureOptions {
                subgroup_size: config.subgroup_size,
                workgroups,
                max_actions: 100_000,
                known_bindings,
            },
        );
        match result.and_then(|trace| Simulator::new(trace, config.clone())) {
            Ok(mut simulator) => {
                let trace_path = format!("kernels/{stem}.trace.json");
                serde_json::to_writer(
                    std::fs::File::create(output.join(&trace_path))?,
                    simulator.trace(),
                )?;
                let mut hotspots = BTreeMap::<String, Value>::new();
                while let Some(step) = simulator.step() {
                    let m = &step.metrics;
                    if m.transactions > m.ideal_transactions
                        || m.bank_conflict_steps > 0
                        || m.cross_subgroup_reuse_bytes > 0
                    {
                        let severity = m.transactions.saturating_sub(m.ideal_transactions)
                            + m.max_bank_degree.saturating_sub(m.ideal_bank_degree);
                        let slot = hotspots
                            .entry(step.source.clone())
                            .or_insert(json!({"severity": 0}));
                        if severity >= slot["severity"].as_u64().unwrap_or(0) {
                            *slot = json!({"severity": severity, "source": step.source, "workgroup": step.workgroup,
                                "subgroup": step.subgroup, "metrics": m, "hints": step.hints});
                        }
                    }
                }
                row["status"] = json!("analyzed");
                row["trace"] = json!(trace_path);
                row["metrics"] = serde_json::to_value(simulator.totals())?;
                row["hotspots"] = json!(hotspots.into_values().collect::<Vec<_>>());
            }
            Err(error) => {
                row["status"] = json!(if error.to_string().contains("out-of-bounds") {
                    "trace_bounds_error"
                } else {
                    "unsupported"
                });
                row["error"] = json!(error.to_string());
            }
        }
        eprintln!(
            "analyzed {}/{} {} {}",
            index + 1,
            inventory.len(),
            ir.name,
            row["status"]
        );
        rows.push(row);
    }
    let analyzed = rows.iter().filter(|r| r["status"] == "analyzed").count();
    let bounds_errors = rows
        .iter()
        .filter(|r| r["status"] == "trace_bounds_error")
        .count();
    let unsupported = rows.len() - analyzed - bounds_errors;
    let report = json!({"version": 1, "model_path": model_path, "model_bytes": std::fs::metadata(&model_path)?.len(),
        "scope": "Distinct uncached lowered IR plus uniform values matching body_filters (empty means all), including tuning candidates; not every possible context length.",
        "sampling": format!("First and last {neighbors} neighboring workgroups in X-fast linear order; overlapping windows deduplicated. Gaps between windows are omitted. Totals are sampled, not full-grid traffic."),
        "body_filters": filters,
        "capture_semantics": "Logical masked IR loads before backend optimization. The emitter can issue clamped loads on masked-out lanes; inspect generated WGSL before interpreting these counts as physical instructions.",
        "dispatch_scope": "Successful nonempty submissions through GpuTarget::resolve, including warm artifact hits and tuning runs. Counts do not establish which plan ultimately wins. Direct Target::launch calls are excluded.",
        "submitted_variants": &*submissions.lock().unwrap(),
        "config": config, "runs": runs, "inference_error": inference_error,
        "bounds_note": "Trace bounds failures compare logical IR addresses with declared resource extents before backend clamping/robustness; they are not proof of a runtime out-of-bounds access.",
        "variants": rows.len(), "analyzed": analyzed, "trace_bounds_errors": bounds_errors, "unsupported": unsupported, "kernels": rows});
    serde_json::to_writer_pretty(std::fs::File::create(output.join("report.json"))?, &report)?;
    println!(
        "{} variants: {analyzed} analyzed, {bounds_errors} trace bounds failures, {unsupported} unsupported. Report: {}",
        rows.len(),
        output.join("report.json").display()
    );
    if let Some(error) = inference_error {
        anyhow::bail!("inference failed; partial inventory retained: {error}");
    }
    anyhow::ensure!(
        !rows.is_empty()
            && report["runs"]
                .as_array()
                .unwrap()
                .iter()
                .all(|r| r["completed"] == true),
        "inference did not complete every requested run; see report"
    );
    Ok(())
}

fn matches_body(id: u128, filters: &[String]) -> bool {
    filters.is_empty()
        || filters
            .iter()
            .any(|prefix| format!("{id:032x}").starts_with(&prefix.to_ascii_lowercase()))
}

fn variant_id(body: u128, grid: [u32; 3], uniforms: Option<&[u8]>) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    grid.hash(&mut hasher);
    uniforms.hash(&mut hasher);
    format!("{body:032x}-{:016x}", hasher.finish())
}

fn sampled_workgroups(grid: [u32; 3], neighbors: u32) -> Vec<[u32; 3]> {
    let count: u64 = grid.iter().map(|&n| u64::from(n)).product();
    let n = u64::from(neighbors).min(count);
    (0..n)
        .chain(count.saturating_sub(n)..count)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|i| {
            [
                (i % u64::from(grid[0])) as u32,
                ((i / u64::from(grid[0])) % u64::from(grid[1])) as u32,
                (i / (u64::from(grid[0]) * u64::from(grid[1]))) as u32,
            ]
        })
        .collect()
}

fn features(ir: &KernelIr) -> BTreeSet<&'static str> {
    fn expr(e: &TileExpr, features: &mut BTreeSet<&'static str>, seen: &mut BTreeSet<usize>) {
        if !seen.insert(e.node_ptr()) {
            return;
        }
        match e.kind() {
            TileExprKind::Load {
                src: fusor_ir::ir::kernel::Source::Quantized(_),
                ..
            } => {
                features.insert("quantized");
            }
            TileExprKind::CoopLoad { .. }
            | TileExprKind::CoopMma { .. }
            | TileExprKind::CoopZero { .. } => {
                features.insert("cooperative_matrix");
            }
            TileExprKind::Reduce { .. } => {
                features.insert("reduction");
            }
            TileExprKind::LoadTile { .. } => {
                features.insert("workgroup_memory");
            }
            _ => {}
        }
        e.kind()
            .visit_children(&mut |child| expr(child, features, seen));
    }
    let mut result = BTreeSet::new();
    let mut seen = BTreeSet::new();
    Stmt::walk(&ir.body, &mut |stmt| {
        match stmt {
            Stmt::FillTile { .. } => {
                result.insert("collective_fill");
            }
            Stmt::Reduce { .. } => {
                result.insert("reduction");
            }
            Stmt::StoreTile { .. } => {
                result.insert("workgroup_memory");
            }
            _ => {}
        }
        stmt.for_each_expr(&mut |e| expr(e, &mut result, &mut seen));
    });
    result
}

fn bounded_debug(ir: &KernelIr) -> String {
    use std::fmt::Write;
    struct Limited(String);
    impl std::fmt::Write for Limited {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            if self.0.len() + s.len() > 1024 * 1024 {
                return Err(std::fmt::Error);
            }
            self.0.push_str(s);
            Ok(())
        }
    }
    let mut out = Limited(String::new());
    if write!(&mut out, "{ir:#?}").is_err() {
        out.0.push_str("\n[debug IR truncated at 1 MiB]\n");
    }
    out.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighboring_windows_cross_grid_rows_without_duplicates() {
        assert_eq!(
            sampled_workgroups([4, 2, 1], 3),
            vec![
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [1, 1, 0],
                [2, 1, 0],
                [3, 1, 0],
            ]
        );
        let all = sampled_workgroups([4, 2, 2], 32);
        assert_eq!(all.len(), 16);
        assert_eq!(all[4], [0, 1, 0]);
        assert_eq!(all[8], [0, 0, 1]);
        assert_eq!(all[15], [3, 1, 1]);
        assert!(sampled_workgroups([0, 2, 1], 1).is_empty());
    }

    #[test]
    fn borrowed_uniform_hash_matches_existing_trace_ids() {
        let bytes = Some(vec![1u8, 2, 3, 4]);
        let grid = [49152u32, 2, 1];
        let mut old = std::collections::hash_map::DefaultHasher::new();
        grid.hash(&mut old);
        bytes.hash(&mut old);
        assert_eq!(
            variant_id(17, grid, bytes.as_deref()),
            format!("{:032x}-{:016x}", 17, old.finish())
        );
    }
}
