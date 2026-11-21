use fusor_sim::{FailureRecord, Limits, campaign_with_limits, minimize_failure, replay_failure};
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("fusor-sim: {error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    let mut args = std::env::args().skip(1);
    match args.next().as_deref() {
        Some("replay") => {
            let path = args.next().ok_or("usage: fusor-sim replay <json>")?;
            let record = read_record(&path)?;
            println!("{}", replay_failure(&record)?);
        }
        Some("minimize") => {
            let path = args.next().ok_or("usage: fusor-sim minimize <json>")?;
            let record = read_record(&path)?;
            let minimized = minimize_failure(&record)?;
            let output = Path::new(&path).with_extension("minimized.json");
            write_record(&output, &minimized)?;
            println!("minimized failure saved to {}", output.display());
        }
        command => {
            let mut seed = 1u64;
            let mut iterations = 4u32;
            let mut limits = Limits::default();
            let mut failure_path = PathBuf::from("target/fusor-sim-first-failure.json");
            if let Some(command) = command
                && command != "campaign"
            {
                return Err(format!(
                    "unknown command {command:?}; use campaign, replay, or minimize"
                ));
            }
            while let Some(flag) = args.next() {
                match flag.as_str() {
                    "--seed" => {
                        seed = args
                            .next()
                            .ok_or("--seed requires a value")?
                            .parse()
                            .map_err(|e| format!("invalid seed: {e}"))?
                    }
                    "--iterations" => {
                        iterations = args
                            .next()
                            .ok_or("--iterations requires a value")?
                            .parse()
                            .map_err(|e| format!("invalid iteration count: {e}"))?
                    }
                    "--max-steps" => {
                        limits.max_steps = args
                            .next()
                            .ok_or("--max-steps requires a value")?
                            .parse()
                            .map_err(|e| format!("invalid step limit: {e}"))?;
                    }
                    "--failure-path" => {
                        failure_path =
                            PathBuf::from(args.next().ok_or("--failure-path requires a path")?)
                    }
                    _ => return Err(format!("unknown campaign option {flag:?}")),
                }
            }
            let result = campaign_with_limits(seed, iterations, limits)?;
            println!(
                "campaign seed={seed} iterations={iterations} cases={} distinct_shapes={} enumerated_candidates={} executed_plans={} unsupported_plans={} execution_errors={} candidate_errors={} dispatches={} steps={}",
                result.cases,
                result.distinct_shapes,
                result.enumerated_candidates,
                result.executed_plans,
                result.unsupported_plans,
                result.execution_errors,
                result.candidate_errors,
                result.dispatches,
                result.steps
            );
            if let Some(failure) = result.first_failure {
                write_record(&failure_path, &failure)?;
                eprintln!(
                    "failure kind={} candidate={} detail={}",
                    failure.kind, failure.candidate, failure.detail
                );
                eprintln!(
                    "replay: cargo run --manifest-path fusor/Cargo.toml -p fusor-sim -- replay {}",
                    failure_path.display()
                );
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

fn read_record(path: &str) -> Result<FailureRecord, String> {
    let bytes = fs::read(path).map_err(|error| format!("read {path}: {error}"))?;
    serde_json::from_slice(&bytes).map_err(|error| format!("parse {path}: {error}"))
}

fn write_record(path: &Path, record: &FailureRecord) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let json =
        serde_json::to_vec_pretty(record).map_err(|error| format!("serialize failure: {error}"))?;
    fs::write(path, json).map_err(|error| format!("write {}: {error}", path.display()))
}
