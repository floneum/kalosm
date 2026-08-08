//! The conformance binary. `--exhaustive` sweeps the full schedule domain
//! rather than the shipped move budget; the default run is the op x backward
//! matrix plus the launch-count and golden asserts.
//!
//! Exit status is the failure count clamped to 1, so a CI step is a plain
//! `cargo run -p fusor2-conformance`. A **skip** is never a pass: a device
//! that cannot run a row is reported on its own line and counted separately,
//! so a suite that skipped its whole f16 matrix cannot read as one that ran
//! it.

use std::process::ExitCode;

use fusor2_conformance::exhaustive;
use fusor2_conformance::harness::{self, Outcome, sessions};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let exhaustive_requested = exhaustive::requested(&args);
    // Everything after the flags is a case-name substring filter.
    let filter = args.iter().find(|a| !a.starts_with("--")).cloned();

    let devices = sessions();
    if devices.is_empty() {
        println!("no device available: every case will be reported as skipped");
    } else {
        let names: Vec<&str> = devices.iter().map(|s| s.device().name()).collect();
        println!("backends: {}", names.join(", "));
    }
    println!("exhaustive: {exhaustive_requested}");

    let reports = harness::run_filtered(filter.as_deref(), |_| {});
    let failures = harness::summarize(&reports);

    // A run that executed nothing is not a green run.
    let ran = reports.iter().filter(|r| !matches!(r.outcome, Outcome::Skipped(_))).count();
    if ran == 0 {
        println!("nothing ran; refusing to report success");
        return ExitCode::FAILURE;
    }
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
