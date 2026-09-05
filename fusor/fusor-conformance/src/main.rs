//! The conformance binary: the op x backward matrix, fuzzed over shapes and
//! over every candidate kernel the compiler can emit.
//!
//! Exit status is the failure count clamped to 1, so a CI step is a plain
//! `cargo run -p fusor-conformance`. A **skip** is never a pass: a device
//! that cannot run a row is reported on its own line and counted separately,
//! so a suite that skipped its whole f16 matrix cannot read as one that ran
//! it.

use std::process::ExitCode;

use fusor_conformance::harness::{self, Outcome, sessions};

fn main() -> ExitCode {
    // Race every class member of every launch, value-checking each against
    // the selected plan, so a case covers the *class* rather than whichever
    // member extraction happened to pick.
    //
    // SAFETY: set before any thread reads the environment.
    unsafe { std::env::set_var("FUSOR_VERIFY_MEMBERS", "1") };
    let args: Vec<String> = std::env::args().skip(1).collect();
    // Everything after the flags is a case-name substring filter.
    let filter = args.iter().find(|a| !a.starts_with("--")).cloned();

    let devices = sessions();
    if devices.is_empty() {
        println!("no device available: every case will be reported as skipped");
    } else {
        let names: Vec<&str> = devices.iter().map(|s| s.device().name()).collect();
        println!("backends: {}", names.join(", "));
    }

    // `FUSOR_CONFORMANCE_PROGRESS` streams each result to stderr as it lands,
    // for watching a run that is slow or growing rather than reading its
    // summary afterwards. The summary below is unchanged.
    let progress = std::env::var_os("FUSOR_CONFORMANCE_PROGRESS").is_some();
    let reports = match &filter {
        Some(f) => harness::Harness::with_filter(f.clone()).run(),
        None => harness::run_all(|r| {
            if progress {
                eprintln!("[progress] {} [{}]", r.case, r.backend);
            }
        }),
    };
    let failures = harness::summarize(&reports);

    // A run that executed nothing is not a green run.
    let ran = reports
        .iter()
        .filter(|r| !matches!(r.outcome, Outcome::Skipped(_)))
        .count();
    if ran == 0 {
        println!("nothing ran; refusing to report success");
        return ExitCode::FAILURE;
    }
    let wrong = fusor::session::wrong_member_count();
    if wrong > 0 {
        println!(
            "{wrong} class member(s) computed wrong values under the member sweep; \
             every one is a live miscompile extraction could select"
        );
        return ExitCode::FAILURE;
    }
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
