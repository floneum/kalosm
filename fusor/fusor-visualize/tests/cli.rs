use std::process::{Command, Output, Stdio};

fn run(args: &[&str]) -> Output {
    // Nextest remaps archived binaries to the test runner's target directory.
    let executable = std::env::var_os("NEXTEST_BIN_EXE_fusor_visualize")
        .or_else(|| std::env::var_os("CARGO_BIN_EXE_fusor-visualize"))
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_fusor-visualize").into());
    Command::new(executable)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

#[test]
fn json_output_has_config_steps_and_an_honest_prefix_summary() {
    let output = run(&["--demo", "strided", "--format", "json", "--max-steps", "1"]);
    assert!(output.status.success());
    let records: Vec<serde_json::Value> = std::str::from_utf8(&output.stdout)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    assert_eq!(records.len(), 3);
    assert_eq!(records[0]["type"], "config");
    assert_eq!(records[1]["step"]["metrics"]["transactions"], 32);
    assert_eq!(records[2]["complete"], false);
    let output = run(&[
        "--demo",
        "coalesced",
        "--format",
        "json",
        "--max-steps",
        "3",
    ]);
    let last: serde_json::Value = serde_json::from_str(
        std::str::from_utf8(&output.stdout)
            .unwrap()
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(last["complete"], true); // limit exactly equals trace length
}

#[test]
fn text_reports_action_and_model_metrics() {
    let output = run(&["--demo", "strided", "--format", "text", "--max-steps", "1"]);
    assert!(output.status.success());
    let output = String::from_utf8(output.stdout).unwrap();
    assert!(output.contains("STEP 0001"));
    assert!(output.contains("sectors 32 / packed lower bound 4"));
    assert!(output.contains("totals are a prefix"));
}

#[test]
fn reuse_advice_only_mentions_cache_hits_when_they_occur() {
    let without_cache = run(&[
        "--demo",
        "reuse",
        "--subgroup-size",
        "16",
        "--cache-bytes",
        "0",
        "--format",
        "text",
        "--max-steps",
        "2",
    ]);
    assert!(without_cache.status.success());
    let without_cache = String::from_utf8(without_cache.stdout).unwrap();
    assert!(without_cache.contains("cache line hits 0 / misses 1"));
    assert!(without_cache.contains("Other subgroups read these bytes: consider workgroup staging"));
    assert!(!without_cache.contains("cache hits already help"));

    let with_cache = run(&[
        "--demo",
        "reuse",
        "--subgroup-size",
        "16",
        "--cache-bytes",
        "128",
        "--format",
        "text",
        "--max-steps",
        "2",
    ]);
    assert!(with_cache.status.success());
    let with_cache = String::from_utf8(with_cache.stdout).unwrap();
    assert!(with_cache.contains("cache line hits 1 / misses 0"));
    assert!(with_cache.contains("cache hits already help"));
}

#[test]
fn invalid_options_and_non_terminal_tui_fail_cleanly() {
    for args in [
        vec!["--subgroup-size", "0"],
        vec!["--demo", "typo"],
        vec!["--format", "tui"],
        vec!["--trace", "unused", "--demo", "strided"],
        vec!["--format", "tui", "--max-steps", "1"],
    ] {
        let output = run(&args);
        assert!(!output.status.success(), "{args:?}");
        assert!(!output.stderr.is_empty());
        assert!(!String::from_utf8_lossy(&output.stderr).contains("panicked"));
    }
}

#[test]
fn portable_trace_fixture_can_be_replayed_with_another_subgroup_size() {
    let manifest =
        std::env::var_os("CARGO_MANIFEST_DIR").unwrap_or_else(|| env!("CARGO_MANIFEST_DIR").into());
    let fixture = std::path::PathBuf::from(manifest).join("examples/trace.json");
    let output = run(&[
        "--trace",
        fixture.to_str().unwrap(),
        "--subgroup-size",
        "2",
        "--format",
        "json",
    ]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let last: serde_json::Value = serde_json::from_str(
        std::str::from_utf8(&output.stdout)
            .unwrap()
            .lines()
            .last()
            .unwrap(),
    )
    .unwrap();
    assert_eq!(last["totals"]["steps"], 4);
    assert_eq!(last["totals"]["transactions"], 6);
    assert_eq!(last["complete"], true);
}
