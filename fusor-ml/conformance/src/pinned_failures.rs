//! Pinned expected failures.
//!
//! The conformance baseline carries five known-red tests; everything else
//! must stay green. Each meta-test here re-runs one pinned suite on its own
//! thread and asserts the failure keeps its measured shape, so a silently
//! passing suite or a changed panic/mismatch message fails fast instead of
//! hiding a behavior change behind an already-red test.

use crate::bench::BenchmarkConfig;
use crate::suite::registry::gpu_test_guard;
use crate::suite::webgpu::run_webgpu_kernel_suite;
use fusor::Device;

/// The `structural_memo` lookup panic shared by the three pinned quantized
/// failures.
const STRUCTURAL_MEMO_PANIC: &str = "no entry found for key";
/// First failing variant of the nary binding-limit suite, and its exact
/// mismatch; the pinned webgpu-suite failure is the same case reached through
/// `run_webgpu_kernel_suite`.
const BINDING_LIMIT_VARIANT: &str = "fusion_behavior::gpu_nary_fusion_respects_binding_limit::correctness::run0::device0_gpu::subgroups_cold_pool";
const BINDING_LIMIT_MISMATCH: &str = "at [0, 1]: expected 108.5, got 108.50001";

enum Outcome {
    NoGpu,
    Passed,
    Failed(String),
    Panicked(String),
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|message| message.to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

/// Run one pinned suite on a fresh thread so an expected panic is captured
/// as an [`Outcome`] instead of failing this test.
fn run_to_outcome(run: impl FnOnce() -> Outcome + Send + 'static) -> Outcome {
    match std::thread::spawn(run).join() {
        Ok(outcome) => outcome,
        Err(payload) => Outcome::Panicked(panic_message(payload)),
    }
}

fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime should build")
        .block_on(future)
}

fn registry_suite_outcome(suite: &'static str) -> Outcome {
    run_to_outcome(move || {
        block_on(async {
            if Device::gpu().await.is_err() {
                return Outcome::NoGpu;
            }
            let assertions = crate::suite::registry::assertions_for_suite(suite)
                .expect("registered conformance suite should exist");
            match crate::suite::registry::run_cases(assertions, |_| {}).await {
                Ok(()) => Outcome::Passed,
                Err(err) => Outcome::Failed(err.to_string()),
            }
        })
    })
}

fn bench_suite_outcome(suite: &'static str) -> Outcome {
    run_to_outcome(move || {
        block_on(async {
            let Ok(device) = Device::gpu().await else {
                return Outcome::NoGpu;
            };
            let cases = crate::bench::registry::cases_for_suite(suite)
                .expect("registered benchmark suite should exist");
            match crate::bench::registry::run_cases(
                &device,
                BenchmarkConfig::smoke(),
                cases,
                |_| {},
            )
            .await
            {
                Ok(_) => Outcome::Passed,
                Err(err) => Outcome::Failed(err.to_string()),
            }
        })
    })
}

fn webgpu_suite_outcome() -> Outcome {
    run_to_outcome(|| {
        block_on(async {
            let Ok(device) = Device::gpu().await else {
                return Outcome::NoGpu;
            };
            match run_webgpu_kernel_suite(&device).await {
                Ok(()) => Outcome::Passed,
                Err(err) => Outcome::Failed(err.to_string()),
            }
        })
    })
}

#[track_caller]
fn expect_panic_containing(suite: &str, outcome: Outcome, substring: &str) {
    match outcome {
        Outcome::NoGpu => {}
        Outcome::Panicked(message) => assert!(
            message.contains(substring),
            "pinned-red {suite} panic message changed: {message}"
        ),
        Outcome::Passed => panic!("pinned-red {suite} now passes"),
        Outcome::Failed(message) => {
            panic!("pinned-red {suite} now fails without panicking: {message}")
        }
    }
}

#[track_caller]
fn expect_failure_containing(suite: &str, outcome: Outcome, substrings: &[&str]) {
    match outcome {
        Outcome::NoGpu => {}
        Outcome::Failed(message) => {
            for substring in substrings {
                assert!(
                    message.contains(substring),
                    "pinned-red {suite} failure message changed: {message}"
                );
            }
        }
        Outcome::Passed => panic!("pinned-red {suite} now passes"),
        Outcome::Panicked(message) => {
            panic!("pinned-red {suite} now panics instead of failing: {message}")
        }
    }
}

#[test]
fn pinned_q4k_paired_silu_bench_panics_in_structural_memo() {
    let _gpu_guard = gpu_test_guard();
    expect_panic_containing(
        "webgpu::q4k_paired_silu",
        bench_suite_outcome("webgpu::q4k_paired_silu"),
        STRUCTURAL_MEMO_PANIC,
    );
}

#[test]
fn pinned_nary_binding_limit_fails_with_item_mismatch() {
    let _gpu_guard = gpu_test_guard();
    expect_failure_containing(
        "fusion_behavior::gpu_nary_fusion_respects_binding_limit",
        registry_suite_outcome("fusion_behavior::gpu_nary_fusion_respects_binding_limit"),
        &[BINDING_LIMIT_VARIANT, BINDING_LIMIT_MISMATCH],
    );
}

#[test]
fn pinned_q4k_concat_split_gated_panics_in_structural_memo() {
    let _gpu_guard = gpu_test_guard();
    expect_panic_containing(
        "quantized_matmul_paired::q4k_concat_split_gated_natural_form_matches_cpu_reference",
        registry_suite_outcome(
            "quantized_matmul_paired::q4k_concat_split_gated_natural_form_matches_cpu_reference",
        ),
        STRUCTURAL_MEMO_PANIC,
    );
}

#[test]
fn pinned_q8_0_qmatmul_epilogue_panics_in_structural_memo() {
    let _gpu_guard = gpu_test_guard();
    expect_panic_containing(
        "quantized_matmul_fusion::q8_0_qmatmul_epilogue_tests",
        registry_suite_outcome("quantized_matmul_fusion::q8_0_qmatmul_epilogue_tests"),
        STRUCTURAL_MEMO_PANIC,
    );
}

#[test]
fn pinned_webgpu_suite_fails_on_binding_limit_case() {
    let _gpu_guard = gpu_test_guard();
    expect_failure_containing(
        "webgpu_kernel_suite",
        webgpu_suite_outcome(),
        &[BINDING_LIMIT_VARIANT, BINDING_LIMIT_MISMATCH],
    );
}
