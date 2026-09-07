use dioxus::prelude::*;
use dioxus_router::{Link, Outlet, Routable, Router, use_route};
use fusor::Device;
use fusor_conformance::bench::{
    BenchmarkConfig, BenchmarkEvent, BenchmarkReport, sweep::BenchmarkSweepEvent,
};
use web_time::{Duration, Instant};

mod components;
use components::badge::{Badge, BadgeVariant};
use components::button::{Button, ButtonVariant};
use components::card::{Card, CardContent, CardDescription, CardHeader, CardTitle};
use components::separator::Separator;

const MAX_RENDERED_STEPS: usize = 80;
const DETAIL_SWEEP_CONFIG: BenchmarkConfig = BenchmarkConfig::new(3, 3, 15);

fn main() {
    dioxus::launch(App);
}

#[derive(Clone, Debug, PartialEq, Routable)]
enum Route {
    #[layout(AppShell)]
    #[redirect("/", || Route::Tests {})]
    #[route("/tests")]
    Tests {},
    #[route("/benchmarks")]
    Benchmarks {},
    #[route("/benchmarks/:case")]
    BenchmarkDetail { case: String },
    #[route("/:..route")]
    NotFound { route: Vec<String> },
}

#[component]
fn App() -> Element {
    rsx! {
        Router::<Route> {}
    }
}

#[component]
fn AppShell() -> Element {
    let route = use_route::<Route>();
    let tests_class = if matches!(route, Route::Tests {}) {
        "nav-link active"
    } else {
        "nav-link"
    };
    let benchmarks_class = if matches!(route, Route::Benchmarks {} | Route::BenchmarkDetail { .. })
    {
        "nav-link active"
    } else {
        "nav-link"
    };

    rsx! {
        document::Stylesheet { href: asset!("/assets/dx-components-theme.css") }
        document::Stylesheet { href: asset!("/assets/runner.css") }
        div { class: "shell",
            div { class: "workspace",
                header { class: "sitebar",
                    div { class: "brand",
                        div { class: "brand-mark", "F" }
                        div { class: "brand-text",
                            p { class: "eyebrow", "Fusor WebGPU" }
                            h1 { "Runner Lab" }
                        }
                    }
                    nav { class: "route-tabs",
                        Link {
                            class: tests_class,
                            to: Route::Tests {},
                            "Tests"
                        }
                        Link {
                            class: benchmarks_class,
                            to: Route::Benchmarks {},
                            "Benchmarks"
                        }
                    }
                }
                Outlet::<Route> {}
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StepState {
    Running,
    Passed,
    Failed,
}

impl StepState {
    fn label(self) -> &'static str {
        match self {
            StepState::Running => "running",
            StepState::Passed => "passed",
            StepState::Failed => "failed",
        }
    }

    fn class(self) -> &'static str {
        match self {
            StepState::Running => "running",
            StepState::Passed => "passed",
            StepState::Failed => "failed",
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct Step {
    name: String,
    state: StepState,
}

impl Step {
    fn new(name: impl Into<String>, state: StepState) -> Self {
        Self {
            name: name.into(),
            state,
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct TestProgressState {
    total: usize,
    passed: usize,
    failed: usize,
    current: String,
    has_running_step: bool,
    recent: Vec<Step>,
}

impl Default for TestProgressState {
    fn default() -> Self {
        Self {
            total: 0,
            passed: 0,
            failed: 0,
            current: "waiting".to_string(),
            has_running_step: false,
            recent: Vec::new(),
        }
    }
}

#[derive(Clone, PartialEq)]
enum RunResult {
    Idle,
    Running,
    Passed(Duration),
    Failed(String),
}

impl RunResult {
    fn class(&self) -> &'static str {
        match self {
            RunResult::Idle => "idle",
            RunResult::Running => "running",
            RunResult::Passed(_) => "passed",
            RunResult::Failed(_) => "failed",
        }
    }

    fn word(&self) -> &'static str {
        match self {
            RunResult::Idle => "ready",
            RunResult::Running => "running",
            RunResult::Passed(_) => "passed",
            RunResult::Failed(_) => "failed",
        }
    }
}

fn mark_running_step(progress: &mut TestProgressState, name: impl Into<String>) {
    if progress.has_running_step {
        progress.passed += 1;
    }
    if let Some(step) = progress.recent.last_mut()
        && step.state == StepState::Running
    {
        step.state = StepState::Passed;
    }
    let name = name.into();
    progress.total += 1;
    progress.current = name.clone();
    progress.has_running_step = true;
    progress.recent.push(Step::new(name, StepState::Running));
    if progress.recent.len() > MAX_RENDERED_STEPS {
        progress.recent.remove(0);
    }
}

fn finish_running_step(progress: &mut TestProgressState, state: StepState) {
    if !progress.has_running_step {
        return;
    }
    match state {
        StepState::Running => {}
        StepState::Passed => progress.passed += 1,
        StepState::Failed => progress.failed += 1,
    }
    if let Some(step) = progress.recent.last_mut()
        && step.state == StepState::Running
    {
        step.state = state;
    }
    progress.has_running_step = false;
    if state == StepState::Passed {
        progress.current = "waiting".to_string();
    }
}

async fn run_test_suite(mut progress: Signal<TestProgressState>, mut result: Signal<RunResult>) {
    use fusor_conformance::harness::{Harness, Outcome, sessions_async};

    result.set(RunResult::Running);
    let started = Instant::now();
    {
        let mut progress = progress.write();
        mark_running_step(&mut progress, "webgpu_device");
    }

    let sessions = sessions_async().await;
    if sessions.is_empty() {
        {
            let mut progress = progress.write();
            finish_running_step(&mut progress, StepState::Failed);
        }
        result.set(RunResult::Failed(
            "no WebGPU device is available".to_string(),
        ));
        return;
    }
    {
        let mut progress = progress.write();
        finish_running_step(&mut progress, StepState::Passed);
    }

    // Every case of the registry, on the one device a browser has, each
    // report landing in the log as it completes.
    let mut first_failure: Option<String> = None;
    let mut failures = 0usize;
    let reports = Harness::new()
        .run_async(&sessions, |report| {
            let mut progress = progress.write();
            mark_running_step(&mut progress, report.case.clone());
            let state = match &report.outcome {
                Outcome::Pass | Outcome::Skipped(_) => StepState::Passed,
                Outcome::Fail(message) => {
                    failures += 1;
                    if first_failure.is_none() {
                        first_failure = Some(format!("{}: {message}", report.case));
                    }
                    StepState::Failed
                }
            };
            finish_running_step(&mut progress, state);
        })
        .await;

    if failures == 0 {
        result.set(RunResult::Passed(started.elapsed()));
    } else {
        result.set(RunResult::Failed(format!(
            "{failures} of {} cases failed; first: {}",
            reports.len(),
            first_failure.unwrap_or_default()
        )));
    }
}

#[component]
fn Tests() -> Element {
    let progress = use_signal(TestProgressState::default);
    let result = use_signal(|| RunResult::Idle);
    let mut run_id = use_signal(|| 0usize);

    let _runner = use_resource(move || async move {
        let _ = run_id();
        run_test_suite(progress, result).await;
    });

    let progress_snapshot = progress.read().clone();
    let total = progress_snapshot.total;
    let passed = progress_snapshot.passed;
    let failed = progress_snapshot.failed;
    let current = progress_snapshot.current.as_str();
    let is_running = matches!(&*result.read(), RunResult::Running);

    let (summary_title, summary_detail) = match &*result.read() {
        RunResult::Idle => ("Ready".to_string(), "WebGPU adapter pending".to_string()),
        RunResult::Running => ("Running conformance".to_string(), current.to_string()),
        RunResult::Passed(duration) => (
            "Conformance passed".to_string(),
            format!("Completed in {:.2}s", duration.as_secs_f64()),
        ),
        RunResult::Failed(error) => ("Conformance failed".to_string(), error.clone()),
    };
    let result_class = result.read().class();
    let status_word = result.read().word();

    rsx! {
        header { class: "topbar",
            div {
                p { class: "eyebrow", "Conformance" }
                h2 { "WebGPU Tests" }
            }
            Button {
                variant: ButtonVariant::Primary,
                disabled: is_running,
                onclick: move |_| {
                    if !matches!(&*result.read(), RunResult::Running) {
                        run_id.with_mut(|id| *id += 1);
                    }
                },
                if is_running { "Running…" } else { "Run tests" }
            }
        }

        Card { class: "summary {result_class}",
            CardHeader {
                div { class: "summary-head",
                    CardTitle { "{summary_title}" }
                    Badge {
                        variant: BadgeVariant::Outline,
                        class: "status-badge {result_class}",
                        "{status_word}"
                    }
                }
                CardDescription { "{summary_detail}" }
            }
            CardContent {
                div { class: "summary-body",
                    div { class: "metrics",
                        div { class: "metric",
                            span { class: "metric-value", "{total}" }
                            span { class: "metric-label", "seen" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{passed}" }
                            span { class: "metric-label", "passed" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{failed}" }
                            span { class: "metric-label", "failed" }
                        }
                    }
                }
            }
        }

        Card { class: "log-panel",
            CardHeader {
                div { class: "log-head",
                    CardTitle { "Case progress" }
                    span { class: "current-case", "{current}" }
                }
            }
            Separator { horizontal: true, decorative: true }
            div { class: "steps",
                if progress_snapshot.recent.is_empty() {
                    div { class: "empty", "Waiting for the runner to start." }
                } else {
                    for (index , step) in progress_snapshot.recent.iter().enumerate() {
                        div { class: "step", key: "{index}-{step.name}",
                            span { class: "step-name", "{step.name}" }
                            Badge {
                                variant: BadgeVariant::Outline,
                                class: "status-badge {step.state.class()}",
                                "{step.state.label()}"
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BenchState {
    Running,
    Passed,
    Failed,
}

impl BenchState {
    fn label(self) -> &'static str {
        match self {
            BenchState::Running => "running",
            BenchState::Passed => "done",
            BenchState::Failed => "failed",
        }
    }

    fn class(self) -> &'static str {
        match self {
            BenchState::Running => "running",
            BenchState::Passed => "passed",
            BenchState::Failed => "failed",
        }
    }
}

#[derive(Clone, PartialEq)]
struct BenchRow {
    name: String,
    state: BenchState,
    detail: String,
    iterations: usize,
    samples: usize,
    mean_ms: Option<f64>,
    median_ms: Option<f64>,
    stddev_ms: Option<f64>,
    min_ms: Option<f64>,
    max_ms: Option<f64>,
    total_ms: Option<f64>,
}

impl BenchRow {
    fn running(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            state: BenchState::Running,
            detail: String::new(),
            iterations: 0,
            samples: 0,
            mean_ms: None,
            median_ms: None,
            stddev_ms: None,
            min_ms: None,
            max_ms: None,
            total_ms: None,
        }
    }

    fn finished(report: BenchmarkReport) -> Self {
        Self {
            name: report.name,
            state: BenchState::Passed,
            detail: report.detail,
            iterations: report.iterations,
            samples: report.samples,
            mean_ms: Some(report.mean_ms),
            median_ms: Some(report.median_ms),
            stddev_ms: Some(report.stddev_ms),
            min_ms: Some(report.min_ms),
            max_ms: Some(report.max_ms),
            total_ms: Some(report.total_ms),
        }
    }
}

#[derive(Clone, PartialEq)]
struct BenchProgressState {
    total: usize,
    completed: usize,
    failed: usize,
    current: String,
    rows: Vec<BenchRow>,
}

impl Default for BenchProgressState {
    fn default() -> Self {
        Self {
            total: 0,
            completed: 0,
            failed: 0,
            current: "waiting".to_string(),
            rows: Vec::new(),
        }
    }
}

fn format_ms(value: Option<f64>) -> String {
    let Some(value) = value else {
        return "-".to_string();
    };
    if value < 1.0 {
        format!("{value:.3}")
    } else if value < 100.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.1}")
    }
}

fn suite_and_case(name: &str) -> Option<(&str, &str)> {
    name.split_once("::")
}

fn benchmark_case_slug(name: &str) -> String {
    suite_and_case(name)
        .map(|(_, case)| case.split_once('@').map_or(case, |(case, _)| case))
        .unwrap_or(name)
        .to_string()
}

/// When a `burn` baseline exists for this non-burn case, returns
/// `(burn_mean_ms, our_mean_ms)` so callers can format and color the comparison.
fn burn_baseline_pair(row: &BenchRow, rows: &[BenchRow]) -> Option<(f64, f64)> {
    let (suite, case) = suite_and_case(&row.name)?;
    if suite == "burn" {
        return None;
    }
    let mean_ms = row.mean_ms?;
    let burn_mean_ms = rows
        .iter()
        .find(|candidate| {
            candidate.state == BenchState::Passed
                && suite_and_case(&candidate.name) == Some(("burn", case))
        })
        .and_then(|candidate| candidate.mean_ms)?;
    if mean_ms <= 0.0 || burn_mean_ms <= 0.0 {
        return None;
    }
    Some((burn_mean_ms, mean_ms))
}

fn format_burn_comparison(row: &BenchRow, rows: &[BenchRow]) -> String {
    if matches!(suite_and_case(&row.name), Some(("burn", _))) {
        return "baseline".to_string();
    }
    match burn_baseline_pair(row, rows) {
        Some((burn_mean_ms, mean_ms)) if mean_ms <= burn_mean_ms => {
            format!("{:.2}x faster", burn_mean_ms / mean_ms)
        }
        Some((burn_mean_ms, mean_ms)) => format!("{:.2}x slower", mean_ms / burn_mean_ms),
        None => "-".to_string(),
    }
}

fn burn_comparison_class(row: &BenchRow, rows: &[BenchRow]) -> &'static str {
    match burn_baseline_pair(row, rows) {
        Some((burn_mean_ms, mean_ms)) if mean_ms <= burn_mean_ms => "cmp-faster",
        Some(_) => "cmp-slower",
        None => "",
    }
}

fn apply_benchmark_event(progress: &mut BenchProgressState, event: BenchmarkEvent) {
    match event {
        BenchmarkEvent::Started(name) => {
            progress.total += 1;
            progress.current = name.clone();
            progress.rows.push(BenchRow::running(name));
        }
        BenchmarkEvent::Finished(report) => {
            progress.completed += 1;
            progress.current = "waiting".to_string();
            let row = BenchRow::finished(report);
            if let Some(existing) =
                progress.rows.iter_mut().rev().find(|existing| {
                    existing.name == row.name && existing.state == BenchState::Running
                })
            {
                *existing = row;
            } else {
                progress.rows.push(row);
            }
        }
    }
}

fn mark_benchmark_failure(progress: &mut BenchProgressState) {
    progress.failed += 1;
    if let Some(row) = progress
        .rows
        .iter_mut()
        .rev()
        .find(|row| row.state == BenchState::Running)
    {
        row.state = BenchState::Failed;
    }
}

async fn run_benchmark_suite(
    mut progress: Signal<BenchProgressState>,
    mut result: Signal<RunResult>,
) {
    progress.set(BenchProgressState::default());
    result.set(RunResult::Running);
    let started = Instant::now();
    let config = BenchmarkConfig::default();

    let device = match Device::gpu().await {
        Ok(device) => device,
        Err(err) => {
            result.set(RunResult::Failed(err.to_string()));
            return;
        }
    };

    let cases = fusor_conformance::bench::registry::cases();
    let suite_result =
        fusor_conformance::bench::registry::run_cases(&device, config, cases, |event| {
            let mut progress = progress.write();
            apply_benchmark_event(&mut progress, event);
        })
        .await;

    match suite_result {
        Ok(_) => {}
        Err(err) => {
            {
                let mut progress = progress.write();
                mark_benchmark_failure(&mut progress);
            }
            result.set(RunResult::Failed(err.to_string()));
            return;
        }
    }

    result.set(RunResult::Passed(started.elapsed()));
}

#[component]
fn Benchmarks() -> Element {
    let progress = use_signal(BenchProgressState::default);
    let result = use_signal(|| RunResult::Idle);
    let mut run_id = use_signal(|| 0usize);

    let _runner = use_resource(move || async move {
        let _ = run_id();
        run_benchmark_suite(progress, result).await;
    });

    let progress_snapshot = progress.read().clone();
    let rows = progress_snapshot.rows.clone();
    let current = progress_snapshot.current.as_str();
    let is_running = matches!(&*result.read(), RunResult::Running);
    let measured_rows = rows
        .iter()
        .filter(|row| row.state == BenchState::Passed)
        .collect::<Vec<_>>();
    let total_measured_ms = measured_rows
        .iter()
        .filter_map(|row| row.total_ms)
        .sum::<f64>();
    let avg_mean_ms = if measured_rows.is_empty() {
        None
    } else {
        Some(
            measured_rows
                .iter()
                .filter_map(|row| row.mean_ms)
                .sum::<f64>()
                / measured_rows.len() as f64,
        )
    };
    let paired_cases = rows
        .iter()
        .filter(|row| {
            matches!(suite_and_case(&row.name), Some(("webgpu", case)) if rows
                .iter()
                .any(|candidate| suite_and_case(&candidate.name) == Some(("burn", case))))
        })
        .count();

    let (summary_title, summary_detail) = match &*result.read() {
        RunResult::Idle => ("Ready".to_string(), "WebGPU adapter pending".to_string()),
        RunResult::Running => ("Running benchmarks".to_string(), current.to_string()),
        RunResult::Passed(duration) => (
            "Benchmarks complete".to_string(),
            format!("Completed in {:.2}s", duration.as_secs_f64()),
        ),
        RunResult::Failed(error) => ("Benchmark failed".to_string(), error.clone()),
    };
    let result_class = result.read().class();
    let status_word = result.read().word();

    rsx! {
        header { class: "topbar",
            div {
                p { class: "eyebrow", "Performance" }
                h2 { "WebGPU Benchmarks" }
            }
            Button {
                variant: ButtonVariant::Primary,
                disabled: is_running,
                onclick: move |_| {
                    if !matches!(&*result.read(), RunResult::Running) {
                        run_id.with_mut(|id| *id += 1);
                    }
                },
                if is_running { "Running…" } else { "Run benchmarks" }
            }
        }

        Card { class: "summary {result_class}",
            CardHeader {
                div { class: "summary-head",
                    CardTitle { "{summary_title}" }
                    Badge {
                        variant: BadgeVariant::Outline,
                        class: "status-badge {result_class}",
                        "{status_word}"
                    }
                }
                CardDescription { "{summary_detail}" }
            }
            CardContent {
                div { class: "summary-body",
                    div { class: "metrics",
                        div { class: "metric",
                            span { class: "metric-value", "{progress_snapshot.total}" }
                            span { class: "metric-label", "seen" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{progress_snapshot.completed}" }
                            span { class: "metric-label", "done" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{paired_cases}" }
                            span { class: "metric-label", "pairs" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{format_ms(avg_mean_ms)}" }
                            span { class: "metric-label", "avg ms" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{format_ms(Some(total_measured_ms))}" }
                            span { class: "metric-label", "total ms" }
                        }
                    }
                }
            }
        }

        Card { class: "log-panel bench-panel",
            CardHeader {
                div { class: "log-head",
                    CardTitle { "Benchmark results" }
                    span { class: "current-case", "{current}" }
                }
            }
            Separator { horizontal: true, decorative: true }
            div { class: "bench-table",
                div { class: "bench-row bench-head",
                    span { "case" }
                    span { "mean ms" }
                    span { "median" }
                    span { "stddev" }
                    span { "range" }
                    span { "samples" }
                    span { "vs burn" }
                    span { "status" }
                }
                if rows.is_empty() {
                    div { class: "empty", "Waiting for benchmarks to start." }
                } else {
                    for (index, row) in rows.iter().enumerate() {
                        Link {
                            class: "bench-row benchmark-link",
                            key: "{index}-{row.name}",
                            to: Route::BenchmarkDetail { case: benchmark_case_slug(&row.name) },
                            div { class: "bench-case",
                                span { class: "step-name", "{row.name}" }
                                span { class: "bench-detail", "{row.detail}" }
                            }
                            span { class: "bench-number", "{format_ms(row.mean_ms)}" }
                            span { class: "bench-number", "{format_ms(row.median_ms)}" }
                            span { class: "bench-number", "{format_ms(row.stddev_ms)}" }
                            span { class: "bench-number", "{format_ms(row.min_ms)}-{format_ms(row.max_ms)}" }
                            span { class: "bench-number", "{row.samples}x{row.iterations}" }
                            span {
                                class: "bench-number {burn_comparison_class(row, &rows)}",
                                "{format_burn_comparison(row, &rows)}"
                            }
                            Badge {
                                variant: BadgeVariant::Outline,
                                class: "status-badge {row.state.class()}",
                                "{row.state.label()}"
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, PartialEq)]
struct SweepRow {
    label: String,
    value: usize,
    burn: Option<BenchmarkReport>,
    webgpu: Option<BenchmarkReport>,
    running_suite: Option<String>,
}

#[derive(Clone, PartialEq)]
struct SweepProgressState {
    current: String,
    rows: Vec<SweepRow>,
}

impl Default for SweepProgressState {
    fn default() -> Self {
        Self {
            current: "waiting".to_string(),
            rows: Vec::new(),
        }
    }
}

impl SweepProgressState {
    fn from_descriptor(
        descriptor: fusor_conformance::bench::sweep::BenchmarkSweepDescriptor,
    ) -> Self {
        Self {
            current: "waiting".to_string(),
            rows: descriptor
                .sizes
                .iter()
                .map(|size| SweepRow {
                    label: size.label.to_string(),
                    value: size.value,
                    burn: None,
                    webgpu: None,
                    running_suite: None,
                })
                .collect(),
        }
    }
}

fn apply_sweep_event(progress: &mut SweepProgressState, event: BenchmarkSweepEvent) {
    match event {
        BenchmarkSweepEvent::Started { suite, label } => {
            progress.current = format!("{suite}::{label}");
            if let Some(row) = progress.rows.iter_mut().find(|row| row.label == label) {
                row.running_suite = Some(suite.to_string());
            }
        }
        BenchmarkSweepEvent::Finished {
            suite,
            label,
            report,
        } => {
            progress.current = "waiting".to_string();
            if let Some(row) = progress.rows.iter_mut().find(|row| row.label == label) {
                match suite {
                    "burn" => row.burn = Some(report),
                    "webgpu" => row.webgpu = Some(report),
                    _ => {}
                }
                if row.running_suite.as_deref() == Some(suite) {
                    row.running_suite = None;
                }
            }
        }
    }
}

fn mark_sweep_failure(progress: &mut SweepProgressState) {
    for row in &mut progress.rows {
        row.running_suite = None;
    }
    progress.current = "failed".to_string();
}

async fn run_benchmark_sweep(
    case: String,
    mut progress: Signal<SweepProgressState>,
    mut result: Signal<RunResult>,
) {
    let Some(descriptor) = fusor_conformance::bench::sweep::descriptor(&case) else {
        result.set(RunResult::Failed(format!("unknown benchmark: {case}")));
        return;
    };

    progress.set(SweepProgressState::from_descriptor(descriptor));
    result.set(RunResult::Running);
    let started = Instant::now();

    let device = match Device::gpu().await {
        Ok(device) => device,
        Err(err) => {
            result.set(RunResult::Failed(err.to_string()));
            return;
        }
    };

    let suite_result =
        fusor_conformance::bench::sweep::run_sweep(&case, &device, DETAIL_SWEEP_CONFIG, |event| {
            let mut progress = progress.write();
            apply_sweep_event(&mut progress, event);
        })
        .await;

    match suite_result {
        Ok(_) => result.set(RunResult::Passed(started.elapsed())),
        Err(err) => {
            {
                let mut progress = progress.write();
                mark_sweep_failure(&mut progress);
            }
            result.set(RunResult::Failed(err.to_string()));
        }
    }
}

fn report_median(report: &Option<BenchmarkReport>) -> Option<f64> {
    report.as_ref().map(|report| report.median_ms)
}

fn report_sample_count(report: &Option<BenchmarkReport>) -> Option<usize> {
    report.as_ref().map(|report| report.samples)
}

fn sweep_sample_count(row: &SweepRow) -> String {
    report_sample_count(&row.webgpu)
        .or_else(|| report_sample_count(&row.burn))
        .map(|samples| samples.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn sweep_max_ms(rows: &[SweepRow]) -> f64 {
    rows.iter()
        .flat_map(|row| [report_median(&row.burn), report_median(&row.webgpu)])
        .flatten()
        .fold(0.0, f64::max)
        .max(1.0)
}

fn bar_style(value: Option<f64>, max_ms: f64) -> String {
    let height = value
        .map(|value| (value / max_ms * 100.0).clamp(3.0, 100.0))
        .unwrap_or(0.0);
    format!("height: {height:.2}%;")
}

fn sweep_ratio(row: &SweepRow) -> Option<(f64, &'static str)> {
    let burn = row.burn.as_ref()?.median_ms;
    let webgpu = row.webgpu.as_ref()?.median_ms;
    if burn <= 0.0 || webgpu <= 0.0 {
        return None;
    }
    if webgpu <= burn {
        Some((burn / webgpu, "faster"))
    } else {
        Some((webgpu / burn, "slower"))
    }
}

fn format_sweep_ratio(row: &SweepRow) -> String {
    match sweep_ratio(row) {
        Some((ratio, direction)) => format!("{ratio:.2}x {direction}"),
        None => "-".to_string(),
    }
}

fn sweep_ratio_class(row: &SweepRow) -> &'static str {
    match sweep_ratio(row) {
        Some((_, "faster")) => "cmp-faster",
        Some(_) => "cmp-slower",
        None => "",
    }
}

fn sweep_status(row: &SweepRow) -> (&'static str, &'static str) {
    if row.running_suite.is_some() {
        ("running", "running")
    } else if row.burn.is_some() && row.webgpu.is_some() {
        ("passed", "done")
    } else {
        ("idle", "queued")
    }
}

fn best_speedup(rows: &[SweepRow]) -> Option<f64> {
    rows.iter()
        .filter_map(|row| match sweep_ratio(row) {
            Some((ratio, "faster")) => Some(ratio),
            _ => None,
        })
        .max_by(f64::total_cmp)
}

#[component]
fn BenchmarkDetail(case: String) -> Element {
    let Some(descriptor) = fusor_conformance::bench::sweep::descriptor(&case) else {
        return rsx! {
            Card { class: "summary failed",
                CardHeader {
                    div { class: "summary-head",
                        CardTitle { "Benchmark not found" }
                        Badge { variant: BadgeVariant::Outline, class: "status-badge failed", "404" }
                    }
                    CardDescription { "{case}" }
                }
                CardContent {
                    Link { class: "nav-link", to: Route::Benchmarks {}, "Benchmarks" }
                }
            }
        };
    };

    let progress = use_signal(|| SweepProgressState::from_descriptor(descriptor));
    let result = use_signal(|| RunResult::Idle);
    let mut run_id = use_signal(|| 0usize);
    let case_for_runner = case.clone();

    let _runner = use_resource(move || {
        let case = case_for_runner.clone();
        async move {
            let _ = run_id();
            run_benchmark_sweep(case, progress, result).await;
        }
    });

    let progress_snapshot = progress.read().clone();
    let rows = progress_snapshot.rows.clone();
    let current = progress_snapshot.current.as_str();
    let is_running = matches!(&*result.read(), RunResult::Running);
    let result_class = result.read().class();
    let status_word = result.read().word();
    let completed_pairs = rows
        .iter()
        .filter(|row| row.burn.is_some() && row.webgpu.is_some())
        .count();
    let runs_per_size = DETAIL_SWEEP_CONFIG.samples;
    let max_ms = sweep_max_ms(&rows);
    let best = best_speedup(&rows)
        .map(|value| format!("{value:.2}x"))
        .unwrap_or_else(|| "-".to_string());

    let (summary_title, summary_detail) = match &*result.read() {
        RunResult::Idle => (descriptor.title.to_string(), descriptor.detail.to_string()),
        RunResult::Running => ("Running size sweep".to_string(), current.to_string()),
        RunResult::Passed(duration) => (
            descriptor.title.to_string(),
            format!("Completed in {:.2}s", duration.as_secs_f64()),
        ),
        RunResult::Failed(error) => ("Sweep failed".to_string(), error.clone()),
    };

    rsx! {
        header { class: "topbar",
            div {
                p { class: "eyebrow", "Benchmark detail" }
                h2 { "{descriptor.title}" }
            }
            div { class: "topbar-actions",
                Link { class: "nav-link", to: Route::Benchmarks {}, "Benchmarks" }
                Button {
                    variant: ButtonVariant::Primary,
                    disabled: is_running,
                    onclick: move |_| {
                        if !matches!(&*result.read(), RunResult::Running) {
                            run_id.with_mut(|id| *id += 1);
                        }
                    },
                    if is_running { "Running…" } else { "Run sweep" }
                }
            }
        }

        Card { class: "summary {result_class}",
            CardHeader {
                div { class: "summary-head",
                    CardTitle { "{summary_title}" }
                    Badge {
                        variant: BadgeVariant::Outline,
                        class: "status-badge {result_class}",
                        "{status_word}"
                    }
                }
                CardDescription { "{summary_detail}" }
            }
            CardContent {
                div { class: "summary-body",
                    div { class: "metrics",
                        div { class: "metric",
                            span { class: "metric-value", "{rows.len()}" }
                            span { class: "metric-label", "sizes" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{completed_pairs}" }
                            span { class: "metric-label", "pairs" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{runs_per_size}" }
                            span { class: "metric-label", "runs/size" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{best}" }
                            span { class: "metric-label", "best" }
                        }
                        div { class: "metric",
                            span { class: "metric-value", "{format_ms(Some(max_ms))}" }
                            span { class: "metric-label", "max ms" }
                        }
                    }
                }
            }
        }

        Card { class: "chart-panel",
            CardHeader {
                div { class: "log-head",
                    CardTitle { "Size sweep" }
                    span { class: "current-case", "{case}" }
                }
            }
            Separator { horizontal: true, decorative: true }
            div { class: "chart-legend",
                span { class: "legend-item burn", "Burn" }
                span { class: "legend-item webgpu", "Fusor" }
            }
            div { class: "bar-chart",
                for row in rows.iter() {
                    div { class: "chart-group", key: "{row.label}",
                        div { class: "chart-bars",
                            div {
                                class: "chart-bar burn",
                                style: "{bar_style(report_median(&row.burn), max_ms)}",
                                span { class: "chart-value", "{format_ms(report_median(&row.burn))}" }
                            }
                            div {
                                class: "chart-bar webgpu",
                                style: "{bar_style(report_median(&row.webgpu), max_ms)}",
                                span { class: "chart-value", "{format_ms(report_median(&row.webgpu))}" }
                            }
                        }
                        div { class: "chart-label", "{row.label}" }
                    }
                }
            }
        }

        Card { class: "log-panel bench-panel",
            CardHeader {
                div { class: "log-head",
                    CardTitle { "Sweep data" }
                    span { class: "current-case", "{current}" }
                }
            }
            Separator { horizontal: true, decorative: true }
            div { class: "sweep-table",
                div { class: "sweep-row sweep-head",
                    span { "size" }
                    span { "burn median" }
                    span { "fusor median" }
                    span { "runs" }
                    span { "vs burn" }
                    span { "status" }
                }
                for row in rows.iter() {
                    {
                        let (state_class, state_label) = sweep_status(row);
                        rsx! {
                            div { class: "sweep-row", key: "{row.label}-data",
                                div { class: "bench-case",
                                    span { class: "step-name", "{row.label}" }
                                    span { class: "bench-detail", "{row.value}" }
                                }
                                span { class: "bench-number", "{format_ms(report_median(&row.burn))}" }
                                span { class: "bench-number", "{format_ms(report_median(&row.webgpu))}" }
                                span { class: "bench-number", "{sweep_sample_count(row)}" }
                                span {
                                    class: "bench-number {sweep_ratio_class(row)}",
                                    "{format_sweep_ratio(row)}"
                                }
                                Badge {
                                    variant: BadgeVariant::Outline,
                                    class: "status-badge {state_class}",
                                    "{state_label}"
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn NotFound(route: Vec<String>) -> Element {
    let path = if route.is_empty() {
        "/".to_string()
    } else {
        format!("/{}", route.join("/"))
    };

    rsx! {
        Card { class: "summary failed",
            CardHeader {
                div { class: "summary-head",
                    CardTitle { "Route not found" }
                    Badge { variant: BadgeVariant::Outline, class: "status-badge failed", "404" }
                }
                CardDescription { "{path}" }
            }
            CardContent {
                div { class: "route-actions",
                    Link { class: "nav-link", to: Route::Tests {}, "Tests" }
                    Link { class: "nav-link", to: Route::Benchmarks {}, "Benchmarks" }
                }
            }
        }
    }
}
