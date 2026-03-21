use async_channel::{Receiver, Sender};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use dioxus::{
    html::{InteractionElementOffset, Key},
    prelude::*,
};
use fusor_nanochat::{
    ComparisonReport, ComparisonSample, DatasetGalleryItem, FirstTokenConstraint, LivePredictor,
    RuntimeConfig, ShapeCount, StrokeTokenizer, build_comparison_report, load_runtime_config,
    load_tokenizer, tokens_to_stroke_scene,
};
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::thread;

const MAIN_CSS: Asset = asset!("/assets/main.css");
const DEFAULT_SKETCH_STAGE_PX: f64 = 480.0;
const PREVIEW_TOKEN_LIMIT: usize = 16;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();
    dioxus::launch(App);
}

#[derive(Clone, Debug, PartialEq)]
enum TaskState {
    Idle,
    Running(String),
    Failed(String),
    Succeeded(String),
}

impl TaskState {
    fn is_running(&self) -> bool {
        matches!(self, Self::Running(_))
    }
}

#[derive(Clone)]
struct PredictorWorker {
    commands: Sender<WorkerCommand>,
}

#[derive(Clone)]
enum WorkerCommand {
    Predict {
        generation: u64,
        revision: u64,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    },
}

#[derive(Clone)]
enum WorkerUpdate {
    Ready {
        generation: u64,
        tokenizer: StrokeTokenizer,
    },
    Prediction {
        generation: u64,
        revision: u64,
        completion_tokens: Vec<u32>,
        /// When true, the worker popped the last prompt token and used constrained
        /// decoding. The first completion token replaces the popped token on accept.
        popped_prompt_token: bool,
    },
    Failed {
        generation: u64,
        message: String,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GridPoint {
    x: i32,
    y: i32,
}

#[component]
fn App() -> Element {
    let mut config = use_signal(load_runtime_config);
    let mut force_cpu = use_signal(|| true);
    let mut status = use_signal(|| TaskState::Idle);
    let report = use_signal(|| None::<ComparisonReport>);

    let busy = status().is_running();

    let run_report = move |_| {
        let runtime = match config.read().as_ref() {
            Ok(runtime) => runtime.clone(),
            Err(error) => {
                status.set(TaskState::Failed(error.clone()));
                return;
            }
        };
        let use_cpu = force_cpu();
        status.set(TaskState::Running(
            "Building the prompted completion comparison report.".to_string(),
        ));
        spawn_report_task(runtime, use_cpu, report, status);
    };

    rsx! {
        document::Title { "Nanochat Studio" }
        document::Stylesheet { href: MAIN_CSS }

        main {
            class: "shell",
            section {
                class: "hero",
                p { class: "eyebrow", "Dioxus Desktop Workbench" }
                h1 { "Nanochat Studio" }
                p {
                    "The app now keeps the nanochat model loaded as a live sketch predictor. "
                    "Draw on the SVG board, watch the completion update in real time, and press Tab to accept the model's continuation."
                }
            }

            div {
                class: "layout",
                aside {
                    class: "sidebar",
                    div {
                        class: "panel",
                        div {
                            class: "panel-head",
                            h2 { "Controls" }
                            p { "Rebuild the report, reload config, or switch between CPU and GPU-backed inference." }
                        }
                        div {
                            class: "panel-body",
                            div {
                                class: "button-stack",
                                button {
                                    class: if force_cpu() { "button button-toggle is-active" } else { "button button-toggle" },
                                    disabled: busy,
                                    onclick: move |_| force_cpu.set(!force_cpu()),
                                    if force_cpu() { "CPU Mode Enabled" } else { "Prefer GPU" }
                                }
                                button {
                                    class: "button button-primary",
                                    disabled: busy,
                                    onclick: run_report,
                                    "Build Report"
                                }
                                button {
                                    class: "button",
                                    disabled: busy,
                                    onclick: move |_| {
                                        config.set(load_runtime_config());
                                        status.set(TaskState::Succeeded(
                                            "Reloaded runtime configuration from nanochat/.env."
                                                .to_string(),
                                        ));
                                    },
                                    "Reload Config"
                                }
                            }
                        }
                    }

                    div {
                        class: "panel",
                        div {
                            class: "panel-head",
                            h2 { "Status" }
                        }
                        div {
                            class: "panel-body",
                            StatusPanel { status: status() }
                        }
                    }

                    div {
                        class: "panel",
                        div {
                            class: "panel-head",
                            h2 { "Runtime" }
                            p { "Loaded directly from nanochat's current env-backed configuration." }
                        }
                        div {
                            class: "panel-body",
                            RuntimePanel {
                                config: config(),
                                force_cpu: force_cpu()
                            }
                        }
                    }
                }

                section {
                    class: "content",
                    div {
                        class: "feature-grid",
                        div {
                            class: "panel",
                            div {
                                class: "panel-head",
                                h2 { "Live Sketch" }
                                p { "Click and drag like a normal sketch pad. New strokes automatically reposition the hidden cursor, and Tab accepts the orange continuation." }
                            }
                            InteractivePad { config, force_cpu }
                        }

                        div {
                            class: "panel",
                            div {
                                class: "panel-head",
                                h2 { "Comparison Report" }
                                p { "Prompted completions and a training gallery rendered from typed nanochat report data." }
                            }
                            ComparisonReportPanel { report }
                        }
                    }
                }
            }
        }
    }
}

#[component]
fn StatusPanel(status: TaskState) -> Element {
    let (class, title, body) = match status {
        TaskState::Idle => (
            "status".to_string(),
            "Ready".to_string(),
            "No report task is running. The sketch predictor loads automatically in the live panel."
                .to_string(),
        ),
        TaskState::Running(message) => (
            "status running".to_string(),
            "Running".to_string(),
            message,
        ),
        TaskState::Failed(message) => (
            "status failed".to_string(),
            "Error".to_string(),
            message,
        ),
        TaskState::Succeeded(message) => (
            "status success".to_string(),
            "Updated".to_string(),
            message,
        ),
    };

    rsx! {
        div {
            class,
            strong { "{title}" }
            p { "{body}" }
        }
    }
}

#[component]
fn RuntimePanel(config: Result<RuntimeConfig, String>, force_cpu: bool) -> Element {
    match config {
        Ok(config) => rsx! {
            div {
                class: "kv",
                RuntimeItem { label: "Device", value: if force_cpu { "CPU".to_string() } else { "GPU if available".to_string() } }
                RuntimeItem { label: "Checkpoint", value: config.gguf_path.display().to_string() }
                RuntimeItem { label: "Dataset", value: config.dataset_path.as_ref().map(|path| path.display().to_string()).unwrap_or_else(|| "bundled examples".to_string()) }
                RuntimeItem { label: "Sample Tokens", value: config.sample_tokens.to_string() }
                RuntimeItem { label: "Prompt Prefix", value: config.sample_prefix_tokens.to_string() }
                RuntimeItem { label: "Block Size", value: config.block_size.to_string() }
                RuntimeItem { label: "Model Width", value: format!("embd {} / layers {}", config.n_embd, config.n_layer) }
            }
        },
        Err(error) => rsx! {
            div {
                class: "empty",
                strong { "Config load failed" }
                p { "{error}" }
            }
        },
    }
}

#[component]
fn RuntimeItem(label: String, value: String) -> Element {
    rsx! {
        div {
            class: "kv-item",
            strong { "{label}" }
            span { "{value}" }
        }
    }
}

#[component]
fn InteractivePad(
    config: Signal<Result<RuntimeConfig, String>>,
    force_cpu: Signal<bool>,
) -> Element {
    let mut prompt_tokens = use_signal(Vec::<u32>::new);
    let mut prediction_tokens = use_signal(Vec::<u32>::new);
    let prediction_popped = use_signal(|| false);
    let mut tokenizer = use_signal(|| None::<StrokeTokenizer>);
    let mut worker = use_signal(|| None::<PredictorWorker>);
    let mut sketch_status = use_signal(|| {
        "SVG board ready. Loading sketch predictor from the current checkpoint…".to_string()
    });
    let mut dragging = use_signal(|| false);
    let mut last_point = use_signal(|| None::<GridPoint>);
    let mut worker_generation = use_signal(|| 0_u64);
    let stage_bounds =
        use_signal(|| StageBounds::new(0.0, 0.0, DEFAULT_SKETCH_STAGE_PX, DEFAULT_SKETCH_STAGE_PX));
    let mut stage_element = use_signal(|| None::<Rc<MountedData>>);
    let mut request_revision = use_signal(|| 0_u64);
    let config_state = config();
    let use_cpu = force_cpu();

    use_effect(use_reactive!(|(config_state, use_cpu)| {
        prompt_tokens.set(Vec::new());
        prediction_tokens.set(Vec::new());
        tokenizer.set(None);
        worker.set(None);
        dragging.set(false);
        last_point.set(None);

        {
            let mut revision = request_revision.write();
            *revision += 1;
        }
        let generation = {
            let mut generation = worker_generation.write();
            *generation += 1;
            *generation
        };

        match config_state.as_ref() {
            Ok(runtime) => {
                let runtime = runtime.clone();
                match load_tokenizer(&runtime) {
                    Ok(loaded_tokenizer) => tokenizer.set(Some(loaded_tokenizer)),
                    Err(error) => {
                        sketch_status.set(error);
                        return;
                    }
                }

                sketch_status.set(
                    "SVG board ready. Loading sketch predictor from the current checkpoint…"
                        .to_string(),
                );
                let predictor_worker = spawn_predictor_worker(
                    runtime,
                    use_cpu,
                    generation,
                    tokenizer,
                    prediction_tokens,
                    prediction_popped,
                    sketch_status,
                    request_revision,
                    worker_generation,
                );
                worker.set(Some(predictor_worker.clone()));
                request_prediction(
                    Some(predictor_worker),
                    generation,
                    request_revision,
                    sketch_status,
                    prediction_tokens,
                    Vec::new(),
                );
            }
            Err(error) => {
                sketch_status.set(error.clone());
            }
        }
    }));

    let tokenizer_loaded = tokenizer();
    let prompt_preview = prompt_tokens();
    let prediction_preview = prediction_tokens();
    let is_popped = prediction_popped();
    // When the prediction was produced with a popped last token, the first
    // completion token overlaps with what the user already drew.  Build
    // display tokens that subtract the overlap so the orange preview starts
    // from the current cursor, not from behind it.
    let display_prediction = if is_popped {
        compute_display_tokens(tokenizer_loaded.as_ref(), &prompt_preview, &prediction_preview)
    } else {
        prediction_preview.clone()
    };
    let prompt_display = tokenizer_loaded
        .as_ref()
        .map(|tokenizer| tokenizer.describe_tokens(&prompt_preview, 80))
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "<empty>".to_string());
    let prediction_display = tokenizer_loaded
        .as_ref()
        .map(|tokenizer| tokenizer.describe_tokens(&display_prediction, 80))
        .filter(|text| !text.is_empty())
        .unwrap_or_else(|| "<none>".to_string());
    let current_cursor = tokenizer_loaded
        .as_ref()
        .map(|tokenizer| tokenizer.cursor_after_tokens(&prompt_preview))
        .unwrap_or((0, 0));
    let grid_size = tokenizer_loaded
        .as_ref()
        .map(interactive_grid_size)
        .unwrap_or(16);
    let stroke_scene = tokenizer_loaded
        .as_ref()
        .map(|tokenizer| tokens_to_stroke_scene(tokenizer, &prompt_preview, &display_prediction));
    let prompt_paths = stroke_scene
        .as_ref()
        .map(|scene| {
            scene
                .prompt_strokes
                .iter()
                .filter_map(|stroke| grid_path_data(&stroke.points))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let prediction_paths = stroke_scene
        .as_ref()
        .map(|scene| {
            scene
                .continuation_strokes
                .iter()
                .filter_map(|stroke| grid_path_data(&stroke.points))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let inner_board_size = grid_size as f32 - 0.28;
    let view_box = format!("0 0 {grid_size} {grid_size}");
    let cursor_x = current_cursor.0 as f32 + 0.5;
    let cursor_y = current_cursor.1 as f32 + 0.5;
    let max_index = grid_size.saturating_sub(1) as i32;
    let neighbor_dots: Vec<(f32, f32)> = [
        (-1, -1), (0, -1), (1, -1),
        (-1,  0),          (1,  0),
        (-1,  1), (0,  1), (1,  1),
    ]
    .iter()
    .filter_map(|&(dx, dy)| {
        let nx = current_cursor.0 + dx;
        let ny = current_cursor.1 + dy;
        if nx >= 0 && ny >= 0 && nx <= max_index && ny <= max_index {
            Some((nx as f32 + 0.5, ny as f32 + 0.5))
        } else {
            None
        }
    })
    .collect();

    rsx! {
        div {
            class: "panel-body",
            div {
                class: "sketch-shell",
                div {
                    class: "sketch-toolbar",
                    div {
                        class: "button-row",
                        button {
                            class: "button",
                            disabled: prompt_tokens.read().is_empty(),
                            onclick: move |_| {
                                let mut next = prompt_tokens();
                                if next.pop().is_some() {
                                    prompt_tokens.set(next.clone());
                                    prediction_tokens.set(Vec::new());
                                    request_prediction(
                                        worker(),
                                        worker_generation(),
                                        request_revision,
                                        sketch_status,
                                        prediction_tokens,
                                        next,
                                    );
                                }
                            },
                            "Undo"
                        }
                        button {
                            class: "button",
                            disabled: prompt_tokens.read().is_empty() && prediction_tokens.read().is_empty(),
                            onclick: move |_| {
                                prompt_tokens.set(Vec::new());
                                prediction_tokens.set(Vec::new());
                                request_prediction(
                                    worker(),
                                    worker_generation(),
                                    request_revision,
                                    sketch_status,
                                    prediction_tokens,
                                    Vec::new(),
                                );
                            },
                            "Clear"
                        }
                        button {
                            class: "button button-primary",
                            disabled: prediction_tokens.read().is_empty(),
                            onclick: move |_| {
                                accept_prediction(
                                    prompt_tokens,
                                    prediction_tokens,
                                    prediction_popped,
                                    worker(),
                                    worker_generation(),
                                    request_revision,
                                    sketch_status,
                                );
                            },
                            "Accept Tab"
                        }
                    }
                }

                div {
                    class: "sketch-stage",
                    tabindex: "0",
                    onmounted: move |event| {
                        let mounted = event.data();
                        stage_element.set(Some(mounted.clone()));
                        spawn(async move {
                            let _ = mounted.set_focus(true).await;
                            if let Ok(rect) = mounted.get_client_rect().await {
                                set_stage_bounds(
                                    stage_bounds,
                                    rect.origin.x,
                                    rect.origin.y,
                                    rect.size.width,
                                    rect.size.height,
                                );
                            }
                        });
                    },
                    onresize: move |event| {
                        if let Ok(size) = event.data().get_border_box_size() {
                            let bounds = stage_bounds();
                            set_stage_bounds(
                                stage_bounds,
                                bounds.left,
                                bounds.top,
                                size.width,
                                size.height,
                            );
                        }
                    },
                    onkeydown: move |event| {
                        match event.key() {
                            Key::Tab => {
                                event.prevent_default();
                                accept_prediction(
                                    prompt_tokens,
                                    prediction_tokens,
                                    prediction_popped,
                                    worker(),
                                    worker_generation(),
                                    request_revision,
                                    sketch_status,
                                );
                            }
                            Key::Backspace => {
                                event.prevent_default();
                                let mut next = prompt_tokens();
                                if next.pop().is_some() {
                                    prompt_tokens.set(next.clone());
                                    prediction_tokens.set(Vec::new());
                                    request_prediction(
                                        worker(),
                                        worker_generation(),
                                        request_revision,
                                        sketch_status,
                                        prediction_tokens,
                                        next,
                                    );
                                }
                            }
                            Key::Escape => {
                                event.prevent_default();
                                prompt_tokens.set(Vec::new());
                                prediction_tokens.set(Vec::new());
                                request_prediction(
                                    worker(),
                                    worker_generation(),
                                    request_revision,
                                    sketch_status,
                                    prediction_tokens,
                                    Vec::new(),
                                );
                            }
                            _ => {}
                        }
                    },
                    onpointerdown: move |event| {
                        event.prevent_default();
                        if let Some(stage) = stage_element() {
                            spawn(async move {
                                let _ = stage.set_focus(true).await;
                            });
                        }
                        let Some(tokenizer) = tokenizer() else {
                            return;
                        };
                        let point = grid_point_from_event(
                            &event,
                            stage_bounds(),
                            interactive_grid_size(&tokenizer),
                        );
                        let cursor = tokenizer.cursor_after_tokens(&prompt_tokens());
                        let current = GridPoint {
                            x: cursor.0,
                            y: cursor.1,
                        };
                        if point != current {
                            let mut next = prompt_tokens();
                            append_tokens_for_path(&tokenizer, &mut next, current, point, false);
                            prompt_tokens.set(next.clone());
                            prediction_tokens.set(Vec::new());
                            request_prediction(
                                worker(),
                                worker_generation(),
                                request_revision,
                                sketch_status,
                                prediction_tokens,
                                next,
                            );
                        }
                        dragging.set(true);
                        last_point.set(Some(point));
                    },
                    onpointermove: move |event| {
                        let Some(tokenizer) = tokenizer() else {
                            return;
                        };
                        if !dragging() {
                            return;
                        }
                        event.prevent_default();
                        let Some(previous_point) = last_point() else {
                            return;
                        };
                        // Snap to the nearest neighbor of the current grid
                        // point so that diagonal connections are as easy to
                        // reach as cardinal ones.
                        let point = grid_point_from_event_constrained(
                            &event,
                            stage_bounds(),
                            interactive_grid_size(&tokenizer),
                            previous_point,
                        );
                        if point == previous_point {
                            return;
                        }
                        let mut next = prompt_tokens();
                        append_tokens_for_path(&tokenizer, &mut next, previous_point, point, true);
                        prompt_tokens.set(next.clone());
                        prediction_tokens.set(Vec::new());
                        last_point.set(Some(point));
                        request_prediction(
                            worker(),
                            worker_generation(),
                            request_revision,
                            sketch_status,
                            prediction_tokens,
                            next,
                        );
                    },
                    onpointerup: move |_| {
                        dragging.set(false);
                        last_point.set(None);
                    },
                    onpointercancel: move |_| {
                        dragging.set(false);
                        last_point.set(None);
                    },
                    onpointerleave: move |_| {
                        dragging.set(false);
                        last_point.set(None);
                    },
                    svg {
                        class: "sketch-board",
                        view_box,
                        preserve_aspect_ratio: "none",
                        pointer_events: "none",
                        title { "Live nanochat sketch board" }
                        rect {
                            x: "0",
                            y: "0",
                            width: "{grid_size}",
                            height: "{grid_size}",
                            fill: "#fdf6e3",
                        }
                        rect {
                            x: "0.14",
                            y: "0.14",
                            width: "{inner_board_size}",
                            height: "{inner_board_size}",
                            rx: "0.48",
                            fill: "#fffaf0",
                            stroke: "#e9dcc3",
                            stroke_width: "0.08",
                        }
                        g {
                            opacity: "0.28",
                            stroke: "#264653",
                            stroke_width: "0.03",
                            for axis in 1..grid_size {
                                line {
                                    x1: "{axis}",
                                    y1: "0",
                                    x2: "{axis}",
                                    y2: "{grid_size}",
                                }
                            }
                            for axis in 1..grid_size {
                                line {
                                    x1: "0",
                                    y1: "{axis}",
                                    x2: "{grid_size}",
                                    y2: "{axis}",
                                }
                            }
                        }
                        g {
                            stroke: "#264653",
                            stroke_width: "0.34",
                            stroke_linecap: "round",
                            stroke_linejoin: "round",
                            fill: "none",
                            opacity: "0.76",
                            for path_data in prompt_paths {
                                path { d: "{path_data}" }
                            }
                        }
                        g {
                            stroke: "#e76f51",
                            stroke_width: "0.34",
                            stroke_linecap: "round",
                            stroke_linejoin: "round",
                            fill: "none",
                            for path_data in prediction_paths {
                                path { d: "{path_data}" }
                            }
                        }
                        // Neighbor dots: show reachable grid points.
                        for (nx, ny) in neighbor_dots.iter().copied() {
                            circle {
                                cx: "{nx}",
                                cy: "{ny}",
                                r: "0.09",
                                fill: "#264653",
                                opacity: "0.32",
                            }
                        }
                        circle {
                            cx: "{cursor_x}",
                            cy: "{cursor_y}",
                            r: "0.28",
                            fill: "#fffaf0",
                            opacity: "0.94",
                        }
                        circle {
                            cx: "{cursor_x}",
                            cy: "{cursor_y}",
                            r: "0.14",
                            fill: "#264653",
                        }
                    }
                    div {
                        class: "sketch-hint",
                        strong { "Tab" }
                        span { "accepts the orange continuation" }
                    }
                }

                div {
                    class: "sketch-meta",
                    div { class: "kv-item", strong { "Predictor" } span { "{sketch_status()}" } }
                    div { class: "kv-item", strong { "Cursor" } span { "x {current_cursor.0} / y {current_cursor.1}" } }
                    div { class: "kv-item", strong { "Grid" } span { "{grid_size}x{grid_size}" } }
                }

                div {
                    class: "sketch-log",
                    div {
                        class: "sketch-log-block",
                        strong { "Prompt" }
                        code { "{prompt_display}" }
                    }
                    div {
                        class: "sketch-log-block",
                        strong { "Prediction" }
                        code { "{prediction_display}" }
                    }
                }
            }
        }
    }
}

#[component]
fn ComparisonReportPanel(report: ReadSignal<Option<ComparisonReport>>) -> Element {
    let report = report.read();
    match report.as_ref() {
        Some(report) => rsx! {
            div {
                class: "panel-body",
                div {
                    class: "stats",
                    StatCard { value: report.train_examples.to_string(), label: "train examples".to_string() }
                    StatCard { value: report.compare_examples.to_string(), label: "eval examples".to_string() }
                    StatCard { value: report.sample_count.to_string(), label: "prompted samples".to_string() }
                    StatCard { value: format!("{:.1}%", report.average_similarity * 100.0), label: "avg similarity".to_string() }
                }
                p {
                    style: "margin: 16px 0 10px; color: var(--muted);",
                    "Dataset label: {report.dataset_label}"
                }
                ul {
                    class: "shape-list",
                    for count in report.shape_counts.iter().cloned() {
                        ShapeCountRow { count }
                    }
                }
                div {
                    style: "height: 18px;"
                }
                div {
                    class: "compare-grid",
                    for sample in report.samples.iter().cloned() {
                        ComparisonCard { sample }
                    }
                }
                div {
                    style: "height: 18px;"
                }
                div {
                    class: "panel-head",
                    h3 { "Training Gallery" }
                    p { "Representative examples from the current training split rendered through the shared SVG helper." }
                }
                div {
                    class: "gallery-grid",
                    for item in report.dataset_gallery.iter().cloned() {
                        GalleryCard { item }
                    }
                }
            }
        },
        None => rsx! {
            div {
                class: "panel-body",
                div {
                    class: "empty",
                    "No comparison report generated yet."
                }
            }
        },
    }
}

#[component]
fn StatCard(value: String, label: String) -> Element {
    rsx! {
        div {
            class: "stat",
            strong { "{value}" }
            span { "{label}" }
        }
    }
}

#[component]
fn ShapeCountRow(count: ShapeCount) -> Element {
    rsx! {
        li {
            strong { "{count.shape}" }
            " : {count.count} prompted comparisons"
        }
    }
}

#[component]
fn ComparisonCard(sample: ComparisonSample) -> Element {
    let similarity = format!("{:.1}%", sample.similarity * 100.0);

    rsx! {
        article {
            class: "compare-card",
            div {
                class: "compare-head",
                h3 { "{sample.example_label}" }
                p {
                    "Prompted from "
                    strong { "{sample.shape}" }
                    " in "
                    code { "{sample.source_path}" }
                }
                p {
                    strong { "Prompt" }
                    code { "{sample.prompt_tokens}" }
                }
            }
            div {
                class: "compare-pair",
                figure {
                    class: "compare-figure",
                    img {
                        src: svg_data_uri(&sample.generated_svg),
                        alt: "Generated completion for {sample.example_label}"
                    }
                    figcaption { "Model completion" }
                }
                figure {
                    class: "compare-figure",
                    img {
                        src: svg_data_uri(&sample.expected_svg),
                        alt: "Expected completion for {sample.example_label}"
                    }
                    figcaption { "Expected continuation" }
                }
            }
            div {
                class: "metrics",
                div {
                    class: "metric",
                    strong { "Edit distance" }
                    span { "{sample.edit_distance}" }
                }
                div {
                    class: "metric",
                    strong { "Similarity" }
                    span { "{similarity}" }
                }
            }
            p {
                strong { "Model continuation" }
                code { "{sample.generated_tokens}" }
            }
            p {
                strong { "Expected continuation" }
                code { "{sample.expected_tokens}" }
            }
        }
    }
}

#[component]
fn GalleryCard(item: DatasetGalleryItem) -> Element {
    rsx! {
        article {
            class: "gallery-card",
            img {
                src: svg_data_uri(&item.svg),
                alt: "{item.shape} training sample"
            }
            h3 { "{item.shape}" }
            p { code { "{item.path}" } }
            p { code { "{item.tokens}" } }
        }
    }
}

fn spawn_report_task(
    runtime: RuntimeConfig,
    force_cpu: bool,
    report_signal: Signal<Option<ComparisonReport>>,
    status_signal: Signal<TaskState>,
) {
    let (report_tx, report_rx) = async_channel::bounded::<Result<ComparisonReport, String>>(1);

    #[cfg(not(target_arch = "wasm32"))]
    thread::spawn(move || {
        let result = build_comparison_report(runtime, force_cpu);
        let _ = report_tx.send_blocking(result);
    });

    #[cfg(target_arch = "wasm32")]
    spawn(async move {
        let result = build_comparison_report(runtime, force_cpu);
        let _ = report_tx.send(result).await;
    });

    spawn(async move {
        if let Ok(result) = report_rx.recv().await {
            handle_report_result(result, report_signal, status_signal);
        }
    });
}

fn handle_report_result(
    result: Result<ComparisonReport, String>,
    mut report_signal: Signal<Option<ComparisonReport>>,
    mut status_signal: Signal<TaskState>,
) {
    match result {
        Ok(report) => {
            let summary = format!(
                "Built comparison report with {} prompted samples from {} training examples.",
                report.sample_count, report.train_examples
            );
            report_signal.set(Some(report));
            status_signal.set(TaskState::Succeeded(summary));
        }
        Err(error) => {
            status_signal.set(TaskState::Failed(error));
        }
    }
}

fn spawn_predictor_worker(
    runtime: RuntimeConfig,
    force_cpu: bool,
    generation: u64,
    mut tokenizer_signal: Signal<Option<StrokeTokenizer>>,
    mut prediction_signal: Signal<Vec<u32>>,
    mut prediction_popped_signal: Signal<bool>,
    mut status_signal: Signal<String>,
    request_revision: Signal<u64>,
    worker_generation: Signal<u64>,
) -> PredictorWorker {
    let (command_tx, command_rx) = async_channel::unbounded::<WorkerCommand>();
    let (update_tx, update_rx) = async_channel::unbounded::<WorkerUpdate>();

    #[cfg(not(target_arch = "wasm32"))]
    thread::spawn(move || {
        run_predictor_loop_blocking(runtime, force_cpu, generation, command_rx, update_tx);
    });

    #[cfg(target_arch = "wasm32")]
    spawn(async move {
        run_predictor_loop_async(runtime, force_cpu, generation, command_rx, update_tx).await;
    });

    spawn(async move {
        while let Ok(update) = update_rx.recv().await {
            match update {
                WorkerUpdate::Ready {
                    generation,
                    tokenizer,
                } => {
                    if generation != worker_generation() {
                        continue;
                    }
                    tokenizer_signal.set(Some(tokenizer));
                    prediction_signal.set(Vec::new());
                    status_signal.set(
                        "Predictor ready. Draw on the pad and press Tab to accept the continuation."
                            .to_string(),
                    );
                }
                WorkerUpdate::Prediction {
                    generation,
                    revision,
                    completion_tokens,
                    popped_prompt_token,
                } => {
                    if generation != worker_generation() || revision != request_revision() {
                        continue;
                    }
                    let has_prediction = !completion_tokens.is_empty();
                    prediction_signal.set(completion_tokens);
                    prediction_popped_signal.set(popped_prompt_token);
                    status_signal.set(if has_prediction {
                        "Prediction updated. Press Tab to accept the continuation.".to_string()
                    } else {
                        "The model has no continuation to suggest from the current sketch."
                            .to_string()
                    });
                }
                WorkerUpdate::Failed {
                    generation,
                    message,
                } => {
                    if generation != worker_generation() {
                        continue;
                    }
                    prediction_signal.set(Vec::new());
                    status_signal.set(message);
                }
            }
        }
    });

    PredictorWorker {
        commands: command_tx,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn run_predictor_loop_blocking(
    runtime: RuntimeConfig,
    force_cpu: bool,
    generation: u64,
    command_rx: Receiver<WorkerCommand>,
    update_tx: Sender<WorkerUpdate>,
) {
    let predictor = match LivePredictor::load(runtime, force_cpu) {
        Ok(predictor) => predictor,
        Err(error) => {
            let _ = update_tx.send_blocking(WorkerUpdate::Failed {
                generation,
                message: error,
            });
            return;
        }
    };

    let _ = update_tx.send_blocking(WorkerUpdate::Ready {
        generation,
        tokenizer: predictor.tokenizer().clone(),
    });

    while let Ok(command) = command_rx.recv_blocking() {
        let mut latest = command;
        while let Ok(queued) = command_rx.try_recv() {
            latest = queued;
        }
        let _ = update_tx.send_blocking(predict_with_command(&predictor, latest));
    }
}

#[cfg(target_arch = "wasm32")]
async fn run_predictor_loop_async(
    runtime: RuntimeConfig,
    force_cpu: bool,
    generation: u64,
    command_rx: Receiver<WorkerCommand>,
    update_tx: Sender<WorkerUpdate>,
) {
    let predictor = match LivePredictor::load(runtime, force_cpu) {
        Ok(predictor) => predictor,
        Err(error) => {
            let _ = update_tx
                .send(WorkerUpdate::Failed {
                    generation,
                    message: error,
                })
                .await;
            return;
        }
    };

    let _ = update_tx
        .send(WorkerUpdate::Ready {
            generation,
            tokenizer: predictor.tokenizer().clone(),
        })
        .await;

    while let Ok(command) = command_rx.recv().await {
        let mut latest = command;
        while let Ok(queued) = command_rx.try_recv() {
            latest = queued;
        }
        let _ = update_tx
            .send(predict_with_command(&predictor, latest))
            .await;
    }
}

fn predict_with_command(predictor: &LivePredictor, command: WorkerCommand) -> WorkerUpdate {
    match command {
        WorkerCommand::Predict {
            generation,
            revision,
            prompt_tokens,
            max_tokens,
        } => {
            // Build a constraint from the last prompt token (BPE boundary fix).
            // Pop the last token if it's a valid segment, predict from the
            // shortened prompt, and constrain the first predicted token to
            // continue in the same mode+direction with count >= user's count.
            let (effective_prompt, constraint, popped) =
                build_prediction_constraint(predictor.tokenizer(), prompt_tokens);

            match predictor.predict_greedy(&effective_prompt, max_tokens, constraint) {
                Ok(completion_tokens) => WorkerUpdate::Prediction {
                    generation,
                    revision,
                    completion_tokens,
                    popped_prompt_token: popped,
                },
                Err(error) => WorkerUpdate::Failed {
                    generation,
                    message: error,
                },
            }
        }
    }
}

/// Pop the last prompt token if it's a valid segment token, and build a
/// `FirstTokenConstraint` so the model's first prediction continues in the
/// same direction with at least the same count.
fn build_prediction_constraint(
    tokenizer: &StrokeTokenizer,
    mut prompt_tokens: Vec<u32>,
) -> (Vec<u32>, Option<FirstTokenConstraint>, bool) {
    if let Some(&last_token) = prompt_tokens.last() {
        if let Some(decoded) = tokenizer.decode_segment_token(last_token) {
            if let (Some(direction_index), Some(count)) =
                (decoded.direction_index, decoded.count)
            {
                prompt_tokens.pop();
                let constraint = FirstTokenConstraint {
                    mode_index: decoded.mode_index,
                    direction_index,
                    min_count: count,
                };
                return (prompt_tokens, Some(constraint), true);
            }
        }
    }
    (prompt_tokens, None, false)
}

fn request_prediction(
    worker: Option<PredictorWorker>,
    generation: u64,
    mut request_revision: Signal<u64>,
    mut status_signal: Signal<String>,
    mut prediction_signal: Signal<Vec<u32>>,
    prompt_tokens: Vec<u32>,
) {
    let Some(worker) = worker else {
        status_signal.set("Predictor is not ready yet.".to_string());
        return;
    };

    let revision = {
        let mut revision = request_revision.write();
        *revision += 1;
        *revision
    };
    prediction_signal.set(Vec::new());

    if let Err(error) = worker.commands.try_send(WorkerCommand::Predict {
        generation,
        revision,
        prompt_tokens,
        max_tokens: PREVIEW_TOKEN_LIMIT,
    }) {
        status_signal.set(format!("predictor worker is unavailable: {error}"));
    }
}

fn accept_prediction(
    mut prompt_tokens: Signal<Vec<u32>>,
    mut prediction_tokens: Signal<Vec<u32>>,
    mut prediction_popped: Signal<bool>,
    worker: Option<PredictorWorker>,
    generation: u64,
    request_revision: Signal<u64>,
    status_signal: Signal<String>,
) {
    let prediction = prediction_tokens();
    if prediction.is_empty() {
        return;
    }

    let is_popped = prediction_popped();
    let mut next = prompt_tokens();
    if is_popped && !next.is_empty() {
        // Replace the last prompt token with the first completion token
        // (which has the full count from constrained decoding)
        *next.last_mut().unwrap() = prediction[0];
        next.extend_from_slice(&prediction[1..]);
    } else {
        next.extend_from_slice(&prediction);
    }
    prompt_tokens.set(next.clone());
    prediction_tokens.set(Vec::new());
    prediction_popped.set(false);
    request_prediction(
        worker,
        generation,
        request_revision,
        status_signal,
        prediction_tokens,
        next,
    );
}

/// When the worker popped the last prompt token and produced a constrained
/// prediction, the first completion token covers what the user already drew
/// plus potentially more.  This function creates display tokens where the
/// first token's count is reduced by the overlap.
fn compute_display_tokens(
    tokenizer: Option<&StrokeTokenizer>,
    prompt_tokens: &[u32],
    completion_tokens: &[u32],
) -> Vec<u32> {
    let Some(tokenizer) = tokenizer else {
        return completion_tokens.to_vec();
    };
    let Some(&first_completion) = completion_tokens.first() else {
        return Vec::new();
    };
    let Some(&last_prompt) = prompt_tokens.last() else {
        return completion_tokens.to_vec();
    };
    let Some(completion_decoded) = tokenizer.decode_segment_token(first_completion) else {
        return completion_tokens.to_vec();
    };
    let Some(prompt_decoded) = tokenizer.decode_segment_token(last_prompt) else {
        return completion_tokens.to_vec();
    };
    let (Some(comp_dir), Some(comp_count)) =
        (completion_decoded.direction_index, completion_decoded.count)
    else {
        return completion_tokens.to_vec();
    };
    let (Some(prompt_dir), Some(prompt_count)) =
        (prompt_decoded.direction_index, prompt_decoded.count)
    else {
        return completion_tokens.to_vec();
    };

    // Only adjust if same mode+direction (the constraint should guarantee this)
    if completion_decoded.mode_index != prompt_decoded.mode_index || comp_dir != prompt_dir {
        return completion_tokens.to_vec();
    }

    let remaining = comp_count.saturating_sub(prompt_count);
    if remaining == 0 {
        // The model chose the exact count the user drew — skip first token entirely
        completion_tokens[1..].to_vec()
    } else {
        let mut display = Vec::with_capacity(completion_tokens.len());
        display.push(tokenizer.token_from_components(
            completion_decoded.mode_index,
            comp_dir,
            remaining,
        ));
        display.extend_from_slice(&completion_tokens[1..]);
        display
    }
}

fn interactive_grid_size(tokenizer: &StrokeTokenizer) -> usize {
    tokenizer.grid_size().max(tokenizer.max_count() + 1).max(2)
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StageBounds {
    left: f64,
    top: f64,
    width: f64,
    height: f64,
}

impl StageBounds {
    const fn new(left: f64, top: f64, width: f64, height: f64) -> Self {
        Self {
            left,
            top,
            width,
            height,
        }
    }
}

fn set_stage_bounds(
    mut stage_bounds: Signal<StageBounds>,
    left: f64,
    top: f64,
    width: f64,
    height: f64,
) {
    stage_bounds.set(StageBounds::new(left, top, width.max(1.0), height.max(1.0)));
}

fn grid_point_from_event(
    event: &PointerEvent,
    stage_bounds: StageBounds,
    grid_size: usize,
) -> GridPoint {
    let point = event.data().element_coordinates();
    let max_index = grid_size.saturating_sub(1) as f64;
    let local_x = point.x.clamp(0.0, stage_bounds.width);
    let local_y = point.y.clamp(0.0, stage_bounds.height);
    let x = ((local_x / stage_bounds.width.max(1.0)) * grid_size as f64 - 0.5)
        .round()
        .clamp(0.0, max_index);
    let y = ((local_y / stage_bounds.height.max(1.0)) * grid_size as f64 - 0.5)
        .round()
        .clamp(0.0, max_index);
    GridPoint {
        x: x as i32,
        y: y as i32,
    }
}

/// During a drag, snap the pointer to one of the 8 neighbors of `anchor`
/// (or `anchor` itself).  The pointer must be within a snap radius of a
/// neighbor dot before the connection is drawn — this prevents the line
/// from appearing before the pointer visually reaches the target dot.
fn grid_point_from_event_constrained(
    event: &PointerEvent,
    stage_bounds: StageBounds,
    grid_size: usize,
    anchor: GridPoint,
) -> GridPoint {
    let point = event.data().element_coordinates();
    let max_index = grid_size.saturating_sub(1) as i32;
    // Raw continuous position in grid coordinates (not rounded).
    let raw_x = (point.x.clamp(0.0, stage_bounds.width) / stage_bounds.width.max(1.0))
        * grid_size as f64
        - 0.5;
    let raw_y = (point.y.clamp(0.0, stage_bounds.height) / stage_bounds.height.max(1.0))
        * grid_size as f64
        - 0.5;

    // Snap radius: the pointer must be within this distance (squared) of a
    // neighbor dot to connect to it.  0.45² = 0.2025 grid units².
    const SNAP_RADIUS_SQ: f64 = 0.45 * 0.45;

    let mut best = anchor;
    let mut best_dist = f64::MAX;
    for dy in -1..=1_i32 {
        for dx in -1..=1_i32 {
            if dx == 0 && dy == 0 {
                continue; // skip anchor itself
            }
            let cx = anchor.x + dx;
            let cy = anchor.y + dy;
            if cx < 0 || cy < 0 || cx > max_index || cy > max_index {
                continue;
            }
            let dist = (raw_x - cx as f64).powi(2) + (raw_y - cy as f64).powi(2);
            if dist < best_dist && dist <= SNAP_RADIUS_SQ {
                best_dist = dist;
                best = GridPoint { x: cx, y: cy };
            }
        }
    }
    best
}

fn grid_path_data(points: &[(i32, i32)]) -> Option<String> {
    let mut points = points.iter();
    let &(first_x, first_y) = points.next()?;
    let mut path = format!("M {:.2} {:.2}", first_x as f32 + 0.5, first_y as f32 + 0.5);
    for &(x, y) in points {
        path.push_str(&format!(" L {:.2} {:.2}", x as f32 + 0.5, y as f32 + 0.5));
    }
    Some(path)
}

fn append_tokens_for_path(
    tokenizer: &StrokeTokenizer,
    prompt_tokens: &mut Vec<u32>,
    from: GridPoint,
    to: GridPoint,
    is_draw: bool,
) {
    let mode = if is_draw { 1u32 } else { 0 };
    let mut current = from;
    while current != to {
        let dx = (to.x - current.x).signum();
        let dy = (to.y - current.y).signum();
        let direction_index = direction_index_from_delta(dx, dy);
        let legal_limit = tokenizer
            .legal_count_limit((current.x, current.y), direction_index)
            .min(tokenizer.max_count())
            .max(1);

        let mut count = 0usize;
        while current != to && count < legal_limit {
            let next_dx = (to.x - current.x).signum();
            let next_dy = (to.y - current.y).signum();
            if next_dx != dx || next_dy != dy {
                break;
            }
            current.x += dx;
            current.y += dy;
            count += 1;
        }

        if count == 0 {
            break;
        }

        // RLE merging: if the last token has the same mode+direction, merge by
        // incrementing the count instead of pushing a new token.
        let merged = if let Some(&last_token) = prompt_tokens.last() {
            if let Some(decoded) = tokenizer.decode_segment_token(last_token) {
                if decoded.mode_index == mode
                    && decoded.direction_index == Some(direction_index)
                {
                    let prev_count = decoded.count.unwrap_or(0);
                    let new_count = prev_count + count;
                    // Check the limit from the cursor before the previous token
                    let prev_cursor =
                        tokenizer.cursor_after_tokens(&prompt_tokens[..prompt_tokens.len() - 1]);
                    let merge_limit = tokenizer
                        .legal_count_limit(prev_cursor, direction_index)
                        .min(tokenizer.max_count());
                    if new_count <= merge_limit {
                        *prompt_tokens.last_mut().unwrap() =
                            tokenizer.token_from_components(mode, direction_index, new_count);
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if !merged {
            prompt_tokens.push(tokenizer.token_from_components(mode, direction_index, count));
        }
    }
}

fn direction_index_from_delta(dx: i32, dy: i32) -> u32 {
    match (dx, dy) {
        (0, -1) => 0,
        (1, -1) => 1,
        (1, 0) => 2,
        (1, 1) => 3,
        (0, 1) => 4,
        (-1, 1) => 5,
        (-1, 0) => 6,
        (-1, -1) => 7,
        other => panic!("unsupported direction delta {other:?}"),
    }
}

fn svg_data_uri(svg: &str) -> String {
    format!(
        "data:image/svg+xml;base64,{}",
        STANDARD.encode(svg.as_bytes())
    )
}
