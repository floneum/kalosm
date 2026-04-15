use dioxus::prelude::*;
use rgliner::{
    relation_decoding::Relation,
    relex::{GlinerRelEx, GlinerRelExSource},
    DecodingMode, Entity, Gliner, GlinerSource,
};

const TOKEN_BUDGET: usize = 128;

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        console_error_panic_hook::set_once();
        tracing_wasm::set_as_global_default();
    }
    dioxus::launch(App);
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ner,
    Relex,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelChoice {
    Edge,
    Small,
    Base,
    Large,
    RelexMulti,
    RelexBase,
    RelexLarge,
}

impl ModelChoice {
    fn label(self) -> &'static str {
        match self {
            ModelChoice::Edge => "edge · 60M · fastest",
            ModelChoice::Small => "small · 108M",
            ModelChoice::Base => "base · 194M",
            ModelChoice::Large => "large · 530M · best NER",
            ModelChoice::RelexMulti => "relex-multi · multilingual",
            ModelChoice::RelexBase => "relex-base · English",
            ModelChoice::RelexLarge => "relex-large · English · best",
        }
    }

    fn value(self) -> &'static str {
        match self {
            ModelChoice::Edge => "edge",
            ModelChoice::Small => "small",
            ModelChoice::Base => "base",
            ModelChoice::Large => "large",
            ModelChoice::RelexMulti => "relex-multi",
            ModelChoice::RelexBase => "relex-base",
            ModelChoice::RelexLarge => "relex-large",
        }
    }

    fn from_value(v: &str) -> Option<Self> {
        Some(match v {
            "edge" => ModelChoice::Edge,
            "small" => ModelChoice::Small,
            "base" => ModelChoice::Base,
            "large" => ModelChoice::Large,
            "relex-multi" => ModelChoice::RelexMulti,
            "relex-base" => ModelChoice::RelexBase,
            "relex-large" => ModelChoice::RelexLarge,
            _ => return None,
        })
    }

    fn default_for(mode: Mode) -> Self {
        match mode {
            Mode::Ner => ModelChoice::Edge,
            Mode::Relex => ModelChoice::RelexMulti,
        }
    }

    fn for_mode(mode: Mode) -> &'static [ModelChoice] {
        match mode {
            Mode::Ner => &[
                ModelChoice::Edge,
                ModelChoice::Small,
                ModelChoice::Base,
                ModelChoice::Large,
            ],
            Mode::Relex => &[
                ModelChoice::RelexMulti,
                ModelChoice::RelexBase,
                ModelChoice::RelexLarge,
            ],
        }
    }
}

enum LoadedModel {
    Ner {
        choice: ModelChoice,
        inner: Gliner,
    },
    Relex {
        choice: ModelChoice,
        inner: GlinerRelEx,
    },
}

impl LoadedModel {
    fn choice(&self) -> ModelChoice {
        match self {
            LoadedModel::Ner { choice, .. } | LoadedModel::Relex { choice, .. } => *choice,
        }
    }
}

#[derive(Clone, Default)]
struct RelexResult {
    entities: Vec<Entity>,
    relations: Vec<Relation>,
}

#[component]
fn App() -> Element {
    let mut mode = use_signal(|| Mode::Ner);
    let mut choice = use_signal(|| ModelChoice::Edge);

    let mut text = use_signal(|| {
        "Apple Inc. was founded by Steve Jobs in California. Microsoft is headquartered in Redmond."
            .to_string()
    });
    let mut entity_labels =
        use_signal(|| "person, organization, location".to_string());
    let mut relation_labels = use_signal(|| "founded by, located in".to_string());

    let mut model = use_signal(|| None::<LoadedModel>);
    let mut loading = use_signal(|| false);
    let mut running = use_signal(|| false);
    let mut error = use_signal(|| None::<String>);
    let mut ner_out = use_signal(Vec::<Entity>::new);
    let mut relex_out = use_signal(RelexResult::default);
    let mut status = use_signal(|| "No model loaded".to_string());

    let mut on_mode_change = move |new_mode: Mode| {
        if mode() != new_mode {
            mode.set(new_mode);
            choice.set(ModelChoice::default_for(new_mode));
            ner_out.write().clear();
            *relex_out.write() = RelexResult::default();
        }
    };

    let on_load = move |_| {
        if loading() {
            return;
        }
        let selected = choice();
        loading.set(true);
        error.set(None);
        status.set(format!("Loading {}…", selected.label()));
        spawn(async move {
            match build_model(selected).await {
                Ok(m) => {
                    let dev = match &m {
                        LoadedModel::Ner { inner, .. } => {
                            if inner.device().is_gpu() { "GPU" } else { "CPU" }
                        }
                        LoadedModel::Relex { inner, .. } => {
                            if inner.device().is_gpu() { "GPU" } else { "CPU" }
                        }
                    };
                    model.set(Some(m));
                    status.set(format!("{} ready on {dev}", selected.label()));
                }
                Err(e) => {
                    error.set(Some(format!("{e}")));
                    status.set("Load failed".to_string());
                }
            }
            loading.set(false);
        });
    };

    let on_extract = move |_| {
        if running() {
            return;
        }
        let current_text = text();
        let ent_labels = parse_labels(&entity_labels());
        let rel_labels = parse_labels(&relation_labels());
        let current_mode = mode();

        running.set(true);
        error.set(None);

        let Some(mut taken) = model.write().take() else {
            error.set(Some("Load a model first.".to_string()));
            running.set(false);
            return;
        };

        spawn(async move {
            let outcome = run_extraction(
                &mut taken,
                current_mode,
                &current_text,
                &ent_labels,
                &rel_labels,
            )
            .await;

            match outcome {
                Ok(ExtractionOutput::Ner(entities)) => {
                    status.set(format!("Extracted {} entities", entities.len()));
                    ner_out.set(entities);
                }
                Ok(ExtractionOutput::Relex(result)) => {
                    status.set(format!(
                        "Extracted {} entities, {} relations",
                        result.entities.len(),
                        result.relations.len()
                    ));
                    relex_out.set(result);
                }
                Err(e) => error.set(Some(e)),
            }

            model.set(Some(taken));
            running.set(false);
        });
    };

    let has_model = model.read().is_some();
    let model_mismatch = model
        .read()
        .as_ref()
        .map(|m| m.choice() != choice())
        .unwrap_or(true);

    rsx! {
        document::Link { rel: "stylesheet", href: asset!("/assets/style.css") }
        div { class: "app",
            header { class: "site-header",
                div {
                    h1 { "rgliner" }
                    div { class: "tag",
                        "GLiNER NER & relation extraction — running locally in your browser with WebGPU."
                    }
                }
                a {
                    href: "https://github.com/floneum/floneum",
                    target: "_blank",
                    "GitHub"
                }
            }

            div { class: "tabs",
                button {
                    class: if mode() == Mode::Ner { "tab active" } else { "tab" },
                    onclick: move |_| on_mode_change(Mode::Ner),
                    "NER"
                }
                button {
                    class: if mode() == Mode::Relex { "tab active" } else { "tab" },
                    onclick: move |_| on_mode_change(Mode::Relex),
                    "NER + Relations"
                }
            }

            if let Some(e) = error() {
                div { class: "err-banner", "{e}" }
            }

            div { class: "panel",
                label { "Model" }
                div { class: "row",
                    select {
                        value: "{choice().value()}",
                        onchange: move |ev| {
                            if let Some(c) = ModelChoice::from_value(&ev.value()) {
                                choice.set(c);
                            }
                        },
                        for c in ModelChoice::for_mode(mode()) {
                            option { value: "{c.value()}", "{c.label()}" }
                        }
                    }
                    button {
                        class: "primary",
                        disabled: loading() || running() || (has_model && !model_mismatch),
                        onclick: on_load,
                        if loading() { "Loading…" }
                        else if has_model && !model_mismatch { "Loaded" }
                        else if has_model { "Reload" }
                        else { "Load model" }
                    }
                    span {
                        class: if error().is_some() { "status err" } else if has_model { "status ok" } else { "status" },
                        "{status()}"
                    }
                }
                p { class: "muted",
                    "First load fetches the GGUF weights (60 MB – 500 MB) from HuggingFace and caches them in the browser's Origin Private File System. Subsequent loads are instant."
                }
            }

            div { class: "panel",
                label { "Text" }
                textarea {
                    value: "{text}",
                    oninput: move |e| text.set(e.value()),
                }
            }

            div { class: "panel",
                label { "Entity labels (comma-separated)" }
                input {
                    r#type: "text",
                    value: "{entity_labels}",
                    oninput: move |e| entity_labels.set(e.value()),
                }
                if mode() == Mode::Relex {
                    div { style: "margin-top: 0.75rem;",
                        label { "Relation labels (comma-separated)" }
                        input {
                            r#type: "text",
                            value: "{relation_labels}",
                            oninput: move |e| relation_labels.set(e.value()),
                        }
                    }
                }
            }

            div { class: "panel",
                div { class: "row",
                    button {
                        class: "primary",
                        disabled: !has_model || running() || loading() || model_mismatch,
                        onclick: on_extract,
                        if running() { "Extracting…" } else { "Extract" }
                    }
                    if model_mismatch && has_model {
                        span { class: "status",
                            "Model selection changed — reload to use it."
                        }
                    }
                }
            }

            { render_results(mode(), text(), ner_out(), relex_out()) }
        }
    }
}

fn parse_labels(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

async fn build_model(
    choice: ModelChoice,
) -> Result<LoadedModel, String> {
    match choice {
        ModelChoice::Edge
        | ModelChoice::Small
        | ModelChoice::Base
        | ModelChoice::Large => {
            let source = match choice {
                ModelChoice::Edge => GlinerSource::edge(),
                ModelChoice::Small => GlinerSource::small(),
                ModelChoice::Base => GlinerSource::base(),
                ModelChoice::Large => GlinerSource::large(),
                _ => unreachable!(),
            };
            let inner = Gliner::builder()
                .with_source(source)
                .with_decoding_mode(DecodingMode::Flat)
                .with_threshold(0.05)
                .build_with_loading_handler(|_| {})
                .await
                .map_err(|e| format!("{e}"))?;
            Ok(LoadedModel::Ner { choice, inner })
        }
        ModelChoice::RelexMulti | ModelChoice::RelexBase | ModelChoice::RelexLarge => {
            let source = match choice {
                ModelChoice::RelexMulti => GlinerRelExSource::relex_multi(),
                ModelChoice::RelexBase => GlinerRelExSource::relex_base(),
                ModelChoice::RelexLarge => GlinerRelExSource::relex_large(),
                _ => unreachable!(),
            };
            let inner = GlinerRelEx::builder()
                .with_source(source)
                .build_with_loading_handler(|_| {})
                .await
                .map_err(|e| format!("{e}"))?;
            Ok(LoadedModel::Relex { choice, inner })
        }
    }
}

enum ExtractionOutput {
    Ner(Vec<Entity>),
    Relex(RelexResult),
}

async fn run_extraction(
    model: &mut LoadedModel,
    mode: Mode,
    text: &str,
    entity_labels: &[String],
    relation_labels: &[String],
) -> Result<ExtractionOutput, String> {
    let ent_refs: Vec<&str> = entity_labels.iter().map(|s| s.as_str()).collect();

    if ent_refs.is_empty() {
        return Err("Add at least one entity label.".to_string());
    }

    match (mode, model) {
        (Mode::Ner, LoadedModel::Ner { inner, .. }) => {
            let entities = inner
                .extract_auto(text, &ent_refs, Some(TOKEN_BUDGET))
                .await
                .map_err(|e| format!("{e}"))?;
            Ok(ExtractionOutput::Ner(entities))
        }
        (Mode::Relex, LoadedModel::Relex { inner, .. }) => {
            let rel_refs: Vec<&str> = relation_labels.iter().map(|s| s.as_str()).collect();
            if rel_refs.is_empty() {
                return Err("Add at least one relation label.".to_string());
            }
            let (entities, relations) = inner
                .extract_auto(text, &ent_refs, &rel_refs, Some(TOKEN_BUDGET))
                .await
                .map_err(|e| format!("{e}"))?;
            Ok(ExtractionOutput::Relex(RelexResult {
                entities,
                relations,
            }))
        }
        _ => Err("Loaded model doesn't match the current tab. Reload the model.".to_string()),
    }
}

fn render_results(
    mode: Mode,
    text: String,
    ner_out: Vec<Entity>,
    relex_out: RelexResult,
) -> Element {
    let (entities, relations): (Vec<Entity>, Vec<Relation>) = match mode {
        Mode::Ner => (ner_out, Vec::new()),
        Mode::Relex => (relex_out.entities, relex_out.relations),
    };

    if entities.is_empty() && relations.is_empty() {
        return rsx! {
            div { class: "panel",
                p { class: "muted", "Run extraction to see results." }
            }
        };
    }

    rsx! {
        div { class: "panel",
            div { class: "results",
                { highlighted_text(text.clone(), entities.clone()) }
            }

            if !entities.is_empty() {
                ul { class: "entity-list",
                    for (i, ent) in entities.iter().enumerate() {
                        li { key: "{i}",
                            span { class: "label", style: "color: {hsl_for(&ent.label)};", "{ent.label}" }
                            " · "
                            span { "{ent.text:?}" }
                            " "
                            span { class: "score", "{format_score(ent.score)}" }
                        }
                    }
                }
            }

            if !relations.is_empty() {
                div { style: "margin-top: 1rem;",
                    label { "Relations" }
                    ul { class: "relation-list",
                        for (i, rel) in relations.iter().enumerate() {
                            li { key: "rel-{i}",
                                span { class: "label", "{rel.head.text}" }
                                " --["
                                span { style: "color: var(--accent);", "{rel.relation}" }
                                "]--> "
                                span { class: "label", "{rel.tail.text}" }
                                " "
                                span { class: "score", "{format_score(rel.score)}" }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn highlighted_text(text: String, entities: Vec<Entity>) -> Element {
    let mut sorted = entities.clone();
    sorted.sort_by_key(|e| e.start_char);

    let mut segments: Vec<(bool, String, Option<String>)> = Vec::new();
    let mut cursor = 0usize;
    let len = text.len();
    for ent in sorted.iter() {
        let start = ent.start_char.min(len);
        let end = ent.end_char.min(len);
        if start < cursor || end <= start {
            continue;
        }
        if !text.is_char_boundary(start) || !text.is_char_boundary(end) {
            continue;
        }
        if start > cursor {
            segments.push((false, text[cursor..start].to_string(), None));
        }
        segments.push((true, text[start..end].to_string(), Some(ent.label.clone())));
        cursor = end;
    }
    if cursor < len {
        segments.push((false, text[cursor..].to_string(), None));
    }

    rsx! {
        for (i, (is_entity, content, label)) in segments.into_iter().enumerate() {
            if is_entity {
                {
                    let label_text = label.clone().unwrap_or_default();
                    let color = hsl_for(&label_text);
                    rsx! {
                        span {
                            key: "seg-{i}",
                            class: "entity",
                            style: "background-color: {color};",
                            "{content}"
                            span { class: "chip", "{label_text}" }
                        }
                    }
                }
            } else {
                span { key: "seg-{i}", "{content}" }
            }
        }
    }
}

fn hsl_for(label: &str) -> String {
    let hash: u32 = label
        .bytes()
        .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
    let hue = hash % 360;
    format!("hsl({hue}, 70%, 72%)")
}

fn format_score(score: f32) -> String {
    format!("{:.2}", score)
}
