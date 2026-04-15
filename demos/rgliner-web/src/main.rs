mod components;

use components::input::Input;
use components::label::Label;
use components::select::{
    Select, SelectItemIndicator, SelectList, SelectOption, SelectTrigger, SelectValue,
};
use dioxus::prelude::*;
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use std::iter::Peekable;
use std::ops::Range;
use rgliner::{
    relation_decoding::Relation,
    relex::{GlinerRelEx, GlinerRelExSource},
    DecodingMode, Entity, Gliner, GlinerSource,
};

const TOKEN_BUDGET: usize = 128;
const DEBOUNCE_MS: u32 = 500;

#[cfg(target_arch = "wasm32")]
async fn sleep_ms(ms: u32) {
    gloo_timers::future::TimeoutFuture::new(ms).await;
}

#[cfg(not(target_arch = "wasm32"))]
async fn sleep_ms(_ms: u32) {}

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
            ModelChoice::Edge => "edge · entities · 60M",
            ModelChoice::Small => "small · entities · 108M",
            ModelChoice::Base => "base · entities · 194M",
            ModelChoice::Large => "large · entities · 530M",
            ModelChoice::RelexMulti => "relex-multi · entities + relations",
            ModelChoice::RelexBase => "relex-base · entities + relations · EN",
            ModelChoice::RelexLarge => "relex-large · entities + relations · EN",
        }
    }

    fn mode(self) -> Mode {
        match self {
            ModelChoice::Edge | ModelChoice::Small | ModelChoice::Base | ModelChoice::Large => {
                Mode::Ner
            }
            _ => Mode::Relex,
        }
    }

    fn all() -> &'static [ModelChoice] {
        &[
            ModelChoice::Edge,
            ModelChoice::Small,
            ModelChoice::Base,
            ModelChoice::Large,
            ModelChoice::RelexMulti,
            ModelChoice::RelexBase,
            ModelChoice::RelexLarge,
        ]
    }
}

enum LoadedModel {
    Ner { choice: ModelChoice, inner: Gliner },
    Relex { choice: ModelChoice, inner: GlinerRelEx },
}

impl LoadedModel {
    fn choice(&self) -> ModelChoice {
        match self {
            LoadedModel::Ner { choice, .. } | LoadedModel::Relex { choice, .. } => *choice,
        }
    }
}

#[derive(Clone, Default)]
struct Extraction {
    entities: Vec<Entity>,
    relations: Vec<Relation>,
}

const DEFAULT_TEXT: &str = "# Silicon Valley, Briefly\n\n*Apple Inc.* was founded by **Steve Jobs** in California. **Microsoft** is headquartered in Redmond, and was founded by Bill Gates.\n\nOpenAI operates out of San Francisco.";

#[component]
fn App() -> Element {
    let mut choice = use_signal(|| ModelChoice::Edge);
    let mut text = use_signal(|| DEFAULT_TEXT.to_string());
    let mut entity_labels = use_signal(|| "person, organization, location".to_string());
    let mut relation_labels = use_signal(|| "founded by, located in, headquartered in".to_string());

    let mut model = use_signal(|| None::<LoadedModel>);
    let mut loading = use_signal(|| false);
    let mut running = use_signal(|| false);
    let mut error = use_signal(|| None::<String>);
    let mut extraction = use_signal(Extraction::default);
    let mut status = use_signal(|| "idle".to_string());

    // Memoised so the effect below only re-runs when a model appears or
    // disappears, not every time `run_extraction` writes the model back.
    let model_ready = use_memo(move || model.read().is_some());

    // One extraction pipeline: watch the inputs, debounce, run.
    // `use_future` cancels and restarts whenever any tracked signal changes,
    // which gives us the debounce for free.
    use_future(move || async move {
        let cur_text = text();
        let ent_raw = entity_labels();
        let rel_raw = relation_labels();
        if !model_ready() {
            return;
        }

        // Cancelled if any of the above change during the wait.
        sleep_ms(DEBOUNCE_MS).await;

        // Detach the extraction so cancellation can't drop it mid-run
        // (which would lose the model we've taken out of the signal).
        spawn(async move {
            if running() {
                return;
            }
            let Some(mut taken) = model.write().take() else {
                return;
            };
            running.set(true);
            error.set(None);
            status.set("extracting…".to_string());

            let ent = parse_labels(&ent_raw);
            let rel = parse_labels(&rel_raw);
            let mode = taken.choice().mode();
            let outcome = run_extraction(&mut taken, mode, &cur_text, &ent, &rel).await;
            model.set(Some(taken));

            match outcome {
                Ok(e) => {
                    status.set(format!(
                        "{} entities · {} relations",
                        e.entities.len(),
                        e.relations.len()
                    ));
                    extraction.set(e);
                }
                Err(e) => {
                    error.set(Some(e));
                    status.set("error".to_string());
                }
            }
            running.set(false);
        });
    });

    let on_choice_change = move |v: Option<ModelChoice>| {
        let Some(c) = v else { return };
        if choice() == c {
            return;
        }
        choice.set(c);
        // Unload the current model; user will see a "load" hint.
        *model.write() = None;
        extraction.set(Extraction::default());
    };

    let on_load = move |_| {
        if loading() {
            return;
        }
        let selected = choice();
        loading.set(true);
        error.set(None);
        status.set(format!("loading {}…", selected.label()));
        spawn(async move {
            match build_model(selected).await {
                Ok(m) => {
                    model.set(Some(m));
                    status.set("ready".to_string());
                }
                Err(e) => {
                    error.set(Some(format!("{e}")));
                    status.set("load failed".to_string());
                }
            }
            loading.set(false);
        });
    };

    let has_model = model.read().is_some();
    let model_matches = model
        .read()
        .as_ref()
        .map(|m| m.choice() == choice())
        .unwrap_or(false);

    let current_choice = choice();
    let current_text = text();
    let cur_extraction = extraction();

    rsx! {
        document::Link { rel: "stylesheet", href: asset!("/assets/dx-components-theme.css") }
        document::Link { rel: "stylesheet", href: asset!("/assets/style.css") }
        main { class: "reader",
            header { class: "masthead",
                div { class: "wordmark",
                    span { class: "mark", "rgliner" }
                    span { class: "byline", "a reader for named entities & their relations" }
                }
                div { class: "picker",
                    Select::<ModelChoice> {
                        placeholder: "choose a model",
                        default_value: current_choice,
                        on_value_change: on_choice_change,
                        SelectTrigger { aria_label: "Model", SelectValue {} }
                        SelectList { aria_label: "Models",
                            for (i, c) in ModelChoice::all().iter().copied().enumerate() {
                                SelectOption::<ModelChoice> {
                                    index: i,
                                    value: c,
                                    text_value: c.label().to_string(),
                                    "{c.label()}"
                                    SelectItemIndicator {}
                                }
                            }
                        }
                    }
                    button {
                        class: "load",
                        disabled: loading() || (has_model && model_matches),
                        onclick: on_load,
                        if loading() {
                            "loading…"
                        } else if has_model && model_matches {
                            "loaded"
                        } else if has_model {
                            "reload"
                        } else {
                            "load"
                        }
                    }
                }
            }

            details { class: "settings",
                summary { "labels" }
                div { class: "settings-body",
                    Label { html_for: "entity-labels", "entities" }
                    Input {
                        id: "entity-labels",
                        r#type: "text",
                        value: "{entity_labels}",
                        oninput: move |e: FormEvent| entity_labels.set(e.value()),
                    }
                    if current_choice.mode() == Mode::Relex {
                        div { style: "margin-top: 0.75rem;",
                            Label { html_for: "relation-labels", "relations" }
                            Input {
                                id: "relation-labels",
                                r#type: "text",
                                value: "{relation_labels}",
                                oninput: move |e: FormEvent| relation_labels.set(e.value()),
                            }
                        }
                    }
                }
            }

            div { class: "status-line",
                span { class: "dot", class: if running() || loading() { "busy" } else if error().is_some() { "err" } else if has_model { "ok" } else { "idle" } }
                span { class: "msg", "{status()}" }
                if let Some(e) = error() {
                    span { class: "err-text", " · {e}" }
                }
            }

            div { class: "split",
                textarea {
                    class: "editor",
                    spellcheck: "false",
                    value: "{current_text}",
                    oninput: move |e: FormEvent| text.set(e.value()),
                }
                article { class: "article",
                    if has_model {
                        { render_article(&current_text, &cur_extraction) }
                    } else {
                        div { class: "placeholder",
                            "load a model to begin reading."
                        }
                    }
                }
            }
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

async fn build_model(choice: ModelChoice) -> Result<LoadedModel, String> {
    match choice {
        ModelChoice::Edge | ModelChoice::Small | ModelChoice::Base | ModelChoice::Large => {
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

async fn run_extraction(
    model: &mut LoadedModel,
    mode: Mode,
    text: &str,
    entity_labels: &[String],
    relation_labels: &[String],
) -> Result<Extraction, String> {
    if entity_labels.is_empty() {
        return Err("add at least one entity label".to_string());
    }
    let ent_refs: Vec<&str> = entity_labels.iter().map(|s| s.as_str()).collect();

    match (mode, model) {
        (Mode::Ner, LoadedModel::Ner { inner, .. }) => {
            let entities = inner
                .extract_auto(text, &ent_refs, Some(TOKEN_BUDGET))
                .await
                .map_err(|e| format!("{e}"))?;
            Ok(Extraction {
                entities,
                relations: Vec::new(),
            })
        }
        (Mode::Relex, LoadedModel::Relex { inner, .. }) => {
            if relation_labels.is_empty() {
                return Err("add at least one relation label".to_string());
            }
            let rel_refs: Vec<&str> = relation_labels.iter().map(|s| s.as_str()).collect();
            let (entities, relations) = inner
                .extract_auto(text, &ent_refs, &rel_refs, Some(TOKEN_BUDGET))
                .await
                .map_err(|e| format!("{e}"))?;
            Ok(Extraction {
                entities,
                relations,
            })
        }
        _ => Err("loaded model doesn't match the selection".to_string()),
    }
}

#[derive(Clone, PartialEq)]
struct EntityView {
    text: String,
    label: String,
    color: String,
    rels: Vec<(String, String, String)>,
}

#[derive(Clone, PartialEq, Default)]
struct Article {
    source: String,
    entities: Vec<EntityView>,
}

/// Splice `<Entity i="N"/>` markers into the markdown source at each entity
/// boundary, and collect the per-entity view the custom component renders.
fn build_article(text: &str, ex: &Extraction) -> Article {
    if ex.entities.is_empty() {
        return Article {
            source: text.to_string(),
            entities: Vec::new(),
        };
    }
    let mut sorted: Vec<&Entity> = ex.entities.iter().collect();
    sorted.sort_by_key(|e| e.start_char);

    let mut source = String::with_capacity(text.len() + sorted.len() * 20);
    let mut entities: Vec<EntityView> = Vec::new();
    let mut cursor = 0usize;
    let len = text.len();

    for ent in sorted {
        let start = ent.start_char.min(len);
        let end = ent.end_char.min(len);
        if start < cursor || end <= start {
            continue;
        }
        if !text.is_char_boundary(start) || !text.is_char_boundary(end) {
            continue;
        }
        source.push_str(&text[cursor..start]);
        let idx = entities.len();
        source.push_str(&format!("<Entity i=\"{idx}\"/>"));

        let ent_text = &text[start..end];
        let rels: Vec<(String, String, String)> = ex
            .relations
            .iter()
            .filter(|r| r.head.text == ent_text || r.tail.text == ent_text)
            .map(|r| (r.head.text.clone(), r.relation.clone(), r.tail.text.clone()))
            .collect();

        entities.push(EntityView {
            text: ent_text.to_string(),
            label: ent.label.clone(),
            color: underline_color(&ent.label),
            rels,
        });
        cursor = end;
    }
    if cursor < len {
        source.push_str(&text[cursor..]);
    }
    Article { source, entities }
}

fn render_entity(view: EntityView) -> Element {
    let has_rels = !view.rels.is_empty();
    rsx! {
        span {
            class: "entity",
            style: "--ec: {view.color};",
            "{view.text}"
            span { class: "pop",
                span { class: "pop-label", "{view.label}" }
                if has_rels {
                    for (i, (head, rel, tail)) in view.rels.iter().cloned().enumerate() {
                        span { key: "r-{i}", class: "pop-rel",
                            "{head}"
                            span { class: "arrow", " → " }
                            span { class: "rel-name", "{rel}" }
                            span { class: "arrow", " → " }
                            "{tail}"
                        }
                    }
                }
            }
        }
    }
}

fn underline_color(label: &str) -> String {
    let hash: u32 = label
        .bytes()
        .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
    let hue = hash % 360;
    format!("hsl({hue}, 75%, 42%)")
}
