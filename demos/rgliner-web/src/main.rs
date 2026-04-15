mod components;

use components::input::Input;
use components::label::Label;
use components::select::{
    Select, SelectItemIndicator, SelectList, SelectOption, SelectTrigger, SelectValue,
};
use dioxus::prelude::*;
use dioxus_markdown::Markdown;
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
    let mut schedule = use_signal(|| 0u64);

    // Bump the schedule whenever any input that affects extraction changes.
    // Do NOT read `model` here — the extractor writes it back on every run,
    // which would re-trigger this effect in an endless loop.
    use_effect(move || {
        let _ = text();
        let _ = entity_labels();
        let _ = relation_labels();
        schedule.with_mut(|s| *s += 1);
    });

    // React to schedule changes: debounce, then extract.
    use_effect(move || {
        let current = schedule();
        if current == 0 {
            return;
        }
        spawn(async move {
            sleep_ms(DEBOUNCE_MS).await;
            if schedule() != current {
                return;
            }
            // Wait for any in-flight extraction to finish; bail if a newer change arrives.
            while running() {
                sleep_ms(80).await;
                if schedule() != current {
                    return;
                }
            }
            let Some(mut taken) = model.write().take() else {
                return;
            };
            running.set(true);
            error.set(None);
            status.set("extracting…".to_string());

            let ent_labels = parse_labels(&entity_labels());
            let rel_labels = parse_labels(&relation_labels());
            let cur_text = text();
            let mode = taken.choice().mode();
            let outcome =
                run_extraction(&mut taken, mode, &cur_text, &ent_labels, &rel_labels).await;
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
                    // Kick an extraction now that a model is available.
                    schedule.with_mut(|s| *s += 1);
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

fn render_article(text: &str, ex: &Extraction) -> Element {
    let spliced = splice_entities(text, &ex.entities, &ex.relations);
    rsx! {
        Markdown { src: spliced }
    }
}

/// Splice `<span class="entity">…</span>` into the markdown source at each
/// entity boundary. The nested `.rels` span renders on hover.
fn splice_entities(text: &str, entities: &[Entity], relations: &[Relation]) -> String {
    if entities.is_empty() {
        return text.to_string();
    }
    let mut sorted: Vec<&Entity> = entities.iter().collect();
    sorted.sort_by_key(|e| e.start_char);

    let mut out = String::with_capacity(text.len() + entities.len() * 64);
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
        out.push_str(&text[cursor..start]);
        let color = underline_color(&ent.label);
        out.push_str(&format!(
            r#"<span class="entity" style="--ec: {color}" data-label="{label}">"#,
            color = color,
            label = escape_attr(&ent.label)
        ));
        // The entity surface text.
        out.push_str(&escape_html_content(&text[start..end]));
        // Popover: label + relations involving this entity.
        out.push_str(r#"<span class="pop">"#);
        out.push_str(r#"<span class="pop-label">"#);
        out.push_str(&escape_html_content(&ent.label));
        out.push_str("</span>");

        let ent_text = &text[start..end];
        let mut rel_lines = 0usize;
        for rel in relations {
            if rel.head.text == ent_text || rel.tail.text == ent_text {
                out.push_str(r#"<span class="pop-rel">"#);
                out.push_str(&escape_html_content(&rel.head.text));
                out.push_str(r#"<span class="arrow"> → </span>"#);
                out.push_str(r#"<span class="rel-name">"#);
                out.push_str(&escape_html_content(&rel.relation));
                out.push_str("</span>");
                out.push_str(r#"<span class="arrow"> → </span>"#);
                out.push_str(&escape_html_content(&rel.tail.text));
                out.push_str("</span>");
                rel_lines += 1;
            }
        }
        if rel_lines == 0 && !relations.is_empty() {
            out.push_str(r#"<span class="pop-empty">no relations</span>"#);
        }
        out.push_str("</span>");
        out.push_str("</span>");
        cursor = end;
    }
    if cursor < len {
        out.push_str(&text[cursor..]);
    }
    out
}

fn escape_html_content(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn underline_color(label: &str) -> String {
    let hash: u32 = label
        .bytes()
        .fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u32));
    let hue = hash % 360;
    format!("hsl({hue}, 75%, 42%)")
}
