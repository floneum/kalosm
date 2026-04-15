mod components;

use components::input::Input;
use components::label::Label;
use components::select::{
    Select, SelectItemIndicator, SelectList, SelectOption, SelectTrigger, SelectValue,
};
use dioxus::prelude::*;
use pulldown_cmark::{Event, HeadingLevel, Options, Parser, Tag};
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
    // `use_resource` re-runs the async body whenever any tracked signal
    // changes, cancelling the previous invocation — the debounce falls
    // out of that cancellation behavior.
    use_resource(move || async move {
        let cur_text = text();
        let ent_raw = entity_labels();
        let rel_raw = relation_labels();
        let ready = model_ready();
        tracing::info!("pipeline tick: ready={ready} text_len={}", cur_text.len());
        if !ready {
            return;
        }

        // Cancelled if any of the above change during the wait.
        sleep_ms(DEBOUNCE_MS).await;
        tracing::info!("pipeline: debounce survived, kicking extraction");

        // Detach the extraction so cancellation can't drop it mid-run
        // (which would lose the model we've taken out of the signal).
        spawn(async move {
            if running() {
                tracing::info!("extraction already running, skipping");
                return;
            }
            let Some(mut taken) = model.write().take() else {
                tracing::warn!("no model in slot at extract-time");
                return;
            };
            running.set(true);
            error.set(None);
            status.set("extracting…".to_string());

            let ent = parse_labels(&ent_raw);
            let rel = parse_labels(&rel_raw);
            let mode = taken.choice().mode();
            let mode_name = match mode {
                Mode::Ner => "ner",
                Mode::Relex => "relex",
            };
            tracing::info!("extracting: mode={mode_name} ent={} rel={}", ent.len(), rel.len());
            let outcome = run_extraction(&mut taken, mode, &cur_text, &ent, &rel).await;
            model.set(Some(taken));

            match outcome {
                Ok(e) => {
                    tracing::info!(
                        "extraction done: {} entities, {} relations",
                        e.entities.len(),
                        e.relations.len()
                    );
                    status.set(format!(
                        "{} entities · {} relations",
                        e.entities.len(),
                        e.relations.len()
                    ));
                    extraction.set(e);
                }
                Err(e) => {
                    tracing::warn!("extraction error: {e}");
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
struct EntitySpan {
    byte_start: usize,
    byte_end: usize,
    label: String,
    color: String,
    rels: Vec<(String, String, String)>,
}

fn collect_entity_spans(text: &str, ex: &Extraction) -> Vec<EntitySpan> {
    let mut sorted: Vec<&Entity> = ex.entities.iter().collect();
    sorted.sort_by_key(|e| e.start_char);

    let mut spans = Vec::new();
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
        let ent_text = &text[start..end];
        let rels: Vec<(String, String, String)> = ex
            .relations
            .iter()
            .filter(|r| r.head.text == ent_text || r.tail.text == ent_text)
            .map(|r| (r.head.text.clone(), r.relation.clone(), r.tail.text.clone()))
            .collect();
        spans.push(EntitySpan {
            byte_start: start,
            byte_end: end,
            label: ent.label.clone(),
            color: underline_color(&ent.label),
            rels,
        });
        cursor = end;
    }
    spans
}

/// Render the markdown article, wrapping entity byte-ranges in interactive
/// spans at the text-event level. We walk pulldown-cmark's flat event stream
/// and build rsx! directly — no HTML serialisation, no custom-component dance.
fn render_article(text: &str, ex: &Extraction) -> Element {
    let spans = collect_entity_spans(text, ex);
    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TABLES);
    let parser = Parser::new_ext(text, opts).into_offset_iter();
    let mut r = Renderer {
        source: text,
        spans: &spans,
        events: parser.peekable(),
    };
    let nodes = r.render_until_end(false);
    rsx! { {nodes.into_iter()} }
}

struct Renderer<'a, I: Iterator<Item = (Event<'a>, Range<usize>)>> {
    source: &'a str,
    spans: &'a [EntitySpan],
    events: Peekable<I>,
}

impl<'a, I: Iterator<Item = (Event<'a>, Range<usize>)>> Renderer<'a, I> {
    /// Consume events until `End(_)` (if `scoped`) or EOF, returning the
    /// rsx! elements they expand to.
    fn render_until_end(&mut self, scoped: bool) -> Vec<Element> {
        let mut nodes: Vec<Element> = Vec::new();
        while let Some((event, range)) = self.events.next() {
            match event {
                Event::Start(tag) => nodes.push(self.render_tag(tag)),
                Event::End(_) if scoped => return nodes,
                Event::End(_) => continue,
                Event::Text(_) => nodes.push(self.render_text_range(range)),
                Event::Code(s) => {
                    let s = s.to_string();
                    nodes.push(rsx! { code { "{s}" } });
                }
                Event::SoftBreak => nodes.push(rsx! { " " }),
                Event::HardBreak => nodes.push(rsx! { br {} }),
                Event::Rule => nodes.push(rsx! { hr {} }),
                Event::Html(s) | Event::InlineHtml(s) => {
                    // Treat raw HTML as literal text — safer than dangerous_inner_html.
                    let s = s.to_string();
                    nodes.push(rsx! { "{s}" });
                }
                Event::FootnoteReference(_) => {}
                Event::TaskListMarker(done) => nodes.push(rsx! {
                    input { r#type: "checkbox", checked: done, disabled: true }
                }),
                _ => {}
            }
        }
        nodes
    }

    fn render_tag(&mut self, tag: Tag<'a>) -> Element {
        match tag {
            Tag::Paragraph => {
                let children = self.render_until_end(true);
                rsx! { p { {children.into_iter()} } }
            }
            Tag::Heading { level, .. } => {
                let children = self.render_until_end(true);
                match level {
                    HeadingLevel::H1 => rsx! { h1 { {children.into_iter()} } },
                    HeadingLevel::H2 => rsx! { h2 { {children.into_iter()} } },
                    HeadingLevel::H3 => rsx! { h3 { {children.into_iter()} } },
                    HeadingLevel::H4 => rsx! { h4 { {children.into_iter()} } },
                    HeadingLevel::H5 => rsx! { h5 { {children.into_iter()} } },
                    HeadingLevel::H6 => rsx! { h6 { {children.into_iter()} } },
                }
            }
            Tag::BlockQuote(_) => {
                let children = self.render_until_end(true);
                rsx! { blockquote { {children.into_iter()} } }
            }
            Tag::CodeBlock(_) => {
                let children = self.render_until_end(true);
                rsx! { pre { code { {children.into_iter()} } } }
            }
            Tag::List(Some(_start)) => {
                let children = self.render_until_end(true);
                rsx! { ol { {children.into_iter()} } }
            }
            Tag::List(None) => {
                let children = self.render_until_end(true);
                rsx! { ul { {children.into_iter()} } }
            }
            Tag::Item => {
                let children = self.render_until_end(true);
                rsx! { li { {children.into_iter()} } }
            }
            Tag::Emphasis => {
                let children = self.render_until_end(true);
                rsx! { em { {children.into_iter()} } }
            }
            Tag::Strong => {
                let children = self.render_until_end(true);
                rsx! { strong { {children.into_iter()} } }
            }
            Tag::Strikethrough => {
                let children = self.render_until_end(true);
                rsx! { s { {children.into_iter()} } }
            }
            Tag::Link { dest_url, title, .. } => {
                let children = self.render_until_end(true);
                let href = dest_url.to_string();
                let title = title.to_string();
                rsx! { a { href: "{href}", title: "{title}", {children.into_iter()} } }
            }
            Tag::Image { dest_url, title, .. } => {
                // Consume inner events (alt text) without rendering them separately.
                let _ = self.render_until_end(true);
                let src = dest_url.to_string();
                let title = title.to_string();
                rsx! { img { src: "{src}", title: "{title}" } }
            }
            _ => {
                let children = self.render_until_end(true);
                rsx! { span { {children.into_iter()} } }
            }
        }
    }

    /// Render the text at `range`, splitting on any entity spans that overlap it.
    fn render_text_range(&self, range: Range<usize>) -> Element {
        let mut parts: Vec<Element> = Vec::new();
        let mut cursor = range.start;
        let slice_end = range.end;

        for span in self.spans.iter() {
            if span.byte_end <= cursor {
                continue;
            }
            if span.byte_start >= slice_end {
                break;
            }
            let s = span.byte_start.max(cursor);
            let e = span.byte_end.min(slice_end);
            if s > cursor {
                let plain = self.source[cursor..s].to_string();
                parts.push(rsx! { "{plain}" });
            }
            let ent_text = self.source[s..e].to_string();
            parts.push(render_entity(&ent_text, span));
            cursor = e;
        }
        if cursor < slice_end {
            let tail = self.source[cursor..slice_end].to_string();
            parts.push(rsx! { "{tail}" });
        }
        rsx! { {parts.into_iter()} }
    }
}

fn render_entity(text: &str, span: &EntitySpan) -> Element {
    let text = text.to_string();
    let label = span.label.clone();
    let color = span.color.clone();
    let rels = span.rels.clone();
    let has_rels = !rels.is_empty();
    rsx! {
        span {
            class: "entity",
            style: "--ec: {color};",
            "{text}"
            span { class: "pop",
                span { class: "pop-label", "{label}" }
                if has_rels {
                    for (i, (head, rel, tail)) in rels.into_iter().enumerate() {
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
