//! The Train route: watch a transformer learn to write, one character at a
//! time.

use std::cell::RefCell;
use std::rc::Rc;

use dioxus::prelude::*;
use web_time::Instant;

use super::corpus::Corpus;
use super::paint;
use super::model::{
    Attention, BLOCKS, CONTEXT, Evaluation, HEADS, Lm, StepStats, TOKENS, parameter_count_for,
};
use crate::components::badge::{Badge, BadgeVariant};
use crate::components::button::{Button, ButtonSize, ButtonVariant};
use crate::components::card::{Card, CardContent, CardDescription, CardHeader, CardTitle};

/// Points the loss curve draws. The history itself is never truncated — a
/// curve whose beginning has scrolled off cannot show what training did,
/// which is the only thing it is for. Longer runs are decimated at render.
const CURVE_POINTS: usize = 320;
/// Optimizer steps queued between host syncs.
///
/// A readback completes on the browser's event loop, so one per step caps
/// training at the frame rate no matter how little work the GPU has.
const STEPS_PER_SYNC: usize = 8;
/// Syncs between held-out evaluations.
const SYNCS_PER_EVAL: usize = 8;
/// Held-out batches averaged per evaluation. Each is one readback, and in a
/// browser a readback completes on the event loop — so this is frames, not
/// arithmetic.
const EVAL_BATCHES: usize = 2;
/// Syncs between re-samples of the live continuation.
const SYNCS_PER_SAMPLE: usize = 24;
/// Characters the live sample writes while training runs. Short on purpose:
/// generation is sequential, so every character is its own dispatch and
/// readback — a frame each in a browser — and a long sample starves the
/// optimizer for a second at a time. The Write button is where a long one
/// belongs.
const LIVE_CHARS: usize = 48;
/// Characters the Write button produces.
const FULL_CHARS: usize = 400;
/// What the demo opens with, and what Reset returns to.
const DEFAULT_PROMPT: &str = "Once upon a time, there was a little girl named";
/// Sampling temperature the demo opens at.
const DEFAULT_TEMPERATURE: f32 = 0.8;
/// Evaluations without a new best held-out loss before training stops.
///
/// Patience rather than a target: a target you never reach never stops, and
/// "it stopped getting better" is the honest reason to halt.
const PATIENCE: usize = 10;
/// Held-out gain in nats that counts as improvement rather than noise.
const MIN_GAIN: f32 = 0.004;

/// A model taken out of its cell for the duration of one run.
///
/// Parking it again on drop is what makes cancellation harmless: whether the
/// run ends normally or the route re-renders the resource away mid-step, the
/// weights and the compiled kernels are back in the cell for the next click.
struct Parked {
    garage: Rc<RefCell<Option<(u32, Lm)>>>,
    seed: u32,
    model: Option<Lm>,
}

impl Parked {
    /// Drive out the model built for `seed`, discarding one built for any
    /// other seed — that is what Reset does.
    fn take(garage: &Rc<RefCell<Option<(u32, Lm)>>>, seed: u32) -> Self {
        let model = garage
            .borrow_mut()
            .take()
            .filter(|(built, _)| *built == seed)
            .map(|(_, model)| model);
        Self {
            garage: garage.clone(),
            seed,
            model,
        }
    }
}

impl Drop for Parked {
    fn drop(&mut self) {
        if let Some(model) = self.model.take() {
            *self.garage.borrow_mut() = Some((self.seed, model));
        }
    }
}

/// Where the loop's wall time goes, and what it costs in dispatches.
///
/// A demo about a compiler should say what it is spending. Training, held-out
/// scoring and writing a sample are three different costs, and lumping them
/// into one "steps per second" hides which one a change actually moved.
#[derive(Clone, Copy, Default, PartialEq)]
struct Cost {
    train: f32,
    eval: f32,
    sample: f32,
    steps: u64,
    dispatches: u64,
    window_steps: u64,
}

impl Cost {
    /// Milliseconds of training per optimizer step.
    fn ms_per_step(&self) -> f32 {
        if self.steps == 0 { 0.0 } else { self.train / self.steps as f32 }
    }

    /// Share of wall time not spent training.
    fn overhead(&self) -> f32 {
        let total = self.train + self.eval + self.sample;
        if total <= 0.0 { 0.0 } else { (self.eval + self.sample) / total }
    }

    /// GPU dispatches per optimizer step.
    fn per_step(&self) -> f32 {
        if self.window_steps == 0 {
            0.0
        } else {
            self.dispatches as f32 / self.window_steps as f32
        }
    }
}

/// What the interpretability panel is currently showing.
#[derive(Clone, PartialEq)]
struct Insight {
    /// The text the maps are about.
    context: String,
    /// `[BLOCKS][HEADS]` maps of `CONTEXT * CONTEXT` probabilities.
    maps: Vec<Vec<Vec<f32>>>,
    /// Real positions in the window; the rest are left padding.
    filled: usize,
    /// Largest logit gap between the model and the lens the maps come from.
    disagreement: f32,
    /// The next-character distribution at the end of `context`.
    next: Vec<f32>,
    /// `vocab * vocab` cosine similarities between token embeddings.
    similarity: Vec<f32>,
}

#[component]
pub fn Train() -> Element {
    let corpus = use_hook(|| Rc::new(Corpus::load()));
    let vocab = corpus.vocab_size();

    let mut prompt = use_signal(|| DEFAULT_PROMPT.to_string());
    let mut temperature = use_signal(|| DEFAULT_TEMPERATURE);
    let mut sample = use_signal(String::new);
    let mut stats = use_signal(|| None::<StepStats>);
    let mut evaluation = use_signal(|| None::<Evaluation>);
    let mut history = use_signal(Vec::<(u64, f32, Option<f32>)>::new);
    let mut rate = use_signal(|| 0.0f32);
    let mut cost = use_signal(Cost::default);
    let mut running = use_signal(|| false);
    let mut status = use_signal(String::new);
    let mut run_id = use_signal(|| 0usize);
    let mut seed = use_signal(|| 0x51ed_c0deu32);
    let mut early_stop = use_signal(|| true);
    let mut insight = use_signal(|| None::<Insight>);
    // What the next turn of the loop should do besides train.
    let mut want_inspect = use_signal(|| false);
    let mut want_write = use_signal(|| false);

    let garage: Rc<RefCell<Option<(u32, Lm)>>> = use_hook(|| Rc::new(RefCell::new(None)));

    let _training = use_resource({
        let garage = garage.clone();
        let corpus = corpus.clone();
        move || {
            let garage = garage.clone();
            let corpus = corpus.clone();
            async move {
                if run_id() == 0 {
                    return;
                }
                // The model is driven out of the cell for the whole run and
                // parked again by `Parked::drop`, so no borrow is ever live
                // across an await — including the one where this future is
                // cancelled out from under us.
                let wanted = *seed.peek();
                let mut parked = Parked::take(&garage, wanted);
                if parked.model.is_none() {
                    status.set("Requesting an adapter and compiling kernels…".into());
                    match Lm::new(corpus.vocab_size(), wanted).await {
                        Ok(mut built) => {
                            if let Err(error) = built.compile_training(Default::default()).await {
                                status.set(error.to_string());
                                running.set(false);
                                return;
                            }
                            parked.model = Some(built);
                        }
                        Err(error) => {
                            status.set(error.to_string());
                            running.set(false);
                            return;
                        }
                    }
                }
                let Some(model) = parked.model.as_mut() else {
                    return;
                };

                // The loop owns the model, so it is also what answers a
                // one-off request: both buttons bump the same `run_id`, so
                // they work whether or not training is going.
                if *want_write.peek() {
                    want_write.set(false);
                    status.set("Writing…".into());
                    let seeded = corpus.encode_all(&prompt.peek().clone());
                    let hot = *temperature.peek();
                    match model.generate(&corpus, &seeded, FULL_CHARS, hot).await {
                        Ok(text) => sample.set(text),
                        Err(error) => status.set(error.to_string()),
                    }
                    status.set(String::new());
                }
                if *want_inspect.peek() {
                    want_inspect.set(false);
                    status.set("Reading the model…".into());
                    let text = prompt.peek().clone();
                    match look_inside(model, &corpus, &text, *temperature.peek()).await {
                        Ok(found) => insight.set(Some(found)),
                        Err(error) => status.set(error.to_string()),
                    }
                    status.set(String::new());
                }

                // `peek` throughout: reading these reactively would re-run the
                // resource — and so restart the loop — every time a step
                // writes a statistic.
                let mut window = (Instant::now(), model.step_count(), model.dispatch_count());
                let mut since_eval = 0usize;
                let mut since_sample = SYNCS_PER_SAMPLE;
                let (mut best, mut stale) = (f32::INFINITY, 0usize);
                if !*running.peek() {
                    return;
                }
                status.set("Training".into());
                while *running.peek() {
                    let before = model.step_count();
                    let at = Instant::now();
                    let outcome = model.train(&corpus, STEPS_PER_SYNC).await;
                    {
                        let mut c = cost.write();
                        c.train += at.elapsed().as_secs_f32() * 1000.0;
                        c.steps += model.step_count() - before;
                    }
                    match outcome {
                        Ok(step) => {
                            history.write().push((step.step, step.loss, None));
                            stats.set(Some(step));
                        }
                        Err(error) => {
                            status.set(error.to_string());
                            break;
                        }
                    }

                    // Held-out text is scored by the same graph resolved for
                    // the loss alone — no update root is asked for, so
                    // measuring cannot train.
                    since_eval += 1;
                    if since_eval >= SYNCS_PER_EVAL {
                        since_eval = 0;
                        let at = Instant::now();
                        let scored = model.evaluate(&corpus, EVAL_BATCHES).await;
                        cost.write().eval += at.elapsed().as_secs_f32() * 1000.0;
                        match scored {
                            Ok(scored) => {
                                if let Some(last) = history.write().last_mut() {
                                    last.2 = Some(scored.loss);
                                }
                                // Stop on the held-out loss, never on the
                                // training loss, which keeps falling long
                                // after the model stops learning anything
                                // transferable.
                                if *early_stop.peek() {
                                    if scored.loss < best - MIN_GAIN {
                                        best = scored.loss;
                                        stale = 0;
                                    } else {
                                        stale += 1;
                                    }
                                }
                                evaluation.set(Some(scored));
                            }
                            Err(error) => status.set(error.to_string()),
                        }
                        if stale >= PATIENCE {
                            running.set(false);
                            status.set(format!(
                                "Stopped: held-out loss bottomed out at {best:.3} nats \
                                 (perplexity {:.2}) and stopped improving.",
                                best.exp(),
                            ));
                            break;
                        }
                    }

                    // The live continuation costs one dispatch per character,
                    // so it is re-sampled every so often rather than every
                    // sync — otherwise the demo would spend its time writing
                    // instead of learning.
                    since_sample += 1;
                    if since_sample >= SYNCS_PER_SAMPLE {
                        since_sample = 0;
                        let seeded = corpus.encode_all(&prompt.peek().clone());
                        let hot = *temperature.peek();
                        let at = Instant::now();
                        let written = model.generate(&corpus, &seeded, LIVE_CHARS, hot).await;
                        cost.write().sample += at.elapsed().as_secs_f32() * 1000.0;
                        if let Ok(text) = written {
                            sample.set(text);
                        }
                    }

                    // Steps are far too quick to time one at a time; the rate
                    // is read off a window of at least a quarter second.
                    let elapsed = window.0.elapsed().as_secs_f32();
                    if elapsed > 0.5 {
                        let steps = model.step_count() - window.1;
                        rate.set(steps as f32 / elapsed);
                        let mut c = cost.write();
                        c.dispatches = model.dispatch_count() - window.2;
                        c.window_steps = steps;
                        window = (Instant::now(), model.step_count(), model.dispatch_count());
                    }
                }
                running.set(false);
                if status.peek().as_str() == "Training" {
                    status.set("Paused".into());
                }
            }
        }
    });

    let excerpt = use_hook({
        let corpus = corpus.clone();
        move || corpus.excerpt(560)
    });
    let alphabet: Vec<char> = corpus.alphabet().to_vec();
    let parameters = Thousands(parameter_count_for(vocab) as u64);
    let step = stats.read().map_or(0, |s| s.step);
    let tokens_seen = step * TOKENS as u64;

    rsx! {
        div { class: "lm-page",
            header { class: "lm-hero",
                h1 { "A transformer that learns to write, in your browser" }
                p { class: "lm-hero-sub",
                    "A {parameters} parameter character-level language model, initialized from "
                    "noise and trained from scratch on WebGPU — no pretrained weights, nothing "
                    "downloaded but the text. Forward, backward and the Adam update are one "
                    "fusor graph, built once and re-run every step."
                }
                div { class: "lm-hero-facts",
                    Fact { value: "{BLOCKS}", label: "blocks" }
                    Fact { value: "{HEADS}", label: "heads" }
                    Fact { value: "{CONTEXT}", label: "context" }
                    Fact { value: "{vocab}", label: "characters" }
                }
            }

            section { class: "lm-controls",
                Button {
                    variant: if running() { ButtonVariant::Secondary } else { ButtonVariant::Primary },
                    onclick: move |_| {
                        let next = !running();
                        running.set(next);
                        if next {
                            run_id += 1;
                        }
                    },
                    if running() { "Pause" } else if step > 0 { "Resume" } else { "Start training" }
                }
                Button {
                    variant: ButtonVariant::Outline,
                    disabled: running(),
                    onclick: move |_| {
                        seed.set(seed() ^ 0x9e37_79b9);
                        history.write().clear();
                        cost.set(Cost::default());
                        stats.set(None);
                        evaluation.set(None);
                        insight.set(None);
                        sample.set(String::new());
                        rate.set(0.0);
                        status.set("Reset — the weights are noise again.".into());
                    },
                    "Reset"
                }
                label { class: "lm-toggle",
                    input {
                        r#type: "checkbox",
                        checked: early_stop(),
                        onchange: move |e| early_stop.set(e.checked()),
                    }
                    "Stop when held-out loss stops improving"
                }
                if !status.read().is_empty() {
                    Badge {
                        variant: if running() { BadgeVariant::Primary } else { BadgeVariant::Secondary },
                        "{status}"
                    }
                }
            }

            section { class: "lm-grid",
                Card { class: "lm-card",
                    CardHeader {
                        CardTitle { "Training" }
                        CardDescription {
                            "Cross-entropy in nats per character. The dashed line is held-out "
                            "text the optimizer never sees."
                        }
                    }
                    CardContent {
                        LossCurve { points: history() }
                        div { class: "lm-metrics",
                            Metric {
                                value: format!("{step}"),
                                label: "steps",
                            }
                            Metric {
                                value: format!("{:.0}/s", rate()),
                                label: "step rate",
                            }
                            Metric {
                                value: stats.read().map_or("—".into(), |s| format!("{:.3}", s.loss)),
                                label: "train loss",
                            }
                            Metric {
                                value: stats.read().map_or("—".into(), |s| format!("{:.2}", s.perplexity())),
                                label: "train perplexity",
                            }
                            Metric {
                                value: evaluation.read().map_or("—".into(), |e| format!("{:.3}", e.loss)),
                                label: "held-out loss",
                            }
                            Metric {
                                value: evaluation.read().map_or("—".into(), |e| format!("{:.2}", e.perplexity())),
                                label: "held-out perplexity",
                            }
                            Metric {
                                value: evaluation.read().map_or("—".into(), |e| format!("{:.0}%", e.accuracy * 100.0)),
                                label: "next-char top-1",
                            }
                            Metric {
                                value: format!("{}", Thousands(tokens_seen)),
                                label: "characters seen",
                            }
                            Metric {
                                value: stats.read().map_or("—".into(), |s| format!("{:.1e}", s.rate)),
                                label: "step size",
                            }
                        }
                        div { class: "lm-metrics",
                            Metric {
                                value: format!("{:.1} ms", cost.read().ms_per_step()),
                                label: "per step",
                            }
                            Metric {
                                value: format!("{:.0}", cost.read().per_step()),
                                label: "dispatches/step",
                            }
                            Metric {
                                value: format!("{:.0}%", cost.read().overhead() * 100.0),
                                label: "spent not training",
                            }
                        }
                        p { class: "lm-note",
                            "Perplexity is how many characters the model is effectively choosing "
                            "between at each position. Untrained, that is all {vocab} of them. "
                            "The second row is what the loop costs: training time per step, the "
                            "GPU dispatches each step issues — which is what a browser pays for "
                            "and a native run barely notices — and the share of wall time going "
                            "to held-out scoring and writing samples rather than to learning."
                        }
                    }
                }

                Card { class: "lm-card",
                    CardHeader {
                        CardTitle { "What it writes" }
                        CardDescription {
                            "Sampled one character at a time from the model's own output, fed "
                            "back into its context."
                        }
                    }
                    CardContent {
                        label { class: "lm-field",
                            span { "Prompt" }
                            textarea {
                                rows: "2",
                                value: "{prompt}",
                                oninput: move |e| prompt.set(e.value()),
                            }
                        }
                        label { class: "lm-field",
                            span { "Temperature {temperature():.2}" }
                            input {
                                r#type: "range",
                                min: "0.1",
                                max: "1.5",
                                step: "0.05",
                                value: "{temperature}",
                                oninput: move |e| {
                                    if let Ok(v) = e.value().parse::<f32>() {
                                        temperature.set(v);
                                    }
                                },
                            }
                        }
                        div { class: "lm-sample",
                            span { class: "lm-sample-prompt", "{prompt}" }
                            if sample.read().is_empty() {
                                span { class: "lm-sample-empty",
                                    "…press Start, or Write, and the continuation appears here."
                                }
                            } else {
                                span { "{sample}" }
                            }
                        }
                        div { class: "lm-row",
                            Button {
                                variant: ButtonVariant::Outline,
                                size: ButtonSize::Sm,
                                onclick: move |_| {
                                    want_write.set(true);
                                    run_id += 1;
                                },
                                "Write {FULL_CHARS} characters"
                            }
                            span { class: "lm-hint",
                                "Low temperature repeats itself; high temperature invents words."
                            }
                        }
                    }
                }
            }

            Card { class: "lm-card",
                CardHeader {
                    CardTitle { "What it is learning from" }
                    CardDescription {
                        "A {Thousands(corpus.len() as u64)} character slice of TinyStories, "
                        "embedded in the page. The last tenth is held out."
                    }
                }
                CardContent {
                    pre { class: "lm-corpus", "{excerpt}…" }
                    div { class: "lm-alphabet",
                        for c in alphabet.iter() {
                            span { class: "lm-char", "{printable(*c)}" }
                        }
                    }
                }
            }

            Card { class: "lm-card",
                CardHeader {
                    CardTitle { "Inside the model" }
                    CardDescription {
                        "Attention maps, the next-character distribution, and which characters "
                        "the embedding has learned to treat alike — all read off the very same "
                        "parameters that are training."
                    }
                }
                CardContent {
                    div { class: "lm-row",
                        Button {
                            variant: ButtonVariant::Outline,
                            size: ButtonSize::Sm,
                            onclick: move |_| {
                                want_inspect.set(true);
                                run_id += 1;
                            },
                            "Read the model"
                        }
                        span { class: "lm-hint", "Reads whatever is currently in the prompt box." }
                    }
                    if let Some(found) = insight.read().as_ref() {
                        Inside { insight: found.clone(), alphabet: alphabet.clone() }
                    } else {
                        p { class: "lm-note",
                            "Nothing read yet. The panels below are computed on demand, because "
                            "every one of them costs a dispatch the optimizer would rather have."
                        }
                    }
                }
            }
        }
    }
}

/// Everything the interpretability card draws once a read has happened.
#[component]
fn Inside(insight: Insight, alphabet: Vec<char>) -> Element {
    let context: Vec<char> = insight.context.chars().collect();
    let shown = context.len().min(insight.filled);
    let tail = context.len().saturating_sub(shown);

    // Rank the next-character distribution once, here, rather than sorting
    // inside the render loop.
    let mut ranked: Vec<(usize, f32)> = insight.next.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let top: Vec<(char, f32)> = ranked
        .iter()
        .take(10)
        .map(|(i, p)| (alphabet.get(*i).copied().unwrap_or('?'), *p))
        .collect();

    // The five characters each character's embedding is closest to. Its own
    // row is dropped: everything is most similar to itself.
    let v = alphabet.len();
    let neighbours: Vec<(char, Vec<char>)> = (0..v)
        .map(|i| {
            let mut row: Vec<(usize, f32)> = insight.similarity[i * v..(i + 1) * v]
                .iter()
                .copied()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .collect();
            row.sort_by(|a, b| b.1.total_cmp(&a.1));
            (
                alphabet[i],
                row.iter()
                    .take(5)
                    .map(|(j, _)| alphabet[*j])
                    .collect::<Vec<char>>(),
            )
        })
        .collect();

    rsx! {
        div { class: "lm-lens-check",
            if insight.disagreement < 1e-2 {
                Badge { variant: BadgeVariant::Secondary,
                    "lens agrees with the model to {insight.disagreement:.1e} logits"
                }
            } else {
                Badge { variant: BadgeVariant::Destructive,
                    "lens disagrees with the model by {insight.disagreement:.3} logits — \
                     the maps below are a picture of something else"
                }
            }
            span { class: "lm-hint",
                "The maps come from an unfused re-implementation of the model's attention. "
                "That is only worth looking at while it computes what the model computes, so "
                "the two are compared on every read."
            }
        }

        h4 { class: "lm-section", "Next character" }
        div { class: "lm-dist",
            for (c, p) in top.iter() {
                div { class: "lm-dist-row",
                    span { class: "lm-dist-char", "{printable(*c)}" }
                    div { class: "lm-dist-bar",
                        div { class: "lm-dist-fill", style: "width: {p * 100.0}%" }
                    }
                    span { class: "lm-dist-value", "{p * 100.0:.1}%" }
                }
            }
        }

        h4 { class: "lm-section", "Attention" }
        p { class: "lm-note",
            "Row {shown} is the position the model is predicting from; a bright cell means "
            "that query position is reading from that key position. Everything above the "
            "diagonal is masked — the model cannot see the future."
        }
        div { class: "lm-attention",
            for (block, heads) in insight.maps.iter().enumerate() {
                for (head, map) in heads.iter().enumerate() {
                    div { class: "lm-head",
                        span { class: "lm-head-label", "block {block} · head {head}" }
                        AttentionMap { map: map.clone(), filled: insight.filled }
                    }
                }
            }
        }
        div { class: "lm-context",
            span { class: "lm-hint", "window:" }
            for c in context[tail..].iter() {
                span { class: "lm-char", "{printable(*c)}" }
            }
        }

        h4 { class: "lm-section", "Learned character similarity" }
        p { class: "lm-note",
            "The five characters each one's embedding points most nearly at. Vowels finding "
            "vowels, and a capital finding its own lower case, is the model having noticed "
            "something about spelling that nobody told it."
        }
        div { class: "lm-neighbours",
            for (c, near) in neighbours.iter() {
                div { class: "lm-neighbour",
                    span { class: "lm-char lm-char-key", "{printable(*c)}" }
                    span { class: "lm-hint", "→" }
                    for n in near.iter() {
                        span { class: "lm-char", "{printable(*n)}" }
                    }
                }
            }
        }
    }
}

/// One head's attention, as an image cropped to the real positions.
///
/// An `<img>` rather than a grid of elements: twelve heads at
/// `CONTEXT x CONTEXT` is fifty thousand cells, which the browser lays out
/// every time the panel opens.
#[component]
fn AttentionMap(map: Vec<f32>, filled: usize) -> Element {
    let n = filled.clamp(1, CONTEXT);
    // The window is filled from the left, so the live block is the top-left
    // `n x n` corner; the rest is padding nobody should read.
    let peak = (0..n)
        .flat_map(|q| (0..=q).map(move |k| (q, k)))
        .map(|(q, k)| map[q * CONTEXT + k])
        .fold(1e-6f32, f32::max);
    let mut pixels = Vec::with_capacity(n * n);
    for q in 0..n {
        for k in 0..n {
            pixels.push(if k > q {
                [22, 25, 29]
            } else {
                // Gamma below one so the long tail of small probabilities is
                // visible at all; attention is usually one bright cell.
                let a = (map[q * CONTEXT + k] / peak).clamp(0.0, 1.0).powf(0.55);
                [
                    (16.0 + a * 48.0) as u8,
                    (19.0 + a * 165.0) as u8,
                    (23.0 + a * 143.0) as u8,
                ]
            });
        }
    }
    let uri = paint::data_uri(&pixels, n, n);
    rsx! {
        img { class: "lm-map", src: "{uri}", alt: "attention probabilities" }
    }
}

/// The loss history as two SVG polylines.
///
/// Every point since the run began is kept; only the drawing is decimated,
/// and by taking the **worst** loss in each bucket rather than the first, so
/// a spike survives being summarized.
#[component]
fn LossCurve(points: Vec<(u64, f32, Option<f32>)>) -> Element {
    const WIDTH: f32 = 320.0;
    const HEIGHT: f32 = 96.0;

    if points.len() < 2 {
        return rsx! {
            div { class: "lm-curve empty-curve", "The loss curve appears once training starts." }
        };
    }

    let buckets = CURVE_POINTS.min(points.len());
    let plotted: Vec<(u64, f32, Option<f32>)> = (0..buckets)
        .filter_map(|b| {
            let from = b * points.len() / buckets;
            let to = (((b + 1) * points.len() / buckets).max(from + 1)).min(points.len());
            points[from..to]
                .iter()
                .copied()
                .reduce(|a, b| (b.0, a.1.max(b.1), b.2.or(a.2)))
        })
        .collect();

    let top = plotted.iter().map(|p| p.1).fold(0.0f32, f32::max).max(1e-3);
    let step = WIDTH / (plotted.len().max(2) - 1) as f32;
    let mut train = String::new();
    let mut held = String::new();
    for (i, (_, loss, test)) in plotted.iter().enumerate() {
        let x = i as f32 * step;
        let y = HEIGHT - (loss / top).clamp(0.0, 1.0) * (HEIGHT - 4.0) - 2.0;
        train.push_str(&format!("{x:.1},{y:.1} "));
        if let Some(t) = test {
            let y = HEIGHT - (t / top).clamp(0.0, 1.0) * (HEIGHT - 4.0) - 2.0;
            held.push_str(&format!("{x:.1},{y:.1} "));
        }
    }
    let last = points.last().map_or(0, |p| p.0);

    rsx! {
        div { class: "lm-curve",
            svg {
                view_box: "0 0 {WIDTH} {HEIGHT}",
                preserve_aspect_ratio: "none",
                width: "100%",
                height: "{HEIGHT}",
                polyline { points: "{train}", fill: "none", stroke: "#40b8a6", stroke_width: "2" }
                polyline {
                    points: "{held}",
                    fill: "none",
                    stroke: "#e2b04a",
                    stroke_width: "2",
                    stroke_dasharray: "4 3",
                }
            }
            span { class: "lm-curve-label",
                "cross-entropy · step 0 to {last} · peak {top:.2} nats"
            }
        }
    }
}

#[component]
fn Metric(value: String, label: String) -> Element {
    rsx! {
        div { class: "metric",
            span { class: "metric-value", "{value}" }
            span { class: "metric-label", "{label}" }
        }
    }
}

#[component]
fn Fact(value: String, label: String) -> Element {
    rsx! {
        div { class: "lm-fact",
            span { class: "lm-fact-value", "{value}" }
            span { class: "lm-fact-label", "{label}" }
        }
    }
}

/// Everything the interpretability card needs, from one pass over the model.
async fn look_inside(
    model: &mut Lm,
    corpus: &Corpus,
    text: &str,
    temperature: f32,
) -> fusor::Result<Insight> {
    let tokens = corpus.encode_all(text);
    let Attention {
        maps,
        filled,
        disagreement,
    } = model.attention(&tokens).await?;
    let next = model.next_char(&tokens, temperature).await?;
    let similarity = model.embedding_similarity().await?;
    Ok(Insight {
        context: text.to_string(),
        maps,
        filled,
        disagreement,
        next,
        similarity,
    })
}

/// A character as something that survives being put in a `<span>`.
fn printable(c: char) -> String {
    match c {
        ' ' => "␣".into(),
        '\n' => "⏎".into(),
        other => other.to_string(),
    }
}

/// Digit grouping, so "characters seen" is readable at seven figures.
struct Thousands(u64);

impl std::fmt::Display for Thousands {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let digits = self.0.to_string();
        for (i, c) in digits.chars().enumerate() {
            if i > 0 && (digits.len() - i).is_multiple_of(3) {
                f.write_str(",")?;
            }
            write!(f, "{c}")?;
        }
        Ok(())
    }
}
