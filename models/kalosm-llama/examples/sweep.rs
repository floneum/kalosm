#![recursion_limit = "256"]
//! Run every `LlamaSource` preset (or a subset) through one chat turn and
//! report load time, time to first token, decode rate and the text, with a
//! cheap degeneration check. `sweep --list` shows which presets are cached;
//! `sweep --cached` runs only those; `sweep a b c` runs named presets;
//! `sweep --all` runs everything, downloading as it goes.

use kalosm_llama::prelude::*;
use std::io::Write;
use std::time::Instant;

macro_rules! presets {
    ($($name:ident),* $(,)?) => {
        fn presets() -> Vec<(&'static str, fn() -> LlamaSource)> {
            vec![$((stringify!($name), LlamaSource::$name as fn() -> LlamaSource)),*]
        }
    };
}

presets!(
    tiny_llama_1_1b_chat,
    tiny_llama_1_1b,
    llama_3_2_1b_chat,
    llama_3_2_3b_chat,
    llama_8b_chat,
    llama_3_1_8b_chat,
    llama_8b_chat_q8,
    llama_8b,
    llama_8b_sppo_iter3,
    llama_7b,
    llama_7b_chat,
    llama_7b_code,
    llama_13b,
    llama_13b_chat,
    llama_13b_code,
    llama_34b_code,
    llama_70b,
    llama_70b_chat,
    phi_3_mini_4k_instruct,
    phi_3_1_mini_4k_instruct,
    phi_3_5_mini_4k_instruct,
    phi_4,
    qwen_2_5_0_5b_instruct,
    qwen_2_5_1_5b_instruct,
    qwen_2_5_3b_instruct,
    qwen_2_5_7b_instruct,
    qwen_3_0_6b_instruct,
    qwen_3_1_7b_instruct,
    qwen_3_4b_instruct,
    qwen_3_8b_instruct,
    qwen_3_14b_instruct,
    qwen_3_32b_instruct,
    deepseek_r1_distill_qwen_1_5b,
    deepseek_r1_distill_qwen_7b,
    deepseek_r1_distill_qwen_14b,
    deepseek_r1_distill_llama_8b,
    gemma_3_270m_chat,
    gemma_3_1b_chat,
    gemma_3_4b_chat,
    gemma_3_12b_chat,
    gemma_3_27b_chat,
    mistral_7b,
    mistral_7b_instruct,
    mistral_7b_instruct_2,
    codestral_22b,
    neural_hermes_2_5_mistral_7b,
    neural_chat_7b_v3_3,
    zephyr_7b_alpha,
    zephyr_7b_beta,
    open_chat_7b,
    starling_7b_alpha,
    starling_7b_beta,
    wizard_lm_7b_v2,
    solar_10_7b,
    solar_10_7b_instruct,
);

const PROMPT: &str = "In two or three sentences, explain why the sky is blue.";
const MAX_TOKENS: usize = 96;

/// Fraction of repeated 4-grams over the token stream: a loop of the same
/// phrase drives this toward 1.
fn repetition(tokens: &[String]) -> f64 {
    if tokens.len() < 8 {
        return 0.0;
    }
    let grams: Vec<String> = tokens.windows(4).map(|w| w.concat()).collect();
    let unique: std::collections::HashSet<&String> = grams.iter().collect();
    1.0 - unique.len() as f64 / grams.len() as f64
}

struct Row {
    name: &'static str,
    load: f64,
    first: f64,
    decode_tps: f64,
    tokens: usize,
    rep: f64,
    text: String,
    error: Option<String>,
}

async fn run(name: &'static str, source: LlamaSource) -> Row {
    let mut row = Row {
        name,
        load: 0.0,
        first: 0.0,
        decode_tps: 0.0,
        tokens: 0,
        rep: 0.0,
        text: String::new(),
        error: None,
    };
    let t = Instant::now();
    let model = match Llama::builder().with_source(source).build().await {
        Ok(m) => m,
        Err(e) => {
            row.error = Some(format!("load: {e}"));
            return row;
        }
    };
    row.load = t.elapsed().as_secs_f64();
    // A warm turn first: the first resolve of a shape compiles and tunes
    // its kernels, which is a one-off cost the steady state never pays.
    {
        let mut chat = model.chat();
        let mut warm = chat(&"Say hello.".to_string());
        let mut n = 0;
        while warm.next().await.is_some() {
            n += 1;
            if n >= 8 {
                break;
            }
        }
    }
    let mut chat = model.chat();
    let mut response = chat(&PROMPT.to_string());
    let t = Instant::now();
    let mut first: Option<Instant> = None;
    let mut tokens: Vec<String> = Vec::new();
    while let Some(tok) = response.next().await {
        if first.is_none() {
            first = Some(Instant::now());
        }
        print!("{tok}");
        let _ = std::io::stdout().flush();
        tokens.push(tok);
        if tokens.len() >= MAX_TOKENS {
            break;
        }
    }
    println!();
    drop(response);
    let Some(first) = first else {
        row.error = Some("no tokens".into());
        return row;
    };
    row.first = (first - t).as_secs_f64();
    let decode = first.elapsed().as_secs_f64();
    row.tokens = tokens.len();
    row.decode_tps = if tokens.len() > 1 {
        (tokens.len() - 1) as f64 / decode
    } else {
        0.0
    };
    row.rep = repetition(&tokens);
    row.text = tokens.concat();
    row
}

fn free_gb() -> f64 {
    std::process::Command::new("df")
        .args(["-k", "/"])
        .output()
        .ok()
        .and_then(|o| {
            let s = String::from_utf8_lossy(&o.stdout);
            s.lines()
                .nth(1)
                .and_then(|l| l.split_whitespace().nth(3))
                .and_then(|v| v.parse::<f64>().ok())
        })
        .map_or(0.0, |kb| kb / 1024.0 / 1024.0)
}

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .try_init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let all = presets();
    if args.iter().any(|a| a == "--list") {
        for (name, make) in &all {
            println!(
                "{:<36} {}",
                name,
                if make().is_cached() {
                    "cached"
                } else {
                    "missing"
                }
            );
        }
        return;
    }
    let selected: Vec<(&'static str, fn() -> LlamaSource)> = if args.iter().any(|a| a == "--all") {
        all
    } else if args.is_empty() || args.iter().any(|a| a == "--cached") {
        all.into_iter().filter(|(_, m)| m().is_cached()).collect()
    } else {
        all.into_iter()
            .filter(|(n, _)| args.iter().any(|a| a == n))
            .collect()
    };
    pollster::block_on(async {
        let mut rows = Vec::new();
        for (name, make) in selected {
            let source = make();
            if !source.is_cached() && free_gb() < 4.0 {
                eprintln!("skipping {name}: not cached and under 4 GB free");
                continue;
            }
            println!("\n===== {name} =====");
            rows.push(run(name, source).await);
        }
        println!(
            "\n{:<34} {:>7} {:>7} {:>8} {:>5} {:>5}  note",
            "preset", "load s", "first s", "tok/s", "toks", "rep"
        );
        for r in &rows {
            let note = match &r.error {
                Some(e) => e.clone(),
                None if r.rep > 0.3 => "REPETITIVE".into(),
                None if r.tokens < 8 => "SHORT".into(),
                None => String::new(),
            };
            println!(
                "{:<34} {:>7.1} {:>7.2} {:>8.1} {:>5} {:>5.2}  {note}",
                r.name, r.load, r.first, r.decode_tps, r.tokens, r.rep
            );
        }
    });
}
