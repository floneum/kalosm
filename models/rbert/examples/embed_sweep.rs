//! Embed a fixed sentence set with every `BertSource` preset and report
//! time plus a sanity check: the two pet sentences must be each other's
//! nearest neighbour.

use rbert::*;
use std::time::Instant;

fn presets() -> Vec<(&'static str, fn() -> BertSource)> {
    vec![
        ("bge_large_en", BertSource::bge_large_en),
        ("bge_base_en", BertSource::bge_base_en),
        ("bge_small_en", BertSource::bge_small_en),
        ("mini_lm_l6_v2", BertSource::mini_lm_l6_v2),
        (
            "snowflake_arctic_embed_extra_small",
            BertSource::snowflake_arctic_embed_extra_small,
        ),
        (
            "snowflake_arctic_embed_small",
            BertSource::snowflake_arctic_embed_small,
        ),
        (
            "snowflake_arctic_embed_medium",
            BertSource::snowflake_arctic_embed_medium,
        ),
        (
            "snowflake_arctic_embed_medium_long",
            BertSource::snowflake_arctic_embed_medium_long,
        ),
        (
            "snowflake_arctic_embed_large",
            BertSource::snowflake_arctic_embed_large,
        ),
        ("qwen3_embedding_0_6b", BertSource::qwen3_embedding_0_6b),
        ("qwen3_embedding_4b", BertSource::qwen3_embedding_4b),
        ("qwen3_embedding_8b", BertSource::qwen3_embedding_8b),
    ]
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let sentences = [
        "Cats are cool",
        "The geopolitical situation is dire",
        "Pets are great",
        "Napoleon is from France",
        "Kalosm supports embedding models",
    ];
    pollster::block_on(async {
        let mut rows = Vec::new();
        for (name, make) in presets() {
            if !args.is_empty() && !args.iter().any(|a| a == name) {
                continue;
            }
            println!("===== {name} =====");
            let t = Instant::now();
            let bert = match Bert::builder().with_source(make()).build().await {
                Ok(b) => b,
                Err(e) => {
                    rows.push((name, 0.0, 0.0, format!("load: {e}")));
                    continue;
                }
            };
            let load = t.elapsed().as_secs_f64();
            // Warm once, then time.
            let _ = bert.embed_batch(sentences).await;
            let t = Instant::now();
            let embeddings = match bert.embed_batch(sentences).await {
                Ok(e) => e,
                Err(e) => {
                    rows.push((name, load, 0.0, format!("embed: {e}")));
                    continue;
                }
            };
            let embed = t.elapsed().as_secs_f64() * 1000.0;
            let sim = |i: usize, j: usize| embeddings[i].cosine_similarity(&embeddings[j]);
            let pets = sim(0, 2);
            let best_other = (1..5)
                .filter(|&j| j != 2)
                .map(|j| sim(0, j))
                .fold(f32::MIN, f32::max);
            let note = if pets > best_other {
                format!("ok (cats~pets {pets:.2} > {best_other:.2})")
            } else {
                format!("SUSPECT (cats~pets {pets:.2} <= {best_other:.2})")
            };
            println!("{note}");
            rows.push((name, load, embed, note));
        }
        println!("\n{:<36} {:>7} {:>9}  note", "preset", "load s", "embed ms");
        for (name, load, embed, note) in rows {
            println!("{name:<36} {load:>7.1} {embed:>9.1}  {note}");
        }
    });
    Ok(())
}
