//! Example of using GlinerRelEx for joint NER and relation extraction.
//!
//! Run with:
//! ```
//! cargo run --example relex -p rgliner
//! ```
//!
//! The GGUF file has the tokenizer and GLiNER config baked in as metadata,
//! so only the model file path is needed.

use rgliner::relex::{GlinerRelEx, GlinerRelExSource};
use std::env;
use std::path::PathBuf;

fn get_model_path() -> Option<PathBuf> {
    let weights_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("weights");
    let default_path = weights_dir.join("gliner-relex-multi-v1.0.gguf");
    let model_path = env::var("GLINER_MODEL")
        .map(PathBuf::from)
        .unwrap_or(default_path);

    if !model_path.exists() {
        eprintln!("Model file not found: {:?}", model_path);
        return None;
    }
    Some(model_path)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model_path = get_model_path().ok_or_else(|| {
        anyhow::anyhow!(
            "Model file not found. Convert with:\n  \
             python scripts/convert_relex_to_gguf.py -m knowledgator/gliner-relex-multi-v1.0 \
             -o weights/gliner-relex-multi-v1.0.gguf"
        )
    })?;

    println!("Loading model from: {:?}", model_path);

    let source = GlinerRelExSource::local(model_path);

    let text = "Perfect! Now I have all the information I need to create a comprehensive GGUF inference framework from scratch. Let me create a single-file implementation with full SIMD support targeting nightly Rust";
    let entity_labels = ["technology", "language", "file format"];
    let relation_labels = ["supported by", "implemented with"];

    println!("\nText: {}", text);
    println!("Entity labels: {:?}", entity_labels);
    println!("Relation labels: {:?}", relation_labels);

    let relex = GlinerRelEx::builder()
        .with_source(source)
        .with_entity_threshold(0.1)
        .build()
        .await?;

    let (entities, relations) = relex.extract(text, &entity_labels, &relation_labels).await?;

    println!("\nEntities found ({}):", entities.len());
    for entity in &entities {
        println!(
            "  {} [{}] (score: {:.3})",
            entity.text, entity.label, entity.score
        );
    }

    println!("\nRelations found ({}):", relations.len());
    for relation in &relations {
        println!(
            "  {} --[{}]--> {} (score: {:.3})",
            relation.head.text, relation.relation, relation.tail.text, relation.score
        );
    }

    Ok(())
}
