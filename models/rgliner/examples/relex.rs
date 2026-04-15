//! Example of using GlinerRelEx for joint NER and relation extraction.
//!
//! Examples:
//! ```
//! cargo run --example relex -p rgliner --release -- \
//!     --text "Apple was founded by Steve Jobs in California." \
//!     --entity-labels person,organization,location \
//!     --relation-labels "founded by,located in"
//! ```
//!
//! The GGUF file has the tokenizer and GLiNER config baked in as metadata,
//! so only the model file path is needed.

use clap::Parser;
use rgliner::relex::{GlinerRelEx, GlinerRelExSource};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    about = "GLiNER-RelEx joint NER and relation extraction",
    long_about = None,
)]
struct Args {
    /// Input text to analyze.
    #[arg(short, long)]
    text: String,

    /// Entity labels to detect (comma-separated).
    #[arg(short = 'e', long, value_delimiter = ',', required = true)]
    entity_labels: Vec<String>,

    /// Relation labels to detect (comma-separated). If empty, only entities are returned.
    #[arg(short = 'r', long, value_delimiter = ',', default_value = "")]
    relation_labels: Vec<String>,

    /// Path to the GGUF model file. If omitted, uses
    /// `<crate>/weights/gliner-relex-multi-v1.0.gguf` or `$GLINER_MODEL`.
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,

    /// Minimum confidence for entity detection.
    #[arg(long, default_value_t = 0.5)]
    entity_threshold: f32,

    /// Minimum confidence for relation classification.
    #[arg(long, default_value_t = 0.5)]
    relation_threshold: f32,

    /// Maximum adjacency score to keep an entity pair.
    #[arg(long, default_value_t = 0.5)]
    adjacency_threshold: f32,
}

fn resolve_model_path(arg: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(path) = arg {
        return Ok(path);
    }
    if let Ok(env_path) = std::env::var("GLINER_MODEL") {
        return Ok(PathBuf::from(env_path));
    }
    let default = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("weights")
        .join("gliner-relex-multi-v1.0.gguf");
    if default.exists() {
        return Ok(default);
    }
    anyhow::bail!(
        "No model path provided. Use --model, set $GLINER_MODEL, or run:\n  \
         python scripts/convert_relex_to_gguf.py -m knowledgator/gliner-relex-multi-v1.0 \
         -o weights/gliner-relex-multi-v1.0.gguf"
    )
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let model_path = resolve_model_path(args.model)?;

    // `value_delimiter = ','` with `default_value = ""` produces a single empty
    // string when the user passes nothing - filter it out.
    let entity_labels: Vec<&str> = args
        .entity_labels
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let relation_labels: Vec<&str> = args
        .relation_labels
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();

    if entity_labels.is_empty() {
        anyhow::bail!("--entity-labels must contain at least one non-empty label");
    }

    println!("Loading model from: {:?}", model_path);
    println!("Text: {}", args.text);
    println!("Entity labels: {:?}", entity_labels);
    println!("Relation labels: {:?}", relation_labels);

    let source = GlinerRelExSource::local(model_path);
    let relex = GlinerRelEx::builder()
        .with_source(source)
        .with_entity_threshold(args.entity_threshold)
        .with_relation_threshold(args.relation_threshold)
        .with_adjacency_threshold(args.adjacency_threshold)
        .build()
        .await?;

    let (entities, relations) = relex
        .extract(&args.text, &entity_labels, &relation_labels)
        .await?;

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
