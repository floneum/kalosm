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
//! The example uses the default GLiNER-RelEx source, which downloads the
//! default GGUF model from Hugging Face.

use clap::Parser;
use rgliner::relex::GlinerRelEx;

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

    /// Minimum confidence for entity detection.
    #[arg(long, default_value_t = 0.5)]
    entity_threshold: f32,

    /// Minimum confidence for relation classification.
    #[arg(long, default_value_t = 0.5)]
    relation_threshold: f32,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

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

    println!("Loading default GLiNER-RelEx model...");
    println!("Text: {}", args.text);
    println!("Entity labels: {:?}", entity_labels);
    println!("Relation labels: {:?}", relation_labels);

    let relex = GlinerRelEx::builder()
        .with_entity_threshold(args.entity_threshold)
        .with_relation_threshold(args.relation_threshold)
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
