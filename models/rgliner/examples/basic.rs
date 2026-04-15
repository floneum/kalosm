use rgliner::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Use local GGUF files if GLINER_MODEL env var is set, otherwise try official HuggingFace edge variant
    let source = if let Ok(model_path) = std::env::var("GLINER_MODEL") {
        // Derive label encoder path from model path
        let label_encoder_path = model_path.replace(".gguf", "-label-encoder.gguf");
        println!("Loading GLiNER model from: {}", model_path);
        println!("Loading label encoder from: {}", label_encoder_path);
        GlinerSource::local(model_path, label_encoder_path)
    } else {
        // Use official HuggingFace GGUF
        println!("Loading GLiNER model from HuggingFace (edge variant)...");
        GlinerSource::edge()
    };

    let mut gliner = Gliner::builder()
        .with_source(source)
        .with_threshold(0.01)
        .build()
        .await?;

    println!("Model loaded!");

    let labels = ["person", "award", "date", "competitions", "teams"];

    // The Ronaldo paragraph from the v2.0 model card (short strings score poorly on v2.0).
    let texts = [
        "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team.",
    ];

    for text in texts {
        println!("\n--- Testing: {} ---", text);
        let entities = gliner.extract(text, &labels).await?;

        println!("Found {} entities:", entities.len());
        for entity in entities {
            println!(
                "  {}: '{}' (score: {:.2})",
                entity.label, entity.text, entity.score
            );
        }
    }

    Ok(())
}
