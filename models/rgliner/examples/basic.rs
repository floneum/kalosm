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

    let mut gliner = Gliner::builder().with_source(source).build().await?;

    println!("Model loaded!");

    let labels = ["person", "organization", "location"];

    // Test with multiple texts to see if the issue is consistent
    let texts = [
        "Apple Inc. was founded by Steve Jobs in California.",
        "Microsoft Corporation is headquartered in Seattle.",
        "Elon Musk is the CEO of Tesla.",
        "Google was founded in Mountain View.",
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
