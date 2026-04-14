use kalosm_model_types::FileSource;
use rgliner::*;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let source = GlinerSource::custom(
        FileSource::local(PathBuf::from("./models/rgliner/weights/test-small-f32.gguf")),
        FileSource::local(PathBuf::from(
            "./models/rgliner/weights/test-small-f32-label-encoder.gguf",
        )),
        FileSource::huggingface(
            "sentence-transformers/all-MiniLM-L12-v2".to_string(),
            "main".to_string(),
            "config.json".to_string(),
        ),
        FileSource::huggingface(
            "sentence-transformers/all-MiniLM-L12-v2".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ),
        FileSource::huggingface(
            "knowledgator/gliner-bi-small-v2.0".to_string(),
            "main".to_string(),
            "tokenizer.json".to_string(),
        ),
        FileSource::huggingface(
            "knowledgator/gliner-bi-small-v2.0".to_string(),
            "main".to_string(),
            "gliner_config.json".to_string(),
        ),
    );
    let mut gliner = Gliner::builder()
        .with_source(source)
        .with_threshold(0.0)
        .build()
        .await?;

    let text = "Cristiano Ronaldo dos Santos Aveiro (Portuguese pronunciation: [kɾiʃˈtjɐnu ʁɔˈnaldu]; born 5 February 1985) is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al Nassr and the Portugal national team.";
    let labels = ["person", "award", "date", "competitions", "teams"];

    let entities = gliner.extract(text, &labels).await?;
    let mut ents = entities;
    ents.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    ents.iter()
        .take(10)
        .for_each(|e| println!("  {} '{}' {:.4}", e.label, e.text, e.score));

    Ok(())
}
