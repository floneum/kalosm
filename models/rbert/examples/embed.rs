use std::str::FromStr;

use fusor::Device;
use rbert::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    struct EmbeddingModel {
        inner: BertSource,
    }

    impl FromStr for EmbeddingModel {
        type Err = String;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                "qwen3" => Ok(Self {
                    inner: BertSource::qwen3_embedding_0_6b(),
                }),
                "snowflake_extra_small" => Ok(Self {
                    inner: BertSource::snowflake_arctic_embed_extra_small(),
                }),
                "snowflake_small" => Ok(Self {
                    inner: BertSource::snowflake_arctic_embed_small(),
                }),
                "snowflake_medium" => Ok(Self {
                    inner: BertSource::snowflake_arctic_embed_medium(),
                }),
                "snowflake_large" => Ok(Self {
                    inner: BertSource::snowflake_arctic_embed_large(),
                }),
                _ => Err(format!("Unknown embedding model: {s}")),
            }
        }
    }

    let embedding_model = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "qwen3".to_string());

    let embedding_model = EmbeddingModel::from_str(&embedding_model)
        .expect("Invalid embedding model. Supported models: qwen3, snowflake_extra_small, snowflake_small, snowflake_medium, snowflake_large");
    let bert = Bert::builder()
        .with_device(if std::env::var("FORCE_CPU").is_ok() {
            Device::Cpu
        } else {
            Device::auto().await
        })
        .with_source(embedding_model.inner)
        .build()
        .await
        .unwrap();
    let sentences = [
        "Cats are cool",
        "The geopolitical situation is dire",
        "Pets are great",
        "Napoleon is from France",
        "Kalosm supports embedding models",
    ];
    let start = std::time::Instant::now();
    // Use embed_batch_for with Query variant to apply the model's instruction prefix
    let embeddings = bert
        .embed_batch_for(sentences.iter().map(|s| EmbeddingInput {
            text: s.to_string(),
            variant: EmbeddingVariant::Query,
        }))
        .await?;
    println!("took {:?}", start.elapsed());

    // Find the cosine similarity between the first two sentences
    let mut similarities = vec![];
    let n_sentences = sentences.len();
    for (i, e_i) in embeddings.iter().enumerate() {
        for j in (i + 1)..n_sentences {
            let e_j = embeddings.get(j).unwrap();
            let cosine_similarity = e_j.cosine_similarity(e_i);
            similarities.push((cosine_similarity, i, j))
        }
    }
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities.iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

    Ok(())
}
