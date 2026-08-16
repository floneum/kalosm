//! Embed three sentences with the cached Qwen3-Embedding-0.6B model — the
//! architecture branch that runs RoPE, RMS norm and quantized projections —
//! and check that the two paraphrases are closer than the unrelated sentence.

use kalosm_model_types::FileSource;
use rbert::*;
use std::path::PathBuf;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(async {
        let kalosm_cache =
            PathBuf::from(std::env::var("HOME")?).join("Library/Application Support/kalosm/cache");
        let source = BertSource::qwen3_embedding_0_6b()
            .with_tokenizer(FileSource::Local(
                kalosm_cache.join("Qwen/Qwen3-Embedding-0.6B/main/tokenizer.json"),
            ))
            .with_model(FileSource::Local(kalosm_cache.join(
                "Qwen/Qwen3-Embedding-0.6B-GGUF/main/Qwen3-Embedding-0.6B-Q8_0.gguf",
            )));

        let bert = Bert::builder().with_source(source).build().await?;

        let sentences = [
            "the cat sat on the mat",
            "a feline rested on the rug",
            "the stock market fell sharply",
        ];
        let embeddings = bert
            .embed_batch_with_pooling(sentences.to_vec(), Pooling::Last)
            .await?;
        let vectors: Vec<&[f32]> = embeddings.iter().map(|e| e.vector()).collect();

        for (sentence, vector) in sentences.iter().zip(&vectors) {
            println!(
                "{sentence:?}: dim {} first 4 dims {:?}",
                vector.len(),
                &vector[..4]
            );
        }
        let cos_01 = cosine(vectors[0], vectors[1]);
        let cos_02 = cosine(vectors[0], vectors[2]);
        println!("cosine(cat/mat, feline/rug)    = {cos_01:.4}");
        println!("cosine(cat/mat, stock market)  = {cos_02:.4}");
        assert!(
            cos_01 > cos_02,
            "paraphrase similarity {cos_01} should exceed unrelated {cos_02}"
        );
        println!("PASS: paraphrase similarity clearly exceeds unrelated similarity");
        Ok(())
    })
}
