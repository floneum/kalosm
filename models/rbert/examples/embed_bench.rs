//! Warm embedding latency of the cached bge-small-en-v1.5 model.
//! `embed_bench [batch] [iterations]`
use kalosm_model_types::FileSource;
use rbert::*;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    pollster::block_on(async {
        let args: Vec<usize> = std::env::args().skip(1).filter_map(|a| a.parse().ok()).collect();
        let batch = args.first().copied().unwrap_or(8);
        let iterations = args.get(1).copied().unwrap_or(20);
        let hf_snapshot = PathBuf::from(std::env::var("HOME")?).join(
            ".cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        );
        let kalosm_cache =
            PathBuf::from(std::env::var("HOME")?).join("Library/Application Support/kalosm/cache");
        let source = BertSource::bge_small_en()
            .with_config(FileSource::Local(hf_snapshot.join("config.json")))
            .with_tokenizer(FileSource::Local(hf_snapshot.join("tokenizer.json")))
            .with_model(FileSource::Local(kalosm_cache.join(
                "CompendiumLabs/bge-small-en-v1.5-gguf/main/bge-small-en-v1.5-q4_k_m.gguf",
            )));
        let bert = Bert::builder().with_source(source).build().await?;
        let base = [
            "the cat sat on the mat",
            "a feline rested on the rug",
            "the stock market fell sharply this morning after the report",
            "rust compiles to fast native code",
        ];
        let sentences: Vec<&str> = (0..batch).map(|i| base[i % base.len()]).collect();
        let mut times = Vec::new();
        let mut first = None;
        for i in 0..iterations {
            let t = std::time::Instant::now();
            let embeddings = bert
                .embed_batch_with_pooling(sentences.clone(), Pooling::CLS)
                .await?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            assert_eq!(embeddings.len(), batch);
            if i == 0 {
                first = Some(embeddings[0].vector()[..4].to_vec());
            } else {
                times.push(ms);
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "batch={batch} warm_min_ms={:.3} warm_median_ms={:.3} first4={:?}",
            times[0],
            times[times.len() / 2],
            first.unwrap()
        );
        Ok(())
    })
}
