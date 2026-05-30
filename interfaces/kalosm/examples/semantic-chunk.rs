use std::time::Instant;

use anyhow::Ok;
use kalosm::language::*;

fn main() -> anyhow::Result<()> {
    pollster::block_on(async {
        let url = std::env::args()
            .nth(1)
            .unwrap_or_else(|| "https://floneum.com/blog/kalosm_0_2".to_string());

        let target_score: f32 = std::env::args()
            .nth(2)
            .unwrap_or_else(|| "0.65".to_string())
            .parse()?;

        let start_loading_time = Instant::now();
        let bert = Bert::new_for_search().await?;
        println!("Loaded in {:?}", start_loading_time.elapsed());

        let semantic_chunker = SemanticChunker::new().with_target_score(target_score);

        let document = Url::parse(&url).unwrap().into_document().await?;
        let start_time = Instant::now();
        let chunks = semantic_chunker.chunk(&document, &bert).await?;
        println!("Chunked in {:?}", start_time.elapsed());

        print!("\n\n\nFINAL CHUNKS:\n");
        for (i, chunk) in chunks.iter().enumerate() {
            println!(
                "CHUNK {}:\n{}\n",
                i,
                &document.body()[chunk.byte_range.clone()].trim()
            );
        }

        Ok(())
    })
}
