//! This example shows how to cache the results of the Bert embedding model.

use rbert::*;
use std::num::NonZeroUsize;

fn main() -> anyhow::Result<()> {
    pollster::block_on(async {
        let bert = Bert::builder()
            .build()
            .await?
            // You can call the `.cached` method to cache the results of the Bert embedding in a LRU cache with the given capacity.
            .cached(NonZeroUsize::new(1000).unwrap());

        // Try to load the cache from the filesystem
        if let Ok(cache) = std::fs::read("cache.bin") {
            let cache: Vec<(EmbeddingInput, Vec<f32>)> = postcard::from_bytes(&cache)?;
            bert.load_cache(cache);
        }

        let start_time = std::time::Instant::now();
        let sentences = [
            "Cats are cool",
            "The geopolitical situation is dire",
            "Pets are great",
            "Napoleon was a tyrant",
            "Napoleon was a great general",
        ];
        // When you embed a new sentence, the cache will store the embedding for that sentence.
        let embeddings = bert.embed_batch(sentences).await?;
        println!("{embeddings:?}");
        println!("embedding uncached took {:?}", start_time.elapsed());

        let start_time = std::time::Instant::now();
        // If you embed the same sentences again, the cache will be used.
        let embeddings = bert.embed_batch(sentences).await?;
        println!("{embeddings:?}");
        println!("embedding cached took {:?}", start_time.elapsed());

        let sentences = [
            "Cats are cool",
            "The geopolitical situation is dire",
            "Pets are great",
            "Napoleon is from France",
            "Kalosm supports embedding models",
        ];
        // When you embed a new sentence, the cache will store the embedding for that sentence.
        let embeddings = bert.embed_batch(sentences).await?;
        println!("{embeddings:?}");
        println!("embedding partially cached took {:?}", start_time.elapsed());

        // Save the cache to the filesystem for future use
        let cache = bert.export_cache();
        let file = std::fs::File::create("cache.bin")?;
        postcard::to_io(&cache, &file)?;

        Ok(())
    })
}
