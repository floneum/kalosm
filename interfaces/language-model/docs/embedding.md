# Embeddings

Embeddings are a way to represent the meaning of text in a numerical format. They can be used to compare the meaning of two different texts, or search for documents with a [embedding database](https://docs.rs/kalosm/latest/kalosm/struct.DocumentTable.html).

## Creating Embeddings

You can create embeddings from text using a [`Bert`](https://docs.rs/kalosm/latest/kalosm/struct.Bert.html) embedding model. You can call `embed` on a `Bert` instance to get an embedding for a single sentence or `embed_batch` to get embeddings for a list of sentences at once:

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
let mut bert = Bert::new().await.unwrap();
let sentences = vec![
    "Kalosm can be used to build local AI applications",
    "With private LLMs data never leaves your computer",
    "The quick brown fox jumps over the lazy dog",
];
let embeddings = bert.embed_batch(&sentences).await.unwrap();
# }
```

Once you have embeddings, you can compare them to each other with a distance metric. The cosine similarity is a common metric for comparing embeddings that measures the cosine of the angle between the two vectors:

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
# let mut bert = Bert::new().await.unwrap();
# let sentences = vec![
#     "Kalosm can be used to build local AI applications",
#     "With private LLMs data never leaves your computer",
#     "The quick brown fox jumps over the lazy dog",
# ];
# let embeddings = bert.embed_batch(&sentences).await.unwrap();
// Find the cosine similarity between each pair of sentences
let n_sentences = sentences.len();
for (i, e_i) in embeddings.iter().enumerate() {
    for j in (i + 1)..n_sentences {
        let e_j = embeddings.get(j).unwrap();
        let cosine_similarity = e_j.cosine_similarity(e_i);
        println!("score: {cosine_similarity:.2} '{}' '{}'", sentences[i], sentences[j])
    }
}
# }
```

You should see that the first two sentences are similar to each other, while the third sentence not similar to either of the first two:

```text
score: 0.82 'Kalosm can be used to build local AI applications' 'With private LLMs data never leaves your computer'
score: 0.72 'With private LLMs data never leaves your computer' 'The quick brown fox jumps over the lazy dog'
score: 0.72 'Kalosm can be used to build local AI applications' 'The quick brown fox jumps over the lazy dog'
```

## Searching for Similar Text

Embeddings can also be a powerful tool for search. Unlike traditional text based search, searching for text with embeddings doesn't directly look for keywords in the text. Instead, it looks for text with similar meanings which can make search more robust and accurate.

In the previous example, we used the cosine similarity to find the similarity between two sentences. Even though the first two sentences have no words in common, their embeddings are similar because they have related meanings.


You can use a vector database to store embedding, value pairs in an easily searchable way. You can create an vector database with [`VectorDB::new`](https://docs.rs/kalosm/latest/kalosm/language/struct.VectorDB.html):

```rust, no_run
# use std::collections::HashMap;
# use kalosm::language::*;
# #[tokio::main]
# async fn main() -> anyhow::Result<()> {
// Create a good default Bert model for search
let bert = Bert::new_for_search().await?;
let sentences = [
    "Kalosm can be used to build local AI applications",
    "With private LLMs data never leaves your computer",
    "The quick brown fox jumps over the lazy dog",
];
// Embed sentences into the vector space
let embeddings = bert.embed_batch(sentences).await?;
println!("embeddings {:?}", embeddings);

// Create a vector database from the embeddings along with a map between the embedding ids and the sentences
let db = VectorDB::new()?;
let embeddings = db.add_embeddings(embeddings)?;
let embedding_id_to_sentence: HashMap<EmbeddingId, &str> =
    HashMap::from_iter(embeddings.into_iter().zip(sentences));

// Embed a query into the vector space. We use `embed_query` instead of `embed` because some models embed queries differently than normal text.
let embedding = bert.embed_query("What is Kalosm?").await?;
let closest = db.search(&embedding).run()?;
if let [closest] = closest.as_slice() {
    let distance = closest.distance;
    let text = embedding_id_to_sentence.get(&closest.value).unwrap();
    println!("distance: {distance}");
    println!("closest:  {text}");
}
# Ok(())
# }
```

The vector database should find that the closest sentence to "What is Kalosm?" is "Kalosm can be used to build local AI applications":

```text
distance: 0.18480265
closest: Kalosm can be used to build local AI applications
```
