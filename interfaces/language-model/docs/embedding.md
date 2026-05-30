# Embeddings

Embeddings are a way to represent the meaning of text in a numerical format. They can be used to compare the meaning of two different texts, search for documents with a [embedding database](https://docs.rs/kalosm/latest/kalosm/struct.DocumentTable.html), or train [classification models](https://docs.rs/kalosm-learning/latest/kalosm_learning/struct.TextClassifier.html).

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

## Classification with Embeddings

Since embeddings represent something about the meaning of text, you can use them to quickly train classification models. Instead of training a whole new model to understand text and classify it, you can just train a classifier on top of a frozen embedding model.


Even with a relatively small dataset, a classifier built on top of an embedding model can achieve impressive results. Lets start by creating a dataset of questions and statements:

```rust, no_run
# use kalosm::language::*;
# use kalosm_learning::*;
# use kalosm_common::accelerated_device_if_available;
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
#[derive(Debug, Clone, Copy, Class)]
enum SentenceType {
    Question,
    Statement,
}
// Create a dataset for the classifier
let bert = Bert::builder()
    .with_source(BertSource::snowflake_arctic_embed_extra_small())
    .build()
    .await?;
let mut dataset = TextClassifierDatasetBuilder::<SentenceType, _>::new(&bert);
const QUESTIONS: [&str; 10] = [
    "What is the capital of France",
    "What is the capital of the United States",
    "What is the best way to learn a new language",
    "What is the best way to learn a new programming language",
    "What is a framework",
    "What is a library",
    "What is a good way to learn a new language",
    "What is a good way to learn a new programming language",
    "What is the city with the most people in the world",
    "What is the most spoken language in the world",
];
const STATEMENTS: [&str; 10] = [
    "The president of France is Emmanuel Macron",
    "The capital of France is Paris",
    "The capital of the United States is Washington, DC",
    "The light bulb was invented by Thomas Edison",
    "The best way to learn a new programming language is to start with the basics and gradually build on them",
    "A framework is a set of libraries and tools that help developers build applications",
    "A library is a collection of code that can be used by other developers",
    "A good way to learn a new language is to practice it every day",
    "The city with the most people in the world is Tokyo",
    "The most spoken language in the United States is English",
];

for question in QUESTIONS {
    dataset.add(question, SentenceType::Question).await?;
}
for statement in STATEMENTS {
    dataset.add(statement, SentenceType::Statement).await?;
}
let dev = accelerated_device_if_available()?;
let dataset = dataset.build(&dev)?;
    // Create a classifier
    let classifier = TextClassifier::<SentenceType>::new(Classifier::new(
        &dev,
        ClassifierConfig::new().layers_dims([10]),
    )?);

    // Train the classifier
    classifier.train(
        &dataset, // The dataset to train on
        100,      // The number of epochs to train for
        0.0003,   // The learning rate
        50,       // The batch size
        |_| {},   // The callback to run as the model trains
    )?;

    loop {
        let input = prompt_input("Input: ").unwrap();
        let embedding = bert.embed(input).await?;
        let output = classifier.run(embedding)?;
        println!("Output: {:?}", output);
    }
}
```

Next, train a classifier on the dataset:

```rust, no_run
# use kalosm::language::*;
# use kalosm_learning::*;
# use kalosm_common::accelerated_device_if_available;
# #[tokio::main]
# async fn main() -> Result<(), Box<dyn std::error::Error>> {
# #[derive(Debug, Clone, Copy, Class)]
# enum SentenceType {
#     Question,
#     Statement,
# }
# // Create a dataset for the classifier
# let bert = Bert::builder()
#     .with_source(BertSource::snowflake_arctic_embed_extra_small())
#     .build()
#     .await?;
# let mut dataset = TextClassifierDatasetBuilder::<SentenceType, _>::new(&bert);
# const QUESTIONS: [&str; 10] = [
#     "What is the capital of France",
#     "What is the capital of the United States",
#     "What is the best way to learn a new language",
#     "What is the best way to learn a new programming language",
#     "What is a framework",
#     "What is a library",
#     "What is a good way to learn a new language",
#     "What is a good way to learn a new programming language",
#     "What is the city with the most people in the world",
#     "What is the most spoken language in the world",
# ];
# const STATEMENTS: [&str; 10] = [
#     "The president of France is Emmanuel Macron",
#     "The capital of France is Paris",
#     "The capital of the United States is Washington, DC",
#     "The light bulb was invented by Thomas Edison",
#     "The best way to learn a new programming language is to start with the basics and gradually build on them",
#     "A framework is a set of libraries and tools that help developers build applications",
#     "A library is a collection of code that can be used by other developers",
#     "A good way to learn a new language is to practice it every day",
#     "The city with the most people in the world is Tokyo",
#     "The most spoken language in the United States is English",
# ];
#
# for question in QUESTIONS {
#     dataset.add(question, SentenceType::Question).await?;
# }
# for statement in STATEMENTS {
#     dataset.add(statement, SentenceType::Statement).await?;
# }
# let dev = accelerated_device_if_available()?;
# let dataset = dataset.build(&dev)?;
// Create a classifier
let classifier = TextClassifier::<SentenceType>::new(Classifier::new(
    &dev,
    ClassifierConfig::new().layers_dims([10]),
)?);

// Train the classifier
classifier.train(
    &dataset, // The dataset to train on
    100,      // The number of epochs to train for
    0.0003,   // The learning rate
    50,       // The batch size
    |_| {},   // The callback to run as the model trains
)?;

// Run the classifier on some input
loop {
    let input = prompt_input("Input: ").unwrap();
    let embedding = bert.embed(input).await?;
    let output = classifier.run(embedding)?;
    println!("Output: {:?}", output);
}
# }
```
