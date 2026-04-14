<div align="center">
  <h1>Kalosm</h1>
</div>
<div align="center">
  <!-- Crates version -->
  <a href="https://crates.io/crates/kalosm">
    <img src="https://img.shields.io/crates/v/kalosm.svg?style=flat-square"
    alt="Crates.io version" />
  </a>
  <!-- Downloads -->
  <a href="https://crates.io/crates/kalosm">
    <img src="https://img.shields.io/crates/d/kalosm.svg?style=flat-square"
      alt="Download" />
  </a>
  <!-- docs -->
  <a href="https://docs.rs/kalosm">
    <img src="https://img.shields.io/badge/docs-latest-blue.svg?style=flat-square"
      alt="docs.rs docs" />
  </a>
</div>

Kalosm is a simple interface for pre-trained models in rust. It makes it easy to interact with pre-trained, language, audio, and image models.

There are three different packages in Kalosm:

- `kalosm::language` - A simple interface for text generation and embedding models and surrounding tools. It includes support for search databases, and text collection from websites, RSS feeds, and search engines.
- `kalosm::audio` - A simple interface for audio transcription and surrounding tools. It includes support for microphone input, transcription with the `whisper` model, and voice activity detection.
- `kalosm::vision` - A simple interface for image generation and segmentation models and surrounding tools. It includes support for the `wuerstchen` and `segment-anything` models and integration with the [image](https://docs.rs/image/latest/image/) crate.

A complete guide for Kalosm is available on the [Kalosm website](https://floneum.com/kalosm/), and examples are available in the [examples folder](https://github.com/floneum/floneum/tree/main/interfaces/kalosm/examples).

## Quickstart!

1. Install [rust](https://rustup.rs/)
2. Create a new project:

```sh
cargo new next-gen-ai
cd ./next-gen-ai
```

3. Add Kalosm as a dependency

```sh
cargo add kalosm --git https://github.com/floneum/floneum --features full
cargo add tokio --features full
```

4. Add this code to your `main.rs` file

```rust, no_run
use std::io::Write;

use kalosm::{*, language::*};

#[tokio::main]
async fn main() {
    let mut llm = Llama::new().await.unwrap();
    let prompt = "The following is a 300 word essay about Paris:";
    print!("{}", prompt);

    let mut stream = llm(prompt);

    stream.to_std_out().await.unwrap();
}
```

5. Run your application with:

```sh
cargo run --release
```

## What can you do with Kalosm?

You can think of Kalosm as the plumbing between different pre-trained models and each other or the surrounding world. Kalosm makes it easy to build applications that use pre-trained models to generate text, audio, and images. Here are some examples of what you can build with Kalosm:

<details>
<summary>Local text generation</summary>

The simplest way to get started with Kalosm language is to pull in one of the local large language models and use it to generate text. Kalosm supports a streaming API that allows you to generate text in real time without blocking your main thread:

```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut llm = Llama::phi_3().await.unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{}", prompt);

    let mut stream = llm(prompt);

    stream.to_std_out().await.unwrap();

    Ok(())
}
```

</details>

<details>
<summary>Structured generation</summary>

Natural language generation is interesting, but the more interesting aspect of text is as a universal data format. You can encode any kind of data into text with a format like json. Kalosm lets you use LLMs with structured generation to create arbitrary types from natural language inputs:

```rust, no_run
use kalosm::language::*;
use std::sync::Arc;

// First, derive an efficient parser for your structured data
#[derive(Parse, Clone, Debug)]
enum Class {
    Thing,
    Person,
    Animal,
}

#[derive(Parse, Clone, Debug)]
struct Response {
    classification: Class,
}

#[tokio::main]
async fn main() {
    // Then set up a task with a prompt and constraints
    let llm = Llama::new_chat().await.unwrap();
    let task = llm.task("You classify the user's message as about a person, animal or thing in a JSON format")
        .with_constraints(Arc::new(Response::new_parser()));

    // Finally, run the task
    let response = task(&"The Kalosm library lets you create structured data from natural language inputs").await.unwrap();
    println!("{:?}", response);
}
```

</details>

<details>
<summary>Cloud models</summary>

Kalosm also supports cloud models like GPT4 with the same streaming API:

```rust, no_run
// You must set the environment variable OPENAI_API_KEY (https://platform.openai.com/account/api-keys) to run this example.
use kalosm::language::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut llm = OpenAICompatibleChatModel::builder()
        .with_gpt_4o_mini()
        .build();

    let mut chat = llm.chat();

    chat(&"What is the capital of France?").to_std_out().await?;

    Ok(())
}
```

</details>

<details>
<summary>Gather context from RSS, websites, local files, search results, and more</summary>

Kalosm makes it easy to collect text data from a variety of sources. For example, you can use Kalosm to collect text from a local folder of documents, an RSS stream, a website, or a search engine:

```rust, no_run
use kalosm::language::*;
use std::convert::TryFrom;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read an RSS stream
    let nyt = RssFeed::new(Url::parse("https://rss.nytimes.com/services/xml/rss/nyt/US.xml").unwrap());
    // Read a local folder of documents
    let mut documents = DocumentFolder::try_from(PathBuf::from("./documents")).unwrap();
    // Read a website (either from the raw HTML or inside of a headless browser)
    let page = Page::new(Url::parse("https://www.nytimes.com/live/2023/09/21/world/zelensky-russia-ukraine-news").unwrap(), BrowserMode::Static).unwrap();
    let document = page.article().await.unwrap();
    println!("Title: {}", document.title());
    println!("Body: {}", document.body());
    // Read pages from a search engine (You must have the SERPER_API_KEY environment variable set to run this example)
    let query = "What is the capital of France?";
    let api_key = std::env::var("SERPER_API_KEY").unwrap();
    let search_query = SearchQuery::new(query, &api_key, 5);
    let documents = search_query.into_documents().await.unwrap();
    let mut text = String::new();
    for document in documents {
        for word in document.body().split(' ').take(300) {
            text.push_str(word);
            text.push(' ');
        }
        text.push('\n');
    }
    println!("{}", text);

    Ok(())
}
```

</details>

<details>
<summary>Embedding powered search</summary>

Once you have your data, Kalosm includes tools to create embedding-powered search indexes. Embedding-based search lets you find documents that are semantically similar to a specific word or phrase even if no words are an exact match:

```rust, no_run
use kalosm::language::*;
use surrealdb::{engine::local::SurrealKv, Surreal};

#[tokio::main]
async fn main() {
    // Create database connection
    let db = Surreal::new::<SurrealKv>(std::env::temp_dir().join("temp.db")).await.unwrap();

    // Select a specific namespace / database
    db.use_ns("search").use_db("documents").await.unwrap();

    // Create a table in the surreal database to store the embeddings
    let document_table = db
        .document_table_builder("documents")
        .build::<Document>()
        .await
        .unwrap();

    // Add documents to the database
    document_table.add_context(DocumentFolder::new("./documents").unwrap()).await.unwrap();

    loop {
        // Get the user's question
        let user_question = prompt_input("Query: ").unwrap();

        let nearest_5 = document_table
            .search(&user_question)
            .with_results(5)
            .await
            .unwrap();

        println!("{:?}", nearest_5);
    }
}
```

</details>

<details>
<summary>Retrieval Augmented Generation</summary>

A large part of making modern LLMs performant is curating the context the models have access to. Retrieval Augmented Generation (or RAG) helps you do this by inserting context into the prompt based on a search query. For example, you can Kalosm to create a chatbot that uses context from local documents to answer questions:

```rust, no_run
use kalosm::language::*;
use surrealdb::{engine::local::SurrealKv, Surreal};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let exists = std::path::Path::new("./db").exists();

    // Create database connection
    let db = Surreal::new::<SurrealKv>("./db/temp.db").await?;

    // Select a specific namespace / database
    db.use_ns("test").use_db("test").await?;

    // Create a table in the surreal database to store the embeddings
    let document_table = db
        .document_table_builder("documents")
        .at("./db/embeddings.db")
        .build::<Document>()
        .await?;

    // If the database is new, add documents to it
    if !exists {
        std::fs::create_dir_all("documents")?;
        let context = [
            "https://floneum.com/kalosm/docs",
            "https://floneum.com/kalosm/docs/guides/retrieval_augmented_generation",
        ]
        .iter()
        .map(|url| Url::parse(url).unwrap());

        document_table.add_context(context).await?;
    }

    // Create a llama chat model
    let model = Llama::new_chat().await?;
    let mut chat = model.chat().with_system_prompt("The assistant help answer questions based on the context given by the user. The model knows that the information the user gives it is always true.");

    loop {
        // Ask the user for a question
        let user_question = prompt_input("\n> ")?;

        // Search for relevant context in the document engine
        let context = document_table
            .search(&user_question)
            .with_results(1)
            .await?
            .into_iter()
            .map(|document| {
                format!(
                    "Title: {}\nBody: {}\n",
                    document.record.title(),
                    document.record.body()
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Format a prompt with the question and context
        let prompt = format!(
            "{context}\n{user_question}"
        );

        // Display the prompt to the user for debugging purposes
        println!("{}", prompt);

        // And finally, respond to the user
        let mut output_stream = chat(&prompt);
        print!("Bot: ");
        output_stream.to_std_out().await?;
    }
}
```

</details>

<details>
<summary>Live audio transcription</summary>

Kalosm makes it easy to build up context about the world around your application.

```rust, no_run
use kalosm::sound::*;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create a new whisper model
    let model = Whisper::new().await?;

    // Stream audio from the microphone
    let mic = MicInput::default();
    let stream = mic.stream();

    // The audio into chunks based on voice activity and then transcribe those chunks
    // The model will transcribe chunks of speech that are separated by silence
    let mut text_stream = stream.transcribe(model);

    // Finally, print the text to the console
    text_stream.to_std_out().await.unwrap();

    Ok(())
}
```

</details>

<details>
<summary>Image generation</summary>

In addition to language, audio, and embedding models, Kalosm also supports image generation. For example, you can use Kalosm to generate images from text:

```rust, no_run
use kalosm::vision::*;

#[tokio::main]
async fn main() {
    let model = Wuerstchen::new().await.unwrap();
    let settings = WuerstchenInferenceSettings::new(
        "a cute cat with a hat in a room covered with fur with incredible detail",
    );
    let mut images = model.run(settings);
    while let Some(image) = images.next().await {
        if let Some(buf) = image.generated_image() {
            buf.save(&format!("{}.png",image.sample_num())).unwrap();
        }
    }
}
```

</details>

<details>
<summary>Image segmentation</summary>

Kalosm also supports image segmentation with the segment-anything model:

```rust, no_run
use kalosm::vision::*;

#[tokio::main]
async fn main() {
    let model = SegmentAnything::builder().build().await.unwrap();
    let image = image::open("examples/landscape.jpg").unwrap();
    let images = model.segment_everything(image).await.unwrap();
    for (i, img) in images.iter().enumerate() {
        img.save(&format!("{}.png", i)).unwrap();
    }
}
```

</details>
