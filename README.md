<h1 align="center">Kalosm</h1>
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
  <!-- Discord -->
  <a href="https://discord.gg/dQdmhuB8q5">
    <img src="https://img.shields.io/discord/1120130300236800062?logo=discord&style=flat-square" alt="Discord Link" />
  </a>
</div>

Kalosm is an ecosystem of crates that make it easy to develop applications that use local or remote AI models. There are try main projects in this repo:

- [Kalosm](./interfaces/kalosm): A simple interface for pre-trained models in rust
- [Fusor](./fusor-ml/core): A runtime for quantized ML inference. Fusor uses WGPU to run models on any accelerator natively or in the browser

## Kalosm

[Kalosm](./interfaces/kalosm/) is a simple interface for pre-trained models in Rust. It makes it easy to interact with pre-trained, language, audio, and image models.

### Model Support

Kalosm supports a variety of models. Here is a list of the models that are currently supported:

| Model            | Modality | Size       | Description                            | Quantized | GPU Accelerated | Example                                                               |
| ---------------- | -------- | ---------- | -------------------------------------- | --------- | --------------- | --------------------------------------------------------------------- |
| Llama            | Text     | 1b-70b     | General purpose language model         | ✅        | ✅              | [llama 3 chat](interfaces/kalosm/examples/chat.rs)                    |
| Mistral          | Text     | 7-13b      | General purpose language model         | ✅        | ✅              | [mistral chat](interfaces/kalosm/examples/chat-mistral-2.rs)          |
| Phi              | Text     | 2b-4b      | Small reasoning focused language model | ✅        | ✅              | [phi 3 chat](interfaces/kalosm/examples/chat-phi-3.rs)                |
| Whisper          | Audio    | 20MB-1GB   | Audio transcription model              | ✅        | ✅              | [live whisper transcription](interfaces/kalosm/examples/transcribe.rs) |
| Segment Anything | Image    | 50MB-400MB | Image segmentation model               | ✅        | ✅              | [Image Segmentation](interfaces/kalosm/examples/segment-image.rs)     |
| Bert             | Text     | 100MB-1GB  | Text embedding model                   | ✅        | ✅              | [Semantic Search](interfaces/kalosm/examples/semantic-search.rs)      |

### Utilities

Kalosm also supports a variety of utilities around pre-trained models. These include:

- [Extracting, formatting and retrieving context for LLMs](./interfaces/kalosm/examples/context_extraction.rs): [Extract context from txt/html/docx/md/pdf](./interfaces/kalosm/examples/context_extraction.rs) [chunk that context](./interfaces/kalosm/examples/chunking.rs) [then search for relevant context with vector database integrations](./interfaces/kalosm/examples/semantic-search.rs)
- [Transcribing audio from your microphone or file](./interfaces/kalosm/examples/transcribe.rs)
- [Crawling and scraping content from web pages](./interfaces/kalosm/examples/crawl.rs)

### Structured Generation

Kalosm supports structured generation with arbitrary parsers. It uses a custom parser engine and sampler and structure-aware acceleration to make structure generation even faster than uncontrolled text generation. You can take any rust type and add `#[derive(Parse, Schema)]` to make it usable with structured generation:

```rust
use kalosm::language::*;

/// A fictional character
#[derive(Parse, Schema, Clone, Debug)]
struct Character {
    /// The name of the character
    #[parse(pattern = "[A-Z][a-z]{2,10} [A-Z][a-z]{2,10}")]
    name: String,
    /// The age of the character
    #[parse(range = 1..=100)]
    age: u8,
    /// A description of the character
    #[parse(pattern = "[A-Za-z ]{40,200}")]
    description: String,
}

#[tokio::main]
async fn main() {
    // First create a model. Chat models tend to work best with structured generation
    let model = Llama::phi_3().await.unwrap();
    // Then create a task with the parser as constraints
    let task = model.task("You generate realistic JSON placeholders for characters")
        .typed();
    // Finally, run the task
    let mut stream = task(&"Create a list of random characters", &model);
    stream.to_std_out().await.unwrap();
    let characters: [Character; 10] = stream.await.unwrap();
    println!("{characters:?}");
}
```

https://github.com/user-attachments/assets/8900f57d-55c8-4d4a-a67b-73beab1e5155

In addition to regex, you can provide your own grammar to generate structured data. This lets you constrain the response to any structure you want including complex data structures like JSON, HTML, and XML.

### Kalosm Quickstart!

This quickstart will get you up and running with a simple chatbot. Let's get started!

> A more complete guide for Kalosm is available on the [Kalosm website](https://floneum.com/kalosm/), and examples are available in the [examples folder](https://github.com/floneum/floneum/tree/main/interfaces/kalosm/examples).

1. Install [rust](https://rustup.rs/)
2. Create a new project:

```sh
cargo new kalosm-hello-world
cd ./kalosm-hello-world
```

3. Add Kalosm as a dependency

```sh
cargo add kalosm --features llama
cargo add tokio --features full
```

4. Add this code to your `main.rs` file

```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  let model = Llama::phi_3().await?;
  let mut chat = model.chat()
    .with_system_prompt("You are a pirate called Blackbeard");

  loop {
    chat(&prompt_input("\n> ")?)
      .to_std_out()
      .await?;
  }
}
```

5. Run your application with:

```sh
cargo run --release
```

[chat bot demo](https://github.com/floneum/floneum/assets/66571940/e4e76efb-6387-4fcd-aa3c-aa556e840334)

## Fusor

⚠️ Fusor is still early in development and is not ready for production use. Fusor will serve as the backend for Kalosm in the 0.5 release to enable web and AMD support

[Fusor](./fusor-ml/core) is a WGPU runtime for quantized ML inference. Fusor works with the gguf file format to load quantized models. It targets uses WebGpu to target many different accelerators including Nvidia GPUs, AMD GPUs, and Metal. Most ML frameworks contain hand optimized kernels that perform a series of operations together. Fusor uses a kernel fusion compiler to make merge custom operation chains into an optimized kernel without dropping down to the shader code. This compiles to a single kernel:
```rust, ignore
fn exp_add_one(tensor: Tensor<2, f32>) -> Tensor<2, f32> {
  1. + (-tensor).exp()
}
```

## Community

If you are interested in either project, you can join the [discord](https://discord.gg/dQdmhuB8q5) to discuss the project and get help.

## Contributing

- Report issues on our [issue tracker](https://github.com/floneum/floneum/issues).
- Help other users in the discord
- If you are interested in contributing, feel free to reach out on discord
