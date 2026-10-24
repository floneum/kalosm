# Kalosm Llama

Kalosm Llama is the transformer implementation backing llama, mistral, phi and qwen models in kalosm. It contains the implementation of the llama model as well as support for
structured generation through the [`Parse`](https://docs.rs/kalosm-sample/latest/kalosm_sample/derive.Parse.html) macro.


The main entrypoint for this crate is the [`Llama`](https://docs.rs/kalosm-llama/latest/kalosm_llama/struct.Llama.html) struct. After you create a model with [`Llama::builder()`](https://docs.rs/kalosm-llama/latest/kalosm_llama/struct.Llama.html#method.builder),
you can use the [`ChatModelExt`](https://docs.rs/kalosm-language/0.4.2/kalosm_language/prelude/trait.ChatModelExt.html#method.chat) trait to start a chat session or start a task.


## Chat

```sh
cargo run -j 1 -p kalosm --example chat --features llama --release
```

The builder uses the ordinary Fusor GPU device. Compiler optimizations are
shared with other tensor workloads and selected from graph structure and
reported device capabilities. No model preset or adapter-name check is needed.
Quantized GGUF weights share their mapped source bytes through Fusor's shared
storage interface; keep those model files unchanged while a model is alive.

## Example

```rust
#![allow(unused)]
use kalosm::language::*;
use std::{io::Write, sync::Arc};

#[tokio::main]
async fn main() {
    let llm = Llama::phi_3().await.unwrap();
    let prompt = "Generate a list of 4 pets in JSON form with a name, description, color, and diet";

    #[derive(Debug, Clone, Parse)]
    struct Pet {
        name: String,
        description: String,
        color: String,
        size: Size,
        diet: Diet,
    }

    #[derive(Debug, Clone, Parse)]
    enum Diet {
        #[parse(rename = "carnivore")]
        Carnivore,
        #[parse(rename = "herbivore")]
        Herbivore,
        #[parse(rename = "omnivore")]
        Omnivore,
    }

    #[derive(Debug, Clone, Parse)]
    enum Size {
        #[parse(rename = "small")]
        Small,
        #[parse(rename = "medium")]
        Medium,
        #[parse(rename = "large")]
        Large,
    }

    println!("# with constraints");

    let task = llm
        .task("You generate realistic JSON placeholders")
        .with_constraints(Arc::new(<[Pet; 4] as Parse>::new_parser()));
    let stream = task.run(prompt);

    time_stream(stream).await;

    println!("\n\n# without constraints");

    let task = llm.task("You generate realistic JSON placeholders");
    let stream = task(&prompt);

    time_stream(stream).await;
}

async fn time_stream(mut stream: impl TextStream + Unpin) {
    let start_time = std::time::Instant::now();
    let mut tokens = 0;
    let mut string_length = 0;
    while let Some(token) = stream.next().await {
        tokens += 1;
        string_length += token.len();
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
    let elapsed = start_time.elapsed();
    println!("\n\nGenerated {tokens} tokens ({string_length} characters) in {elapsed:?}");
    println!(
        "Tokens per second: {:.2}",
        tokens as f64 / elapsed.as_secs_f64()
    );
}
```
