# Text Completion Models

[`TextCompletionModelExt`] is the main trait for text generation models. Any model that implements either [TextCompletionModel`] or [`StructuredTextCompletionModel`] can be used with this trait.

The simplest way to use a model is to create a model and call [`TextCompletionModelExt::complete`]. The response builder that is returned can awaited to get the full response:

```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() {
    let mut llm = Llama::new().await.unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{prompt}");
    let mut completion = llm
        .complete(prompt)
        .await
        .unwrap();
    println!("{completion}");
}
```

Or use the response as a [`Stream`]:

```rust, no_run
use kalosm::language::*;
use std::io::Write;

#[tokio::main]
async fn main() {
    let mut llm = Llama::new().await.unwrap();
    let prompt = "The following is a 300 word essay about why the capital of France is Paris:";
    print!("{prompt}");
    let mut completion = llm
        .complete(prompt);
    while let Some(token) = completion.next().await {
        print!("{token}");
        std::io::stdout().flush().unwrap();
    }
}
```

## Changing the sampler

You can modify the response builder any time before reading the response. The sampler chooses the next token from the probability distribution the model generates. It can make the response more or less predictable and prevent repetition. You can call [`TextCompletionBuilder::with_sampler`] to set a sampler the model will use when completing the text:

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
let model = Llama::new().await.unwrap();
// Create the generation parameters to use for the text completion session
let sampler = GenerationParameters::default().with_temperature(0.8);
// Create a completion request with the generation parameters
let mut stream = model.complete("Here is a list of 5 primes: ").with_sampler(sampler);
stream.to_std_out().await.unwrap();
# }
```

## Structured Generation

Along with the sampler, you can use structured generation to force the output of a model to conform to a specific format a parser defines.

### Defining the parser

There are a few different ways create a parser for structured generation:

1. Derive a parser for your data
2. Create a parser from the set of prebuilt combinators
3. Create a parser from a regex

#### Deriving a parser from a struct

The simplest way to get started is to derive a parser for your data:

```rust, no_run
# use kalosm::language::*;
#[derive(Parse, Clone)]
struct Pet {
    name: String,
    age: u32,
    description: String,
}
```

Then you can generate text that works with the parser in a [`Task`](https://docs.rs/kalosm/latest/kalosm/language/struct.Task.html):

```rust, no_run
# use kalosm::language::*;
# #[derive(Parse, Clone, Debug)]
# struct Pet {
#     name: String,
#     age: u32,
#     description: String,
# }
#[tokio::main]
async fn main() {
    // First create a model
    let model = Llama::new().await.unwrap();
    // Then create a parser for your data. Any type that implements the `Parse` trait has the `new_parser` method
    let parser = Pet::new_parser();
    // Create a text completion stream with the constraints
    let description = model.complete("JSON for an adorable dog named ruffles: ")
        .with_constraints(parser);
    // Finally, await the stream to get the parsed response
    let pet: Pet = description.await.unwrap();
    println!("{pet:?}");
}
```

#### Creating a Parser from the Set of Prebuilt Combinators

Kalosm also provides a set of prebuilt combinators for creating more complex parsers. You can use these combinators to create a parser with a custom format:

```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() {
    // First create a model
    let model = Llama::new().await.unwrap();
    // Then create a parser for your custom format
    let parser = LiteralParser::from("[")
        .ignore_output_then(String::new_parser())
        .then_literal(", ")
        .then(u8::new_parser())
        .then_literal(", ")
        .then(String::new_parser())
        .then_literal("]");
    // Create a text completion stream with the constraints
    let description = model.complete("JSON for an adorable dog named ruffles: ")
        .with_constraints(parser);
    // Finally, await the stream to get the parsed response
    let ((name, age), description) = description.await.unwrap();
    println!("{name} {age} {description}");
}
```

#### Creating a Parser from a Regex

You can also create a parser from a regex:

```rust, no_run
use kalosm::language::*;

#[tokio::main]
async fn main() {
    // First create a model
    let model = Llama::new().await.unwrap();
    // Then create a parser for your data. Any
    let parser = RegexParser::new(r"\[(\w+), (\d+), (\w+)\]").unwrap();
    // Create a text completion stream with the constraints
    let mut description = model.complete("JSON for an adorable dog named ruffles in the form [\"Pet name\", age number, \"Pet description\"]: ")
        .with_constraints(parser);
    // Finally, run the task. Unlike derived and custom parsers, regex parsers do not provide a useful output type
    description.to_std_out().await.unwrap();
}
```

### Text Completion with Constraints

Once you have a parser, you can force the model to generate text that conforms to that parser with the [`TextCompletionBuilder::with_constraints`]:

```rust, no_run
use kalosm::language::*;

#[derive(Parse, Clone, Debug)]
struct Pet {
    name: String,
    age: u32,
    description: String,
}

#[tokio::main]
async fn main() {
    // First create a model
    let model = Llama::new().await.unwrap();
    // Then create a parser for your data. Any type that implements the `Parse` trait has the `new_parser` method
    let parser = Pet::new_parser();
    // Create a text completion stream with the constraints
    let description = model.complete("JSON for an adorable dog named ruffles: ")
        .with_constraints(parser);
    // Finally, await the stream to get the parsed response
    let pet: Pet = description.await.unwrap();
    println!("{pet:?}");
}
```
