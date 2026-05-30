Let's start with a simple chat application:

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
// Before you create a chat session, you need a model. Llama::new_chat will create a good default chat model.
let model = Llama::new_chat().await.unwrap();
// Then you can build a chat session that uses that model
let mut chat = model.chat()
    // The builder exposes methods for settings like the system prompt and constraints the bot response must follow
    .with_system_prompt("The assistant will act like a pirate");

loop {
    // To use the chat session, you need to add messages to it
    let mut response_stream = chat(&prompt_input("\n> ").unwrap());
    // And then display the response stream to the user
    response_stream.to_std_out().await.unwrap();
}
# }
```

LLMs are powerful because of their generality, but sometimes you need more control over the output. For example, you might want the assistant to start with a certain phrase, or to follow a certain format.

In kalosm, you can use constraints to guide the model's response. Constraints are a way to specify the format of the output. When generating with constraints, the model will always respond with the specified format.

Let's create a chat application that uses constraints to guide the assistant's response to always start with "Yes!":

```rust, no_run
# use kalosm::language::*;
# #[tokio::main]
# async fn main() {
let model = Llama::new_chat().await.unwrap();
// Create constraints that parses Yes! and then stops on the end of the assistant's response
let constraints = LiteralParser::new("Yes!")
    .then(model.default_assistant_constraints());
// Create a chat session with the model and the constraints
let mut chat = model.chat();

// Chat with the user
loop {
    let mut output_stream = chat(&prompt_input("\n> ").unwrap()).with_constraints(constraints.clone());
    output_stream.to_std_out().await.unwrap();
}
# }
```
