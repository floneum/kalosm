#![allow(unused)]
use kalosm::language::*;

// You can derive an efficient parser for your struct with the `Parse` trait
#[derive(Schema, Parse, Clone, Debug)]
struct Account {
    /// A summary of the account holder
    #[parse(pattern = r"[a-zA-Z,.?!\d ]{1,80}")]
    summary: String,
    /// The name of the account holder. This may be the full name of the user or a pseudonym
    #[parse(pattern = "[a-zA-Z ]{1,20}")]
    name: String,
    /// The age of the account holder
    #[parse(range = 1..=100)]
    age: u8,
}

fn main() -> anyhow::Result<()> {
    pollster::block_on(async {
        // Download default chat model or get the model from the cache
        let llm = Llama::new_chat().await?;

        // Create a task that generates a list of accounts
        let task = llm
            .task("You generate accounts based on a description of the account holder")
            // The typed combinator constraints the task for the default parser for the type you get out. (In
            // this case, Account)
            .typed();

        // Task can be called like a function with the input to the task. You can await the stream, modify the
        // constraints, or sampler
        let account: Account =
            task(&"Candice is the CEO of a fortune 500 company. She is a 30 years old.").await?;

        println!("{account:#?}");

        Ok(())
    })
}
