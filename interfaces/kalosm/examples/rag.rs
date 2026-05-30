use kalosm::language::*;
use surrealdb::{engine::local::SurrealKv, Surreal};

fn main() -> anyhow::Result<()> {
    pollster::block_on(async {
        let exists = std::path::Path::new("./db").exists();

        // Create database connection
        let db = Surreal::new::<SurrealKv>("./db/temp.db").await?;

        // Select a specific namespace / database
        db.use_ns("test").use_db("test").await?;

        // Create a chunker splits the document into chunks to be embedded
        let chunker = SemanticChunker::new();

        // Create a table in the surreal database to store the embeddings
        let document_table = db
            .document_table_builder("documents")
            .with_chunker(chunker)
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
            let prompt = format!("{context}\n{user_question}");

            // Display the prompt to the user for debugging purposes
            println!("{prompt}");

            // And finally, respond to the user
            let mut output_stream = chat(&prompt);
            print!("Bot: ");
            output_stream.to_std_out().await?;
        }
    })
}
