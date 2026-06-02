use comfy_table::{Cell, Color, Row, Table};
use kalosm::language::*;
use surrealdb::{engine::local::SurrealKv, Surreal};

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt::init();

        let exists = std::path::Path::new("./db").exists();

        // Create database connection
        let db = Surreal::new::<SurrealKv>("./db/temp.db").await.unwrap();

        // Select a specific namespace / database
        db.use_ns("test").use_db("test").await.unwrap();

        // Create a chunker splits the document into chunks to be embedded
        let chunker = SemanticChunker::new();

        // Create a table in the surreal database to store the embeddings
        let document_table = db
            .document_table_builder("documents")
            .with_chunker(chunker)
            .at("./db/embeddings.db")
            .build::<Document>()
            .await
            .unwrap();

        if !exists {
            // If the database is new, add documents to it
            let start_time = std::time::Instant::now();
            std::fs::create_dir_all("documents").unwrap();
            let context = [
                "https://floneum.com/kalosm/docs",
                "https://floneum.com/kalosm/docs/reference/web_scraping",
                "https://floneum.com/kalosm/docs/reference/transcription",
                "https://floneum.com/kalosm/docs/reference/image_segmentation",
                "https://floneum.com/kalosm/docs/reference/image_generation",
                "https://floneum.com/kalosm/docs/reference/llms",
                "https://floneum.com/kalosm/docs/reference/llms/structured_generation",
                "https://floneum.com/kalosm/docs/reference/llms/context",
                "https://floneum.com/kalosm/docs/guides/retrieval_augmented_generation",
            ]
            .iter()
            .map(|url| Url::parse(url).unwrap());

            document_table.add_context(context).await.unwrap();
            println!("Added context in {:?}", start_time.elapsed());
        }

        loop {
            // Get the user's question
            let user_question = prompt_input("Query: ").unwrap();

            let nearest_5 = document_table
                .search(user_question)
                .with_results(5)
                .await
                .unwrap();

            // Display the results in a pretty table
            let mut table = Table::new();
            table.set_content_arrangement(comfy_table::ContentArrangement::DynamicFullWidth);
            table.load_preset(comfy_table::presets::UTF8_FULL);
            table.apply_modifier(comfy_table::modifiers::UTF8_ROUND_CORNERS);
            table.set_header(vec!["Score", "Value"]);

            for result in nearest_5 {
                let mut row = Row::new();
                let color = if result.distance < 0.25 {
                    Color::Green
                } else if result.distance < 0.75 {
                    Color::Yellow
                } else {
                    Color::Red
                };
                row.add_cell(Cell::new(result.distance).fg(color))
                    .add_cell(Cell::new(result.text()));
                table.add_row(row);
            }

            println!("{table}");
        }
    });
}
