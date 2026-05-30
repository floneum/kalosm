use comfy_table::{Cell, Color, Row, Table};
use kalosm::language::*;
use surrealdb::{engine::local::SurrealKv, Surreal};

fn main() {
    pollster::block_on(async {
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
            .with_hybrid_search()
            .at("./db/embeddings.db")
            .build::<Document>()
            .await
            .unwrap();

        // Populate db if new
        if !exists {
            println!("📝 Adding sample articles...\n");

            let articles = vec![
                Document::from_parts(
                    "Getting Started with Rust".to_string(),
                    "Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. It achieves memory safety without garbage collection.".to_string(),
                ),
                Document::from_parts(
                    "Introduction to Python".to_string(),
                    "Python is an interpreted, high-level programming language known for its simplicity and readability. It's widely used in web development, data science, and machine learning.".to_string(),
                ),
                Document::from_parts(
                    "Building Web APIs with Rust".to_string(),
                    "Actix-web and Rocket are popular frameworks for building fast, reliable web APIs in Rust. They leverage Rust's type system for compile-time guarantees.".to_string(),
                ),
                Document::from_parts(
                    "Machine Learning in Python".to_string(),
                    "Python's ecosystem includes powerful libraries like TensorFlow, PyTorch, and scikit-learn for machine learning. These tools make it easy to build and train neural networks.".to_string(),
                ),
                Document::from_parts(
                    "Database Design Patterns".to_string(),
                    "Effective database design involves normalization, indexing strategies, and understanding query patterns. Vector databases are emerging for similarity search in AI applications.".to_string(),
                ),
                Document::from_parts(
                    "Async Programming in Rust".to_string(),
                    "Rust's async/await syntax enables efficient concurrent programming. Tokio is the most popular async runtime for building scalable network applications.".to_string(),
                ),
            ];

            document_table
                .add_context(articles.clone())
                .await
                .expect("documents should be added as context");
            println!("✅ Added {} articles", articles.len());
        } else {
            println!("✅ Using existing database");
        }

        loop {
            // Get the user's question
            let user_question = prompt_input("Query: ").unwrap();

            let nearest_3 = document_table
                .hybrid_search(user_question)
                .with_results(3)
                .run_weighted()
                .await
                .unwrap();

            // Display the results in a pretty table
            let mut table = Table::new();
            table.set_content_arrangement(comfy_table::ContentArrangement::DynamicFullWidth);
            table.load_preset(comfy_table::presets::UTF8_FULL);
            table.apply_modifier(comfy_table::modifiers::UTF8_ROUND_CORNERS);
            table.set_header(vec!["Score", "Value"]);

            for result in nearest_3 {
                let mut row = Row::new();
                let color = if result.score >= 0.75 {
                    Color::Green
                } else if result.score > 0.55 {
                    Color::Yellow
                } else {
                    Color::Red
                };
                row.add_cell(Cell::new(result.score).fg(color))
                    .add_cell(Cell::new(result.record.body()));
                table.add_row(row);
            }

            println!("{table}");
        }
    });
}
