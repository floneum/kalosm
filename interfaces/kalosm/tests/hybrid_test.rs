use kalosm::language::*;
// use std::future::Future;
use surrealdb::{
    engine::local::{Db, Mem},
    Surreal,
};

// Helper to create test database
async fn setup_db() -> Result<Surreal<Db>, Box<dyn std::error::Error>> {
    let db = Surreal::new::<Mem>(()).await?;
    db.use_ns("test").use_db("test").await?;

    Ok(db)
}

// Helpers to create sample documents
fn sample_blog_posts() -> Vec<Document> {
    vec![
        Document::from_parts(
            "Getting Started with Rust Async",
            "Async programming in Rust uses the async/await syntax. This allows you to write asynchronous code that looks like synchronous code. Tokio is the most popular async runtime for Rust.",
        ),
        Document::from_parts(
            "Understanding Rust Ownership",
            "Ownership is Rust's most unique feature. It enables Rust to make memory safety guarantees without needing a garbage collector. Every value has a variable that's its owner.",
        ),
        Document::from_parts(
            "Building REST APIs with Python Flask",
            "Flask is a lightweight WSGI web application framework in Python. It's designed to make getting started quick and easy, with the ability to scale up to complex applications.",
        ),
        Document::from_parts(
            "Introduction to Vector Databases",
            "Vector databases store data as high-dimensional vectors, which are mathematical representations of features. They enable similarity search and are crucial for AI applications like RAG.",
        ),
        Document::from_parts(
            "Python vs Rust: Performance Comparison",
            "While Python is easier to learn and faster to write, Rust offers superior performance and memory safety. The choice depends on your use case and team expertise.",
        ),
    ]
}

fn sample_products() -> Vec<Document> {
    vec![
        Document::from_parts(
            "Mechanical Keyboard",
            "Professional mechanical keyboard with Cherry MX Blue switches. Perfect for programmers who love tactile feedback. RGB backlit with customizable keys.",
        ),
        Document::from_parts(
            "Ergonomic Office Chair",
            "Premium ergonomic office chair with lumbar support. Adjustable height and armrests. Mesh back for breathability during long coding sessions.",
        ),
        Document::from_parts(
            "USB-C Docking Station",
            "Universal USB-C dock with dual 4K monitor support. Includes multiple USB ports, ethernet, and power delivery. Compatible with MacBook and laptops.",
        ),
        Document::from_parts(
            "Standing Desk Converter",
            "Height adjustable standing desk converter. Easy to use, fits on existing desk. Promotes better posture and reduces back pain from sitting.",
        ),
        Document::from_parts(
            "Wireless Mouse",
            "Ergonomic wireless mouse with precision tracking. Long battery life and comfortable grip for extended use. Works on any surface.",
        ),
    ]
}

#[tokio::test]
async fn test_rrf_hybrid_search() -> Result<(), Box<dyn std::error::Error>> {
    let table_name = "blog_posts_rrf";
    let db = setup_db().await?;
    let chunker = SemanticChunker::new();

    let table = db
        .document_table_builder(table_name)
        .with_chunker(chunker)
        .with_hybrid_search()
        .build::<Document>()
        .await?;

    // Insert documents
    let docs = sample_blog_posts();
    table.add_context(docs).await?;

    let results = table
        .hybrid_search("async programming")
        .with_results(3)
        .run_rrf()
        .await?;

    println!("\n=== RRF Hybrid Search Results ===");
    for (i, result) in results.iter().enumerate() {
        println!("{}. {}", i + 1, result.record.title());
        println!("   RRF Score: {:.4}", result.score);
        println!(
            "   Semantic: {:.4}, Keyword: {:.4}",
            result.semantic_score, result.keyword_score
        );
        println!();
    }

    // First result should be about async programming
    assert!(
        results[0].record.title().contains("Async") && results[0].record.body().contains("async")
    );

    Ok(())
}

#[tokio::test]
async fn test_weighted_hybrid_search() -> Result<(), Box<dyn std::error::Error>> {
    let table_name = "blog_posts_weighted";
    let db = setup_db().await?;
    let chunker = SemanticChunker::new();

    let table = db
        .document_table_builder(table_name)
        .with_chunker(chunker)
        .with_hybrid_search()
        .build::<Document>()
        .await?;

    let docs = sample_blog_posts();
    table.add_context(docs).await?;

    let results = table
        .hybrid_search("ownership memory safety")
        .with_results(3)
        .with_semantic_weight(0.6)
        .with_keyword_weight(0.4)
        .run_weighted()
        .await?;

    println!("\n=== Weighted Hybrid Search Results ===");
    for (i, result) in results.iter().enumerate() {
        println!("{}. {}", i + 1, result.record.title());
        println!("   Combined Score: {:.4}", result.score);
        println!(
            "   Semantic: {:.4}, Keyword: {:.4}",
            result.semantic_score, result.keyword_score
        );
        println!();
    }

    // Should find the ownership article in top results
    assert!(results
        .iter()
        .any(|r| r.record.title().contains("Ownership")));

    Ok(())
}

#[tokio::test]
async fn test_keyword_only_match() -> Result<(), Box<dyn std::error::Error>> {
    let table_name = "products_keyword";
    let db = setup_db().await?;
    let chunker = SemanticChunker::new();

    let table = db
        .document_table_builder(table_name)
        .with_chunker(chunker)
        .with_hybrid_search()
        .build::<Document>()
        .await?;

    let docs = sample_products();
    table.add_context(docs).await?;

    // Search for exact product name
    let results = table
        .hybrid_search("USB-C")
        .with_results(5)
        .run_rrf()
        .await?;

    println!("\n=== Keyword Match Test ===");
    for (i, result) in results.iter().enumerate() {
        println!("{}. {}", i + 1, result.record.title());
        println!("   Score: {:.4}", result.score);
        println!();
    }

    // USB-C docking station should be first
    assert!(results[0].record.title().contains("USB-C"));

    Ok(())
}
