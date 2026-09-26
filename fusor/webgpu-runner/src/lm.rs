//! The "watch a transformer learn to write" demo: a small language model with
//! a byte-pair tokenizer, trained live, in the browser, on a slice of
//! TinyStories.

pub mod config;
pub mod corpus;
pub mod model;
pub mod paint;
pub mod rng;
pub mod tokenizer;
pub mod ui;
