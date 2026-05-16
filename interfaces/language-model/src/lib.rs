//! # Language Model
//!
//! This crate provides a unified interface for language models. It supports streaming text, sampling, and embedding.
//!
//! ## Usage (with the kalosm-llama implementation crate)
//!
//! ```rust, no_run
//! use kalosm::language::*;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut model = Llama::phi_3().await.unwrap();
//!     let prompt = "The capital of France is ";
//!     let mut result = model.complete(prompt);
//!
//!     print!("{prompt}");
//!     while let Some(token) = result.next().await {
//!         print!("{token}");
//!     }
//! }
//! ```

#![warn(missing_docs)]

pub use futures_util::StreamExt;
pub use kalosm_sample;

#[cfg(feature = "openai")]
mod openai;
#[cfg(feature = "openai")]
pub use openai::*;
#[cfg(feature = "anthropic")]
mod claude;
#[cfg(feature = "anthropic")]
pub use claude::*;

mod embedding;
pub use embedding::*;
mod model;
pub use model::*;
mod chat;
pub use chat::*;
