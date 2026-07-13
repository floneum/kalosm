//! Native-only conformance cases.
//!
//! These cases use the CPU device as a differential baseline and are never run
//! in the browser. Most are registered in [`super::registry`], which generates
//! one `#[tokio::test]` per case; the cases that aren't (`should_panic`,
//! `vision_block_pattern`) carry their own `#[cfg(test)]` test module.

pub mod attention_ops;
pub mod cache_ops;
pub mod dtypes;
pub mod elementwise_ops;
pub mod fusion_behavior;
pub mod fusion_correctness;
pub mod layer_ops;
pub mod layout_ops;
pub mod matmul_conv_pool;
pub mod normalization_ops;
pub mod quantized_matmul;
pub mod quantized_matmul_batched;
pub mod quantized_matmul_fusion;
pub mod quantized_matmul_paired;
pub mod rank_and_empty;
pub mod reductions_indexing;
pub mod rope_ops;
pub mod should_panic;
pub mod tensor_construction_smoke;

// Uses `std::time`/`std::env` for a flush-timing comparison; native-only.
#[cfg(not(target_arch = "wasm32"))]
pub mod vision_block_pattern;
