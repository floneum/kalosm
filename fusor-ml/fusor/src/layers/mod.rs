//! Neural network layer implementations that work on both CPU and GPU backends.
//!
//! These layers wrap the Tensor tensor operations into convenient layer abstractions.
//!
//! All layers support loading from GGUF files via `VarBuilder` for f32 types.

mod batch_norm;
mod conv1d;
mod conv2d;
mod embedding;
mod layer_norm;
mod linear;
mod rms_norm;

pub use batch_norm::BatchNorm1d;
pub use conv1d::{Conv1d, Conv1dConfig};
pub use conv2d::{Conv2d, Conv2dConfig};
pub use embedding::Embedding;
pub use layer_norm::LayerNorm;
pub use linear::Linear;
pub use rms_norm::RmsNorm;
