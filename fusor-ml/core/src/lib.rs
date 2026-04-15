// This crate is still heavily in development. We don't need as strict of a linting policy as the
// most of the crates in the workspace
#![allow(missing_docs)]

pub use device::*;
pub use element_wise::CastTensor;
pub use fusor_gguf::GgufReadError;
pub use fusor_types::{Layout, SlidingWindow, TILE_SIZE, TensorSlice, slice_shape, slice_strides};
pub use quantized::*;
pub use rank::*;
pub use reduce::*;
pub use tensor::MappedBuffer;
pub use tensor::*;

// Re-export wasm-compatible Send/Sync traits
pub use wgpu::{WasmNotSend, WasmNotSendSync, WasmNotSync};

pub(crate) use element_wise::*;
pub use matmul::*;
pub(crate) use pair_wise::*;
pub use resize::ShapeWithOneHole;
pub use varbuilder::{ShardedVarBuilder, VarBuilder};

pub mod cache;
mod composite;
pub use composite::{ToVec1, ToVec2, ToVec3};
mod compute_graph;
mod device;
mod element_wise;
mod index_select;
pub mod layers;
mod layout;
mod map_layout;
pub mod matmul;
mod mir;
mod nary_wise;
mod pair_wise;
mod quantized;
mod quantized_types_wgsl;
mod rank;
mod reduce;
mod resize;
mod slice_assign;
mod tensor;
mod util;
mod varbuilder;
mod visit_tiled;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Failed to find a suitable device {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
    #[error("GGUF error {0}")]
    GgufError(#[from] fusor_gguf::GgufReadError),
    #[error("WGSL async buffer error {0}")]
    BufferAsyncError(#[from] wgpu::BufferAsyncError),
    #[error("VarBuilder error {0}")]
    VarBuilder(String),
    #[error("Other error {0}")]
    Other(String),
}

impl Error {
    pub fn msg<S: Into<String>>(s: S) -> Self {
        Error::Other(s.into())
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::GgufError(e.into())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
