#[derive(Clone, Debug, PartialEq, Eq)]
/// Runtime tensor strides and base offset.
pub struct TensorMeta {
    /// Row-major logical strides for the tensor rank used by the kernel.
    pub strides: Vec<u32>,
    /// Element offset into the bound buffer.
    pub offset: u32,
}

impl TensorMeta {
    /// Create tensor metadata from strides and an element offset.
    pub fn new(strides: Vec<u32>, offset: u32) -> Self {
        Self { strides, offset }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Metadata for one top-k chunk pass.
pub struct TopKChunkMeta {
    /// Input length covered by all chunks.
    pub input_len: u32,
    /// Number of candidates emitted per chunk.
    pub output_per_chunk: u32,
    /// Input tensor offset.
    pub input_offset: u32,
    /// Input tensor stride.
    pub input_stride: u32,
    /// Whether multiple processors contribute chunk outputs.
    pub processors: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Metadata for exactness validation of chunked top-k.
pub struct TopKExactnessMeta {
    /// Number of chunks.
    pub chunks: u32,
    /// Candidate count per chunk.
    pub candidate_count: u32,
    /// Number of values emitted per chunk.
    pub output_per_chunk: u32,
    /// Requested final top-k.
    pub top_k: u32,
    /// Offset of the merged top values.
    pub top_values_offset: u32,
    /// Stride of the merged top values.
    pub top_values_stride: u32,
    /// Offset of chunk-local values.
    pub chunk_values_offset: u32,
    /// Stride of chunk-local values.
    pub chunk_values_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Metadata for merging sorted top-k chunks.
pub struct MergeTopKMeta {
    /// Number of chunks to merge.
    pub chunks: u32,
    /// Number of candidates per chunk.
    pub chunk_len: u32,
    /// Stride between chunk candidates.
    pub chunk_stride: u32,
    /// Original input length.
    pub input_len: u32,
    /// Requested final top-k.
    pub k: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Metadata for Mirostat v2 sampling over a sorted top-k list.
pub struct Mirostat2Meta {
    /// Number of sorted candidates.
    pub top_k: u32,
    /// Token id offset.
    pub ids_offset: u32,
    /// Token id stride.
    pub ids_stride: u32,
    /// Value/logit offset.
    pub values_offset: u32,
    /// Value/logit stride.
    pub values_stride: u32,
    /// Whether an exactness flag binding is provided.
    pub has_exactness_flag: bool,
}
