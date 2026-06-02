use fusor_tile_ir::F32Bits;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Flash-attention tensor dimensions.
pub struct FlashAttentionDims {
    /// Batch size.
    pub batch: u32,
    /// Number of query heads.
    pub num_heads: u32,
    /// Number of key/value heads.
    pub num_kv_heads: u32,
    /// Query sequence length.
    pub q_seq_len: u32,
    /// Key/value sequence length.
    pub kv_seq_len: u32,
    /// Per-head embedding dimension.
    pub head_dim: u32,
}

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

#[derive(Clone, Debug, PartialEq, Eq)]
/// Metadata for streaming flash attention.
pub struct FlashAttentionMeta {
    /// Logical attention dimensions.
    pub dims: FlashAttentionDims,
    /// Attention scale applied to QK scores.
    pub scale: F32Bits,
    /// Query tensor metadata.
    pub q_meta: TensorMeta,
    /// Key tensor metadata.
    pub k_meta: TensorMeta,
    /// Value tensor metadata.
    pub v_meta: TensorMeta,
    /// Optional additive mask metadata.
    pub mask_meta: Option<TensorMeta>,
    /// Output tensor metadata.
    pub output_meta: TensorMeta,
    /// Dispatch grid used for the generated tile program.
    pub dispatch_size: [u32; 3],
    /// When `true`, the kernel applies a strict lower-triangular causal mask
    /// (kv_idx <= q_idx) by skipping out-of-bound KV chunks/lanes. The
    /// `mask_meta` field must be `None` in this case — no additive mask is
    /// loaded.
    pub causal: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Metadata for the small F32 flash-decode kernel.
pub struct FlashDecodeSmallMeta {
    /// Logical decode dimensions.
    pub dims: FlashAttentionDims,
    /// Attention scale applied to QK scores.
    pub scale: F32Bits,
    /// Default active KV length. Runtime params may override this.
    pub active_kv_len: u32,
    /// Workgroup block size to use for decode.
    pub decode_block: u32,
    /// Whether to use the tiled decode path for long active KV lengths.
    pub tiled: bool,
    /// Number of KV tiles used by the split decode path.
    pub split_blocks: u32,
    /// Query-heads per KV-head group.
    pub groups: u32,
    /// Query tensor element offset.
    pub q_offset: u32,
    /// Key tensor element offset.
    pub k_offset: u32,
    /// Value tensor element offset.
    pub v_offset: u32,
    /// Output tensor element offset.
    pub output_offset: u32,
    /// Query strides.
    pub q_strides: [u32; 4],
    /// Key strides.
    pub k_strides: [u32; 4],
    /// Value strides.
    pub v_strides: [u32; 4],
    /// Output strides.
    pub output_strides: [u32; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// Metadata for a direct softmax kernel over one tensor axis.
pub struct SoftmaxMeta {
    /// Logical tensor shape.
    pub shape: Vec<u32>,
    /// Softmax axis.
    pub axis: u32,
    /// Number of logical rows after removing the softmax axis.
    pub rows: u32,
    /// Length of the softmax axis.
    pub axis_len: u32,
    /// Workgroup block size used for one axis tile.
    pub block: u32,
    /// Number of axis tiles for split softmax.
    pub split_blocks: u32,
    /// Input tensor metadata.
    pub input_meta: TensorMeta,
    /// Output tensor metadata.
    pub output_meta: TensorMeta,
    /// Dispatch grid used by the generated tile program.
    pub dispatch_size: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Vec4 RMS-norm offsets, strides, and scalar parameters.
pub struct RmsNormVec4Meta {
    /// Dense column count.
    pub cols: u32,
    /// Column count in packed vec4 elements.
    pub cols_vec: u32,
    /// Epsilon added before reciprocal square root.
    pub eps: F32Bits,
    /// Input vec4 offset.
    pub input_offset_vec: u32,
    /// Input row stride in vec4 elements.
    pub input_row_stride_vec: u32,
    /// Optional residual vec4 offset.
    pub residual_offset_vec: Option<u32>,
    /// Residual row stride in vec4 elements.
    pub residual_row_stride_vec: u32,
    /// Weight vec4 offset.
    pub weight_offset_vec: u32,
    /// Optional bias vec4 offset.
    pub bias_offset_vec: Option<u32>,
    /// Output vec4 offset.
    pub output_offset_vec: u32,
    /// Output row stride in vec4 elements.
    pub output_row_stride_vec: u32,
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
