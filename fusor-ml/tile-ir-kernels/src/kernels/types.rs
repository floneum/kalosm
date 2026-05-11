use fusor_tile_ir::F32Bits;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashAttentionDims {
    pub batch: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub q_seq_len: u32,
    pub kv_seq_len: u32,
    pub head_dim: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorMeta {
    pub strides: Vec<u32>,
    pub offset: u32,
}

impl TensorMeta {
    pub fn new(strides: Vec<u32>, offset: u32) -> Self {
        Self { strides, offset }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlashAttentionMeta {
    pub dims: FlashAttentionDims,
    pub scale: F32Bits,
    pub q_meta: TensorMeta,
    pub k_meta: TensorMeta,
    pub v_meta: TensorMeta,
    pub mask_meta: Option<TensorMeta>,
    pub output_meta: TensorMeta,
    pub dispatch_size: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashDecodeSmallMeta {
    pub dims: FlashAttentionDims,
    pub scale: F32Bits,
    pub active_kv_len: u32,
    pub decode_block: u32,
    pub tiled: bool,
    pub groups: u32,
    pub q_offset: u32,
    pub k_offset: u32,
    pub v_offset: u32,
    pub output_offset: u32,
    pub q_strides: [u32; 4],
    pub k_strides: [u32; 4],
    pub v_strides: [u32; 4],
    pub output_strides: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RmsNormVec4Meta {
    pub cols: u32,
    pub cols_vec: u32,
    pub eps: F32Bits,
    pub input_offset_vec: u32,
    pub input_row_stride_vec: u32,
    pub residual_offset_vec: Option<u32>,
    pub residual_row_stride_vec: u32,
    pub weight_offset_vec: u32,
    pub bias_offset_vec: Option<u32>,
    pub output_offset_vec: u32,
    pub output_row_stride_vec: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopKChunkMeta {
    pub input_len: u32,
    pub output_per_chunk: u32,
    pub input_offset: u32,
    pub input_stride: u32,
    pub processors: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TopKExactnessMeta {
    pub chunks: u32,
    pub candidate_count: u32,
    pub output_per_chunk: u32,
    pub top_k: u32,
    pub top_values_offset: u32,
    pub top_values_stride: u32,
    pub chunk_values_offset: u32,
    pub chunk_values_stride: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MergeTopKMeta {
    pub chunks: u32,
    pub chunk_len: u32,
    pub chunk_stride: u32,
    pub input_len: u32,
    pub k: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mirostat2Meta {
    pub top_k: u32,
    pub ids_offset: u32,
    pub ids_stride: u32,
    pub values_offset: u32,
    pub values_stride: u32,
    pub has_exactness_flag: bool,
}
