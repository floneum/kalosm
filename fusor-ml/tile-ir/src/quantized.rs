use crate::ir::StorageView;

/// GGML quantization formats represented by the tiled qmatmul path.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GgmlQuantFormat {
    /// GGML `Q4_0`.
    Q4_0,
    /// GGML `Q4_1`.
    Q4_1,
    /// GGML `Q5_0`.
    Q5_0,
    /// GGML `Q5_1`.
    Q5_1,
    /// GGML `Q8_0`.
    Q8_0,
    /// GGML `Q8_1`.
    Q8_1,
    /// GGML K-quant `Q2_K`.
    Q2K,
    /// GGML K-quant `Q3_K`.
    Q3K,
    /// GGML K-quant `Q4_K`.
    Q4K,
    /// GGML K-quant `Q5_K`.
    Q5K,
    /// GGML K-quant `Q6_K`.
    Q6K,
    /// GGML K-quant `Q8_K`.
    Q8K,
}

impl GgmlQuantFormat {
    /// Number of dense rows/K elements contained in one quantized block.
    pub const fn block_elements(self) -> u32 {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Number of u32 words in the f32-scale shader layout for one block.
    pub const fn block_words(self) -> u32 {
        match self {
            Self::Q4_0 => 5,
            Self::Q4_1 => 6,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 9,
            Self::Q8_1 => 10,
            Self::Q2K => 22,
            Self::Q3K => 28,
            Self::Q4K => 37,
            Self::Q5K => 45,
            Self::Q6K => 53,
            Self::Q8K => 73,
        }
    }
}

/// A packed quantized storage matrix — kernel-input handle pairing a tile-IR
/// storage view with the quantization format and matrix dimensions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedMatrix {
    /// Storage view containing packed quantized block words.
    pub data: StorageView,
    /// Quantization format used by `data`.
    pub format: GgmlQuantFormat,
    /// Dense row/K dimension.
    pub rows: u32,
    /// Dense output-column dimension.
    pub cols: u32,
}
