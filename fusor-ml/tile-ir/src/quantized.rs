use crate::ir::StorageView;

/// GGML quantization formats represented by the tiled qmatmul path.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GgmlQuantFormat {
    /// GGML `Q4_0`.
    Q4_0,
    /// GGML `Q4_0` in the raw native GGUF layout.
    Q4_0Native,
    /// GGML `Q4_1`.
    Q4_1,
    /// GGML `Q5_0`.
    Q5_0,
    /// GGML `Q5_0` in the raw native GGUF layout.
    Q5_0Native,
    /// GGML `Q5_1`.
    Q5_1,
    /// GGML `Q8_0`.
    Q8_0,
    /// GGML `Q8_0` in the raw native GGUF layout.
    Q8_0Native,
    /// GGML `Q8_1`.
    Q8_1,
    /// GGML K-quant `Q2_K`.
    Q2K,
    /// GGML K-quant `Q3_K`.
    Q3K,
    /// GGML K-quant `Q4_K`.
    Q4K,
    /// GGML K-quant `Q4_K` in the raw native GGUF layout.
    Q4KNative,
    /// GGML K-quant `Q5_K`.
    Q5K,
    /// GGML K-quant `Q5_K` in the raw native GGUF layout.
    Q5KNative,
    /// GGML K-quant `Q6_K`.
    Q6K,
    /// GGML K-quant `Q6_K` in the raw native GGUF layout.
    Q6KNative,
    /// GGML K-quant `Q8_K`.
    Q8K,
}

impl GgmlQuantFormat {
    /// Number of dense rows/K elements contained in one quantized block.
    pub const fn block_elements(self) -> u32 {
        match self {
            Self::Q4_0
            | Self::Q4_0Native
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_0Native
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_0Native
            | Self::Q8_1 => 32,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q4KNative
            | Self::Q5K
            | Self::Q5KNative
            | Self::Q6K
            | Self::Q6KNative
            | Self::Q8K => 256,
        }
    }

    /// Number of u32 words needed to cover one stored block.
    pub const fn block_words(self) -> u32 {
        self.block_bytes().div_ceil(4)
    }

    /// Number of bytes in one stored block.
    pub const fn block_bytes(self) -> u32 {
        match self {
            Self::Q4_0 => 20,
            Self::Q4_0Native => 18,
            Self::Q4_1 => 24,
            Self::Q5_0 => 24,
            Self::Q5_0Native => 22,
            Self::Q5_1 => 28,
            Self::Q8_0 => 36,
            Self::Q8_0Native => 34,
            Self::Q8_1 => 40,
            Self::Q2K => 88,
            Self::Q3K => 112,
            Self::Q4K => 148,
            Self::Q4KNative => 144,
            Self::Q5K => 180,
            Self::Q5KNative => 176,
            Self::Q6K => 212,
            Self::Q6KNative => 210,
            Self::Q8K => 292,
        }
    }

    pub const fn matrix_storage_words(self, rows: u32, cols: u32) -> u32 {
        let blocks = (rows / self.block_elements()) * cols;
        (blocks * self.block_bytes()).div_ceil(4)
    }

    pub const fn uses_byte_addressed_blocks(self) -> bool {
        matches!(
            self,
            Self::Q4_0Native | Self::Q5_0Native | Self::Q8_0Native | Self::Q6KNative
        )
    }

    pub const fn is_q4k_family(self) -> bool {
        matches!(self, Self::Q4K | Self::Q4KNative)
    }

    pub const fn is_q8_0_family(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q8_0Native)
    }

    pub const fn is_q5_0_family(self) -> bool {
        matches!(self, Self::Q5_0 | Self::Q5_0Native)
    }

    pub const fn is_q5k_family(self) -> bool {
        matches!(self, Self::Q5K | Self::Q5KNative)
    }

    pub const fn is_q6k_family(self) -> bool {
        matches!(self, Self::Q6K | Self::Q6KNative)
    }

    pub const fn has_native_f16_scales(self) -> bool {
        matches!(
            self,
            Self::Q4_0Native
                | Self::Q5_0Native
                | Self::Q8_0Native
                | Self::Q4KNative
                | Self::Q5KNative
                | Self::Q6KNative
        )
    }
}

/// A packed quantized storage matrix — kernel-input handle pairing a tile-IR
/// storage view with the quantization format and matrix dimensions.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
