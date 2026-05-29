use crate::ir::StorageView;

/// GGML quantization formats represented by the tiled qmatmul path.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum GgmlQuantFormat {
    /// GGML `Q4_0`.
    Q4_0,
    /// GGML `Q4_0` in the native/padded GPU layout with an fp16 scale.
    Q4_0Native,
    /// GGML `Q4_1`.
    Q4_1,
    /// GGML `Q5_0`.
    Q5_0,
    /// GGML `Q5_0` in the native/padded GPU layout with an fp16 scale.
    Q5_0Native,
    /// GGML `Q5_1`.
    Q5_1,
    /// GGML `Q8_0`.
    Q8_0,
    /// GGML `Q8_0` in the native/padded GPU layout with an fp16 scale.
    Q8_0Native,
    /// GGML `Q8_1`.
    Q8_1,
    /// GGML K-quant `Q2_K`.
    Q2K,
    /// GGML K-quant `Q3_K`.
    Q3K,
    /// GGML K-quant `Q4_K`.
    Q4K,
    /// GGML K-quant `Q4_K` in the native/raw 36-word block layout.
    Q4KNative,
    /// GGML K-quant `Q5_K`.
    Q5K,
    /// GGML K-quant `Q5_K` in the native/raw 44-word block layout.
    Q5KNative,
    /// GGML K-quant `Q6_K`.
    Q6K,
    /// GGML K-quant `Q6_K` in the native/padded 53-word block layout.
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

    /// Number of u32 words in the f32-scale shader layout for one block.
    pub const fn block_words(self) -> u32 {
        match self {
            Self::Q4_0 | Self::Q4_0Native => 5,
            Self::Q4_1 => 6,
            Self::Q5_0 | Self::Q5_0Native => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 | Self::Q8_0Native => 9,
            Self::Q8_1 => 10,
            Self::Q2K => 22,
            Self::Q3K => 28,
            Self::Q4K => 37,
            Self::Q4KNative => 36,
            Self::Q5K => 45,
            Self::Q5KNative => 44,
            Self::Q6K | Self::Q6KNative => 53,
            Self::Q8K => 73,
        }
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
