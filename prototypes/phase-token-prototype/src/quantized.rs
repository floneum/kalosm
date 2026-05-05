use crate::ir::StorageView;

/// GGML quantization formats represented by the prototype qmatmul path.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GgmlQuantFormat {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlQuantFormat {
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

    pub const fn qgemv_cols_per_workgroup(self) -> u32 {
        self.qgemv_subgroups_per_workgroup() * self.qgemv_cols_per_subgroup()
    }

    pub const fn qgemv_cols_per_workgroup_for_shape(self, rows: u32, cols: u32) -> u32 {
        self.qgemv_subgroups_per_workgroup_for_shape(rows, cols) * self.qgemv_cols_per_subgroup()
    }

    pub const fn qgemv_cols_per_subgroup(self) -> u32 {
        match self {
            Self::Q2K => 4,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_1 => 4,
            Self::Q5_0 => 4,
            Self::Q3K | Self::Q8K => 2,
            Self::Q4K => 8,
            Self::Q6K => 4,
            Self::Q8_0 | Self::Q8_1 => 4,
            Self::Q5K => 1,
        }
    }

    pub const fn qgemv_subgroups_per_workgroup(self) -> u32 {
        match self {
            Self::Q4K | Self::Q6K | Self::Q8_0 | Self::Q8_1 => 4,
            _ => 2,
        }
    }

    pub const fn qgemv_subgroups_per_workgroup_for_shape(self, rows: u32, _cols: u32) -> u32 {
        match self {
            Self::Q6K if rows > 4096 => 8,
            _ => self.qgemv_subgroups_per_workgroup(),
        }
    }
}

/// A packed quantized storage matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedMatrix {
    pub data: StorageView,
    pub format: GgmlQuantFormat,
    pub rows: u32,
    pub cols: u32,
}
