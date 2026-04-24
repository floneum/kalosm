use fusor_gguf::GgmlType;
use std::fmt::Display;
use std::fmt::Write;

#[derive(Clone, Debug)]
pub(crate) struct QMatrixInput {
    pub(crate) matrix_binding: u32,
    pub(crate) info_binding: u32,
    pub(crate) datatype: GgmlType,
    pub(crate) rank: u32,
}

impl QMatrixInput {
    pub(crate) fn get_matrix_binding(&self) -> u32 {
        self.matrix_binding
    }

    pub(crate) fn get_info_binding(&self) -> u32 {
        self.info_binding
    }

    fn info_binding(&self) -> String {
        format!("i_{}", self.get_info_binding())
    }

    pub(crate) fn strided_index(
        &self,
        write: &mut impl std::fmt::Write,
        indexes: impl IntoIterator<Item = String>,
    ) {
        if self.rank == 0 {
            write!(write, "0u").unwrap();
            return;
        }
        let mut strides = Vec::new();
        let mut product = "1".to_string();
        for i in (0..self.rank).rev() {
            let mut shape = self.shape_binding(i);
            if i == self.rank - 1 {
                write!(&mut shape, " / {}", self.datatype.block_size()).unwrap();
            }
            let new = format!("{product} * {shape}");
            strides.push(product);
            product = new;
        }
        for (i, index) in indexes.into_iter().enumerate().take(self.rank as usize) {
            let stride = &strides[strides.len() - i - 1];
            write!(write, "({index})*{stride}").unwrap();
            if i < (self.rank - 1) as usize {
                write!(write, " + ").unwrap();
            }
        }
    }

    pub(crate) fn shape_binding(&self, rank: u32) -> String {
        format!("{}.shape_{}", self.info_binding(), rank)
    }
}

impl Display for QMatrixInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "i_{}", self.get_matrix_binding())
    }
}
