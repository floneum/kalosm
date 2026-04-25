use crate::Layout;
use crate::{DataType, DataTypeEnum, LazyTensorData, Tensor, TensorData};

use super::QMatrix;

impl QMatrix {
    pub fn dequantize<const R: usize, T: DataType>(&self) -> Tensor<R, T>
    where
        f32: crate::CastTensor<T>,
    {
        assert_eq!(
            self.shape.len(),
            R,
            "Dequantize: expected {}D tensor, got {}D tensor. Shape: {:?}",
            R,
            self.shape.len(),
            self.shape
        );

        let dequantized =
            Tensor::<R, f32>::from_parts(LazyTensorData::new(TensorData::new_from_parts(
                &self.device,
                self.dequantized_f32_buffer(),
                Layout::contiguous(&self.shape),
                DataTypeEnum::F32,
            )));
        dequantized.cast()
    }
}
