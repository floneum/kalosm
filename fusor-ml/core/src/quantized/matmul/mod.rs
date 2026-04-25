use crate::{DataType, DataTypeEnum, Layout, LazyTensorData, Tensor, TensorData};

use super::QMatrix;

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        assert_eq!(
            T::WGSL_TYPE,
            DataTypeEnum::F32,
            "q_mat_mul currently materializes dequantized weights as f32"
        );
        assert!(R >= 2, "q_mat_mul requires rank >= 2");
        let n = other.shape()[0];
        let k = other.shape()[1];
        assert_eq!(self.shape()[R - 1], k);

        let base_weight_shape: [usize; R] = std::array::from_fn(|axis| {
            if axis < R - 2 {
                self.shape()[axis]
            } else if axis == R - 2 {
                k
            } else {
                n
            }
        });
        let weight_strides = (0..R)
            .map(|axis| {
                if axis < R - 2 {
                    0
                } else if axis == R - 2 {
                    1
                } else {
                    k
                }
            })
            .collect::<Box<[_]>>();
        let weights = Tensor::from_parts(LazyTensorData::new(TensorData::new_from_parts(
            other.device(),
            other.dequantized_f32_buffer(),
            Layout::from_parts(0, base_weight_shape.into(), weight_strides),
            DataTypeEnum::F32,
        )));
        self.mat_mul(&weights)
    }
}
