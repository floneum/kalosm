use crate::{
    Tensor,
    nary_wise::{NaryExpr, NaryOperation},
};

/// Compute the output shape for an index_select operation.
pub(crate) fn index_select_output_shape(
    dimension: usize,
    value_shape: &[usize],
    indexes_shape: &[usize],
) -> Box<[usize]> {
    value_shape
        .iter()
        .enumerate()
        .map(|(i, dim)| {
            if i == dimension {
                indexes_shape[0]
            } else {
                *dim
            }
        })
        .collect()
}

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn index_select(&self, dimension: usize, indexes: &Tensor<1, u32>) -> Self {
        assert!(dimension < R);
        let output_shape = index_select_output_shape(dimension, self.shape(), indexes.shape());
        let nary = NaryOperation {
            inputs: vec![self.key(), indexes.key()],
            expression: NaryExpr::index_select(R, dimension),
            shape: output_shape.clone(),
            output_datatype: T::WGSL_TYPE,
        };
        let device = self.device().clone();
        let info = crate::tensor::TensorInfo::new(output_shape, T::WGSL_TYPE);
        let key = device.compute_graph().create_nary(nary);
        Self::from_parts(crate::tensor::LazyTensorData::from_parts(device, info, key))
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_dim_0() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &[1, 0]);
    let tensor = tensor.index_select(0, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    let expected_data = [[4., 5., 6.], [1., 2., 3.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_large_dim_0() {
    use rand::seq::SliceRandom;

    use crate::Device;

    let device = Device::test_instance();

    const SIZE_1: usize = 100;
    const SIZE_0: usize = 100;
    let mut indexes_array: [u32; SIZE_0] = std::array::from_fn(|i| i as u32);
    indexes_array.shuffle(&mut rand::rng());
    let data: [[f32; SIZE_1]; SIZE_0] =
        std::array::from_fn(|i| std::array::from_fn(|j| (i * SIZE_1 + j) as f32));
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &indexes_array);
    let tensor = tensor.index_select(0, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    let expected_data: [[f32; SIZE_1]; SIZE_0] = std::array::from_fn(|i| {
        let index = indexes_array[i];
        data[index as usize]
    });
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_dim_1() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &[1, 2, 0]);
    let tensor = tensor.index_select(1, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    let expected_data = [[2., 3., 1.], [5., 6., 4.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_large_dim_1() {
    use rand::seq::SliceRandom;

    use crate::Device;

    let device = Device::test_instance();

    const SIZE_1: usize = 100;
    const SIZE_0: usize = 100;
    let mut indexes_array: [u32; SIZE_1] = std::array::from_fn(|i| i as u32);
    indexes_array.shuffle(&mut rand::rng());
    let data: [[f32; SIZE_1]; SIZE_0] =
        std::array::from_fn(|i| std::array::from_fn(|j| (i * SIZE_1 + j) as f32));
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &indexes_array);
    let tensor = tensor.index_select(1, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    let expected_data: [[f32; SIZE_1]; SIZE_0] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let index = indexes_array[j];
            data[i][index as usize]
        })
    });
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_multiply_before_index_select() {
    use crate::Device;

    let device = Device::test_instance();

    // Test that multiply works correctly
    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor * 3.;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("multiply result: {as_slice:?}");
    let expected_data = [[3., 6., 9.], [12., 15., 18.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_prefused() {
    use crate::Device;

    let device = Device::test_instance();

    // Test just tensor * 3. -> index_select (pre-fusion only)
    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &[1, 0]);
    let tensor = (tensor * 3.).index_select(1, &indexes);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("prefused: {as_slice:?}");
    // tensor * 3 = [[3, 6, 9], [12, 15, 18]]
    // index_select(1, [1, 0]) -> [[6, 3], [15, 12]]
    let expected_data = [[6., 3.], [15., 12.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_index_select_fused() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2., 3.], [4., 5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let indexes = Tensor::new(&device, &[1, 0]);
    let tensor = (tensor * 3.).index_select(1, &(indexes * 2u32)) * 2.0;
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    let expected_data = [[3. * 3. * 2., 1. * 3. * 2.], [6. * 3. * 2., 4. * 3. * 2.]];
    let expected_tensor = Tensor::new(&device, &expected_data);
    let expected_as_slice = expected_tensor.as_slice().await.unwrap();
    assert_eq!(as_slice, expected_as_slice);
}
