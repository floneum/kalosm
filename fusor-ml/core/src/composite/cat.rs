use crate::{DataType, Dim, NextRank, Tensor};

impl<const R: usize, D: DataType> Tensor<R, D> {
    pub fn cat(vectors: impl IntoIterator<Item = Self>, dim: impl Dim<R>) -> Self {
        let dim = dim.resolve();
        let vectors = vectors.into_iter().collect::<Vec<_>>();
        assert!(!vectors.is_empty(), "cat requires at least one tensor");
        let mut shape = [0; R];
        for (i, v) in vectors[0].shape().iter().enumerate() {
            if i != dim {
                shape[i] = *v;
            }
        }
        for vector in &vectors {
            let vector_shape = vector.shape();
            for (i, shape) in shape.iter_mut().enumerate() {
                if i == dim {
                    *shape += vector_shape[i];
                } else {
                    assert_eq!(*shape, vector_shape[i]);
                }
            }
        }
        let device = vectors[0].device().clone();
        let mut index = 0;
        let mut larger = Tensor::zeros(&device, shape);
        for vector in vectors {
            let length = vector.shape()[dim];
            if vector.is_zero_splat() {
                index += length;
                continue;
            }
            let slice = std::array::from_fn(|i| {
                if i == dim {
                    index..(index + length)
                } else {
                    0..shape[i]
                }
            });
            larger = larger.slice_assign(slice, &vector);
            index += length;
        }
        larger
    }
}

impl<const R1: usize, D: DataType> Tensor<R1, D> {
    pub fn stack<const R2: usize>(
        vectors: impl IntoIterator<Item = Self>,
        dim: impl Dim<R2>,
    ) -> Tensor<R2, D>
    where
        Self: NextRank<R2, D>,
    {
        let dim = dim.resolve();
        Tensor::cat(vectors.into_iter().map(|t| t.unsqueeze(dim)), dim)
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_cat() {
    use crate::Device;

    let device = Device::test_instance();

    let data1 = [[1., -2.], [-3., 4.], [5., -6.]];
    let tensor1 = Tensor::new(&device, &data1);
    let data2 = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor2 = Tensor::new(&device, &data2);

    let tensor = Tensor::cat([tensor1, tensor2], 1);
    assert_eq!(*tensor.shape(), [3, 4]);

    let output = tensor.as_slice().await.unwrap();
    println!("{output:?}");
    assert_eq!(output[[0, 0]], 1.);
    assert_eq!(output[[0, 1]], -2.);
    assert_eq!(output[[0, 2]], 1.);
    assert_eq!(output[[0, 3]], 2.);

    assert_eq!(output[[1, 0]], -3.);
    assert_eq!(output[[1, 1]], 4.);
    assert_eq!(output[[1, 2]], 3.);
    assert_eq!(output[[1, 3]], 4.);

    assert_eq!(output[[2, 0]], 5.);
    assert_eq!(output[[2, 1]], -6.);
    assert_eq!(output[[2, 2]], 5.);
    assert_eq!(output[[2, 3]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_multi_dim_cat() {
    use crate::{D, Device};

    let device = Device::test_instance();

    let data1 = vec![vec![vec![1f32; 32]; 11]; 3];
    let tensor1 = Tensor::new(&device, &data1).reshape([1, 3, 11, 32, 1]);
    let data2 = vec![vec![vec![2f32; 32]; 11]; 3];
    let tensor2 = Tensor::new(&device, &data2).reshape([1, 3, 11, 32, 1]);

    let tensor = Tensor::cat([tensor1, tensor2], D::Minus1);
    println!("tensor shape: {:?}", tensor.shape());

    assert_eq!(*tensor.shape(), [1, 3, 11, 32, 2]);

    let output = tensor.i((0usize, .., .., .., ..)).as_slice().await.unwrap();
    println!("{output:?}");

    for i in 0..3 {
        for j in 0..11 {
            for k in 0..32 {
                for l in 0..2 {
                    let value = output[[i, j, k, l]];
                    let expected = if l == 0 { 1f32 } else { 2f32 };
                    assert_eq!(value, expected);
                }
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_rel_shift_cat_pattern_matches_cpu() {
    use crate::Device;

    let gpu = Device::test_instance();

    let data = vec![vec![vec![vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![6.0, 7.0, 8.0, 9.0, 10.0]]]];
    let gpu_x = Tensor::new(&gpu, &data);

    let run = |x: Tensor<4, f32>| -> Tensor<4, f32> {
        let [batch, heads, q_len, pos_len] = *x.shape();
        let zeros = Tensor::zeros(&x.device(), [batch, heads, q_len, 1]);
        let padded = Tensor::cat([zeros, x], 3);
        let reshaped = padded.reshape([batch, heads, pos_len + 1, q_len]);
        reshaped
            .narrow(2, 1, pos_len)
            .reshape([batch, heads, q_len, pos_len])
    };

    let gpu_out = run(gpu_x).as_slice().await.unwrap();
    let expected = [
        [[[2.0f32, 3.0, 4.0, 5.0, 0.0], [6.0, 7.0, 8.0, 9.0, 10.0]]]
    ];

    for b in 0..1 {
        for h in 0..1 {
            for q in 0..2 {
                for p in 0..5 {
                    assert_eq!(gpu_out[[b, h, q, p]], expected[b][h][q][p]);
                }
            }
        }
    }
}
