use fusor::{
    AddOp, CastTensor, CastTo, Device, FloatDataType, FloatOps, MatmulImpl, MulOp, SimdBinaryOp,
    SimdElement, SimdReduceOp, SumOp, Tensor, VarBuilder,
};

#[derive(Debug, Clone, Copy)]
pub struct Conv3dConfig {
    pub padding: usize,
    pub stride: usize,
}

impl Default for Conv3dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
        }
    }
}

pub struct Conv3d<T: fusor::DataType + SimdElement> {
    weight: Tensor<5, T>, // (out_channels, in_channels, kernel_h, kernel_w, temporal)
    bias: Option<Tensor<1, T>>, // (out_channels,)
    config: Conv3dConfig,
}

impl<T: FloatDataType + SimdElement + FloatOps + MatmulImpl> Conv3d<T>
where
    MulOp: SimdBinaryOp<T>,
    AddOp: SimdBinaryOp<T>,
    SumOp: SimdReduceOp<T>,
{
    pub fn new(weight: Tensor<5, T>, bias: Option<Tensor<1, T>>, config: Conv3dConfig) -> Self {
        Self {
            weight,
            bias,
            config,
        }
    }

    pub fn forward(&self, input: &Tensor<5, T>) -> Tensor<5, T> {
        input.conv(
            &self.weight,
            self.bias.as_ref(),
            [
                self.config.padding,
                self.config.padding,
                self.config.padding,
            ],
            [self.config.stride, self.config.stride, self.config.stride],
        )
    }
}

pub(crate) struct Qwen2_5VisionPatchEmbed<F: FloatDataType + SimdElement> {
    patch_size: usize,
    temporal_patch_size: usize,
    in_channels: usize,
    embed_dim: usize,
    conv: Conv3d<F>,
}

impl<F: FloatDataType + SimdElement + FloatOps + MatmulImpl> Qwen2_5VisionPatchEmbed<F>
where
    F: CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
    MulOp: SimdBinaryOp<F>,
    AddOp: SimdBinaryOp<F>,
    SumOp: SimdReduceOp<F>,
{
    pub fn new(
        patch_size: usize,
        temporal_patch_size: usize,
        in_channels: usize,
        hidden_size: usize,
        vb: &mut VarBuilder,
        device: &Device,
    ) -> fusor::Result<Self> {
        // GGUF stores Conv3D weights split into two Conv2D tensors along the temporal dimension
        // - "weight" contains temporal slice 0: [out_channels, in_channels, kernel_h, kernel_w]
        // - "weight.1" contains temporal slice 1: [out_channels, in_channels, kernel_h, kernel_w]
        // We need to stack them to form [out_channels, in_channels, temporal, kernel_h, kernel_w]
        assert_eq!(
            temporal_patch_size, 2,
            "Only 2 temporal patch size is supported for 5D weights"
        );

        let weight_0 = vb.get("weight", device)?;
        let weight_1 = vb.get("weight.1", device)?;

        // Dequantize to 4D f32 tensors then cast to F
        let weight_0: Tensor<4, F> = weight_0.dequantize().cast();
        let weight_1: Tensor<4, F> = weight_1.dequantize().cast();

        // Stack along the temporal dimension (dim 2) to create 5D tensor
        // [out_channels, in_channels, kernel_h, kernel_w] -> [out_channels, in_channels, 2, kernel_h, kernel_w]
        let weight: Tensor<5, F> = Tensor::stack([weight_0, weight_1], 2);

        let cfg = Conv3dConfig {
            stride: patch_size,
            ..Default::default()
        };

        Ok(Self {
            patch_size,
            temporal_patch_size,
            in_channels,
            embed_dim: hidden_size,
            conv: Conv3d::new(weight, None, cfg),
        })
    }

    pub fn temporal_patch_size(&self) -> usize {
        self.temporal_patch_size
    }

    pub fn forward(&self, hidden_states: &Tensor<2, F>) -> fusor::Result<Tensor<2, F>> {
        let [num_patches, _] = hidden_states.shape();

        // Input: (num_patches, in_channels * temporal * patch * patch)
        // Reshape to (num_patches, in_channels, temporal, patch, patch)
        let x = hidden_states
            .reshape([
                num_patches,
                self.in_channels,
                self.temporal_patch_size,
                self.patch_size,
                self.patch_size,
            ])
            .to_concrete();

        let out = self.conv.forward(&x);
        Ok(out
            .reshape([
                out.shape().iter().product::<usize>() / self.embed_dim,
                self.embed_dim,
            ])
            .to_concrete())
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_vision_patch_embed() {
    use fusor::ToVec2;

    let embed_dim = 4;
    let in_channels = 3;
    let temporal_patch_size = 2;
    let patch_size = 2;

    let weights = [
        [
            [
                [[-2.5095e-02, 4.8800e-03], [7.8459e-03, 2.8647e-04]],
                [[6.4076e-03, 5.8325e-03], [1.0669e-02, -4.5015e-03]],
            ],
            [
                [[-1.8527e-03, 7.5276e-03], [4.0476e-03, 1.7847e-03]],
                [[2.6491e-03, 1.2732e-02], [-1.3109e-05, -3.0360e-03]],
            ],
            [
                [[-1.4570e-02, -1.0234e-03], [-5.9915e-03, 4.7706e-03]],
                [[7.2618e-03, 9.1152e-04], [-3.8907e-03, 5.2792e-03]],
            ],
        ],
        [
            [
                [[-1.2685e-04, 2.4084e-03], [1.3254e-03, 7.6424e-03]],
                [[1.0950e-02, 3.3989e-03], [7.1997e-03, 4.1141e-03]],
            ],
            [
                [[1.9312e-02, 1.0119e-02], [-1.4364e-02, -1.1299e-02]],
                [[-1.3603e-03, 1.6354e-02], [6.5474e-03, 5.7600e-03]],
            ],
            [
                [[1.1415e-02, 1.8565e-04], [-1.8058e-02, 9.2543e-03]],
                [[-3.7534e-03, 1.0331e-02], [-6.8665e-03, 6.3681e-03]],
            ],
        ],
        [
            [
                [[-9.7267e-03, 9.5846e-03], [1.6192e-02, 1.4506e-02]],
                [[2.6948e-03, -2.1038e-03], [-7.3280e-03, 1.0430e-03]],
            ],
            [
                [[3.4875e-03, 9.6759e-03], [-4.6569e-03, 1.6048e-02]],
                [[-2.4801e-02, -4.1754e-03], [-1.1955e-02, 8.1234e-03]],
            ],
            [
                [[-1.9006e-02, 2.2858e-03], [2.4859e-04, -3.4595e-03]],
                [[2.8683e-03, -7.3084e-03], [1.7482e-03, -1.0939e-02]],
            ],
        ],
        [
            [
                [[-1.6022e-02, 1.3529e-02], [1.2888e-02, 5.2295e-04]],
                [[-1.5469e-02, 7.5671e-03], [7.7552e-03, 2.0265e-02]],
            ],
            [
                [[3.5818e-04, 1.2059e-03], [-8.0566e-03, -2.0758e-03]],
                [[-9.3195e-03, -1.5910e-02], [-1.1360e-02, -5.2260e-03]],
            ],
            [
                [[-5.1877e-03, -1.5013e-02], [-1.9267e-02, 1.2785e-03]],
                [[1.0229e-02, -5.5580e-03], [7.0427e-03, 7.0988e-03]],
            ],
        ],
    ];

    let device = Device::new().await.unwrap();

    let weight = Tensor::new(&device, &weights);

    let patch_embed = Qwen2_5VisionPatchEmbed {
        patch_size,
        temporal_patch_size,
        in_channels,
        embed_dim,
        conv: Conv3d::new(
            weight,
            None,
            Conv3dConfig {
                stride: patch_size,
                ..Default::default()
            },
        ),
    };

    let input = [[
        [[[0., 1.], [2., 3.]], [[4., 5.], [6., 7.]]],
        [[[8., 9.], [10., 11.]], [[12., 13.], [14., 15.]]],
        [[[16., 17.], [18., 19.]], [[20., 21.], [22., 23.]]],
    ]];
    let input = Tensor::new(&device, &input);

    let output = patch_embed
        .forward(&input.reshape((1, ())).to_concrete())
        .unwrap()
        .cast::<f32>();
    let output_slice = output.as_slice().await.unwrap();
    let output_vec = output_slice.to_vec2();
    println!("Output: {output_vec:?}");
    let expected_output = [[0.3058, 0.6866, -0.7391, -0.6952]];
    assert_2d_vec_eq(output_vec, expected_output, 1e-2);
}

#[cfg(test)]
pub(crate) fn assert_2d_vec_eq(
    a: impl IntoIterator<Item = impl IntoIterator<Item = f32>>,
    b: impl IntoIterator<Item = impl IntoIterator<Item = f32>>,
    tolerance: f32,
) {
    let a = a
        .into_iter()
        .map(|x| x.into_iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let b = b
        .into_iter()
        .map(|x| x.into_iter().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    assert_eq!(a.len(), b.len());
    for (a_row, b_row) in a.iter().zip(b.iter()) {
        assert_eq!(a_row.len(), b_row.len());
        for (a_elem, b_elem) in a_row.iter().zip(b_row.iter()) {
            assert!(
                (a_elem - b_elem).abs() < tolerance,
                "Elements are not equal: {a_elem} vs {b_elem}"
            );
        }
    }
}
