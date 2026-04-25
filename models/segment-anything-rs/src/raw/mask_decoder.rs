//! Mask decoder: predicts masks from image+prompt embeddings.

use fusor::layers::{Embedding, LayerNormNd, Linear};
use fusor::{ConcreteTensor, Device, Tensor, VarBuilder};

use super::transformer::TwoWayTransformer;
use super::Result;

/// Mask-decoder transformer config, matching Meta's official SAM checkpoint.
/// Promoted to named constants so SAM2 / future ports don't have to chase
/// magic literals through `MaskDecoder::load`.
const TRANSFORMER_DEPTH: usize = 2;
const TRANSFORMER_NUM_HEADS: usize = 8;
const TRANSFORMER_MLP_DIM: usize = 2048;
/// Hyper-network MLPs that turn each mask token into a per-pixel kernel.
const HYPER_MLP_LAYERS: usize = 3;
/// Expected upscaling kernel shape (2x2 stride-2 transposed conv).
const UPSCALE_KERNEL_HW: [usize; 2] = [2, 2];

/// Private 2x2 stride-2 learned upsampler used by the SAM output head.
///
/// This is intentionally local to the SAM port rather than exposed as a generic
/// `fusor` layer. The implementation relies on the exact kernel/stride pattern
/// used by SAM and reorders the result into image space with a pixel-shuffle.
struct SamUpscale2x2 {
    weight: Tensor<4, f32, ConcreteTensor<f32, 4>>,
    bias: Option<Tensor<1, f32, ConcreteTensor<f32, 1>>>,
}

impl SamUpscale2x2 {
    fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let weight: Tensor<4, f32> = vb.get("weight", device)?.dequantize();
        let bias: Option<Tensor<1, f32, ConcreteTensor<f32, 1>>> =
            vb.get("bias", device).ok().map(|b| b.dequantize());
        let weight_shape = weight.shape();
        let kh = weight_shape[2];
        let kw = weight_shape[3];
        if [kh, kw] != UPSCALE_KERNEL_HW {
            return Err(fusor::Error::msg(format!(
                "SAM upscaling expects a {:?} transposed-conv kernel, got {:?}",
                UPSCALE_KERNEL_HW,
                [kh, kw]
            )));
        }
        Ok(Self {
            weight: weight.to_concrete(),
            bias,
        })
    }

    fn forward(
        &self,
        input: &Tensor<4, f32, ConcreteTensor<f32, 4>>,
    ) -> Tensor<4, f32, ConcreteTensor<f32, 4>> {
        let shape = input.shape();
        let b = shape[0];
        let in_ch = shape[1];
        let h = shape[2];
        let w = shape[3];

        let weight_shape = self.weight.shape();
        let out_ch = weight_shape[1];
        let kh = weight_shape[2];
        let kw = weight_shape[3];

        let input_flat = input.reshape([b, in_ch, h * w]);
        let input_flat = input_flat.transpose(1, 2);
        let input_flat = input_flat.reshape([b * h * w, in_ch]);
        let weight_flat = self.weight.reshape([in_ch, out_ch * kh * kw]);
        let result = input_flat.mat_mul(&weight_flat);

        let result = result.reshape([b, h, w, out_ch, kh, kw]);
        let result = result.transpose(2, 3);
        let result = result.transpose(1, 2);
        let result = result.transpose(3, 4);
        let result = result.reshape([b, out_ch, h * kh, w * kw]);

        if let Some(bias) = &self.bias {
            let bias = bias.reshape([1, out_ch, 1, 1]);
            let bias_4d = bias.broadcast_as([b, out_ch, h * kh, w * kw]);
            (result + bias_4d).to_concrete()
        } else {
            result.to_concrete()
        }
    }
}

struct MlpMaskDecoder {
    layers: Vec<Linear<f32>>,
    sigmoid_output: bool,
}

impl MlpMaskDecoder {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        num_layers: usize,
        sigmoid_output: bool,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = Linear::load(device, &mut vb.pp(format!("layers.{i}")))?;
            layers.push(layer);
        }
        Ok(Self {
            layers,
            sigmoid_output,
        })
    }

    fn forward(&self, xs: &Tensor<2, f32>) -> Tensor<2, f32> {
        let mut xs = xs.to_concrete();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_2d(&xs);
            if i + 1 < self.layers.len() {
                xs = xs.relu();
            }
        }
        if self.sigmoid_output {
            xs.sigmoid()
        } else {
            xs
        }
    }
}

/// SAM mask decoder head.
///
/// `forward(image_embeddings, image_pe, sparse_prompt, dense_prompt, multimask)`
/// returns `(masks, iou_predictions)`:
/// - `masks`: `(batch, n_masks, low_res_h, low_res_w)`. `n_masks` = 3 if
///   `multimask=true`, else 1. The masks are at 1/4 resolution of `IMAGE_SIZE`.
/// - `iou_predictions`: `(batch, n_masks)` quality scores in `[0, 1]`.
pub struct MaskDecoder {
    iou_token: Embedding<f32>,
    mask_tokens: Embedding<f32>,
    iou_prediction_head: MlpMaskDecoder,
    output_upscaling_conv1: SamUpscale2x2,
    output_upscaling_ln: LayerNormNd<f32>,
    output_upscaling_conv2: SamUpscale2x2,
    num_mask_tokens: usize,
    output_hypernetworks_mlps: Vec<MlpMaskDecoder>,
    transformer: TwoWayTransformer,
}

impl MaskDecoder {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        transformer_dim: usize,
        num_multimask_outputs: usize,
        iou_head_depth: usize,
    ) -> Result<Self> {
        let num_mask_tokens = num_multimask_outputs + 1;
        let iou_prediction_head = MlpMaskDecoder::load(
            device,
            &mut vb.pp("iou_prediction_head"),
            iou_head_depth,
            false,
        )?;
        let iou_token = Embedding::load(device, &mut vb.pp("iou_token"))?;
        let mask_tokens = Embedding::load(device, &mut vb.pp("mask_tokens"))?;
        let output_upscaling_conv1 = SamUpscale2x2::load(device, &mut vb.pp("output_upscaling.0"))?;
        let output_upscaling_ln =
            LayerNormNd::<f32>::load_over_axis(device, &mut vb.pp("output_upscaling.1"), 1, 1e-6)?;
        let output_upscaling_conv2 = SamUpscale2x2::load(device, &mut vb.pp("output_upscaling.3"))?;
        let mut output_hypernetworks_mlps = Vec::with_capacity(num_mask_tokens);
        for i in 0..num_mask_tokens {
            let mlp = MlpMaskDecoder::load(
                device,
                &mut vb.pp(format!("output_hypernetworks_mlps.{i}")),
                HYPER_MLP_LAYERS,
                false,
            )?;
            output_hypernetworks_mlps.push(mlp);
        }
        let transformer = TwoWayTransformer::load(
            device,
            &mut vb.pp("transformer"),
            TRANSFORMER_DEPTH,
            transformer_dim,
            TRANSFORMER_NUM_HEADS,
            TRANSFORMER_MLP_DIM,
        )?;
        Ok(Self {
            iou_token,
            mask_tokens,
            iou_prediction_head,
            output_upscaling_conv1,
            output_upscaling_ln,
            output_upscaling_conv2,
            num_mask_tokens,
            output_hypernetworks_mlps,
            transformer,
        })
    }

    pub fn forward(
        &self,
        image_embeddings: &Tensor<4, f32>,
        image_pe: &Tensor<4, f32>,
        sparse_prompt_embeddings: &Tensor<3, f32>,
        dense_prompt_embeddings: &Tensor<4, f32>,
        multimask_output: bool,
    ) -> (Tensor<4, f32>, Tensor<2, f32>) {
        let (masks, iou_pred) = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        );
        if multimask_output {
            // masks[:, 1:], iou_pred[:, 1:]
            let masks_shape = masks.shape();
            let masks = masks.narrow(1, 1, masks_shape[1] - 1).to_concrete();
            let iou_shape = iou_pred.shape();
            let iou_pred = iou_pred.narrow(1, 1, iou_shape[1] - 1).to_concrete();
            (masks, iou_pred)
        } else {
            // masks[:, 0:1], iou_pred[:, 0:1]
            let masks = masks.narrow(1, 0, 1).to_concrete();
            let iou_pred = iou_pred.narrow(1, 0, 1).to_concrete();
            (masks, iou_pred)
        }
    }

    fn predict_masks(
        &self,
        image_embeddings: &Tensor<4, f32>,
        image_pe: &Tensor<4, f32>,
        sparse_prompt_embeddings: &Tensor<3, f32>,
        dense_prompt_embeddings: &Tensor<4, f32>,
    ) -> (Tensor<4, f32>, Tensor<2, f32>) {
        // Concatenate output tokens: [iou_token, mask_tokens]
        let iou_emb = self.iou_token.embeddings(); // (1, dim)
        let mask_emb = self.mask_tokens.embeddings(); // (num_mask_tokens, dim)
        let output_tokens: Tensor<2, f32> = Tensor::cat([iou_emb.clone(), mask_emb.clone()], 0); // (1+num_mask_tokens, dim)

        let sparse_shape = sparse_prompt_embeddings.shape();
        let batch_size = sparse_shape[0];
        let token_shape = output_tokens.shape();
        let num_tokens = token_shape[0];
        let dim = token_shape[1];

        // Expand to batch: (batch, num_tokens, dim)
        let output_tokens: Tensor<3, f32> = output_tokens
            .reshape([1, num_tokens, dim])
            .broadcast_as([batch_size, num_tokens, dim])
            .to_concrete();

        // Cat with sparse prompt embeddings: (batch, num_tokens + num_sparse, dim)
        let tokens: Tensor<3, f32> =
            Tensor::cat([output_tokens, sparse_prompt_embeddings.to_concrete()], 1);

        // Expand image data per mask
        let img_shape = image_embeddings.shape();
        let c = img_shape[1];
        let h = img_shape[2];
        let w = img_shape[3];

        let src = repeat_interleave_4d(image_embeddings, batch_size);
        let src: Tensor<4, f32> = (src + dense_prompt_embeddings).to_concrete();
        let pos_src = repeat_interleave_4d(image_pe, batch_size);

        // Run the transformer
        let (hs, src) = self.transformer.forward(&src, &pos_src, &tokens);

        // Extract token outputs
        let iou_token_out: Tensor<2, f32> = hs
            .narrow(1, 0, 1)
            .to_concrete()
            .reshape([batch_size, dim])
            .to_concrete();
        let mask_tokens_out: Tensor<3, f32> = hs.narrow(1, 1, self.num_mask_tokens).to_concrete();

        // Upscale mask embeddings for the whole prompt batch at once.
        let src: Tensor<4, f32> = src
            .transpose(1, 2)
            .to_concrete()
            .reshape([batch_size, c, h, w])
            .to_concrete();
        let upscaled = self.output_upscaling_conv1.forward(&src);
        let upscaled = self.output_upscaling_ln.forward(&upscaled);
        let upscaled = upscaled.gelu().to_concrete();
        let upscaled = self.output_upscaling_conv2.forward(&upscaled);
        let upscaled: Tensor<4, f32> = upscaled.gelu().to_concrete();

        // Predict masks using hypernetwork MLPs
        let mut hyper_in_list = Vec::with_capacity(self.num_mask_tokens);
        for (i, mlp) in self.output_hypernetworks_mlps.iter().enumerate() {
            let token_i: Tensor<2, f32> = mask_tokens_out
                .narrow(1, i, 1)
                .to_concrete()
                .reshape([batch_size, dim])
                .to_concrete();
            let h = mlp.forward(&token_i);
            hyper_in_list.push(h);
        }
        // Stack into (batch, num_mask_tokens, dim/8)
        let hyper_in: Tensor<3, f32> = Tensor::stack(hyper_in_list, 1).to_concrete();

        let up_shape = upscaled.shape();
        let up_c = up_shape[1];
        let up_h = up_shape[2];
        let up_w = up_shape[3];

        // masks = hyper_in @ upscaled.reshape(b, c, h*w)
        let upscaled_flat: Tensor<3, f32> = upscaled
            .to_concrete()
            .reshape([batch_size, up_c, up_h * up_w])
            .to_concrete();
        let masks = hyper_in.mat_mul(&upscaled_flat);
        let masks_shape = masks.shape();
        let num_masks = masks_shape[1];
        let masks: Tensor<4, f32> = masks
            .reshape([batch_size, num_masks, up_h, up_w])
            .to_concrete();

        // Generate mask quality predictions
        let iou_pred = self.iou_prediction_head.forward(&iou_token_out);

        (masks, iou_pred)
    }
}

/// Equivalent to torch.repeat_interleave for 4D tensors along dim 0.
fn repeat_interleave_4d(
    img: &Tensor<4, f32>,
    repeats: usize,
) -> Tensor<4, f32, ConcreteTensor<f32, 4>> {
    let shape = img.shape();
    let b = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    // unsqueeze(1) -> (b, 1, c, h, w), broadcast to (b, repeats, c, h, w), flatten(0,1)
    img.reshape([b, 1, c, h, w])
        .broadcast_as([b, repeats, c, h, w])
        .to_concrete()
        .reshape([b * repeats, c, h, w])
        .to_concrete()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn to_vec<const R: usize>(tensor: &Tensor<R, f32, ConcreteTensor<f32, R>>) -> Vec<f32> {
        let len = tensor.shape().iter().product::<usize>();
        tensor
            .reshape([len])
            .to_concrete()
            .as_slice()
            .await
            .unwrap()
            .as_slice()
            .to_vec()
    }

    #[tokio::test]
    async fn test_upscaling_batched_matches_per_item() {
        let device = Device::new().await.unwrap();

        const BATCH: usize = 4;
        const IN_CH: usize = 8;
        const MID_CH: usize = 4;
        const OUT_CH: usize = 2;
        const H: usize = 3;
        const W: usize = 5;

        let input_data: Vec<f32> = (0..BATCH * IN_CH * H * W)
            .map(|i| (i as f32 * 0.03125).sin())
            .collect();
        let input: Tensor<4, f32> = Tensor::from_slice(&device, [BATCH, IN_CH, H, W], &input_data);

        let conv1_weight_data: Vec<f32> = (0..IN_CH * MID_CH * 2 * 2)
            .map(|i| (i as f32 * 0.0625).cos() * 0.25)
            .collect();
        let conv1_bias_data: Vec<f32> = (0..MID_CH).map(|i| i as f32 * 0.1 - 0.15).collect();
        let conv1 = SamUpscale2x2 {
            weight: Tensor::from_slice(&device, [IN_CH, MID_CH, 2, 2], &conv1_weight_data)
                .to_concrete(),
            bias: Some(Tensor::from_slice(&device, [MID_CH], &conv1_bias_data).to_concrete()),
        };

        let ln_weight_data: Vec<f32> = (0..MID_CH).map(|i| 0.75 + i as f32 * 0.1).collect();
        let ln_bias_data: Vec<f32> = (0..MID_CH).map(|i| -0.2 + i as f32 * 0.05).collect();
        let ln = LayerNormNd::new_over_axis(
            Tensor::from_slice(&device, [MID_CH], &ln_weight_data).to_concrete(),
            Some(Tensor::from_slice(&device, [MID_CH], &ln_bias_data).to_concrete()),
            1,
            1e-6,
        );

        let conv2_weight_data: Vec<f32> = (0..MID_CH * OUT_CH * 2 * 2)
            .map(|i| (i as f32 * 0.046875).sin() * 0.2)
            .collect();
        let conv2_bias_data: Vec<f32> = (0..OUT_CH).map(|i| i as f32 * 0.08 - 0.04).collect();
        let conv2 = SamUpscale2x2 {
            weight: Tensor::from_slice(&device, [MID_CH, OUT_CH, 2, 2], &conv2_weight_data)
                .to_concrete(),
            bias: Some(Tensor::from_slice(&device, [OUT_CH], &conv2_bias_data).to_concrete()),
        };

        let batched = conv1.forward(&input.to_concrete());
        let batched = ln.forward(&batched);
        let batched = batched.gelu().to_concrete();
        let batched = conv2.forward(&batched);
        let batched: Tensor<4, f32> = batched.gelu().to_concrete();

        let mut per_item = Vec::with_capacity(BATCH);
        for i in 0..BATCH {
            let item: Tensor<4, f32> = input.narrow(0, i, 1).to_concrete();
            let item = conv1.forward(&item);
            let item = ln.forward(&item);
            let item = item.gelu().to_concrete();
            let item = conv2.forward(&item);
            per_item.push(item.gelu().to_concrete());
        }
        let per_item: Tensor<4, f32> = Tensor::cat(per_item, 0).to_concrete();

        let batched_vec = to_vec(&batched.to_concrete()).await;
        let per_item_vec = to_vec(&per_item.to_concrete()).await;
        let max_diff = batched_vec
            .iter()
            .zip(per_item_vec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 0.001,
            "batched upscaling diverged from per-item path: {max_diff}",
        );
    }
}
