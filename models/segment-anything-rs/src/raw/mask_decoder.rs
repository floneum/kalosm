//! Mask decoder: predicts masks from image+prompt embeddings.

use fusor2::layers::{Embedding, LayerNorm, Linear};
use fusor2::{stack, Device, Error, Tensor};
use fusor2_gguf::VarBuilder;

use super::transformer::TwoWayTransformer;
use super::{channel_layer_norm, linear, load_dense, Result};

/// Mask-decoder transformer config, matching Meta's official SAM checkpoint.
const TRANSFORMER_DEPTH: usize = 2;
const TRANSFORMER_NUM_HEADS: usize = 8;
const TRANSFORMER_MLP_DIM: usize = 2048;
/// Hyper-network MLPs that turn each mask token into a per-pixel kernel.
const HYPER_MLP_LAYERS: usize = 3;
/// Expected upscaling kernel shape. SAM stores this layer in checkpoints as
/// a 2x2 stride-2 `ConvTranspose2d`; because stride == kernel and there is no
/// kernel overlap in the output, the math is equivalent to (and implemented
/// here as) a `(in_ch, out_ch * 4)` matmul followed by a pixel-shuffle.
const UPSCALE_KERNEL_HW: [usize; 2] = [2, 2];

/// Private 2x2 stride-2 pixel-shuffle upsampler used by the SAM output head.
///
/// Mathematically equivalent to `ConvTranspose2d(in, out, kernel=2, stride=2)`,
/// but implemented as a `(in_ch, out_ch * 4)` matmul followed by a
/// pixel-shuffle because stride == kernel means output windows never overlap
/// (so no per-pixel accumulation is required).
///
/// This is intentionally local to the SAM port rather than exposed as a
/// generic fusor2 layer.
struct SamPixelShuffleUpscale2x2 {
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
}

impl SamPixelShuffleUpscale2x2 {
    fn load(device: &Device, vb: &VarBuilder) -> Result<Self> {
        let weight: Tensor<4> = load_dense(vb, device, "weight")?;
        let bias = if vb.contains_key("bias") {
            Some(load_dense::<1>(vb, device, "bias")?)
        } else {
            None
        };
        let [_, _, kh, kw] = weight.shape();
        if [kh, kw] != UPSCALE_KERNEL_HW {
            return Err(Error::Shape(format!(
                "SAM upscaling expects a {:?} transposed-conv kernel, got {:?}",
                UPSCALE_KERNEL_HW,
                [kh, kw]
            )));
        }
        Ok(Self { weight, bias })
    }

    fn forward(&self, input: &Tensor<4>) -> Tensor<4> {
        let [b, in_ch, h, w] = input.shape();
        let [_, out_ch, kh, kw] = self.weight.shape();

        let input_flat = input
            .reshape([b, in_ch, h * w])
            .transpose(1, 2)
            .reshape([b * h * w, in_ch]);
        let weight_flat = self.weight.reshape([in_ch, out_ch * kh * kw]);
        let result = input_flat.matmul(&weight_flat);

        let result = result
            .reshape([b, h, w, out_ch, kh, kw])
            // (b, h, w, out, kh, kw) -> (b, out, h, kh, w, kw)
            .permute([0, 3, 1, 4, 2, 5])
            .reshape([b, out_ch, h * kh, w * kw]);

        match &self.bias {
            Some(bias) => result.add_(&bias.reshape([1, out_ch, 1, 1])),
            None => result,
        }
    }
}

struct MlpMaskDecoder {
    layers: Vec<Linear>,
    sigmoid_output: bool,
}

impl MlpMaskDecoder {
    fn load(
        device: &Device,
        vb: &VarBuilder,
        num_layers: usize,
        sigmoid_output: bool,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer = linear(&vb.pp(format!("layers.{i}")), device)?;
            layers.push(layer);
        }
        Ok(Self {
            layers,
            sigmoid_output,
        })
    }

    fn forward<const R: usize>(&self, xs: &Tensor<R>) -> Tensor<R> {
        let mut xs = xs.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(&xs);
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
    iou_token: Embedding,
    mask_tokens: Embedding,
    iou_prediction_head: MlpMaskDecoder,
    output_upscaling_conv1: SamPixelShuffleUpscale2x2,
    output_upscaling_ln: LayerNorm,
    output_upscaling_conv2: SamPixelShuffleUpscale2x2,
    num_mask_tokens: usize,
    output_hypernetworks_mlps: Vec<MlpMaskDecoder>,
    transformer: TwoWayTransformer,
}

impl MaskDecoder {
    pub fn load(
        device: &Device,
        vb: &VarBuilder,
        transformer_dim: usize,
        num_multimask_outputs: usize,
        iou_head_depth: usize,
    ) -> Result<Self> {
        let graph = device.graph().handle();
        let num_mask_tokens = num_multimask_outputs + 1;
        let iou_prediction_head =
            MlpMaskDecoder::load(device, &vb.pp("iou_prediction_head"), iou_head_depth, false)?;
        let iou_token = Embedding::load(&vb.pp("iou_token"), graph)?;
        let mask_tokens = Embedding::load(&vb.pp("mask_tokens"), graph)?;
        let output_upscaling_conv1 =
            SamPixelShuffleUpscale2x2::load(device, &vb.pp("output_upscaling.0"))?;
        let output_upscaling_ln = LayerNorm::load(&vb.pp("output_upscaling.1"), graph, 1e-6)?;
        let output_upscaling_conv2 =
            SamPixelShuffleUpscale2x2::load(device, &vb.pp("output_upscaling.3"))?;
        let mut output_hypernetworks_mlps = Vec::with_capacity(num_mask_tokens);
        for i in 0..num_mask_tokens {
            let mlp = MlpMaskDecoder::load(
                device,
                &vb.pp(format!("output_hypernetworks_mlps.{i}")),
                HYPER_MLP_LAYERS,
                false,
            )?;
            output_hypernetworks_mlps.push(mlp);
        }
        let transformer = TwoWayTransformer::load(
            device,
            &vb.pp("transformer"),
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
        image_embeddings: &Tensor<4>,
        image_pe: &Tensor<4>,
        sparse_prompt_embeddings: &Tensor<3>,
        dense_prompt_embeddings: &Tensor<4>,
        multimask_output: bool,
    ) -> (Tensor<4>, Tensor<2>) {
        let (masks, iou_pred) = self.predict_masks(
            image_embeddings,
            image_pe,
            sparse_prompt_embeddings,
            dense_prompt_embeddings,
        );
        if multimask_output {
            // masks[:, 1:], iou_pred[:, 1:]
            let n_masks = masks.shape()[1];
            let n_iou = iou_pred.shape()[1];
            (
                masks.narrow(1, 1, n_masks - 1),
                iou_pred.narrow(1, 1, n_iou - 1),
            )
        } else {
            // masks[:, 0:1], iou_pred[:, 0:1]
            (masks.narrow(1, 0, 1), iou_pred.narrow(1, 0, 1))
        }
    }

    fn predict_masks(
        &self,
        image_embeddings: &Tensor<4>,
        image_pe: &Tensor<4>,
        sparse_prompt_embeddings: &Tensor<3>,
        dense_prompt_embeddings: &Tensor<4>,
    ) -> (Tensor<4>, Tensor<2>) {
        // Concatenate output tokens: [iou_token, mask_tokens]
        let iou_emb = &self.iou_token.table; // (1, dim)
        let mask_emb = &self.mask_tokens.table; // (num_mask_tokens, dim)
        let output_tokens = Tensor::cat([iou_emb.clone(), mask_emb.clone()], 0);

        let batch_size = sparse_prompt_embeddings.shape()[0];
        let [num_tokens, dim] = output_tokens.shape();

        // Expand to batch: (batch, num_tokens, dim)
        let output_tokens = output_tokens
            .reshape([1, num_tokens, dim])
            .broadcast_as([batch_size, num_tokens, dim]);

        // Cat with sparse prompt embeddings: (batch, num_tokens + num_sparse, dim)
        let tokens = Tensor::cat([output_tokens, sparse_prompt_embeddings.clone()], 1);

        // Expand image data per mask
        let [_, c, h, w] = image_embeddings.shape();

        let src = repeat_interleave_4d(image_embeddings, batch_size).add(dense_prompt_embeddings);
        let pos_src = repeat_interleave_4d(image_pe, batch_size);

        // Run the transformer
        let (hs, src) = self.transformer.forward(&src, &pos_src, &tokens);

        // Extract token outputs
        let iou_token_out = hs.narrow(1, 0, 1).reshape([batch_size, dim]);
        let mask_tokens_out = hs.narrow(1, 1, self.num_mask_tokens);

        // Upscale mask embeddings for the whole prompt batch at once.
        let src = src.transpose(1, 2).reshape([batch_size, c, h, w]);
        let upscaled = self.output_upscaling_conv1.forward(&src);
        let upscaled = channel_layer_norm(&self.output_upscaling_ln, &upscaled).gelu();
        let upscaled = self.output_upscaling_conv2.forward(&upscaled).gelu();

        // Predict masks using hypernetwork MLPs
        let mut hyper_in_list = Vec::with_capacity(self.num_mask_tokens);
        for (i, mlp) in self.output_hypernetworks_mlps.iter().enumerate() {
            let token_i = mask_tokens_out.narrow(1, i, 1).reshape([batch_size, dim]);
            hyper_in_list.push(mlp.forward(&token_i));
        }
        // Stack into (batch, num_mask_tokens, dim/8)
        let hyper_in: Tensor<3> = stack(hyper_in_list, 1);

        let [_, up_c, up_h, up_w] = upscaled.shape();

        // masks = hyper_in @ upscaled.reshape(b, c, h*w)
        let upscaled_flat = upscaled.reshape([batch_size, up_c, up_h * up_w]);
        let masks = hyper_in.matmul(&upscaled_flat);
        let num_masks = masks.shape()[1];
        let masks = masks.reshape([batch_size, num_masks, up_h, up_w]);

        // Generate mask quality predictions
        let iou_pred = self.iou_prediction_head.forward(&iou_token_out);

        (masks, iou_pred)
    }
}

/// Equivalent to torch.repeat_interleave for 4D tensors along dim 0.
fn repeat_interleave_4d(img: &Tensor<4>, repeats: usize) -> Tensor<4> {
    let [b, c, h, w] = img.shape();
    // unsqueeze(1) -> (b, 1, c, h, w), broadcast to (b, repeats, c, h, w), flatten(0,1)
    img.reshape([b, 1, c, h, w])
        .broadcast_as([b, repeats, c, h, w])
        .reshape([b * repeats, c, h, w])
}
