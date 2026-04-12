//! Mask decoder: predicts masks from image+prompt embeddings.

use fusor::layers::{ConvTranspose2d, Embedding, LayerNorm2d, Linear};
use fusor::{ConcreteTensor, Device, Tensor, VarBuilder};

use super::transformer::TwoWayTransformer;
use super::Result;

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
            let layer = Linear::load(device, &mut vb.pp(&format!("layers.{i}")))?;
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

pub struct MaskDecoder {
    pub(crate) iou_token: Embedding<f32>,
    pub(crate) mask_tokens: Embedding<f32>,
    iou_prediction_head: MlpMaskDecoder,
    pub(crate) output_upscaling_conv1: ConvTranspose2d,
    pub(crate) output_upscaling_ln: LayerNorm2d,
    pub(crate) output_upscaling_conv2: ConvTranspose2d,
    pub(crate) num_mask_tokens: usize,
    output_hypernetworks_mlps: Vec<MlpMaskDecoder>,
    pub(crate) transformer: TwoWayTransformer,
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
        let output_upscaling_conv1 =
            ConvTranspose2d::load(device, &mut vb.pp("output_upscaling.0"), [2, 2])?;
        let output_upscaling_ln =
            LayerNorm2d::load(device, &mut vb.pp("output_upscaling.1"), 1e-6)?;
        let output_upscaling_conv2 =
            ConvTranspose2d::load(device, &mut vb.pp("output_upscaling.3"), [2, 2])?;
        let mut output_hypernetworks_mlps = Vec::with_capacity(num_mask_tokens);
        for i in 0..num_mask_tokens {
            let mlp = MlpMaskDecoder::load(
                device,
                &mut vb.pp(&format!("output_hypernetworks_mlps.{i}")),
                3,
                false,
            )?;
            output_hypernetworks_mlps.push(mlp);
        }
        let transformer = TwoWayTransformer::load(
            device,
            &mut vb.pp("transformer"),
            2,
            transformer_dim,
            8,
            2048,
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

        // Upscale mask embeddings
        // Process each batch item individually through the upscaling pipeline
        // to work around a GPU framework issue with batch > 1 in ConvTranspose2d/LayerNorm2d.
        let src: Tensor<4, f32> = src
            .transpose(1, 2)
            .to_concrete()
            .reshape([batch_size, c, h, w])
            .to_concrete();

        let upscaled = if batch_size == 1 {
            let upscaled = self.output_upscaling_conv1.forward(&src);
            let upscaled = self.output_upscaling_ln.forward(&upscaled);
            let upscaled = upscaled.gelu();
            let upscaled = self.output_upscaling_conv2.forward(&upscaled.to_concrete());
            upscaled.gelu().to_concrete()
        } else {
            let mut upscaled_items = Vec::with_capacity(batch_size);
            for i in 0..batch_size {
                let src_i: Tensor<4, f32> = src.narrow(0, i, 1).to_concrete();
                let up_i = self.output_upscaling_conv1.forward(&src_i);
                let up_i = self.output_upscaling_ln.forward(&up_i);
                let up_i = up_i.gelu();
                let up_i = self.output_upscaling_conv2.forward(&up_i.to_concrete());
                let up_i = up_i.gelu();
                upscaled_items.push(up_i.to_concrete());
            }
            Tensor::cat(upscaled_items, 0).to_concrete()
        };

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
