//! Top-level Sam model: ties together image encoder, prompt encoder, and mask decoder.

use fusor::{ConcreteTensor, Device, Tensor, TensorBacking, VarBuilder};

use super::image_encoder::ImageEncoderViT;
use super::mask_decoder::MaskDecoder;
use super::prompt_encoder::PromptEncoder;
use super::tiny_vit::{tiny_vit_5m, TinyViT};
use super::Result;

const PROMPT_EMBED_DIM: usize = 256;
/// The expected image size (both width and height) for the SAM model.
pub const IMAGE_SIZE: usize = 1024;
const VIT_PATCH_SIZE: usize = 16;
pub(crate) const PRED_IOU_THRESH: f32 = 0.78;
pub(crate) const STABILITY_SCORE_OFFSET: f32 = 1.0;
pub(crate) const STABILITY_SCORE_THRESHOLD: f32 = 0.88;
pub(crate) const MODEL_MASK_THRESHOLD: f32 = 0.0;
pub(crate) const CROP_NMS_THRESH: f32 = 0.7;

pub(crate) enum ImageEncoder {
    Original(Box<ImageEncoderViT>),
    TinyViT(Box<TinyViT>),
}

impl ImageEncoder {
    fn forward(&self, xs: &Tensor<4, f32, impl TensorBacking<4, Elem = f32>>) -> Tensor<4, f32> {
        match self {
            Self::Original(vit) => vit.forward(xs),
            Self::TinyViT(vit) => vit.forward(xs),
        }
    }
}

/// The Segment Anything Model.
pub struct Sam {
    pub(crate) image_encoder: ImageEncoder,
    pub(crate) prompt_encoder: PromptEncoder,
    pub(crate) mask_decoder: MaskDecoder,
    pub(crate) pixel_mean: [f32; 3],
    pub(crate) pixel_std: [f32; 3],
}

impl Sam {
    /// Load a ViT-B based SAM model.
    pub fn load_vit_b(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        Self::load_vit(
            device,
            vb,
            768,            // embed_dim
            12,             // depth
            12,             // num_heads
            &[2, 5, 8, 11], // global_attn_indexes
        )
    }

    /// Load a ViT-based SAM model with custom architecture parameters.
    pub fn load_vit(
        device: &Device,
        vb: &mut VarBuilder,
        encoder_embed_dim: usize,
        encoder_depth: usize,
        encoder_num_heads: usize,
        encoder_global_attn_indexes: &[usize],
    ) -> Result<Self> {
        let image_embedding_size = IMAGE_SIZE / VIT_PATCH_SIZE;

        let image_encoder = ImageEncoderViT::load(
            device,
            &mut vb.pp("image_encoder"),
            IMAGE_SIZE,
            VIT_PATCH_SIZE,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            PROMPT_EMBED_DIM,
            true, // use_rel_pos
            true, // use_abs_pos
            14,   // window_size
            encoder_global_attn_indexes,
        )?;

        let prompt_encoder = PromptEncoder::load(
            device,
            &mut vb.pp("prompt_encoder"),
            PROMPT_EMBED_DIM,
            (image_embedding_size, image_embedding_size),
            (IMAGE_SIZE, IMAGE_SIZE),
        )?;

        let mask_decoder = MaskDecoder::load(
            device,
            &mut vb.pp("mask_decoder"),
            PROMPT_EMBED_DIM,
            3, // num_multimask_outputs
            3, // iou_head_depth
        )?;

        Ok(Self {
            image_encoder: ImageEncoder::Original(Box::new(image_encoder)),
            prompt_encoder,
            mask_decoder,
            pixel_mean: [123.675, 116.28, 103.53],
            pixel_std: [58.395, 57.12, 57.375],
        })
    }

    /// Load a TinyViT-based (MobileSAM) model.
    pub fn load_tiny(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let image_embedding_size = IMAGE_SIZE / VIT_PATCH_SIZE;

        let image_encoder = tiny_vit_5m(device, &mut vb.pp("image_encoder"))?;

        let prompt_encoder = PromptEncoder::load(
            device,
            &mut vb.pp("prompt_encoder"),
            PROMPT_EMBED_DIM,
            (image_embedding_size, image_embedding_size),
            (IMAGE_SIZE, IMAGE_SIZE),
        )?;

        let mask_decoder = MaskDecoder::load(
            device,
            &mut vb.pp("mask_decoder"),
            PROMPT_EMBED_DIM,
            3, // num_multimask_outputs
            3, // iou_head_depth
        )?;

        Ok(Self {
            image_encoder: ImageEncoder::TinyViT(Box::new(image_encoder)),
            prompt_encoder,
            mask_decoder,
            pixel_mean: [123.675, 116.28, 103.53],
            pixel_std: [58.395, 57.12, 57.375],
        })
    }

    /// Compute image embeddings.
    pub fn embeddings(&self, img: &Tensor<3, f32, ConcreteTensor<f32, 3>>) -> Tensor<4, f32> {
        let img = self.preprocess(img);
        // Add batch dim: (C, H, W) -> (1, C, H, W)
        let shape = img.shape();
        let img = img.reshape([1, shape[0], shape[1], shape[2]]);
        self.image_encoder.forward(&img)
    }

    /// Forward pass: image -> masks + IoU predictions.
    ///
    /// Points format: `(x, y, is_foreground)` where x,y are in [0,1] normalized coords.
    pub fn forward(
        &self,
        img: &Tensor<3, f32, ConcreteTensor<f32, 3>>,
        points: &[(f64, f64, bool)],
        multimask_output: bool,
    ) -> (Tensor<4, f32>, Tensor<2, f32>) {
        let shape = img.shape();
        let original_h = shape[1];
        let original_w = shape[2];

        let img = self.preprocess(img);
        // (C, H, W) -> (1, C, H, W)
        let img_shape = img.shape();
        let img = img.reshape([1, img_shape[0], img_shape[1], img_shape[2]]);
        let img_embeddings = self.image_encoder.forward(&img);

        let (low_res_mask, iou) = self.forward_for_embeddings(
            &img_embeddings,
            original_h,
            original_w,
            points,
            multimask_output,
        );

        // Upsample to IMAGE_SIZE
        let upscaled: Tensor<4, f32> = low_res_mask.upsample_nearest2d(
            IMAGE_SIZE / low_res_mask.shape()[2],
            IMAGE_SIZE / low_res_mask.shape()[3],
        );

        // Crop to original size: narrow on H and W dims
        let cropped_h = upscaled.narrow(2, 0, original_h);
        let cropped = cropped_h.narrow(3, 0, original_w);

        (cropped.to_concrete(), iou)
    }

    /// Generate mask and IoU predictions from pre-computed image embeddings.
    ///
    /// Points format: `(x, y, is_foreground)` where x,y are normalized to [0,1].
    pub fn forward_for_embeddings(
        &self,
        img_embeddings: &Tensor<4, f32>,
        original_h: usize,
        original_w: usize,
        points: &[(f64, f64, bool)],
        multimask_output: bool,
    ) -> (Tensor<4, f32>, Tensor<2, f32>) {
        let device = img_embeddings.device();
        let image_pe = self.prompt_encoder.get_dense_pe();

        let points_data = if points.is_empty() {
            None
        } else {
            let n_points = points.len();
            let xys: Vec<f32> = points
                .iter()
                .flat_map(|(x, y, _b)| {
                    let x = (*x as f32) * (original_w as f32);
                    let y = (*y as f32) * (original_h as f32);
                    [x, y]
                })
                .collect();
            let labels: Vec<f32> = points
                .iter()
                .map(|(_x, _y, b)| if *b { 1f32 } else { 0f32 })
                .collect();
            let pts: Tensor<3, f32> = Tensor::from_slice(&device, [1, n_points, 2], &xys);
            let lbls: Tensor<2, f32> = Tensor::from_slice(&device, [1, n_points], &labels);
            Some((pts, lbls))
        };

        let points_ref = points_data
            .as_ref()
            .map(|(pts, lbls)| (pts as &Tensor<3, f32>, lbls as &Tensor<2, f32>));

        let (sparse_prompt_embeddings, dense_prompt_embeddings) =
            self.prompt_encoder.forward(points_ref, None, None);

        self.mask_decoder.forward(
            img_embeddings,
            &image_pe,
            &sparse_prompt_embeddings,
            &dense_prompt_embeddings,
            multimask_output,
        )
    }

    /// Generate masks and IoU predictions for a batch of single-point prompts
    /// from pre-computed image embeddings.
    ///
    /// Each element in `points` is `(x, y, is_foreground)` and becomes one batch item.
    /// All points are processed in a single pass through the prompt encoder and mask decoder.
    ///
    /// Returns `(masks, iou_predictions)`:
    /// - masks: `(batch, n_masks, h, w)` where n_masks=3 if multimask_output
    /// - iou_predictions: `(batch, n_masks)`
    pub fn forward_for_embeddings_batched(
        &self,
        img_embeddings: &Tensor<4, f32>,
        original_h: usize,
        original_w: usize,
        points: &[(f64, f64, bool)],
        multimask_output: bool,
    ) -> (Tensor<4, f32>, Tensor<2, f32>) {
        let device = img_embeddings.device();
        let image_pe = self.prompt_encoder.get_dense_pe();
        let batch_size = points.len();

        // Build batched point tensors: (batch_size, 1, 2) and (batch_size, 1)
        let xys: Vec<f32> = points
            .iter()
            .flat_map(|(x, y, _b)| {
                let x = (*x as f32) * (original_w as f32);
                let y = (*y as f32) * (original_h as f32);
                [x, y]
            })
            .collect();
        let labels: Vec<f32> = points
            .iter()
            .map(|(_x, _y, b)| if *b { 1f32 } else { 0f32 })
            .collect();

        let pts: Tensor<3, f32> = Tensor::from_slice(&device, [batch_size, 1, 2], &xys);
        let lbls: Tensor<2, f32> = Tensor::from_slice(&device, [batch_size, 1], &labels);

        let (sparse_prompt_embeddings, dense_prompt_embeddings) =
            self.prompt_encoder.forward(Some((&pts, &lbls)), None, None);

        self.mask_decoder.forward(
            img_embeddings,
            &image_pe,
            &sparse_prompt_embeddings,
            &dense_prompt_embeddings,
            multimask_output,
        )
    }

    /// Preprocess an image tensor: normalize by pixel mean/std and pad to IMAGE_SIZE.
    pub(crate) fn preprocess(&self, img: &Tensor<3, f32>) -> Tensor<3, f32> {
        let shape = img.shape();
        let c = shape[0];
        let h = shape[1];
        let w = shape[2];
        let device = img.device();

        // Create mean and std tensors: (3, 1, 1) broadcast to (3, H, W)
        let mean_base = Tensor::from_slice(&device, [3, 1, 1], &self.pixel_mean);
        let mean = mean_base.broadcast_as([c, h, w]);
        let std_base = Tensor::from_slice(&device, [3, 1, 1], &self.pixel_std);
        let std = std_base.broadcast_as([c, h, w]);

        let img: Tensor<3, f32> = ((img - mean) / std).to_concrete();

        // Pad to IMAGE_SIZE
        let img = if h < IMAGE_SIZE {
            img.pad_with_zeros(1, 0, IMAGE_SIZE - h).to_concrete()
        } else {
            img
        };
        let img = if w < IMAGE_SIZE {
            img.pad_with_zeros(2, 0, IMAGE_SIZE - w).to_concrete()
        } else {
            img
        };
        img.to_concrete()
    }
}

// Grid-based mask generation support

struct CropBox {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    layer_idx: usize,
}

fn generate_crop_boxes(
    (im_h, im_w): (usize, usize),
    n_layers: usize,
    overlap_ratio: f64,
) -> Vec<CropBox> {
    fn crop_len(orig_len: usize, n_crops: usize, overlap: usize) -> usize {
        f64::ceil((overlap * (n_crops - 1) + orig_len) as f64 / n_crops as f64) as usize
    }

    let short_side = usize::min(im_h, im_w);
    let mut crop_boxes = Vec::new();

    crop_boxes.push(CropBox {
        x0: 0,
        y0: 0,
        x1: im_w,
        y1: im_h,
        layer_idx: 0,
    });

    for layer_idx in 1..=n_layers {
        let n_crops_per_side = 1 << layer_idx;
        let overlap = (overlap_ratio * short_side as f64 * 2.0 / n_crops_per_side as f64) as usize;
        let crop_w = crop_len(im_w, n_crops_per_side, overlap);
        let crop_h = crop_len(im_h, n_crops_per_side, overlap);

        for i_x in 0..n_crops_per_side {
            let x0 = (crop_w - overlap) * i_x;
            for i_y in 0..n_crops_per_side {
                let y0 = (crop_h - overlap) * i_y;
                let x1 = usize::min(im_w, x0 + crop_w);
                let y1 = usize::min(im_h, y0 + crop_h);
                crop_boxes.push(CropBox {
                    x0,
                    y0,
                    x1,
                    y1,
                    layer_idx,
                });
            }
        }
    }

    crop_boxes
}

pub(crate) fn build_point_grid(n_per_side: usize) -> Vec<(f64, f64)> {
    let offset = 1f64 / (2 * n_per_side) as f64;
    let mut points = Vec::with_capacity(n_per_side * n_per_side);
    for i_x in 0..n_per_side {
        let x = offset + i_x as f64 / n_per_side as f64;
        for i_y in 0..n_per_side {
            let y = offset + i_y as f64 / n_per_side as f64;
            points.push((x, y));
        }
    }
    points
}

fn build_all_layer_point_grids(
    n_per_side: usize,
    n_layers: usize,
    scale_per_layer: usize,
) -> Vec<Vec<(f64, f64)>> {
    let mut points_by_layer = Vec::with_capacity(n_layers + 1);
    for i in 0..=n_layers {
        let n_points = n_per_side / scale_per_layer.pow(i as u32);
        points_by_layer.push(build_point_grid(n_points));
    }
    points_by_layer
}
