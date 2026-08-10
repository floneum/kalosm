//! Top-level Sam model: ties together image encoder, prompt encoder, and mask decoder.

use fusor2::composite::upsample::upsample_nearest2d;
use fusor2::graph::Graph;
use fusor2::tensor::Dyn as Tensor;
use fusor2_gguf::VarBuilder;

use super::image_encoder::ImageEncoderViT;
use super::mask_decoder::MaskDecoder;
use super::prompt_encoder::PromptEncoder;
use super::tiny_vit::{tiny_vit_5m, TinyViT};
use super::{dims, udim, Result};

const PROMPT_EMBED_DIM: usize = 256;
/// The expected image size (both width and height) for the SAM model.
pub const IMAGE_SIZE: usize = 1024;
/// Patch size for the standard ViT image encoder. The TinyViT/MobileSAM encoder
/// also happens to downsample by 16 across its full stride stack - we rely on
/// this coincidence so a single `IMAGE_SIZE / VIT_PATCH_SIZE` constant works
/// for the prompt-encoder geometry. SAM2 variants must NOT reuse this constant.
const VIT_PATCH_SIZE: usize = 16;
/// Pixel-mean used to normalize input images (matches Meta's SAM checkpoint).
const PIXEL_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
/// Pixel-std used to normalize input images.
const PIXEL_STD: [f32; 3] = [58.395, 57.12, 57.375];

enum ImageEncoder {
    Original(Box<ImageEncoderViT>),
    TinyViT(Box<TinyViT>),
}

impl ImageEncoder {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Original(vit) => vit.forward(xs),
            Self::TinyViT(vit) => vit.forward(xs),
        }
    }
}

/// The Segment Anything Model.
pub struct Sam {
    graph: Graph,
    image_encoder: ImageEncoder,
    prompt_encoder: PromptEncoder,
    mask_decoder: MaskDecoder,
}

impl Sam {
    /// Load a ViT-B based SAM model.
    pub fn load_vit_b(graph: &Graph, vb: &VarBuilder) -> Result<Self> {
        Self::load_vit(
            graph,
            vb,
            768,            // embed_dim
            12,             // depth
            12,             // num_heads
            &[2, 5, 8, 11], // global_attn_indexes
        )
    }

    /// Load a ViT-based SAM model with custom architecture parameters.
    pub fn load_vit(
        graph: &Graph,
        vb: &VarBuilder,
        encoder_embed_dim: usize,
        encoder_depth: usize,
        encoder_num_heads: usize,
        encoder_global_attn_indexes: &[usize],
    ) -> Result<Self> {
        let image_embedding_size = IMAGE_SIZE / VIT_PATCH_SIZE;

        let image_encoder = ImageEncoderViT::load(
            graph,
            &vb.pp("image_encoder"),
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
            graph,
            &vb.pp("prompt_encoder"),
            PROMPT_EMBED_DIM,
            (image_embedding_size, image_embedding_size),
            (IMAGE_SIZE, IMAGE_SIZE),
        )?;

        let mask_decoder = MaskDecoder::load(
            graph,
            &vb.pp("mask_decoder"),
            PROMPT_EMBED_DIM,
            3, // num_multimask_outputs
            3, // iou_head_depth
        )?;

        Ok(Self {
            graph: graph.clone(),
            image_encoder: ImageEncoder::Original(Box::new(image_encoder)),
            prompt_encoder,
            mask_decoder,
        })
    }

    /// Load a TinyViT-based (MobileSAM) model.
    pub fn load_tiny(graph: &Graph, vb: &VarBuilder) -> Result<Self> {
        let image_embedding_size = IMAGE_SIZE / VIT_PATCH_SIZE;

        let image_encoder = tiny_vit_5m(graph, &vb.pp("image_encoder"))?;

        let prompt_encoder = PromptEncoder::load(
            graph,
            &vb.pp("prompt_encoder"),
            PROMPT_EMBED_DIM,
            (image_embedding_size, image_embedding_size),
            (IMAGE_SIZE, IMAGE_SIZE),
        )?;

        let mask_decoder = MaskDecoder::load(
            graph,
            &vb.pp("mask_decoder"),
            PROMPT_EMBED_DIM,
            3, // num_multimask_outputs
            3, // iou_head_depth
        )?;

        Ok(Self {
            graph: graph.clone(),
            image_encoder: ImageEncoder::TinyViT(Box::new(image_encoder)),
            prompt_encoder,
            mask_decoder,
        })
    }

    /// The graph this model's weights live in. Inputs must be built in it.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Compute image embeddings from a `(C, H, W)` image tensor.
    pub fn embeddings(&self, img: &Tensor) -> Result<Tensor> {
        let img = self.preprocess(img)?;
        // Add batch dim: (C, H, W) -> (1, C, H, W)
        let img = img.unsqueeze(0)?;
        self.image_encoder.forward(&img)
    }

    /// Forward pass: image -> masks + IoU predictions.
    ///
    /// Points format: `(x, y, is_foreground)` where x,y are in [0,1] normalized coords.
    pub fn forward(
        &self,
        img: &Tensor,
        points: &[(f64, f64, bool)],
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor)> {
        let original_h = udim(img, 1);
        let original_w = udim(img, 2);

        let img_embeddings = self.embeddings(img)?;

        let (low_res_mask, iou) = self.forward_for_embeddings(
            &img_embeddings,
            original_h,
            original_w,
            points,
            multimask_output,
        )?;

        // Upsample to IMAGE_SIZE.
        // Low-res masks come back at exactly IMAGE_SIZE/4 (256). If a
        // future model changes the upsampling ratio this assert will catch it
        // before `upsample_nearest2d` silently truncates.
        let lr_h = udim(&low_res_mask, 2);
        let lr_w = udim(&low_res_mask, 3);
        let scale_h = IMAGE_SIZE / lr_h;
        let scale_w = IMAGE_SIZE / lr_w;
        assert_eq!(
            scale_h * lr_h,
            IMAGE_SIZE,
            "low-res mask H ({lr_h}) must divide IMAGE_SIZE ({IMAGE_SIZE})",
        );
        assert_eq!(
            scale_w * lr_w,
            IMAGE_SIZE,
            "low-res mask W ({lr_w}) must divide IMAGE_SIZE ({IMAGE_SIZE})",
        );
        let upscaled = upsample_nearest2d(&low_res_mask, scale_h as u32, scale_w as u32)?;

        // Crop to original size: narrow on H and W dims
        let cropped = upscaled
            .narrow(2, 0, original_h)?
            .narrow(3, 0, original_w)?;

        Ok((cropped, iou))
    }

    /// Generate mask and IoU predictions from pre-computed image embeddings.
    ///
    /// Points format: `(x, y, is_foreground)` where x,y are normalized to [0,1].
    pub fn forward_for_embeddings(
        &self,
        img_embeddings: &Tensor,
        original_h: usize,
        original_w: usize,
        points: &[(f64, f64, bool)],
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor)> {
        // Single-batch path; equivalent to calling the batched variant with
        // batch_size = 1 but producing a `(1, 1, 2)` point tensor.
        let image_pe = self.prompt_encoder.get_dense_pe()?;

        let points_data = if points.is_empty() {
            None
        } else {
            Some(build_point_tensors(
                &self.graph,
                points,
                original_h,
                original_w,
                1,
            )?)
        };

        let points_ref = points_data.as_ref().map(|(pts, lbls)| (pts, lbls));

        let (sparse_prompt_embeddings, dense_prompt_embeddings) =
            self.prompt_encoder.forward(points_ref, None, None)?;

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
        img_embeddings: &Tensor,
        original_h: usize,
        original_w: usize,
        points: &[(f64, f64, bool)],
        multimask_output: bool,
    ) -> Result<(Tensor, Tensor)> {
        let image_pe = self.prompt_encoder.get_dense_pe()?;
        let batch_size = points.len();

        let (pts, lbls) =
            build_point_tensors(&self.graph, points, original_h, original_w, batch_size)?;

        let (sparse_prompt_embeddings, dense_prompt_embeddings) =
            self.prompt_encoder.forward(Some((&pts, &lbls)), None, None)?;

        self.mask_decoder.forward(
            img_embeddings,
            &image_pe,
            &sparse_prompt_embeddings,
            &dense_prompt_embeddings,
            multimask_output,
        )
    }

    /// Preprocess an image tensor: normalize by pixel mean/std and pad to IMAGE_SIZE.
    pub(crate) fn preprocess(&self, img: &Tensor) -> Result<Tensor> {
        let c = udim(img, 0);
        let h = udim(img, 1);
        let w = udim(img, 2);
        // Callers (`image_to_tensor`) resize so the longer side is exactly
        // IMAGE_SIZE; assert here so a mistake elsewhere fails loudly instead
        // of producing a `pad_with_zeros(.., IMAGE_SIZE - h)` underflow deep
        // in the tensor stack.
        assert!(
            h <= IMAGE_SIZE && w <= IMAGE_SIZE,
            "preprocess input ({h}x{w}) exceeds IMAGE_SIZE ({IMAGE_SIZE}); resize before calling",
        );
        let graph = self.graph.handle();

        // Create mean and std tensors: (3, 1, 1) broadcast to (3, H, W)
        let mean = Tensor::from_elements(graph, &dims(&[3, 1, 1]), &PIXEL_MEAN)?
            .broadcast_as(&dims(&[c, h, w]))?;
        let std = Tensor::from_elements(graph, &dims(&[3, 1, 1]), &PIXEL_STD)?
            .broadcast_as(&dims(&[c, h, w]))?;

        let img = img.sub(&mean)?.div(&std)?;

        // Pad to IMAGE_SIZE
        let img = if h < IMAGE_SIZE {
            img.pad_with_zeros(1, 0, IMAGE_SIZE - h)?
        } else {
            img
        };
        let img = if w < IMAGE_SIZE {
            img.pad_with_zeros(2, 0, IMAGE_SIZE - w)?
        } else {
            img
        };
        Ok(img)
    }
}

/// Convert normalized `(x, y, is_foreground)` prompt points into the
/// `(batch_size, n_points_per_batch, 2)` xy tensor and `(batch_size,
/// n_points_per_batch)` label tensor expected by the prompt encoder.
///
/// The two callers use this in different modes:
/// - `forward_for_embeddings` passes `batch_size = 1`, so `points` is laid out
///   as `[1, points.len(), 2]` (one prompt with N points).
/// - `forward_for_embeddings_batched` passes `batch_size = points.len()`, so
///   `points` is laid out as `[N, 1, 2]` (N prompts with one point each).
///
/// `points.len()` must equal `batch_size * n_points_per_batch` exactly.
fn build_point_tensors(
    graph: &Graph,
    points: &[(f64, f64, bool)],
    original_h: usize,
    original_w: usize,
    batch_size: usize,
) -> Result<(Tensor, Tensor)> {
    assert!(
        batch_size > 0 && points.len().is_multiple_of(batch_size),
        "build_point_tensors: points.len() ({}) must be a multiple of batch_size ({batch_size})",
        points.len(),
    );
    let n_per_batch = points.len() / batch_size;
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
    let pts = Tensor::from_elements(graph.handle(), &dims(&[batch_size, n_per_batch, 2]), &xys)?;
    let lbls = Tensor::from_elements(graph.handle(), &dims(&[batch_size, n_per_batch]), &labels)?;
    Ok((pts, lbls))
}

/// Build a uniform `n_per_side` by `n_per_side` grid of normalized `(x, y)`
/// coordinates in `(0, 1)`. Used as the prompt grid for `segment_everything`.
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
