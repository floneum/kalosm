//! # Segment Anything RS
//! A rust wrapper for [Segment Anything](https://segment-anything.com/)
//!
//! The model uses fusor tensors and prefers GPU execution when available,
//! automatically falling back to CPU otherwise.
//!
//! ## Usage
//!
//! ```rust, no_run
//! use segment_anything_rs::*;
//!
//! # async fn run() {
//! let model = SegmentAnything::builder().build().await.unwrap();
//! let image = image::open("examples/landscape.jpg").unwrap();
//! let images = model.segment_everything(image).await.unwrap();
//! for (i, img) in images.iter().enumerate() {
//!     img.save(&format!("{}.png", i)).unwrap();
//! }
//! # }
//! ```

#![warn(missing_docs)]

mod raw;

use fusor::{Concrete, Device, Tensor, ToVec1, VarBuilder};
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgba};
use kalosm_model_types::FileSource;
use raw::sam::{
    build_point_grid, Sam, CROP_NMS_THRESH, IMAGE_SIZE, MODEL_MASK_THRESHOLD, PRED_IOU_THRESH,
    STABILITY_SCORE_OFFSET, STABILITY_SCORE_THRESHOLD,
};

/// Number of grid points evaluated per forward pass in `segment_everything`.
const SEGMENT_EVERYTHING_BATCH_SIZE: usize = 64;

/// A builder for [`SegmentAnything`].
#[derive(Default)]
pub struct SegmentAnythingBuilder {
    source: SegmentAnythingSource,
    device: Option<Device>,
    local_path: Option<std::path::PathBuf>,
}

impl SegmentAnythingBuilder {
    /// Sets the source of the model.
    pub fn source(mut self, source: SegmentAnythingSource) -> Self {
        self.source = source;
        self
    }

    /// Sets the fusor device used to load and run the model.
    ///
    /// When not specified, the builder prefers GPU and falls back to CPU.
    pub fn device(mut self, device: Device) -> Self {
        self.device = Some(device);
        self
    }

    /// Load weights from a local GGUF file instead of downloading from
    /// Hugging Face. Pair with [`SegmentAnythingBuilder::source`] (or rely on
    /// the default `tiny` source) to indicate which architecture the file
    /// contains - the source's `tiny` flag controls whether the loader uses
    /// `Sam::load_tiny` or `Sam::load_vit_b`.
    pub fn gguf_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.local_path = Some(path.into());
        self
    }

    /// Builds the [`SegmentAnything`] model.
    pub async fn build(self) -> Result<SegmentAnything, LoadSegmentAnythingError> {
        SegmentAnything::new(self).await
    }
}

/// The source of the model.
pub struct SegmentAnythingSource {
    model: String,
    filename: String,
    tiny: bool,
}

impl SegmentAnythingSource {
    /// Creates a new [`SegmentAnythingSource`].
    pub fn new(model: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            filename: filename.into(),
            tiny: false,
        }
    }

    /// Create the tiny SAM model source.
    pub fn tiny() -> Self {
        let mut self_ = Self::new("Demonthos/MobileSamGguf", "mobile_sam-tiny-vitt.gguf");
        self_.tiny = true;
        self_
    }

    /// Create a normal sized model source.
    pub fn medium() -> Self {
        Self::new("Demonthos/MobileSamGguf", "sam_vit_b_01ec64.gguf")
    }
}

impl Default for SegmentAnythingSource {
    fn default() -> Self {
        Self::tiny()
    }
}

/// Settings for running inference on [`SegmentAnything`].
pub struct SegmentAnythingInferenceSettings {
    threshold: f32,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image).
    goal_points: Vec<(f64, f64)>,

    /// List of x,y coordinates, between 0 and 1 (0.5 is at the middle of the image).
    avoid_points: Vec<(f64, f64)>,

    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
}

impl SegmentAnythingInferenceSettings {
    /// Creates a new [`SegmentAnythingInferenceSettings`] from an image.
    pub fn new<I: GenericImageView<Pixel = Rgba<u8>>>(input: I) -> Self {
        let mut image = ImageBuffer::new(input.width(), input.height());
        image.copy_from(&input, 0, 0).unwrap();
        Self {
            threshold: 0.,
            goal_points: Vec::new(),
            avoid_points: Vec::new(),
            image,
        }
    }

    /// Sets the detection threshold for the mask, 0 is the default value.
    /// - A negative values makes the model return a larger mask.
    /// - A positive makes the model return a smaller mask.
    pub fn set_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Add a point to the list of points to segment.
    ///
    /// Coordinates are normalized to `[0, 1]` (0.5 is the middle of the image).
    /// Renamed from `add_goal_point` in 0.5 to flag the absolute-to-normalized
    /// coordinate switch - old pixel-based callers should divide by image size.
    pub fn add_goal_point_normalized(mut self, x: impl Into<f64>, y: impl Into<f64>) -> Self {
        self.goal_points.push((x.into(), y.into()));
        self
    }

    /// Set the list of points to segment.
    ///
    /// Coordinates are normalized to `[0, 1]`.
    pub fn set_goal_points(mut self, points: Vec<(f64, f64)>) -> Self {
        self.goal_points = points;
        self
    }

    /// Add a point to the list of points to avoid.
    ///
    /// Coordinates are normalized to `[0, 1]` (0.5 is the middle of the image).
    /// Renamed from `add_avoid_points` in 0.5 to flag the absolute-to-normalized
    /// coordinate switch and fix the singular/plural mismatch.
    pub fn add_avoid_point_normalized(mut self, x: impl Into<f64>, y: impl Into<f64>) -> Self {
        self.avoid_points.push((x.into(), y.into()));
        self
    }

    /// Set the list of points to avoid.
    ///
    /// Coordinates are normalized to `[0, 1]`.
    pub fn set_avoid_points(mut self, points: Vec<(f64, f64)>) -> Self {
        self.avoid_points = points;
        self
    }

    /// Set the image to segment.
    pub fn set_image<I: GenericImageView<Pixel = Rgba<u8>>>(
        mut self,
        image: I,
    ) -> Result<Self, image::ImageError> {
        self.image = ImageBuffer::new(image.width(), image.height());
        self.image.copy_from(&image, 0, 0)?;
        Ok(self)
    }
}

/// An error that can occur when loading a [`SegmentAnything`] model.
#[derive(Debug, thiserror::Error)]
pub enum LoadSegmentAnythingError {
    /// An error that can occur when initializing the runtime for a [`SegmentAnything`] model.
    #[error("Failed to initialize model runtime: {0}")]
    LoadModel(#[from] fusor::Error),
    /// An error that can occur when downloading a [`SegmentAnything`] model from Hugging Face.
    #[error("Failed to download model from Hugging Face: {0}")]
    DownloadModel(#[from] kalosm_common::CacheError),
    /// An IO error opening the model file.
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// An error that can occur when running a [`SegmentAnything`] model.
#[derive(Debug, thiserror::Error)]
pub enum SegmentAnythingInferenceError {
    /// An error that can occur when trying to run a [`SegmentAnything`] model.
    #[error("Failed to run model: {0}")]
    RunModel(#[from] fusor::Error),
    /// An error that can occur when converting the result of a [`SegmentAnything`] model to an image.
    #[error("Failed to merge masks")]
    MergeMasks,
}

/// The [segment anything](https://segment-anything.com/) model.
pub struct SegmentAnything {
    sam: Sam,
    device: Device,
}

impl SegmentAnything {
    /// Creates a new [`SegmentAnythingBuilder`].
    pub fn builder() -> SegmentAnythingBuilder {
        SegmentAnythingBuilder::default()
    }

    async fn new(settings: SegmentAnythingBuilder) -> Result<Self, LoadSegmentAnythingError> {
        let SegmentAnythingBuilder {
            source,
            device,
            local_path,
        } = settings;
        let model_path = match local_path {
            Some(path) => path,
            None => {
                let source =
                    FileSource::huggingface(source.model, "main".to_string(), source.filename);
                kalosm_common::Cache::default().get(&source, |_| {}).await?
            }
        };
        let device = match device {
            Some(device) => device,
            None => Device::auto().await,
        };
        let mut reader = std::io::BufReader::new(std::fs::File::open(&model_path)?);
        let mut vb = VarBuilder::from_gguf(&mut reader)
            .map_err(|e| fusor::Error::msg(format!("Failed to read GGUF: {e}")))?;
        let sam = if source.tiny {
            Sam::load_tiny(&device, &mut vb)?
        } else {
            Sam::load_vit_b(&device, &mut vb)?
        };
        Ok(Self { sam, device })
    }

    /// Segment an image from a list of points. Returns a [`DynamicImage`] mask.
    ///
    /// # Example
    /// ```rust, no_run
    /// use segment_anything_rs::*;
    ///
    /// # async fn run() {
    /// let model = SegmentAnything::builder().build().await.unwrap();
    /// let image = image::open("examples/landscape.jpg").unwrap();
    /// let images = model
    ///     .segment_from_points(SegmentAnythingInferenceSettings::new(image).add_goal_point_normalized(0.5, 0.25))
    ///     .await
    ///     .unwrap();
    ///
    /// images.save("out.png").unwrap();
    /// # }
    /// ```
    pub async fn segment_from_points(
        &self,
        settings: SegmentAnythingInferenceSettings,
    ) -> Result<DynamicImage, SegmentAnythingInferenceError> {
        let SegmentAnythingInferenceSettings {
            threshold,
            goal_points,
            avoid_points,
            image,
        } = settings;

        let image = image::DynamicImage::ImageRgba8(image);
        let image_width = image.width();
        let image_height = image.height();

        let image_tensor = self.image_to_tensor(image);

        let points = {
            let mut points = Vec::new();
            for (x, y) in goal_points {
                points.push((x, y, true));
            }
            for (x, y) in avoid_points {
                points.push((x, y, false));
            }
            points
        };

        let (mask, _iou_predictions) = self.sam.forward(&image_tensor, &points, false);

        let mask_shape = mask.shape();
        let h = mask_shape[2];
        let w = mask_shape[3];

        // Get first mask (batch=0, mask=0)
        let mask_n0 = mask.narrow(0, 0, 1);
        let mask_n1 = mask_n0.narrow(1, 0, 1);
        let mask_2d = mask_n1.reshape([h, w]);

        // Threshold: >= threshold -> 255, else 0
        let threshold_mask = mask_2d.gt_scalar(threshold - 1e-6);

        let mask_u8: Tensor<2, f32> = threshold_mask.mul_scalar(255.0f32);

        // Expand to 3 channels: (H, W) -> (3, H, W)
        let mask_reshaped = mask_u8.reshape([1, h, w]);
        let mask_3ch = mask_reshaped.broadcast_as([3, h, w]);

        // Permute to (H, W, 3) and flatten
        let mask_t1 = mask_3ch.transpose(0, 1); // (H, 3, W)
        let mask_hwc = mask_t1.transpose(1, 2); // (H, W, 3);
        let mask_flat = mask_hwc.reshape([h * w * 3]);
        let mask_slice = mask_flat.as_slice().await?;
        let mask_pixels: Vec<u8> = mask_slice.to_vec1().iter().map(|&v| v as u8).collect();

        let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            image::ImageBuffer::from_raw(w as u32, h as u32, mask_pixels)
                .ok_or(SegmentAnythingInferenceError::MergeMasks)?;

        Ok(image::DynamicImage::from(mask_img).resize_to_fill(
            image_width,
            image_height,
            image::imageops::FilterType::CatmullRom,
        ))
    }

    fn image_to_tensor(&self, image: DynamicImage) -> Tensor<3, f32, Concrete<f32, 3>> {
        let image = {
            let resize_longest = IMAGE_SIZE;
            let (height, width) = (image.height(), image.width());
            let resize_longest = resize_longest as u32;
            let (height, width) = if height < width {
                let h = (resize_longest * height) / width;
                (h, resize_longest)
            } else {
                let w = (resize_longest * width) / height;
                (resize_longest, w)
            };
            image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
        };
        let (height, width) = (image.height() as usize, image.width() as usize);
        let img = image.to_rgb8();
        let data = img.into_raw();
        // Convert u8 to f32
        let data_f32: Vec<f32> = data.iter().map(|&v| v as f32).collect();
        let device = &self.device;
        // (H, W, 3) -> permute to (3, H, W)
        let image: Tensor<3, f32, Concrete<f32, 3>> =
            Tensor::from_slice(device, [height, width, 3], &data_f32)
                .transpose(1, 2) // (H, 3, W)
                .transpose(0, 1) // (3, H, W)
                .to_concrete();
        image
    }

    /// Segment everything in an image. Returns a list of [`DynamicImage`] masks.
    ///
    /// # Example
    ///
    /// ```rust, no_run
    /// use segment_anything_rs::*;
    ///
    /// # async fn run() {
    /// let model = SegmentAnything::builder().build().await.unwrap();
    /// let image = image::open("examples/landscape.jpg").unwrap();
    /// let images = model.segment_everything(image).await.unwrap();
    /// for (i, img) in images.iter().enumerate() {
    ///     img.save(&format!("{}.png", i)).unwrap();
    /// }
    /// # }
    /// ```
    pub async fn segment_everything(
        &self,
        image: DynamicImage,
    ) -> Result<Vec<DynamicImage>, SegmentAnythingInferenceError> {
        let image_width = image.width();
        let image_height = image.height();
        let image_tensor = self.image_to_tensor(image);
        let original_h = image_tensor.shape()[1];
        let original_w = image_tensor.shape()[2];

        // Compute image embeddings once
        let img_embeddings = self.sam.embeddings(&image_tensor);

        // Build a 32x32 grid of points (1024 points)
        let point_grid = build_point_grid(32);

        let mut candidates: Vec<MaskCandidate> = Vec::new();

        for chunk in point_grid.chunks(SEGMENT_EVERYTHING_BATCH_SIZE) {
            let batch_points: Vec<(f64, f64, bool)> =
                chunk.iter().map(|&(px, py)| (px, py, true)).collect();

            let (low_res_masks, iou_preds) = self.sam.forward_for_embeddings_batched(
                &img_embeddings,
                original_h,
                original_w,
                &batch_points,
                true, // multimask_output: get 3 mask alternatives per point
            );

            // Read masks and IoU predictions to CPU in one shot
            let masks_shape = low_res_masks.shape(); // (batch, 3, h, w)
            let batch = masks_shape[0];
            let n_masks_per_point = masks_shape[1];
            let mask_h = masks_shape[2];
            let mask_w = masks_shape[3];
            let mask_pixels = mask_h * mask_w;
            let total_mask_elems = batch * n_masks_per_point * mask_pixels;

            // The low-res masks are at 1/4 of IMAGE_SIZE (256x256) and represent the
            // padded 1024x1024 image space. The actual image occupies only the top-left
            // (original_h, original_w) region of that 1024x1024 space. Compute the
            // corresponding crop region at the low-res mask scale.
            let crop_h = low_res_crop_extent(original_h, mask_h);
            let crop_w = low_res_crop_extent(original_w, mask_w);

            let masks_flat = low_res_masks.reshape([total_mask_elems]);
            let masks_slice = masks_flat.as_slice().await?;
            let masks_vec = masks_slice.to_vec1();

            let total_iou_elems = batch * n_masks_per_point;
            let iou_flat = iou_preds.reshape([total_iou_elems]);
            let iou_slice = iou_flat.as_slice().await?;
            let iou_vec = iou_slice.to_vec1();

            collect_mask_candidates(
                &masks_vec,
                &iou_vec,
                batch,
                n_masks_per_point,
                mask_w,
                crop_h,
                crop_w,
                &mut candidates,
            );
        }

        let candidates = non_maximum_suppression(candidates, CROP_NMS_THRESH);
        let mut masks = Vec::new();
        for candidate in candidates {
            let rgb_pixels = candidate.mask.to_rgb_pixels();
            let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                image::ImageBuffer::from_raw(candidate.w as u32, candidate.h as u32, rgb_pixels)
                    .ok_or(SegmentAnythingInferenceError::MergeMasks)?;

            let image = image::DynamicImage::from(mask_img).resize_exact(
                image_width,
                image_height,
                image::imageops::FilterType::Nearest,
            );
            masks.push(image);
        }

        Ok(masks)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoundingBox {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl BoundingBox {
    fn area(self) -> usize {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }
}

#[derive(Clone, Debug)]
struct BinaryMask {
    bits: Vec<u64>,
    len: usize,
}

impl BinaryMask {
    fn from_thresholded(mask: &[f32], width: usize, threshold: f32) -> Option<(Self, BoundingBox)> {
        let mut bits = vec![0u64; mask.len().div_ceil(64)];
        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut has_foreground = false;

        for (idx, &value) in mask.iter().enumerate() {
            if value < threshold {
                continue;
            }

            has_foreground = true;
            bits[idx / 64] |= 1u64 << (idx % 64);

            let x = idx % width;
            let y = idx / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        has_foreground.then_some((
            Self {
                bits,
                len: mask.len(),
            },
            BoundingBox {
                x0: min_x,
                y0: min_y,
                x1: max_x + 1,
                y1: max_y + 1,
            },
        ))
    }

    fn bit(&self, idx: usize) -> bool {
        let word = self.bits[idx / 64];
        ((word >> (idx % 64)) & 1) != 0
    }

    fn to_rgb_pixels(&self) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(self.len * 3);
        for idx in 0..self.len {
            let value = if self.bit(idx) { 255u8 } else { 0u8 };
            pixels.extend_from_slice(&[value, value, value]);
        }
        pixels
    }
}

#[derive(Clone, Debug)]
struct MaskCandidate {
    mask: BinaryMask,
    bbox: BoundingBox,
    score: f32,
    h: usize,
    w: usize,
}

fn bbox_iou(a: BoundingBox, b: BoundingBox) -> f32 {
    let x0 = a.x0.max(b.x0);
    let y0 = a.y0.max(b.y0);
    let x1 = a.x1.min(b.x1);
    let y1 = a.y1.min(b.y1);

    if x0 >= x1 || y0 >= y1 {
        return 0.0;
    }

    let intersection = (x1 - x0) * (y1 - y0);
    let union = a.area() + b.area() - intersection;
    intersection as f32 / union as f32
}

fn non_maximum_suppression(
    mut candidates: Vec<MaskCandidate>,
    threshold: f32,
) -> Vec<MaskCandidate> {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<MaskCandidate> = Vec::with_capacity(candidates.len());
    'candidate: for candidate in candidates {
        for kept_candidate in &kept {
            if bbox_iou(candidate.bbox, kept_candidate.bbox) > threshold {
                continue 'candidate;
            }
        }
        kept.push(candidate);
    }

    kept
}

fn low_res_crop_extent(original_len: usize, low_res_len: usize) -> usize {
    (original_len * low_res_len)
        .div_ceil(IMAGE_SIZE)
        .min(low_res_len)
}

/// Filter a batch of low-res masks by IoU and stability score and push the survivors
/// into `candidates`. The masks in `masks_vec` are laid out as
/// `[batch, n_masks_per_point, mask_h, mask_w]` (row-major), but we only read the
/// top-left `crop_h` by `crop_w` region of each mask (the part that corresponds to the
/// actual image rather than the padded 1024 by 1024 input).
#[allow(clippy::too_many_arguments)]
fn collect_mask_candidates(
    masks_vec: &[f32],
    iou_vec: &[f32],
    batch: usize,
    n_masks_per_point: usize,
    mask_w: usize,
    crop_h: usize,
    crop_w: usize,
    candidates: &mut Vec<MaskCandidate>,
) {
    let mask_pixels = masks_vec.len() / (batch * n_masks_per_point);
    let hi_thresh = MODEL_MASK_THRESHOLD + STABILITY_SCORE_OFFSET;
    let lo_thresh = MODEL_MASK_THRESHOLD - STABILITY_SCORE_OFFSET;

    for point_idx in 0..batch {
        for mask_idx in 0..n_masks_per_point {
            let flat_idx = point_idx * n_masks_per_point + mask_idx;
            let iou = iou_vec[flat_idx];
            if iou < PRED_IOU_THRESH {
                continue;
            }

            let mask_start = flat_idx * mask_pixels;
            let mut cropped_mask = Vec::with_capacity(crop_h * crop_w);
            for y in 0..crop_h {
                let row_start = mask_start + y * mask_w;
                cropped_mask.extend_from_slice(&masks_vec[row_start..row_start + crop_w]);
            }

            let intersections = cropped_mask.iter().filter(|&&v| v >= hi_thresh).count() as f32;
            let unions = cropped_mask.iter().filter(|&&v| v >= lo_thresh).count() as f32;
            let stability_score = if unions > 0.0 {
                intersections / unions
            } else {
                0.0
            };
            if stability_score < STABILITY_SCORE_THRESHOLD {
                continue;
            }

            if let Some((mask, bbox)) =
                BinaryMask::from_thresholded(&cropped_mask, crop_w, MODEL_MASK_THRESHOLD)
            {
                candidates.push(MaskCandidate {
                    mask,
                    bbox,
                    score: iou,
                    h: crop_h,
                    w: crop_w,
                });
            }
        }
    }
}
