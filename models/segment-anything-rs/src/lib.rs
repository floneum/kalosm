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

#[cfg(test)]
pub(crate) async fn gpu_device_for_test() -> Option<Device> {
    match Device::new().await {
        Ok(device) => Some(device),
        Err(err) => {
            eprintln!("skipping GPU-only test: {err}");
            None
        }
    }
}

use fusor::{ConcreteTensor, Device, Tensor, ToVec1, VarBuilder};
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgba};
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
    /// contains — the source's `tiny` flag controls whether the loader uses
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
    /// Renamed from `add_goal_point` in 0.5 to flag the absolute→normalized
    /// coordinate switch — old pixel-based callers should divide by image size.
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
    /// Renamed from `add_avoid_points` in 0.5 to flag the absolute→normalized
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
    DownloadModel(#[from] hf_hub::api::sync::ApiError),
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
                let api = hf_hub::api::sync::Api::new()?;
                let api = api.model(source.model.clone());
                api.get(&source.filename)?
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

    fn image_to_tensor(&self, image: DynamicImage) -> Tensor<3, f32, ConcreteTensor<f32, 3>> {
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
        let image: Tensor<3, f32, ConcreteTensor<f32, 3>> =
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
/// top-left `crop_h × crop_w` region of each mask (the part that corresponds to the
/// actual image rather than the padded 1024×1024 input).
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn fetch_tiny_gguf_path() -> PathBuf {
        let source = SegmentAnythingSource::tiny();
        let api = hf_hub::api::sync::Api::new().expect("create hf api");
        api.model(source.model)
            .get(&source.filename)
            .expect("download tiny SAM gguf")
    }

    async fn f_to_vec<const R: usize>(t: &Tensor<R, f32>) -> Vec<f32> {
        let shape = t.shape();
        let n: usize = shape.iter().product();
        let ones: Tensor<R, f32> = Tensor::from_slice(&t.device(), shape, &vec![1.0f32; n]);
        let materialized = t * ones;
        let flat = materialized.reshape([n]);
        let s = flat.as_slice().await.unwrap();
        s.to_vec1()
    }

    fn load_tiny_sam(device: &Device, gguf_path: &Path) -> raw::sam::Sam {
        let mut reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut vb = VarBuilder::from_gguf(&mut reader).unwrap();
        raw::sam::Sam::load_tiny(device, &mut vb).unwrap()
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    /// End-to-end smoke test: load model, run inference, verify mask is non-trivial.
    #[tokio::test]
    async fn test_load_tiny_model() {
        let Some(device) = crate::gpu_device_for_test().await else {
            return;
        };
        let model = SegmentAnything::builder()
            .device(device)
            .build()
            .await
            .expect("Failed to load model");
        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).expect("Failed to open test image");
        let (w, h) = (image.width(), image.height());

        let settings =
            SegmentAnythingInferenceSettings::new(image).add_goal_point_normalized(0.5, 0.25);

        let mask = model
            .segment_from_points(settings)
            .await
            .expect("Failed to run inference");

        assert_eq!(mask.width(), w);
        assert_eq!(mask.height(), h);

        let mask_rgb = mask.to_rgb8();
        let pixels: &[u8] = mask_rgb.as_raw();
        let total = pixels.len();
        let white_count = pixels.iter().filter(|&&v| v == 255).count();
        let black_count = pixels.iter().filter(|&&v| v == 0).count();
        let white_frac = white_count as f64 / total as f64;
        let black_frac = black_count as f64 / total as f64;
        assert!(white_frac > 0.01, "Mask is all black");
        assert!(black_frac > 0.01, "Mask is all white");
    }

    #[tokio::test]
    async fn test_load_tiny_model_cpu_runtime() {
        let model = SegmentAnything::builder()
            .device(Device::cpu())
            .build()
            .await
            .expect("Failed to load model on CPU");

        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).expect("Failed to open test image");

        let mask = model
            .segment_from_points(
                SegmentAnythingInferenceSettings::new(image).add_goal_point_normalized(0.5, 0.25),
            )
            .await
            .expect("Failed to run CPU inference");

        assert!(mask.width() > 0);
        assert!(mask.height() > 0);
        assert!(model.device.is_cpu());
    }

    /// CPU vs GPU mask decoder: dense PE, prompt encoder, transformer, masks, IoU.
    #[tokio::test]
    async fn test_mask_decoder_cpu_vs_gpu() {
        use fusor::{Device, Tensor};

        let gguf_path = fetch_tiny_gguf_path();

        let cpu = Device::cpu();
        let Some(gpu) = crate::gpu_device_for_test().await else {
            return;
        };
        let cpu_sam = load_tiny_sam(&cpu, &gguf_path);
        let gpu_sam = load_tiny_sam(&gpu, &gguf_path);

        // Dense PE
        let cpu_pe = cpu_sam.prompt_encoder.get_dense_pe();
        let gpu_pe = gpu_sam.prompt_encoder.get_dense_pe();
        let pe_diff = max_diff(&f_to_vec(&cpu_pe).await, &f_to_vec(&gpu_pe).await);
        assert!(pe_diff < 0.001, "dense PE diverged: {}", pe_diff);

        // Prompt encoder — scale normalized points by the padded SAM input size for x,
        // and by the landscape.jpg test-image height for y.
        let points = [(0.5, 0.25, true)];
        const TEST_IMAGE_HEIGHT: f32 = 771.0;
        let xys: Vec<f32> = points
            .iter()
            .flat_map(|(x, y, _)| {
                [
                    (*x as f32) * IMAGE_SIZE as f32,
                    (*y as f32) * TEST_IMAGE_HEIGHT,
                ]
            })
            .collect();
        let labels: Vec<f32> = points
            .iter()
            .map(|(_, _, b)| if *b { 1f32 } else { 0f32 })
            .collect();
        let cpu_pts: Tensor<3, f32> = Tensor::from_slice(&cpu, [1, 1, 2], &xys);
        let cpu_lbls: Tensor<2, f32> = Tensor::from_slice(&cpu, [1, 1], &labels);
        let gpu_pts: Tensor<3, f32> = Tensor::from_slice(&gpu, [1, 1, 2], &xys);
        let gpu_lbls: Tensor<2, f32> = Tensor::from_slice(&gpu, [1, 1], &labels);
        let (cpu_sparse, cpu_dense) =
            cpu_sam
                .prompt_encoder
                .forward(Some((&cpu_pts, &cpu_lbls)), None, None);
        let (gpu_sparse, gpu_dense) =
            gpu_sam
                .prompt_encoder
                .forward(Some((&gpu_pts, &gpu_lbls)), None, None);
        assert!(
            max_diff(
                &(f_to_vec(&cpu_sparse).await),
                &(f_to_vec(&gpu_sparse).await)
            ) < 0.001,
            "sparse prompt diverged"
        );

        // Mask decoder with synthetic embeddings
        let emb_data: Vec<f32> = (0..256 * 64 * 64)
            .map(|i| ((i as f32) * 0.001).sin() * 0.1)
            .collect();
        let cpu_emb: Tensor<4, f32> = Tensor::from_slice(&cpu, [1, 256, 64, 64], &emb_data);
        let gpu_emb: Tensor<4, f32> = Tensor::from_slice(&gpu, [1, 256, 64, 64], &emb_data);

        let (cpu_masks, cpu_iou) =
            cpu_sam
                .mask_decoder
                .forward(&cpu_emb, &cpu_pe, &cpu_sparse, &cpu_dense, false);
        let (gpu_masks, gpu_iou) =
            gpu_sam
                .mask_decoder
                .forward(&gpu_emb, &gpu_pe, &gpu_sparse, &gpu_dense, false);
        assert!(
            max_diff(&f_to_vec(&cpu_masks).await, &f_to_vec(&gpu_masks).await) < 0.01,
            "mask output diverged"
        );
        assert!(
            max_diff(&f_to_vec(&cpu_iou).await, &f_to_vec(&gpu_iou).await) < 0.01,
            "IoU prediction diverged"
        );
    }

    /// Regression test for the shared-node `sin`/`cos` path used by `pe_encoding`.
    ///
    /// When a single lazy graph node feeds into two consumers (sin() and cos()),
    /// both GPU consumers must still see the correct values.
    /// This reproduces the exact chain from PositionEmbeddingRandom::pe_encoding().
    #[tokio::test]
    async fn test_dual_consumer_gpu_bug() {
        use fusor::{ConcreteTensor, Device, Tensor};

        let gguf_path = fetch_tiny_gguf_path();

        let cpu = Device::cpu();
        let Some(gpu) = crate::gpu_device_for_test().await else {
            return;
        };

        // Load full model to warm up GPU buffer pool and get GGUF gaussian matrix.
        let cpu_sam = load_tiny_sam(&cpu, &gguf_path);
        let gpu_sam = load_tiny_sam(&gpu, &gguf_path);

        let h = 64usize;
        let w = 64usize;

        fn build_pe_shared(
            device: &Device,
            gm: &Tensor<2, f32, ConcreteTensor<f32, 2>>,
            h: usize,
            w: usize,
        ) -> Tensor<3, f32> {
            let x: Tensor<1, f32> =
                fusor::arange_step::<f32>(device, 0.5, w as f32 + 0.5, 1.0).div_scalar(w as f32);
            let y: Tensor<1, f32> =
                fusor::arange_step::<f32>(device, 0.5, h as f32 + 0.5, 1.0).div_scalar(h as f32);
            let x_2d = x.reshape([1, w]);
            let x_broadcast = x_2d.broadcast_as([h, w]);
            let xu = x_broadcast.reshape([h, w, 1]);
            let y_2d = y.reshape([h, 1]);
            let y_broadcast = y_2d.broadcast_as([h, w]);
            let yu = y_broadcast.reshape([h, w, 1]);
            let coords = Tensor::cat([xu, yu], 2).mul_scalar(2.0) + (-1.0f32);
            let gm_shape = gm.shape();
            let gm_broadcast = gm.reshape([1, gm_shape[0], gm_shape[1]]);
            let gm_broadcast = gm_broadcast.broadcast_as([h, gm_shape[0], gm_shape[1]]);
            let mm = coords.mat_mul(&gm_broadcast);
            let scaled = mm.mul_scalar(2.0 * std::f32::consts::PI);
            // Dual consumer: sin() and cos() both read from `scaled`
            Tensor::cat([scaled.sin().to_concrete(), scaled.cos().to_concrete()], 2)
        }

        fn build_pe_separate(
            device: &Device,
            gm: &Tensor<2, f32, ConcreteTensor<f32, 2>>,
            h: usize,
            w: usize,
        ) -> Tensor<3, f32> {
            let x: Tensor<1, f32> =
                fusor::arange_step::<f32>(device, 0.5, w as f32 + 0.5, 1.0).div_scalar(w as f32);
            let y: Tensor<1, f32> =
                fusor::arange_step::<f32>(device, 0.5, h as f32 + 0.5, 1.0).div_scalar(h as f32);
            let x_2d = x.reshape([1, w]);
            let x_broadcast = x_2d.broadcast_as([h, w]);
            let xu = x_broadcast.reshape([h, w, 1]);
            let y_2d = y.reshape([h, 1]);
            let y_broadcast = y_2d.broadcast_as([h, w]);
            let yu = y_broadcast.reshape([h, w, 1]);
            let coords = Tensor::cat([xu, yu], 2).mul_scalar(2.0) + (-1.0f32);
            let gm_shape = gm.shape();
            let gm_broadcast = gm.reshape([1, gm_shape[0], gm_shape[1]]);
            let gm_broadcast = gm_broadcast.broadcast_as([h, gm_shape[0], gm_shape[1]]);
            let mm = coords.mat_mul(&gm_broadcast);
            // Control path: force distinct mul_scalar nodes for comparison.
            let for_sin = mm.mul_scalar(2.0 * std::f32::consts::PI);
            let for_cos = mm.mul_scalar(2.0 * std::f32::consts::PI);
            Tensor::cat(
                [for_sin.sin().to_concrete(), for_cos.cos().to_concrete()],
                2,
            )
        }

        let cpu_gm = &cpu_sam
            .prompt_encoder
            .pe_layer
            .positional_encoding_gaussian_matrix;
        let gpu_gm = &gpu_sam
            .prompt_encoder
            .pe_layer
            .positional_encoding_gaussian_matrix;

        let cpu_result = build_pe_shared(&cpu, cpu_gm, h, w);
        let gpu_shared = build_pe_shared(&gpu, gpu_gm, h, w);
        let gpu_separate = build_pe_separate(&gpu, gpu_gm, h, w);

        let cpu_result_vec = f_to_vec(&cpu_result).await;
        let gpu_shared_vec = f_to_vec(&gpu_shared).await;
        let gpu_separate_vec = f_to_vec(&gpu_separate).await;

        let diff_separate = max_diff(&cpu_result_vec, &gpu_separate_vec);
        assert!(
            diff_separate < 0.001,
            "separate consumers diverged (unexpected): {}",
            diff_separate
        );

        let diff_shared = max_diff(&cpu_result_vec, &gpu_shared_vec);
        assert!(
            diff_shared < 0.01,
            "shared consumers diverged: {}",
            diff_shared
        );
    }

    #[tokio::test]
    async fn test_position_encoding_intermediates_cpu_vs_gpu() {
        use fusor::{Device, Tensor};

        let gguf_path = fetch_tiny_gguf_path();

        let cpu = Device::cpu();
        let Some(gpu) = crate::gpu_device_for_test().await else {
            return;
        };

        let cpu_sam = load_tiny_sam(&cpu, &gguf_path);
        let gpu_sam = load_tiny_sam(&gpu, &gguf_path);

        let h = 64usize;
        let w = 64usize;

        let cpu_gm = &cpu_sam
            .prompt_encoder
            .pe_layer
            .positional_encoding_gaussian_matrix;
        let gpu_gm = &gpu_sam
            .prompt_encoder
            .pe_layer
            .positional_encoding_gaussian_matrix;

        let gm_diff = max_diff(&f_to_vec(cpu_gm).await, &f_to_vec(gpu_gm).await);
        assert!(gm_diff < 0.001, "gaussian matrix diverged: {}", gm_diff);

        let cpu_x: Tensor<1, f32> =
            fusor::arange_step::<f32>(&cpu, 0.5, w as f32 + 0.5, 1.0).div_scalar(w as f32);
        let gpu_x: Tensor<1, f32> =
            fusor::arange_step::<f32>(&gpu, 0.5, w as f32 + 0.5, 1.0).div_scalar(w as f32);
        let x_diff = max_diff(&f_to_vec(&cpu_x).await, &f_to_vec(&gpu_x).await);
        assert!(x_diff < 0.001, "x grid diverged: {}", x_diff);

        let cpu_y: Tensor<1, f32> =
            fusor::arange_step::<f32>(&cpu, 0.5, h as f32 + 0.5, 1.0).div_scalar(h as f32);
        let gpu_y: Tensor<1, f32> =
            fusor::arange_step::<f32>(&gpu, 0.5, h as f32 + 0.5, 1.0).div_scalar(h as f32);
        let y_diff = max_diff(&f_to_vec(&cpu_y).await, &f_to_vec(&gpu_y).await);
        assert!(y_diff < 0.001, "y grid diverged: {}", y_diff);

        let cpu_x_2d = cpu_x.reshape([1, w]);
        let cpu_x_broadcast = cpu_x_2d.broadcast_as([h, w]);
        let cpu_x = cpu_x_broadcast.reshape([h, w, 1]);
        let gpu_x_2d = gpu_x.reshape([1, w]);
        let gpu_x_broadcast = gpu_x_2d.broadcast_as([h, w]);
        let gpu_x = gpu_x_broadcast.reshape([h, w, 1]);
        let cpu_y_2d = cpu_y.reshape([h, 1]);
        let cpu_y_broadcast = cpu_y_2d.broadcast_as([h, w]);
        let cpu_y = cpu_y_broadcast.reshape([h, w, 1]);
        let gpu_y_2d = gpu_y.reshape([h, 1]);
        let gpu_y_broadcast = gpu_y_2d.broadcast_as([h, w]);
        let gpu_y = gpu_y_broadcast.reshape([h, w, 1]);
        let cpu_coords: Tensor<3, f32> = Tensor::cat([cpu_x, cpu_y], 2);
        let gpu_coords: Tensor<3, f32> = Tensor::cat([gpu_x, gpu_y], 2);

        let coords_diff = max_diff(&f_to_vec(&cpu_coords).await, &f_to_vec(&gpu_coords).await);
        assert!(coords_diff < 0.001, "coords diverged: {}", coords_diff);

        let cpu_coords = (cpu_coords.mul_scalar(2.0) + (-1.0f32)).to_concrete();
        let gpu_coords = (gpu_coords.mul_scalar(2.0) + (-1.0f32)).to_concrete();
        let centered_diff = max_diff(&f_to_vec(&cpu_coords).await, &f_to_vec(&gpu_coords).await);
        assert!(
            centered_diff < 0.001,
            "centered coords diverged: {}",
            centered_diff
        );

        let cpu_gm_shape = cpu_gm.shape();
        let gpu_gm_shape = gpu_gm.shape();
        let cpu_gm_broadcast = cpu_gm.reshape([1, cpu_gm_shape[0], cpu_gm_shape[1]]);
        let cpu_gm_broadcast = cpu_gm_broadcast.broadcast_as([h, cpu_gm_shape[0], cpu_gm_shape[1]]);
        let gpu_gm_broadcast = gpu_gm.reshape([1, gpu_gm_shape[0], gpu_gm_shape[1]]);
        let gpu_gm_broadcast = gpu_gm_broadcast.broadcast_as([h, gpu_gm_shape[0], gpu_gm_shape[1]]);
        let cpu_mm = cpu_coords.mat_mul(&cpu_gm_broadcast).to_concrete();
        let gpu_mm = gpu_coords.mat_mul(&gpu_gm_broadcast).to_concrete();
        let mm_diff = max_diff(&f_to_vec(&cpu_mm).await, &f_to_vec(&gpu_mm).await);
        assert!(mm_diff < 0.001, "matmul diverged: {}", mm_diff);

        let cpu_scaled = cpu_mm.mul_scalar(2.0 * std::f32::consts::PI);
        let gpu_scaled = gpu_mm.mul_scalar(2.0 * std::f32::consts::PI);
        let scaled_diff = max_diff(&f_to_vec(&cpu_scaled).await, &f_to_vec(&gpu_scaled).await);
        assert!(scaled_diff < 0.001, "scaled diverged: {}", scaled_diff);

        let cpu_sin = cpu_scaled.sin().to_concrete();
        let gpu_sin = gpu_scaled.sin().to_concrete();
        let sin_diff = max_diff(&f_to_vec(&cpu_sin).await, &f_to_vec(&gpu_sin).await);
        assert!(sin_diff < 0.001, "sin diverged: {}", sin_diff);

        let cpu_cos = cpu_scaled.cos().to_concrete();
        let gpu_cos = gpu_scaled.cos().to_concrete();
        let cos_diff = max_diff(&f_to_vec(&cpu_cos).await, &f_to_vec(&gpu_cos).await);
        assert!(cos_diff < 0.001, "cos diverged: {}", cos_diff);
    }

    /// Compare batched vs unbatched forward_for_embeddings to verify numerical equivalence.
    /// Uses the exact same reshape+as_slice reading pattern as segment_everything.
    #[tokio::test]
    async fn test_batched_vs_unbatched() {
        let Some(device) = crate::gpu_device_for_test().await else {
            return;
        };
        let model = SegmentAnything::builder()
            .device(device)
            .build()
            .await
            .expect("Failed to load model");

        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).expect("Failed to open test image");
        let image_tensor = model.image_to_tensor(image);
        let original_h = image_tensor.shape()[1];
        let original_w = image_tensor.shape()[2];

        let img_embeddings = model.sam.embeddings(&image_tensor);

        // Use 8 test points (more than 4 to stress-test)
        let test_points: Vec<(f64, f64, bool)> = vec![
            (0.25, 0.25, true),
            (0.5, 0.25, true),
            (0.75, 0.5, true),
            (0.5, 0.75, true),
            (0.1, 0.1, true),
            (0.9, 0.9, true),
            (0.3, 0.7, true),
            (0.7, 0.3, true),
        ];

        // --- Batched: all points at once, read via reshape+as_slice (same as segment_everything) ---
        let (batched_masks, batched_iou) = model.sam.forward_for_embeddings_batched(
            &img_embeddings,
            original_h,
            original_w,
            &test_points,
            true,
        );
        let bm_shape = batched_masks.shape(); // (batch, 3, h, w)
        let batch = bm_shape[0];
        let n_masks = bm_shape[1];
        let mask_h = bm_shape[2];
        let mask_w = bm_shape[3];
        let mask_pixels = mask_h * mask_w;

        // Read masks via reshape+as_slice (the pattern used in segment_everything)
        let total_mask = batch * n_masks * mask_pixels;
        let masks_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> =
            batched_masks.reshape([total_mask]).to_concrete();
        let masks_slice = masks_flat.as_slice().await.unwrap();
        let batched_masks_data = masks_slice.to_vec1();

        let total_iou = batch * n_masks;
        let iou_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> =
            batched_iou.reshape([total_iou]).to_concrete();
        let iou_slice = iou_flat.as_slice().await.unwrap();
        let batched_iou_data = iou_slice.to_vec1();

        // --- Unbatched: one point at a time, read via reshape+as_slice ---
        let mut unbatched_masks_data = Vec::new();
        let mut unbatched_iou_data = Vec::new();
        for point in &test_points {
            let (masks, iou) = model.sam.forward_for_embeddings(
                &img_embeddings,
                original_h,
                original_w,
                &[*point],
                true,
            );
            let ms = masks.shape();
            let m_total = ms[0] * ms[1] * ms[2] * ms[3];
            let mf: Tensor<1, f32, ConcreteTensor<f32, 1>> = masks.reshape([m_total]).to_concrete();
            let ms_data = mf.as_slice().await.unwrap();
            unbatched_masks_data.extend(ms_data.to_vec1());

            let is = iou.shape();
            let i_total = is[0] * is[1];
            let if_: Tensor<1, f32, ConcreteTensor<f32, 1>> = iou.reshape([i_total]).to_concrete();
            let is_data = if_.as_slice().await.unwrap();
            unbatched_iou_data.extend(is_data.to_vec1());
        }

        let mask_diff = max_diff(&batched_masks_data, &unbatched_masks_data);
        let iou_diff = max_diff(&batched_iou_data, &unbatched_iou_data);

        assert!(
            mask_diff < 0.01,
            "Batched vs unbatched mask divergence too large: {mask_diff}"
        );
        assert!(
            iou_diff < 0.01,
            "Batched vs unbatched IoU divergence too large: {iou_diff}"
        );
    }

    #[test]
    fn test_build_point_grid_matches_expected_layout() {
        let grid = build_point_grid(4);
        assert_eq!(grid.len(), 16);

        // Offset is 1 / (2 * 4) = 0.125, then step by 1/4 = 0.25.
        let coord = |i: usize| 0.125 + i as f64 * 0.25;
        // Implementation iterates `i_x` in the outer loop and `i_y` inner,
        // so points are emitted column-major (all y for x=0, then x=1, …).
        let expected: Vec<(f64, f64)> = (0..4)
            .flat_map(|ix| (0..4).map(move |iy| (coord(ix), coord(iy))))
            .collect();

        for (got, want) in grid.iter().zip(expected.iter()) {
            assert!(
                (got.0 - want.0).abs() < 1e-9 && (got.1 - want.1).abs() < 1e-9,
                "got {got:?} want {want:?}",
            );
        }

        // Every coordinate should fall inside (0, 1).
        for &(x, y) in &grid {
            assert!(x > 0.0 && x < 1.0, "x out of range: {x}");
            assert!(y > 0.0 && y < 1.0, "y out of range: {y}");
        }
    }

    #[test]
    fn test_nms_handles_empty_input() {
        let kept = non_maximum_suppression(Vec::new(), 0.5);
        assert!(kept.is_empty());
    }

    #[test]
    fn test_low_res_crop_extent_uses_ceil_coverage() {
        assert_eq!(low_res_crop_extent(771, 256), 193);
        assert_eq!(low_res_crop_extent(769, 256), 193);
        assert_eq!(low_res_crop_extent(1024, 256), 256);
        assert_eq!(low_res_crop_extent(1, 256), 1);
    }

    #[test]
    fn test_binary_mask_bbox_tracks_foreground_extent() {
        let mask = vec![
            -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
        ];
        let (binary, bbox) = BinaryMask::from_thresholded(&mask, 4, 0.0).unwrap();

        assert_eq!(
            bbox,
            BoundingBox {
                x0: 1,
                y0: 1,
                x1: 3,
                y1: 3,
            }
        );
        assert!(binary.bit(5));
        assert!(binary.bit(6));
        assert!(binary.bit(10));
        assert!(!binary.bit(0));
        assert!(!binary.bit(11));
    }

    #[test]
    fn test_bbox_nms_prefers_highest_scoring_candidate() {
        let candidate = |bbox, score| MaskCandidate {
            mask: BinaryMask {
                bits: vec![1],
                len: 1,
            },
            bbox,
            score,
            h: 1,
            w: 1,
        };

        let kept = non_maximum_suppression(
            vec![
                candidate(
                    BoundingBox {
                        x0: 0,
                        y0: 0,
                        x1: 10,
                        y1: 10,
                    },
                    0.95,
                ),
                candidate(
                    BoundingBox {
                        x0: 0,
                        y0: 0,
                        x1: 10,
                        y1: 10,
                    },
                    0.80,
                ),
                candidate(
                    BoundingBox {
                        x0: 20,
                        y0: 20,
                        x1: 30,
                        y1: 30,
                    },
                    0.70,
                ),
            ],
            0.7,
        );

        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].score, 0.95);
        assert_eq!(kept[1].score, 0.70);
    }
}
