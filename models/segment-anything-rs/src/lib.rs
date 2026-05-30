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

mod mask_generation;
mod raw;

use fusor::{Concrete, Device, Tensor, ToVec1, VarBuilder};
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgba};
use kalosm_model_types::FileSource;
use mask_generation::LowResMaskBatch;
use raw::sam::{build_point_grid, Sam, IMAGE_SIZE};

/// Number of prompt-grid points per image side in `segment_everything`.
const SEGMENT_EVERYTHING_POINTS_PER_SIDE: usize = 32;
/// Number of grid points evaluated per forward pass in `segment_everything`.
const SEGMENT_EVERYTHING_BATCH_SIZE: usize = 64;

/// A builder for [`SegmentAnything`].
#[derive(Default)]
pub struct SegmentAnythingBuilder {
    source: SegmentAnythingSource,
    local_path: Option<std::path::PathBuf>,
}

impl SegmentAnythingBuilder {
    /// Sets the source of the model.
    pub fn source(mut self, source: SegmentAnythingSource) -> Self {
        self.source = source;
        self
    }

    /// Load weights from a local GGUF file instead of downloading from
    /// Hugging Face. Pair with [`SegmentAnythingBuilder::source`] (or rely on
    /// the default `tiny` source) to indicate which architecture the file
    /// contains.
    pub fn gguf_path(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.local_path = Some(path.into());
        self
    }

    /// Builds the [`SegmentAnything`] model.
    pub async fn build(self) -> Result<SegmentAnything, LoadSegmentAnythingError> {
        SegmentAnything::new(self).await
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentAnythingArchitecture {
    MobileSamTiny,
    SamVitB,
}

/// The source of the model.
pub struct SegmentAnythingSource {
    model: String,
    filename: String,
    architecture: SegmentAnythingArchitecture,
}

impl SegmentAnythingSource {
    /// Creates a new [`SegmentAnythingSource`] for a SAM ViT-B checkpoint.
    pub fn new(model: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            filename: filename.into(),
            architecture: SegmentAnythingArchitecture::SamVitB,
        }
    }

    /// Creates a new [`SegmentAnythingSource`] for a MobileSAM TinyViT checkpoint.
    pub fn mobile_sam_tiny(model: impl Into<String>, filename: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            filename: filename.into(),
            architecture: SegmentAnythingArchitecture::MobileSamTiny,
        }
    }

    /// Create the tiny SAM model source.
    pub fn tiny() -> Self {
        Self::mobile_sam_tiny("Demonthos/MobileSamGguf", "mobile_sam-tiny-vitt.gguf")
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
        let SegmentAnythingBuilder { source, local_path } = settings;
        let model_path = match local_path {
            Some(path) => path,
            None => {
                let source =
                    FileSource::huggingface(source.model, "main".to_string(), source.filename);
                kalosm_common::Cache::default().get(&source, |_| {}).await?
            }
        };
        let device = Device::auto().await;
        let mut reader = std::io::BufReader::new(std::fs::File::open(&model_path)?);
        let mut vb = VarBuilder::from_gguf(&mut reader)
            .map_err(|e| fusor::Error::msg(format!("Failed to read GGUF: {e}")))?;
        let sam = match source.architecture {
            SegmentAnythingArchitecture::MobileSamTiny => Sam::load_tiny(&device, &mut vb)?,
            SegmentAnythingArchitecture::SamVitB => Sam::load_vit_b(&device, &mut vb)?,
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

        let point_grid = build_point_grid(SEGMENT_EVERYTHING_POINTS_PER_SIDE);

        let mut candidates = Vec::new();

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
            let crop_h = mask_generation::low_res_crop_extent(original_h, mask_h);
            let crop_w = mask_generation::low_res_crop_extent(original_w, mask_w);

            let masks_flat = low_res_masks.reshape([total_mask_elems]);
            let masks_slice = masks_flat.as_slice().await?;
            let masks_vec = masks_slice.to_vec1();

            let total_iou_elems = batch * n_masks_per_point;
            let iou_flat = iou_preds.reshape([total_iou_elems]);
            let iou_slice = iou_flat.as_slice().await?;
            let iou_vec = iou_slice.to_vec1();

            mask_generation::collect_mask_candidates(
                LowResMaskBatch {
                    masks: &masks_vec,
                    iou: &iou_vec,
                    batch,
                    masks_per_point: n_masks_per_point,
                    mask_w,
                    crop_h,
                    crop_w,
                },
                &mut candidates,
            );
        }

        let candidates = mask_generation::suppress_overlaps(candidates);
        let mut masks = Vec::new();
        for candidate in candidates {
            let mask_img = candidate
                .into_image()
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
