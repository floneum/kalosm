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

use fusor::{ConcreteTensor, Device, Tensor, VarBuilder};
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgba};
use raw::sam::{
    build_point_grid, Sam, CROP_NMS_THRESH, IMAGE_SIZE, MODEL_MASK_THRESHOLD, PRED_IOU_THRESH,
    STABILITY_SCORE_OFFSET, STABILITY_SCORE_THRESHOLD,
};

/// A builder for [`SegmentAnything`].
#[derive(Default)]
pub struct SegmentAnythingBuilder {
    source: SegmentAnythingSource,
    device: Option<Device>,
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
    pub fn add_goal_point(mut self, x: impl Into<f64>, y: impl Into<f64>) -> Self {
        self.goal_points.push((x.into(), y.into()));
        self
    }

    /// Set the list of points to segment.
    pub fn set_goal_points(mut self, points: Vec<(f64, f64)>) -> Self {
        self.goal_points = points;
        self
    }

    /// Add a point to the list of points to avoid.
    pub fn add_avoid_points(mut self, x: impl Into<f64>, y: impl Into<f64>) -> Self {
        self.avoid_points.push((x.into(), y.into()));
        self
    }

    /// Set the list of points to avoid.
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
        let SegmentAnythingBuilder { source, device } = settings;
        let model_path = {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model(source.model);
            api.get(&source.filename)?
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
    ///     .segment_from_points(SegmentAnythingInferenceSettings::new(image).add_goal_point(0.5, 0.25))
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
        let mask_2d: Tensor<2, f32> = mask
            .narrow(0, 0, 1)
            .to_concrete()
            .narrow(1, 0, 1)
            .to_concrete()
            .reshape([h, w])
            .to_concrete();

        // Threshold: >= threshold -> 255, else 0
        let threshold_mask: Tensor<2, f32> = mask_2d.gt_scalar(threshold - 1e-6).to_concrete();

        let mask_u8: Tensor<2, f32> = threshold_mask.mul_scalar(255.0f32);

        // Expand to 3 channels: (H, W) -> (3, H, W)
        let mask_3ch: Tensor<3, f32> = mask_u8
            .reshape([1, h, w])
            .broadcast_as([3, h, w])
            .to_concrete();

        // Permute to (H, W, 3) and flatten
        let mask_hwc: Tensor<3, f32> = mask_3ch
            .transpose(0, 1) // (H, 3, W)
            .to_concrete()
            .transpose(1, 2) // (H, W, 3)
            .to_concrete();
        let mask_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> =
            mask_hwc.reshape([h * w * 3]).to_concrete();
        let mask_slice = mask_flat.as_slice().await?;
        let mask_pixels: Vec<u8> = mask_slice.as_slice().iter().map(|&v| v as u8).collect();

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
            Tensor::from_slice(&device, [height, width, 3], &data_f32)
                .transpose(1, 2) // (H, 3, W)
                .to_concrete()
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

        struct MaskCandidate {
            mask_data: Vec<f32>,
            iou: f32,
            h: usize,
            w: usize,
        }

        let mut candidates: Vec<MaskCandidate> = Vec::new();

        const BATCH_SIZE: usize = 64;

        for chunk in point_grid.chunks(BATCH_SIZE) {
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

            let masks_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> =
                low_res_masks.reshape([total_mask_elems]).to_concrete();
            let masks_slice = masks_flat.as_slice().await?;
            let masks_vec = masks_slice.as_slice();

            let total_iou_elems = batch * n_masks_per_point;
            let iou_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> =
                iou_preds.reshape([total_iou_elems]).to_concrete();
            let iou_slice = iou_flat.as_slice().await?;
            let iou_vec = iou_slice.as_slice();

            for point_idx in 0..batch {
                for mask_idx in 0..n_masks_per_point {
                    let flat_idx = point_idx * n_masks_per_point + mask_idx;
                    let iou = iou_vec[flat_idx];
                    if iou < PRED_IOU_THRESH {
                        continue;
                    }

                    let mask_start = flat_idx * mask_pixels;

                    // Crop the mask to the actual image region (remove padding)
                    let mut cropped_mask = Vec::with_capacity(crop_h * crop_w);
                    for y in 0..crop_h {
                        let row_start = mask_start + y * mask_w;
                        cropped_mask.extend_from_slice(&masks_vec[row_start..row_start + crop_w]);
                    }

                    // Compute stability score on the cropped mask
                    let hi_thresh = MODEL_MASK_THRESHOLD + STABILITY_SCORE_OFFSET;
                    let lo_thresh = MODEL_MASK_THRESHOLD - STABILITY_SCORE_OFFSET;
                    let intersections =
                        cropped_mask.iter().filter(|&&v| v >= hi_thresh).count() as f32;
                    let unions = cropped_mask.iter().filter(|&&v| v >= lo_thresh).count() as f32;
                    let stability_score = if unions > 0.0 {
                        intersections / unions
                    } else {
                        0.0
                    };
                    if stability_score < STABILITY_SCORE_THRESHOLD {
                        continue;
                    }

                    candidates.push(MaskCandidate {
                        mask_data: cropped_mask,
                        iou,
                        h: crop_h,
                        w: crop_w,
                    });
                }
            }
        }

        // NMS: sort by IoU descending, suppress overlapping masks
        candidates.sort_by(|a, b| {
            b.iou
                .partial_cmp(&a.iou)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut keep = vec![true; candidates.len()];
        for i in 0..candidates.len() {
            if !keep[i] {
                continue;
            }
            for j in (i + 1)..candidates.len() {
                if !keep[j] {
                    continue;
                }
                let iou = mask_iou(&candidates[i].mask_data, &candidates[j].mask_data);
                if iou > CROP_NMS_THRESH {
                    keep[j] = false;
                }
            }
        }

        // Convert surviving masks to images
        let mut masks = Vec::new();
        for (i, candidate) in candidates.iter().enumerate() {
            if !keep[i] {
                continue;
            }

            // Threshold mask at 0.0 -> binary 0/255
            let rgb_pixels: Vec<u8> = candidate
                .mask_data
                .iter()
                .flat_map(|&v| {
                    let p = if v >= MODEL_MASK_THRESHOLD {
                        255u8
                    } else {
                        0u8
                    };
                    [p, p, p]
                })
                .collect();

            let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                image::ImageBuffer::from_raw(candidate.w as u32, candidate.h as u32, rgb_pixels)
                    .ok_or(SegmentAnythingInferenceError::MergeMasks)?;

            let image = image::DynamicImage::from(mask_img).resize_exact(
                image_width,
                image_height,
                image::imageops::FilterType::CatmullRom,
            );
            masks.push(image);
        }

        Ok(masks)
    }

    /// Load from a local GGUF file path using fusor, preferring GPU but falling back to CPU.
    pub async fn from_gguf_path(
        path: &std::path::Path,
        tiny: bool,
    ) -> Result<Self, LoadSegmentAnythingError> {
        let device = Device::auto().await;
        let mut reader = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut vb = VarBuilder::from_gguf(&mut reader)
            .map_err(|e| fusor::Error::msg(format!("Failed to read GGUF: {e}")))?;
        let sam = if tiny {
            raw::sam::Sam::load_tiny(&device, &mut vb)?
        } else {
            raw::sam::Sam::load_vit_b(&device, &mut vb)?
        };
        Ok(Self { sam, device })
    }
}

/// Compute IoU (Intersection over Union) between two masks on CPU.
/// Masks are raw float values; pixels >= MODEL_MASK_THRESHOLD count as foreground.
fn mask_iou(a: &[f32], b: &[f32]) -> f32 {
    let mut intersection = 0u32;
    let mut area_a = 0u32;
    let mut area_b = 0u32;
    for (&av, &bv) in a.iter().zip(b.iter()) {
        let a_fg = av >= MODEL_MASK_THRESHOLD;
        let b_fg = bv >= MODEL_MASK_THRESHOLD;
        area_a += a_fg as u32;
        area_b += b_fg as u32;
        intersection += (a_fg && b_fg) as u32;
    }
    let union = area_a + area_b - intersection;
    if union > 0 {
        intersection as f32 / union as f32
    } else {
        0.0
    }
}

fn low_res_crop_extent(original_len: usize, low_res_len: usize) -> usize {
    (original_len * low_res_len)
        .div_ceil(IMAGE_SIZE)
        .min(low_res_len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn load_model_with_device(
        gguf_path: &Path,
        device: Device,
        tiny: bool,
    ) -> Result<SegmentAnything, LoadSegmentAnythingError> {
        let mut reader = std::io::BufReader::new(std::fs::File::open(gguf_path)?);
        let mut vb = VarBuilder::from_gguf(&mut reader)
            .map_err(|e| fusor::Error::msg(format!("Failed to read GGUF: {e}")))?;
        let sam = if tiny {
            raw::sam::Sam::load_tiny(&device, &mut vb)?
        } else {
            raw::sam::Sam::load_vit_b(&device, &mut vb)?
        };
        Ok(SegmentAnything { sam, device })
    }

    fn f_to_vec<const R: usize>(t: &Tensor<R, f32>) -> Vec<f32> {
        let shape = t.shape();
        let n: usize = shape.iter().product();
        let ones: Tensor<R, f32> = Tensor::from_slice(&t.device(), shape, &vec![1.0f32; n]);
        let materialized: Tensor<R, f32> = (t * ones).to_concrete();
        let flat: Tensor<1, f32, ConcreteTensor<f32, 1>> = materialized.reshape([n]).to_concrete();
        let s = pollster::block_on(flat.as_slice()).unwrap();
        s.as_slice().to_vec()
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
        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !gguf_path.exists() {
            return;
        }

        let model = SegmentAnything::from_gguf_path(gguf_path, true)
            .await
            .expect("Failed to load model");

        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).expect("Failed to open test image");
        let (w, h) = (image.width(), image.height());

        let settings = SegmentAnythingInferenceSettings::new(image).add_goal_point(0.5, 0.25);

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
        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !gguf_path.exists() {
            return;
        }

        let model = load_model_with_device(gguf_path, Device::cpu(), true)
            .expect("Failed to load model on CPU");

        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).expect("Failed to open test image");

        let mask = model
            .segment_from_points(
                SegmentAnythingInferenceSettings::new(image).add_goal_point(0.5, 0.25),
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
        use fusor::{Device, Tensor, VarBuilder};

        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !gguf_path.exists() {
            return;
        }

        let cpu = Device::cpu();
        let gpu = Device::new().await.unwrap();
        let mut cpu_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut cpu_vb = VarBuilder::from_gguf(&mut cpu_reader).unwrap();
        let cpu_sam = raw::sam::Sam::load_tiny(&cpu, &mut cpu_vb).unwrap();
        let mut gpu_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut gpu_vb = VarBuilder::from_gguf(&mut gpu_reader).unwrap();
        let gpu_sam = raw::sam::Sam::load_tiny(&gpu, &mut gpu_vb).unwrap();

        // Dense PE
        let cpu_pe = cpu_sam.prompt_encoder.get_dense_pe();
        let gpu_pe = gpu_sam.prompt_encoder.get_dense_pe();
        let pe_diff = max_diff(&f_to_vec(&cpu_pe), &f_to_vec(&gpu_pe));
        assert!(pe_diff < 0.001, "dense PE diverged: {}", pe_diff);

        // Prompt encoder
        let points = vec![(0.5, 0.25, true)];
        let xys: Vec<f32> = points
            .iter()
            .flat_map(|(x, y, _)| [(*x as f32) * 1024.0, (*y as f32) * 771.0])
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
            max_diff(&f_to_vec(&cpu_sparse), &f_to_vec(&gpu_sparse)) < 0.001,
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
            max_diff(&f_to_vec(&cpu_masks), &f_to_vec(&gpu_masks)) < 0.01,
            "mask output diverged"
        );
        assert!(
            max_diff(&f_to_vec(&cpu_iou), &f_to_vec(&gpu_iou)) < 0.01,
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

        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !gguf_path.exists() {
            return;
        }

        let cpu = Device::cpu();
        let gpu = Device::new().await.unwrap();

        // Load full model to warm up GPU buffer pool and get GGUF gaussian matrix.
        let mut cpu_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut cpu_vb = VarBuilder::from_gguf(&mut cpu_reader).unwrap();
        let cpu_sam = raw::sam::Sam::load_tiny(&cpu, &mut cpu_vb).unwrap();
        let mut gpu_reader = std::io::BufReader::new(std::fs::File::open(gguf_path).unwrap());
        let mut gpu_vb = VarBuilder::from_gguf(&mut gpu_reader).unwrap();
        let gpu_sam = raw::sam::Sam::load_tiny(&gpu, &mut gpu_vb).unwrap();

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
            let xu: Tensor<3, f32> = x
                .reshape([1, w])
                .broadcast_as([h, w])
                .to_concrete()
                .reshape([h, w, 1])
                .to_concrete();
            let yu: Tensor<3, f32> = y
                .reshape([h, 1])
                .broadcast_as([h, w])
                .to_concrete()
                .reshape([h, w, 1])
                .to_concrete();
            let coords: Tensor<3, f32> =
                (Tensor::cat([xu, yu], 2).mul_scalar(2.0) + (-1.0f32)).to_concrete();
            let gm_shape = gm.shape();
            let gm3: Tensor<3, f32> = gm
                .reshape([1, gm_shape[0], gm_shape[1]])
                .broadcast_as([h, gm_shape[0], gm_shape[1]])
                .to_concrete();
            let scaled = coords.mat_mul(&gm3).mul_scalar(2.0 * std::f32::consts::PI);
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
            let xu: Tensor<3, f32> = x
                .reshape([1, w])
                .broadcast_as([h, w])
                .to_concrete()
                .reshape([h, w, 1])
                .to_concrete();
            let yu: Tensor<3, f32> = y
                .reshape([h, 1])
                .broadcast_as([h, w])
                .to_concrete()
                .reshape([h, w, 1])
                .to_concrete();
            let coords: Tensor<3, f32> =
                (Tensor::cat([xu, yu], 2).mul_scalar(2.0) + (-1.0f32)).to_concrete();
            let gm_shape = gm.shape();
            let gm3: Tensor<3, f32> = gm
                .reshape([1, gm_shape[0], gm_shape[1]])
                .broadcast_as([h, gm_shape[0], gm_shape[1]])
                .to_concrete();
            let mm = coords.mat_mul(&gm3);
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

        let diff_separate = max_diff(&f_to_vec(&cpu_result), &f_to_vec(&gpu_separate));
        assert!(
            diff_separate < 0.001,
            "separate consumers diverged (unexpected): {}",
            diff_separate
        );

        let diff_shared = max_diff(&f_to_vec(&cpu_result), &f_to_vec(&gpu_shared));
        assert!(
            diff_shared < 0.01,
            "shared consumers diverged: {}",
            diff_shared
        );
    }

    /// Benchmark: fusor GPU vs candle CPU/Metal end-to-end inference.
    #[tokio::test]
    async fn bench_fusor_vs_candle() {
        use candle_core::{DType, Device as CDevice, Tensor as CTensor};
        use candle_nn::VarBuilder as CVarBuilder;
        use candle_transformers::models::segment_anything::sam::Sam as CSam;
        use std::time::Instant;

        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        let safetensors_path = std::path::Path::new(&std::env::var("HOME").unwrap())
            .join(".cache/huggingface/hub/models--lmz--candle-sam/snapshots/8b4cb7c743f3b3cb8afd212a86ae15b1bbfdac97/mobile_sam-tiny-vitt.safetensors");
        if !gguf_path.exists() || !safetensors_path.exists() {
            return;
        }

        let image_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/landscape.jpg");
        let image = image::open(&image_path).expect("open image");
        let points = vec![(0.5, 0.25, true)];
        let n_warmup = 2;
        let n_iters = 5;

        // --- Candle (CPU) ---
        let c_device = CDevice::Cpu;
        let c_vb = unsafe {
            CVarBuilder::from_mmaped_safetensors(&[&safetensors_path], DType::F32, &c_device)
                .expect("load candle vb")
        };
        let c_sam = CSam::new_tiny(c_vb).expect("load candle model");

        let c_image = {
            let resize_longest = 1024u32;
            let (h, w) = (image.height(), image.width());
            let (h, w) = if h < w {
                ((resize_longest * h) / w, resize_longest)
            } else {
                (resize_longest, (resize_longest * w) / h)
            };
            let img = image.resize_exact(w, h, image::imageops::FilterType::CatmullRom);
            let (h, w) = (img.height() as usize, img.width() as usize);
            let data = img.to_rgb8().into_raw();
            CTensor::from_vec(data, (h, w, 3), &c_device)
                .unwrap()
                .permute((2, 0, 1))
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
        };

        for _ in 0..n_warmup {
            let _ = c_sam.forward(&c_image, &points, false).unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..n_iters {
            let _ = c_sam.forward(&c_image, &points, false).unwrap();
        }
        let candle_per_iter = t0.elapsed() / n_iters as u32;

        // --- Fusor (GPU) ---
        let f_model = SegmentAnything::from_gguf_path(gguf_path, true)
            .await
            .expect("load fusor");
        let f_image_tensor = f_model.image_to_tensor(image.clone());

        for _ in 0..n_warmup {
            let (mask, _) = f_model.sam.forward(&f_image_tensor, &points, false);
            let n: usize = mask.shape().iter().product();
            let flat: fusor::Tensor<1, f32, fusor::ConcreteTensor<f32, 1>> =
                mask.reshape([n]).to_concrete();
            let _ = flat.as_slice().await.unwrap();
        }
        let t0 = Instant::now();
        for _ in 0..n_iters {
            let (mask, _) = f_model.sam.forward(&f_image_tensor, &points, false);
            let n: usize = mask.shape().iter().product();
            let flat: fusor::Tensor<1, f32, fusor::ConcreteTensor<f32, 1>> =
                mask.reshape([n]).to_concrete();
            let _ = flat.as_slice().await.unwrap();
        }
        let fusor_per_iter = t0.elapsed() / n_iters as u32;

        // --- Candle (Metal GPU) ---
        let cm_result = CDevice::new_metal(0).ok().and_then(|cm_device| {
            let cm_vb = unsafe {
                CVarBuilder::from_mmaped_safetensors(&[&safetensors_path], DType::F32, &cm_device)
                    .ok()?
            };
            let cm_sam = CSam::new_tiny(cm_vb).ok()?;
            let cm_image = c_image.to_device(&cm_device).ok()?;
            for _ in 0..n_warmup {
                cm_sam.forward(&cm_image, &points, false).ok()?;
            }
            let t0 = Instant::now();
            for _ in 0..n_iters {
                cm_sam.forward(&cm_image, &points, false).ok()?;
            }
            let per_iter = t0.elapsed() / n_iters as u32;

            let cm_emb = cm_sam.embeddings(&cm_image).ok()?;
            for _ in 0..n_warmup {
                cm_sam
                    .forward_for_embeddings(&cm_emb, 771, 1024, &points, false)
                    .ok()?;
            }
            let t0 = Instant::now();
            for _ in 0..20u32 {
                cm_sam
                    .forward_for_embeddings(&cm_emb, 771, 1024, &points, false)
                    .ok()?;
            }
            Some((per_iter, t0.elapsed() / 20))
        });

        eprintln!("=== MobileSAM end-to-end ({n_iters} iters) ===");
        eprintln!("  Candle (CPU):   {candle_per_iter:?}/iter");
        if let Some((cm, _)) = cm_result {
            eprintln!("  Candle (Metal): {cm:?}/iter");
        } else {
            eprintln!("  Candle (Metal): N/A");
        }
        eprintln!("  Fusor  (GPU):   {fusor_per_iter:?}/iter");
        eprintln!(
            "  Fusor vs Candle CPU: {:.2}x",
            candle_per_iter.as_secs_f64() / fusor_per_iter.as_secs_f64()
        );

        // Decoder-only benchmark
        let f_emb = f_model.sam.embeddings(&f_image_tensor);
        let c_emb = c_sam.embeddings(&c_image).unwrap();
        let n_dec = 20u32;

        for _ in 0..n_warmup {
            let (mask, _) = f_model
                .sam
                .forward_for_embeddings(&f_emb, 771, 1024, &points, false);
            let n: usize = mask.shape().iter().product();
            let flat: fusor::Tensor<1, f32, fusor::ConcreteTensor<f32, 1>> =
                mask.reshape([n]).to_concrete();
            let _ = flat.as_slice().await.unwrap();
        }
        for _ in 0..n_warmup {
            let _ = c_sam
                .forward_for_embeddings(&c_emb, 771, 1024, &points, false)
                .unwrap();
        }

        let t0 = Instant::now();
        for _ in 0..n_dec {
            let _ = c_sam
                .forward_for_embeddings(&c_emb, 771, 1024, &points, false)
                .unwrap();
        }
        let candle_dec = t0.elapsed() / n_dec;

        let t0 = Instant::now();
        for _ in 0..n_dec {
            let (mask, _) = f_model
                .sam
                .forward_for_embeddings(&f_emb, 771, 1024, &points, false);
            let n: usize = mask.shape().iter().product();
            let flat: fusor::Tensor<1, f32, fusor::ConcreteTensor<f32, 1>> =
                mask.reshape([n]).to_concrete();
            let _ = flat.as_slice().await.unwrap();
        }
        let fusor_dec = t0.elapsed() / n_dec;

        eprintln!("=== Decoder only ({n_dec} iters) ===");
        eprintln!("  Candle (CPU):   {candle_dec:?}/iter");
        if let Some((_, cm)) = cm_result {
            eprintln!("  Candle (Metal): {cm:?}/iter");
        } else {
            eprintln!("  Candle (Metal): N/A");
        }
        eprintln!("  Fusor  (GPU):   {fusor_dec:?}/iter");
        eprintln!(
            "  Fusor vs Candle CPU: {:.2}x",
            candle_dec.as_secs_f64() / fusor_dec.as_secs_f64()
        );
    }

    /// Compare batched vs unbatched forward_for_embeddings to verify numerical equivalence.
    /// Uses the exact same reshape+as_slice reading pattern as segment_everything.
    #[tokio::test]
    async fn test_batched_vs_unbatched() {
        let gguf_path = Path::new("/tmp/mobile_sam-tiny-vitt.gguf");
        if !gguf_path.exists() {
            return;
        }

        let model = SegmentAnything::from_gguf_path(gguf_path, true)
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
        let batched_masks_data = masks_slice.as_slice();

        let total_iou = batch * n_masks;
        let iou_flat: Tensor<1, f32, ConcreteTensor<f32, 1>> =
            batched_iou.reshape([total_iou]).to_concrete();
        let iou_slice = iou_flat.as_slice().await.unwrap();
        let batched_iou_data = iou_slice.as_slice();

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
            unbatched_masks_data.extend_from_slice(ms_data.as_slice());

            let is = iou.shape();
            let i_total = is[0] * is[1];
            let if_: Tensor<1, f32, ConcreteTensor<f32, 1>> = iou.reshape([i_total]).to_concrete();
            let is_data = if_.as_slice().await.unwrap();
            unbatched_iou_data.extend_from_slice(is_data.as_slice());
        }

        // --- Compare ---
        let mask_diff = max_diff(batched_masks_data, &unbatched_masks_data);
        let iou_diff = max_diff(batched_iou_data, &unbatched_iou_data);

        eprintln!("=== Batched vs Unbatched (reshape+as_slice) ===");
        eprintln!("  Batch={batch}, n_masks={n_masks}, mask_size={mask_h}x{mask_w}");
        eprintln!("  Mask max diff:  {mask_diff}");
        eprintln!("  IoU  max diff:  {iou_diff}");
        eprintln!(
            "  Total mask elements:  batched={} unbatched={}",
            batched_masks_data.len(),
            unbatched_masks_data.len()
        );

        // Also verify per-point IoU values match
        for i in 0..test_points.len() {
            for m in 0..n_masks {
                let idx = i * n_masks + m;
                eprintln!(
                    "  Point {i} Mask {m}: batched_iou={:.4} unbatched_iou={:.4}",
                    batched_iou_data[idx], unbatched_iou_data[idx]
                );
            }
        }

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
    fn test_low_res_crop_extent_uses_ceil_coverage() {
        assert_eq!(low_res_crop_extent(771, 256), 193);
        assert_eq!(low_res_crop_extent(769, 256), 193);
        assert_eq!(low_res_crop_extent(1024, 256), 256);
        assert_eq!(low_res_crop_extent(1, 256), 1);
    }
}
