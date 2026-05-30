//! # Segment Anything RS
//! A rust wrapper for [Segment Anything](https://segment-anything.com/)
//!
//! ## Usage
//!
//! ```rust, no_run
//! use segment_anything_rs::*;
//!
//! let model = SegmentAnything::builder().build().unwrap();
//! let image = image::open("examples/landscape.jpg").unwrap();
//! let images = model.segment_everything(image).unwrap();
//! for (i, img) in images.iter().enumerate() {
//!     img.save(&format!("{}.png", i)).unwrap();
//! }
//! ```

#![warn(missing_docs)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle_core::DType;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::segment_anything::sam::{self, Sam};
use image::{DynamicImage, GenericImage, GenericImageView, ImageBuffer, Rgba};
use kalosm_common::CacheError;

/// A builder for [`SegmentAnything`].
#[derive(Default)]
pub struct SegmentAnythingBuilder {
    source: SegmentAnythingSource,
}

impl SegmentAnythingBuilder {
    /// Sets the source of the model.
    pub fn source(mut self, source: SegmentAnythingSource) -> Self {
        self.source = source;
        self
    }

    /// Builds the [`SegmentAnything`] model.
    pub fn build(self) -> Result<SegmentAnything, LoadSegmentAnythingError> {
        SegmentAnything::new(self)
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
        let mut self_ = Self::new("lmz/candle-sam", "mobile_sam-tiny-vitt.safetensors");
        self_.tiny = true;
        self_
    }

    /// Create a normal sized model source.
    pub fn medium() -> Self {
        Self::new("lmz/candle-sam", "sam_vit_b_01ec64.safetensors")
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
    /// An error that can occur when trying to load a [`SegmentAnything`] model into a device.
    #[error("Failed to load model into device: {0}")]
    LoadModel(#[from] candle_core::Error),
    /// An error that can occur when downloading a [`SegmentAnything`] model from Hugging Face.
    #[error("Failed to download model from Hugging Face: {0}")]
    DownloadModel(#[from] CacheError),
}

/// An error that can occur when running a [`SegmentAnything`] model.
#[derive(Debug, thiserror::Error)]
pub enum SegmentAnythingInferenceError {
    /// An error that can occur when trying to run a [`SegmentAnything`] model.
    #[error("Failed to run model: {0}")]
    RunModel(#[from] candle_core::Error),
    /// An error that can occur when converting the result of a [`SegmentAnything`] model to an image.
    #[error("Failed to merge masks")]
    MergeMasks,
}

/// The [segment anything](https://segment-anything.com/) model.
pub struct SegmentAnything {
    device: Device,
    sam: Sam,
}

impl SegmentAnything {
    /// Creates a new [`SegmentAnythingBuilder`].
    pub fn builder() -> SegmentAnythingBuilder {
        SegmentAnythingBuilder::default()
    }

    fn new(settings: SegmentAnythingBuilder) -> Result<Self, LoadSegmentAnythingError> {
        let SegmentAnythingBuilder { source } = settings;
        let model = {
            let source = kalosm_model_types::FileSource::huggingface(
                source.model,
                "main".to_string(),
                source.filename,
            );
            pollster::block_on(kalosm_common::Cache::default().get(&source, |_| {}))?
        };
        // Currently, candle doesn't support some operations that are required for segment anything
        // let device = kalosm_common::accelerated_device_if_available()?;
        let device = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
        let sam = if source.tiny {
            sam::Sam::new_tiny(vb)? // tiny vit_t
        } else {
            sam::Sam::new(768, 12, 12, &[2, 5, 8, 11], vb)? // sam_vit_b
        };
        Ok(Self { device, sam })
    }

    /// Segment an image from a list of points. Returns a [`DynamicImage`] mask.
    ///
    /// # Example
    /// ```rust, no_run
    /// use segment_anything_rs::*;
    ///
    /// let model = SegmentAnything::builder().build().unwrap();
    /// let image = image::open("examples/landscape.jpg").unwrap();
    /// let x = image.width() / 2;
    /// let y = image.height() / 4;
    /// let images = model
    ///     .segment_from_points(SegmentAnythingInferenceSettings::new(image).add_goal_point(x, y))
    ///     .unwrap();
    ///
    /// images.save("out.png").unwrap();
    /// ```
    pub fn segment_from_points(
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

        let image_tensor = self.image_to_tensor(image)?;

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

        let (mask, _iou_predictions) = self.sam.forward(&image_tensor, &points, false)?;

        let mask = (mask.ge(threshold)? * 255.)?;
        let (_one, h, w) = mask.dims3()?;
        let mask = mask.expand((3, h, w))?;

        let mask_pixels = mask.permute((1, 2, 0))?.flatten_all()?.to_vec1::<u8>()?;
        let mask_img: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
            image::ImageBuffer::from_raw(w as u32, h as u32, mask_pixels)
                .ok_or(SegmentAnythingInferenceError::MergeMasks)?;

        Ok(image::DynamicImage::from(mask_img).resize_to_fill(
            image_width,
            image_height,
            image::imageops::FilterType::CatmullRom,
        ))
    }

    fn image_to_tensor(&self, image: DynamicImage) -> candle_core::Result<Tensor> {
        let image = {
            let resize_longest = sam::IMAGE_SIZE;
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
        let image = Tensor::from_vec(data, (height, width, 3), &self.device)?.permute((2, 0, 1))?;

        let image = image.to_device(&self.device)?;

        Ok(image)
    }

    /// Segment everything in an image. Returns a list of [`DynamicImage`] masks.
    ///
    /// # Example
    ///
    /// ```rust, no_run
    /// use segment_anything_rs::*;
    ///
    /// let model = SegmentAnything::builder().build().unwrap();
    /// let image = image::open("examples/landscape.jpg").unwrap();
    /// let images = model.segment_everything(image).unwrap();
    /// for (i, img) in images.iter().enumerate() {
    ///     img.save(&format!("{}.png", i)).unwrap();
    /// }
    /// ```
    pub fn segment_everything(
        &self,
        image: DynamicImage,
    ) -> Result<Vec<DynamicImage>, SegmentAnythingInferenceError> {
        let image = self.image_to_tensor(image)?;

        let bboxes = self.sam.generate_masks(&image, 32, 0, 512. / 1500., 1)?;
        let mut masks = Vec::new();
        for bbox in bboxes {
            let mask = (&bbox.data.to_dtype(DType::U8)? * 255.)?;
            let (h, w) = mask.dims2()?;
            let mask = mask.broadcast_as((3, h, w))?;
            let (channel, height, width) = mask.dims3()?;
            if channel != 3 {
                return Err(candle_core::Error::Msg(
                    "save_image expects an input of shape (3, height, width)".to_string(),
                )
                .into());
            }
            let mask = mask.permute((1, 2, 0))?.flatten_all()?;
            let pixels = mask.to_vec1::<u8>()?;
            let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                image::ImageBuffer::from_raw(width as u32, height as u32, pixels)
                    .ok_or(SegmentAnythingInferenceError::MergeMasks)?;
            let image = image::DynamicImage::from(image);
            let image =
                image.resize_to_fill(w as u32, h as u32, image::imageops::FilterType::CatmullRom);
            masks.push(image);
        }

        Ok(masks)
    }
}
