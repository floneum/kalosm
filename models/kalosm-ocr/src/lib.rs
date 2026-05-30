//! # Kalosm OCR
//!
//! A rust wrapper for [TR OCR](https://huggingface.co/docs/transformers/model_doc/trocr)
//!
//! ## Usage
//!
//! ```rust, no_run
//! # #[tokio::main]
//! # async fn main() {
//! use kalosm_ocr::*;
//!
//! let mut model = Ocr::builder().build().await.unwrap();
//! let image = image::open("examples/ocr.png").unwrap();
//! let text = model
//!     .recognize_text(OcrInferenceSettings::new(image))
//!     .unwrap();
//!
//! println!("{}", text);
//! # }
//! ```

#![warn(missing_docs)]
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

mod image_processor;

use candle_core::DType;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::trocr;
use candle_transformers::models::vit;
use image::{GenericImage, GenericImageView, ImageBuffer, Rgba};
use kalosm_common::*;
use kalosm_model_types::{FileSource, ModelLoadingProgress};
use tokenizers::Tokenizer;

/// A builder for [`Ocr`].
#[derive(Default)]
pub struct OcrBuilder {
    source: OcrSource,
}

impl OcrBuilder {
    /// Sets the source of the model.
    pub fn with_source(mut self, source: OcrSource) -> Self {
        self.source = source;
        self
    }

    /// Builds the [`Ocr`] model.
    pub async fn build(self) -> Result<Ocr, LoadOcrError> {
        Ocr::new(self, |_| {}).await
    }

    /// Builds the [`Ocr`] model.
    pub async fn build_with_loading_handler(
        self,
        handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Ocr, LoadOcrError> {
        Ocr::new(self, handler).await
    }
}

/// The source of the model.
pub struct OcrSource {
    model: FileSource,
    config: FileSource,
    tokenizer: FileSource,
}

impl OcrSource {
    /// Creates a new [`OcrSource`].
    pub fn new(model: FileSource, config: FileSource) -> Self {
        Self {
            model,
            config,
            tokenizer: FileSource::huggingface(
                "ToluClassics/candle-trocr-tokenizer",
                "main",
                "tokenizer.json",
            ),
        }
    }

    /// Set the tokenizer source.
    pub fn with_tokenizer(mut self, tokenizer: FileSource) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// Create the base model source.
    pub fn base() -> Self {
        Self::new(
            FileSource::huggingface(
                "microsoft/trocr-base-handwritten".to_string(),
                "refs/pr/3".to_string(),
                "model.safetensors".to_string(),
            ),
            FileSource::huggingface(
                "microsoft/trocr-base-handwritten".to_string(),
                "refs/pr/3".to_string(),
                "config.json".to_string(),
            ),
        )
    }

    /// Create a normal sized model source.
    pub fn large() -> Self {
        Self::new(
            FileSource::huggingface(
                "microsoft/trocr-large-handwritten".to_string(),
                "refs/pr/6".to_string(),
                "model.safetensors".to_string(),
            ),
            FileSource::huggingface(
                "microsoft/trocr-large-handwritten".to_string(),
                "refs/pr/6".to_string(),
                "config.json".to_string(),
            ),
        )
    }

    /// Create a base printed model source.
    pub fn base_printed() -> Self {
        Self::new(
            FileSource::huggingface(
                "microsoft/trocr-base-printed".to_string(),
                "refs/pr/7".to_string(),
                "model.safetensors".to_string(),
            ),
            FileSource::huggingface(
                "microsoft/trocr-base-printed".to_string(),
                "refs/pr/7".to_string(),
                "config.json".to_string(),
            ),
        )
    }

    /// Create a large printed model source.
    pub fn large_printed() -> Self {
        Self::new(
            FileSource::huggingface(
                "microsoft/trocr-large-printed".to_string(),
                "main".to_string(),
                "model.safetensors".to_string(),
            ),
            FileSource::huggingface(
                "microsoft/trocr-large-printed".to_string(),
                "main".to_string(),
                "config.json".to_string(),
            ),
        )
    }

    async fn varbuilder(
        &self,
        device: &Device,
        mut handler: impl FnMut(ModelLoadingProgress) + Send + Sync,
    ) -> Result<VarBuilder<'_>, LoadOcrError> {
        let source = format!("Model ({})", self.model);
        let mut create_progress = ModelLoadingProgress::downloading_progress(source);
        let cache = Cache::default();
        let filename = cache
            .get(&self.model, |progress| handler(create_progress(progress)))
            .await?;
        Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[filename], DType::F32, device)? })
    }

    async fn config(
        &self,
        mut handler: impl FnMut(ModelLoadingProgress) + Send + Sync,
    ) -> Result<(vit::Config, trocr::TrOCRConfig), LoadOcrError> {
        #[derive(Debug, Clone, serde::Deserialize)]
        struct Config {
            encoder: vit::Config,
            decoder: trocr::TrOCRConfig,
        }

        let (encoder_config, decoder_config) = {
            let source = format!("Config ({})", self.model);
            let mut create_progress = ModelLoadingProgress::downloading_progress(source);
            let cache = Cache::default();
            let config_bytes = cache
                .get_bytes(&self.config, |progress| handler(create_progress(progress)))
                .await?;
            let config: Config =
                serde_json::from_slice(&config_bytes).map_err(LoadOcrError::LoadConfig)?;
            (config.encoder, config.decoder)
        };

        Ok((encoder_config, decoder_config))
    }

    async fn tokenizer(
        &self,
        mut handler: impl FnMut(ModelLoadingProgress) + Send + Sync,
    ) -> Result<Tokenizer, LoadOcrError> {
        let source = format!("Tokenizer ({})", self.tokenizer);
        let mut create_progress = ModelLoadingProgress::downloading_progress(source);
        let cache = Cache::default();
        let tokenizer = cache
            .get_bytes(&self.tokenizer, |progress| {
                handler(create_progress(progress))
            })
            .await?;
        Tokenizer::from_bytes(&tokenizer).map_err(LoadOcrError::LoadTokenizer)
    }
}

impl Default for OcrSource {
    fn default() -> Self {
        Self::base()
    }
}

/// Settings for running inference on [`Ocr`].
pub struct OcrInferenceSettings {
    image: ImageBuffer<image::Rgba<u8>, Vec<u8>>,
}

impl OcrInferenceSettings {
    /// Creates a new [`OcrInferenceSettings`] from an image.
    pub fn new<I: GenericImageView<Pixel = Rgba<u8>>>(input: I) -> Self {
        let mut image = ImageBuffer::new(input.width(), input.height());
        image.copy_from(&input, 0, 0).unwrap();
        Self { image }
    }
}

/// An error that can occur when loading an [`Ocr`] model.
#[derive(Debug, thiserror::Error)]
pub enum LoadOcrError {
    /// An error that can occur when trying to load an [`Ocr`] model into a device.
    #[error("Failed to load model into device: {0}")]
    LoadModel(#[from] candle_core::Error),
    /// An error that can occur when downloading an [`Ocr`] model from the cache.
    #[error("Failed to download model: {0}")]
    DownloadModel(#[from] CacheError),
    /// An error that can occur when loading the tokenizer.
    #[error("Failed to load tokenizer: {0}")]
    LoadTokenizer(tokenizers::Error),
    /// An error that can occur when loading the config.
    #[error("Failed to load config: {0}")]
    LoadConfig(serde_json::Error),
}

/// An error that can occur when running an [`Ocr`] model.
#[derive(Debug, thiserror::Error)]
pub enum OcrInferenceError {
    /// An error that can occur when trying to run an [`Ocr`] model.
    #[error("Failed to run model: {0}")]
    RunModel(#[from] candle_core::Error),
    /// An error that can occur when decoding the result of an [`Ocr`] model.
    #[error("Failed to decode: {0}")]
    Decode(tokenizers::Error),
}

/// The [trocs](https://www.microsoft.com/en-us/research/publication/trocr-transformer-based-optical-character-recognition-with-pre-trained-models/) optical character recognition model.
pub struct Ocr {
    device: Device,
    decoder: trocr::TrOCRModel,
    decoder_config: trocr::TrOCRConfig,
    processor: image_processor::ViTImageProcessor,
    tokenizer_dec: Tokenizer,
}

impl Ocr {
    /// Creates a new [`OcrBuilder`].
    pub fn builder() -> OcrBuilder {
        OcrBuilder::default()
    }

    async fn new(
        settings: OcrBuilder,
        mut handler: impl FnMut(ModelLoadingProgress) + Send + Sync + 'static,
    ) -> Result<Self, LoadOcrError> {
        let OcrBuilder { source } = settings;
        let tokenizer_dec = source.tokenizer(&mut handler).await?;
        let device = accelerated_device_if_available()?;

        let vb = source.varbuilder(&device, &mut handler).await?;

        let (encoder_config, decoder_config) = source.config(&mut handler).await?;

        let model = trocr::TrOCRModel::new(&encoder_config, &decoder_config, vb)?;

        let config = image_processor::ProcessorConfig::default();
        let processor = image_processor::ViTImageProcessor::new(&config);

        Ok(Self {
            device,
            decoder: model,
            processor,
            decoder_config,
            tokenizer_dec,
        })
    }

    /// Recognize text from an image. Returns the recognized text.
    ///
    /// # Example
    /// ```rust, no_run
    /// # #[tokio::main]
    /// # async fn main() {
    /// use kalosm_ocr::*;
    ///
    /// let mut model = Ocr::builder().build().await.unwrap();
    /// let image = image::open("examples/ocr.png").unwrap();
    /// let text = model
    ///     .recognize_text(OcrInferenceSettings::new(image))
    ///     .unwrap();
    ///
    /// println!("{}", text);
    /// # }
    /// ```
    pub fn recognize_text(
        &mut self,
        settings: OcrInferenceSettings,
    ) -> Result<String, OcrInferenceError> {
        let OcrInferenceSettings { image } = settings;

        let image = image::DynamicImage::ImageRgba8(image);

        let image = vec![image];
        let image = self.processor.preprocess(image, &self.device)?;

        let encoder_xs = self.decoder.encoder().forward(&image)?;

        let mut logits_processor =
            candle_transformers::generation::LogitsProcessor::new(1337, None, None);

        let mut token_ids: Vec<u32> = vec![self.decoder_config.decoder_start_token_id];
        for index in 0..1000 {
            let context_size = if index >= 1 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids = Tensor::new(&token_ids[start_pos..], &self.device)?.unsqueeze(0)?;

            let logits = self.decoder.decode(&input_ids, &encoder_xs, start_pos)?;

            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let token = logits_processor.sample(&logits)?;
            token_ids.push(token);

            if token == self.decoder_config.eos_token_id {
                break;
            }
        }

        let decoded = self
            .tokenizer_dec
            .decode(&token_ids, true)
            .map_err(OcrInferenceError::Decode)?;

        Ok(decoded)
    }
}
