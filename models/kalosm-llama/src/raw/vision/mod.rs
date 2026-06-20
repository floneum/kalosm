mod gemma;
mod qwen;
mod qwen_image_processing;
mod qwen_patch_merger;
mod qwen_rope;
mod qwen_vision;
mod qwen_vision_block;
mod qwen_vision_embed;

use std::ops::Range;

use fusor::{CastTensor, CastTo, Device, FloatDataType, Result, SimdElement, Tensor};
use fusor_gguf::GgufMetadata;

pub(crate) use gemma::GemmaVisionTransformer;
pub(crate) use qwen::QwenVisionTransformer;

pub const QWEN_EPS: f64 = 1e-6;

pub(crate) enum VisionTransformer<F: FloatDataType + SimdElement = f32> {
    Qwen(QwenVisionTransformer<F>),
    Gemma(GemmaVisionTransformer<F>),
}

impl<F: FloatDataType + SimdElement> VisionTransformer<F>
where
    F: CastTo<f32> + CastTensor<f32> + fusor::FloatOps + fusor::MatmulImpl + Default,
    f32: CastTo<F> + CastTensor<F>,
    fusor::MulOp: fusor::SimdBinaryOp<F>,
    fusor::AddOp: fusor::SimdBinaryOp<F>,
    fusor::SumOp: fusor::SimdReduceOp<F>,
{
    pub(crate) fn from_gguf(
        vision_ct: GgufMetadata,
        vision_bytes: &[u8],
        device: &Device,
    ) -> Result<Self> {
        let projector_type = vision_ct
            .metadata
            .get("clip.vision.projector_type")
            .and_then(|x| x.to_string().ok())
            .map(|x| x.to_string());

        match projector_type.as_deref() {
            Some("gemma4v") => Ok(Self::Gemma(GemmaVisionTransformer::from_gguf(
                vision_ct,
                vision_bytes,
                device,
            )?)),
            _ => Ok(Self::Qwen(QwenVisionTransformer::from_gguf(
                vision_ct,
                vision_bytes,
                device,
            )?)),
        }
    }

    pub(crate) fn preprocess_image(
        &self,
        image: &image::DynamicImage,
        min_pixels: Option<u32>,
        max_pixels: Option<u32>,
    ) -> Result<(Tensor<2, f32>, [u32; 3])> {
        match self {
            Self::Qwen(vision) => vision.preprocess_image(image, min_pixels, max_pixels),
            Self::Gemma(vision) => vision.preprocess_image(image, min_pixels, max_pixels),
        }
    }

    pub(crate) fn image_token_count(&self, grid: [u32; 3]) -> u32 {
        match self {
            Self::Qwen(vision) => {
                grid.iter().product::<u32>() / (vision.spacial_merge_size as u32).pow(2)
            }
            Self::Gemma(vision) => vision.image_token_count(grid),
        }
    }

    pub(crate) fn expand_image_tokens(
        &self,
        raw_tokens: &[u32],
        image_pad_token: u32,
        vision_start_token: Option<u32>,
        image_start_token: Option<u32>,
        image_end_token: Option<u32>,
        grid_thw: &[[u32; 3]],
    ) -> Result<(Vec<u32>, Vec<Range<usize>>)> {
        match self {
            Self::Qwen(_) => {
                let Some(vision_start_token) = vision_start_token else {
                    return Ok((raw_tokens.to_vec(), Vec::new()));
                };
                let mut tokens = Vec::new();
                let mut token_iter = raw_tokens.iter().copied();
                let mut image_iter = grid_thw.iter();
                let mut image_token_ranges = Vec::new();
                while let Some(token) = token_iter.next() {
                    tokens.push(token);
                    let start_index = tokens.len();
                    if token == vision_start_token {
                        match token_iter.next() {
                            Some(next) if next == image_pad_token => {
                                let grid = *image_iter.next().ok_or_else(|| {
                                    fusor::Error::msg(
                                        "Image pad token found without matching image.",
                                    )
                                })?;
                                for _ in 0..self.image_token_count(grid) {
                                    tokens.push(image_pad_token);
                                }
                                image_token_ranges.push(start_index..tokens.len());
                            }
                            Some(next) => {
                                tokens.push(next);
                            }
                            None => break,
                        }
                    }
                }
                Ok((tokens, image_token_ranges))
            }
            Self::Gemma(_) => {
                let mut tokens = Vec::new();
                let mut image_iter = grid_thw.iter();
                let mut image_token_ranges = Vec::new();
                for token in raw_tokens.iter().copied() {
                    if token == image_pad_token {
                        let grid = *image_iter.next().ok_or_else(|| {
                            fusor::Error::msg("Image token found without matching image.")
                        })?;
                        if let Some(image_start_token) = image_start_token {
                            tokens.push(image_start_token);
                        }
                        let start_index = tokens.len();
                        for _ in 0..self.image_token_count(grid) {
                            tokens.push(image_pad_token);
                        }
                        image_token_ranges.push(start_index..tokens.len());
                        if let Some(image_end_token) = image_end_token {
                            tokens.push(image_end_token);
                        }
                    } else {
                        tokens.push(token);
                    }
                }
                Ok((tokens, image_token_ranges))
            }
        }
    }

    pub(crate) fn get_rope_index(
        &self,
        input_ids: &[u32],
        grid_thw: &[[u32; 3]],
        config: &crate::raw::LlamaConfig<F>,
        start_time: u32,
    ) -> Result<Option<(Tensor<2, u32>, u32)>> {
        match self {
            Self::Qwen(vision) => vision
                .get_rope_index(input_ids, grid_thw, config, start_time)
                .map(Some),
            Self::Gemma(_) => Ok(None),
        }
    }

    pub(crate) fn forward_image(
        &self,
        pixels: &Tensor<2, F>,
        grid: [u32; 3],
    ) -> Result<Tensor<2, F>> {
        match self {
            Self::Qwen(vision) => vision.forward_image(pixels, grid),
            Self::Gemma(vision) => vision.forward_image(pixels, grid),
        }
    }

    pub(crate) fn outputs_on_isolated_device(&self) -> bool {
        match self {
            Self::Qwen(_) => false,
            Self::Gemma(vision) => vision.uses_isolated_device(),
        }
    }
}
