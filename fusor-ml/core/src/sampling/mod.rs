use crate::{Device, tensor::TensorData};

mod mirostat;
mod pipeline;
pub(crate) mod processors;
mod qmat_topk;
mod topk;

#[cfg(test)]
mod tests;

pub(crate) use pipeline::{mirostat2_sample_token_to_host, qmat_mirostat2_sample_token_to_host};
pub(crate) use topk::{
    MergeSortedChunkTopKParams, chunk_top_k_pair_data_with_encoder,
    merge_sorted_chunk_top_k_pair_data_with_encoder,
};

pub(crate) const TOP_K_BLOCK: u32 = 256;
pub(crate) const TOP_K_CHUNK: usize = TOP_K_BLOCK as usize;
pub(crate) const DEFAULT_MIN_TOP_K_CANDIDATES_PER_CHUNK: usize = 16;
pub(crate) const GPU_SAMPLER_PREVIOUS_TOKENS: usize = 64;
pub(crate) const GPU_SAMPLE_RESULT_WORDS: usize = 2;
pub(crate) const GPU_SAMPLE_STATUS_RETRY_NEEDED: u32 = 0;
pub(crate) const GPU_SAMPLE_STATUS_SAMPLED: u32 = 1;
pub(crate) const GPU_SAMPLE_STATUS_INVALID: u32 = 2;

pub(crate) fn min_top_k_candidates_per_chunk() -> usize {
    std::env::var("FUSOR_TOP_K_MIN_CANDIDATES_PER_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MIN_TOP_K_CANDIDATES_PER_CHUNK)
        .min(TOP_K_CHUNK)
}

#[derive(Clone, Copy, Debug)]
pub struct GpuMirostat2SamplerParams {
    pub top_k: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub tau: f32,
    pub eta: f32,
    pub random: f32,
}

#[derive(Clone, Debug)]
pub struct GpuMirostat2Sampler {
    pub(crate) state: TensorData,
}

impl GpuMirostat2Sampler {
    pub fn new(device: &Device, mu: f32) -> Self {
        let state = TensorData::new_splat(device, &[1], mu);
        Self { state }
    }
}
