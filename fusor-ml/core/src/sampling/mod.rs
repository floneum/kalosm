use crate::{Device, Tensor, tensor::TensorData};

mod mirostat;
mod pipeline;
pub(crate) mod processors;
pub(crate) mod row_kernels;
mod standard_sampler;
mod topk;

#[cfg(test)]
mod tests;

pub(crate) use pipeline::{GpuSamplerRequest, sample_token_pending, sample_token_to_host};
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

pub struct PendingGpuSampledToken {
    token: Tensor,
    download: wgpu::Buffer,
    receiver: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
}

impl PendingGpuSampledToken {
    pub(crate) fn new(
        token: Tensor,
        download: wgpu::Buffer,
        receiver: futures_channel::oneshot::Receiver<Result<(), wgpu::BufferAsyncError>>,
    ) -> Self {
        Self {
            token,
            download,
            receiver,
        }
    }

    pub fn token_tensor(&self) -> &Tensor {
        &self.token
    }

    pub async fn read_token(self) -> Result<Option<u32>, wgpu::BufferAsyncError> {
        // Diagnostic: time the wait for GPU completion + token readback (the
        // per-token sync the pending decode path stalls on). On wasm this is
        // always on; native gates on the trace config flags.
        let config = self.token.device().config();
        let trace =
            cfg!(target_arch = "wasm32") || config.trace_decode || config.trace_sampler;
        let await_start = trace.then(web_time::Instant::now);
        self.receiver.await.map_err(|_| wgpu::BufferAsyncError)??;
        if let Some(start) = await_start {
            tracing::info!("sampler_trace pending_await elapsed={:?}", start.elapsed());
        }

        let view = self.download.slice(..).get_mapped_range();
        let word_size = std::mem::size_of::<u32>();
        let status = view
            .get(..word_size)
            .map(bytemuck::from_bytes::<u32>)
            .copied()
            .unwrap_or(GPU_SAMPLE_STATUS_INVALID);
        let token = view
            .get(word_size..word_size * GPU_SAMPLE_RESULT_WORDS)
            .map(bytemuck::from_bytes::<u32>)
            .copied()
            .unwrap_or_default();
        drop(view);
        self.download.unmap();

        match status {
            GPU_SAMPLE_STATUS_SAMPLED => Ok(Some(token)),
            _ => Ok(None),
        }
    }
}

pub(crate) fn min_top_k_candidates_per_chunk(config: &fusor_tile_ir_runtime::FusorConfig) -> usize {
    config
        .top_k_min_candidates_per_chunk
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

#[derive(Clone, Copy, Debug)]
pub struct GpuStandardSamplerParams {
    pub top_k: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub top_p: f32,
    pub min_p: f32,
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
