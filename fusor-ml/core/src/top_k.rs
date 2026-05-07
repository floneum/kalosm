pub use crate::sampling::{GpuMirostat2Sampler, GpuMirostat2SamplerParams};

pub(crate) use crate::sampling::{
    MIN_TOP_K_CANDIDATES_PER_CHUNK, TOP_K_CHUNK, chunk_top_k_pair_data,
    merge_sorted_chunk_top_k_pair_data, mirostat2_sample_token_to_host,
    qmat_mirostat2_sample_token_to_host,
};

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};

    #[tokio::test]
    async fn facade_keeps_top_k_pairs_coverage_addressable() {
        let device = Device::new().await.unwrap();
        let values = [0.25, f32::NAN, 7.0, -3.0, 2.5, 9.0, 8.5, 9.0];
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(4).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(4);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
