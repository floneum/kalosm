use crate::{
    Layout,
    quantized::{QMatrix, matmul::QMatMulOperation},
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::CommandEncoder;

use super::{MIN_TOP_K_CANDIDATES_PER_CHUNK, TOP_K_CHUNK};

pub(super) fn qmat_logits_data_with_encoder(
    hidden: &TensorData,
    matrix: &QMatrix,
    encoder: &mut CommandEncoder,
) -> Option<TensorData> {
    if hidden.datatype() != DataTypeEnum::F32 || hidden.layout().rank() != 1 {
        return None;
    }
    let hidden_len = hidden.layout().shape()[0];
    let hidden_stride = hidden.layout().strides()[0];
    let [vocab_len, matrix_hidden_len] = matrix.shape() else {
        return None;
    };
    if hidden_len != *matrix_hidden_len || *vocab_len == 0 {
        return None;
    }

    let device = hidden.device();
    let logits = TensorData::new_for_shape(device, &[*vocab_len], DataTypeEnum::F32);
    let hidden_2d = TensorData::new_from_parts(
        device,
        hidden.buffer().clone(),
        Layout::from_parts(
            hidden.layout().offset(),
            Box::new([1, hidden_len]),
            Box::new([0, hidden_stride]),
        ),
        DataTypeEnum::F32,
    );
    let logits_2d = TensorData::new_from_parts(
        device,
        logits.buffer().clone(),
        Layout::from_parts(0, Box::new([1, *vocab_len]), Box::new([*vocab_len, 1])),
        DataTypeEnum::F32,
    );
    let kernel = QMatMulOperation::direct_kernel_for_tensors(
        device,
        &hidden_2d,
        matrix,
        &logits_2d,
        "q_mat_logits_for_sampler",
        None,
        None,
        None,
    )?;
    kernel.run(device.kernel_cache(), encoder);

    Some(logits)
}

pub(super) fn initial_sampler_candidate_count(top_k: usize, chunks: usize) -> usize {
    top_k
        .div_ceil(chunks)
        .max(MIN_TOP_K_CANDIDATES_PER_CHUNK)
        .min(top_k)
        .min(TOP_K_CHUNK)
}

pub(super) fn sampler_output_per_chunk(candidate_count: usize) -> usize {
    if candidate_count >= TOP_K_CHUNK {
        TOP_K_CHUNK
    } else {
        candidate_count + 1
    }
}

pub(super) fn next_sampler_candidate_count(candidate_count: usize, top_k: usize) -> usize {
    candidate_count
        .saturating_mul(2)
        .min(top_k)
        .min(TOP_K_CHUNK)
}
