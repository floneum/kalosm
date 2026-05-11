use fusor_tile_ir::{
    F32Bits, KernelBuilder, KernelTensorRef, Layout, MemoryLevel, NagaKernel, Shape, F32,
};
use fusor_tile_ir_kernels::{
    flash_attention, linear_storage_layout, rms_norm_vec4, FlashAttentionDims, FlashAttentionMeta,
    RmsNormVec4Meta, TensorMeta,
};

fn lower_or_fail(ir: &fusor_tile_ir::KernelIr, label: &str) -> NagaKernel {
    ir.lower_to_naga()
        .unwrap_or_else(|error| panic!("{label} lowering failed: {error}"))
}

#[test]
fn streaming_flash_attention_regression_shape_lowers_to_naga() {
    let layout = linear_storage_layout();
    let mut kb = KernelBuilder::<()>::new();
    flash_attention::<F32, ()>(
        &mut kb,
        KernelTensorRef::new((), layout.clone()),
        KernelTensorRef::new((), layout.clone()),
        KernelTensorRef::new((), layout.clone()),
        None,
        KernelTensorRef::new((), layout),
        FlashAttentionMeta {
            dims: FlashAttentionDims {
                batch: 1,
                num_heads: 32,
                num_kv_heads: 8,
                q_seq_len: 48,
                kv_seq_len: 48,
                head_dim: 128,
            },
            scale: F32Bits::new(1.0 / 128.0f32.sqrt()),
            q_meta: TensorMeta::new(vec![196_608, 6_144, 128, 1], 0),
            k_meta: TensorMeta::new(vec![49_152, 6_144, 128, 1], 0),
            v_meta: TensorMeta::new(vec![49_152, 6_144, 128, 1], 0),
            mask_meta: None,
            output_meta: TensorMeta::new(vec![196_608, 6_144, 128, 1], 0),
            dispatch_size: [16, 1536, 1],
        },
    )
    .expect("streaming flash attention should build");
    let (ir, _) = kb.finish();

    lower_or_fail(&ir, "streaming flash attention");
}

#[test]
fn rms_norm_vec4_minimal_lowers() {
    let layout = Layout::strided(MemoryLevel::Storage, Shape::new([1]), &[1]);
    let mut kb = KernelBuilder::<()>::new();
    let input = KernelTensorRef::with_offset((), layout.clone(), 0);
    let weight = KernelTensorRef::with_offset((), layout.clone(), 0);
    let output = KernelTensorRef::with_offset((), layout.clone(), 0);
    let meta = RmsNormVec4Meta {
        cols: 4,
        cols_vec: 1,
        eps: F32Bits::new(1e-5),
        input_offset_vec: 0,
        input_row_stride_vec: 1,
        residual_offset_vec: None,
        residual_row_stride_vec: 0,
        weight_offset_vec: 0,
        bias_offset_vec: None,
        output_offset_vec: 0,
        output_row_stride_vec: 1,
    };
    rms_norm_vec4(&mut kb, input, None, weight, None, output, meta, 1).unwrap();
    let (ir, _) = kb.finish();
    lower_or_fail(&ir, "rms_norm_vec4");
}
