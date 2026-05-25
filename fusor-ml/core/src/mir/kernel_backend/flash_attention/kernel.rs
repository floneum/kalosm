use std::{any::TypeId, hash::Hash, sync::Arc};

use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;

use crate::{
    DataTypeEnum,
    compute_graph::{ComputeGraphInner, NodeIndex},
    kernel_selection::KernelDeviceCaps,
    mir::{
        inputs::MirValue,
        kernel_backend,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    tensor::TensorData,
};

use super::{
    DECODE_HEAD_DIM, DECODE_SMALL_BLOCK, FLASH_STREAMING_SUBGROUP_SIZES,
    FLASH_STREAMING_TILED_Q_BLOCK, FlashAttentionDirectKernelVariant, FlashAttentionKernelVariant,
    FlashAttentionOperation, FlashDecodeSmallMeta, FlashDecodeSmallTensors, TensorMeta,
    build_flash_decode_small_meta, dispatch_streaming_flash_attention,
    dispatch_streaming_tiled_flash_attention, flash_attention_module_cache,
    flash_streaming_tiled_eligible, select_flash_attention_variant, streaming_dispatch_size,
};

fn flash_decode_cache_variant(
    variant: FlashAttentionKernelVariant,
    scale: f32,
    meta: &FlashDecodeSmallMeta,
) -> kernel_backend::KernelVariantKey {
    kernel_backend::KernelVariantKey::with_payload::<FlashAttentionDirectKernelVariant>(|state| {
        variant.hash(state);
        scale.to_bits().hash(state);
        meta.decode_block.hash(state);
        meta.tiled.hash(state);
        meta.split_blocks.hash(state);
    })
}

fn hash_flash_decode_dims(
    state: &mut rustc_hash::FxHasher,
    dims: &tile_ir_kernels::FlashAttentionDims,
) {
    dims.batch.hash(state);
    dims.num_heads.hash(state);
    dims.num_kv_heads.hash(state);
    dims.q_seq_len.hash(state);
    dims.kv_seq_len.hash(state);
    dims.head_dim.hash(state);
}

pub(super) fn flash_decode_module_key(
    workgroup_shape: Option<&WorkgroupShape>,
    dispatch_size: [u32; 3],
    input_dtype: DataTypeEnum,
    scale: f32,
    meta: &FlashDecodeSmallMeta,
) -> kernel_backend::KernelCacheKey {
    flash_decode_module_key_for_variant(
        FlashAttentionKernelVariant::DecodeSmall,
        workgroup_shape,
        dispatch_size,
        input_dtype,
        scale,
        meta,
    )
}

fn flash_decode_module_key_for_variant(
    decode_variant: FlashAttentionKernelVariant,
    workgroup_shape: Option<&WorkgroupShape>,
    dispatch_size: [u32; 3],
    input_dtype: DataTypeEnum,
    scale: f32,
    meta: &FlashDecodeSmallMeta,
) -> kernel_backend::KernelCacheKey {
    let variant = flash_decode_cache_variant(decode_variant, scale, meta);
    kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
        // Decode kernels take the active KV length from a params buffer. Do
        // not hash `active_kv_len`, or every generated token would miss the
        // module cache even though the IR is otherwise bucketed by block size.
        2u64.hash(state);
        variant.hash(state);
        TypeId::of::<FlashAttentionOperation>().hash(state);
        workgroup_shape
            .map(|workgroup_shape| workgroup_shape.shape())
            .hash(state);
        dispatch_size.hash(state);
        input_dtype.hash(state);
        hash_flash_decode_dims(state, &meta.dims);
        meta.decode_block.hash(state);
        meta.tiled.hash(state);
        meta.split_blocks.hash(state);
        meta.groups.hash(state);
        meta.q_offset.hash(state);
        meta.k_offset.hash(state);
        meta.v_offset.hash(state);
        meta.output_offset.hash(state);
        meta.q_strides.hash(state);
        meta.k_strides.hash(state);
        meta.v_strides.hash(state);
        meta.output_strides.hash(state);
    })
}

impl Operation for FlashAttentionOperation {
    fn workgroup_shape_constraints(&self, _device: &crate::Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::Equals(1));
        constraints.add_constraint(1, Constraint::Equals(1));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(&self, _workgroup_shape: &WorkgroupShape, _inputs: &[MirValue]) -> [u32; 3] {
        // The streaming kernel's per-axis dispatch depends on the device's
        // hardware subgroup size, which isn't known at this layer; the
        // workgroup-shape pass only uses the row axis (Y/Z) to size local
        // work. Conservatively report the smallest streaming variant's
        // per-axis dispatch; `build_direct_kernel` recomputes the real
        // dispatch with the correct subgroup width before launch.
        let dims = self.dims().expect("flash attention dimensions fit in u32");
        let outputs_per_workgroup = tile_ir_kernels::flash_outputs_per_workgroup(
            *FLASH_STREAMING_SUBGROUP_SIZES.last().expect("non-empty"),
        );
        streaming_dispatch_size(dims, outputs_per_workgroup)
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.q);
        f(self.k);
        f(self.v);
        if let Some(mask) = self.mask {
            f(mask);
        }
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let q = nodes.get_result(self.q).unwrap();
        let k = nodes.get_result(self.k).unwrap();
        let v = nodes.get_result(self.v).unwrap();
        let output = TensorData::new_for_shape(q.device(), &self.out_shape, self.input_dtype);

        let mut inputs = vec![q.into(), k.into(), v.into()];
        if let Some(mask) = self.mask {
            inputs.push(nodes.get_result(mask).unwrap().into());
        }
        inputs.push(output.into());
        inputs
    }

    fn build_direct_kernel(
        &self,
        graph: &ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let q = inputs.first()?.as_tensor()?.clone();
        let k = inputs.get(1)?.as_tensor()?.clone();
        let v = inputs.get(2)?.as_tensor()?.clone();
        let (mask, output_index) = if self.mask.is_some() {
            (Some(inputs.get(3)?.as_tensor()?.clone()), 4)
        } else {
            (None, 3)
        };
        let output = inputs.get(output_index)?.as_tensor()?.clone();
        let device = graph.device();

        // Streaming kernel: pick the effective hardware subgroup size and
        // dispatch a monomorphization tiled to match.
        let streaming_subgroup_size = device.fixed_width_subgroup_size()?;
        if !FLASH_STREAMING_SUBGROUP_SIZES.contains(&streaming_subgroup_size) {
            return None;
        }

        let input_dtype = self.input_dtype;
        if !matches!(input_dtype, DataTypeEnum::F32 | DataTypeEnum::F16) {
            return None;
        }
        if q.datatype() != input_dtype
            || k.datatype() != input_dtype
            || v.datatype() != input_dtype
            || output.datatype() != input_dtype
            || mask
                .as_ref()
                .is_some_and(|mask| mask.datatype() != input_dtype)
        {
            return None;
        }

        let dims = self.dims()?;
        if dims.batch == 0
            || dims.num_heads == 0
            || dims.num_kv_heads == 0
            || dims.q_seq_len == 0
            || dims.kv_seq_len == 0
            || dims.head_dim == 0
        {
            return None;
        }

        let q_meta = TensorMeta::new(&q)?;
        let k_meta = TensorMeta::new(&k)?;
        let v_meta = TensorMeta::new(&v)?;
        let mask_meta = if let Some(mask) = mask.as_ref() {
            Some(TensorMeta::new(mask)?)
        } else {
            None
        };
        let output_meta = TensorMeta::new(&output)?;

        if q_meta.datatype != input_dtype
            || k_meta.datatype != input_dtype
            || v_meta.datatype != input_dtype
            || output_meta.datatype != input_dtype
            || mask_meta
                .as_ref()
                .is_some_and(|mask| mask.datatype != input_dtype)
        {
            return None;
        }

        let caps = KernelDeviceCaps::from_device(&device);
        let selected_variant = select_flash_attention_variant(dims, mask_meta.is_some(), caps);
        // Decode-small kernel only supports f32; force streaming for other dtypes.
        let decode_eligible =
            input_dtype == DataTypeEnum::F32 && selected_variant.decode_block().is_some();
        let decode_candidate = mask_meta.is_none()
            && dims.q_seq_len == 1
            && dims.head_dim == DECODE_HEAD_DIM
            && input_dtype == DataTypeEnum::F32;
        assert!(
            !decode_candidate || selected_variant.decode_block().is_some(),
            "decode attention refused slow fallback: device must support at least {DECODE_SMALL_BLOCK} workgroup invocations on x"
        );
        let decode_meta = if decode_eligible {
            let meta = build_flash_decode_small_meta(
                dims,
                self.scale,
                caps,
                FlashDecodeSmallTensors {
                    q: q_meta.clone(),
                    k: k_meta.clone(),
                    v: v_meta.clone(),
                    mask: mask_meta.as_ref(),
                    output: output_meta.clone(),
                },
            )?;
            assert_eq!(
                Some(meta.decode_block),
                selected_variant.decode_block(),
                "flash attention selector and decode meta disagree"
            );
            Some(meta)
        } else {
            None
        };
        let variant = if decode_eligible {
            selected_variant.kernel_variant()
        } else if flash_streaming_tiled_eligible(dims) {
            FlashAttentionKernelVariant::StreamingTiled
        } else {
            FlashAttentionKernelVariant::Streaming
        };
        let dispatch_size = match variant {
            FlashAttentionKernelVariant::Streaming => streaming_dispatch_size(
                dims,
                tile_ir_kernels::flash_outputs_per_workgroup(streaming_subgroup_size),
            ),
            FlashAttentionKernelVariant::StreamingTiled => {
                tile_ir_kernels::flash_tiled_dispatch_size(
                    dims,
                    tile_ir_kernels::flash_tiled_outputs_per_workgroup(streaming_subgroup_size),
                    FLASH_STREAMING_TILED_Q_BLOCK,
                )
            }
            FlashAttentionKernelVariant::DecodeSmall => [
                dims.batch
                    .checked_mul(dims.num_heads)
                    .expect("flash decode dispatch overflow"),
                1,
                1,
            ],
            FlashAttentionKernelVariant::DecodeSplitPartials
            | FlashAttentionKernelVariant::DecodeSplitReduce => {
                unreachable!("split decode kernels are built as a sequence")
            }
        };
        if dispatch_size
            .iter()
            .any(|dim| *dim > device.limits().max_compute_workgroups_per_dimension)
        {
            return None;
        }

        if let Some(meta) = decode_meta.filter(|meta| meta.tiled) {
            let rows = meta
                .dims
                .batch
                .checked_mul(meta.dims.num_heads)
                .expect("flash split decode row overflow");
            let scratch_elements =
                rows as u64 * meta.split_blocks as u64 * (DECODE_HEAD_DIM as u64 + 2);
            let scratch_buffer = device.create_buffer(
                scratch_elements * std::mem::size_of::<f32>() as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            );
            let params = [meta.active_kv_len, 0, 0, 0];
            let params_buffer = device.create_buffer_init(
                bytemuck::cast_slice(&params),
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            let partial_dispatch = [
                rows.checked_mul(meta.split_blocks)
                    .expect("flash split decode partial dispatch overflow"),
                1,
                1,
            ];
            let reduce_dispatch = [rows, 1, 1];
            let layout = tile_ir_kernels::linear_storage_layout();

            let partial_key = flash_decode_module_key_for_variant(
                FlashAttentionKernelVariant::DecodeSplitPartials,
                Some(workgroup_shape),
                partial_dispatch,
                input_dtype,
                self.scale,
                &meta,
            );
            let partial_buffers = vec![
                q.buffer().clone(),
                k.buffer().clone(),
                v.buffer().clone(),
                scratch_buffer.clone(),
                params_buffer,
            ];
            let partial_layout = layout.clone();
            let partial_kernel = kernel_backend::dynamic_kernel_from_hashed_ir(
                device.kernel_cache(),
                flash_attention_module_cache(),
                "flash_attention_decode_split_partials",
                partial_key,
                partial_buffers,
                partial_dispatch,
                move || {
                    let mut kb = tile_ir::KernelBuilder::<()>::new();
                    let q_ref = tile_ir::KernelTensorRef::new((), partial_layout.clone());
                    let k_ref = tile_ir::KernelTensorRef::new((), partial_layout.clone());
                    let v_ref = tile_ir::KernelTensorRef::new((), partial_layout.clone());
                    let scratch_ref = tile_ir::KernelTensorRef::new((), partial_layout.clone());
                    let params_ref = tile_ir::KernelTensorRef::new((), partial_layout);
                    tile_ir_kernels::flash_decode_split_partials(
                        &mut kb,
                        q_ref,
                        k_ref,
                        v_ref,
                        scratch_ref,
                        params_ref,
                        meta,
                    )?;
                    Some(kb.finish().0)
                },
            )?;

            let reduce_key = flash_decode_module_key_for_variant(
                FlashAttentionKernelVariant::DecodeSplitReduce,
                Some(workgroup_shape),
                reduce_dispatch,
                input_dtype,
                self.scale,
                &meta,
            );
            let reduce_buffers = vec![scratch_buffer, output.buffer().clone()];
            let reduce_layout = layout.clone();
            let reduce_kernel = kernel_backend::dynamic_kernel_from_hashed_ir(
                device.kernel_cache(),
                flash_attention_module_cache(),
                "flash_attention_decode_split_reduce",
                reduce_key,
                reduce_buffers,
                reduce_dispatch,
                move || {
                    let mut kb = tile_ir::KernelBuilder::<()>::new();
                    let scratch_ref = tile_ir::KernelTensorRef::new((), reduce_layout.clone());
                    let output_ref = tile_ir::KernelTensorRef::new((), reduce_layout);
                    tile_ir_kernels::flash_decode_split_reduce(
                        &mut kb,
                        scratch_ref,
                        output_ref,
                        meta,
                    )?;
                    Some(kb.finish().0)
                },
            )?;

            return Some(kernel_backend::DirectKernel::sequence(
                "flash_attention_decode_split",
                vec![partial_kernel, reduce_kernel],
            ));
        }

        let kernel_label = match variant {
            FlashAttentionKernelVariant::Streaming => "flash_attention",
            FlashAttentionKernelVariant::StreamingTiled => "flash_attention_tiled",
            FlashAttentionKernelVariant::DecodeSmall => "flash_attention_decode",
            FlashAttentionKernelVariant::DecodeSplitPartials => {
                "flash_attention_decode_split_partials"
            }
            FlashAttentionKernelVariant::DecodeSplitReduce => "flash_attention_decode_split_reduce",
        };
        let module_key = if let Some(meta) = decode_meta.as_ref() {
            flash_decode_module_key(
                Some(workgroup_shape),
                dispatch_size,
                input_dtype,
                self.scale,
                meta,
            )
        } else {
            let cache_variant = kernel_backend::KernelVariantKey::with_payload::<
                FlashAttentionDirectKernelVariant,
            >(|state| {
                variant.hash(state);
                self.scale.to_bits().hash(state);
                streaming_subgroup_size.hash(state);
                self.causal.hash(state);
            });
            self.kernel_module_key_with_dispatch(
                cache_variant,
                Some(workgroup_shape),
                dispatch_size,
                inputs,
            )
        };

        let _ = output_index; // Bindings are derived from the kernel IR.
        let layout = tile_ir_kernels::linear_storage_layout();
        let q_buffer = q.buffer().clone();
        let k_buffer = k.buffer().clone();
        let v_buffer = v.buffer().clone();
        let mask_buffer = mask.as_ref().map(|m| m.buffer().clone());
        let output_buffer = output.buffer().clone();
        let scale = self.scale;
        let causal = self.causal;
        let q_tile_meta = q_meta.tile.clone();
        let k_tile_meta = k_meta.tile.clone();
        let v_tile_meta = v_meta.tile.clone();
        let mask_tile_meta = mask_meta.clone().map(|meta| meta.tile);
        let output_tile_meta = output_meta.tile.clone();

        // Hoist params-buffer upload (decode path only) and the buffer-list
        // collection OUTSIDE the IR-build closure so cache hits skip the
        // entire IR construction. The closure runs only on cache miss.
        let mut buffers: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(6);
        buffers.push(q_buffer.clone());
        buffers.push(k_buffer.clone());
        buffers.push(v_buffer.clone());
        if let Some(mask_buf) = mask_buffer.as_ref() {
            buffers.push(mask_buf.clone());
        }
        buffers.push(output_buffer.clone());
        if let Some(meta) = decode_meta {
            let params = [meta.active_kv_len, 0, 0, 0];
            let params_buffer = device.create_buffer_init(
                bytemuck::cast_slice(&params),
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            );
            buffers.push(params_buffer);
        }

        kernel_backend::dynamic_kernel_from_hashed_ir(
            device.kernel_cache(),
            flash_attention_module_cache(),
            kernel_label,
            module_key,
            buffers,
            dispatch_size,
            move || {
                let mut kb = tile_ir::KernelBuilder::<()>::new();
                let q_ref = tile_ir::KernelTensorRef::new((), layout.clone());
                let k_ref = tile_ir::KernelTensorRef::new((), layout.clone());
                let v_ref = tile_ir::KernelTensorRef::new((), layout.clone());
                let mask_ref = mask_buffer
                    .as_ref()
                    .map(|_| tile_ir::KernelTensorRef::new((), layout.clone()));
                let output_ref = tile_ir::KernelTensorRef::new((), layout.clone());
                if let Some(meta) = decode_meta {
                    let params_ref = tile_ir::KernelTensorRef::new((), layout.clone());
                    tile_ir_kernels::flash_decode_small(
                        &mut kb, q_ref, k_ref, v_ref, output_ref, params_ref, meta,
                    )
                } else {
                    let stream_meta = tile_ir_kernels::FlashAttentionMeta {
                        dims,
                        scale: tile_ir::F32Bits::new(scale),
                        q_meta: q_tile_meta,
                        k_meta: k_tile_meta,
                        v_meta: v_tile_meta,
                        mask_meta: mask_tile_meta,
                        output_meta: output_tile_meta,
                        dispatch_size,
                        causal,
                    };
                    match variant {
                        FlashAttentionKernelVariant::StreamingTiled => {
                            dispatch_streaming_tiled_flash_attention(
                                &mut kb,
                                q_ref,
                                k_ref,
                                v_ref,
                                mask_ref,
                                output_ref,
                                stream_meta,
                                input_dtype,
                                streaming_subgroup_size,
                            )
                        }
                        _ => dispatch_streaming_flash_attention(
                            &mut kb,
                            q_ref,
                            k_ref,
                            v_ref,
                            mask_ref,
                            output_ref,
                            stream_meta,
                            input_dtype,
                            streaming_subgroup_size,
                        ),
                    }
                }?;
                Some(kb.finish().0)
            },
        )
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs.last().unwrap().clone()
    }

    fn name(&self) -> String {
        format!(
            "flash_attention_{}_{}x{}x{}x{}_by_{}x{}",
            self.input_dtype,
            self.q_shape[0],
            self.q_shape[1],
            self.q_shape[2],
            self.q_shape[3],
            self.k_shape[1],
            self.k_shape[2],
        )
    }
}
