use std::{
    hash::Hash,
    sync::{Arc, OnceLock},
};

use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;

use crate::{
    DataTypeEnum,
    compute_graph::{ComputeGraphInner, NodeIndex},
    kernel_selection::{
        Axis, DimConstraint, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, eq, range,
    },
    mir::{
        direct_kernel::DirectKernel,
        inputs::MirValue,
        kernel_backend,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    tensor::TensorData,
};

const BLOCK: usize = 256;
const SIMD_WIDTH: usize = 32;
const OUTPUTS_PER_WORKGROUP: usize = BLOCK / SIMD_WIDTH;
const DECODE_SMALL_BLOCK: u32 = 128;
const DECODE_MEDIUM_BLOCK: u32 = 512;
const DECODE_LARGE_BLOCK: u32 = 1024;
const DECODE_HEAD_DIM: u32 = 128;
const FLASH_ATTENTION_MODULE_CACHE_SIZE: usize = 128;

fn flash_attention_module_cache() -> &'static kernel_backend::ModuleCache {
    static CACHE: OnceLock<kernel_backend::ModuleCache> = OnceLock::new();
    CACHE.get_or_init(|| kernel_backend::module_cache(FLASH_ATTENTION_MODULE_CACHE_SIZE))
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum FlashAttentionKernelVariant {
    Streaming,
    DecodeSmall,
}

struct FlashAttentionDirectKernelVariant;

const FLASH_Q_SEQ: Axis<3> = Axis;
const FLASH_KV_SEQ: Axis<4> = Axis;
const FLASH_HEAD_DIM: Axis<5> = Axis;

type FlashAttentionDims = tile_ir_kernels::FlashAttentionDims;
type FlashDecodeSmallMeta = tile_ir_kernels::FlashDecodeSmallMeta;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeBlock {
    Small,
    Medium,
    Large,
}

impl DecodeBlock {
    const ALL: [Self; 3] = [Self::Small, Self::Medium, Self::Large];

    fn size(self) -> u32 {
        match self {
            Self::Small => DECODE_SMALL_BLOCK,
            Self::Medium => DECODE_MEDIUM_BLOCK,
            Self::Large => DECODE_LARGE_BLOCK,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlashAttentionSelectedVariant {
    Streaming,
    DecodeSmall(DecodeBlock),
}

impl FlashAttentionSelectedVariant {
    fn kernel_variant(self) -> FlashAttentionKernelVariant {
        match self {
            Self::Streaming => FlashAttentionKernelVariant::Streaming,
            Self::DecodeSmall(_) => FlashAttentionKernelVariant::DecodeSmall,
        }
    }

    fn decode_block(self) -> Option<u32> {
        match self {
            Self::Streaming => None,
            Self::DecodeSmall(block) => Some(block.size()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FlashAttentionSelectionCtx {
    has_mask: bool,
}

fn decode_block_supported(block: DecodeBlock, caps: KernelDeviceCaps) -> bool {
    let size = block.size();
    size <= caps.max_compute_invocations_per_workgroup && size <= caps.max_compute_workgroup_size_x
}

fn choose_decode_block(kv_seq_len: u32, caps: KernelDeviceCaps) -> Option<DecodeBlock> {
    if kv_seq_len == 0 {
        return None;
    }

    let mut largest_supported = None;
    for block in DecodeBlock::ALL {
        if !decode_block_supported(block, caps) {
            continue;
        }
        largest_supported = Some(block);
        if kv_seq_len <= block.size() {
            return Some(block);
        }
    }
    largest_supported
}

fn selected_decode_block_for_shape(
    shape: KernelShape<6>,
    ctx: &FlashAttentionSelectionCtx,
    caps: KernelDeviceCaps,
) -> Option<DecodeBlock> {
    if ctx.has_mask || shape[FLASH_KV_SEQ] == 0 {
        return None;
    }
    choose_decode_block(shape[FLASH_KV_SEQ].try_into().ok()?, caps)
}

fn decode_shape_rule(
    block: DecodeBlock,
    kv_constraint: Option<DimConstraint>,
) -> ShapeRule<6, FlashAttentionSelectionCtx> {
    let mut rule = ShapeRule::new()
        .axis(FLASH_Q_SEQ, eq(1))
        .axis(FLASH_HEAD_DIM, eq(DECODE_HEAD_DIM as usize));
    if let Some(kv_constraint) = kv_constraint {
        rule = rule.axis(FLASH_KV_SEQ, kv_constraint);
    }
    rule.when(move |shape, ctx, caps| {
        selected_decode_block_for_shape(shape, ctx, caps) == Some(block)
    })
}

fn flash_attention_selector()
-> ShapeSelector<6, FlashAttentionSelectionCtx, FlashAttentionSelectedVariant> {
    ShapeSelector::new()
        .rule(
            FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Small),
            decode_shape_rule(
                DecodeBlock::Small,
                Some(range(1..=DECODE_SMALL_BLOCK as usize)),
            ),
        )
        .rule(
            FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Medium),
            decode_shape_rule(
                DecodeBlock::Medium,
                Some(range(
                    (DECODE_SMALL_BLOCK as usize + 1)..=DECODE_MEDIUM_BLOCK as usize,
                )),
            ),
        )
        .rule(
            FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Large),
            decode_shape_rule(
                DecodeBlock::Large,
                Some(range(
                    (DECODE_MEDIUM_BLOCK as usize + 1)..=DECODE_LARGE_BLOCK as usize,
                )),
            ),
        )
        .rule(
            FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Small),
            decode_shape_rule(DecodeBlock::Small, None),
        )
        .rule(
            FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Medium),
            decode_shape_rule(DecodeBlock::Medium, None),
        )
        .rule(
            FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Large),
            decode_shape_rule(DecodeBlock::Large, None),
        )
        .rule(FlashAttentionSelectedVariant::Streaming, ShapeRule::new())
}

fn select_flash_attention_variant(
    dims: FlashAttentionDims,
    has_mask: bool,
    caps: KernelDeviceCaps,
) -> FlashAttentionSelectedVariant {
    let shape = KernelShape::new([
        dims.batch as usize,
        dims.num_heads as usize,
        dims.num_kv_heads as usize,
        dims.q_seq_len as usize,
        dims.kv_seq_len as usize,
        dims.head_dim as usize,
    ]);
    let ctx = FlashAttentionSelectionCtx { has_mask };
    flash_attention_selector()
        .select(shape, &ctx, caps)
        .expect("flash attention selector has a catch-all rule")
}

#[derive(Clone, Debug)]
pub(crate) struct FlashAttentionOperation {
    pub(crate) q: NodeIndex,
    pub(crate) k: NodeIndex,
    pub(crate) v: NodeIndex,
    pub(crate) mask: Option<NodeIndex>,
    pub(crate) out_shape: Box<[usize]>,
    q_shape: Box<[usize]>,
    k_shape: Box<[usize]>,
    scale: f32,
    input_dtype: DataTypeEnum,
}

impl FlashAttentionOperation {
    pub(crate) fn new(
        q: NodeIndex,
        k: NodeIndex,
        v: NodeIndex,
        mask: Option<NodeIndex>,
        q_shape: &[usize],
        k_shape: &[usize],
        v_shape: &[usize],
        scale: f32,
        input_dtype: DataTypeEnum,
    ) -> Self {
        assert_eq!(q_shape.len(), 4, "Q must be rank-4");
        assert_eq!(k_shape.len(), 4, "K must be rank-4");
        assert_eq!(v_shape.len(), 4, "V must be rank-4");
        assert_eq!(q_shape[0], k_shape[0], "Q and K batch dimensions differ");
        assert_eq!(q_shape[0], v_shape[0], "Q and V batch dimensions differ");
        assert_eq!(k_shape[1], v_shape[1], "K and V head dimensions differ");
        assert_eq!(k_shape[2], v_shape[2], "K and V sequence dimensions differ");
        assert_eq!(q_shape[3], k_shape[3], "Q and K head dimensions differ");
        assert_eq!(q_shape[3], v_shape[3], "Q and V head dimensions differ");
        assert!(
            q_shape[1].is_multiple_of(k_shape[1]),
            "Number of Q heads ({}) must be divisible by number of K/V heads ({})",
            q_shape[1],
            k_shape[1]
        );

        Self {
            q,
            k,
            v,
            mask,
            out_shape: q_shape.into(),
            q_shape: q_shape.into(),
            k_shape: k_shape.into(),
            scale,
            input_dtype,
        }
    }

    fn dims(&self) -> Option<FlashAttentionDims> {
        Some(FlashAttentionDims {
            batch: self.q_shape[0].try_into().ok()?,
            num_heads: self.q_shape[1].try_into().ok()?,
            num_kv_heads: self.k_shape[1].try_into().ok()?,
            q_seq_len: self.q_shape[2].try_into().ok()?,
            kv_seq_len: self.k_shape[2].try_into().ok()?,
            head_dim: self.q_shape[3].try_into().ok()?,
        })
    }
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
        let dims = self.dims().expect("flash attention dimensions fit in u32");
        [
            dims.head_dim.div_ceil(OUTPUTS_PER_WORKGROUP as u32),
            dims.batch
                .checked_mul(dims.num_heads)
                .and_then(|value| value.checked_mul(dims.q_seq_len))
                .expect("flash attention row dispatch overflow"),
            1,
        ]
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

        if !device.subgroups_supported()
            || device.min_subgroup_size() > SIMD_WIDTH as u32
            || device.max_subgroup_size() < SIMD_WIDTH as u32
        {
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
                q_meta.clone(),
                k_meta.clone(),
                v_meta.clone(),
                mask_meta.as_ref(),
                output_meta.clone(),
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
        } else {
            FlashAttentionKernelVariant::Streaming
        };
        let dispatch_size = match variant {
            FlashAttentionKernelVariant::Streaming => {
                self.dispatch_size(&WorkgroupShape::new(1, 1, 1), inputs)
            }
            FlashAttentionKernelVariant::DecodeSmall => [
                dims.batch
                    .checked_mul(dims.num_heads)
                    .expect("flash decode dispatch overflow"),
                1,
                1,
            ],
        };
        if dispatch_size
            .iter()
            .any(|dim| *dim > device.limits().max_compute_workgroups_per_dimension)
        {
            return None;
        }

        let kernel_label = match variant {
            FlashAttentionKernelVariant::Streaming => "flash_attention",
            FlashAttentionKernelVariant::DecodeSmall => "flash_attention_decode",
        };
        let cache_variant = kernel_backend::KernelVariantKey::with_payload::<
            FlashAttentionDirectKernelVariant,
        >(|state| {
            variant.hash(state);
            self.scale.to_bits().hash(state);
            if let Some(meta) = decode_meta.as_ref() {
                meta.decode_block.hash(state);
                meta.tiled.hash(state);
            }
        });
        let module_key = self.kernel_module_key_with_dispatch(
            cache_variant,
            Some(workgroup_shape),
            dispatch_size,
            inputs,
        );

        let _ = output_index; // Bindings are derived from the kernel IR.
        let layout = tile_ir_kernels::linear_storage_layout();
        let q_buffer = q.buffer().clone();
        let k_buffer = k.buffer().clone();
        let v_buffer = v.buffer().clone();
        let mask_buffer = mask.as_ref().map(|m| m.buffer().clone());
        let output_buffer = output.buffer().clone();
        let scale = self.scale;
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
            &device,
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
                let result = if let Some(meta) = decode_meta {
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
                    };
                    match input_dtype {
                        DataTypeEnum::F32 => tile_ir_kernels::flash_attention::<tile_ir::F32, _>(
                            &mut kb,
                            q_ref,
                            k_ref,
                            v_ref,
                            mask_ref,
                            output_ref,
                            stream_meta,
                        ),
                        DataTypeEnum::F16 => tile_ir_kernels::flash_attention::<tile_ir::F16, _>(
                            &mut kb,
                            q_ref,
                            k_ref,
                            v_ref,
                            mask_ref,
                            output_ref,
                            stream_meta,
                        ),
                        _ => None,
                    }
                }?;
                let _ = result;
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

#[derive(Clone)]
pub(crate) struct TensorMeta {
    datatype: DataTypeEnum,
    tile: tile_ir_kernels::TensorMeta,
}

impl TensorMeta {
    fn new(tensor: &TensorData) -> Option<Self> {
        let strides = tensor
            .layout()
            .strides()
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let offset = tensor.layout().offset().try_into().ok()?;
        Some(Self {
            datatype: tensor.datatype(),
            tile: tile_ir_kernels::TensorMeta::new(strides, offset),
        })
    }

    fn stride4(&self) -> Option<[u32; 4]> {
        self.tile.strides.as_slice().try_into().ok()
    }
}

fn build_flash_decode_small_meta(
    dims: FlashAttentionDims,
    scale: f32,
    caps: KernelDeviceCaps,
    q_meta: TensorMeta,
    k_meta: TensorMeta,
    v_meta: TensorMeta,
    mask_meta: Option<&TensorMeta>,
    output_meta: TensorMeta,
) -> Option<FlashDecodeSmallMeta> {
    if mask_meta.is_some()
        || dims.q_seq_len != 1
        || dims.head_dim != DECODE_HEAD_DIM
        || dims.kv_seq_len == 0
    {
        return None;
    }
    let decode_block = choose_decode_block(dims.kv_seq_len, caps)?.size();
    let tiled = dims.kv_seq_len > decode_block;

    let groups = dims.num_heads.checked_div(dims.num_kv_heads)?;
    if groups == 0 {
        return None;
    }

    let mut module_dims = dims;
    module_dims.kv_seq_len = decode_block;

    Some(FlashDecodeSmallMeta {
        dims: module_dims,
        scale: tile_ir::F32Bits::new(scale),
        active_kv_len: dims.kv_seq_len,
        decode_block,
        tiled,
        groups,
        q_offset: q_meta.tile.offset,
        k_offset: k_meta.tile.offset,
        v_offset: v_meta.tile.offset,
        output_offset: output_meta.tile.offset,
        q_strides: q_meta.stride4()?,
        k_strides: k_meta.stride4()?,
        v_strides: v_meta.stride4()?,
        output_strides: output_meta.stride4()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor, kernel_selection::assert_selector_generates};

    const TEST_HEAD_DIM: usize = DECODE_HEAD_DIM as usize;

    fn caps(max_compute_invocations_per_workgroup: u32) -> KernelDeviceCaps {
        KernelDeviceCaps {
            subgroups_supported: true,
            cooperative_matrix_supported: false,
            min_subgroup_size: 32,
            max_subgroup_size: 32,
            max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: 64 * 1024,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroups_per_dimension: 65_535,
        }
    }

    fn tensor_meta4() -> TensorMeta {
        TensorMeta {
            datatype: DataTypeEnum::F32,
            tile: tile_ir_kernels::TensorMeta::new(vec![65_536, 8_192, 128, 1], 0),
        }
    }

    fn decode_dims(kv_seq_len: u32) -> FlashAttentionDims {
        FlashAttentionDims {
            batch: 1,
            num_heads: 32,
            num_kv_heads: 8,
            q_seq_len: 1,
            kv_seq_len,
            head_dim: DECODE_HEAD_DIM,
        }
    }

    fn decode_shape(kv_seq_len: usize) -> KernelShape<6> {
        KernelShape::new([1, 32, 8, 1, kv_seq_len, DECODE_HEAD_DIM as usize])
    }

    #[test]
    fn decode_block_choice_uses_smallest_covering_supported_block() {
        assert_eq!(
            choose_decode_block(64, caps(DECODE_LARGE_BLOCK)),
            Some(DecodeBlock::Small)
        );
        assert_eq!(
            choose_decode_block(200, caps(DECODE_LARGE_BLOCK)),
            Some(DecodeBlock::Medium)
        );
        assert_eq!(
            choose_decode_block(600, caps(DECODE_LARGE_BLOCK)),
            Some(DecodeBlock::Large)
        );
        assert_eq!(
            choose_decode_block(600, caps(DECODE_MEDIUM_BLOCK)),
            Some(DecodeBlock::Medium)
        );
        assert_eq!(
            choose_decode_block(DECODE_LARGE_BLOCK + 1, caps(DECODE_LARGE_BLOCK)),
            Some(DecodeBlock::Large)
        );
        assert_eq!(
            choose_decode_block(DECODE_SMALL_BLOCK, caps(DECODE_SMALL_BLOCK - 1)),
            None
        );
    }

    #[test]
    fn decode_small_meta_buckets_dynamic_kv_len() {
        let meta = build_flash_decode_small_meta(
            decode_dims(DECODE_SMALL_BLOCK + 1),
            1.0,
            caps(DECODE_LARGE_BLOCK),
            tensor_meta4(),
            tensor_meta4(),
            tensor_meta4(),
            None,
            tensor_meta4(),
        )
        .unwrap();

        assert_eq!(meta.active_kv_len, DECODE_SMALL_BLOCK + 1);
        assert_eq!(meta.decode_block, DECODE_MEDIUM_BLOCK);
        assert_eq!(meta.dims.kv_seq_len, DECODE_MEDIUM_BLOCK);
        assert!(!meta.tiled);
    }

    #[test]
    fn decode_small_meta_tiles_with_largest_supported_block() {
        let meta = build_flash_decode_small_meta(
            decode_dims(DECODE_MEDIUM_BLOCK + 1),
            1.0,
            caps(DECODE_MEDIUM_BLOCK),
            tensor_meta4(),
            tensor_meta4(),
            tensor_meta4(),
            None,
            tensor_meta4(),
        );

        let meta = meta.unwrap();
        assert_eq!(meta.active_kv_len, DECODE_MEDIUM_BLOCK + 1);
        assert_eq!(meta.decode_block, DECODE_MEDIUM_BLOCK);
        assert_eq!(meta.dims.kv_seq_len, DECODE_MEDIUM_BLOCK);
        assert!(meta.tiled);
    }

    #[test]
    fn decode_small_meta_requires_minimum_workgroup_limit() {
        let meta = build_flash_decode_small_meta(
            decode_dims(DECODE_SMALL_BLOCK),
            1.0,
            caps(DECODE_SMALL_BLOCK - 1),
            tensor_meta4(),
            tensor_meta4(),
            tensor_meta4(),
            None,
            tensor_meta4(),
        );

        assert!(meta.is_none());
    }

    #[test]
    fn flash_attention_selector_selects_decode_block_buckets() {
        let selector = flash_attention_selector();
        let decode_ctx = FlashAttentionSelectionCtx { has_mask: false };
        let masked_ctx = FlashAttentionSelectionCtx { has_mask: true };

        assert_eq!(
            selector.select(decode_shape(64), &decode_ctx, caps(DECODE_LARGE_BLOCK)),
            Some(FlashAttentionSelectedVariant::DecodeSmall(
                DecodeBlock::Small
            ))
        );
        assert_eq!(
            selector.select(decode_shape(200), &decode_ctx, caps(DECODE_LARGE_BLOCK)),
            Some(FlashAttentionSelectedVariant::DecodeSmall(
                DecodeBlock::Medium
            ))
        );
        assert_eq!(
            selector.select(decode_shape(600), &decode_ctx, caps(DECODE_LARGE_BLOCK)),
            Some(FlashAttentionSelectedVariant::DecodeSmall(
                DecodeBlock::Large
            ))
        );
        assert_eq!(
            selector.select(decode_shape(600), &decode_ctx, caps(DECODE_MEDIUM_BLOCK)),
            Some(FlashAttentionSelectedVariant::DecodeSmall(
                DecodeBlock::Medium
            ))
        );
        assert_eq!(
            selector.select(
                decode_shape(DECODE_LARGE_BLOCK as usize + 1),
                &decode_ctx,
                caps(DECODE_LARGE_BLOCK)
            ),
            Some(FlashAttentionSelectedVariant::DecodeSmall(
                DecodeBlock::Large
            ))
        );
        assert_eq!(
            selector.select(decode_shape(200), &decode_ctx, caps(DECODE_SMALL_BLOCK - 1)),
            Some(FlashAttentionSelectedVariant::Streaming)
        );
        assert_eq!(
            selector.select(decode_shape(200), &masked_ctx, caps(DECODE_LARGE_BLOCK)),
            Some(FlashAttentionSelectedVariant::Streaming)
        );
    }

    #[test]
    fn flash_attention_selector_generates_each_variant() {
        let selector = flash_attention_selector();
        let decode_ctx = FlashAttentionSelectionCtx { has_mask: false };
        let streaming_ctx = FlashAttentionSelectionCtx { has_mask: true };
        let cases = [
            (
                FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Small),
                decode_ctx,
                caps(DECODE_SMALL_BLOCK),
            ),
            (
                FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Medium),
                decode_ctx,
                caps(DECODE_MEDIUM_BLOCK),
            ),
            (
                FlashAttentionSelectedVariant::DecodeSmall(DecodeBlock::Large),
                decode_ctx,
                caps(DECODE_LARGE_BLOCK),
            ),
            (
                FlashAttentionSelectedVariant::Streaming,
                streaming_ctx,
                caps(DECODE_LARGE_BLOCK),
            ),
        ];
        assert_selector_generates(&selector, cases);
    }

    type AttentionFixture = Vec<Vec<Vec<Vec<f32>>>>;

    fn attention_fixture(
        heads: usize,
        tokens: usize,
        f: impl Fn(usize, usize, usize) -> f32,
    ) -> AttentionFixture {
        vec![
            (0..heads)
                .map(|head| {
                    (0..tokens)
                        .map(|token| (0..TEST_HEAD_DIM).map(|dim| f(head, token, dim)).collect())
                        .collect()
                })
                .collect(),
        ]
    }

    fn decode_q() -> AttentionFixture {
        attention_fixture(1, 1, |_, _, dim| ((dim % 17) as f32 - 8.0) * 0.0075)
    }

    fn decode_k(kv_len: usize) -> AttentionFixture {
        attention_fixture(1, kv_len, |_, token, dim| {
            let value = ((token * 13 + dim * 7) % 31) as f32 - 15.0;
            value * 0.004
        })
    }

    fn decode_v(kv_len: usize) -> AttentionFixture {
        attention_fixture(1, kv_len, |_, token, dim| {
            let value = ((token * 5 + dim * 11) % 37) as f32 - 18.0;
            0.25 + value * 0.01
        })
    }

    fn decode_q_gqa(num_heads: usize) -> AttentionFixture {
        attention_fixture(num_heads, 1, |head, _, dim| {
            let value = ((head * 19 + dim * 7) % 43) as f32 - 21.0;
            value * 0.003
        })
    }

    fn decode_k_gqa(num_kv_heads: usize, kv_len: usize) -> AttentionFixture {
        attention_fixture(num_kv_heads, kv_len, |kv_head, token, dim| {
            let value = ((kv_head * 23 + token * 13 + dim * 5) % 47) as f32 - 23.0;
            value * 0.0025
        })
    }

    fn decode_v_gqa(num_kv_heads: usize, kv_len: usize) -> AttentionFixture {
        attention_fixture(num_kv_heads, kv_len, |kv_head, token, dim| {
            let value = ((kv_head * 29 + token * 3 + dim * 11) % 53) as f32 - 26.0;
            0.05 + value * 0.004
        })
    }

    fn cpu_decode_reference(q: &[f32], k: &[Vec<f32>], v: &[Vec<f32>], scale: f32) -> Vec<f32> {
        let scores = k
            .iter()
            .map(|key| {
                q.iter()
                    .zip(key)
                    .map(|(q, k)| (*q as f64) * (*k as f64))
                    .sum::<f64>()
                    * scale as f64
            })
            .collect::<Vec<_>>();
        let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denom = scores
            .iter()
            .map(|score| (score - max_score).exp())
            .sum::<f64>();
        let mut output = vec![0.0; TEST_HEAD_DIM];
        for (token, score) in scores.iter().copied().enumerate() {
            let prob = (score - max_score).exp() / denom;
            for (dim, output) in output.iter_mut().enumerate() {
                *output += prob * v[token][dim] as f64;
            }
        }
        output.into_iter().map(|value| value as f32).collect()
    }

    fn decode_max_error(
        num_heads: usize,
        groups: usize,
        q_data: &AttentionFixture,
        k_data: &AttentionFixture,
        v_data: &AttentionFixture,
        scale: f32,
        actual: impl Fn(usize, usize) -> f32,
    ) -> (f32, usize, usize, f32, f32) {
        let mut max_error = 0.0f32;
        let mut max_head = 0usize;
        let mut max_dim = 0usize;
        let mut max_actual = 0.0f32;
        let mut max_expected = 0.0f32;
        for head in 0..num_heads {
            let kv_head = head / groups;
            let expected = cpu_decode_reference(
                &q_data[0][head][0],
                &k_data[0][kv_head],
                &v_data[0][kv_head],
                scale,
            );
            for (dim, expected) in expected.into_iter().enumerate() {
                let actual = actual(head, dim);
                let error = (actual - expected).abs();
                if error > max_error {
                    max_error = error;
                    max_head = head;
                    max_dim = dim;
                    max_actual = actual;
                    max_expected = expected;
                }
            }
        }
        (max_error, max_head, max_dim, max_actual, max_expected)
    }

    #[tokio::test]
    async fn tiled_decode_attention_matches_cpu_reference() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let kv_len = DECODE_LARGE_BLOCK as usize + 1;
        let q_data = decode_q();
        let k_data = decode_k(kv_len);
        let v_data = decode_v(kv_len);
        let scale = 1.0 / f32::sqrt(TEST_HEAD_DIM as f32);

        let q = Tensor::new(&device, &q_data);
        let k = Tensor::new(&device, &k_data);
        let v = Tensor::new(&device, &v_data);
        let output = q.try_flash_attention_direct(&k, &v, scale, None).unwrap();
        let output = output.as_slice().await.unwrap();
        let (max_error, _, max_dim, max_actual, max_expected) =
            decode_max_error(1, 1, &q_data, &k_data, &v_data, scale, |_, dim| {
                output[[0, 0, 0, dim]]
            });
        assert!(
            max_error < 2.0e-4,
            "dim {max_dim}: actual={max_actual} expected={max_expected} error={max_error}"
        );
    }

    #[tokio::test]
    async fn tiled_decode_attention_gqa_matches_cpu_reference() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let num_heads = 32;
        let num_kv_heads = 8;
        let groups = num_heads / num_kv_heads;
        let kv_len = DECODE_LARGE_BLOCK as usize + 1;
        let q_data = decode_q_gqa(num_heads);
        let k_data = decode_k_gqa(num_kv_heads, kv_len);
        let v_data = decode_v_gqa(num_kv_heads, kv_len);
        let scale = 1.0 / f32::sqrt(TEST_HEAD_DIM as f32);

        let q = Tensor::new(&device, &q_data);
        let k = Tensor::new(&device, &k_data);
        let v = Tensor::new(&device, &v_data);
        let output = q.try_flash_attention_direct(&k, &v, scale, None).unwrap();
        let output = output.as_slice().await.unwrap();

        let (max_error, max_head, max_dim, max_actual, max_expected) = decode_max_error(
            num_heads,
            groups,
            &q_data,
            &k_data,
            &v_data,
            scale,
            |head, dim| output[[0, head, 0, dim]],
        );
        assert!(
            max_error < 3.0e-4,
            "head {max_head} dim {max_dim}: actual={max_actual} expected={max_expected} error={max_error}"
        );
    }

    /// Regression test for the non-tiled 512/1024-thread decode blocks.
    /// Before the fix, the per-thread score loop folded its 128 q*k
    /// accumulations into a single deeply-nested Naga expression, which
    /// miscompiled on Metal once the kernel's `workgroup_size` exceeded 128;
    /// the kernel produced all-zero output. The fix emits the dot product as a
    /// shader loop with a function-scope accumulator.
    #[tokio::test]
    async fn decode_gqa_non_tiled_large_blocks_match_cpu_reference() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let num_heads = 32;
        let num_kv_heads = 8;
        let groups = num_heads / num_kv_heads;
        let caps = KernelDeviceCaps::from_device(&device);

        // On devices that support the larger workgroups, 200 uses the 512
        // block and 600 uses the 1024 block.
        for (kv_len, expected_block) in [(200usize, DecodeBlock::Medium), (600, DecodeBlock::Large)]
        {
            if choose_decode_block(kv_len as u32, caps) != Some(expected_block) {
                continue;
            }
            let q_data = decode_q_gqa(num_heads);
            let k_data = decode_k_gqa(num_kv_heads, kv_len);
            let v_data = decode_v_gqa(num_kv_heads, kv_len);
            let scale = 1.0 / f32::sqrt(TEST_HEAD_DIM as f32);

            let q = Tensor::new(&device, &q_data);
            let k = Tensor::new(&device, &k_data);
            let v = Tensor::new(&device, &v_data);
            let output = q.try_flash_attention_direct(&k, &v, scale, None).unwrap();
            let output = output.as_slice().await.unwrap();

            let (max_error, max_head, max_dim, max_actual, max_expected) = decode_max_error(
                num_heads,
                groups,
                &q_data,
                &k_data,
                &v_data,
                scale,
                |head, dim| output[[0, head, 0, dim]],
            );
            assert!(
                max_error < 5.0e-4,
                "kv_len={kv_len} head={max_head} dim={max_dim}: actual={max_actual} expected={max_expected} error={max_error}"
            );
        }
    }

    #[tokio::test]
    async fn streaming_gqa_regression_shape_builds_direct_kernel() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let q_data = vec![
            (0..32)
                .map(|head| {
                    (0..48)
                        .map(|token| {
                            (0..TEST_HEAD_DIM)
                                .map(|dim| {
                                    let value =
                                        ((head * 17 + token * 11 + dim * 5) % 41) as f32 - 20.0;
                                    value * 0.002
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        ];
        let k_data = decode_k_gqa(8, 48);
        let v_data = decode_v_gqa(8, 48);
        let scale = 1.0 / f32::sqrt(TEST_HEAD_DIM as f32);

        let q = Tensor::new(&device, &q_data);
        let k = Tensor::new(&device, &k_data);
        let v = Tensor::new(&device, &v_data);
        let output = q
            .try_flash_attention_direct(&k, &v, scale, None)
            .expect("streaming flash attention direct kernel should build");
        output.materialize().await;
    }
}
