use std::sync::OnceLock;

use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;

use crate::{
    DataTypeEnum,
    compute_graph::NodeIndex,
    kernel_selection::{
        Axis, DimConstraint, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, eq, range,
    },
    mir::kernel_backend,
    tensor::TensorData,
};

mod kernel;
#[cfg(test)]
mod tests;

const DECODE_SMALL_BLOCK: u32 = 128;
const DECODE_MEDIUM_BLOCK: u32 = 512;
const DECODE_LARGE_BLOCK: u32 = 1024;
const DECODE_HEAD_DIM: u32 = 128;
const FLASH_ATTENTION_MODULE_CACHE_SIZE: usize = 128;
/// Hardware subgroup sizes the streaming flash kernel emits IR for. The
/// kernel layout assumes one subgroup per output dim and uses subgroup
/// reductions across a `SIZE`-wide KV chunk, so the runtime subgroup width
/// must match one of these exactly.
const FLASH_STREAMING_SUBGROUP_SIZES: &[u32] = &[4, 8, 16, 32, 64];

fn flash_attention_module_cache() -> &'static kernel_backend::ModuleCache {
    static CACHE: OnceLock<kernel_backend::ModuleCache> = OnceLock::new();
    CACHE.get_or_init(|| kernel_backend::module_cache(FLASH_ATTENTION_MODULE_CACHE_SIZE))
}

#[allow(clippy::too_many_arguments)]
fn dispatch_streaming_flash_attention(
    kb: &mut tile_ir::KernelBuilder<()>,
    q: tile_ir::KernelTensorRef<()>,
    k: tile_ir::KernelTensorRef<()>,
    v: tile_ir::KernelTensorRef<()>,
    mask: Option<tile_ir::KernelTensorRef<()>>,
    output: tile_ir::KernelTensorRef<()>,
    meta: tile_ir_kernels::FlashAttentionMeta,
    input_dtype: DataTypeEnum,
    subgroup_size: u32,
) -> Option<()> {
    macro_rules! emit {
        ($element:ty, $size:literal) => {
            tile_ir_kernels::flash_attention::<$element, $size, _>(kb, q, k, v, mask, output, meta)
        };
    }
    match (input_dtype, subgroup_size) {
        (DataTypeEnum::F32, 4) => emit!(tile_ir::F32, 4),
        (DataTypeEnum::F32, 8) => emit!(tile_ir::F32, 8),
        (DataTypeEnum::F32, 16) => emit!(tile_ir::F32, 16),
        (DataTypeEnum::F32, 32) => emit!(tile_ir::F32, 32),
        (DataTypeEnum::F32, 64) => emit!(tile_ir::F32, 64),
        (DataTypeEnum::F16, 4) => emit!(tile_ir::F16, 4),
        (DataTypeEnum::F16, 8) => emit!(tile_ir::F16, 8),
        (DataTypeEnum::F16, 16) => emit!(tile_ir::F16, 16),
        (DataTypeEnum::F16, 32) => emit!(tile_ir::F16, 32),
        (DataTypeEnum::F16, 64) => emit!(tile_ir::F16, 64),
        _ => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_streaming_tiled_flash_attention(
    kb: &mut tile_ir::KernelBuilder<()>,
    q: tile_ir::KernelTensorRef<()>,
    k: tile_ir::KernelTensorRef<()>,
    v: tile_ir::KernelTensorRef<()>,
    mask: Option<tile_ir::KernelTensorRef<()>>,
    output: tile_ir::KernelTensorRef<()>,
    meta: tile_ir_kernels::FlashAttentionMeta,
    input_dtype: DataTypeEnum,
    subgroup_size: u32,
) -> Option<()> {
    macro_rules! emit {
        ($element:ty, $size:literal) => {
            tile_ir_kernels::flash_attention_tiled::<
                $element,
                $size,
                { FLASH_STREAMING_TILED_Q_BLOCK },
                _,
            >(kb, q, k, v, mask, output, meta)
        };
    }
    match (input_dtype, subgroup_size) {
        (DataTypeEnum::F32, 4) => emit!(tile_ir::F32, 4),
        (DataTypeEnum::F32, 8) => emit!(tile_ir::F32, 8),
        (DataTypeEnum::F32, 16) => emit!(tile_ir::F32, 16),
        (DataTypeEnum::F32, 32) => emit!(tile_ir::F32, 32),
        (DataTypeEnum::F32, 64) => emit!(tile_ir::F32, 64),
        (DataTypeEnum::F16, 4) => emit!(tile_ir::F16, 4),
        (DataTypeEnum::F16, 8) => emit!(tile_ir::F16, 8),
        (DataTypeEnum::F16, 16) => emit!(tile_ir::F16, 16),
        (DataTypeEnum::F16, 32) => emit!(tile_ir::F16, 32),
        (DataTypeEnum::F16, 64) => emit!(tile_ir::F16, 64),
        _ => None,
    }
}

/// Returns true when the per-shape gating for the tiled (Q-batched) streaming
/// kernel is satisfied. Decode (q_seq_len < threshold) keeps the existing
/// streaming kernel because the per-workgroup K-cache load offers no reuse
/// when there's only one query.
pub(crate) fn flash_streaming_tiled_eligible(dims: FlashAttentionDims) -> bool {
    dims.q_seq_len >= FLASH_STREAMING_TILED_MIN_Q
        && dims
            .head_dim
            .is_multiple_of(FLASH_STREAMING_TILED_HEAD_DIM_ALIGN)
}

fn streaming_dispatch_size(dims: FlashAttentionDims, outputs_per_workgroup: u32) -> [u32; 3] {
    [
        dims.head_dim.div_ceil(outputs_per_workgroup),
        dims.batch
            .checked_mul(dims.num_heads)
            .and_then(|value| value.checked_mul(dims.q_seq_len))
            .expect("flash attention row dispatch overflow"),
        1,
    ]
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum FlashAttentionKernelVariant {
    Streaming,
    StreamingTiled,
    DecodeSmall,
}

/// Q-block size used by the tiled (Q-batched) flash attention kernel: each
/// workgroup processes this many contiguous query rows, sharing one K/V
/// workgroup-memory load across them.
pub(crate) const FLASH_STREAMING_TILED_Q_BLOCK: u32 = 8;
/// Minimum query sequence length to switch to the tiled kernel. Below this
/// the per-workgroup setup overhead dominates the K/V reuse win.
pub(crate) const FLASH_STREAMING_TILED_MIN_Q: u32 = 128;
/// Required head_dim alignment for the tiled kernel. Apple subgroup width is
/// 32, but we round-robin K loads in groups of 8 (the subgroup-warp width on
/// Metal); enforce 8-alignment so the cooperative K-cache load issues stay
/// well-formed without per-lane tail handling.
pub(crate) const FLASH_STREAMING_TILED_HEAD_DIM_ALIGN: u32 = 8;

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

    // Metal currently miscomputes the non-tiled 512/1024-thread decode
    // variants on some GQA shapes. The 128-thread tiled path has the same
    // semantics and passes the reference tests, so prefer it on Metal.
    if caps.backend == wgpu::Backend::Metal {
        return decode_block_supported(DecodeBlock::Small, caps).then_some(DecodeBlock::Small);
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
    pub(crate) causal: bool,
}

pub(crate) struct FlashAttentionInputs<'a> {
    pub(crate) q: NodeIndex,
    pub(crate) k: NodeIndex,
    pub(crate) v: NodeIndex,
    pub(crate) mask: Option<NodeIndex>,
    pub(crate) q_shape: &'a [usize],
    pub(crate) k_shape: &'a [usize],
    pub(crate) v_shape: &'a [usize],
    pub(crate) scale: f32,
    pub(crate) input_dtype: DataTypeEnum,
    pub(crate) causal: bool,
}

impl FlashAttentionOperation {
    pub(crate) fn new(inputs: FlashAttentionInputs<'_>) -> Self {
        let FlashAttentionInputs {
            q,
            k,
            v,
            mask,
            q_shape,
            k_shape,
            v_shape,
            scale,
            input_dtype,
            causal,
        } = inputs;
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

        if causal {
            assert!(
                mask.is_none(),
                "causal flash attention cannot accept an additive mask"
            );
            assert_eq!(
                q_shape[2], k_shape[2],
                "causal flash attention requires q_seq_len == kv_seq_len, got {} vs {}",
                q_shape[2], k_shape[2]
            );
        }

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
            causal,
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

struct FlashDecodeSmallTensors<'a> {
    q: TensorMeta,
    k: TensorMeta,
    v: TensorMeta,
    mask: Option<&'a TensorMeta>,
    output: TensorMeta,
}

fn build_flash_decode_small_meta(
    dims: FlashAttentionDims,
    scale: f32,
    caps: KernelDeviceCaps,
    tensors: FlashDecodeSmallTensors<'_>,
) -> Option<FlashDecodeSmallMeta> {
    let FlashDecodeSmallTensors {
        q: q_meta,
        k: k_meta,
        v: v_meta,
        mask: mask_meta,
        output: output_meta,
    } = tensors;
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
