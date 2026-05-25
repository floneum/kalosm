use std::{
    fmt,
    hash::{Hash, Hasher},
};

use crate::{
    DataTypeEnum, Device, Layout, Tensor, TensorData,
    compute_graph::NodeIndex,
    kernel_selection::{
        Axis, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, eq, multiple_of, range,
    },
    matmul::MatMulOperation,
    mir::{
        inputs::MirValue,
        kernel_backend,
        kernel_backend::DirectKernel,
        operation::Operation,
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout,
            tile_storage_write_with_direct_layout,
        },
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
    nary_direct::apply_single_input_elementwise_expr,
    nary_wise::{NaryExpr, NaryOp, NaryOperation},
};
use fusor_gguf::GgmlType;
use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use rustc_hash::FxHasher;

use super::{QMatMulDirectPipelineKey, QMatrix, dequantize::DequantizeOperation};

mod kernel;
mod paired;
#[cfg(test)]
mod tests;

/// Hash the qmatmul format and (M, K, N) into `state`. Shared by the cache
/// keys at every qmatmul-direct call site so they stay in lockstep.
fn hash_qmatmul_shape(
    state: &mut FxHasher,
    format: tile_ir::GgmlQuantFormat,
    m: u32,
    k: u32,
    n: u32,
) {
    format.hash(state);
    m.hash(state);
    k.hash(state);
    n.hash(state);
}

/// Hash the dispatch tuple and input/output layout's offset/shape/strides
/// into `state`. Shared by qmatmul-direct cache keys whose layout shape
/// participates in the key.
fn hash_qmatmul_dispatch_layouts(
    state: &mut FxHasher,
    dispatch_size: [u32; 3],
    input_layout: &Layout,
    output_layout: &Layout,
) {
    dispatch_size.hash(state);
    input_layout.offset().hash(state);
    input_layout.shape().hash(state);
    input_layout.strides().hash(state);
    output_layout.offset().hash(state);
    output_layout.shape().hash(state);
    output_layout.strides().hash(state);
}

/// Build a qmatmul-direct cache key in either the operation-bound or
/// module-only path. Both arms wrap the same `KernelVariantKey<M>(payload)`;
/// the unbound arm additionally hashes `outer` into the module key.
fn qmatmul_direct_module_key<M: 'static>(
    payload_hash: impl Fn(&mut FxHasher),
    outer_hash: impl Fn(&mut FxHasher),
    dispatch_size: [u32; 3],
    operation_key: Option<(&dyn Operation, &[MirValue])>,
) -> kernel_backend::KernelCacheKey {
    let cache_variant = kernel_backend::KernelVariantKey::with_payload::<M>(payload_hash);
    match operation_key {
        Some((operation, operation_inputs)) => operation.kernel_cache_key_with_dispatch(
            cache_variant,
            None,
            dispatch_size,
            operation_inputs,
        ),
        None => kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
            cache_variant.hash(state);
            outer_hash(&mut *state);
        }),
    }
}

const QMAT_M: Axis<0> = Axis;
const QMAT_K: Axis<1> = Axis;
const QMAT_N: Axis<2> = Axis;

/// (BM, BN) tile dimensions for a cooperative-matrix qmatmul tile. BK is
/// pinned to 32 (cooperative-matrix MMA shape: 8x8x8 along K, 4 lanes per
/// subgroup) so it isn't carried in the struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct QCoopTile {
    pub bm: u32,
    pub bn: u32,
}

impl QCoopTile {
    pub(super) const fn new(bm: u32, bn: u32) -> Self {
        Self { bm, bn }
    }
}

/// Chosen kernel path for a direct quantized matmul.
///
/// - [`SingleRow`](Self::SingleRow) / [`Q5SmallSingleRow`](Self::Q5SmallSingleRow):
///   single-row qgemv specializations; the [`QCoopTile`] payload is not used.
/// - [`Q8Wide`](Self::Q8Wide): Q8_0-specific tile path that shares the IR of
///   [`Tile`](Self::Tile) `{tile=(64, 128)}` but lives in its own cache slot
///   (selector rule gates on Q8_0 + N>=8192 + 32 KB workgroup storage).
/// - [`Tile`](Self::Tile) `{cached: true}` uses the precomputed
///   `[n/64, m/64, 1]` dispatch + storage3 cached-pipeline fast path
///   (previously `Tile64x64Cached`); `cached: false` is the IR-build path
///   used by every other tile geometry including the unaligned catch-all.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum QMatmulPath {
    SingleRow,
    Q5SmallSingleRow,
    Q8Wide(QCoopTile),
    Tile { tile: QCoopTile, cached: bool },
}

struct QMatmulDirectFastKernelVariant;
struct QMatmulDirectEpilogueKernelVariant;
struct QMatmulPairedKernelVariant;
struct QMatmulPairedExtrasKernelVariant;
struct QMatmulPairedDenseFallbackKernelVariant;

const QMATMUL_DIRECT_KERNEL_GENERATION: u64 = 0x514D_4154_4D49_5831;

#[derive(Clone, Copy, Debug)]
struct QMatmulDirectCtx {
    format: tile_ir::GgmlQuantFormat,
    y_supports_coop: bool,
}

fn qmatmul_direct_selector() -> ShapeSelector<3, QMatmulDirectCtx, QMatmulPath> {
    /// Helper to build a `Tile` variant with a given tile + cached flag.
    const fn tile(bm: u32, bn: u32, cached: bool) -> QMatmulPath {
        QMatmulPath::Tile {
            tile: QCoopTile::new(bm, bn),
            cached,
        }
    }
    ShapeSelector::new()
        .rule(
            QMatmulPath::Q5SmallSingleRow,
            ShapeRule::<3, QMatmulDirectCtx>::new()
                .axis(QMAT_M, eq(1))
                .axis(QMAT_K, range(0..=1024))
                .axis(QMAT_N, range(0..=4096))
                .when_ctx(|ctx: &QMatmulDirectCtx| ctx.format == tile_ir::GgmlQuantFormat::Q5_0),
        )
        .rule(
            QMatmulPath::SingleRow,
            ShapeRule::new().axis(QMAT_M, eq(1)),
        )
        .rule(
            QMatmulPath::Q8Wide(QCoopTile::new(64, 128)),
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(64))
                .axis(QMAT_K, range(0..=1024))
                .axis(QMAT_N, multiple_of(128))
                .when(|shape: KernelShape<3>, ctx: &QMatmulDirectCtx, caps| {
                    ctx.format == tile_ir::GgmlQuantFormat::Q8_0
                        && shape[QMAT_N] >= 8192
                        && caps.max_compute_invocations_per_workgroup >= 512
                        && caps.max_compute_workgroup_storage_size >= 32 * 1024
                }),
        )
        .rule(
            tile(128, 128, false),
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(128))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(128))
                .when(|shape, ctx, caps| {
                    qmatmul_coop_rule_supported(shape, ctx, caps)
                        && caps.max_compute_invocations_per_workgroup >= 512
                        && caps.max_compute_workgroup_storage_size >= 32 * 1024
                }),
        )
        .rule(
            tile(128, 64, false),
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(128))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(64))
                .when(qmatmul_coop_rule_supported),
        )
        .rule(
            tile(64, 128, false),
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(64))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(128))
                .when(qmatmul_coop_rule_supported),
        )
        .rule(
            tile(64, 64, true), // Tile64x64Cached
            ShapeRule::<3, QMatmulDirectCtx>::new()
                .axis(QMAT_M, multiple_of(64))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(64))
                .when(
                    |shape: KernelShape<3>, ctx: &QMatmulDirectCtx, caps: KernelDeviceCaps| {
                        ctx.format != tile_ir::GgmlQuantFormat::Q4K
                            && qmatmul_coop_rule_supported(shape, ctx, caps)
                    },
                ),
        )
        .rule(
            tile(64, 32, false),
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(64))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(32))
                .when(
                    |shape: KernelShape<3>, ctx: &QMatmulDirectCtx, caps: KernelDeviceCaps| {
                        ctx.format == tile_ir::GgmlQuantFormat::Q4K
                            && qmatmul_coop_rule_supported(shape, ctx, caps)
                    },
                ),
        )
        .rule(tile(64, 64, false), ShapeRule::new())
}

fn qmatmul_coop_rule_supported(
    shape: KernelShape<3>,
    ctx: &QMatmulDirectCtx,
    _caps: KernelDeviceCaps,
) -> bool {
    ctx.y_supports_coop && shape[QMAT_M] > 1
}

fn select_qmatmul_direct_variant(
    format: tile_ir::GgmlQuantFormat,
    m: u32,
    k: u32,
    n: u32,
    y_supports_coop: bool,
    caps: KernelDeviceCaps,
) -> QMatmulPath {
    let shape = KernelShape::new([m as usize, k as usize, n as usize]);
    let ctx = QMatmulDirectCtx {
        format,
        y_supports_coop,
    };
    qmatmul_direct_selector()
        .select(shape, &ctx, caps)
        .expect("quantized matmul selector has a catch-all rule")
}


fn matmul_m_size(shape: &[usize]) -> u32 {
    shape[shape.len() - 2] as u32
}

fn qmatmul_operation_inputs(
    input: NodeIndex,
    matrix: &QMatrix,
    out_shape: &[usize],
    nodes: &crate::compute_graph::ComputeGraphInner,
) -> Vec<MirValue> {
    let input = nodes.get_result(input).unwrap();
    let q_matrix = matrix.clone();
    let device = input.device();
    let output_tensor = TensorData::new_for_shape(device, out_shape, input.datatype());
    vec![input.into(), q_matrix.into(), output_tensor.into()]
}

fn qmatmul_operation_output(inputs: &[MirValue]) -> MirValue {
    let output_tensor = inputs.last().unwrap().as_tensor().unwrap();
    output_tensor.clone().into()
}

fn qmatmul_shape_key(shape: &[usize]) -> String {
    shape
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join("x")
}

fn qmatmul_operation_name(
    kind: &str,
    input_datatype: DataTypeEnum,
    in_shape: &[usize],
    matrix: &QMatrix,
) -> String {
    format!(
        "q_mat_{kind}_{}_{}_{}_{}",
        input_datatype,
        qmatmul_shape_key(in_shape),
        matrix.datatype,
        qmatmul_shape_key(&matrix.shape)
    )
}

/// Paired-mode configuration on a `QMatMulOperation`. When present, the
/// matmul produces `[gate; up]` columns and applies `epilogue.apply(gate, up,
/// extras…)` to emit one output column per pair — `extras` are per-column
/// broadcast tensors (e.g. bias vectors) the kernel loads at epilogue time.
/// Populated by the resolver's `try_fuse_paired_qmatmul` rule.
#[derive(Debug, Clone)]
pub(crate) struct PairedConfig {
    pub(crate) epilogue: tile_ir_kernels::PairedEpilogue,
    pub(crate) pair_len: usize,
    pub(crate) extras: Vec<NodeIndex>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct ElementwiseEpilogue {
    pub(crate) expression: NaryExpr,
    pub(crate) extras: Vec<NodeIndex>,
    pub(crate) input_datatype: DataTypeEnum,
    pub(crate) output_datatype: DataTypeEnum,
}

#[derive(Debug, Clone)]
pub(crate) struct QMatMulOperation {
    pub(crate) input_datatype: DataTypeEnum,
    pub(crate) input: NodeIndex,
    pub(crate) matrix: QMatrix,
    pub(crate) in_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    /// General single-input element-wise expression applied to each loaded
    /// activation before the dot product.
    pub(crate) pre_element_wise_expr: Option<ElementwiseEpilogue>,
    /// General single-input element-wise expression applied after reduction
    /// and before store. This covers composite unary expressions like GELU
    /// that are not representable as a linear unary chain.
    pub(crate) post_element_wise_expr: Option<ElementwiseEpilogue>,
    /// When `Some`, this operation produces a paired output (`out_shape[-1]`
    /// = `paired.pair_len`, half of `matrix.shape[0]`) and dispatches to the
    /// `qgemv_q4k_paired_*` kernel family. When `None`, it's a plain
    /// quantized matmul with optional pre/post expression epilogues.
    pub(crate) paired: Option<PairedConfig>,
}

impl QMatMulOperation {
    pub(crate) fn new(
        input_datatype: DataTypeEnum,
        input_shape: &[usize],
        input: NodeIndex,
        matrix: QMatrix,
    ) -> Self {
        let last_dim = input_shape.len() - 1;
        let mut out_shape = input_shape.to_vec();
        out_shape[last_dim] = matrix.shape[0];
        assert_eq!(input_shape[last_dim], matrix.shape[1]);
        let out_shape = out_shape.into_boxed_slice();
        QMatMulOperation {
            input_datatype,
            input,
            matrix,
            in_shape: input_shape.into(),
            out_shape,
            pre_element_wise_expr: None,
            post_element_wise_expr: None,
            paired: None,
        }
    }

    /// Build a paired-mode QMatMul that emits one output column per
    /// gate/up pair via the supplied epilogue. `pair_len` must equal
    /// `matrix.shape[0] / 2`; `extras.len()` must equal
    /// `epilogue.extras_count()`. Used by the resolver's paired-fusion
    /// rewrite.
    pub(crate) fn new_paired(
        input_datatype: DataTypeEnum,
        input_shape: &[usize],
        input: NodeIndex,
        matrix: QMatrix,
        pair_len: usize,
        epilogue: tile_ir_kernels::PairedEpilogue,
        extras: Vec<NodeIndex>,
    ) -> Self {
        assert_eq!(
            extras.len(),
            epilogue.extras_count(),
            "QMatMulOperation::new_paired: extras.len() must equal epilogue.extras_count()"
        );
        let last_dim = input_shape.len() - 1;
        let mut out_shape = input_shape.to_vec();
        out_shape[last_dim] = pair_len;
        assert_eq!(input_shape[last_dim], matrix.shape[1]);
        assert_eq!(matrix.shape[0], pair_len * 2);
        let out_shape = out_shape.into_boxed_slice();
        QMatMulOperation {
            input_datatype,
            input,
            matrix,
            in_shape: input_shape.into(),
            out_shape,
            pre_element_wise_expr: None,
            post_element_wise_expr: None,
            paired: Some(PairedConfig {
                epilogue,
                pair_len,
                extras,
            }),
        }
    }

    fn m_size(&self) -> u32 {
        matmul_m_size(&self.in_shape)
    }

    fn n_size(&self) -> u32 {
        self.matrix.shape[0] as u32
    }
}

pub(crate) struct DirectKernelTensors<'a> {
    pub input: &'a TensorData,
    pub matrix: &'a QMatrix,
    pub pre_extra_tensors: &'a [&'a TensorData],
    pub post_extra_tensors: &'a [&'a TensorData],
    pub output: &'a TensorData,
}

pub(crate) enum QMatMulKernelPlan {
    EmptyOutput,
    Kernels(Vec<DirectKernel>),
}

impl QMatMulKernelPlan {
    fn from_kernels(kernels: Vec<DirectKernel>) -> Option<Self> {
        (!kernels.is_empty()).then_some(Self::Kernels(kernels))
    }

    pub(crate) fn dispatch_count(&self) -> usize {
        match self {
            Self::EmptyOutput => 0,
            Self::Kernels(kernels) => kernels.len(),
        }
    }

    pub(crate) fn into_kernels(self) -> Vec<DirectKernel> {
        match self {
            Self::EmptyOutput => Vec::new(),
            Self::Kernels(kernels) => kernels,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct QMatMulLoweringError {
    operation: String,
}

impl QMatMulLoweringError {
    fn new(operation: String) -> Self {
        Self { operation }
    }
}

impl fmt::Display for QMatMulLoweringError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "QMatMul lowering produced no kernel plan for {}",
            self.operation
        )
    }
}

pub(crate) struct DirectKernelChains<'a> {
    pub pre_expr: Option<&'a ElementwiseEpilogue>,
    pub post_expr: Option<&'a ElementwiseEpilogue>,
}

fn qmatmul_post_expr_is_column_add(expr: &ElementwiseEpilogue) -> bool {
    if expr.extras.len() != 1
        || expr.input_datatype != DataTypeEnum::F32
        || expr.output_datatype != DataTypeEnum::F32
    {
        return false;
    }
    let NaryExpr::Op { children, function } = &expr.expression else {
        return false;
    };
    if function.op != NaryOp::Add || children.len() != 2 {
        return false;
    }
    let is_input = |child: &NaryExpr, expected| {
        matches!(
            child,
            NaryExpr::IndexedInput { input_idx, indices }
                if *input_idx == expected && NaryExpr::is_elementwise_indices(indices)
        )
    };
    (is_input(&children[0], 0) && is_input(&children[1], 1))
        || (is_input(&children[0], 1) && is_input(&children[1], 0))
}

fn qmatmul_variant_supports_coop_acc_init(
    variant: QMatmulPath,
    m: u32,
    k: u32,
    n: u32,
    y_supports_coop: bool,
) -> bool {
    if !y_supports_coop || m <= 1 || !k.is_multiple_of(32) {
        return false;
    }
    let tile = match variant {
        QMatmulPath::Q8Wide(tile) | QMatmulPath::Tile { tile, .. } => tile,
        QMatmulPath::Q5SmallSingleRow | QMatmulPath::SingleRow => return false,
    };
    m.is_multiple_of(tile.bm) && n.is_multiple_of(tile.bn)
}


enum QmatmulExtraStorage {
    Column(tile_ir::tile::Storage<tile_ir::F32, 1>),
    Pointwise(tile_ir::tile::Storage<tile_ir::F32, 2>),
}

impl QmatmulExtraStorage {
    fn as_extra(&self) -> tile_ir_kernels::QmatmulExtra<'_> {
        match self {
            Self::Column(storage) => tile_ir_kernels::QmatmulExtra::Column(storage),
            Self::Pointwise(storage) => tile_ir_kernels::QmatmulExtra::Pointwise(storage),
        }
    }
}

fn qmatrix_direct_quant_format(matrix: &QMatrix) -> Option<tile_ir::GgmlQuantFormat> {
    Some(match matrix.datatype() {
        GgmlType::Q4_0 => tile_ir::GgmlQuantFormat::Q4_0,
        GgmlType::Q4_1 => tile_ir::GgmlQuantFormat::Q4_1,
        GgmlType::Q5_0 => tile_ir::GgmlQuantFormat::Q5_0,
        GgmlType::Q5_1 => tile_ir::GgmlQuantFormat::Q5_1,
        GgmlType::Q8_0 => tile_ir::GgmlQuantFormat::Q8_0,
        GgmlType::Q8_1 => tile_ir::GgmlQuantFormat::Q8_1,
        GgmlType::Q2K => tile_ir::GgmlQuantFormat::Q2K,
        GgmlType::Q3K => tile_ir::GgmlQuantFormat::Q3K,
        GgmlType::Q4K => tile_ir::GgmlQuantFormat::Q4K,
        GgmlType::Q5K => tile_ir::GgmlQuantFormat::Q5K,
        GgmlType::Q6K => tile_ir::GgmlQuantFormat::Q6K,
        GgmlType::Q8K => tile_ir::GgmlQuantFormat::Q8K,
        GgmlType::F16 | GgmlType::F32 => return None,
    })
}

#[allow(clippy::too_many_arguments)]
fn cached_qmatmul_direct_kernel(
    kernel_name: &str,
    matrix: &QMatrix,
    pipeline_key: &QMatMulDirectPipelineKey,
    input: &TensorData,
    output: &TensorData,
    dispatch_size: [u32; 3],
) -> Option<DirectKernel> {
    let pipeline = matrix
        .direct_pipeline_cache()
        .write()
        .get(pipeline_key)
        .cloned()?;
    Some(
        kernel_backend::DirectKernel::from_prepared_three_buffer_pipeline(
            kernel_name.to_owned(),
            pipeline,
            input.buffer().clone(),
            matrix.buffer().clone(),
            output.buffer().clone(),
            dispatch_size,
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn qmatmul_direct_kernel_from_ir(
    device: &Device,
    cached_kernel_name: String,
    kernel_name: String,
    cache_key: kernel_backend::KernelCacheKey,
    matrix: &QMatrix,
    pipeline_key: QMatMulDirectPipelineKey,
    input: &TensorData,
    pre_extra_tensors: &[&TensorData],
    post_extra_tensors: &[&TensorData],
    output: &TensorData,
    dispatch_size: [u32; 3],
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<DirectKernel> {
    if !pre_extra_tensors.is_empty() || !post_extra_tensors.is_empty() {
        let buffers = std::iter::once(input.buffer().clone())
            .chain(std::iter::once(matrix.buffer().clone()))
            .chain(
                pre_extra_tensors
                    .iter()
                    .map(|tensor| tensor.buffer().clone()),
            )
            .chain(
                post_extra_tensors
                    .iter()
                    .map(|tensor| tensor.buffer().clone()),
            )
            .chain(std::iter::once(output.buffer().clone()));
        return kernel_backend::dynamic_kernel_from_ir(
            device.kernel_cache(),
            kernel_name,
            cache_key,
            build_ir,
            buffers,
            dispatch_size,
        );
    }
    if let Some(kernel) = cached_qmatmul_direct_kernel(
        &cached_kernel_name,
        matrix,
        &pipeline_key,
        input,
        output,
        dispatch_size,
    ) {
        return Some(kernel);
    }
    let pipeline = kernel_backend::three_buffer_pipeline_from_ir(
        device.kernel_cache(),
        &kernel_name,
        cache_key,
        build_ir,
    )?;
    let pipeline = matrix
        .direct_pipeline_cache()
        .write()
        .get_or_insert(pipeline_key, || pipeline.clone())
        .clone();
    Some(
        kernel_backend::DirectKernel::from_prepared_three_buffer_pipeline(
            kernel_name,
            pipeline,
            input.buffer().clone(),
            matrix.buffer().clone(),
            output.buffer().clone(),
            dispatch_size,
        ),
    )
}

fn split_workgroups_2d(
    total_workgroups: u32,
    max_workgroups_per_dimension: u32,
) -> Option<[u32; 2]> {
    if total_workgroups == 0 {
        return Some([1, 1]);
    }

    let max_workgroups_per_dimension = max_workgroups_per_dimension.max(1);
    let x = total_workgroups.min(max_workgroups_per_dimension);
    let y = total_workgroups.div_ceil(x);
    (y <= max_workgroups_per_dimension).then_some([x, y])
}

fn effective_qmatmul_max_workgroups_per_dimension(limits: &wgpu::Limits) -> u32 {
    limits.max_compute_workgroups_per_dimension.max(1)
}

fn tile_cooperative_store_layout_supported(layout: &tile_ir::Layout) -> bool {
    if !layout.is_affine() || layout.shape().rank() != 2 {
        return false;
    }
    let strides = layout.affine_strides();
    strides[0] == 1 || strides[1] == 1
}

/// Output columns per workgroup for the direct qgemv path, by (format, K, N).
/// Each branch below corresponds to one rule of the former `QgemvColsVariant`
/// selector — collapsed to its numeric value here.
fn qgemv_cols_per_workgroup_for_direct(format: tile_ir::GgmlQuantFormat, k: u32, n: u32) -> u32 {
    use tile_ir::GgmlQuantFormat::{Q4K, Q5_0, Q6K, Q8_0};
    // Q4K specializations.
    if format == Q4K && k <= 4096 && (4096..8192).contains(&n) {
        return 4; // was Q4KSmallWide4
    }
    if format == Q4K && k <= 4096 && n >= 8192 {
        return 8; // was Q4KSmallWide8
    }
    if format == Q4K && n <= 4096 && k > 4096 {
        return 8; // was Q4KLargeNarrow8
    }
    // Q6K specializations.
    if format == Q6K && k <= 4096 && n >= 8192 {
        return 8; // was Q6KSmallWide8
    }
    if format == Q6K && n <= 4096 && k > 4096 {
        return 4; // was Q6KLargeNarrow4
    }
    // Q8_0 wide.
    if format == Q8_0 && k <= 1024 && n >= 8192 {
        return 32; // was Q8WideAccelerated32
    }
    // FormatAccelerated: Q5_0 mid (K,N both 2048..=4096), Q4K/Q6K general,
    // or Q5_0 large (K*N >= 4M). Delegates to the format-aware helper.
    let q5_mid = format == Q5_0 && (2048..=4096).contains(&k) && (2048..=4096).contains(&n);
    let q5_large = format == Q5_0
        && (k as u64)
            .checked_mul(n as u64)
            .is_some_and(|elements| elements >= 4 * 1024 * 1024);
    if q5_mid || format == Q4K || format == Q6K || q5_large {
        return tile_ir_kernels::qgemv_cols_per_workgroup_for_shape(format, k, n);
    }
    if format == Q5_0 && k <= 1024 && n <= 4096 {
        return 8; // was Q5Small8
    }
    4 // was Default4
}

/// Public re-export of [`qmatmul_m_pad_target`] for crate-internal callers
/// outside this module (e.g. the fused `q_mat_mul_*` helpers on `Tensor`).
pub(crate) fn qmatmul_m_pad_target_pub(m: usize, n: usize) -> Option<usize> {
    qmatmul_m_pad_target(m, n)
}

/// When `M` (= `input_shape[R-2]`) is unaligned and would land on the
/// catch-all `Tile64x64` variant without `coop_acc_init`, return the M
/// padding target that lets `qmatmul_variant_supports_coop_acc_init` fire.
/// Returns `None` if no padding helps (M already aligned, M == 1, or
/// matrix N is too small for any coop tile).
fn qmatmul_m_pad_target(m: usize, n: usize) -> Option<usize> {
    // SingleRow path handles M == 1 specially; don't pad.
    if m <= 1 {
        return None;
    }
    // Already aligned to 64 — selector will already pick a coop variant.
    if m.is_multiple_of(64) {
        return None;
    }
    // Coop variants all need N % 64 == 0 at minimum.
    if !n.is_multiple_of(64) {
        return None;
    }
    // Prefer 128-aligned padding when N % 128 == 0 so we can hit
    // Tile128x128 / Tile128x64.
    let pad = if n.is_multiple_of(128) { 128 } else { 64 };
    let padded = m.div_ceil(pad) * pad;
    Some(padded)
}

impl Tensor {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        match self.datatype() {
            DataTypeEnum::F16 => {
                return self
                    .cast_to(DataTypeEnum::F32)
                    .q_mat_mul(other)
                    .cast_to(DataTypeEnum::F16);
            }
            DataTypeEnum::F32 => {}
            DataTypeEnum::U32 => panic!("q_mat_mul requires f32/f16 tensors"),
        }

        if self.rank() < 2 {
            return self.add_q_mat_mul(other);
        }
        let in_shape = self.shape();
        let m_axis = self.rank() - 2;
        let m = in_shape[m_axis];
        let n = other.shape()[0];
        let Some(padded_m) = qmatmul_m_pad_target(m, n) else {
            return self.add_q_mat_mul(other);
        };

        // Build padded input shape: replace the M dim with padded_m.
        let mut padded_shape = in_shape.to_vec();
        padded_shape[m_axis] = padded_m;

        // Resize writes zeros outside the copied region so the trailing
        // `padded_m - m` rows contribute nothing to the dot product.
        let padded_input = self.resize(padded_shape);

        // Run the aligned matmul.
        let padded_out = padded_input.add_q_mat_mul(other);

        // Narrow the output back to the caller's M along dim R-2 via
        // a restride view. All other dims are full-size, so this is a
        // pure layout change (no copy).
        let out_shape = padded_out.shape();
        let specs: Vec<crate::StrideSpec> = (0..padded_out.rank())
            .map(|i| {
                if i == m_axis {
                    crate::StrideSpec::dim(i, m)
                } else {
                    crate::StrideSpec::dim(i, out_shape[i])
                }
            })
            .collect();
        padded_out.restride(specs)
    }
}
