use std::hash::Hash;

use crate::{
    DataTypeEnum, Device, Layout, Tensor, TensorData,
    compute_graph::NodeIndex,
    kernel_selection::{
        Axis, CooperativeMatrixKind, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, eq,
        multiple_of, range,
    },
    matmul::{CoopTile, MatMulOperation},
    mir::{
        inputs::MirValue,
        kernel_backend,
        kernel_backend::DirectKernel,
        operation::{DirectKernelLoweringError, DirectKernelPlan, Operation},
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout,
            tile_storage_read_with_direct_layout_typed, tile_storage_write_with_direct_layout,
            tile_storage_write_with_direct_layout_typed,
        },
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
    nary_direct::{apply_multi_input_elementwise_expr, apply_single_input_elementwise_expr},
    nary_wise::{NaryExpr, NaryOp},
};
use fusor_gguf::GgmlType;
use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use rustc_hash::FxHasher;

use super::{QMatMulDirectPipelineKey, QMatrix, QMatrixStorageLayout};

mod kernel;
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
/// standalone path. Both arms wrap the same `KernelVariantKey<M>(payload)`;
/// the unbound arm additionally hashes `outer` into the key.
fn qmatmul_direct_cache_key<M: 'static>(
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

const QMATMUL_COOP_BK: u32 = 32;

/// Chosen kernel path for a direct quantized matmul.
///
/// - [`SingleRow`](Self::SingleRow) / [`Q5SmallSingleRow`](Self::Q5SmallSingleRow):
///   single-row qgemv specializations; the [`CoopTile`] payload is not used.
/// - [`Q8Wide`](Self::Q8Wide): Q8_0-specific tile path that shares the IR of
///   [`Tile`](Self::Tile) `{tile=(64, 128, 32)}` but lives in its own cache slot
///   (selector rule gates on Q8_0 + N>=8192 + 32 KB workgroup storage).
/// - [`Tile`](Self::Tile) `{cached: true}` uses the precomputed
///   `[n/64, m/64, 1]` dispatch + storage3 cached-pipeline fast path
///   (previously `Tile64x64Cached`); `cached: false` is the IR-build path.
/// - [`Workgroup`](Self::Workgroup) uses the non-subgroup fallback.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum QMatmulPath {
    Workgroup,
    SingleRow,
    Q5SmallSingleRow,
    Q8Wide(CoopTile),
    Tile { tile: CoopTile, cached: bool },
}

struct QMatmulDirectEpilogueKernelVariant;

const QMATMUL_DIRECT_KERNEL_GENERATION: u64 = 0x514D_4154_4D49_5832;
#[derive(Clone, Copy, Debug)]
struct QMatmulDirectCtx {
    format: tile_ir::GgmlQuantFormat,
}

fn qmatmul_coop_supported(caps: KernelDeviceCaps) -> bool {
    fixed_subgroup_runtime_supported(caps)
        && caps
            .cooperative_matrix
            .supports(CooperativeMatrixKind::F32F32M8N8K8)
        && subgroup_workgroup_size_supported(caps, 2)
}

fn qmatmul_path_is_single_row_qgemv(path: QMatmulPath) -> bool {
    matches!(path, QMatmulPath::SingleRow | QMatmulPath::Q5SmallSingleRow)
}

#[cfg(test)]
fn qmatmul_path_requires_coop(path: QMatmulPath) -> bool {
    matches!(path, QMatmulPath::Q8Wide(_) | QMatmulPath::Tile { .. })
}

fn subgroup_runtime_supported(caps: KernelDeviceCaps) -> bool {
    caps.subgroups_supported
        && caps.min_subgroup_size > 0
        && caps.max_subgroup_size >= caps.min_subgroup_size
}

fn fixed_subgroup_runtime_supported(caps: KernelDeviceCaps) -> bool {
    subgroup_runtime_supported(caps) && caps.min_subgroup_size == caps.max_subgroup_size
}

fn subgroup_workgroup_size_supported(caps: KernelDeviceCaps, subgroups: u32) -> bool {
    subgroups
        .checked_mul(caps.max_subgroup_size)
        .is_some_and(|block| {
            block <= caps.max_compute_invocations_per_workgroup
                && block <= caps.max_compute_workgroup_size_x
        })
}

fn qgemv_subgroup_supported(
    format: tile_ir::GgmlQuantFormat,
    k: u32,
    n: u32,
    caps: KernelDeviceCaps,
) -> bool {
    let subgroups = tile_ir_kernels::qgemv_subgroups_per_workgroup_for_shape(format, k, n);
    subgroup_runtime_supported(caps) && subgroup_workgroup_size_supported(caps, subgroups)
}

fn qmatmul_coop_tile_subgroup_groups(tile: CoopTile) -> u32 {
    (tile.bm / 32) * (tile.bn / 32)
}

fn qmatmul_coop_tile_supported(tile: CoopTile, caps: KernelDeviceCaps) -> bool {
    qmatmul_coop_supported(caps)
        && subgroup_workgroup_size_supported(caps, qmatmul_coop_tile_subgroup_groups(tile))
}

fn qmatmul_direct_selector() -> ShapeSelector<3, QMatmulDirectCtx, QMatmulPath> {
    /// Helper to build a `Tile` variant with a given tile + cached flag.
    const fn tile(bm: u32, bn: u32, cached: bool) -> QMatmulPath {
        QMatmulPath::Tile {
            tile: CoopTile::new(bm, bn, QMATMUL_COOP_BK),
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
                .when(|shape: KernelShape<3>, ctx: &QMatmulDirectCtx, caps| {
                    ctx.format.is_q5_0_family()
                        && qgemv_subgroup_supported(
                            ctx.format,
                            shape[QMAT_K] as u32,
                            shape[QMAT_N] as u32,
                            caps,
                        )
                }),
        )
        .rule(
            QMatmulPath::SingleRow,
            ShapeRule::new().axis(QMAT_M, eq(1)).when(
                |shape: KernelShape<3>, ctx: &QMatmulDirectCtx, caps| {
                    qgemv_subgroup_supported(
                        ctx.format,
                        shape[QMAT_K] as u32,
                        shape[QMAT_N] as u32,
                        caps,
                    )
                },
            ),
        )
        .rule(
            QMatmulPath::Q8Wide(CoopTile::new(64, 128, QMATMUL_COOP_BK)),
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(64))
                .axis(QMAT_K, range(0..=1024))
                .axis(QMAT_N, multiple_of(128))
                .when(|shape: KernelShape<3>, ctx: &QMatmulDirectCtx, caps| {
                    let tile = CoopTile::new(64, 128, QMATMUL_COOP_BK);
                    ctx.format.is_q8_0_family()
                        && qmatmul_direct_tile_supported(shape, caps, tile)
                        && shape[QMAT_N] >= 8192
                        && caps.max_compute_workgroup_storage_size >= 32 * 1024
                }),
        )
        .rule(
            tile(128, 128, false),
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(128))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(128))
                .when(|shape, _ctx, caps| {
                    qmatmul_direct_tile_supported(
                        shape,
                        caps,
                        CoopTile::new(128, 128, QMATMUL_COOP_BK),
                    ) && caps.max_compute_workgroup_storage_size >= 32 * 1024
                }),
        )
        .rule(
            tile(128, 64, false),
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(128))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(64))
                .when(|shape, _ctx, caps| {
                    qmatmul_direct_tile_supported(
                        shape,
                        caps,
                        CoopTile::new(128, 64, QMATMUL_COOP_BK),
                    )
                }),
        )
        .rule(
            tile(64, 128, false),
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(64))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(128))
                .when(|shape, _ctx, caps| {
                    qmatmul_direct_tile_supported(
                        shape,
                        caps,
                        CoopTile::new(64, 128, QMATMUL_COOP_BK),
                    )
                }),
        )
        .rule(
            tile(64, 64, true), // Tile64x64Cached
            ShapeRule::<3, QMatmulDirectCtx>::new()
                .axis(QMAT_M, multiple_of(64))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(64))
                .when(
                    |shape: KernelShape<3>, ctx: &QMatmulDirectCtx, caps: KernelDeviceCaps| {
                        !ctx.format.is_q4k_family()
                            && qmatmul_direct_tile_supported(
                                shape,
                                caps,
                                CoopTile::new(64, 64, QMATMUL_COOP_BK),
                            )
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
                        ctx.format.is_q4k_family()
                            && qmatmul_direct_tile_supported(
                                shape,
                                caps,
                                CoopTile::new(64, 32, QMATMUL_COOP_BK),
                            )
                    },
                ),
        )
        .rule(
            tile(64, 64, false),
            ShapeRule::new().when(|shape, _ctx, caps| {
                qmatmul_direct_tile_supported(shape, caps, CoopTile::new(64, 64, QMATMUL_COOP_BK))
            }),
        )
        .rule(QMatmulPath::Workgroup, ShapeRule::new())
}

fn qmatmul_direct_tile_supported(
    shape: KernelShape<3>,
    caps: KernelDeviceCaps,
    tile: CoopTile,
) -> bool {
    qmatmul_coop_tile_supported(tile, caps) && shape[QMAT_M] > 1
}

fn select_qmatmul_direct_variant(
    format: tile_ir::GgmlQuantFormat,
    m: u32,
    k: u32,
    n: u32,
    caps: KernelDeviceCaps,
) -> QMatmulPath {
    let shape = KernelShape::new([m as usize, k as usize, n as usize]);
    let ctx = QMatmulDirectCtx { format };
    qmatmul_direct_selector()
        .select(shape, &ctx, caps)
        .expect("quantized matmul selector has a catch-all rule")
}

fn matmul_m_size(shape: &[usize]) -> u32 {
    // Rank-1 activations are a single matrix row.
    if shape.len() < 2 {
        return 1;
    }
    shape[shape.len() - 2] as u32
}

fn qmatmul_operation_inputs(
    input: NodeIndex,
    matrix: &QMatrix,
    out_shape: &[usize],
    m_pad: Option<usize>,
    nodes: &crate::compute_graph::ComputeGraphInner,
) -> Vec<MirValue> {
    let input = nodes.get_result(input).unwrap();
    let q_matrix = matrix.clone();
    let device = input.device();
    let output_tensor = match m_pad {
        // M-padded lowering writes `padded_m` rows: allocate the padded
        // buffer but expose it cropped to the real `M` (the trailing rows
        // are slack the kernel scribbles into).
        Some(padded_m) => {
            let m_axis = out_shape.len() - 2;
            let mut padded_shape = out_shape.to_vec();
            padded_shape[m_axis] = padded_m;
            let padded = TensorData::new_for_shape(device, &padded_shape, input.datatype());
            TensorData::new_from_parts(
                device,
                padded.buffer().clone(),
                Layout::from_parts(
                    0,
                    out_shape.into(),
                    Layout::continuous_strides(&padded_shape),
                ),
                input.datatype(),
            )
        }
        None => TensorData::new_for_shape(device, out_shape, input.datatype()),
    };
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
    /// Matrix-column offsets for accumulator values consumed by
    /// `post_element_wise_expr`. Empty means the default single accumulator at
    /// the output column. Non-empty is only lowered by the single-row qgemv
    /// path.
    pub(crate) post_accumulator_offsets: Box<[u32]>,
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
            post_accumulator_offsets: Box::new([]),
        }
    }

    fn m_size(&self) -> u32 {
        matmul_m_size(&self.in_shape)
    }

    /// The padded `M` this operation's lowering will use, when M-padding
    /// applies: the activation is copied into a zero-padded scratch tensor so
    /// the selector can pick a coop tile variant, and the output buffer is
    /// over-allocated to hold the padded rows (exposed cropped to the real
    /// `M`). Epilogue-carrying operations never pad — the padded views don't
    /// carry epilogue extras, so fusion declines those ops instead (see
    /// [`Self::supports_elementwise_epilogue_fusion`]).
    pub(crate) fn m_pad_target(&self, caps: KernelDeviceCaps) -> Option<usize> {
        if self.pre_element_wise_expr.is_some()
            || self.post_element_wise_expr.is_some()
            || !self.post_accumulator_offsets.is_empty()
        {
            return None;
        }
        if self.in_shape.len() < 2 {
            return None;
        }
        let m = self.in_shape[self.in_shape.len() - 2];
        let n = self.out_shape[self.out_shape.len() - 1];
        qmatmul_m_pad_target_for_caps(m, n, caps)
    }

    /// Whether the resolver may fold a general element-wise expression into
    /// this operation's pre/post epilogue. Dense-storage (F32/F16) matrices
    /// lower through the epilogue-less dense kernel, and an operation whose
    /// lowering will M-pad keeps element-wise work as separate kernels.
    pub(crate) fn supports_elementwise_epilogue_fusion(&self, device: &Device) -> bool {
        if qmatrix_direct_quant_format(&self.matrix).is_none() {
            return false;
        }
        if self.in_shape.len() < 2 {
            return true;
        }
        let m = self.in_shape[self.in_shape.len() - 2];
        let n = self.out_shape[self.out_shape.len() - 1];
        qmatmul_m_pad_target_for_caps(m, n, KernelDeviceCaps::from_device(device)).is_none()
    }

    fn n_size(&self) -> u32 {
        self.out_shape[self.out_shape.len() - 1] as u32
    }

    pub(crate) fn supports_indexed_post_accumulator_offsets(
        &self,
        device: &Device,
        output_shape: &[usize],
        accumulator_offsets: &[u32],
    ) -> bool {
        if self.input_datatype != DataTypeEnum::F32 {
            return false;
        }
        let Some(format) = qmatrix_direct_quant_format(&self.matrix) else {
            return false;
        };
        let Some((m, k)) = flattened_matrix_shape_u32(&self.in_shape) else {
            return false;
        };
        let Some((y_m, n)) = flattened_matrix_shape_u32(output_shape) else {
            return false;
        };
        if m != y_m {
            return false;
        }
        let Some(&matrix_n) = self.matrix.shape().first() else {
            return false;
        };
        let Ok(matrix_n) = u32::try_from(matrix_n) else {
            return false;
        };
        let Some(max_accumulator_offset) = accumulator_offsets.iter().copied().max() else {
            return false;
        };

        let caps = KernelDeviceCaps::from_device(device);
        let variant = select_qmatmul_direct_variant(format, m, k, n, caps);
        qmatmul_custom_accumulator_offsets_supported(
            format,
            variant,
            m,
            k,
            n,
            matrix_n,
            max_accumulator_offset,
            effective_qmatmul_max_workgroups_per_dimension(&device.limits()),
        )
    }

    /// Number of distinct storage-buffer bindings this operation needs:
    /// activation input, the quantized matrix, every pre/post epilogue extra,
    /// and the output. Mirrors the binding layout produced by
    /// [`QMatMulOperation::inputs`].
    pub(crate) fn binding_count(&self) -> usize {
        let pre = self
            .pre_element_wise_expr
            .as_ref()
            .map_or(0, |epilogue| epilogue.extras.len());
        let post = self
            .post_element_wise_expr
            .as_ref()
            .map_or(0, |epilogue| epilogue.extras.len());
        // input + matrix + extras + output
        3 + pre + post
    }

    /// Whether the fused operation's bindings fit within the device's
    /// per-shader storage-buffer budget. Fusing an epilogue that would exceed
    /// the budget is infeasible, so the resolver must leave it as separate
    /// kernels.
    pub(crate) fn fits_binding_budget(&self, device: &Device) -> bool {
        let limits = device.limits();
        let budget = (limits.max_storage_buffers_per_shader_stage as usize)
            .min(limits.max_bindings_per_bind_group as usize);
        self.binding_count() <= budget
    }
}

pub(crate) struct DirectKernelTensors<'a> {
    pub input: &'a TensorData,
    pub matrix: &'a QMatrix,
    pub pre_extra_tensors: &'a [&'a TensorData],
    pub post_extra_tensors: &'a [&'a TensorData],
    pub output: &'a TensorData,
}

pub(crate) struct DirectKernelChains<'a> {
    pub pre_expr: Option<&'a ElementwiseEpilogue>,
    pub post_expr: Option<&'a ElementwiseEpilogue>,
    pub post_accumulator_offsets: &'a [u32],
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

fn qmatmul_custom_accumulator_offsets_cover_output(
    m: u32,
    n: u32,
    matrix_n: u32,
    max_accumulator_offset: u32,
) -> bool {
    m == 1 && n.checked_add(max_accumulator_offset) == Some(matrix_n)
}

fn qmatmul_variant_supports_coop_acc_init(
    variant: QMatmulPath,
    m: u32,
    k: u32,
    n: u32,
    y_supports_coop: bool,
) -> bool {
    if !y_supports_coop || m <= 1 {
        return false;
    }
    let tile = match variant {
        QMatmulPath::Q8Wide(tile) | QMatmulPath::Tile { tile, .. } => tile,
        QMatmulPath::Workgroup | QMatmulPath::Q5SmallSingleRow | QMatmulPath::SingleRow => {
            return false;
        }
    };
    if !k.is_multiple_of(tile.bk) {
        return false;
    }
    m.is_multiple_of(tile.bm) && n.is_multiple_of(tile.bn)
}

fn qmatmul_qgemv_dispatch_supported(
    format: tile_ir::GgmlQuantFormat,
    k: u32,
    n: u32,
    max_workgroups_per_dimension: u32,
) -> bool {
    let qgemv_cols_per_workgroup = qgemv_cols_per_workgroup_for_direct(format, k, n);
    let qgemv_workgroups = n.div_ceil(qgemv_cols_per_workgroup);
    split_workgroups_2d(qgemv_workgroups, max_workgroups_per_dimension).is_some()
}

#[allow(clippy::too_many_arguments)]
fn qmatmul_custom_accumulator_offsets_supported(
    format: tile_ir::GgmlQuantFormat,
    variant: QMatmulPath,
    m: u32,
    k: u32,
    n: u32,
    matrix_n: u32,
    max_accumulator_offset: u32,
    max_workgroups_per_dimension: u32,
) -> bool {
    qmatmul_custom_accumulator_offsets_cover_output(m, n, matrix_n, max_accumulator_offset)
        && qmatmul_path_is_single_row_qgemv(variant)
        && qmatmul_qgemv_dispatch_supported(format, k, n, max_workgroups_per_dimension)
}

fn flattened_matrix_shape_u32(shape: &[usize]) -> Option<(u32, u32)> {
    if shape.len() < 2 || shape.contains(&0) {
        return None;
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
    Some((rows.try_into().ok()?, (*shape.last()?).try_into().ok()?))
}

enum QmatmulExtraStorage {
    Column(tile_ir::tile::Storage),
    Pointwise(tile_ir::tile::Storage),
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
        GgmlType::Q4_0 if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q4_0Native
        }
        GgmlType::Q4_0 => tile_ir::GgmlQuantFormat::Q4_0,
        GgmlType::Q4_1 => tile_ir::GgmlQuantFormat::Q4_1,
        GgmlType::Q5_0 if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q5_0Native
        }
        GgmlType::Q5_0 => tile_ir::GgmlQuantFormat::Q5_0,
        GgmlType::Q5_1 => tile_ir::GgmlQuantFormat::Q5_1,
        GgmlType::Q8_0 if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q8_0Native
        }
        GgmlType::Q8_0 => tile_ir::GgmlQuantFormat::Q8_0,
        GgmlType::Q8_1 => tile_ir::GgmlQuantFormat::Q8_1,
        GgmlType::Q2K => tile_ir::GgmlQuantFormat::Q2K,
        GgmlType::Q3K => tile_ir::GgmlQuantFormat::Q3K,
        GgmlType::Q4K if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q4KNative
        }
        GgmlType::Q4K => tile_ir::GgmlQuantFormat::Q4K,
        GgmlType::Q5K if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q5KNative
        }
        GgmlType::Q5K => tile_ir::GgmlQuantFormat::Q5K,
        GgmlType::Q6K if matrix.storage_layout() == QMatrixStorageLayout::Native => {
            tile_ir::GgmlQuantFormat::Q6KNative
        }
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
            None,
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
    let (pipeline, cached) = kernel_backend::three_buffer_pipeline_from_ir(
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
            Some(cached),
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
    let max_workgroups_per_dimension = max_workgroups_per_dimension.max(1);
    let x = total_workgroups.min(max_workgroups_per_dimension);
    let y = total_workgroups.div_ceil(x.max(1)).max(1);
    (y <= max_workgroups_per_dimension).then_some([x, y])
}

fn effective_qmatmul_max_workgroups_per_dimension(limits: &wgpu::Limits) -> u32 {
    limits.max_compute_workgroups_per_dimension.max(1)
}

/// Output columns per workgroup for the direct qgemv path, by (format, K, N).
fn qgemv_cols_per_workgroup_for_direct(format: tile_ir::GgmlQuantFormat, k: u32, n: u32) -> u32 {
    // Q4K specializations.
    if format.is_q4k_family() && k <= 4096 && (4096..8192).contains(&n) {
        return 4; // was Q4KSmallWide4
    }
    if format.is_q4k_family() && k <= 4096 && n >= 8192 {
        return 8; // was Q4KSmallWide8
    }
    if format.is_q4k_family() && n <= 4096 && k > 4096 {
        return 8; // was Q4KLargeNarrow8
    }
    // Q6K specializations.
    if format.is_q6k_family() && k <= 4096 && n >= 8192 {
        return 8; // was Q6KSmallWide8
    }
    if format.is_q6k_family() && n <= 4096 && k > 4096 {
        return 4; // was Q6KLargeNarrow4
    }
    // Q8_0 wide.
    if format.is_q8_0_family() && k <= 1024 && n >= 8192 {
        return 32; // was Q8WideAccelerated32
    }
    // FormatAccelerated: Q5_0 mid (K,N both 2048..=4096), Q4K/Q6K general,
    // or Q5_0 large (K*N >= 4M). Delegates to the format-aware helper.
    let q5_mid =
        format.is_q5_0_family() && (2048..=4096).contains(&k) && (2048..=4096).contains(&n);
    let q5_large = format.is_q5_0_family()
        && (k as u64)
            .checked_mul(n as u64)
            .is_some_and(|elements| elements >= 4 * 1024 * 1024);
    if q5_mid || format.is_q4k_family() || format.is_q6k_family() || q5_large {
        return tile_ir_kernels::qgemv_cols_per_workgroup_for_shape(format, k, n);
    }
    if format.is_q5_0_family() && k <= 1024 && n <= 4096 {
        return 8; // was Q5Small8
    }
    4 // was Default4
}

fn qmatmul_m_pad_target_for_caps(m: usize, n: usize, caps: KernelDeviceCaps) -> Option<usize> {
    if !qmatmul_coop_supported(caps) {
        return None;
    }
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
            DataTypeEnum::F16 | DataTypeEnum::F32 => {}
            DataTypeEnum::U32 => panic!("q_mat_mul requires f32/f16 tensors"),
        }
        // M-padding for the coop tile variants is a lowering decision: the
        // recognized operation pads its own activation scratch and output
        // slack (see `QMatMulOperation::m_pad_target`).
        self.add_q_mat_mul(other)
    }
}
