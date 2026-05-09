use crate::{
    DataTypeEnum, Device, Layout, Tensor, TensorData,
    compute_graph::NodeIndex,
    kernel_selection::{
        Axis, KernelDeviceCaps, KernelShape, ShapeRule, ShapeSelector, eq, multiple_of, range,
    },
    matmul::MatMulOperation,
    mir::{
        direct_kernel::DirectKernel,
        inputs::MirValue,
        kernel_backend,
        operation::Operation,
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout,
            tile_storage_write_with_direct_layout,
        },
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
};
use fusor_gguf::GgmlType;
use fusor_tile_ir as tile_ir;

use super::{QMatMulDirectPipelineKey, QMatrix};

const QMAT_M: Axis<0> = Axis;
const QMAT_K: Axis<1> = Axis;
const QMAT_N: Axis<2> = Axis;
const QGEMV_K: Axis<0> = Axis;
const QGEMV_N: Axis<1> = Axis;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QMatmulDirectVariant {
    Q5SmallSingleRow,
    SingleRow,
    Q8Wide64x128,
    Tile128x128,
    Tile128x64,
    Tile64x128,
    Tile64x64Fast,
    Tile64x64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QgemvColsVariant {
    Q4KSmallWide4,
    Q4KSmallWide8,
    Q4KLargeNarrow8,
    Q6KSmallWide8,
    Q6KLargeNarrow4,
    Q8WideAccelerated32,
    FormatAccelerated,
    Q5Small8,
    Default4,
}

#[derive(Clone, Copy, Debug)]
struct QMatmulDirectCtx {
    format: tile_ir::GgmlQuantFormat,
    y_supports_coop: bool,
}

#[derive(Clone, Copy, Debug)]
struct QgemvColsCtx {
    format: tile_ir::GgmlQuantFormat,
}

fn qmatmul_direct_selector() -> ShapeSelector<3, QMatmulDirectCtx, QMatmulDirectVariant> {
    ShapeSelector::new()
        .rule(
            QMatmulDirectVariant::Q5SmallSingleRow,
            ShapeRule::new()
                .axis(QMAT_M, eq(1))
                .axis(QMAT_K, range(0..=1024))
                .axis(QMAT_N, range(0..=4096))
                .when_ctx(|ctx: &QMatmulDirectCtx| ctx.format == tile_ir::GgmlQuantFormat::Q5_0),
        )
        .rule(
            QMatmulDirectVariant::SingleRow,
            ShapeRule::new().axis(QMAT_M, eq(1)),
        )
        .rule(
            QMatmulDirectVariant::Q8Wide64x128,
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
            QMatmulDirectVariant::Tile128x128,
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
            QMatmulDirectVariant::Tile128x64,
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(128))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(64))
                .when(qmatmul_coop_rule_supported),
        )
        .rule(
            QMatmulDirectVariant::Tile64x128,
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(64))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(128))
                .when(qmatmul_coop_rule_supported),
        )
        .rule(
            QMatmulDirectVariant::Tile64x64Fast,
            ShapeRule::new()
                .axis(QMAT_M, multiple_of(64))
                .axis(QMAT_K, multiple_of(32))
                .axis(QMAT_N, multiple_of(64))
                .when(qmatmul_coop_rule_supported),
        )
        .rule(QMatmulDirectVariant::Tile64x64, ShapeRule::new())
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
) -> QMatmulDirectVariant {
    let shape = KernelShape::new([m as usize, k as usize, n as usize]);
    let ctx = QMatmulDirectCtx {
        format,
        y_supports_coop,
    };
    qmatmul_direct_selector()
        .select(shape, &ctx, caps)
        .expect("quantized matmul selector has a catch-all rule")
}

fn qgemv_cols_selector() -> ShapeSelector<2, QgemvColsCtx, QgemvColsVariant> {
    ShapeSelector::new()
        .rule(
            QgemvColsVariant::Q4KSmallWide4,
            ShapeRule::new()
                .axis(QGEMV_K, range(0..=4096))
                .axis(QGEMV_N, range(4096..8192))
                .when_ctx(|ctx: &QgemvColsCtx| ctx.format == tile_ir::GgmlQuantFormat::Q4K),
        )
        .rule(
            QgemvColsVariant::Q4KSmallWide8,
            ShapeRule::new()
                .axis(QGEMV_K, range(0..=4096))
                .axis(QGEMV_N, range(8192..=16384))
                .when_ctx(|ctx: &QgemvColsCtx| ctx.format == tile_ir::GgmlQuantFormat::Q4K),
        )
        .rule(
            QgemvColsVariant::Q4KSmallWide8,
            ShapeRule::new().axis(QGEMV_K, range(0..=4096)).when(
                |shape: KernelShape<2>, ctx: &QgemvColsCtx, _caps| {
                    ctx.format == tile_ir::GgmlQuantFormat::Q4K && shape[QGEMV_N] >= 8192
                },
            ),
        )
        .rule(
            QgemvColsVariant::Q4KLargeNarrow8,
            ShapeRule::new().axis(QGEMV_N, range(0..=4096)).when(
                |shape: KernelShape<2>, ctx: &QgemvColsCtx, _caps| {
                    ctx.format == tile_ir::GgmlQuantFormat::Q4K && shape[QGEMV_K] > 4096
                },
            ),
        )
        .rule(
            QgemvColsVariant::Q6KSmallWide8,
            ShapeRule::new()
                .axis(QGEMV_K, range(0..=4096))
                .axis(QGEMV_N, range(8192..=16384))
                .when_ctx(|ctx: &QgemvColsCtx| ctx.format == tile_ir::GgmlQuantFormat::Q6K),
        )
        .rule(
            QgemvColsVariant::Q6KSmallWide8,
            ShapeRule::new().axis(QGEMV_K, range(0..=4096)).when(
                |shape: KernelShape<2>, ctx: &QgemvColsCtx, _caps| {
                    ctx.format == tile_ir::GgmlQuantFormat::Q6K && shape[QGEMV_N] >= 8192
                },
            ),
        )
        .rule(
            QgemvColsVariant::Q6KLargeNarrow4,
            ShapeRule::new().axis(QGEMV_N, range(0..=4096)).when(
                |shape: KernelShape<2>, ctx: &QgemvColsCtx, _caps| {
                    ctx.format == tile_ir::GgmlQuantFormat::Q6K && shape[QGEMV_K] > 4096
                },
            ),
        )
        .rule(
            QgemvColsVariant::Q8WideAccelerated32,
            ShapeRule::new()
                .axis(QGEMV_K, range(0..=1024))
                .axis(QGEMV_N, range(8192..=16384))
                .when_ctx(|ctx: &QgemvColsCtx| ctx.format == tile_ir::GgmlQuantFormat::Q8_0),
        )
        .rule(
            QgemvColsVariant::Q8WideAccelerated32,
            ShapeRule::new().axis(QGEMV_K, range(0..=1024)).when(
                |shape: KernelShape<2>, ctx: &QgemvColsCtx, _caps| {
                    ctx.format == tile_ir::GgmlQuantFormat::Q8_0 && shape[QGEMV_N] >= 8192
                },
            ),
        )
        .rule(
            QgemvColsVariant::FormatAccelerated,
            ShapeRule::new()
                .axis(QGEMV_K, range(2048..=4096))
                .axis(QGEMV_N, range(2048..=4096))
                .when_ctx(|ctx: &QgemvColsCtx| ctx.format == tile_ir::GgmlQuantFormat::Q5_0),
        )
        .rule(
            QgemvColsVariant::FormatAccelerated,
            ShapeRule::new().when(|shape: KernelShape<2>, ctx: &QgemvColsCtx, _caps| {
                ctx.format == tile_ir::GgmlQuantFormat::Q4K
                    || ctx.format == tile_ir::GgmlQuantFormat::Q6K
                    || (ctx.format == tile_ir::GgmlQuantFormat::Q5_0
                        && shape[QGEMV_K]
                            .checked_mul(shape[QGEMV_N])
                            .is_some_and(|elements| elements >= 4 * 1024 * 1024))
            }),
        )
        .rule(
            QgemvColsVariant::Q5Small8,
            ShapeRule::new()
                .axis(QGEMV_K, range(0..=1024))
                .axis(QGEMV_N, range(0..=4096))
                .when_ctx(|ctx: &QgemvColsCtx| ctx.format == tile_ir::GgmlQuantFormat::Q5_0),
        )
        .rule(QgemvColsVariant::Default4, ShapeRule::new())
}

fn select_qgemv_cols_variant(format: tile_ir::GgmlQuantFormat, k: u32, n: u32) -> QgemvColsVariant {
    qgemv_cols_selector()
        .select(
            KernelShape::new([k as usize, n as usize]),
            &QgemvColsCtx { format },
            KernelDeviceCaps {
                subgroups_supported: false,
                cooperative_matrix_supported: false,
                min_subgroup_size: 0,
                max_subgroup_size: 0,
                max_compute_invocations_per_workgroup: 0,
                max_compute_workgroup_storage_size: 0,
                max_compute_workgroup_size_x: 0,
                max_compute_workgroups_per_dimension: 0,
            },
        )
        .expect("qgemv column selector has a catch-all rule")
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
    let output_tensor = inputs[2].as_tensor().unwrap();
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

#[derive(Debug, Clone)]
pub(crate) struct QMatMulOperation {
    pub(crate) input_datatype: DataTypeEnum,
    pub(crate) input: NodeIndex,
    pub(crate) matrix: QMatrix,
    pub(crate) in_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
}

/// Paired-activation quantized matmul: produces `[gate; up]` columns and applies
/// `activation.apply(gate, up)` to emit one output column per pair. Replaces
/// the SwiGLU-specific operation with a pluggable activation parameter.
#[derive(Debug, Clone)]
pub(crate) struct QMatMulPairedOperation {
    pub(crate) input_datatype: DataTypeEnum,
    pub(crate) input: NodeIndex,
    pub(crate) matrix: QMatrix,
    pub(crate) in_shape: Box<[usize]>,
    pub(crate) out_shape: Box<[usize]>,
    pub(crate) pair_len: usize,
    pub(crate) activation: tile_ir::PairedActivation,
}

impl QMatMulPairedOperation {
    pub(crate) fn new(
        input_datatype: DataTypeEnum,
        input_shape: &[usize],
        input: NodeIndex,
        matrix: QMatrix,
        pair_len: usize,
        activation: tile_ir::PairedActivation,
    ) -> Self {
        let last_dim = input_shape.len() - 1;
        let mut out_shape = input_shape.to_vec();
        out_shape[last_dim] = pair_len;
        assert_eq!(input_shape[last_dim], matrix.shape[1]);
        assert_eq!(matrix.shape[0], pair_len * 2);
        let out_shape = out_shape.into_boxed_slice();
        Self {
            input_datatype,
            input,
            matrix,
            in_shape: input_shape.into(),
            out_shape,
            pair_len,
            activation,
        }
    }

    fn m_size(&self) -> u32 {
        matmul_m_size(&self.in_shape)
    }

    fn direct_quant_format(&self) -> Option<tile_ir::GgmlQuantFormat> {
        match self.matrix.datatype {
            GgmlType::Q4K => Some(tile_ir::GgmlQuantFormat::Q4K),
            _ => None,
        }
    }
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
        }
    }

    fn m_size(&self) -> u32 {
        matmul_m_size(&self.in_shape)
    }

    fn n_size(&self) -> u32 {
        self.matrix.shape[0] as u32
    }

    pub(crate) fn direct_kernel_for_tensors(
        device: &Device,
        input: &TensorData,
        matrix: &QMatrix,
        output: &TensorData,
        kernel_name: impl Into<String>,
    ) -> Option<DirectKernel> {
        if input.datatype() != DataTypeEnum::F32 || output.datatype() != DataTypeEnum::F32 {
            return None;
        }
        if matrix.datatype() == GgmlType::F32 {
            return None;
        }
        let input_rank = input.layout().shape().len();
        if input_rank != output.layout().shape().len() {
            return None;
        }

        let format = qmatrix_direct_quant_format(matrix)?;
        let a_view = flatten_matrix_layout(input.layout())?;
        let y_view = flatten_matrix_layout(output.layout())?;
        let m = a_view.rows;
        let k = a_view.cols;
        let y_m = y_view.rows;
        let n = y_view.cols;
        if m != y_m || k != matrix.shape[1] as u32 || n != matrix.shape[0] as u32 {
            return None;
        }

        let limits = device.limits();
        let caps = KernelDeviceCaps::from_device(device);
        let max_workgroups = limits.max_compute_workgroups_per_dimension;
        let mut qmatmul_workgroups_x = 1;
        let y_supports_coop = tile_cooperative_store_layout_supported(&y_view.layout);
        let variant = select_qmatmul_direct_variant(format, m, k, n, y_supports_coop, caps);
        let fast_dispatch_size = match variant {
            QMatmulDirectVariant::Q5SmallSingleRow | QMatmulDirectVariant::SingleRow => {
                let qgemv_cols_per_workgroup = qgemv_cols_per_workgroup_for_direct(format, k, n);
                let qgemv_workgroups = n.div_ceil(qgemv_cols_per_workgroup);
                let [dispatch_x, _] = split_workgroups_2d(qgemv_workgroups, max_workgroups)?;
                qmatmul_workgroups_x = dispatch_x;
                Some([
                    qmatmul_workgroups_x,
                    qgemv_workgroups.div_ceil(qmatmul_workgroups_x),
                    1,
                ])
            }
            QMatmulDirectVariant::Q8Wide64x128 => Some([n / 128, m / 64, 1]),
            QMatmulDirectVariant::Tile128x128 => Some([n / 128, m / 128, 1]),
            QMatmulDirectVariant::Tile128x64 => Some([n / 64, m / 128, 1]),
            QMatmulDirectVariant::Tile64x128 => Some([n / 128, m / 64, 1]),
            QMatmulDirectVariant::Tile64x64Fast => Some([n / 64, m / 64, 1]),
            QMatmulDirectVariant::Tile64x64 => None,
        };
        let kernel_name = kernel_name.into();
        if let Some(dispatch_size) = fast_dispatch_size {
            if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
                return None;
            }
            let pipeline_key = QMatMulDirectPipelineKey::new(
                matrix.datatype(),
                m,
                k,
                n,
                dispatch_size,
                input.layout(),
                output.layout(),
            );
            if let Some(kernel) = cached_qmatmul_direct_kernel(
                &kernel_name,
                matrix,
                &pipeline_key,
                input,
                output,
                dispatch_size,
            ) {
                return Some(kernel);
            }
            let cache_key = format!(
                "{kernel_name}:direct:{format:?}:m={m}:k={k}:n={n}:dispatch={dispatch_size:?}:{:?}:{:?}",
                input.layout(),
                output.layout()
            );
            if let Some(pipeline) = kernel_backend::storage3_pipeline_from_cached_module(
                device,
                &kernel_name,
                &cache_key,
            ) {
                matrix
                    .direct_pipeline_cache()
                    .write()
                    .get_or_insert(pipeline_key, || pipeline.clone());
                return Some(kernel_backend::storage3_kernel_with_prepared_pipeline(
                    kernel_name.clone(),
                    cache_key,
                    pipeline,
                    input.buffer().clone(),
                    matrix.buffer().clone(),
                    output.buffer().clone(),
                    dispatch_size,
                ));
            }
        }
        let ir = tile_ir::tile::build(move |phase| {
            let a = tile_storage_read_with_direct_layout(phase, a_view);
            let b = phase.quantized_matrix(format, k, n);
            let y = tile_storage_write_with_direct_layout(phase, y_view);
            match variant {
                QMatmulDirectVariant::Q5SmallSingleRow => {
                    phase.qgemv::<8, 32>(&a, &b, &y, 4, qmatmul_workgroups_x);
                }
                QMatmulDirectVariant::SingleRow => {
                    phase.qgemv::<4, 64>(&a, &b, &y, 4, qmatmul_workgroups_x);
                }
                QMatmulDirectVariant::Q8Wide64x128 | QMatmulDirectVariant::Tile64x128 => {
                    phase.qmatmul::<64, 128, 32>(&a, &b, &y, 4);
                }
                QMatmulDirectVariant::Tile128x128 => {
                    phase.qmatmul::<128, 128, 32>(&a, &b, &y, 4);
                }
                QMatmulDirectVariant::Tile128x64 => {
                    phase.qmatmul::<128, 64, 32>(&a, &b, &y, 4);
                }
                QMatmulDirectVariant::Tile64x64Fast | QMatmulDirectVariant::Tile64x64 => {
                    phase.qmatmul::<64, 64, 32>(&a, &b, &y, 4);
                }
            }
        });
        let dispatch_size = ir.single_tile_program_grid()?;
        if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
            return None;
        }
        let pipeline_key = QMatMulDirectPipelineKey::new(
            matrix.datatype(),
            m,
            k,
            n,
            dispatch_size,
            input.layout(),
            output.layout(),
        );
        let cache_key = format!(
            "{kernel_name}:direct:{format:?}:m={m}:k={k}:n={n}:dispatch={dispatch_size:?}:{:?}:{:?}",
            input.layout(),
            output.layout()
        );
        qmatmul_direct_kernel_from_ir(
            device,
            kernel_name.clone(),
            kernel_name,
            cache_key,
            matrix,
            pipeline_key,
            input,
            output,
            dispatch_size,
            || Some(ir),
        )
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
    Some(kernel_backend::storage3_kernel_with_prepared_pipeline(
        kernel_name.to_owned(),
        "",
        pipeline,
        input.buffer().clone(),
        matrix.buffer().clone(),
        output.buffer().clone(),
        dispatch_size,
    ))
}

#[allow(clippy::too_many_arguments)]
fn qmatmul_direct_kernel_from_ir(
    device: &Device,
    cached_kernel_name: String,
    kernel_name: String,
    cache_key: String,
    matrix: &QMatrix,
    pipeline_key: QMatMulDirectPipelineKey,
    input: &TensorData,
    output: &TensorData,
    dispatch_size: [u32; 3],
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<DirectKernel> {
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
    let pipeline = kernel_backend::storage3_pipeline_from_ir(
        device,
        &kernel_name,
        cache_key.clone(),
        build_ir,
    )?;
    let pipeline = matrix
        .direct_pipeline_cache()
        .write()
        .get_or_insert(pipeline_key, || pipeline.clone())
        .clone();
    Some(kernel_backend::storage3_kernel_with_prepared_pipeline(
        kernel_name,
        cache_key,
        pipeline,
        input.buffer().clone(),
        matrix.buffer().clone(),
        output.buffer().clone(),
        dispatch_size,
    ))
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

fn tile_cooperative_store_layout_supported(layout: &tile_ir::Layout) -> bool {
    layout.shape().rank() == 2
        && layout.strides().rank() == 2
        && (layout.strides().values()[0] == 1 || layout.strides().values()[1] == 1)
}

fn qgemv_cols_per_workgroup_for_direct(format: tile_ir::GgmlQuantFormat, k: u32, n: u32) -> u32 {
    match select_qgemv_cols_variant(format, k, n) {
        QgemvColsVariant::Q4KSmallWide4 => 4,
        QgemvColsVariant::Q4KSmallWide8 => 8,
        QgemvColsVariant::Q4KLargeNarrow8 => 8,
        QgemvColsVariant::Q6KSmallWide8 => 8,
        QgemvColsVariant::Q6KLargeNarrow4 => 4,
        QgemvColsVariant::Q8WideAccelerated32 => 4 * 8,
        QgemvColsVariant::FormatAccelerated => format.qgemv_cols_per_workgroup_for_shape(k, n),
        QgemvColsVariant::Q5Small8 => 8,
        QgemvColsVariant::Default4 => 4,
    }
}

impl<const R: usize> Tensor<R, f32> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        self.add_q_mat_mul(other)
    }

    pub fn q_mat_mul_swiglu(&self, other: &QMatrix, pair_len: usize) -> Self {
        self.add_q_mat_mul_paired(other, pair_len, tile_ir::PairedActivation::SwiGLU)
    }

    pub fn q_mat_mul_geglu(&self, other: &QMatrix, pair_len: usize) -> Self {
        self.add_q_mat_mul_paired(other, pair_len, tile_ir::PairedActivation::GeGLU)
    }

    pub fn q_mat_mul_reglu(&self, other: &QMatrix, pair_len: usize) -> Self {
        self.add_q_mat_mul_paired(other, pair_len, tile_ir::PairedActivation::ReGLU)
    }
}

impl<const R: usize> Tensor<R, half::f16> {
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        self.cast::<f32>().q_mat_mul(other).cast()
    }
}

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::kernel_selection::assert_selector_generates;

    fn caps(high_tile_limits: bool) -> KernelDeviceCaps {
        KernelDeviceCaps {
            subgroups_supported: true,
            cooperative_matrix_supported: true,
            min_subgroup_size: 32,
            max_subgroup_size: 32,
            max_compute_invocations_per_workgroup: if high_tile_limits { 1024 } else { 256 },
            max_compute_workgroup_storage_size: if high_tile_limits {
                64 * 1024
            } else {
                16 * 1024
            },
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroups_per_dimension: 65_535,
        }
    }

    fn ctx(format: tile_ir::GgmlQuantFormat, y_supports_coop: bool) -> QMatmulDirectCtx {
        QMatmulDirectCtx {
            format,
            y_supports_coop,
        }
    }

    #[test]
    fn qmatmul_direct_selector_generates_each_variant() {
        let selector = qmatmul_direct_selector();
        let q4 = tile_ir::GgmlQuantFormat::Q4_0;
        let cases = [
            (
                QMatmulDirectVariant::Q5SmallSingleRow,
                ctx(tile_ir::GgmlQuantFormat::Q5_0, false),
                caps(false),
            ),
            (QMatmulDirectVariant::SingleRow, ctx(q4, false), caps(false)),
            (
                QMatmulDirectVariant::Q8Wide64x128,
                ctx(tile_ir::GgmlQuantFormat::Q8_0, false),
                caps(true),
            ),
            (QMatmulDirectVariant::Tile128x128, ctx(q4, true), caps(true)),
            (QMatmulDirectVariant::Tile128x64, ctx(q4, true), caps(false)),
            (QMatmulDirectVariant::Tile64x128, ctx(q4, true), caps(false)),
            (
                QMatmulDirectVariant::Tile64x64Fast,
                ctx(q4, true),
                caps(false),
            ),
            (QMatmulDirectVariant::Tile64x64, ctx(q4, false), caps(false)),
        ];
        assert_selector_generates(&selector, cases);
    }

    #[test]
    fn qgemv_cols_selector_generates_each_variant() {
        let selector = qgemv_cols_selector();
        let cases = [
            (
                QgemvColsVariant::Q4KSmallWide4,
                QgemvColsCtx {
                    format: tile_ir::GgmlQuantFormat::Q4K,
                },
            ),
            (
                QgemvColsVariant::Q4KSmallWide8,
                QgemvColsCtx {
                    format: tile_ir::GgmlQuantFormat::Q4K,
                },
            ),
            (
                QgemvColsVariant::Q4KLargeNarrow8,
                QgemvColsCtx {
                    format: tile_ir::GgmlQuantFormat::Q4K,
                },
            ),
            (
                QgemvColsVariant::Q6KSmallWide8,
                QgemvColsCtx {
                    format: tile_ir::GgmlQuantFormat::Q6K,
                },
            ),
            (
                QgemvColsVariant::Q6KLargeNarrow4,
                QgemvColsCtx {
                    format: tile_ir::GgmlQuantFormat::Q6K,
                },
            ),
            (
                QgemvColsVariant::Q8WideAccelerated32,
                QgemvColsCtx {
                    format: tile_ir::GgmlQuantFormat::Q8_0,
                },
            ),
            (
                QgemvColsVariant::FormatAccelerated,
                QgemvColsCtx {
                    format: tile_ir::GgmlQuantFormat::Q5_0,
                },
            ),
            (
                QgemvColsVariant::Q5Small8,
                QgemvColsCtx {
                    format: tile_ir::GgmlQuantFormat::Q5_0,
                },
            ),
            (
                QgemvColsVariant::Default4,
                QgemvColsCtx {
                    format: tile_ir::GgmlQuantFormat::Q4_0,
                },
            ),
        ];
        assert_selector_generates(&selector, cases.map(|(variant, ctx)| (variant, ctx, caps(false))));
    }
}

impl Operation for QMatMulOperation {
    fn workgroup_shape_constraints(
        &self,
        _device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        if self.m_size() == 1 {
            constraints.add_constraint(0, Constraint::Equals(1));
        } else {
            constraints.add_constraint(0, Constraint::Equals(32));
        }
        constraints.add_constraint(1, Constraint::Equals(1));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
    ) -> [u32; 3] {
        let n = self.n_size();
        let m = self.m_size();
        // Calculate batch size for dimensions beyond the last two (M, K)
        let batch_size: u32 = self
            .in_shape
            .iter()
            .rev()
            .skip(2)
            .map(|x| *x as u32)
            .product();

        if m == 1 {
            [n, 1, batch_size]
        } else {
            [n, m, batch_size]
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        qmatmul_operation_inputs(self.input, &self.matrix, &self.out_shape, nodes)
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let [input, matrix, output] = inputs else {
            return None;
        };
        let input = input.as_tensor()?;
        let MirValue::QMatrix(matrix) = matrix else {
            return None;
        };
        let output = output.as_tensor()?;
        if input.datatype() != DataTypeEnum::F32 || output.datatype() != DataTypeEnum::F32 {
            return None;
        }
        if matrix.datatype() == GgmlType::F32 {
            return self.build_dense_direct_kernel(graph, input, matrix, output);
        }
        Self::direct_kernel_for_tensors(&graph.device(), input, matrix, output, self.name())
    }

    fn requires_single_kernel_batch(&self) -> bool {
        true
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        qmatmul_operation_output(inputs)
    }

    fn name(&self) -> String {
        qmatmul_operation_name("mul", self.input_datatype, &self.in_shape, &self.matrix)
    }
}

/// Distinct hash perturbation per paired activation so the pipeline cache key
/// (which already mixes `pair_len`) doesn't alias kernels for different
/// activations on the same shape.
fn activation_cache_salt(activation: tile_ir::PairedActivation) -> u32 {
    match activation {
        tile_ir::PairedActivation::SwiGLU => 0,
        tile_ir::PairedActivation::GeGLU => 0x4000_0000,
        tile_ir::PairedActivation::ReGLU => 0x8000_0000,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Q4KPairedTile {
    X4x1,
    X4x4,
    X8x1,
    X8x2,
    X2x2,
    X2x4,
}

/// Tuning table: `(variant, env-var name, cols_per_workgroup)`. Drives both
/// `from_env` lookup and the `name`/`cols_per_workgroup` accessors so adding
/// a tile only requires one row.
const Q4K_PAIRED_TILES: &[(Q4KPairedTile, &str, u32)] = &[
    (Q4KPairedTile::X4x1, "4x1", 4),
    (Q4KPairedTile::X2x2, "2x2", 4),
    (Q4KPairedTile::X2x4, "2x4", 8),
    (Q4KPairedTile::X8x1, "8x1", 8),
    (Q4KPairedTile::X4x4, "4x4", 16),
    (Q4KPairedTile::X8x2, "8x2", 16),
];

impl Q4KPairedTile {
    fn from_env() -> Self {
        let tile_choice = std::env::var("FUSOR_Q4K_PAIRED_TILE")
            .or_else(|_| std::env::var("FUSOR_Q4K_SWIGLU_TILE"))
            .unwrap_or_default();
        Q4K_PAIRED_TILES
            .iter()
            .find(|(_, name, _)| *name == tile_choice)
            .map(|(tile, _, _)| *tile)
            .unwrap_or(Self::X8x2)
    }

    fn spec(self) -> &'static (Self, &'static str, u32) {
        Q4K_PAIRED_TILES
            .iter()
            .find(|(tile, _, _)| *tile == self)
            .expect("Q4KPairedTile variant must appear in Q4K_PAIRED_TILES")
    }

    fn name(self) -> &'static str {
        self.spec().1
    }

    fn cols_per_workgroup(self) -> u32 {
        self.spec().2
    }

    fn emit(
        self,
        phase: &mut tile_ir::tile::Program,
        a: &tile_ir::tile::Storage<tile_ir::F32, 2>,
        b: &tile_ir::QuantizedMatrix,
        y: &tile_ir::tile::Storage<tile_ir::F32, 2>,
        pair_len: u32,
        m: u32,
        workgroups_x: u32,
        activation: tile_ir::PairedActivation,
    ) {
        macro_rules! emit_tile {
            ($method:ident) => {
                phase.$method(a, b, y, pair_len, m, workgroups_x, activation)
            };
        }

        match self {
            Self::X4x1 => emit_tile!(qgemv_q4k_paired_4x1),
            Self::X4x4 => emit_tile!(qgemv_q4k_paired_4x4),
            Self::X8x1 => emit_tile!(qgemv_q4k_paired_8x1),
            Self::X8x2 => emit_tile!(qgemv_q4k_paired_8x2),
            Self::X2x2 => emit_tile!(qgemv_q4k_paired_2x2),
            Self::X2x4 => emit_tile!(qgemv_q4k_paired_2x4),
        }
    }
}

impl Operation for QMatMulPairedOperation {
    fn workgroup_shape_constraints(&self, _device: &Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::Equals(1));
        constraints.add_constraint(1, Constraint::Equals(1));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[MirValue],
    ) -> [u32; 3] {
        [self.pair_len as u32, self.m_size(), 1]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn inputs(&self, nodes: &crate::compute_graph::ComputeGraphInner) -> Vec<MirValue> {
        qmatmul_operation_inputs(self.input, &self.matrix, &self.out_shape, nodes)
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let [input, matrix, output] = inputs else {
            return None;
        };
        let input = input.as_tensor()?;
        let MirValue::QMatrix(matrix) = matrix else {
            return None;
        };
        let output = output.as_tensor()?;
        if input.datatype() != DataTypeEnum::F32 || output.datatype() != DataTypeEnum::F32 {
            return None;
        }

        let format = self.direct_quant_format()?;
        let a_view = flatten_matrix_layout(input.layout())?;
        let y_view = flatten_matrix_layout(output.layout())?;
        let m = a_view.rows;
        let k = a_view.cols;
        let pair_len = self.pair_len as u32;
        if m == 0
            || y_view.rows != m
            || y_view.cols != pair_len
            || k != self.matrix.shape[1] as u32
            || self.matrix.shape[0] as u32 != pair_len.checked_mul(2)?
        {
            return None;
        }

        let limits = graph.device().limits();
        let max_workgroups = limits.max_compute_workgroups_per_dimension;
        let tile = Q4KPairedTile::from_env();
        let tile_name = tile.name();
        let cols_per_workgroup = tile.cols_per_workgroup();
        let cols_workgroups = pair_len.div_ceil(cols_per_workgroup);
        let total_workgroups = cols_workgroups.checked_mul(m)?;
        let [workgroups_x, _] = split_workgroups_2d(total_workgroups, max_workgroups)?;
        let dispatch_size = [workgroups_x, total_workgroups.div_ceil(workgroups_x), 1];
        if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
            return None;
        }

        let activation = self.activation;
        let pipeline_key = QMatMulDirectPipelineKey::new(
            matrix.datatype(),
            m,
            k,
            pair_len.wrapping_add(activation_cache_salt(activation)),
            dispatch_size,
            input.layout(),
            output.layout(),
        );
        let kernel_name = self.name();
        let cache_key = format!(
            "{kernel_name}:direct:{format:?}:tile={tile_name}:m={m}:k={k}:pair={pair_len}:act={activation:?}:dispatch={dispatch_size:?}:{:?}:{:?}",
            input.layout(),
            output.layout()
        );
        qmatmul_direct_kernel_from_ir(
            &graph.device(),
            "q_mat_paired".to_owned(),
            kernel_name,
            cache_key,
            matrix,
            pipeline_key,
            input,
            output,
            dispatch_size,
            || {
                Some(tile_ir::tile::build(move |phase| {
                    let a = tile_storage_read_with_direct_layout(phase, a_view);
                    let b = phase.quantized_matrix(format, k, pair_len * 2);
                    let y = tile_storage_write_with_direct_layout(phase, y_view);
                    tile.emit(phase, &a, &b, &y, pair_len, m, workgroups_x, activation);
                }))
            },
        )
    }

    fn requires_single_kernel_batch(&self) -> bool {
        true
    }

    fn output(&self, _: &crate::compute_graph::ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        qmatmul_operation_output(inputs)
    }

    fn name(&self) -> String {
        qmatmul_operation_name(
            self.activation.label(),
            self.input_datatype,
            &self.in_shape,
            &self.matrix,
        )
    }
}

impl QMatMulOperation {
    fn build_dense_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        input: &TensorData,
        matrix: &QMatrix,
        output: &TensorData,
    ) -> Option<DirectKernel> {
        let [n, k] = matrix.shape() else {
            return None;
        };
        let (n, k) = (*n, *k);
        let input_shape = input.layout().shape();
        let rank = input_shape.len();
        if rank < 2 {
            return None;
        }
        let mut dense_shape = input_shape.to_vec();
        dense_shape[rank - 2] = k;
        dense_shape[rank - 1] = n;
        let mut dense_strides = vec![0; rank];
        dense_strides[rank - 2] = 1;
        dense_strides[rank - 1] = k;
        let dense_weight_t = TensorData::new_from_parts(
            matrix.device(),
            matrix.buffer().clone(),
            Layout::from_parts(
                0,
                dense_shape.into_boxed_slice(),
                dense_strides.into_boxed_slice(),
            ),
            DataTypeEnum::F32,
        );
        let device = graph.device();
        let dense_matmul = MatMulOperation::new(
            DataTypeEnum::F32,
            self.input,
            self.input,
            input.layout().shape(),
            dense_weight_t.layout().shape(),
            None,
            &device,
        );
        dense_matmul.build_direct_kernel(
            graph,
            &dense_matmul
                .workgroup_shape_constraints(&device)
                .solve(device.max_subgroup_size())?,
            &[
                input.clone().into(),
                dense_weight_t.into(),
                output.clone().into(),
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use std::{mem::size_of, sync::Arc};

    use fusor_gguf::{BlockQ4_0, BlockQ8_0, GgufBlock};

    use super::*;
    use crate::{
        compute_graph::{ComputeGraphInner, ComputeGraphNodes},
        mir::direct_kernel::DirectKernelBinding,
        mir::workgroup_shape::WorkgroupShape,
    };

    fn padded_copy_size(size: u64) -> u64 {
        let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
        ((size + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT)
    }

    #[tokio::test]
    async fn qmatmul_direct_kernel_binds_compact_quantized_weight_buffer() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let weight_shape = [128usize, 256usize];
        let element_count = weight_shape.iter().product::<usize>();
        let block_count = element_count / BlockQ4_0::BLOCK_SIZE;
        let raw_bytes = vec![0; block_count * size_of::<BlockQ4_0>()];
        let matrix =
            QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4_0).unwrap();

        let compact_len = block_count * size_of::<<BlockQ4_0 as GgufBlock>::BytesF32>();
        let dense_len = element_count * size_of::<f32>();
        assert_eq!(matrix.buffer().size(), padded_copy_size(compact_len as u64));
        assert!(matrix.buffer().size() < padded_copy_size(dense_len as u64));

        let input = TensorData::new_for_shape(&device, &[1, weight_shape[1]], DataTypeEnum::F32);
        let output = TensorData::new_for_shape(&device, &[1, weight_shape[0]], DataTypeEnum::F32);
        let graph = ComputeGraphInner {
            device: device.downgrade(),
            nodes: ComputeGraphNodes::default(),
        };
        let operation = QMatMulOperation {
            input_datatype: DataTypeEnum::F32,
            input: NodeIndex::new(0),
            matrix: matrix.clone(),
            in_shape: Box::new([1, weight_shape[1]]),
            out_shape: Box::new([1, weight_shape[0]]),
        };
        let kernel = operation
            .build_direct_kernel(
                &graph,
                &WorkgroupShape::new(256, 1, 1),
                &[input.into(), matrix.clone().into(), output.into()],
            )
            .expect("qmatmul should build a direct quantized kernel");

        let bindings = kernel.bindings_for_test();
        assert_eq!(bindings.len(), 3);
        let DirectKernelBinding::Storage {
            binding,
            buffer,
            read_only,
        } = &bindings[1];
        assert_eq!(*binding, 1);
        assert!(*read_only);
        assert!(Arc::ptr_eq(buffer, matrix.buffer()));
    }

    #[tokio::test]
    async fn qmatmul_accepts_dense_f32_qmatrix_without_generic_fallback() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let matrix = QMatrix::from_parts(
            &device,
            bytemuck::cast_slice(&weights),
            Box::new([2usize, 4usize]),
            GgmlType::F32,
        )
        .unwrap();
        let input_rows = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
        let input: Tensor<2, f32> = Tensor::new(&device, &input_rows);

        let result = input.q_mat_mul(&matrix).as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2]);
        assert!((result[[0, 0]] - 30.0).abs() < 1e-4);
        assert!((result[[0, 1]] - 70.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn q5_0_qgemv_matches_expected_values() {
        let Ok(device) = Device::new().await else {
            return;
        };

        fn q5_0_block(scale: f32, high_bits: [u8; 4], low_bits: u8) -> Vec<u8> {
            let mut bytes = Vec::with_capacity(22);
            bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
            bytes.extend_from_slice(&high_bits);
            bytes.extend(std::iter::repeat_n(low_bits, 16));
            bytes
        }

        let mut raw_bytes = Vec::new();
        raw_bytes.extend(q5_0_block(1.0, [0xff; 4], 0x11));
        raw_bytes.extend(q5_0_block(1.0, [0x00; 4], 0xff));
        let matrix =
            QMatrix::from_parts(&device, &raw_bytes, Box::new([2, 32]), GgmlType::Q5_0).unwrap();
        let input_rows = vec![(1..=32).map(|value| value as f32).collect::<Vec<_>>()];
        let input: Tensor<2, f32> = Tensor::new(&device, &input_rows);

        let result = input.q_mat_mul(&matrix).as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, 2]);
        assert!((result[[0, 0]] - 528.0).abs() < 1e-3);
        assert!((result[[0, 1]] + 528.0).abs() < 1e-3);
    }

    #[tokio::test]
    async fn f16_qmatmul_casts_through_f32_direct_path() {
        let Ok(device) = Device::new().await else {
            return;
        };
        if !device.f16_supported() {
            return;
        }

        let weight_shape = [4usize, BlockQ8_0::BLOCK_SIZE];
        let block_count = weight_shape.iter().product::<usize>() / BlockQ8_0::BLOCK_SIZE;
        let raw_bytes = vec![0; block_count * size_of::<BlockQ8_0>()];
        let matrix =
            QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q8_0).unwrap();
        let input_rows = vec![vec![half::f16::from_f32(0.25); weight_shape[1]]];
        let input: Tensor<2, half::f16> = Tensor::new(&device, &input_rows);

        let result = input.q_mat_mul(&matrix).as_slice().await.unwrap();

        assert_eq!(result.shape(), &[1, weight_shape[0]]);
        assert!(
            result
                .as_slice()
                .iter()
                .take(weight_shape[0])
                .all(|value| *value == half::f16::from_f32(0.0))
        );
    }
}
