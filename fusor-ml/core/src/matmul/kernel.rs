use std::hash::Hash;

use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use rustc_hash::FxHasher;

use crate::{
    Device,
    compute_graph::NodeIndex,
    mir::{
        kernel_backend::{self, DirectKernel},
        operation::Operation,
        tile_direct::{
            flatten_matrix_layout_split, tile_storage_read_with_direct_layout_typed,
            tile_storage_write_with_direct_layout_typed,
        },
    },
    nary_direct::apply_typed_unary_function_chain,
    nary_wise::{NaryExpr, NaryFunction, NaryOp, NaryScalar, UnaryFunctionChain},
    reduce::{ReduceFunction, ReduceOp, ReduceOperation},
    tensor::{DataTypeEnum, TensorData},
};

use super::{
    MatMulOperation, MatMulParams, MatrixOperand,
    cost::CoopDispatch,
    sgemm, sgemv,
    variants::{CoopTile, dense_coop_kinds_from_datatype, select_dense_matmul_params},
};

fn device_supported<T>(value: Option<T>) -> Result<T, kernel_backend::DeviceNotSupported> {
    value.ok_or(kernel_backend::DeviceNotSupported)
}

/// The validated views and geometry of one cooperative-matrix matmul
/// lowering (see [`MatMulOperation::hardware_matmul_prep`]).
struct HardwareMatmulPrep {
    a_view: crate::mir::tile_direct::DirectMatrixLayout,
    b_view: crate::mir::tile_direct::DirectMatrixLayout,
    y_view: crate::mir::tile_direct::DirectMatrixLayout,
    shape: tile_ir_kernels::DenseMatmulShape,
    tile: CoopTile,
    row_groups: u32,
    col_groups: u32,
    batch_m_padded: u32,
    n_padded: u32,
}

/// The split count an already-allocated output can host: the partials occupy
/// scratch slices `1..=splits` of the output backing, which was sized before
/// the wave was partitioned. Splitting fewer times is always a correct
/// kernel, so a build that wants a deeper split than the allocation carries
/// takes what fits instead of falling back to the single-pass body.
fn splits_fitting_allocation(
    splits: u32,
    output: &TensorData,
    batch_m_padded: u32,
    n_padded: u32,
    datatype: DataTypeEnum,
) -> u32 {
    if splits <= 1 {
        return 1;
    }
    let slice_bytes =
        u64::from(batch_m_padded) * u64::from(n_padded) * datatype.element_size() as u64;
    if slice_bytes == 0 {
        return 1;
    }
    let slices = output.buffer().size() / slice_bytes;
    splits
        .min(u32::try_from(slices.saturating_sub(1)).unwrap_or(u32::MAX))
        .max(1)
}

impl MatMulOperation {
    pub fn new(
        datatype: DataTypeEnum,
        first: NodeIndex,
        second: NodeIndex,
        first_shape: &[usize],
        second_shape: &[usize],
        parameters: Option<MatMulParams>,
        device: &Device,
    ) -> Self {
        let parameters = parameters.unwrap_or_else(|| {
            let n = second_shape[second_shape.len() - 1];
            let m = first_shape[first_shape.len() - 2];
            let k = first_shape[first_shape.len() - 1];
            select_dense_matmul_params(m, n, k, device, dense_coop_kinds_from_datatype(datatype))
        });
        Self::new_with_parameters(
            datatype,
            first,
            second,
            first_shape,
            second_shape,
            parameters,
        )
    }

    pub(crate) fn new_with_parameters(
        datatype: DataTypeEnum,
        first: NodeIndex,
        second: NodeIndex,
        first_shape: &[usize],
        second_shape: &[usize],
        parameters: MatMulParams,
    ) -> Self {
        let last_dim = first_shape.len() - 1;
        let second_to_last_dim = first_shape.len() - 2;
        let mut out_shape = first_shape.to_vec();
        out_shape[second_to_last_dim] = first_shape[second_to_last_dim];
        out_shape[last_dim] = second_shape[last_dim];
        assert_eq!(first_shape[last_dim], second_shape[second_to_last_dim]);
        assert!(
            first_shape
                .iter()
                .rev()
                .skip(2)
                .zip(second_shape.iter().rev().skip(2))
                .all(|(a, b)| a == b)
        );

        Self {
            first,
            second,
            a: MatrixOperand::plain(first_shape),
            b: MatrixOperand::plain(second_shape),
            out_shape: out_shape.into(),
            datatype,
            pre_element_wise: [
                UnaryFunctionChain::empty(datatype),
                UnaryFunctionChain::empty(datatype),
            ],
            post_element_wise: UnaryFunctionChain::empty(datatype),
            parameters,
        }
    }

    fn can_use_hardware_matmul(&self) -> bool {
        matches!(self.datatype, DataTypeEnum::F32 | DataTypeEnum::F16)
    }

    /// The cooperative kernel hosts dtype-preserving unary chains, plus post
    /// chains that widen f16 operands into an f32 output (the fused form of
    /// matmul-then-cast, which training's mixed-precision backward emits for
    /// every weight gradient). The store lands in the chain's output dtype;
    /// the in-place epilogue rounds back to the operand dtype before the
    /// chain reads it, so the fused result matches the unfused one exactly.
    /// Narrowing chains would round the accumulator ahead of the chain, so
    /// they keep using the generic fused reduction.
    fn coop_epilogues_supported(&self) -> bool {
        let post_out = self.post_element_wise.out_datatype();
        self.pre_element_wise.iter().all(|chain| {
            chain.input_datatype() == self.datatype && chain.out_datatype() == self.datatype
        }) && self.post_element_wise.input_datatype() == self.datatype
            && (post_out == self.datatype
                || (self.datatype == DataTypeEnum::F16 && post_out == DataTypeEnum::F32))
    }

    fn has_elementwise_epilogues(&self) -> bool {
        self.pre_element_wise
            .iter()
            .any(|chain| !chain.functions.is_empty())
            || !self.post_element_wise.functions.is_empty()
    }

    /// The contraction in its composed map-reduce form: a multiply over the
    /// `[batch.., m, n, k]` index space summed along `k`, with the fused
    /// pre/post chains inlined and accumulation upgraded to f32 (matching
    /// the dedicated kernels' accumulator). Routes that aren't
    /// hardware-specialized lower through this — the same generic tiled
    /// reduce any composed contraction gets.
    fn as_fused_reduce(&self) -> ReduceOperation {
        let batch = self.a.batch_dims;
        let (m_dim, n_dim, k_dim) = (batch, batch + 1, batch + 2);
        let mut index_space: Vec<usize> = self.a.batch_shape().to_vec();
        index_space.extend([self.a.rows(), self.b.cols(), self.a.cols()]);

        let apply_chain = |mut expr: NaryExpr, chain: &UnaryFunctionChain| {
            for function in &chain.functions {
                expr = NaryExpr::Op {
                    children: vec![expr],
                    function: function.clone(),
                };
            }
            expr
        };
        let cast_to = |expr: NaryExpr, from: DataTypeEnum, to: DataTypeEnum| {
            if from == to {
                expr
            } else {
                NaryExpr::Op {
                    children: vec![expr],
                    function: NaryFunction::unary(Some("cast".to_string()), NaryOp::Cast, from, to),
                }
            }
        };

        let a_indices = self.a.index_expressions(m_dim, k_dim);
        let b_indices = self.b.index_expressions(k_dim, n_dim);

        let acc_dtype = match self.pre_element_wise[0].out_datatype() {
            DataTypeEnum::U32 => DataTypeEnum::U32,
            DataTypeEnum::F32 | DataTypeEnum::F16 => DataTypeEnum::F32,
        };
        let a = apply_chain(
            NaryExpr::indexed_input(0, a_indices),
            &self.pre_element_wise[0],
        );
        let a = cast_to(a, self.pre_element_wise[0].out_datatype(), acc_dtype);
        let b = apply_chain(
            NaryExpr::indexed_input(1, b_indices),
            &self.pre_element_wise[1],
        );
        let b = cast_to(b, self.pre_element_wise[1].out_datatype(), acc_dtype);
        let expression = NaryExpr::mul(a, b, acc_dtype);

        let initial_value = match acc_dtype {
            DataTypeEnum::U32 => NaryScalar::U32(0),
            _ => NaryScalar::F32(0.0),
        };
        let result_dtype = self.post_element_wise.input_datatype();
        let mut post_functions = Vec::new();
        if acc_dtype != result_dtype {
            post_functions.push(NaryFunction::unary(
                Some("cast".to_string()),
                NaryOp::Cast,
                acc_dtype,
                result_dtype,
            ));
        }
        post_functions.extend(self.post_element_wise.functions.iter().cloned());

        ReduceOperation {
            inputs: vec![self.first, self.second],
            expression,
            shape: index_space.into(),
            function: ReduceFunction {
                name: Some("sum".to_string()),
                op: ReduceOp::Sum,
                initial_value,
                datatype: acc_dtype,
            },
            post_element_wise: UnaryFunctionChain::new(post_functions, acc_dtype),
            axis: k_dim,
        }
    }

    /// The static half of [`Self::build_hardware_matmul`]'s gates: whether
    /// this contraction will reach the cooperative-matrix kernel on this
    /// device. The resolver uses it to decide if reading an operand through
    /// its un-flattened producer is profitable — the coop kernel's tile
    /// staging amortizes the per-load coordinate decomposition, while the
    /// generic reduce re-derives it for every load and loses to a one-time
    /// gather.
    pub(crate) fn hardware_matmul_statically_viable(&self, device: &Device) -> bool {
        self.coop_tile(device).is_some()
    }

    /// The contraction as the planner sees it, at the given launched-segment
    /// count. `None` when the shape is not a coop contraction at all.
    fn coop_dispatch(&self, segments: u32) -> Option<super::cost::CoopDispatch> {
        Some(super::cost::CoopDispatch {
            m: self.a.rows().try_into().ok()?,
            k: self.a.cols().try_into().ok()?,
            n: self.b.cols().try_into().ok()?,
            batch: self
                .a
                .batch_shape()
                .iter()
                .try_fold(1u32, |acc, &dim| acc.checked_mul(u32::try_from(dim).ok()?))?,
            segments,
            datatype: self.datatype,
            has_epilogues: self.has_elementwise_epilogues(),
        })
    }

    /// Split count and staged tile pairs for the dispatch that will actually
    /// run this contraction: `(1, _)` for the single-pass body. `segments` is
    /// the real launched grid depth — 1 standalone, `segments.len()` for a
    /// merged dispatch — never a probe.
    fn coop_splits(
        &self,
        device: &Device,
        tile: CoopTile,
        rg: u32,
        cg: u32,
        segments: u32,
    ) -> (u32, u32) {
        let Some(dispatch) = self.coop_dispatch(segments) else {
            return (1, 2);
        };
        let (splits, buffers) = super::cost::plan_coop_splits(
            dispatch,
            tile,
            rg,
            cg,
            &device.dispatch_policy(),
            device.max_subgroup_size(),
        );
        if device.config().trace_splitk {
            let CoopDispatch { m, k, n, batch, .. } = dispatch;
            eprintln!(
                "matmul_plan name={} m={m} k={k} n={n} batch={batch} segments={segments} \
                 tile={}x{}x{} rg={rg} cg={cg} splits={splits} buffers={buffers}",
                self.name(),
                tile.bm,
                tile.bn,
                tile.bk,
            );
        }
        (splits, buffers)
    }

    /// The tile geometry the cooperative-matrix kernel would run with on
    /// this device, `None` when any static gate fails and the contraction is
    /// bound for the generic path. Shapes need not divide the tile: edge
    /// tiles mask their fills and the output allocation pads to whole tiles.
    pub(crate) fn coop_tile(&self, device: &Device) -> Option<(CoopTile, u32, u32)> {
        if !self.can_use_hardware_matmul()
            || (self.datatype == DataTypeEnum::F16 && !device.f16_supported())
            || !self.coop_epilogues_supported()
        {
            return None;
        }
        let MatMulParams::CoopMatMul = &self.parameters else {
            return None;
        };
        let kind = *dense_coop_kinds_from_datatype(self.datatype).first()?;
        device.coop_token(kind)?;
        let subgroup_config = device.subgroup_config()?;
        if !subgroup_config.is_fixed() {
            return None;
        }
        let CoopDispatch {
            m,
            k,
            n,
            batch,
            has_epilogues,
            ..
        } = self.coop_dispatch(1)?;
        let limits = device.limits();
        // Memoized on the device: the scored selection is asked once per
        // static-viability probe, once per prep, once per allocation and once
        // per trace for every matmul in every resolve, and it enumerates the
        // whole table against every legal split count each time.
        let probe_group = super::cost::tile_probe_group(device, has_epilogues);
        let [bm, bn, bk, row_groups, col_groups] = device.coop_tile_memo(
            crate::device::CoopTileKey {
                m,
                k,
                n,
                batch,
                datatype: self.datatype,
                has_epilogues,
                probe_group,
            },
            || {
                super::cost::plan_coop_tile(
                    m,
                    k,
                    n,
                    batch,
                    self.datatype,
                    has_epilogues,
                    probe_group,
                    &device.dispatch_policy(),
                    subgroup_config.max_size(),
                )
                .map(|(tile, rg, cg)| [tile.bm, tile.bn, tile.bk, rg, cg])
            },
        )?;
        let tile = CoopTile::new(bm, bn, bk);
        // The 1D->3D grid spread plus the kernels' overhang guard cover any
        // u32 tile count; the checked math above is the only real bound. A
        // per-dimension cap here silently dropped real-vocab lm-head shapes
        // (16384x384x32768 = 65536 tiles) onto the generic fallback at ~17x
        // the cost.
        let _ = m
            .div_ceil(tile.bm)
            .checked_mul(n.div_ceil(tile.bn))
            .and_then(|tiles| tiles.checked_mul(batch))?;
        let _ = limits;
        Some((tile, row_groups, col_groups))
    }

    /// Row-major strides of the logical output over its padded backing:
    /// rows step `n_padded`, each batch block spans `m_padded * n_padded`.
    fn padded_out_strides(out_shape: &[usize], m_padded: usize, n_padded: usize) -> Box<[usize]> {
        let rank = out_shape.len();
        let mut strides = vec![0usize; rank];
        strides[rank - 1] = 1;
        strides[rank - 2] = n_padded;
        if rank >= 3 {
            strides[rank - 3] = m_padded * n_padded;
            for axis in (0..rank - 3).rev() {
                strides[axis] = strides[axis + 1] * out_shape[axis + 1];
            }
        }
        strides.into()
    }

    /// The shared head of the cooperative-matrix lowering: flatten the
    /// operand layouts, validate the contraction geometry, pick the tile,
    /// and verify the output allocation carries the tile-padded backing.
    /// Used by both the standalone [`Self::build_hardware_matmul`] and the
    /// horizontally merged builder ([`build_merged_matmul_kernel`]), so the
    /// two agree on every gate by construction.
    fn hardware_matmul_prep(
        &self,
        device: &Device,
        input_a: &TensorData,
        input_b: &TensorData,
        output: &TensorData,
    ) -> Result<HardwareMatmulPrep, kernel_backend::DeviceNotSupported> {
        // Operands with a base map read their producer through it: compose
        // with the runtime buffer layout, then flatten with the operand's
        // dim grouping.
        let operand_layout =
            |operand: &MatrixOperand, input: &TensorData| -> Option<crate::Layout> {
                match &operand.base_map {
                    Some(map) => crate::view::compose_layouts(&map.layout, input.layout()),
                    None => Some(input.layout().clone()),
                }
            };
        let a_layout = device_supported(operand_layout(&self.a, input_a))?;
        let b_layout = device_supported(operand_layout(&self.b, input_b))?;
        let a_view = device_supported(flatten_matrix_layout_split(&a_layout, self.a.split()))?;
        let b_view = device_supported(flatten_matrix_layout_split(&b_layout, self.b.split()))?;

        let m: u32 = self
            .a
            .rows()
            .try_into()
            .map_err(|_| kernel_backend::DeviceNotSupported)?;
        let k: u32 = self
            .a
            .cols()
            .try_into()
            .map_err(|_| kernel_backend::DeviceNotSupported)?;
        let n: u32 = self
            .b
            .cols()
            .try_into()
            .map_err(|_| kernel_backend::DeviceNotSupported)?;
        let batch: u32 = device_supported(
            self.a
                .batch_shape()
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim)),
        )?
        .try_into()
        .map_err(|_| kernel_backend::DeviceNotSupported)?;
        let batch_m = device_supported(batch.checked_mul(m))?;
        let batch_k = device_supported(batch.checked_mul(k))?;
        if a_view.rows != batch_m || a_view.cols != k || b_view.rows != batch_k || b_view.cols != n
        {
            return Err(kernel_backend::DeviceNotSupported);
        }
        let shape = tile_ir_kernels::DenseMatmulShape { batch, m, k, n };

        // Only the cooperative-matrix route stays hand-specialized; gemv
        // shapes and dtype-changing fused chains lower through the generic
        // row reduction. Dtype-preserving unary chains remain hosted here.
        let (tile, row_groups, col_groups) = device_supported(self.coop_tile(device))?;

        // The store covers whole tiles, so `y` is the padded matrix: rows
        // padded to `ceil(m / bm) * bm` per batch and columns to
        // `ceil(n / bn) * bn`, allocated by `inputs()` with the logical
        // output viewing it. Verify the output really has that geometry —
        // a mismatch (the allocation predicted a different tile) falls back
        // to the generic path, which writes through the logical layout.
        let m_padded = m.div_ceil(tile.bm) * tile.bm;
        let n_padded = n.div_ceil(tile.bn) * tile.bn;
        let expected_strides =
            Self::padded_out_strides(&self.out_shape, m_padded as usize, n_padded as usize);
        let padded_elements = device_supported(
            (batch as usize)
                .checked_mul(m_padded as usize)
                .and_then(|rows| rows.checked_mul(n_padded as usize)),
        )?;
        let padded_bytes =
            padded_elements as u64 * self.post_element_wise.out_datatype().element_size() as u64;
        if output.layout().offset() != 0
            || output.layout().strides() != &*expected_strides
            || padded_bytes > output.buffer().size()
        {
            return Err(kernel_backend::DeviceNotSupported);
        }
        let batch_m_padded = device_supported(batch.checked_mul(m_padded))?;
        let y_view = crate::mir::tile_direct::DirectMatrixLayout {
            rows: batch_m_padded,
            cols: n_padded,
            offset: 0,
            layout: tile_ir::Layout::strided(
                tile_ir::MemoryLevel::Storage,
                tile_ir::Shape::new([batch_m_padded, n_padded]),
                &[n_padded, 1],
            ),
        };
        Ok(HardwareMatmulPrep {
            a_view,
            b_view,
            y_view,
            shape,
            tile,
            row_groups,
            col_groups,
            batch_m_padded,
            n_padded,
        })
    }

    fn build_hardware_matmul(
        &self,
        device: &Device,
        input_a: &TensorData,
        input_b: &TensorData,
        output: &TensorData,
    ) -> Result<DirectKernel, kernel_backend::DeviceNotSupported> {
        let HardwareMatmulPrep {
            a_view,
            b_view,
            y_view,
            shape,
            tile,
            row_groups,
            col_groups,
            batch_m_padded,
            n_padded,
        } = self.hardware_matmul_prep(device, input_a, input_b, output)?;
        let subgroup_config = device_supported(device.subgroup_config())?;
        let MatMulParams::CoopMatMul = &self.parameters else {
            return Err(kernel_backend::DeviceNotSupported);
        };
        let kind = *device_supported(dense_coop_kinds_from_datatype(self.datatype).first())?;
        let coop = device_supported(device.coop_token(kind))?;

        let max_wg_per_dim = device.limits().max_compute_workgroups_per_dimension;
        let datatype = self.datatype;

        let make_epilogue = |label, chain: &UnaryFunctionChain| {
            if chain.functions.is_empty() {
                return None;
            }
            let chain = chain.clone();
            Some(tile_ir_kernels::UnaryEpilogue::new(label, move |value| {
                apply_typed_unary_function_chain(value, datatype, &chain)
                    .expect("cooperative matmul epilogue validated before kernel construction")
                    .0
            }))
        };
        let pre_a = make_epilogue("dense_matmul_pre_a", &self.pre_element_wise[0]);
        let pre_b = make_epilogue("dense_matmul_pre_b", &self.pre_element_wise[1]);
        let post = make_epilogue("dense_matmul_post", &self.post_element_wise);

        // Starved tile grids with a long contraction split K across
        // workgroups: partials land in scratch slices of the over-allocated
        // output buffer and a combine kernel folds them (sum-reorder-only
        // numerics). A weight-gradient shape like 64×2048×64 otherwise runs
        // as a single workgroup.
        let (standalone_splits, stage_buffers) =
            self.coop_splits(device, tile, row_groups, col_groups, 1);
        let splits = splits_fitting_allocation(
            standalone_splits,
            output,
            batch_m_padded,
            n_padded,
            self.datatype,
        );
        if splits > 1
            && let Some(kernel) = self.build_split_k_matmul(
                device,
                input_a,
                input_b,
                output,
                &a_view,
                &b_view,
                &y_view,
                shape,
                tile,
                row_groups,
                col_groups,
                subgroup_config,
                coop,
                batch_m_padded,
                n_padded,
                splits,
            )
        {
            return Ok(kernel);
        }

        let used = std::cell::Cell::new(false);
        let ir = tile_ir::tile::build(|phase| {
            let element = match datatype {
                DataTypeEnum::F32 => tile_ir::ElementType::F32,
                DataTypeEnum::F16 => tile_ir::ElementType::F16,
                _ => unreachable!("hardware matmul only supports f32/f16"),
            };
            let out_element = match self.post_element_wise.out_datatype() {
                DataTypeEnum::F32 => tile_ir::ElementType::F32,
                DataTypeEnum::F16 => tile_ir::ElementType::F16,
                _ => unreachable!("hardware matmul only supports f32/f16 outputs"),
            };
            let a = tile_storage_read_with_direct_layout_typed(phase, element, a_view.clone());
            let b = tile_storage_read_with_direct_layout_typed(phase, element, b_view.clone());
            let y = tile_storage_write_with_direct_layout_typed(phase, out_element, y_view.clone());
            used.set(tile_ir_kernels::try_batched_coop_matmul(
                phase,
                tile_ir_kernels::DenseMatmulTensors {
                    a: &a,
                    b: &b,
                    y: &y,
                },
                shape,
                &tile_ir_kernels::DenseMatmulEpilogues {
                    pre_a: pre_a.as_ref(),
                    pre_b: pre_b.as_ref(),
                    post: post.as_ref(),
                },
                max_wg_per_dim,
                tile_ir_kernels::DenseCoopMatmulConfig {
                    coop,
                    subgroups: subgroup_config,
                    tile: tile_ir_kernels::DenseCoopMatmulTile {
                        bm: tile.bm,
                        bn: tile.bn,
                        bk: tile.bk,
                    },
                    row_groups,
                    col_groups,
                    staging: None,
                    stage_buffers,
                    swizzle_group_m: super::cost::swizzle_group_m(
                        self.a.rows(),
                        self.a.cols(),
                        self.b.cols(),
                        self.datatype,
                    ),
                },
            ));
        });
        if !used.get() {
            return Err(kernel_backend::DeviceNotSupported);
        }
        let dispatch_size = ir.grid;
        if dispatch_size.iter().any(|dim| *dim > max_wg_per_dim) {
            return Err(kernel_backend::DeviceNotSupported);
        }
        let inputs = [
            input_a.clone().into(),
            input_b.clone().into(),
            output.clone().into(),
        ];
        let variant =
            kernel_backend::KernelVariantKey::with_payload::<HardwareMatmulVariant>(|state| {
                tile.hash(state);
                subgroup_config.hash(state);
            });
        let cache_key = self.kernel_cache_key_with_dispatch(variant, None, dispatch_size, &inputs);

        let name = self.name();
        let (pipeline, cached) = kernel_backend::three_buffer_pipeline_from_ir(
            device.kernel_cache(),
            &name,
            cache_key,
            || Some(ir),
        )
        .ok_or(kernel_backend::DeviceNotSupported)?;
        Ok(
            kernel_backend::DirectKernel::from_prepared_three_buffer_pipeline(
                name,
                pipeline,
                Some(cached),
                input_a.buffer().clone(),
                input_b.buffer().clone(),
                output.buffer().clone(),
                dispatch_size,
            ),
        )
    }

    /// Split-K route for coop matmuls whose tile grid starves the GPU: the
    /// partials kernel runs `splits × total_tiles` workgroups, each covering
    /// one K-span with the standard coop tile loop and storing an
    /// unnormalized partial into one scratch slice of the over-allocated
    /// output buffer (slices `1..=splits`, allocated by [`Self::inputs`]);
    /// a combine kernel sums the slices into the padded output at slice 0.
    /// Keeping the scratch inside the output allocation means every bound
    /// buffer stays slot-attributable, so flush-plan recording keeps
    /// working. Numerics differ from the single-pass kernel only in
    /// summation order. Returns `None` (single-pass coop path proceeds)
    /// when the geometry, allocation, or device declines.
    #[allow(clippy::too_many_arguments)]
    fn build_split_k_matmul(
        &self,
        device: &Device,
        input_a: &TensorData,
        input_b: &TensorData,
        output: &TensorData,
        a_view: &crate::mir::tile_direct::DirectMatrixLayout,
        b_view: &crate::mir::tile_direct::DirectMatrixLayout,
        y_view: &crate::mir::tile_direct::DirectMatrixLayout,
        shape: tile_ir_kernels::DenseMatmulShape,
        tile: CoopTile,
        row_groups: u32,
        col_groups: u32,
        subgroup_config: fusor_tile_ir_kernels::SubgroupConfig,
        coop: tile_ir::CoopMatrixToken,
        batch_m_padded: u32,
        n_padded: u32,
        splits: u32,
    ) -> Option<DirectKernel> {
        let slice_elements = batch_m_padded.checked_mul(n_padded)?;
        let total_elements = slice_elements.checked_mul(splits.checked_add(1)?)?;
        let required_bytes = total_elements as u64 * self.datatype.element_size() as u64;
        // The output allocation must carry the scratch slices; an exact
        // allocation (a plan built before the split decision, or an aliased
        // buffer) falls back to the single-pass kernel.
        if output.buffer().size() < required_bytes {
            return None;
        }
        let scratch_rows = splits.checked_mul(batch_m_padded)?;
        let scratch_view = crate::mir::tile_direct::DirectMatrixLayout {
            rows: scratch_rows,
            cols: n_padded,
            offset: slice_elements,
            layout: tile_ir::Layout::strided(
                tile_ir::MemoryLevel::Storage,
                tile_ir::Shape::new([scratch_rows, n_padded]),
                &[n_padded, 1],
            ),
        };
        let element = match self.datatype {
            DataTypeEnum::F32 => tile_ir::ElementType::F32,
            DataTypeEnum::F16 => tile_ir::ElementType::F16,
            _ => return None,
        };
        let max_wg_per_dim = device.limits().max_compute_workgroups_per_dimension;

        let used = std::cell::Cell::new(false);
        let ir = tile_ir::tile::build(|phase| {
            let a = tile_storage_read_with_direct_layout_typed(phase, element, a_view.clone());
            let b = tile_storage_read_with_direct_layout_typed(phase, element, b_view.clone());
            let y =
                tile_storage_write_with_direct_layout_typed(phase, element, scratch_view.clone());
            used.set(tile_ir_kernels::try_batched_coop_matmul_split_k(
                phase,
                tile_ir_kernels::DenseMatmulTensors {
                    a: &a,
                    b: &b,
                    y: &y,
                },
                shape,
                splits,
                max_wg_per_dim,
                tile_ir_kernels::DenseCoopMatmulConfig {
                    coop,
                    subgroups: subgroup_config,
                    tile: tile_ir_kernels::DenseCoopMatmulTile {
                        bm: tile.bm,
                        bn: tile.bn,
                        bk: tile.bk,
                    },
                    row_groups,
                    col_groups,
                    staging: None,
                    // The partials body stages one pair; see `staging_depths`.
                    stage_buffers: 1,
                    swizzle_group_m: super::cost::swizzle_group_m(
                        self.a.rows(),
                        self.a.cols(),
                        self.b.cols(),
                        self.datatype,
                    ),
                },
            ));
        });
        if !used.get() {
            if device.config().trace_splitk {
                eprintln!("splitk_declined_by_kernel name={}", self.name());
            }
            return None;
        }
        let dispatch_size = ir.grid;
        if dispatch_size.iter().any(|dim| *dim > max_wg_per_dim) {
            return None;
        }
        let inputs = [
            input_a.clone().into(),
            input_b.clone().into(),
            output.clone().into(),
        ];
        let variant =
            kernel_backend::KernelVariantKey::with_payload::<SplitKMatmulVariant>(|state| {
                tile.hash(state);
                subgroup_config.hash(state);
                splits.hash(state);
                1u64.hash(state);
            });
        let cache_key = self.kernel_cache_key_with_dispatch(variant, None, dispatch_size, &inputs);
        let name = self.name();
        let (pipeline, cached) = kernel_backend::three_buffer_pipeline_from_ir(
            device.kernel_cache(),
            &name,
            cache_key,
            || Some(ir),
        )?;
        let partials = kernel_backend::DirectKernel::from_prepared_three_buffer_pipeline(
            name.clone(),
            pipeline,
            Some(cached),
            input_a.buffer().clone(),
            input_b.buffer().clone(),
            output.buffer().clone(),
            dispatch_size,
        );

        // One read-write view over all `1 + splits` slices: the combine
        // reads the partial slices and stores slice 0, through a single
        // binding of the shared buffer.
        debug_assert_eq!(y_view.offset, 0);
        debug_assert_eq!(
            y_view.rows, batch_m_padded,
            "split-K expects the padded output view"
        );
        let all_rows = scratch_rows + batch_m_padded;
        let combine_view = crate::mir::tile_direct::DirectMatrixLayout {
            rows: all_rows,
            cols: n_padded,
            offset: 0,
            layout: tile_ir::Layout::strided(
                tile_ir::MemoryLevel::Storage,
                tile_ir::Shape::new([all_rows, n_padded]),
                &[n_padded, 1],
            ),
        };
        let combine_ir = tile_ir::tile::build(|phase| {
            let y =
                tile_storage_write_with_direct_layout_typed(phase, element, combine_view.clone());
            tile_ir_kernels::split_k_combine(
                phase,
                &y,
                batch_m_padded,
                n_padded,
                splits,
                max_wg_per_dim,
            );
        });
        let combine_dispatch = combine_ir.grid;
        if combine_dispatch.iter().any(|dim| *dim > max_wg_per_dim) {
            return None;
        }
        let combine_variant =
            kernel_backend::KernelVariantKey::with_payload::<SplitKMatmulVariant>(|state| {
                tile.hash(state);
                subgroup_config.hash(state);
                splits.hash(state);
                2u64.hash(state);
            });
        let combine_key =
            self.kernel_cache_key_with_dispatch(combine_variant, None, combine_dispatch, &inputs);
        let combine = kernel_backend::dynamic_kernel_from_ir(
            device.kernel_cache(),
            format!("{name}_split_combine"),
            combine_key,
            || Some(combine_ir),
            [output.buffer().clone()],
            combine_dispatch,
        )?;

        Some(kernel_backend::DirectKernel::sequence(
            name,
            vec![partials, combine],
        ))
    }
}

struct HardwareMatmulVariant;
struct SplitKMatmulVariant;
struct MergedMatmulVariant;

/// The horizontal-merge compatibility key of a dense matmul: two matmuls
/// merge into one dispatch only when every field matches, which makes the
/// guarded segment bodies identical up to their storage bindings (same
/// logical shape, tile geometry, workgroup size, split factor, and element
/// type). Only matmuls that will take the cooperative-matrix route produce a
/// key.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct MatmulMergeKey {
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    /// Whether this profile would split its K loop as a standalone dispatch.
    /// The tile and the split count themselves are pure functions of the
    /// fields above, so they cannot discriminate two keys these do not; this
    /// one derived bool survives only to keep the split-K wave category where
    /// it is.
    split_candidate: bool,
    datatype: DataTypeEnum,
}

impl MatmulMergeKey {
    /// Whether the profile is a split-K candidate, which is the wave category
    /// its one consumer wants.
    pub(crate) fn splits(&self) -> Option<u32> {
        self.split_candidate.then_some(1)
    }
}

impl MatMulOperation {
    /// See [`MatmulMergeKey`]. `None` = not horizontally mergeable.
    pub(crate) fn merge_profile(&self, device: &Device) -> Option<MatmulMergeKey> {
        // Standalone coop kernels host unary epilogues. The guarded merged
        // body does not yet carry per-segment epilogue identities/bindings.
        if self.has_elementwise_epilogues() {
            return None;
        }
        let (tile, row_groups, col_groups) = self.coop_tile(device)?;
        let CoopDispatch { m, k, n, batch, .. } = self.coop_dispatch(1)?;
        Some(MatmulMergeKey {
            m,
            k,
            n,
            batch,
            // Scored at the probe group, not at 1: the category exists to
            // keep split-K profiles in their own merged wave, so the question
            // is what the merged dispatch will do, and the merged dispatch is
            // bounded by exactly this group.
            split_candidate: self
                .coop_splits(
                    device,
                    tile,
                    row_groups,
                    col_groups,
                    super::cost::tile_probe_group(device, false),
                )
                .0
                > 1,
            datatype: self.datatype,
        })
    }
}

/// One kernel running several independent same-profile dense matmuls (see
/// [`MatmulMergeKey`]): the guarded-segment counterpart of the standalone
/// cooperative-matrix lowering, emitted through
/// [`tile_ir_kernels::try_merged_coop_matmul`]. Split-K profiles produce a
/// two-dispatch sequence — all segments' partials in one kernel, then all
/// combines in another — with each segment's scratch carved out of its own
/// over-allocated output (slot-attributable for flush-plan recording, like
/// the standalone split-K route).
///
/// Returns `None` when any segment fails the hardware gates (the caller
/// falls back to per-segment kernels and poisons the recording).
pub(crate) fn build_merged_matmul_kernel(
    graph: &crate::compute_graph::ComputeGraphInner,
    segments: &[MatMulOperation],
    segment_inputs: &[Vec<crate::mir::inputs::MirValue>],
) -> Option<DirectKernel> {
    let device = graph.device();
    macro_rules! decline {
        ($reason:expr) => {{
            if device.config().trace_matmul_merge {
                eprintln!("matmul_merge_decline reason={}", $reason);
            }
            return None;
        }};
    }
    let first = segments.first()?;
    let mut tensors = Vec::with_capacity(segments.len());
    for (op, inputs) in segments.iter().zip(segment_inputs) {
        let [input_a, input_b, output] = inputs.as_slice() else {
            decline!("input_arity");
        };
        let (Some(input_a), Some(input_b), Some(output)) =
            (input_a.as_tensor(), input_b.as_tensor(), output.as_tensor())
        else {
            decline!("input_tensors");
        };
        if !op.can_use_hardware_matmul()
            || input_a.datatype() != op.datatype
            || input_b.datatype() != op.datatype
            || output.datatype() != op.datatype
            || (op.datatype == DataTypeEnum::F16 && !device.f16_supported())
        {
            decline!("datatype_gate");
        }
        tensors.push((input_a, input_b, output));
    }

    let mut preps = Vec::with_capacity(segments.len());
    for (op, (input_a, input_b, output)) in segments.iter().zip(&tensors) {
        let Ok(prep) = op.hardware_matmul_prep(&device, input_a, input_b, output) else {
            decline!(format!("prep {}", op.name()));
        };
        preps.push(prep);
    }
    let tile = preps[0].tile;
    let shape = preps[0].shape;
    let batch_m_padded = preps[0].batch_m_padded;
    let n_padded = preps[0].n_padded;
    // The merge key guarantees profile equality; re-verify structurally so a
    // drifted caller can never emit mismatched guarded bodies.
    if preps.iter().any(|prep| {
        prep.tile != tile
            || prep.shape.batch != shape.batch
            || prep.shape.m != shape.m
            || prep.shape.k != shape.k
            || prep.shape.n != shape.n
            || prep.batch_m_padded != batch_m_padded
            || prep.n_padded != n_padded
    }) {
        decline!("profile_mismatch");
    }
    if segments.iter().any(|op| op.datatype != first.datatype) {
        decline!("datatype_mismatch");
    }
    // The one place the grid that actually launches is known:
    // `splits x tiles x segments.len()` workgroups. The tile was scored at a
    // fixed probe group (allocation had to precede this partition), so the
    // split the real grid wants may exceed the scratch that was allocated;
    // clamping is always legal — fewer splits is a correct kernel.
    let (merged_splits, stage_buffers) = first.coop_splits(
        &device,
        tile,
        preps[0].row_groups,
        preps[0].col_groups,
        segments.len() as u32,
    );
    let splits = splits_fitting_allocation(
        merged_splits,
        tensors
            .iter()
            .map(|(_, _, output)| *output)
            .min_by_key(|output| output.buffer().size())
            .expect("merged matmul builds from a non-empty segment list"),
        batch_m_padded,
        n_padded,
        first.datatype,
    );

    let Some(subgroup_config) = device.subgroup_config() else {
        decline!("subgroups");
    };
    let MatMulParams::CoopMatMul = &first.parameters else {
        decline!("params");
    };
    let Some(&kind) = dense_coop_kinds_from_datatype(first.datatype).first() else {
        decline!("coop_kind");
    };
    let Some(coop) = device.coop_token(kind) else {
        decline!("coop_token");
    };
    let element = match first.datatype {
        DataTypeEnum::F32 => tile_ir::ElementType::F32,
        DataTypeEnum::F16 => tile_ir::ElementType::F16,
        _ => decline!("element"),
    };
    let max_wg_per_dim = device.limits().max_compute_workgroups_per_dimension;
    let config = tile_ir_kernels::DenseCoopMatmulConfig {
        coop,
        subgroups: subgroup_config,
        tile: tile_ir_kernels::DenseCoopMatmulTile {
            bm: tile.bm,
            bn: tile.bn,
            bk: tile.bk,
        },
        row_groups: preps[0].row_groups,
        col_groups: preps[0].col_groups,
        staging: None,
        stage_buffers,
        swizzle_group_m: super::cost::swizzle_group_m(
            first.a.rows(),
            first.a.cols(),
            first.b.cols(),
            first.datatype,
        ),
    };

    // Split-K segments store partials into scratch slices of their own
    // output allocation; verify every allocation carries the slices.
    let slice_elements = batch_m_padded.checked_mul(n_padded)?;
    if splits > 1 {
        let total_elements = slice_elements.checked_mul(splits.checked_add(1)?)?;
        let required_bytes = total_elements as u64 * first.datatype.element_size() as u64;
        if tensors
            .iter()
            .any(|(_, _, output)| output.buffer().size() < required_bytes)
        {
            decline!("scratch_capacity");
        }
    }

    let name = if device.config().trace_decode_names {
        format!(
            "merged_matmul[{}]",
            segments
                .iter()
                .map(|op| op.name())
                .collect::<Vec<_>>()
                .join("; ")
        )
    } else {
        format!("merged_matmul_x{}", segments.len())
    };

    let used = std::cell::Cell::new(false);
    let ir = tile_ir::tile::build(|phase| {
        let mut storages = Vec::with_capacity(segments.len());
        for prep in &preps {
            let a = tile_storage_read_with_direct_layout_typed(phase, element, prep.a_view.clone());
            let b = tile_storage_read_with_direct_layout_typed(phase, element, prep.b_view.clone());
            // Partials land in the scratch slices (`1..=splits`).
            let y_view = if splits > 1 {
                crate::mir::tile_direct::DirectMatrixLayout {
                    rows: splits * batch_m_padded,
                    cols: n_padded,
                    offset: slice_elements,
                    layout: tile_ir::Layout::strided(
                        tile_ir::MemoryLevel::Storage,
                        tile_ir::Shape::new([splits * batch_m_padded, n_padded]),
                        &[n_padded, 1],
                    ),
                }
            } else {
                prep.y_view.clone()
            };
            let y = tile_storage_write_with_direct_layout_typed(phase, element, y_view);
            storages.push((a, b, y));
        }
        let segment_tensors: Vec<tile_ir_kernels::DenseMatmulTensors> = storages
            .iter()
            .map(|(a, b, y)| tile_ir_kernels::DenseMatmulTensors { a, b, y })
            .collect();
        used.set(tile_ir_kernels::try_merged_coop_matmul(
            phase,
            &segment_tensors,
            shape,
            splits,
            max_wg_per_dim,
            config,
        ));
    });
    if !used.get() {
        decline!("tile_ir_declined");
    }
    let dispatch_size = ir.grid;
    if dispatch_size.iter().any(|dim| *dim > max_wg_per_dim) {
        return None;
    }
    let cache_key = kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
        kernel_backend::KernelVariantKey::of::<MergedMatmulVariant>().hash(state);
        dispatch_size.hash(state);
        tile.hash(state);
        subgroup_config.hash(state);
        splits.hash(state);
        1u64.hash(state);
        crate::compute_graph::resolve::plan_cache::hash_merged_segments(
            state,
            segments.iter(),
            segment_inputs,
        );
    });
    let buffers: Vec<std::sync::Arc<wgpu::Buffer>> = tensors
        .iter()
        .flat_map(|(input_a, input_b, output)| {
            [
                input_a.buffer().clone(),
                input_b.buffer().clone(),
                output.buffer().clone(),
            ]
        })
        .collect();
    let Some(main) = kernel_backend::dynamic_kernel_from_ir(
        device.kernel_cache(),
        name.clone(),
        cache_key,
        move || Some(ir),
        buffers,
        dispatch_size,
    ) else {
        decline!("pipeline");
    };
    if splits <= 1 {
        return Some(main);
    }

    // The merged combine: every segment's `(1 + splits)`-slice buffer bound
    // once read-write, folded by guarded ranges in the same segment order.
    let all_rows = (splits + 1) * batch_m_padded;
    let combine_ir = tile_ir::tile::build(|phase| {
        let storages: Vec<tile_ir::tile::Storage> = preps
            .iter()
            .map(|_| {
                tile_storage_write_with_direct_layout_typed(
                    phase,
                    element,
                    crate::mir::tile_direct::DirectMatrixLayout {
                        rows: all_rows,
                        cols: n_padded,
                        offset: 0,
                        layout: tile_ir::Layout::strided(
                            tile_ir::MemoryLevel::Storage,
                            tile_ir::Shape::new([all_rows, n_padded]),
                            &[n_padded, 1],
                        ),
                    },
                )
            })
            .collect();
        let ys: Vec<&tile_ir::tile::Storage> = storages.iter().collect();
        tile_ir_kernels::merged_split_k_combine(
            phase,
            &ys,
            batch_m_padded,
            n_padded,
            splits,
            max_wg_per_dim,
        );
    });
    let combine_dispatch = combine_ir.grid;
    if combine_dispatch.iter().any(|dim| *dim > max_wg_per_dim) {
        return None;
    }
    let combine_key = kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
        kernel_backend::KernelVariantKey::of::<MergedMatmulVariant>().hash(state);
        combine_dispatch.hash(state);
        tile.hash(state);
        subgroup_config.hash(state);
        splits.hash(state);
        2u64.hash(state);
        crate::compute_graph::resolve::plan_cache::hash_merged_segments(
            state,
            segments.iter(),
            segment_inputs,
        );
    });
    let combine_buffers: Vec<std::sync::Arc<wgpu::Buffer>> = tensors
        .iter()
        .map(|(_, _, output)| output.buffer().clone())
        .collect();
    let Some(combine) = kernel_backend::dynamic_kernel_from_ir(
        device.kernel_cache(),
        format!("{name}_split_combine"),
        combine_key,
        move || Some(combine_ir),
        combine_buffers,
        combine_dispatch,
    ) else {
        decline!("combine_pipeline");
    };
    Some(kernel_backend::DirectKernel::sequence(
        name,
        vec![main, combine],
    ))
}

impl Operation for MatMulOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        self.datatype.hash(state);
        self.a.hash(state);
        self.b.hash(state);
        self.out_shape.hash(state);
        self.pre_element_wise.hash(state);
        self.post_element_wise.hash(state);
        self.parameters.hash(state);
    }

    fn workgroup_shape_constraints(
        &self,
        device: &Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        match &self.parameters {
            MatMulParams::Vector(sgemv_params) => {
                sgemv::workgroup_shape_constraints(self, device, sgemv_params)
            }
            MatMulParams::MatMul(sgemm_params) => {
                sgemm::workgroup_shape_constraints(self, device, sgemm_params)
            }
            // The cooperative kernels carry their own grid and block
            // (`ir.grid`); this shape only reaches the generic fused-reduce
            // fallback, so it is the fallback's own policy.
            MatMulParams::CoopMatMul => {
                crate::row_program::RowProgramOperation::from_reduce(&self.as_fused_reduce())
                    .workgroup_shape_constraints(device)
            }
        }
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        let [input_a, _input_b, _output] = inputs else {
            panic!("MatMulOperation requires 3 inputs");
        };
        // The logical contraction shape: an un-flattened operand's runtime
        // layout has a different rank, so the runtime layouts can't be used.
        let input_a = input_a.as_tensor().unwrap();
        let last_dim_size = self.b.cols();
        let second_to_last_dim_size = self.a.rows();
        let batch_size = self.a.batch_shape().iter().product::<usize>();

        match &self.parameters {
            MatMulParams::Vector(sgemv_params) => sgemv::dispatch_size(
                second_to_last_dim_size as u32,
                last_dim_size as u32,
                batch_size as u32,
                input_a
                    .device()
                    .limits()
                    .max_compute_workgroups_per_dimension,
                workgroup_shape,
                sgemv_params,
            ),
            MatMulParams::MatMul(sgemm_params) => sgemm::dispatch_size(
                last_dim_size,
                second_to_last_dim_size,
                batch_size,
                workgroup_shape,
                sgemm_params,
            ),
            MatMulParams::CoopMatMul => {
                crate::row_program::RowProgramOperation::from_reduce(&self.as_fused_reduce())
                    .dispatch_size(workgroup_shape, inputs)
            }
        }
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.first);
        f(self.second);
    }

    fn visit_dependencies_mut(&mut self, f: &mut dyn FnMut(&mut NodeIndex)) {
        f(&mut self.first);
        f(&mut self.second);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        let a = nodes.get_result(self.first).unwrap();
        let b = nodes.get_result(self.second).unwrap();
        let device = a.device();
        let datatype = self.post_element_wise.out_datatype();
        // The coop kernel stores whole tiles: pad the backing to tile
        // multiples and view the logical shape over it (consumers never
        // read the pad region). Split-K shapes over-allocate one extra
        // padded slice per split for the partials scratch (slice 0 is the
        // output; the combine kernel folds slices 1..=splits into it).
        // Shapes that already divide the tile — and anything bound for the
        // generic path — allocate exactly.
        let (m, n) = (self.a.rows(), self.b.cols());
        let padded = self.coop_tile(device).and_then(|(tile, rg, cg)| {
            let m_padded = m.div_ceil(tile.bm as usize) * tile.bm as usize;
            let n_padded = n.div_ceil(tile.bn as usize) * tile.bn as usize;
            // Allocation predates the merge partition, so it sizes against the
            // same fixed probe group the tile was scored at. A merged build of
            // a shorter tail chunk may want more splits than this; that build
            // clamps, rather than every matmul over-allocating for the worst
            // case the way the group-1 sizing did.
            let probe = super::cost::tile_probe_group(device, self.has_elementwise_epilogues());
            let slices = self.coop_splits(device, tile, rg, cg, probe).0 as usize + 1;
            (slices > 1 || m_padded != m || n_padded != n).then_some((m_padded, n_padded, slices))
        });
        let output_tensor = match padded {
            Some((m_padded, n_padded, slices)) => {
                let batch: usize = self.a.batch_shape().iter().product();
                let backing = TensorData::new_for_shape(
                    device,
                    &[slices * batch, m_padded, n_padded],
                    datatype,
                );
                TensorData::new_from_parts(
                    device,
                    backing.buffer().clone(),
                    crate::Layout::from_parts(
                        0,
                        self.out_shape.clone(),
                        Self::padded_out_strides(&self.out_shape, m_padded, n_padded),
                    ),
                    datatype,
                )
            }
            None => TensorData::new_for_shape(device, &self.out_shape, datatype),
        };
        vec![a.into(), b.into(), output_tensor.into()]
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> Option<DirectKernel> {
        let [input_a, input_b, output] = inputs else {
            return None;
        };
        let input_a = input_a.as_tensor()?;
        let input_b = input_b.as_tensor()?;
        let output = output.as_tensor()?;
        if self.can_use_hardware_matmul()
            && input_a.datatype() == self.datatype
            && input_b.datatype() == self.datatype
            && output.datatype() == self.post_element_wise.out_datatype()
            && (self.datatype != DataTypeEnum::F16 || graph.device().f16_supported())
            && let Ok(kernel) =
                self.build_hardware_matmul(&graph.device(), input_a, input_b, output)
        {
            return Some(kernel);
        }
        // Everything else is the composed contraction's own lowering: the
        // generic tiled (or serial) fused reduce, identical to what any
        // unrecognized contraction gets.
        if std::env::var_os("FUSOR_TRACE_MATMUL_FALLBACK").is_some() {
            tracing::warn!(
                "matmul fallback to row reduce: dtype={:?} params={:?} m={} k={} n={} batch={:?} hw={} epi={} f16_dev={} coop_tile={:?} a_dt={:?} b_dt={:?} out_dt={:?}",
                self.datatype,
                std::mem::discriminant(&self.parameters),
                self.a.rows(),
                self.a.cols(),
                self.b.cols(),
                self.a.batch_shape(),
                self.can_use_hardware_matmul(),
                self.coop_epilogues_supported(),
                graph.device().f16_supported(),
                self.coop_tile(&graph.device())
                    .map(|(tile, ..)| (tile.bm, tile.bn, tile.bk)),
                input_a.datatype(),
                input_b.datatype(),
                output.datatype(),
            );
        }
        let reduce = self.as_fused_reduce();
        crate::row_program::RowProgramOperation::from_reduce(&reduce).build_direct_kernel(
            graph,
            workgroup_shape,
            inputs,
        )
    }

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> crate::mir::inputs::MirValue {
        let output_tensor = inputs[2].as_tensor().unwrap().clone();
        output_tensor.into()
    }

    fn name(&self) -> String {
        format!(
            "matmul_{}_{}_by_{}",
            self.datatype,
            self.a
                .shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.b
                .shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}
