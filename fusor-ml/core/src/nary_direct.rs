use std::hash::Hash;

use fusor_tile_ir as tile_ir;

use crate::{
    mir::{
        inputs::MirValue,
        kernel_backend::{self, DirectKernel},
        operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_wise::{
        ElementwiseOperation, NaryExpr, NaryFunction, NaryOp, NaryScalar, UnaryFunctionChain,
    },
    quantized::QMatrix,
    tensor::{DataTypeEnum, TensorData},
    visit_tiled::MaybeQData,
};

const BLOCK: usize = 256;
const SMALL_BLOCK: usize = 1;

struct NaryDirectKernelVariant;

pub(crate) fn build_nary_direct_kernel(
    operation: &ElementwiseOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    build_nary_direct_kernel_with_output_index(operation, graph, workgroup_shape, inputs, None)
}

pub(crate) fn build_nary_direct_kernel_to_output(
    operation: &ElementwiseOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
    output_index: usize,
) -> Option<DirectKernel> {
    build_nary_direct_kernel_with_output_index(
        operation,
        graph,
        workgroup_shape,
        inputs,
        Some(output_index),
    )
}

fn build_nary_direct_kernel_with_output_index(
    operation: &ElementwiseOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
    forced_output_index: Option<usize>,
) -> Option<DirectKernel> {
    let output_index = forced_output_index.or_else(|| operation.output_tensor_index(inputs))?;
    let values = inputs
        .iter()
        .map(|input| MaybeQData::try_from(input.clone()).ok())
        .collect::<Option<Vec<_>>>()?;
    let MaybeQData::Tensor(_) = values.get(output_index)? else {
        return None;
    };

    if values.iter().any(|value| {
        let datatype = match value {
            MaybeQData::Tensor(tensor) => tensor.datatype(),
            MaybeQData::QMatrix(matrix) => match matrix.datatype() {
                fusor_gguf::GgmlType::F16 => DataTypeEnum::F16,
                _ => return false,
            },
        };
        datatype == DataTypeEnum::F16 && !graph.device().f16_supported()
    }) {
        return None;
    }

    let total_elements = total_elements(&operation.shape)?;
    let plan = plan_nary_tiling(operation, &graph.device(), &values, output_index);
    if let Some(plan) = &plan
        && std::env::var_os("FUSOR_TRACE_REDUCE_TILED").is_some()
    {
        eprintln!(
            "nary_tiled dim={} invariant={:?} threads={} shape={:?}",
            plan.dim, plan.invariant, plan.total_threads, operation.shape,
        );
    }
    let small_dispatch = total_elements < BLOCK as u32;
    let dispatch_size = if let Some(plan) = &plan {
        let max_per_dim = graph.device().limits().max_compute_workgroups_per_dimension;
        crate::visit_tiled::distribute_workgroups(
            plan.total_threads.div_ceil(BLOCK as u32),
            max_per_dim,
        )
    } else if small_dispatch {
        [total_elements, 1, 1]
    } else {
        operation.dispatch_size(workgroup_shape, inputs)
    };
    let variant =
        kernel_backend::KernelVariantKey::with_payload::<NaryDirectKernelVariant>(|state| {
            output_index.hash(state);
            if let Some(plan) = &plan {
                plan.dim.hash(state);
                plan.invariant.hash(state);
            }
        });
    let cache_key = operation.kernel_cache_key_with_dispatch(
        variant,
        Some(workgroup_shape),
        dispatch_size,
        inputs,
    );
    let name = if std::env::var_os("FUSOR_TRACE_DECODE_NAMES").is_some() {
        operation.name()
    } else {
        format!("nary_direct_out_{output_index}")
    };
    kernel_backend::run_kernel(
        graph.device().kernel_cache(),
        name,
        cache_key,
        dispatch_size,
        |kb| {
            if let Some(plan) = &plan {
                build_nary_tiled_ir(operation, &values, output_index, plan, dispatch_size, kb)
            } else if small_dispatch {
                build_nary_tile_ir::<SMALL_BLOCK>(
                    operation,
                    &values,
                    output_index,
                    dispatch_size,
                    kb,
                )
            } else {
                build_nary_tile_ir::<BLOCK>(operation, &values, output_index, dispatch_size, kb)
            }
        },
    )
}

struct MergedRegionKernelVariant;

/// Bind a value tile into a register so later statements can reuse it
/// without re-evaluating. Statement values are cast to a storable dtype
/// before binding, so `Bool` never reaches here.
/// For every tensor value across all segments (flattened), the index of its
/// first occurrence — the cross-segment binding-sharing pattern.
fn cross_segment_alias_classes<'a>(segments: impl Iterator<Item = &'a [MaybeQData]>) -> Vec<usize> {
    // Sharing equality must match the declare-time dedup exactly: same
    // buffer, same datatype, same layout.
    let mut seen: Vec<(usize, Option<(DataTypeEnum, crate::Layout)>)> = Vec::new();
    let mut classes = Vec::new();
    for segment in segments {
        for value in segment {
            let key = match value {
                MaybeQData::Tensor(tensor) => (
                    std::sync::Arc::as_ptr(tensor.buffer()) as usize,
                    Some((tensor.datatype(), tensor.layout().clone())),
                ),
                MaybeQData::QMatrix(matrix) => {
                    (std::sync::Arc::as_ptr(matrix.buffer()) as usize, None)
                }
            };
            let class = seen
                .iter()
                .position(|entry| *entry == key)
                .unwrap_or(seen.len());
            classes.push(class);
            seen.push(key);
        }
    }
    classes
}

fn bind_value_tile(program: &mut tile_ir::tile::TileBlock<'_>, value: ValueTile) -> ValueTile {
    match value {
        ValueTile::F32(tile) => ValueTile::F32(program.bind(tile)),
        ValueTile::F16(tile) => ValueTile::F16(program.bind(tile)),
        ValueTile::U32(tile) => ValueTile::U32(program.bind(tile)),
        ValueTile::Bool(_) => unreachable!("region statements are cast to storable dtypes"),
    }
}

/// One kernel executing several independent multi-output regions: like
/// [`build_merged_nary_kernel`], each segment owns a contiguous range of
/// workgroups behind a uniform guard, but a segment's body is a statement
/// chain — statement values stay in registers (the `extras` slots of
/// [`eval_nary_expr`]) and every externally-live statement stores to its own
/// output binding. Segment values are inputs-then-outputs.
pub(crate) fn build_merged_region_kernel(
    graph: &crate::compute_graph::ComputeGraphInner,
    segments: &[crate::region::ElementwiseRegionOperation],
    segment_inputs: &[Vec<MirValue>],
) -> Option<DirectKernel> {
    let device = graph.device();
    let max_per_dim = device.limits().max_compute_workgroups_per_dimension;

    struct Segment {
        values: Vec<MaybeQData>,
        /// Per output (relative index): the input slot whose buffer the
        /// output writes in place, bound once as read-write.
        fold: Vec<Option<usize>>,
        elements: u32,
        base: u32,
        groups: u32,
    }
    let mut prepared = Vec::with_capacity(segments.len());
    let mut total_groups = 0u32;
    for (op, inputs) in segments.iter().zip(segment_inputs) {
        let values = inputs
            .iter()
            .map(|input| MaybeQData::try_from(input.clone()).ok())
            .collect::<Option<Vec<_>>>()?;
        if values
            .iter()
            .any(|value| matches!(value, MaybeQData::QMatrix(_)))
        {
            return None;
        }
        if values.iter().any(|value| {
            matches!(value, MaybeQData::Tensor(tensor)
                if tensor.datatype() == DataTypeEnum::F16 && !device.f16_supported())
        }) {
            return None;
        }
        debug_assert_eq!(values.len(), op.inputs.len() + op.output_count());
        let input_count = op.inputs.len();
        let buffer_of = |value: &MaybeQData| match value {
            MaybeQData::Tensor(tensor) => Some(std::sync::Arc::as_ptr(tensor.buffer()) as usize),
            _ => None,
        };
        // An output sharing an input's buffer must bind it exactly once,
        // read-write: wgpu rejects one buffer bound read-only and read-write
        // in the same dispatch.
        let fold: Vec<Option<usize>> = values[input_count..]
            .iter()
            .map(|output| {
                let output_ptr = buffer_of(output)?;
                values[..input_count]
                    .iter()
                    .position(|input| buffer_of(input) == Some(output_ptr))
            })
            .collect();
        let elements = total_elements(&op.shape)?;
        let groups = elements.div_ceil(BLOCK as u32);
        prepared.push(Segment {
            values,
            fold,
            elements,
            base: total_groups,
            groups,
        });
        total_groups = total_groups.checked_add(groups)?;
    }

    let dispatch_size = crate::visit_tiled::distribute_workgroups(total_groups, max_per_dim);
    let cache_key = kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
        kernel_backend::KernelVariantKey::of::<MergedRegionKernelVariant>().hash(state);
        dispatch_size.hash(state);
        segments.len().hash(state);
        for ((op, inputs), segment) in segments.iter().zip(segment_inputs).zip(&prepared) {
            op.hash_kernel_fields(state);
            segment.fold.hash(state);
            inputs.len().hash(state);
            for input in inputs {
                crate::mir::operation::hash_mir_value(state, input);
            }
        }
        // Shared read-only inputs bind once across segments; the sharing
        // pattern changes the generated bindings, so it keys the kernel.
        for class in
            cross_segment_alias_classes(prepared.iter().map(|segment| segment.values.as_slice()))
        {
            class.hash(state);
        }
    });
    let name = if std::env::var_os("FUSOR_TRACE_DECODE_NAMES").is_some() {
        format!(
            "merged_region[{}]",
            segments
                .iter()
                .map(|op| op.name())
                .collect::<Vec<_>>()
                .join("; ")
        )
    } else {
        format!("merged_region_x{}", segments.len())
    };

    kernel_backend::run_kernel(
        device.kernel_cache(),
        name,
        cache_key,
        dispatch_size,
        move |kb| {
            let mut declared = Vec::with_capacity(prepared.len());
            // Read-only inputs shared across segments (a learning-rate
            // tensor read by every optimizer segment) bind once: wgpu
            // rejects one buffer bound at several slots of a dispatch, and
            // one binding is cheaper anyway.
            let mut shared_reads: Vec<(usize, DataTypeEnum, crate::Layout, Storage2, TensorMeta)> =
                Vec::new();
            for (op, segment) in segments.iter().zip(&prepared) {
                let input_count = op.inputs.len();
                let folded_inputs: rustc_hash::FxHashSet<usize> =
                    segment.fold.iter().flatten().copied().collect();
                let mut storages: Vec<Storage2> = Vec::with_capacity(segment.values.len());
                let mut metas: Vec<TensorMeta> = Vec::with_capacity(segment.values.len());
                for (binding, value) in segment.values.iter().enumerate() {
                    if binding >= input_count
                        && let Some(source) = segment.fold[binding - input_count]
                    {
                        storages.push(storages[source].clone());
                        metas.push(metas[source].clone());
                        continue;
                    }
                    let write = binding >= input_count || folded_inputs.contains(&binding);
                    if !write && let MaybeQData::Tensor(tensor) = value {
                        let ptr = std::sync::Arc::as_ptr(tensor.buffer()) as usize;
                        let datatype = tensor.datatype();
                        // Only reads through the identical view share a
                        // binding: the same buffer read through different
                        // layouts needs its own metadata.
                        if let Some((.., storage, meta)) =
                            shared_reads.iter().find(|(p, d, layout, _, _)| {
                                *p == ptr && *d == datatype && layout == tensor.layout()
                            })
                        {
                            storages.push(storage.clone());
                            metas.push(meta.clone());
                            continue;
                        }
                        let (storage, meta) = declare_value(kb, value, false)?;
                        shared_reads.push((
                            ptr,
                            datatype,
                            tensor.layout().clone(),
                            storage.clone(),
                            meta.clone(),
                        ));
                        storages.push(storage);
                        metas.push(meta);
                        continue;
                    }
                    let (storage, meta) = declare_value(kb, value, write)?;
                    storages.push(storage);
                    metas.push(meta);
                }
                declared.push((storages, metas));
            }

            kb.program()
                .program_grid(BLOCK as u32, dispatch_size, |program| {
                    let lane = program.lane();
                    let group = program.bind(linear_group(program, dispatch_size));
                    for (op, (segment, (storages, metas))) in
                        segments.iter().zip(prepared.iter().zip(&declared))
                    {
                        let in_segment = group.clone().ge(segment.base)
                            & group.clone().lt(segment.base + segment.groups);
                        program.if_then(in_segment, |program| {
                            let flat = (group.clone() - segment.base) * BLOCK as u32 + lane.clone();
                            let in_bounds = flat.clone().lt(segment.elements);
                            let dims = output_dims_from_flat(flat, &op.shape);
                            let input_count = op.inputs.len();
                            let (input_storages, output_storages) = storages.split_at(input_count);
                            let (input_metas, output_metas) = metas.split_at(input_count);
                            let mut extras: Vec<(ValueTile, DataTypeEnum)> = Vec::new();
                            let mut out_idx = 0usize;
                            for statement in &op.statements {
                                let (value, _) = eval_nary_expr(
                                    program,
                                    &statement.expression,
                                    &dims,
                                    input_storages,
                                    input_metas,
                                    in_bounds.clone(),
                                    &extras,
                                );
                                let value =
                                    bind_value_tile(program, value.cast_to(statement.datatype));
                                if statement.output.is_some() {
                                    let index = layout_index(&output_metas[out_idx], &dims);
                                    output_storages[out_idx].store(
                                        program,
                                        index,
                                        value.clone(),
                                        in_bounds.clone(),
                                    );
                                    out_idx += 1;
                                }
                                extras.push((value, statement.datatype));
                            }
                        });
                    }
                });
            Some(())
        },
    )
}

/// Outputs per thread along the tiled dim of a reuse-tiled elementwise
/// kernel.
const NARY_TM: u32 = 4;
/// Floor on post-tiling thread count: trading threads for register reuse
/// must leave the device saturated.
const MIN_TILED_THREADS: u32 = 65536;

/// A register-reuse tiling for an elementwise kernel: each thread covers
/// `NARY_TM` outputs along `dim`, loading the inputs that are invariant
/// along `dim` once instead of per output.
struct NaryTilePlan {
    dim: usize,
    /// Per input: invariant along `dim` (its loads hoist out of the run).
    invariant: Vec<bool>,
    /// Per input: index-space dim read by each input dimension.
    dims: Vec<Vec<usize>>,
    /// The output shape with `dim` divided by `NARY_TM`.
    thread_shape: Vec<usize>,
    total_threads: u32,
}

/// Tile only when the hoisted loads buy real bandwidth: the invariant
/// inputs must exceed the device's cache-residency threshold (cache-resident
/// re-reads are free, and the tiling costs thread-level parallelism), the
/// tiled dim must not be the innermost output dim (thread-local runs there
/// break inter-thread store coalescing), and enough threads must remain.
fn plan_nary_tiling(
    operation: &ElementwiseOperation,
    device: &crate::Device,
    values: &[MaybeQData],
    output_index: usize,
) -> Option<NaryTilePlan> {
    let input_count = operation.inputs.len();
    let rank = operation.shape.len();
    if rank < 2 || output_index != input_count || values.len() != input_count + 1 {
        return None;
    }
    // Quantized inputs decode through a dedicated load path the value-tile
    // evaluator can't hoist.
    if values[..input_count]
        .iter()
        .any(|value| matches!(value, MaybeQData::QMatrix(_)))
    {
        return None;
    }
    let metas: Vec<TensorMeta> = values[..input_count]
        .iter()
        .map(|value| match value {
            MaybeQData::Tensor(tensor) => TensorMeta::new(tensor),
            MaybeQData::QMatrix(_) => None,
        })
        .collect::<Option<_>>()?;
    let access =
        crate::access_analysis::InputAccesses::collect(&operation.expression, input_count, &metas)?;

    let mut best: Option<(u64, usize)> = None;
    for dim in 0..rank.saturating_sub(1) {
        if operation.shape[dim] < NARY_TM as usize {
            continue;
        }
        let invariant_bytes: u64 = (0..input_count)
            .filter(|&i| !access.depends_on(i, dim))
            .map(|i| input_allocation_bytes(&metas[i], &values[i]))
            .sum();
        if invariant_bytes < device.last_level_cache_bytes() {
            continue;
        }
        if best
            .as_ref()
            .is_none_or(|(best_bytes, _)| invariant_bytes > *best_bytes)
        {
            best = Some((invariant_bytes, dim));
        }
    }
    let (_, dim) = best?;
    let invariant: Vec<bool> = (0..input_count)
        .map(|i| !access.depends_on(i, dim))
        .collect();

    let mut thread_shape = operation.shape.to_vec();
    thread_shape[dim] = thread_shape[dim].div_ceil(NARY_TM as usize);
    let total_threads = total_elements(&thread_shape)?;
    if total_threads < MIN_TILED_THREADS {
        return None;
    }
    Some(NaryTilePlan {
        dim,
        invariant,
        dims: access.dims,
        thread_shape,
        total_threads,
    })
}

fn build_nary_tiled_ir(
    operation: &ElementwiseOperation,
    values: &[MaybeQData],
    output_index: usize,
    plan: &NaryTilePlan,
    dispatch_size: [u32; 3],
    kb: &mut tile_ir::KernelBuilder<std::sync::Arc<wgpu::Buffer>>,
) -> Option<()> {
    let mut storages = Vec::with_capacity(values.len());
    let mut metas = Vec::with_capacity(values.len());
    for (binding, value) in values.iter().enumerate() {
        let (storage, meta) = declare_value(kb, value, binding == output_index)?;
        storages.push(storage);
        metas.push(meta);
    }
    let extent = operation.shape[plan.dim] as u32;
    let total_threads = plan.total_threads;
    let input_count = operation.inputs.len();

    let input_coords = |coords: &[tile_ir::tile::Tile], i: usize| -> Vec<tile_ir::tile::Tile> {
        plan.dims[i].iter().map(|&d| coords[d].clone()).collect()
    };

    kb.program()
        .program_grid(BLOCK as u32, dispatch_size, |program| {
            let lane = program.lane();
            let group = linear_group(program, dispatch_size);
            let flat = group * BLOCK as u32 + lane;
            let in_bounds = flat.clone().lt(total_threads);
            let mut coords = output_dims_from_flat(flat, &plan.thread_shape);
            let base = program.bind(coords[plan.dim].clone() * NARY_TM);

            // Invariant loads hoist out of the per-output run; the base
            // coordinate is always in range for an in-bounds thread.
            coords[plan.dim] = base.clone();
            let hoisted: Vec<Option<(ValueTile, DataTypeEnum)>> = (0..input_count)
                .map(|i| {
                    if !plan.invariant[i] {
                        return None;
                    }
                    let index = layout_index(&metas[i], &input_coords(&coords, i));
                    let loaded = storages[i].load(program, index, in_bounds.clone());
                    let native = match loaded {
                        ValueTile::F32(v) | ValueTile::F16(v) | ValueTile::U32(v) => v,
                        ValueTile::Bool(_) => unreachable!("tensor inputs are f32/f16/u32"),
                    };
                    Some((
                        value_tile_of(metas[i].datatype, program.bind(native)),
                        metas[i].datatype,
                    ))
                })
                .collect();

            for j in 0..NARY_TM {
                let coord = base.clone() + j;
                let in_bounds_j = in_bounds.clone() & coord.clone().lt(extent);
                coords[plan.dim] = coord;
                let slot_values: Vec<(ValueTile, DataTypeEnum)> = (0..input_count)
                    .map(|i| match &hoisted[i] {
                        Some(value) => value.clone(),
                        None => {
                            let index = layout_index(&metas[i], &input_coords(&coords, i));
                            (
                                storages[i].load(program, index, in_bounds_j.clone()),
                                metas[i].datatype,
                            )
                        }
                    })
                    .collect();
                let (value, value_ty) =
                    eval_nary_expr_on_value_tiles(&operation.expression, &slot_values);
                let value = value.cast_to(operation.output_datatype);
                debug_assert_eq!(value_ty, operation.output_datatype);
                let output_index_value = layout_index(&metas[output_index], &coords);
                storages[output_index].store(program, output_index_value, value, in_bounds_j);
            }
        });
    Some(())
}

fn value_tile_of(datatype: DataTypeEnum, value: tile_ir::tile::Tile) -> ValueTile {
    match datatype {
        DataTypeEnum::F32 => ValueTile::F32(value),
        DataTypeEnum::F16 => ValueTile::F16(value),
        DataTypeEnum::U32 => ValueTile::U32(value),
    }
}

fn total_elements(shape: &[usize]) -> Option<u32> {
    shape
        .iter()
        .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))
}

impl ElementwiseOperation {
    pub(crate) fn output_tensor_index(&self, inputs: &[MirValue]) -> Option<usize> {
        inputs.len().checked_sub(1)
    }
}

/// Map a [`DataTypeEnum`] to the runtime tile-ir element type. Used to carry
/// the element type of a `ValueTile`/`Storage2`. The `Tile`/`Storage` values
/// are non-generic and the element travels in the IR, so the
/// `ValueTile`/`Storage2` tag exists only to drive the cast/load/store routing
/// below.
pub(crate) fn datatype_element(datatype: DataTypeEnum) -> tile_ir::ElementType {
    match datatype {
        DataTypeEnum::F32 => tile_ir::ElementType::F32,
        DataTypeEnum::F16 => tile_ir::ElementType::F16,
        DataTypeEnum::U32 => tile_ir::ElementType::U32,
    }
}

#[derive(Clone)]
pub(crate) enum ValueTile {
    F32(tile_ir::tile::Tile),
    F16(tile_ir::tile::Tile),
    U32(tile_ir::tile::Tile),
    Bool(tile_ir::tile::Mask),
}

impl ValueTile {
    pub(crate) fn cast_to(self, target: DataTypeEnum) -> Self {
        match (self, target) {
            (Self::F32(v), DataTypeEnum::F32) => Self::F32(v),
            (Self::F32(v), DataTypeEnum::F16) => Self::F16(v.cast(tile_ir::ElementType::F16)),
            (Self::F32(v), DataTypeEnum::U32) => Self::U32(v.cast(tile_ir::ElementType::U32)),
            (Self::F16(v), DataTypeEnum::F32) => Self::F32(v.cast(tile_ir::ElementType::F32)),
            (Self::F16(v), DataTypeEnum::F16) => Self::F16(v),
            (Self::F16(v), DataTypeEnum::U32) => Self::U32(v.cast(tile_ir::ElementType::U32)),
            (Self::U32(v), DataTypeEnum::F32) => Self::F32(v.cast(tile_ir::ElementType::F32)),
            (Self::U32(v), DataTypeEnum::F16) => Self::F16(v.cast(tile_ir::ElementType::F16)),
            (Self::U32(v), DataTypeEnum::U32) => Self::U32(v),
            (Self::Bool(v), DataTypeEnum::F32) => Self::F32(bool_as_f32(v)),
            (Self::Bool(v), DataTypeEnum::F16) => {
                Self::F16(bool_as_f32(v).cast(tile_ir::ElementType::F16))
            }
            (Self::Bool(v), DataTypeEnum::U32) => Self::U32(bool_as_u32(v)),
        }
    }

    pub(crate) fn into_f32(self) -> tile_ir::tile::Tile {
        match self.cast_to(DataTypeEnum::F32) {
            Self::F32(v) => v,
            _ => unreachable!(),
        }
    }

    pub(crate) fn into_mask(self) -> tile_ir::tile::Mask {
        match self {
            Self::Bool(v) => v,
            Self::F32(v) => v.ne(0.0),
            Self::F16(v) => v.ne(tile_ir::tile::Tile::f16_bits(0)),
            Self::U32(v) => v.ne(0u32),
        }
    }

    fn unary(self, op: tile_ir::TileUnaryOp) -> Self {
        match self {
            Self::F32(v) => Self::F32(v.unary(op)),
            Self::F16(v) => Self::F16(v.unary(op)),
            Self::U32(v) => Self::U32(v.unary(op)),
            Self::Bool(v) => Self::Bool(v.unary(op)),
        }
    }

    pub(crate) fn binary(self, op: tile_ir::TileBinaryOp, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::F32(a), Self::F32(b)) => Self::F32(a.binary(op, b)),
            (Self::F16(a), Self::F16(b)) => Self::F16(a.binary(op, b)),
            (Self::U32(a), Self::U32(b)) => Self::U32(a.binary(op, b)),
            (Self::Bool(a), Self::Bool(b)) => Self::Bool(a.binary(op, b)),
            _ => panic!("nary direct binary op called with mismatched tile types"),
        }
    }

    fn compare(self, op: tile_ir::TileCompareOp, rhs: Self, output: DataTypeEnum) -> Self {
        let mask = match (self, rhs) {
            (Self::F32(a), Self::F32(b))
            | (Self::F16(a), Self::F16(b))
            | (Self::U32(a), Self::U32(b))
            | (Self::Bool(a), Self::Bool(b)) => compare_bool(op, a, b),
            _ => panic!("nary direct compare called with mismatched tile types"),
        };
        ValueTile::Bool(mask).cast_to(output)
    }
}

fn compare_bool(
    op: tile_ir::TileCompareOp,
    left: tile_ir::tile::Tile,
    right: tile_ir::tile::Tile,
) -> tile_ir::tile::Mask {
    match op {
        tile_ir::TileCompareOp::Lt => left.lt(right),
        tile_ir::TileCompareOp::Le => left.le(right),
        tile_ir::TileCompareOp::Gt => left.gt(right),
        tile_ir::TileCompareOp::Ge => left.ge(right),
        tile_ir::TileCompareOp::Eq => left.eq(right),
        tile_ir::TileCompareOp::Ne => left.ne(right),
    }
}

fn bool_as_f32(value: tile_ir::tile::Mask) -> tile_ir::tile::Tile {
    tile_ir::tile::Tile::select(value, 1.0.into(), 0.0.into())
}

fn bool_as_u32(value: tile_ir::tile::Mask) -> tile_ir::tile::Tile {
    tile_ir::tile::Tile::select(value, 1u32.into(), 0u32.into())
}

#[derive(Clone)]
pub(crate) enum Storage2 {
    F32(tile_ir::tile::Storage),
    F16(tile_ir::tile::Storage),
    U32(tile_ir::tile::Storage),
    /// A block-quantized matrix loaded per element (block decode in the
    /// lowerer). Loads through [`eval_nary_expr`]'s dedicated path — `load`
    /// on this variant is unreachable.
    Quantized(tile_ir::QuantizedMatrix),
}

impl Storage2 {
    pub(crate) fn load(
        &self,
        program: &tile_ir::tile::TileBlock<'_>,
        index: tile_ir::tile::Tile,
        mask: tile_ir::tile::Mask,
    ) -> ValueTile {
        let index = tile_ir::tile::Tile::select(mask, index, tile_u32(0));
        let mask = tile_ir::tile::Mask::all();
        match self {
            Self::F32(storage) => ValueTile::F32(program.load(
                storage.at((0u32, index)),
                mask,
                zero_literal(DataTypeEnum::F32),
            )),
            Self::F16(storage) => ValueTile::F16(program.load(
                storage.at((0u32, index)),
                mask,
                zero_literal(DataTypeEnum::F16),
            )),
            Self::U32(storage) => ValueTile::U32(program.load(
                storage.at((0u32, index)),
                mask,
                zero_literal(DataTypeEnum::U32),
            )),
            Self::Quantized(_) => {
                unreachable!("quantized inputs load through the row/col path")
            }
        }
    }

    pub(crate) fn store(
        &self,
        program: &mut tile_ir::tile::TileBlock<'_>,
        index: tile_ir::tile::Tile,
        value: ValueTile,
        mask: tile_ir::tile::Mask,
    ) {
        match self {
            Self::F32(storage) => {
                if let ValueTile::F32(value) = value.cast_to(DataTypeEnum::F32) {
                    program.store(storage.at((0u32, index)), value, mask);
                }
            }
            Self::F16(storage) => {
                if let ValueTile::F16(value) = value.cast_to(DataTypeEnum::F16) {
                    program.store(storage.at((0u32, index)), value, mask);
                }
            }
            Self::U32(storage) => {
                if let ValueTile::U32(value) = value.cast_to(DataTypeEnum::U32) {
                    program.store(storage.at((0u32, index)), value, mask);
                }
            }
            Self::Quantized(_) => {
                unreachable!("quantized matrices are read-only inputs")
            }
        }
    }
}

/// Declare one n-ary input or output as a kernel binding: dense tensors and
/// dense-storage matrices as flat strided reads/writes, block-quantized
/// matrices through the format-aware quantized binding.
pub(crate) fn declare_value(
    kb: &mut tile_ir::KernelBuilder<std::sync::Arc<wgpu::Buffer>>,
    value: &MaybeQData,
    write: bool,
) -> Option<(Storage2, TensorMeta)> {
    let declare_dense = |kb: &mut tile_ir::KernelBuilder<std::sync::Arc<wgpu::Buffer>>,
                         buffer: std::sync::Arc<wgpu::Buffer>,
                         meta: &TensorMeta| {
        let tensor = tile_ir::KernelTensorRef::new(buffer, flat_layout(meta.allocation_len));
        let element = datatype_element(meta.datatype);
        let storage = if write {
            kb.write(element, tensor)
        } else {
            kb.read(element, tensor)
        };
        match meta.datatype {
            DataTypeEnum::F32 => Storage2::F32(storage),
            DataTypeEnum::F16 => Storage2::F16(storage),
            DataTypeEnum::U32 => Storage2::U32(storage),
        }
    };
    match value {
        MaybeQData::Tensor(tensor) => {
            let meta = TensorMeta::new(tensor)?;
            let storage = declare_dense(kb, tensor.buffer().clone(), &meta);
            Some((storage, meta))
        }
        MaybeQData::QMatrix(matrix) => {
            if write {
                return None;
            }
            let meta = TensorMeta::for_matrix(matrix)?;
            match crate::quantized::dequantize::quant_format(matrix) {
                Some(format) => {
                    // `QuantizedMatrix::rows` is the dense row *length* (the
                    // contiguous K axis); `cols` counts the rows.
                    let row_len = *matrix.shape().last()? as u32;
                    let row_count = matrix.shape()[..matrix.shape().len() - 1]
                        .iter()
                        .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))?;
                    let storage = fusor_tile_ir_kernels::quantized_matrix_for(
                        kb,
                        matrix.buffer().clone(),
                        format,
                        row_len,
                        row_count,
                    );
                    Some((Storage2::Quantized(storage), meta))
                }
                // Dense f16/f32 storage reads like a plain row-major tensor.
                None => {
                    let storage = declare_dense(kb, matrix.buffer().clone(), &meta);
                    Some((storage, meta))
                }
            }
        }
    }
}

fn build_nary_tile_ir<const BLOCK_SIZE: usize>(
    operation: &ElementwiseOperation,
    values: &[MaybeQData],
    output_index: usize,
    dispatch_size: [u32; 3],
    kb: &mut tile_ir::KernelBuilder<std::sync::Arc<wgpu::Buffer>>,
) -> Option<()> {
    let total_elements = total_elements(&operation.shape)?;
    let mut storages = Vec::with_capacity(values.len());
    let mut tensor_metas = Vec::with_capacity(values.len());
    for (binding, value) in values.iter().enumerate() {
        let (storage, meta) = declare_value(kb, value, binding == output_index)?;
        storages.push(storage);
        tensor_metas.push(meta);
    }

    {
        let phase = kb.program();
        phase.program_grid(BLOCK_SIZE as u32, dispatch_size, |program| {
            let lane = program.lane();
            let group = linear_group(program, dispatch_size);
            let flat_index = group * BLOCK_SIZE as u32 + lane.clone();
            let in_bounds = flat_index.lt(total_elements);
            let dims = output_dims_from_flat(flat_index.clone(), &operation.shape);
            let (value, value_ty) = eval_nary_expr(
                program,
                &operation.expression,
                &dims,
                &storages,
                &tensor_metas,
                in_bounds.clone(),
                &[],
            );
            let value = value.cast_to(operation.output_datatype);
            debug_assert_eq!(value_ty, operation.output_datatype);
            let output_index_value = layout_index(&tensor_metas[output_index], &dims);
            storages[output_index].store(program, output_index_value, value, in_bounds);
        });
    }
    Some(())
}

/// Evaluate an expression at one coordinate. Slot indices beyond the bound
/// storages resolve from `extras` — per-row scalars a surrounding kernel has
/// already computed (row-program phases).
pub(crate) fn eval_nary_expr(
    program: &mut tile_ir::tile::TileBlock<'_>,
    expr: &NaryExpr,
    dims: &[tile_ir::tile::Tile],
    storages: &[Storage2],
    metas: &[TensorMeta],
    mask: tile_ir::tile::Mask,
    extras: &[(ValueTile, DataTypeEnum)],
) -> (ValueTile, DataTypeEnum) {
    if let Some(value) =
        eval_associative_binary_tree(program, expr, dims, storages, metas, mask.clone(), extras)
    {
        return value;
    }

    match expr {
        NaryExpr::Op { children, function } => {
            let mut values = children
                .iter()
                .zip(&function.input_types)
                .map(|(child, expected)| {
                    let (value, ty) =
                        eval_nary_expr(program, child, dims, storages, metas, mask.clone(), extras);
                    (value.cast_to(*expected), ty)
                })
                .collect::<Vec<_>>();
            (emit_function(function, &mut values), function.output_type)
        }
        NaryExpr::IndexedInput { input_idx, indices } => {
            if *input_idx >= storages.len() {
                return extras[*input_idx - storages.len()].clone();
            }
            let meta = &metas[*input_idx];
            let coords = indices
                .iter()
                .map(|index| {
                    let (value, _) =
                        eval_nary_expr(program, index, dims, storages, metas, mask.clone(), extras);
                    match value.cast_to(DataTypeEnum::U32) {
                        ValueTile::U32(value) => value,
                        _ => unreachable!(),
                    }
                })
                .collect::<Vec<_>>();
            if let Storage2::Quantized(matrix) = &storages[*input_idx] {
                // Block-quantized loads address by (position within the row,
                // which row): the position is the last coordinate, and the
                // row is the row-major flattening of every leading
                // coordinate (the meta carries those strides with a 0 in the
                // final slot).
                let along_row = coords.last().cloned().unwrap_or_else(|| tile_u32(0));
                let which_row = layout_index(meta, &coords);
                let value = program.load_quantized(matrix, along_row, which_row, mask, 0.0);
                return (ValueTile::F32(value), DataTypeEnum::F32);
            }
            let index = layout_index(meta, &coords);
            let value = storages[*input_idx].load(program, index, mask);
            (value, meta.datatype)
        }
        NaryExpr::DimIndex(dim) => (ValueTile::U32(dims[*dim].clone()), DataTypeEnum::U32),
        NaryExpr::Scalar(value) => (tile_literal(*value), value.datatype()),
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_associative_binary_tree(
    program: &mut tile_ir::tile::TileBlock<'_>,
    expr: &NaryExpr,
    dims: &[tile_ir::tile::Tile],
    storages: &[Storage2],
    metas: &[TensorMeta],
    mask: tile_ir::tile::Mask,
    extras: &[(ValueTile, DataTypeEnum)],
) -> Option<(ValueTile, DataTypeEnum)> {
    let (op, datatype, terms) = flatten_associative_binary_terms(expr)?;
    let mut terms = terms.into_iter();
    let first = terms.next()?;
    let (value, _) = eval_nary_expr(program, first, dims, storages, metas, mask.clone(), extras);
    let mut value = value.cast_to(datatype);
    for term in terms {
        let (rhs, _) = eval_nary_expr(program, term, dims, storages, metas, mask.clone(), extras);
        value = value.binary(op, rhs.cast_to(datatype));
    }

    Some((value, datatype))
}

fn flatten_associative_binary_terms(
    expr: &NaryExpr,
) -> Option<(tile_ir::TileBinaryOp, DataTypeEnum, Vec<&NaryExpr>)> {
    let NaryExpr::Op { function, .. } = expr else {
        return None;
    };
    let op = associative_tile_op(function)?;
    let datatype = function.output_type;

    let mut stack = vec![expr];
    let mut terms = Vec::new();
    while let Some(current) = stack.pop() {
        match current {
            NaryExpr::Op { children, function }
                if associative_tile_op(function) == Some(op)
                    && function.output_type == datatype
                    && children.len() == 2 =>
            {
                stack.push(&children[1]);
                stack.push(&children[0]);
            }
            _ => terms.push(current),
        }
    }

    (terms.len() > 1).then_some((op, datatype, terms))
}

fn associative_tile_op(function: &NaryFunction) -> Option<tile_ir::TileBinaryOp> {
    let [left, right] = function.input_types.as_slice() else {
        return None;
    };
    if *left != function.output_type || *right != function.output_type {
        return None;
    }

    match function.op {
        NaryOp::Add => Some(tile_ir::TileBinaryOp::Add),
        NaryOp::Mul => Some(tile_ir::TileBinaryOp::Mul),
        _ => None,
    }
}

fn emit_function(function: &NaryFunction, values: &mut [(ValueTile, DataTypeEnum)]) -> ValueTile {
    match function.op {
        NaryOp::Add => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Add, values[1].0.clone()),
        NaryOp::Sub => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Sub, values[1].0.clone()),
        NaryOp::Mul => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Mul, values[1].0.clone()),
        NaryOp::Div => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Div, values[1].0.clone()),
        NaryOp::Pow => values[0]
            .0
            .clone()
            .binary(tile_ir::TileBinaryOp::Pow, values[1].0.clone()),
        NaryOp::Neg => values[0].0.clone().unary(tile_ir::TileUnaryOp::Neg),
        NaryOp::Cast => values[0].0.clone().cast_to(function.output_type),
        NaryOp::Select => match values[1].0.clone().cast_to(function.output_type) {
            ValueTile::F32(a) => {
                if let ValueTile::F32(b) = values[2].0.clone().cast_to(function.output_type) {
                    ValueTile::F32(tile_ir::tile::Tile::select(
                        values[0].0.clone().into_mask(),
                        a,
                        b,
                    ))
                } else {
                    unreachable!()
                }
            }
            ValueTile::F16(a) => {
                if let ValueTile::F16(b) = values[2].0.clone().cast_to(function.output_type) {
                    ValueTile::F16(tile_ir::tile::Tile::select(
                        values[0].0.clone().into_mask(),
                        a,
                        b,
                    ))
                } else {
                    unreachable!()
                }
            }
            ValueTile::U32(a) => {
                if let ValueTile::U32(b) = values[2].0.clone().cast_to(function.output_type) {
                    ValueTile::U32(tile_ir::tile::Tile::select(
                        values[0].0.clone().into_mask(),
                        a,
                        b,
                    ))
                } else {
                    unreachable!()
                }
            }
            ValueTile::Bool(a) => {
                if let ValueTile::Bool(b) = values[2].0.clone().cast_to(function.output_type) {
                    ValueTile::Bool(tile_ir::tile::Tile::select(
                        values[0].0.clone().into_mask(),
                        a,
                        b,
                    ))
                } else {
                    unreachable!()
                }
            }
        },
        NaryOp::Exp | NaryOp::ApproximateExp | NaryOp::LessApproximateExp => {
            values[0].0.clone().unary(tile_ir::TileUnaryOp::Exp)
        }
        NaryOp::Exp2 => values[0].0.clone().unary(tile_ir::TileUnaryOp::Exp2),
        NaryOp::Log => values[0].0.clone().unary(tile_ir::TileUnaryOp::Log),
        NaryOp::Log2 => values[0].0.clone().unary(tile_ir::TileUnaryOp::Log2),
        NaryOp::Sqrt => values[0].0.clone().unary(tile_ir::TileUnaryOp::Sqrt),
        NaryOp::Sin => values[0].0.clone().unary(tile_ir::TileUnaryOp::Sin),
        NaryOp::Cos => values[0].0.clone().unary(tile_ir::TileUnaryOp::Cos),
        NaryOp::Tan => values[0].0.clone().unary(tile_ir::TileUnaryOp::Tan),
        NaryOp::Tanh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Tanh),
        NaryOp::TanhExact => tanh_exact(values[0].0.clone()),
        NaryOp::Asin => values[0].0.clone().unary(tile_ir::TileUnaryOp::Asin),
        NaryOp::Acos => values[0].0.clone().unary(tile_ir::TileUnaryOp::Acos),
        NaryOp::Atan => values[0].0.clone().unary(tile_ir::TileUnaryOp::Atan),
        NaryOp::Sinh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Sinh),
        NaryOp::Cosh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Cosh),
        NaryOp::Asinh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Asinh),
        NaryOp::Acosh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Acosh),
        NaryOp::Atanh => values[0].0.clone().unary(tile_ir::TileUnaryOp::Atanh),
        NaryOp::Abs => values[0].0.clone().unary(tile_ir::TileUnaryOp::Abs),
        NaryOp::LessEqual => values[0].0.clone().compare(
            tile_ir::TileCompareOp::Le,
            values[1].0.clone(),
            function.output_type,
        ),
        NaryOp::Less => values[0].0.clone().compare(
            tile_ir::TileCompareOp::Lt,
            values[1].0.clone(),
            function.output_type,
        ),
        NaryOp::Equal => values[0].0.clone().compare(
            tile_ir::TileCompareOp::Eq,
            values[1].0.clone(),
            function.output_type,
        ),
        NaryOp::NotEqual => values[0].0.clone().compare(
            tile_ir::TileCompareOp::Ne,
            values[1].0.clone(),
            function.output_type,
        ),
        NaryOp::Greater => values[0].0.clone().compare(
            tile_ir::TileCompareOp::Gt,
            values[1].0.clone(),
            function.output_type,
        ),
        NaryOp::GreaterEqual => values[0].0.clone().compare(
            tile_ir::TileCompareOp::Ge,
            values[1].0.clone(),
            function.output_type,
        ),
        NaryOp::AddConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Add,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::SubConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Sub,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::RSubConst(scalar) => tile_literal(scalar)
            .cast_to(values[0].1)
            .binary(tile_ir::TileBinaryOp::Sub, values[0].0.clone()),
        NaryOp::MulConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Mul,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::DivConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Div,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::RDivConst(scalar) => tile_literal(scalar)
            .cast_to(values[0].1)
            .binary(tile_ir::TileBinaryOp::Div, values[0].0.clone()),
        NaryOp::RemConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Rem,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::RRemConst(scalar) => tile_literal(scalar)
            .cast_to(values[0].1)
            .binary(tile_ir::TileBinaryOp::Rem, values[0].0.clone()),
        NaryOp::PowConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Pow,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::MinConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Min,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::MaxConst(scalar) => values[0].0.clone().binary(
            tile_ir::TileBinaryOp::Max,
            tile_literal(scalar).cast_to(values[0].1),
        ),
        NaryOp::EqualConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Eq,
            &values[0],
            scalar,
            function.output_type,
        ),
        NaryOp::LessConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Lt,
            &values[0],
            scalar,
            function.output_type,
        ),
        NaryOp::LessEqualConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Le,
            &values[0],
            scalar,
            function.output_type,
        ),
        NaryOp::GreaterConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Gt,
            &values[0],
            scalar,
            function.output_type,
        ),
        NaryOp::GreaterEqualConst(scalar) => compare_const(
            tile_ir::TileCompareOp::Ge,
            &values[0],
            scalar,
            function.output_type,
        ),
    }
}

pub(crate) fn eval_nary_expr_on_value_tiles(
    expr: &NaryExpr,
    inputs: &[(ValueTile, DataTypeEnum)],
) -> (ValueTile, DataTypeEnum) {
    if let Some((op, datatype, terms)) = flatten_associative_binary_terms(expr) {
        let mut terms = terms.into_iter();
        let first = terms
            .next()
            .expect("associative expression should have at least two terms");
        let (value, _) = eval_nary_expr_on_value_tiles(first, inputs);
        let mut value = value.cast_to(datatype);
        for term in terms {
            let (rhs, _) = eval_nary_expr_on_value_tiles(term, inputs);
            value = value.binary(op, rhs.cast_to(datatype));
        }
        return (value, datatype);
    }

    match expr {
        NaryExpr::Op { children, function } => {
            let mut values = children
                .iter()
                .zip(&function.input_types)
                .map(|(child, expected)| {
                    let (value, ty) = eval_nary_expr_on_value_tiles(child, inputs);
                    (value.cast_to(*expected), ty)
                })
                .collect::<Vec<_>>();
            (emit_function(function, &mut values), function.output_type)
        }
        NaryExpr::IndexedInput { input_idx, .. } => inputs[*input_idx].clone(),
        NaryExpr::Scalar(value) => (tile_literal(*value), value.datatype()),
        NaryExpr::DimIndex(_) => {
            panic!("eval_nary_expr_on_tiles called with a DimIndex leaf — not supported");
        }
    }
}

pub(crate) fn apply_unary_function_chain(
    value: tile_ir::tile::Tile,
    value_ty: DataTypeEnum,
    chain: &UnaryFunctionChain,
) -> Option<(tile_ir::tile::Tile, DataTypeEnum)> {
    if chain.input_datatype() != value_ty {
        return None;
    }

    let mut value = ValueTile::F32(value).cast_to(value_ty);
    let mut value_ty = value_ty;
    for function in &chain.functions {
        if function.input_types.as_slice() != [value_ty] {
            return None;
        }
        let mut values = [(value, value_ty)];
        value = emit_function(function, &mut values);
        value_ty = function.output_type;
    }
    Some((value.into_f32(), value_ty))
}

pub(crate) fn apply_single_input_elementwise_expr(
    value: tile_ir::tile::Tile,
    value_ty: DataTypeEnum,
    expr: &NaryExpr,
    output_ty: DataTypeEnum,
    extras: &[(ValueTile, DataTypeEnum)],
) -> Option<(tile_ir::tile::Tile, DataTypeEnum)> {
    let value = ValueTile::F32(value).cast_to(value_ty);
    let mut inputs = Vec::with_capacity(1 + extras.len());
    inputs.push((value, value_ty));
    inputs.extend_from_slice(extras);
    let (value, actual_ty) = eval_nary_expr_on_value_tiles(expr, &inputs);
    if actual_ty != output_ty {
        return None;
    }
    Some((value.into_f32(), actual_ty))
}

pub(crate) fn apply_multi_input_elementwise_expr(
    values: &[(tile_ir::tile::Tile, DataTypeEnum)],
    expr: &NaryExpr,
    output_ty: DataTypeEnum,
    extras: &[(ValueTile, DataTypeEnum)],
) -> Option<(tile_ir::tile::Tile, DataTypeEnum)> {
    let mut inputs = Vec::with_capacity(values.len() + extras.len());
    inputs.extend(
        values
            .iter()
            .cloned()
            .map(|(value, ty)| (ValueTile::F32(value).cast_to(ty), ty)),
    );
    inputs.extend_from_slice(extras);
    let (value, actual_ty) = eval_nary_expr_on_value_tiles(expr, &inputs);
    if actual_ty != output_ty {
        return None;
    }
    Some((value.into_f32(), actual_ty))
}

fn tanh_exact(value: ValueTile) -> ValueTile {
    let exp_pos = value.clone().unary(tile_ir::TileUnaryOp::Exp);
    let exp_neg = value
        .unary(tile_ir::TileUnaryOp::Neg)
        .unary(tile_ir::TileUnaryOp::Exp);
    exp_pos
        .clone()
        .binary(tile_ir::TileBinaryOp::Sub, exp_neg.clone())
        .binary(
            tile_ir::TileBinaryOp::Div,
            exp_pos.binary(tile_ir::TileBinaryOp::Add, exp_neg),
        )
}

fn compare_const(
    op: tile_ir::TileCompareOp,
    left: &(ValueTile, DataTypeEnum),
    scalar: NaryScalar,
    output: DataTypeEnum,
) -> ValueTile {
    left.0
        .clone()
        .compare(op, tile_literal(scalar).cast_to(left.1), output)
}

pub(crate) fn output_dims_from_flat(
    flat: tile_ir::tile::Tile,
    shape: &[usize],
) -> Vec<tile_ir::tile::Tile> {
    // Peel innermost-out with a running quotient rather than dividing the
    // flat index by each axis' suffix product. Load-bearing, not a style
    // choice: the Apple Metal compiler miscompiles chains of u32 div/mod by
    // large non-power-of-two constants (e.g. `flat / 393216` for a
    // [64,6,256,256] delinearize) once the grid needs a second dispatch
    // dimension — stores land at wild addresses, which with unchecked
    // shaders corrupts arbitrary GPU memory (observed M2 Max, macOS 26;
    // reproduced with stock wgpu + trivial WGSL). The peeled form only ever
    // divides by a single dimension extent, on a quotient already reduced by
    // the inner axes, which compiles correctly at any size.
    // The outermost non-trivial axis takes the raw quotient without a `%`:
    // for in-bounds lanes it is already < dim, out-of-bounds lanes never
    // touch memory (stores are branch-masked, masked loads clamp their index
    // to 0), and a trailing non-power-of-two `%` re-triggers the miscompile
    // (observed with `% 48` on a [48,6,256,256] delinearize).
    let mut dims = vec![tile_u32(0); shape.len()];
    let mut rest = flat;
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis] as u32;
        if dim == 1 {
            continue;
        }
        if shape[..axis].iter().any(|&outer| outer != 1) {
            dims[axis] = rest.clone() % tile_u32(dim);
            rest = rest / tile_u32(dim);
        } else {
            dims[axis] = rest;
            break;
        }
    }
    dims
}

pub(crate) fn layout_index(
    meta: &TensorMeta,
    coords: &[tile_ir::tile::Tile],
) -> tile_ir::tile::Tile {
    let mut index = tile_u32(meta.offset);
    for (axis, (coord, stride)) in coords.iter().zip(&meta.strides).enumerate() {
        if *stride == 0 || meta.shape.get(axis).copied() == Some(1) {
            continue;
        }
        index = index + coord.clone() * tile_u32(*stride);
    }
    index
}

pub(crate) fn linear_group(
    program: &tile_ir::tile::TileBlock<'_>,
    dispatch_size: [u32; 3],
) -> tile_ir::tile::Tile {
    program.program_id(tile_ir::WorkgroupAxis::X)
        + program.program_id(tile_ir::WorkgroupAxis::Y) * dispatch_size[0]
        + program.program_id(tile_ir::WorkgroupAxis::Z)
            * dispatch_size[0].saturating_mul(dispatch_size[1])
}

pub(crate) fn tile_literal(value: NaryScalar) -> ValueTile {
    match value {
        NaryScalar::F32(value) => ValueTile::F32(tile_ir::tile::Tile::literal(
            tile_ir::TileLiteral::F32(tile_ir::F32Bits::new(value)),
        )),
        NaryScalar::F16(value) => ValueTile::F16(tile_ir::tile::Tile::literal(
            tile_ir::TileLiteral::F16(value.to_bits()),
        )),
        NaryScalar::U32(value) => ValueTile::U32(tile_ir::tile::Tile::literal(
            tile_ir::TileLiteral::U32(value),
        )),
    }
}

pub(crate) fn tile_u32(value: u32) -> tile_ir::tile::Tile {
    tile_ir::tile::Tile::literal(tile_ir::TileLiteral::U32(value))
}

pub(crate) fn zero_literal(value: DataTypeEnum) -> tile_ir::TileLiteral {
    match value {
        DataTypeEnum::F32 => tile_ir::TileLiteral::F32(tile_ir::F32Bits::new(0.0)),
        DataTypeEnum::F16 => tile_ir::TileLiteral::F16(half::f16::from_f32(0.0).to_bits()),
        DataTypeEnum::U32 => tile_ir::TileLiteral::U32(0),
    }
}

#[derive(Clone)]
pub(crate) struct TensorMeta {
    pub(crate) datatype: DataTypeEnum,
    pub(crate) shape: Vec<u32>,
    pub(crate) strides: Vec<u32>,
    pub(crate) offset: u32,
    pub(crate) allocation_len: u32,
}

pub(crate) fn input_allocation_bytes(meta: &TensorMeta, value: &MaybeQData) -> u64 {
    let element_bytes = match value {
        MaybeQData::Tensor(tensor) => match tensor.datatype() {
            DataTypeEnum::F16 => 2,
            DataTypeEnum::F32 | DataTypeEnum::U32 => 4,
        },
        MaybeQData::QMatrix(_) => 4,
    };
    meta.allocation_len as u64 * element_bytes
}

impl TensorMeta {
    /// Metadata for a quantized-or-dense matrix input. Quantized matrices
    /// load by (row, col), so the strides flatten the leading dims row-major
    /// and zero out the final (column) slot; dense-storage matrices read as
    /// plain row-major tensors of their storage element type.
    pub(crate) fn for_matrix(matrix: &QMatrix) -> Option<Self> {
        let shape: Vec<u32> = matrix
            .shape()
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<Result<_, _>>()
            .ok()?;
        let allocation_len = shape
            .iter()
            .try_fold(1u32, |acc, dim| acc.checked_mul(*dim))?;
        let quantized = crate::quantized::dequantize::quant_format(matrix).is_some();
        let datatype = if quantized {
            DataTypeEnum::F32
        } else {
            match matrix.datatype() {
                fusor_gguf::GgmlType::F32 => DataTypeEnum::F32,
                fusor_gguf::GgmlType::F16 => DataTypeEnum::F16,
                _ => return None,
            }
        };
        let mut strides = vec![0u32; shape.len()];
        let mut acc = 1u32;
        let stride_dims = if quantized {
            shape.len().saturating_sub(1)
        } else {
            shape.len()
        };
        for dim in (0..stride_dims).rev() {
            strides[dim] = acc;
            acc = acc.checked_mul(shape[dim])?;
        }
        Some(Self {
            datatype,
            shape,
            strides,
            offset: 0,
            allocation_len,
        })
    }

    pub(crate) fn new(tensor: &TensorData) -> Option<Self> {
        Some(Self {
            datatype: tensor.datatype(),
            shape: tensor
                .layout()
                .shape()
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<_, _>>()
                .ok()?,
            strides: tensor
                .layout()
                .strides()
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()?,
            offset: tensor.layout().offset().try_into().ok()?,
            allocation_len: layout_allocation_len(tensor.layout())?,
        })
    }
}

pub(crate) fn flat_layout(allocation_len: u32) -> tile_ir::Layout {
    tile_ir::Layout::strided(
        tile_ir::MemoryLevel::Storage,
        tile_ir::Shape::new([1, allocation_len]),
        &[0, 1],
    )
}

pub(crate) fn layout_allocation_len(layout: &crate::Layout) -> Option<u32> {
    let max_index = layout
        .shape()
        .iter()
        .zip(layout.strides())
        .try_fold(layout.offset(), |acc, (dim, stride)| {
            acc.checked_add(dim.saturating_sub(1).checked_mul(*stride)?)
        })?;
    max_index.checked_add(1)?.try_into().ok()
}
