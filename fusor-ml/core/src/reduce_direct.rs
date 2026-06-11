use fusor_tile_ir as tile_ir;
use fusor_tile_ir_kernels as tile_ir_kernels;
use std::hash::Hash;

use crate::{
    access_analysis::InputAccesses,
    mir::{
        inputs::MirValue, kernel_backend, kernel_backend::DirectKernel, operation::Operation,
        workgroup_shape::WorkgroupShape,
    },
    nary_direct::{
        ValueTile, apply_unary_function_chain, declare_value, eval_nary_expr, linear_group,
        output_dims_from_flat, tile_u32,
    },
    nary_wise::NaryScalar,
    reduce::{ReduceOp, ReduceOperation},
    reduce_tiled::{datatype_scalar, input_allocation_bytes},
    tensor::DataTypeEnum,
    visit_tiled::{MaybeQData, distribute_workgroups},
};

const BLOCK: usize = 256;
/// Output rows per workgroup on the subgroup-per-output route.
const SUBGROUP_ROWS: u32 = 4;
/// Consecutive reduce-axis values each lane folds per iteration on the
/// subgroup-per-output route (matches the dense gemv kernel's run length).
const SUBGROUP_VALUES_PER_LANE: u32 = 4;

struct ReduceDirectKernelVariant;

/// How the fused reduce distributes its fold across threads.
#[derive(Clone, Copy)]
enum ReduceRoute {
    /// One workgroup per output, 256 lanes splitting the reduce axis. The
    /// final cross-lane combine uses subgroup collectives (one reduction per
    /// subgroup, partials through workgroup memory, one more reduction) when
    /// the device has a fixed subgroup width; otherwise the scratch tree.
    Cooperative {
        subgroup_finish: Option<tile_ir_kernels::SubgroupConfig>,
    },
    /// One subgroup per output: lanes walk the reduce axis in
    /// `SUBGROUP_VALUES_PER_LANE`-long runs (consecutive lanes read adjacent
    /// runs — coalesced when the dominant input is contiguous along the
    /// axis) and a single subgroup collective combines them.
    SubgroupRow(tile_ir_kernels::SubgroupConfig),
    /// One thread per output with a per-thread fold.
    Serial,
}

/// The subgroup-per-output route only pays when adjacent lanes read adjacent
/// addresses: the byte-dominant reduce-axis-dependent input must be
/// contiguous along the axis. Otherwise the serial layout (adjacent threads
/// = adjacent outputs) is the coalesced one.
fn dominant_k_dep_input_is_k_contiguous(
    operation: &ReduceOperation,
    values: &[MaybeQData],
) -> bool {
    let Some(metas) = values
        .iter()
        .map(crate::reduce_tiled::analysis_meta)
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    let Some(access) =
        InputAccesses::collect(&operation.expression, operation.inputs.len(), &metas)
    else {
        return false;
    };
    let axis = operation.axis;
    let mut best: Option<(u64, bool)> = None;
    for i in 0..operation.inputs.len() {
        if !access.depends_on(i, axis) {
            continue;
        }
        let contiguous = access.dims[i]
            .iter()
            .enumerate()
            .any(|(j, &d)| d == axis && metas[i].strides.get(j).copied() == Some(1));
        let bytes = input_allocation_bytes(&metas[i], &values[i]);
        if best.as_ref().is_none_or(|(b, _)| bytes > *b) {
            best = Some((bytes, contiguous));
        }
    }
    best.is_some_and(|(_, contiguous)| contiguous)
}

/// The shared preamble for every fused-reduce kernel builder: split the
/// MirValues into producer values + output, gate on device support, and
/// declare the kernel bindings.
pub(crate) struct ReduceKernelInputs {
    pub(crate) values: Vec<MaybeQData>,
    pub(crate) output_shape: Vec<u32>,
    pub(crate) total_outputs: u32,
}

impl ReduceKernelInputs {
    pub(crate) fn parse(
        operation: &ReduceOperation,
        graph: &crate::compute_graph::ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Option<Self> {
        let (output, producers) = inputs.split_last()?;
        let output = output.as_tensor()?;
        let values = producers
            .iter()
            .map(|input| MaybeQData::try_from(input.clone()).ok())
            .collect::<Option<Vec<_>>>()?;

        let f16_unsupported = !graph.device().f16_supported();
        if f16_unsupported {
            let uses_f16 = output.datatype() == DataTypeEnum::F16
                || operation.function.datatype() == DataTypeEnum::F16
                || values.iter().any(|value| match value {
                    MaybeQData::Tensor(tensor) => tensor.datatype() == DataTypeEnum::F16,
                    MaybeQData::QMatrix(matrix) => {
                        matches!(matrix.datatype(), fusor_gguf::GgmlType::F16)
                    }
                });
            if uses_f16 {
                return None;
            }
        }

        if operation.post_element_wise.input_datatype() != operation.function.datatype() {
            return None;
        }

        let output_shape = output
            .layout()
            .shape()
            .iter()
            .copied()
            .map(u32::try_from)
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        let total_outputs = output_shape
            .iter()
            .try_fold(1u32, |acc, dim| acc.checked_mul(*dim))?;

        Some(Self {
            values,
            output_shape,
            total_outputs,
        })
    }
}

pub(crate) fn build_reduce_direct_kernel(
    operation: &ReduceOperation,
    graph: &crate::compute_graph::ComputeGraphInner,
    workgroup_shape: &WorkgroupShape,
    inputs: &[MirValue],
) -> Option<DirectKernel> {
    let parsed = ReduceKernelInputs::parse(operation, graph, inputs)?;
    let output = inputs.last()?.as_tensor()?.clone();
    let reduce_size: u32 = operation.reduce_size().try_into().ok()?;

    let reduce_dtype = operation.function.datatype();
    let reduce_op = tile_reduce_op(operation.function.op);
    let initial = operation.function.initial_value;

    let device = graph.device();
    let limits = device.limits();
    let subgroup_config = device.subgroup_config();
    let route =
        if crate::reduce::use_cooperative_reduce(parsed.total_outputs, reduce_size, BLOCK as u32) {
            // The two-stage subgroup finish needs a compile-time subgroup count
            // small enough for the second collective to cover the partials.
            let subgroup_finish = subgroup_config.filter(|config| {
                config.is_fixed()
                    && (BLOCK as u32).is_multiple_of(config.max_size())
                    && BLOCK as u32 / config.max_size() <= config.max_size()
            });
            ReduceRoute::Cooperative { subgroup_finish }
        } else {
            let row_config = subgroup_config.filter(|config| {
                let block = config.block_for_subgroups(SUBGROUP_ROWS);
                reduce_size >= 32
                    && block.is_power_of_two()
                    && block
                        <= limits
                            .max_compute_workgroup_size_x
                            .min(limits.max_compute_invocations_per_workgroup)
                    && dominant_k_dep_input_is_k_contiguous(operation, &parsed.values)
            });
            match row_config {
                Some(config) => ReduceRoute::SubgroupRow(config),
                None => ReduceRoute::Serial,
            }
        };
    let dispatch_size = match route {
        ReduceRoute::SubgroupRow(_) => distribute_workgroups(
            parsed.total_outputs.div_ceil(SUBGROUP_ROWS),
            limits.max_compute_workgroups_per_dimension,
        ),
        _ => operation.dispatch_size(workgroup_shape, inputs),
    };
    let variant =
        kernel_backend::KernelVariantKey::with_payload::<ReduceDirectKernelVariant>(|state| {
            match route {
                ReduceRoute::Cooperative { subgroup_finish } => {
                    0u8.hash(state);
                    subgroup_finish.hash(state);
                }
                ReduceRoute::SubgroupRow(config) => {
                    1u8.hash(state);
                    config.hash(state);
                }
                ReduceRoute::Serial => 2u8.hash(state),
            }
        });
    let cache_key = operation.kernel_cache_key_with_dispatch(
        variant,
        Some(workgroup_shape),
        dispatch_size,
        inputs,
    );

    let axis = operation.axis;
    let expression = operation.expression.clone();
    let post_chain = operation.post_element_wise.clone();
    let output_dtype = output.datatype();
    let output_value = MaybeQData::Tensor(output);
    let ReduceKernelInputs {
        values,
        output_shape,
        total_outputs,
    } = parsed;

    kernel_backend::run_kernel(
        graph.device().kernel_cache(),
        operation.name(),
        cache_key,
        dispatch_size,
        move |kb| {
            let mut storages = Vec::with_capacity(values.len());
            let mut metas = Vec::with_capacity(values.len());
            for value in &values {
                let (storage, meta) = declare_value(kb, value, false)?;
                storages.push(storage);
                metas.push(meta);
            }
            let (output_storage, output_meta) = declare_value(kb, &output_value, true)?;

            // Evaluate the fused producer at one index-space coordinate: the
            // output dims with the running reduce index inserted at `axis`.
            let value_at = |program: &mut tile_ir::tile::TileBlock<'_>,
                            dims_out: &[tile_ir::tile::Tile],
                            reduce_index: tile_ir::tile::Tile,
                            mask: tile_ir::tile::Mask| {
                let mut coords = dims_out.to_vec();
                coords.insert(axis, reduce_index);
                let (value, _) =
                    eval_nary_expr(program, &expression, &coords, &storages, &metas, mask, &[]);
                value.cast_to(reduce_dtype)
            };

            let store_reduced = |program: &mut tile_ir::tile::TileBlock<'_>,
                                 reduced: ValueTile,
                                 dims_out: &[tile_ir::tile::Tile],
                                 mask: tile_ir::tile::Mask| {
                let (reduced, reduced_ty) =
                    apply_unary_function_chain(reduced.into_f32(), reduce_dtype, &post_chain)
                        .expect("validated reduce post_element_wise chain");
                let reduced = ValueTile::F32(reduced)
                    .cast_to(reduced_ty)
                    .cast_to(output_dtype);
                let output_index = crate::nary_direct::layout_index(&output_meta, dims_out);
                output_storage.store(program, output_index, reduced, mask);
            };

            let identity = || tile_ir::tile::Tile::literal(tile_literal_for(initial, reduce_dtype));
            let untag = |value: ValueTile| match reduce_dtype {
                DataTypeEnum::F32 => value.into_f32(),
                DataTypeEnum::F16 => value.into_f16(),
                DataTypeEnum::U32 => value.into_u32(),
            };
            let tag = |value: tile_ir::tile::Tile| match reduce_dtype {
                DataTypeEnum::F32 => ValueTile::F32(value),
                DataTypeEnum::F16 => ValueTile::F16(value),
                DataTypeEnum::U32 => ValueTile::U32(value),
            };

            // The per-lane fold every route shares: `k_of(loop_index, run)`
            // gives this lane's reduce coordinate for each of `runs` values
            // per iteration, and out-of-range slots collapse to the identity.
            // A route whose coordinates never leave the axis (serial) skips
            // the select by passing `mask_oob: false`.
            let fold_reduce_lane =
                |program: &mut tile_ir::tile::TileBlock<'_>,
                 dims_out: &[tile_ir::tile::Tile],
                 iterations: tile_ir::tile::FoldIter,
                 in_bounds: &tile_ir::tile::Mask,
                 runs: u32,
                 mask_oob: bool,
                 k_of: &dyn Fn(&tile_ir::tile::Tile, u32) -> tile_ir::tile::Tile|
                 -> tile_ir::tile::Tile {
                    let reduce_binary = reduce_op.binary();
                    let [acc] =
                        program.fold(iterations, [identity()], |program, loop_index, [acc]| {
                            let mut acc = acc;
                            for run in 0..runs {
                                let k_index = k_of(&loop_index, run);
                                let active = in_bounds.clone().and(k_index.clone().lt(reduce_size));
                                let value = value_at(program, dims_out, k_index, active.clone());
                                let value = if mask_oob {
                                    tile_ir::tile::Tile::select(active, untag(value), identity())
                                } else {
                                    untag(value)
                                };
                                acc = acc.binary(reduce_binary, value);
                            }
                            [acc]
                        });
                    acc
                };

            match route {
                ReduceRoute::Cooperative { subgroup_finish } => {
                    let chunks = reduce_size.div_ceil(BLOCK as u32);
                    let phase = kb.program();
                    let scratch = subgroup_finish
                        .map(|config| BLOCK as u32 / config.max_size())
                        .filter(|partials| *partials > 1)
                        .map(|partials| {
                            phase.alloc_workgroup_array(datatype_scalar(reduce_dtype), partials)
                        });
                    phase.program_grid(BLOCK as u32, dispatch_size, |program| {
                        let lane = program.lane();
                        let output_flat = linear_group(program, dispatch_size);
                        let in_bounds = output_flat.lt(total_outputs);
                        let dims_out =
                            output_dims_from_flat_usize(output_flat.clone(), &output_shape);

                        let partial = fold_reduce_lane(
                            program,
                            &dims_out,
                            tile_ir::tile::range(chunks),
                            &in_bounds,
                            1,
                            true,
                            &|loop_index, _| loop_index.clone() * BLOCK as u32 + lane.clone(),
                        );
                        let (reduced, store_mask) = match subgroup_finish {
                            // One collective per subgroup, partials through
                            // workgroup memory, one more collective: two
                            // barriers and two subgroup ops instead of the
                            // log2(block) scratch-tree rounds.
                            Some(config) => {
                                let token = config.token();
                                let first = token.subgroup_reduce(program, reduce_op, partial);
                                let partials = BLOCK as u32 / config.max_size();
                                let sg_lane = token.subgroup_lane(program);
                                let sg_id = token.subgroup_id(program);
                                let reduced = if partials <= 1 {
                                    first
                                } else {
                                    let scratch = scratch.as_ref().unwrap();
                                    let first = program.bind(first);
                                    program.if_then(sg_lane.clone().eq(0u32), |program| {
                                        program.store_workgroup(
                                            scratch,
                                            sg_id.clone(),
                                            first.clone(),
                                        );
                                    });
                                    program.workgroup_barrier();
                                    let slot = tile_ir::tile::Tile::select(
                                        lane.clone().lt(partials),
                                        lane.clone(),
                                        tile_u32(0),
                                    );
                                    let loaded = program.load_workgroup(scratch, slot);
                                    let masked = tile_ir::tile::Tile::select(
                                        lane.clone().lt(partials),
                                        loaded,
                                        identity(),
                                    );
                                    token.subgroup_reduce(program, reduce_op, masked)
                                };
                                // The combined value is uniform across
                                // subgroup 0; store from its first lane
                                // rather than assuming workgroup lane 0
                                // belongs to it.
                                let store_mask =
                                    in_bounds.clone().and(sg_id.eq(0u32)).and(sg_lane.eq(0u32));
                                (reduced, store_mask)
                            }
                            None => (
                                program.group_reduce(reduce_op, BLOCK as u32, partial),
                                in_bounds.clone().and(lane.eq(0u32)),
                            ),
                        };
                        store_reduced(program, tag(reduced), &dims_out, store_mask);
                    });
                }
                ReduceRoute::SubgroupRow(config) => {
                    let block = config.block_for_subgroups(SUBGROUP_ROWS);
                    let token = config.token();
                    kb.program().program_grid(block, dispatch_size, |program| {
                        let group = linear_group(program, dispatch_size);
                        let row = group * SUBGROUP_ROWS + token.subgroup_id(program);
                        let in_bounds = row.clone().lt(total_outputs);
                        let dims_out = output_dims_from_flat_usize(row, &output_shape);
                        let sg_lane = token.subgroup_lane(program);
                        let k_per_iter =
                            program.bind(token.subgroup_size(program) * SUBGROUP_VALUES_PER_LANE);
                        let k_iterations = (tile_u32(reduce_size) + k_per_iter.clone() - 1u32)
                            / k_per_iter.clone();

                        let acc = fold_reduce_lane(
                            program,
                            &dims_out,
                            tile_ir::tile::range(k_iterations),
                            &in_bounds,
                            SUBGROUP_VALUES_PER_LANE,
                            true,
                            &|loop_index, run| {
                                loop_index.clone() * k_per_iter.clone()
                                    + sg_lane.clone() * SUBGROUP_VALUES_PER_LANE
                                    + run
                            },
                        );
                        let reduced = token.subgroup_reduce(program, reduce_op, acc);
                        let store_mask = in_bounds.and(sg_lane.eq(0u32));
                        store_reduced(program, tag(reduced), &dims_out, store_mask);
                    });
                }
                ReduceRoute::Serial => {
                    kb.program()
                        .program_grid(BLOCK as u32, dispatch_size, |program| {
                            let lane = program.lane();
                            let group = linear_group(program, dispatch_size);
                            let flat = group * BLOCK as u32 + lane.clone();
                            let in_bounds = flat.lt(total_outputs);
                            let dims_out = output_dims_from_flat_usize(flat.clone(), &output_shape);

                            let acc = fold_reduce_lane(
                                program,
                                &dims_out,
                                tile_ir::tile::range(reduce_size),
                                &in_bounds,
                                1,
                                false,
                                &|loop_index, _| loop_index.clone(),
                            );
                            store_reduced(program, tag(acc), &dims_out, in_bounds);
                        });
                }
            }
            Some(())
        },
    )
}

pub(crate) fn tile_literal_for(value: NaryScalar, target: DataTypeEnum) -> tile_ir::TileLiteral {
    match target {
        DataTypeEnum::F32 => match value {
            NaryScalar::F32(value) => tile_ir::TileLiteral::f32(value),
            NaryScalar::F16(value) => tile_ir::TileLiteral::f32(value.to_f32()),
            NaryScalar::U32(value) => tile_ir::TileLiteral::f32(value as f32),
        },
        DataTypeEnum::F16 => match value {
            NaryScalar::F32(value) => {
                tile_ir::TileLiteral::F16(half::f16::from_f32(value).to_bits())
            }
            NaryScalar::F16(value) => tile_ir::TileLiteral::F16(value.to_bits()),
            NaryScalar::U32(value) => {
                tile_ir::TileLiteral::F16(half::f16::from_f32(value as f32).to_bits())
            }
        },
        DataTypeEnum::U32 => match value {
            NaryScalar::F32(value) => tile_ir::TileLiteral::U32(value as u32),
            NaryScalar::F16(value) => tile_ir::TileLiteral::U32(value.to_f32() as u32),
            NaryScalar::U32(value) => tile_ir::TileLiteral::U32(value),
        },
    }
}

pub(crate) fn output_dims_from_flat_usize(
    flat: tile_ir::tile::Tile,
    shape: &[u32],
) -> Vec<tile_ir::tile::Tile> {
    let shape = shape.iter().map(|dim| *dim as usize).collect::<Vec<_>>();
    output_dims_from_flat(flat, &shape)
}

pub(crate) fn tile_reduce_op(op: ReduceOp) -> tile_ir::TileReduceOp {
    match op {
        ReduceOp::Sum => tile_ir::TileReduceOp::Sum,
        ReduceOp::Product => tile_ir::TileReduceOp::Product,
        ReduceOp::Max => tile_ir::TileReduceOp::Max,
        ReduceOp::Min => tile_ir::TileReduceOp::Min,
    }
}
