use fusor_tile_ir::tile::{Mask, Tile, TileBlock, WorkgroupTile};
use fusor_tile_ir::{ElementType, ScalarElement, TileLiteral, WorkgroupAxis};

use super::{
    helpers::{reduce_workgroup, scalar_of, supports_float, u32_tile, zero_fill, NEG_MAX_F32},
    types::SoftmaxMeta,
};

/// The per-workgroup softmax statistics produced by [`workgroup_softmax_block`].
/// Runtime-typed (ARBOR_DESIGN.md §2): every tile carries its `ElementType` as
/// data — these are all F32 accumulators.
#[derive(Clone)]
pub(super) struct WorkgroupSoftmaxBlock {
    pub max: Tile,
    pub denom: Tile,
    pub prob: Tile,
}

/// Storage-typed `-f32::MAX` literal for the kernel's `element` (F32 or F16).
fn neg_max_fill(element: ElementType) -> TileLiteral {
    match scalar_of(element) {
        ScalarElement::F32 => TileLiteral::f32(NEG_MAX_F32),
        ScalarElement::F16 => TileLiteral::F16(half::f16::from_f32(NEG_MAX_F32).to_bits()),
        _ => panic!("softmax only supports F32 and F16 element types"),
    }
}

/// The runtime block dispatch (ARBOR_DESIGN.md §5): the old `softmax`/
/// `softmax_partials`/`softmax_reduce`/`softmax_write` each had a 4-way
/// `match meta.block { 128 | 512 | 1024 => ..::<BLOCK>, _ => None }`. The block
/// is now a runtime `u32`, so the only thing the match guarded — rejecting
/// unsupported sizes before `program_grid` asserts on them — survives as this
/// predicate. `program_grid` bakes `block` into `@workgroup_size`, so the
/// emitted Naga and cache key are identical to the monomorphized version.
fn supported_block(block: u32) -> bool {
    matches!(block, 128 | 512 | 1024)
}

fn linear_group(program: &TileBlock<'_>, dispatch_size: [u32; 3]) -> Tile {
    let x = program.program_id(WorkgroupAxis::X);
    let y = program.program_id(WorkgroupAxis::Y);
    let z = program.program_id(WorkgroupAxis::Z);
    x + y * u32_tile(dispatch_size[0]) + z * u32_tile(dispatch_size[0] * dispatch_size[1])
}

fn storage_index(
    program: &mut TileBlock<'_>,
    meta: &SoftmaxMeta,
    row: Tile,
    axis_value: Tile,
    output: bool,
) -> Tile {
    let tensor_meta = if output {
        &meta.output_meta
    } else {
        &meta.input_meta
    };
    let strides = tensor_meta.strides.as_slice();
    let axis = meta.axis as usize;
    let mut remaining = row;
    let mut index = u32_tile(tensor_meta.offset);

    for dim in (0..meta.shape.len()).rev() {
        let coord = if dim == axis {
            axis_value.clone()
        } else {
            let size = meta.shape[dim];
            let coord = if size == 1 {
                u32_tile(0)
            } else {
                remaining.clone() % u32_tile(size)
            };
            if size != 1 {
                remaining = program.bind(remaining / u32_tile(size));
            }
            coord
        };
        match strides[dim] {
            0 => {}
            1 => {
                index = index + coord;
            }
            stride => {
                index = index + coord * u32_tile(stride);
            }
        }
    }

    program.bind(index)
}

pub(super) fn softmax_partial_scale(block_max: Tile, global_max: Tile) -> Tile {
    (block_max - global_max).exp()
}

pub(super) fn workgroup_softmax_block(
    program: &mut TileBlock<'_>,
    lane: Tile,
    score: Tile,
    valid: Mask,
    reduce: &WorkgroupTile,
    probs: Option<&WorkgroupTile>,
) -> WorkgroupSoftmaxBlock {
    let score = Tile::select(
        valid.clone(),
        score,
        Tile::literal(TileLiteral::f32(NEG_MAX_F32)),
    );
    program.store_workgroup(reduce, lane.clone(), score.clone());
    program.workgroup_barrier();
    reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs.max(rhs));

    let max_local = program.private(ElementType::F32);
    let max_score = program.load_workgroup(reduce, 0u32);
    program.store_local(&max_local, max_score);
    let max_score = program.load_local(&max_local);

    program.workgroup_barrier();
    let raw_prob = (score - max_score.clone()).exp();
    let prob = Tile::select(valid, raw_prob, Tile::literal(TileLiteral::f32(0.0)));
    if let Some(probs) = probs {
        program.store_workgroup(probs, lane.clone(), prob.clone());
    }
    program.store_workgroup(reduce, lane.clone(), prob.clone());
    program.workgroup_barrier();
    reduce_workgroup(program, reduce, lane, |lhs, rhs| lhs + rhs);
    let denom = program.load_workgroup(reduce, 0u32);

    WorkgroupSoftmaxBlock {
        max: max_score,
        denom,
        prob,
    }
}

pub fn softmax<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    element: ElementType,
    input: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    if !supports_float(element) || !supported_block(meta.block) || meta.split_blocks != 1 {
        return None;
    }
    let input = kb.read(element, input);
    let output = kb.write(element, output);
    let phase = kb.program();
    let reduce = phase.alloc_workgroup_array(ScalarElement::F32, meta.block);

    phase.program_grid(meta.block, meta.dispatch_size, |program| {
        let lane = program.lane();
        let row = linear_group(program, meta.dispatch_size);
        let axis_value = lane.clone();
        let valid = row
            .clone()
            .lt(u32_tile(meta.rows))
            .and(axis_value.clone().lt(u32_tile(meta.axis_len)));
        let input_index = storage_index(program, &meta, row.clone(), axis_value.clone(), false);
        let score = program
            .load(input.at(input_index), valid.clone(), neg_max_fill(element))
            .cast(ElementType::F32);
        let stats = workgroup_softmax_block(program, lane, score, valid.clone(), &reduce, None);
        let output_index = storage_index(program, &meta, row, axis_value, true);
        let value = (stats.prob / stats.denom).cast(element);
        program.store(output.at(output_index), value, valid);
    });
    Some(())
}

pub fn softmax_partials<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    element: ElementType,
    input: fusor_tile_ir::KernelTensorRef<B>,
    scratch: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    if !supports_float(element) || !supported_block(meta.block) || meta.split_blocks < 2 {
        return None;
    }
    let input = kb.read(element, input);
    let scratch = kb.write(ElementType::F32, scratch);
    let phase = kb.program();
    let reduce = phase.alloc_workgroup_array(ScalarElement::F32, meta.block);

    phase.program_grid(meta.block, meta.dispatch_size, |program| {
        let lane = program.lane();
        let group = linear_group(program, meta.dispatch_size);
        let total_groups = u32_tile(meta.rows * meta.split_blocks);
        let group_valid = group.clone().lt(total_groups);
        let row = program.bind(group.clone() % u32_tile(meta.rows));
        let split = program.bind(group / u32_tile(meta.rows));
        let axis_value = program.bind(split.clone() * u32_tile(meta.block) + lane.clone());
        let valid = group_valid
            .clone()
            .and(axis_value.clone().lt(u32_tile(meta.axis_len)));
        let input_index = storage_index(program, &meta, row.clone(), axis_value, false);
        let score = program
            .load(input.at(input_index), valid.clone(), neg_max_fill(element))
            .cast(ElementType::F32);
        let stats = workgroup_softmax_block(program, lane.clone(), score, valid, &reduce, None);
        let partial_base = program.bind((row * u32_tile(meta.split_blocks) + split) * u32_tile(2));
        program.if_then(group_valid.and(lane.eq(u32_tile(0))), |program| {
            program.store(scratch.at(partial_base.clone()), stats.denom, Mask::all());
            program.store(
                scratch.at(partial_base + u32_tile(1)),
                stats.max,
                Mask::all(),
            );
        });
    });
    Some(())
}

pub fn softmax_write<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    element: ElementType,
    input: fusor_tile_ir::KernelTensorRef<B>,
    global: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    if !supports_float(element) || !supported_block(meta.block) || meta.split_blocks < 2 {
        return None;
    }
    let input = kb.read(element, input);
    let global = kb.read(ElementType::F32, global);
    let output = kb.write(element, output);
    let phase = kb.program();

    phase.program_grid(meta.block, meta.dispatch_size, |program| {
        let lane = program.lane();
        let group = linear_group(program, meta.dispatch_size);
        let total_groups = u32_tile(meta.rows * meta.split_blocks);
        let group_valid = group.clone().lt(total_groups);
        let row = program.bind(group.clone() % u32_tile(meta.rows));
        let split = program.bind(group / u32_tile(meta.rows));
        let axis_value = program.bind(split * u32_tile(meta.block) + lane.clone());

        let row_base = program.bind(row.clone() * u32_tile(2));
        let denom = program.load(
            global.at(row_base.clone()),
            group_valid.clone(),
            TileLiteral::f32(0.0),
        );
        let max_score = program.load(
            global.at(row_base + u32_tile(1)),
            group_valid.clone(),
            TileLiteral::f32(NEG_MAX_F32),
        );

        let valid = group_valid.and(axis_value.clone().lt(u32_tile(meta.axis_len)));
        let input_index = storage_index(program, &meta, row.clone(), axis_value.clone(), false);
        let output_index = storage_index(program, &meta, row, axis_value, true);
        let value = program
            .load(input.at(input_index), valid.clone(), zero_fill(element))
            .cast(ElementType::F32);
        let prob = (value - max_score).exp() / denom;
        program.store(output.at(output_index), prob.cast(element), valid);
    });
    Some(())
}

pub fn softmax_reduce<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    scratch: fusor_tile_ir::KernelTensorRef<B>,
    global: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    if !supported_block(meta.block) || meta.split_blocks < 2 {
        return None;
    }
    let scratch = kb.read(ElementType::F32, scratch);
    let global = kb.write(ElementType::F32, global);
    let phase = kb.program();
    let block = meta.block;

    phase.program_grid(block, meta.dispatch_size, |program| {
        let lane = program.lane();
        let row = linear_group(program, meta.dispatch_size);
        let row_valid = row.clone().lt(u32_tile(meta.rows));
        let partial_row_base = program.bind(row.clone() * u32_tile(meta.split_blocks * 2));

        program.if_then(row_valid.and(lane.eq(u32_tile(0))), |program| {
            let mut max_score = Tile::literal(TileLiteral::f32(NEG_MAX_F32));
            for split in 0..meta.split_blocks {
                let block_max = program.load(
                    scratch.at(partial_row_base.clone() + u32_tile(split * 2 + 1)),
                    Mask::all(),
                    TileLiteral::f32(NEG_MAX_F32),
                );
                max_score = max_score.max(block_max);
            }

            let mut denom = Tile::literal(TileLiteral::f32(0.0));
            for split in 0..meta.split_blocks {
                let block_base = partial_row_base.clone() + u32_tile(split * 2);
                let block_denom = program.load(
                    scratch.at(block_base.clone()),
                    Mask::all(),
                    TileLiteral::f32(0.0),
                );
                let block_max = program.load(
                    scratch.at(block_base + u32_tile(1)),
                    Mask::all(),
                    TileLiteral::f32(NEG_MAX_F32),
                );
                denom = denom + block_denom * softmax_partial_scale(block_max, max_score.clone());
            }

            let global_base = program.bind(row * u32_tile(2));
            program.store(global.at(global_base.clone()), denom, Mask::all());
            program.store(global.at(global_base + u32_tile(1)), max_score, Mask::all());
        });
    });
    Some(())
}
