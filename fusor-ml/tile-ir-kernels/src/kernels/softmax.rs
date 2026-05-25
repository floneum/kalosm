use fusor_tile_ir::{
    ElementType, F32, Numeric, TileLiteral, U32, WorkgroupAxis,
    tile::{Mask, Tile, TileBlock, Workgroup},
};

use super::{
    helpers::{NEG_MAX_F32, reduce_workgroup},
    types::SoftmaxMeta,
};

#[derive(Clone)]
pub(super) struct WorkgroupSoftmaxBlock {
    pub max: Tile<F32>,
    pub denom: Tile<F32>,
    pub prob: Tile<F32>,
}

fn zero_fill<E: Numeric>() -> TileLiteral {
    match E::ELEMENT {
        ElementType::F32 => TileLiteral::f32(0.0),
        ElementType::F16 => TileLiteral::F16(half::f16::from_f32(0.0).to_bits()),
        _ => panic!("softmax only supports F32 and F16 element types"),
    }
}

fn neg_max_fill<E: Numeric>() -> TileLiteral {
    match E::ELEMENT {
        ElementType::F32 => TileLiteral::f32(NEG_MAX_F32),
        ElementType::F16 => TileLiteral::F16(half::f16::from_f32(NEG_MAX_F32).to_bits()),
        _ => panic!("softmax only supports F32 and F16 element types"),
    }
}

fn supports_float<E: Numeric>() -> bool {
    matches!(E::ELEMENT, ElementType::F32 | ElementType::F16)
}

fn u32_tile(value: u32) -> Tile<U32> {
    Tile::literal(TileLiteral::U32(value))
}

fn linear_group(program: &TileBlock<'_>, dispatch_size: [u32; 3]) -> Tile<U32> {
    let x = program.program_id(WorkgroupAxis::X);
    let y = program.program_id(WorkgroupAxis::Y);
    let z = program.program_id(WorkgroupAxis::Z);
    x + y * u32_tile(dispatch_size[0]) + z * u32_tile(dispatch_size[0] * dispatch_size[1])
}

fn storage_index(
    program: &mut TileBlock<'_>,
    meta: &SoftmaxMeta,
    row: Tile<U32>,
    axis_value: Tile<U32>,
    output: bool,
) -> Tile<U32> {
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

pub(super) fn softmax_partial_scale(block_max: Tile<F32>, global_max: Tile<F32>) -> Tile<F32> {
    (block_max - global_max).exp()
}

pub(super) fn workgroup_softmax_block(
    program: &mut TileBlock<'_>,
    lane: Tile<U32>,
    score: Tile<F32>,
    valid: Mask,
    reduce: Workgroup<F32>,
    probs: Option<Workgroup<F32>>,
) -> WorkgroupSoftmaxBlock {
    let score = Tile::select(
        valid.clone(),
        score,
        Tile::literal(TileLiteral::f32(NEG_MAX_F32)),
    );
    program.store_workgroup(reduce, lane.clone(), score.clone());
    program.workgroup_barrier();
    reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs.max(rhs));

    let max_local = program.private::<F32>();
    let max_score = program.load_workgroup(reduce, 0);
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
    let denom = program.load_workgroup(reduce, 0);

    WorkgroupSoftmaxBlock {
        max: max_score,
        denom,
        prob,
    }
}

fn softmax_block<E: Numeric, const BLOCK: usize, B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    input: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    if !supports_float::<E>() || meta.block != BLOCK as u32 || meta.split_blocks != 1 {
        return None;
    }
    let input = kb.read::<E, 1>(input);
    let output = kb.write::<E, 1>(output);
    let phase = kb.program();
    let reduce = phase.alloc_workgroup_array::<F32>(BLOCK as u32);

    phase.program_grid::<BLOCK>(meta.dispatch_size, |program| {
        let lane = program.lane();
        let row = linear_group(program, meta.dispatch_size);
        let axis_value = program.index(lane.clone());
        let valid = row
            .clone()
            .lt(u32_tile(meta.rows))
            .and(axis_value.clone().lt(u32_tile(meta.axis_len)));
        let input_index = storage_index(program, &meta, row.clone(), axis_value.clone(), false);
        let score = program
            .load(input.at(input_index), valid.clone(), neg_max_fill::<E>())
            .cast::<F32>();
        let stats = workgroup_softmax_block(program, lane, score, valid.clone(), reduce, None);
        let output_index = storage_index(program, &meta, row, axis_value, true);
        let value = (stats.prob / stats.denom).cast::<E>();
        program.store(output.at(output_index), value, valid);
    });
    Some(())
}

fn softmax_partials_block<E: Numeric, const BLOCK: usize, B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    input: fusor_tile_ir::KernelTensorRef<B>,
    scratch: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    if !supports_float::<E>() || meta.block != BLOCK as u32 || meta.split_blocks < 2 {
        return None;
    }
    let input = kb.read::<E, 1>(input);
    let scratch = kb.write::<F32, 1>(scratch);
    let phase = kb.program();
    let reduce = phase.alloc_workgroup_array::<F32>(BLOCK as u32);

    phase.program_grid::<BLOCK>(meta.dispatch_size, |program| {
        let lane = program.lane();
        let group = linear_group(program, meta.dispatch_size);
        let total_groups = u32_tile(meta.rows * meta.split_blocks);
        let group_valid = group.clone().lt(total_groups);
        let row = program.bind(group.clone() % u32_tile(meta.rows));
        let split = program.bind(group / u32_tile(meta.rows));
        let axis_value =
            program.bind(split.clone() * u32_tile(meta.block) + program.index(lane.clone()));
        let valid = group_valid
            .clone()
            .and(axis_value.clone().lt(u32_tile(meta.axis_len)));
        let input_index = storage_index(program, &meta, row.clone(), axis_value, false);
        let score = program
            .load(input.at(input_index), valid.clone(), neg_max_fill::<E>())
            .cast::<F32>();
        let stats = workgroup_softmax_block(program, lane.clone(), score, valid, reduce, None);
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

fn softmax_write_block<E: Numeric, const BLOCK: usize, B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    input: fusor_tile_ir::KernelTensorRef<B>,
    global: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    if !supports_float::<E>() || meta.block != BLOCK as u32 || meta.split_blocks < 2 {
        return None;
    }
    let input = kb.read::<E, 1>(input);
    let global = kb.read::<F32, 1>(global);
    let output = kb.write::<E, 1>(output);
    let phase = kb.program();

    phase.program_grid::<BLOCK>(meta.dispatch_size, |program| {
        let lane = program.lane();
        let group = linear_group(program, meta.dispatch_size);
        let total_groups = u32_tile(meta.rows * meta.split_blocks);
        let group_valid = group.clone().lt(total_groups);
        let row = program.bind(group.clone() % u32_tile(meta.rows));
        let split = program.bind(group / u32_tile(meta.rows));
        let axis_value = program.bind(split * u32_tile(meta.block) + program.index(lane.clone()));

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
            .load(input.at(input_index), valid.clone(), zero_fill::<E>())
            .cast::<F32>();
        let prob = (value - max_score).exp() / denom;
        program.store(output.at(output_index), prob.cast::<E>(), valid);
    });
    Some(())
}

fn softmax_reduce_block<const BLOCK: usize, B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    scratch: fusor_tile_ir::KernelTensorRef<B>,
    global: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    if meta.block != BLOCK as u32 || meta.split_blocks < 2 {
        return None;
    }
    let scratch = kb.read::<F32, 1>(scratch);
    let global = kb.write::<F32, 1>(global);
    let phase = kb.program();
    let reduce = phase.alloc_workgroup_array::<F32>(BLOCK as u32);

    phase.program_grid::<BLOCK>(meta.dispatch_size, |program| {
        let lane = program.lane();
        let row = linear_group(program, meta.dispatch_size);
        let row_valid = row.clone().lt(u32_tile(meta.rows));
        let partial_row_base = program.bind(row.clone() * u32_tile(meta.split_blocks * 2));

        let mut local_max = Tile::literal(TileLiteral::f32(NEG_MAX_F32));
        for split_base in (0..meta.split_blocks).step_by(BLOCK) {
            let split = u32_tile(split_base) + lane.clone();
            let valid = row_valid
                .clone()
                .and(split.clone().lt(u32_tile(meta.split_blocks)));
            let block_max = program.load(
                scratch.at(partial_row_base.clone() + split * u32_tile(2) + u32_tile(1)),
                valid,
                TileLiteral::f32(NEG_MAX_F32),
            );
            local_max = local_max.max(block_max);
        }
        program.store_workgroup(reduce, lane.clone(), local_max);
        program.workgroup_barrier();
        reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs.max(rhs));

        let max_local = program.private::<F32>();
        let max_score = program.load_workgroup(reduce, 0);
        program.store_local(&max_local, max_score);
        let max_score = program.load_local(&max_local);

        program.workgroup_barrier();
        let mut local_denom = Tile::literal(TileLiteral::f32(0.0));
        for split_base in (0..meta.split_blocks).step_by(BLOCK) {
            let split = u32_tile(split_base) + lane.clone();
            let valid = row_valid
                .clone()
                .and(split.clone().lt(u32_tile(meta.split_blocks)));
            let block_base = partial_row_base.clone() + split * u32_tile(2);
            let block_denom = program.load(
                scratch.at(block_base.clone()),
                valid.clone(),
                TileLiteral::f32(0.0),
            );
            let block_max = program.load(
                scratch.at(block_base + u32_tile(1)),
                valid,
                TileLiteral::f32(NEG_MAX_F32),
            );
            local_denom =
                local_denom + block_denom * softmax_partial_scale(block_max, max_score.clone());
        }
        program.store_workgroup(reduce, lane.clone(), local_denom);
        program.workgroup_barrier();
        reduce_workgroup(program, reduce, lane.clone(), |lhs, rhs| lhs + rhs);

        let global_base = program.bind(row * u32_tile(2));
        program.if_then(row_valid.and(lane.eq(u32_tile(0))), |program| {
            program.store(
                global.at(global_base.clone()),
                program.load_workgroup(reduce, 0),
                Mask::all(),
            );
            program.store(global.at(global_base + u32_tile(1)), max_score, Mask::all());
        });
    });
    Some(())
}

pub fn softmax<E: Numeric, B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    input: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    match meta.block {
        128 => softmax_block::<E, 128, B>(kb, input, output, meta),
        512 => softmax_block::<E, 512, B>(kb, input, output, meta),
        1024 => softmax_block::<E, 1024, B>(kb, input, output, meta),
        _ => None,
    }
}

pub fn softmax_partials<E: Numeric, B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    input: fusor_tile_ir::KernelTensorRef<B>,
    scratch: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    match meta.block {
        128 => softmax_partials_block::<E, 128, B>(kb, input, scratch, meta),
        512 => softmax_partials_block::<E, 512, B>(kb, input, scratch, meta),
        1024 => softmax_partials_block::<E, 1024, B>(kb, input, scratch, meta),
        _ => None,
    }
}

pub fn softmax_reduce<B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    scratch: fusor_tile_ir::KernelTensorRef<B>,
    global: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    match meta.block {
        128 => softmax_reduce_block::<128, B>(kb, scratch, global, meta),
        512 => softmax_reduce_block::<512, B>(kb, scratch, global, meta),
        1024 => softmax_reduce_block::<1024, B>(kb, scratch, global, meta),
        _ => None,
    }
}

pub fn softmax_write<E: Numeric, B>(
    kb: &mut fusor_tile_ir::KernelBuilder<B>,
    input: fusor_tile_ir::KernelTensorRef<B>,
    global: fusor_tile_ir::KernelTensorRef<B>,
    output: fusor_tile_ir::KernelTensorRef<B>,
    meta: SoftmaxMeta,
) -> Option<()> {
    match meta.block {
        128 => softmax_write_block::<E, 128, B>(kb, input, global, output, meta),
        512 => softmax_write_block::<E, 512, B>(kb, input, global, output, meta),
        1024 => softmax_write_block::<E, 1024, B>(kb, input, global, output, meta),
        _ => None,
    }
}
