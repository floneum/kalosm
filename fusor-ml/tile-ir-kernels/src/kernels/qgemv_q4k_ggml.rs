use fusor_tile_ir::tile::{Mask, Storage, Tile, TileBlock};
use fusor_tile_ir::{ElementType, ScalarElement};

/// Q4K subgroup-lane decomposition: `ix = lane / 8` selects the super-block
/// within this runtime subgroup pass; `(iq, ir) = ((lane % 8) / 4, lane % 4)`
/// addresses the lane's 8-byte sub-region within that block.
pub(crate) struct Q4KLane {
    pub(crate) ix: Tile,
    pub(crate) iq: Tile,
    pub(crate) ir: Tile,
}

pub(crate) fn q4k_lane_decomposition(lane: &Tile) -> Q4KLane {
    let ix = lane.clone() / 8u32;
    let it = lane.clone() % 8u32;
    let iq = it.clone() / 4u32;
    let ir = it % 4u32;
    Q4KLane { ix, iq, ir }
}

/// Per-lane ggml activations for one Q4K super-block: 16 "low" + 16 "high" f32
/// values gathered with the strided nibble layout, plus the 4 partial sums used
/// by the affine-min correction.
pub(crate) struct Q4KGgmlActs {
    low: Vec<Tile>,
    high: Vec<Tile>,
    sums: [Tile; 4],
}

pub(crate) fn load_q4k_ggml_activations(
    program: &mut TileBlock<'_>,
    a: &Storage,
    row: &Tile,
    vector_base: &Tile,
    in_bounds: &Tile,
) -> Q4KGgmlActs {
    let load_quad = |program: &mut TileBlock<'_>, base: u32| -> Vec<Tile> {
        (0..16u32)
            .map(|j| {
                let offset = if j < 8 { j } else { (j - 8) + 32 } + base;
                let scalar = program.load(
                    a.at((row.clone(), vector_base.clone() + offset)),
                    in_bounds.clone(),
                    0.0,
                );
                program.bind(scalar)
            })
            .collect()
    };
    let low = load_quad(program, 0);
    let high = load_quad(program, 128);
    let mut sums: [Tile; 4] = std::array::from_fn(|_| Tile::f32(0.0));
    for j in 0..8 {
        sums[0] = sums[0].clone() + low[j].clone();
        sums[1] = sums[1].clone() + low[j + 8].clone();
        sums[2] = sums[2].clone() + high[j].clone();
        sums[3] = sums[3].clone() + high[j + 8].clone();
    }
    Q4KGgmlActs { low, high, sums }
}

/// Compose one Q4K ggml per-column dot from tile primitives.
#[allow(clippy::too_many_arguments)]
pub(crate) fn q4k_ggml_dot_tiles(
    program: &mut TileBlock<'_>,
    qwords: &Storage,
    blocks_per_col: u32,
    block_words: u32,
    native: bool,
    block: &Tile,
    col: &Tile,
    lane: &Q4KLane,
    mask: Mask,
    acts: &Q4KGgmlActs,
) -> Tile {
    let base = (col.clone() * blocks_per_col + block.clone()) * block_words;
    let load = |program: &mut TileBlock<'_>, offset: Tile| -> Tile {
        program.load(qwords.at(base.clone() + offset), mask.clone(), 0u32)
    };

    let (scale0, data_base) = if native { (1u32, 4u32) } else { (2u32, 5u32) };

    let (d, dmin) = if native {
        let pair = load(program, Tile::u32(0)).unpack2x16float();
        let lo = program.compose_vector::<2>(ScalarElement::F32, [Tile::f32(1.0), Tile::f32(0.0)]);
        let hi = program.compose_vector::<2>(ScalarElement::F32, [Tile::f32(0.0), Tile::f32(1.0)]);
        (
            program.vector_dot(pair.clone(), lo),
            program.vector_dot(pair, hi),
        )
    } else {
        (
            load(program, Tile::u32(0)).bitcast(ElementType::F32),
            load(program, Tile::u32(1)).bitcast(ElementType::F32),
        )
    };

    let scale_shift = lane.iq.clone().shift_left(4u32);
    let sc0 = load(program, Tile::u32(scale0)).shift_right(scale_shift.clone());
    let sc1 = load(program, Tile::u32(scale0 + 1)).shift_right(scale_shift.clone());
    let sc2 = load(program, Tile::u32(scale0 + 2)).shift_right(scale_shift);
    let first_two = sc0.clone() & 0x3f3fu32;
    let second_two = sc1.clone() & 0x3f3fu32;
    let third_low = sc2.clone() & 0x0f0fu32;
    let third_high = (sc0 & 0xc0c0u32).shift_right(2u32);
    let third_two = third_low | third_high;
    let fourth_low = sc2.shift_right(4u32) & 0x0f0fu32;
    let fourth_high = (sc1 & 0xc0c0u32).shift_right(2u32);
    let fourth_two = fourth_low | fourth_high;

    let u8_f32 = |x: &Tile, byte: u32| -> Tile {
        (x.clone().shift_right(byte * 8) & 0xffu32).cast(ElementType::F32)
    };
    let odd = [
        u8_f32(&first_two, 0),
        u8_f32(&first_two, 1),
        u8_f32(&third_two, 0),
        u8_f32(&third_two, 1),
    ];
    let even = [
        u8_f32(&second_two, 0),
        u8_f32(&second_two, 1),
        u8_f32(&fourth_two, 0),
        u8_f32(&fourth_two, 1),
    ];

    let data_offset = lane.iq.clone().shift_left(3u32) + lane.ir.clone().shift_left(1u32);
    let mut first_sums: [Tile; 4] = std::array::from_fn(|_| Tile::f32(0.0));
    let mut second_sums: [Tile; 4] = std::array::from_fn(|_| Tile::f32(0.0));
    for j in 0..2u32 {
        let word = load(program, data_offset.clone() + (data_base + j));
        accumulate_q4k_word(&word, &acts.low, (j * 4) as usize, &mut first_sums);
        let word_high = load(program, data_offset.clone() + (data_base + 16 + j));
        accumulate_q4k_word(&word_high, &acts.high, (j * 4) as usize, &mut second_sums);
    }

    let inv_256 = Tile::f32(1.0 / 256.0);
    let inv_16 = Tile::f32(1.0 / 16.0);
    let combined: [Tile; 4] = [
        first_sums[0].clone() + first_sums[1].clone() * inv_256.clone(),
        first_sums[2].clone() + first_sums[3].clone() * inv_256.clone(),
        second_sums[0].clone() + second_sums[1].clone() * inv_256.clone(),
        second_sums[2].clone() + second_sums[3].clone() * inv_256.clone(),
    ];
    let scaled_dot = combined[0].clone() * odd[0].clone()
        + combined[1].clone() * odd[1].clone() * inv_16.clone()
        + combined[2].clone() * odd[2].clone()
        + combined[3].clone() * odd[3].clone() * inv_16;
    let scaled_dot = d * scaled_dot;

    let min_dot = acts.sums[0].clone() * even[0].clone()
        + acts.sums[1].clone() * even[1].clone()
        + acts.sums[2].clone() * even[2].clone()
        + acts.sums[3].clone() * even[3].clone();
    let min_dot = dmin * min_dot;

    scaled_dot - min_dot
}

fn accumulate_q4k_word(word: &Tile, acts: &[Tile], act_base: usize, sums: &mut [Tile; 4]) {
    let high_word = word.clone().shift_right(16u32);
    for (source, base) in [(word.clone(), act_base), (high_word, act_base + 2)] {
        sums[0] = sums[0].clone()
            + acts[base].clone() * (source.clone() & 0x000fu32).cast(ElementType::F32);
        sums[1] = sums[1].clone()
            + acts[base + 1].clone() * (source.clone() & 0x0f00u32).cast(ElementType::F32);
        sums[2] = sums[2].clone()
            + acts[base + 8].clone() * (source.clone() & 0x00f0u32).cast(ElementType::F32);
        sums[3] =
            sums[3].clone() + acts[base + 9].clone() * (source & 0xf000u32).cast(ElementType::F32);
    }
}
