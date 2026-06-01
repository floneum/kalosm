//! Non-paired Q6K qgemv built on the ggml super-block-amortized decode.
//!
//! Mirrors [`crate::kernels::qgemv::qgemv_q4k_ggml`] exactly, but for the Q6K
//! 256-element super-block layout: `d` (one f32 scale), 128 bytes of low 4-bit
//! weights (`ql`), 64 bytes of high 2-bit weights (`qh`), then 16 signed 8-bit
//! sub-block scales. A 32-lane subgroup covers 2 super-blocks per pass
//! (`ix = lane % 2`); the 16 lanes assigned to a super-block each decode the
//! per-super-block `d` once and the four sub-block scales for their 16-element
//! region, instead of re-decoding that metadata for every 8/16-element chunk
//! the generic `quantized_dot_f32` path takes.
//!
//! Only the f32-scale [`GgmlQuantFormat::Q6K`] layout is handled here. The
//! f16-native [`GgmlQuantFormat::Q6KNative`] block is 210 bytes — its block
//! stride is not word-aligned, so the raw-word-load addressing this kernel uses
//! does not apply; that layout stays on the generic perf path.

use fusor_tile_ir::tile::{range, Program, Storage, Tile, TileBlock};
use fusor_tile_ir::{ElementType, GgmlQuantFormat, QuantizedMatrix};

use crate::dispatch::QgemvShape;
use crate::grid::{
    qgemv_grid, qgemv_program_scope, store_qgemv_sums_with_epilogue, QgemvStoreTarget,
};
use crate::types::{matrix_shape, QmatmulEpilogues};

/// Q6K word offset of the super-block scale `d` (an f32 word for the f32-scale
/// layout). The block is `ql[128] qh[64] scales[16] d[4]` = 212 bytes.
const Q6K_D_WORD: u32 = 52;
/// Q6K word offset of the first `qh` (high 2-bit) word: `ql` is 128 bytes.
const Q6K_QH_WORD_BASE: u32 = 32;
/// Q6K word offset of the first scale word: `ql` + `qh` = 192 bytes.
const Q6K_SCALE_WORD_BASE: u32 = 48;

/// Q6K subgroup-lane decomposition. A 32-lane subgroup covers 2 super-blocks per
/// pass: `ix = lane % 2` selects the super-block, `tid = lane / 2` (0..15)
/// addresses one 16-element region inside it — `ip = tid / 8` selects the
/// 128-element half and `il = tid % 8` the group of 4 (`l0 = il * 4`).
pub(crate) struct Q6KLane {
    pub(crate) ix: Tile,
    pub(crate) ip: Tile,
    pub(crate) il: Tile,
    pub(crate) l0: Tile,
}

pub(crate) fn q6k_lane_decomposition(lane: &Tile) -> Q6KLane {
    let tid = lane.clone() / 2u32;
    let ix = lane.clone() % 2u32;
    let ip = tid.clone() / 8u32;
    let il = tid % 8u32;
    let l0 = il.clone() * 4u32;
    Q6KLane { ix, ip, il, l0 }
}

/// Per-lane ggml activations for one Q6K super-block: 16 f32 values gathered
/// with the strided weight layout (`offset = j/4 + (j%4)*32`).
pub(crate) struct Q6KGgmlActs {
    acts: Vec<Tile>,
}

pub(crate) fn load_q6k_ggml_activations(
    program: &mut TileBlock<'_>,
    a: &Storage,
    row: &Tile,
    vector_base: &Tile,
    in_bounds: &Tile,
) -> Q6KGgmlActs {
    let acts = (0..16u32)
        .map(|j| {
            let offset = j / 4 + (j % 4) * 32;
            let scalar = program.load(
                a.at((row.clone(), vector_base.clone() + offset)),
                in_bounds.clone(),
                0.0,
            );
            program.bind(scalar)
        })
        .collect();
    Q6KGgmlActs { acts }
}

/// Non-paired Q6K qgemv built on the ggml super-block-amortized decode — the
/// Q6K analogue of [`crate::kernels::qgemv::qgemv_q4k_ggml`]. A 32-lane subgroup
/// covers 2 super-blocks per pass (`ix = lane % 2`); each of the 16 lanes per
/// super-block decodes `d` and its four sub-block scales once and consumes a
/// strided 16-element region. Only valid with an empty pre-epilogue and the
/// word-aligned f32-scale [`GgmlQuantFormat::Q6K`] layout.
pub(crate) fn qgemv_q6k_ggml(
    program: &mut Program,
    a: &Storage,
    b: &QuantizedMatrix,
    y: &Storage,
    workgroups_x: u32,
    epilogues: &QmatmulEpilogues<'_>,
    shape: QgemvShape,
) {
    const SUBGROUP_SIZE: u32 = 32;
    let block = shape.block;
    let subgroups = shape.subgroups;
    let cols_per_subgroup = shape.cols_per_subgroup;
    debug_assert_eq!(subgroups * SUBGROUP_SIZE, block);
    debug_assert_eq!(b.format, GgmlQuantFormat::Q6K);
    let grid = qgemv_grid(subgroups, cols_per_subgroup, b.cols, workgroups_x);
    let [_, k] = matrix_shape(a.layout());
    let block_count = k.div_ceil(256);
    // 2 super-blocks per subgroup pass (`ix = lane % 2`).
    let block_iterations = block_count.div_ceil(2);
    let full_block_iterations = block_count.is_multiple_of(2);
    let blocks_per_col = b.rows / b.format.block_elements();
    let block_words = b.format.block_words();
    let qwords = Storage::from_view(b.data.clone());
    let cols_usize = cols_per_subgroup as usize;
    let row = Tile::u32(0);

    program.program_grid(block, [grid.workgroups_x, grid.dispatch_y, 1], |program| {
        let scope = qgemv_program_scope(program, grid, cols_per_subgroup);
        let col0 = scope.col0;
        let lane = scope.lane;
        let q6k_lane = q6k_lane_decomposition(&lane);

        let sums: Vec<Tile> = program.fold_vec(
            range(block_iterations),
            vec![Tile::f32(0.0); cols_usize],
            |program, loop_index, accs| {
                let block_idx = loop_index * 2u32 + q6k_lane.ix.clone();
                let in_bounds = if full_block_iterations {
                    Tile::bool(true)
                } else {
                    block_idx.clone().lt(block_count)
                };
                let vector_base =
                    block_idx.clone() * 256u32 + q6k_lane.ip.clone() * 128u32 + q6k_lane.l0.clone();
                let acts = load_q6k_ggml_activations(program, a, &row, &vector_base, &in_bounds);
                accs.into_iter()
                    .enumerate()
                    .map(|(c, acc)| {
                        let col = col0.clone() + c as u32;
                        acc + q6k_ggml_dot_tiles(
                            program,
                            &qwords,
                            blocks_per_col,
                            block_words,
                            &block_idx,
                            &col,
                            &q6k_lane,
                            &acts,
                        )
                    })
                    .collect()
            },
        );

        store_qgemv_sums_with_epilogue(
            program,
            sums,
            QgemvStoreTarget {
                y,
                col0,
                lane,
                full_cols: grid.full_cols,
                n_cols: grid.n_cols,
                epilogues,
            },
        );
    });
}

/// `signed_byte_f32`: reinterpret an unsigned 0..255 byte as a signed 8-bit
/// value in f32. `(byte ^ 128)` casts the unsigned magnitude, then `- 128`
/// restores the sign — identical to the lowering's `signed_byte_f32`.
fn signed_byte_f32(byte: Tile) -> Tile {
    (byte ^ 128u32).cast(ElementType::F32) - Tile::f32(128.0)
}

/// Extract byte `lane` (0..3) from a packed u32 word as a u32 in 0..255.
fn byte_at(word: &Tile, lane: u32) -> Tile {
    word.clone().shift_right(lane * 8) & 0xffu32
}

/// Compose one Q6K ggml per-column dot from tile primitives. Reads the lane's
/// `ql`/`qh` words and four sub-block scales straight from the matrix buffer,
/// reconstructs the 6-bit weights (`low4 | high2<<4`), centers them by 32, and
/// folds the four 16-element sub-block partial sums against their scales scaled
/// by the per-super-block `d`. Decodes `d`/scales once per 16-element lane
/// region instead of once per 8.
#[allow(clippy::too_many_arguments)]
pub(crate) fn q6k_ggml_dot_tiles(
    program: &mut TileBlock<'_>,
    qwords: &Storage,
    blocks_per_col: u32,
    block_words: u32,
    block: &Tile,
    col: &Tile,
    lane: &Q6KLane,
    acts: &Q6KGgmlActs,
) -> Tile {
    let base = (col.clone() * blocks_per_col + block.clone()) * block_words;
    // Weights/scales are read unconditionally (constant-true mask) so each lowers
    // to a direct pointer load. Out-of-bounds K is zeroed via the masked
    // activations; out-of-bounds columns are discarded by the store mask.
    let load = |program: &mut TileBlock<'_>, offset: Tile| -> Tile {
        program.load(qwords.at(base.clone() + offset), Tile::all(), 0u32)
    };

    // Super-block scale `d` (f32-scale layout): one f32 word at word 52.
    let d = load(program, Tile::u32(Q6K_D_WORD)).bitcast(ElementType::F32);

    // Low 4-bit weights (`ql`): word offset `(ip*64 + il*4) >> 2`, second word +8.
    let low_byte_offset = lane.ip.clone() * 64u32 + lane.l0.clone();
    let low_word_offset = low_byte_offset.shift_right(2u32);
    let q1_word = load(program, low_word_offset.clone());
    let q2_word = load(program, low_word_offset + 8u32);

    // High 2-bit weights (`qh`): word offset `((ip*32 + il*4) >> 2) + 32`.
    let high_byte_offset = lane.ip.clone() * 32u32 + lane.l0.clone();
    let high_word_offset = high_byte_offset.shift_right(2u32) + Q6K_QH_WORD_BASE;
    let qh_word = load(program, high_word_offset);

    // Four sub-block scales: index `ip*8 + (il >> 2)`, two words at `(idx>>2)+48`
    // and `+1`, bytes at `idx&3` and `idx&3 + 2`.
    let scale_index = lane.ip.clone() * 8u32 + lane.il.clone().shift_right(2u32);
    let scale_word0_offset = scale_index.clone().shift_right(2u32) + Q6K_SCALE_WORD_BASE;
    let scale_word1_offset = scale_word0_offset.clone() + 1u32;
    let scale_word0 = load(program, scale_word0_offset);
    let scale_word1 = load(program, scale_word1_offset);
    let scale_lane0 = scale_index & 3u32;
    let scale_lane1 = scale_lane0.clone() + 2u32;
    let scale_lane0_shift = scale_lane0.shift_left(3u32);
    let scale_lane1_shift = scale_lane1.shift_left(3u32);
    let scale_byte = |word: &Tile, shift: &Tile| -> Tile {
        signed_byte_f32(word.clone().shift_right(shift.clone()) & 0xffu32)
    };
    let scales = [
        scale_byte(&scale_word0, &scale_lane0_shift),
        scale_byte(&scale_word0, &scale_lane1_shift),
        scale_byte(&scale_word1, &scale_lane0_shift),
        scale_byte(&scale_word1, &scale_lane1_shift),
    ];

    // Accumulate the four sub-block partial sums over the 4 byte-lanes.
    let mut sums: [Tile; 4] = std::array::from_fn(|_| Tile::f32(0.0));
    let center = Tile::f32(32.0);
    for l in 0..4u32 {
        let q1_byte = byte_at(&q1_word, l);
        let q2_byte = byte_at(&q2_word, l);
        let qh_byte = byte_at(&qh_word, l);

        let q0 = (q1_byte.clone() & 0x0fu32) | (qh_byte.clone() & 0x03u32).shift_left(4u32);
        let q1 = (q2_byte.clone() & 0x0fu32) | (qh_byte.clone() & 0x0cu32).shift_left(2u32);
        let q2 = q1_byte.shift_right(4u32) | (qh_byte.clone() & 0x30u32);
        let q3 = q2_byte.shift_right(4u32) | (qh_byte & 0xc0u32).shift_right(2u32);

        let quants = [q0, q1, q2, q3];
        for (s, quant) in quants.into_iter().enumerate() {
            let centered = quant.cast(ElementType::F32) - center.clone();
            let act = acts.acts[(4 * l as usize) + s].clone();
            sums[s] = sums[s].clone() + act * centered;
        }
    }

    let weighted = sums[0].clone() * scales[0].clone()
        + sums[1].clone() * scales[1].clone()
        + sums[2].clone() * scales[2].clone()
        + sums[3].clone() * scales[3].clone();
    d * weighted
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::QgemvShape;
    use crate::kernels::quantized_matrix;
    use crate::types::QmatmulEpilogues;
    use fusor_tile_ir::{tile, Shape};

    fn qgemv_shape(block: u32, subgroups: u32, cols_per_subgroup: u32) -> QgemvShape {
        QgemvShape {
            subgroups,
            cols_per_subgroup,
            block,
        }
    }

    fn build_and_lower(rows: u32, cols: u32, shape: QgemvShape) {
        let ir = tile::build(|program| {
            let a = program.storage_read(ElementType::F32, Shape::new([1, rows]));
            let b = quantized_matrix(program, GgmlQuantFormat::Q6K, rows, cols);
            let y = program.storage_write(ElementType::F32, Shape::new([1, cols]));
            let epilogues = QmatmulEpilogues::default();
            qgemv_q6k_ggml(program, &a, &b, &y, 1, &epilogues, shape);
        });
        // Lowering to Naga validates the IR structure end to end.
        ir.lower_to_naga()
            .expect("qgemv_q6k_ggml IR must lower to a valid Naga module");
    }

    #[test]
    fn lowers_small_shape() {
        // Two super-blocks, single column: exercises the partial-iteration mask
        // (block_count = 2 is a multiple of 2, so full iterations here).
        build_and_lower(512, 4, qgemv_shape(256, 8, 4));
    }

    #[test]
    fn lowers_partial_block_iteration() {
        // block_count = 3 (768/256) is not a multiple of 2: exercises the
        // partial-block-iteration bounds mask.
        build_and_lower(768, 4, qgemv_shape(128, 4, 4));
    }

    #[test]
    fn lowers_llama_decode_shape() {
        // The regression target: 4096x14336 Q6K decode matmul.
        build_and_lower(4096, 14336, qgemv_shape(256, 8, 4));
    }
}
