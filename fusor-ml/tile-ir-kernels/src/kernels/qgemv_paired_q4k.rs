//! Q4K paired-epilogue GEMV program kernels.

use fusor_tile_ir::tile::{range, Program, Storage, Tile, TileBlock};
use fusor_tile_ir::{ElementType, GgmlQuantFormat, QuantizedMatrix, ScalarElement, WorkgroupAxis};

use crate::types::{matrix_shape, PairedEpilogue};

/// Each lane owns one whole 32-element Q4K sub-block, so the per-column dot
/// decodes the sub-block scale/min exactly once (`quantized_dot_f32` for 32
/// values) instead of re-decoding it every 8 elements. The K-loop advances 32
/// elements per lane.
const VALUES_PER_LANE: u32 = 32;
/// Subgroup width assumed by the Q4K paired tiling.
const SUBGROUP_SIZE: u32 = 32;

/// Tile shape for `qgemv_q4k_paired_ggml`. The kernel only takes `block` (the
/// workgroup-size literal) as a runtime arg; `subgroups` and `pairs_per_subgroup`
/// ride along as runtime args, and `dots_per_subgroup` always equals
/// `pairs_per_subgroup * 2`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Q4KPairedShape {
    pub subgroups: u32,
    pub pairs_per_subgroup: u32,
    pub block: u32,
}

impl Q4KPairedShape {
    pub const fn new(subgroups: u32, pairs_per_subgroup: u32, block: u32) -> Self {
        Self {
            subgroups,
            pairs_per_subgroup,
            block,
        }
    }

    pub const fn pairs_per_workgroup(self) -> u32 {
        self.subgroups * self.pairs_per_subgroup
    }
}

const Q4K_PAIRED_TILES: &[(&str, Q4KPairedShape)] = &[
    ("ggml_2x1", Q4KPairedShape::new(2, 1, 64)),
    ("ggml_2x2", Q4KPairedShape::new(2, 2, 64)),
    ("ggml_2x4", Q4KPairedShape::new(2, 4, 64)),
    ("ggml_4x1", Q4KPairedShape::new(4, 1, 128)),
    ("ggml_4x2", Q4KPairedShape::new(4, 2, 128)),
    ("ggml_4x4", Q4KPairedShape::new(4, 4, 128)),
    ("ggml_8x1", Q4KPairedShape::new(8, 1, 256)),
    ("ggml_8x2", Q4KPairedShape::new(8, 2, 256)),
];

fn q4k_paired_shape() -> Q4KPairedShape {
    const DEFAULT: Q4KPairedShape = Q4KPairedShape::new(4, 4, 128);
    let Ok(value) = std::env::var("FUSOR_Q4K_PAIRED_TILE") else {
        return DEFAULT;
    };
    Q4K_PAIRED_TILES
        .iter()
        .find(|(name, _)| *name == value)
        .map(|(_, shape)| *shape)
        .unwrap_or(DEFAULT)
}

/// Compute launch geometry for the paired Q4K GEMV kernel.
pub fn qgemv_q4k_paired_dispatch(
    pair_cols: u32,
    m_rows: u32,
    max_workgroups_per_dimension: u32,
) -> Option<([u32; 3], u32, Q4KPairedShape)> {
    let shape = q4k_paired_shape();
    let cols_workgroups = pair_cols.div_ceil(shape.pairs_per_workgroup());
    let total_workgroups = cols_workgroups.checked_mul(m_rows.max(1))?;
    let workgroups_x = total_workgroups.min(max_workgroups_per_dimension).max(1);
    let dispatch_size = [workgroups_x, total_workgroups.div_ceil(workgroups_x), 1];
    dispatch_size
        .iter()
        .all(|dim| *dim <= max_workgroups_per_dimension)
        .then_some((dispatch_size, workgroups_x, shape))
}

/// Inputs and launch geometry for the Q4K paired-epilogue GEMV kernels.
///
/// These kernels consume a Q4K matrix whose columns are laid out as
/// `[gate columns, up columns]`. Each kernel computes both halves for a
/// column pair, applies `epilogue` in-register, and writes the paired result.
///
/// ```no_run
/// # use fusor_tile_ir::{tile, GgmlQuantFormat, ScalarElement, Shape};
/// # use fusor_tile_ir_kernels::{
/// #     PairedEpilogue, Q4KPairedGgml, Q4KPairedShape, qgemv_q4k_paired, quantized_matrix,
/// # };
/// let epilogue =
///     PairedEpilogue::with_extras("mul", 0, |tiles| tiles[0].clone() * tiles[1].clone());
/// let ir = tile::build(|program| {
///     let f32 = ScalarElement::F32.element();
///     let a = program.storage_read(f32, Shape::new([1, 4096]));
///     let b = quantized_matrix(program, GgmlQuantFormat::Q4K, 4096, 8192);
///     let y = program.storage_write(f32, Shape::new([1, 4096]));
///     qgemv_q4k_paired(
///         program,
///         Q4KPairedGgml {
///             a: &a,
///             b: &b,
///             y: &y,
///             pair_cols: 4096,
///             m_rows: 1,
///             workgroups_x: 1,
///             shape: Q4KPairedShape::new(8, 2, 256),
///             epilogue: &epilogue,
///             extras: &[],
///         },
///     );
/// });
/// ```
pub struct Q4KPairedGgml<'a> {
    /// Single-row or batched activation matrix.
    pub a: &'a Storage,
    /// Q4K matrix with `pair_cols * 2` columns.
    pub b: &'a QuantizedMatrix,
    /// Output matrix with `pair_cols` columns.
    pub y: &'a Storage,
    /// Number of gate/up pairs in `b`.
    pub pair_cols: u32,
    /// Number of rows from `a` and `y` covered by the launch.
    pub m_rows: u32,
    /// Preferred dispatch width on X. Clamped to the kernel's total workgroup count.
    pub workgroups_x: u32,
    /// Workgroup/subgroup decomposition for each paired output tile.
    pub shape: Q4KPairedShape,
    /// Register-level operation applied to each `(gate, up)` pair.
    pub epilogue: &'a PairedEpilogue,
    /// One-dimensional extra tensors consumed by `epilogue`.
    pub extras: &'a [Storage],
}

/// Q4K paired-epilogue qgemv. Reduces the gate and up halves of a `[gate; up]`
/// matmul output and applies the supplied `PairedEpilogue` in-register before
/// the single output store.
///
/// Dispatches on the matrix layout: the GPU f32-scale layout
/// ([`GgmlQuantFormat::Q4K`]) takes the fast ggml decomposition
/// ([`qgemv_q4k_paired_ggml`]); any other layout (e.g. the f16-native scales)
/// falls back to the format-agnostic dequantize path
/// ([`qgemv_q4k_paired_dequant`]).
pub fn qgemv_q4k_paired(program: &mut Program, spec: Q4KPairedGgml<'_>) {
    if spec.b.format.is_q4k_family() {
        qgemv_q4k_paired_ggml(program, spec);
    } else {
        qgemv_q4k_paired_dequant(program, spec);
    }
}

/// Format-agnostic fallback: each lane owns one whole 32-element Q4K sub-block,
/// so the per-column dot decodes the sub-block scale/min once
/// (`quantized_dot_f32` for 32 values). Correct for every Q4K storage layout but
/// slower than the ggml path because it materializes the weights to f32.
fn qgemv_q4k_paired_dequant(program: &mut Program, spec: Q4KPairedGgml<'_>) {
    let Q4KPairedGgml {
        a,
        b,
        y,
        pair_cols,
        m_rows,
        workgroups_x,
        shape,
        epilogue,
        extras,
    } = spec;
    let subgroups = shape.subgroups;
    let pairs_per_subgroup = shape.pairs_per_subgroup;
    let pairs_per_subgroup_usize = pairs_per_subgroup as usize;
    let dots_per_subgroup_usize = pairs_per_subgroup_usize * 2;
    debug_assert_eq!(subgroups * SUBGROUP_SIZE, shape.block);
    debug_assert_eq!(
        extras.len(),
        epilogue.extras_count(),
        "kernel extras count must match epilogue arity"
    );
    debug_assert!(b.format.is_q4k_family());
    debug_assert_eq!(b.cols, pair_cols * 2);

    let [_, k] = matrix_shape(a.layout());
    let cols_per_workgroup = subgroups * pairs_per_subgroup;
    let cols_workgroups = pair_cols.div_ceil(cols_per_workgroup);
    let m_rows = m_rows.max(1);
    let total_workgroups = cols_workgroups * m_rows;
    let workgroups_x = workgroups_x.min(total_workgroups.max(1));
    let dispatch_y = total_workgroups.div_ceil(workgroups_x);
    let k_per_iter = SUBGROUP_SIZE * VALUES_PER_LANE;
    let k_iterations = k.div_ceil(k_per_iter);
    let full_k_iterations = k.is_multiple_of(k_per_iter);
    let full_cols = pair_cols.is_multiple_of(cols_per_workgroup);
    let b_cloned = b.clone();

    program.program_grid(shape.block, [workgroups_x, dispatch_y, 1], |program| {
        let workgroup_idx = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * workgroups_x;
        let row = workgroup_idx.clone() / cols_workgroups;
        let col_workgroup = workgroup_idx % cols_workgroups;
        let row_in_bounds = row.clone().lt(m_rows);
        let col_group_base = col_workgroup * cols_per_workgroup;
        let subgroup_col_base = program.subgroup_id() * pairs_per_subgroup;
        let col0 = col_group_base + subgroup_col_base;
        let lane = program.subgroup_lane();

        let sums: Vec<Tile> = program.fold_vec(
            range(k_iterations),
            vec![program.f32(0.0); dots_per_subgroup_usize],
            |program, loop_index, accs| {
                let k_base = loop_index * k_per_iter + lane.clone() * VALUES_PER_LANE;
                let in_bounds_k = if full_k_iterations {
                    row_in_bounds.clone()
                } else {
                    row_in_bounds.clone().and(k_base.lt(k))
                };

                // Activations are shared across every gate/up dot in this
                // subgroup pass: 32 contiguous f32 loads at `k_base + i`.
                let acts: Vec<Tile> = (0..VALUES_PER_LANE)
                    .map(|i| {
                        let scalar = program.load(
                            a.at((row.clone(), k_base.clone() + i)),
                            in_bounds_k.clone(),
                            0.0,
                        );
                        program.bind(scalar)
                    })
                    .collect();

                accs.into_iter()
                    .enumerate()
                    .map(|(idx, acc)| {
                        let offset = idx % pairs_per_subgroup_usize;
                        let gate = col0.clone() + offset as u32;
                        let col = if idx < pairs_per_subgroup_usize {
                            gate.clone()
                        } else {
                            gate.clone() + pair_cols
                        };
                        let mask = if full_cols {
                            in_bounds_k.clone()
                        } else {
                            in_bounds_k.clone().and(gate.lt(pair_cols))
                        };
                        acc + program.quantized_dot_f32(&acts, &b_cloned, &k_base, &col, mask, 0.0)
                    })
                    .collect()
            },
        );

        for offset in 0..pairs_per_subgroup_usize {
            let col = col0.clone() + offset as u32;
            let gate = program.subgroup_reduce_sum(sums[offset].clone());
            let up = program.subgroup_reduce_sum(sums[offset + pairs_per_subgroup_usize].clone());
            let store_lane = if full_cols {
                lane.eq(0u32)
            } else {
                lane.eq(0u32).and(col.lt(pair_cols))
            };
            let mask = store_lane.and(row_in_bounds.clone());
            // Load any per-column extras (e.g. bias vectors) at the current
            // output column. Indexing is `extras[k][col]` — extras are 1D
            // tensors of length `pair_cols`.
            let extra_tiles: Vec<Tile> = extras
                .iter()
                .map(|extra| program.load(extra.at(col.clone()), mask.clone(), 0.0))
                .collect();
            let value = epilogue.apply(gate, up, &extra_tiles);
            program.store(y.at((row.clone(), col)), value, mask);
        }
    });
}

/// Fast Q4K paired qgemv built entirely from tile primitives — no bespoke Naga
/// op. Restores the ggml subgroup decomposition: a 32-lane subgroup covers 4
/// super-blocks per pass (`ix = lane / 8`), each lane owning an `(iq, ir)`
/// 8-byte sub-region. Raw block words are read straight from the matrix buffer
/// (`Storage::from_view(b.data)`); the 6-bit scales/mins are decoded and the
/// 4-bit weights consumed with the mask-multiply-without-shift trick (isolate
/// each nibble by mask only, defer its positional `{1, 1/16} * 1/256` scale to a
/// single fold). Decoding the super-block once per 32-element lane — instead of
/// once per 8 — is what recovers the ~2x over the dequantize path. Handles both
/// Q4K storage layouts: f32-scale and the f16-native `d`/`dmin` header.
fn qgemv_q4k_paired_ggml(program: &mut Program, spec: Q4KPairedGgml<'_>) {
    let Q4KPairedGgml {
        a,
        b,
        y,
        pair_cols,
        m_rows,
        workgroups_x,
        shape,
        epilogue,
        extras,
    } = spec;
    let subgroups = shape.subgroups;
    let pairs_per_subgroup = shape.pairs_per_subgroup;
    let pairs_per_subgroup_usize = pairs_per_subgroup as usize;
    let dots_per_subgroup_usize = pairs_per_subgroup_usize * 2;
    debug_assert_eq!(subgroups * SUBGROUP_SIZE, shape.block);
    debug_assert_eq!(
        extras.len(),
        epilogue.extras_count(),
        "kernel extras count must match epilogue arity"
    );
    debug_assert!(b.format.is_q4k_family());
    debug_assert_eq!(b.cols, pair_cols * 2);

    let [_, k] = matrix_shape(a.layout());
    let cols_per_workgroup = subgroups * pairs_per_subgroup;
    let cols_workgroups = pair_cols.div_ceil(cols_per_workgroup);
    let m_rows = m_rows.max(1);
    let total_workgroups = cols_workgroups * m_rows;
    let workgroups_x = workgroups_x.min(total_workgroups.max(1));
    let dispatch_y = total_workgroups.div_ceil(workgroups_x);
    // 4 super-blocks per subgroup pass (`ix = lane / 8`).
    let block_count = k.div_ceil(256);
    let block_iterations = block_count.div_ceil(4);
    let full_block_iterations = block_count.is_multiple_of(4);
    let full_cols = pair_cols.is_multiple_of(cols_per_workgroup);
    let blocks_per_col = b.rows / b.format.block_elements();
    let block_words = b.format.block_words();
    // `Q4KNative` keeps the ggml f16 `d`/`dmin` header (unpacked per lane);
    // `Q4K` stores them as two f32 words. The two layouts also shift the scale
    // and data word offsets by one.
    let native = b.format == GgmlQuantFormat::Q4KNative;
    let qwords = Storage::from_view(b.data.clone());

    program.program_grid(shape.block, [workgroups_x, dispatch_y, 1], |program| {
        let workgroup_idx = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * workgroups_x;
        let row = workgroup_idx.clone() / cols_workgroups;
        let col_workgroup = workgroup_idx % cols_workgroups;
        let row_in_bounds = row.clone().lt(m_rows);
        let col_group_base = col_workgroup * cols_per_workgroup;
        let subgroup_col_base = program.subgroup_id() * pairs_per_subgroup;
        let col0 = col_group_base + subgroup_col_base;
        let lane = program.subgroup_lane();
        let q4k_lane = q4k_lane_decomposition(&lane);

        let sums: Vec<Tile> = program.fold_vec(
            range(block_iterations),
            vec![Tile::f32(0.0); dots_per_subgroup_usize],
            |program, loop_index, accs| {
                let block = loop_index * 4u32 + q4k_lane.ix.clone();
                let in_bounds = if full_block_iterations {
                    row_in_bounds.clone()
                } else {
                    row_in_bounds.clone().and(block.clone().lt(block_count))
                };
                // The lane's 32 activations are gathered with the same strided
                // pattern the weight nibbles use, plus 4 partial sums for the
                // affine-min term — shared across every gate/up dot this pass.
                let vector_base = block.clone() * 256u32
                    + q4k_lane.iq.clone() * 64u32
                    + q4k_lane.ir.clone() * 8u32;
                let acts = load_q4k_ggml_activations(program, a, &row, &vector_base, &in_bounds);

                accs.into_iter()
                    .enumerate()
                    .map(|(idx, acc)| {
                        let offset = idx % pairs_per_subgroup_usize;
                        let gate = col0.clone() + offset as u32;
                        let col = if idx < pairs_per_subgroup_usize {
                            gate.clone()
                        } else {
                            gate.clone() + pair_cols
                        };
                        acc + q4k_ggml_dot_tiles(
                            program,
                            &qwords,
                            blocks_per_col,
                            block_words,
                            native,
                            &block,
                            &col,
                            &q4k_lane,
                            &acts,
                        )
                    })
                    .collect()
            },
        );

        for offset in 0..pairs_per_subgroup_usize {
            let col = col0.clone() + offset as u32;
            let gate = program.subgroup_reduce_sum(sums[offset].clone());
            let up = program.subgroup_reduce_sum(sums[offset + pairs_per_subgroup_usize].clone());
            let store_lane = if full_cols {
                lane.eq(0u32)
            } else {
                lane.eq(0u32).and(col.lt(pair_cols))
            };
            let mask = store_lane.and(row_in_bounds.clone());
            let extra_tiles: Vec<Tile> = extras
                .iter()
                .map(|extra| program.load(extra.at(col.clone()), mask.clone(), 0.0))
                .collect();
            let value = epilogue.apply(gate, up, &extra_tiles);
            program.store(y.at((row.clone(), col)), value, mask);
        }
    });
}

/// Q4K subgroup-lane decomposition: `ix = lane / 8` selects one of the 4
/// super-blocks for this pass; `(iq, ir) = ((lane % 8) / 4, lane % 4)` addresses
/// the lane's 8-byte sub-region within that block.
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

/// Compose one Q4K ggml per-column dot from tile primitives. Reads raw block
/// words from `qwords`, decodes the lane's 6-bit scales/mins, accumulates the
/// 4-bit weights against the pre-gathered activations with the
/// mask-multiply-without-shift trick, then folds in the deferred positional
/// scale, the per-sub-block scale `d`, and the affine `dmin` term.
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
    acts: &Q4KGgmlActs,
) -> Tile {
    let base = (col.clone() * blocks_per_col + block.clone()) * block_words;
    // Weights are read unconditionally (constant-true mask) so each lowers to a
    // direct pointer load. Out-of-bounds K is already zeroed via the masked
    // activations, and out-of-bounds columns are discarded by the store mask.
    let load = |program: &mut TileBlock<'_>, offset: Tile| -> Tile {
        program.load(qwords.at(base.clone() + offset), Tile::all(), 0u32)
    };

    // The native (f16 header) and f32-scale layouts shift the scale/data word
    // offsets by one: scales live at `scale0..scale0+3`, data from `data_base`.
    let (scale0, data_base) = if native { (1u32, 4u32) } else { (2u32, 5u32) };

    // Super-block scales `d`/`dmin`: a packed f16 pair (native) or two f32 words.
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

    // 6-bit sub-block scales/mins, interleaved across three words (ggml layout).
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

    // 4-bit weights: word offset is data_base + iq*8 + ir*2, low/high split.
    let data_offset = lane.iq.clone().shift_left(3u32) + lane.ir.clone().shift_left(1u32);
    let mut first_sums: [Tile; 4] = std::array::from_fn(|_| Tile::f32(0.0));
    let mut second_sums: [Tile; 4] = std::array::from_fn(|_| Tile::f32(0.0));
    for j in 0..2u32 {
        let word = load(program, data_offset.clone() + (data_base + j));
        accumulate_q4k_word(&word, &acts.low, (j * 4) as usize, &mut first_sums);
        let word_high = load(program, data_offset.clone() + (data_base + 16 + j));
        accumulate_q4k_word(&word_high, &acts.high, (j * 4) as usize, &mut second_sums);
    }

    // Deferred positional fold: the small-shift nibbles keep weight 1, the
    // large-shift nibbles are scaled by 1/256 (and 1/16 within each pair).
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

/// Accumulate one packed 4-bit weight word against 8 activations using the
/// mask-multiply-without-shift trick: each nibble is isolated by mask only, so
/// its value carries an implicit positional scale (1, 256, 16, 4096) that the
/// caller's deferred fold corrects.
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
