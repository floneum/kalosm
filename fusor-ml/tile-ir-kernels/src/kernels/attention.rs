//! Fused scaled-dot-product attention over cooperative matrices.
//!
//! One workgroup owns a `BR`-row query tile of one (batch, head) and streams
//! the KV axis in `BC`-wide tiles: QKᵀ and P·V run on simdgroup matrices
//! staged through workgroup memory, the online-softmax row statistics run on
//! per-row lanes over the staged score tile, and the output accumulates in
//! per-lane registers rescaled by the running max. Unlike the attention row
//! program (one query row per workgroup — decode's shape), K/V tiles are
//! shared across all `BR` rows in the workgroup, so prefill/training shapes
//! keep full data reuse.

use fusor_tile_ir::tile::{Program, Storage, Tile};
use fusor_tile_ir::{CoopMatrixToken, ElementType, ScalarElement, WorkgroupAxis};

use crate::dispatch::SubgroupConfig;
use crate::kernels::helpers::{
    coop_load_a_fragments, coop_load_b_fragments, coop_mma_grid, dispatch_grid_1d,
    zero_coop_acc_grid,
};

/// Query rows per workgroup.
const BR: u32 = 32;
/// KV positions per streamed tile.
const BC: u32 = 16;
/// Cooperative fragment side.
const COOP_DIM: u32 = 8;
/// Finite stand-in for -inf: masked scores exp to zero without the
/// `(-inf) - (-inf)` NaN when a row is entirely masked so far.
const MASKED_SCORE: f32 = -3.0e38;

/// Shape of one fused attention dispatch. Sequences are indexed per
/// (batch·head) row block: `q`/`o` are `[bh * q_len, head_dim]`, `k`/`v` are
/// `[bh * kv_len, head_dim]`, and the optional additive mask is
/// `[q_len, kv_len]`.
#[derive(Clone, Copy, Debug)]
pub struct FlashAttentionShape {
    /// Batch × query heads.
    pub bh: u32,
    /// Query sequence length.
    pub q_len: u32,
    /// KV sequence length.
    pub kv_len: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Score scale (typically `head_dim^-0.5`).
    pub scale: f32,
    /// Causal masking via index comparison (mutually exclusive with `mask`).
    pub causal: bool,
}

/// Whether [`flash_attention_f32`] can host this shape.
pub fn flash_attention_supported(shape: &FlashAttentionShape, subgroups: SubgroupConfig) -> bool {
    let block = subgroups.block_for_subgroups(4);
    subgroups.is_fixed()
        && shape.q_len % BR == 0
        && shape.kv_len % BC == 0
        && shape.head_dim % 16 == 0
        && shape.head_dim >= 16
        && shape.head_dim <= 80
        && block % BR == 0
        && shape.head_dim % (block / BR) == 0
        && shape.bh > 0
}

/// Emit the fused attention kernel. Returns `false` (program untouched) when
/// the shape fails [`flash_attention_supported`] or the tile grid exceeds the
/// dispatch limit.
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_f32(
    program: &mut Program,
    q: &Storage,
    k: &Storage,
    v: &Storage,
    mask: Option<&Storage>,
    o: &Storage,
    shape: FlashAttentionShape,
    subgroups: SubgroupConfig,
    coop: CoopMatrixToken,
    max_workgroups_per_dimension: u32,
) -> bool {
    if !flash_attention_supported(&shape, subgroups) || (shape.causal && mask.is_some()) {
        return false;
    }
    let block = subgroups.block_for_subgroups(4);
    let d = shape.head_dim;
    let q_tiles = shape.q_len / BR;
    let kv_tiles = shape.kv_len / BC;
    let total_tiles = shape.bh * q_tiles;
    if total_tiles > max_workgroups_per_dimension {
        return false;
    }
    let scalar = ScalarElement::F32;
    // Per-lane output slice: `block / BR` lanes share a row, each owning
    // `d / (block / BR)` contiguous columns in registers.
    let lanes_per_row = block / BR;
    let cols_per_lane = d / lanes_per_row;

    // Staged tiles (+1 pad on the inner stride against bank conflicts).
    // `s_tile` holds the BR×BC score/probability tile, then is overwritten
    // with the BR×d P·V partial after a barrier — probabilities are dead once
    // every subgroup's MMAs have read them.
    let q_tile = program.alloc_workgroup_tile_padded(scalar, BR, d, 1);
    let kt_tile = program.alloc_workgroup_tile_padded(scalar, d, BC, 1);
    let v_tile = program.alloc_workgroup_tile_padded(scalar, BC, d, 1);
    let s_tile = program.alloc_workgroup_tile_padded(scalar, BR, d.max(BC), 1);
    let m_arr = program.alloc_workgroup_array(scalar, BR);
    let l_arr = program.alloc_workgroup_array(scalar, BR);
    let alpha_arr = program.alloc_workgroup_array(scalar, BR);

    let kt_stride = BC + 1;
    let s_stride = d.max(BC) + 1;

    let grid = dispatch_grid_1d(total_tiles, max_workgroups_per_dimension);
    program.program_grid(block, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let bh = tile_id.clone() / q_tiles;
        let qt = tile_id % q_tiles;
        // Row bases into the [bh * len, d] storages.
        let q_row_base = bh.clone() * shape.q_len + qt.clone() * BR;
        let kv_row_base = bh * shape.kv_len;
        // Position of this workgroup's first query row within the sequence
        // (mask/causal indices are per-sequence, not per-storage-row).
        let q_pos_base = program.bind(qt * BR);

        let lane = program.lane();
        // Zero the running statistics and the output registers.
        program.if_then(lane.clone().lt(BR), |program| {
            program.store_workgroup(&m_arr, lane.clone(), Tile::f32(MASKED_SCORE));
            program.store_workgroup(&l_arr, lane.clone(), Tile::f32(0.0));
        });
        let o_regs: Vec<_> = (0..cols_per_lane)
            .map(|_| {
                let reg = program.private(ElementType::F32);
                program.store_local(&reg, Tile::f32(0.0));
                reg
            })
            .collect();
        let o_row = program.bind(lane.clone() / lanes_per_row);
        let o_col_base = program.bind((lane.clone() % lanes_per_row) * cols_per_lane);

        // Q tile is loop-invariant: staged once.
        program.fill_tile(&q_tile, q, q_row_base.clone(), 0u32);

        program.loop_range(kv_tiles, |program, kv_t| {
            let kv_pos_base = program.bind(kv_t * BC);
            if shape.causal {
                // Tiles entirely past the query diagonal contribute nothing.
                program.break_if(kv_pos_base.clone().gt(q_pos_base.clone() + (BR - 1)));
            }

            // Stage Kᵀ ([d][BC], transposed at staging time so QKᵀ is a plain
            // A×B over workgroup tiles) and V ([BC][d]).
            let kt_elems = d * BC;
            debug_assert_eq!(kt_elems % block, 0);
            for i in 0..kt_elems / block {
                let flat = program.bind(lane.clone() + i * block);
                let kv_local = flat.clone() / d;
                let dim = flat % d;
                let value = program.load(
                    k.at((
                        kv_row_base.clone() + kv_pos_base.clone() + kv_local.clone(),
                        dim.clone(),
                    )),
                    Tile::all(),
                    0.0,
                );
                program.store_workgroup(&kt_tile, dim * kt_stride + kv_local, value);
            }
            program.fill_tile(&v_tile, v, kv_row_base.clone() + kv_pos_base.clone(), 0u32);
            program.workgroup_barrier();

            // S = Q · Kᵀ on fragments: 2×2 subgroup grid, each owning
            // BR/2 rows × BC/2 columns of the score tile.
            let subgroup_id = subgroups.token().subgroup_id(program);
            let sg_row = program.bind(subgroup_id.clone() / 2);
            let sg_col = program.bind(subgroup_id % 2);
            let s_rows = BR / 2 / COOP_DIM;
            let s_cols = BC / 2 / COOP_DIM;
            let sg_row_base = program.bind(sg_row * (BR / 2));
            let sg_col_base = program.bind(sg_col.clone() * (BC / 2));
            let s_accs = zero_coop_acc_grid(program, coop, scalar, s_rows, s_cols);
            for kk in 0..d / COOP_DIM {
                let a_frags = coop_load_a_fragments(
                    program,
                    coop,
                    &q_tile,
                    &sg_row_base,
                    kk,
                    s_rows,
                    scalar,
                );
                let b_frags = coop_load_b_fragments(
                    program,
                    coop,
                    &kt_tile,
                    &sg_col_base,
                    kk,
                    s_cols,
                    scalar,
                );
                coop_mma_grid(program, coop, &s_accs, &a_frags, &b_frags);
            }
            for (r, row_accs) in s_accs.iter().enumerate() {
                for (c, acc) in row_accs.iter().enumerate() {
                    coop.coop_store_tile(
                        program,
                        acc,
                        &s_tile,
                        sg_row_base.clone() + r as u32 * COOP_DIM,
                        sg_col_base.clone() + c as u32 * COOP_DIM,
                    );
                }
            }
            program.workgroup_barrier();

            // Online softmax over the staged scores: one lane per query row.
            program.if_then(lane.clone().lt(BR), |program| {
                let row = lane.clone();
                let q_pos = program.bind(q_pos_base.clone() + row.clone());
                let m_old = program.bind(program.load_workgroup(&m_arr, row.clone()));
                let vals: Vec<Tile> = (0..BC)
                    .map(|c| {
                        let raw = program.load_workgroup(&s_tile, row.clone() * s_stride + c)
                            * shape.scale;
                        let masked = if shape.causal {
                            let allowed =
                                (kv_pos_base.clone() + c).le(q_pos.clone());
                            Tile::select(allowed, raw, Tile::f32(MASKED_SCORE))
                        } else if let Some(mask) = mask {
                            raw + program.load(
                                mask.at((q_pos.clone(), kv_pos_base.clone() + c)),
                                Tile::all(),
                                0.0,
                            )
                        } else {
                            raw
                        };
                        program.bind(masked)
                    })
                    .collect();
                let mut m_tile = vals[0].clone();
                for val in &vals[1..] {
                    m_tile = m_tile.max(val.clone());
                }
                let m_new = program.bind(m_old.clone().max(m_tile));
                let alpha = program.bind((m_old - m_new.clone()).exp());
                let mut row_sum = Tile::f32(0.0);
                for (c, val) in vals.iter().enumerate() {
                    let p = program.bind((val.clone() - m_new.clone()).exp());
                    program.store_workgroup(
                        &s_tile,
                        row.clone() * s_stride + c as u32,
                        p.clone(),
                    );
                    row_sum = row_sum + p;
                }
                let l_old = program.load_workgroup(&l_arr, row.clone());
                program.store_workgroup(&l_arr, row.clone(), l_old * alpha.clone() + row_sum);
                program.store_workgroup(&m_arr, row.clone(), m_new);
                program.store_workgroup(&alpha_arr, row, alpha);
            });
            program.workgroup_barrier();

            // P·V on fragments: each subgroup owns d/4 output columns across
            // all BR rows. Accumulators stay in registers past the barrier
            // below, so overwriting the probabilities is safe.
            let pv_rows = BR / COOP_DIM;
            let pv_cols = d / 4 / COOP_DIM;
            let sg_d_base = program.bind(subgroups.token().subgroup_id(program) * (d / 4));
            let pv_accs = zero_coop_acc_grid(program, coop, scalar, pv_rows, pv_cols);
            let zero_row = Tile::u32(0);
            for kk in 0..BC / COOP_DIM {
                let a_frags = coop_load_a_fragments(
                    program,
                    coop,
                    &s_tile,
                    &zero_row,
                    kk,
                    pv_rows,
                    scalar,
                );
                let b_frags = coop_load_b_fragments(
                    program,
                    coop,
                    &v_tile,
                    &sg_d_base,
                    kk,
                    pv_cols,
                    scalar,
                );
                coop_mma_grid(program, coop, &pv_accs, &a_frags, &b_frags);
            }
            // Every subgroup finishes reading P before any overwrites it.
            program.workgroup_barrier();
            for (r, row_accs) in pv_accs.iter().enumerate() {
                for (c, acc) in row_accs.iter().enumerate() {
                    coop.coop_store_tile(
                        program,
                        acc,
                        &s_tile,
                        Tile::u32(r as u32 * COOP_DIM),
                        sg_d_base.clone() + c as u32 * COOP_DIM,
                    );
                }
            }
            program.workgroup_barrier();

            // Fold the P·V partial into the output registers, rescaling the
            // previous accumulation by this tile's alpha.
            let alpha = program.bind(program.load_workgroup(&alpha_arr, o_row.clone()));
            for (i, reg) in o_regs.iter().enumerate() {
                let partial = program.load_workgroup(
                    &s_tile,
                    o_row.clone() * s_stride + o_col_base.clone() + i as u32,
                );
                let folded = program.load_local(reg) * alpha.clone() + partial;
                program.store_local(reg, folded);
            }
            // The next iteration's staging writes k/v tiles only; the score
            // store is gated by the post-staging barrier above.
            program.workgroup_barrier();
        });

        // Normalize by the softmax denominator and store.
        let inv_l = program.bind(Tile::f32(1.0) / program.load_workgroup(&l_arr, o_row.clone()));
        for (i, reg) in o_regs.iter().enumerate() {
            program.store(
                o.at((
                    q_row_base.clone() + o_row.clone(),
                    o_col_base.clone() + i as u32,
                )),
                program.load_local(reg) * inv_l.clone(),
                Tile::all(),
            );
        }
    });
    true
}
