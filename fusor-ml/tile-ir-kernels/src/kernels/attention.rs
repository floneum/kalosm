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
//!
//! Operands are addressed through explicit per-axis element strides baked at
//! build time, so arbitrary strided rank-4 views (transposes, offsets) and
//! grouped-query K/V (`kv_groups > 1`, KV head = head / groups) need no
//! materialization.

use fusor_tile_ir::tile::{Program, Storage, Tile, TileBlock};
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

/// Shape of one fused attention dispatch over rank-4
/// `[batch, heads, seq, head_dim]` operands.
#[derive(Clone, Copy, Debug)]
pub struct FlashAttentionShape {
    /// Batch size.
    pub batch: u32,
    /// Query heads.
    pub heads: u32,
    /// Query heads per KV head (1 = multi-head attention). K/V are addressed
    /// at head `h / kv_groups`.
    pub kv_groups: u32,
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

/// Element strides of one rank-4 `[batch, heads, seq, head_dim]` operand into
/// its linear storage. K/V use their own head count (`heads / kv_groups`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlashOperandLayout {
    /// Element offset of `[0, 0, 0, 0]`.
    pub offset: u32,
    /// Elements between consecutive batches.
    pub batch_stride: u32,
    /// Elements between consecutive heads.
    pub head_stride: u32,
    /// Elements between consecutive sequence positions.
    pub seq_stride: u32,
    /// Elements between consecutive head-dim positions.
    pub dim_stride: u32,
}

impl FlashOperandLayout {
    /// The contiguous `[batch, heads, seq, head_dim]` layout.
    pub fn contiguous(heads: u32, seq: u32, head_dim: u32) -> Self {
        Self {
            offset: 0,
            batch_stride: heads * seq * head_dim,
            head_stride: seq * head_dim,
            seq_stride: head_dim,
            dim_stride: 1,
        }
    }
}

/// Element strides of the additive rank-2 `[q_len, kv_len]` mask, broadcast
/// over every (batch, head).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlashMaskLayout {
    /// Element offset of `[0, 0]`.
    pub offset: u32,
    /// Elements between consecutive query positions.
    pub q_stride: u32,
    /// Elements between consecutive KV positions.
    pub kv_stride: u32,
}

/// Per-operand layouts for one flash-attention dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlashAttentionLayouts {
    pub q: FlashOperandLayout,
    pub k: FlashOperandLayout,
    pub v: FlashOperandLayout,
    pub o: FlashOperandLayout,
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
        && shape.batch > 0
        && shape.heads > 0
        && shape.kv_groups > 0
        && shape.heads % shape.kv_groups == 0
}

/// Decompose an operand's (batch, head) base into dynamic (row, col) origins
/// over a `[seq_stride, 1]`-strided rank-2 view of the same buffer: base
/// components divisible by the sequence stride advance whole rows, the rest
/// lands in the unit-stride column axis. Only unit-`dim_stride` operands
/// qualify (checked by the caller); this keeps `fill_tile`'s contiguous vec4
/// staging path for every practical layout.
fn rank2_origins(
    program: &mut TileBlock,
    layout: FlashOperandLayout,
    b: &Tile,
    h: &Tile,
) -> (Tile, Tile) {
    let ss = layout.seq_stride;
    let mut row = Tile::u32(0);
    let mut col = Tile::u32(0);
    for (coeff, stride) in [(b, layout.batch_stride), (h, layout.head_stride)] {
        if stride % ss == 0 {
            row = row + coeff.clone() * (stride / ss);
        } else {
            col = col + coeff.clone() * stride;
        }
    }
    if layout.offset % ss == 0 {
        row = row + layout.offset / ss;
    } else {
        col = col + layout.offset;
    }
    (program.bind(row), program.bind(col))
}

/// The workgroup grid [`flash_attention_f32`] dispatches over: one workgroup
/// per `BR`-row query tile of one (batch, head).
pub fn flash_attention_dispatch(
    shape: &FlashAttentionShape,
    max_workgroups_per_dimension: u32,
) -> [u32; 3] {
    let total_tiles = shape.batch * shape.heads * (shape.q_len / BR);
    dispatch_grid_1d(total_tiles, max_workgroups_per_dimension)
}

/// Emit the fused attention kernel. Returns `false` (program untouched) when
/// the shape fails [`flash_attention_supported`].
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_f32(
    program: &mut Program,
    q: &Storage,
    k: &Storage,
    v: &Storage,
    mask: Option<(&Storage, FlashMaskLayout)>,
    o: &Storage,
    layouts: &FlashAttentionLayouts,
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
    let scalar = ScalarElement::F32;
    let (lq, lk, lv, lo) = (layouts.q, layouts.k, layouts.v, layouts.o);
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

    let q_stride = d + 1;
    let kt_stride = BC + 1;
    let v_stride = d + 1;
    let s_stride = d.max(BC) + 1;

    // Rank-2 `[seq positions, head dim]` views over the same buffers keep
    // `fill_tile`'s vectorized staging whenever the head dim is unit-stride;
    // other layouts stage through the scalar fallback loops.
    let kv_heads = shape.heads / shape.kv_groups;
    let q_fast = (lq.dim_stride == 1 && lq.seq_stride > 0).then(|| {
        q.restride([shape.batch * shape.heads * shape.q_len, d], [lq.seq_stride, 1])
    });
    let v_fast = (lv.dim_stride == 1 && lv.seq_stride > 0).then(|| {
        v.restride([shape.batch * kv_heads * shape.kv_len, d], [lv.seq_stride, 1])
    });

    let grid = flash_attention_dispatch(&shape, max_workgroups_per_dimension);
    program.program_grid(block, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let bh = tile_id.clone() / q_tiles;
        let qt = program.bind(tile_id % q_tiles);
        let b = program.bind(bh.clone() / shape.heads);
        let h = program.bind(bh % shape.heads);
        let kv_h = program.bind(h.clone() / shape.kv_groups);
        // Element bases of this workgroup's operand slices.
        let k_base = program.bind(
            b.clone() * lk.batch_stride + kv_h.clone() * lk.head_stride + lk.offset,
        );
        let o_base = program.bind(
            b.clone() * lo.batch_stride
                + h.clone() * lo.head_stride
                + qt.clone() * (BR * lo.seq_stride)
                + lo.offset,
        );
        // Position of this workgroup's first query row within the sequence
        // (mask/causal indices are per-sequence, not per-storage-row).
        let q_pos_base = program.bind(qt.clone() * BR);

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
        if let Some(q_view) = &q_fast {
            let (row0, col0) = rank2_origins(program, lq, &b, &h);
            program.fill_tile(&q_tile, q_view, row0 + q_pos_base.clone(), col0);
        } else {
            let q_base = program.bind(
                b.clone() * lq.batch_stride
                    + h.clone() * lq.head_stride
                    + qt.clone() * (BR * lq.seq_stride)
                    + lq.offset,
            );
            let q_elems = BR * d;
            debug_assert_eq!(q_elems % block, 0);
            for i in 0..q_elems / block {
                let flat = program.bind(lane.clone() + i * block);
                let r = program.bind(flat.clone() / d);
                let c = program.bind(flat % d);
                let value = program.load(
                    q.at(q_base.clone() + r.clone() * lq.seq_stride + c.clone() * lq.dim_stride),
                    Tile::all(),
                    0.0,
                );
                program.store_workgroup(&q_tile, r * q_stride + c, value);
            }
        }
        let v_origins = v_fast
            .as_ref()
            .map(|_| rank2_origins(program, lv, &b, &kv_h));
        let v_base = v_fast.is_none().then(|| {
            program.bind(b.clone() * lv.batch_stride + kv_h.clone() * lv.head_stride + lv.offset)
        });

        program.loop_range(kv_tiles, |program, kv_t| {
            let kv_pos_base = program.bind(kv_t * BC);
            if shape.causal {
                // Tiles entirely past the query diagonal contribute nothing.
                program.break_if(kv_pos_base.clone().gt(q_pos_base.clone() + (BR - 1)));
            }

            // Stage Kᵀ ([d][BC], transposed at staging time so QKᵀ is a plain
            // A×B over workgroup tiles) and V ([BC][d]).
            let k_tile_base =
                program.bind(k_base.clone() + kv_pos_base.clone() * lk.seq_stride);
            let kt_elems = d * BC;
            debug_assert_eq!(kt_elems % block, 0);
            for i in 0..kt_elems / block {
                let flat = program.bind(lane.clone() + i * block);
                let kv_local = program.bind(flat.clone() / d);
                let dim = program.bind(flat % d);
                let value = program.load(
                    k.at(k_tile_base.clone()
                        + kv_local.clone() * lk.seq_stride
                        + dim.clone() * lk.dim_stride),
                    Tile::all(),
                    0.0,
                );
                program.store_workgroup(&kt_tile, dim * kt_stride + kv_local, value);
            }
            if let (Some(v_view), Some((v_row0, v_col0))) = (&v_fast, &v_origins) {
                program.fill_tile(
                    &v_tile,
                    v_view,
                    v_row0.clone() + kv_pos_base.clone(),
                    v_col0.clone(),
                );
            } else {
                let v_base = v_base.as_ref().expect("scalar V staging needs a base");
                let v_tile_base =
                    program.bind(v_base.clone() + kv_pos_base.clone() * lv.seq_stride);
                let v_elems = BC * d;
                debug_assert_eq!(v_elems % block, 0);
                for i in 0..v_elems / block {
                    let flat = program.bind(lane.clone() + i * block);
                    let r = program.bind(flat.clone() / d);
                    let c = program.bind(flat % d);
                    let value = program.load(
                        v.at(v_tile_base.clone()
                            + r.clone() * lv.seq_stride
                            + c.clone() * lv.dim_stride),
                        Tile::all(),
                        0.0,
                    );
                    program.store_workgroup(&v_tile, r * v_stride + c, value);
                }
            }
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
                        } else if let Some((mask, lm)) = &mask {
                            raw + program.load(
                                mask.at(q_pos.clone() * lm.q_stride
                                    + (kv_pos_base.clone() + c) * lm.kv_stride
                                    + lm.offset),
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
                o.at(o_base.clone()
                    + o_row.clone() * lo.seq_stride
                    + (o_col_base.clone() + i as u32) * lo.dim_stride),
                program.load_local(reg) * inv_l.clone(),
                Tile::all(),
            );
        }
    });
    true
}
