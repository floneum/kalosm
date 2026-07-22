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
    scalar_of,
    coop_load_a_fragments, coop_load_b_fragments, coop_load_b_fragments_transposed,
    coop_mma_grid, dispatch_grid_1d, zero_coop_acc_grid,
};

/// Query rows per workgroup.
const BR: u32 = 32;
/// KV positions per streamed tile.
const BC: u32 = 16;
/// Cooperative fragment side.
const COOP_DIM: u32 = 8;
/// Element the Q/K/V/dO operand tiles stage in: f16 halves their threadgroup
/// footprint and the MMAs run f16 fragments against f32 accumulators at full
/// rate. Tiles that receive f32 accumulator stores (scores, dP) stay f32.

/// Finite stand-in for -inf: masked scores exp to zero without the
/// `(-inf) - (-inf)` NaN when a row is entirely masked so far.
const MASKED_SCORE: f32 = -3.0e38;

/// The kernel's math runs in f32; convert at the storage boundary when the
/// operand tensors are f16 (per-lane stores don't convert element types).
fn cast_to_storage(dst: &Storage, value: Tile) -> Tile {
    match scalar_of(dst.element()) {
        ScalarElement::F32 => value,
        elem => value.cast(elem.element()),
    }
}

/// Widen a storage-loaded value into the kernel's f32 math.
fn cast_from_storage(src: &Storage, value: Tile) -> Tile {
    match scalar_of(src.element()) {
        ScalarElement::F32 => value,
        _ => value.cast(ElementType::F32),
    }
}

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
///
/// The P·V pass splits the head dim across the four subgroups in 8-wide
/// fragments, so `head_dim` must be a multiple of 32.
pub fn flash_attention_supported(shape: &FlashAttentionShape, subgroups: SubgroupConfig) -> bool {
    let block = subgroups.block_for_subgroups(4);
    subgroups.is_fixed()
        && shape.q_len % BR == 0
        && shape.kv_len % BC == 0
        && shape.head_dim % 32 == 0
        && shape.head_dim >= 32
        && shape.head_dim <= 80
        && block % BR == 0
        && shape.head_dim % (block / BR) == 0
        && shape.batch > 0
        && shape.heads > 0
        && shape.kv_groups > 0
        && shape.heads % shape.kv_groups == 0
}

/// Workgroup-memory footprint of [`flash_attention_f32`] in bytes for one
/// head dim and stage element: the staged Q/KV/P operand tiles in `stage`
/// (K and V share one tile) plus the f32 score tile and the three per-row
/// statistic arrays, each tile row carrying one pad element against bank
/// conflicts. Asserted equal to the lowered IR's `workgroup_bytes` in
/// `tests/footprint.rs`.
pub const fn flash_attention_workgroup_bytes(head_dim: u32, stage: ScalarElement) -> u64 {
    let d = head_dim as u64;
    let (br, bc) = (BR as u64, BC as u64);
    let s_cols = if d > bc { d } else { bc };
    // A padded tile spans `rows * (cols + 1) - 1` elements: the pad after
    // its last row is never addressed and is not allocated.
    let stage_elements = (br * (d + 1) - 1) + (bc * (d + 1) - 1) + (br * (bc + 1) - 1);
    let f32_elements = (br * (s_cols + 1) - 1) + 3 * br;
    stage_elements * stage.byte_size() + f32_elements * ScalarElement::F32.byte_size()
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
    // Operands stage in f16; `s_tile` receives the f32 QKᵀ accumulators (cols
    // 0..BC) and, after a barrier, the f32 P·V partial (cols 0..d) —
    // probabilities live in the small f16 `p_tile` so they can feed the P·V
    // MMA as f16 A-fragments.
    // Stage operands in their own element type: casting f32 operands to
    // f16 tiles injects ~2e-4 noise per attention op, which measurably
    // degrades training and compounds to NaN within a few hundred steps.
    let stage = scalar_of(q.element());
    let q_tile = program.alloc_workgroup_tile_padded(stage, BR, d, 1);
    // K and V share one tile: K is dead once the score MMA's post-store
    // barrier passes, and V is not read until the barrier before the P*V
    // MMA, so V stages into K's slot between the two existing barriers.
    // The sharing costs nothing and holds the f16 kernel's footprint at
    // 16.0 KB - the two-workgroups-per-core residency boundary.
    let kv_tile = program.alloc_workgroup_tile_padded(stage, BC, d, 1);
    let s_tile = program.alloc_workgroup_tile_padded(scalar, BR, d.max(BC), 1);
    let p_tile = program.alloc_workgroup_tile_padded(stage, BR, BC, 1);
    let m_arr = program.alloc_workgroup_array(scalar, BR);
    let l_arr = program.alloc_workgroup_array(scalar, BR);
    let alpha_arr = program.alloc_workgroup_array(scalar, BR);

    let s_stride = d.max(BC) + 1;
    let p_stride = BC + 1;

    let kv_heads = shape.heads / shape.kv_groups;
    let q_fast = fast_rows_view(q, lq, shape.batch * shape.heads * shape.q_len, d);
    let k_fast = fast_rows_view(k, lk, shape.batch * kv_heads * shape.kv_len, d);
    let v_fast = fast_rows_view(v, lv, shape.batch * kv_heads * shape.kv_len, d);

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
        stage_rows(
            program, q, &q_fast, lq, &q_tile, &b, &h, &q_pos_base, BR, d, &lane, block,
        );

        program.loop_range(kv_tiles, |program, kv_t| {
            let kv_pos_base = program.bind(kv_t * BC);
            if shape.causal {
                // Tiles entirely past the query diagonal contribute nothing.
                program.break_if(kv_pos_base.clone().gt(q_pos_base.clone() + (BR - 1)));
            }

            // Stage K row-major; QKᵀ reads it through transposed fragment
            // loads. V stages into the same tile after the score phase.
            stage_rows(
                program, k, &k_fast, lk, &kv_tile, &b, &kv_h, &kv_pos_base, BC, d, &lane, block,
            );
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
                    stage,
                );
                let b_frags = coop_load_b_fragments_transposed(
                    program,
                    coop,
                    &kv_tile,
                    &sg_col_base,
                    kk,
                    s_cols,
                    stage,
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

            // Every subgroup's score MMA reads of K completed before the
            // barrier above, so V now stages into the shared tile while the
            // BR softmax lanes work the staged scores; the barrier below
            // gates the P·V reads of V.
            stage_rows(
                program, v, &v_fast, lv, &kv_tile, &b, &kv_h, &kv_pos_base, BC, d, &lane, block,
            );

            // Online softmax over the staged scores: one lane per query row.
            // Probabilities land in the f16 `p_tile` — the A operand of the
            // P·V MMA below.
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
                            raw + cast_from_storage(
                                mask,
                                program.load(
                                    mask.at(q_pos.clone() * lm.q_stride
                                        + (kv_pos_base.clone() + c) * lm.kv_stride
                                        + lm.offset),
                                    Tile::all(),
                                    0.0,
                                ),
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
                        &p_tile,
                        row.clone() * p_stride + c as u32,
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
            // below, so overwriting the score region is safe.
            let pv_rows = BR / COOP_DIM;
            let pv_cols = d / 4 / COOP_DIM;
            let sg_d_base = program.bind(subgroups.token().subgroup_id(program) * (d / 4));
            let pv_accs = zero_coop_acc_grid(program, coop, scalar, pv_rows, pv_cols);
            let zero_row = Tile::u32(0);
            for kk in 0..BC / COOP_DIM {
                let a_frags = coop_load_a_fragments(
                    program,
                    coop,
                    &p_tile,
                    &zero_row,
                    kk,
                    pv_rows,
                    stage,
                );
                let b_frags = coop_load_b_fragments(
                    program,
                    coop,
                    &kv_tile,
                    &sg_d_base,
                    kk,
                    pv_cols,
                    stage,
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
            // The trailing barrier gates this iteration's `s_tile`/`alpha_arr`
            // reads against the next iteration's shared-memory writes. An
            // attempted elision here (reasoning the post-staging barrier
            // covers it) produced run-to-run nondeterminism and eventual NaNs
            // at training step ~300 — the next iteration touches shared
            // state before its first collective barrier.
            program.workgroup_barrier();
        });

        // Normalize by the softmax denominator and store.
        let inv_l = program.bind(Tile::f32(1.0) / program.load_workgroup(&l_arr, o_row.clone()));
        for (i, reg) in o_regs.iter().enumerate() {
            program.store(
                o.at(o_base.clone()
                    + o_row.clone() * lo.seq_stride
                    + (o_col_base.clone() + i as u32) * lo.dim_stride),
                cast_to_storage(o, program.load_local(reg) * inv_l.clone()),
                Tile::all(),
            );
        }
    });
    true
}

// ---- backward -------------------------------------------------------------

/// Element strides of one rank-3 `[batch, heads, seq]` row statistic (the
/// forward's log-sum-exp, the backward's `rowsum(dO ∘ O)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlashRowLayout {
    /// Element offset of `[0, 0, 0]`.
    pub offset: u32,
    /// Elements between consecutive batches.
    pub batch_stride: u32,
    /// Elements between consecutive heads.
    pub head_stride: u32,
    /// Elements between consecutive sequence positions.
    pub seq_stride: u32,
}

/// Per-operand layouts for the backward kernels. `out` is dq for
/// [`flash_bwd_q_f32`]; for [`flash_bwd_kv_f32`] it is the combined dk/dv
/// tensor whose sequence axis spans `2 * kv_len` (dk rows first, dv rows at
/// `kv_len + position`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FlashBwdLayouts {
    pub q: FlashOperandLayout,
    pub k: FlashOperandLayout,
    pub v: FlashOperandLayout,
    pub grad_o: FlashOperandLayout,
    pub lse: FlashRowLayout,
    pub dsum: FlashRowLayout,
    pub out: FlashOperandLayout,
}

/// Whether the backward family ([`flash_lse_f32`], [`flash_bwd_q_f32`],
/// [`flash_bwd_kv_f32`]) can host this shape. Grouped-query K/V is not yet
/// implemented on the kv pass, so the whole family requires `kv_groups == 1`.
pub fn flash_attention_bwd_supported(
    shape: &FlashAttentionShape,
    subgroups: SubgroupConfig,
) -> bool {
    flash_attention_supported(shape, subgroups)
        && shape.q_len % BR == 0
        && shape.kv_len % BR == 0
        && shape.kv_groups == 1
}

/// One workgroup per `BR`-row query tile of one (batch, head).
pub fn flash_lse_dispatch(
    shape: &FlashAttentionShape,
    max_workgroups_per_dimension: u32,
) -> [u32; 3] {
    dispatch_grid_1d(
        shape.batch * shape.heads * (shape.q_len / BR),
        max_workgroups_per_dimension,
    )
}

/// One workgroup per `BR`-row query tile of one (batch, head).
pub fn flash_bwd_q_dispatch(
    shape: &FlashAttentionShape,
    max_workgroups_per_dimension: u32,
) -> [u32; 3] {
    dispatch_grid_1d(
        shape.batch * shape.heads * (shape.q_len / BR),
        max_workgroups_per_dimension,
    )
}

/// One workgroup per `BR`-row KV tile of one (batch, head).
pub fn flash_bwd_kv_dispatch(
    shape: &FlashAttentionShape,
    max_workgroups_per_dimension: u32,
) -> [u32; 3] {
    dispatch_grid_1d(
        shape.batch * shape.heads * (shape.kv_len / BR),
        max_workgroups_per_dimension,
    )
}

/// The vectorized-staging rank-2 view of an operand, when its head dim is
/// unit-stride.
fn fast_rows_view(
    storage: &Storage,
    layout: FlashOperandLayout,
    total_rows: u32,
    d: u32,
) -> Option<Storage> {
    (layout.dim_stride == 1 && layout.seq_stride > 0)
        .then(|| storage.restride([total_rows, d], [layout.seq_stride, 1]))
}

/// Stage `rows` consecutive sequence rows into a `[rows][d + 1]` workgroup
/// tile: `fill_tile`'s vectorized path via the rank-2 view when available,
/// otherwise a scalar strided loop.
#[allow(clippy::too_many_arguments)]
fn stage_rows(
    program: &mut TileBlock,
    src: &Storage,
    fast: &Option<Storage>,
    layout: FlashOperandLayout,
    tile: &fusor_tile_ir::tile::WorkgroupTile,
    b: &Tile,
    h: &Tile,
    seq_base: &Tile,
    rows: u32,
    d: u32,
    lane: &Tile,
    block: u32,
) {
    if let Some(view) = fast {
        let (row0, col0) = rank2_origins(program, layout, b, h);
        program.fill_tile(tile, view, row0 + seq_base.clone(), col0);
        return;
    }
    let base = program.bind(
        b.clone() * layout.batch_stride
            + h.clone() * layout.head_stride
            + seq_base.clone() * layout.seq_stride
            + layout.offset,
    );
    let elems = rows * d;
    debug_assert_eq!(elems % block, 0);
    let stride = d + 1;
    for i in 0..elems / block {
        let flat = program.bind(lane.clone() + i * block);
        let r = program.bind(flat.clone() / d);
        let c = program.bind(flat % d);
        let value = program.load(
            src.at(base.clone() + r.clone() * layout.seq_stride + c.clone() * layout.dim_stride),
            Tile::all(),
            0.0,
        );
        program.store_workgroup(tile, r * stride + c, value);
    }
}

/// Load `rows` per-row statistics into a workgroup array (one lane each).
#[allow(clippy::too_many_arguments)]
fn load_row_stats(
    program: &mut TileBlock,
    src: &Storage,
    layout: FlashRowLayout,
    arr: &fusor_tile_ir::tile::WorkgroupTile,
    b: &Tile,
    h: &Tile,
    seq_base: &Tile,
    rows: u32,
    lane: &Tile,
) {
    let base = program.bind(
        b.clone() * layout.batch_stride
            + h.clone() * layout.head_stride
            + seq_base.clone() * layout.seq_stride
            + layout.offset,
    );
    let seq_stride = layout.seq_stride;
    program.if_then(lane.clone().lt(rows), |program| {
        let value = program.load(
            src.at(base.clone() + lane.clone() * seq_stride),
            Tile::all(),
            0.0,
        );
        program.store_workgroup(arr, lane.clone(), value);
    });
}

/// Store a `[rows][d]` f32 accumulator grid — each of the four subgroups
/// owning the `d/4`-column slice at `subgroup_id * d/4` — to `out` at
/// `out_base + r * seq_stride + col * dim_stride`, staged through the given
/// dead f32 `[rows][BC + 1]` tiles. One subgroup stages per tile per round
/// (guarded coop stores are subgroup-uniform), then every lane copies out.
/// Requires `d / 4 <= BC` so a subgroup slice fits a tile; the last loop
/// iteration's trailing barrier covers the first round's tile reuse.
#[allow(clippy::too_many_arguments)]
fn store_acc_grid_chunked(
    program: &mut TileBlock,
    coop: CoopMatrixToken,
    subgroups: SubgroupConfig,
    accs: &[Vec<fusor_tile_ir::tile::CoopAcc>],
    tiles: &[&fusor_tile_ir::tile::WorkgroupTile],
    out: &Storage,
    layout: FlashOperandLayout,
    out_base: &Tile,
    rows: u32,
    d: u32,
    lane: &Tile,
    block: u32,
) {
    let chunk_cols = d / 4;
    debug_assert!(chunk_cols <= BC && (rows * chunk_cols).is_multiple_of(block));
    let stride = BC + 1;
    let sgid = subgroups.token().subgroup_id(program);
    let mut sg = 0u32;
    while sg < 4 {
        if sg > 0 {
            // The previous round's copies finish before the tiles are reused.
            program.workgroup_barrier();
        }
        let in_round = (tiles.len() as u32).min(4 - sg);
        for i in 0..in_round {
            let owner = sg + i;
            program.if_then(sgid.clone().eq(owner), |program| {
                for (r, row_accs) in accs.iter().enumerate() {
                    for (c, acc) in row_accs.iter().enumerate() {
                        coop.coop_store_tile(
                            program,
                            acc,
                            tiles[i as usize],
                            Tile::u32(r as u32 * COOP_DIM),
                            Tile::u32(c as u32 * COOP_DIM),
                        );
                    }
                }
            });
        }
        program.workgroup_barrier();
        for i in 0..in_round {
            let col_base = (sg + i) * chunk_cols;
            for pass in 0..rows * chunk_cols / block {
                let flat = program.bind(lane.clone() + pass * block);
                let r = program.bind(flat.clone() / chunk_cols);
                let c = program.bind(flat % chunk_cols);
                let value =
                    program.load_workgroup(tiles[i as usize], r.clone() * stride + c.clone());
                program.store(
                    out.at(out_base.clone()
                        + r * layout.seq_stride
                        + (c + col_base) * layout.dim_stride),
                    cast_to_storage(out, value),
                    Tile::all(),
                );
            }
        }
        sg += in_round;
    }
}

/// Store a `[rows][d]` f32 accumulator grid to `out` staged through one dead
/// f32 `[rows][d + 1]` tile — the wide-head fallback when a subgroup's
/// column slice exceeds the chunk tiles.
#[allow(clippy::too_many_arguments)]
fn store_acc_grid_whole(
    program: &mut TileBlock,
    coop: CoopMatrixToken,
    subgroups: SubgroupConfig,
    accs: &[Vec<fusor_tile_ir::tile::CoopAcc>],
    tile: &fusor_tile_ir::tile::WorkgroupTile,
    out: &Storage,
    layout: FlashOperandLayout,
    out_base: &Tile,
    rows: u32,
    d: u32,
    lane: &Tile,
    block: u32,
) {
    let sg_d_base = program.bind(subgroups.token().subgroup_id(program) * (d / 4));
    for (r, row_accs) in accs.iter().enumerate() {
        for (c, acc) in row_accs.iter().enumerate() {
            coop.coop_store_tile(
                program,
                acc,
                tile,
                Tile::u32(r as u32 * COOP_DIM),
                sg_d_base.clone() + c as u32 * COOP_DIM,
            );
        }
    }
    program.workgroup_barrier();
    let lanes_per_row = block / rows;
    let cols_per_lane = d / lanes_per_row;
    let o_row = program.bind(lane.clone() / lanes_per_row);
    let o_col_base = program.bind((lane.clone() % lanes_per_row) * cols_per_lane);
    let stride = d + 1;
    for i in 0..cols_per_lane {
        let value =
            program.load_workgroup(tile, o_row.clone() * stride + o_col_base.clone() + i);
        program.store(
            out.at(out_base.clone()
                + o_row.clone() * layout.seq_stride
                + (o_col_base.clone() + i) * layout.dim_stride),
            cast_to_storage(out, value),
            Tile::all(),
        );
    }
}

/// Emit the row log-sum-exp kernel: `lse[row] = m + ln Σ exp(scale·q·kᵀ
/// [+ mask] − m)` — the forward statistic that reconstructs probabilities
/// per tile. Returns `false` when the shape fails
/// [`flash_attention_bwd_supported`].
#[allow(clippy::too_many_arguments)]
pub fn flash_lse_f32(
    program: &mut Program,
    q: &Storage,
    k: &Storage,
    mask: Option<(&Storage, FlashMaskLayout)>,
    lse_out: &Storage,
    q_layout: FlashOperandLayout,
    k_layout: FlashOperandLayout,
    lse_layout: FlashRowLayout,
    shape: FlashAttentionShape,
    subgroups: SubgroupConfig,
    coop: CoopMatrixToken,
    max_workgroups_per_dimension: u32,
) -> bool {
    if !flash_attention_bwd_supported(&shape, subgroups) || (shape.causal && mask.is_some()) {
        return false;
    }
    let block = subgroups.block_for_subgroups(4);
    let d = shape.head_dim;
    let q_tiles = shape.q_len / BR;
    let kv_tiles = shape.kv_len / BC;
    let scalar = ScalarElement::F32;

    // Stage operands in their own element type: casting f32 operands to
    // f16 tiles injects ~2e-4 noise per attention op, which measurably
    // degrades training and compounds to NaN within a few hundred steps.
    let stage = scalar_of(q.element());
    let q_tile = program.alloc_workgroup_tile_padded(stage, BR, d, 1);
    // Stage operands in their own element type: casting f32 operands to
    // f16 tiles injects ~2e-4 noise per attention op, which measurably
    // degrades training and compounds to NaN within a few hundred steps.
    let stage = scalar_of(q.element());
    let k_tile = program.alloc_workgroup_tile_padded(stage, BC, d, 1);
    let s_tile = program.alloc_workgroup_tile_padded(scalar, BR, BC, 1);
    let m_arr = program.alloc_workgroup_array(scalar, BR);
    let l_arr = program.alloc_workgroup_array(scalar, BR);
    let s_stride = BC + 1;

    let q_fast = fast_rows_view(q, q_layout, shape.batch * shape.heads * shape.q_len, d);
    let k_fast = fast_rows_view(k, k_layout, shape.batch * shape.heads * shape.kv_len, d);

    let grid = flash_lse_dispatch(&shape, max_workgroups_per_dimension);
    program.program_grid(block, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let bh = tile_id.clone() / q_tiles;
        let qt = program.bind(tile_id % q_tiles);
        let b = program.bind(bh.clone() / shape.heads);
        let h = program.bind(bh % shape.heads);
        let kv_h = program.bind(h.clone() / shape.kv_groups);
        let q_pos_base = program.bind(qt * BR);
        let lane = program.lane();

        program.if_then(lane.clone().lt(BR), |program| {
            program.store_workgroup(&m_arr, lane.clone(), Tile::f32(MASKED_SCORE));
            program.store_workgroup(&l_arr, lane.clone(), Tile::f32(0.0));
        });
        stage_rows(
            program, q, &q_fast, q_layout, &q_tile, &b, &h, &q_pos_base, BR, d, &lane, block,
        );

        program.loop_range(kv_tiles, |program, kv_t| {
            let kv_pos_base = program.bind(kv_t * BC);
            if shape.causal {
                program.break_if(kv_pos_base.clone().gt(q_pos_base.clone() + (BR - 1)));
            }
            stage_rows(
                program, k, &k_fast, k_layout, &k_tile, &b, &kv_h, &kv_pos_base, BC, d, &lane,
                block,
            );
            program.workgroup_barrier();

            let subgroup_id = subgroups.token().subgroup_id(program);
            let sg_row = program.bind(subgroup_id.clone() / 2);
            let sg_col = program.bind(subgroup_id % 2);
            let s_rows = BR / 2 / COOP_DIM;
            let s_cols = BC / 2 / COOP_DIM;
            let sg_row_base = program.bind(sg_row * (BR / 2));
            let sg_col_base = program.bind(sg_col * (BC / 2));
            let s_accs = zero_coop_acc_grid(program, coop, scalar, s_rows, s_cols);
            for kk in 0..d / COOP_DIM {
                let a_frags =
                    coop_load_a_fragments(program, coop, &q_tile, &sg_row_base, kk, s_rows, stage);
                let b_frags = coop_load_b_fragments_transposed(
                    program, coop, &k_tile, &sg_col_base, kk, s_cols, stage,
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

            program.if_then(lane.clone().lt(BR), |program| {
                let row = lane.clone();
                let q_pos = program.bind(q_pos_base.clone() + row.clone());
                let m_old = program.bind(program.load_workgroup(&m_arr, row.clone()));
                let vals: Vec<Tile> = (0..BC)
                    .map(|c| {
                        let raw = program.load_workgroup(&s_tile, row.clone() * s_stride + c)
                            * shape.scale;
                        let masked = if shape.causal {
                            let allowed = (kv_pos_base.clone() + c).le(q_pos.clone());
                            Tile::select(allowed, raw, Tile::f32(MASKED_SCORE))
                        } else if let Some((mask, lm)) = &mask {
                            raw + cast_from_storage(
                                mask,
                                program.load(
                                    mask.at(q_pos.clone() * lm.q_stride
                                        + (kv_pos_base.clone() + c) * lm.kv_stride
                                        + lm.offset),
                                    Tile::all(),
                                    0.0,
                                ),
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
                for val in &vals {
                    row_sum = row_sum + (val.clone() - m_new.clone()).exp();
                }
                let l_old = program.load_workgroup(&l_arr, row.clone());
                program.store_workgroup(&l_arr, row.clone(), l_old * alpha + row_sum);
                program.store_workgroup(&m_arr, row, m_new);
            });
            program.workgroup_barrier();
        });

        let lse_base = program.bind(
            b * lse_layout.batch_stride
                + h * lse_layout.head_stride
                + q_pos_base * lse_layout.seq_stride
                + lse_layout.offset,
        );
        program.if_then(lane.clone().lt(BR), |program| {
            let m = program.load_workgroup(&m_arr, lane.clone());
            let l = program.load_workgroup(&l_arr, lane.clone());
            program.store(
                lse_out.at(lse_base.clone() + lane.clone() * lse_layout.seq_stride),
                cast_to_storage(lse_out, m + l.log()),
                Tile::all(),
            );
        });
    });
    true
}

/// Emit the dQ kernel: per `BR`-row query tile, stream KV tiles
/// reconstructing `P = exp(scale·q·kᵀ [+ mask] − lse)`, form
/// `dS = P ∘ (dO·vᵀ − dsum) · scale`, and accumulate `dq = Σ dS·k` in
/// cooperative registers. Returns `false` when the shape fails
/// [`flash_attention_bwd_supported`].
#[allow(clippy::too_many_arguments)]
pub fn flash_bwd_q_f32(
    program: &mut Program,
    q: &Storage,
    k: &Storage,
    v: &Storage,
    grad_o: &Storage,
    lse: &Storage,
    dsum: &Storage,
    mask: Option<(&Storage, FlashMaskLayout)>,
    dq_out: &Storage,
    layouts: &FlashBwdLayouts,
    shape: FlashAttentionShape,
    subgroups: SubgroupConfig,
    coop: CoopMatrixToken,
    max_workgroups_per_dimension: u32,
) -> bool {
    if !flash_attention_bwd_supported(&shape, subgroups) || (shape.causal && mask.is_some()) {
        return false;
    }
    let block = subgroups.block_for_subgroups(4);
    let d = shape.head_dim;
    let q_tiles = shape.q_len / BR;
    let kv_tiles = shape.kv_len / BC;
    let scalar = ScalarElement::F32;
    let (lq, lk, lv, ldo) = (layouts.q, layouts.k, layouts.v, layouts.grad_o);

    // Stage operands in their own element type: casting f32 operands to
    // f16 tiles injects ~2e-4 noise per attention op, which measurably
    // degrades training and compounds to NaN within a few hundred steps.
    let stage = scalar_of(q.element());
    let q_tile = program.alloc_workgroup_tile_padded(stage, BR, d, 1);
    let do_tile = program.alloc_workgroup_tile_padded(stage, BR, d, 1);
    // Stage operands in their own element type: casting f32 operands to
    // f16 tiles injects ~2e-4 noise per attention op, which measurably
    // degrades training and compounds to NaN within a few hundred steps.
    let stage = scalar_of(q.element());
    let k_tile = program.alloc_workgroup_tile_padded(stage, BC, d, 1);
    let v_tile = program.alloc_workgroup_tile_padded(stage, BC, d, 1);
    let s_tile = program.alloc_workgroup_tile_padded(scalar, BR, BC, 1);
    let dp_tile = program.alloc_workgroup_tile_padded(scalar, BR, BC, 1);
    // dS feeds the dq MMA as f16 A-fragments; the f32 score/dP tiles stay the
    // accumulator staging (and later the chunked output staging).
    let ds_tile = program.alloc_workgroup_tile_padded(stage, BR, BC, 1);
    let lse_arr = program.alloc_workgroup_array(scalar, BR);
    let d_arr = program.alloc_workgroup_array(scalar, BR);
    let s_stride = BC + 1;
    // Output staging: dq columns leave through the dead f32 score/dP tiles
    // when a subgroup's `d/4` slice fits; wider heads stage through one
    // dedicated f32 tile.
    let out_whole =
        (d / 4 > BC).then(|| program.alloc_workgroup_tile_padded(scalar, BR, d, 1));

    let q_fast = fast_rows_view(q, lq, shape.batch * shape.heads * shape.q_len, d);
    let do_fast = fast_rows_view(grad_o, ldo, shape.batch * shape.heads * shape.q_len, d);
    let k_fast = fast_rows_view(k, lk, shape.batch * shape.heads * shape.kv_len, d);
    let v_fast = fast_rows_view(v, lv, shape.batch * shape.heads * shape.kv_len, d);

    let grid = flash_bwd_q_dispatch(&shape, max_workgroups_per_dimension);
    program.program_grid(block, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let bh = tile_id.clone() / q_tiles;
        let qt = program.bind(tile_id % q_tiles);
        let b = program.bind(bh.clone() / shape.heads);
        let h = program.bind(bh % shape.heads);
        let q_pos_base = program.bind(qt * BR);
        let lane = program.lane();

        stage_rows(
            program, q, &q_fast, lq, &q_tile, &b, &h, &q_pos_base, BR, d, &lane, block,
        );
        stage_rows(
            program, grad_o, &do_fast, ldo, &do_tile, &b, &h, &q_pos_base, BR, d, &lane, block,
        );
        load_row_stats(
            program, lse, layouts.lse, &lse_arr, &b, &h, &q_pos_base, BR, &lane,
        );
        load_row_stats(
            program, dsum, layouts.dsum, &d_arr, &b, &h, &q_pos_base, BR, &lane,
        );

        // Each subgroup owns `d / 4` output columns across all rows.
        let sg_d_base = program.bind(subgroups.token().subgroup_id(program) * (d / 4));
        let dq_rows = BR / COOP_DIM;
        let dq_cols = d / 4 / COOP_DIM;
        let dq_accs = zero_coop_acc_grid(program, coop, scalar, dq_rows, dq_cols);
        let zero_row = Tile::u32(0);

        program.loop_range(kv_tiles, |program, kv_t| {
            let kv_pos_base = program.bind(kv_t * BC);
            if shape.causal {
                program.break_if(kv_pos_base.clone().gt(q_pos_base.clone() + (BR - 1)));
            }
            stage_rows(
                program, k, &k_fast, lk, &k_tile, &b, &h, &kv_pos_base, BC, d, &lane, block,
            );
            stage_rows(
                program, v, &v_fast, lv, &v_tile, &b, &h, &kv_pos_base, BC, d, &lane, block,
            );
            program.workgroup_barrier();

            // S = q·kᵀ and dP = dO·vᵀ on one 2×2 subgroup grid, K/V read
            // through transposed fragment loads.
            let subgroup_id = subgroups.token().subgroup_id(program);
            let sg_row = program.bind(subgroup_id.clone() / 2);
            let sg_col = program.bind(subgroup_id % 2);
            let sg_row_base = program.bind(sg_row * (BR / 2));
            let sg_col_base = program.bind(sg_col * (BC / 2));
            let s_rows = BR / 2 / COOP_DIM;
            let s_cols = BC / 2 / COOP_DIM;
            let s_accs = zero_coop_acc_grid(program, coop, scalar, s_rows, s_cols);
            let dp_accs = zero_coop_acc_grid(program, coop, scalar, s_rows, s_cols);
            for kk in 0..d / COOP_DIM {
                let a_frags =
                    coop_load_a_fragments(program, coop, &q_tile, &sg_row_base, kk, s_rows, stage);
                let b_frags = coop_load_b_fragments_transposed(
                    program, coop, &k_tile, &sg_col_base, kk, s_cols, stage,
                );
                coop_mma_grid(program, coop, &s_accs, &a_frags, &b_frags);
                let da_frags =
                    coop_load_a_fragments(program, coop, &do_tile, &sg_row_base, kk, s_rows, stage);
                let db_frags = coop_load_b_fragments_transposed(
                    program, coop, &v_tile, &sg_col_base, kk, s_cols, stage,
                );
                coop_mma_grid(program, coop, &dp_accs, &da_frags, &db_frags);
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
            for (r, row_accs) in dp_accs.iter().enumerate() {
                for (c, acc) in row_accs.iter().enumerate() {
                    coop.coop_store_tile(
                        program,
                        acc,
                        &dp_tile,
                        sg_row_base.clone() + r as u32 * COOP_DIM,
                        sg_col_base.clone() + c as u32 * COOP_DIM,
                    );
                }
            }
            program.workgroup_barrier();

            // dS = P ∘ (dP − dsum) · scale, written to the f16 dS tile (the
            // A operand of the dq MMA).
            program.if_then(lane.clone().lt(BR), |program| {
                let row = lane.clone();
                let q_pos = program.bind(q_pos_base.clone() + row.clone());
                let lse_row = program.bind(program.load_workgroup(&lse_arr, row.clone()));
                let d_row = program.bind(program.load_workgroup(&d_arr, row.clone()));
                for c in 0..BC {
                    let raw = program.load_workgroup(&s_tile, row.clone() * s_stride + c)
                        * shape.scale;
                    let masked = if shape.causal {
                        let allowed = (kv_pos_base.clone() + c).le(q_pos.clone());
                        Tile::select(allowed, raw, Tile::f32(MASKED_SCORE))
                    } else if let Some((mask, lm)) = &mask {
                        raw + cast_from_storage(
                            mask,
                            program.load(
                                mask.at(q_pos.clone() * lm.q_stride
                                    + (kv_pos_base.clone() + c) * lm.kv_stride
                                    + lm.offset),
                                Tile::all(),
                                0.0,
                            ),
                        )
                    } else {
                        raw
                    };
                    let p = program.bind((masked - lse_row.clone()).exp());
                    let dp = program.load_workgroup(&dp_tile, row.clone() * s_stride + c);
                    let ds = p * (dp - d_row.clone()) * shape.scale;
                    program.store_workgroup(&ds_tile, row.clone() * s_stride + c, ds);
                }
            });
            program.workgroup_barrier();

            // dq += dS·k.
            for kk in 0..BC / COOP_DIM {
                let a_frags =
                    coop_load_a_fragments(program, coop, &ds_tile, &zero_row, kk, dq_rows, stage);
                let b_frags =
                    coop_load_b_fragments(program, coop, &k_tile, &sg_d_base, kk, dq_cols, stage);
                coop_mma_grid(program, coop, &dq_accs, &a_frags, &b_frags);
            }
            program.workgroup_barrier();
        });

        // Stage the accumulators through dead f32 tiles and store.
        let lo = layouts.out;
        let out_base = program.bind(
            b * lo.batch_stride
                + h * lo.head_stride
                + q_pos_base * lo.seq_stride
                + lo.offset,
        );
        if let Some(out_whole) = &out_whole {
            store_acc_grid_whole(
                program, coop, subgroups, &dq_accs, out_whole, dq_out, lo, &out_base, BR, d,
                &lane, block,
            );
        } else {
            store_acc_grid_chunked(
                program,
                coop,
                subgroups,
                &dq_accs,
                &[&s_tile, &dp_tile],
                dq_out,
                lo,
                &out_base,
                BR,
                d,
                &lane,
                block,
            );
        }
    });
    true
}

/// Which gradients [`flash_bwd_kv_f32`] emits. The independent modes exist so
/// each recognized contraction can lower alone; a later horizontal merge can
/// combine two single-output dispatches back into `Both`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FlashKvOutputs {
    /// dk and dv into one tensor whose sequence axis spans `2·kv_len`
    /// (dk rows first, dv rows at `kv_len + position`).
    Both,
    /// dk only, into a `[batch, heads, kv_len, head_dim]` tensor.
    Dk,
    /// dv only, into a `[batch, heads, kv_len, head_dim]` tensor.
    Dv,
}

/// Emit the dK/dV kernel: per `BR`-row KV tile, stream query tiles
/// (descending when causal so the break skips the empty upper triangle),
/// reconstruct `Pᵀ` from the forward statistics, and accumulate
/// `dv = Σ Pᵀ·dO` and/or `dk = Σ dSᵀ·q` in cooperative registers per
/// `outputs`. Returns `false` when the shape fails
/// [`flash_attention_bwd_supported`].
#[allow(clippy::too_many_arguments)]
pub fn flash_bwd_kv_f32(
    program: &mut Program,
    q: &Storage,
    k: &Storage,
    v: Option<&Storage>,
    grad_o: &Storage,
    lse: &Storage,
    dsum: Option<&Storage>,
    mask: Option<(&Storage, FlashMaskLayout)>,
    dkv_out: &Storage,
    layouts: &FlashBwdLayouts,
    outputs: FlashKvOutputs,
    shape: FlashAttentionShape,
    subgroups: SubgroupConfig,
    coop: CoopMatrixToken,
    max_workgroups_per_dimension: u32,
) -> bool {
    if !flash_attention_bwd_supported(&shape, subgroups) || (shape.causal && mask.is_some()) {
        return false;
    }
    let emit_dk = outputs != FlashKvOutputs::Dv;
    let emit_dv = outputs != FlashKvOutputs::Dk;
    let block = subgroups.block_for_subgroups(4);
    let d = shape.head_dim;
    let kv_tiles = shape.kv_len / BR;
    let q_tiles = shape.q_len / BC;
    let scalar = ScalarElement::F32;
    let (lq, lk, lv, ldo) = (layouts.q, layouts.k, layouts.v, layouts.grad_o);

    // Stage operands in their own element type: casting f32 operands to
    // f16 tiles injects ~2e-4 noise per attention op, which measurably
    // degrades training and compounds to NaN within a few hundred steps.
    let stage = scalar_of(q.element());
    let k_tile = program.alloc_workgroup_tile_padded(stage, BR, d, 1);
    // dPᵀ = v·dOᵀ is only needed for dk.
    let v_tile = emit_dk.then(|| program.alloc_workgroup_tile_padded(stage, BR, d, 1));
    // Stage operands in their own element type: casting f32 operands to
    // f16 tiles injects ~2e-4 noise per attention op, which measurably
    // degrades training and compounds to NaN within a few hundred steps.
    let stage = scalar_of(q.element());
    let q_tile = program.alloc_workgroup_tile_padded(stage, BC, d, 1);
    let do_tile = program.alloc_workgroup_tile_padded(stage, BC, d, 1);
    let st_tile = program.alloc_workgroup_tile_padded(scalar, BR, BC, 1);
    let dpt_tile = emit_dk.then(|| program.alloc_workgroup_tile_padded(scalar, BR, BC, 1));
    // Pᵀ and dSᵀ feed the dv/dk MMAs as f16 A-fragments; the f32 tiles above
    // keep the accumulator staging (and later the chunked output staging).
    let pt_tile = emit_dv.then(|| program.alloc_workgroup_tile_padded(stage, BR, BC, 1));
    let dst_tile = emit_dk.then(|| program.alloc_workgroup_tile_padded(stage, BR, BC, 1));
    let lse_arr = program.alloc_workgroup_array(scalar, BC);
    let d_arr = emit_dk.then(|| program.alloc_workgroup_array(scalar, BC));
    let st_stride = BC + 1;
    // Wide-head output staging fallback, as in the dq kernel.
    let out_whole =
        (d / 4 > BC).then(|| program.alloc_workgroup_tile_padded(scalar, BR, d, 1));

    let q_fast = fast_rows_view(q, lq, shape.batch * shape.heads * shape.q_len, d);
    let do_fast = fast_rows_view(grad_o, ldo, shape.batch * shape.heads * shape.q_len, d);
    let k_fast = fast_rows_view(k, lk, shape.batch * shape.heads * shape.kv_len, d);
    let v_fast = v.and_then(|v| fast_rows_view(v, lv, shape.batch * shape.heads * shape.kv_len, d));

    let grid = flash_bwd_kv_dispatch(&shape, max_workgroups_per_dimension);
    program.program_grid(block, grid, |program| {
        let tile_id = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * grid[0]
            + program.program_id(WorkgroupAxis::Z) * grid[0] * grid[1];
        let bh = tile_id.clone() / kv_tiles;
        let kvt = program.bind(tile_id % kv_tiles);
        let b = program.bind(bh.clone() / shape.heads);
        let h = program.bind(bh % shape.heads);
        let kv_pos_base = program.bind(kvt * BR);
        let lane = program.lane();

        stage_rows(
            program, k, &k_fast, lk, &k_tile, &b, &h, &kv_pos_base, BR, d, &lane, block,
        );
        if let Some(v_tile) = &v_tile {
            let v = v.expect("dk emission requires the values operand");
            stage_rows(
                program, v, &v_fast, lv, v_tile, &b, &h, &kv_pos_base, BR, d, &lane, block,
            );
        }

        let sg_d_base = program.bind(subgroups.token().subgroup_id(program) * (d / 4));
        let acc_rows = BR / COOP_DIM;
        let acc_cols = d / 4 / COOP_DIM;
        let dk_accs =
            emit_dk.then(|| zero_coop_acc_grid(program, coop, scalar, acc_rows, acc_cols));
        let dv_accs =
            emit_dv.then(|| zero_coop_acc_grid(program, coop, scalar, acc_rows, acc_cols));
        let zero_row = Tile::u32(0);

        program.loop_range(q_tiles, |program, i| {
            // Causal iterates query tiles descending: once a tile is fully
            // above the diagonal every remaining one is too.
            let q_t = if shape.causal {
                program.bind(Tile::u32(q_tiles - 1) - i)
            } else {
                program.bind(i)
            };
            let q_pos_base = program.bind(q_t * BC);
            if shape.causal {
                program.break_if((q_pos_base.clone() + (BC - 1)).lt(kv_pos_base.clone()));
            }
            stage_rows(
                program, q, &q_fast, lq, &q_tile, &b, &h, &q_pos_base, BC, d, &lane, block,
            );
            stage_rows(
                program, grad_o, &do_fast, ldo, &do_tile, &b, &h, &q_pos_base, BC, d, &lane,
                block,
            );
            load_row_stats(
                program, lse, layouts.lse, &lse_arr, &b, &h, &q_pos_base, BC, &lane,
            );
            if let Some(d_arr) = &d_arr {
                let dsum = dsum.expect("dk emission requires the dsum operand");
                load_row_stats(
                    program, dsum, layouts.dsum, d_arr, &b, &h, &q_pos_base, BC, &lane,
                );
            }
            program.workgroup_barrier();

            // S̃ = k·qᵀ (and dPᵀ = v·dOᵀ when dk is emitted) on one 2×2
            // subgroup grid; Q/dO read through transposed fragment loads.
            let subgroup_id = subgroups.token().subgroup_id(program);
            let sg_row = program.bind(subgroup_id.clone() / 2);
            let sg_col = program.bind(subgroup_id % 2);
            let sg_row_base = program.bind(sg_row * (BR / 2));
            let sg_col_base = program.bind(sg_col * (BC / 2));
            let s_rows = BR / 2 / COOP_DIM;
            let s_cols = BC / 2 / COOP_DIM;
            let st_accs = zero_coop_acc_grid(program, coop, scalar, s_rows, s_cols);
            let dpt_accs =
                emit_dk.then(|| zero_coop_acc_grid(program, coop, scalar, s_rows, s_cols));
            for kk in 0..d / COOP_DIM {
                let a_frags =
                    coop_load_a_fragments(program, coop, &k_tile, &sg_row_base, kk, s_rows, stage);
                let b_frags = coop_load_b_fragments_transposed(
                    program, coop, &q_tile, &sg_col_base, kk, s_cols, stage,
                );
                coop_mma_grid(program, coop, &st_accs, &a_frags, &b_frags);
                if let (Some(dpt_accs), Some(v_tile)) = (&dpt_accs, &v_tile) {
                    let da_frags = coop_load_a_fragments(
                        program, coop, v_tile, &sg_row_base, kk, s_rows, stage,
                    );
                    let db_frags = coop_load_b_fragments_transposed(
                        program, coop, &do_tile, &sg_col_base, kk, s_cols, stage,
                    );
                    coop_mma_grid(program, coop, dpt_accs, &da_frags, &db_frags);
                }
            }
            for (r, row_accs) in st_accs.iter().enumerate() {
                for (c, acc) in row_accs.iter().enumerate() {
                    coop.coop_store_tile(
                        program,
                        acc,
                        &st_tile,
                        sg_row_base.clone() + r as u32 * COOP_DIM,
                        sg_col_base.clone() + c as u32 * COOP_DIM,
                    );
                }
            }
            if let (Some(dpt_accs), Some(dpt_tile)) = (&dpt_accs, &dpt_tile) {
                for (r, row_accs) in dpt_accs.iter().enumerate() {
                    for (c, acc) in row_accs.iter().enumerate() {
                        coop.coop_store_tile(
                            program,
                            acc,
                            dpt_tile,
                            sg_row_base.clone() + r as u32 * COOP_DIM,
                            sg_col_base.clone() + c as u32 * COOP_DIM,
                        );
                    }
                }
            }
            program.workgroup_barrier();

            // Pᵀ over the score tile (and dSᵀ over the dPᵀ tile when dk is
            // emitted). Row = KV position, column = query position,
            // statistics indexed by query.
            program.if_then(lane.clone().lt(BR), |program| {
                let kv_row = lane.clone();
                let kv_pos = program.bind(kv_pos_base.clone() + kv_row.clone());
                for c in 0..BC {
                    let q_pos = program.bind(q_pos_base.clone() + c);
                    let raw = program
                        .load_workgroup(&st_tile, kv_row.clone() * st_stride + c)
                        * shape.scale;
                    let masked = if shape.causal {
                        let allowed = kv_pos.clone().le(q_pos.clone());
                        Tile::select(allowed, raw, Tile::f32(MASKED_SCORE))
                    } else if let Some((mask, lm)) = &mask {
                        raw + cast_from_storage(
                            mask,
                            program.load(
                                mask.at(q_pos.clone() * lm.q_stride
                                    + kv_pos.clone() * lm.kv_stride
                                    + lm.offset),
                                Tile::all(),
                                0.0,
                            ),
                        )
                    } else {
                        raw
                    };
                    let lse_col = program.load_workgroup(&lse_arr, Tile::u32(c));
                    let p = program.bind((masked - lse_col).exp());
                    if let (Some(d_arr), Some(dpt_tile), Some(dst_tile)) =
                        (&d_arr, &dpt_tile, &dst_tile)
                    {
                        let d_col = program.load_workgroup(d_arr, Tile::u32(c));
                        let dp =
                            program.load_workgroup(dpt_tile, kv_row.clone() * st_stride + c);
                        let ds = p.clone() * (dp - d_col) * shape.scale;
                        program.store_workgroup(dst_tile, kv_row.clone() * st_stride + c, ds);
                    }
                    if let Some(pt_tile) = &pt_tile {
                        program.store_workgroup(pt_tile, kv_row.clone() * st_stride + c, p);
                    }
                }
            });
            program.workgroup_barrier();

            // dv += Pᵀ·dO and dk += dSᵀ·q.
            for kk in 0..BC / COOP_DIM {
                if let (Some(dv_accs), Some(pt_tile)) = (&dv_accs, &pt_tile) {
                    let a_frags = coop_load_a_fragments(
                        program, coop, pt_tile, &zero_row, kk, acc_rows, stage,
                    );
                    let b_frags = coop_load_b_fragments(
                        program, coop, &do_tile, &sg_d_base, kk, acc_cols, stage,
                    );
                    coop_mma_grid(program, coop, dv_accs, &a_frags, &b_frags);
                }
                if let (Some(dk_accs), Some(dst_tile)) = (&dk_accs, &dst_tile) {
                    let ka_frags = coop_load_a_fragments(
                        program, coop, dst_tile, &zero_row, kk, acc_rows, stage,
                    );
                    let kb_frags = coop_load_b_fragments(
                        program, coop, &q_tile, &sg_d_base, kk, acc_cols, stage,
                    );
                    coop_mma_grid(program, coop, dk_accs, &ka_frags, &kb_frags);
                }
            }
            program.workgroup_barrier();
        });

        // Stage each accumulator grid through dead f32 tiles and store. In
        // `Both` mode dk lands at sequence `kv_pos` and dv at
        // `kv_len + kv_pos`; single-output modes write at `kv_pos`.
        let dv_seq_offset = if emit_dk { shape.kv_len } else { 0 };
        let lo = layouts.out;
        let mut stores: Vec<(u32, &Vec<Vec<fusor_tile_ir::tile::CoopAcc>>)> = Vec::new();
        if let Some(dk_accs) = &dk_accs {
            stores.push((0, dk_accs));
        }
        if let Some(dv_accs) = &dv_accs {
            stores.push((dv_seq_offset, dv_accs));
        }
        let mut chunk_tiles: Vec<&fusor_tile_ir::tile::WorkgroupTile> = vec![&st_tile];
        if let Some(dpt_tile) = &dpt_tile {
            chunk_tiles.push(dpt_tile);
        }
        for (i, (seq_offset, accs)) in stores.iter().enumerate() {
            if i > 0 {
                // The first output's copies finish before its staging tiles
                // are reused.
                program.workgroup_barrier();
            }
            let out_base = program.bind(
                b.clone() * lo.batch_stride
                    + h.clone() * lo.head_stride
                    + (kv_pos_base.clone() + *seq_offset) * lo.seq_stride
                    + lo.offset,
            );
            if let Some(out_whole) = &out_whole {
                store_acc_grid_whole(
                    program, coop, subgroups, accs, out_whole, dkv_out, lo, &out_base, BR, d,
                    &lane, block,
                );
            } else {
                store_acc_grid_chunked(
                    program,
                    coop,
                    subgroups,
                    accs,
                    &chunk_tiles,
                    dkv_out,
                    lo,
                    &out_base,
                    BR,
                    d,
                    &lane,
                    block,
                );
            }
        }
    });
    true
}
