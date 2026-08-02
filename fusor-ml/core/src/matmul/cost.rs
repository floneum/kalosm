//! General cooperative-tile selection.
//!
//! One scalar additive cost over the joint choice of (tile entry) x (split
//! count) x (staged tile pairs), in integer femtoseconds with physical device
//! rates. A lexicographic
//! key cannot trade split-K scratch against tile geometry — the levels have
//! incommensurate units — so the whole decision surface is a single argmin
//! ([`score_fs`]), with a tie-break chain that is reached and therefore
//! load-bearing.
//!
//! Alignment enters only through what it does to the cost (edge tiles execute
//! padded MACs); the device enters through legality, four measured rates and
//! one occupancy target. Padding is the dominant measured effect — a
//! 33%-padded tile ran ~2x slower than its zero-pad neighbor at N=384, and a
//! 6.7%-padded 128x512 ran ~3x slower than the least-padded choice on the
//! vision shape — and the MAC term is proportional to padded MACs, so it
//! carries that without a rule.
//!
//! Single-buffered profiles are excluded from automatic selection outright:
//! they cannot horizontally merge (the merged body shares one double-buffered
//! tile pair across guarded segments), and their best standalone shape
//! (4096^3) measures 5.57 TF/s against the selected 128x64's 7.64 — even at
//! parity it would not pay for decomposing a merged wave into standalone
//! dispatches (~15% on the vision QKV family). They stay in the kernel table
//! for forced experiments, where `coop_tile_conformance` keeps them verified.
//!
//! Workgroup-memory footprint enters twice. As legality: entries whose
//! `CoopTileEntry::workgroup_bytes` for the contraction's stage element
//! exceed the device's workgroup-storage limit are unselectable (WebGPU's
//! 16 KB default would otherwise fail at pipeline creation). And as
//! residency: shared memory is carved from a per-core pool, so the footprint
//! divides into how many workgroups a core holds at once, and co-resident
//! workgroups cover each other's epilogue drain. That second role is what
//! makes the staging depth a decision rather than a table column — one
//! staged pair loses the load/MMA overlap on every K iteration and buys back
//! a once-per-workgroup drain, so deep-K contractions want two pairs and
//! shallow ones want one.
//!
//! Measured anchors on M2 Max, per-entry minima over the round-robin reps of
//! `ITERS=8 REPS=6 cargo run --release -p fusor-core --example
//! bench_coop_tiles`. f32, `128x64 / 64x64 / 128x128 / 64x16 / 16x64` in ms:
//! 16384x384x384 0.799 / 0.822 / 1.011 / 0.990 / 1.197; 16384x384x1536
//! 2.705 / 2.893 / 3.456 / 3.823 / 3.996; 16384x1536x384 2.698 / 2.885 /
//! 3.646 / 3.836 / 4.023; 16384x3072x1536 20.41 / 21.16 / 28.30 / 30.58 /
//! 32.44; 1024^3 0.445 / 0.445 / 0.577 / 0.446 / 0.634. The K-deep half of
//! the calibration set: every one of them wants the 128-wide profile, which
//! is what stops the epilogue term from taking 64x64 everywhere. f16 128x64
//! vs 64x64 is inside 0.3% on the first three and 3-4% the other way on
//! 16384x3072x1536 (19.21 / 18.46) and 4096^3 (17.22 / 16.65), so the two
//! f16 rows the golden table moves onto 64x64 are neutral-to-positive.
//! Earlier anchors from the same harness, still the padding evidence:
//! 4096^3 — 256x256 5.57 TF/s (excluded as unmergeable) / 128x512 5.20;
//! 1944x1280x3840 — 64x64 4.19 (least padded) vs 128x64 2.70 ~= 64x128
//! 2.61, padded 128x512 1.47; 1000x1024x1024 — 64x64 3.11 / 128x64 2.62,
//! padded 128x512 1.09.
//!
//! Warm-resolve span_ms against the previous calibration of these same terms
//! (both arms one binary behind one env switch, 20 interleaved processes per
//! arm, order flipped every round): 64x2048x64 0.228 -> 0.200, 64x2048x256
//! 0.617 -> 0.593, 256x2048x64 0.690 -> 0.594, 2048x64x64 0.261 -> 0.220,
//! 2048x256x64 0.738 -> 0.578, 2048x64x256 0.876 -> 0.683, batched 64x64x16
//! 0.290 -> 0.291, 384x16384x1536 x4 14.46 -> 14.22; softmax control flat at
//! 1.090 -> 1.086. Adding the staging depth then moved the two shapes whose
//! chosen depth changed: 2048x64x64 0.217 -> 0.187 and 2048x64x256 0.671 ->
//! 0.597, every other plan bit-identical and the control at 0.0%. The depth
//! itself was calibrated by forcing it: the same merged body staged from one
//! pair instead of two runs -14.0% / -11.0% on those two, +0.8% on
//! 2048x256x64 (16 K iterations) and +7.2% on 384x16384x1536 (1024), which
//! is the crossover the two constants encode. Every one of those moves is the same decision: six of the
//! eight shapes had been taking a 128-wide or 16-wide profile where the
//! 64x64 one measures fastest, because a flat per-element epilogue rate
//! cannot see that a wider workgroup drains its accumulator more slowly, and
//! a linear occupancy law over-splits a starved grid.

use std::cmp::Reverse;

use fusor_tile_ir::ScalarElement;
use fusor_tile_ir_kernels::{CoopTileEntry, coop_tile_entries};

use super::variants::CoopTile;
use crate::occupancy::DispatchPolicy;

/// Cooperative-matrix fragment side; every per-subgroup fragment grid counts
/// whole 8x8 fragments.
const COOP_DIM: u64 = 8;

/// Storage bindings one matmul segment declares: A, B and its output. The
/// horizontal merger budgets a merged wave with the same count, which is what
/// bounds the group [`tile_probe_group`] scores against.
const MATMUL_SEGMENT_BINDINGS: usize = 3;

/// The contraction plus how many co-located segments the dispatch that will
/// run it carries. `segments` is 1 for a standalone kernel and
/// `segments.len()` inside `build_merged_matmul_kernel` — the launched grid is
/// `tiles * splits * segments`, never `tiles * splits`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct CoopDispatch {
    pub(crate) m: u32,
    pub(crate) k: u32,
    pub(crate) n: u32,
    pub(crate) batch: u32,
    pub(crate) segments: u32,
    pub(crate) datatype: crate::DataTypeEnum,
    pub(crate) has_epilogues: bool,
}

/// The probe group the TILE is scored against, everywhere and always.
///
/// Allocation runs before the wave is partitioned, and `hardware_matmul_prep`
/// compares the output's strides for exact equality against the build tile's
/// padding — a tile that moved between `inputs()` and the build does not waste
/// memory, it silently falls back to the generic path. So the tile is scored
/// at a fixed group both can compute, never at the real one: 1 for a
/// contraction that can never merge (epilogues disqualify it at
/// `merge_profile`), otherwise the merger's own maximum group size.
pub(crate) fn tile_probe_group(device: &crate::Device, has_epilogues: bool) -> u32 {
    if has_epilogues {
        1
    } else {
        (device.nary_direct_input_binding_budget() / MATMUL_SEGMENT_BINDINGS).max(1) as u32
    }
}

/// The element the kernels stage operand tiles in. Production paths leave
/// `staging` off, so this is the storage element; it sets both the staged
/// byte count and the workgroup-memory footprint the residency term reads.
fn stage_element(datatype: crate::DataTypeEnum) -> ScalarElement {
    match datatype {
        crate::DataTypeEnum::F16 => ScalarElement::F16,
        _ => ScalarElement::F32,
    }
}

/// Cost of running one contraction with one (tile, subgroup split, split
/// count) choice, in integer femtoseconds.
///
/// Roofline in its literal form: the three issue-side terms sum (MMA issue,
/// threadgroup traffic and the accumulator store all contend for the same
/// per-core issue and load/store slots), DRAM overlaps them (`max`), and the
/// combine dispatch adds behind its barrier. Occupancy scales the issue side
/// only, by the cube root of the residency shortfall against
/// `prefetched_saturation_lanes` — that target is the only role a
/// parallelism floor plays here, never an execution width and never a
/// MAC-equivalent.
///
/// Integer, not f64: candidates tie exactly (the tile table is built from
/// powers of two and the score is invariant to `n_passes` by construction), so
/// the tie-break chain is load-bearing and must be reached deterministically.
#[allow(clippy::too_many_arguments)]
fn score_fs(
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    segments: u32,
    entry: &CoopTileEntry,
    rg: u32,
    cg: u32,
    splits: u32,
    buffers: u32,
    stage: ScalarElement,
    policy: &DispatchPolicy,
    subgroup_width: u32,
) -> u128 {
    let rates = policy.matmul_rates();
    let elem_bytes = stage.byte_size();
    let (bm, bn, bk) = (
        u64::from(entry.tile.bm),
        u64::from(entry.tile.bn),
        u64::from(entry.tile.bk),
    );
    let n_passes = u64::from(entry.n_passes);
    let bn_pass = bn / n_passes;
    // Per subgroup per kk-step: `tr` A-fragment loads and `tc` B-fragment
    // loads feed `tr * tc` MMAs (`coop_load_a_fragments` emits `rows`,
    // `coop_load_b_fragments` emits `cols`, `coop_mma_grid` does rows*cols).
    let tr = bm / (COOP_DIM * u64::from(rg));
    let tc = bn_pass / (COOP_DIM * u64::from(cg));
    let subgroups = u64::from(rg) * u64::from(cg);
    let threads = subgroups * u64::from(subgroup_width);

    let tiles_m = u64::from(m.div_ceil(entry.tile.bm));
    let tiles_n = u64::from(n.div_ceil(entry.tile.bn));
    let tiles_per_segment = tiles_m * tiles_n * u64::from(batch);
    let k_iterations = u64::from(k.div_ceil(entry.tile.bk));
    let span_iterations = k_iterations.div_ceil(u64::from(splits));
    let workgroups = tiles_per_segment * u64::from(splits) * u64::from(segments);
    let m_padded = tiles_m * bm;
    let n_padded = tiles_n * bn;
    let per_workgroup = u128::from(workgroups) * u128::from(span_iterations);

    // T1 MMA issue. Per kk-step a workgroup issues `subgroups * tr * tc * 512`
    // MACs = `bm * bn_pass * 8`; over `bk/8` kk-steps and `n_passes` passes
    // that is `bm * bn * bk` per workgroup per K iteration.
    let t_mma = per_workgroup * u128::from(bm * bn * bk) * 1_000_000
        / u128::from(rates.mac_per_ns);

    // T2 threadgroup traffic: cooperative fragment loads plus operand staging,
    // at one rate. Their per-load byte ratio varies only 96..128 across the
    // whole table, so a fit that separates them is not identifiable.
    let fragment_bytes = n_passes * subgroups * (tr + tc) * (bk / COOP_DIM) * 64 * elem_bytes;
    let stage_bytes = n_passes * (bm * bk + bk * bn_pass) * elem_bytes;
    // One staged pair loses the load/MMA overlap the rates were fitted on.
    let overlap_pct = if buffers == 1 {
        rates.single_buffered_traffic_pct
    } else {
        100
    };
    let t_threadgroup = per_workgroup * u128::from(fragment_bytes + stage_bytes) * 1_000_000
        * u128::from(overlap_pct)
        / (u128::from(rates.workgroup_bytes_per_ns) * 100);

    // T3 accumulator zeroing, the cooperative store's fragment shuffles and
    // the store itself, over the padded output every workgroup emits. This is
    // what makes split-K expensive: every split writes a full padded tile.
    //
    // Per element AND per subgroup of the emitting workgroup. The epilogue is
    // a whole-workgroup drain — every subgroup's accumulator fragments shuffle
    // through the one staged tile pair behind the workgroup's own barrier, so
    // a wider workgroup serializes more of its output through the same
    // threadgroup port and the workgroup cannot retire until the last
    // subgroup lands. Measured on 2048x64x256, where every profile does the
    // same MACs and emits the same padded output, the implied per-element
    // epilogue rate is 32 ps (64x16) and 36 ps (64x64) against 60 ps
    // (128x64), 63 ps (128x128) and 72 ps (64x128): the split is exactly the
    // 4-vs-8 subgroup count. Nothing else in the table sorts it — 64x64 and
    // 64x128 have the same workgroup-memory footprint and land on opposite
    // sides, 128x64 and 64x128 have the same tile area and land together.
    //
    // This is a per-workgroup cost, so it amortizes away down a deep K loop,
    // which is why 384x16384x1536 (1024 K-iterations) still takes the widest
    // profile while the shallow-K shapes do not.
    //
    // It is also what a second co-resident workgroup covers. A core's
    // shared memory holds `core_workgroup_slots` workgroups of this
    // footprint; while one drains its accumulators another issues MMAs, so
    // the drain the dispatch actually waits on is the term divided by that
    // count. Measured by halving the footprint at fixed tile, splits and
    // grid: 2048x64x64 -14.0%, 2048x64x256 -11.0%, against 2048x256x64
    // +0.8% and 384x16384x1536 +7.2% where the deeper K loop makes the
    // staging penalty above outweigh it. A fourth root, not a reciprocal:
    // the drain is only partly hideable, and the raw ratio predicts a 55%
    // swing where the measured pair is 14%.
    let slots = policy.core_workgroup_slots(entry.workgroup_bytes_at(stage, buffers));
    let t_store = u128::from(workgroups)
        * u128::from(bm * bn)
        * u128::from(rates.store_fs_per_element * subgroups)
        * 1_000
        / integer_root(u128::from(slots) * 1_000_000_000_000, 4);

    // T4 DRAM: operands once per segment plus every split's padded output.
    let dram_bytes = u128::from(segments)
        * u128::from(elem_bytes)
        * (u128::from(batch) * (u128::from(m) * u128::from(k) + u128::from(k) * u128::from(n))
            + u128::from(splits) * u128::from(batch) * u128::from(m_padded * n_padded));
    let t_dram = dram_bytes * 10_000_000 / u128::from(rates.dram_decibytes_per_ns);

    // T5 combine: reads every partial slice and writes the output, in its own
    // barrier-separated dispatch, so it adds rather than overlaps.
    let t_combine = if splits > 1 {
        u128::from(segments)
            * u128::from(splits + 1)
            * u128::from(batch)
            * u128::from(m_padded * n_padded)
            * u128::from(elem_bytes)
            * 10_000_000
            / u128::from(rates.dram_decibytes_per_ns)
    } else {
        0
    };

    let issue = t_mma + t_threadgroup + t_store;
    let resident = u128::from(workgroups) * u128::from(threads);
    let target = u128::from(policy.prefetched_saturation_lanes());
    // A grid short of the floor does not lose issue rate in proportion to the
    // lanes it is missing: the lanes it does have keep more of the core's
    // issue slots, its threadgroup port and its share of L2 to themselves, so
    // each runs faster than it would in a full grid. Measured on the split-K
    // sweeps, where the split count is exactly a lane-count dial: 64x2048x64
    // at 64x64 runs 0.202 ms with 20,480 lanes resident and 0.310 with 81,920
    // — the linear law says the starved grid should have been 3.2x slower and
    // it is 0.65x. A cube root reproduces that and the 64x2048x256 and
    // 256x2048x64 split curves; a linear one over-splits every one of them by
    // 2-4x.
    let issue_scaled = if resident >= target {
        issue
    } else {
        issue * integer_root(target * 1_000_000_000 / resident.max(1), 3) / 1_000
    };
    issue_scaled.max(t_dram) + t_combine
}

/// Floor of the `n`th root, by Newton iteration on integers so the argmin
/// stays exactly reproducible across platforms (a floating `powf` is not).
fn integer_root(value: u128, n: u32) -> u128 {
    if value < 2 {
        return value;
    }
    let mut x = 1u128 << (value.ilog2() / n + 1);
    loop {
        let next = ((u128::from(n) - 1) * x + value / x.pow(n - 1)) / u128::from(n);
        if next >= x {
            return x;
        }
        x = next;
    }
}

/// Split counts worth scoring: never splitting, plus every divisor of the K
/// loop that leaves at least two iterations per workgroup. The 64 ceiling
/// bounds the candidate count; epilogues make splitting illegal outright
/// (partials carry no epilogue identity).
fn split_candidates(k_iterations: u32, has_epilogues: bool) -> impl Iterator<Item = u32> {
    let limit = if has_epilogues {
        1
    } else {
        (k_iterations / 2).min(64)
    };
    (1..=limit.max(1)).filter(move |d| *d == 1 || k_iterations.is_multiple_of(*d))
}

/// Staged operand tile pairs worth scoring. Two pairs overlap the next K
/// tile's fill with the current tile's MMAs; one pair halves the workgroup's
/// threadgroup footprint, so a core holds more of them and their epilogue
/// drains cover each other. The split-K partials body is one pair outright:
/// a split grid exists to raise occupancy and a second pair halves it, which
/// measured 55% of wall time on 64x2048x256 at identical tile, splits and
/// grid.
fn staging_depths(splits: u32) -> impl Iterator<Item = u32> {
    if splits > 1 { 1..=1 } else { 1..=2 }
}

/// The best split count and staging depth for one fixed geometry on the grid
/// that actually launches. Monotone non-increasing in `dispatch.segments`: every term is
/// linear in the segment count except the occupancy scaling, whose starved
/// branch is independent of it, so a larger group reaches the saturated
/// plateau at an ever-smaller split count.
pub(super) fn plan_coop_splits(
    dispatch: CoopDispatch,
    tile: CoopTile,
    rg: u32,
    cg: u32,
    policy: &DispatchPolicy,
    subgroup_width: u32,
) -> (u32, u32) {
    let Some(entry) = coop_tile_entries()
        .iter()
        .find(|entry| entry.tile.bm == tile.bm && entry.tile.bn == tile.bn && entry.tile.bk == tile.bk)
    else {
        return (1, 2);
    };
    let stage = stage_element(dispatch.datatype);
    let k_iterations = dispatch.k.div_ceil(tile.bk);
    let mut best: Option<(u128, u32, u32)> = None;
    for splits in split_candidates(k_iterations, dispatch.has_epilogues) {
        for buffers in staging_depths(splits) {
            let score = score_fs(
                dispatch.m,
                dispatch.k,
                dispatch.n,
                dispatch.batch,
                dispatch.segments.max(1),
                entry,
                rg,
                cg,
                splits,
                buffers,
                stage,
                policy,
                subgroup_width,
            );
            // Ascending candidates with a strict comparison: an exact tie
            // keeps the smaller split count (less scratch) and, within a
            // split count, the shallower staging (less workgroup memory).
            if best.is_none_or(|(best_score, ..)| score < best_score) {
                best = Some((score, splits, buffers));
            }
        }
    }
    best.map_or((1, 2), |(_, splits, buffers)| (splits, buffers))
}

/// The tile geometry and its subgroup split, scored at `probe_group` with
/// every candidate free to pick its own best split count. `None` routes the
/// contraction to the vector/generic families.
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_coop_tile(
    m: u32,
    k: u32,
    n: u32,
    batch: u32,
    datatype: crate::DataTypeEnum,
    has_epilogues: bool,
    probe_group: u32,
    policy: &DispatchPolicy,
    subgroup_width: u32,
) -> Option<(CoopTile, u32, u32)> {
    if m == 0 || n == 0 || k == 0 || batch == 0 {
        return None;
    }
    // The kernels stage operand tiles in the storage element (`staging` stays
    // off in production paths).
    let stage = stage_element(datatype);
    let mut best: Option<((u128, u32, Reverse<u32>, Reverse<u32>), CoopTile, u32, u32)> = None;
    for entry in coop_tile_entries() {
        let (bm, bn, bk) = (entry.tile.bm, entry.tile.bn, entry.tile.bk);
        if entry.single_buffered {
            continue;
        }
        if entry.workgroup_bytes(stage) > u64::from(policy.max_workgroup_storage_bytes()) {
            continue;
        }
        let (rg, cg) = entry.subgroup_split();
        let threads = rg * cg * subgroup_width;
        if threads == 0 || threads > policy.max_workgroup_lanes() {
            continue;
        }
        let k_iterations = k.div_ceil(bk);
        let Some(score) = split_candidates(k_iterations, has_epilogues)
            .flat_map(|splits| staging_depths(splits).map(move |buffers| (splits, buffers)))
            .map(|(splits, buffers)| {
                score_fs(
                    m,
                    k,
                    n,
                    batch,
                    probe_group.max(1),
                    entry,
                    rg,
                    cg,
                    splits,
                    buffers,
                    stage,
                    policy,
                    subgroup_width,
                )
            })
            .min()
        else {
            continue;
        };
        // The score is invariant to `n_passes` by construction — a p-pass
        // profile does p times the per-workgroup work over a p-times-smaller
        // grid — and the recorded sweeps say fewer passes wins at every
        // measured shape, so it leads the tie-break. The two Reverse levels
        // reproduce the previous selector's tie-breaks and are reached.
        // A tile that wastes more than a quarter of its work on padding is
        // not a candidate — a truly degenerate (gemv-shaped) contraction is
        // one where *no* tile clears the bar, and that is the routing signal
        // this function returns `None` for. Screening each candidate rather
        // than vetoing the score winner matters: the table's narrow entries
        // (16x64, 64x16) fit an M or N of 196 to within 6%, while the winner
        // on score is a wide tile that pads it to 256 and trips the bar. The
        // veto form declined those shapes outright, and since the dense
        // selector commits to the cooperative family from device capability
        // alone, "declined" meant the generic fused reduce — ~40x slower on
        // exactly the conv-backward shapes that reach it.
        //
        // It compares against kernel families this model cannot cost, so it
        // stays a guard rather than a term.
        let padded_macs = u128::from(m.div_ceil(bm))
            * u128::from(n.div_ceil(bn))
            * u128::from(batch)
            * u128::from(bm)
            * u128::from(bn)
            * u128::from(u64::from(k.div_ceil(bk)) * u64::from(bk));
        let useful_macs = u128::from(m)
            * u128::from(n)
            * u128::from(batch)
            * (u128::from(k.div_ceil(bk)) * u128::from(bk));
        if padded_macs * 4 > useful_macs * 5 && std::env::var_os("FUSOR_OLD_PAD_GUARD").is_none() {
            continue;
        }
        // The score is invariant to `n_passes` by construction — a p-pass
        // profile does p times the per-workgroup work over a p-times-smaller
        // grid — and the recorded sweeps say fewer passes wins at every
        // measured shape, so it leads the tie-break. The two Reverse levels
        // reproduce the previous selector's tie-breaks and are reached.
        let key = (score, entry.n_passes, Reverse(bm), Reverse(bn));
        if best.as_ref().is_none_or(|(best_key, ..)| key < *best_key) {
            best = Some((key, CoopTile::new(bm, bn, bk), rg, cg));
        }
    }
    let (_, tile, rg, cg) = best?;
    Some((tile, rg, cg))
}

/// Traversal-order parameter for the dense coop grid, decided per shape
/// from the raw swizzle sweeps (`bench_coop_tiles` `SWIZZLE=1,4,8,16`, both
/// element types, round-robin minima; winners with margins):
///
/// | shape (m,k,n)      | f32                | f16          |
/// |--------------------|--------------------|--------------|
/// | 16384,384,1536     | tie                | tie          |
/// | 16384,1536,384     | sw8 +4.5%          | sw1 +4.4%    |
/// | 384,16384,1536     | sw8 +4%            | sw8 +3.6%    |
/// | 16384,384,384      | tie                | tie          |
/// | 4096,4096,4096     | sw4 +2.7% (sw1 2nd)| sw1 +4.3%    |
/// | 16384,3072,1536    | sw1 +7%            | sw1 +2.2%    |
///
/// The mechanisms the winners trace: the swizzle exists to share B column
/// slabs across the resident wavefront, and it pays exactly when B is too
/// big to sit in the LLC; K-deep contractions are A-streaming-bound and
/// prefer plain row-major order (the swizzle's M-jumps break the A read
/// stream); halved f16 operands halve cache pressure and shift small-B
/// shapes to row-major too. The one point the rule leaves on the table is
/// 4096-cube f32 (sw4 beats the rule's sw8 by 2.7%) — an isolated winner a
/// dedicated branch would overfit.
pub(super) fn swizzle_group_m(
    m: usize,
    k: usize,
    n: usize,
    datatype: crate::DataTypeEnum,
) -> u32 {
    let _ = m;
    let element_size = datatype.element_size() as u128;
    let b_bytes = k as u128 * n as u128 * element_size;
    const LLC_CLASS: u128 = 32 << 20;
    const SMALL_B: u128 = 4 << 20;
    if k >= 2048 && b_bytes <= LLC_CLASS {
        // K-deep, cacheable B: streaming A wins (f32 +7%, f16 +2.2%;
        // 4096-cube f16 +4.3%).
        return 1;
    }
    if element_size == 2 && b_bytes <= SMALL_B {
        // f16 with a small B: row-major wins (+4.4% at 16384x1536x384).
        return 1;
    }
    // B-slab sharing pays when B outsizes the LLC (+4-4.5% at
    // 384x16384x1536, both dtypes) and measures as a tie on small grids.
    fusor_tile_ir_kernels::DEFAULT_SWIZZLE_GROUP_M
}

/// The dense matmul family's complete routing for one contraction, pure in
/// device-derived inputs: family variant, cooperative plan, and traversal
/// group. One entry point makes the whole decision surface golden-testable —
/// new scoring terms change this function's output table or they change
/// nothing.
#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct DensePlan {
    pub variant: super::variants::DenseMatmulVariant,
    /// `(tile, row_groups, col_groups, splits, stage_buffers)`.
    pub coop: Option<(CoopTile, u32, u32, u32, u32)>,
    pub swizzle_group_m: u32,
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_dense_matmul(
    m: usize,
    k: usize,
    n: usize,
    batch: u32,
    probe_group: u32,
    segments: u32,
    datatype: crate::DataTypeEnum,
    policy: &DispatchPolicy,
    max_subgroup_size: u32,
    caps: crate::kernel_selection::KernelDeviceCaps,
) -> DensePlan {
    let ctx = super::variants::DenseMatmulCtx {
        coop_kinds: super::variants::dense_coop_kinds_from_datatype(datatype),
    };
    let shape = crate::kernel_selection::KernelShape::new([m, k, n]);
    let variant = super::variants::dense_matmul_selector()
        .select(shape, &ctx, caps)
        .expect("dense matmul selector has a catch-all rule");
    // Exactly how production composes the two stages: the tile at the fixed
    // probe group (allocation precedes the merge partition), the split count
    // at the grid that actually launches.
    let coop = (variant == super::variants::DenseMatmulVariant::Coop)
        .then(|| {
            let (tile, rg, cg) = plan_coop_tile(
                m as u32,
                k as u32,
                n as u32,
                batch,
                datatype,
                false,
                probe_group,
                policy,
                max_subgroup_size,
            )?;
            let (splits, buffers) = plan_coop_splits(
                CoopDispatch {
                    m: m as u32,
                    k: k as u32,
                    n: n as u32,
                    batch,
                    segments,
                    datatype,
                    has_epilogues: false,
                },
                tile,
                rg,
                cg,
                policy,
                max_subgroup_size,
            );
            Some((tile, rg, cg, splits, buffers))
        })
        .flatten();
    DensePlan {
        variant,
        coop,
        swizzle_group_m: swizzle_group_m(m, k, n, datatype),
    }
}
