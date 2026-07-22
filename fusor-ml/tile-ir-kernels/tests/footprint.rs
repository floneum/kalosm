//! Deviceless IR-vs-analytic workgroup-footprint checks: every
//! `COOP_TILE_TABLE` entry and the flash-attention forward kernel must lower
//! to exactly the byte count their `workgroup_bytes` formulas report, so the
//! selection layers can plan occupancy from pure arithmetic.

use fusor_tile_ir::{tile, ScalarElement, Shape};
use fusor_tile_ir_kernels::{
    coop_tile_entries, flash_attention_f32, flash_attention_supported,
    flash_attention_workgroup_bytes, try_batched_coop_matmul, try_batched_coop_matmul_split_k,
    CoopTileEntry, DenseCoopMatmulConfig, DenseMatmulEpilogues, DenseMatmulShape,
    DenseMatmulTensors, FlashAttentionLayouts, FlashAttentionShape, FlashOperandLayout,
    SubgroupConfig, DEFAULT_SWIZZLE_GROUP_M,
};

fn subgroup_config() -> SubgroupConfig {
    SubgroupConfig::fixed(fusor_tile_ir::SubgroupToken::new_unchecked(), 32)
}

/// Build one table entry's kernel IR without buffers: the standard path for
/// `splits: None` (single- or double-buffered per the entry), the split-K
/// partials kernel otherwise.
fn coop_matmul_ir(
    entry: &CoopTileEntry,
    storage: ScalarElement,
    staging: Option<ScalarElement>,
    splits: Option<u32>,
) -> fusor_tile_ir::KernelIr {
    let geometry = entry.tile;
    tile::build(|program| {
        let shape = DenseMatmulShape {
            batch: 1,
            m: geometry.bm,
            k: geometry.bk * 4,
            n: geometry.bn,
        };
        let a = program.storage_read(storage.element(), Shape::new([shape.m, shape.k]));
        let b = program.storage_read(storage.element(), Shape::new([shape.k, shape.n]));
        // Split-K over-allocates the output with one scratch slice per split.
        let y_rows = shape.m * (splits.unwrap_or(0) + 1);
        let y = program.storage_write(storage.element(), Shape::new([y_rows, shape.n]));
        let tensors = DenseMatmulTensors {
            a: &a,
            b: &b,
            y: &y,
        };
        let config = DenseCoopMatmulConfig {
            coop: fusor_tile_ir::CoopMatrixToken::new_unchecked(),
            subgroups: subgroup_config(),
            tile: geometry,
            staging,
            swizzle_group_m: DEFAULT_SWIZZLE_GROUP_M,
        };
        let emitted = match splits {
            Some(splits) => {
                try_batched_coop_matmul_split_k(program, tensors, shape, splits, 65_535, config)
            }
            None => try_batched_coop_matmul(
                program,
                tensors,
                shape,
                &DenseMatmulEpilogues::empty(),
                65_535,
                config,
            ),
        };
        assert!(emitted, "kernel declined {geometry:?}");
    })
}

/// Every table entry's lowered footprint equals the analytic formula, for
/// both storage elements, through both the double-buffered perf body and
/// the single-buffered body (the entry's flag routes it).
#[test]
fn coop_table_footprints_match_ir() {
    for entry in coop_tile_entries() {
        for storage in [ScalarElement::F32, ScalarElement::F16] {
            let ir = coop_matmul_ir(entry, storage, None, None);
            assert_eq!(
                ir.workgroup_bytes(),
                entry.workgroup_bytes(storage),
                "{:?} {storage:?}",
                entry.tile,
            );
        }
    }
}

/// The split-K partials kernel stages exactly one tile pair.
#[test]
fn split_k_footprints_match_ir() {
    for entry in coop_tile_entries() {
        // The split path declines single-buffered geometry.
        if entry.single_buffered {
            continue;
        }
        let ir = coop_matmul_ir(entry, ScalarElement::F32, None, Some(2));
        assert_eq!(
            ir.workgroup_bytes(),
            entry.tile.stage_pair_elements(entry.n_passes) * ScalarElement::F32.byte_size(),
            "{:?} split-k",
            entry.tile,
        );
    }
}

/// `staging: Some(F16)` over f32 storage stages the whole pair set in f16 —
/// the formula's stage axis, validated without enabling staging in any
/// production path.
#[test]
fn f16_staging_over_f32_storage_matches_ir() {
    for entry in coop_tile_entries() {
        // The single-buffered body ignores staging.
        if entry.single_buffered {
            continue;
        }
        let ir = coop_matmul_ir(entry, ScalarElement::F32, Some(ScalarElement::F16), None);
        assert_eq!(
            ir.workgroup_bytes(),
            entry.workgroup_bytes(ScalarElement::F16),
            "{:?} staged f16",
            entry.tile,
        );
    }
}

/// `single_buffered` is a derived property, not a tuning choice: set exactly
/// when two f32 pairs would overflow Apple's 32 KB threadgroup-memory limit.
#[test]
fn single_buffered_is_exactly_the_32kb_overflow() {
    for entry in coop_tile_entries() {
        let two_pair_f32 =
            2 * entry.tile.stage_pair_elements(entry.n_passes) * ScalarElement::F32.byte_size();
        assert_eq!(
            entry.single_buffered,
            two_pair_f32 > 32 * 1024,
            "{:?}: two-pair f32 footprint {two_pair_f32}",
            entry.tile,
        );
    }
}

fn flash_shape(head_dim: u32) -> FlashAttentionShape {
    FlashAttentionShape {
        batch: 1,
        heads: 1,
        kv_groups: 1,
        q_len: 32,
        kv_len: 32,
        head_dim,
        scale: 1.0,
        causal: false,
    }
}

fn flash_ir(head_dim: u32, storage: ScalarElement) -> fusor_tile_ir::KernelIr {
    tile::build(|program| {
        let shape = flash_shape(head_dim);
        let q_elems = shape.q_len * head_dim;
        let kv_elems = shape.kv_len * head_dim;
        let q = program.storage_read(storage.element(), Shape::new([q_elems]));
        let k = program.storage_read(storage.element(), Shape::new([kv_elems]));
        let v = program.storage_read(storage.element(), Shape::new([kv_elems]));
        let o = program.storage_write(storage.element(), Shape::new([q_elems]));
        let layouts = FlashAttentionLayouts {
            q: FlashOperandLayout::contiguous(1, shape.q_len, head_dim),
            k: FlashOperandLayout::contiguous(1, shape.kv_len, head_dim),
            v: FlashOperandLayout::contiguous(1, shape.kv_len, head_dim),
            o: FlashOperandLayout::contiguous(1, shape.q_len, head_dim),
        };
        assert!(flash_attention_f32(
            program,
            &q,
            &k,
            &v,
            None,
            &o,
            &layouts,
            shape,
            subgroup_config(),
            fusor_tile_ir::CoopMatrixToken::new_unchecked(),
            65_535,
        ));
    })
}

/// The forward kernel's lowered footprint equals the analytic formula for
/// every supported head dim and stage element.
#[test]
fn flash_forward_footprints_match_ir() {
    let mut covered = 0;
    for head_dim in [32, 64, 80] {
        if !flash_attention_supported(&flash_shape(head_dim), subgroup_config()) {
            continue;
        }
        for storage in [ScalarElement::F32, ScalarElement::F16] {
            let ir = flash_ir(head_dim, storage);
            assert_eq!(
                ir.workgroup_bytes(),
                flash_attention_workgroup_bytes(head_dim, storage),
                "d={head_dim} {storage:?}",
            );
            covered += 1;
        }
    }
    assert!(covered >= 4, "supported-shape sweep collapsed: {covered}");
}

/// The f16 d=64 forward kernel sits at 16.0 KB — the two-workgroups-per-core
/// residency boundary on Apple's 32 KB budget, inside WebGPU's 16 KB default
/// workgroup-storage limit.
#[test]
fn flash_f16_d64_holds_the_residency_boundary() {
    let bytes = flash_attention_workgroup_bytes(64, ScalarElement::F16);
    assert_eq!(bytes, 16_022);
    assert!(bytes <= 16 << 10);
}

fn flash_ir_byte_arena(head_dim: u32, storage: ScalarElement) -> fusor_tile_ir::KernelIr {
    tile::build(|program| {
        program.enable_byte_arena(fusor_tile_ir::ByteArenaToken::new_unchecked());
        let shape = flash_shape(head_dim);
        let q_elems = shape.q_len * head_dim;
        let kv_elems = shape.kv_len * head_dim;
        let q = program.storage_read(storage.element(), Shape::new([q_elems]));
        let k = program.storage_read(storage.element(), Shape::new([kv_elems]));
        let v = program.storage_read(storage.element(), Shape::new([kv_elems]));
        let o = program.storage_write(storage.element(), Shape::new([q_elems]));
        let layouts = FlashAttentionLayouts {
            q: FlashOperandLayout::contiguous(1, shape.q_len, head_dim),
            k: FlashOperandLayout::contiguous(1, shape.kv_len, head_dim),
            v: FlashOperandLayout::contiguous(1, shape.kv_len, head_dim),
            o: FlashOperandLayout::contiguous(1, shape.q_len, head_dim),
        };
        assert!(flash_attention_f32(
            program,
            &q,
            &k,
            &v,
            None,
            &o,
            &layouts,
            shape,
            subgroup_config(),
            fusor_tile_ir::CoopMatrixToken::new_unchecked(),
            65_535,
        ));
    })
}

/// The byte arena self-selects: it only replaces typed regions when
/// cross-stride reuse actually shrinks the footprint. The f16 forward
/// kernel's tiles are all live across the KV loop, so today the arena finds
/// nothing and the footprint must stay exactly the regions number.
#[test]
fn flash_f16_byte_arena_footprint() {
    let regions = flash_ir(64, ScalarElement::F16).workgroup_bytes();
    let packed = flash_ir_byte_arena(64, ScalarElement::F16).workgroup_bytes();
    assert_eq!(regions, 16_022);
    assert_eq!(packed, regions);
}
