//! Workgroup-tile arena sharing: barrier soundness fixtures.
//!
//! Tile A (8x8 f32, 256 B) and tile B (4x8 f32, 128 B) touch on either side
//! of a barrier. Sharing collapses the footprint to 256 B; refusing keeps
//! 384 B. A barrier inside a loop that may break early, return, or run zero
//! iterations can be skipped at runtime, so it must not enable sharing for
//! tiles living outside that loop.

use super::*;
use crate::tile;

const SHARED: u64 = 256;
const UNSHARED: u64 = 256 + 128;

/// Build the two-tile fixture with `body` between the touches. `body` runs
/// with the program block and must contain the only barrier.
fn two_tile_fixture(
    between: impl FnOnce(&mut tile::TileBlock),
) -> KernelIr {
    tile::build(|phase| {
        let a = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
        let b = phase.alloc_workgroup_tile(ScalarElement::F32, 4, 8);
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            program.store_workgroup(&a, lane.clone(), 1.0f32);
            between(program);
            program.store_workgroup(&b, lane, 2.0f32);
        });
    })
}

#[test]
fn shares_through_top_level_barrier() {
    let ir = two_tile_fixture(|program| {
        program.workgroup_barrier();
    });
    assert_eq!(ir.workgroup_bytes(), SHARED);
}

#[test]
fn shares_through_static_loop_barrier() {
    // A static-count, break-free loop executes its body on every pass, so an
    // in-loop barrier is guaranteed and sharing through it is sound.
    let ir = two_tile_fixture(|program| {
        program.loop_range(4, |program, _| {
            program.workgroup_barrier();
        });
    });
    assert_eq!(ir.workgroup_bytes(), SHARED);
}

#[test]
fn no_share_through_unstructured_loop_barrier() {
    // An unstructured loop's count is data-dependent: the barrier may never
    // execute (or not on the exit path), so it cannot separate the tiles.
    let ir = two_tile_fixture(|program| {
        let lane = program.lane();
        program.loop_forever(|program| {
            program.workgroup_barrier();
            program.break_if(lane.clone().lt(32u32));
        });
    });
    assert_eq!(ir.workgroup_bytes(), UNSHARED);
}

#[test]
fn no_share_through_breaking_static_loop_barrier() {
    // A break before the barrier can skip it on the final iteration.
    let ir = two_tile_fixture(|program| {
        let lane = program.lane();
        program.loop_range(4, |program, _| {
            program.break_if(lane.clone().lt(32u32));
            program.workgroup_barrier();
        });
    });
    assert_eq!(ir.workgroup_bytes(), UNSHARED);
}

#[test]
fn no_share_through_returning_static_loop_barrier() {
    // A mid-body return exits before later barriers just like a break.
    let ir = two_tile_fixture(|program| {
        let lane = program.lane();
        program.loop_range(4, |program, _| {
            program.if_then(lane.clone().lt(32u32), |program| program.return_());
            program.workgroup_barrier();
        });
    });
    assert_eq!(ir.workgroup_bytes(), UNSHARED);
}

#[test]
fn accumulator_update_is_live_across_loop() {
    // An accumulator update executes at the end of EVERY iteration — after
    // any in-loop barrier — so a tile read only by the update is live across
    // the whole loop and an in-loop barrier cannot separate it from a tile
    // touched after the loop. Built as raw IR: no builder emits a counted
    // accumulator loop without a break, but the IR admits it.
    let workgroup =
        |shape: [u32; 2]| Layout::contiguous(MemoryLevel::Workgroup, Shape::new(shape));
    let tile_a: Tile = std::rc::Rc::new(TileDecl {
        element: ElementType::F32,
        layout: workgroup([8, 8]),
    });
    let tile_b: Tile = std::rc::Rc::new(TileDecl {
        element: ElementType::F32,
        layout: workgroup([4, 8]),
    });
    let acc_local: Local = std::rc::Rc::new(LocalDecl {
        element: ElementType::F32,
    });
    let index_local: Local = std::rc::Rc::new(LocalDecl {
        element: ElementType::U32,
    });
    let lit_u32 = |value: u32| Expr::new(ExprKind::Literal(TileLiteral::U32(value)), ElementType::U32);
    let lit_f32 = |value: f32| {
        Expr::new(
            ExprKind::Literal(TileLiteral::f32(value)),
            ElementType::F32,
        )
    };

    let mut ir = KernelIr::default();
    ir.block = 32;
    ir.body = vec![
        Stmt::Loop {
            count: Some(lit_u32(4)),
            index: Some(index_local),
            accumulators: vec![Accumulator {
                local: acc_local,
                init: lit_f32(0.0),
                update: Expr::new(
                    ExprKind::LoadTile {
                        tile: tile_a,
                        index: Box::new(lit_u32(0)),
                    },
                    ElementType::F32,
                ),
            }],
            body: vec![Stmt::Barrier],
        },
        Stmt::StoreTile {
            dst: tile_b,
            index: Box::new(lit_u32(0)),
            value: lit_f32(2.0),
        },
    ];
    assert_eq!(ir.workgroup_bytes(), UNSHARED);
}

#[test]
fn cross_type_tiles_share_one_region() {
    // f32 and u32 have the same 4-byte stride: barrier-separated disjoint
    // tiles share one region, emitted with the class-neutral u32 type and a
    // value bitcast at each access.
    let ir = tile::build(|phase| {
        let a = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
        let b = phase.alloc_workgroup_tile(ScalarElement::U32, 4, 8);
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            program.store_workgroup(&a, lane.clone(), 1.0f32);
            let read_back = program.load_workgroup(&a, lane.clone());
            program.store_workgroup(&a, lane.clone(), read_back);
            program.workgroup_barrier();
            program.store_workgroup(&b, lane, 2u32);
        });
    });
    assert_eq!(ir.workgroup_bytes(), SHARED);
    let lowered = lower_or_fail(&ir, "cross-type region");
    let function = &lowered.module().entry_points[0].function;
    let bitcasts = function
        .expressions
        .iter()
        .filter(|(_, expr)| {
            matches!(
                expr,
                naga::Expression::As {
                    convert: None,
                    ..
                }
            )
        })
        .count();
    // Both f32 stores, the f32 load, and the u32 store's canonical is u32
    // itself: three f32<->u32 casts.
    assert_eq!(bitcasts, 3);
}

#[test]
fn coop_consumed_tile_pins_its_region_type() {
    // A tile consumed as a raw cooperative-matrix pointer cannot live in a
    // widened region: the u32 tile must get its own allocation.
    let ir = tile::build(|phase| {
        let coop = crate::CoopMatrixToken::new_unchecked();
        let a = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
        let b = phase.alloc_workgroup_tile(ScalarElement::U32, 4, 8);
        let y = phase.storage_write(ScalarElement::F32.element(), Shape::new([8, 8]));
        phase.program_grid(32, [1, 1, 1], |program| {
            let acc = coop.alloc_coop_acc(program, ScalarElement::F32, 8, 8);
            let a_frag = coop.coop_load_a(program, &a, 0u32, 0u32, ScalarElement::F32, 8, 8);
            let b_frag = coop.coop_load_b(program, &a, 0u32, 0u32, ScalarElement::F32, 8, 8);
            let c = coop.coop_zero(program, ScalarElement::F32, 8, 8);
            coop.coop_store_local(program, &acc, coop.coop_mma(program, a_frag, b_frag, c));
            coop.coop_store(program, &acc, &y, 0u32, 0u32);
            program.workgroup_barrier();
            let lane = program.lane();
            program.store_workgroup(&b, lane, 2u32);
        });
    });
    assert_eq!(ir.workgroup_bytes(), UNSHARED);
}

#[test]
fn mixed_stride_tiles_pack_into_byte_arena() {
    // With the byte-arena backend proved, an f16 tile reuses the f32 tile's
    // bytes after a barrier: footprint collapses to the f32 extent (16-byte
    // aligned) instead of the sum.
    let mut ir = tile::build(|phase| {
        let a = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
        let b = phase.alloc_workgroup_tile(ScalarElement::F16, 4, 8);
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            program.store_workgroup(&a, lane.clone(), 1.0f32);
            program.workgroup_barrier();
            program.store_workgroup(&b, lane, 2.0f32);
        });
    });
    // Regions mode first: f16 cannot join the 4-byte class, so both
    // allocations exist (256 + 64 bytes).
    assert_eq!(ir.workgroup_bytes(), 256 + 64);
    ir.byte_arena = true;
    assert_eq!(ir.workgroup_bytes(), 256);
    // The alias emission path validates end-to-end: one arena global plus
    // typed aliased globals, accepted by the fork validator.
    let lowered = lower_or_fail(&ir, "byte-arena emission");
    let aliased = lowered
        .module()
        .global_variables
        .iter()
        .filter(|(_, global)| global.workgroup_alias.is_some())
        .count();
    assert_eq!(aliased, 2);
}

#[test]
fn byte_arena_keeps_concurrent_tiles_disjoint() {
    // Without a separating barrier the tiles overlap in time: the packer
    // pushes the f16 tile past the f32 extent.
    let mut ir = tile::build(|phase| {
        let a = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
        let b = phase.alloc_workgroup_tile(ScalarElement::F16, 4, 8);
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            program.store_workgroup(&a, lane.clone(), 1.0f32);
            program.store_workgroup(&b, lane, 2.0f32);
        });
    });
    ir.byte_arena = true;
    assert_eq!(ir.workgroup_bytes(), 256 + 64);
}

fn barrier_count(stmts: &[Stmt]) -> usize {
    stmts
        .iter()
        .map(|stmt| match stmt {
            Stmt::Barrier => 1,
            Stmt::Loop { body, .. } => barrier_count(body),
            Stmt::If { accept, reject, .. } => barrier_count(accept) + barrier_count(reject),
            _ => 0,
        })
        .sum()
}

#[test]
fn elision_removes_duplicate_barrier() {
    let ir = two_tile_fixture(|program| {
        program.workgroup_barrier();
        program.workgroup_barrier();
    });
    assert_eq!(barrier_count(&ir.body), 1);
    assert_eq!(ir.workgroup_bytes(), SHARED);
}

#[test]
fn elision_removes_trailing_barrier() {
    let ir = tile::build(|phase| {
        let a = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            program.store_workgroup(&a, lane, 1.0f32);
            program.workgroup_barrier();
        });
    });
    assert_eq!(barrier_count(&ir.body), 0);
}

#[test]
fn elision_keeps_wrap_around_separators() {
    // The reverted-elision incident in kernel form: inside a loop, one
    // barrier orders this iteration's write before its read, the other
    // orders the read before the NEXT iteration's write. Neither backs the
    // other up; both must survive.
    let ir = tile::build(|phase| {
        let t = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            let scratch = program.private(ElementType::F32);
            program.loop_range(4, |program, _| {
                program.store_workgroup(&t, lane.clone(), 1.0f32);
                program.workgroup_barrier();
                let value = program.load_workgroup(&t, lane.clone());
                program.store_local(&scratch, value);
                program.workgroup_barrier();
            });
        });
    });
    assert_eq!(barrier_count(&ir.body), 2);
}

/// Phased in-loop fixture: tiles touched in disjoint per-iteration phases,
/// with barriers controlled by the caller. A is 8x8 f32, B is 4x8 f32.
fn phased_loop_fixture(
    barrier_between: bool,
    barrier_at_end: bool,
    unstructured: bool,
) -> KernelIr {
    tile::build(|phase| {
        let a = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
        let b = phase.alloc_workgroup_tile(ScalarElement::F32, 4, 8);
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            let scratch = program.private(ElementType::F32);
            let body = |program: &mut tile::TileBlock| {
                program.store_workgroup(&a, lane.clone(), 1.0f32);
                let read = program.load_workgroup(&a, lane.clone());
                program.store_local(&scratch, read);
                if barrier_between {
                    program.workgroup_barrier();
                }
                program.store_workgroup(&b, lane.clone(), 2.0f32);
                let read = program.load_workgroup(&b, lane.clone());
                program.store_local(&scratch, read);
                if barrier_at_end {
                    program.workgroup_barrier();
                }
            };
            if unstructured {
                program.loop_forever(|program| {
                    body(program);
                    program.break_if(lane.clone().lt(32u32));
                });
            } else {
                program.loop_range(4, |program, _| body(program));
            }
        });
    })
}

#[test]
fn phased_in_loop_tiles_share() {
    // Disjoint phases, a barrier between them, and a barrier covering the
    // wrap back to the first phase: the tiles share one region.
    let ir = phased_loop_fixture(true, true, false);
    assert_eq!(ir.workgroup_bytes(), SHARED);
    lower_or_fail(&ir, "phased in-loop sharing");
}

#[test]
fn phase_sharing_needs_the_forward_barrier() {
    let ir = phased_loop_fixture(false, true, false);
    assert_eq!(ir.workgroup_bytes(), UNSHARED);
}

#[test]
fn phase_sharing_needs_the_wrap_barrier() {
    let ir = phased_loop_fixture(true, false, false);
    assert_eq!(ir.workgroup_bytes(), UNSHARED);
}

#[test]
fn phase_sharing_survives_break_loops() {
    // Taking the back edge means the full body executed, so in-loop
    // barriers stay valid phase separators even in unstructured loops.
    let ir = phased_loop_fixture(true, true, true);
    assert_eq!(ir.workgroup_bytes(), SHARED);
}

#[test]
fn phase_sharing_does_not_chain_transitively() {
    // A->B and B->C phase separation does NOT cover the C->A wrap: with
    // barriers only between the phases, C must not join A and B's region.
    let ir = tile::build(|phase| {
        let a = phase.alloc_workgroup_tile(ScalarElement::F32, 8, 8);
        let b = phase.alloc_workgroup_tile(ScalarElement::F32, 4, 8);
        let c = phase.alloc_workgroup_tile(ScalarElement::F32, 4, 4);
        phase.program_grid(32, [1, 1, 1], |program| {
            let lane = program.lane();
            let scratch = program.private(ElementType::F32);
            program.loop_range(4, |program, _| {
                let mut touch = |program: &mut tile::TileBlock, tile: &tile::WorkgroupTile| {
                    program.store_workgroup(tile, lane.clone(), 1.0f32);
                    let read = program.load_workgroup(tile, lane.clone());
                    program.store_local(&scratch, read);
                };
                touch(program, &a);
                program.workgroup_barrier();
                touch(program, &b);
                program.workgroup_barrier();
                touch(program, &c);
            });
        });
    });
    // A and B share (forward barrier between phases, wrap covered by the
    // second barrier for B and the first for... A's wrap: second barrier
    // sits after B's phase, before the back edge). C shares with neither:
    // its own phase has no trailing barrier, so the C->occupant wraps are
    // uncovered.
    assert_eq!(ir.workgroup_bytes(), 256 + 64);
}
