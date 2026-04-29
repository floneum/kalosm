use super::*;

fn storage_view(buffer: BufferRef, layout: Layout) -> StorageView {
    StorageView::root(buffer, layout)
}

fn block_contains_gemm(block: &Block) -> bool {
    block.ops().iter().any(|op| match op {
        Op::Gemm(_) => true,
        Op::Block(op) => block_contains_gemm(&op.body),
        Op::Loop(op) => block_contains_gemm(&op.body),
        Op::Partition(op) => block_contains_gemm(&op.body),
        _ => false,
    })
}

fn count_mmas(block: &Block) -> usize {
    block
        .ops()
        .iter()
        .map(|op| match op {
            Op::Mma(_) => 1,
            Op::Block(op) => count_mmas(&op.body),
            Op::Loop(op) => count_mmas(&op.body),
            Op::Partition(op) => count_mmas(&op.body),
            _ => 0,
        })
        .sum()
}

fn block_contains_partition(block: &Block) -> bool {
    block.ops().iter().any(|op| match op {
        Op::Partition(_) => true,
        Op::Block(op) => block_contains_partition(&op.body),
        Op::Loop(op) => block_contains_partition(&op.body),
        _ => false,
    })
}

#[test]
fn loop_body_must_end_with_sync_witness() {
    let shape = Shape::new([32]);
    let layout = Layout::contiguous(MemoryLevel::Workgroup, shape.clone());
    let buffer_layout = Layout::contiguous(MemoryLevel::Storage, shape.clone());
    let src = BufferRef::new(BufferId(0), ElementType::F32);
    let dst = BufferRef::new(BufferId(1), ElementType::F32);
    let tile = TileRef::new(TileId(0), ElementType::F32);
    let ir = build(|mut phase| {
        let src_tensor = phase.storage_tensor::<F32>(shape.clone());
        let dst_tensor = phase.storage_tensor::<F32>(shape.clone());
        phase.range_step(
            |mut phase, _| {
                let tile = phase.alloc_workgroup_tile::<F32>(shape.clone());
                let pending = phase.cooperative_load(tile, &src_tensor);
                let (ready, mut phase) = pending.sync_tile();

                phase.store_ready_to_storage(&ready, &dst_tensor);
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    assert_eq!(
        ir.buffers(),
        &[
            BufferDecl {
                id: BufferId(0),
                element: ElementType::F32,
                layout: buffer_layout.clone(),
                access: BufferAccess::ReadWrite,
            },
            BufferDecl {
                id: BufferId(1),
                element: ElementType::F32,
                layout: buffer_layout,
                access: BufferAccess::ReadWrite,
            },
        ],
    );
    assert_eq!(
        ir.tiles(),
        &[TileDecl {
            id: TileId(0),
            element: ElementType::F32,
            layout,
            level: TileLevel::Workgroup,
            origin: TileOrigin::Allocation,
        }],
    );
    assert_eq!(
        ir.body(),
        &Block::from_ops(vec![Op::Loop(LoopOp {
            kind: LoopKind::RangeStep {
                induction: Dim(0),
                iterations: 1,
            },
            body: Block::from_ops(vec![
                Op::CooperativeLoad(CooperativeLoadOp {
                    dst: tile,
                    src: storage_view(src, Layout::contiguous(MemoryLevel::Storage, shape.clone())),
                    level: TileLevel::Workgroup,
                }),
                Op::Barrier(BarrierOp {
                    scope: BarrierScope::Workgroup,
                }),
                Op::StoreTile(StoreTileOp {
                    src: tile,
                    dst: storage_view(dst, Layout::contiguous(MemoryLevel::Storage, shape.clone())),
                }),
                Op::Barrier(BarrierOp {
                    scope: BarrierScope::Workgroup,
                }),
            ]),
        })]),
    );
}

#[test]
fn outer_ready_tile_can_be_read_inside_loop_body() {
    let shape = Shape::new([32]);
    let layout = Layout::contiguous(MemoryLevel::Workgroup, shape.clone());
    let buffer_layout = Layout::contiguous(MemoryLevel::Storage, shape.clone());
    let src = BufferRef::new(BufferId(0), ElementType::F32);
    let dst = BufferRef::new(BufferId(1), ElementType::F32);
    let tile = TileRef::new(TileId(0), ElementType::F32);
    let ir = build(|mut phase| {
        let src_tensor = phase.storage_tensor::<F32>(shape.clone());
        let dst_tensor = phase.storage_tensor::<F32>(shape.clone());
        let tile = phase.alloc_workgroup_tile::<F32>(shape.clone());
        let pending = phase.cooperative_load(tile, &src_tensor);
        let (ready, phase) = pending.sync_tile();

        phase.range_step(
            |mut phase, _| {
                phase.store_ready_to_storage(&ready, &dst_tensor);
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    assert_eq!(
        ir.buffers(),
        &[
            BufferDecl {
                id: BufferId(0),
                element: ElementType::F32,
                layout: buffer_layout.clone(),
                access: BufferAccess::ReadWrite,
            },
            BufferDecl {
                id: BufferId(1),
                element: ElementType::F32,
                layout: buffer_layout,
                access: BufferAccess::ReadWrite,
            },
        ],
    );
    assert_eq!(
        ir.tiles(),
        &[TileDecl {
            id: TileId(0),
            element: ElementType::F32,
            layout,
            level: TileLevel::Workgroup,
            origin: TileOrigin::Allocation,
        }],
    );
    assert_eq!(
        ir.body(),
        &Block::from_ops(vec![
            Op::CooperativeLoad(CooperativeLoadOp {
                dst: tile,
                src: storage_view(src, Layout::contiguous(MemoryLevel::Storage, shape.clone())),
                level: TileLevel::Workgroup,
            }),
            Op::Barrier(BarrierOp {
                scope: BarrierScope::Workgroup,
            }),
            Op::Loop(LoopOp {
                kind: LoopKind::RangeStep {
                    induction: Dim(0),
                    iterations: 1,
                },
                body: Block::from_ops(vec![
                    Op::StoreTile(StoreTileOp {
                        src: tile,
                        dst: storage_view(
                            dst,
                            Layout::contiguous(MemoryLevel::Storage, shape.clone()),
                        ),
                    }),
                    Op::Barrier(BarrierOp {
                        scope: BarrierScope::Workgroup,
                    }),
                ]),
            }),
        ]),
    );
}

#[test]
fn matmul_tile_body_is_represented_in_the_ir() {
    let bm = 16;
    let bn = 16;
    let bk = 8;

    let acc = TileRef::new(TileId(0), ElementType::F32);
    let a = TileRef::new(TileId(1), ElementType::F32);
    let b = TileRef::new(TileId(2), ElementType::F32);
    let a_buf = BufferRef::new(BufferId(0), ElementType::F32);
    let b_buf = BufferRef::new(BufferId(1), ElementType::F32);
    let c_buf = BufferRef::new(BufferId(2), ElementType::F32);

    let ir = build(|mut phase| {
        let a_in = phase.storage_tensor::<F32>(Shape::new([bm, bk]));
        let b_in = phase.storage_tensor::<F32>(Shape::new([bk, bn]));
        let c_out = phase.storage_tensor::<F32>(Shape::new([bm, bn]));
        let mut acc = phase.alloc_fragment::<F32>(Shape::new([bm, bn]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;

        phase.range_step(
            |mut phase, _| {
                let a = phase.alloc_workgroup_tile::<F32>(Shape::new([bm, bk]));
                let b = phase.alloc_workgroup_tile::<F32>(Shape::new([bk, bn]));
                let pending = phase.cooperative_load_pair(a, &a_in, b, &b_in);
                let (a, b, mut phase) = pending.sync_tiles();

                kernels::gemm::tiled(
                    &mut phase,
                    &a,
                    &b,
                    &mut acc,
                    kernels::gemm::GemmTilePlan::portable(bm, bn, bk),
                );
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
        )
    });

    assert_eq!(
        ir.buffers(),
        &[
            BufferDecl {
                id: BufferId(0),
                element: ElementType::F32,
                layout: Layout::contiguous(MemoryLevel::Storage, Shape::new([bm, bk])),
                access: BufferAccess::ReadWrite,
            },
            BufferDecl {
                id: BufferId(1),
                element: ElementType::F32,
                layout: Layout::contiguous(MemoryLevel::Storage, Shape::new([bk, bn])),
                access: BufferAccess::ReadWrite,
            },
            BufferDecl {
                id: BufferId(2),
                element: ElementType::F32,
                layout: Layout::contiguous(MemoryLevel::Storage, Shape::new([bm, bn])),
                access: BufferAccess::ReadWrite,
            },
        ],
    );
    assert_eq!(ir.tiles().len(), 54);
    assert_eq!(ir.tiles()[0].origin, TileOrigin::Allocation);
    assert_eq!(ir.tiles()[1].origin, TileOrigin::Allocation);
    assert_eq!(ir.tiles()[2].origin, TileOrigin::Allocation);
    assert!(block_contains_partition(ir.body()));
    assert_eq!(count_mmas(ir.body()), 16);
    assert!(!block_contains_gemm(ir.body()));

    let Op::Loop(loop_op) = &ir.body().ops()[1] else {
        panic!("expected loop after accumulator fill");
    };
    assert_eq!(
        &loop_op.body.ops()[0..3],
        &[
            Op::CooperativeLoad(CooperativeLoadOp {
                dst: a,
                src: storage_view(
                    a_buf,
                    Layout::contiguous(MemoryLevel::Storage, Shape::new([bm, bk])),
                ),
                level: TileLevel::Workgroup,
            }),
            Op::CooperativeLoad(CooperativeLoadOp {
                dst: b,
                src: storage_view(
                    b_buf,
                    Layout::contiguous(MemoryLevel::Storage, Shape::new([bk, bn])),
                ),
                level: TileLevel::Workgroup,
            }),
            Op::Barrier(BarrierOp {
                scope: BarrierScope::Workgroup,
            }),
        ],
    );
    assert_eq!(
        &ir.body().ops()[2],
        &Op::StoreTile(StoreTileOp {
            src: acc,
            dst: storage_view(
                c_buf,
                Layout::contiguous(MemoryLevel::Storage, Shape::new([bm, bn])),
            ),
        }),
    );
}

#[test]
fn userland_gemm_emits_partitioned_mma_ir() {
    let bm = 16;
    let bn = 16;
    let bk = 8;

    let ir = build(|mut phase| {
        let a_in = phase.storage_tensor::<F32>(Shape::new([bm, bk]));
        let b_in = phase.storage_tensor::<F32>(Shape::new([bk, bn]));
        let c_out = phase.storage_tensor::<F32>(Shape::new([bm, bn]));
        let mut acc = phase.alloc_fragment::<F32>(Shape::new([bm, bn]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;

        phase.range_step(
            |mut phase, _| {
                let a = phase.alloc_workgroup_tile::<F32>(Shape::new([bm, bk]));
                let b = phase.alloc_workgroup_tile::<F32>(Shape::new([bk, bn]));
                let pending = phase.cooperative_load_pair(a, &a_in, b, &b_in);
                let (a, b, mut phase) = pending.sync_tiles();

                kernels::gemm::tiled(
                    &mut phase,
                    &a,
                    &b,
                    &mut acc,
                    kernels::gemm::GemmTilePlan::portable(bm, bn, bk),
                );
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
        )
    });

    assert_eq!(ir.tiles().len(), 54);
    assert_eq!(
        ir.tiles()[5].origin,
        TileOrigin::View {
            source: TileRef::new(TileId(0), ElementType::F32),
            mapping: ViewMapping::Partition {
                level: TileLevel::Subgroup,
                origin: [0, 0],
            },
        },
    );
    assert_eq!(
        ir.tiles()[8].origin,
        TileOrigin::View {
            source: TileRef::new(TileId(5), ElementType::F32),
            mapping: ViewMapping::Partition {
                level: TileLevel::Thread,
                origin: [0, 0],
            },
        },
    );
    assert_eq!(ir.tiles()[3].layout.shape(), &Shape::new([16, 8]));
    assert_eq!(ir.tiles()[4].layout.shape(), &Shape::new([8, 16]));
    assert_eq!(ir.tiles()[5].layout.shape(), &Shape::new([16, 16]));
    assert_eq!(ir.tiles()[6].layout.shape(), &Shape::new([4, 8]));
    assert_eq!(ir.tiles()[7].layout.shape(), &Shape::new([8, 4]));
    assert_eq!(ir.tiles()[8].layout.shape(), &Shape::new([4, 4]));

    let Op::Loop(loop_op) = &ir.body().ops()[1] else {
        panic!("expected loop after accumulator fill");
    };
    assert!(matches!(loop_op.body.ops()[3], Op::Partition(_)));
    assert_eq!(count_mmas(&loop_op.body), 16);
    assert_eq!(
        ir.tiles()[53].origin,
        TileOrigin::View {
            source: TileRef::new(TileId(5), ElementType::F32),
            mapping: ViewMapping::Partition {
                level: TileLevel::Thread,
                origin: [12, 12],
            },
        },
    );
    assert!(!block_contains_gemm(ir.body()));
}

#[test]
fn legacy_gemm_op_still_expands_to_partitioned_mma_ir() {
    let bm = 16;
    let bn = 16;
    let bk = 8;
    let a = TileRef::new(TileId(0), ElementType::F32);
    let b = TileRef::new(TileId(1), ElementType::F32);
    let acc = TileRef::new(TileId(2), ElementType::F32);
    let ir = KernelIr {
        buffers: Vec::new(),
        tiles: vec![
            TileDecl {
                id: TileId(0),
                element: ElementType::F32,
                layout: Layout::contiguous(MemoryLevel::Workgroup, Shape::new([bm, bk])),
                level: TileLevel::Workgroup,
                origin: TileOrigin::Allocation,
            },
            TileDecl {
                id: TileId(1),
                element: ElementType::F32,
                layout: Layout::contiguous(MemoryLevel::Workgroup, Shape::new([bk, bn])),
                level: TileLevel::Workgroup,
                origin: TileOrigin::Allocation,
            },
            TileDecl {
                id: TileId(2),
                element: ElementType::F32,
                layout: Layout::contiguous(MemoryLevel::Private, Shape::new([bm, bn])),
                level: TileLevel::Thread,
                origin: TileOrigin::Allocation,
            },
        ],
        body: Block::from_ops(vec![Op::Gemm(GemmOp {
            a,
            b,
            acc,
            tiling: GemmTiling::portable(bm, bn, bk),
            backend: MmaBackend::FmaPortable,
        })]),
        next_buffer: 0,
        next_tile: 3,
    };

    let expanded = ir.expand_gemm_to_mma();
    assert_eq!(expanded.tiles().len(), 54);
    assert_eq!(count_mmas(expanded.body()), 16);
    assert!(!block_contains_gemm(expanded.body()));
}

#[test]
fn lowers_to_valid_naga_module() {
    let ir = build(|mut phase| {
        let src = phase.storage_tensor::<F32>(Shape::new([32]));
        let dst = phase.storage_tensor::<F32>(Shape::new([32]));
        phase.range_step(
            |mut phase, _| {
                let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
                let pending = phase.cooperative_load(tile, &src);
                let (ready, mut phase) = pending.sync_tile();
                phase.store_ready_to_storage(&ready, &dst);
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    let lowered = ir.lower_to_naga().unwrap();
    assert_eq!(lowered.module().entry_points.len(), 1);
    assert_eq!(lowered.module().global_variables.iter().count(), 3);
    assert_eq!(lowered.module().entry_points[0].workgroup_size, [16, 16, 1]);
    assert_eq!(lowered.module().entry_points[0].function.arguments.len(), 2);
}

#[test]
fn lowers_to_naga_module_with_loop_and_barriers() {
    let ir = build(|mut phase| {
        let src = phase.storage_tensor::<F32>(Shape::new([32]));
        let dst = phase.storage_tensor::<F32>(Shape::new([32]));
        phase.range_step(
            |mut phase, _| {
                let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
                let pending = phase.cooperative_load(tile, &src);
                let (ready, mut phase) = pending.sync_tile();
                phase.store_ready_to_storage(&ready, &dst);
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    let lowered = ir.lower_to_naga().unwrap();
    let entry = &lowered.module().entry_points[0];
    assert_eq!(entry.function.body.len(), 2);
}

#[test]
fn lowers_partition_views_as_aliases_of_source_tiles() {
    let ir = build(|mut phase| {
        let src = phase.storage_tensor::<F32>(Shape::new([32]));
        let dst = phase.storage_tensor::<F32>(Shape::new([16]));
        phase.range_step(
            |mut phase, _| {
                let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
                let pending = phase.cooperative_load(tile, &src);
                let (ready, mut phase) = pending.sync_tile();
                phase.partition(
                    &ready,
                    TileLevel::Subgroup,
                    Shape::new([16]),
                    |phase, child| {
                        phase.store_ready_to_storage(&child, &dst);
                    },
                );
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });

    assert_eq!(ir.tiles().len(), 2);
    assert_eq!(
        ir.tiles()[1].origin,
        TileOrigin::View {
            source: TileRef::new(TileId(0), ElementType::F32),
            mapping: ViewMapping::Partition {
                level: TileLevel::Subgroup,
                origin: [0, 0],
            },
        },
    );

    let lowered = ir.lower_to_naga().unwrap();
    assert_eq!(lowered.module().global_variables.iter().count(), 3);
}

#[test]
fn lowers_gemm_to_naga_module() {
    let ir = build(|mut phase| {
        let a_in = phase.storage_tensor::<F32>(Shape::new([1, 2]));
        let b_in = phase.storage_tensor::<F32>(Shape::new([2, 1]));
        let c_out = phase.storage_tensor::<F32>(Shape::new([1, 1]));
        let mut acc = phase.alloc_fragment::<F32>(Shape::new([1, 1]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;

        phase.range_step(
            |mut phase, _| {
                let a = phase.alloc_workgroup_tile::<F32>(Shape::new([1, 2]));
                let b = phase.alloc_workgroup_tile::<F32>(Shape::new([2, 1]));
                let pending = phase.cooperative_load_pair(a, &a_in, b, &b_in);
                let (a, b, mut phase) = pending.sync_tiles();

                kernels::gemm::tiled(
                    &mut phase,
                    &a,
                    &b,
                    &mut acc,
                    kernels::gemm::GemmTilePlan::portable(1, 1, 2),
                );
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
        )
    });

    let lowered = ir.lower_to_naga().unwrap();
    let entry = &lowered.module().entry_points[0];
    assert_eq!(lowered.module().global_variables.iter().count(), 3);
    assert_eq!(entry.function.local_variables.iter().count(), 15);
}

#[test]
fn userland_gemm_triggers_cooperative_fast_path() {
    let ir = build(|mut phase| {
        let a_in = phase.storage_tensor::<F32>(Shape::new([16, 8]));
        let b_in = phase.storage_tensor::<F32>(Shape::new([8, 16]));
        let c_out = phase.storage_tensor::<F32>(Shape::new([16, 16]));
        let mut acc = phase.alloc_fragment::<F32>(Shape::new([16, 16]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;
        phase.range_step(
            |mut phase, _| {
                let a = phase.alloc_workgroup_tile::<F32>(Shape::new([16, 8]));
                let b = phase.alloc_workgroup_tile::<F32>(Shape::new([8, 16]));
                let pending = phase.cooperative_load_pair(a, &a_in, b, &b_in);
                let (a, b, mut phase) = pending.sync_tiles();
                kernels::gemm::tiled(
                    &mut phase,
                    &a,
                    &b,
                    &mut acc,
                    kernels::gemm::GemmTilePlan::portable(16, 16, 8),
                );
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
        )
    });

    let lowered = ir.lower_to_naga().unwrap();
    assert!(
        lowered
            .module()
            .types
            .iter()
            .any(|(_, ty)| matches!(ty.inner, naga::TypeInner::CooperativeMatrix { .. })),
        "expected primitive userland GEMM to be recognized by the cooperative fast path"
    );
}

#[test]
fn layout_is_structured_shape_strides_and_memory_level() {
    let shape = Shape::new([4, 8]);
    let row_major = Layout::contiguous(MemoryLevel::Workgroup, shape.clone());
    let col_major = Layout::strided(
        MemoryLevel::Workgroup,
        shape,
        Strides::col_major_for(&Shape::new([4, 8])),
    );

    assert_eq!(row_major.memory_level(), MemoryLevel::Workgroup);
    assert_eq!(row_major.shape().dims()[0].get(), 4);
    assert_eq!(row_major.strides().values(), &[8, 1]);
    assert!(row_major.is_row_major());
    assert!(!row_major.is_col_major());
    assert_eq!(row_major.element_count().get(), 32);
    assert_eq!(row_major.allocation_element_count().get(), 32);

    assert_eq!(col_major.strides().values(), &[1, 4]);
    assert!(col_major.is_col_major());
    assert!(!col_major.is_row_major());

    let padded = Layout::strided(
        MemoryLevel::Workgroup,
        Shape::new([4, 8]),
        Strides::new([12, 1]),
    );
    assert_eq!(padded.element_count().get(), 32);
    assert_eq!(padded.allocation_element_count().get(), 44);
}

#[test]
#[should_panic(expected = "gemm K dimensions must match")]
fn userland_gemm_rejects_shape_mismatch() {
    build(|mut phase| {
        let a_in = phase.storage_tensor::<F32>(Shape::new([2, 3]));
        let b_in = phase.storage_tensor::<F32>(Shape::new([2, 2]));
        let c_out = phase.storage_tensor::<F32>(Shape::new([2, 2]));
        let mut acc = phase.alloc_fragment::<F32>(Shape::new([2, 2]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;
        phase.range_step(
            |mut phase, _| {
                let a = phase.alloc_workgroup_tile::<F32>(Shape::new([2, 3]));
                let b = phase.alloc_workgroup_tile::<F32>(Shape::new([2, 2]));
                let pending = phase.cooperative_load_pair(a, &a_in, b, &b_in);
                let (a, b, mut phase) = pending.sync_tiles();
                kernels::gemm::tiled(
                    &mut phase,
                    &a,
                    &b,
                    &mut acc,
                    kernels::gemm::GemmTilePlan::portable(2, 2, 3),
                );
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
        )
    });
}

#[test]
#[should_panic(expected = "M must divide subgroup_m")]
fn userland_gemm_rejects_non_divisible_tile_plan() {
    build(|mut phase| {
        let a_in = phase.storage_tensor::<F32>(Shape::new([4, 4]));
        let b_in = phase.storage_tensor::<F32>(Shape::new([4, 4]));
        let c_out = phase.storage_tensor::<F32>(Shape::new([4, 4]));
        let mut acc = phase.alloc_fragment::<F32>(Shape::new([4, 4]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;
        phase.range_step(
            |mut phase, _| {
                let a = phase.alloc_workgroup_tile::<F32>(Shape::new([4, 4]));
                let b = phase.alloc_workgroup_tile::<F32>(Shape::new([4, 4]));
                let pending = phase.cooperative_load_pair(a, &a_in, b, &b_in);
                let (a, b, mut phase) = pending.sync_tiles();
                kernels::gemm::tiled(
                    &mut phase,
                    &a,
                    &b,
                    &mut acc,
                    kernels::gemm::GemmTilePlan {
                        subgroup_m: 3,
                        subgroup_n: 4,
                        subgroup_k: 4,
                        thread_m: 1,
                        thread_n: 1,
                        thread_k: 4,
                    },
                );
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
        )
    });
}

#[test]
#[should_panic(expected = "partition view must stay within parent tile shape")]
fn explicit_partition_rejects_out_of_bounds_origin() {
    build(|mut phase| {
        let src = phase.storage_tensor::<F32>(Shape::new([4, 4]));
        let dst = phase.storage_tensor::<F32>(Shape::new([2, 2]));
        phase.range_step(
            |mut phase, _| {
                let tile = phase.alloc_workgroup_tile::<F32>(Shape::new([4, 4]));
                let pending = phase.cooperative_load(tile, &src);
                let (ready, mut phase) = pending.sync_tile();
                phase.partition_at(
                    &ready,
                    TileLevel::Subgroup,
                    Shape::new([2, 2]),
                    [3, 0],
                    |phase, child| {
                        phase.store_ready_to_storage(&child, &dst);
                    },
                );
                phase.sync_end()
            },
            |phase| phase.finish(),
        )
    });
}

#[test]
#[should_panic(expected = "nested dynamic storage views are not supported")]
fn nested_dynamic_storage_views_are_rejected() {
    build(|mut phase| {
        let full = phase.storage_tensor::<F32>(Shape::new([64, 64]));
        let row_tile = full.workgroup_tile_2d(
            Shape::new([16, 64]),
            Some(WorkgroupOffset::new(WorkgroupAxis::Y, 16)),
            None,
        );
        let _k_tile = row_tile.dynamic_tile_2d(
            Shape::new([16, 8]),
            None,
            Some(DynamicOffset::Loop(LoopOffset::new(8))),
        );
        phase.finish()
    });
}

#[test]
#[should_panic(expected = "gemv partials must contain 128 elements per row")]
fn gemv_builder_rejects_wrong_scratch_tile_size() {
    build(|mut phase| {
        let a = phase.storage_tensor_read::<F32>(Shape::new([4, 8]));
        let x = phase.storage_tensor_read::<F32>(Shape::new([8, 1]));
        let y = phase.storage_tensor::<F32>(Shape::new([4, 1]));
        let partials = phase.alloc_workgroup_tile::<F32>(Shape::new([32]));
        phase.gemv_tiled(&a, &x, &y, partials, 1, 1);
        phase.finish()
    });
}
