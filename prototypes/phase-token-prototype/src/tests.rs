use super::*;

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
            kind: LoopKind::RangeStep { induction: Dim(0) },
            body: Block::from_ops(vec![
                Op::CooperativeLoad(CooperativeLoadOp {
                    dst: tile,
                    src: StorageView {
                        buffer: src,
                        offset: 0,
                    },
                    level: TileLevel::Workgroup,
                }),
                Op::Barrier(BarrierOp {
                    scope: BarrierScope::Workgroup,
                }),
                Op::StoreTile(StoreTileOp {
                    src: tile,
                    dst: StorageView {
                        buffer: dst,
                        offset: 0,
                    },
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
                src: StorageView {
                    buffer: src,
                    offset: 0,
                },
                level: TileLevel::Workgroup,
            }),
            Op::Barrier(BarrierOp {
                scope: BarrierScope::Workgroup,
            }),
            Op::Loop(LoopOp {
                kind: LoopKind::RangeStep { induction: Dim(0) },
                body: Block::from_ops(vec![
                    Op::StoreTile(StoreTileOp {
                        src: tile,
                        dst: StorageView {
                            buffer: dst,
                            offset: 0,
                        },
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

                phase.gemm(&a, &b, &mut acc);
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
    assert_eq!(
        ir.tiles(),
        &[
            TileDecl {
                id: TileId(0),
                element: ElementType::F32,
                layout: Layout::contiguous(MemoryLevel::Private, Shape::new([bm, bn])),
                level: TileLevel::Thread,
                origin: TileOrigin::Allocation,
            },
            TileDecl {
                id: TileId(1),
                element: ElementType::F32,
                layout: Layout::contiguous(MemoryLevel::Workgroup, Shape::new([bm, bk])),
                level: TileLevel::Workgroup,
                origin: TileOrigin::Allocation,
            },
            TileDecl {
                id: TileId(2),
                element: ElementType::F32,
                layout: Layout::contiguous(MemoryLevel::Workgroup, Shape::new([bk, bn])),
                level: TileLevel::Workgroup,
                origin: TileOrigin::Allocation,
            },
        ],
    );
    assert_eq!(
        ir.body(),
        &Block::from_ops(vec![
            Op::FillTile(FillTileOp {
                dst: acc,
                value: FillValue::Zero,
            }),
            Op::Loop(LoopOp {
                kind: LoopKind::RangeStep { induction: Dim(0) },
                body: Block::from_ops(vec![
                    Op::CooperativeLoad(CooperativeLoadOp {
                        dst: a,
                        src: StorageView {
                            buffer: a_buf,
                            offset: 0,
                        },
                        level: TileLevel::Workgroup,
                    }),
                    Op::CooperativeLoad(CooperativeLoadOp {
                        dst: b,
                        src: StorageView {
                            buffer: b_buf,
                            offset: 0,
                        },
                        level: TileLevel::Workgroup,
                    }),
                    Op::Barrier(BarrierOp {
                        scope: BarrierScope::Workgroup,
                    }),
                    Op::Gemm(GemmOp {
                        a,
                        b,
                        acc,
                        tiling: GemmTiling::portable(bm, bn, bk),
                        backend: MmaBackend::FmaPortable,
                    }),
                    Op::Barrier(BarrierOp {
                        scope: BarrierScope::Workgroup,
                    }),
                ]),
            }),
            Op::StoreTile(StoreTileOp {
                src: acc,
                dst: StorageView {
                    buffer: c_buf,
                    offset: 0,
                },
            }),
        ]),
    );
}

#[test]
fn gemm_expands_to_partitioned_mma_ir() {
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

                phase.gemm(&a, &b, &mut acc);
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
        )
    });

    let expanded = ir.expand_gemm_to_mma();
    assert_eq!(expanded.tiles().len(), 54);
    assert_eq!(
        expanded.tiles()[5].origin,
        TileOrigin::View {
            source: TileRef::new(TileId(0), ElementType::F32),
            mapping: ViewMapping::Partition {
                level: TileLevel::Subgroup,
                origin: [0, 0],
            },
        },
    );
    assert_eq!(
        expanded.tiles()[8].origin,
        TileOrigin::View {
            source: TileRef::new(TileId(5), ElementType::F32),
            mapping: ViewMapping::Partition {
                level: TileLevel::Thread,
                origin: [0, 0],
            },
        },
    );
    assert_eq!(expanded.tiles()[3].layout.shape(), &Shape::new([16, 8]));
    assert_eq!(expanded.tiles()[4].layout.shape(), &Shape::new([8, 16]));
    assert_eq!(expanded.tiles()[5].layout.shape(), &Shape::new([16, 16]));
    assert_eq!(expanded.tiles()[6].layout.shape(), &Shape::new([4, 8]));
    assert_eq!(expanded.tiles()[7].layout.shape(), &Shape::new([8, 4]));
    assert_eq!(expanded.tiles()[8].layout.shape(), &Shape::new([4, 4]));

    let Op::Loop(loop_op) = &expanded.body().ops()[1] else {
        panic!("expected loop after accumulator fill");
    };
    let Op::Block(gemm_block) = &loop_op.body.ops()[3] else {
        panic!("expected gemm to expand into a block of thread mmas");
    };
    assert_eq!(gemm_block.body.ops().len(), 16);

    let Op::Mma(mma) = &gemm_block.body.ops()[0] else {
        panic!("expected expanded gemm block to contain mma");
    };
    assert_eq!(mma.a, TileRef::new(TileId(6), ElementType::F32));
    assert_eq!(mma.b, TileRef::new(TileId(7), ElementType::F32));
    assert_eq!(mma.acc, TileRef::new(TileId(8), ElementType::F32));

    let Op::Mma(last_mma) = &gemm_block.body.ops()[15] else {
        panic!("expected expanded gemm block to contain mma");
    };
    assert_eq!(last_mma.a, TileRef::new(TileId(51), ElementType::F32));
    assert_eq!(last_mma.b, TileRef::new(TileId(52), ElementType::F32));
    assert_eq!(last_mma.acc, TileRef::new(TileId(53), ElementType::F32));
    assert_eq!(
        expanded.tiles()[53].origin,
        TileOrigin::View {
            source: TileRef::new(TileId(5), ElementType::F32),
            mapping: ViewMapping::Partition {
                level: TileLevel::Thread,
                origin: [12, 12],
            },
        },
    );
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
    assert_eq!(lowered.module().entry_points[0].workgroup_size, [256, 1, 1]);
    assert_eq!(lowered.module().entry_points[0].function.arguments.len(), 1);
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

                phase.gemm(&a, &b, &mut acc);
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
    assert_eq!(lowered.module().global_variables.iter().count(), 5);
    assert_eq!(entry.function.local_variables.iter().count(), 7);
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

    assert_eq!(col_major.strides().values(), &[1, 4]);
    assert!(col_major.is_col_major());
    assert!(!col_major.is_row_major());
}
