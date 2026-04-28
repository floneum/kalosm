use std::fmt;
use std::marker::PhantomData;

use crate::{
    BarrierOp, BarrierScope, Block, BufferAccess, BufferDecl, BufferRef, CooperativeLoadOp, Dim,
    ElementType, FillTileOp, FillValue, GemmOp, GemmTiling, KernelIr, Layout, LoopKind, LoopOp,
    MemoryLevel, MmaBackend, MmaOp, Op, PartitionBinding, PartitionOp, Shape, StorageView,
    StoreTileOp, TileDecl, TileLevel, TileOrigin, TileRef, ViewMapping,
};

/// A sample numeric marker.
#[derive(Copy, Clone, Debug)]
pub struct F32;

/// Numeric element markers that can appear in the typed IR.
pub trait Numeric {
    const ELEMENT: ElementType;
}

impl Numeric for F32 {
    const ELEMENT: ElementType = ElementType::F32;
}

/// Build a kernel IR with a generative kernel lifetime and entry phase.
pub fn build(
    f: impl for<'k, 'entry, 'cx> FnOnce(Phase<'cx, 'k, 'entry>) -> KernelDone,
) -> KernelIr {
    let mut ir = KernelIr::default();
    let mut cx = KernelBuilder {
        ir: &mut ir,
        blocks: vec![Block::new()],
        _kernel: PhantomData,
    };

    let phase = Phase {
        cx: &mut cx,
        state: Clean,
        _phase: PhantomData,
    };
    let KernelDone(()) = f(phase);
    cx.finish_ir();
    ir
}

/// The IR builder. Users never receive this directly; they operate through
/// phase-scoped [`Phase`] handles.
pub struct KernelBuilder<'k> {
    pub(crate) ir: &'k mut KernelIr,
    pub(crate) blocks: Vec<Block>,
    pub(crate) _kernel: PhantomData<&'k mut ()>,
}

impl<'k> KernelBuilder<'k> {
    fn alloc_buffer<T: Numeric>(&mut self, layout: Layout, access: BufferAccess) -> BufferRef {
        assert_eq!(
            layout.memory_level(),
            MemoryLevel::Storage,
            "storage buffers must use MemoryLevel::Storage"
        );
        let id = crate::BufferId(self.ir.next_buffer);
        self.ir.next_buffer += 1;
        let buffer = BufferRef::new(id, T::ELEMENT);
        self.ir.buffers.push(BufferDecl {
            id,
            element: T::ELEMENT,
            layout,
            access,
        });
        buffer
    }

    fn alloc_tile<T: Numeric>(&mut self, layout: Layout, level: TileLevel) -> TileRef {
        let id = crate::TileId(self.ir.next_tile);
        self.ir.next_tile += 1;
        let tile = TileRef::new(id, T::ELEMENT);
        self.ir.tiles.push(TileDecl {
            id,
            element: T::ELEMENT,
            layout,
            level,
            origin: TileOrigin::Allocation,
        });
        tile
    }

    fn alloc_tile_view<T: Numeric>(
        &mut self,
        source: TileRef,
        layout: Layout,
        level: TileLevel,
    ) -> TileRef {
        let id = crate::TileId(self.ir.next_tile);
        self.ir.next_tile += 1;
        let tile = TileRef::new(id, T::ELEMENT);
        self.ir.tiles.push(TileDecl {
            id,
            element: T::ELEMENT,
            layout,
            level,
            origin: TileOrigin::View {
                source,
                mapping: ViewMapping::Partition {
                    level,
                    origin: [0, 0],
                },
            },
        });
        tile
    }

    fn tile_layout(&self, tile: TileRef) -> &Layout {
        let decl = self
            .ir
            .tiles()
            .get(tile.id.index())
            .expect("tile reference must point at a declared tile");
        assert_eq!(
            decl.element, tile.element,
            "tile reference element must match its declaration"
        );
        &decl.layout
    }

    fn push_op(&mut self, op: Op) {
        self.blocks
            .last_mut()
            .expect("builder always has a current block")
            .push(op);
    }

    fn begin_block(&mut self) {
        self.blocks.push(Block::new());
    }

    fn end_block(&mut self) -> Block {
        assert!(self.blocks.len() > 1, "cannot pop the root block");
        self.blocks.pop().expect("checked block stack length")
    }

    fn finish_ir(&mut self) {
        assert_eq!(self.blocks.len(), 1, "unclosed IR block");
        self.ir.body = self.blocks.pop().expect("checked block stack length");
    }
}

/// A phase-scoped builder handle.
///
/// `State` is either [`Clean`] or [`Pending`]. Only a clean phase can finish a
/// loop body. Creating a pending cooperative load consumes the clean phase and
/// returns a pending phase; synchronizing it consumes the pending phase and
/// returns a clean phase again.
pub struct Phase<'cx, 'k, 'flow, State = Clean> {
    cx: &'cx mut KernelBuilder<'k>,
    state: State,
    _phase: PhantomData<fn(&'flow ()) -> &'flow ()>,
}

/// A phase with no unsynchronized workgroup writes.
pub struct Clean;

/// A phase with one unsynchronized cooperative load.
pub struct Pending<T> {
    tile: TileRef,
    _ty: PhantomData<T>,
}

/// A phase with two unsynchronized cooperative loads.
pub struct Pending2<A, B> {
    first: TileRef,
    second: TileRef,
    _ty: PhantomData<(A, B)>,
}

impl<'cx, 'k, 'flow> Phase<'cx, 'k, 'flow, Clean> {
    /// Declare a storage buffer tensor bound to this kernel.
    pub fn storage_tensor<T: Numeric>(&mut self, shape: Shape) -> StorageTensor<'k, T> {
        let buffer = self.cx.alloc_buffer::<T>(
            Layout::contiguous(MemoryLevel::Storage, shape),
            BufferAccess::ReadWrite,
        );
        StorageTensor {
            buffer,
            _ty: PhantomData,
            _kernel: PhantomData,
        }
    }

    /// Allocate a workgroup tile whose contents are not yet initialized.
    pub fn alloc_workgroup<T: Numeric>(&mut self) -> UninitTile<'k, T> {
        self.alloc_workgroup_tile(Shape::tile())
    }

    /// Allocate a shaped workgroup tile whose contents are not yet initialized.
    pub fn alloc_workgroup_tile<T: Numeric>(&mut self, shape: Shape) -> UninitTile<'k, T> {
        self.alloc_tile_with_layout(Layout::contiguous(MemoryLevel::Workgroup, shape))
    }

    /// Allocate a tile with an explicit layout.
    pub fn alloc_tile_with_layout<T: Numeric>(&mut self, layout: Layout) -> UninitTile<'k, T> {
        assert_eq!(
            layout.memory_level(),
            MemoryLevel::Workgroup,
            "this prototype only lowers workgroup tile allocations"
        );
        let tile = self.cx.alloc_tile::<T>(layout, TileLevel::Workgroup);
        UninitTile {
            tile,
            _ty: PhantomData,
            _kernel: PhantomData,
        }
    }

    /// Allocate a private/register-resident tile.
    pub fn alloc_private_tile<T: Numeric>(&mut self, shape: Shape) -> RegTile<'k, T> {
        self.alloc_thread_tile(shape)
    }

    /// Allocate a private/register-resident tile with an explicit layout.
    pub fn alloc_private_tile_with_layout<T: Numeric>(&mut self, layout: Layout) -> RegTile<'k, T> {
        self.alloc_thread_tile_with_layout(layout)
    }

    /// Allocate a thread-owned register tile.
    pub fn alloc_thread_tile<T: Numeric>(&mut self, shape: Shape) -> RegTile<'k, T> {
        self.alloc_thread_tile_with_layout(Layout::contiguous(MemoryLevel::Private, shape))
    }

    /// Allocate an accumulator fragment.
    pub fn alloc_fragment<T: Numeric>(&mut self, shape: Shape) -> RegTile<'k, T> {
        self.alloc_thread_tile(shape)
    }

    /// Allocate a thread-owned register tile with an explicit layout.
    pub fn alloc_thread_tile_with_layout<T: Numeric>(&mut self, layout: Layout) -> RegTile<'k, T> {
        assert_eq!(
            layout.memory_level(),
            MemoryLevel::Private,
            "thread tile allocations must use MemoryLevel::Private"
        );
        let tile = self.cx.alloc_tile::<T>(layout, TileLevel::Thread);
        RegTile {
            tile,
            _ty: PhantomData,
            _kernel: PhantomData,
        }
    }

    /// Fill a private/register tile with zero.
    pub fn fill_zero<T>(&mut self, dst: &mut RegTile<'k, T>) {
        self.cx.push_op(Op::FillTile(FillTileOp {
            dst: dst.tile,
            value: FillValue::Zero,
        }));
    }

    /// Emit a cooperative load into an uninitialized tile.
    ///
    /// This consumes the clean phase and returns a pending phase. The pending
    /// phase has no `finish`, `range_step`, or `sync_end` methods, so user code
    /// must synchronize it before it can finish the control-flow body.
    pub fn cooperative_load<T: Numeric>(
        self,
        dst: UninitTile<'k, T>,
        src: &StorageTensor<'k, T>,
    ) -> Phase<'cx, 'k, 'flow, Pending<T>> {
        self.cx.push_op(Op::CooperativeLoad(CooperativeLoadOp {
            dst: dst.tile,
            src: src.view(),
            level: TileLevel::Workgroup,
        }));
        Phase {
            cx: self.cx,
            state: Pending {
                tile: dst.tile,
                _ty: PhantomData,
            },
            _phase: PhantomData,
        }
    }

    /// Emit two cooperative loads that are synchronized together.
    pub fn cooperative_load_pair<A: Numeric, B: Numeric>(
        self,
        first: UninitTile<'k, A>,
        first_src: &StorageTensor<'k, A>,
        second: UninitTile<'k, B>,
        second_src: &StorageTensor<'k, B>,
    ) -> Phase<'cx, 'k, 'flow, Pending2<A, B>> {
        self.cx.push_op(Op::CooperativeLoad(CooperativeLoadOp {
            dst: first.tile,
            src: first_src.view(),
            level: TileLevel::Workgroup,
        }));
        self.cx.push_op(Op::CooperativeLoad(CooperativeLoadOp {
            dst: second.tile,
            src: second_src.view(),
            level: TileLevel::Workgroup,
        }));
        Phase {
            cx: self.cx,
            state: Pending2 {
                first: first.tile,
                second: second.tile,
                _ty: PhantomData,
            },
            _phase: PhantomData,
        }
    }

    /// Emit an end-of-phase barrier and return the `Synced` witness required by
    /// loop bodies.
    ///
    /// This consumes the phase handle, which makes the barrier structurally the
    /// last IR-emitting operation available in the body.
    pub fn sync_end(self) -> Synced<'flow> {
        self.cx.push_op(Op::Barrier(BarrierOp {
            scope: BarrierScope::Workgroup,
        }));
        Synced {
            _phase: PhantomData,
        }
    }

    /// Store a ready workgroup tile to a storage buffer.
    pub fn store_ready_to_storage<'ready, T: Numeric>(
        &mut self,
        src: &ReadyTile<'k, 'ready, T>,
        dst: &StorageTensor<'k, T>,
    ) {
        self.cx.push_op(Op::StoreTile(StoreTileOp {
            src: src.tile,
            dst: dst.view(),
        }));
    }

    /// Store a private/register tile to a storage buffer.
    pub fn store_fragment_to_storage<T: Numeric>(
        &mut self,
        src: &RegTile<'k, T>,
        dst: &StorageTensor<'k, T>,
    ) {
        self.cx.push_op(Op::StoreTile(StoreTileOp {
            src: src.tile,
            dst: dst.view(),
        }));
    }

    /// Emit a high-level tiled GEMM over ready parent tiles.
    pub fn gemm<'ready, TA, TB, TC>(
        &mut self,
        a: &ReadyTile<'k, 'ready, TA>,
        b: &ReadyTile<'k, 'ready, TB>,
        acc: &mut RegTile<'k, TC>,
    ) {
        let [m, k] = matrix_shape(self.cx.tile_layout(a.tile));
        let [k_b, n] = matrix_shape(self.cx.tile_layout(b.tile));
        let [m_acc, n_acc] = matrix_shape(self.cx.tile_layout(acc.tile));
        assert_eq!(k, k_b, "gemm K dimensions must match");
        assert_eq!(m, m_acc, "gemm M dimension must match accumulator");
        assert_eq!(n, n_acc, "gemm N dimension must match accumulator");

        self.cx.push_op(Op::Gemm(GemmOp {
            a: a.tile,
            b: b.tile,
            acc: acc.tile,
            tiling: GemmTiling::portable(m, n, k),
            backend: MmaBackend::FmaPortable,
        }));
    }

    /// Emit a tile-level matrix multiply-accumulate.
    pub fn mma<'ready, TA, TB, TC>(
        &mut self,
        a: &ReadyTile<'k, 'ready, TA>,
        b: &ReadyTile<'k, 'ready, TB>,
        acc: &mut RegTile<'k, TC>,
    ) {
        self.cx.push_op(Op::Mma(MmaOp {
            a: a.tile,
            b: b.tile,
            acc: acc.tile,
            level: TileLevel::Thread,
            backend: MmaBackend::FmaPortable,
        }));
    }

    /// Partition a ready tile into a lower-level tile view.
    ///
    /// The closure is executed once while building the IR. The resulting IR is
    /// a structured [`PartitionOp`] whose nested body carries the semantic
    /// level, much like Triton/TileLang block programs describe tile lanes
    /// without spelling a physical per-thread loop in user code.
    pub fn partition<T: Numeric>(
        &mut self,
        tile: &ReadyTile<'k, '_, T>,
        level: TileLevel,
        shape: Shape,
        body: impl for<'part> FnOnce(&mut Self, ReadyTile<'k, 'part, T>),
    ) {
        let source_layout = self.cx.tile_layout(tile.tile);
        let view_layout = Layout::strided(
            source_layout.memory_level(),
            shape,
            source_layout.strides().clone(),
        );
        let view = self.cx.alloc_tile_view::<T>(tile.tile, view_layout, level);
        let ready = ReadyTile {
            tile: view,
            _ty: PhantomData,
            _kernel: PhantomData,
            _phase: PhantomData,
        };

        self.cx.begin_block();
        body(self, ready);
        let partition_body = self.cx.end_block();
        self.cx.push_op(Op::Partition(PartitionOp {
            bindings: vec![PartitionBinding {
                source: tile.tile,
                view,
            }],
            level,
            body: partition_body,
        }));
    }

    /// Build a symbolic stepped loop.
    ///
    /// The loop body is generic over an iteration phase lifetime. It receives a
    /// phase handle and must return `Synced<'iter>`, not the handle itself, so
    /// the body must end by consuming its handle with a sync method. The
    /// continuation after the loop gets a fresh phase handle, which prevents
    /// values branded by the iteration phase from escaping.
    pub fn range_step<R>(
        self,
        body: impl for<'iter, 'body> FnOnce(Phase<'body, 'k, 'iter, Clean>, Dim) -> Synced<'iter>,
        after: impl for<'after, 'after_body> FnOnce(Phase<'after_body, 'k, 'after, Clean>) -> R,
    ) -> R {
        let cx = self.cx;
        cx.begin_block();

        let iter_phase = Phase {
            cx,
            state: Clean,
            _phase: PhantomData,
        };
        let synced = body(iter_phase, Dim(0));
        drop(synced);

        let body = cx.end_block();
        cx.push_op(Op::Loop(LoopOp {
            kind: LoopKind::RangeStep { induction: Dim(0) },
            body,
        }));

        let after_phase = Phase {
            cx,
            state: Clean,
            _phase: PhantomData,
        };
        after(after_phase)
    }

    /// Consume the final phase handle and finish kernel construction.
    pub fn finish(self) -> KernelDone {
        KernelDone(())
    }
}

fn matrix_shape(layout: &Layout) -> [u32; 2] {
    assert_eq!(layout.shape().rank(), 2, "gemm operands must be rank-2");
    [
        layout.shape().dims()[0].get(),
        layout.shape().dims()[1].get(),
    ]
}

impl<'cx, 'k, 'flow, T> Phase<'cx, 'k, 'flow, Pending<T>> {
    /// Emit a barrier, consume the pending load, and return a ready tile plus a
    /// clean phase handle.
    pub fn sync_tile(self) -> (ReadyTile<'k, 'flow, T>, Phase<'cx, 'k, 'flow, Clean>) {
        self.cx.push_op(Op::Barrier(BarrierOp {
            scope: BarrierScope::Workgroup,
        }));
        let ready = ReadyTile {
            tile: self.state.tile,
            _ty: PhantomData,
            _kernel: PhantomData,
            _phase: PhantomData,
        };
        let phase = Phase {
            cx: self.cx,
            state: Clean,
            _phase: PhantomData,
        };
        (ready, phase)
    }
}

impl<'cx, 'k, 'flow, A, B> Phase<'cx, 'k, 'flow, Pending2<A, B>> {
    /// Emit a barrier, consume two pending loads, and return ready tiles plus a
    /// clean phase handle.
    pub fn sync_tiles(
        self,
    ) -> (
        ReadyTile<'k, 'flow, A>,
        ReadyTile<'k, 'flow, B>,
        Phase<'cx, 'k, 'flow, Clean>,
    ) {
        self.cx.push_op(Op::Barrier(BarrierOp {
            scope: BarrierScope::Workgroup,
        }));
        let first = ReadyTile {
            tile: self.state.first,
            _ty: PhantomData,
            _kernel: PhantomData,
            _phase: PhantomData,
        };
        let second = ReadyTile {
            tile: self.state.second,
            _ty: PhantomData,
            _kernel: PhantomData,
            _phase: PhantomData,
        };
        let phase = Phase {
            cx: self.cx,
            state: Clean,
            _phase: PhantomData,
        };
        (first, second, phase)
    }
}

/// A barrier witness. This is intentionally not constructible outside the crate.
pub struct Synced<'flow> {
    _phase: PhantomData<fn(&'flow ()) -> &'flow ()>,
}

/// A marker returned by [`Phase::finish`].
pub struct KernelDone(());

/// Workgroup memory with undefined contents.
pub struct UninitTile<'k, T> {
    pub(crate) tile: TileRef,
    _ty: PhantomData<T>,
    _kernel: PhantomData<&'k mut ()>,
}

/// Workgroup memory that has been cooperatively written but not synchronized.
pub type PendingTile<'cx, 'k, 'flow, T> = Phase<'cx, 'k, 'flow, Pending<T>>;

/// Two pending cooperative loads that must be synchronized together.
pub type PendingTilePair<'cx, 'k, 'flow, A, B> = Phase<'cx, 'k, 'flow, Pending2<A, B>>;

/// A private/register-resident tile.
#[derive(Copy, Clone)]
pub struct RegTile<'k, T> {
    pub(crate) tile: TileRef,
    _ty: PhantomData<T>,
    _kernel: PhantomData<&'k mut ()>,
}

/// A storage buffer tensor bound by the kernel.
pub struct StorageTensor<'k, T> {
    pub(crate) buffer: BufferRef,
    _ty: PhantomData<T>,
    _kernel: PhantomData<&'k mut ()>,
}

impl<T> StorageTensor<'_, T> {
    fn view(&self) -> StorageView {
        StorageView {
            buffer: self.buffer,
            offset: 0,
        }
    }
}

/// Workgroup memory that can be read after its producing barrier.
///
/// This is intentionally not `Copy`; future reload APIs can consume it to
/// invalidate stale ready views.
pub struct ReadyTile<'k, 'flow, T> {
    pub(crate) tile: TileRef,
    _ty: PhantomData<T>,
    _kernel: PhantomData<&'k ()>,
    _phase: PhantomData<&'flow ()>,
}

impl<T> fmt::Debug for ReadyTile<'_, '_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadyTile")
            .field("tile", &self.tile)
            .finish_non_exhaustive()
    }
}
