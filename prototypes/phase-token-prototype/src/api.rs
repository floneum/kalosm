use std::fmt;
use std::marker::PhantomData;

use crate::{
    BarrierOp, BarrierScope, Block, BufferAccess, BufferDecl, BufferRef, CooperativeLoadOp, Dim,
    DynamicOffset, ElementType, FillTileOp, FillValue, GemvOp, KernelIr, Layout, LoopKind, LoopOp,
    MemoryLevel, MmaBackend, MmaOp, Op, PartitionBinding, PartitionOp, Shape, StorageView,
    StoreTileOp, TileDecl, TileLevel, TileOrigin, TileRef, ViewMapping, WorkgroupOffset,
};

const GEMV_WORKGROUP_INVOCATIONS: u32 = 128;

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
        self.alloc_tile_view_at::<T>(source, layout, level, [0, 0])
    }

    fn alloc_tile_view_at<T: Numeric>(
        &mut self,
        source: TileRef,
        layout: Layout,
        level: TileLevel,
        origin: [u32; 2],
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
                mapping: ViewMapping::Partition { level, origin },
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
        self.storage_tensor_with_access::<T>(shape, BufferAccess::ReadWrite)
    }

    /// Declare a read-only storage buffer tensor bound to this kernel.
    pub fn storage_tensor_read<T: Numeric>(&mut self, shape: Shape) -> StorageTensor<'k, T> {
        self.storage_tensor_with_access::<T>(shape, BufferAccess::Read)
    }

    /// Declare a read-only storage buffer tensor with an explicit layout.
    pub fn storage_tensor_read_with_layout<T: Numeric>(
        &mut self,
        layout: Layout,
    ) -> StorageTensor<'k, T> {
        assert_eq!(
            layout.memory_level(),
            MemoryLevel::Storage,
            "storage tensors must use MemoryLevel::Storage"
        );
        let buffer = self
            .cx
            .alloc_buffer::<T>(layout.clone(), BufferAccess::Read);
        StorageTensor {
            buffer,
            view: StorageView::root(buffer, layout),
            _ty: PhantomData,
            _kernel: PhantomData,
        }
    }

    fn storage_tensor_with_access<T: Numeric>(
        &mut self,
        shape: Shape,
        access: BufferAccess,
    ) -> StorageTensor<'k, T> {
        let layout = Layout::contiguous(MemoryLevel::Storage, shape);
        let buffer = self.cx.alloc_buffer::<T>(layout.clone(), access);
        StorageTensor {
            buffer,
            view: StorageView::root(buffer, layout),
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

    /// Emit a row-parallel GEMV.
    ///
    /// This is not modeled as a skinny GEMM: one workgroup owns one output row,
    /// and the workgroup lanes cooperatively reduce the K dimension through the
    /// supplied scratch tile.
    pub fn gemv<T: Numeric>(
        &mut self,
        a: &StorageTensor<'k, T>,
        x: &StorageTensor<'k, T>,
        y: &StorageTensor<'k, T>,
        partials: UninitTile<'k, T>,
    ) {
        self.gemv_with_vector_width(a, x, y, partials, 4);
    }

    /// Emit a row-parallel GEMV with explicit per-lane K unrolling.
    pub fn gemv_with_vector_width<T: Numeric>(
        &mut self,
        a: &StorageTensor<'k, T>,
        x: &StorageTensor<'k, T>,
        y: &StorageTensor<'k, T>,
        partials: UninitTile<'k, T>,
        vector_width: u32,
    ) {
        self.gemv_tiled(a, x, y, partials, 1, vector_width);
    }

    /// Emit a row-parallel GEMV with explicit rows per workgroup and K unroll.
    pub fn gemv_tiled<T: Numeric>(
        &mut self,
        a: &StorageTensor<'k, T>,
        x: &StorageTensor<'k, T>,
        y: &StorageTensor<'k, T>,
        partials: UninitTile<'k, T>,
        rows_per_workgroup: u32,
        vector_width: u32,
    ) {
        assert!(
            rows_per_workgroup > 0,
            "gemv rows per workgroup must be non-zero"
        );
        assert!(
            rows_per_workgroup <= 4,
            "this prototype currently lowers at most four GEMV rows per workgroup"
        );
        assert!(vector_width > 0, "gemv vector width must be non-zero");
        let [m, k] = matrix_shape(&a.view.layout);
        let [x_k, x_cols] = matrix_shape(&x.view.layout);
        let [y_m, y_cols] = matrix_shape(&y.view.layout);
        assert_eq!(k, x_k, "gemv K dimensions must match");
        assert_eq!(x_cols, 1, "gemv vector must be shaped [K, 1]");
        assert_eq!(m, y_m, "gemv output row count must match A");
        assert_eq!(y_cols, 1, "gemv output must be shaped [M, 1]");
        assert_eq!(
            m % rows_per_workgroup,
            0,
            "gemv row count must divide rows per workgroup"
        );

        let partial_layout = self.cx.tile_layout(partials.tile);
        assert_eq!(
            partial_layout.memory_level(),
            MemoryLevel::Workgroup,
            "gemv partials must live in workgroup memory"
        );
        assert_eq!(
            partial_layout.shape().rank(),
            1,
            "gemv partials must be rank-1"
        );
        assert_eq!(
            partial_layout.element_count().get(),
            GEMV_WORKGROUP_INVOCATIONS * rows_per_workgroup,
            "gemv partials must contain 128 elements per row"
        );

        self.cx.push_op(Op::Gemv(GemvOp {
            a: a.view(),
            x: x.view(),
            y: y.view(),
            partials: partials.tile,
            rows_per_workgroup,
            vector_width,
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

    pub(crate) fn tile_matrix_shape(&self, tile: TileRef) -> [u32; 2] {
        matrix_shape(self.cx.tile_layout(tile))
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

    /// Partition a ready tile into an explicit-origin lower-level tile view.
    pub fn partition_at<T: Numeric>(
        &mut self,
        tile: &ReadyTile<'k, '_, T>,
        level: TileLevel,
        shape: Shape,
        origin: [u32; 2],
        body: impl for<'part> FnOnce(&mut Self, ReadyTile<'k, 'part, T>),
    ) {
        let source_layout = self.cx.tile_layout(tile.tile);
        validate_partition_view(source_layout, &shape, origin);
        let view_layout = Layout::strided(
            source_layout.memory_level(),
            shape,
            source_layout.strides().clone(),
        );
        let view = self
            .cx
            .alloc_tile_view_at::<T>(tile.tile, view_layout, level, origin);
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

    /// Partition a private/register tile into an explicit-origin lower-level view.
    pub fn partition_private_at<T: Numeric>(
        &mut self,
        tile: &mut RegTile<'k, T>,
        level: TileLevel,
        shape: Shape,
        origin: [u32; 2],
        body: impl FnOnce(&mut Self, RegTile<'k, T>),
    ) {
        let source_layout = self.cx.tile_layout(tile.tile);
        validate_partition_view(source_layout, &shape, origin);
        let view_layout = Layout::strided(
            source_layout.memory_level(),
            shape,
            source_layout.strides().clone(),
        );
        let view = self
            .cx
            .alloc_tile_view_at::<T>(tile.tile, view_layout, level, origin);
        let reg = RegTile {
            tile: view,
            _ty: PhantomData,
            _kernel: PhantomData,
        };

        self.cx.begin_block();
        body(self, reg);
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
            kind: LoopKind::RangeStep {
                induction: Dim(0),
                iterations: 1,
            },
            body,
        }));

        let after_phase = Phase {
            cx,
            state: Clean,
            _phase: PhantomData,
        };
        after(after_phase)
    }

    /// Build a counted stepped loop with a statically known trip count.
    pub fn range_step_count<R>(
        self,
        iterations: u32,
        body: impl for<'iter, 'body> FnOnce(Phase<'body, 'k, 'iter, Clean>, Dim) -> Synced<'iter>,
        after: impl for<'after, 'after_body> FnOnce(Phase<'after_body, 'k, 'after, Clean>) -> R,
    ) -> R {
        assert!(iterations > 0, "loop iteration count must be non-zero");
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
            kind: LoopKind::RangeStep {
                induction: Dim(0),
                iterations,
            },
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

fn validate_partition_view(source: &Layout, shape: &Shape, origin: [u32; 2]) {
    assert_eq!(source.shape().rank(), 2, "partition source must be rank-2");
    assert_eq!(shape.rank(), 2, "partition view must be rank-2");
    for (axis, (origin, dim)) in origin.iter().zip(shape.dims()).enumerate() {
        let parent = source.shape().dims()[axis].get();
        let end = origin
            .checked_add(dim.get())
            .expect("partition view origin overflow");
        assert!(
            end <= parent,
            "partition view must stay within parent tile shape"
        );
    }
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
pub struct RegTile<'k, T> {
    pub(crate) tile: TileRef,
    _ty: PhantomData<T>,
    _kernel: PhantomData<&'k mut ()>,
}

impl<'k, T> Copy for RegTile<'k, T> {}

impl<'k, T> Clone for RegTile<'k, T> {
    fn clone(&self) -> Self {
        *self
    }
}

/// A storage buffer tensor bound by the kernel.
pub struct StorageTensor<'k, T> {
    pub(crate) buffer: BufferRef,
    pub(crate) view: StorageView,
    _ty: PhantomData<T>,
    _kernel: PhantomData<&'k mut ()>,
}

impl<'k, T> StorageTensor<'k, T> {
    /// Create a rank-2 tile view whose logical origin is offset by workgroup id.
    pub fn workgroup_tile_2d(
        &self,
        shape: Shape,
        row_offset: Option<WorkgroupOffset>,
        col_offset: Option<WorkgroupOffset>,
    ) -> Self {
        self.dynamic_tile_2d(
            shape,
            row_offset.map(DynamicOffset::Workgroup),
            col_offset.map(DynamicOffset::Workgroup),
        )
    }

    /// Create a rank-2 tile view whose logical origin has dynamic offsets.
    pub fn dynamic_tile_2d(
        &self,
        shape: Shape,
        row_offset: Option<DynamicOffset>,
        col_offset: Option<DynamicOffset>,
    ) -> Self {
        assert_eq!(self.view.layout.shape().rank(), 2, "parent view must be 2D");
        assert_eq!(shape.rank(), 2, "tile view must be 2D");
        assert!(
            self.view.dynamic_offsets.iter().all(Option::is_none),
            "nested dynamic storage views are not supported"
        );
        let layout = Layout::strided(
            MemoryLevel::Storage,
            shape,
            self.view.layout.strides().clone(),
        );
        Self {
            buffer: self.buffer,
            view: StorageView {
                buffer: self.buffer,
                offset: self.view.offset,
                layout,
                dynamic_offsets: vec![row_offset, col_offset],
            },
            _ty: PhantomData,
            _kernel: PhantomData,
        }
    }

    fn view(&self) -> StorageView {
        self.view.clone()
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
