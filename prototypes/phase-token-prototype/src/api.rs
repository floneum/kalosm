use std::fmt;
use std::marker::PhantomData;

use crate::{Event, KernelIr, TileId};

/// A symbolic dimension used only to make the loop API look like an IR builder.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dim(pub u32);

/// A sample numeric marker.
#[derive(Copy, Clone, Debug)]
pub struct F32;

/// Build a toy kernel IR with a generative kernel lifetime and entry phase.
pub fn build(
    f: impl for<'k, 'entry, 'cx> FnOnce(Phase<'cx, 'k, 'entry>) -> KernelDone,
) -> KernelIr {
    let mut ir = KernelIr::default();
    let mut cx = KernelBuilder {
        ir: &mut ir,
        _kernel: PhantomData,
    };

    let phase = Phase {
        cx: &mut cx,
        state: Clean,
        _phase: PhantomData,
    };
    let KernelDone(()) = f(phase);
    ir
}

/// The toy IR builder. Users never receive this directly; they operate through
/// phase-scoped [`Phase`] handles.
pub struct KernelBuilder<'k> {
    pub(crate) ir: &'k mut KernelIr,
    pub(crate) _kernel: PhantomData<&'k mut ()>,
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
    tile: TileId,
    _ty: PhantomData<T>,
}

impl<'cx, 'k, 'flow> Phase<'cx, 'k, 'flow, Clean> {
    /// Allocate a workgroup tile whose contents are not yet initialized.
    pub fn alloc_workgroup<T>(&mut self) -> UninitTile<'k, T> {
        let id = TileId(self.cx.ir.next_tile);
        self.cx.ir.next_tile += 1;
        self.cx.ir.events.push(Event::AllocWorkgroup { tile: id });
        UninitTile {
            id,
            _ty: PhantomData,
            _kernel: PhantomData,
        }
    }

    /// Emit a cooperative load into an uninitialized tile.
    ///
    /// This consumes the clean phase and returns a pending phase. The pending
    /// phase has no `finish`, `range_step`, or `sync_end` methods, so user code
    /// must synchronize it before it can finish the control-flow body.
    pub fn cooperative_load<T>(self, dst: UninitTile<'k, T>) -> Phase<'cx, 'k, 'flow, Pending<T>> {
        self.cx
            .ir
            .events
            .push(Event::CooperativeLoad { tile: dst.id });
        Phase {
            cx: self.cx,
            state: Pending {
                tile: dst.id,
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
        self.cx.ir.events.push(Event::WorkgroupBarrier);
        Synced {
            _phase: PhantomData,
        }
    }

    /// Emit a read from any ready tile. If the tile exists at this construction
    /// point, the linear phase handle has already sequenced its producing
    /// barrier before this read.
    pub fn read<'ready, T>(&mut self, tile: &ReadyTile<'k, 'ready, T>) {
        self.cx.ir.events.push(Event::ReadReady { tile: tile.id });
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
        cx.ir.events.push(Event::RangeStepStart);

        let iter_phase = Phase {
            cx,
            state: Clean,
            _phase: PhantomData,
        };
        let synced = body(iter_phase, Dim(0));
        drop(synced);

        cx.ir.events.push(Event::RangeStepEnd);

        let after_phase = Phase {
            cx,
            state: Clean,
            _phase: PhantomData,
        };
        after(after_phase)
    }

    /// Consume the final phase handle and finish kernel construction.
    pub fn finish(self) -> KernelDone {
        self.cx.ir.events.push(Event::Finish);
        KernelDone(())
    }
}

impl<'cx, 'k, 'flow, T> Phase<'cx, 'k, 'flow, Pending<T>> {
    /// Emit a barrier, consume the pending load, and return a ready tile plus a
    /// clean phase handle.
    pub fn sync_tile(self) -> (ReadyTile<'k, 'flow, T>, Phase<'cx, 'k, 'flow, Clean>) {
        self.cx.ir.events.push(Event::WorkgroupBarrier);
        let ready = ReadyTile {
            id: self.state.tile,
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

/// A barrier witness. This is intentionally not constructible outside the crate.
pub struct Synced<'flow> {
    _phase: PhantomData<fn(&'flow ()) -> &'flow ()>,
}

/// A marker returned by [`Phase::finish`].
pub struct KernelDone(());

/// Workgroup memory with undefined contents.
pub struct UninitTile<'k, T> {
    pub(crate) id: TileId,
    _ty: PhantomData<T>,
    _kernel: PhantomData<&'k mut ()>,
}

/// Workgroup memory that has been cooperatively written but not synchronized.
pub type PendingTile<'cx, 'k, 'flow, T> = Phase<'cx, 'k, 'flow, Pending<T>>;

/// Workgroup memory that can be read after its producing barrier.
///
/// This is intentionally not `Copy`; future reload APIs can consume it to
/// invalidate stale ready views.
pub struct ReadyTile<'k, 'flow, T> {
    pub(crate) id: TileId,
    _ty: PhantomData<T>,
    _kernel: PhantomData<&'k ()>,
    _phase: PhantomData<&'flow ()>,
}

impl<T> fmt::Debug for ReadyTile<'_, '_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadyTile")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}
