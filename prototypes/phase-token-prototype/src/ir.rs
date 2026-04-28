use crate::{LowerError, NagaKernel};

/// The event log emitted by the prototype builder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelIr {
    pub(crate) events: Vec<Event>,
    pub(crate) next_tile: u32,
}

impl KernelIr {
    /// Returns the events emitted while building the prototype IR.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Lower this toy IR into a validated Naga module.
    pub fn lower_to_naga(&self) -> Result<NagaKernel, LowerError> {
        crate::lower::lower_to_naga(self)
    }
}

/// Events emitted by the toy builder.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Event {
    /// A workgroup tile was allocated.
    AllocWorkgroup { tile: TileId },
    /// A cooperative load into a tile was emitted.
    CooperativeLoad { tile: TileId },
    /// A workgroup barrier was emitted.
    WorkgroupBarrier,
    /// A read from a ready workgroup tile was emitted.
    ReadReady { tile: TileId },
    /// A symbolic loop was opened.
    RangeStepStart,
    /// A symbolic loop was closed.
    RangeStepEnd,
    /// Kernel construction consumed the final phase handle.
    Finish,
}

/// A tiny tile identifier for the toy event log.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileId(pub(crate) u32);
