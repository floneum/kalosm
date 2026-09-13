//! A fixed dispatch program with owned pool leases. Holding WGPU bind groups
//! alone is insufficient: the pool can otherwise reuse their backing buffers.
use crate::{launch::CommandRecord, pool::GpuBuffer};
use fusor_ir::{Result, error::Error, target::Buf};
use std::sync::Arc;

pub struct GpuExecutable {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    commands: Vec<CommandRecord>,
    retained: Vec<Buf>,
    inputs: Vec<Buf>,
}
impl GpuExecutable {
    #[cfg(feature = "prototype-capture")]
    pub(crate) fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        commands: Vec<CommandRecord>,
        retained: Vec<Buf>,
        inputs: Vec<Buf>,
    ) -> Result<Self> {
        if commands.is_empty()
            || commands
                .iter()
                .any(|r| !matches!(r, CommandRecord::Dispatch { .. }))
        {
            return Err(Error::Plan(
                "fixed executable requires a nonempty dispatch-only program".into(),
            ));
        }
        Ok(Self {
            device,
            queue,
            commands,
            retained,
            inputs,
        })
    }
    /// Encode into a caller-owned pass, without allocation, lookup, or binding
    /// preparation. Shapes, uniforms and buffer identities are fixed at capture.
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>) {
        for command in &self.commands {
            if let CommandRecord::Dispatch {
                pipeline,
                bind_group,
                grid,
                ..
            } = command
            {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group.as_ref(), &[]);
                pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
            }
        }
    }
    /// Submit one execution. Queue ordering protects prior uses of the same
    /// owned arena; independent overlapping executions need separate instances.
    pub fn submit(&self) -> wgpu::SubmissionIndex {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.encode(&mut pass);
        }
        self.queue.submit([encoder.finish()])
    }
    pub fn dispatches(&self) -> usize {
        self.commands.len()
    }
    pub fn retained_buffers(&self) -> &[Buf] {
        &self.retained
    }
    /// Non-uniform bindings read but never written by the captured dispatches.
    pub fn inputs(&self) -> &[Buf] {
        &self.inputs
    }
    /// Update the contents of a retained binding, preserving its identity and
    /// compiled shape. Replacing buffers or changing dimensions requires a new
    /// executable; this method cannot silently keep a stale bind group.
    pub fn write(&self, buffer: &Buf, offset: u64, bytes: &[u8]) -> Result<()> {
        if !self.inputs.iter().any(|b| b.addr() == buffer.addr()) {
            return Err(Error::Device(
                "only a fixed read-only input may be updated".into(),
            ));
        }
        let gpu = buffer
            .downcast_ref::<GpuBuffer>()
            .ok_or_else(|| Error::Device("binding is not a GPU buffer".into()))?;
        if !gpu.usage.contains(wgpu::BufferUsages::COPY_DST)
            || offset % 4 != 0
            || bytes.len() % 4 != 0
            || offset
                .checked_add(bytes.len() as u64)
                .is_none_or(|end| end > gpu.size)
        {
            return Err(Error::Device(
                "buffer update is unaligned or outside the fixed binding".into(),
            ));
        }
        self.queue.write_buffer(&gpu.buffer, offset, bytes);
        Ok(())
    }
}

#[cfg(feature = "prototype-capture")]
pub(crate) struct Capture {
    pub owner: std::thread::ThreadId,
    pub commands: Vec<CommandRecord>,
    pub buffers: rustc_hash::FxHashMap<usize, Buf>,
    pub groups: rustc_hash::FxHashSet<usize>,
    pub reads: rustc_hash::FxHashSet<usize>,
    pub writes: rustc_hash::FxHashSet<usize>,
}
#[cfg(feature = "prototype-capture")]
impl Capture {
    pub(crate) fn new() -> Self {
        Self {
            owner: std::thread::current().id(),
            commands: vec![],
            buffers: Default::default(),
            groups: Default::default(),
            reads: Default::default(),
            writes: Default::default(),
        }
    }
    pub(crate) fn on_thread(&self) -> bool {
        self.owner == std::thread::current().id()
    }
}
#[cfg(feature = "prototype-capture")]
pub(crate) struct CancelCapture<'a>(pub Option<&'a parking_lot::Mutex<Option<Capture>>>);
#[cfg(feature = "prototype-capture")]
impl Drop for CancelCapture<'_> {
    fn drop(&mut self) {
        if let Some(slot) = self.0 {
            *slot.lock() = None;
        }
    }
}
