//! Pairs the tile-ir [`Program`] storage declarations with a generic list of
//! runtime "binding sources", so that callers (such as the wgpu backend) can
//! declare an IR storage and remember which runtime buffer it corresponds to
//! via a single call.
//!
//! The generic `B` is opaque to tile-ir. Callers choose what to track —
//! `core` uses `Arc<wgpu::Buffer>`; tests can use `()`.

use crate::{
    tile::{Program, RuntimeElement, Storage},
    ElementType, KernelIr, Layout, Numeric,
};

/// A runtime binding paired with the IR layout that describes how the kernel
/// will access it. Constructed by the caller (typically core's
/// `KernelTensor`); consumed by [`KernelBuilder`] to declare a storage.
pub struct KernelTensorRef<B> {
    /// Caller-owned runtime binding associated with this tensor.
    pub binding: B,
    /// Logical layout used by the generated IR.
    pub layout: Layout,
    /// Element offset applied to the storage view.
    pub offset: u32,
}

impl<B> KernelTensorRef<B> {
    /// Create a tensor reference with zero element offset.
    pub fn new(binding: B, layout: Layout) -> Self {
        Self::with_offset(binding, layout, 0)
    }

    /// Create a tensor reference with an element offset.
    pub fn with_offset(binding: B, layout: Layout, offset: u32) -> Self {
        Self {
            binding,
            layout,
            offset,
        }
    }
}

/// Owns a [`Program`] and the parallel list of runtime bindings.
///
/// Each `read`/`write` call declares an IR storage and pushes the matching
/// `B` onto the binding list — so the order of declarations and the binding
/// indices in the lowered Naga module match the order of bindings here. Call
/// [`finish`](Self::finish) to consume the builder and get the [`KernelIr`]
/// plus the binding list back.
pub struct KernelBuilder<B> {
    program: Program,
    bindings: Vec<B>,
}

impl<B> Default for KernelBuilder<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B> KernelBuilder<B> {
    /// Create an empty kernel builder.
    ///
    /// ```
    /// use fusor_tile_ir::{
    ///     KernelBuilder, KernelTensorRef, Layout, MemoryLevel, Shape, TileLiteral, F32,
    /// };
    ///
    /// let layout = Layout::contiguous(MemoryLevel::Storage, Shape::new([16]));
    /// let mut kb = KernelBuilder::<&'static str>::new();
    /// let input = kb.read::<F32, 1>(KernelTensorRef::new("input", layout.clone()));
    /// let output = kb.write::<F32, 1>(KernelTensorRef::new("output", layout));
    ///
    /// kb.program().program_grid::<16>([1, 1, 1], |block| {
    ///     let lane = block.lane();
    ///     let mask = lane.clone().lt(16u32);
    ///     let value = block.load(input.at(lane.clone()), mask.clone(), TileLiteral::f32(0.0));
    ///     block.store(output.at(lane), value, mask);
    /// });
    ///
    /// let (_ir, bindings) = kb.finish();
    /// assert_eq!(bindings, ["input", "output"]);
    /// ```
    pub fn new() -> Self {
        Self {
            program: Program::new(),
            bindings: Vec::new(),
        }
    }

    /// Direct access to the underlying [`Program`] for grid construction
    /// and other operations that aren't per-tensor.
    pub fn program(&mut self) -> &mut Program {
        &mut self.program
    }

    /// Declare a read-only typed storage binding.
    pub fn read<T: Numeric, const R: usize>(
        &mut self,
        tensor: KernelTensorRef<B>,
    ) -> Storage<T, R> {
        self.declare_storage(tensor, |program, layout, offset| {
            program.storage_read_with_layout_offset::<T, R>(layout, offset)
        })
    }

    /// Declare a read-write typed storage binding.
    pub fn write<T: Numeric, const R: usize>(
        &mut self,
        tensor: KernelTensorRef<B>,
    ) -> Storage<T, R> {
        self.declare_storage(tensor, |program, layout, offset| {
            program.storage_write_with_layout_offset::<T, R>(layout, offset)
        })
    }

    /// Declare a read-only storage binding whose element type is known at runtime.
    pub fn read_element<const R: usize>(
        &mut self,
        element: ElementType,
        tensor: KernelTensorRef<B>,
    ) -> Storage<RuntimeElement, R> {
        self.declare_storage(tensor, |program, layout, offset| {
            program.storage_read_element_with_layout_offset::<R>(element, layout, offset)
        })
    }

    /// Declare a read-write storage binding whose element type is known at runtime.
    pub fn write_element<const R: usize>(
        &mut self,
        element: ElementType,
        tensor: KernelTensorRef<B>,
    ) -> Storage<RuntimeElement, R> {
        self.declare_storage(tensor, |program, layout, offset| {
            program.storage_write_element_with_layout_offset::<R>(element, layout, offset)
        })
    }

    /// Push the binding and call `declare` with the program plus the
    /// tensor's layout and offset. Shared by every
    /// `read`/`write`/`read_element`/`write_element` entry point.
    fn declare_storage<S>(
        &mut self,
        tensor: KernelTensorRef<B>,
        declare: impl FnOnce(&mut Program, Layout, u32) -> S,
    ) -> S {
        self.bindings.push(tensor.binding);
        declare(&mut self.program, tensor.layout, tensor.offset)
    }

    /// Append a binding without declaring an IR storage. Used by downstream
    /// helpers (e.g. `fusor-tile-ir-kernels::quantized_matrix_for`) that need
    /// to declare a quantized matrix backed by a runtime binding without
    /// going through the typed `read`/`write` paths.
    pub fn push_binding(&mut self, binding: B) {
        self.bindings.push(binding);
    }

    /// Finish building and return the IR plus bindings in declaration order.
    pub fn finish(self) -> (KernelIr, Vec<B>) {
        let Self { program, bindings } = self;
        (program.into_ir(), bindings)
    }
}
