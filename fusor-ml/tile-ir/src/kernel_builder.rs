//! Pairs the tile-ir [`Program`] storage declarations with a generic list of
//! runtime "binding sources", so that callers (such as the wgpu backend) can
//! declare an IR storage and remember which runtime buffer it corresponds to
//! via a single call.
//!
//! The generic `B` is opaque to tile-ir. Callers choose what to track —
//! `core` uses `Arc<wgpu::Buffer>`; tests can use `()`.

use crate::{
    ElementType, KernelIr, Layout, Numeric, StorageIndexMap,
    quantized::{GgmlQuantFormat, QuantizedMatrix},
    tile::{ErasedStorage, Program, Storage},
};

/// A runtime binding paired with the IR layout that describes how the kernel
/// will access it. Constructed by the caller (typically core's
/// `KernelTensor`); consumed by [`KernelBuilder`] to declare a storage.
pub struct KernelTensorRef<B> {
    pub binding: B,
    pub layout: Layout,
    pub offset: u32,
    pub index_map: Option<StorageIndexMap>,
}

impl<B> KernelTensorRef<B> {
    pub fn new(binding: B, layout: Layout) -> Self {
        Self {
            binding,
            layout,
            offset: 0,
            index_map: None,
        }
    }

    pub fn with_offset(binding: B, layout: Layout, offset: u32) -> Self {
        Self {
            binding,
            layout,
            offset,
            index_map: None,
        }
    }

    pub fn with_offset_and_index_map(
        binding: B,
        layout: Layout,
        offset: u32,
        index_map: Option<StorageIndexMap>,
    ) -> Self {
        Self {
            binding,
            layout,
            offset,
            index_map,
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

    pub fn read<T: Numeric, const R: usize>(
        &mut self,
        tensor: KernelTensorRef<B>,
    ) -> Storage<T, R> {
        self.bindings.push(tensor.binding);
        match tensor.index_map {
            Some(index_map) => self
                .program
                .storage_read_with_layout_offset_and_index_map::<T, R>(
                    tensor.layout,
                    tensor.offset,
                    index_map,
                ),
            None => self
                .program
                .storage_read_with_layout_offset::<T, R>(tensor.layout, tensor.offset),
        }
    }

    pub fn write<T: Numeric, const R: usize>(
        &mut self,
        tensor: KernelTensorRef<B>,
    ) -> Storage<T, R> {
        self.bindings.push(tensor.binding);
        match tensor.index_map {
            Some(index_map) => self
                .program
                .storage_write_with_layout_offset_and_index_map::<T, R>(
                    tensor.layout,
                    tensor.offset,
                    index_map,
                ),
            None => self
                .program
                .storage_write_with_layout_offset::<T, R>(tensor.layout, tensor.offset),
        }
    }

    pub fn read_erased<const R: usize>(
        &mut self,
        element: ElementType,
        tensor: KernelTensorRef<B>,
    ) -> ErasedStorage<R> {
        self.bindings.push(tensor.binding);
        self.program.storage_read_element_with_layout_offset::<R>(
            element,
            tensor.layout,
            tensor.offset,
        )
    }

    pub fn write_erased<const R: usize>(
        &mut self,
        element: ElementType,
        tensor: KernelTensorRef<B>,
    ) -> ErasedStorage<R> {
        self.bindings.push(tensor.binding);
        self.program.storage_write_element_with_layout_offset::<R>(
            element,
            tensor.layout,
            tensor.offset,
        )
    }

    /// Declare a quantized matrix backed by `binding`.
    pub fn quantized_matrix(
        &mut self,
        binding: B,
        format: GgmlQuantFormat,
        rows: u32,
        cols: u32,
    ) -> QuantizedMatrix {
        self.bindings.push(binding);
        self.program.quantized_matrix(format, rows, cols)
    }

    pub fn finish(self) -> (KernelIr, Vec<B>) {
        let Self { program, bindings } = self;
        (program.into_ir(), bindings)
    }
}
