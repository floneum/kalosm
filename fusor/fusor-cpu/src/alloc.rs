//! 64-byte-aligned buffers: SIMD loads never split a cache line.

use fusor_ir::Result;
use fusor_ir::error::Error;
use std::alloc::{Layout, alloc_zeroed, dealloc};

/// A 64-byte-aligned byte buffer.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: exclusive ownership, no interior mutability.
unsafe impl Send for AlignedBuf {}
// SAFETY: read-only access, plus [`AlignedBuf::as_mut_ptr`] (see its contract).
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub const ALIGN: usize = 64;

    pub fn zeroed(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }
        let layout = Layout::from_size_align(len, Self::ALIGN)
            .map_err(|e| Error::Device(format!("bad allocation layout: {e}")))?;
        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(Error::Device(format!("out of memory allocating {len} B")));
        }
        Ok(Self { ptr, len })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` is a live allocation of `len` initialized bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `&mut self` proves exclusivity over a live allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Aliasing escape hatch: every worker gets the same buffer and writes a
    /// disjoint slice, which `verify_launch` proves before launch.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.len == 0 || self.ptr.is_null() {
            return;
        }
        // SAFETY: same size and alignment `zeroed` allocated with.
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.len, Self::ALIGN);
            dealloc(self.ptr, layout);
        }
    }
}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBuf({} B @ {:p})", self.len, self.ptr)
    }
}
